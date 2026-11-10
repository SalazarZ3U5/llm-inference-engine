#include "cuda/kernels.cuh"

#include <cuda_bf16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ie::cuda {

namespace {

// Check a CUDA API call immediately. An unchecked failure stays in CUDA's
// "last error" slot and is reported by the next launch check instead, which
// then blames the wrong kernel.
void check_api(cudaError_t err, const char* what) {
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(err));
}

void check_launch(const char* name) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("CUDA kernel ") + name + " failed to launch: " +
                             cudaGetErrorString(err));
}

// ------------------------------------------------------- reductions ----

__device__ __forceinline__ float warp_sum(float v) {
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;  // every lane gets the total
}

__device__ __forceinline__ float warp_max(float v) {
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}

// Sum (or max) over the whole block; every thread gets the result.
template <bool kMax>
__device__ float block_reduce(float v) {
  __shared__ float partial[32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;
  v = kMax ? warp_max(v) : warp_sum(v);
  __syncthreads();  // `partial` may still be in use by a previous call
  if (lane == 0) partial[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float u = lane < num_warps ? partial[lane] : (kMax ? -INFINITY : 0.0f);
    u = kMax ? warp_max(u) : warp_sum(u);
    if (lane == 0) partial[0] = u;
  }
  __syncthreads();
  return partial[0];
}

// --------------------------------------------------- weight formats ----
//
// Each format says how many weights one 16-byte load holds and how to turn
// them into floats. The matmul kernels are templates over the format.

struct Bf16 {
  using Elem = uint16_t;
  static constexpr int kPerLoad = 8;  // 8 x 2 bytes
  static constexpr bool kScaled = false;
  // Two BF16 values per 32-bit word: the low half is element 0.
  __device__ static __forceinline__ void unpack(const uint4 p, float w[kPerLoad]) {
    w[0] = __uint_as_float(p.x << 16);
    w[1] = __uint_as_float(p.x & 0xFFFF0000u);
    w[2] = __uint_as_float(p.y << 16);
    w[3] = __uint_as_float(p.y & 0xFFFF0000u);
    w[4] = __uint_as_float(p.z << 16);
    w[5] = __uint_as_float(p.z & 0xFFFF0000u);
    w[6] = __uint_as_float(p.w << 16);
    w[7] = __uint_as_float(p.w & 0xFFFF0000u);
  }
  __device__ static __forceinline__ float load1(const Elem* row, int k) {
    return __uint_as_float(uint32_t(row[k]) << 16);
  }
};

struct Int8 {
  using Elem = int8_t;
  static constexpr int kPerLoad = 16;  // 16 x 1 byte
  static constexpr bool kScaled = true;  // the row scale is applied once, after the dot product
  __device__ static __forceinline__ void unpack4(uint32_t v, float* w) {
    w[0] = float(int8_t(v & 0xFF));
    w[1] = float(int8_t((v >> 8) & 0xFF));
    w[2] = float(int8_t((v >> 16) & 0xFF));
    w[3] = float(int8_t(v >> 24));
  }
  __device__ static __forceinline__ void unpack(const uint4 p, float w[kPerLoad]) {
    unpack4(p.x, w);
    unpack4(p.y, w + 4);
    unpack4(p.z, w + 8);
    unpack4(p.w, w + 12);
  }
  __device__ static __forceinline__ float load1(const Elem* row, int k) { return float(row[k]); }
};

// The core of every matmul: one warp computes dot(wrow, x[i]) for nt <= NT tokens.
// Lane j reads 16 bytes at [j*16, j*16+16), then 512 bytes further, and so on:
// the 32 lanes together read 512 consecutive bytes per step, which is what GPU
// memory is fastest at (a "coalesced" access). Each weight is converted once
// and used for all nt tokens.
template <class W, int NT>
__device__ __forceinline__ void warp_dot(const typename W::Elem* __restrict__ wrow,
                                         const float* __restrict__ x, int cols, int nt,
                                         float acc[NT]) {
  constexpr int P = W::kPerLoad;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < NT; ++i) acc[i] = 0.0f;
  if (cols % P == 0) {  // fast path: 16-byte loads (every row of this model)
    for (int k = lane * P; k < cols; k += 32 * P) {
      float w[P];
      W::unpack(__ldg(reinterpret_cast<const uint4*>(wrow + k)), w);
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        if (i < nt) {
          const float4* xp = reinterpret_cast<const float4*>(x + size_t(i) * cols + k);
          float s = acc[i];
#pragma unroll
          for (int j = 0; j < P / 4; ++j) {
            const float4 a = xp[j];
            s = fmaf(w[4 * j + 0], a.x, s);
            s = fmaf(w[4 * j + 1], a.y, s);
            s = fmaf(w[4 * j + 2], a.z, s);
            s = fmaf(w[4 * j + 3], a.w, s);
          }
          acc[i] = s;
        }
      }
    }
  } else {  // general path: one weight per lane per step
    for (int k = lane; k < cols; k += 32) {
      const float w = W::load1(wrow, k);
#pragma unroll
      for (int i = 0; i < NT; ++i)
        if (i < nt) acc[i] = fmaf(w, x[size_t(i) * cols + k], acc[i]);
    }
  }
#pragma unroll
  for (int i = 0; i < NT; ++i) acc[i] = warp_sum(acc[i]);
}

constexpr int kWarpsPerBlock = 8;  // 256 threads = 8 output rows per block

// ----------------------------------------------------------- kernels ----

template <class W>
__global__ void embedding_kernel(float* __restrict__ x, const int32_t* __restrict__ ids,
                                 const typename W::Elem* __restrict__ E,
                                 const float* __restrict__ scale, int hidden) {
  const int t = blockIdx.x, id = ids[t];
  const typename W::Elem* row = E + size_t(id) * hidden;
  const float s = W::kScaled ? scale[id] : 1.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) x[size_t(t) * hidden + i] = W::load1(row, i) * s;
}

__global__ void rmsnorm_kernel(float* y, const float* x, const float* __restrict__ w, int dim,
                               float eps) {
  const float* xt = x + size_t(blockIdx.x) * dim;
  float* yt = y + size_t(blockIdx.x) * dim;
  float ss = 0.0f;  // FP32 always: x can hold values near 1,600, whose squares overflow FP16
  for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xt[i] * xt[i];
  ss = block_reduce<false>(ss);
  const float inv_rms = 1.0f / sqrtf(ss / float(dim) + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x) yt[i] = xt[i] * inv_rms * w[i];
}

template <class W, int NT>
__global__ void matmul_kernel(float* __restrict__ y, const float* __restrict__ x,
                              const typename W::Elem* __restrict__ Wm,
                              const float* __restrict__ scale, const float* __restrict__ bias,
                              int T, int rows, int cols, bool accumulate) {
  const int row = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  if (row >= rows) return;  // whole warp leaves together, so shuffles stay safe
  const typename W::Elem* wrow = Wm + size_t(row) * cols;
  const float s = W::kScaled ? scale[row] : 1.0f;
  const float b = bias ? bias[row] : 0.0f;
  for (int t0 = 0; t0 < T; t0 += NT) {
    const int nt = min(NT, T - t0);
    float acc[NT];
    warp_dot<W, NT>(wrow, x + size_t(t0) * cols, cols, nt, acc);
    if ((threadIdx.x & 31) == 0) {
      for (int i = 0; i < nt; ++i) {
        float* dst = y + size_t(t0 + i) * rows + row;
        const float v = acc[i] * s + b;
        *dst = accumulate ? *dst + v : v;
      }
    }
  }
}

template <class W, int NT>
__global__ void gate_up_silu_kernel(float* __restrict__ out, const float* __restrict__ x,
                                    const typename W::Elem* __restrict__ Wg,
                                    const float* __restrict__ sg,
                                    const typename W::Elem* __restrict__ Wu,
                                    const float* __restrict__ su, int T, int rows, int cols) {
  const int row = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  if (row >= rows) return;
  const float scale_g = W::kScaled ? sg[row] : 1.0f, scale_u = W::kScaled ? su[row] : 1.0f;
  for (int t0 = 0; t0 < T; t0 += NT) {
    const int nt = min(NT, T - t0);
    float g[NT], u[NT];
    warp_dot<W, NT>(Wg + size_t(row) * cols, x + size_t(t0) * cols, cols, nt, g);
    warp_dot<W, NT>(Wu + size_t(row) * cols, x + size_t(t0) * cols, cols, nt, u);
    if ((threadIdx.x & 31) == 0) {
      for (int i = 0; i < nt; ++i) {
        const float gi = g[i] * scale_g;
        out[size_t(t0 + i) * rows + row] = gi / (1.0f + expf(-gi)) * (u[i] * scale_u);
      }
    }
  }
}

__global__ void rope_cache_kernel(float* __restrict__ q, const float* __restrict__ k,
                                  const float* __restrict__ v, float* __restrict__ k_cache,
                                  float* __restrict__ v_cache, const float* __restrict__ inv_freq,
                                  const int* __restrict__ pos_ptr, int n_heads, int n_kv_heads,
                                  int head_dim) {
  const int t = blockIdx.x, pos = *pos_ptr + t, half = head_dim / 2;
  const int q_dim = n_heads * head_dim, kv_dim = n_kv_heads * head_dim;
  // One thread per (head, pair): query heads first, then key heads.
  for (int j = threadIdx.x; j < (n_heads + n_kv_heads) * half; j += blockDim.x) {
    const int head = j / half, i = j % half;
    const float angle = float(pos) * inv_freq[i];
    const float c = cosf(angle), s = sinf(angle);
    if (head < n_heads) {
      float* hv = q + size_t(t) * q_dim + head * head_dim;
      const float x0 = hv[i], x1 = hv[i + half];
      hv[i] = x0 * c - x1 * s;
      hv[i + half] = x1 * c + x0 * s;
    } else {
      const int kh = head - n_heads;
      const float* src = k + size_t(t) * kv_dim + kh * head_dim;
      float* dst = k_cache + size_t(pos) * kv_dim + kh * head_dim;  // rotated key goes straight to the cache
      const float x0 = src[i], x1 = src[i + half];
      dst[i] = x0 * c - x1 * s;
      dst[i + half] = x1 * c + x0 * s;
    }
  }
  for (int i = threadIdx.x; i < kv_dim; i += blockDim.x)
    v_cache[size_t(pos) * kv_dim + i] = v[size_t(t) * kv_dim + i];
}

// One block per (head, query token). Scores for every visible position live in
// shared memory, so the context is limited by shared memory size (see
// attention_max_tokens). A tiled "flash" version without that limit comes later.
__global__ void attention_kernel(float* __restrict__ out, const float* __restrict__ q,
                                 const float* __restrict__ k_cache,
                                 const float* __restrict__ v_cache, const int* __restrict__ pos_ptr,
                                 int n_heads, int n_kv_heads, int head_dim, float scale) {
  extern __shared__ float smem[];
  const int h = blockIdx.x, t = blockIdx.y, p = *pos_ptr + t;
  const int kvh = h / (n_heads / n_kv_heads);  // GQA
  const int q_dim = n_heads * head_dim, kv_dim = n_kv_heads * head_dim;
  float* qs = smem;                 // [head_dim]
  float* scores = smem + head_dim;  // [p + 1]

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    qs[d] = q[size_t(t) * q_dim + h * head_dim + d];
  __syncthreads();

  // 1. scores for positions 0..p (causal: nothing beyond p is ever read)
  float local_max = -INFINITY;
  for (int s = threadIdx.x; s <= p; s += blockDim.x) {
    const float* kr = k_cache + size_t(s) * kv_dim + kvh * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) dot = fmaf(qs[d], kr[d], dot);
    dot *= scale;
    scores[s] = dot;
    local_max = fmaxf(local_max, dot);
  }
  // 2. softmax (subtract the max so expf can't overflow)
  const float mx = block_reduce<true>(local_max);
  float local_sum = 0.0f;
  for (int s = threadIdx.x; s <= p; s += blockDim.x) {
    const float e = expf(scores[s] - mx);
    scores[s] = e;
    local_sum += e;
  }
  const float inv_sum = 1.0f / block_reduce<false>(local_sum);  // its barriers publish `scores`

  // 3. weighted sum of values; thread d owns output dimension d
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    const float* vc = v_cache + kvh * head_dim + d;
    float acc = 0.0f;
    for (int s = 0; s <= p; ++s) acc = fmaf(scores[s], vc[size_t(s) * kv_dim], acc);
    out[size_t(t) * q_dim + h * head_dim + d] = acc * inv_sum;
  }
}


// ------------------------------------------------- tensor-core GEMM ----
//
// Used whenever several tokens go through a matrix at once (prefill). Each
// block computes a 64 x 64 tile of the output (64 tokens x 64 output rows)
// with the GPU's tensor cores, via the WMMA API: 16x16x16 matrix multiplies
// on BF16 inputs with FP32 accumulation.
//
// Weights are BF16 already (or INT8, whose values -127..127 are exact in BF16;
// the row scale is applied at the end). Activations are FP32, and rounding them
// to BF16 would keep only ~3 significant digits. Instead each activation is
// split into two BF16 numbers, x = hi + lo (lo = x - hi, rounded), and the tile
// is computed as W·hi + W·lo. That keeps ~16 bits of mantissa: FP32-level
// accuracy for twice the tensor-core work, which is still far faster than the
// CUDA-core kernel.

namespace wmma = nvcuda::wmma;

constexpr int kTileM = 64, kTileN = 64, kTileK = 32;
constexpr int kLd = kTileK + 8;   // shared-memory row stride (BF16) for A and B tiles: the padding
                                  // spreads rows across memory banks; WMMA needs a multiple of 8
constexpr int kLdC = kTileN + 4;  // row stride (floats) of the output tile
constexpr int kTcThreads = 128;   // 4 warps in a 2 x 2 grid, each owning a 32 x 32 sub-tile
constexpr int kTileBytes = 64 * kLd * 2;

// A tile: 64 tokens x 32 columns of x, split into hi and lo BF16 parts.
__device__ __forceinline__ void tc_load_a(__nv_bfloat16* a_hi, __nv_bfloat16* a_lo,
                                          const float* __restrict__ x, int T, int K, int m0, int k0) {
  for (int i = threadIdx.x; i < kTileM * kTileK / 4; i += kTcThreads) {
    const int m = i / (kTileK / 4), c = (i % (kTileK / 4)) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (m0 + m < T) v = *reinterpret_cast<const float4*>(x + size_t(m0 + m) * K + k0 + c);
    const float vals[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const __nv_bfloat16 hi = __float2bfloat16_rn(vals[j]);
      a_hi[m * kLd + c + j] = hi;
      a_lo[m * kLd + c + j] = __float2bfloat16_rn(vals[j] - __bfloat162float(hi));
    }
  }
}

// B tile: 64 weight rows x 32 columns, as BF16. Rows past N are zero.
template <class W>
__device__ __forceinline__ void tc_load_b(__nv_bfloat16* b, const typename W::Elem* __restrict__ Wm,
                                          int N, int K, int n0, int k0) {
  if constexpr (!W::kScaled) {  // BF16: copy the bits, 8 per 16-byte load
    for (int i = threadIdx.x; i < kTileN * kTileK / 8; i += kTcThreads) {
      const int n = i / (kTileK / 8), c = (i % (kTileK / 8)) * 8;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (n0 + n < N) v = __ldg(reinterpret_cast<const uint4*>(Wm + size_t(n0 + n) * K + k0 + c));
      *reinterpret_cast<uint4*>(b + n * kLd + c) = v;
    }
  } else {  // INT8: 16 per load, each converted exactly to BF16
    for (int i = threadIdx.x; i < kTileN * kTileK / 16; i += kTcThreads) {
      const int n = i / (kTileK / 16), c = (i % (kTileK / 16)) * 16;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (n0 + n < N) v = __ldg(reinterpret_cast<const uint4*>(Wm + size_t(n0 + n) * K + k0 + c));
      float w[16];
      W::unpack(v, w);
#pragma unroll
      for (int j = 0; j < 16; ++j) b[n * kLd + c + j] = __float2bfloat16_rn(w[j]);
    }
  }
}

// kGateUp = false: y[t, n] (+)= (x·W^T)[t, n] * scale[n] + bias[n]
// kGateUp = true:  y[t, n]   = silu((x·Wg^T)[t, n] * sg[n]) * (x·Wu^T)[t, n] * su[n]
template <class W, bool kGateUp>
__global__ void __launch_bounds__(kTcThreads)
tc_gemm_kernel(float* __restrict__ y, const float* __restrict__ x,
               const typename W::Elem* __restrict__ W0, const float* __restrict__ s0,
               const typename W::Elem* __restrict__ W1, const float* __restrict__ s1,
               const float* __restrict__ bias, int T, int N, int K, bool accumulate) {
  constexpr int kNumB = kGateUp ? 2 : 1;
  constexpr int kMainBytes = (2 + kNumB) * kTileBytes;      // A hi, A lo, B tile(s)
  constexpr int kEpiBytes = kNumB * kTileM * kLdC * 4;      // FP32 output tile(s)
  __shared__ __align__(128) unsigned char smem[kMainBytes > kEpiBytes ? kMainBytes : kEpiBytes];
  auto* a_hi = reinterpret_cast<__nv_bfloat16*>(smem);
  auto* a_lo = reinterpret_cast<__nv_bfloat16*>(smem + kTileBytes);
  __nv_bfloat16* b_tile[2] = {reinterpret_cast<__nv_bfloat16*>(smem + 2 * kTileBytes),
                              reinterpret_cast<__nv_bfloat16*>(smem + 3 * kTileBytes)};

  const int n0 = blockIdx.x * kTileN, m0 = blockIdx.y * kTileM;
  const int warp = threadIdx.x / 32, wm = warp / 2, wn = warp % 2;
  const typename W::Elem* mats[2] = {W0, W1};

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[kNumB][2][2];
#pragma unroll
  for (int b = 0; b < kNumB; ++b)
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[b][i][j], 0.0f);

  for (int k0 = 0; k0 < K; k0 += kTileK) {
    tc_load_a(a_hi, a_lo, x, T, K, m0, k0);
#pragma unroll
    for (int b = 0; b < kNumB; ++b) tc_load_b<W>(b_tile[b], mats[b], N, K, n0, k0);
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kTileK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ahi[2], alo[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        wmma::load_matrix_sync(ahi[i], a_hi + (wm * 32 + i * 16) * kLd + kk, kLd);
        wmma::load_matrix_sync(alo[i], a_lo + (wm * 32 + i * 16) * kLd + kk, kLd);
      }
#pragma unroll
      for (int b = 0; b < kNumB; ++b) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          // B is stored [n][k], i.e. column-major as a K x N matrix.
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
          wmma::load_matrix_sync(bf, b_tile[b] + (wn * 32 + j * 16) * kLd + kk, kLd);
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            wmma::mma_sync(acc[b][i][j], ahi[i], bf, acc[b][i][j]);
            wmma::mma_sync(acc[b][i][j], alo[i], bf, acc[b][i][j]);
          }
        }
      }
    }
    __syncthreads();  // before the next k step overwrites the tiles
  }

  // Epilogue: park the accumulators in shared memory (reusing the tile space),
  // then write them out with scale, bias, residual add or SwiGLU applied.
  float* c_tile[2] = {reinterpret_cast<float*>(smem),
                      reinterpret_cast<float*>(smem) + kTileM * kLdC};
#pragma unroll
  for (int b = 0; b < kNumB; ++b)
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j)
        wmma::store_matrix_sync(c_tile[b] + (wm * 32 + i * 16) * kLdC + wn * 32 + j * 16,
                                acc[b][i][j], kLdC, wmma::mem_row_major);
  __syncthreads();

  for (int idx = threadIdx.x; idx < kTileM * kTileN; idx += kTcThreads) {
    const int m = idx / kTileN, n = idx % kTileN, gm = m0 + m, gn = n0 + n;
    if (gm >= T || gn >= N) continue;
    float* dst = y + size_t(gm) * N + gn;
    if constexpr (kGateUp) {
      const float g = c_tile[0][m * kLdC + n] * (W::kScaled ? s0[gn] : 1.0f);
      const float u = c_tile[1][m * kLdC + n] * (W::kScaled ? s1[gn] : 1.0f);
      *dst = g / (1.0f + expf(-g)) * u;
    } else {
      const float v = c_tile[0][m * kLdC + n] * (W::kScaled ? s0[gn] : 1.0f) + (bias ? bias[gn] : 0.0f);
      *dst = accumulate ? *dst + v : v;
    }
  }
}

bool g_tensor_cores = true;

// Tensor cores for any multi-token call whose width splits into 32-column tiles.
bool use_tensor_cores(int T, int cols) { return g_tensor_cores && T >= 2 && cols % kTileK == 0; }

template <class W, bool kGateUp>
void launch_tc(float* y, const float* x, const DeviceMatrix& A, const DeviceMatrix* B,
               const float* bias, int T, bool accumulate, cudaStream_t stream) {
  const dim3 grid((A.rows + kTileN - 1) / kTileN, (T + kTileM - 1) / kTileM);
  using E = typename W::Elem;
  tc_gemm_kernel<W, kGateUp><<<grid, kTcThreads, 0, stream>>>(
      y, x, static_cast<const E*>(A.data), A.scale, B ? static_cast<const E*>(B->data) : nullptr,
      B ? B->scale : nullptr, bias, T, A.rows, A.cols, accumulate);
}

// Largest dynamic shared memory the attention kernel may request per block.
int max_attention_smem_bytes() {
  static const int bytes = [] {
    int dev = 0, optin = 0;
    check_api(cudaGetDevice(&dev), "cudaGetDevice");
    check_api(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev),
              "cudaDeviceGetAttribute");
    // The per-block limit (99 KB on Ada) covers the kernel's *static* shared
    // memory too (block_reduce's arrays, 256 bytes), so only the rest can be
    // requested dynamically at launch.
    cudaFuncAttributes attr{};
    check_api(cudaFuncGetAttributes(&attr, attention_kernel), "cudaFuncGetAttributes");
    const int dynamic_max = optin - int(attr.sharedSizeBytes);
    // Raise the kernel's limit above the default 48 KB.
    check_api(cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   dynamic_max),
              "cudaFuncSetAttribute");
    return dynamic_max;
  }();
  return bytes;
}

int blocks_for(int rows) { return (rows + kWarpsPerBlock - 1) / kWarpsPerBlock; }

template <class W>
void launch_matmul(float* y, const float* x, const DeviceMatrix& M, const float* bias, int T,
                   bool accumulate, cudaStream_t stream) {
  const auto* data = static_cast<const typename W::Elem*>(M.data);
  if (T == 1)
    matmul_kernel<W, 1><<<blocks_for(M.rows), kWarpsPerBlock * 32, 0, stream>>>(
        y, x, data, M.scale, bias, T, M.rows, M.cols, accumulate);
  else
    matmul_kernel<W, 4><<<blocks_for(M.rows), kWarpsPerBlock * 32, 0, stream>>>(
        y, x, data, M.scale, bias, T, M.rows, M.cols, accumulate);
}

template <class W>
void launch_gate_up(float* out, const float* x, const DeviceMatrix& G, const DeviceMatrix& U,
                    int T, cudaStream_t stream) {
  const auto* g = static_cast<const typename W::Elem*>(G.data);
  const auto* u = static_cast<const typename W::Elem*>(U.data);
  if (T == 1)
    gate_up_silu_kernel<W, 1><<<blocks_for(G.rows), kWarpsPerBlock * 32, 0, stream>>>(
        out, x, g, G.scale, u, U.scale, T, G.rows, G.cols);
  else
    gate_up_silu_kernel<W, 4><<<blocks_for(G.rows), kWarpsPerBlock * 32, 0, stream>>>(
        out, x, g, G.scale, u, U.scale, T, G.rows, G.cols);
}

}  // namespace

// ------------------------------------------------------------ launch ----

void embedding(float* x, const int32_t* ids, const DeviceMatrix& E, int T, cudaStream_t stream) {
  if (E.int8)
    embedding_kernel<Int8><<<T, 256, 0, stream>>>(x, ids, static_cast<const int8_t*>(E.data), E.scale, E.cols);
  else
    embedding_kernel<Bf16><<<T, 256, 0, stream>>>(x, ids, static_cast<const uint16_t*>(E.data), nullptr, E.cols);
  check_launch("embedding");
}

void rmsnorm(float* y, const float* x, const float* weight, int T, int dim, float eps,
             cudaStream_t stream) {
  rmsnorm_kernel<<<T, 256, 0, stream>>>(y, x, weight, dim, eps);
  check_launch("rmsnorm");
}

void set_tensor_cores_impl(bool on) { g_tensor_cores = on; }
bool tensor_cores() { return g_tensor_cores; }

void matmul(float* y, const float* x, const DeviceMatrix& W, const float* bias, int T,
            bool accumulate, cudaStream_t stream) {
  if (use_tensor_cores(T, W.cols)) {
    if (W.int8) launch_tc<Int8, false>(y, x, W, nullptr, bias, T, accumulate, stream);
    else launch_tc<Bf16, false>(y, x, W, nullptr, bias, T, accumulate, stream);
  } else {
    if (W.int8) launch_matmul<Int8>(y, x, W, bias, T, accumulate, stream);
    else launch_matmul<Bf16>(y, x, W, bias, T, accumulate, stream);
  }
  check_launch("matmul");
}

void gate_up_silu(float* out, const float* x, const DeviceMatrix& Wg, const DeviceMatrix& Wu,
                  int T, cudaStream_t stream) {
  if (use_tensor_cores(T, Wg.cols)) {
    if (Wg.int8) launch_tc<Int8, true>(out, x, Wg, &Wu, nullptr, T, false, stream);
    else launch_tc<Bf16, true>(out, x, Wg, &Wu, nullptr, T, false, stream);
  } else {
    if (Wg.int8) launch_gate_up<Int8>(out, x, Wg, Wu, T, stream);
    else launch_gate_up<Bf16>(out, x, Wg, Wu, T, stream);
  }
  check_launch("gate_up_silu");
}

void rope_and_cache(float* q, const float* k, const float* v, float* k_cache, float* v_cache,
                    const float* inv_freq, int T, const int* pos, int n_heads, int n_kv_heads,
                    int head_dim, cudaStream_t stream) {
  const int threads = std::min(1024, std::max(32, (n_heads + n_kv_heads) * head_dim / 2));
  rope_cache_kernel<<<T, threads, 0, stream>>>(q, k, v, k_cache, v_cache, inv_freq, pos, n_heads,
                                               n_kv_heads, head_dim);
  check_launch("rope_and_cache");
}

int attention_max_tokens() {
  return max_attention_smem_bytes() / int(sizeof(float)) - 256;  // leave room for q (head_dim <= 256)
}

void attention(float* out, const float* q, const float* k_cache, const float* v_cache, int T,
               const int* pos, int max_tokens, int n_heads, int n_kv_heads, int head_dim,
               cudaStream_t stream) {
  // Sized for the whole cache, so the same launch works at every position.
  const size_t smem = sizeof(float) * (size_t(head_dim) + max_tokens);
  if (smem > size_t(max_attention_smem_bytes()))
    throw std::runtime_error("attention: context too long for the shared-memory kernel");
  const dim3 grid(n_heads, T);
  attention_kernel<<<grid, 128, smem, stream>>>(out, q, k_cache, v_cache, pos, n_heads, n_kv_heads,
                                                head_dim, 1.0f / sqrtf(float(head_dim)));
  check_launch("attention");
}

}  // namespace ie::cuda
