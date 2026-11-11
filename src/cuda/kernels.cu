#include "cuda/kernels.cuh"

#include <cuda_bf16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ie::cuda {

namespace {

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

// Offset (in floats) of position `pos` of sequence `seq` in a layer's page pool.
__device__ __forceinline__ size_t kv_offset(const KvLayout& kv, int seq, int pos, int kv_dim) {
  const int page = kv.tables[seq * kv.max_pages + pos / kKvPageSize];
  return (size_t(page) * kKvPageSize + pos % kKvPageSize) * kv_dim;
}

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
                                  const float* __restrict__ v, float* __restrict__ k_pool,
                                  float* __restrict__ v_pool, const float* __restrict__ inv_freq,
                                  const int* __restrict__ row_pos, const int* __restrict__ row_seq,
                                  KvLayout kv, int n_heads, int n_kv_heads, int head_dim) {
  const int t = blockIdx.x, pos = row_pos[t], half = head_dim / 2;
  const int q_dim = n_heads * head_dim, kv_dim = n_kv_heads * head_dim;
  const size_t dst_row = kv_offset(kv, row_seq[t], pos, kv_dim);  // this token's slot in the cache
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
      float* dst = k_pool + dst_row + kh * head_dim;  // rotated key goes straight to the cache
      const float x0 = src[i], x1 = src[i + half];
      dst[i] = x0 * c - x1 * s;
      dst[i + half] = x1 * c + x0 * s;
    }
  }
  for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) v_pool[dst_row + i] = v[size_t(t) * kv_dim + i];
}

// ------------------------------------------- flash-decoding attention ----
//
// The cached positions 0..p that a query can see are cut into chunks of
// kAttnChunk. Each block handles one (head, query token, chunk) and produces a
// *partial* softmax result over just its chunk:
//   m = max score in the chunk
//   l = sum over the chunk of exp(score - m)
//   o = sum over the chunk of exp(score - m) * value      (not yet divided by l)
// A second kernel merges the chunks exactly:
//   M = max m_c,  L = sum l_c * exp(m_c - M),  out = sum o_c * exp(m_c - M) / L
// So a long context is spread over many blocks instead of looping inside one,
// and nothing needs to fit in shared memory except one chunk's scores.
//
// The grid always has `num_chunks` chunks per query; chunks that start past the
// query's position exit at once. That keeps the launch identical at every
// position, which CUDA graphs need.

constexpr int kAttnChunk = 256;   // cached positions per block
constexpr int kAttnWarps = 4;     // 128 threads
constexpr int kMaxHeadDim = 128;  // lanes hold head_dim / 32 <= 4 values each
constexpr int kAttnUnroll = 4;   // positions per warp per loop step (memory-level parallelism)
constexpr int kChunkPages = kAttnChunk / kKvPageSize;  // pages one chunk spans
static_assert(kAttnChunk % kKvPageSize == 0, "a chunk must cover whole pages");
static_assert(kChunkPages <= kAttnWarps * 32, "one thread loads each page id");

__global__ void __launch_bounds__(kAttnWarps * 32)
attention_partial_kernel(float* __restrict__ part_o, float* __restrict__ part_ml,
                         const float* __restrict__ q, const float* __restrict__ k_pool,
                         const float* __restrict__ v_pool, const int* __restrict__ row_pos,
                         const int* __restrict__ row_seq, KvLayout kv, int n_heads,
                         int n_kv_heads, int head_dim, float scale, int num_chunks) {
  __shared__ float scores[kAttnChunk];
  __shared__ float partial_o[kAttnWarps][kMaxHeadDim];
  __shared__ int chunk_pages[kChunkPages];
  const int h = blockIdx.x, t = blockIdx.y, c = blockIdx.z;
  const int p = row_pos[t];  // last position this query may see (causal)
  const int seq = row_seq[t];
  const int s0 = c * kAttnChunk;
  const size_t slot = (size_t(t) * n_heads + h) * num_chunks + c;
  if (s0 > p) {  // chunk lies entirely in the future: contributes nothing
    if (threadIdx.x == 0) {
      part_ml[2 * slot] = -INFINITY;
      part_ml[2 * slot + 1] = 0.0f;
    }
    return;  // the whole block leaves together, so later barriers are safe
  }
  const int s1 = min(s0 + kAttnChunk, p + 1);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int kvh = h / (n_heads / n_kv_heads);  // GQA
  const int q_dim = n_heads * head_dim, kv_dim = n_kv_heads * head_dim;
  const int per_lane = head_dim / 32;

  // The chunk's page ids, looked up once: the loops below then find a position's
  // row with a shift and a shared-memory read, instead of a division and a
  // dependent global load for every position.
  if (threadIdx.x < (s1 - s0 + kKvPageSize - 1) / kKvPageSize)
    chunk_pages[threadIdx.x] = kv.tables[seq * kv.max_pages + s0 / kKvPageSize + threadIdx.x];
  __syncthreads();
  auto row_of = [&](int s) {  // s0 is a multiple of the page size
    const int i = s - s0;
    return (size_t(chunk_pages[i / kKvPageSize]) * kKvPageSize + i % kKvPageSize) * kv_dim;
  };

  // Each lane keeps its slice of the query in registers: dims lane, lane+32, ...
  float qr[kMaxHeadDim / 32];
#pragma unroll
  for (int j = 0; j < kMaxHeadDim / 32; ++j)
    qr[j] = j < per_lane ? q[size_t(t) * q_dim + h * head_dim + lane + 32 * j] : 0.0f;

  // 1. Scores: one warp per position, lanes across the head dimension, so the
  //    32 lanes read one contiguous key row (coalesced), then sum by shuffles.
  //    Each warp works on kAttnUnroll positions at a time: their loads are
  //    independent, so all are in flight together instead of one memory
  //    round trip per position (few warps run during decode to hide it).
  for (int s = s0 + warp; s < s1; s += kAttnWarps * kAttnUnroll) {
    float dot[kAttnUnroll];
#pragma unroll
    for (int u = 0; u < kAttnUnroll; ++u) {
      // Past the end, re-read the last valid row (and discard the result):
      // no branch, so the compiler issues all the loads back to back.
      const int su = min(s + u * kAttnWarps, s1 - 1);
      const float* kr = k_pool + row_of(su) + kvh * head_dim;
      dot[u] = 0.0f;
#pragma unroll
      for (int j = 0; j < kMaxHeadDim / 32; ++j)
        if (j < per_lane) dot[u] = fmaf(qr[j], kr[lane + 32 * j], dot[u]);
    }
#pragma unroll
    for (int u = 0; u < kAttnUnroll; ++u) {
      const float d = warp_sum(dot[u]);
      const int su = s + u * kAttnWarps;
      if (lane == 0 && su < s1) scores[su - s0] = d * scale;
    }
  }
  __syncthreads();

  // 2. Softmax within the chunk (subtract the chunk max so expf can't overflow).
  const int n = s1 - s0;
  float local_max = -INFINITY;
  for (int i = threadIdx.x; i < n; i += blockDim.x) local_max = fmaxf(local_max, scores[i]);
  const float m = block_reduce<true>(local_max);
  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float e = expf(scores[i] - m);
    scores[i] = e;
    local_sum += e;
  }
  const float l = block_reduce<false>(local_sum);  // its barriers also publish `scores`

  // 3. Weighted values: again one warp per position, lanes across dimensions.
  //    Same unrolling; positions are still added in increasing order, so the
  //    result is bit-identical to one position at a time.
  float acc[kMaxHeadDim / 32] = {};
  for (int s = s0 + warp; s < s1; s += kAttnWarps * kAttnUnroll) {
#pragma unroll
    for (int u = 0; u < kAttnUnroll; ++u) {
      // Past the end: weight 0 on the last valid row, and fmaf(0, v, acc) == acc.
      const int su = s + u * kAttnWarps, sc = min(su, s1 - 1);
      const float w = su < s1 ? scores[sc - s0] : 0.0f;
      const float* vr = v_pool + row_of(sc) + kvh * head_dim;
#pragma unroll
      for (int j = 0; j < kMaxHeadDim / 32; ++j)
        if (j < per_lane) acc[j] = fmaf(w, vr[lane + 32 * j], acc[j]);
    }
  }
#pragma unroll
  for (int j = 0; j < kMaxHeadDim / 32; ++j)
    if (j < per_lane) partial_o[warp][lane + 32 * j] = acc[j];
  __syncthreads();

  // Add the 4 warps' partial outputs and store this chunk's result.
  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float o = 0.0f;
#pragma unroll
    for (int w = 0; w < kAttnWarps; ++w) o += partial_o[w][d];
    part_o[slot * head_dim + d] = o;
  }
  if (threadIdx.x == 0) {
    part_ml[2 * slot] = m;
    part_ml[2 * slot + 1] = l;
  }
}

// One block per (head, query token), one thread per output dimension.
__global__ void attention_merge_kernel(float* __restrict__ out, const float* __restrict__ part_o,
                                       const float* __restrict__ part_ml, int n_heads,
                                       int head_dim, int num_chunks) {
  const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
  const size_t base = (size_t(t) * n_heads + h) * num_chunks;
  float M = -INFINITY;
  for (int c = 0; c < num_chunks; ++c) M = fmaxf(M, part_ml[2 * (base + c)]);
  float L = 0.0f, acc = 0.0f;
  for (int c = 0; c < num_chunks; ++c) {
    const float l = part_ml[2 * (base + c) + 1];
    if (l == 0.0f) continue;  // empty (future) chunk
    const float w = expf(part_ml[2 * (base + c)] - M);
    L += l * w;
    acc += part_o[(base + c) * head_dim + d] * w;
  }
  out[size_t(t) * n_heads * head_dim + h * head_dim + d] = acc / L;
}

__global__ void gather_rows_kernel(float* __restrict__ dst, const float* __restrict__ src,
                                   const int* __restrict__ rows, int dim) {
  const float* in = src + size_t(rows[blockIdx.x]) * dim;
  float* out = dst + size_t(blockIdx.x) * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) out[i] = in[i];
}

// One block per row: each thread scans a strided slice, then the block keeps the
// largest value (lowest index on ties, matching the CPU's argmax).
__global__ void argmax_rows_kernel(int32_t* __restrict__ out, const float* __restrict__ x, int n) {
  __shared__ float best_v[1024];
  __shared__ int best_i[1024];
  const float* row = x + size_t(blockIdx.x) * n;
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = row[i];
    if (v > bv) { bv = v; bi = i; }  // i only increases, so the first max is kept
  }
  best_v[threadIdx.x] = bv;
  best_i[threadIdx.x] = bi;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const float ov = best_v[threadIdx.x + stride];
      const int oi = best_i[threadIdx.x + stride];
      if (ov > best_v[threadIdx.x] || (ov == best_v[threadIdx.x] && oi < best_i[threadIdx.x])) {
        best_v[threadIdx.x] = ov;
        best_i[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = best_i[0];
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

// Tensor cores for prefill-sized calls. Below 32 rows (decode, small batches) the
// 64-row tiles would be mostly empty and a 896-row matrix would make only 14
// blocks, too few to keep 24 SMs busy; the warp-per-row kernel is faster there,
// and it computes each row exactly as it would alone, so batching doesn't change
// any sequence's output.
constexpr int kTcMinTokens = 32;
bool use_tensor_cores(int T, int cols) {
  return g_tensor_cores && T >= kTcMinTokens && cols % kTileK == 0;
}

template <class W, bool kGateUp>
void launch_tc(float* y, const float* x, const DeviceMatrix& A, const DeviceMatrix* B,
               const float* bias, int T, bool accumulate, cudaStream_t stream) {
  const dim3 grid((A.rows + kTileN - 1) / kTileN, (T + kTileM - 1) / kTileM);
  using E = typename W::Elem;
  tc_gemm_kernel<W, kGateUp><<<grid, kTcThreads, 0, stream>>>(
      y, x, static_cast<const E*>(A.data), A.scale, B ? static_cast<const E*>(B->data) : nullptr,
      B ? B->scale : nullptr, bias, T, A.rows, A.cols, accumulate);
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

void rope_and_cache(float* q, const float* k, const float* v, float* k_pool, float* v_pool,
                    const float* inv_freq, int T, const int* row_pos, const int* row_seq,
                    KvLayout kv, int n_heads, int n_kv_heads, int head_dim, cudaStream_t stream) {
  if (kv.page_size != kKvPageSize) throw std::runtime_error("rope_and_cache: KV page size must be 16");
  const int threads = std::min(1024, std::max(32, (n_heads + n_kv_heads) * head_dim / 2));
  rope_cache_kernel<<<T, threads, 0, stream>>>(q, k, v, k_pool, v_pool, inv_freq, row_pos, row_seq,
                                               kv, n_heads, n_kv_heads, head_dim);
  check_launch("rope_and_cache");
}

int attention_chunks(int context_tokens) { return (context_tokens + kAttnChunk - 1) / kAttnChunk; }

void attention(float* out, const float* q, const float* k_pool, const float* v_pool, int T,
               const int* row_pos, const int* row_seq, KvLayout kv, int num_chunks,
               float* part_o, float* part_ml, int n_heads, int n_kv_heads, int head_dim,
               cudaStream_t stream) {
  if (head_dim % 32 != 0 || head_dim > kMaxHeadDim)
    throw std::runtime_error("attention: head_dim must be a multiple of 32, at most 128");
  if (kv.page_size != kKvPageSize) throw std::runtime_error("attention: KV page size must be 16");
  const dim3 grid(n_heads, T, num_chunks);
  attention_partial_kernel<<<grid, kAttnWarps * 32, 0, stream>>>(
      part_o, part_ml, q, k_pool, v_pool, row_pos, row_seq, kv, n_heads, n_kv_heads, head_dim,
      1.0f / sqrtf(float(head_dim)), num_chunks);
  check_launch("attention_partial");
  attention_merge_kernel<<<dim3(n_heads, T), head_dim, 0, stream>>>(out, part_o, part_ml, n_heads,
                                                                    head_dim, num_chunks);
  check_launch("attention_merge");
}

void gather_rows(float* dst, const float* src, const int* rows, int R, int dim, cudaStream_t stream) {
  gather_rows_kernel<<<R, 256, 0, stream>>>(dst, src, rows, dim);
  check_launch("gather_rows");
}

void argmax_rows(int32_t* out, const float* x, int R, int n, cudaStream_t stream) {
  argmax_rows_kernel<<<R, 1024, 0, stream>>>(out, x, n);
  check_launch("argmax_rows");
}

}  // namespace ie::cuda
