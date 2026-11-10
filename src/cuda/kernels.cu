#include "cuda/kernels.cuh"

#include <cuda_runtime.h>

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

__device__ __forceinline__ float bf16_to_f32(uint16_t h) { return __uint_as_float(uint32_t(h) << 16); }

// 8 BF16 values packed in 16 bytes -> 8 floats (element 0 is the low half of .x).
__device__ __forceinline__ void unpack8(const uint4 p, float w[8]) {
  w[0] = __uint_as_float(p.x << 16);
  w[1] = __uint_as_float(p.x & 0xFFFF0000u);
  w[2] = __uint_as_float(p.y << 16);
  w[3] = __uint_as_float(p.y & 0xFFFF0000u);
  w[4] = __uint_as_float(p.z << 16);
  w[5] = __uint_as_float(p.z & 0xFFFF0000u);
  w[6] = __uint_as_float(p.w << 16);
  w[7] = __uint_as_float(p.w & 0xFFFF0000u);
}

// The core of every matmul: one warp computes dot(wrow, x[i]) for nt <= NT tokens.
// Lane j reads weights [8j, 8j+8), [8j+256, ...), ...: the 32 lanes together
// read 512 consecutive bytes per step, which is what GPU memory is fastest at
// (a "coalesced" access). Each weight is converted once and used for all nt tokens.
template <int NT>
__device__ __forceinline__ void warp_dot(const uint16_t* __restrict__ wrow,
                                         const float* __restrict__ x, int cols, int nt,
                                         float acc[NT]) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < NT; ++i) acc[i] = 0.0f;
  if ((cols & 7) == 0) {  // fast path: 16-byte loads (every row of this model)
    for (int k = lane * 8; k < cols; k += 32 * 8) {
      float w[8];
      unpack8(__ldg(reinterpret_cast<const uint4*>(wrow + k)), w);
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        if (i < nt) {
          const float4* xp = reinterpret_cast<const float4*>(x + size_t(i) * cols + k);
          const float4 a = xp[0], b = xp[1];
          float s = acc[i];
          s = fmaf(w[0], a.x, s); s = fmaf(w[1], a.y, s);
          s = fmaf(w[2], a.z, s); s = fmaf(w[3], a.w, s);
          s = fmaf(w[4], b.x, s); s = fmaf(w[5], b.y, s);
          s = fmaf(w[6], b.z, s); s = fmaf(w[7], b.w, s);
          acc[i] = s;
        }
      }
    }
  } else {  // general path: one weight per lane per step
    for (int k = lane; k < cols; k += 32) {
      const float w = bf16_to_f32(wrow[k]);
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

__global__ void embedding_kernel(float* __restrict__ x, const int32_t* __restrict__ ids,
                                 const uint16_t* __restrict__ embed, int hidden) {
  const int t = blockIdx.x;
  const uint16_t* row = embed + size_t(ids[t]) * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) x[size_t(t) * hidden + i] = bf16_to_f32(row[i]);
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

template <int NT>
__global__ void matmul_kernel(float* __restrict__ y, const float* __restrict__ x,
                              const uint16_t* __restrict__ W, const float* __restrict__ bias,
                              int T, int rows, int cols, bool accumulate) {
  const int row = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  if (row >= rows) return;  // whole warp leaves together, so shuffles stay safe
  const uint16_t* wrow = W + size_t(row) * cols;
  const float b = bias ? bias[row] : 0.0f;
  for (int t0 = 0; t0 < T; t0 += NT) {
    const int nt = min(NT, T - t0);
    float acc[NT];
    warp_dot<NT>(wrow, x + size_t(t0) * cols, cols, nt, acc);
    if ((threadIdx.x & 31) == 0) {
      for (int i = 0; i < nt; ++i) {
        float* dst = y + size_t(t0 + i) * rows + row;
        const float v = acc[i] + b;
        *dst = accumulate ? *dst + v : v;
      }
    }
  }
}

template <int NT>
__global__ void gate_up_silu_kernel(float* __restrict__ out, const float* __restrict__ x,
                                    const uint16_t* __restrict__ Wg,
                                    const uint16_t* __restrict__ Wu, int T, int rows, int cols) {
  const int row = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  if (row >= rows) return;
  for (int t0 = 0; t0 < T; t0 += NT) {
    const int nt = min(NT, T - t0);
    float g[NT], u[NT];
    warp_dot<NT>(Wg + size_t(row) * cols, x + size_t(t0) * cols, cols, nt, g);
    warp_dot<NT>(Wu + size_t(row) * cols, x + size_t(t0) * cols, cols, nt, u);
    if ((threadIdx.x & 31) == 0)
      for (int i = 0; i < nt; ++i)
        out[size_t(t0 + i) * rows + row] = g[i] / (1.0f + expf(-g[i])) * u[i];
  }
}

__global__ void rope_cache_kernel(float* __restrict__ q, const float* __restrict__ k,
                                  const float* __restrict__ v, float* __restrict__ k_cache,
                                  float* __restrict__ v_cache, const float* __restrict__ inv_freq,
                                  int pos0, int n_heads, int n_kv_heads, int head_dim) {
  const int t = blockIdx.x, pos = pos0 + t, half = head_dim / 2;
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
                                 const float* __restrict__ v_cache, int pos0, int n_heads,
                                 int n_kv_heads, int head_dim, float scale) {
  extern __shared__ float smem[];
  const int h = blockIdx.x, t = blockIdx.y, p = pos0 + t;
  const int kvh = h / (n_heads / n_kv_heads);  // GQA
  const int q_dim = n_heads * head_dim, kv_dim = n_kv_heads * head_dim;
  float* qs = smem;                // [head_dim]
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

}  // namespace

// ------------------------------------------------------------ launch ----

void embedding(float* x, const int32_t* ids, const uint16_t* embed, int T, int hidden) {
  embedding_kernel<<<T, 256>>>(x, ids, embed, hidden);
  check_launch("embedding");
}

void rmsnorm(float* y, const float* x, const float* weight, int T, int dim, float eps) {
  rmsnorm_kernel<<<T, 256>>>(y, x, weight, dim, eps);
  check_launch("rmsnorm");
}

void matmul_bf16(float* y, const float* x, const uint16_t* W, const float* bias, int T, int rows,
                 int cols, bool accumulate) {
  const int blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (T == 1)
    matmul_kernel<1><<<blocks, kWarpsPerBlock * 32>>>(y, x, W, bias, T, rows, cols, accumulate);
  else
    matmul_kernel<4><<<blocks, kWarpsPerBlock * 32>>>(y, x, W, bias, T, rows, cols, accumulate);
  check_launch("matmul_bf16");
}

void gate_up_silu(float* out, const float* x, const uint16_t* Wg, const uint16_t* Wu, int T,
                  int rows, int cols) {
  const int blocks = (rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (T == 1)
    gate_up_silu_kernel<1><<<blocks, kWarpsPerBlock * 32>>>(out, x, Wg, Wu, T, rows, cols);
  else
    gate_up_silu_kernel<4><<<blocks, kWarpsPerBlock * 32>>>(out, x, Wg, Wu, T, rows, cols);
  check_launch("gate_up_silu");
}

void rope_and_cache(float* q, const float* k, const float* v, float* k_cache, float* v_cache,
                    const float* inv_freq, int T, int pos0, int n_heads, int n_kv_heads,
                    int head_dim) {
  const int threads = std::min(1024, std::max(32, (n_heads + n_kv_heads) * head_dim / 2));
  rope_cache_kernel<<<T, threads>>>(q, k, v, k_cache, v_cache, inv_freq, pos0, n_heads,
                                    n_kv_heads, head_dim);
  check_launch("rope_and_cache");
}

int attention_max_tokens() {
  return max_attention_smem_bytes() / int(sizeof(float)) - 256;  // leave room for q (head_dim <= 256)
}

void attention(float* out, const float* q, const float* k_cache, const float* v_cache, int T,
               int pos0, int n_heads, int n_kv_heads, int head_dim) {
  const size_t smem = sizeof(float) * (size_t(head_dim) + pos0 + T);
  if (smem > size_t(max_attention_smem_bytes()))
    throw std::runtime_error("attention: context too long for the shared-memory kernel");
  const dim3 grid(n_heads, T);
  attention_kernel<<<grid, 128, smem>>>(out, q, k_cache, v_cache, pos0, n_heads, n_kv_heads,
                                        head_dim, 1.0f / sqrtf(float(head_dim)));
  check_launch("attention");
}

}  // namespace ie::cuda
