#include "cuda/kernels.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <type_traits>

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

// The core of every CUDA-core matmul: one warp computes dot(W[r], x[i]) for R
// weight rows r and nt <= NT tokens i. Lane j reads 16 bytes of each row at
// [j*16, j*16+16), then 512 bytes further, and so on: the 32 lanes together read
// 512 consecutive bytes per step, which is what GPU memory is fastest at (a
// "coalesced" access). Each weight is converted once and used for all nt
// tokens, and each activation load is used for all R rows (with R = 1 every
// row would re-read the activations, and small batches would be limited by
// on-chip bandwidth instead of by the weights).
//
// For every (row, token), the lane adds its products in the same order and the
// warp sums lanes the same way whatever R, NT and nt are, so a token's result is
// bit-identical whether it is computed alone or in a batch.
template <class W, int NT, int R>
__device__ __forceinline__ void warp_dot(const typename W::Elem* const (&wrow)[R],
                                         const float* __restrict__ x, int cols, int nt,
                                         float (&acc)[R][NT]) {
  constexpr int P = W::kPerLoad;
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int i = 0; i < NT; ++i) acc[r][i] = 0.0f;
  if (cols % P == 0) {  // fast path: 16-byte loads (every row of this model)
    for (int k = lane * P; k < cols; k += 32 * P) {
      float w[R][P];
#pragma unroll
      for (int r = 0; r < R; ++r) W::unpack(__ldg(reinterpret_cast<const uint4*>(wrow[r] + k)), w[r]);
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        if (i < nt) {
          const float4* xp = reinterpret_cast<const float4*>(x + size_t(i) * cols + k);
          float4 a[P / 4];
#pragma unroll
          for (int j = 0; j < P / 4; ++j) a[j] = xp[j];
#pragma unroll
          for (int r = 0; r < R; ++r) {
            float s = acc[r][i];
#pragma unroll
            for (int j = 0; j < P / 4; ++j) {
              s = fmaf(w[r][4 * j + 0], a[j].x, s);
              s = fmaf(w[r][4 * j + 1], a[j].y, s);
              s = fmaf(w[r][4 * j + 2], a[j].z, s);
              s = fmaf(w[r][4 * j + 3], a[j].w, s);
            }
            acc[r][i] = s;
          }
        }
      }
    }
  } else {  // general path: one weight per lane per step
    for (int k = lane; k < cols; k += 32) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const float w = W::load1(wrow[r], k);
#pragma unroll
        for (int i = 0; i < NT; ++i)
          if (i < nt) acc[r][i] = fmaf(w, x[size_t(i) * cols + k], acc[r][i]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int i = 0; i < NT; ++i) acc[r][i] = warp_sum(acc[r][i]);
}

constexpr int kWarpsPerBlock = 8;  // 256 threads = 8 output rows per block

// Offset (in floats) of position `pos` of sequence `seq` in a layer's page pool.
// The KV cache holds FP32 or FP16 values (KvLayout::f16); all math is FP32.
__device__ __forceinline__ float kv_load(const float* p) { return *p; }
__device__ __forceinline__ float kv_load(const __half* p) { return __half2float(*p); }
__device__ __forceinline__ void kv_store(float* p, float v) { *p = v; }
__device__ __forceinline__ void kv_store(__half* p, float v) { *p = __float2half_rn(v); }
// Four consecutive values (16 bytes as FP32, 8 as FP16; p is aligned to that).
__device__ __forceinline__ float4 kv_load4(const float* p) { return *reinterpret_cast<const float4*>(p); }
__device__ __forceinline__ float4 kv_load4(const __half* p) {
  const uint2 u = *reinterpret_cast<const uint2*>(p);
  const float2 a = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
  const float2 b = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
  return make_float4(a.x, a.y, b.x, b.y);
}

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

// One warp per R consecutive output rows. Rows past the end re-read the last
// real row (results discarded), so every warp runs the same code.
template <class W, int NT, int R>
__global__ void matmul_kernel(float* __restrict__ y, const float* __restrict__ x,
                              const typename W::Elem* __restrict__ Wm,
                              const float* __restrict__ scale, const float* __restrict__ bias,
                              int T, int rows, int cols, bool accumulate) {
  const int row0 = (blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5)) * R;
  if (row0 >= rows) return;  // whole warp leaves together, so shuffles stay safe
  const typename W::Elem* wrow[R];
#pragma unroll
  for (int r = 0; r < R; ++r) wrow[r] = Wm + size_t(min(row0 + r, rows - 1)) * cols;
  for (int t0 = 0; t0 < T; t0 += NT) {
    const int nt = min(NT, T - t0);
    float acc[R][NT];
    warp_dot<W, NT, R>(wrow, x + size_t(t0) * cols, cols, nt, acc);
    if ((threadIdx.x & 31) == 0) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
        if (row >= rows) break;
        const float s = W::kScaled ? scale[row] : 1.0f, b = bias ? bias[row] : 0.0f;
        for (int i = 0; i < nt; ++i) {
          float* dst = y + size_t(t0 + i) * rows + row;
          const float v = acc[r][i] * s + b;
          *dst = accumulate ? *dst + v : v;
        }
      }
    }
  }
}

template <class W, int NT, int R>
__global__ void gate_up_silu_kernel(float* __restrict__ out, const float* __restrict__ x,
                                    const typename W::Elem* __restrict__ Wg,
                                    const float* __restrict__ sg,
                                    const typename W::Elem* __restrict__ Wu,
                                    const float* __restrict__ su, int T, int rows, int cols) {
  const int row0 = (blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5)) * R;
  if (row0 >= rows) return;
  // Gate and up rows go through one warp_dot call as 2R rows, sharing the activation loads.
  const typename W::Elem* wrow[2 * R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const size_t off = size_t(min(row0 + r, rows - 1)) * cols;
    wrow[r] = Wg + off;
    wrow[R + r] = Wu + off;
  }
  for (int t0 = 0; t0 < T; t0 += NT) {
    const int nt = min(NT, T - t0);
    float acc[2 * R][NT];
    warp_dot<W, NT, 2 * R>(wrow, x + size_t(t0) * cols, cols, nt, acc);
    if ((threadIdx.x & 31) == 0) {
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
        if (row >= rows) break;
        const float scale_g = W::kScaled ? sg[row] : 1.0f, scale_u = W::kScaled ? su[row] : 1.0f;
        for (int i = 0; i < nt; ++i) {
          const float gi = acc[r][i] * scale_g;
          out[size_t(t0 + i) * rows + row] = gi / (1.0f + expf(-gi)) * (acc[R + r][i] * scale_u);
        }
      }
    }
  }
}

template <class KV>
__global__ void rope_cache_kernel(float* __restrict__ q, const float* __restrict__ k,
                                  const float* __restrict__ v, KV* __restrict__ k_pool,
                                  KV* __restrict__ v_pool, const float* __restrict__ inv_freq,
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
      KV* dst = k_pool + dst_row + kh * head_dim;  // rotated key goes straight to the cache
      const float x0 = src[i], x1 = src[i + half];
      kv_store(dst + i, x0 * c - x1 * s);
      kv_store(dst + i + half, x1 * c + x0 * s);
    }
  }
  for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) kv_store(v_pool + dst_row + i, v[size_t(t) * kv_dim + i]);
}

// ------------------------------------------- flash-decoding attention ----
//
// The cached positions 0..p that a query can see are cut into chunks of
// kAttnChunk. Each block handles one (KV head, query token, chunk) and produces
// a *partial* softmax result over just its chunk, for every query head that
// shares that KV head (grouped-query attention: 7 query heads per KV head here):
//   m = max score in the chunk
//   l = sum over the chunk of exp(score - m)
//   o = sum over the chunk of exp(score - m) * value      (not yet divided by l)
// A second kernel merges the chunks exactly:
//   M = max m_c,  L = sum l_c * exp(m_c - M),  out = sum o_c * exp(m_c - M) / L
// So a long context is spread over many blocks instead of looping inside one,
// each K/V row is read once for all the query heads of its group, and short
// chunks give even a single decoding sequence enough blocks to fill the GPU.
//
// The grid always has `num_chunks` chunks per query; chunks that start past the
// query's position exit at once. That keeps the launch identical at every
// position, which CUDA graphs need.

constexpr int kAttnChunk = 64;    // cached positions per block
constexpr int kAttnWarps = 4;     // 128 threads
constexpr int kMaxHeadDim = 128;  // lanes hold head_dim / 32 <= 4 values each
constexpr int kMaxGroup = 8;      // query heads per KV head that one block can serve
constexpr int kAttnUnroll = 4;    // positions per warp per loop step (memory-level parallelism)
constexpr int kChunkPages = kAttnChunk / kKvPageSize;  // pages one chunk spans
static_assert(kAttnChunk % kKvPageSize == 0, "a chunk must cover whole pages");
static_assert(kChunkPages <= kAttnWarps * 32, "one thread loads each page id");

// PL = head_dim / 32: values of one head each lane holds.
template <int PL, class KV>
__global__ void __launch_bounds__(kAttnWarps * 32)
attention_partial_kernel(float* __restrict__ part_o, float* __restrict__ part_ml,
                         const float* __restrict__ q, const KV* __restrict__ k_pool,
                         const KV* __restrict__ v_pool, const int* __restrict__ row_pos,
                         const int* __restrict__ row_seq, KvLayout kv, int n_heads,
                         int n_kv_heads, float scale, int num_chunks) {
  constexpr int HD = PL * 32;
  __shared__ float scores[kMaxGroup][kAttnChunk];
  __shared__ float partial_o[kAttnWarps][kMaxGroup][HD];
  __shared__ float group_ml[kMaxGroup][2];
  __shared__ int chunk_pages[kChunkPages];
  const int kvh = blockIdx.x, t = blockIdx.y, c = blockIdx.z;
  const int group = n_heads / n_kv_heads, h0 = kvh * group;  // this block's query heads
  const int p = row_pos[t];  // last position this query may see (causal)
  const int seq = row_seq[t];
  const int s0 = c * kAttnChunk;
  auto slot_of = [&](int g) { return (size_t(t) * n_heads + h0 + g) * num_chunks + c; };
  if (s0 > p) {  // chunk lies entirely in the future: contributes nothing
    if (threadIdx.x < group) {
      part_ml[2 * slot_of(threadIdx.x)] = -INFINITY;
      part_ml[2 * slot_of(threadIdx.x) + 1] = 0.0f;
    }
    return;  // the whole block leaves together, so later barriers are safe
  }
  const int s1 = min(s0 + kAttnChunk, p + 1);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int q_dim = n_heads * HD, kv_dim = n_kv_heads * HD;

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

  // Each lane keeps its slice of every group head's query: dims lane, lane+32, ...
  float qr[kMaxGroup][PL];
#pragma unroll
  for (int g = 0; g < kMaxGroup; ++g)
#pragma unroll
    for (int j = 0; j < PL; ++j)
      qr[g][j] = g < group ? q[size_t(t) * q_dim + (h0 + g) * HD + lane + 32 * j] : 0.0f;

  // 1. Scores: one warp per position, lanes across the head dimension (the 32
  //    lanes read one contiguous key row), one dot product per group head, then
  //    sums by shuffles. Each warp works on kAttnUnroll positions at a time: their
  //    loads are independent, so all are in flight together.
  for (int s = s0 + warp; s < s1; s += kAttnWarps * kAttnUnroll) {
    float dot[kAttnUnroll][kMaxGroup];
#pragma unroll
    for (int u = 0; u < kAttnUnroll; ++u) {
      // Past the end, re-read the last valid row (and discard the result):
      // no branch, so the compiler issues all the loads back to back.
      const int su = min(s + u * kAttnWarps, s1 - 1);
      const KV* kr = k_pool + row_of(su) + kvh * HD;
      float kk[PL];
#pragma unroll
      for (int j = 0; j < PL; ++j) kk[j] = kv_load(kr + lane + 32 * j);
#pragma unroll
      for (int g = 0; g < kMaxGroup; ++g) {
        dot[u][g] = 0.0f;
#pragma unroll
        for (int j = 0; j < PL; ++j) dot[u][g] = fmaf(qr[g][j], kk[j], dot[u][g]);
      }
    }
#pragma unroll
    for (int u = 0; u < kAttnUnroll; ++u) {
      const int su = s + u * kAttnWarps;
#pragma unroll
      for (int g = 0; g < kMaxGroup; ++g) {
        if (g >= group) break;  // same for the whole warp
        const float d = warp_sum(dot[u][g]);
        if (lane == 0 && su < s1) scores[g][su - s0] = d * scale;
      }
    }
  }
  __syncthreads();

  // 2. Softmax within the chunk, one warp per group head (subtract the chunk max
  //    so expf can't overflow).
  const int n = s1 - s0;
  for (int g = warp; g < group; g += kAttnWarps) {
    float mx = -INFINITY;
    for (int i = lane; i < n; i += 32) mx = fmaxf(mx, scores[g][i]);
    mx = warp_max(mx);
    float sum = 0.0f;
    for (int i = lane; i < n; i += 32) {
      const float e = expf(scores[g][i] - mx);
      scores[g][i] = e;
      sum += e;
    }
    sum = warp_sum(sum);
    if (lane == 0) {
      group_ml[g][0] = mx;
      group_ml[g][1] = sum;
    }
  }
  __syncthreads();

  // 3. Weighted values: again one warp per position, lanes across dimensions,
  //    each value row read once for all group heads. Positions are added in
  //    increasing order within each warp.
  float acc[kMaxGroup][PL];
#pragma unroll
  for (int g = 0; g < kMaxGroup; ++g)
#pragma unroll
    for (int j = 0; j < PL; ++j) acc[g][j] = 0.0f;
  for (int s = s0 + warp; s < s1; s += kAttnWarps * kAttnUnroll) {
#pragma unroll
    for (int u = 0; u < kAttnUnroll; ++u) {
      // Past the end: weight 0 on the last valid row, and fmaf(0, v, acc) == acc.
      const int su = s + u * kAttnWarps, sc = min(su, s1 - 1);
      const KV* vr = v_pool + row_of(sc) + kvh * HD;
      float vv[PL];
#pragma unroll
      for (int j = 0; j < PL; ++j) vv[j] = kv_load(vr + lane + 32 * j);
#pragma unroll
      for (int g = 0; g < kMaxGroup; ++g) {
        if (g >= group) break;
        const float w = su < s1 ? scores[g][sc - s0] : 0.0f;
#pragma unroll
        for (int j = 0; j < PL; ++j) acc[g][j] = fmaf(w, vv[j], acc[g][j]);
      }
    }
  }
#pragma unroll
  for (int g = 0; g < kMaxGroup; ++g) {
    if (g >= group) break;
#pragma unroll
    for (int j = 0; j < PL; ++j) partial_o[warp][g][lane + 32 * j] = acc[g][j];
  }
  __syncthreads();

  // Add the 4 warps' partial outputs and store this chunk's results.
  for (int i = threadIdx.x; i < group * HD; i += blockDim.x) {
    const int g = i / HD, d = i % HD;
    float o = 0.0f;
#pragma unroll
    for (int w = 0; w < kAttnWarps; ++w) o += partial_o[w][g][d];
    part_o[slot_of(g) * HD + d] = o;
  }
  if (threadIdx.x < group) {
    part_ml[2 * slot_of(threadIdx.x)] = group_ml[threadIdx.x][0];
    part_ml[2 * slot_of(threadIdx.x) + 1] = group_ml[threadIdx.x][1];
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

// ---------------------------------------------- prefill attention ----
//
// A prompt's T query rows each attend to every earlier position, so the
// decode kernel above (each block reads the cache for one query) reads the K/V
// cache T times: memory traffic grows with T^2. Here a block takes kPfQ
// consecutive queries and all the query heads that share one KV head (7 for
// Qwen2.5-0.5B), loads K/V kPfKeys positions at a time into shared memory, and
// every one of those kPfQ x 7 queries uses each loaded tile: one thread per
// (query, head), holding its query and its output in registers.
//
// Softmax is computed "online" (as in FlashAttention): each thread keeps the
// running max m, the running sum l and the unnormalized output acc; when a new
// tile raises the max, l and acc are rescaled by exp(m_old - m_new). No scores
// are stored, and no partial results need merging.
//
// Rows must be consecutive positions of one sequence (a prompt, or the next
// part of one), which is what lets a block share its K/V tiles.

constexpr int kPfQ = 16;      // query tokens per block
constexpr int kPfSlots = 8;   // threads per query token: one per query head of the group (<= 8)
constexpr int kPfKeys = 16;   // cached positions per shared-memory tile
constexpr int kPfDim = 64;    // head_dim this kernel is compiled for
constexpr int kPfThreads = kPfQ * kPfSlots;  // 128

template <class KV>
__global__ void __launch_bounds__(kPfThreads)
attention_prefill_kernel(float* __restrict__ out, const float* __restrict__ q,
                         const KV* __restrict__ k_pool, const KV* __restrict__ v_pool,
                         int T, const int* __restrict__ row_pos, const int* __restrict__ row_seq,
                         KvLayout kv, int n_heads, int n_kv_heads, float scale) {
  constexpr int D = kPfDim;
  __shared__ __align__(16) float ks[kPfKeys][D];
  __shared__ __align__(16) float vs[kPfKeys][D];
  const int kvh = blockIdx.y, group = n_heads / n_kv_heads;
  const int t0 = blockIdx.x * kPfQ;
  const int qi = threadIdx.x / kPfSlots, g = threadIdx.x % kPfSlots, t = t0 + qi;
  const bool active = g < group && t < T;
  const int seq = row_seq[0];
  const int max_pos = row_pos[min(t0 + kPfQ, T) - 1];  // positions increase by 1 per row
  const int pos = active ? row_pos[t] : -1;            // inactive threads see nothing
  const int h = kvh * group + g;
  const int q_dim = n_heads * D, kv_dim = n_kv_heads * D;

  float qr[D], acc[D];
#pragma unroll
  for (int c = 0; c < D / 4; ++c) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (active) v = *reinterpret_cast<const float4*>(q + size_t(t) * q_dim + h * D + 4 * c);
    qr[4 * c + 0] = v.x * scale;  // fold 1/sqrt(head_dim) into the query once
    qr[4 * c + 1] = v.y * scale;
    qr[4 * c + 2] = v.z * scale;
    qr[4 * c + 3] = v.w * scale;
  }
#pragma unroll
  for (int d = 0; d < D; ++d) acc[d] = 0.0f;
  float m = -INFINITY, l = 0.0f;

  for (int s0 = 0; s0 <= max_pos; s0 += kPfKeys) {
    __syncthreads();  // everyone is done with the previous tile
    for (int i = threadIdx.x; i < kPfKeys * D / 4; i += kPfThreads) {
      const int j = i / (D / 4), c = (i % (D / 4)) * 4;
      const int sp = min(s0 + j, max_pos);  // past the end: a valid row, masked below
      const size_t off = kv_offset(kv, seq, sp, kv_dim) + kvh * D + c;
      *reinterpret_cast<float4*>(&ks[j][c]) = kv_load4(k_pool + off);
      *reinterpret_cast<float4*>(&vs[j][c]) = kv_load4(v_pool + off);
    }
    __syncthreads();

    // Scores for this tile. All threads of a warp read the same key row at the
    // same time, which shared memory serves as a single broadcast.
    float sc[kPfKeys];
    float tile_max = -INFINITY;
#pragma unroll
    for (int j = 0; j < kPfKeys; ++j) {
      float dot = 0.0f;
#pragma unroll
      for (int c = 0; c < D / 4; ++c) {
        const float4 kk = *reinterpret_cast<const float4*>(&ks[j][4 * c]);
        dot = fmaf(qr[4 * c + 0], kk.x, dot);
        dot = fmaf(qr[4 * c + 1], kk.y, dot);
        dot = fmaf(qr[4 * c + 2], kk.z, dot);
        dot = fmaf(qr[4 * c + 3], kk.w, dot);
      }
      sc[j] = s0 + j <= pos ? dot : -INFINITY;  // causal mask
      tile_max = fmaxf(tile_max, sc[j]);
    }
    const float m_new = fmaxf(m, tile_max);
    if (m_new == -INFINITY) continue;  // nothing visible to this thread (inactive)
    const float corr = expf(m - m_new);  // 0 on the first tile (m = -inf)
    l *= corr;
#pragma unroll
    for (int d = 0; d < D; ++d) acc[d] *= corr;
#pragma unroll
    for (int j = 0; j < kPfKeys; ++j) {
      const float p = expf(sc[j] - m_new);  // masked: exp(-inf) = 0
      l += p;
#pragma unroll
      for (int c = 0; c < D / 4; ++c) {
        const float4 vv = *reinterpret_cast<const float4*>(&vs[j][4 * c]);
        acc[4 * c + 0] = fmaf(p, vv.x, acc[4 * c + 0]);
        acc[4 * c + 1] = fmaf(p, vv.y, acc[4 * c + 1]);
        acc[4 * c + 2] = fmaf(p, vv.z, acc[4 * c + 2]);
        acc[4 * c + 3] = fmaf(p, vv.w, acc[4 * c + 3]);
      }
    }
    m = m_new;
  }

  if (!active) return;
  const float inv_l = 1.0f / l;
  float* o = out + size_t(t) * q_dim + h * D;
#pragma unroll
  for (int c = 0; c < D / 4; ++c)
    *reinterpret_cast<float4*>(o + 4 * c) =
        make_float4(acc[4 * c] * inv_l, acc[4 * c + 1] * inv_l, acc[4 * c + 2] * inv_l, acc[4 * c + 3] * inv_l);
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

// Tile loading is split in two so it can overlap with the math: fetch() issues
// the global loads for the next k step into registers, the MMAs on the current
// tile run while those loads are in flight, then store() converts and writes
// them into the other shared-memory buffer.
constexpr int kAPerThread = kTileM * kTileK / 4 / kTcThreads;  // float4 loads of x per thread

template <class W>
struct TcFetch {
  static constexpr int kPerLoad = W::kScaled ? 16 : 8;  // weights per 16-byte load
  static constexpr int kBPerThread = kTileN * kTileK / kPerLoad / kTcThreads;
  float4 a[kAPerThread];
  uint4 b[2][kBPerThread];
};

// A: 64 tokens x 32 columns of x.  B: 64 weight rows x 32 columns.
// Tokens past T and weight rows past N load as zeros.
template <class W, int kNumB>
__device__ __forceinline__ void tc_fetch(TcFetch<W>& f, const float* __restrict__ x,
                                         const typename W::Elem* const (&mats)[2], int T, int N,
                                         int K, int m0, int n0, int k0) {
#pragma unroll
  for (int i = 0; i < kAPerThread; ++i) {
    const int idx = threadIdx.x + i * kTcThreads;
    const int m = idx / (kTileK / 4), c = (idx % (kTileK / 4)) * 4;
    f.a[i] = m0 + m < T ? *reinterpret_cast<const float4*>(x + size_t(m0 + m) * K + k0 + c)
                        : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  constexpr int P = TcFetch<W>::kPerLoad;
#pragma unroll
  for (int b = 0; b < kNumB; ++b)
#pragma unroll
    for (int i = 0; i < TcFetch<W>::kBPerThread; ++i) {
      const int idx = threadIdx.x + i * kTcThreads;
      const int n = idx / (kTileK / P), c = (idx % (kTileK / P)) * P;
      f.b[b][i] = n0 + n < N ? __ldg(reinterpret_cast<const uint4*>(mats[b] + size_t(n0 + n) * K + k0 + c))
                             : make_uint4(0, 0, 0, 0);
    }
}

// x is split into two BF16 parts, x = hi + lo (see above); BF16 weights are
// copied as they are, INT8 weights converted exactly to BF16.
template <class W, int kNumB>
__device__ __forceinline__ void tc_store(const TcFetch<W>& f, __nv_bfloat16* a_hi,
                                         __nv_bfloat16* a_lo, __nv_bfloat16* const (&b_tile)[2]) {
#pragma unroll
  for (int i = 0; i < kAPerThread; ++i) {
    const int idx = threadIdx.x + i * kTcThreads;
    const int m = idx / (kTileK / 4), c = (idx % (kTileK / 4)) * 4;
    const float vals[4] = {f.a[i].x, f.a[i].y, f.a[i].z, f.a[i].w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const __nv_bfloat16 hi = __float2bfloat16_rn(vals[j]);
      a_hi[m * kLd + c + j] = hi;
      a_lo[m * kLd + c + j] = __float2bfloat16_rn(vals[j] - __bfloat162float(hi));
    }
  }
  constexpr int P = TcFetch<W>::kPerLoad;
#pragma unroll
  for (int b = 0; b < kNumB; ++b)
#pragma unroll
    for (int i = 0; i < TcFetch<W>::kBPerThread; ++i) {
      const int idx = threadIdx.x + i * kTcThreads;
      const int n = idx / (kTileK / P), c = (idx % (kTileK / P)) * P;
      if constexpr (!W::kScaled) {
        *reinterpret_cast<uint4*>(b_tile[b] + n * kLd + c) = f.b[b][i];
      } else {
        float w[16];
        W::unpack(f.b[b][i], w);
#pragma unroll
        for (int j = 0; j < 16; ++j) b_tile[b][n * kLd + c + j] = __float2bfloat16_rn(w[j]);
      }
    }
}

// kGateUp = false: y[t, n] (+)= (x·W^T)[t, n] * scale[n] + bias[n]
// kGateUp = true:  y[t, n]   = silu((x·Wg^T)[t, n] * sg[n]) * (x·Wu^T)[t, n] * su[n]
template <class W, bool kGateUp>
__global__ void __launch_bounds__(kTcThreads, 3)
tc_gemm_kernel(float* __restrict__ y, const float* __restrict__ x,
               const typename W::Elem* __restrict__ W0, const float* __restrict__ s0,
               const typename W::Elem* __restrict__ W1, const float* __restrict__ s1,
               const float* __restrict__ bias, int T, int N, int K, bool accumulate,
               int k_split, float* __restrict__ partial) {
  constexpr int kNumB = kGateUp ? 2 : 1;
  constexpr int kStageBytes = (2 + kNumB) * kTileBytes;   // A hi, A lo, B tile(s)
  constexpr int kMainBytes = 2 * kStageBytes;              // two stages: compute one, fill the other
  constexpr int kEpiBytes = kNumB * kTileM * kLdC * 4;     // FP32 output tile(s)
  __shared__ __align__(128) unsigned char smem[kMainBytes > kEpiBytes ? kMainBytes : kEpiBytes];
  auto a_hi = [&](int st) { return reinterpret_cast<__nv_bfloat16*>(smem + st * kStageBytes); };
  auto a_lo = [&](int st) { return reinterpret_cast<__nv_bfloat16*>(smem + st * kStageBytes + kTileBytes); };
  auto b_tiles = [&](int st, __nv_bfloat16* (&out)[2]) {
    for (int b = 0; b < 2; ++b)
      out[b] = reinterpret_cast<__nv_bfloat16*>(smem + st * kStageBytes + (2 + (b < kNumB ? b : 0)) * kTileBytes);
  };

  const int n0 = blockIdx.x * kTileN, m0 = blockIdx.y * kTileM;
  const int warp = threadIdx.x / 32, wm = warp / 2, wn = warp % 2;
  const typename W::Elem* const mats[2] = {W0, W1};

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[kNumB][2][2];
#pragma unroll
  for (int b = 0; b < kNumB; ++b)
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[b][i][j], 0.0f);

  // Split-K: block z covers columns [z * k_split, (z + 1) * k_split) only.
  const int k_begin = blockIdx.z * k_split, k_end = min(K, k_begin + k_split);
  TcFetch<W> f;
  {
    __nv_bfloat16* bt[2];
    b_tiles(0, bt);
    tc_fetch<W, kNumB>(f, x, mats, T, N, K, m0, n0, k_begin);
    tc_store<W, kNumB>(f, a_hi(0), a_lo(0), bt);
  }
  __syncthreads();
  int st = 0;
  for (int k0 = k_begin; k0 < k_end; k0 += kTileK) {
    const bool more = k0 + kTileK < k_end;
    if (more) tc_fetch<W, kNumB>(f, x, mats, T, N, K, m0, n0, k0 + kTileK);  // in flight during the MMAs
    __nv_bfloat16* bt[2];
    b_tiles(st, bt);
#pragma unroll
    for (int kk = 0; kk < kTileK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> ahi[2], alo[2];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        wmma::load_matrix_sync(ahi[i], a_hi(st) + (wm * 32 + i * 16) * kLd + kk, kLd);
        wmma::load_matrix_sync(alo[i], a_lo(st) + (wm * 32 + i * 16) * kLd + kk, kLd);
      }
#pragma unroll
      for (int b = 0; b < kNumB; ++b) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          // B is stored [n][k], i.e. column-major as a K x N matrix.
          wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> bf;
          wmma::load_matrix_sync(bf, bt[b] + (wn * 32 + j * 16) * kLd + kk, kLd);
#pragma unroll
          for (int i = 0; i < 2; ++i) {
            wmma::mma_sync(acc[b][i][j], ahi[i], bf, acc[b][i][j]);
            wmma::mma_sync(acc[b][i][j], alo[i], bf, acc[b][i][j]);
          }
        }
      }
    }
    if (more) {
      // The other stage was last read in the previous step, which every thread
      // finished before that step's barrier.
      __nv_bfloat16* nb[2];
      b_tiles(st ^ 1, nb);
      tc_store<W, kNumB>(f, a_hi(st ^ 1), a_lo(st ^ 1), nb);
    }
    __syncthreads();  // the new stage is complete, and nobody still reads the old one
    st ^= 1;
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
    if (partial) {  // split-K: raw sums only; splitk_reduce_kernel finishes the job
#pragma unroll
      for (int b = 0; b < kNumB; ++b)
        partial[((size_t(blockIdx.z) * kNumB + b) * T + gm) * N + gn] = c_tile[b][m * kLdC + n];
      continue;
    }
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

// Adds the split-K partial sums in a fixed order (deterministic, unlike atomics),
// then applies the same epilogue as tc_gemm_kernel. One thread per output.
template <class W, bool kGateUp>
__global__ void splitk_reduce_kernel(float* __restrict__ y, const float* __restrict__ partial,
                                     const float* __restrict__ s0, const float* __restrict__ s1,
                                     const float* __restrict__ bias, int T, int N, int splits,
                                     bool accumulate) {
  constexpr int kNumB = kGateUp ? 2 : 1;
  const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x, plane = size_t(T) * N;
  if (idx >= plane) return;
  const int gn = int(idx % N);
  float sum[kNumB] = {};
  for (int z = 0; z < splits; ++z)
#pragma unroll
    for (int b = 0; b < kNumB; ++b) sum[b] += partial[(size_t(z) * kNumB + b) * plane + idx];
  if constexpr (kGateUp) {
    const float g = sum[0] * (W::kScaled ? s0[gn] : 1.0f);
    const float u = sum[1] * (W::kScaled ? s1[gn] : 1.0f);
    y[idx] = g / (1.0f + expf(-g)) * u;
  } else {
    const float v = sum[0] * (W::kScaled ? s0[gn] : 1.0f) + (bias ? bias[gn] : 0.0f);
    y[idx] = accumulate ? y[idx] + v : v;
  }
}

// Split-K scratch, shared by every matmul (they run one after another on a stream).
float* g_splitk_ws = nullptr;
size_t g_splitk_floats = 0;
int g_num_sms = 0;

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

// How many ways to split K. A 64-token batch through a 896-row matrix makes only
// 14 output tiles: 14 blocks on a 24-SM GPU, each walking all K columns alone
// (4,864 for down_proj). Splitting K gives every SM work; the price is writing
// and re-reading the partial sums, which is small next to the weights.
// The target is 4 blocks per SM: this kernel waits on its loads with only 4 warps
// per block, so it needs several blocks per SM to keep memory busy.
int splitk_ways(int tiles, int K, int outputs) {
  const int target = 4 * g_num_sms;
  if (!g_splitk_ws || tiles >= target) return 1;
  const int steps = K / kTileK;
  int ways = std::min({(target + tiles - 1) / tiles, std::max(1, steps / 2), 16});
  while (ways > 1 && size_t(ways) * outputs > g_splitk_floats) --ways;
  return ways;
}

template <class W, bool kGateUp>
void launch_tc(float* y, const float* x, const DeviceMatrix& A, const DeviceMatrix* B,
               const float* bias, int T, bool accumulate, cudaStream_t stream) {
  constexpr int kNumB = kGateUp ? 2 : 1;
  const int N = A.rows, K = A.cols;
  dim3 grid((N + kTileN - 1) / kTileN, (T + kTileM - 1) / kTileM);
  const int ways = splitk_ways(int(grid.x * grid.y), K, kNumB * T * N);
  // Columns per split, a whole number of tiles; recount the splits that are non-empty.
  const int k_split = ((K / kTileK + ways - 1) / ways) * kTileK;
  const int splits = (K + k_split - 1) / k_split;
  grid.z = splits;
  using E = typename W::Elem;
  tc_gemm_kernel<W, kGateUp><<<grid, kTcThreads, 0, stream>>>(
      y, x, static_cast<const E*>(A.data), A.scale, B ? static_cast<const E*>(B->data) : nullptr,
      B ? B->scale : nullptr, bias, T, N, K, accumulate, k_split, splits > 1 ? g_splitk_ws : nullptr);
  if (splits > 1) {
    const size_t outputs = size_t(T) * N;
    splitk_reduce_kernel<W, kGateUp><<<unsigned((outputs + 255) / 256), 256, 0, stream>>>(
        y, g_splitk_ws, A.scale, B ? B->scale : nullptr, bias, T, N, splits, accumulate);
  }
}

// Decode (T = 1) streams the weights: one row per warp keeps the most loads in
// flight. Small batches (T = 2..31) reuse each activation load for several rows.
constexpr int kBatchRows = 4;     // rows per warp, matmul (T > 1)
constexpr int kBatchRowsGU = 2;   // rows per warp, gate_up (x2: gate and up)
constexpr int kBatchTokens = 8;   // tokens per pass over the weights (T > 1)

int blocks_for(int rows, int rows_per_warp) {
  const int per_block = kWarpsPerBlock * rows_per_warp;
  return (rows + per_block - 1) / per_block;
}

template <class W>
void launch_matmul(float* y, const float* x, const DeviceMatrix& M, const float* bias, int T,
                   bool accumulate, cudaStream_t stream) {
  const auto* data = static_cast<const typename W::Elem*>(M.data);
  if (T == 1)
    matmul_kernel<W, 1, 1><<<blocks_for(M.rows, 1), kWarpsPerBlock * 32, 0, stream>>>(
        y, x, data, M.scale, bias, T, M.rows, M.cols, accumulate);
  else
    matmul_kernel<W, kBatchTokens, kBatchRows>
        <<<blocks_for(M.rows, kBatchRows), kWarpsPerBlock * 32, 0, stream>>>(
            y, x, data, M.scale, bias, T, M.rows, M.cols, accumulate);
}

template <class W>
void launch_gate_up(float* out, const float* x, const DeviceMatrix& G, const DeviceMatrix& U,
                    int T, cudaStream_t stream) {
  const auto* g = static_cast<const typename W::Elem*>(G.data);
  const auto* u = static_cast<const typename W::Elem*>(U.data);
  if (T == 1)
    gate_up_silu_kernel<W, 1, 1><<<blocks_for(G.rows, 1), kWarpsPerBlock * 32, 0, stream>>>(
        out, x, g, G.scale, u, U.scale, T, G.rows, G.cols);
  else
    gate_up_silu_kernel<W, kBatchTokens, kBatchRowsGU>
        <<<blocks_for(G.rows, kBatchRowsGU), kWarpsPerBlock * 32, 0, stream>>>(
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

void init_matmul_workspace() {
  if (g_splitk_ws) return;
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess ||
      cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
    throw std::runtime_error("init_matmul_workspace: cannot query the GPU");
  constexpr size_t kFloats = size_t(8) << 20;  // 32 MiB, kept for the life of the process
  if (cudaMalloc(&g_splitk_ws, kFloats * sizeof(float)) != cudaSuccess)
    throw std::runtime_error("init_matmul_workspace: out of GPU memory");
  g_splitk_floats = kFloats;
}
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

void rope_and_cache(float* q, const float* k, const float* v, void* k_pool, void* v_pool,
                    const float* inv_freq, int T, const int* row_pos, const int* row_seq,
                    KvLayout kv, int n_heads, int n_kv_heads, int head_dim, cudaStream_t stream) {
  if (kv.page_size != kKvPageSize) throw std::runtime_error("rope_and_cache: KV page size must be 16");
  const int threads = std::min(1024, std::max(32, (n_heads + n_kv_heads) * head_dim / 2));
  if (kv.f16)
    rope_cache_kernel<__half><<<T, threads, 0, stream>>>(
        q, k, v, static_cast<__half*>(k_pool), static_cast<__half*>(v_pool), inv_freq, row_pos,
        row_seq, kv, n_heads, n_kv_heads, head_dim);
  else
    rope_cache_kernel<float><<<T, threads, 0, stream>>>(
        q, k, v, static_cast<float*>(k_pool), static_cast<float*>(v_pool), inv_freq, row_pos,
        row_seq, kv, n_heads, n_kv_heads, head_dim);
  check_launch("rope_and_cache");
}

bool attention_prefill_supported(int n_heads, int n_kv_heads, int head_dim) {
  return head_dim == kPfDim && n_kv_heads > 0 && n_heads % n_kv_heads == 0 &&
         n_heads / n_kv_heads <= kPfSlots;
}

void attention_prefill(float* out, const float* q, const void* k_pool, const void* v_pool, int T,
                       const int* row_pos, const int* row_seq, KvLayout kv, int n_heads,
                       int n_kv_heads, int head_dim, cudaStream_t stream) {
  if (!attention_prefill_supported(n_heads, n_kv_heads, head_dim))
    throw std::runtime_error("attention_prefill: needs head_dim 64 and at most 8 query heads per KV head");
  if (kv.page_size != kKvPageSize) throw std::runtime_error("attention_prefill: KV page size must be 16");
  const dim3 grid((T + kPfQ - 1) / kPfQ, n_kv_heads);
  const float scale = 1.0f / sqrtf(float(head_dim));
  if (kv.f16)
    attention_prefill_kernel<__half><<<grid, kPfThreads, 0, stream>>>(
        out, q, static_cast<const __half*>(k_pool), static_cast<const __half*>(v_pool), T, row_pos,
        row_seq, kv, n_heads, n_kv_heads, scale);
  else
    attention_prefill_kernel<float><<<grid, kPfThreads, 0, stream>>>(
        out, q, static_cast<const float*>(k_pool), static_cast<const float*>(v_pool), T, row_pos,
        row_seq, kv, n_heads, n_kv_heads, scale);
  check_launch("attention_prefill");
}

int attention_chunks(int context_tokens) { return (context_tokens + kAttnChunk - 1) / kAttnChunk; }

void attention(float* out, const float* q, const void* k_pool, const void* v_pool, int T,
               const int* row_pos, const int* row_seq, KvLayout kv, int num_chunks,
               float* part_o, float* part_ml, int n_heads, int n_kv_heads, int head_dim,
               cudaStream_t stream) {
  if (head_dim % 32 != 0 || head_dim > kMaxHeadDim)
    throw std::runtime_error("attention: head_dim must be a multiple of 32, at most 128");
  if (n_kv_heads <= 0 || n_heads % n_kv_heads != 0 || n_heads / n_kv_heads > kMaxGroup)
    throw std::runtime_error("attention: at most 8 query heads per KV head");
  if (kv.page_size != kKvPageSize) throw std::runtime_error("attention: KV page size must be 16");
  const dim3 grid(n_kv_heads, T, num_chunks);
  const float scale = 1.0f / sqrtf(float(head_dim));
  // One kernel per (values per lane, cache element type).
  const int pl = head_dim / 32;
#define IE_ATTN_CASE(PL, KV)                                                                      \
  if (pl == PL && kv.f16 == std::is_same<KV, __half>::value)                                      \
    attention_partial_kernel<PL, KV><<<grid, kAttnWarps * 32, 0, stream>>>(                       \
        part_o, part_ml, q, static_cast<const KV*>(k_pool), static_cast<const KV*>(v_pool),       \
        row_pos, row_seq, kv, n_heads, n_kv_heads, scale, num_chunks);
  IE_ATTN_CASE(1, float)
  IE_ATTN_CASE(2, float)
  IE_ATTN_CASE(3, float)
  IE_ATTN_CASE(4, float)
  IE_ATTN_CASE(1, __half)
  IE_ATTN_CASE(2, __half)
  IE_ATTN_CASE(3, __half)
  IE_ATTN_CASE(4, __half)
#undef IE_ATTN_CASE
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
