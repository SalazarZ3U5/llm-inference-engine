#pragma once
// GPU kernels for the forward pass. Each function queues one kernel on
// `stream`; all pointers are device pointers.
//
// Layout conventions match the CPU code: activations are row-major [T, dim];
// weight matrices are [rows, cols] (nn.Linear's [out, in]); everything that
// isn't a weight matrix is FP32.
//
// Every row (token) of a forward pass carries its own sequence id and position,
// read from device memory (`row_seq`, `row_pos`). One call can therefore mix
// tokens from different sequences (batching), and a recorded CUDA graph stays
// valid for every position: only the values in memory change between replays.
//
// The KV cache is *paged*: each layer has a pool of pages of `page_size` token
// rows, and each sequence has a page table listing its pages in order. Token
// position p of sequence s lives in page tables[s * max_pages + p / page_size],
// row p % page_size. Sequences grow one page at a time and never need one big
// contiguous block, so memory isn't reserved for the maximum length up front.

#include <cuda_runtime.h>

#include <cstdint>

namespace ie::cuda {

// Token rows per KV page. A compile-time constant so kernels find a position's
// page with a shift instead of a division; the attention chunk (256) is a multiple.
constexpr int kKvPageSize = 16;

// Where each sequence's KV rows live (see above). Passed to kernels by value.
// The cache stores FP32 values, or FP16 when f16 is set (half the memory and
// memory traffic; FP16 keeps ~3 significant digits, and all math stays FP32).
struct KvLayout {
  const int* tables = nullptr;  // [num_sequences * max_pages] page ids
  int max_pages = 0;            // row stride of `tables`
  int page_size = kKvPageSize;  // must equal kKvPageSize (checked at launch)
  bool f16 = false;             // element type of k_pool / v_pool: false = float, true = __half
};

// A weight matrix in VRAM, stored as BF16 or as INT8 with one scale per row.
struct DeviceMatrix {
  const void* data = nullptr;    // uint16 BF16 bits or int8 values, [rows, cols]
  const float* scale = nullptr;  // INT8 only: [rows]
  bool int8 = false;
  int rows = 0, cols = 0;
};

// x[t, :] = E[ids[t], :] converted to FP32
void embedding(float* x, const int32_t* ids, const DeviceMatrix& E, int T, cudaStream_t stream);

// y[t, :] = x[t, :] / rms(x[t, :]) * weight     (y may equal x)
void rmsnorm(float* y, const float* x, const float* weight, int T, int dim, float eps,
             cudaStream_t stream);

// y[t, o] (+)= dot(W[o, :], x[t, :]) + bias[o]
//   accumulate = true adds into y (fuses the residual connection: x += W·a).
//   T < 32 (decode, small batches): one warp per output row, streaming weights at
//     full bandwidth; a token's result doesn't depend on the other rows in the batch.
//   T >= 32 (prefill, big batches): tensor cores on 64 x 64 output tiles, with K split
//     across blocks when there are too few tiles to fill the GPU (see init_matmul_workspace).
void matmul(float* y, const float* x, const DeviceMatrix& W, const float* bias, int T,
            bool accumulate, cudaStream_t stream);

// out[t, o] = silu(dot(Wg[o], x[t])) * dot(Wu[o], x[t])
//   The MLP's gate and up projections and the SwiGLU, fused into one kernel.
void gate_up_silu(float* out, const float* x, const DeviceMatrix& Wg, const DeviceMatrix& Wu,
                  int T, cudaStream_t stream);

// Applies RoPE to q (in place) and k, then writes row t's key and value into the
// paged cache of its sequence at its position. inv_freq: [head_dim / 2].
//   k_pool, v_pool: this layer's page pool, [num_pages * page_size, kv_dim], float or __half
void rope_and_cache(float* q, const float* k, const float* v, void* k_pool, void* v_pool,
                    const float* inv_freq, int T, const int* row_pos, const int* row_seq,
                    KvLayout kv, int n_heads, int n_kv_heads, int head_dim, cudaStream_t stream);

// Causal grouped-query attention for T rows, each over its own sequence's cached
// positions 0..row_pos[t], "flash-decoding" style: cached positions are split
// into chunks handled by separate blocks, then merged exactly.
//   q: [T, n_heads*head_dim]   out: [T, n_heads*head_dim]
//   num_chunks: attention_chunks(context) for the longest context this launch may
//     see (the maximum sequence length when recording a graph; extra chunks exit).
//   part_o: scratch [T * n_heads * num_chunks * head_dim]
//   part_ml: scratch [T * n_heads * num_chunks * 2]
void attention(float* out, const float* q, const void* k_pool, const void* v_pool, int T,
               const int* row_pos, const int* row_seq, KvLayout kv, int num_chunks,
               float* part_o, float* part_ml, int n_heads, int n_kv_heads, int head_dim,
               cudaStream_t stream);

// The same attention for a prompt: T rows that are consecutive positions of
// ONE sequence (row_pos[t] = row_pos[0] + t), e.g. a prompt or the next turn of
// a chat. Blocks share K/V tiles between 16 queries and all query heads of a KV
// head, instead of reading the cache once per query; no scratch needed.
bool attention_prefill_supported(int n_heads, int n_kv_heads, int head_dim);
void attention_prefill(float* out, const float* q, const void* k_pool, const void* v_pool, int T,
                       const int* row_pos, const int* row_seq, KvLayout kv, int n_heads,
                       int n_kv_heads, int head_dim, cudaStream_t stream);

// Number of chunks needed to cover `context_tokens` cached positions.
int attention_chunks(int context_tokens);

// dst[r, :] = src[rows[r], :] for r < R   (picks the rows that need logits)
void gather_rows(float* dst, const float* src, const int* rows, int R, int dim, cudaStream_t stream);

// out[r] = index of the largest value in x[r, 0..n), first one on ties (greedy decoding
// on the GPU, so only R ints come back instead of R x vocab floats).
void argmax_rows(int32_t* out, const float* x, int R, int n, cudaStream_t stream);

// Allocates the scratch the tensor-core matmul uses to split small problems
// across more blocks ("split-K"). Call once before launching (and before
// recording a CUDA graph); without it, matmuls simply don't split.
void init_matmul_workspace();

// Multi-token matmuls (prefill) use tensor cores by default; switch off to compare.
void set_tensor_cores_impl(bool on);
// Tensor-core matmuls run two BF16 passes by default (x = hi + lo, FP32-level
// accuracy); on = one FP16 pass (x rounded to FP16 once): half the tensor-core
// work, ~3 significant digits on the activations, as most engines do.
void set_tc_fp16_impl(bool on);
bool tc_fp16();
bool tensor_cores();

}  // namespace ie::cuda
