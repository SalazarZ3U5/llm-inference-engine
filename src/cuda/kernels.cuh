#pragma once
// GPU kernels for the forward pass. Each function queues one kernel on
// `stream`; all pointers are device pointers.
//
// Layout conventions match the CPU code: activations are row-major [T, dim];
// weight matrices are [rows, cols] (nn.Linear's [out, in]); everything that
// isn't a weight matrix is FP32.
//
// The current position (number of tokens already in the KV cache) is read from
// device memory (`pos`), not passed as an argument. That way a recorded CUDA
// graph of the decode step stays valid for every position: only the value in
// memory changes between replays.

#include <cuda_runtime.h>

#include <cstdint>

namespace ie::cuda {

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
//   T = 1 (decode): one warp per output row, streaming weights at full bandwidth.
//   T >= 2 (prefill): tensor cores on 64 x 64 output tiles (see set_tensor_cores_impl).
void matmul(float* y, const float* x, const DeviceMatrix& W, const float* bias, int T,
            bool accumulate, cudaStream_t stream);

// out[t, o] = silu(dot(Wg[o], x[t])) * dot(Wu[o], x[t])
//   The MLP's gate and up projections and the SwiGLU, fused into one kernel.
void gate_up_silu(float* out, const float* x, const DeviceMatrix& Wg, const DeviceMatrix& Wu,
                  int T, cudaStream_t stream);

// Applies RoPE to q (in place) and k, then writes k and v into the KV cache at
// rows *pos .. *pos+T-1. inv_freq: [head_dim / 2] rotation frequencies.
void rope_and_cache(float* q, const float* k, const float* v, float* k_cache, float* v_cache,
                    const float* inv_freq, int T, const int* pos, int n_heads, int n_kv_heads,
                    int head_dim, cudaStream_t stream);

// Causal grouped-query attention for T new tokens over the cache (*pos + T rows).
//   q: [T, n_heads*head_dim]   k_cache, v_cache: [max_tokens, n_kv_heads*head_dim]
//   out: [T, n_heads*head_dim]. Shared memory is sized for max_tokens.
void attention(float* out, const float* q, const float* k_cache, const float* v_cache, int T,
               const int* pos, int max_tokens, int n_heads, int n_kv_heads, int head_dim,
               cudaStream_t stream);

// Largest context the attention kernel supports (its scores live in shared memory).
int attention_max_tokens();

// Multi-token matmuls (prefill) use tensor cores by default; switch off to compare.
void set_tensor_cores_impl(bool on);
bool tensor_cores();

}  // namespace ie::cuda
