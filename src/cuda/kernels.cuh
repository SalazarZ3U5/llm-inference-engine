#pragma once
// GPU kernels for the forward pass. Each function launches one kernel on the
// default stream; all pointers are device pointers.
//
// Layout conventions match the CPU code: activations are row-major [T, dim];
// weight matrices are [rows, cols] (nn.Linear's [out, in]) stored as BF16 bits
// (uint16_t); everything else is FP32.

#include <cstdint>

namespace ie::cuda {

// x[t, :] = embed[ids[t], :]   (BF16 -> FP32)
void embedding(float* x, const int32_t* ids, const uint16_t* embed, int T, int hidden);

// y[t, :] = x[t, :] / rms(x[t, :]) * weight     (y may equal x)
void rmsnorm(float* y, const float* x, const float* weight, int T, int dim, float eps);

// y[t, o] (+)= dot(W[o, :], x[t, :]) + bias[o]
//   accumulate = true adds into y (fuses the residual connection: x += W·a).
//   One warp per output row; each weight is loaded once per group of 4 tokens.
void matmul_bf16(float* y, const float* x, const uint16_t* W, const float* bias, int T, int rows,
                 int cols, bool accumulate);

// out[t, o] = silu(dot(Wg[o], x[t])) * dot(Wu[o], x[t])
//   The MLP's gate and up projections and the SwiGLU, fused into one kernel.
void gate_up_silu(float* out, const float* x, const uint16_t* Wg, const uint16_t* Wu, int T,
                  int rows, int cols);

// Applies RoPE to q (in place) and k, then writes k and v into the KV cache at
// rows pos0 .. pos0+T-1. inv_freq: [head_dim / 2] rotation frequencies.
void rope_and_cache(float* q, const float* k, const float* v, float* k_cache, float* v_cache,
                    const float* inv_freq, int T, int pos0, int n_heads, int n_kv_heads,
                    int head_dim);

// Causal grouped-query attention for T new tokens over the cache (pos0 + T rows).
//   q: [T, n_heads*head_dim]   k_cache, v_cache: [>= pos0+T, n_kv_heads*head_dim]
//   out: [T, n_heads*head_dim]
void attention(float* out, const float* q, const float* k_cache, const float* v_cache, int T,
               int pos0, int n_heads, int n_kv_heads, int head_dim);

// Largest context the attention kernel supports (its scores live in shared memory).
int attention_max_tokens();

}  // namespace ie::cuda
