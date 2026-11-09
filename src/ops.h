#pragma once
// CPU reference implementations of every operation in the forward pass.
// Plain loops in FP32, written for clarity rather than speed. They stay in the
// project permanently: faster CPU code and CUDA kernels are tested against them.
//
// Convention: activations are row-major [T, dim], one row per token.

namespace ie {

// y[t][o] = sum_i W[o][i] * x[t][i] + b[o]
//   x: [T, in]   W: [out, in]   b: [out] or nullptr   y: [T, out]
void linear(float* y, const float* x, const float* W, const float* b, int T, int in, int out);

// y[t][i] = x[t][i] / sqrt(mean_j(x[t][j]^2) + eps) * weight[i]
//   x, y: [T, dim]   weight: [dim]   (y may equal x)
void rmsnorm(float* y, const float* x, const float* weight, int T, int dim, float eps);

// Rotary position embedding, applied in place.
//   x: [T, n_heads * head_dim]; token t sits at position pos0 + t.
//   Pairs element i with element i + head_dim/2 ("rotate half", as in HF Llama/Qwen).
void rope(float* x, int T, int n_heads, int head_dim, int pos0, float theta);

// Numerically stable softmax over n values, in place.
void softmax(float* x, int n);

// SiLU (a.k.a. swish): x * sigmoid(x)
float silu(float x);

// Causal multi-head attention with grouped-query support.
//   q: [T, n_heads * head_dim]   k, v: [T, n_kv_heads * head_dim]   out: [T, n_heads * head_dim]
//   Query head h reads key/value head h / (n_heads / n_kv_heads).
//   Token t attends to tokens 0..t only.
void attention(float* out, const float* q, const float* k, const float* v, int T, int n_heads,
               int n_kv_heads, int head_dim);

}  // namespace ie
