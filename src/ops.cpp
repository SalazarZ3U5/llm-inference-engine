#include "ops.h"

#include <cmath>
#include <vector>

#include "threadpool.h"

namespace ie {

void linear(float* y, const float* x, const float* W, const float* b, int T, int in, int out) {
  for (int t = 0; t < T; ++t) {
    const float* xt = x + size_t(t) * in;
    float* yt = y + size_t(t) * out;
    for (int o = 0; o < out; ++o) {
      const float* row = W + size_t(o) * in;  // one contiguous row of W per output
      float acc = b ? b[o] : 0.0f;
      for (int i = 0; i < in; ++i) acc += row[i] * xt[i];
      yt[o] = acc;
    }
  }
}

void rmsnorm(float* y, const float* x, const float* weight, int T, int dim, float eps) {
  for (int t = 0; t < T; ++t) {
    const float* xt = x + size_t(t) * dim;
    float* yt = y + size_t(t) * dim;
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; ++i) sum_sq += xt[i] * xt[i];
    const float inv_rms = 1.0f / std::sqrt(sum_sq / float(dim) + eps);
    for (int i = 0; i < dim; ++i) yt[i] = xt[i] * inv_rms * weight[i];
  }
}

void rope(float* x, int T, int n_heads, int head_dim, int pos0, float theta) {
  const int half = head_dim / 2;
  // Frequency of pair i: theta^(-2i / head_dim). Computed in FP32 exactly as
  // Hugging Face does, so angles match the reference bit for bit.
  std::vector<float> inv_freq(half);
  for (int i = 0; i < half; ++i)
    inv_freq[i] = 1.0f / std::pow(theta, float(2 * i) / float(head_dim));

  for (int t = 0; t < T; ++t) {
    const float pos = float(pos0 + t);
    for (int h = 0; h < n_heads; ++h) {
      float* v = x + (size_t(t) * n_heads + h) * head_dim;  // this head's slice
      for (int i = 0; i < half; ++i) {
        const float angle = pos * inv_freq[i];
        const float c = std::cos(angle), s = std::sin(angle);
        const float x0 = v[i], x1 = v[i + half];
        v[i] = x0 * c - x1 * s;
        v[i + half] = x1 * c + x0 * s;
      }
    }
  }
}

void softmax(float* x, int n) {
  float mx = x[0];
  for (int i = 1; i < n; ++i) mx = std::fmax(mx, x[i]);
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - mx);  // subtracting the max keeps exp() from overflowing
    sum += x[i];
  }
  for (int i = 0; i < n; ++i) x[i] /= sum;
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

void attention(float* out, const float* q, const float* k, const float* v, int T, int pos0,
               int n_heads, int n_kv_heads, int head_dim) {
  const int group = n_heads / n_kv_heads;          // query heads per KV head
  const float scale = 1.0f / std::sqrt(float(head_dim));
  const size_t q_stride = size_t(n_heads) * head_dim;   // floats per token in q / out
  const size_t kv_stride = size_t(n_kv_heads) * head_dim;  // floats per token in k / v

  // Heads are independent, so each thread takes a slice of them.
  thread_pool().parallel_for(n_heads, [&](int64_t h0, int64_t h1) {
    std::vector<float> scores(size_t(pos0) + T);  // per-thread scratch
    for (int h = int(h0); h < int(h1); ++h) {
      const int kvh = h / group;  // GQA: which KV head this query head shares
      for (int t = 0; t < T; ++t) {
        const int p = pos0 + t;  // absolute position of this query token
        const float* qt = q + t * q_stride + size_t(h) * head_dim;

        // 1. scores against every token up to and including this one (s <= p):
        //    the causal mask is simply not computing s > p at all.
        for (int s = 0; s <= p; ++s) {
          const float* ks = k + s * kv_stride + size_t(kvh) * head_dim;
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d) dot += qt[d] * ks[d];
          scores[s] = dot * scale;
        }
        // 2. scores -> weights
        softmax(scores.data(), p + 1);

        // 3. weighted sum of value vectors
        float* ot = out + t * q_stride + size_t(h) * head_dim;
        for (int d = 0; d < head_dim; ++d) ot[d] = 0.0f;
        for (int s = 0; s <= p; ++s) {
          const float* vs = v + s * kv_stride + size_t(kvh) * head_dim;
          const float w = scores[s];
          for (int d = 0; d < head_dim; ++d) ot[d] += w * vs[d];
        }
      }
    }
  });
}

}  // namespace ie
