#include "model.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "ops.h"

namespace ie {

namespace {
const float* ptr_or_null(const std::vector<float>& v) { return v.empty() ? nullptr : v.data(); }
}  // namespace

std::vector<float> Model::forward(const std::vector<int32_t>& ids,
                                  const StageCallback& on_stage) const {
  const int T = static_cast<int>(ids.size());
  const int H = cfg_.hidden_size, I = cfg_.intermediate_size, V = cfg_.vocab_size;
  const int QD = cfg_.q_dim(), KVD = cfg_.kv_dim(), HD = cfg_.head_dim;
  auto emit = [&](const std::string& name, const std::vector<float>& buf, int cols) {
    if (on_stage) on_stage(name, buf.data(), T, cols);
  };

  // Working buffers, one row per token.
  std::vector<float> x(size_t(T) * H);     // residual stream
  std::vector<float> xn(size_t(T) * H);    // normalized copy of x that each block reads
  std::vector<float> q(size_t(T) * QD), k(size_t(T) * KVD), v(size_t(T) * KVD);
  std::vector<float> attn(size_t(T) * QD);  // attention output, all heads side by side
  std::vector<float> proj(size_t(T) * H);   // o_proj / down_proj output, added back to x
  std::vector<float> gate(size_t(T) * I), up(size_t(T) * I);

  // 1. Embedding: copy one row of the table per token.
  for (int t = 0; t < T; ++t) {
    if (ids[t] < 0 || ids[t] >= V) throw std::runtime_error("token id out of range");
    std::memcpy(&x[size_t(t) * H], &w_.embed[size_t(ids[t]) * H], sizeof(float) * H);
  }
  emit("embed", x, H);

  // 2. Transformer layers.
  for (int l = 0; l < cfg_.num_layers; ++l) {
    const LayerWeights& L = w_.layers[l];

    // --- attention block: x = x + o_proj(attention(rmsnorm(x))) ---
    rmsnorm(xn.data(), x.data(), L.input_norm.data(), T, H, cfg_.rms_norm_eps);
    linear(q.data(), xn.data(), L.wq.data(), ptr_or_null(L.bq), T, H, QD);
    linear(k.data(), xn.data(), L.wk.data(), ptr_or_null(L.bk), T, H, KVD);
    linear(v.data(), xn.data(), L.wv.data(), ptr_or_null(L.bv), T, H, KVD);
    rope(q.data(), T, cfg_.num_heads, HD, /*pos0=*/0, cfg_.rope_theta);
    rope(k.data(), T, cfg_.num_kv_heads, HD, /*pos0=*/0, cfg_.rope_theta);
    attention(attn.data(), q.data(), k.data(), v.data(), T, cfg_.num_heads, cfg_.num_kv_heads, HD);
    linear(proj.data(), attn.data(), L.wo.data(), nullptr, T, QD, H);
    for (size_t i = 0; i < x.size(); ++i) x[i] += proj[i];

    // --- MLP block: x = x + down(silu(gate(rmsnorm(x))) * up(rmsnorm(x))) ---
    rmsnorm(xn.data(), x.data(), L.post_attn_norm.data(), T, H, cfg_.rms_norm_eps);
    linear(gate.data(), xn.data(), L.w_gate.data(), nullptr, T, H, I);
    linear(up.data(), xn.data(), L.w_up.data(), nullptr, T, H, I);
    for (size_t i = 0; i < gate.size(); ++i) gate[i] = silu(gate[i]) * up[i];
    linear(proj.data(), gate.data(), L.w_down.data(), nullptr, T, I, H);
    for (size_t i = 0; i < x.size(); ++i) x[i] += proj[i];

    char name[32];
    std::snprintf(name, sizeof(name), "layer_%02d", l);
    emit(name, x, H);
  }

  // 3. Final norm, then compare against every vocabulary entry.
  rmsnorm(xn.data(), x.data(), w_.final_norm.data(), T, H, cfg_.rms_norm_eps);
  emit("final_norm", xn, H);

  std::vector<float> logits(size_t(T) * V);
  linear(logits.data(), xn.data(), w_.lm_head_ptr(), nullptr, T, H, V);
  emit("logits", logits, V);
  return logits;
}

}  // namespace ie
