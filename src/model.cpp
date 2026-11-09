#include "model.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "matrix.h"
#include "ops.h"

namespace ie {

namespace {
const float* ptr_or_null(const std::vector<float>& v) { return v.empty() ? nullptr : v.data(); }
}  // namespace

size_t KVCache::bytes() const {
  size_t n = 0;
  for (size_t l = 0; l < k.size(); ++l) n += (k[l].size() + v[l].size()) * sizeof(float);
  return n;
}

KVCache Model::make_cache(int max_tokens) const {
  KVCache c;
  c.max_tokens = max_tokens;
  c.k.assign(cfg_.num_layers, std::vector<float>(size_t(max_tokens) * cfg_.kv_dim()));
  c.v.assign(cfg_.num_layers, std::vector<float>(size_t(max_tokens) * cfg_.kv_dim()));
  return c;
}

std::vector<float> Model::forward(const std::vector<int32_t>& ids, KVCache& cache, bool all_logits,
                                  const StageCallback& on_stage) const {
  const int T = static_cast<int>(ids.size());
  const int pos0 = cache.len;  // position of the first new token
  const int H = cfg_.hidden_size, I = cfg_.intermediate_size, V = cfg_.vocab_size;
  const int QD = cfg_.q_dim(), KVD = cfg_.kv_dim(), HD = cfg_.head_dim;
  if (T == 0) throw std::runtime_error("forward: no tokens");
  if (pos0 + T > cache.max_tokens)
    throw std::runtime_error("forward: KV cache full (" + std::to_string(cache.max_tokens) +
                             " tokens)");
  auto emit = [&](const std::string& name, const std::vector<float>& buf, int rows, int cols) {
    if (on_stage) on_stage(name, buf.data(), rows, cols);
  };

  // Working buffers for the T new tokens, one row per token.
  std::vector<float> x(size_t(T) * H);     // residual stream
  std::vector<float> xn(size_t(T) * H);    // normalized copy of x that each block reads
  std::vector<float> q(size_t(T) * QD), k(size_t(T) * KVD), v(size_t(T) * KVD);
  std::vector<float> attn(size_t(T) * QD);  // attention output, all heads side by side
  std::vector<float> proj(size_t(T) * H);   // o_proj / down_proj output, added back to x
  std::vector<float> gate(size_t(T) * I), up(size_t(T) * I);

  // 1. Embedding: copy (and convert to FP32) one row of the table per token.
  for (int t = 0; t < T; ++t) {
    if (ids[t] < 0 || ids[t] >= V) throw std::runtime_error("token id out of range");
    w_.embed.row_to_f32(ids[t], &x[size_t(t) * H]);
  }
  emit("embed", x, T, H);

  // 2. Transformer layers.
  for (int l = 0; l < cfg_.num_layers; ++l) {
    const LayerWeights& L = w_.layers[l];
    float* k_cache = cache.k[l].data();
    float* v_cache = cache.v[l].data();

    // --- attention block: x = x + o_proj(attention(rmsnorm(x))) ---
    rmsnorm(xn.data(), x.data(), L.input_norm.data(), T, H, cfg_.rms_norm_eps);
    matmul(q.data(), xn.data(), L.wq, ptr_or_null(L.bq), T);
    matmul(k.data(), xn.data(), L.wk, ptr_or_null(L.bk), T);
    matmul(v.data(), xn.data(), L.wv, ptr_or_null(L.bv), T);
    rope(q.data(), T, cfg_.num_heads, HD, pos0, cfg_.rope_theta);
    rope(k.data(), T, cfg_.num_kv_heads, HD, pos0, cfg_.rope_theta);

    // Append this step's keys/values after the cached ones, then attend over all of them.
    std::memcpy(k_cache + size_t(pos0) * KVD, k.data(), sizeof(float) * k.size());
    std::memcpy(v_cache + size_t(pos0) * KVD, v.data(), sizeof(float) * v.size());
    attention(attn.data(), q.data(), k_cache, v_cache, T, pos0, cfg_.num_heads, cfg_.num_kv_heads,
              HD);

    matmul(proj.data(), attn.data(), L.wo, nullptr, T);
    for (size_t i = 0; i < x.size(); ++i) x[i] += proj[i];

    // --- MLP block: x = x + down(silu(gate(rmsnorm(x))) * up(rmsnorm(x))) ---
    rmsnorm(xn.data(), x.data(), L.post_attn_norm.data(), T, H, cfg_.rms_norm_eps);
    matmul(gate.data(), xn.data(), L.w_gate, nullptr, T);
    matmul(up.data(), xn.data(), L.w_up, nullptr, T);
    for (size_t i = 0; i < gate.size(); ++i) gate[i] = silu(gate[i]) * up[i];
    matmul(proj.data(), gate.data(), L.w_down, nullptr, T);
    for (size_t i = 0; i < x.size(); ++i) x[i] += proj[i];

    char name[32];
    std::snprintf(name, sizeof(name), "layer_%02d", l);
    emit(name, x, T, H);
  }
  cache.len += T;

  // 3. Final norm, then compare against every vocabulary entry.
  rmsnorm(xn.data(), x.data(), w_.final_norm.data(), T, H, cfg_.rms_norm_eps);
  emit("final_norm", xn, T, H);

  // During generation only the last token's logits matter, and the LM head is
  // the single biggest matrix (151936 x 896), so skip it for the other rows.
  const int rows = all_logits ? T : 1;
  const float* first_row = xn.data() + size_t(T - rows) * H;
  std::vector<float> logits(size_t(rows) * V);
  matmul(logits.data(), first_row, w_.lm_head_matrix(), nullptr, rows);
  emit("logits", logits, rows, V);
  return logits;
}

}  // namespace ie
