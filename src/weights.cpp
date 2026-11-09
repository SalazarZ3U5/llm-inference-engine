#include "weights.h"

#include <stdexcept>
#include <string>

namespace ie {

namespace {

std::string shape_str(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) out += (i ? ", " : "") + std::to_string(s[i]);
  return out + "]";
}

// Reads one tensor, checks it has exactly the expected shape, returns FP32.
std::vector<float> load(const SafeTensorsFile& file, const std::string& name,
                        const std::vector<int64_t>& expected_shape) {
  const TensorInfo& t = file.info(name);
  if (t.shape != expected_shape)
    throw std::runtime_error(name + ": expected shape " + shape_str(expected_shape) + ", file has " +
                             shape_str(t.shape));

  std::vector<uint8_t> raw = file.read_bytes(name);
  std::vector<float> out(t.num_elements());
  if (t.dtype == "BF16") {
    const uint8_t* p = raw.data();
    for (size_t i = 0; i < out.size(); ++i)
      out[i] = bf16_to_f32(uint16_t(p[2 * i] | (p[2 * i + 1] << 8)));  // little-endian
  } else if (t.dtype == "F32") {
    std::memcpy(out.data(), raw.data(), raw.size());
  } else {
    throw std::runtime_error(name + ": dtype " + t.dtype + " not supported yet");
  }
  return out;
}

// Same, but returns an empty vector if the tensor doesn't exist (optional biases).
std::vector<float> load_optional(const SafeTensorsFile& file, const std::string& name,
                                 const std::vector<int64_t>& expected_shape) {
  if (!file.contains(name)) return {};
  return load(file, name, expected_shape);
}

}  // namespace

size_t Weights::num_params() const {
  size_t n = embed.size() + final_norm.size() + lm_head.size();
  for (const LayerWeights& l : layers) {
    n += l.input_norm.size() + l.wq.size() + l.bq.size() + l.wk.size() + l.bk.size() +
         l.wv.size() + l.bv.size() + l.wo.size() + l.post_attn_norm.size() + l.w_gate.size() +
         l.w_up.size() + l.w_down.size();
  }
  return n;
}

Weights load_weights(const SafeTensorsFile& file, const Config& cfg) {
  const int64_t H = cfg.hidden_size, I = cfg.intermediate_size, V = cfg.vocab_size;
  const int64_t QD = cfg.q_dim(), KVD = cfg.kv_dim();

  Weights w;
  w.embed = load(file, "model.embed_tokens.weight", {V, H});
  w.final_norm = load(file, "model.norm.weight", {H});
  if (!cfg.tie_word_embeddings) w.lm_head = load(file, "lm_head.weight", {V, H});

  w.layers.resize(cfg.num_layers);
  for (int l = 0; l < cfg.num_layers; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    LayerWeights& L = w.layers[l];
    L.input_norm = load(file, p + "input_layernorm.weight", {H});
    L.wq = load(file, p + "self_attn.q_proj.weight", {QD, H});
    L.bq = load_optional(file, p + "self_attn.q_proj.bias", {QD});
    L.wk = load(file, p + "self_attn.k_proj.weight", {KVD, H});
    L.bk = load_optional(file, p + "self_attn.k_proj.bias", {KVD});
    L.wv = load(file, p + "self_attn.v_proj.weight", {KVD, H});
    L.bv = load_optional(file, p + "self_attn.v_proj.bias", {KVD});
    L.wo = load(file, p + "self_attn.o_proj.weight", {H, QD});
    L.post_attn_norm = load(file, p + "post_attention_layernorm.weight", {H});
    L.w_gate = load(file, p + "mlp.gate_proj.weight", {I, H});
    L.w_up = load(file, p + "mlp.up_proj.weight", {I, H});
    L.w_down = load(file, p + "mlp.down_proj.weight", {H, I});
  }
  return w;
}

}  // namespace ie
