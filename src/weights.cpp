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

const TensorInfo& checked_info(const SafeTensorsFile& file, const std::string& name,
                               const std::vector<int64_t>& expected_shape) {
  const TensorInfo& t = file.info(name);
  if (t.shape != expected_shape)
    throw std::runtime_error(name + ": expected shape " + shape_str(expected_shape) + ", file has " +
                             shape_str(t.shape));
  return t;
}

// Reads one tensor as FP32 (used for small vectors: norms and biases).
std::vector<float> load_vector(const SafeTensorsFile& file, const std::string& name,
                               const std::vector<int64_t>& expected_shape) {
  const TensorInfo& t = checked_info(file, name, expected_shape);
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
  return load_vector(file, name, expected_shape);
}

// Reads a 2-D weight matrix and stores it as `dtype`.
Matrix load_matrix(const SafeTensorsFile& file, const std::string& name, int64_t rows, int64_t cols,
                   DType dtype) {
  const std::vector<int64_t> shape = {rows, cols};
  const TensorInfo& t = checked_info(file, name, shape);
  if (t.dtype == "BF16") {
    std::vector<uint8_t> raw = file.read_bytes(name);
    std::vector<uint16_t> bits(t.num_elements());
    std::memcpy(bits.data(), raw.data(), raw.size());  // safetensors and x86 are both little-endian
    raw.clear();
    raw.shrink_to_fit();
    return make_matrix_bf16(std::move(bits), int(rows), int(cols), dtype);
  }
  return make_matrix(load_vector(file, name, shape), int(rows), int(cols), dtype);
}

size_t matrix_params(const Matrix& m) { return size_t(m.rows) * m.cols; }

}  // namespace

size_t Weights::num_params() const {
  size_t n = matrix_params(embed) + final_norm.size() + matrix_params(lm_head);
  for (const LayerWeights& l : layers) {
    n += l.input_norm.size() + matrix_params(l.wq) + l.bq.size() + matrix_params(l.wk) +
         l.bk.size() + matrix_params(l.wv) + l.bv.size() + matrix_params(l.wo) +
         l.post_attn_norm.size() + matrix_params(l.w_gate) + matrix_params(l.w_up) +
         matrix_params(l.w_down);
  }
  return n;
}

size_t Weights::matrix_bytes() const {
  size_t n = embed.bytes() + lm_head.bytes();
  for (const LayerWeights& l : layers)
    n += l.wq.bytes() + l.wk.bytes() + l.wv.bytes() + l.wo.bytes() + l.w_gate.bytes() +
         l.w_up.bytes() + l.w_down.bytes();
  return n;
}

Weights load_weights(const SafeTensorsFile& file, const Config& cfg, DType dtype) {
  const int64_t H = cfg.hidden_size, I = cfg.intermediate_size, V = cfg.vocab_size;
  const int64_t QD = cfg.q_dim(), KVD = cfg.kv_dim();

  Weights w;
  w.dtype = dtype;
  w.embed = load_matrix(file, "model.embed_tokens.weight", V, H, dtype);
  w.final_norm = load_vector(file, "model.norm.weight", {H});
  if (!cfg.tie_word_embeddings) w.lm_head = load_matrix(file, "lm_head.weight", V, H, dtype);

  w.layers.resize(cfg.num_layers);
  for (int l = 0; l < cfg.num_layers; ++l) {
    const std::string p = "model.layers." + std::to_string(l) + ".";
    LayerWeights& L = w.layers[l];
    L.input_norm = load_vector(file, p + "input_layernorm.weight", {H});
    L.wq = load_matrix(file, p + "self_attn.q_proj.weight", QD, H, dtype);
    L.bq = load_optional(file, p + "self_attn.q_proj.bias", {QD});
    L.wk = load_matrix(file, p + "self_attn.k_proj.weight", KVD, H, dtype);
    L.bk = load_optional(file, p + "self_attn.k_proj.bias", {KVD});
    L.wv = load_matrix(file, p + "self_attn.v_proj.weight", KVD, H, dtype);
    L.bv = load_optional(file, p + "self_attn.v_proj.bias", {KVD});
    L.wo = load_matrix(file, p + "self_attn.o_proj.weight", H, QD, dtype);
    L.post_attn_norm = load_vector(file, p + "post_attention_layernorm.weight", {H});
    L.w_gate = load_matrix(file, p + "mlp.gate_proj.weight", I, H, dtype);
    L.w_up = load_matrix(file, p + "mlp.up_proj.weight", I, H, dtype);
    L.w_down = load_matrix(file, p + "mlp.down_proj.weight", H, I, dtype);
  }
  return w;
}

}  // namespace ie
