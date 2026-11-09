#include "config.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace ie {

Config load_config(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  nlohmann::json j = nlohmann::json::parse(f);

  Config c;
  c.hidden_size = j.at("hidden_size").get<int>();
  c.intermediate_size = j.at("intermediate_size").get<int>();
  c.num_layers = j.at("num_hidden_layers").get<int>();
  c.num_heads = j.at("num_attention_heads").get<int>();
  c.num_kv_heads = j.value("num_key_value_heads", c.num_heads);  // absent = no GQA
  c.vocab_size = j.at("vocab_size").get<int>();
  c.max_position_embeddings = j.value("max_position_embeddings", 0);
  c.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
  c.rope_theta = j.value("rope_theta", 10000.0f);
  c.tie_word_embeddings = j.value("tie_word_embeddings", false);
  c.head_dim = j.value("head_dim", c.hidden_size / c.num_heads);

  if (c.hidden_size % c.num_heads != 0 && !j.contains("head_dim"))
    throw std::runtime_error("hidden_size is not divisible by num_attention_heads");
  if (c.num_heads % c.num_kv_heads != 0)
    throw std::runtime_error("num_attention_heads is not divisible by num_key_value_heads");
  if (c.head_dim % 2 != 0) throw std::runtime_error("RoPE needs an even head_dim");
  return c;
}

}  // namespace ie
