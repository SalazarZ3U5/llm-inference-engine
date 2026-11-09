#pragma once
// Model hyperparameters, read from the model's config.json.
// Nothing in the math hardcodes these numbers, so the same code runs
// Qwen2.5-0.5B, Qwen2.5-1.5B, or a Llama-style model of any size.

#include <string>

namespace ie {

struct Config {
  int hidden_size = 0;           // width of the residual stream (896)
  int intermediate_size = 0;     // MLP width (4864)
  int num_layers = 0;            // transformer layers (24)
  int num_heads = 0;             // query heads (14)
  int num_kv_heads = 0;          // key/value heads (2), < num_heads means GQA
  int head_dim = 0;              // hidden_size / num_heads (64)
  int vocab_size = 0;            // 151936
  int max_position_embeddings = 0;
  float rms_norm_eps = 1e-6f;
  float rope_theta = 10000.0f;
  bool tie_word_embeddings = false;  // true: LM head reuses the embedding table

  int q_dim() const { return num_heads * head_dim; }       // 896
  int kv_dim() const { return num_kv_heads * head_dim; }   // 128
  int kv_group() const { return num_heads / num_kv_heads; }  // query heads per KV head (7)
};

Config load_config(const std::string& config_json_path);

}  // namespace ie
