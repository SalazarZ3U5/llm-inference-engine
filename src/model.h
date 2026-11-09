#pragma once
// The transformer forward pass: token IDs in, logits out.
//
//   x = embed[ids]
//   for each layer:
//     x = x + o_proj( attention( rope(q), rope(k), v ) )   where q,k,v come from rmsnorm(x)
//     x = x + down( silu(gate(rmsnorm(x))) * up(rmsnorm(x)) )
//   logits = rmsnorm(x) . lm_head^T
//
// Generation uses a KV cache: every token's key and value vectors (per layer)
// are stored once, so each new token only needs its own q/k/v computed.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "config.h"
#include "weights.h"

namespace ie {

// Called after each stage with its output, so a caller can inspect or check it.
// Stage names match the reference files: "embed", "layer_00".."layer_23",
// "final_norm", "logits". `data` holds `rows` x `cols` floats.
using StageCallback =
    std::function<void(const std::string& stage, const float* data, int rows, int cols)>;

// Keys and values of every token processed so far, for every layer.
// k[l] and v[l] are [max_tokens, kv_dim]; rows 0..len-1 are filled.
// Keys are stored *after* RoPE, so they never need rotating again.
struct KVCache {
  int max_tokens = 0;
  int len = 0;  // tokens currently cached = position of the next token
  std::vector<std::vector<float>> k, v;

  void clear() { len = 0; }
  size_t bytes() const;
};

class Model {
 public:
  Model(Config cfg, Weights w) : cfg_(std::move(cfg)), w_(std::move(w)) {}

  KVCache make_cache(int max_tokens) const;

  // Runs `ids` through the model as the next tokens after whatever is in `cache`
  // (positions cache.len .. cache.len + ids.size() - 1), and appends their keys
  // and values to the cache.
  //   all_logits = false: returns logits for the last token only, [vocab_size].
  //   all_logits = true:  returns logits for every new token, [ids.size(), vocab_size].
  std::vector<float> forward(const std::vector<int32_t>& ids, KVCache& cache,
                             bool all_logits = false,
                             const StageCallback& on_stage = nullptr) const;

  const Config& config() const { return cfg_; }
  const Weights& weights() const { return w_; }

 private:
  Config cfg_;
  Weights w_;
};

}  // namespace ie
