#pragma once
// The transformer forward pass: token IDs in, logits out.
//
//   x = embed[ids]
//   for each layer:
//     x = x + o_proj( attention( rope(q), rope(k), v ) )   where q,k,v come from rmsnorm(x)
//     x = x + down( silu(gate(rmsnorm(x))) * up(rmsnorm(x)) )
//   logits = rmsnorm(x) . lm_head^T

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

class Model {
 public:
  Model(Config cfg, Weights w) : cfg_(std::move(cfg)), w_(std::move(w)) {}

  // Runs all tokens through the model in one pass (no KV cache yet).
  // Returns logits for every position: [ids.size(), vocab_size].
  std::vector<float> forward(const std::vector<int32_t>& ids,
                             const StageCallback& on_stage = nullptr) const;

  const Config& config() const { return cfg_; }
  const Weights& weights() const { return w_; }

 private:
  Config cfg_;
  Weights w_;
};

}  // namespace ie
