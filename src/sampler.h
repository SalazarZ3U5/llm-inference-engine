#pragma once
// Picks the next token from the model's logits.
//
// Applied in the same order as Hugging Face's generate():
//   1. repetition penalty  make tokens that already appeared less likely
//   2. temperature         divide logits by T (T < 1: sharper, T > 1: flatter)
//   3. top-k               keep only the k highest-scoring tokens
//   4. top-p (nucleus)     of those, keep the smallest set whose probability sums to p
//   5. sample              draw one token at random from what is left
// temperature <= 0 means greedy: always take the highest logit.

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace ie {

struct SamplingParams {
  float temperature = 0.7f;   // defaults from Qwen2.5-Instruct's generation_config.json
  int top_k = 20;             // 0 = no limit
  float top_p = 0.8f;         // 1.0 = no limit
  float repetition_penalty = 1.1f;  // 1.0 = off
  uint64_t seed = 0;          // 0 = pick a random seed
};

class Sampler {
 public:
  explicit Sampler(const SamplingParams& p);

  // `logits` is one row of vocab_size scores; `history` is every token so far
  // (prompt included), used by the repetition penalty.
  int32_t sample(const float* logits, int vocab_size, const std::vector<int32_t>& history);

  uint64_t seed() const { return seed_; }

 private:
  SamplingParams p_;
  uint64_t seed_;
  std::mt19937_64 rng_;
};

int32_t argmax(const float* x, int n);

// The model's recommended settings from generation_config.json (temperature,
// top_k, top_p, repetition_penalty); fields it doesn't set keep the defaults above.
// Note: Hugging Face applies repetition_penalty even when sampling is off (greedy).
SamplingParams load_generation_config(const std::string& model_dir);

}  // namespace ie
