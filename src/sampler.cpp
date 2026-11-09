#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>

#include <nlohmann/json.hpp>

namespace ie {

SamplingParams load_generation_config(const std::string& model_dir) {
  SamplingParams p;
  std::ifstream f(std::filesystem::path(model_dir) / "generation_config.json");
  if (!f) return p;
  const nlohmann::json j = nlohmann::json::parse(f);
  p.temperature = j.value("temperature", p.temperature);
  p.top_k = j.value("top_k", p.top_k);
  p.top_p = j.value("top_p", p.top_p);
  p.repetition_penalty = j.value("repetition_penalty", p.repetition_penalty);
  if (!j.value("do_sample", true)) p.temperature = 0.0f;
  return p;
}

int32_t argmax(const float* x, int n) {
  int32_t best = 0;
  for (int i = 1; i < n; ++i)
    if (x[i] > x[best]) best = i;
  return best;
}

Sampler::Sampler(const SamplingParams& p)
    : p_(p), seed_(p.seed ? p.seed : std::random_device{}()), rng_(seed_) {}

int32_t Sampler::sample(const float* logits, int vocab_size, const std::vector<int32_t>& history) {
  std::vector<float> l(logits, logits + vocab_size);

  // 1. Repetition penalty (as in the CTRL paper / HF): push already-seen tokens
  //    toward "less likely" whichever sign their logit has.
  if (p_.repetition_penalty != 1.0f) {
    std::vector<bool> seen(vocab_size, false);
    for (int32_t id : history) {
      if (id < 0 || id >= vocab_size || seen[id]) continue;
      seen[id] = true;
      l[id] = l[id] > 0 ? l[id] / p_.repetition_penalty : l[id] * p_.repetition_penalty;
    }
  }

  if (p_.temperature <= 0.0f) return argmax(l.data(), vocab_size);

  // 2. Temperature
  for (float& x : l) x /= p_.temperature;

  // 3. Top-k: sort only the k best candidates (partial sort is much cheaper
  //    than sorting all 151,936).
  std::vector<int32_t> idx(vocab_size);
  std::iota(idx.begin(), idx.end(), 0);
  const int k = (p_.top_k > 0 && p_.top_k < vocab_size) ? p_.top_k : vocab_size;
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&](int32_t a, int32_t b) { return l[a] > l[b]; });
  idx.resize(k);

  // Softmax over the survivors (subtract the max for numerical safety).
  std::vector<double> prob(k);
  const double mx = l[idx[0]];
  double sum = 0;
  for (int i = 0; i < k; ++i) sum += prob[i] = std::exp(double(l[idx[i]]) - mx);
  for (double& pr : prob) pr /= sum;

  // 4. Top-p: keep tokens until the running total reaches p (always keep one).
  int keep = k;
  if (p_.top_p < 1.0f) {
    double cum = 0;
    for (int i = 0; i < k; ++i) {
      cum += prob[i];
      if (cum >= p_.top_p) {
        keep = i + 1;
        break;
      }
    }
  }

  // 5. Draw from the remaining distribution.
  double total = 0;
  for (int i = 0; i < keep; ++i) total += prob[i];
  double r = std::uniform_real_distribution<double>(0.0, total)(rng_);
  for (int i = 0; i < keep; ++i) {
    r -= prob[i];
    if (r <= 0) return idx[i];
  }
  return idx[keep - 1];
}

}  // namespace ie
