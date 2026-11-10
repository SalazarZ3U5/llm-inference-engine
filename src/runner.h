#pragma once
// A Runner owns a model's weights and its KV cache on one device (CPU or GPU),
// and runs tokens through it. All the commands (chat, generate, verify, bench,
// perplexity) talk to this interface, so they work unchanged on either device.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "config.h"
#include "matrix.h"
#include "model.h"
#include "weights.h"

namespace ie {

class Runner {
 public:
  virtual ~Runner() = default;

  // Runs `ids` as the next tokens after whatever is in the KV cache and appends
  // them to it. Returns logits for the last token ([vocab]), or for every new
  // token ([ids.size(), vocab]) when all_logits is true. `on_stage` receives
  // each stage's output on the host (slow on the GPU; meant for verification).
  virtual std::vector<float> forward(const std::vector<int32_t>& ids, bool all_logits = false,
                                     const StageCallback& on_stage = nullptr) = 0;

  virtual void reset() = 0;               // empty the KV cache
  virtual int position() const = 0;       // tokens currently cached
  virtual int max_tokens() const = 0;     // KV cache capacity

  virtual const Config& config() const = 0;
  virtual std::string device_name() const = 0;
  virtual DType weight_dtype() const = 0;
  virtual size_t weight_bytes() const = 0;    // matrix bytes streamed per decode step
  virtual size_t num_params() const = 0;
  virtual size_t kv_cache_bytes() const = 0;
};

// The CPU implementation (Model + KVCache from Phases 0-2).
std::unique_ptr<Runner> make_cpu_runner(Config cfg, Weights w, int max_tokens);

}  // namespace ie
