#include "runner.h"

#include "threadpool.h"

namespace ie {

namespace {

class CpuRunner : public Runner {
 public:
  CpuRunner(Config cfg, Weights w, int max_tokens)
      : model_(std::move(cfg), std::move(w)), cache_(model_.make_cache(max_tokens)) {}

  std::vector<float> forward(const std::vector<int32_t>& ids, bool all_logits,
                             const StageCallback& on_stage) override {
    return model_.forward(ids, cache_, all_logits, on_stage);
  }
  void reset() override { cache_.clear(); }
  int position() const override { return cache_.len; }
  int max_tokens() const override { return cache_.max_tokens; }

  const Config& config() const override { return model_.config(); }
  std::string device_name() const override {
    return "cpu, " + std::to_string(thread_pool().num_threads()) + " threads, " +
           kernel_name(current_kernel()) + " kernels";
  }
  DType weight_dtype() const override { return model_.weights().dtype; }
  size_t weight_bytes() const override { return model_.weights().matrix_bytes(); }
  size_t num_params() const override { return model_.weights().num_params(); }
  size_t kv_cache_bytes() const override { return cache_.bytes(); }

 private:
  Model model_;
  KVCache cache_;
};

}  // namespace

std::unique_ptr<Runner> make_cpu_runner(Config cfg, Weights w, int max_tokens) {
  return std::make_unique<CpuRunner>(std::move(cfg), std::move(w), max_tokens);
}

}  // namespace ie
