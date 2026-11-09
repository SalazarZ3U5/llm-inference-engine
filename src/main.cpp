// ie: command-line entry point for the inference engine.
//
//   ie gpu-check                          GPU info + vector-add test + bandwidth test
//   ie inspect <model.safetensors>        list every tensor: name, dtype, shape, size
//   ie verify  <model_dir> <ref_dir>      run the forward pass and check every stage
//                                         against the reference .npy files

#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "config.h"
#include "model.h"
#include "npy.h"
#include "safetensors.h"
#include "weights.h"

#ifdef IE_WITH_CUDA
#include "cuda/device_check.h"
#endif

namespace fs = std::filesystem;

namespace {

void usage() {
  std::printf(
      "usage:\n"
      "  ie gpu-check\n"
      "  ie inspect <model.safetensors>\n"
      "  ie verify  <model_dir> <reference_dir>\n");
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::string shape_str(const ie::TensorInfo& t) {
  std::string s = "[";
  for (size_t i = 0; i < t.shape.size(); ++i) {
    if (i) s += ", ";
    s += std::to_string(t.shape[i]);
  }
  return s + "]";
}

int inspect(const std::string& path) {
  ie::SafeTensorsFile f(path);
  std::printf("%-60s %-6s %-18s %10s\n", "tensor", "dtype", "shape", "MiB");
  uint64_t total_params = 0, total_bytes = 0;
  for (const auto& [name, t] : f.tensors()) {
    std::printf("%-60s %-6s %-18s %10.2f\n", name.c_str(), t.dtype.c_str(), shape_str(t).c_str(),
                t.num_bytes() / 1048576.0);
    total_params += t.num_elements();
    total_bytes += t.num_bytes();
  }
  std::printf("\n%zu tensors, %.1fM parameters, %.1f MiB of weights\n", f.tensors().size(),
              total_params / 1e6, total_bytes / 1048576.0);
  for (const auto& [k, v] : f.metadata()) std::printf("metadata %s = %s\n", k.c_str(), v.c_str());
  return 0;
}

// Error budget per stage. The embedding is a pure copy of converted weights,
// so it must match exactly. Everything else involves sums whose rounding
// depends on the order of additions, which differs from PyTorch's.
//
// Why 1e-3 and not tighter: token 0 carries a value of ~1,574 through layers
// 2-20. FP32 rounding on numbers that size is ~1e-4 in absolute terms. At
// layer 21 the model cancels that value out, the largest remaining value drops
// to ~50, and the carried-over rounding shows up as ~1e-4 *relative* error.
// That is expected arithmetic, not a bug. Real bugs (missing bias, wrong RoPE
// pairing, wrong GQA mapping) produce errors of 1e-2 to 1 and still fail.
double tolerance(const std::string& stage) { return stage == "embed" ? 0.0 : 1e-3; }

int verify(const fs::path& model_dir, const fs::path& ref_dir) {
  auto t0 = std::chrono::steady_clock::now();
  ie::Config cfg = ie::load_config((model_dir / "config.json").string());
  std::printf("config: %d layers, hidden %d, %d query heads / %d KV heads (head_dim %d), vocab %d\n",
              cfg.num_layers, cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim,
              cfg.vocab_size);

  ie::SafeTensorsFile st((model_dir / "model.safetensors").string());
  ie::Model model(cfg, ie::load_weights(st, cfg));
  std::printf("weights: %.1fM parameters loaded as FP32 in %.1f s\n",
              model.weights().num_params() / 1e6, seconds_since(t0));

  ie::NpyArray ids_npy = ie::load_npy((ref_dir / "input_ids.npy").string());
  std::vector<int32_t> ids;
  for (int64_t id : ids_npy.i64) ids.push_back(static_cast<int32_t>(id));
  std::printf("prompt ids:");
  for (int32_t id : ids) std::printf(" %d", id);
  std::printf("\n\n%-12s %14s %14s %12s  %s\n", "stage", "max_abs_err", "max_abs_ref", "rel_err",
              "result");

  int failures = 0, checked = 0;
  auto check = [&](const std::string& stage, const float* data, int rows, int cols) {
    fs::path ref_path = ref_dir / (stage + ".npy");
    if (!fs::exists(ref_path)) {
      std::printf("%-12s %14s\n", stage.c_str(), "(no reference)");
      return;
    }
    ie::NpyArray ref = ie::load_npy(ref_path.string());
    if (ref.size() != size_t(rows) * cols) {
      std::printf("%-12s size mismatch: got %d x %d, reference has %zu values  FAIL\n",
                  stage.c_str(), rows, cols, ref.size());
      ++failures;
      return;
    }
    ie::CompareResult r = ie::compare(data, ref.f32.data(), ref.size());
    bool ok = r.rel_err <= tolerance(stage);
    ++checked;
    if (!ok) ++failures;
    std::printf("%-12s %14.3e %14.3e %12.3e  %s", stage.c_str(), r.max_abs_err, r.max_abs_ref,
                r.rel_err, ok ? "PASS" : "FAIL");
    if (!ok)
      std::printf("   worst at token %zu, index %zu", r.worst_index / cols, r.worst_index % cols);
    std::printf("\n");
  };

  auto t1 = std::chrono::steady_clock::now();
  std::vector<float> logits = model.forward(ids, check);
  double fwd_s = seconds_since(t1);

  // Which token would the model pick next? (argmax of the last row)
  const float* last = logits.data() + (ids.size() - 1) * size_t(cfg.vocab_size);
  int best = 0;
  for (int i = 1; i < cfg.vocab_size; ++i)
    if (last[i] > last[best]) best = i;
  std::printf("\nforward pass: %.2f s for %zu tokens\n", fwd_s, ids.size());
  std::printf("next token (greedy): %d", best);
  fs::path gen_path = ref_dir / "generated_ids.npy";
  if (fs::exists(gen_path)) {
    int64_t expected = ie::load_npy(gen_path.string()).i64.at(0);
    bool ok = best == expected;
    ++checked;
    if (!ok) ++failures;
    std::printf("   reference: %lld  %s", static_cast<long long>(expected), ok ? "PASS" : "FAIL");
  }
  std::printf("\n\n%s: %d checks, %d failed\n", failures ? "FAILED" : "ALL PASSED", checked,
              failures);
  return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 1;
  }
  const std::string cmd = argv[1];
  try {
    if (cmd == "gpu-check") {
#ifdef IE_WITH_CUDA
      return ie::cuda::run_device_check();
#else
      std::printf("Built without CUDA (IE_ENABLE_CUDA=OFF).\n");
      return 1;
#endif
    }
    if (cmd == "inspect" && argc >= 3) return inspect(argv[2]);
    if (cmd == "verify" && argc >= 4) return verify(argv[2], argv[3]);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  usage();
  return 1;
}
