// ie: command-line entry point for the inference engine.
//
//   ie gpu-check                      GPU info + vector-add test + bandwidth test
//   ie inspect <model.safetensors>    list every tensor: name, dtype, shape, size

#include <cstdio>
#include <exception>
#include <string>

#include "safetensors.h"

#ifdef IE_WITH_CUDA
#include "cuda/device_check.h"
#endif

namespace {

void usage() {
  std::printf(
      "usage:\n"
      "  ie gpu-check\n"
      "  ie inspect <model.safetensors>\n");
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
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  usage();
  return 1;
}
