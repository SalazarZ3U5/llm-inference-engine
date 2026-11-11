#pragma once
// The GPU implementation of Runner. Plain C++ header: no CUDA types, so the
// rest of the program can include it without being compiled by nvcc.

#include <memory>
#include <string>

#include "runner.h"

namespace ie::cuda {

// True if a CUDA device and driver are present.
bool device_available();
std::string device_description();

// Copies the model to the GPU. `w` must hold BF16 or INT8 weights; it can be
// freed afterwards. max_tokens sets the KV cache size in VRAM. use_graphs:
// record each decode step once as a CUDA graph and replay it.
std::unique_ptr<Runner> make_runner(const Config& cfg, const Weights& w, int max_tokens,
                                    bool use_graphs = true);

// Batched serving with a paged KV cache: up to max_batch sequences of up to
// max_seq_len tokens, sharing a page pool of about kv_bytes bytes.
std::unique_ptr<BatchRunner> make_batch_runner(const Config& cfg, const Weights& w, int max_batch,
                                               int max_seq_len, size_t kv_bytes,
                                               bool use_graphs = true);

// Multi-token matmuls (prefill) use tensor cores by default; switch off to compare.
void set_tensor_cores(bool on);

// Compares every CUDA kernel with the CPU reference on random data. Returns 0 if all pass.
int run_selftest();

}  // namespace ie::cuda
