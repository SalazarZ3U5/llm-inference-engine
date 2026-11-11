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

// Serves many sequences at once from one copy of the weights (GPU only).
//
// Each sequence's KV cache lives in fixed-size pages taken from a shared pool
// as it grows and returned when it is closed, so memory follows the actual
// lengths instead of max_batch x max_seq_len. A decode step runs one new token
// for each of several sequences in a single forward pass: the weights are read
// once for all of them, which is where batching gets its throughput.
class BatchRunner {
 public:
  virtual ~BatchRunner() = default;

  virtual int open_sequence() = 0;           // returns a sequence id, or -1 if all slots are busy
  virtual void close_sequence(int seq) = 0;  // frees its pages
  virtual int sequence_length(int seq) const = 0;

  // Appends a prompt to one sequence; returns the logits [vocab] of its last token.
  virtual std::vector<float> prefill(int seq, const std::vector<int32_t>& ids) = 0;
  // One new token for each listed sequence; returns logits [n, vocab].
  virtual std::vector<float> decode(const std::vector<int>& seqs,
                                    const std::vector<int32_t>& tokens) = 0;
  // Greedy versions: the argmax is taken on the GPU, so only token ids come back.
  virtual int32_t prefill_greedy(int seq, const std::vector<int32_t>& ids) = 0;
  virtual std::vector<int32_t> decode_greedy(const std::vector<int>& seqs,
                                             const std::vector<int32_t>& tokens) = 0;

  virtual int max_batch() const = 0;     // sequences that can be open at once
  virtual int max_seq_len() const = 0;   // tokens per sequence
  virtual int page_size() const = 0;     // tokens per KV page
  virtual int total_pages() const = 0;
  virtual int free_pages() const = 0;

  virtual const Config& config() const = 0;
  virtual std::string device_name() const = 0;
  virtual size_t weight_bytes() const = 0;
  virtual size_t kv_cache_bytes() const = 0;
};

}  // namespace ie
