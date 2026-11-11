#include "cuda/cuda_runner.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda/kernels.cuh"
#include "npy.h"
#include "ops.h"
#include "sampler.h"

// NVTX: named ranges that show up on the Nsight Systems timeline ("prefill",
// "decode", ...). Header-only, and costs nothing when no profiler is attached.
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define IE_NVTX_PUSH(name) nvtxRangePushA(name)
#define IE_NVTX_POP() nvtxRangePop()
#else
#define IE_NVTX_PUSH(name) ((void)0)
#define IE_NVTX_POP() ((void)0)
#endif

namespace ie::cuda {

namespace {

constexpr int kPageSize = kKvPageSize;  // tokens per KV-cache page (16)
constexpr int kPromptMinRows = 16;      // prompts this long use the prefill attention kernel

struct NvtxRange {
  explicit NvtxRange(const char* name) { IE_NVTX_PUSH(name); }
  ~NvtxRange() { IE_NVTX_POP(); }
};

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(err));
}

// Owns every device allocation of one engine and frees them together.
class DeviceMemory {
 public:
  ~DeviceMemory() {
    for (void* p : ptrs_) cudaFree(p);
  }
  template <class T>
  T* alloc(size_t count) {
    if (count == 0) return nullptr;
    void* p = nullptr;
    check(cudaMalloc(&p, count * sizeof(T)), "cudaMalloc");
    ptrs_.push_back(p);
    return static_cast<T*>(p);
  }
  template <class T>
  T* upload(const std::vector<T>& host) {
    T* d = alloc<T>(host.size());
    if (d) check(cudaMemcpy(d, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
  }
  void release(void* p) {
    auto it = std::find(ptrs_.begin(), ptrs_.end(), p);
    if (it != ptrs_.end()) {
      cudaFree(p);
      ptrs_.erase(it);
    }
  }
  // Uploads a weight matrix in its storage format (BF16, or INT8 + row scales).
  DeviceMatrix upload_matrix(const Matrix& m) {
    DeviceMatrix d;
    d.rows = m.rows;
    d.cols = m.cols;
    if (m.dtype == DType::BF16) {
      d.data = upload(m.bf16);
    } else if (m.dtype == DType::I8) {
      d.data = upload(m.i8);
      d.scale = upload(m.scale);
      d.int8 = true;
    } else {
      throw std::runtime_error("the CUDA backend stores weights as bf16 or int8");
    }
    return d;
  }

 private:
  std::vector<void*> ptrs_;
};

// Page-locked ("pinned") host memory. Copies between it and the GPU can run
// asynchronously on a stream; copies from ordinary memory cannot.
template <class T>
class PinnedBuffer {
 public:
  ~PinnedBuffer() { reset(); }
  void resize(size_t n) {  // grows only; contents are not preserved
    if (n <= n_) return;
    reset();
    void* p = nullptr;
    check(cudaMallocHost(&p, n * sizeof(T)), "cudaMallocHost");
    p_ = static_cast<T*>(p);
    n_ = n;
  }
  T* data() { return p_; }

 private:
  void reset() {
    if (p_) cudaFreeHost(p_);
    p_ = nullptr;
    n_ = 0;
  }
  T* p_ = nullptr;
  size_t n_ = 0;
};

struct DeviceLayer {
  float* input_norm = nullptr;
  DeviceMatrix wq, wk, wv, wo;
  float *bq = nullptr, *bk = nullptr, *bv = nullptr;
  float* post_attn_norm = nullptr;
  DeviceMatrix w_gate, w_up, w_down;
};

// One sequence's view of the paged KV cache.
struct Sequence {
  bool open = false;
  int len = 0;             // tokens cached
  std::vector<int> pages;  // page ids, in order
};

// The GPU engine. Every forward pass is a list of token rows, each with its own
// sequence and position, so the same code serves:
//   - Runner (chat, generate, verify, bench): one sequence, id 0;
//   - BatchRunner (serve-bench): many sequences, prefilled one at a time and
//     then decoded together, one token each per step.
class CudaEngine final : public Runner, public BatchRunner {
 public:
  CudaEngine(const Config& cfg, const Weights& w, int max_batch, int max_seq_len, int num_pages,
             bool use_graphs, bool kv_f16)
      : cfg_(cfg), max_batch_(max_batch), max_seq_len_(max_seq_len), num_pages_(num_pages),
        max_pages_per_seq_((max_seq_len + kPageSize - 1) / kPageSize), use_graphs_(use_graphs),
        kv_f16_(kv_f16), dtype_(w.dtype), seqs_(max_batch) {
    if (w.dtype != DType::BF16 && w.dtype != DType::I8)
      throw std::runtime_error("the CUDA backend runs bf16 or int8 weights (not f32)");
    if (num_pages < 1) throw std::runtime_error("KV cache: no pages");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    name_ = std::string("cuda, ") + prop.name + (use_graphs_ ? ", graphs" : ", no graphs") +
            (kv_f16_ ? ", f16 KV cache" : "");
    check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
    init_matmul_workspace();  // before any graph is recorded

    // Weights: copied once, stay in VRAM in their storage format.
    embed_ = mem_.upload_matrix(w.embed);
    lm_head_ = w.lm_head.rows ? mem_.upload_matrix(w.lm_head) : embed_;  // tied: reuse the embedding
    final_norm_ = mem_.upload(w.final_norm);
    for (const LayerWeights& L : w.layers) {
      DeviceLayer d;
      d.input_norm = mem_.upload(L.input_norm);
      d.wq = mem_.upload_matrix(L.wq);
      d.wk = mem_.upload_matrix(L.wk);
      d.wv = mem_.upload_matrix(L.wv);
      d.wo = mem_.upload_matrix(L.wo);
      d.bq = mem_.upload(L.bq);
      d.bk = mem_.upload(L.bk);
      d.bv = mem_.upload(L.bv);
      d.post_attn_norm = mem_.upload(L.post_attn_norm);
      d.w_gate = mem_.upload_matrix(L.w_gate);
      d.w_up = mem_.upload_matrix(L.w_up);
      d.w_down = mem_.upload_matrix(L.w_down);
      layers_.push_back(d);
    }
    weight_bytes_ = w.matrix_bytes();
    num_params_ = w.num_params();

    // RoPE frequencies, computed exactly like the CPU version.
    const int half = cfg_.head_dim / 2;
    std::vector<float> inv_freq(half);
    for (int i = 0; i < half; ++i)
      inv_freq[i] = 1.0f / std::pow(cfg_.rope_theta, float(2 * i) / float(cfg_.head_dim));
    inv_freq_ = mem_.upload(inv_freq);

    // Paged KV cache: per layer, a pool of num_pages pages of kPageSize rows,
    // as FP32 or FP16 values (2 or 4 bytes each).
    const size_t layer_bytes = size_t(num_pages_) * kPageSize * cfg_.kv_dim() * kv_elem_bytes();
    k_pool_ = mem_.alloc<unsigned char>(layer_bytes * cfg_.num_layers);
    v_pool_ = mem_.alloc<unsigned char>(layer_bytes * cfg_.num_layers);
    kv_bytes_ = 2 * layer_bytes * cfg_.num_layers;
    for (int p = num_pages_ - 1; p >= 0; --p) free_pages_.push_back(p);

    // Page tables, one row per sequence slot, mirrored in pinned host memory.
    const size_t table_ints = size_t(max_batch_) * max_pages_per_seq_;
    tables_ = mem_.alloc<int>(table_ints);
    h_tables_.resize(table_ints);
    std::fill(h_tables_.data(), h_tables_.data() + table_ints, 0);
    tables_dirty_ = true;
  }

  ~CudaEngine() override {
    drop_graphs();
    if (stream_) cudaStreamDestroy(stream_);
  }

  // ---------------------------------------------------------- Runner ----
  // (one sequence: id 0, opened on first use)

  std::vector<float> forward(const std::vector<int32_t>& ids, bool all_logits,
                             const StageCallback& on_stage) override {
    if (!seqs_[0].open) open_slot(0);
    const int T = int(ids.size()), start = seqs_[0].len;
    std::vector<int> rows_seq(T, 0), rows_pos(T), logit_rows;
    for (int i = 0; i < T; ++i) rows_pos[i] = start + i;
    if (all_logits) {
      logit_rows.resize(T);
      std::iota(logit_rows.begin(), logit_rows.end(), 0);
    } else {
      logit_rows = {T - 1};
    }
    const bool graph = T == 1 && !all_logits && !on_stage;
    Output out = run(ids, rows_seq, rows_pos, logit_rows, graph, false, on_stage ? &on_stage : nullptr);
    seqs_[0].len += T;
    return std::move(out.logits);
  }
  void reset() override {
    if (seqs_[0].open) close_sequence(0);
  }
  int position() const override { return seqs_[0].len; }
  int max_tokens() const override { return max_seq_len_; }

  // ----------------------------------------------------- BatchRunner ----

  int open_sequence() override {
    for (int s = 0; s < max_batch_; ++s)
      if (!seqs_[s].open) {
        open_slot(s);
        return s;
      }
    return -1;
  }
  void close_sequence(int seq) override {
    Sequence& sq = slot(seq);
    for (int p : sq.pages) free_pages_.push_back(p);
    sq = Sequence{};
  }
  int sequence_length(int seq) const override { return seqs_.at(seq).len; }

  std::vector<float> prefill(int seq, const std::vector<int32_t>& ids) override {
    return std::move(run_prefill(seq, ids, false).logits);
  }
  int32_t prefill_greedy(int seq, const std::vector<int32_t>& ids) override {
    return run_prefill(seq, ids, true).tokens.at(0);
  }
  std::vector<float> decode(const std::vector<int>& seqs, const std::vector<int32_t>& tokens) override {
    return std::move(run_decode(seqs, tokens, false).logits);
  }
  std::vector<int32_t> decode_greedy(const std::vector<int>& seqs,
                                     const std::vector<int32_t>& tokens) override {
    return std::move(run_decode(seqs, tokens, true).tokens);
  }

  int max_batch() const override { return max_batch_; }
  int max_seq_len() const override { return max_seq_len_; }
  int page_size() const override { return kPageSize; }
  int total_pages() const override { return num_pages_; }
  int free_pages() const override { return int(free_pages_.size()); }

  // ------------------------------------------------------------ both ----

  const Config& config() const override { return cfg_; }
  std::string device_name() const override { return name_; }
  DType weight_dtype() const override { return dtype_; }
  size_t weight_bytes() const override { return weight_bytes_; }
  size_t num_params() const override { return num_params_; }
  size_t kv_cache_bytes() const override { return kv_bytes_; }

 private:
  struct Output {
    std::vector<float> logits;   // [logit rows, vocab]   (unless greedy)
    std::vector<int32_t> tokens;  // [logit rows]          (greedy)
  };

  Sequence& slot(int seq) {
    if (seq < 0 || seq >= max_batch_ || !seqs_[seq].open)
      throw std::runtime_error("invalid or closed sequence id " + std::to_string(seq));
    return seqs_[seq];
  }
  void open_slot(int s) {
    seqs_[s] = Sequence{};
    seqs_[s].open = true;
  }

  // Makes sure `seq` has pages for positions 0..len-1, taking pages from the pool.
  void ensure_pages(int seq, int len) {
    Sequence& sq = slot(seq);
    if (len > max_seq_len_)
      throw std::runtime_error("sequence longer than the maximum (" + std::to_string(max_seq_len_) + " tokens)");
    while (int(sq.pages.size()) * kPageSize < len) {
      if (free_pages_.empty()) throw std::runtime_error("KV cache full: no free pages");
      const int page = free_pages_.back();
      free_pages_.pop_back();
      h_tables_.data()[size_t(seq) * max_pages_per_seq_ + sq.pages.size()] = page;
      sq.pages.push_back(page);
      tables_dirty_ = true;
    }
  }

  Output run_prefill(int seq, const std::vector<int32_t>& ids, bool greedy) {
    Sequence& sq = slot(seq);
    const int T = int(ids.size());
    std::vector<int> rows_seq(T, seq), rows_pos(T);
    for (int i = 0; i < T; ++i) rows_pos[i] = sq.len + i;
    Output out = run(ids, rows_seq, rows_pos, {T - 1}, /*graph=*/T == 1, greedy, nullptr);
    sq.len += T;
    return out;
  }

  Output run_decode(const std::vector<int>& seqs, const std::vector<int32_t>& tokens, bool greedy) {
    const int n = int(seqs.size());
    if (n != int(tokens.size())) throw std::runtime_error("decode: one token per sequence");
    std::vector<int> rows_pos(n), logit_rows(n);
    for (int i = 0; i < n; ++i) {
      rows_pos[i] = slot(seqs[i]).len;
      logit_rows[i] = i;
    }
    Output out = run(tokens, seqs, rows_pos, logit_rows, /*graph=*/true, greedy, nullptr);
    for (int s : seqs) seqs_[s].len += 1;
    return out;
  }

  // The core: one forward pass over `ids`, row t belonging to sequence
  // rows_seq[t] at position rows_pos[t]. Logits (or their argmax) come back for
  // the rows listed in logit_rows. `graph`: this is a pure decode step (one
  // token per sequence, logits for every row), which can replay a recording.
  Output run(const std::vector<int32_t>& ids, const std::vector<int>& rows_seq,
             const std::vector<int>& rows_pos, const std::vector<int>& logit_rows, bool graph,
             bool greedy, const StageCallback* on_stage) {
    const int T = int(ids.size()), R = int(logit_rows.size()), V = cfg_.vocab_size;
    if (T == 0) throw std::runtime_error("forward: no tokens");
    for (int32_t id : ids)
      if (id < 0 || id >= V) throw std::runtime_error("token id out of range");
    NvtxRange range(T == R ? (T == 1 ? "decode" : "decode batch") : "prefill");

    int max_pos = 0;
    for (int t = 0; t < T; ++t) {
      ensure_pages(rows_seq[t], rows_pos[t] + 1);
      max_pos = std::max(max_pos, rows_pos[t]);
    }
    graph = graph && use_graphs_ && !on_stage;
    // A prompt (16+ consecutive positions of one sequence) uses the prefill
    // attention kernel, signalled to enqueue() by chunks = 0. The choice depends
    // only on the request itself, never on what it is batched with.
    bool prompt = !graph && T >= kPromptMinRows &&
                  attention_prefill_supported(cfg_.num_heads, cfg_.num_kv_heads, cfg_.head_dim);
    for (int t = 1; prompt && t < T; ++t)
      prompt = rows_seq[t] == rows_seq[0] && rows_pos[t] == rows_pos[0] + t;
    // Otherwise, attention chunks: a recorded graph must cover any position a
    // sequence can reach; a direct launch only needs the context it actually has.
    const int chunks = prompt ? 0 : attention_chunks(graph ? max_seq_len_ : max_pos + 1);
    reserve(T, R, chunks);  // before any capture: no allocations allowed inside a graph
    if (graph && !graphs_.count(T)) capture_graph(T, chunks);

    // Inputs, read by the kernels from VRAM.
    std::memcpy(h_ids_.data(), ids.data(), sizeof(int32_t) * T);
    std::memcpy(h_row_seq_.data(), rows_seq.data(), sizeof(int) * T);
    std::memcpy(h_row_pos_.data(), rows_pos.data(), sizeof(int) * T);
    std::memcpy(h_logit_rows_.data(), logit_rows.data(), sizeof(int) * R);
    check(cudaMemcpyAsync(ids_, h_ids_.data(), sizeof(int32_t) * T, cudaMemcpyHostToDevice, stream_), "ids");
    check(cudaMemcpyAsync(row_seq_, h_row_seq_.data(), sizeof(int) * T, cudaMemcpyHostToDevice, stream_), "row_seq");
    check(cudaMemcpyAsync(row_pos_, h_row_pos_.data(), sizeof(int) * T, cudaMemcpyHostToDevice, stream_), "row_pos");
    check(cudaMemcpyAsync(logit_rows_, h_logit_rows_.data(), sizeof(int) * R, cudaMemcpyHostToDevice, stream_), "logit_rows");
    if (tables_dirty_) {  // only after a sequence gained or lost a page
      const size_t n = size_t(max_batch_) * max_pages_per_seq_;
      check(cudaMemcpyAsync(tables_, h_tables_.data(), n * sizeof(int), cudaMemcpyHostToDevice, stream_), "tables");
      tables_dirty_ = false;
    }

    if (graph)
      check(cudaGraphLaunch(graphs_.at(T), stream_), "cudaGraphLaunch");  // the whole step, one call
    else
      enqueue(T, R, chunks, on_stage);

    Output out;
    if (greedy) {
      argmax_rows(argmax_, logits_, R, V, stream_);
      check(cudaMemcpyAsync(h_argmax_.data(), argmax_, sizeof(int32_t) * R, cudaMemcpyDeviceToHost, stream_), "argmax");
    } else {
      check(cudaMemcpyAsync(h_logits_.data(), logits_, size_t(R) * V * sizeof(float), cudaMemcpyDeviceToHost, stream_), "logits");
    }
    // Waits for everything queued above; errors from inside kernels surface here.
    check(cudaStreamSynchronize(stream_), "forward");
    if (greedy) {
      out.tokens.assign(h_argmax_.data(), h_argmax_.data() + R);
    } else {
      out.logits.assign(h_logits_.data(), h_logits_.data() + size_t(R) * V);
      if (on_stage) (*on_stage)("logits", out.logits.data(), R, V);
    }
    return out;
  }

  // Queues the whole forward pass for T rows on stream_. Used directly for
  // prefill, and recorded once per batch size into a decode graph.
  void enqueue(int T, int R, int chunks, const StageCallback* on_stage) {
    const int H = cfg_.hidden_size, NH = cfg_.num_heads, NKV = cfg_.num_kv_heads;
    const int HD = cfg_.head_dim;
    auto stage = [&](const char* name, const float* d, int cols) {
      if (!on_stage) return;
      std::vector<float> h(size_t(T) * cols);
      check(cudaMemcpyAsync(h.data(), d, h.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_), "stage copy");
      check(cudaStreamSynchronize(stream_), "stage copy");
      (*on_stage)(name, h.data(), T, cols);
    };
    KvLayout kv;
    kv.tables = tables_;
    kv.max_pages = max_pages_per_seq_;
    kv.page_size = kPageSize;
    kv.f16 = kv_f16_;
    const size_t layer_bytes = size_t(num_pages_) * kPageSize * cfg_.kv_dim() * kv_elem_bytes();

    embedding(x_, ids_, embed_, T, stream_);
    stage("embed", x_, H);

    for (int l = 0; l < cfg_.num_layers; ++l) {
      const DeviceLayer& L = layers_[l];
      void* kp = k_pool_ + l * layer_bytes;
      void* vp = v_pool_ + l * layer_bytes;

      // attention block: x += o_proj(attention(rope(q), rope(k), v))
      rmsnorm(xn_, x_, L.input_norm, T, H, cfg_.rms_norm_eps, stream_);
      matmul(q_, xn_, L.wq, L.bq, T, false, stream_);
      matmul(k_, xn_, L.wk, L.bk, T, false, stream_);
      matmul(v_, xn_, L.wv, L.bv, T, false, stream_);
      rope_and_cache(q_, k_, v_, kp, vp, inv_freq_, T, row_pos_, row_seq_, kv, NH, NKV, HD, stream_);
      if (chunks == 0)  // a prompt: queries share K/V tiles
        attention_prefill(attn_, q_, kp, vp, T, row_pos_, row_seq_, kv, NH, NKV, HD, stream_);
      else
        attention(attn_, q_, kp, vp, T, row_pos_, row_seq_, kv, chunks, part_o_, part_ml_, NH, NKV,
                  HD, stream_);
      matmul(x_, attn_, L.wo, nullptr, T, /*accumulate=*/true, stream_);

      // MLP block: x += down(silu(gate(xn)) * up(xn))
      rmsnorm(xn_, x_, L.post_attn_norm, T, H, cfg_.rms_norm_eps, stream_);
      gate_up_silu(mid_, xn_, L.w_gate, L.w_up, T, stream_);
      matmul(x_, mid_, L.w_down, nullptr, T, /*accumulate=*/true, stream_);

      if (on_stage) {
        char name[32];
        std::snprintf(name, sizeof(name), "layer_%02d", l);
        stage(name, x_, H);
      }
    }

    rmsnorm(xn_, x_, final_norm_, T, H, cfg_.rms_norm_eps, stream_);
    stage("final_norm", xn_, H);
    // Only the rows that need logits go through the 151,936-row LM head.
    gather_rows(xg_, xn_, logit_rows_, R, H, stream_);
    matmul(logits_, xg_, lm_head_, nullptr, R, false, stream_);
  }

  // Records one decode step for a batch of T sequences into a CUDA graph.
  // Replaying it costs a single launch instead of ~250, which matters most on
  // Windows, where each individual launch has noticeable driver overhead.
  void capture_graph(int T, int chunks) {
    NvtxRange range("capture decode graph");
    cudaGraph_t graph = nullptr;
    check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
    try {
      enqueue(T, T, chunks, nullptr);
    } catch (...) {
      cudaStreamEndCapture(stream_, &graph);
      if (graph) cudaGraphDestroy(graph);
      throw;
    }
    check(cudaStreamEndCapture(stream_, &graph), "cudaStreamEndCapture");
    cudaGraphExec_t exec = nullptr;
    cudaError_t err = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    check(err, "cudaGraphInstantiate");
    graphs_[T] = exec;
  }

  void drop_graphs() {
    for (auto& g : graphs_) cudaGraphExecDestroy(g.second);
    graphs_.clear();
  }

  // Scratch buffers sized for T rows, R logit rows and the attention chunks,
  // grown on demand. Recorded graphs hold the old buffer addresses, so they are
  // dropped after any reallocation and re-recorded when next needed.
  void reserve(int T, int R, int chunks) {
    if (T > cap_rows_) {
      drop_graphs();
      for (void* p : {(void*)ids_, (void*)row_seq_, (void*)row_pos_, (void*)x_, (void*)xn_, (void*)q_,
                      (void*)k_, (void*)v_, (void*)attn_, (void*)mid_})
        mem_.release(p);
      cap_rows_ = std::max(T, 16);
      const size_t n = size_t(cap_rows_);
      ids_ = mem_.alloc<int32_t>(n);
      row_seq_ = mem_.alloc<int>(n);
      row_pos_ = mem_.alloc<int>(n);
      x_ = mem_.alloc<float>(n * cfg_.hidden_size);
      xn_ = mem_.alloc<float>(n * cfg_.hidden_size);
      q_ = mem_.alloc<float>(n * cfg_.q_dim());
      k_ = mem_.alloc<float>(n * cfg_.kv_dim());
      v_ = mem_.alloc<float>(n * cfg_.kv_dim());
      attn_ = mem_.alloc<float>(n * cfg_.q_dim());
      mid_ = mem_.alloc<float>(n * cfg_.intermediate_size);
      h_ids_.resize(n);
      h_row_seq_.resize(n);
      h_row_pos_.resize(n);
    }
    if (R > cap_logit_rows_) {
      drop_graphs();
      for (void* p : {(void*)logit_rows_, (void*)xg_, (void*)logits_, (void*)argmax_}) mem_.release(p);
      cap_logit_rows_ = std::max(R, max_batch_);
      const size_t r = size_t(cap_logit_rows_);
      logit_rows_ = mem_.alloc<int>(r);
      xg_ = mem_.alloc<float>(r * cfg_.hidden_size);
      logits_ = mem_.alloc<float>(r * cfg_.vocab_size);
      argmax_ = mem_.alloc<int32_t>(r);
      h_logit_rows_.resize(r);
      h_logits_.resize(r * cfg_.vocab_size);
      h_argmax_.resize(r);
    }
    // Per-chunk partial attention results: one slot per (row, head, chunk).
    const size_t slots = size_t(T) * cfg_.num_heads * chunks;
    if (slots > cap_attn_slots_) {
      drop_graphs();
      mem_.release(part_o_);
      mem_.release(part_ml_);
      cap_attn_slots_ = slots;
      part_o_ = mem_.alloc<float>(slots * cfg_.head_dim);
      part_ml_ = mem_.alloc<float>(slots * 2);
    }
  }

  Config cfg_;
  int max_batch_ = 1, max_seq_len_ = 0, num_pages_ = 0, max_pages_per_seq_ = 0;
  bool use_graphs_ = true;
  bool kv_f16_ = false;
  DType dtype_;
  std::string name_;
  size_t weight_bytes_ = 0, num_params_ = 0, kv_bytes_ = 0;
  DeviceMemory mem_;
  cudaStream_t stream_ = nullptr;
  std::map<int, cudaGraphExec_t> graphs_;  // decode graphs by batch size

  DeviceMatrix embed_, lm_head_;
  float* final_norm_ = nullptr;
  std::vector<DeviceLayer> layers_;
  float* inv_freq_ = nullptr;

  // Paged KV cache
  unsigned char *k_pool_ = nullptr, *v_pool_ = nullptr;  // [layer][page][row][kv_dim], f32 or f16
  size_t kv_elem_bytes() const { return kv_f16_ ? 2 : 4; }
  int* tables_ = nullptr;                        // [sequence][page index] -> page id
  PinnedBuffer<int> h_tables_;
  bool tables_dirty_ = true;
  std::vector<int> free_pages_;
  std::vector<Sequence> seqs_;

  // Scratch
  int cap_rows_ = 0, cap_logit_rows_ = 0;
  size_t cap_attn_slots_ = 0;
  int32_t* ids_ = nullptr;
  int *row_seq_ = nullptr, *row_pos_ = nullptr, *logit_rows_ = nullptr;
  float *x_ = nullptr, *xn_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr;
  float *attn_ = nullptr, *mid_ = nullptr, *xg_ = nullptr, *logits_ = nullptr;
  int32_t* argmax_ = nullptr;
  float *part_o_ = nullptr, *part_ml_ = nullptr;  // flash-decoding partial results
  PinnedBuffer<int32_t> h_ids_, h_argmax_;
  PinnedBuffer<int> h_row_seq_, h_row_pos_, h_logit_rows_;
  PinnedBuffer<float> h_logits_;
};

int pages_for(int tokens) { return (tokens + kPageSize - 1) / kPageSize; }

}  // namespace

bool device_available() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) {
    cudaGetLastError();  // clear the error so later calls aren't affected
    return false;
  }
  return n > 0;
}

std::string device_description() {
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return "no CUDA device";
  return prop.name;
}

void set_tensor_cores(bool on) { ie::cuda::set_tensor_cores_impl(on); }
void set_tc_fp16(bool on) { ie::cuda::set_tc_fp16_impl(on); }

std::unique_ptr<Runner> make_runner(const Config& cfg, const Weights& w, int max_tokens,
                                    bool use_graphs, bool kv_f16) {
  return std::make_unique<CudaEngine>(cfg, w, /*max_batch=*/1, max_tokens, pages_for(max_tokens),
                                      use_graphs, kv_f16);
}

std::unique_ptr<BatchRunner> make_batch_runner(const Config& cfg, const Weights& w, int max_batch,
                                               int max_seq_len, size_t kv_bytes, bool use_graphs,
                                               bool kv_f16) {
  const size_t page_bytes =
      size_t(2) * kPageSize * cfg.kv_dim() * (kv_f16 ? 2 : 4) * cfg.num_layers;
  // Enough pages for the byte budget, but never more than every sequence at full length.
  const size_t want = std::max<size_t>(1, kv_bytes / page_bytes);
  const size_t most = size_t(max_batch) * pages_for(max_seq_len);
  return std::make_unique<CudaEngine>(cfg, w, max_batch, max_seq_len, int(std::min(want, most)),
                                      use_graphs, kv_f16);
}

// ---------------------------------------------------------- self-test ----

namespace {

std::vector<float> random_vec(std::mt19937& rng, size_t n, float scale) {
  std::uniform_real_distribution<float> uni(-scale, scale);
  std::vector<float> v(n);
  for (float& x : v) x = uni(rng);
  return v;
}

template <class T>
std::vector<T> download(const T* d, size_t n) {
  std::vector<T> h(n);
  check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
  return h;
}

// The FP32 values a quantized matrix actually holds (what the GPU should compute with).
std::vector<float> dequantized(const Matrix& m) {
  std::vector<float> out(size_t(m.rows) * m.cols);
  for (int r = 0; r < m.rows; ++r) m.row_to_f32(r, &out[size_t(r) * m.cols]);
  return out;
}

struct Tally {
  int tests = 0, failures = 0;
  double worst = 0;
  void add(const char* what, const std::vector<float>& got, const std::vector<float>& want,
           double tol = 2e-5) {
    CompareResult r = compare(got.data(), want.data(), want.size());
    ++tests;
    worst = std::max(worst, r.rel_err);
    if (!(r.rel_err <= tol)) {
      ++failures;
      std::printf("FAIL %-44s rel_err %.3e (worst at index %zu)\n", what, r.rel_err, r.worst_index);
    }
  }
  void add_exact(const char* what, bool ok) {
    ++tests;
    if (!ok) {
      ++failures;
      std::printf("FAIL %s\n", what);
    }
  }
};

// FP16 copies for testing the FP16 KV cache: the values rounded to FP16, as
// floats (for the CPU reference) and as __half (for the GPU).
std::vector<float> round_f16(const std::vector<float>& v) {
  std::vector<float> r(v.size());
  for (size_t i = 0; i < v.size(); ++i) r[i] = __half2float(__float2half_rn(v[i]));
  return r;
}
std::vector<__half> to_f16(const std::vector<float>& v) {
  std::vector<__half> r(v.size());
  for (size_t i = 0; i < v.size(); ++i) r[i] = __float2half_rn(v[i]);
  return r;
}
std::vector<float> from_f16(const std::vector<__half>& v) {
  std::vector<float> r(v.size());
  for (size_t i = 0; i < v.size(); ++i) r[i] = __half2float(v[i]);
  return r;
}

// A paged cache for tests: `lens[s]` positions per sequence, pages handed out in
// a shuffled order so that no sequence is contiguous.
struct TestPages {
  int page_size = 16, max_pages = 0, num_pages = 0;
  std::vector<int> tables;  // [seq * max_pages + i]
  TestPages(const std::vector<int>& lens, std::mt19937& rng) {
    for (int len : lens) {
      max_pages = std::max(max_pages, (len + page_size - 1) / page_size);
      num_pages += (len + page_size - 1) / page_size;
    }
    std::vector<int> order(num_pages);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    tables.assign(lens.size() * size_t(max_pages), 0);
    int next = 0;
    for (size_t s = 0; s < lens.size(); ++s)
      for (int i = 0; i < (lens[s] + page_size - 1) / page_size; ++i)
        tables[s * max_pages + i] = order[next++];
  }
  size_t row(int seq, int pos, int kv_dim) const {
    return (size_t(tables[size_t(seq) * max_pages + pos / page_size]) * page_size + pos % page_size) * kv_dim;
  }
};

}  // namespace

int run_selftest() {
  std::printf("GPU: %s\n", device_description().c_str());
  std::mt19937 rng(42);
  Tally tally;
  DeviceMemory mem;
  char what[160];
  const cudaStream_t s = nullptr;  // the default stream is fine for tests
  init_matmul_workspace();

  // matmul and gate_up_silu, for both weight formats, against ie::linear on the
  // exact values the GPU holds. Column counts 72 and 75 exercise the slow path
  // (not a multiple of the 16-byte load width). Tensor cores engage from 32 rows;
  // small shapes there split K across blocks, and 3200 x 96 has enough output
  // tiles (50 at 64 rows) to run unsplit.
  struct Shape { int rows, cols; };
  for (int mode : {0, 1, 2})  // CUDA cores; tensor cores, exact BF16 split; tensor cores, one FP16 pass
  for (DType dt : {DType::BF16, DType::I8}) {
    set_tensor_cores_impl(mode > 0);
    set_tc_fp16_impl(mode == 2);
    const char* dn = dtype_name(dt);
    const char* path = mode == 0 ? "cuda-core" : mode == 1 ? "tc" : "tc-fp16";
    // Split-BF16 tensor-core results are within ~1e-5 of FP32; one FP16 pass
    // rounds every activation to 11 significant bits (~2.4e-4 relative).
    const double tol = mode == 0 ? 2e-5 : mode == 1 ? 1e-4 : 2e-3;
    for (Shape sh : {Shape{37, 72}, Shape{37, 75}, Shape{128, 896}, Shape{300, 896}, Shape{64, 4864}, Shape{3200, 96}}) {
      Matrix m = make_matrix(random_vec(rng, size_t(sh.rows) * sh.cols, 0.1f), sh.rows, sh.cols, dt);
      Matrix m2 = make_matrix(random_vec(rng, size_t(sh.rows) * sh.cols, 0.1f), sh.rows, sh.cols, dt);
      std::vector<float> deq = dequantized(m), deq2 = dequantized(m2);
      std::vector<float> bias = random_vec(rng, sh.rows, 1.0f);
      DeviceMatrix dW = mem.upload_matrix(m), dW2 = mem.upload_matrix(m2);
      float* dbias = mem.upload(bias);
      for (int T : {1, 2, 3, 4, 5, 9, 17, 32, 64, 100}) {
        std::vector<float> x = random_vec(rng, size_t(T) * sh.cols, 1.0f);
        std::vector<float> y0 = random_vec(rng, size_t(T) * sh.rows, 1.0f);
        float* dx = mem.upload(x);
        float* dy = mem.upload(y0);
        std::vector<float> want(size_t(T) * sh.rows);
        linear(want.data(), x.data(), deq.data(), bias.data(), T, sh.cols, sh.rows);

        matmul(dy, dx, dW, dbias, T, false, s);
        std::snprintf(what, sizeof(what), "matmul %s %s %dx%d T=%d", path, dn, sh.rows, sh.cols, T);
        tally.add(what, download(dy, want.size()), want, tol);

        check(cudaMemcpy(dy, y0.data(), y0.size() * sizeof(float), cudaMemcpyHostToDevice), "reset y");
        matmul(dy, dx, dW, dbias, T, true, s);
        std::vector<float> want_acc(want);
        for (size_t i = 0; i < want_acc.size(); ++i) want_acc[i] += y0[i];
        std::snprintf(what, sizeof(what), "matmul+residual %s %s %dx%d T=%d", path, dn, sh.rows, sh.cols, T);
        tally.add(what, download(dy, want.size()), want_acc, tol);

        std::vector<float> g(want.size()), u(want.size()), want_mlp(want.size());
        linear(g.data(), x.data(), deq.data(), nullptr, T, sh.cols, sh.rows);
        linear(u.data(), x.data(), deq2.data(), nullptr, T, sh.cols, sh.rows);
        for (size_t i = 0; i < g.size(); ++i) want_mlp[i] = silu(g[i]) * u[i];
        gate_up_silu(dy, dx, dW, dW2, T, s);
        std::snprintf(what, sizeof(what), "gate_up_silu %s %s %dx%d T=%d", path, dn, sh.rows, sh.cols, T);
        tally.add(what, download(dy, want.size()), want_mlp, tol);
      }
    }

    // embedding (a lookup: must be exact)
    const int vocab = 100, hidden = 896;
    Matrix e = make_matrix(random_vec(rng, size_t(vocab) * hidden, 0.1f), vocab, hidden, dt);
    std::vector<int32_t> ids = {5, 0, 99, 5};
    std::vector<float> want(ids.size() * hidden);
    for (size_t t = 0; t < ids.size(); ++t) e.row_to_f32(ids[t], &want[t * hidden]);
    DeviceMatrix de = mem.upload_matrix(e);
    int32_t* dids = mem.upload(ids);
    float* dx = mem.alloc<float>(want.size());
    embedding(dx, dids, de, int(ids.size()), s);
    std::snprintf(what, sizeof(what), "embedding %s", dn);
    tally.add(what, download(dx, want.size()), want, 0.0);
  }
  set_tensor_cores_impl(true);
  set_tc_fp16_impl(false);

  // Batching must not change results: on the CUDA-core path (batches under 32),
  // a token computed in a batch must equal the same token computed alone, bit
  // for bit, for both kernels and both weight formats, including partial groups
  // (batches that aren't a multiple of the 8 tokens processed per pass) and a
  // row count (900) that isn't a multiple of the rows per warp.
  for (DType dt : {DType::BF16, DType::I8}) {
    const int rows = 900, cols = 896;
    Matrix m = make_matrix(random_vec(rng, size_t(rows) * cols, 0.1f), rows, cols, dt);
    Matrix m2 = make_matrix(random_vec(rng, size_t(rows) * cols, 0.1f), rows, cols, dt);
    DeviceMatrix dW = mem.upload_matrix(m), dW2 = mem.upload_matrix(m2);
    float* dbias = mem.upload(random_vec(rng, rows, 1.0f));
    for (int T : {2, 8, 13, 31}) {
      std::vector<float> x = random_vec(rng, size_t(T) * cols, 1.0f);
      float* dx = mem.upload(x);
      float* dyb = mem.alloc<float>(size_t(T) * rows);
      float* dy1 = mem.alloc<float>(rows);
      bool same = true;
      for (int kernel = 0; kernel < 2; ++kernel) {
        if (kernel == 0) matmul(dyb, dx, dW, dbias, T, false, s);
        else gate_up_silu(dyb, dx, dW, dW2, T, s);
        for (int t = 0; t < T; ++t) {
          if (kernel == 0) matmul(dy1, dx + size_t(t) * cols, dW, dbias, 1, false, s);
          else gate_up_silu(dy1, dx + size_t(t) * cols, dW, dW2, 1, s);
          std::vector<float> a = download(dyb + size_t(t) * rows, rows), b = download(dy1, rows);
          same = same && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
        }
      }
      std::snprintf(what, sizeof(what), "batch of %d == %d single tokens, bit for bit (%s)", T, T,
                    dtype_name(dt));
      tally.add_exact(what, same);
    }
  }

  // rmsnorm
  {
    const int T = 3, dim = 896;
    std::vector<float> x = random_vec(rng, size_t(T) * dim, 2.0f), w = random_vec(rng, dim, 1.0f);
    x[5] = 1574.0f;  // a "massive activation", like token 0 in the real model
    std::vector<float> want(x.size());
    ie::rmsnorm(want.data(), x.data(), w.data(), T, dim, 1e-6f);
    float* dx = mem.upload(x);
    float* dw = mem.upload(w);
    float* dy = mem.alloc<float>(x.size());
    cuda::rmsnorm(dy, dx, dw, T, dim, 1e-6f, s);
    tally.add("rmsnorm 3x896", download(dy, x.size()), want);
  }

  const int NH = 14, NKV = 2, HD = 64, QD = NH * HD, KVD = NKV * HD;

  // rope_and_cache into a paged cache: rows from two sequences at different
  // positions, written to the right page and row of each sequence.
  {
    const float theta = 1e6f;
    TestPages pages({40, 20}, rng);  // sequence lengths after this step
    struct Row { int seq, pos; };
    const std::vector<Row> rows = {{0, 20}, {0, 21}, {1, 3}, {0, 39}};
    const int T = int(rows.size());
    std::vector<float> q = random_vec(rng, size_t(T) * QD, 1.0f);
    std::vector<float> k = random_vec(rng, size_t(T) * KVD, 1.0f);
    std::vector<float> v = random_vec(rng, size_t(T) * KVD, 1.0f);
    const size_t pool = size_t(pages.num_pages) * pages.page_size * KVD;
    std::vector<float> k_pool = random_vec(rng, pool, 1.0f), v_pool = random_vec(rng, pool, 1.0f);
    std::vector<float> inv_freq(HD / 2);
    for (int i = 0; i < HD / 2; ++i) inv_freq[i] = 1.0f / std::pow(theta, float(2 * i) / float(HD));

    std::vector<float> q_ref = q, kp_ref = k_pool, vp_ref = v_pool;
    std::vector<int> row_seq, row_pos;
    for (int t = 0; t < T; ++t) {
      ie::rope(&q_ref[size_t(t) * QD], 1, NH, HD, rows[t].pos, theta);
      std::vector<float> kr(k.begin() + size_t(t) * KVD, k.begin() + size_t(t + 1) * KVD);
      ie::rope(kr.data(), 1, NKV, HD, rows[t].pos, theta);
      const size_t dst = pages.row(rows[t].seq, rows[t].pos, KVD);
      std::copy(kr.begin(), kr.end(), kp_ref.begin() + dst);
      std::copy(v.begin() + size_t(t) * KVD, v.begin() + size_t(t + 1) * KVD, vp_ref.begin() + dst);
      row_seq.push_back(rows[t].seq);
      row_pos.push_back(rows[t].pos);
    }

    KvLayout kv;
    kv.tables = mem.upload(pages.tables);
    kv.max_pages = pages.max_pages;
    kv.page_size = pages.page_size;
    float* dq = mem.upload(q);
    float* dk = mem.upload(k);
    float* dv = mem.upload(v);
    float* dkp = mem.upload(k_pool);
    float* dvp = mem.upload(v_pool);
    rope_and_cache(dq, dk, dv, dkp, dvp, mem.upload(inv_freq), T, mem.upload(row_pos),
                   mem.upload(row_seq), kv, NH, NKV, HD, s);
    tally.add("paged rope (queries)", download(dq, q.size()), q_ref);
    tally.add("paged rope + cache write (keys)", download(dkp, pool), kp_ref);
    tally.add("paged cache write (values)", download(dvp, pool), vp_ref, 0.0);

    // The same into an FP16 cache: written rows hold the FP32 results rounded to
    // FP16, other rows are untouched. Keys may differ by one FP16 step where the
    // GPU's and the CPU's rotation round to different sides of a boundary.
    kv.f16 = true;
    __half* hkp = mem.upload(to_f16(k_pool));
    __half* hvp = mem.upload(to_f16(v_pool));
    rope_and_cache(mem.upload(q), dk, dv, hkp, hvp, mem.upload(inv_freq), T, mem.upload(row_pos),
                   mem.upload(row_seq), kv, NH, NKV, HD, s);
    tally.add("paged rope + cache write (keys, f16 cache)", from_f16(download(hkp, pool)),
              round_f16(kp_ref), 2e-3);
    tally.add("paged cache write (values, f16 cache)", from_f16(download(hvp, pool)),
              round_f16(vp_ref), 0.0);
  }

  // Flash-decoding attention over paged caches vs the CPU on contiguous copies:
  // single sequences, batches, mixed prefill + decode rows, chunk boundaries
  // (256), long context, and extra empty chunks as when replaying a graph.
  {
    struct Row { int seq, pos; };
    // prompt: rows are consecutive positions of one sequence, so the prefill
    // kernel can run them too (both kernels are checked).
    struct Case { std::vector<int> lens; std::vector<Row> rows; int capacity; const char* name; bool prompt = false; };
    std::vector<Case> cases;
    cases.push_back({{1}, {{0, 0}}, 1, "first token"});
    cases.push_back({{64}, {{0, 63}}, 64, "chunk edge 63"});
    cases.push_back({{65}, {{0, 64}}, 1000, "chunk edge 64 + empty chunks"});
    cases.push_back({{256}, {{0, 255}}, 256, "position 255"});
    cases.push_back({{257}, {{0, 256}}, 4096, "position 256 + empty chunks"});
    cases.push_back({{3000}, {{0, 2999}}, 3000, "long context"});
    {
      Case c{{540}, {}, 540, "prompt continuing at 500 (40 rows)", true};
      for (int p = 500; p < 540; ++p) c.rows.push_back({0, p});
      cases.push_back(c);
    }
    for (int len : {1, 17, 300, 1000}) {  // whole prompts from position 0
      Case c{{len}, {}, len, nullptr, true};
      for (int p = 0; p < len; ++p) c.rows.push_back({0, p});
      cases.push_back(c);
    }
    {
      Case c{{100, 300}, {}, 300, "prompt of the 2nd of 2 sequences (300 rows)", true};
      for (int p = 0; p < 300; ++p) c.rows.push_back({1, p});
      cases.push_back(c);
    }
    cases.push_back({{6, 17, 256, 257, 3000}, {{0, 5}, {1, 16}, {2, 255}, {3, 256}, {4, 2999}}, 4096,
                     "batch of 5 decodes"});
    {
      Case c{{101, 30, 1001}, {{0, 100}, {2, 1000}}, 1024, "mixed prefill + decode"};
      for (int p = 0; p < 30; ++p) c.rows.push_back({1, p});
      cases.push_back(c);
    }
    // Each case runs with an FP32 and an FP16 cache; for FP16 the cache values
    // are rounded first and the CPU reference uses the same rounded values.
    for (bool f16 : {false, true})
    for (const Case& c : cases) {
      TestPages pages(c.lens, rng);
      const int T = int(c.rows.size());
      const size_t pool = size_t(pages.num_pages) * pages.page_size * KVD;
      std::vector<float> k_pool = random_vec(rng, pool, 1.0f), v_pool = random_vec(rng, pool, 1.0f);
      if (f16) {
        k_pool = round_f16(k_pool);
        v_pool = round_f16(v_pool);
      }
      std::vector<float> q = random_vec(rng, size_t(T) * QD, 1.0f), want(size_t(T) * QD);
      std::vector<int> row_seq, row_pos;
      for (int t = 0; t < T; ++t) {
        const Row r = c.rows[t];
        // CPU reference: this row alone over a contiguous copy of its sequence.
        std::vector<float> kc(size_t(r.pos + 1) * KVD), vc(kc.size());
        for (int p = 0; p <= r.pos; ++p) {
          const size_t src = pages.row(r.seq, p, KVD);
          std::copy(k_pool.begin() + src, k_pool.begin() + src + KVD, kc.begin() + size_t(p) * KVD);
          std::copy(v_pool.begin() + src, v_pool.begin() + src + KVD, vc.begin() + size_t(p) * KVD);
        }
        ie::attention(&want[size_t(t) * QD], &q[size_t(t) * QD], kc.data(), vc.data(), 1, r.pos, NH, NKV, HD);
        row_seq.push_back(r.seq);
        row_pos.push_back(r.pos);
      }
      KvLayout kv;
      kv.tables = mem.upload(pages.tables);
      kv.max_pages = pages.max_pages;
      kv.page_size = pages.page_size;
      kv.f16 = f16;
      const void* dkp = f16 ? static_cast<const void*>(mem.upload(to_f16(k_pool)))
                            : static_cast<const void*>(mem.upload(k_pool));
      const void* dvp = f16 ? static_cast<const void*>(mem.upload(to_f16(v_pool)))
                            : static_cast<const void*>(mem.upload(v_pool));
      const int chunks = attention_chunks(c.capacity);
      float* part_o = mem.alloc<float>(size_t(T) * NH * chunks * HD);
      float* part_ml = mem.alloc<float>(size_t(T) * NH * chunks * 2);
      float* dout = mem.alloc<float>(want.size());
      cuda::attention(dout, mem.upload(q), dkp, dvp, T, mem.upload(row_pos), mem.upload(row_seq), kv,
                      chunks, part_o, part_ml, NH, NKV, HD, s);
      char name[80];
      if (c.name) std::snprintf(name, sizeof(name), "%s%s", c.name, f16 ? " (f16 cache)" : "");
      else std::snprintf(name, sizeof(name), "prompt of %d rows%s", T, f16 ? " (f16 cache)" : "");
      std::snprintf(what, sizeof(what), "paged attention: %s", name);
      tally.add(what, download(dout, want.size()), want);
      if (c.prompt) {
        float* dout2 = mem.alloc<float>(want.size());
        attention_prefill(dout2, mem.upload(q), dkp, dvp, T, mem.upload(row_pos), mem.upload(row_seq),
                          kv, NH, NKV, HD, s);
        std::snprintf(what, sizeof(what), "prefill attention: %s", name);
        tally.add(what, download(dout2, want.size()), want);
      }
    }
  }

  // gather_rows and argmax_rows
  {
    const int rows = 9, dim = 896;
    std::vector<float> x = random_vec(rng, size_t(rows) * dim, 1.0f);
    std::vector<int> pick = {8, 0, 3};
    std::vector<float> want;
    for (int r : pick) want.insert(want.end(), x.begin() + size_t(r) * dim, x.begin() + size_t(r + 1) * dim);
    float* dst = mem.alloc<float>(want.size());
    gather_rows(dst, mem.upload(x), mem.upload(pick), int(pick.size()), dim, s);
    tally.add("gather_rows", download(dst, want.size()), want, 0.0);

    const int n = 151936;  // the real vocabulary size
    std::vector<float> logits = random_vec(rng, size_t(3) * n, 10.0f);
    logits[size_t(0) * n + 151935] = 50.0f;                                      // max at the very end
    logits[size_t(1) * n + 7] = 50.0f;  logits[size_t(1) * n + 90000] = 50.0f;  // tie: first one wins
    logits[size_t(2) * n + 0] = 50.0f;                                           // max at the start
    int32_t* dout = mem.alloc<int32_t>(3);
    argmax_rows(dout, mem.upload(logits), 3, n, s);
    std::vector<int32_t> got = download(dout, 3);
    bool ok = true;
    for (int r = 0; r < 3; ++r) ok = ok && got[r] == ie::argmax(&logits[size_t(r) * n], n);
    tally.add_exact("argmax_rows (end, tie, start)", ok && got[1] == 7);
  }

  check(cudaDeviceSynchronize(), "self-test");
  std::printf("%s: %d CUDA kernel tests, %d failed (worst relative error %.2e)\n",
              tally.failures ? "FAILED" : "ALL PASSED", tally.tests, tally.failures, tally.worst);
  return tally.failures ? 1 : 0;
}

}  // namespace ie::cuda
