#include "cuda/cuda_runner.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda/kernels.cuh"
#include "npy.h"
#include "ops.h"

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

struct NvtxRange {
  explicit NvtxRange(const char* name) { IE_NVTX_PUSH(name); }
  ~NvtxRange() { IE_NVTX_POP(); }
};

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess)
    throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(err));
}

// Owns every device allocation of one runner and frees them together.
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
  void resize(size_t n) {
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

class CudaRunner : public Runner {
 public:
  CudaRunner(const Config& cfg, const Weights& w, int max_tokens, bool use_graphs)
      : cfg_(cfg), max_tokens_(max_tokens), use_graphs_(use_graphs), dtype_(w.dtype) {
    if (w.dtype != DType::BF16 && w.dtype != DType::I8)
      throw std::runtime_error("the CUDA backend runs bf16 or int8 weights (not f32)");
    if (max_tokens > attention_max_tokens())
      throw std::runtime_error("--ctx " + std::to_string(max_tokens) + " is larger than the GPU attention kernel supports (" +
                               std::to_string(attention_max_tokens()) + ")");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    name_ = std::string("cuda, ") + prop.name + (use_graphs_ ? ", graphs" : ", no graphs");
    check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");

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

    // KV cache: [layer][position][kv_dim] for keys and for values.
    const size_t per_layer = size_t(max_tokens_) * cfg_.kv_dim();
    k_cache_ = mem_.alloc<float>(per_layer * cfg_.num_layers);
    v_cache_ = mem_.alloc<float>(per_layer * cfg_.num_layers);
    kv_bytes_ = 2 * per_layer * cfg_.num_layers * sizeof(float);

    pos_ = mem_.alloc<int>(1);
    h_pos_.resize(1);
  }

  ~CudaRunner() override {
    drop_graph();
    if (stream_) cudaStreamDestroy(stream_);
  }

  std::vector<float> forward(const std::vector<int32_t>& ids, bool all_logits,
                             const StageCallback& on_stage) override {
    const int T = int(ids.size()), V = cfg_.vocab_size;
    if (T == 0) throw std::runtime_error("forward: no tokens");
    if (len_ + T > max_tokens_)
      throw std::runtime_error("forward: KV cache full (" + std::to_string(max_tokens_) + " tokens)");
    for (int32_t id : ids)
      if (id < 0 || id >= V) throw std::runtime_error("token id out of range");
    const int logit_rows = all_logits ? T : 1;
    NvtxRange range(T == 1 ? "decode" : "prefill");
    reserve(T, logit_rows);  // before any capture: no allocations allowed inside a graph

    // A decode step (1 token, no inspection) replays the recorded graph.
    const bool use_graph = use_graphs_ && T == 1 && !all_logits && !on_stage;
    if (use_graph && !graph_) capture_decode_graph();

    // Inputs: the token ids and the current position, read by the kernels from VRAM.
    std::memcpy(h_ids_.data(), ids.data(), sizeof(int32_t) * T);
    *h_pos_.data() = len_;
    check(cudaMemcpyAsync(ids_, h_ids_.data(), sizeof(int32_t) * T, cudaMemcpyHostToDevice, stream_), "ids");
    check(cudaMemcpyAsync(pos_, h_pos_.data(), sizeof(int), cudaMemcpyHostToDevice, stream_), "pos");

    if (use_graph)
      check(cudaGraphLaunch(graph_, stream_), "cudaGraphLaunch");  // ~243 kernels, one call
    else
      enqueue(T, logit_rows, on_stage ? &on_stage : nullptr);

    const size_t n = size_t(logit_rows) * V;
    check(cudaMemcpyAsync(h_logits_.data(), logits_, n * sizeof(float), cudaMemcpyDeviceToHost, stream_), "logits");
    // Waits for everything queued above; errors from inside kernels surface here.
    check(cudaStreamSynchronize(stream_), "forward");
    len_ += T;

    std::vector<float> out(h_logits_.data(), h_logits_.data() + n);
    if (on_stage) on_stage("logits", out.data(), logit_rows, V);
    return out;
  }

  void reset() override { len_ = 0; }
  int position() const override { return len_; }
  int max_tokens() const override { return max_tokens_; }
  const Config& config() const override { return cfg_; }
  std::string device_name() const override { return name_; }
  DType weight_dtype() const override { return dtype_; }
  size_t weight_bytes() const override { return weight_bytes_; }
  size_t num_params() const override { return num_params_; }
  size_t kv_cache_bytes() const override { return kv_bytes_; }

 private:
  // Queues the whole forward pass for T tokens on stream_. Used directly for
  // prefill, and recorded once (T = 1) into the decode graph.
  void enqueue(int T, int logit_rows, const StageCallback* on_stage) {
    const int H = cfg_.hidden_size, NH = cfg_.num_heads, NKV = cfg_.num_kv_heads;
    const int HD = cfg_.head_dim, KVD = cfg_.kv_dim();
    auto stage = [&](const char* name, const float* d, int cols) {
      if (!on_stage) return;
      std::vector<float> h(size_t(T) * cols);
      check(cudaMemcpyAsync(h.data(), d, h.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_), "stage copy");
      check(cudaStreamSynchronize(stream_), "stage copy");
      (*on_stage)(name, h.data(), T, cols);
    };

    embedding(x_, ids_, embed_, T, stream_);
    stage("embed", x_, H);

    const size_t cache_stride = size_t(max_tokens_) * KVD;
    for (int l = 0; l < cfg_.num_layers; ++l) {
      const DeviceLayer& L = layers_[l];
      float* kc = k_cache_ + l * cache_stride;
      float* vc = v_cache_ + l * cache_stride;

      // attention block: x += o_proj(attention(rope(q), rope(k), v))
      rmsnorm(xn_, x_, L.input_norm, T, H, cfg_.rms_norm_eps, stream_);
      matmul(q_, xn_, L.wq, L.bq, T, false, stream_);
      matmul(k_, xn_, L.wk, L.bk, T, false, stream_);
      matmul(v_, xn_, L.wv, L.bv, T, false, stream_);
      rope_and_cache(q_, k_, v_, kc, vc, inv_freq_, T, pos_, NH, NKV, HD, stream_);
      attention(attn_, q_, kc, vc, T, pos_, max_tokens_, NH, NKV, HD, stream_);
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
    matmul(logits_, xn_ + size_t(T - logit_rows) * H, lm_head_, nullptr, logit_rows, false, stream_);
  }

  // Records one decode step's kernel launches into a CUDA graph. Replaying it
  // costs a single launch instead of ~243, which matters most on Windows, where
  // each individual launch has noticeable driver overhead.
  void capture_decode_graph() {
    NvtxRange range("capture decode graph");
    cudaGraph_t graph = nullptr;
    check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
    try {
      enqueue(1, 1, nullptr);
    } catch (...) {
      cudaStreamEndCapture(stream_, &graph);
      if (graph) cudaGraphDestroy(graph);
      throw;
    }
    check(cudaStreamEndCapture(stream_, &graph), "cudaStreamEndCapture");
    cudaError_t err = cudaGraphInstantiate(&graph_, graph, 0);
    cudaGraphDestroy(graph);
    check(err, "cudaGraphInstantiate");
  }

  void drop_graph() {
    if (graph_) cudaGraphExecDestroy(graph_);
    graph_ = nullptr;
  }

  // Scratch buffers sized for T tokens, grown on demand. A recorded graph holds
  // the old buffer addresses, so it must be re-recorded after any reallocation.
  void reserve(int T, int logit_rows) {
    if (T > cap_tokens_) {
      drop_graph();
      for (void* p : {(void*)ids_, (void*)x_, (void*)xn_, (void*)q_, (void*)k_, (void*)v_,
                      (void*)attn_, (void*)mid_})
        mem_.release(p);
      cap_tokens_ = std::max(T, 16);
      const size_t n = size_t(cap_tokens_);
      ids_ = mem_.alloc<int32_t>(n);
      x_ = mem_.alloc<float>(n * cfg_.hidden_size);
      xn_ = mem_.alloc<float>(n * cfg_.hidden_size);
      q_ = mem_.alloc<float>(n * cfg_.q_dim());
      k_ = mem_.alloc<float>(n * cfg_.kv_dim());
      v_ = mem_.alloc<float>(n * cfg_.kv_dim());
      attn_ = mem_.alloc<float>(n * cfg_.q_dim());
      mid_ = mem_.alloc<float>(n * cfg_.intermediate_size);
      h_ids_.resize(n);
    }
    if (logit_rows > cap_logit_rows_) {
      drop_graph();
      mem_.release(logits_);
      cap_logit_rows_ = logit_rows;
      logits_ = mem_.alloc<float>(size_t(logit_rows) * cfg_.vocab_size);
      h_logits_.resize(size_t(logit_rows) * cfg_.vocab_size);
    }
  }

  Config cfg_;
  int max_tokens_ = 0, len_ = 0;
  bool use_graphs_ = true;
  DType dtype_;
  std::string name_;
  size_t weight_bytes_ = 0, num_params_ = 0, kv_bytes_ = 0;
  DeviceMemory mem_;
  cudaStream_t stream_ = nullptr;
  cudaGraphExec_t graph_ = nullptr;

  DeviceMatrix embed_, lm_head_;
  float* final_norm_ = nullptr;
  std::vector<DeviceLayer> layers_;
  float* inv_freq_ = nullptr;
  float *k_cache_ = nullptr, *v_cache_ = nullptr;
  int* pos_ = nullptr;

  int cap_tokens_ = 0, cap_logit_rows_ = 0;
  int32_t* ids_ = nullptr;
  float *x_ = nullptr, *xn_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr;
  float *attn_ = nullptr, *mid_ = nullptr, *logits_ = nullptr;
  PinnedBuffer<int32_t> h_ids_;
  PinnedBuffer<int> h_pos_;
  PinnedBuffer<float> h_logits_;
};

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

std::unique_ptr<Runner> make_runner(const Config& cfg, const Weights& w, int max_tokens,
                                    bool use_graphs) {
  return std::make_unique<CudaRunner>(cfg, w, max_tokens, use_graphs);
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
};

}  // namespace

int run_selftest() {
  std::printf("GPU: %s\n", device_description().c_str());
  std::mt19937 rng(42);
  Tally tally;
  DeviceMemory mem;
  char what[128];
  const cudaStream_t s = nullptr;  // the default stream is fine for tests

  // matmul and gate_up_silu, for both weight formats, against ie::linear on the
  // exact values the GPU holds. Column counts 72 and 75 exercise the slow path
  // (not a multiple of the 16-byte load width).
  struct Shape { int rows, cols; };
  for (bool tc : {false, true})
  for (DType dt : {DType::BF16, DType::I8}) {
    set_tensor_cores_impl(tc);
    const char* dn = dtype_name(dt);
    const char* path = tc ? "tc" : "cuda-core";
    // Split-BF16 tensor-core results are within ~1e-5 of FP32; allow a little more.
    const double tol = tc ? 1e-4 : 2e-5;
    for (Shape sh : {Shape{37, 72}, Shape{37, 75}, Shape{128, 896}, Shape{300, 896}, Shape{64, 4864}}) {
      Matrix m = make_matrix(random_vec(rng, size_t(sh.rows) * sh.cols, 0.1f), sh.rows, sh.cols, dt);
      Matrix m2 = make_matrix(random_vec(rng, size_t(sh.rows) * sh.cols, 0.1f), sh.rows, sh.cols, dt);
      std::vector<float> deq = dequantized(m), deq2 = dequantized(m2);
      std::vector<float> bias = random_vec(rng, sh.rows, 1.0f);
      DeviceMatrix dW = mem.upload_matrix(m), dW2 = mem.upload_matrix(m2);
      float* dbias = mem.upload(bias);
      for (int T : {1, 2, 3, 4, 5, 9, 17, 64, 100}) {
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

  // rope_and_cache + attention, with tokens already in the cache (position 6)
  {
    const int NH = 14, NKV = 2, HD = 64, T = 3, pos0 = 6, max_tokens = 16;
    const int QD = NH * HD, KVD = NKV * HD;
    const float theta = 1e6f;
    std::vector<float> q = random_vec(rng, size_t(T) * QD, 1.0f);
    std::vector<float> k = random_vec(rng, size_t(T) * KVD, 1.0f);
    std::vector<float> v = random_vec(rng, size_t(T) * KVD, 1.0f);
    std::vector<float> k_cache = random_vec(rng, size_t(max_tokens) * KVD, 1.0f);
    std::vector<float> v_cache = random_vec(rng, size_t(max_tokens) * KVD, 1.0f);
    std::vector<float> inv_freq(HD / 2);
    for (int i = 0; i < HD / 2; ++i) inv_freq[i] = 1.0f / std::pow(theta, float(2 * i) / float(HD));

    // CPU reference: rotate, store into the cache, attend
    std::vector<float> q_ref = q, k_rot = k, kc_ref = k_cache, vc_ref = v_cache;
    ie::rope(q_ref.data(), T, NH, HD, pos0, theta);
    ie::rope(k_rot.data(), T, NKV, HD, pos0, theta);
    std::copy(k_rot.begin(), k_rot.end(), kc_ref.begin() + size_t(pos0) * KVD);
    std::copy(v.begin(), v.end(), vc_ref.begin() + size_t(pos0) * KVD);
    std::vector<float> attn_ref(size_t(T) * QD);
    ie::attention(attn_ref.data(), q_ref.data(), kc_ref.data(), vc_ref.data(), T, pos0, NH, NKV, HD);

    float* dq = mem.upload(q);
    float* dk = mem.upload(k);
    float* dv = mem.upload(v);
    float* dkc = mem.upload(k_cache);
    float* dvc = mem.upload(v_cache);
    float* dinv = mem.upload(inv_freq);
    int* dpos = mem.upload(std::vector<int>{pos0});
    float* dout = mem.alloc<float>(attn_ref.size());
    rope_and_cache(dq, dk, dv, dkc, dvc, dinv, T, dpos, NH, NKV, HD, s);
    tally.add("rope (queries)", download(dq, q.size()), q_ref);
    tally.add("rope + cache write (keys)", download(dkc, k_cache.size()), kc_ref);
    tally.add("cache write (values)", download(dvc, v_cache.size()), vc_ref, 0.0);
    cuda::attention(dout, dq, dkc, dvc, T, dpos, max_tokens, NH, NKV, HD, s);
    tally.add("attention T=3 pos=6", download(dout, attn_ref.size()), attn_ref);
  }

  check(cudaDeviceSynchronize(), "self-test");
  std::printf("%s: %d CUDA kernel tests, %d failed (worst relative error %.2e)\n",
              tally.failures ? "FAILED" : "ALL PASSED", tally.tests, tally.failures, tally.worst);
  return tally.failures ? 1 : 0;
}

}  // namespace ie::cuda
