#include "cuda/cuda_runner.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda/kernels.cuh"
#include "npy.h"
#include "ops.h"

namespace ie::cuda {

namespace {

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
 private:
  std::vector<void*> ptrs_;
};

struct DeviceLayer {
  float* input_norm = nullptr;
  uint16_t *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
  float *bq = nullptr, *bk = nullptr, *bv = nullptr;
  float* post_attn_norm = nullptr;
  uint16_t *w_gate = nullptr, *w_up = nullptr, *w_down = nullptr;
};

class CudaRunner : public Runner {
 public:
  CudaRunner(const Config& cfg, const Weights& w, int max_tokens) : cfg_(cfg), max_tokens_(max_tokens) {
    if (w.dtype != DType::BF16)
      throw std::runtime_error("the CUDA backend runs BF16 weights (use --weights bf16)");
    if (max_tokens > attention_max_tokens())
      throw std::runtime_error("--ctx " + std::to_string(max_tokens) + " is larger than the GPU attention kernel supports (" +
                               std::to_string(attention_max_tokens()) + ")");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    name_ = std::string("cuda, ") + prop.name;

    // Weights: copied once, stay in VRAM as BF16.
    embed_ = mem_.upload(w.embed.bf16);
    lm_head_ = w.lm_head.rows ? mem_.upload(w.lm_head.bf16) : embed_;  // tied: reuse the embedding
    final_norm_ = mem_.upload(w.final_norm);
    for (const LayerWeights& L : w.layers) {
      DeviceLayer d;
      d.input_norm = mem_.upload(L.input_norm);
      d.wq = mem_.upload(L.wq.bf16);
      d.wk = mem_.upload(L.wk.bf16);
      d.wv = mem_.upload(L.wv.bf16);
      d.wo = mem_.upload(L.wo.bf16);
      d.bq = mem_.upload(L.bq);
      d.bk = mem_.upload(L.bk);
      d.bv = mem_.upload(L.bv);
      d.post_attn_norm = mem_.upload(L.post_attn_norm);
      d.w_gate = mem_.upload(L.w_gate.bf16);
      d.w_up = mem_.upload(L.w_up.bf16);
      d.w_down = mem_.upload(L.w_down.bf16);
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
  }

  std::vector<float> forward(const std::vector<int32_t>& ids, bool all_logits,
                             const StageCallback& on_stage) override {
    const int T = int(ids.size()), pos0 = len_;
    const int H = cfg_.hidden_size, I = cfg_.intermediate_size, V = cfg_.vocab_size;
    const int QD = cfg_.q_dim(), KVD = cfg_.kv_dim(), HD = cfg_.head_dim;
    const int NH = cfg_.num_heads, NKV = cfg_.num_kv_heads;
    if (T == 0) throw std::runtime_error("forward: no tokens");
    if (pos0 + T > max_tokens_)
      throw std::runtime_error("forward: KV cache full (" + std::to_string(max_tokens_) + " tokens)");
    for (int32_t id : ids)
      if (id < 0 || id >= V) throw std::runtime_error("token id out of range");
    const int logit_rows = all_logits ? T : 1;
    reserve(T, logit_rows);

    auto stage = [&](const char* name, const float* d, int rows, int cols) {
      if (!on_stage) return;
      std::vector<float> h(size_t(rows) * cols);
      check(cudaMemcpy(h.data(), d, h.size() * sizeof(float), cudaMemcpyDeviceToHost), "stage copy");
      on_stage(name, h.data(), rows, cols);
    };

    check(cudaMemcpy(ids_, ids.data(), sizeof(int32_t) * T, cudaMemcpyHostToDevice), "ids");
    embedding(x_, ids_, embed_, T, H);
    stage("embed", x_, T, H);

    const size_t cache_stride = size_t(max_tokens_) * KVD;
    for (int l = 0; l < cfg_.num_layers; ++l) {
      const DeviceLayer& L = layers_[l];
      float* kc = k_cache_ + l * cache_stride;
      float* vc = v_cache_ + l * cache_stride;

      // attention block: x += o_proj(attention(rope(q), rope(k), v))
      rmsnorm(xn_, x_, L.input_norm, T, H, cfg_.rms_norm_eps);
      matmul_bf16(q_, xn_, L.wq, L.bq, T, QD, H, false);
      matmul_bf16(k_, xn_, L.wk, L.bk, T, KVD, H, false);
      matmul_bf16(v_, xn_, L.wv, L.bv, T, KVD, H, false);
      rope_and_cache(q_, k_, v_, kc, vc, inv_freq_, T, pos0, NH, NKV, HD);
      attention(attn_, q_, kc, vc, T, pos0, NH, NKV, HD);
      matmul_bf16(x_, attn_, L.wo, nullptr, T, H, QD, /*accumulate=*/true);

      // MLP block: x += down(silu(gate(xn)) * up(xn))
      rmsnorm(xn_, x_, L.post_attn_norm, T, H, cfg_.rms_norm_eps);
      gate_up_silu(mid_, xn_, L.w_gate, L.w_up, T, I, H);
      matmul_bf16(x_, mid_, L.w_down, nullptr, T, H, I, /*accumulate=*/true);

      if (on_stage) {
        char name[32];
        std::snprintf(name, sizeof(name), "layer_%02d", l);
        stage(name, x_, T, H);
      }
    }
    len_ += T;

    rmsnorm(xn_, x_, final_norm_, T, H, cfg_.rms_norm_eps);
    stage("final_norm", xn_, T, H);
    matmul_bf16(logits_, xn_ + size_t(T - logit_rows) * H, lm_head_, nullptr, logit_rows, V, H, false);

    std::vector<float> out(size_t(logit_rows) * V);
    // The copy waits for every queued kernel, so errors inside them surface here.
    check(cudaMemcpy(out.data(), logits_, out.size() * sizeof(float), cudaMemcpyDeviceToHost), "forward");
    if (on_stage) on_stage("logits", out.data(), logit_rows, V);
    return out;
  }

  void reset() override { len_ = 0; }
  int position() const override { return len_; }
  int max_tokens() const override { return max_tokens_; }
  const Config& config() const override { return cfg_; }
  std::string device_name() const override { return name_; }
  DType weight_dtype() const override { return DType::BF16; }
  size_t weight_bytes() const override { return weight_bytes_; }
  size_t num_params() const override { return num_params_; }
  size_t kv_cache_bytes() const override { return kv_bytes_; }

 private:
  // Scratch buffers sized for T tokens, grown on demand.
  void reserve(int T, int logit_rows) {
    if (T > cap_tokens_) {
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
    }
    if (logit_rows > cap_logit_rows_) {
      mem_.release(logits_);
      cap_logit_rows_ = logit_rows;
      logits_ = mem_.alloc<float>(size_t(logit_rows) * cfg_.vocab_size);
    }
  }

  Config cfg_;
  int max_tokens_ = 0, len_ = 0;
  std::string name_;
  size_t weight_bytes_ = 0, num_params_ = 0, kv_bytes_ = 0;
  DeviceMemory mem_;

  uint16_t *embed_ = nullptr, *lm_head_ = nullptr;
  float* final_norm_ = nullptr;
  std::vector<DeviceLayer> layers_;
  float* inv_freq_ = nullptr;
  float *k_cache_ = nullptr, *v_cache_ = nullptr;

  int cap_tokens_ = 0, cap_logit_rows_ = 0;
  int32_t* ids_ = nullptr;
  float *x_ = nullptr, *xn_ = nullptr, *q_ = nullptr, *k_ = nullptr, *v_ = nullptr;
  float *attn_ = nullptr, *mid_ = nullptr, *logits_ = nullptr;
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

std::unique_ptr<Runner> make_runner(const Config& cfg, const Weights& w, int max_tokens) {
  return std::make_unique<CudaRunner>(cfg, w, max_tokens);
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
      std::printf("FAIL %-40s rel_err %.3e (worst at index %zu)\n", what, r.rel_err, r.worst_index);
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

  // matmul_bf16 and gate_up_silu against ie::linear on the same (BF16-rounded) weights
  struct Shape { int rows, cols; };
  for (Shape sh : {Shape{37, 72}, Shape{37, 75}, Shape{128, 896}, Shape{300, 896}, Shape{64, 4864}}) {
    Matrix m = make_matrix(random_vec(rng, size_t(sh.rows) * sh.cols, 0.1f), sh.rows, sh.cols, DType::BF16);
    Matrix m2 = make_matrix(random_vec(rng, size_t(sh.rows) * sh.cols, 0.1f), sh.rows, sh.cols, DType::BF16);
    std::vector<float> deq(size_t(sh.rows) * sh.cols), deq2(deq.size());
    for (int r = 0; r < sh.rows; ++r) {
      m.row_to_f32(r, &deq[size_t(r) * sh.cols]);
      m2.row_to_f32(r, &deq2[size_t(r) * sh.cols]);
    }
    std::vector<float> bias = random_vec(rng, sh.rows, 1.0f);
    uint16_t* dW = mem.upload(m.bf16);
    uint16_t* dW2 = mem.upload(m2.bf16);
    float* dbias = mem.upload(bias);
    for (int T : {1, 2, 3, 4, 5, 9}) {
      std::vector<float> x = random_vec(rng, size_t(T) * sh.cols, 1.0f);
      std::vector<float> y0 = random_vec(rng, size_t(T) * sh.rows, 1.0f);
      float* dx = mem.upload(x);
      float* dy = mem.upload(y0);
      std::vector<float> want(size_t(T) * sh.rows);
      linear(want.data(), x.data(), deq.data(), bias.data(), T, sh.cols, sh.rows);

      matmul_bf16(dy, dx, dW, dbias, T, sh.rows, sh.cols, false);
      std::snprintf(what, sizeof(what), "matmul %dx%d T=%d", sh.rows, sh.cols, T);
      tally.add(what, download(dy, want.size()), want);

      check(cudaMemcpy(dy, y0.data(), y0.size() * sizeof(float), cudaMemcpyHostToDevice), "reset y");
      matmul_bf16(dy, dx, dW, dbias, T, sh.rows, sh.cols, true);
      std::vector<float> want_acc(want);
      for (size_t i = 0; i < want_acc.size(); ++i) want_acc[i] += y0[i];
      std::snprintf(what, sizeof(what), "matmul+residual %dx%d T=%d", sh.rows, sh.cols, T);
      tally.add(what, download(dy, want.size()), want_acc);

      std::vector<float> g(want.size()), u(want.size()), want_mlp(want.size());
      linear(g.data(), x.data(), deq.data(), nullptr, T, sh.cols, sh.rows);
      linear(u.data(), x.data(), deq2.data(), nullptr, T, sh.cols, sh.rows);
      for (size_t i = 0; i < g.size(); ++i) want_mlp[i] = silu(g[i]) * u[i];
      gate_up_silu(dy, dx, dW, dW2, T, sh.rows, sh.cols);
      std::snprintf(what, sizeof(what), "gate_up_silu %dx%d T=%d", sh.rows, sh.cols, T);
      tally.add(what, download(dy, want.size()), want_mlp);
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
    cuda::rmsnorm(dy, dx, dw, T, dim, 1e-6f);
    tally.add("rmsnorm 3x896", download(dy, x.size()), want);
  }

  // embedding
  {
    const int vocab = 100, hidden = 896;
    Matrix e = make_matrix(random_vec(rng, size_t(vocab) * hidden, 0.1f), vocab, hidden, DType::BF16);
    std::vector<int32_t> ids = {5, 0, 99, 5};
    std::vector<float> want(ids.size() * hidden);
    for (size_t t = 0; t < ids.size(); ++t) e.row_to_f32(ids[t], &want[t * hidden]);
    uint16_t* de = mem.upload(e.bf16);
    int32_t* dids = mem.upload(ids);
    float* dx = mem.alloc<float>(want.size());
    embedding(dx, dids, de, int(ids.size()), hidden);
    tally.add("embedding", download(dx, want.size()), want, 0.0);
  }

  // rope_and_cache + attention, with tokens already in the cache (pos0 > 0)
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
    float* dout = mem.alloc<float>(attn_ref.size());
    rope_and_cache(dq, dk, dv, dkc, dvc, dinv, T, pos0, NH, NKV, HD);
    tally.add("rope (queries)", download(dq, q.size()), q_ref);
    tally.add("rope + cache write (keys)", download(dkc, k_cache.size()), kc_ref);
    tally.add("cache write (values)", download(dvc, v_cache.size()), vc_ref, 0.0);
    cuda::attention(dout, dq, dkc, dvc, T, pos0, NH, NKV, HD);
    tally.add("attention T=3 pos0=6", download(dout, attn_ref.size()), attn_ref);
  }

  check(cudaDeviceSynchronize(), "self-test");
  std::printf("%s: %d CUDA kernel tests, %d failed (worst relative error %.2e)\n",
              tally.failures ? "FAILED" : "ALL PASSED", tally.tests, tally.failures, tally.worst);
  return tally.failures ? 1 : 0;
}

}  // namespace ie::cuda
