#include "matrix.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "threadpool.h"

#if defined(__AVX2__)
#include <immintrin.h>
#define IE_HAVE_AVX2 1
#else
#define IE_HAVE_AVX2 0
#endif

namespace ie {

// ------------------------------------------------------------ dtypes ----

DType parse_dtype(const std::string& s) {
  if (s == "f32" || s == "fp32") return DType::F32;
  if (s == "bf16") return DType::BF16;
  if (s == "int8" || s == "i8" || s == "q8") return DType::I8;
  throw std::runtime_error("unknown weight type '" + s + "' (use f32, bf16 or int8)");
}
const char* dtype_name(DType d) {
  switch (d) {
    case DType::F32: return "f32";
    case DType::BF16: return "bf16";
    case DType::I8: return "int8";
  }
  return "?";
}
size_t dtype_bytes(DType d) { return d == DType::F32 ? 4 : d == DType::BF16 ? 2 : 1; }

namespace {

inline float bf16_bits_to_f32(uint16_t h) {
  uint32_t bits = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

inline uint16_t f32_to_bf16_bits(float f) {  // round to nearest even
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  if ((bits & 0x7F800000u) == 0x7F800000u) return uint16_t(bits >> 16);  // inf / NaN
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return uint16_t(bits >> 16);
}

}  // namespace

size_t Matrix::bytes() const {
  return f32.size() * 4 + bf16.size() * 2 + i8.size() + scale.size() * 4;
}

void Matrix::row_to_f32(int r, float* out) const {
  const size_t base = size_t(r) * cols;
  switch (dtype) {
    case DType::F32: std::memcpy(out, &f32[base], sizeof(float) * cols); break;
    case DType::BF16: for (int i = 0; i < cols; ++i) out[i] = bf16_bits_to_f32(bf16[base + i]); break;
    case DType::I8: for (int i = 0; i < cols; ++i) out[i] = scale[r] * float(i8[base + i]); break;
  }
}

Matrix make_matrix(const std::vector<float>& w, int rows, int cols, DType dtype) {
  if (w.size() != size_t(rows) * cols) throw std::runtime_error("make_matrix: size mismatch");
  Matrix m;
  m.dtype = dtype;
  m.rows = rows;
  m.cols = cols;
  switch (dtype) {
    case DType::F32:
      m.f32 = w;
      break;
    case DType::BF16:
      m.bf16.resize(w.size());
      for (size_t i = 0; i < w.size(); ++i) m.bf16[i] = f32_to_bf16_bits(w[i]);
      break;
    case DType::I8:
      // Symmetric per-row quantization: each row gets its own scale so that the
      // row's largest |weight| maps to 127. One outlier only hurts its own row.
      m.i8.resize(w.size());
      m.scale.resize(rows);
      for (int r = 0; r < rows; ++r) {
        const float* row = &w[size_t(r) * cols];
        float amax = 0.0f;
        for (int i = 0; i < cols; ++i) amax = std::max(amax, std::fabs(row[i]));
        const float s = amax > 0 ? amax / 127.0f : 1.0f;
        m.scale[r] = s;
        for (int i = 0; i < cols; ++i) {
          const float q = std::nearbyint(row[i] / s);
          m.i8[size_t(r) * cols + i] = int8_t(std::max(-127.0f, std::min(127.0f, q)));
        }
      }
      break;
  }
  return m;
}

Matrix make_matrix_bf16(std::vector<uint16_t> bits, int rows, int cols, DType dtype) {
  if (bits.size() != size_t(rows) * cols) throw std::runtime_error("make_matrix_bf16: size mismatch");
  if (dtype == DType::BF16) {
    Matrix m;
    m.dtype = DType::BF16;
    m.rows = rows;
    m.cols = cols;
    m.bf16 = std::move(bits);
    return m;
  }
  std::vector<float> w(bits.size());
  for (size_t i = 0; i < bits.size(); ++i) w[i] = bf16_bits_to_f32(bits[i]);
  bits.clear();
  bits.shrink_to_fit();
  return make_matrix(w, rows, cols, dtype);
}

// ----------------------------------------------------------- kernels ----
//
// Every kernel computes NT dot products at once (NT = 1..4 tokens) against ONE
// weight row: each weight is loaded/converted once and used NT times. During
// prefill that cuts weight traffic by up to 4x; during decode NT = 1.

namespace {

// Scalar: one weight at a time. Same blocking as the SIMD version.
template <DType D, int NT>
void dot_scalar(const Matrix& W, int o, const float* x, float* out) {
  const int n = W.cols;
  const size_t base = size_t(o) * n;
  float acc[NT] = {};
  for (int k = 0; k < n; ++k) {
    float w;
    if constexpr (D == DType::F32) w = W.f32[base + k];
    else if constexpr (D == DType::BF16) w = bf16_bits_to_f32(W.bf16[base + k]);
    else w = float(W.i8[base + k]);
    for (int t = 0; t < NT; ++t) acc[t] += w * x[size_t(t) * n + k];
  }
  const float s = (D == DType::I8) ? W.scale[o] : 1.0f;
  for (int t = 0; t < NT; ++t) out[t] = acc[t] * s;
}

#if IE_HAVE_AVX2
// Load 8 consecutive weights of row-major storage as 8 floats.
template <DType D>
inline __m256 load8(const Matrix& W, size_t idx) {
  if constexpr (D == DType::F32) {
    return _mm256_loadu_ps(&W.f32[idx]);
  } else if constexpr (D == DType::BF16) {
    // 8 x uint16 -> 8 x uint32, shift into the top half: that IS the float.
    __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&W.bf16[idx]));
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16));
  } else {
    // 8 x int8 -> 8 x int32 -> 8 x float (the row scale is applied at the end).
    __m128i q = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&W.i8[idx]));
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q));
  }
}

inline float hsum(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}

// AVX2 + FMA: 8 multiply-adds per instruction. Two accumulators per token
// (16 weights per iteration) keep the FMA units busy instead of waiting on
// each previous add to finish.
template <DType D, int NT>
void dot_avx2(const Matrix& W, int o, const float* x, float* out) {
  const int n = W.cols;
  const size_t base = size_t(o) * n;
  __m256 acc0[NT], acc1[NT];
  for (int t = 0; t < NT; ++t) acc0[t] = acc1[t] = _mm256_setzero_ps();
  int k = 0;
  for (; k + 16 <= n; k += 16) {
    const __m256 w0 = load8<D>(W, base + k);
    const __m256 w1 = load8<D>(W, base + k + 8);
    for (int t = 0; t < NT; ++t) {
      const float* xt = x + size_t(t) * n + k;
      acc0[t] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xt), acc0[t]);
      acc1[t] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xt + 8), acc1[t]);
    }
  }
  for (; k + 8 <= n; k += 8) {
    const __m256 w0 = load8<D>(W, base + k);
    for (int t = 0; t < NT; ++t)
      acc0[t] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x + size_t(t) * n + k), acc0[t]);
  }
  const float s = (D == DType::I8) ? W.scale[o] : 1.0f;
  for (int t = 0; t < NT; ++t) {
    float sum = hsum(_mm256_add_ps(acc0[t], acc1[t]));
    for (int kk = k; kk < n; ++kk) {  // leftover columns (none for this model)
      float w;
      if constexpr (D == DType::F32) w = W.f32[base + kk];
      else if constexpr (D == DType::BF16) w = bf16_bits_to_f32(W.bf16[base + kk]);
      else w = float(W.i8[base + kk]);
      sum += w * x[size_t(t) * n + kk];
    }
    out[t] = sum * s;
  }
}
#endif

using DotFn = void (*)(const Matrix&, int, const float*, float*);

template <DType D>
DotFn pick(Kernel k, int nt) {
#if IE_HAVE_AVX2
  if (k == Kernel::Avx2) {
    switch (nt) {
      case 1: return dot_avx2<D, 1>;
      case 2: return dot_avx2<D, 2>;
      case 3: return dot_avx2<D, 3>;
      default: return dot_avx2<D, 4>;
    }
  }
#endif
  (void)k;
  switch (nt) {
    case 1: return dot_scalar<D, 1>;
    case 2: return dot_scalar<D, 2>;
    case 3: return dot_scalar<D, 3>;
    default: return dot_scalar<D, 4>;
  }
}

DotFn pick(DType d, Kernel k, int nt) {
  switch (d) {
    case DType::F32: return pick<DType::F32>(k, nt);
    case DType::BF16: return pick<DType::BF16>(k, nt);
    case DType::I8: return pick<DType::I8>(k, nt);
  }
  return nullptr;
}

Kernel g_kernel = IE_HAVE_AVX2 ? Kernel::Avx2 : Kernel::Scalar;

// Rows [r0, r1) of y = x . W^T, blocked for cache reuse:
//   - 16 rows at a time, so that block of W stays in L1/L2 while
//   - tokens are processed 4 at a time against it.
// Each weight is fetched from memory once per call, however many tokens there are.
void matmul_rows(float* y, const float* x, const Matrix& W, const float* bias, int T, int r0,
                 int r1) {
  constexpr int kRowBlock = 16, kTokBlock = 4;
  const int n = W.cols, m = W.rows;
  DotFn dots[kTokBlock + 1] = {nullptr};
  for (int nt = 1; nt <= kTokBlock; ++nt) dots[nt] = pick(W.dtype, g_kernel, nt);

  for (int rb = r0; rb < r1; rb += kRowBlock) {
    const int re = std::min(rb + kRowBlock, r1);
    for (int t = 0; t < T; t += kTokBlock) {
      const int nt = std::min(kTokBlock, T - t);
      const float* xt = x + size_t(t) * n;
      for (int o = rb; o < re; ++o) {
        float out[kTokBlock];
        dots[nt](W, o, xt, out);
        const float b = bias ? bias[o] : 0.0f;
        for (int i = 0; i < nt; ++i) y[size_t(t + i) * m + o] = out[i] + b;
      }
    }
  }
}

}  // namespace

Kernel parse_kernel(const std::string& s) {
  if (s == "scalar") return Kernel::Scalar;
  if (s == "avx2") {
    if (!avx2_available()) throw std::runtime_error("this build has no AVX2 kernels (IE_AVX2=OFF)");
    return Kernel::Avx2;
  }
  throw std::runtime_error("unknown kernel '" + s + "' (use scalar or avx2)");
}
const char* kernel_name(Kernel k) { return k == Kernel::Avx2 ? "avx2" : "scalar"; }
bool avx2_available() { return IE_HAVE_AVX2 != 0; }
void set_kernel(Kernel k) { g_kernel = k; }
Kernel current_kernel() { return g_kernel; }

void matmul(float* y, const float* x, const Matrix& W, const float* bias, int T) {
  // Output rows are independent, so each thread takes a contiguous slice of them.
  thread_pool().parallel_for(W.rows, [&](int64_t r0, int64_t r1) {
    matmul_rows(y, x, W, bias, T, int(r0), int(r1));
  });
}

}  // namespace ie
