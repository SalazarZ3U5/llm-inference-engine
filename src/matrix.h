#pragma once
// Weight matrices in different storage formats, and the matrix multiply that
// dominates the forward pass.
//
// Storage formats (dtype):
//   F32   4 bytes per weight.
//   BF16  2 bytes. The model ships in BF16, so this is exact: converting a
//         BF16 weight to FP32 for the multiply loses nothing.
//   I8    1 byte + one FP32 scale per row. w ≈ scale[row] * q, q in [-127, 127].
//         Lossy, but halves memory traffic again.
//
// Decoding one token reads every weight once, so bytes per weight is roughly
// what sets the speed limit: FP32 -> BF16 -> INT8 is 4 -> 2 -> 1 byte.

#include <cstdint>
#include <string>
#include <vector>

namespace ie {

enum class DType { F32, BF16, I8 };
DType parse_dtype(const std::string& s);  // "f32" | "bf16" | "int8"
const char* dtype_name(DType d);
size_t dtype_bytes(DType d);

struct Matrix {
  DType dtype = DType::F32;
  int rows = 0, cols = 0;      // [out, in], row-major, like nn.Linear
  std::vector<float> f32;      // used when dtype == F32
  std::vector<uint16_t> bf16;  // used when dtype == BF16
  std::vector<int8_t> i8;      // used when dtype == I8
  std::vector<float> scale;    // I8 only: one per row

  size_t bytes() const;
  void row_to_f32(int r, float* out) const;  // dequantize one row (e.g. an embedding)
};

// Build a matrix of the given dtype from FP32 values ([rows, cols], row-major).
Matrix make_matrix(const std::vector<float>& w, int rows, int cols, DType dtype);
// Build directly from BF16 bits (avoids a temporary FP32 copy for BF16 storage).
Matrix make_matrix_bf16(std::vector<uint16_t> bits, int rows, int cols, DType dtype);

enum class Kernel { Scalar, Avx2 };
Kernel parse_kernel(const std::string& s);  // "scalar" | "avx2"
const char* kernel_name(Kernel k);
bool avx2_available();          // compiled in?
void set_kernel(Kernel k);      // default: AVX2 when compiled in
Kernel current_kernel();

// y[t][o] = sum_i W[o][i] * x[t][i] + bias[o]
//   x: [T, W.cols]   y: [T, W.rows]   bias: [W.rows] or nullptr
// Runs on the global thread pool, split across output rows.
void matmul(float* y, const float* x, const Matrix& W, const float* bias, int T);

}  // namespace ie
