#pragma once
// Reader for NumPy .npy files, used to load the reference activations that
// scripts/reference.py saves, plus a helper to compare two float arrays.
//
// File layout (format versions 1.0 / 2.0 / 3.0):
//   6 bytes   magic "\x93NUMPY"
//   2 bytes   major, minor version
//   2 or 4    little-endian header length (2 bytes in v1, 4 bytes in v2/v3)
//   N bytes   ASCII Python dict, e.g. {'descr': '<f4', 'fortran_order': False, 'shape': (5, 896), }
//   rest      raw array data, row-major (C order)

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ie {

struct NpyArray {
  std::string dtype;            // "<f4" (float32) or "<i8" (int64)
  std::vector<int64_t> shape;
  std::vector<float> f32;       // filled when dtype is "<f4"
  std::vector<int64_t> i64;     // filled when dtype is "<i8"

  size_t size() const;          // total element count (product of shape)
};

NpyArray load_npy(const std::string& path);

struct CompareResult {
  double max_abs_err = 0;   // largest |got - want|
  double max_abs_ref = 0;   // largest |want|, used to make the error relative
  double rel_err = 0;       // max_abs_err / max_abs_ref
  size_t worst_index = 0;   // flat index where the largest error occurred
};

// Compares n floats. rel_err is relative to the largest reference value, so
// one huge activation does not make tiny errors elsewhere look large.
CompareResult compare(const float* got, const float* want, size_t n);

}  // namespace ie
