#pragma once
// All model weights, held in CPU memory.
//
// The big matrices are `Matrix` objects stored as F32, BF16 or INT8 (see
// matrix.h). Small vectors (norm weights, biases) are always FP32.
//
// Every matrix is stored the way PyTorch's nn.Linear stores it: [out, in],
// row-major. So W[o * in + i] is the weight from input i to output o, and
// each output is the dot product of one contiguous row with the input.

#include <cstdint>
#include <cstring>
#include <vector>

#include "config.h"
#include "matrix.h"
#include "safetensors.h"

namespace ie {

struct LayerWeights {
  std::vector<float> input_norm;      // [hidden]            RMSNorm before attention
  Matrix wq, wk, wv;                  // [q_dim, hidden], [kv_dim, hidden] x2
  std::vector<float> bq, bk, bv;      // [q_dim], [kv_dim] x2; empty if the model has none
  Matrix wo;                          // [hidden, q_dim]     no bias
  std::vector<float> post_attn_norm;  // [hidden]            RMSNorm before the MLP
  Matrix w_gate, w_up;                // [intermediate, hidden]
  Matrix w_down;                      // [hidden, intermediate]
};

struct Weights {
  DType dtype = DType::BF16;          // storage format of the matrices
  Matrix embed;                       // [vocab, hidden]
  std::vector<LayerWeights> layers;
  std::vector<float> final_norm;      // [hidden]
  Matrix lm_head;                     // [vocab, hidden]; empty when tied to embed

  // The matrix used to turn the final hidden state into logits.
  const Matrix& lm_head_matrix() const { return lm_head.rows ? lm_head : embed; }
  size_t num_params() const;
  size_t matrix_bytes() const;        // bytes of all matrices (what decode streams)
};

// BF16 is the upper 16 bits of an IEEE FP32, so conversion is a shift.
inline float bf16_to_f32(uint16_t h) {
  uint32_t bits = uint32_t(h) << 16;
  float f;
  static_assert(sizeof(f) == sizeof(bits), "float must be 32-bit");
  std::memcpy(&f, &bits, 4);  // reinterpret the bits; a pointer cast would be undefined behaviour
  return f;
}

// Loads every tensor the config calls for, checks its shape, and stores the
// matrices as `dtype`.
Weights load_weights(const SafeTensorsFile& file, const Config& cfg, DType dtype = DType::BF16);

}  // namespace ie
