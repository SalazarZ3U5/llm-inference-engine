#pragma once
// All model weights, converted to FP32 and held in CPU memory.
//
// Every matrix is stored the way PyTorch's nn.Linear stores it: [out, in],
// row-major. So W[o * in + i] is the weight from input i to output o, and
// each output is the dot product of one contiguous row with the input.

#include <cstdint>
#include <cstring>
#include <vector>

#include "config.h"
#include "safetensors.h"

namespace ie {

struct LayerWeights {
  std::vector<float> input_norm;      // [hidden]            RMSNorm before attention
  std::vector<float> wq, bq;          // [q_dim, hidden], [q_dim]
  std::vector<float> wk, bk;          // [kv_dim, hidden], [kv_dim]
  std::vector<float> wv, bv;          // [kv_dim, hidden], [kv_dim]
  std::vector<float> wo;              // [hidden, q_dim]     no bias
  std::vector<float> post_attn_norm;  // [hidden]            RMSNorm before the MLP
  std::vector<float> w_gate, w_up;    // [intermediate, hidden]
  std::vector<float> w_down;          // [hidden, intermediate]
  // Biases are empty for models that have none (e.g. Llama).
};

struct Weights {
  std::vector<float> embed;           // [vocab, hidden]
  std::vector<LayerWeights> layers;
  std::vector<float> final_norm;      // [hidden]
  std::vector<float> lm_head;         // [vocab, hidden]; empty when tied to embed

  // The matrix used to turn the final hidden state into logits.
  const float* lm_head_ptr() const { return lm_head.empty() ? embed.data() : lm_head.data(); }
  size_t num_params() const;
};

// BF16 is the upper 16 bits of an IEEE FP32, so conversion is a shift.
inline float bf16_to_f32(uint16_t h) {
  uint32_t bits = uint32_t(h) << 16;
  float f;
  static_assert(sizeof(f) == sizeof(bits), "float must be 32-bit");
  std::memcpy(&f, &bits, 4);  // reinterpret the bits; a pointer cast would be undefined behaviour
  return f;
}

// Loads every tensor the config calls for, checks its shape, converts to FP32.
Weights load_weights(const SafeTensorsFile& file, const Config& cfg);

}  // namespace ie
