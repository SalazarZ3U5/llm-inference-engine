#pragma once
// Minimal reader for the .safetensors format used by Hugging Face models.
//
// File layout:
//   [8 bytes]  little-endian uint64 N = size of the JSON header
//   [N bytes]  JSON header: { "<tensor name>": { "dtype": "BF16", "shape": [..],
//                                                "data_offsets": [begin, end] }, ...,
//                             "__metadata__": { ... } }
//   [rest]     raw tensor bytes; data_offsets are relative to the start of this region
//
// Spec: https://github.com/huggingface/safetensors

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ie {

struct TensorInfo {
  std::string name;
  std::string dtype;            // "F32", "F16", "BF16", "I8", ...
  std::vector<int64_t> shape;
  uint64_t begin = 0;           // byte offset into the data region
  uint64_t end = 0;

  uint64_t num_bytes() const { return end - begin; }
  uint64_t num_elements() const;
};

// Bytes per element for a safetensors dtype string; throws on unknown dtypes.
size_t dtype_size(const std::string& dtype);

class SafeTensorsFile {
 public:
  // Parses and validates the header. Tensor data is read lazily.
  explicit SafeTensorsFile(const std::string& path);

  const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }
  const std::map<std::string, std::string>& metadata() const { return metadata_; }
  const TensorInfo& info(const std::string& name) const;
  bool contains(const std::string& name) const { return tensors_.count(name) != 0; }

  // Reads one tensor's raw bytes from disk (no dtype conversion).
  std::vector<uint8_t> read_bytes(const std::string& name) const;

  uint64_t file_size() const { return file_size_; }
  uint64_t data_offset() const { return data_offset_; }

 private:
  std::string path_;
  uint64_t file_size_ = 0;
  uint64_t data_offset_ = 0;  // 8 + header size
  std::map<std::string, TensorInfo> tensors_;
  std::map<std::string, std::string> metadata_;
};

}  // namespace ie
