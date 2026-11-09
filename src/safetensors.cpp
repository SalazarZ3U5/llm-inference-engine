#include "safetensors.h"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace ie {

uint64_t TensorInfo::num_elements() const {
  uint64_t n = 1;
  for (int64_t d : shape) n *= static_cast<uint64_t>(d);
  return n;
}

size_t dtype_size(const std::string& dtype) {
  if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
  if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") return 2;
  if (dtype == "I8" || dtype == "U8" || dtype == "BOOL" || dtype == "F8_E4M3" ||
      dtype == "F8_E5M2")
    return 1;
  throw std::runtime_error("unknown safetensors dtype: " + dtype);
}

SafeTensorsFile::SafeTensorsFile(const std::string& path) : path_(path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  file_size_ = static_cast<uint64_t>(f.tellg());
  f.seekg(0);

  if (file_size_ < 8) throw std::runtime_error("file too small to be safetensors: " + path);
  unsigned char len_bytes[8];
  f.read(reinterpret_cast<char*>(len_bytes), 8);
  uint64_t header_len = 0;
  for (int i = 7; i >= 0; --i) header_len = (header_len << 8) | len_bytes[i];  // little-endian
  if (header_len > file_size_ - 8)
    throw std::runtime_error("header length exceeds file size: " + path);

  std::string header(header_len, '\0');
  f.read(header.data(), static_cast<std::streamsize>(header_len));
  data_offset_ = 8 + header_len;
  const uint64_t data_size = file_size_ - data_offset_;

  nlohmann::json j = nlohmann::json::parse(header);
  for (auto it = j.begin(); it != j.end(); ++it) {
    if (it.key() == "__metadata__") {
      for (auto m = it->begin(); m != it->end(); ++m)
        metadata_[m.key()] = m->is_string() ? m->get<std::string>() : m->dump();
      continue;
    }
    TensorInfo t;
    t.name = it.key();
    t.dtype = it->at("dtype").get<std::string>();
    t.shape = it->at("shape").get<std::vector<int64_t>>();
    auto offs = it->at("data_offsets").get<std::vector<uint64_t>>();
    if (offs.size() != 2) throw std::runtime_error("bad data_offsets for " + t.name);
    t.begin = offs[0];
    t.end = offs[1];

    // Validate: offsets in range and consistent with dtype * shape.
    if (t.end < t.begin || t.end > data_size)
      throw std::runtime_error("data_offsets out of range for " + t.name);
    if (t.num_elements() * dtype_size(t.dtype) != t.num_bytes())
      throw std::runtime_error("size mismatch for " + t.name);

    tensors_.emplace(t.name, std::move(t));
  }
}

const TensorInfo& SafeTensorsFile::info(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::runtime_error("tensor not found: " + name);
  return it->second;
}

std::vector<uint8_t> SafeTensorsFile::read_bytes(const std::string& name) const {
  const TensorInfo& t = info(name);
  std::ifstream f(path_, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path_);
  f.seekg(static_cast<std::streamoff>(data_offset_ + t.begin));
  std::vector<uint8_t> buf(t.num_bytes());
  f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
  if (!f) throw std::runtime_error("short read for " + name);
  return buf;
}

}  // namespace ie
