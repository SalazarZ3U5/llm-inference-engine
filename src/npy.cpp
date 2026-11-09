#include "npy.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace ie {

size_t NpyArray::size() const {
  size_t n = 1;
  for (int64_t d : shape) n *= static_cast<size_t>(d);
  return n;
}

namespace {

// Returns the text value that follows `key` in the header dict,
// e.g. key "'descr':" -> "'<f4'" (still quoted) or "(5, 896)".
std::string dict_value(const std::string& header, const std::string& key) {
  size_t k = header.find(key);
  if (k == std::string::npos) throw std::runtime_error("npy header missing " + key);
  size_t start = k + key.size();
  while (start < header.size() && header[start] == ' ') ++start;
  size_t end;
  if (header[start] == '(') {
    end = header.find(')', start) + 1;
  } else if (header[start] == '\'') {
    end = header.find('\'', start + 1) + 1;
  } else {
    end = header.find(',', start);
  }
  return header.substr(start, end - start);
}

std::vector<int64_t> parse_shape(const std::string& tuple) {
  // "(5, 896)" or "(20,)" or "()" for a scalar
  std::vector<int64_t> shape;
  std::string num;
  for (char c : tuple) {
    if (c >= '0' && c <= '9') {
      num += c;
    } else if (!num.empty()) {
      shape.push_back(std::stoll(num));
      num.clear();
    }
  }
  return shape;
}

}  // namespace

NpyArray load_npy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);

  char magic[6];
  f.read(magic, 6);
  if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0)
    throw std::runtime_error("not a .npy file: " + path);

  unsigned char ver[2];
  f.read(reinterpret_cast<char*>(ver), 2);
  uint32_t header_len = 0;
  if (ver[0] == 1) {
    unsigned char b[2];
    f.read(reinterpret_cast<char*>(b), 2);
    header_len = b[0] | (b[1] << 8);
  } else {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    header_len = b[0] | (b[1] << 8) | (b[2] << 16) | (uint32_t(b[3]) << 24);
  }
  std::string header(header_len, '\0');
  f.read(header.data(), header_len);

  NpyArray arr;
  std::string descr = dict_value(header, "'descr':");
  arr.dtype = descr.substr(1, descr.size() - 2);  // strip quotes
  if (dict_value(header, "'fortran_order':").find("True") != std::string::npos)
    throw std::runtime_error("fortran_order arrays are not supported: " + path);
  arr.shape = parse_shape(dict_value(header, "'shape':"));

  const size_t n = arr.size();
  if (arr.dtype == "<f4") {
    arr.f32.resize(n);
    f.read(reinterpret_cast<char*>(arr.f32.data()), static_cast<std::streamsize>(n * 4));
  } else if (arr.dtype == "<i8") {
    arr.i64.resize(n);
    f.read(reinterpret_cast<char*>(arr.i64.data()), static_cast<std::streamsize>(n * 8));
  } else {
    throw std::runtime_error("unsupported npy dtype " + arr.dtype + " in " + path);
  }
  if (!f) throw std::runtime_error("short read in " + path);
  return arr;
}

CompareResult compare(const float* got, const float* want, size_t n) {
  CompareResult r;
  for (size_t i = 0; i < n; ++i) {
    double err = std::fabs(double(got[i]) - double(want[i]));
    if (!(err <= r.max_abs_err)) {  // also catches NaN
      r.max_abs_err = err;
      r.worst_index = i;
    }
    r.max_abs_ref = std::fmax(r.max_abs_ref, std::fabs(double(want[i])));
  }
  r.rel_err = r.max_abs_ref > 0 ? r.max_abs_err / r.max_abs_ref : r.max_abs_err;
  return r;
}

}  // namespace ie
