#include "tokenizer.h"

#include <algorithm>
#include <climits>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "unicode_tables.h"

namespace ie {

namespace {

// ---------------------------------------------------------------- UTF-8 ----

std::string utf8_encode(uint32_t cp) {
  std::string s;
  if (cp < 0x80) {
    s += char(cp);
  } else if (cp < 0x800) {
    s += char(0xC0 | (cp >> 6));
    s += char(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    s += char(0xE0 | (cp >> 12));
    s += char(0x80 | ((cp >> 6) & 0x3F));
    s += char(0x80 | (cp & 0x3F));
  } else {
    s += char(0xF0 | (cp >> 18));
    s += char(0x80 | ((cp >> 12) & 0x3F));
    s += char(0x80 | ((cp >> 6) & 0x3F));
    s += char(0x80 | (cp & 0x3F));
  }
  return s;
}

// Decodes UTF-8 into code points, remembering where each one starts so chunks
// can be cut from the original bytes. Invalid bytes become U+FFFD, one byte each.
void utf8_decode(const std::string& s, std::vector<uint32_t>& cps, std::vector<size_t>& offs) {
  cps.clear();
  offs.clear();
  size_t i = 0;
  while (i < s.size()) {
    const unsigned char c = s[i];
    uint32_t cp = 0xFFFD;
    size_t len = 1;
    if (c < 0x80) {
      cp = c;
    } else if ((c >> 5) == 0x6) {
      len = 2;
    } else if ((c >> 4) == 0xE) {
      len = 3;
    } else if ((c >> 3) == 0x1E) {
      len = 4;
    }
    if (len > 1) {
      bool ok = i + len <= s.size();
      for (size_t k = 1; ok && k < len; ++k) ok = (static_cast<unsigned char>(s[i + k]) >> 6) == 0x2;
      if (ok) {
        cp = c & (0xFF >> (len + 1));
        for (size_t k = 1; k < len; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
      } else {
        len = 1;  // invalid sequence: consume one byte
      }
    }
    cps.push_back(cp);
    offs.push_back(i);
    i += len;
  }
  offs.push_back(s.size());
}

// ------------------------------------------------- character classes ----

template <size_t N>
bool in_ranges(uint32_t cp, const unicode::CodepointRange (&table)[N]) {
  // Binary search for the last range starting at or before cp.
  size_t lo = 0, hi = N;
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (table[mid].first <= cp) lo = mid + 1; else hi = mid;
  }
  return lo > 0 && cp <= table[lo - 1].last;
}

bool is_letter(uint32_t cp) {  // \p{L}
  if (cp < 0x80) return (cp | 0x20) >= 'a' && (cp | 0x20) <= 'z';
  return in_ranges(cp, unicode::kLetters);
}
bool is_number(uint32_t cp) {  // \p{N}
  if (cp < 0x80) return cp >= '0' && cp <= '9';
  return in_ranges(cp, unicode::kNumbers);
}
bool is_space(uint32_t cp) {   // \s (Unicode White_Space)
  return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
         (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
         cp == 0x205F || cp == 0x3000;
}
bool is_newline(uint32_t cp) { return cp == '\r' || cp == '\n'; }
bool is_other(uint32_t cp) { return !is_space(cp) && !is_letter(cp) && !is_number(cp); }
uint32_t ascii_lower(uint32_t cp) { return (cp >= 'A' && cp <= 'Z') ? cp + 32 : cp; }

// ------------------------------------------------------- pre-split ----
//
// Hand-written matcher for Qwen2's pre-tokenizer regex:
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)      1. English contractions
//   |[^\r\n\p{L}\p{N}]?\p{L}+         2. a word, optionally with one leading non-letter (" hello")
//   |\p{N}                            3. a single digit
//   | ?[^\s\p{L}\p{N}]+[\r\n]*        4. a punctuation/symbol run, optional leading space
//   |\s*[\r\n]+                       5. whitespace ending in newline(s)
//   |\s+(?!\S)                        6. whitespace not followed by a non-space
//   |\s+                              7. any other whitespace
//
// Like a regex engine, at each position the first alternative that matches wins.
// Returns the end index (exclusive) of the chunk starting at i.
size_t match_chunk(const std::vector<uint32_t>& cp, size_t i) {
  const size_t n = cp.size();
  const uint32_t c = cp[i];

  // 1. contractions
  if (c == '\'' && i + 1 < n) {
    const uint32_t a = ascii_lower(cp[i + 1]);
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return i + 2;
    if (i + 2 < n) {
      const uint32_t b = ascii_lower(cp[i + 2]);
      if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return i + 3;
    }
  }

  // 2. [^\r\n\p{L}\p{N}]?\p{L}+
  size_t start = n;
  if (is_letter(c)) {
    start = i;
  } else if (!is_newline(c) && !is_number(c) && i + 1 < n && is_letter(cp[i + 1])) {
    start = i + 1;
  }
  if (start != n) {
    size_t j = start + 1;
    while (j < n && is_letter(cp[j])) ++j;
    return j;
  }

  // 3. \p{N}: digits are always split one by one
  if (is_number(c)) return i + 1;

  // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
  size_t s = (c == ' ' && i + 1 < n && is_other(cp[i + 1])) ? i + 1 : i;
  if (is_other(cp[s])) {
    size_t j = s + 1;
    while (j < n && is_other(cp[j])) ++j;
    while (j < n && is_newline(cp[j])) ++j;
    return j;
  }

  // 5-7. whitespace runs
  if (is_space(c)) {
    size_t j = i;
    size_t last_newline = n;
    while (j < n && is_space(cp[j])) {
      if (is_newline(cp[j])) last_newline = j;
      ++j;
    }
    if (last_newline != n) return last_newline + 1;  // 5. up to and including the last newline
    if (j == n) return j;                             // 6. trailing whitespace: take it all
    if (j - i >= 2) return j - 1;                     // 6. leave one space to attach to the next word
    return j;                                         // 7. a single space before a non-space
  }
  return i + 1;  // unreachable for valid input; never stall
}

}  // namespace

// ----------------------------------------------------------- loading ----

Tokenizer::Tokenizer(const std::string& model_dir) {
  // GPT-2 byte-level mapping: printable bytes stand for themselves; the other
  // 68 bytes (control chars, space, ...) are shifted to code points 256+.
  // This way every byte has a visible character and a space shows up as "Ġ".
  std::vector<int> bs;
  for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
  for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
  for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);
  std::unordered_map<uint32_t, uint8_t> char_to_byte;
  int extra = 0;
  for (int b = 0; b < 256; ++b) {
    const bool printable = std::find(bs.begin(), bs.end(), b) != bs.end();
    const uint32_t ch = printable ? uint32_t(b) : uint32_t(256 + extra++);
    byte_to_char_[b] = utf8_encode(ch);
    char_to_byte[ch] = uint8_t(b);
  }

  const std::string path = (std::filesystem::path(model_dir) / "tokenizer.json").string();
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  const nlohmann::json j = nlohmann::json::parse(f);
  const nlohmann::json& model = j.at("model");
  if (model.at("type").get<std::string>() != "BPE")
    throw std::runtime_error("only BPE tokenizers are supported");

  // Vocabulary
  int32_t max_id = -1;
  for (auto it = model.at("vocab").begin(); it != model.at("vocab").end(); ++it) {
    const int32_t id = it.value().get<int32_t>();
    token_to_id_[it.key()] = id;
    max_id = std::max(max_id, id);
  }
  for (const auto& a : j.at("added_tokens")) max_id = std::max(max_id, a.at("id").get<int32_t>());
  id_to_token_.resize(size_t(max_id) + 1);
  for (const auto& [tok, id] : token_to_id_) id_to_token_[id] = tok;

  // Merges, in priority order. Stored as "a b" strings or ["a", "b"] pairs
  // depending on the tokenizers version that saved the file.
  int32_t rank = 0;
  for (const auto& m : model.at("merges")) {
    std::string key = m.is_string() ? m.get<std::string>()
                                    : m.at(0).get<std::string>() + " " + m.at(1).get<std::string>();
    merge_rank_.emplace(std::move(key), rank++);
  }

  // Added tokens (<|im_start|>, <|endoftext|>, ...)
  special_.assign(id_to_token_.size(), false);
  for (const auto& a : j.at("added_tokens")) {
    const std::string content = a.at("content").get<std::string>();
    const int32_t id = a.at("id").get<int32_t>();
    added_.emplace_back(content, id);
    id_to_token_[id] = content;
    special_[id] = true;
  }
  // Longest first, so a longer token wins when one is a prefix of another.
  std::sort(added_.begin(), added_.end(),
            [](const auto& x, const auto& y) { return x.first.size() > y.first.size(); });

  // Precompute the raw bytes behind every id, for fast decoding.
  id_to_bytes_.resize(id_to_token_.size());
  std::vector<uint32_t> cps;
  std::vector<size_t> offs;
  for (size_t id = 0; id < id_to_token_.size(); ++id) {
    if (special_[id]) {
      id_to_bytes_[id] = id_to_token_[id];
      continue;
    }
    utf8_decode(id_to_token_[id], cps, offs);
    std::string bytes;
    for (uint32_t ch : cps) {
      auto it = char_to_byte.find(ch);
      if (it != char_to_byte.end()) bytes += char(it->second);
    }
    id_to_bytes_[id] = std::move(bytes);
  }
}

// ---------------------------------------------------------- encoding ----

std::vector<std::string> Tokenizer::pre_split(const std::string& text) const {
  std::vector<uint32_t> cps;
  std::vector<size_t> offs;
  utf8_decode(text, cps, offs);
  std::vector<std::string> chunks;
  for (size_t i = 0; i < cps.size();) {
    const size_t j = match_chunk(cps, i);
    chunks.push_back(text.substr(offs[i], offs[j] - offs[i]));
    i = j;
  }
  return chunks;
}

void Tokenizer::bpe(const std::string& chunk, std::vector<int32_t>& out) const {
  auto cached = cache_.find(chunk);
  if (cached != cache_.end()) {
    out.insert(out.end(), cached->second.begin(), cached->second.end());
    return;
  }

  // Start with one symbol per byte, written as its stand-in character.
  std::vector<std::string> sym;
  for (unsigned char b : chunk) sym.push_back(byte_to_char_[b]);

  // Repeatedly merge the adjacent pair that was learned earliest in training.
  while (sym.size() > 1) {
    int32_t best = INT32_MAX;
    size_t best_i = 0;
    for (size_t i = 0; i + 1 < sym.size(); ++i) {
      auto it = merge_rank_.find(sym[i] + " " + sym[i + 1]);
      if (it != merge_rank_.end() && it->second < best) {
        best = it->second;
        best_i = i;
      }
    }
    if (best == INT32_MAX) break;  // no listed pair left

    // Merge every occurrence of that pair, left to right.
    const std::string a = sym[best_i], b = sym[best_i + 1];
    std::vector<std::string> merged;
    for (size_t i = 0; i < sym.size();) {
      if (i + 1 < sym.size() && sym[i] == a && sym[i + 1] == b) {
        merged.push_back(a + b);
        i += 2;
      } else {
        merged.push_back(sym[i]);
        i += 1;
      }
    }
    sym.swap(merged);
  }

  std::vector<int32_t> ids;
  for (const std::string& s : sym) {
    auto it = token_to_id_.find(s);
    if (it == token_to_id_.end()) throw std::runtime_error("BPE produced a piece not in the vocabulary");
    ids.push_back(it->second);
  }
  out.insert(out.end(), ids.begin(), ids.end());
  cache_.emplace(chunk, std::move(ids));
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool allow_special) const {
  std::vector<int32_t> out;
  auto encode_plain = [&](size_t from, size_t to) {
    if (from >= to) return;
    for (const std::string& chunk : pre_split(text.substr(from, to - from))) bpe(chunk, out);
  };

  size_t seg_start = 0;
  if (allow_special) {
    for (size_t i = 0; i < text.size();) {
      bool matched = false;
      if (text[i] == '<') {  // every added token starts with '<'; skip the scan otherwise
        for (const auto& [content, id] : added_) {
          if (text.compare(i, content.size(), content) == 0) {
            encode_plain(seg_start, i);
            out.push_back(id);
            i += content.size();
            seg_start = i;
            matched = true;
            break;
          }
        }
      }
      if (!matched) ++i;
    }
  }
  encode_plain(seg_start, text.size());
  return out;
}

// ---------------------------------------------------------- decoding ----

std::string Tokenizer::token_bytes(int32_t id) const {
  if (id < 0 || size_t(id) >= id_to_bytes_.size()) return {};
  return id_to_bytes_[id];
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
  std::string s;
  for (int32_t id : ids) s += token_bytes(id);
  return s;
}

int32_t Tokenizer::special_id(const std::string& text) const {
  for (const auto& [content, id] : added_)
    if (content == text) return id;
  throw std::runtime_error("unknown special token " + text);
}

bool Tokenizer::is_special(int32_t id) const {
  return id >= 0 && size_t(id) < special_.size() && special_[id];
}

std::string StreamDecoder::push(int32_t id) {
  pending_ += tok_.token_bytes(id);
  // Find where the last (possibly incomplete) UTF-8 character starts.
  size_t n = pending_.size();
  size_t lead = n;
  for (size_t back = 1; back <= 4 && back <= n; ++back) {
    const unsigned char c = pending_[n - back];
    if ((c >> 6) != 0x2) {  // not a continuation byte
      lead = n - back;
      break;
    }
  }
  size_t ready = n;
  if (lead < n) {
    const unsigned char c = pending_[lead];
    const size_t need = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
    if (n - lead < need) ready = lead;  // hold back the incomplete character
  }
  std::string out = pending_.substr(0, ready);
  pending_.erase(0, ready);
  return out;
}

std::string StreamDecoder::flush() {
  std::string out;
  out.swap(pending_);
  return out;
}

}  // namespace ie
