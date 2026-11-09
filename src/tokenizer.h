#pragma once
// Byte-level BPE tokenizer, compatible with Hugging Face's tokenizer for
// Qwen2 / Qwen2.5 (the same scheme as GPT-2 and Llama 3, with a different
// pre-split pattern).
//
// Encoding text -> ids:
//   1. Cut out special tokens such as <|im_start|>; they map straight to their id.
//   2. Pre-split the rest into chunks (words, single digits, punctuation runs,
//      whitespace) with the model's regex. BPE never merges across chunks.
//   3. Map every byte of a chunk to a printable stand-in character (byte-level).
//   4. Apply BPE: repeatedly merge the adjacent pair with the best (lowest)
//      rank in merges.txt until no listed pair remains.
//   5. Look each resulting piece up in vocab.json.
//
// Decoding ids -> text reverses step 3 and concatenates the bytes.
//
// Limitation: Hugging Face applies Unicode NFC normalization first. Text typed
// on a keyboard is almost always already NFC, so this is skipped.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace ie {

class Tokenizer {
 public:
  // Reads the vocabulary, merge list and added (special) tokens from tokenizer.json.
  explicit Tokenizer(const std::string& model_dir);

  // allow_special = true: text like "<|im_end|>" becomes the special token.
  // allow_special = false: it is tokenized as ordinary characters.
  std::vector<int32_t> encode(const std::string& text, bool allow_special = true) const;

  // Raw bytes a token stands for. May be part of a multi-byte UTF-8 character.
  std::string token_bytes(int32_t id) const;
  std::string decode(const std::vector<int32_t>& ids) const;

  int32_t special_id(const std::string& text) const;  // throws if unknown
  bool is_special(int32_t id) const;
  int32_t vocab_size() const { return static_cast<int32_t>(id_to_token_.size()); }

  // Pre-split step only (exposed for testing): byte ranges of each chunk.
  std::vector<std::string> pre_split(const std::string& text) const;

 private:
  void bpe(const std::string& chunk, std::vector<int32_t>& out) const;

  std::unordered_map<std::string, int32_t> token_to_id_;  // byte-level string -> id
  std::vector<std::string> id_to_token_;                  // id -> byte-level string
  std::vector<std::string> id_to_bytes_;                  // id -> the raw bytes it stands for
  std::unordered_map<std::string, int32_t> merge_rank_;   // "a b" -> rank (0 = merged first)
  std::vector<std::pair<std::string, int32_t>> added_;    // special/added tokens, longest first
  std::vector<bool> special_;                              // id -> is an added token
  std::string byte_to_char_[256];                          // byte -> UTF-8 of its stand-in
  mutable std::unordered_map<std::string, std::vector<int32_t>> cache_;  // chunk -> ids
};

// Turns a stream of token ids into printable text. A token can end in the
// middle of a multi-byte UTF-8 character (common for emoji and non-Latin
// scripts), so incomplete bytes are held back until the character completes.
class StreamDecoder {
 public:
  explicit StreamDecoder(const Tokenizer& tok) : tok_(tok) {}
  std::string push(int32_t id);  // returns text that is now complete
  std::string flush();           // whatever is left, at the end

 private:
  const Tokenizer& tok_;
  std::string pending_;
};

}  // namespace ie
