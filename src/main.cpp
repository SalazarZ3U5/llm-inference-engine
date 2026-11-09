// ie: command-line entry point for the inference engine.
//
//   ie generate <model_dir> "<prompt>" [options]   continue a prompt (add --chat to ask as a user)
//   ie chat     <model_dir> [options]              interactive multi-turn chat
//   ie tokenize <model_dir> "<text>"               show how text splits into tokens
//   ie verify   <model_dir> <ref_dir>              check every stage + generation against reference
//   ie test-tokenizer <model_dir> <cases.json>     check the tokenizer against Hugging Face's
//   ie inspect  <model.safetensors>                list every tensor in a weights file
//   ie gpu-check                                   GPU info + vector-add test + bandwidth test
//
// Sampling options: --temp T  --top-k K  --top-p P  --rep R  --seed S  --greedy
//   (defaults come from the model's generation_config.json)
// Other options:    --max-tokens N (default 256)  --ctx N (KV cache size, default 4096)
//                   --system "<text>" (chat)  --chat (generate)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "config.h"
#include "model.h"
#include "npy.h"
#include "safetensors.h"
#include "sampler.h"
#include "tokenizer.h"
#include "weights.h"

#ifdef IE_WITH_CUDA
#include "cuda/device_check.h"
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#ifdef _MSC_VER
#pragma comment(lib, "shell32.lib")
#endif
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

const char* kDefaultSystem = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

void usage() {
  std::printf(
      "usage:\n"
      "  ie generate <model_dir> \"<prompt>\" [--chat] [options]\n"
      "  ie chat     <model_dir> [--system \"<text>\"] [options]\n"
      "  ie tokenize <model_dir> \"<text>\"\n"
      "  ie verify   <model_dir> <reference_dir>\n"
      "  ie test-tokenizer <model_dir> <cases.json>\n"
      "  ie inspect  <model.safetensors>\n"
      "  ie gpu-check\n"
      "\n"
      "options: --max-tokens N  --ctx N  --temp T  --top-k K  --top-p P  --rep R  --seed S  --greedy\n");
}

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// ------------------------------------------------- text input (UTF-8) ----
//
// The engine works in UTF-8 everywhere. On Windows, argv and console input
// arrive in the legacy code page by default, which garbles Hindi, Chinese,
// emoji and so on, so read them as UTF-16 and convert.

#ifdef _WIN32
std::string wide_to_utf8(const wchar_t* w, int len) {
  const int bytes = WideCharToMultiByte(CP_UTF8, 0, w, len, nullptr, 0, nullptr, nullptr);
  std::string s(bytes > 0 ? bytes : 0, '\0');
  if (bytes > 0) WideCharToMultiByte(CP_UTF8, 0, w, len, s.data(), bytes, nullptr, nullptr);
  return s;
}
#endif

std::vector<std::string> utf8_args(int argc, char** argv) {
  std::vector<std::string> args;
#ifdef _WIN32
  int n = 0;
  if (LPWSTR* w = CommandLineToArgvW(GetCommandLineW(), &n)) {
    for (int i = 0; i < n; ++i) args.push_back(wide_to_utf8(w[i], int(wcslen(w[i]))));
    LocalFree(w);
    return args;
  }
#endif
  for (int i = 0; i < argc; ++i) args.push_back(argv[i]);
  return args;
}

bool read_line(std::string& line) {
#ifdef _WIN32
  HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
  DWORD mode = 0;
  if (GetConsoleMode(in, &mode)) {  // an interactive console, not a pipe or file
    std::wstring w;
    wchar_t buf[512];
    DWORD got = 0;
    while (w.empty() || w.back() != L'\n') {
      if (!ReadConsoleW(in, buf, 512, &got, nullptr) || got == 0) {
        if (w.empty()) return false;
        break;
      }
      w.append(buf, got);
    }
    while (!w.empty() && (w.back() == L'\n' || w.back() == L'\r')) w.pop_back();
    if (w == L"\x1a") return false;  // Ctrl+Z then Enter: end of input
    line = wide_to_utf8(w.data(), int(w.size()));
    return true;
  }
#endif
  return bool(std::getline(std::cin, line));
}

// ------------------------------------------------------------- options ----

struct Options {
  std::vector<std::string> positional;
  // Sampling overrides; anything not given comes from generation_config.json.
  std::optional<float> temperature, top_p, repetition_penalty;
  std::optional<int> top_k;
  uint64_t seed = 0;
  int max_tokens = 256;
  int ctx = 4096;
  bool chat = false;
  std::string system = kDefaultSystem;
};

Options parse_options(const std::vector<std::string>& args, size_t first) {
  Options o;
  for (size_t i = first; i < args.size(); ++i) {
    const std::string a = args[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= args.size()) throw std::runtime_error("missing value after " + a);
      return args[++i];
    };
    if (a == "--temp") o.temperature = std::stof(next());
    else if (a == "--top-k") o.top_k = std::stoi(next());
    else if (a == "--top-p") o.top_p = std::stof(next());
    else if (a == "--rep") o.repetition_penalty = std::stof(next());
    else if (a == "--seed") o.seed = std::stoull(next());
    else if (a == "--greedy") o.temperature = 0.0f;
    else if (a == "--max-tokens") o.max_tokens = std::stoi(next());
    else if (a == "--ctx") o.ctx = std::stoi(next());
    else if (a == "--chat") o.chat = true;
    else if (a == "--system") o.system = next();
    else if (a.rfind("--", 0) == 0) throw std::runtime_error("unknown option " + a);
    else o.positional.push_back(a);
  }
  return o;
}

ie::SamplingParams sampling_for(const Options& o, const std::string& model_dir) {
  ie::SamplingParams p = ie::load_generation_config(model_dir);
  if (o.temperature) p.temperature = *o.temperature;
  if (o.top_k) p.top_k = *o.top_k;
  if (o.top_p) p.top_p = *o.top_p;
  if (o.repetition_penalty) p.repetition_penalty = *o.repetition_penalty;
  p.seed = o.seed;
  if (p.temperature <= 0)
    std::fprintf(stderr, "[sampling: greedy, repetition penalty %.2f]\n", p.repetition_penalty);
  else
    std::fprintf(stderr, "[sampling: temperature %.2f, top-k %d, top-p %.2f, repetition penalty %.2f]\n",
                 p.temperature, p.top_k, p.top_p, p.repetition_penalty);
  return p;
}

// ----------------------------------------------------------- loading ----

std::unique_ptr<ie::Model> load_model(const fs::path& model_dir) {
  auto t0 = Clock::now();
  ie::Config cfg = ie::load_config((model_dir / "config.json").string());
  ie::SafeTensorsFile st((model_dir / "model.safetensors").string());
  auto model = std::make_unique<ie::Model>(cfg, ie::load_weights(st, cfg));
  std::fprintf(stderr, "[loaded %.1fM parameters in %.1f s]\n",
               model->weights().num_params() / 1e6, seconds_since(t0));
  return model;
}

// --------------------------------------------------------- generation ----

struct GenResult {
  int prompt_tokens = 0, new_tokens = 0;
  double prefill_s = 0, decode_s = 0;
  bool hit_stop = false;
};

// Feeds `prompt` after whatever is already in `cache`, then samples up to
// `max_new` tokens, printing text as it is produced. Every generated token
// except a stop token is run through the model, so the cache always holds the
// full conversation. `history` collects all token ids (for the repetition penalty).
GenResult generate(const ie::Model& model, ie::KVCache& cache, const ie::Tokenizer& tok,
                   ie::Sampler& sampler, std::vector<int32_t>& history,
                   const std::vector<int32_t>& prompt, int max_new,
                   const std::vector<int32_t>& stop_ids) {
  GenResult r;
  const int V = model.config().vocab_size;
  r.prompt_tokens = static_cast<int>(prompt.size());

  auto t0 = Clock::now();
  std::vector<float> logits = model.forward(prompt, cache);  // prefill
  r.prefill_s = seconds_since(t0);
  history.insert(history.end(), prompt.begin(), prompt.end());

  ie::StreamDecoder out(tok);
  auto t1 = Clock::now();
  for (int i = 0; i < max_new; ++i) {
    const int32_t next = sampler.sample(logits.data(), V, history);
    if (std::find(stop_ids.begin(), stop_ids.end(), next) != stop_ids.end()) {
      r.hit_stop = true;
      break;
    }
    history.push_back(next);
    ++r.new_tokens;
    std::fputs(out.push(next).c_str(), stdout);
    std::fflush(stdout);
    if (cache.len >= cache.max_tokens) {
      std::fputs("\n[context full]", stdout);
      break;
    }
    logits = model.forward({next}, cache);  // decode step: one token
  }
  std::fputs(out.flush().c_str(), stdout);
  r.decode_s = seconds_since(t1);
  return r;
}

void print_stats(const GenResult& r) {
  std::fprintf(stderr, "\n[prompt %d tokens in %.2f s (%.1f tok/s) | generated %d tokens in %.2f s (%.2f tok/s)]\n",
               r.prompt_tokens, r.prefill_s, r.prompt_tokens / std::max(r.prefill_s, 1e-9),
               r.new_tokens, r.decode_s, r.new_tokens / std::max(r.decode_s, 1e-9));
}

std::string chat_turn(const std::string& system, const std::string& user, bool first) {
  // Qwen2.5's ChatML format (see chat_template in tokenizer_config.json).
  std::string s;
  if (first) s += "<|im_start|>system\n" + system + "<|im_end|>\n";
  else s += "<|im_end|>\n";  // close the previous assistant reply
  s += "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
  return s;
}

int cmd_generate(const Options& o) {
  if (o.positional.size() < 2) { usage(); return 1; }
  const fs::path model_dir = o.positional[0];
  auto model = load_model(model_dir);
  ie::Tokenizer tok(model_dir.string());
  ie::Sampler sampler(sampling_for(o, model_dir.string()));

  std::string text = o.chat ? chat_turn(o.system, o.positional[1], true) : o.positional[1];
  std::vector<int32_t> prompt = tok.encode(text);
  ie::KVCache cache = model->make_cache(std::min<int>(o.ctx, int(prompt.size()) + o.max_tokens + 1));
  std::vector<int32_t> history;
  std::vector<int32_t> stops = {tok.special_id("<|im_end|>"), tok.special_id("<|endoftext|>")};

  if (!o.chat) std::fputs(o.positional[1].c_str(), stdout);
  GenResult r = generate(*model, cache, tok, sampler, history, prompt, o.max_tokens, stops);
  std::fputs("\n", stdout);
  print_stats(r);
  std::fprintf(stderr, "[seed %llu]\n", static_cast<unsigned long long>(sampler.seed()));
  return 0;
}

int cmd_chat(const Options& o) {
  if (o.positional.empty()) { usage(); return 1; }
  const fs::path model_dir = o.positional[0];
  auto model = load_model(model_dir);
  ie::Tokenizer tok(model_dir.string());
  ie::Sampler sampler(sampling_for(o, model_dir.string()));
  ie::KVCache cache = model->make_cache(o.ctx);
  std::fprintf(stderr, "[KV cache: %d tokens, %.0f MiB]\n", o.ctx, cache.bytes() / 1048576.0);
  std::vector<int32_t> stops = {tok.special_id("<|im_end|>"), tok.special_id("<|endoftext|>")};
  std::vector<int32_t> history;
  bool first = true;

  std::printf("Chat with the model. Commands: /reset (new conversation), /exit\n");
  std::string line;
  while (true) {
    std::printf("\n> ");
    std::fflush(stdout);
    if (!read_line(line)) break;
    if (line == "/exit" || line == "/quit") break;
    if (line == "/reset") {
      cache.clear();
      history.clear();
      first = true;
      std::printf("[conversation cleared]\n");
      continue;
    }
    if (line.empty()) continue;

    std::vector<int32_t> prompt = tok.encode(chat_turn(o.system, line, first));
    if (cache.len + int(prompt.size()) + o.max_tokens + 1 > cache.max_tokens) {
      std::printf("[context full: starting a new conversation]\n");
      cache.clear();
      history.clear();
      prompt = tok.encode(chat_turn(o.system, line, true));
    }
    first = false;
    std::printf("\n");
    GenResult r = generate(*model, cache, tok, sampler, history, prompt, o.max_tokens, stops);
    std::printf("\n");
    print_stats(r);
  }
  return 0;
}

bool valid_utf8(const std::string& s) {
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = s[i];
    const size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
    if (len == 0 || i + len > s.size()) return false;
    for (size_t k = 1; k < len; ++k)
      if ((static_cast<unsigned char>(s[i + k]) >> 6) != 0x2) return false;
    i += len;
  }
  return true;
}

int cmd_tokenize(const Options& o) {
  if (o.positional.size() < 2) { usage(); return 1; }
  ie::Tokenizer tok(o.positional[0]);
  std::vector<int32_t> ids = tok.encode(o.positional[1]);
  std::printf("%zu tokens\n", ids.size());
  for (int32_t id : ids) {
    // A token can hold part of a multi-byte character; show such bytes as \xNN.
    const std::string b = tok.token_bytes(id);
    const bool whole_chars = valid_utf8(b);
    std::string shown;
    for (unsigned char c : b) {
      char hex[8];
      if (c == '\n') shown += "\\n";
      else if (c == '\t') shown += "\\t";
      else if (c == '\r') shown += "\\r";
      else if (c >= 0x80 && !whole_chars) { std::snprintf(hex, sizeof(hex), "\\x%02X", c); shown += hex; }
      else shown += char(c);
    }
    std::printf("%8d  \"%s\"\n", id, shown.c_str());
  }
  return 0;
}

int cmd_test_tokenizer(const Options& o) {
  if (o.positional.size() < 2) { usage(); return 1; }
  ie::Tokenizer tok(o.positional[0]);
  std::ifstream f(fs::path(o.positional[1]));
  if (!f) throw std::runtime_error("cannot open " + o.positional[1]);
  nlohmann::json cases = nlohmann::json::parse(f);

  int failures = 0, total_tokens = 0;
  auto t0 = Clock::now();
  for (const auto& c : cases) {
    const std::string text = c.at("text").get<std::string>();
    const std::vector<int32_t> want = c.at("ids").get<std::vector<int32_t>>();
    const std::vector<int32_t> got = tok.encode(text);
    total_tokens += int(want.size());
    const bool ids_ok = got == want;
    const bool roundtrip_ok = tok.decode(got) == text;
    if (ids_ok && roundtrip_ok) continue;
    if (++failures <= 10) {
      std::printf("FAIL %s: %s\n", ids_ok ? "decode" : "encode", c.at("text").dump().c_str());
      std::printf("  want:");
      for (int32_t id : want) std::printf(" %d", id);
      std::printf("\n  got: ");
      for (int32_t id : got) std::printf(" %d", id);
      std::printf("\n");
    }
  }
  std::printf("%s: %zu cases (%d tokens), %d failed, %.0f ms\n", failures ? "FAILED" : "ALL PASSED",
              cases.size(), total_tokens, failures, seconds_since(t0) * 1e3);
  return failures ? 1 : 0;
}

// ------------------------------------------------------------- verify ----

// Error budget per stage. The embedding is a pure copy of converted weights,
// so it must match exactly. Everything else involves sums whose rounding
// depends on the order of additions, which differs from PyTorch's.
//
// Why 1e-3 and not tighter: token 0 carries a value of ~1,574 through layers
// 2-20. FP32 rounding on numbers that size is ~1e-4 in absolute terms. At
// layer 21 the model cancels that value out, the largest remaining value drops
// to ~50, and the carried-over rounding shows up as ~1e-4 *relative* error.
// That is expected arithmetic, not a bug. Real bugs (missing bias, wrong RoPE
// pairing, wrong GQA mapping) produce errors of 1e-2 to 1 and still fail.
double tolerance(const std::string& stage) { return stage == "embed" ? 0.0 : 1e-3; }

int cmd_verify(const Options& o) {
  if (o.positional.size() < 2) { usage(); return 1; }
  const fs::path model_dir = o.positional[0], ref_dir = o.positional[1];
  auto model = load_model(model_dir);
  const ie::Config& cfg = model->config();
  ie::Tokenizer tok(model_dir.string());

  ie::NpyArray ids_npy = ie::load_npy((ref_dir / "input_ids.npy").string());
  std::vector<int32_t> ids;
  for (int64_t id : ids_npy.i64) ids.push_back(static_cast<int32_t>(id));

  int failures = 0, checked = 0;
  auto report = [&](const char* name, bool ok, const std::string& detail) {
    ++checked;
    if (!ok) ++failures;
    std::printf("%-12s %-44s %s\n", name, detail.c_str(), ok ? "PASS" : "FAIL");
  };

  // --- 1. every stage of one full forward pass ---
  std::printf("\n%-12s %14s %14s %12s  %s\n", "stage", "max_abs_err", "max_abs_ref", "rel_err",
              "result");
  auto check = [&](const std::string& stage, const float* data, int rows, int cols) {
    fs::path ref_path = ref_dir / (stage + ".npy");
    if (!fs::exists(ref_path)) return;
    ie::NpyArray ref = ie::load_npy(ref_path.string());
    ++checked;
    if (ref.size() != size_t(rows) * cols) {
      std::printf("%-12s size mismatch: got %d x %d, reference has %zu values  FAIL\n",
                  stage.c_str(), rows, cols, ref.size());
      ++failures;
      return;
    }
    ie::CompareResult r = ie::compare(data, ref.f32.data(), ref.size());
    bool ok = r.rel_err <= tolerance(stage);
    if (!ok) ++failures;
    std::printf("%-12s %14.3e %14.3e %12.3e  %s", stage.c_str(), r.max_abs_err, r.max_abs_ref,
                r.rel_err, ok ? "PASS" : "FAIL");
    if (!ok)
      std::printf("   worst at token %zu, index %zu", r.worst_index / cols, r.worst_index % cols);
    std::printf("\n");
  };
  {
    ie::KVCache cache = model->make_cache(int(ids.size()));
    auto t0 = Clock::now();
    model->forward(ids, cache, /*all_logits=*/true, check);
    std::printf("[forward pass over %zu tokens: %.2f s]\n\n", ids.size(), seconds_since(t0));
  }

  // --- 2. tokenizer: same prompt ids, same decoded text ---
  std::ifstream mf(ref_dir / "meta.json");
  nlohmann::json meta = nlohmann::json::parse(mf);
  {
    std::vector<int32_t> enc = tok.encode(meta.at("prompt").get<std::string>());
    report("tokenize", enc == ids, "encode(prompt) == input_ids.npy");
  }
  ie::NpyArray gen_npy = ie::load_npy((ref_dir / "generated_ids.npy").string());
  std::vector<int32_t> want;
  for (int64_t id : gen_npy.i64) want.push_back(static_cast<int32_t>(id));
  report("detokenize", tok.decode(want) == meta.at("generated_text").get<std::string>(),
         "decode(generated_ids) == generated_text");

  // --- 3. greedy generation with the KV cache: prefill once, then 1 token per step ---
  // reference.py called HF generate(do_sample=False), which still applies the
  // repetition penalty from generation_config.json, so do the same here.
  {
    ie::SamplingParams greedy = ie::load_generation_config(model_dir.string());
    greedy.temperature = 0.0f;
    ie::Sampler sampler(greedy);
    ie::KVCache cache = model->make_cache(int(ids.size() + want.size()));
    auto t0 = Clock::now();
    std::vector<float> logits = model->forward(ids, cache);
    std::vector<int32_t> history = ids, got;
    for (size_t i = 0; i < want.size(); ++i) {
      got.push_back(sampler.sample(logits.data(), cfg.vocab_size, history));
      history.push_back(got.back());
      if (i + 1 < want.size()) logits = model->forward({got.back()}, cache);
    }
    const double s = seconds_since(t0);
    size_t first_bad = 0;
    while (first_bad < want.size() && got[first_bad] == want[first_bad]) ++first_bad;
    char detail[96];
    if (first_bad == want.size())
      std::snprintf(detail, sizeof(detail), "%zu/%zu greedy tokens match (%.2f tok/s)", want.size(),
                    want.size(), want.size() / s);
    else
      std::snprintf(detail, sizeof(detail), "first mismatch at token %zu: %d vs %d", first_bad,
                    got[first_bad], want[first_bad]);
    report("generate", first_bad == want.size(), detail);
    std::printf("             \"%s\"\n", tok.decode(got).c_str());
  }

  std::printf("\n%s: %d checks, %d failed\n", failures ? "FAILED" : "ALL PASSED", checked, failures);
  return failures ? 1 : 0;
}

// ------------------------------------------------------------ inspect ----

int cmd_inspect(const Options& o) {
  if (o.positional.empty()) { usage(); return 1; }
  ie::SafeTensorsFile f(o.positional[0]);
  std::printf("%-60s %-6s %-18s %10s\n", "tensor", "dtype", "shape", "MiB");
  uint64_t total_params = 0, total_bytes = 0;
  for (const auto& [name, t] : f.tensors()) {
    std::string shape = "[";
    for (size_t i = 0; i < t.shape.size(); ++i) shape += (i ? ", " : "") + std::to_string(t.shape[i]);
    shape += "]";
    std::printf("%-60s %-6s %-18s %10.2f\n", name.c_str(), t.dtype.c_str(), shape.c_str(),
                t.num_bytes() / 1048576.0);
    total_params += t.num_elements();
    total_bytes += t.num_bytes();
  }
  std::printf("\n%zu tensors, %.1fM parameters, %.1f MiB of weights\n", f.tensors().size(),
              total_params / 1e6, total_bytes / 1048576.0);
  for (const auto& [k, v] : f.metadata()) std::printf("metadata %s = %s\n", k.c_str(), v.c_str());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
  SetConsoleOutputCP(CP_UTF8);  // print UTF-8 (emoji, Hindi, ...) correctly
#endif
  const std::vector<std::string> args = utf8_args(argc, argv);
  if (args.size() < 2) {
    usage();
    return 1;
  }
  const std::string cmd = args[1];
  try {
    if (cmd == "gpu-check") {
#ifdef IE_WITH_CUDA
      return ie::cuda::run_device_check();
#else
      std::printf("Built without CUDA (IE_ENABLE_CUDA=OFF).\n");
      return 1;
#endif
    }
    Options o = parse_options(args, 2);
    if (cmd == "generate") return cmd_generate(o);
    if (cmd == "chat") return cmd_chat(o);
    if (cmd == "tokenize") return cmd_tokenize(o);
    if (cmd == "verify") return cmd_verify(o);
    if (cmd == "test-tokenizer") return cmd_test_tokenizer(o);
    if (cmd == "inspect") return cmd_inspect(o);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  usage();
  return 1;
}
