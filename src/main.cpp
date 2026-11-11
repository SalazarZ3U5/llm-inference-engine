// ie: command-line entry point for the inference engine.
//
//   ie generate <model_dir> "<prompt>" [options]   continue a prompt (add --chat to ask as a user)
//   ie chat     <model_dir> [options]              interactive multi-turn chat
//   ie tokenize <model_dir> "<text>"               show how text splits into tokens
//   ie verify   <model_dir> <ref_dir>              check every stage + generation against reference
//   ie test-tokenizer <model_dir> <cases.json>     check the tokenizer against Hugging Face's
//   ie inspect  <model.safetensors>                list every tensor in a weights file
//   ie bench    <model_dir> [--sweep]              measure prefill and decode speed
//   ie serve-bench <model_dir> [--check]           many requests at once: continuous batching (GPU)
//   ie serve    <model_dir> [--port 8080]          OpenAI-compatible HTTP server (GPU)
//   ie perplexity <model_dir> <text_file>          measure model quality on a text
//   ie test-kernels                                check the fast matmul kernels against the reference
//   ie test-cuda                                   check every CUDA kernel against the CPU version
//   ie gpu-check                                   GPU info + vector-add test + bandwidth test
//
// Performance options: --device cpu|cuda (default: cuda if available)
//                      --no-graph, --no-tensor-cores (CUDA, for comparisons)
//                      --tc-fp16 (CUDA: tensor-core matmuls in one FP16 pass, ~2x faster prefill)
//                      --kv f32|f16 (CUDA KV cache precision, default f32)
//                      --weights f32|bf16|int8 (default bf16)  --threads N (default: physical cores)
//                      --kernel scalar|avx2 (default: avx2 if compiled in)
// Sampling options: --temp T  --top-k K  --top-p P  --rep R  --seed S  --greedy
//   (defaults come from the model's generation_config.json)
// Serving options:  --requests N (default 64)  --concurrency 1,4,16,... (batch sizes to try)
//                   --kv-gb G (KV cache budget, default 2)  --check (compare with one at a time)
//                   --same-length (every request exactly --prompt-tokens + --gen-tokens)
// Server options:   --host H (default 127.0.0.1)  --port N (default 8080)
//                   --max-batch N (requests at once, default 32); --ctx, --kv-gb and the
//                   sampling options set the defaults for requests that don't say
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
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "config.h"
#include "matrix.h"
#include "model.h"
#include "npy.h"
#include "runner.h"
#include "safetensors.h"
#include "ops.h"
#include "sampler.h"
#include "server.h"
#include "threadpool.h"
#include "tokenizer.h"
#include "weights.h"

#ifdef IE_WITH_CUDA
#include "cuda/cuda_runner.h"
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
      "  ie bench    <model_dir> [--sweep] [--prompt-tokens N] [--gen-tokens N]\n"
      "  ie serve-bench <model_dir> [--requests N] [--concurrency 1,4,16,32,64] [--kv-gb G] [--check]\n"
      "              [--same-length]\n"
      "  ie serve    <model_dir> [--host 127.0.0.1] [--port 8080] [--max-batch 32] [--ctx N] [--kv-gb G]\n"
      "              [--kv f32|f16]\n"
      "  ie perplexity <model_dir> <text_file>\n"
      "  ie test-kernels\n"
      "  ie test-cuda\n"
      "  ie gpu-check\n"
      "\n"
      "options: --max-tokens N  --ctx N  --temp T  --top-k K  --top-p P  --rep R  --seed S  --greedy\n"
      "         --device cpu|cuda  --weights f32|bf16|int8  --threads N  --kernel scalar|avx2\n"
      "         --no-graph  --no-tensor-cores  --tc-fp16  --kv f32|f16\n");
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
  // Performance
  std::string device = "auto";  // auto = cuda if built with CUDA and a GPU is present
  bool graphs = true;            // CUDA: replay each decode step as a recorded graph
  bool tensor_cores = true;      // CUDA: tensor-core GEMM for multi-token (prefill) matmuls
  bool tc_fp16 = false;          // CUDA: those matmuls in one FP16 pass instead of the exact BF16 split
  bool kv_f16 = false;           // CUDA: KV cache as FP16 (half the memory) instead of FP32
  std::string weights = "bf16";
  int threads = 0;           // 0 = one per physical core
  std::string kernel;        // empty = default
  int prompt_tokens = 64, gen_tokens = 32;
  bool sweep = false;
  // Serving
  int requests = 64;
  std::string concurrency = "1,4,16,32,64";
  double kv_gb = 2.0;
  bool check = false;
  bool same_length = false;  // serve-bench: no spread in request lengths (to compare with other engines)
  // HTTP server
  std::string host = "127.0.0.1";
  int port = 8080;
  int max_batch = 32;
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
    else if (a == "--weights") o.weights = next();
    else if (a == "--device") o.device = next();
    else if (a == "--no-graph") o.graphs = false;
    else if (a == "--no-tensor-cores") o.tensor_cores = false;
    else if (a == "--tc-fp16") o.tc_fp16 = true;
    else if (a == "--kv") {
      const std::string v = next();
      if (v != "f32" && v != "f16") throw std::runtime_error("--kv must be f32 or f16");
      o.kv_f16 = v == "f16";
    }
    else if (a == "--threads") o.threads = std::stoi(next());
    else if (a == "--kernel") o.kernel = next();
    else if (a == "--prompt-tokens") o.prompt_tokens = std::stoi(next());
    else if (a == "--gen-tokens") o.gen_tokens = std::stoi(next());
    else if (a == "--sweep") o.sweep = true;
    else if (a == "--requests") o.requests = std::stoi(next());
    else if (a == "--concurrency") o.concurrency = next();
    else if (a == "--kv-gb") o.kv_gb = std::stod(next());
    else if (a == "--check") o.check = true;
    else if (a == "--same-length") o.same_length = true;
    else if (a == "--host") o.host = next();
    else if (a == "--port") o.port = std::stoi(next());
    else if (a == "--max-batch") o.max_batch = std::stoi(next());
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

int hardware_threads() { return int(std::max(1u, std::thread::hardware_concurrency())); }

// Applies --threads and --kernel to the global compute settings.
void set_tensor_cores(bool on) {
#ifdef IE_WITH_CUDA
  ie::cuda::set_tensor_cores(on);
#else
  (void)on;
#endif
}

void apply_compute_options(const Options& o) {
  ie::set_num_threads(o.threads);  // <= 0: one per physical core
  if (!o.kernel.empty()) ie::set_kernel(ie::parse_kernel(o.kernel));
  set_tensor_cores(o.tensor_cores);
#ifdef IE_WITH_CUDA
  ie::cuda::set_tc_fp16(o.tc_fp16);
#endif
}

bool cuda_available() {
#ifdef IE_WITH_CUDA
  return ie::cuda::device_available();
#else
  return false;
#endif
}

// "auto" -> "cuda" when this build has CUDA and a GPU is present, else "cpu".
std::string resolve_device(const std::string& d) {
  if (d == "auto") return cuda_available() ? "cuda" : "cpu";
  if (d == "cuda" && !cuda_available())
    throw std::runtime_error("--device cuda: no CUDA GPU available (or this build has no CUDA support)");
  if (d != "cpu" && d != "cuda") throw std::runtime_error("unknown device '" + d + "' (use cpu or cuda)");
  return d;
}

// Loads the model onto `device` with room for `max_tokens` tokens of context.
std::unique_ptr<ie::Runner> load_runner(const fs::path& model_dir, const std::string& device,
                                        ie::DType dtype, int max_tokens, bool graphs = true,
                                        bool kv_f16 = false) {
  (void)graphs;  // only the CUDA backend uses these
  (void)kv_f16;
  auto t0 = Clock::now();
  ie::Config cfg = ie::load_config((model_dir / "config.json").string());
  ie::SafeTensorsFile st((model_dir / "model.safetensors").string());
  std::unique_ptr<ie::Runner> runner;
  if (device == "cuda") {
#ifdef IE_WITH_CUDA
    if (dtype == ie::DType::F32)
      throw std::runtime_error("the CUDA backend runs bf16 or int8 weights (use --weights bf16 or int8)");
    ie::Weights w = ie::load_weights(st, cfg, dtype);
    runner = ie::cuda::make_runner(cfg, w, max_tokens, graphs, kv_f16);  // copies to VRAM; `w` is freed here
#endif
  } else {
    runner = ie::make_cpu_runner(cfg, ie::load_weights(st, cfg, dtype), max_tokens);
  }
  std::fprintf(stderr, "[loaded %.1fM parameters as %s (%.0f MiB) in %.1f s | %s]\n",
               runner->num_params() / 1e6, ie::dtype_name(dtype), runner->weight_bytes() / 1048576.0,
               seconds_since(t0), runner->device_name().c_str());
  return runner;
}

std::unique_ptr<ie::Runner> load_runner(const fs::path& model_dir, const Options& o, int max_tokens) {
  apply_compute_options(o);
  return load_runner(model_dir, resolve_device(o.device), ie::parse_dtype(o.weights), max_tokens,
                     o.graphs, o.kv_f16);
}

// --------------------------------------------------------- generation ----

struct GenResult {
  int prompt_tokens = 0, new_tokens = 0;
  double prefill_s = 0, decode_s = 0;
  bool hit_stop = false;
};

// Feeds `prompt` after whatever is already in the KV cache, then samples up to
// `max_new` tokens, printing text as it is produced. Every generated token
// except a stop token is run through the model, so the cache always holds the
// full conversation. `history` collects all token ids (for the repetition penalty).
GenResult generate(ie::Runner& model, const ie::Tokenizer& tok,
                   ie::Sampler& sampler, std::vector<int32_t>& history,
                   const std::vector<int32_t>& prompt, int max_new,
                   const std::vector<int32_t>& stop_ids) {
  GenResult r;
  const int V = model.config().vocab_size;
  r.prompt_tokens = static_cast<int>(prompt.size());

  auto t0 = Clock::now();
  std::vector<float> logits = model.forward(prompt);  // prefill
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
    if (model.position() >= model.max_tokens()) {
      std::fputs("\n[context full]", stdout);
      break;
    }
    logits = model.forward({next});  // decode step: one token
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
  ie::Tokenizer tok(model_dir.string());
  std::string text = o.chat ? chat_turn(o.system, o.positional[1], true) : o.positional[1];
  std::vector<int32_t> prompt = tok.encode(text);
  auto model = load_runner(model_dir, o, std::min<int>(o.ctx, int(prompt.size()) + o.max_tokens + 1));
  ie::Sampler sampler(sampling_for(o, model_dir.string()));
  std::vector<int32_t> history;
  std::vector<int32_t> stops = {tok.special_id("<|im_end|>"), tok.special_id("<|endoftext|>")};

  if (!o.chat) std::fputs(o.positional[1].c_str(), stdout);
  GenResult r = generate(*model, tok, sampler, history, prompt, o.max_tokens, stops);
  std::fputs("\n", stdout);
  print_stats(r);
  std::fprintf(stderr, "[seed %llu]\n", static_cast<unsigned long long>(sampler.seed()));
  return 0;
}

int cmd_chat(const Options& o) {
  if (o.positional.empty()) { usage(); return 1; }
  const fs::path model_dir = o.positional[0];
  auto model = load_runner(model_dir, o, o.ctx);
  ie::Tokenizer tok(model_dir.string());
  ie::Sampler sampler(sampling_for(o, model_dir.string()));
  std::fprintf(stderr, "[KV cache: %d tokens, %.0f MiB]\n", model->max_tokens(),
               model->kv_cache_bytes() / 1048576.0);
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
      model->reset();
      history.clear();
      first = true;
      std::printf("[conversation cleared]\n");
      continue;
    }
    if (line.empty()) continue;

    std::vector<int32_t> prompt = tok.encode(chat_turn(o.system, line, first));
    if (model->position() + int(prompt.size()) + o.max_tokens + 1 > model->max_tokens()) {
      std::printf("[context full: starting a new conversation]\n");
      model->reset();
      history.clear();
      prompt = tok.encode(chat_turn(o.system, line, true));
    }
    first = false;
    std::printf("\n");
    GenResult r = generate(*model, tok, sampler, history, prompt, o.max_tokens, stops);
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

// -------------------------------------------------------------- bench ----

const char* kBenchText =
    "The history of computing is a story of making machines do more work with less effort. Early "
    "calculators were mechanical, built from gears and levers, and every answer had to be read off "
    "a dial by hand. Electronic computers replaced the gears with vacuum tubes, then with "
    "transistors, and finally with integrated circuits holding billions of switches. ";

struct BenchResult {
  double prefill_tps = 0, decode_tps = 0, gbps = 0;
};

// Prefill `prompt_tokens` tokens in one call, then decode `gen_tokens` tokens
// one at a time (greedy, no sampling cost), with the current thread/kernel settings.
BenchResult run_bench(ie::Runner& model, const ie::Tokenizer& tok, int prompt_tokens,
                      int gen_tokens) {
  std::vector<int32_t> unit = tok.encode(kBenchText), prompt;
  while (int(prompt.size()) < prompt_tokens) prompt.insert(prompt.end(), unit.begin(), unit.end());
  prompt.resize(prompt_tokens);
  const int V = model.config().vocab_size;

  // Warm-up: wakes the threads / GPU, faults in memory, allocates scratch buffers.
  model.reset();
  model.forward(prompt);
  model.forward({prompt[0]});
  model.reset();

  auto t0 = Clock::now();
  std::vector<float> logits = model.forward(prompt);
  const double prefill_s = seconds_since(t0);
  auto t1 = Clock::now();
  for (int i = 0; i < gen_tokens; ++i) logits = model.forward({ie::argmax(logits.data(), V)});
  const double decode_s = seconds_since(t1);

  BenchResult r;
  r.prefill_tps = prompt_tokens / prefill_s;
  r.decode_tps = gen_tokens / decode_s;
  // Each decode step streams every weight matrix once (the tied embedding is
  // read in full as the LM head), so this is the memory bandwidth achieved.
  r.gbps = r.decode_tps * double(model.weight_bytes()) / 1e9;
  return r;
}

int cmd_bench(const Options& o) {
  if (o.positional.empty()) { usage(); return 1; }
  const fs::path model_dir = o.positional[0];
  ie::Tokenizer tok(model_dir.string());

  struct Cfg {
    std::string device, weights;
    ie::Kernel kernel;
    int threads;
    bool graphs = true, tensor_cores = true;
  };
  std::vector<Cfg> cfgs;
  const int all = hardware_threads(), cores = ie::physical_cores();
  const ie::Kernel fast = ie::avx2_available() ? ie::Kernel::Avx2 : ie::Kernel::Scalar;
  if (o.sweep) {
    // The whole story, one step at a time: Phase 2 on the CPU, then the GPU.
    cfgs = {{"cpu", "f32", ie::Kernel::Scalar, 1}, {"cpu", "f32", fast, 1},
            {"cpu", "f32", fast, cores},           {"cpu", "f32", fast, all},
            {"cpu", "bf16", fast, cores},          {"cpu", "bf16", fast, all},
            {"cpu", "int8", fast, cores},          {"cpu", "int8", fast, all}};
    if (cuda_available()) {  // each GPU optimization switched on in turn
      cfgs.push_back({"cuda", "bf16", fast, cores, false, false});
      cfgs.push_back({"cuda", "bf16", fast, cores, true, false});
      cfgs.push_back({"cuda", "bf16", fast, cores, true, true});
      cfgs.push_back({"cuda", "int8", fast, cores, true, false});
      cfgs.push_back({"cuda", "int8", fast, cores, true, true});
    }
  } else {
    cfgs = {{resolve_device(o.device), o.weights,
             o.kernel.empty() ? ie::current_kernel() : ie::parse_kernel(o.kernel),
             o.threads > 0 ? o.threads : cores, o.graphs, o.tensor_cores}};
  }
  // On machines with few threads some sweep entries coincide; run each once.
  cfgs.erase(std::unique(cfgs.begin(), cfgs.end(),
                         [](const Cfg& a, const Cfg& b) {
                           return a.device == b.device && a.weights == b.weights &&
                                  a.kernel == b.kernel && a.threads == b.threads &&
                                  a.graphs == b.graphs && a.tensor_cores == b.tensor_cores;
                         }),
             cfgs.end());

  std::printf("prompt %d tokens, generate %d tokens, %d physical cores, %d hardware threads\n\n",
              o.prompt_tokens, o.gen_tokens, cores, all);
  std::printf("| Device | Weights | Kernel | Threads | Prefill tok/s | Decode tok/s | Decode GB/s |\n");
  std::printf("|--------|---------|--------|--------:|--------------:|-------------:|------------:|\n");
  std::unique_ptr<ie::Runner> model;
  std::string loaded;
  for (const Cfg& c : cfgs) {
    ie::set_num_threads(c.threads);
    ie::set_kernel(c.kernel);
    set_tensor_cores(c.tensor_cores);
    const std::string key = c.device + "/" + c.weights + (c.graphs ? "/graphs" : "");
    if (loaded != key) {
      model.reset();  // free the previous copy first
      model = load_runner(model_dir, c.device, ie::parse_dtype(c.weights),
                          o.prompt_tokens + o.gen_tokens + 8, c.graphs, o.kv_f16);
      loaded = key;
    }
    BenchResult r = run_bench(*model, tok, o.prompt_tokens, o.gen_tokens);
    const bool gpu = c.device == "cuda";
    std::printf("| %s | %s | %s | %s | %.1f | %.1f | %.1f |\n", c.device.c_str(), c.weights.c_str(),
                gpu ? (std::string("cuda") + (c.graphs ? " graph" : "") + (c.tensor_cores ? (o.tc_fp16 ? " + tc fp16" : " + tc") : "")).c_str()
                    : ie::kernel_name(c.kernel),
                gpu ? "-" : std::to_string(c.threads).c_str(), r.prefill_tps, r.decode_tps, r.gbps);
    std::fflush(stdout);
  }
  return 0;
}

// -------------------------------------------------------- serve-bench ----
//
// Serving many requests at once. All requests arrive at time 0; up to
// `concurrency` of them are in flight. A request is admitted when a slot is
// free and the KV cache has pages for its full length, is prefilled on its own,
// and then joins the running batch, which advances every sequence by one token
// per step. Finished sequences leave and free their pages at once, so the next
// request can join mid-flight ("continuous batching") instead of the batch
// waiting for its slowest member.

struct Request {
  std::vector<int32_t> prompt;
  int gen = 0;  // tokens to generate (greedy, no stop token: lengths are fixed)
};

struct ServeResult {
  std::vector<std::vector<int32_t>> out;  // generated tokens per request
  double seconds = 0, mean_ttft = 0, mean_latency = 0, p95_latency = 0;
  int steps = 0, peak_batch = 0, peak_pages = 0;
  long long tokens = 0;
};

#ifdef IE_WITH_CUDA
ServeResult serve(ie::BatchRunner& eng, const std::vector<Request>& reqs, int concurrency) {
  const int P = eng.page_size();
  auto pages_for = [&](const Request& r) { return (int(r.prompt.size()) + r.gen + P - 1) / P; };
  struct Active {
    int seq, req;
    int32_t last;  // token to feed in the next step
  };
  std::vector<Active> active;
  ServeResult res;
  res.out.resize(reqs.size());
  std::vector<double> ttft(reqs.size()), done(reqs.size());
  size_t next = 0, finished = 0;
  int reserved = 0;  // pages promised to running sequences (their full length)
  const auto t0 = Clock::now();

  while (finished < reqs.size()) {
    // Admit waiting requests while there is a free slot and room in the cache.
    while (next < reqs.size() && int(active.size()) < concurrency &&
           reserved + pages_for(reqs[next]) <= eng.total_pages()) {
      const Request& r = reqs[next];
      const int seq = eng.open_sequence();
      if (seq < 0) break;
      const int32_t tok = eng.prefill_greedy(seq, r.prompt);
      ttft[next] = seconds_since(t0);
      res.out[next].push_back(tok);
      if (r.gen <= 1) {
        eng.close_sequence(seq);
        done[next] = ttft[next];
        ++finished;
      } else {
        reserved += pages_for(r);
        active.push_back({seq, int(next), tok});
      }
      ++next;
    }
    if (active.empty()) {
      if (finished < reqs.size() && next < reqs.size())
        throw std::runtime_error("a request needs more KV-cache pages than the cache has (raise --kv-gb)");
      continue;
    }

    // One decode step for every running sequence.
    std::vector<int> seqs;
    std::vector<int32_t> toks;
    for (const Active& a : active) {
      seqs.push_back(a.seq);
      toks.push_back(a.last);
    }
    res.peak_batch = std::max(res.peak_batch, int(active.size()));
    const std::vector<int32_t> nxt = eng.decode_greedy(seqs, toks);
    ++res.steps;
    res.peak_pages = std::max(res.peak_pages, eng.total_pages() - eng.free_pages());
    const double now = seconds_since(t0);

    std::vector<Active> still;
    for (size_t i = 0; i < active.size(); ++i) {
      Active a = active[i];
      std::vector<int32_t>& o = res.out[a.req];
      o.push_back(nxt[i]);
      if (int(o.size()) >= reqs[a.req].gen) {  // done: leave the batch, free the pages
        eng.close_sequence(a.seq);
        reserved -= pages_for(reqs[a.req]);
        done[a.req] = now;
        ++finished;
      } else {
        a.last = nxt[i];
        still.push_back(a);
      }
    }
    active.swap(still);
  }

  res.seconds = seconds_since(t0);
  for (const auto& o : res.out) res.tokens += long(o.size());
  for (size_t i = 0; i < reqs.size(); ++i) {
    res.mean_ttft += ttft[i] / reqs.size();
    res.mean_latency += done[i] / reqs.size();
  }
  std::vector<double> lat = done;
  std::sort(lat.begin(), lat.end());
  res.p95_latency = lat[std::min(lat.size() - 1, size_t(0.95 * lat.size()))];
  return res;
}

std::vector<int> parse_int_list(const std::string& s) {
  std::vector<int> v;
  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    const int n = std::stoi(s.substr(i, j - i));
    if (n < 1) throw std::runtime_error("--concurrency values must be >= 1");
    v.push_back(n);
    i = j + 1;
  }
  if (v.empty()) throw std::runtime_error("--concurrency: empty list");
  return v;
}
#endif

int cmd_serve_bench(const Options& o) {
  if (o.positional.empty()) { usage(); return 1; }
#ifndef IE_WITH_CUDA
  std::printf("serve-bench needs the CUDA backend (this build has none).\n");
  return 1;
#else
  if (resolve_device(o.device == "auto" ? "cuda" : o.device) != "cuda")
    throw std::runtime_error("serve-bench runs on the GPU only (--device cuda)");
  apply_compute_options(o);
  const fs::path model_dir = o.positional[0];
  ie::Tokenizer tok(model_dir.string());
  std::vector<int> levels = parse_int_list(o.concurrency);
  const int max_batch = *std::max_element(levels.begin(), levels.end());

  // Requests: different slices of the bench text, with lengths spread between
  // half and all of --prompt-tokens / --gen-tokens, so sequences finish at
  // different times (which is what continuous batching is for).
  std::vector<int32_t> unit = tok.encode(kBenchText), text;
  while (int(text.size()) < o.prompt_tokens + int(unit.size()))
    text.insert(text.end(), unit.begin(), unit.end());
  std::mt19937 rng(1234);
  auto between = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
  std::vector<Request> reqs(std::max(1, o.requests));
  int longest = 0;
  for (Request& r : reqs) {
    const int len = o.same_length ? std::max(1, o.prompt_tokens)
                                  : between(std::max(1, o.prompt_tokens / 2), std::max(1, o.prompt_tokens));
    const int start = between(0, int(unit.size()) - 1);
    r.prompt.assign(text.begin() + start, text.begin() + start + len);
    r.gen = o.same_length ? std::max(1, o.gen_tokens)
                          : between(std::max(1, o.gen_tokens / 2), std::max(1, o.gen_tokens));
    longest = std::max(longest, len + r.gen);
  }

  // Load the model once, sized for the largest batch.
  const auto t0 = Clock::now();
  const ie::DType dtype = ie::parse_dtype(o.weights);
  if (dtype == ie::DType::F32) throw std::runtime_error("the CUDA backend runs bf16 or int8 weights");
  ie::Config cfg = ie::load_config((model_dir / "config.json").string());
  std::unique_ptr<ie::BatchRunner> eng;
  {
    ie::SafeTensorsFile st((model_dir / "model.safetensors").string());
    ie::Weights w = ie::load_weights(st, cfg, dtype);
    eng = ie::cuda::make_batch_runner(cfg, w, max_batch, longest, size_t(o.kv_gb * 1073741824.0),
                                      o.graphs, o.kv_f16);
  }
  std::fprintf(stderr, "[loaded as %s in %.1f s | %s]\n", ie::dtype_name(dtype), seconds_since(t0),
               eng->device_name().c_str());

  // Warm-up: a full-length prefill, then decode steps at every batch size from
  // max_batch down to 1, so buffers are allocated and each batch size's graph
  // is recorded before anything is timed.
  {
    std::vector<int> seqs;
    for (int i = 0; i < max_batch; ++i) {
      const int s = eng->open_sequence();
      if (s < 0) break;
      if (i == 0) eng->prefill_greedy(s, std::vector<int32_t>(text.begin(), text.begin() + o.prompt_tokens));
      else eng->prefill_greedy(s, {text[i % text.size()]});
      seqs.push_back(s);
    }
    while (!seqs.empty()) {
      eng->decode_greedy(seqs, std::vector<int32_t>(seqs.size(), text[0]));
      eng->close_sequence(seqs.back());
      seqs.pop_back();
    }
  }

  const int min_prompt = o.same_length ? std::max(1, o.prompt_tokens) : std::max(1, o.prompt_tokens / 2);
  const int min_gen = o.same_length ? std::max(1, o.gen_tokens) : std::max(1, o.gen_tokens / 2);
  std::printf("%d requests, prompts %d-%d tokens, generating %d-%d tokens each (greedy), %s weights\n",
              int(reqs.size()), min_prompt, std::max(1, o.prompt_tokens), min_gen,
              std::max(1, o.gen_tokens), ie::dtype_name(dtype));
  std::printf("KV cache: %d pages of %d tokens (%.0f MiB), up to %d sequences at once\n\n",
              eng->total_pages(), eng->page_size(), eng->kv_cache_bytes() / 1048576.0, max_batch);

  // The reference for --check: every request run alone (concurrency 1).
  std::optional<ServeResult> ref;
  if (o.check && std::find(levels.begin(), levels.end(), 1) == levels.end())
    ref = serve(*eng, reqs, 1);

  std::printf("| Concurrency | Peak batch | Steps | Time s | Throughput tok/s | Mean TTFT ms | Mean latency ms | p95 latency ms | Peak KV pages |%s\n",
              o.check ? " Same as one at a time |" : "");
  std::printf("|------------:|-----------:|------:|-------:|-----------------:|-------------:|----------------:|---------------:|--------------:|%s\n",
              o.check ? ":---|" : "");
  bool all_same = true;
  for (int c : levels) {
    ServeResult r = serve(*eng, reqs, c);
    if (c == 1 && o.check && !ref) ref = r;
    std::string match;
    if (o.check) {
      int same = 0, first_diff = -1, diff_req = -1;
      for (size_t i = 0; i < reqs.size(); ++i) {
        if (r.out[i] == ref->out[i]) { ++same; continue; }
        if (diff_req < 0) {
          diff_req = int(i);
          first_diff = 0;
          while (first_diff < int(r.out[i].size()) && r.out[i][first_diff] == ref->out[i][first_diff]) ++first_diff;
        }
      }
      char buf[160];
      if (same == int(reqs.size())) {
        std::snprintf(buf, sizeof(buf), " %d/%d identical |", same, int(reqs.size()));
      } else {
        all_same = false;
        std::snprintf(buf, sizeof(buf), " %d/%d identical (request %d differs from token %d) |", same,
                      int(reqs.size()), diff_req, first_diff);
      }
      match = buf;
    }
    std::printf("| %d | %d | %d | %.2f | %.1f | %.1f | %.1f | %.1f | %d |%s\n", c, r.peak_batch, r.steps,
                r.seconds, r.tokens / r.seconds, r.mean_ttft * 1e3, r.mean_latency * 1e3,
                r.p95_latency * 1e3, r.peak_pages, match.c_str());
    std::fflush(stdout);
  }
  if (o.check && !all_same && o.tensor_cores && max_batch >= 32)
    std::printf("\nnote: batches of 32+ tokens use tensor cores, which round differently in the last bits;\n"
                "      a near-tie between two tokens can then go the other way. --no-tensor-cores keeps\n"
                "      every batch size bit-exact.\n");
  return o.check && !all_same ? 2 : 0;
#endif
}

// -------------------------------------------------------------- serve ----

// The OpenAI-compatible HTTP server (src/server.cpp) on the batched GPU engine.
int cmd_serve(const Options& o) {
  if (o.positional.empty()) { usage(); return 1; }
#ifndef IE_WITH_CUDA
  std::printf("serve needs the CUDA backend (this build has none).\n");
  return 1;
#else
  if (resolve_device(o.device == "auto" ? "cuda" : o.device) != "cuda")
    throw std::runtime_error("serve runs on the GPU only (--device cuda)");
  if (o.max_batch < 1) throw std::runtime_error("--max-batch must be >= 1");
  if (o.port < 0 || o.port > 65535) throw std::runtime_error("--port must be 0-65535");
  apply_compute_options(o);
  const fs::path model_dir = o.positional[0];
  ie::Tokenizer tok(model_dir.string());

  ie::ServerOptions so;
  so.host = o.host;
  so.port = o.port;
  // The model's id in the API: its folder name ("qwen2.5-0.5b-instruct").
  fs::path name = model_dir.lexically_normal();
  if (name.filename().empty()) name = name.parent_path();
  so.model_id = name.filename().string();
  so.default_system = o.system;
  so.sampling = sampling_for(o, model_dir.string());
  so.default_max_tokens = o.max_tokens;

  const auto t0 = Clock::now();
  const ie::DType dtype = ie::parse_dtype(o.weights);
  if (dtype == ie::DType::F32) throw std::runtime_error("the CUDA backend runs bf16 or int8 weights");
  ie::Config cfg = ie::load_config((model_dir / "config.json").string());
  std::unique_ptr<ie::BatchRunner> eng;
  {
    ie::SafeTensorsFile st((model_dir / "model.safetensors").string());
    ie::Weights w = ie::load_weights(st, cfg, dtype);
    eng = ie::cuda::make_batch_runner(cfg, w, o.max_batch, o.ctx, size_t(o.kv_gb * 1073741824.0),
                                      o.graphs, o.kv_f16);
  }
  std::fprintf(stderr, "[loaded as %s in %.1f s | %s | KV cache %.0f MiB]\n", ie::dtype_name(dtype),
               seconds_since(t0), eng->device_name().c_str(), eng->kv_cache_bytes() / 1048576.0);
  ie::run_server(*eng, tok, so);
  return 0;
#endif
}

// --------------------------------------------------------- perplexity ----

// Perplexity = exp(average negative log-likelihood of each next token).
// Lower is better; it is the standard way to check that a faster or smaller
// version of a model (e.g. INT8) still predicts text as well as the original.
int cmd_perplexity(const Options& o) {
  if (o.positional.size() < 2) { usage(); return 1; }
  const fs::path model_dir = o.positional[0];
  ie::Tokenizer tok(model_dir.string());
  std::ifstream f(fs::path(o.positional[1]), std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + o.positional[1]);
  const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

  std::vector<int32_t> ids = tok.encode(text, /*allow_special=*/false);
  if (int(ids.size()) > o.ctx) ids.resize(o.ctx);
  if (ids.size() < 2) throw std::runtime_error("text too short");
  auto model = load_runner(model_dir, o, int(ids.size()));
  const int V = model->config().vocab_size, chunk = 128;

  // Feed the text in chunks through the KV cache; logits at position i score token i+1.
  double nll = 0;
  int scored = 0;
  auto t0 = Clock::now();
  for (size_t start = 0; start < ids.size(); start += chunk) {
    const size_t end = std::min(ids.size(), start + chunk);
    std::vector<int32_t> part(ids.begin() + start, ids.begin() + end);
    std::vector<float> logits = model->forward(part, /*all_logits=*/true);
    for (size_t i = start; i < end && i + 1 < ids.size(); ++i) {
      const float* row = logits.data() + (i - start) * size_t(V);
      double mx = row[0];
      for (int v = 1; v < V; ++v) mx = std::max(mx, double(row[v]));
      double sum = 0;
      for (int v = 0; v < V; ++v) sum += std::exp(double(row[v]) - mx);
      nll += (mx + std::log(sum)) - double(row[ids[i + 1]]);  // -log softmax(row)[target]
      ++scored;
    }
  }
  const double s = seconds_since(t0);
  std::printf("%s weights on %s: perplexity %.4f over %d tokens (%.1f s, %.1f tok/s)\n",
              ie::dtype_name(model->weight_dtype()), model->device_name().c_str(),
              std::exp(nll / scored), scored, s,
              ids.size() / s);
  return 0;
}

// ------------------------------------------------------- test-kernels ----

// Checks every matmul kernel (each weight type x scalar/AVX2 x thread count x
// token count) against the plain reference `ie::linear` on random data.
int cmd_test_kernels(const Options&) {
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  struct Shape { int rows, cols; };
  const Shape shapes[] = {{37, 75}, {128, 896}, {64, 4864}};
  const int token_counts[] = {1, 2, 3, 4, 5, 9};
  std::vector<ie::Kernel> kernels = {ie::Kernel::Scalar};
  if (ie::avx2_available()) kernels.push_back(ie::Kernel::Avx2);

  int failures = 0, tests = 0;
  double worst = 0;
  for (const Shape& sh : shapes) {
    std::vector<float> w(size_t(sh.rows) * sh.cols), bias(sh.rows);
    for (float& x : w) x = 0.1f * uni(rng);
    for (float& x : bias) x = uni(rng);
    for (ie::DType dt : {ie::DType::F32, ie::DType::BF16, ie::DType::I8}) {
      ie::Matrix m = ie::make_matrix(w, sh.rows, sh.cols, dt);
      std::vector<float> deq(w.size());  // the exact values the kernel should be using
      for (int r = 0; r < sh.rows; ++r) m.row_to_f32(r, &deq[size_t(r) * sh.cols]);
      for (int T : token_counts) {
        std::vector<float> x(size_t(T) * sh.cols), want(size_t(T) * sh.rows), got(want.size());
        for (float& v : x) v = uni(rng);
        ie::linear(want.data(), x.data(), deq.data(), bias.data(), T, sh.cols, sh.rows);
        for (ie::Kernel k : kernels) {
          for (int threads : {1, 3}) {
            ie::set_kernel(k);
            ie::set_num_threads(threads);
            std::fill(got.begin(), got.end(), 0.0f);
            ie::matmul(got.data(), x.data(), m, bias.data(), T);
            ie::CompareResult r = ie::compare(got.data(), want.data(), want.size());
            ++tests;
            worst = std::max(worst, r.rel_err);
            if (r.rel_err > 1e-5) {
              ++failures;
              std::printf("FAIL %s %s threads=%d T=%d %dx%d: rel_err %.2e\n", ie::dtype_name(dt),
                          ie::kernel_name(k), threads, T, sh.rows, sh.cols, r.rel_err);
            }
          }
        }
      }
    }
  }
  std::printf("%s: %d kernel tests, %d failed (worst relative error %.2e)\n",
              failures ? "FAILED" : "ALL PASSED", tests, failures, worst);
  return failures ? 1 : 0;
}

int cmd_test_cuda(const Options&) {
#ifdef IE_WITH_CUDA
  if (!ie::cuda::device_available()) {
    std::printf("No CUDA GPU found.\n");
    return 1;
  }
  return ie::cuda::run_selftest();
#else
  std::printf("Built without CUDA (IE_ENABLE_CUDA=OFF).\n");
  return 1;
#endif
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
  auto model = load_runner(model_dir, o, o.ctx);
  const ie::Config& cfg = model->config();
  ie::Tokenizer tok(model_dir.string());

  ie::NpyArray ids_npy = ie::load_npy((ref_dir / "input_ids.npy").string());
  std::vector<int32_t> ids;
  for (int64_t id : ids_npy.i64) ids.push_back(static_cast<int32_t>(id));

  // INT8 weights are an approximation, so their numbers can't match the
  // reference to 1e-3. Show the errors for information, and judge quality
  // with `ie perplexity` instead.
  const bool lossy = model->weight_dtype() == ie::DType::I8;
  if (lossy)
    std::printf("\nINT8 weights are approximate: model errors below are shown for information only.\n"
                "Use `ie perplexity` to measure INT8 quality.\n");

  int failures = 0, checked = 0;
  auto report = [&](const char* name, bool ok, const std::string& detail, bool info = false) {
    if (!info) {
      ++checked;
      if (!ok) ++failures;
    }
    std::printf("%-12s %-44s %s\n", name, detail.c_str(), info ? "(info)" : ok ? "PASS" : "FAIL");
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
    if (lossy) {
      --checked;
      std::printf("%-12s %14.3e %14.3e %12.3e  (info)\n", stage.c_str(), r.max_abs_err,
                  r.max_abs_ref, r.rel_err);
      return;
    }
    if (!ok) ++failures;
    std::printf("%-12s %14.3e %14.3e %12.3e  %s", stage.c_str(), r.max_abs_err, r.max_abs_ref,
                r.rel_err, ok ? "PASS" : "FAIL");
    if (!ok)
      std::printf("   worst at token %zu, index %zu", r.worst_index / cols, r.worst_index % cols);
    std::printf("\n");
  };
  {
    model->reset();
    auto t0 = Clock::now();
    model->forward(ids, /*all_logits=*/true, check);
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
    model->reset();
    auto t0 = Clock::now();
    std::vector<float> logits = model->forward(ids);
    std::vector<int32_t> history = ids, got;
    for (size_t i = 0; i < want.size(); ++i) {
      got.push_back(sampler.sample(logits.data(), cfg.vocab_size, history));
      history.push_back(got.back());
      if (i + 1 < want.size()) logits = model->forward({got.back()});
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
    report("generate", first_bad == want.size(), detail, lossy);
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
    if (cmd == "bench") return cmd_bench(o);
    if (cmd == "serve-bench") return cmd_serve_bench(o);
    if (cmd == "serve") return cmd_serve(o);
    if (cmd == "perplexity") return cmd_perplexity(o);
    if (cmd == "test-kernels") return cmd_test_kernels(o);
    if (cmd == "test-cuda") return cmd_test_cuda(o);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  usage();
  return 1;
}
