// The OpenAI-compatible server: request parsing, the engine thread and SSE streaming.

// First: on Windows httplib.h includes winsock2.h, which must come before any windows.h.
#include <httplib.h>

#include "server.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "threadpool.h"

namespace ie {
namespace {

using json = nlohmann::ordered_json;  // keeps fields in the order written, as OpenAI sends them
using Clock = std::chrono::steady_clock;
using std::chrono::milliseconds;

double ms_between(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// Model output can contain invalid UTF-8 (a token can be a stray byte), which
// nlohmann::json refuses to serialize; such bytes become U+FFFD instead.
std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

// Greedy without a repetition penalty needs only the argmax, which the GPU
// computes, so the 600 KB row of logits never has to be copied back.
bool argmax_only(const SamplingParams& p) {
  return p.temperature <= 0.0f && p.repetition_penalty == 1.0f;
}

// ------------------------------------------------------------- requests ----

// One request, shared by its HTTP thread and the engine thread.
struct Job {
  // Set by the HTTP thread before it is submitted, read-only afterwards.
  std::string id;
  std::vector<int32_t> prompt;
  int max_new = 0;
  SamplingParams sampling;
  std::vector<std::string> stop;  // stop strings; the text ends just before the first match
  Clock::time_point arrived;
  int64_t created = 0;            // Unix time, for the response

  // Set by the HTTP thread when its client has gone away; the engine then drops the request.
  std::atomic<bool> cancelled{false};

  // Engine -> HTTP thread, guarded by mu.
  std::mutex mu;
  std::condition_variable cv;
  std::string text;               // produced but not yet taken
  bool finished = false;
  std::string finish_reason;      // "stop", "length", "cancelled" or "error"
  std::string error;
  int completion_tokens = 0;

  // HTTP side: waits up to `timeout` for new text or the end, and takes the text.
  std::string take(milliseconds timeout, bool& done) {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait_for(lock, timeout, [&] { return !text.empty() || finished; });
    done = finished;
    std::string out;
    out.swap(text);
    return out;
  }
};

// Length of the longest end of `s` that is the start of a stop string: text
// that may still turn into a stop string, so it can't be sent yet.
size_t partial_stop(const std::string& s, const std::vector<std::string>& stops) {
  size_t best = 0;
  for (const std::string& stop : stops)
    for (size_t k = std::min(s.size(), stop.size() - 1); k > best; --k)
      if (s.compare(s.size() - k, k, stop, 0, k) == 0) {
        best = k;
        break;
      }
  return best;
}

// --------------------------------------------------------------- engine ----
//
// The continuous-batching loop of serve-bench, fed by a queue instead of a
// fixed list. Only this thread touches the BatchRunner.

class Engine {
 public:
  Engine(BatchRunner& eng, const Tokenizer& tok)
      : eng_(eng), tok_(tok), vocab_(eng.config().vocab_size),
        stop_ids_{tok.special_id("<|im_end|>"), tok.special_id("<|endoftext|>")} {
    free_pages = eng.free_pages();
    thread_ = std::thread([this] { loop(); });
  }
  ~Engine() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      quit_ = true;
    }
    cv_.notify_one();
    thread_.join();
  }
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  void submit(std::shared_ptr<Job> job) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      incoming_.push_back(std::move(job));
    }
    cv_.notify_one();
  }

  // Pages reserved for a request: its whole length, so it can never run out mid-reply.
  int pages_for(const Job& j) const {
    const int P = eng_.page_size();
    return (int(j.prompt.size()) + j.max_new + P - 1) / P;
  }

  // Scheduler state for /health, published after every step.
  std::atomic<int> running{0}, waiting{0}, free_pages{0}, peak_batch{0};
  std::atomic<long long> served{0};

 private:
  struct Running {
    Running(std::shared_ptr<Job> j, const Tokenizer& tok)
        : job(std::move(j)), sampler(job->sampling), decoder(tok), history(job->prompt) {}
    std::shared_ptr<Job> job;
    Sampler sampler;               // its own seed and random stream
    StreamDecoder decoder;         // never splits a multi-byte UTF-8 character
    std::vector<int32_t> history;  // prompt + generated, for the repetition penalty
    std::string held;              // decoded text not sent yet (might begin a stop string)
    int seq = -1, pages = 0;
    int32_t last = 0;              // the token to feed in the next decode step
    int generated = 0;
    bool hit_stop_string = false;
    Clock::time_point first_token;
  };

  void loop() {
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mu_);
        // Sleep while there is nothing to do.
        cv_.wait(lock, [&] {
          return quit_ || !incoming_.empty() || !waiting_.empty() || !running_.empty();
        });
        if (quit_) break;
        for (auto& j : incoming_) waiting_.push_back(std::move(j));
        incoming_.clear();
      }
      admit();
      step();
      running = int(running_.size());
      waiting = int(waiting_.size());
      free_pages = eng_.free_pages();
    }
    for (auto& r : running_) retire(*r, "error", "the server is shutting down");
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& j : incoming_) waiting_.push_back(std::move(j));
    for (auto& j : waiting_) finish(*j, "error", "the server is shutting down", 0, "");
  }

  // Starts waiting requests, oldest first, while there is a free slot and the
  // KV cache can hold each one's full length. Each prompt is prefilled on its own.
  void admit() {
    while (!waiting_.empty() && int(running_.size()) < eng_.max_batch()) {
      std::shared_ptr<Job> job = waiting_.front();
      if (job->cancelled) {
        waiting_.pop_front();
        Running r(job, tok_);
        retire(r, "cancelled");
        continue;
      }
      const int pages = pages_for(*job);
      if (reserved_ + pages > eng_.total_pages()) break;
      const int seq = eng_.open_sequence();
      if (seq < 0) break;
      waiting_.pop_front();
      auto r = std::make_unique<Running>(job, tok_);
      r->seq = seq;
      r->pages = pages;
      reserved_ += pages;
      std::string reason, error;
      try {
        int32_t next = 0;
        if (argmax_only(job->sampling)) {
          next = eng_.prefill_greedy(seq, job->prompt);
        } else {
          const std::vector<float> logits = eng_.prefill(seq, job->prompt);
          next = r->sampler.sample(logits.data(), vocab_, r->history);
        }
        r->first_token = Clock::now();
        reason = accept(*r, next);
      } catch (const std::exception& e) {
        reason = "error";
        error = e.what();
      }
      if (reason.empty()) running_.push_back(std::move(r));
      else retire(*r, reason, error);
    }
  }

  // One decode step: every running request gets its next token.
  void step() {
    // Clients that went away: close their sequences, which frees their pages.
    for (auto& r : running_)
      if (r->job->cancelled) {
        retire(*r, "cancelled");
        r.reset();
      }
    running_.erase(std::remove(running_.begin(), running_.end(), nullptr), running_.end());
    if (running_.empty()) return;

    const int n = int(running_.size());
    std::vector<int> seqs(n);
    std::vector<int32_t> tokens(n), next(n);
    bool fast = true;
    for (int i = 0; i < n; ++i) {
      seqs[i] = running_[i]->seq;
      tokens[i] = running_[i]->last;
      fast = fast && argmax_only(running_[i]->job->sampling);
    }
    peak_batch = std::max(peak_batch.load(), n);
    try {
      if (fast) {
        next = eng_.decode_greedy(seqs, tokens);
      } else {
        const std::vector<float> logits = eng_.decode(seqs, tokens);
        // Sampling a row takes about a millisecond (repetition penalty and
        // top-k over 151,936 logits), so the rows are spread over the CPU threads.
        thread_pool().parallel_for(n, [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; ++i) {
            Running& r = *running_[size_t(i)];
            const float* row = logits.data() + size_t(i) * size_t(vocab_);
            next[size_t(i)] = argmax_only(r.job->sampling)
                                  ? argmax(row, vocab_)
                                  : r.sampler.sample(row, vocab_, r.history);
          }
        });
      }
    } catch (const std::exception& e) {
      for (auto& r : running_) retire(*r, "error", e.what());  // the step failed for all of them
      running_.clear();
      return;
    }
    for (int i = 0; i < n; ++i) {
      const std::string reason = accept(*running_[i], next[i]);
      if (!reason.empty()) {
        retire(*running_[i], reason);  // leaves the batch now; its slot and pages are free
        running_[i].reset();
      }
    }
    running_.erase(std::remove(running_.begin(), running_.end(), nullptr), running_.end());
  }

  // Takes a request's next token. Returns why it finished, or "" to go on.
  std::string accept(Running& r, int32_t token) {
    Job& job = *r.job;
    ++r.generated;
    if (std::find(stop_ids_.begin(), stop_ids_.end(), token) != stop_ids_.end()) return "stop";
    r.history.push_back(token);
    r.held += r.decoder.push(token);
    size_t at = std::string::npos;
    for (const std::string& s : job.stop) at = std::min(at, r.held.find(s));
    if (at != std::string::npos) {
      r.held.resize(at);
      r.hit_stop_string = true;
      return "stop";
    }
    if (r.generated >= job.max_new) return "length";
    const size_t ready = r.held.size() - partial_stop(r.held, job.stop);
    if (ready > 0) {
      send(job, r.held.substr(0, ready));
      r.held.erase(0, ready);
    }
    r.last = token;
    return "";
  }

  void send(Job& job, const std::string& text) {
    {
      std::lock_guard<std::mutex> lock(job.mu);
      job.text += text;
    }
    job.cv.notify_all();
  }

  void finish(Job& job, const std::string& reason, const std::string& error, int tokens,
              const std::string& tail) {
    {
      std::lock_guard<std::mutex> lock(job.mu);
      job.text += tail;
      job.finished = true;
      job.finish_reason = reason;
      job.error = error;
      job.completion_tokens = tokens;
    }
    job.cv.notify_all();
  }

  // Ends a request: frees its sequence, sends the rest of its text, logs it.
  void retire(Running& r, const std::string& reason, const std::string& error = "") {
    if (r.seq >= 0) {
      try {
        eng_.close_sequence(r.seq);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "[%s: closing sequence %d failed: %s]\n", r.job->id.c_str(), r.seq,
                     e.what());
      }
      reserved_ -= r.pages;
      r.seq = -1;
    }
    std::string tail;
    if (reason == "stop" || reason == "length") {
      tail = r.held;
      if (!r.hit_stop_string) tail += r.decoder.flush();
    }
    finish(*r.job, reason, error, r.generated, tail);
    ++served;

    const Job& job = *r.job;
    const auto now = Clock::now();
    char speed[64] = "-";
    if (r.generated > 1)
      std::snprintf(speed, sizeof(speed), "%.1f tok/s",
                    (r.generated - 1) / std::max(ms_between(r.first_token, now) / 1e3, 1e-9));
    char ttft[64] = "-";
    if (r.generated > 0)
      std::snprintf(ttft, sizeof(ttft), "%.0f ms", ms_between(job.arrived, r.first_token));
    std::fprintf(stderr, "[%s: %d prompt + %d generated tokens | first token %s | %s | %s%s%s]\n",
                 job.id.c_str(), int(job.prompt.size()), r.generated, ttft, speed, reason.c_str(),
                 error.empty() ? "" : ": ", error.c_str());
  }

  BatchRunner& eng_;
  const Tokenizer& tok_;
  const int vocab_;
  const std::vector<int32_t> stop_ids_;

  std::mutex mu_;
  std::condition_variable cv_;
  std::vector<std::shared_ptr<Job>> incoming_;  // guarded by mu_
  bool quit_ = false;                           // guarded by mu_

  // Engine thread only.
  std::deque<std::shared_ptr<Job>> waiting_;
  std::vector<std::unique_ptr<Running>> running_;
  int reserved_ = 0;  // pages promised to running requests
  std::thread thread_;
};

// -------------------------------------------------------------- parsing ----

// A request the server rejects, with what OpenAI's API would say.
struct ApiError : std::runtime_error {
  ApiError(int http_status, const std::string& message, std::string field_name = "",
           std::string error_code = "")
      : std::runtime_error(message), status(http_status), param(std::move(field_name)),
        code(std::move(error_code)) {}
  int status;
  std::string param, code;
};

ApiError bad_field(const std::string& name, const std::string& problem) {
  return ApiError(400, "'" + name + "' " + problem, name);
}

// Optional fields: missing and null both mean "use the default".
const json* field(const json& body, const char* name) {
  auto it = body.find(name);
  return it == body.end() || it->is_null() ? nullptr : &*it;
}

double number_field(const json& body, const char* name, double def, double lo, double hi) {
  const json* f = field(body, name);
  if (!f) return def;
  if (!f->is_number()) throw bad_field(name, "must be a number");
  const double v = f->get<double>();
  if (!(v >= lo && v <= hi))
    throw bad_field(name, "must be between " + json(lo).dump() + " and " + json(hi).dump());
  return v;
}

int64_t int_field(const json& body, const char* name, int64_t def, int64_t lo) {
  const json* f = field(body, name);
  if (!f) return def;
  if (!f->is_number_integer()) throw bad_field(name, "must be an integer");
  const int64_t v = f->is_number_unsigned()
                        ? int64_t(std::min<uint64_t>(f->get<uint64_t>(), INT64_MAX))
                        : f->get<int64_t>();
  if (v < lo) throw bad_field(name, "must be at least " + std::to_string(lo));
  return v;
}

bool bool_field(const json& body, const char* name, bool def) {
  const json* f = field(body, name);
  if (!f) return def;
  if (!f->is_boolean()) throw bad_field(name, "must be true or false");
  return f->get<bool>();
}

// A message's text: a string, or a list of {"type": "text", "text": ...} parts.
std::string message_text(const json& m) {
  const json* c = field(m, "content");
  if (!c) return "";
  if (c->is_string()) return c->get<std::string>();
  if (!c->is_array())
    throw bad_field("messages", "content must be a string or a list of text parts");
  std::string s;
  for (const json& part : *c) {
    const json* text = part.is_object() ? field(part, "text") : nullptr;
    if (!text || !text->is_string() || part.value("type", std::string()) != "text")
      throw bad_field("messages", "content parts must be text (the model reads no images or audio)");
    s += text->get<std::string>();
  }
  return s;
}

// Qwen2.5's ChatML template (chat_template in tokenizer_config.json, without
// tools): a leading system message replaces the default one, and the prompt
// ends by opening the assistant's turn.
std::string chat_prompt(const json& body, const std::string& default_system) {
  const json* messages = field(body, "messages");
  if (!messages || !messages->is_array() || messages->empty())
    throw bad_field("messages", "must be a non-empty list of messages");
  std::string s;
  for (size_t i = 0; i < messages->size(); ++i) {
    const json& m = (*messages)[i];
    const json* role_field = m.is_object() ? field(m, "role") : nullptr;
    if (!role_field || !role_field->is_string())
      throw bad_field("messages", "each message needs a role");
    std::string role = role_field->get<std::string>();
    if (role == "developer") role = "system";  // OpenAI's newer name for the system role
    if (role != "system" && role != "user" && role != "assistant")
      throw bad_field("messages",
                      "role '" + role + "' is not supported (use system, user or assistant)");
    if (i == 0 && role != "system") s += "<|im_start|>system\n" + default_system + "<|im_end|>\n";
    s += "<|im_start|>" + role + "\n" + message_text(m) + "<|im_end|>\n";
  }
  return s + "<|im_start|>assistant\n";
}

// /v1/completions: the prompt as given, no template.
std::string completion_prompt(const json& body) {
  const json* p = field(body, "prompt");
  if (p && p->is_string()) return p->get<std::string>();
  if (p && p->is_array() && p->size() == 1 && (*p)[0].is_string())
    return (*p)[0].get<std::string>();
  throw bad_field("prompt", "must be a string");
}

std::vector<std::string> stop_strings(const json& body) {
  const json* f = field(body, "stop");
  std::vector<std::string> out;
  if (!f) return out;
  if (f->is_string()) {
    out.push_back(f->get<std::string>());
  } else if (f->is_array()) {
    for (const json& s : *f) {
      if (!s.is_string()) throw bad_field("stop", "must be a string or a list of strings");
      out.push_back(s.get<std::string>());
    }
  } else {
    throw bad_field("stop", "must be a string or a list of strings");
  }
  out.erase(std::remove(out.begin(), out.end(), std::string()), out.end());
  return out;
}

// Request fields over the server's defaults (generation_config.json + command line).
SamplingParams sampling_from(const json& body, const SamplingParams& def) {
  SamplingParams p = def;
  p.temperature = float(number_field(body, "temperature", def.temperature, 0.0, 2.0));
  p.top_p = float(number_field(body, "top_p", def.top_p, 0.0, 1.0));
  p.top_k = int(std::min<int64_t>(int_field(body, "top_k", def.top_k, -1), 1 << 30));
  if (p.top_k < 0) p.top_k = 0;  // -1 (as in vLLM) = no limit
  p.repetition_penalty =
      float(number_field(body, "repetition_penalty", def.repetition_penalty, 0.01, 10.0));
  if (const json* s = field(body, "seed")) {
    if (!s->is_number_integer()) throw bad_field("seed", "must be an integer");
    p.seed = s->is_number_unsigned() ? s->get<uint64_t>() : uint64_t(s->get<int64_t>());
    // The Sampler reads 0 as "pick a random seed", but a client sending seed 0
    // wants repeatable output like any other seed.
    if (p.seed == 0) p.seed = 0x9E3779B97F4A7C15ull;
  }
  return p;
}

void send_error(httplib::Response& res, int status, const std::string& message,
                const std::string& param = "", const std::string& code = "") {
  const json error = {{"message", message},
                      {"type", status >= 500 ? "server_error" : "invalid_request_error"},
                      {"param", param.empty() ? json(nullptr) : json(param)},
                      {"code", code.empty() ? json(nullptr) : json(code)}};
  res.status = status;
  res.set_content(dump({{"error", error}}), "application/json");
}

// ------------------------------------------------------------------ API ----

class Api {
 public:
  Api(BatchRunner& eng, const Tokenizer& tok, const ServerOptions& opt)
      : tok_(tok), opt_(opt), max_batch_(eng.max_batch()), total_pages_(eng.total_pages()),
        context_(std::min(eng.max_seq_len(), eng.total_pages() * eng.page_size())),
        started_(std::time(nullptr)), engine_(eng, tok) {}

  void routes(httplib::Server& svr) {
    using Req = httplib::Request;
    using Res = httplib::Response;
    svr.Post("/v1/chat/completions", [this](const Req& req, Res& res) { complete(req, res, true); });
    svr.Post("/v1/completions", [this](const Req& req, Res& res) { complete(req, res, false); });
    svr.Get("/v1/models", [this](const Req&, Res& res) {
      const json list = {{"object", "list"}, {"data", json::array({model_card()})}};
      res.set_content(dump(list), "application/json");
    });
    svr.Get("/v1/models/(.+)", [this](const Req& req, Res& res) {
      if (req.matches[1] == opt_.model_id)
        res.set_content(dump(model_card()), "application/json");
      else
        send_error(res, 404, "no model named '" + req.matches[1].str() + "'", "model",
                   "model_not_found");
    });
    svr.Get("/health", [this](const Req&, Res& res) {
      const json h = {{"status", "ok"},
                      {"model", opt_.model_id},
                      {"running", engine_.running.load()},
                      {"waiting", engine_.waiting.load()},
                      {"max_batch", max_batch_},
                      {"peak_batch", engine_.peak_batch.load()},
                      {"requests_served", engine_.served.load()},
                      {"kv_pages_free", engine_.free_pages.load()},
                      {"kv_pages_total", total_pages_},
                      {"context_tokens", context_}};
      res.set_content(dump(h), "application/json");
    });
  }

 private:
  json model_card() const {
    return {{"id", opt_.model_id}, {"object", "model"}, {"created", started_}, {"owned_by", "ie"}};
  }

  void complete(const httplib::Request& req, httplib::Response& res, bool chat) {
    auto job = std::make_shared<Job>();
    job->arrived = Clock::now();
    job->created = std::time(nullptr);
    bool stream = false, include_usage = false;
    try {
      const json body = json::parse(req.body);
      if (!body.is_object()) throw ApiError(400, "the request body must be a JSON object");
      const json* n = field(body, "n");
      if (n && !(n->is_number_integer() && n->get<int64_t>() == 1))
        throw bad_field("n", "must be 1 (one reply per request)");
      if (field(body, "tools")) throw bad_field("tools", "is not supported by this server");
      stream = bool_field(body, "stream", false);
      if (const json* so = field(body, "stream_options"); so && so->is_object())
        include_usage = bool_field(*so, "include_usage", false);
      job->sampling = sampling_from(body, opt_.sampling);
      job->stop = stop_strings(body);
      const std::string text =
          chat ? chat_prompt(body, opt_.default_system) : completion_prompt(body);
      prepare(*job, body, text, chat ? "messages" : "prompt");
    } catch (const ApiError& e) {
      send_error(res, e.status, e.what(), e.param, e.code);
      return;
    } catch (const std::exception& e) {  // malformed JSON, or a field of the wrong type
      send_error(res, 400, std::string("invalid request: ") + e.what());
      return;
    }
    job->id = (chat ? "chatcmpl-" : "cmpl-") + std::to_string(++next_id_);
    engine_.submit(job);
    if (stream) stream_reply(res, job, chat, include_usage);
    else full_reply(req, res, *job, chat);
  }

  // Tokenizes the prompt and fits the reply into the context.
  void prepare(Job& job, const json& body, const std::string& text, const char* prompt_field) {
    {
      std::lock_guard<std::mutex> lock(tok_mu_);  // encode() fills a cache: one thread at a time
      job.prompt = tok_.encode(text);
    }
    const int n = int(job.prompt.size());
    if (n == 0) throw bad_field(prompt_field, "is empty");
    if (n >= context_)
      throw ApiError(400,
                     "the prompt is " + std::to_string(n) + " tokens, but this server's context " +
                         "holds " + std::to_string(context_) +
                         " tokens including the reply (start it with a larger --ctx)",
                     prompt_field, "context_length_exceeded");
    int64_t want = int_field(body, "max_tokens", opt_.default_max_tokens, 1);
    want = int_field(body, "max_completion_tokens", want, 1);  // the newer name wins
    // A reply that would overflow the context is cut short (finish_reason "length").
    job.max_new = int(std::min<int64_t>(want, context_ - n));
  }

  json usage(const Job& job) const {
    const int p = int(job.prompt.size());
    return {{"prompt_tokens", p}, {"completion_tokens", job.completion_tokens},
            {"total_tokens", p + job.completion_tokens}};
  }

  json base(const Job& job, const char* object) const {
    return {{"id", job.id}, {"object", object}, {"created", job.created}, {"model", opt_.model_id}};
  }

  void full_reply(const httplib::Request& req, httplib::Response& res, Job& job, bool chat) {
    std::string text;
    bool done = false;
    while (!done) {
      text += job.take(milliseconds(200), done);
      if (!done && req.is_connection_closed()) {  // nobody is waiting for the answer any more
        job.cancelled = true;
        return;
      }
    }
    if (job.finish_reason != "stop" && job.finish_reason != "length") {
      send_error(res, 500, job.error.empty() ? "request " + job.finish_reason : job.error);
      return;
    }
    json choice = {{"index", 0}};
    if (chat) choice["message"] = {{"role", "assistant"}, {"content", text}};
    else choice["text"] = text;
    choice["logprobs"] = nullptr;
    choice["finish_reason"] = job.finish_reason;
    json reply = base(job, chat ? "chat.completion" : "text_completion");
    reply["choices"] = json::array({choice});
    reply["usage"] = usage(job);
    res.set_content(dump(reply), "application/json");
  }

  // Server-Sent Events: "data: <chunk JSON>\n\n" per piece of text, then "data: [DONE]\n\n".
  void stream_reply(httplib::Response& res, std::shared_ptr<Job> job, bool chat,
                    bool include_usage) {
    res.set_header("Cache-Control", "no-cache");
    auto chunk = [this, job, chat](const json& delta_or_text, const json& finish_reason) {
      json choice = {{"index", 0}};
      if (chat) choice["delta"] = delta_or_text;
      else choice["text"] = delta_or_text;
      choice["logprobs"] = nullptr;
      choice["finish_reason"] = finish_reason;
      json c = base(*job, chat ? "chat.completion.chunk" : "text_completion");
      c["choices"] = json::array({choice});
      return c;
    };
    auto provider = [this, job, chat, include_usage, chunk, started = false](
                        size_t, httplib::DataSink& sink) mutable -> bool {
      auto event = [&](const json& j) {
        const std::string s = "data: " + dump(j) + "\n\n";
        return sink.write(s.data(), s.size());
      };
      auto gone = [&] {  // the client disconnected: the engine drops the request
        job->cancelled = true;
        return false;
      };
      if (!started) {
        started = true;
        if (chat) return event(chunk({{"role", "assistant"}, {"content", ""}}, nullptr)) || gone();
      }
      std::string text;
      bool done = false;
      while (text.empty() && !done) {
        text = job->take(milliseconds(200), done);
        if (text.empty() && !done && !sink.is_writable()) return gone();
      }
      if (!text.empty()) {
        const json piece = chat ? json{{"content", text}} : json(text);
        if (!event(chunk(piece, nullptr))) return gone();
      }
      if (done) {
        if (job->finish_reason == "stop" || job->finish_reason == "length") {
          event(chunk(chat ? json::object() : json(""), job->finish_reason));
          if (include_usage) {
            json u = base(*job, chat ? "chat.completion.chunk" : "text_completion");
            u["choices"] = json::array();
            u["usage"] = usage(*job);
            event(u);
          }
        } else {
          const std::string message =
              job->error.empty() ? "request " + job->finish_reason : job->error;
          event({{"error", {{"message", message}, {"type", "server_error"}, {"param", nullptr},
                            {"code", nullptr}}}});
        }
        const char* end = "data: [DONE]\n\n";
        sink.write(end, std::strlen(end));
        sink.done();
      }
      return true;
    };
    res.set_chunked_content_provider("text/event-stream", provider, [job](bool success) {
      if (!success) job->cancelled = true;
    });
  }

  const Tokenizer& tok_;
  std::mutex tok_mu_;
  const ServerOptions opt_;
  const int max_batch_, total_pages_, context_;
  const int64_t started_;
  std::atomic<long long> next_id_{0};
  Engine engine_;
};

// Sizes the GPU buffers and records the decode graph of every batch size up
// front, so the first requests don't pay for it.
void warm_up(BatchRunner& eng, const Tokenizer& tok) {
  const std::vector<int32_t> word = tok.encode(" the");
  const int32_t t = word.empty() ? 0 : word[0];
  const int P = eng.page_size(), steps = eng.max_batch();
  int budget = eng.free_pages();
  std::vector<int> seqs;
  for (int i = 0; i < eng.max_batch(); ++i) {
    // Each sequence also gets one token per decode step below (up to max_batch of them).
    const int len = i == 0 ? std::min(512, eng.max_seq_len() - steps) : 1;
    const int pages = (len + steps + P - 1) / P;
    if (len < 1 || len + steps > eng.max_seq_len() || pages > budget) break;
    budget -= pages;
    const int s = eng.open_sequence();
    if (s < 0) break;
    eng.prefill_greedy(s, std::vector<int32_t>(size_t(len), t));
    seqs.push_back(s);
  }
  while (!seqs.empty()) {
    eng.decode_greedy(seqs, std::vector<int32_t>(seqs.size(), t));
    eng.close_sequence(seqs.back());
    seqs.pop_back();
  }
}

}  // namespace

void run_server(BatchRunner& engine, const Tokenizer& tok, const ServerOptions& opt) {
  warm_up(engine, tok);
  Api api(engine, tok, opt);
  httplib::Server svr;
  // Each request holds an HTTP thread until its reply is complete, so allow
  // more threads than requests that can run at once (the rest wait in the queue).
  const size_t threads = size_t(engine.max_batch()) + 8;
  svr.new_task_queue = [threads] { return new httplib::ThreadPool(threads, 4 * threads); };
  api.routes(svr);
  svr.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
    if (!res.body.empty()) return httplib::Server::HandlerResponse::Unhandled;
    send_error(res, res.status, res.status == 404 ? "no route for " + req.method + " " + req.path
                                                  : "HTTP error " + std::to_string(res.status));
    return httplib::Server::HandlerResponse::Handled;
  });
  svr.set_exception_handler([](const httplib::Request&, httplib::Response& res,
                               std::exception_ptr ep) {
    std::string what = "internal error";
    try {
      if (ep) std::rethrow_exception(ep);
    } catch (const std::exception& e) {
      what = e.what();
    } catch (...) {
    }
    send_error(res, 500, what);
  });

  if (!svr.bind_to_port(opt.host, opt.port))
    throw std::runtime_error("cannot listen on " + opt.host + ":" + std::to_string(opt.port) +
                             " (is another server using the port?)");
  std::fprintf(stderr,
               "[serving %s on http://%s:%d | max %d concurrent requests | KV cache %d pages of %d "
               "tokens | context %d tokens]\n",
               opt.model_id.c_str(), opt.host.c_str(), opt.port, engine.max_batch(),
               engine.total_pages(), engine.page_size(),
               std::min(engine.max_seq_len(), engine.total_pages() * engine.page_size()));
  if (!svr.listen_after_bind()) throw std::runtime_error("the HTTP server stopped unexpectedly");
}

}  // namespace ie
