#pragma once
// An OpenAI-compatible HTTP server on top of a BatchRunner.
//
//   POST /v1/chat/completions   messages -> Qwen's ChatML template -> reply (optionally streamed)
//   POST /v1/completions        raw prompt -> continuation
//   GET  /v1/models             the one model being served
//   GET  /health                liveness + scheduler state
//
// Threads: the HTTP library runs each connection on a worker thread, which
// parses the request, tokenizes it and hands it to a queue. One engine thread
// owns the BatchRunner (it is not thread-safe) and runs the continuous-batching
// loop: admit waiting requests, prefill them, then advance every running
// request by one token per decode step. Text flows back to each request's
// HTTP thread through its own small mutex-protected buffer.
//
// Nothing here depends on CUDA: the server only needs a BatchRunner&, so it
// builds in the CPU-only configuration and can be tested with a fake engine.

#include <string>

#include "runner.h"
#include "sampler.h"
#include "tokenizer.h"

namespace ie {

struct ServerOptions {
  std::string host = "127.0.0.1";
  int port = 8080;
  std::string model_id;          // reported by /v1/models and in every response
  std::string default_system;    // used when a chat request has no system message
  SamplingParams sampling;       // defaults for fields a request leaves out
  int default_max_tokens = 256;  // when a request sets neither max_tokens nor max_completion_tokens
};

// Serves until the process ends. Throws if the address cannot be bound.
// `engine` must not be used by anything else while this runs.
void run_server(BatchRunner& engine, const Tokenizer& tok, const ServerOptions& opt);

}  // namespace ie
