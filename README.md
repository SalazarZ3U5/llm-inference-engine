# Inference Engine

An LLM inference engine written from scratch in C++17 and CUDA, with no ML frameworks.
It loads Hugging Face `safetensors` weights and generates text on an NVIDIA GPU.

Target model: **Qwen2.5-0.5B-Instruct** (download needs no license approval). Development GPU: RTX 4060 (sm_89, 8 GB).

> Status: **Phase 1 complete.** Text generation and multi-turn chat on the CPU with a KV cache,
> a from-scratch BPE tokenizer and sampling. Output matches Hugging Face at every layer, the tokenizer
> matches on 155k+ test tokens, and greedy generation reproduces HF token for token.
> Next: Phase 2, CPU performance.

## Layout

```
CMakeLists.txt          build (CUDA on by default; -DIE_ENABLE_CUDA=OFF for CPU-only)
CMakePresets.json       release / debug / cpu-only presets (Ninja + MSVC)
.clangd                 editor config so clangd understands .cu files
src/
  main.cpp              `ie` CLI: generate, chat, tokenize, verify, test-tokenizer, inspect, gpu-check
  safetensors.{h,cpp}   weight-file reader (header parse + validation + lazy tensor reads)
  npy.{h,cpp}           .npy reader for reference activations + array comparison
  config.{h,cpp}        Config struct loaded from config.json
  weights.{h,cpp}       all weights as FP32, shape-checked against the config
  ops.{h,cpp}           CPU reference ops: linear, rmsnorm, rope, softmax, silu, attention
  model.{h,cpp}         forward pass with KV cache: embedding -> 24 layers -> final norm -> logits
  tokenizer.{h,cpp}     byte-level BPE tokenizer (Qwen2 pre-split rules, special tokens, streaming decode)
  unicode_tables.h      letter/number code-point ranges for the tokenizer (generated)
  sampler.{h,cpp}       temperature, top-k, top-p, repetition penalty; defaults from generation_config.json
  cuda/
    cuda_check.cuh      CUDA_CHECK error macro
    device_check.cu     GPU info, vector-add test, memory-bandwidth test
scripts/
  download_model.py     fetch weights + tokenizer into models/
  reference.py          dump ground-truth activations from HF transformers into reference/
  make_tokenizer_cases.py   expected token ids from HF's tokenizer -> tests/tokenizer_cases.json
  gen_unicode_tables.py     regenerate src/unicode_tables.h
tests/
  tokenizer_cases.json  347 strings (English, code, Hindi, Gujarati, CJK, emoji, random) + expected ids
```

## Setup (Windows)

Prerequisites: NVIDIA driver, Visual Studio 2022 with "Desktop development with C++", the CUDA Toolkit, Git and Python.

### 1. Build

From the **x64 Native Tools Command Prompt for VS 2022**, in this folder:

```
cmake --preset release
cmake --build --preset release
build\release\ie.exe gpu-check
```

In Antigravity or VS Code with the CMake Tools extension, pick the `CUDA Release` preset and build.
The preset sets up the MSVC environment itself.

### 2. Python reference tools

```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r scripts\requirements.txt
python scripts\download_model.py
python scripts\reference.py
```

### 3. Look inside the model

```
build\release\ie.exe inspect models\qwen2.5-0.5b-instruct\model.safetensors
```

### 4. Check the forward pass against the reference

```
build\release\ie.exe verify models\qwen2.5-0.5b-instruct reference\default
```

Runs the prompt through the engine and compares every stage (embedding, each of the 24 layers,
final norm, logits) with the Hugging Face reference, then checks the tokenizer and 20 tokens of greedy
generation through the KV cache. Every row should say PASS.

```
build\release\ie.exe test-tokenizer models\qwen2.5-0.5b-instruct tests\tokenizer_cases.json
```

### 5. Generate text and chat

```
build\release\ie.exe generate models\qwen2.5-0.5b-instruct "Once upon a time"
build\release\ie.exe generate models\qwen2.5-0.5b-instruct "Explain a KV cache in one sentence." --chat
build\release\ie.exe chat models\qwen2.5-0.5b-instruct
```

Options: `--max-tokens N`, `--temp T`, `--top-k K`, `--top-p P`, `--rep R`, `--seed S`, `--greedy`,
`--ctx N` (KV cache size), `--system "..."` (chat). Sampling defaults come from the model's
`generation_config.json`. In chat, `/reset` starts a new conversation and `/exit` quits.

## Roadmap

### Phase 0: correct forward pass (CPU, FP32) ✅
- [x] Load config.json (hidden size, heads, KV heads, layers, eps, rope theta)
- [x] Load weights; convert BF16 to FP32 (BF16 is the top 16 bits of an FP32)
- [x] Embedding lookup: matches `reference/default/embed.npy` exactly
- [x] RMSNorm
- [x] Q/K/V projections with Qwen2's q/k/v biases (optional, so Llama-style models also load)
- [x] RoPE (rotate-half variant as used by HF; theta from config)
- [x] Grouped-query attention with causal mask (7 query heads share each K/V head)
- [x] SwiGLU MLP: `down(silu(gate(x)) * up(x))`
- [x] Final norm + LM head tied to `embed_tokens`
- [x] `ie verify`: all 24 layers, final norm and logits within 1e-4 relative error; greedy next token = 12095 (" Paris")

### Phase 1: generation ✅
- [x] KV cache (keys stored after RoPE); cached decode is bit-identical to a full recompute
- [x] Greedy decode reproduces `generated_ids.npy` exactly (20/20 tokens)
- [x] Sampling: temperature, top-k, top-p, repetition penalty (HF order and defaults)
- [x] Byte-level BPE tokenizer: matches HF on 5,000+ strings / 155k tokens incl. Hindi, CJK, emoji, code
- [x] `generate` and multi-turn `chat` with Qwen's ChatML template, streaming UTF-8 output

### Phase 2: CPU performance
- [ ] Multithreaded matmul, SIMD (AVX2)
- [ ] INT8 weight quantization
- [ ] Benchmark harness: prefill tok/s, decode tok/s, time to first token

### Phase 3: CUDA
- [ ] Weights resident on GPU in FP16/BF16
- [ ] cuBLAS GEMM baseline end to end
- [ ] Custom kernels: RMSNorm, RoPE, softmax, fused SiLU*up
- [ ] Own GEMV kernel for decode (memory-bound): approach the `gpu-check` bandwidth
- [ ] Profile with Nsight Systems / Nsight Compute; document each optimization

### Phase 4: advanced
- [ ] Flash-style fused attention
- [ ] INT4 weight-only quantization
- [ ] Continuous batching + paged KV cache
- [ ] Speculative decoding

### Phase 5: present it
- [ ] OpenAI-compatible HTTP server with streaming
- [ ] Benchmarks vs llama.cpp / PyTorch on the same GPU, with charts in this README

## Performance log

| Date | Change | Decode tok/s | Prefill tok/s | Notes |
|------|--------|-------------:|--------------:|-------|
| 2026-11-09 | Phase 1 baseline: CPU, FP32, single thread, plain loops | 2.84 | 4.0 | `chat`: 30-token prompt, 53 tokens generated |

The decode speed limit is roughly memory bandwidth ÷ model size in bytes.
`ie gpu-check` measures the bandwidth.
