# Inference Engine

An LLM inference engine written from scratch in C++17 and CUDA, with no ML frameworks.
It loads Hugging Face `safetensors` weights and generates text on an NVIDIA GPU.

Target model: **Qwen2.5-0.5B-Instruct** (download needs no license approval). Development GPU: RTX 4060 (sm_89, 8 GB).

> Status: **Phase 3 (CUDA) nearly complete.** The whole forward pass runs on an RTX 4060 with
> hand-written CUDA kernels and CUDA graphs: **213.8 tok/s decode with BF16** (211 GB/s, 97% of the
> measured memory bandwidth) and **361.6 tok/s with INT8**, 127x the first working version, while still
> matching Hugging Face at every layer. Next: tensor-core prefill and profiling, then Phase 4.

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
  matrix.{h,cpp}        weight matrices as f32 / bf16 / int8; blocked matmul with scalar and AVX2 kernels
  threadpool.{h,cpp}    fixed thread pool for parallel loops (spins briefly, then sleeps)
  runner.{h,cpp}        Runner interface (weights + KV cache on one device); CPU implementation
  cuda/
    cuda_check.cuh      CUDA_CHECK error macro
    device_check.cu     GPU info, vector-add test, memory-bandwidth test
    kernels.{cuh,cu}    GPU kernels: embedding, RMSNorm, BF16 matmul (warp per row), fused
                        gate/up/SiLU, RoPE + KV-cache write, attention
    cuda_runner.{h,cu}  GPU Runner: BF16 weights and KV cache in VRAM; `test-cuda` self-test
scripts/
  download_model.py     fetch weights + tokenizer into models/
  reference.py          dump ground-truth activations from HF transformers into reference/
  make_tokenizer_cases.py   expected token ids from HF's tokenizer -> tests/tokenizer_cases.json
  gen_unicode_tables.py     regenerate src/unicode_tables.h
tests/
  tokenizer_cases.json  347 strings (English, code, Hindi, Gujarati, CJK, emoji, random) + expected ids
  perplexity.txt        fixed ~700-token text for measuring model quality
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

Performance options for every command that runs the model: `--weights f32|bf16|int8` (default `bf16`),
`--threads N` (default: one per physical core), `--kernel scalar|avx2` (default `avx2`).

### 6. Measure speed and quality

```
build\release\ie.exe bench models\qwen2.5-0.5b-instruct --sweep
build\release\ie.exe perplexity models\qwen2.5-0.5b-instruct tests\perplexity.txt --weights int8
build\release\ie.exe test-kernels
build\release\ie.exe test-cuda
```

`bench --sweep` walks through every optimization step (scalar → AVX2 → threads → BF16 → INT8) and
prints a table of prefill and decode speed. `perplexity` scores how well the model predicts a fixed
text (lower is better), which is how INT8 is checked against BF16. `test-kernels` checks every matmul
kernel against the plain reference implementation, and `test-cuda` does the same for every GPU kernel.

Every command that runs the model takes `--device cpu|cuda`. The default is the GPU when the build has
CUDA and one is present.

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

### Phase 2: CPU performance ✅
- [x] Benchmark harness (`ie bench`): prefill tok/s, decode tok/s, achieved memory bandwidth
- [x] Thread pool; matmul split across output rows, attention across heads
- [x] Cache blocking: 16 weight rows x 4 tokens, so each weight is read once per call during prefill
- [x] AVX2 + FMA kernels for f32, bf16 and int8 weights (216 kernel tests against the reference)
- [x] BF16 weight storage: half the memory traffic, still exact; `ie verify` passes
- [x] INT8 per-row quantization: perplexity 14.85 vs 14.78 for BF16 (+0.5%), same greedy output

### Phase 3: CUDA
- [x] Runner interface so every command runs on CPU or GPU (`--device`)
- [x] BF16 weights and FP32 KV cache resident in VRAM; whole forward pass on the GPU
- [x] Own kernels: BF16 matmul (one warp per row, coalesced 16-byte loads), RMSNorm, RoPE + cache
      write, attention, fused gate/up/SiLU, residual adds fused into the matmuls
- [x] `ie verify --device cuda` passes (30/30); `test-cuda` checks every kernel against the CPU code (96/96)
- [x] CUDA graphs: each decode step (~243 kernels) recorded once and replayed with one call (+8%)
- [x] INT8 weights on the GPU (per-row scales); perplexity identical to the CPU INT8 path (14.8453)
- [ ] Tensor-core GEMM for prefill
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
| 2026-11-10 | Cache-blocked matmul (16 rows x 4 tokens), scalar, 1 thread | 2.7 | 14.2 | prefill 3.5x from weight reuse alone |
| 2026-11-10 | AVX2 + FMA kernels, 1 thread | 11.1 | 70.6 | 8 multiply-adds per instruction |
| 2026-11-10 | Thread pool, 6 threads (one per physical core) | 17.0 | 184.7 | decode now limited by RAM bandwidth (33.6 GB/s) |
| 2026-11-10 | BF16 weight storage, 6 threads | 31.5 | 207.2 | half the bytes per token, still exact |
| 2026-11-10 | INT8 weights (per-row scales), 6 threads | 59.8 | 222.9 | quarter of FP32's bytes; perplexity +0.5% |
| 2026-11-10 | GPU: hand-written CUDA kernels, BF16, RTX 4060 | 190.8 | 1752.2 | 188.5 GB/s = 87% of measured VRAM bandwidth |
| 2026-11-11 | GPU: CUDA graphs for decode, BF16 | 213.8 | 1763.8 | 211.3 GB/s = 97% of measured VRAM bandwidth |
| 2026-11-11 | **GPU: INT8 weights + CUDA graphs** | **361.6** | 1100.5 | 127x the Phase 1 baseline; perplexity +0.5% |

Full sweep (`ie bench --sweep`, 64-token prompt, 32 generated tokens) on a Ryzen 5 3600XT
(6 cores / 12 threads, AVX2, dual-channel DDR4) and an RTX 4060 (8 GB):

| Device | Weights | Kernel | Threads | Prefill tok/s | Decode tok/s | Decode GB/s |
|--------|---------|--------|--------:|--------------:|-------------:|------------:|
| cpu | f32 | scalar | 1 | 14.6 | 2.9 | 5.7 |
| cpu | f32 | avx2 | 1 | 74.8 | 13.1 | 25.8 |
| cpu | f32 | avx2 | 6 | 299.9 | 20.1 | 39.7 |
| cpu | f32 | avx2 | 12 | 300.6 | 17.1 | 33.8 |
| cpu | bf16 | avx2 | 6 | 330.6 | 38.3 | 37.8 |
| cpu | bf16 | avx2 | 12 | 339.4 | 32.0 | 31.6 |
| cpu | int8 | avx2 | 6 | 359.7 | 70.6 | 35.0 |
| cpu | int8 | avx2 | 12 | 369.7 | 59.5 | 29.5 |
| cuda | bf16 | cuda | - | 1529.2 | 197.4 | 195.1 |
| cuda | bf16 | cuda graph | - | 1763.8 | 213.8 | 211.3 |
| cuda | int8 | cuda graph | - | 1100.5 | 361.6 | 179.3 |

CPU numbers vary by roughly 10-20% between runs with background load (the per-step rows above the
table come from an earlier run). INT8 on the GPU no longer saturates memory bandwidth: converting
int8 weights to floats now costs as much time as reading them, which also makes INT8 prefill slower
than BF16. Quantizing activations too and using int8 dot-product instructions (`dp4a`) is the next step.

Using all 12 hardware threads helps prefill (large, compute-bound calls) but cuts decode by up to 58%.
Decode makes ~200 small parallel calls per token and each waits for its slowest thread; with every
hyper-thread busy, the OS has to pause one of them now and then, and every call waits for it. The engine
therefore defaults to one thread per physical core.

The decode speed limit is roughly memory bandwidth ÷ model size in bytes.
`ie gpu-check` measures the bandwidth.
