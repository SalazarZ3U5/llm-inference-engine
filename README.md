# Inference Engine

An LLM inference engine written from scratch in C++17 and CUDA, with no ML frameworks.
It loads Hugging Face `safetensors` weights and generates text on an NVIDIA GPU.

Target model: **Qwen2.5-0.5B-Instruct** (download needs no license approval). Development GPU: RTX 4060 (sm_89, 8 GB).

> Status: **Phase 4 (serving) in progress.** The whole forward pass runs on an RTX 4060 with
> hand-written CUDA kernels and CUDA graphs: **213.8 tok/s decode with BF16** (211 GB/s, 97% of the
> measured memory bandwidth) and **361.6 tok/s with INT8**, 127x the first working version, while still
> matching Hugging Face at every layer. Prefill uses tensor cores and FlashAttention-style tiled
> attention (**13,148 tok/s** on a 512-token prompt, **9,432 tok/s** on 4,000 tokens), and
> flash-decoding keeps decode fast at long context (**~180 tok/s after a 4,000-token prompt**). A paged
> KV cache and continuous batching serve many requests at once: **2,346 tok/s across 64 concurrent
> requests**, 12.2x serving them one at a time, with every request's output identical to serving it alone.

## Layout

```
CMakeLists.txt          build (CUDA on by default; -DIE_ENABLE_CUDA=OFF for CPU-only)
CMakePresets.json       release / debug / cpu-only presets (Ninja + MSVC)
.clangd                 editor config so clangd understands .cu files
src/
  main.cpp              `ie` CLI: generate, chat, tokenize, verify, bench, serve-bench, perplexity, ...
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
  runner.{h,cpp}        Runner interface (weights + KV cache on one device); CPU implementation;
                        BatchRunner interface (many sequences, paged KV cache)
  cuda/
    cuda_check.cuh      CUDA_CHECK error macro
    device_check.cu     GPU info, vector-add test, memory-bandwidth test
    kernels.{cuh,cu}    GPU kernels: embedding, RMSNorm, matmul (warp per row; tensor cores for
                        prefill), fused gate/up/SiLU, RoPE + paged KV-cache write, flash-decoding
                        attention over paged caches, row gather, argmax
    cuda_runner.{h,cu}  GPU engine (Runner + BatchRunner): weights and paged KV cache in VRAM,
                        one CUDA graph per batch size; `test-cuda` self-test
scripts/
  download_model.py     fetch weights + tokenizer into models/
  profile.bat           Nsight Systems profiles of decode, long-context decode and serving
  profile_summary.py    per-kernel GPU time from a profile, split by phase (prefill / decode / batch)
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
build\release\ie.exe serve-bench models\qwen2.5-0.5b-instruct --prompt-tokens 512 --gen-tokens 256 --check
```

`bench --sweep` walks through every optimization step (scalar → AVX2 → threads → BF16 → INT8) and
prints a table of prefill and decode speed. `perplexity` scores how well the model predicts a fixed
text (lower is better), which is how INT8 is checked against BF16. `test-kernels` checks every matmul
kernel against the plain reference implementation, and `test-cuda` does the same for every GPU kernel.
`serve-bench` sends 64 requests of varied lengths at once and serves them at several concurrency levels
(`--concurrency 1,4,16,32,64`), reporting throughput, time to first token and latency; `--check` confirms
every request gets exactly the tokens it gets when served alone.

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
- [x] `ie verify --device cuda` passes (30/30); `test-cuda` checks every kernel against the CPU code (758 tests, incl. paged caches)
- [x] CUDA graphs: each decode step (~243 kernels) recorded once and replayed with one call (+8%)
- [x] INT8 weights on the GPU (per-row scales); perplexity identical to the CPU INT8 path (14.8453)
- [x] Tensor-core GEMM for prefill (WMMA, 64x64 tiles): FP32 activations split into two BF16 parts
      (hi + lo) to keep FP32-level accuracy; BF16 and INT8 weights; 3.5x faster prefill, same perplexity
- [x] Profile with Nsight Systems (`scripts\profile.bat`): per-kernel time for every phase, below

### Phase 4: advanced
- [x] Flash-decoding attention: cached positions split into 256-token chunks across blocks, partial
      softmax per chunk, exact merge; no context-length limit; works with CUDA graphs
- [x] Prefill attention: each block shares K/V tiles between 16 prompt tokens and all 7 query heads
      of a KV head, with an online softmax (FlashAttention-style, CUDA cores): 4,000-token prefill 4.0x
- [x] Decode attention: one block per KV head and 64-position chunk, reading each K/V row once for
      all 7 query heads (decode attention -42% at 512 tokens; -30% at 64 sequences)
- [x] Tensor-core matmul: the next tile's loads overlap the current tile's math (two shared-memory
      buffers, one barrier per step): prefill +15%
- [ ] Tensor-core (WMMA) version of the prefill attention
- [ ] INT4 weight-only quantization
- [x] Paged KV cache: 16-token pages from a shared pool, one page table per sequence; memory follows
      actual lengths instead of max batch x max length
- [x] Batched decode: one token for each of up to 64 sequences per forward pass, weights read once for
      all of them; one CUDA graph per batch size; greedy argmax on the GPU
- [x] Continuous batching scheduler: requests join as soon as a slot and pages are free, and leave
      the moment they finish (`ie serve-bench`)
- [x] Batching never changes results: batches under 32 run each row exactly as it would alone
      (`serve-bench --check`, and a bit-for-bit test in `test-cuda`)
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
| 2026-11-11 | GPU: tensor-core prefill, BF16, 512-token prompt | 181.9 | 5507.9 | vs 1571.7 on CUDA cores (3.5x); decode is slower at this context length, see below |
| 2026-11-11 | GPU: flash-decoding attention, 512-token prompt | 191.4 | **7895.8** | attention spread over chunks: +5% decode, +43% prefill |
| 2026-11-11 | GPU: flash-decoding attention, 4,000-token prompt | 184.5 | 2492.9 | 8x the context costs only 3.6% decode speed |
| 2026-11-11 | GPU: paged KV cache, 512-token prompt | 187.8 | 7401.5 | page lookups hoisted out of the attention loop + 4 loads in flight per warp |
| 2026-11-11 | GPU: paged KV cache, 4,000-token prompt | 174.5 | 2174.9 | single-stream cost of paging, see below |
| 2026-11-11 | GPU: continuous batching, 64 concurrent requests | 1371.6 | - | total across all requests: 7.4x one at a time, output identical |
| 2026-11-11 | GPU: profile-guided batched matmuls (split-K, 4 rows per warp) | 1766.8 | - | 64 requests: 9.4x one at a time; 16 requests 802 → 1,046 |
| 2026-11-11 | GPU: tiled prefill attention (shared K/V tiles, online softmax), 512-token prompt | 189.2 | **11481.2** | attention 24 ms → 5.4 ms per prompt |
| 2026-11-11 | GPU: tiled prefill attention, 4,000-token prompt | 179.9 | **8731.6** | 4.0x prefill: attention 1.53 s → 0.19 s |
| 2026-11-11 | GPU: same, serving 64 concurrent requests | 1975.5 | - | 10.6x one at a time; time to first token 1.65 s → 1.24 s |
| 2026-11-11 | GPU: GQA-shared decode attention + pipelined tensor-core matmul, 512-token prompt | 186.4 | **13147.6** | decode attention 687 → 397 us per step |
| 2026-11-11 | GPU: same, 4,000-token prompt | 180.9 | 9432.4 | |
| 2026-11-11 | **GPU: same, serving 64 concurrent requests** | **2346.3** | - | 12.2x one at a time; 64-sequence step 16.4 → 12.9 ms |

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

At longer context, decode used to slow down (213.8 tok/s after a 64-token prompt, 181.9 after 512)
because the first attention kernel looped over every cached position with one block per head.
Flash-decoding attention spreads the positions over many blocks: 191.4 tok/s after 512 tokens and
184.5 after 4,000. Prefill attention used the same kernel and read the whole cache once per prompt
token, so its cost grew with the square of the prompt (2,493 tok/s for 4,000 tokens); the tiled
prefill kernel (below) fixed that.

### Serving many requests (`ie serve-bench`)

64 requests arrive at once (prompts of 256-512 tokens, 128-256 generated tokens each, greedy, BF16).
Up to `Concurrency` of them run together; each decode step advances every running sequence by one
token, finished sequences leave immediately and waiting ones join (continuous batching). `--check`
compares every request's tokens with serving it alone.

| Concurrency | Throughput tok/s | Mean time to first token | Mean latency | p95 latency | Peak KV pages | Same as one at a time |
|------------:|-----------------:|-------------------------:|-------------:|------------:|--------------:|:---|
| 1 | 192.1 | 30.9 s | 31.9 s | 60.3 s | 48 | 64/64 |
| 4 | 605.2 | 9.3 s | 10.5 s | 19.3 s | 152 | 64/64 |
| 16 | 1,144.7 | 4.0 s | 6.4 s | 10.3 s | 556 | 64/64 |
| 32 | 1,645.1 | 2.0 s | 5.0 s | 7.2 s | 1,097 | 64/64 |
| 64 | **2,346.3** | 1.1 s | 4.5 s | 5.2 s | 2,146 | 64/64 |

Decode streams all the weights once per step whether it serves one sequence or 64, so throughput
grows with the batch: 3.2x at 4, 12.2x at 64. Time to first token drops from 31 s to 1.1 s because
requests no longer queue behind each other. Prompts are prefilled on their own between decode steps
(chunked prefill, which mixes them into decode steps, is a possible next step).

The KV cache is split into 16-token pages handed out as sequences grow (peak 2,146 pages = 805 MiB for
64 sequences, instead of reserving 64 x the maximum length). Batches under 32 rows run every matmul row
exactly as it would run alone, so batching cannot change any request's output; the tensor-core path
used for larger batches rounds differently in the last bits, but all 64 requests still matched here.

### Where the time goes (Nsight Systems)

`scripts\profile.bat` records every kernel with Nsight Systems; `scripts\profile_summary.py` splits the
time by the engine's NVTX ranges (prefill, decode, batched decode). Times per forward pass, BF16:

| Phase | Time | Biggest costs |
|-------|-----:|---------------|
| Decode, 1 sequence, 512 context | 5.5 ms | weight matmuls 87% (~238 GB/s while running; the card's peak is 272), attention 9% |
| Decode, 64 sequences | 12.9 ms | attention 32% (near the memory floor for an FP32 cache), gate/up 22%, other matmuls 24%, LM head 12% |
| Prefill, 512 tokens | 43 ms | tensor-core matmuls 81%, attention 15% |
| Prefill, 4,000 tokens | 0.43 s | attention 47%, tensor-core matmuls 51% |

Each row is the latest profile; the history (first profile: decode at 64 sequences 23.7 ms, prefill of
4,000 tokens 1.82 s) is in the performance log above.

The first profile changed the plan. Batches of 2-31 sequences gained almost nothing from batching (each
extra sequence added ~0.7 ms per step): the matmul kernel re-read every activation once per weight row
and ran out of on-chip bandwidth. Each warp now computes 4 weight rows (2 for gate/up) with every
activation it loads, still adding in the same order so results stay bit-identical to running alone
(+30% at 16 requests). At 64 sequences, a 896-row matmul made only 14 output tiles for 24 SMs, and the
128-row K/V projections only 2, each block walking the whole inner dimension alone (44 us for a 0.2 MB
matrix). Split-K now spreads small problems over 4 blocks per SM and adds the partial sums in a fixed
order (+29% at 64 requests, and 16% faster prefill as a side effect).

Prefill attention used the decode kernel: one block per query token, each reading the whole cache, so
a 4,000-token prompt read the cache 4,000 times (84% of prefill). The prefill kernel gives each block
16 consecutive prompt tokens and the 7 query heads that share a KV head, loads the cache 16 positions
at a time into shared memory, and lets all 112 queries use each tile; each thread keeps a running
softmax (max, sum, output) that it rescales when a new tile raises the max, as in FlashAttention.
Attention for a 4,000-token prompt went from 1.53 s to 0.19 s (perplexity 14.7774 → 14.7775: the same
sums in a different order). It is used for any pass of 16+ consecutive positions of one sequence, a
rule that depends only on the request, so batching still cannot change an output.

Decode attention then got the same treatment: one block per KV head (instead of per query head)
reads each cached row once for all 7 query heads, and chunks of 64 positions instead of 256 give even
one decoding sequence enough blocks to fill the GPU (attention per step 687 → 397 us at 512 tokens).
The tensor-core matmul now starts loading the next tile into registers before computing on the
current one and writes it into a second shared-memory buffer, so memory latency hides behind the math
(prefill +15%).

What's left: at 64 sequences, attention reads ~0.9 GB of FP32 K/V per step, close to what the memory
can deliver; storing the cache in 16-bit would halve that (and double how many tokens fit). Batches of
17-31 sequences still take the CUDA-core matmul path, which is slower there than tensor cores (a
24-sequence step takes 16 ms, a 32-sequence one 11.6 ms); it is kept below 32 because it gives
bit-identical results to running alone.

Paging is not free for a single stream: finding a position's page adds a lookup to the attention loop.
Loading each chunk's page ids once into shared memory and keeping 4 key/value loads in flight per warp
won most of it back. Against the contiguous cache the remaining cost is about 2% decode / 6% prefill
at 512 tokens and 5% / 13% at 4,000, measured in different sessions, where run-to-run variation on
this machine is a few percent.

Using all 12 hardware threads helps prefill (large, compute-bound calls) but cuts decode by up to 58%.
Decode makes ~200 small parallel calls per token and each waits for its slowest thread; with every
hyper-thread busy, the OS has to pause one of them now and then, and every call waits for it. The engine
therefore defaults to one thread per physical core.

The decode speed limit is roughly memory bandwidth ÷ model size in bytes.
`ie gpu-check` measures the bandwidth.
