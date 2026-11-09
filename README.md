# Inference Engine

An LLM inference engine written from scratch in C++17 and CUDA, with no ML frameworks.
It loads Hugging Face `safetensors` weights and generates text on an NVIDIA GPU.

Target model: **Qwen2.5-0.5B-Instruct** (download needs no license approval). Development GPU: RTX 4060 (sm_89, 8 GB).

> Status: Phase 0, setup. `ie gpu-check` and `ie inspect` work. The forward pass is next.

## Layout

```
CMakeLists.txt          build (CUDA on by default; -DIE_ENABLE_CUDA=OFF for CPU-only)
CMakePresets.json       release / debug / cpu-only presets (Ninja + MSVC)
.clangd                 editor config so clangd understands .cu files
src/
  main.cpp              `ie` CLI
  safetensors.{h,cpp}   weight-file reader (header parse + validation + lazy tensor reads)
  cuda/
    cuda_check.cuh      CUDA_CHECK error macro
    device_check.cu     GPU info, vector-add test, memory-bandwidth test
scripts/
  download_model.py     fetch weights + tokenizer into models/
  reference.py          dump ground-truth activations from HF transformers into reference/
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

## Roadmap

### Phase 0: correct forward pass (CPU, FP32)
- [ ] Load config.json (hidden size, heads, KV heads, layers, eps, rope theta)
- [ ] Load weights; convert BF16 to FP32 (BF16 is the top 16 bits of an FP32)
- [ ] Embedding lookup: compare with `reference/default/embed.npy`
- [ ] RMSNorm
- [ ] Q/K/V projections. **Qwen2 has biases on q/k/v**; Llama does not
- [ ] RoPE (rotate-half variant as used by HF; theta from config)
- [ ] Grouped-query attention with causal mask (several Q heads share each K/V head)
- [ ] SwiGLU MLP: `down(silu(gate(x)) * up(x))`
- [ ] Final norm + LM head (**weights tied to `embed_tokens`**: there may be no separate `lm_head.weight`)
- [ ] Each `layer_XX.npy` matches to about 1e-4; `logits.npy` argmax matches

### Phase 1: generation
- [ ] KV cache
- [ ] Greedy decode reproduces `generated_ids.npy` exactly
- [ ] Sampling: temperature, top-k, top-p
- [ ] BPE tokenizer from `tokenizer.json`: encode matches `input_ids.npy`
- [ ] Interactive chat CLI using the chat template

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
|      |        |              |               |       |

The decode speed limit is roughly memory bandwidth ÷ model size in bytes.
`ie gpu-check` measures the bandwidth.
