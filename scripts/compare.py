"""Compare this engine with llama.cpp and PyTorch on the same GPU and the same model.

    python scripts/compare.py --llama-dir C:\\tools\\llama.cpp     # all three engines
    python scripts/compare.py --skip-torch --skip-llama            # only this engine
    python scripts/compare.py --charts-only                         # redraw from docs/benchmarks.json

Workloads (greedy decoding, every engine on the GPU, 16-bit weights and KV cache):
  - one request: prefill a 512- and a 4,000-token prompt, then decode 128 tokens after it
  - many requests at once: B sequences of 512 prompt + 128 generated tokens, all
    starting together (llama.cpp's llama-batched-bench workload), B = 1, 4, 16, 32, 64;
    reported as generated tokens per second over the whole run, prefill included
  - 8-bit weights, one request: this engine's INT8 vs llama.cpp's Q8_0

Engines:
  - this engine: `ie bench` and `ie serve-bench --same-length`, BF16 weights, FP16 KV cache
  - llama.cpp: a prebuilt CUDA release (llama-bench, llama-batched-bench), the official
    FP16 and Q8_0 GGUF files (downloaded into models/gguf once), flash attention on
  - PyTorch: Hugging Face transformers, BF16, SDPA attention, no torch.compile

Writes docs/benchmarks.json (merged with earlier runs, so engines can be run
separately), docs/bench_single.png, docs/bench_batched.png, and prints Markdown tables.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
RESULTS = DOCS / "benchmarks.json"

PROMPTS = [512, 4000]
GEN = 128
BATCHES = [1, 4, 16, 32, 64]
BATCH_PROMPT = 512

GGUF_REPO = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
GGUF_FILES = {"f16": "qwen2.5-0.5b-instruct-fp16.gguf", "q8_0": "qwen2.5-0.5b-instruct-q8_0.gguf"}

ENGINES = ["this engine", "llama.cpp", "PyTorch"]
LABELS = {  # what each engine runs, for the tables
    "this engine": "this engine (BF16)",
    "llama.cpp": "llama.cpp (FP16 GGUF)",
    "PyTorch": "PyTorch + transformers (BF16)",
}

# ------------------------------------------------------------------ helpers --


def run(cmd, timeout=3600):
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"exit code {p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return p.stdout + "\n" + p.stderr


def run_first(cmds, timeout=3600):
    """Runs the first command line that works (option spellings differ between llama.cpp builds)."""
    error = None
    for cmd in cmds:
        try:
            return run(cmd, timeout)
        except RuntimeError as e:
            error = e
            print("  (failed, trying the next form)", flush=True)
    raise error


def table_rows(text, first_cell):
    """Cells of the Markdown table rows whose first cell is `first_cell`."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|") and line.count("|") > 2:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cells[0] == first_cell:
                rows.append(cells)
    return rows


# ------------------------------------------------------------- this engine --


def bench_ie(ie, model, out):
    single = out.setdefault("single", {}).setdefault("this engine", {})
    for p in PROMPTS:
        text = run([ie, "bench", model, "--device", "cuda", "--weights", "bf16", "--kv", "f16",
                    "--prompt-tokens", p, "--gen-tokens", GEN])
        cells = table_rows(text, "cuda")[-1]  # | cuda | bf16 | kernel | - | prefill | decode | GB/s |
        single[f"prefill_{p}"] = float(cells[4])
        single[f"decode_{p}"] = float(cells[5])
    text = run([ie, "bench", model, "--device", "cuda", "--weights", "int8", "--kv", "f16",
                "--prompt-tokens", BATCH_PROMPT, "--gen-tokens", GEN])
    cells = table_rows(text, "cuda")[-1]
    out.setdefault("eight_bit", {})["this engine (INT8)"] = {
        "prefill_512": float(cells[4]), "decode_512": float(cells[5])}

    batched = out.setdefault("batched", {}).setdefault("this engine", {})
    for b in BATCHES:
        text = run([ie, "serve-bench", model, "--same-length", "--kv", "f16",
                    "--prompt-tokens", BATCH_PROMPT, "--gen-tokens", GEN,
                    "--requests", b, "--concurrency", b])
        cells = table_rows(text, str(b))[-1]  # | concurrency | peak | steps | time | tok/s | ...
        batched[str(b)] = float(cells[4])


# ---------------------------------------------------------------- llama.cpp --


def gguf_path(kind):
    path = ROOT / "models" / "gguf" / GGUF_FILES[kind]
    if not path.exists():
        from huggingface_hub import hf_hub_download
        print(f"downloading {GGUF_FILES[kind]} from {GGUF_REPO} ...", flush=True)
        hf_hub_download(GGUF_REPO, GGUF_FILES[kind], local_dir=path.parent)
    return path


def llama_exe(llama_dir, name):
    for candidate in (Path(llama_dir) / f"{name}.exe", Path(llama_dir) / name):
        if candidate.exists():
            return candidate
    sys.exit(f"{name} not found in {llama_dir} (point --llama-dir at the unzipped llama.cpp release)")


def llama_bench_ts(exe, gguf, args):
    """tokens/s of one llama-bench test, as reported in its JSON output."""
    base = [exe, "-m", gguf, "-ngl", "99", "-r", "5", "-o", "json", *args]
    text = run_first([base + ["-fa", "1"], base + ["-fa", "on"], base])
    data = json.loads(text[text.index("["): text.rindex("]") + 1])
    return float(data[-1]["avg_ts"])


def bench_llama(llama_dir, out):
    bench = llama_exe(llama_dir, "llama-bench")
    batched_bench = llama_exe(llama_dir, "llama-batched-bench")
    f16, q8 = gguf_path("f16"), gguf_path("q8_0")

    single = out.setdefault("single", {}).setdefault("llama.cpp", {})
    for p in PROMPTS:
        single[f"prefill_{p}"] = llama_bench_ts(bench, f16, ["-p", p, "-n", 0])
        # Decode measured after a prompt of the same length (-d: tokens already in the cache).
        single[f"decode_{p}"] = llama_bench_ts(bench, f16, ["-p", 0, "-n", GEN, "-d", p])
    out.setdefault("eight_bit", {})["llama.cpp (Q8_0)"] = {
        "prefill_512": llama_bench_ts(bench, q8, ["-p", BATCH_PROMPT, "-n", 0]),
        "decode_512": llama_bench_ts(bench, q8, ["-p", 0, "-n", GEN, "-d", BATCH_PROMPT]),
    }

    ctx = max(BATCHES) * (BATCH_PROMPT + GEN)
    base = [batched_bench, "-m", f16, "-ngl", "99", "-c", ctx, "-b", 2048, "-ub", 512,
            "-npp", BATCH_PROMPT, "-ntg", GEN, "-npl", ",".join(str(b) for b in BATCHES)]
    text = run_first([base + ["-fa", "on"], base + ["-fa"], base])
    # Rows: | PP | TG | B | N_KV | T_PP s | S_PP t/s | T_TG s | S_TG t/s | T s | S t/s |
    batched = out.setdefault("batched", {}).setdefault("llama.cpp", {})
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 10 and cells[0] == str(BATCH_PROMPT) and cells[1] == str(GEN):
            b, total_s = int(cells[2]), float(cells[8])
            batched[str(b)] = b * GEN / total_s  # generated tokens per second, prefill included


# ------------------------------------------------------------------ PyTorch --


def bench_torch(model_dir, out):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        sys.exit("PyTorch has no CUDA here: install a CUDA build (see the README) or use --skip-torch")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    text = (ROOT / "tests" / "perplexity.txt").read_text(encoding="utf-8")
    unit = tok(text)["input_ids"]

    def prompt(n, batch=1):
        ids = (unit * (n // len(unit) + 1))[:n]
        return torch.tensor([ids] * batch, device="cuda")

    def seconds(fn):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return time.perf_counter() - t

    single = out.setdefault("single", {}).setdefault("PyTorch", {})
    with torch.inference_mode():
        for p in PROMPTS:
            ids = prompt(p)
            model(ids)  # warm-up
            single[f"prefill_{p}"] = p / statistics.median(seconds(lambda: model(ids)) for _ in range(3))

            def decode():
                o = model(ids, use_cache=True)
                past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(GEN):
                    o = model(nxt, past_key_values=past, use_cache=True)
                    past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
                torch.cuda.synchronize()
                return time.perf_counter() - t

            decode()  # warm-up
            single[f"decode_{p}"] = GEN / decode()

        batched = out.setdefault("batched", {}).setdefault("PyTorch", {})
        greedy = dict(do_sample=False, temperature=None, top_p=None, top_k=None,
                      repetition_penalty=1.0, max_new_tokens=GEN, min_new_tokens=GEN,
                      pad_token_id=tok.eos_token_id)
        for b in BATCHES:
            ids = prompt(BATCH_PROMPT, b)
            mask = torch.ones_like(ids)
            model.generate(ids[:, :32], attention_mask=mask[:, :32], **{**greedy, "max_new_tokens": 4,
                                                                         "min_new_tokens": 4})
            t = seconds(lambda: model.generate(ids, attention_mask=mask, **greedy))
            batched[str(b)] = b * GEN / t


# ------------------------------------------------------------------- output --


def tables(r):
    lines = []
    s = r.get("single", {})
    lines.append("| Engine | Prefill 512 tok/s | Decode @512 tok/s | Prefill 4,000 tok/s | Decode @4,000 tok/s |")
    lines.append("|---|--:|--:|--:|--:|")
    for e in ENGINES:
        if e in s:
            v = s[e]
            lines.append(f"| {LABELS[e]} | {v['prefill_512']:,.0f} | {v['decode_512']:,.1f} | "
                         f"{v['prefill_4000']:,.0f} | {v['decode_4000']:,.1f} |")
    lines.append("")
    b = r.get("batched", {})
    lines.append("| Engine | " + " | ".join(f"{n} at once" for n in BATCHES) + " |")
    lines.append("|---|" + "--:|" * len(BATCHES))
    for e in ENGINES:
        if e in b:
            lines.append(f"| {LABELS[e]} | " + " | ".join(
                f"{b[e][str(n)]:,.0f}" if str(n) in b[e] else "-" for n in BATCHES) + " |")
    lines.append("")
    q = r.get("eight_bit", {})
    if q:
        lines.append("| 8-bit weights | Prefill 512 tok/s | Decode @512 tok/s |")
        lines.append("|---|--:|--:|")
        for name, v in q.items():
            lines.append(f"| {name} | {v['prefill_512']:,.0f} | {v['decode_512']:,.1f} |")
    return "\n".join(lines)


# Reference categorical palette, slots 1-3 in fixed order; text in ink colors, never series colors.
COLORS = {"this engine": "#2a78d6", "llama.cpp": "#eb6834", "PyTorch": "#1baf7a"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def charts(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK_2})
    thousands = FuncFormatter(lambda v, _: f"{v:,.0f}")
    gpu = r.get("gpu", "GPU")

    # One request: decode and prefill, two context lengths, one bar per engine.
    s = r.get("single", {})
    engines = [e for e in ENGINES if e in s]
    if engines:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE)
        for ax, (metric, title) in zip(axes, [("decode", "Decode, one request (tokens/s)"),
                                              ("prefill", "Prefill, one request (tokens/s)")]):
            style(ax)
            width = 0.8 / len(engines)
            for i, e in enumerate(engines):
                xs = [g + (i - (len(engines) - 1) / 2) * width for g in range(len(PROMPTS))]
                vals = [s[e][f"{metric}_{p}"] for p in PROMPTS]
                bars = ax.bar(xs, vals, width, color=COLORS[e], edgecolor=SURFACE, linewidth=2,
                              label=LABELS[e])
                for bar, v in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:,.0f}", ha="center",
                            va="bottom", fontsize=8.5, color=INK_2)
            ax.set_xticks(range(len(PROMPTS)))
            ax.set_xticklabels([f"{p:,}-token prompt" for p in PROMPTS])
            ax.yaxis.set_major_formatter(thousands)
            ax.set_title(title, loc="left", fontsize=11, color=INK)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="lower center", ncol=len(engines))
        fig.suptitle(f"Qwen2.5-0.5B-Instruct on an {gpu}, greedy, 16-bit weights", x=0.01, ha="left",
                     fontsize=12, color=INK)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(DOCS / "bench_single.png", dpi=200, facecolor=SURFACE)
        plt.close(fig)

    # Many requests at once: total generated tokens/s against the number of requests.
    b = r.get("batched", {})
    engines = [e for e in ENGINES if e in b]
    if engines:
        fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
        style(ax)
        for e in engines:
            ns = [n for n in BATCHES if str(n) in b[e]]
            vals = [b[e][str(n)] for n in ns]
            ax.plot(ns, vals, color=COLORS[e], linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=LABELS[e])
            ax.annotate(f"{vals[-1]:,.0f}", (ns[-1], vals[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK_2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BATCHES)
        ax.set_xticklabels([str(n) for n in BATCHES])
        ax.set_xlim(0.8, max(BATCHES) * 1.6)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(thousands)
        ax.set_xlabel("requests served at once (each: 512-token prompt + 128 generated tokens)")
        ax.set_title(f"Total generated tokens/s, prefill included ({gpu}, greedy, 16-bit)",
                     loc="left", fontsize=11, color=INK)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(DOCS / "bench_batched.png", dpi=200, facecolor=SURFACE)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models" / "qwen2.5-0.5b-instruct"))
    ap.add_argument("--ie", default=str(ROOT / "build" / "release" / ("ie.exe" if os.name == "nt" else "ie")))
    ap.add_argument("--llama-dir", help="folder of the unzipped llama.cpp CUDA release")
    ap.add_argument("--skip-ie", action="store_true")
    ap.add_argument("--skip-llama", action="store_true")
    ap.add_argument("--skip-torch", action="store_true")
    ap.add_argument("--charts-only", action="store_true", help="redraw from docs/benchmarks.json")
    args = ap.parse_args()

    DOCS.mkdir(exist_ok=True)
    r = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    if not args.charts_only:
        if not args.skip_ie:
            text = run([args.ie, "gpu-check"])
            for line in text.splitlines():
                if "NVIDIA" in line or "GeForce" in line:
                    r["gpu"] = "RTX 4060" if "4060" in line else line.strip()
                    break
            bench_ie(args.ie, args.model, r)
        if not args.skip_llama:
            if not args.llama_dir:
                sys.exit("--llama-dir is required for llama.cpp (or pass --skip-llama)")
            bench_llama(args.llama_dir, r)
        if not args.skip_torch:
            bench_torch(args.model, r)
        RESULTS.write_text(json.dumps(r, indent=2) + "\n")
    charts(r)
    print()
    print(tables(r))
    print(f"\nwrote {RESULTS.relative_to(ROOT)}, docs/bench_single.png, docs/bench_batched.png")


if __name__ == "__main__":
    main()
