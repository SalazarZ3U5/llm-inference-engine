"""Dump ground-truth activations from Hugging Face transformers.

Your engine must reproduce these numbers. Compare layer by layer: the first
layer whose output diverges is where your bug is.

Writes to reference/<name>/:
    input_ids.npy        int64  [seq]           token ids of the prompt
    embed.npy            f32    [seq, hidden]   output of the embedding lookup
    layer_00.npy ...     f32    [seq, hidden]   output of each decoder layer
    final_norm.npy       f32    [seq, hidden]   after the last RMSNorm
    logits.npy           f32    [seq, vocab]
    generated_ids.npy    int64  [n]             greedy continuation (no sampling)
    meta.json                                   prompt, decoded text, config summary

Usage:
    python scripts/reference.py
    python scripts/reference.py --prompt "Once upon a time" --gen 32
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent


def to_np(x):
    if isinstance(x, (tuple, list)):
        x = x[0]
    return x[0].detach().float().cpu().numpy()  # drop batch dim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "qwen2.5-0.5b-instruct"))
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--gen", type=int, default=20, help="greedy tokens to generate")
    ap.add_argument("--name", default="default")
    args = ap.parse_args()

    out = ROOT / "reference" / args.name
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    # FP32 on CPU: the most precise reference to compare against.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    captured = {}

    def hook(name):
        def fn(_module, _inputs, output):
            captured[name] = to_np(output)
        return fn

    inner = model.model
    hooks = [inner.embed_tokens.register_forward_hook(hook("embed")),
             inner.norm.register_forward_hook(hook("final_norm"))]
    for i, layer in enumerate(inner.layers):
        hooks.append(layer.register_forward_hook(hook(f"layer_{i:02d}")))

    ids = tok(args.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits
    for h in hooks:
        h.remove()

    np.save(out / "input_ids.npy", ids[0].numpy())
    np.save(out / "logits.npy", to_np(logits))
    for name, arr in captured.items():
        np.save(out / f"{name}.npy", arr)

    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=args.gen, do_sample=False)
    new_ids = gen[0, ids.shape[1]:]
    np.save(out / "generated_ids.npy", new_ids.numpy())

    cfg = model.config
    meta = {
        "prompt": args.prompt,
        "input_ids": ids[0].tolist(),
        "generated_ids": new_ids.tolist(),
        "generated_text": tok.decode(new_ids),
        "config": {k: getattr(cfg, k, None) for k in [
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "vocab_size",
            "rms_norm_eps", "tie_word_embeddings", "max_position_embeddings"]},
    }
    # Newer transformers versions move rope_theta into config.rope_parameters.
    rope = getattr(cfg, "rope_parameters", None) or {}
    meta["config"]["rope_theta"] = getattr(cfg, "rope_theta", None) or rope.get("rope_theta")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"prompt     : {args.prompt!r}")
    print(f"input_ids  : {ids[0].tolist()}")
    print(f"greedy out : {tok.decode(new_ids)!r}")
    print(f"next token : {tok.decode([int(logits[0, -1].argmax())])!r}")
    print(f"saved {len(captured) + 3} arrays to {out}")


if __name__ == "__main__":
    main()
