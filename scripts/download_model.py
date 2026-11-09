"""Download a model's weights, config and tokenizer files into models/.

Usage:
    python scripts/download_model.py                       # Qwen2.5-0.5B-Instruct (default)
    python scripts/download_model.py --repo Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = ap.parse_args()

    local_dir = ROOT / "models" / args.repo.split("/")[-1].lower()
    path = snapshot_download(
        repo_id=args.repo,
        local_dir=local_dir,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*"],
    )
    print(f"Downloaded {args.repo} -> {path}")
    for f in sorted(Path(path).iterdir()):
        if f.is_file():
            print(f"  {f.name:40s} {f.stat().st_size / 2**20:10.1f} MiB")


if __name__ == "__main__":
    main()
