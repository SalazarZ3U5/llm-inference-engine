"""Generate tests/tokenizer_cases.json: input strings and the token ids that
Hugging Face's tokenizer produces for them. `ie test-tokenizer` must match
every case exactly.

Usage:
    python scripts/make_tokenizer_cases.py
    python scripts/make_tokenizer_cases.py --model models/qwen2.5-0.5b-instruct
"""

import argparse
import json
import random
import unicodedata
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent

HAND_WRITTEN = [
    "The capital of France is",
    "Hello, world!",
    "",
    " ",
    "   leading spaces",
    "trailing spaces   ",
    "a  b   c    d",
    "tabs\tand\t\tmore\ttabs",
    "line one\nline two\n\nline four\n",
    "windows\r\nline\r\nendings",
    "\n\n\n",
    "  \n  indented\n    more\n",
    "I'm sure they'll say it's fine, but we'd rather you've checked.",
    "DON'T SHOUT. I'M HERE. 'S 'T 'RE",
    "'sup, 'twas 'hello' and ''quoted''",
    "12345 67890",
    "3.14159 and -2.5e10",
    "$1,000,000.00 (USD) or ₹83,000",
    "Version 2.0.1-beta+build.42",
    "x² + y² = z²; ½ of Ⅻ is Ⅵ",
    "café naïve résumé façade",
    "Ünïcödé ÀÉÎÕÜ ñ ß",
    "नमस्ते दुनिया, आप कैसे हैं?",
    "કેમ છો? મજામાં.",
    "你好，世界！今天天气很好。",
    "こんにちは、世界。カタカナ ひらがな",
    "안녕하세요 세계",
    "Привет, мир! Как дела?",
    "مرحبا بالعالم",
    "Γειά σου Κόσμε",
    "emoji 👍 👍🏽 👨‍👩‍👧 🇮🇳 ✨🔥",
    "a non-breaking space and　ideographic space",
    "zero​width​space",
    "def f(x):\n    return x * 2  # double\n",
    "for (int i = 0; i < n; ++i) { sum += a[i]; }",
    '{"key": "value", "list": [1, 2, 3], "nested": {"a": null}}',
    "https://example.com/path?query=1&other=two#frag",
    "user@example.com wrote: >>> quoted",
    "# Heading\n\n- item **bold** _italic_ `code`\n\n```python\nprint('hi')\n```",
    "Wait... what?!?! No way!!!",
    "---===***+++",
    "∑ x_i ≥ 0 ∀ i ∈ ℕ, ∫ f(x) dx ≈ π",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n",
    "text<|endoftext|>more text",
    "<tool_call>{\"name\": \"f\"}</tool_call>",
    "a <|not_special|> b < | im_start | >",
    "The quick brown fox jumps over the lazy dog. " * 4,
]


def random_cases(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    pool = (list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            + list("     \t\n\r'\".,;:!?-_()[]{}<>/\\|@#$%^&*+=~`")
            + list("éñüßçøåæœ") + list("नमस्तेकैसे") + list("你好世界") + list("ありがとう")
            + list("Привет") + ["👍", "🔥", "🏽", "‍", " ", "　", "²", "Ⅻ", "₹"])
    out = []
    for _ in range(n):
        length = rng.randint(1, 60)
        out.append("".join(rng.choice(pool) for _ in range(length)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "qwen2.5-0.5b-instruct"))
    ap.add_argument("--out", default=str(ROOT / "tests" / "tokenizer_cases.json"))
    ap.add_argument("--random", type=int, default=300)
    args = ap.parse_args()

    tok = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    cases = []
    for text in HAND_WRITTEN + random_cases(args.random, seed=1234):
        # The engine skips NFC normalization, so feed both sides NFC text.
        text = unicodedata.normalize("NFC", text)
        cases.append({"text": text, "ids": tok.encode(text).ids})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
