"""Where does the GPU time go? Summarizes an Nsight Systems profile of `ie`.

    python scripts/profile_summary.py build/prof/decode512.sqlite

The engine marks each forward pass with an NVTX range ("prefill", "decode",
"decode batch"). For each kind, this prints:
  - the wall time of one pass and how much of it the GPU spent in kernels,
    in copies, and idle (launch overhead, waiting on the CPU);
  - every kernel's share of the kernel time and its average time per pass.
The .sqlite file comes from `nsys profile --export sqlite` (see scripts/profile.bat).
"""

import re
import sqlite3
import sys
from bisect import bisect_right
from collections import defaultdict

PHASES = ("prefill", "decode", "decode batch")


def short_name(name: str) -> str:
    """'void ie::cuda::(anonymous namespace)::matmul_kernel<Bf16, 1>(float *, ...)' -> 'matmul_kernel<Bf16, 1>'"""
    name = re.sub(r"^void ", "", name)
    # GCC writes "(anonymous namespace)::", MSVC "<unnamed>::"
    name = re.sub(r"ie::cuda::|\(anonymous namespace\)::|<unnamed>::|_GLOBAL__N_\w+::|ie::", "", name)
    # Cut the parameter list: the first '(' outside the template arguments
    # (which can themselves hold parentheses, e.g. MSVC's "(bool)0").
    depth = 0
    for i, ch in enumerate(name):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch == "(" and depth == 0 and i > 0:
            name = name[:i]
            break
    name = re.sub(r"\(int\)(-?\d+)", r"\1", name)  # MSVC: <Bf16, (int)4> -> <Bf16, 4>
    return name.replace("(bool)0", "false").replace("(bool)1", "true")


def load(path):
    db = sqlite3.connect(path)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))

    ranges = defaultdict(list)  # phase -> [(start, end)]
    cols = {r[1] for r in db.execute("PRAGMA table_info(NVTX_EVENTS)")}
    text_expr = "COALESCE(text, (SELECT value FROM StringIds WHERE id = textId))" if "textId" in cols else "text"
    for text, start, end in db.execute(f"SELECT {text_expr}, start, end FROM NVTX_EVENTS WHERE end IS NOT NULL"):
        if text in PHASES:
            ranges[text].append((start, end))

    kernels = [(s, e, short_name(strings.get(n, str(n))))
               for s, e, n in db.execute("SELECT start, end, demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    copies = []
    if db.execute("SELECT name FROM sqlite_master WHERE name = 'CUPTI_ACTIVITY_KIND_MEMCPY'").fetchone():
        copies = list(db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY"))
    return ranges, sorted(kernels), sorted(copies)


def assign(events, spans):
    """Groups events (sorted by start) by the span (sorted, non-overlapping) they start in."""
    starts = [s for s, _ in spans]
    out = defaultdict(list)
    for ev in events:
        i = bisect_right(starts, ev[0]) - 1
        if i >= 0 and ev[0] < spans[i][1]:
            out[i].append(ev)
    return out


def main(path):
    ranges, kernels, copies = load(path)
    if not ranges:
        sys.exit("no NVTX ranges found: was the profile taken with -t cuda,nvtx?")
    for phase in PHASES:
        spans = sorted(ranges.get(phase, []))
        if not spans:
            continue
        by_span = assign(kernels, spans)
        copy_span = assign(copies, spans)
        n = len(spans)
        wall = sum(e - s for s, e in spans)
        k_time = sum(e - s for evs in by_span.values() for s, e, _ in evs)
        c_time = sum(e - s for evs in copy_span.values() for s, e in evs)
        per_kernel = defaultdict(lambda: [0, 0])  # name -> [total ns, calls]
        for evs in by_span.values():
            for s, e, name in evs:
                per_kernel[name][0] += e - s
                per_kernel[name][1] += 1

        print(f"\n## {phase}: {n} passes, {wall / n / 1e3:,.1f} us each")
        print(f"GPU busy in kernels {100 * k_time / wall:.1f}% | copies {100 * c_time / wall:.1f}% | "
              f"other (launch, sync, CPU) {100 * max(0, wall - k_time - c_time) / wall:.1f}%\n")
        print("| Kernel | Share of kernel time | us per pass | Calls per pass |")
        print("|--------|---------------------:|------------:|---------------:|")
        for name, (t, calls) in sorted(per_kernel.items(), key=lambda kv: -kv[1][0]):
            print(f"| `{name}` | {100 * t / k_time:.1f}% | {t / n / 1e3:,.1f} | {calls / n:g} |")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
