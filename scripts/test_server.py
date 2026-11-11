"""Smoke test for a running `ie serve` (standard library only).

    build\\release\\ie.exe serve models\\qwen2.5-0.5b-instruct
    python scripts\\test_server.py --url http://127.0.0.1:8080

Checks /health, one chat completion without and one with streaming, a burst of
concurrent requests (aggregate tokens per second), and that a greedy request
(temperature 0, repetition penalty 1) gives the same text alone and while the
server is busy with other requests, which is the promise of batching.
"""

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request

results = []  # (name, ok, detail)


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    return ok


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, json.loads(r.read())


def post(url: str, body: dict, timeout: float = 600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def chat(base: str, body: dict) -> dict:
    with post(base + "/v1/chat/completions", body) as r:
        return json.loads(r.read())


def chat_stream(base: str, body: dict):
    """Returns (text, finish_reason, usage, seconds to the first text, events, saw [DONE] last)."""
    t0 = time.perf_counter()
    text, finish, usage, first, events, done_last = "", None, None, None, 0, False
    body = dict(body, stream=True, stream_options={"include_usage": True})
    with post(base + "/v1/chat/completions", body) as r:
        for raw in r:  # one line at a time; events are "data: ...\n\n"
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line:
                continue
            if not line.startswith("data: "):
                raise ValueError(f"unexpected line in the stream: {line!r}")
            data = line[len("data: "):]
            events += 1
            done_last = data == "[DONE]"
            if done_last:
                continue
            chunk = json.loads(data)
            if "error" in chunk:
                raise ValueError(chunk["error"]["message"])
            usage = chunk.get("usage") or usage
            for c in chunk["choices"]:
                piece = c["delta"].get("content") or ""
                if piece and first is None:
                    first = time.perf_counter() - t0
                text += piece
                finish = c["finish_reason"] or finish
    return text, finish, usage, first, events, done_last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="requests at once for the load test (under 32 keeps every batch bit-exact)")
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()
    base = args.url.rstrip("/")

    # 1. Is it up?
    try:
        status, health = get_json(base + "/health")
    except (urllib.error.URLError, OSError) as e:
        print(f"FAIL  cannot reach {base}: {e}")
        return 1
    check("GET /health", status == 200 and health.get("status") == "ok",
          f"model {health.get('model')}, max batch {health.get('max_batch')}, "
          f"KV cache {health.get('kv_pages_total')} pages")
    status, models = get_json(base + "/v1/models")
    model = models["data"][0]["id"] if models.get("data") else "?"
    check("GET /v1/models", status == 200 and model == health.get("model"), model)

    question = [{"role": "user", "content": "Explain what a KV cache is in two sentences."}]

    # 2. One chat completion.
    t0 = time.perf_counter()
    r = chat(base, {"model": model, "messages": question, "max_tokens": args.max_tokens})
    secs = time.perf_counter() - t0
    reply = r["choices"][0]["message"]["content"]
    u = r["usage"]
    check("chat completion", r["object"] == "chat.completion" and len(reply) > 0
          and r["choices"][0]["finish_reason"] in ("stop", "length")
          and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"],
          f"{u['prompt_tokens']} + {u['completion_tokens']} tokens in {secs:.2f} s")
    print(f"      {reply[:200]!r}")

    # 3. The same, streamed.
    text, finish, usage, first, events, done_last = chat_stream(
        base, {"model": model, "messages": question, "max_tokens": args.max_tokens})
    check("streamed chat completion", len(text) > 0 and finish in ("stop", "length") and done_last
          and usage is not None and usage["completion_tokens"] > 0,
          f"{events} events, first text after {1000 * (first or 0):.0f} ms, ends with [DONE]")

    # 4. Greedy reference, alone.
    greedy = {"model": model, "max_tokens": args.max_tokens, "temperature": 0, "repetition_penalty": 1,
              "messages": [{"role": "user",
                            "content": "List five uses of a hash table, one per line."}]}
    alone = chat(base, greedy)["choices"][0]["message"]["content"]

    # 5. Many at once (the server batches them), with the greedy request in the middle.
    n = max(1, args.concurrency)
    topics = ["rivers", "compilers", "volcanoes", "chess", "bread", "satellites", "jazz", "glaciers",
              "databases", "bees", "lighthouses", "origami", "tides", "keyboards", "comets", "maps"]
    load_out, errors = [None] * n, []

    def worker(i: int) -> None:
        try:
            topic = topics[i % len(topics)]
            body = {"model": model, "max_tokens": args.max_tokens, "seed": i + 1,
                    "messages": [{"role": "user", "content": f"Write a short paragraph about {topic}."}]}
            load_out[i] = chat(base, body)
        except Exception as e:  # report every failure, keep the others going
            errors.append(f"request {i}: {e}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(0.05)  # let the load start first
    busy = chat(base, greedy)["choices"][0]["message"]["content"]
    for t in threads:
        t.join()
    secs = time.perf_counter() - t0
    done = [r for r in load_out if r]
    tokens = sum(r["usage"]["completion_tokens"] for r in done)
    detail = f"{tokens} tokens in {secs:.2f} s = {tokens / secs:.1f} tok/s in total"
    check(f"{n} concurrent requests", not errors and len(done) == n,
          detail + (f"; {errors[0]}" if errors else ""))
    _, health = get_json(base + "/health")
    check("server batched them", health.get("peak_batch", 0) > 1, f"peak batch {health.get('peak_batch')}")

    same = busy == alone
    detail = "identical" if same else f"alone: {alone[:80]!r} | under load: {busy[:80]!r}"
    if not same and n + 1 >= 32:
        detail += " (batches of 32+ use tensor cores, which can round a near-tie the other way)"
    check("greedy output alone == under load", same, detail)

    passed = sum(ok for _, ok, _ in results)
    print(f"\n{'PASS' if passed == len(results) else 'FAIL'}: {passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
