#!/usr/bin/env python3
"""Streaming TTFT / TPOT client for one running sglang server.

Fixed random input_ids (seed 20260920, same as runner/bench_speed_arms.py) so
every arm sees identical tokens; tokenizer time is excluded.
"""
import argparse, json, random, statistics, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--arm", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lengths", type=int, nargs="+", default=[131072])
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--new-tokens", type=int, default=128)
args = ap.parse_args()


def ids_for(n, seed=20260920):
    rng = random.Random(seed * 1000003 + n)
    return [rng.randrange(100, 50000) for _ in range(n)]


def one(ids, new_tokens):
    payload = json.dumps({
        "input_ids": ids, "stream": True,
        "sampling_params": {"temperature": 0, "max_new_tokens": new_tokens, "ignore_eos": True},
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/generate", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps = []  # (t, completion_tokens)
    meta = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            d = json.loads(body)
            meta = d.get("meta_info", {})
            stamps.append((time.perf_counter(), int(meta.get("completion_tokens", 0))))
    # first stamp with >=1 token = TTFT; per-token intervals from then on
    first = next((t for t, c in stamps if c >= 1), None)
    last_t, last_c = stamps[-1]
    ttft = first - t0
    tpot = (last_t - first) / max(last_c - 1, 1)
    return dict(ttft_s=ttft, tpot_ms=tpot * 1e3, tokens=last_c, total_s=last_t - t0,
                e2e_latency_server=meta.get("e2e_latency"), prompt_tokens=meta.get("prompt_tokens"))


rows = []
with open(args.out, "a") as f:
    for n in args.lengths:
        ids = ids_for(n)
        samples = []
        for i in range(args.warmup + args.reps):
            r = one(ids, args.new_tokens)
            r.update(arm=args.arm, input_len=n, rep=i, warmup=i < args.warmup)
            f.write(json.dumps(r) + "\n"); f.flush()
            print(json.dumps(r), flush=True)
            if i >= args.warmup:
                samples.append(r)
        s = dict(arm=args.arm, input_len=n, summary=True,
                 ttft_s_median=statistics.median(x["ttft_s"] for x in samples),
                 ttft_s_min=min(x["ttft_s"] for x in samples),
                 tpot_ms_median=statistics.median(x["tpot_ms"] for x in samples),
                 tpot_ms_min=min(x["tpot_ms"] for x in samples))
        f.write(json.dumps(s) + "\n"); f.flush()
        print(json.dumps(s), flush=True)
