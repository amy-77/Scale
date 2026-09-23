#!/usr/bin/env python3
"""Fair DSA / fixed-HISA-K64 / Adaptive-HISA decode+TTFT benchmark harness.

Three independently frozen arms on the same hardware, TP, seed, candidate
budget (8192), Top-2048, and CUDA-graph policy. Uses ``bench_one_batch``-style
fixed ``input_ids`` so tokenizer / chat-template time is excluded.

Usage (after the live e2e releases the GPUs)::

  # Adaptive (this package)
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
  SGLANG_NSA_ADAPTIVE_HISA_MODE=adaptive_decode \\
  SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC=key_sse \\
  SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY=sync_nonoverlap \\
  SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR=64 \\
  SGLANG_NSA_ADAPTIVE_HISA_DECODE_CHUNK=64 \\
  SGLANG_NSA_ADAPTIVE_HISA_DECODE_TIMING=1 \\
  python -m sglang.bench_one_batch ... --batch-size 1 --input-len 32768 \\
      --output-len 128 --warmup 10

  # Fixed HISA K=64 (upstream hisa path; NOT Adaptive fixed8)
  SGLANG_NSA_ADAPTIVE_HISA_MODE=off \\
  SGLANG_HISA_ENABLED=1 ...

  # DSA baseline
  SGLANG_NSA_ADAPTIVE_HISA_MODE=off \\
  SGLANG_HISA_ENABLED=0 ...

This script materialises fixed input_ids dumps and summarises JSONL results.
It does not launch a server while another e2e owns the cards.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path

import torch


ARMS = ("dsa", "hisa_k64", "adaptive_pkey")
LENGTHS = (32768, 65536, 131072)


def materialise_inputs(out_dir: Path, lengths=LENGTHS, seed: int = 20260920) -> dict:
    """Write one contiguous int32 dump per length so every arm sees identical IDs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"seed": seed, "lengths": {}}
    g = torch.Generator().manual_seed(seed)
    for n in lengths:
        path = out_dir / f"input_ids_{n}.pt"
        if not path.exists():
            # Vocab-agnostic placeholders in [100, 50000); real runs should
            # overwrite with tokenizer-encoded LongBench/RULER prompts if desired.
            ids = torch.randint(100, 50000, (1, n), generator=g, dtype=torch.int32)
            torch.save(ids, path)
        meta["lengths"][str(n)] = str(path)
    (out_dir / "inputs_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def env_for_arm(arm: str) -> dict[str, str]:
    """Environment deltas relative to a shared TP8 DeepSeek-V3.2 launch."""
    common = {
        "SGLANG_NSA_INDEX_TOPK": "2048",
        "SGLANG_NSA_CANDIDATE_TOKENS": "8192",
    }
    if arm == "dsa":
        return {
            **common,
            "SGLANG_NSA_ADAPTIVE_HISA_MODE": "off",
            "SGLANG_HISA_ENABLED": "0",
        }
    if arm == "hisa_k64":
        return {
            **common,
            "SGLANG_NSA_ADAPTIVE_HISA_MODE": "off",
            "SGLANG_HISA_ENABLED": "1",
            "SGLANG_HISA_CHUNK_SIZE": "64",
        }
    if arm == "adaptive_pkey":
        return {
            **common,
            "SGLANG_NSA_ADAPTIVE_HISA_MODE": "adaptive_decode",
            "SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC": "key_sse",
            "SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY": "sync_nonoverlap",
            "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR": "64",
            "SGLANG_NSA_ADAPTIVE_HISA_DECODE_CHUNK": "64",
            "SGLANG_NSA_ADAPTIVE_HISA_BUILD_SUMMARIES": "1",
            "SGLANG_NSA_ADAPTIVE_HISA_SKIP_CLEAN": "1",
            "SGLANG_NSA_ADAPTIVE_HISA_FUSE_PAGE_TABLE": "1",
            "SGLANG_NSA_ADAPTIVE_HISA_RAW_FP8_BUILDER": "1",
            "SGLANG_NSA_ADAPTIVE_HISA_DECODE_TIMING": "1",
            "SGLANG_HISA_ENABLED": "0",
        }
    raise ValueError(arm)


def summarise_decode(samples_ms: list[float], warmup: int = 10) -> dict:
    """Drop the first ``warmup`` decode steps; report median / p95 ms/token."""
    body = samples_ms[warmup:] if len(samples_ms) > warmup else samples_ms
    if not body:
        return {"n": 0}
    ordered = sorted(body)
    return {
        "n": len(body),
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "mean_ms": statistics.fmean(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def gate_vs_dsa(adaptive: dict, dsa: dict, lengths=(32768, 131072)) -> dict:
    """Performance gate: Adaptive median decode must beat DSA at 32K and 128K."""
    verdict = {}
    for n in lengths:
        a = adaptive.get(str(n), {}).get("median_ms")
        d = dsa.get(str(n), {}).get("median_ms")
        if a is None or d is None:
            verdict[str(n)] = {"ok": False, "reason": "missing"}
            continue
        verdict[str(n)] = {
            "ok": a < d,
            "adaptive_ms": a,
            "dsa_ms": d,
            "speedup": (d / a) if a else None,
        }
    return verdict


def ingest_bench_jsonl(path: Path, warmup: int = 10) -> dict:
    """Convert ``bench_one_batch`` JSONL rows into the results schema of ``summarise_decode``."""
    by_len: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n = str(row.get("input_len") or row.get("input_length"))
        lat = row.get("decode_latencies")
        if lat is None and "median_decode_latency" in row:
            # Fallback: synthesise a one-sample list in ms.
            lat = [float(row["median_decode_latency"]) * 1000.0]
        else:
            lat = [float(x) * 1000.0 for x in (lat or [])]
        by_len[n] = {
            "decode_ms": lat,
            "warmup": int(row.get("decode_warmup", warmup)),
            "ttft_ms": float(row["ttft"]) * 1000.0 if row.get("ttft") is not None
            else (float(row["prefill_latency"]) * 1000.0 if row.get("prefill_latency") else None),
            "p95_ms": float(row["p95_decode_latency"]) * 1000.0
            if row.get("p95_decode_latency") is not None else None,
        }
    return by_len


def write_launch_scripts(out_dir: Path, model_path: str, tp: int = 8) -> None:
    """Emit one shell launcher per arm that pins identical input dumps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = out_dir / "inputs"
    materialise_inputs(inputs)
    for arm, env in ((a, env_for_arm(a)) for a in ARMS):
        script = out_dir / f"run_{arm}.sh"
        exports = "\n".join(f'export {k}="{v}"' for k, v in env.items())
        body = f"""#!/usr/bin/env bash
set -euo pipefail
{exports}
export SGLANG_BENCH_DECODE_WARMUP=10
# Prefer a free GPU set; do not attach to a live LongBench e2e.
python -m sglang.bench_one_batch \\
  --model-path {model_path} \\
  --tp-size {tp} \\
  --batch-size 1 \\
  --input-len 32768 65536 131072 \\
  --output-len 128 \\
  --run-name {arm} \\
  --result-filename {out_dir / (arm + '.jsonl')} \\
  --disable-cuda-graph
"""
        script.write_text(body)
        script.chmod(0o755)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--materialise-only", action="store_true")
    ap.add_argument("--write-launchers", action="store_true")
    ap.add_argument("--model-path", type=str, default="")
    ap.add_argument("--results-json", type=Path, default=None,
                    help="Optional JSON with per-arm per-length decode samples")
    ap.add_argument("--ingest-jsonl-dir", type=Path, default=None,
                    help="Directory with dsa.jsonl / hisa_k64.jsonl / adaptive_pkey.jsonl")
    args = ap.parse_args()
    meta = materialise_inputs(args.out_dir / "inputs", seed=args.seed)
    arms = {arm: env_for_arm(arm) for arm in ARMS}
    (args.out_dir / "arm_envs.json").write_text(json.dumps(arms, indent=2))
    if args.write_launchers:
        if not args.model_path:
            raise SystemExit("--model-path required with --write-launchers")
        write_launch_scripts(args.out_dir, args.model_path)
    print(json.dumps({"inputs": meta, "arms": list(arms)}, indent=2))
    raw = None
    if args.ingest_jsonl_dir is not None:
        raw = {}
        for arm in ARMS:
            path = args.ingest_jsonl_dir / f"{arm}.jsonl"
            if path.exists():
                raw[arm] = ingest_bench_jsonl(path)
        (args.out_dir / "results_raw.json").write_text(json.dumps(raw, indent=2))
    elif args.results_json is not None:
        raw = json.loads(args.results_json.read_text())
    if args.materialise_only or raw is None:
        return
    summary = {}
    for arm, by_len in raw.items():
        summary[arm] = {
            n: summarise_decode(v.get("decode_ms", []), warmup=int(v.get("warmup", 10)))
            for n, v in by_len.items()
        }
        for n, v in by_len.items():
            if v.get("ttft_ms") is not None:
                summary[arm][n]["ttft_ms"] = v["ttft_ms"]
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if "adaptive_pkey" in summary and "dsa" in summary:
        gates = gate_vs_dsa(summary["adaptive_pkey"], summary["dsa"])
        (args.out_dir / "gates_vs_dsa.json").write_text(json.dumps(gates, indent=2))
        print(json.dumps({"gates_vs_dsa": gates}, indent=2))
    # Attach Adaptive stage timing snapshot if present.
    try:
        from sglang.srt.layers.attention.nsa.adaptive_hisa.decode_timing import snapshot

        stages = snapshot(synchronize=False)
        if stages:
            (args.out_dir / "adaptive_stage_timing.json").write_text(
                json.dumps(stages, indent=2)
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
