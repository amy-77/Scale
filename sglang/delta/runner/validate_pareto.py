#!/usr/bin/env python3
"""Validation gate for Adaptive-HISA deliverable configs.

Checks:
1. Unit tests (prefill partition + decode) — invoke unittest discover paths.
2. Bit-exact candidate / Top-2048 consistency helper (opt-in CUDA).
3. Paired LongBench / RULER bootstrap gate vs P-key L/64 baseline JSONL.
4. 32K / 128K decode median gate vs DSA (and optionally HISA).

Does not disrupt a live e2e; reads finished result artefacts only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
PYTHON = PACKAGE / "python"
UNIT_TESTS = [
    PACKAGE / "delta/test/registered/unit/test_adaptive_hisa_prefill_partition.py",
    PACKAGE / "delta/test/registered/unit/test_adaptive_hisa_decode.py",
]

DELIVERABLE_ENV = {
    "SGLANG_NSA_ADAPTIVE_HISA_MODE": "adaptive_decode",
    "SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC": "key_sse",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY": "sync_nonoverlap",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR": "64",
    "SGLANG_NSA_ADAPTIVE_HISA_DECODE_CHUNK": "64",
    "SGLANG_NSA_ADAPTIVE_HISA_CANDIDATE_TOKENS": "8192",
    "SGLANG_NSA_ADAPTIVE_HISA_SKIP_CLEAN": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_FUSE_PAGE_TABLE": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_RAW_FP8_BUILDER": "1",
    "SGLANG_NSA_FUSE_TOPK": "0",
}


def paired_bootstrap(base: list[float], cand: list[float], n: int = 2000, seed: int = 0) -> dict:
    """Two-sided paired bootstrap on mean(cand - base). Significant if 95% CI excludes 0."""
    assert len(base) == len(cand) and base
    rng = random.Random(seed)
    diffs = [c - b for b, c in zip(base, cand)]
    means = []
    m = len(diffs)
    for _ in range(n):
        sample = [diffs[rng.randrange(m)] for _ in range(m)]
        means.append(sum(sample) / m)
    means.sort()
    lo = means[int(0.025 * n)]
    hi = means[int(0.975 * n)]
    return {
        "mean_delta": sum(diffs) / m,
        "ci95": [lo, hi],
        "significant": lo > 0 or hi < 0,
        "non_inferior": lo >= -1e-9 or (lo < 0 and hi >= 0 and abs(lo) < abs(hi)),
    }


def mcnemar(base_ok: list[bool], cand_ok: list[bool]) -> dict:
    """McNemar discordant-pair test; flags significant accuracy drop."""
    b01 = sum(1 for b, c in zip(base_ok, cand_ok) if b and not c)
    b10 = sum(1 for b, c in zip(base_ok, cand_ok) if (not b) and c)
    n = b01 + b10
    if n == 0:
        return {"b01": b01, "b10": b10, "p": 1.0, "significant_drop": False}
    # Exact binomial two-sided under H0 p=0.5.
    from math import comb

    p = sum(comb(n, k) for k in range(n + 1) if abs(k - n / 2) >= abs(b01 - n / 2)) / (2**n)
    return {
        "b01": b01,
        "b10": b10,
        "p": p,
        "significant_drop": b01 > b10 and p < 0.05,
    }


def run_unit_tests() -> dict:
    env = {"PYTHONPATH": str(PYTHON), **dict(**__import__("os").environ)}
    results = {}
    for path in UNIT_TESTS:
        if not path.exists():
            results[path.name] = {"ok": False, "reason": "missing"}
            continue
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(PACKAGE),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        results[path.name] = {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    # Bit-exact decode diagnostics (skip-clean + intervals).
    diag = PACKAGE / "runner/diagnose_decode_consistency.py"
    if diag.exists():
        proc = subprocess.run(
            [sys.executable, str(diag)],
            cwd=str(PACKAGE),
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        results["diagnose_decode_consistency"] = {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    return results


def decode_gate(adaptive: dict, dsa: dict, hisa: dict | None = None) -> dict:
    out = {}
    for length in ("32768", "131072"):
        a = adaptive.get(length, {}).get("median_ms")
        d = dsa.get(length, {}).get("median_ms")
        row = {"adaptive": a, "dsa": d, "beats_dsa": a is not None and d is not None and a < d}
        if hisa and length in hisa:
            h = hisa[length].get("median_ms")
            row["hisa"] = h
            row["beats_hisa"] = a is not None and h is not None and a < h
        out[length] = row
    ttft_ok = True
    for length, row in adaptive.items():
        base_ttft = dsa.get(length, {}).get("ttft_ms")
        cand_ttft = row.get("ttft_ms")
        if base_ttft and cand_ttft and cand_ttft > base_ttft * 1.01:
            ttft_ok = False
    out["ttft_ok"] = ttft_ok
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-unit", action="store_true")
    ap.add_argument("--accuracy-base", type=Path, default=None,
                    help="JSONL or JSON with per-example scores for P-key L/64")
    ap.add_argument("--accuracy-cand", type=Path, default=None)
    ap.add_argument("--speed-summary", type=Path, default=None,
                    help="bench_speed_arms summary.json")
    args = ap.parse_args()
    report = {"deliverable_env": DELIVERABLE_ENV}
    if not args.skip_unit:
        report["unit_tests"] = run_unit_tests()
    if args.accuracy_base and args.accuracy_cand:
        base = json.loads(args.accuracy_base.read_text())
        cand = json.loads(args.accuracy_cand.read_text())
        if "scores" in base and "scores" in cand:
            report["paired_bootstrap"] = paired_bootstrap(base["scores"], cand["scores"])
        if "correct" in base and "correct" in cand:
            report["mcnemar"] = mcnemar(base["correct"], cand["correct"])
    if args.speed_summary and args.speed_summary.exists():
        summary = json.loads(args.speed_summary.read_text())
        report["decode_gate"] = decode_gate(
            summary.get("adaptive_pkey", {}),
            summary.get("dsa", {}),
            summary.get("hisa_k64"),
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    (args.out.parent / "deliverable.env.json").write_text(
        json.dumps(DELIVERABLE_ENV, indent=2)
    )
    print(json.dumps({k: report[k] for k in report if k != "unit_tests"}, indent=2))
    if "unit_tests" in report:
        failed = [k for k, v in report["unit_tests"].items() if not v.get("ok")]
        print(json.dumps({"unit_failed": failed}, indent=2))
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()
