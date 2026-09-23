#!/usr/bin/env python3
"""Algorithm-knob sweep: chunk64 → Merge {64,96,128} → candidate {8192,6144,4096}.

Does not launch GPUs while the frozen e2e owns the cards. Emits a JSON plan of
runs and a lightweight recall/consistency checklist. Execute each config on a
free machine with the env blocks below, then feed result JSONLs back through
``--summarise``.

Accuracy gate: vs current P-key L/64, LongBench 401 + RULER 364 must show no
statistically significant regression (paired bootstrap / McNemar).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

BASE = {
    "SGLANG_NSA_ADAPTIVE_HISA_MODE": "adaptive_decode",
    "SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC": "key_sse",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY": "sync_nonoverlap",
    "SGLANG_NSA_ADAPTIVE_HISA_BUILD_SUMMARIES": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_SKIP_CLEAN": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_FUSE_PAGE_TABLE": "1",
}

# Phase order from the plan.
PHASE1_CHUNK = [{"DECODE_CHUNK": 64, "MERGE_TARGET_DIVISOR": 64, "CANDIDATE_TOKENS": 8192}]
PHASE2_MERGE = [
    {"DECODE_CHUNK": 64, "MERGE_TARGET_DIVISOR": d, "CANDIDATE_TOKENS": 8192}
    for d in (64, 96, 128)
]
PHASE3_CAND = [
    {"DECODE_CHUNK": 64, "MERGE_TARGET_DIVISOR": 64, "CANDIDATE_TOKENS": c}
    for c in (8192, 6144, 4096)
]


def env_block(knobs: dict) -> dict[str, str]:
    out = dict(BASE)
    for k, v in knobs.items():
        out[f"SGLANG_NSA_ADAPTIVE_HISA_{k}"] = str(v)
    return out


def build_plan() -> list[dict]:
    plan = []
    for phase, rows in (
        ("chunk64", PHASE1_CHUNK),
        ("merge_divisor", PHASE2_MERGE),
        ("candidate_budget", PHASE3_CAND),
    ):
        for knobs in rows:
            name = (
                f"chunk{knobs['DECODE_CHUNK']}"
                f"_merge{knobs['MERGE_TARGET_DIVISOR']}"
                f"_cand{knobs['CANDIDATE_TOKENS']}"
            )
            plan.append({"phase": phase, "name": name, "env": env_block(knobs), "knobs": knobs})
    return plan


def summarise(results: dict) -> dict:
    """``results`` maps config name → {recall@2048, longbench, ruler, decode_ms}."""
    ranked = sorted(
        (
            (
                name,
                float(v.get("decode_ms_p50", 1e9)),
                float(v.get("longbench", 0)),
                float(v.get("ruler", 0)),
                float(v.get("recall_2048", 0)),
            )
            for name, v in results.items()
        ),
        key=lambda t: (t[1], -t[2], -t[3]),
    )
    return {
        "ranked_by_decode_then_accuracy": [
            {
                "name": n,
                "decode_ms_p50": d,
                "longbench": lb,
                "ruler": ru,
                "recall_2048": rc,
            }
            for n, d, lb, ru, rc in ranked
        ],
        "recommend": ranked[0][0] if ranked else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--summarise", type=Path, default=None,
                    help="JSON of measured results to rank")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan = build_plan()
    (args.out_dir / "sweep_plan.json").write_text(json.dumps(plan, indent=2))
    for row in plan:
        (args.out_dir / f"env_{row['name']}.json").write_text(json.dumps(row["env"], indent=2))
    print(json.dumps({"n_configs": len(plan), "names": [r["name"] for r in plan]}, indent=2))
    if args.summarise and args.summarise.exists():
        summary = summarise(json.loads(args.summarise.read_text()))
        (args.out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
