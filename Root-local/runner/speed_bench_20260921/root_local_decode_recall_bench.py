#!/usr/bin/env python3
"""Global split32/merge128 vs root-local recall on real decode queries."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_MODE", "adaptive_decode")
os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_SUMMARY_COMPRESSION", "32")
os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY", "sync_nonoverlap")
os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR", "128")
os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_ROUNDS", "4")
os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_RAW_FP8_BUILDER", "1")

import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa.config import get_config
from sglang.srt.layers.attention.nsa.adaptive_hisa.pkey_fp8_builder import (
    build_partition_from_fp8,
    summaries_from_totals,
)

from runner.compare_fixed_adaptive_chunks import (
    LeafMap,
    discover_samples,
    load_sample,
    recall_at_k,
    select_and_rerank,
)


def partition_arm(sample, cfg):
    part = build_partition_from_fp8(
        sample["k_fp8"][: sample["n_complete"]],
        sample["k_scale"][: sample["n_complete"]],
        sample["n_complete"],
        cfg,
    )
    n = int(part.num_leaves.item())
    leaves = LeafMap(
        start=part.leaf_start[:n].contiguous(),
        length=part.leaf_len[:n].contiguous(),
        kind="root_local" if cfg.root_local_partition else "global",
        chunk=128,
    )
    totals = part.meta.pop("_merge_totals")[:n].contiguous()
    mean_fp8, mean_scale = summaries_from_totals(totals, leaves.length, "ue8m0")
    return leaves, mean_fp8, mean_scale.reshape(-1)


def stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    x = torch.tensor(values, dtype=torch.float64)
    std = float(x.std(unbiased=True)) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": float(x.mean()),
        "std": std,
        "mean_ci95": 1.96 * std / (len(values) ** 0.5),
        "median": float(x.median()),
        "p05": float(torch.quantile(x, 0.05)),
        "p95": float(torch.quantile(x, 0.95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dump",
        type=Path,
        default=Path("/data/jz/adaptive_hisa_dump_complete_20260923/converted/rank_00"),
    )
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--layers", nargs="*", default=None)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/data/jz/speed_957_0923/stats/root_local_decode_recall.json"),
    )
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.device}")

    base = dataclasses.replace(
        get_config(),
        mode="adaptive_decode",
        summary_compression=32,
        merge_target_divisor=128,
        merge_target_rounds=4,
        root_local_partition=False,
    ).validate()
    root_cfg = dataclasses.replace(base, root_local_partition=True).validate()
    sample_dirs = discover_samples(args.dump)
    if args.layers:
        allowed = set(args.layers)
        sample_dirs = [p for p in sample_dirs if p.parent.name in allowed]
    if args.max_samples:
        sample_dirs = sample_dirs[: args.max_samples]

    rows = []
    values = defaultdict(list)
    grouped = defaultdict(lambda: defaultdict(list))
    for idx, sample_dir in enumerate(sample_dirs):
        sample = load_sample(sample_dir, device)
        global_leaves, global_mean, global_scale = partition_arm(sample, base)
        root_leaves, root_mean, root_scale = partition_arm(sample, root_cfg)
        keys = sample["k_fp8"].float() * sample["k_scale"][:, None]
        sample_metrics = defaultdict(list)

        for q in sample["queries"]:
            ke = min(int(q["ke"]), int(keys.shape[0]))
            if ke <= base.index_topk:
                continue
            exact = (
                q["w"][:, None] * torch.relu(q["q_fp8"].float() @ keys[:ke].T)
            ).sum(0)
            ref = torch.topk(exact, base.index_topk).indices
            global_sel = select_and_rerank(
                q["q_fp8"],
                q["w"],
                sample["k_fp8"],
                sample["k_scale"],
                global_leaves,
                global_mean,
                global_scale,
                ke,
                base.candidate_tokens,
                base.sink_tokens,
                base.tail_tokens,
                base.index_topk,
            )
            root_sel = select_and_rerank(
                q["q_fp8"],
                q["w"],
                sample["k_fp8"],
                sample["k_scale"],
                root_leaves,
                root_mean,
                root_scale,
                ke,
                root_cfg.candidate_tokens,
                root_cfg.sink_tokens,
                root_cfg.tail_tokens,
                root_cfg.index_topk,
            )
            metrics = {
                "global_candidate_recall": recall_at_k(ref, global_sel["candidates"]),
                "root_candidate_recall": recall_at_k(ref, root_sel["candidates"]),
                "global_rerank_recall": recall_at_k(ref, global_sel["topk"]),
                "root_rerank_recall": recall_at_k(ref, root_sel["topk"]),
                "root_vs_global": recall_at_k(global_sel["topk"], root_sel["topk"]),
            }
            metrics["candidate_delta"] = (
                metrics["root_candidate_recall"] - metrics["global_candidate_recall"]
            )
            metrics["rerank_delta"] = (
                metrics["root_rerank_recall"] - metrics["global_rerank_recall"]
            )
            for name, value in metrics.items():
                values[name].append(value)
                sample_metrics[name].append(value)
                grouped[("layer", sample["layer"])][name].append(value)
                task = str(sample["manifest"].get("sample_meta", {}).get("task", "unknown"))
                grouped[("task", task)][name].append(value)

        task = str(sample["manifest"].get("sample_meta", {}).get("task", "unknown"))
        row = {
            "sample": str(sample_dir.relative_to(args.dump)),
            "task": task,
            "layer": sample["layer"],
            "queries": len(sample_metrics["rerank_delta"]),
            "n_complete": sample["n_complete"],
            "global_leaves": global_leaves.num_leaves,
            "root_leaves": root_leaves.num_leaves,
            **{
                name: sum(v) / len(v)
                for name, v in sample_metrics.items()
                if v
            },
        }
        rows.append(row)
        print(json.dumps({"progress": idx + 1, **row}), flush=True)
        del sample, keys
        torch.cuda.empty_cache()

    sample_rerank_delta = [r["rerank_delta"] for r in rows if "rerank_delta" in r]
    summary = {
        "samples": len(rows),
        "queries": len(values["rerank_delta"]),
        **{name: stats(v) for name, v in values.items()},
        "sample_mean_rerank_delta": stats(sample_rerank_delta),
        "root_better_queries": sum(x > 0 for x in values["rerank_delta"]),
        "root_equal_queries": sum(x == 0 for x in values["rerank_delta"]),
        "root_worse_queries": sum(x < 0 for x in values["rerank_delta"]),
    }
    breakdown = {}
    for (kind, key), metrics in grouped.items():
        breakdown.setdefault(kind, {})[key] = {
            name: stats(v) for name, v in metrics.items()
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "breakdown": breakdown, "rows": rows}, indent=2)
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
