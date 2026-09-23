#!/usr/bin/env python3
"""Compare exact and relaxed P-key split/merge paths on real decode dumps."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from sglang.srt.layers.attention.nsa.adaptive_hisa.config import (  # noqa: E402
    PartitionConfig,
)
from sglang.srt.layers.attention.nsa.adaptive_hisa.decode_select import (  # noqa: E402
    weighted_select_candidates,
)
from sglang.srt.layers.attention.nsa.adaptive_hisa.pkey_fp8_builder import (  # noqa: E402
    build_partition_from_fp8,
    summaries_from_totals,
)


def discover_samples(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("layer_*/req_*/manifest.json"))


def load_sample(sample_dir: Path, device: torch.device) -> dict:
    manifest = json.loads((sample_dir / "manifest.json").read_text())
    keys, scales = [], []
    for name in manifest["key_shards"]:
        blob = torch.load(sample_dir / name, map_location="cpu", weights_only=False)
        keys.append(blob["k_fp8"])
        scales.append(blob["k_scale"].reshape(-1))
    n_complete = (int(keys[0].shape[0]) // 256) * 256
    k_fp8 = torch.cat(keys).to(device)
    if k_fp8.dtype == torch.uint8:
        k_fp8 = k_fp8.view(torch.float8_e4m3fn)
    k_scale = torch.cat(scales).to(device=device, dtype=torch.float32)
    queries = []
    for name in manifest["query_shards"]:
        blob = torch.load(sample_dir / name, map_location="cpu", weights_only=False)
        q_fp8 = blob["q_fp8"][0].to(device)
        if q_fp8.dtype == torch.uint8:
            q_fp8 = q_fp8.view(torch.float8_e4m3fn)
        queries.append(
            {
                "q_fp8": q_fp8,
                "w": blob["w"][0].to(device=device, dtype=torch.float32),
                "ke": int(blob["ke"][0]),
            }
        )
    return {
        "k_fp8": k_fp8,
        "k_scale": k_scale,
        "n_complete": n_complete,
        "queries": queries,
        "layer": sample_dir.parent.name,
        "task": manifest.get("sample_meta", {}).get("task", "unknown"),
    }


def configs() -> dict[str, PartitionConfig]:
    base = PartitionConfig(
        mode="adaptive_decode",
        atom=1,
        summary_compression=32,
        split_backend="gpu",
        lambda_candidates=16,
        lambda_max_rounds=6,
        relaxed_split=False,
        merge_policy="sync_nonoverlap",
        merge_target_divisor=128,
        merge_target_rounds=4,
        soft_merge_rounds=0,
        build_summaries=True,
    ).validate()
    return {
        "exact": base,
        "relaxed": dataclasses.replace(
            base,
            lambda_max_rounds=2,
            relaxed_split=True,
            soft_merge_rounds=2,
            merge_target_rounds=1,
            merge_alpha=2.0,
        ).validate(),
    }


def build_arm(sample: dict, cfg: PartitionConfig) -> tuple[dict, dict]:
    args = (
        sample["k_fp8"][: sample["n_complete"]],
        sample["k_scale"][: sample["n_complete"]],
        sample["n_complete"],
        cfg,
    )
    warm = build_partition_from_fp8(*args)
    warm.meta.pop("_merge_totals", None)
    torch.cuda.synchronize()
    part = build_partition_from_fp8(*args, profile=True)
    totals = part.meta.pop("_merge_totals")
    mean_fp8, mean_scale = summaries_from_totals(totals, part.leaf_len, "ue8m0")
    torch.cuda.synchronize()
    n = int(part.num_leaves)
    timing = {
        name.removesuffix("_s") + "_ms": start.elapsed_time(end)
        for name, (start, end) in part.events.items()
    }
    timing["split_ms"] = timing["dp_ms"] + timing["repair_ms"]
    round_merges = [int(x) for x in part.round_merges]
    force_rounds = int(part.meta.get("force_target_rounds", 0))
    forced_merges = sum(round_merges[-force_rounds:]) if force_rounds else 0
    timing.update(
        {
            "leaves": n,
            "capacity": part.capacity,
            "dp_leaves": int(part.dp_leaves),
            "repairs": int(part.repairs),
            "forced_merges": forced_merges,
            "overflow_fallback": int(forced_merges > 0),
            "status_ok": bool(part.status_ok),
            "round_merges": round_merges,
        }
    )
    built = {
        "start": part.leaf_start[:n].contiguous(),
        "length": part.leaf_len[:n].contiguous(),
        "mean_fp8": mean_fp8[:n].contiguous(),
        "mean_scale": mean_scale.reshape(-1)[:n].contiguous(),
    }
    return built, timing


def select(arm: dict, sample: dict, query: dict, cfg: PartitionConfig) -> torch.Tensor:
    means = arm["mean_fp8"].float() * arm["mean_scale"][:, None]
    coarse = (
        query["w"][:, None] * torch.relu(query["q_fp8"].float() @ means.T)
    ).sum(0).contiguous()
    capacity = arm["start"].numel()
    candidates, count = weighted_select_candidates(
        coarse,
        arm["length"],
        arm["start"],
        torch.tensor([capacity], dtype=torch.int32, device=coarse.device),
        torch.tensor([query["ke"]], dtype=torch.int32, device=coarse.device),
        torch.zeros(capacity, dtype=torch.int32, device=coarse.device),
        cfg.candidate_tokens,
        cfg.sink_tokens,
        cfg.tail_tokens,
    )
    ids = candidates[0, : int(count)].long()
    keys = sample["k_fp8"][ids].float() * sample["k_scale"][ids, None]
    fine = (
        query["w"][:, None] * torch.relu(query["q_fp8"].float() @ keys.T)
    ).sum(0)
    order = torch.argsort(ids, stable=True)
    order = order[torch.argsort(-fine[order], stable=True)]
    return ids[order[: cfg.index_topk]]


def recall(ref: torch.Tensor, pred: torch.Tensor) -> float:
    return float(torch.isin(pred, ref).sum().item()) / max(1, int(ref.numel()))


def stats(values: list[float]) -> dict:
    x = torch.tensor(values, dtype=torch.float64)
    if not x.numel():
        return {"count": 0}
    return {
        "count": x.numel(),
        "mean": float(x.mean()),
        "p50": float(x.quantile(0.5)),
        "p95": float(x.quantile(0.95)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dump",
        type=Path,
        default=Path("/data/jz/adaptive_hisa_dump_complete_20260923/converted/rank_00"),
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--layers", nargs="*", default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "stats" / "relaxed_partition_bench.json",
    )
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}")
    cfgs = configs()
    samples = discover_samples(args.dump)
    if args.layers:
        samples = [p for p in samples if p.parent.name in set(args.layers)]
    if args.max_samples:
        samples = samples[: args.max_samples]

    rows = []
    recall_values = defaultdict(list)
    task_recall = defaultdict(lambda: defaultdict(list))
    latency_values = defaultdict(lambda: defaultdict(list))
    for index, sample_dir in enumerate(samples):
        sample = load_sample(sample_dir, device)
        arms, timings = {}, {}
        for name, cfg in cfgs.items():
            arms[name], timings[name] = build_arm(sample, cfg)
            for metric, value in timings[name].items():
                if metric.endswith("_ms") or metric in (
                    "leaves",
                    "dp_leaves",
                    "repairs",
                    "forced_merges",
                    "overflow_fallback",
                ):
                    latency_values[name][metric].append(float(value))

        keys = sample["k_fp8"].float() * sample["k_scale"][:, None]
        query_rows = []
        for query in sample["queries"]:
            ke = min(query["ke"], keys.shape[0])
            if ke <= cfgs["exact"].index_topk:
                continue
            exact_scores = (
                query["w"][:, None]
                * torch.relu(query["q_fp8"].float() @ keys[:ke].T)
            ).sum(0)
            ref = torch.topk(exact_scores, cfgs["exact"].index_topk).indices
            recalls = {
                name: recall(ref, select(arm, sample, {**query, "ke": ke}, cfgs[name]))
                for name, arm in arms.items()
            }
            for name, value in recalls.items():
                recall_values[name].append(value)
                task_recall[sample["task"]][name].append(value)
                if name != "exact":
                    recall_values[name + "_delta"].append(value - recalls["exact"])
                    task_recall[sample["task"]][name + "_delta"].append(
                        value - recalls["exact"]
                    )
            query_rows.append(recalls)

        row = {
            "sample": str(sample_dir.relative_to(args.dump)),
            "layer": sample["layer"],
            "task": sample["task"],
            "queries": len(query_rows),
            "timings": timings,
            "mean_recall": {
                name: sum(q[name] for q in query_rows) / len(query_rows)
                for name in cfgs
                if query_rows
            },
        }
        rows.append(row)
        print(json.dumps({"progress": index + 1, **row}), flush=True)
        del sample, arms, keys
        torch.cuda.empty_cache()

    result = {
        "configs": {name: dataclasses.asdict(cfg) for name, cfg in cfgs.items()},
        "summary": {
            "recall": {name: stats(values) for name, values in recall_values.items()},
            "recall_by_task": {
                task: {name: stats(values) for name, values in metrics.items()}
                for task, metrics in task_recall.items()
            },
            "partition": {
                arm: {name: stats(values) for name, values in metrics.items()}
                for arm, metrics in latency_values.items()
            },
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
