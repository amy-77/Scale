#!/usr/bin/env python3
"""Summarize broad fixed-64 vs P-key split-8/merge-64 paired results."""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "candidate_recall_all",
    "summary_topk_recall_stable",
    "summary_topk_recall_expected",
    "token_score_relmse",
    "chunk_score_relmse",
)


def bootstrap(values: list[float], *, seed: int, rounds: int = 5000) -> dict:
    if not values:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = random.Random(seed)
    n = len(values)
    means = [
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(rounds)
    ]
    means.sort()
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "ci95": [means[int(0.025 * rounds)], means[int(0.975 * rounds)]],
        "n": n,
        "frac_positive": sum(value > 0 for value in values) / n,
    }


def aggregate_samples(samples: list[dict], seed: int) -> dict:
    out = {"n_samples": len(samples)}
    for metric in METRICS:
        fixed = [row[f"fixed_{metric}"] for row in samples]
        adaptive = [row[f"adaptive_{metric}"] for row in samples]
        delta = [a - f for f, a in zip(fixed, adaptive)]
        out[metric] = {
            "fixed_mean": statistics.mean(fixed),
            "adaptive_mean": statistics.mean(adaptive),
            "paired_delta": bootstrap(delta, seed=seed + len(metric)),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", type=Path, required=True)
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in (args.exp / "per_query_rows.jsonl").read_text().splitlines()
        if line.strip()
    ]
    runner_summary = json.loads((args.exp / "summary.json").read_text())
    buckets: dict[str, dict] = {}
    layer_buckets: dict[tuple[str, str], dict] = {}
    for row in rows:
        meta = row["sample_meta"]
        tag = meta["tag"]
        bucket = buckets.setdefault(tag, {"meta": meta, "values": defaultdict(list)})
        layer_key = (tag, row["layer"])
        layer_bucket = layer_buckets.setdefault(
            layer_key,
            {"meta": meta, "layer": row["layer"], "values": defaultdict(list)},
        )
        fixed_name = next(name for name in row["arms"] if name.startswith("fixed_"))
        adapt_name = next(name for name in row["arms"] if name.startswith("adaptive_"))
        fixed = row["arms"][fixed_name]["queries"]
        adaptive = row["arms"][adapt_name]["queries"]
        for f_query, a_query in zip(fixed, adaptive):
            for metric in METRICS:
                bucket["values"][f"fixed_{metric}"].append(f_query[metric])
                bucket["values"][f"adaptive_{metric}"].append(a_query[metric])
                layer_bucket["values"][f"fixed_{metric}"].append(f_query[metric])
                layer_bucket["values"][f"adaptive_{metric}"].append(a_query[metric])

    per_sample = []
    for tag, bucket in sorted(buckets.items()):
        row = dict(bucket["meta"])
        row["tag"] = tag
        for name, values in bucket["values"].items():
            row[name] = statistics.mean(values)
        for metric in METRICS:
            row[f"delta_{metric}"] = (
                row[f"adaptive_{metric}"] - row[f"fixed_{metric}"]
            )
        per_sample.append(row)

    per_layer_sample = []
    for (_, layer), bucket in sorted(layer_buckets.items()):
        row = dict(bucket["meta"])
        row["layer"] = layer
        for name, values in bucket["values"].items():
            row[name] = statistics.mean(values)
        for metric in METRICS:
            row[f"delta_{metric}"] = (
                row[f"adaptive_{metric}"] - row[f"fixed_{metric}"]
            )
        per_layer_sample.append(row)

    def grouped(keys: tuple[str, ...], source: list[dict], seed: int) -> list[dict]:
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for row in source:
            groups[tuple(row[key] for key in keys)].append(row)
        output = []
        for index, (key, values) in enumerate(sorted(groups.items())):
            output.append(
                {
                    **dict(zip(keys, key)),
                    **aggregate_samples(values, seed + index * 100),
                }
            )
        return output

    summary = {
        "protocol": {
            "sample_unit": "one representative prompt per available task-length group",
            "layers": sorted({row["layer"] for row in rows}),
            "queries_per_layer": 16,
            "candidate_tokens": 8192,
            "dense_topk": 2048,
            "fixed": runner_summary["config"].get("ks", [64]),
            "adaptive": {
                "split_divisor": runner_summary["config"].get("split_divisor"),
                "merge_target_divisor": runner_summary["config"].get("merge_target_divisor"),
                "merge_policy": runner_summary["config"].get("merge_policy"),
            },
            "primary_metric": runner_summary["config"]["primary_metric"],
        },
        "coverage": {
            "samples": len(per_sample),
            "layer_samples": len(per_layer_sample),
            "query_pairs": sum(
                len(next(v["queries"] for k, v in row["arms"].items() if k.startswith("fixed_")))
                for row in rows
            ),
            "truncated_samples": sum(bool(row["truncated"]) for row in per_sample),
        },
        "overall": aggregate_samples(per_sample, 20260923),
        "by_dataset": grouped(("dataset",), per_sample, 1000),
        "by_dataset_length": grouped(
            ("dataset", "length"), per_sample, 2000
        ),
        "by_longbench_domain_length": grouped(
            ("task", "length"),
            [row for row in per_sample if row["dataset"] == "longbench_v2"],
            3000,
        ),
        "by_layer": grouped(("layer",), per_layer_sample, 4000),
        "per_sample": per_sample,
    }
    (args.exp / "broad_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    csv_fields = [
        "dataset",
        "task",
        "subtask",
        "length",
        "difficulty",
        "source_id",
        "tag",
        "prompt_tokens_original",
        "prompt_tokens",
        "truncated",
    ]
    for metric in METRICS:
        csv_fields.extend(
            [f"fixed_{metric}", f"adaptive_{metric}", f"delta_{metric}"]
        )
    with (args.exp / "per_sample_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_sample)

    print(
        json.dumps(
            {
                "coverage": summary["coverage"],
                "overall": summary["overall"],
                "by_dataset_length": summary["by_dataset_length"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
