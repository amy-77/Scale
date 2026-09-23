#!/usr/bin/env python3
"""Global split32/merge128 vs root-local prefill Top-2048 recall.

Both arms consume the same real prefill dump and use the production sparse
indexer with the dense local-window path. The oracle is a full-context
DeepGEMM score matrix with an explicit per-query causal mask; clean_logits=False
does not initialize masked columns, so omitting this mask invalidates recall.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_MODE", "adaptive_decode")
os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_SUMMARY_COMPRESSION", "32")
os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY", "sync_nonoverlap")
os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR", "128")
os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_ROUNDS", "4")
os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL", "1")
os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_ROWS", "8192")
os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_PREFILL_SELECT", "weighted")

import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa import prefill_select as ps
from sglang.srt.layers.attention.nsa.adaptive_hisa.config import get_config
from sglang.srt.layers.attention.nsa.adaptive_hisa.pkey_fp8_builder import (
    build_partition_from_fp8,
    summaries_from_totals,
)


def load_layer(path: Path, device):
    shards = sorted(path.glob("prefill_*.pt"))
    blobs = [torch.load(p, map_location="cpu", weights_only=False) for p in shards]
    k = torch.cat([x["k_fp8_u8"] for x in blobs]).to(device).view(torch.float8_e4m3fn)
    s = torch.cat([x["k_scale"] for x in blobs]).to(device=device, dtype=torch.float32)
    return shards, blobs, k, s


def row_recall(ref: torch.Tensor, got: torch.Tensor) -> torch.Tensor:
    """Per-row set recall; invalid/padded output ids count as misses."""
    width = int(ref.max().item()) + 1
    mask = torch.zeros((ref.shape[0], width), dtype=torch.bool, device=ref.device)
    mask.scatter_(1, ref.long(), True)
    valid = (got >= 0) & (got < width)
    safe = got.clamp(0, width - 1).long()
    return (valid & mask.gather(1, safe)).float().mean(dim=1)


def sparse_output(
    cfg,
    q,
    w,
    pos1,
    k,
    s,
    n_complete: int,
    seq_len: int,
    dummy_pages,
    dummy_table,
):
    part = build_partition_from_fp8(k, s, n_complete, cfg)
    totals = part.meta.pop("_merge_totals")
    summary, summary_scale = summaries_from_totals(totals, part.leaf_len, None)
    out = ps.sparse_topk_core(
        q,
        w,
        pos1,
        dummy_pages,
        dummy_table,
        summary,
        summary_scale.reshape(-1),
        part.leaf_start,
        part.leaf_len,
        part.num_leaves,
        n_complete,
        seq_len,
        cfg,
        k_flat=(k, s),
        block_local=False,
    )
    return out, int(part.num_leaves.item())


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
        "--root",
        type=Path,
        default=Path("/data/jz/adaptive_hisa_dump_ruler_full_20260923/raw"),
    )
    ap.add_argument("--max-samples", type=int, default=0, help="0 means all 128K samples")
    ap.add_argument(
        "--chunks",
        type=int,
        nargs="*",
        default=None,
        help="Chunk indices after chunk 0; default is every available sparse chunk",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/data/jz/speed_957_0923/stats/root_local_recall_ruler_full.json"),
    )
    args = ap.parse_args()

    device = torch.device("cuda")
    base = dataclasses.replace(
        get_config(),
        mode="adaptive_decode",
        summary_compression=32,
        merge_target_divisor=128,
        merge_target_rounds=4,
        sparse_prefill=True,
        sparse_prefill_rows=8192,
        root_local_partition=False,
    ).validate()
    root_local = dataclasses.replace(base, root_local_partition=True).validate()
    layer_dirs = sorted(
        p
        for p in args.root.glob("*128k*/L*")
        if p.is_dir() and any(p.glob("prefill_*.pt"))
    )
    samples: list[str] = []
    chosen: list[Path] = []
    for path in layer_dirs:
        sample = path.parent.name
        if sample not in samples:
            if args.max_samples and len(samples) >= args.max_samples:
                continue
            samples.append(sample)
        if sample in samples:
            chosen.append(path)

    import deep_gemm

    dummy_pages = torch.zeros((1, 64 * 132), dtype=torch.uint8, device=device)
    dummy_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    rows = []
    all_values = defaultdict(list)
    grouped = defaultdict(lambda: defaultdict(list))

    for path in chosen:
        shards, blobs, k, s = load_layer(path, device)
        chunk_ids = args.chunks if args.chunks is not None else range(1, len(shards))
        for ci in chunk_ids:
            if ci <= 0 or ci >= len(shards):
                continue
            blob = blobs[ci]
            start = int(blob["positions"][0])
            n_complete = start // base.root * base.root
            if n_complete < base.prefill_candidate_tokens:
                continue
            q = blob["q_fp8_u8"].to(device).view(torch.float8_e4m3fn)
            w = blob["weights"].to(device=device, dtype=torch.float32)
            pos1 = blob["q_positions"].to(device=device, dtype=torch.int32) + 1
            seq_len = int(blob["seq_len"])

            global_out, global_leaves = sparse_output(
                base,
                q,
                w,
                pos1,
                k,
                s,
                n_complete,
                seq_len,
                dummy_pages,
                dummy_table,
            )
            root_out, root_leaves = sparse_output(
                root_local,
                q,
                w,
                pos1,
                k,
                s,
                n_complete,
                seq_len,
                dummy_pages,
                dummy_table,
            )
            dense = deep_gemm.fp8_mqa_logits(
                q,
                (k, s),
                w,
                torch.zeros_like(pos1),
                pos1,
                clean_logits=False,
            )[:, :seq_len]
            cols = torch.arange(seq_len, device=device)
            dense.masked_fill_(cols.unsqueeze(0) >= pos1.unsqueeze(1), float("-inf"))
            ref = torch.topk(dense, base.index_topk, dim=1).indices

            global_recall = row_recall(ref, global_out)
            root_recall = row_recall(ref, root_out)
            overlap = row_recall(global_out, root_out)
            delta = root_recall - global_recall
            case = {
                "sample": path.parent.name,
                "layer": path.name,
                "chunk": int(ci),
                "n_complete": n_complete,
                "queries": int(q.shape[0]),
                "global_leaves": global_leaves,
                "root_leaves": root_leaves,
                "global_recall": float(global_recall.mean()),
                "root_recall": float(root_recall.mean()),
                "delta": float(delta.mean()),
                "root_vs_global": float(overlap.mean()),
            }
            rows.append(case)
            print(json.dumps(case), flush=True)

            for name, tensor in (
                ("global_recall", global_recall),
                ("root_recall", root_recall),
                ("delta", delta),
                ("root_vs_global", overlap),
            ):
                vals = tensor.cpu().tolist()
                all_values[name].extend(vals)
                grouped[("layer", path.name)][name].extend(vals)
                grouped[("chunk", str(ci))][name].extend(vals)
                grouped[("sample", path.parent.name)][name].extend(vals)
            del dense, ref, global_out, root_out

        del k, s, blobs
        torch.cuda.empty_cache()

    sample_delta_means = [
        sum(metrics["delta"]) / len(metrics["delta"])
        for (kind, _key), metrics in grouped.items()
        if kind == "sample" and metrics["delta"]
    ]
    summary = {
        "samples": len(samples),
        "layers": len(chosen),
        "cases": len(rows),
        "queries": len(all_values["delta"]),
        **{name: stats(values) for name, values in all_values.items()},
        "sample_mean_delta": stats(sample_delta_means),
        "root_better_queries": sum(x > 0 for x in all_values["delta"]),
        "root_equal_queries": sum(x == 0 for x in all_values["delta"]),
        "root_worse_queries": sum(x < 0 for x in all_values["delta"]),
    }
    breakdown = {}
    for (kind, key), metrics in grouped.items():
        breakdown.setdefault(kind, {})[key] = {
            name: stats(values) for name, values in metrics.items()
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "breakdown": breakdown, "rows": rows}, indent=2)
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
