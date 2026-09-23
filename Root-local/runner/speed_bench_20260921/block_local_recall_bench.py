#!/usr/bin/env python3
"""Dense-local vs block-local Top-2048 recall on real prefill dumps."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_MODE", "adaptive_decode")
os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_SUMMARY_COMPRESSION", "32")
os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY", "sync_nonoverlap")
os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR", "128")
os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_ROUNDS", "4")
os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL", "1")
os.environ.setdefault("SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_ROWS", "2048")
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


def recall(ref: torch.Tensor, got: torch.Tensor) -> float:
    width = int(ref.max().item()) + 1
    mask = torch.zeros((ref.shape[0], width), dtype=torch.bool, device=ref.device)
    mask.scatter_(1, ref.long(), True)
    valid = (got >= 0) & (got < width)
    safe = got.clamp(0, width - 1).long()
    return float((valid & mask.gather(1, safe)).float().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/data/jz/adaptive_hisa_dump_broad_20260923/raw"),
    )
    ap.add_argument("--max-samples", type=int, default=3)
    ap.add_argument("--chunks", type=int, nargs="+", default=[1, 8, 15])
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/data/jz/speed_957_0923/stats/block_local_recall.json"),
    )
    args = ap.parse_args()
    device = torch.device("cuda")
    cfg = dataclasses.replace(
        get_config(),
        mode="adaptive_decode",
        summary_compression=32,
        merge_target_divisor=128,
        merge_target_rounds=4,
        sparse_prefill=True,
        sparse_prefill_rows=2048,
    ).validate()
    layer_dirs = sorted(
        p
        for p in args.root.glob("*128k*/L*")
        if p.is_dir() and any(p.glob("prefill_*.pt"))
    )
    sample_names = []
    chosen = []
    for path in layer_dirs:
        sample = path.parent.name
        if sample not in sample_names:
            if len(sample_names) >= args.max_samples:
                continue
            sample_names.append(sample)
        if sample in sample_names:
            chosen.append(path)

    rows = []
    dummy_pages = torch.zeros((1, 64 * 132), dtype=torch.uint8, device=device)
    dummy_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
    import deep_gemm

    for path in chosen:
        shards, blobs, k, s = load_layer(path, device)
        for ci in args.chunks:
            if ci <= 0 or ci >= len(shards):
                continue
            blob = blobs[ci]
            start = int(blob["positions"][0])
            n_complete = start // cfg.root * cfg.root
            q = blob["q_fp8_u8"].to(device).view(torch.float8_e4m3fn)
            w = blob["weights"].to(device=device, dtype=torch.float32)
            pos1 = blob["q_positions"].to(device=device, dtype=torch.int32) + 1
            seq_len = int(blob["seq_len"])
            part = build_partition_from_fp8(k, s, n_complete, cfg)
            totals = part.meta.pop("_merge_totals")
            summary, summary_scale = summaries_from_totals(
                totals, part.leaf_len, None
            )
            common = (
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
            )
            dense_local = ps.sparse_topk_core(
                *common, k_flat=(k, s), block_local=False
            )
            block_local = ps.sparse_topk_core(
                *common, k_flat=(k, s), block_local=True
            )
            dense_scores = deep_gemm.fp8_mqa_logits(
                q,
                (k, s),
                w,
                torch.zeros_like(pos1),
                pos1,
                clean_logits=False,
            )[:, :seq_len]
            # clean_logits=False leaves columns outside each row's causal
            # prefix unspecified; mask them before using dense Top-K as oracle.
            col = torch.arange(seq_len, device=device)
            dense_scores.masked_fill_(
                col.unsqueeze(0) >= pos1.unsqueeze(1), float("-inf")
            )
            ref = torch.topk(dense_scores, cfg.index_topk, dim=1).indices
            row = {
                "sample": path.parent.name,
                "layer": path.name,
                "chunk": ci,
                "n_complete": n_complete,
                "queries": int(q.shape[0]),
                "dense_local_recall": recall(ref, dense_local),
                "block_local_recall": recall(ref, block_local),
                "block_vs_dense": recall(dense_local, block_local),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            del dense_scores, ref, dense_local, block_local, part

    summary = {
        "n_cases": len(rows),
        "n_queries": sum(x["queries"] for x in rows),
        "dense_local_recall": sum(x["dense_local_recall"] for x in rows)
        / max(1, len(rows)),
        "block_local_recall": sum(x["block_local_recall"] for x in rows)
        / max(1, len(rows)),
        "block_vs_dense": sum(x["block_vs_dense"] for x in rows)
        / max(1, len(rows)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
