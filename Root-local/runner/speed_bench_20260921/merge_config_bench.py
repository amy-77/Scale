#!/usr/bin/env python3
"""Merge rounds / split-merge granularity sweep on real 128K index-K dumps.

For each (summary_compression C, merge_target_divisor D, merge_target_rounds R):
  * partition build of the sealed 15-shard prefix (122,880 tokens): stage times,
    pure-GPU time via CUDA-graph replay, merge rounds that actually merged
    something, whether the L/D target was reached;
  * prefill Top-2048 recall of the sparse indexer (summary coarse -> weighted
    select -> fine + local window -> Top-K) against the dense DSA Top-2048 for
    the 128 dumped queries of the final chunk. ``recall_prefix`` restricts both
    sets to the partitioned prefix [0, n_complete), which is the part the
    partition can influence.

Usage (inside the eval container)::

  PYTHONPATH=python python runner/speed_bench_20260921/merge_config_bench.py \
      --sample dump:ruler:niah_single_2:128k:6 --layers 0 30 \
      --configs 8:64:8 8:64:6 8:64:5 8:64:4 16:96:8 16:128:8 32:128:8
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/qyl/code/adaptive_0921_h202/python")
ENV = {
    "SGLANG_NSA_ADAPTIVE_HISA_MODE": "adaptive_decode", "SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC": "key_sse",
    "SGLANG_NSA_ADAPTIVE_HISA_SPLIT_BACKEND": "gpu", "SGLANG_NSA_ADAPTIVE_HISA_BUILD_SUMMARIES": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY": "sync_nonoverlap", "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR": "64",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_ROUNDS": "8", "SGLANG_NSA_ADAPTIVE_HISA_MAX_MERGE_LEN": "0",
    "SGLANG_NSA_ADAPTIVE_HISA_GRAPH_BUILD": "1", "SGLANG_NSA_ADAPTIVE_HISA_GPU_STREAM": "side",
    "SGLANG_NSA_ADAPTIVE_HISA_SINK_TOKENS": "64", "SGLANG_NSA_ADAPTIVE_HISA_TAIL_TOKENS": "256",
    "SGLANG_NSA_ADAPTIVE_HISA_CANDIDATE_TOKENS": "8192", "SGLANG_NSA_ADAPTIVE_HISA_DECODE_CHUNK": "64",
    "SGLANG_NSA_ADAPTIVE_HISA_SKIP_CLEAN": "1", "SGLANG_NSA_ADAPTIVE_HISA_FUSE_PAGE_TABLE": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_RAW_FP8_BUILDER": "1", "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_ROWS": "2048",
}
os.environ.update(ENV)

import torch  # noqa: E402

DUMP = Path("/workspace/qyl/data/adaptive_hisa_dump_20260918/dump")


def load_layer(sample: str, layer: int, device):
    d = DUMP / sample / f"L{layer:02d}"
    shards = sorted(p for p in d.iterdir() if p.name.startswith("prefill_"))
    ks, ss = [], []
    for p in shards:
        t = torch.load(p, map_location="cpu", weights_only=False)
        ks.append(t["k_fp8_u8"])
        ss.append(t["k_scale"])
    last = torch.load(shards[-1], map_location="cpu", weights_only=False)
    k = torch.cat(ks).to(device).view(torch.float8_e4m3fn)
    s = torch.cat(ss).to(device).float()
    n_complete = sum(int(x.shape[0]) for x in ks[:-1])
    q = last["q_fp8_u8"].to(device).view(torch.float8_e4m3fn)
    w = last["weights"].to(device).float()
    pos1 = (last["q_positions"].to(device).to(torch.int32) + 1)
    return k, s, n_complete, q, w, pos1


def build(cfg, k, s, n_complete):
    from sglang.srt.layers.attention.nsa.adaptive_hisa.pkey_fp8_builder import (
        build_partition_from_fp8,
        summaries_from_totals,
    )

    for _ in range(2):
        part = build_partition_from_fp8(k[:n_complete], s[:n_complete], n_complete, cfg, profile=True)
        totals = part.meta.pop("_merge_totals")
        keys, scales = summaries_from_totals(totals, part.leaf_len, None)
    torch.cuda.synchronize()
    stages = {key: a.elapsed_time(b) for key, (a, b) in part.events.items()}
    rounds = part.round_merges.tolist()
    # pure GPU time: graph replay
    side = torch.cuda.Stream()
    kk, ss = k[:n_complete].clone(), s[:n_complete].clone()
    with torch.cuda.stream(side):
        for _ in range(2):
            build_partition_from_fp8(kk, ss, n_complete, cfg)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side):
        p2 = build_partition_from_fp8(kk, ss, n_complete, cfg)
        summaries_from_totals(p2.meta.pop("_merge_totals"), p2.leaf_len, None)
    g.replay()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(10):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return part, keys, scales, stages, rounds, a.elapsed_time(b) / 10


def recall(cfg, part, keys, scales, k, s, n_complete, q, w, pos1):
    import deep_gemm

    from sglang.srt.layers.attention.nsa.adaptive_hisa import prefill_select as ps

    dev = k.device
    seq_len = int(pos1.max())
    dummy_pages = torch.zeros((1, 64 * 132), dtype=torch.uint8, device=dev)
    table = torch.zeros((1, 1), dtype=torch.int32, device=dev)
    out = ps.sparse_topk_core(
        q, w, pos1, dummy_pages, table, keys, scales, part.leaf_start, part.leaf_len,
        part.num_leaves, n_complete, seq_len, cfg, k_flat=(k, s),
    ).long()
    dense = deep_gemm.fp8_mqa_logits(q, (k[:seq_len], s[:seq_len]), w, torch.zeros_like(pos1), pos1,
                                     clean_logits=False)[:, :seq_len]
    dense = dense.masked_fill(torch.arange(seq_len, device=dev)[None] >= pos1[:, None].long(), float("-inf"))
    ref = torch.topk(dense, 2048, dim=1).indices
    ref_mask = torch.zeros_like(dense, dtype=torch.bool).scatter_(1, ref, True)
    valid = out >= 0
    hit = ref_mask.gather(1, out.clamp_min(0)) & valid
    r_all = hit.float().sum(1) / ref_mask.float().sum(1)
    prefix_ref = ref_mask.clone()
    prefix_ref[:, n_complete:] = False
    hit_p = hit & (out < n_complete)
    r_prefix = hit_p.float().sum(1) / prefix_ref.float().sum(1).clamp_min(1)
    return float(r_all.mean()), float(r_prefix.mean()), float(prefix_ref.float().sum(1).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="dump:ruler:niah_single_2:128k:6")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 30])
    ap.add_argument("--configs", nargs="+", default=["8:64:8", "8:64:6", "8:64:5", "8:64:4", "8:64:3",
                                                     "16:96:8", "16:128:8", "32:128:8", "32:128:3"],
                    help="summary_compression:merge_divisor:merge_rounds")
    ap.add_argument("--out", type=Path, default=Path("/tmp/merge_config_bench.json"))
    args = ap.parse_args()
    from sglang.srt.layers.attention.nsa.adaptive_hisa.config import get_config

    base = get_config()
    dev = torch.device("cuda")
    rows = []
    for layer in args.layers:
        k, s, n_complete, q, w, pos1 = load_layer(args.sample, layer, dev)
        for spec in args.configs:
            c, d, r = (int(x) for x in spec.split(":"))
            cfg = dataclasses.replace(base, summary_compression=c, merge_target_divisor=d, merge_target_rounds=r).validate()
            part, keys, scales, stages, rounds, gpu_ms = build(cfg, k, s, n_complete)
            leaves = int(part.num_leaves)
            target = n_complete // d
            r_all, r_prefix, n_prefix = recall(cfg, part, keys, scales, k, s, n_complete, q, w, pos1)
            row = dict(layer=layer, compression=c, divisor=d, rounds=r, n_complete=n_complete,
                       leaves=leaves, target=target, target_reached=leaves <= target,
                       rounds_used=sum(1 for x in rounds if x > 0), merges_per_round=rounds,
                       build_gpu_ms=round(gpu_ms, 3),
                       **{key[:-2] + "_ms": round(v, 3) for key, v in stages.items()},
                       recall_all=round(r_all, 4), recall_prefix=round(r_prefix, 4), prefix_ref_mean=round(n_prefix, 1))
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
