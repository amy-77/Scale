#!/usr/bin/env python3
"""Top-2048 recall + candidate-set diagnostics for Merge divisor / candidate sweeps.

Compares Adaptive leaf maps / Top-2048 against a fixed HISA-K64 style uniform
chunking oracle on synthetic keys (no full model). Emits JSON suitable for
``sweep_algorithm_knobs.py --summarise``.

Usage::

  PYTHONPATH=python python runner/diagnose_topk_recall.py --out /tmp/recall.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "python"))


def hisa_uniform_leaves(n_complete: int, chunk: int = 64):
    starts = list(range(0, n_complete, chunk))
    lengths = [min(chunk, n_complete - s) for s in starts]
    return starts, lengths


def recall_at_k(ref: set[int], pred: list[int], k: int = 2048) -> float:
    if not ref:
        return 1.0
    hit = sum(1 for x in pred[:k] if x in ref)
    return hit / min(k, len(ref))


def run_one(n: int, merge_divisor: int, candidate_tokens: int, device="cuda") -> dict:
    from sglang.srt.layers.attention.nsa.adaptive_hisa.config import PartitionConfig
    from sglang.srt.layers.attention.nsa.adaptive_hisa.pkey_fp8_builder import (
        build_partition_from_fp8,
    )
    from sglang.srt.layers.attention.nsa.adaptive_hisa.decode_select import (
        weighted_select_candidates,
    )

    torch.manual_seed(n + merge_divisor)
    # Synthetic FP8-ish keys.
    keys = torch.randn(n, 128, device=device)
    scale = keys.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) / 448.0
    k_fp8 = (keys / scale).to(torch.float8_e4m3fn)
    k_scale = scale.squeeze(1).to(torch.float32)

    cfg = PartitionConfig(
        mode="adaptive_decode",
        partition_metric="key_sse",
        merge_policy="sync_nonoverlap",
        merge_target_divisor=merge_divisor,
        merge_target_rounds=8,
        candidate_tokens=candidate_tokens,
        decode_chunk=64,
        build_summaries=True,
    ).validate()
    part = build_partition_from_fp8(k_fp8, k_scale, n, cfg)
    count = int(part.num_leaves.item())
    starts = part.leaf_start[:count].tolist()
    lengths = part.leaf_len[:count].tolist()

    # Coarse scores: mean key L2 as a cheap stand-in.
    deq = k_fp8.to(torch.float32) * k_scale[:, None]
    leaf_scores = []
    for s, ln in zip(starts, lengths):
        leaf_scores.append(float(deq[s : s + ln].pow(2).mean().item()) if ln else float("-inf"))
    capacity = part.capacity
    gpu_scores = torch.tensor(
        leaf_scores + [float("-inf")] * (capacity - count),
        dtype=torch.float32,
        device=device,
    )
    gpu_starts = part.leaf_start
    gpu_lens = part.leaf_len
    num_leaves = torch.tensor([count], dtype=torch.int32, device=device)
    seq_t = torch.tensor([n], dtype=torch.int32, device=device)
    prefix = torch.zeros(capacity, dtype=torch.int32, device=device)
    cand, ccount = weighted_select_candidates(
        gpu_scores, gpu_lens, gpu_starts, num_leaves, seq_t, prefix,
        candidate_tokens, cfg.sink_tokens, cfg.tail_tokens,
    )
    adaptive_set = set(cand[0, : int(ccount.item())].tolist())

    # HISA-K64 oracle: uniform chunks, take top by same score rule until budget.
    h_starts, h_lens = hisa_uniform_leaves(n, 64)
    h_scores = []
    for s, ln in zip(h_starts, h_lens):
        h_scores.append(float(deq[s : s + ln].pow(2).mean().item()))
    # Fill budget with ranked HISA leaves (same sink/tail).
    sink, tail = cfg.sink_tokens, cfg.tail_tokens
    sink_end = min(sink, n)
    tail_start = max(sink_end, n - tail)
    room = candidate_tokens - sink_end - (n - tail_start)
    order = sorted(range(len(h_lens)), key=lambda i: (-h_scores[i], i))
    hisa_tokens = list(range(sink_end))
    for i in order:
        first = min(max(h_starts[i], sink_end), tail_start)
        last = min(max(h_starts[i] + h_lens[i], sink_end), tail_start)
        clipped = max(last - first, 0)
        if room <= 0 or clipped <= 0:
            if room <= 0:
                break
            continue
        take = min(clipped, room)
        hisa_tokens.extend(range(first, first + take))
        room -= take
    hisa_tokens.extend(range(tail_start, n))
    hisa_set = set(hisa_tokens)

    # Proxy Top-2048: highest |key| tokens inside each candidate set.
    def topk_from(cset: set[int], k=2048):
        idx = torch.tensor(sorted(cset), device=device, dtype=torch.long)
        energy = deq[idx].pow(2).sum(-1)
        pick = energy.topk(min(k, idx.numel())).indices
        return set(idx[pick].tolist())

    ref_top = topk_from(hisa_set | adaptive_set, 2048)  # soft oracle over union
    pred_top = sorted(topk_from(adaptive_set, 2048))
    return {
        "n": n,
        "merge_divisor": merge_divisor,
        "candidate_tokens": candidate_tokens,
        "adaptive_leaves": count,
        "hisa_leaves": len(h_lens),
        "candidate_overlap": len(adaptive_set & hisa_set) / max(1, len(hisa_set)),
        "recall_2048": recall_at_k(ref_top, pred_top, 2048),
        "status_ok": bool(part.status_ok.item()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 32768])
    ap.add_argument("--merges", type=int, nargs="+", default=[64, 96, 128])
    ap.add_argument("--candidates", type=int, nargs="+", default=[8192])
    args = ap.parse_args()
    if not torch.cuda.is_available():
        args.out.write_text(json.dumps({"ok": False, "reason": "no CUDA"}))
        sys.exit(2)
    rows = []
    for n in args.lengths:
        for d in args.merges:
            for c in args.candidates:
                rows.append(run_one(n, d, c))
                print(json.dumps(rows[-1]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"ok": True, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
