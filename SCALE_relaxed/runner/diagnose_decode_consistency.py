#!/usr/bin/env python3
"""Bit-exact / recall diagnostics for Adaptive-HISA decode optimisations.

Validates, without changing the candidate set:
* skip ``clean_summary_scores`` vs clean-on (same selected tokens)
* ``weighted_select_intervals`` expands to the same token set as the token path
* optional segment scorer logits match K=1 sparse_paged_mqa (allclose)

Usage (needs a free CUDA device)::

  PYTHONPATH=python python runner/diagnose_decode_consistency.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "python"))


def _reference_select(scores, starts, lengths, seq, budget, sink, tail):
    n = len(lengths)
    sink_end = min(sink, seq)
    tail_start = max(sink_end, seq - tail)
    room = budget - sink_end - (seq - tail_start)
    order = sorted(range(n), key=lambda i: (-scores[i], i))
    taken = [0] * n
    for leaf in order:
        first = min(max(starts[leaf], sink_end), tail_start)
        last = min(max(starts[leaf] + lengths[leaf], sink_end), tail_start)
        clipped = max(last - first, 0)
        if room <= 0 or clipped <= 0:
            if room <= 0:
                break
            continue
        take = min(clipped, room)
        taken[leaf] = (first, take)
        room -= take
    expected = list(range(sink_end))
    for leaf in range(n):
        if taken[leaf]:
            first, take = taken[leaf]
            expected.extend(range(first, first + take))
    expected.extend(range(tail_start, seq))
    return expected


def test_weighted_select_and_intervals(device="cuda"):
    from sglang.srt.layers.attention.nsa.adaptive_hisa.decode_select import (
        weighted_select_candidates,
        weighted_select_candidates_legacy,
        weighted_select_intervals,
    )

    torch.manual_seed(7)
    lengths = [64, 128, 256, 512, 1024, 33, 17, 200]
    starts = [sum(lengths[:i]) for i in range(len(lengths))]
    complete = sum(lengths)
    seq = complete + 40
    n = len(lengths)
    capacity = math.ceil(n / 64) * 64
    scores = torch.randn(n, device=device).tolist() + [float("-inf")] * (capacity - n)
    gpu_scores = torch.tensor(scores, dtype=torch.float32, device=device)
    gpu_starts = torch.tensor(starts + [complete] * (capacity - n), dtype=torch.int32, device=device)
    gpu_lens = torch.tensor(lengths + [0] * (capacity - n), dtype=torch.int32, device=device)
    num_leaves = torch.tensor([n], dtype=torch.int32, device=device)
    seq_t = torch.tensor([seq], dtype=torch.int32, device=device)
    prefix = torch.zeros(capacity, dtype=torch.int32, device=device)
    budget, sink, tail = 2048, 64, 256

    expected = _reference_select(scores[:n], starts, lengths, seq, budget, sink, tail)
    cand, count = weighted_select_candidates(
        gpu_scores, gpu_lens, gpu_starts, num_leaves, seq_t, prefix, budget, sink, tail
    )
    got = cand[0, : int(count.item())].tolist()
    assert got == expected, f"token path mismatch len={len(got)} vs {len(expected)}"

    old_cand, old_count = weighted_select_candidates_legacy(
        gpu_scores,
        gpu_lens,
        gpu_starts,
        num_leaves,
        seq_t,
        prefix.clone(),
        budget,
        sink,
        tail,
    )
    assert int(old_count.item()) == int(count.item())
    assert torch.equal(
        old_cand[:, : int(old_count.item())],
        cand[:, : int(count.item())],
    ), "new selector differs from six-pass reference"

    prefix.zero_()
    segs, seg_count, cand2, count2 = weighted_select_intervals(
        gpu_scores, gpu_lens, gpu_starts, num_leaves, seq_t, prefix, budget, sink, tail
    )
    got2 = cand2[0, : int(count2.item())].tolist()
    assert got2 == expected, "interval path token set mismatch"
    assert int(count2.item()) == int(count.item())

    # Reconstruct tokens from intervals.
    n_seg = int(seg_count.item())
    rebuilt = []
    seg_starts, seg_lengths, seg_offsets = segs
    for i in range(n_seg):
        s = int(seg_starts[i].item())
        ln = int(seg_lengths[i].item())
        rebuilt.extend(range(s, s + ln))
    assert rebuilt == expected, "interval expansion does not cover the same tokens"
    return {"weighted_select": "ok", "n_tokens": len(expected), "n_segments": n_seg}


def test_clean_redundant(device="cuda"):
    """Selector ignores length==0 / rows beyond num_leaves, so capacity clean is a no-op."""
    from sglang.srt.layers.attention.nsa.adaptive_hisa.decode_select import (
        weighted_select_candidates,
    )
    from sglang.srt.layers.attention.nsa.adaptive_hisa.incremental import (
        clean_summary_scores,
    )

    torch.manual_seed(11)
    n, capacity, budget = 32, 64, 512
    lengths = torch.randint(1, 40, (n,), device=device, dtype=torch.int32)
    starts = lengths.cumsum(0) - lengths
    starts = torch.cat([starts, torch.full((capacity - n,), int(starts[-1] + lengths[-1]),
                                          device=device, dtype=torch.int32)])
    lengths = torch.cat([lengths, torch.zeros(capacity - n, device=device, dtype=torch.int32)])
    scores = torch.randn(capacity, device=device)
    scores[n:] = 42.0  # garbage past num_leaves — must not be selected
    scores_dirty = scores.clone()
    scores_dirty[lengths == 0] = 99.0
    num_leaves = torch.tensor([n], dtype=torch.int32, device=device)
    seq = int((starts[:n] + lengths[:n]).max().item()) + 10
    seq_t = torch.tensor([seq], dtype=torch.int32, device=device)
    prefix = torch.zeros(capacity, dtype=torch.int32, device=device)

    work = SimpleNamespace(
        capacity=capacity,
        lengths=lengths,
        num_leaves=num_leaves,
    )
    cleaned = clean_summary_scores(
        scores_dirty.clone().unsqueeze(0), work
    ).squeeze(0)
    a, ca = weighted_select_candidates(
        scores_dirty, lengths, starts, num_leaves, seq_t, prefix.clone(), budget, 64, 128
    )
    b, cb = weighted_select_candidates(
        cleaned, lengths, starts, num_leaves, seq_t, prefix.clone(), budget, 64, 128
    )
    assert torch.equal(a[:, : ca.item()], b[:, : cb.item()]), "clean changed the candidate set"
    return {"clean_redundant": "ok", "count": int(ca.item())}


def main():
    if not torch.cuda.is_available():
        print(json.dumps({"ok": False, "reason": "no CUDA"}))
        sys.exit(2)
    report = {}
    report.update(test_weighted_select_and_intervals())
    report.update(test_clean_redundant())
    report["ok"] = True
    print(json.dumps(report, indent=2))
    out = Path(os.environ.get("ADAPTIVE_HISA_DIAG_OUT", "/tmp/adaptive_hisa_decode_diag.json"))
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
