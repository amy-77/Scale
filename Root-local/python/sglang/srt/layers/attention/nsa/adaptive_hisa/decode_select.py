"""Exact graph-safe token-budgeted leaf selector.

The CUDA op is JIT-compiled once and cached by PyTorch. Its production path
uses a token-weighted threshold histogram, packs only the threshold bucket,
and refines that small set; capacities above 4096 use the original exact
six-pass selector. The selected token set is identical to the h20-1 reference
``select_candidates`` (sink + tail guards, clipped leaves, stable score/index
ties, exact fill with a partial crossing leaf). Candidates are emitted in
logical-token order by direct interval expansion.

Optional interval expansion emits ``(start, length, output_offset)`` segments
covering the same token set, for the segment-aware fine scorer.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_HERE = Path(__file__).resolve().parent
_PROFILE_STATS: dict[str, torch.Tensor] = {}
_PROFILE_SIZE = 106


def warp_opt_mask() -> int:
    """Compile-time bitmask for the fast-path warp-level variants
    (1 = warp-aggregated Phase A histogram, 2 = ballot compaction of the
    threshold bucket, 4 = warp-aggregated Phase C histograms). Selection is
    identical for every value; only the number of shared atomics changes.
    Default 0: on H20 the per-leaf shared atomics beat ``__match_any_sync``
    (see runner/bench_weighted_select.py). ``ADAPTIVE_HISA_SELECT_WARP_OPT=7``
    enables all three."""
    raw = os.environ.get("ADAPTIVE_HISA_SELECT_WARP_OPT", "0").strip().lower()
    if raw in ("", "0", "false", "off", "no"):
        return 0
    if raw in ("1", "true", "on", "yes", "all"):
        return 7
    return int(raw) & 7


def _load_module():
    load(
        name="adaptive_hisa_weighted_select",
        sources=[str(_HERE / "csrc" / "weighted_select.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            f"-DADAPTIVE_HISA_WS_WARP_OPT={warp_opt_mask()}",
        ],
        is_python_module=False,
        verbose=bool(
            int(os.environ.get("ADAPTIVE_HISA_SELECT_VERBOSE", "0"))
        ),
    )


_load_module()


def weighted_select_candidates(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    starts: torch.Tensor,
    num_leaves: torch.Tensor,
    seq_len: torch.Tensor,
    selected_prefix: torch.Tensor,
    budget: int,
    sink: int,
    tail: int,
    *,
    candidates_out: torch.Tensor | None = None,
    count_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Emit ``[0, sink) + clipped score-prefix leaves + [seq-tail, seq)``.

    The reference emits leaves in score order. This op emits the same selected
    token set in logical order, which is deterministic and does not affect
    exact reranking or the final Top-K.
    """
    if candidates_out is None:
        candidates_out = torch.empty(
            (1, budget), dtype=torch.int32, device=scores.device
        )
    if count_out is None:
        count_out = torch.empty((1,), dtype=torch.int32, device=scores.device)
    torch.ops.adaptive_hisa_select.weighted_select(
        scores,
        lengths,
        starts,
        num_leaves,
        seq_len.reshape(-1),
        selected_prefix,
        candidates_out,
        count_out,
        budget,
        sink,
        tail,
    )
    return candidates_out, count_out


def weighted_select_candidates_legacy(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    starts: torch.Tensor,
    num_leaves: torch.Tensor,
    seq_len: torch.Tensor,
    selected_prefix: torch.Tensor,
    budget: int,
    sink: int,
    tail: int,
    *,
    candidates_out: torch.Tensor | None = None,
    count_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Benchmark/reference entry point for the original six-pass selector."""
    if candidates_out is None:
        candidates_out = torch.empty(
            (1, budget), dtype=torch.int32, device=scores.device
        )
    if count_out is None:
        count_out = torch.empty((1,), dtype=torch.int32, device=scores.device)
    torch.ops.adaptive_hisa_select.weighted_select_legacy(
        scores,
        lengths,
        starts,
        num_leaves,
        seq_len.reshape(-1),
        selected_prefix,
        candidates_out,
        count_out,
        budget,
        sink,
        tail,
    )
    return candidates_out, count_out


def _profile_buffer(device: torch.device) -> torch.Tensor:
    key = str(torch.device(device))
    value = _PROFILE_STATS.get(key)
    if value is None:
        value = torch.zeros(_PROFILE_SIZE, dtype=torch.int32, device=device)
        _PROFILE_STATS[key] = value
    return value


def weighted_prefix_profile(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    starts: torch.Tensor,
    num_leaves: torch.Tensor,
    seq_len: torch.Tensor,
    selected_prefix: torch.Tensor,
    budget: int,
    sink: int,
    tail: int,
) -> None:
    """Launch only prefix selection and accumulate threshold-bucket ratios."""
    torch.ops.adaptive_hisa_select.weighted_prefix_profile(
        scores,
        lengths,
        starts,
        num_leaves,
        seq_len.reshape(-1),
        selected_prefix,
        _profile_buffer(scores.device),
        budget,
        sink,
        tail,
    )


def expand_selected_candidates(
    selected_prefix: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    num_leaves: torch.Tensor,
    seq_len: torch.Tensor,
    budget: int,
    sink: int,
    tail: int,
    *,
    candidates_out: torch.Tensor,
    count_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch only direct interval-to-token expansion."""
    torch.ops.adaptive_hisa_select.expand_selected(
        selected_prefix,
        starts,
        lengths,
        seq_len.reshape(-1),
        num_leaves,
        candidates_out,
        count_out,
        budget,
        sink,
        tail,
    )
    return candidates_out, count_out


def selector_profile_snapshot(reset: bool = False) -> dict:
    """Synchronize and summarize real threshold-bucket ratio observations."""
    if not _PROFILE_STATS:
        return {}
    merged = [0] * _PROFILE_SIZE
    for tensor in _PROFILE_STATS.values():
        values = tensor.cpu().tolist()
        merged = [a + int(b) for a, b in zip(merged, values)]
    histogram = merged[:101]
    fast_calls = sum(histogram)

    def percentile(q: float) -> int | None:
        if not fast_calls:
            return None
        target = max(1, math.ceil(fast_calls * q))
        cumulative = 0
        for ratio, count in enumerate(histogram):
            cumulative += count
            if cumulative >= target:
                return ratio
        return 100

    result = {
        "calls": merged[101],
        "fast_calls": fast_calls,
        "fallback_calls": merged[105],
        "mean_bucket_ratio_pct": (
            100.0 * merged[103] / merged[102] if merged[102] else None
        ),
        "p50_bucket_ratio_pct": percentile(0.50),
        "p90_bucket_ratio_pct": percentile(0.90),
        "p99_bucket_ratio_pct": percentile(0.99),
        "max_bucket_leaves": merged[104],
        "ratio_histogram": histogram,
    }
    if reset:
        for tensor in _PROFILE_STATS.values():
            tensor.zero_()
    return result


def weighted_select_intervals(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    starts: torch.Tensor,
    num_leaves: torch.Tensor,
    seq_len: torch.Tensor,
    selected_prefix: torch.Tensor,
    budget: int,
    sink: int,
    tail: int,
    *,
    candidates_out: torch.Tensor | None = None,
    count_out: torch.Tensor | None = None,
    seg_starts_out: torch.Tensor | None = None,
    seg_lengths_out: torch.Tensor | None = None,
    seg_offsets_out: torch.Tensor | None = None,
    seg_count_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same token set as ``weighted_select_candidates``, plus interval metadata.

    Returns ``(seg_starts, seg_count, candidates, count)``. Segment buffers
    store packed ``(start, length, output_offset)`` rows; ``seg_count`` is the
    number of valid segments (≤ num_leaves + 2).
    """
    capacity = int(scores.numel())
    if candidates_out is None:
        candidates_out = torch.empty(
            (1, budget), dtype=torch.int32, device=scores.device
        )
    if count_out is None:
        count_out = torch.empty((1,), dtype=torch.int32, device=scores.device)
    if seg_starts_out is None:
        seg_starts_out = torch.empty(
            (capacity + 2,), dtype=torch.int32, device=scores.device
        )
    if seg_lengths_out is None:
        seg_lengths_out = torch.empty_like(seg_starts_out)
    if seg_offsets_out is None:
        seg_offsets_out = torch.empty_like(seg_starts_out)
    if seg_count_out is None:
        seg_count_out = torch.empty((1,), dtype=torch.int32, device=scores.device)
    torch.ops.adaptive_hisa_select.weighted_select_intervals(
        scores,
        lengths,
        starts,
        num_leaves,
        seq_len.reshape(-1),
        selected_prefix,
        candidates_out,
        count_out,
        seg_starts_out,
        seg_lengths_out,
        seg_offsets_out,
        seg_count_out,
        budget,
        sink,
        tail,
    )
    return (
        (seg_starts_out, seg_lengths_out, seg_offsets_out),
        seg_count_out,
        candidates_out,
        count_out,
    )
