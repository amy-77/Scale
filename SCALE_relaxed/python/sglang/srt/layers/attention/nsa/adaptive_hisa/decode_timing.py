"""Opt-in CUDA-event / NVTX stage timing for Adaptive-HISA decode.

Enable with ``SGLANG_NSA_ADAPTIVE_HISA_DECODE_TIMING=1``. Optional NVTX ranges
also require ``SGLANG_NSA_ADAPTIVE_HISA_DECODE_NVTX=1`` (nsys-friendly).

Stages: seal, schedule, coarse_mqa, weighted_select, fine_rescore, topk.
With ``SGLANG_NSA_ADAPTIVE_HISA_SELECTOR_PROFILE=1``, weighted_select also
contains separately measured ``weighted_prefix`` and ``weighted_expand``.
"""
from __future__ import annotations

import os
import threading
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

_TRUE = {"1", "true", "yes", "on"}


def timing_enabled() -> bool:
    return os.environ.get("SGLANG_NSA_ADAPTIVE_HISA_DECODE_TIMING", "0").strip().lower() in _TRUE


def nvtx_enabled() -> bool:
    return os.environ.get("SGLANG_NSA_ADAPTIVE_HISA_DECODE_NVTX", "0").strip().lower() in _TRUE


@dataclass
class _StageAcc:
    count: int = 0
    total_ms: float = 0.0
    samples_ms: list = field(default_factory=list)

    def add(self, ms: float) -> None:
        self.count += 1
        self.total_ms += ms
        if len(self.samples_ms) < 4096:
            self.samples_ms.append(ms)


_LOCK = threading.Lock()
_STAGES: dict[str, _StageAcc] = defaultdict(_StageAcc)
_PENDING: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []


def reset() -> None:
    with _LOCK:
        _STAGES.clear()
        _PENDING.clear()


def _flush_pending() -> None:
    if not _PENDING:
        return
    # Resolve completed pairs; leave unfinished ones for the next flush.
    keep: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    for name, start, end in _PENDING:
        if not end.query():
            keep.append((name, start, end))
            continue
        ms = start.elapsed_time(end)
        _STAGES[name].add(ms)
    _PENDING[:] = keep


@contextmanager
def stage(name: str, device=None, enabled: bool | None = None):
    """Record one named stage. No-op when timing is disabled."""
    on = timing_enabled() if enabled is None else enabled
    use_nvtx = on and nvtx_enabled()
    if use_nvtx:
        torch.cuda.nvtx.range_push(f"adaptive_hisa/{name}")
    if not on or device is None or torch.device(device).type != "cuda":
        try:
            yield
        finally:
            if use_nvtx:
                torch.cuda.nvtx.range_pop()
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield
    finally:
        end.record()
        with _LOCK:
            _PENDING.append((name, start, end))
            if len(_PENDING) >= 64:
                _flush_pending()
        if use_nvtx:
            torch.cuda.nvtx.range_pop()


def snapshot(synchronize: bool = True) -> dict:
    """Return per-stage count / mean / p50 / p95 in milliseconds."""
    with _LOCK:
        if synchronize and _PENDING:
            torch.cuda.synchronize()
            _flush_pending()
        out = {}
        for name, acc in _STAGES.items():
            if not acc.count:
                continue
            samples = sorted(acc.samples_ms) or [0.0]
            p50 = samples[len(samples) // 2]
            p95 = samples[min(len(samples) - 1, int(len(samples) * 0.95))]
            out[name] = {
                "count": acc.count,
                "mean_ms": acc.total_ms / acc.count,
                "p50_ms": p50,
                "p95_ms": p95,
                "total_ms": acc.total_ms,
            }
        return out
