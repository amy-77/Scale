"""Single-layer raw-FP8 partition microbenchmark.

Records CUDA events, launch counts, peak memory, and Triton register/spill
metadata for the fused tree and summary kernels. Intended for one free GPU.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa.config import get_config, reset_config_cache
from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_kernels as pk
from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_select as _sel
from sglang.srt.layers.attention.nsa.adaptive_hisa.pkey_fp8_builder import (
    build_key_tree_from_fp8,
    build_partition_from_fp8,
)


def _spill(fn) -> list[dict]:
    """Register / spill / shared-memory metadata of every compiled specialization."""
    out = []
    for val in getattr(fn, "device_caches", {}).values():
        cache = val[0] if isinstance(val, tuple) else val
        for comp in cache.values():
            try:
                comp._init_handles()
                out.append({"n_regs": comp.n_regs, "n_spills": comp.n_spills, "shared": comp.metadata.shared})
            except Exception as exc:  # pragma: no cover
                out.append({"error": repr(exc)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=32768)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--top", type=int, default=12, help="kernels to list by GPU time")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    os.environ.update({
        "SGLANG_NSA_ADAPTIVE_HISA_MODE": "build_only",
        "SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC": "key_sse",
        "SGLANG_NSA_ADAPTIVE_HISA_SPLIT_BACKEND": "gpu",
        "SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY": "sync_nonoverlap",
        "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR": "64",
    })
    reset_config_cache()
    cfg = get_config()
    gen = torch.Generator(device="cpu").manual_seed(0)
    raw = torch.randint(0, 126, (args.n, 128), generator=gen, dtype=torch.uint8)
    keys = raw.view(torch.float8_e4m3fn).to(device)
    scale = torch.linspace(0.5, 1.5, args.n, device=device)
    for _ in range(args.warmup):
        build_partition_from_fp8(keys, scale, args.n, cfg, profile=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        part = None
        for _ in range(args.iters):
            part = build_partition_from_fp8(keys, scale, args.n, cfg, profile=True)
    end.record()
    end.synchronize()
    rows = [e for e in prof.key_averages() if e.self_device_time_total]
    rows.sort(key=lambda e: -e.self_device_time_total)
    top = [
        {"kernel": e.key[:70], "us_per_iter": e.self_device_time_total / args.iters, "calls_per_iter": e.count / args.iters}
        for e in rows[:args.top]
    ]
    gpu_us = sum(e.self_device_time_total for e in rows) / args.iters
    flat, _ = build_key_tree_from_fp8(keys, scale, args.n, atom=cfg.atom, root=cfg.root)
    result = {
        "n": args.n,
        "radix_select": _sel.available(keys),
        "ms_per_iter": start.elapsed_time(end) / args.iters,
        "gpu_us_per_iter": gpu_us,
        "launches_per_iter": sum(e.count for e in rows) / args.iters,
        "peak_mb": torch.cuda.max_memory_allocated() / (1 << 20),
        "flat_bytes": flat.numel() * flat.element_size(),
        "num_leaves": int(part.num_leaves.item()) if part is not None else 0,
        "stage_ms_last_iter": {
            name: round(a.elapsed_time(b), 3) for name, (a, b) in (part.events.items() if part is not None else [])
        },
        "top_kernels": top,
        "tree_chunk_kernel": _spill(pk._fp8_chunk_tree_kernel),
        "tree_upper_kernel": _spill(pk._fp8_upper_tree_kernel),
        "leaf_totals_kernel": _spill(pk._leaf_totals_fp8_kernel),
        "leaf_gain_kernel": _spill(pk._dp_leaf_gain_kernel),
        "merge_cost_kernel": _spill(pk._merge_cost_kernel),
        "merge_match_kernel": _spill(pk._merge_match_kernel),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
