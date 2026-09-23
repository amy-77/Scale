#!/usr/bin/env python3
"""Paper microbenchmark: PyTorch/reference vs optimized partition operators.

The script intentionally compares GPU implementations only.  CPU reference
code is used by unit tests but is not reported as a GPU speedup baseline.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys
from pathlib import Path

import torch


PACKAGE = Path("/workspace/qyl/code/adaptive_0921_h202")
sys.path.insert(0, str(PACKAGE / "python"))
os.environ.update(
    {
        "SGLANG_NSA_ADAPTIVE_HISA_MODE": "adaptive_decode",
        "SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC": "key_sse",
        "SGLANG_NSA_ADAPTIVE_HISA_SPLIT_BACKEND": "gpu",
        "SGLANG_NSA_ADAPTIVE_HISA_BUILD_SUMMARIES": "1",
        "SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY": "sync_nonoverlap",
        "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR": "64",
        "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_ROUNDS": "8",
        "SGLANG_NSA_ADAPTIVE_HISA_RAW_FP8_BUILDER": "1",
        "SGLANG_NSA_ADAPTIVE_HISA_KEY_MOMENT_DTYPE": "fp64",
    }
)

from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_gpu as pg  # noqa: E402
from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_kernels as pk  # noqa: E402
from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_select as ps  # noqa: E402
from sglang.srt.layers.attention.nsa.adaptive_hisa.config import get_config  # noqa: E402
from sglang.srt.layers.attention.nsa.adaptive_hisa.pkey_fp8_builder import (  # noqa: E402
    _build_key_tree_torch,
    build_key_tree_from_fp8,
    build_partition_from_fp8,
    leaf_totals_from_fp8,
    summaries_from_totals,
)


@contextlib.contextmanager
def torch_reference_mode():
    """Force the vectorized PyTorch/sort branches in partition_gpu."""
    old_use = pk.use_triton
    old_available = ps.available
    pk.use_triton = lambda _tensor: False
    ps.available = lambda _tensor: False
    try:
        yield
    finally:
        pk.use_triton = old_use
        ps.available = old_available


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(q * len(ordered) + 0.999999) - 1))]


def measure(fn, warmup: int, samples: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(begin.elapsed_time(end)))
    return {
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
    }


def peak_mb(fn) -> float:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    # Keep the result alive until after the peak is sampled.
    _ = out
    return (torch.cuda.max_memory_allocated() - before) / (1 << 20)


def split(flat, layout, cfg):
    lam, _count0, _rounds, _used = pg.search_lambda(
        flat,
        layout,
        layout.n_tokens // cfg.summary_compression,
        candidates=cfg.lambda_candidates,
        rel_tol=cfg.lambda_rel_tol,
        max_rounds=cfg.lambda_max_rounds,
        deficit_tol=cfg.lambda_deficit_tol,
    )
    is_leaf, gain = pg.dp_leaf_and_gain(flat, layout, lam)
    is_leaf, repairs, _passes = pg.repair_leaves(
        flat,
        layout,
        is_leaf,
        layout.n_tokens // cfg.summary_compression,
        tie_passes=cfg.repair_tie_passes,
        gain=gain,
    )
    start, length = pg.emit_leaves(
        is_leaf, layout, layout.n_tokens // cfg.summary_compression
    )
    return lam, is_leaf, repairs, start, length


def merge(start, length, totals, lam, n_tokens, cfg):
    count = torch.tensor(
        start.numel(), dtype=torch.int64, device=start.device
    )
    target = torch.tensor(
        n_tokens // cfg.merge_target_divisor,
        dtype=torch.int64,
        device=start.device,
    )
    return pg.merge_sync_nonoverlap(
        start,
        length,
        count,
        totals,
        lam * cfg.merge_alpha,
        rounds=cfg.merge_target_rounds,
        max_merge_len=cfg.max_merge_len,
        n_tokens=n_tokens,
        target=target,
    )


def partition_signature(result) -> tuple[torch.Tensor, torch.Tensor, int]:
    start, length, count = result[0], result[1], result[2]
    n = int(count.item())
    return start[:n], length[:n], n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[8192, 16384, 32768, 65536, 131072],
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    cfg = get_config()
    device = torch.device("cuda")
    max_n = max(args.lengths)
    if max_n % cfg.root:
        raise ValueError("every length must be root-aligned")

    # Piecewise-correlated keys are closer to a production index cache than IID.
    gen = torch.Generator(device=device).manual_seed(args.seed)
    segments = torch.randn(
        (max_n + 63) // 64, 128, generator=gen, device=device
    )
    noise = 0.25 * torch.randn(
        max_n, 128, generator=gen, device=device
    )
    keys_fp32 = segments.repeat_interleave(64, dim=0)[:max_n] + noise
    scales = keys_fp32.abs().amax(dim=1).clamp_min(1e-6) / 448.0
    keys = (keys_fp32 / scales[:, None]).to(torch.float8_e4m3fn)
    del keys_fp32, noise, segments

    rows = []
    for n in args.lengths:
        layout = pg.TreeLayout(cfg.atom, cfg.root, n)

        def tree_torch():
            return _build_key_tree_torch(keys[:n], scales[:n], layout)

        def tree_triton():
            return build_key_tree_from_fp8(
                keys[:n], scales[:n], n, atom=cfg.atom, root=cfg.root
            )[0]

        old_flat = None
        if n > args.lengths[0]:
            old_n = n - args.lengths[0]
            old_flat = build_key_tree_from_fp8(
                keys[:old_n],
                scales[:old_n],
                old_n,
                atom=cfg.atom,
                root=cfg.root,
            )[0]

            def tree_incremental():
                return build_key_tree_from_fp8(
                    keys[:n],
                    scales[:n],
                    n,
                    atom=cfg.atom,
                    root=cfg.root,
                    tree_cache=(old_flat, old_n),
                )[0]
        else:
            tree_incremental = tree_triton

        flat_opt = tree_triton()
        flat_ref = tree_torch()
        tree_max_abs = float((flat_ref - flat_opt).abs().max().item())
        tree_equal = bool(torch.equal(flat_ref, flat_opt))
        inc_equal = bool(torch.equal(tree_incremental(), flat_opt))

        opt_split = split(flat_opt, layout, cfg)
        with torch_reference_mode():
            ref_split = split(flat_opt, layout, cfg)
        split_equal = bool(
            torch.equal(opt_split[1], ref_split[1])
            and torch.equal(opt_split[3], ref_split[3])
            and torch.equal(opt_split[4], ref_split[4])
        )

        totals = leaf_totals_from_fp8(
            keys[:n], scales[:n], opt_split[3], opt_split[4], n
        )
        opt_merge = merge(
            opt_split[3], opt_split[4], totals, opt_split[0], n, cfg
        )
        with torch_reference_mode():
            ref_merge = merge(
                opt_split[3], opt_split[4], totals, opt_split[0], n, cfg
            )
        os_, ol_, on = partition_signature(opt_merge)
        rs_, rl_, rn = partition_signature(ref_merge)
        merge_equal = bool(
            on == rn and torch.equal(os_, rs_) and torch.equal(ol_, rl_)
        )

        tree_ref_time = measure(tree_torch, args.warmup, args.samples)
        tree_opt_time = measure(tree_triton, args.warmup, args.samples)
        tree_inc_time = measure(tree_incremental, args.warmup, args.samples)
        with torch_reference_mode():
            split_ref_time = measure(
                lambda: split(flat_opt, layout, cfg), args.warmup, args.samples
            )
        split_opt_time = measure(
            lambda: split(flat_opt, layout, cfg), args.warmup, args.samples
        )
        with torch_reference_mode():
            merge_ref_time = measure(
                lambda: merge(
                    opt_split[3],
                    opt_split[4],
                    totals,
                    opt_split[0],
                    n,
                    cfg,
                ),
                args.warmup,
                args.samples,
            )
        merge_opt_time = measure(
            lambda: merge(
                opt_split[3],
                opt_split[4],
                totals,
                opt_split[0],
                n,
                cfg,
            ),
            args.warmup,
            args.samples,
        )

        def full_build():
            part = build_partition_from_fp8(keys[:n], scales[:n], n, cfg)
            summary = summaries_from_totals(
                part.meta.pop("_merge_totals"), part.leaf_len, None
            )
            return part, summary

        full_time = measure(full_build, args.warmup, args.samples)
        row = {
            "n": n,
            "seed": args.seed,
            "tree_parity": {
                "bitwise": tree_equal,
                "max_abs": tree_max_abs,
                "incremental_bitwise": inc_equal,
            },
            "split_partition_equal": split_equal,
            "merge_partition_equal": merge_equal,
            "tree_torch": tree_ref_time,
            "tree_raw_fp8_triton": tree_opt_time,
            "tree_incremental_triton": tree_inc_time,
            "split_torch_sort": split_ref_time,
            "split_fused_dp_radix": split_opt_time,
            "merge_torch_sort": merge_ref_time,
            "merge_triton_radix": merge_opt_time,
            "full_partition_summary": full_time,
            "tree_peak_mb_torch": peak_mb(tree_torch),
            "tree_peak_mb_triton": peak_mb(tree_triton),
            "final_leaves": on,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    payload = {
        "hardware": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "config": {
            "atom": cfg.atom,
            "root": cfg.root,
            "summary_compression": cfg.summary_compression,
            "merge_target_divisor": cfg.merge_target_divisor,
            "merge_target_rounds": cfg.merge_target_rounds,
            "lambda_candidates": cfg.lambda_candidates,
            "lambda_max_rounds": cfg.lambda_max_rounds,
        },
        "warmup": args.warmup,
        "samples": args.samples,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
