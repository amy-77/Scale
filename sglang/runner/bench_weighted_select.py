#!/usr/bin/env python3
"""Microbenchmark old/new Adaptive selector and fixed-block HISA Stage 3.

The Adaptive timings include exact guard handling and materialisation of the
8192 candidate token IDs. HISA emits only 128 block IDs, so it is a useful
lower bound rather than a semantics-equivalent competitor.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "python"))


def length_distribution(
    name: str,
    n: int,
    generator: torch.Generator,
    real_lengths: list[int] | None,
) -> list[int]:
    if name == "fixed64":
        return [64] * n
    if name == "adaptive_real":
        if real_lengths:
            return [max(1, int(real_lengths[i % len(real_lengths)])) for i in range(n)]
        # Calibrated to the saved target64 stats: median near 64, a small
        # short-leaf tail, and occasional 128-208 token leaves.
        support = torch.tensor(
            [8, 16, 24, 32, 40, 48, 56, 64, 64, 64, 72, 80, 96, 112, 128, 160, 208]
        )
        pick = torch.randint(
            0, support.numel(), (n,), generator=generator
        )
        return support[pick].tolist()
    if name == "highly_skewed":
        support = torch.tensor([1, 2, 4, 8, 16, 64, 256, 512, 1024])
        pick = torch.randint(
            0, support.numel(), (n,), generator=generator
        )
        return support[pick].tolist()
    raise ValueError(name)


def measure(fn, warmup: int, iterations: int, rounds: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000.0 / iterations)
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(ordered),
        "p95_us": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
        "min_us": ordered[0],
    }


def run_case(
    n: int,
    distribution: str,
    *,
    budget: int,
    sink: int,
    tail: int,
    warmup: int,
    iterations: int,
    rounds: int,
    real_lengths: list[int] | None,
) -> dict:
    from sglang.srt.layers.attention.nsa.adaptive_hisa.decode_select import (
        expand_selected_candidates,
        selector_profile_snapshot,
        weighted_select_candidates,
        weighted_select_candidates_legacy,
        weighted_prefix_profile,
    )

    generator = torch.Generator().manual_seed(20260920 + n)
    lengths = length_distribution(distribution, n, generator, real_lengths)
    starts = torch.tensor(
        [0] + lengths[:-1], dtype=torch.int64
    ).cumsum(0).to(torch.int32)
    lengths_t = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    starts_t = starts.to(device="cuda")
    # Quantisation creates realistic ties and exercises stable leaf-index
    # semantics while retaining multiple FP32 high-byte buckets.
    scores = (
        torch.randint(-4096, 4097, (n,), generator=generator).float() / 128.0
    ).to(device="cuda")
    num_leaves = torch.tensor([n], dtype=torch.int32, device="cuda")
    seq_len = torch.tensor([sum(lengths)], dtype=torch.int32, device="cuda")
    prefix_fast = torch.empty(n, dtype=torch.int32, device="cuda")
    prefix_old = torch.empty_like(prefix_fast)
    candidates_fast = torch.empty((1, budget), dtype=torch.int32, device="cuda")
    candidates_old = torch.empty_like(candidates_fast)
    count_fast = torch.empty(1, dtype=torch.int32, device="cuda")
    count_old = torch.empty_like(count_fast)

    def fast():
        weighted_select_candidates(
            scores,
            lengths_t,
            starts_t,
            num_leaves,
            seq_len,
            prefix_fast,
            budget,
            sink,
            tail,
            candidates_out=candidates_fast,
            count_out=count_fast,
        )

    def legacy():
        weighted_select_candidates_legacy(
            scores,
            lengths_t,
            starts_t,
            num_leaves,
            seq_len,
            prefix_old,
            budget,
            sink,
            tail,
            candidates_out=candidates_old,
            count_out=count_old,
        )

    def prefix_only():
        weighted_prefix_profile(
            scores,
            lengths_t,
            starts_t,
            num_leaves,
            seq_len,
            prefix_fast,
            budget,
            sink,
            tail,
        )

    def expand_only():
        expand_selected_candidates(
            prefix_fast,
            starts_t,
            lengths_t,
            num_leaves,
            seq_len,
            budget,
            sink,
            tail,
            candidates_out=candidates_fast,
            count_out=count_fast,
        )

    fast()
    legacy()
    torch.cuda.synchronize()
    count = int(count_fast.item())
    if int(count_old.item()) != count or not torch.equal(
        candidates_fast[:, :count], candidates_old[:, :count]
    ):
        raise AssertionError(f"selector parity failed for n={n} {distribution}")

    result = {
        "n": n,
        "distribution": distribution,
        "selected_tokens": count,
        "new_weighted": measure(fast, warmup, iterations, rounds),
        "old_weighted": measure(legacy, warmup, iterations, rounds),
    }
    result["speedup"] = (
        result["old_weighted"]["median_us"]
        / result["new_weighted"]["median_us"]
    )
    selector_profile_snapshot(reset=True)
    prefix_only()
    torch.cuda.synchronize()
    result["new_prefix_profile"] = measure(
        prefix_only, warmup, iterations, rounds
    )
    result["new_expand"] = measure(expand_only, warmup, iterations, rounds)
    result["threshold_bucket"] = selector_profile_snapshot(reset=True)

    try:
        from sglang.srt.layers.attention.nsa.hisa.fast_topk_runtime import (
            fast_topk_runtime,
        )

        def hisa():
            fast_topk_runtime(scores.view(1, -1), min(128, n))

        result["hisa_fast_topk"] = measure(hisa, warmup, iterations, rounds)
    except Exception as error:
        result["hisa_fast_topk_error"] = repr(error)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[512, 1024, 1536, 2048, 2560]
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=["fixed64", "adaptive_real", "highly_skewed"],
        choices=["fixed64", "adaptive_real", "highly_skewed"],
    )
    parser.add_argument(
        "--real-lengths",
        type=Path,
        help="optional JSON list of measured adaptive leaf lengths",
    )
    parser.add_argument("--budget", type=int, default=8192)
    parser.add_argument("--sink", type=int, default=64)
    parser.add_argument("--tail", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("/tmp/weighted_select_bench.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    real_lengths = None
    if args.real_lengths:
        real_lengths = json.loads(args.real_lengths.read_text(encoding="utf-8"))
        if not isinstance(real_lengths, list) or not real_lengths:
            raise ValueError("--real-lengths must contain a non-empty JSON list")

    report = []
    for distribution in args.distributions:
        for n in args.sizes:
            row = run_case(
                n,
                distribution,
                budget=args.budget,
                sink=args.sink,
                tail=args.tail,
                warmup=args.warmup,
                iterations=args.iterations,
                rounds=args.rounds,
                real_lengths=real_lengths,
            )
            report.append(row)
            print(json.dumps(row), flush=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
