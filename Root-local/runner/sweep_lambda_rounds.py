"""λ-search round sweep: how many full-tree DP rounds does the exact repair make redundant?

Arms (all K=16 candidates):
  A  rel_tol=1e-12                 -> 10 rounds (current default)
  B  rel_tol=1e-6                  ->  5 rounds
  C  rel_tol=1e-4                  ->  4 rounds
  D  max 5 rounds + leaf-deficit early stop (ratio sweep)

Per arm and sequence length we record the rounds actually executed, DP leaves,
repair deficit ratio, λ-search / repair stage time (CUDA events) and the final
leaf-map agreement with arm A (fraction of A's leaves reproduced exactly).
Structural columns are deterministic; timings are only indicative on a shared GPU.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa.config import get_config, reset_config_cache
from sglang.srt.layers.attention.nsa.adaptive_hisa.pkey_fp8_builder import build_partition_from_fp8

BASE_ENV = {
    "SGLANG_NSA_ADAPTIVE_HISA_MODE": "build_only",
    "SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC": "key_sse",
    "SGLANG_NSA_ADAPTIVE_HISA_SPLIT_BACKEND": "gpu",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY": "sync_nonoverlap",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR": "64",
}

ARMS = [
    ("A_tol1e-12", {"LAMBDA_REL_TOL": "1e-12", "LAMBDA_MAX_ROUNDS": "0", "LAMBDA_DEFICIT_TOL": "-1"}),
    ("B_tol1e-6", {"LAMBDA_REL_TOL": "1e-6", "LAMBDA_MAX_ROUNDS": "0", "LAMBDA_DEFICIT_TOL": "-1"}),
    ("C_tol1e-4", {"LAMBDA_REL_TOL": "1e-4", "LAMBDA_MAX_ROUNDS": "0", "LAMBDA_DEFICIT_TOL": "-1"}),
    ("D_max5_exact", {"LAMBDA_REL_TOL": "1e-12", "LAMBDA_MAX_ROUNDS": "5", "LAMBDA_DEFICIT_TOL": "0"}),
    ("D_max5_0.05%", {"LAMBDA_REL_TOL": "1e-12", "LAMBDA_MAX_ROUNDS": "5", "LAMBDA_DEFICIT_TOL": "0.0005"}),
    ("D_max5_0.1%", {"LAMBDA_REL_TOL": "1e-12", "LAMBDA_MAX_ROUNDS": "5", "LAMBDA_DEFICIT_TOL": "0.001"}),
    ("D_max5_0.25%", {"LAMBDA_REL_TOL": "1e-12", "LAMBDA_MAX_ROUNDS": "5", "LAMBDA_DEFICIT_TOL": "0.0025"}),
    ("D_max5_0.5%", {"LAMBDA_REL_TOL": "1e-12", "LAMBDA_MAX_ROUNDS": "5", "LAMBDA_DEFICIT_TOL": "0.005"}),
    ("D_max10_0.1%", {"LAMBDA_REL_TOL": "1e-12", "LAMBDA_MAX_ROUNDS": "0", "LAMBDA_DEFICIT_TOL": "0.001"}),
]


def _cfg(extra: dict):
    env = dict(BASE_ENV)
    env.update({f"SGLANG_NSA_ADAPTIVE_HISA_{k}": v for k, v in extra.items()})
    os.environ.update(env)
    reset_config_cache()
    return get_config()


def _keys(n: int, seed: int, device, kind: str):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "uniform":
        raw = torch.randint(0, 126, (n, 128), generator=gen, dtype=torch.uint8)
    else:
        # piecewise-constant "segments" with noise: closer to real KV structure
        n_seg = max(1, n // 48)
        centers = torch.randint(0, 126, (n_seg, 128), generator=gen, dtype=torch.uint8)
        lengths = torch.randint(8, 96, (n_seg,), generator=gen)
        idx = torch.repeat_interleave(torch.arange(n_seg), lengths)[:n]
        if idx.numel() < n:
            idx = torch.cat((idx, torch.full((n - idx.numel(),), n_seg - 1)))
        noise = torch.randint(-3, 4, (n, 128), generator=gen)
        raw = (centers[idx].to(torch.int64) + noise).clamp(0, 125).to(torch.uint8)
    keys = raw.view(torch.float8_e4m3fn).to(device)
    scale = (torch.rand(n, generator=gen) + 0.5).to(device)
    return keys, scale


def _leaf_set(part) -> set[tuple[int, int]]:
    n = int(part.num_leaves.item())
    s = part.leaf_start[:n].tolist()
    l = part.leaf_len[:n].tolist()
    return set(zip(s, l))


def _stage_ms(part, name: str) -> float:
    a, b = part.events[name]
    return a.elapsed_time(b)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[32768, 131072])
    parser.add_argument("--kind", choices=["uniform", "segments"], default="segments")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--reps", type=int, default=3, help="timing repetitions per arm")
    parser.add_argument("--json", type=str, default="")
    parser.add_argument(
        "--per-round", action="store_true",
        help="instead of the arms, cap rounds at 1..10 to trace deficit ratio vs DP rounds",
    )
    args = parser.parse_args()
    device = torch.device("cuda")
    arms = ARMS
    if args.per_round:
        arms = [
            (f"R{r:02d}", {"LAMBDA_REL_TOL": "1e-12", "LAMBDA_MAX_ROUNDS": str(r), "LAMBDA_DEFICIT_TOL": "-1"})
            for r in range(1, 11)
        ]
        arms.insert(0, ("A_tol1e-12", dict(ARMS[0][1])))  # reference for leaf-map match
    # warm up Triton/CUDA compilation so the first arm is not charged for it
    k0, s0 = _keys(min(args.lengths), 12345, device, args.kind)
    build_partition_from_fp8(k0, s0, min(args.lengths), _cfg(ARMS[0][1]), profile=True)
    torch.cuda.synchronize()
    rows = []
    for n in args.lengths:
        budget = n // 8
        for seed in range(args.seeds):
            keys, scale = _keys(n, seed, device, args.kind)
            ref_set = None
            for name, extra in arms:
                cfg = _cfg(extra)
                part = None
                dp_ms = rep_ms = 0.0
                dp_min = rep_min = float("inf")
                for _ in range(args.reps):
                    part = build_partition_from_fp8(keys, scale, n, cfg, profile=True)
                    torch.cuda.synchronize()
                    d, r = _stage_ms(part, "dp_s"), _stage_ms(part, "repair_s")
                    dp_ms += d
                    rep_ms += r
                    dp_min, rep_min = min(dp_min, d), min(rep_min, r)
                leaves = _leaf_set(part)
                if ref_set is None:
                    ref_set = leaves
                dp_leaves = int(part.dp_leaves.item())
                row = {
                    "n": n,
                    "seed": seed,
                    "arm": name,
                    "rounds_max": int(part.meta["lambda_rounds"]),
                    "rounds_used": int(part.meta["lambda_rounds_used"].item()),
                    "dp_leaves": dp_leaves,
                    "deficit": budget - dp_leaves,
                    "deficit_ratio": (budget - dp_leaves) / budget,
                    "repairs": int(part.repairs.item()),
                    "lambda": float(part.lam.item()),
                    "final_leaves": int(part.num_leaves.item()),
                    "leafmap_match_vs_A": len(leaves & ref_set) / max(1, len(ref_set)),
                    "dp_ms": dp_ms / args.reps,
                    "repair_ms": rep_ms / args.reps,
                    # min over reps: robust to co-tenant stalls on a shared GPU
                    "dp_ms_min": dp_min,
                    "repair_ms_min": rep_min,
                }
                rows.append(row)
                print(
                    f"n={n:6d} seed={seed} {name:14s} rounds={row['rounds_used']:2d}/{row['rounds_max']:2d} "
                    f"dp_leaves={dp_leaves:6d} deficit={row['deficit']:5d} ({row['deficit_ratio']*100:.3f}%) "
                    f"match_A={row['leafmap_match_vs_A']*100:6.2f}% dp={row['dp_ms_min']:.3f}ms repair={row['repair_ms_min']:.3f}ms "
                    f"lambda={row['lambda']:.6g}"
                )
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
