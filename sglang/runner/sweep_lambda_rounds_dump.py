"""λ-search round sweep on real DeepSeek-V3.2 indexer dumps (score-space energies).

For every (layer, request) directory of an ``adaptive_hisa_dump`` tree the
score field is rebuilt from the exported FP8 keys and the saved query rows
(``score[c, t] = k_scale[t] * sum_h w[c,h] * relu(q_fp8[c,h] . k_fp8[t])``, the
DSA indexer logit), and the paper schedule (dyadic λ-split to ``M = N/8``,
exact repair, two synchronous Ward rounds at α=1) is run with the λ search
capped at ``R = 1..10`` rounds. Recorded per sample and ``R``:

  rounds_used, dp_leaves, deficit ratio, split-level and post-merge leaf
  agreement with the ``R = 10`` reference, and the λ found.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa.config import get_config, reset_config_cache
from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import build_partition_gpu

ENV_BASE = {
    "MODE": "build_only",
    "SPLIT_BACKEND": "gpu",
    "PARTITION_VALIDATE": "0",
    "MERGE_ROUNDS": "2",
    "MERGE_ALPHA": "1.0",
    "MAX_MERGE_LEN": "0",
    "LAMBDA_REL_TOL": "1e-12",
    "LAMBDA_DEFICIT_TOL": "-1",
}


def _cfg(**extra):
    env = dict(ENV_BASE)
    env.update(extra)
    for k, v in env.items():
        os.environ[f"SGLANG_NSA_ADAPTIVE_HISA_{k}"] = v
    reset_config_cache()
    return get_config()


def load_scores(sample_dir: Path, device) -> tuple[torch.Tensor, int]:
    m = json.load(open(sample_dir / "manifest.json"))
    k = torch.load(sample_dir / m["key_shards"][0], map_location=device)
    kf = k["k_fp8"].float()
    ks = k["k_scale"]
    n = kf.shape[0]
    rows = []
    for qs in m["query_shards"]:
        q = torch.load(sample_dir / qs, map_location=device)
        qf = q["q_fp8"][0].float()  # [H, D]
        w = q["w"][0]  # [H], already includes q_scale and the 1/sqrt scalings
        rows.append((w[:, None] * torch.relu(qf @ kf.T)).sum(0) * ks)
    return torch.stack(rows), n  # [C, N] float32


def _leaf_set(part) -> set[tuple[int, int]]:
    k = int(part.num_leaves.item())
    return set(zip(part.leaf_start[:k].tolist(), part.leaf_len[:k].tolist()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/data/jz/adaptive_hisa_dump_sweep_20260918/rank_00")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    device = torch.device("cuda")
    dirs = sorted(p for p in Path(args.dump).glob("layer_*/req_*") if (p / "manifest.json").exists())
    if args.max_samples:
        dirs = dirs[: args.max_samples]
    rows = []
    for d in dirs:
        scores, n = load_scores(d, device)
        layer, req = d.parent.name, d.name
        ref_split = ref_final = None
        for r in range(10, 0, -1):  # R=10 first: it is the reference
            cfg_split = _cfg(MERGE_POLICY="off", LAMBDA_MAX_ROUNDS=str(r))
            split = build_partition_gpu(scores, n, cfg_split)
            cfg_final = _cfg(MERGE_POLICY="sync_nonoverlap", LAMBDA_MAX_ROUNDS=str(r))
            final = build_partition_gpu(scores, n, cfg_final)
            torch.cuda.synchronize()
            s_set, f_set = _leaf_set(split), _leaf_set(final)
            if r == 10:
                ref_split, ref_final = s_set, f_set
            budget = split.capacity
            dp_leaves = int(split.dp_leaves.item())
            row = {
                "layer": layer, "req": req, "n_complete": split.n_complete, "budget": budget, "R": r,
                "rounds_used": int(split.meta["lambda_rounds_used"].item()),
                "dp_leaves": dp_leaves, "deficit_ratio": (budget - dp_leaves) / budget,
                "lambda": float(split.lam.item()),
                "split_agree": len(s_set & ref_split) / len(ref_split),
                "final_leaves": len(f_set),
                "final_agree": len(f_set & ref_final) / len(ref_final),
                "round_merges": final.round_merges.tolist(),
            }
            rows.append(row)
        last = [x for x in rows if x["layer"] == layer and x["req"] == req]
        print(f"{layer} {req} N={n} M={budget}: " + " ".join(
            f"R{x['R']}:{x['deficit_ratio']*100:.2f}%/{x['split_agree']*100:.0f}%/{x['final_agree']*100:.0f}%"
            for x in sorted(last, key=lambda x: x["R"])[2:7]))
    if args.json:
        json.dump(rows, open(args.json, "w"))
    # aggregate
    print("\nR  max_deficit  mean_deficit  min_split_agree  mean_split_agree  min_final_agree  mean_final_agree")
    for r in range(1, 11):
        sel = [x for x in rows if x["R"] == r]
        f = lambda key, fn: fn(x[key] for x in sel)
        print(f"{r:2d} {f('deficit_ratio', max)*100:10.3f}% {f('deficit_ratio', lambda g: sum(g)/len(sel))*100:11.3f}% "
              f"{f('split_agree', min)*100:14.1f}% {f('split_agree', lambda g: sum(g)/len(sel))*100:15.1f}% "
              f"{f('final_agree', min)*100:14.1f}% {f('final_agree', lambda g: sum(g)/len(sel))*100:15.1f}%")


if __name__ == "__main__":
    main()
