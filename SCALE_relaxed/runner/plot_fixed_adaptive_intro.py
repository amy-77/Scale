#!/usr/bin/env python3
"""Render the Fixed vs Adaptive intro figure from experiment outputs.

Panels
  A  mechanism example (exact token scores vs fixed/adaptive mean estimates)
  B  matched-summary Pareto: Top-2048 recall vs #summaries (= L/K)
  C  paired adaptive−fixed recall ECDF (+ high-hetero callout)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def _exact_scores(sample_dir: Path, query_name: str, device="cpu"):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    manifest = json.loads((sample_dir / "manifest.json").read_text())
    key_parts, scale_parts = [], []
    for name in manifest["key_shards"]:
        blob = torch.load(sample_dir / name, map_location="cpu", weights_only=False)
        key_parts.append(blob["k_fp8"])
        scale_parts.append(blob["k_scale"].reshape(-1))
    n_complete = int(key_parts[0].shape[0])
    k_fp8 = torch.cat(key_parts, 0)
    if k_fp8.dtype == torch.uint8:
        k_fp8 = k_fp8.view(torch.float8_e4m3fn)
    k_scale = torch.cat(scale_parts, 0).float()
    q = torch.load(sample_dir / query_name, map_location="cpu", weights_only=False)
    q_fp8 = q["q_fp8"][0]
    if q_fp8.dtype == torch.uint8:
        q_fp8 = q_fp8.view(torch.float8_e4m3fn)
    w = q["w"][0].float()
    ke = int(q["ke"][0])
    keys = k_fp8[:ke].float() * k_scale[:ke, None]
    exact = (w[:, None] * torch.relu(q_fp8.float() @ keys.T)).sum(0)
    return exact.numpy(), n_complete, ke


def _expand(leaf_start, leaf_len, scores, n):
    out = np.zeros(n, dtype=np.float64)
    for s, ln, sc in zip(leaf_start, leaf_len, scores):
        if ln > 0:
            out[s : s + ln] = sc
    return out


def _leaf_means_scores(exact, leaf_start, leaf_len):
    """Mean of exact token scores inside each leaf (visual proxy for mean-key estimate)."""
    vals = []
    for s, ln in zip(leaf_start, leaf_len):
        if ln <= 0:
            vals.append(0.0)
        else:
            vals.append(float(exact[s : s + ln].mean()))
    return np.asarray(vals, dtype=np.float64)


def panel_a(ax, summary: dict, full_arms_dir: Path):
    ex = summary.get("intro_example")
    if not ex:
        ax.text(0.5, 0.5, "no intro example", ha="center")
        ax.set_axis_off()
        return
    k = int(ex["k"])
    sample_dir = Path(ex["sample_dir"])
    exact, n_complete, ke = _exact_scores(sample_dir, ex["query"])
    snap = torch.load(
        full_arms_dir / f"{ex['layer']}__{ex['req']}.pt",
        map_location="cpu",
        weights_only=False,
    )
    fixed = snap["arms"][f"fixed_{k}"]
    adapt = snap["arms"][f"adaptive_{k}"]
    # Focus window around the densest top tokens in the first 8K for readability.
    window = min(4096, n_complete)
    # Prefer a window that covers many oracle top tokens if reference exists.
    top = np.argpartition(-exact[:n_complete], min(64, n_complete - 1))[
        : min(64, n_complete)
    ]
    center = int(np.median(top))
    lo = max(0, center - window // 2)
    hi = min(n_complete, lo + window)
    lo = max(0, hi - window)

    f_scores = _leaf_means_scores(exact, fixed["leaf_start"], fixed["leaf_len"])
    a_scores = _leaf_means_scores(exact, adapt["leaf_start"], adapt["leaf_len"])
    f_exp = _expand(fixed["leaf_start"], fixed["leaf_len"], f_scores, n_complete)
    a_exp = _expand(adapt["leaf_start"], adapt["leaf_len"], a_scores, n_complete)

    xs = np.arange(lo, hi)
    ax.plot(xs, exact[lo:hi], color="#222222", lw=0.8, label="exact token score")
    ax.plot(xs, f_exp[lo:hi], color="#C44E52", lw=1.2, label=f"fixed-{k} mean")
    ax.plot(xs, a_exp[lo:hi], color="#4C72B0", lw=1.2, label="adaptive mean")
    # Fixed boundaries.
    for s in fixed["leaf_start"]:
        if lo <= s < hi:
            ax.axvline(s, color="#C44E52", lw=0.4, alpha=0.35)
    for s in adapt["leaf_start"]:
        if lo <= s < hi:
            ax.axvline(s, color="#4C72B0", lw=0.4, alpha=0.25, ls="--")
    # Mark top-K tokens inside the window that fixed dilutes more.
    top2048 = set(np.argpartition(-exact[:n_complete], min(2047, n_complete - 1))[:2048].tolist())
    diluted = []
    for t in sorted(top2048):
        if not (lo <= t < hi):
            continue
        # Leaf containing t under fixed vs adaptive length.
        def leaf_len_of(starts, lens, t):
            for s, ln in zip(starts, lens):
                if s <= t < s + ln:
                    return ln
            return 0

        fl = leaf_len_of(fixed["leaf_start"], fixed["leaf_len"], t)
        al = leaf_len_of(adapt["leaf_start"], adapt["leaf_len"], t)
        if fl >= 2 * max(al, 1):
            diluted.append(t)
    if diluted:
        ax.scatter(
            diluted,
            exact[diluted],
            s=18,
            color="#55A868",
            zorder=5,
            label="Top-2048 diluted by fixed",
        )
    ax.set_xlim(lo, hi)
    ax.set_xlabel("token index")
    ax.set_ylabel("indexer score")
    ax.set_title(
        f"A  {ex['layer']}/{ex['req']}/{ex['query']}\n"
        f"gain={ex['gain']:+.3f} (high-hetero median exemplar)"
    )
    ax.legend(loc="upper right", fontsize=8, frameon=False)


def panel_b(ax, summary: dict):
    ks = [int(k) for k in summary.get("ks", [])]
    # Approximate #summaries as L/K using mean n from paired n if available;
    # fall back to nominal 1/K on a unit length axis labelled compression.
    xs_fixed, ys_fixed, yerr_lo, yerr_hi = [], [], [], []
    xs_adapt, ys_adapt, a_lo, a_hi = [], [], [], []
    for k in ks:
        f = summary["arms"].get(f"fixed_{k}")
        a = summary["arms"].get(f"adaptive_{k}")
        if not f or not a:
            continue
        # x = relative summary count = 1/K (higher = finer)
        x = 1.0 / k
        fr = f["recall_all"]
        ar = a["recall_all"]
        xs_fixed.append(x)
        ys_fixed.append(fr["mean"])
        yerr_lo.append(fr["mean"] - fr["ci95"][0])
        yerr_hi.append(fr["ci95"][1] - fr["mean"])
        xs_adapt.append(x)
        ys_adapt.append(ar["mean"])
        a_lo.append(ar["mean"] - ar["ci95"][0])
        a_hi.append(ar["ci95"][1] - ar["mean"])
    ax.errorbar(
        xs_fixed,
        ys_fixed,
        yerr=[yerr_lo, yerr_hi],
        fmt="o-",
        color="#C44E52",
        label="fixed-K",
        capsize=3,
    )
    ax.errorbar(
        xs_adapt,
        ys_adapt,
        yerr=[a_lo, a_hi],
        fmt="s-",
        color="#4C72B0",
        label="adaptive → L/K",
        capsize=3,
    )
    # Mark deployed HISA-64.
    if 64 in ks and f"fixed_64" in summary["arms"]:
        x = 1.0 / 64
        y = summary["arms"]["fixed_64"]["recall_all"]["mean"]
        ax.scatter([x], [y], s=80, marker="*", color="#DD8452", zorder=5, label="HISA-64")
    ax.set_xscale("log")
    ax.set_xlabel("relative summary count (1/K)")
    ax.set_ylabel("Top-2048 recall")
    ax.set_title("B  matched-budget recall vs granularity")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(True, which="both", ls=":", alpha=0.4)


def panel_c(ax, rows_path: Path, summary: dict, k: int = 128):
    gains = []
    hetero = []
    fixed_name, adapt_name = f"fixed_{k}", f"adaptive_{k}"
    with rows_path.open() as f:
        for line in f:
            row = json.loads(line)
            if fixed_name not in row["arms"] or adapt_name not in row["arms"]:
                continue
            h = row["arms"][fixed_name]["heterogeneity"]
            for a, b in zip(row["arms"][adapt_name]["queries"], row["arms"][fixed_name]["queries"]):
                gains.append(a["recall_all"] - b["recall_all"])
                hetero.append(h)
    if not gains:
        ax.text(0.5, 0.5, "no pairs", ha="center")
        ax.set_axis_off()
        return
    gains = np.asarray(gains)
    hetero = np.asarray(hetero)
    xs = np.sort(gains)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.plot(xs, ys, color="#4C72B0", lw=1.5, label="all queries")
    thr = summary.get("paired", {}).get(str(k), {}).get("high_hetero_threshold")
    if thr is not None:
        hi = gains[hetero >= thr]
        if hi.size:
            hx = np.sort(hi)
            hy = np.arange(1, len(hx) + 1) / len(hx)
            ax.plot(hx, hy, color="#55A868", lw=1.5, label="high-hetero quartile")
    ax.axvline(0.0, color="#666666", lw=0.8, ls="--")
    paired = summary.get("paired", {}).get(str(k), {})
    mean = paired.get("recall_all", {}).get("mean")
    ci = paired.get("recall_all", {}).get("ci95")
    frac = paired.get("recall_all", {}).get("frac_positive")
    title = f"C  adaptive − fixed-{k} recall"
    if mean is not None and ci is not None:
        title += f"\nmean {mean:+.4f}  CI95 [{ci[0]:+.4f}, {ci[1]:+.4f}]  P(>0)={frac:.2f}"
    ax.set_title(title)
    ax.set_xlabel("recall gain")
    ax.set_ylabel("ECDF")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.grid(True, ls=":", alpha=0.4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", type=Path, required=True, help="experiment output directory")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--k", type=int, default=128)
    args = ap.parse_args()
    summary = load_summary(args.exp / "summary.json")
    rows_path = args.exp / "per_query_rows.jsonl"
    full_arms = args.exp / "full_arms"
    out = args.out or (args.exp / "intro_fixed_vs_adaptive")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    panel_a(axes[0], summary, full_arms)
    panel_b(axes[1], summary)
    panel_c(axes[2], rows_path, summary, k=args.k)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=200, bbox_inches="tight")
    print(json.dumps({"pdf": str(out) + ".pdf", "png": str(out) + ".png"}, indent=2))


if __name__ == "__main__":
    main()
