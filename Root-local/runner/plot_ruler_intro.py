#!/usr/bin/env python3
"""RULER-only intro figure: HISA fixed-8 vs P-key adaptive split-8.

Matched budget: both arms keep ~L/8 summaries. Candidate Top-2048 recall
inside an 8,192-token budget. One prompt per RULER task × length.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASK_ORDER = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]

TASK_LABEL = {
    "niah_single_1": "NIAH-S1",
    "niah_single_2": "NIAH-S2",
    "niah_single_3": "NIAH-S3",
    "niah_multikey_1": "NIAH-MK1",
    "niah_multikey_2": "NIAH-MK2",
    "niah_multikey_3": "NIAH-MK3",
    "niah_multivalue": "NIAH-MV",
    "niah_multiquery": "NIAH-MQ",
    "vt": "VT",
    "cwe": "CWE",
    "fwe": "FWE",
    "qa_1": "QA-1",
    "qa_2": "QA-2",
}

FIXED_COLOR = "#F4B6A6"
ADAPT_COLOR = "#9EC5E8"
EDGE_COLOR = "#555555"


def load_rows(path: Path) -> list[dict]:
    out = []
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row["dataset"] != "ruler":
                continue
            out.append(
                {
                    "task": row["task"],
                    "length": row["length"],
                    "fixed": 100.0 * float(row["fixed_candidate_recall_all"]),
                    "adaptive": 100.0 * float(row["adaptive_candidate_recall_all"]),
                    "delta": 100.0 * float(row["delta_candidate_recall_all"]),
                }
            )
    return out


def ordered(rows: list[dict], length: str) -> list[dict]:
    by_task = {row["task"]: row for row in rows if row["length"] == length}
    return [by_task[task] for task in TASK_ORDER if task in by_task]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
            "font.size": 12,
            "font.weight": "bold",
            "axes.labelsize": 14,
            "axes.labelweight": "bold",
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def panel_recall(ax, rows: list[dict], title: str) -> None:
    labels = [TASK_LABEL[row["task"]] for row in rows]
    x = np.arange(len(rows))
    width = 0.38
    ax.bar(
        x - width / 2,
        [row["fixed"] for row in rows],
        width,
        color=FIXED_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.5,
        label="HISA fixed-8",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        [row["adaptive"] for row in rows],
        width,
        color=ADAPT_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.5,
        label="Adaptive split-8 (L/8)",
        zorder=3,
    )
    hi = max(max(row["fixed"] for row in rows), max(row["adaptive"] for row in rows)) + 3
    ax.set_ylim(50, hi)
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=40,
        ha="right",
        rotation_mode="anchor",
        fontweight="bold",
    )
    ax.set_ylabel("Candidate Top-2048 recall (%)", fontweight="bold")
    ax.set_title("RULER 128K  ·  HISA fixed-8 vs adaptive split-8", loc="left", pad=8)
    ax.tick_params(axis="both", which="major", labelsize=12, width=1.0)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(axis="y", ls=":", alpha=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_delta(ax, rows32: list[dict], rows128: list[dict]) -> None:
    labels = [TASK_LABEL[row["task"]] for row in rows128]
    x = np.arange(len(labels))
    width = 0.38
    d32 = [row["delta"] for row in rows32]
    d128 = [row["delta"] for row in rows128]
    ax.bar(x - width / 2, d32, width, color="#8DA0CB", label="32K", zorder=3)
    ax.bar(x + width / 2, d128, width, color=ADAPT_COLOR, label="128K", zorder=3)
    ax.axhline(0.0, color="#444444", lw=0.8, zorder=2)
    mean32 = float(np.mean(d32))
    mean128 = float(np.mean(d128))
    ax.axhline(mean32, color="#8DA0CB", lw=1.0, ls="--", alpha=0.9)
    ax.axhline(mean128, color=ADAPT_COLOR, lw=1.0, ls="--", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Adaptive − fixed (pp)")
    ax.set_title("B  Paired gain on every RULER task", loc="left", pad=8)
    ax.grid(axis="y", ls=":", alpha=0.45, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.text(
        0.02,
        0.98,
        f"mean  32K {mean32:+.2f} pp   128K {mean128:+.2f} pp",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.5,
        color="#333333",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path(
            "/data/jz/speed_957_0923/stats/"
            "pkey_split8_vs_fixed8_ruler_20260923/per_sample_summary.csv"
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/data/jz/speed_957_0923/stats/"
            "pkey_split8_vs_fixed8_ruler_20260923/intro_ruler_fixed8_vs_adaptive"
        ),
    )
    args = ap.parse_args()

    style()
    rows = load_rows(args.csv)
    rows32 = ordered(rows, "32k")
    rows128 = ordered(rows, "128k")

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    panel_recall(ax, rows128, "RULER 128K  ·  HISA fixed-8 vs adaptive split-8")
    ax.set_title("RULER 128K  ·  HISA fixed-8 vs adaptive split-8", loc="left", pad=8)
    ax.legend(
        frameon=False,
        loc="upper left",
        ncol=2,
        fontsize=11,
        columnspacing=1.6,
        handletextpad=0.6,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(args.out) + ".png", dpi=300, bbox_inches="tight")
    print(
        json.dumps(
            {
                "pdf": str(args.out) + ".pdf",
                "png": str(args.out) + ".png",
                "ruler_32k_mean_pp": float(np.mean([r["delta"] for r in rows32])),
                "ruler_128k_mean_pp": float(np.mean([r["delta"] for r in rows128])),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
