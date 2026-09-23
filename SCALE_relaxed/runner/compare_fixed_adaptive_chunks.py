#!/usr/bin/env python3
"""Fixed vs Adaptive chunk experiment on real indexer dumps.

Fair matched-budget protocol (same final summary count ≈ L/K and the same
8192-token candidate budget):

  * fixed-K: uniform leaves of length K (optional offsets for sensitivity)
  * adaptive-K: P-key λ-split to L/S leaves then target-count Ward merge to
                ∼ L/K leaves (S and the merge-round cap are explicit CLI args)

Both paths use production FP8 leaf means (requantised totals), DSA coarse
scoring ``sum_h w_h relu(q_h · mean)``, weighted token-budgeted selection,
and exact FP8 token rerank. Recomputed dense Top-2048 is the default retrieval
oracle. The primary metric is direct candidate-set coverage
``|Candidates_8192 ∩ DenseTop_2048| / 2048``; exact-rerank recall is retained
as an equivalence check. Token-expanded and chunk-estimate relative MSE are
reported as explanatory metrics.

Outputs per-query JSONL + a bootstrap summary suitable for the intro figure.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "python"))

ENV = {
    "SGLANG_NSA_ADAPTIVE_HISA_MODE": "adaptive_decode",
    "SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC": "key_sse",
    "SGLANG_NSA_ADAPTIVE_HISA_SPLIT_BACKEND": "gpu",
    "SGLANG_NSA_ADAPTIVE_HISA_BUILD_SUMMARIES": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY": "sync_nonoverlap",
    "SGLANG_NSA_ADAPTIVE_HISA_SUMMARY_COMPRESSION": "32",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_ROUNDS": "4",
    "SGLANG_NSA_ADAPTIVE_HISA_MAX_MERGE_LEN": "0",
    "SGLANG_NSA_ADAPTIVE_HISA_SINK_TOKENS": "64",
    "SGLANG_NSA_ADAPTIVE_HISA_TAIL_TOKENS": "256",
    "SGLANG_NSA_ADAPTIVE_HISA_CANDIDATE_TOKENS": "8192",
    "SGLANG_NSA_ADAPTIVE_HISA_DECODE_CHUNK": "64",
    "SGLANG_NSA_ADAPTIVE_HISA_RAW_FP8_BUILDER": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_LAMBDA_MAX_ROUNDS": "6",
}
os.environ.update(ENV)

import torch  # noqa: E402


@dataclass
class LeafMap:
    start: torch.Tensor  # int32 [M]
    length: torch.Tensor  # int32 [M]
    kind: str
    chunk: int
    offset: int = 0

    @property
    def num_leaves(self) -> int:
        return int(self.start.numel())


def discover_samples(root: Path) -> list[Path]:
    return sorted(
        p.parent
        for p in root.glob("layer_*/req_*/manifest.json")
        if p.is_file()
    )


def load_sample(sample_dir: Path, device: torch.device):
    manifest = json.loads((sample_dir / "manifest.json").read_text())
    key_parts = []
    scale_parts = []
    for name in manifest["key_shards"]:
        blob = torch.load(sample_dir / name, map_location="cpu", weights_only=False)
        key_parts.append(blob["k_fp8"])
        scale = blob["k_scale"]
        scale_parts.append(scale.reshape(-1))
    # The first shard is the saved prefill prefix. Production P-key partitions
    # complete 256-token roots only; the short remainder stays in the raw tail.
    # Use that same complete-root prefix for fixed-K so both arms summarize
    # exactly the same token range and have the same L/K leaf count.
    n_saved_prefix = int(key_parts[0].shape[0])
    n_complete = (n_saved_prefix // 256) * 256
    k_fp8 = torch.cat(key_parts, dim=0).to(device)
    if k_fp8.dtype == torch.uint8:
        k_fp8 = k_fp8.view(torch.float8_e4m3fn)
    k_scale = torch.cat(scale_parts, dim=0).to(device=device, dtype=torch.float32)
    queries = []
    for name in manifest["query_shards"]:
        q = torch.load(sample_dir / name, map_location="cpu", weights_only=False)
        queries.append(
            {
                "name": name,
                "q_fp8": q["q_fp8"][0].to(device),
                "w": q["w"][0].to(device=device, dtype=torch.float32),
                "ke": int(q["ke"][0]),
                "reference_topk": q.get("reference_topk"),
            }
        )
        if queries[-1]["q_fp8"].dtype == torch.uint8:
            queries[-1]["q_fp8"] = queries[-1]["q_fp8"].view(torch.float8_e4m3fn)
        if queries[-1]["reference_topk"] is not None:
            queries[-1]["reference_topk"] = (
                queries[-1]["reference_topk"][0].to(device=device, dtype=torch.long)
            )
    return {
        "dir": sample_dir,
        "layer": sample_dir.parent.name,
        "req": sample_dir.name,
        "n_complete": n_complete,
        "n_saved_prefix": n_saved_prefix,
        "k_fp8": k_fp8,
        "k_scale": k_scale,
        "queries": queries,
        "manifest": manifest,
    }


def fixed_leaves(n_complete: int, chunk: int, offset: int, device) -> LeafMap:
    if offset:
        starts = [0] + list(range(offset, n_complete, chunk))
        # Deduplicate / keep increasing.
        starts = sorted(set(s for s in starts if 0 <= s < n_complete))
    else:
        starts = list(range(0, n_complete, chunk))
    lengths = []
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else n_complete
        lengths.append(end - s)
    return LeafMap(
        start=torch.tensor(starts, dtype=torch.int32, device=device),
        length=torch.tensor(lengths, dtype=torch.int32, device=device),
        kind="fixed",
        chunk=chunk,
        offset=offset,
    )


def adaptive_leaves(
    k_fp8,
    k_scale,
    n_complete: int,
    divisor: int,
    split_divisor: int,
    merge_rounds: int,
    device,
    merge_policy: str = "sync_nonoverlap",
) -> LeafMap:
    from dataclasses import replace

    from sglang.srt.layers.attention.nsa.adaptive_hisa.config import get_config, reset_config_cache
    from sglang.srt.layers.attention.nsa.adaptive_hisa.pkey_fp8_builder import (
        build_partition_from_fp8,
    )

    os.environ["SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR"] = str(divisor)
    os.environ["SGLANG_NSA_ADAPTIVE_HISA_SUMMARY_COMPRESSION"] = str(split_divisor)
    os.environ["SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY"] = merge_policy
    reset_config_cache()
    cfg = replace(
        get_config(),
        summary_compression=int(split_divisor),
        merge_target_divisor=int(divisor),
        merge_target_rounds=int(merge_rounds),
        merge_policy=merge_policy,
    ).validate()
    part = build_partition_from_fp8(
        k_fp8[:n_complete], k_scale[:n_complete], n_complete, cfg
    )
    n = int(part.num_leaves.item())
    return LeafMap(
        start=part.leaf_start[:n].contiguous(),
        length=part.leaf_len[:n].contiguous(),
        kind="adaptive",
        chunk=int(divisor),
        offset=0,
    ), part


def build_summaries(k_fp8, k_scale, leaves: LeafMap, n_complete: int):
    from sglang.srt.layers.attention.nsa.adaptive_hisa.pkey_fp8_builder import (
        leaf_totals_from_fp8,
        summaries_from_totals,
    )

    totals = leaf_totals_from_fp8(
        k_fp8, k_scale, leaves.start, leaves.length, n_complete
    )
    fp8, scale = summaries_from_totals(totals, leaves.length, "ue8m0")
    return fp8, scale.reshape(-1), totals


def key_reconstruction_sse(k_fp8, k_scale, leaves: LeafMap, n_complete: int) -> float:
    """Mean per-token SSE of dequantised keys vs their leaf means."""
    keys = k_fp8[:n_complete].float() * k_scale[:n_complete, None]
    sse = 0.0
    count = 0
    starts = leaves.start.tolist()
    lengths = leaves.length.tolist()
    for s, ln in zip(starts, lengths):
        if ln <= 0:
            continue
        block = keys[s : s + ln]
        mean = block.mean(dim=0, keepdim=True)
        sse += float((block - mean).pow(2).sum().item())
        count += ln
    return sse / max(count, 1)


def block_heterogeneity(k_fp8, k_scale, leaves: LeafMap, n_complete: int) -> float:
    """Length-weighted mean of per-block key SSE (same units as reconstruction)."""
    keys = k_fp8[:n_complete].float() * k_scale[:n_complete, None]
    total = 0.0
    weight = 0
    for s, ln in zip(leaves.start.tolist(), leaves.length.tolist()):
        if ln <= 1:
            continue
        block = keys[s : s + ln]
        mean = block.mean(dim=0, keepdim=True)
        total += float((block - mean).pow(2).sum().item())
        weight += ln
    return total / max(weight, 1)


def exact_token_scores(q_fp8, w, k_fp8, k_scale, ke: int) -> torch.Tensor:
    """DSA indexer logits for one query against tokens ``[0, ke)``."""
    q = q_fp8.float()  # [H, D]
    keys = k_fp8[:ke].float() * k_scale[:ke, None]
    return (w[:, None] * torch.relu(q @ keys.T)).sum(0)


def coarse_leaf_scores(q_fp8, w, mean_fp8, mean_scale, leaves: LeafMap) -> torch.Tensor:
    means = mean_fp8.float() * mean_scale[:, None]
    return (w[:, None] * torch.relu(q_fp8.float() @ means.T)).sum(0)


def expand_scores(leaf_scores: torch.Tensor, leaves: LeafMap, n: int) -> torch.Tensor:
    out = torch.zeros(n, dtype=torch.float32, device=leaf_scores.device)
    for i, (s, ln) in enumerate(zip(leaves.start.tolist(), leaves.length.tolist())):
        if ln > 0:
            out[s : s + ln] = leaf_scores[i]
    return out


def select_and_rerank(
    q_fp8,
    w,
    k_fp8,
    k_scale,
    leaves: LeafMap,
    mean_fp8,
    mean_scale,
    ke: int,
    budget: int,
    sink: int,
    tail: int,
    topk: int,
):
    from sglang.srt.layers.attention.nsa.adaptive_hisa.decode_select import (
        weighted_select_candidates,
    )

    device = q_fp8.device
    coarse = coarse_leaf_scores(q_fp8, w, mean_fp8, mean_scale, leaves).contiguous()
    # Pad to capacity buffers expected by the selector.
    capacity = int(leaves.start.numel())
    lengths = leaves.length
    starts = leaves.start
    selected_prefix = torch.zeros(capacity, dtype=torch.int32, device=device)
    num_leaves = torch.tensor([capacity], dtype=torch.int32, device=device)
    seq = torch.tensor([ke], dtype=torch.int32, device=device)
    candidates, count = weighted_select_candidates(
        coarse,
        lengths,
        starts,
        num_leaves,
        seq,
        selected_prefix,
        budget,
        sink,
        tail,
    )
    n_cand = int(count.item())
    ids = candidates[0, :n_cand].to(torch.long)
    # Exact rerank on selected tokens.
    keys = k_fp8[ids].float() * k_scale[ids, None]
    fine = (w[:, None] * torch.relu(q_fp8.float() @ keys.T)).sum(0)
    order = torch.argsort(ids, stable=True)
    order = order[torch.argsort(-fine[order], stable=True)]
    picked = ids[order[:topk]]
    return {
        "candidates": ids,
        "topk": picked,
        "coarse": coarse,
        "n_cand": n_cand,
    }


def recall_at_k(ref: torch.Tensor, pred: torch.Tensor) -> float:
    ref_set = set(int(x) for x in ref.tolist() if int(x) >= 0)
    if not ref_set:
        return 1.0
    hit = sum(1 for x in pred.tolist() if int(x) in ref_set)
    return hit / min(len(ref_set), int(pred.numel()))


def summary_topk_recalls(
    ref: torch.Tensor, approx: torch.Tensor, topk: int
) -> tuple[float, float]:
    """Recall from token scores obtained by expanding each chunk-summary score.

    ``stable`` breaks equal-score ties by logical token index. ``expected``
    removes that arbitrary choice by taking the expected overlap under uniform
    tie-breaking at the Top-K boundary.
    """
    k = min(topk, int(approx.numel()))
    if k <= 0:
        return 1.0, 1.0
    order = torch.argsort(approx, descending=True, stable=True)
    stable = recall_at_k(ref, order[:k])

    threshold = approx[order[k - 1]]
    above = approx > threshold
    tied = approx == threshold
    n_above = int(above.sum().item())
    n_tied = int(tied.sum().item())
    remaining = max(0, k - n_above)
    ref_mask = torch.zeros_like(approx, dtype=torch.bool)
    ref_mask[ref.to(torch.long)] = True
    hits_above = int((above & ref_mask).sum().item())
    hits_tied = int((tied & ref_mask).sum().item())
    expected_hits = hits_above
    if n_tied:
        expected_hits += remaining * hits_tied / n_tied
    expected = expected_hits / max(int(ref.numel()), 1)
    return stable, expected


def dense_topk(q_fp8, w, k_fp8, k_scale, ke: int, topk: int) -> torch.Tensor:
    scores = exact_token_scores(q_fp8, w, k_fp8, k_scale, ke)
    return torch.topk(scores, min(topk, ke)).indices


def score_errors(
    exact: torch.Tensor,
    leaf_scores: torch.Tensor,
    leaves: LeafMap,
    n_complete: int,
    approx: torch.Tensor | None = None,
) -> tuple[float, float]:
    """Return token-expanded RelMSE and length-weighted chunk-estimate RelMSE.

    The first metric includes the within-chunk dilution that affects retrieval.
    The second isolates summary-score bias relative to each chunk's mean exact
    token score; length weighting keeps short adaptive chunks from dominating.
    """
    if approx is None:
        approx = expand_scores(leaf_scores, leaves, n_complete)
    e = exact[:n_complete]
    a = approx[:n_complete]
    denom = float(e.pow(2).sum().clamp_min(1e-12).item())
    token_relmse = float((e - a).pow(2).sum().item() / denom)

    starts = leaves.start.to(torch.long)
    lengths = leaves.length.to(torch.long).clamp_min(1)
    ends = (starts + lengths).clamp_max(n_complete)
    prefix = torch.cat([e.new_zeros(1), e.cumsum(0)])
    exact_chunk_mean = (prefix[ends] - prefix[starts]) / lengths
    weights = lengths.to(torch.float32)
    chunk_denom = (weights * exact_chunk_mean.pow(2)).sum().clamp_min(1e-12)
    chunk_relmse = float(
        (weights * (leaf_scores - exact_chunk_mean).pow(2)).sum().item()
        / chunk_denom.item()
    )
    return token_relmse, chunk_relmse


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0) -> dict:
    if not values:
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = random.Random(seed)
    m = len(values)
    means = []
    for _ in range(n):
        sample = [values[rng.randrange(m)] for _ in range(m)]
        means.append(sum(sample) / m)
    means.sort()
    return {
        "mean": sum(values) / m,
        "ci95": [means[int(0.025 * n)], means[int(0.975 * n)]],
        "n": m,
        "std": (sum((x - sum(values) / m) ** 2 for x in values) / max(m - 1, 1)) ** 0.5,
    }


def paired_diff_ci(base: list[float], cand: list[float], n: int = 2000, seed: int = 1) -> dict:
    assert len(base) == len(cand)
    diffs = [c - b for b, c in zip(base, cand)]
    out = bootstrap_ci(diffs, n=n, seed=seed)
    out["frac_positive"] = sum(1 for d in diffs if d > 0) / max(len(diffs), 1)
    return out


def evaluate_arm(
    sample,
    leaves: LeafMap,
    mean_fp8,
    mean_scale,
    *,
    budget: int,
    sink: int,
    tail: int,
    topk: int,
    use_dump_ref: bool,
):
    n_complete = sample["n_complete"]
    k_fp8, k_scale = sample["k_fp8"], sample["k_scale"]
    sse = key_reconstruction_sse(k_fp8, k_scale, leaves, n_complete)
    hetero = block_heterogeneity(k_fp8, k_scale, leaves, n_complete)
    rows = []
    for qi, q in enumerate(sample["queries"]):
        ke = min(int(q["ke"]), int(k_fp8.shape[0]))
        if ke <= topk:
            continue
        exact = exact_token_scores(q["q_fp8"], q["w"], k_fp8, k_scale, ke)
        if use_dump_ref and q["reference_topk"] is not None:
            ref = q["reference_topk"]
        else:
            ref = torch.topk(exact, min(topk, ke)).indices
        sel = select_and_rerank(
            q["q_fp8"],
            q["w"],
            k_fp8,
            k_scale,
            leaves,
            mean_fp8,
            mean_scale,
            ke,
            budget,
            sink,
            tail,
            topk,
        )
        # Prefix-restricted recall: only oracle tokens inside the partitioned prefix.
        ref_prefix = ref[(ref >= 0) & (ref < n_complete)]
        cand_prefix = sel["candidates"][
            (sel["candidates"] >= 0) & (sel["candidates"] < n_complete)
        ]
        candidate_recall = recall_at_k(ref, sel["candidates"])
        rerank_recall = recall_at_k(ref, sel["topk"])
        candidate_recall_prefix = (
            recall_at_k(ref_prefix, cand_prefix) if ref_prefix.numel() else 1.0
        )
        approx = expand_scores(sel["coarse"], leaves, n_complete)
        summary_ref = torch.argsort(
            exact[:n_complete], descending=True, stable=True
        )[: min(topk, n_complete)]
        summary_topk_stable, summary_topk_expected = summary_topk_recalls(
            summary_ref, approx, topk
        )
        token_relmse, chunk_relmse = score_errors(
            exact, sel["coarse"], leaves, n_complete, approx
        )
        rows.append(
            {
                "query_idx": qi,
                "query": q["name"],
                "ke": ke,
                # ``recall_*`` aliases keep the plotting script compatible;
                # candidate coverage is the primary metric for this experiment.
                "recall_all": candidate_recall,
                "recall_prefix": candidate_recall_prefix,
                "candidate_recall_all": candidate_recall,
                "candidate_recall_prefix": candidate_recall_prefix,
                "rerank_recall_all": rerank_recall,
                "candidate_rerank_gap": candidate_recall - rerank_recall,
                "summary_topk_recall_stable": summary_topk_stable,
                "summary_topk_recall_expected": summary_topk_expected,
                "n_cand": sel["n_cand"],
                "score_err": token_relmse,
                "token_score_relmse": token_relmse,
                "chunk_score_relmse": chunk_relmse,
                "topk": sel["topk"].detach().cpu().tolist(),
                "ref": ref.detach().cpu().tolist(),
            }
        )
    return {
        "sse": sse,
        "heterogeneity": hetero,
        "num_leaves": leaves.num_leaves,
        "queries": rows,
    }


def run_sample(
    sample,
    ks: list[int],
    budget: int,
    sink: int,
    tail: int,
    topk: int,
    offsets: bool,
    split_divisor: int,
    merge_rounds: int,
    use_dump_ref: bool,
    merge_target_divisor: int | None = None,
    merge_policy: str = "sync_nonoverlap",
):
    device = sample["k_fp8"].device
    n_complete = sample["n_complete"]
    arms = {}
    # Fixed arms (+ optional offsets for the primary K=128 / K=64).
    for k in ks:
        offs = [0]
        if offsets and k in (64, 128):
            offs = sorted(set([0, k // 4, k // 2, (3 * k) // 4]))
        for off in offs:
            name = f"fixed_{k}" if off == 0 else f"fixed_{k}_off{off}"
            leaves = fixed_leaves(n_complete, k, off, device)
            mean_fp8, mean_scale, _ = build_summaries(
                sample["k_fp8"], sample["k_scale"], leaves, n_complete
            )
            arms[name] = {
                "leaves": leaves,
                "mean_fp8": mean_fp8,
                "mean_scale": mean_scale,
                "meta": {
                    "kind": "fixed",
                    "chunk": k,
                    "offset": off,
                    "target_leaves": math.ceil(n_complete / k),
                },
            }
    # Adaptive arms. Default matches fixed-K (L/K). 0 = threshold merge;
    # merge_policy=off keeps the split-L/S leaves and does not align counts.
    for k in ks:
        merge_div = k if merge_target_divisor is None else int(merge_target_divisor)
        if merge_policy == "off":
            merge_div = 0
        leaves, part = adaptive_leaves(
            sample["k_fp8"],
            sample["k_scale"],
            n_complete,
            merge_div,
            split_divisor,
            merge_rounds,
            device,
            merge_policy=merge_policy,
        )
        # Prefer merge totals when present.
        from sglang.srt.layers.attention.nsa.adaptive_hisa.pkey_fp8_builder import (
            summaries_from_totals,
            leaf_totals_from_fp8,
        )

        totals = part.meta.pop("_merge_totals", None)
        if totals is None:
            totals = leaf_totals_from_fp8(
                sample["k_fp8"],
                sample["k_scale"],
                leaves.start,
                leaves.length,
                n_complete,
            )
        else:
            totals = totals[: leaves.num_leaves].contiguous()
        mean_fp8, mean_scale = summaries_from_totals(totals, leaves.length, "ue8m0")
        mean_scale = mean_scale.reshape(-1)
        if merge_policy == "off":
            adapt_name = f"adaptive_split{split_divisor}"
        elif merge_target_divisor == 0:
            adapt_name = f"adaptive_split{split_divisor}_threshold"
        elif merge_target_divisor not in (None, k):
            adapt_name = f"adaptive_split{split_divisor}_merge{merge_div}"
        else:
            adapt_name = f"adaptive_{k}"
        arms[adapt_name] = {
            "leaves": leaves,
            "mean_fp8": mean_fp8,
            "mean_scale": mean_scale,
            "meta": {
                "kind": "adaptive",
                "chunk": k,
                "split_divisor": split_divisor,
                "merge_target_divisor": merge_div,
                "merge_policy": merge_policy,
                "merge_target_rounds": merge_rounds,
                "offset": 0,
                "target_leaves": (
                    0 if merge_policy == "off" or merge_div == 0 else n_complete // merge_div
                ),
                "split_target_leaves": int(part.meta.get("premerge_M", 0)),
                "status_ok": bool(part.status_ok.item()),
                "dp_leaves": int(part.dp_leaves.item()),
                "repairs": int(part.repairs.item()),
                "rounds_used": int(part.meta.get("lambda_rounds_used", torch.tensor(0)).item())
                if torch.is_tensor(part.meta.get("lambda_rounds_used", 0))
                else int(part.meta.get("lambda_rounds_used", 0)),
            },
        }

    out_arms = {}
    for name, arm in arms.items():
        ev = evaluate_arm(
            sample,
            arm["leaves"],
            arm["mean_fp8"],
            arm["mean_scale"],
            budget=budget,
            sink=sink,
            tail=tail,
            topk=topk,
            use_dump_ref=use_dump_ref,
        )
        out_arms[name] = {
            **arm["meta"],
            "num_leaves": ev["num_leaves"],
            "sse": ev["sse"],
            "heterogeneity": ev["heterogeneity"],
            "queries": ev["queries"],
            "leaf_start": arm["leaves"].start.detach().cpu().tolist(),
            "leaf_len": arm["leaves"].length.detach().cpu().tolist(),
        }
    return out_arms


def aggregate(
    rows: list[dict], ks: list[int], primary_metric: str = "candidate_recall_all"
) -> dict:
    """Bootstrap summary across all (sample, query) pairs."""
    summary = {"ks": ks, "arms": {}, "paired": {}}
    # Collect per-arm per-query metrics.
    per_arm = {}
    for row in rows:
        for arm, data in row["arms"].items():
            bucket = per_arm.setdefault(
                arm,
                {
                    "recall_all": [],
                    "recall_prefix": [],
                    "candidate_recall_all": [],
                    "candidate_recall_prefix": [],
                    "rerank_recall_all": [],
                    "candidate_rerank_gap": [],
                    "summary_topk_recall_stable": [],
                    "summary_topk_recall_expected": [],
                    "sse": [],
                    "heterogeneity": [],
                    "score_err": [],
                    "token_score_relmse": [],
                    "chunk_score_relmse": [],
                    "num_leaves": [],
                },
            )
            bucket["sse"].append(data["sse"])
            bucket["heterogeneity"].append(data["heterogeneity"])
            bucket["num_leaves"].append(data["num_leaves"])
            for q in data["queries"]:
                bucket["recall_all"].append(q["recall_all"])
                bucket["recall_prefix"].append(q["recall_prefix"])
                bucket["candidate_recall_all"].append(q["candidate_recall_all"])
                bucket["candidate_recall_prefix"].append(q["candidate_recall_prefix"])
                bucket["rerank_recall_all"].append(q["rerank_recall_all"])
                bucket["candidate_rerank_gap"].append(q["candidate_rerank_gap"])
                bucket["summary_topk_recall_stable"].append(
                    q["summary_topk_recall_stable"]
                )
                bucket["summary_topk_recall_expected"].append(
                    q["summary_topk_recall_expected"]
                )
                bucket["score_err"].append(q["score_err"])
                bucket["token_score_relmse"].append(q["token_score_relmse"])
                bucket["chunk_score_relmse"].append(q["chunk_score_relmse"])
    for arm, bucket in per_arm.items():
        summary["arms"][arm] = {
            key: bootstrap_ci(vals, seed=hash(arm + key) % 10_000)
            for key, vals in bucket.items()
        }
        summary["arms"][arm]["kind"] = (
            "fixed" if arm.startswith("fixed_") else "adaptive"
        )
    # Paired adaptive - fixed. Prefer matched adaptive_{k}; otherwise the
    # unmatched adaptive arm against fixed of the first K.
    adaptive_names = [name for name in per_arm if name.startswith("adaptive_")]
    for k in ks:
        fixed_name = f"fixed_{k}"
        adapt_name = f"adaptive_{k}"
        if adapt_name not in per_arm:
            adapt_name = next(
                (name for name in adaptive_names),
                "",
            )
        if fixed_name not in per_arm or adapt_name not in per_arm:
            continue
        # Align by walking rows in order (same query order across arms).
        f_all, a_all, f_pre, a_pre, f_token_mse, a_token_mse = [], [], [], [], [], []
        f_chunk_mse, a_chunk_mse, hetero = [], [], []
        f_summary_stable, a_summary_stable = [], []
        f_summary_expected, a_summary_expected = [], []
        for row in rows:
            if fixed_name not in row["arms"] or adapt_name not in row["arms"]:
                continue
            fq = row["arms"][fixed_name]["queries"]
            aq = row["arms"][adapt_name]["queries"]
            if len(fq) != len(aq):
                continue
            hetero.extend([row["arms"][fixed_name]["heterogeneity"]] * len(fq))
            for a, b in zip(aq, fq):
                a_all.append(a["recall_all"])
                f_all.append(b["recall_all"])
                a_pre.append(a["recall_prefix"])
                f_pre.append(b["recall_prefix"])
                a_summary_stable.append(a["summary_topk_recall_stable"])
                f_summary_stable.append(b["summary_topk_recall_stable"])
                a_summary_expected.append(a["summary_topk_recall_expected"])
                f_summary_expected.append(b["summary_topk_recall_expected"])
                a_token_mse.append(a["token_score_relmse"])
                f_token_mse.append(b["token_score_relmse"])
                a_chunk_mse.append(a["chunk_score_relmse"])
                f_chunk_mse.append(b["chunk_score_relmse"])
        summary["paired"][str(k)] = {
            "primary_metric": primary_metric,
            "candidate_recall_all": paired_diff_ci(f_all, a_all, seed=10 + k),
            "recall_all": paired_diff_ci(f_all, a_all, seed=10 + k),
            "recall_prefix": paired_diff_ci(f_pre, a_pre, seed=20 + k),
            "summary_topk_recall_stable": paired_diff_ci(
                f_summary_stable, a_summary_stable, seed=70 + k
            ),
            "summary_topk_recall_expected": paired_diff_ci(
                f_summary_expected, a_summary_expected, seed=80 + k
            ),
            "token_score_relmse": paired_diff_ci(
                f_token_mse, a_token_mse, seed=50 + k
            ),
            "chunk_score_relmse": paired_diff_ci(
                f_chunk_mse, a_chunk_mse, seed=60 + k
            ),
            "heterogeneity_fixed": bootstrap_ci(hetero, seed=30 + k),
            "n_pairs": len(f_all),
        }
        # Gain vs heterogeneity correlation (Pearson on host).
        if len(f_all) >= 3:
            gains = [a - b for a, b in zip(a_all, f_all)]
            mh = sum(hetero) / len(hetero)
            mg = sum(gains) / len(gains)
            num = sum((h - mh) * (g - mg) for h, g in zip(hetero, gains))
            den_h = math.sqrt(sum((h - mh) ** 2 for h in hetero))
            den_g = math.sqrt(sum((g - mg) ** 2 for g in gains))
            corr = num / (den_h * den_g) if den_h > 0 and den_g > 0 else 0.0
            summary["paired"][str(k)]["corr_gain_vs_hetero"] = corr
            # High-heterogeneity quartile paired gain.
            thr = sorted(hetero)[max(0, int(0.75 * len(hetero)) - 1)]
            hi_f = [f for f, h in zip(f_all, hetero) if h >= thr]
            hi_a = [a for a, h in zip(a_all, hetero) if h >= thr]
            summary["paired"][str(k)]["high_hetero_recall_all"] = paired_diff_ci(
                hi_f, hi_a, seed=40 + k
            )
            summary["paired"][str(k)]["high_hetero_threshold"] = thr
    return summary


def pick_intro_example(rows: list[dict], k: int = 128) -> dict | None:
    """High-hetero quartile, gain closest to that group's median."""
    fixed_name = f"fixed_{k}"
    candidates = []
    for row in rows:
        adapt_name = next(
            (name for name in row.get("arms", {}) if name.startswith("adaptive_")),
            "",
        )
        if fixed_name not in row["arms"] or not adapt_name:
            continue
        hetero = row["arms"][fixed_name]["heterogeneity"]
        fq = row["arms"][fixed_name]["queries"]
        aq = row["arms"][adapt_name]["queries"]
        for a, b in zip(aq, fq):
            candidates.append(
                {
                    "layer": row["layer"],
                    "req": row["req"],
                    "query": a["query"],
                    "query_idx": a["query_idx"],
                    "heterogeneity": hetero,
                    "gain": a["recall_all"] - b["recall_all"],
                    "recall_fixed": b["recall_all"],
                    "recall_adaptive": a["recall_all"],
                    "sample_dir": row["sample_dir"],
                }
            )
    if not candidates:
        return None
    thr = sorted(c["heterogeneity"] for c in candidates)[
        max(0, int(0.75 * len(candidates)) - 1)
    ]
    hi = [c for c in candidates if c["heterogeneity"] >= thr]
    if not hi:
        hi = candidates
    gains = sorted(c["gain"] for c in hi)
    median = gains[len(gains) // 2]
    best = min(hi, key=lambda c: (abs(c["gain"] - median), -c["heterogeneity"]))
    best["high_hetero_threshold"] = thr
    best["group_median_gain"] = median
    best["k"] = k
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dump",
        type=Path,
        default=Path("/data/jz/adaptive_hisa_dump_sweep_20260918/rank_00"),
    )
    ap.add_argument("--ks", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--layers", nargs="*", default=None, help="e.g. layer_003 layer_030")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--budget", type=int, default=8192)
    ap.add_argument("--sink", type=int, default=64)
    ap.add_argument("--tail", type=int, default=256)
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument(
        "--primary-metric",
        choices=["candidate_recall_all", "summary_topk_recall_expected"],
        default="candidate_recall_all",
    )
    ap.add_argument(
        "--split-divisor",
        type=int,
        default=32,
        help="P-key split leaf count is L/S (use 8 for the production split-8 arm)",
    )
    ap.add_argument(
        "--merge-rounds",
        type=int,
        default=4,
        help="target-count merge round cap (use 8 for split-8 -> merge-64)",
    )
    ap.add_argument(
        "--merge-target-divisor",
        type=int,
        default=-1,
        help="Adaptive merge target L/D. -1 = match --ks (aligned). 0 = threshold merge, no count match.",
    )
    ap.add_argument(
        "--merge-policy",
        choices=["sync_nonoverlap", "off"],
        default="sync_nonoverlap",
        help="off = keep split leaves, do not merge down to HISA's chunk count",
    )
    ap.add_argument("--only-dataset", default="", help="e.g. ruler")
    ap.add_argument(
        "--use-dump-ref",
        action="store_true",
        help="use saved reference_topk instead of recomputing dense Top-2048",
    )
    ap.add_argument("--offsets", action="store_true", help="also sweep fixed offsets")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("/tmp/fixed_adaptive_chunks"))
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    samples = discover_samples(args.dump)
    if args.layers:
        allow = set(args.layers)
        samples = [s for s in samples if s.parent.name in allow]
    if args.only_dataset:
        keep = []
        for path in samples:
            meta = json.loads((path / "manifest.json").read_text()).get("sample_meta", {})
            if meta.get("dataset") == args.only_dataset:
                keep.append(path)
        samples = keep
    if args.max_samples:
        samples = samples[: args.max_samples]
    merge_target = None if args.merge_target_divisor < 0 else args.merge_target_divisor
    args.out.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out / "per_query_rows.jsonl"
    rows = []
    with jsonl_path.open("w") as stream:
        for i, path in enumerate(samples, 1):
            sample = load_sample(path, device)
            print(
                f"[{i}/{len(samples)}] {sample['layer']}/{sample['req']} "
                f"N={sample['n_complete']} Q={len(sample['queries'])}",
                flush=True,
            )
            arms = run_sample(
                sample,
                args.ks,
                args.budget,
                args.sink,
                args.tail,
                args.topk,
                offsets=args.offsets,
                split_divisor=args.split_divisor,
                merge_rounds=args.merge_rounds,
                use_dump_ref=args.use_dump_ref,
                merge_target_divisor=merge_target,
                merge_policy=args.merge_policy,
            )
            row = {
                "sample_dir": str(path),
                "layer": sample["layer"],
                "req": sample["req"],
                "n_complete": sample["n_complete"],
                "n_saved_prefix": sample["n_saved_prefix"],
                "n_queries": len(sample["queries"]),
                "sample_meta": sample["manifest"].get("sample_meta", {}),
                "arms": arms,
            }
            # Drop bulky leaf lists from the JSONL default; keep in a side file
            # only for the intro example later.
            slim_arms = {}
            for name, data in arms.items():
                slim = dict(data)
                slim.pop("leaf_start", None)
                slim.pop("leaf_len", None)
                # Drop per-query topk/ref lists from aggregate stream.
                slim_q = []
                for q in slim["queries"]:
                    slim_q.append(
                        {
                            k: v
                            for k, v in q.items()
                            if k not in ("topk", "ref")
                        }
                    )
                slim["queries"] = slim_q
                slim_arms[name] = slim
            row_slim = dict(row)
            row_slim["arms"] = slim_arms
            stream.write(json.dumps(row_slim) + "\n")
            stream.flush()
            rows.append(row_slim)
            # Keep one full arm snapshot path for plotting later.
            (args.out / "full_arms").mkdir(exist_ok=True)
            torch.save(
                {
                    "sample_dir": str(path),
                    "layer": sample["layer"],
                    "req": sample["req"],
                    "n_complete": sample["n_complete"],
                    "arms": {
                        name: {
                            "leaf_start": data["leaf_start"],
                            "leaf_len": data["leaf_len"],
                            "heterogeneity": data["heterogeneity"],
                            "sse": data["sse"],
                            "num_leaves": data["num_leaves"],
                        }
                        for name, data in arms.items()
                    },
                },
                args.out / "full_arms" / f"{sample['layer']}__{sample['req']}.pt",
            )
    summary = aggregate(rows, args.ks, args.primary_metric)
    summary["config"] = {
        "dump": str(args.dump),
        "adaptive_method": "P-key",
        "split_divisor": args.split_divisor,
        "merge_target_divisor": args.merge_target_divisor,
        "merge_policy": args.merge_policy,
        "only_dataset": args.only_dataset or None,
        "merge_target_rounds": args.merge_rounds,
        "candidate_tokens": args.budget,
        "sink_tokens": args.sink,
        "tail_tokens": args.tail,
        "topk": args.topk,
        "oracle": "dump_reference_topk" if args.use_dump_ref else "recomputed_dense_topk",
        "primary_metric": args.primary_metric,
    }
    example = pick_intro_example(rows, k=128 if 128 in args.ks else args.ks[0])
    summary["intro_example"] = example
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"samples": len(rows), "summary_arms": list(summary["arms"])}, indent=2))
    print(json.dumps(summary.get("paired", {}), indent=2))
    if example:
        print("intro_example:", json.dumps(example, indent=2))


if __name__ == "__main__":
    main()
