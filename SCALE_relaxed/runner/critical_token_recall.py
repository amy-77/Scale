#!/usr/bin/env python3
"""Why a 0.5-point candidate-recall gain can sit next to a large RULER gap.

On the same 128K RULER replay that produced the matched N/64 curve, report:

* average Top-2048 candidate recall (the published +0.49 point number);
* recall restricted to oracle tokens that sit in high Key-SSE fixed blocks,
  which is where mean pooling is a bad summary;
* whether the gold answer span itself is inside the 8192 candidates.

The replay is partition fidelity only. It does not apply the dense-final
prefill policy used by the published RULER numbers.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "runner"))

from compare_fixed_adaptive_chunks import (  # noqa: E402
    adaptive_leaves,
    build_summaries,
    exact_token_scores,
    fixed_leaves,
    load_sample,
    recall_at_k,
    select_and_rerank,
)

ROWS = Path(os.environ.get("SCALE_REPLAY_ROWS", "/datasets/replay/per_query_rows.jsonl"))
RULER = Path(os.environ.get("RULER_DATA_ROOT", "/datasets/ruler")) / "128k"
INVENTORY = Path(os.environ.get("SCALE_REPLAY_INVENTORY", "/datasets/replay/inventory.json"))
MODEL = Path(os.environ.get("MODEL_PATH", "/models/DeepSeek-V3.2"))
OUT = Path(os.environ.get("SCALE_CRITICAL_RESULTS", PACKAGE.parent / "results/critical_token"))
MAX_PROMPT = 130560
BUDGET = 8192
SINK = 64
TAIL = 256
TOPK = 2048


def remap(pos: int, original: int, truncated: bool) -> int | None:
    if not truncated:
        return pos
    head = MAX_PROMPT // 2
    tail = MAX_PROMPT - head
    if pos < head:
        return pos
    tail_start = original - tail
    if pos >= tail_start:
        return head + (pos - tail_start)
    return None


def load_records() -> dict[tuple[str, str], dict]:
    found = {}
    for path in sorted(RULER.glob("*/validation.jsonl")):
        task = path.parent.name
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            found[(task, str(row["index"]))] = row
    return found


def answer_spans(tokenizer, ids: list[int], record: dict) -> tuple[list[int], str]:
    """Return served-sequence token ids of the gold answer, plus a status."""
    answers = [str(x) for x in record.get("outputs") or []]
    if not answers:
        return [], "no_answer"
    spans = []
    ambiguous = False
    preferred = record.get("token_position_answer")
    for answer in answers:
        needle = tokenizer.encode(answer, add_special_tokens=False)
        if not needle:
            continue
        hits = []
        n = len(needle)
        for start in range(0, len(ids) - n + 1):
            if ids[start : start + n] == needle:
                hits.append(list(range(start, start + n)))
        if preferred is not None:
            anchored = [span for span in hits if span[0] == int(preferred)]
            if anchored:
                hits = anchored
        if len(hits) > 4:
            ambiguous = True
            continue
        spans.extend(hits)
    if ambiguous and not spans:
        return [], "ambiguous"
    flat = sorted({token for span in spans for token in span})
    if not flat:
        return [], "not_found"
    return flat, "ambiguous" if ambiguous else "ok"


def block_sse(keys: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    n = int(keys.shape[0])
    out = torch.empty(n, device=keys.device, dtype=torch.float32)
    for start in range(0, n, chunk):
        block = keys[start : start + chunk]
        err = (block - block.mean(0, keepdim=True)).pow(2).sum(dim=-1)
        out[start : start + chunk] = err
    return out


def cover(tokens: list[int], selected: torch.Tensor) -> float:
    if not tokens:
        return float("nan")
    chosen = set(int(x) for x in selected.tolist() if int(x) >= 0)
    return sum(1 for token in tokens if token in chosen) / len(tokens)


def main() -> None:
    from transformers import AutoTokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    records = load_records()
    inventory = {
        (row["task"], str(row["source_id"])): row
        for row in json.loads(INVENTORY.read_text())
        if row.get("length") == "128k"
    }

    samples = []
    seen = set()
    for line in ROWS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row["sample_meta"]
        if meta.get("dataset") != "ruler" or meta.get("length") != "128k":
            continue
        sample_dir_text = str(row["sample_dir"])
        old_prefix = os.environ.get("SCALE_REPLAY_OLD_PREFIX")
        new_prefix = os.environ.get("SCALE_REPLAY_NEW_PREFIX", "")
        if old_prefix:
            sample_dir_text = sample_dir_text.replace(old_prefix, new_prefix)
        sample_dir = Path(sample_dir_text)
        key = str(sample_dir)
        if key in seen:
            continue
        seen.add(key)
        samples.append((sample_dir, meta))

    details = []
    cache: dict[tuple[str, str], tuple[list[int], str, int]] = {}
    for index, (sample_dir, meta) in enumerate(samples, 1):
        task = meta["task"]
        source = str(meta["source_id"])
        print(f"[{index}/{len(samples)}] {task} {sample_dir.parent.name}", flush=True)
        sample = load_sample(sample_dir, device)
        n_complete = int(sample["n_complete"])
        fixed = fixed_leaves(n_complete, 64, 0, device)
        adaptive, _part = adaptive_leaves(
            sample["k_fp8"],
            sample["k_scale"],
            n_complete,
            divisor=64,
            split_divisor=8,
            merge_rounds=8,
            device=device,
        )
        fixed_mean, fixed_scale, _ = build_summaries(
            sample["k_fp8"], sample["k_scale"], fixed, n_complete
        )
        adapt_mean, adapt_scale, _ = build_summaries(
            sample["k_fp8"], sample["k_scale"], adaptive, n_complete
        )
        keys = sample["k_fp8"][:n_complete].float() * sample["k_scale"][:n_complete, None]
        sse = block_sse(keys).detach()

        ident = (task, source)
        if ident not in cache:
            record = records.get(ident)
            info = inventory.get(ident)
            if record is None or info is None:
                cache[ident] = ([], "missing_record", 0)
            else:
                ids = tokenizer.encode(record["input"], add_special_tokens=False)
                if len(ids) != int(info["prompt_tokens_original"]):
                    cache[ident] = ([], "length_mismatch", len(ids))
                else:
                    truncated = len(ids) > MAX_PROMPT
                    served = []
                    for pos in range(len(ids)):
                        mapped = remap(pos, len(ids), truncated)
                        if mapped is not None:
                            served.append(ids[pos])
                    # answer positions are in the original encoding
                    original_spans, status = answer_spans(tokenizer, ids, record)
                    mapped_tokens = []
                    for pos in original_spans:
                        mapped = remap(pos, len(ids), truncated)
                        if mapped is not None and mapped < n_complete:
                            mapped_tokens.append(mapped)
                    cache[ident] = (mapped_tokens, status if mapped_tokens else "dropped", len(ids))
        critical, status, encoded_len = cache[ident]

        for query in sample["queries"]:
            ke = min(int(query["ke"]), int(sample["k_fp8"].shape[0]))
            if ke <= TOPK:
                continue
            exact = exact_token_scores(
                query["q_fp8"], query["w"], sample["k_fp8"], sample["k_scale"], ke
            )
            ref = torch.topk(exact, min(TOPK, ke)).indices
            fixed_sel = select_and_rerank(
                query["q_fp8"], query["w"], sample["k_fp8"], sample["k_scale"],
                fixed, fixed_mean, fixed_scale, ke, BUDGET, SINK, TAIL, TOPK,
            )
            adapt_sel = select_and_rerank(
                query["q_fp8"], query["w"], sample["k_fp8"], sample["k_scale"],
                adaptive, adapt_mean, adapt_scale, ke, BUDGET, SINK, TAIL, TOPK,
            )
            ref_pos = ref[(ref >= 0) & (ref < n_complete)]
            ref_sse = sse[ref_pos]
            threshold = torch.quantile(ref_sse, 0.75)
            high = ref_pos[ref_sse >= threshold]
            low = ref_pos[ref_sse <= torch.quantile(ref_sse, 0.25)]
            details.append(
                {
                    "task": task,
                    "source_id": source,
                    "layer": sample["layer"],
                    "query": query["name"],
                    "fixed_recall": recall_at_k(ref, fixed_sel["candidates"]),
                    "adaptive_recall": recall_at_k(ref, adapt_sel["candidates"]),
                    "fixed_high_sse_recall": recall_at_k(high, fixed_sel["candidates"]),
                    "adaptive_high_sse_recall": recall_at_k(high, adapt_sel["candidates"]),
                    "fixed_low_sse_recall": recall_at_k(low, fixed_sel["candidates"]),
                    "adaptive_low_sse_recall": recall_at_k(low, adapt_sel["candidates"]),
                    "critical_status": status,
                    "n_critical": len(critical),
                    "encoded_len": encoded_len,
                    "fixed_critical_hit": cover(critical, fixed_sel["candidates"]),
                    "adaptive_critical_hit": cover(critical, adapt_sel["candidates"]),
                    "fixed_critical_topk": cover(critical, fixed_sel["topk"]),
                    "adaptive_critical_topk": cover(critical, adapt_sel["topk"]),
                }
            )
            del exact
        del keys, sse, sample
        torch.cuda.empty_cache()

    (OUT / "per_query.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in details)
    )
    summary = summarize(details)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def mean(values: list[float]) -> float | None:
    usable = [value for value in values if value == value]
    if not usable:
        return None
    return sum(usable) / len(usable)


def summarize(rows: list[dict]) -> dict:
    def pack(subset: list[dict]) -> dict:
        return {
            "n": len(subset),
            "fixed_candidate_recall": mean([row["fixed_recall"] for row in subset]),
            "adaptive_candidate_recall": mean([row["adaptive_recall"] for row in subset]),
            "delta_candidate_recall": mean(
                [row["adaptive_recall"] - row["fixed_recall"] for row in subset]
            ),
            "fixed_high_sse_recall": mean([row["fixed_high_sse_recall"] for row in subset]),
            "adaptive_high_sse_recall": mean(
                [row["adaptive_high_sse_recall"] for row in subset]
            ),
            "delta_high_sse_recall": mean(
                [
                    row["adaptive_high_sse_recall"] - row["fixed_high_sse_recall"]
                    for row in subset
                ]
            ),
            "delta_low_sse_recall": mean(
                [
                    row["adaptive_low_sse_recall"] - row["fixed_low_sse_recall"]
                    for row in subset
                ]
            ),
            "fixed_critical_hit": mean([row["fixed_critical_hit"] for row in subset]),
            "adaptive_critical_hit": mean(
                [row["adaptive_critical_hit"] for row in subset]
            ),
            "fixed_critical_topk": mean([row["fixed_critical_topk"] for row in subset]),
            "adaptive_critical_topk": mean(
                [row["adaptive_critical_topk"] for row in subset]
            ),
        }

    by_task = {}
    tasks = sorted({row["task"] for row in rows})
    for task in tasks:
        by_task[task] = pack([row for row in rows if row["task"] == task])
    retrieval = [
        row
        for row in rows
        if row["task"].startswith("niah_") or row["task"] == "vt"
    ]
    return {
        "protocol": {
            "split": "N/8",
            "merge": "N/64",
            "merge_rounds": 8,
            "candidate_tokens": BUDGET,
            "topk": TOPK,
            "note": "Same replay as the matched-budget recall curve. Not dense-final generation.",
        },
        "overall": pack(rows),
        "retrieval_tasks": pack(retrieval),
        "by_task": by_task,
    }


if __name__ == "__main__":
    main()
