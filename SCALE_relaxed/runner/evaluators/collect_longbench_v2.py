#!/usr/bin/env python3
"""Send a prompt-stratified LongBench-v2 split to an SGLang server."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


class JsonTokenizer:
    def __init__(self, model_path: str):
        self.tokenizer = Tokenizer.from_file(str(Path(model_path) / "tokenizer.json"))

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        return self.tokenizer.encode(
            text, add_special_tokens=add_special_tokens
        ).ids


def build_parts(sample: dict[str, Any]) -> tuple[str, str, str]:
    prefix = "Please read the following text and answer the question below.\n\n<text>\n"
    suffix = (
        "\n</text>\n\nWhat is the correct answer to this question: "
        f"{sample['question']}\nChoices:\n"
        f"(A) {sample['choice_A']}\n"
        f"(B) {sample['choice_B']}\n"
        f"(C) {sample['choice_C']}\n"
        f"(D) {sample['choice_D']}\n\n"
        'Format your response as follows: "The correct answer is (insert answer here)".'
    )
    return prefix, sample["context"], suffix


def encode_prompt(
    tokenizer: Any,
    sample: dict[str, Any],
    max_context_tokens: int,
    *,
    chat_template: bool = False,
) -> list[int]:
    prefix, context, suffix = build_parts(sample)
    if chat_template:
        prefix = "<｜User｜>" + prefix
        suffix += "<｜Assistant｜></think>"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    if max_context_tokens <= 0:
        return prefix_ids + context_ids + suffix_ids
    budget = max_context_tokens - len(prefix_ids) - len(suffix_ids)
    if budget <= 0:
        raise ValueError("Question and choices exceed the context budget")
    if len(context_ids) > budget:
        head = budget // 2
        context_ids = context_ids[:head] + context_ids[-(budget - head) :]
    return prefix_ids + context_ids + suffix_ids


SPLIT_OFFSETS = {
    "train": (0, 16),
    "validation": (16, 17),
    "calibration": (17, 18),
    "test": (18, 19),
}


def select_stratified(
    records: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    """Select 16/1/1/1 records from each difficulty×length stratum.

    Samples are first ordered with a stable hash inside each domain, then
    domains are round-robined so the 19 selected records remain diverse.
    """
    if split not in SPLIT_OFFSETS:
        raise ValueError(split)
    groups: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample in records:
        key = (sample["difficulty"], sample["length"])
        groups[key][sample["domain"]].append(sample)

    selected: list[dict[str, Any]] = []
    begin, end = SPLIT_OFFSETS[split]
    for stratum in sorted(groups):
        by_domain = groups[stratum]
        queues = {}
        for domain, values in by_domain.items():
            values.sort(
                key=lambda row: hashlib.sha256(
                    str(row["_id"]).encode()
                ).digest()
            )
            queues[domain] = deque(values)

        ordered = []
        domains = sorted(queues)
        while domains and len(ordered) < end:
            next_domains = []
            for domain in domains:
                if queues[domain] and len(ordered) < end:
                    ordered.append(queues[domain].popleft())
                if queues[domain]:
                    next_domains.append(domain)
            domains = next_domains
        if len(ordered) < end:
            raise RuntimeError(
                f"Stratum {stratum} has only {len(ordered)} usable samples; "
                f"need {end}"
            )
        selected.extend(ordered[begin:end])
    return selected


def generate(server: str, input_ids: list[int], max_new_tokens: int) -> dict[str, Any]:
    payload = json.dumps(
        {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        }
    ).encode()
    request = urllib.request.Request(
        f"{server.rstrip('/')}/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def completed_source_ids(manifest_path: Path, split: str) -> set[str]:
    if not manifest_path.exists():
        return set()
    completed = set()
    with manifest_path.open(encoding="utf-8") as manifest:
        for line in manifest:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("split") == split and row.get("status") == "ok":
                completed.add(str(row["source_id"]))
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:31000")
    parser.add_argument("--control-file", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--split",
        choices=["train", "validation", "calibration", "test"],
        required=True,
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=0,
        help="0 preserves the original LongBench-v2 length",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as stream:
        records = json.load(stream)
    selected = select_stratified(records, args.split)
    tokenizer = JsonTokenizer(args.model)

    control_path = Path(args.control_file)
    control_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        completed = completed_source_ids(manifest_path, args.split)
        selected = [sample for sample in selected if sample["_id"] not in completed]
        print(
            f"[resume] skipped {len(completed)} completed {args.split} prompts; "
            f"{len(selected)} remain",
            flush=True,
        )

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for index, sample in enumerate(selected):
            started = time.time()
            input_ids = encode_prompt(tokenizer, sample, args.max_context_tokens)
            control_path.write_text(
                json.dumps({"split": args.split, "source_id": sample["_id"]}),
                encoding="utf-8",
            )
            status = "ok"
            error = None
            output: dict[str, Any] = {}
            try:
                output = generate(args.server, input_ids, args.max_new_tokens)
            except Exception as exc:  # Keep collection moving and log the failure.
                status = "error"
                error = repr(exc)

            row = {
                "source_id": sample["_id"],
                "split": args.split,
                "domain": sample["domain"],
                "difficulty": sample["difficulty"],
                "length": sample["length"],
                "prompt_tokens": len(input_ids),
                "status": status,
                "error": error,
                "elapsed_seconds": round(time.time() - started, 3),
                "generated_text": output.get("text"),
            }
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest.flush()
            print(
                f"[{index + 1}/{len(selected)}] {sample['_id']} "
                f"{len(input_ids)} tokens {status} {row['elapsed_seconds']}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
