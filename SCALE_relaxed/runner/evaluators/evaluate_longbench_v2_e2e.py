#!/usr/bin/env python3
"""Run and score the complete LongBench-v2 multiple-choice set."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from collect_longbench_v2 import JsonTokenizer, encode_prompt


ANSWER_PATTERNS = (
    re.compile(r"correct answer is\s*\(?([ABCD])\)?", re.IGNORECASE),
    re.compile(r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([ABCD])\)?", re.IGNORECASE),
    re.compile(r"^\s*\(?([ABCD])\)?(?:[.)\s]|$)", re.IGNORECASE),
)


def generate(
    server: str, input_ids: list[int], max_new_tokens: int
) -> dict[str, Any]:
    """Generate a direct answer, allowing EOS but leaving room for formatting.

    The data-collection helper intentionally forces every requested token with
    ``ignore_eos=True``.  That is useful for fixed-length probe collection but
    biases downstream accuracy when a short cap cuts the option letter off.
    """
    payload = json.dumps(
        {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": False,
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


def extract_answer(text: str) -> str | None:
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok":
                completed.add(str(row["source_id"]))
    return completed


def latest_rows(path: Path) -> list[dict[str, Any]]:
    """Return one final row per source id from an append/resume output."""
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[str(row["source_id"])] = row
    return sorted(latest.values(), key=lambda row: int(row.get("index", 0)))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["status"] == "ok"]
    summary: dict[str, Any] = {
        "examples": len(rows),
        "successful": len(successful),
        "accuracy": (
            sum(float(row["score"]) for row in successful) / len(successful)
            if successful
            else 0.0
        ),
    }
    for field in ("length", "difficulty", "domain"):
        buckets: dict[str, list[float]] = defaultdict(list)
        for row in successful:
            buckets[str(row[field])].append(float(row["score"]))
        summary[f"by_{field}"] = {
            key: sum(values) / len(values) for key, values in sorted(buckets.items())
        }
    return summary


def evaluate_sample(
    index: int,
    sample: dict[str, Any],
    *,
    tokenizer: JsonTokenizer,
    server: str,
    max_context_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    source_id = str(sample["_id"])
    started = time.time()
    input_ids = encode_prompt(
        tokenizer,
        sample,
        max_context_tokens,
        chat_template=True,
    )
    status = "ok"
    error = None
    prediction = None
    text = ""
    try:
        response = generate(server, input_ids, max_new_tokens)
        text = str(response.get("text") or "")
        prediction = extract_answer(text)
    except Exception as exc:
        status = "error"
        error = repr(exc)
    return {
        "source_id": source_id,
        "index": index,
        "length": sample["length"],
        "difficulty": sample["difficulty"],
        "domain": sample["domain"],
        "prompt_tokens": len(input_ids),
        "prediction": prediction,
        "answer": str(sample["answer"]).upper(),
        "score": (
            float(prediction == str(sample["answer"]).upper())
            if status == "ok"
            else None
        ),
        "generated_text": text,
        "status": status,
        "error": error,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:31000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-context-tokens", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of simultaneous server requests. Match server max-running-requests.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")

    with args.data.open(encoding="utf-8") as stream:
        records = json.load(stream)
    if args.max_records is not None:
        records = records[: args.max_records]
    tokenizer = JsonTokenizer(args.model)
    done = completed_ids(args.output) if args.resume else set()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as output:
        pending = [
            (index, sample)
            for index, sample in enumerate(records)
            if str(sample["_id"]) not in done
        ]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:
            futures = [
                executor.submit(
                    evaluate_sample,
                    index,
                    sample,
                    tokenizer=tokenizer,
                    server=args.server,
                    max_context_tokens=args.max_context_tokens,
                    max_new_tokens=args.max_new_tokens,
                )
                for index, sample in pending
            ]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(
                    f"longbench:{row['index']}:{row['source_id']}: "
                    f"{row['status']} score={row['score']}",
                    flush=True,
                )

    rows = latest_rows(args.output)
    report = summarize(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
