#!/usr/bin/env python3
"""Run a fixed or expanded RULER split against an SGLang server."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from collect_longbench_v2 import JsonTokenizer, generate
from collect_ruler import (
    infer_length,
    infer_task,
    prompt_text,
    read_records,
    sample_index,
    selected_records,
)


TASK_CONFIG = {
    "niah_single_1": ("all", 128),
    "niah_single_2": ("all", 128),
    "niah_single_3": ("all", 128),
    "niah_multikey_1": ("all", 128),
    "niah_multikey_2": ("all", 128),
    "niah_multikey_3": ("all", 128),
    "niah_multivalue": ("all", 128),
    "niah_multiquery": ("all", 128),
    "vt": ("all", 64),
    "cwe": ("all", 256),
    "fwe": ("all", 128),
    "qa_1": ("part", 64),
    "qa_2": ("part", 64),
}


def score_prediction(prediction: str, references: list[str], mode: str) -> float:
    hits = [float(reference.lower() in prediction.lower()) for reference in references]
    return max(hits) if mode == "part" else sum(hits) / max(len(hits), 1)


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok":
            completed[str(row["source_id"])] = row
    return completed


def latest_rows(path: Path) -> list[dict[str, Any]]:
    """Return one final row per source id from an append/resume output."""
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[str(row["source_id"])] = row
    return sorted(
        latest.values(),
        key=lambda row: (
            str(row.get("task", "")),
            str(row.get("length", "")),
            int(row.get("index", 0)),
        ),
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("status") == "ok"]
    by_length = {}
    by_task = {}
    for field, target in (("length", by_length), ("task", by_task)):
        values = sorted({str(row[field]) for row in ok})
        for value in values:
            selected = [row for row in ok if str(row[field]) == value]
            target[value] = sum(float(row["score"]) for row in selected) / len(selected)
    return {
        "examples": len(rows),
        "successful": len(ok),
        "accuracy": sum(float(row["score"]) for row in ok) / max(len(ok), 1),
        "by_length": by_length,
        "by_task": by_task,
        "elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
    }


def all_generated_records(
    data_root: Path,
) -> list[tuple[str, str, int, dict[str, Any]]]:
    """Load every generated validation record, including indices beyond 0-7."""
    result = []
    for path in sorted(data_root.rglob("validation.jsonl")):
        records = read_records(path)
        for fallback, record in enumerate(records):
            length = infer_length(path, record)
            if length not in {"32k", "128k"}:
                continue
            result.append(
                (
                    infer_task(path, record),
                    length,
                    sample_index(record, fallback),
                    record,
                )
            )
    return sorted(result, key=lambda item: (item[0], item[1], item[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:31000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--lengths",
        nargs="+",
        choices=["32k", "128k"],
        default=["32k", "128k"],
    )
    parser.add_argument("--tasks", nargs="+", help="restrict to these tasks")
    parser.add_argument(
        "--all-records",
        action="store_true",
        help="evaluate every record rather than only fixed test index 7",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    tokenizer = JsonTokenizer(args.model)
    selected = (
        all_generated_records(args.data_root)
        if args.all_records
        else selected_records(args.data_root, "test")
    )
    selected = [item for item in selected if item[1] in args.lengths]
    if args.tasks:
        selected = [item for item in selected if item[0] in args.tasks]
    if args.max_records is not None:
        selected = selected[: args.max_records]
    completed = load_completed(args.output) if args.resume else {}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as stream:
        for task, length, index, record in selected:
            source_id = f"ruler:{task}:{length}:{index}"
            if source_id in completed:
                continue
            metric_mode, max_new_tokens = TASK_CONFIG[task]
            chat_prompt = (
                "<｜User｜>"
                + prompt_text(record)
                + "<｜Assistant｜></think>"
            )
            input_ids = tokenizer.encode(chat_prompt, add_special_tokens=True)
            started = time.time()
            try:
                output = generate(args.server, input_ids, max_new_tokens)
                prediction = str(output.get("text", ""))
                row = {
                    "source_id": source_id,
                    "task": task,
                    "length": length,
                    "index": index,
                    "prompt_tokens": len(input_ids),
                    "prediction": prediction,
                    "references": record["outputs"],
                    "score": score_prediction(
                        prediction, list(record["outputs"]), metric_mode
                    ),
                    "status": "ok",
                    "error": None,
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            except Exception as exc:
                row = {
                    "source_id": source_id,
                    "task": task,
                    "length": length,
                    "index": index,
                    "status": "error",
                    "error": repr(exc),
                    "elapsed_seconds": round(time.time() - started, 3),
                }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{source_id}: {row['status']} score={row.get('score')}", flush=True)

    rows = latest_rows(args.output)
    report = summarize(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
