#!/usr/bin/env python3
"""Collect the fixed RULER 32K/128K split through an SGLang server.

For every task and length, records 0-4 are train, 5 validation, 6 calibration,
and 7 test. With 13 tasks and two lengths this yields 130/26/26/26 prompts.
Input files may be JSON arrays or JSONL and must expose ``prompt`` or ``input``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from collect_longbench_v2 import (  # pyright: ignore[reportMissingImports]
    JsonTokenizer,
    completed_source_ids,
    generate,
)


SPLIT_INDICES = {
    "train": {0, 1, 2, 3, 4},
    "validation": {5},
    "calibration": {6},
    "test": {7},
}
LENGTHS = {"32k", "128k"}


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "records", "examples"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported RULER file format: {path}")


def infer_length(path: Path, record: dict[str, Any]) -> str | None:
    candidates = [
        str(record.get("length", "")),
        str(record.get("context_length", "")),
        str(path),
    ]
    joined = " ".join(candidates).lower().replace("_", "")
    if "128k" in joined or "131072" in joined:
        return "128k"
    if "32k" in joined or "32768" in joined:
        return "32k"
    return None


def infer_task(path: Path, record: dict[str, Any]) -> str:
    for key in ("task", "task_name", "sub_domain", "dataset"):
        if record.get(key):
            return str(record[key])
    return path.parent.name if path.parent.name else path.stem


def prompt_text(record: dict[str, Any]) -> str:
    for key in ("prompt", "input", "text"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    raise KeyError("RULER record must contain prompt, input, or text")


def sample_index(record: dict[str, Any], fallback: int) -> int:
    """Return the fixed 0-7 dataset sample index, not NIAH insertion index."""
    sample_id = str(record.get("_id", ""))
    suffix = sample_id.rsplit("-", 1)[-1]
    return int(suffix) if suffix.isdigit() else fallback


def selected_records(
    data_root: Path,
    split: str,
) -> list[tuple[str, str, int, dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    files = (
        [data_root]
        if data_root.is_file()
        else sorted(
            [
                *data_root.rglob("*.jsonl"),
                *data_root.rglob("*.json"),
            ]
        )
    )
    for path in files:
        for record in read_records(path):
            length = infer_length(path, record)
            if length not in LENGTHS:
                continue
            task = infer_task(path, record)
            grouped.setdefault((task, length), []).append(record)

    result = []
    wanted = SPLIT_INDICES[split]
    for (task, length), records in sorted(grouped.items()):
        records = [
            row
            for _, row in sorted(
                (sample_index(row, offset), row)
                for offset, row in enumerate(records)
            )
        ]
        if len(records) < 8:
            raise RuntimeError(
                f"RULER {task}/{length} has {len(records)} records; need 8"
            )
        for index in sorted(wanted):
            actual_index = sample_index(records[index], index)
            if actual_index != index:
                raise RuntimeError(
                    f"RULER {task}/{length} is missing index {index}"
                )
            result.append((task, length, index, records[index]))
    if len(grouped) != 26:
        raise RuntimeError(
            f"Expected 13 tasks x 2 lengths, found {len(grouped)} groups"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:31000")
    parser.add_argument("--control-file", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=["train", "validation", "calibration", "test"],
        required=True,
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--lengths",
        nargs="+",
        choices=sorted(LENGTHS),
        default=sorted(LENGTHS),
        help="Only collect the selected context lengths",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        help="Optional cap after split and length filtering",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    tokenizer = JsonTokenizer(args.model)
    selected = selected_records(args.data_root, args.split)
    selected = [item for item in selected if item[1] in args.lengths]
    if args.max_records is not None:
        selected = selected[: args.max_records]
    args.control_file.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    completed = (
        completed_source_ids(args.manifest, args.split) if args.resume else set()
    )

    with args.manifest.open("a", encoding="utf-8") as manifest:
        for task, length, index, record in selected:
            source_id = f"ruler:{task}:{length}:{index}"
            if source_id in completed:
                continue
            input_ids = tokenizer.encode(
                prompt_text(record), add_special_tokens=True
            )
            args.control_file.write_text(
                json.dumps({"split": args.split, "source_id": source_id}),
                encoding="utf-8",
            )
            started = time.time()
            status = "ok"
            error = None
            output: dict[str, Any] = {}
            try:
                output = generate(
                    args.server, input_ids, args.max_new_tokens
                )
            except Exception as exc:
                status = "error"
                error = repr(exc)
            row = {
                "source_id": source_id,
                "split": args.split,
                "task": task,
                "length": length,
                "index": index,
                "prompt_tokens": len(input_ids),
                "status": status,
                "error": error,
                "elapsed_seconds": round(time.time() - started, 3),
                "generated_text": output.get("text"),
            }
            manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest.flush()
            print(f"{source_id}: {status}", flush=True)


if __name__ == "__main__":
    main()
