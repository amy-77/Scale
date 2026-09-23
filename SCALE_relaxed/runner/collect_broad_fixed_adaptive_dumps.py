#!/usr/bin/env python3
"""Collect one real dump per available RULER/LongBench-v2 task-length group.

Selection is deterministic:

* RULER: one median-character-length record from every available
  ``<length>/<task>/validation.jsonl`` file.
* LongBench-v2: one median-context-length record from every available
  ``(domain, sub_domain, short|medium|long)`` group.

The server must be launched with ``SGLANG_NSA_INDEXER_DUMP_DIR`` and an RID
prefix matching ``broad:``.  Results are resumable through ``manifest.jsonl``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path


def longbench_prompt(row: dict) -> str:
    choices = [row.get(f"choice_{x}", row.get(x, "")) for x in "ABCD"]
    return (
        "Please read the following text and answer the question below.\n\n"
        f"<text>\n{row.get('context', '').strip()}\n</text>\n\n"
        f"What is the correct answer to this question: {row['question'].strip()}\n"
        "Choices:\n"
        + "\n".join(
            f"({letter}) {choice}" for letter, choice in zip("ABCD", choices)
        )
        + '\n\nFormat your response as: "The correct answer is (A/B/C/D)".'
    )


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def select_ruler(root: Path, evaluator_dir: Path) -> list[dict]:
    sys.path.insert(0, str(evaluator_dir))
    from collect_ruler import prompt_text, read_records  # noqa: PLC0415

    out = []
    for path in sorted(root.glob("*/*/validation.jsonl")):
        length, task = path.parent.parent.name, path.parent.name
        records = read_records(path)
        if not records:
            continue
        ranked = sorted(
            records,
            key=lambda row: (
                len(row.get("input", "")),
                str(row.get("index", "")),
            ),
        )
        row = ranked[len(ranked) // 2]
        out.append(
            {
                "dataset": "ruler",
                "task": task,
                "subtask": task,
                "length": length,
                "difficulty": None,
                "source_id": str(row.get("index", "")),
                "tag": f"ruler:{length}:{task}",
                "rid": f"broad:ruler:{length}:{task}",
                "text": prompt_text(row),
            }
        )
    return out


def select_longbench(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in data:
        groups[(row["domain"], row["sub_domain"], row["length"])].append(row)
    out = []
    for (domain, subdomain, length), rows in sorted(groups.items()):
        rows.sort(key=lambda row: (len(row.get("context", "")), row.get("_id", "")))
        row = rows[len(rows) // 2]
        tag = f"lbv2:{length}:{slug(domain)}:{slug(subdomain)}"
        out.append(
            {
                "dataset": "longbench_v2",
                "task": domain,
                "subtask": subdomain,
                "length": length,
                "difficulty": row.get("difficulty"),
                "source_id": row.get("_id"),
                "tag": tag,
                "rid": "broad:" + tag,
                "text": longbench_prompt(row),
            }
        )
    del data
    return out


def prompt_ids(tokenizer, text: str, max_tokens: int) -> tuple[list[int], int]:
    if tokenizer.chat_template:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        ids = tokenizer.encode(text, add_special_tokens=True)
    original = len(ids)
    if original > max_tokens:
        head = max_tokens // 2
        ids = ids[:head] + ids[-(max_tokens - head) :]
    return list(map(int, ids)), original


def post_generate(
    server: str,
    input_ids: list[int],
    rid: str,
    max_new_tokens: int,
    timeout: int,
    retries: int = 2,
) -> dict:
    payload = json.dumps(
        {
            "input_ids": input_ids,
            "rid": rid,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        }
    ).encode()
    error = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                f"{server.rstrip('/')}/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            error = exc
            if attempt < retries:
                time.sleep(10)
    raise RuntimeError(f"generation failed for {rid}: {error}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", default="http://127.0.0.1:31929")
    ap.add_argument(
        "--model",
        default="/share-evpfs/flagos/models/DeepSeek-V3.2",
    )
    ap.add_argument(
        "--ruler-root",
        type=Path,
        default=Path(
            "/data/jz/sglang_coarse_adapt/h20-9-57/evaluation_inputs/ruler"
        ),
    )
    ap.add_argument(
        "--ruler-evaluator",
        type=Path,
        default=Path(
            "/data/jz/9-57-0920/results_pkey_target64_e2e_20260920/evaluators"
        ),
    )
    ap.add_argument(
        "--longbench",
        type=Path,
        default=Path("/data/jz/longbench_heldout.json"),
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-prompt-tokens", type=int, default=131072 - 512)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--inventory-only", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    specs = select_ruler(args.ruler_root, args.ruler_evaluator)
    specs += select_longbench(args.longbench)
    if args.only:
        specs = [x for x in specs if any(v in x["tag"] for v in args.only)]

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"
    completed = set()
    if manifest_path.exists():
        for line in manifest_path.read_text().splitlines():
            if line.strip():
                completed.add(json.loads(line)["tag"])

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    inventory = []
    for index, spec in enumerate(specs, 1):
        ids, original_tokens = prompt_ids(
            tokenizer, spec["text"], args.max_prompt_tokens
        )
        meta = {k: v for k, v in spec.items() if k != "text"}
        meta.update(
            {
                "selection_index": index,
                "prompt_tokens_original": original_tokens,
                "prompt_tokens": len(ids),
                "truncated": original_tokens != len(ids),
            }
        )
        inventory.append(meta)
        status = (
            "done"
            if spec["tag"] in completed
            else ("inventory" if args.inventory_only else "run")
        )
        print(
            f"[{index}/{len(specs)} {status}] {spec['tag']} "
            f"tokens={len(ids)} original={original_tokens}",
            flush=True,
        )
        if status != "run":
            continue
        start = time.time()
        response = post_generate(
            args.server,
            ids,
            spec["rid"],
            args.max_new_tokens,
            args.timeout,
        )
        row = dict(meta)
        row.update(
            {
                "seconds": round(time.time() - start, 3),
                "completion_tokens": response.get("meta_info", {}).get(
                    "completion_tokens"
                ),
                "response_id": response.get("meta_info", {}).get("id"),
                "output": response.get("text", "")[:256],
            }
        )
        with manifest_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        completed.add(spec["tag"])

    (args.out / "inventory.json").write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False)
    )
    counts = defaultdict(int)
    for row in inventory:
        counts[f"{row['dataset']}:{row['length']}"] += 1
    print(
        json.dumps(
            {
                "samples": len(inventory),
                "completed": len(completed & {x["tag"] for x in inventory}),
                "counts": dict(sorted(counts.items())),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
