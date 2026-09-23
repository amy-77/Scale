#!/usr/bin/env python3
"""Drive a dump-enabled SGLang server over a small, diverse prompt set.

Selects
  * RULER regression-subset prompts (tasks where Static Group16 lost the most
    against Official DSA), ``--ruler-per-task`` per task at ``--ruler-length``;
  * LongBench-v2 prompts, one per (length, difficulty) group;
sends them one at a time to ``/generate`` with ``rid="dump:..."`` so that
``sglang.srt.layers.attention.nsa.indexer_dump`` records the indexer state of
every layer, and writes a manifest with the generated text.

Example:
  python scripts/collect_indexer_dump.py \
      --server http://127.0.0.1:31730 \
      --tokenizer /DATA/disk0/qyl/models/deepseek-v3.2 \
      --ruler-subset /DATA/disk0/qyl/data/ruler_low_score_subset_20260917 \
      --longbench-json /DATA/disk0/qyl/data/misa_assignment_router_v2/longbench_v2.json \
      --out /DATA/disk0/qyl/data/adaptive_hisa_dump_20260918/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))  # /DATA/disk0/qyl/code/dpskv32 -> collect_ruler.py
from collect_ruler import prompt_text, read_records, sample_index  # noqa: E402

LB_GROUPS = [("short", "easy"), ("short", "hard"), ("medium", "easy"), ("medium", "hard"), ("long", "easy"), ("long", "hard")]


def longbench_prompt(row: dict) -> str:
    choices = [row.get(f"choice_{x}", row.get(x, "")) for x in "ABCD"]
    return (
        "Please read the following text and answer the question below.\n\n"
        f"<text>\n{row.get('context', '').strip()}\n</text>\n\n"
        f"What is the correct answer to this question: {row['question'].strip()}\n"
        "Choices:\n"
        + "\n".join(f"({letter}) {choice}" for letter, choice in zip("ABCD", choices))
        + '\n\nFormat your response as: "The correct answer is (A/B/C/D)".'
    )


def select_ruler(subset: Path, tasks: list[str], length: str, per_task: int) -> list[dict]:
    manifest = [json.loads(l) for l in (subset / "manifest.jsonl").read_text().splitlines() if l.strip()]
    out = []
    for task in tasks:
        cands = sorted(
            (m for m in manifest if m["task"] == task and m["length"] == length),
            key=lambda m: (-m["official_advantage"], m["index"]),
        )[:per_task]
        if not cands:
            print(f"[warn] no RULER candidates for {task}/{length}", file=sys.stderr)
        path = subset / "ruler" / length / task / "validation.jsonl"
        records = read_records(path)
        # The subset tree keeps exactly the selected records.  The manifest
        # index is the held-out position (0-13) while the raw record carries
        # the original dataset index (held-out offset + 6), so align by rank.
        all_idx = sorted(m["index"] for m in manifest if m["task"] == task and m["length"] == length)
        rec_sorted = sorted(enumerate(records), key=lambda t: (t[1].get("index", sample_index(t[1], t[0]))))
        if len(all_idx) == len(rec_sorted):
            by_index = {mi: rec for mi, (_, rec) in zip(all_idx, rec_sorted)}
        else:
            by_index = {}
            for i, rec in enumerate(records):
                by_index.setdefault(sample_index(rec, i), rec)
        for m in cands:
            rec = by_index.get(m["index"])
            if rec is None:
                print(f"[warn] record {m['source_id']} missing in {path}", file=sys.stderr)
                continue
            out.append(
                {
                    "dataset": "ruler",
                    "tag": f"ruler:{task}:{length}:{m['index']}",
                    "task": task,
                    "length": length,
                    "static_score": m["static_score"],
                    "official_score": m["official_score"],
                    "text": prompt_text(rec),
                    "answer": rec.get("outputs") or rec.get("answer"),
                }
            )
    return out


def select_longbench(path: Path, per_group: int) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for length, diff in LB_GROUPS:
        group = [r for r in data if r.get("length") == length and r.get("difficulty") == diff]
        group.sort(key=lambda r: (len(r.get("context", "")), r.get("_id", "")))
        if not group:
            continue
        # deterministic: median context length within the group, then step outwards
        mid = len(group) // 2
        order = [mid] + [i for k in range(1, len(group)) for i in (mid - k, mid + k) if 0 <= i < len(group)]
        for i in order[:per_group]:
            r = group[i]
            out.append(
                {
                    "dataset": "longbench_v2",
                    "tag": f"lbv2:{length}-{diff}:{r['_id']}",
                    "task": r.get("domain"),
                    "length": length,
                    "difficulty": diff,
                    "text": longbench_prompt(r),
                    "answer": r.get("answer"),
                }
            )
    return out


def prompt_ids(tokenizer, text: str, max_tokens: int) -> list[int]:
    if tokenizer.chat_template:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)
    else:
        ids = tokenizer.encode(f"<｜User｜>{text}<｜Assistant｜></think>", add_special_tokens=True)
    if len(ids) > max_tokens:
        head = max_tokens // 2
        ids = ids[:head] + ids[-(max_tokens - head) :]
    return list(map(int, ids))


def post_generate(server: str, input_ids: list[int], rid: str, max_new_tokens: int, timeout: int, retries: int = 2) -> dict:
    payload = json.dumps(
        {
            "input_ids": input_ids,
            "rid": rid,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        }
    ).encode()
    err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(f"{server.rstrip('/')}/generate", data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            err = exc
            if attempt < retries:
                time.sleep(10)
    raise RuntimeError(f"generation failed for {rid}: {err}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:31730")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--ruler-subset", type=Path, required=True)
    ap.add_argument("--ruler-tasks", nargs="+", default=["cwe", "niah_multikey_3", "niah_multiquery", "niah_multivalue", "niah_single_2", "qa_1"])
    ap.add_argument("--ruler-length", default="128k")
    ap.add_argument("--ruler-per-task", type=int, default=2)
    ap.add_argument("--longbench-json", type=Path, required=True)
    ap.add_argument("--longbench-per-group", type=int, default=1)
    ap.add_argument("--max-prompt-tokens", type=int, default=131072 - 512)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="select + tokenize only")
    ap.add_argument("--force", action="store_true", help="replay even when the tag is already in --out")
    ap.add_argument("--only", nargs="*", default=None, help="tags to run (substring match)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    prompts = select_ruler(args.ruler_subset, args.ruler_tasks, args.ruler_length, args.ruler_per_task)
    prompts += select_longbench(args.longbench_json, args.longbench_per_group)
    if args.only:
        prompts = [p for p in prompts if any(s in p["tag"] for s in args.only)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for l in args.out.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["tag"])
    for p in prompts:
        ids = prompt_ids(tok, p["text"], args.max_prompt_tokens)
        p["prompt_tokens"] = len(ids)
        rid = "dump:" + p["tag"]
        status = "done" if p["tag"] in done and not args.force else ("dry" if args.dry_run else "run")
        print(f"[{status}] {rid}  tokens={len(ids)}  task={p['task']}", flush=True)
        if status != "run":
            continue
        t0 = time.time()
        resp = post_generate(args.server, ids, rid, args.max_new_tokens, args.timeout)
        row = {k: v for k, v in p.items() if k != "text"}
        row.update({"rid": rid, "output": resp.get("text", ""), "seconds": round(time.time() - t0, 1), "meta": resp.get("meta_info", {})})
        with args.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"      -> {row['seconds']}s  output={row['output'][:80]!r}", flush=True)


if __name__ == "__main__":
    main()
