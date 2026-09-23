#!/usr/bin/env python3
"""Batch-convert broad online dumps into the paired-runner tree layout."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "runner"))

from convert_raw_indexer_dump import convert_layer  # noqa: E402


def sanitize_rid(rid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:=-]+", "_", rid)[:160]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only-dataset")
    ap.add_argument("--index-offset", type=int, default=0)
    args = ap.parse_args()

    samples = [
        json.loads(line)
        for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]
    if args.only_dataset:
        samples = [
            row for row in samples if row["dataset"] == args.only_dataset
        ]
    converted = []
    for sample_index, meta in enumerate(samples, 1 + args.index_offset):
        raw_request = args.raw_root / sanitize_rid(meta["rid"])
        if not raw_request.is_dir():
            raise FileNotFoundError(f"missing raw request directory: {raw_request}")
        req_name = f"req_{sample_index:05d}_epoch_000"
        layer_rows = []
        for raw_layer in sorted(raw_request.glob("L*")):
            if not raw_layer.is_dir():
                continue
            layer_id = int(raw_layer.name[1:])
            out_dir = (
                args.out
                / "rank_00"
                / f"layer_{layer_id:03d}"
                / req_name
            )
            row = convert_layer(raw_layer, out_dir)
            manifest_path = out_dir / "manifest.json"
            layer_manifest = json.loads(manifest_path.read_text())
            layer_manifest["sample_meta"] = meta
            layer_manifest["sample_index"] = sample_index
            manifest_path.write_text(json.dumps(layer_manifest, indent=2))
            layer_rows.append(row)
        if not layer_rows:
            raise RuntimeError(f"no layer dumps under {raw_request}")
        record = {
            "sample_index": sample_index,
            "req": req_name,
            "meta": meta,
            "layers": layer_rows,
        }
        converted.append(record)
        print(
            json.dumps(
                {
                    "sample": sample_index,
                    "total": len(samples),
                    "tag": meta["tag"],
                    "layers": [row["layer"] for row in layer_rows],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = args.only_dataset or "all"
    (args.out / f"conversion_{suffix}_{args.index_offset}.json").write_text(
        json.dumps(converted, indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
