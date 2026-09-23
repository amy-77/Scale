#!/usr/bin/env python3
"""Convert an ``indexer_dump.py`` request into paired-runner manifest format.

The raw online dump stores one file per prefill chunk and decode step.  This
converter packs the prefill keys into the first key shard (the partition
prefix), retains each decode key as a subsequent shard, and emits one query
shard per decode step.  No score or reference Top-K is precomputed: the paired
runner recomputes the dense DSA oracle from the converted FP8 tensors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def convert_layer(raw_layer: Path, out_dir: Path) -> dict:
    prefill_paths = sorted(raw_layer.glob("prefill_*.pt"))
    decode_paths = sorted(raw_layer.glob("decode_*.pt"))
    if not prefill_paths or not decode_paths:
        raise ValueError(
            f"{raw_layer}: need both prefill and decode shards "
            f"(got {len(prefill_paths)} and {len(decode_paths)})"
        )

    prefill = [_load(p) for p in prefill_paths]
    positions = torch.cat([x["positions"].reshape(-1) for x in prefill])
    expected = torch.arange(positions.numel(), dtype=positions.dtype)
    if not torch.equal(positions, expected):
        raise ValueError(f"{raw_layer}: prefill positions are not contiguous from zero")

    out_dir.mkdir(parents=True, exist_ok=True)
    n_prefill = int(positions.numel())
    prefix_name = f"keys_00000000_{n_prefill:08d}.pt"
    torch.save(
        {
            "k_fp8": torch.cat([x["k_fp8_u8"] for x in prefill], dim=0),
            "k_scale": torch.cat([x["k_scale"].reshape(-1) for x in prefill]),
        },
        out_dir / prefix_name,
    )

    key_shards = [prefix_name]
    query_shards = []
    next_pos = n_prefill
    for qi, path in enumerate(decode_paths):
        row = _load(path)
        pos = row["positions"].reshape(-1)
        if pos.numel() != 1 or int(pos.item()) != next_pos:
            raise ValueError(
                f"{path}: expected one decode key at {next_pos}, got {pos.tolist()}"
            )
        key_name = f"keys_{next_pos:08d}_{next_pos + 1:08d}.pt"
        torch.save(
            {
                "k_fp8": row["k_fp8_u8"],
                "k_scale": row["k_scale"].reshape(-1),
            },
            out_dir / key_name,
        )
        key_shards.append(key_name)

        qpos = row["q_positions"].reshape(-1)
        if qpos.numel() != 1 or int(qpos.item()) != next_pos:
            raise ValueError(f"{path}: query position does not match its decode key")
        query_name = f"queries_{qi:04d}.pt"
        torch.save(
            {
                "q_fp8": row["q_fp8_u8"],
                "w": row["weights"].to(torch.float32),
                "ke": torch.tensor([int(row["seq_len"])], dtype=torch.int32),
            },
            out_dir / query_name,
        )
        query_shards.append(query_name)
        next_pos += 1

    layer_id = int(prefill[0]["layer_id"])
    manifest = {
        "schema_version": 1,
        "request_slot": 1,
        "layer_id": layer_id,
        "rank": 0,
        "epoch": 0,
        "source_raw_layer": str(raw_layer),
        "q_semantics": "q_fp8 codes; weights already include q_scale and DSA scaling",
        "w_semantics": "raw_gate * H^-0.5 * q_scale * head_dim^-0.5",
        "k_semantics": "k_fp8 codes with per-token k_scale",
        "causality": "each query row carries local [0,ke)",
        "key_shards": key_shards,
        "query_shards": query_shards,
        "saved_key_tokens": next_pos,
        "saved_query_rows": len(query_shards),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return {
        "layer": layer_id,
        "prefill_tokens": n_prefill,
        "decode_queries": len(query_shards),
        "out": str(out_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--raw-request",
        type=Path,
        required=True,
        help="request directory containing Lxx subdirectories",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output root; rank_00/layer_xxx/req_00001_epoch_000 is created",
    )
    args = ap.parse_args()

    rows = []
    for raw_layer in sorted(args.raw_request.glob("L*")):
        if not raw_layer.is_dir():
            continue
        layer_id = int(raw_layer.name[1:])
        out_dir = (
            args.out
            / "rank_00"
            / f"layer_{layer_id:03d}"
            / "req_00001_epoch_000"
        )
        row = convert_layer(raw_layer, out_dir)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if not rows:
        raise SystemExit(f"no Lxx directories under {args.raw_request}")
    (args.out / "conversion.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
