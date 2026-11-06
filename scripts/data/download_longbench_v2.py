#!/usr/bin/env python3
"""Download pinned official LongBench v2, optionally reconstructing paper401."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import download, hf_endpoint, record_hash, sha256, write_json

REPO = Path(__file__).resolve().parents[2]
DATASET = 'zai-org/LongBench-v2'  # Official THUDM/LongBench-v2 URL redirects here.
REVISION = '2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9'
FIELDS = {'_id', 'context', 'question', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'answer', 'domain', 'sub_domain', 'length', 'difficulty'}


def verify_paper_rule(rows: list[dict], spec: dict) -> None:
    """Recompute the original 503 - 96 train - 6 validation selection."""
    if len(rows) != 503:
        raise ValueError('Reconstructing paper401 requires the full official 503 records')
    sys.path.insert(0, str(REPO/'SCALE_relaxed/runner/evaluators'))
    from collect_longbench_v2 import select_stratified

    excluded = {
        split: {str(row['_id']) for row in select_stratified(rows, split)}
        for split in ('train', 'validation')
    }
    if len(excluded['train']) != 96 or len(excluded['validation']) != 6:
        raise ValueError('Unexpected original train/validation split sizes')
    excluded_ids = excluded['train'] | excluded['validation']
    expected_ids = [str(row['_id']) for row in rows if str(row['_id']) not in excluded_ids]
    if len(excluded_ids) != 102 or expected_ids != spec['ids']:
        raise ValueError('Original selection rule does not reproduce the frozen paper401 IDs/order')
    recorded = spec.get('selection', {}).get('excluded_ids')
    if recorded is not None and recorded != {k: sorted(v) for k, v in excluded.items()}:
        raise ValueError('Recorded exclusion IDs disagree with the original selector')


def select(rows: list[dict], subset: str, manifest: Path) -> list[dict]:
    by_id = {str(row['_id']): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('Duplicate LongBench v2 IDs')
    for row in rows:
        if not FIELDS.issubset(row):
            raise ValueError(f"Missing fields in {row.get('_id')}")
    if subset == 'full':
        if len(rows) != 503:
            raise ValueError(f'Expected the pinned official 503 records, got {len(rows)}')
        return rows
    spec = json.loads(manifest.read_text())
    verify_paper_rule(rows, spec)
    selected = []
    for ident in spec['ids']:
        if ident not in by_id:
            raise ValueError(f'Missing paper record {ident}')
        row = by_id[ident]
        if record_hash(row) != spec['records'][ident]:
            raise ValueError(f'Paper record content differs: {ident}')
        selected.append(row)
    if len(selected) != spec['count']:
        raise ValueError('Unexpected paper split size')
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subset', choices=['full', 'paper401'], default='full')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, default=REPO/'data/cache/longbench_v2')
    parser.add_argument('--input-file', type=Path, help='Use an already downloaded official data.json')
    parser.add_argument('--manifest', type=Path, default=REPO/'experiments/20261107/longbench_heldout_manifest.json')
    args = parser.parse_args()
    source = args.input_file or args.cache_dir/REVISION/'data.json'
    url = f'{hf_endpoint()}/datasets/{DATASET}/resolve/{REVISION}/data.json'
    if args.input_file is None:
        download(url, source)
    rows = json.loads(source.read_text(encoding='utf-8'))
    selected = select(rows, args.subset, args.manifest)
    write_json(args.output, selected)
    write_json(args.output.with_suffix('.manifest.json'), {
        'dataset': DATASET, 'revision': REVISION, 'url': url,
        'official_split_name': 'train', 'subset': args.subset,
        'examples': len(selected), 'source_sha256': sha256(source),
        'output_sha256': sha256(args.output),
        'paper_records_verified': args.subset == 'paper401',
        'paper_selection_rule_verified': args.subset == 'paper401',
    })
    print(f'Wrote {len(selected)} records to {args.output}')


if __name__ == '__main__':
    main()
