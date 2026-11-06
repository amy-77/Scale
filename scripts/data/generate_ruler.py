#!/usr/bin/env python3
"""Generate RULER with the pinned NVIDIA code used by the original experiments."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

from common import download, hf_endpoint, sha256, write_json

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = 'https://github.com/NVIDIA/RULER.git'
REVISION = 'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a'
TASKS = ['niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1',
         'niah_multikey_2', 'niah_multikey_3', 'niah_multivalue', 'niah_multiquery',
         'vt', 'cwe', 'fwe', 'qa_1', 'qa_2']
ESSAY_TASKS = {'niah_single_2', 'niah_single_3', 'niah_multikey_1', 'niah_multivalue', 'niah_multiquery'}


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def prepare_repo(repo: Path) -> None:
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', UPSTREAM, str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', REVISION], check=True)
    if git(repo, 'rev-parse', 'HEAD') != REVISION:
        raise ValueError(f'RULER checkout must be at {REVISION}; use a fresh --ruler-repo directory')
    if git(repo, 'diff', 'HEAD', '--', 'scripts'):
        raise ValueError('RULER generator has tracked modifications')


def prepare_assets(repo: Path, tasks: list[str]) -> dict:
    import nltk
    nltk_root = repo/'.cache/nltk_data'
    for name in ['punkt', 'punkt_tab']:
        directory = nltk_root/'tokenizers'/name
        if directory.exists():
            continue
        archive = nltk_root/'tokenizers'/f'{name}.zip'
        for attempt in range(3):
            download(f'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/{name}.zip', archive)
            try:
                with zipfile.ZipFile(archive) as bundle:
                    if bundle.testzip() is not None:
                        raise zipfile.BadZipFile('Corrupt NLTK archive')
                    for member in bundle.infolist():
                        destination = (archive.parent/member.filename).resolve()
                        if not destination.is_relative_to(archive.parent.resolve()):
                            raise ValueError('Unexpected path in NLTK archive')
                    bundle.extractall(archive.parent)
                break
            except zipfile.BadZipFile:
                archive.unlink(missing_ok=True)
                if attempt == 2:
                    raise
    nltk.data.path.insert(0, str(nltk_root))
    asset_dir = repo/'scripts/data/synthetic/json'
    if ESSAY_TASKS.intersection(tasks):
        target = asset_dir/'PaulGrahamEssays.json'
        if not target.exists():
            result = subprocess.run([sys.executable, 'download_paulgraham_essay.py'], cwd=asset_dir, text=True, capture_output=True, check=True)
            if 'Fail download' in result.stdout:
                raise RuntimeError('Some essay downloads failed; remove the incomplete PaulGrahamEssays.json and retry.\n' + result.stdout)
        if not json.loads(target.read_text()).get('text'):
            raise ValueError('Empty Paul Graham essay corpus')
    if 'qa_1' in tasks:
        download('https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json', asset_dir/'squad.json')
    if 'qa_2' in tasks:
        # This pinned fallback is also listed by NVIDIA's download_qa_dataset.sh.
        download(f'{hf_endpoint()}/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json', asset_dir/'hotpotqa.json')
    return {p.name: sha256(p) for p in sorted(asset_dir.glob('*.json'))}


def verify_paper(output: Path, files: list[dict]) -> None:
    spec = json.loads((ROOT/'experiments/20261107/ruler_manifest.json').read_text())
    expected = {row['path']: row['sha256'] for row in spec['files']}
    bad = [row['path'] for row in files if expected.get(row['path']) != row['sha256']]
    if bad:
        raise ValueError(f'Generated data differs from frozen paper bytes: {bad}. Check tokenizer, corpus hashes and dependency versions. Do not mix these runs with the frozen-paper results.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenizer', required=True, help='Local model/tokenizer directory or a pinned local HF snapshot')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ruler-repo', type=Path, default=ROOT/'third_party/NVIDIA-RULER')
    parser.add_argument('--lengths', nargs='+', type=int, default=[32768, 131072])
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=TASKS)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260823)
    parser.add_argument('--paper-subset', action='store_true', help='Generate 20 records, retain rows 6:20 (zero-based), matching the paper selection rule')
    parser.add_argument('--verify-paper', action='store_true', help='Require generated files to match the frozen paper SHA256 values')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.samples <= 0 or any(n <= 0 or n % 1024 for n in args.lengths):
        parser.error('samples must be positive and lengths must be positive multiples of 1024')
    if args.paper_subset and args.samples != 20:
        parser.error('--paper-subset requires --samples 20')
    if args.verify_paper and not args.paper_subset:
        parser.error('--verify-paper requires --paper-subset')
    # Upstream prepare.py builds a shell command; reject shell metacharacters in paths.
    args.output = args.output.resolve()
    args.ruler_repo = args.ruler_repo.resolve()
    if Path(args.tokenizer).exists():
        args.tokenizer = str(Path(args.tokenizer).resolve())
    for value in [str(args.output), str(args.ruler_repo), args.tokenizer]:
        if not re.fullmatch(r'[A-Za-z0-9_./:@+,-]+', value):
            parser.error('Upstream generator requires paths without spaces or shell metacharacters')
    # Keep the raw 20 outside the evaluation tree; its evaluator uses rglob.
    generated_root = args.output.with_name(args.output.name + '_raw20') if args.paper_subset else args.output
    commands = []
    for length in args.lengths:
        for task in args.tasks:
            label = f'{length // 1024}k'
            command = [sys.executable, str(args.ruler_repo/'scripts/data/prepare.py'),
                       '--save_dir', str(generated_root/label), '--benchmark', 'synthetic',
                       '--task', task, '--subset', 'validation', '--tokenizer_path', args.tokenizer,
                       '--tokenizer_type', 'hf', '--max_seq_length', str(length),
                       '--model_template_type', 'base', '--num_samples', str(args.samples),
                       '--random_seed', str(args.seed)]
            commands.append((label, task, command))
    if args.dry_run:
        for _, _, command in commands:
            print(shlex.join(command))
        return
    prepare_repo(args.ruler_repo)
    assets = prepare_assets(args.ruler_repo, args.tasks)
    versions = {name: importlib.metadata.version(name) for name in ['transformers', 'tokenizers', 'numpy', 'nltk', 'wonderwords', 'scipy']}
    tokenizer_path = Path(args.tokenizer)
    tokenizer_hashes = {p.name: sha256(p) for p in [tokenizer_path/'tokenizer.json', tokenizer_path/'tokenizer_config.json'] if p.is_file()}
    config = {'upstream':UPSTREAM, 'revision':REVISION, 'tokenizer':args.tokenizer, 'tokenizer_hashes':tokenizer_hashes,
              'lengths':args.lengths, 'tasks':args.tasks, 'samples':args.samples, 'seed':args.seed,
              'paper_subset':args.paper_subset, 'assets':assets, 'versions':versions}
    config_file = args.output/'generation_config.json'
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError('Output directory belongs to a different generation configuration; choose another --output')
    if not config_file.exists() and any(args.output.rglob('validation.jsonl')):
        raise ValueError('Existing data has no generation_config.json; use a fresh output directory')
    if not config_file.exists() and args.paper_subset and any(generated_root.rglob('validation.jsonl')):
        raise ValueError('Raw generation directory already contains untracked data; choose another --output')
    write_json(config_file, config)
    env = os.environ.copy()
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    env['NLTK_DATA'] = str(args.ruler_repo/'.cache/nltk_data')
    files = []
    for label, task, command in commands:
        source = generated_root/label/task/'validation.jsonl'
        log = args.output/'logs'/f'{label}_{task}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('a') as stream:
            subprocess.run(command, cwd=args.ruler_repo/'scripts/data', env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        lines = source.read_bytes().splitlines(keepends=True)
        # Upstream prepare.py catches child failures: check output ourselves.
        if len(lines) != args.samples:
            raise RuntimeError(f'{source}: expected {args.samples} records, got {len(lines)}; see {log}')
        for line in lines:
            row = json.loads(line)
            if not isinstance(row.get('input'), str) or not isinstance(row.get('outputs'), list):
                raise ValueError(f'Invalid RULER record in {source}')
        target = args.output/label/task/'validation.jsonl'
        if args.paper_subset:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b''.join(lines[6:20]))
        files.append({'path':str(target.relative_to(args.output)), 'examples':14 if args.paper_subset else args.samples, 'sha256':sha256(target)})
        print(f'Prepared {label}/{task}', flush=True)
    if args.verify_paper:
        verify_paper(args.output, files)
    write_json(args.output/'generation_manifest.json', {'config':config, 'examples':sum(row['examples'] for row in files), 'files':files, 'paper_bytes_verified':args.verify_paper})


if __name__ == '__main__':
    main()
