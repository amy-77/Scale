#!/usr/bin/env python3
"""Launch one DeepSeek-V3.2 SCALE/HISA/DSA arm and run a resumable evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'SCALE_relaxed/runner'))
from run_attribution_suite import hisa_env, scale_env, stop


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['scale', 'hisa', 'dsa'], default='scale')
    parser.add_argument('--dataset', choices=['ruler', 'longbench-v2'], required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hisa-source', type=Path, default=REPO/'HISA_current')
    parser.add_argument('--dense-final', type=int, choices=[0, 1], default=0)
    parser.add_argument('--hisa-block', type=int, choices=[8, 16, 32, 64, 128], default=64)
    parser.add_argument('--scale-average', type=int, choices=[8, 16, 32, 64, 128], default=64)
    parser.add_argument('--candidate-tokens', type=int, default=8192)
    parser.add_argument('--tp', type=int, default=8)
    parser.add_argument('--port', type=int, default=31946)
    parser.add_argument('--mem-fraction', type=float, default=0.82)
    parser.add_argument('--context-length', type=int, default=163840)
    parser.add_argument('--max-input-tokens', type=int, default=131072, help='LongBench head/tail input cap, preserving the question/options')
    parser.add_argument('--max-new-tokens', type=int, default=128, help='LongBench answer cap; RULER uses per-task limits')
    parser.add_argument('--lengths', nargs='+', choices=['32k', '128k'], default=['32k', '128k'])
    parser.add_argument('--max-records', type=int, help='Explicit smoke-test subset; omitted for full input files')
    parser.add_argument('--server', help='Evaluate an existing server without launching/stopping it')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.method == 'dsa' and args.dense_final:
        parser.error('dense-final is a hierarchical-indexer ablation; use --dense-final 0 for DSA')
    if args.tp <= 0 or not 0 < args.mem_fraction < 1:
        parser.error('tp must be positive and mem-fraction must be between zero and one')
    if args.candidate_tokens < 2048:
        parser.error('Candidate budget must accommodate final Top-2048')
    if args.method == 'hisa' and (args.candidate_tokens % args.hisa_block or args.candidate_tokens // args.hisa_block > 2048):
        parser.error('HISA requires an integral block Top-K no greater than 2048')
    if args.dataset == 'longbench-v2' and args.max_input_tokens + args.max_new_tokens > args.context_length:
        parser.error('LongBench input and generation limits exceed server context length')
    for key in ['model', 'data', 'output', 'hisa_source']:
        setattr(args, key, getattr(args, key).resolve())
    source = REPO/'SCALE_relaxed' if args.method == 'scale' else args.hisa_source
    env_updates = scale_env(bool(args.dense_final)) if args.method == 'scale' else hisa_env(bool(args.dense_final))
    if args.method == 'scale':
        env_updates['SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR'] = str(args.scale_average)
        env_updates['SGLANG_NSA_ADAPTIVE_HISA_CANDIDATE_TOKENS'] = str(args.candidate_tokens)
    override = {'use_hisa':False} if args.method == 'dsa' else None
    if args.method == 'hisa':
        override = {'use_hisa':True, 'hisa_k_block_size':args.hisa_block, 'hisa_block_topk':args.candidate_tokens//args.hisa_block}
    server_command = [sys.executable, '-m', 'sglang.launch_server', '--model-path', str(args.model),
                      '--served-model-name', 'deepseek-v3.2', '--tp-size', str(args.tp), '--host', '127.0.0.1',
                      '--port', str(args.port), '--context-length', str(args.context_length), '--trust-remote-code',
                      '--reasoning-parser', 'deepseek-v3', '--chunked-prefill-size', '8192',
                      '--mem-fraction-static', str(args.mem_fraction), '--max-running-requests', '1',
                      '--disable-radix-cache', '--random-seed', '20260921', '--watchdog-timeout', '900', '--cuda-graph-bs', '1']
    if override is not None:
        server_command.extend(['--json-model-override-args', json.dumps(override)])
    url = (args.server or f'http://127.0.0.1:{args.port}').rstrip('/')
    evaluator = REPO/'SCALE_relaxed/runner/evaluators'/('evaluate_ruler_e2e.py' if args.dataset == 'ruler' else 'evaluate_longbench_v2_e2e.py')
    eval_command = [sys.executable, str(evaluator), '--model', str(args.model), '--server', url,
                    '--output', str(args.output/'predictions.jsonl'), '--summary', str(args.output/'summary.json')]
    if args.dataset == 'ruler':
        eval_command.extend(['--data-root', str(args.data), '--all-records', '--lengths', *args.lengths])
    else:
        eval_command.extend(['--data', str(args.data), '--max-context-tokens', str(args.max_input_tokens), '--max-new-tokens', str(args.max_new_tokens), '--concurrency', '1'])
    if args.resume:
        eval_command.append('--resume')
    if args.max_records is not None:
        if args.max_records <= 0:
            parser.error('max-records must be positive')
        eval_command.extend(['--max-records', str(args.max_records)])
    print('Server:', shlex.join(server_command) if not args.server else url)
    print('Environment:', json.dumps(env_updates, sort_keys=True))
    print('Evaluation:', shlex.join(eval_command))
    if args.dry_run:
        return
    if not args.model.is_dir() or not args.data.exists():
        parser.error('Model/tokenizer directory and dataset must exist')
    if not args.server and not (source/'python/sglang').is_dir():
        parser.error('Serving source is missing; prepare the HISA baseline if needed')
    # This wrapper adds metadata safeguards without changing the evaluator.
    digest = hashlib.sha256()
    data_files = sorted(args.data.rglob('validation.jsonl')) if args.data.is_dir() else [args.data]
    for path in data_files:
        digest.update(str(path.relative_to(args.data) if args.data.is_dir() else path.name).encode())
        digest.update(b'\0')
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
    metadata = {'method':args.method, 'dataset':args.dataset, 'data':str(args.data), 'data_sha256':digest.hexdigest(), 'model':str(args.model),
                'environment':env_updates, 'override':override, 'max_records':args.max_records,
                'lengths':args.lengths, 'max_input_tokens':args.max_input_tokens,
                'max_new_tokens':args.max_new_tokens, 'context_length':args.context_length,
                'tp':args.tp, 'mem_fraction':args.mem_fraction, 'external_server':args.server}
    previous = args.output/'run_config.json'
    if previous.exists():
        if not args.resume or json.loads(previous.read_text()) != metadata:
            raise ValueError('Output already belongs to a run; use --resume with the same configuration or a new output directory')
    elif (args.output/'predictions.jsonl').exists():
        raise ValueError('Cannot append to predictions with no matching run_config.json')
    args.output.mkdir(parents=True, exist_ok=True)
    previous.write_text(json.dumps(metadata, indent=2)+'\n')
    process = None
    server_log = None
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    try:
        if not args.server:
            import socket
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1', args.port)) == 0:
                    raise RuntimeError(f'Port {args.port} is already in use')
            env = os.environ.copy()
            env.update(env_updates)
            env['PYTHONPATH'] = str(source/'python')
            server_log = (args.output/'server.log').open('a')
            process = subprocess.Popen(server_command, cwd=source, env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic()+3600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Server exited {process.returncode}; inspect server.log')
                try:
                    with urllib.request.urlopen(url+'/health', timeout=3) as response:
                        if response.status == 200:
                            break
                except Exception:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('Server startup exceeded one hour')
                time.sleep(5)
        try:
            with urllib.request.urlopen(url+'/get_server_info', timeout=30) as response:
                info = json.load(response)
            (args.output/'server_info.json').write_text(json.dumps(info, indent=2)+'\n')
        except Exception as error:
            print(f'Server info unavailable: {error}', file=sys.stderr)
        subprocess.run(eval_command, check=True)
        summary = json.loads((args.output/'summary.json').read_text())
        if not summary.get('examples') or summary.get('successful') != summary.get('examples'):
            raise RuntimeError('Evaluation incomplete; rerun with --resume')
    finally:
        if process is not None:
            stop(process)
        if server_log is not None:
            server_log.close()


if __name__ == '__main__':
    main()
