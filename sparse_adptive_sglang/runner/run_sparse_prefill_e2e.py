"""Adaptive-HISA sparse prefill (adaptive_0921_h202) end-to-end queue on h20-9-57.

Same protocol/env as ``run_speedopt_e2e_h957.py`` (P-key, L/8 split -> L/64
target merge, adaptive decode) plus ``SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL=1``:
every prefill chunk partitions its sealed prefix and the next chunk's indexer
takes Top-2048 from summary-selected leaves + the causal local window instead
of the dense DSA logits. Freeze source, warm-up method check, 20K smoke with
sparse-prefill evidence, LongBench-v2 401, RULER 32k/128k 364, comparison
against official DSA and the decode-only P-key run. Owns only the container
named below.

Resume: rerun the script; evaluators append/skip completed rows.
"""
import datetime
import fcntl
import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path('/DATA/disk0/qyl')
PACKAGE = ROOT / 'code/adaptive_0921_h202'
OUT = ROOT / 'data/adaptive_0921_h202_sparse_prefill_e2e_20260921'
FROZEN = OUT / 'source'
CACHE = ROOT / 'cache/adaptive_0921_h202'
EVALUATORS_SRC = ROOT / 'code/dpskv32'
HELD = ROOT / 'data/static_group16_e2e_20260913/strict_heldout'
OFFICIAL_HIST = HELD / 'official_dsa'
PKEY_DECODE_ONLY = ROOT / 'data/pkey_align_e2e_20260920/pkey'
IMAGE = 'qyl/sglang-hisa:eval'
MODEL = '/workspace/qyl/models/deepseek-v3.2'
PORT = 31923
CONTEXT_LEN = 163840
ARM = 'sparse_prefill'
NAME = 'adaptive-0921-h202-sparse-prefill-e2e'
EXPECTED_METRIC = 'key_sse'
EXPECTED_METHOD = 'P-key-sync_nonoverlap-target64'
EXPECTED_DECODE_CHUNK = '64'
ACTIVE = None


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def cpath(path):
    return str(path).replace(str(ROOT), '/workspace/qyl', 1)


def event(event_name, **data):
    rec = dict(event=event_name, time=now(), **data)
    print(json.dumps(rec, ensure_ascii=False), flush=True)
    with (OUT / 'events.jsonl').open('a') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def status(**values):
    values['updated_utc'] = now()
    (OUT / 'progress.json').write_text(json.dumps(values, indent=2))


def sh(args, check=True):
    r = subprocess.run(args, text=True, capture_output=True)
    if check and r.returncode:
        raise RuntimeError(f'{args} -> {r.returncode}: {r.stderr.strip()[-2000:]}')
    return (r.stdout or '').strip()


def dlogs(name, since=None):
    args = ['docker', 'logs'] + (['--since', since] if since else []) + [name]
    r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace')
    return r.stdout or ''


def running(name):
    r = subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', name], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == 'true'


def exists(name):
    return subprocess.run(['docker', 'inspect', name], capture_output=True).returncode == 0


def gpu_pids():
    out = sh(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])
    return sorted({l.strip() for l in out.splitlines() if l.strip()})


def health(path='/health_generate', timeout=5):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}{path}', timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def wait_ready(limit=1800):
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if not running(NAME):
            raise RuntimeError(NAME + ' exited before becoming ready:\n' + dlogs(NAME)[-4000:])
        if health():
            event('SERVER_READY', name=NAME, port=PORT)
            return
        time.sleep(5)
    raise RuntimeError(NAME + ' readiness timeout')


def freeze_source():
    FROZEN.mkdir(parents=True, exist_ok=True)
    sh(['rsync', '-a', '--delete', '--exclude', '__pycache__', str(PACKAGE / 'python') + '/', str(FROZEN / 'python') + '/'])
    pkg = FROZEN / 'python/sglang/srt/layers/attention/nsa/adaptive_hisa'
    shas = {}
    for p in sorted(list(pkg.rglob('*.py')) + list(pkg.rglob('*.cu'))
                    + [FROZEN / 'python/sglang/srt/layers/attention/nsa/nsa_indexer.py']):
        shas[str(p.relative_to(FROZEN))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / 'source.sha256.json').write_text(json.dumps(shas, indent=2))
    prov = {
        'package': str(PACKAGE),
        'package_git_head': (PACKAGE / 'git_head.txt').read_text().strip() if (PACKAGE / 'git_head.txt').exists() else None,
        'frozen_utc': now(),
    }
    (OUT / 'source_provenance.json').write_text(json.dumps(prov, indent=2))
    (OUT / 'evaluators').mkdir(exist_ok=True)
    for name in ('evaluate_longbench_v2_e2e.py', 'collect_longbench_v2.py',
                 'evaluate_ruler_e2e.py', 'collect_ruler.py'):
        (OUT / 'evaluators' / name).write_bytes((EVALUATORS_SRC / name).read_bytes())
    event('SOURCE_FROZEN', files=len(shas))


def server_env():
    return {
        'PYTHONPATH': f'{cpath(FROZEN)}/python',
        'PYTHONUNBUFFERED': '1',
        'SGLANG_NSA_FUSE_TOPK': '0',
        'SGLANG_NSA_PER_HEAD_INDEX': '0',
        'SGLANG_NSA_EXPERIMENTAL_PREFILL_PER_HEAD': '0',
        'SGLANG_JIT_DEEPGEMM_PRECOMPILE': '0',
        'SGLANG_NSA_ADAPTIVE_HISA_MODE': 'adaptive_decode',
        'SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC': EXPECTED_METRIC,
        'SGLANG_NSA_ADAPTIVE_HISA_SPLIT_BACKEND': 'gpu',
        'SGLANG_NSA_ADAPTIVE_HISA_BUILD_SUMMARIES': '1',
        'SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY': 'sync_nonoverlap',
        'SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR': '64',
        'SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_ROUNDS': '8',
        'SGLANG_NSA_ADAPTIVE_HISA_MAX_MERGE_LEN': '0',
        'SGLANG_NSA_ADAPTIVE_HISA_GRAPH_BUILD': '1',
        'SGLANG_NSA_ADAPTIVE_HISA_GPU_STREAM': 'side',
        'SGLANG_NSA_ADAPTIVE_HISA_FALLBACK_LAYERS': '0',
        'SGLANG_NSA_ADAPTIVE_HISA_SINK_TOKENS': '64',
        'SGLANG_NSA_ADAPTIVE_HISA_TAIL_TOKENS': '256',
        'SGLANG_NSA_ADAPTIVE_HISA_CANDIDATE_TOKENS': '8192',
        'SGLANG_NSA_ADAPTIVE_HISA_DECODE_CHUNK': EXPECTED_DECODE_CHUNK,
        'SGLANG_NSA_ADAPTIVE_HISA_SKIP_CLEAN': '1',
        'SGLANG_NSA_ADAPTIVE_HISA_FUSE_PAGE_TABLE': '1',
        'SGLANG_NSA_ADAPTIVE_HISA_RAW_FP8_BUILDER': '1',
        'SGLANG_NSA_ADAPTIVE_HISA_DECODE_TIMING': '0',
        'SGLANG_NSA_ADAPTIVE_HISA_SELECTOR_PROFILE': '0',
        'SGLANG_NSA_ADAPTIVE_HISA_PARTITION_VALIDATE': '1',
        'SGLANG_NSA_ADAPTIVE_HISA_FORWARD_TIMING': '0',
        'SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL': '1',
        'SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_ROWS': '2048',
    }


def server_cmd():
    return ['python', '-m', 'sglang.launch_server',
            '--model-path', MODEL, '--served-model-name', 'deepseek-v3.2',
            '--tp-size', '8', '--host', '127.0.0.1', '--port', str(PORT),
            '--context-length', str(CONTEXT_LEN),
            '--trust-remote-code', '--reasoning-parser', 'deepseek-v3',
            '--mem-fraction-static', '0.82', '--max-running-requests', '1',
            '--disable-cuda-graph', '--disable-radix-cache',
            '--random-seed', '20260921', '--watchdog-timeout', '900']


def start_server():
    global ACTIVE
    if exists(NAME):
        if running(NAME):
            event('SERVER_RESUMED', name=NAME)
            ACTIVE = NAME
            wait_ready()
            return
        sh(['docker', 'rm', NAME])
    busy = gpu_pids()
    if busy:
        raise RuntimeError(f'GPU processes are active; refusing to compete: {busy}')
    if health('/health'):
        raise RuntimeError(f'port {PORT} already serving')
    CACHE.mkdir(parents=True, exist_ok=True)
    argv = ['docker', 'run', '-d', '--name', NAME, '--label', 'qyl.adaptive_0921_h202=sparse_prefill_e2e',
            '--gpus', 'all', '--ipc', 'host', '--network', 'host',
            '-v', f'{ROOT}:/workspace/qyl', '-v', f'{CACHE}:/root/.cache']
    for k, v in server_env().items():
        argv += ['-e', f'{k}={v}']
    argv += [IMAGE, *server_cmd()]
    (OUT / ARM / 'launch.json').write_text(json.dumps(argv, indent=2))
    event('SERVER_START', arm=ARM, name=NAME)
    sh(argv)
    ACTIVE = NAME
    wait_ready()
    logs = dlogs(NAME)
    if 'Traceback' in logs:
        raise RuntimeError('traceback during server start-up:\n' + logs[-4000:])


def confirm_method(logs):
    # The server warm-up request is short and takes the k-only indexer path, so the
    # adaptive-hisa kernel warm-up (and its metric/method line) only appears on the
    # first real long prefill, i.e. after the smoke request.
    m = re.search(r'gpu kernels warm in [\d.]+s metric=(\w+) method=(\S+)', logs)
    if not m:
        raise RuntimeError('no kernel-warm/metric line in the server log after smoke')
    event('METRIC_CONFIRMED', metric=m.group(1), method=m.group(2))
    if m.group(1) != EXPECTED_METRIC or m.group(2) != EXPECTED_METHOD:
        raise RuntimeError(f'server runs {m.group(1)}/{m.group(2)}, expected {EXPECTED_METRIC}/{EXPECTED_METHOD}')


def smoke():
    ids = [100 + (i * 7919) % 30000 for i in range(20000)]
    payload = json.dumps({'input_ids': ids, 'sampling_params': {'temperature': 0, 'max_new_tokens': 16, 'ignore_eos': True}}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/generate', data=payload, headers={'Content-Type': 'application/json'})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    time.sleep(2)
    confirm_method(dlogs(NAME))
    logs = dlogs(NAME, '300s')
    built = re.findall(r'forward summary sync_built=(\d+) gpu_built=(\d+)', logs)
    bound = re.findall(r'decode bound req=\d+ layers=(\d+) eligible=(\d+) candidate_tokens=\d+ sink=(\d+) tail=(\d+) decode_chunk=(\d+) fallback_layers=(\d+)', logs)
    skipped = re.findall(r'skipped prefill partition layer=\d+ req=\d+ reason=(\w+)', logs)
    sparse = re.findall(r'sparse prefill layer=(\d+) rows=(\d+) n_complete=(\d+) chunk_start=(\d+) seq_len=(\d+)', logs)
    tb = logs.count('Traceback') + len(re.findall(r'adaptive-hisa .*failed', logs))
    rec = dict(wall_s=round(time.monotonic() - t0, 2), completion_tokens=out.get('meta_info', {}).get('completion_tokens'),
               forward_summaries=built[-8:], decode_bound=bound[-8:], skipped=sorted(set(skipped)), tracebacks=tb,
               sparse_prefill_layers=len({m[0] for m in sparse}), sparse_prefill_first=sparse[:2])
    (OUT / ARM / 'smoke.json').write_text(json.dumps(rec, indent=2))
    event('SMOKE', **rec)
    if tb or not built or not bound:
        raise RuntimeError(f'smoke lacks required evidence: {rec}')
    # 20K prompt = chunks [0,8192) [8192,16384) [16384,20000): chunk 2 must use the
    # sparse path (n_complete=16384 >= 8192) on every indexer layer.
    if rec['sparse_prefill_layers'] < 60:
        raise RuntimeError(f'smoke shows sparse prefill on only {rec["sparse_prefill_layers"]} layers: {rec}')
    if any(int(s) + int(g) < 60 for s, g in built[-8:]):
        raise RuntimeError(f'smoke built fewer than 60 layers: {rec}')
    last = bound[-1]
    if not (int(last[0]) == int(last[1]) >= 60 and last[2:] == ('64', '256', EXPECTED_DECODE_CHUNK, '0')):
        raise RuntimeError(f'smoke decode binding mismatch: {last}')


def run_eval(cmd, log_path):
    with log_path.open('a') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)


def evaluate_longbench():
    dst = OUT / ARM
    cmd = ['docker', 'run', '--rm', '--network', 'host', '-v', f'{ROOT}:/workspace/qyl', IMAGE,
           'python', cpath(OUT / 'evaluators/evaluate_longbench_v2_e2e.py'),
           '--model', MODEL, '--server', f'http://127.0.0.1:{PORT}',
           '--data', cpath(HELD / 'longbench_heldout.json'),
           '--output', cpath(dst / 'longbench_predictions.jsonl'),
           '--summary', cpath(dst / 'longbench_summary.json'),
           '--max-context-tokens', '131072', '--max-new-tokens', '128', '--concurrency', '1', '--resume']
    event('EVAL_START', bench='longbench_v2')
    run_eval(cmd, dst / 'longbench.log')
    s = json.loads((dst / 'longbench_summary.json').read_text())
    event('EVAL_DONE', bench='longbench_v2', examples=s.get('examples'), successful=s.get('successful'),
          accuracy=s.get('accuracy'), by_length=s.get('by_length'), by_difficulty=s.get('by_difficulty'))
    if s.get('examples') != s.get('successful'):
        raise RuntimeError(f'longbench request errors: {s}')


def evaluate_ruler():
    dst = OUT / ARM
    cmd = ['docker', 'run', '--rm', '--network', 'host', '-v', f'{ROOT}:/workspace/qyl', IMAGE,
           'python', cpath(OUT / 'evaluators/evaluate_ruler_e2e.py'),
           '--data-root', cpath(HELD / 'ruler'),
           '--model', MODEL, '--server', f'http://127.0.0.1:{PORT}',
           '--output', cpath(dst / 'ruler_predictions.jsonl'),
           '--summary', cpath(dst / 'ruler_summary.json'),
           '--lengths', '32k', '128k', '--all-records', '--resume']
    event('EVAL_START', bench='ruler')
    run_eval(cmd, dst / 'ruler.log')
    s = json.loads((dst / 'ruler_summary.json').read_text())
    event('EVAL_DONE', bench='ruler', examples=s.get('examples'), successful=s.get('successful'),
          accuracy=s.get('accuracy'), by_length=s.get('by_length'), by_task=s.get('by_task'))
    if s.get('examples') != s.get('successful'):
        raise RuntimeError(f'ruler request errors: {s}')


def stop_server():
    global ACTIVE
    logs = dlogs(NAME)
    (OUT / ARM / 'server.log').write_text(logs)
    audit = {
        'tracebacks': logs.count('Traceback'),
        'adaptive_failed': len(re.findall(r'adaptive-hisa .*failed', logs)),
        'forward_summaries': len(re.findall(r'forward summary', logs)),
        'sparse_prefill_lines': len(re.findall(r'sparse prefill layer=', logs)),
        'sparse_prefill_dense_fallback_notes': len(re.findall(r'sparse prefill needs unfused', logs)),
        'decode_bound': len(re.findall(r'decode bound', logs)),
        'skip_reasons': dict(sorted(Counter(re.findall(r'skipped prefill partition layer=\d+ req=\d+ reason=(\w+)', logs)).items())),
        'stale_dropped': logs.count('dropped stale partition'),
    }
    (OUT / ARM / 'server_audit.json').write_text(json.dumps(audit, indent=2))
    event('SERVER_AUDIT', **audit)
    sh(['docker', 'stop', '--timeout', '60', NAME], check=False)
    sh(['docker', 'rm', NAME], check=False)
    ACTIVE = None


def rows(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[str(r['source_id'])] = r
    return out


def compare():
    report = {'updated_utc': now(), 'benches': {}}
    for bench, fname in (('longbench_v2', 'longbench_predictions.jsonl'), ('ruler', 'ruler_predictions.jsonl')):
        arm = rows(OUT / ARM / fname)
        entry = {'rows_sparse_prefill': len(arm), 'baselines': {}}
        for label, path in (('official_dsa_hist_20260913', OFFICIAL_HIST / fname),
                            ('pkey_decode_only_20260920', PKEY_DECODE_ONLY / fname)):
            base = rows(path)
            common = [k for k in sorted(arm.keys() & base.keys())
                      if arm[k].get('status') == base[k].get('status') == 'ok']
            e = {'rows_baseline': len(base), 'matched': len(common)}
            if common:
                acc = lambda side, keys: sum(float(side[k]['score']) for k in keys) / len(keys)
                e.update(acc_sparse_prefill=acc(arm, common), acc_baseline=acc(base, common),
                         delta_pp=100 * (acc(arm, common) - acc(base, common)),
                         sparse_prefill_wins=sum(float(arm[k]['score']) > float(base[k]['score']) for k in common),
                         baseline_wins=sum(float(arm[k]['score']) < float(base[k]['score']) for k in common))
                for field in ('length', 'difficulty', 'task'):
                    if field not in arm[common[0]]:
                        continue
                    e['by_' + field] = {}
                    for value in sorted({str(arm[k][field]) for k in common}):
                        keys = [k for k in common if str(arm[k][field]) == value]
                        e['by_' + field][value] = {'n': len(keys), 'acc_sparse_prefill': acc(arm, keys), 'acc_baseline': acc(base, keys)}
            entry['baselines'][label] = e
        report['benches'][bench] = entry
    (OUT / 'comparison.json').write_text(json.dumps(report, indent=2))
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ARM).mkdir(exist_ok=True)
    lock = (OUT / 'runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if '--compare-only' in sys.argv:
        print(json.dumps(compare(), indent=2))
        return
    if not (FROZEN / 'python').exists() or '--refreeze' in sys.argv:
        freeze_source()
    try:
        status(arm=ARM, state='starting_server')
        start_server()
        status(arm=ARM, state='smoke')
        smoke()
        status(arm=ARM, state='evaluating_longbench')
        evaluate_longbench()
        compare()
        status(arm=ARM, state='evaluating_ruler')
        evaluate_ruler()
        compare()
        stop_server()
        status(arm=ARM, state='all_completed')
        event('FINISHED', status='passed')
    except Exception as e:
        status(state='failed', arm=ARM, error=repr(e))
        event('FAILED', arm=ARM, error=repr(e))
        if ACTIVE:
            try:
                stop_server()
            except Exception:
                pass
        raise


if __name__ == '__main__':
    main()
