"""P-key L/8 -> L/64 target-count merge, end-to-end queue on the h20-1-aligned dpskv32 decode.

Same protocol as ``pkey_align_e2e_20260920/run_pkey_align.py`` (single arm,
LongBench-v2 then RULER 32k + 128k, all records; source tree frozen into this
directory; official DSA held-out as the comparison side), with one config
change: ``SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR=64``. The lambda-DP
split still produces L/8 leaves; the merge then reaches exactly L/64 chunks,
the same chunk count as HISA at chunk size 64, so accuracy differences are not
explained by a higher chunk count (the threshold-merge P-key run ended at
~L/9.2).

Owns only the container ``pkey-target64-pkey_t64-20260920``, port 31922.
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
from pathlib import Path

ROOT = Path('/DATA/disk0/qyl')
OUT = Path('/DATA/disk0/qyl/data/pkey_target64_e2e_20260920')
TREE = ROOT / 'code/dpskv32/sglang-hisa'
FROZEN = OUT / 'source'
EVALUATORS = ROOT / 'code/adaptive_accuracy_20260919/evaluators'
HELD = ROOT / 'data/static_group16_e2e_20260913/strict_heldout'
ORIGINAL_INFO = ROOT / 'data/pkey_lbv2_e2e_20260919/original_server.json'
PORT = 31922
MODEL = '/workspace/qyl/models/deepseek-v3.2'
CONTEXT_LEN = 163840
ARM, METRIC = 'pkey_t64', 'key_sse'
MERGE_TARGET_DIVISOR = 64
EXPECTED_METHOD = f'P-key-sync_nonoverlap-target{MERGE_TARGET_DIVISOR}'
NAME = f'pkey-target64-{ARM}-20260920'
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


def sh(args, check=True, capture=True):
    r = subprocess.run(args, text=True, capture_output=capture)
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


def health(port, path='/health_generate', timeout=5):
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def wait_ready(name, port, limit=1500):
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if not running(name):
            raise RuntimeError(name + ' exited before becoming ready')
        if health(port):
            event('SERVER_READY', name=name, port=port)
            return
        time.sleep(5)
    raise RuntimeError(name + ' readiness timeout')


def freeze_source():
    FROZEN.mkdir(parents=True, exist_ok=True)
    sh(['rsync', '-a', '--delete', '--exclude', '__pycache__', str(TREE / 'python') + '/', str(FROZEN / 'python') + '/'])
    pkg = FROZEN / 'python/sglang/srt/layers/attention/nsa/adaptive_hisa'
    shas = {}
    for p in sorted(list(pkg.rglob('*.py')) + list(pkg.rglob('*.cu'))
                    + [FROZEN / 'python/sglang/srt/layers/attention/nsa/nsa_indexer.py']):
        shas[str(p.relative_to(FROZEN))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT / 'source.sha256.json').write_text(json.dumps(shas, indent=2))
    prov = {
        'tree': str(TREE),
        'git_head': sh(['git', '-C', str(TREE), 'rev-parse', 'HEAD'], check=False),
        'git_status': sh(['git', '-C', str(TREE), 'status', '--short'], check=False),
        'frozen_utc': now(),
    }
    (OUT / 'source_provenance.json').write_text(json.dumps(prov, indent=2))
    (OUT / 'evaluators').mkdir(exist_ok=True)
    for name in ('evaluate_longbench_v2_e2e.py', 'collect_longbench_v2.py',
                 'evaluate_ruler_e2e.py', 'collect_ruler.py'):
        (OUT / 'evaluators' / name).write_bytes((EVALUATORS / name).read_bytes())
    event('SOURCE_FROZEN', files=len(shas))


def start_server(info):
    global ACTIVE
    if exists(NAME):
        if running(NAME):
            event('SERVER_RESUMED', name=NAME)
            ACTIVE = NAME
            wait_ready(NAME, PORT)
            return NAME
        sh(['docker', 'rm', NAME])
    env = [e for e in info['env'] if e.startswith('SGLANG_')
           and not e.startswith(('SGLANG_BUILD_', 'SGLANG_IMAGE_TAG=', 'SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC=',
                                 'SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_'))]
    env += [f'PYTHONPATH={cpath(FROZEN)}/python', f'SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC={METRIC}',
            f'SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR={MERGE_TARGET_DIVISOR}']
    cmd = list(info['cmd'])
    cmd[cmd.index('--port') + 1] = str(PORT)
    cmd[cmd.index('--context-length') + 1] = str(CONTEXT_LEN)
    argv = ['docker', 'run', '-d', '--name', NAME, '--label', 'qyl.pkey_target64=20260920',
            '--gpus', 'all', '--ipc', 'host', '--network', 'host']
    for b in info['binds']:
        argv += ['-v', b]
    for e in env:
        argv += ['-e', e]
    argv += ['--entrypoint', info['entrypoint'][0] if info['entrypoint'] else 'python3', info['image'], *cmd]
    (OUT / ARM / 'launch.json').write_text(json.dumps(argv, indent=2))
    event('SERVER_START', arm=ARM, name=NAME, metric=METRIC, merge_target_divisor=MERGE_TARGET_DIVISOR, context_len=CONTEXT_LEN)
    sh(argv)
    ACTIVE = NAME
    wait_ready(NAME, PORT)
    for _ in range(60):
        logs = dlogs(NAME)
        m = re.search(r'gpu kernels warm in [\d.]+s metric=(\w+) method=(\S+)', logs)
        if m:
            event('METRIC_CONFIRMED', arm=ARM, metric=m.group(1), method=m.group(2))
            if m.group(1) != METRIC:
                raise RuntimeError(f'{NAME} runs metric {m.group(1)}, expected {METRIC}')
            if m.group(2) != EXPECTED_METHOD:
                raise RuntimeError(f'{NAME} runs method {m.group(2)}, expected {EXPECTED_METHOD}')
            break
        time.sleep(5)
    else:
        raise RuntimeError(f'{NAME}: no kernel-warm/metric line in the server log')
    return NAME


def smoke():
    """One 20K-token request: every layer must build, decode must bind with fallback_layers=0."""
    ids = [100 + (i * 7919) % 30000 for i in range(20000)]
    payload = json.dumps({'input_ids': ids, 'sampling_params': {'temperature': 0, 'max_new_tokens': 16, 'ignore_eos': True}}).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/generate', data=payload, headers={'Content-Type': 'application/json'})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    logs = dlogs(NAME, '120s')
    built = re.findall(r'forward summary sync_built=(\d+) gpu_built=(\d+)', logs)
    bound = re.findall(r'decode bound req=\d+ layers=(\d+) eligible=(\d+) candidate_tokens=\d+ sink=(\d+) tail=(\d+) decode_chunk=(\d+) fallback_layers=(\d+)', logs)
    skipped = re.findall(r'skipped prefill partition layer=\d+ req=\d+ reason=(\w+)', logs)
    tb = logs.count('Traceback') + len(re.findall(r'adaptive-hisa .*failed', logs))
    rec = dict(arm=ARM, wall_s=round(time.monotonic() - t0, 2), completion_tokens=out.get('meta_info', {}).get('completion_tokens'),
               forward_summaries=built[-8:], decode_bound=bound[-8:], skipped=sorted(set(skipped)), tracebacks=tb)
    (OUT / ARM / 'smoke.json').write_text(json.dumps(rec, indent=2))
    event('SMOKE', **rec)
    if tb or not built or any(int(b[1]) + int(b[0]) < 60 for b in built[-8:]) or not bound:
        raise RuntimeError(f'{ARM} smoke failed: {rec}')
    last = bound[-1]
    if not (int(last[0]) == int(last[1]) >= 60 and last[2:] == ('64', '256', '8', '0')):
        raise RuntimeError(f'{ARM} decode binding is not the aligned configuration: {last}')


def evaluate_longbench():
    dst = OUT / ARM
    cmd = ['docker', 'run', '--rm', '--network', 'host', '-v', str(ROOT) + ':/workspace/qyl', 'qyl/sglang-hisa:eval',
           'python', cpath(OUT / 'evaluators/evaluate_longbench_v2_e2e.py'),
           '--model', MODEL, '--server', f'http://127.0.0.1:{PORT}',
           '--data', cpath(HELD / 'longbench_heldout.json'),
           '--output', cpath(dst / 'longbench_predictions.jsonl'),
           '--summary', cpath(dst / 'longbench_summary.json'),
           # same prompt budget as the official DSA baseline (context_len 163840)
           '--max-context-tokens', '131072', '--max-new-tokens', '128', '--concurrency', '1', '--resume']
    (dst / 'longbench.command.json').write_text(json.dumps(cmd, indent=2))
    event('EVAL_START', arm=ARM, bench='longbench_v2')
    with (dst / 'longbench.log').open('a') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    summary = json.loads((dst / 'longbench_summary.json').read_text())
    event('EVAL_DONE', arm=ARM, bench='longbench_v2', examples=summary.get('examples'), successful=summary.get('successful'),
          accuracy=summary.get('accuracy'), by_length=summary.get('by_length'), by_difficulty=summary.get('by_difficulty'))
    if summary.get('examples') != summary.get('successful'):
        raise RuntimeError(f'{ARM}: longbench request errors: {summary}')


def evaluate_ruler():
    dst = OUT / ARM
    cmd = ['docker', 'run', '--rm', '--network', 'host', '-v', str(ROOT) + ':/workspace/qyl', 'qyl/sglang-hisa:eval',
           'python', cpath(OUT / 'evaluators/evaluate_ruler_e2e.py'),
           '--data-root', cpath(HELD / 'ruler'),
           '--model', MODEL, '--server', f'http://127.0.0.1:{PORT}',
           '--output', cpath(dst / 'ruler_predictions.jsonl'),
           '--summary', cpath(dst / 'ruler_summary.json'),
           '--lengths', '32k', '128k', '--all-records', '--resume']
    (dst / 'ruler.command.json').write_text(json.dumps(cmd, indent=2))
    event('EVAL_START', arm=ARM, bench='ruler')
    with (dst / 'ruler.log').open('a') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    summary = json.loads((dst / 'ruler_summary.json').read_text())
    event('EVAL_DONE', arm=ARM, bench='ruler', examples=summary.get('examples'), successful=summary.get('successful'),
          accuracy=summary.get('accuracy'), by_length=summary.get('by_length'), by_task=summary.get('by_task'))
    if summary.get('examples') != summary.get('successful'):
        raise RuntimeError(f'{ARM}: ruler request errors: {summary}')


def stop_server():
    global ACTIVE
    logs = dlogs(NAME)
    (OUT / ARM / 'server.log').write_text(logs)
    audit = {
        'tracebacks': logs.count('Traceback'),
        'adaptive_failed': len(re.findall(r'adaptive-hisa .*failed', logs)),
        'forward_summaries': len(re.findall(r'forward summary', logs)),
        'decode_bound': len(re.findall(r'decode bound', logs)),
        'skip_reasons': dict(sorted(
            ((k, v) for k, v in
             __import__('collections').Counter(re.findall(r'skipped prefill partition layer=\d+ req=\d+ reason=(\w+)', logs)).items()))),
        'stale_dropped': logs.count('dropped stale partition'),
    }
    (OUT / ARM / 'server_audit.json').write_text(json.dumps(audit, indent=2))
    event('SERVER_AUDIT', arm=ARM, **audit)
    sh(['docker', 'stop', '--timeout', '60', NAME], check=False)
    sh(['docker', 'rm', NAME], check=False)
    ACTIVE = None


def rows(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[str(r['source_id'])] = r  # last row per id wins (resume)
    return out


def compare():
    report = {'updated_utc': now(), 'benches': {}}
    for bench, fname in (('longbench_v2', 'longbench_predictions.jsonl'), ('ruler', 'ruler_predictions.jsonl')):
        arms = {ARM: rows(OUT / ARM / fname), 'official_dsa_20260913': rows(HELD / 'official_dsa' / fname)}
        a, b = ARM, 'official_dsa_20260913'
        common = [k for k in sorted(arms[a].keys() & arms[b].keys())
                  if arms[a][k].get('status') == arms[b][k].get('status') == 'ok']
        entry = {'n_rows': {k: len(v) for k, v in arms.items()}, 'matched': len(common)}
        if common:
            acc = lambda side, keys: sum(arms[side][k]['score'] for k in keys) / len(keys)
            entry.update(acc_pkey=acc(a, common), acc_official=acc(b, common),
                         delta_pp=100 * (acc(a, common) - acc(b, common)),
                         pkey_wins=sum(arms[a][k]['score'] > arms[b][k]['score'] for k in common),
                         official_wins=sum(arms[a][k]['score'] < arms[b][k]['score'] for k in common))
            for field in ('length', 'difficulty', 'task'):
                if field not in arms[a][common[0]]:
                    continue
                entry['by_' + field] = {}
                for value in sorted({str(arms[a][k][field]) for k in common}):
                    keys = [k for k in common if str(arms[a][k][field]) == value]
                    entry['by_' + field][value] = {'n': len(keys), 'acc_pkey': acc(a, keys), 'acc_official': acc(b, keys)}
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
    info = json.loads(ORIGINAL_INFO.read_text())
    if not (FROZEN / 'python').exists() or '--refreeze' in sys.argv:
        freeze_source()
    try:
        status(arm=ARM, state='starting_server')
        start_server(info)
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
