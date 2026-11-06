#!/usr/bin/env python3
"""Resumable HISA-16/8 RULER quality queue on h20-9-5.

Shares GPUs 6/7 with two pre-existing small services; timings are diagnostic
only. Stops only its own server process group. Never changes source on h20-1.
"""
from __future__ import annotations
import importlib.metadata
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get('SCALE_FINE_ROOT', REPO/'results/fine_blocks_20261107'))
SOURCE = Path(os.environ.get('HISA_SOURCE', REPO/'HISA_current'))
EVAL_DIR = REPO/'SCALE_relaxed/runner/evaluators'
DATA = Path(os.environ.get('RULER_DATA_ROOT', '/datasets/ruler'))
MODEL = os.environ.get('MODEL_PATH', '/models/DeepSeek-V3.2')
OUT = ROOT/'results'
PORT = int(os.environ.get('SCALE_PORT', '31958'))
ACTIVE = None

def log(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%S%z'), message, flush=True)

def write_json(path, obj):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)

def stop_owned():
    global ACTIVE
    if ACTIVE is None:
        return
    process = ACTIVE
    ACTIVE = None
    # Every launch has its own session/process group. Do not kill other GPU jobs.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)

def on_signal(signum, frame):
    stop_owned()
    raise SystemExit(128+signum)

def get_json(path, timeout=5):
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}{path}', timeout=timeout) as r:
        return json.load(r)

def ready():
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=3) as r:
            return r.status == 200
    except Exception:
        return False

def sparse_smoke():
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(MODEL+'/tokenizer.json')
    chunk = tokenizer.encode('A short factual note about a library and its books. ', add_special_tokens=False).ids
    ids = (chunk*(16384//len(chunk)+1))[:16384]
    data = json.dumps({'input_ids':ids, 'sampling_params':{'temperature':0, 'max_new_tokens':4}, 'stream':False}).encode()
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}/generate', data=data, headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=900) as response:
        result = json.load(response)
    assert result.get('meta_info', {}).get('completion_tokens', 0) > 0, result
    return result

def main():
    global ACTIVE
    OUT.mkdir(parents=True, exist_ok=True)
    lock_file = (OUT/'queue.lock').open('a')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old_failures = OUT/'failures.json'
    if old_failures.exists():
        old_failures.rename(OUT/f'failures.previous_attempt.{int(time.time())}.json')
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    versions = {}
    for name in ['torch','triton','sglang-kernel','sgl-deep-gemm','flashinfer-python','flashinfer-cubin','transformers','numpy']:
        versions[name] = importlib.metadata.version(name)
    write_json(OUT/'environment.json', versions)
    failures = []
    for block in [16,8]:
        name = f'hisa{block}_c8192_all_sparse'
        dst = OUT/name
        dst.mkdir(exist_ok=True)
        if (dst/'DONE').exists():
            log(f'skip completed {name}')
            continue
        if ready():
            raise RuntimeError(f'Port {PORT} already serves a process; refusing to replace it.')
        env = os.environ.copy()
        env.update({
            'PYTHONUNBUFFERED':'1', 'PYTHONPATH':str(SOURCE/'python'),
            'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7',
            'SGLANG_NSA_FUSE_TOPK':'0', 'SGLANG_NSA_PER_HEAD_INDEX':'0',
            'SGLANG_NSA_EXPERIMENTAL_PREFILL_PER_HEAD':'0',
            'SGLANG_JIT_DEEPGEMM_PRECOMPILE':'0', 'SGLANG_NSA_ADAPTIVE_HISA_MODE':'off',
            'SGLANG_HISA_HEADWISE_HIERARCHICAL':'0', 'SGLANG_HISA_EAGER_CORRECTNESS':'0',
            'SGLANG_HISA_EAGER_PREFILL':'0', 'SGLANG_HISA_LEARNED_COARSE':'0',
            'SGLANG_HISA_DENSE_FINAL':'0', 'MAX_JOBS':'4',
        })
        command = [sys.executable,'-m','sglang.launch_server','--model-path',MODEL,
            '--served-model-name','deepseek-v3.2','--tp-size','8','--host','127.0.0.1',
            '--port',str(PORT),'--context-length','163840','--trust-remote-code',
            '--reasoning-parser','deepseek-v3','--chunked-prefill-size','8192',
            '--mem-fraction-static','0.76','--max-total-tokens','163840',
            '--max-running-requests','1','--disable-radix-cache','--random-seed','20260921',
            '--watchdog-timeout','900','--cuda-graph-bs','1',
            '--json-model-override-args',json.dumps({'use_hisa':True,'hisa_k_block_size':block,'hisa_block_topk':8192//block})]
        write_json(dst/'launch.json', {'command':command,'env':{k:v for k,v in env.items() if k.startswith(('SGLANG_','CUDA_','PYTHONPATH','MAX_JOBS','LD_LIBRARY_PATH'))},
            'candidate_tokens':8192,'final_topk':2048,'block_tokens':block,'dense_final':False,
            'timing_valid_for_paper':False,'timing_note':'Co-located llama services on GPUs 6 and 7; this queue measures quality only.',
            'source_snapshot':'experiments/20261107/provenance.json','h20_1_scale_commit':'b3106dcfd6ec97c706fb244db7815578c3983765',
            'model_revision':'a7e62ac04ecb2c0a54d736dc46601c5606cf10a6'})
        write_json(OUT/'status.json', {'status':'starting','arm':name,'time':time.time()})
        log(f'launch {name}')
        try:
            with (dst/'server.log').open('a') as server_log:
                ACTIVE = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic()+3600
                while not ready():
                    if ACTIVE.poll() is not None:
                        raise RuntimeError(f'server exited {ACTIVE.returncode}; see server.log')
                    if time.monotonic() > deadline:
                        raise TimeoutError('server startup exceeded 3600 seconds')
                    time.sleep(5)
                try:
                    write_json(dst/'server_info.json',get_json('/get_server_info',timeout=30))
                except Exception as e:
                    log(f'server info unavailable: {e!r}')
                write_json(dst/'smoke.json',sparse_smoke())
                log(f'sparse 16K prefill/decode smoke passed: {name}')
                for length in ['128k','32k']:
                    prediction = dst/f'ruler_{length}_predictions.jsonl'
                    summary_path = dst/f'ruler_{length}_summary.json'
                    write_json(OUT/'status.json', {'status':'evaluating','arm':name,'length':length,'time':time.time()})
                    eval_command = [sys.executable,str(EVAL_DIR/'evaluate_ruler_e2e.py'), '--data-root',str(DATA),
                        '--model',MODEL,'--server',f'http://127.0.0.1:{PORT}','--output',str(prediction),
                        '--summary',str(summary_path),'--lengths',length,'--all-records','--resume']
                    with (dst/f'evaluator_{length}.log').open('a') as eval_log:
                        subprocess.run(eval_command, cwd=EVAL_DIR, env=env, stdout=eval_log, stderr=subprocess.STDOUT,check=True)
                    summary = json.loads(summary_path.read_text())
                    if summary.get('examples') != 182 or summary.get('successful') != 182:
                        raise RuntimeError(f'{length}: expected 182 successful examples; got {summary}')
                    log(f'finished {name} {length}: accuracy={summary["accuracy"]}')
                (dst/'DONE').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z')+'\n')
        except Exception as e:
            log(f'failed {name}: {e!r}')
            failures.append({'arm':name,'error':repr(e),'time':time.time()})
            write_json(OUT/'failures.json',failures)
        finally:
            stop_owned()
            time.sleep(10)
    write_json(OUT/'status.json', {'status':'complete' if not failures else 'failed','failures':failures,'time':time.time()})
    if failures:
        raise SystemExit(1)
    (OUT/'SUITE_DONE').write_text(time.strftime('%Y-%m-%dT%H:%M:%S%z')+'\n')
    log('all fine-block RULER quality arms complete')

if __name__ == '__main__':
    main()
