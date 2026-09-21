#!/usr/bin/env python3
"""Start one docker server per arm, measure TTFT/TPOT with bench_client.py, stop it."""
import json, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path("/DATA/disk0/qyl")
OUT = ROOT / "data/speed_bench_20260921"
IMAGE = "qyl/sglang-hisa:eval"
MODEL = "/workspace/qyl/models/deepseek-v3.2"
PORT = 31930
NAME = "speedbench-arm"
CACHE = ROOT / "cache/adaptive_0921_h201"

COMMON_ENV = {
    "PYTHONUNBUFFERED": "1",
    "SGLANG_NSA_FUSE_TOPK": "0",
    "SGLANG_NSA_PER_HEAD_INDEX": "0",
    "SGLANG_NSA_EXPERIMENTAL_PREFILL_PER_HEAD": "0",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
}
ADAPTIVE_ENV = {
    "SGLANG_NSA_ADAPTIVE_HISA_MODE": "adaptive_decode",
    "SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC": "key_sse",
    "SGLANG_NSA_ADAPTIVE_HISA_SPLIT_BACKEND": "gpu",
    "SGLANG_NSA_ADAPTIVE_HISA_BUILD_SUMMARIES": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY": "sync_nonoverlap",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR": "64",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_ROUNDS": "8",
    "SGLANG_NSA_ADAPTIVE_HISA_MAX_MERGE_LEN": "0",
    "SGLANG_NSA_ADAPTIVE_HISA_GRAPH_BUILD": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_GPU_STREAM": "side",
    "SGLANG_NSA_ADAPTIVE_HISA_FALLBACK_LAYERS": "0",
    "SGLANG_NSA_ADAPTIVE_HISA_SINK_TOKENS": "64",
    "SGLANG_NSA_ADAPTIVE_HISA_TAIL_TOKENS": "256",
    "SGLANG_NSA_ADAPTIVE_HISA_CANDIDATE_TOKENS": "8192",
    "SGLANG_NSA_ADAPTIVE_HISA_DECODE_CHUNK": "64",
    "SGLANG_NSA_ADAPTIVE_HISA_SKIP_CLEAN": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_FUSE_PAGE_TABLE": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_RAW_FP8_BUILDER": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_DECODE_TIMING": "0",
    "SGLANG_NSA_ADAPTIVE_HISA_SELECTOR_PROFILE": "0",
    "SGLANG_NSA_ADAPTIVE_HISA_PARTITION_VALIDATE": "0",
    "SGLANG_NSA_ADAPTIVE_HISA_FORWARD_TIMING": "0",
}

ARMS = {
    "dsa": dict(pythonpath="/workspace/qyl/code/sglang_hisa/python", env={}, override=None),
    "hisa64": dict(pythonpath="/workspace/qyl/code/sglang_hisa_fixed_e399d9f/python",
                   env={"SGLANG_HISA_HEADWISE_HIERARCHICAL": "0", "SGLANG_HISA_EAGER_CORRECTNESS": "0",
                        "SGLANG_HISA_EAGER_PREFILL": "0", "SGLANG_HISA_LEARNED_COARSE": "0"},
                   override='{"use_hisa":true,"hisa_k_block_size":64,"hisa_block_topk":128}'),
    "adaptive": dict(pythonpath="/workspace/qyl/code/adaptive_0921_h201/python", env=ADAPTIVE_ENV, override=None),
    "adaptive_sp": dict(pythonpath="/workspace/qyl/code/adaptive_0921_h202/python",
                        env={**ADAPTIVE_ENV, "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL": "1",
                             "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_ROWS": "2048"}, override=None),
    # h202 with the batched weighted radix selector in prefill (default since 2026-09-21 22:xx);
    # adaptive_sp_argsort pins the legacy argsort+cumsum+binary-search path for A/B.
    "adaptive_sp_ws": dict(pythonpath="/workspace/qyl/code/adaptive_0921_h202/python",
                           env={**ADAPTIVE_ENV, "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL": "1",
                                "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_ROWS": "2048",
                                "SGLANG_NSA_ADAPTIVE_HISA_PREFILL_SELECT": "weighted"}, override=None),
    "adaptive_sp_argsort": dict(pythonpath="/workspace/qyl/code/adaptive_0921_h202/python",
                                env={**ADAPTIVE_ENV, "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL": "1",
                                     "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_ROWS": "2048",
                                     "SGLANG_NSA_ADAPTIVE_HISA_PREFILL_SELECT": "argsort"}, override=None),
    # 2026-09-21 23:xx: + incremental Key-SSE tree across chunks (TREE_CACHE) + fine scorer on the
    # flat index-K via HISA's persistent block-sparse kernel (K=1). *_notree isolates the fine-scorer part.
    "adaptive_sp_v3": dict(pythonpath="/workspace/qyl/code/adaptive_0921_h202/python",
                           env={**ADAPTIVE_ENV, "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL": "1",
                                "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_ROWS": "2048"}, override=None),
    "adaptive_sp_v3_notree": dict(pythonpath="/workspace/qyl/code/adaptive_0921_h202/python",
                                  env={**ADAPTIVE_ENV, "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL": "1",
                                       "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_ROWS": "2048",
                                       "SGLANG_NSA_ADAPTIVE_HISA_TREE_CACHE": "0"}, override=None),
    # 2026-09-22: + decode selector fast path with warp threshold-bin search and lane-parallel
    # interval expansion (weighted_select.cu); prefill unchanged vs adaptive_sp_v3.
    "adaptive_sp_v4": dict(pythonpath="/workspace/qyl/code/adaptive_0921_h202/python",
                           env={**ADAPTIVE_ENV, "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL": "1",
                                "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_ROWS": "2048"}, override=None),
    "adaptive_sp_df": dict(pythonpath="/workspace/qyl/code/adaptive_0921_h202/python",
                           env={**ADAPTIVE_ENV, "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL": "1",
                                "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_ROWS": "2048",
                                "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_DENSE_FINAL": "1"}, override=None),
}


def sh(args, check=True):
    r = subprocess.run(args, text=True, capture_output=True)
    if check and r.returncode:
        raise RuntimeError(f"{args} -> {r.returncode}: {r.stderr[-2000:]}")
    return r.stdout


def health(path="/health_generate"):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def running():
    r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", NAME], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == "true"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (OUT / "driver.log").open("a") as f:
        f.write(line + "\n")


def run_arm(arm, graph):
    tag = f"{arm}_{'graph' if graph else 'nograph'}"
    dst = OUT / tag
    dst.mkdir(parents=True, exist_ok=True)
    if (dst / "DONE").exists():
        log(f"skip {tag} (done)"); return
    spec = ARMS[arm]
    subprocess.run(["docker", "rm", "-f", NAME], capture_output=True)
    busy = sh(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]).strip()
    if busy:
        raise RuntimeError(f"GPU busy: {busy}")
    env = {**COMMON_ENV, **spec["env"], "PYTHONPATH": spec["pythonpath"]}
    argv = ["docker", "run", "-d", "--name", NAME, "--gpus", "all", "--ipc", "host", "--network", "host",
            "-v", f"{ROOT}:/workspace/qyl", "-v", f"{CACHE}:/root/.cache"]
    for k, v in env.items():
        argv += ["-e", f"{k}={v}"]
    cmd = ["python", "-m", "sglang.launch_server", "--model-path", MODEL, "--served-model-name", "deepseek-v3.2",
           "--tp-size", "8", "--host", "127.0.0.1", "--port", str(PORT), "--context-length", "163840",
           "--trust-remote-code", "--reasoning-parser", "deepseek-v3", "--chunked-prefill-size", "8192",
           "--mem-fraction-static", "0.82", "--max-running-requests", "1", "--disable-radix-cache",
           "--random-seed", "20260921", "--watchdog-timeout", "900"]
    cmd += ["--cuda-graph-bs", "1"] if graph else ["--disable-cuda-graph"]
    if spec["override"]:
        cmd += ["--json-model-override-args", spec["override"]]
    argv += [IMAGE, *cmd]
    (dst / "launch.json").write_text(json.dumps(argv, indent=2))
    log(f"start {tag}")
    t0 = time.monotonic()
    sh(argv)
    try:
        while True:
            if not running():
                raise RuntimeError("server exited during start-up")
            if health():
                break
            if time.monotonic() - t0 > 2400:
                raise RuntimeError("readiness timeout")
            time.sleep(5)
        log(f"ready {tag} after {time.monotonic() - t0:.0f}s")
        res = dst / "results.jsonl"
        res.unlink(missing_ok=True)
        with (dst / "client.log").open("w") as f:
            subprocess.run([sys.executable, "/tmp/hisa_speed/bench_client.py", "--port", str(PORT), "--arm", tag,
                            "--out", str(res)], stdout=f, stderr=subprocess.STDOUT, check=True)
        (dst / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
        log(f"done {tag}")
    except Exception as e:
        log(f"FAILED {tag}: {e!r}")
        (dst / "FAILED").write_text(repr(e))
    finally:
        with (dst / "server.log").open("w") as f:
            subprocess.run(["docker", "logs", NAME], stdout=f, stderr=subprocess.STDOUT)
        subprocess.run(["docker", "stop", "--timeout", "60", NAME], capture_output=True)
        subprocess.run(["docker", "rm", NAME], capture_output=True)


OUT.mkdir(parents=True, exist_ok=True)
queue = [("adaptive", False), ("dsa", False), ("hisa64", False), ("dsa", True), ("adaptive", True)]
if len(sys.argv) > 1:
    queue = [(a.split(":")[0], a.endswith(":graph")) for a in sys.argv[1:]]
for arm, graph in queue:
    run_arm(arm, graph)
log("ALL DONE")
