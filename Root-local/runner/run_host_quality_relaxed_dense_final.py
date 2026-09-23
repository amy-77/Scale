#!/usr/bin/env python3
"""Run full LongBench-v2 and RULER for relaxed global λ-DP + dense final.

The run is host-native because the historical evaluator depends on a missing
Docker image and /DATA/disk0/qyl layout.  Outputs are resumable: rerunning this
script skips successful prediction rows already present in the output files.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/share-evpfs/flagos/models/DeepSeek-V3.2")
EVALUATORS = Path(
    "/data/jz/sglang_coarse_adapt/h20-9-57/accuracy_v2/evaluators"
)
LONGBENCH_DATA = Path("/data/jz/longbench_heldout.json")
RULER_DATA = Path(
    "/data/jz/sglang_coarse_adapt/h20-9-57/evaluation_inputs/ruler"
)
OUT = ROOT / "stats/quality_relaxed_dense_final_20260923"
PORT = 31932
EXPECTED_METHOD = "P-key-sync_nonoverlap-target128-relaxed-soft2"


SERVER_ENV = {
    "PYTHONUNBUFFERED": "1",
    "SGLANG_NSA_FUSE_TOPK": "0",
    "SGLANG_NSA_PER_HEAD_INDEX": "0",
    "SGLANG_NSA_EXPERIMENTAL_PREFILL_PER_HEAD": "0",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
    "SGLANG_NSA_ADAPTIVE_HISA_MODE": "adaptive_decode",
    "SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC": "key_sse",
    "SGLANG_NSA_ADAPTIVE_HISA_SPLIT_BACKEND": "gpu",
    "SGLANG_NSA_ADAPTIVE_HISA_BUILD_SUMMARIES": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY": "sync_nonoverlap",
    "SGLANG_NSA_ADAPTIVE_HISA_SUMMARY_COMPRESSION": "32",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR": "128",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_ROUNDS": "1",
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
    "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_DENSE_FINAL": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_ROWS": "8192",
    "SGLANG_NSA_ADAPTIVE_HISA_PREFILL_SELECT": "weighted",
    "SGLANG_NSA_ADAPTIVE_HISA_TREE_CACHE": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_ROOT_LOCAL_PARTITION": "0",
    "SGLANG_NSA_ADAPTIVE_HISA_ROOT_LOCAL_FUSED_SUMMARIES": "0",
    "SGLANG_NSA_ADAPTIVE_HISA_RELAXED_SPLIT": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_SOFT_MERGE_ROUNDS": "2",
    "SGLANG_NSA_ADAPTIVE_HISA_LAMBDA_MAX_ROUNDS": "2",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_ALPHA": "2.0",
}


def server_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(MODEL),
        "--served-model-name",
        "deepseek-v3.2",
        "--tp-size",
        "8",
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "--context-length",
        "163840",
        "--trust-remote-code",
        "--reasoning-parser",
        "deepseek-v3",
        "--chunked-prefill-size",
        "8192",
        "--mem-fraction-static",
        "0.82",
        "--max-running-requests",
        "1",
        "--disable-radix-cache",
        "--random-seed",
        "20260921",
        "--watchdog-timeout",
        "900",
        "--cuda-graph-bs",
        "1",
    ]


def write_progress(state: str, **values) -> None:
    payload = {
        "state": state,
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **values,
    }
    (OUT / "progress.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload), flush=True)


def health() -> bool:
    for path in ("/health_generate", "/health"):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}{path}", timeout=3
            ) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
    return False


def gpu_pids() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return sorted(
        {line.strip() for line in result.stdout.splitlines() if line.strip()}
    )


def wait_for_idle_gpus(timeout: int = 86400) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while True:
        busy = gpu_pids()
        if not busy:
            return
        if busy != last:
            write_progress("waiting_for_gpus", pids=busy)
            last = busy
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GPUs remained busy for {timeout}s: {busy}")
        time.sleep(30)


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def smoke() -> None:
    ids = [100 + (index * 7919) % 30000 for index in range(20000)]
    payload = json.dumps(
        {
            "input_ids": ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 8,
                "ignore_eos": True,
            },
        }
    ).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        result = json.load(response)
    record = {
        "prompt_tokens": result.get("meta_info", {}).get("prompt_tokens"),
        "completion_tokens": result.get("meta_info", {}).get("completion_tokens"),
    }
    (OUT / "smoke.json").write_text(json.dumps(record, indent=2))


def completed_count(path: Path) -> int:
    if not path.exists():
        return 0
    completed = set()
    for line in path.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if row.get("status") == "ok":
            completed.add(str(row.get("source_id")))
    return len(completed)


def run_evaluator(
    bench: str,
    command: list[str],
    output: Path,
    expected: int,
) -> dict:
    write_progress(
        f"evaluating_{bench}",
        completed=completed_count(output),
        expected=expected,
    )
    log_path = OUT / f"{bench}.log"
    with log_path.open("a") as log_file:
        process = subprocess.Popen(
            command,
            cwd=EVALUATORS,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        seen = 0
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            print(line, end="", flush=True)
            seen += 1
            if seen % 10 == 0:
                write_progress(
                    f"evaluating_{bench}",
                    completed=completed_count(output),
                    expected=expected,
                )
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"{bench} evaluator exited with {returncode}")
    summary_path = OUT / f"{bench}_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("examples") != expected or summary.get("successful") != expected:
        raise RuntimeError(f"incomplete {bench} evaluation: {summary}")
    write_progress(f"{bench}_completed", summary=summary)
    return summary


def evaluator_commands() -> list[tuple[str, list[str], Path, int]]:
    server = f"http://127.0.0.1:{PORT}"
    longbench_output = OUT / "longbench_v2_predictions.jsonl"
    ruler_output = OUT / "ruler_predictions.jsonl"
    return [
        (
            "longbench_v2",
            [
                sys.executable,
                str(EVALUATORS / "evaluate_longbench_v2_e2e.py"),
                "--model",
                str(MODEL),
                "--server",
                server,
                "--data",
                str(LONGBENCH_DATA),
                "--output",
                str(longbench_output),
                "--summary",
                str(OUT / "longbench_v2_summary.json"),
                "--max-context-tokens",
                "131072",
                "--max-new-tokens",
                "128",
                "--concurrency",
                "1",
                "--resume",
            ],
            longbench_output,
            401,
        ),
        (
            "ruler",
            [
                sys.executable,
                str(EVALUATORS / "evaluate_ruler_e2e.py"),
                "--data-root",
                str(RULER_DATA),
                "--model",
                str(MODEL),
                "--server",
                server,
                "--output",
                str(ruler_output),
                "--summary",
                str(OUT / "ruler_summary.json"),
                "--lengths",
                "32k",
                "128k",
                "--all-records",
                "--resume",
            ],
            ruler_output,
            364,
        ),
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lock_file = (OUT / "runner.lock").open("w")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    required = [
        MODEL,
        EVALUATORS / "evaluate_longbench_v2_e2e.py",
        EVALUATORS / "evaluate_ruler_e2e.py",
        LONGBENCH_DATA,
        RULER_DATA,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing evaluation inputs: {missing}")

    env = os.environ.copy()
    env.update(SERVER_ENV)
    env["PYTHONPATH"] = str(ROOT / "python")
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    command = server_command()
    (OUT / "launch.json").write_text(
        json.dumps(
            {
                "command": command,
                "env": SERVER_ENV,
                "source": str(ROOT),
                "model": str(MODEL),
                "longbench_data": str(LONGBENCH_DATA),
                "ruler_data": str(RULER_DATA),
            },
            indent=2,
        )
    )

    wait_for_idle_gpus()
    if health():
        raise RuntimeError(f"port {PORT} is already serving")

    write_progress("starting_server")
    server_log = (OUT / "server.log").open("a")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    started = time.monotonic()
    summaries = {}
    try:
        while not health():
            if process.poll() is not None:
                raise RuntimeError(
                    f"server exited with {process.returncode}; see {OUT / 'server.log'}"
                )
            if time.monotonic() - started > 2400:
                raise TimeoutError("server startup exceeded 2400 seconds")
            time.sleep(5)
        write_progress("smoke", startup_s=time.monotonic() - started)
        smoke()
        time.sleep(2)
        log_text = (OUT / "server.log").read_text(errors="replace")
        if EXPECTED_METHOD not in log_text:
            raise RuntimeError(f"server did not report {EXPECTED_METHOD}")
        if "final=1" in log_text:
            raise RuntimeError("dense-final run unexpectedly used sparse final prefill")
        if "final=0" not in log_text:
            raise RuntimeError("smoke did not exercise sparse intermediate prefill")
        for bench, eval_command, output, expected in evaluator_commands():
            summaries[bench] = run_evaluator(
                bench, eval_command, output, expected
            )
        (OUT / "summary.json").write_text(json.dumps(summaries, indent=2))
        (OUT / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        write_progress("all_completed", summaries=summaries)
    except BaseException as error:
        write_progress("failed", error=repr(error))
        raise
    finally:
        stop_process(process)
        server_log.close()
        text = (OUT / "server.log").read_text(errors="replace")
        evidence = {
            "tracebacks": text.count("Traceback"),
            "method_mentions": text.count(EXPECTED_METHOD),
            "sparse_intermediate_mentions": len(
                re.findall(r"sparse prefill .+ final=0", text)
            ),
            "sparse_final_mentions": len(
                re.findall(r"sparse prefill .+ final=1", text)
            ),
        }
        (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
