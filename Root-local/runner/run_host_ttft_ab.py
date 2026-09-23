#!/usr/bin/env python3
"""Host-native TP8 TTFT/TPOT A/B on the current 8xH20 machine.

The historical harness is tied to a Docker image and /DATA/disk0/qyl.  This
runner keeps its measurement protocol (fixed input ids, one warmup, three
128-token samples) but launches the checked-out source and the model available
on this host directly.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISA_ROOT = Path("/data/jz/hisa_pr_upstream")
CLIENT = ROOT / "runner/speed_bench_20260921/bench_client.py"
DEFAULT_MODEL = "/share-evpfs/flagos/models/DeepSeek-V3.2"
DEFAULT_OUT = ROOT / "stats/host_ttft_ab_20260923"

COMMON = {
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
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_ROUNDS": "4",
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
    "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_DENSE_FINAL": "0",
    "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_ROWS": "8192",
    "SGLANG_NSA_ADAPTIVE_HISA_PREFILL_SELECT": "weighted",
    "SGLANG_NSA_ADAPTIVE_HISA_TREE_CACHE": "1",
}

ARMS = {
    "rootlocal_exact": {
        "SGLANG_NSA_ADAPTIVE_HISA_ROOT_LOCAL_PARTITION": "1",
        "SGLANG_NSA_ADAPTIVE_HISA_ROOT_LOCAL_FUSED_SUMMARIES": "0",
        "SGLANG_NSA_ADAPTIVE_HISA_RELAXED_SPLIT": "0",
        "SGLANG_NSA_ADAPTIVE_HISA_SOFT_MERGE_ROUNDS": "0",
    },
    "rootlocal_fused": {
        "SGLANG_NSA_ADAPTIVE_HISA_ROOT_LOCAL_PARTITION": "1",
        "SGLANG_NSA_ADAPTIVE_HISA_ROOT_LOCAL_FUSED_SUMMARIES": "1",
        "SGLANG_NSA_ADAPTIVE_HISA_RELAXED_SPLIT": "0",
        "SGLANG_NSA_ADAPTIVE_HISA_SOFT_MERGE_ROUNDS": "0",
    },
    "global_exact": {
        "SGLANG_NSA_ADAPTIVE_HISA_ROOT_LOCAL_PARTITION": "0",
        "SGLANG_NSA_ADAPTIVE_HISA_ROOT_LOCAL_FUSED_SUMMARIES": "0",
        "SGLANG_NSA_ADAPTIVE_HISA_RELAXED_SPLIT": "0",
        "SGLANG_NSA_ADAPTIVE_HISA_SOFT_MERGE_ROUNDS": "0",
        "SGLANG_NSA_ADAPTIVE_HISA_LAMBDA_MAX_ROUNDS": "6",
        "SGLANG_NSA_ADAPTIVE_HISA_MERGE_ALPHA": "1.0",
        "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_ROUNDS": "4",
    },
    "global_relaxed": {
        "SGLANG_NSA_ADAPTIVE_HISA_ROOT_LOCAL_PARTITION": "0",
        "SGLANG_NSA_ADAPTIVE_HISA_ROOT_LOCAL_FUSED_SUMMARIES": "0",
        "SGLANG_NSA_ADAPTIVE_HISA_RELAXED_SPLIT": "1",
        "SGLANG_NSA_ADAPTIVE_HISA_SOFT_MERGE_ROUNDS": "2",
        "SGLANG_NSA_ADAPTIVE_HISA_LAMBDA_MAX_ROUNDS": "2",
        "SGLANG_NSA_ADAPTIVE_HISA_MERGE_ALPHA": "2.0",
        "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_ROUNDS": "1",
    },
    "hisa64": {
        "SGLANG_NSA_ADAPTIVE_HISA_MODE": "off",
        "SGLANG_HISA_HEADWISE_HIERARCHICAL": "0",
        "SGLANG_HISA_EAGER_CORRECTNESS": "0",
        "SGLANG_HISA_EAGER_PREFILL": "0",
        "SGLANG_HISA_LEARNED_COARSE": "0",
    },
}


def arm_source(arm: str) -> Path:
    return HISA_ROOT if arm == "hisa64" else ROOT


def health(port: int) -> bool:
    for path in ("/health_generate", "/health"):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{path}", timeout=3
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
    return sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def read_completed_summaries(out: Path) -> dict[str, dict]:
    summaries = {}
    for arm in ARMS:
        arm_dir = out / arm
        results = arm_dir / "results.jsonl"
        if not (arm_dir / "DONE").exists() or not results.exists():
            continue
        rows = read_rows(results)
        summaries[arm] = next(
            row for row in reversed(rows) if row.get("summary")
        )
    return summaries


def wait_for_idle_gpus(timeout: int) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while True:
        busy = gpu_pids()
        if not busy:
            return
        current = tuple(busy)
        if current != last:
            print(f"waiting for active GPU processes: {busy}", flush=True)
            last = current
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GPUs remained busy for {timeout}s: {busy}")
        time.sleep(30)


def run_arm(args, arm: str) -> dict:
    dst = args.out / arm
    dst.mkdir(parents=True, exist_ok=True)
    done = dst / "DONE"
    if done.exists() and not args.force:
        rows = read_rows(dst / "results.jsonl")
        return next(row for row in reversed(rows) if row.get("summary"))

    wait_for_idle_gpus(args.gpu_wait_timeout)
    if health(args.port):
        raise RuntimeError(f"port {args.port} is already serving")

    env = os.environ.copy()
    env.update(COMMON)
    env.update(ARMS[arm])
    source = arm_source(arm)
    env["PYTHONPATH"] = str(source / "python")
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--served-model-name",
        "deepseek-v3.2",
        "--tp-size",
        "8",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
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
    if arm == "hisa64":
        command.extend(
            [
                "--json-model-override-args",
                '{"use_hisa":true,"hisa_k_block_size":64,"hisa_block_topk":128}',
            ]
        )
    (dst / "launch.json").write_text(
        json.dumps(
            {
                "command": command,
                "env": {name: env[name] for name in sorted(env) if name.startswith("SGLANG_")},
                "model": args.model,
                "source": str(source),
            },
            indent=2,
        )
    )

    log_file = (dst / "server.log").open("w")
    process = subprocess.Popen(
        command,
        cwd=source,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    started = time.monotonic()
    try:
        while not health(args.port):
            if process.poll() is not None:
                raise RuntimeError(
                    f"{arm} server exited with {process.returncode}; "
                    f"see {dst / 'server.log'}"
                )
            if time.monotonic() - started > args.startup_timeout:
                raise TimeoutError(f"{arm} startup exceeded {args.startup_timeout}s")
            time.sleep(5)
        (dst / "ready.json").write_text(
            json.dumps({"startup_s": time.monotonic() - started}, indent=2)
        )

        smoke = dst / "smoke.jsonl"
        smoke.unlink(missing_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(CLIENT),
                "--port",
                str(args.port),
                "--arm",
                arm,
                "--out",
                str(smoke),
                "--lengths",
                "20000",
                "--reps",
                "1",
                "--warmup",
                "0",
                "--new-tokens",
                "8",
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )

        results = dst / "results.jsonl"
        results.unlink(missing_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(CLIENT),
                "--port",
                str(args.port),
                "--arm",
                arm,
                "--out",
                str(results),
                "--lengths",
                *[str(value) for value in args.lengths],
                "--reps",
                str(args.reps),
                "--warmup",
                str(args.warmup),
                "--new-tokens",
                str(args.new_tokens),
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        rows = read_rows(results)
        summary = next(row for row in reversed(rows) if row.get("summary"))
        done.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        return summary
    finally:
        stop_process(process)
        log_file.close()
        text = (dst / "server.log").read_text(errors="replace")
        evidence = {
            "tracebacks": text.count("Traceback"),
            "adaptive_config": re.findall(r"adaptive-hisa enabled: (.+)", text)[-4:],
            "hisa_config": re.findall(r"NSA indexer: use_hisa=True .+", text)[-4:],
            "kernel_warm": re.findall(r"gpu kernels warm in .+", text)[-4:],
            "fused_summary_mentions": text.count("fused_summaries=True"),
        }
        (dst / "evidence.json").write_text(json.dumps(evidence, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--port", type=int, default=31931)
    parser.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=list(ARMS))
    parser.add_argument("--lengths", nargs="+", type=int, default=[131072])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--startup-timeout", type=int, default=2400)
    parser.add_argument("--gpu-wait-timeout", type=int, default=7200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summaries = read_completed_summaries(args.out)
    for arm in args.arms:
        summaries[arm] = run_arm(args, arm)
        (args.out / "summary.json").write_text(json.dumps(summaries, indent=2))

    paired = {}
    for left, right in (
        ("rootlocal_exact", "rootlocal_fused"),
        ("global_exact", "global_relaxed"),
    ):
        if left not in summaries or right not in summaries:
            continue
        a, b = summaries[left], summaries[right]
        paired[f"{left}_vs_{right}"] = {
            "ttft_delta_s": b["ttft_s_median"] - a["ttft_s_median"],
            "ttft_ratio": b["ttft_s_median"] / a["ttft_s_median"],
            "tpot_delta_ms": b["tpot_ms_median"] - a["tpot_ms_median"],
            "tpot_ratio": b["tpot_ms_median"] / a["tpot_ms_median"],
        }
    (args.out / "paired.json").write_text(json.dumps(paired, indent=2))
    print(json.dumps({"summaries": summaries, "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
