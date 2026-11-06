#!/usr/bin/env python3
"""Paired experiments for the three attribution questions.

1. Dense-final factorial on RULER: {SCALE-64, HISA-64} x {flat DSA last chunk,
   hierarchical last chunk}. Same prompts, seed, and refinement budget.
2. Latency sweep of HISA block size and candidate budget, then RULER at the
   HISA point whose prefill time matches SCALE-64.
3. The critical-token replay is a separate script; this runner does not redo it.

Every arm is resumable. A finished arm has DONE in its directory.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(os.environ.get("SCALE_SOURCE", Path(__file__).resolve().parents[1]))
HISA = Path(os.environ.get("HISA_SOURCE", ROOT.parent / "HISA_current"))
MODEL = os.environ.get("MODEL_PATH", "/models/DeepSeek-V3.2")
RULER = Path(os.environ.get("RULER_DATA_ROOT", "/datasets/ruler"))
EVALUATOR = Path(os.environ.get(
    "RULER_EVALUATOR", ROOT / "runner/evaluators/evaluate_ruler_e2e.py"
))
CLIENT = ROOT / "runner/speed_bench_20260921/bench_client.py"
OUT = Path(os.environ.get("SCALE_RESULTS", ROOT.parent / "results/attribution_20261107"))
PORT = int(os.environ.get("SCALE_PORT", "31946"))


SCALE_ENV = {
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
    "SGLANG_NSA_ADAPTIVE_HISA_SUMMARY_COMPRESSION": "8",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR": "64",
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


def hisa_env(dense_final: bool) -> dict[str, str]:
    env = {
        "PYTHONUNBUFFERED": "1",
        "SGLANG_NSA_FUSE_TOPK": "0",
        "SGLANG_NSA_PER_HEAD_INDEX": "0",
        "SGLANG_NSA_EXPERIMENTAL_PREFILL_PER_HEAD": "0",
        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
        "SGLANG_NSA_ADAPTIVE_HISA_MODE": "off",
        "SGLANG_HISA_HEADWISE_HIERARCHICAL": "0",
        "SGLANG_HISA_EAGER_CORRECTNESS": "0",
        "SGLANG_HISA_EAGER_PREFILL": "0",
        "SGLANG_HISA_LEARNED_COARSE": "0",
        "SGLANG_HISA_DENSE_FINAL": "1" if dense_final else "0",
    }
    return env


def scale_env(dense_final: bool) -> dict[str, str]:
    env = dict(SCALE_ENV)
    env["SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_DENSE_FINAL"] = (
        "1" if dense_final else "0"
    )
    return env


# name, source, env, extra server args, role
def arms() -> list[dict]:
    rows = [
        {
            "name": "scale64_dense_final",
            "source": ROOT,
            "env": scale_env(True),
            "override": None,
            "family": "scale",
            "dense_final": True,
            "block": None,
            "topk": None,
            "candidates": 8192,
        },
        {
            "name": "scale64_all_sparse",
            "source": ROOT,
            "env": scale_env(False),
            "override": None,
            "family": "scale",
            "dense_final": False,
            "block": None,
            "topk": None,
            "candidates": 8192,
        },
    ]
    hisa_points = [
        ("hisa64", 64, 128, False),
        ("hisa64_dense_final", 64, 128, True),
        ("hisa32", 32, 256, False),
        ("hisa16", 16, 512, False),
        ("hisa64_c16384", 64, 256, False),
        ("hisa64_c24576", 64, 384, False),
        ("hisa32_c16384", 32, 512, False),
    ]
    for name, block, topk, dense in hisa_points:
        rows.append(
            {
                "name": name,
                "source": HISA,
                "env": hisa_env(dense),
                "override": {
                    "use_hisa": True,
                    "hisa_k_block_size": block,
                    "hisa_block_topk": topk,
                },
                "family": "hisa",
                "dense_final": dense,
                "block": block,
                "topk": topk,
                "candidates": block * topk,
            }
        )
    return rows


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def gpu_pids() -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=True,
    )
    return sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})


def wait_for_idle(timeout: int = 48 * 3600) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while True:
        busy = gpu_pids()
        if not busy:
            return
        if busy != last:
            log(f"waiting for GPUs, pids={busy}")
            last = busy
        if time.monotonic() >= deadline:
            raise TimeoutError(f"GPUs stayed busy: {busy}")
        time.sleep(30)


def health() -> bool:
    for path in ("/health_generate", "/health"):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=3) as response:
                if response.status == 200:
                    return True
        except Exception:
            pass
    return False


def stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def server_command(arm: dict) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
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
    if arm["override"] is not None:
        command.extend(["--json-model-override-args", json.dumps(arm["override"])])
    return command


def launch(arm: dict, dst: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(arm["env"])
    env["PYTHONPATH"] = str(arm["source"] / "python")
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    command = server_command(arm)
    (dst / "launch.json").write_text(
        json.dumps(
            {
                "command": command,
                "env": arm["env"],
                "source": str(arm["source"]),
                "model": MODEL,
                "block": arm["block"],
                "topk": arm["topk"],
                "candidates": arm["candidates"],
                "dense_final": arm["dense_final"],
            },
            indent=2,
        )
    )
    log_file = (dst / "server.log").open("w")
    process = subprocess.Popen(
        command,
        cwd=arm["source"],
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    process._log_file = log_file  # type: ignore[attr-defined]
    return process


def wait_ready(process: subprocess.Popen, dst: Path, timeout: int = 2400) -> None:
    started = time.monotonic()
    while not health():
        if process.poll() is not None:
            raise RuntimeError(f"server exited {process.returncode}; see {dst / 'server.log'}")
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"startup exceeded {timeout}s; see {dst / 'server.log'}")
        time.sleep(5)
    (dst / "ready.json").write_text(json.dumps({"startup_s": time.monotonic() - started}))


def close_server(process: subprocess.Popen) -> None:
    stop(process)
    log_file = getattr(process, "_log_file", None)
    if log_file is not None:
        log_file.close()


def read_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in reversed(rows):
        if row.get("summary"):
            return row
    return None


def run_ttft(arm: dict) -> None:
    dst = OUT / "ttft" / arm["name"]
    dst.mkdir(parents=True, exist_ok=True)
    if (dst / "DONE").exists() and read_summary(dst / "results.jsonl"):
        log(f"ttft skip {arm['name']}")
        return
    wait_for_idle()
    if health():
        raise RuntimeError(f"port {PORT} is already serving")
    log(f"ttft start {arm['name']}")
    process = launch(arm, dst)
    try:
        wait_ready(process, dst)
        results = dst / "results.jsonl"
        results.unlink(missing_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(CLIENT),
                "--port",
                str(PORT),
                "--arm",
                arm["name"],
                "--out",
                str(results),
                "--lengths",
                "131072",
                "--reps",
                "3",
                "--warmup",
                "1",
                "--new-tokens",
                "128",
            ],
            check=True,
        )
        summary = read_summary(results)
        if summary is None:
            raise RuntimeError("ttft produced no summary")
        (dst / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        log(f"ttft done {arm['name']} ttft={summary['ttft_s_median']:.2f}s")
    finally:
        close_server(process)


def run_ruler(arm: dict) -> None:
    dst = OUT / "ruler" / arm["name"]
    dst.mkdir(parents=True, exist_ok=True)
    if (dst / "DONE").exists():
        log(f"ruler skip {arm['name']}")
        return
    wait_for_idle()
    if health():
        raise RuntimeError(f"port {PORT} is already serving")
    log(f"ruler start {arm['name']}")
    process = launch(arm, dst)
    try:
        wait_ready(process, dst)
        output = dst / "ruler_predictions.jsonl"
        summary_path = dst / "ruler_summary.json"
        subprocess.run(
            [
                sys.executable,
                str(EVALUATOR),
                "--data-root",
                str(RULER),
                "--model",
                MODEL,
                "--server",
                f"http://127.0.0.1:{PORT}",
                "--output",
                str(output),
                "--summary",
                str(summary_path),
                "--lengths",
                "32k",
                "128k",
                "--all-records",
                "--resume",
            ],
            cwd=EVALUATOR.parent,
            check=True,
        )
        summary = json.loads(summary_path.read_text())
        if summary.get("successful") != summary.get("examples"):
            raise RuntimeError(f"incomplete ruler: {summary}")
        (dst / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        log(f"ruler done {arm['name']} acc={summary.get('accuracy')}")
    finally:
        close_server(process)


def choose_iso(catalog: list[dict]) -> dict:
    timings = []
    for arm in catalog:
        summary = read_summary(OUT / "ttft" / arm["name"] / "results.jsonl")
        if summary is None:
            continue
        timings.append({**arm, "ttft_s": summary["ttft_s_median"], "tpot_ms": summary["tpot_ms_median"]})
    scale = next(row for row in timings if row["name"] == "scale64_dense_final")
    target = float(scale["ttft_s"])
    pool = [
        row
        for row in timings
        if row["family"] == "hisa" and not row["dense_final"]
    ]
    reached = [row for row in pool if row["ttft_s"] + 0.25 >= target]
    if reached:
        chosen = min(reached, key=lambda row: (row["ttft_s"] - target, row["ttft_s"]))
        reason = "slowest-not-required: closest HISA prefill at or above SCALE-64"
    else:
        chosen = max(pool, key=lambda row: row["ttft_s"])
        reason = "no HISA point reached SCALE-64 prefill; using the slowest measured HISA point"
    decision = {
        "scale64_dense_final_ttft_s": target,
        "chosen": chosen["name"],
        "chosen_ttft_s": chosen["ttft_s"],
        "reason": reason,
        "points": [
            {
                "name": row["name"],
                "family": row["family"],
                "dense_final": row["dense_final"],
                "block": row["block"],
                "topk": row["topk"],
                "candidates": row["candidates"],
                "ttft_s": row["ttft_s"],
                "tpot_ms": row["tpot_ms"],
            }
            for row in timings
        ],
    }
    (OUT / "iso_latency_choice.json").write_text(json.dumps(decision, indent=2))
    log(f"iso-latency choice {chosen['name']} at {chosen['ttft_s']:.2f}s vs SCALE {target:.2f}s")
    return chosen


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = arms()
    (OUT / "catalog.json").write_text(
        json.dumps(
            [
                {
                    **{key: (str(value) if isinstance(value, Path) else value) for key, value in arm.items() if key != "env"},
                    "env": arm["env"],
                }
                for arm in catalog
            ],
            indent=2,
        )
    )
    failures = []
    for arm in catalog:
        try:
            run_ttft(arm)
        except Exception as error:
            log(f"ttft failed {arm['name']}: {error!r}")
            failures.append({"phase": "ttft", "arm": arm["name"], "error": repr(error)})
            (OUT / "failures.json").write_text(json.dumps(failures, indent=2))
    try:
        chosen = choose_iso(catalog)
        extra = [chosen["name"]]
    except Exception as error:
        log(f"iso-latency choice failed: {error!r}")
        failures.append({"phase": "iso", "arm": "", "error": repr(error)})
        extra = []
    quality_names = [
        "scale64_dense_final",
        "scale64_all_sparse",
        "hisa64",
        "hisa64_dense_final",
        *extra,
    ]
    by_name = {arm["name"]: arm for arm in catalog}
    seen = set()
    for name in quality_names:
        if name in seen:
            continue
        seen.add(name)
        try:
            run_ruler(by_name[name])
        except Exception as error:
            log(f"ruler failed {name}: {error!r}")
            failures.append({"phase": "ruler", "arm": name, "error": repr(error)})
            (OUT / "failures.json").write_text(json.dumps(failures, indent=2))
    (OUT / "SUITE_DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    log("suite finished")


if __name__ == "__main__":
    main()
