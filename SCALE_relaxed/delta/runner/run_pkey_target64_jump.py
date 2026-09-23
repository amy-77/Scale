"""P-key L/8 -> L/64 Adaptive-HISA queue for jump_h20-1.

This is the host-process equivalent of run_pkey_target64.py.  It never
inspects, starts, stops, or removes Docker containers.  It owns only the
process group that it starts and writes into a new result directory.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request


PACKAGE = Path("/data/jz/9-57-0920/adaptive_hisa_pkey_target64_20260920")
BASE = Path("/data/jz/sglang_coarse_adapt/h20-9-57")
OUT = Path(
    os.environ.get(
        "ADAPTIVE_HISA_RUN_OUT",
        "/data/jz/9-57-0920/results_pkey_target64_e2e_20260920",
    )
)
SOURCE = OUT / "source"
EVALUATORS = OUT / "evaluators"
HELD = BASE / "evaluation_inputs"
OFFICIAL = HELD / "official_dsa"
MODEL = "/share-evpfs/flagos/models/DeepSeek-V3.2"
PORT = int(os.environ.get("ADAPTIVE_HISA_RUN_PORT", "31922"))
CONTEXT_LENGTH = 163840
PYTHON = sys.executable
EXPECTED_METHOD = "P-key-sync_nonoverlap-target64"

ACTIVE: subprocess.Popen | None = None
SERVER_STREAM = None


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def event(name: str, **values) -> None:
    record = {"event": name, "time": now(), **values}
    print(json.dumps(record, ensure_ascii=False), flush=True)
    with (OUT / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def status(**values) -> None:
    values["updated_utc"] = now()
    (OUT / "progress.json").write_text(
        json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_source(refreeze: bool = False) -> None:
    source_python = SOURCE / "python"
    if refreeze and SOURCE.exists():
        shutil.rmtree(SOURCE)
    if not source_python.exists():
        source_python.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            PACKAGE / "python",
            source_python,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            symlinks=True,
        )
    EVALUATORS.mkdir(parents=True, exist_ok=True)
    for name in (
        "evaluate_longbench_v2_e2e.py",
        "collect_longbench_v2.py",
        "evaluate_ruler_e2e.py",
        "collect_ruler.py",
    ):
        shutil.copy2(BASE / "accuracy_v2/evaluators" / name, EVALUATORS / name)

    files = sorted(
        list(
            (
                source_python
                / "sglang/srt/layers/attention/nsa/adaptive_hisa"
            ).rglob("*.py")
        )
        + list(
            (
                source_python
                / "sglang/srt/layers/attention/nsa/adaptive_hisa"
            ).rglob("*.cu")
        )
        + [source_python / "sglang/srt/layers/attention/nsa/nsa_indexer.py"]
    )
    manifest = {
        str(path.relative_to(SOURCE)): sha256(path)
        for path in files
        if path.is_file()
    }
    (OUT / "source.sha256.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    provenance = {
        "package": str(PACKAGE),
        "package_git_head": (PACKAGE / "git_head.txt").read_text().strip(),
        "frozen_source": str(source_python),
        "frozen_utc": now(),
        "runner": str(Path(__file__).resolve()),
    }
    (OUT / "source_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    event("SOURCE_FROZEN", files=len(manifest), refreeze=refreeze)


def gpu_pids() -> list[str]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ],
        text=True,
    )
    return sorted({line.strip() for line in output.splitlines() if line.strip()})


def wait_for_free_gpus(limit_seconds: int = 12 * 3600) -> None:
    deadline = time.monotonic() + limit_seconds
    while True:
        busy = gpu_pids()
        if not busy:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPUs remained occupied by foreign PIDs: {busy}")
        status(state="waiting_for_free_gpus", foreign_pids=busy)
        time.sleep(60)


def health(path: str = "/health", timeout: int = 5) -> bool:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{PORT}{path}", timeout=timeout
        ) as response:
            return response.status == 200
    except Exception:
        return False


def read_server_log(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def base_env(profile: bool) -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("SGLANG_NSA_"):
            env.pop(key)
    env.update(
        {
            "PYTHONPATH": str(SOURCE / "python"),
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
            "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR": "64",
            "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_ROUNDS": "8",
            "SGLANG_NSA_ADAPTIVE_HISA_MAX_MERGE_LEN": "0",
            "SGLANG_NSA_ADAPTIVE_HISA_GRAPH_BUILD": "1",
            "SGLANG_NSA_ADAPTIVE_HISA_GPU_STREAM": "side",
            "SGLANG_NSA_ADAPTIVE_HISA_FALLBACK_LAYERS": "0",
            "SGLANG_NSA_ADAPTIVE_HISA_SINK_TOKENS": "64",
            "SGLANG_NSA_ADAPTIVE_HISA_TAIL_TOKENS": "256",
            "SGLANG_NSA_ADAPTIVE_HISA_DECODE_CHUNK": "64",
            "SGLANG_NSA_ADAPTIVE_HISA_SKIP_CLEAN": "1",
            "SGLANG_NSA_ADAPTIVE_HISA_FUSE_PAGE_TABLE": "1",
            "SGLANG_NSA_ADAPTIVE_HISA_RAW_FP8_BUILDER": "1",
            "SGLANG_NSA_ADAPTIVE_HISA_DECODE_TIMING": "1" if profile else "0",
            "SGLANG_NSA_ADAPTIVE_HISA_SELECTOR_PROFILE": "1" if profile else "0",
            "SGLANG_NSA_ADAPTIVE_HISA_PARTITION_VALIDATE": "1",
            "SGLANG_NSA_ADAPTIVE_HISA_FORWARD_TIMING": "1" if profile else "0",
        }
    )
    return env


def server_command() -> list[str]:
    return [
        PYTHON,
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
        str(CONTEXT_LENGTH),
        "--trust-remote-code",
        "--reasoning-parser",
        "deepseek-v3",
        "--mem-fraction-static",
        "0.82",
        "--max-running-requests",
        "1",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--random-seed",
        "20260919",
        "--watchdog-timeout",
        "900",
        "--json-model-override-args",
        '{"use_hisa":false}',
    ]


def start_server(profile: bool) -> tuple[subprocess.Popen, Path, dict[str, str]]:
    global ACTIVE, SERVER_STREAM
    wait_for_free_gpus()
    if health():
        raise RuntimeError(f"port {PORT} is already serving; refusing to interfere")
    env = base_env(profile)
    command = server_command()
    label = "profile" if profile else "full"
    log_path = OUT / f"server_{label}.log"
    launch = {
        "cmd": command,
        "cwd": str(OUT),
        "profile": profile,
        "env": {
            key: value
            for key, value in env.items()
            if key == "PYTHONPATH"
            or key == "PYTHONUNBUFFERED"
            or key.startswith("SGLANG_")
        },
    }
    (OUT / f"launch_{label}.json").write_text(
        json.dumps(launch, indent=2), encoding="utf-8"
    )
    SERVER_STREAM = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        cwd=OUT,
        env=env,
        stdout=SERVER_STREAM,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    ACTIVE = proc
    (OUT / "server_pid.json").write_text(
        json.dumps({"pid": proc.pid, "profile": profile, "port": PORT}, indent=2),
        encoding="utf-8",
    )
    event("SERVER_START", pid=proc.pid, profile=profile, port=PORT)
    for _ in range(240):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup: {proc.returncode}")
        if health():
            break
        time.sleep(10)
    else:
        raise RuntimeError("server readiness timeout")

    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        logs = read_server_log(log_path)
        method = re.search(
            r"gpu kernels warm in [\d.]+s metric=(\w+) method=(\S+)", logs
        )
        if method:
            if method.group(1) != "key_sse" or method.group(2) != EXPECTED_METHOD:
                raise RuntimeError(
                    f"wrong method: metric={method.group(1)} method={method.group(2)}"
                )
            event(
                "METHOD_CONFIRMED",
                metric=method.group(1),
                method=method.group(2),
            )
            return proc, log_path, env
        time.sleep(5)
    raise RuntimeError("no metric/method warmup line in server log")


def stop_server() -> None:
    global ACTIVE, SERVER_STREAM
    proc = ACTIVE
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)
    if SERVER_STREAM is not None:
        SERVER_STREAM.close()
    ACTIVE = None
    SERVER_STREAM = None
    event("SERVER_STOPPED", gpu_pids=gpu_pids())


def generate(input_ids: list[int], max_new_tokens: int = 16) -> dict:
    payload = json.dumps(
        {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
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
        return json.load(response)


def synthetic_ids(length: int) -> list[int]:
    return [100 + (index * 7919) % 30000 for index in range(length)]


def smoke(log_path: Path, output_name: str = "smoke.json") -> dict:
    started = time.monotonic()
    output = generate(synthetic_ids(20000), 16)
    time.sleep(2)
    logs = read_server_log(log_path)
    built = re.findall(
        r"forward summary sync_built=(\d+) gpu_built=(\d+)", logs
    )
    bound = re.findall(
        r"decode bound req=\d+ layers=(\d+) eligible=(\d+) "
        r"candidate_tokens=\d+ sink=(\d+) tail=(\d+) decode_chunk=(\d+) "
        r"fallback_layers=(\d+)",
        logs,
    )
    errors = logs.count("Traceback") + len(
        re.findall(r"adaptive-hisa .*failed", logs)
    )
    record = {
        "wall_seconds": time.monotonic() - started,
        "completion_tokens": output.get("meta_info", {}).get("completion_tokens"),
        "forward_summaries": built[-8:],
        "decode_bound": bound[-8:],
        "tracebacks_or_failures": errors,
    }
    (OUT / output_name).write_text(json.dumps(record, indent=2), encoding="utf-8")
    if errors or not built or not bound:
        raise RuntimeError(f"smoke lacks required evidence: {record}")
    if any(int(sync) + int(gpu) < 60 for sync, gpu in built[-8:]):
        raise RuntimeError(f"smoke built fewer than 60 layers: {record}")
    last = bound[-1]
    if not (
        int(last[0]) == int(last[1]) >= 60
        and last[2:] == ("64", "256", "64", "0")
    ):
        raise RuntimeError(f"smoke decode binding mismatch: {last}")
    event("SMOKE_PASSED", **record)
    return record


def profile_requests(log_path: Path) -> None:
    records = []
    for length in (16384, 32768, 131072):
        before = len(read_server_log(log_path))
        started = time.monotonic()
        output = generate(synthetic_ids(length), 8)
        elapsed = time.monotonic() - started
        time.sleep(2)
        new_logs = read_server_log(log_path)[before:]
        summaries = re.findall(
            r"adaptive-hisa forward summary .*?builder_sums score_s=([\d.]+) "
            r"tree_s=([\d.]+) dp_s=([\d.]+) repair_s=([\d.]+) "
            r"merge_s=([\d.]+) summary_s=([\d.]+) total_s=([\d.]+)",
            new_logs,
        )
        methods = re.findall(
            r"metric=(\w+) method=(P-key-sync_nonoverlap-target64)", new_logs
        )
        selector_profiles = re.findall(
            r"adaptive-hisa selector profile req=\d+ (\{.*\})", new_logs
        )
        stage_timings = re.findall(
            r"adaptive-hisa stage timing req=\d+ (\{.*\})", new_logs
        )
        records.append(
            {
                "prompt_tokens": length,
                "wall_seconds": elapsed,
                "completion_tokens": output.get("meta_info", {}).get(
                    "completion_tokens"
                ),
                "forward_summary_timings": summaries,
                "method_lines": methods,
                "selector_profile": (
                    ast.literal_eval(selector_profiles[-1])
                    if selector_profiles
                    else None
                ),
                "decode_stage_timing": (
                    ast.literal_eval(stage_timings[-1])
                    if stage_timings
                    else None
                ),
                "tracebacks": new_logs.count("Traceback"),
            }
        )
        event("PROFILE_REQUEST", **records[-1])
    (OUT / "profile_requests.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


def run_evaluator(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    with log_path.open("a", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=EVALUATORS,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )


def longbench_command(
    output: Path, summary: Path, data: Path
) -> list[str]:
    return [
        PYTHON,
        str(EVALUATORS / "evaluate_longbench_v2_e2e.py"),
        "--model",
        MODEL,
        "--server",
        f"http://127.0.0.1:{PORT}",
        "--data",
        str(data),
        "--output",
        str(output),
        "--summary",
        str(summary),
        "--max-context-tokens",
        "131072",
        "--max-new-tokens",
        "128",
        "--concurrency",
        "1",
        "--resume",
    ]


def ruler_command(
    output: Path, summary: Path, data_root: Path, max_records: int | None = None
) -> list[str]:
    command = [
        PYTHON,
        str(EVALUATORS / "evaluate_ruler_e2e.py"),
        "--model",
        MODEL,
        "--server",
        f"http://127.0.0.1:{PORT}",
        "--data-root",
        str(data_root),
        "--output",
        str(output),
        "--summary",
        str(summary),
        "--lengths",
        "32k",
        "128k",
        "--all-records",
        "--resume",
    ]
    if max_records is not None:
        command += ["--max-records", str(max_records)]
    return command


def run_pilot(env: dict[str, str]) -> None:
    pilot = OUT / "pilot"
    pilot.mkdir(exist_ok=True)
    run_evaluator(
        longbench_command(
            pilot / "longbench_predictions.jsonl",
            pilot / "longbench_summary.json",
            BASE / "results_jump_exact/longbench_small_pilot.json",
        ),
        pilot / "longbench.log",
        env,
    )
    run_evaluator(
        ruler_command(
            pilot / "ruler_predictions.jsonl",
            pilot / "ruler_summary.json",
            BASE / "results_jump_exact/ruler_pilot",
            max_records=3,
        ),
        pilot / "ruler.log",
        env,
    )
    for name in ("longbench", "ruler"):
        summary = json.loads(
            (pilot / f"{name}_summary.json").read_text(encoding="utf-8")
        )
        if summary["examples"] != summary["successful"]:
            raise RuntimeError(f"{name} pilot failed: {summary}")
    event("PILOT_PASSED")


def run_full(env: dict[str, str]) -> None:
    arm = OUT / "pkey_t64"
    arm.mkdir(exist_ok=True)
    status(state="evaluating_longbench", arm="pkey_t64")
    run_evaluator(
        longbench_command(
            arm / "longbench_predictions.jsonl",
            arm / "longbench_summary.json",
            HELD / "longbench_heldout.json",
        ),
        arm / "longbench.log",
        env,
    )
    longbench = json.loads(
        (arm / "longbench_summary.json").read_text(encoding="utf-8")
    )
    if longbench["examples"] != 401 or longbench["successful"] != 401:
        raise RuntimeError(f"LongBench incomplete: {longbench}")
    compare()
    event("EVAL_DONE", bench="longbench_v2", summary=longbench)

    status(state="evaluating_ruler", arm="pkey_t64")
    run_evaluator(
        ruler_command(
            arm / "ruler_predictions.jsonl",
            arm / "ruler_summary.json",
            HELD / "ruler",
        ),
        arm / "ruler.log",
        env,
    )
    ruler = json.loads((arm / "ruler_summary.json").read_text(encoding="utf-8"))
    if ruler["examples"] != 364 or ruler["successful"] != 364:
        raise RuntimeError(f"RULER incomplete: {ruler}")
    compare()
    event("EVAL_DONE", bench="ruler", summary=ruler)


def rows(path: Path) -> dict[str, dict]:
    output = {}
    if not path.exists():
        return output
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            output[str(row["source_id"])] = row
    return output


def grouped(
    arm: dict[str, dict],
    official: dict[str, dict],
    common: list[str],
    fields: tuple[str, ...],
) -> dict:
    groups: dict[tuple[str, ...], list[str]] = {}
    for source_id in common:
        key = tuple(str(arm[source_id][field]) for field in fields)
        groups.setdefault(key, []).append(source_id)
    output = {}
    for key, keys in sorted(groups.items()):
        arm_accuracy = sum(float(arm[k]["score"]) for k in keys) / len(keys)
        official_accuracy = sum(float(official[k]["score"]) for k in keys) / len(keys)
        output["|".join(key)] = {
            "n": len(keys),
            "accuracy_pkey_target64": arm_accuracy,
            "accuracy_official": official_accuracy,
            "delta_pp": 100 * (arm_accuracy - official_accuracy),
        }
    return output


def compare() -> dict:
    report = {"updated_utc": now(), "benches": {}}
    for bench, filename in (
        ("longbench_v2", "longbench_predictions.jsonl"),
        ("ruler", "ruler_predictions.jsonl"),
    ):
        arm = rows(OUT / "pkey_t64" / filename)
        official = rows(OFFICIAL / filename)
        common = [
            key
            for key in sorted(arm.keys() & official.keys())
            if arm[key].get("status") == official[key].get("status") == "ok"
        ]
        entry = {
            "rows_pkey_target64": len(arm),
            "rows_official": len(official),
            "matched": len(common),
        }
        if common:
            pkey_accuracy = sum(float(arm[k]["score"]) for k in common) / len(common)
            official_accuracy = sum(float(official[k]["score"]) for k in common) / len(
                common
            )
            entry.update(
                accuracy_pkey_target64=pkey_accuracy,
                accuracy_official=official_accuracy,
                delta_pp=100 * (pkey_accuracy - official_accuracy),
                pkey_wins=sum(
                    float(arm[k]["score"]) > float(official[k]["score"])
                    for k in common
                ),
                official_wins=sum(
                    float(arm[k]["score"]) < float(official[k]["score"])
                    for k in common
                ),
            )
            if bench == "longbench_v2":
                entry["by_domain"] = grouped(
                    arm, official, common, ("domain",)
                )
                entry["by_difficulty_length"] = grouped(
                    arm, official, common, ("difficulty", "length")
                )
            else:
                entry["by_length"] = grouped(
                    arm, official, common, ("length",)
                )
                entry["by_task"] = grouped(arm, official, common, ("task",))
                entry["by_task_length"] = grouped(
                    arm, official, common, ("task", "length")
                )
        report["benches"][bench] = entry
    (OUT / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def audit(log_path: Path, label: str) -> None:
    logs = read_server_log(log_path)
    data = {
        "tracebacks": logs.count("Traceback"),
        "adaptive_failed": len(re.findall(r"adaptive-hisa .*failed", logs)),
        "forward_summaries": logs.count("adaptive-hisa forward summary"),
        "decode_bound": logs.count("decode bound"),
        "skip_reasons": dict(
            Counter(
                re.findall(
                    r"skipped prefill partition layer=\d+ req=\d+ reason=(\w+)",
                    logs,
                )
            )
        ),
        "stale_dropped": logs.count("dropped stale partition"),
    }
    (OUT / f"server_audit_{label}.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )
    event("SERVER_AUDIT", label=label, **data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("profile", "full", "compare"), required=True
    )
    parser.add_argument("--refreeze", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.phase == "compare":
        print(json.dumps(compare(), indent=2, ensure_ascii=False))
        return

    freeze_source(refreeze=args.refreeze)
    profile = args.phase == "profile"
    label = "profile" if profile else "full"
    proc = None
    log_path = OUT / f"server_{label}.log"
    try:
        status(state="starting_server", phase=args.phase)
        proc, log_path, env = start_server(profile=profile)
        smoke(log_path, output_name=f"smoke_{label}.json")
        if profile:
            status(state="profiling")
            profile_requests(log_path)
            audit(log_path, label)
            status(state="profile_completed")
        else:
            status(state="running_pilot")
            run_pilot(env)
            run_full(env)
            audit(log_path, label)
            status(state="all_completed")
    except BaseException as error:
        status(state="failed", phase=args.phase, error=repr(error))
        event("FAILED", phase=args.phase, error=repr(error))
        raise
    finally:
        if proc is not None:
            stop_server()


if __name__ == "__main__":
    main()
