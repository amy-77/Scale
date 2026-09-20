#!/usr/bin/env python3
"""Wait for an owned experiment and idle GPUs, then profile the selector."""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


POLL_SECONDS = 30
IDLE_POLLS_REQUIRED = 4
RUNNER = Path(__file__).with_name("run_pkey_target64_jump.py")
OUT = Path("/data/jz/9-57-0920/results_selector_profile_chunk64_20260920")
STATUS = OUT / "wait_status.json"
EVENTS = OUT / "wait_events.jsonl"


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def event(name: str, **values: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    record = {"event": name, "time": now(), **values}
    with EVENTS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    STATUS.write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False), flush=True)


def process_start_time(pid: int) -> str | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(
            ")", 1
        )[1].split()
        return fields[19]
    except (FileNotFoundError, IndexError, PermissionError, ProcessLookupError):
        return None


def gpu_pids() -> list[int] | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        event("GPU_QUERY_FAILED", error=repr(error))
        return None
    return sorted(
        {
            int(line.strip())
            for line in output.splitlines()
            if line.strip().isdigit()
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch-pid", type=int, required=True)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "waiter.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUT / "waiter.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")

    watched_start = process_start_time(args.watch_pid)
    event(
        "WAITING_FOR_PID",
        watch_pid=args.watch_pid,
        watch_start_time=watched_start,
        policy="never signal other processes",
    )
    while watched_start is not None:
        current_start = process_start_time(args.watch_pid)
        if current_start is None or current_start != watched_start:
            break
        time.sleep(POLL_SECONDS)
    event("WATCH_PID_EXITED", watch_pid=args.watch_pid)

    idle_polls = 0
    last_busy: list[int] | None = None
    while idle_polls < IDLE_POLLS_REQUIRED:
        busy = gpu_pids()
        if busy is None:
            idle_polls = 0
        elif busy:
            idle_polls = 0
            if busy != last_busy:
                event("WAITING_FOR_ALL_GPUS", process_ids=busy)
            last_busy = busy
        else:
            idle_polls += 1
            event(
                "GPU_IDLE_POLL",
                idle_polls=idle_polls,
                required=IDLE_POLLS_REQUIRED,
            )
        time.sleep(POLL_SECONDS)

    command = ["/usr/bin/python3", str(RUNNER), "--phase", "profile"]
    env = os.environ.copy()
    env["ADAPTIVE_HISA_RUN_OUT"] = str(OUT)
    env["ADAPTIVE_HISA_RUN_PORT"] = "31923"
    event(
        "LAUNCHING_SELECTOR_PROFILE",
        command=command,
        output=str(OUT),
        port=31923,
    )
    completed = subprocess.run(command, env=env, check=False)
    event("SELECTOR_PROFILE_EXITED", returncode=completed.returncode)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
