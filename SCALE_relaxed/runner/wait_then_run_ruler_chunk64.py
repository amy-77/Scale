#!/usr/bin/env python3
"""Wait for foreign GPU work to finish, then launch the chunk64 RULER run.

This watcher never signals another process.  It first waits for WATCH_PID to
exit, then requires every visible GPU to have no compute processes for several
consecutive polls.  The experiment runner repeats the all-GPU-free check before
starting SGLang, closing the race between this watcher and server startup.
"""

from __future__ import annotations

import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


WATCH_PID = 593576
POLL_SECONDS = 30
IDLE_POLLS_REQUIRED = 4
RUNNER = Path(__file__).with_name("run_ruler_chunk64_only.py")
OUT = Path("/data/jz/9-57-0920/results_pkey_target64_chunk64_ruler_20260920")
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
        # Field 22 is the kernel start time. Account for "(comm with spaces)".
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
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "waiter.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUT / "waiter.pid").write_text(f"{os.getpid()}\n", encoding="utf-8")

    watched_start = process_start_time(WATCH_PID)
    event(
        "WAITING_FOR_PID",
        watch_pid=WATCH_PID,
        watch_start_time=watched_start,
        policy="never signal foreign processes",
    )
    while watched_start is not None:
        current_start = process_start_time(WATCH_PID)
        if current_start is None or current_start != watched_start:
            break
        time.sleep(POLL_SECONDS)
    event("WATCH_PID_EXITED", watch_pid=WATCH_PID)

    idle_polls = 0
    last_busy: list[int] | None = None
    while idle_polls < IDLE_POLLS_REQUIRED:
        busy = gpu_pids()
        if busy is None:
            idle_polls = 0
        elif busy:
            idle_polls = 0
            if busy != last_busy:
                event("WAITING_FOR_ALL_GPUS", foreign_pids=busy)
            last_busy = busy
        else:
            idle_polls += 1
            event(
                "GPU_IDLE_POLL",
                idle_polls=idle_polls,
                required=IDLE_POLLS_REQUIRED,
            )
        time.sleep(POLL_SECONDS)

    command = ["/usr/bin/python3", str(RUNNER)]
    event("LAUNCHING_RULER", command=command, decode_chunk=64)
    completed = subprocess.run(command, check=False)
    event("RULER_RUN_EXITED", returncode=completed.returncode)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
