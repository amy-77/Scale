#!/usr/bin/env python3
"""Run a fresh P-key target64 RULER evaluation with decode chunk 64 only."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path

import run_pkey_target64_jump as core


OUT = Path("/data/jz/9-57-0920/results_pkey_target64_chunk64_ruler_20260920")


def configure_output() -> None:
    core.OUT = OUT
    core.SOURCE = OUT / "source"
    core.EVALUATORS = OUT / "evaluators"


def main() -> None:
    configure_output()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    proc = None
    log_path = OUT / "server_full.log"
    try:
        # The source is frozen when the waiter is installed, before any
        # unvalidated selector optimisation is edited into the package.
        core.freeze_source(refreeze=False)
        core.status(state="starting_server", phase="ruler_chunk64")
        proc, log_path, env = core.start_server(profile=False)
        core.smoke(log_path, output_name="smoke_ruler.json")

        arm = OUT / "pkey_t64"
        arm.mkdir(exist_ok=True)
        core.status(
            state="evaluating_ruler",
            arm="pkey_t64",
            decode_chunk=64,
            fresh_run=True,
        )
        core.run_evaluator(
            core.ruler_command(
                arm / "ruler_predictions.jsonl",
                arm / "ruler_summary.json",
                core.HELD / "ruler",
            ),
            arm / "ruler.log",
            env,
        )
        summary = json.loads(
            (arm / "ruler_summary.json").read_text(encoding="utf-8")
        )
        if summary["examples"] != 364 or summary["successful"] != 364:
            raise RuntimeError(f"RULER incomplete: {summary}")

        comparison = core.compare()
        core.event("EVAL_DONE", bench="ruler", decode_chunk=64, summary=summary)
        core.audit(log_path, "ruler")
        core.status(
            state="all_completed",
            phase="ruler_chunk64",
            decode_chunk=64,
            summary=summary,
            comparison=comparison["benches"].get("ruler", {}),
        )
    except BaseException as error:
        core.status(
            state="failed",
            phase="ruler_chunk64",
            decode_chunk=64,
            error=repr(error),
        )
        core.event("FAILED", phase="ruler_chunk64", error=repr(error))
        raise
    finally:
        if proc is not None:
            core.stop_server()


if __name__ == "__main__":
    main()
