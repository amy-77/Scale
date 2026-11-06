# Current experiment snapshot: 2026-11-07

For dataset preparation and single-configuration runs on another machine, see
the [quick start](../../scripts/README.md).

The running SCALE implementation is `SCALE_relaxed/`, configured as relaxed
global Key-SSE split S8 / merge target M64. It is not the `Root-local/` variant.
The serving source was already published: server commit `b3106dc` and GitHub
commit `70b5a2c` share tree `c70d12523100e587406c35229f05206138d335bb`.
This update publishes the previously untracked experiment runners and their
evaluation dependencies. `provenance.json` records the original script hashes.
Only deployment paths/ports were made configurable in those runners; the SCALE
serving implementation and experiment parameters are unchanged.

## Configurations and measurement scope

| Runner | Work |
| --- | --- |
| `SCALE_relaxed/runner/run_attribution_suite.py` | SCALE-64 and HISA-64 with dense-final on/off; HISA block/candidate latency sweep; RULER quality for the factorial and selected timing point |
| `SCALE_relaxed/runner/run_fine_blocks.py` | HISA-16 then HISA-8, C=8192, dense-final off; 128K then 32K RULER, resumable |
| `SCALE_relaxed/runner/critical_token_recall.py` | Offline replay of candidate/gold-span coverage, separate from end-to-end generation |

SCALE uses S8/M64, two lambda rounds without exact repair, two soft merge
rounds, and one target-count round. Candidate budget is 8192 and final Top-K
is 2048. Both dense-final arms are explicit in `arms.json`. Dense-final retains
flat DSA scoring on the last prompt chunk; it does not turn all attention dense.
The old `deliverable_speed.env` uses S32/M128 and is a different operating point.

RULER uses the existing frozen subset: 13 tasks x 14 prompts x two lengths
(32K and 128K), 364 examples per configuration. `--all-records` evaluates all
records in those files, not just the old collector's test index 7.
`ruler_manifest.json` provides file hashes; dataset prompts are not included.
The original generator is NVIDIA/RULER commit `c3f5e3b`, seed `20260823`.
Each task originally had 20 rows; the paper retains zero-based rows `[6:20]`.
All 26 frozen files were verified byte-for-byte against that source slice.
The model/tokenizer is DeepSeek-V3.2 at the revision in `provenance.json`.

The fine-block deployment measures quality only because other GPU services
coexist on that host. Its elapsed times are diagnostic. The attribution runner
measures 128K input / 128 output, one warmup and three requests. Its `choose_iso`
selects a nearby measured point against SCALE+dense-final; this is not a complete
matched-quality/latency frontier and does not isolate partitioning across policies.
The 4-token kernel configuration is not implemented by this snapshot.

## Prepare the fixed-block baseline

The current baseline has changes relative to the older `sglang/` snapshot.
Create a separate checkout directory and apply the supplied patch there:

```bash
cp -a sglang HISA_current
git -C HISA_current init -q
git -C HISA_current apply --check ../experiments/20261107/hisa_current.patch
git -C HISA_current apply ../experiments/20261107/hisa_current.patch
export HISA_SOURCE="$PWD/HISA_current"
```

The patch reconstructs the Python source used by the current baseline; the
tree fingerprint is recorded in `provenance.json`. Use a Linux filesystem that
distinguishes `README.md` from `readme.md` when cloning this historical repository.
Model weights, generated datasets and GPU/Python environments must be supplied
separately. The fine-block host environment is recorded in provenance; it does
not imply identical latency across environments.

## Run

Use an environment with the required serving dependencies and eight available
GPUs. From the repository root, set paths to the existing model and frozen data:

```bash
export MODEL_PATH=/models/DeepSeek-V3.2
export RULER_DATA_ROOT=/datasets/ruler
export HISA_SOURCE="$PWD/HISA_current"
export SCALE_RESULTS="$PWD/results/attribution_20261107"
python SCALE_relaxed/runner/run_attribution_suite.py
```

This runs the full attribution suite and waits for idle GPUs. For the separate
fine-block quality queue, use a separate allocation and output directory:

```bash
export SCALE_FINE_ROOT="$PWD/results/fine_blocks_20261107"
python SCALE_relaxed/runner/run_fine_blocks.py
```

For a SCALE-only launch, source the current preset and choose dense-final:

```bash
source experiments/20261107/scale_s8_m64.env
export SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_DENSE_FINAL=0  # or 1
export PYTHONPATH="$PWD/SCALE_relaxed/python"
python -m sglang.launch_server --model-path "$MODEL_PATH" \
  --served-model-name deepseek-v3.2 --tp-size 8 --host 127.0.0.1 --port 31946 \
  --context-length 163840 --trust-remote-code --reasoning-parser deepseek-v3 \
  --chunked-prefill-size 8192 --mem-fraction-static 0.82 \
  --max-running-requests 1 --disable-radix-cache --random-seed 20260921 \
  --watchdog-timeout 900 --cuda-graph-bs 1
```

Then run the included evaluator:

```bash
python SCALE_relaxed/runner/evaluators/evaluate_ruler_e2e.py \
  --data-root "$RULER_DATA_ROOT" --model "$MODEL_PATH" \
  --server http://127.0.0.1:31946 --lengths 32k 128k --all-records --resume \
  --output ruler_predictions.jsonl --summary ruler_summary.json
```

Offline critical-token analysis additionally needs `SCALE_REPLAY_ROWS` and
`SCALE_REPLAY_INVENTORY`. Set `SCALE_REPLAY_OLD_PREFIX` and
`SCALE_REPLAY_NEW_PREFIX` if paths in the supplied replay inventory were moved.
Replay inputs are not included and this analysis is not the serving algorithm.
