# SCALE relaxed λ-DP

Measured on 8×H20, TP=8, 128K input / 128 output, CUDA Graph BS=1, 8K chunked prefill, 1 warmup + 3 timed requests:

**TTFT 38.92 s，E2E 41.11 s，TPOT 17.20 ms**

This is a speed point, not a quality-qualified serving configuration. Same-stack HISA-64 is 35.65 s / 17.55 ms; exact global λ-DP (S32/M128) is 40.10 s / 17.15 ms.

Derived from the original [`sparse_adptive_sglang`](https://github.com/amy-77/Scale/tree/main/sparse_adptive_sglang) tree (`2c38146`, split L/32 → exact λ-DP → merge L/128).

## Changes versus the original tree

The original path still exists (`RELAXED_SPLIT=0`). This directory turns on a relaxed global split/merge that **keeps the same algorithm family** (build Key-SSE tree → λ-DP split → Ward merge) but drops the exact-count contract.

1. **Shorter λ search.** `LAMBDA_MAX_ROUNDS` 6 → 2. The DP no longer spends extra bracket rounds forcing a near-exact leaf count.
2. **No exact repair.** Relaxed split skips the gain-ordered repair that padded/trimmed leaves to exactly `N/32`.
3. **Softer merge.** Two threshold Ward rounds (`SOFT_MERGE_ROUNDS=2`, `MERGE_ALPHA=2.0`) plus one target-count fallback (`MERGE_TARGET_ROUNDS=1`). Logical leaf count may sit slightly under capacity; the physical tensor shape stays graph-stable.
4. **Still global, still atom=1.** This is not root-local. Boundaries can still cross 256-token roots during merge. CUDA Graph only needs a fixed *capacity*, not a fixed logical chunk count.
5. **All-sparse prefill.** Intermediate and final prompt chunks use the adaptive indexer (`DENSE_FINAL=0`).
6. **Shared later kernels** inherited after the original snapshot:
   - incremental Key-SSE tree (`TREE_CACHE=1`): only new 256-token roots are rebuilt
   - batched weighted radix select for prefill
   - HISA persistent K=1 fine scorer
   - decode `weighted_select` fast path
   - side-stream partition build

Offline dump recall (3504 queries, 219 partitions) stayed at 0.6241 → 0.6245 versus exact λ-DP. Downstream RULER / LongBench for this exact arm is still running and is **not** attached to these timing numbers.

## How to run

```bash
cd SCALE_relaxed
source runner/deliverable_speed.env
export PYTHONPATH="$PWD/python"
python -m sglang.launch_server \
  --model-path <DeepSeek-V3.2> \
  --tp-size 8 --chunked-prefill-size 8192 --cuda-graph-bs 1
```

Host-native A/B harness: `runner/run_host_ttft_ab.py --arms global_relaxed`.
