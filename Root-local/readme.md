# SCALE root-local fused

Measured on 8×H20, TP=8, 128K input / 128 output, CUDA Graph BS=1, 8K chunked prefill, 1 warmup + 3 timed requests:

**TTFT 36.95 s，E2E 39.13 s，TPOT 17.17 ms**

This is a speed point, not a quality-qualified serving configuration. Same-stack HISA-64 is 35.65 s / 17.55 ms, so prefill is still ~1.3 s slower and decode is ~0.37 ms/token faster.

Derived from the original [`sparse_adptive_sglang`](https://github.com/amy-77/Scale/tree/main/sparse_adptive_sglang) tree (`2c38146`, global split L/32 → exact λ-DP → merge L/128).

## Changes versus the original tree

This path **does not** run λ search, global DP, exact repair, or Ward merge. It replaces that serial pipeline with a per-root local cut plus fused summaries.

1. **Root-local partition.** Every sealed 256-token root independently picks one 8-token-aligned cut that maximizes Key-SSE gain, producing exactly two leaves per root (`N/128` leaves, no cross-root merge).
2. **No λ / DP / repair / merge.** The original global `search_lambda` → `repair_leaves` → `merge_sync_nonoverlap` path is skipped (`ROOT_LOCAL_PARTITION=1`).
3. **Fused FP8 summaries.** `ROOT_LOCAL_FUSED_SUMMARIES=1` reuses HISA 8-token grouped FP8 mean-pooling and writes the two leaf summaries in the same kernel, instead of scanning raw-K leaf totals and launching a separate requant.
4. **Incremental Key-SSE tree.** Across 8K prefill chunks, only newly completed roots are built (`TREE_CACHE=1`, 1.19 → 0.10 ms at 128K).
5. **All-sparse prefill.** Same two-level indexer on intermediate and final chunks (`DENSE_FINAL=0`), plus the later weighted-select / K=1 fine-scorer / decode fast-path kernels.

Microbenchmarks (single layer, H20):

| stage | original global | this path |
|---|---|---|
| 128K partition + summary | 3.459 ms | 0.248 ms (fused) |
| 128K Key-SSE tree | 1.19 ms (full) | 0.10 ms (new roots) |

Offline dump recall versus exact root-local summaries: 0.6189 → 0.6187 (−0.017 pp on 3504 queries). Versus original global λ-DP the partition itself is a different approximation (atom 8, no cross-root merge); do not treat it as equivalent to the paper's global atom-1 DP.

## How to run

```bash
cd Root-local
source runner/deliverable_speed.env
export PYTHONPATH="$PWD/python"
python -m sglang.launch_server \
  --model-path <DeepSeek-V3.2> \
  --tp-size 8 --chunked-prefill-size 8192 --cuda-graph-bs 1
```

Host-native A/B harness: `runner/run_host_ttft_ab.py --arms rootlocal_fused`.
