# adaptive_0921_h202

Derived from `adaptive_0921_h201` on 2026-09-21 05:00 (frozen copy of that state:
`adaptive_0921_h201_backup_20260921_0500/`). h201 is left untouched from here on;
all further kernel work and the adaptive sparse prefill land in this tree.

State inherited from h201 at fork time (all bit-identical to the 09-20 snapshot on
5 synthetic + 5 real-dump configs):
- tree kernels: ATOM=1 level skips SSE, `* (1/cnt)` instead of `/ cnt`
- `_leaf_totals_fp8_kernel`: full-row tiles (3.3x)
- merge: row means travel with rows (`row_means`, compact kernel writes means),
  cost kernel no longer divides; idle-round device early exits in match/index/
  compact and in the CUDA radix selects
- λ-DP: KC=8, num_warps=1 (1.7x); `lambda_max_rounds=6` default
- 128K single layer partition build: 10.4 -> 4.3 ms GPU time

## 2026-09-21 (evening) — sparse prefill evaluated

- `prefill_select.py`: attention-sink leaves forced first in the coarse ranking; per-chunk leaf budget
  threaded through admission (`sparse_prefill_admission` returns the budget).
- `config.py`: `SPARSE_PREFILL_CANDIDATES`, `SPARSE_PREFILL_FINAL_CANDIDATES`, `SPARSE_PREFILL_DENSE_FINAL`.
- `runner/run_sparse_prefill_e2e.py`: `--arm {sparse_prefill,_c16k,_c32k,_final32k,_dense_final}`, `--quick-niah`,
  per-arm frozen source / comparison.
- e2e of the all-chunks-sparse arm done: LongBench 0.479 (h201 0.501), RULER 0.727 (h201 0.853; 128k NIAH collapse).
  `dense_final` quick check: 128k niah_multikey 0.71/0.93/0.93. See README.md and
  `delta/docs_research/adaptive_hisa_sparse_prefill.md`.

## 2026-09-21 (night) — prefill indexer speed work (128K TTFT 47.13 -> 45.44 s, TPOT unchanged)

All three changes are output-identical to the previous version (unit-tested `torch.equal` / bitwise).

- Coarse selection: `select_candidates_weighted` + new `csrc/prefill_weighted_select.cu` — batched
  token-weighted radix select (one CTA per query row, the decode `weighted_select` algorithm) replaces
  `argsort + cumsum + per-slot binary search`; 2.0 -> 0.5 ms per chunk (4.1-4.7x). Legacy path:
  `SGLANG_NSA_ADAPTIVE_HISA_PREFILL_SELECT=argsort`. TTFT 47.25 -> 46.12 s.
- Fine scorer: `prefill_select._fine_scores` uses HISA's persistent `block_sparse_mqa_triton` with the
  new `kv_block_size=1` (token-id candidates, -1 padding) over the flat index-K when `k_flat` is given;
  G=64/KC=32/4 warps, 249 TFLOPS on H20 (84% of fp8 peak). 5.4 -> 4.4 ms per chunk. Measured first:
  HISA's own K=64/GROUP_SIZE=4 kernel on the same 8192 tokens is *slower* (5.58 ms) — the fine stage is
  compute-bound, contiguity buys nothing. TTFT 46.12 -> 45.32 s.
- Incremental Key-SSE tree across chunks (`config.tree_cache`, env `TREE_CACHE`, default on):
  `partition_kernels.raw_fp8_tree_triton_incremental` copies the sealed prefix's levels
  (`_tree_copy_levels_kernel`) and runs `_fp8_chunk_tree_kernel` / `_fp8_upper_tree_kernel` for the new
  roots only (new `c0` / `r0` offsets); `prefill_runtime._TreeCache` keeps the previous chunk's tree per
  (layer, req, epoch), dropped after the final chunk. Bitwise identical to the full build; 1.19 -> 0.10 ms
  at 128K, not visible end-to-end (build is on the side stream; merge dominates it at ~2.7 ms).
- Stage costs per layer per chunk at 128K (H20): coarse 1.3 + select 0.5 + fine 4.4 + local 2.4 +
  cat/topk 2.1 ≈ 10.7 ms; partition build 4.7 ms GPU on the side stream. Details, microbenchmarks and
  e2e tables: `delta/docs_research/speed_bench_128k_20260921.md`, scripts in `runner/speed_bench_20260921/`.
- Tests: `test_adaptive_hisa_sparse_prefill.py` (+`TestWeightedSelect`, flat-vs-paged fine parity),
  `test_adaptive_hisa_prefill_partition.py` (+`test_incremental_tree_is_bitwise_identical`,
  `test_tree_cache_epoch_and_final`); 61 passed.
