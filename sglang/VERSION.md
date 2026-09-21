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

## 2026-09-22 — decode selector (`csrc/weighted_select.cu`) fast path: 38.8 -> 14.2 µs at n=2048

Reference: `Adaptive_HISA_WeightedSelect_Warp优化_Cursor修改指南.pdf` (warp-aggregated histogram atomics +
ballot compaction of the threshold bucket). Selection contract, host API, fallback kernel and final
logical scan unchanged; everything below is bit-exact against the six-pass legacy kernel
(`test_weighted_selector_fast_path_bit_exact_sweep`: n = 0/1/31/32/33/1023/1024/1025/4096, ties, ±0,
denormal/huge, zero/guard-clipped/crossing leaves, budget < guards and > everything, 60 random
production-shape cases, CUDA-graph replay).

- Implemented the guide's two warp optimisations behind the compile-time bitmask
  `ADAPTIVE_HISA_WS_WARP_OPT` (env `ADAPTIVE_HISA_SELECT_WARP_OPT`: 1 = Phase A `__match_any_sync` +
  `__reduce_add_sync` histogram, 2 = Phase B ballot compaction, 4 = Phase C histograms; sm70 shuffle
  fallback). Measured on H20 they are **not** a win: Phase A +2.6 µs, Phase C +1.7 µs, Phase B ±0
  (n=1024; worse at 4096). Shared atomics on ~10-25 hot bins are cheaper than `__match_any_sync` for a
  single 1024-thread CTA; the guide's own risk list anticipated this. Default is 0 (per-leaf atomics).
- Profiling showed the actual hot spot was the *single-thread* 256-bin descending scan that picks the
  threshold byte, executed 4 times (Phase A + three Phase C passes): ~3.5 µs each of dependent shared
  loads. Replaced by `warp_threshold_bin` (one warp, 8 bins/lane, shuffle prefix + ballot); same
  result, prefix kernel 22.4 -> 8.9 µs (n=1024), 25.4 -> 14.4 µs (n=4096).
- `expand_selected_kernel`: each warp inspected leaves one at a time (n/32 dependent global round
  trips per warp). Now 32 leaves per warp per round with coalesced loads and ballot-driven interval
  writes: 15.3 -> 8.5 µs (n=2048), 27.6 -> 8.6 µs (n=4096), independent of n.
- `runner/bench_weighted_select.py`: `--scores {quantized,relu}` (positive, concentrated scores like the
  real coarse output). Full selector (prefix + expand), adaptive_real lengths, median µs:
  n=1024 31.3 -> 11.5, n=2048 38.8 -> 14.2, n=4096 52.9 -> 19.3 (HISA `fast_topk` on the same n: 10.1).
- End-to-end (128K / 128 out, graph, 1 warmup + 3 reps, `adaptive_sp_v4_graph`): TPOT 19.16 -> **18.10 ms**
  (18.09 / 18.10 / 18.12; -1.06 ms, matches 61 layers x ~20 µs), TTFT 45.44 -> 45.33 s (noise; prefill
  untouched). DSA 20.3 ms, HISA-64 17.85 ms.

## 2026-09-22 (01:00) — prefill Top-K without torch.cat (128K TTFT 45.33 -> 44.66 s); chunk 4096 measured

- `hisa/csrc/hisa_topk_fused.cu`: `SplitRow` + templated `fast_topk_cuda_tl`; new op `topk_candidates_split`
  (`hisa_topk_candidates_split`) selects Top-2048 over [leaf logits | local-window logits] read in place.
  `prefill_select.sparse_topk_core` no longer concatenates logits/candidate ids or pads the window:
  2.09 -> 1.27 ms per layer per chunk (`fine_scorer_stage_bench.py`). Redundant `.contiguous()` on row
  slices removed (no-ops). Test: `test_split_topk_equals_fused_topk_on_concatenation` (== fused-on-cat and
  == exact torch.topk, sets; HISA's radix kernel resolves threshold ties with atomics so order is unspecified).
- e2e: TTFT 45.33 -> 44.66 s (3 reps 44.83 / 44.65 / 44.66), TPOT 18.07 ms.
- `--chunked-prefill-size 4096` with the same code: TTFT 50.05 s (+5.4 s). Per-chunk prefill time is ~flat in
  the prefix length (2.85 s per 8192 chunk, 1.5 s per 4096 chunk), so halving the chunk doubles the per-chunk
  fixed cost (61 x 4.7 ms partition build per chunk ~= 0.29 s of the +0.34 s per extra chunk). Not adopted.
