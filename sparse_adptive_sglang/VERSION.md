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
