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

## 2026-09-23 — split32/merge128、无 dense-final 的 Prefill 热路径清理

- 确认当前 production scorer 已经复用 HISA 的 Tensor-Core 结构：
  coarse 是 batched `deep_gemm.fp8_mqa_logits`，fine 是 HISA persistent K=1
  kernel；K=64/segment 版本在 H20 的实测反而更慢，因此未机械照搬指南。
- `prefill_weighted_select.cu` 在把 leaf score/length 装入 shared memory 时
  同步完成 sink leaf 的 `+inf` promotion；padding 本来就由 `num_leaves` /
  `leaf_len` 忽略。`coarse_leaf_scores(mask_metadata=False)` 因此只保留
  Tensor-Core scorer，不再对整张 `[query, leaf]` 矩阵追加两次 metadata
  mask 写回。候选 token 与旧路径逐元素相同（包括 ties、padding、local window）。
- `sparse_topk_core` 复用一份 query-start zero buffer 和一份 `num_leaves`
  i32 标量；Top-K 直接写最终 `out[rows]`，不再为每个 2048-row 子批分配
  临时 `[rows,2048]` 后做 D2D copy；valid-length 也移出子批循环。
- H20，128K 最后一个 chunk，split L/32 -> merge L/128（960 leaves）：
  coarse+metadata 0.637 ms -> pure coarse 0.519 ms（-0.118 ms）；
  split Top-K 临时输出+copy 1.244 ms -> direct output 1.200 ms（-0.044 ms）。
  这只是单层单 chunk 微基准，未把它写成端到端 TTFT 结果。
- 新增 opt-in `ROOT_LOCAL_PARTITION=1`：每个 256-token root 独立在 8-token
  边界上选择 Key-SSE gain 最大的二段切分，直接得到 L/128 动态 leaves，
  消除 global λ-DP / repair / merge 串行路径；原 split32/merge128 exact
  global contract 仍由 `ROOT_LOCAL_PARTITION=0` 保留。
- H20 128K 单层 partition+summary 微基准：global 3.459 ms，root-local
  0.503 ms（6.9x）；root-local coverage CUDA 单测通过。真实 RULER dump
  的 18 个 query 样本上，dense-local Top-2048 recall 平均值未下降
  （global 0.1635，root-local 0.2635；该小样本仅作退化检查）。
- `SPARSE_PREFILL_ROWS=8192` 与 root-local、dense-local 组合的 8 卡
  128K/128-out 实测 TTFT 为 37.1113 / 37.0105 / 37.0227 s（中位数
  **37.0227 s**），TPOT 均约 **17.2 ms**；相对 40.399 s 基线 TTFT
  降低 3.38 s，达到低于 38 s 的目标。
- `runner/deliverable_speed.env` 和 `run_sparse_prefill_e2e.py` 默认 arm
  现在显式为 sparse prefill、split32/merge128、`DENSE_FINAL=0`。
- 验证：完整 sparse-prefill suite 11 passed；selector 独立 parity 覆盖
  1..4096 leaves、ties、padding、budget 超界和 local window。

## 2026-09-23 — relaxed split/merge 与 root-local summary fusion

- 新增 opt-in relaxed global 路径：λ 搜索最多 2 轮、跳过 exact repair，
  先做 2 轮 threshold Ward merge，再用固定容量 fallback 保证物理输出
  shape；logical leaf count 保持动态，因此 CUDA Graph 的地址和 launch
  sequence 不变。32K/64K/128K 单层 partition 中位数分别
  2.848→1.824 ms、2.844→1.847 ms、3.026→2.145 ms（-29%～-36%）。
- relaxed 路径在三层完整 real dump（219 partitions / 3504 queries）上，
  exact recall 0.624122，relaxed recall 0.624530，差值 +0.000408
  （+0.0408 个百分点）。soft merge 仅在当前 count 已不超过 target 时
  停止，避免继续无意义合并；该路径仍保持 opt-in。
- 实测了“保留旧 partition、仅处理新增 8K roots”的方案。root-local
  partition 的边界可与 full rebuild bitwise 一致，但 kernel launch 数没有
  减少，原实现累计仅快 0.30%；summary fusion 后，warmed 16-chunk
  累计 full rebuild 为 2.146 ms，增量 append 因 `cat`/metadata launch
  反而为 3.148 ms（慢 47%）。因此回退了这条增量代码，不作为交付路径。
- 新增 opt-in `ROOT_LOCAL_FUSED_SUMMARIES=1`：复用 HISA 的 8-token
  grouped FP8 mean-pooling，root-local kernel 在选出边界后直接生成两行
  FP8 leaf summaries，省掉 raw-K leaf-total 扫描和独立 requant launch。
  含 SummaryPool 写入的 128K 单层中位数 0.471→0.248 ms（1.90x）；
  8K/32K/64K 也分别降低 27.8%/28.1%/27.8%。
- fused summaries 相对 exact raw-K summaries 的完整 real-dump recall：
  0.618895→0.618724，差值 -0.000171（-0.0171 个百分点，95% CI
  半宽 0.0101 个百分点；3504 queries）。因为仍缺 task-level E2E
  质量结果，该开关默认关闭。
- 原 h20-9-57 harness 依赖的 `/DATA/disk0/qyl` 布局和
  `qyl/sglang-hisa:eval` 镜像在当前机器不可用，因此它不能原样运行。
  当前机器实际有 8×H20，并有
  `/share-evpfs/flagos/models/DeepSeek-V3.2` 与可用的 host SGLang /
  DeepGEMM / TileLang 环境；可以改成 host-native launcher 做同机 A/B，
  但结果不能与旧容器的 37.02 s 直接作严格跨环境对比。

## 2026-09-23 — 当前 8×H20 host-native 128K TTFT/TPOT

- 新增 `runner/run_host_ttft_ab.py`，统一使用固定 input IDs、TP=8、
  8K chunked prefill、CUDA Graph BS=1、128K input / 128 output、1 次
  warmup + 3 次计时；同机结果写入 `stats/host_ttft_ab_20260923/`。
- root-local exact：TTFT 中位数 37.0614 s，TPOT 中位数 17.1702 ms；
  root-local fused：36.9519 s / 17.1747 ms。fusion 的 TTFT 差值
  -0.1096 s（-0.30%），TPOT 无可分辨变化。
- fixed HISA-64 使用 `/data/jz/hisa_pr_upstream`，确认服务日志实际进入
  `use_hisa=True, hisa_k_block_size=64, hisa_block_topk=128`。结果为
  TTFT 35.6492 s、TPOT 17.5468 ms；3 次计时 TTFT
  35.6492/35.6410/35.6777 s，且无 traceback。
- 相对 root-local exact，HISA-64 的 TTFT 快 1.4122 s（3.81%），但
  TPOT 慢 0.3767 ms/token（2.19%）。因此当前 adaptive 路径的收益主要
  在 decode；固定 HISA-64 在 prefill 仍更快。
- relaxed global λ-DP（2 轮 λ 搜索、无 exact repair、2 轮 soft merge）
  的同机结果为 TTFT **38.9215 s**、TPOT **17.1982 ms/token**；
  三次 TTFT 为 39.0205/38.9215/38.8897 s。相对 strict global exact
  的 40.1014 s / 17.1529 ms，TTFT 降低 1.1798 s（2.94%），TPOT
  增加 0.0453 ms（0.26%）。服务日志确认进入
  `target128-relaxed-soft2`，且无 traceback。
