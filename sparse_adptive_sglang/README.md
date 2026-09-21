# Adaptive-HISA 稀疏 Prefill — `adaptive_0921_h202`（代码快照 2026-09-21）

来源机器 h20-9-57，目录 `/DATA/disk0/qyl/code/adaptive_0921_h202`。
基线：https://github.com/xuyufei-a/sglang_hisa 分支 `hisa_pr`，git HEAD `faa198b4e`（`git_head.txt`）；
工作树 = HEAD + `changes_vs_HEAD.patch` + `delta/` 里列出的未跟踪文件。

`h202` 从 `adaptive_0921_h201` 分出来（P-key 自适应分区：L/8 λ-DP split → L/64 target merge，两级选择只用在 **decode**），
把同一套两级选择搬进了 **prefill**。h201 继承下来的 kernel 加速改动见 `VERSION.md` / `SPEEDOPT_NOTES.md`。

## 目录

- `python/`   完整可直接使用的 `python/` 树（`PYTHONPATH=<本目录>/python`，e2e runner 冻结、挂载的就是它）
- `delta/`    只包含相对 git HEAD 有差异的文件，便于 review（内容与 `python/` 一致）
  - `delta/python/sglang/srt/layers/attention/nsa/adaptive_hisa/`   分区 / summary / decode / **prefill_select.py** 整个包
  - `delta/python/sglang/srt/layers/attention/nsa/nsa_indexer.py`   稀疏 prefill 的挂载点
  - `delta/test/registered/unit/test_adaptive_hisa_sparse_prefill.py`   6 个单测（GPU 全过）
  - `delta/docs_research/adaptive_hisa_sparse_prefill.md`   设计 + 全部测量（recall/mass 扫描、e2e 表、NIAH 诊断）
  - `delta/docs_research/speed_bench_128k_20260921.md`       128K TTFT/TPOT：DSA vs HISA-64 vs adaptive
- `runner/run_sparse_prefill_e2e.py`   e2e 队列（LongBench v2 held-out 401 条 + RULER 32k/128k 364 条），`--arm`、`--quick-niah`
- `changes_vs_HEAD.patch`、`speedopt_vs_snapshot1219.patch`   diff（说明见 `SPEEDOPT_NOTES.md`）

下面所有路径都相对 `python/`（`delta/python/` 下是同一份文件），前缀 `nsa/` = `sglang/srt/layers/attention/nsa/`。

## 需要重点检查的算子（完整路径）

### A. 分区构建：建树 → λ-DP split → Ward merge（`nsa/adaptive_hisa/`）

| 算子 | 文件 : 函数 | 作用 |
|---|---|---|
| 总入口 | `nsa/adaptive_hisa/pkey_fp8_builder.py : build_partition_from_fp8` | 一次完整的 P-key 建树 → split → merge，输出 `leaf_start/leaf_len/num_leaves` 和 `_merge_totals`；CUDA graph 捕获的对象 |
| 建树（FP8 直接） | `nsa/adaptive_hisa/partition_kernels.py : _raw_fp8_score_tree_kernel`（Triton） | 由 FP8 key + scale 直接算每个树节点的 SSE 得分（fp64） |
| 建树（分块版） | `partition_kernels.py : _fp8_chunk_tree_kernel`, `_fp8_upper_tree_kernel` | 行分块建树，上层节点合并 |
| λ 搜索 | `nsa/adaptive_hisa/partition_gpu.py : search_lambda` → `partition_kernels.py : search_rounds_triton` → `_dp_count_reduce_kernel`（KC=8, num_warps=1）, `_dp_count_bracket_kernel`, `_bracket_update_kernel`, `_bracket_from_totals_kernel` | 二分搜 λ 使叶子数 ≤ L/8；上限 `lambda_max_rounds=6` |
| DP 叶子标记 | `partition_kernels.py : _dp_leaf_mask_kernel`, `_dp_leaf_gain_kernel` | 在给定 λ 下标记叶子并算 split gain |
| 补叶子 | `partition_gpu.py : repair_leaves` → `partition_kernels.py : _staircase_kernel` | 按最大 gain 把叶子数补到恰好 L/8 |
| 叶子发射 | `partition_gpu.py : emit_leaves` | 生成 `leaf_start / leaf_len` |
| 叶子 totals | `partition_kernels.py : _leaf_totals_fp8_kernel` | 每个叶子的 key 和（整行 tile，为 merge cost 和 summary 用） |
| Merge 调度 | `partition_gpu.py : merge_sync_nonoverlap` | 每轮：cost → 阈值选择 → 不重叠匹配 → 合并；收敛到恰好 L/64 |
| Merge cost | `partition_kernels.py : _merge_cost_kernel` | 相邻叶子对的 Ward cost（用行均值，不再除法） |
| Merge 阈值（selection，非 sort） | `nsa/adaptive_hisa/partition_select.py : kth_threshold_f64`, `lex_select_mask` → `nsa/adaptive_hisa/csrc/radix_select.cu : kth_f64_small_kernel / kth_f64_pass_kernel / kth_f64_finish_kernel / lex_small_kernel / lex_pass_kernel / lex_mask_kernel` | 第 k 小 cost 的 radix selection（含 k≤0 空闲轮早退） |
| Merge 匹配/压缩 | `partition_kernels.py : _merge_match_kernel`, `_merge_index_kernel`, `_merge_compact_kernel`（COMPACT_BLOCK=16） | 不重叠匹配、写新叶子及行均值；空闲轮 device 侧早退 |
| Summary 重量化 | `nsa/adaptive_hisa/summary_kernels.py : requant_from_totals` → `_totals_requant_kernel`；`leaf_key_means` → `_leaf_mean_kernel` | 叶子 key 均值 → FP8 + scale，写入 summary pool |
| Summary 页池 | `nsa/adaptive_hisa/summary_pool.py : SummaryPool.allocate / read`, `build_summaries`, `get_summary_pool` | 复用 index-K 页池存 summary；`allocate` 释放同 (layer, req) 的旧版本 |
| CUDA graph 包装 | `partition_gpu.py : graphed_builder`, `_evict_graphs`, `reset_graph_cache` | 按 (rows, n) 缓存 graph，OOM 时驱逐重试 |

### B. 稀疏 prefill 选择（本版新增，`nsa/adaptive_hisa/prefill_select.py`）

| 算子 | 文件 : 函数 | 作用 |
|---|---|---|
| 准入 | `prefill_select.py : sparse_prefill_admission` | B=1、非 capture/spec/CP/PP、entry epoch 匹配、`budget ≤ n_complete ≤ chunk_start`、非 fused Top-K；最后 chunk 按 `DENSE_FINAL / FINAL_CANDIDATES` 处理 |
| 粗筛 | `prefill_select.py : coarse_leaf_scores` → `deep_gemm.fp8_mqa_logits(q, summaries)` | `[n_q, capacity]` summary logits；padding/空叶子 −inf；起点在 `[0, sink)` 的叶子 +inf 强制排第一 |
| 排序 | `prefill_select.py : rank_leaves`（`torch.argsort` + `cumsum`） | 每行叶子降序 + 叶长前缀和 |
| 展开 | `prefill_select.py : expand_candidates` → `_expand_slots_kernel`（Triton） | 每行 `budget` 个 slot 在前缀和上二分找到所在叶子，写出 token id；跨预算叶子自动截断 |
| 精排（叶子候选） | `nsa/hisa/triton_kernels.py : sparse_paged_mqa_triton`（K=1） | 对候选 token 在 paged index-K cache 上算精确 fp8 logits |
| 精排（局部窗口） | `prefill_select.py : _local_keys` + `deep_gemm.fp8_mqa_logits`（`ke` 逐行因果） | 因果局部窗口 `[n_complete, pos]` 连续，用密集 kernel 打分后 `cat` |
| Top-2048 | `nsa/hisa/hisa_topk_fused.py : hisa_topk_candidates_fused` | 输出 request-relative token id，格式同 `fast_topk_v2(..., row_starts=ks)` |
| 核心编排 | `prefill_select.py : sparse_topk_core`（可独立测试）、`sparse_prefill_topk`（服务内包装） | 行按 `SPARSE_PREFILL_ROWS` 分批，全程无 `.item()` |

### C. Decode 侧（h201 继承，未改）

`nsa/adaptive_hisa/decode_runtime.py : maybe_decode_topk → select_topk`，候选选择 `nsa/adaptive_hisa/decode_select.py : weighted_select_candidates`
→ `nsa/adaptive_hisa/csrc/weighted_select.cu : weighted_prefix_fast_kernel / expand_selected_kernel / expand_intervals_kernel`，
连续段精排 `nsa/adaptive_hisa/segment_scorer.py : _segment_score_kernel`。

## 稀疏 prefill 的关键调用链

sglang chunked prefill 把 prompt 切成 `--chunked-prefill-size`（8192）个 query 的 chunk 顺序前向。以 128K = 16 个 chunk 为例：

```
scheduler 组 batch
  sglang/srt/managers/schedule_batch.py : ScheduleBatch.get_model_worker_batch
    └─ prefill_runtime.partition_request_flags  → forward_batch.prefill_final_cpu / prefill_new_tokens_cpu
                                                   （标记这是不是该请求的最后一个 chunk、本 chunk 新 token 数）

每个 forward（一个 chunk）
  sglang/srt/models/deepseek_v2.py : DeepseekV2Model.forward
    ├─ _begin_adaptive_partitions  → nsa/adaptive_hisa/prefill_runtime.py : begin_forward      （回收上一轮 side-stream 任务）
    ├─ 61 层 × {
    │    sglang/srt/layers/attention/nsa/nsa_indexer.py : Indexer.forward_indexer
    │      └─ _get_topk_ragged                                                           （L1812；B=1 extend 路径）
    │           ├─ [chunk 0/1 或不准入]  官方密集 DSA：deep_gemm.fp8_mqa_logits [n_q, N] → metadata.topk_transform
    │           └─ [准入]  L1936-1959：
    │                prefill_select.sparse_prefill_admission(forward_batch, layer_id, metadata)
    │                  ├─ prefill_runtime.get_summary_entry(layer_id, req_idx)           （上一个 chunk 建好的分区 + summary）
    │                  └─ 返回 (cfg, entry, chunk_start, seq_len, budget) 或 None
    │                prefill_select.sparse_prefill_topk(...)
    │                  ├─ wait_event(entry.ready_event)                                  （等 side stream 上的分区建完）
    │                  ├─ summary_pool.SummaryPool.read(entry)  → summaries fp8 [cap,128] + scale
    │                  └─ sparse_topk_core(q, w, seq_lens, raw_pages, block_table, summaries, leaf_start/len, ...)
    │                       coarse_leaf_scores → rank_leaves → 按 2048 行分批 {
    │                         expand_candidates(_expand_slots_kernel)
    │                         sparse_paged_mqa_triton(K=1)         ← 叶子候选精排
    │                         deep_gemm.fp8_mqa_logits(局部窗口)   ← [n_complete, pos] 精排
    │                         hisa_topk_candidates_fused           ← Top-2048
    │                       }
    │                topk_result[:q_offset] = 上面的输出
    │           └─ nsa_indexer._schedule_adaptive_partition(forward_batch, layer_id, k_fp8, k_scale)
    │                └─ prefill_runtime.schedule_prefill_partition
    │                     ├─ prepare_prefill_partition   （cfg.sparse_prefill 时中间 chunk 也准入；n_complete = floor(seq_len/256)·256）
    │                     └─ _schedule_key_partition → _build_gpu_raw_fp8   （side stream，CUDA graph）
    │                          ├─ pkey_fp8_builder.build_partition_from_fp8(k_fp8[:n_complete], k_scale, ...)
    │                          │    build_key_tree_from_fp8 → search_lambda → dp_leaf_and_gain → repair_leaves
    │                          │    → emit_leaves → leaf_totals_from_fp8 → merge_sync_nonoverlap
    │                          ├─ summaries_from_totals(_merge_totals, leaf_len)  → FP8 summary
    │                          ├─ summary_pool.allocate(layer, req, epoch, part) + 写页    （释放同 (layer, req) 的上一版）
    │                          └─ record ready_event                                        （下一个 chunk 的 indexer 等它）
    │  }
    └─ _finish_adaptive_partitions → prefill_runtime.finish_prefill_partitions   （forward 末尾与 side stream 合流）

decode（与 h201 相同）
  nsa_indexer._get_topk_paged → decode_runtime.maybe_decode_topk → select_topk（用最后一个 chunk 建的分区）
```

时序关系：chunk c 的 indexer 用的是 chunk c−1 结束时建好的分区（覆盖 `[0, n_complete_{c-1})`），chunk c 自己的 8192 个 token
和 `[n_complete, chunk_start)` 之间未 seal 的尾巴由"局部窗口"覆盖。chunk 0/1 一定走密集（前缀 < 8192 预算）。

## 开关（环境变量前缀 `SGLANG_NSA_ADAPTIVE_HISA_`，解析在 `nsa/adaptive_hisa/config.py`）

| env | 默认 | 含义 |
|---|---|---|
| `SPARSE_PREFILL` | 0 | 开启稀疏 prefill（0 = h201 行为） |
| `SPARSE_PREFILL_ROWS` | 2048 | 精排每批 query 行数 |
| `SPARSE_PREFILL_CANDIDATES` | 0 | prefill 叶子 token 预算（0 = 沿用 decode 的 `CANDIDATE_TOKENS`=8192） |
| `SPARSE_PREFILL_FINAL_CANDIDATES` | 0 | 只给最后一个 chunk 用的预算 |
| `SPARSE_PREFILL_DENSE_FINAL` | 0 | 最后一个 chunk 保留官方密集 DSA indexer（中间 chunk 仍稀疏） |

## 结果（128K，8×H20 TP8，DeepSeek-V3.2）

速度（`delta/docs_research/speed_bench_128k_20260921.md`，128K 输入 / 128 输出，decode 开 CUDA graph）：

| | TTFT | TPOT |
|---|---:|---:|
| DSA 官方 | 68.4 s | 20.3 ms |
| HISA-64（固定块，prefill+decode 两级） | 38.4 s | 17.8 ms |
| h201 adaptive，仅 decode 稀疏 | 68.9 s | 19.2 ms |
| h202 稀疏 prefill（单请求，无 graph） | 56.3 s | – |

精度（arm `sparse_prefill`：全 chunk 稀疏，预算 8192，sink 64；与 h201 同一 held-out）：

| | LongBench v2（401） | RULER 32k（182） | RULER 128k（182） |
|---|---:|---:|---:|
| DSA 官方 | 0.504 | 0.872 | 0.821 |
| h201 仅 decode 稀疏 | 0.501 | 0.882 | 0.825 |
| **h202 稀疏 prefill 8192** | **0.479** | 0.855 | **0.598** |

128k 的损失集中在 NIAH（`niah_multikey_1/2/3` 0.43 / 0.36 / 0.21，h201 是 0.93 / 1.0 / 1.0）。原因：第一个输出 token 由
**最后一个** prompt chunk 的末尾几行产生，它们要在 128K 前缀里定位 needle；均匀 haystack 上均值 summary 把 needle 所在叶子稀释掉
（NIAH dump 上前缀 Top-2048 recall 只有 0.55，第 3 层 0.18；固定块 HISA-64 完全一样）。预算加到 16384 只回到 0.57 / 0.43 / 0.54。

正在验证的修法：`SPARSE_PREFILL_DENSE_FINAL=1`（中间 chunk 稀疏，最后一个 chunk 密集）。42 条 128k `niah_multikey_1/2/3`：
0.71 / 0.93 / 0.93；代价约为密集 indexer 总量的 1/8（128K 上 ~1.5–2 s TTFT）。下一步用这个 arm 跑完整 LongBench + RULER 并重测 TTFT。

各任务明细、recall/mass 扫描：`delta/docs_research/adaptive_hisa_sparse_prefill.md`。

## 运行

    PYTHONPATH=<本目录>/python
    SGLANG_NSA_ADAPTIVE_HISA_MODE=adaptive_decode
    SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC=key_sse  SGLANG_NSA_ADAPTIVE_HISA_SPLIT_BACKEND=gpu
    SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY=sync_nonoverlap  SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR=64
    SGLANG_NSA_ADAPTIVE_HISA_RAW_FP8_BUILDER=1  SGLANG_NSA_ADAPTIVE_HISA_GPU_STREAM=side
    SGLANG_NSA_ADAPTIVE_HISA_SINK=64  SGLANG_NSA_ADAPTIVE_HISA_TAIL=256  SGLANG_NSA_ADAPTIVE_HISA_CANDIDATE_TOKENS=8192
    SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL=1  [SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_DENSE_FINAL=1]
    SGLANG_NSA_FUSE_TOPK=0

服务参数：`--chunked-prefill-size 8192 --max-running-requests 1 --disable-radix-cache`。
到第三个 chunk 起日志应出现 `adaptive-hisa sparse prefill layer=... final=...`。

单测（GPU）：`PYTHONPATH=python python -m pytest delta/test/registered/unit/test_adaptive_hisa_sparse_prefill.py -q`
