# Adaptive-HISA 稀疏 Prefill（adaptive_0921_h202）

日期：2026-09-21。代码：`python/sglang/srt/layers/attention/nsa/adaptive_hisa/prefill_select.py`，
入口挂在 `nsa_indexer.py::_get_topk_ragged`。开关 `SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL=1`（默认 0，关掉即 h201 行为）。

## 动机

128K 输入的速度测试（`delta/docs_research/speed_bench_128k_20260921.md`）：

| 方法 | TTFT | TPOT (graph) |
|---|---:|---:|
| DSA | 68.4 s | 20.3 ms |
| HISA-64（固定块两级索引） | 38.4 s | 17.8 ms |
| Adaptive h201（只在 decode 用两级索引） | 68.9 s | 19.2 ms |

h201 的 prefill 索引仍是官方密集 DSA：每个 8192-query chunk 对整个前缀算 `[8192, N]` logits 再 Top-2048，
128K 时占 prefill 约一半时间。HISA 因为 prefill 也走"块均值粗筛 → 块内精排"，省了 30 s。
本版把 adaptive 的两级选择搬进 prefill。

## 设计

sglang chunked prefill 把 prompt 切成 `--chunked-prefill-size`（8192）个 query 的 chunk 顺序前向。

1. **每个 chunk 都建分区**（`prefill_runtime.prepare_prefill_partition`：`cfg.sparse_prefill` 时不再要求 `prefill_final`）。
   chunk c 在每层 indexer 之后，把 `[0, e_c)` 的 sealed 前缀 `n_complete = floor(e_c/256)·256` 交给 raw-FP8 builder
   （side stream，CUDA graph），产出 `leaf_start/leaf_len`（≈ L/64 个叶子）和 FP8 summary（叶子内 key 均值重量化），
   写进 summary pool；`pool.allocate` 会释放同一 (layer, req) 的上一版页。最后一个 chunk 的分区同时供 decode 使用（与 h201 相同）。
2. **chunk c+1 的 indexer 用 chunk c 的分区**（`prefill_select.sparse_prefill_admission`）：
   要求 B=1、非 capture/spec/CP/PP、entry epoch 匹配、`candidate_tokens ≤ n_complete ≤ chunk_start`、非 fused Top-K 输出。
   不满足就走原密集路径（chunk 0/1 一定是密集：前缀不足 8192）。
3. **两级 Top-2048**（`sparse_topk_core`，全程无 `.item()`，CPU 不停顿）：
   - coarse：`deep_gemm.fp8_mqa_logits(q[n_q], summaries[cap])` → `[n_q, cap]`，padding/空叶子置 -inf；
   - rank：每行 `argsort` 降序 + 叶长 `cumsum`；
   - expand（Triton `_expand_slots_kernel`）：每行 8192 个 slot 各自在 `incl` 上二分找到所在叶子，写出 token id
     （跨越预算的叶子自动截断；叶子按分数序、叶内按位置序，无重复）；
   - fine：叶子候选用 `sparse_paged_mqa_triton(K=1)` 在 paged index-K cache 上精确打分；
     因果局部窗口 `[n_complete, pos]`（当前 chunk + 未 seal 的尾巴，≤ 8192+255 token）是连续的，
     用 `deep_gemm.fp8_mqa_logits`（`ke` 逐行给因果）打分再 `cat`；
   - Top-2048：`hisa_topk_candidates_fused(score, cand, count, seq_lens)`，`count = 8192 + (pos − n_complete + 1)`
     屏蔽窗口里越过因果的列，输出 request-relative token id，格式同 `fast_topk_v2(..., row_starts=ks)`。
   - 行按 `SPARSE_PREFILL_ROWS`（2048）分批，候选/logits 工作区 ≈ rows × 16K × 8 B。

tail 的角色在 prefill 由局部窗口承担。**sink 必须保留**：起点落在 `[0, sink_tokens)` 的叶子在 coarse 分数上置 +inf
强制排第一（不单独列 sink token，避免重复）。原因见下面的 mass 指标。

## 单层单 chunk 验证（真实 dump `dump:ruler:cwe:128k:3` L30，`/tmp/hisa_sparse_prefill_test.py`）

chunk 8（chunk_start 65536，128 条真实 query）：

- 输出全部合法（< pos+1）、无重复；
- 输出 ⊂ "密集 logits 限制在候选集上的 Top-2048"：99.8%（fine 阶段与密集 kernel 数值一致，剩余为 fp8 平局噪声）；
- 相对密集 DSA Top-2048 的 recall：均值 0.800（其中局部窗口部分 100%，密集 Top-2048 有 50.6% 落在 sealed 前缀）；
- **前缀部分 recall：adaptive 叶子+均值 summary 0.597，HISA-64 固定块+均值 summary 0.597**（同预算 8192），
  用"叶内最大 logit"做 oracle 排序反而只有 0.46 —— 均值 summary 选的是"整体相关"的块，比单点最大值更能覆盖 Top-2048。

### 多层/多 chunk 扫描与 attention-sink（`/tmp/hisa_recall_sweep.py`）

指标：prefix recall（同上）、overall recall（局部窗口 100% 覆盖）、**mass** = 我们选出的 2048 个 token 的
relu(logit) 之和 / 密集 Top-2048 的 relu(logit) 之和（更接近注意力质量）。

首版没有 sink 时（lbv2 short-hard 33K dump）：L10 recall 0.97 但 mass 只有 0.39–0.69，L30 mass 0.79–0.84 ——
少数 token（位置 0 附近的 attention sink）占了很大一部分 logit 质量，而均值 summary 把它们稀释掉了，
叶子选不中。HISA-64 的固定块+均值 summary 同样中招（0.44 / 0.74）。
强制 sink 叶子后：L10 mass → 1.000，L30 → 0.88–0.93；128K ruler dump chunk 8 的 mass 0.52 → 0.83。

强制 sink 后 28 个 (layer, chunk) 的均值（ruler cwe 128K，L10/L30，chunk 2–15）：
prefix recall adaptive 0.772 vs hisa64 0.770；overall 0.886 vs 0.885；mass 0.920 vs 0.917。
lbv2 33K dump（L0/3/10/20/30/45/60 × chunk 2–4，21 组）：prefix 0.750 vs 0.742，overall 0.812 vs 0.805。
adaptive 在每个 (layer, chunk) 上都 ≥ hisa64，但优势很小（+0.5–1 pp）。

没有 sink 的首版 e2e 跑了 61 条 LongBench 后停掉（`data/..._nosink_partial/`）：同 61 条上 0.410 vs DSA/P-key 0.508，
2 胜 6 负；带 sink 的版本重新跑。

单元测试 `delta/test/registered/unit/test_adaptive_hisa_sparse_prefill.py`（6 个，GPU 上全部通过）：
配置/环境变量、expand kernel 对 Python 参考、准入规则、Top-K 与"密集限制在候选集"一致、dense-local 与全稀疏一致。

## 单层单 chunk 耗时（8192 query，GPU 与 e2e 服务共享，绝对值偏大）

N=131072（n_complete=122880，1920 叶）：

| 阶段 | ms |
|---|---:|
| coarse deep_gemm `[8192,1920]` | 2.5 |
| rank（argsort+cumsum） | 2.5 |
| expand（Triton，4×2048 行） | ~2.2（只展开叶子） |
| fine：叶子 K=1 sparse mqa + 窗口 dense | ~16 |
| Top-2048 fused | 3.0 |
| **合计 `sparse_topk_core`** | **26.8**（searchsorted + 全稀疏窗口的首版为 38.0） |
| 对照：密集 `deep_gemm [8192,131072]` + topk | ~190（N=73K 时测得 111） |

e2e 服务里（`--disable-cuda-graph`，LongBench 前 8 条）128K prompt 的单请求耗时：sparse prefill 56.3 s，
h201（decode-only）68.9 s，DSA 71.4 s（含十几到几十个 token 的无 graph decode）。

## 下一步

- fine 阶段仍占 60%：叶子候选是 ~64 token 的连续段，可用 segment scorer（`segment_scorer.sparse_paged_mqa_segments`
  的多行版本）代替 K=1 逐 token gather；或把叶子预算内的候选按页对齐。
- 每个 chunk 都建分区：61 层 × 16 chunk 的 raw-FP8 builder（side stream + CUDA graph）；forward 末尾的合流等待
  是每 chunk 的固定成本，可以把最后几层的 build 推迟到下一 chunk 开头。
- 精度：LongBench v2 / RULER 32k+128k e2e 见 `data/adaptive_0921_h202_sparse_prefill_e2e_20260921/`
  （runner：`runner/run_sparse_prefill_e2e.py`）。
