# Adaptive-HISA 加速改动快照 (2026-09-20 晚)

基线 = 同目录 12:19 的 `adaptive_hisa_pkey_target64_20260920` 快照（P-key + L/8 split -> L/64 target merge，
e2e 已跑通的那一版）。本包 = 基线 + 本轮 decode/prefill 冗余削减改动。

- 相对基线的完整 diff：`speedopt_vs_snapshot1219.patch`（31 个文件）
- `delta/` 已与 `python/` 主树对齐，可直接用于 review
- `changes_vs_HEAD.patch` 仍是**基线相对 git HEAD** 的 diff，未重新生成（本机无 git 仓库）

## 新增文件

| 文件 | 作用 |
| --- | --- |
| `python/.../adaptive_hisa/pkey_fp8_builder.py` | raw-FP8 prefill builder 路径（FP8 K+scale 直接建树/totals/summary） |
| `python/.../adaptive_hisa/segment_scorer.py` | segment-aware 连续访存精排 kernel |
| `python/.../adaptive_hisa/decode_timing.py` | decode 逐阶段 NVTX / event 计时 |
| `runner/bench_speed_arms.py` | DSA / HISA-K64 / Adaptive 三臂速度基准 |
| `runner/sweep_algorithm_knobs.py` | chunk / merge divisor / 候选预算扫描 |
| `runner/validate_pareto.py` | 单测 + 配对统计 + decode gate 校验 |
| `runner/diagnose_decode_consistency.py` | weighted_select 区间输出与原实现的一致性诊断 |
| `runner/deliverable_speed.env` | 可交付速度配置的环境变量 |
| `runner/run_pkey_target64_jump.py` | jump_h20-1 上的启动脚本 |

## 主要修改

- P-key-only 清理：删除 P-qfull calibration query tail、复用官方 logits、
  `schedule_prefill_partition_scores` 和 score-SSE 配置分支；`key_sse` 现在是唯一 metric
- `decode_runtime.py`：共享 DeepGEMM schedule、复用 workspace 输出缓冲、边界触发 seal、page-table 融合
- `incremental.py`：`SKIP_CLEAN` 默认跳过全容量 `clean_summary_scores` 扫描
- `partition_gpu.py`：`capacity` 改为按合并后 `L/D` 而非 `L/8`；kth 阈值用预分配缓冲；merge 支持 ping-pong buffer
- `summary_pool.py`：`pages_for` 按 `merge_divisor` 计算最终行数，消除容量膨胀
- `prefill_runtime.py`：接入 raw-FP8 builder；P-key 路径去掉多余 `clone()`
- `csrc/weighted_select.cu` + `decode_select.py`：新增 `weighted_select_intervals`（输出 start/length/offset 区间）
- `nsa_backend.py`：page-table 由 Adaptive-HISA 融合产出时跳过 `transform_index_page_table_decode`
- `config.py`：`DECODE_CHUNK` 默认 8 -> 64；新增 `SKIP_CLEAN` / `FUSE_PAGE_TABLE` / `SEGMENT_SCORER` / `RAW_FP8_BUILDER` / `DECODE_TIMING` 开关

## 运行

```bash
export PYTHONPATH=<解包路径>/python
source <解包路径>/runner/deliverable_speed.env
```

`SEGMENT_SCORER` 默认关闭，需在区间一致性诊断通过后再开。

## 尚未验证（本机 GPU 被占，全部改动只做了静态检查）

1. `test_adaptive_hisa_decode.py::test_graph_replay_after_workspace_rebind_and_new_tail`
   仍按 `DECODE_CHUNK=8` 断言 32 个 sealed block，chunk 改 64 后实际为 4，需要更新期望值。
2. `runner/diagnose_decode_consistency.py` 的 `test_clean_redundant` 有 `NameError: capacity`。
3. `bench_speed_arms.py` 的 HISA 臂：`SGLANG_HISA_ENABLED` / `SGLANG_HISA_CHUNK_SIZE`
   在 python 侧搜不到消费点，该臂大概率没真正跑 HISA，结论不可用之前必须先修。
4. 固定 `input_ids` 是否被 `bench_one_batch.py` 真正读取，未在 GPU 上确认。
5. raw-FP8 builder 目前仍会落 FP32 中间量，不是完全 raw；merge 仍用 `torch.sort` 未换 radix。
