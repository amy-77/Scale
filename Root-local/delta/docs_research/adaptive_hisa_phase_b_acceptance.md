# Adaptive-HISA Phase B 验收记录（2026-09-19）

后续更新：`adaptive_decode` 已完成第一版接线，使用方式与边界见 [Adaptive-HISA decode](adaptive_hisa_decode.md)。本文保留 Phase B 当时的验收记录；下文“decode 尚未接线”是当时状态。

目标（来自 `gpt_code_advice/adaptive_hisa_phaseB_revision_plan.md`）：在不重写打分器、不恢复整套旧 HISA 的前提下，把
「GPU 全局切分 → 真正的两轮合并 → GPU FP8 摘要」这条链补完整，并回归 build_only（官方 Top-2048 不变）。
四个小步：B0 锁配置和参考算法，B1 GPU 分数树 + 全局切分，B2 两轮合并 + 摘要，B3 接线、回归、真机 A/B。

结论先行：

* 四步全部完成。`MODE=build_only SPLIT_BACKEND=gpu`（默认）时，每层在主 stream 上完成
  分数树 → λ-DP → 精确修复 → 两轮 sync_nonoverlap 合并 → 叶子 FP8 均值摘要写入分页池，全程没有 host 同步、
  没有从 device 读回形状。
* 与 CPU numba 参考实现（旧 A/B 的 `v7_reuse` 路径原样保留为 `cpu_reference`）在随机 / 全零 / 分段常数 / 稀疏 /
  整数平局输入上逐位一致；GPU 上 Triton 融合核与 torch 逐算子实现也逐位一致（λ 仅末位 ulp 差别，来自 FMA）。
* 真机（DeepSeek-V3.2，TP8，H20，32K prompt，4×8192 chunk，构建落在最后一个 chunk）：见 §5。
  最后一个 chunk 的开销从第一版 GPU 链的 +1.24 s 压到同步 GPU 链的 +0.14 s（切分+两轮合并+摘要，61 层，
  CUDA graph 回放，host 侧 54 ms；剩下的几乎全是构建本身的 GPU 时间 61 × 1.7 ms），再把这条链放到第二条 stream
  上与后续层重叠（`GPU_STREAM=side`，现在的默认）后是 **+0.04 s**，与 `v7_reuse`（CPU numba + 工作线程重叠，
  +0.036 s）持平，且带完整的 GPU FP8 摘要。`GPU_STREAM=main` / `GRAPH_BUILD=0` 可分别回到同步、逐 launch。

## 1. 文件

`python/sglang/srt/layers/attention/nsa/adaptive_hisa/`

| 文件 | 作用 |
| --- | --- |
| `config.py` | `PartitionConfig`：`MODE / SPLIT_BACKEND / MERGE_POLICY / MERGE_ROUNDS / MERGE_ALPHA / MAX_MERGE_LEN / BUILD_SUMMARIES / PARTITION_OVERLAP / PARTITION_REUSE_LOGITS / LAMBDA_CANDIDATES …` 的统一解析与校验。旧变量 `PREFILL_PARTITION=1` 等价于 `MODE=build_only`；`PARTITION_MERGE=1` 不再隐含算法，必须显式 `MERGE_POLICY`；与 MISA / per-head 实验开关同时打开时直接报错。 |
| `partition_reference.py` | CPU 参考：numba λ-DP + 堆序精确修复；`heap_reference`（旧的顺序堆合并，带 256 上限）；`sync_nonoverlap`（论文算法，移植自 sibling `batched_adjacent_merge_rounds`）。 |
| `prefill_partition.py` | `PrefillPartition` 数据结构、校准打分 torch 回退、FP64 矩树、cpu_reference 后端的编排。 |
| `partition_gpu.py` | GPU 后端：`TreeLayout`、分数树、K 路 λ 搜索、单遍精确修复（staircase 字典序）、无排序叶子压缩、sync_nonoverlap 合并、`GpuPartition`（device 常驻，`to_host()` 懒转换）；`GraphedBuilder`：整条构建按 `(C, N_complete)` 捕获成一张 CUDA graph，同一请求的 61 层各回放一次（LRU 2 张，捕获失败自动退回 eager）。torch 实现同时是 Triton 核的参考。 |
| `partition_kernels.py` | Triton 融合核：每个 root 一个 program 的分数树 / λ 计数（16 个候选共享一次树读取）/ 括号更新 / 叶子掩码 / gain / staircase，以及每轮 3 个 launch 的合并（代价、匹配+压缩映射、gather）。所有运行期整数 `do_not_specialize`，block 固定，`warmup_triton_kernels` 一次编译全部变体。 |
| `summary_kernels.py` | Triton 分段 FP8 反量化均值；`requant_summaries` 复用生产 `act_quant`（含 ue8m0）。 |
| `summary_pool.py` | 分页 FP8 摘要池（与 index-K 池同样的 64 行/页 SoA 布局，`SetKAndS` 写入），按 (layer, req) 分配 / 复用 / 释放，页表 pinned 非阻塞上传。 |
| `prefill_runtime.py` | 运行时状态：滚动校准尾（24 条 q/w）、准入判断（B=1、非 spec/CP、最后一个 chunk、prefix-hit 拒绝）、`gpu`（主 stream 或 `GPU_STREAM=side` 第二条 stream，forward 结束时 join）/ `cpu_reference(+overlap)` 构建路径、`BuildContext`、摘要池接入、请求释放、计时统计、首个 prefill 时的预热（Triton 编译 + 首次 graph 捕获 + 摘要池分配 + pinned 页表 + `SetKAndS`）。 |
| `legacy/` | 旧的 P-radius / CPU decode 原型（`runtime.py, sidecar.py, cuda_select.py, reference.py, csrc/`），不在任何路径上。 |

其他改动：`nsa_indexer.py`（两处 hook 传 `BuildContext` / `scale_fmt`）、`deepseek_v2.py`（forward 起止 hook）、
`schedule_batch.py` / `forward_batch_info.py`（`prefill_final_cpu`、`prefill_new_tokens_cpu`）、`memory_pool.py`
（请求释放时回收摘要页）、`scripts/run_indexer_dump.sh`（新变量）。

## 2. 算法契约

* 度量 P-qfull：`E(node) = Σ_c Σ_{s∈node} (I[c,s] − mean_c)^2`，24 条校准 query 的未掩码分数，FP64。
* 树：`ATOM=1`，`ROOT=256`，9 层；只对已封口的 `N_complete = floor(N/256)·256` 建树，尾部不建。
* 预算：全局 `M0 = N_complete / 8`；λ-DP（惩罚 λ，`split < keep` 严格，平局保留父节点）K 路括号搜索到相对
  1e-12；不足的叶子按 gain 降序（start / level / idx 升序打平）精确补齐到 M0。
* 合并 `sync_nonoverlap`（默认，2 轮）：冻结叶子，给所有相邻对算 Ward 代价，按 `(cost, start)` 顺序取
  互不重叠的对（`cost < alpha·λ`，可选 `len_l+len_r ≤ MAX_MERGE_LEN`），同时合并，重复。
  与旧的顺序堆合并**不是同一算法**：反例见 `adaptive_hisa_runtime_audit.md §4.5`。
* 摘要：每个叶子 FP8 key 反量化后的 FP32 均值 → `act_quant` → (FP8[128], scale) 一行，写入摘要池；
  容量固定 M0 行（合并后的空行为 padding，`num_leaves` 为有效行数）。

## 3. 测试

`test/registered/unit/test_adaptive_hisa_prefill_partition.py`：59 个用例，CPU 与 CUDA 各跑一遍（CUDA 用例在
`qyl/sglang-hisa:eval` 镜像内通过，`Ran 59 tests … OK (skipped=1)`）。

* 配置：默认值、旧变量迁移与报错、后端规则、MISA 冲突拒绝。
* CPU 参考：矩树 = 直接方差、9 层覆盖、λ-DP 预算/覆盖/尾部、平局修复顺序、预算流向高方差 root、
  页置换不改变逻辑分区、堆合并全塌缩/严格阈值/上限、**评审反例上 sync_nonoverlap ≠ heap**、与 sibling 离线实现逐位一致。
* GPU 后端：四种分布 × 三种长度的切分逐位对齐参考；atom=4/compression=16；短尾无叶子；两轮合并（含上限）逐位对齐
  numpy 参考并保持 capacity=M0、padding 约定；**Triton 融合核 vs torch 回退逐位一致**；**CUDA graph 回放 vs eager
  逐位一致**（四种长度 × 两个种子、strided 输入、同长度复用同一张图、短尾不会把 graph 路径关掉）。
* 摘要：均值 vs 参考、padding 为零、分页往返逐位、分配复用/释放/耗尽、短尾零分配、按 KV 池定尺寸、ue8m0 与 `act_quant` 逐位。
* build_only 接线：调度器标志、分块 prefill 滚动尾 → 构建、**官方 Top-2048 与原始 K 不被触碰**、复用 logits 路径 = 回退打分器、
  短尾/prefix-hit/无尾/非最后 chunk 的四种回退与原因、请求释放后槽位复用同一批页、合并开启时 capacity 与 padding、
  cpu_reference 仍可接线（无摘要）、两后端经运行时得到相同叶子、DeepGEMM 搭车行不改官方行、overlap 与同步一致、ready_event/计时元数据、
  **`GPU_STREAM=side` 与主 stream 得到相同叶子和逐位相同的摘要页，forward 结束时 join**。

```bash
PY=/DATA/disk1/jiabei/compensability-20260918/env/bin/python
cd /DATA/disk0/qyl/code/dpskv32/sglang-hisa
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/python $PY test/registered/unit/test_adaptive_hisa_prefill_partition.py
# 镜像内
docker run --rm --gpus '"device=0"' --ipc host -v /DATA/disk0/qyl:/workspace/qyl \
  -e PYTHONPATH=/workspace/qyl/code/dpskv32/sglang-hisa/python -w /workspace/qyl/code/dpskv32/sglang-hisa \
  --entrypoint python3 qyl/sglang-hisa:eval test/registered/unit/test_adaptive_hisa_prefill_partition.py
```

## 4. 单层构建微基准（H20 空闲卡，warm，`/tmp` 脚本）

| 阶段 @32K | 第一版 GPU 链（torch 逐算子） | 融合后 | 说明 |
| --- | --- | --- | --- |
| tree | 0.60 ms | 0.11 ms | 1 launch |
| λ 搜索 | 5.88 ms（8 轮 × ~70 launch） | 0.42 ms（10 轮 × 2 launch） | K=16；FP64 在 H20 上很慢，少候选多轮更优 |
| leaf mask | 0.65 ms | 0.06 ms | 1 launch |
| repair | 2.91 ms + 1 次 host 同步 | 0.70 ms，无同步 | 单遍 staircase + 4 次排序 |
| emit | 0.37 ms | 0.15 ms | 去掉排序 |
| merge（2 轮） | 1.66 ms | 0.24 ms | 每轮 3 launch |
| summaries | 0.34 ms + 1 次 host 同步 | 0.36 ms，无同步 | 页表 pinned 非阻塞 |
| **整体（host）** | **12.3 ms，1448 launch** | **1.85 ms，216 launch** | GPU 时间 1.9 ms |
| 整体 @128K | — | host 1.9 ms / GPU 3.7 ms | λ 1.4 ms，tree 1.1 ms |

CUDA graph 回放（`GRAPH_BUILD=1`，默认）在上面的基础上把 host 时间再压掉（镜像内，逐位一致）：

| N | eager host / 整体 | graph host / 整体 | 捕获（每个新长度一次） |
| --- | --- | --- | --- |
| 256 | 1.7 / 1.7 ms | 0.06 / 1.0 ms | 5.8 ms |
| 4096 | 1.9 / 1.9 ms | 1.1 / 1.3 ms（host 被 GPU 队列顶住） | 6.5 ms |
| 32 868 | 1.9 / 2.1 ms | 0.14 / 1.8 ms | 7.6 ms |
| 131 072 | 4.3 / 4.7 ms | 1.7 / 4.6 ms | 7.5 ms |

捕获用 `capture_begin/capture_end` 而不是 `torch.cuda.graph`（后者进入时会 `synchronize + gc.collect + empty_cache`，
在 forward 中间做不得）。冷容器里 Triton 首次编译约 6.6 s；编译、首次 graph 捕获、摘要池 310 MiB 分配、首个 pinned 页表和
`SetKAndS` 路径都在进程首个 prefill（server 自身的 warmup 请求）里预热，不落在请求内——第三轮 A/B 里摘要阶段
0.22 s 的「一次性成本」就是这几项，预热后降到 0.004 s。

## 5. 真机 A/B（32K prompt，`ruler:cwe:128k:3` 截到 32768 token，4×8192 chunk，B=1，TP8）

命令（`scripts/run_indexer_dump.sh`，`FORWARD_TIMING=1`，每个配置单独起 server 跑一条 prompt）：

```bash
COMMON="LAYERS=999 RULER_PER_TASK=1 LB_PER_GROUP=0 MAX_NEW_TOKENS=1 MAX_DECODE_STEPS=0 FORWARD_TIMING=1"
EXTRA='--only ruler:cwe:128k --max-prompt-tokens 32768'
env $COMMON EXP_NAME=phaseb_off DUMP_PORT=31801 ADAPTIVE_MODE=off EXTRA_ARGS="$EXTRA" bash scripts/run_indexer_dump.sh
env $COMMON EXP_NAME=phaseb_v7_reuse DUMP_PORT=31802 ADAPTIVE_MODE=build_only SPLIT_BACKEND=cpu_reference \
    MERGE_POLICY=heap_reference MAX_MERGE_LEN=256 PARTITION_OVERLAP=1 BUILD_SUMMARIES=0 EXTRA_ARGS="$EXTRA" bash scripts/run_indexer_dump.sh
env $COMMON EXP_NAME=phaseb_gpu_sync_v4 DUMP_PORT=31832 ADAPTIVE_MODE=build_only SPLIT_BACKEND=gpu \
    MERGE_POLICY=sync_nonoverlap MERGE_ROUNDS=2 BUILD_SUMMARIES=1 GRAPH_BUILD=1 EXTRA_ARGS="$EXTRA" bash scripts/run_indexer_dump.sh
# 第二条 stream（opt-in）
env $COMMON EXP_NAME=phaseb_gpu_side_v5 DUMP_PORT=31841 ADAPTIVE_MODE=build_only SPLIT_BACKEND=gpu \
    MERGE_POLICY=sync_nonoverlap MERGE_ROUNDS=2 BUILD_SUMMARIES=1 GRAPH_BUILD=1 GPU_STREAM=side EXTRA_ARGS="$EXTRA" bash scripts/run_indexer_dump.sh
```

一次只能跑一个实例：第三轮曾把脚本起了两份，两组容器同时装模型、互相 `docker rm`，那一轮的数字全部作废（不在下表）。
第四/五轮都是单实例顺序跑的。

最后一个 8192-token chunk 的 forward 墙钟（TP0，`adaptive-hisa extend forward tokens=8192 wall_s`），构建全部发生在这个 chunk：

| 配置 | 最后 chunk | 相对 off | 61 层构建（GPU event 累计 / 模型线程 launch 墙钟） |
| --- | --- | --- | --- |
| `off`（第一轮两次 / 第四轮） | 3.113 / 3.108 / 3.084 s | — | — |
| `v7_reuse`（cpu_reference + heap 256 + overlap，三次） | 3.140 / 3.153 / 3.120 s | +0.03～0.04 s | D2H 等待 1.19 + dp 0.28 + merge 0.12 s 在工作线程；`final_wait` 0.04 s |
| 第一版 GPU 链（torch 逐算子 + 摘要） | 4.357 s | +1.24 s | 0.84 s / 2.21 s（36 ms/层 launch） |
| 融合核、**未预热**（编译落在请求内） | 9.159 s | +6.05 s | 5.64 s / 6.09 s（几乎全是 Triton 编译） |
| 融合核 + 预热 + 摘要，eager（`gpu_eager_v4`） | 3.226 s | +0.142 s | 0.106 s（tree 0.025 / dp 0.026 / repair 0.037 / merge 0.018 / 摘要 0.004）/ 0.166 s |
| **融合核 + 预热 + 摘要，CUDA graph 回放（`gpu_sync_v4`，默认配置）** | **3.221 s** | **+0.137 s** | 0.105 s（摘要 0.004）/ **0.054 s** |
| 同上，无摘要（`gpu_nosummary_v4`） | 3.231 s | +0.147 s | 0.098 s / 0.023 s |
| 同上，不合并（`gpu_nomerge_v5`） | 3.210 s | +0.12 s | 0.082 s（摘要 0.005）/ 0.052 s |
| `off`（第五轮，同一时段基线） | 3.093 s | — | — |
| graph 回放（第五轮重复，`gpu_sync_v5`） | 3.224 s | +0.13 s | 0.098 s / 0.055 s |
| **第二条 stream + graph（`gpu_side_v5`，opt-in）** | **3.127 s** | **+0.035～0.043 s** | event 跨度 1.26 s（每层 ~20 ms，与模型核共享 SM 被拉长，但都在 forward 内完成）/ 0.058 s；`final_wait` 0.050 s |
| 第二条 stream，eager（`gpu_side_eager_v5`） | 3.141 s | +0.05 s | 1.29 s / 0.171 s |

| **默认配置（side + graph + 摘要，`gpu_default_v6`，两次）** | **3.133 / 3.127 s** | **+0.03～0.04 s** | event 跨度 1.26 s / 0.06 s；`final_wait` 0.050 s；allocator 无 retry |
| `off`（第六轮） | 3.096 s | — | — |

读法：

* eager 与 graph 的 host 时间差 0.11 s，但主 stream 上的墙钟只差 5 ms——预热之后模型线程本来就跑在 GPU 前面，host 不是瓶颈。
  同步 GPU 链剩下的 +0.14 s 就是 61 层构建自己的 GPU 时间（61 × 1.7 ms：FP64 λ 搜索 + 4 次排序的修复 + 两轮合并）排在主 stream 上，
  摘要（0.004 s）和合并（对比 nomerge，~10 ms）都在噪声边缘。
* 把同一条链放到第二条 stream 上（scores 就位后 `wait_stream`，forward 结束时 join），构建与后续层的模型核共享 SM，
  每层的 event 跨度从 1.7 ms 拉长到 ~20 ms，但都在下一层结束前完成，最后一个 chunk 只剩 join 时最后一两层的尾巴：
  **+0.04 s，与 `v7_reuse` 相同，而且带完整的 GPU 摘要**。于是 `GPU_STREAM=side` 成为默认（`GPU_STREAM=main` 回到同步）。
  side 路径先在主 stream 上把 24 × N_complete 的校准行 clone 出来（3 MB），再交给第二条 stream，避免 `record_stream`
  把 1 GiB 的官方 logits 块钉住；gather 出来的 K（32K 时 4 MB）按层 `record_stream`，实际滞后约一层。
* 前三个 chunk（不构建）所有配置都是 3.45 / 2.58 / 2.85 s 左右；`off` 三次 3.084～3.093 s，噪声约 ±10 ms。

分区形状（`gpu_sync`，`PARTITION_LOG_LAYERS=1`，61 层）：合并前每层 4096 叶，两轮后平均 2353（2069～3136）；
第一轮平均合 1207 对、第二轮 536 对；修复平均 2.6 个节点，61 层 `status_ok` 全真。合并后叶长分布：
1:23.6%、2:16.7%、3:4.8%、4:7.9%、5–7:5.3%、8:5.5%、9–15:8.6%、16:4.3%、17–31:9.3%、32:2.7%、33–63:6.7%、
64–127:4.0%、128–511:0.9%、512+:<0.05%（不设上限时极少数叶子超过 256）。摘要池 61 层 × 630 页 = 310 MiB。

## 6. 与评审意见的对应

* 「找到真正的两轮合并配对代码」：`partition_reference.sync_nonoverlap_merge` 是 sibling
  `batched_adjacent_merge_rounds` 的直译，`partition_gpu.merge_sync_nonoverlap` / `partition_kernels._merge_match_kernel`
  用「到单调段局部最小的距离为偶数」在并行下精确复现同一贪心匹配；审计 §4.5 的错误论断已改正并附反例。
* 「隔离旧 radius/CPU decode 原型」：移入 `legacy/`，`memory_pool.release_request` 不再调用 sidecar。
* 「保留本地基线与已有优化」：`cpu_reference + heap_reference + MAX_MERGE_LEN=256 + PARTITION_OVERLAP=1 + REUSE_LOGITS=1`
  就是之前 A/B 的 `v7_reuse`，本次重跑数字一致（+0.03～0.04 s）。
* 「GPU 化不自动等于更快，要和 v7_reuse 比」：同意，§5 的表就是这么比的；第一版 GPU 链和未预热版本都明显更慢，
  根因分别是 launch-bound（1448 launch/层 + 2 次 host 同步）和请求内 JIT 编译，两者都已修。同步 GPU 链（+0.14 s）
  仍比 `v7_reuse`（+0.04 s）慢，差的是构建自己的 GPU 时间。
* 「异步留作独立优化、stream 并发不保证重叠」：这一项做成了独立开关 `GPU_STREAM` 并单独 A/B（§5 末四行）：
  side + graph +0.04 s，side + eager +0.05 s，main + graph +0.14 s。因为有真机数字且与主 stream 逐位一致、
  可一键回退，才把它设为默认；不是假定并发就更快。

## 7. 限制与下一步

1. side stream 的收益依赖 SM 余量：8192-token chunk 下模型核之间有足够空隙让 200 个小核挤进去；更小的 chunk
   或更满的卡上收益会缩小，最坏退化到同步 GPU 链的 +0.14 s（不会更差，因为 forward 结束时 join）。
   构建滞后模型约一层，每层多钉住 3 MB 校准行 + gather 出来的 K（32K 时 4 MB，128K 时 16 MB）直到该层构建结束。
2. λ 搜索是 FP64 计算，H20 的 FP64 吞吐很低（~1 TFLOP）；128K 时单层 GPU 时间 3.7 ms 里 λ 占 1.4 ms。
   降到 FP32 会破坏与参考实现的逐位一致，暂不做。
3. 修复阶段仍有 4 次 `torch.sort`（gain 排名 + 3 段字典序），约 0.7 ms/层；可再融合成一个 per-root 排序。
   在 side stream 上这部分已不在关键路径上，优先级下降。
4. graph 模式下没有每阶段的 event 计时（只有整条构建的 `total_s`）；要看阶段分解用 `GRAPH_BUILD=0`。
   每个新的 `N_complete` 捕获一次（~8 ms），LRU 保留 2 张图；长度频繁变化的负载每个请求都会付这 8 ms。
5. `MODE=adaptive_decode` 仍被配置校验拒绝：生成阶段用摘要池替换官方打分尚未接线；读摘要页前要等
   `SummaryEntry.ready_event`（side stream 上记录）。
6. B>1、投机、CP/PP、prefix-cache 命中（新 token < 24）都走显式回退并记录原因，不构建。
7. 32K prompt 的 A/B 各配置每次只跑一条 prompt；`off` 四次落在 3.084～3.113 s，同一时段内噪声约 ±10 ms。
   一次只能跑一个 A/B 实例（第三轮的教训）。
