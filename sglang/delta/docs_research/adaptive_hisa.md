# Adaptive-HISA：目前做了什么

长上下文里，模型没法对前面每一个 token 都做完整 attention。DeepSeek 的做法是：先用一个便宜的打分器，从十几万个已有 token 里挑出大约 2048 个「看起来重要」的，真正的 attention 只打这些。

挑这 2048 个本身也不便宜——每个新生成的 token 都要对整段前文打一遍分。HISA 的加速办法是：先把前文切成等长小段，每段只留一个平均值，用平均值粗选，再只对选中的小段做精确打分。

我们在问一件事：**小段为什么必须一样长？** 有的地方分数很平，一大段用一个平均值就够；有的地方分数很尖，应该切得很细，甚至细到单个 token。Adaptive-HISA 就是按这个想法，在读完 prompt 之后给每一层画一张「哪里该细、哪里该粗」的地图，生成阶段复用这张地图。

模型本身、attention 公式、最后仍然只选 2048 个 token，都没改。开关默认关着，线上行为和原来一模一样。

## 我们试出来的结果

在真实的 128k 数据上对比「精确挑出的 2048 个」和「用平均值粗筛后再精排」能找回多少。

1. **等长切块几乎没有改进空间。** 切成 16 个一组还是 64 个一组，按 key 的几何形状决定哪里该细，效果和永远用固定 64 几乎一样。原因是：这些 key 经过旋转和 Hadamard 之后，各段看起来差不多，算法找不到「该细」的地方，最后还是均匀切。
2. **真正丢的是很远的 token。** 靠近当前位置的几乎都能找回；三万 token 以外的重要位置，粗筛几乎全丢（大约只找回 9%）。平均值代表不了远处那些尖峰。
3. **要按真实打分来切，而且允许切到单个 token。** 读完 prompt 时，拿最后大约 24 个 query 对全文打一遍分，看哪一段分数波动大就切细、哪一段太平就合并。这样远处重要 token 的找回率大约从 9% 到 97%。切完后再把相邻、分数差不多的段并回去，摘要数量能少将近一半，质量只掉一点。
4. **浅层（layer 0）这条路走不通。** 那里的重要 token 是散落在各段里的孤立尖峰，平均值再细也排不出来，以后要单独回退。

所以结论不是「自适应比固定块强一点」，而是：**必须看见真实分数，才知道预算该花在哪。**

## 代码做到哪了

读 prompt 结束时建这张地图，已经接到 SGLang 上。打开开关后：

- 调度器知道「这条请求的 prompt 这一拍写完了」
- 每一层留下最后 24 个 query
- 用官方 indexer 已经准备好的 key，按上面的分数波动切一遍
- **读 prompt 时官方选出的 2048 个 token 完全不动**，我们只是把地图存下来，给后面生成用

Phase B（2026-09-19）把这条链在 GPU 上补完整了：切分、两轮合并、以及每个叶子的 FP8 摘要都在主 stream 上算完并写进一个分页的摘要池（和官方 index-K 池同样的页布局），生成阶段可以直接按页表读。旧的 CPU numba 路径（此前 A/B 过的 `v7_reuse`）保留为参考实现，两条路径逐位对齐。

生成阶段用这张地图去替换官方打分还没有接线；`MODE=adaptive_decode` 目前会被配置校验直接拒绝。旧的 P-radius/CPU decode 原型移到了 `adaptive_hisa/legacy/`，不参与任何路径。

单测在 `test/registered/unit/test_adaptive_hisa_prefill_partition.py`，不需要起 server。离线实验和完整数字在另一份仓库的 `experiments/adaptive_hisa/results_20260918.md`；Phase B 的验收记录见 `adaptive_hisa_phase_b_acceptance.md`。

## 开关（默认全关）

所有变量以 `SGLANG_NSA_ADAPTIVE_HISA_` 开头，由 `adaptive_hisa/config.py` 统一解析和校验，互相冲突的组合会在启动时报错而不是静默混用。

| 环境变量 | 实际效果 |
| --- | --- |
| `MODE=off\|build_only` | `build_only`：读完 prompt 后建地图和摘要，官方 Top-2048 不变（默认 `off`） |
| `SPLIT_BACKEND=gpu\|cpu_reference` | 切分/合并在 GPU 主 stream 上做（默认），或走 numba 参考实现 |
| `MERGE_POLICY=off\|sync_nonoverlap\|heap_reference` | 不合并 / 两轮同步不重叠合并（默认，论文算法）/ 旧的顺序堆合并 |
| `MERGE_ROUNDS`、`MERGE_ALPHA`、`MAX_MERGE_LEN` | 合并轮数（默认 2）、阈值倍率（默认 1）、合并后叶子长度上限（默认 0 = 不限） |
| `BUILD_SUMMARIES=1` | 写 FP8 叶子摘要到摘要池（仅 `gpu` 后端，默认开） |
| `GRAPH_BUILD=1` | 仅 `gpu`：每个请求长度只捕获一次 CUDA graph，61 层各回放一次（默认开；捕获失败自动退回逐 launch） |
| `GPU_STREAM=side\|main` | 仅 `gpu`：构建放到第二条 stream 与后续层重叠、forward 结束时 join（默认 `side`，32K 时 +0.04 s）；`main` 为同步构建（+0.14 s） |
| `PARTITION_REUSE_LOGITS=1` | 校准分数搭官方 FP8 打分器的便车而不是重新扫一遍 K（默认开） |
| `PARTITION_OVERLAP=1` | 仅 `cpu_reference`：numba 阶段放到工作线程与后续层重叠 |
| `FORWARD_TIMING=1`、`PARTITION_LOG_LAYERS=1` | A/B 计时日志、每层阶段耗时与直方图 |

旧变量 `PREFILL_PARTITION=1` 仍等价于 `MODE=build_only`；`PARTITION_MERGE=1` 不再隐含算法选择，必须显式给 `MERGE_POLICY`。

## 下一步

把读 prompt 时建好的地图和摘要池接到生成阶段，真正替换官方那次「对全文打分、再取 Top-2048」。浅层要有回退。构建已经在独立 stream 上与后续层重叠（`GPU_STREAM=side`），读摘要页的一方要先等 `SummaryEntry.ready_event`。在这之前，默认路径不会变。
