# Adaptive-HISA decode（Phase C，2026-09-19）

`MODE=adaptive_decode` 已接入 NSA indexer 的真实生成路径。默认仍为 `off`，`build_only` 仍只建图，不改变官方检索结果。

```bash
SGLANG_NSA_ADAPTIVE_HISA_MODE=adaptive_decode
# 可选：前 N 层保留官方全量打分，默认 N=1。
SGLANG_NSA_ADAPTIVE_HISA_FALLBACK_LAYERS=1
```

需要默认的 `SPLIT_BACKEND=gpu` 和 `BUILD_SUMMARIES=1`。当前生产目标为 DeepSeek-V3.2、CUDA、NSA、普通 B=1 decode；支持整模型 B=1 CUDA graph，以及 `SGLANG_NSA_FUSE_TOPK` 的逻辑索引/物理页表两种返回约定。

## 与 fixed-size HISA 相同的检索结构

1. 使用官方 paged DeepGEMM 在 FP8 摘要池上粗打分。每个叶子一条均值 key，估计量仍为 `sum_h weights[h] * relu(dot(q[h], mean_key))`。
2. 按粗分数降序排列叶子；按 `leaf_len` 累计 token 数，保留累计长度不超过剩余预算的完整叶子前缀。
3. 用 `(leaf_start, leaf_len)` 展开候选 token；无条件追加尚未封口的原始尾巴（不足 64 token）。候选表固定宽度 8192，未使用位置为 -1。
4. 调用 HISA `sparse_paged_mqa_triton`，以 `K=1` 将候选 token ID 解释成块 ID；沿用原 grouped kernel 的分页读 key、FP8 点积、scale、逐头 ReLU 和加权求和。没有独立 gather key 中间张量，也不再调用 dense scorer。
5. 使用 HISA `hisa_topk_fused` 原 radix selector；新 epilogue 从候选表查 token ID，将 Top-2048、坐标映射、有效性 mask 以及可选的物理页表转换融合在一个 CUDA kernel 中。

summary 只用于筛候选。精排和最终 attention 都使用原始 key/KV；没有用 summary 代替最终 attention。HISA 算子来源为 `origin/hisa_pr` 的 `e399d9f33`。固定块路径保留，`K=1` 新增候选 token 寻址以及读 key 前的负索引防护。融合 Top-K 保留原选择算法，只扩展输出映射。

与 HISA 对齐：首叶子强制进入候选；最后一个未封口块直接追加原始 token，恰好封口时强制选择最后一条 summary。均值粗分数与逐 token 分数的平均并不相同，因为逐头 ReLU 是非线性的；FP8 均值量化也有误差。当前仍用通用稳定排序和前缀和实现变长预算选择，这部分尚未优化。

## 回退条件与预算

默认 layer 0 使用官方全量打分。`FALLBACK_LAYERS=N` 可扩展为前 N 层；`LAYERS` 未选中的层也保持官方行为。

缺少有效地图、请求不满足原有 prefill 建图准入条件、B>1、speculative/CP/PP/DP-attention、上下文不超过候选预算、封口前缀不足 8192、叶子元数据容量不足，均回退官方路径。最大叶长限制为 `min(candidate_tokens-index_topk, (candidate_tokens-63)//2)`（默认 4064），保证首尾叶子及尾巴能装入预算，并有至少 2048 个候选。

prefill 自适应叶子冻结；从 `n_complete` 开始的后续区域采用固定 64-token 块，每完成一块便增量生成 summary。`incremental.py` 移植并适配 HISA completed-block TileLang 算子的 FP32 均值、FP8 重新量化和 SoA 写回流程，只调整输入起点和输出叶子偏移；单测与原 HISA 算子直接对照。水位保存在 GPU，支持间隔若干步后补齐全部新块。原始尾巴始终小于 64，不再因生成达到 8192 token 耗尽候选预算。

## CUDA graph 和请求生命周期

模型 graph 在启动时捕获，此时还没有真实请求地图。每层使用固定地址、固定最大容量的 decode 工作区捕获整个筛选流程；第一次 decode 前等待 ready event，复制该请求的叶子边界及 prefill 摘要到私有分页工作区，后续已完成的生成块追加在有效叶子之后。布局仍为 64 行 SoA，页表地址固定；容量 16384 时，每层私有摘要占约 2.06 MiB。

graph 回放前在主机检查 B=1、请求 epoch、地图完整性和元数据容量。只有所有需要自适应检索的层都具备有效绑定，才允许回放自适应模型 graph；否则执行 eager forward，各层按条件使用自适应或官方路径。短请求/不支持的 batch 因而可能不使用模型 graph，这是当前回退实现的性能限制。

释放请求时使绑定失效；复用同一 request slot 时通过新 epoch 重新绑定。旧地图不会因为 CUDA graph 持有固定地址而继续参与新请求。

## 代码与验证入口

- `adaptive_hisa/decode_runtime.py`：分数计算、token 预算、精排、层回退和 graph 工作区/准入。
- `adaptive_hisa/decode_kernels.py`：变长候选展开；旧 gather 保留作参考测试。
- `adaptive_hisa/incremental.py`：HISA completed-block 均值池的偏移适配、GPU 水位及首尾候选保留。
- `hisa/triton_kernels.py`、`hisa/hisa_topk_fused.py`：复用与扩展的原 HISA 精排、融合 Top-K 算子。
- `nsa_indexer.py`：decode hook；prefill Top-2048 不变。
- `cuda_graph_runner.py`：回放前的有效性检查。
- `test/registered/unit/test_adaptive_hisa_decode.py`：完整叶子选择、尾巴覆盖、非连续物理页、精排分数与 Top-K 阈值、graph 重绑定、epoch/释放/预算回退。

HISA 算子版本的测试输出、真实 query 对照及端到端日志保存在 `/DATA/disk0/qyl/data/adaptive_decode_hisa_ops_20260919`。早期 gather+dense 版本记录保留在 `/DATA/disk0/qyl/data/adaptive_decode_impl_20260919`，其速度不代表当前算子版本。两轮修改前的文件备份分别在对应 `original/` 子目录，用户已有未提交改动予以保留。
