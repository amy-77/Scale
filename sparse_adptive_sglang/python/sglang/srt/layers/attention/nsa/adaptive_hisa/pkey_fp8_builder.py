"""P-key 的建树、Split、Merge 和 summary 构建入口。

这个文件负责把一层的 ``K_fp8[N,128] + scale[N]`` 变成供 Decode 使用的
自适应 chunk：

1. 反量化 K，并为长度 1/2/4/.../256 的区间计算 Key-SSE 能量树；
2. 用全局 λ-DP 做 Split，先得到恰好 ``N / summary_compression`` 个叶子；
3. 用相邻 Ward cost 做 Merge，将叶子数继续压到 ``N / merge_target_divisor``；
4. 用最终叶子的 ``sum(K) / leaf_len`` 生成 FP8 summary。

重要说明：

* P-key 能量确实在本文件的 :func:`build_key_tree_from_fp8` 中计算；
* Split 的 DP/repair 主逻辑在 ``partition_gpu.py``，底层 Triton 算子在
  ``partition_kernels.py``；
* Merge 的调度在 ``partition_gpu.merge_sync_nonoverlap``，真正的 cost、
  matching、compact Triton 算子也在 ``partition_kernels.py``；
* 名字虽然叫 raw-FP8 builder，但 CUDA 默认路径现在直接从 ``K_fp8 + scale``
  计算 SSE 和叶子 ``sum(K)``，不再物化 ``[N,128]`` FP32 K 或完整 FP64 prefix。
  CPU / FP32 moment 仍保留物化 fallback。

环境变量 ``SGLANG_NSA_ADAPTIVE_HISA_RAW_FP8_BUILDER=1`` 开启此路径。
FP64 moment 下叶子划分应与旧 ``key_values`` 路径一致；FP32 moment 需要单独
通过叶子一致性和任务准确率验证。
"""
from __future__ import annotations

import os

import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import TreeLayout

_TRUE = {"1", "true", "yes", "on"}


def raw_fp8_builder_enabled() -> bool:
    """是否启用本文件的 P-key builder。"""
    return os.environ.get("SGLANG_NSA_ADAPTIVE_HISA_RAW_FP8_BUILDER", "0").strip().lower() in _TRUE


def _moment_dtype() -> torch.dtype:
    """SSE moment 的累加精度；默认 FP64，FP32 仅供实验。"""
    raw = os.environ.get("SGLANG_NSA_ADAPTIVE_HISA_KEY_MOMENT_DTYPE", "fp64").strip().lower()
    if raw in ("fp32", "float32", "f32"):
        return torch.float32
    return torch.float64


def dequant_keys_fp32(k_fp8: torch.Tensor, k_scale: torch.Tensor, n: int) -> torch.Tensor:
    """将前 ``n`` 个 FP8 K 反量化成行主序 ``[n,128]`` FP32。

    只给 CPU 和 FP32 moment fallback 使用；CUDA FP64 路径不调用它。
    """
    if k_fp8.dtype == torch.uint8:
        k_fp8 = k_fp8.view(torch.float8_e4m3fn)
    scale = k_scale.reshape(-1).to(torch.float32)
    return k_fp8[:n].to(torch.float32) * scale[:n, None]


def build_key_tree_from_fp8(
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    n_complete: int,
    *,
    atom: int,
    root: int,
    tree_cache: tuple[torch.Tensor, int] | None = None,
) -> tuple[torch.Tensor, TreeLayout]:
    """计算所有 dyadic 区间的 Key-SSE 能量树。

    对每个区间保存两个 moment：

    ``moment = sum(K)``，``moment_sq = sum(K²)``。

    区间的 P-key 能量为：

    ``SSE = sum_d(moment_sq[d] - moment[d]² / token_count)``。

    CUDA + FP64 走 Triton：每个 root 在寄存器中累计 moment，只把 SSE 树写回。
    FP32 实验或 CPU fallback 才使用下方 PyTorch 实现。

    ``tree_cache=(flat_old, n_old)`` 是同一请求、同一层上一个 chunk 建好的树
    （sealed prefix ``[0, n_old)``）。dyadic 节点不跨 root，所以旧前缀内每个
    节点的 SSE 在更长前缀的树里完全不变：只复制旧层、只为新增 root
    ``[n_old, n_complete)`` 读 FP8 K 建节点（``raw_fp8_tree_triton_incremental``）。
    结果与整棵重建逐位相同；随后的 λ 搜索 / Split / Merge 仍在整棵树上做，
    因为 M0、λ 和 merge target 都随前缀长度变化，旧 partition 不能复用。
    """
    if root % atom or (root // atom) & (root // atom - 1):
        raise ValueError("root/atom must be a power of two")
    if n_complete % root:
        raise ValueError(f"sealed length {n_complete} is not a multiple of root {root}")
    layout = TreeLayout(atom, root, n_complete)
    if (
        _moment_dtype() == torch.float64
        and k_fp8.is_cuda
        and k_fp8.shape[1] == 128
    ):
        from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_kernels as _pk

        if _pk.use_triton(k_fp8) and layout.levels > 1:
            if tree_cache is not None:
                flat_old, n_old = tree_cache
                return (
                    _pk.raw_fp8_tree_triton_incremental(
                        k_fp8, k_scale, layout, flat_old, int(n_old)
                    ),
                    layout,
                )
            return _pk.raw_fp8_tree_triton(k_fp8, k_scale, layout), layout
    return _build_key_tree_torch(k_fp8, k_scale, layout), layout


def _build_key_tree_torch(
    k_fp8: torch.Tensor, k_scale: torch.Tensor, layout: TreeLayout
) -> torch.Tensor:
    """CPU/FP32 reference. Materializes dequantized keys; not the CUDA path."""
    dtype = _moment_dtype()
    keys = dequant_keys_fp32(k_fp8, k_scale, layout.n_tokens)  # [N, D]
    grouped = keys.reshape(layout.n_atoms, layout.atom, keys.shape[1]).to(dtype)
    moment = grouped.sum(dim=1)
    moment_sq = grouped.square().sum(dim=1)
    del keys, grouped
    count = float(layout.atom)
    energies = []
    for lv in range(layout.levels):
        energies.append((moment_sq - moment.square() / count).sum(-1))
        if lv + 1 < layout.levels:
            moment = moment[0::2] + moment[1::2]
            moment_sq = moment_sq[0::2] + moment_sq[1::2]
            count *= 2.0
    return torch.cat(energies).to(torch.float64)


def leaf_totals_from_fp8(
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    leaf_start: torch.Tensor,
    leaf_len: torch.Tensor,
    n_complete: int,
) -> torch.Tensor:
    """计算每个叶子的 ``sum(K)``，返回 ``[M,128]`` FP64。

    CUDA + FP64 按叶子分段累加，不建立 ``[N,128]`` 或 ``[N+1,128]`` 中间量。
    """
    if (
        _moment_dtype() == torch.float64
        and k_fp8.is_cuda
        and k_fp8.shape[1] == 128
        and leaf_start.is_cuda
    ):
        from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_kernels as _pk

        if _pk.use_triton(k_fp8):
            return _pk.leaf_totals_fp8_triton(
                k_fp8, k_scale, leaf_start, leaf_len, n_complete
            )
    keys = dequant_keys_fp32(k_fp8, k_scale, n_complete)
    dtype = _moment_dtype()
    prefix = torch.zeros((n_complete + 1, keys.shape[1]), dtype=dtype, device=keys.device)
    torch.cumsum(keys.to(dtype), dim=0, out=prefix[1:])
    start = leaf_start.to(torch.long).clamp(0, n_complete)
    end = (leaf_start.to(torch.long) + leaf_len.to(torch.long)).clamp(0, n_complete)
    return (prefix[end] - prefix[start]).to(torch.float64)


def summaries_from_totals(
    totals: torch.Tensor,
    leaf_len: torch.Tensor,
    scale_fmt: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把最终叶子的 ``sum(K)`` 直接量化为 FP8 summary。"""
    from sglang.srt.layers.attention.nsa.adaptive_hisa.summary_kernels import requant_from_totals

    return requant_from_totals(totals, leaf_len, scale_fmt)


def build_partition_from_fp8(
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    n_tokens: int,
    cfg,
    *,
    profile: bool = False,
    tree_cache: tuple[torch.Tensor, int] | None = None,
):
    """完整执行一次 P-key ``建树 → Split → Merge``。

    ``tree_cache=(flat_old, n_old)``：上一个 chunk 同层的 SSE 树，见
    :func:`build_key_tree_from_fp8`。本次的树放在 ``part.meta["_tree_flat"]``
    供下一个 chunk 复用。

    这是本文件最重要的总入口。调用关系如下：

    ``build_key_tree_from_fp8``
      → ``search_lambda``（搜索能产生不超过 L/8 个叶子的 λ）
      → ``dp_leaf_mask``（按 λ 做动态规划并输出叶子 mask）
      → ``repair_leaves``（按最大 split gain 补到恰好 L/8）
      → ``emit_leaves``（生成 leaf_start / leaf_len）
      → ``merge_sync_nonoverlap``（按 Ward cost 从 L/8 合并到目标 L/D）。

    FP64 moment 下，叶子结果应与
    ``build_partition_gpu(key_values(...))`` 一致。
    """
    from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import (
        GpuPartition,
        dp_leaf_and_gain,
        emit_leaves,
        merge_sync_nonoverlap,
        repair_leaves,
        search_lambda,
        _mark,
    )
    from sglang.srt.layers.attention.nsa.adaptive_hisa.prefill_partition import (
        n_complete_tokens,
    )

    n_tokens = int(n_tokens)
    # 只对完整 root 做 partition；不足一个 root 的尾部不进入 P-key 树。
    done = n_complete_tokens(n_tokens, cfg.root)
    device = k_fp8.device
    # Split 的第一阶段预算 M0，默认 summary_compression=8，即 L/8。
    budget = done // cfg.summary_compression
    method = cfg.method_name if cfg.merge_policy == "off" else f"{cfg.method_name}-{cfg.merge_policy}"
    if cfg.merge_target_divisor:
        method += f"-target{cfg.merge_target_divisor}-rawfp8"
    base_meta = {
        "metric": cfg.partition_metric,
        "atom": cfg.atom,
        "root": cfg.root,
        "backend": "gpu_raw_fp8",
        "merge_policy": cfg.merge_policy,
        "merge_rounds": cfg.merge_rounds,
        "merge_alpha": cfg.merge_alpha,
        "max_merge_len": cfg.max_merge_len,
        "max_leaf_len": cfg.max_merge_len if cfg.merge_policy != "off" else cfg.root,
        "merge_target": cfg.merge_target(done),
        "raw_fp8": True,
    }
    if done == 0:
        zero = torch.zeros((), dtype=torch.int64, device=device)
        return GpuPartition(
            n_tokens, 0, 0,
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            zero, torch.zeros((), dtype=torch.float64, device=device), zero, zero,
            torch.zeros(0, dtype=torch.int64, device=device),
            torch.ones((), dtype=torch.bool, device=device),
            method, base_meta, {},
        )

    # ------------------------------------------------------------------
    # 阶段 1：计算 Key-SSE 能量树。
    # ------------------------------------------------------------------
    t0 = _mark(device, profile)
    flat, layout = build_key_tree_from_fp8(
        k_fp8, k_scale, done, atom=cfg.atom, root=cfg.root, tree_cache=tree_cache
    )
    t1 = _mark(device, profile)
    if not layout.n_roots <= budget <= layout.n_atoms:
        raise ValueError(f"M={budget} infeasible: roots={layout.n_roots}, atoms={layout.n_atoms}")
    # ------------------------------------------------------------------
    # 阶段 2（Split）：λ-DP 先选出 <= budget 个叶子。
    #
    # search_lambda 最终会进入 partition_kernels.py 的：
    #   _dp_count_bracket_kernel + _bracket_update_kernel
    # dp_leaf_mask 最终会进入：
    #   _dp_leaf_mask_kernel
    # ------------------------------------------------------------------
    lam, _count0, rounds, rounds_used = search_lambda(
        flat, layout, budget, candidates=cfg.lambda_candidates, rel_tol=cfg.lambda_rel_tol,
        max_rounds=cfg.lambda_max_rounds, deficit_tol=cfg.lambda_deficit_tol,
    )
    is_leaf, leaf_gain = dp_leaf_and_gain(flat, layout, lam)
    dp_leaves = is_leaf.sum()
    t2 = _mark(device, profile)
    # λ-DP 通常只保证叶子数 <= budget。repair 按 split gain 从大到小继续切，
    # 直到恰好 budget=L/8；fused gain 避免再次扫描 flat。
    is_leaf, repairs, passes = repair_leaves(
        flat, layout, is_leaf, budget, tie_passes=cfg.repair_tie_passes, gain=leaf_gain
    )
    num_leaves = is_leaf.sum()
    status_ok = num_leaves == budget
    start, length = emit_leaves(is_leaf, layout, budget)
    t3 = _mark(device, profile)
    # ------------------------------------------------------------------
    # 阶段 3（Merge）：把 L/8 的 dyadic 叶子继续合并到目标 L/D（默认 D=64）。
    #
    # merge_sync_nonoverlap 内部真正调用的关键 Triton kernel：
    #   _merge_cost_kernel    计算相邻叶子的 Ward cost
    #   _merge_match_kernel   选择互不重叠的相邻边
    #   _merge_compact_kernel 合并 totals/length 并压紧有效行
    # ------------------------------------------------------------------
    round_merges: list[torch.Tensor] = []
    count = num_leaves
    merge_target = cfg.merge_target(done)
    merge_rounds = cfg.merge_target_rounds if merge_target else cfg.merge_rounds
    finals = None

    
#     Merge 入口：
    if cfg.merge_policy == "sync_nonoverlap" and merge_rounds > 0:
        totals = leaf_totals_from_fp8(k_fp8, k_scale, start, length, done)
        target = (
            torch.full((), merge_target, dtype=torch.int64, device=device) if merge_target else None
        )
        start, length, count, finals, round_merges = merge_sync_nonoverlap(
            start, length, count, totals, lam * cfg.merge_alpha,
            rounds=merge_rounds, max_merge_len=cfg.max_merge_len, n_tokens=done, target=target,
        )
    t4 = _mark(device, profile)
    events = {}
    if profile:
        events = {
            "tree_s": (t0, t1), "dp_s": (t1, t2), "repair_s": (t2, t3),
            "merge_s": (t3, t4), "total_s": (t0, t4),
        }
    meta = dict(base_meta)
    meta.update({
        "lambda_rounds": rounds,
        "lambda_rounds_used": rounds_used,  # device int64 (early stop)
        "lambda_candidates": cfg.lambda_candidates,
        "repair_passes": passes,
    })
    storage = budget
    if merge_target:
        storage = max(int(merge_target), 1)
        storage = -(-storage // 64) * 64
        start = start[:storage].contiguous()
        length = length[:storage].contiguous()
        if finals is not None:
            finals = finals[:storage].contiguous()
        meta["premerge_M"] = budget
        meta["storage_M"] = storage
    part = GpuPartition(
        n_tokens,
        done,
        storage,
        start.to(torch.int32),
        length.to(torch.int32),
        count,
        lam,
        dp_leaves,
        repairs,
        torch.stack(round_merges) if round_merges else torch.zeros(0, dtype=torch.int64, device=device),
        status_ok,
        method,
        meta,
        events,
    )
    if finals is not None:
        # 最终叶子的 sum(K) 直接交给 summary writer，避免再次扫描 raw K。
        part.meta["_merge_totals"] = finals
    # 下一个 chunk 只需为新增 root 建节点（prefill_runtime 负责按请求缓存）。
    part.meta["_tree_flat"] = flat
    return part
