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
* 名字虽然叫 raw-FP8 builder，但当前实现仍会短暂物化 ``[N,128]`` FP32 K，
  只是避免了旧路径的 ``[128,N]`` 转置和完整 ``[128,N+1]`` FP64 prefix。

环境变量 ``SGLANG_NSA_ADAPTIVE_HISA_RAW_FP8_BUILDER=1`` 开启此路径。
FP64 moment 下叶子划分应与旧 ``key_values`` 路径一致；FP32 moment 需要单独
通过叶子一致性和任务准确率验证。
"""
from __future__ import annotations

import os

import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import TreeLayout
from sglang.srt.layers.attention.nsa.adaptive_hisa.summary_kernels import requant_summaries

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

    这里就是当前实现仍然会产生 ``[N,128]`` FP32 临时量的位置。
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
) -> tuple[torch.Tensor, TreeLayout]:
    """计算所有 dyadic 区间的 Key-SSE 能量树。

    对每个区间保存两个 moment：

    ``moment = sum(K)``，``moment_sq = sum(K²)``。

    区间的 P-key 能量为：

    ``SSE = sum_d(moment_sq[d] - moment[d]² / token_count)``。

    相邻两个子节点只需把 moment 相加，就能得到父节点，因此按
    ``atom → 2*atom → ... → root`` 自底向上建树。

    返回：

    * ``flat[n_nodes]``：所有节点的 FP64 SSE，按 level-major 排列；
    * ``layout``：各层节点数量、偏移和 token 长度。

    注意：这段当前是 PyTorch tensor 运算，不是
    ``partition_kernels._score_tree_kernel``。
    """
    if root % atom or (root // atom) & (root // atom - 1):
        raise ValueError("root/atom must be a power of two")
    if n_complete % root:
        raise ValueError(f"sealed length {n_complete} is not a multiple of root {root}")
    layout = TreeLayout(atom, root, n_complete)
    dtype = _moment_dtype()
    # 只处理可被 root 整除的 sealed prefix，尾部留给 raw tail。
    keys = dequant_keys_fp32(k_fp8, k_scale, n_complete)  # [N, D]
    # 第一级：每个 atom 的 sum(K) 与 sum(K²)，shape=[n_atoms,D]。
    grouped = keys.reshape(layout.n_atoms, atom, keys.shape[1]).to(dtype)
    moment = grouped.sum(dim=1)
    moment_sq = grouped.square().sum(dim=1)
    del keys, grouped
    count = float(atom)
    energies = []
    for lv in range(layout.levels):
        # 这行就是 P-key / Key-SSE 的核心公式。
        energies.append((moment_sq - moment.square() / count).sum(-1))
        if lv + 1 < layout.levels:
            # 两个相邻子节点合并成父节点；moment 可以直接相加。
            moment = moment[0::2] + moment[1::2]
            moment_sq = moment_sq[0::2] + moment_sq[1::2]
            count *= 2.0
    flat = torch.cat(energies).to(torch.float64)
    return flat, layout


def leaf_totals_from_fp8(
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    leaf_start: torch.Tensor,
    leaf_len: torch.Tensor,
    n_complete: int,
) -> torch.Tensor:
    """计算每个叶子的 ``sum(K)``，返回 ``[M,128]`` FP64。

    Merge 的 Ward cost 和最终 summary 都需要叶子总和。这里沿 token 维建立
    ``[N+1,128]`` prefix，再用 ``prefix[end]-prefix[start]`` 取区间和。
    """
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
    """把最终叶子的 ``sum(K)`` 转成 ``mean(K)``，再量化为 FP8 summary。"""
    denom = leaf_len.to(totals.dtype).clamp_min(1).unsqueeze(-1)
    means = (totals / denom).to(torch.float32)
    means = torch.where(leaf_len.unsqueeze(-1) > 0, means, torch.zeros_like(means))
    return requant_summaries(means, scale_fmt)


def build_partition_from_fp8(
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    n_tokens: int,
    cfg,
    *,
    profile: bool = False,
):
    """完整执行一次 P-key ``建树 → Split → Merge``。

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
        dp_leaf_mask,
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
        k_fp8, k_scale, done, atom=cfg.atom, root=cfg.root
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
    lam, _count0, rounds = search_lambda(
        flat, layout, budget, candidates=cfg.lambda_candidates, rel_tol=cfg.lambda_rel_tol
    )
    is_leaf = dp_leaf_mask(flat, layout, lam)
    dp_leaves = is_leaf.sum()
    t2 = _mark(device, profile)
    # λ-DP 通常只保证叶子数 <= budget。repair 按 split gain 从大到小继续切，
    # 直到恰好 budget=L/8；关键 kernel 是 _gain_kernel/_staircase_kernel。
    is_leaf, repairs, passes = repair_leaves(
        flat, layout, is_leaf, budget, tie_passes=cfg.repair_tie_passes
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
    return part
