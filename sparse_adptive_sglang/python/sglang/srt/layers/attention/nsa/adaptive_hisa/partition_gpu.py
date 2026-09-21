"""GPU backend for the P-key partition: tree, global λ-DP, exact repair, merge.

Everything runs on the *current* stream, never reads a device value back
(no host sync anywhere in the build), and never decides a tensor shape from
device data. Shapes are fixed by host-known integers (``N_complete``,
``M0 = N_complete / summary_compression``); the actual leaf count after
merging is a device scalar (``num_leaves``).

On CUDA the per-root tree passes and the merge rounds are fused Triton
kernels (``partition_kernels``); the torch implementations in this module are
the reference for those kernels and the path used on CPU tensors. At 32K the
whole build is ~1.9 ms of host time and ~210 launches per layer.

Algorithms are the ones in ``partition_reference`` (parity-tested):

* ``dp_counts``: penalised keep/split DP for ``K`` λ candidates; ties keep
  the parent. Fused: one program per (root, 16 candidates).
* λ search: ``K``-way bracketing with the bracket kept on the device until the
  relative width is below ``lambda_rel_tol`` (the CPU reference bisects to the
  same tolerance). Fused: two launches per round (counts, bracket update).
* repair: exact replay of the heap order (gain desc, start asc, level asc,
  idx asc) in a single pass. Best-first traversal visits nodes in ascending
  lexicographic order of their *staircase* (the record-low keys along the
  path from the frontier leaf down); the first ``deficit`` reachable nodes in
  that order are exactly the popped set. ``repair_leaves_passes`` keeps the
  older multi-pass bottleneck formulation as a cross-check.
* merge ``sync_nonoverlap``: greedy non-overlapping matching by ``(cost,
  start)`` on a path is computed exactly in parallel: an eligible edge is
  selected iff its distance to the local-minimum-rank edge of its monotone
  run is even; a local-maximum edge is selected iff neither neighbour is.
  Fused: three launches per round (costs, matching + compaction map, gather).

FP64 is used for energies, DP and costs to match the reference bit for bit
where the arithmetic is identical.
"""

from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_kernels as _pk
from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_select as _sel
from sglang.srt.layers.attention.nsa.adaptive_hisa.config import PartitionConfig
from sglang.srt.layers.attention.nsa.adaptive_hisa.prefill_partition import (
    PrefillPartition,
    n_complete_tokens,
)

_NEG_INF = float("-inf")
_BIG = 1 << 40


# --------------------------------------------------------------------------- #
# Tree layout and score tree
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TreeLayout:
    atom: int
    root: int
    n_tokens: int

    @property
    def levels(self) -> int:
        return (self.root // self.atom).bit_length()

    @property
    def n_atoms(self) -> int:
        return self.n_tokens // self.atom

    @property
    def n_roots(self) -> int:
        return self.n_tokens // self.root

    def size(self, level: int) -> int:
        return self.n_atoms >> level

    def length(self, level: int) -> int:
        return self.atom << level

    @property
    def offs(self) -> tuple[int, ...]:
        out = [0]
        for lv in range(self.levels):
            out.append(out[-1] + self.size(lv))
        return tuple(out)

    def slice(self, level: int) -> slice:
        offs = self.offs
        return slice(offs[level], offs[level + 1])

    @property
    def n_nodes(self) -> int:
        return self.offs[-1]


@dataclass
class _LayoutStatics:
    """Tensors that depend only on the layout (cached per layout and device)."""

    start: torch.Tensor  # [n_nodes] node start token
    length: torch.Tensor  # [n_nodes] node length
    tiebreak: torch.Tensor  # [n_nodes] (start << 36) | (level << 32) | idx
    tie_order: torch.Tensor  # argsort(tiebreak)
    parent: torch.Tensor  # [n_nodes] parent node index (self at roots)


_STATICS: dict[tuple, _LayoutStatics] = {}


def layout_statics(layout: TreeLayout, device: torch.device) -> _LayoutStatics:
    key = (layout.atom, layout.root, layout.n_tokens, str(device))
    hit = _STATICS.get(key)
    if hit is not None:
        return hit
    starts, lengths, ties, parents = [], [], [], []
    offs = layout.offs
    for lv in range(layout.levels):
        idx = torch.arange(layout.size(lv), device=device)
        starts.append(idx * layout.length(lv))
        lengths.append(torch.full_like(idx, layout.length(lv)))
        ties.append(((idx * layout.length(lv)) << 36) | (lv << 32) | idx)
        if lv + 1 < layout.levels:
            parents.append(offs[lv + 1] + (idx >> 1))
        else:
            parents.append(offs[lv] + idx)
    tiebreak = torch.cat(ties)
    statics = _LayoutStatics(
        start=torch.cat(starts),
        length=torch.cat(lengths),
        tiebreak=tiebreak,
        tie_order=torch.sort(tiebreak, stable=True).indices,
        parent=torch.cat(parents),
    )
    if len(_STATICS) > 64:
        _STATICS.clear()
    _STATICS[key] = statics
    return statics


def build_score_tree(
    scores: torch.Tensor, *, atom: int, root: int
) -> tuple[torch.Tensor, TreeLayout, torch.Tensor]:
    """FP64 Key-SSE per dyadic node, level-major flat, plus key prefix sums.

    Returns ``(flat_energy[n_nodes], layout, prefix[C, N+1])``. ``prefix`` is
    used for exact leaf score sums after merging.
    """
    if scores.dim() != 2:
        raise ValueError("scores must be [C,N]")
    n_tokens = int(scores.shape[1])
    if root % atom or (root // atom) & (root // atom - 1):
        raise ValueError("root/atom must be a power of two")
    if n_tokens % root:
        raise ValueError(f"sealed length {n_tokens} is not a multiple of root {root}")
    layout = TreeLayout(atom, root, n_tokens)
    queries = scores.shape[0]
    if layout.levels > 1 and _pk.use_triton(scores):
        flat = _pk.score_tree_triton(scores, layout)
        prefix = torch.zeros((queries, n_tokens + 1), dtype=torch.float64, device=scores.device)
        # Row-chunked scan: rows are independent, so each chunk's result is
        # bit-identical to one full cumsum, but the FP64 upcast temporary is
        # bounded (~16 MiB) instead of a full [C, N] copy (76 MiB for P-key at
        # 76K tokens; that copy was the allocation that OOMed inside graph
        # capture on a server with little free device memory).
        step = max(1, (16 << 20) // (8 * max(n_tokens, 1)))
        for r0 in range(0, queries, step):
            r1 = min(queries, r0 + step)
            torch.cumsum(scores[r0:r1], dim=1, dtype=torch.float64, out=prefix[r0:r1, 1:])
        return flat, layout, prefix
    values = scores.to(torch.float64)
    grouped = values.reshape(queries, layout.n_atoms, atom)
    moment = grouped.sum(-1).transpose(0, 1).contiguous()
    moment_sq = grouped.square().sum(-1).transpose(0, 1).contiguous()
    count = float(atom)
    energies = []
    for lv in range(layout.levels):
        energies.append((moment_sq - moment.square() / count).sum(-1))
        if lv + 1 < layout.levels:
            moment = moment[0::2] + moment[1::2]
            moment_sq = moment_sq[0::2] + moment_sq[1::2]
            count *= 2.0
    flat = torch.cat(energies)
    prefix = torch.zeros((queries, n_tokens + 1), dtype=torch.float64, device=values.device)
    torch.cumsum(values, dim=1, out=prefix[:, 1:])
    return flat, layout, prefix


# --------------------------------------------------------------------------- #
# λ-DP
# --------------------------------------------------------------------------- #


def dp_counts(flat: torch.Tensor, layout: TreeLayout, lams: torch.Tensor) -> torch.Tensor:
    """给每个候选 λ 计算惩罚 DP 的总叶子数。"""
    if layout.levels == 1:
        return torch.full((lams.numel(),), layout.n_roots, dtype=torch.int64, device=flat.device)
    if _pk.use_triton(flat):
        return _pk.dp_counts_triton(flat, layout, lams)
    lams = lams.to(torch.float64).reshape(-1, 1)
    cost = flat[layout.slice(0)][None, :] + lams
    cnt = None
    for lv in range(1, layout.levels):
        energy = flat[layout.slice(lv)]
        split = cost[:, 0::2] + cost[:, 1::2]
        keep = energy[None, :] + lams
        do_split = split < keep
        cost = torch.where(do_split, split, keep)
        if cnt is None:
            cnt = torch.where(do_split, 2, 1)
        else:
            cnt = torch.where(do_split, cnt[:, 0::2] + cnt[:, 1::2], 1)
    return cnt.sum(dim=1)


def dp_leaf_mask(flat: torch.Tensor, layout: TreeLayout, lam: torch.Tensor) -> torch.Tensor:
    """用最终 λ 重新跑 DP，并从 root 向下生成扁平叶子 mask。"""
    leaf, _gain = dp_leaf_and_gain(flat, layout, lam)
    return leaf


def dp_leaf_and_gain(
    flat: torch.Tensor, layout: TreeLayout, lam: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One fused traversal when Triton is available; gain is ``None`` on fallback."""
    if layout.levels > 1 and _pk.use_triton(flat):
        return _pk.dp_leaf_and_gain_triton(flat, layout, lam)
    return _dp_leaf_mask_torch(flat, layout, lam), None


def _dp_leaf_mask_torch(flat: torch.Tensor, layout: TreeLayout, lam: torch.Tensor) -> torch.Tensor:
    cost = flat[layout.slice(0)] + lam
    do_split = [torch.zeros(layout.size(0), dtype=torch.bool, device=flat.device)]
    for lv in range(1, layout.levels):
        split = cost[0::2] + cost[1::2]
        keep = flat[layout.slice(lv)] + lam
        ds = split < keep
        cost = torch.where(ds, split, keep)
        do_split.append(ds)
    is_leaf = torch.zeros(layout.n_nodes, dtype=torch.bool, device=flat.device)
    active = torch.ones(layout.n_roots, dtype=torch.bool, device=flat.device)
    for lv in range(layout.levels - 1, 0, -1):
        is_leaf[layout.slice(lv)] = active & ~do_split[lv]
        active = torch.repeat_interleave(active & do_split[lv], 2)
    is_leaf[layout.slice(0)] = active
    return is_leaf


def search_lambda(
    flat: torch.Tensor,
    layout: TreeLayout,
    m_leaves: int,
    *,
    candidates: int,
    rel_tol: float,
    max_rounds: int = 0,
    deficit_tol: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """搜索使 ``leaf_count <= M`` 的最小 λ。

    λ 越大，每个叶子的惩罚越大，算法越倾向保留粗节点，因此叶子越少。
    轮数由 ``rel_tol``（区间相对宽度）决定，``max_rounds > 0`` 可再加上限。
    ``deficit_tol >= 0`` 打开按叶子数的 device 侧提前收敛：一旦可行端 ``hi``
    的 DP 叶子数满足 ``M - count <= deficit_tol * M``，剩余轮次立即返回
    （launch 数不变，省掉整棵树的 DP），差额交给 exact repair。
    返回 ``(lambda, count_at_zero, rounds, rounds_used)``，标量留在 device 上。
    """
    device = flat.device
    root_energy = flat[layout.slice(layout.levels - 1)]
    hi = root_energy.max().clamp_min(0.0) * 2.0 + 1.0
    lo = torch.zeros((), dtype=torch.float64, device=device)
    count0 = dp_counts(flat, layout, lo.reshape(1))[0]
    zero_ok = count0 <= m_leaves
    k = int(candidates)
    rounds = max(1, math.ceil(math.log(1.0 / rel_tol) / math.log(k + 1)))
    if max_rounds > 0:
        rounds = min(rounds, int(max_rounds))
    tol_leaves = int(math.floor(deficit_tol * m_leaves)) if deficit_tol >= 0 else -1
    # hi >= 2*max root SSE + 1 exceeds every split gain, so no root is split.
    hi_count = torch.full((), float(layout.n_roots), dtype=torch.float64, device=device)
    if layout.levels > 1 and _pk.use_triton(flat) and k & (k - 1) == 0:
        zero = torch.zeros((), dtype=torch.float64, device=device)
        bracket = torch.stack((lo, hi, hi_count, zero, zero))
        _pk.search_rounds_triton(
            flat, layout, bracket, m_leaves, candidates=k, rounds=rounds, deficit_tol=tol_leaves
        )
        best = torch.where(zero_ok, lo, bracket[1])
        return best, count0, rounds, bracket[4].to(torch.int64)
    frac = torch.arange(1, k + 1, dtype=torch.float64, device=device) / (k + 1)
    done = torch.zeros((), dtype=torch.bool, device=device)
    used = torch.zeros((), dtype=torch.int64, device=device)
    for _ in range(rounds):
        cands = lo + (hi - lo) * frac
        # counts are monotone in λ, so feasibility is a suffix of the grid:
        # grid = [lo, c_1..c_K, hi]; with n feasible candidates the first
        # feasible point is grid[K - n + 1] and the last infeasible grid[K - n].
        counts = dp_counts(flat, layout, cands)
        n_feasible = (counts <= m_leaves).sum()
        grid = torch.cat((lo.reshape(1), cands, hi.reshape(1)))
        idx = (k - n_feasible).reshape(1)
        cgrid = torch.cat((counts.to(torch.float64), hi_count.reshape(1)))
        new_lo, new_hi, new_count = grid[idx][0], grid[idx + 1][0], cgrid[idx][0]
        lo = torch.where(done, lo, new_lo)
        hi = torch.where(done, hi, new_hi)
        hi_count = torch.where(done, hi_count, new_count)
        used = used + (~done).to(torch.int64)
        if tol_leaves >= 0:
            done = done | (m_leaves - hi_count <= tol_leaves)
    best = torch.where(zero_ok, lo * 0.0, hi)
    return best, count0, rounds, used


# --------------------------------------------------------------------------- #
# Exact repair
# --------------------------------------------------------------------------- #


def _node_keys(flat: torch.Tensor, layout: TreeLayout) -> tuple[torch.Tensor, torch.Tensor]:
    """Heap key per node: ``(gain desc, start asc, level asc, idx asc)``.

    ``gain`` is ``-inf`` for atoms (never splittable). The tiebreak packs
    ``start``, ``level`` and ``idx`` into one int64 (smaller pops first).
    """
    device = flat.device
    tiebreak = layout_statics(layout, device).tiebreak
    if layout.levels > 1 and _pk.use_triton(flat):
        return _pk.gain_triton(flat, layout), tiebreak
    gain = torch.full((layout.n_nodes,), _NEG_INF, dtype=torch.float64, device=device)
    for lv in range(1, layout.levels):
        child = flat[layout.slice(lv - 1)]
        gain[layout.slice(lv)] = flat[layout.slice(lv)] - (child[0::2] + child[1::2])
    return gain, tiebreak


def _later(ga, ta, gb, tb) -> torch.Tensor:
    """True where key ``b`` pops *after* key ``a`` (b has lower priority)."""
    return (gb < ga) | ((gb == ga) & (tb > ta))


def _bottleneck(
    gain: torch.Tensor,
    tiebreak: torch.Tensor,
    frontier: torch.Tensor,
    layout: TreeLayout,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reachability and bottleneck key of every splittable node below ``frontier``."""
    device = gain.device
    reach = torch.zeros(layout.n_nodes, dtype=torch.bool, device=device)
    m_gain = torch.full((layout.n_nodes,), _NEG_INF, dtype=torch.float64, device=device)
    m_tie = torch.zeros(layout.n_nodes, dtype=torch.int64, device=device)
    parent_reach = torch.zeros(layout.n_roots, dtype=torch.bool, device=device)
    parent_gain = torch.full((layout.n_roots,), _NEG_INF, dtype=torch.float64, device=device)
    parent_tie = torch.zeros(layout.n_roots, dtype=torch.int64, device=device)
    for lv in range(layout.levels - 1, 0, -1):
        sl = layout.slice(lv)
        own = frontier[sl]
        g, t = gain[sl], tiebreak[sl]
        via = parent_reach & ~own
        take_own = _later(parent_gain, parent_tie, g, t)  # own key is the new bottleneck
        mg = torch.where(own, g, torch.where(via, torch.where(take_own, g, parent_gain), _NEG_INF))
        mt = torch.where(own, t, torch.where(via, torch.where(take_own, t, parent_tie), 0))
        r = own | via
        reach[sl] = r
        m_gain[sl] = mg
        m_tie[sl] = mt
        parent_reach = torch.repeat_interleave(r, 2)
        parent_gain = torch.repeat_interleave(mg, 2)
        parent_tie = torch.repeat_interleave(mt, 2)
    return reach, m_gain, m_tie


def _children_mask(mask: torch.Tensor, layout: TreeLayout) -> torch.Tensor:
    """``out[v] = mask[parent(v)]`` for non-root nodes, False at roots."""
    out = mask[layout_statics(layout, mask.device).parent]
    out[layout.slice(layout.levels - 1)] = False
    return out


def _pop_top(
    gain: torch.Tensor,
    tiebreak: torch.Tensor,
    frontier: torch.Tensor,
    deficit: torch.Tensor,
    layout: TreeLayout,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One bottleneck pass. Returns ``(popped, tie_node, residual)``.

    ``popped`` are the nodes with a strictly better bottleneck than the
    ``deficit``-th plus the tie node ``u`` itself; ``residual`` pops remain
    inside ``u``'s subtree (0 when there was no partial group).
    """
    reach, m_gain, m_tie = _bottleneck(gain, tiebreak, frontier, layout)
    # Sort candidates by priority (gain desc, tiebreak asc); non-candidates last.
    idx1 = torch.sort(m_tie, stable=True).indices
    idx2 = torch.sort(m_gain[idx1], descending=True, stable=True).indices
    order = idx1[idx2]
    have = deficit >= 1
    pick = order[(deficit - 1).clamp_min(0)]
    theta_g, theta_t = m_gain[pick], m_tie[pick]
    strictly_better = reach & _later(m_gain, m_tie, theta_g.expand_as(m_gain), theta_t.expand_as(m_tie))
    tie_node = reach & (gain == theta_g) & (tiebreak == theta_t)
    popped = (strictly_better | tie_node) & have
    residual = torch.where(have, deficit - strictly_better.sum() - 1, deficit * 0)
    return popped, tie_node & have, residual


def _pack_ranks(ranks: torch.Tensor, bits: int) -> list[torch.Tensor]:
    """Pack ``ranks[n, L]`` (``-1`` padding) into ``ceil(L / per)`` int64 keys.

    Lexicographic order of the key list equals lexicographic order of the rows.
    """
    per = 63 // bits
    shifted = (ranks + 1).to(torch.int64)  # padding -1 -> 0 sorts first
    keys = []
    for k0 in range(0, ranks.shape[1], per):
        chunk = shifted[:, k0 : k0 + per]
        key = torch.zeros(ranks.shape[0], dtype=torch.int64, device=ranks.device)
        for c in range(chunk.shape[1]):
            key = (key << bits) | chunk[:, c]
        if chunk.shape[1] < per:
            key = key << (bits * (per - chunk.shape[1]))
        keys.append(key)
    return keys


def _staircase(
    rank: torch.Tensor, frontier: torch.Tensor, layout: TreeLayout
) -> tuple[torch.Tensor, torch.Tensor]:
    """Best-first pop order of a tree as a lexicographic key.

    Popping a max-heap that starts with the frontier leaves and pushes the
    children of every popped node visits nodes in ascending lexicographic
    order of their *staircase*: the strictly increasing (in priority)
    sequence of record-low keys along the path from the frontier node down,
    padded with ``-1`` (a proper prefix pops before its extensions). ``rank``
    is the heap priority (0 pops first). Returns ``(stairs[n_nodes, L], reach)``.
    """
    device = rank.device
    levels = layout.levels
    stairs = torch.full((layout.n_nodes, levels), -1, dtype=torch.int64, device=device)
    reach = torch.zeros(layout.n_nodes, dtype=torch.bool, device=device)
    cols = torch.arange(levels, device=device)
    top = layout.slice(levels - 1)
    parent_s = torch.where(
        (cols[None, :] == 0) & frontier[top][:, None], rank[top][:, None], -1
    )
    parent_reach = frontier[top].clone()
    stairs[top] = parent_s
    reach[top] = parent_reach
    for lv in range(levels - 2, -1, -1):
        sl = layout.slice(lv)
        own_f, own_r = frontier[sl], rank[sl]
        p_s = torch.repeat_interleave(parent_s, 2, dim=0)
        p_reach = torch.repeat_interleave(parent_reach, 2)
        keep = p_s > own_r[:, None]  # records with lower priority than this node
        n_keep = keep.sum(dim=1, keepdim=True)
        ext = torch.where(keep, p_s, -1)
        ext = torch.where(cols[None, :] == n_keep, own_r[:, None], ext)
        via = p_reach & ~own_f
        own_s = torch.where(cols[None, :] == 0, own_r[:, None], -1)
        s = torch.where(own_f[:, None], own_s, torch.where(via[:, None], ext, -1))
        r = own_f | via
        stairs[sl] = s
        reach[sl] = r
        parent_s, parent_reach = s, r
    return stairs, reach


def repair_leaves(
    flat: torch.Tensor,
    layout: TreeLayout,
    is_leaf: torch.Tensor,
    m_leaves: int,
    *,
    tie_passes: str | int = "auto",
    gain: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """按最大 split gain 继续切分，直到叶子数恰好等于 ``m_leaves``。

    ``gain`` 可由 fused leaf-mask kernel 传入，避免再次读取整棵 SSE 树。
    """
    device = flat.device
    statics = layout_statics(layout, device)
    if gain is None:
        gain, _ = _node_keys(flat, layout)
    deficit = m_leaves - is_leaf.sum()
    frontier = is_leaf.clone()
    frontier[layout.slice(0)] = False
    # heap priority: gain desc, tiebreak asc -> rank 0 pops first
    idx1 = statics.tie_order
    idx2 = torch.sort(gain[idx1], descending=True, stable=True).indices
    order = idx1[idx2]
    arange = torch.arange(layout.n_nodes, device=device)
    rank = torch.empty_like(order)
    rank[order] = arange
    bits = max(1, int(layout.n_nodes + 1).bit_length())
    if layout.levels > 1 and _pk.use_triton(flat) and bits * _pk.STAIR_PER <= 63:
        packed, reach = _pk.staircase_keys_triton(rank, frontier, layout, bits)
        if _sel.available(packed):
            # Pop the `deficit` lexicographically smallest reachable keys by
            # exact radix *selection*: keys are unique per reachable node, so
            # "<= k-th smallest" is exactly the first `deficit` of the sorted
            # order, without sorting (and re-permuting) 3 int64 arrays.
            popped = _sel.lex_select_mask(packed, reach, deficit, key_bits=bits * _pk.STAIR_PER)
            new_leaf = (is_leaf & ~popped) | (_children_mask(popped, layout) & ~popped)
            return new_leaf, popped.sum(), 1
        keys = [packed[:, k] for k in range(packed.shape[1])]
    else:
        stairs, reach = _staircase(rank, frontier, layout)
        keys = _pack_ranks(stairs, bits)
    # unreachable nodes sort last
    keys[0] = torch.where(reach, keys[0], torch.iinfo(torch.int64).max)
    perm = arange
    for key in reversed(keys):  # least significant key first (stable radix)
        perm = perm[torch.sort(key[perm], stable=True).indices]
    pos = torch.empty_like(perm)
    pos[perm] = arange
    popped = reach & (pos < deficit)
    new_leaf = (is_leaf & ~popped) | (_children_mask(popped, layout) & ~popped)
    return new_leaf, popped.sum(), 1


def repair_leaves_passes(
    flat: torch.Tensor,
    layout: TreeLayout,
    is_leaf: torch.Tensor,
    m_leaves: int,
    *,
    tie_passes: str | int = "auto",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Reference multi-pass bottleneck repair (kept for parity tests).

    With ``tie_passes="auto"`` the residual deficit after the first pass is
    read once (one stream sync); an integer runs that many extra masked passes.
    """
    gain, tiebreak = _node_keys(flat, layout)
    deficit = m_leaves - is_leaf.sum()
    frontier = is_leaf.clone()
    frontier[layout.slice(0)] = False
    popped_all = torch.zeros_like(is_leaf)
    popped, tie_node, residual = _pop_top(gain, tiebreak, frontier, deficit, layout)
    popped_all |= popped
    passes = 1
    max_extra = layout.levels - 1
    extra = max_extra if tie_passes == "auto" else int(tie_passes)
    for _ in range(extra):
        if tie_passes == "auto" and int(residual.item()) <= 0:
            break
        frontier = _children_mask(tie_node, layout)
        frontier[layout.slice(0)] = False
        popped, tie_node, residual = _pop_top(gain, tiebreak, frontier, residual, layout)
        popped_all |= popped
        passes += 1
    new_leaf = (is_leaf & ~popped_all) | (_children_mask(popped_all, layout) & ~popped_all)
    return new_leaf, popped_all.sum(), passes


# --------------------------------------------------------------------------- #
# Leaf emission and merge
# --------------------------------------------------------------------------- #


def _node_coords(layout: TreeLayout, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    statics = layout_statics(layout, device)
    return statics.start, statics.length


def emit_leaves(
    is_leaf: torch.Tensor, layout: TreeLayout, capacity: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact leaf ``(start, len)`` sorted by start into ``[capacity]`` buffers.

    Padding rows have ``start = n_tokens`` and ``len = 0``.
    """
    device = is_leaf.device
    start_all, len_all = _node_coords(layout, device)
    # Output slot of a leaf = number of leaves starting before it. Leaves are
    # disjoint, so marking each leaf's first atom and prefix-summing over atoms
    # gives that count without a sort.
    atom_of = start_all // layout.atom
    mark = torch.zeros(layout.n_atoms + 1, dtype=torch.int64, device=device)
    mark.scatter_(0, torch.where(is_leaf, atom_of, layout.n_atoms), 1)
    slot = torch.cumsum(mark, dim=0) - 1
    dst = torch.where(is_leaf, slot[atom_of], capacity).clamp_max(capacity)
    out_start = torch.full((capacity + 1,), layout.n_tokens, dtype=torch.int64, device=device)
    out_len = torch.zeros(capacity + 1, dtype=torch.int64, device=device)
    out_start.scatter_(0, dst, start_all)
    out_len.scatter_(0, dst, len_all)
    return out_start[:capacity], out_len[:capacity]


def leaf_totals(prefix: torch.Tensor, start: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """``[M, C]`` FP64 score sums from ``prefix[C, N+1]``; padding rows are zero."""
    end = (start + length).clamp_max(prefix.shape[1] - 1)
    begin = start.clamp_max(prefix.shape[1] - 1)
    return (prefix[:, end] - prefix[:, begin]).transpose(0, 1).contiguous()


# Edges admitted per target-merge round, as a multiple of the remaining deficit.
# The non-overlapping matching keeps roughly half of a run of admitted edges,
# so admitting 2x the deficit and capping the merges at the deficit converges
# in ~2 rounds instead of ~9 with no overshoot.
TARGET_ADMIT_FACTOR = 2


def merge_deficit(count: torch.Tensor, target: torch.Tensor, n_edges: int) -> torch.Tensor:
    """Pairs still to merge, ``clamp(count - target, 0, n_edges)`` (device int64 scalar)."""
    return (count.to(torch.int64).reshape(()) - target.to(torch.int64).reshape(())).clamp(0, n_edges)


def kth_cost_threshold(cost: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Device threshold admitting the ``k`` cheapest entries of ``cost`` (ties included).

    ``cost`` is fp64 with ``+inf`` on ineligible entries. Returns the successor
    of the k-th smallest value (``cost < thr`` then admits boundary ties too)
    or ``-inf`` for ``k == 0``. Uses gather, not tensor indexing, so there is
    no host sync and it captures into a CUDA graph.
    """
    k = k.to(torch.int64).reshape(())
    if _sel.available(cost):
        # exact radix selection: one scalar written, no sorted copy in HBM
        return _sel.kth_threshold_f64(cost, k).reshape(())
    kth = torch.sort(cost).values.gather(0, (k - 1).clamp(0, cost.shape[0] - 1).reshape(1)).reshape(())
    inf = torch.full((), float("inf"), dtype=torch.float64, device=cost.device)
    return torch.where(k > 0, torch.nextafter(kth, inf), -inf)


def target_merge_threshold(
    cost: torch.Tensor, count: torch.Tensor, target: torch.Tensor, admit: int = 1
) -> torch.Tensor:
    """Threshold admitting the ``admit * (count - target)`` cheapest eligible edges."""
    deficit = merge_deficit(count, target, cost.shape[0])
    return kth_cost_threshold(cost, (deficit * int(admit)).clamp_max(cost.shape[0]))


def merge_sync_nonoverlap(
    start: torch.Tensor,
    length: torch.Tensor,
    count: torch.Tensor,
    totals: torch.Tensor,
    threshold: torch.Tensor,
    *,
    rounds: int,
    max_merge_len: int,
    n_tokens: int,
    target: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """同步、相邻、互不重叠的 Ward Merge 总调度。

    每轮依次执行：

    1. 计算所有相邻叶子的 Ward cost；
    2. 按 cost 选择互不重叠的边；
    3. 合并被选中的叶子并 compact。

    target 模式下，每轮根据 ``count-target`` 动态决定阈值，最终从 L/8
    收敛到 L/D。``count`` 是 device 标量，全程不需要 host 同步。
    """
    capacity = start.shape[0]
    device = start.device
    threshold = threshold.to(torch.float64).reshape(())
    per_round = []
    if _pk.use_triton(start) and capacity >= 2:
        ws: dict = {}
        # Row means (totals / len) travel with the rows: computed once here and
        # then by the compact kernel, so the cost kernel never divides.
        means = _pk.row_means(totals, length)
        for _ in range(int(rounds)):
            if target is None:
                start, length, totals, means, count, n_sel = _pk.merge_round_triton(
                    start, length, totals, count, threshold,
                    max_merge_len=max_merge_len, n_tokens=n_tokens, workspace=ws, means=means,
                )
            else:
                deficit = merge_deficit(count, target, capacity - 1)
                cost = _pk.merge_costs_triton(
                    length, means, count, max_merge_len=max_merge_len, workspace=ws, active=deficit > 0
                )
                thr = kth_cost_threshold(cost, (deficit * TARGET_ADMIT_FACTOR).clamp_max(capacity - 1))
                start, length, totals, means, count, n_sel = _pk.merge_round_triton(
                    start, length, totals, count, thr,
                    max_merge_len=max_merge_len, n_tokens=n_tokens, workspace=ws,
                    means=means, ecost=cost, cap_merges=deficit,
                )
            per_round.append(n_sel)
        # The compact kernel only writes live rows (no per-round tail padding
        # traffic); normalise the tail once here so consumers see the usual
        # (start=n_tokens, len=0, totals=0) padding.
        live = torch.arange(capacity, device=device) < count
        start = torch.where(live, start, torch.full_like(start, n_tokens))
        length = torch.where(live, length, torch.zeros_like(length))
        totals = torch.where(live[:, None], totals, torch.zeros_like(totals))
        return start, length, count, totals, per_round
    rows = torch.arange(capacity, device=device)
    edge_pos = torch.arange(capacity - 1, device=device)
    for _ in range(int(rounds)):
        valid = rows < count
        n_l = length[:-1].to(torch.float64).clamp_min(1.0)
        n_r = length[1:].to(torch.float64).clamp_min(1.0)
        delta = totals[:-1] / n_l[:, None] - totals[1:] / n_r[:, None]
        cost = n_l * n_r / (n_l + n_r) * (delta * delta).sum(dim=1)
        eligible = valid[:-1] & valid[1:]
        if max_merge_len > 0:
            eligible &= length[:-1] + length[1:] <= max_merge_len
        thr = threshold
        if target is not None:
            deficit = merge_deficit(count, target, capacity - 1)
            thr = kth_cost_threshold(
                torch.where(eligible, cost, torch.full_like(cost, float("inf"))),
                (deficit * TARGET_ADMIT_FACTOR).clamp_max(capacity - 1),
            )
        eligible &= cost < thr
        # Rank eligible edges by (cost asc, start asc). Starts are ascending
        # already, so a stable sort on cost is the reference's sorted(eligible).
        ranked_cost = torch.where(eligible, cost, torch.full_like(cost, float("inf")))
        order = torch.sort(ranked_cost, stable=True).indices
        rank = torch.empty_like(order)
        rank[order] = edge_pos
        rank = torch.where(eligible, rank, torch.full_like(rank, _BIG))
        big = torch.full((1,), _BIG, dtype=rank.dtype, device=device)
        rank_left = torch.cat((big, rank[:-1]))
        rank_right = torch.cat((rank[1:], big))
        lower_left = rank_left < rank
        lower_right = rank_right < rank
        local_min = eligible & ~lower_left & ~lower_right
        local_max = eligible & lower_left & lower_right
        to_right = eligible & lower_right & ~lower_left
        to_left = eligible & lower_left & ~lower_right
        next_min = torch.flip(torch.cummin(torch.flip(torch.where(local_min, edge_pos, _BIG), (0,)), 0).values, (0,))
        prev_min = torch.cummax(torch.where(local_min, edge_pos, -_BIG), 0).values
        dist = torch.where(to_right, next_min - edge_pos, torch.where(to_left, edge_pos - prev_min, 0))
        sel_run = eligible & ~local_max & (dist % 2 == 0)
        false1 = torch.zeros(1, dtype=torch.bool, device=device)
        sel_left = torch.cat((false1, sel_run[:-1]))
        sel_right = torch.cat((sel_run[1:], false1))
        selected = sel_run | (local_max & ~sel_left & ~sel_right)
        if target is not None:
            # keep the `deficit` cheapest matched pairs (boundary ties included)
            thr2 = kth_cost_threshold(torch.where(selected, cost, torch.full_like(cost, float("inf"))), deficit)
            selected &= cost < thr2
        # Merge selected pairs simultaneously and compact.
        is_right = torch.cat((false1, selected))
        is_left = torch.cat((selected, false1))
        keep = valid & ~is_right
        zero_len = torch.zeros(1, dtype=length.dtype, device=device)
        new_len = length + torch.where(is_left, torch.cat((length[1:], zero_len)), 0)
        shifted = torch.cat((totals[1:], torch.zeros_like(totals[:1])))
        new_tot = totals + torch.where(is_left[:, None], shifted, 0.0)
        pos = torch.cumsum(keep, dim=0) - 1
        dst = torch.where(keep, pos, capacity)
        out_start = torch.full((capacity + 1,), n_tokens, dtype=start.dtype, device=device)
        out_len = torch.zeros(capacity + 1, dtype=length.dtype, device=device)
        out_tot = torch.zeros((capacity + 1, totals.shape[1]), dtype=totals.dtype, device=device)
        out_start.scatter_(0, dst, start)
        out_len.scatter_(0, dst, new_len)
        out_tot.scatter_(0, dst[:, None].expand(-1, totals.shape[1]), new_tot)
        start, length, totals = out_start[:capacity], out_len[:capacity], out_tot[:capacity]
        n_sel = selected.sum()
        count = count - n_sel
        per_round.append(n_sel)
    return start, length, count, totals, per_round


# --------------------------------------------------------------------------- #
# Whole build
# --------------------------------------------------------------------------- #


@dataclass
class GpuPartition:
    """Device-resident partition of one (layer, request) sealed prefix.

    ``leaf_start`` / ``leaf_len`` are ``int32[storage]`` (padding: start=n_complete,
    len=0); ``num_leaves`` is the valid row count after merging. With target
    merge, ``capacity`` is the post-merge storage (≈ L/D), not pre-merge M0.
    Scalars stay on the device; ``to_host()`` copies them for logging and tests.
    """

    n_tokens: int
    n_complete: int
    capacity: int  # post-merge storage (or M0 if no target merge)
    leaf_start: torch.Tensor
    leaf_len: torch.Tensor
    num_leaves: torch.Tensor  # int64 scalar
    lam: torch.Tensor  # float64 scalar
    dp_leaves: torch.Tensor
    repairs: torch.Tensor
    round_merges: torch.Tensor  # int64 [rounds]
    status_ok: torch.Tensor  # bool scalar: split produced exactly M0 leaves
    method: str
    meta: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)

    @property
    def tail_len(self) -> int:
        return self.n_tokens - self.n_complete

    def to_host(self) -> PrefillPartition:
        num = int(self.num_leaves.item()) if self.n_complete else 0
        meta = dict(self.meta)
        # A round that merges nothing leaves the leaves unchanged, so every later
        # round is also empty: trailing zeros are the reference's early ``break``.
        round_counts = self.round_merges.tolist()
        while round_counts and round_counts[-1] == 0:
            round_counts.pop()
        meta.update(
            {
                "M": self.capacity,
                "lambda": float(self.lam.item()) if self.n_complete else 0.0,
                "dp_leaves": int(self.dp_leaves.item()) if self.n_complete else 0,
                "repairs": int(self.repairs.item()) if self.n_complete else 0,
                "round_merge_counts": round_counts,
                "merge_rounds_run": len(round_counts),
                "merge_count": int(sum(round_counts)),
                "status_ok": bool(self.status_ok.item()) if self.n_complete else True,
                "M_after_merge": num,
            }
        )
        if meta.get("merge_count"):
            meta["premerge_M"] = self.capacity
            meta["arbitrary_leaves"] = True
        for key, (t0, t1) in self.events.items():
            if hasattr(t0, "elapsed_time"):
                t1.synchronize()
                meta[key] = t0.elapsed_time(t1) / 1000.0
            else:
                meta[key] = t1 - t0
        return PrefillPartition(
            self.leaf_start[:num].to("cpu"),
            self.leaf_len[:num].to("cpu"),
            self.n_complete,
            self.tail_len,
            self.n_tokens,
            method=self.method,
            meta=meta,
        )


def _mark(device: torch.device, enabled: bool):
    if not enabled:
        return None
    if device.type != "cuda":
        return time.perf_counter()
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def build_partition_gpu(
    scores: torch.Tensor,
    n_tokens: int,
    cfg: PartitionConfig,
    *,
    profile: bool = False,
) -> GpuPartition:
    """Split (+merge) the sealed prefix of ``scores[C, >=N_complete]`` on the device."""
    n_tokens = int(n_tokens)
    if scores.dim() != 2:
        raise ValueError(f"scores must be [C,N], got {tuple(scores.shape)}")
    done = n_complete_tokens(n_tokens, cfg.root)
    if done > scores.shape[1]:
        raise ValueError(f"sealed tokens={done} outside scores {scores.shape[1]}")
    device = scores.device
    budget = done // cfg.summary_compression
    method = cfg.method_name if cfg.merge_policy == "off" else f"{cfg.method_name}-{cfg.merge_policy}"
    if cfg.merge_target_divisor:
        method += f"-target{cfg.merge_target_divisor}"
    base_meta = {
        "metric": cfg.partition_metric,
        "atom": cfg.atom,
        "root": cfg.root,
        "backend": "gpu",
        "merge_policy": cfg.merge_policy,
        "merge_rounds": cfg.merge_rounds,
        "merge_alpha": cfg.merge_alpha,
        "max_merge_len": cfg.max_merge_len,
        "max_leaf_len": cfg.max_merge_len if cfg.merge_policy != "off" else cfg.root,
        "merge_target": cfg.merge_target(done),
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
            method, dict(base_meta),
        )
    events: dict = {}
    t0 = _mark(device, profile)
    flat, layout, prefix = build_score_tree(scores[:, :done], atom=cfg.atom, root=cfg.root)
    if not layout.n_roots <= budget <= layout.n_atoms:
        raise ValueError(f"M={budget} infeasible: roots={layout.n_roots}, atoms={layout.n_atoms}")
    t1 = _mark(device, profile)
    lam, _count0, rounds, rounds_used = search_lambda(
        flat, layout, budget, candidates=cfg.lambda_candidates, rel_tol=cfg.lambda_rel_tol,
        max_rounds=cfg.lambda_max_rounds, deficit_tol=cfg.lambda_deficit_tol,
    )
    is_leaf, leaf_gain = dp_leaf_and_gain(flat, layout, lam)
    dp_leaves = is_leaf.sum()
    t2 = _mark(device, profile)
    is_leaf, repairs, passes = repair_leaves(
        flat, layout, is_leaf, budget, tie_passes=cfg.repair_tie_passes, gain=leaf_gain
    )
    num_leaves = is_leaf.sum()
    status_ok = num_leaves == budget
    start, length = emit_leaves(is_leaf, layout, budget)
    t3 = _mark(device, profile)
    round_merges: list[torch.Tensor] = []
    count = num_leaves
    merge_target = cfg.merge_target(done)
    merge_rounds = cfg.merge_target_rounds if merge_target else cfg.merge_rounds
    if cfg.merge_policy == "sync_nonoverlap" and merge_rounds > 0:
        totals = leaf_totals(prefix, start, length)
        target = (
            torch.full((), merge_target, dtype=torch.int64, device=device) if merge_target else None
        )
        start, length, count, _totals, round_merges = merge_sync_nonoverlap(
            start, length, count, totals, lam * cfg.merge_alpha,
            rounds=merge_rounds, max_merge_len=cfg.max_merge_len, n_tokens=done, target=target,
        )
    elif cfg.merge_policy == "heap_reference":
        raise NotImplementedError(
            "heap_reference merge is a CPU reference; use split_backend=cpu_reference"
        )
    t4 = _mark(device, profile)
    if profile:
        events = {"tree_s": (t0, t1), "dp_s": (t1, t2), "repair_s": (t2, t3), "merge_s": (t3, t4), "total_s": (t0, t4)}
    meta = dict(base_meta)
    meta.update({
        "lambda_rounds": rounds,
        "lambda_rounds_used": rounds_used,  # device int64 (early stop)
        "lambda_candidates": cfg.lambda_candidates,
        "repair_passes": passes,
    })
    # After target merge, only ~L/D rows are live; shrink storage from M0=L/8 so
    # the summary pool and decode workspace stop carrying 7/8 padding.
    storage = budget
    if merge_target:
        storage = max(int(merge_target), 1)
        storage = -(-storage // 64) * 64
        start = start[:storage].contiguous()
        length = length[:storage].contiguous()
        meta["premerge_M"] = budget
        meta["storage_M"] = storage
    return GpuPartition(
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


# --------------------------------------------------------------------------- #
# CUDA-graph replay
# --------------------------------------------------------------------------- #


class GraphedBuilder:
    """``build_partition_gpu`` captured once and replayed per layer.

    Every layer of a request seals the same prefix, so the ~200 launches of one
    build are captured for ``(C, N_complete)`` and replayed 61 times; the host
    cost per layer becomes one strided input copy, one ``replay`` and the
    output clones. The build has no host reads, which is what makes it
    capturable. Per-stage events are not available in graph mode (only
    ``total_s`` around the replay).
    """

    def __init__(self, n_queries: int, n_tokens: int, cfg: PartitionConfig, device: torch.device):
        self.cfg = cfg
        self.n_queries = int(n_queries)
        self.done = n_complete_tokens(int(n_tokens), cfg.root)
        if self.done == 0:
            raise ValueError("nothing to capture: no sealed root")
        self.device = device
        self.scores = torch.zeros(self.n_queries, self.done, dtype=torch.float32, device=device)
        main = torch.cuda.current_stream(device)
        side = torch.cuda.Stream(device=device)
        side.wait_stream(main)
        self.graph = torch.cuda.CUDAGraph()
        # Raw capture_begin/end instead of ``torch.cuda.graph``: that context
        # manager starts with ``torch.cuda.synchronize(); gc.collect();
        # torch.cuda.empty_cache()``, which would drain the model's queue and
        # flush the allocator cache in the middle of a forward. Capture happens
        # on our own side stream; other threads' CUDA calls are not checked.
        try:
            with torch.cuda.stream(side):
                # Eager warm-up on the capture stream: lazy CUDA/Triton module loads
                # must not happen inside the capture.
                build_partition_gpu(self.scores, self.done, cfg)
                self.graph.capture_begin(capture_error_mode="thread_local")
                try:
                    self.out = build_partition_gpu(self.scores, self.done, cfg)
                finally:
                    self.graph.capture_end()
        except BaseException:
            # A failed capture (e.g. OOM inside the private pool) must not leak
            # the half-built graph or leave the main stream unordered after the
            # warm-up kernels still queued on ``side``.
            self.graph.reset()
            main.wait_stream(side)
            raise
        main.wait_stream(side)
        self.replays = 0

    def matches(self, n_queries: int, n_tokens: int, cfg: PartitionConfig) -> bool:
        return (
            self.n_queries == int(n_queries)
            and self.done == n_complete_tokens(int(n_tokens), cfg.root)
            and self.cfg == cfg
        )

    def build(self, scores: torch.Tensor, n_tokens: int, *, profile: bool = False) -> GpuPartition:
        if scores.shape[0] != self.n_queries or scores.shape[1] < self.done:
            raise ValueError(f"scores {tuple(scores.shape)} do not fit graph ({self.n_queries}, >={self.done})")
        t0 = _mark(self.device, profile)
        self.scores.copy_(scores[:, : self.done])
        self.graph.replay()
        self.replays += 1
        out = self.out
        part = GpuPartition(
            int(n_tokens),
            out.n_complete,
            out.capacity,
            out.leaf_start.clone(),
            out.leaf_len.clone(),
            out.num_leaves.clone(),
            out.lam.clone(),
            out.dp_leaves.clone(),
            out.repairs.clone(),
            out.round_merges.clone(),
            out.status_ok.clone(),
            out.method,
            dict(out.meta),
        )
        t1 = _mark(self.device, profile)
        if profile:
            part.events = {"total_s": (t0, t1)}
        part.meta["graph_replay"] = True
        return part


_GRAPH_CACHE: "OrderedDict[tuple, GraphedBuilder]" = OrderedDict()
_GRAPH_CACHE_MAX = 2
_GRAPH_DISABLED = False
# After an OOM during capture the next ``_GRAPH_OOM_COOLDOWN`` builds (one
# request's worth of layers) run eager before capture is attempted again.
_GRAPH_OOM_COOLDOWN = 61
_GRAPH_SKIP = 0
_GRAPH_STATS = {"captures": 0, "oom": 0, "oom_retry_ok": 0}


def _evict_graphs(keep: int) -> None:
    """Drop the oldest cached graphs until ``keep`` remain and release their
    private memory pools to the driver (the pool is freed once the graph is
    reset and the builder's output tensors are gone)."""
    while len(_GRAPH_CACHE) > keep:
        _, old = _GRAPH_CACHE.popitem(last=False)
        old.graph.reset()
        del old


def graphed_builder(
    n_queries: int, n_tokens: int, cfg: PartitionConfig, device: torch.device
) -> GraphedBuilder | None:
    """Cached ``GraphedBuilder`` for ``(C, N_complete, cfg)``; ``None`` when the
    graph path is off, unusable on this device, capture failed for a non-memory
    reason, or an OOM during capture put the graph path on cooldown.

    Graph capture allocates from a private memory pool, which cannot reuse the
    blocks cached in the default allocator pool; on a server whose static
    memory fraction leaves little free device memory a capture for a new
    ``N_complete`` can therefore OOM while the eager build (default pool)
    succeeds. So: the cache is trimmed *before* capturing (old private pools
    are returned to the driver first), an OOM is retried once after also
    releasing the default pool's cached blocks, and a second OOM only skips
    the graph path for the next ``_GRAPH_OOM_COOLDOWN`` builds.
    """
    global _GRAPH_DISABLED, _GRAPH_SKIP
    if _GRAPH_DISABLED or not cfg.graph_build or device.type != "cuda":
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    done = n_complete_tokens(int(n_tokens), cfg.root)
    if done == 0:
        return None  # nothing sealed: the eager path returns the empty partition
    key = (int(n_queries), done, cfg, str(device))
    hit = _GRAPH_CACHE.get(key)
    if hit is not None:
        _GRAPH_CACHE.move_to_end(key)
        return hit
    if _GRAPH_SKIP > 0:
        _GRAPH_SKIP -= 1
        return None
    log = logging.getLogger(__name__)
    _evict_graphs(_GRAPH_CACHE_MAX - 1)  # make room before, not after, the capture
    builder = None
    for attempt in range(2):
        try:
            builder = GraphedBuilder(n_queries, n_tokens, cfg, device)
            if attempt:
                _GRAPH_STATS["oom_retry_ok"] += 1
            break
        except torch.OutOfMemoryError as exc:
            _GRAPH_STATS["oom"] += 1
            if attempt == 0:
                # Free every cached graph pool and the default pool's idle
                # blocks so the new private pool can cudaMalloc, then retry.
                _evict_graphs(0)
                torch.cuda.empty_cache()
                continue
            _GRAPH_SKIP = _GRAPH_OOM_COOLDOWN
            log.warning(
                "adaptive-hisa CUDA graph capture OOM (C=%d, N=%d): eager builds for the "
                "next %d layers, then capture is retried (%s)",
                int(n_queries), done, _GRAPH_OOM_COOLDOWN, str(exc).split(".")[0],
            )
            return None
        except Exception:
            _GRAPH_DISABLED = True
            log.exception("adaptive-hisa CUDA graph capture failed; falling back to eager builds")
            return None
    _GRAPH_STATS["captures"] += 1
    _GRAPH_CACHE[key] = builder
    _evict_graphs(_GRAPH_CACHE_MAX)
    return builder


def reset_graph_cache() -> None:
    global _GRAPH_DISABLED, _GRAPH_SKIP
    _evict_graphs(0)
    _GRAPH_DISABLED = False
    _GRAPH_SKIP = 0
    _GRAPH_STATS.update({k: 0 for k in _GRAPH_STATS})
