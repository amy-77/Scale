"""GPU partition 的关键 Triton kernel。

Split 核心：

* ``_dp_count_bracket_kernel``：同时评估多个 λ 对应的叶子数；
* ``_bracket_update_kernel``：缩小 λ 搜索区间；
* ``_dp_leaf_mask_kernel``：按最终 λ 输出叶子 mask；
* ``_gain_kernel`` / ``_staircase_kernel``：把叶子数精确补到 L/8。

Merge 核心：

* ``_merge_cost_kernel``：计算相邻叶子的 Ward cost；
* ``_merge_match_kernel``：选择互不重叠的相邻 merge 边；
* ``_merge_compact_kernel``：真正合并 length/totals 并压紧输出。

每个 256-token root 的子树相互独立，因此 λ-DP 可由一个 Triton program
处理一棵 root，整次 bottom-up / top-down 尽量留在寄存器中。
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit(do_not_specialize=["n_roots"])
    def _dp_count_kernel(
        flat_ptr,
        lam_ptr,
        out_ptr,  # int32 [K, n_roots]
        n_roots,
        R: tl.constexpr,
        LEVELS: tl.constexpr,
        KC: tl.constexpr,  # λ candidates per program (tree loaded once)
    ):
        r = tl.program_id(0)
        kb = tl.program_id(1)
        ks = kb * KC + tl.arange(0, KC)
        lam = tl.load(lam_ptr + ks)
        i = tl.arange(0, R)
        cost = tl.load(flat_ptr + r * R + i)[None, :] + lam[:, None]
        cnt = tl.full((KC, R), 1, tl.int32)
        off = n_roots * R
        for l in tl.static_range(1, LEVELS):
            split = tl.sum(tl.reshape(cost, (KC, (R >> l), 2)), axis=2)
            cnt2 = tl.sum(tl.reshape(cnt, (KC, (R >> l), 2)), axis=2)
            j = tl.arange(0, (R >> l))
            keep = tl.load(flat_ptr + off + r * (R >> l) + j)[None, :] + lam[:, None]
            ds = split < keep
            cost = tl.where(ds, split, keep)
            cnt = tl.where(ds, cnt2, 1)
            off += n_roots * (R >> l)
        tl.store(out_ptr + ks * n_roots + r, tl.sum(cnt, axis=1))

    @triton.jit(do_not_specialize=["n_roots"])
    def _dp_count_bracket_kernel(
        flat_ptr,
        bracket_ptr,  # fp64 [2]: lo, hi
        out_ptr,  # int32 [K, n_roots]
        n_roots,
        K: tl.constexpr,
        R: tl.constexpr,
        LEVELS: tl.constexpr,
        KC: tl.constexpr,
    ):
        """Split 算子 1：一次评估 K 个候选 λ 各自产生多少叶子。"""
        r = tl.program_id(0)
        kb = tl.program_id(1)
        ks = kb * KC + tl.arange(0, KC)
        lo = tl.load(bracket_ptr)
        hi = tl.load(bracket_ptr + 1)
        frac = (ks + 1).to(tl.float64) / (K + 1)
        lam = lo + (hi - lo) * frac
        i = tl.arange(0, R)
        cost = tl.load(flat_ptr + r * R + i)[None, :] + lam[:, None]
        cnt = tl.full((KC, R), 1, tl.int32)
        off = n_roots * R
        for l in tl.static_range(1, LEVELS):
            split = tl.sum(tl.reshape(cost, (KC, (R >> l), 2)), axis=2)
            cnt2 = tl.sum(tl.reshape(cnt, (KC, (R >> l), 2)), axis=2)
            j = tl.arange(0, (R >> l))
            keep = tl.load(flat_ptr + off + r * (R >> l) + j)[None, :] + lam[:, None]
            ds = split < keep
            cost = tl.where(ds, split, keep)
            cnt = tl.where(ds, cnt2, 1)
            off += n_roots * (R >> l)
        tl.store(out_ptr + ks * n_roots + r, tl.sum(cnt, axis=1))

    @triton.jit(do_not_specialize=["n_roots", "m_leaves"])
    def _bracket_update_kernel(
        counts_ptr,  # int32 [K, n_roots]
        bracket_ptr,  # fp64 [2] in/out
        n_roots,
        m_leaves,
        K: tl.constexpr,
        RB: tl.constexpr,
    ):
        """Split 算子 2：把 λ 区间缩到叶子数跨过目标 M 的网格小区间。"""
        ks = tl.arange(0, K)
        total = tl.zeros((K,), tl.int64)
        for r0 in range(0, n_roots, RB):
            rs = r0 + tl.arange(0, RB)
            blk = tl.load(counts_ptr + ks[:, None] * n_roots + rs[None, :], mask=rs[None, :] < n_roots, other=0)
            total += tl.sum(blk.to(tl.int64), axis=1)
        lo = tl.load(bracket_ptr)
        hi = tl.load(bracket_ptr + 1)
        frac = (ks + 1).to(tl.float64) / (K + 1)
        cands = lo + (hi - lo) * frac
        n_feasible = tl.sum((total <= m_leaves).to(tl.int64))
        idx = K - n_feasible  # grid index of the last infeasible point (0 = lo)
        c_lo = tl.sum(tl.where(ks == idx - 1, cands, 0.0))
        c_hi = tl.sum(tl.where(ks == idx, cands, 0.0))
        new_lo = tl.where(idx == 0, lo, c_lo)
        new_hi = tl.where(idx == K, hi, c_hi)
        tl.store(bracket_ptr, new_lo)
        tl.store(bracket_ptr + 1, new_hi)

    @triton.jit(do_not_specialize=["n_roots"])
    def _dp_leaf_mask_kernel(
        flat_ptr,
        lam_ptr,
        leaf_ptr,  # uint8 [n_nodes] out
        split_ptr,  # uint8 [n_nodes] scratch: do_split, then active&split
        n_roots,
        R: tl.constexpr,
        LEVELS: tl.constexpr,
    ):
        # Split 算子 3：bottom-up 计算 split/keep，再 top-down 写最终叶子。
        r = tl.program_id(0)
        lam = tl.load(lam_ptr)
        i = tl.arange(0, R)
        cost = tl.load(flat_ptr + r * R + i) + lam
        off = n_roots * R
        # bottom-up: do_split per internal node
        for l in tl.static_range(1, LEVELS):
            split = tl.sum(tl.reshape(cost, ((R >> l), 2)), axis=1)
            j = tl.arange(0, (R >> l))
            keep = tl.load(flat_ptr + off + r * (R >> l) + j) + lam
            ds = split < keep
            cost = tl.where(ds, split, keep)
            tl.store(split_ptr + off + r * (R >> l) + j, ds.to(tl.uint8))
            off += n_roots * (R >> l)
        tl.debug_barrier()
        # top-down: active(root) = 1; child active = parent active & parent split
        for l2 in tl.static_range(1, LEVELS):
            # level l = LEVELS - l2 runs from LEVELS-1 (roots) down to 1
            base = n_roots * (2 * R - ((2 * R) >> (LEVELS - l2)))
            j = tl.arange(0, (R >> (LEVELS - l2)))
            ds = tl.load(split_ptr + base + r * (R >> (LEVELS - l2)) + j) != 0
            if l2 == 1:
                active = j >= 0
            else:
                pbase = n_roots * (2 * R - ((2 * R) >> (LEVELS - l2 + 1)))
                active = tl.load(split_ptr + pbase + r * (R >> (LEVELS - l2 + 1)) + (j // 2)) != 0
            tl.store(leaf_ptr + base + r * (R >> (LEVELS - l2)) + j, (active & (ds == 0)).to(tl.uint8))
            tl.store(split_ptr + base + r * (R >> (LEVELS - l2)) + j, (active & ds).to(tl.uint8))
            tl.debug_barrier()
        # atoms: leaf iff parent active & split
        pbase = n_roots * (2 * R - ((2 * R) >> 1))
        act0 = tl.load(split_ptr + pbase + r * (R >> 1) + (i // 2)) != 0
        tl.store(leaf_ptr + r * R + i, act0.to(tl.uint8))


    @triton.jit(do_not_specialize=["stride_c", "n_roots", "C"])
    def _score_tree_kernel(
        scores_ptr,  # [C, N] any float, row stride = stride_c
        stride_c,
        flat_ptr,  # fp64 [n_nodes] out
        n_roots,
        C,
        ATOM: tl.constexpr,
        R: tl.constexpr,
        LEVELS: tl.constexpr,
    ):
        """旧 ``[128,N]`` 路径的 Key-SSE 建树 kernel（raw builder 当前不走它）。"""
        r = tl.program_id(0)
        i = tl.arange(0, R)
        a = tl.arange(0, ATOM)
        tok = (r * R + i)[:, None] * ATOM + a[None, :]
        for c in range(0, C):
            v = tl.load(scores_ptr + c * stride_c + tok).to(tl.float64)
            m = tl.sum(v, axis=1)
            m2 = tl.sum(v * v, axis=1)
            cnt = ATOM * 1.0
            e = m2 - m * m / cnt
            tl.store(flat_ptr + r * R + i, tl.load(flat_ptr + r * R + i) + e)
            off = n_roots * R
            for l in tl.static_range(1, LEVELS):
                m = tl.sum(tl.reshape(m, ((R >> l), 2)), axis=1)
                m2 = tl.sum(tl.reshape(m2, ((R >> l), 2)), axis=1)
                cnt = cnt * 2.0
                e = m2 - m * m / cnt
                j = tl.arange(0, (R >> l))
                tl.store(flat_ptr + off + r * (R >> l) + j, tl.load(flat_ptr + off + r * (R >> l) + j) + e)
                off += n_roots * (R >> l)

    @triton.jit(do_not_specialize=["n_roots"])
    def _gain_kernel(
        flat_ptr,
        gain_ptr,  # fp64 [n_nodes] out (-inf at atoms)
        n_roots,
        R: tl.constexpr,
        LEVELS: tl.constexpr,
    ):
        # repair 算子：gain = parent_sse - left_sse - right_sse。
        r = tl.program_id(0)
        i = tl.arange(0, R)
        child = tl.load(flat_ptr + r * R + i)
        tl.store(gain_ptr + r * R + i, tl.full((R,), float("-inf"), tl.float64))
        off = n_roots * R
        for l in tl.static_range(1, LEVELS):
            pair = tl.sum(tl.reshape(child, ((R >> l), 2)), axis=1)
            j = tl.arange(0, (R >> l))
            own = tl.load(flat_ptr + off + r * (R >> l) + j)
            tl.store(gain_ptr + off + r * (R >> l) + j, own - pair)
            child = own
            off += n_roots * (R >> l)

    @triton.jit(do_not_specialize=["n_roots", "bits"])
    def _staircase_kernel(
        rank_ptr,  # int64 [n_nodes] heap priority (0 pops first)
        frontier_ptr,  # uint8 [n_nodes]
        stairs_ptr,  # int64 [n_nodes, LPAD] scratch
        keys_ptr,  # int64 [n_nodes, NKEYS] out (lexicographic pop order)
        reach_ptr,  # uint8 [n_nodes] out
        n_roots,
        bits,
        R: tl.constexpr,
        LEVELS: tl.constexpr,
        LPAD: tl.constexpr,
        PER: tl.constexpr,
        NKEYS: tl.constexpr,
    ):
        r = tl.program_id(0)
        cols = tl.arange(0, LPAD)
        for l2 in tl.static_range(0, LEVELS):
            # level l = LEVELS-1-l2, top-down
            base = n_roots * (2 * R - ((2 * R) >> (LEVELS - 1 - l2)))
            j = tl.arange(0, (R >> (LEVELS - 1 - l2)))
            node = base + r * (R >> (LEVELS - 1 - l2)) + j
            own_f = tl.load(frontier_ptr + node) != 0
            own_r = tl.load(rank_ptr + node)
            if l2 == 0:
                p_s = tl.full(((R >> (LEVELS - 1 - l2)), LPAD), -1, tl.int64)
                p_reach = j < 0
            else:
                pbase = n_roots * (2 * R - ((2 * R) >> (LEVELS - l2)))
                pnode = pbase + r * (R >> (LEVELS - l2)) + (j // 2)
                p_s = tl.load(stairs_ptr + pnode[:, None] * LPAD + cols[None, :])
                p_reach = tl.load(reach_ptr + pnode) != 0
            keep = p_s > own_r[:, None]
            n_keep = tl.sum(keep.to(tl.int64), axis=1)
            ext = tl.where(keep, p_s, -1)
            ext = tl.where(cols[None, :] == n_keep[:, None], own_r[:, None], ext)
            via = p_reach & (own_f == 0)
            own_s = tl.where(cols[None, :] == 0, own_r[:, None], -1)
            s = tl.where(own_f[:, None], own_s, tl.where(via[:, None], ext, -1))
            reach_v = own_f | via
            tl.store(stairs_ptr + node[:, None] * LPAD + cols[None, :], s)
            tl.store(reach_ptr + node, reach_v.to(tl.uint8))
            # pack PER ranks per int64 key, most significant first; -1 -> 0
            shifted = (s + 1).to(tl.int64)
            for kk in tl.static_range(0, NKEYS):
                in_chunk = (cols >= kk * PER) & (cols < (kk + 1) * PER)
                sh = tl.maximum(((kk + 1) * PER - 1 - cols) * bits, 0)
                term = tl.where(in_chunk[None, :], shifted << sh[None, :], 0)
                key = tl.sum(term, axis=1)
                tl.store(keys_ptr + node * NKEYS + kk, key)
            tl.debug_barrier()


    # ----------------------------------------------------------------- Merge 核心算子

    @triton.jit
    def _min_i64(a, b):
        return tl.minimum(a, b)

    @triton.jit
    def _max_i64(a, b):
        return tl.maximum(a, b)

    @triton.jit(do_not_specialize=["n_edges", "max_merge_len"])
    def _merge_cost_kernel(
        len_ptr,  # int64 [cap]
        tot_ptr,  # fp64 [cap, C]
        count_ptr,  # int64 scalar: valid rows
        thr_ptr,  # fp64 scalar
        ecost_ptr,  # fp64 [cap-1] out: Ward cost of eligible edges, +inf otherwise
        n_edges,
        max_merge_len,
        C: tl.constexpr,
        CPAD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # 对每条相邻边计算 Ward cost；越小越适合合并。
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < n_edges
        count = tl.load(count_ptr)
        thr = tl.load(thr_ptr)
        cols = tl.arange(0, CPAD)
        cm = cols < C
        # thr == -inf marks an idle round (target already reached): nothing can
        # be eligible, so skip the fp64 traffic and just write +inf.
        live = m & (thr > float("-inf"))
        len_l = tl.load(len_ptr + i, mask=live, other=1)
        len_r = tl.load(len_ptr + i + 1, mask=live, other=1)
        n_l = tl.maximum(len_l.to(tl.float64), 1.0)
        n_r = tl.maximum(len_r.to(tl.float64), 1.0)
        t_l = tl.load(tot_ptr + i[:, None] * C + cols[None, :], mask=live[:, None] & cm[None, :], other=0.0)
        t_r = tl.load(tot_ptr + (i + 1)[:, None] * C + cols[None, :], mask=live[:, None] & cm[None, :], other=0.0)
        delta = t_l / n_l[:, None] - t_r / n_r[:, None]
        cost = n_l * n_r / (n_l + n_r) * tl.sum(delta * delta, axis=1)
        eligible = m & (i + 1 < count) & (cost < thr)
        if max_merge_len > 0:
            eligible = eligible & (len_l + len_r <= max_merge_len)
        tl.store(ecost_ptr + i, tl.where(eligible, cost, float("inf")), mask=m)

    @triton.jit(do_not_specialize=["n_edges"])
    def _merge_match_kernel(
        ecost_ptr,  # fp64 [cap-1]
        count_ptr,  # int64 scalar in: valid rows
        sel_ptr,  # uint8 [cap-1] out: selected edges (scratch: sel_run first)
        src_ptr,  # int64 [cap] out: src[dst] = row for kept rows
        new_count_ptr,  # int64 scalar out
        nsel_ptr,  # int64 scalar out
        n_edges,
        BLOCK: tl.constexpr,
    ):
        """按 ``(cost,start)`` 贪心选择互不重叠的相邻边。"""
        BIG: tl.constexpr = 1 << 40
        i = tl.arange(0, BLOCK)
        m = i < n_edges
        inf = float("inf")
        c = tl.load(ecost_ptr + i, mask=m, other=inf)
        c_l = tl.load(ecost_ptr + i - 1, mask=m & (i > 0), other=inf)
        c_r = tl.load(ecost_ptr + i + 1, mask=m & (i + 1 < n_edges), other=inf)
        elig = c < inf
        # ranks are (cost asc, start asc); starts ascend along the path
        lower_left = (c_l < c) | (c_l == c)
        lower_right = c_r < c
        local_min = elig & (lower_left == 0) & (lower_right == 0)
        local_max = elig & lower_left & lower_right
        to_right = elig & lower_right & (lower_left == 0)
        to_left = elig & lower_left & (lower_right == 0)
        i64 = i.to(tl.int64)
        nm = tl.associative_scan(tl.where(local_min, i64, BIG), 0, _min_i64, reverse=True)
        pm = tl.associative_scan(tl.where(local_min, i64, -BIG), 0, _max_i64)
        dist = tl.where(to_right, nm - i64, tl.where(to_left, i64 - pm, 0))
        sel_run = elig & (local_max == 0) & (dist % 2 == 0)
        tl.store(sel_ptr + i, sel_run.to(tl.uint8), mask=m)
        tl.debug_barrier()
        sl = tl.load(sel_ptr + i - 1, mask=m & (i > 0), other=0) != 0
        sr = tl.load(sel_ptr + i + 1, mask=m & (i + 1 < n_edges), other=0) != 0
        selected = sel_run | (local_max & (sl == 0) & (sr == 0))
        tl.debug_barrier()
        tl.store(sel_ptr + i, selected.to(tl.uint8), mask=m)
        tl.debug_barrier()
        # rows j in [0, cap): right members of selected pairs disappear
        count = tl.load(count_ptr)
        j = i  # BLOCK >= cap
        jm = j < n_edges + 1
        is_right = tl.load(sel_ptr + j - 1, mask=jm & (j > 0), other=0) != 0
        keep = jm & (j < count) & (is_right == 0)
        pos = tl.cumsum(keep.to(tl.int64), 0) - 1
        tl.store(src_ptr + pos, j.to(tl.int64), mask=keep)
        n_sel = tl.sum(selected.to(tl.int64))
        tl.store(nsel_ptr, n_sel)
        tl.store(new_count_ptr, count - n_sel)

    @triton.jit(do_not_specialize=["n_edges"])
    def _merge_index_kernel(
        sel_ptr,  # uint8 [cap-1] selected edges
        count_ptr,  # int64 scalar in: valid rows
        src_ptr,  # int64 [cap] out: src[dst] = row for kept rows
        new_count_ptr,  # int64 scalar out
        nsel_ptr,  # int64 scalar out
        n_edges,
        BLOCK: tl.constexpr,
    ):
        """Row compaction index for a given selection (the tail of the match kernel)."""
        j = tl.arange(0, BLOCK)
        jm = j < n_edges + 1
        count = tl.load(count_ptr)
        selected = tl.load(sel_ptr + j, mask=j < n_edges, other=0) != 0
        is_right = tl.load(sel_ptr + j - 1, mask=jm & (j > 0), other=0) != 0
        keep = jm & (j < count) & (is_right == 0)
        pos = tl.cumsum(keep.to(tl.int64), 0) - 1
        tl.store(src_ptr + pos, j.to(tl.int64), mask=keep)
        n_sel = tl.sum(selected.to(tl.int64))
        tl.store(nsel_ptr, n_sel)
        tl.store(new_count_ptr, count - n_sel)

    @triton.jit(do_not_specialize=["cap", "n_tokens"])
    def _merge_compact_kernel(
        start_ptr,
        len_ptr,
        tot_ptr,
        sel_ptr,
        src_ptr,
        new_count_ptr,
        out_start_ptr,
        out_len_ptr,
        out_tot_ptr,
        cap,
        n_tokens,
        C: tl.constexpr,
        CPAD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # 选中边的右叶子消失；左叶子累加右侧 length 和 sum(K)，再压紧。
        pid = tl.program_id(0)
        j = pid * BLOCK + tl.arange(0, BLOCK)
        m = j < cap
        n_keep = tl.load(new_count_ptr)
        live = m & (j < n_keep)
        src = tl.load(src_ptr + j, mask=live, other=0)
        merged = tl.load(sel_ptr + src, mask=live & (src + 1 < cap), other=0) != 0
        s = tl.load(start_ptr + src, mask=live, other=n_tokens)
        ln = tl.load(len_ptr + src, mask=live, other=0)
        ln_r = tl.load(len_ptr + src + 1, mask=live & merged, other=0)
        tl.store(out_start_ptr + j, tl.where(live, s, n_tokens), mask=m)
        tl.store(out_len_ptr + j, tl.where(live, ln + ln_r, 0), mask=m)
        cols = tl.arange(0, CPAD)
        cm = cols < C
        t = tl.load(tot_ptr + src[:, None] * C + cols[None, :], mask=live[:, None] & cm[None, :], other=0.0)
        t_r = tl.load(
            tot_ptr + (src + 1)[:, None] * C + cols[None, :],
            mask=(live & merged)[:, None] & cm[None, :],
            other=0.0,
        )
        tl.store(out_tot_ptr + j[:, None] * C + cols[None, :], t + t_r, mask=m[:, None] & cm[None, :])


MATCH_BLOCK = 16384
# 32-row tiles halve the fp64 cost kernel's time vs 64 at C=128 (P-key).
COST_BLOCK = 32


def _merge_workspace(ws: dict, cap: int, c: int, device) -> None:
    if ws.get("cap") != cap or ws.get("c") != c:
        ws.clear()
        ws["cap"] = cap
        ws["c"] = c
        ws["ecost"] = torch.empty(cap - 1, dtype=torch.float64, device=device)
        ws["ecost2"] = torch.empty(cap - 1, dtype=torch.float64, device=device)
        ws["sorted_cost"] = torch.empty(cap - 1, dtype=torch.float64, device=device)
        ws["sorted_idx"] = torch.empty(cap - 1, dtype=torch.int64, device=device)
        ws["sel"] = torch.empty(cap - 1, dtype=torch.uint8, device=device)
        ws["src"] = torch.empty(cap, dtype=torch.int64, device=device)
        ws["scalars"] = torch.zeros(2, dtype=torch.int64, device=device)
        ws["inf"] = torch.full((1,), float("inf"), dtype=torch.float64, device=device)
        ws["buf_start"] = [None, None]
        ws["buf_len"] = [None, None]
        ws["buf_tot"] = [None, None]
        ws["ping"] = 0


def merge_costs_triton(
    length: torch.Tensor,
    totals: torch.Tensor,
    count: torch.Tensor,
    *,
    max_merge_len: int,
    workspace: dict,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ward cost of every eligible adjacent edge (``+inf`` otherwise), fp64 ``[cap-1]``.

    ``active`` (device bool scalar) false marks an idle round: the kernel then
    skips the fp64 traffic and returns all ``+inf``. Returns a fresh tensor
    (the shared workspace buffer is reused by ``merge_round_triton``).
    """
    cap = length.shape[0]
    n_edges = cap - 1
    c = totals.shape[1]
    _merge_workspace(workspace, cap, c, length.device)
    cpad = 1 << (c - 1).bit_length()
    count = count.to(torch.int64).reshape(1).contiguous()
    thr = workspace["inf"] if active is None else torch.where(active.reshape(1), workspace["inf"], -workspace["inf"])
    _merge_cost_kernel[(triton.cdiv(n_edges, COST_BLOCK),)](
        length, totals, count, thr, workspace["ecost"], n_edges, int(max_merge_len),
        C=c, CPAD=cpad, BLOCK=COST_BLOCK,
    )
    # Twin buffer instead of .clone() alloc; safe for the caller to sort/mutate.
    workspace["ecost2"].copy_(workspace["ecost"])
    return workspace["ecost2"]


def merge_round_triton(
    start: torch.Tensor,
    length: torch.Tensor,
    totals: torch.Tensor,
    count: torch.Tensor,
    threshold: torch.Tensor,
    *,
    max_merge_len: int,
    n_tokens: int,
    workspace: dict,
    ecost: torch.Tensor | None = None,
    cap_merges: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One synchronous non-overlapping merge round in three launches.

    Returns ``(start, length, totals, count, n_selected)`` (new buffers).
    Same semantics as the torch loop body in ``partition_gpu.merge_sync_nonoverlap``.

    ``ecost`` (fp64 ``[cap-1]``, ``+inf`` on ineligible edges) skips the cost
    kernel; ``threshold`` is then applied elementwise. ``cap_merges`` (device
    int64 scalar) keeps only that many cheapest selected edges (ties at the
    boundary included) - the target-count merge's per-round budget.
    """
    cap = start.shape[0]
    n_edges = cap - 1
    c = totals.shape[1]
    device = start.device
    if cap < 2:
        zero = torch.zeros((), dtype=torch.int64, device=device)
        return start, length, totals, count, zero
    cpad = 1 << (c - 1).bit_length()
    # One compiled variant covers every capacity up to 16K rows (128K tokens
    # at compression 8); larger caps compile their own block on demand.
    block_rows = max(MATCH_BLOCK, 1 << max(1, (cap - 1).bit_length()))
    ws = workspace
    _merge_workspace(ws, cap, c, device)
    sel, src, scalars = ws["sel"], ws["src"], ws["scalars"]
    new_count, n_sel = scalars[0:1], scalars[1:2]
    count = count.to(torch.int64).reshape(1).contiguous()
    threshold = threshold.to(torch.float64).reshape(1).contiguous()
    blk = 64
    if ecost is None:
        ecost = ws["ecost"]
        _merge_cost_kernel[(triton.cdiv(n_edges, COST_BLOCK),)](
            length, totals, count, threshold, ecost, n_edges, int(max_merge_len), C=c, CPAD=cpad, BLOCK=COST_BLOCK
        )
    else:
        ecost = torch.where(ecost < threshold, ecost, ws["inf"])
    _merge_match_kernel[(1,)](ecost, count, sel, src, new_count, n_sel, n_edges, BLOCK=block_rows, num_warps=32)
    if cap_merges is not None:
        picked = sel != 0
        sel_cost = torch.where(picked, ecost, ws["inf"])
        torch.sort(sel_cost, out=(ws["sorted_cost"], ws["sorted_idx"]))
        kth = ws["sorted_cost"].gather(
            0, (cap_merges.to(torch.int64).reshape(1) - 1).clamp(0, n_edges - 1)
        )
        thr2 = torch.where(cap_merges.reshape(1) > 0, torch.nextafter(kth, ws["inf"]), -ws["inf"])
        sel = (picked & (ecost < thr2)).to(torch.uint8)
        _merge_index_kernel[(1,)](sel, count, src, new_count, n_sel, n_edges, BLOCK=block_rows, num_warps=32)
    ping = int(ws["ping"])
    if ws["buf_start"][ping] is None or ws["buf_start"][ping].dtype != start.dtype:
        ws["buf_start"][ping] = torch.empty(cap, dtype=start.dtype, device=device)
        ws["buf_len"][ping] = torch.empty(cap, dtype=length.dtype, device=device)
        ws["buf_tot"][ping] = torch.empty((cap, c), dtype=totals.dtype, device=device)
    out_start = ws["buf_start"][ping]
    out_len = ws["buf_len"][ping]
    out_tot = ws["buf_tot"][ping]
    ws["ping"] = 1 - ping
    _merge_compact_kernel[(triton.cdiv(cap, blk),)](
        start, length, totals, sel, src, new_count, out_start, out_len, out_tot, cap, int(n_tokens), C=c, CPAD=cpad, BLOCK=blk
    )
    return out_start, out_len, out_tot, new_count.clone().reshape(()), n_sel.clone().reshape(())


def _check(layout) -> tuple[int, int]:
    r_atoms = layout.root // layout.atom
    if r_atoms & (r_atoms - 1) or r_atoms < 2:
        raise ValueError("root/atom must be a power of two >= 2")
    return r_atoms, layout.levels


def dp_counts_triton(flat: torch.Tensor, layout, lams: torch.Tensor) -> torch.Tensor:
    """``[K]`` int64 leaf counts for every λ (same result as ``partition_gpu.dp_counts``)."""
    r_atoms, levels = _check(layout)
    lams = lams.to(torch.float64).reshape(-1).contiguous()
    k = lams.numel()
    kc = 16 if k % 16 == 0 else (8 if k % 8 == 0 else (4 if k % 4 == 0 else (2 if k % 2 == 0 else 1)))
    out = torch.empty((k, layout.n_roots), dtype=torch.int32, device=flat.device)
    _dp_count_kernel[(layout.n_roots, k // kc)](flat, lams, out, layout.n_roots, R=r_atoms, LEVELS=levels, KC=kc)
    return out.sum(dim=1, dtype=torch.int64)


def search_rounds_triton(
    flat: torch.Tensor, layout, bracket: torch.Tensor, m_leaves: int, *, candidates: int, rounds: int
) -> None:
    """Run ``rounds`` K-way bracketing rounds in place on ``bracket = [lo, hi]``.

    Two launches per round; identical arithmetic to ``partition_gpu.search_lambda``.
    """
    r_atoms, levels = _check(layout)
    k = int(candidates)
    if k & (k - 1) or k < 1:
        raise ValueError("lambda_candidates must be a power of two for the fused search")
    kc = min(16, k)
    counts = torch.empty((k, layout.n_roots), dtype=torch.int32, device=flat.device)
    for _ in range(int(rounds)):
        _dp_count_bracket_kernel[(layout.n_roots, k // kc)](
            flat, bracket, counts, layout.n_roots, K=k, R=r_atoms, LEVELS=levels, KC=kc
        )
        _bracket_update_kernel[(1,)](counts, bracket, layout.n_roots, int(m_leaves), K=k, RB=128)


def dp_leaf_mask_triton(flat: torch.Tensor, layout, lam: torch.Tensor) -> torch.Tensor:
    """Flat bool leaf mask at one λ (same result as ``partition_gpu.dp_leaf_mask``)."""
    r_atoms, levels = _check(layout)
    lam = lam.to(torch.float64).reshape(1).contiguous()
    leaf = torch.empty(layout.n_nodes, dtype=torch.uint8, device=flat.device)
    scratch = torch.empty(layout.n_nodes, dtype=torch.uint8, device=flat.device)
    _dp_leaf_mask_kernel[(layout.n_roots,)](flat, lam, leaf, scratch, layout.n_roots, R=r_atoms, LEVELS=levels)
    return leaf.to(torch.bool)


def score_tree_triton(scores: torch.Tensor, layout) -> torch.Tensor:
    """FP64 level-major energy tree from ``scores[C, N]`` (one launch)."""
    r_atoms, levels = _check(layout)
    if scores.stride(1) != 1:
        scores = scores.contiguous()
    flat = torch.zeros(layout.n_nodes, dtype=torch.float64, device=scores.device)
    _score_tree_kernel[(layout.n_roots,)](
        scores,
        scores.stride(0),
        flat,
        layout.n_roots,
        scores.shape[0],
        ATOM=layout.atom,
        R=r_atoms,
        LEVELS=levels,
    )
    return flat


def gain_triton(flat: torch.Tensor, layout) -> torch.Tensor:
    """Split gain ``E(node) - E(left) - E(right)`` per node, ``-inf`` at atoms."""
    r_atoms, levels = _check(layout)
    gain = torch.empty(layout.n_nodes, dtype=torch.float64, device=flat.device)
    _gain_kernel[(layout.n_roots,)](flat, gain, layout.n_roots, R=r_atoms, LEVELS=levels)
    return gain


STAIR_PER = 3  # ranks per int64 key (needs bits * 3 <= 63)


def staircase_keys_triton(
    rank: torch.Tensor, frontier: torch.Tensor, layout, bits: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packed lexicographic pop-order keys ``[n_nodes, NKEYS]`` and ``reach``.

    Same definition as ``partition_gpu._staircase`` + ``_pack_ranks``.
    """
    r_atoms, levels = _check(layout)
    if bits * STAIR_PER > 63:
        raise ValueError("tree too large for packed staircase keys")
    lpad = 1 << (levels - 1).bit_length()
    nkeys = -(-levels // STAIR_PER)
    n = layout.n_nodes
    stairs = torch.empty((n, lpad), dtype=torch.int64, device=rank.device)
    keys = torch.empty((n, nkeys), dtype=torch.int64, device=rank.device)
    reach = torch.empty(n, dtype=torch.uint8, device=rank.device)
    _staircase_kernel[(layout.n_roots,)](
        rank,
        frontier.to(torch.uint8),
        stairs,
        keys,
        reach,
        layout.n_roots,
        bits,
        R=r_atoms,
        LEVELS=levels,
        LPAD=lpad,
        PER=STAIR_PER,
        NKEYS=nkeys,
    )
    return keys, reach.to(torch.bool)


def use_triton(tensor: torch.Tensor) -> bool:
    return HAS_TRITON and tensor.is_cuda


def warmup_triton_kernels(device, cfg) -> None:
    """Compile every kernel variant the build can use (call once per process).

    Runtime ints are ``do_not_specialize``d and block sizes are fixed, so one
    representative build compiles everything except the >16K-row merge block.
    """
    if not HAS_TRITON:
        return
    from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import build_partition_gpu
    from sglang.srt.layers.attention.nsa.adaptive_hisa.summary_kernels import (
        leaf_key_means,
        requant_summaries,
    )

    n = cfg.root * 17 + 8  # 17 roots, ragged tail
    scores = torch.rand(cfg.energy_rows, n, device=device)
    part = build_partition_gpu(scores, n, cfg)
    build_partition_gpu(scores[:, : cfg.root], cfg.root, cfg)  # single-root variant
    if cfg.build_summaries:
        keys = torch.zeros(n, 128, dtype=torch.float8_e4m3fn, device=device)
        scale = torch.ones(n, dtype=torch.float32, device=device)
        means = leaf_key_means(keys, scale, part.leaf_start, part.leaf_len)
        for fmt in (None, "ue8m0"):
            requant_summaries(means, fmt)
    torch.cuda.synchronize(device)
