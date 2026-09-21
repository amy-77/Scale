"""GPU partition 的关键 Triton kernel。

建树（raw-FP8 → FP64 Key-SSE 树）：

* ``_fp8_chunk_tree_kernel``：每 16 token 一个 program，反量化后的
  ``[16,128]`` fp64 tile 留在寄存器里逐层归约，只写出 chunk 的两个 ``[D]`` 矩；
* ``_fp8_upper_tree_kernel``：每棵 root 一个 program，归约 16 个 chunk 矩得到上层节点；
  两者都带 chunk/root 偏移（``c0``/``r0``），chunked prefill 下
  ``raw_fp8_tree_triton_incremental`` 只为新增 root 建节点，旧前缀各 level 由
  ``_tree_copy_levels_kernel`` 拷入新树（结果与整棵重建逐位相同）；
* ``_leaf_totals_fp8_kernel``：按叶子分段求 ``sum(K)``，不落地 ``[N,D]`` / prefix。

Split 核心：

* ``_dp_count_reduce_kernel`` / ``_bracket_from_totals_kernel``：同时评估多个 λ
  的叶子数（原子归约成 ``[K]``）并缩小 λ 区间；
* ``_dp_leaf_gain_kernel``：按最终 λ 一次扫描输出叶子 mask 和 split gain；
* ``_staircase_kernel``：repair 的字典序 pop key；实际选出前 deficit 个用
  ``partition_select``（CUDA radix selection，无全量 sort）。

Merge 核心：

* ``_merge_cost_kernel``：由每行的均值 ``totals/len`` 计算相邻叶子的 Ward cost
  （不再在 kernel 里对两侧 totals 各做 128 次 fp64 除法）；
* ``_merge_match_kernel``：选择互不重叠的相邻 merge 边；
* ``_merge_compact_kernel``：真正合并 length/totals 并压紧输出（只写活跃行），
  同时写出新行的均值供下一轮 cost 使用。
* target 模式的 k-th cost 阈值同样走 ``partition_select``。
* 阈值为 ``-inf`` 的空转轮（target 已达成）在 match/index/compact/k-th 中
  device 侧直接早退，launch 数不变（CUDA graph 友好）。

每个 256-token root 的子树相互独立，因此 λ-DP 可由一个 Triton program
处理一棵 root，整次 bottom-up / top-down 尽量留在寄存器中。
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_select as _sel

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


    @triton.jit(do_not_specialize=["n_roots", "n_tokens"])
    def _raw_fp8_score_tree_kernel(
        k_ptr,  # fp8 [N, D], row-major
        scale_ptr,  # fp32 [N]
        flat_ptr,  # fp64 [n_nodes] out, level-major
        n_roots,
        n_tokens,
        ATOM: tl.constexpr,
        R: tl.constexpr,
        LEVELS: tl.constexpr,
        D: tl.constexpr,
    ):
        """Reference single-program variant (one root per program).

        Holds the whole ``[R*ATOM, D]`` root in registers: at R=256, D=128
        that spills heavily, so production uses the two-stage chunked kernels
        below. Kept for parity tests.
        """
        r = tl.program_id(0)
        i = tl.arange(0, R)
        a = tl.arange(0, ATOM)
        d = tl.arange(0, D)
        tok = (r * R + i)[:, None] * ATOM + a[None, :]
        valid = tok < n_tokens
        raw = tl.load(
            k_ptr + tok[:, :, None] * D + d[None, None, :],
            mask=valid[:, :, None],
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(scale_ptr + tok, mask=valid, other=0.0).to(tl.float32)
        v = (raw * scale[:, :, None]).to(tl.float64)
        m = tl.sum(v, axis=1)
        m2 = tl.sum(v * v, axis=1)
        cnt = ATOM * 1.0
        if ATOM == 1:
            # single-token SSE is identically zero (v*v - v*v); skip the fp64 work
            e = tl.zeros((R,), tl.float64)
        else:
            e = tl.sum(m2 - m * m * (1.0 / cnt), axis=1)
        tl.store(flat_ptr + r * R + i, e)
        off = n_roots * R
        for l in tl.static_range(1, LEVELS):
            m = tl.sum(tl.reshape(m, ((R >> l), 2, D)), axis=1)
            m2 = tl.sum(tl.reshape(m2, ((R >> l), 2, D)), axis=1)
            cnt = cnt * 2.0
            # cnt is a power of two: x * (1/cnt) is bit-identical to x / cnt
            e = tl.sum(m2 - m * m * (1.0 / cnt), axis=1)
            j = tl.arange(0, (R >> l))
            tl.store(flat_ptr + off + r * (R >> l) + j, e)
            off += n_roots * (R >> l)

    @triton.jit(do_not_specialize=["n_roots", "n_tokens", "c0"])
    def _fp8_chunk_tree_kernel(
        k_ptr,  # fp8 [N, D], row-major
        scale_ptr,  # fp32 [N]
        flat_ptr,  # fp64 [n_nodes] out, level-major (levels 0..log2(CA))
        mom_ptr,  # fp64 [n_chunks - c0, 2, D] out: chunk sum(K), sum(K^2)
        n_roots,
        n_tokens,
        c0,  # first chunk handled by this launch (incremental tree: old roots kept)
        ATOM: tl.constexpr,
        R: tl.constexpr,
        CA: tl.constexpr,  # atoms per chunk (power of two, <= R)
        LC: tl.constexpr,  # log2(CA)
        D: tl.constexpr,
    ):
        """Stage 1: one chunk of ``CA*ATOM`` tokens per program.

        The dequantized ``[CA*ATOM, D]`` tile (16x128 fp64 = 16 values per
        thread at 4 warps) stays in registers; SSE for every node inside the
        chunk is reduced pairwise from register moments and only the chunk's
        two ``[D]`` moment vectors leave the SM.

        Node positions use the full tree (``n_roots``); ``c0 > 0`` fills only
        chunks ``[c0, c0 + grid)`` (the sealed prefix's nodes are copied from
        the previous chunk's tree) and stores moments relative to ``c0``.
        """
        c = c0 + tl.program_id(0)
        i = tl.arange(0, CA)
        a = tl.arange(0, ATOM)
        d = tl.arange(0, D)
        tok = (c * CA + i)[:, None] * ATOM + a[None, :]
        valid = tok < n_tokens
        raw = tl.load(
            k_ptr + tok[:, :, None] * D + d[None, None, :],
            mask=valid[:, :, None],
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(scale_ptr + tok, mask=valid, other=0.0).to(tl.float32)
        v = (raw * scale[:, :, None]).to(tl.float64)
        m = tl.sum(v, axis=1)
        m2 = tl.sum(v * v, axis=1)
        cnt = ATOM * 1.0
        if ATOM == 1:
            # single-token SSE is identically zero (v*v - v*v): this used to be
            # N*D fp64 mul+div+sub for nothing (the largest fp64 term at ATOM=1)
            e = tl.zeros((CA,), tl.float64)
        else:
            e = tl.sum(m2 - m * m * (1.0 / cnt), axis=1)
        tl.store(flat_ptr + c * CA + i, e)
        off = n_roots * R
        for l in tl.static_range(1, LC + 1):
            m = tl.sum(tl.reshape(m, ((CA >> l), 2, D)), axis=1)
            m2 = tl.sum(tl.reshape(m2, ((CA >> l), 2, D)), axis=1)
            cnt = cnt * 2.0
            # cnt is a power of two: x * (1/cnt) is bit-identical to x / cnt and
            # avoids the div.rn.f64 sequence (not folded by the compiler)
            e = tl.sum(m2 - m * m * (1.0 / cnt), axis=1)
            j = tl.arange(0, (CA >> l))
            tl.store(flat_ptr + off + c * (CA >> l) + j, e)
            off += n_roots * (R >> l)
        tl.store(mom_ptr + ((c - c0) * 2) * D + d, tl.reshape(m, (D,)))
        tl.store(mom_ptr + ((c - c0) * 2 + 1) * D + d, tl.reshape(m2, (D,)))

    @triton.jit(do_not_specialize=["n_roots", "r0"])
    def _fp8_upper_tree_kernel(
        mom_ptr,  # fp64 [n_chunks - r0 * NC, 2, D]
        flat_ptr,  # fp64 [n_nodes] out (levels log2(CA)+1 .. LEVELS-1)
        n_roots,
        r0,  # first root handled by this launch (moments are relative to it)
        ATOM: tl.constexpr,
        R: tl.constexpr,
        CA: tl.constexpr,
        LC: tl.constexpr,  # log2(CA)
        LNC: tl.constexpr,  # log2(R // CA)
        D: tl.constexpr,
    ):
        """Stage 2: one root per program, reduces its ``R/CA`` chunk moments."""
        NC: tl.constexpr = R // CA
        r = r0 + tl.program_id(0)
        q = tl.arange(0, NC)
        d = tl.arange(0, D)
        base = ((r - r0) * NC + q)[:, None] * (2 * D) + d[None, :]
        m = tl.load(mom_ptr + base)
        m2 = tl.load(mom_ptr + base + D)
        cnt = (CA * ATOM) * 1.0
        off = n_roots * (2 * R - (2 * R >> LC))  # first node of level LC
        off += n_roots * (R >> LC)  # -> level LC + 1
        for l2 in tl.static_range(1, LNC + 1):
            m = tl.sum(tl.reshape(m, ((NC >> l2), 2, D)), axis=1)
            m2 = tl.sum(tl.reshape(m2, ((NC >> l2), 2, D)), axis=1)
            cnt = cnt * 2.0
            e = tl.sum(m2 - m * m * (1.0 / cnt), axis=1)  # power-of-two cnt: exact
            j = tl.arange(0, (NC >> l2))
            tl.store(flat_ptr + off + r * (NC >> l2) + j, e)
            off += n_roots * (NC >> l2)

    @triton.jit(do_not_specialize=["n_old", "n_new"])
    def _tree_copy_levels_kernel(
        src_ptr,  # fp64 level-major tree of n_old atoms
        dst_ptr,  # fp64 level-major tree of n_new atoms (n_new >= n_old)
        n_old,
        n_new,
        LEVELS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Copy every level of the old tree to the front of the same level of
        the new tree (grid: ``(LEVELS, cdiv(n_old, BLOCK))``).

        Level ``lv`` holds ``n >> lv`` nodes and starts at
        ``sum_{l < lv} (n >> l)``; the sealed prefix's dyadic nodes are the
        first ``n_old >> lv`` of each level in both trees.
        """
        lv = tl.program_id(0)
        blk = tl.program_id(1)
        src_off = 0
        dst_off = 0
        for l in tl.static_range(LEVELS):
            take = l < lv
            src_off += tl.where(take, n_old >> l, 0)
            dst_off += tl.where(take, n_new >> l, 0)
        size = n_old >> lv
        j = blk * BLOCK + tl.arange(0, BLOCK)
        mask = j < size
        vals = tl.load(src_ptr + src_off + j, mask=mask, other=0.0)
        tl.store(dst_ptr + dst_off + j, vals, mask=mask)

    @triton.jit(do_not_specialize=["n_leaves", "n_tokens"])
    def _leaf_totals_fp8_kernel(
        k_ptr,
        scale_ptr,
        start_ptr,
        len_ptr,
        out_ptr,  # fp64 [M, D]
        n_leaves,
        n_tokens,
        D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        """Segmented FP64 ``sum(K)``; no dense ``[N,D]`` or prefix materialization.

        One program per leaf, full ``[BLOCK_T, D]`` row tiles: every fp8 row is
        one contiguous 128-byte load and ``scale`` is read once per row. (The
        earlier 4 x ``[32, 32]`` column passes issued 32-byte row fragments and
        spent 4 masked fp64 passes on an average 8-token leaf.)
        """
        leaf = tl.program_id(0)
        if leaf >= n_leaves:
            return
        start = tl.load(start_ptr + leaf).to(tl.int64)
        length = tl.load(len_ptr + leaf).to(tl.int64)
        cols = tl.arange(0, D)
        t = tl.arange(0, BLOCK_T)
        acc = tl.zeros((D,), tl.float64)
        for t0 in range(0, length, BLOCK_T):
            rows = start + t0 + t
            valid_t = ((t0 + t) < length) & (rows < n_tokens)
            raw = tl.load(k_ptr + rows[:, None] * D + cols[None, :], mask=valid_t[:, None], other=0.0).to(tl.float32)
            scale = tl.load(scale_ptr + rows, mask=valid_t, other=0.0).to(tl.float32)
            acc += tl.sum(raw.to(tl.float64) * scale.to(tl.float64)[:, None], axis=0)
        tl.store(out_ptr + leaf * D + cols, acc)

    @triton.jit(do_not_specialize=["n_roots"])
    def _dp_leaf_gain_kernel(
        flat_ptr,
        lam_ptr,
        leaf_ptr,  # uint8 [n_nodes] out
        gain_ptr,  # fp64 [n_nodes] out; -inf at atoms
        split_ptr,  # uint8 [n_nodes] scratch
        n_roots,
        R: tl.constexpr,
        LEVELS: tl.constexpr,
    ):
        """One ``flat`` load produces both the λ leaf mask and split gains."""
        r = tl.program_id(0)
        lam = tl.load(lam_ptr)
        i = tl.arange(0, R)
        child = tl.load(flat_ptr + r * R + i)
        cost = child + lam
        tl.store(gain_ptr + r * R + i, tl.full((R,), float("-inf"), tl.float64))
        off = n_roots * R
        for l in tl.static_range(1, LEVELS):
            pair = tl.sum(tl.reshape(child, ((R >> l), 2)), axis=1)
            split = tl.sum(tl.reshape(cost, ((R >> l), 2)), axis=1)
            j = tl.arange(0, (R >> l))
            own = tl.load(flat_ptr + off + r * (R >> l) + j)
            keep = own + lam
            ds = split < keep
            cost = tl.where(ds, split, keep)
            tl.store(split_ptr + off + r * (R >> l) + j, ds.to(tl.uint8))
            tl.store(gain_ptr + off + r * (R >> l) + j, own - pair)
            child = own
            off += n_roots * (R >> l)
        tl.debug_barrier()
        for l2 in tl.static_range(1, LEVELS):
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
        pbase = n_roots * (2 * R - ((2 * R) >> 1))
        act0 = tl.load(split_ptr + pbase + r * (R >> 1) + (i // 2)) != 0
        tl.store(leaf_ptr + r * R + i, act0.to(tl.uint8))

    @triton.jit(do_not_specialize=["n_roots"])
    def _dp_count_reduce_kernel(
        flat_ptr,
        bracket_ptr,  # fp64 [5]: lo, hi, count(hi), done, rounds_used
        total_ptr,  # int64 [K] out, exact sum over roots
        n_roots,
        K: tl.constexpr,
        R: tl.constexpr,
        LEVELS: tl.constexpr,
        KC: tl.constexpr,
    ):
        """Evaluate K lambdas and atomically reduce every root's leaf count.

        Returns immediately once the bracket kernel has flagged convergence
        (``done``): the launch stays in the CUDA graph but skips the DP.
        """
        if tl.load(bracket_ptr + 3) != 0.0:
            return
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
        tl.atomic_add(total_ptr + ks, tl.sum(cnt, axis=1).to(tl.int64))

    @triton.jit(do_not_specialize=["m_leaves", "deficit_tol"])
    def _bracket_from_totals_kernel(
        total_ptr,  # int64 [K]
        bracket_ptr,  # fp64 [5] in/out: lo, hi, count(hi), done, rounds_used
        m_leaves,
        deficit_tol,  # int64: stop once m_leaves - count(hi) <= tol (< 0 = never)
        K: tl.constexpr,
    ):
        if tl.load(bracket_ptr + 3) != 0.0:
            return
        ks = tl.arange(0, K)
        total = tl.load(total_ptr + ks)
        lo = tl.load(bracket_ptr)
        hi = tl.load(bracket_ptr + 1)
        hi_count = tl.load(bracket_ptr + 2)
        frac = (ks + 1).to(tl.float64) / (K + 1)
        cands = lo + (hi - lo) * frac
        n_feasible = tl.sum((total <= m_leaves).to(tl.int64))
        idx = K - n_feasible
        c_lo = tl.sum(tl.where(ks == idx - 1, cands, 0.0))
        c_hi = tl.sum(tl.where(ks == idx, cands, 0.0))
        c_count = tl.sum(tl.where(ks == idx, total, 0)).to(tl.float64)
        new_lo = tl.where(idx == 0, lo, c_lo)
        new_hi = tl.where(idx == K, hi, c_hi)
        new_count = tl.where(idx == K, hi_count, c_count)
        deficit = m_leaves.to(tl.float64) - new_count
        done = (deficit_tol >= 0) & (deficit <= deficit_tol.to(tl.float64))
        tl.store(bracket_ptr, new_lo)
        tl.store(bracket_ptr + 1, new_hi)
        tl.store(bracket_ptr + 2, new_count)
        tl.store(bracket_ptr + 3, tl.where(done, 1.0, 0.0))
        tl.store(bracket_ptr + 4, tl.load(bracket_ptr + 4) + 1.0)

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
            e = m2 - m * m * (1.0 / cnt)  # power-of-two cnt: bit-identical to / cnt
            tl.store(flat_ptr + r * R + i, tl.load(flat_ptr + r * R + i) + e)
            off = n_roots * R
            for l in tl.static_range(1, LEVELS):
                m = tl.sum(tl.reshape(m, ((R >> l), 2)), axis=1)
                m2 = tl.sum(tl.reshape(m2, ((R >> l), 2)), axis=1)
                cnt = cnt * 2.0
                e = m2 - m * m * (1.0 / cnt)
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
        mean_ptr,  # fp64 [cap, C]: totals / max(len, 1) per row
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
        #
        # Row means are precomputed once per row (compact kernel / round-0
        # setup) instead of dividing both totals rows here: that removed 2*C
        # fp64 divides per edge (each row's mean was computed twice, as the
        # left and the right member), leaving this kernel memory-bound. The
        # value is the same IEEE ``t / n`` either way, so costs are bit-identical.
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
        m_l = tl.load(mean_ptr + i[:, None] * C + cols[None, :], mask=live[:, None] & cm[None, :], other=0.0)
        m_r = tl.load(mean_ptr + (i + 1)[:, None] * C + cols[None, :], mask=live[:, None] & cm[None, :], other=0.0)
        delta = m_l - m_r
        cost = n_l * n_r / (n_l + n_r) * tl.sum(delta * delta, axis=1)
        eligible = m & (i + 1 < count) & (cost < thr)
        if max_merge_len > 0:
            eligible = eligible & (len_l + len_r <= max_merge_len)
        tl.store(ecost_ptr + i, tl.where(eligible, cost, float("inf")), mask=m)

    @triton.jit(do_not_specialize=["n_edges"])
    def _merge_match_kernel(
        ecost_ptr,  # fp64 [cap-1]
        count_ptr,  # int64 scalar in: valid rows
        thr_ptr,  # fp64 scalar in: -inf = idle round (nothing eligible)
        sel_ptr,  # uint8 [cap-1] out: selected edges (scratch: sel_run first)
        src_ptr,  # int64 [cap] out: src[dst] = row for kept rows
        new_count_ptr,  # int64 scalar out
        nsel_ptr,  # int64 scalar out
        n_edges,
        BLOCK: tl.constexpr,
    ):
        """按 ``(cost,start)`` 贪心选择互不重叠的相邻边。"""
        # Idle round (target reached, threshold -inf): no edge can be selected.
        # Only the scalars are produced; sel/src are not touched and the
        # compact kernel takes its identity path off the same flag.
        if tl.load(thr_ptr) == float("-inf"):
            tl.store(new_count_ptr, tl.load(count_ptr))
            tl.store(nsel_ptr, tl.zeros((), tl.int64))
            return
        # int32 positions (cap < 2^30) halve the register footprint of the
        # scans; BLOCK is one program so every array is live at once.
        BIG: tl.constexpr = 1 << 30
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
        nm = tl.associative_scan(tl.where(local_min, i, BIG), 0, _min_i64, reverse=True)
        pm = tl.associative_scan(tl.where(local_min, i, -BIG), 0, _max_i64)
        dist = tl.where(to_right, nm - i, tl.where(to_left, i - pm, 0))
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
        thr_ptr,  # fp64 scalar in: -inf = idle round
        src_ptr,  # int64 [cap] out: src[dst] = row for kept rows
        new_count_ptr,  # int64 scalar out
        nsel_ptr,  # int64 scalar out
        n_edges,
        BLOCK: tl.constexpr,
    ):
        """Row compaction index for a given selection (the tail of the match kernel)."""
        if tl.load(thr_ptr) == float("-inf"):
            tl.store(new_count_ptr, tl.load(count_ptr))
            tl.store(nsel_ptr, tl.zeros((), tl.int64))
            return
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
        thr_ptr,  # fp64 scalar in: -inf = idle round (identity compaction)
        out_start_ptr,
        out_len_ptr,
        out_tot_ptr,
        out_mean_ptr,  # fp64 [cap, C] out: new totals / new length (next round's costs)
        cap,
        n_tokens,
        C: tl.constexpr,
        CPAD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # 选中边的右叶子消失；左叶子累加右侧 length 和 sum(K)，再压紧。
        pid = tl.program_id(0)
        n_keep = tl.load(new_count_ptr)
        if pid * BLOCK >= n_keep:
            return  # only live rows are written; dead tail programs exit at once
        j = pid * BLOCK + tl.arange(0, BLOCK)
        m = j < cap
        live = m & (j < n_keep)
        # Idle round: the match/index kernels did not write sel/src; every live
        # row maps to itself and nothing is merged (still copied into the
        # ping-pong output so the caller's buffer alternation stays valid).
        idle = tl.load(thr_ptr) == float("-inf")
        src = tl.load(src_ptr + j, mask=live & (idle == 0), other=0)
        src = tl.where(idle, j.to(tl.int64), src)
        merged = tl.load(sel_ptr + src, mask=live & (idle == 0) & (src + 1 < cap), other=0) != 0
        s = tl.load(start_ptr + src, mask=live, other=n_tokens)
        ln = tl.load(len_ptr + src, mask=live, other=0)
        ln_r = tl.load(len_ptr + src + 1, mask=live & merged, other=0)
        new_len = ln + ln_r
        tl.store(out_start_ptr + j, tl.where(live, s, n_tokens), mask=live)
        tl.store(out_len_ptr + j, tl.where(live, new_len, 0), mask=live)
        cols = tl.arange(0, CPAD)
        cm = cols < C
        lm = live[:, None] & cm[None, :]
        t = tl.load(tot_ptr + src[:, None] * C + cols[None, :], mask=lm, other=0.0)
        t_r = tl.load(
            tot_ptr + (src + 1)[:, None] * C + cols[None, :],
            mask=(live & merged)[:, None] & cm[None, :],
            other=0.0,
        )
        new_tot = t + t_r
        tl.store(out_tot_ptr + j[:, None] * C + cols[None, :], new_tot, mask=lm)
        # Row mean for the next round's Ward costs: the same IEEE ``t / n`` the
        # cost kernel used to do twice per edge, now once per live row. Idle
        # rounds (threshold -inf) never read the means again, so skip them.
        if idle == 0:
            n_new = tl.maximum(new_len.to(tl.float64), 1.0)
            tl.store(out_mean_ptr + j[:, None] * C + cols[None, :], new_tot / n_new[:, None], mask=lm)


MATCH_BLOCK_MIN = 1024
# 16-row tiles: 2 x [16,128] fp64 operands = 64 fp64 per thread at 4 warps,
# no spills (32 rows spilled at 255 registers).
COST_BLOCK = 16
# Compact rows per program. It now carries the per-row fp64 division for the
# means, so small tiles (many programs) hide the divide latency better than the
# old 64-row copy tiles; programs past ``new_count`` exit immediately.
COMPACT_BLOCK = 16


def _merge_workspace(ws: dict, cap: int, c: int, device) -> None:
    if ws.get("cap") != cap or ws.get("c") != c:
        ws.clear()
        ws["cap"] = cap
        ws["c"] = c
        ws["ecost"] = torch.empty(cap - 1, dtype=torch.float64, device=device)
        ws["kth"] = torch.empty(cap - 1, dtype=torch.float64, device=device)
        ws["thr"] = torch.empty(1, dtype=torch.float64, device=device)
        ws["sel"] = torch.empty(cap - 1, dtype=torch.uint8, device=device)
        ws["src"] = torch.empty(cap, dtype=torch.int64, device=device)
        ws["scalars"] = torch.zeros(2, dtype=torch.int64, device=device)
        ws["inf"] = torch.full((1,), float("inf"), dtype=torch.float64, device=device)
        ws["buf_start"] = [None, None]
        ws["buf_len"] = [None, None]
        ws["buf_tot"] = [None, None]
        ws["buf_mean"] = [None, None]
        ws["ping"] = 0


def row_means(totals: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    """``totals / max(len, 1)`` per row, fp64 ``[cap, C]``.

    Round-0 input of the merge; later rounds get their means from the compact
    kernel. Both are the plain IEEE division the cost kernel used to do inline,
    so Ward costs are unchanged bit for bit. Padding rows (len 0, totals 0) are 0.
    """
    n = length.to(torch.float64).clamp_min(1.0)
    return totals / n[:, None]


def merge_costs_triton(
    length: torch.Tensor,
    means: torch.Tensor,
    count: torch.Tensor,
    *,
    max_merge_len: int,
    workspace: dict,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ward cost of every eligible adjacent edge (``+inf`` otherwise), fp64 ``[cap-1]``.

    ``means`` is ``row_means(totals, length)`` (or the compact kernel's output).
    The returned tensor aliases the reusable workspace buffer. Callers that need
    to retain it across another cost launch must copy it themselves.
    """
    cap = length.shape[0]
    n_edges = cap - 1
    c = means.shape[1]
    _merge_workspace(workspace, cap, c, length.device)
    cpad = 1 << (c - 1).bit_length()
    count = count.to(torch.int64).reshape(1).contiguous()
    thr = workspace["inf"] if active is None else torch.where(active.reshape(1), workspace["inf"], -workspace["inf"])
    _merge_cost_kernel[(triton.cdiv(n_edges, COST_BLOCK),)](
        length, means, count, thr, workspace["ecost"], n_edges, int(max_merge_len),
        C=c, CPAD=cpad, BLOCK=COST_BLOCK,
    )
    return workspace["ecost"]


def _kth_threshold(cost: torch.Tensor, k: torch.Tensor, ws: dict) -> torch.Tensor:
    """``[1]`` fp64 successor of the k-th smallest ``cost`` (``-inf`` for ``k <= 0``).

    CUDA: exact radix selection (``partition_select``), one scalar written.
    Fallback: full sort of a reusable workspace copy.
    """
    k = k.to(torch.int64).reshape(1)
    if _sel.available(cost):
        return _sel.kth_threshold_f64(cost, k, out=ws["thr"])
    ws["kth"].copy_(cost)
    kth = torch.sort(ws["kth"]).values.gather(0, (k - 1).clamp(0, cost.shape[0] - 1))
    return torch.where(k > 0, torch.nextafter(kth, ws["inf"]), -ws["inf"])


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
    means: torch.Tensor | None = None,
    ecost: torch.Tensor | None = None,
    cap_merges: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One synchronous non-overlapping merge round in three launches.

    Returns ``(start, length, totals, means, count, n_selected)`` (new buffers).
    Same semantics as the torch loop body in ``partition_gpu.merge_sync_nonoverlap``.

    ``means`` is ``row_means(totals, length)`` of the input rows (computed here
    when omitted); the returned ``means`` belong to the returned rows and feed
    the next round's ``merge_costs_triton`` / ``merge_round_triton``.
    ``ecost`` (fp64 ``[cap-1]``, ``+inf`` on ineligible edges) skips the cost
    kernel; ``threshold`` is then applied elementwise. ``cap_merges`` (device
    int64 scalar) keeps only that many cheapest selected edges (ties at the
    boundary included) - the target-count merge's per-round budget.

    A ``threshold`` of ``-inf`` (target already reached) is an idle round: the
    match/index kernels only write the scalars, the compact kernel copies rows
    through unchanged, and the k-th selection returns immediately for ``k <= 0``.

    Only rows ``< new_count`` of the returned buffers are written; the tail is
    stale and must be normalised once by the caller after the last round
    (``merge_sync_nonoverlap`` does).
    """
    cap = start.shape[0]
    n_edges = cap - 1
    c = totals.shape[1]
    device = start.device
    if cap < 2:
        zero = torch.zeros((), dtype=torch.int64, device=device)
        if means is None:
            means = row_means(totals, length)
        return start, length, totals, means, count, zero
    cpad = 1 << (c - 1).bit_length()
    # Size the single-program match kernel to the capacity (power of two, at
    # least 1K rows): fewer idle lanes and no register spills up to 8K rows.
    # Each distinct capacity compiles once (disk-cached by Triton).
    block_rows = max(MATCH_BLOCK_MIN, 1 << max(1, (cap - 1).bit_length()))
    ws = workspace
    _merge_workspace(ws, cap, c, device)
    sel, src, scalars = ws["sel"], ws["src"], ws["scalars"]
    new_count, n_sel = scalars[0:1], scalars[1:2]
    count = count.to(torch.int64).reshape(1).contiguous()
    threshold = threshold.to(torch.float64).reshape(1).contiguous()
    blk = COMPACT_BLOCK
    if ecost is None:
        if means is None:
            means = row_means(totals, length)
        ecost = ws["ecost"]
        _merge_cost_kernel[(triton.cdiv(n_edges, COST_BLOCK),)](
            length, means, count, threshold, ecost, n_edges, int(max_merge_len), C=c, CPAD=cpad, BLOCK=COST_BLOCK
        )
    else:
        ecost = torch.where(ecost < threshold, ecost, ws["inf"])
    _merge_match_kernel[(1,)](
        ecost, count, threshold, sel, src, new_count, n_sel, n_edges, BLOCK=block_rows, num_warps=32
    )
    if cap_merges is not None:
        picked = sel != 0
        sel_cost = torch.where(picked, ecost, ws["inf"])
        thr2 = _kth_threshold(sel_cost, cap_merges, ws)
        sel = (picked & (ecost < thr2)).to(torch.uint8)
        _merge_index_kernel[(1,)](
            sel, count, threshold, src, new_count, n_sel, n_edges, BLOCK=block_rows, num_warps=32
        )
    ping = int(ws["ping"])
    if ws["buf_start"][ping] is None or ws["buf_start"][ping].dtype != start.dtype:
        ws["buf_start"][ping] = torch.empty(cap, dtype=start.dtype, device=device)
        ws["buf_len"][ping] = torch.empty(cap, dtype=length.dtype, device=device)
        ws["buf_tot"][ping] = torch.empty((cap, c), dtype=totals.dtype, device=device)
        ws["buf_mean"][ping] = torch.empty((cap, c), dtype=torch.float64, device=device)
    out_start = ws["buf_start"][ping]
    out_len = ws["buf_len"][ping]
    out_tot = ws["buf_tot"][ping]
    out_mean = ws["buf_mean"][ping]
    ws["ping"] = 1 - ping
    _merge_compact_kernel[(triton.cdiv(cap, blk),)](
        start, length, totals, sel, src, new_count, threshold,
        out_start, out_len, out_tot, out_mean, cap, int(n_tokens), C=c, CPAD=cpad, BLOCK=blk,
    )
    return out_start, out_len, out_tot, out_mean, new_count.clone().reshape(()), n_sel.clone().reshape(())


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
    kc = DP_KC if k % DP_KC == 0 else (4 if k % 4 == 0 else (2 if k % 2 == 0 else 1))
    out = torch.empty((k, layout.n_roots), dtype=torch.int32, device=flat.device)
    _dp_count_kernel[(layout.n_roots, k // kc)](
        flat, lams, out, layout.n_roots, R=r_atoms, LEVELS=levels, KC=kc, num_warps=DP_WARPS
    )
    return out.sum(dim=1, dtype=torch.int64)


BRACKET_STATE = 5  # fp64: lo, hi, count(hi), done flag, rounds actually run
# λ-DP program shape. The per-level ``reshape + sum`` pair reductions are warp
# shuffles inside one warp but shared-memory layout conversions across warps,
# so one warp per (root, 8 λ) is ~1.7x faster than 4 warps x 16 λ on H20
# (measured 117 -> 69 us per round at 128K, 41 -> 24 us at 32K; results are
# integer sums of elementwise fp64 compares, so the choice cannot change them).
DP_KC = 8
DP_WARPS = 1


def search_rounds_triton(
    flat: torch.Tensor,
    layout,
    bracket: torch.Tensor,
    m_leaves: int,
    *,
    candidates: int,
    rounds: int,
    deficit_tol: int = -1,
) -> None:
    """Run up to ``rounds`` K-way bracketing rounds in place on the bracket state.

    ``bracket`` is fp64 ``[BRACKET_STATE]`` = ``[lo, hi, count(hi), done, used]``.
    Each round reduces leaf counts directly to ``[K]`` int64 (no ``[K, roots]``
    table). With ``deficit_tol >= 0`` the bracket kernel flags ``done`` once
    ``m_leaves - count(hi) <= deficit_tol``; the remaining launches then return
    immediately (fixed launch count for CUDA graphs, no host sync).
    """
    r_atoms, levels = _check(layout)
    k = int(candidates)
    if k & (k - 1) or k < 1:
        raise ValueError("lambda_candidates must be a power of two for the fused search")
    kc = min(DP_KC, k)
    totals = torch.empty(k, dtype=torch.int64, device=flat.device)
    for _ in range(int(rounds)):
        totals.zero_()
        _dp_count_reduce_kernel[(layout.n_roots, k // kc)](
            flat, bracket, totals, layout.n_roots, K=k, R=r_atoms, LEVELS=levels, KC=kc, num_warps=DP_WARPS
        )
        _bracket_from_totals_kernel[(1,)](totals, bracket, int(m_leaves), int(deficit_tol), K=k)


def dp_leaf_mask_triton(flat: torch.Tensor, layout, lam: torch.Tensor) -> torch.Tensor:
    """Flat bool leaf mask at one λ (same result as ``partition_gpu.dp_leaf_mask``)."""
    leaf, _gain = dp_leaf_and_gain_triton(flat, layout, lam)
    return leaf


def dp_leaf_and_gain_triton(
    flat: torch.Tensor, layout, lam: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Leaf mask and split gain from one traversal of ``flat``."""
    r_atoms, levels = _check(layout)
    lam = lam.to(torch.float64).reshape(1).contiguous()
    leaf = torch.empty(layout.n_nodes, dtype=torch.uint8, device=flat.device)
    gain = torch.empty(layout.n_nodes, dtype=torch.float64, device=flat.device)
    scratch = torch.empty(layout.n_nodes, dtype=torch.uint8, device=flat.device)
    _dp_leaf_gain_kernel[(layout.n_roots,)](
        flat, lam, leaf, gain, scratch, layout.n_roots, R=r_atoms, LEVELS=levels
    )
    return leaf.to(torch.bool), gain


TREE_CHUNK_TOKENS = 16  # stage-1 tile rows for the raw-FP8 tree (D=128)
# Row tile of the segmented leaf-sum kernel: [8,128] fp64 = 8 values per thread
# at 4 warps; the mean leaf is 8 tokens at compression 8, longer leaves loop.
LEAF_TOTALS_ROWS = 8


def raw_fp8_tree_triton(
    k_fp8: torch.Tensor, k_scale: torch.Tensor, layout, *, single_program: bool = False
) -> torch.Tensor:
    """FP64 level-major Key-SSE tree directly from row-major FP8 keys.

    ``single_program=True`` selects the one-root-per-program reference kernel
    (spills at R=256; parity/debug only).
    """
    r_atoms, levels = _check(layout)
    dim = int(k_fp8.shape[1])
    if dim != 128:
        raise ValueError(f"raw-FP8 tree supports D=128, got {dim}")
    if k_fp8.dtype == torch.uint8:
        k_fp8 = k_fp8.view(torch.float8_e4m3fn)
    if not k_fp8.is_contiguous():
        k_fp8 = k_fp8.contiguous()
    scale = k_scale.reshape(-1).to(torch.float32).contiguous()
    flat = torch.empty(layout.n_nodes, dtype=torch.float64, device=k_fp8.device)
    if single_program:
        _raw_fp8_score_tree_kernel[(layout.n_roots,)](
            k_fp8, scale, flat, layout.n_roots, layout.n_tokens,
            ATOM=layout.atom, R=r_atoms, LEVELS=levels, D=dim, num_warps=4,
        )
        return flat
    # Stage 1 tiles of TREE_CHUNK_TOKENS tokens (16x128 fp64 fits registers);
    # stage 2 reduces the R/CA chunk moments of each root.
    ca = max(1, min(r_atoms, TREE_CHUNK_TOKENS // layout.atom))
    lc = ca.bit_length() - 1
    nc = r_atoms // ca
    lnc = nc.bit_length() - 1
    _fill_fp8_tree_roots(k_fp8, scale, flat, layout, 0, r_atoms, ca, lc, nc, lnc, dim)
    return flat


def _fill_fp8_tree_roots(k_fp8, scale, flat, layout, first_root, r_atoms, ca, lc, nc, lnc, dim):
    """Write the nodes of roots ``[first_root, n_roots)`` of ``layout`` into ``flat``."""
    new_roots = layout.n_roots - first_root
    if new_roots <= 0:
        return
    n_chunks = new_roots * nc
    mom = torch.empty((n_chunks, 2, dim), dtype=torch.float64, device=k_fp8.device)
    _fp8_chunk_tree_kernel[(n_chunks,)](
        k_fp8, scale, flat, mom, layout.n_roots, layout.n_tokens, first_root * nc,
        ATOM=layout.atom, R=r_atoms, CA=ca, LC=lc, D=dim, num_warps=4,
    )
    if nc > 1:
        _fp8_upper_tree_kernel[(new_roots,)](
            mom, flat, layout.n_roots, first_root,
            ATOM=layout.atom, R=r_atoms, CA=ca, LC=lc, LNC=lnc, D=dim, num_warps=4,
        )


TREE_COPY_BLOCK = 1024


def raw_fp8_tree_triton_incremental(
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    layout,
    cached_flat: torch.Tensor,
    cached_tokens: int,
) -> torch.Tensor:
    """Extend the previous chunk's tree instead of rebuilding it.

    Dyadic nodes never straddle a root, so for the sealed prefix
    ``[0, cached_tokens)`` every node's ``sum(K)`` / ``sum(K^2)`` — and hence
    its SSE — is identical in the tree of the longer prefix. Only the level
    offsets move (each level grows). The old levels are copied into place and
    the stage-1/stage-2 kernels run for the new roots only, so the raw FP8
    keys of the prefix are not read again. Bitwise identical to
    :func:`raw_fp8_tree_triton` on the full prefix (per-node fp64 reductions
    are the same); λ-search, Split and Merge then run on the complete tree as
    before, because their global targets (M0, λ, merge target) do change.

    Falls back to the full build when the cache does not line up (different
    atom/root, not a whole number of roots, or not a strict prefix).
    """
    r_atoms, _levels = _check(layout)
    n_old = int(cached_tokens)
    old_layout = type(layout)(layout.atom, layout.root, n_old)
    if (
        n_old <= 0
        or n_old % layout.root
        or n_old >= layout.n_tokens
        or cached_flat.numel() != old_layout.n_nodes
        or cached_flat.dtype != torch.float64
        or cached_flat.device != k_fp8.device
    ):
        return raw_fp8_tree_triton(k_fp8, k_scale, layout)
    dim = int(k_fp8.shape[1])
    if dim != 128:
        raise ValueError(f"raw-FP8 tree supports D=128, got {dim}")
    if k_fp8.dtype == torch.uint8:
        k_fp8 = k_fp8.view(torch.float8_e4m3fn)
    if not k_fp8.is_contiguous():
        k_fp8 = k_fp8.contiguous()
    scale = k_scale.reshape(-1).to(torch.float32).contiguous()
    flat = torch.empty(layout.n_nodes, dtype=torch.float64, device=k_fp8.device)
    grid = (layout.levels, triton.cdiv(old_layout.n_atoms, TREE_COPY_BLOCK))
    _tree_copy_levels_kernel[grid](
        cached_flat, flat, old_layout.n_atoms, layout.n_atoms,
        LEVELS=layout.levels, BLOCK=TREE_COPY_BLOCK, num_warps=4,
    )
    ca = max(1, min(r_atoms, TREE_CHUNK_TOKENS // layout.atom))
    lc = ca.bit_length() - 1
    nc = r_atoms // ca
    lnc = nc.bit_length() - 1
    _fill_fp8_tree_roots(
        k_fp8, scale, flat, layout, old_layout.n_roots, r_atoms, ca, lc, nc, lnc, dim
    )
    return flat


def leaf_totals_fp8_triton(
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    leaf_start: torch.Tensor,
    leaf_len: torch.Tensor,
    n_tokens: int,
) -> torch.Tensor:
    """FP64 ``[M,128]`` segmented key sums, without a dense prefix."""
    dim = int(k_fp8.shape[1])
    if k_fp8.dtype == torch.uint8:
        k_fp8 = k_fp8.view(torch.float8_e4m3fn)
    if not k_fp8.is_contiguous():
        k_fp8 = k_fp8.contiguous()
    scale = k_scale.reshape(-1).to(torch.float32).contiguous()
    capacity = int(leaf_start.numel())
    out = torch.zeros((capacity, dim), dtype=torch.float64, device=k_fp8.device)
    if capacity:
        _leaf_totals_fp8_kernel[(capacity,)](
            k_fp8,
            scale,
            leaf_start.to(torch.int64).contiguous(),
            leaf_len.to(torch.int64).contiguous(),
            out,
            capacity,
            int(n_tokens),
            D=dim,
            BLOCK_T=LEAF_TOTALS_ROWS,
            num_warps=4,
        )
    return out


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
    representative build compiles everything except the capacity-sized merge
    match/index blocks, which are compiled explicitly below (up to 16K rows).
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
    # The match/index kernels are sized to the merge capacity; compile every
    # power-of-two block up to 16K rows (128K tokens at compression 8) now.
    ws: dict = {}
    for cap in (1024, 2048, 4096, 8192, 16384):
        start = torch.arange(cap, dtype=torch.int64, device=device)
        length = torch.ones(cap, dtype=torch.int64, device=device)
        totals = torch.zeros((cap, 128), dtype=torch.float64, device=device)
        count = torch.full((), cap, dtype=torch.int64, device=device)
        thr = torch.full((), float("inf"), dtype=torch.float64, device=device)
        means = row_means(totals, length)
        merge_round_triton(start, length, totals, count, thr, max_merge_len=0, n_tokens=cap, workspace=ws, means=means)
        cost = merge_costs_triton(length, means, count, max_merge_len=0, workspace=ws)
        merge_round_triton(
            start, length, totals, count, thr, max_merge_len=0, n_tokens=cap, workspace=ws,
            means=means, ecost=cost, cap_merges=count,
        )
        # idle-round variant (threshold -inf) so its code path is compiled too
        merge_round_triton(
            start, length, totals, count, -thr, max_merge_len=0, n_tokens=cap, workspace=ws,
            means=means, ecost=cost, cap_merges=count * 0,
        )
    if cfg.build_summaries:
        keys = torch.zeros(n, 128, dtype=torch.float8_e4m3fn, device=device)
        scale = torch.ones(n, dtype=torch.float32, device=device)
        means = leaf_key_means(keys, scale, part.leaf_start, part.leaf_len)
        for fmt in (None, "ue8m0"):
            requant_summaries(means, fmt)
    if cfg.root // cfg.atom >= 2:
        # raw-FP8 tree: full build and the incremental (cached prefix) variant.
        from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import TreeLayout

        keys = torch.zeros(2 * cfg.root, 128, dtype=torch.float8_e4m3fn, device=device)
        scale = torch.ones(2 * cfg.root, dtype=torch.float32, device=device)
        old = raw_fp8_tree_triton(keys, scale, TreeLayout(cfg.atom, cfg.root, cfg.root))
        raw_fp8_tree_triton_incremental(
            keys, scale, TreeLayout(cfg.atom, cfg.root, 2 * cfg.root), old, cfg.root
        )
    torch.cuda.synchronize(device)
