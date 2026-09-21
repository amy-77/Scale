"""Segment-aware sparse FP8 MQA scorer for Adaptive-HISA fine rescore.

Scores the same token set as ``sparse_paged_mqa_triton`` with K=1, but walks
contiguous ``(start, length, output_offset)`` intervals so consecutive tokens
share page-table lookups. Enabled via
``SGLANG_NSA_ADAPTIVE_HISA_SEGMENT_SCORER=1``.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False

PAGE = 64
DIM = 128


if _HAS_TRITON:

    @triton.jit
    def _segment_score_kernel(
        Q_ptr,  # [H, D] fp8
        W_ptr,  # [H] f32
        Kv8_ptr,  # [num_phys, PAGE*(D+4)] fp8 view of flat page
        Kv32_ptr,  # same bytes as f32 view
        PageTable_ptr,  # [max_pages] i32
        Context_ptr,  # [1] i32
        SegStart_ptr,
        SegLen_ptr,
        SegOff_ptr,
        SegCount_ptr,
        Logits_ptr,  # [budget] f32
        stride_kv8_p,
        stride_kv8_b,
        stride_kv32_p,
        stride_kv32_b,
        max_pages,
        num_phys,
        budget,
        H: tl.constexpr,
        D: tl.constexpr,
        PAGE: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        seg = tl.program_id(0)
        n_seg = tl.load(SegCount_ptr)
        if seg >= n_seg:
            return
        start = tl.load(SegStart_ptr + seg)
        length = tl.load(SegLen_ptr + seg)
        out_off = tl.load(SegOff_ptr + seg)
        context_len = tl.load(Context_ptr)
        if length <= 0:
            return

        h = tl.arange(0, H)
        d = tl.arange(0, D)
        q = tl.load(Q_ptr + h[:, None] * D + d[None, :])
        w = tl.load(W_ptr + h)

        SCALE_OFFSET = PAGE * D // 4
        for t0 in range(0, length, BLOCK_T):
            t = t0 + tl.arange(0, BLOCK_T)
            active = t < length
            token = start + t
            in_ctx = active & (token >= 0) & (token < context_len)
            logical_page = token // PAGE
            row = token % PAGE
            page_ok = in_ctx & (logical_page >= 0) & (logical_page < max_pages)
            safe_page = tl.where(page_ok, logical_page, 0)
            phys = tl.load(PageTable_ptr + safe_page, mask=page_ok, other=0)
            phys_ok = page_ok & (phys >= 0) & (phys < num_phys)
            safe_phys = tl.where(phys_ok, phys, 0)
            safe_row = tl.where(phys_ok, row, 0)

            k = tl.load(
                Kv8_ptr
                + safe_phys[:, None] * stride_kv8_p
                + (safe_row[:, None] * D + d[None, :]) * stride_kv8_b,
                mask=phys_ok[:, None],
                other=0.0,
            )
            k_scale = tl.load(
                Kv32_ptr
                + safe_phys * stride_kv32_p
                + (SCALE_OFFSET + safe_row) * stride_kv32_b,
                mask=phys_ok,
                other=0.0,
            )
            s = tl.dot(k, tl.trans(q), out_dtype=tl.float32)
            s = s * k_scale[:, None]
            s = tl.maximum(s, 0.0)
            s = s * w[None, :]
            logits = tl.sum(s, axis=1)
            logits = tl.where(phys_ok, logits, float("-inf"))

            out_idx = out_off + t
            out_ok = active & (out_idx >= 0) & (out_idx < budget)
            tl.store(Logits_ptr + out_idx, logits, mask=out_ok)


def sparse_paged_mqa_segments(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    seg_starts: torch.Tensor,
    seg_lengths: torch.Tensor,
    seg_offsets: torch.Tensor,
    seg_count: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    budget: int,
    *,
    logits_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score ``budget`` candidate slots from packed intervals.

    ``q_fp8`` is ``[B=1, seq=1, H, D]`` FP8. Returns ``[1, budget]`` logits.
    """
    assert q_fp8.shape[0] == 1 and q_fp8.shape[1] == 1
    _, _, H, D = q_fp8.shape
    assert D == DIM
    num_phys, paged, _, dplus = kv_cache_fp8.shape
    assert paged == PAGE and dplus == D + 4
    if logits_out is None:
        logits_out = torch.full(
            (1, budget), float("-inf"), dtype=torch.float32, device=q_fp8.device
        )
    else:
        logits_out.fill_(float("-inf"))

    if not _HAS_TRITON or not q_fp8.is_cuda:
        n_seg = int(seg_count.item())
        tokens = []
        for i in range(n_seg):
            s = int(seg_starts[i].item())
            n = int(seg_lengths[i].item())
            tokens.extend(range(s, s + n))
        while len(tokens) < budget:
            tokens.append(-1)
        from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
            sparse_paged_mqa_triton,
        )

        cand = torch.tensor(tokens[:budget], device=q_fp8.device, dtype=torch.int32)
        scored = sparse_paged_mqa_triton(
            q_fp8,
            kv_cache_fp8,
            cand.view(1, 1, -1),
            1,
            weights,
            context_lens,
            block_tables,
        ).squeeze(1)
        logits_out.copy_(scored)
        return logits_out

    q = q_fp8[0, 0].contiguous()
    w = weights.reshape(-1).contiguous()
    assert w.numel() == H
    kv_flat = kv_cache_fp8.view(num_phys, -1)
    kv8 = kv_flat.view(torch.float8_e4m3fn)
    kv32 = kv_flat.view(torch.float32)
    pt = block_tables.reshape(-1).contiguous()
    ctx = context_lens.reshape(-1).contiguous()
    n_seg_cap = int(seg_starts.numel())
    _segment_score_kernel[(n_seg_cap,)](
        q,
        w,
        kv8,
        kv32,
        pt,
        ctx,
        seg_starts,
        seg_lengths,
        seg_offsets,
        seg_count,
        logits_out.view(-1),
        kv8.stride(0),
        kv8.stride(1),
        kv32.stride(0),
        kv32.stride(1),
        pt.numel(),
        num_phys,
        budget,
        H=H,
        D=D,
        PAGE=PAGE,
        BLOCK_T=64,
    )
    return logits_out
