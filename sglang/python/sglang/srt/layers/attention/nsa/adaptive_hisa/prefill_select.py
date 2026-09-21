"""Adaptive sparse prefill: Top-2048 from summary-selected candidates.

Chunked prefill (B=1) processes the prompt in ``chunked_prefill_size`` query
chunks. With ``SPARSE_PREFILL=1`` every chunk builds the partition + FP8
summaries of its sealed prefix on the side stream (``prefill_runtime``), so
when the *next* chunk reaches a layer's indexer the following is available for
the prefix ``[0, n_complete)``::

    leaf_start / leaf_len   int32 [capacity]   (padding rows: len = 0)
    summaries               fp8   [capacity, 128] + fp32 scale

For that chunk's ``n_q`` queries the official DSA indexer would compute dense
``[n_q, N]`` logits and take Top-2048 per row. This module replaces that with
the same two-level scheme the decode path uses, batched over query rows:

1. coarse   ``[n_q, capacity]`` = fp8_mqa_logits(q, summaries); leaves that
            start in ``[0, sink_tokens)`` are forced to the top (attention sink)
2. select   per row: leaves in descending coarse score until
            ``prefill_candidate_tokens`` raw tokens (the crossing leaf is clipped);
            a batched token-weighted radix select (one CTA per row, same
            algorithm as the decode ``weighted_select``) replaces the former
            argsort + cumsum + per-slot binary search and writes each selected
            interval directly (``select_candidates_weighted``)
3. local    every row also gets its causal local window
            ``[n_complete, pos]`` (the current chunk + unsealed tail), which is
            what the decode path calls sink/tail
4. fine     exact fp8 logits on the candidate tokens only: HISA's persistent
            block-sparse kernel with K=1 over the flat index-K (``_fine_scores``;
            falls back to the paged K=1 kernel without ``k_flat``)
5. Top-2048 ``hisa_topk_candidates_fused`` -> request-relative token ids,
            identical in format to ``fast_topk_v2(..., row_starts=ks)``

Rows are processed in ``sparse_prefill_rows`` sub-batches to bound the
candidate/logit workspaces. Everything is device-side (no ``.item()``), so the
CPU keeps running ahead of the GPU exactly as in the dense path.

Admission (else the caller falls back to the dense DSA path):
* B=1 extend, no capture / spec / CP / PP (same as the partition builder)
* a summary entry of the current epoch with ``prefill_candidate_tokens <= n_complete
  <= chunk_start`` (chunk 0 and 1 are dense; the leaf budget always fills)
* unfused ragged Top-K output (``SGLANG_NSA_FUSE_TOPK=0`` or forced unfused)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import triton
import triton.language as tl

from .config import get_config, wants_layer

logger = logging.getLogger(__name__)
_NOTED: set = set()
_HERE = Path(__file__).resolve().parent


def _note(key: str, msg: str, *args) -> None:
    if key not in _NOTED:
        _NOTED.add(key)
        logger.info(msg, *args)


def _request_index(forward_batch) -> int | None:
    slots = getattr(forward_batch, "req_pool_indices_cpu", None)
    if slots is not None and len(slots) == 1 and slots[0] is not None:
        return int(slots[0])
    try:
        return int(forward_batch.req_pool_indices[0].item())
    except Exception:
        return None


def _fused_topk_requested(metadata) -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_NSA_FUSE_TOPK.get()) and not getattr(
        metadata, "force_unfused_topk", False
    )


def _is_final_chunk(forward_batch) -> bool:
    final = getattr(forward_batch, "prefill_final_cpu", None)
    return bool(final) and bool(final[0])


def sparse_prefill_admission(forward_batch, layer_id: int, metadata) -> tuple | None:
    """Return ``(cfg, entry, chunk_start, seq_len, budget)`` or ``None`` (dense path).

    The final prompt chunk produces the first output token, so it gets its own
    treatment: ``sparse_prefill_dense_final`` keeps the dense DSA indexer for
    it, ``sparse_prefill_final_candidates`` gives it a larger leaf budget.
    """
    cfg = get_config()
    if not cfg.sparse_prefill or not cfg.enabled or not wants_layer(layer_id):
        return None
    final = _is_final_chunk(forward_batch)
    if final and cfg.sparse_prefill_dense_final:
        return None
    budget = cfg.final_candidate_tokens if final else cfg.prefill_candidate_tokens
    from .prefill_runtime import STATE, _admission_skip, get_summary_entry

    if _admission_skip(forward_batch) is not None:
        return None
    if forward_batch.seq_lens_cpu is None or forward_batch.extend_seq_lens_cpu is None:
        return None
    if _fused_topk_requested(metadata):
        _note("fused", "adaptive-hisa sparse prefill needs unfused Top-K output; dense path kept")
        return None
    req_idx = _request_index(forward_batch)
    if req_idx is None:
        return None
    entry = get_summary_entry(layer_id, req_idx)
    if entry is None or entry.capacity == 0 or entry.epoch != STATE.epoch(req_idx):
        return None
    seq_len = int(forward_batch.seq_lens_cpu[0])
    extend = int(forward_batch.extend_seq_lens_cpu[0])
    chunk_start = seq_len - extend
    if extend <= 0 or entry.n_complete < budget or entry.n_complete > chunk_start:
        return None
    return cfg, entry, chunk_start, seq_len, budget


def _summary_kv(pool, entry) -> tuple[torch.Tensor, torch.Tensor]:
    """Leaf-order ``(fp8 [capacity,128], fp32 [capacity])`` from the summary pages."""
    keys, scales = pool.read(entry)
    return keys.contiguous(), scales.contiguous()


def coarse_leaf_scores(q_fp8, weights, keys, scales, num_leaves, leaf_len,
                       leaf_start=None, sink: int = 0) -> torch.Tensor:
    """``[n_q, capacity]`` fp32 summary logits; padding / empty leaves = -inf.

    Leaves that start inside ``[0, sink)`` get +inf so they rank first: the
    attention-sink tokens carry a large share of the logit mass at many layers
    but a mean summary dilutes them (on a 33K LongBench dump, layer 10's
    selected relu-logit mass went from 0.39 to 1.00 of the dense Top-2048 with
    the sink leaf forced). This is the prefill counterpart of the decode
    ``sink_tokens`` guard, without duplicating tokens.
    """
    import deep_gemm

    n_q = q_fp8.shape[0]
    ks = torch.zeros(n_q, dtype=torch.int32, device=q_fp8.device)
    ke = num_leaves.reshape(1).to(torch.int32).expand(n_q).contiguous()
    keys = keys.reshape(-1, keys.shape[-1]).contiguous()
    scales = scales.reshape(-1).to(torch.float32).contiguous()
    coarse = deep_gemm.fp8_mqa_logits(q_fp8, (keys, scales), weights, ks, ke, clean_logits=False)
    cap = keys.shape[0]
    coarse = coarse[:, :cap]
    col = torch.arange(cap, device=q_fp8.device)
    invalid = (col >= num_leaves.reshape(1)) | (leaf_len[:cap] <= 0)
    coarse = coarse.masked_fill(invalid.unsqueeze(0), float("-inf"))
    if sink > 0 and leaf_start is not None:
        forced = (leaf_start[:cap] < sink) & ~invalid
        coarse.masked_fill_(forced.unsqueeze(0), float("inf"))
    return coarse


def rank_leaves(coarse: torch.Tensor, leaf_start, leaf_len):
    """Descending score order per row (int32 leaf ids) and the inclusive token prefix.

    Legacy path (``SGLANG_NSA_ADAPTIVE_HISA_PREFILL_SELECT=argsort``); the
    default is the batched weighted radix selector below, which needs neither
    the sort nor the prefix.
    """
    order = torch.argsort(coarse, dim=1, descending=True)
    incl = torch.cumsum(leaf_len[order], dim=1, dtype=torch.int32)
    return order.to(torch.int32), incl


WEIGHTED_SELECT_MAX_LEAVES = 4096
_SELECT_OPS = None


def _weighted_select_ops():
    """JIT-compile (once, cached by torch) the batched weighted selector."""
    global _SELECT_OPS
    if _SELECT_OPS is None:
        from torch.utils.cpp_extension import load

        extra = os.environ.get("ADAPTIVE_HISA_PREFILL_SELECT_CFLAGS", "").split()
        load(
            name="adaptive_hisa_prefill_select",
            sources=[str(_HERE / "csrc" / "prefill_weighted_select.cu")],
            extra_cuda_cflags=["-O3", *extra],
            is_python_module=False,
            verbose=bool(int(os.environ.get("ADAPTIVE_HISA_SELECT_VERBOSE", "0"))),
        )
        _SELECT_OPS = torch.ops.adaptive_hisa_prefill_select
    return _SELECT_OPS


def select_mode() -> str:
    """``weighted`` (default, batched radix selector) or ``argsort`` (legacy)."""
    mode = os.environ.get("SGLANG_NSA_ADAPTIVE_HISA_PREFILL_SELECT", "weighted").strip().lower()
    if mode not in ("weighted", "argsort"):
        raise ValueError(f"SGLANG_NSA_ADAPTIVE_HISA_PREFILL_SELECT={mode!r}; use weighted|argsort")
    return mode


def select_candidates_weighted(
    coarse: torch.Tensor,
    leaf_start: torch.Tensor,
    leaf_len: torch.Tensor,
    num_leaves: torch.Tensor,
    rows: slice,
    budget: int,
    n_complete: int,
    local_len: int,
    *,
    intervals: bool = False,
):
    """Batched weighted radix select: ``[R, budget + local_len]`` candidates for ``rows``.

    Same output contract as ``expand_candidates`` (leaf tokens first, then the
    local window), computed straight from ``coarse`` without argsort/cumsum:
    per row a token-weighted 4-pass radix select finds the exact score
    threshold, a logical-order scan clips the crossing leaf (ties broken by
    leaf index), and one warp writes each selected interval. Leaf tokens come
    out in logical order (fine scoring and Top-K are permutation invariant).

    With ``intervals=True`` also returns ``((seg_start, seg_len, seg_offset),
    seg_count)`` — packed ``[R, capacity]`` int32 segments per row — for a
    fine scorer that consumes intervals instead of token ids.
    """
    row0, row1 = rows.indices(coarse.shape[0])[:2]
    R = row1 - row0
    total = budget + local_len
    device = coarse.device
    out = torch.empty((R, total), dtype=torch.int32, device=device)
    if num_leaves.dtype != torch.int32:
        num_leaves = num_leaves.reshape(1).to(torch.int32)
    seg = None
    seg_count = None
    if intervals:
        cap = coarse.shape[1]
        seg = tuple(torch.empty((R, cap), dtype=torch.int32, device=device) for _ in range(3))
        seg_count = torch.empty((R,), dtype=torch.int32, device=device)
    _weighted_select_ops().select(
        coarse, row0, R, leaf_len, leaf_start, num_leaves.reshape(1), out,
        *(seg if seg is not None else (None, None, None)), seg_count,
        budget, n_complete, total,
    )
    if intervals:
        return out, seg, seg_count
    return out


@triton.jit
def _expand_slots_kernel(
    order_ptr, incl_ptr, start_ptr, len_ptr, out_ptr,
    stride_rank, stride_out, row0, M, budget, n_complete, total,
    LOG_M: tl.constexpr, BLOCK: tl.constexpr,
):
    """out[r, s] = token of candidate slot ``s`` for ranked row ``row0 + r``.

    s <  budget : lower_bound(incl[r], s) gives the ranked leaf holding slot s
                  (the crossing leaf is clipped automatically); token =
                  leaf_start + (s - (incl - leaf_len)).
    s >= budget : local window token ``n_complete + (s - budget)``.
    """
    r = tl.program_id(0)
    s = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    smask = s < total
    rank_base = (row0 + r) * stride_rank
    lo = tl.zeros([BLOCK], dtype=tl.int32)
    hi = tl.full([BLOCK], M, dtype=tl.int32)
    for _ in range(LOG_M):
        active = lo < hi
        mid = (lo + hi) // 2
        v = tl.load(incl_ptr + rank_base + tl.minimum(mid, M - 1))
        gt = v > s
        hi = tl.where(active & gt, mid, hi)
        lo = tl.where(active & (gt == 0), mid + 1, lo)
    j = tl.minimum(lo, M - 1)
    leaf = tl.load(order_ptr + rank_base + j)
    start = tl.load(start_ptr + leaf)
    length = tl.load(len_ptr + leaf)
    incl = tl.load(incl_ptr + rank_base + j)
    leaf_tok = start + (s - (incl - length))
    local_tok = n_complete + (s - budget)
    tok = tl.where(s < budget, leaf_tok, local_tok)
    tl.store(out_ptr + r * stride_out + s, tok, mask=smask)


def expand_candidates(order, incl, leaf_start, leaf_len, rows: slice, budget: int,
                      n_complete: int, local_len: int) -> torch.Tensor:
    """``[R, budget + local_len]`` int32 candidate token ids for ``rows``.

    Slot ``s < budget`` maps to the ranked leaf whose inclusive prefix first
    exceeds ``s`` (clipping the crossing leaf); slots ``>= budget`` hold the
    local window ``n_complete + t``. Rows whose window is shorter get ids beyond
    their position, which the fine kernel scores as -inf. Requires
    ``incl[:, -1] >= budget`` (sealed prefix at least ``budget`` tokens).
    """
    row0, row1 = rows.indices(order.shape[0])[:2]
    R = row1 - row0
    M = order.shape[1]
    total = budget + local_len
    out = torch.empty((R, total), dtype=torch.int32, device=order.device)
    BLOCK = 256
    grid = (R, triton.cdiv(total, BLOCK))
    _expand_slots_kernel[grid](
        order, incl, leaf_start, leaf_len, out,
        order.stride(0), out.stride(0), row0, M, budget, n_complete, total,
        LOG_M=max(1, (M + 1).bit_length()), BLOCK=BLOCK,
    )
    return out


def sparse_prefill_topk(
    forward_batch,
    layer_id: int,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens_expanded: torch.Tensor,
    block_tables: torch.Tensor,
    admission: tuple,
    k_flat: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """``[n_q, 2048]`` int32 request-relative Top-K token ids (-1 padded).

    ``k_flat`` is the dense path's already gathered ``(k_fp8 [N,128], k_scale [N])``
    of this request (B=1, so flat index == position); it saves re-gathering
    the local window from the paged cache.
    """
    from .summary_pool import get_summary_pool

    cfg, entry, chunk_start, seq_len, budget = admission
    device = q_fp8.device
    if entry.ready_event is not None:
        torch.cuda.current_stream(device).wait_event(entry.ready_event)
    pool = get_summary_pool(forward_batch.token_to_kv_pool, cfg.summary_compression)
    keys, scales = _summary_kv(pool, entry)
    part = entry.partition
    raw_pages = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(layer_id)
    out = sparse_topk_core(
        q_fp8, weights, seq_lens_expanded, raw_pages, block_tables[:1],
        keys, scales, part.leaf_start[: entry.capacity], part.leaf_len[: entry.capacity],
        part.num_leaves, int(entry.n_complete), seq_len, cfg, k_flat=k_flat, budget=budget,
    )
    _note(
        ("first", layer_id),
        "adaptive-hisa sparse prefill layer=%d rows=%d n_complete=%d chunk_start=%d "
        "seq_len=%d capacity=%d budget=%d rows_per_step=%d final=%d",
        layer_id, q_fp8.shape[0], entry.n_complete, chunk_start, seq_len, entry.capacity,
        budget, cfg.sparse_prefill_rows, _is_final_chunk(forward_batch),
    )
    return out


def sparse_topk_core(
    q_fp8: torch.Tensor,            # [n_q, H, 128] fp8
    weights: torch.Tensor,          # [n_q, H] fp32
    seq_lens_expanded: torch.Tensor,  # [n_q] position + 1
    raw_pages: torch.Tensor,        # [num_pages, 64 * 132] uint8 index-K cache
    table: torch.Tensor,            # [1, max_pages] int32 logical -> physical page
    keys: torch.Tensor,             # [capacity, 128] fp8 summaries (leaf order)
    scales: torch.Tensor,           # [capacity] fp32
    leaf_start: torch.Tensor,
    leaf_len: torch.Tensor,
    num_leaves: torch.Tensor,
    n_complete: int,
    seq_len: int,
    cfg,
    k_flat: tuple[torch.Tensor, torch.Tensor] | None = None,  # optional flat (fp8 [>=seq_len,128], fp32 [>=seq_len])
    dense_local: bool = True,
    budget: int | None = None,      # leaf-token budget; default cfg.prefill_candidate_tokens
) -> torch.Tensor:
    """Two-level Top-K for one chunk of queries (see module docstring).

    The leaf candidates (token ids of the selected leaves) are scored by
    ``_fine_scores``: HISA's persistent block-sparse kernel over the flat
    index-K when ``k_flat`` is given (one CTA per row and 2048 candidates, Q
    loaded once, [64,128] fp8 tiles — ~250 TFLOPS on H20, i.e. at the fp8
    tensor-core rate; the former per-64-token paged K=1 launch was ~18%
    slower), otherwise the paged K=1 kernel. The causal local window
    ``[n_complete, pos]`` is contiguous, so with ``dense_local`` it is scored
    with the dense DeepGEMM logits kernel on the window's keys (``ke`` per row
    gives causality) and concatenated — about 2x cheaper than gathering the
    window token by token.
    """
    import deep_gemm

    from sglang.srt.layers.attention.nsa.hisa.hisa_topk_fused import hisa_topk_candidates_fused

    n_q = q_fp8.shape[0]
    device = q_fp8.device
    leaf_start = leaf_start.to(torch.int32)
    leaf_len = leaf_len.to(torch.int32)
    coarse = coarse_leaf_scores(q_fp8, weights, keys, scales, num_leaves, leaf_len,
                                leaf_start, int(cfg.sink_tokens))
    budget = int(cfg.prefill_candidate_tokens if budget is None else budget)
    n_complete = int(n_complete)
    local_len = int(seq_len) - n_complete  # last row's window; shorter rows mask
    weighted = select_mode() == "weighted" and coarse.shape[1] <= WEIGHTED_SELECT_MAX_LEAVES
    if weighted:
        num_leaves_i32 = num_leaves.reshape(1).to(torch.int32)

        def select(rows, extra_local):
            return select_candidates_weighted(
                coarse, leaf_start, leaf_len, num_leaves_i32, rows, budget, n_complete, extra_local
            )
    else:
        order, incl = rank_leaves(coarse, leaf_start, leaf_len)
        del coarse

        def select(rows, extra_local):
            return expand_candidates(order, incl, leaf_start, leaf_len, rows, budget, n_complete, extra_local)

    raw_pages = raw_pages.view(-1, 64, 1, raw_pages.shape[-1] // 64)
    table = table.to(torch.int32)
    ctx = seq_lens_expanded.to(torch.int32)
    flat = _flat_keys(k_flat, int(seq_len))
    if dense_local and local_len > 0:
        k_loc, s_loc = _local_keys(raw_pages, table, k_flat, n_complete, int(seq_len))
        local_ids = torch.arange(n_complete, n_complete + local_len, dtype=torch.int32, device=device)
        ke_loc = (ctx - n_complete).contiguous()
        ks_loc = torch.zeros_like(ke_loc)
    out = torch.empty((n_q, cfg.index_topk), dtype=torch.int32, device=device)
    step = int(cfg.sparse_prefill_rows)
    for r0 in range(0, n_q, step):
        rows = slice(r0, min(r0 + step, n_q))
        ctx_rows = ctx[rows].contiguous()
        w_rows = weights[rows].contiguous()
        if dense_local and local_len > 0:
            cand_leaf = select(rows, 0)
            R = cand_leaf.shape[0]
            fine_leaf = _fine_scores(q_fp8[rows], w_rows, cand_leaf, ctx_rows, flat, raw_pages, table)
            local = deep_gemm.fp8_mqa_logits(
                q_fp8[rows], (k_loc, s_loc), w_rows, ks_loc[rows].contiguous(), ke_loc[rows].contiguous(),
                clean_logits=False,
            )[:, :local_len]
            if local.shape[1] < local_len:
                # DeepGEMM sizes the output by this sub-batch's max ke; the
                # missing columns are beyond every row's causal end and are
                # excluded by ``count`` below, so pad them with -inf.
                pad = torch.full((R, local_len - local.shape[1]), float("-inf"),
                                 dtype=local.dtype, device=device)
                local = torch.cat((local, pad), dim=1)
            score = torch.cat((fine_leaf, local), dim=1)
            cand = torch.cat((cand_leaf, local_ids.unsqueeze(0).expand(R, -1)), dim=1)
            # valid prefix per row: budget leaf tokens + the causal part of the window
            count = (budget + ke_loc[rows]).contiguous()
        else:
            cand = select(rows, local_len)
            R = cand.shape[0]
            score = _fine_scores(q_fp8[rows], w_rows, cand, ctx_rows, flat, raw_pages, table)
            count = torch.full((R,), cand.shape[1], dtype=torch.int32, device=device)
        out[rows] = hisa_topk_candidates_fused(score, cand, count, ctx_rows, None)
    return out


def _flat_keys(k_flat, seq_len: int):
    """``(fp8 [>=seq_len,128], fp32 [>=seq_len])`` contiguous, or ``None``."""
    if k_flat is None:
        return None
    k, s = k_flat
    if k.shape[0] < seq_len or s.reshape(-1).shape[0] < seq_len:
        return None
    if k.dtype == torch.uint8:
        k = k.view(torch.float8_e4m3fn)
    return k.contiguous(), s.reshape(-1).to(torch.float32).contiguous()


def _fine_scores(q_rows, w_rows, cand, ctx_rows, flat, raw_pages, table) -> torch.Tensor:
    """``[R, C]`` fp32 exact indexer logits of the candidate token ids ``cand``
    (-1 padded); tokens at or beyond each row's causal end (``ctx_rows`` =
    position + 1) and padding score -inf.

    With the flat index-K (``flat``) this is HISA's persistent block-sparse
    kernel with K=1: token ids are the "blocks", gathered straight from the
    flat ``[N,128]`` keys (no page-table indirection, Q loaded once per row,
    the gather of step i+1 overlaps the GEMM of step i). Contiguity of the
    candidates is irrelevant for it: every fp8 key row is one 128-byte line,
    and on H20 the kernel already runs at the fp8 tensor-core rate.
    """
    from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
        block_sparse_mqa_triton,
        sparse_paged_mqa_triton,
    )

    R = cand.shape[0]
    if flat is not None:
        k, s = flat
        ks = torch.zeros_like(ctx_rows)
        return block_sparse_mqa_triton(q_rows.contiguous(), k, s, cand, 1, w_rows, ks, ctx_rows)
    return sparse_paged_mqa_triton(
        q_rows.unsqueeze(1), raw_pages, cand.unsqueeze(1), 1,
        w_rows.unsqueeze(1), ctx_rows, table.expand(R, -1),
    ).squeeze(1)


def _local_keys(raw_pages, table, k_flat, n_complete: int, seq_len: int):
    """Flat ``(fp8 [L_pad,128], fp32 [L_pad])`` keys of ``[n_complete, seq_len)``.

    ``n_complete`` is a multiple of the 64-token page, so the window starts on a
    page boundary; the tail is padded to whole pages (masked by ``ke``).
    """
    if k_flat is not None:
        k, s = k_flat
        return k[n_complete:seq_len].contiguous(), s[n_complete:seq_len].reshape(-1).to(torch.float32).contiguous()
    p0, p1 = n_complete // 64, -(-seq_len // 64)
    pages = raw_pages.view(raw_pages.shape[0], -1)[table[0, p0:p1].long()]  # [P, 64*132]
    keys = pages[:, : 64 * 128].reshape(-1, 128).view(torch.float8_e4m3fn)
    scales = pages[:, 64 * 128 :].contiguous().view(torch.float32).reshape(-1)
    return keys.contiguous(), scales.contiguous()
