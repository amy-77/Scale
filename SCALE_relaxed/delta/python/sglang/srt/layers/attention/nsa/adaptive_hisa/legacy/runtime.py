"""LEGACY radius/CPU-rerank decode prototype. Not wired into ``nsa_indexer``.

Kept only as a historical reference: it seals roots by P-radius, loops over
requests in Python and reranks candidates on the CPU. The Phase B contract
(P-key split on GPU, sync target merge, FP8 summaries) lives in
``adaptive_hisa.partition_gpu`` / ``summary_pool``; decode consumption is
Phase C and will not reuse this module.

Decode is the primary path. Prefill is opt-in (``adaptive_hisa_prefill``) and
keeps the official indexer for rows whose visible prefix is still below
``MIN_SEQ_LEN``; those rows are scored with the same FP8 kernel, but only the
short rows are passed in. Rows at or beyond the threshold share one sealed
partition per request and do not see leaves that end after their own ``ke``.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa.config import (
    CANDIDATE_TOKENS,
    HEAD_DIM,
    INDEX_TOPK,
    MIN_SEQ_LEN,
    PAGE_SIZE,
    ROOT,
    enabled,
    wants_layer,
)


def policy_name() -> str:
    """Legacy radius/key/fixed policy switch; only this prototype reads it."""
    import os

    return os.environ.get("SGLANG_NSA_ADAPTIVE_HISA_LEGACY_POLICY", "radius").strip().lower() or "radius"


def prefill_enabled() -> bool:
    """Legacy prefill selector switch. Never true in the Phase B contract."""
    import os

    return os.environ.get("SGLANG_NSA_ADAPTIVE_HISA_LEGACY_PREFILL", "0").strip().lower() in {"1", "true", "on"}
from sglang.srt.layers.attention.nsa.adaptive_hisa.legacy.cuda_select import gather_compact, score_leaves
from sglang.srt.layers.attention.nsa.adaptive_hisa.legacy.reference import (
    canonical_scores,
    rerank,
    select_candidates,
)
from sglang.srt.layers.attention.nsa.adaptive_hisa.legacy.sidecar import SLOTS, on_req_free
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

__all__ = ["maybe_topk", "on_req_free", "note_stored_keys"]


def note_stored_keys(pool, layer_id: int, key: torch.Tensor, loc: torch.Tensor) -> None:
    if not enabled() or key.numel() == 0:
        return
    from sglang.srt.layers.attention.nsa.adaptive_hisa.legacy.sidecar import PageAtoms, update_atom_stats

    slot = layer_id - pool.start_layer
    atoms = getattr(pool, "_adaptive_atoms", None)
    if atoms is None or len(atoms) != pool.layer_num:
        pages = pool.index_k_with_scale_buffer[0].shape[0]
        atoms = [PageAtoms.empty(pages, pool.device) for _ in range(pool.layer_num)]
        pool._adaptive_atoms = atoms
    update_atom_stats(atoms[slot], key, loc, page_size=pool.page_size)


def _handles(forward_batch, layer_id: int, *, prefill: bool) -> bool:
    if not enabled() or not wants_layer(layer_id) or get_is_capture_mode():
        return False
    mode = forward_batch.forward_mode
    if mode.is_decode_or_idle():
        return True
    return prefill and prefill_enabled() and mode.is_extend_without_speculative()


def maybe_topk(indexer, forward_batch, layer_id: int, q_fp8: torch.Tensor, weights: torch.Tensor, metadata):
    if not _handles(forward_batch, layer_id, prefill=True):
        return None
    if forward_batch.forward_mode.is_decode_or_idle():
        return _decode(indexer, forward_batch, layer_id, q_fp8, weights, metadata)
    return _prefill(indexer, forward_batch, layer_id, q_fp8, weights, metadata)


def _weights2d(weights: torch.Tensor) -> torch.Tensor:
    if weights.dim() == 3:
        weights = weights.squeeze(-1)
    return weights


def _page_table(metadata):
    table = metadata.get_page_table_64()
    return table


def _dequant_range(pool, layer_id: int, page_table: torch.Tensor, start: int, end: int) -> torch.Tensor:
    ids = torch.arange(start, end, device=page_table.device, dtype=torch.long)
    buf = pool.get_index_k_with_scale_buffer(layer_id)
    keys_u8, scale = gather_compact(buf, ids, page_table, pool.page_size, pool.index_head_dim)
    if keys_u8.device != page_table.device:
        keys_u8 = keys_u8.to(page_table.device)
        scale = scale.to(page_table.device)
    return keys_u8.view(fp8_dtype).to(torch.float32) * scale.to(torch.float32)[:, None]


def _seal_to(pool, layer_id: int, req_idx: int, page_table: torch.Tensor, seq_len: int, policy: str):
    state = SLOTS.state(layer_id, req_idx)
    while state.sealed_end + ROOT <= seq_len:
        root = _dequant_range(pool, layer_id, page_table, state.sealed_end, state.sealed_end + ROOT)
        state.append_root(root, policy=policy, offset=state.sealed_end)
    return state


def _one_query(state, query_fp8, weight, ke: int, page_table, pool, layer_id: int, topk: int) -> torch.Tensor:
    q = query_fp8.to(torch.float32)
    cpu_w = weight.detach().to(torch.float32).cpu()
    gpu_scores = None
    if q.device.type == "cuda" and state.mean.numel():
        gpu_scores = score_leaves(
            q.unsqueeze(0), cpu_w.unsqueeze(0).to(q.device), state.mean.to(q.device)
        )[0].detach().cpu()
    candidates = select_candidates(
        state,
        ke,
        budget=min(CANDIDATE_TOKENS, (max(ke, 1) // 4) * 4 or 4),
        query=q.detach().cpu(),
        weights=cpu_w,
        scores=gpu_scores,
    )
    buf = pool.get_index_k_with_scale_buffer(layer_id)
    k_u8, scale = gather_compact(
        buf,
        candidates.to(page_table.device),
        page_table,
        pool.page_size,
        pool.index_head_dim,
    )
    k_fp8 = k_u8.view(fp8_dtype)
    scores = canonical_scores(
        query_fp8.detach().cpu(),
        weight.detach().to(torch.float32).cpu(),
        k_fp8.detach().cpu(),
        scale.detach().to(torch.float32).cpu(),
        candidates.cpu(),
    )
    return rerank(scores, candidates.cpu(), topk)


def _decode(indexer, forward_batch, layer_id, q_fp8, weights, metadata) -> torch.Tensor:
    weights = _weights2d(weights)
    seqlens = metadata.get_seqlens_int32()
    table = _page_table(metadata)
    reqs = forward_batch.req_pool_indices
    bsz = seqlens.shape[0]
    topk = indexer.index_topk
    out = torch.full((q_fp8.shape[0], topk), -1, dtype=torch.int32, device=q_fp8.device)
    policy = policy_name()
    pool = forward_batch.token_to_kv_pool
    for b in range(bsz):
        seq = int(seqlens[b].item())
        if seq <= topk:
            out[b, :seq] = torch.arange(seq, device=out.device)
            continue
        state = _seal_to(pool, layer_id, int(reqs[b].item()), table[b], seq, policy)
        picked = _one_query(state, q_fp8[b], weights[b], seq, table[b], pool, layer_id, topk)
        out[b] = picked.to(device=out.device, dtype=torch.int32)
    return out


def _prefill(indexer, forward_batch, layer_id, q_fp8, weights, metadata) -> Optional[torch.Tensor]:
    if forward_batch.batch_size not in (None, 1) and len(metadata.get_page_table_64()) != 1:
        return None
    weights = _weights2d(weights)
    row_ke = metadata.get_seqlens_expanded()
    n = row_ke.shape[0]
    topk = indexer.index_topk
    long = row_ke >= MIN_SEQ_LEN
    if not bool(long.any()):
        return None
    out = torch.full((q_fp8.shape[0], topk), -1, dtype=torch.int32, device=q_fp8.device)
    short = ~long
    short = short[:n]
    if bool(short.any()):
        official = indexer._get_topk_ragged(False, forward_batch, layer_id, q_fp8, weights, metadata)
        out[:n] = official[:n]
    table = _page_table(metadata)[0]
    req = int(forward_batch.req_pool_indices[0].item())
    pool = forward_batch.token_to_kv_pool
    max_ke = int(row_ke[long].max().item()) if bool(long.any()) else 0
    state = _seal_to(pool, layer_id, req, table, max_ke, policy_name())
    for row in torch.nonzero(long).flatten().tolist():
        ke = int(row_ke[row].item())
        if ke <= topk:
            out[row, :ke] = torch.arange(ke, device=out.device)
            continue
        picked = _one_query(state, q_fp8[row], weights[row], ke, table, pool, layer_id, topk)
        out[row] = picked.to(device=out.device, dtype=torch.int32)
    return out
