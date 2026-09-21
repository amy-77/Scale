"""Adaptive-HISA B=1 decode, including stable buffers for model CUDA graphs.

Graphs are captured before request maps exist. Each eligible layer therefore
captures this pipeline against a permanent workspace. Before replay, the host
checks the request epoch and binds its map to those buffers once. Missing maps,
unsupported batches/modes, short contexts, and insufficient metadata capacity use the
official eager path; a graph containing stale metadata must never be replayed.

Hot-path reductions (same candidate set):
* ``get_paged_mqa_logits_metadata`` once per forward, shared across layers
* reusable candidates / logits / topk buffers on ``DecodeWorkspace``
* skip capacity-wide ``clean_summary_scores`` (selector already masks)
* skip empty seal launches outside CUDA-graph capture
* always fuse Top-2048 → physical page-table indices when available
"""
from __future__ import annotations

import logging
import math
import os

import torch

from .config import get_config, wants_layer
from .decode_select import weighted_select_candidates
from .decode_timing import stage as timing_stage
from .incremental import update_completed_blocks

logger = logging.getLogger(__name__)
_WORKSPACES = {}
_GENERATION = 0
_TRUE = {"1", "true", "yes", "on"}


def _skip_clean_enabled() -> bool:
    # Default on: selector already ignores length==0 / rows beyond num_leaves.
    return os.environ.get("SGLANG_NSA_ADAPTIVE_HISA_SKIP_CLEAN", "1").strip().lower() in _TRUE


def _fuse_page_table_enabled() -> bool:
    # Prefer fused physical indices even when SGLANG_NSA_FUSE_TOPK=0.
    return os.environ.get("SGLANG_NSA_ADAPTIVE_HISA_FUSE_PAGE_TABLE", "1").strip().lower() in _TRUE


def _selector_profile_enabled() -> bool:
    return os.environ.get(
        "SGLANG_NSA_ADAPTIVE_HISA_SELECTOR_PROFILE", "0"
    ).strip().lower() in _TRUE


class DecodeWorkspace:
    def __init__(self, capacity, device, *, candidate_tokens=8192, index_topk=2048):
        self.capacity = math.ceil(capacity / 64) * 64
        self.candidate_tokens = int(candidate_tokens)
        self.index_topk = int(index_topk)
        self.starts = torch.zeros(self.capacity, dtype=torch.int32, device=device)
        self.lengths = torch.zeros_like(self.starts)
        self.selected_prefix = torch.zeros_like(self.starts)
        self.num_leaves = torch.ones((1, 1), dtype=torch.int32, device=device)
        self.complete = torch.zeros((1,), dtype=torch.int32, device=device)
        self.base_complete = torch.zeros_like(self.complete)
        self.base_rows = torch.ones_like(self.complete)
        self.previous_blocks = torch.zeros_like(self.complete)
        self.page_table = torch.arange(self.capacity // 64, dtype=torch.int32, device=device).reshape(1, -1)
        # Same HISA 64-row SoA layout; stable addresses for graph replay.
        # Copy the frozen prefill rows once, then append completed decode blocks.
        self.summary_pages = torch.zeros((self.capacity // 64, 64 * 132), dtype=torch.uint8, device=device)
        self.row_ids = torch.arange(self.capacity, device=device)
        self.zero = torch.zeros((1,), dtype=torch.int32, device=device)
        # Reused per-token outputs (eliminate torch.empty / contiguous each layer).
        self.candidates = torch.full(
            (1, self.candidate_tokens), -1, dtype=torch.int32, device=device
        )
        self.candidate_count = torch.zeros((1,), dtype=torch.int32, device=device)
        self.coarse_logits = torch.empty((1, self.capacity), dtype=torch.float32, device=device)
        self.fine_logits = torch.empty(
            (1, self.candidate_tokens), dtype=torch.float32, device=device
        )
        self.topk_indices = torch.full(
            (1, self.index_topk), -1, dtype=torch.int32, device=device
        )
        # Interval expansion buffers (start, length, output_offset) for segment scorer.
        self.seg_starts = torch.zeros(self.capacity + 2, dtype=torch.int32, device=device)
        self.seg_lengths = torch.zeros_like(self.seg_starts)
        self.seg_offsets = torch.zeros_like(self.seg_starts)
        self.seg_count = torch.zeros((1,), dtype=torch.int32, device=device)
        self.binding = None
        self.n_tokens = self.n_complete = self.max_leaf_len = 0
        self.valid = False
        self.graph_ready = False
        self.n_base_leaves = 0
        self._host_previous_blocks = 0
        self._entry = self._source_pages = None
        self._summaries_bound = False

    def bind(self, entry, source_pages=None):
        key = (entry.req_idx, entry.epoch, id(entry))
        if self.binding == key:
            return
        self.valid = False
        if entry.capacity > self.capacity:
            return
        if entry.ready_event is not None:
            torch.cuda.current_stream(self.starts.device).wait_event(entry.ready_event)
        # These host reads happen once per binding, never inside graph replay.
        part = entry.partition
        count = int(part.num_leaves.item())
        if not bool(part.status_ok.item()) or not 0 < count <= entry.capacity:
            return
        self.max_leaf_len = int(part.leaf_len[:count].max().item())
        self.starts.zero_()
        self.lengths.zero_()
        self.starts[:entry.capacity].copy_(part.leaf_start)
        self.lengths[:entry.capacity].copy_(part.leaf_len)
        self.num_leaves.fill_(count)
        self.complete.fill_(entry.n_complete)
        self.base_complete.fill_(entry.n_complete)
        self.base_rows.fill_(count)
        self.previous_blocks.zero_()
        self._host_previous_blocks = 0
        self.n_base_leaves = count
        self.n_tokens, self.n_complete = entry.n_tokens, entry.n_complete
        self._entry = entry
        self._summaries_bound = False
        self.bind_summaries(source_pages if source_pages is not None else self._source_pages)
        self.binding = key
        self.valid = True

    def bind_summaries(self, source_pages):
        if self._summaries_bound or self._entry is None or source_pages is None:
            return
        self._source_pages = source_pages
        self.summary_pages.zero_()
        pages = source_pages[self._entry.page_table.long()]
        self.summary_pages[:pages.shape[0]].copy_(pages)
        self._summaries_bound = True

    def eligible(self, seq_len, cfg):
        # The candidate budget is filled exactly (sink + clipped leaves with a
        # partial crossing leaf + raw tail), so it covers topk whenever the
        # context exceeds the budget. Sealed decode chunks must fit the rows.
        return (
            self.valid
            and seq_len >= self.n_tokens
            and seq_len > cfg.candidate_tokens
            and self.n_base_leaves + (seq_len - self.n_complete) // cfg.decode_chunk <= self.capacity
        )


def reset_decode_state():
    global _GENERATION
    _WORKSPACES.clear()
    _GENERATION += 1


def release_decode_request(req_idx):
    global _GENERATION
    _GENERATION += 1
    if _selector_profile_enabled():
        from .decode_select import selector_profile_snapshot
        from .decode_timing import reset as reset_timing
        from .decode_timing import snapshot as timing_snapshot

        profile = selector_profile_snapshot(reset=True)
        if profile:
            logger.info(
                "adaptive-hisa selector profile req=%d %s",
                req_idx,
                profile,
            )
        timing = timing_snapshot()
        if timing:
            logger.info(
                "adaptive-hisa stage timing req=%d %s",
                req_idx,
                timing,
            )
            reset_timing()
    for work in _WORKSPACES.values():
        if work.binding is not None and work.binding[0] == req_idx:
            work.valid = False
            work.binding = None


def _uses_layer(layer_id, cfg):
    return layer_id >= cfg.fallback_layers and wants_layer(layer_id)


def _supported_batch(batch):
    if batch.batch_size != 1 or not batch.forward_mode.is_decode():
        return False
    if getattr(batch, 'attn_cp_metadata', None) is not None:
        return False
    spec = getattr(batch, 'spec_algorithm', None)
    if spec is not None and not spec.is_none():
        return False
    if batch.token_to_kv_pool.page_size != 64:
        return False
    from sglang.srt.server_args import get_global_server_args

    args = get_global_server_args()
    return (int(getattr(args, 'pp_size', 1)) == 1
            and not getattr(args, 'enable_nsa_prefill_context_parallel', False)
            and not getattr(args, 'enable_dp_attention', False))


def _workspace(batch, layer_id):
    cfg = get_config()
    pool = get_summary_pool_safe(batch, cfg)
    key = (id(pool), layer_id)
    if key not in _WORKSPACES:
        context = batch.req_to_token_pool.max_context_len
        # Prefill leaves ≈ N/merge_divisor; sealed decode chunks add G/decode_chunk.
        leaf_div = cfg.merge_target_divisor or cfg.summary_compression
        capacity = -(-context // min(leaf_div, cfg.decode_chunk))
        _WORKSPACES[key] = DecodeWorkspace(
            max(capacity, 64),
            pool.device,
            candidate_tokens=cfg.candidate_tokens,
            index_topk=cfg.index_topk,
        )
    return pool, _WORKSPACES[key]


def get_summary_pool_safe(batch, cfg):
    from .summary_pool import get_summary_pool

    return get_summary_pool(batch.token_to_kv_pool, cfg.summary_compression)


def _shared_schedule(batch, num_leaves, capacity, device):
    """One DeepGEMM schedule per forward; reused by every Adaptive layer."""
    import deep_gemm

    key = ("_adaptive_hisa_schedule", int(capacity), id(device))
    cached = getattr(batch, "_adaptive_hisa_mqa_schedule", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    schedule = deep_gemm.get_paged_mqa_logits_metadata(num_leaves, 64, sms)
    batch._adaptive_hisa_mqa_schedule = (key, schedule)
    return schedule


def prepare_decode(batch):
    """Bind maps once per request; cache this forward's eligibility on the batch."""
    cfg = get_config()
    if cfg.mode != 'adaptive_decode' or not _supported_batch(batch):
        return {}
    if batch.seq_lens_cpu is None or len(batch.seq_lens_cpu) != 1:
        return {}
    slots = getattr(batch, 'req_pool_indices_cpu', None)
    if slots is None or len(slots) != 1 or slots[0] is None:
        return {}
    req_idx = int(slots[0])
    seq_len = int(batch.seq_lens_cpu[0])
    key = (seq_len, req_idx, _GENERATION)
    if getattr(batch, '_adaptive_decode_prepared_key', None) == key:
        return batch._adaptive_decode_prepared
    result = {}
    batch._adaptive_decode_prepared_key = key
    batch._adaptive_decode_prepared = result
    # Drop stale schedule when the sequence advances.
    batch._adaptive_hisa_mqa_schedule = None
    if seq_len <= cfg.candidate_tokens:
        return result
    from .prefill_runtime import get_summary_entry

    count = len(batch.token_to_kv_pool.index_k_with_scale_buffer)
    newly_bound = 0
    for layer_id in range(count):
        if not _uses_layer(layer_id, cfg):
            continue
        entry = get_summary_entry(layer_id, req_idx)
        if entry is None:
            continue
        pool, work = _workspace(batch, layer_id)
        previous = work.binding
        work.bind(entry, pool.bufs[layer_id])
        newly_bound += int(work.binding != previous)
        if work.eligible(seq_len, cfg):
            result[layer_id] = (pool, work)
    if newly_bound:
        logger.info('adaptive-hisa decode bound req=%d layers=%d eligible=%d candidate_tokens=%d '
                    'sink=%d tail=%d decode_chunk=%d fallback_layers=%d',
                    req_idx, newly_bound, len(result), cfg.candidate_tokens,
                    cfg.sink_tokens, cfg.tail_tokens, cfg.decode_chunk, cfg.fallback_layers)
    return result


def allow_cuda_graph(batch):
    """Called outside capture, before any model graph containing this path runs."""
    cfg = get_config()
    if cfg.mode != 'adaptive_decode':
        return True
    if not _supported_batch(batch):
        return False
    prepared = prepare_decode(batch)
    required = [i for i in range(len(batch.token_to_kv_pool.index_k_with_scale_buffer))
                if _uses_layer(i, cfg)]
    return bool(required) and all(i in prepared and prepared[i][1].graph_ready for i in required)


def _maybe_seal(work, raw_pages, page_table, seq_len, chunk, capturing: bool):
    """Seal only on chunk boundaries; skip empty 4-CTA launches outside capture."""
    if not capturing:
        # Host-known seq length when prepare_decode ran; fall back to tensor path.
        seq_host = getattr(work, "_pending_seq_host", None)
        if seq_host is not None:
            completed = max(int(seq_host) - work.n_complete, 0) // int(chunk)
            if completed == work._host_previous_blocks:
                return
            work._host_previous_blocks = completed
    with timing_stage("seal", work.starts.device):
        update_completed_blocks(work, raw_pages, page_table, seq_len, chunk)


def select_topk(work, summary_pages, raw_pages, page_table, seq_len, q, weights, cfg,
                *, return_debug=False, output_page_table=None, schedule=None,
                capturing=False, use_segments=False):
    """No host reads; all tensor shapes fixed, suitable for model graph capture."""
    import deep_gemm
    from sglang.srt.layers.attention.nsa.hisa.hisa_topk_fused import hisa_topk_candidates_fused

    work.bind_summaries(summary_pages)
    _maybe_seal(work, raw_pages, page_table, seq_len, cfg.decode_chunk, capturing)

    with timing_stage("schedule", q.device):
        if schedule is None:
            sms = torch.cuda.get_device_properties(q.device).multi_processor_count
            schedule = deep_gemm.get_paged_mqa_logits_metadata(work.num_leaves, 64, sms)

    with timing_stage("coarse_mqa", q.device):
        coarse = deep_gemm.fp8_paged_mqa_logits(
            q.unsqueeze(1), work.summary_pages.view(-1, 64, 1, 132), weights,
            work.num_leaves, work.page_table, schedule, work.capacity, clean_logits=False)
        # Write into the stable workspace view used by the selector.
        work.coarse_logits.copy_(coarse[:, : work.capacity])
        coarse_view = work.coarse_logits
        if not _skip_clean_enabled():
            from .incremental import clean_summary_scores

            coarse_view = clean_summary_scores(coarse_view, work)

    with timing_stage("weighted_select", q.device):
        profile_selector = (
            _selector_profile_enabled() and not capturing and not use_segments
        )
        if profile_selector:
            from .decode_select import (
                expand_selected_candidates,
                weighted_prefix_profile,
            )

            with timing_stage("weighted_prefix", q.device):
                weighted_prefix_profile(
                    coarse_view[0],
                    work.lengths,
                    work.starts,
                    work.num_leaves,
                    seq_len,
                    work.selected_prefix,
                    cfg.candidate_tokens,
                    cfg.sink_tokens,
                    cfg.tail_tokens,
                )
            with timing_stage("weighted_expand", q.device):
                candidates, count = expand_selected_candidates(
                    work.selected_prefix,
                    work.starts,
                    work.lengths,
                    work.num_leaves,
                    seq_len,
                    cfg.candidate_tokens,
                    cfg.sink_tokens,
                    cfg.tail_tokens,
                    candidates_out=work.candidates,
                    count_out=work.candidate_count,
                )
            segments = seg_count = None
        elif use_segments:
            from .decode_select import weighted_select_intervals

            segments, seg_count, candidates, count = weighted_select_intervals(
                coarse_view[0],
                work.lengths,
                work.starts,
                work.num_leaves,
                seq_len,
                work.selected_prefix,
                cfg.candidate_tokens,
                cfg.sink_tokens,
                cfg.tail_tokens,
                candidates_out=work.candidates,
                count_out=work.candidate_count,
                seg_starts_out=work.seg_starts,
                seg_lengths_out=work.seg_lengths,
                seg_offsets_out=work.seg_offsets,
                seg_count_out=work.seg_count,
            )
        else:
            candidates, count = weighted_select_candidates(
                coarse_view[0],
                work.lengths,
                work.starts,
                work.num_leaves,
                seq_len,
                work.selected_prefix,
                cfg.candidate_tokens,
                cfg.sink_tokens,
                cfg.tail_tokens,
                candidates_out=work.candidates,
                count_out=work.candidate_count,
            )
            segments = seg_count = None

    with timing_stage("fine_rescore", q.device):
        if use_segments and segments is not None:
            from .segment_scorer import sparse_paged_mqa_segments

            logits = sparse_paged_mqa_segments(
                q.unsqueeze(1),
                raw_pages.view(-1, 64, 1, 132),
                work.seg_starts,
                work.seg_lengths,
                work.seg_offsets,
                seg_count,
                weights,
                seq_len.reshape(-1),
                page_table,
                cfg.candidate_tokens,
                logits_out=work.fine_logits,
            )
        else:
            from sglang.srt.layers.attention.nsa.hisa.triton_kernels import sparse_paged_mqa_triton

            # K=1 interprets each HISA "block ID" as one candidate token ID.
            scored = sparse_paged_mqa_triton(
                q.unsqueeze(1), raw_pages.view(-1, 64, 1, 132), candidates.unsqueeze(1),
                1, weights, seq_len.reshape(-1), page_table).squeeze(1)
            work.fine_logits.copy_(scored)
            logits = work.fine_logits

    with timing_stage("topk", q.device):
        result = hisa_topk_candidates_fused(
            logits, candidates, count, seq_len, output_page_table)
        # Keep a workspace mirror for diagnostics; return the fresh tensor so
        # later layers cannot overwrite indices still held by attention.
        work.topk_indices.copy_(result[:1])

    if return_debug:
        # Keep the historical 5-tuple contract used by unit tests.
        return result, candidates, count, coarse_view, logits
    return result


def maybe_decode_topk(batch, layer_id, q, weights, metadata, index_topk):
    cfg = get_config()
    if cfg.mode != 'adaptive_decode' or not _uses_layer(layer_id, cfg):
        return None
    if not _supported_batch(batch) or q.shape[0] < 1 or index_topk != cfg.index_topk:
        return None
    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

    capturing = bool(get_is_capture_mode())
    if capturing:
        # Warmup/capture uses a safe dummy workspace. Host replay admission
        # above guarantees that real request metadata is bound before replay.
        pool, work = _workspace(batch, layer_id)
        if not work.graph_ready and layer_id == cfg.fallback_layers:
            logger.info('adaptive-hisa capturing B=1 decode pipeline capacity=%d query_rows=%d',
                        work.capacity, q.shape[0])
        work.graph_ready = True
        seq_host = None
    else:
        prepared = prepare_decode(batch)
        if layer_id not in prepared:
            return None
        pool, work = prepared[layer_id]
        seq_host = int(batch.seq_lens_cpu[0]) if batch.seq_lens_cpu is not None else None
    work._pending_seq_host = seq_host

    page_table_64 = metadata.get_page_table_64()
    seq_len = metadata.get_seqlens_int32()
    schedule = None if capturing else _shared_schedule(
        batch, work.num_leaves, work.capacity, q.device
    )
    from sglang.srt.environ import envs

    # Fuse Top-2048 → physical page indices inside hisa_topk when requested.
    # Mark the batch so nsa_backend skips the separate 2048-entry gather.
    fuse_pt = (
        _fuse_page_table_enabled()
        or (envs.SGLANG_NSA_FUSE_TOPK.get()
            and not getattr(metadata, "force_unfused_topk", False))
    )
    if getattr(metadata, "force_unfused_topk", False):
        fuse_pt = False
    output_page_table = metadata.get_page_table_1() if fuse_pt else None
    batch._adaptive_hisa_fused_topk = bool(fuse_pt)
    use_segments = os.environ.get(
        "SGLANG_NSA_ADAPTIVE_HISA_SEGMENT_SCORER", "0"
    ).strip().lower() in _TRUE

    result = select_topk(
        work, pool.bufs[layer_id], batch.token_to_kv_pool.get_index_k_with_scale_buffer(layer_id),
        page_table_64, seq_len, q[:1],
        weights[:1].reshape(1, -1).contiguous(), cfg,
        output_page_table=output_page_table,
        schedule=schedule,
        capturing=capturing,
        use_segments=use_segments,
    )
    if q.shape[0] > 1:
        result = torch.cat((result, torch.full((q.shape[0]-1, cfg.index_topk), -1,
                                               device=q.device, dtype=torch.int32)))
    return result
