"""逐 Indexer head 的 chunk16-quota 分页检索（DSA per-head index 契约）。

每个 Indexer head ``h`` 各自选出一套 token indices；MLA 侧消费
``[nq, G, topk]``，固定 1:2 映射为 Indexer head ``h`` -> MLA heads
``2h, 2h+1``。在 attention-TP 下，本 rank 只计算本地 MLA heads 对应的
``G = 64 / tp`` 个 Indexer heads。

选择流程（chunk = 16 tokens，每个 64-token page 含 4 个 chunks）：

  stage 0  粗分：对本 request 的所有 chunks 做 ``q_h · chunk_sum``
           （cuBLAS bf16 GEMM）。``chunk_sum`` 在写入 K 时增量维护
           （``update_chunk_sum``）；完整 chunk 的 sum 排序与 mean 等价，
           唯一的半满 chunk（tail）本身会被强制保留。真实 trace 上
           mean 优于 min+max extrema（见 exp_scheme_cmp.py）。
  stage 1  （融合 CUDA kernel）在 SMEM 中对 score 行做精确 radix
           Top-SEL(128) chunk 选择；sink 与 tail chunk 强制保留。
  stage 2  （同一 kernel）经 block table 对选中 chunks 做精确 fp8 QK。
           最好的 NDENSE(52) 个 chunks 各保留 8 个 tokens，其余 76 个各
           保留 4 个：52*8 + 76*4 = 720 indices / head，不足处用 -1 填充；
           sink 与当前 token 强制保留。中间分数不回写 HBM。

上述 boundary guarantee 适用于全量和 fraction-pruned 路径。实验性的
``prune_topk + mean_ranked_chunks`` 路径严格使用 mean 排名，不额外插入或
提升 sink/tail；这两种 contract 不应混用。

当 ``len <= 720`` 时，该行保留全部可见 tokens（预算已覆盖），行为与
dense indexer 一致。kernel 见 ``csrc/chunk16_quota_paged.cu``；contiguous K
的独立对照实现见 ``code/headwise_minmax/``。

通过 ``SGLANG_NSA_PER_HEAD_INDEX=1`` 开启（需要 ``--disable-cuda-graph``）。
``SGLANG_NSA_PER_HEAD_Q_BLOCK`` 限制每次 launch 的粗分 buffer 规模，单位为
(row, head) pairs。
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import NamedTuple, Optional, Sequence, Tuple

import torch

PAGE = 64
CHUNK = 16
CHUNKS_PER_PAGE = PAGE // CHUNK
DIM = 128


class MISASelection(NamedTuple):
    candidate_head_ids: torch.Tensor
    coarse_block_ids: torch.Tensor
    candidate_importance: torch.Tensor
    candidate_context_summary: torch.Tensor
    candidate_context_stats: torch.Tensor

# selection scheme constants; must match csrc/chunk16_quota_paged.cu
SEL = 128
NDENSE = 52
QDENSE = 8
QSPARSE = 4
OUTK = NDENSE * QDENSE + (SEL - NDENSE) * QSPARSE  # 720
MISA_DEFAULT_POOLING_BLOCK = 256
MISA_DEFAULT_PRUNE_TOPK = 8
MISA_DEFAULT_PRUNE_KEEP_FRACTION = 0.60

# FlashMLA's sparse FP8 decode kernel asserts ``topk % TOPK_BLOCK_SIZE == 0``
# (sm90/decode/sparse_fp8/splitkv_mla.cuh).  The width is otherwise free: 4096
# and 8192 are accepted, while a bare OUTK=720 is not.  Every per-head index
# table therefore carries a width rounded up to this block.
FLASHMLA_TOPK_BLOCK = 64


def pad_index_width(width: int) -> int:
    """Round a per-head index width up to what the attention kernel accepts."""
    block = FLASHMLA_TOPK_BLOCK
    return max(block, -(-width // block) * block)

_FALSE = {"", "0", "false", "no", "off"}


def per_head_index_enabled() -> bool:
    return os.environ.get("SGLANG_NSA_PER_HEAD_INDEX", "0").lower() not in _FALSE


def per_head_pairs_per_launch() -> int:
    # (row, head) pairs per kernel launch; the coarse-score buffer is
    # pairs * C * 2 bytes (~270 MB at 128K context with the default).
    return int(os.environ.get("SGLANG_NSA_PER_HEAD_Q_BLOCK", "16384"))


def per_head_decode_start_token() -> int:
    """First generated token that may use per-head/router indices.

    The count excludes prompt tokens.  A value of zero preserves the original
    behavior.  This is particularly useful for reasoning evaluations such as
    AIME, where the early chain of thought should retain the official shared
    DSA selector before switching to the experimental sparse policy.
    """
    raw = os.environ.get(
        "SGLANG_NSA_EXPERIMENTAL_DECODE_START_TOKEN",
        os.environ.get("SGLANG_NSA_PER_HEAD_DECODE_START_TOKEN", "0"),
    )
    value = int(raw)
    if value < 0:
        raise ValueError(
            "SGLANG_NSA_EXPERIMENTAL_DECODE_START_TOKEN must be >= 0"
        )
    return value


def per_head_min_seq_len() -> int:
    """Minimum visible context length for experimental sparse retrieval."""
    value = int(os.environ.get("SGLANG_NSA_EXPERIMENTAL_MIN_SEQ_LEN", "4096"))
    if value < OUTK:
        raise ValueError(
            "SGLANG_NSA_EXPERIMENTAL_MIN_SEQ_LEN must be at least "
            f"OUTK={OUTK}, got {value}"
        )
    return value


def prefill_per_head_enabled() -> bool:
    """Whether prefill uses the per-head contract instead of shared Top-K.

    Prefill and decode run the same policy by default.  Setting this to 0 keeps
    the official shared selector during prefill while decode stays per-head,
    which is the decode-only configuration earlier results were measured under
    and the only way to attribute a change to prefill alone.
    """
    return (
        os.environ.get("SGLANG_NSA_EXPERIMENTAL_PREFILL_PER_HEAD", "1").lower()
        not in _FALSE
    )


def trace_prefill_enabled() -> bool:
    """Log one line per prefill chunk at layer 0 to audit the per-head path."""
    return (
        os.environ.get("SGLANG_NSA_EXPERIMENTAL_TRACE_PREFILL", "0").lower()
        not in _FALSE
    )


def local_head_range(n_heads: int) -> Tuple[int, int]:
    """Indexer heads whose MLA heads (1:2) live on this attention-TP rank."""
    from sglang.srt.layers.dp_attention import (
        get_attention_tp_rank,
        get_attention_tp_size,
    )

    tp = get_attention_tp_size()
    rank = get_attention_tp_rank()
    assert n_heads % tp == 0, f"{n_heads} indexer heads not divisible by tp={tp}"
    g = n_heads // tp
    return rank * g, (rank + 1) * g


# --------------------------------------------------------------------------
# per-chunk K sums, maintained at K-store time
# --------------------------------------------------------------------------

@torch.no_grad()
def update_chunk_sum(
    chunk_sum: torch.Tensor,  # [num_pages * 4, D] f32
    key: torch.Tensor,  # [n, D] bf16 (pre-quantisation K of the new tokens)
    loc: torch.Tensor,  # [n] token slots in the paged cache
) -> None:
    """Fold the new tokens into their chunks' K sum.

    A token at slot 0 of a chunk is the first write into that chunk for the
    owning request (pages are request-exclusive and prefix hits are
    page-aligned, hence chunk-aligned), so the stale sum is reset first;
    later tokens of the same chunk (chunked prefill, decode) accumulate.
    """
    if key.shape[0] == 0:
        return
    loc = loc.to(torch.int64)
    chunk = loc // CHUNK
    fresh = chunk[(loc % CHUNK) == 0]
    if fresh.numel():
        chunk_sum[fresh] = 0.0
    idx = chunk.view(-1, 1).expand(-1, key.shape[1])
    chunk_sum.scatter_add_(0, idx, key.to(chunk_sum.dtype))


# --------------------------------------------------------------------------
# fused kernel (JIT, cached in $TORCH_EXTENSIONS_DIR after first compile)
# --------------------------------------------------------------------------

_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        csrc = Path(__file__).resolve().parent / "csrc"
        _ext = load(
            name="chunk16_quota_paged_ext",
            sources=[str(csrc / "chunk16_quota_paged.cu")],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=bool(int(os.environ.get("SGLANG_NSA_PER_HEAD_VERBOSE", "0"))),
        )
    return _ext


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def _chunk_ids(bt_rows: torch.Tensor, n_chunks: int) -> torch.Tensor:
    """Physical chunk ids [..., n_chunks] from block-table rows [..., W].

    ``n_chunks`` is padded to a multiple of 8 for the kernel and may cover a
    page or two past the table width; those trailing ids alias page 0 and are
    masked by the kernel's visible range.
    """
    n_pages = (n_chunks + CHUNKS_PER_PAGE - 1) // CHUNKS_PER_PAGE
    w = bt_rows.shape[-1]
    if w < n_pages:
        pad = bt_rows.new_zeros(*bt_rows.shape[:-1], n_pages - w)
        bt_rows = torch.cat([bt_rows, pad], dim=-1)
    else:
        bt_rows = bt_rows[..., :n_pages]
    off = torch.arange(CHUNKS_PER_PAGE, device=bt_rows.device, dtype=torch.int64)
    ids = bt_rows.to(torch.int64).clamp(min=0).unsqueeze(-1) * CHUNKS_PER_PAGE + off
    return ids.reshape(*bt_rows.shape[:-1], -1)[..., :n_chunks]


def _pad8(n: int) -> int:
    return (n + 7) // 8 * 8


@torch.no_grad()
def misa_topk_heads_paged(
    q: torch.Tensor,  # [nq, H, D] quantized query; q scale is folded into weights
    weights: torch.Tensor,  # [nq, H] or [nq, H, 1]
    chunk_sum: torch.Tensor,  # [P*4, D] f32 sums for physical 16-token chunks
    block_tables: torch.Tensor,  # [B, max_pages] i32 physical page ids
    row_batch: torch.Tensor,  # [nq] request index per row
    row_len: torch.Tensor,  # [nq] visible tokens per row
    topk_heads: int,
    *,
    pooling_block_size: int = MISA_DEFAULT_POOLING_BLOCK,
    prune_keep_fraction: Optional[float] = None,
    prune_topk: Optional[int] = None,
    context_temperature: float = 1.0,
    return_metadata: bool = False,
    rows_per_launch: Optional[int] = None,
    importance_variant_output: Optional[dict[str, torch.Tensor]] = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | MISASelection:
    """Select query-dependent Indexer heads with the MISA pooled-key router.

    The cache already maintains exact sums for physical 16-token chunks.  This
    function gathers the logical chunks through the request block table, folds
    them into ``pooling_block_size`` token means, and evaluates

    ``w_h * sum_b relu(q_h @ mean_k_b)``.

    ``prune_topk=8, pooling_block_size=256`` returns the selected heads' eight
    highest mean-affinity regions (negative affinities remain ordered). Stage B
    expands these into 128 chunk16 candidates; it must use mean_ranked_chunks.
    In this fixed-count policy, head importance is the signed gate times the sum
    of those same top coarse affinities after ReLU. Explicit
    ``prune_keep_fraction`` and unpruned calls retain the legacy all-block sum.
    Fraction pruning also retains its legacy boundary-pinned region selection.

    Only the relative ordering across heads matters, so the paper's common
    ``1 / num_blocks`` factor is omitted.  The query quantization scale is
    already folded into ``weights`` by the DSA Indexer path.
    """
    if q.ndim != 3 or q.shape[-1] != DIM:
        raise ValueError(f"q must have shape [nq, H, {DIM}], got {tuple(q.shape)}")
    if weights.ndim == 3 and weights.shape[-1] == 1:
        weights = weights.squeeze(-1)
    if weights.ndim != 2 or weights.shape != q.shape[:2]:
        raise ValueError(
            f"weights must have shape {tuple(q.shape[:2])}, got {tuple(weights.shape)}"
        )
    if row_batch.ndim != 1 or row_len.ndim != 1 or row_batch.shape != row_len.shape:
        raise ValueError("row_batch and row_len must be one-dimensional and aligned")
    if q.shape[0] != row_len.shape[0]:
        raise ValueError(f"q rows {q.shape[0]} != row_len rows {row_len.shape[0]}")
    if pooling_block_size < CHUNK or pooling_block_size % CHUNK:
        raise ValueError(
            f"pooling_block_size must be a multiple of {CHUNK}, got {pooling_block_size}"
        )
    if topk_heads <= 0 or topk_heads > q.shape[1]:
        raise ValueError(f"topk_heads must be in [1, {q.shape[1]}], got {topk_heads}")
    if prune_keep_fraction is not None and not 0 < prune_keep_fraction <= 1:
        raise ValueError(
            f"prune_keep_fraction must be in (0, 1], got {prune_keep_fraction}"
        )
    if prune_topk is not None:
        if not isinstance(prune_topk, int) or isinstance(prune_topk, bool) or prune_topk <= 0:
            raise ValueError("prune_topk must be a positive integer")
        if prune_keep_fraction is not None:
            raise ValueError("prune_topk and prune_keep_fraction are mutually exclusive")
    pruning = prune_topk is not None or prune_keep_fraction is not None
    if context_temperature <= 0:
        raise ValueError("context_temperature must be positive")

    nq = q.shape[0]
    if nq == 0:
        if importance_variant_output is not None:
            importance_variant_output.clear()
        empty_heads = torch.empty(
            (0, topk_heads), dtype=torch.long, device=q.device
        )
        if not pruning:
            return empty_heads
        empty_blocks = torch.empty(
            (0, topk_heads, 0), dtype=torch.int32, device=q.device
        )
        if return_metadata:
            return MISASelection(
                candidate_head_ids=empty_heads,
                coarse_block_ids=empty_blocks,
                candidate_importance=torch.empty(
                    (0, topk_heads), dtype=torch.float32, device=q.device
                ),
                candidate_context_summary=torch.empty(
                    (0, topk_heads, DIM), dtype=torch.float32, device=q.device
                ),
                candidate_context_stats=torch.empty(
                    (0, topk_heads, 4), dtype=torch.float32, device=q.device
                ),
            )
        return empty_heads, empty_blocks
    if return_metadata and not pruning:
        raise ValueError("return_metadata requires prune_topk or prune_keep_fraction")

    rows_per = rows_per_launch
    if rows_per is None:
        pinned = os.environ.get("SGLANG_NSA_MISA_ROUTER_ROWS_PER_LAUNCH")
        if pinned is not None:
            rows_per = int(pinned)
        else:
            # Size the row group by the pooled-key gather it implies.  Decode
            # has one row per request, but ragged prefill has thousands, where a
            # fixed small group would cost 61 * nq/rows_per launches per chunk.
            budget = int(
                os.environ.get("SGLANG_NSA_MISA_ROUTER_GATHER_BYTES", 1 << 29)
            )
            worst_chunks = max(
                1, (int(row_len.max().item()) + CHUNK - 1) // CHUNK
            )
            rows_per = budget // (worst_chunks * DIM * 4)
    rows_per = max(1, min(rows_per, nq))
    chunks_per_pool = pooling_block_size // CHUNK
    q_bf = q.to(torch.bfloat16)
    # Keep the signed Indexer gate.  The official DSA score is
    # sum_h gate_h * relu(q_h @ k); taking abs(gate_h) here systematically
    # promotes strongly negative heads that suppress, rather than support,
    # the shared selector.
    weights_f32 = weights.float()
    row_batch = row_batch.to(torch.int64)
    row_len = row_len.to(torch.int64)
    bt = block_tables.to(torch.int32).contiguous()
    selected = torch.empty((nq, topk_heads), dtype=torch.long, device=q.device)
    selected_importance = torch.empty(
        (nq, topk_heads), dtype=torch.float32, device=q.device
    )
    variant_importance = None
    if importance_variant_output is not None:
        importance_variant_output.clear()
        variant_importance = {
            name: torch.empty(
                (nq, q.shape[1]), dtype=torch.float32, device=q.device
            )
            for name in (
                "baseline_all",
                "raw_top_count_8",
                "top_count_8",
                "top_count_16",
                "top_count_32",
                "top_count_60",
                "top_fraction_10",
                "top_fraction_25",
                "top_fraction_50",
            )
        }
    candidate_context_summary = torch.empty(
        (nq, topk_heads, DIM), dtype=torch.float32, device=q.device
    )
    candidate_context_stats = torch.empty(
        (nq, topk_heads, 4), dtype=torch.float32, device=q.device
    )
    selected_blocks = None
    if pruning:
        max_pooling_blocks = max(
            1,
            (
                int(row_len.max().item())
                + pooling_block_size
                - 1
            )
            // pooling_block_size,
        )
        max_kept_blocks = (
            prune_topk if prune_topk is not None else
            max(2, math.ceil(max_pooling_blocks * prune_keep_fraction))
        )
        selected_blocks = torch.full(
            (nq, topk_heads, max_kept_blocks),
            -1,
            dtype=torch.int32,
            device=q.device,
        )

    for start in range(0, nq, rows_per):
        end = min(start + rows_per, nq)
        batch_rows = end - start
        lens = row_len[start:end]
        max_tokens = int(lens.max().item())
        num_pooling_blocks = max(
            1, (max_tokens + pooling_block_size - 1) // pooling_block_size
        )
        num_chunks = num_pooling_blocks * chunks_per_pool
        pages = bt[row_batch[start:end]]
        logical_chunk_ids = _chunk_ids(pages, num_chunks)
        chunk_keys = chunk_sum[logical_chunk_ids]

        # Physical pages may be reused after the visible tail.  Mask every
        # logical 16-token chunk before pooling so stale sums cannot leak in.
        chunk_starts = torch.arange(num_chunks, device=q.device) * CHUNK
        valid_chunks = chunk_starts.view(1, -1) < lens.view(-1, 1)
        chunk_keys = chunk_keys * valid_chunks.unsqueeze(-1)
        pooled_sum = chunk_keys.view(
            batch_rows, num_pooling_blocks, chunks_per_pool, DIM
        ).sum(dim=2)

        pool_starts = (
            torch.arange(num_pooling_blocks, device=q.device) * pooling_block_size
        )
        pool_counts = (lens.view(-1, 1) - pool_starts.view(1, -1)).clamp(
            min=0, max=pooling_block_size
        )
        pooled_mean = pooled_sum / pool_counts.clamp_min(1).unsqueeze(-1)
        affinities = torch.bmm(
            q_bf[start:end], pooled_mean.to(torch.bfloat16).transpose(1, 2)
        ).float()
        positive_affinities = torch.relu(affinities)
        all_block_importance = (
            positive_affinities * weights_f32[start:end].unsqueeze(-1)
        ).sum(dim=-1)
        if prune_topk is not None:
            # Stage A ranks each Indexer head by exactly the coarse regions that
            # Stage B will expand. Invisible blocks have zero positive affinity,
            # so short rows need no special-case masking for this sum.
            importance = (
                torch.topk(
                    positive_affinities,
                    k=min(prune_topk, num_pooling_blocks),
                    dim=-1,
                    largest=True,
                    sorted=False,
                )
                .values.sum(dim=-1)
                * weights_f32[start:end]
            )
        else:
            importance = all_block_importance
        if variant_importance is not None:
            variant_importance["baseline_all"][start:end] = all_block_importance
            # Sort once, then use prefix sums to evaluate every fixed-count and
            # fixed-fraction selector without repeating the pooled QK matmul.
            prefix = positive_affinities.sort(
                dim=-1, descending=True
            ).values.cumsum(dim=-1)
            visible_blocks = (
                (lens + pooling_block_size - 1) // pooling_block_size
            ).clamp_min(1)
            keep_counts = {
                "top_count_8": visible_blocks.clamp(max=8),
                "top_count_16": visible_blocks.clamp(max=16),
                "top_count_32": visible_blocks.clamp(max=32),
                "top_count_60": visible_blocks.clamp(max=60),
                "top_fraction_10": torch.ceil(
                    visible_blocks.to(torch.float64) * 0.10
                ).long(),
                "top_fraction_25": torch.ceil(
                    visible_blocks.to(torch.float64) * 0.25
                ).long(),
                "top_fraction_50": torch.ceil(
                    visible_blocks.to(torch.float64) * 0.50
                ).long(),
            }
            for name, counts in keep_counts.items():
                prefix_index = (
                    (counts - 1)
                    .view(-1, 1, 1)
                    .expand(-1, q.shape[1], 1)
                )
                top_sum = prefix.gather(2, prefix_index).squeeze(-1)
                variant_importance[name][start:end] = (
                    top_sum * weights_f32[start:end]
                )
                if name == "top_count_8":
                    variant_importance["raw_top_count_8"][start:end] = top_sum
        selected[start:end] = torch.topk(
            importance, k=topk_heads, dim=-1, largest=True, sorted=False
        ).indices
        selected_importance[start:end] = torch.gather(
            importance, 1, selected[start:end]
        )
        if selected_blocks is not None:
            chosen_affinity = torch.gather(
                affinities,
                1,
                selected[start:end]
                .unsqueeze(-1)
                .expand(-1, -1, num_pooling_blocks),
            )
            raw_chosen_affinity = chosen_affinity.clone()
            visible_blocks = (
                lens + pooling_block_size - 1
            ) // pooling_block_size
            block_ids = torch.arange(num_pooling_blocks, device=q.device)
            stats_mask = block_ids.view(1, 1, -1) < visible_blocks.view(
                -1, 1, 1
            )
            positive = torch.relu(chosen_affinity).masked_fill(~stats_mask, 0)
            count = visible_blocks.float().view(-1, 1).clamp_min(1)
            mean = positive.sum(-1) / count
            variance = (
                (positive - mean.unsqueeze(-1)).square() * stats_mask
            ).sum(-1) / count
            maximum = positive.masked_fill(~stats_mask, -torch.inf).max(-1).values
            positive_fraction = ((chosen_affinity > 0) & stats_mask).sum(
                -1
            ).float() / count
            candidate_context_stats[start:end] = torch.stack(
                [mean, variance.sqrt(), maximum, positive_fraction], dim=-1
            )
            chosen_affinity.masked_fill_(
                block_ids.view(1, 1, -1)
                >= visible_blocks.view(-1, 1, 1),
                -torch.inf,
            )
            if prune_topk is not None:
                # Pure mean ranking: no extra sink/tail region is inserted.
                # Stage B consumes these exact IDs without selecting regions again.
                kept_per_row = visible_blocks.clamp(max=prune_topk)
            else:
                chosen_affinity[:, :, 0] = torch.inf
                tail = (
                    (visible_blocks - 1)
                    .view(-1, 1, 1)
                    .expand(-1, topk_heads, 1)
                )
                chosen_affinity.scatter_(2, tail, torch.inf)
                kept_per_row = torch.ceil(
                    visible_blocks.to(torch.float64) * prune_keep_fraction
                ).long().clamp(min=2)
                kept_per_row = torch.minimum(kept_per_row, visible_blocks)
            batch_max_kept = int(kept_per_row.max().item())
            block_choice = torch.topk(
                chosen_affinity,
                k=batch_max_kept,
                dim=-1,
                largest=True,
                # Shorter rows retain only the first ``kept_per_row`` entries
                # from this batch-wide Top-K. Keep them score-ordered so that
                # prefix is the row's actual best subset.
                sorted=True,
            ).indices.to(torch.int32)
            valid_rank = (
                torch.arange(batch_max_kept, device=q.device).view(1, 1, -1)
                < kept_per_row.view(-1, 1, 1)
            )
            selected_blocks[start:end, :, :batch_max_kept] = torch.where(
                valid_rank, block_choice, -1
            )
            summary_score = torch.gather(
                raw_chosen_affinity, 2, block_choice.long()
            ).masked_fill(~valid_rank, -torch.inf)
            summary_weight = torch.softmax(
                summary_score / context_temperature, dim=-1
            )
            batch_index = torch.arange(
                batch_rows, device=q.device
            ).view(-1, 1, 1)
            selected_means = pooled_mean[
                batch_index, block_choice.long()
            ]
            candidate_context_summary[start:end] = (
                summary_weight.unsqueeze(-1) * selected_means
            ).sum(dim=2)

    if importance_variant_output is not None:
        assert variant_importance is not None
        importance_variant_output.update(variant_importance)

    if selected_blocks is not None:
        if return_metadata:
            return MISASelection(
                candidate_head_ids=selected,
                coarse_block_ids=selected_blocks,
                candidate_importance=selected_importance,
                candidate_context_summary=candidate_context_summary,
                candidate_context_stats=candidate_context_stats,
            )
        return selected, selected_blocks
    return selected


@torch.no_grad()
def per_head_topk_paged(
    q: torch.Tensor,  # [nq, H, D] fp8/bf16 (per-head scale irrelevant to ranking)
    head_range: Optional[Tuple[int, int]],
    kv_cache: torch.Tensor,  # [P, 64*132] uint8 NSA indexer cache
    chunk_sum: torch.Tensor,  # [P*4, D] f32
    block_tables: torch.Tensor,  # [B, max_pages] i32 physical page ids
    row_batch: torch.Tensor,  # [nq] i32 request index per row
    row_len: torch.Tensor,  # [nq] i32 visible tokens per row (request relative)
    topk: int,
    *,
    head_ids: Optional[Sequence[int] | torch.Tensor] = None,
    coarse_block_ids: Optional[torch.Tensor] = None,
    coarse_block_size: int = MISA_DEFAULT_POOLING_BLOCK,
    mean_ranked_chunks: bool = False,
    segments: Optional[Sequence[Tuple[int, int, int]]] = None,  # (start, end, batch)
    pairs_per_launch: Optional[int] = None,
) -> torch.Tensor:
    """Returns ``[nq, G, topk]`` i32 request-relative token positions, valid
    picks first, -1 padded (at most ``OUTK`` = 720 valid per row).

    With ``mean_ranked_chunks=True``, mapped fine chunks are ranked by their
    actual visible-token means, with no boundary-chunk or boundary-token boost.
    The first 52 receive quota 8 and the remaining 76 receive quota 4. Incomplete
    tails can leave fewer than 720 valid picks, padded with -1.

    ``segments`` lists contiguous row ranges per request for ragged prefill;
    when omitted every row is its own request (decode).
    """
    assert topk >= OUTK, f"index_topk={topk} < per-head budget {OUTK}"
    pairs = per_head_pairs_per_launch() if pairs_per_launch is None else pairs_per_launch
    if (head_range is None) == (head_ids is None):
        raise ValueError("exactly one of head_range or head_ids must be provided")
    if coarse_block_ids is not None and head_ids is None:
        raise ValueError("coarse_block_ids requires explicit head_ids")
    if coarse_block_size < CHUNK or coarse_block_size % CHUNK:
        raise ValueError(
            f"coarse_block_size must be a multiple of {CHUNK}, "
            f"got {coarse_block_size}"
        )
    if head_ids is not None:
        head_index = torch.as_tensor(head_ids, dtype=torch.long, device=q.device)
        if head_index.ndim not in (1, 2):
            raise ValueError(
                f"head_ids must have shape [G] or [nq, G], got {tuple(head_index.shape)}"
            )
        if head_index.ndim == 2 and head_index.shape[0] != q.shape[0]:
            raise ValueError(
                f"dynamic head_ids rows {head_index.shape[0]} != q rows {q.shape[0]}"
            )
        if head_index.numel() == 0:
            raise ValueError("head_ids must not be empty")
        if bool(((head_index < 0) | (head_index >= q.shape[1])).any().item()):
            raise ValueError(f"head_ids out of range for H={q.shape[1]}: {head_ids}")
        g = int(head_index.shape[-1])
        # ``q`` is normally float8 here.  Some CUDA/PyTorch combinations have
        # produced invalid values for non-contiguous float8 index_select;
        # dequantize first for the arbitrary global-head gather.  The fixed
        # contiguous-head path below keeps the cheaper sliced conversion.
        q_bf = q.to(torch.bfloat16)
        if head_index.ndim == 1:
            q_selected = q_bf.index_select(1, head_index)
        else:
            q_selected = torch.gather(
                q_bf,
                dim=1,
                index=head_index.unsqueeze(-1).expand(-1, -1, q.shape[-1]),
            )
    else:
        assert head_range is not None
        h0, h1 = head_range
        g = h1 - h0
        q_selected = q[:, h0:h1]
    nq = q.shape[0]
    device = q.device
    out = torch.full((nq, g, topk), -1, dtype=torch.int32, device=device)
    if nq == 0 or g == 0:
        return out

    ext = _get_ext()
    q_bf = q_selected.to(torch.bfloat16).contiguous()  # [nq, G, D]
    row_len = row_len.to(torch.int32)
    bt = block_tables.to(torch.int32).contiguous()
    kv = kv_cache.view(kv_cache.shape[0], -1)
    rows_per = max(1, pairs // g)
    candidate_chunk_ids = None
    if coarse_block_ids is not None:
        coarse = coarse_block_ids.to(device=device, dtype=torch.int64)
        if coarse.ndim != 3 or coarse.shape[:2] != (nq, g):
            raise ValueError(
                "coarse_block_ids must have shape "
                f"[{nq}, {g}, K], got {tuple(coarse.shape)}"
            )
        chunks_per_coarse = coarse_block_size // CHUNK
        offsets = torch.arange(chunks_per_coarse, device=device)
        candidate_chunk_ids = (
            coarse.unsqueeze(-1) * chunks_per_coarse + offsets
        ).flatten(2)
        visible_chunks = (
            row_len.to(torch.int64) + CHUNK - 1
        ) // CHUNK
        coarse_valid = (
            (coarse.unsqueeze(-1) >= 0)
            .expand(-1, -1, -1, chunks_per_coarse)
            .flatten(2)
        )
        valid = coarse_valid & (
            candidate_chunk_ids < visible_chunks.view(-1, 1, 1)
        )
        candidate_chunk_ids = torch.where(
            valid, candidate_chunk_ids, -1
        ).to(torch.int32).contiguous()
        pad = (-candidate_chunk_ids.shape[-1]) % 8
        if pad:
            candidate_chunk_ids = torch.cat(
                [
                    candidate_chunk_ids,
                    candidate_chunk_ids.new_full((*candidate_chunk_ids.shape[:2], pad), -1),
                ],
                dim=-1,
            )

    debug = os.environ.get("SGLANG_NSA_PER_HEAD_DEBUG", "0").lower() not in _FALSE

    def _run(
        s: int,
        e: int,
        scores: torch.Tensor,
        batch_m: torch.Tensor,
        chunk_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Launch the fused kernel for rows [s, e) and fill ``out``."""
        b = e - s
        m = b * g
        if debug:
            q_finite = bool(torch.isfinite(q_bf[s:e].float()).all().item())
            scores_finite = bool(torch.isfinite(scores.float()).all().item())
            print(
                "[per_head_paged debug] "
                f"q_finite={q_finite} scores_finite={scores_finite} "
                f"scores={tuple(scores.shape)}",
                flush=True,
            )
            if not q_finite or not scores_finite:
                raise RuntimeError(
                    "per-head candidate scan received non-finite q/coarse scores"
                )
        ke_m = row_len[s:e].view(b, 1).expand(b, g).reshape(m).contiguous()
        tokens = torch.full((m, OUTK), -1, dtype=torch.int32, device=device)
        if chunk_ids is None:
            ext.chunk16_quota_paged(
                scores, q_bf[s:e].reshape(m, DIM), kv, bt, batch_m, ke_m, tokens
            )
        else:
            mapped_quota = (
                ext.chunk16_quota_paged_mapped_mean if mean_ranked_chunks
                else ext.chunk16_quota_paged_mapped
            )
            mapped_quota(
                scores,
                q_bf[s:e].reshape(m, DIM),
                kv,
                bt,
                batch_m,
                ke_m,
                chunk_ids,
                tokens,
            )
        if debug:
            torch.cuda.synchronize(device)
            invalid = (tokens < -1) | (tokens >= ke_m.view(-1, 1))
            invalid_count = int(invalid.sum().item())
            valid_count = int((tokens >= 0).sum().item())
            valid_max = int(tokens.max().item())
            print(
                "[per_head_paged debug] "
                f"candidate_valid={valid_count} invalid={invalid_count} "
                f"candidate_max={valid_max}",
                flush=True,
            )
            if invalid_count:
                raise RuntimeError(
                    f"per-head candidate kernel returned {invalid_count} "
                    "out-of-range token indices"
                )
        # valid picks first, -1 padding last: the attention kernels read the
        # leading entries per row
        tokens = torch.sort(tokens, dim=1, descending=True).values
        out[s:e, :, :OUTK] = tokens.view(b, g, OUTK)

    row_batch = row_batch.to(torch.int32)
    if debug and segments is None:
        logical_page = torch.div(
            row_len.to(torch.int64) - 1, PAGE, rounding_mode="floor"
        )
        max_logical_page = int(logical_page.max().item())
        if max_logical_page >= bt.shape[1]:
            raise RuntimeError(
                "per-head tail page exceeds block-table width: "
                f"max logical page={max_logical_page}, width={bt.shape[1]}"
            )
        physical_page = bt[
            row_batch.to(torch.int64), logical_page
        ].to(torch.int64)
        min_physical_page = int(physical_page.min().item())
        max_physical_page = int(physical_page.max().item())
        if min_physical_page < 0 or max_physical_page >= kv.shape[0]:
            raise RuntimeError(
                "per-head tail page is invalid: "
                f"physical=[{min_physical_page}, {max_physical_page}], "
                f"kv_pages={kv.shape[0]}, row_len={row_len.tolist()}"
            )
        print(
            "[per_head_paged debug] "
            f"row_len={row_len.tolist()} bt={tuple(bt.shape)} "
            f"tail_physical={physical_page.tolist()} kv_pages={kv.shape[0]}",
            flush=True,
        )

    if candidate_chunk_ids is not None:
        # Coarse-pruned scan.  ``row_batch`` and ``row_len`` are per row, and
        # the kernel resolves each row's pages itself, so this path serves both
        # decode (one row per request) and ragged prefill (many rows sharing a
        # request) without a segment-specific branch.
        for s in range(0, nq, rows_per):
            e = min(s + rows_per, nq)
            b = e - s
            m = b * g
            batch_m = (
                row_batch[s:e]
                .view(b, 1)
                .expand(b, g)
                .reshape(m)
                .contiguous()
            )
            ke_m = (
                row_len[s:e]
                .view(b, 1)
                .expand(b, g)
                .reshape(m)
                .contiguous()
            )
            chunk_ids_m = candidate_chunk_ids[s:e].reshape(
                m, -1
            ).contiguous()
            scores = torch.empty(
                chunk_ids_m.shape,
                dtype=torch.bfloat16,
                device=device,
            )
            score_chunks = (
                ext.selected_chunk_mean_scores_paged if mean_ranked_chunks
                else ext.selected_chunk_scores_paged
            )
            score_chunks(
                scores,
                q_bf[s:e].reshape(m, DIM),
                chunk_sum,
                bt,
                batch_m,
                ke_m,
                chunk_ids_m,
            )
            _run(s, e, scores, batch_m, chunk_ids_m)
    elif segments is None:
        # decode: one row per request
        n_chunks = _pad8(int((int(row_len.max().item()) + CHUNK - 1) // CHUNK))
        for s in range(0, nq, rows_per):
            e = min(s + rows_per, nq)
            b = e - s
            pages_b = bt[row_batch[s:e].to(torch.int64)]  # [b, W]
            stat = chunk_sum[_chunk_ids(pages_b, n_chunks)].to(torch.bfloat16)
            scores = torch.bmm(q_bf[s:e], stat.transpose(1, 2))  # [b, G, C]
            batch_m = (
                row_batch[s:e].view(b, 1).expand(b, g).reshape(b * g).contiguous()
            )
            _run(s, e, scores.reshape(b * g, n_chunks), batch_m)
    else:
        # ragged prefill: rows of a segment share one request's chunks
        for start, end, req in segments:
            if end <= start:
                continue
            n_chunks = _pad8(int((int(row_len[end - 1].item()) + CHUNK - 1) // CHUNK))
            stat = chunk_sum[_chunk_ids(bt[req], n_chunks)].to(torch.bfloat16)  # [C, D]
            for s in range(start, end, rows_per):
                e = min(s + rows_per, end)
                m = (e - s) * g
                scores = q_bf[s:e].reshape(m, DIM) @ stat.T  # [m, C]
                batch_m = torch.full((m,), req, dtype=torch.int32, device=device)
                _run(s, e, scores, batch_m)

    # rows whose whole prefix fits in the budget keep every token, matching
    # the dense indexer (the quota would otherwise drop tokens needlessly)
    short = row_len <= OUTK
    if bool(short.any()):
        lens = row_len[short]
        max_l = int(lens.max())
        ar = torch.arange(max_l, dtype=torch.int32, device=device)
        fill = torch.where(ar.view(1, -1) < lens.view(-1, 1), ar.view(1, -1), -1)
        out[short] = -1
        out[short, :, :max_l] = fill.view(-1, 1, max_l).expand(-1, g, max_l)
    return out
