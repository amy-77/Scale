"""Group16 all-64 teacher probe.

Each TP rank writes one ``[64 candidates, 16 local MLA heads]`` mass matrix.
The offline trainer averages the 16 heads and joins eight ranks by request ID.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NamedTuple, Optional

import torch


class ProbeInput(NamedTuple):
    request_id: str
    candidate_token_ids: torch.Tensor
    selected_indexer_head_id: Optional[int]
    selected_indexer_exact_top720_recall: Optional[float]
    exact_indexer_topk_ids: Optional[dict[int, torch.Tensor]]
    exact_merged_topk_ids: Optional[dict[int, torch.Tensor]]
    selector_ablation_per_indexer_ids: Optional[dict[str, torch.Tensor]]
    selector_ablation_shared_ids: Optional[dict[str, torch.Tensor]]
    selector_ablation_stats: Optional[dict[str, float]]
    all_indexer_q: Optional[torch.Tensor]
    all_indexer_gate: Optional[torch.Tensor]
    candidate_head_ids: Optional[torch.Tensor]
    candidate_importance: Optional[torch.Tensor]
    candidate_context_summary: Optional[torch.Tensor]
    candidate_context_stats: Optional[torch.Tensor]
    candidate_top8_sum: Optional[torch.Tensor]
    candidate_top8_context_summary: Optional[torch.Tensor]
    importance_variants: Optional[dict[str, torch.Tensor]]


class RequestInfo(NamedTuple):
    request_id: str
    sample_id: int
    expected_seq_len: int
    prefix_hash: str


_INPUTS: dict[int, ProbeInput] = {}
_COUNTS: dict[int, int] = {}
_BUFFERS: dict[int, list[dict]] = {}
_SEEN: dict[int, set[str]] = {}
_REQUEST_PREFIX = "group16-probe-"


def request_info(forward_batch) -> Optional[RequestInfo]:
    """Decode the stable identity supplied by the replay driver."""
    rids = getattr(forward_batch, "rids", None)
    if rids is None or len(rids) != 1:
        return None
    request_id = str(rids[0])
    if not request_id.startswith(_REQUEST_PREFIX):
        return None
    try:
        sample, seq_len, prefix_hash = request_id[len(_REQUEST_PREFIX) :].split(
            "-", 2
        )
        return RequestInfo(
            request_id=request_id,
            sample_id=int(sample),
            expected_seq_len=int(seq_len),
            prefix_hash=prefix_hash,
        )
    except (TypeError, ValueError):
        return None


def enabled() -> bool:
    return bool(os.environ.get("SGLANG_NSA_HEADMAP_PROBE_DIR"))


def wants_layer(layer_id: int) -> bool:
    configured = os.environ.get(
        "SGLANG_NSA_HEADMAP_PROBE_LAYERS", "0,30,60"
    )
    return layer_id in {int(value) for value in configured.split(",") if value}


def min_len() -> int:
    return int(os.environ.get("SGLANG_NSA_HEADMAP_PROBE_MIN_LEN", "120000"))


def max_samples() -> int:
    return int(os.environ.get("SGLANG_NSA_HEADMAP_PROBE_MAX_SAMPLES", "1"))


def wants_sample(layer_id: int) -> bool:
    return _COUNTS.get(layer_id, 0) < max_samples()


def candidate_budget() -> int:
    return int(os.environ.get("SGLANG_NSA_HEADMAP_PROBE_MISA_BUDGET", "8"))


def pooling_block_size() -> int:
    return int(os.environ.get("SGLANG_NSA_HEADMAP_PROBE_MISA_BLOCK_SIZE", "256"))


def prune_keep_fraction() -> Optional[float]:
    raw = os.environ.get("SGLANG_NSA_HEADMAP_PROBE_MISA_KEEP")
    return None if raw is None else float(raw)


def prune_topk() -> Optional[int]:
    raw = os.environ.get("SGLANG_NSA_HEADMAP_PROBE_MISA_TOPK")
    if raw is None:
        return 8 if prune_keep_fraction() is None else None
    return int(raw)


def shard_size() -> int:
    return int(os.environ.get("SGLANG_NSA_HEADMAP_PROBE_SHARD_SIZE", "16"))


def save_exact_topk_ids() -> bool:
    return os.environ.get(
        "SGLANG_NSA_HEADMAP_PROBE_SAVE_EXACT_TOPK_IDS", "0"
    ).lower() in ("1", "true", "yes")


def exact_indexer_recall_enabled() -> bool:
    return os.environ.get(
        "SGLANG_NSA_HEADMAP_PROBE_EXACT_INDEXER_RECALL", "0"
    ).lower() in ("1", "true", "yes")


def exact_dense_ablation_enabled() -> bool:
    return os.environ.get(
        "SGLANG_NSA_HEADMAP_PROBE_EXACT_DENSE_ABLATION", "0"
    ).lower() in ("1", "true", "yes")


def selected_static_indexer(layer_id: int, tp_rank: int) -> Optional[int]:
    path = os.environ.get("SGLANG_NSA_HEADMAP_PROBE_STATIC_CONFIG")
    if not path:
        return None
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    assignment = config.get("static_assignment_by_layer", {}).get(str(layer_id))
    if assignment is None:
        return None
    if len(assignment) != 8 or not 0 <= tp_rank < len(assignment):
        raise RuntimeError(
            f"invalid static probe assignment for layer={layer_id}, rank={tp_rank}"
        )
    return int(assignment[tp_rank])


def save_group_router_features() -> bool:
    return os.environ.get(
        "SGLANG_NSA_HEADMAP_PROBE_SAVE_GROUP_ROUTER_FEATURES", "0"
    ).lower() in ("1", "true", "yes")


def put_picks(
    layer_id: int,
    picks: torch.Tensor,
    *,
    request_id: str,
    selected_indexer_head_id: Optional[int] = None,
    selected_indexer_exact_top720_recall: Optional[float] = None,
    exact_indexer_topk_ids: Optional[dict[int, torch.Tensor]] = None,
    exact_merged_topk_ids: Optional[dict[int, torch.Tensor]] = None,
    selector_ablation_per_indexer_ids: Optional[
        dict[str, torch.Tensor]
    ] = None,
    selector_ablation_shared_ids: Optional[dict[str, torch.Tensor]] = None,
    selector_ablation_stats: Optional[dict[str, float]] = None,
    all_indexer_q: Optional[torch.Tensor] = None,
    all_indexer_gate: Optional[torch.Tensor] = None,
    candidate_head_ids: Optional[torch.Tensor] = None,
    candidate_importance: Optional[torch.Tensor] = None,
    candidate_context_summary: Optional[torch.Tensor] = None,
    candidate_context_stats: Optional[torch.Tensor] = None,
    candidate_top8_sum: Optional[torch.Tensor] = None,
    candidate_top8_context_summary: Optional[torch.Tensor] = None,
    importance_variants: Optional[dict[str, torch.Tensor]] = None,
) -> None:
    _INPUTS[layer_id] = ProbeInput(
        request_id=request_id,
        candidate_token_ids=picks,
        selected_indexer_head_id=selected_indexer_head_id,
        selected_indexer_exact_top720_recall=selected_indexer_exact_top720_recall,
        exact_indexer_topk_ids=exact_indexer_topk_ids,
        exact_merged_topk_ids=exact_merged_topk_ids,
        selector_ablation_per_indexer_ids=selector_ablation_per_indexer_ids,
        selector_ablation_shared_ids=selector_ablation_shared_ids,
        selector_ablation_stats=selector_ablation_stats,
        all_indexer_q=all_indexer_q,
        all_indexer_gate=all_indexer_gate,
        candidate_head_ids=candidate_head_ids,
        candidate_importance=candidate_importance,
        candidate_context_summary=candidate_context_summary,
        candidate_context_stats=candidate_context_stats,
        candidate_top8_sum=candidate_top8_sum,
        candidate_top8_context_summary=candidate_top8_context_summary,
        importance_variants=importance_variants,
    )


def _required(value: Optional[torch.Tensor], name: str) -> torch.Tensor:
    if value is None:
        raise RuntimeError(f"teacher probe is missing {name}")
    return value


def _flush(out: Path, layer_id: int, rank: int) -> None:
    records = _BUFFERS.get(layer_id, [])
    if not records:
        return
    first, last = records[0]["sample_id"], records[-1]["sample_id"]
    torch.save(
        records,
        out
        / f"probe_shard_L{layer_id:02d}_rank{rank}_s{first:05d}-{last:05d}.pt",
    )
    records.clear()


@torch.no_grad()
def maybe_dump_decode(
    layer,
    forward_batch,
    metadata,
    q_all: torch.Tensor,
    shared_indices: torch.Tensor,
) -> None:
    layer_id = int(layer.layer_id)
    identity = request_info(forward_batch)
    if (
        not enabled()
        or not wants_layer(layer_id)
        or not wants_sample(layer_id)
        or identity is None
        or identity.request_id in _SEEN.get(layer_id, set())
        or q_all.shape[0] != 1
        or layer_id not in _INPUTS
    ):
        return

    seq_len = int(forward_batch.seq_lens[0])
    probe = _INPUTS[layer_id]
    if (
        seq_len < min_len()
        or seq_len != identity.expected_seq_len
        or probe.request_id != identity.request_id
    ):
        return
    locations = metadata.page_table_1[0, :seq_len].long()
    k_nope, k_rope = forward_batch.token_to_kv_pool.get_mla_kv_buffer(
        layer, locations, dst_dtype=torch.bfloat16
    )
    key = torch.cat([k_nope.squeeze(1), k_rope.squeeze(1)], dim=-1)
    attention = torch.softmax(
        torch.matmul(q_all[0].float(), key.float().T) * layer.scaling,
        dim=-1,
    )

    _INPUTS.pop(layer_id)
    token_ids = probe.candidate_token_ids[0].long()
    out_of_range = (token_ids != -1) & (
        (token_ids < 0) | (token_ids >= seq_len)
    )
    if out_of_range.any():
        raise RuntimeError("Group16 candidate set contains out-of-range token IDs")
    valid = token_ids != -1
    if (valid.sum(dim=-1) > 720).any():
        raise RuntimeError("Group16 candidate set contains more than 720 tokens")
    ordered_ids = token_ids.masked_fill(~valid, seq_len).sort(dim=-1).values
    duplicates = (
        (ordered_ids[:, 1:] == ordered_ids[:, :-1])
        & (ordered_ids[:, 1:] < seq_len)
    )
    if duplicates.any():
        raise RuntimeError("Group16 candidate set contains duplicate token IDs")
    gather_ids = token_ids.clamp(0, seq_len - 1)
    # [local MLA heads, M, K] -> [M, local MLA heads]
    candidate_mass = attention[:, gather_ids].masked_fill(
        ~valid.unsqueeze(0), 0
    ).sum(-1).transpose(0, 1).contiguous()
    if candidate_mass.max() > 1.0001:
        raise RuntimeError("candidate attention mass exceeds one")
    shared_token_ids = shared_indices[0].long()
    shared_valid = (shared_token_ids >= 0) & (shared_token_ids < seq_len)
    shared_gather_ids = shared_token_ids.clamp(0, seq_len - 1)
    official_shared_mass = attention[:, shared_gather_ids].masked_fill(
        ~shared_valid.unsqueeze(0), 0
    ).sum(-1)
    exact_topk = min(720, seq_len)
    exact_mla_top720 = attention.topk(exact_topk, dim=-1, sorted=False)
    exact_mla_top720_mass = exact_mla_top720.values.sum(-1)
    group_exact_top720_mass = attention.mean(dim=0).topk(
        exact_topk, sorted=False
    ).values.sum()

    try:
        from sglang.srt.layers.dp_attention import get_attention_tp_rank

        rank = int(get_attention_tp_rank())
    except Exception:
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
    local_heads = q_all.shape[1]
    global_head_ids = torch.arange(
        rank * local_heads,
        (rank + 1) * local_heads,
        device=q_all.device,
    )
    candidate_head_ids = _required(
        probe.candidate_head_ids, "candidate_head_ids"
    )[0].long()
    importance_variants = {
        name: values[0, candidate_head_ids].float().cpu()
        for name, values in (probe.importance_variants or {}).items()
    }
    all_indexer_q = _required(probe.all_indexer_q, "all_indexer_q")
    all_indexer_gate = _required(probe.all_indexer_gate, "all_indexer_gate")
    sample_id = identity.sample_id
    record = {
        "sample_id": sample_id,
        "request_id": identity.request_id,
        "prefix_hash": identity.prefix_hash,
        "query_position": seq_len - 1,
        "query_token_id": int(forward_batch.input_ids.reshape(-1)[0]),
        "layer": layer_id,
        "tp_rank": rank,
        "seq_len": seq_len,
        "global_mla_head_ids": global_head_ids.cpu(),
        "q_mla": q_all[0].to(torch.bfloat16).cpu(),
        "candidate_head_ids": candidate_head_ids.cpu(),
        "candidate_q_indexer": all_indexer_q[
            0, candidate_head_ids
        ].to(torch.bfloat16).cpu(),
        "candidate_context_summary": _required(
            probe.candidate_context_summary, "candidate_context_summary"
        )[0].to(torch.bfloat16).cpu(),
        "candidate_gate": all_indexer_gate[
            0, candidate_head_ids
        ].float().cpu(),
        "candidate_importance": _required(
            probe.candidate_importance, "candidate_importance"
        )[0].float().cpu(),
        "candidate_context_stats": _required(
            probe.candidate_context_stats, "candidate_context_stats"
        )[0].float().cpu(),
        "importance_variants": importance_variants,
        "attention_mass_matrix": candidate_mass.cpu(),
        "official_shared_mass": official_shared_mass.cpu(),
        "exact_mla_top720_mass": exact_mla_top720_mass.cpu(),
        "group_exact_top720_mass": group_exact_top720_mass.cpu(),
    }
    if probe.exact_indexer_topk_ids is not None:
        record["exact_indexer_topk_attention_mass"] = {
            str(k): attention[:, ids[0].long()].sum(-1).transpose(0, 1).cpu()
            for k, ids in probe.exact_indexer_topk_ids.items()
        }
    if probe.exact_merged_topk_ids is not None:
        record["exact_merged_topk_attention_mass"] = {
            str(k): attention[:, ids[0].long()].sum(-1).cpu()
            for k, ids in probe.exact_merged_topk_ids.items()
        }
        merged_2048 = probe.exact_merged_topk_ids.get(2048)
        if merged_2048 is not None:
            exact_ids = merged_2048[0].long()
            record["official_vs_recomputed_merged_top2048_recall"] = (
                torch.isin(exact_ids, shared_token_ids[shared_valid])
                .float()
                .mean()
                .cpu()
            )
    if probe.selector_ablation_per_indexer_ids is not None:
        per_indexer_mass = {}
        per_indexer_recall = {}
        exact_top720 = (
            probe.exact_indexer_topk_ids.get(720)[0].long()
            if probe.exact_indexer_topk_ids is not None
            and probe.exact_indexer_topk_ids.get(720) is not None
            else None
        )
        for name, ids in probe.selector_ablation_per_indexer_ids.items():
            token_ids = ids[0].long()
            valid_ids = (token_ids >= 0) & (token_ids < seq_len)
            gather_ids = token_ids.clamp(0, seq_len - 1)
            per_indexer_mass[name] = (
                attention[:, gather_ids]
                .masked_fill(~valid_ids.unsqueeze(0), 0)
                .sum(-1)
                .transpose(0, 1)
                .cpu()
            )
            if exact_top720 is not None:
                sorted_ids = token_ids.masked_fill(~valid_ids, seq_len).sort(
                    dim=-1
                ).values
                positions = torch.searchsorted(sorted_ids, exact_top720)
                in_range = positions < sorted_ids.shape[-1]
                matched_ids = sorted_ids.gather(
                    1, positions.clamp_max(sorted_ids.shape[-1] - 1)
                )
                per_indexer_recall[name] = (
                    (in_range & (matched_ids == exact_top720))
                    .float()
                    .mean(dim=-1)
                    .cpu()
                )
        record["selector_ablation_per_indexer_attention_mass"] = (
            per_indexer_mass
        )
        if per_indexer_recall:
            record["selector_ablation_per_indexer_exact_top720_recall"] = (
                per_indexer_recall
            )
    if probe.selector_ablation_shared_ids is not None:
        shared_mass = {}
        for name, ids in probe.selector_ablation_shared_ids.items():
            token_ids = ids[0].long()
            valid_ids = (token_ids >= 0) & (token_ids < seq_len)
            gather_ids = token_ids.clamp(0, seq_len - 1)
            shared_mass[name] = (
                attention[:, gather_ids]
                .masked_fill(~valid_ids.unsqueeze(0), 0)
                .sum(-1)
                .cpu()
            )
        record["selector_ablation_shared_attention_mass"] = shared_mass
    if probe.selector_ablation_stats is not None:
        record["selector_ablation_stats"] = probe.selector_ablation_stats
    if probe.selected_indexer_head_id is not None:
        record["selected_indexer_head_id"] = probe.selected_indexer_head_id
    if probe.selected_indexer_exact_top720_recall is not None:
        record["selected_indexer_exact_top720_recall"] = (
            probe.selected_indexer_exact_top720_recall
        )
    if probe.candidate_top8_sum is not None:
        record["candidate_top8_sum"] = probe.candidate_top8_sum[0].float().cpu()
    if probe.candidate_top8_context_summary is not None:
        record["candidate_top8_context_summary"] = (
            probe.candidate_top8_context_summary[0].to(torch.bfloat16).cpu()
        )
    if save_exact_topk_ids():
        record["exact_mla_top720_ids"] = exact_mla_top720.indices.to(
            torch.int32
        ).cpu()

    out = Path(os.environ["SGLANG_NSA_HEADMAP_PROBE_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    buffer = _BUFFERS.setdefault(layer_id, [])
    buffer.append(record)
    _SEEN.setdefault(layer_id, set()).add(identity.request_id)
    _COUNTS[layer_id] = _COUNTS.get(layer_id, 0) + 1
    if len(buffer) >= shard_size() or _COUNTS[layer_id] >= max_samples():
        _flush(out, layer_id, rank)
    if _COUNTS[layer_id] >= max_samples():
        marker = out / f"complete_L{layer_id:02d}_rank{rank}.json"
        marker.write_text(
            json.dumps(
                {
                    "layer": layer_id,
                    "rank": rank,
                    "records": _COUNTS[layer_id],
                }
            )
            + "\n",
            encoding="utf-8",
        )
