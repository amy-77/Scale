"""Inference adapter for dynamic MISA selection + TP-local group assignment.

The selector and router deliberately have separate phases:

1. The indexer selects ``M`` query-dependent heads from pooled historical keys
   (MISA), scans those heads, and returns ``[tokens, M, topk]`` candidates.  A
   fixed offline set remains available as an explicit ablation.
2. The group16 model maps the existing 16 local MLA heads to one candidate.
   Its token indices fill the existing per-pair attention contract
   ``[tokens, local_pairs, topk]`` without moving heads. Explicit legacy
   checkpoints retain their original pair assignment path.

Enable with ``SGLANG_NSA_OFFLINE_ROUTER_CONFIG=/path/to/config.json`` and
disable CUDA graphs.  The JSON is intentionally self-contained and records
the checkpoint, budget, and per-layer selected sets used by an experiment.
Layers absent from ``selected_heads_by_layer`` keep the official shared DSA
selector; this makes partial-layer canaries explicit instead of silently
pretending a seven-layer checkpoint covers the full model.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.nsa.assignment_router_model import (
    MISAAssignmentRouter,
)

if TYPE_CHECKING:
    from sglang.srt.layers.attention.nsa.per_head_paged import MISASelection


_CONFIG_ENV = "SGLANG_NSA_OFFLINE_ROUTER_CONFIG"


@dataclass(frozen=True)
class RouterConfig:
    path: Path
    checkpoint: Optional[Path]
    budget: int
    assignment_mode: str
    head_selection_mode: str
    misa_chunk_size: int
    misa_prune_keep_fraction: Optional[float]
    misa_prune_topk: Optional[int]
    mla_heads_per_group: Optional[int]
    min_seq_len: int
    selected_heads_by_layer: Dict[int, tuple[int, ...]]
    static_assignment_by_layer: Optional[Dict[int, tuple[int, ...]]]


_config: Optional[RouterConfig] = None
_checkpoint: Optional[dict[str, Any]] = None
_device_states: dict[str, dict[str, torch.Tensor]] = {}
_device_models: dict[str, MISAAssignmentRouter] = {}
_traced_routes: set[tuple[int, int]] = set()


def enabled() -> bool:
    return bool(os.environ.get(_CONFIG_ENV, "").strip())


def _load_config() -> RouterConfig:
    global _config
    if _config is not None:
        return _config

    raw_path = os.environ.get(_CONFIG_ENV, "").strip()
    if not raw_path:
        raise RuntimeError(f"{_CONFIG_ENV} is not set")
    path = Path(raw_path).expanduser().resolve()
    raw = json.loads(path.read_text())
    budget = int(raw["budget"])
    if budget <= 0 or budget > 64:
        raise ValueError(f"offline router budget must be in [1, 64], got {budget}")

    assignment_mode = str(raw.get("assignment_mode", "learned"))
    if assignment_mode not in ("learned", "static"):
        raise ValueError(f"unsupported router assignment_mode: {assignment_mode}")
    # Learned-router configs created before MISA did not carry this field.
    # Treat them as MISA so enabling the existing experiment config exercises
    # the new first-stage selector.  Static pair-assignment configs retain the
    # original fixed-set ablation unless explicitly rebuilt otherwise.
    head_selection_mode = str(
        raw.get(
            "head_selection_mode",
            "misa" if assignment_mode == "learned" else "static",
        )
    )
    if head_selection_mode not in ("misa", "static"):
        raise ValueError(
            f"unsupported router head_selection_mode: {head_selection_mode}"
        )
    if head_selection_mode == "misa" and assignment_mode == "static":
        raise ValueError(
            "head_selection_mode=misa requires assignment_mode=learned; "
            "a static pair map cannot address query-dependent heads"
        )
    misa_chunk_size = int(raw.get("misa_chunk_size", 256))
    if misa_chunk_size <= 0 or misa_chunk_size % 16:
        raise ValueError(
            "offline router misa_chunk_size must be a positive multiple of 16, "
            f"got {misa_chunk_size}"
        )
    # Explicit legacy fraction configs keep their original policy. New configs
    # default to eight mean-ranked regions, shared between stages A and B.
    raw_fraction = raw.get("misa_prune_keep_fraction")
    raw_topk = raw.get("misa_prune_topk", 8 if raw_fraction is None else None)
    misa_prune_keep_fraction = None if raw_fraction is None else float(raw_fraction)
    misa_prune_topk = raw_topk
    if misa_prune_keep_fraction is not None and not 0 < misa_prune_keep_fraction <= 1:
        raise ValueError("offline router misa_prune_keep_fraction must be in (0, 1]")
    if misa_prune_topk is not None:
        if (not isinstance(misa_prune_topk, int) or isinstance(misa_prune_topk, bool)
                or misa_prune_topk <= 0):
            raise ValueError("offline router misa_prune_topk must be a positive integer")
        if misa_prune_keep_fraction is not None:
            raise ValueError("misa_prune_topk and misa_prune_keep_fraction are mutually exclusive")
    elif misa_prune_keep_fraction is None:
        raise ValueError("configure misa_prune_topk or misa_prune_keep_fraction")
    mla_heads_per_group = raw.get("mla_heads_per_group")
    if mla_heads_per_group not in (None, 2, 16):
        raise ValueError("mla_heads_per_group must be 16 (TP-local) or explicit legacy 2")
    if mla_heads_per_group == 16:
        valid_group16_mode = (
            assignment_mode == "learned" and head_selection_mode == "misa"
        ) or (
            assignment_mode == "static" and head_selection_mode == "static"
        )
        if not valid_group16_mode:
            raise ValueError(
                "TP-local group16 requires learned+MISA or static+static"
            )
        if assignment_mode == "static" and budget != 1:
            raise ValueError(
                "TP-local static Group16 scans exactly one Indexer per TP rank; "
                "configure budget=1"
            )
    raw_checkpoint = raw.get("checkpoint")
    checkpoint = None
    if raw_checkpoint is not None:
        checkpoint = Path(raw_checkpoint).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = (path.parent / checkpoint).resolve()
    if assignment_mode == "learned" and checkpoint is None:
        raise ValueError("learned assignment requires checkpoint")
    min_seq_len = int(raw.get("min_seq_len", 4096))
    if min_seq_len < 720:
        raise ValueError(
            f"offline router min_seq_len must be at least 720, got {min_seq_len}"
        )

    static_assignment = None
    if assignment_mode == "static":
        raw_assignment = raw.get("static_assignment_by_layer")
        if not isinstance(raw_assignment, dict):
            raise ValueError(
                "assignment_mode=static requires static_assignment_by_layer"
            )
        static_assignment = {
            int(layer): tuple(int(head) for head in heads)
            for layer, heads in raw_assignment.items()
        }
        for layer, assignment in static_assignment.items():
            expected_entries = 8 if mla_heads_per_group == 16 else 64
            if len(assignment) != expected_entries:
                raise ValueError(
                    f"layer {layer} has {len(assignment)} static assignment "
                    f"entries, expected {expected_entries}"
                )
            if any(head < 0 or head >= 64 for head in assignment):
                raise ValueError(
                    f"layer {layer} has invalid static assignment: {assignment}"
                )

    raw_selected = raw.get("selected_heads_by_layer")
    if raw_selected is None:
        if static_assignment is None:
            raise ValueError("selected_heads_by_layer is required")
        selected = {
            layer: tuple(dict.fromkeys(assignment))
            for layer, assignment in static_assignment.items()
        }
    else:
        selected = {
            int(layer): tuple(int(head) for head in heads)
            for layer, heads in raw_selected.items()
        }
    if static_assignment is not None and set(static_assignment) != set(selected):
        raise ValueError(
            "static assignment layers must exactly match selected-head layers"
        )
    for layer, heads in selected.items():
        if mla_heads_per_group == 16 and assignment_mode == "static":
            expected_heads = set(static_assignment[layer])
            if set(heads) != expected_heads:
                raise ValueError(
                    f"layer {layer} selected heads do not cover its Group16 assignment"
                )
        elif len(heads) != budget:
            raise ValueError(
                f"layer {layer} has {len(heads)} selected heads, expected M={budget}"
            )
        if len(set(heads)) != len(heads) or any(head < 0 or head >= 64 for head in heads):
            raise ValueError(f"layer {layer} has invalid selected heads: {heads}")
        if static_assignment is not None:
            outside = sorted(set(static_assignment[layer]) - set(heads))
            if outside:
                raise ValueError(
                    f"layer {layer} static assignment uses heads outside its "
                    f"global set: {outside}"
                )

    _config = RouterConfig(
        path=path,
        checkpoint=checkpoint,
        budget=budget,
        assignment_mode=assignment_mode,
        head_selection_mode=head_selection_mode,
        misa_chunk_size=misa_chunk_size,
        misa_prune_keep_fraction=misa_prune_keep_fraction,
        misa_prune_topk=misa_prune_topk,
        mla_heads_per_group=mla_heads_per_group,
        min_seq_len=min_seq_len,
        selected_heads_by_layer=selected,
        static_assignment_by_layer=static_assignment,
    )
    return _config


def candidate_head_ids(layer_id: int) -> Optional[tuple[int, ...]]:
    """Return the Indexer heads this TP rank must scan, or ``None``.

    In MISA mode the tuple marks layer coverage and its values are not used for
    selection; the runtime supplies a query-dependent ``[tokens, M]`` tensor.
    TP-local static Group16 returns one head from its frozen ``[layer, 8]`` map.
    """
    if not enabled():
        return None
    config = _load_config()
    configured = config.selected_heads_by_layer.get(int(layer_id))
    if configured is None:
        return None
    if config.assignment_mode == "static" and config.mla_heads_per_group == 16:
        assert config.static_assignment_by_layer is not None
        from sglang.srt.layers.dp_attention import (
            get_attention_tp_rank,
            get_attention_tp_size,
        )

        tp_size = get_attention_tp_size()
        tp_rank = get_attention_tp_rank()
        if tp_size != 8 or not 0 <= tp_rank < tp_size:
            raise ValueError(
                f"TP-local static Group16 requires attention TP=8, got "
                f"rank={tp_rank}, size={tp_size}"
            )
        return (config.static_assignment_by_layer[int(layer_id)][tp_rank],)
    return configured


def budget() -> int:
    return _load_config().budget


def head_selection_mode() -> str:
    return _load_config().head_selection_mode


def assignment_mode() -> str:
    return _load_config().assignment_mode


def static_group16_enabled() -> bool:
    config = _load_config()
    return config.assignment_mode == "static" and config.mla_heads_per_group == 16


def misa_chunk_size() -> int:
    return _load_config().misa_chunk_size


def misa_prune_keep_fraction() -> Optional[float]:
    return _load_config().misa_prune_keep_fraction


def misa_prune_topk() -> Optional[int]:
    return _load_config().misa_prune_topk


def min_seq_len() -> int:
    return _load_config().min_seq_len


def _load_checkpoint() -> dict[str, Any]:
    global _checkpoint
    if _checkpoint is None:
        config = _load_config()
        if config.checkpoint is None:
            raise ValueError("this router mode has no checkpoint")
        checkpoint = torch.load(
            config.checkpoint, map_location="cpu", weights_only=False
        )
        if checkpoint.get("model_class") not in (
            "OfflineRouter",
            "MISAAssignmentRouter",
        ):
            raise ValueError(
                f"unsupported router checkpoint class: {checkpoint.get('model_class')}"
            )
        group_size = int(checkpoint.get("mla_heads_per_group", 2))
        if config.mla_heads_per_group is not None and group_size != config.mla_heads_per_group:
            raise ValueError("router group size mismatch: group16 cannot load a pair-router checkpoint")
        if group_size == 16:
            if (checkpoint.get("model_class") != "MISAAssignmentRouter"
                    or checkpoint.get("attention_tp_size") != 8
                    or checkpoint.get("assignment_unit") != "tp_local"):
                raise ValueError("group16 checkpoint requires TP-local 16-head metadata and attention TP=8")
        elif group_size != 2:
            raise ValueError(f"unsupported router group size: {group_size}")
        model_config = checkpoint["model_config"]
        if checkpoint.get("model_class") == "OfflineRouter" and bool(
            model_config.get("dynamic_indexer_keys", False)
        ):
            raise NotImplementedError(
                "legacy router runtime does not support dynamic indexer keys"
            )
        checkpoint_layers = {int(layer) for layer in checkpoint["layers"]}
        missing = set(config.selected_heads_by_layer) - checkpoint_layers
        if missing:
            raise ValueError(
                f"router config selects layers absent from checkpoint: {sorted(missing)}"
            )
        _checkpoint = checkpoint
    return _checkpoint


def _state_for(device: torch.device) -> dict[str, torch.Tensor]:
    key = str(device)
    state = _device_states.get(key)
    if state is None:
        checkpoint = _load_checkpoint()
        prefix = "assignment_scorer."
        state = {
            name[len(prefix) :]: tensor.to(device=device, dtype=torch.float32)
            for name, tensor in checkpoint["state_dict"].items()
            if name.startswith(prefix)
        }
        required = {
            "q_proj.weight",
            "indexer_embedding",
            "layer_embedding.weight",
            "prior",
            "pair_prior",
            "gate_scale",
        }
        missing = required - set(state)
        if missing:
            raise ValueError(f"router checkpoint is missing tensors: {sorted(missing)}")
        _device_states[key] = state
    return state


def _model_for(device: torch.device) -> MISAAssignmentRouter:
    key = str(device)
    model = _device_models.get(key)
    if model is None:
        checkpoint = _load_checkpoint()
        if checkpoint.get("model_class") != "MISAAssignmentRouter":
            raise ValueError("candidate-wise model requested for a legacy checkpoint")
        model = MISAAssignmentRouter(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device=device, dtype=torch.float32).eval()
        _device_models[key] = model
    return model


def _trace_misa_route(
    *,
    tp_rank: int,
    layer_id: int,
    q_group: torch.Tensor,
    mla_heads_per_group: int,
    candidates: torch.Tensor,
    features: dict[str, torch.Tensor],
    score: torch.Tensor,
    choice: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Persist one optional runtime record per rank/layer for exact replay."""
    trace_dir = os.environ.get("SGLANG_NSA_ROUTER_TRACE_DIR", "").strip()
    key = (tp_rank, int(layer_id))
    if not trace_dir or key in _traced_routes:
        return
    _traced_routes.add(key)
    path = Path(trace_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "tp_rank": tp_rank,
        "layer_id": int(layer_id),
        "q_group": q_group.detach().cpu(),
        "mla_heads_per_group": mla_heads_per_group,
        "candidates": candidates.detach().cpu(),
        **{name: value.detach().cpu() for name, value in features.items()},
        "score": score.detach().cpu(),
        "choice": choice.detach().cpu(),
        "output": output.detach().cpu(),
    }
    torch.save(payload, path / f"route_L{int(layer_id):02d}_rank{tp_rank}.pt")


def mark_router_bypassed(forward_batch: Any, layer_id: int) -> None:
    """Record that this layer already emitted final per-pair indices.

    The dense warmup below ``min_seq_len`` shares one index list across heads and
    never runs MISA, so there are no candidate lists to assign.  It still uses the
    per-head tensor contract, which routing would otherwise mistake for candidates.
    """
    cache = getattr(forward_batch, "offline_router_bypass", None)
    if cache is None:
        cache = set()
        setattr(forward_batch, "offline_router_bypass", cache)
    cache.add(int(layer_id))


def stash_gate(forward_batch: Any, layer_id: int, gate: torch.Tensor) -> None:
    """Retain the indexer's 64-way gate until the MLA query is available."""
    cache = getattr(forward_batch, "offline_router_gates", None)
    if cache is None:
        cache = {}
        setattr(forward_batch, "offline_router_gates", cache)
    cache[int(layer_id)] = gate.detach()


def _pop_gate(forward_batch: Any, layer_id: int) -> torch.Tensor:
    cache = getattr(forward_batch, "offline_router_gates", None)
    if cache is None or int(layer_id) not in cache:
        raise RuntimeError(f"missing offline-router gate for layer {layer_id}")
    return cache.pop(int(layer_id))


def stash_selected_heads(
    forward_batch: Any,
    layer_id: int,
    selected_heads: torch.Tensor,
    *,
    total_rows: int,
) -> None:
    """Retain the query-dependent MISA set until MLA pair assignment."""
    if selected_heads.ndim != 2 or selected_heads.shape[1] != budget():
        raise ValueError(
            "MISA selected heads must have shape "
            f"[n, {budget()}], got {tuple(selected_heads.shape)}"
        )
    if selected_heads.shape[0] > total_rows:
        raise ValueError(
            f"MISA selected rows {selected_heads.shape[0]} exceed total {total_rows}"
        )
    if selected_heads.shape[0] < total_rows:
        pad = torch.zeros(
            total_rows - selected_heads.shape[0],
            selected_heads.shape[1],
            dtype=selected_heads.dtype,
            device=selected_heads.device,
        )
        selected_heads = torch.cat([selected_heads, pad], dim=0)
    cache = getattr(forward_batch, "offline_router_selected_heads", None)
    if cache is None:
        cache = {}
        setattr(forward_batch, "offline_router_selected_heads", cache)
    cache[int(layer_id)] = selected_heads.detach()


def _pop_selected_heads(forward_batch: Any, layer_id: int) -> torch.Tensor:
    cache = getattr(forward_batch, "offline_router_selected_heads", None)
    if cache is None or int(layer_id) not in cache:
        raise RuntimeError(f"missing MISA selected heads for layer {layer_id}")
    return cache.pop(int(layer_id))


def stash_misa_features(
    forward_batch: Any,
    layer_id: int,
    indexer_q: torch.Tensor,
    gate: torch.Tensor,
    selection: "MISASelection",
    *,
    total_rows: int,
) -> None:
    """Retain candidate-wise MISA inputs until local MLA queries are available."""
    heads = selection.candidate_head_ids.long()
    if gate.ndim == 3 and gate.shape[-1] == 1:
        gate = gate.squeeze(-1)
    if indexer_q.ndim != 3 or indexer_q.shape[:2] != gate.shape:
        raise ValueError("indexer query and gate shapes disagree")
    gather_q = heads.unsqueeze(-1).expand(-1, -1, indexer_q.shape[-1])
    candidate_q = indexer_q.gather(1, gather_q)
    candidate_gate = gate.gather(1, heads)
    features = {
        "candidate_head_ids": heads,
        "candidate_q_indexer": candidate_q,
        "candidate_context_summary": selection.candidate_context_summary,
        "candidate_gate": candidate_gate,
        "candidate_importance": selection.candidate_importance,
        "candidate_context_stats": selection.candidate_context_stats,
    }
    for name, value in tuple(features.items()):
        if value.shape[0] < total_rows:
            pad = torch.zeros(
                (total_rows - value.shape[0], *value.shape[1:]),
                dtype=value.dtype,
                device=value.device,
            )
            features[name] = torch.cat([value, pad], dim=0)
    cache = getattr(forward_batch, "offline_router_misa_features", None)
    if cache is None:
        cache = {}
        setattr(forward_batch, "offline_router_misa_features", cache)
    cache[int(layer_id)] = {name: value.detach() for name, value in features.items()}


def _pop_misa_features(forward_batch: Any, layer_id: int) -> dict[str, torch.Tensor]:
    cache = getattr(forward_batch, "offline_router_misa_features", None)
    if cache is None or int(layer_id) not in cache:
        raise RuntimeError(f"missing MISA candidate features for layer {layer_id}")
    return cache.pop(int(layer_id))


@torch.no_grad()
def route_candidates(
    *,
    layer_id: int,
    q_mla: torch.Tensor,
    candidates: torch.Tensor,
    forward_batch: Any,
) -> torch.Tensor:
    """Assign each local MLA group to one globally budgeted candidate list.

    Args:
        q_mla: ``[tokens, local_mla_heads, 576]`` absorbed MLA query.
        candidates: ``[tokens, M, topk]`` picks for the configured unique set.
    Returns:
        ``[tokens, local_mla_pairs, topk]`` request-relative token positions.
    """
    configured_heads = candidate_head_ids(layer_id)
    if configured_heads is None:
        return candidates
    bypass = getattr(forward_batch, "offline_router_bypass", None)
    if bypass is not None and int(layer_id) in bypass:
        bypass.discard(int(layer_id))
        return candidates
    config = _load_config()
    if candidates.ndim != 3 or candidates.shape[1] != config.budget:
        raise ValueError(
            f"layer {layer_id} candidates must have shape [n, {config.budget}, k], "
            f"got {tuple(candidates.shape)}"
        )
    if q_mla.ndim != 3 or q_mla.shape[0] != candidates.shape[0]:
        raise ValueError(
            f"q/candidate shape mismatch: {tuple(q_mla.shape)} vs {tuple(candidates.shape)}"
        )
    if q_mla.shape[1] % 2:
        raise ValueError(f"MLA head count must be even, got {q_mla.shape[1]}")

    local_pairs = q_mla.shape[1] // 2
    from sglang.srt.layers.dp_attention import (
        get_attention_tp_rank,
        get_attention_tp_size,
    )

    tp_rank = get_attention_tp_rank()
    tp_size = get_attention_tp_size()
    if 64 % tp_size:
        raise ValueError(f"64 MLA pairs are not divisible by attention TP={tp_size}")
    expected_local_pairs = 64 // tp_size
    if local_pairs != expected_local_pairs:
        raise ValueError(
            f"local pair count {local_pairs} disagrees with attention TP={tp_size}"
        )
    pair_ids = torch.arange(
        tp_rank * local_pairs,
        (tp_rank + 1) * local_pairs,
        dtype=torch.long,
        device=q_mla.device,
    )

    if config.assignment_mode == "static":
        assert config.static_assignment_by_layer is not None
        assignment = config.static_assignment_by_layer[int(layer_id)]
        if config.mla_heads_per_group == 16:
            if tp_size != 8 or q_mla.shape[1] != 16:
                raise ValueError(
                    "TP-local static Group16 requires attention TP=8 with "
                    "16 local MLA heads"
                )
            expected_head = assignment[tp_rank]
            if configured_heads != (expected_head,) or candidates.shape[1] != 1:
                raise ValueError(
                    f"layer {layer_id} rank {tp_rank} must scan only static "
                    f"Indexer {expected_head}"
                )
            # The attention interface still has eight local two-head slots.
            # Reuse the single already-final candidate list without a router.
            return candidates.expand(-1, local_pairs, -1).contiguous()
        slot_by_head = {head: slot for slot, head in enumerate(configured_heads)}
        local_slots = [
            slot_by_head[assignment[pair_id]]
            for pair_id in range(
                tp_rank * local_pairs, (tp_rank + 1) * local_pairs
            )
        ]
        choice = torch.tensor(local_slots, dtype=torch.long, device=q_mla.device)
        return candidates.index_select(1, choice).contiguous()

    checkpoint = _load_checkpoint()
    layer_to_id = {
        int(layer): index for index, layer in enumerate(checkpoint["layers"])
    }
    checkpoint_layer_id = layer_to_id[int(layer_id)]
    if checkpoint.get("model_class") == "MISAAssignmentRouter":
        group_size = int(checkpoint.get("mla_heads_per_group", 2))
        if group_size == 16 and (tp_size != 8 or q_mla.shape[1] != 16):
            raise ValueError("TP-local group16 requires attention TP=8 with 16 local MLA heads")
        local_groups = q_mla.shape[1] // group_size
        features = _pop_misa_features(forward_batch, layer_id)
        selected_tensor = features["candidate_head_ids"].to(
            device=q_mla.device, dtype=torch.long
        )
        stashed_heads = _pop_selected_heads(forward_batch, layer_id).to(
            device=q_mla.device, dtype=torch.long
        )
        if not torch.equal(selected_tensor, stashed_heads):
            raise ValueError("MISA candidate feature heads disagree with selected heads")
        expected_q_dim = int(checkpoint["model_config"]["q_dim"])
        q_group = q_mla.reshape(q_mla.shape[0], local_groups, -1)
        if q_group.shape[-1] != expected_q_dim:
            raise ValueError(
                "router q_dim mismatch: "
                f"runtime={q_group.shape[-1]}, checkpoint={expected_q_dim}"
            )
        score = _model_for(q_mla.device)(
            q_group.float(),
            selected_tensor,
            features["candidate_q_indexer"].float(),
            features["candidate_context_summary"].float(),
            features["candidate_gate"].float(),
            features["candidate_importance"].float(),
            features["candidate_context_stats"].float(),
            torch.full(
                (q_mla.shape[0],),
                checkpoint_layer_id,
                dtype=torch.long,
                device=q_mla.device,
            ),
        )
        choice = score.argmax(-1)
        gather = choice.unsqueeze(-1).expand(-1, -1, candidates.shape[-1])
        expanded = candidates.unsqueeze(1).expand(-1, local_groups, -1, -1)
        output = (
            expanded.gather(2, gather.unsqueeze(2))
            .squeeze(2)
            .contiguous()
        )
        # The attention kernel still exposes two-head slots. Repeat only token
        # indices locally: all eight slots consume this rank's single decision.
        if group_size == 16:
            output = output.repeat_interleave(8, dim=1)
        _trace_misa_route(
            tp_rank=tp_rank,
            layer_id=layer_id,
            q_group=q_group,
            mla_heads_per_group=group_size,
            candidates=candidates,
            features=features,
            score=score,
            choice=choice,
            output=output,
        )
        return output

    state = _state_for(q_mla.device)
    expected_q_dim = int(checkpoint["model_config"]["q_dim"])
    q_pair = q_mla.reshape(q_mla.shape[0], local_pairs, 2, -1).reshape(
        q_mla.shape[0], local_pairs, -1
    )
    if q_pair.shape[-1] != expected_q_dim:
        raise ValueError(
            f"router q_dim mismatch: runtime={q_pair.shape[-1]}, checkpoint={expected_q_dim}"
        )
    q_pair = F.layer_norm(q_pair.float(), (q_pair.shape[-1],))

    pair_latent = F.linear(q_pair, state["q_proj.weight"])
    pair_latent = pair_latent + state["layer_embedding.weight"][
        checkpoint_layer_id
    ].view(1, 1, -1)
    dynamic = torch.einsum(
        "npr,ir->npi", pair_latent, state["indexer_embedding"]
    )

    gate = _pop_gate(forward_batch, layer_id).float()
    if gate.ndim == 3 and gate.shape[-1] == 1:
        gate = gate.squeeze(-1)
    if gate.ndim != 2 or gate.shape != (q_mla.shape[0], 64):
        raise ValueError(
            f"router gate must have shape {(q_mla.shape[0], 64)}, got {tuple(gate.shape)}"
        )
    gate = (gate - gate.mean(-1, keepdim=True)) / gate.std(
        -1, keepdim=True
    ).clamp_min(1e-5)

    score = (
        dynamic
        + state["prior"][checkpoint_layer_id].view(1, 1, 64)
        + state["pair_prior"][checkpoint_layer_id, pair_ids].unsqueeze(0)
        + gate.unsqueeze(1) * state["gate_scale"].view(1, 1, 64)
    )
    if config.head_selection_mode == "misa":
        selected_tensor = _pop_selected_heads(forward_batch, layer_id).to(
            device=q_mla.device, dtype=torch.long
        )
        if selected_tensor.shape != (q_mla.shape[0], config.budget):
            raise ValueError(
                "MISA selected-head shape mismatch: expected "
                f"{(q_mla.shape[0], config.budget)}, got {tuple(selected_tensor.shape)}"
            )
        if bool(((selected_tensor < 0) | (selected_tensor >= 64)).any().item()):
            raise ValueError("MISA selected heads must be in [0, 64)")
        selected_score = score.gather(
            -1,
            selected_tensor.unsqueeze(1).expand(-1, local_pairs, -1),
        )
    else:
        selected_tensor = torch.tensor(
            configured_heads, dtype=torch.long, device=q_mla.device
        )
        selected_score = score.index_select(-1, selected_tensor)
    choice = selected_score.argmax(-1)
    gather = choice.unsqueeze(-1).expand(-1, -1, candidates.shape[-1])
    expanded = candidates.unsqueeze(1).expand(-1, local_pairs, -1, -1)
    return expanded.gather(2, gather.unsqueeze(2)).squeeze(2).contiguous()


def reset_for_tests() -> None:
    """Clear process-global caches; only intended for unit tests."""
    global _config, _checkpoint
    _config = None
    _checkpoint = None
    _device_states.clear()
    _device_models.clear()
    _traced_routes.clear()
