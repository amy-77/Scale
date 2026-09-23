"""Raw DSA-indexer state dump for the offline Adaptive-HISA study.

Enabled by ``SGLANG_NSA_INDEXER_DUMP_DIR``.  For every indexer forward on TP rank 0
(prefill chunk or decode step) of the configured layers, writes one ``.pt`` file:

    {dir}/{rid}/L{layer:02d}/{mode}_{pos_first:07d}_{pos_last:07d}.pt

with

    layer_id, rid, mode ("prefill" | "decode"), seq_len (= max position + 1)
    positions      [T]        int32   absolute positions of the tokens in this forward
    k_bf16         [T, 128]   bf16    indexer key, post-RoPE + post-Hadamard (what act_quant sees)
    k_fp8_u8       [T, 128]   uint8   act_quant(k_bf16) e4m3 bits (what the KV cache stores)
    k_scale        [T]        f32
    q_positions    [Tq]       int32   subsampled query rows (prefill: position % stride == 0; decode: all)
    q_fp8_u8       [Tq, 64, 128] uint8
    q_scale        [Tq, 64]   f32
    weights        [Tq, 64]   f32     w_dg = raw_gate * H^-0.5 * q_scale * softmax_scale (as fed to fp8_mqa_logits)

Keys of the whole prefix are recovered offline by concatenating all files of a
``(rid, layer)`` sorted by position.  Nothing here changes the served result.

Environment:
    SGLANG_NSA_INDEXER_DUMP_DIR               output root (required)
    SGLANG_NSA_INDEXER_DUMP_LAYERS            "all" (default) or comma list, e.g. "0,15,30,45,60"
    SGLANG_NSA_INDEXER_DUMP_Q_STRIDE          prefill query-row stride (default 64)
    SGLANG_NSA_INDEXER_DUMP_MAX_DECODE_STEPS  decode steps kept per (rid, layer) (default 64)
    SGLANG_NSA_INDEXER_DUMP_RID_PREFIX        only dump requests whose rid starts with this (default "dump:")
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Optional

import torch

_FALSE = {"0", "false", "no", "off", ""}
_lock = threading.Lock()
_decode_steps: dict[tuple[str, int], int] = {}
_layer_cache: Optional[set[int] | str] = None


def enabled() -> bool:
    return bool(os.environ.get("SGLANG_NSA_INDEXER_DUMP_DIR", "").strip())


def _dump_dir() -> Path:
    return Path(os.environ["SGLANG_NSA_INDEXER_DUMP_DIR"])


def wants_layer(layer_id: int) -> bool:
    global _layer_cache
    if _layer_cache is None:
        raw = os.environ.get("SGLANG_NSA_INDEXER_DUMP_LAYERS", "all").strip().lower()
        _layer_cache = "all" if raw in ("", "all") else {int(v) for v in raw.split(",") if v.strip()}
    return _layer_cache == "all" or layer_id in _layer_cache  # type: ignore[operator]


def _q_stride() -> int:
    return max(1, int(os.environ.get("SGLANG_NSA_INDEXER_DUMP_Q_STRIDE", "64")))


def _max_decode_steps() -> int:
    return int(os.environ.get("SGLANG_NSA_INDEXER_DUMP_MAX_DECODE_STEPS", "64"))


def _rid_prefix() -> str:
    return os.environ.get("SGLANG_NSA_INDEXER_DUMP_RID_PREFIX", "dump:")


def _sanitize(rid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:=-]+", "_", rid)[:160]


def request_id(forward_batch) -> Optional[str]:
    rids = getattr(forward_batch, "rids", None)
    if rids is None or len(rids) != 1:
        return None
    rid = str(rids[0])
    if not rid.startswith(_rid_prefix()):
        return None
    return rid


def maybe_dump(
    *,
    forward_batch,
    layer_id: int,
    positions: torch.Tensor,
    key_bf16: torch.Tensor,
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    weights: torch.Tensor,
    act_quant,
    block_size: int,
    scale_fmt,
) -> None:
    """Called from ``Indexer.forward_cuda`` after q/k/weights are formed.  Never raises."""
    try:
        if not enabled() or not wants_layer(layer_id):
            return
        from sglang.srt.layers.dp_attention import get_attention_tp_rank
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        if get_is_capture_mode() or int(get_attention_tp_rank()) != 0:
            return
        rid = request_id(forward_batch)
        if rid is None:
            return
        mode_obj = forward_batch.forward_mode
        if mode_obj.is_extend_without_speculative():
            mode = "prefill"
        elif mode_obj.is_decode():
            mode = "decode"
        else:
            return
        T = int(positions.shape[0])
        if T == 0 or key_bf16.shape[0] != T or q_fp8.shape[0] != T:
            return
        if mode == "decode":
            with _lock:
                n = _decode_steps.get((rid, layer_id), 0)
                if n >= _max_decode_steps():
                    return
                _decode_steps[(rid, layer_id)] = n + 1
            q_rows = torch.arange(T, device=positions.device)
        else:
            q_rows = torch.nonzero(positions % _q_stride() == 0).flatten()
        k_fp8, k_scale = act_quant(key_bf16, block_size, scale_fmt)
        pos_cpu = positions.to(torch.int32).cpu()
        payload = {
            "layer_id": int(layer_id),
            "rid": rid,
            "mode": mode,
            "seq_len": int(pos_cpu.max().item()) + 1,
            "positions": pos_cpu,
            "k_bf16": key_bf16.detach().to(torch.bfloat16).cpu(),
            "k_fp8_u8": k_fp8.detach().view(torch.uint8).reshape(T, -1).cpu(),
            "k_scale": k_scale.detach().to(torch.float32).reshape(T, -1)[:, 0].cpu(),
            "q_positions": pos_cpu[q_rows.cpu()],
            "q_fp8_u8": q_fp8[q_rows].detach().view(torch.uint8).cpu(),
            "q_scale": q_scale[q_rows].detach().to(torch.float32).reshape(q_rows.numel(), -1).cpu(),
            "weights": weights[q_rows].detach().to(torch.float32).reshape(q_rows.numel(), -1).cpu(),
            "weights_semantics": "w_dg_includes_q_scale_H-0.5_D-0.5",
            "key_semantics": "post_rope64_post_hadamard_bf16; k_fp8 = act_quant(k_bf16)",
        }
        p0, p1 = int(pos_cpu.min().item()), int(pos_cpu.max().item())
        out_dir = _dump_dir() / _sanitize(rid) / f"L{layer_id:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / f".{mode}_{p0:07d}_{p1:07d}.pt.tmp"
        torch.save(payload, tmp)
        tmp.rename(out_dir / f"{mode}_{p0:07d}_{p1:07d}.pt")
    except Exception as exc:  # pragma: no cover - never break serving
        import logging

        logging.getLogger(__name__).warning("indexer dump failed (layer %s): %r", layer_id, exc)
