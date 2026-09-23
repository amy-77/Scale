"""CUDA leaf scoring and compact gather. Compilation is lazy and optional."""

from __future__ import annotations

import os
from pathlib import Path

import torch

_EXT = None
_FAILED = False


def extension():
    global _EXT, _FAILED
    if _EXT is not None or _FAILED or not torch.cuda.is_available():
        return _EXT
    try:
        from torch.utils.cpp_extension import load

        src = Path(__file__).resolve().parent / "csrc" / "adaptive_select.cu"
        _EXT = load(
            name="adaptive_hisa_select",
            sources=[str(src)],
            extra_cuda_cflags=["-O3"],
            verbose=bool(int(os.environ.get("SGLANG_NSA_ADAPTIVE_HISA_VERBOSE", "0"))),
        )
    except Exception:
        _FAILED = True
        _EXT = None
    return _EXT


def score_leaves(query: torch.Tensor, weights: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    ext = extension()
    if ext is None:
        from sglang.srt.layers.attention.nsa.adaptive_hisa.legacy.reference import leaf_scores

        rows = []
        for i in range(query.shape[0]):
            rows.append(leaf_scores(mean, query[i], weights[i]))
        return torch.stack(rows, dim=0)
    return ext.score_leaves(query, weights, mean)[0]


def gather_compact(buf: torch.Tensor, token_ids: torch.Tensor, page_table: torch.Tensor, page_size: int, dim: int):
    ext = extension()
    if ext is None:
        from sglang.srt.layers.attention.nsa.adaptive_hisa.legacy.reference import gather_compact as gather_ref

        return gather_ref(buf.cpu(), page_table.cpu(), token_ids.cpu(), page_size=page_size, dim=dim)
    return ext.gather_compact(buf, token_ids, page_table, page_size, dim)
