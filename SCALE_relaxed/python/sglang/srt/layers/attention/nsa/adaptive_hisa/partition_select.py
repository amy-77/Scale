"""Exact radix *selection* replacing threshold-only ``torch.sort`` calls.

Two places in the P-key partition sorted a whole array just to read one
threshold: the target merge (k cheapest Ward edges per round, twice per round)
and the split repair (the ``deficit`` lexicographically smallest staircase
keys). ``csrc/radix_select.cu`` selects the k-th element with shared-memory
histograms: keys are read once per digit and only a scalar (or a 1-byte
mask) is written back, versus several full HBM rewrites of keys+indices for a
radix sort (11-bit digits: 6 passes for 64-bit keys). Results are
bit-identical to the sort path (see the parity tests).

The CUDA op is JIT-compiled on first use (same convention as
``decode_select.py``). If compilation is unavailable the callers fall back to
the sort-based implementation, so this module is an accelerator, not a
dependency.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SCRATCH_WORDS = 2055  # kMaxKeys + 3 + kRadix in radix_select.cu
_STATE: dict = {"loaded": None}
_SCRATCH: dict[torch.device, torch.Tensor] = {}


def _try_load() -> bool:
    if _STATE["loaded"] is not None:
        return _STATE["loaded"]
    if os.environ.get("ADAPTIVE_HISA_DISABLE_RADIX_SELECT", "0") == "1":
        _STATE["loaded"] = False
        return False
    try:
        from torch.utils.cpp_extension import load

        load(
            name="adaptive_hisa_radix_select",
            sources=[str(_HERE / "csrc" / "radix_select.cu")],
            extra_cuda_cflags=["-O3"],
            is_python_module=False,
            verbose=bool(int(os.environ.get("ADAPTIVE_HISA_SELECT_VERBOSE", "0"))),
        )
        _STATE["loaded"] = True
    except Exception as exc:  # pragma: no cover - toolchain dependent
        _STATE["loaded"] = False
        _STATE["error"] = repr(exc)
    return _STATE["loaded"]


def available(tensor: torch.Tensor) -> bool:
    """True when the CUDA selection op can serve ``tensor``."""
    return tensor.is_cuda and _try_load()


def _ops():
    if not _try_load():
        raise RuntimeError(f"adaptive_hisa_radix_select unavailable: {_STATE.get('error')}")
    return torch.ops.adaptive_hisa_radix_select


def _scratch(device: torch.device) -> torch.Tensor:
    buf = _SCRATCH.get(device)
    if buf is None:
        buf = torch.zeros(_SCRATCH_WORDS, dtype=torch.int64, device=device)
        _SCRATCH[device] = buf
    else:
        buf.zero_()
    return buf


def kth_threshold_f64(cost: torch.Tensor, k: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``[1]`` fp64 successor of the k-th smallest ``cost`` (``-inf`` if ``k <= 0``).

    Same contract as ``partition_gpu.kth_cost_threshold``: ``cost < thr``
    admits the ``k`` cheapest entries, boundary ties included, ``k`` clamped
    to ``cost.numel()``. ``k`` is a device int64 scalar; no host sync.
    """
    cost = cost.reshape(-1)
    if cost.dtype != torch.float64:
        cost = cost.to(torch.float64)
    cost = cost.contiguous()
    k = k.to(torch.int64).reshape(1)
    if out is None:
        out = torch.empty(1, dtype=torch.float64, device=cost.device)
    _ops().kth_threshold_f64(cost, k, out, _scratch(cost.device))
    return out


def lex_select_mask(
    keys: torch.Tensor, valid: torch.Tensor | None, k: torch.Tensor, *, key_bits: int = 64
) -> torch.Tensor:
    """Bool ``[n]``: ``valid`` rows whose key tuple is lexicographically <= the k-th smallest valid tuple.

    ``keys`` is int64 ``[n, n_keys]`` (any strides), most significant key first.
    ``key_bits < 64`` promises non-negative keys below ``2**key_bits`` and
    skips the unused high digits. With unique keys among valid rows exactly
    ``min(k, n_valid)`` rows are set; ``k <= 0`` yields an all-false mask.
    """
    n = keys.shape[0]
    k = k.to(torch.int64).reshape(1)
    if valid is not None and valid.dtype != torch.uint8 and valid.dtype != torch.bool:
        valid = valid != 0
    if valid is not None:
        valid = valid.contiguous()
    mask = torch.empty(n, dtype=torch.uint8, device=keys.device)
    _ops().lex_select_mask(keys, valid, k, mask, _scratch(keys.device), int(key_bits))
    return mask.view(torch.bool)
