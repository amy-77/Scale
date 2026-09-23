"""Leaf key summaries: segmented FP8 dequant → mean → production requant.

``leaf_key_means`` reads the raw ``k_fp8[N, 128]`` / ``k_scale[N]`` that the
official prefill already gathered (logical order), and returns one FP32 mean
per leaf. Only the final leaves are summarised; the tree never stores 128-D
means. ``requant_summaries`` applies the model's ``act_quant(..., scale_fmt)``
convention so summaries share the FP8 + FP32-scale layout of raw index keys.

Padding leaves (``len == 0``) produce zero means and are written as zeros.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - CPU-only environments
    _HAS_TRITON = False

FP8_MAX = 448.0


if _HAS_TRITON:

    @triton.jit
    def _leaf_mean_kernel(
        k_ptr,
        scale_ptr,
        start_ptr,
        len_ptr,
        out_ptr,
        BLOCK_T: tl.constexpr,
        D: tl.constexpr,
    ):
        leaf = tl.program_id(0)
        start = tl.load(start_ptr + leaf).to(tl.int64)
        length = tl.load(len_ptr + leaf).to(tl.int64)
        d = tl.arange(0, D)
        t = tl.arange(0, BLOCK_T)
        acc = tl.zeros((D,), dtype=tl.float32)
        for t0 in range(0, length, BLOCK_T):
            rows = start + t0 + t
            mask = (t0 + t) < length
            k = tl.load(k_ptr + rows[:, None] * D + d[None, :], mask=mask[:, None], other=0.0)
            s = tl.load(scale_ptr + rows, mask=mask, other=0.0)
            acc += tl.sum(k.to(tl.float32) * s.to(tl.float32)[:, None], axis=0)
        denom = tl.maximum(length, 1).to(tl.float32)
        tl.store(out_ptr + leaf * D + d, acc / denom)


def leaf_key_means(
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    leaf_start: torch.Tensor,
    leaf_len: torch.Tensor,
) -> torch.Tensor:
    """``[M_cap, 128]`` FP32 mean of dequantised keys per leaf (zeros for padding)."""
    if k_fp8.dim() != 2:
        raise ValueError(f"k_fp8 must be [N, D], got {tuple(k_fp8.shape)}")
    if k_fp8.dtype == torch.uint8:
        k_fp8 = k_fp8.view(torch.float8_e4m3fn)
    n_tokens, dim = k_fp8.shape
    scale = k_scale.reshape(-1)
    if scale.dtype == torch.uint8:
        scale = scale.view(torch.float32)
    if scale.numel() < n_tokens:
        raise ValueError("k_scale shorter than k_fp8")
    capacity = int(leaf_start.numel())
    out = torch.zeros((capacity, dim), dtype=torch.float32, device=k_fp8.device)
    if capacity == 0:
        return out
    if k_fp8.is_cuda and _HAS_TRITON and dim & (dim - 1) == 0:
        k_fp8 = k_fp8.contiguous()
        scale = scale.to(torch.float32).contiguous()
        _leaf_mean_kernel[(capacity,)](
            k_fp8,
            scale,
            leaf_start.contiguous(),
            leaf_len.contiguous(),
            out,
            BLOCK_T=64,
            D=dim,
        )
        return out
    # Torch fallback (CPU tests): prefix sums of dequantised keys.
    keys = k_fp8.to(torch.float32) * scale[:n_tokens, None].to(torch.float32)
    prefix = torch.zeros((n_tokens + 1, dim), dtype=torch.float64, device=keys.device)
    torch.cumsum(keys.to(torch.float64), dim=0, out=prefix[1:])
    start = leaf_start.to(torch.long).clamp(0, n_tokens)
    end = (leaf_start.to(torch.long) + leaf_len.to(torch.long)).clamp(0, n_tokens)
    sums = prefix[end] - prefix[start]
    denom = leaf_len.to(torch.float64).clamp_min(1.0)[:, None]
    return (sums / denom).to(torch.float32)


def _torch_act_quant(x: torch.Tensor, round_scale: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Replica of ``triton_kernel._act_quant_kernel`` for one 128-wide block."""
    x32 = x.to(torch.float32)
    amax = x32.abs().amax(dim=-1).clamp_min(1e-4)
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(amax / FP8_MAX)))
    else:
        scale = amax / FP8_MAX
    y = (x32 / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return y, scale.reshape(-1, 1)


def requant_summaries(
    means: torch.Tensor, scale_fmt: str | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 means → (fp8[M,128], fp32 scale[M,1]) with the production convention.

    The K path quantises bf16 keys, so means are rounded to bf16 first.
    """
    x = means.to(torch.bfloat16).contiguous()
    if x.is_cuda and _HAS_TRITON:
        from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

        fp8, scale = act_quant(x, x.shape[-1], scale_fmt)
        return fp8, scale.reshape(-1, 1).to(torch.float32).contiguous()
    return _torch_act_quant(x, scale_fmt is not None)
