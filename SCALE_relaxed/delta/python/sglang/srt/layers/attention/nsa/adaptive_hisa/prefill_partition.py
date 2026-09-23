"""P-key partition of a sealed prefix: shared pieces and the CPU-reference build.

The energy rows are dequantised keys ``[D, N]``. Every dyadic node stores the
Key-SSE ``sum_s ||k_s - mean(k)||²`` used by split and Ward merge.

The tree is a forest of ``root``-token roots (default 256). Every root shares
one summary budget ``M0 = N_complete / summary_compression`` (default 8), so a
flat root can stay coarse while a spiky root takes more leaves. Split leaves
are dyadic and atom-aligned. The optional merge produces arbitrary
atom-aligned lengths. Tokens after the last complete root are not partitioned.

Coordinates are logical token ids. Physical 64-token pages are resolved before
this module sees the keys.

Two backends share the contract:

* ``cpu_reference`` (this module): GPU scorer rows → FP64 score-moment tree on
  the device → one D2H copy → numba λ-DP / repair / merge on the host. This is
  the path that was A/B'ed as ``v7_reuse``.
* ``gpu`` (``partition_gpu``): the same tree, λ search, repair, merge and the
  FP8 summaries stay on the current CUDA stream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa.config import (
    ATOM,
    ROOT,
    PartitionConfig,
    get_config,
    partition_validate_enabled,
)
from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_reference import (
    _dp_solve,
    _leaf_totals,
    flatten_energies,
    lambda_dp_repair,
    merge_reference,
    warmup_host_kernels,
)

__all__ = [
    "PrefillPartition",
    "PartitionInputs",
    "build_prefill_partition_from_keys",
    "greedy_adjacent_merge",
    "key_values",
    "lambda_dp_repair",
    "n_complete_tokens",
    "partition_cpu_phase",
    "partition_keys_gpu_phase",
    "scale_bucket_histogram",
    "scale_histogram",
    "score_energies",
    "validate_leaves",
    "warmup_host_kernels",
]


@dataclass
class PrefillPartition:
    leaf_start: torch.Tensor  # int32 [M], logical token
    leaf_len: torch.Tensor  # int32 [M]
    n_complete: int
    tail_len: int
    n_tokens: int
    method: str = "P-key"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_complete == 0:
            if self.leaf_start.numel() or self.leaf_len.numel():
                raise ValueError("an empty sealed prefix cannot have leaves")
            return
        if partition_validate_enabled():
            validate_leaves(
                self.leaf_start,
                self.leaf_len,
                self.n_complete,
                int(self.meta.get("atom", ATOM)),
                allow_arbitrary=bool(self.meta.get("arbitrary_leaves", False)),
                max_len=int(self.meta.get("max_leaf_len", 0)),
            )

    @property
    def num_leaves(self) -> int:
        return int(self.leaf_start.numel())

    # Histograms are computed on demand: the worker thread must not spend GIL
    # time on Python bookkeeping while the model thread is launching kernels.
    def histogram(self) -> dict[int, dict[str, int]]:
        return scale_histogram(self.leaf_len)

    def bucket_histogram(self) -> dict[str, dict[str, int]]:
        return scale_bucket_histogram(self.leaf_len)


def n_complete_tokens(n_tokens: int, root: int = ROOT) -> int:
    return (int(n_tokens) // root) * root


def validate_leaves(
    leaf_start: torch.Tensor,
    leaf_len: torch.Tensor,
    n_complete: int,
    atom: int = ATOM,
    *,
    allow_arbitrary: bool = False,
    max_len: int = 0,
    root: int = ROOT,
) -> None:
    """Coverage / alignment check. ``max_len=0`` means uncapped (merged leaves)."""
    if leaf_start.shape != leaf_len.shape or leaf_start.dim() != 1:
        raise ValueError("leaf arrays must be 1-D and the same length")
    if n_complete == 0:
        if leaf_start.numel():
            raise ValueError("empty prefix must have no leaves")
        return
    if leaf_start.numel() == 0:
        raise ValueError("non-empty prefix must have leaves")
    start = leaf_start.to(torch.long).cpu()
    length = leaf_len.to(torch.long).cpu()
    if int(start[0]) != 0:
        raise ValueError("partition must start at logical token 0")
    if not bool(torch.equal(start, torch.sort(start).values)):
        raise ValueError("leaves must be sorted by start")
    if bool((start % atom).any()) or bool((length % atom).any()) or bool((length <= 0).any()):
        raise ValueError("leaves must be positive and atom-aligned")
    cap = max_len if max_len > 0 else (0 if allow_arbitrary else root)
    if cap and bool((length > cap).any()):
        raise ValueError(f"leaf lengths must not exceed {cap}")
    if not allow_arbitrary:
        allowed = {atom * (1 << level) for level in range(32) if atom * (1 << level) <= root}
        if not set(torch.unique(length).tolist()).issubset(allowed):
            raise ValueError(f"leaf lengths must be dyadic in {sorted(allowed)}")
    ends = start + length
    if int(ends[-1]) != n_complete or not bool(torch.equal(ends[:-1], start[1:])):
        raise ValueError("leaves must cover the sealed prefix without gaps or overlap")


def scale_histogram(leaf_len: torch.Tensor) -> dict[int, dict[str, int]]:
    values, counts = torch.unique(leaf_len.to(torch.long).cpu(), return_counts=True)
    return {
        int(v): {"leaves": int(c), "tokens": int(v) * int(c)}
        for v, c in zip(values.tolist(), counts.tolist())
    }


def scale_bucket_histogram(leaf_len: torch.Tensor) -> dict[str, dict[str, int]]:
    """Compact histogram for arbitrary merged lengths."""
    bounds = (
        ("1", 1, 1),
        ("2", 2, 2),
        ("3", 3, 3),
        ("4", 4, 4),
        ("5-7", 5, 7),
        ("8", 8, 8),
        ("9-15", 9, 15),
        ("16", 16, 16),
        ("17-31", 17, 31),
        ("32", 32, 32),
        ("33-63", 33, 63),
        ("64", 64, 64),
        ("65-127", 65, 127),
        ("128", 128, 128),
        ("129-255", 129, 255),
        ("256", 256, 256),
        ("257-511", 257, 511),
        ("512+", 512, 1 << 31),
    )
    length = leaf_len.to(torch.long).cpu()
    out: dict[str, dict[str, int]] = {}
    for name, low, high in bounds:
        mask = (length >= low) & (length <= high)
        if bool(mask.any()):
            out[name] = {"leaves": int(mask.sum()), "tokens": int(length[mask].sum())}
    return out


# --------------------------------------------------------------------------- #
# Device-side pieces shared by both backends
# --------------------------------------------------------------------------- #


def key_values(k_fp8: torch.Tensor, k_scale: torch.Tensor, n_complete: int) -> torch.Tensor:
    """Dequantised keys as ``[D, n_complete]`` float32 value rows (P-key input).

    Uses the production dequantisation ``k = fp8 * scale``. Each key dimension
    is one value row consumed by :func:`score_energies` and the Ward merge.
    """
    if n_complete < 0 or n_complete > k_fp8.shape[0]:
        raise ValueError(f"n_complete={n_complete} outside keys {k_fp8.shape[0]}")
    if k_fp8.dim() != 2:
        raise ValueError(f"k_fp8 must be [N,D], got {tuple(k_fp8.shape)}")
    if k_fp8.dtype == torch.uint8:
        k_fp8 = k_fp8.view(torch.float8_e4m3fn)
    scale = k_scale.reshape(-1).to(torch.float32)
    if scale.shape[0] < n_complete:
        raise ValueError(f"k_scale has {scale.shape[0]} entries < n_complete={n_complete}")
    keys = k_fp8[:n_complete].to(torch.float32) * scale[:n_complete, None]
    return keys.T.contiguous()


def score_energies(scores: torch.Tensor, *, atom: int = ATOM, root: int = ROOT) -> list[torch.Tensor]:
    """Per-node Key-SSE over ``[D,N]`` dequantised key rows.

    The energy is ``sum_s ||k_s - mu_b||²``. ``N`` is a multiple of ``root``;
    adjacent nodes merge by summing ``sum`` and ``sum_sq`` moments.
    """
    if scores.dim() != 2:
        raise ValueError("scores must be [C,N]")
    _check_shape(scores.shape[1], atom, root)
    values = scores.to(torch.float64)
    dims, n_tokens = values.shape
    n_atoms = n_tokens // atom
    grouped = values.reshape(dims, n_atoms, atom)
    moment = grouped.sum(-1).transpose(0, 1).contiguous()
    moment_sq = grouped.square().sum(-1).transpose(0, 1).contiguous()
    count = torch.full((n_atoms,), float(atom), dtype=torch.float64, device=values.device)
    energies: list[torch.Tensor] = []
    length = atom
    while True:
        energies.append((moment_sq - moment.square() / count[:, None]).sum(-1))
        if length == root:
            return energies
        moment = moment[0::2] + moment[1::2]
        moment_sq = moment_sq[0::2] + moment_sq[1::2]
        count = count[0::2] + count[1::2]
        length *= 2


_flatten_energies = flatten_energies


def greedy_adjacent_merge(
    leaf_start: torch.Tensor,
    leaf_len: torch.Tensor,
    values: torch.Tensor,
    threshold: float,
    *,
    atom: int = ATOM,
    max_leaf_len: int = ROOT,
    policy: str = "heap_reference",
    rounds: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Host merge on CPU tensors (``heap_reference`` by default, or ``sync_nonoverlap``).

    The merge cost is the exact increase in Key-SSE:
    ``n_l*n_r/(n_l+n_r) * ||mean_l-mean_r||²``. ``max_leaf_len=0`` disables the cap.
    """
    if values.dim() != 2:
        raise ValueError("values must be [C,N]")
    if partition_validate_enabled():
        validate_leaves(leaf_start, leaf_len, values.shape[1], atom)
    count = int(leaf_start.numel())
    lam = float(threshold)
    meta = {"premerge_M": count, "merge_count": 0, "merge_threshold": lam, "merge_policy": policy}
    if count <= 1 or lam <= 0:
        return leaf_start.clone(), leaf_len.clone(), meta
    starts = leaf_start.to(torch.long).cpu().numpy()
    lengths = leaf_len.to(torch.long).cpu().numpy()
    scores = values.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    totals = _leaf_totals(scores, starts, lengths)
    out_start, out_len, info = merge_reference(
        policy, starts, lengths, totals, lam, rounds=rounds, max_merge_len=int(max_leaf_len)
    )
    meta.update(info)
    meta["arbitrary_leaves"] = True
    return torch.from_numpy(out_start), torch.from_numpy(out_len), meta


# --------------------------------------------------------------------------- #
# cpu_reference backend: GPU tree → D2H → numba host phase
# --------------------------------------------------------------------------- #


@dataclass
class PartitionInputs:
    """Host-side inputs produced by the device phase of the CPU-reference backend.

    ``ready`` is a CUDA event recorded after the D2H copies; ``None`` on CPU.
    ``gpu_events`` are optional ``(t0, t1, t2)`` timing events around key
    dequantisation and tree construction. ``ready`` is the only event needed
    for correctness.
    """

    n_tokens: int
    n_complete: int
    # Dequantised key rows [D, n_complete] used by the host Ward merge.
    scores: torch.Tensor | None
    energies: torch.Tensor | None  # flat float64 on host
    offs: np.ndarray | None
    ready: torch.cuda.Event | None
    gpu_events: tuple | None
    wall_t0: float


def partition_keys_gpu_phase(
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    n_tokens: int,
    *,
    atom: int = ATOM,
    root: int = ROOT,
    non_blocking: bool = False,
    profile: bool = False,
) -> PartitionInputs:
    """P-key device phase: dequantise the sealed keys, then the same moment tree.

    ``inputs.scores`` holds the ``[D, n_complete]`` key rows so the host merge
    computes the Ward cost on key means. The dequantisation is accounted as
    ``score_s`` so the timing fields keep their meaning across metrics.
    """
    n_tokens = int(n_tokens)
    if n_tokens < 0 or n_tokens > k_fp8.shape[0]:
        raise ValueError(f"n_tokens={n_tokens} outside key tensor {k_fp8.shape[0]}")
    wall_t0 = time.perf_counter()
    done = n_complete_tokens(n_tokens, root)
    if done == 0:
        return PartitionInputs(n_tokens, 0, None, None, None, None, None, wall_t0)
    use_cuda = k_fp8.device.type == "cuda"
    collect_timing = profile or not use_cuda
    t0 = _mark(k_fp8.device) if collect_timing else None
    values = key_values(k_fp8, k_scale, done)
    t1 = _mark(values.device) if collect_timing else None
    flat, offs = flatten_energies(score_energies(values, atom=atom, root=root))
    t2 = _mark(values.device) if collect_timing else None
    if use_cuda:
        values_h = values.to("cpu", non_blocking=non_blocking)
        flat_h = flat.to("cpu", non_blocking=non_blocking)
        ready = torch.cuda.Event()
        ready.record()
    else:
        values_h, flat_h, ready = values, flat, None
    gpu_events = (t0, t1, t2) if collect_timing else None
    return PartitionInputs(n_tokens, done, values_h, flat_h, offs, ready, gpu_events, wall_t0)


def _empty_partition(inputs: PartitionInputs, cfg: PartitionConfig) -> PrefillPartition:
    return PrefillPartition(
        torch.empty(0, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        0,
        inputs.n_tokens - inputs.n_complete,
        inputs.n_tokens,
        meta={
            "M": 0,
            "lambda": 0.0,
            "dp_leaves": 0,
            "repairs": 0,
            "atom": cfg.atom,
            "score_s": 0.0,
            "tree_s": 0.0,
            "dp_s": 0.0,
            "merge_s": 0.0,
            "total_s": 0.0,
            "wall_s": time.perf_counter() - inputs.wall_t0,
        },
    )


def partition_cpu_phase(
    inputs: PartitionInputs,
    *,
    atom: int | None = None,
    merge: bool | None = None,
    cfg: PartitionConfig | None = None,
) -> PrefillPartition:
    """λ-DP, exact-budget repair and optional merge on the host (numba)."""
    cfg = cfg or get_config()
    atom = cfg.atom if atom is None else int(atom)
    merge = cfg.merge_enabled if merge is None else bool(merge)
    policy = cfg.merge_policy if cfg.merge_policy != "off" else "heap_reference"
    done, n_tokens = inputs.n_complete, inputs.n_tokens
    tail = n_tokens - done
    if done == 0:
        return _empty_partition(inputs, cfg)
    if inputs.ready is not None:
        inputs.ready.synchronize()
    budget = done // cfg.summary_compression
    t2 = time.perf_counter()
    start, length, lam, dp_leaves, repairs = _dp_solve(
        inputs.energies.numpy(), inputs.offs, budget, atom, 64
    )
    meta = {
        "M": budget,
        "lambda": float(lam),
        "dp_leaves": int(dp_leaves),
        "repairs": int(repairs),
        "atom": atom,
        "backend": "cpu_reference",
        "metric": cfg.partition_metric,
    }
    t3 = time.perf_counter()
    split_len = length
    merge_target = cfg.merge_target(done)
    if merge and (lam > 0 or merge_target) and start.shape[0] > 1:
        totals = _leaf_totals(inputs.scores.numpy(), start, length)
        threshold = float(lam) * float(cfg.merge_alpha)
        start, length, info = merge_reference(
            policy,
            start,
            length,
            totals,
            threshold,
            rounds=cfg.merge_target_rounds if merge_target else cfg.merge_rounds,
            max_merge_len=cfg.max_merge_len,
            target=merge_target,
        )
        meta.update(info)
        meta.update(
            {
                "premerge_M": budget,
                "merge_threshold": threshold,
                "arbitrary_leaves": True,
                "max_leaf_len": cfg.max_merge_len,
            }
        )
    t4 = time.perf_counter()
    if inputs.gpu_events is None:
        score_s = tree_s = 0.0
    elif inputs.ready is not None:
        e0, e1, e2 = inputs.gpu_events
        score_s, tree_s = e0.elapsed_time(e1) / 1000.0, e1.elapsed_time(e2) / 1000.0
    else:
        e0, e1, e2 = inputs.gpu_events
        score_s, tree_s = e1 - e0, e2 - e1
    return PrefillPartition(
        torch.from_numpy(start).to(torch.int32),
        torch.from_numpy(length).to(torch.int32),
        done,
        tail,
        n_tokens,
        method=(
            f"{cfg.method_name}-{policy}" + (f"-target{cfg.merge_target_divisor}" if merge_target else "")
            if merge
            else cfg.method_name
        ),
        meta={
            **meta,
            "split_leaf_len": torch.from_numpy(split_len).to(torch.int32),
            "score_s": score_s,
            "tree_s": tree_s,
            "dp_s": t3 - t2,
            "merge_s": t4 - t3,
            "total_s": score_s + tree_s + (t4 - t2),
            "wall_s": time.perf_counter() - inputs.wall_t0,
        },
    )


def build_prefill_partition_from_keys(
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    n_tokens: int,
    *,
    atom: int | None = None,
    root: int | None = None,
    merge: bool | None = None,
    profile: bool = False,
    cfg: PartitionConfig | None = None,
) -> PrefillPartition:
    """Synchronous P-key build: dequantised keys, tree, one D2H, host phase."""
    cfg = cfg or get_config()
    inputs = partition_keys_gpu_phase(
        k_fp8,
        k_scale,
        n_tokens,
        atom=cfg.atom if atom is None else atom,
        root=cfg.root if root is None else root,
        profile=profile,
    )
    return partition_cpu_phase(inputs, atom=atom, merge=merge, cfg=cfg)


def _mark(device: torch.device):
    if device.type != "cuda":
        return time.perf_counter()
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _check_shape(n_tokens: int, atom: int, root: int) -> None:
    if root % atom or (root // atom) & (root // atom - 1):
        raise ValueError("root/atom must be a power of two")
    if n_tokens % root:
        raise ValueError(f"sealed length {n_tokens} is not a multiple of root {root}")
