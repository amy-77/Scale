"""Adaptive-HISA (SCALE) configuration and algorithm contract.

Prefill contract fixed for Phase B (2026-09-19); adaptive_decode consumes the
resulting summaries in the B=1 decode pipeline without changing prefill:

* Prefill keeps the full official DSA indexer. Only the **final** prefill
  chunk of a request builds a partition of the sealed 256-token roots, for
  later decode. The partition never changes the prefill Top-2048.
* The only partition metric is ``key_sse`` (P-key), which is query-independent.
  The per-block energy is ``sum_s ||k_s - mu_b||^2 = |b| tr(Sigma_b)`` on the
  dequantised keys; merge uses the Ward cost on key means. No query tail or
  calibration-score path exists.
* One global budget ``M0 = N_complete / summary_compression`` shared by all
  roots; ``lambda_dp_repair`` = one λ for the forest, bisection to
  ``count <= M0``, then split-by-largest-gain repair to exactly ``M0``. This is
  an exact leaf-count solution, not the exact constrained-DP optimum.
* Merge: ``sync_nonoverlap`` = the sibling reference
  ``batched_adjacent_merge_rounds`` (E10): each round freezes the leaves,
  scores all adjacent pairs, greedily matches non-overlapping pairs by
  ``(cost, start)`` below ``merge_alpha * lambda``, merges them at once, and
  stops after ``merge_rounds`` rounds. ``heap_reference`` is the old
  sequential heap that keeps merging until no pair is below the threshold.
* Target-count merge (``merge_target_divisor = D > 0``): instead of the
  ``merge_alpha * lambda`` threshold, every round admits the
  ``2 * (count - target)`` cheapest adjacent pairs, matches them
  non-overlapping, and merges the ``count - target`` cheapest matched pairs,
  until the leaf count reaches ``n_complete // D`` (never overshot except by
  exact cost ties). A round can at most halve the count, so ``L/8 -> L/64``
  needs >= 3 rounds; measured 4-5 on the 128K dumps, ``merge_target_rounds``
  (default 8) caps it and idle rounds are cheap. ``D = 64`` matches HISA's
  chunk count at chunk size 64 so accuracy comparisons are not confounded by
  a higher chunk count.
* ``split_backend = gpu`` is the production path (tree, λ search, repair,
  merge and FP8 summaries on the current CUDA stream). ``cpu_reference`` is
  the numba path that was A/B'ed before (``v7_reuse``), kept as a reference
  and for the CPU overlap experiment.

Legacy environment variables are still honoured where their meaning did not
change. ``SGLANG_NSA_ADAPTIVE_HISA_PARTITION_MERGE=1`` used to select the heap
merge implicitly; it now requires an explicit ``..._MERGE_POLICY`` so the
two-round algorithm is never enabled by accident.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from functools import lru_cache

ATOM = 1
ROOT = 256
SUMMARY_COMPRESSION = 8  # M0 = N_complete / 8
CANDIDATE_TOKENS = 8192
INDEX_TOPK = 2048
FP8_MAX = 448.0
HEAD_DIM = 128
PAGE_SIZE = 64
# Decode guards (shared with the h20-1 accuracy reference): the first
# ``SINK_TOKENS`` and the most recent ``TAIL_TOKENS`` are always raw candidates
# and count toward ``CANDIDATE_TOKENS``. Generated tokens older than the tail
# are sealed into fixed ``DECODE_CHUNK``-token FP8 summaries.
SINK_TOKENS = 64
TAIL_TOKENS = 256
# Sealed decode generation leaves; 64 matches HISA's growth rate and shrinks
# the fixed workspace from ~L/8 toward ~L/64. Override with
# SGLANG_NSA_ADAPTIVE_HISA_DECODE_CHUNK for A/B (legacy accuracy runs used 8).
DECODE_CHUNK = 64
# Legacy constant used only by ``adaptive_hisa.legacy`` (radius prototype).
MIN_SEQ_LEN = 4096

MODES = ("off", "build_only", "adaptive_decode")
PARTITION_METRICS = ("key_sse",)
SPLIT_BACKENDS = ("gpu", "cpu_reference")
MERGE_POLICIES = ("off", "sync_nonoverlap", "heap_reference")
GPU_STREAMS = ("main", "side")

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}
_ENV = "SGLANG_NSA_ADAPTIVE_HISA_"


class PartitionConfigError(ValueError):
    """Raised for conflicting or unmigrated Adaptive-HISA settings."""


def _env(name: str) -> str | None:
    raw = os.environ.get(_ENV + name)
    return None if raw is None else raw.strip()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUE


def _parse_bool(name: str, raw: str) -> bool:
    low = raw.lower()
    if low in _TRUE:
        return True
    if low in _FALSE:
        return False
    raise PartitionConfigError(f"{_ENV}{name}={raw!r} is not a boolean")


@dataclass(frozen=True)
class PartitionConfig:
    mode: str = "off"
    partition_metric: str = "key_sse"
    atom: int = ATOM
    root: int = ROOT
    summary_compression: int = SUMMARY_COMPRESSION
    split_backend: str = "gpu"
    split_solver: str = "lambda_dp_repair"
    # GPU λ search: ``lambda_candidates`` DP evaluations per round (one fused
    # Triton launch) until the bracket is below ``lambda_rel_tol`` (relative to
    # the upper bound); 16 candidates -> 10 rounds for 1e-12 (FP64 is slow on H20,
    # so fewer candidates per round wins over fewer rounds).
    lambda_candidates: int = 16
    lambda_rel_tol: float = 1e-12
    # Hard cap on λ rounds (0 = derive from ``lambda_rel_tol`` only). 6 rounds
    # with 16 candidates (bracket width 17^-6 ~ 4e-8 relative) reproduced the
    # 10-round split and merge leaf-for-leaf on real 33K-128K indexer dumps;
    # 5 rounds already changed leaves. Each round is one full-tree fp64 DP.
    lambda_max_rounds: int = 6
    # Device-side early stop: once the feasible bracket end already yields
    # ``budget - count <= lambda_deficit_tol * budget`` DP leaves, later rounds
    # return immediately (launch count is unchanged for CUDA graphs; only the
    # full-tree DP work is skipped). The exact repair fills the remaining
    # deficit. ``0.0`` accepts only an exact count; ``< 0`` disables.
    lambda_deficit_tol: float = -1.0
    # Kept for compatibility: the exact repair is single-pass (lexicographic
    # staircase order, no host read) and ignores this value.
    repair_tie_passes: str = "auto"
    merge_policy: str = "sync_nonoverlap"
    merge_rounds: int = 2
    merge_alpha: float = 1.0
    # 0 = no cap (the E10 reference that produced 9,166 / 83.52% had none).
    # The earlier heap A/B (``v7_reuse``) used 256; set it explicitly for that.
    max_merge_len: int = 0
    # 0 = threshold merge (merge_alpha * lambda, merge_rounds rounds). D > 0 =
    # merge the cheapest pairs down to n_complete // D leaves.
    merge_target_divisor: int = 0
    merge_target_rounds: int = 8
    candidate_tokens: int = CANDIDATE_TOKENS
    index_topk: int = INDEX_TOPK
    # Decode guards and sealing granularity (see module constants).
    sink_tokens: int = SINK_TOKENS
    tail_tokens: int = TAIL_TOKENS
    decode_chunk: int = DECODE_CHUNK
    # First N layers retain official full-context scoring during decode.
    # 0 = every indexer layer uses Adaptive-HISA (aligned with h20-1).
    fallback_layers: int = 0
    build_summaries: bool = True
    # gpu backend: capture split+merge once per (C, N_complete) as a CUDA graph
    # and replay it for every layer of the request (host cost per layer drops
    # from ~200 launches to one replay). Falls back to eager if capture fails.
    graph_build: bool = True
    # raw-fp8 builder: keep each (layer, request)'s Key-SSE tree between prefill
    # chunks and only add the nodes of the newly sealed roots (the prefix's
    # dyadic nodes are unchanged); λ/Split/Merge still run on the full tree.
    tree_cache: bool = True
    # gpu backend: "main" runs the build in-stream (its GPU time lands on the
    # critical path of the last chunk: +0.14 s at 32K); "side" issues it on a
    # second stream that waits on the scores and is joined at forward end, so
    # it overlaps the remaining layers (+0.04 s at 32K, same as the CPU
    # overlap baseline). Overlap depends on SM headroom; GPU_STREAM=main reverts.
    gpu_stream: str = "side"
    cpu_overlap: bool = False
    # Adaptive sparse prefill: every prefill chunk builds the partition of its
    # sealed prefix (not only the final chunk), and the next chunk's indexer
    # scores summaries first, expands the best leaves to ``candidate_tokens``
    # raw tokens, adds the causal local window, and takes Top-2048 from that
    # candidate set instead of the dense ``[n_q, N]`` DSA logits.
    sparse_prefill: bool = False
    # Query rows per sparse-prefill sub-batch (bounds the candidate/logit
    # workspaces: rows x (candidate_tokens + chunk) x 8 bytes).
    sparse_prefill_rows: int = 2048
    # Raw-token leaf budget for the prefill two-level selection; 0 = use
    # ``candidate_tokens`` (the decode budget). Prefill amortises the fine
    # pass over 8192 query rows, so a larger budget is cheap there: on a 128K
    # dump 16384 lifts Top-2048 recall 0.886 -> 0.945 for ~+16 ms/layer-chunk.
    sparse_prefill_candidates: int = 0
    # The final prompt chunk produces the first output token (RULER NIAH is
    # decided there). ``dense_final`` keeps the dense DSA indexer for that chunk;
    # ``final_candidates`` (0 = same as prefill budget) gives it a larger budget.
    sparse_prefill_dense_final: bool = False
    sparse_prefill_final_candidates: int = 0
    profile: bool = False

    @property
    def prefill_candidate_tokens(self) -> int:
        return self.sparse_prefill_candidates or self.candidate_tokens

    @property
    def final_candidate_tokens(self) -> int:
        return self.sparse_prefill_final_candidates or self.prefill_candidate_tokens

    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def merge_enabled(self) -> bool:
        return self.enabled and self.merge_policy != "off"

    @property
    def method_name(self) -> str:
        return "P-key"

    def merge_target(self, n_complete: int) -> int:
        """Post-merge leaf-count target for ``n_complete`` sealed tokens (0 = threshold merge)."""
        if not self.merge_target_divisor:
            return 0
        return max(1, int(n_complete) // int(self.merge_target_divisor))

    @property
    def energy_rows(self) -> int:
        """Rows of the P-key energy input: one per key dimension."""
        return HEAD_DIM

    @property
    def levels(self) -> int:
        return (self.root // self.atom).bit_length()

    def validate(self) -> "PartitionConfig":
        if self.mode not in MODES:
            raise PartitionConfigError(f"mode={self.mode!r} not in {MODES}")
        if self.fallback_layers < 0:
            raise PartitionConfigError("FALLBACK_LAYERS must be non-negative")
        if self.index_topk != INDEX_TOPK:
            raise PartitionConfigError("INDEX_TOPK must remain 2048")
        if self.candidate_tokens < self.index_topk or self.candidate_tokens % PAGE_SIZE:
            raise PartitionConfigError("CANDIDATE_TOKENS must be a multiple of 64 and >= 2048")
        if self.sink_tokens < 0 or self.tail_tokens < 0:
            raise PartitionConfigError("SINK_TOKENS / TAIL_TOKENS must be non-negative")
        if self.sink_tokens + self.tail_tokens + self.index_topk > self.candidate_tokens:
            raise PartitionConfigError(
                "SINK_TOKENS + TAIL_TOKENS + INDEX_TOPK must fit in CANDIDATE_TOKENS"
            )
        if self.decode_chunk < 1 or PAGE_SIZE % self.decode_chunk:
            raise PartitionConfigError("DECODE_CHUNK must divide the 64-token page")
        if self.mode == "adaptive_decode" and (self.split_backend != "gpu" or not self.build_summaries):
            raise PartitionConfigError("adaptive_decode requires GPU partitions and FP8 summaries")
        if self.partition_metric not in PARTITION_METRICS:
            raise PartitionConfigError(
                f"partition_metric={self.partition_metric!r} not in {PARTITION_METRICS}"
            )
        if self.split_solver != "lambda_dp_repair":
            raise PartitionConfigError("split_solver must be lambda_dp_repair")
        if self.split_backend not in SPLIT_BACKENDS:
            raise PartitionConfigError(
                f"split_backend={self.split_backend!r} not in {SPLIT_BACKENDS}"
            )
        if self.merge_policy not in MERGE_POLICIES:
            raise PartitionConfigError(
                f"merge_policy={self.merge_policy!r} not in {MERGE_POLICIES}"
            )
        if self.atom < 1 or self.root % self.atom:
            raise PartitionConfigError(f"root={self.root} must be a multiple of atom={self.atom}")
        ratio = self.root // self.atom
        if ratio & (ratio - 1):
            raise PartitionConfigError("root/atom must be a power of two")
        if self.summary_compression < 1 or self.root % self.summary_compression:
            raise PartitionConfigError("summary_compression must divide root")
        if self.root // self.summary_compression < 1:
            raise PartitionConfigError("summary_compression larger than root")
        if self.lambda_candidates < 1:
            raise PartitionConfigError("lambda_candidates must be positive")
        if self.lambda_max_rounds < 0:
            raise PartitionConfigError("lambda_max_rounds must be >= 0")
        if self.merge_rounds < 0:
            raise PartitionConfigError("merge_rounds must be non-negative")
        if self.merge_alpha <= 0:
            raise PartitionConfigError("merge_alpha must be positive")
        if self.max_merge_len < 0:
            raise PartitionConfigError("max_merge_len must be >= 0 (0 = no cap)")
        if self.merge_target_divisor < 0:
            raise PartitionConfigError("merge_target_divisor must be >= 0 (0 = threshold merge)")
        if self.merge_target_divisor:
            if self.merge_target_divisor < self.summary_compression:
                raise PartitionConfigError(
                    "merge_target_divisor must be >= summary_compression "
                    f"({self.merge_target_divisor} < {self.summary_compression})"
                )
            if self.merge_policy != "sync_nonoverlap":
                raise PartitionConfigError("merge_target_divisor requires merge_policy=sync_nonoverlap")
            if self.merge_target_rounds < 1:
                raise PartitionConfigError("merge_target_rounds must be positive")
        if self.repair_tie_passes != "auto":
            try:
                if int(self.repair_tie_passes) < 0:
                    raise ValueError
            except ValueError:
                raise PartitionConfigError(
                    "repair_tie_passes must be 'auto' or a non-negative integer"
                ) from None
        if self.cpu_overlap and self.split_backend != "cpu_reference":
            raise PartitionConfigError(
                "PARTITION_OVERLAP (CPU worker overlap) only exists for "
                "SPLIT_BACKEND=cpu_reference; the gpu backend runs in-stream"
            )
        if self.build_summaries and self.split_backend != "gpu":
            raise PartitionConfigError(
                "FP8 summaries are produced by the gpu backend only; set "
                "BUILD_SUMMARIES=0 for cpu_reference"
            )
        if self.gpu_stream not in GPU_STREAMS:
            raise PartitionConfigError(f"GPU_STREAM must be one of {GPU_STREAMS}, got {self.gpu_stream!r}")
        if self.gpu_stream == "side" and self.split_backend != "gpu":
            raise PartitionConfigError("GPU_STREAM=side only applies to SPLIT_BACKEND=gpu")
        if self.sparse_prefill and (self.split_backend != "gpu" or not self.build_summaries):
            raise PartitionConfigError("SPARSE_PREFILL requires GPU partitions and FP8 summaries")
        if self.sparse_prefill_rows < 1:
            raise PartitionConfigError("SPARSE_PREFILL_ROWS must be positive")
        for name in ("sparse_prefill_candidates", "sparse_prefill_final_candidates"):
            v = getattr(self, name)
            if v < 0 or v % 256:
                raise PartitionConfigError(f"{name.upper()} must be 0 or a positive multiple of 256")
        if self.enabled:
            _refuse_conflicting_experiments()
        return self

    def describe(self) -> str:
        return (
            f"mode={self.mode} metric={self.partition_metric} atom={self.atom} "
            f"root={self.root} compression={self.summary_compression} "
            f"D={HEAD_DIM} backend={self.split_backend} "
            f"solver={self.split_solver} "
            f"lambda=K{self.lambda_candidates}/tol{self.lambda_rel_tol:g}"
            f"/max{self.lambda_max_rounds or 'auto'}/deficit{self.lambda_deficit_tol if self.lambda_deficit_tol >= 0 else 'off'} "
            f"merge={self.merge_policy} "
            f"rounds={self.merge_rounds} alpha={self.merge_alpha} "
            f"max_merge_len={self.max_merge_len or 'none'} "
            f"merge_target={('L/%d@%d' % (self.merge_target_divisor, self.merge_target_rounds)) if self.merge_target_divisor else 'off'} "
            f"summaries={self.build_summaries} "
            f"graph_build={self.graph_build} tree_cache={self.tree_cache} gpu_stream={self.gpu_stream} cpu_overlap={self.cpu_overlap} "
            f"fallback_layers={self.fallback_layers} candidate_tokens={self.candidate_tokens} "
            f"sink={self.sink_tokens} tail={self.tail_tokens} decode_chunk={self.decode_chunk} "
            f"sparse_prefill={self.sparse_prefill}@{self.sparse_prefill_rows}rows"
            f"/{self.prefill_candidate_tokens}cand"
            f"/final={'dense' if self.sparse_prefill_dense_final else self.final_candidate_tokens}"
        )


def _refuse_conflicting_experiments() -> None:
    """SCALE runs on the full 64-head DSA; it is not combined with MISA,
    per-head indexing or the offline head router. Refuse instead of mixing."""
    conflicts = []
    if os.environ.get("SGLANG_NSA_PER_HEAD_INDEX", "0").strip().lower() not in _FALSE:
        conflicts.append("SGLANG_NSA_PER_HEAD_INDEX")
    if os.environ.get("SGLANG_NSA_OFFLINE_ROUTER_CONFIG", "").strip():
        conflicts.append("SGLANG_NSA_OFFLINE_ROUTER_CONFIG")
    if os.environ.get("SGLANG_NSA_HEADMAP_PROBE_DIR", "").strip():
        conflicts.append("SGLANG_NSA_HEADMAP_PROBE_DIR")
    if conflicts:
        raise PartitionConfigError(
            "Adaptive-HISA partition mode cannot run together with "
            + ", ".join(conflicts)
            + "; disable them or set SGLANG_NSA_ADAPTIVE_HISA_MODE=off"
        )


def config_from_env() -> PartitionConfig:
    cfg = PartitionConfig()
    updates: dict = {}
    for removed in ("CALIB_QUERIES", "PARTITION_REUSE_LOGITS"):
        if _env(removed) is not None:
            raise PartitionConfigError(
                f"{_ENV}{removed} was removed with the P-qfull path; "
                "P-key does not use calibration queries or indexer logits"
            )

    mode = _env("MODE")
    legacy_partition = _env("PREFILL_PARTITION")
    if mode is not None and mode != "":
        updates["mode"] = mode.lower()
        if legacy_partition is not None and _parse_bool("PREFILL_PARTITION", legacy_partition) != (
            updates["mode"] != "off"
        ):
            raise PartitionConfigError(
                f"{_ENV}MODE={mode} conflicts with {_ENV}PREFILL_PARTITION={legacy_partition}"
            )
    elif legacy_partition is not None:
        updates["mode"] = "build_only" if _parse_bool("PREFILL_PARTITION", legacy_partition) else "off"

    for name, field in (("ATOM", "atom"), ("ROOT", "root"), ("SUMMARY_COMPRESSION", "summary_compression"),
                        ("LAMBDA_CANDIDATES", "lambda_candidates"),
                        ("MERGE_ROUNDS", "merge_rounds"), ("MAX_MERGE_LEN", "max_merge_len"),
                        ("MERGE_TARGET_DIVISOR", "merge_target_divisor"),
                        ("MERGE_TARGET_ROUNDS", "merge_target_rounds"),
                        ("CANDIDATE_TOKENS", "candidate_tokens"), ("INDEX_TOPK", "index_topk"),
                        ("FALLBACK_LAYERS", "fallback_layers"), ("SINK_TOKENS", "sink_tokens"),
                        ("TAIL_TOKENS", "tail_tokens"), ("DECODE_CHUNK", "decode_chunk")):
        raw = _env(name)
        if raw:
            try:
                updates[field] = int(raw)
            except ValueError:
                raise PartitionConfigError(f"{_ENV}{name}={raw!r} is not an integer") from None
    raw = _env("MERGE_ALPHA")
    if raw:
        updates["merge_alpha"] = float(raw)
    raw = _env("LAMBDA_REL_TOL")
    if raw:
        updates["lambda_rel_tol"] = float(raw)
    raw = _env("LAMBDA_MAX_ROUNDS")
    if raw:
        updates["lambda_max_rounds"] = int(raw)
    raw = _env("LAMBDA_DEFICIT_TOL")
    if raw:
        updates["lambda_deficit_tol"] = float(raw)
    raw = _env("REPAIR_TIE_PASSES")
    if raw:
        updates["repair_tie_passes"] = raw.lower()
    raw = _env("SPLIT_BACKEND")
    if raw:
        updates["split_backend"] = raw.lower()
    raw = _env("PARTITION_METRIC")
    if raw:
        updates["partition_metric"] = raw.lower()

    policy = _env("MERGE_POLICY")
    legacy_merge = _env("PARTITION_MERGE")
    if policy:
        updates["merge_policy"] = policy.lower()
        if legacy_merge is not None and _parse_bool("PARTITION_MERGE", legacy_merge) != (
            updates["merge_policy"] != "off"
        ):
            raise PartitionConfigError(
                f"{_ENV}MERGE_POLICY={policy} conflicts with {_ENV}PARTITION_MERGE={legacy_merge}"
            )
    elif legacy_merge is not None:
        if _parse_bool("PARTITION_MERGE", legacy_merge):
            raise PartitionConfigError(
                f"{_ENV}PARTITION_MERGE=1 no longer selects an algorithm. Set "
                f"{_ENV}MERGE_POLICY=heap_reference (old sequential heap, add "
                f"{_ENV}MAX_MERGE_LEN=256 to reproduce v7_reuse) or "
                f"{_ENV}MERGE_POLICY=sync_nonoverlap (two synchronous rounds)."
            )
        updates["merge_policy"] = "off"

    raw = _env("PARTITION_OVERLAP")
    if raw is not None:
        updates["cpu_overlap"] = _parse_bool("PARTITION_OVERLAP", raw)
    raw = _env("BUILD_SUMMARIES")
    if raw is not None:
        updates["build_summaries"] = _parse_bool("BUILD_SUMMARIES", raw)
    elif updates.get("split_backend", cfg.split_backend) != "gpu":
        updates["build_summaries"] = False
    raw = _env("GRAPH_BUILD")
    if raw is not None:
        updates["graph_build"] = _parse_bool("GRAPH_BUILD", raw)
    raw = _env("TREE_CACHE")
    if raw is not None:
        updates["tree_cache"] = _parse_bool("TREE_CACHE", raw)
    raw = _env("SPARSE_PREFILL")
    if raw is not None:
        updates["sparse_prefill"] = _parse_bool("SPARSE_PREFILL", raw)
    raw = _env("SPARSE_PREFILL_ROWS")
    if raw:
        updates["sparse_prefill_rows"] = int(raw)
    raw = _env("SPARSE_PREFILL_CANDIDATES")
    if raw:
        updates["sparse_prefill_candidates"] = int(raw)
    raw = _env("SPARSE_PREFILL_FINAL_CANDIDATES")
    if raw:
        updates["sparse_prefill_final_candidates"] = int(raw)
    raw = _env("SPARSE_PREFILL_DENSE_FINAL")
    if raw is not None:
        updates["sparse_prefill_dense_final"] = _parse_bool("SPARSE_PREFILL_DENSE_FINAL", raw)
    raw = _env("GPU_STREAM")
    if raw is not None:
        updates["gpu_stream"] = raw.lower()
    elif updates.get("split_backend", cfg.split_backend) != "gpu":
        updates["gpu_stream"] = "main"  # the stream choice is a gpu-backend knob
    updates["profile"] = forward_timing_enabled() or partition_layer_log_enabled()
    return replace(cfg, **updates).validate()


@lru_cache(maxsize=1)
def _cached_config() -> PartitionConfig:
    return config_from_env()


def get_config() -> PartitionConfig:
    """Process-wide configuration, resolved once from the environment."""
    return _cached_config()


def reset_config_cache() -> None:
    _cached_config.cache_clear()


# --------------------------------------------------------------------------- #
# Thin predicates kept for the existing hook sites.
# --------------------------------------------------------------------------- #


def partition_enabled() -> bool:
    """Build the prefill P-key partition (mode != off)."""
    return get_config().enabled


enabled = partition_enabled


def partition_overlap_enabled() -> bool:
    """Legacy CPU worker overlap (cpu_reference backend only)."""
    cfg = get_config()
    return cfg.enabled and cfg.cpu_overlap


def partition_merge_enabled() -> bool:
    return get_config().merge_enabled


def partition_layer_log_enabled() -> bool:
    """Log one line per layer with split/merge histograms (analysis runs only)."""
    return _env_flag(_ENV + "PARTITION_LOG_LAYERS")


def forward_timing_enabled() -> bool:
    """Log the wall time of every extend forward (for A/B latency measurements)."""
    return _env_flag(_ENV + "FORWARD_TIMING")


def partition_validate_enabled() -> bool:
    """Re-check leaf coverage/alignment after every build (tests and debugging)."""
    return _env_flag(_ENV + "PARTITION_VALIDATE")


def wants_layer(layer_id: int) -> bool:
    raw = os.environ.get(_ENV + "LAYERS", "all").strip().lower()
    if raw in ("", "all"):
        return True
    return layer_id in {int(v) for v in raw.split(",") if v.strip()}
