"""Paged FP8 summary pool for Adaptive-HISA leaves (build_only phase).

Layout mirrors the raw index-K pool: per layer a ``uint8[S_pages, 64*(128+4)]``
buffer, each page holding 64 summary rows as ``64x128`` FP8 bytes followed by
``64`` FP32 scales (SoA per page). Rows are addressed by a per-request page
table exactly like raw tokens, so a later paged DeepGEMM scorer can consume
them without a new layout.

Per (layer, request) the pool keeps ``leaf_start/leaf_len int32[storage]``
(padding ``len=0``), ``num_leaves`` (device scalar), ``n_complete``, epoch and a
``ready_event`` recorded after the last builder kernel. With target merge,
``storage`` is the post-merge capacity (≈ L/D), not pre-merge M0.

Phase B is synchronous on the main stream, so freeing pages when the scheduler
releases a request is stream-ordered against every kernel that wrote them.
The asynchronous builder (later) must add an explicit lease/consumed event.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import GpuPartition
from sglang.srt.layers.attention.nsa.adaptive_hisa.summary_kernels import (
    leaf_key_means,
    requant_summaries,
)

logger = logging.getLogger(__name__)

ROWS_PER_PAGE = 64
HEAD_DIM = 128
ROW_BYTES = HEAD_DIM + 4
PAGE_BYTES = ROWS_PER_PAGE * ROW_BYTES


class SummaryPoolExhausted(RuntimeError):
    pass


@dataclass
class SummaryEntry:
    layer_id: int
    req_idx: int
    epoch: int
    n_tokens: int
    n_complete: int
    capacity: int
    pages: list[int]
    page_table: torch.Tensor  # int32 [n_pages]
    row_loc: torch.Tensor  # int64 [capacity]
    partition: GpuPartition
    ready_event: torch.cuda.Event | None = None
    meta: dict = field(default_factory=dict)

    @property
    def leaf_start(self) -> torch.Tensor:
        return self.partition.leaf_start

    @property
    def leaf_len(self) -> torch.Tensor:
        return self.partition.leaf_len

    @property
    def num_leaves(self) -> torch.Tensor:
        return self.partition.num_leaves

    def wait(self) -> None:
        if self.ready_event is not None:
            self.ready_event.synchronize()


class SummaryPool:
    """B=1 private page allocation per (layer, request); shared allocator later."""

    page_size = ROWS_PER_PAGE  # for index_buf_accessor.SetKAndS

    def __init__(self, num_layers: int, pages_per_layer: int, device: torch.device | str):
        self.num_layers = int(num_layers)
        self.pages_per_layer = int(pages_per_layer)
        self.device = torch.device(device)
        self.bufs = [
            torch.zeros((self.pages_per_layer, PAGE_BYTES), dtype=torch.uint8, device=self.device)
            for _ in range(self.num_layers)
        ]
        self._free: list[list[int]] = [list(range(self.pages_per_layer - 1, -1, -1)) for _ in range(self.num_layers)]
        self._entries: dict[tuple[int, int], SummaryEntry] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    @property
    def bytes_allocated(self) -> int:
        return self.num_layers * self.pages_per_layer * PAGE_BYTES

    def free_pages(self, layer_id: int) -> int:
        return len(self._free[layer_id])

    def get(self, layer_id: int, req_idx: int) -> SummaryEntry | None:
        with self._lock:
            return self._entries.get((int(layer_id), int(req_idx)))

    def entries_for(self, req_idx: int) -> list[SummaryEntry]:
        with self._lock:
            return [e for (_, r), e in self._entries.items() if r == int(req_idx)]

    def release(self, req_idx: int) -> int:
        """Free every layer's pages of a request. Returns the number of entries."""
        with self._lock:
            keys = [k for k in self._entries if k[1] == int(req_idx)]
            for key in keys:
                entry = self._entries.pop(key)
                self._free[entry.layer_id].extend(entry.pages)
        return len(keys)

    def _release_locked(self, key: tuple[int, int]) -> None:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._free[entry.layer_id].extend(entry.pages)

    def allocate(
        self,
        layer_id: int,
        req_idx: int,
        epoch: int,
        partition: GpuPartition,
    ) -> SummaryEntry:
        """Reserve ``ceil(M0/64)`` pages; a previous entry of the slot is freed."""
        layer_id, req_idx = int(layer_id), int(req_idx)
        if not 0 <= layer_id < self.num_layers:
            raise ValueError(f"layer {layer_id} outside pool ({self.num_layers} layers)")
        capacity = int(partition.capacity)
        n_pages = math.ceil(capacity / ROWS_PER_PAGE) if capacity else 0
        with self._lock:
            self._release_locked((layer_id, req_idx))
            free = self._free[layer_id]
            if len(free) < n_pages:
                raise SummaryPoolExhausted(
                    f"layer {layer_id}: need {n_pages} summary pages, {len(free)} free"
                )
            pages = [free.pop() for _ in range(n_pages)]
            host_table = torch.tensor(pages, dtype=torch.int32)
            if self.device.type == "cuda":
                # pinned + non_blocking: no stream sync in the model thread
                page_table = host_table.pin_memory().to(self.device, non_blocking=True)
            else:
                page_table = host_table
            rows = torch.arange(capacity, device=self.device, dtype=torch.int64)
            row_loc = (
                page_table.to(torch.int64)[rows // ROWS_PER_PAGE] * ROWS_PER_PAGE + rows % ROWS_PER_PAGE
                if capacity
                else rows
            )
            entry = SummaryEntry(
                layer_id, req_idx, int(epoch), partition.n_tokens, partition.n_complete,
                capacity, pages, page_table, row_loc, partition,
            )
            self._entries[(layer_id, req_idx)] = entry
        return entry

    def write(self, entry: SummaryEntry, fp8: torch.Tensor, scale: torch.Tensor) -> None:
        """Write ``capacity`` rows (padding rows included) into the entry's pages."""
        if entry.capacity == 0:
            return
        if fp8.shape != (entry.capacity, HEAD_DIM):
            raise ValueError(f"summary fp8 shape {tuple(fp8.shape)} != {(entry.capacity, HEAD_DIM)}")
        scale = scale.reshape(entry.capacity, 1).to(torch.float32).contiguous()
        buf = self.bufs[entry.layer_id]
        if buf.is_cuda:
            from sglang.srt.layers.attention.nsa.index_buf_accessor import SetKAndS

            SetKAndS.execute(
                pool=self, buf=buf, loc=entry.row_loc, index_k=fp8.contiguous(), index_k_scale=scale
            )
            return
        # Torch fallback for CPU tests: same SoA page layout.
        loc = entry.row_loc
        page, row = loc // ROWS_PER_PAGE, loc % ROWS_PER_PAGE
        key_bytes = fp8.contiguous().view(torch.uint8)
        scale_bytes = scale.view(torch.uint8)  # [capacity, 4]
        for i in range(entry.capacity):
            p, r = int(page[i]), int(row[i])
            buf[p, r * HEAD_DIM : (r + 1) * HEAD_DIM] = key_bytes[i]
            off = ROWS_PER_PAGE * HEAD_DIM + r * 4
            buf[p, off : off + 4] = scale_bytes[i]

    def read(self, entry: SummaryEntry) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(fp8[capacity,128], scale[capacity])`` in leaf order (tests/logging)."""
        if entry.capacity == 0:
            empty = torch.empty(0, HEAD_DIM, dtype=torch.float8_e4m3fn, device=self.device)
            return empty, torch.empty(0, dtype=torch.float32, device=self.device)
        pages = self.bufs[entry.layer_id][entry.page_table.to(torch.long)]  # [n_pages, PAGE_BYTES]
        keys = pages[:, : ROWS_PER_PAGE * HEAD_DIM].reshape(-1, HEAD_DIM).view(torch.float8_e4m3fn)
        scales = pages[:, ROWS_PER_PAGE * HEAD_DIM :].contiguous().view(torch.float32).reshape(-1)
        return keys[: entry.capacity], scales[: entry.capacity]

    def stats(self) -> dict:
        with self._lock:
            used = sum(self.pages_per_layer - len(f) for f in self._free)
            return {
                "layers": self.num_layers,
                "pages_per_layer": self.pages_per_layer,
                "pages_used": used,
                "entries": len(self._entries),
                "bytes": self.bytes_allocated,
            }


def build_summaries(
    pool: SummaryPool,
    entry: SummaryEntry,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    *,
    scale_fmt: str | None,
) -> None:
    """Dequant → segmented mean → act_quant → paged write for the entry's leaves."""
    if entry.capacity == 0:
        return
    if k_fp8.shape[0] < entry.n_complete:
        raise ValueError(f"raw keys {k_fp8.shape[0]} shorter than sealed prefix {entry.n_complete}")
    means = leaf_key_means(k_fp8, k_scale, entry.leaf_start, entry.leaf_len)
    fp8, scale = requant_summaries(means, scale_fmt)
    pool.write(entry, fp8, scale)


def pages_for(raw_index_pages: int, compression: int, slack_pages: int = 16,
              merge_divisor: int | None = None) -> int:
    """Summary pages needed if every raw token belongs to a sealed prefix.

    When ``merge_divisor`` is set (target merge L/D), size for L/D rows instead
    of the pre-merge M0=L/compression padding.
    """
    div = max(1, int(merge_divisor or compression))
    return math.ceil(raw_index_pages / div) + slack_pages


_POOL: SummaryPool | None = None
_POOL_LOCK = threading.Lock()


def get_summary_pool(token_to_kv_pool, compression: int) -> SummaryPool:
    """Lazily size the pool from the raw index-K pool of the model (once per process)."""
    global _POOL
    from sglang.srt.layers.attention.nsa.adaptive_hisa.config import get_config

    cfg = get_config()
    merge_div = cfg.merge_target_divisor or None
    with _POOL_LOCK:
        if _POOL is None:
            bufs = token_to_kv_pool.index_k_with_scale_buffer
            raw_pages = int(bufs[0].shape[0])
            num_layers = len(bufs)
            _POOL = SummaryPool(
                num_layers,
                pages_for(raw_pages, compression, merge_divisor=merge_div),
                bufs[0].device,
            )
            logger.info(
                "adaptive-hisa summary pool: layers=%d pages/layer=%d bytes=%.1f MiB "
                "(compression=%d merge_divisor=%s)",
                num_layers,
                _POOL.pages_per_layer,
                _POOL.bytes_allocated / 2**20,
                compression,
                merge_div,
            )
        return _POOL


def set_summary_pool(pool: SummaryPool | None) -> None:
    global _POOL
    with _POOL_LOCK:
        _POOL = pool
