"""Prefill 阶段的 P-key partition 调度器。

这个文件不负责实现 P-key 数学算子，而是负责把整条构建流程接到模型运行时：

1. 判断当前请求是否是最后一个 Prefill chunk；
2. 取出当前层已经 gather 好的 ``K_fp8[N,128]`` 和 ``K_scale[N]``；
3. 选择 raw-FP8 GPU、普通 GPU 或 CPU-reference backend；
4. 在主 CUDA stream 或旁路 stream 上启动建树、Split、Merge；
5. 为最终叶子生成 FP8 summary 并写入 ``SummaryPool``；
6. 以 ``(layer_id, req_idx, epoch)`` 保存结果，供 Decode 阶段读取；
7. 在整次模型 forward 结束时等待旁路 stream，保证 Decode 不会读到半成品。

当前生产路径通常是：

``schedule_prefill_partition``
  → ``_schedule_key_partition``
  → ``_build_gpu_raw_fp8``
  → ``pkey_fp8_builder.build_partition_from_fp8``
  → ``SummaryPool.write``。

本文件不会改变当前官方 Prefill Top-2048；目前它仍是在官方 Top-2048 完成后
异步为后续 Decode 构建 partition。
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa.config import (
    PartitionConfig,
    forward_timing_enabled,
    get_config,
    partition_enabled,
    partition_layer_log_enabled,
    wants_layer,
)
from sglang.srt.layers.attention.nsa.adaptive_hisa.prefill_partition import (
    PartitionInputs,
    PrefillPartition,
    key_values,
    n_complete_tokens,
    partition_cpu_phase,
    partition_keys_gpu_phase,
    scale_histogram,
)

logger = logging.getLogger(__name__)


def _maybe_warmup_host_kernels() -> None:
    """进程启动时预热 CPU-reference 的 numba kernel。"""
    cfg = get_config()
    if cfg.enabled and cfg.split_backend == "cpu_reference":
        try:
            from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_reference import (
                warmup_host_kernels,
            )

            warmup_host_kernels()
        except Exception:
            logger.exception("adaptive-hisa host kernel warmup failed")


_maybe_warmup_host_kernels()

_GPU_WARM: set = set()


def _maybe_warmup_gpu_kernels(device: torch.device, forward_batch=None) -> None:
    """在每张 GPU 上只执行一次构建路径预热。

    提前完成 Triton 编译、CUDA graph 首次 capture、SummaryPool 分配以及
    ``SetKAndS`` 初始化，避免这些一次性开销落入用户请求的 TTFT。
    """
    if device.type != "cuda" or device in _GPU_WARM:
        return
    _GPU_WARM.add(device)
    cfg = get_config()
    if not (cfg.enabled and cfg.split_backend == "gpu"):
        return
    try:
        from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import (
            build_partition_gpu,
            graphed_builder,
        )
        from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_kernels import (
            warmup_triton_kernels,
        )
        from sglang.srt.layers.attention.nsa.adaptive_hisa.summary_pool import (
            build_summaries,
            get_summary_pool,
        )

        t0 = time.perf_counter()
        warmup_triton_kernels(device, cfg)
        n = cfg.root * 17
        rows = cfg.energy_rows  # 128 key dimensions
        graph = graphed_builder(rows, n, cfg, device)  # 首次 capture，同时初始化 graph。
        if graph is not None:
            part = graph.build(torch.rand(rows, n, device=device), n)
        else:
            part = build_partition_gpu(torch.rand(rows, n, device=device), n, cfg)
        kv_pool = getattr(forward_batch, "token_to_kv_pool", None)
        if cfg.build_summaries and kv_pool is not None:
            pool = get_summary_pool(kv_pool, cfg.summary_compression)
            keys = torch.zeros(n, 128, dtype=torch.float8_e4m3fn, device=device)
            scale = torch.ones(n, dtype=torch.float32, device=device)
            entry = pool.allocate(0, 0, -1, part)
            build_summaries(pool, entry, keys, scale, scale_fmt="ue8m0")
            pool.release(0)
        torch.cuda.synchronize(device)
        logger.info(
            "adaptive-hisa gpu kernels warm in %.2fs metric=%s method=%s energy_rows=%d",
            time.perf_counter() - t0, cfg.partition_metric, part.method, rows,
        )
    except Exception:
        logger.exception("adaptive-hisa gpu kernel warmup failed")


def partition_request_flags(
    reqs,
    chunked_req,
    *,
    is_extend: bool,
    seq_lens: list[int],
) -> tuple[list[bool], list[int]]:
    """为 batch 中每个请求返回 ``(是否最终 Prefill, 本轮新算 token 数)``。

    只有本轮真正完成 prompt 的请求才允许构建最终 partition；仍有后续 prompt
    token 的 ``chunked_req`` 不能构建。新 token 数不包含 radix-cache 命中的前缀。
    """
    if len(seq_lens) != len(reqs):
        raise ValueError(f"seq_lens={len(seq_lens)} does not match reqs={len(reqs)}")
    final: list[bool] = []
    new_tokens: list[int] = []
    for req, seq in zip(reqs, seq_lens):
        prefill_row = (
            is_extend
            and len(getattr(req, "output_ids", []) or []) == 0
            and int(getattr(req, "extend_input_len", 0) or 0) > 0
        )
        final.append(prefill_row and req is not chunked_req)
        if prefill_row:
            cached = int(getattr(req, "cached_tokens", 0) or 0)
            new_tokens.append(max(0, int(seq) - cached))
        else:
            new_tokens.append(0)
    return final, new_tokens


class _RequestEpoch:
    """按请求保存 partition，并用 epoch 防止请求槽复用导致读到旧数据。

    模型线程、scheduler 的释放路径和 CPU overlap worker 都会访问这些表，
    因此每次读写都持锁。``_parts`` 可以保存 CPU ``PrefillPartition`` 或 GPU
    ``GpuPartition``；只有日志/测试真正需要时，``get`` 才把 GPU 结果转到 host。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._epoch: dict[int, int] = {}
        self._parts: dict[tuple[int, int], object] = {}
        self._host: dict[tuple[int, int], PrefillPartition] = {}
        self._skip: dict[tuple[int, int], str] = {}

    def epoch(self, req_idx: int) -> int:
        return self._epoch.get(int(req_idx), 0)

    def release(self, req_idx: int) -> None:
        req_idx = int(req_idx)
        with self._lock:
            self._epoch[req_idx] = self._epoch.get(req_idx, 0) + 1
            for table in (self._parts, self._host, self._skip):
                for key in [key for key in table if key[1] == req_idx]:
                    del table[key]

    def put(self, layer_id: int, req_idx: int, epoch: int, part) -> bool:
        key = (int(layer_id), int(req_idx))
        with self._lock:
            if epoch != self._epoch.get(key[1], 0):
                logger.warning("adaptive-hisa dropped stale partition layer=%s req=%s", layer_id, req_idx)
                return False
            self._parts[key] = part
            self._host.pop(key, None)
            self._skip.pop(key, None)
        return True

    def raw(self, layer_id: int, req_idx: int):
        with self._lock:
            return self._parts.get((int(layer_id), int(req_idx)))

    def get(self, layer_id: int, req_idx: int) -> PrefillPartition | None:
        key = (int(layer_id), int(req_idx))
        with self._lock:
            host = self._host.get(key)
            part = self._parts.get(key)
        if host is not None:
            return host
        if part is None:
            return None
        if isinstance(part, PrefillPartition):
            return part
        host = part.to_host()  # 只做一次 device→host，并缓存结果。
        with self._lock:
            if self._parts.get(key) is part:
                self._host[key] = host
        return host

    def skip(self, layer_id: int, req_idx: int, reason: str) -> None:
        key = (int(layer_id), int(req_idx))
        with self._lock:
            first_for_request = not any(k[1] == key[1] and r == reason for k, r in self._skip.items())
            self._skip[key] = reason
        # 同一请求、同一原因只用 INFO 打一次，其余层降到 DEBUG，避免刷屏。
        logger.log(
            logging.INFO if first_for_request else logging.DEBUG,
            "adaptive-hisa skipped prefill partition layer=%s req=%s reason=%s",
            layer_id,
            req_idx,
            reason,
        )

    def skip_reason(self, layer_id: int, req_idx: int) -> str | None:
        with self._lock:
            return self._skip.get((int(layer_id), int(req_idx)))

    def clear(self) -> None:
        with self._lock:
            self._epoch.clear()
            self._parts.clear()
            self._host.clear()
            self._skip.clear()


@dataclass
class PreparedPartition:
    """一次构建所需的请求/层身份，以及可进入树的 sealed prefix 长度。"""

    layer_id: int
    req_idx: int
    epoch: int
    n_tokens: int
    n_complete: int
    # 最后一个 prefill chunk：之后不会再扩展这棵树。
    final: bool = False


@dataclass
class PendingPartition:
    """CPU-reference overlap worker 的待处理任务。"""

    layer_id: int
    req_idx: int
    epoch: int
    inputs: PartitionInputs


@dataclass
class BuildContext:
    """写最终 summary 所需的原始 K 和 KV pool 元数据。

    K 已由 ``_get_topk_ragged`` 按逻辑 token 顺序 gather：
    ``k_fp8=[N,128]``，``k_scale=[N]``。
    """

    k_fp8: torch.Tensor | None = None
    k_scale: torch.Tensor | None = None
    scale_fmt: str | None = None
    token_to_kv_pool: object | None = None


class _TreeCache:
    """上一个 prefill chunk 的 Key-SSE 树，按 ``(layer, req)`` 保存。

    sealed prefix 内的 dyadic 节点在更长前缀的树里不变，所以下一个 chunk 只需
    为新增 root 建节点（``raw_fp8_tree_triton_incremental``）。条目带 epoch，
    请求槽复用后旧树不会被误用；``release_request`` 时释放。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._trees: dict[tuple[int, int], tuple[int, int, torch.Tensor]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, layer_id: int, req_idx: int, epoch: int, n_complete: int):
        """返回 ``(flat, n_old)``；没有可扩展的严格前缀树时返回 ``None``。"""
        key = (int(layer_id), int(req_idx))
        with self._lock:
            hit = self._trees.get(key)
        if hit is None or hit[0] != epoch or not 0 < hit[1] < n_complete:
            self.misses += 1
            return None
        self.hits += 1
        return hit[2], hit[1]

    def put(self, layer_id: int, req_idx: int, epoch: int, n_complete: int, flat: torch.Tensor) -> None:
        key = (int(layer_id), int(req_idx))
        with self._lock:
            self._trees[key] = (int(epoch), int(n_complete), flat)

    def drop(self, layer_id: int, req_idx: int) -> None:
        with self._lock:
            self._trees.pop((int(layer_id), int(req_idx)), None)

    def release(self, req_idx: int) -> None:
        req_idx = int(req_idx)
        with self._lock:
            for key in [key for key in self._trees if key[1] == req_idx]:
                del self._trees[key]

    def clear(self) -> None:
        with self._lock:
            self._trees.clear()
        self.hits = self.misses = 0


STATE = _RequestEpoch()
TREES = _TreeCache()
_STREAM: torch.cuda.Stream | None = None
_WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="adaptive-hisa-partition")
_PENDING: list[Future] = []
# 本轮 forward 已经提交完成的 GPU partition；只在 forward 末尾读取标量。
_GPU_DONE: list[tuple[int, int, object]] = []
_STATS = {"launched": 0, "launch_s": 0.0, "built": 0, "gpu_built": 0, "first_t": 0.0, "mem0": {}}
_TIMING = {
    "score_s": 0.0,
    "tree_s": 0.0,
    "dp_s": 0.0,
    "repair_s": 0.0,
    "merge_s": 0.0,
    "summary_s": 0.0,
    "total_s": 0.0,
    "launch_wall_s": 0.0,
}
_STATS_LOCK = threading.Lock()
_NULL_CTX = contextlib.nullcontext()
# CPU worker 最多允许落后模型线程 8 层，超过后模型线程才等待。
_MAX_PENDING = 8


def reset_prefill_state() -> None:
    """清空 partition、decode workspace、summary pool 和旁路 stream 状态。"""
    global _STREAM
    finish_prefill_partitions()
    STATE.clear()
    TREES.clear()
    _STREAM = None
    import sys

    decode = sys.modules.get(__package__ + ".decode_runtime")
    if decode is not None:
        decode.reset_decode_state()
    try:
        from sglang.srt.layers.attention.nsa.adaptive_hisa.summary_pool import set_summary_pool

        set_summary_pool(None)
    except Exception:
        pass


def release_request(req_idx: int) -> None:
    """请求释放时递增 epoch，并释放该请求的 partition 和 summary 页。

    epoch 递增后，即使旧构建稍后完成，``STATE.put`` 也会拒绝写回旧请求槽。
    """
    STATE.release(req_idx)
    TREES.release(req_idx)
    import sys

    decode = sys.modules.get(__package__ + ".decode_runtime")
    if decode is not None:
        decode.release_decode_request(req_idx)
    try:
        from sglang.srt.layers.attention.nsa.adaptive_hisa import summary_pool as _sp

        if _sp._POOL is not None:
            _sp._POOL.release(req_idx)
    except Exception:
        logger.exception("adaptive-hisa summary release failed")


def get_prefill_partition(layer_id: int, req_idx: int) -> PrefillPartition | None:
    return STATE.get(layer_id, req_idx)


def get_gpu_partition(layer_id: int, req_idx: int):
    """取得 device 上的 ``GpuPartition``；不存在或是 CPU 结果时返回 ``None``。"""
    part = STATE.raw(layer_id, req_idx)
    return None if isinstance(part, PrefillPartition) else part


def get_summary_entry(layer_id: int, req_idx: int):
    from sglang.srt.layers.attention.nsa.adaptive_hisa import summary_pool as _sp

    return None if _sp._POOL is None else _sp._POOL.get(layer_id, req_idx)


def prefill_skip_reason(layer_id: int, req_idx: int) -> str | None:
    return STATE.skip_reason(layer_id, req_idx)


def _admission_skip(forward_batch) -> str | None:
    """判断当前 batch 是否支持 P-key 构建；返回 ``None`` 表示允许。"""
    try:
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        if get_is_capture_mode():
            return "cuda_graph"
    except Exception:
        pass
    if int(getattr(forward_batch, "batch_size", 0) or 0) != 1:
        return "batch_size"
    mode = forward_batch.forward_mode
    if not mode.is_extend_without_speculative() or mode.is_split_prefill() or mode.is_dllm_extend():
        return "forward_mode"
    spec = getattr(forward_batch, "spec_algorithm", None)
    if spec is not None and hasattr(spec, "is_none") and not spec.is_none():
        return "speculative"
    if getattr(forward_batch, "attn_cp_metadata", None) is not None:
        return "context_parallel"
    try:
        from sglang.srt.server_args import get_global_server_args

        args = get_global_server_args()
        if int(getattr(args, "pp_size", 1) or 1) > 1:
            return "pipeline_parallel"
        if bool(getattr(args, "enable_nsa_prefill_context_parallel", False)):
            return "context_parallel"
    except Exception:
        pass
    return None


def _request_index(forward_batch) -> int:
    return int(forward_batch.req_pool_indices[0].item())


def prepare_prefill_partition(
    forward_batch,
    layer_id: int,
    *,
    max_tokens: int | None = None,
) -> PreparedPartition | None:
    """校验当前请求并计算可参与 P-key 的 sealed prefix。

    这里只允许 B=1、最终 Prefill chunk、非 speculative/CP/PP 的请求。
    ``n_complete = floor(n_tokens / root) * root``，不足一个 root 的尾部不建树。
    """
    cfg = get_config()
    if not cfg.enabled or not wants_layer(layer_id):
        return None
    try:
        if _admission_skip(forward_batch) is not None:
            return None
        final = getattr(forward_batch, "prefill_final_cpu", None)
        if not final:
            return None
        new_tokens = getattr(forward_batch, "prefill_new_tokens_cpu", None)
        computed = int(new_tokens[0]) if new_tokens else 0
        if not bool(final[0]):
            # Sparse prefill also partitions intermediate chunks: the next
            # chunk's indexer scores this chunk's sealed prefix. Only real
            # prefill rows (new prompt tokens) qualify; decode rows never do.
            if not cfg.sparse_prefill or computed <= 0:
                return None
        req_idx = _request_index(forward_batch)
        if computed <= 0:
            STATE.skip(layer_id, req_idx, "new_tokens_0")
            return None
        n_tokens = (
            int(forward_batch.seq_lens_cpu[0].item())
            if forward_batch.seq_lens_cpu is not None
            else int(max_tokens or 0)
        )
        if max_tokens is not None:
            n_tokens = min(n_tokens, int(max_tokens))
        return PreparedPartition(
            int(layer_id),
            req_idx,
            STATE.epoch(req_idx),
            n_tokens,
            n_complete_tokens(n_tokens, cfg.root),
            bool(final[0]),
        )
    except Exception:
        logger.exception("adaptive-hisa failed to prepare prefill partition")
        return None


def _first_build_bookkeeping(cfg: PartitionConfig) -> None:
    """记录本轮第一次构建的时间和 allocator 计数。"""
    if not _STATS["launched"] and not _STATS["built"] and not _STATS["gpu_built"]:
        _STATS["first_t"] = time.perf_counter()
        _STATS["mem0"] = _alloc_counters() if cfg.profile else {}


def schedule_prefill_partition(
    forward_batch,
    layer_id: int,
    *,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    scale_fmt: str | None = None,
) -> None:
    """Prefill 侧的 P-key 总调度入口。

    输入是当前层按逻辑顺序 gather 好的 FP8 K。通过准入检查后，统一交给
    :func:`_schedule_key_partition` 选择 backend。这个函数不接收 Query。
    """
    cfg = get_config()
    if not cfg.enabled or not wants_layer(layer_id):
        return
    try:
        _maybe_warmup_gpu_kernels(k_fp8.device, forward_batch)
        prepared = prepare_prefill_partition(
            forward_batch, layer_id, max_tokens=int(k_fp8.shape[0])
        )
        if prepared is None:
            return
        k_fp8, k_scale = _normalize_keys(k_fp8, k_scale)
        _first_build_bookkeeping(cfg)
        _schedule_key_partition(
            prepared, k_fp8, k_scale, scale_fmt, forward_batch, cfg
        )
    except Exception:
        logger.exception("adaptive-hisa prefill partition failed")


def _schedule_key_partition(
    prepared: PreparedPartition,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    scale_fmt: str | None,
    forward_batch,
    cfg: PartitionConfig,
) -> None:
    """选择 P-key backend。

    优先级：

    1. GPU + ``RAW_FP8_BUILDER=1``：进入 ``_build_gpu_raw_fp8``；
    2. 普通 GPU：先产生 ``[128,N]`` FP32，再进入 ``_build_gpu``；
    3. CPU-reference overlap：GPU 建 moment 后 D2H，交给 worker；
    4. 同步 CPU-reference。
    """
    from sglang.srt.layers.attention.nsa.adaptive_hisa.pkey_fp8_builder import (
        raw_fp8_builder_enabled,
    )

    if cfg.split_backend == "gpu" and raw_fp8_builder_enabled():
        # 当前交付配置走这里；真正的 P-key/Split/Merge 总入口在
        # pkey_fp8_builder.build_partition_from_fp8。
        _build_gpu_raw_fp8(
            prepared, k_fp8, k_scale, scale_fmt, forward_batch, cfg
        )
        return
    if cfg.split_backend == "gpu":
        # 回退路径：物化旧式 [128,N] FP32 key rows。
        t0 = _mark(k_fp8.device, cfg.profile)
        values = key_values(k_fp8, k_scale, prepared.n_complete)  # [128, N_complete]
        t1 = _mark(k_fp8.device, cfg.profile)
        _build_gpu(
            prepared,
            values,
            cfg,
            BuildContext(k_fp8, k_scale, scale_fmt, getattr(forward_batch, "token_to_kv_pool", None)),
            score_events=(t0, t1),
        )
        return
    if cfg.cpu_overlap and k_fp8.is_cuda:
        t0 = time.perf_counter()
        side = _partition_stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for tensor in (k_fp8, k_scale):
                tensor.record_stream(side)
            inputs = partition_keys_gpu_phase(
                k_fp8, k_scale, prepared.n_tokens,
                atom=cfg.atom, root=cfg.root, non_blocking=True, profile=cfg.profile,
            )
        _submit(PendingPartition(prepared.layer_id, prepared.req_idx, prepared.epoch, inputs), cfg)
        _STATS["launched"] += 1
        _STATS["launch_s"] += time.perf_counter() - t0
        return
    inputs = partition_keys_gpu_phase(
        k_fp8, k_scale, prepared.n_tokens, atom=cfg.atom, root=cfg.root, profile=cfg.profile
    )
    part = partition_cpu_phase(inputs, cfg=cfg)
    if STATE.put(prepared.layer_id, prepared.req_idx, prepared.epoch, part):
        _STATS["built"] += 1
        _account(prepared.layer_id, prepared.req_idx, part, overlapped=False)


def _build_gpu_raw_fp8(
    prepared: PreparedPartition,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    scale_fmt: str | None,
    forward_batch,
    cfg: PartitionConfig,
) -> None:
    """生产用 raw-FP8 P-key 路径。

    建树、叶子 totals 和 summary 都直接从 FP8 K 计算，不再物化
    ``[N,128]`` FP32 K、``[N+1,128]`` FP64 prefix 或 ``[M,128]`` FP32 means。
    """
    from sglang.srt.layers.attention.nsa.adaptive_hisa.pkey_fp8_builder import (
        build_partition_from_fp8,
        summaries_from_totals,
    )
    from sglang.srt.layers.attention.nsa.adaptive_hisa.summary_pool import get_summary_pool

    wall0 = time.perf_counter()
    side = None
    k_fp8, k_scale = _normalize_keys(k_fp8, k_scale)
    if cfg.gpu_stream == "side" and k_fp8.is_cuda:
        # 旁路 stream 必须先等待模型 stream 上的 K gather 完成。
        side = _partition_stream()
        side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side) if side is not None else _NULL_CTX:
        if side is not None:
            k_fp8.record_stream(side)
            k_scale.record_stream(side)
        tree_cache = None
        if cfg.tree_cache:
            tree_cache = TREES.get(
                prepared.layer_id, prepared.req_idx, prepared.epoch, prepared.n_complete
            )
        part = build_partition_from_fp8(
            k_fp8,
            k_scale,
            prepared.n_tokens,
            cfg,
            profile=cfg.profile,
            tree_cache=tree_cache,
            summary_scale_fmt=scale_fmt,
        )
        flat = part.meta.pop("_tree_flat", None)
        if prepared.final:
            # 最后一个 chunk：这棵树不会再被扩展，立刻释放（16 B/token/层）。
            TREES.drop(prepared.layer_id, prepared.req_idx)
        elif cfg.tree_cache and flat is not None and part.n_complete:
            TREES.put(prepared.layer_id, prepared.req_idx, prepared.epoch, part.n_complete, flat)
        part.meta["tree_incremental"] = tree_cache is not None
        if tree_cache is not None and TREES.hits == 1:
            logger.info(
                "adaptive-hisa incremental Key-SSE tree: layer=%d req=%d prefix %d -> %d tokens "
                "(only the %d new roots are built)",
                prepared.layer_id, prepared.req_idx, int(tree_cache[1]), part.n_complete,
                (part.n_complete - int(tree_cache[1])) // cfg.root,
            )
        part.meta["layer_id"] = prepared.layer_id
        part.meta["req_idx"] = prepared.req_idx
        part.meta["stream"] = "side" if side is not None else "main"
        if cfg.build_summaries and part.n_complete:
            # 按最终 post-merge capacity 分配 summary 页。
            token_pool = getattr(forward_batch, "token_to_kv_pool", None)
            if token_pool is None:
                raise ValueError("raw-fp8 builder needs forward_batch.token_to_kv_pool")
            pool = get_summary_pool(token_pool, cfg.summary_compression)
            t0 = _mark(k_fp8.device, cfg.profile)
            entry = pool.allocate(
                prepared.layer_id, prepared.req_idx, prepared.epoch, part
            )
            fp8 = part.meta.pop("_summary_fp8", None)
            scale = part.meta.pop("_summary_scale", None)
            fused_summary = fp8 is not None
            if fp8 is None:
                totals = part.meta.pop("_merge_totals", None)
                if totals is None:
                    # Merge 关闭或未返回 totals 时，重新按最终叶子计算 sum(K)。
                    from sglang.srt.layers.attention.nsa.adaptive_hisa.pkey_fp8_builder import (
                        leaf_totals_from_fp8,
                    )

                    totals = leaf_totals_from_fp8(
                        k_fp8,
                        k_scale,
                        entry.leaf_start,
                        entry.leaf_len,
                        part.n_complete,
                    )
                fp8, scale = summaries_from_totals(
                    totals, entry.leaf_len, scale_fmt
                )
            pool.write(entry, fp8, scale)
            part.meta["summary_from_totals"] = not fused_summary
            part.meta["summary_fused_from_atoms"] = fused_summary
            t1 = _mark(k_fp8.device, cfg.profile)
            if k_fp8.is_cuda:
                entry.ready_event = torch.cuda.Event()
                entry.ready_event.record()
            if cfg.profile:
                part.events["summary_s"] = (t0, t1)
            part.meta["summary_pages"] = len(entry.pages)
    if side is not None:
        _STATS["side_used"] = True
    part.meta["launch_wall_s"] = time.perf_counter() - wall0
    if STATE.put(prepared.layer_id, prepared.req_idx, prepared.epoch, part):
        # 只保存 epoch 仍匹配的结果，防止请求槽已复用。
        _STATS["gpu_built"] += 1
        _GPU_DONE.append((prepared.layer_id, prepared.req_idx, part))
    else:
        try:
            from sglang.srt.layers.attention.nsa.adaptive_hisa import summary_pool as _sp

            if _sp._POOL is not None:
                _sp._POOL.release(prepared.req_idx)
        except Exception:
            pass


def _build_gpu(
    prepared: PreparedPartition,
    values: torch.Tensor,
    cfg: PartitionConfig,
    ctx: BuildContext,
    *,
    score_events=None,
) -> None:
    """普通 GPU 回退路径：``[128,N]`` key rows → Split → Merge → summary。

    可在主 stream 或 builder 旁路 stream 上执行；旁路 stream 最终由
    :func:`finish_prefill_partitions` 合回模型 stream。
    """
    from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import (
        build_partition_gpu,
        graphed_builder,
    )

    wall0 = time.perf_counter()
    side = None
    if cfg.gpu_stream == "side" and values.is_cuda:
        n_done = n_complete_tokens(prepared.n_tokens, cfg.root)
        if values.shape[1] > n_done:
            values = values[:, :n_done]
        side = _partition_stream()
        side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side) if side is not None else _NULL_CTX:
        if side is not None:
            values.record_stream(side)
        graph = graphed_builder(values.shape[0], prepared.n_tokens, cfg, values.device)
        if graph is not None:
            part = graph.build(values, prepared.n_tokens, profile=cfg.profile)
        else:
            part = build_partition_gpu(values, prepared.n_tokens, cfg, profile=cfg.profile)
        if score_events is not None and cfg.profile:
            part.events["score_s"] = score_events
        part.meta["layer_id"] = prepared.layer_id
        part.meta["req_idx"] = prepared.req_idx
        part.meta["stream"] = "side" if side is not None else "main"
        if cfg.build_summaries and part.n_complete and ctx.k_fp8 is not None:
            from sglang.srt.layers.attention.nsa.adaptive_hisa.summary_pool import (
                build_summaries,
                get_summary_pool,
            )

            if ctx.token_to_kv_pool is None:
                raise ValueError("gpu backend needs forward_batch.token_to_kv_pool to size the summary pool")
            k_fp8, k_scale = _normalize_keys(ctx.k_fp8, ctx.k_scale)
            if side is not None:
                ctx.k_fp8.record_stream(side)
                ctx.k_scale.record_stream(side)
                k_scale.record_stream(side)
            pool = get_summary_pool(ctx.token_to_kv_pool, cfg.summary_compression)
            t0 = _mark(values.device, cfg.profile)
            entry = pool.allocate(prepared.layer_id, prepared.req_idx, prepared.epoch, part)
            build_summaries(pool, entry, k_fp8, k_scale, scale_fmt=ctx.scale_fmt)
            t1 = _mark(values.device, cfg.profile)
            if values.is_cuda:
                entry.ready_event = torch.cuda.Event()
                entry.ready_event.record()
            if cfg.profile:
                part.events["summary_s"] = (t0, t1)
            part.meta["summary_pages"] = len(entry.pages)
    if side is not None:
        _STATS["side_used"] = True
    part.meta["launch_wall_s"] = time.perf_counter() - wall0
    if STATE.put(prepared.layer_id, prepared.req_idx, prepared.epoch, part):
        _STATS["gpu_built"] += 1
        _GPU_DONE.append((prepared.layer_id, prepared.req_idx, part))
    else:
        # epoch 已过期：请求在 forward 中途释放，同时归还已分配的 summary 页。
        try:
            from sglang.srt.layers.attention.nsa.adaptive_hisa import summary_pool as _sp

            if _sp._POOL is not None:
                _sp._POOL.release(prepared.req_idx)
        except Exception:
            pass


def _submit(job: PendingPartition, cfg: PartitionConfig) -> None:
    """提交 CPU-reference worker；积压达到上限时等待最老任务。"""
    _reap()
    while len(_PENDING) >= _MAX_PENDING:
        _PENDING.pop(0).result()
    _PENDING.append(_WORKER.submit(_host_phase, job, cfg))


def _host_phase(job: PendingPartition, cfg: PartitionConfig) -> None:
    """Worker 线程：等待 D2H 完成，再执行不持 GIL 的 numba Split/Merge。"""
    part = partition_cpu_phase(job.inputs, cfg=cfg)
    if STATE.put(job.layer_id, job.req_idx, job.epoch, part):
        _account(job.layer_id, job.req_idx, part, overlapped=True)


def _alloc_counters() -> dict[str, int]:
    """读取 allocator 计数；频繁 retry 表示 builder 与模型激活在争抢显存。"""
    if not torch.cuda.is_available():
        return {}
    stats = torch.cuda.memory_stats()
    return {k: int(stats.get(k, 0)) for k in ("num_device_alloc", "num_device_free", "num_alloc_retries", "num_sync_all_streams")}


def _reap() -> None:
    while _PENDING and _PENDING[0].done():
        _PENDING.pop(0).result()


def begin_forward(forward_batch) -> None:
    """记录一次 extend forward 的起点，供 A/B wall-time 日志使用。"""
    _STATS["fwd_t0"] = 0.0
    if forward_timing_enabled() and forward_batch.forward_mode.is_extend():
        torch.cuda.synchronize()
        _STATS["fwd_t0"] = time.perf_counter()
        _STATS["fwd_tokens"] = int(forward_batch.input_ids.numel())


def finish_prefill_partitions() -> None:
    """在模型 forward 末尾收口所有 partition 构建。

    CPU worker 全部等待完成；若 GPU 使用 side stream，则让模型 stream 等待
    builder stream。此函数返回后，Decode 才能安全读取所有层的 summary。
    """
    t0 = time.perf_counter()
    backlog = len(_PENDING)
    while _PENDING:
        _PENDING.pop(0).result()
    if _STATS.get("side_used"):
        # 合流后，模型 stream 随后的 Decode 一定能看到完整 partition。
        torch.cuda.current_stream().wait_stream(_partition_stream())
        _STATS["side_used"] = False
    if _STATS.get("fwd_t0"):
        torch.cuda.synchronize()
        logger.info(
            "adaptive-hisa extend forward tokens=%d wall_s=%.4f",
            _STATS.get("fwd_tokens", 0),
            time.perf_counter() - _STATS["fwd_t0"],
        )
        _STATS["fwd_t0"] = 0.0
    gpu_done = list(_GPU_DONE)
    _GPU_DONE.clear()
    if gpu_done:
        cfg = get_config()
        # 每轮 forward 只在这里集中读取 device 标量，避免逐层同步。
        for layer_id, req_idx, part in gpu_done:
            if cfg.profile:
                host = STATE.get(layer_id, req_idx)
                if host is not None:
                    _account(layer_id, req_idx, host, overlapped=False)
            else:
                with _STATS_LOCK:
                    _TIMING["launch_wall_s"] += float(part.meta.get("launch_wall_s") or 0.0)
    if _STATS["launched"] or _STATS["built"] or _STATS["gpu_built"]:
        with _STATS_LOCK:
            sums = dict(_TIMING)
            _TIMING.update({k: 0.0 for k in _TIMING})
        allocator_delta = {}
        if _STATS["mem0"]:
            allocator_delta = {
                key: value - _STATS["mem0"].get(key, 0)
                for key, value in _alloc_counters().items()
            }
        logger.info(
            "adaptive-hisa forward summary sync_built=%d gpu_built=%d launched=%d "
            "model_thread_launch_s=%.4f final_backlog=%d final_wait_s=%.4f "
            "span_first_build_to_finish_s=%.4f builder_sums score_s=%.4f tree_s=%.4f "
            "dp_s=%.4f repair_s=%.4f merge_s=%.4f summary_s=%.4f total_s=%.4f "
            "gpu_launch_wall_s=%.4f allocator_delta=%s",
            _STATS["built"],
            _STATS["gpu_built"],
            _STATS["launched"],
            _STATS["launch_s"],
            backlog,
            time.perf_counter() - t0,
            time.perf_counter() - _STATS["first_t"],
            sums["score_s"],
            sums["tree_s"],
            sums["dp_s"],
            sums["repair_s"],
            sums["merge_s"],
            sums["summary_s"],
            sums["total_s"],
            sums["launch_wall_s"],
            allocator_delta,
        )
        _STATS.update(launched=0, launch_s=0.0, built=0, gpu_built=0, first_t=0.0, mem0={})


def _partition_stream() -> torch.cuda.Stream:
    """懒创建所有层共享的一条低优先级 builder stream。"""
    global _STREAM
    if _STREAM is None:
        priority = 0
        if hasattr(torch.cuda, "stream_priority_range"):
            least, _greatest = torch.cuda.stream_priority_range()
            priority = int(least)
        _STREAM = torch.cuda.Stream(priority=priority)
    return _STREAM


def _mark(device: torch.device, enabled: bool):
    if not enabled:
        return None
    if device.type != "cuda":
        return time.perf_counter()
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _normalize_keys(k_fp8: torch.Tensor, k_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """统一 K/scale 的 dtype 和 shape，不复制 K 主体。"""
    if k_fp8.dtype == torch.uint8:
        k_fp8 = k_fp8.view(torch.float8_e4m3fn)
    scale = k_scale.detach()
    if scale.dtype == torch.uint8:
        scale = scale.view(torch.float32)
    return k_fp8, scale.reshape(-1).to(torch.float32)


def _account(layer_id: int, req_idx: int, part: PrefillPartition, *, overlapped: bool) -> None:
    """汇总分阶段耗时；详细逐层直方图仅在显式开启时打印。"""
    meta = part.meta
    with _STATS_LOCK:
        for key in _TIMING:
            _TIMING[key] += float(meta.get(key) or 0.0)
    # 直方图格式化会占用 GIL，因此默认不在每层执行。
    if partition_layer_log_enabled():
        split = meta.get("split_leaf_len")
        logger.info(
            "adaptive-hisa partition layer=%s req=%s backend=%s leaves=%s tail=%s "
            "score_s=%.4f tree_s=%.4f dp_s=%.4f repair_s=%.4f merge_s=%.4f summary_s=%.4f "
            "total_s=%.4f wall_s=%.4f overlap=%s premerge=%s merges=%s round_merges=%s "
            "repairs=%s status_ok=%s split_hist=%s hist=%s",
            layer_id,
            req_idx,
            meta.get("backend", "cpu_reference"),
            part.leaf_start.numel(),
            part.tail_len,
            float(meta.get("score_s") or 0.0),
            float(meta.get("tree_s") or 0.0),
            float(meta.get("dp_s") or 0.0),
            float(meta.get("repair_s") or 0.0),
            float(meta.get("merge_s") or 0.0),
            float(meta.get("summary_s") or 0.0),
            float(meta.get("total_s") or 0.0),
            float(meta.get("wall_s") or meta.get("launch_wall_s") or 0.0),
            overlapped,
            meta.get("premerge_M"),
            meta.get("merge_count", 0),
            meta.get("round_merge_counts"),
            meta.get("repairs", 0),
            meta.get("status_ok", True),
            scale_histogram(split) if split is not None else {},
            part.bucket_histogram(),
        )
