from __future__ import annotations

import contextlib
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from einops import rearrange

from sglang.jit_kernel.fused_store_index_cache import (
    can_use_nsa_fused_store,
    fused_store_index_k_cache,
)
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor
from sglang.srt.layers.layernorm import LayerNorm
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.state_capturer.indexer_topk import (
    maybe_capture_indexer_topk,
)
from sglang.srt.utils import (
    add_prefix,
    ceil_align,
    get_bool_env_var,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_npu,
)

global _use_multi_stream
_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_fp8_fnuz = is_fp8_fnuz()
_is_gfx95_supported = is_gfx95_supported()
if _is_cuda:
    try:
        import deep_gemm
    except ImportError as e:
        deep_gemm = e

if _use_aiter:
    from aiter.ops.cache import indexer_k_quant_and_cache

if is_npu():
    import torch_npu
    from sglang.srt.hardware_backend.npu.utils import get_indexer_weight_stream

from sglang.srt.distributed import (
    get_attn_context_model_parallel_rank,
    get_attn_context_model_parallel_world_size,
)
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.nsa import headmap_probe as _headmap_probe
from sglang.srt.layers.attention.nsa import indexer_dump as _indexer_dump
from sglang.srt.layers.attention.nsa import per_head_paged as _per_head_paged
from sglang.srt.layers.attention.nsa import offline_unique_head_router as _offline_router
from sglang.srt.layers.attention.nsa.adaptive_hisa.config import (
    get_config as get_adaptive_hisa_config,
    partition_enabled,
)
from sglang.srt.layers.attention.nsa.utils import (
    is_nsa_enable_prefill_cp,
    is_nsa_prefill_cp_in_seq_split,
)
from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args

_use_ag_after_qlora = envs.SGLANG_USE_AG_AFTER_QLORA.get()


def _schedule_adaptive_partition(
    forward_batch, layer_id, k_fp8, k_scale, scale_fmt=None
) -> None:
    if not partition_enabled():
        return
    from sglang.srt.layers.attention.nsa.adaptive_hisa.prefill_runtime import (
        schedule_prefill_partition,
    )

    schedule_prefill_partition(
        forward_batch, layer_id, k_fp8=k_fp8, k_scale=k_scale, scale_fmt=scale_fmt
    )

def _skip_adaptive_partition(forward_batch, layer_id, reason: str) -> None:
    if not partition_enabled():
        return
    from sglang.srt.layers.attention.nsa.adaptive_hisa.prefill_runtime import STATE

    final = getattr(forward_batch, "prefill_final_cpu", None)
    if not final or not bool(final[0]):
        return
    try:
        req_idx = int(forward_batch.req_pool_indices[0].item())
    except Exception:
        return
    STATE.skip(layer_id, req_idx, reason)
if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool


DUAL_STREAM_TOKEN_THRESHOLD = 1024 if _is_cuda else 0


class BaseIndexerMetadata(ABC):
    @abstractmethod
    def get_seqlens_int32(self) -> torch.Tensor:
        """
        Return: (batch_size,) int32 tensor
        """

    @abstractmethod
    def get_page_table_64(self) -> torch.Tensor:
        """
        Return: (batch_size, num_blocks) int32, page table.
                The page size of the table is 64.
        """

    @abstractmethod
    def get_page_table_1(self) -> torch.Tensor:
        """
        Return: (batch_size, num_blocks) int32, page table.
                The page size of the table is 1.
        """

    @abstractmethod
    def get_seqlens_expanded(self) -> torch.Tensor:
        """
        Return: (sum_extend_seq_len,) int32 tensor
        """

    def get_indexer_kvcache_range(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return: (tokens, ), (tokens, ) int32, k_start and k_end in kv cache(token,xxx) for each token.
        """

    def get_indexer_seq_len_cpu(self) -> torch.Tensor:
        """
        Return: seq lens for each batch.
        """

    def get_indexer_seq_len(self) -> torch.Tensor:
        """
        Return: seq lens for each batch.
        """

    def get_nsa_extend_len_cpu(self) -> List[int]:
        """
        Return: extend seq lens for each batch.
        """

    def get_token_to_batch_idx(self) -> torch.Tensor:
        """
        Return: batch idx for each token.
        """

    @abstractmethod
    def topk_transform(
        self,
        logits: torch.Tensor,
        topk: int,
    ) -> torch.Tensor:
        """
        Perform topk selection on the logits and possibly transform the result.

        NOTE that attention backend may override this function to do some
        transformation, which means the result of this topk_transform may not
        be the topk indices of the input logits.

        Return: Anything, since it will be passed to the attention backend
                for further processing on sparse attention computation.
                Don't assume it is the topk indices of the input logits.
        """


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    # from sgl_kernel import hadamard_transform
    if _is_hip:
        from fast_hadamard_transform import hadamard_transform
    else:
        from sglang.jit_kernel.hadamard import hadamard_transform

    hidden_size = x.size(-1)
    assert (
        hidden_size & (hidden_size - 1)
    ) == 0, "Hidden size must be a power of 2 for Hadamard transform."
    return hadamard_transform(x, scale=hidden_size**-0.5)


class Indexer(MultiPlatformOp):
    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        max_position_embeddings: int,
        rope_theta: float,
        layer_id: int,
        scale_fmt: Optional[str],
        block_size: int = 128,
        rope_scaling: Optional[Dict[str, Any]] = None,
        is_neox_style: bool = True,
        prefix: str = "",
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attn_context_model_parallel_world_size()
            self.cp_rank = get_attn_context_model_parallel_rank()
        else:
            self.cp_size = None
            self.cp_rank = None
        if _is_cuda:
            self.sm_count = deep_gemm.get_num_sms()
            self.half_device_sm_count = ceil_align(self.sm_count // 2, 8)
            pp_size = get_global_server_args().pp_size
            self.logits_with_pp_recv = pp_size > 1 and not get_pp_group().is_last_rank
        else:
            self.logits_with_pp_recv = False

        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )

        self.wk = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wk", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            self.hidden_size,
            self.n_heads,
            bias=False,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.k_norm = LayerNorm(
            self.head_dim, dtype=torch.bfloat16 if _use_aiter else torch.float32
        )
        self.rotary_emb = get_rope_wrapper(
            rope_head_dim,
            rotary_dim=rope_head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,  # type: ignore
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            device=get_global_server_args().device,
        )
        self.block_size = block_size
        self.scale_fmt = scale_fmt
        self.softmax_scale = self.head_dim**-0.5




    @contextlib.contextmanager
    def _with_real_sm_count(self):
        # When pipeline parallelism is enabled, each PP rank initiates a recv operation after the _pp_launch_batch
        # request to receive the PP proxy tensor or output from the previous stage, occupying one SM resource.
        # Model execution runs in parallel with the recv operation, so the SMs available to the indexer must be reduced
        # by 1. Currently, the last rank starts the send result + recv request only after waiting for execution results.
        if self.logits_with_pp_recv:
            pp_recv_sm_count = 1
            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.sm_count - pp_recv_sm_count
            ):
                yield
        else:
            yield




    def _weights_proj_bf16_in_fp32_out(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> torch.Tensor:
        # aiter (ROCm gfx95): extract the passthrough bf16 tensor from the
        # 3-tuple (fp8, scale, bf16) produced by fused_rms_fp8_group_quant,
        # avoiding an expensive FP8-to-bf16 dequantization.
        if _use_aiter and _is_gfx95_supported and isinstance(x, tuple) and len(x) == 3:
            x = x[2]
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            weight = self.weights_proj.weight
            out = torch.empty(
                (x.shape[0], weight.shape[0]),
                dtype=torch.float32,
                device=x.device,
            )
            deep_gemm_wrapper.gemm_nt_bf16bf16f32(x, weight, out)
            return out

        weights, _ = self.weights_proj(x)
        if _is_hip:
            # Return bf16; multiplying with q_scale promotes back to fp32.
            return weights
        return weights.float()




    @torch.compile(dynamic=True)
    def _project_raw_head_gates(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ):
        return self._weights_proj_bf16_in_fp32_out(x)




    @torch.compile(dynamic=True)
    def _get_raw_and_logits_head_gate(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor, ...]], q_scale: torch.Tensor
    ):
        raw_gate = self._weights_proj_bf16_in_fp32_out(x)
        logits_gate = raw_gate * self.n_heads**-0.5
        logits_gate = logits_gate.unsqueeze(-1) * q_scale * self.softmax_scale
        return raw_gate, logits_gate





    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
        forward_batch: ForwardBatch,
    ):
        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)

            with deep_gemm_wrapper.configure_deep_gemm_num_sms(
                self.half_device_sm_count
            ):
                query, _ = self.wq_b(q_lora)
                query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
                q_rope, _ = torch.split(
                    query,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )
            with torch.cuda.stream(self.alt_stream):
                # TODO we should also put DeepGEMM half SM here?
                key, _ = self.wk(x)
                key = self.k_norm(key)

                k_rope, _ = torch.split(
                    key,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )

            current_stream.wait_stream(self.alt_stream)
        else:
            query, _ = self.wq_b(q_lora)
            query = rearrange(query, "l (h d) -> l h d", d=self.head_dim)
            q_rope, _ = torch.split(
                query, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )
            key, _ = self.wk(x)
            key = self.k_norm(key)
            k_rope, _ = torch.split(
                key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
            )

        q_rope, k_rope = self.rotary_emb(positions, q_rope, k_rope)

        self._update_rope_guarded(query[..., : self.rope_head_dim], q_rope)
        self._update_rope_guarded(key[..., : self.rope_head_dim], k_rope)

        if enable_dual_stream:
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query = rotate_activation(query)

            with torch.cuda.stream(self.alt_stream):
                key = rotate_activation(key)
            current_stream.wait_stream(self.alt_stream)
        elif (
            self.alt_stream is not None
            and forward_batch.attn_cp_metadata is not None
            and self.nsa_enable_prefill_cp
        ):
            key = rotate_activation(key)
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            query = rotate_activation(query)

            with torch.cuda.stream(self.alt_stream):
                key = cp_all_gather_rerange_output(
                    key.contiguous(),
                    self.cp_size,
                    forward_batch,
                    torch.cuda.current_stream(),
                )
            current_stream.wait_stream(self.alt_stream)
            return query, key
        else:
            query = rotate_activation(query)
            key = rotate_activation(key)

        # allgather+rerrange
        if forward_batch.attn_cp_metadata is not None and self.nsa_enable_prefill_cp:
            key = cp_all_gather_rerange_output(
                key.contiguous(),
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
        return query, key




    def _get_k_bf16(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        enable_dual_stream: bool,
    ):
        # Compute only key, skip query
        key, _ = self.wk(x)
        key = self.k_norm(key)
        k_rope, _ = torch.split(
            key, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )

        _, k_rope = self.rotary_emb(positions, k_rope, k_rope)
        self._update_rope_guarded(key[..., : self.rope_head_dim], k_rope)
        key = rotate_activation(key)

        return key




    @staticmethod
    def _update_rope_guarded(dst: torch.Tensor, src: torch.Tensor) -> None:
        # On AMD with in-place RoPE kernels, self-aliasing can occur;
        # skip write-back when src/dst tensors point to a single memory.
        if src.data_ptr() == dst.data_ptr():
            return
        dst.copy_(src)





    def _per_head_paged_topk(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        metadata: BaseIndexerMetadata,
        row_batch: torch.Tensor,
        row_len: torch.Tensor,
        segments: Optional[List[Tuple[int, int, int]]],
        head_range: Optional[Tuple[int, int]] = None,
        head_ids: Optional[List[int] | torch.Tensor] = None,
        coarse_block_ids: Optional[torch.Tensor] = None,
        coarse_block_size: int = _per_head_paged.MISA_DEFAULT_POOLING_BLOCK,
        mean_ranked_chunks: bool = False,
    ) -> torch.Tensor:
        """Per-head index contract: ``[num_tokens, G, topk]`` request-relative
        token positions for this rank's G indexer heads (see per_head_paged).
        Rows beyond the real queries (TP / DP padding) are ``-1``.
        """
        if get_is_capture_mode() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "SGLANG_NSA_PER_HEAD_INDEX=1 requires --disable-cuda-graph"
            )
        pool = forward_batch.token_to_kv_pool
        assert pool.page_size == 64, "per-head paged indexer needs page size 64"
        if head_ids is None:
            h0, h1 = (
                _per_head_paged.local_head_range(self.n_heads)
                if head_range is None
                else head_range
            )
            output_heads = h1 - h0
        else:
            if head_range is not None:
                raise ValueError("head_range and head_ids are mutually exclusive")
            h0 = h1 = 0
            output_heads = (
                int(head_ids.shape[-1])
                if isinstance(head_ids, torch.Tensor)
                else len(head_ids)
            )
        q_offset = row_len.shape[0]

        picks = _per_head_paged.per_head_topk_paged(
            q_fp8[:q_offset],
            (h0, h1) if head_ids is None else None,
            pool.get_index_k_with_scale_buffer(layer_id=layer_id),
            pool.get_index_k_chunk_sum(layer_id),
            metadata.get_page_table_64(),
            row_batch,
            row_len,
            self.index_topk,
            head_ids=head_ids,
            coarse_block_ids=coarse_block_ids,
            coarse_block_size=coarse_block_size,
            mean_ranked_chunks=mean_ranked_chunks,
            segments=segments,
        )

        if q_offset < q_fp8.shape[0]:
            pad = torch.full(
                (q_fp8.shape[0] - q_offset, output_heads, self.index_topk),
                -1,
                dtype=picks.dtype,
                device=picks.device,
            )
            picks = torch.cat([picks, pad], dim=0)
        return picks

    def _static_group16_coarse_blocks(
        self,
        *,
        q: torch.Tensor,
        weights: torch.Tensor,
        chunk_sum: torch.Tensor,
        block_tables: torch.Tensor,
        row_batch: torch.Tensor,
        row_len: torch.Tensor,
        head_ids: Tuple[int, ...],
    ) -> torch.Tensor:
        """Score coarse regions for one frozen Indexer, never all 64 heads."""
        if len(head_ids) != 1:
            raise ValueError(
                f"static Group16 expects one rank-local Indexer, got {head_ids}"
            )
        static_ids = torch.as_tensor(
            head_ids, dtype=torch.long, device=q.device
        )
        selection = _per_head_paged.misa_topk_heads_paged(
            q.to(torch.bfloat16).index_select(1, static_ids),
            weights.index_select(1, static_ids),
            chunk_sum,
            block_tables,
            row_batch,
            row_len,
            1,
            pooling_block_size=_offline_router.misa_chunk_size(),
            prune_topk=_offline_router.misa_prune_topk(),
            prune_keep_fraction=_offline_router.misa_prune_keep_fraction(),
            return_metadata=True,
        )
        assert isinstance(selection, _per_head_paged.MISASelection)
        return selection.coarse_block_ids

    @staticmethod
    def _early_prefill_rows(
        forward_batch: ForwardBatch,
        metadata: BaseIndexerMetadata,
        min_seq_len: int,
    ) -> Tuple[List[int], int]:
        """Count the dense-warmup rows at the head of each prefill segment.

        A prefill row's visible prefix grows monotonically inside its request, so
        the rows below ``min_seq_len`` are always a prefix of that request's
        segment.  Deriving the counts from the CPU-side extend metadata keeps the
        boundary decision off the critical path: no device synchronisation.
        Returns the per-segment counts and the longest dense prefix seen.
        """
        assert forward_batch.seq_lens_cpu is not None
        seq_lens = forward_batch.seq_lens_cpu.tolist()
        counts: List[int] = []
        widest = 0
        for b, extend_len in enumerate(metadata.get_nsa_extend_len_cpu()):
            extend_len = int(extend_len)
            cached = int(seq_lens[b]) - extend_len
            count = min(max(min_seq_len - 1 - cached, 0), extend_len)
            counts.append(count)
            if count:
                widest = max(widest, cached + count)
        return counts, widest






    def _per_head_ragged_topk(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        row_batch: torch.Tensor,
        row_len: torch.Tensor,
        segments: List[Tuple[int, int, int]],
        *,
        router_heads: Optional[Any],
        router_uses_misa: bool,
    ) -> torch.Tensor:
        """Prefill picks under the decode policy.

        Every prefill row is one query with its own causal prefix, so the sparse
        threshold, the dynamic MISA head set and the coarse pruning are all
        evaluated per row exactly as decode evaluates them per request.  Returns
        ``[num_tokens, G, width]`` request-relative positions, where G is the
        MISA budget under the router and this rank's local head count otherwise.
        """
        nq = row_len.shape[0]
        total_rows = q_fp8.shape[0]
        device = q_fp8.device
        h0, h1 = _per_head_paged.local_head_range(self.n_heads)
        local_pairs = h1 - h0
        if router_heads is not None:
            # Sparse rows carry M candidate lists for the router to assign;
            # dense rows carry one final list per local pair.
            heads = _offline_router.budget()
            min_seq_len = _offline_router.min_seq_len()
        else:
            heads = local_pairs
            min_seq_len = _per_head_paged.per_head_min_seq_len()

        early_counts, dense_width = self._early_prefill_rows(
            forward_batch, metadata, min_seq_len
        )
        n_early = sum(early_counts)
        if layer_id == 0 and _per_head_paged.trace_prefill_enabled():
            print(
                "[experimental_prefill_per_head] "
                f"rows={nq} dense_rows={n_early} dense_width={dense_width} "
                f"min_seq_len={min_seq_len} heads={heads} "
                f"misa={router_uses_misa}",
                flush=True,
            )

        def dense_table(width: int) -> torch.Tensor:
            """``[total_rows, width]`` every visible key, shared across heads."""
            lens = row_len.to(torch.int32)
            if nq < total_rows:
                lens = torch.cat(
                    [
                        lens,
                        torch.zeros(
                            total_rows - nq, dtype=torch.int32, device=device
                        ),
                    ]
                )
            ids = torch.arange(width, dtype=torch.int32, device=device)
            return torch.where(ids.view(1, -1) < lens.view(-1, 1), ids.view(1, -1), -1)

        if n_early == nq:
            # The whole chunk is still below the threshold, so the Assignment
            # Router is bypassed just as it is on dense decode.
            width = _per_head_paged.pad_index_width(dense_width)
            if router_heads is not None:
                _offline_router.mark_router_bypassed(forward_batch, layer_id)
            return dense_table(width).unsqueeze(1).expand(-1, local_pairs, -1)

        coarse_block_ids = None
        misa_selection = None
        head_ids: Optional[Union[torch.Tensor, List[int]]] = None
        router_uses_static_group16 = (
            router_heads is not None and _offline_router.static_group16_enabled()
        )
        if router_uses_misa:
            misa_selection = _per_head_paged.misa_topk_heads_paged(
                q_fp8[:nq],
                weights[:nq],
                forward_batch.token_to_kv_pool.get_index_k_chunk_sum(layer_id),
                metadata.get_page_table_64(),
                row_batch,
                row_len,
                _offline_router.budget(),
                pooling_block_size=_offline_router.misa_chunk_size(),
                prune_topk=_offline_router.misa_prune_topk(),
                prune_keep_fraction=_offline_router.misa_prune_keep_fraction(),
                return_metadata=True,
            )
            assert isinstance(misa_selection, _per_head_paged.MISASelection)
            head_ids = misa_selection.candidate_head_ids
            coarse_block_ids = misa_selection.coarse_block_ids
        elif router_uses_static_group16:
            head_ids = list(router_heads)
            coarse_block_ids = self._static_group16_coarse_blocks(
                q=q_fp8[:nq],
                weights=weights[:nq],
                chunk_sum=forward_batch.token_to_kv_pool.get_index_k_chunk_sum(
                    layer_id
                ),
                block_tables=metadata.get_page_table_64(),
                row_batch=row_batch,
                row_len=row_len,
                head_ids=router_heads,
            )
        elif router_heads is not None:
            head_ids = list(router_heads)


        picks = self._per_head_paged_topk(
            forward_batch,
            layer_id,
            q_fp8,
            metadata,
            row_batch,
            row_len,
            segments,
            head_ids=head_ids,
            coarse_block_ids=coarse_block_ids,
            mean_ranked_chunks=(
                (router_uses_misa or router_uses_static_group16)
                and _offline_router.misa_prune_topk() is not None
            ),
            coarse_block_size=(
                _offline_router.misa_chunk_size()
                if coarse_block_ids is not None
                else _per_head_paged.MISA_DEFAULT_POOLING_BLOCK
            ),
        )

        if (
            router_heads is not None
            and _offline_router.assignment_mode() == "learned"
        ):
            _offline_router.stash_gate(forward_batch, layer_id, weights)
            if router_uses_misa:
                assert isinstance(head_ids, torch.Tensor)
                assert misa_selection is not None
                _offline_router.stash_selected_heads(
                    forward_batch, layer_id, head_ids, total_rows=picks.shape[0]
                )
                _offline_router.stash_misa_features(
                    forward_batch,
                    layer_id,
                    q_fp8[:nq],
                    weights[:nq],
                    misa_selection,
                    total_rows=picks.shape[0],
                )

        if not n_early:
            return picks

        # Mixed chunk: attention takes one index width for the whole batch, so
        # widen the sparse table and swap the dense rows in.
        width = _per_head_paged.pad_index_width(max(picks.shape[-1], dense_width))
        if picks.shape[-1] < width:
            picks = torch.cat(
                [
                    picks,
                    torch.full(
                        (picks.shape[0], heads, width - picks.shape[-1]),
                        -1,
                        dtype=picks.dtype,
                        device=picks.device,
                    ),
                ],
                dim=-1,
            )
        early_mask = torch.zeros(picks.shape[0], dtype=torch.bool, device=device)
        for (start, _end, _req), count in zip(segments, early_counts):
            if count:
                early_mask[start : start + count] = True
        return torch.where(
            early_mask.view(-1, 1, 1),
            dense_table(width).unsqueeze(1).expand(-1, heads, -1),
            picks,
        )






    def _probe_selected_indexer_exact_top720_recall(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        metadata: BaseIndexerMetadata,
        seqlens_32: torch.Tensor,
        candidate_head_ids: torch.Tensor,
        picks: torch.Tensor,
    ) -> tuple[Optional[int], Optional[float]]:
        """Compare the deployed candidate set with exact full-context Top-720.

        This probe-only path runs one single-head DeepGEMM scan for the static
        Indexer assigned to this TP rank.  Unit weight is intentional: query
        quantization scale is positive and constant across keys, so it cannot
        change a single head's token ranking.
        """
        if not _headmap_probe.exact_indexer_recall_enabled():
            return None, None
        from sglang.srt.layers.dp_attention import get_attention_tp_rank

        selected_head = _headmap_probe.selected_static_indexer(
            layer_id, int(get_attention_tp_rank())
        )
        if selected_head is None:
            raise RuntimeError(
                "exact Indexer recall probe requires "
                "SGLANG_NSA_HEADMAP_PROBE_STATIC_CONFIG"
            )
        slots = (candidate_head_ids[0].long() == selected_head).nonzero().flatten()
        if slots.numel() != 1:
            raise RuntimeError(
                f"static Indexer {selected_head} is absent or duplicated in probe candidates"
            )

        seq_len = int(seqlens_32.reshape(-1)[0].item())
        exact_k = min(_per_head_paged.OUTK, seq_len)
        if exact_k == 0:
            return selected_head, 1.0
        page_table = metadata.get_page_table_64()[:1]
        kv_cache_fp8 = (
            forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
                layer_id=layer_id
            )
        )
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0],
            64,
            1,
            self.head_dim + 4,
        )
        seqlens_2d = seqlens_32.reshape(-1)[:1].to(torch.int32).unsqueeze(-1)
        schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
            seqlens_2d, 64, self.sm_count
        )
        if q_fp8.ndim != 4 or q_fp8.shape[1] != 1:
            raise RuntimeError(
                f"exact Indexer recall probe expects [B,1,H,D] q, got {q_fp8.shape}"
            )
        selected_q = q_fp8[:1]
        # SM90 DeepGEMM only instantiates H=32/64 paged MQA.  Keep the native
        # 64-head shape and zero all but the selected head; the fused reduction
        # is then exactly that head's ReLU(QK) score.
        unit_weight = torch.zeros(
            (1, selected_q.shape[2]), dtype=torch.float32, device=q_fp8.device
        )
        unit_weight[0, selected_head] = 1.0
        logits = deep_gemm.fp8_paged_mqa_logits(
            selected_q,
            kv_cache_fp8,
            unit_weight,
            seqlens_2d,
            page_table,
            schedule_metadata,
            page_table.shape[1] * 64,
            clean_logits=False,
        )
        exact_ids = logits[0, :seq_len].topk(
            exact_k, largest=True, sorted=False
        ).indices
        approximate_ids = picks[0, int(slots.item())].long()
        approximate_ids = approximate_ids[
            (approximate_ids >= 0) & (approximate_ids < seq_len)
        ]
        recall = torch.isin(exact_ids, approximate_ids).float().mean()
        return selected_head, float(recall.item())


    def _probe_exact_dense_indexer_topk(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        seqlens_32: torch.Tensor,
    ) -> tuple[
        Optional[dict[int, torch.Tensor]],
        Optional[dict[int, torch.Tensor]],
        Optional[dict[str, torch.Tensor]],
        Optional[dict[str, torch.Tensor]],
        Optional[dict[str, float]],
    ]:
        """Compute full-context FP8 Indexer Top-K ablations for probe rows.

        Unlike the deployed chunk selector, this materializes all 64 by
        sequence-length QK scores.  It is intentionally probe-only.
        """
        if not _headmap_probe.exact_dense_ablation_enabled():
            return None, None, None, None, None
        if q_fp8.ndim != 4 or q_fp8.shape[:2] != (1, 1):
            raise RuntimeError(
                "exact dense Indexer ablation requires q_fp8 shaped [1,1,64,D], "
                f"got {tuple(q_fp8.shape)}"
            )
        if weights.shape != (1, q_fp8.shape[2]):
            raise RuntimeError(
                "exact dense Indexer ablation requires weights shaped [1,64], "
                f"got {tuple(weights.shape)}"
            )

        seq_len = int(seqlens_32.reshape(-1)[0].item())
        page_table = metadata.get_page_table_64()[:1]
        k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
            layer_id, seq_len, page_table[0]
        )
        k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
            layer_id, seq_len, page_table[0]
        )
        k_dtype = (
            torch.float8_e4m3fnuz if _is_fp8_fnuz else torch.float8_e4m3fn
        )
        key = (
            k_fp8.view(k_dtype).float()
            * k_scale.view(torch.float32).squeeze(-1).unsqueeze(-1)
        )
        raw_qk = torch.matmul(q_fp8[0, 0].float(), key.transpose(0, 1))
        # The per-head chunk16 selector ranks raw QK.  ReLU is applied only
        # when reproducing the production 64-head merged DSA score.
        per_indexer_score = raw_qk
        merged_score = (
            torch.relu(raw_qk) * weights[0].float().unsqueeze(-1)
        ).sum(dim=0)

        per_indexer_ids = {}
        merged_ids = {}
        for budget in (720, 2048):
            k = min(budget, seq_len)
            per_indexer_ids[budget] = per_indexer_score.topk(
                k, dim=-1, largest=True, sorted=False
            ).indices.unsqueeze(0)
            merged_ids[budget] = merged_score.topk(
                k, largest=True, sorted=False
            ).indices.unsqueeze(0)

        chunk_size = 16
        chunks = (seq_len + chunk_size - 1) // chunk_size
        padded_len = chunks * chunk_size
        raw_qk_padded = torch.full(
            (raw_qk.shape[0], padded_len),
            -torch.inf,
            dtype=raw_qk.dtype,
            device=raw_qk.device,
        )
        raw_qk_padded[:, :seq_len] = raw_qk
        qk_blocks = raw_qk_padded.view(raw_qk.shape[0], chunks, chunk_size)
        chunk_counts = torch.full(
            (chunks,), chunk_size, dtype=torch.float32, device=raw_qk.device
        )
        chunk_counts[-1] = seq_len - (chunks - 1) * chunk_size
        finite_qk = torch.where(
            torch.isfinite(qk_blocks), qk_blocks, torch.zeros_like(qk_blocks)
        )
        mean_chunk_score = finite_qk.sum(dim=-1) / chunk_counts.unsqueeze(0)

        key_min_pad = torch.full(
            (padded_len, key.shape[1]),
            torch.inf,
            dtype=key.dtype,
            device=key.device,
        )
        key_max_pad = torch.full_like(key_min_pad, -torch.inf)
        key_min_pad[:seq_len] = key
        key_max_pad[:seq_len] = key
        chunk_min = key_min_pad.view(chunks, chunk_size, -1).amin(dim=1)
        chunk_max = key_max_pad.view(chunks, chunk_size, -1).amax(dim=1)
        q = q_fp8[0, 0].float()
        quest_chunk_score = torch.matmul(
            torch.clamp_min(q, 0), chunk_max.transpose(0, 1)
        ) + torch.matmul(
            torch.clamp_max(q, 0), chunk_min.transpose(0, 1)
        )

        def quota_ids(
            chunk_score: torch.Tensor,
            selected_chunks: int,
            dense_chunks: int,
        ) -> torch.Tensor:
            ranked_score = chunk_score.clone()
            ranked_score[:, 0] = torch.finfo(ranked_score.dtype).max
            if chunks > 1:
                ranked_score[:, -1] = (
                    torch.finfo(ranked_score.dtype).max / 2
                )
            selected = ranked_score.topk(
                selected_chunks, dim=-1, largest=True, sorted=True
            ).indices
            selected_qk = qk_blocks.gather(
                1, selected.unsqueeze(-1).expand(-1, -1, chunk_size)
            )
            relative = selected_qk.topk(
                8, dim=-1, largest=True, sorted=False
            ).indices
            token_ids = selected.unsqueeze(-1) * chunk_size + relative
            dense = token_ids[:, :dense_chunks, :8].reshape(
                token_ids.shape[0], -1
            )
            sparse = token_ids[:, dense_chunks:, :4].reshape(
                token_ids.shape[0], -1
            )
            result = torch.cat([dense, sparse], dim=-1)
            return result.masked_fill(result >= seq_len, -1)

        fixed_chunks = min(_per_head_paged.SEL, chunks)
        fixed_dense = min(_per_head_paged.NDENSE, fixed_chunks)
        per_indexer_ablation_ids = {
            "mean_fixed_128_quota_52x8_76x4": quota_ids(
                mean_chunk_score, fixed_chunks, fixed_dense
            ).unsqueeze(0),
            "quest_fixed_128_quota_52x8_76x4": quota_ids(
                quest_chunk_score, fixed_chunks, fixed_dense
            ).unsqueeze(0),
        }
        shared_ablation_ids = {}
        ablation_stats = {}
        for fraction in (0.05, 0.10):
            candidate_chunks = min(
                chunks, max(1, math.ceil(chunks * fraction))
            )
            quest_ranked_score = quest_chunk_score.clone()
            quest_ranked_score[:, 0] = torch.finfo(
                quest_ranked_score.dtype
            ).max
            if chunks > 1:
                quest_ranked_score[:, -1] = (
                    torch.finfo(quest_ranked_score.dtype).max / 2
                )
            candidate_ids = quest_ranked_score.topk(
                candidate_chunks, dim=-1, largest=True, sorted=False
            ).indices
            candidate_mask = torch.zeros_like(
                quest_chunk_score, dtype=torch.bool
            )
            candidate_mask.scatter_(1, candidate_ids, True)
            candidate_mean_score = mean_chunk_score.masked_fill(
                ~candidate_mask, -torch.inf
            )
            final_ids = quota_ids(
                candidate_mean_score, fixed_chunks, fixed_dense
            )
            prefix = f"quest_{int(fraction * 100)}pct_then_mean"
            per_indexer_ablation_ids[
                f"{prefix}_fixed128_quota_top720"
            ] = final_ids.unsqueeze(0)

            candidate_qk = qk_blocks.gather(
                1,
                candidate_ids.unsqueeze(-1).expand(-1, -1, chunk_size),
            ).reshape(raw_qk.shape[0], -1)
            candidate_token_ids = (
                candidate_ids.unsqueeze(-1) * chunk_size
                + torch.arange(
                    chunk_size, device=raw_qk.device
                ).view(1, 1, -1)
            ).reshape(raw_qk.shape[0], -1)
            hard_topk_slots = candidate_qk.topk(
                min(_per_head_paged.OUTK, candidate_qk.shape[-1]),
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
            hard_topk_ids = candidate_token_ids.gather(1, hard_topk_slots)
            per_indexer_ablation_ids[
                f"quest_{int(fraction * 100)}pct_all_keys_hard_top720"
            ] = hard_topk_ids.unsqueeze(0)

            if fraction == 0.10:
                mean_ranked_score = mean_chunk_score.clone()
                mean_ranked_score[:, 0] = torch.finfo(
                    mean_ranked_score.dtype
                ).max
                if chunks > 1:
                    mean_ranked_score[:, -1] = (
                        torch.finfo(mean_ranked_score.dtype).max / 2
                    )
                mean_candidate_ids = mean_ranked_score.topk(
                    candidate_chunks, dim=-1, largest=True, sorted=False
                ).indices
                mean_candidate_qk = qk_blocks.gather(
                    1,
                    mean_candidate_ids.unsqueeze(-1).expand(
                        -1, -1, chunk_size
                    ),
                ).reshape(raw_qk.shape[0], -1)
                mean_candidate_token_ids = (
                    mean_candidate_ids.unsqueeze(-1) * chunk_size
                    + torch.arange(
                        chunk_size, device=raw_qk.device
                    ).view(1, 1, -1)
                ).reshape(raw_qk.shape[0], -1)
                mean_hard_slots = mean_candidate_qk.topk(
                    min(
                        _per_head_paged.OUTK,
                        mean_candidate_qk.shape[-1],
                    ),
                    dim=-1,
                    largest=True,
                    sorted=False,
                ).indices
                per_indexer_ablation_ids[
                    "mean_10pct_all_keys_hard_top720"
                ] = mean_candidate_token_ids.gather(
                    1, mean_hard_slots
                ).unsqueeze(0)

            ablation_stats[f"{prefix}_candidate_chunks_per_head"] = float(
                candidate_chunks
            )
            ablation_stats[f"{prefix}_candidate_chunk_fraction"] = float(
                candidate_chunks / chunks
            )
        return (
            per_indexer_ids,
            merged_ids,
            per_indexer_ablation_ids,
            shared_ablation_ids,
            ablation_stats,
        )


    def _prepare_group16_all64_probe(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        seqlens_32: torch.Tensor,
        raw_query: torch.Tensor,
        raw_gate: torch.Tensor,
    ) -> None:
        """Prepare one identified candidate set for the Group16 teacher."""
        identity = _headmap_probe.request_info(forward_batch)
        if (
            not _headmap_probe.enabled()
            or identity is None
            or not _headmap_probe.wants_layer(layer_id)
            or not _headmap_probe.wants_sample(layer_id)
            or not forward_batch.forward_mode.is_decode_or_idle()
            or seqlens_32.numel() != 1
            or int(seqlens_32.reshape(-1)[0].item())
            != identity.expected_seq_len
        ):
            return

        row_batch = torch.zeros(1, dtype=torch.int32, device=q_fp8.device)
        chunk_sum = (
            forward_batch.token_to_kv_pool.get_index_k_chunk_sum(layer_id)
        )
        page_table = metadata.get_page_table_64()
        row_len = seqlens_32.reshape(-1)[:1]
        importance_variants: dict[str, torch.Tensor] = {}
        selection = _per_head_paged.misa_topk_heads_paged(
            q_fp8.squeeze(1)[:1],
            weights[:1],
            chunk_sum,
            page_table,
            row_batch,
            row_len,
            _headmap_probe.candidate_budget(),
            pooling_block_size=_headmap_probe.pooling_block_size(),
            prune_keep_fraction=_headmap_probe.prune_keep_fraction(),
            prune_topk=_headmap_probe.prune_topk(),
            return_metadata=True,
            importance_variant_output=importance_variants,
        )
        assert isinstance(selection, _per_head_paged.MISASelection)
        if selection.candidate_head_ids.shape[1] == q_fp8.shape[2]:
            # All-64 collection has no head-selection stage. Canonicalize the
            # candidate axis so every TP rank writes IDs in exact 0..63 order.
            order = selection.candidate_head_ids.argsort(dim=1)

            def gather_candidate(value: torch.Tensor) -> torch.Tensor:
                index = order
                while index.ndim < value.ndim:
                    index = index.unsqueeze(-1)
                return torch.gather(value, 1, index.expand_as(value))

            selection = _per_head_paged.MISASelection(
                candidate_head_ids=gather_candidate(
                    selection.candidate_head_ids
                ),
                coarse_block_ids=gather_candidate(selection.coarse_block_ids),
                candidate_importance=gather_candidate(
                    selection.candidate_importance
                ),
                candidate_context_summary=gather_candidate(
                    selection.candidate_context_summary
                ),
                candidate_context_stats=gather_candidate(
                    selection.candidate_context_stats
                ),
            )

        candidate_top8_sum = None
        candidate_top8_context_summary = None
        if _headmap_probe.save_group_router_features():
            router_variants: dict[str, torch.Tensor] = {}
            router_selection = _per_head_paged.misa_topk_heads_paged(
                raw_query[:1],
                raw_gate[:1],
                chunk_sum,
                page_table,
                row_batch,
                row_len,
                raw_query.shape[1],
                pooling_block_size=256,
                prune_topk=8,
                return_metadata=True,
                importance_variant_output=router_variants,
            )
            assert isinstance(router_selection, _per_head_paged.MISASelection)
            primary_ids = selection.candidate_head_ids
            candidate_top8_sum = torch.gather(
                router_variants["raw_top_count_8"], 1, primary_ids
            )
            inverse_slots = torch.empty_like(router_selection.candidate_head_ids)
            inverse_slots.scatter_(
                1,
                router_selection.candidate_head_ids,
                torch.arange(
                    router_selection.candidate_head_ids.shape[1],
                    device=q_fp8.device,
                ).view(1, -1),
            )
            router_slots = torch.gather(inverse_slots, 1, primary_ids)
            candidate_top8_context_summary = torch.gather(
                router_selection.candidate_context_summary,
                1,
                router_slots.unsqueeze(-1).expand(
                    -1,
                    -1,
                    router_selection.candidate_context_summary.shape[-1],
                ),
            )

        picks = self._per_head_paged_topk(
            forward_batch,
            layer_id,
            q_fp8.squeeze(1),
            metadata,
            row_batch,
            row_len,
            None,
            head_ids=selection.candidate_head_ids,
            coarse_block_ids=selection.coarse_block_ids,
            coarse_block_size=_headmap_probe.pooling_block_size(),
            mean_ranked_chunks=_headmap_probe.prune_topk() is not None,
        )
        selected_indexer_head_id, selected_indexer_exact_top720_recall = (
            self._probe_selected_indexer_exact_top720_recall(
                forward_batch,
                layer_id,
                q_fp8,
                metadata,
                seqlens_32,
                selection.candidate_head_ids,
                picks,
            )
        )
        (
            exact_indexer_topk_ids,
            exact_merged_topk_ids,
            selector_ablation_per_indexer_ids,
            selector_ablation_shared_ids,
            selector_ablation_stats,
        ) = self._probe_exact_dense_indexer_topk(
            forward_batch,
            layer_id,
            q_fp8,
            weights,
            metadata,
            seqlens_32,
        )
        _headmap_probe.put_picks(
            layer_id,
            picks,
            request_id=identity.request_id,
            selected_indexer_head_id=selected_indexer_head_id,
            selected_indexer_exact_top720_recall=(
                selected_indexer_exact_top720_recall
            ),
            exact_indexer_topk_ids=exact_indexer_topk_ids,
            exact_merged_topk_ids=exact_merged_topk_ids,
            selector_ablation_per_indexer_ids=(
                selector_ablation_per_indexer_ids
            ),
            selector_ablation_shared_ids=selector_ablation_shared_ids,
            selector_ablation_stats=selector_ablation_stats,
            all_indexer_q=raw_query,
            all_indexer_gate=raw_gate,
            candidate_head_ids=selection.candidate_head_ids,
            candidate_importance=selection.candidate_importance,
            candidate_context_summary=selection.candidate_context_summary,
            candidate_context_stats=selection.candidate_context_stats,
            candidate_top8_sum=candidate_top8_sum,
            candidate_top8_context_summary=candidate_top8_context_summary,
            importance_variants=importance_variants,
        )





    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        raw_query: torch.Tensor,
        raw_gate: torch.Tensor,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        if _is_cuda and get_adaptive_hisa_config().mode == "adaptive_decode":
            from sglang.srt.layers.attention.nsa.adaptive_hisa.decode_runtime import maybe_decode_topk

            adaptive_result = maybe_decode_topk(
                forward_batch, layer_id, q_fp8, weights, metadata, self.index_topk
            )
            if adaptive_result is not None:
                return adaptive_result

        per_head_enabled = _per_head_paged.per_head_index_enabled() and not _is_hip
        router_heads = (
            _offline_router.candidate_head_ids(layer_id) if not _is_hip else None
        )
        router_uses_misa = (
            router_heads is not None and _offline_router.head_selection_mode() == "misa"
        )
        router_uses_static_group16 = (
            router_heads is not None and _offline_router.static_group16_enabled()
        )
        if router_heads is not None and per_head_enabled:
            raise RuntimeError(
                "offline unique-head router and fixed per-head index mode cannot "
                "be enabled together"
            )

        experimental_picks = None
        experimental_early_mask = None
        all_late = True
        all_early = False
        if per_head_enabled or router_heads is not None:
            if not forward_batch.forward_mode.is_decode_or_idle():
                raise NotImplementedError(
                    "experimental paged indexer: speculative decoding is not supported"
                )
            seqlens = metadata.get_seqlens_int32()
            bsz = seqlens.shape[0]
            decode_start = _per_head_paged.per_head_decode_start_token()
            min_seq_len = (
                _offline_router.min_seq_len()
                if router_heads is not None
                else _per_head_paged.per_head_min_seq_len()
            )
            seq_lens_cpu = forward_batch.seq_lens_cpu
            if seq_lens_cpu is None or len(seq_lens_cpu) != bsz:
                raise RuntimeError(
                    "seq_lens_cpu and indexer batch size disagree: "
                    f"{None if seq_lens_cpu is None else len(seq_lens_cpu)} != {bsz}"
                )
            generated_lens = forward_batch.generated_lens_cpu
            if decode_start > 0:
                if generated_lens is None:
                    raise RuntimeError(
                        "SGLANG_NSA_EXPERIMENTAL_DECODE_START_TOKEN requires "
                        "ForwardBatch.generated_lens_cpu"
                    )
                if len(generated_lens) != bsz:
                    raise RuntimeError(
                        "generated_lens_cpu and indexer batch size disagree: "
                        f"{len(generated_lens)} != {bsz}"
                    )
            early_flags = [
                seq_len < min_seq_len
                or (
                    decode_start > 0
                    and generated_lens is not None
                    and generated_len < decode_start
                )
                for seq_len, generated_len in zip(
                    seq_lens_cpu,
                    generated_lens if generated_lens is not None else [0] * bsz,
                )
            ]
            all_late = not any(early_flags)
            all_early = all(early_flags)
            # Emit an auditable, low-volume marker at either policy boundary.
            # Restricting this to layer 0 avoids 61 log lines per worker.
            if layer_id == 0 and (
                any(length == min_seq_len for length in seq_lens_cpu)
                or (
                    decode_start > 0
                    and generated_lens is not None
                    and any(length == decode_start for length in generated_lens)
                )
            ):
                print(
                    "[experimental_sparse_transition] "
                    f"min_seq_len={min_seq_len} decode_start={decode_start} "
                    f"seq_lens={seq_lens_cpu} generated_lens={generated_lens} "
                    f"early={early_flags}",
                    flush=True,
                )
            # Avoid a GPU allocation in the common all-early/all-late cases.
            if not all_late and not all_early:
                experimental_early_mask = torch.tensor(
                    early_flags,
                    dtype=torch.bool,
                    device=seqlens.device,
                )

            if all_early:
                # True dense warmup: before the sparse threshold, return every
                # visible request-relative key rather than the model's shared
                # Top-K (2048).  The list is shared across heads, so the
                # Assignment Router is intentionally bypassed, but it is still
                # emitted on the per-head contract because only that attention
                # path accepts a width other than index_topk.
                width = _per_head_paged.pad_index_width(
                    max(int(length) for length in seq_lens_cpu)
                )
                token_ids = torch.arange(
                    width, dtype=torch.int32, device=seqlens.device
                )
                full_indices = torch.where(
                    token_ids.view(1, -1) < seqlens.view(-1, 1),
                    token_ids.view(1, -1),
                    -1,
                )
                if bsz < q_fp8.shape[0]:
                    full_indices = torch.cat(
                        [
                            full_indices,
                            torch.full(
                                (q_fp8.shape[0] - bsz, width),
                                -1,
                                dtype=torch.int32,
                                device=seqlens.device,
                            ),
                        ],
                        dim=0,
                    )
                # The head axis is this rank's local pair count, not the MISA
                # budget: these are already final per-pair indices, so routing
                # must be skipped rather than fed 8 identical candidate lists.
                h0, h1 = _per_head_paged.local_head_range(self.n_heads)
                if router_heads is not None:
                    _offline_router.mark_router_bypassed(forward_batch, layer_id)
                return full_indices.unsqueeze(1).expand(-1, h1 - h0, -1)

            # With no delay, or once every request crossed the delay, retain
            # the original fast path and avoid computing the official shared
            # selector.  If every request is still early, fall through to the
            # shared selector below.  A mixed batch computes both policies and
            # merges per row after the shared result is available.
            coarse_block_ids = None
            misa_selection = None
            if router_uses_misa and not all_early:
                misa_selection = _per_head_paged.misa_topk_heads_paged(
                    q_fp8[:bsz],
                    weights[:bsz],
                    forward_batch.token_to_kv_pool.get_index_k_chunk_sum(layer_id),
                    metadata.get_page_table_64(),
                    torch.arange(bsz, device=seqlens.device, dtype=torch.int32),
                    seqlens,
                    _offline_router.budget(),
                    pooling_block_size=_offline_router.misa_chunk_size(),
                    prune_topk=_offline_router.misa_prune_topk(),
                    prune_keep_fraction=_offline_router.misa_prune_keep_fraction(),
                    return_metadata=True,
                )
                assert isinstance(misa_selection, _per_head_paged.MISASelection)
                head_ids = misa_selection.candidate_head_ids
                coarse_block_ids = misa_selection.coarse_block_ids
            elif router_uses_static_group16 and not all_early:
                # Score coarse blocks for this rank's one frozen Indexer only.
                # Passing all 64 heads here would erase the Static speedup.
                head_ids = list(router_heads)
                coarse_block_ids = self._static_group16_coarse_blocks(
                    q=q_fp8[:bsz],
                    weights=weights[:bsz],
                    chunk_sum=forward_batch.token_to_kv_pool.get_index_k_chunk_sum(
                        layer_id
                    ),
                    block_tables=metadata.get_page_table_64(),
                    row_batch=torch.arange(
                        bsz, device=seqlens.device, dtype=torch.int32
                    ),
                    row_len=seqlens,
                    head_ids=router_heads,
                )
            elif router_uses_misa:
                head_ids = None
            else:
                head_ids = list(router_heads) if router_heads is not None else None
            if all_late:
                picks = self._per_head_paged_topk(
                    forward_batch,
                    layer_id,
                    q_fp8,
                    metadata,
                    torch.arange(bsz, device=seqlens.device, dtype=torch.int32),
                    seqlens,
                    None,
                    head_ids=head_ids,
                    coarse_block_ids=coarse_block_ids,
                    coarse_block_size=_offline_router.misa_chunk_size(),
                    mean_ranked_chunks=(
                        (router_uses_misa or router_uses_static_group16)
                        and _offline_router.misa_prune_topk() is not None
                    ),
                )
                if (
                    router_heads is not None
                    and _offline_router.assignment_mode() == "learned"
                ):
                    _offline_router.stash_gate(forward_batch, layer_id, weights)
                    if router_uses_misa:
                        assert isinstance(head_ids, torch.Tensor)
                        assert misa_selection is not None
                        _offline_router.stash_selected_heads(
                            forward_batch,
                            layer_id,
                            head_ids,
                            total_rows=picks.shape[0],
                        )
                        _offline_router.stash_misa_features(
                            forward_batch,
                            layer_id,
                            q_fp8[:bsz],
                            weights[:bsz],
                            misa_selection,
                            total_rows=picks.shape[0],
                        )
                return picks
            if not all_early:
                experimental_picks = self._per_head_paged_topk(
                    forward_batch,
                    layer_id,
                    q_fp8,
                    metadata,
                    torch.arange(bsz, device=seqlens.device, dtype=torch.int32),
                    seqlens,
                    None,
                    head_ids=head_ids,
                    coarse_block_ids=coarse_block_ids,
                    coarse_block_size=_offline_router.misa_chunk_size(),
                    mean_ranked_chunks=(
                        (router_uses_misa or router_uses_static_group16)
                        and _offline_router.misa_prune_topk() is not None
                    ),
                )
                if (
                    router_heads is not None
                    and _offline_router.assignment_mode() == "learned"
                ):
                    _offline_router.stash_gate(forward_batch, layer_id, weights)
                    if router_uses_misa:
                        assert isinstance(head_ids, torch.Tensor)
                        assert misa_selection is not None
                        _offline_router.stash_selected_heads(
                            forward_batch,
                            layer_id,
                            head_ids,
                            total_rows=experimental_picks.shape[0],
                        )
                        _offline_router.stash_misa_features(
                            forward_batch,
                            layer_id,
                            q_fp8[:bsz],
                            weights[:bsz],
                            misa_selection,
                            total_rows=experimental_picks.shape[0],
                        )

        page_size = forward_batch.token_to_kv_pool.page_size
        # NOTE(dark): blocksize = 64 is hardcoded in deep_gemm
        if _is_hip:
            assert page_size == 1, "only support page size 1"
            block_tables = metadata.get_page_table_1()
        else:
            assert page_size == 64, "only support page size 64"
            # NOTE(dark): this support extend/decode/decode+graph
            block_tables = metadata.get_page_table_64()

        max_seq_len = block_tables.shape[1] * page_size
        kv_cache_fp8 = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer_id
        )

        blocksize = page_size
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            seqlens_32 = metadata.get_seqlens_expanded()
        else:
            seqlens_32 = metadata.get_seqlens_int32()

        # Reuse pre-computed schedule metadata if available (from init_forward_metadata), otherwise fall back to computing it here.
        schedule_metadata = getattr(metadata, "paged_mqa_schedule_metadata", None)
        # DeepGEMM release-0426 requires context_lens of shape [batch_size, next_n]
        # to match q.shape = [batch_size, next_n, heads, head_dim]. The indexer uses
        # next_n=1 with batch_size=N_total via q_fp8.unsqueeze(1) below, so mirror that layout here.

        if seqlens_32.dim() == 2:
            seqlens_32_2d = seqlens_32
        else:
            seqlens_32_2d = seqlens_32.unsqueeze(-1)
        if _is_cuda:
            if schedule_metadata is None:
                schedule_metadata = deep_gemm.get_paged_mqa_logits_metadata(
                    seqlens_32_2d, blocksize, self.sm_count)

        assert len(q_fp8.shape) == 3
        q_fp8 = q_fp8.unsqueeze(1)  # the next_n dim is 1 now
        assert len(kv_cache_fp8.shape) == 2
        block_kv = 1 if _is_hip else 64
        num_heads_kv = 1
        head_dim_with_sf = 132
        if _is_hip:
            kv_cache_fp8 = kv_cache_fp8.view(
                -1, block_kv, num_heads_kv, head_dim_with_sf
            )
        else:
            kv_cache_fp8 = kv_cache_fp8.view(
                kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
            )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)

        # When attn_tp_size > 1 or in the MAX_LEN padding mode, padding may exist in the hidden states,
        # and it is necessary to extract the actual q length.
        q_offset = sum(metadata.get_nsa_extend_len_cpu())
        if _is_hip:
            from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

            batch_size, next_n, heads, _ = q_fp8.shape
            logits = torch.full(
                (batch_size * next_n, max_seq_len),
                float("-inf"),
                device=q_fp8.device,
                dtype=torch.float32,
            )
            deepgemm_fp8_paged_mqa_logits(
                q_fp8,
                kv_cache_fp8,
                weights,
                logits,
                seqlens_32,
                block_tables,
                max_seq_len,
                Preshuffle=False,
                KVBlockSize=block_kv,
            )
        else:
            logits = deep_gemm.fp8_paged_mqa_logits(
                q_fp8[:q_offset],
                kv_cache_fp8,
                weights[:q_offset],
                seqlens_32_2d,
                block_tables,
                schedule_metadata,
                max_seq_len,
                clean_logits=False,
            )

        # NOTE(dark): logits should be cleaned in topk_transform
        topk_result = metadata.topk_transform(logits, self.index_topk)
        # Restore possible padding exist in the hidden states.
        if not _is_hip and q_offset < q_fp8.shape[0]:
            pad_len = q_fp8.shape[0] - q_offset
            padding = torch.full(
                (pad_len, topk_result.shape[1]),
                -1,
                dtype=topk_result.dtype,
                device=topk_result.device,
            )
            topk_result = torch.cat([topk_result, padding], dim=0)
        if experimental_picks is not None:
            # Attention accepts a single rank for the whole batch.  Promote
            # every visible key to the per-head contract for early rows, while
            # retaining sparse per-head picks for requests past the threshold.
            assert experimental_early_mask is not None
            g = experimental_picks.shape[1]
            if experimental_early_mask.shape[0] < topk_result.shape[0]:
                padding = torch.ones(
                    topk_result.shape[0] - experimental_early_mask.shape[0],
                    dtype=torch.bool,
                    device=experimental_early_mask.device,
                )
                experimental_early_mask = torch.cat(
                    [experimental_early_mask, padding], dim=0
                )
            dense_width = _per_head_paged.pad_index_width(
                max(
                    experimental_picks.shape[-1],
                    max(
                        int(length)
                        for length, early in zip(seq_lens_cpu, early_flags)
                        if early
                    ),
                )
            )
            dense_ids = torch.arange(
                dense_width, dtype=torch.int32, device=topk_result.device
            )
            dense_shared = torch.full(
                (topk_result.shape[0], dense_width),
                -1,
                dtype=torch.int32,
                device=topk_result.device,
            )
            dense_shared[:bsz] = torch.where(
                dense_ids.view(1, -1) < seqlens.view(-1, 1),
                dense_ids.view(1, -1),
                -1,
            )
            dense_per_head = dense_shared.unsqueeze(1).expand(-1, g, -1)
            if experimental_picks.shape[-1] < dense_width:
                experimental_picks = torch.cat(
                    [
                        experimental_picks,
                        torch.full(
                            (
                                experimental_picks.shape[0],
                                g,
                                dense_width - experimental_picks.shape[-1],
                            ),
                            -1,
                            dtype=experimental_picks.dtype,
                            device=experimental_picks.device,
                        ),
                    ],
                    dim=-1,
                )
            topk_result = torch.where(
                experimental_early_mask.view(-1, 1, 1),
                dense_per_head,
                experimental_picks,
            )

        self._prepare_group16_all64_probe(
            forward_batch,
            layer_id,
            q_fp8,
            weights,
            metadata,
            seqlens_32,
            raw_query,
            raw_gate,
        )
        return topk_result





    def _should_chunk_mqa_logits(
        self, num_q: int, num_k: int, device: torch.device
    ) -> Tuple[bool, int]:
        """
        Detect whether we need to chunk the MQA logits computation to avoid OOM
        Return: (need_chunk, free_mem)
        """
        # Quick static check for normal batches
        if num_q * num_k < 8_000_000:  # 8M elements ≈ 32MB logits
            return False, 0

        free_mem, total_mem = torch.cuda.mem_get_info(device)
        bytes_per_elem = 4  # float32
        logits_bytes = num_q * num_k * bytes_per_elem

        # Logits should not exceed 50% of free memory or 30% of total memory
        need_chunk = (logits_bytes * 2 > free_mem) or (logits_bytes > total_mem * 0.3)
        return need_chunk, free_mem




    # prefill入口
    def _get_topk_ragged(
        self,
        enable_dual_stream: bool,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)
        assert forward_batch.forward_mode.is_extend_without_speculative()
        page_size = forward_batch.token_to_kv_pool.page_size
        if _is_hip:
            assert page_size == 1, "only support page size 1"
        else:
            assert page_size == 64, "only support page size 64"

        assert len(weights.shape) == 3
        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        weights = weights.squeeze(-1)

        if _is_hip:
            block_tables = metadata.get_page_table_1()
        else:
            block_tables = metadata.get_page_table_64()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )

        batch_size = len(block_tables)
        token_nums, _, _ = q_fp8.shape
        device = q_fp8.device
        per_head = _per_head_paged.per_head_index_enabled() and not _is_hip
        router_heads = (
            _offline_router.candidate_head_ids(layer_id) if not _is_hip else None
        )
        router_uses_misa = (
            router_heads is not None and _offline_router.head_selection_mode() == "misa"
        )
        if router_heads is not None and per_head:
            raise RuntimeError(
                "offline unique-head router and fixed per-head index mode cannot "
                "be enabled together"
            )
        if not _per_head_paged.prefill_per_head_enabled():
            # Decode-only configuration: prefill keeps the official shared
            # selector, so fall through to the baseline path below.
            per_head = False
            router_heads = None
            router_uses_misa = False

        if batch_size == 0:
            shape = (token_nums, self.index_topk)
            if per_head:
                h0, h1 = _per_head_paged.local_head_range(self.n_heads)
                shape = (token_nums, h1 - h0, self.index_topk)
            elif router_heads is not None:
                shape = (token_nums, _offline_router.budget(), self.index_topk)
            return torch.full(shape, -1, device=device, dtype=torch.int32)

        if per_head or router_heads is not None:
            # Paged per-head path: no K gather, no packed-buffer chunking.
            row_len = metadata.get_seqlens_expanded()  # position + 1 per query
            segments = []
            start = 0
            for b, n in enumerate(metadata.get_nsa_extend_len_cpu()):
                segments.append((start, start + n, b))
                start += n

            topk_result = self._per_head_ragged_topk(
                forward_batch,
                layer_id,
                q_fp8,
                weights,
                metadata,
                metadata.get_token_to_batch_idx(),
                row_len,
                segments,
                router_heads=router_heads,
                router_uses_misa=router_uses_misa,
            )
            # Experimental per-head/MISA index: no shared K, so no partition.
            return topk_result

        topk_result = torch.full(
            (token_nums, self.index_topk),
            -1,
            device=device,
            dtype=torch.int32,
        )
        ks, ke = metadata.get_indexer_kvcache_range()
        indexer_seq_lens_cpu = metadata.get_indexer_seq_len_cpu()
        seq_len_sum = torch.sum(indexer_seq_lens_cpu).item()
        max_seq_len = torch.max(indexer_seq_lens_cpu).item()

        k_fp8, k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_buffer(
            layer_id,
            metadata.get_indexer_seq_len(),
            block_tables,
            seq_len_sum,
            max_seq_len,
        )

        if _is_fp8_fnuz:
            k_fp8 = k_fp8.view(torch.float8_e4m3fnuz)
        else:
            k_fp8 = k_fp8.view(torch.float8_e4m3fn)

        k_scale = k_scale.view(torch.float32).squeeze(-1)
        kv_fp8 = (k_fp8, k_scale)

        # Check if we need to chunk to avoid OOM
        seq_lens_expanded = metadata.get_seqlens_expanded()
        token_to_batch_idx = metadata.get_token_to_batch_idx()
        q_offset = ks.shape[0]
        k_offset = k_fp8.shape[0]
        build_partition = partition_enabled()

        if build_partition and not _is_hip and seq_lens_expanded.shape[0] == q_offset:
            # Adaptive sparse prefill: Top-2048 from summary-selected leaves +
            # the causal local window of the previous chunk's partition.
            from sglang.srt.layers.attention.nsa.adaptive_hisa.prefill_select import (
                sparse_prefill_admission,
                sparse_prefill_topk,
            )

            admission = sparse_prefill_admission(forward_batch, layer_id, metadata)
            if admission is not None:
                with self._with_real_sm_count():
                    topk_result[:q_offset] = sparse_prefill_topk(
                        forward_batch,
                        layer_id,
                        q_fp8[:q_offset],
                        weights[:q_offset],
                        seq_lens_expanded,
                        block_tables,
                        admission,
                        k_flat=(k_fp8, k_scale) if k_fp8.shape[0] >= admission[3] else None,
                    )
                _schedule_adaptive_partition(
                    forward_batch, layer_id, k_fp8, k_scale, self.scale_fmt
                )
                return topk_result

        # Baseline DSA: use one logits call when memory allows, otherwise run
        # the same algorithm in row chunks.
        need_chunk, free_mem = self._should_chunk_mqa_logits(
            q_offset, k_offset, device
        )

        if not need_chunk:
            assert q_fp8[:q_offset].shape[0] != 0
            with self._with_real_sm_count():
                if _is_hip:
                    from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

                    kv, scale = kv_fp8
                    logits = fp8_mqa_logits(q_fp8[:q_offset], kv, scale, weights[:q_offset], ks, ke)
                else:
                    logits = deep_gemm.fp8_mqa_logits(
                        q_fp8[:q_offset],
                        kv_fp8,
                        weights[:q_offset],
                        ks,
                        ke,
                        clean_logits=False,
                    )

            official_logits = logits[:q_offset]
            assert official_logits.shape[0] == len(seq_lens_expanded)
            assert logits.shape[1] == k_offset
            raw_topk_result = metadata.topk_transform(
                official_logits, self.index_topk, ks=ks
            )
            topk_result[:q_offset] = raw_topk_result
            if build_partition:
                _schedule_adaptive_partition(
                    forward_batch, layer_id, k_fp8, k_scale, self.scale_fmt
                )
            return topk_result

        # Chunk path
        bytes_per_elem = 4  # float32
        bytes_per_row = k_offset * bytes_per_elem
        # Reserve 50% of free memory for logits
        max_rows = max(1, int((free_mem * 0.5) // max(bytes_per_row, 1)))
        max_rows = min(max_rows, q_offset)

        global_topk_offset = metadata.attn_metadata.topk_indices_offset

        assert (
            seq_lens_expanded.shape[0] == q_offset
        ), f"seq_lens_expanded length mismatch: {seq_lens_expanded.shape[0]} != {q_offset}"
        if global_topk_offset is not None:
            assert (
                global_topk_offset.shape[0] >= q_offset
            ), f"topk_indices_offset too short: {global_topk_offset.shape[0]} < {q_offset}"

        start = 0
        while start < q_offset:
            end = min(start + max_rows, q_offset)

            with self._with_real_sm_count():
                if _is_hip:
                    from aiter.ops.triton.fp8_mqa_logits import fp8_mqa_logits

                    kv, scale = kv_fp8
                    logits_chunk = fp8_mqa_logits(
                        q_fp8[start:end],
                        kv,
                        scale,
                        weights[start:end],
                        ks[start:end],
                        ke[start:end],
                    )

                else:
                    logits_chunk = deep_gemm.fp8_mqa_logits(
                        q_fp8[start:end],
                        kv_fp8,
                        weights[start:end],
                        ks[start:end],
                        ke[start:end],
                        clean_logits=False,
                    )

            official_rows = end - start
            official_logits_chunk = logits_chunk[:official_rows]
            lengths_chunk = seq_lens_expanded[start:end]

            # RAGGED: use global offset; PAGED: construct local cu_seqlens_q per chunk
            if global_topk_offset is not None:
                # RAGGED path
                topk_offset_chunk = global_topk_offset[start:end]
                cu_seqlens_q_chunk = None
                batch_idx_chunk = None
            else:
                # PAGED path: treat each token as a length-1 sequence
                topk_offset_chunk = None
                B_chunk = official_logits_chunk.shape[0]
                cu_seqlens_q_chunk = torch.ones(
                    B_chunk, dtype=torch.int32, device=device
                )
                batch_idx_chunk = token_to_batch_idx[start:end]

            raw_topk_chunk = metadata.topk_transform(
                official_logits_chunk,
                self.index_topk,
                ks=ks[start:end],
                cu_seqlens_q=cu_seqlens_q_chunk,
                ke_offset=lengths_chunk,
                batch_idx_list=batch_idx_chunk,
                topk_indices_offset_override=topk_offset_chunk,
            )
            topk_result[start:end] = raw_topk_chunk
            start = end

        if build_partition:
            _schedule_adaptive_partition(
                forward_batch, layer_id, k_fp8, k_scale, self.scale_fmt
            )
        return topk_result







    def _forward_cuda_k_only(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        act_quant,
        enable_dual_stream: bool,
        metadata: BaseIndexerMetadata,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        assert forward_batch.forward_mode.is_extend_without_speculative()
        x_meta = x[0] if isinstance(x, tuple) else x

        # Fast path: only compute and store k cache, skip all q and weights ops
        key = self._get_k_bf16(x, positions, enable_dual_stream)

        if not forward_batch.out_cache_loc.is_contiguous():
            forward_batch.out_cache_loc = forward_batch.out_cache_loc.contiguous()

        self._store_index_k_cache(
            forward_batch=forward_batch,
            layer_id=layer_id,
            key=key,
            act_quant=act_quant,
        )
        _skip_adaptive_partition(forward_batch, layer_id, "k_only_no_query")

        # MHA doesn't need topk_indices
        if not return_indices:
            return None

        # MLA: use dummy logits with topk kernel's fast path to generate indices
        # When length <= 2048, naive_topk_cuda directly generates [0,1,...,length-1,-1,...]
        seq_lens_expanded = metadata.get_seqlens_expanded()
        dummy_logits = torch.zeros(
            seq_lens_expanded.shape[0],
            self.index_topk,
            dtype=torch.float32,
            device=x_meta.device,
        )
        return metadata.topk_transform(dummy_logits, self.index_topk)




    def _get_topk_ragged_with_cp(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
        kv_len: int,
        actual_seq_q: int,
        cp_index: List[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"
        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)
        k_fp8_list = []
        k_scale_list = []
        ks_list = []
        ke_offset_list = []
        offset = 0
        actual_seq_q_list = []
        batch_idx_list = []

        block_tables = metadata.get_page_table_64()

        assert (
            forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )
        if cp_index is not None:
            # TODO Multi-batch support has accuracy issues
            for batch_idx, start_seq_position, end_seq_position in cp_index:
                pre_chunk_offset = (
                    forward_batch.seq_lens_cpu[batch_idx].item()
                    - forward_batch.extend_seq_lens_cpu[batch_idx]
                )
                start_seq_position += pre_chunk_offset
                end_seq_position += pre_chunk_offset
                if offset == 0 and batch_idx != 0:
                    offset += forward_batch.extend_seq_lens_cpu[batch_idx - 1]
                k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                    layer_id,
                    end_seq_position,
                    block_tables[batch_idx],
                )
                k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                    layer_id,
                    end_seq_position,
                    block_tables[batch_idx],
                )

                extend_seq_len = end_seq_position - start_seq_position
                ks = torch.full(
                    (extend_seq_len,), offset, dtype=torch.int32, device="cuda"
                )
                k_fp8_list.append(k_fp8)
                k_scale_list.append(k_scale)
                ks_list.append(ks)
                ke_offset = torch.arange(
                    start_seq_position + 1,
                    end_seq_position + 1,
                    dtype=torch.int32,
                    device="cuda",
                )
                ke_offset_list.append(ke_offset)
                actual_seq_q = torch.tensor(
                    [extend_seq_len], dtype=torch.int32, device="cuda"
                )
                actual_seq_q_list.append(actual_seq_q)
                batch_idx_list.append(batch_idx)

            k_fp8 = torch.cat(k_fp8_list, dim=0).view(torch.float8_e4m3fn)
            k_scale = torch.cat(k_scale_list, dim=0).view(torch.float32).squeeze(-1)
            kv_fp8 = (k_fp8, k_scale)
            ks = torch.cat(ks_list, dim=0)
            ke_offset = torch.cat(ke_offset_list, dim=0)
            ke = ks + ke_offset
            actual_seq_q = torch.cat(actual_seq_q_list, dim=0)
            with self._with_real_sm_count():
                logits = deep_gemm.fp8_mqa_logits(
                    q_fp8,
                    kv_fp8,
                    weights,
                    ks,
                    ke,
                    clean_logits=False,
                )
            topk_result = metadata.topk_transform(
                logits,
                self.index_topk,
                ks=ks,
                cu_seqlens_q=actual_seq_q,
                ke_offset=ke_offset,
                batch_idx_list=batch_idx_list,
            )
        else:
            kv_len = (
                forward_batch.seq_lens_cpu[0].item()
                - forward_batch.extend_seq_lens_cpu[0]
                + kv_len
            )
            k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                layer_id,
                kv_len,
                block_tables[0],
            )
            k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                layer_id,
                kv_len,
                block_tables[0],
            )

            k_fp8 = k_fp8.view(torch.float8_e4m3fn)
            k_scale = k_scale.view(torch.float32).squeeze(-1)
            kv_fp8 = (k_fp8, k_scale)
            ks = torch.full((actual_seq_q,), offset, dtype=torch.int32, device="cuda")
            ke_offset = torch.arange(
                (kv_len - actual_seq_q) + 1,
                kv_len + 1,
                dtype=torch.int32,
                device="cuda",
            )
            ke = ks + ke_offset

            with self._with_real_sm_count():
                logits = deep_gemm.fp8_mqa_logits(
                    q_fp8,
                    kv_fp8,
                    weights,
                    ks,
                    ke,
                    clean_logits=False,
                )
            actual_seq_q = torch.tensor([actual_seq_q], dtype=torch.int32).to(
                device="cuda", non_blocking=True
            )
            topk_result = metadata.topk_transform(
                logits,
                self.index_topk,
                ks=ks,
                cu_seqlens_q=actual_seq_q,
                ke_offset=ke_offset,
            )

        return topk_result



    def forward_indexer(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        forward_batch: ForwardBatch,
        topk: int,
        layer_id: int,
    ) -> Optional[torch.Tensor]:
        if not _is_npu:
            from sglang.srt.layers.attention.nsa.tilelang_kernel import fp8_index

        page_size = forward_batch.token_to_kv_pool.page_size
        assert page_size == 64, "only support page size 64"

        assert len(weights.shape) == 3
        weights = weights.squeeze(-1)

        # logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)
        k_fp8_list = []
        k_scale_list = []

        topk_indices_list = []

        block_tables = forward_batch.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :
        ]
        strided_indices = torch.arange(
            0, block_tables.shape[-1], page_size, device="cuda"
        )
        block_tables = block_tables[:, strided_indices] // page_size

        q_len_start = 0

        for i in range(forward_batch.batch_size):
            seq_len = forward_batch.seq_lens[i].item()
            q_len = (
                forward_batch.extend_seq_lens_cpu[i]
                if forward_batch.forward_mode.is_extend()
                else 1
            )
            q_len_end = q_len_start + q_len

            q_fp8_partial = q_fp8[q_len_start:q_len_end]
            q_fp8_partial = q_fp8_partial.unsqueeze(0).contiguous()

            weights_partial = weights[q_len_start:q_len_end]
            weights_partial = weights_partial.squeeze(-1).unsqueeze(0).contiguous()

            k_fp8 = forward_batch.token_to_kv_pool.get_index_k_continuous(
                layer_id,
                seq_len,
                block_tables[i],
            )
            k_scale = forward_batch.token_to_kv_pool.get_index_k_scale_continuous(
                layer_id,
                seq_len,
                block_tables[i],
            )

            k_fp8 = k_fp8.view(torch.float8_e4m3fn).unsqueeze(0).contiguous()
            k_scale = k_scale.view(torch.float32).squeeze(-1).unsqueeze(0).contiguous()

            index_score = fp8_index(
                q_fp8_partial,
                weights_partial,
                k_fp8,
                k_scale,
            )
            end_pos = seq_len
            topk_indices = index_score.topk(min(topk, end_pos), dim=-1)[1].squeeze(0)

            pad_len = ceil_align(topk_indices.shape[-1], 2048) - topk_indices.shape[-1]
            topk_indices = torch.nn.functional.pad(
                topk_indices, (0, pad_len), "constant", -1
            )

            topk_indices_list.append(topk_indices)

            q_len_start = q_len_end

        topk_indices = torch.cat(topk_indices_list, dim=0)
        return topk_indices

    def _store_index_k_cache(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        key: torch.Tensor,
        *,
        act_quant=None,  # fallback only
    ) -> None:
        """
        Store NSA indexer K cache for current step.

        Preferred: fused_store_index_k_cache(key, cache, out_cache_loc, page_size)
        Fallback : act_quant(key) + token_to_kv_pool.set_index_k_scale_buffer(...)
        """

        # Fast path: JIT fused store (CUDA, page_size=64, non-fnuz)
        if (
            _is_cuda
            and (not _is_fp8_fnuz)
            and can_use_nsa_fused_store(
                key.dtype,
                forward_batch.out_cache_loc.dtype,
                forward_batch.token_to_kv_pool.page_size,
            )
        ):
            # NOTE: wrapper already normalizes shape/contiguity and asserts dtypes.
            buf = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
                layer_id=layer_id
            )
            fused_store_index_k_cache(
                key,
                buf,
                forward_batch.out_cache_loc,
                forward_batch.token_to_kv_pool.page_size,
            )
            if (
                _per_head_paged.per_head_index_enabled()
                or _offline_router.candidate_head_ids(layer_id) is not None
                or (
                    _headmap_probe.enabled()
                    and _headmap_probe.wants_layer(layer_id)
                )
            ):
                forward_batch.token_to_kv_pool.update_index_k_chunk_sum(
                    layer_id, key, forward_batch.out_cache_loc
                )
            return

        # Fast path: AITER fused quant + cache store (HIP, page_size=1)
        if _use_aiter:
            buf = forward_batch.token_to_kv_pool.get_index_k_with_scale_buffer(
                layer_id=layer_id
            )
            # Reshape from (num_pages, 132) uint8 to (num_pages, 1, 132) fp8
            # to match kernel's (num_blocks, block_size, head_dim + scale_bytes) layout
            kv_cache = buf.unsqueeze(1).view(fp8_dtype)
            out_loc = forward_batch.out_cache_loc
            if not out_loc.is_contiguous():
                out_loc = out_loc.contiguous()
            indexer_k_quant_and_cache(
                key, kv_cache, out_loc, self.block_size, self.scale_fmt
            )
            return

        # Fallback: original path
        assert act_quant is not None
        k_fp8, k_scale = act_quant(key, self.block_size, self.scale_fmt)

        out_loc = forward_batch.out_cache_loc
        if not out_loc.is_contiguous():
            out_loc = out_loc.contiguous()

        forward_batch.token_to_kv_pool.set_index_k_scale_buffer(
            layer_id=layer_id,
            loc=out_loc,
            index_k=k_fp8,
            index_k_scale=k_scale,
        )
        if (
            _per_head_paged.per_head_index_enabled()
            or _offline_router.candidate_head_ids(layer_id) is not None
            or (
                _headmap_probe.enabled()
                and _headmap_probe.wants_layer(layer_id)
            )
        ):
            forward_batch.token_to_kv_pool.update_index_k_chunk_sum(
                layer_id, key, out_loc
            )

    def forward_xpu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        return self.forward_cuda(
            x, q_lora, positions, forward_batch, layer_id, return_indices
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        return_indices: bool = True,
    ) -> Optional[torch.Tensor]:
        if _is_hip:
            from sglang.srt.layers.attention.nsa.tilelang_kernel import act_quant
        elif not _is_npu:
            from sglang.srt.layers.attention.nsa.triton_kernel import act_quant

        if TYPE_CHECKING:
            assert isinstance(forward_batch.token_to_kv_pool, NSATokenToKVPool)

        # When upstream uses fused FP8 RMSNorm+quant, activations may be passed as
        # a tuple like (x_fp8, x_scale[, y]). Use `x_meta` for shape/device queries.
        x_meta = x[0] if isinstance(x, tuple) else x

        metadata = forward_batch.attn_backend.get_indexer_metadata(layer_id, forward_batch)

        enable_dual_stream = (
            self.alt_stream is not None
            and get_is_capture_mode()
            and q_lora.shape[0] > 0
            and q_lora.shape[0] <= DUAL_STREAM_TOKEN_THRESHOLD
        )

        # skip NSA if attention backend choose to skip this batch
        if metadata is None:
            return None

        # Determine if should skip topk based on sequence length
        # We can only skip the logits computation if cuda graph is not involved
        skip_logits_computation = False
        if forward_batch.forward_mode.is_extend_without_speculative():
            if forward_batch.seq_lens_cpu is not None:
                max_kv_len = forward_batch.seq_lens_cpu.max().item()
                skip_logits_computation = max_kv_len <= self.index_topk

        # Optimization: fast path when skipping topk computation
        if skip_logits_computation and (not self.nsa_enable_prefill_cp):
            return maybe_capture_indexer_topk(
                layer_id,
                self._forward_cuda_k_only(
                    x,
                    positions,
                    forward_batch,
                    layer_id,
                    act_quant,
                    enable_dual_stream,
                    metadata,
                    return_indices,
                ),
            )

        if enable_dual_stream and forward_batch.forward_mode.is_decode_or_idle():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            raw_head_gates = self._project_raw_head_gates(x)
            head_gates = raw_head_gates * self.n_heads**-0.5
            query, key = self._get_q_k_bf16(
                q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
            )
            q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
            with torch.cuda.stream(self.alt_stream):
                self._store_index_k_cache(
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    key=key,
                    act_quant=act_quant,
                )
            current_stream.wait_stream(self.alt_stream)
            weights = head_gates.unsqueeze(-1) * q_scale * self.softmax_scale
        else:
            query, key = self._get_q_k_bf16(
                q_lora, x, positions, enable_dual_stream, forward_batch=forward_batch
            )

            if enable_dual_stream:
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)

                q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
                with torch.cuda.stream(self.alt_stream):
                    self._store_index_k_cache(
                        forward_batch=forward_batch,
                        layer_id=layer_id,
                        key=key,
                        act_quant=act_quant,
                    )
                current_stream.wait_stream(self.alt_stream)
            else:
                q_fp8, q_scale = act_quant(query, self.block_size, self.scale_fmt)
                self._store_index_k_cache(
                    forward_batch=forward_batch,
                    layer_id=layer_id,
                    key=key,
                    act_quant=act_quant,
                )

            # aiter (ROCm gfx95): the 3-tuple (fp8, scale, bf16) from
            # fused_rms_fp8_group_quant is passed directly to the gate projection,
            # which extracts the bf16 tensor via _weights_proj_bf16_in_fp32_out,
            # completely skipping the FP8 dequantization path below.
            if (
                _use_aiter
                and _is_gfx95_supported
                and isinstance(x, tuple)
                and len(x) == 3
            ):
                x_for_gate = x
            elif isinstance(x, tuple):
                assert len(x) in (
                    2,
                    3,
                ), "For tuple input, only (x, x_s) or (x, x_s, y) formats are accepted"
                x_q, x_s = x[0], x[1]
                if (
                    x_s is not None
                    and x_q.dim() == 2
                    and x_s.dim() == 2
                    and x_q.shape[0] == x_s.shape[0]
                ):
                    m, n = x_q.shape
                    ng = x_s.shape[1]
                    if ng > 0 and n % ng == 0:
                        group = n // ng
                        x_for_gate = (
                            x_q.to(torch.float32)
                            .view(m, ng, group)
                            .mul_(x_s.to(torch.float32).unsqueeze(-1))
                            .view(m, n)
                            .to(torch.bfloat16)
                        )
                    else:
                        x_for_gate = x_q.to(torch.bfloat16)
                else:
                    x_for_gate = x_q.to(torch.bfloat16)
            else:
                x_for_gate = x

            raw_head_gates, weights = self._get_raw_and_logits_head_gate(
                x_for_gate, q_scale
            )

        if _indexer_dump.enabled():
            _indexer_dump.maybe_dump(
                forward_batch=forward_batch,
                layer_id=layer_id,
                positions=positions,
                key_bf16=key,
                q_fp8=q_fp8,
                q_scale=q_scale,
                weights=weights,
                act_quant=act_quant,
                block_size=self.block_size,
                scale_fmt=self.scale_fmt,
            )

        if _is_cuda or _is_hip:
            assert forward_batch.seq_lens_cpu is not None
            if len(forward_batch.seq_lens_cpu) == 0:
                # this seems b/c max-pad, no worries?
                # if x.shape[0] != 0:
                #     print(
                #         "HACK: seq_lens empty but x not empty, hackily return all-invalid topk_result"
                #     )
                return maybe_capture_indexer_topk(
                    layer_id,
                    torch.full(
                        (x_meta.shape[0], self.index_topk),
                        -1,
                        dtype=torch.int,
                        device=x_meta.device,
                    ),
                )

            if (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
                or forward_batch.forward_mode.is_draft_extend(include_v2=True)
            ):
                topk_result = self._get_topk_paged(
                    forward_batch,
                    layer_id,
                    q_fp8,
                    weights,
                    metadata,
                    query,
                    raw_head_gates,
                )
            else:
                if (
                    forward_batch.attn_cp_metadata is not None
                    and is_nsa_prefill_cp_in_seq_split()
                ):
                    kv_len_prev = forward_batch.attn_cp_metadata.kv_len_prev
                    kv_len_next = forward_batch.attn_cp_metadata.kv_len_next
                    actual_seq_q_prev = forward_batch.attn_cp_metadata.actual_seq_q_prev
                    actual_seq_q_next = forward_batch.attn_cp_metadata.actual_seq_q_next

                    # TODO support mutil-batch
                    # cp_batch_seq_index_prev = forward_batch.attn_cp_metadata["cp_batch_seq_index_prev"]
                    # cp_batch_seq_index_next = forward_batch.attn_cp_metadata["cp_batch_seq_index_next"]
                    # TODO prev, next, combined into a single call
                    q_fp8_prev, q_fp8_next = torch.split(
                        q_fp8, (q_fp8.shape[0] + 1) // 2, dim=0
                    )
                    weights_prev, weights_next = torch.split(
                        weights, (weights.shape[0] + 1) // 2, dim=0
                    )
                    topk_result_prev = self._get_topk_ragged_with_cp(
                        forward_batch,
                        layer_id,
                        q_fp8_prev,
                        weights_prev,
                        metadata,
                        kv_len_prev,
                        actual_seq_q_prev,
                    )

                    topk_result_next = self._get_topk_ragged_with_cp(
                        forward_batch,
                        layer_id,
                        q_fp8_next,
                        weights_next,
                        metadata,
                        kv_len_next,
                        actual_seq_q_next,
                    )
                    return maybe_capture_indexer_topk(
                        layer_id,
                        torch.cat([topk_result_prev, topk_result_next], dim=0),
                    )
                else:
                    topk_result = self._get_topk_ragged(
                        enable_dual_stream,
                        forward_batch,
                        layer_id,
                        q_fp8,
                        weights,
                        metadata,
                    )
        else:
            topk_result = self.forward_indexer(
                q_fp8.contiguous(),
                weights,
                forward_batch,
                topk=self.index_topk,
                layer_id=layer_id,
            )
        return maybe_capture_indexer_topk(layer_id, topk_result)






    def forward_npu(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        layer_scatter_modes=None,
        dynamic_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        if forward_batch.attn_backend.forward_metadata.seq_lens_cpu_int is None:
            actual_seq_lengths_kv = forward_batch.attn_backend.forward_metadata.seq_lens
        else:
            actual_seq_lengths_kv = (
                forward_batch.attn_backend.forward_metadata.seq_lens_cpu_int
            )
        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
            and not forward_batch.forward_mode.is_draft_extend()
        )

        bs = q_lora.shape[0]

        if self.rotary_emb.is_neox_style:
            if not hasattr(forward_batch, "npu_indexer_sin_cos_cache"):
                cos_sin = self.rotary_emb.cos_sin_cache[positions]
                cos, sin = cos_sin.chunk(2, dim=-1)
                cos = cos.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                sin = sin.repeat(1, 2).view(-1, 1, 1, self.rope_head_dim)
                forward_batch.npu_indexer_sin_cos_cache = (sin, cos)
            else:
                sin, cos = forward_batch.npu_indexer_sin_cos_cache

            if self.alt_stream is not None:
                self.alt_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(self.alt_stream):
                    q_lora = (
                        (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                    )
                    q = self.wq_b(q_lora)[
                        0
                    ]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
                    wq_b_event = self.alt_stream.record_event()
                    q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
                    q_pe, q_nope = torch.split(
                        q,
                        [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                        dim=-1,
                    )  # [bs, 64, 64 + 64]
                    q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
                    q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(
                        bs, self.n_heads, self.rope_head_dim
                    )  # [bs, n, d]
                    q = torch.cat([q_pe, q_nope], dim=-1)
                    q.record_stream(self.alt_stream)
                    q_rope_event = self.alt_stream.record_event()
            else:
                q_lora = (
                    (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
                )
                q = self.wq_b(q_lora)[
                    0
                ]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
                q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
                q_pe, q_nope = torch.split(
                    q,
                    [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                    dim=-1,
                )  # [bs, 64, 64 + 64]
                q_pe = q_pe.view(bs, self.n_heads, 1, self.rope_head_dim)
                q_pe = torch_npu.npu_rotary_mul(q_pe, cos, sin).view(
                    bs, self.n_heads, self.rope_head_dim
                )  # [bs, n, d]
                q = torch.cat([q_pe, q_nope], dim=-1)

            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0].to(torch.bfloat16)
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0].to(torch.bfloat16)

            k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
            k = self.k_norm(k_proj)
            if (
                _use_ag_after_qlora
                and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
                and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
            ):
                k = scattered_to_tp_attn_full(k, forward_batch)
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64 + 64]

            k_pe = k_pe.view(-1, 1, 1, self.rope_head_dim)
            k_pe = torch.ops.npu.npu_rotary_mul(k_pe, cos, sin).view(
                bs, 1, self.rope_head_dim
            )  # [bs, 1, d]
            k = torch.cat([k_pe, k_nope.unsqueeze(1)], dim=-1)  # [bs, 1, 128]

        else:
            if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
                indexer_weight_stream = get_indexer_weight_stream()
                indexer_weight_stream.wait_stream(torch.npu.current_stream())
                with torch.npu.stream(indexer_weight_stream):
                    x = x.view(-1, self.hidden_size)
                    weights = self.weights_proj(x.float())[0].to(torch.bfloat16)
                    weights.record_stream(indexer_weight_stream)
                    weights_event = indexer_weight_stream.record_event()
            else:
                x = x.view(-1, self.hidden_size)
                weights = self.weights_proj(x.float())[0].to(torch.bfloat16)

            q_lora = (q_lora, dynamic_scale) if dynamic_scale is not None else q_lora
            q = self.wq_b(q_lora)[0]  # [bs, 1536] @ [1536, 64 * 128] = [bs, 64 * 128]
            q = q.view(bs, self.n_heads, self.head_dim)  # [bs, 64, 128]
            q_pe, q_nope = torch.split(
                q,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64, 64 + 64]

            k_proj = self.wk(x)[0]  # [b, s, 7168] @ [7168, 128] = [b, s, 128]
            k = self.k_norm(k_proj)
            k_pe, k_nope = torch.split(
                k,
                [self.rope_head_dim, self.head_dim - self.rope_head_dim],
                dim=-1,
            )  # [bs, 64 + 64]

            k_pe = k_pe.unsqueeze(1)

            if layer_id == 0:
                self.rotary_emb.sin_cos_cache = (
                    self.rotary_emb.cos_sin_cache.index_select(0, positions)
                )

            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            k_pe = k_pe.squeeze(1)
            q = torch.cat([q_pe, q_nope], dim=-1)
            k = torch.cat([k_pe, k_nope], dim=-1)

        if (
            is_prefill
            and self.nsa_enable_prefill_cp
            and forward_batch.attn_cp_metadata is not None
        ):
            k = cp_all_gather_rerange_output(
                k.contiguous().view(-1, self.head_dim),
                self.cp_size,
                forward_batch,
                torch.npu.current_stream(),
            )

        forward_batch.token_to_kv_pool.set_index_k_buffer(
            layer_id, forward_batch.out_cache_loc, k
        )
        if is_prefill:
            if (
                self.nsa_enable_prefill_cp
                and forward_batch.attn_cp_metadata is not None
            ):
                forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q = (
                    forward_batch.attn_cp_metadata.actual_seq_q_prev_tensor,
                    forward_batch.attn_cp_metadata.actual_seq_q_next_tensor,
                )
                if sum(forward_batch.extend_prefix_lens_cpu) > 0:
                    total_kv_len_prev_tensor = (
                        forward_batch.attn_cp_metadata.kv_len_prev_tensor
                        + forward_batch.extend_prefix_lens.squeeze()
                    )
                    total_kv_len_next_tensor = (
                        forward_batch.attn_cp_metadata.kv_len_next_tensor
                        + forward_batch.extend_prefix_lens.squeeze()
                    )
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_kv = (
                        total_kv_len_prev_tensor,
                        total_kv_len_next_tensor,
                    )
                else:
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_kv = (
                        forward_batch.attn_cp_metadata.kv_len_prev_tensor,
                        forward_batch.attn_cp_metadata.kv_len_next_tensor,
                    )
                actual_seq_lengths_q = (
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q
                )
                actual_seq_lengths_kv = (
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_kv
                )
            else:
                actual_seq_lengths_kv = forward_batch.seq_lens
                actual_seq_lengths_q = forward_batch.extend_seq_lens.cumsum(dim=0)
        else:
            if forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q is None:
                if (
                    forward_batch.forward_mode.is_draft_extend_v2()
                    or forward_batch.forward_mode.is_target_verify()
                    or forward_batch.forward_mode.is_draft_extend()
                ):
                    num_draft_tokens = (
                        forward_batch.attn_backend.speculative_num_draft_tokens
                    )
                    actual_seq_lengths_q = torch.arange(
                        num_draft_tokens,
                        num_draft_tokens + bs,
                        num_draft_tokens,
                        dtype=torch.int32,
                        device=k.device,
                    )
                else:
                    actual_seq_lengths_q = torch.tensor(
                        [1 + i * 1 for i in range(bs)],
                        dtype=torch.int32,
                        device=k.device,
                    )
            else:
                actual_seq_lengths_q = (
                    forward_batch.attn_backend.forward_metadata.actual_seq_lengths_q
                )

        past_key_states = forward_batch.token_to_kv_pool.get_index_k_buffer(layer_id)

        if self.rotary_emb.is_neox_style and self.alt_stream is not None:
            torch.npu.current_stream().wait_event(q_rope_event)
        if envs.SGLANG_NPU_USE_MULTI_STREAM.get():
            torch.npu.current_stream().wait_event(weights_event)
        if (
            _use_ag_after_qlora
            and layer_scatter_modes.layer_input_mode == ScatterMode.SCATTERED
            and layer_scatter_modes.attn_mode == ScatterMode.TP_ATTN_FULL
        ):
            weights = scattered_to_tp_attn_full(weights, forward_batch)
        block_table = forward_batch.attn_backend.forward_metadata.block_tables
        if (
            is_prefill
            and self.nsa_enable_prefill_cp
            and forward_batch.attn_cp_metadata is not None
        ):
            block_table = block_table[: actual_seq_lengths_q[0].numel()]
            topk_indices = self.do_npu_cp_balance_indexer(
                q.view(-1, self.n_heads, self.head_dim),
                past_key_states,
                weights,
                actual_seq_lengths_q,
                actual_seq_lengths_kv,
                block_table,
            )
            return topk_indices
        else:
            block_table = (
                block_table[: actual_seq_lengths_q.size()[0]]
                if is_prefill
                else block_table
            )

            topk_indices = torch_npu.npu_lightning_indexer(
                query=q.view(-1, self.n_heads, self.head_dim),
                key=past_key_states,
                weights=weights,
                actual_seq_lengths_query=actual_seq_lengths_q.to(torch.int32),
                actual_seq_lengths_key=actual_seq_lengths_kv.to(k.device).to(
                    torch.int32
                ),
                block_table=block_table,
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=3,
            )
            return topk_indices[0]

    def do_npu_cp_balance_indexer(
        self,
        q,
        past_key_states,
        indexer_weights,
        actual_seq_lengths_q,
        actual_seq_lengths_kv,
        block_table,
    ):
        q_prev, q_next = torch.split(q, (q.size(0) + 1) // 2, dim=0)
        weights_prev, weights_next = None, None
        if indexer_weights is not None:
            weights_prev, weights_next = torch.split(
                indexer_weights, (indexer_weights.size(0) + 1) // 2, dim=0
            )
            weights_prev = weights_prev.contiguous().view(-1, weights_prev.shape[-1])
            weights_next = weights_next.contiguous().view(-1, weights_next.shape[-1])

        actual_seq_lengths_q_prev, actual_seq_lengths_q_next = actual_seq_lengths_q
        actual_seq_lengths_kv_prev, actual_seq_lengths_kv_next = actual_seq_lengths_kv

        topk_indices_prev = torch_npu.npu_lightning_indexer(
            query=q_prev,
            key=past_key_states,
            weights=weights_prev,
            actual_seq_lengths_query=actual_seq_lengths_q_prev.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv_prev.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )
        topk_indices_next = torch_npu.npu_lightning_indexer(
            query=q_next,
            key=past_key_states,
            weights=weights_next,
            actual_seq_lengths_query=actual_seq_lengths_q_next.to(
                device=q.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=actual_seq_lengths_kv_next.to(
                device=q.device, dtype=torch.int32
            ),
            block_table=block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=3,
        )
        return topk_indices_prev[0], topk_indices_next[0]


def scattered_to_tp_attn_full(
    hidden_states: torch.Tensor,
    forward_batch,
) -> torch.Tensor:
    hidden_states, local_hidden_states = (
        torch.empty(
            (forward_batch.input_ids.shape[0], hidden_states.shape[1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        ),
        hidden_states,
    )
    attn_tp_all_gather_into_tensor(hidden_states, local_hidden_states.contiguous())
    return hidden_states
