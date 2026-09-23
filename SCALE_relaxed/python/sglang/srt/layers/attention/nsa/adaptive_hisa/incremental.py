"""Fixed-chunk sealing of generated tokens behind a frozen adaptive prefix.

Aligned with the h20-1 accuracy reference: tokens generated after the prefill
partition (``base_complete``) are sealed into fixed ``chunk``-token leaves
(default 64) whose FP8 summary is the requantised mean of the dequantised keys,
exactly like the prefill leaf summaries (``summary_kernels.leaf_key_means`` +
``requant_summaries``). The FP32 mean / FP8 requantisation and the 64-row SoA
writeback follow hisa.tilelang_kernels.fp8_native_paged_mean_pooling_completed_blocks
(origin/hisa_pr e399d9f33); only the addressing differs: input chunks start
at ``base_complete`` (any multiple of ``chunk`` inside a 64-token page) and
output rows start at ``base_rows``. Four CTAs cover newly sealed chunks,
including catch-up after an eager/batched interval.

Leaves whose tokens fall inside the raw guards (first ``sink`` tokens, most
recent ``tail`` tokens) are clipped by the selector, so sealing may run ahead
of the tail without changing the candidate set.
"""
import tilelang
import torch
import triton
import triton.language as tl
from tilelang import language as T

PAGE = 64


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def _sealed_chunks(chunk: int = 64, dim: int = 128, threads: int = 128):
    pages = T.dynamic("pages")
    columns = T.dynamic("columns")
    summary_pages = T.dynamic("summary_pages")
    capacity = T.dynamic("capacity")
    scale_base = PAGE * dim // 4

    @T.prim_func
    def kernel(
        K: T.Tensor([pages, PAGE * (dim + 4)], T.float8_e4m3fn),
        KS: T.Tensor([pages, PAGE * (dim + 4) // 4], T.float32),
        RawPages: T.Tensor([1, columns], T.int32),
        Seq: T.Tensor([1], T.int32),
        BaseComplete: T.Tensor([1], T.int32),
        BaseRows: T.Tensor([1], T.int32),
        PreviousBlocks: T.Tensor([1], T.int32),
        Summary: T.Tensor([summary_pages, PAGE * (dim + 4)], T.float8_e4m3fn),
        SummaryScale: T.Tensor([summary_pages, PAGE * (dim + 4) // 4], T.float32),
        Starts: T.Tensor([capacity], T.int32),
        Lengths: T.Tensor([capacity], T.int32),
    ):
        with T.Kernel(4, threads=threads) as by:
            matrix = T.alloc_fragment([chunk, dim], T.float32)
            acc = T.alloc_fragment([dim], T.float32)
            max_abs = T.alloc_fragment([1], T.float32)
            base = BaseComplete[0]
            previous = PreviousBlocks[0]
            completed = T.max(Seq[0] - base, 0) // chunk
            n_new = T.max(completed - previous, 0)
            for step in T.serial(T.ceildiv(n_new, 4)):
                relative = by + step * 4
                block = previous + relative
                row = BaseRows[0] + block
                if relative < n_new and row < capacity:
                    start = base + block * chunk
                    # Keep speculative address calculation in bounds, as in
                    # the original HISA completed-block kernel.
                    safe_page = T.min(start // PAGE, columns - 1)
                    physical = RawPages[0, safe_page]
                    first_slot = start % PAGE
                    for n, d in T.Parallel(chunk, dim):
                        matrix[n, d] = (
                            T.cast(K[physical, (first_slot + n) * dim + d], T.float32)
                            * KS[physical, scale_base + first_slot + n]
                        )
                    T.reduce_sum(matrix, acc, dim=0, clear=True)
                    for d in T.Parallel(dim):
                        acc[d] = acc[d] * (1.0 / chunk)
                    T.reduce_absmax(acc, max_abs, dim=0, clear=True)
                    scale = T.max(max_abs[0] * (1.0 / 448.0), T.cast(1e-10, T.float32))
                    inv_scale = T.cast(1.0, T.float32) / scale
                    page = row // PAGE
                    slot = row % PAGE
                    for d in T.Parallel(dim):
                        Summary[page, slot * dim + d] = T.cast(acc[d] * inv_scale, T.float8_e4m3fn)
                    SummaryScale[page, scale_base + slot] = scale
                    Starts[row] = start
                    Lengths[row] = chunk
    return kernel


@triton.jit
def _commit(Seq, BaseComplete, BaseRows, PreviousBlocks, Complete, NumLeaves,
            CHUNK: tl.constexpr):
    base = tl.load(BaseComplete)
    blocks = tl.maximum(tl.load(Seq) - base, 0) // CHUNK
    tl.store(PreviousBlocks, blocks)
    tl.store(Complete, base + blocks * CHUNK)
    tl.store(NumLeaves, tl.load(BaseRows) + blocks)


@triton.jit
def _clean_scores(Scores, Lengths, NumLeaves, CAPACITY: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(NumLeaves)
    valid = (row < CAPACITY) & (row < count)
    length = tl.load(Lengths + row, row < CAPACITY, 0)
    score = tl.load(Scores + row, row < CAPACITY, float('-inf'))
    score = tl.where(valid & (length > 0), score, float('-inf'))
    tl.store(Scores + row, score, row < CAPACITY)


def update_completed_blocks(work, raw_pages, raw_page_table, seq_len, chunk=64):
    """Seal every complete ``chunk`` of generated tokens since the last call."""
    seq = seq_len.reshape(-1)
    _sealed_chunks(chunk)(
        raw_pages.view(torch.float8_e4m3fn), raw_pages.view(torch.float32),
        raw_page_table, seq, work.base_complete, work.base_rows,
        work.previous_blocks, work.summary_pages.view(torch.float8_e4m3fn),
        work.summary_pages.view(torch.float32), work.starts, work.lengths,
    )
    _commit[(1,)](seq, work.base_complete, work.base_rows, work.previous_blocks,
                  work.complete, work.num_leaves, CHUNK=chunk)


def clean_summary_scores(scores, work, seq_len=None):
    """Mask padding / unused rows. Guards are raw tokens, so no leaf is forced."""
    _clean_scores[(triton.cdiv(work.capacity, 256),)](
        scores, work.lengths, work.num_leaves,
        CAPACITY=work.capacity, BLOCK=256,
    )
    return scores
