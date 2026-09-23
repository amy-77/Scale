"""Fixed-shape candidate expansion and raw paged FP8 gathers for B=1 decode."""
import torch
import triton
import triton.language as tl


@triton.jit
def _expand(order, prefix, starts, complete, seq_len, output, count,
            M: tl.constexpr, BUDGET: tl.constexpr, BLOCK: tl.constexpr, STEPS: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    sealed = tl.load(complete)
    tail = tl.minimum(tl.maximum(tl.load(seq_len) - sealed, 0), BUDGET)
    leaf_tokens = tl.load(prefix + M - 1)
    total = tl.minimum(leaf_tokens + tail, BUDGET)
    # Upper-bound search: first selected prefix sum strictly greater than pos.
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), M, tl.int32)
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        value = tl.load(prefix + mid, mask=mid < M, other=2147483647)
        right = value <= pos
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(right, hi, mid)
    leaf = tl.load(order + lo, mask=lo < M, other=0)
    start = tl.load(starts + leaf)
    previous = tl.load(prefix + lo - 1, mask=(lo > 0) & (lo <= M), other=0)
    token = tl.where(pos < leaf_tokens, start + pos - previous,
                     sealed + pos - leaf_tokens)
    tl.store(output + pos, tl.where(pos < total, token, -1), mask=pos < BUDGET)
    if tl.program_id(0) == 0:
        tl.store(count + pos, total, mask=pos == 0)


def expand_candidates(order, prefix, starts, complete, seq_len, budget):
    candidates = torch.empty((1, budget), dtype=torch.int32, device=starts.device)
    count = torch.empty((1,), dtype=torch.int32, device=starts.device)
    _expand[(triton.cdiv(budget, 256),)](
        order, prefix, starts, complete, seq_len, candidates, count,
        M=starts.numel(), BUDGET=budget, BLOCK=256, STEPS=starts.numel().bit_length())
    return candidates, count


@triton.jit
def _expand_ranked_budget(
    order,
    ranked_prefix,
    starts,
    complete,
    seq_len,
    output,
    count,
    M: tl.constexpr,
    BUDGET: tl.constexpr,
    BLOCK: tl.constexpr,
    STEPS: tl.constexpr,
):
    """Expand the score-ranked prefix that fits the token budget.

    ``ranked_prefix`` is the inclusive cumsum of *all* ranked leaf lengths.
    The old path materialised ``where(prefix <= available, length, 0)`` and
    ran a second cumsum to turn the suffix into a plateau.  Because lengths
    are non-negative, the accepted leaves are already one prefix: find its
    endpoint here and use the original prefix for token-to-leaf lookup.
    """
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    sealed = tl.load(complete)
    tail = tl.minimum(tl.maximum(tl.load(seq_len) - sealed, 0), BUDGET)
    available = BUDGET - tail

    # upper_bound(ranked_prefix, available): number of complete leaves that
    # fit. Every lane intentionally performs the same scalar search; keeping
    # it in this launch is still cheaper than materialising another [M] tensor
    # and launching a second scan.
    end_lo = tl.zeros((BLOCK,), tl.int32)
    end_hi = tl.full((BLOCK,), M, tl.int32)
    for _ in range(STEPS):
        end_mid = (end_lo + end_hi) // 2
        end_value = tl.load(
            ranked_prefix + end_mid, mask=end_mid < M, other=2147483647
        )
        end_right = end_value <= available
        end_lo = tl.where(end_right, end_mid + 1, end_lo)
        end_hi = tl.where(end_right, end_hi, end_mid)
    selected_count = end_lo
    leaf_tokens = tl.load(
        ranked_prefix + selected_count - 1,
        mask=selected_count > 0,
        other=0,
    )
    total = tl.minimum(leaf_tokens + tail, BUDGET)

    # upper_bound(ranked_prefix, pos): rank of the leaf containing this token.
    lo = tl.zeros((BLOCK,), tl.int32)
    hi = selected_count
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        value = tl.load(
            ranked_prefix + mid,
            mask=(mid < selected_count) & (mid < M),
            other=2147483647,
        )
        right = value <= pos
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(right, hi, mid)
    leaf = tl.load(order + lo, mask=lo < selected_count, other=0)
    start = tl.load(starts + leaf)
    previous = tl.load(
        ranked_prefix + lo - 1,
        mask=(lo > 0) & (lo <= selected_count),
        other=0,
    )
    token = tl.where(
        pos < leaf_tokens, start + pos - previous, sealed + pos - leaf_tokens
    )
    tl.store(
        output + pos, tl.where(pos < total, token, -1), mask=pos < BUDGET
    )
    if tl.program_id(0) == 0:
        tl.store(count + pos, total, mask=pos == 0)


def expand_ranked_candidates(
    order, ranked_prefix, starts, complete, seq_len, budget
):
    """Budget and expand a fully ranked leaf list without a second cumsum."""
    candidates = torch.empty((1, budget), dtype=torch.int32, device=starts.device)
    count = torch.empty((1,), dtype=torch.int32, device=starts.device)
    _expand_ranked_budget[(triton.cdiv(budget, 256),)](
        order,
        ranked_prefix,
        starts,
        complete,
        seq_len,
        candidates,
        count,
        M=starts.numel(),
        BUDGET=budget,
        BLOCK=256,
        STEPS=starts.numel().bit_length(),
    )
    return candidates, count


@triton.jit
def _gather(raw_bytes, raw_scales, pages, candidates, seq_len, keys, scales,
            BUDGET: tl.constexpr, PAGE_COLS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, 128)
    token = tl.load(candidates + row, mask=row < BUDGET, other=-1)
    valid = (row < BUDGET) & (token >= 0) & (token < tl.load(seq_len))
    logical_page = token // 64
    valid = valid & (logical_page < PAGE_COLS)
    physical_page = tl.load(pages + logical_page, mask=valid, other=0).to(tl.int64)
    slot = token % 64
    bits = tl.load(raw_bytes + physical_page[:, None] * 8448 + slot[:, None] * 128 + d[None, :],
                   mask=valid[:, None], other=0)
    scale = tl.load(raw_scales + physical_page * 2112 + 2048 + slot, mask=valid, other=0.0)
    tl.store(keys + row[:, None] * 128 + d[None, :], bits, mask=(row < BUDGET)[:, None])
    tl.store(scales + row, scale, mask=row < BUDGET)


def gather_candidate_keys(raw_pages, page_table, candidates, seq_len):
    budget = candidates.shape[1]
    keys = torch.empty((budget, 128), dtype=torch.float8_e4m3fn, device=raw_pages.device)
    scales = torch.empty((budget,), dtype=torch.float32, device=raw_pages.device)
    _gather[(triton.cdiv(budget, 32),)](
        raw_pages, raw_pages.view(torch.float32), page_table, candidates, seq_len,
        keys.view(torch.uint8), scales, BUDGET=budget, PAGE_COLS=page_table.shape[1], BLOCK=32)
    return keys, scales
