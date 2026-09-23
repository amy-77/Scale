#!/usr/bin/env python3
"""A/B the full 128K sparse-prefill indexer with dense vs block local keys."""
from __future__ import annotations

import os
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.nsa.adaptive_hisa import prefill_select as ps


torch.manual_seed(0)
dev = torch.device("cuda")
n_tokens = 131072
n_complete = n_tokens - 8192
n_q = 8192
heads = int(os.environ.get("BENCH_HEADS", "64"))
budget = 8192
rows = int(os.environ.get("BENCH_ROWS", "2048"))

k = torch.randn(n_tokens, 128, device=dev).to(torch.float8_e4m3fn)
s = torch.rand(n_tokens, device=dev) + 0.5
q = torch.randn(n_q, heads, 128, device=dev).to(torch.float8_e4m3fn)
w = torch.rand(n_q, heads, device=dev)
pos1 = torch.arange(n_complete + 1, n_tokens + 1, device=dev, dtype=torch.int32)

pages = n_tokens // 64
buf = torch.zeros((pages, 64 * 132), dtype=torch.uint8, device=dev)
buf[:, : 64 * 128] = k.view(torch.uint8).view(pages, -1)
buf[:, 64 * 128 :] = s.view(pages, 64).view(torch.uint8).view(pages, 256)
table = torch.arange(pages, device=dev, dtype=torch.int32)[None]

n_leaves = n_complete // 128
leaf_start = torch.arange(
    0, n_complete, 128, dtype=torch.int32, device=dev
)
leaf_len = torch.full(
    (n_leaves,), 128, dtype=torch.int32, device=dev
)
keys, scales, _, _ = ps.local_block_leaves(
    k, s, 0, n_complete, block=128
)
num_leaves = torch.tensor([n_leaves], dtype=torch.int32, device=dev)
cfg = SimpleNamespace(
    prefill_candidate_tokens=budget,
    index_topk=2048,
    sparse_prefill_rows=rows,
    sink_tokens=64,
)


def run(block_local: bool):
    return ps.sparse_topk_core(
        q,
        w,
        pos1,
        buf,
        table,
        keys,
        scales,
        leaf_start,
        leaf_len,
        num_leaves,
        n_complete,
        n_tokens,
        cfg,
        k_flat=(k, s),
        block_local=block_local,
    )


def bench(block_local: bool, reps: int = 5) -> float:
    for _ in range(2):
        out = run(block_local)
        del out
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(reps):
        out = run(block_local)
        del out
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) / reps


dense_ms = bench(False)
torch.cuda.empty_cache()
block_ms = bench(True)
print(
    f"dense_local_ms={dense_ms:.3f} block_local_ms={block_ms:.3f} "
    f"saved_ms={dense_ms - block_ms:.3f} speedup={dense_ms / block_ms:.3f}x"
)
