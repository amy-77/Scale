"""Per-layer, per-chunk cost of the sparse prefill indexer stages at 128K (synthetic shapes)."""
import os
import sys
sys.path.insert(0, "/workspace/qyl/code/adaptive_0921_h202/python")
import torch, triton
import deep_gemm
from sglang.srt.layers.attention.nsa.hisa.triton_kernels import (
    sparse_paged_mqa_triton, block_sparse_mqa_triton, _block_sparse_mqa_persistent_kernel)
from sglang.srt.layers.attention.nsa.hisa.hisa_topk_fused import hisa_topk_candidates_fused
from sglang.srt.layers.attention.nsa.adaptive_hisa import prefill_select as ps

dev = "cuda"
torch.manual_seed(0)
N = 131072; n_complete = N - 8192; n_q = 8192; H = 64; D = 128; budget = 8192; step = 2048
k = torch.randn(N, D, device=dev).to(torch.float8_e4m3fn)
s = torch.rand(N, device=dev) + 0.5
q = torch.randn(n_q, H, D, device=dev).to(torch.float8_e4m3fn)
w = torch.rand(n_q, H, device=dev)
pos1 = torch.arange(n_complete + 1, N + 1, device=dev, dtype=torch.int32)
pages = N // 64
buf = torch.zeros((pages, 64 * 132), dtype=torch.uint8, device=dev)
buf[:, : 64 * 128] = k.view(torch.uint8).view(pages, -1)
buf[:, 64 * 128:] = s.view(pages, 64).view(torch.uint8).view(pages, 256)
raw_pages = buf.view(pages, 64, 1, 132)
table = torch.arange(pages, device=dev, dtype=torch.int32)[None]

# split L/32 -> merge L/128 production shape: random adaptive lengths averaging
# 128 tokens and exactly the final summary capacity.
n = n_complete // 128
wl = torch.rand(n, device=dev) * 1.5 + 0.25
lens = torch.floor(wl * ((n_complete - n) / wl.sum().item())).to(torch.int32) + 1
lens[: n_complete - int(lens.sum())] += 1
starts = torch.zeros(n, dtype=torch.int32, device=dev); starts[1:] = torch.cumsum(lens, 0)[:-1].to(torch.int32)
nl = torch.tensor([n], dtype=torch.int32, device=dev)
keys = torch.randn(n, D, device=dev).to(torch.float8_e4m3fn); scales = torch.ones(n, device=dev)


def timeit(fn, iters=10):
    for _ in range(2): fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


coarse_masked = ps.coarse_leaf_scores(q, w, keys, scales, nl, lens, starts, 64)
coarse = ps.coarse_leaf_scores(
    q, w, keys, scales, nl, lens, starts, 64, mask_metadata=False
)
t_coarse_masked = timeit(
    lambda: ps.coarse_leaf_scores(q, w, keys, scales, nl, lens, starts, 64)
)
t_coarse = timeit(
    lambda: ps.coarse_leaf_scores(
        q, w, keys, scales, nl, lens, starts, 64, mask_metadata=False
    )
)
cands_old = [
    ps.select_candidates_weighted(
        coarse_masked, starts, lens, nl, slice(r0, r0 + step), budget, n_complete, 0
    )
    for r0 in range(0, n_q, step)
]
cands = [
    ps.select_candidates_weighted(
        coarse, starts, lens, nl, slice(r0, r0 + step), budget, n_complete, 0, sink=64
    )
    for r0 in range(0, n_q, step)
]
assert all(torch.equal(a, b) for a, b in zip(cands_old, cands))
t_select = timeit(
    lambda: [
        ps.select_candidates_weighted(
            coarse, starts, lens, nl, slice(r0, r0 + step), budget, n_complete, 0, sink=64
        )
        for r0 in range(0, n_q, step)
    ]
)
print(
    f"coarse GEMM [8192 x {n}]: masked={t_coarse_masked:.3f} ms "
    f"pure={t_coarse:.3f} ms; weighted select + fused metadata: {t_select:.3f} ms"
)

# (a) current fine scorer: K=1 paged, per 2048-row sub-batch
def fine_paged():
    outs = []
    for i, r0 in enumerate(range(0, n_q, step)):
        rows = slice(r0, r0 + step)
        outs.append(sparse_paged_mqa_triton(q[rows].unsqueeze(1), raw_pages, cands[i].unsqueeze(1), 1,
                                            w[rows].unsqueeze(1), pos1[rows], table.expand(step, -1)).squeeze(1))
    return outs
t_a = timeit(fine_paged)
print(f"(a) fine K=1 paged (current): {t_a:.3f} ms")

# (b) HISA persistent kernel on flat K with K=1 (token ids as block ids)
ks0 = torch.zeros(n_q, dtype=torch.int32, device=dev)
def fine_flat_k1(GROUP_SIZE=256, K_CHUNKS=32, num_warps=8, num_stages=3):
    outs = []
    for i, r0 in enumerate(range(0, n_q, step)):
        rows = slice(r0, r0 + step)
        cand = cands[i]
        logits = torch.empty((step, budget), device=dev, dtype=torch.float32)
        num_chunks = (budget + GROUP_SIZE - 1) // GROUP_SIZE
        outer = (num_chunks + K_CHUNKS - 1) // K_CHUNKS
        _block_sparse_mqa_persistent_kernel[(step, outer)](
            q[rows], k, s, cand, logits, w[rows], ks0[rows], pos1[rows],
            q.stride(0), q.stride(1), k.stride(0), cand.stride(0), logits.stride(0), w.stride(0),
            N, budget, HEADS=H, DIM=D, KV_BLOCK_SIZE=1, GROUP_SIZE=GROUP_SIZE, K_CHUNKS=K_CHUNKS,
            num_warps=num_warps, num_stages=num_stages)
        outs.append(logits)
    return outs
for cfg in [(256, 32, 8, 3), (128, 64, 8, 3), (256, 32, 4, 3), (64, 128, 4, 2)]:
    t_b = timeit(lambda: fine_flat_k1(*cfg))
    print(f"(b) fine K=1 flat persistent G={cfg[0]} KC={cfg[1]} warps={cfg[2]} stages={cfg[3]}: {t_b:.3f} ms")
# correctness of (b) vs (a)
ref = fine_paged(); alt = fine_flat_k1()
for x, y in zip(ref, alt):
    m = torch.isfinite(x)
    assert torch.equal(m, torch.isfinite(y)), "mask mismatch"
    print("   max |diff| (b vs a):", (x[m] - y[m]).abs().max().item())
    break

# (c) HISA K=64 (128 aligned blocks per row) — contiguity ceiling
blocks = torch.stack([torch.randperm(n_complete // 64, device=dev)[:128].sort().values for _ in range(n_q)]).to(torch.int64)
def fine_k64():
    outs = []
    for r0 in range(0, n_q, step):
        rows = slice(r0, r0 + step)
        outs.append(block_sparse_mqa_triton(q[rows], k, s, blocks[rows], 64, w[rows], ks0[rows], pos1[rows]))
    return outs
t_c = timeit(fine_k64)
print(f"(c) fine HISA K=64 (128 blocks) flat persistent: {t_c:.3f} ms")

# local window dense GEMM + topk
k_loc, s_loc = k[n_complete:], s[n_complete:]
ke_loc = (pos1 - n_complete).contiguous()
def local():
    return [deep_gemm.fp8_mqa_logits(q[r0:r0 + step], (k_loc, s_loc), w[r0:r0 + step], ks0[r0:r0 + step], ke_loc[r0:r0 + step], clean_logits=False)
            for r0 in range(0, n_q, step)]
t_loc = timeit(local)
print(f"local window dense GEMM (8192 x 8192 causal): {t_loc:.3f} ms")
local_ids = torch.arange(n_complete, N, dtype=torch.int32, device=dev)
fine = fine_paged(); loc = local()
def topk():
    outs = []
    for i, r0 in enumerate(range(0, n_q, step)):
        R = step
        score = torch.cat((fine[i], loc[i][:, :8192]), dim=1)
        cand = torch.cat((cands[i], local_ids.unsqueeze(0).expand(R, -1)), dim=1)
        count = (budget + ke_loc[r0:r0 + R]).contiguous()
        outs.append(hisa_topk_candidates_fused(score, cand, count, pos1[r0:r0 + R], None))
    return outs
t_topk = timeit(topk)
print(f"cat + topk (16K candidates -> 2048): {t_topk:.3f} ms")
from sglang.srt.layers.attention.nsa.hisa.hisa_topk_fused import hisa_topk_candidates_split
def topk_split():
    outs = []
    for i, r0 in enumerate(range(0, n_q, step)):
        R = step
        count = (budget + ke_loc[r0:r0 + R]).contiguous()
        outs.append(hisa_topk_candidates_split(fine[i], loc[i][:, :8192], count, cands[i], n_complete, pos1[r0:r0 + R]))
    return outs
t_split = timeit(topk_split)
print(f"split topk, temporary output only: {t_split:.3f} ms")
topk_copy_out = torch.empty((n_q, 2048), dtype=torch.int32, device=dev)
def topk_split_copy():
    for i, r0 in enumerate(range(0, n_q, step)):
        rows = slice(r0, r0 + step)
        count = (budget + ke_loc[rows]).contiguous()
        topk_copy_out[rows] = hisa_topk_candidates_split(
            fine[i], loc[i][:, :8192], count, cands[i], n_complete, pos1[rows]
        )
    return topk_copy_out
t_split_copy = timeit(topk_split_copy)
print(f"split topk, old temporary + final copy: {t_split_copy:.3f} ms")
topk_out = torch.empty((n_q, 2048), dtype=torch.int32, device=dev)
def topk_split_direct():
    for i, r0 in enumerate(range(0, n_q, step)):
        rows = slice(r0, r0 + step)
        count = (budget + ke_loc[rows]).contiguous()
        hisa_topk_candidates_split(
            fine[i], loc[i][:, :8192], count, cands[i], n_complete, pos1[rows],
            out=topk_out[rows],
        )
    return topk_out
t_split_direct = timeit(topk_split_direct)
print(f"split topk, direct final output (no temp/copy): {t_split_direct:.3f} ms")
# The radix Top-K breaks threshold ties with atomics (two runs of the same
# path already differ in the odd tied token), so compare selected sets.
agree = [(x.sort(1).values == y.sort(1).values).float().mean().item() for x, y in zip(topk(), topk_split())]
print(f"split vs cat selected-set agreement: {min(agree):.6f}")
# dense DSA reference for this chunk
def dense():
    return deep_gemm.fp8_mqa_logits(q, (k, s), w, ks0, pos1, clean_logits=False)
if os.environ.get("BENCH_DENSE_DSA", "0") == "1":
    t_dense = timeit(dense, iters=3)
    print(f"dense DSA logits [8192 x 131072]: {t_dense:.3f} ms (no topk)")
else:
    print("dense DSA logits: skipped (set BENCH_DENSE_DSA=1; output needs ~4 GiB)")
