import sys
sys.path.insert(0, "/workspace/qyl/code/adaptive_0921_h202/python")
import torch
from sglang.srt.layers.attention.nsa.hisa.triton_kernels import _block_sparse_mqa_persistent_kernel, sparse_paged_mqa_triton

dev = "cuda"
torch.manual_seed(0)
N = 131072; n_complete = N - 8192; H = 64; D = 128; budget = 8192; step = 2048
k = torch.randn(N, D, device=dev).to(torch.float8_e4m3fn)
s = torch.rand(N, device=dev) + 0.5
q = torch.randn(step, H, D, device=dev).to(torch.float8_e4m3fn)
w = torch.rand(step, H, device=dev)
pos1 = torch.arange(n_complete + 1, n_complete + step + 1, device=dev, dtype=torch.int32)
ks0 = torch.zeros(step, dtype=torch.int32, device=dev)
# candidates: ~130 random contiguous runs of avg 64 tokens per row (sorted by position)
cand = torch.empty(step, budget, dtype=torch.int32, device=dev)
for r in range(step):
    starts = torch.randint(0, n_complete - 256, (128,), device=dev).sort().values
    toks = (starts[:, None] + torch.arange(64, device=dev)[None, :]).reshape(-1)
    cand[r] = toks
pages = N // 64
buf = torch.zeros((pages, 64 * 132), dtype=torch.uint8, device=dev)
buf[:, : 64 * 128] = k.view(torch.uint8).view(pages, -1)
buf[:, 64 * 128:] = s.view(pages, 64).view(torch.uint8).view(pages, 256)
raw_pages = buf.view(pages, 64, 1, 132)
table = torch.arange(pages, device=dev, dtype=torch.int32)[None].expand(step, -1).contiguous()


def timeit(fn, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


t0 = timeit(lambda: sparse_paged_mqa_triton(q.unsqueeze(1), raw_pages, cand.unsqueeze(1), 1, w.unsqueeze(1), pos1, table))
print(f"current K=1 paged: {t0:.3f} ms per 2048 rows  ({4 * t0:.2f} ms per 8192-row chunk)")


def flat(G, KC, warps, stages):
    logits = torch.empty((step, budget), device=dev, dtype=torch.float32)
    num_chunks = (budget + G - 1) // G
    outer = (num_chunks + KC - 1) // KC
    _block_sparse_mqa_persistent_kernel[(step, outer)](
        q, k, s, cand, logits, w, ks0, pos1,
        q.stride(0), q.stride(1), k.stride(0), cand.stride(0), logits.stride(0), w.stride(0),
        N, budget, HEADS=H, DIM=D, KV_BLOCK_SIZE=1, GROUP_SIZE=G, K_CHUNKS=KC,
        num_warps=warps, num_stages=stages)
    return logits


ref = sparse_paged_mqa_triton(q.unsqueeze(1), raw_pages, cand.unsqueeze(1), 1, w.unsqueeze(1), pos1, table).squeeze(1)
assert torch.equal(ref, flat(64, 128, 4, 2))
best = None
for G in (32, 64, 128, 256):
    for KC in (16, 32, 64, 128, 256):
        if G * KC > budget or G * KC < 2048:
            continue
        for warps in (4, 8):
            for stages in (2, 3, 4):
                try:
                    t = timeit(lambda: flat(G, KC, warps, stages))
                except Exception as e:
                    print(f"G={G} KC={KC} warps={warps} stages={stages}: FAILED {str(e)[:60]}"); continue
                print(f"G={G:3d} KC={KC:3d} warps={warps} stages={stages}: {t:.3f} ms")
                if best is None or t < best[0]:
                    best = (t, G, KC, warps, stages)
print("best:", best, f"-> {4 * best[0]:.2f} ms per chunk vs {4 * t0:.2f}")
