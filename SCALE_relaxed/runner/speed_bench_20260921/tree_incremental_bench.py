import sys, torch
sys.path.insert(0, "/workspace/qyl/code/adaptive_0921_h202/python")
from sglang.srt.layers.attention.nsa.adaptive_hisa import partition_kernels as pk
from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import TreeLayout

dev = "cuda"
torch.manual_seed(0)
N = 131072
k = torch.randn(N, 128, device=dev).to(torch.float8_e4m3fn)
s = torch.rand(N, device=dev) + 0.5
for atom, root in ((1, 256), (8, 256), (1, 64), (4, 128)):
    cached, n_old = None, 0
    step = 8192
    for n in range(step, N + 1, step):
        layout = TreeLayout(atom, root, n)
        full = pk.raw_fp8_tree_triton(k[:n], s[:n], layout)
        if cached is None:
            inc = full
        else:
            inc = pk.raw_fp8_tree_triton_incremental(k[:n], s[:n], layout, cached, n_old)
        assert inc.shape == full.shape and torch.equal(inc, full), (atom, root, n)
        cached, n_old = inc, n
    # odd extension (one root) and fallback cases
    layout = TreeLayout(atom, root, N)
    inc = pk.raw_fp8_tree_triton_incremental(k, s, layout, pk.raw_fp8_tree_triton(k[: N - root], s[: N - root], TreeLayout(atom, root, N - root)), N - root)
    assert torch.equal(inc, pk.raw_fp8_tree_triton(k, s, layout))
    assert torch.equal(pk.raw_fp8_tree_triton_incremental(k, s, layout, cached, N), pk.raw_fp8_tree_triton(k, s, layout))  # not a strict prefix -> full
    print(f"atom={atom} root={root}: incremental == full (bitwise) over {N // step} chunks")


def timeit(fn, iters=20):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


for n in (16384, 65536, 131072):
    layout = TreeLayout(1, 256, n)
    old = pk.raw_fp8_tree_triton(k[: n - 8192], s[: n - 8192], TreeLayout(1, 256, n - 8192))
    t_full = timeit(lambda: pk.raw_fp8_tree_triton(k[:n], s[:n], layout))
    t_inc = timeit(lambda: pk.raw_fp8_tree_triton_incremental(k[:n], s[:n], layout, old, n - 8192))
    print(f"N={n:6d}: full tree {t_full:.3f} ms  incremental (+8192 tokens) {t_inc:.3f} ms")
