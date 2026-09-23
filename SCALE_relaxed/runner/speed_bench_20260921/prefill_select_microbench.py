"""Parity + micro-benchmark: batched weighted select vs argsort/cumsum/expand."""
import os, sys, time
sys.path.insert(0, "/workspace/qyl/code/adaptive_0921_h202/python")
import torch
from sglang.srt.layers.attention.nsa.adaptive_hisa import prefill_select as ps

torch.manual_seed(0)
dev = "cuda"


def make_leaves(n_complete, cap, n):
    # random dyadic-ish lengths summing to n_complete over n leaves
    assert n_complete >= n
    w = torch.randint(1, 5, (n,), device=dev).float()
    lens = torch.floor(w * ((n_complete - n) / w.sum().item())).to(torch.int32) + 1
    rem = n_complete - int(lens.sum())
    assert 0 <= rem < n + 1
    lens[:rem] += 1
    assert lens.min() > 0 and int(lens.sum()) == n_complete
    starts = torch.zeros(n, dtype=torch.int32, device=dev)
    starts[1:] = torch.cumsum(lens, 0)[:-1].to(torch.int32)
    leaf_start = torch.zeros(cap, dtype=torch.int32, device=dev)
    leaf_len = torch.zeros(cap, dtype=torch.int32, device=dev)
    leaf_start[:n] = starts
    leaf_len[:n] = lens
    return leaf_start, leaf_len


def reference_rows(coarse, leaf_start, leaf_len, n, budget):
    """(score desc, index asc) selection, clipped crossing leaf -> sorted token set."""
    out = []
    sc = coarse[:, :n].cpu()
    st = leaf_start[:n].cpu().tolist()
    ln = leaf_len[:n].cpu().tolist()
    for r in range(sc.shape[0]):
        row = sc[r].tolist()
        order = sorted(range(n), key=lambda i: (-row[i], i))
        toks = []
        rem = budget
        for i in order:
            if rem <= 0:
                break
            if ln[i] <= 0 or row[i] == float("-inf"):
                continue
            take = min(rem, ln[i])
            toks.extend(range(st[i], st[i] + take))
            rem -= take
        out.append(sorted(toks))
    return out


def check(n_q, n_complete, cap, n, budget, local_len, ties=False, seed=1):
    torch.manual_seed(seed)
    leaf_start, leaf_len = make_leaves(n_complete, cap, n)
    num_leaves = torch.tensor([n], dtype=torch.int32, device=dev)
    if ties:
        coarse = torch.randint(0, 8, (n_q, cap), device=dev).float()
    else:
        coarse = torch.randn(n_q, cap, device=dev)
    col = torch.arange(cap, device=dev)
    coarse.masked_fill_((col >= n).unsqueeze(0), float("-inf"))
    coarse.masked_fill_(((leaf_start < 64) & (col < n)).unsqueeze(0), float("inf"))
    rows = slice(0, n_q)
    new = ps.select_candidates_weighted(coarse, leaf_start, leaf_len, num_leaves, rows, budget, n_complete, local_len)
    new2, seg, seg_count = ps.select_candidates_weighted(
        coarse, leaf_start, leaf_len, num_leaves, rows, budget, n_complete, local_len, intervals=True)
    torch.cuda.synchronize()
    assert torch.equal(new, new2)
    assert new.shape == (n_q, budget + local_len)
    # local window slots
    if local_len:
        exp_local = torch.arange(n_complete, n_complete + local_len, dtype=torch.int32, device=dev)
        assert torch.equal(new[:, budget:], exp_local.unsqueeze(0).expand(n_q, -1))
    ref = reference_rows(coarse, leaf_start, leaf_len, n, budget)
    leaf_part = new[:, :budget].cpu()
    for r in range(n_q):
        vals = leaf_part[r].tolist()
        got = sorted(v for v in vals if v >= 0)
        assert got == ref[r], f"row {r}: mismatch (ties={ties}) len {len(got)} vs {len(ref[r])}"
        assert all(v == -1 for v in vals[len(ref[r]):]) and all(v >= 0 for v in vals[: len(ref[r])])
    # segments reproduce the same tokens
    ss, sl, so = (t.cpu() for t in seg)
    sc = seg_count.cpu()
    for r in range(n_q):
        k = int(sc[r])
        toks = []
        segs = sorted((int(so[r, j]), int(ss[r, j]), int(sl[r, j])) for j in range(k))
        for off, st, ln in segs:
            assert off == len(toks) and ln > 0
            toks.extend(range(st, st + ln))
        assert toks == leaf_part[r].tolist()[: len(toks)]
        assert len(toks) == min(budget, sum(1 for _ in ref[r]))
    # legacy path parity (argsort has unspecified tie order -> only for distinct scores;
    # legacy requires the leaves to hold >= budget tokens)
    if not ties and n_complete >= budget:
        order, incl = ps.rank_leaves(coarse, leaf_start, leaf_len)
        old = ps.expand_candidates(order, incl, leaf_start, leaf_len, rows, budget, n_complete, local_len)
        old_sorted = torch.sort(old[:, :budget], dim=1).values
        new_sorted = torch.sort(new[:, :budget], dim=1).values
        assert torch.equal(old_sorted, new_sorted), "legacy parity failed"
        assert torch.equal(old[:, budget:], new[:, budget:])
    print(f"ok n_q={n_q} n_complete={n_complete} cap={cap} n={n} budget={budget} local={local_len} ties={ties}")


check(37, 4096, 128, 100, 1000, 17)
check(64, 122880, 1920, 1920, 8192, 0)
check(64, 122880, 1920, 1900, 8192, 1000)
check(64, 122880, 1920, 1900, 8192, 0, ties=True)
check(8, 20000, 512, 300, 16384, 0)           # budget < total but a big fraction
check(8, 5000, 512, 100, 8192, 0)             # budget > total -> select_all, -1 padded
check(16, 200000, 4096, 4096, 8192, 100)      # max capacity


def bench(n_q, cap, n, n_complete, budget, step=2048, iters=20):
    leaf_start, leaf_len = make_leaves(n_complete, cap, n)
    num_leaves = torch.tensor([n], dtype=torch.int32, device=dev)
    coarse = torch.randn(n_q, cap, device=dev)
    col = torch.arange(cap, device=dev)
    coarse.masked_fill_((col >= n).unsqueeze(0), float("-inf"))
    coarse.masked_fill_(((leaf_start < 64) & (col < n)).unsqueeze(0), float("inf"))

    def old():
        order, incl = ps.rank_leaves(coarse, leaf_start, leaf_len)
        outs = []
        for r0 in range(0, n_q, step):
            outs.append(ps.expand_candidates(order, incl, leaf_start, leaf_len, slice(r0, min(r0 + step, n_q)), budget, n_complete, 0))
        return outs

    def old_rank_only():
        return ps.rank_leaves(coarse, leaf_start, leaf_len)

    def new():
        outs = []
        for r0 in range(0, n_q, step):
            outs.append(ps.select_candidates_weighted(coarse, leaf_start, leaf_len, num_leaves, slice(r0, min(r0 + step, n_q)), budget, n_complete, 0))
        return outs

    def timeit(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    t_old = timeit(old); t_rank = timeit(old_rank_only); t_new = timeit(new)
    print(f"bench n_q={n_q} cap={cap} n={n} budget={budget}: "
          f"old(argsort+cumsum+expand)={t_old:.3f} ms [rank={t_rank:.3f}, expand={t_old - t_rank:.3f}]  "
          f"new(weighted select)={t_new:.3f} ms  speedup={t_old / t_new:.1f}x")


bench(8192, 1920, 1920, 122880, 8192)     # 128K prompt, last chunk
bench(8192, 1024, 960, 61440, 8192)       # 64K
bench(8192, 512, 480, 30720, 8192)        # 32K
bench(8192, 1920, 1920, 122880, 16384)    # final chunk with larger budget
bench(2048, 1920, 1920, 122880, 8192)     # 2048-row chunk
