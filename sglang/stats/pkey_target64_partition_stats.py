"""P-key target-count merge (L/8 split -> L/64 merged) on the 17 indexer dumps: convergence + build time."""
import os, sys, json, time
os.environ["SGLANG_NSA_ADAPTIVE_HISA_MODE"] = "build_only"
os.environ["SGLANG_NSA_ADAPTIVE_HISA_SPLIT_BACKEND"] = "gpu"
os.environ["SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC"] = "key_sse"
sys.path.insert(0, "/workspace/qyl/code/sglang_hisa/experiments")
import torch
from dataclasses import replace
from adaptive_hisa.schema import list_request_dumps, load_request_dump
from sglang.srt.layers.attention.nsa.adaptive_hisa import config as C
from sglang.srt.layers.attention.nsa.adaptive_hisa.prefill_partition import key_values, n_complete_tokens
from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_gpu import build_partition_gpu, graphed_builder, reset_graph_cache
from sglang.srt.layers.attention.nsa.adaptive_hisa.partition_kernels import warmup_triton_kernels

C.reset_config_cache(); base = C.get_config(); dev = torch.device("cuda")
DIV = int(os.environ.get("DIV", "64")); ROUNDS = int(os.environ.get("ROUNDS", "8"))
tgt = replace(base, merge_target_divisor=DIV, merge_target_rounds=ROUNDS)
warmup_triton_kernels(dev, tgt)
LAYERS = [0, 3, 7, 12, 18, 24, 30, 38, 46, 54, 60]
root = "/workspace/qyl/data/adaptive_hisa_dump_20260918/dump"
out = []
for req in list_request_dumps(root):
    rows = []
    for L in LAYERS:
        d = load_request_dump(req, L, queries="prefill", max_queries=1)
        n = d.k_fp8.shape[0]; done = n_complete_tokens(n, base.root)
        vals = key_values(d.k_fp8.to(dev), d.k_scale.to(dev), done)
        part = build_partition_gpu(vals, n, tgt).to_host()
        ln = part.leaf_len.to(torch.long)
        rm = list(part.meta.get("round_merge_counts") or [])
        rows.append(dict(layer=L, n=n, done=done, M0=done // 8, target=done // DIV, M=int(ln.numel()),
                         rounds=len(rm), merges=rm, max_len=int(ln.max()), p50=int(ln.median()),
                         frac_le8=float((ln <= 8).sum()) / ln.numel(), frac_ge128=float((ln >= 128).sum()) / ln.numel(),
                         tok_ge256=float(ln[ln >= 256].sum()) / done))
    r0 = rows[0]
    over = [r["M"] - r["target"] for r in rows]
    rec = dict(request=req.name.replace("dump:", ""), n=r0["n"], done=r0["done"], M0=r0["M0"], target=r0["target"],
               M_mean=sum(r["M"] for r in rows) / len(rows), over_max=max(over),
               rounds_max=max(r["rounds"] for r in rows), rounds_mean=sum(r["rounds"] for r in rows) / len(rows),
               max_len=max(r["max_len"] for r in rows), p50=sum(r["p50"] for r in rows) / len(rows),
               frac_le8=sum(r["frac_le8"] for r in rows) / len(rows), frac_ge128=sum(r["frac_ge128"] for r in rows) / len(rows),
               tok_ge256=sum(r["tok_ge256"] for r in rows) / len(rows), layers=rows)
    out.append(rec)
    print(f"{rec['request']:<44} N={rec['n']:>7} M0={rec['M0']:>6} target={rec['target']:>5} M={rec['M_mean']:>7.1f} "
          f"over_max={rec['over_max']:>3} rounds={rec['rounds_mean']:.1f}/{rec['rounds_max']} p50len={rec['p50']:.0f} "
          f"max_len={rec['max_len']:>5} le8={rec['frac_le8']:.2f} ge128={rec['frac_ge128']:.2f} tok>=256:{rec['tok_ge256']:.2f}", flush=True)
    print("   rounds(layer 30):", rows[5]["merges"], flush=True)
json.dump(out, open(os.environ.get("OUT", "/tmp/pkey_target_check.json"), "w"), indent=1)

# ---- build time: threshold (2 rounds) vs target (L/64) at the longest dump, graph replay path
longest = max(list_request_dumps(root), key=lambda r: load_request_dump(r, 0, queries="prefill", max_queries=1).k_fp8.shape[0])
d = load_request_dump(longest, 30, queries="prefill", max_queries=1)
n = d.k_fp8.shape[0]; done = n_complete_tokens(n, base.root)
vals = key_values(d.k_fp8.to(dev), d.k_scale.to(dev), done)
for name, cfg in (("threshold-2r", base), ("target-L/%d@%d" % (DIV, ROUNDS), tgt)):
    reset_graph_cache()
    g = graphed_builder(vals.shape[0], n, cfg, dev)
    for _ in range(3):
        g.build(vals, n)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(20):
        p = g.build(vals, n)
    torch.cuda.synchronize(); graph_ms = (time.perf_counter() - t) / 20 * 1e3
    for _ in range(2):
        build_partition_gpu(vals, n, cfg)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(10):
        build_partition_gpu(vals, n, cfg)
    torch.cuda.synchronize(); eager_ms = (time.perf_counter() - t) / 10 * 1e3
    M = int(p.to_host().leaf_len.numel())
    print(f"TIME {name:<18} N={n} M={M} graph={graph_ms:.3f} ms/layer eager={eager_ms:.3f} ms/layer", flush=True)
