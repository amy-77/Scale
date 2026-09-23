"""Partition build cost split per chunk (tree / dp / repair / merge / summary)."""
import os, sys
sys.path.insert(0, "/workspace/qyl/code/adaptive_0921_h202/python")
ENV = {
    "SGLANG_NSA_ADAPTIVE_HISA_MODE": "adaptive_decode", "SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC": "key_sse",
    "SGLANG_NSA_ADAPTIVE_HISA_SPLIT_BACKEND": "gpu", "SGLANG_NSA_ADAPTIVE_HISA_BUILD_SUMMARIES": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY": "sync_nonoverlap", "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR": "64",
    "SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_ROUNDS": "8", "SGLANG_NSA_ADAPTIVE_HISA_MAX_MERGE_LEN": "0",
    "SGLANG_NSA_ADAPTIVE_HISA_GRAPH_BUILD": "1", "SGLANG_NSA_ADAPTIVE_HISA_GPU_STREAM": "side",
    "SGLANG_NSA_ADAPTIVE_HISA_SINK_TOKENS": "64", "SGLANG_NSA_ADAPTIVE_HISA_TAIL_TOKENS": "256",
    "SGLANG_NSA_ADAPTIVE_HISA_CANDIDATE_TOKENS": "8192", "SGLANG_NSA_ADAPTIVE_HISA_DECODE_CHUNK": "64",
    "SGLANG_NSA_ADAPTIVE_HISA_SKIP_CLEAN": "1", "SGLANG_NSA_ADAPTIVE_HISA_FUSE_PAGE_TABLE": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_RAW_FP8_BUILDER": "1", "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL": "1",
    "SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_ROWS": "2048",
}
os.environ.update(ENV)
import torch
from sglang.srt.layers.attention.nsa.adaptive_hisa.config import get_config
from sglang.srt.layers.attention.nsa.adaptive_hisa.pkey_fp8_builder import build_partition_from_fp8, summaries_from_totals

dev = "cuda"
cfg = get_config()
print("atom", cfg.atom, "root", cfg.root, "compression", cfg.summary_compression, "lambda_max_rounds", cfg.lambda_max_rounds)
torch.manual_seed(0)
Nmax = 131072
k = (torch.randn(Nmax, 128, device=dev) + torch.randn(Nmax // 64, 128, device=dev).repeat_interleave(64, 0)).to(torch.float8_e4m3fn)
s = torch.rand(Nmax, device=dev) + 0.5


def ms(ev):
    a, b = ev
    return a.elapsed_time(b)


for N in (16384, 32768, 65536, 98304, 131072):
    for _ in range(2):
        part = build_partition_from_fp8(k[:N], s[:N], N, cfg, profile=True)
        totals = part.meta.pop("_merge_totals")
        summaries_from_totals(totals, part.leaf_len, None)
    torch.cuda.synchronize()
    reps = 5
    acc = {}
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        part = build_partition_from_fp8(k[:N], s[:N], N, cfg, profile=True)
        totals = part.meta.pop("_merge_totals")
        t0 = torch.cuda.Event(enable_timing=True); t0.record()
        summaries_from_totals(totals, part.leaf_len, None)
        t1 = torch.cuda.Event(enable_timing=True); t1.record()
        part.events["summary_s"] = (t0, t1)
        torch.cuda.synchronize()
        for key, ev in part.events.items():
            acc[key] = acc.get(key, 0.0) + ms(ev) / reps
    b.record(); torch.cuda.synchronize()
    wall = a.elapsed_time(b) / reps
    print(f"N={N:6d}: " + "  ".join(f"{key[:-2]}={val:.2f}" for key, val in acc.items()) + f"  wall={wall:.2f} ms  leaves={int(part.num_leaves)}")

# pure GPU time via CUDA graph replay (removes CPU launch gaps)
for N in (16384, 65536, 131072):
    kk, ss = k[:N].clone(), s[:N].clone()
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(2):
            part = build_partition_from_fp8(kk, ss, N, cfg)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g, stream=side):
            part = build_partition_from_fp8(kk, ss, N, cfg)
            totals = part.meta.pop("_merge_totals")
            summaries_from_totals(totals, part.leaf_len, None)
    except Exception as e:
        print("capture failed:", repr(e)[:200]); break
    g.replay(); torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(10): g.replay()
    b.record(); torch.cuda.synchronize()
    print(f"N={N:6d}: graph replay (pure GPU) = {a.elapsed_time(b) / 10:.2f} ms")
