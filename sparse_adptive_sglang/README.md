# Adaptive-HISA sparse prefill — `adaptive_0921_h202` (snapshot 2026-09-21)

Source machine h20-9-57, tree `/DATA/disk0/qyl/code/adaptive_0921_h202`.
Base: https://github.com/xuyufei-a/sglang_hisa branch `hisa_pr`, git HEAD `faa198b4e` (`git_head.txt`);
working tree = HEAD + `changes_vs_HEAD.patch` + the untracked files listed in `delta/`.

`h202` forks `adaptive_0921_h201` (P-key partition, L/8 λ-DP split → L/64 target merge, two-level selection
in **decode** only) and moves the same two-level selection into **prefill**. Kernel-side changes inherited
from h201 are listed in `VERSION.md` / `SPEEDOPT_NOTES.md`.

## Layout

- `python/`   complete drop-in `python/` tree (`PYTHONPATH=<this>/python`; exactly what the e2e runner freezes and mounts)
- `delta/`    only the files that differ from git HEAD, for review
  - `python/.../nsa/adaptive_hisa/`   partition / summary / decode / **prefill_select.py** package
  - `python/.../nsa/nsa_indexer.py`   the sparse-prefill hook in `_get_topk_ragged`
  - `test/registered/unit/test_adaptive_hisa_sparse_prefill.py`   6 tests (config, expand kernel vs Python, admission, Top-K vs dense-restricted)
  - `docs_research/adaptive_hisa_sparse_prefill.md`   design + all measurements (zh)
  - `docs_research/speed_bench_128k_20260921.md`       128K TTFT/TPOT: DSA vs HISA-64 vs adaptive (zh)
- `runner/run_sparse_prefill_e2e.py`   e2e queue (LongBench v2 held-out 401 + RULER 32k/128k 364); `--arm`, `--quick-niah`
- `changes_vs_HEAD.patch`, `speedopt_vs_snapshot1219.patch`   diffs (see `SPEEDOPT_NOTES.md`)

## What sparse prefill does

sglang chunked prefill runs the prompt in 8192-query chunks. With
`SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL=1`:

1. every chunk builds the P-key partition + FP8 mean summaries of its sealed prefix on the side stream
   (`prefill_runtime.prepare_prefill_partition` no longer requires the final chunk);
2. the next chunk's indexer (`prefill_select.sparse_topk_core`) replaces the dense `[8192, N]` DSA logits with
   coarse summary logits → per-row leaf ranking → expand the best leaves to `budget` raw tokens
   (Triton `_expand_slots_kernel`, crossing leaf clipped) → exact fp8 logits on those tokens + the causal local
   window → fused Top-2048. Leaves that start inside `[0, sink_tokens)` are forced to the top (attention sink).
   All device-side, no `.item()`.
3. chunks 0/1 stay dense (prefix < budget); decode is unchanged from h201.

Knobs (env prefix `SGLANG_NSA_ADAPTIVE_HISA_`):

| env | default | meaning |
|---|---|---|
| `SPARSE_PREFILL` | 0 | enable (0 = h201 behaviour) |
| `SPARSE_PREFILL_ROWS` | 2048 | query rows per fine sub-batch |
| `SPARSE_PREFILL_CANDIDATES` | 0 | leaf-token budget for prefill (0 = decode `CANDIDATE_TOKENS`, 8192) |
| `SPARSE_PREFILL_FINAL_CANDIDATES` | 0 | budget for the final prompt chunk only |
| `SPARSE_PREFILL_DENSE_FINAL` | 0 | keep the dense DSA indexer for the final prompt chunk |

## Results (128K, 8×H20 TP8, DeepSeek-V3.2)

Speed (`speed_bench_128k_20260921.md`, 128K in / 128 out, CUDA graph for decode):

| | TTFT | TPOT |
|---|---:|---:|
| DSA | 68.4 s | 20.3 ms |
| HISA-64 (fixed blocks, two-level prefill+decode) | 38.4 s | 17.8 ms |
| h201 adaptive, decode-only | 68.9 s | 19.2 ms |
| h202 sparse prefill (single request, no graph) | 56.3 s | – |

Accuracy of the all-chunks-sparse arm (`sparse_prefill`, budget 8192, sink 64) vs h201 on the same held-out rows:

| | LongBench v2 (401) | RULER 32k (182) | RULER 128k (182) |
|---|---:|---:|---:|
| DSA | 0.504 | 0.872 | 0.821 |
| h201 decode-only | 0.501 | 0.882 | 0.825 |
| **h202 sparse prefill 8192** | **0.479** | 0.855 | **0.598** |

The 128k loss is concentrated in NIAH (`niah_multikey_1/2/3` 0.43 / 0.36 / 0.21 vs 0.93 / 1.0 / 1.0). The first
output token is produced by the **final** prompt chunk, whose last rows must locate the needle in the 128K prefix;
mean summaries over a uniform haystack dilute the needle's leaf (prefix Top-2048 recall 0.55 on the NIAH dump,
layer 3 only 0.18 — identical for fixed HISA-64 blocks).

Fix under evaluation: `SPARSE_PREFILL_DENSE_FINAL=1` (intermediate chunks sparse, final chunk dense).
128k `niah_multikey_1/2/3` on the 42 held-out rows: 0.71 / 0.93 / 0.93 (h201: 0.93 / 1.0 / 1.0); cost ≈ 1/8 of the
dense indexer time (~1.5–2 s TTFT at 128K). Full LongBench + RULER with this arm is next.

Details, per-task tables and the recall/mass sweeps: `delta/docs_research/adaptive_hisa_sparse_prefill.md`.

## Run

    PYTHONPATH=<this>/python
    SGLANG_NSA_ADAPTIVE_HISA_MODE=adaptive_decode
    SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC=key_sse  SGLANG_NSA_ADAPTIVE_HISA_SPLIT_BACKEND=gpu
    SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY=sync_nonoverlap  SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR=64
    SGLANG_NSA_ADAPTIVE_HISA_RAW_FP8_BUILDER=1  SGLANG_NSA_ADAPTIVE_HISA_GPU_STREAM=side
    SGLANG_NSA_ADAPTIVE_HISA_SINK=64  SGLANG_NSA_ADAPTIVE_HISA_TAIL=256  SGLANG_NSA_ADAPTIVE_HISA_CANDIDATE_TOKENS=8192
    SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL=1  [SGLANG_NSA_ADAPTIVE_HISA_SPARSE_PREFILL_DENSE_FINAL=1]
    SGLANG_NSA_FUSE_TOPK=0

Server: `--chunked-prefill-size 8192 --max-running-requests 1 --disable-radix-cache`. The log must contain
`adaptive-hisa sparse prefill layer=... final=...` lines once the third chunk is reached.

Unit tests (GPU): `PYTHONPATH=python python -m pytest delta/test/registered/unit/test_adaptive_hisa_sparse_prefill.py -q`
