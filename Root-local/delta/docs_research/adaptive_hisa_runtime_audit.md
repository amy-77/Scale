# Adaptive-HISA runtime audit (Phase A, no code changes)

Date: 2026-09-19. Checkout: `/DATA/disk0/qyl/code/dpskv32/sglang-hisa`, branch `hisa_pr`.
Everything below was read from the local tree (plus `git show origin/hisa_pr:<path>`
for the public branch). Line numbers refer to the local working tree.

---

## 0. State of the local checkout

| Item | Value |
| --- | --- |
| Local HEAD | `faa198b4e` "Add MISA head selection and assignment router on top of DSA indexer." |
| `origin/hisa_pr` | `e399d9f33` (= `HEAD~1`). Local is **ahead by 1 commit**, behind by 0. |
| Tracked files modified in the working tree | `nsa/nsa_indexer.py` (+596), `nsa/headmap_probe.py`, `managers/schedule_batch.py`, `mem_cache/memory_pool.py`, `model_executor/forward_batch_info.py`, `models/deepseek_v2.py`, `scripts/run_group16_all64_router.sh` |
| Untracked (this work) | `nsa/adaptive_hisa/` (config, prefill_partition, prefill_runtime, reference, runtime, sidecar, cuda_select, csrc/adaptive_select.cu), `test/registered/unit/test_adaptive_hisa_prefill_partition.py`, `docs/research/adaptive_hisa.md`, `nsa/indexer_dump.py`, `scripts/run_indexer_dump.sh`, `scripts/collect_indexer_dump.py`, analysis scripts |

### 0.1 The public HISA implementation is not in the local tree

Commit `faa198b4e` **deleted** the public fixed-block HISA and replaced it with the
MISA/per-head experiment. The files named in the brief exist only on `origin/hisa_pr`:

| Requested file | Local | `origin/hisa_pr` |
| --- | --- | --- |
| `nsa/hisa/indexer.py` (`HisaIndexer`) | 0-byte placeholder | 1772 lines |
| `nsa/hisa/orchestrator.py` (4-stage pipeline) | missing | 305 lines |
| `mem_cache/hisa_memory_pool.py` (`HisaNSATokenToKVPool`, pool pages, allocator, watermark) | missing | 502 lines |
| `nsa/hisa/triton_kernels.py` (pool MQA, sparse paged MQA, mean pooling, force-maintain, coord transform) | missing | 2472 lines |
| `nsa/hisa/tilelang_kernels.py` (fp8 block/paged mean pooling) | missing | 765 lines |
| `nsa/hisa/hisa_topk_fused.py` + `csrc/hisa_topk_fused.cu` | missing | 321 + 994 lines |
| `nsa/hisa/fast_topk_runtime.py` + `.cu` | present, **no callers** | present |
| `config.use_hisa`, `hisa_k_block_size`, `hisa_block_topk` (`deepseek_v2.py:1372`, `model_runner_kv_cache_mixin.py:465`) | **absent** | present |
| `nsa_backend.py` HISA metadata (+269 lines), `mem_cache/common.py` pool hooks | absent | present |

What the local tree has instead:

* Official DSA indexer (`nsa_indexer.py`): DeepGEMM ragged/paged logits + `sgl_kernel.fast_topk_*` (Top-2048).
* MISA / per-head experiment (`per_head_paged.py`, `csrc/chunk16_quota_paged.cu`, `offline_unique_head_router.py`): 16-token `chunk_sum` (f32) maintained in `_store_index_k_cache`, 256-token pooled means scored with `torch.bmm`, fused FP8 quota refine to 720 picks per head. Opt-in only (`SGLANG_NSA_PER_HEAD_INDEX`, router JSON, headmap probe), `--disable-cuda-graph` required.
* Adaptive-HISA prefill partition builder (`adaptive_hisa/prefill_*.py`), wired into `_get_topk_ragged`; decode prototype (`adaptive_hisa/runtime.py`) **not wired**.

**Decision needed before Phase B** (see §5): build adaptive on the local base and cherry-pick the
public pool-page infrastructure, or restore `HisaIndexer` wholesale. Recommendation: the former.

---

## 1. Official DSA dataflow in this tree (what adaptive must plug into)

### 1.1 Indexer K cache (`memory_pool.py:1929-1948`, `NSATokenToKVPool`)

* Per layer `index_k_with_scale_buffer[l]`: `uint8 [num_pages, 64*(128+4)] = [num_pages, 8448]`,
  `num_pages = (index_buf_size + 64 + 1) // 64`. Page = 64 tokens; inside a page bytes
  `[0, 8192)` are fp8 keys (token-major, 128 B each), bytes `[8192, 8448)` are 64 fp32 scales.
  **132 B/token/layer**; single shared key head (`wk` is `ReplicatedLinear(hidden, 128)`).
* Physical pages are free-list allocated and **permuted**; logical page `i` of request `b` is
  `page_table_64[b, i]` (`nsa_backend.py` builds it from `req_to_token`). Radix-cache hits share
  physical pages between requests (page-aligned).
* Accessors (`memory_pool.py:1983-2061`, `index_buf_accessor.py`): `get_index_k_with_scale_buffer`
  (raw paged view, no copy), `get_index_k_scale_buffer` → Triton `GetKAndS` gathers the **whole
  visible prefix** into contiguous `k_fp8 [seq_len_sum,128] + k_scale [seq_len_sum,4]` (temp 132 B/token,
  every prefill call, every layer), `set_index_k_scale_buffer` → Triton `SetKAndS` (fp8+scale by slot).
* Store (`nsa_indexer.py:2453-2541`): CUDA fused `fused_store_index_k_cache` (bf16 → fp8 ue8m0 + scale, in place);
  `update_index_k_chunk_sum` only when MISA/router/probe flags are on.
* TP: indexer projections are all `ReplicatedLinear` (`nsa_indexer.py:300-321`); the K cache is a full
  replica on every attention-TP rank. CP prefill all-gathers K (`cp_all_gather_rerange_output`, `:477-497`).

### 1.2 Production scorer convention (`nsa_indexer.py:396-402`, `:2617-2631`, `:2651`)

```
q_fp8, q_scale = act_quant(query_bf16 [T,64,128], block_size=128, scale_fmt="ue8m0")  # fp8 [T,64,128], f32 [T,64,1]
raw_gate       = weights_proj(x)                                                     # f32 [T,64]
weights        = raw_gate * 64**-0.5 * q_scale * 128**-0.5                            # f32 [T,64,1] -> squeeze -> [T,64]
I[t,s]         = sum_h weights[t,h] * relu( q_fp8[t,h,:] . (k_fp8[s,:] * k_scale[s]) )  # DeepGEMM
```

`weights` **already contains `q_scale`**. Any calibration scorer must take `q_fp8` unscaled with these
`weights` (this is what `_append_adaptive_calibration_rows` does; parity test
`test_appended_fp8_scorer_preserves_official_rows_and_partition` passes at rel. 5e-4).

### 1.3 Prefill (`forward_cuda` → `_get_topk_ragged`, `nsa_indexer.py:2556-2812`, `:1940-2170`)

```
_get_q_k_bf16 → act_quant(q) → _store_index_k_cache(K)            (:2633-2657; dual-stream only under graph capture)
→ _get_raw_and_logits_head_gate → weights                             (:2700)
→ _note_adaptive_queries(q_fp8, weights)                              (:2718)  <- adaptive tail-24 capture
→ _get_topk_ragged:
     ks, ke = metadata.get_indexer_kvcache_range()                    (:1953)
     k_fp8, k_scale = pool.get_index_k_scale_buffer(...)  (full-prefix gather, logical order) (:1958)
     [adaptive] prepared = _prepare_adaptive_partition(...)           (:1985)
     [adaptive] q/w/ks/ke += 24 calibration rows (ks=0, ke=N_complete) (:2015)
     logits = deep_gemm.fp8_mqa_logits(q, (k_fp8,k_scale), w, ks, ke, clean_logits=False)   (:2030 / chunked :2100)
     official_logits = logits[:q_offset]; topk = metadata.topk_transform(official_logits, 2048, ks)  (unchanged)
     [adaptive] _schedule_adaptive_partition_scores(prepared, logits[q_offset:q_offset+24, :N_complete])  (:2040)
```

Chunked prefill: `_should_chunk_mqa_logits` (`:1834`) slabs the q rows when `num_q*num_k >= 8e6` and
memory is tight; calibration rows are appended only to the last slab. `skip_topk` layers /
`max_kv_len <= 2048` go through `_forward_cuda_k_only` (`:2173`), which stores K and returns no logits.

### 1.4 Decode (`_get_topk_paged`, `nsa_indexer.py:1384-1760`)

```
kv_view = index_k_with_scale_buffer[l].view(num_pages, 64, 1, 132)
schedule = metadata.paged_mqa_schedule_metadata  (precomputed / graph-stable; else get_paged_mqa_logits_metadata)
logits = deep_gemm.fp8_paged_mqa_logits(q_fp8[B,1,64,128], kv_view, weights[B,64], seqlens[B,1], page_table_64[B,P], schedule, P*64)
topk   = metadata.topk_transform(logits, 2048)   # sgl_kernel fast_topk_v2 / fused transform, CUDA-graph friendly
```

The DeepGEMM paged kernel is instantiated with `kNextN * kNumHeads == 64` (`sm90_fp8_paged_mqa_logits.cuh:22-42`),
i.e. `next_n = 1` with 64 heads. `indices=` in the Python signature is a SM100 varlen batch-pairing
helper (`scheduler/paged_mqa_logits.cuh:29,166`), **not** a sparse token list.

### 1.5 Public HISA (origin/hisa_pr) for reference

`orchestrator.py`: (1) fixed `k_block_size` mean pool (tilelang, incremental
`update_pool_for_completed_blocks` in `_store_index_k_cache` with a per-request watermark), (2) pool
scoring with `deep_gemm.fp8_paged_mqa_logits` over `pool_k_pages` (same 132-B page layout, per-request
pool page table, `context_lens = ceil(N/K)`), then `clean_and_force_maintain_logits_decode_triton`,
(3) `fast_topk_runtime(block_topk = 8192 // K)`, (4) `sparse_paged_mqa_triton` /
`block_sparse_mqa_triton` exact rescoring of the selected fixed-size blocks, then Top-2048.
`HisaNSATokenToKVPool` sizes the pool as `ceil(size / (K*64)) * 1.5 + 4` pages. All stage-3/4 kernels
assume equal block length `K ∈ {16..128}`.

---

## 2. Current Adaptive-HISA prefill builder: what exists, what it costs, where it breaks the contract

### 2.1 Implemented (working tree)

* Scheduler flag `prefill_final_cpu` / `prefill_new_tokens_cpu` (`schedule_batch.py:2549`, `forward_batch_info.py`) → last chunk detection without inferring from positions.
* Per-(layer, req, epoch) rolling tail-24 of `q_fp8`/`weights` (`prefill_runtime._RequestEpoch.note`, `.clone()` of 24 rows/layer/chunk).
* Admission (`prefill_runtime._admission_skip`): B=1, extend w/o speculative, no CP/PP, not under graph capture; prefix hit with <24 new tokens → skip with logged reason.
* Calibration scores: appended to the official `fp8_mqa_logits` call (`partition_reuse_logits_enabled`, default on) — zero extra K traffic; fallback canonical torch scorer (`calibration_scores`, key_tile 16384).
* Score-moment tree (`score_energies`): atom=1 … root=256, per node `count/sum[C]/sum_sq` → energy `Σ_c (sum_sq - sum²/n)`; torch level-wise ops (~40 launches/layer).
* Global λ-DP with exact `M = N_complete/8` and heap repair (`_dp_solve`, Numba, CPU), parity-tested against the offline `lambda_dp_repair`.
* One greedy adjacent-merge pass (`_merge_loop`, Numba, CPU): **sequential min-heap greedy**, threshold = split λ, `max_len = 256`, Ward cost in score space. Parity-tested against the offline heap version.
* Overlap mode: GPU phase on a side stream (`record_stream` on inputs, non-blocking D2H into pinned buffers), CPU phase on a 1-thread `ThreadPoolExecutor`, `_MAX_PENDING = 8`, host wait in `finish_prefill_partitions()` at the end of `DeepseekV2Model.forward` (`deepseek_v2.py:2281/2304/2306`).
* Output: `PrefillPartition(leaf_start int32[M], leaf_len uint16[M], n_complete, tail_len, lambda, repair_count, timings)` kept in host `STATE`. **No leaf means, no decode consumption.**

### 2.2 Measured (32K prompt, 4 × 8192 chunks, B=1, H20-3e, 61 layers)

| Run | e2e prefill (s) | Δ vs baseline | builder sums over 61 layers |
| --- | --- | --- | --- |
| `v6_baseline` | 12.221 | — | — |
| `v6_sync_all` (sync, torch scorer) | 12.970 | +0.75 (+6.1%); last chunk +0.65 | score 0.144, tree 0.023, dp 0.294, merge 0.115, total 0.576 |
| `v7_old_all` (overlap, torch scorer) | 12.472 | +0.25 (+2.1%) | side-stream events inflated by contention |
| `v7_reuse_all` (overlap, DeepGEMM piggyback) | 12.308 | **+0.09 (+0.7%)**; last chunk +0.037 | score 0, dp 0.290, merge 0.115; `final_wait 0.043` |
| `v7_reuse_l0` (layer 0 only) | 12.143 | ≈0 | `final_wait 0.807` (!) |

Per layer at 32K (sync): tree 0.37 ms GPU, λ-DP 4.8 ms CPU, merge 1.9 ms CPU → CPU work is 73% of
builder time. Non-final chunks carry +0.02–0.06 s each (per-layer admission checks + clones).
The 0.8 s `final_wait` in the layer-0-only run is the Numba on-disk cache load on the worker's
first job (hidden behind 60 other layers in the all-layers runs) — a first-request stall today.

### 2.3 Contract deviations in the current version (must be fixed in Phase B/D)

1. λ-DP, repair and merge run on the **CPU** after a D2H copy of the energy tree (§7, §18 of the brief).
2. Host-side wait (`finish_prefill_partitions`) at the end of every final-chunk forward instead of CUDA events; worker thread + backpressure instead of stream ordering.
3. Score-moment tree = ~40 small torch kernels per layer; pinned staging allocations per layer.
4. `ATOM`, `ROOT`, `LEAVES_PER_ROOT` are module constants (functions accept `atom/root` kwargs, but there is no server-arg plumbing).
5. No K summaries, no adaptive pool, no decode path; `adaptive_hisa/runtime.py` + `sidecar.PageAtoms` (P-radius sealing, CPU rerank; `[pages,64,128] f32` atoms = 4× the K cache at atom=1) are dead code and contradict the new contract → delete.
6. Prefix-cache hit with <24 fresh tokens is skipped, not rebuilt from cached raw K.
7. Merge semantics: current runtime and offline reference are **sequential heap greedy** (cheapest adjacent pair first, re-evaluate neighbours). The brief's "synchronous non-overlapping rounds" is a *different* algorithm. See §4.5 for a GPU scheme that reproduces the sequential result exactly.

---

## 3. Answers to the ten questions

**Q1. HEAD / diffs.** `hisa_pr @ faa198b4e`, 1 commit ahead of `origin/hisa_pr (e399d9f33)`.
That commit removes `hisa/indexer.py`, `hisa/orchestrator.py`, `hisa/triton_kernels.py`,
`hisa/tilelang_kernels.py`, `hisa/hisa_topk_fused.*`, `mem_cache/hisa_memory_pool.py`,
`nsa_mtp_verification.py`, the `use_hisa` wiring in `deepseek_v2.py` / `model_runner_kv_cache_mixin.py` /
`nsa_backend.py` / `mem_cache/common.py`, and adds `per_head_paged.py`, `csrc/chunk16_quota_paged.cu`,
`offline_unique_head_router.py`, `headmap_probe.py`. The working tree then adds `adaptive_hisa/`
and modifies `nsa_indexer.py`, `schedule_batch.py`, `memory_pool.py`, `forward_batch_info.py`,
`deepseek_v2.py` (full list in §0). Only `hisa/fast_topk_runtime.*` survives from the public HISA, unused.

**Q2. Where tail q_fp8 / weights are free.** `nsa_indexer.py:2718` (`_note_adaptive_queries`), right after
`weights` is formed at `:2700` (`_get_raw_and_logits_head_gate`) and `q_fp8` at `:2651`
(`act_quant`), before the Top-K dispatch. Rows `q_fp8[q_offset-24:q_offset]` with
`q_offset = sum(metadata.get_nsa_extend_len_cpu())` are the prompt tail of this chunk; for chunks
shorter than 24 the rolling ring supplies the rest. No projection is recomputed. In the graph-capture
dual-stream path the same point is after `current_stream.wait_stream(alt_stream)` (`:2630/2649`), so
the tensors are complete.

**Q3. Paged DeepGEMM without a K gather.** Yes, two ways. (a) During prefill the official path has
*already* gathered the prefix contiguously (`get_index_k_scale_buffer`), so appending 24 rows to
`fp8_mqa_logits` costs no extra K traffic — implemented, `score_s = 0`. (b) Outside prefill (prefix-cache
rebuild, k-only layers) call `deep_gemm.fp8_paged_mqa_logits(q[24,1,64,128], kv_view, w[24,64],
context_lens=[24,1]·N_complete, block_table=page_table_64[b].expand(24,-1).contiguous(), schedule, P*64)`
— 24 "requests" sharing one block table, no gather, output `[24, P*64]` sliced to `N_complete`.
Constraint: `next_n=1` (kernel is compiled for `next_n*heads == 64`). Cost at 128K: K is streamed once per
query row (24 × 17 MB ≈ 0.4 GB, ~0.15 ms) + 51 GFLOP fp8 (~0.1–0.2 ms).

**Q4. Redundant fixed pooling.** In the default official-DSA configuration nothing pooled is built, so
nothing is wasted today. Redundant *if* adaptive replaces the coarse stage: (i) MISA `update_index_k_chunk_sum`
(`nsa_indexer.py:2488-2498, 2531-2541`; f32 `[num_pages*4,128]` = 32 B/token/layer) — gate it off under
`hisa_mode=adaptive`; (ii) the public HISA `update_pool_for_completed_blocks` + watermark (only on
`origin/hisa_pr`) — must not be enabled together with adaptive; (iii) `sidecar.PageAtoms`
(`note_stored_keys`, unwired) — delete; (iv) `runtime._seal_to` per-root P-radius sealing — delete.

**Q5. Proposed state / buffers and memory at 128K** (T = pool token capacity, N = 131072, 61 layers, per TP rank):

| Buffer | Layout | Size |
| --- | --- | --- |
| `summary_pool[l]` | `uint8 [S_pages, 64*132]`, same layout as `index_k_with_scale_buffer`; one row per leaf; `S_pages = ceil((T/8)/64)*slack` (hard cap from `M_max = ceil(N_max/8)` per request) | `T/8 × 132 B × 61` = **1/8 of the index-K cache** (T=500K → 0.50 GB); per 128K request 16384 rows × 132 B × 61 = **132 MB** |
| `summary_meta[l]` | `int32 [S_pages*64, 2]` (`leaf_start`, `leaf_len`) addressed by the same summary slot; `leaf_len` bounded by `ROOT` | `T/8 × 8 B × 61` (T=500K → 30 MB); per 128K request 8 MB |
| `req_summary_pages` | `int32 [max_reqs+1, max_summary_pages_per_req]` page table (public `HisaReqToPoolPagePool` pattern) | 128K → 256 pages/req → 1 KB/req |
| `num_leaves` / `epoch` | `int32 [61, max_reqs+1]` GPU + CPU mirror; `index_ready[req]` CUDA event (CPU) | negligible |
| `calib_ring` | `fp8 [slots, 61, 24, 64, 128]` + `f32 [slots, 61, 24, 64]`; only chunk-prefilling requests need it (1 chunked req at a time in the scheduler) → `slots = 2–4` | 12.4 MB per slot |
| shared scratch (reused across layers, sized for `N_max`) | scores `[24, N_max] f32` only in the paged path (piggyback slices the official logits) 12.6 MB; moment tree `~(24+2)·4 B·2N` ≈ 27 MB; DP/merge arrays ≈ 2 MB; leaf-mean f32 `[M_max,128]` 8 MB | ≈ 50 MB total |
| decode scratch per step | leaf scores `[B, M_max] f32`, sorted ids, `cand_ids [B, 8192] i32`, gathered `cand_K [B·8192, 132 B]`, `cand_scores [B, 8192]` | ≈ 1.2 MB × B |

Total resident ≈ 12.5% of the index-K cache + <100 MB; per 128K request ≈ 140 MB vs 1.06 GB of raw index K.

**Q6. Exact launch point for layer L.** `_get_topk_ragged`, immediately after `deep_gemm.fp8_mqa_logits`
returns for the final slab (`nsa_indexer.py:~2030` unchunked, `~2110` chunked) and **before**
`topk_transform`: `inputs_ready[L] = current_stream.record_event()`; `builder_stream.wait_event(inputs_ready[L])`;
inputs = `logits[q_offset:q_offset+24, :N_complete]` (score signature) and `k_fp8/k_scale` (contiguous,
logical order — leaf means need no page table); both `record_stream(builder_stream)`. The main stream
continues into `topk_transform` and layer L+1 with no wait. For layers without logits (`_forward_cuda_k_only`,
MHA prefill with `return_indices=False`) record `inputs_ready[L]` after `_store_index_k_cache` and use the
paged calibration path (Q3b). Do not use `alt_stream`: `forward_cuda` waits on it within the same layer
(`:2630`, `:2649`), which would serialize the builder.

**Q7. Where the first decode waits.** One stream-side wait, no host sync: at the end of the final-chunk
prefill forward (`deepseek_v2.py:2281/2304/2306`, today's `_finish_adaptive_partitions`) do
`torch.cuda.current_stream().wait_event(index_ready[req])`, where `index_ready` is recorded on the
builder stream after the last layer's summary write. Every later kernel on the model stream (sampling
of this forward, the first decode batch) is ordered after it, including CUDA-graph replays, without
putting an external-event wait inside a captured graph. Exposed cost = builder tail of the last
~1–2 layers. Per-layer lazy waits inside `_get_topk_paged` are possible in eager mode but not under
graph replay; not needed. Requests without a partition (`num_leaves == 0`: prefix hit, admission skip)
must route the batch to the official paged path — a batch-level decision in the scheduler/metadata,
not a per-row branch inside a captured kernel.

**Q8. Reusable as-is.**
* `deep_gemm.fp8_mqa_logits` — calibration rows (done).
* `deep_gemm.fp8_paged_mqa_logits` — (a) paged calibration (Q3b); (b) **decode leaf scoring** over
  `summary_pool` with `req_summary_pages` as block table and `context_lens = num_leaves` — exactly the
  public HISA stage 2; the kernel does not care that rows summarize different token counts.
* `act_quant` (bf16 → fp8 ue8m0, block 128) — requantize leaf means with the production convention;
  `SetKAndS` (Triton) — write summary rows by slot; `GetKAndS`/`gather_compact` — gather ≤8192 candidate keys (1 MB).
* `sgl_kernel.fast_topk_v2` — final Top-2048 over the `[B, 8192]` candidate scores (row ≥ 2048 ✓).
* From `origin/hisa_pr` (restore as files): `HisaPoolPageAllocator`, `HisaReqToPoolPagePool`, watermark/free
  hooks in `mem_cache/common.py`, `clean_and_force_maintain_logits_decode_triton` (mask ≥ `num_leaves`, force
  sink/tail), `hisa_coord_transform` idea. `sparse_paged_mqa_triton` / `block_sparse_mqa_triton` are fixed-K
  only; with atom=1 leaves the exact refine is gather + production scorer instead.
* Existing NSA CUDA-graph machinery: `paged_mqa_schedule_metadata` precompute/refresh (`nsa_backend.py`).

**Q9. Truly new kernels** (Triton unless noted):
1. Segmented leaf-mean: contiguous `k_fp8/k_scale` (prefill) or paged K (rebuild) → dequant in registers →
   segmented sum over `[leaf_start, leaf_start+leaf_len)` → mean → `act_quant`-compatible fp8 + scale →
   `summary_pool` slot. One program per leaf looping ≤4 page tiles; benchmark vs a bucketed-by-length variant.
2. Budgeted leaf → token expansion: sorted leaf order + prefix-summed lengths → `cand_ids [B, 8192]` (-1 pad),
   sink/tail guards, partial last leaf, deterministic ties (score desc, then start asc). Fixed shapes for graph capture.
3. GPU λ-DP: level-wise cost/keep over all roots for a batch of λ candidates (`[K_λ, nodes]`), leaf-count
   reduction, 2–3 refinement rounds → ≈30 launches; exact-M repair as batched top-d split with a sequential
   fallback when a child gain exceeds a competing leaf gain (§4.4). Can start as torch ops.
4. Merge rounds on GPU (§4.5). Can start as torch ops.
5. Optional fusion of the score-moment tree (one program per 256-root computing all 8 levels) to replace ~40 launches.

**Q10. Predicted latency and bottleneck** (B=1, H20-3e, 61 layers; 32K numbers measured, 128K extrapolated):

| Stage | 32K today | 128K, after Phase D (GPU) |
| --- | --- | --- |
| calibration_score (piggyback) | ≈0 (24 extra GEMM rows) | ≈0; paged fallback 0.3–0.5 ms/layer |
| score_tree_build | 0.37 ms/layer GPU (40 launches) | ~1 ms/layer torch, ~0.2 ms fused |
| lambda_dp + repair | 4.8 ms/layer **CPU** | 0.3–0.6 ms/layer GPU |
| merge | 1.9 ms/layer **CPU** | 0.3–0.6 ms/layer GPU (10–20 rounds) |
| key_summary_build | not built | 17 MB read → <0.1 ms + launch |
| adaptive_total / 61 layers | 0.58 s builder (+0.75 s e2e sync incl. stalls), +0.09 s e2e overlapped | ≈0.1–0.2 s sync (0.2–0.4% of a ~50 s prefill), mostly hidden |
| first-decode stall | host wait 27–43 ms (0.8 s Numba cold) | builder tail of 1–2 layers, ≈3–5 ms |
| decode indexer / layer | official: 17 MB paged scan + topk ≈ 0.2–0.3 ms | leaf score (M=16K rows) + sort + expand + gather + rescore 8K + topk ≈ 0.15–0.3 ms, **8–10 launches** |

Biggest bottlenecks: today, the CPU λ-DP/merge (73% of builder time) and the Numba cold start; after
Phase D, **decode launch count** — the official decode indexer at 128K/B=1 is already only ~0.25 ms/layer,
so the adaptive decode path only wins if it is captured in the CUDA graph and its 8–10 small kernels are
fused/persistent; on the prefill side, the builder's HBM contention with the next layer (measured: overlap
hides ~85% at 32K). Recall, not latency, is the reason to do this (offline: far-token recall 9% → 97%).

---

## 4. Proposed dataflow

### 4.1 Gate

`--nsa-hisa-mode {off, adaptive}` (server arg → `get_global_server_args()`), default `off`. `off` keeps
today's bytes on every path (official DSA; MISA flags untouched). `adaptive` allocates the buffers of Q5,
disables `update_index_k_chunk_sum`, enables the prefill builder and the decode path. `atom`, `root`,
`leaves_per_root` (M = N/leaves_per_root), `merge_rounds` (0/1), `candidate_tokens=8192`, `calib_queries=24`
become configurable fields; no hard-coded 4/8/16/64 in the adaptive path.

### 4.2 Prefill (per layer L, final chunk only)

```
main stream            builder stream (dedicated, created once)
----------------       ----------------------------------------
gather K (GetKAndS)
fp8_mqa_logits(+24 rows)
record inputs_ready[L] -> wait_event(inputs_ready[L])
topk_transform (official)   scores = logits[q_off:q_off+24, :N_complete]     (view, record_stream)
layer L+1 ...               moment tree  (level-wise, score space, count/sum[C]/sum_sq)
                            λ-DP over all roots, exact M, deterministic ties  (GPU)
                            merge round(s)                                     (GPU, gated)
                            leaf means from contiguous k_fp8/k_scale -> act_quant -> summary_pool slots
                            summary_meta[slot] = (leaf_start, leaf_len); num_leaves[L, req] = M'
                            if L == last: record index_ready[req]
end of forward: current_stream.wait_event(index_ready[req])      (stream-side only)
```

Scratch is one shared workspace sized for `N_max`, reused layer to layer; the builder is serial on its
stream so reuse is safe. K-only layers use the paged calibration kernel. Non-final chunks only append to
the calibration ring (24 rows/layer). Nothing is retained per layer except the summary rows and metadata.

### 4.3 Decode (per layer, B requests, all with `num_leaves > 0`)

```
leaf_scores = fp8_paged_mqa_logits(q_fp8, summary_pool[l].view(S,64,1,132), weights, num_leaves[l][B,1], req_summary_pages, sched, M_max)
clean(-inf beyond num_leaves), force sink/tail leaves
order      = sort leaf_scores desc (M ≤ 16K; deterministic tie on start)      # or radix select on a token-budget cutoff
cand_ids   = expand(order, summary_meta, budget=8192)  -> [B, 8192] logical token ids, -1 pad (partial last leaf)
cand_K     = gather_compact(index_k_with_scale_buffer[l], cand_ids, page_table_64)   # 8192 × 132 B
cand_score = production scorer on (q_fp8, cand_K)  (fp8_mqa_logits with ks=0/ke=8192 per row, or paged over scratch pages)
topk2048   = fast_topk_v2(cand_score masked) -> map slot -> cand_ids -> logical positions (same contract as topk_transform)
```

Logical ↔ physical mapping happens only in `gather_compact` (page table applied once), so page permutation
cannot change the candidate set. `fast_topk_runtime` (unsorted output) is not used for the budget cut.

### 4.4 λ-DP on GPU with exact M

Bisection evaluates `K_λ` candidates per round in one level-wise pass (cost arrays `[K_λ, nodes_at_level]`),
counts leaves per λ with one reduction, narrows the bracket, 2–3 rounds (~30 launches, no D2H). Repair:
deficit `d = M − count(λ_hi)`; take the top-`d` split gains among current splittable leaves and split them at
once; this equals the sequential heap when no newly created child gain exceeds the smallest chosen gain
(checked on GPU with one comparison); otherwise fall back to sequential steps for the remaining deficit
(each step = argmax + 2 updates). Ties: larger gain first, then smaller start — the reference's order.
Both `lambda_dp_repair` (reference) and the GPU solver stay in the tree; parity test on random and
adversarial trees.

### 4.5 Merge semantics (corrected 2026-09-19)

The first version of this section claimed that merging all reciprocal-nearest-neighbour adjacent
pairs in parallel reproduces the sequential heap exactly because Ward's linkage is reducible. That
is wrong for *adjacent-only* Ward merging with a length cap. Counterexample (from the external review,
`merge_semantics_checks.json`, now `test_sync_nonoverlap_differs_from_heap_on_review_counterexample`):
leaves `(start,len,mean) = (0,128,1) (128,64,4) (192,64,0) (256,64,3) (320,64,100) (384,128,200)`,
threshold 500, cap 256. Adjacent costs are `384, 512, 288, …`; the heap pops `(192,256)` (288), then the
*recomputed* pair `(128,[192..320))` (266.7) and stops → `[(0,128),(128,192),(320,64),(384,128)]`. One
synchronous round selects the non-overlapping local minima `(0,128)` and `(192,256)` together →
`[(0,192),(192,128),(320,64),(384,128)]`. The two algorithms produce different partitions, and the
reducibility argument does not apply (`ΔE(A∪B,C)` can be smaller than both old distances when
`mean(A∪B)` lands near `mean(C)`).

What Phase B implements, therefore, is the paper's algorithm as written in the sibling reference
`batched_adjacent_merge_rounds` (E10): `merge_policy=sync_nonoverlap` freezes the leaves, scores every
adjacent pair, visits eligible pairs (`cost < alpha·λ`, optional `len_l+len_r <= max_merge_len`) in
`(cost, start)` order, takes a pair when neither leaf is already taken this round, merges all taken
pairs at once, and repeats for `merge_rounds` (default 2) rounds or until a round takes nothing. The
GPU version (`partition_gpu.merge_sync_nonoverlap`) is bit-identical to the numpy port
(`partition_reference.sync_nonoverlap_merge`) on random / zero / piecewise / sparse inputs, with and
without a cap. The old sequential heap stays as `merge_policy=heap_reference` (CPU only, used by the
`v7_reuse` A/B baseline) and is never selected implicitly.

### 4.6 Phases

* **B** (sync correctness): gate + buffers + summary_meta; builder in the main stream after `fp8_mqa_logits`;
  torch-op GPU tree/DP/merge or the current CPU solver behind a flag as reference; leaf means + `act_quant`
  + `SetKAndS`; tests 1–8, 12, 13.
* **C** (decode correctness): leaf scoring via paged DeepGEMM, GPU reference selector (sort + cumsum + expand),
  gather + production rescore + `fast_topk_v2`; compare candidate sets and Top-2048 recall with the offline simulator; tests 9–11.
* **D**: GPU λ-DP/merge (4.4/4.5), fused moment tree, shared workspace, no pinned staging, no worker thread.
* **E**: builder stream + `inputs_ready`/`index_ready`; bit-identical vs sync (test 14); TTFT A/B sync/async/off at 32K/64K/128K.
* **F**: CUDA-graph decode (static shapes, precomputed schedule for the summary pool), B>1, prefix-cache rebuild via paged calibration, slot reuse reset (`num_leaves=0`, epoch++ in `ReqToTokenPool.free`), CP/PP explicit fallback with log.

---

## 5. Open decisions / risks

1. **Base branch.** The public HISA infrastructure exists only on `origin/hisa_pr`. Proposal: keep the local
   base (official DSA + adaptive prefill) and restore only `hisa_memory_pool.py`'s allocator/page-table
   classes and the two small Triton helpers as a starting point for `summary_pool`. Restoring `HisaIndexer`
   wholesale would re-introduce fixed-K assumptions (`block_topk = 8192 // K`, stage-4 tile = block).
2. **Merge semantics.** Resolved: the two algorithms differ (§4.5 counterexample). Phase B ships the
   brief's synchronous non-overlapping rounds as the default (`sync_nonoverlap`, 2 rounds) and keeps
   the sequential heap as an explicit CPU reference (`heap_reference`).
3. **Decode benefit at B=1.** The official paged indexer at 128K costs ~0.25 ms/layer; the adaptive decode
   path must be graph-captured to avoid regressing latency. Recall is the primary goal.
4. **Layer 0** (and possibly other shallow layers) should fall back to the official indexer per the offline study;
   `wants_layer` already exists for this.
5. **Docs hook.** `scripts/ci/check_no_docs_changes.py` rejects committed changes under `docs/`; this note lives
   next to `docs/research/adaptive_hisa.md` (both untracked) until a location is agreed.
