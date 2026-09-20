# Adaptive-HISA — P-key, L/8 split -> L/64 target merge (code snapshot 2026-09-20)

Source machine: h20-9-57, tree `/DATA/disk0/qyl/code/dpskv32/sglang-hisa`
Repo: https://github.com/xuyufei-a/sglang_hisa, branch hisa_pr
Git HEAD: see `git_head.txt` (faa198b4e); working tree = HEAD + `changes_vs_HEAD.patch` + the untracked files under `delta/`.

## Layout

- `python/`                 complete drop-in `python/` tree (point `PYTHONPATH` at it, exactly what the e2e runner freezes and mounts)
- `delta/`                  only the files that differ from git HEAD, for review:
  - `python/.../nsa/adaptive_hisa/`   the whole partition/decode package (untracked in git)
  - `python/.../nsa/nsa_indexer.py`, `managers/schedule_batch.py`, `mem_cache/memory_pool.py`,
    `model_executor/{cuda_graph_runner,forward_batch_info}.py`, `models/deepseek_v2.py`   modified hooks
  - `python/.../nsa/indexer_dump.py`, `scripts/{run_indexer_dump.sh,collect_indexer_dump.py}`   offline dump tooling
  - `test/registered/unit/test_adaptive_hisa_{prefill_partition,decode}.py`   unit tests (74/74 pass for prefill_partition)
  - `docs_research/`        design notes (zh)
- `changes_vs_HEAD.patch`   `git diff HEAD -- python/` for the tracked files
- `runner/run_pkey_target64.py`   e2e queue (LongBench v2 + RULER 32k/128k) for this configuration; `run_pkey_align.py` is the previous P-key run it derives from
- `stats/`                  chunk-length distributions and target-merge convergence on 17 indexer dumps x 11 layers

## What is new vs. the 2026-09-19 snapshot / the finished P-key e2e run

1. `partition_metric=key_sse` (P-key): partition energy is the key SSE `sum_i ||k_i - mu_b||^2`, query independent.
2. Target-count merge (`config.merge_target_divisor`, env `SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR=64`):
   the lambda-DP split still produces L/8 leaves; the merge then converges to exactly `floor(L/64)` chunks
   (same chunk count as HISA at chunk size 64). Per round, on device and CUDA-graph capturable:
   Ward cost of every adjacent pair -> admit the `2*(count-target)` cheapest -> non-overlapping matching ->
   merge the `count-target` cheapest matched pairs. Never overshoots; 187/187 layer builds hit the target
   exactly in <= 5 rounds (`merge_target_rounds=8` is the cap). Method label: `P-key-sync_nonoverlap-target64`.
3. CUDA-graph capture is OOM-robust (evict-before-capture, one retry after `empty_cache`, cooldown instead of
   permanent disable) and the fp64 tree build is row-chunked.
4. Merge cost kernel tile 64 -> 32 rows (fp64, C=128), ~40% faster.

## Run

Server env (same image/cmd as production, plus):

    PYTHONPATH=<this>/python
    SGLANG_NSA_ADAPTIVE_HISA_PARTITION_METRIC=key_sse
    SGLANG_NSA_ADAPTIVE_HISA_MERGE_TARGET_DIVISOR=64

The warm-up log line must read `metric=key_sse method=P-key-sync_nonoverlap-target64`.
Without `MERGE_TARGET_DIVISOR` the old threshold merge (2 rounds, lambda*alpha, ~L/9.2 chunks) is used.

Unit tests (GPU):

    cd <tree>; PYTHONPATH=python python -m pytest test/registered/unit/test_adaptive_hisa_prefill_partition.py -q

Chunk-length statistics from `stats/` (17 dumps x 11 layers): after split mean 8.0 (p50 8, max 128);
after target merge mean 64.0 for every length (p10 ~40, p50 64, p90 ~90, p99 ~125, max 416).
