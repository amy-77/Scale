#!/usr/bin/env bash
# Launch a dump-enabled official-DSA server and replay the Adaptive-HISA prompt set.
#
#   EXP_NAME=adaptive_hisa_dump_20260918 bash scripts/run_indexer_dump.sh
#
# Output: ${ROOT}/data/${EXP_NAME}/{dump/<rid>/L??/*.pt, manifest.jsonl, server.log}
set -euo pipefail

ROOT=/DATA/disk0/qyl
IMAGE=qyl/sglang-hisa:eval
MODEL=/workspace/qyl/models/deepseek-v3.2
PORT="${DUMP_PORT:-31730}"
EXP_NAME="${EXP_NAME:-adaptive_hisa_dump_$(date +%Y%m%d)}"
EXP="${ROOT}/data/${EXP_NAME}"
CONTAINER_EXP="/workspace/qyl/data/${EXP_NAME}"
CONTAINER="dsv32-${EXP_NAME//_/-}"
LAYERS="${LAYERS:-all}"
Q_STRIDE="${Q_STRIDE:-64}"
MAX_DECODE_STEPS="${MAX_DECODE_STEPS:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
RULER_PER_TASK="${RULER_PER_TASK:-2}"
LB_PER_GROUP="${LB_PER_GROUP:-1}"
# Adaptive-HISA Phase B configuration (see adaptive_hisa/config.py).
#   ADAPTIVE_MODE=off|build_only   SPLIT_BACKEND=gpu|cpu_reference
#   MERGE_POLICY=off|sync_nonoverlap|heap_reference   MERGE_ROUNDS, MAX_MERGE_LEN
#   BUILD_SUMMARIES=1 (gpu only)   PARTITION_OVERLAP=1 (cpu_reference only)
#   GRAPH_BUILD=1 (gpu only: CUDA-graph replay of the per-layer build)
#   GPU_STREAM=main|side (gpu only: build in-stream, or on a second stream joined at forward end)
# v7_reuse (the earlier A/B): SPLIT_BACKEND=cpu_reference MERGE_POLICY=heap_reference
#   MAX_MERGE_LEN=256 PARTITION_OVERLAP=1 BUILD_SUMMARIES=0
ADAPTIVE_MODE="${ADAPTIVE_MODE:-off}"
SPLIT_BACKEND="${SPLIT_BACKEND:-gpu}"
MERGE_POLICY="${MERGE_POLICY:-sync_nonoverlap}"
MERGE_ROUNDS="${MERGE_ROUNDS:-2}"
MAX_MERGE_LEN="${MAX_MERGE_LEN:-0}"
BUILD_SUMMARIES="${BUILD_SUMMARIES:-}"
GRAPH_BUILD="${GRAPH_BUILD:-}"
GPU_STREAM="${GPU_STREAM:-}"
PARTITION_OVERLAP="${PARTITION_OVERLAP:-0}"
PARTITION_REUSE_LOGITS="${PARTITION_REUSE_LOGITS:-1}"
PARTITION_LAYERS="${PARTITION_LAYERS:-all}"
PARTITION_LOG_LAYERS="${PARTITION_LOG_LAYERS:-0}"
FORWARD_TIMING="${FORWARD_TIMING:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
KEEP_SERVER="${KEEP_SERVER:-0}"

mkdir -p "${EXP}/dump"
log() { echo "[$(date '+%F %T')] $*"; }

cleanup() {
    docker logs "${CONTAINER}" >"${EXP}/server.log" 2>&1 || true
    if [[ "${KEEP_SERVER}" != 1 ]]; then
        docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if [[ "$(docker inspect --format '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)" != true ]]; then
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    if curl --fail --silent --max-time 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "port ${PORT} already served by another process" >&2
        exit 1
    fi
    log "starting ${CONTAINER} (layers=${LAYERS}, q_stride=${Q_STRIDE}, decode_steps=${MAX_DECODE_STEPS})"
    docker run -d \
        --name "${CONTAINER}" \
        --gpus all --ipc host --network host \
        -v "${ROOT}:/workspace/qyl" \
        -v "${ROOT}/cache/misa_router_pilot_v1:/root/.cache" \
        -e PYTHONPATH=/workspace/qyl/code/dpskv32/sglang-hisa/python \
        -e SGLANG_NSA_FUSE_TOPK=0 \
        -e SGLANG_NSA_PER_HEAD_INDEX=0 \
        -e SGLANG_NSA_EXPERIMENTAL_PREFILL_PER_HEAD=0 \
        -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
        -e SGLANG_NSA_INDEXER_DUMP_DIR="${CONTAINER_EXP}/dump" \
        -e SGLANG_NSA_INDEXER_DUMP_LAYERS="${LAYERS}" \
        -e SGLANG_NSA_INDEXER_DUMP_Q_STRIDE="${Q_STRIDE}" \
        -e SGLANG_NSA_INDEXER_DUMP_MAX_DECODE_STEPS="${MAX_DECODE_STEPS}" \
        -e SGLANG_NSA_ADAPTIVE_HISA_MODE="${ADAPTIVE_MODE}" \
        -e SGLANG_NSA_ADAPTIVE_HISA_SPLIT_BACKEND="${SPLIT_BACKEND}" \
        -e SGLANG_NSA_ADAPTIVE_HISA_MERGE_POLICY="${MERGE_POLICY}" \
        -e SGLANG_NSA_ADAPTIVE_HISA_MERGE_ROUNDS="${MERGE_ROUNDS}" \
        -e SGLANG_NSA_ADAPTIVE_HISA_MAX_MERGE_LEN="${MAX_MERGE_LEN}" \
        ${BUILD_SUMMARIES:+-e SGLANG_NSA_ADAPTIVE_HISA_BUILD_SUMMARIES="${BUILD_SUMMARIES}"} \
        ${GRAPH_BUILD:+-e SGLANG_NSA_ADAPTIVE_HISA_GRAPH_BUILD="${GRAPH_BUILD}"} \
        ${GPU_STREAM:+-e SGLANG_NSA_ADAPTIVE_HISA_GPU_STREAM="${GPU_STREAM}"} \
        -e SGLANG_NSA_ADAPTIVE_HISA_PARTITION_OVERLAP="${PARTITION_OVERLAP}" \
        -e SGLANG_NSA_ADAPTIVE_HISA_PARTITION_REUSE_LOGITS="${PARTITION_REUSE_LOGITS}" \
        -e SGLANG_NSA_ADAPTIVE_HISA_LAYERS="${PARTITION_LAYERS}" \
        -e SGLANG_NSA_ADAPTIVE_HISA_PARTITION_LOG_LAYERS="${PARTITION_LOG_LAYERS}" \
        -e SGLANG_NSA_ADAPTIVE_HISA_FORWARD_TIMING="${FORWARD_TIMING}" \
        "${IMAGE}" \
        python -m sglang.launch_server \
        --model-path "${MODEL}" \
        --served-model-name deepseek-v3.2 \
        --tp-size 8 --host 0.0.0.0 --port "${PORT}" \
        --trust-remote-code --reasoning-parser deepseek-v3 \
        --mem-fraction-static 0.75 --max-running-requests 1 \
        --disable-cuda-graph --disable-radix-cache \
        --json-model-override-args '{"use_hisa":false}' >/dev/null
fi

log "waiting for server"
for _ in $(seq 1 240); do
    if curl --fail --silent --max-time 5 "http://127.0.0.1:${PORT}/health_generate" >/dev/null 2>&1; then
        break
    fi
    if [[ "$(docker inspect --format '{{.State.Running}}' "${CONTAINER}" 2>/dev/null || true)" != true ]]; then
        docker logs "${CONTAINER}" | tail -50 >&2 || true
        echo "server container exited" >&2
        exit 1
    fi
    sleep 10
done
curl --fail --silent --max-time 5 "http://127.0.0.1:${PORT}/health_generate" >/dev/null

log "replaying prompts"
python3 "$(dirname "$0")/collect_indexer_dump.py" \
    --server "http://127.0.0.1:${PORT}" \
    --tokenizer "${ROOT}/models/deepseek-v3.2" \
    --ruler-subset "${ROOT}/data/ruler_low_score_subset_20260917" \
    --ruler-per-task "${RULER_PER_TASK}" \
    --longbench-json "${ROOT}/data/misa_assignment_router_v2/longbench_v2.json" \
    --longbench-per-group "${LB_PER_GROUP}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --out "${EXP}/manifest.jsonl" ${EXTRA_ARGS}

log "done: $(find "${EXP}/dump" -name '*.pt' | wc -l) files, $(du -sh "${EXP}/dump" | cut -f1)"
