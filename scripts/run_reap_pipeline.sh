#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${LLAMA_WEBUI_REAP_MODEL_DIR:-$ROOT/models/reap_gguf}"
LOG_DIR="$MODEL_DIR/logs"
BENCH_DIR="$ROOT/data/benchmarks"
PY_BENCH="$ROOT/scripts/reap_benchmark.py"
PIPELINE_LOG="$LOG_DIR/pipeline.log"

mkdir -p "$MODEL_DIR" "$LOG_DIR" "$BENCH_DIR"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" >>"$PIPELINE_LOG"
}

download_exact() {
  local label="$1"
  local url="$2"
  local dst="$3"
  local expected_size="$4"
  local log_file="$5"

  mkdir -p "$(dirname "$dst")"
  touch "$log_file"

  while true; do
    local current_size=0
    if [[ -f "$dst" ]]; then
      current_size="$(stat -c%s "$dst" 2>/dev/null || echo 0)"
    fi

    if [[ "$current_size" == "$expected_size" ]]; then
      log "$label already complete: $dst ($current_size bytes)"
      return 0
    fi

    log "starting/resuming $label download: $dst"
    set +e
    wget -c --tries=0 --timeout=30 --waitretry=15 -O "$dst" "$url" >>"$log_file" 2>&1
    local rc=$?
    set -e

    current_size="$(stat -c%s "$dst" 2>/dev/null || echo 0)"
    if [[ "$current_size" == "$expected_size" ]]; then
      log "$label download complete: $dst ($current_size bytes)"
      return 0
    fi

    log "$label download incomplete (rc=$rc size=$current_size expected=$expected_size), retrying in 30s"
    sleep 30
  done
}

run_bench() {
  local label="$1"
  local model_path="$2"
  local bench_log="$3"

  log "benchmark start: $label"
  python3 "$PY_BENCH" \
    --model "$model_path" \
    --label "$label" \
    >>"$bench_log" 2>&1
  log "benchmark finished: $label"
}

main() {
  log "pipeline started"

  download_exact \
    "glm-reap-q3km" \
    "https://huggingface.co/unsloth/GLM-4.7-Flash-REAP-23B-A3B-GGUF/resolve/main/GLM-4.7-Flash-REAP-23B-A3B-Q3_K_M.gguf?download=1" \
    "$MODEL_DIR/GLM-4.7-Flash-REAP-23B-A3B-Q3_K_M.gguf" \
    "11273383648" \
    "$LOG_DIR/glm-download.log"

  run_bench \
    "glm47flash-reap23b-a3b-q3km" \
    "$MODEL_DIR/GLM-4.7-Flash-REAP-23B-A3B-Q3_K_M.gguf" \
    "$LOG_DIR/glm-benchmark.log"

  download_exact \
    "qwen-reap-q4km" \
    "https://huggingface.co/barozp/Qwen-3.5-28B-A3B-REAP-GGUF/resolve/main/Qwen-3.5-28B-A3B-REAP-Q4_K_M.gguf?download=1" \
    "$MODEL_DIR/Qwen-3.5-28B-A3B-REAP-Q4_K_M.gguf" \
    "17264580864" \
    "$LOG_DIR/qwen-download.log"

  run_bench \
    "qwen35-reap28b-a3b-q4km" \
    "$MODEL_DIR/Qwen-3.5-28B-A3B-REAP-Q4_K_M.gguf" \
    "$LOG_DIR/qwen-benchmark.log"

  log "pipeline complete"
}

main "$@"
