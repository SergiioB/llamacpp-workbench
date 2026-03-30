#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${LLAMA_WEBUI_REAP_MODEL_DIR:-$ROOT/models/reap_gguf}"
LOG_DIR="$MODEL_DIR/logs"
PID_FILE="$LOG_DIR/pipeline.pid"
TMUX_FILE="$LOG_DIR/pipeline.tmux"
GLM_PID_FILE="$LOG_DIR/glm-direct.pid"

echo "== files =="
ls -lh "$MODEL_DIR" 2>/dev/null || true
echo

echo "== disk =="
df -h "$MODEL_DIR" 2>/dev/null || true
echo

echo "== memory =="
free -h
echo

echo "== pipeline =="
if [[ -f "$TMUX_FILE" ]]; then
  session="$(cat "$TMUX_FILE")"
  if tmux has-session -t "$session" 2>/dev/null; then
    tmux list-panes -t "$session" -F 'session=#{session_name} pane_pid=#{pane_pid} command=#{pane_current_command}'
  else
    echo "tmux session not running: $session"
  fi
fi
if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  ps -p "$pid" -o pid,etime,pcpu,pmem,stat,cmd || true
else
  echo "no pid file"
fi
if [[ -f "$GLM_PID_FILE" ]]; then
  pid="$(cat "$GLM_PID_FILE")"
  ps -p "$pid" -o pid,etime,pcpu,pmem,stat,cmd || true
fi
echo

echo "== recent log =="
tail -n 40 "$LOG_DIR/pipeline.log" 2>/dev/null || true
tail -n 20 "$LOG_DIR/glm-direct.log" 2>/dev/null || true
