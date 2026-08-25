#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/c/Dev/vllm-qwen38-bench
LITELLM_HOME=/home/gkhmyznikov/litellm-hybrid
PID_FILE="$LITELLM_HOME/litellm-proxy.pid"

export GITHUB_COPILOT_TOKEN_DIR=/home/gkhmyznikov/.config/litellm/github_copilot
export HYBRID_ROUTING_LOG="$LITELLM_HOME/routing-events.jsonl"
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-hybrid-copilot}"
export LOCAL_QWEN_API_KEY="${LOCAL_QWEN_API_KEY:-local-copilot}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

curl --fail --silent --show-error http://127.0.0.1:8001/health >/dev/null
if curl --fail --silent \
    -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
    "http://127.0.0.1:${LITELLM_PORT:-4000}/health/liveliness" >/dev/null 2>&1; then
    echo "LiteLLM hybrid proxy is already healthy on port ${LITELLM_PORT:-4000}"
    exit 0
fi

if [[ -f "$PID_FILE" ]]; then
    old_pid=$(cat "$PID_FILE")
    old_cmd=$(tr '\0' ' ' < "/proc/$old_pid/cmdline" 2>/dev/null || true)
    if [[ "$old_cmd" == *"litellm"* && "$old_cmd" == *"litellm_hybrid_config.yaml"* ]]; then
        echo "LiteLLM hybrid proxy already runs as PID $old_pid"
        exit 0
    fi
    rm -f "$PID_FILE"
fi

echo $$ > "$PID_FILE"
cd "$ROOT"
exec "$LITELLM_HOME/.venv/bin/litellm" \
    --config "$ROOT/litellm_hybrid_config.yaml" \
    --host 0.0.0.0 \
    --port "${LITELLM_PORT:-4000}" \
    --num_workers 1