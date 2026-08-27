#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOCAL_HYBRID_AGENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LITELLM_HOME="${LITELLM_HOME:-$HOME/litellm-hybrid}"
PID_FILE="$LITELLM_HOME/litellm-proxy.pid"

export GITHUB_COPILOT_TOKEN_DIR="${GITHUB_COPILOT_TOKEN_DIR:-$HOME/.config/litellm/github_copilot}"
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
    if [[ -r "/proc/$old_pid/cmdline" ]]; then
        old_cmd=$(tr '\0' ' ' < "/proc/$old_pid/cmdline")
    else
        old_cmd=""
    fi
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
