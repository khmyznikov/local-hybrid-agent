#!/usr/bin/env bash
set -euo pipefail

LITELLM_HOME="${LITELLM_HOME:-$HOME/litellm-hybrid}"
PID_FILE="$LITELLM_HOME/litellm-proxy.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "No tracked LiteLLM hybrid proxy"
    exit 0
fi

pid=$(cat "$PID_FILE")
if [[ -r "/proc/$pid/cmdline" ]]; then
    command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline")
else
    command_line=""
fi
if [[ "$command_line" == *"litellm"* && "$command_line" == *"litellm_hybrid_config.yaml"* ]]; then
    kill -TERM "$pid"
    echo "Stopped LiteLLM hybrid proxy PID $pid"
else
    echo "Ignored stale LiteLLM PID $pid"
fi
rm -f "$PID_FILE"
