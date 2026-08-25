#!/usr/bin/env bash
set -euo pipefail

PID_FILE=/home/gkhmyznikov/litellm-hybrid/litellm-proxy.pid

if [[ ! -f "$PID_FILE" ]]; then
    echo "No tracked LiteLLM hybrid proxy"
    exit 0
fi

pid=$(cat "$PID_FILE")
command_line=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
if [[ "$command_line" == *"litellm"* && "$command_line" == *"litellm_hybrid_config.yaml"* ]]; then
    kill -TERM "$pid"
    echo "Stopped LiteLLM hybrid proxy PID $pid"
else
    echo "Ignored stale LiteLLM PID $pid"
fi
rm -f "$PID_FILE"