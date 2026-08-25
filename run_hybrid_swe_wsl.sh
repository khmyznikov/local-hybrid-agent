#!/usr/bin/env bash
set -euo pipefail

exec /home/gkhmyznikov/litellm-hybrid/.venv/bin/python \
    /mnt/c/Dev/vllm-qwen38-bench/hybrid_swe_experiment.py "$@"