#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOCAL_HYBRID_AGENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LITELLM_HOME="${LITELLM_HOME:-$HOME/litellm-hybrid}"

exec "$LITELLM_HOME/.venv/bin/python" \
    "$ROOT/hybrid_swe_experiment.py" "$@"
