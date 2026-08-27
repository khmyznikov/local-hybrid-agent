#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOCAL_HYBRID_AGENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
LITELLM_HOME="${LITELLM_HOME:-$HOME/litellm-hybrid}"
UV="${UV:-$HOME/.local/bin/uv}"
COPILOT="${COPILOT:-$HOME/.local/bin/copilot}"
export LITELLM_HOME

mkdir -p "$LITELLM_HOME"
"$UV" venv --python 3.13 "$LITELLM_HOME/.venv"
"$UV" pip install \
    --python "$LITELLM_HOME/.venv/bin/python" \
    'litellm[proxy]==1.99.0rc1'

if [[ ! -x "$COPILOT" ]]; then
    curl -fsSL https://gh.io/copilot-install | PREFIX="$HOME/.local" bash
fi

mkdir -p "$LITELLM_HOME/swebench"
"$LITELLM_HOME/.venv/bin/python" - <<'PY'
import os

from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="princeton-nlp/SWE-bench_Verified",
    repo_type="dataset",
    filename="data/test-00000-of-00001.parquet",
    local_dir=os.path.join(os.environ["LITELLM_HOME"], "swebench"),
)
PY

echo "LiteLLM: $($LITELLM_HOME/.venv/bin/litellm --version 2>&1 | tail -1)"
echo "Copilot: $($COPILOT --version | head -1)"
echo "Repository: $ROOT"
echo "Run authenticate_litellm_copilot_wsl.py if Copilot OAuth is not initialized."
