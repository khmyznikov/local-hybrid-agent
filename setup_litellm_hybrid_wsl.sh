#!/usr/bin/env bash
set -euo pipefail

HOME_DIR=/home/gkhmyznikov
LITELLM_HOME="$HOME_DIR/litellm-hybrid"

mkdir -p "$LITELLM_HOME"
"$HOME_DIR/.local/bin/uv" venv --python 3.13 "$LITELLM_HOME/.venv"
"$HOME_DIR/.local/bin/uv" pip install \
    --python "$LITELLM_HOME/.venv/bin/python" \
    'litellm[proxy]==1.99.0rc1'

if [[ ! -x "$HOME_DIR/.local/bin/copilot" ]]; then
    curl -fsSL https://gh.io/copilot-install | PREFIX="$HOME_DIR/.local" bash
fi

mkdir -p "$LITELLM_HOME/swebench"
"$LITELLM_HOME/.venv/bin/python" - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="princeton-nlp/SWE-bench_Verified",
    repo_type="dataset",
    filename="data/test-00000-of-00001.parquet",
    local_dir="/home/gkhmyznikov/litellm-hybrid/swebench",
)
PY

echo "LiteLLM: $($LITELLM_HOME/.venv/bin/litellm --version | tail -1)"
echo "Copilot: $($HOME_DIR/.local/bin/copilot --version | head -1)"
echo "Run authenticate_litellm_copilot_wsl.py if Copilot OAuth is not initialized."
