# Qwen3.8 NVFP4 on WSL2 ARM64

This guide documents the validated WSL2 ARM64 runtime, model preparation,
serving profile, context limits, and stability checks for an NVIDIA RTX Spark
N1X. It intentionally omits performance measurements.

## Validated Components

- Windows 11 ARM64
- WSL2 Ubuntu ARM64
- NVIDIA RTX Spark N1X, SM12.1
- Python 3.13
- vLLM 0.27.1
- PyTorch with CUDA 13.x
- FlashInfer 0.6.16.post3
- Qwen3.8-27B ModelOpt NVFP4 checkpoint
- FP8 KV cache
- FlashInfer attention and ModelOpt NVFP4 linear kernels
- Triton/FLA Gated DeltaNet kernels

The current local model path used by the launchers is:

```text
/home/gkhmyznikov/models/Qwen3.8-27B-NVFP4-RTX5090
```

Adjust paths when using a different account or installation directory.

## WSL Memory Configuration

Create or update `%USERPROFILE%\.wslconfig`:

```ini
[wsl2]
memory=8GB
swap=16GB
```

Apply the configuration:

```powershell
wsl --shutdown
```

Eight GiB is the validated operational WSL allocation. Swap is a safety net;
normal serving should not actively depend on it.

## Install Build Tools and uv

Inside Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential ninja-build gcc g++ curl git
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Create the environment:

```bash
mkdir -p ~/vllm-qwen38-wsl
cd ~/vllm-qwen38-wsl
uv venv --python 3.13
source .venv/bin/activate
```

## Install the Runtime

```bash
uv pip install "vllm==0.27.1" --torch-backend=auto
uv pip install \
  "flashinfer-python==0.6.16.post3" \
  "cuda-bindings==13.3.1" \
  "cuda-tile==1.5.0" \
  "nvidia-cutlass-dsl==4.6.0" \
  "quack-kernels==0.6.1" \
  "fastsafetensors==0.3.3" \
  "tilelang==0.1.12" \
  "humming-kernels==0.1.10"
```

Align CUDA compilation packages with the PyTorch CUDA version installed in the
environment. The validated environment used CUDA 13 packages.

## Expose the CUDA Toolkit

The pip CUDA toolkit is under the virtual environment. Create a conventional
toolkit path and linker aliases:

```bash
sudo ln -sfn \
  "$HOME/vllm-qwen38-wsl/.venv/lib/python3.13/site-packages/nvidia/cu13" \
  /usr/local/cuda
sudo ln -sfn lib /usr/local/cuda/lib64
sudo ln -sfn /usr/lib/wsl/lib/libcuda.so /usr/local/cuda/lib/libcuda.so
sudo ln -sfn libcudart.so.13 /usr/local/cuda/lib/libcudart.so
```

Use these variables for compilation and serving:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export MAX_JOBS=1
```

## Model Download

Download the ModelOpt checkpoint from Hugging Face:

```bash
source ~/vllm-qwen38-wsl/.venv/bin/activate
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090",
    local_dir="/home/gkhmyznikov/models/Qwen3.8-27B-NVFP4-RTX5090",
    allow_patterns=[
        "*.safetensors",
        "*.json",
        "*.jinja",
        "*.txt",
        "*.model",
    ],
)
PY
```

No model files are stored in this repository.

## Prebuild FlashInfer FP4

The first ModelOpt load may compile a large SM12.1 FlashInfer module. Compile it
before loading model weights so the 8 GiB WSL allocation is not shared between
the model and compiler processes:

```bash
cd /mnt/c/Dev/local-hybrid-agent
source ~/vllm-qwen38-wsl/.venv/bin/activate
MAX_JOBS=1 python prebuild_flashinfer_fp4_wsl.py
```

The reusable module is stored in the FlashInfer cache under the user's home
directory.

## Validated Server Profile

Use the PowerShell launcher:

```powershell
& .\start_qwen38_copilot_server.ps1
```

The launcher configures:

- OpenAI-compatible endpoint on `127.0.0.1:8001`
- Served model name `qwen3.8-27b-local`
- 11 GiB explicit FP8 hybrid cache
- 262,144-token native model limit
- Four active sequence slots
- Decode graphs for batch sizes 1, 2, and 4
- No Inductor compilation
- FlashInfer autotuning and optional warmups disabled
- `qwen3_xml` tool parsing
- Qwen thinking disabled by default for bounded local-agent work

The process is owned by a hidden Windows `wsl.exe` host so the WSL distribution
remains alive. Stop it with:

```powershell
& .\stop_qwen38_copilot_server.ps1
```

## Health and API Checks

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8001/health

$headers = @{ Authorization = 'Bearer local-copilot' }
Invoke-RestMethod http://127.0.0.1:8001/v1/models -Headers $headers
```

The API key is a loopback-only placeholder used by the launchers. Change it if
the endpoint is exposed beyond the local machine.

## Capacity and Stability

The validated 11 GiB cache exposes 351,058 shared token slots. Practical
placements are approximately:

- One maximum-context request
- Two medium/large requests around half the native context
- Four small/medium requests around one quarter of the native context

The full native-window request completed without meaningful swap. Two large
requests around half context also completed, but their prefills behaved mostly
as queued work. Four short requests are supported by the active sequence slots.

Use `benchmark_qwen38_wsl_limits.py` to validate capacity and memory after
changing hardware partitioning, runtime packages, checkpoint, cache size, or
graph configuration. The benchmark records token counts, elapsed times, GPU
memory, WSL memory, swap, PSI, faults, and OOM events.

Example init-only capacity check:

```bash
export BENCHMARK_MODEL=/home/gkhmyznikov/models/Qwen3.8-27B-NVFP4-RTX5090
export BENCHMARK_CONTEXT_LENGTHS=262080
export BENCHMARK_OUTPUT_TOKENS=32
export BENCHMARK_KV_CACHE_GIB=11
export BENCHMARK_MAX_NUM_BATCHED_TOKENS=4096
export BENCHMARK_LINEAR_BACKEND=auto
export BENCHMARK_COMPILE=0
export BENCHMARK_CUDAGRAPH_MODE=FULL_DECODE_ONLY
export BENCHMARK_NOMINAL_GPU_MIB=32704
export BENCHMARK_INIT_ONLY=1
python benchmark_qwen38_wsl_limits.py
```

## Model and Runtime Changes

Rebuild or revalidate after changing any of:

- Python, PyTorch, CUDA, vLLM, or FlashInfer versions
- GPU architecture or firmware memory partition
- Checkpoint format or quantization recipe
- KV cache dtype or size
- Maximum context, sequence count, or scheduler token budget
- CUDA graph or compilation settings

The FlashInfer JIT cache and vLLM compilation artifacts are version-, ABI-,
CUDA-, and architecture-specific.