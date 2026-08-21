# local-hybrid-agent

Local Qwen3.8-27B serving and hybrid cloud/local agent experiments for an
NVIDIA RTX Spark N1X running Windows 11 ARM64 and WSL2 Ubuntu ARM64.

The project includes:

- A validated vLLM/FlashInfer WSL setup for SM12.1.
- An OpenAI-compatible local server with FP8 KV cache and CUDA decode graphs.
- Direct-local GitHub Copilot CLI BYOK launchers.
- A cloud-primary MCP sidekick with deterministic search, local model calls,
  full local Copilot-agent delegation, and parallel small-agent batching.
- Reproducible context, fidelity, MTP, cache, concurrency, and
  cloud-credit evaluation tools.

## Validated Local Profile

- Checkpoint: `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`
- Runtime: vLLM 0.27.1, PyTorch CUDA 13.x, FlashInfer 0.6.16.post3
- GPU: RTX Spark N1X, SM12.1, 32,704 MiB partition
- Resident model: approximately 16.19 GiB
- FP8 hybrid cache: 11 GiB / 351,058 tokens
- Native request limit: 262,144 tokens
- Active requests: up to 4
- Decode graphs: batch sizes 1, 2, and 4
- Local endpoint: `http://127.0.0.1:8001/v1`
- Served model: `qwen3.8-27b-local`

The complete 262,080-token prompt plus 32 output tokens passed. Four short
requests completed concurrently with similar per-request latency to a single
short request.

## Quick Start

Read [wsl_qwen38_setup_and_stress.md](wsl_qwen38_setup_and_stress.md) for the
runtime installation and CUDA/FlashInfer setup.

Start the local server from PowerShell:

```powershell
& .\start_qwen38_copilot_server.ps1
```

Run the entire Copilot CLI harness against local Qwen:

```powershell
& .\start_local_copilot.ps1
```

Run a cloud-primary Copilot session with the local MCP sidekick:

```powershell
& .\start_hybrid_copilot.ps1
```

Stop the server:

```powershell
& .\stop_qwen38_copilot_server.ps1
```

The launchers contain machine-specific default paths for the validated machine.
Adjust the WSL distribution, model path, virtual environment path, and local
checkout path when using a different host.

## Hybrid Agent Tools

The MCP server in [local_sidekick_mcp.py](local_sidekick_mcp.py) exposes:

- `local_capacity`: machine-readable request/cache scheduling limits.
- `local_search`: deterministic count/match search with optional local summary.
- `local_sidekick`: bounded one-shot local model work.
- `local_agent`: a nested GitHub Copilot CLI process using the local provider.
- `local_agent_batch`: 2-4 independent small local Copilot agents in parallel.

The advertised placement policy is:

- Up to 4 independent small tasks.
- Up to 2 medium tasks.
- 1 exclusive large or maximum-context task.

See [HYBRID_COPILOT.md](HYBRID_COPILOT.md) for architecture, commands,
correctness results, concurrency measurements, and cloud-credit tradeoffs.

## Key Measurements

- Native cloud subagent: 25.51 s and 0.927 Copilot AI credits for the validated
  fixture task.
- Cloud parent plus local full-harness subagent: 74.58 s and 0.471 cloud AI
  credits, saving 49.2% of cloud credits with a 2.92x latency penalty.
- Two concurrent approximately 128K prompts fit but behaved mostly as queued
  prefills rather than latency-improving parallel work.

These are machine-specific engineering measurements, not general model quality
claims. Run representative workloads and use the included JSONL evaluator before
choosing a routing policy.

## Repository Layout

- `start_*copilot*.ps1`: server and Copilot launchers.
- `local_sidekick_mcp.py`: MCP bridge and admission control.
- `evaluate_hybrid_copilot.py`: local/cloud/hybrid evaluator.
- `benchmark_qwen38_*.py`: long-context, cache, fidelity, and MTP tests.
- `windows_native_runtime_gaps.md`: native Windows ARM64 runtime gaps.
- `prebuild_flashinfer_fp4_wsl.py`: constrained-memory FP4 JIT prebuild.
- `shard_qwen38_raw.py`: safetensors resharing utility.

Generated model files, wheels, virtual environments, logs, traces, metrics, and
benchmark result JSONL are intentionally excluded from Git.