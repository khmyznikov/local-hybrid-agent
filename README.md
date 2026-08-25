# local-hybrid-agent

Local Qwen3.8-27B serving and hybrid cloud/local agent experiments for an
NVIDIA RTX Spark N1X running Windows 11 ARM64 and WSL2 Ubuntu ARM64.

The project includes:

- A validated vLLM/FlashInfer WSL setup for SM12.1.
- An OpenAI-compatible local server with explicit quality and capacity profiles.
- Direct-local GitHub Copilot CLI BYOK launchers.
- A cloud-primary MCP sidekick with deterministic search, local model calls,
  full local Copilot-agent delegation, and parallel small-agent batching.
- A LiteLLM pre-routing experiment combining local Gittensor Qwen with the
  GitHub Copilot model pool through one BYOM endpoint.
- Reproducible context, fidelity, MTP, cache, concurrency, and
  cloud-credit evaluation tools.

## Validated Local Profiles

- Checkpoint: `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`
- Runtime: vLLM 0.27.1, PyTorch CUDA 13.x, FlashInfer 0.6.16.post3
- GPU: RTX Spark N1X, SM12.1, 32,704 MiB partition
- Resident model: approximately 16.19 GiB
- Default quality profile: BF16 KV, 11 GiB / 175,529 tokens, 131,072-token limit
- Opt-in capacity profile: FP8 KV, 11 GiB / 351,058 tokens, 262,144-token limit
- Active requests: up to 4
- Decode graphs: batch sizes 1, 2, and 4
- Local endpoint: `http://127.0.0.1:8001/v1`
- Served model: `qwen3.8-27b-local`

The capacity profile completed a 262,080-token prompt plus 32 output tokens.
Its checkpoint does not contain q/k/v/prob attention scale tensors, so vLLM
uses scale 1.0 and warns that FP8 attention may lose accuracy. It is not the
default quality profile.

A controlled greedy literal/tool-call canary at 40K and 96K produced identical
output token IDs with BF16 and FP8 KV under pinned FlashInfer attention. This is
a narrow regression result, not evidence that uncalibrated FP8 is equivalent on
arbitrary agentic workloads. The included fidelity probe should be rerun with
representative captured work after changing any inference component.

The same canary compared the Gittensor capacity profile with
`unsloth/Qwen3.8-27B-NVFP4` at 40K, 96K, and 240K. Both checkpoints passed and
produced identical output token IDs. Gittensor used 26.4% less resident model
memory and had lower elapsed time at every tier; Unsloth supplied calibrated
k/v scales and avoided Gittensor's k/v scale-1 fallback. See
[QWEN38_NVFP4_COMPARISON.md](QWEN38_NVFP4_COMPARISON.md) for methodology,
measurements, caveats, and the checkpoint recommendation.

## Quick Start

Read [wsl_qwen38_setup_and_stress.md](wsl_qwen38_setup_and_stress.md) for the
runtime installation and CUDA/FlashInfer setup.

Start the local server from PowerShell:

```powershell
& .\start_qwen38_copilot_server.ps1
```

Request the maximum-context FP8 profile explicitly:

```powershell
& .\start_qwen38_copilot_server.ps1 -Profile Capacity
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
See [LITELLM_HYBRID_EXPERIMENT.md](LITELLM_HYBRID_EXPERIMENT.md) for the WSL
pre-routing setup and focused SWE-bench Verified pilot.

## Key Measurements

- Native cloud subagent: 25.51 s and 0.927 Copilot AI credits for the validated
  fixture task.
- Cloud parent plus local full-harness subagent: 74.58 s and 0.471 cloud AI
  credits, saving 49.2% of cloud credits with a 2.92x latency penalty.
- Two concurrent approximately 128K prompts fit but behaved mostly as queued
  prefills rather than latency-improving parallel work.
- The six-case WSL LiteLLM/SWE-bench pilot resolved 5/6 focused tests: 2/3 on
  local Gittensor and 3/3 on Copilot GPT-5.4. Cloud-only also resolved 5/6.
  Hybrid reduced Copilot calls by 39.0% and cloud token volume by 40.4%, but
  increased total agent time by 7.41x.
- GPT-5.6 Sol cloud-only also resolved 5/6, using 1.9% fewer tokens but taking
  2.01x as long as GPT-5.4. The Sol hybrid resolved 5/6 in its corrected latest
  results, reduced Sol token volume by 53.8%, and took 4.89x as long as forced
  Sol. Sol requires the Responses API; one local Responses adapter failure led
  to explicit hybrid-alias cloud fallbacks.
- Two new GPT-5.4 cases preserved identical per-case correctness between
  hybrid and forced cloud. Across the expanded eight-case latest-result view,
  both resolved 6/8; hybrid reduced cloud calls by 33.3% and cloud tokens by
  28.2%, while increasing total agent time by 6.00x.

These are machine-specific engineering measurements, not general model quality
claims. Run representative workloads and use the included JSONL evaluator before
choosing a routing policy.

## Repository Layout

- `start_*copilot*.ps1`: server and Copilot launchers.
- `local_sidekick_mcp.py`: MCP bridge and admission control.
- `evaluate_hybrid_copilot.py`: local/cloud/hybrid evaluator.
- `benchmark_qwen38_*.py`: long-context, cache, fidelity, and MTP tests.
- `hybrid_swe_experiment.py`: WSL-native LiteLLM/SWE-bench pilot harness.
- `QWEN38_NVFP4_COMPARISON.md`: controlled Gittensor/Unsloth comparison.
- `windows_native_runtime_gaps.md`: native Windows ARM64 runtime gaps.
- `prebuild_flashinfer_fp4_wsl.py`: constrained-memory FP4 JIT prebuild.
- `sync_qwen38_reference_template.py`: pinned first-party template audit input.
- `shard_qwen38_raw.py`: safetensors resharing utility.

Generated model files, wheels, virtual environments, logs, traces, metrics, and
benchmark result JSONL are intentionally excluded from Git.