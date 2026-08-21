# Hybrid Copilot with local Qwen sidekick

## Architecture

The stable setup keeps GitHub Copilot's API model as the primary reasoner and
orchestrator. A local MCP server exposes two bounded tools backed by the local
Qwen3.8-27B ModelOpt checkpoint:

- `local_sidekick`: extraction, classification, summarization, drafting, simple
  code inspection, and tool planning.
- `local_search`: workspace-contained text search with deterministic count or
  bounded-match modes and optional local-model summarization.

The primary model must verify important local results. Architecture decisions,
ambiguous reasoning, security conclusions, destructive actions, and final
verification stay with the API model.

Direct local mode is also available. It runs the complete Copilot CLI agent
harness against local Qwen through Copilot's OpenAI-compatible BYOK support.

The `local_agent` MCP tool runs a nested Copilot CLI process in BYOK mode. This
is the same agent loop and workspace tool harness, not a raw completion. Copilot
1.0.80 only supports per-subagent `model`, `effortLevel`, and `contextTier`;
custom-provider selection is process-wide. Therefore a built-in subagent cannot
use vLLM while its parent uses GitHub routing in the same process. The nested
process is the compatibility bridge.

## Server profile

The launcher uses the validated N1X profile:

- Model: `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`
- OpenAI model name: `qwen3.8-27b-local`
- URL: `http://127.0.0.1:8001/v1`
- API key: `local-copilot`
- FP8 hybrid cache: 11 GiB / 351,058 tokens
- Native request limit: 262,144 tokens
- Up to four active requests (`max_num_seqs=4`) with decode graphs captured for
  batches 1, 2, and 4
- FlashInfer SM121 ModelOpt NVFP4 linear kernels
- FlashInfer full attention, Triton/FLA GDN
- `FULL_DECODE_ONLY` CUDA graph, no Inductor
- `qwen3_xml` tool-call parser
- Thinking disabled by default for Copilot requests to reduce local agent-loop
  latency; the primary cloud model retains responsibility for deep reasoning

The start script keeps a hidden Windows `wsl.exe` host process alive while vLLM
runs in the foreground inside WSL. This is intentional: detached user/systemd
units were stopped when the non-lingering WSL interop session ended. The stop
script terminates both the Windows host and Linux server PID.

The full 262,080-token prompt plus 32 output tokens completed with 2,179 MiB
physical GPU headroom and no meaningful swap. With four sequence slots and
batch 1/2/4 graphs, the persistent server uses about 30.8 GiB after warmup and
retains about 1.8-2.0 GiB free.

## Concurrency

The N1X is not limited to a single request. Decode at batch one is mostly
weight-bandwidth bound, so batching several sequences amortizes each model-weight
read across several generated tokens. With short prompts and 256 output tokens:

| Active requests | Mean request time |
|---:|---:|
| 1 | 20.45 s |
| 2 | 20.82 s |
| 4 | 21.65 s |

Four short requests therefore complete with only a small increase in
per-request latency.

Prefill behaves differently. With approximately 8K prompt tokens and 64 output
tokens per request:

| Active requests | Mean request time |
|---:|---:|
| 1 | 9.70 s |
| 2 | 14.62 s |
| 4 | 25.76 s |

Chunked prefill shares the 4,096-token scheduler budget, so concurrent long
prompts increase aggregate work but raise latency. Two concurrent ~128K prompts
also fit (256,048 prompt tokens total): one completed in 150 s and the other in
287 s. This is useful queue processing, not interactive latency improvement.

The 351,058-token cache is shared across active requests. Practical combinations
are approximately one 262K request, two 128K requests, or four 64K requests;
request metadata and aligned GDN state require some reserve. At maximum native
context there is only room for one request. For interactive work, use up to four
short/medium requests, but avoid launching several large prefills together.

### Exposing capacity to the primary model

The MCP bridge exposes the scheduling envelope in three machine-readable places:

1. MCP initialization instructions state the placement rule: up to four small,
  at most two medium, or one exclusive large task.
2. `local_capacity` returns JSON with sequence slots, shared-cache tokens,
  context thresholds, and measured scheduling behavior.
3. `local_agent.workload_class` declares `small`, `medium`, or `large`; the
  parent should call `local_capacity` when uncertain.

For independent small work, the parent should make **one**
`local_agent_batch` call containing 2-4 tasks. The bridge launches isolated
nested Copilot agents concurrently and preserves result order. Do not issue four
separate MCP calls: one batch call communicates independence and lets the bridge
own admission control.

Measured local-agent batches on deterministic read/search tasks:

| Batch | Wall time | Result |
|---:|---:|---|
| 2 small agents | 94.9 s | both correct |
| 4 small agents | 125.4 s | all four correct |

Individual agents slow down while sharing decode bandwidth, but batch wall time
is substantially lower than serial full-harness execution. Each nested process
uses an isolated temporary `COPILOT_HOME` to prevent concurrent session-state
writes. Medium and large tasks are intentionally excluded from the batch tool.

## Commands

Start or confirm the server:

```powershell
& C:\Dev\vllm-qwen38-bench\start_qwen38_copilot_server.ps1
```

Run Copilot entirely against local Qwen:

```powershell
& C:\Dev\vllm-qwen38-bench\start_local_copilot.ps1
```

Run cloud-primary Copilot with the local MCP sidekick:

```powershell
& C:\Dev\vllm-qwen38-bench\start_hybrid_copilot.ps1
```

Arguments after the launcher name are passed to Copilot. For example:

```powershell
& C:\Dev\vllm-qwen38-bench\start_hybrid_copilot.ps1 --model gpt-5.4
```

Use `-ServerPort` when changing the local server port; other short flags such
as Copilot's `-p` pass through to Copilot unchanged.

Stop the server:

```powershell
& C:\Dev\vllm-qwen38-bench\stop_qwen38_copilot_server.ps1
```

Run the small repeatable comparison suite:

```powershell
& C:\Dev\vllm-qwen38-bench\.venv\Scripts\python.exe `
  C:\Dev\vllm-qwen38-bench\evaluate_hybrid_copilot.py
```

Results append to `hybrid_eval_results.jsonl`; local call latency and token use
append to `local_sidekick_metrics.jsonl`.

## Initial measurements

Three objective bounded tasks (extraction, classification, arithmetic) produced:

| Mode | Correct | Mean wall time | Mean Copilot AI credits |
|---|---:|---:|---:|
| Local-only | 3/3 | 24.38 s | 0 |
| Cloud-only (`auto`) | 3/3 | 9.07 s | 0.328 |
| Hybrid, forced one local call | 3/3 | 11.99 s | 0.381 |

The hybrid local calls themselves took 0.45-0.72 seconds and consumed 377 local
prompt tokens plus 14 local completion tokens in total. The rest of the hybrid
overhead came from cloud tool planning and verification.

A 60-match search-condensation task also scored correctly in all modes:

| Mode | Wall time | Copilot AI credits | Local work |
|---|---:|---:|---|
| Local-only | 43.17 s | 0 | full Copilot harness local |
| Cloud-only | 12.27 s | 0.364 | built-in search |
| Hybrid summary | 32.35 s | 0.381 | 20.05 s, 1,782 + 252 tokens |

Replacing local-model summarization with deterministic `result_mode=count`
removed that 20-second local generation step. With both cloud-only and hybrid
pinned to `gpt-5.6-luna`, the same task measured:

| Mode | Wall time | Copilot AI credits | Local work |
|---|---:|---:|---|
| Cloud-only | 11.65 s | 0.404 | built-in search |
| Hybrid deterministic search | **10.70 s** | 0.417 | 2.7 ms, no local LLM tokens |

This optimized route was 8.1% faster but still used 3.0% more cloud AI credits.
The MCP tool description and orchestration prompt are themselves context, so
even nearly free local computation does not automatically reduce cloud cost.
When using `--model auto`, the tool-bearing prompt may also select a more
expensive cloud model; pin the same cloud model when comparing economics.

These are smoke tests, not an accuracy benchmark. They show that forced local
delegation does not save money on small tasks: it adds a cloud tool round trip,
increasing both latency and AI-credit use. Local summarization only becomes
plausible when it prevents a much larger result from entering cloud context.

## Practical routing policy

Use local-first without a cloud call for isolated low-risk jobs when savings are
the goal. Use the direct-local launcher or invoke the MCP tool from an existing
session only when the main model already needs to remain in control.

Within a hybrid session:

1. Prefer deterministic `local_search` with `result_mode=count` or `matches`.
2. Use `result_mode=summary` only for genuinely large/noisy results.
3. Delegate drafting, extraction, and classification when their output can be
   cheaply checked.
4. Keep deep reasoning and final verification in the API model.
5. Do not delegate tiny tasks merely because the local model is available; the
   orchestration round trip costs more than direct cloud completion.

The launcher explicitly pre-authorizes `local-qwen-sidekick`; this matters for
non-interactive `-p` runs, where Copilot cannot display an MCP permission prompt.
A verified launcher smoke added one local metrics row and returned the expected
cloud-verified answer. In the full vLLM workspace that forced-tool turn still
used 4.33 AI credits and a 62K-token Copilot harness context. Repository context,
tool schemas, and orchestration can dominate the few local inference tokens, so
measure from the directory and model you actually intend to use.

## Measuring savings

Copilot reports AI credits, not a universal currency amount. Dollar conversion
depends on the account's plan and billing arrangement. For a workload, compare:

```text
savings = (cloud-only AI credits - hybrid AI credits) * account cost per credit
```

The current bounded-task result is negative: hybrid used about 16% more AI
credits than cloud-only. The optimized deterministic-search case narrowed that
to 3% more credits while improving latency by 8%. The next meaningful experiment
is a representative multi-step workload where local work eliminates a cloud
subagent call or a large repeated context transfer, not a single cheap cloud
answer.

## Same-harness subagent comparison

A valid same-workspace comparison asked both subagents to inspect a 90-line file
and derive a count without receiving the expected answer. Both returned the
correct count of 60 with tool evidence.

| Path | Whole-turn time | Cloud AI credits | Child execution |
|---|---:|---:|---:|
| Native cloud `general-purpose` subagent | **25.51 s** | 0.927 | 9.98 s, 2 tools |
| Cloud parent + local full-harness agent | 74.58 s | **0.471** | 58.40 s, 3 model calls / 3 tools |

The local full-harness path reduced cloud credits by **49.2%** but made the
whole turn **2.92x slower**. Local inference itself costs no Copilot credits;
the remaining 0.471 credits are the cloud parent's delegation and verification
turns plus MCP/tool schema context.

With Qwen thinking enabled, the same local child took 136 seconds. Disabling
thinking reduced it to roughly 58-63 seconds while preserving this deterministic
task's accuracy once exact-count tool evidence was required. For open-ended
reasoning, disabling thinking may reduce quality, so keep those tasks in the
cloud parent.

The cloud-search and deterministic-local-search paths had similar wall time
because search execution was not the bottleneck: local count took about 3 ms,
while both runs paid 8-10 seconds for Copilot startup, prompt construction, and
cloud inference. The native cloud subagent itself is fast because the cloud
model processes its large harness context and tool turns much faster than the
local decoder.