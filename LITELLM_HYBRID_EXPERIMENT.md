# LiteLLM hybrid Copilot experiment

This experiment places LiteLLM in front of one local Gittensor Qwen deployment
and either GitHub Copilot GPT-5.4 or GPT-5.6 Sol. Copilot CLI keeps its normal
agent/tool harness, while LiteLLM chooses the model before each inference call.

## Architecture

```text
Copilot CLI agent harness
        |
        v
LiteLLM aliases (WSL, port 4000)
        |-- hybrid-copilot (Chat Completions)
        |     |-- SIMPLE/MEDIUM -> Gittensor Qwen3.8-27B, BF16 KV
        |     `-- COMPLEX/REASONING -> GitHub Copilot GPT-5.4
        `-- hybrid-copilot-sol (Responses)
              |-- SIMPLE/MEDIUM -> Gittensor Qwen3.8-27B, BF16 KV
              `-- COMPLEX/REASONING -> GitHub Copilot GPT-5.6 Sol
```

This is different from the MCP sidekick. The MCP path pays for a cloud parent
to delegate and verify local work. The LiteLLM path runs only the model selected
for each inference request.

GPT-5.6 Sol rejects Chat Completions requests with
`unsupported_api_for_model`. Its direct and hybrid aliases therefore use the
Responses API; the GPT-5.4 aliases continue to use Chat Completions.

## Validated environment

- WSL2 Ubuntu ARM64
- LiteLLM `1.99.0rc1` in `/home/gkhmyznikov/litellm-hybrid/.venv`
- Copilot CLI `1.0.80` Linux ARM64 in `~/.local/bin/copilot`
- Gittensor quality profile at `http://127.0.0.1:8001/v1`
- LiteLLM proxy at `http://127.0.0.1:4000/v1`
- GitHub Copilot OAuth credentials stored under
  `~/.config/litellm/github_copilot` with mode `600`

The GitHub CLI OAuth token cannot be reused for this provider. LiteLLM uses its
own documented GitHub device flow and exchanges that token for short-lived
Copilot API credentials.

## Install in WSL

```bash
bash /mnt/c/Dev/vllm-qwen38-bench/setup_litellm_hybrid_wsl.sh
```

Initialize OAuth once:

```bash
~/litellm-hybrid/.venv/bin/python \
        /mnt/c/Dev/vllm-qwen38-bench/authenticate_litellm_copilot_wsl.py
```

Open the printed URL, enter the device code, and approve the application. Never
commit `access-token` or `api-key.json`.

## Start the proxy

Start or confirm Gittensor quality mode from PowerShell:

```powershell
& .\start_qwen38_copilot_server.ps1 -Profile Quality
```

Then start LiteLLM in WSL:

```bash
bash /mnt/c/Dev/vllm-qwen38-bench/start_litellm_hybrid_wsl.sh
```

Stop it with:

```bash
bash /mnt/c/Dev/vllm-qwen38-bench/stop_litellm_hybrid_wsl.sh
```

The proxy uses [litellm_hybrid_config.yaml](litellm_hybrid_config.yaml). Route
events append to `/home/gkhmyznikov/litellm-hybrid/routing-events.jsonl`; prompts
and response text are deliberately excluded.

## Run Copilot CLI through the router

```bash
export COPILOT_PROVIDER_BASE_URL=http://127.0.0.1:4000/v1
export COPILOT_PROVIDER_TYPE=openai
export COPILOT_PROVIDER_API_KEY=hybrid-copilot
export COPILOT_PROVIDER_WIRE_API=completions
export COPILOT_MODEL=hybrid-copilot
export COPILOT_PROVIDER_MAX_PROMPT_TOKENS=120000
export COPILOT_PROVIDER_MAX_OUTPUT_TOKENS=8192

copilot
```

Use the Responses-only Sol lane with:

```bash
export COPILOT_PROVIDER_BASE_URL=http://127.0.0.1:4000/v1
export COPILOT_PROVIDER_TYPE=openai
export COPILOT_PROVIDER_API_KEY=hybrid-copilot
export COPILOT_PROVIDER_WIRE_API=responses
export COPILOT_MODEL=hybrid-copilot-sol
export COPILOT_PROVIDER_MODEL_ID=gpt-5.6-sol
export COPILOT_PROVIDER_WIRE_MODEL=hybrid-copilot-sol
export COPILOT_PROVIDER_MAX_PROMPT_TOKENS=120000
export COPILOT_PROVIDER_MAX_OUTPUT_TOKENS=8192

copilot
```

Copilot CLI reports `premiumRequests: 0` for this BYOM lane even when LiteLLM
routes to the GitHub Copilot provider. Use proxy route events to count cloud
calls; do not interpret the CLI field as proof that the upstream Copilot call
was unmetered.

## Routing coverage

The local LiteLLM heuristic was run over all 500 SWE-bench Verified issue
statements without model inference:

| Tier | Cases | Route |
|---|---:|---|
| SIMPLE | 70 | Local Gittensor |
| MEDIUM | 256 | Local Gittensor |
| COMPLEX | 174 | Copilot GPT-5.4 |
| REASONING | 0 | Copilot GPT-5.4 |

This gives 326 local and 174 cloud issue-level routes. The stock heuristic is
not a learned SWE capability predictor; these counts describe the configured
experiment, not expected resolution quality.

## Executable SWE subset

`hybrid_swe_cases.json` fixes six SWE-bench Verified cases across pytest,
Sphinx, and Pylint: one local and one cloud-classified case from each repository.
Every case has a focused official FAIL_TO_PASS test that was validated to fail
at the base commit on WSL ARM64.

```bash
# Show all-500 routing coverage and the six selected routes.
bash run_hybrid_swe_wsl.sh classify

# Prepare isolated worktrees and Python 3.9 test environments.
bash run_hybrid_swe_wsl.sh prepare --mode hybrid

# Verify the official focused tests fail before model execution.
bash run_hybrid_swe_wsl.sh baseline --mode hybrid

# Run the six-case hybrid experiment.
bash run_hybrid_swe_wsl.sh run --mode hybrid

# Prepare, validate, and run the Responses-based Sol comparison.
bash run_hybrid_swe_wsl.sh prepare --mode cloud-sol
bash run_hybrid_swe_wsl.sh baseline --mode cloud-sol
bash run_hybrid_swe_wsl.sh run --mode cloud-sol
bash run_hybrid_swe_wsl.sh prepare --mode hybrid-sol
bash run_hybrid_swe_wsl.sh baseline --mode hybrid-sol
bash run_hybrid_swe_wsl.sh run --mode hybrid-sol

# Summarize the latest result per mode/case.
bash run_hybrid_swe_wsl.sh summarize
```

Each agent is capped at ten minutes. A timed-out patch is still captured and
tested. Generated worktrees, virtual environments, transcripts, patches, and
results stay under `/home/gkhmyznikov/litellm-hybrid/swebench` and are not
committed.

This is a WSL-native focused evaluation, not the canonical containerized
SWE-bench score. Docker/Podman is not available in the validated WSL instance,
so the harness evaluates the official FAIL_TO_PASS nodes after applying the
official test patch. It does not reproduce SWE-bench's per-instance Docker image
or full PASS_TO_PASS regression suite. Report it as a six-case focused pilot.

## Initial pilot results

All six selected official test patches produced the intended failing baseline
before model execution. The latest hybrid result for each case was:

| Instance | Initial route | Focused test | Elapsed | Model calls | Tools |
|---|---|---:|---:|---:|---:|
| `pylint-dev__pylint-4970` | Local | Failed | 606.47 s | 20 | 22 |
| `pylint-dev__pylint-6903` | Cloud | Passed | 38.39 s | 14 | 20 |
| `pytest-dev__pytest-7521` | Cloud | Passed | 37.87 s | 15 | 22 |
| `pytest-dev__pytest-7571` | Local | Passed | 366.64 s | 18 | 17 |
| `sphinx-doc__sphinx-10323` | Cloud | Passed | 22.45 s | 7 | 14 |
| `sphinx-doc__sphinx-9367` | Local | Passed | 157.37 s | 7 | 7 |

Overall, five of six focused FAIL_TO_PASS evaluations resolved. The local lane
resolved two of three cases and averaged 376.83 seconds; the Copilot GPT-5.4
lane resolved three of three and averaged 32.91 seconds. The unresolved local
case reached the ten-minute cap, edited the correct gold implementation file,
but still failed the official focused assertion. Across the pilot, LiteLLM
recorded 45 local model calls and 36 Copilot model calls.

Two agents added their own regression tests in files also touched by the
official test patch. The evaluator stores the complete agent patch, restores
those agent-authored test files to the base commit, and then applies the official
test patch before scoring. This prevents a correct implementation from being
marked unresolved because two equivalent test edits conflict textually.

The configured `session_affinity` did not produce affinity-pin decisions because
Copilot CLI's BYOM requests did not expose a LiteLLM `session_id`. LiteLLM
reclassified every agent model call, although every case remained on its initial
local or cloud lane because the issue text and score stayed stable. A future
iteration should add a stable session identifier at the client/proxy boundary
before relying on prompt-cache affinity.

These six cases are sufficient to validate the WSL architecture and expose the
local agent-loop latency gap. They are not sufficient to estimate a population
SWE-bench resolution rate or prove that the stock heuristic is an optimal
capability router.

## Cloud-only comparison

The same six cases were rerun in fresh worktrees with Copilot GPT-5.4 forced for
every model call. Setup and failing-baseline validation were identical and kept
outside measured agent time.

| Instance | Hybrid route | Hybrid | Cloud only | Hybrid / cloud |
|---|---|---:|---:|---:|
| `pylint-dev__pylint-4970` | Local | Failed, 606.47 s | Failed, 48.13 s | 12.60x |
| `pylint-dev__pylint-6903` | Cloud | Passed, 38.39 s | Passed, 33.92 s | 1.13x |
| `pytest-dev__pytest-7521` | Cloud | Passed, 37.87 s | Passed, 17.99 s | 2.11x |
| `pytest-dev__pytest-7571` | Local | Passed, 366.64 s | Passed, 19.87 s | 18.45x |
| `sphinx-doc__sphinx-10323` | Cloud | Passed, 22.45 s | Passed, 33.98 s | 0.66x |
| `sphinx-doc__sphinx-9367` | Local | Passed, 157.37 s | Passed, 12.07 s | 13.04x |

Both modes resolved the same five of six focused evaluations; both missed
`pylint-dev__pylint-4970`. Hybrid therefore showed no measured quality loss on
this subset, but total agent time increased from 165.96 seconds to 1,229.20
seconds (7.41x, or 17 minutes 43 seconds additional wall time).

Hybrid made 36 Copilot calls versus 59 for cloud only, a 39.0% reduction. It
sent 852,068 tokens through Copilot versus 1,429,951 for cloud only, a 40.4%
reduction (577,883 fewer cloud tokens). At the user-task level, three of six
sessions stayed local, so the expected premium-request-unit reduction is 50%
under GitHub's documented one-multiplied-request-per-user-prompt accounting.
The live enterprise model catalog does not expose GPT-5.4's multiplier, so no
exact Copilot-billed credit or dollar figure is claimed.

Using the supplied global GPT-5.4 rates for requests below 272K context
($2.50/M input, $0.25/M cached input, and $15/M output), the measured token
volumes have the following API-price equivalents:

| Assumed cached input | Hybrid | Cloud only | Saved |
|---:|---:|---:|---:|
| 0% | $2.217 | $3.727 | $1.510 (40.5%) |
| 50% | $1.267 | $2.132 | $0.865 (40.6%) |
| 75% | $0.791 | $1.334 | $0.543 (40.7%) |
| 90% | $0.506 | $0.856 | $0.350 (40.9%) |

LiteLLM did not receive cached-input token counts from the Copilot provider, so
the cache rows are scenarios rather than observed billing. These values are
direct-API price equivalents, not confirmed Copilot Enterprise charges. Against
the additional 1,063 seconds of wall time, the uncached saving values the delay
at about $5.11/hour; at 90% cache it falls to about $1.18/hour.

### Local electricity

Assuming the entire local system draws its 90 W peak continuously for all
1,130.48 seconds spent on the three local-routed cases gives a conservative
upper bound of 0.0283 kWh. Actual inference energy should be lower because the
wall interval includes file inspection, tool execution, and tests.

| Electricity price | Local energy cost | Net saving, uncached API equivalent | Net saving, 90% cached input |
|---:|---:|---:|---:|
| $0.10/kWh | $0.0028 | $1.507 | $0.347 |
| $0.20/kWh | $0.0057 | $1.504 | $0.344 |
| $0.30/kWh | $0.0085 | $1.501 | $0.341 |
| $0.40/kWh | $0.0113 | $1.498 | $0.339 |
| $0.50/kWh | $0.0141 | $1.495 | $0.336 |

Electricity would break even with the measured API-price saving only around
$53.41/kWh with uncached input, or $12.38/kWh under the 90%-cached scenario.
Thus electricity is negligible for this pilot. Latency, hardware amortization,
and the opportunity cost of occupying the local GPU are the meaningful costs.

The result argues against routing full Copilot agent loops to local Qwen solely
for cost. Local execution preserved quality in two of three routed tasks but
used long, repeated harness contexts and dominated latency. A better policy
would reserve local Qwen for bounded one-shot work or sharply cap local agent
turns before escalating the same task to cloud.

## GPT-5.6 Sol cloud-only comparison

The same six cases were also run in fresh worktrees with GPT-5.6 Sol forced for
every model call. Sol uses the Responses API; the GPT-5.4 reference uses Chat
Completions. Repository revisions, prompts, ten-minute caps, Python 3.9
environments, and official focused tests were unchanged.

| Instance | GPT-5.4 cloud only | GPT-5.6 Sol cloud only | Sol / GPT-5.4 |
|---|---:|---:|---:|
| `pylint-dev__pylint-4970` | Failed, 48.13 s | Failed, 66.04 s | 1.37x |
| `pylint-dev__pylint-6903` | Passed, 33.92 s | Passed, 50.46 s | 1.49x |
| `pytest-dev__pytest-7521` | Passed, 17.99 s | Passed, 47.88 s | 2.66x |
| `pytest-dev__pytest-7571` | Passed, 19.87 s | Passed, 35.10 s | 1.77x |
| `sphinx-doc__sphinx-10323` | Passed, 33.98 s | Passed, 108.06 s | 3.18x |
| `sphinx-doc__sphinx-9367` | Passed, 12.07 s | Passed, 26.00 s | 2.15x |

| Metric | GPT-5.4 | GPT-5.6 Sol | Sol difference |
|---|---:|---:|---:|
| Focused tests resolved | 5/6 | 5/6 | No change |
| Agent time | 165.96 s | 333.53 s | +101.0% |
| Model calls | 59 | 54 | -8.5% |
| Tool calls | 92 | 125 | +35.9% |
| Input tokens | 1,417,789 | 1,387,880 | -2.1% |
| Output tokens | 12,162 | 15,069 | +23.9% |
| Total tokens | 1,429,951 | 1,402,949 | -1.9% |

Both models missed `pylint-dev__pylint-4970`. Sol edited the correct
implementation file, but returned before suppressing the command's final
summary output, so the official assertion still failed. On this subset, Sol
did not improve focused correctness and was slower on every case. It used five
fewer model calls and slightly fewer total tokens, but performed 33 more tool
calls.

GitHub's published GPT-5.6 Sol promotional rates through September 3, 2026 are
$2.00/M uncached input tokens, $0.20/M cached input tokens, $2.50/M cache-write
tokens, and $10.00/M output tokens. At uncached rates, this run is equivalent to
$2.926, versus $3.727 for GPT-5.4: $0.800 or 21.5% lower. LiteLLM did not expose
cached-read or cache-write token counts, so no exact cached-cost claim is made.
The Sol promotion is explicitly temporary; its post-promotion comparison may
reverse even if token usage remains unchanged.

## GPT-5.6 Sol hybrid comparison

The Responses-based hybrid lane was run across all six cases rather than
reusing local results from the Chat Completions experiment. The latest scored
result for each case is:

| Instance | Route | Focused test | Elapsed | Model calls | Tools |
|---|---|---:|---:|---:|---:|
| `pylint-dev__pylint-4970` | Local | Failed | 600.51 s | 25 | 24 |
| `pylint-dev__pylint-6903` | Sol | Passed | 75.06 s | 11 | 20 |
| `pytest-dev__pytest-7521` | Sol | Passed | 61.89 s | 9 | 20 |
| `pytest-dev__pytest-7571` | Local | Passed | 600.04 s | 24 | 24 |
| `sphinx-doc__sphinx-10323` | Sol | Passed | 55.74 s | 8 | 16 |
| `sphinx-doc__sphinx-9367` | Local | Passed | 237.31 s | 9 | 9 |

Hybrid Sol resolved five of six focused tests: two of three local-routed cases
and all three Sol-routed cases. Total agent time was 1,630.54 seconds, 4.89x
the 333.53-second forced-Sol run and 1.33x the GPT-5.4 hybrid run. The local
lane averaged 479.28 seconds; the Sol lane averaged 64.23 seconds.

The hybrid made 28 Sol calls versus 54 for forced Sol, a 48.1% reduction. It
sent 647,598 tokens through Sol versus 1,402,949, a 53.8% reduction. At Sol's
current uncached promotional rates, those cloud tokens are equivalent to
$1.351 versus $2.926, saving $1.575 (53.8%). Compared with the GPT-5.4 hybrid,
the Sol hybrid used 22.2% fewer cloud calls, 24.0% fewer cloud tokens, and had
a 39.0% lower uncached cloud-price equivalent, but took 32.7% longer overall.

The first hybrid-Sol pass resolved only four cases. After 15 successful local
turns on `pytest-dev__pytest-7571`, vLLM 0.27.1's Responses input converter
received a typed `ResponseCustomToolCall` where it assumed a mapping and raised
`AttributeError`. The hybrid alias did not initially have its own fallback, so
Copilot CLI exhausted five retries and exited. The proxy now gives both hybrid
aliases direct cloud fallbacks. A deterministic custom-tool probe reproduced
the local 500 and completed through Sol with the expected answer.

The reported five-of-six hybrid result uses a targeted rerun of that one
infrastructure-failed case. The rerun stayed local, reached the ten-minute cap,
and left the correct two-line implementation patch; the official focused test
passed. It is therefore a corrected latest-result comparison, not a second
clean six-case run under the final fallback configuration. The harness now also
waits for late proxy callbacks and records whether route telemetry is complete,
preventing a timed-out request from being attributed to the next case.
