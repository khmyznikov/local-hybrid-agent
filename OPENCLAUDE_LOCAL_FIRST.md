# OpenClaude local-first through LiteLLM

This workflow keeps GPT-5.4 in charge of dispatch and failure recovery while
moving repository inspection, implementation, and focused tests to local Qwen.
Both providers are pinned behind LiteLLM for one credential, telemetry, and
transport layer.

```text
Issue
  -> GPT-5.4 dispatcher (maximum four parent turns)
  -> one local Qwen implementation child (maximum 12 tool uses)
  -> external official SWE-bench FAIL_TO_PASS scorer
  -> GPT-5.4 repair only when scoring fails (maximum eight turns)
  -> external scorer again
```

The dynamic `hybrid-copilot` complexity router is not used. OpenClaude owns
workflow and escalation. LiteLLM routes explicit aliases:

- `copilot-gpt-5.4` to GitHub Copilot GPT-5.4.
- `qwen-local` to `qwen3.8-27b-local` on `http://127.0.0.1:8001/v1`.

## Validated result

Six selected SWE-bench Verified cases were run in WSL with official
FAIL_TO_PASS tests. This is focused scoring, not canonical Docker/PASS_TO_PASS
scoring.

| Matching policy | Focused tests | Cloud calls | Cloud tokens | Local calls | Local tokens |
|---|---:|---:|---:|---:|---:|
| GPT-5.4 implementer + selective repair | 6/6 | 90 | 897,360 | 0 | 0 |
| Qwen implementer + selective GPT repair | 6/6 | 46 | 384,619 | 66 | 1,078,818 |

Local-first preserved 6/6 focused correctness and reduced cloud tokens by
57.1% under equal orchestration. Including a conservative 90W local-system
energy bound at $0.20/kWh, estimated cost fell by 58.0-61.0% across 0-90%
assumed cloud input caching.

Three local patches failed their first external score and all three were
recovered by bounded GPT-5.4 repair. Two cloud-implementer patches also required
repair. No LiteLLM route failed.

### Per-case result

| Instance | Cloud policy | Local first | Cloud repair | Local repair | Cloud tokens | Local-first cloud tokens |
|---|---:|---:|---:|---:|---:|---:|
| `pytest-dev__pytest-7571` | Pass | Pass | No | No | 146,553 | 9,591 |
| `pytest-dev__pytest-7521` | Pass | Pass | No | No | 209,862 | 163,067 |
| `sphinx-doc__sphinx-9367` | Pass | Pass | No | No | 41,911 | 9,454 |
| `sphinx-doc__sphinx-10323` | Pass | Pass | Yes | Yes | 209,260 | 71,507 |
| `pylint-dev__pylint-4970` | Pass | Pass | Yes | Yes | 232,316 | 73,950 |
| `pylint-dev__pylint-6903` | Pass | Pass | No | Yes | 57,458 | 57,050 |

`pytest-7521` generated an anomalous 22 GPT-5.4 calls in the local-first arm,
limiting its saving to 22.3%. That overhead is retained in the aggregate rather
than removed as an outlier.

## Prerequisites

The validated host uses:

- Windows 11 ARM64 with WSL2 Ubuntu ARM64.
- NVIDIA RTX Spark N1X and the vLLM environment described in
  [wsl_qwen38_setup_and_stress.md](wsl_qwen38_setup_and_stress.md).
- `uv` at `~/.local/bin/uv`.
- Git and `curl` in WSL.
- Access to GitHub Copilot GPT-5.4.

The scripts use overridable defaults:

| Variable | Default |
|---|---|
| `LOCAL_HYBRID_AGENT_ROOT` | Directory containing this repository |
| `LITELLM_HOME` | `~/litellm-hybrid` |
| `LOCAL_HYBRID_STATE_HOME` | `~/.local/share/local-hybrid-agent` |
| `LOCAL_HYBRID_SWEBENCH_HOME` | `$LITELLM_HOME/swebench` |
| `OPENCLAUDE_ROOT` | `$LOCAL_HYBRID_STATE_HOME/openclaude` |
| `LITELLM_URL` | `http://127.0.0.1:4000/v1` |
| `LITELLM_MASTER_KEY` | `hybrid-copilot` |
| `LITELLM_PARENT_ALIAS` | `copilot-gpt-5.4` |
| `LITELLM_LOCAL_ALIAS` | `qwen-local` |

The WSL mount path must point to this clone. With the documented Windows path,
it is `/mnt/c/Dev/local-hybrid-agent`.

## Install

### 1. Prepare local vLLM

Follow [wsl_qwen38_setup_and_stress.md](wsl_qwen38_setup_and_stress.md) once.
The default launcher expects:

- vLLM environment: `~/vllm-qwen38-wsl/.venv`
- model: `~/models/Qwen3.8-27B-NVFP4-RTX5090`

The PowerShell server launcher currently contains machine-specific defaults.
Adjust those paths in [start_qwen38_copilot_server.ps1](start_qwen38_copilot_server.ps1)
for a different machine.

### 2. Install LiteLLM and SWE-bench data

From WSL:

```bash
cd /mnt/c/Dev/local-hybrid-agent
bash setup_litellm_hybrid_wsl.sh
```

This creates `~/litellm-hybrid/.venv`, installs pinned LiteLLM, installs
Copilot CLI if absent, and downloads the SWE-bench Verified parquet file.

### 3. Authorize LiteLLM for GitHub Copilot

Run once from WSL:

```bash
~/litellm-hybrid/.venv/bin/python \
  /mnt/c/Dev/local-hybrid-agent/authenticate_litellm_copilot_wsl.py
```

Open the displayed URL, enter the device code, and approve the application.
OAuth files remain under `~/.config/litellm/github_copilot` with mode 600.
Never commit those files.

### 4. Install the pinned OpenClaude build

From WSL:

```bash
cd /mnt/c/Dev/local-hybrid-agent
bash setup_openclaude_local_first_wsl.sh
```

The setup script:

1. Installs Node 22 ARM64 and Bun 1.3.13 under user-local state when needed.
2. Clones OpenClaude into `~/.local/share/local-hybrid-agent/openclaude`.
3. Checks out commit `8db8830666bfa115bf6f9586525fae0e3585bf23`.
4. Applies [the compatibility patch](openclaude_harness/openclaude-0.29.1-local-first.patch).
5. Runs `bun install --frozen-lockfile` and builds `dist/cli.mjs`.

The patch is intentionally pinned. It provides:

- stable Responses tool-call correlation by `output_index`;
- named local-agent routing when strict schemas synthesize `model: inherit`;
- an opt-out from nested Agent worktrees because the SWE harness already uses
  isolated worktrees.

The script is idempotent for the expected patch and fails closed if the
OpenClaude checkout contains unrelated changes.

## Start services

From PowerShell in this repository:

```powershell
& .\start_qwen38_copilot_server.ps1 -Profile Quality
```

Then start LiteLLM in a dedicated WSL terminal and leave it running:

```bash
cd /mnt/c/Dev/local-hybrid-agent
bash start_litellm_hybrid_wsl.sh
```

Run the evaluation commands below from a second WSL terminal.

Confirm both endpoints if needed:

```bash
curl -fsS http://127.0.0.1:8001/health
curl -fsS -H 'Authorization: Bearer hybrid-copilot' \
  http://127.0.0.1:4000/health/liveliness
```

## Run the focused evaluation

### One command

Run both matching dispatch arms, selectively repair only failed patches, and
write the policy summary:

```bash
cd /mnt/c/Dev/local-hybrid-agent
bash run_openclaude_local_first_wsl.sh run --force
```

The six-case run is sequential because one local server is shared. It can take
well over 30 minutes; latency is not the optimization target.

### Staged commands

Use staged commands for inspection or recovery:

```bash
# Prepare worktrees and Python environments.
bash run_openclaude_local_first_wsl.sh prepare

# Confirm official tests fail at every base commit.
bash run_openclaude_local_first_wsl.sh baseline

# Run both cloud-implementer and local-implementer dispatch arms.
bash run_openclaude_local_first_wsl.sh dispatch --force

# Invoke GPT-5.4 repair only for failed external scores.
bash run_openclaude_local_first_wsl.sh repair

# Regenerate the aggregate report.
bash run_openclaude_local_first_wsl.sh summarize
```

Run one arm or case with repeatable selectors:

```bash
bash run_openclaude_local_first_wsl.sh dispatch \
  --arm hybrid \
  --case pytest-dev__pytest-7571 \
  --force

bash run_openclaude_local_first_wsl.sh repair \
  --arm hybrid \
  --case pytest-dev__pytest-7571
```

`control` uses GPT-5.4 for both dispatcher and implementation child. `hybrid`
uses GPT-5.4 for dispatch and repair and Qwen for the implementation child.
The control is required for attribution, not as the local-inference solution.

## Outputs

Generated files are under:

```text
~/litellm-hybrid/swebench/results/openclaude-local-first/
```

Important artifacts:

- `results.jsonl`: one dispatch result per arm/case.
- `repairs.jsonl`: selective repair results.
- `policy-summary.json`: latest aggregate quality, token, route, and cost data.
- `<arm>/<case>/agent.patch`: dispatch-stage implementation patch.
- `<arm>/<case>/repaired.patch`: final repaired patch when repair ran.
- `<arm>/<case>/focused-test.log`: first external score.
- `<arm>/<case>/repair-focused-test.log`: score after repair.
- `<arm>/<case>/stream.jsonl`: OpenClaude dispatch stream.
- `<arm>/<case>/litellm-routes.jsonl`: immutable dispatch route slice.
- `<arm>/<case>/repair-litellm-routes.jsonl`: immutable repair route slice.

The LiteLLM observer records model names, duration, and token counts. It does
not record prompts or response text. Generated artifacts are outside this Git
checkout by default.

The committed
[validated_result.json](openclaude_harness/validated_result.json) captures the
reference six-case result in machine-readable form.

## Cost interpretation

LiteLLM currently reports total Copilot prompt tokens without a cached-input
breakdown. `policy-summary.json` therefore gives equal-cache scenarios at 0%,
50%, 75%, and 90% for both policies. Prices are:

- $2.50/M uncached input tokens.
- $0.25/M cached input tokens.
- $15/M output tokens.
- $0.20/kWh local electricity assumption.
- 90W continuous-system-draw upper bound during local request time.

These are API-price equivalents, not confirmed Copilot Enterprise charges.

## Run on another case set

Edit [openclaude_harness/swebench_cases.json](openclaude_harness/swebench_cases.json)
and list SWE-bench Verified instance IDs. Re-run `prepare`, `baseline`, and
`run`. Historical Python packages may need additional compatibility pins in
[hybrid_swe_experiment.py](hybrid_swe_experiment.py).

## Stop services

Stop LiteLLM from WSL:

```bash
bash /mnt/c/Dev/local-hybrid-agent/stop_litellm_hybrid_wsl.sh
```

Stop Qwen from PowerShell:

```powershell
& .\stop_qwen38_copilot_server.ps1
```

## Limitations

- Six selected cases are too few for a population quality estimate.
- Scoring runs official FAIL_TO_PASS nodes in WSL, not per-case Docker images or
  the full PASS_TO_PASS suite.
- Independent GPT/Qwen runs can take different paths.
- A trusted external verifier is essential; without it, failed local patches
  would not trigger repair reliably.
- The local model can exhaust its tool budget after diagnosis without editing.
  Selective cloud repair recovered those cases in the measured run.
- Do not enable LiteLLM's dynamic router for this workflow. Explicit aliases
  keep GPT dispatch/repair and Qwen implementation ownership deterministic.
- Do not send unrelated traffic through the same LiteLLM proxy during an
  evaluation. Per-case route attribution uses byte offsets in the shared route
  log and assumes exclusive benchmark use.
