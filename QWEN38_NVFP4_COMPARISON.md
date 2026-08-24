# Qwen3.8-27B NVFP4 checkpoint comparison

This comparison evaluates two Qwen3.8-27B NVFP4 checkpoints on the validated
RTX Spark N1X WSL runtime:

- `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`
- `unsloth/Qwen3.8-27B-NVFP4`

The goal was not to rank general intelligence from one prompt. It was to test a
representative long-context failure mode: retrieving exact operational literals
and emitting a structurally strict tool call.

## Method

Both runs used:

- vLLM 0.27.1 on SM12.1
- FP8 KV cache
- Pinned FlashInfer full attention
- Triton/FLA Gated DeltaNet prefill
- `FULL_DECODE_ONLY` CUDA graphs, batch size 1
- 4,096 scheduler tokens
- Greedy generation with thinking disabled
- Checkpoint-specific chat templates
- 40K, 96K, and 240K rendered prompt tiers
- 128 maximum output tokens

The canary required one `execute_cli` call containing five exact literals. A
pass required the correct function and arguments, opening and closing tool tags,
and no prose or Markdown fences outside the XML tool envelope. The harness
recorded rendered-template hashes, prompt hashes, output token IDs, exact output
hashes, and elapsed time.

The cache allocation could not be identical because of checkpoint size:

- Gittensor fit an 11 GiB FP8 cache.
- Unsloth OOMed with 8.25 GiB during the first 40K prefill. A 7.75 GiB cache was
  stable and still exposed enough slots for the 240K request.

This difference is part of the result, not a tuning advantage granted to either
checkpoint.

## Results

| Prompt tier | Gittensor elapsed | Unsloth elapsed | Result |
|---:|---:|---:|---|
| 40K | 39.05 s | 45.24 s | Both passed; Gittensor 13.7% lower elapsed time |
| 96K | 109.61 s | 121.08 s | Both passed; Gittensor 9.5% lower elapsed time |
| 240K | 423.17 s | 442.84 s | Both passed; Gittensor 4.4% lower elapsed time |

Despite different checkpoint templates, both models produced byte-identical
output text and identical output token IDs at every tier. The Unsloth 96K prompt
rendered one token shorter; the other prompt lengths matched exactly.

## Memory and cache

| Measurement | Gittensor | Unsloth |
|---|---:|---:|
| Safetensors size | 17.48 GiB | 21.81 GiB |
| Resident model memory | 16.19 GiB | 20.47 GiB |
| Stable FP8 cache allocation | 11 GiB | 7.75 GiB |
| Exposed cache slots in this run | 350,343 | 246,311 |

Unsloth used 26.4% more resident model memory and exposed 29.7% fewer cache
slots in the stable configuration. Gittensor therefore leaves substantially
more room for native-context headroom and concurrent requests.

## Numerical configuration

The checkpoints take materially different quantization paths:

- Gittensor uses ModelOpt NVFP4 and the FlashInfer CUTLASS NVFP4 linear kernel.
- Unsloth uses a mixed compressed-tensors recipe: CUTLASS FP8 linears for its
  W8A8 targets and FlashInfer CUTLASS NVFP4 for its FP4 targets.
- Unsloth contains 32 calibrated `k_scale`/`v_scale` tensors, one pair for each
  full-attention layer. vLLM loaded them without k/v fallback warnings.
- Gittensor contains no q/k/v/prob attention-scale tensors. With FP8 KV, vLLM
  substitutes scale 1.0 and warns that q/prob are uncalibrated.

The Unsloth safetensors do not contain standalone `q_scale` or `prob_scale`
keys, but its compressed-tensors runtime path emitted no missing-scale or
uncalibrated-scale warnings. That behavior should not be assumed equivalent to
ModelOpt's explicitly warned scale-1 fallback without a broader logit-level
comparison.

## Recommendation

Keep Gittensor as the local sidekick checkpoint on this 32.7 GiB partition:

- It was faster at all tested context tiers.
- It uses substantially less model memory.
- It provides more cache capacity and concurrency headroom.
- It produced the same exact tool output as Unsloth in this canary.
- The default BF16-KV quality profile avoids its missing FP8 KV-scale path.

Use Unsloth with calibrated FP8 KV when preserving k/v calibration is more
important than memory, elapsed time, and concurrency. It is a useful candidate
for broader quality evaluation, but this single literal-copy canary does not
establish generally better reasoning or tool use.

All results are machine- and runtime-specific. Re-run the fidelity harness after
changing the checkpoint, template, attention backend, KV dtype, CUDA/PyTorch,
vLLM, FlashInfer, or GPU architecture.