# Native Qwen3.8 / DFlash2 validation — 2026-09-10

## Outcome

**The real NVFP4 target and BF16 DFlash2 draft load and generate natively on
Windows ARM64.** No vLLM source changes, wheel rebuild, WSL, FlashInfer, or
CuTeDSL were needed for these runs.

Recommended conservative configuration: **target-only, one request at a time,
BF16 KV cache, CUTLASS NVFP4 linear kernels, Triton attention/GDN prefill,
decode-only CUDA graphs, torch.compile disabled**. This passed the tested 32K
retrieval canary and preserved the eager baseline's tokens at batch size 1.

DFlash2 is substantially faster on this small workload, but **not qualified as
lossless relative to the non-speculative baseline**. An open-ended response
differs reproducibly starting at output token 13. A target-only probe also
reproduces that next-token change when the same prefix is re-prefilled instead
of decoded incrementally. The exact responsible numerical kernel is not isolated.

This is a local compatibility and regression evaluation, not a model-accuracy
benchmark or production certification. No server is left running.

## Stack and checkpoint provenance

- GPU: NVIDIA RTX Spark N1X, Blackwell SM121, 32,704 MiB physical GPU partition;
  driver 616.30, Windows ARM64.
- Installed vLLM: `0.28.0+cu134.pr3.1`, built from actual upstream 0.28 plus PR #3
  Windows ARM64 integration in the sibling vLLM checkout.
- CPython 3.13.10; custom Torch `2.11.0a0+git70d99e9`, CUDA 13.4;
  `triton-windows 3.8.0+git461876e8`; Transformers 5.15.1.
- Runtime: sibling vLLM checkout's isolated WOA prefix and scoped PowerShell
  Python helper. The installed wheel was used, not an unbuilt source import.
- Upstream vLLM expects Torch 2.13. These results qualify only this custom stack.
- Model directories were absent; pinned checkpoints were downloaded to the
  DevDrive. No remote model code was enabled.

| Checkpoint | Hugging Face revision | Local directory |
|---|---|---|
| `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` | `5b7a687fc8211a5d631c8ca6a593dd37eb26ce33` | X:/models/Qwen3.8-27B-NVFP4-RTX5090 |
| `incoai/Qwen3.8-27B-DFlash2` | `dedf8df68adfb1afeaf7b7480c0a0243108177b4` | X:/models/Qwen3.8-27B-DFlash2 |

The target's actual config uses `Qwen3_5ForConditionalGeneration`: 64 layers,
48 linear/GDN attention and 16 full-attention layers. It is dense, not MoE, but
it is **not** an ordinary attention-only Qwen3 model. Text-only mode skips vision.
The draft has five BF16 layers, noncausal sliding attention, block size eight,
and seven speculative tokens.

Target shard CRC32 checks matched the published manifest:

- Shard 1: 9,972,777,720 bytes, `617cc98f`.
- Shard 2: 7,943,334,864 bytes, `e1291b6b`.

Draft: 3,848,817,896 bytes; SHA256 matched pinned Hugging Face LFS metadata:
`67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c`.

Do not compare these timings directly with older WSL/5090 reports: hardware,
engine, kernel choices, prompts, checkpoint revision, and cache settings differ.

## Method

[benchmark_qwen38_native.py](benchmark_qwen38_native.py) runs each configuration
in a fresh process, sequentially on the GPU. It records package versions,
configuration/template/prompt hashes, every output token, text, latency,
selected-token log-probability finiteness, repeat equality, and engine counters.

- V2 model runner explicitly enabled; BF16 activations and KV cache.
- Three timed repetitions after a two-token warmup for each prompt shape.
- Identical checkpoint chat template, thinking disabled, greedy temperature zero.
- Prefix cache disabled; every recorded request reports zero cached tokens.
- Explicit 3 GiB KV allocation for matched baseline/DFlash2 comparisons,
  1,024-token prefill chunks, batch size 1, no CPU offload.
- Five canaries: multiplication, sorted JSON array, exact JSON record extraction,
  strict XML/JSON tool envelope and all five arguments at 2K and 8K context.
- One open-ended coding tutorial prompt, fixed 192 output tokens with EOS ignored,
  for sustained decode and cross-configuration token comparison. It is not a
  code-execution or code-quality test.
- Longer contexts use deterministic repeated background text, with the record
  at the beginning. This is a retrieval canary, not diverse long-context QA.
- Timings are offline `LLM.generate` measurements, not HTTP serving latency.
- Performance is reported only as relative percentage changes between options:
  `100 * (candidate / reference - 1)`, calculated from the unrounded saved
  measurements. Each comparison names its reference. Positive throughput changes
  mean higher throughput; negative time changes mean lower latency. With
  speculation, tokens arrive in bursts, so wall time is the primary end-to-end
  comparison. Comparisons use medians, not confidence intervals or peak rates.
- No engine preemptions were recorded in the matched graph and batch tests.

[compare_qwen38_native.py](compare_qwen38_native.py) rejects mismatched inputs,
compares candidate outputs with baseline repeat zero, and refuses to qualify a
speedup as token-preserving when outputs differ. Acceptance uses before/after
counter differences to exclude warmup.

## Single-request performance

Same 3 GiB cache, 49-token coding prompt, 192 generated tokens, three runs:

Both percentage columns use **target-only eager** as the reference.

| Mode | Median decode throughput change | Median request wall-time change | GPU memory after tests |
|---|---:|---:|---:|
| Target-only eager (reference) | 0.00% | 0.00% | 20,620 MiB |
| Target-only decode graphs | +30.19% | -23.04% | 20,724 MiB |
| DFlash2 eager | +230.34% | -68.64% | 24,721 MiB |
| DFlash2 decode graphs, capture 8 tokens | +318.73% | -75.17% | 24,762 MiB |

Actual graph capture was logged: approximately 0.05 GiB for the target-only
graph and 0.07 GiB for the DFlash2 run. These are model-runner capture deltas,
not total process memory or a claim that every operation is graphed.

**DFlash2 graph versus target-only graph:** median decode throughput increased
by **221.63%** and median request wall time decreased by **67.74%** on the coding
prompt, but the generated token sequences differ. This is not a lossless speedup
claim. DFlash2 acceptance across timed requests was
1,281 / 1,911 draft tokens = **67.03%**, over 273 draft rounds; mean accepted
length including the bonus token was about 5.69.

Selected canaries, with identical output tokens between these two modes.
Wall-time changes compare **DFlash2 graphs against target-only graphs**:

| Case | Prompt tokens | Output tokens | Median request wall-time change |
|---|---:|---:|---:|
| Arithmetic | 43 | 4 | -4.53% |
| JSON sorting | 48 | 16 | -68.65% |
| Exact record extraction | 118 | 72 | -79.48% |
| 2K tool call | 2,048 | 115 | -70.48% |
| 8K tool call | 8,193 | 115 | -43.40% |

DFlash2 does not improve every latency metric: compared with target-only graphs,
its median time to first token increased by **2.83%** for the 8K prompt and
**16.50%** for the short coding prompt. The gain is predominantly generation
throughput.

Target weights occupied 16.19 GiB per worker allocation logging; target plus
draft occupied 19.99 GiB. In the matched graph runs, DFlash2 warm-cache
initialization took **16.64% longer** than the target-only baseline. Initial cold
target startup included first-use Triton compilation. Larger/batched new shapes
can take longer.

**Memory measurement caveat:** CUDA `mem_get_info()` in the parent process on
this WDDM stack reported about 46.55 GiB, not the physical partition. Use the
recorded device-wide `nvidia-smi` readings for physical memory. The table shows
post-test samples, **not sampled peak usage**.

## Correctness findings

1. All **15/15 canaries per matched configuration** passed. Selected-token
   log probabilities were finite. All six cases repeated exactly within each
   configuration, including the open-ended coding response.
2. Target-only graph outputs matched target-only eager outputs token-for-token.
3. DFlash2 graph outputs matched DFlash2 eager outputs token-for-token.
4. DFlash2 matched the target baseline on all five canaries, but **not** the
   coding response. The first difference is index 12 (the 13th output token):
   baseline token 32, `A`; DFlash2 token 1919, `This`.
5. [inspect_qwen38_native_divergence.py](inspect_qwen38_native_divergence.py)
   reproduced the same flip with **only the target loaded**, twice:

   | Target execution | `A` log probability | `This` log probability | Selected |
   |---|---:|---:|---|
   | Incremental autoregressive decode | -0.601175 | -0.851175 | `A` |
   | Re-prefill identical accepted prefix | -0.836194 | -0.586194 | `This` |

   This establishes target execution-path sensitivity independently of the
   drafter. It does not identify whether the difference originates in NVFP4
   GEMM, hybrid-state accumulation, attention, or another operation. Do not
   describe it as a fixed bug, harmless rounding, or proof of a faulty sampler.

6. **32K target-only graph test:** two exact retrieval passes at 32,768 prompt
  tokens; identical 115-token tool calls. Other short canaries in that run passed
  too. No matched DFlash2 run at this context was performed, so no between-option
  performance comparison is reported. Not a maximum-capacity test.
7. **Two-request target-only graph batch:** all 30 canaries passed and outputs
  repeated per slot. Compared with the single-request target-only graph run,
  median aggregate end-to-end output throughput increased by **98.45%**, while
  median batch wall time increased by **0.78%**. Both throughput measurements
  include prefill; this is distinct from the decode-only metric above. Both
  requests used identical prompts. Coding output differed from batch size 1;
  do not promise batch-invariant greedy outputs.

## Numerical and harness regression coverage

- 12 installed-wheel CUTLASS NVFP4 GEMM cases against dequantized PyTorch matmul
  references, FP16/BF16 including padded shapes; upstream tolerance 0.1 absolute
  and relative. This checks implementation against quantized values, **not**
  quantization error versus original BF16 model weights.
- Seven packed recurrent GDN decode tests against the reference path, including
  strided inputs, state updates and padding. Upstream tolerances retained.
- **19 CPU-only harness tests** cover strict envelopes/values, token differences,
  flat tokenizer IDs, input mismatch guards, repeat drift, and warmup-excluded
  acceptance counters.
- The earlier 55 native port/kernel/DFlash2 unit regressions were also rerun in
  final verification. The focused native regression total is **74**, not the
  entire upstream suite.

## Resolved setup issues and excluded attempts

- DFlash2 loaded both models with 1 GiB KV but rejected startup: 8,449 maximum
  sequence length required 2.06 GiB for this hybrid/draft layout. Explicit 3 GiB
  passed and provided 12,283 shared token capacity in that run. Do not reuse
  target-only capacity estimates for DFlash2.
- V2 graph capture sizes count **tokens**, not requests. Capture size `[1]`
  with seven draft tokens silently left no eligible verification graph. The
  corrected size `[8]` actually captured and increased median decode throughput
  by **26.76% versus DFlash2 eager**. The initial graph-requested run is retained
  but excluded as a graph-acceleration result.
- One eager DFlash2 attempt received SIGINT during repetition two. Its partial
  result remains `testing`; it is excluded. A fresh three-repeat run completed.
- Transformers 5 chat-template tokenization can return a mapping. The harness
  explicitly renders text then encodes it to flat IDs; prompt sizes are checked.
- Transformers printed nonfatal missing-docstring diagnostics for video
  processor arguments even in text-only import. No dependency code was patched
  merely to hide these diagnostics. Actual inference completed.
- No driver, system memory, security, WSL, or global Python changes were made.

## Reproduce locally

Run sequentially using the existing native runtime helper. The paths below are
machine-specific and require the previously provisioned native stack.

```powershell
$workspace = 'X:\GitHub\local-hybrid-agent'
$python = 'X:\GitHub\vllm-windows\tools\run-woa-python.ps1'

& $python "$workspace\benchmark_qwen38_native.py" `
  --kv-gib 3 --graph FULL_DECODE_ONLY --contexts '2048,8192' `
  --output-tokens 192 --repeats 3 `
  --results "$workspace\results\native-v028\baseline-rerun.json"

& $python "$workspace\benchmark_qwen38_native.py" `
  --mode dflash --kv-gib 3 --graph FULL_DECODE_ONLY --contexts '2048,8192' `
  --output-tokens 192 --repeats 3 `
  --results "$workspace\results\native-v028\dflash-rerun.json"

& $python "$workspace\compare_qwen38_native.py" `
  "$workspace\results\native-v028\baseline-rerun.json" `
  "$workspace\results\native-v028\dflash-rerun.json" `
  --output "$workspace\results\native-v028\comparison-rerun.json"

& $python "$workspace\test_benchmark_qwen38_native.py" -v
```

Use `--graph NONE` for eager mode. For the baseline-only extended tests, use
`--contexts '32768' --repeats 2` or `--batch-size 2`. Do not assume those limits
are qualified with DFlash2. The helper sets spawn and disables FlashInfer
sampling, scopes DLL search paths, and restores its environment after exit.

## Raw evidence

Generated measurements are retained locally under the ignored results directory;
the scripts and this report are source-controlled candidates, not committed.

- [Matched eager baseline](results/native-v028/baseline-eager-3g.json)
- [Matched eager DFlash2](results/native-v028/dflash-eager-3g-complete.json)
- [Target-only graph baseline](results/native-v028/baseline-graph-3g.json)
- [Correctly captured DFlash2 graphs](results/native-v028/dflash-graph8-3g.json)
- [Graph baseline versus DFlash2 comparison](results/native-v028/comparison-graph8.json)
- [Eager versus graph baseline equality](results/native-v028/comparison-baseline-graph.json)
- [Target-only next-token diagnostic](results/native-v028/target-divergence-probe.json)
- [32K retrieval results](results/native-v028/baseline-graph-32k.json)
- [Two-request batch results](results/native-v028/baseline-graph-batch2.json)

Full worker logs are retained under the sibling native checkout's build
directory, with the `native-qwen-` prefix. The native source merge remains
uncommitted; the previously archived `pr3.1` wheel was not changed.

## Not qualified

Full-precision reference accuracy/perplexity, sampled-distribution equivalence,
general code quality, representative long-context datasets, multimodal inputs,
FP8 KV, prefix caching, contexts beyond the tested 32K baseline / 8K draft,
DFlash2 concurrency, torch.compile, and HTTP server throughput are untested.
Exact greedy invariance across prefill/decode paths, speculation and batch sizes
**failed** the targeted comparison and remains an explicit limitation.