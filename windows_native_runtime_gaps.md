# Native Windows ARM64 vLLM: What Is Still Missing?

Date: 2026-08-18

## Short Answer

Native text generation works. The custom stack can load and run large NVFP4
models without x64 emulation, CPU offload, or dequantizing the model weights.

What is missing is mostly the surrounding acceleration and packaging ecosystem:

1. There is no complete set of public Windows ARM64 wheels for PyTorch CUDA,
   Triton, vLLM, and their optional acceleration dependencies.
2. FlashInfer's Python wheel can be installed, but its useful CUDA paths still
   depend on missing or x64-only native components.
3. NVIDIA CUTLASS DSL/CuTeDSL has no complete Windows ARM64 runtime, blocking
   FA4 and the Blackwell FlashInfer GDN path.
4. Some optional Triton kernels are absent or packaged under a different Python
   namespace, causing noisy import failures and fallback behavior.
5. Large single-file safetensors checkpoints are unreliable on this UMA host;
   standard smaller shards are required.
6. Several multimedia and specialized-kernel libraries still have no native
   Windows ARM64 build.

The current result is a functional native core with good fallbacks, not yet a
one-command, fully accelerated public distribution.

## What Already Works

These components are not missing:

- CPython 3.13 ARM64
- PyTorch CUDA on ARM64, using the locally built PyTorch wheel
- CUDA 13.4 and SM12.1 execution
- Triton on Windows ARM64, using the NVIDIA-only Triton Windows fork
- Native vLLM ARM64 wheel and Rust frontend
- Native FA2 with SM120 SASS
- CUTLASS FP8 and NVFP4 linear kernels
- compressed-tensors mixed FP8/NVFP4 checkpoints
- Triton full attention
- Triton/FLA Gated DeltaNet prefill and recurrent decode
- FP8 K/V cache and hybrid GDN state cache
- Qwen3.8 one-step MTP speculative decoding
- Experimental hybrid prefix caching in Mamba `align` mode

This is enough to run Gemma 4 and Qwen3.8 NVFP4 end to end.

## Missing Public Core Wheels

### PyTorch family

There is no official public PyTorch CUDA wheel for native Windows ARM64 in the
configuration used here. The stack relies on locally built ARM64 wheels for
PyTorch and related packages.

Consequences:

- A normal `pip install vllm` cannot reproduce this environment.
- Dependency resolvers may try to replace the custom build with an x64 or
  unsupported package.
- Torch, torchvision, and torchaudio versions must be kept together manually.

Needed:

- Published and tested Windows ARM64 CUDA wheels for the PyTorch family.

### Triton

Upstream Triton does not provide the required native Windows ARM64 package.
This environment uses `triton-windows 3.8.0+git461876e8` from a custom fork.

Needed:

- A maintained Windows ARM64 Triton wheel with SM12x support.
- CI that covers JIT compilation, tensor descriptors, scratch allocation, and
  Torch compilation integration on Windows ARM64.

### vLLM

The custom vLLM wheel is native and works, but it is not backed by a complete
public dependency set. Installation currently requires a known-good local
wheelhouse and `--no-deps` for selected packages.

Needed:

- A published Windows ARM64 wheel.
- ARM64-aware dependency markers and optional dependency groups.
- A tested lock or wheelhouse that does not resolve x64 packages.

## FlashInfer Is Not Really Pure Python

`flashinfer-python` is tagged as a platform-independent Python wheel, but it is
only the control layer. Actual operation requires CUDA JIT compilation, native
CUDA Python bindings, CuTeDSL, precompiled cubins, or a JIT cache.

### What was tested

The SystemPanic FlashInfer Windows core wheel was installed in an isolated
environment. Two packaging bugs were bypassed locally:

1. Its Windows CUDA loader preferred `CUDA\bin\x64` whenever that directory
   existed. On ARM64 this loaded an x64 `cudart64_13.dll` and failed with
   `WinError 193`.
2. Its generated `nvcc -ccbin` command did not quote the ARM64 MSVC path, so
   spaces were interpreted as extra input files.

After correcting those two issues, Python and `flashinfer.gdn_prefill` imported.
The GDN kernel still could not build.

### Native dependencies that are missing

- `cuda-bindings` has Windows x64 wheels, but no Windows ARM64 wheel.
- `cuda-tile` has Windows x64 wheels, but no Windows ARM64 wheel.
- The available SystemPanic FlashInfer JIT-cache wheel is `win_amd64`.
- Official FlashInfer documentation lists Linux as the supported OS.
- The generic cubin wheel does not replace the tested GDN JIT path.

### Kernel/backend mismatch

The tested FlashInfer release supports Blackwell GDN through CuTeDSL only for
the SM100/SM103 path. SM12.1 did not select that path. It fell back to the SM90a
C++/CUDA JIT kernel, which native Windows ARM64 NVCC rejected with:

```text
nvcc fatal: Host compiler targets unsupported OS.
```

Consequences:

- FlashInfer attention, sampler, and GDN acceleration are disabled.
- vLLM uses its native sampler, Triton attention, and in-tree Triton/FLA GDN.
- The fallback is correct but leaves performance on the table.

Needed:

1. Publish Windows ARM64 `cuda-bindings` and `cuda-tile` wheels.
2. Fix FlashInfer's Windows architecture and compiler-path detection.
3. Add and test SM120/SM121 GDN backend selection.
4. Publish a native Windows ARM64 JIT-cache or architecture-neutral GDN cubins.
5. Validate all FlashInfer imports so optional communication modules do not
   load x64 DLLs on ARM64.

### Same-hardware acceleration opportunity

The official Linux ARM64 stack was tested under WSL2 on the same machine as a
proxy for what native Windows could gain once equivalent dependencies and
runtime support exist. These percentages compare adjacent matched profiles;
they are opportunity estimates, not a guarantee that a future Windows port
will reproduce every result.

- Replacing Triton full attention with FlashInfer full attention improved 8K
  prefill by approximately 23%. Decode improved by only about 2% because Qwen
  has 16 full-attention layers while its other 48 layers use recurrent GDN.
- The FlashInfer sampler showed no repeatable benefit over vLLM's native
  sampler. Porting it is therefore lower priority for this model.
- FlashInfer FP4 linear selected successfully on Linux. A model-free serial JIT
   build then produced and loaded the shape-generic SM121 release module within
   the existing 10 GiB WSL allocation. Minimum available RAM was approximately
   1.4 GiB with no swap use. A standalone quantize-and-GEMM smoke test returned
   finite output. In repeated end-to-end eager tests, median prefill regressed by
   approximately 7%, median decode improved by approximately 5%, and total
   request latency was unchanged. A compiled CUDA-graph profile showed unstable
   sustained replay. Native CUTLASS therefore remains the preferred linear
   backend for this checkpoint; a Windows FP4 JIT cache is not currently a
   demonstrated performance priority.
- Direct FlashInfer GDN probes on SM12.1 produced non-finite output/state. GDN
  must remain correctness-blocked until the SM121 path is validated against
  Triton/FLA; availability alone is not enough.

FlashInfer's demonstrated value for this checkpoint is currently concentrated
in prompt processing. The larger decode opportunity comes from stable Torch
compilation, CUDA graph replay, and MTP integration described below.

## CuTeDSL / CUTLASS DSL

The top-level `nvidia-cutlass-dsl` Python package may look portable, but its
runtime library packages are not available for Windows ARM64. The published
runtime artifacts are Linux wheels.

This blocks:

- Standard FA4 on SM12x
- FlashInfer's Blackwell CuTeDSL GDN backend
- QuACK and other CuTeDSL-based kernels

Current fallback:

- FA2 for supported attention shapes
- Triton attention
- Triton/FLA GDN

Needed:

- Windows ARM64 runtime wheels for CUTLASS DSL and CUDA Tile.
- Windows loader, compiler, and cache-path support.
- SM12.1 validation for FA4 and GDN kernels.

## Triton Kernels Packaging Gap

The tested optimized vLLM wheel contains 47 files under
`vllm.third_party.triton_kernels`, but it does not contain `matmul_ogs.py`.
The newer contribution wheel contains `matmul_ogs` files, but they are still
vendored under `vllm.third_party.triton_kernels`; they must be exposed through
vLLM's top-level `triton_kernels` alias before any direct import.

The optimized runtime repeatedly logged:

```text
No module named 'triton_kernels.matmul_ogs'
```

Consequences:

- Optional Triton MXFP4/MoE paths are unavailable.
- Startup logs report errors even when the selected CUTLASS NVFP4 path works.
- Users can mistake a clean fallback for a fatal configuration failure.

Current fallback:

- CUTLASS FP8/NVFP4 for the tested models.

Needed:

- Package the complete matching Triton kernels source.
- Ensure `import_triton_kernels()` runs before all direct imports.
- Add a wheel-level import test for `triton_kernels.matmul_ogs`.
- Downgrade unavailable optional-backend messages from `ERROR` when a working
  fallback is selected.

## Triton Runtime Gaps

### Tensor Memory Accelerator scratch allocation

Enabling `FLA_USE_TMA=1` failed because the Triton kernel required runtime
scratch allocation and no allocator was registered:

```text
RuntimeError: Kernel requires a runtime memory allocation, but no allocator was set.
```

Needed:

- Register a Windows-compatible Triton scratch allocator before TMA launches.
- Validate tensor descriptors and TMA on SM12.1.

### Torch compilation integration

vLLM compilation without CUDA graphs completed, but Torch's Triton wrapper
emitted repeated:

```text
IndexError('Function argument index out of range')
```

The compiled Qwen run was slower for prefill than eager execution.

On the same hardware under Linux, compilation worked with the official Triton
and PyTorch stack. In a matched 8K test with sustained generation, compilation
combined with full decode-only CUDA graphs improved prefill by approximately
45% and decode by approximately 32% over eager execution. Cold compilation
added startup latency, but generated artifacts were reusable from cache.

Needed:

- Fix Triton kernel metadata/introspection under Torch compilation on Windows.
- Add Qwen GDN compile tests for 4K chunk shapes.

### CUDA graphs

Earlier Gemma testing showed that graph capture and one replay can succeed, but
multi-token replay stalls. Qwen was therefore tested with CUDA graphs disabled.

Linux full decode-only graph capture completed sustained Qwen generation and
improved decode by approximately 26% over the matched eager profile, with
negligible additional allocator reservation. This is the clearest measured
benefit currently blocked by the native Windows replay issue.

One-step MTP and full decode-only FlashInfer graphs were not additive: vLLM
rejected that combination because FlashInfer speculative decode only advertised
uniform single-token graph support and fell back to eager MTP.

One-step MTP by itself improved decode by approximately 58% in the synthetic
test. Combining MTP with compilation, but not CUDA graphs, reproduced a roughly
90-98% decode improvement. The deterministic prompt had 100% draft acceptance,
so this is an upper-bound result; natural prompts can accept fewer drafts and
receive a smaller gain. MTP also adds model memory and can reduce prefill, so it
remains a long-output profile rather than the general default.

Needed:

- Diagnose sustained replay around split attention, KV-cache updates, and
  recurrent-state mutation.
- Require multi-token generation tests, not capture-only tests.

## Model Loading and Safetensors

The 21.0 GiB Qwen `model.safetensors` file caused Python to exit while opening
the checkpoint, without a Python traceback. The same class of failure was seen
with large Gemma checkpoint mappings on this UMA machine.

Resharing Qwen into standard approximately 1 GiB safetensors files fixed
loading. One 2.37 GiB shard remained because a single tensor cannot be split.
Tensor dtypes, shapes, index mappings, and representative source bytes were
validated; tensor bytes and quantization were unchanged.

The optional `fastsafetensors` loader is excluded because there is no selected
Windows ARM64 artifact.

Consequences:

- Some published checkpoints cannot be loaded directly.
- Users must manually reshard models or increase pagefile/commit capacity.
- Lazy loading is slower and depends on careful mmap lifetime synchronization.

Needed:

- A native Windows ARM64 `fastsafetensors`/DirectStorage wheel.
- Better failure diagnostics for Windows mmap and commit-limit failures.
- Optional automatic resharing or a documented preprocessing command.

## Other Missing Optional Libraries

These do not block the tested text-only Qwen/Gemma path, but their features are
not complete on native Windows ARM64:

| Library | Missing feature or consequence | Current policy |
|---|---|---|
| xFormers | Optional Pixtral path | Excluded on ARM64 |
| TorchCodec | Multimedia codec backend | Excluded on ARM64 |
| PyNvVideoCodec | NVIDIA video decoding | Excluded; source required |
| fastsafetensors | Faster/DirectStorage loading | Excluded on ARM64 |
| TileLang | Specialized MHC kernels | Excluded on ARM64 |
| Humming | Optional quantized GEMM backend | Excluded on ARM64 |
| cuDNN frontend | Optional frontend kernels | Requires local source build |
| QuACK | CuTeDSL kernel dependency | Linux-only path |
| tokenspeed-mla | Specialized MLA acceleration | Linux-only |

Image-only processing can use locally built OpenCV. Full video support requires
native FFmpeg, TorchCodec, NVIDIA Video Codec SDK components, and PyNvVideoCodec.

## Priority Order

### P0: Make installation reproducible

1. Publish the custom PyTorch family, Triton, and vLLM Windows ARM64 wheels.
2. Publish a tested dependency lock/wheelhouse.
3. Add wheel import and PE-machine validation in CI.

### P1: Unlock the main performance paths

1. Port `cuda-bindings` and `cuda-tile` to Windows ARM64.
2. Publish a Windows ARM64 CUTLASS DSL runtime.
3. Port FlashInfer attention and an SM12.1 JIT cache; validate GDN correctness
   before enabling it.
4. Fix complete Triton-kernels packaging and imports.
5. Prioritize FlashInfer attention over FP4 linear and sampling artifacts. FP4
   had no reliable end-to-end latency benefit, the sampler had no measured
   advantage, and GDN remains correctness-blocked for this checkpoint.

### P2: Fix runtime quality

1. Register Triton scratch allocation for TMA.
2. Fix Torch compile/Triton-wrapper compatibility; matched Linux testing shows
   approximately 32% decode upside when combined with decode-only graphs.
3. Diagnose multi-token CUDA-graph replay stalls; decode-only graphs alone
   showed approximately 26% decode upside.
4. Improve optional-backend logging and fallback diagnostics.

### P3: Complete loading and multimedia

1. Port fastsafetensors/DirectStorage.
2. Build TorchCodec and shared FFmpeg for ARM64.
3. Build PyNvVideoCodec against the NVIDIA Video Codec SDK.
4. Port xFormers, TileLang, Humming, and cuDNN frontend as model needs require.

## Practical Current Configuration

For native text serving today:

- Use the custom ARM64 PyTorch, Triton, and vLLM wheels.
- Use CUTLASS for FP8/NVFP4 linear layers.
- Use FA2/Triton attention and Triton/FLA GDN.
- Keep FlashInfer, CuTeDSL, TMA, and CUDA graphs disabled.
- Use standard safetensors lazy loading with Windows-friendly shards.
- Use FP8 K/V cache and checkpoint-requested FP32 GDN state.
- Use 4K prefill chunks for Qwen3.8 on this system.
- Enable experimental prefix caching only when repeated prefixes justify its
  lower cache capacity.
- Enable one-step MTP for long output generation when its additional memory is
  acceptable.

This configuration is native and reliable. The missing libraries primarily
affect installation convenience, optional models/media, and peak performance,
not basic text-generation correctness.