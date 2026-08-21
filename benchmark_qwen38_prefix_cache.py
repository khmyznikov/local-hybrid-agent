import os
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

import torch
from benchmark_qwen38_64k import KV_CACHE_GIB, MAX_MODEL_LEN, MODEL, make_prompt

from vllm import LLM, SamplingParams
from vllm.config.kernel import KernelConfig


def run(llm: LLM, prompt: str, iteration: int) -> None:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = llm.generate(
        [prompt],
        SamplingParams(temperature=0, max_tokens=8, ignore_eos=True),
        use_tqdm=False,
    )[0]
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    print(f"iteration_{iteration}_wall_seconds={wall:.3f}", flush=True)
    print(f"iteration_{iteration}_cached_tokens={result.num_cached_tokens}", flush=True)
    print(f"iteration_{iteration}_response={result.outputs[0].text!r}", flush=True)
    if result.metrics is not None:
        print(
            f"iteration_{iteration}_first_token_latency="
            f"{result.metrics.first_token_latency:.3f}",
            flush=True,
        )


def main() -> None:
    prompt, _, digest = make_prompt()
    print(f"model={MODEL} prompt_sha256={digest}", flush=True)
    llm = LLM(
        model=str(MODEL),
        load_format="safetensors",
        safetensors_load_strategy="lazy",
        language_model_only=True,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.01,
        kv_cache_memory_bytes=int(KV_CACHE_GIB * 1024**3),
        kv_cache_dtype="fp8",
        enable_prefix_caching=True,
        cpu_offload_gb=0,
        disable_custom_all_reduce=True,
        disable_log_stats=False,
        kernel_config=KernelConfig(
            linear_backend="cutlass",
            enable_flashinfer_autotune=False,
            enable_cutedsl_warmup=False,
            enable_jit_warmup=False,
        ),
    )
    run(llm, prompt, 1)
    run(llm, prompt, 2)


if __name__ == "__main__":
    main()
