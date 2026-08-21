import hashlib
import os
import time
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = os.getenv("BENCHMARK_USE_V2", "0")

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.config.kernel import KernelConfig

MODEL = Path(
    os.getenv(
        "BENCHMARK_MODEL",
        r"C:\Dev\models\Qwen3.8-27B-NVFP4-sharded-1g",
    )
)
PROMPT_TOKENS = int(os.getenv("BENCHMARK_PROMPT_TOKENS", "65536"))
OUTPUT_TOKENS = int(os.getenv("BENCHMARK_OUTPUT_TOKENS", "32"))
KV_CACHE_GIB = float(os.getenv("BENCHMARK_KV_CACHE_GIB", "6"))
MAX_NUM_BATCHED_TOKENS = int(
    os.getenv("BENCHMARK_MAX_NUM_BATCHED_TOKENS", "2048")
)
ENABLE_COMPILE = os.getenv("BENCHMARK_COMPILE", "0") == "1"
SPEC_TOKENS = int(os.getenv("BENCHMARK_SPEC_TOKENS", "0"))
MAMBA_SSM_CACHE_DTYPE = os.getenv("BENCHMARK_MAMBA_SSM_CACHE_DTYPE", "auto")
MAX_MODEL_LEN = PROMPT_TOKENS + OUTPUT_TOKENS + 32


def make_prompt() -> tuple[str, int, str]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    for token_id in range(256, min(tokenizer.vocab_size, 20_000)):
        piece = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not piece.strip() or not any(character.isalnum() for character in piece):
            continue
        probe = tokenizer.decode(
            [token_id] * 64,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if tokenizer.encode(probe, add_special_tokens=False) != [token_id] * 64:
            continue
        prompt = tokenizer.decode(
            [token_id] * PROMPT_TOKENS,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        if encoded != [token_id] * PROMPT_TOKENS:
            continue
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        return prompt, token_id, digest
    raise RuntimeError("Could not find a stable repeatable model token")


def main() -> None:
    if not (MODEL / "model.safetensors.index.json").exists():
        raise FileNotFoundError("Checkpoint index is missing")
    if not any(MODEL.glob("*.safetensors")):
        raise FileNotFoundError("Checkpoint weights are missing")

    prompt, token_id, digest = make_prompt()
    print(
        f"model={MODEL} prompt_content_tokens={PROMPT_TOKENS} "
        f"characters={len(prompt)} "
        f"token_id={token_id} sha256={digest}",
        flush=True,
    )

    started = time.perf_counter()
    llm_args = dict(
        model=str(MODEL),
        load_format="safetensors",
        safetensors_load_strategy="lazy",
        language_model_only=True,
        enforce_eager=not ENABLE_COMPILE,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        gpu_memory_utilization=0.01,
        kv_cache_memory_bytes=int(KV_CACHE_GIB * 1024**3),
        kv_cache_dtype="fp8",
        mamba_ssm_cache_dtype=MAMBA_SSM_CACHE_DTYPE,
        enable_prefix_caching=False,
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
    if ENABLE_COMPILE:
        llm_args["compilation_config"] = {
            "mode": 3,
            "cudagraph_mode": "NONE",
            "cudagraph_capture_sizes": [],
        }
    if SPEC_TOKENS:
        llm_args["spec_method"] = "mtp"
        llm_args["spec_tokens"] = SPEC_TOKENS
    llm = LLM(**llm_args)
    print(f"load_and_init_seconds={time.perf_counter() - started:.3f}", flush=True)
    print(f"use_v2_model_runner={llm.llm_engine.vllm_config.use_v2_model_runner}")

    warmup_tokens = MAX_NUM_BATCHED_TOKENS if ENABLE_COMPILE else 128
    llm.generate(
        [{"prompt_token_ids": [2] + [1000] * (warmup_tokens - 1)}],
        SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
        use_tqdm=False,
    )

    torch.cuda.synchronize()
    started = time.perf_counter()
    result = llm.generate(
        [prompt],
        SamplingParams(
            temperature=0,
            max_tokens=OUTPUT_TOKENS,
            ignore_eos=True,
        ),
        use_tqdm=False,
    )[0]
    torch.cuda.synchronize()
    wall = time.perf_counter() - started

    prompt_count = len(result.prompt_token_ids)
    output_count = len(result.outputs[0].token_ids)
    print(f"wall_seconds={wall:.3f}", flush=True)
    print(f"prompt_tokens_reported={prompt_count}", flush=True)
    print(f"output_tokens={output_count}", flush=True)
    print(f"response={result.outputs[0].text!r}", flush=True)
    print(f"num_cached_tokens={result.num_cached_tokens}", flush=True)

    metrics = result.metrics
    if metrics is not None:
        prefill_seconds = metrics.first_token_latency
        decode_seconds = metrics.last_token_ts - metrics.first_token_ts
        print(f"prefill_to_first_token_seconds={prefill_seconds:.3f}", flush=True)
        print(f"decode_31_intervals_seconds={decode_seconds:.3f}", flush=True)
        print(f"metrics={metrics!r}", flush=True)


if __name__ == "__main__":
    main()