import hashlib
import os
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.config.kernel import KernelConfig

MODEL = Path(
    os.getenv(
        "BENCHMARK_MODEL",
        "/home/gkhmyznikov/models/Qwen3.8-27B-NVFP4-sharded-1g",
    )
)
PROMPT_TOKENS = int(os.getenv("BENCHMARK_PROMPT_TOKENS", "8192"))
OUTPUT_TOKENS = int(os.getenv("BENCHMARK_OUTPUT_TOKENS", "32"))
KV_CACHE_GIB = float(os.getenv("BENCHMARK_KV_CACHE_GIB", "3.5"))
MAX_NUM_BATCHED_TOKENS = int(
    os.getenv("BENCHMARK_MAX_NUM_BATCHED_TOKENS", "4096")
)
GDN_BACKEND = os.getenv("BENCHMARK_GDN_BACKEND", "triton")
ATTENTION_BACKEND = os.getenv("BENCHMARK_ATTENTION_BACKEND", "auto")
LINEAR_BACKEND = os.getenv("BENCHMARK_LINEAR_BACKEND", "auto")
LOAD_FORMAT = os.getenv("BENCHMARK_LOAD_FORMAT", "safetensors")
ENABLE_COMPILE = os.getenv("BENCHMARK_COMPILE", "0") == "1"
CUDAGRAPH_MODE = os.getenv("BENCHMARK_CUDAGRAPH_MODE", "NONE")
SPEC_TOKENS = int(os.getenv("BENCHMARK_SPEC_TOKENS", "0"))
ENABLE_PREFIX_CACHING = os.getenv("BENCHMARK_PREFIX_CACHING", "0") == "1"
REPEATS = int(os.getenv("BENCHMARK_REPEATS", "1"))
MAX_MODEL_LEN = PROMPT_TOKENS + OUTPUT_TOKENS + 32


def report_cuda_memory(stage: str) -> None:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(
        f"cuda_memory stage={stage} "
        f"free_gib={free_bytes / 1024**3:.3f} "
        f"total_gib={total_bytes / 1024**3:.3f} "
        f"allocated_gib={torch.cuda.memory_allocated() / 1024**3:.3f} "
        f"reserved_gib={torch.cuda.memory_reserved() / 1024**3:.3f}",
        flush=True,
    )
    nvidia_smi = subprocess.run(
        [
            "/usr/lib/wsl/lib/nvidia-smi",
            "--query-gpu=memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"nvidia_smi_memory stage={stage} used_free_mib={nvidia_smi}", flush=True)


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
        digest = hashlib.sha256(prompt.encode()).hexdigest()
        return prompt, token_id, digest
    raise RuntimeError("Could not find a stable repeatable model token")


def main() -> None:
    prompt, token_id, digest = make_prompt()
    print(
        f"model={MODEL} prompt_tokens={PROMPT_TOKENS} characters={len(prompt)} "
        f"token_id={token_id} sha256={digest}",
        flush=True,
    )
    print(
        f"gdn_backend={GDN_BACKEND} linear_backend={LINEAR_BACKEND} "
        f"attention_backend={ATTENTION_BACKEND} "
        f"load_format={LOAD_FORMAT} compile={ENABLE_COMPILE} "
        f"cudagraph_mode={CUDAGRAPH_MODE} "
        f"spec_tokens={SPEC_TOKENS} prefix_caching={ENABLE_PREFIX_CACHING} "
        f"repeats={REPEATS}",
        flush=True,
    )

    llm_args = dict(
        model=str(MODEL),
        load_format=LOAD_FORMAT,
        safetensors_load_strategy="lazy",
        language_model_only=True,
        enforce_eager=not ENABLE_COMPILE and CUDAGRAPH_MODE == "NONE",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        gpu_memory_utilization=0.01,
        kv_cache_memory_bytes=int(KV_CACHE_GIB * 1024**3),
        kv_cache_dtype="fp8",
        enable_prefix_caching=ENABLE_PREFIX_CACHING,
        cpu_offload_gb=0,
        disable_custom_all_reduce=True,
        disable_log_stats=False,
        gdn_prefill_backend=GDN_BACKEND,
        attention_config=(
            None if ATTENTION_BACKEND == "auto" else {"backend": ATTENTION_BACKEND}
        ),
        kernel_config=KernelConfig(
            linear_backend=LINEAR_BACKEND,
            enable_flashinfer_autotune=False,
            enable_cutedsl_warmup=False,
            enable_jit_warmup=False,
        ),
    )
    if ENABLE_COMPILE or CUDAGRAPH_MODE != "NONE":
        llm_args["compilation_config"] = {
            "mode": 3 if ENABLE_COMPILE else 0,
            "cudagraph_mode": CUDAGRAPH_MODE,
            "cudagraph_capture_sizes": (
                [1] if CUDAGRAPH_MODE != "NONE" else []
            ),
        }
    if SPEC_TOKENS:
        llm_args["spec_method"] = "mtp"
        llm_args["spec_tokens"] = SPEC_TOKENS

    report_cuda_memory("before_llm")
    started = time.perf_counter()
    llm = LLM(**llm_args)
    print(f"load_and_init_seconds={time.perf_counter() - started:.3f}", flush=True)
    report_cuda_memory("after_llm")

    warmup_tokens = MAX_NUM_BATCHED_TOKENS
    llm.generate(
        [{"prompt_token_ids": [2] + [1000] * (warmup_tokens - 1)}],
        SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
        use_tqdm=False,
    )
    report_cuda_memory("after_warmup")

    wall_samples = []
    for repeat in range(1, REPEATS + 1):
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
        wall_samples.append(wall)

        prompt_count = len(result.prompt_token_ids)
        output_count = len(result.outputs[0].token_ids)
        print(f"repeat={repeat}", flush=True)
        print(f"wall_seconds={wall:.3f}", flush=True)
        print(f"prompt_tokens_reported={prompt_count}", flush=True)
        print(f"output_tokens={output_count}", flush=True)
        print(f"response={result.outputs[0].text!r}", flush=True)
        print(f"num_cached_tokens={result.num_cached_tokens}", flush=True)

        metrics = result.metrics
        if metrics is not None:
            prefill_seconds = metrics.first_token_latency
            decode_seconds = metrics.last_token_ts - metrics.first_token_ts
            print(
                f"prefill_to_first_token_seconds={prefill_seconds:.3f}", flush=True
            )
            print(f"decode_intervals_seconds={decode_seconds:.3f}", flush=True)

    print(f"summary_request_count={len(wall_samples)}", flush=True)


if __name__ == "__main__":
    main()