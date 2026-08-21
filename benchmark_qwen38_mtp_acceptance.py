import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.config.kernel import KernelConfig
from vllm.v1.spec_decode.metrics import SpecDecodingStats

MODEL = Path(
    os.getenv(
        "BENCHMARK_MODEL",
        "/home/gkhmyznikov/models/Qwen3.8-27B-NVFP4-sharded-1g",
    )
)
OUTPUT_TOKENS = int(os.getenv("BENCHMARK_OUTPUT_TOKENS", "256"))
MIN_OUTPUT_TOKENS = int(os.getenv("BENCHMARK_MIN_OUTPUT_TOKENS", "128"))
SPEC_TOKENS = int(os.getenv("BENCHMARK_SPEC_TOKENS", "1"))
KV_CACHE_GIB = float(os.getenv("BENCHMARK_KV_CACHE_GIB", "3.5"))
ENABLE_COMPILE = os.getenv("BENCHMARK_COMPILE", "0") == "1"
CUDAGRAPH_MODE = os.getenv("BENCHMARK_CUDAGRAPH_MODE", "NONE")
ENABLE_FLASHINFER_AUTOTUNE = (
    os.getenv("BENCHMARK_FLASHINFER_AUTOTUNE", "0") == "1"
)
LINEAR_BACKEND = os.getenv("BENCHMARK_LINEAR_BACKEND", "cutlass")
MAX_NUM_BATCHED_TOKENS = int(
    os.getenv("BENCHMARK_MAX_NUM_BATCHED_TOKENS", "1024")
)
MAX_MODEL_LEN = int(os.getenv("BENCHMARK_MAX_MODEL_LEN", "4096"))
RESULTS_JSON = os.getenv("BENCHMARK_RESULTS_JSON")
LONG_CONTEXT_TOKENS = int(os.getenv("BENCHMARK_LONG_CONTEXT_TOKENS", "0"))
PROMPT_FILTER = {
    name.strip()
    for name in os.getenv("BENCHMARK_PROMPT_FILTER", "").split(",")
    if name.strip()
}

PROMPTS = (
    (
        "factual",
        "Explain how a modern heat pump works. Compare heating efficiency "
        "with an electric resistance heater, discuss cold-weather behavior, "
        "and give practical guidance for choosing a residential system.",
    ),
    (
        "reasoning",
        "A warehouse has three bins. Bin A contains twice as many parts as "
        "Bin B. Bin C contains 120 fewer parts than Bin A. Together they "
        "contain 1,080 parts. Solve for each bin, verify the result, and "
        "explain every algebraic step.",
    ),
    (
        "coding",
        "Write a robust Python implementation of an asynchronous bounded "
        "worker pool. Include type hints, graceful shutdown, exception "
        "propagation, and a short usage example. Explain the concurrency "
        "invariants after the code.",
    ),
    (
        "creative",
        "Write a detailed science-fiction scene in which an engineer on a "
        "generation ship discovers that the navigation computer has been "
        "quietly correcting for an impossible star. Use dialogue and end on "
        "a concrete revelation rather than a vague cliffhanger.",
    ),
)

ACCEPTANCE_OBSERVATIONS: list[tuple[int, int]] = []
ORIGINAL_OBSERVE_DRAFT = SpecDecodingStats.observe_draft


def observe_draft_with_capture(
    self: SpecDecodingStats,
    num_draft_tokens: int,
    num_accepted_tokens: int,
) -> None:
    ACCEPTANCE_OBSERVATIONS.append((num_draft_tokens, num_accepted_tokens))
    ORIGINAL_OBSERVE_DRAFT(self, num_draft_tokens, num_accepted_tokens)


SpecDecodingStats.observe_draft = observe_draft_with_capture


def report_cuda_memory(stage: str) -> None:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(
        f"cuda_memory stage={stage} free_gib={free_bytes / 1024**3:.3f} "
        f"total_gib={total_bytes / 1024**3:.3f} "
        f"allocated_gib={torch.cuda.memory_allocated() / 1024**3:.3f} "
        f"reserved_gib={torch.cuda.memory_reserved() / 1024**3:.3f}",
        flush=True,
    )


def main() -> None:
    if SPEC_TOKENS < 0:
        raise ValueError("BENCHMARK_SPEC_TOKENS must not be negative")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    selected_prompts = tuple(
        item for item in PROMPTS if not PROMPT_FILTER or item[0] in PROMPT_FILTER
    )
    if not selected_prompts:
        raise ValueError("BENCHMARK_PROMPT_FILTER did not match a prompt")

    workloads = []
    for name, prompt in selected_prompts:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if LONG_CONTEXT_TOKENS:
            tail_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
            if len(tail_ids) >= LONG_CONTEXT_TOKENS:
                raise ValueError("Long-context target is shorter than the prompt tail")
            bos_token_id = tokenizer.bos_token_id or 2
            filler_count = LONG_CONTEXT_TOKENS - len(tail_ids) - 1
            prompt_input = {
                "prompt_token_ids": [bos_token_id] + [259] * filler_count + tail_ids
            }
            prompt_token_count = LONG_CONTEXT_TOKENS
        else:
            prompt_input = formatted_prompt
            prompt_token_count = len(
                tokenizer.encode(formatted_prompt, add_special_tokens=False)
            )
        workloads.append((name, prompt_input, prompt_token_count))

    max_prompt_tokens = max(token_count for _, _, token_count in workloads)
    if max_prompt_tokens + OUTPUT_TOKENS > MAX_MODEL_LEN:
        raise ValueError("BENCHMARK_MAX_MODEL_LEN is too small for this workload")

    print(
        f"model={MODEL} spec_tokens={SPEC_TOKENS} output_tokens={OUTPUT_TOKENS} "
        f"min_output_tokens={MIN_OUTPUT_TOKENS} compile={ENABLE_COMPILE} "
        f"cudagraph_mode={CUDAGRAPH_MODE} "
        f"flashinfer_autotune={ENABLE_FLASHINFER_AUTOTUNE} "
        f"linear_backend={LINEAR_BACKEND} "
        f"max_num_batched_tokens={MAX_NUM_BATCHED_TOKENS} "
        f"prompts={len(workloads)} long_context_tokens={LONG_CONTEXT_TOKENS}",
        flush=True,
    )
    report_cuda_memory("before_llm")
    started = time.perf_counter()
    llm_args = dict(
        model=str(MODEL),
        load_format="safetensors",
        safetensors_load_strategy="lazy",
        language_model_only=True,
        enforce_eager=not ENABLE_COMPILE and CUDAGRAPH_MODE == "NONE",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        gpu_memory_utilization=0.01,
        kv_cache_memory_bytes=int(KV_CACHE_GIB * 1024**3),
        kv_cache_dtype="fp8",
        enable_prefix_caching=False,
        cpu_offload_gb=0,
        disable_custom_all_reduce=True,
        disable_log_stats=False,
        gdn_prefill_backend="triton",
        kernel_config=KernelConfig(
            linear_backend=LINEAR_BACKEND,
            enable_flashinfer_autotune=ENABLE_FLASHINFER_AUTOTUNE,
            enable_cutedsl_warmup=False,
            enable_jit_warmup=False,
        ),
        compilation_config=(
            {
                "mode": 3 if ENABLE_COMPILE else 0,
                "cudagraph_mode": CUDAGRAPH_MODE,
                "cudagraph_capture_sizes": (
                    [1] if CUDAGRAPH_MODE != "NONE" else []
                ),
            }
            if ENABLE_COMPILE or CUDAGRAPH_MODE != "NONE"
            else None
        ),
    )
    if SPEC_TOKENS:
        llm_args["spec_method"] = "mtp"
        llm_args["spec_tokens"] = SPEC_TOKENS
    llm = LLM(**llm_args)
    print(f"load_and_init_seconds={time.perf_counter() - started:.3f}", flush=True)
    report_cuda_memory("after_llm")

    llm.generate(
        [{"prompt_token_ids": [2] + [1000] * 255}],
        SamplingParams(temperature=0, max_tokens=2, ignore_eos=True),
        use_tqdm=False,
    )
    ACCEPTANCE_OBSERVATIONS.clear()
    report_cuda_memory("after_warmup")

    result_records = []
    for name, prompt_input, prompt_token_count in workloads:
        ACCEPTANCE_OBSERVATIONS.clear()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = llm.generate(
            [prompt_input],
            SamplingParams(
                temperature=0,
                min_tokens=MIN_OUTPUT_TOKENS,
                max_tokens=OUTPUT_TOKENS,
            ),
            use_tqdm=False,
        )[0]
        torch.cuda.synchronize()
        wall = time.perf_counter() - started

        drafted = sum(draft for draft, _ in ACCEPTANCE_OBSERVATIONS)
        accepted = sum(accepted for _, accepted in ACCEPTANCE_OBSERVATIONS)
        drafts = len(ACCEPTANCE_OBSERVATIONS)
        acceptance_rate = accepted / drafted if drafted else 0
        mean_acceptance_length = 1 + accepted / drafts if drafts else 1
        position_rates = (
            [
                sum(
                    accepted_count > position
                    for _, accepted_count in ACCEPTANCE_OBSERVATIONS
                )
                / drafts
                for position in range(SPEC_TOKENS)
            ]
            if drafts
            else []
        )

        output = result.outputs[0]
        output_count = len(output.token_ids)
        metrics = result.metrics
        decode_seconds = 0.0
        if metrics is not None and output_count > 1:
            decode_seconds = metrics.last_token_ts - metrics.first_token_ts

        digest = hashlib.sha256(output.text.encode()).hexdigest()
        preview = output.text[:240].replace("\n", "\\n")
        result_records.append(
            {
                "name": name,
                "prompt_token_count": prompt_token_count,
                "prompt_token_ids": result.prompt_token_ids,
                "output_token_ids": output.token_ids,
                "output_text": output.text,
                "wall_seconds": wall,
                "decode_seconds": decode_seconds,
                "draft_steps": drafts,
                "drafted_tokens": drafted,
                "accepted_tokens": accepted,
                "acceptance_rate": acceptance_rate,
                "mean_acceptance_length": mean_acceptance_length,
                "position_rates": position_rates,
                "acceptance_observations": ACCEPTANCE_OBSERVATIONS.copy(),
            }
        )
        print(
            f"acceptance name={name} output_tokens={output_count} "
            f"wall_seconds={wall:.3f} decode_seconds={decode_seconds:.3f} "
            f"draft_steps={drafts} drafted_tokens={drafted} "
            f"accepted_tokens={accepted} rejected_tokens={drafted - accepted} "
            f"acceptance_rate={acceptance_rate:.4f} "
            f"mean_acceptance_length={mean_acceptance_length:.4f} "
            f"position_rates={','.join(f'{rate:.4f}' for rate in position_rates)} "
            f"output_sha256={digest} preview={preview!r}",
            flush=True,
        )

    if RESULTS_JSON:
        results_path = Path(RESULTS_JSON)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(
            json.dumps(result_records, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"results_json={results_path}", flush=True)


if __name__ == "__main__":
    main()