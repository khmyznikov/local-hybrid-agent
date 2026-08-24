import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.config.kernel import KernelConfig


MODEL = Path(
    os.getenv(
        "BENCHMARK_MODEL",
        "/home/gkhmyznikov/models/Qwen3.8-27B-NVFP4-RTX5090",
    )
)
SOURCE_ROOT = Path(
    os.getenv("BENCHMARK_SOURCE_ROOT", "/mnt/c/Dev/vllm-qwen38-bench")
)
CONTEXT_LENGTHS = tuple(
    int(value)
    for value in os.getenv("BENCHMARK_CONTEXT_LENGTHS", "40000,96000").split(",")
)
OUTPUT_TOKENS = int(os.getenv("BENCHMARK_OUTPUT_TOKENS", "128"))
KV_CACHE_GIB = float(os.getenv("BENCHMARK_KV_CACHE_GIB", "11"))
KV_CACHE_DTYPE = os.getenv("BENCHMARK_KV_CACHE_DTYPE", "bfloat16")
ATTENTION_BACKEND = os.getenv("BENCHMARK_ATTENTION_BACKEND", "auto")
CHAT_TEMPLATE = os.getenv("BENCHMARK_CHAT_TEMPLATE", "")
CUDAGRAPH_MODE = os.getenv("BENCHMARK_CUDAGRAPH_MODE", "FULL_DECODE_ONLY")
RESULTS_JSON = Path(
    os.getenv(
        "BENCHMARK_RESULTS_JSON",
        f"/home/gkhmyznikov/vllm-qwen38-wsl/results/long_fidelity_{KV_CACHE_DTYPE}_{ATTENTION_BACKEND}.json",
    )
)
MAX_MODEL_LEN = max(CONTEXT_LENGTHS) + OUTPUT_TOKENS + 64

EXPECTED_DEVICE = "edge-rtr-7"
EXPECTED_INTERFACE = "GigabitEthernet0/0/1.201"
EXPECTED_COMMAND = (
    "show mac address-table dynamic interface GigabitEthernet0/0/1.201"
)
EXPECTED_DB_HOST = "db.tenant.internal"
EXPECTED_DB_PORT = "5432"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_cli",
            "description": "Execute one read-only command on a managed device.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device": {"type": "string"},
                    "interface": {"type": "string"},
                    "command": {"type": "string"},
                    "database_host": {"type": "string"},
                    "database_port": {"type": "integer"},
                },
                "required": [
                    "device",
                    "interface",
                    "command",
                    "database_host",
                    "database_port",
                ],
            },
        },
    }
]


def load_archive_text() -> str:
    parts = []
    for pattern in ("*.md", "*.py"):
        for path in sorted(SOURCE_ROOT.glob(pattern)):
            if path.name == Path(__file__).name or path.stat().st_size > 200_000:
                continue
            parts.append(f"\n--- {path.name} ---\n")
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    archive = "".join(parts)
    if not archive:
        raise RuntimeError(f"No source text found under {SOURCE_ROOT}")
    return archive


def render_prompt(
    tokenizer: AutoTokenizer,
    archive_text: str,
    target_tokens: int,
) -> tuple[str, list[int]]:
    record = (
        "Authoritative operational record. Preserve every literal exactly.\n"
        f"DEVICE={EXPECTED_DEVICE}\n"
        f"INTERFACE={EXPECTED_INTERFACE}\n"
        f"COMMAND={EXPECTED_COMMAND}\n"
        f"DB_HOST={EXPECTED_DB_HOST}\n"
        f"DB_PORT={EXPECTED_DB_PORT}\n"
    )
    final_task = (
        "Use the authoritative record from the beginning of the conversation. "
        "Call execute_cli exactly once. Copy all five values exactly; do not "
        "abbreviate, normalize, or invent any literal."
    )

    def build(background: str) -> tuple[str, list[int]]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a read-only infrastructure agent. Tool arguments are "
                    "operational literals and must be copied exactly."
                ),
            },
            {
                "role": "user",
                "content": record + "\nBackground archive:\n" + background,
            },
            {
                "role": "assistant",
                "content": "I have retained the authoritative record.",
            },
            {"role": "user", "content": final_task},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return text, tokenizer.encode(text, add_special_tokens=False)

    archive_ids = tokenizer.encode(archive_text, add_special_tokens=False)
    _, fixed_ids = build("")
    needed = max(1, target_tokens - len(fixed_ids))
    repeated_ids = (archive_ids * (needed // len(archive_ids) + 1))[:needed]
    text, token_ids = build(tokenizer.decode(repeated_ids))
    for _ in range(4):
        difference = target_tokens - len(token_ids)
        if abs(difference) <= 8:
            break
        needed = max(1, needed + difference)
        repeated_ids = (archive_ids * (needed // len(archive_ids) + 1))[:needed]
        text, token_ids = build(tokenizer.decode(repeated_ids))
    return text, token_ids


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    if CHAT_TEMPLATE:
        tokenizer.chat_template = Path(CHAT_TEMPLATE).read_text(encoding="utf-8")
    template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    archive_text = load_archive_text()
    prompts = [
        (target, *render_prompt(tokenizer, archive_text, target))
        for target in CONTEXT_LENGTHS
    ]
    actual_max = max(len(token_ids) for _, _, token_ids in prompts)
    if actual_max + OUTPUT_TOKENS > MAX_MODEL_LEN:
        raise ValueError(
            f"Rendered prompt {actual_max} exceeds max_model_len={MAX_MODEL_LEN}"
        )

    print(
        f"model={MODEL} kv_cache_dtype={KV_CACHE_DTYPE} "
        f"attention_backend={ATTENTION_BACKEND} contexts={CONTEXT_LENGTHS} "
        f"actual_max_prompt={actual_max} chat_template={CHAT_TEMPLATE or 'checkpoint'} "
        f"chat_template_sha256={template_hash}",
        flush=True,
    )
    llm = LLM(
        model=str(MODEL),
        load_format="safetensors",
        safetensors_load_strategy="lazy",
        language_model_only=True,
        enforce_eager=CUDAGRAPH_MODE == "NONE",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.01,
        kv_cache_memory_bytes=int(KV_CACHE_GIB * 1024**3),
        kv_cache_dtype=KV_CACHE_DTYPE,
        enable_prefix_caching=False,
        cpu_offload_gb=0,
        disable_custom_all_reduce=True,
        gdn_prefill_backend="triton",
        attention_config=(
            None
            if ATTENTION_BACKEND == "auto"
            else {"backend": ATTENTION_BACKEND}
        ),
        kernel_config=KernelConfig(
            linear_backend="auto",
            enable_flashinfer_autotune=False,
            enable_cutedsl_warmup=False,
            enable_jit_warmup=False,
        ),
        compilation_config=(
            None
            if CUDAGRAPH_MODE == "NONE"
            else {
                "mode": 0,
                "cudagraph_mode": CUDAGRAPH_MODE,
                "cudagraph_capture_sizes": [1],
            }
        ),
    )
    llm.generate(
        [{"prompt_token_ids": [2] + [1000] * 255}],
        SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
        use_tqdm=False,
    )

    records = []
    for target, prompt_text, prompt_token_ids in prompts:
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = llm.generate(
            [{"prompt_token_ids": prompt_token_ids}],
            SamplingParams(
                temperature=0,
                max_tokens=OUTPUT_TOKENS,
                ignore_eos=False,
            ),
            use_tqdm=False,
        )[0]
        torch.cuda.synchronize()
        output = result.outputs[0]
        text = output.text
        stripped_text = text.strip()
        checks = {
            "device": EXPECTED_DEVICE in text,
            "interface": EXPECTED_INTERFACE in text,
            "command": EXPECTED_COMMAND in text,
            "database_host": EXPECTED_DB_HOST in text,
            "database_port": EXPECTED_DB_PORT in text,
            "tool_open": "<tool_call>" in text,
            "tool_close": "</tool_call>" in text,
            "function_name": "execute_cli" in text,
            "strict_tool_envelope": (
                stripped_text.startswith("<tool_call>")
                and stripped_text.endswith("</tool_call>")
                and "```" not in stripped_text
            ),
        }
        record = {
            "target_context": target,
            "prompt_tokens": len(prompt_token_ids),
            "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
            "output_token_ids": list(output.token_ids),
            "output_text": text,
            "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "wall_seconds": time.perf_counter() - started,
            "checks": checks,
            "all_checks_passed": all(checks.values()),
        }
        records.append(record)
        print(
            f"context={target} prompt_tokens={len(prompt_token_ids)} "
            f"output_tokens={len(output.token_ids)} passed={record['all_checks_passed']} "
            f"checks={json.dumps(checks, sort_keys=True)} output={text!r}",
            flush=True,
        )

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(
        json.dumps(
            {
                "model": str(MODEL),
                "kv_cache_dtype": KV_CACHE_DTYPE,
                "attention_backend": ATTENTION_BACKEND,
                "chat_template": CHAT_TEMPLATE or "checkpoint",
                "chat_template_sha256": template_hash,
                "cudagraph_mode": CUDAGRAPH_MODE,
                "records": records,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    print(f"results_json={RESULTS_JSON}", flush=True)


if __name__ == "__main__":
    main()