import hashlib
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import psutil
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
CONTEXT_LENGTHS = tuple(
    int(value)
    for value in os.getenv(
        "BENCHMARK_CONTEXT_LENGTHS", "180000,200000,262080"
    ).split(",")
)
OUTPUT_TOKENS = int(os.getenv("BENCHMARK_OUTPUT_TOKENS", "32"))
KV_CACHE_GIB = float(os.getenv("BENCHMARK_KV_CACHE_GIB", "11"))
MAX_NUM_BATCHED_TOKENS = int(
    os.getenv("BENCHMARK_MAX_NUM_BATCHED_TOKENS", "4096")
)
LINEAR_BACKEND = os.getenv("BENCHMARK_LINEAR_BACKEND", "cutlass")
ENABLE_COMPILE = os.getenv("BENCHMARK_COMPILE", "1") == "1"
CUDAGRAPH_MODE = os.getenv(
    "BENCHMARK_CUDAGRAPH_MODE", "FULL_DECODE_ONLY"
)
NOMINAL_GPU_MIB = int(os.getenv("BENCHMARK_NOMINAL_GPU_MIB", "32768"))
INIT_ONLY = os.getenv("BENCHMARK_INIT_ONLY", "0") == "1"
NATIVE_CONTEXT_LIMIT = 262_144
MAX_MODEL_LEN = max(CONTEXT_LENGTHS) + OUTPUT_TOKENS + 32
SAMPLE_INTERVAL_SECONDS = 2.0


@dataclass
class ResourceSample:
    memory_total: int
    memory_available: int
    swap_used: int
    gpu_used_mib: int


class ResourceMonitor:
    def __init__(self) -> None:
        self.samples: list[ResourceSample] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.vmstat_start: dict[str, int] = {}
        self.psi_start: dict[str, int] = {}

    @staticmethod
    def read_vmstat() -> dict[str, int]:
        values = {}
        for line in Path("/proc/vmstat").read_text().splitlines():
            key, value = line.split()
            if key in {"pswpin", "pswpout", "pgmajfault", "oom_kill"}:
                values[key] = int(value)
        return values

    @staticmethod
    def read_psi_totals() -> dict[str, int]:
        values = {}
        for line in Path("/proc/pressure/memory").read_text().splitlines():
            match = re.search(r"^(some|full).* total=(\d+)$", line)
            if match:
                values[match.group(1)] = int(match.group(2))
        return values

    @staticmethod
    def read_gpu_used_mib() -> int:
        output = subprocess.run(
            [
                "/usr/lib/wsl/lib/nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return int(output)

    def sample(self) -> None:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        self.samples.append(
            ResourceSample(
                memory_total=memory.total,
                memory_available=memory.available,
                swap_used=swap.used,
                gpu_used_mib=self.read_gpu_used_mib(),
            )
        )

    def run(self) -> None:
        while not self.stop_event.wait(SAMPLE_INTERVAL_SECONDS):
            self.sample()

    def __enter__(self) -> "ResourceMonitor":
        self.vmstat_start = self.read_vmstat()
        self.psi_start = self.read_psi_totals()
        self.sample()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.sample()

    def report(self, stage: str) -> None:
        vmstat_end = self.read_vmstat()
        psi_end = self.read_psi_totals()
        max_gpu_used_mib = max(sample.gpu_used_mib for sample in self.samples)
        print(
            f"resource stage={stage} "
            f"wsl_total_gib={self.samples[0].memory_total / 1024**3:.3f} "
            f"max_wsl_active_gib="
            f"{max(sample.memory_total - sample.memory_available for sample in self.samples) / 1024**3:.3f} "
            f"min_wsl_available_gib="
            f"{min(sample.memory_available for sample in self.samples) / 1024**3:.3f} "
            f"max_swap_used_gib="
            f"{max(sample.swap_used for sample in self.samples) / 1024**3:.3f} "
            f"max_gpu_used_mib={max_gpu_used_mib} "
            f"nominal_gpu_headroom_mib={NOMINAL_GPU_MIB - max_gpu_used_mib} "
            f"psi_some_seconds="
            f"{(psi_end['some'] - self.psi_start['some']) / 1_000_000:.3f} "
            f"psi_full_seconds="
            f"{(psi_end['full'] - self.psi_start['full']) / 1_000_000:.3f} "
            f"swap_in_pages="
            f"{vmstat_end['pswpin'] - self.vmstat_start['pswpin']} "
            f"swap_out_pages="
            f"{vmstat_end['pswpout'] - self.vmstat_start['pswpout']} "
            f"major_faults="
            f"{vmstat_end['pgmajfault'] - self.vmstat_start['pgmajfault']} "
            f"oom_kills={vmstat_end['oom_kill'] - self.vmstat_start['oom_kill']}",
            flush=True,
        )


def report_cuda_memory(stage: str) -> None:
    free_bytes, total_bytes = torch.cuda.mem_get_info()
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
    print(
        f"cuda_memory stage={stage} "
        f"free_gib={free_bytes / 1024**3:.3f} "
        f"total_gib={total_bytes / 1024**3:.3f} "
        f"allocated_gib={torch.cuda.memory_allocated() / 1024**3:.3f} "
        f"reserved_gib={torch.cuda.memory_reserved() / 1024**3:.3f} "
        f"nvidia_smi_used_free_mib={nvidia_smi}",
        flush=True,
    )


def find_stable_token(tokenizer: AutoTokenizer) -> int:
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
        if tokenizer.encode(probe, add_special_tokens=False) == [token_id] * 64:
            return token_id
    raise RuntimeError("Could not find a stable repeatable model token")


def make_prompt(tokenizer: AutoTokenizer, token_id: int, length: int) -> str:
    prompt = tokenizer.decode(
        [token_id] * length,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    if encoded != [token_id] * length:
        raise RuntimeError(f"Prompt did not round-trip at {length} tokens")
    return prompt


def run_context(
    llm: LLM,
    tokenizer: AutoTokenizer,
    token_id: int,
    context_length: int,
) -> None:
    prompt = make_prompt(tokenizer, token_id, context_length)
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    print(
        f"context_start prompt_tokens={context_length} characters={len(prompt)} "
        f"sha256={digest}",
        flush=True,
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    with ResourceMonitor() as monitor:
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
    metrics = result.metrics
    print(
        f"context_result prompt_tokens={prompt_count} output_tokens={output_count} "
        f"wall_seconds={wall:.3f} response={result.outputs[0].text!r} "
        f"num_cached_tokens={result.num_cached_tokens}",
        flush=True,
    )
    if metrics is not None:
        prefill_seconds = metrics.first_token_latency
        decode_seconds = metrics.last_token_ts - metrics.first_token_ts
        print(
            f"context_metrics prompt_tokens={prompt_count} "
            f"prefill_seconds={prefill_seconds:.3f} "
            f"decode_seconds={decode_seconds:.3f}",
            flush=True,
        )
    monitor.report(f"context_{context_length}")
    report_cuda_memory(f"after_context_{context_length}")


def main() -> None:
    if MAX_MODEL_LEN > NATIVE_CONTEXT_LIMIT:
        raise ValueError(
            f"max_model_len={MAX_MODEL_LEN} exceeds native limit "
            f"{NATIVE_CONTEXT_LIMIT}"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    token_id = find_stable_token(tokenizer)
    print(
        f"model={MODEL} contexts={CONTEXT_LENGTHS} output_tokens={OUTPUT_TOKENS} "
        f"max_model_len={MAX_MODEL_LEN} kv_cache_gib={KV_CACHE_GIB} "
        f"chunk_tokens={MAX_NUM_BATCHED_TOKENS} token_id={token_id} "
        f"nominal_gpu_mib={NOMINAL_GPU_MIB}",
        flush=True,
    )
    print(
        f"profile=flashinfer_attention+{LINEAR_BACKEND}_linear+triton_gdn+"
        f"native_sampler compile={ENABLE_COMPILE} "
        f"cudagraph_mode={CUDAGRAPH_MODE} prefix_caching=False",
        flush=True,
    )

    report_cuda_memory("before_llm")
    started = time.perf_counter()
    with ResourceMonitor() as load_monitor:
        llm = LLM(
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
                enable_flashinfer_autotune=False,
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
    print(f"load_and_init_seconds={time.perf_counter() - started:.3f}", flush=True)
    load_monitor.report("load_and_init")
    report_cuda_memory("after_llm")

    llm.generate(
        [
            {
                "prompt_token_ids": [2]
                + [1000] * (MAX_NUM_BATCHED_TOKENS - 1)
            }
        ],
        SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
        use_tqdm=False,
    )
    report_cuda_memory("after_warmup")

    if INIT_ONLY:
        return

    for context_length in CONTEXT_LENGTHS:
        run_context(llm, tokenizer, token_id, context_length)


if __name__ == "__main__":
    main()