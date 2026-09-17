"""Native installed-wheel Qwen baseline/DFlash2 validation; no model code patches.

Run with the native repository's scoped run-woa-python.ps1 wrapper. Each mode
must run in a fresh process, sequentially, not with two resident target models.
Correctness canaries are not a substitute for a full-precision model reference.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

TARGET_REVISION = "5b7a687fc8211a5d631c8ca6a593dd37eb26ce33"
DRAFT_REVISION = "dedf8df68adfb1afeaf7b7480c0a0243108177b4"
EXPECTED = {
    "device": "edge-rtr-7",
    "interface": "GigabitEthernet0/0/1.201",
    "command": "show mac address-table dynamic interface GigabitEthernet0/0/1.201",
    "database_host": "db.tenant.internal",
    "database_port": 5432,
}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_cli",
            "description": "Execute one read-only command on a managed device.",
            "parameters": {
                "type": "object",
                "properties": {
                    key: {"type": "integer" if isinstance(value, int) else "string"}
                    for key, value in EXPECTED.items()
                },
                "required": list(EXPECTED),
                "additionalProperties": False,
            },
        },
    }
]


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def check_output(kind: str, text: str) -> bool:
    """Strict values and envelope checks; never execute generated commands/code."""
    text = text.strip()
    try:
        if kind == "arithmetic":
            return text == "408"
        if kind == "sort":
            return json.loads(text) == [-4, 0, 2, 2, 9]
        if kind == "extract":
            return json.loads(text) == EXPECTED
        if kind == "tool":
            match = re.fullmatch(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
            if not match:
                return False
            body = match[1]
            if body.startswith("{"):
                return json.loads(body) == {
                    "name": "execute_cli",
                    "arguments": EXPECTED,
                }
            match = re.fullmatch(
                r"<function=execute_cli>\s*(.*?)\s*</function>", body, re.DOTALL
            )
            if not match:
                return False
            parameters = re.findall(
                r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", match[1], re.DOTALL
            )
            residue = re.sub(
                r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
                "",
                match[1],
                flags=re.DOTALL,
            )
            return (
                not residue.strip()
                and len(parameters) == len(EXPECTED)
                and dict(parameters) == {k: str(v) for k, v in EXPECTED.items()}
            )
    except (ValueError, TypeError):
        return False
    raise ValueError(f"Unknown correctness case: {kind}")


def make_cases(tokenizer, contexts: list[int], smoke: bool) -> list[dict]:
    def render(user: str, tools=None) -> list[int]:
        text = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "Follow the requested output format exactly.",
                },
                {"role": "user", "content": user},
            ],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # Transformers 5 can return BatchEncoding for tokenize=True. Explicit
        # encoding keeps the vLLM TokensPrompt payload a flat list of integers.
        return tokenizer.encode(text, add_special_tokens=False)

    cases = [
        {
            "name": "arithmetic",
            "kind": "arithmetic",
            "ids": render(
                "What is 17 multiplied by 24? Output only the integer, without explanation."
            ),
        }
    ]
    if smoke:
        return cases
    cases.extend(
        [
            {
                "name": "sort",
                "kind": "sort",
                "ids": render(
                    "Sort [9, 2, -4, 2, 0] ascending. Return only a JSON array."
                ),
            },
            {
                "name": "extract",
                "kind": "extract",
                "ids": render(
                    f"Record: {json.dumps(EXPECTED)}\nCopy this record exactly as a JSON object. "
                    "Keep the port numeric. No Markdown or other text."
                ),
            },
        ]
    )
    filler = tokenizer.encode(
        "Background archive: routine maintenance completed; no changes to the authoritative record.\n",
        add_special_tokens=False,
    )
    for target in contexts:

        def build(count):
            background = tokenizer.decode((filler * (count // len(filler) + 1))[:count])
            return render(
                f"Authoritative record: {json.dumps(EXPECTED)}\nBackground archive:\n"
                f"{background}\nUse the authoritative record at the beginning. "
                "Call execute_cli exactly once, copying all five values literally.",
                TOOLS,
            )

        count = max(0, target - len(build(0)))
        ids = build(count)
        for _ in range(5):
            if abs(len(ids) - target) <= 8:
                break
            count = max(0, count + target - len(ids))
            ids = build(count)
        cases.append({"name": f"tool_{target}", "kind": "tool", "ids": ids})
    cases.append(
        {
            "name": "decode",
            "kind": None,
            "ids": render(
                "Write a detailed tutorial on implementing a bounded asynchronous Python "
                "worker pool, with type hints, graceful shutdown, error propagation and examples."
            ),
        }
    )
    return cases


def memory_snapshot() -> dict:
    import psutil
    import torch

    free, total = torch.cuda.mem_get_info()
    # WDDM's CUDA virtual budget is not physical VRAM (and the engine lives
    # in another process). Record device-wide driver memory separately.
    driver_memory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return {
        "gpu_free_bytes": free,
        "gpu_total_bytes": total,
        "cuda_memory_scope": "parent-process CUDA/WDDM budget, not physical VRAM usage",
        "nvidia_smi_name_total_used_free_mib": driver_memory,
        "host_available_bytes": psutil.virtual_memory().available,
        "process_rss_bytes": psutil.Process().memory_info().rss,
    }


def save(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args, report: dict) -> None:
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.config.kernel import KernelConfig

    report["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "triton-windows", "transformers")
        },
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "target_revision": TARGET_REVISION,
        "draft_revision": DRAFT_REVISION,
        "config_sha256": digest((args.model / "config.json").read_text()),
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    report["chat_template_sha256"] = digest(tokenizer.chat_template)
    cases = make_cases(tokenizer, args.contexts, args.smoke)
    max_len = max(len(case["ids"]) for case in cases) + args.output_tokens + 64
    report["prompt_manifest"] = [
        {
            "name": c["name"],
            "tokens": len(c["ids"]),
            "sha256": digest(json.dumps(c["ids"])),
        }
        for c in cases
    ]
    kwargs = {
        "model": str(args.model),
        "dtype": "bfloat16",
        "load_format": "safetensors",
        "safetensors_load_strategy": "lazy",
        "language_model_only": True,
        "enforce_eager": args.graph == "NONE",
        "max_model_len": max_len,
        "max_num_seqs": args.batch_size,
        "max_num_batched_tokens": args.prefill_chunk,
        "gpu_memory_utilization": 0.01,
        "kv_cache_memory_bytes": int(args.kv_gib * 1024**3),
        "kv_cache_dtype": "bfloat16",
        "enable_prefix_caching": False,
        "cpu_offload_gb": 0,
        "disable_custom_all_reduce": True,
        "disable_log_stats": False,
        "seed": 0,
        "gdn_prefill_backend": "triton",
        "attention_backend": args.attention_backend,
        "kernel_config": KernelConfig(
            linear_backend="cutlass",
            enable_flashinfer_autotune=False,
            enable_cutedsl_warmup=False,
            enable_jit_warmup=False,
        ),
    }
    if args.graph != "NONE":
        # V2 capture sizes are token counts, not request counts. A DFlash2
        # verification step has one anchor plus seven draft tokens per request.
        capture_tokens = args.batch_size * (8 if args.mode == "dflash" else 1)
        kwargs["compilation_config"] = {
            "mode": 0,
            "cudagraph_mode": args.graph,
            "cudagraph_capture_sizes": [capture_tokens],
        }
    if args.mode == "dflash":
        kwargs["speculative_config"] = {
            "method": "dflash",
            "model": str(args.draft),
            "num_speculative_tokens": 7,
            "attention_backend": "TRITON_ATTN",
            "kv_cache_dtype": "bfloat16",
        }
    report["engine_kwargs"] = {
        **kwargs,
        "kernel_config": asdict(kwargs["kernel_config"]),
    }
    report["memory_before_load"] = memory_snapshot()
    report["status"] = "loading"
    save(args.results, report)
    started = time.perf_counter()
    llm = LLM(**kwargs)
    report["load_and_init_seconds"] = time.perf_counter() - started
    resolved = llm.llm_engine.vllm_config.compilation_config
    report["resolved_graph_configuration"] = {
        "mode": str(resolved.cudagraph_mode),
        "capture_token_sizes": resolved.cudagraph_capture_sizes,
        "max_capture_token_size": resolved.max_cudagraph_capture_size,
        "note": "Verify capture/replay in the worker log; configuration alone is not proof.",
    }
    report["memory_after_load"] = memory_snapshot()
    report["status"] = "testing"
    save(args.results, report)
    try:
        # Warm every prompt/batch shape. No prefix cache; timed runs recompute prefill.
        for case in cases:
            started = time.perf_counter()
            llm.generate(
                [{"prompt_token_ids": case["ids"]}] * args.batch_size,
                SamplingParams(temperature=0, max_tokens=2, ignore_eos=True),
                use_tqdm=False,
            )
            print(
                f"WARMUP {case['name']} {time.perf_counter() - started:.3f}s",
                flush=True,
            )
        report["metrics_before"] = [asdict(metric) for metric in llm.get_metrics()]
        previous = {}
        for repeat in range(args.repeats):
            for case in cases:
                params = SamplingParams(
                    temperature=0,
                    max_tokens=args.output_tokens,
                    ignore_eos=case["kind"] is None,
                    logprobs=1 if case["kind"] else None,
                )
                started = time.perf_counter()
                results = llm.generate(
                    [{"prompt_token_ids": case["ids"]}] * args.batch_size,
                    params,
                    use_tqdm=False,
                )
                wall = time.perf_counter() - started
                for slot, result in enumerate(results):
                    output = result.outputs[0]
                    token_ids = list(output.token_ids)
                    metrics = result.metrics
                    decode_seconds = (
                        metrics.last_token_ts - metrics.first_token_ts if metrics else 0
                    )
                    selected_logprobs = (
                        [
                            step[token].logprob
                            for step, token in zip(output.logprobs, token_ids)
                        ]
                        if output.logprobs
                        else []
                    )
                    key = (case["name"], slot)
                    record = {
                        "case": case["name"],
                        "kind": case["kind"],
                        "repeat": repeat,
                        "slot": slot,
                        "prompt_sha256": digest(json.dumps(case["ids"])),
                        "prompt_tokens": len(result.prompt_token_ids),
                        "output_tokens": len(token_ids),
                        "output_token_ids": token_ids,
                        "output_text": output.text,
                        "finish_reason": output.finish_reason,
                        "wall_seconds_batch": wall,
                        "cached_tokens": result.num_cached_tokens,
                        "ttft_seconds": metrics.first_token_latency
                        if metrics
                        else None,
                        "decode_seconds": decode_seconds,
                        "decode_tokens_per_second": (len(token_ids) - 1)
                        / decode_seconds
                        if decode_seconds > 0 and len(token_ids) > 1
                        else None,
                        "finite_selected_logprobs": all(
                            math.isfinite(p) for p in selected_logprobs
                        )
                        if selected_logprobs
                        else None,
                        "correctness_passed": check_output(case["kind"], output.text)
                        if case["kind"]
                        else None,
                        "repeat_exact_match": previous[key] == token_ids
                        if key in previous
                        else None,
                    }
                    previous.setdefault(key, token_ids)
                    report["records"].append(record)
                    print(
                        json.dumps(
                            {k: v for k, v in record.items() if k != "output_token_ids"}
                        ),
                        flush=True,
                    )
                save(args.results, report)
        report["metrics_after"] = [asdict(metric) for metric in llm.get_metrics()]
        report["memory_after_tests"] = memory_snapshot()
        report["status"] = "completed"
        rates = [
            r["decode_tokens_per_second"]
            for r in report["records"]
            if r["case"] == "decode"
        ]
        report["summary"] = {
            "correctness_passed": all(
                r["correctness_passed"] is not False for r in report["records"]
            ),
            "repeat_exact_match": all(
                r["repeat_exact_match"] is not False for r in report["records"]
            ),
            "median_decode_tokens_per_second": statistics.median(r for r in rates if r)
            if any(rates)
            else None,
        }
    finally:
        llm.llm_engine.engine_core.shutdown()
        del llm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path(r"X:\models\Qwen3.8-27B-NVFP4-RTX5090")
    )
    parser.add_argument(
        "--draft", type=Path, default=Path(r"X:\models\Qwen3.8-27B-DFlash2")
    )
    parser.add_argument("--mode", choices=["baseline", "dflash"], default="baseline")
    parser.add_argument(
        "--contexts",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[2048, 8192],
    )
    parser.add_argument("--output-tokens", type=int, default=192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefill-chunk", type=int, default=1024)
    parser.add_argument("--kv-gib", type=float, default=1.0)
    parser.add_argument(
        "--attention-backend",
        choices=["TRITON_ATTN", "FLASH_ATTN"],
        default="TRITON_ATTN",
    )
    parser.add_argument("--graph", choices=["NONE", "FULL_DECODE_ONLY"], default="NONE")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(
            args.output_tokens,
            args.repeats,
            args.batch_size,
            args.prefill_chunk,
            args.kv_gib,
            *args.contexts,
        )
        <= 0
    ):
        parser.error(
            "Token counts, repeats, batch/cache sizes and contexts must be positive"
        )
    report = {
        "status": "starting",
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "records": [],
    }
    try:
        run(args, report)
    except Exception:
        report["status"] = "error"
        report["error"] = traceback.format_exc()
        raise
    finally:
        save(args.results, report)
        print(f"RESULTS {args.results} STATUS {report['status']}", flush=True)


if __name__ == "__main__":
    main()
