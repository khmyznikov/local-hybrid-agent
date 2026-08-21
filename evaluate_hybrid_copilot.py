import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "hybrid_eval_cases.json"
DEFAULT_RESULTS = ROOT / "hybrid_eval_results.jsonl"
DEFAULT_METRICS = ROOT / "local_sidekick_metrics.jsonl"
MCP_CONFIG = ROOT / "copilot-local-sidekick.mcp.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        default="local,cloud,hybrid",
        help="Comma-separated subset of local,cloud,hybrid.",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--model", default="qwen3.8-27b-local")
    parser.add_argument("--cloud-model", default="auto")
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def extract_final_output(stdout: str) -> str:
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    for event in reversed(events):
        if event.get("type") == "assistant.message":
            content = event.get("data", {}).get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        for key in ("result", "content", "message", "text"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                text = value.get("content") or value.get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
    return stdout.strip()


def extract_cloud_usage(stdout: str) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = event.get("type")
        data = event.get("data", {})
        if event_type == "session.auto_mode_resolved":
            usage["resolved_model"] = data.get("chosenModel")
            usage["resolved_reasoning_bucket"] = data.get("reasoningBucket")
        elif event_type == "model.call_start":
            model = data.get("model")
            if model:
                usage.setdefault("models_called", [])
                if model not in usage["models_called"]:
                    usage["models_called"].append(model)
        elif event_type == "session.usage_checkpoint":
            usage["total_nano_aiu"] = data.get("totalNanoAiu")
            usage["total_ai_credits"] = (
                data.get("totalNanoAiu", 0) / 1_000_000_000
            )
            usage["total_premium_requests"] = data.get("totalPremiumRequests")
        elif event_type == "result":
            usage["result"] = event.get("usage", {})
        elif event_type == "model.call_end":
            call_usage = data.get("usage")
            if isinstance(call_usage, dict):
                usage.setdefault("model_calls", []).append(call_usage)
    return usage


def base_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in list(environment):
        if name == "COPILOT_MODEL" or name.startswith("COPILOT_PROVIDER_"):
            environment.pop(name, None)
    return environment


def command_for_mode(
    mode: str,
    prompt: str,
    hybrid_instruction: str,
    args: argparse.Namespace,
) -> tuple[list[str], dict[str, str]]:
    environment = base_environment()
    command = [
        "copilot",
        "-p",
        prompt,
        "--allow-all-tools",
        "--disable-builtin-mcps",
        "--no-custom-instructions",
        "--output-format",
        "json",
        "--stream",
        "off",
    ]
    if mode == "local":
        environment.update(
            {
                "COPILOT_PROVIDER_BASE_URL": args.base_url,
                "COPILOT_PROVIDER_TYPE": "openai",
                "COPILOT_PROVIDER_API_KEY": "local-copilot",
                "COPILOT_PROVIDER_WIRE_API": "completions",
                "COPILOT_MODEL": args.model,
                "COPILOT_PROVIDER_MAX_PROMPT_TOKENS": "240000",
                "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS": "8192",
            }
        )
    elif mode == "cloud":
        command.extend(["--model", args.cloud_model])
    elif mode == "hybrid":
        environment.update(
            {
                "LOCAL_SIDEKICK_BASE_URL": args.base_url,
                "LOCAL_SIDEKICK_API_KEY": "local-copilot",
                "LOCAL_SIDEKICK_MODEL": args.model,
                "LOCAL_SIDEKICK_TIMEOUT_SECONDS": "300",
                "LOCAL_SIDEKICK_MAX_CONTEXT_CHARS": "60000",
            }
        )
        command[2] = f"{hybrid_instruction}\n\nTask: {prompt}"
        command.extend(
            [
                "--model",
                args.cloud_model,
                "--additional-mcp-config",
                f"@{MCP_CONFIG}",
                "--allow-all-mcp-server-instructions",
            ]
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return command, environment


def main() -> None:
    args = parse_args()
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    args.results.parent.mkdir(parents=True, exist_ok=True)

    with args.results.open("a", encoding="utf-8") as output:
        for case in cases:
            for mode in modes:
                before_metrics = read_metrics(args.metrics)
                command, environment = command_for_mode(
                    mode,
                    case["prompt"],
                    case.get("hybrid_instruction", ""),
                    args,
                )
                started = time.perf_counter()
                process = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=args.timeout,
                    check=False,
                )
                elapsed = time.perf_counter() - started
                after_metrics = read_metrics(args.metrics)
                local_metrics = after_metrics[len(before_metrics) :]
                answer = extract_final_output(process.stdout)
                passed = bool(re.fullmatch(case["expected_regex"], answer.strip()))
                record = {
                    "timestamp": time.time(),
                    "case": case["name"],
                    "mode": mode,
                    "passed": passed,
                    "answer": answer,
                    "elapsed_seconds": elapsed,
                    "exit_code": process.returncode,
                    "cloud_usage": extract_cloud_usage(process.stdout),
                    "local_metrics": local_metrics,
                    "stderr": process.stderr[-2000:],
                }
                output.write(json.dumps(record, ensure_ascii=True) + "\n")
                output.flush()
                print(
                    f"case={case['name']} mode={mode} passed={passed} "
                    f"elapsed={elapsed:.2f}s local_calls={len(local_metrics)} "
                    f"answer={answer!r}",
                    flush=True,
                )


if __name__ == "__main__":
    main()