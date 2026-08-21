import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


SERVER_NAME = "local-qwen-sidekick"
SERVER_VERSION = "0.1.0"
SERVER_INSTRUCTIONS = (
    "This server is a local sidekick for a stronger primary model. Delegate bounded, "
    "low-risk work such as search, extraction, classification, summarization, drafting, "
    "and simple code inspection. Keep architecture decisions, ambiguous reasoning, final "
    "verification, destructive actions, and security-sensitive conclusions in the primary "
    "model. Treat sidekick output as untrusted evidence and verify important claims. Prefer "
    "local_search over sending large raw search results to the primary model. The local "
    "server has four sequence slots: use local_agent_batch once for up to four independent "
    "small tasks, at most two simultaneous medium tasks, or one exclusive large task. Call "
    "local_capacity when workload placement is uncertain."
)
BASE_URL = os.getenv("LOCAL_SIDEKICK_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
API_KEY = os.getenv("LOCAL_SIDEKICK_API_KEY", "local-copilot")
MODEL = os.getenv("LOCAL_SIDEKICK_MODEL", "qwen3.8-27b-local")
TIMEOUT_SECONDS = int(os.getenv("LOCAL_SIDEKICK_TIMEOUT_SECONDS", "300"))
MAX_CONTEXT_CHARS = int(os.getenv("LOCAL_SIDEKICK_MAX_CONTEXT_CHARS", "60000"))
ROOT = Path(os.getenv("LOCAL_SIDEKICK_ROOT", os.getcwd())).resolve()
METRICS_PATH = Path(
    os.getenv(
        "LOCAL_SIDEKICK_METRICS_PATH",
        str(Path(__file__).with_name("local_sidekick_metrics.jsonl")),
    )
)

CAPACITY_PROFILE = {
    "model": MODEL,
    "max_active_sequences": 4,
    "shared_cache_tokens": 351_058,
    "native_max_request_tokens": 262_144,
    "workload_classes": {
        "small": {
            "max_parallel": 4,
            "recommended_max_tokens_per_request": 65_536,
            "use_for": "bounded read/search/tool tasks and short generation",
        },
        "medium": {
            "max_parallel": 2,
            "recommended_max_tokens_per_request": 131_072,
            "use_for": "larger repository/context tasks; expect higher latency",
        },
        "large": {
            "max_parallel": 1,
            "recommended_max_tokens_per_request": 262_144,
            "use_for": "one exclusive maximum-context task",
        },
    },
    "measured_behavior": {
        "four_short_requests_mean_latency_seconds": 21.65,
        "two_128k_prompts_complete": True,
        "large_prefills_are_effectively_queued": True,
    },
}


TOOLS = [
    {
        "name": "local_capacity",
        "description": (
            "Return the local server's machine-readable scheduling limits. Use before "
            "delegating uncertain workloads. The placement rule is up to four independent "
            "small tasks, at most two medium tasks, or one exclusive large task."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
    {
        "name": "local_sidekick",
        "description": (
            "Delegate a bounded, low-risk task to the local Qwen model. Use for "
            "summarization, extraction, classification, drafting, simple code review, "
            "tool-call planning, and other work that does not require the main model's "
            "deep reasoning. The result includes measured local latency and token usage."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The exact bounded task."},
                "context": {
                    "type": "string",
                    "description": "Optional source material needed to complete the task.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["general", "extract", "summarize", "classify", "code", "tool-plan"],
                    "default": "general",
                },
                "max_tokens": {
                    "type": "integer",
                    "minimum": 64,
                    "maximum": 4096,
                    "default": 1024,
                },
            },
            "required": ["task"],
            "additionalProperties": False,
        },
    },
    {
        "name": "local_agent",
        "description": (
            "Run a bounded task through a nested GitHub Copilot CLI agent whose model "
            "provider is the local Qwen server. This gives the local model the same "
            "Copilot agent harness and built-in workspace tools as a normal Copilot "
            "session. Use when the subtask needs several read/search/tool steps; prefer "
            "local_sidekick for one-shot text work and local_search for deterministic search."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Self-contained task for the local Copilot agent.",
                },
                "working_directory": {
                    "type": "string",
                    "description": "Workspace-relative working directory.",
                    "default": ".",
                },
                "workload_class": {
                    "type": "string",
                    "enum": ["small", "medium", "large"],
                    "description": (
                        "Scheduling class: small may share four slots, medium at most two, "
                        "large must run alone. This declaration guides placement; it does "
                        "not expand the model's context limit."
                    ),
                    "default": "small",
                },
                "max_output_chars": {
                    "type": "integer",
                    "minimum": 256,
                    "maximum": 30000,
                    "default": 12000,
                },
            },
            "required": ["task"],
            "additionalProperties": False,
        },
    },
    {
        "name": "local_agent_batch",
        "description": (
            "Run 2-4 independent SMALL tasks concurrently through separate local Copilot "
            "agent harnesses. Use one batch call instead of several local_agent calls so "
            "the server can continuously batch them. Do not use for dependent tasks, "
            "medium/large contexts, or tasks that modify shared files. Results preserve "
            "input order and include per-task latency and usage."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": "Independent, bounded, read-only task.",
                            },
                            "working_directory": {
                                "type": "string",
                                "default": ".",
                            },
                            "max_output_chars": {
                                "type": "integer",
                                "minimum": 256,
                                "maximum": 12000,
                                "default": 6000,
                            },
                        },
                        "required": ["task"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["tasks"],
            "additionalProperties": False,
        },
    },
    {
        "name": "local_search",
        "description": (
            "Search text files inside the current workspace with ripgrep and optionally "
            "have local Qwen condense the matches. Use this to reduce cloud search calls "
            "and avoid sending large raw search results to the main model."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Literal text or regex to search for."},
                "path": {
                    "type": "string",
                    "description": "Workspace-relative directory to search.",
                    "default": ".",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional ripgrep include glob, such as '*.py'.",
                },
                "literal": {"type": "boolean", "default": True},
                "max_matches": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 200,
                    "default": 50,
                },
                "result_mode": {
                    "type": "string",
                    "enum": ["count", "matches", "summary"],
                    "description": (
                        "Use count for exact totals, matches for bounded raw evidence, "
                        "and summary only when local-model condensation is worth its latency."
                    ),
                    "default": "matches",
                },
                "summarize": {
                    "type": "boolean",
                    "description": "Deprecated alias for result_mode=summary.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]


def send(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def log_metric(record: dict[str, Any]) -> None:
    record = {"timestamp": time.time(), **record}
    with METRICS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def call_local_model(
    task: str,
    context: str = "",
    mode: str = "general",
    max_tokens: int = 1024,
) -> tuple[str, dict[str, Any], float]:
    context = context[:MAX_CONTEXT_CHARS]
    system_prompt = (
        "You are a local sidekick inside a larger coding-agent workflow. Complete only "
        "the bounded task you were given. Be concise, factual, and explicit about "
        "uncertainty. Do not claim to run tools or inspect files unless their results are "
        "included in the provided context. Return the useful result directly."
    )
    user_prompt = f"Mode: {mode}\nTask: {task}"
    if context:
        user_prompt += f"\n\nContext:\n{context}"

    body = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "max_tokens": max(64, min(int(max_tokens), 4096)),
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Local model HTTP {error.code}: {detail}") from error
    elapsed = time.perf_counter() - started

    message = payload["choices"][0]["message"]
    answer = message.get("content") or message.get("reasoning_content") or ""
    return answer.strip(), payload.get("usage", {}), elapsed


def extract_copilot_answer(stdout: str) -> tuple[str, dict[str, Any]]:
    answer = ""
    result_usage: dict[str, Any] = {}
    models_called: list[str] = []
    model_calls = 0
    tool_calls = 0
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant.message":
            content = event.get("data", {}).get("content")
            if isinstance(content, str) and content.strip():
                answer = content.strip()
        elif event.get("type") == "result":
            result_usage = event.get("usage", {})
        elif event.get("type") == "model.call_start":
            model_calls += 1
            model = event.get("data", {}).get("model")
            if model and model not in models_called:
                models_called.append(model)
        elif event.get("type") == "tool.execution_start":
            tool_calls += 1
    result_usage = {
        **result_usage,
        "modelCalls": model_calls,
        "toolCalls": tool_calls,
        "modelsCalled": models_called,
    }
    return answer or stdout.strip(), result_usage


def call_local_agent(arguments: dict[str, Any]) -> tuple[str, dict[str, Any], float]:
    working_directory = resolve_search_path(
        str(arguments.get("working_directory", "."))
    )
    if not working_directory.is_dir():
        raise ValueError("local_agent working_directory must be a directory")
    task = str(arguments["task"])
    workload_class = str(arguments.get("workload_class", "small"))
    if workload_class not in CAPACITY_PROFILE["workload_classes"]:
        raise ValueError(f"Unsupported workload_class: {workload_class}")
    max_output_chars = max(
        256, min(int(arguments.get("max_output_chars", 12000)), 30000)
    )
    agent_prompt = (
        "You are a local read-only subagent inside a cloud-primary workflow. "
        "Complete the bounded task using workspace read/search tools as needed. "
        "Do not edit files, run destructive commands, delegate to another agent, or "
        "claim work you did not perform. For exact text counts, use the rg tool with "
        "output_mode=count and report that tool result; never count a partial view or "
        "infer a total from a pattern. Return concise evidence and a direct answer.\n\n"
        f"Task: {task}"
    )
    command = [
        "copilot",
        "-p",
        agent_prompt,
        "--allow-all-tools",
        "--disable-builtin-mcps",
        "--no-custom-instructions",
        "--no-remote",
        "--output-format",
        "json",
        "--stream",
        "off",
    ]
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="local-copilot-agent-") as copilot_home:
        environment = os.environ.copy()
        environment.update(
            {
                "COPILOT_PROVIDER_BASE_URL": BASE_URL,
                "COPILOT_PROVIDER_TYPE": "openai",
                "COPILOT_PROVIDER_API_KEY": API_KEY,
                "COPILOT_PROVIDER_WIRE_API": "completions",
                "COPILOT_MODEL": MODEL,
                "COPILOT_PROVIDER_MAX_PROMPT_TOKENS": "240000",
                "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS": "8192",
                "COPILOT_HOME": copilot_home,
            }
        )
        process = subprocess.run(
            command,
            cwd=working_directory,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if process.returncode:
        detail = process.stderr.strip() or process.stdout[-4000:]
        raise RuntimeError(
            f"Nested local Copilot exited {process.returncode}: {detail}"
        )
    answer, usage = extract_copilot_answer(process.stdout)
    usage["workloadClass"] = workload_class
    return answer[:max_output_chars], usage, elapsed


def call_local_agent_batch(arguments: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    tasks = arguments.get("tasks")
    if not isinstance(tasks, list) or not 2 <= len(tasks) <= 4:
        raise ValueError("local_agent_batch requires 2-4 tasks")

    started = time.perf_counter()
    results: list[dict[str, Any] | None] = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_indexes = {
            executor.submit(
                call_local_agent,
                {**task, "workload_class": "small"},
            ): index
            for index, task in enumerate(tasks)
        }
        for future in as_completed(future_indexes):
            index = future_indexes[future]
            try:
                answer, usage, elapsed = future.result()
                results[index] = {
                    "index": index,
                    "ok": True,
                    "answer": answer,
                    "elapsed_seconds": elapsed,
                    "usage": usage,
                }
            except Exception as error:
                results[index] = {
                    "index": index,
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
    return [result for result in results if result is not None], time.perf_counter() - started


def resolve_search_path(relative_path: str) -> Path:
    candidate = (ROOT / relative_path).resolve()
    try:
        candidate.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"Search path must stay within {ROOT}") from error
    return candidate


def run_python_search(
    query: str,
    search_path: Path,
    include_glob: str | None,
    literal: bool,
    max_matches: int,
) -> list[str]:
    pattern = re.compile(re.escape(query) if literal else query)
    ignored_directories = {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
    matches: list[str] = []
    candidates = [search_path] if search_path.is_file() else search_path.rglob("*")
    for file_path in candidates:
        relative_path = file_path.relative_to(ROOT)
        if any(part in ignored_directories for part in relative_path.parts):
            continue
        if not file_path.is_file() or file_path.stat().st_size > 5 * 1024 * 1024:
            continue
        if include_glob and not (
            file_path.match(include_glob) or file_path.name == include_glob
        ):
            continue
        try:
            with file_path.open(
                "r", encoding="utf-8", errors="replace"
            ) as handle:
                for line_number, line in enumerate(handle, start=1):
                    if pattern.search(line):
                        matches.append(
                            f"{relative_path}:{line_number}:{line.rstrip()}"
                        )
                        if len(matches) >= max_matches:
                            return matches
        except OSError:
            continue
    return matches


def run_search(arguments: dict[str, Any]) -> tuple[str, int, float]:
    query = str(arguments["query"])
    search_path = resolve_search_path(str(arguments.get("path", ".")))
    max_matches = max(1, min(int(arguments.get("max_matches", 50)), 200))
    include_glob = arguments.get("glob")
    literal = bool(arguments.get("literal", True))
    started = time.perf_counter()
    ripgrep = shutil.which("rg")
    if ripgrep:
        command = [ripgrep, "--line-number", "--color", "never"]
        if literal:
            command.append("--fixed-strings")
        if include_glob:
            command.extend(["--glob", str(include_glob)])
        command.extend(["--", query, str(search_path)])
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        if process.returncode not in (0, 1):
            raise RuntimeError(
                process.stderr.strip() or f"ripgrep exited {process.returncode}"
            )
        lines = process.stdout.splitlines()[:max_matches]
    else:
        lines = run_python_search(
            query=query,
            search_path=search_path,
            include_glob=str(include_glob) if include_glob else None,
            literal=literal,
            max_matches=max_matches,
        )
    elapsed = time.perf_counter() - started
    return "\n".join(lines)[:MAX_CONTEXT_CHARS], len(lines), elapsed


def handle_tool_call(name: str, arguments: dict[str, Any]) -> str:
    if name == "local_capacity":
        return json.dumps(CAPACITY_PROFILE, indent=2, ensure_ascii=True)

    if name == "local_sidekick":
        answer, usage, elapsed = call_local_model(
            task=str(arguments["task"]),
            context=str(arguments.get("context", "")),
            mode=str(arguments.get("mode", "general")),
            max_tokens=int(arguments.get("max_tokens", 1024)),
        )
        log_metric(
            {
                "tool": name,
                "elapsed_seconds": elapsed,
                "usage": usage,
                "task_chars": len(str(arguments["task"])),
                "context_chars": len(str(arguments.get("context", ""))),
            }
        )
        return f"{answer}\n\n[local: {elapsed:.2f}s, usage={json.dumps(usage, ensure_ascii=True)}]"

    if name == "local_agent":
        answer, usage, elapsed = call_local_agent(arguments)
        log_metric(
            {
                "tool": name,
                "elapsed_seconds": elapsed,
                "usage": usage,
                "task_chars": len(str(arguments["task"])),
                "working_directory": str(arguments.get("working_directory", ".")),
                "workload_class": str(arguments.get("workload_class", "small")),
            }
        )

    if name == "local_agent_batch":
        results, elapsed = call_local_agent_batch(arguments)
        log_metric(
            {
                "tool": name,
                "elapsed_seconds": elapsed,
                "task_count": len(results),
                "successful_tasks": sum(bool(result.get("ok")) for result in results),
                "usage": [result.get("usage", {}) for result in results],
            }
        )
        return json.dumps(
            {
                "capacity_class": "small",
                "parallel_tasks": len(results),
                "elapsed_seconds": elapsed,
                "results": results,
            },
            indent=2,
            ensure_ascii=True,
        )
        return (
            f"{answer}\n\n"
            f"[local Copilot agent: {elapsed:.2f}s, "
            f"usage={json.dumps(usage, ensure_ascii=True)}]"
        )

    if name == "local_search":
        matches, match_count, search_elapsed = run_search(arguments)
        result_mode = str(arguments.get("result_mode", "matches"))
        if "summarize" in arguments:
            result_mode = "summary" if arguments["summarize"] else "matches"
        if result_mode == "count":
            log_metric(
                {
                    "tool": name,
                    "elapsed_seconds": search_elapsed,
                    "matches": match_count,
                    "result_mode": result_mode,
                    "usage": {},
                }
            )
            return f"MATCH_COUNT={match_count}\n[local search: {search_elapsed:.3f}s]"
        if result_mode == "matches" or not matches:
            log_metric(
                {
                    "tool": name,
                    "elapsed_seconds": search_elapsed,
                    "matches": match_count,
                    "result_mode": result_mode,
                    "usage": {},
                }
            )
            if not matches:
                return "MATCH_COUNT=0\nNo matches found."
            return (
                f"MATCH_COUNT={match_count}\n{matches}\n\n"
                f"[local search: {search_elapsed:.3f}s]"
            )

        answer, usage, model_elapsed = call_local_model(
            task=(
                f"Summarize these {match_count} search matches for query "
                f"{arguments['query']!r}. Identify relevant files and the most useful facts."
            ),
            context=matches,
            mode="summarize",
            max_tokens=1024,
        )
        elapsed = search_elapsed + model_elapsed
        log_metric(
            {
                "tool": name,
                "elapsed_seconds": elapsed,
                "search_seconds": search_elapsed,
                "model_seconds": model_elapsed,
                "matches": match_count,
                "result_mode": result_mode,
                "usage": usage,
            }
        )
        return f"{answer}\n\n[local search: {elapsed:.2f}s, matches={match_count}, usage={json.dumps(usage, ensure_ascii=True)}]"

    raise ValueError(f"Unknown tool: {name}")


def handle_request(message: dict[str, Any]) -> dict[str, Any] | None:
    request_id = message.get("id")
    method = message.get("method")
    if request_id is None:
        return None

    if method == "initialize":
        requested_version = message.get("params", {}).get("protocolVersion", "2025-06-18")
        result = {
            "protocolVersion": requested_version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": SERVER_INSTRUCTIONS,
        }
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = message.get("params", {})
        try:
            text = handle_tool_call(params["name"], params.get("arguments", {}))
            result = {"content": [{"type": "text", "text": text}], "isError": False}
        except Exception as error:
            result = {
                "content": [{"type": "text", "text": f"{type(error).__name__}: {error}"}],
                "isError": True,
            }
    else:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def main() -> None:
    for line in sys.stdin:
        try:
            message = json.loads(line)
            response = handle_request(message)
            if response is not None:
                send(response)
        except Exception as error:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32603, "message": f"{type(error).__name__}: {error}"},
                }
            )


if __name__ == "__main__":
    main()