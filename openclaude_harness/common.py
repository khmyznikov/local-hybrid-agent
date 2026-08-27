import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

HARNESS = Path(__file__).resolve().parent
ROOT = Path(os.getenv("LOCAL_HYBRID_AGENT_ROOT", HARNESS.parent))
STATE_HOME = Path(
    os.getenv(
        "LOCAL_HYBRID_STATE_HOME",
        Path.home() / ".local/share/local-hybrid-agent",
    )
)
LITELLM_HOME = Path(os.getenv("LITELLM_HOME", Path.home() / "litellm-hybrid"))
LAB = Path(
    os.getenv("LOCAL_HYBRID_SWEBENCH_HOME", LITELLM_HOME / "swebench")
)
OPENCLAUDE_ROOT = Path(
    os.getenv("OPENCLAUDE_ROOT", STATE_HOME / "openclaude")
)
ROUTING_LOG = Path(
    os.getenv("HYBRID_ROUTING_LOG", LITELLM_HOME / "routing-events.jsonl")
)
RESULTS = LAB / "results/openclaude-local-first"
RESULTS_LOG = RESULTS / "results.jsonl"
REPAIRS_LOG = RESULTS / "repairs.jsonl"
CASE_CONFIG = HARNESS / "swebench_cases.json"
AGENT_CONFIG = HARNESS / "agents-implementer.json"
NODE = Path(os.getenv("OPENCLAUDE_NODE", Path.home() / ".local/bin/node"))
CLI = OPENCLAUDE_ROOT / "dist/cli.mjs"
LITELLM_URL = os.getenv("LITELLM_URL", "http://127.0.0.1:4000/v1")
LITELLM_KEY = os.getenv("LITELLM_MASTER_KEY", "hybrid-copilot")
PARENT_ALIAS = os.getenv("LITELLM_PARENT_ALIAS", "copilot-gpt-5.4")
LOCAL_ALIAS = os.getenv("LITELLM_LOCAL_ALIAS", "qwen-local")
PARENT_UPSTREAM_MODEL = "gpt-5.4"
LOCAL_UPSTREAM_MODEL = "qwen3.8-27b-local"
PARENT_MAX_TURNS = "4"
PARENT_ALLOWED_TOOLS = [
    "Agent",
    "Bash",
    "Read",
    "Grep",
    "Glob",
    "Edit",
    "Write",
]
MODE_DIR = {
    "control": "openclaude-local-first-control",
    "hybrid": "openclaude-local-first-hybrid",
}

sys.path.insert(0, str(ROOT))
import hybrid_swe_experiment as bench  # noqa: E402

bench.ROOT = ROOT
bench.LAB = LAB
bench.DATASET = LAB / "data/test-00000-of-00001.parquet"
bench.CASE_CONFIG = CASE_CONFIG
bench.ROUTING_LOG = ROUTING_LOG
bench.UV = Path(os.getenv("UV", Path.home() / ".local/bin/uv"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_runtime() -> None:
    require(
        NODE.is_file(),
        f"Node runtime is missing: {NODE}. Run setup_openclaude_local_first_wsl.sh.",
    )
    require(
        CLI.is_file(),
        f"OpenClaude build is missing: {CLI}. Run setup_openclaude_local_first_wsl.sh.",
    )
    require(
        bench.DATASET.is_file(),
        f"SWE-bench dataset is missing: {bench.DATASET}. "
        "Run setup_litellm_hybrid_wsl.sh.",
    )
    health_url = f"{LITELLM_URL.removesuffix('/v1')}/health/liveliness"
    request = urllib.request.Request(
        health_url,
        headers={"Authorization": f"Bearer {LITELLM_KEY}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5):
            pass
    except OSError as error:
        raise RuntimeError(
            f"LiteLLM is not healthy at {health_url}. "
            "Run start_litellm_hybrid_wsl.sh."
        ) from error


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 900,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def load_cases(case_ids: list[str] | None = None) -> list[dict[str, Any]]:
    records = bench.load_cases()
    if not case_ids:
        return records
    selected = set(case_ids)
    matched = [
        record for record in records if record["instance_id"] in selected
    ]
    if len(matched) != len(selected):
        found = {record["instance_id"] for record in matched}
        raise ValueError(f"Unknown cases: {sorted(selected - found)}")
    return matched


def shared_environment(record: dict[str, Any]) -> Path:
    environment = LAB / "envs/cloud" / record["instance_id"]
    if (environment / ".ready").exists():
        return environment
    worktree = bench.prepare_worktree(record, "cloud", False)
    return bench.prepare_environment(
        record,
        "cloud",
        worktree,
        False,
    )


def prepare_case(
    record: dict[str, Any], arm: str, force: bool
) -> tuple[Path, Path]:
    worktree = bench.prepare_worktree(record, MODE_DIR[arm], force)
    environment = shared_environment(record)
    bench.clean_worktree(record, worktree)
    bench.refresh_editable(record, worktree, environment)
    return worktree, environment


def baseline_case(
    record: dict[str, Any], arm: str, force: bool
) -> dict[str, Any]:
    worktree, environment = prepare_case(record, arm, force)
    require(
        bench.apply_patch(worktree, record["test_patch"]),
        f"Could not apply official test patch for {record['instance_id']}",
    )
    test = bench.run_tests(record, worktree, environment)
    output = (test.stdout + test.stderr)[-12000:]
    bench.clean_worktree(record, worktree)
    bench.refresh_editable(record, worktree, environment)
    expected_failure = test.returncode == 1 and "FAILED " in output
    require(
        expected_failure,
        f"Baseline did not fail for {record['instance_id']}",
    )
    return {
        "instance_id": record["instance_id"],
        "arm": arm,
        "baseline_exit_code": test.returncode,
        "expected_failure": expected_failure,
    }


def make_prompt(record: dict[str, Any]) -> str:
    tests = " ".join(record["FAIL_TO_PASS"])
    return (
        "Use the experiment-implementer agent exactly once to solve the issue "
        "below in this already-isolated benchmark checkout. Do not request Agent "
        "worktree isolation. Tell the child to edit the checkout and run these "
        f"focused tests when they exist: {tests}. The official node may be "
        "introduced only by the evaluator's hidden test patch; if it is absent, "
        "implement from repository evidence and run the nearest existing tests "
        "instead of stopping. After the child returns, invoke no more tools and "
        "immediately return a concise relay of its changed files and tests. "
        "External harness scoring owns final verification. Do not invoke any "
        "other Agent, investigate independently, commit, read outside this "
        "checkout, or search for the upstream solution.\n\n"
        f"Issue:\n{record['problem_statement']}"
    )


def parse_events(stream: str) -> list[dict[str, Any]]:
    events = []
    for line in stream.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def assistant_events(
    events: list[dict[str, Any]], parent_tool_use_id: str | None
) -> list[dict[str, Any]]:
    selected = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        if event.get("parent_tool_use_id") != parent_tool_use_id:
            continue
        model = event.get("message", {}).get("model")
        if isinstance(model, str) and model != "<synthetic>":
            selected.append(event)
    return selected


def tool_uses(
    events: list[dict[str, Any]], parent_tool_use_id: str | None
) -> list[dict[str, Any]]:
    uses = []
    for event in assistant_events(events, parent_tool_use_id):
        content = event.get("message", {}).get("content", [])
        if isinstance(content, dict):
            content = [content]
        if isinstance(content, list):
            uses.extend(
                block
                for block in content
                if isinstance(block, dict) and block.get("type") == "tool_use"
            )
    return uses


def write_settings(
    run_dir: Path,
    arm: str,
    filename: str = "settings.json",
) -> Path:
    child_alias = PARENT_ALIAS if arm == "control" else LOCAL_ALIAS
    settings = {
        "env": {
            "OPENAI_MODEL": PARENT_ALIAS,
            "GITHUB_COPILOT_FORCE_SYNC_SUBAGENTS": "1",
            "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
            "CLAUDE_CODE_USE_OPENAI": "1",
            "OPENAI_BASE_URL": LITELLM_URL,
            "OPENAI_API_KEY": LITELLM_KEY,
        },
        "permissions": {
            "deny": [
                "Agent(general-purpose)",
                "Agent(statusline-setup)",
                "Agent(code-reviewer)",
                "Agent(Explore)",
                "Agent(Plan)",
                "Agent(verification)",
            ]
        },
        "agentModels": {
            "litellm-implementer": {
                "model": child_alias,
                "base_url": LITELLM_URL,
                "api_key": LITELLM_KEY,
            }
        },
        "agentRouting": {
            "experiment-implementer": "litellm-implementer",
        },
    }
    destination = run_dir / filename
    destination.write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def route_offset() -> int:
    return ROUTING_LOG.stat().st_size if ROUTING_LOG.exists() else 0


def wait_for_routes(
    offset: int,
    route_slice: Path,
    timeout: float = 15,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    lines: list[str] = []
    previous_count = -1
    stable_since = None
    while time.monotonic() < deadline:
        if ROUTING_LOG.exists():
            with ROUTING_LOG.open("rb") as handle:
                handle.seek(offset)
                lines = handle.read().decode("utf-8").splitlines()
        completed = sum(
            1
            for line in lines
            if json.loads(line).get("status") in {"success", "failure"}
        )
        if completed > 0 and completed == previous_count:
            stable_since = stable_since or time.monotonic()
        else:
            stable_since = None
        if stable_since and time.monotonic() - stable_since >= 2:
            break
        previous_count = completed
        time.sleep(0.2)
    routes = [json.loads(line) for line in lines if line]
    route_slice.write_text(
        "".join(f"{json.dumps(event)}\n" for event in routes),
        encoding="utf-8",
    )
    return routes


def route_usage(routes: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "input_tokens": sum(
            int(event.get("prompt_tokens", 0) or 0) for event in routes
        ),
        "cache_read_input_tokens": 0,
        "output_tokens": sum(
            int(event.get("completion_tokens", 0) or 0) for event in routes
        ),
    }


def local_usage(routes: list[dict[str, Any]]) -> dict[str, Any]:
    usage = route_usage(routes)
    requests = []
    for event in routes:
        request_usage = {
            "prompt_tokens": int(event.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(
                event.get("completion_tokens", 0) or 0
            ),
        }
        request_usage["total_tokens"] = sum(request_usage.values())
        requests.append(
            {
                "status": 200,
                "latencyMs": int(
                    float(event.get("duration_seconds", 0) or 0) * 1000
                ),
                "usage": request_usage,
                "model": event.get("requested_model"),
                "stream": False,
            }
        )
    return {
        "calls": len(routes),
        "prompt_tokens": usage["input_tokens"],
        "completion_tokens": usage["output_tokens"],
        "total_tokens": usage["input_tokens"] + usage["output_tokens"],
        "requests": requests,
    }


def parse_metrics(
    events: list[dict[str, Any]],
    arm: str,
    routes: list[dict[str, Any]],
) -> dict[str, Any]:
    init = next(
        event
        for event in events
        if event.get("type") == "system" and event.get("subtype") == "init"
    )
    result = next(
        (event for event in reversed(events) if event.get("type") == "result"),
        None,
    )
    parent_tools = tool_uses(events, None)
    agent_calls = [tool for tool in parent_tools if tool.get("name") == "Agent"]
    require(len(agent_calls) == 1, f"{arm}: expected exactly one Agent call")
    successes = [route for route in routes if route.get("status") == "success"]
    failures = [route for route in routes if route.get("status") == "failure"]
    cloud_routes = [
        route
        for route in successes
        if route.get("requested_model") == PARENT_UPSTREAM_MODEL
    ]
    local_routes = [
        route
        for route in successes
        if route.get("requested_model") == LOCAL_UPSTREAM_MODEL
    ]
    unexpected = [
        route
        for route in successes
        if route not in cloud_routes and route not in local_routes
    ]
    require(cloud_routes, f"{arm}: no GPT-5.4 routes observed")
    require(
        not failures and not unexpected,
        f"{arm}: failures={len(failures)}, unexpected={len(unexpected)}",
    )
    if arm == "control":
        require(not local_routes, f"{arm}: unexpected local routes")
    else:
        require(local_routes, f"{arm}: no local implementer routes")
    agent_id = agent_calls[0].get("id")
    return {
        "session_id": init.get("session_id"),
        "parent_model": init.get("model"),
        "child_model": PARENT_ALIAS if arm == "control" else LOCAL_ALIAS,
        "parent_response_events": len(assistant_events(events, None)),
        "child_response_events": len(assistant_events(events, agent_id)),
        "cloud_model_calls": len(cloud_routes),
        "parent_tool_calls": len(parent_tools),
        "child_tool_calls": len(tool_uses(events, agent_id)),
        "cloud_usage": route_usage(cloud_routes),
        "local_usage": local_usage(local_routes),
        "duration_ms": (result or {}).get("duration_ms"),
        "result_is_error": (result or {}).get("is_error"),
        "complete_result_event": result is not None,
        "final_answer": (result or {}).get("result", "")[:4000],
        "litellm_successful_routes": len(successes),
        "litellm_failed_routes": len(failures),
    }


def append_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")
