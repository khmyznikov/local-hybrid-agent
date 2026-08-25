import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl
from litellm.router_strategy.complexity_router.complexity_router import (
    ComplexityRouter,
)

ROOT = Path("/mnt/c/Dev/vllm-qwen38-bench")
LAB = Path("/home/gkhmyznikov/litellm-hybrid/swebench")
DATASET = LAB / "data/test-00000-of-00001.parquet"
CASE_CONFIG = ROOT / "hybrid_swe_cases.json"
ROUTING_LOG = Path("/home/gkhmyznikov/litellm-hybrid/routing-events.jsonl")
UV = Path("/home/gkhmyznikov/.local/bin/uv")
COPILOT = Path("/home/gkhmyznikov/.local/bin/copilot")

MODEL_BY_MODE = {
    "local": "qwen-local",
    "cloud": "copilot-gpt-5.4",
    "hybrid": "hybrid-copilot",
    "cloud-sol": "copilot-gpt-5.6-sol",
    "hybrid-sol": "hybrid-copilot-sol",
}
REPO_DIRS = {
    "pytest-dev/pytest": "pytest",
    "sphinx-doc/sphinx": "sphinx",
    "pylint-dev/pylint": "pylint",
}
REPO_URLS = {
    repo: f"https://github.com/{repo}.git" for repo in REPO_DIRS
}
ROUTER_CONFIG = {
    "tiers": {
        "SIMPLE": "qwen-local",
        "MEDIUM": "qwen-local",
        "COMPLEX": "copilot-gpt-5.4",
        "REASONING": "copilot-gpt-5.4",
    }
}


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def require_success(
    process: subprocess.CompletedProcess[str], description: str
) -> None:
    if process.returncode:
        detail = process.stderr.strip() or process.stdout[-4000:]
        raise RuntimeError(f"{description} failed ({process.returncode}): {detail}")


def load_cases() -> list[dict[str, Any]]:
    selected = set(json.loads(CASE_CONFIG.read_text(encoding="utf-8"))["cases"])
    frame = pl.read_parquet(DATASET).filter(pl.col("instance_id").is_in(selected))
    records = frame.to_dicts()
    if len(records) != len(selected):
        found = {record["instance_id"] for record in records}
        raise RuntimeError(f"Missing selected cases: {sorted(selected - found)}")
    for record in records:
        for field in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            record[field] = json.loads(record[field])
    return sorted(records, key=lambda record: record["instance_id"])


def classify_record(
    router: ComplexityRouter, record: dict[str, Any]
) -> tuple[str, float, list[str], str]:
    tier, score, signals = router.classify(record["problem_statement"])
    route = "local" if tier.value in {"SIMPLE", "MEDIUM"} else "cloud"
    return tier.value, score, signals, route


def classify_all() -> None:
    frame = pl.read_parquet(DATASET)
    router = ComplexityRouter(
        "hybrid-copilot",
        None,
        ROUTER_CONFIG,
        derive_savings_baseline=False,
    )
    counts: Counter[str] = Counter()
    selected = {record["instance_id"] for record in load_cases()}
    selected_rows = []
    for record in frame.iter_rows(named=True):
        tier, score, signals, route = classify_record(router, record)
        counts[tier] += 1
        if record["instance_id"] in selected:
            selected_rows.append(
                {
                    "instance_id": record["instance_id"],
                    "repo": record["repo"],
                    "difficulty": record["difficulty"],
                    "tier": tier,
                    "route": route,
                    "score": score,
                    "signals": signals,
                }
            )
    print(
        json.dumps(
            {
                "dataset_cases": frame.height,
                "tier_counts": dict(counts),
                "route_counts": {
                    "local": counts["SIMPLE"] + counts["MEDIUM"],
                    "cloud": counts["COMPLEX"] + counts["REASONING"],
                },
                "selected": selected_rows,
            },
            indent=2,
            ensure_ascii=True,
        )
    )


def repo_dir(record: dict[str, Any]) -> Path:
    return LAB / "repos" / REPO_DIRS[record["repo"]]


def ensure_repo(record: dict[str, Any]) -> Path:
    destination = repo_dir(record)
    if not (destination / ".git").exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        process = run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                REPO_URLS[record["repo"]],
                str(destination),
            ],
            timeout=900,
        )
        require_success(process, f"clone {record['repo']}")
    exists = run(
        ["git", "cat-file", "-e", f"{record['base_commit']}^{{commit}}"],
        cwd=destination,
    )
    if exists.returncode:
        process = run(
            ["git", "fetch", "--depth=1", "origin", record["base_commit"]],
            cwd=destination,
            timeout=900,
        )
        require_success(process, f"fetch {record['instance_id']}")
    return destination


def prepare_worktree(record: dict[str, Any], mode: str, force: bool) -> Path:
    repository = ensure_repo(record)
    destination = LAB / "worktrees" / mode / record["instance_id"]
    if destination.exists() and force:
        run(["git", "worktree", "remove", "--force", str(destination)], cwd=repository)
        shutil.rmtree(destination, ignore_errors=True)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "worktree", "prune"], cwd=repository)
        process = run(
            [
                "git",
                "worktree",
                "add",
                "--detach",
                str(destination),
                record["base_commit"],
            ],
            cwd=repository,
            timeout=900,
        )
        require_success(process, f"worktree {record['instance_id']}")
    return destination


def prepare_environment(
    record: dict[str, Any], mode: str, worktree: Path, force: bool
) -> Path:
    environment = LAB / "envs" / mode / record["instance_id"]
    ready = environment / ".ready"
    if force:
        shutil.rmtree(environment, ignore_errors=True)
    if ready.exists():
        return environment
    environment.parent.mkdir(parents=True, exist_ok=True)
    process = run(
        [str(UV), "venv", "--python", "3.9", str(environment)],
        timeout=900,
    )
    require_success(process, f"create environment {record['instance_id']}")
    python = environment / "bin/python"
    target = str(worktree)
    install_env = os.environ.copy()
    if record["repo"] == "pytest-dev/pytest":
        install_env["SETUPTOOLS_SCM_PRETEND_VERSION"] = "6.0.0"
        install = [str(UV), "pip", "install", "--python", str(python), "-e", f"{target}[testing]", "pygments"]
    elif record["repo"] == "sphinx-doc/sphinx":
        pytest_version = "6.2.5" if record["version"] == "4.1" else "7.1.2"
        install = [
            str(UV),
            "pip",
            "install",
            "--python",
            str(python),
            "-e",
            f"{target}[test]",
            f"pytest=={pytest_version}",
            "setuptools<81",
        ]
    else:
        install = [
            str(UV),
            "pip",
            "install",
            "--python",
            str(python),
            "-e",
            target,
            "-r",
            str(worktree / "requirements_test.txt"),
        ]
    process = run(install, cwd=worktree, env=install_env, timeout=1200)
    require_success(process, f"install environment {record['instance_id']}")
    ready.write_text(record["base_commit"], encoding="ascii")
    clean_worktree(record, worktree)
    refresh_editable(record, worktree, environment)
    return environment


def clean_worktree(record: dict[str, Any], worktree: Path) -> None:
    process = run(["git", "reset", "--hard", record["base_commit"]], cwd=worktree)
    require_success(process, f"reset {record['instance_id']}")
    process = run(["git", "clean", "-fdx"], cwd=worktree)
    require_success(process, f"clean {record['instance_id']}")


def refresh_editable(
    record: dict[str, Any], worktree: Path, environment: Path
) -> None:
    env = os.environ.copy()
    if record["repo"] == "pytest-dev/pytest":
        env["SETUPTOOLS_SCM_PRETEND_VERSION"] = "6.0.0"
    process = run(
        [
            str(UV),
            "pip",
            "install",
            "--python",
            str(environment / "bin/python"),
            "--no-deps",
            "-e",
            str(worktree),
        ],
        cwd=worktree,
        env=env,
        timeout=600,
    )
    require_success(process, f"refresh editable {record['instance_id']}")


def apply_patch(worktree: Path, patch: str, reverse: bool = False) -> bool:
    command = ["git", "apply", "--whitespace=nowarn"]
    if reverse:
        command.append("--reverse")
    return run(command, cwd=worktree, input_text=patch).returncode == 0


def run_tests(
    record: dict[str, Any], worktree: Path, environment: Path
) -> subprocess.CompletedProcess[str]:
    python = environment / "bin/python"
    env = os.environ.copy()
    env["PATH"] = f"{environment / 'bin'}:{env['PATH']}"
    env["VIRTUAL_ENV"] = str(environment)
    return run(
        [str(python), "-m", "pytest", "-q", *record["FAIL_TO_PASS"]],
        cwd=worktree,
        env=env,
        timeout=600,
    )


def parse_agent_output(stdout: str) -> dict[str, Any]:
    answer = ""
    usage: dict[str, Any] = {}
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
        elif event.get("type") == "model.call_start":
            model_calls += 1
        elif event.get("type") == "tool.execution_start":
            tool_calls += 1
        elif event.get("type") == "result":
            usage = event.get("usage", {})
    return {
        "answer": answer[:4000],
        "usage": usage,
        "model_calls": model_calls,
        "tool_calls": tool_calls,
    }


def read_new_routing_events(offset: int) -> list[dict[str, Any]]:
    if not ROUTING_LOG.exists():
        return []
    with ROUTING_LOG.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        events = []
        for line in handle:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events


def wait_for_routing_events(
    offset: int,
    expected_count: int,
    timeout: float = 610,
    settle_seconds: float = 2,
) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    events = read_new_routing_events(offset)
    last_change = time.monotonic()
    while time.monotonic() < deadline:
        if (
            len(events) >= expected_count
            and time.monotonic() - last_change >= settle_seconds
        ):
            break
        time.sleep(0.25)
        updated = read_new_routing_events(offset)
        if len(updated) != len(events):
            last_change = time.monotonic()
            events = updated
    return events


def run_agent(
    record: dict[str, Any], mode: str, worktree: Path, environment: Path
) -> tuple[subprocess.CompletedProcess[str], float, list[dict[str, Any]]]:
    prompt = (
        "Resolve the following repository issue. Inspect the implementation, make the "
        "smallest correct code change, and run focused tests when practical. Do not "
        "commit. Do not read files outside this checkout or search for the upstream "
        "solution. Return a concise summary and tests run.\n\n"
        f"Issue:\n{record['problem_statement']}"
    )
    env = os.environ.copy()
    sol_mode = mode.endswith("-sol")
    env.update(
        {
            "PATH": f"{environment / 'bin'}:/home/gkhmyznikov/.local/bin:{env['PATH']}",
            "VIRTUAL_ENV": str(environment),
            "COPILOT_PROVIDER_BASE_URL": "http://127.0.0.1:4000/v1",
            "COPILOT_PROVIDER_TYPE": "openai",
            "COPILOT_PROVIDER_API_KEY": "hybrid-copilot",
            "COPILOT_PROVIDER_WIRE_API": "responses" if sol_mode else "completions",
            "COPILOT_MODEL": MODEL_BY_MODE[mode],
            "COPILOT_PROVIDER_MAX_PROMPT_TOKENS": "120000",
            "COPILOT_PROVIDER_MAX_OUTPUT_TOKENS": "8192",
        }
    )
    if sol_mode:
        env.update(
            {
                "COPILOT_PROVIDER_MODEL_ID": "gpt-5.6-sol",
                "COPILOT_PROVIDER_WIRE_MODEL": MODEL_BY_MODE[mode],
            }
        )
    routing_offset = ROUTING_LOG.stat().st_size if ROUTING_LOG.exists() else 0
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"copilot-{record['instance_id']}-") as home:
        env["COPILOT_HOME"] = home
        command = [
            str(COPILOT),
            "-p",
            prompt,
            "--allow-all-tools",
            "--disable-builtin-mcps",
            "--no-custom-instructions",
            "--no-remote",
            "--output-format",
            "json",
            "--stream",
            "off",
        ]
        try:
            process = run(
                command,
                cwd=worktree,
                env=env,
                timeout=600,
            )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            process = subprocess.CompletedProcess(
                command,
                124,
                stdout=stdout,
                stderr=stderr or "Agent timed out after 600 seconds.",
            )
    elapsed = time.perf_counter() - started
    model_calls = parse_agent_output(process.stdout)["model_calls"]
    routing_events = wait_for_routing_events(routing_offset, model_calls)
    return process, elapsed, routing_events


def changed_files(worktree: Path, base_commit: str) -> list[str]:
    process = run(["git", "diff", "--name-only", base_commit], cwd=worktree)
    return [line for line in process.stdout.splitlines() if line]


def patch_files(patch: str) -> set[str]:
    files = set()
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            files.add(line[6:])
    return files


def restore_official_test_files(
    record: dict[str, Any], worktree: Path
) -> list[str]:
    files = sorted(
        patch_files(record["test_patch"]) - patch_files(record["patch"])
    )
    if not files:
        return files
    process = run(
        ["git", "restore", "--source", record["base_commit"], "--", *files],
        cwd=worktree,
    )
    require_success(process, f"restore official test files {record['instance_id']}")
    return files


def append_result(record: dict[str, Any]) -> None:
    path = LAB / "results/results.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def latest_result(instance_id: str, mode: str) -> dict[str, Any] | None:
    path = LAB / "results/results.jsonl"
    if not path.exists():
        return None
    latest = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        if record.get("instance_id") == instance_id and record.get("mode") == mode:
            latest = record
    return latest


def rescore_case(record: dict[str, Any], mode: str) -> dict[str, Any]:
    previous = latest_result(record["instance_id"], mode)
    if previous is None:
        raise RuntimeError(f"No saved result for {mode} {record['instance_id']}")
    patch_path = LAB / "results/patches" / mode / f"{record['instance_id']}.patch"
    if not patch_path.exists():
        raise RuntimeError(f"No saved patch for {mode} {record['instance_id']}")

    worktree = prepare_worktree(record, mode, False)
    environment = prepare_environment(record, mode, worktree, False)
    clean_worktree(record, worktree)
    refresh_editable(record, worktree, environment)
    agent_patch = patch_path.read_text(encoding="utf-8")
    if agent_patch and not apply_patch(worktree, agent_patch):
        raise RuntimeError(f"Could not restore agent patch for {record['instance_id']}")
    restored_test_files = restore_official_test_files(record, worktree)
    test_patch_applied = apply_patch(worktree, record["test_patch"])
    if test_patch_applied:
        test = run_tests(record, worktree, environment)
        test_output = (test.stdout + test.stderr)[-12000:]
        test_exit_code = test.returncode
    else:
        test_output = "Official test patch conflicted after restoring agent test files."
        test_exit_code = None

    updated = {
        **previous,
        "rescored": True,
        "restored_agent_test_files": restored_test_files,
        "test_patch_applied": test_patch_applied,
        "focused_tests_passed": test_exit_code == 0,
        "test_exit_code": test_exit_code,
        "test_output": test_output,
    }
    append_result(updated)
    return updated


def execute_case(record: dict[str, Any], mode: str, force: bool) -> dict[str, Any]:
    router = ComplexityRouter(
        "hybrid-copilot",
        None,
        ROUTER_CONFIG,
        derive_savings_baseline=False,
    )
    tier, score, signals, expected_route = classify_record(router, record)
    worktree = prepare_worktree(record, mode, force)
    environment = prepare_environment(record, mode, worktree, force)
    clean_worktree(record, worktree)
    refresh_editable(record, worktree, environment)

    if not apply_patch(worktree, record["test_patch"]):
        raise RuntimeError(f"Could not apply test patch for {record['instance_id']}")
    baseline = run_tests(record, worktree, environment)
    baseline_log = (baseline.stdout + baseline.stderr)[-12000:]
    clean_worktree(record, worktree)
    refresh_editable(record, worktree, environment)
    if baseline.returncode != 1 or "FAILED " not in baseline_log:
        result = {
            "instance_id": record["instance_id"],
            "mode": mode,
            "status": "unsupported_baseline",
            "baseline_exit_code": baseline.returncode,
            "baseline_output": baseline_log,
        }
        append_result(result)
        return result

    agent, elapsed, routing_events = run_agent(record, mode, worktree, environment)
    agent_data = parse_agent_output(agent.stdout)
    diff_process = run(
        ["git", "diff", "--binary", record["base_commit"]], cwd=worktree
    )
    agent_patch = diff_process.stdout
    patch_dir = LAB / "results/patches" / mode
    patch_dir.mkdir(parents=True, exist_ok=True)
    (patch_dir / f"{record['instance_id']}.patch").write_text(
        agent_patch, encoding="utf-8"
    )
    transcript_dir = LAB / "results/transcripts" / mode
    transcript_dir.mkdir(parents=True, exist_ok=True)
    (transcript_dir / f"{record['instance_id']}.jsonl").write_text(
        agent.stdout, encoding="utf-8"
    )

    restored_test_files = restore_official_test_files(record, worktree)
    test_patch_applied = apply_patch(worktree, record["test_patch"])
    if test_patch_applied:
        test = run_tests(record, worktree, environment)
        test_output = (test.stdout + test.stderr)[-12000:]
        test_exit_code = test.returncode
    else:
        test_output = "Official test patch conflicted with the agent changes."
        test_exit_code = None

    agent_files = changed_files(worktree, record["base_commit"])
    gold_files = patch_files(record["patch"])
    implementation_files = [
        name for name in agent_files if name not in patch_files(record["test_patch"])
    ]
    result = {
        "instance_id": record["instance_id"],
        "repo": record["repo"],
        "difficulty": record["difficulty"],
        "mode": mode,
        "requested_model": MODEL_BY_MODE[mode],
        "offline_tier": tier,
        "offline_score": score,
        "offline_signals": signals,
        "expected_route": expected_route,
        "status": "complete" if agent.returncode == 0 else "agent_failed",
        "agent_exit_code": agent.returncode,
        "elapsed_seconds": elapsed,
        "model_calls": agent_data["model_calls"],
        "tool_calls": agent_data["tool_calls"],
        "reported_usage": agent_data["usage"],
        "routing_events": routing_events,
        "routing_events_complete": len(routing_events) >= agent_data["model_calls"],
        "changed_files": implementation_files,
        "gold_files": sorted(gold_files),
        "gold_file_recall": (
            len(gold_files.intersection(implementation_files)) / len(gold_files)
            if gold_files
            else None
        ),
        "restored_agent_test_files": restored_test_files,
        "test_patch_applied": test_patch_applied,
        "focused_tests_passed": test_exit_code == 0,
        "test_exit_code": test_exit_code,
        "test_output": test_output,
        "baseline_exit_code": baseline.returncode,
        "agent_answer": agent_data["answer"],
        "agent_stderr": agent.stderr[-4000:],
    }
    append_result(result)
    return result


def select_cases(case_ids: list[str] | None) -> list[dict[str, Any]]:
    records = load_cases()
    if not case_ids:
        return records
    selected = set(case_ids)
    matched = [record for record in records if record["instance_id"] in selected]
    if len(matched) != len(selected):
        found = {record["instance_id"] for record in matched}
        raise ValueError(f"Unknown selected cases: {sorted(selected - found)}")
    return matched


def summarize() -> None:
    path = LAB / "results/results.jsonl"
    if not path.exists():
        print("No results yet.")
        return
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    latest = {}
    for record in records:
        latest[(record.get("mode"), record["instance_id"])] = record
    summary = Counter()
    routes = Counter()
    elapsed = Counter()
    lanes = Counter()
    for (mode, _instance), record in latest.items():
        summary[(mode, "cases")] += 1
        summary[(mode, "resolved")] += int(record.get("focused_tests_passed", False))
        elapsed[mode] += record.get("elapsed_seconds", 0)
        lane = (
            "local"
            if mode == "local"
            else "cloud"
            if mode.startswith("cloud")
            else record.get("expected_route", "unknown")
        )
        lanes[(mode, lane, "cases")] += 1
        lanes[(mode, lane, "resolved")] += int(
            record.get("focused_tests_passed", False)
        )
        lanes[(mode, lane, "elapsed")] += record.get("elapsed_seconds", 0)
        lanes[(mode, lane, "model_calls")] += record.get("model_calls", 0) or 0
        for event in record.get("routing_events", []):
            route = event.get("routed_model") or event.get("response_model")
            if route:
                routes[(mode, route)] += 1
    print(
        json.dumps(
            {
                "modes": {
                    mode: {
                        "cases": summary[(mode, "cases")],
                        "focused_tests_resolved": summary[(mode, "resolved")],
                        "elapsed_seconds": elapsed[mode],
                        "routed_calls": {
                            route: count
                            for (route_mode, route), count in routes.items()
                            if route_mode == mode
                        },
                        "lanes": {
                            lane: {
                                "cases": lanes[(mode, lane, "cases")],
                                "focused_tests_resolved": lanes[
                                    (mode, lane, "resolved")
                                ],
                                "mean_elapsed_seconds": (
                                    lanes[(mode, lane, "elapsed")]
                                    / lanes[(mode, lane, "cases")]
                                ),
                                "model_calls": lanes[(mode, lane, "model_calls")],
                            }
                            for lane in sorted(
                                {
                                    (
                                        record_mode
                                        if record_mode == "local"
                                        else "cloud"
                                        if record_mode.startswith("cloud")
                                        else record.get("expected_route", "unknown")
                                    )
                                    for (record_mode, _), record in latest.items()
                                    if record_mode == mode
                                }
                            )
                        },
                    }
                    for mode in sorted({mode for mode, _ in latest})
                }
            },
            indent=2,
        )
    )


def verify_baseline(
    record: dict[str, Any], mode: str, force: bool
) -> dict[str, Any]:
    worktree = prepare_worktree(record, mode, force)
    environment = prepare_environment(record, mode, worktree, force)
    clean_worktree(record, worktree)
    refresh_editable(record, worktree, environment)
    if not apply_patch(worktree, record["test_patch"]):
        raise RuntimeError(f"Could not apply test patch for {record['instance_id']}")
    test = run_tests(record, worktree, environment)
    output = (test.stdout + test.stderr)[-4000:]
    clean_worktree(record, worktree)
    expected_failure = test.returncode == 1 and "FAILED " in output
    return {
        "instance_id": record["instance_id"],
        "baseline_exit_code": test.returncode,
        "expected_failure": expected_failure,
        "output": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("classify", "prepare", "baseline", "run", "rescore", "summarize"),
    )
    parser.add_argument("--mode", choices=tuple(MODEL_BY_MODE), default="hybrid")
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.command == "classify":
        classify_all()
        return
    if args.command == "summarize":
        summarize()
        return
    records = select_cases(args.case_ids)
    if args.command == "prepare":
        for record in records:
            worktree = prepare_worktree(record, args.mode, args.force)
            environment = prepare_environment(
                record, args.mode, worktree, args.force
            )
            print(f"prepared {record['instance_id']} env={environment}")
        return
    if args.command == "baseline":
        for record in records:
            result = verify_baseline(record, args.mode, args.force)
            print(json.dumps(result, ensure_ascii=True), flush=True)
        return
    if args.command == "rescore":
        for record in records:
            result = rescore_case(record, args.mode)
            print(json.dumps(result, ensure_ascii=True), flush=True)
        return
    for record in records:
        print(f"running {args.mode} {record['instance_id']}", flush=True)
        try:
            result = execute_case(record, args.mode, args.force)
        except Exception as error:  # noqa: BLE001
            result = {
                "instance_id": record["instance_id"],
                "mode": args.mode,
                "status": "harness_error",
                "error": str(error),
            }
            append_result(result)
        print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()