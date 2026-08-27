import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import common


def execute_case(record, arm, force):
    worktree, environment = common.prepare_case(record, arm, force)
    common.require(
        common.bench.apply_patch(worktree, record["test_patch"]),
        f"Could not apply official test patch for {record['instance_id']}",
    )
    baseline = common.bench.run_tests(record, worktree, environment)
    baseline_output = (baseline.stdout + baseline.stderr)[-12000:]
    common.require(
        baseline.returncode == 1 and "FAILED " in baseline_output,
        f"Baseline did not fail for {record['instance_id']}",
    )
    common.bench.clean_worktree(record, worktree)
    common.bench.refresh_editable(record, worktree, environment)

    run_dir = common.RESULTS / arm / record["instance_id"]
    if force:
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = common.write_settings(run_dir, arm)
    stream_path = run_dir / "stream.jsonl"
    stderr_path = run_dir / "stderr.log"
    route_slice = run_dir / "litellm-routes.jsonl"
    offset = common.route_offset()
    (run_dir / "litellm-route-offset.txt").write_text(
        str(offset), encoding="utf-8"
    )

    config_dir = (
        common.STATE_HOME
        / "openclaude-config/dispatch"
        / arm
        / record["instance_id"]
    )
    shutil.rmtree(config_dir, ignore_errors=True)
    config_dir.mkdir(parents=True, mode=0o700)
    env = os.environ.copy()
    env.update(
        {
            "PATH": (
                f"{environment / 'bin'}:{Path.home() / '.local/bin'}:"
                f"{env['PATH']}"
            ),
            "VIRTUAL_ENV": str(environment),
            "OPENCLAUDE_CONFIG_DIR": str(config_dir),
            "OPENCLAUDE_DISABLE_AGENT_WORKTREE_ISOLATION": "1",
            "CLAUDE_CODE_USE_OPENAI": "1",
            "OPENAI_BASE_URL": common.LITELLM_URL,
            "OPENAI_API_KEY": common.LITELLM_KEY,
            "OPENAI_MODEL": common.PARENT_ALIAS,
        }
    )
    command = [
        str(common.NODE),
        str(common.CLI),
        "--print",
        common.make_prompt(record),
        "--provider",
        "openai",
        "--model",
        common.PARENT_ALIAS,
        "--settings",
        str(settings),
        "--agents",
        common.AGENT_CONFIG.read_text(encoding="utf-8"),
        "--allowedTools",
        *common.PARENT_ALLOWED_TOOLS,
        "--permission-mode",
        "dontAsk",
        "--no-session-persistence",
        "--output-format",
        "stream-json",
        "--max-turns",
        common.PARENT_MAX_TURNS,
        "--verbose",
    ]
    started = time.perf_counter()
    try:
        process = common.run(
            command,
            cwd=worktree,
            env=env,
            timeout=900,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or "Agent timed out after 900 seconds."
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        process = subprocess.CompletedProcess(command, 124, stdout, stderr)
    elapsed = time.perf_counter() - started
    stream_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")
    if process.returncode and not process.stdout.strip():
        detail = process.stderr.strip() or f"exit code {process.returncode}"
        raise RuntimeError(f"OpenClaude failed before streaming: {detail}")
    routes = common.wait_for_routes(offset, route_slice)
    events = common.parse_events(process.stdout)

    patch = common.run(
        ["git", "diff", "--binary", record["base_commit"]],
        cwd=worktree,
    ).stdout
    (run_dir / "agent.patch").write_text(patch, encoding="utf-8")
    restored = common.bench.restore_official_test_files(record, worktree)
    test_patch_applied = common.bench.apply_patch(
        worktree, record["test_patch"]
    )
    if test_patch_applied:
        test = common.bench.run_tests(record, worktree, environment)
        test_output = (test.stdout + test.stderr)[-12000:]
        test_exit_code = test.returncode
    else:
        test_output = "Official test patch conflicted with agent changes."
        test_exit_code = None
    (run_dir / "focused-test.log").write_text(
        test_output, encoding="utf-8"
    )

    agent_files = common.bench.changed_files(worktree, record["base_commit"])
    test_files = common.bench.patch_files(record["test_patch"])
    implementation_files = [
        name for name in agent_files if name not in test_files
    ]
    gold_files = common.bench.patch_files(record["patch"])
    metrics = common.parse_metrics(events, arm, routes)
    result = {
        "instance_id": record["instance_id"],
        "repo": record["repo"],
        "difficulty": record["difficulty"],
        "arm": arm,
        "status": "complete" if process.returncode == 0 else "agent_failed",
        "agent_exit_code": process.returncode,
        "elapsed_seconds": elapsed,
        "baseline_exit_code": baseline.returncode,
        "test_patch_applied": test_patch_applied,
        "focused_tests_passed": test_exit_code == 0,
        "test_exit_code": test_exit_code,
        "changed_files": implementation_files,
        "gold_files": sorted(gold_files),
        "gold_file_recall": (
            len(gold_files.intersection(implementation_files))
            / len(gold_files)
            if gold_files
            else None
        ),
        "restored_agent_test_files": restored,
        "metrics": metrics,
    }
    common.append_record(common.RESULTS_LOG, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "baseline", "run"))
    parser.add_argument("--arm", choices=tuple(common.MODE_DIR), action="append")
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "run":
        common.validate_runtime()
    arms = args.arm or list(common.MODE_DIR)
    for record in common.load_cases(args.case_ids):
        for arm in arms:
            if args.command == "prepare":
                worktree, environment = common.prepare_case(
                    record, arm, args.force
                )
                print(
                    f"prepared {arm} {record['instance_id']} "
                    f"env={environment}",
                    flush=True,
                )
            elif args.command == "baseline":
                print(
                    json.dumps(
                        common.baseline_case(record, arm, args.force)
                    ),
                    flush=True,
                )
            else:
                print(f"running {arm} {record['instance_id']}", flush=True)
                try:
                    result = execute_case(record, arm, args.force)
                except Exception as error:  # noqa: BLE001
                    result = {
                        "instance_id": record["instance_id"],
                        "arm": arm,
                        "status": "harness_error",
                        "error": str(error),
                    }
                    common.append_record(common.RESULTS_LOG, result)
                print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
