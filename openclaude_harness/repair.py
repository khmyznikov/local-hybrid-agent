import argparse
import json
import os
import shutil
import time
from pathlib import Path

import common


def latest_dispatch(instance_id: str, arm: str) -> dict:
    latest = None
    for line in common.RESULTS_LOG.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        if record.get("instance_id") == instance_id and record.get("arm") == arm:
            latest = record
    if latest is None or "metrics" not in latest:
        raise RuntimeError(f"No {arm} dispatch result for {instance_id}")
    return latest


def repair_case(record: dict, arm: str) -> dict:
    instance_id = record["instance_id"]
    initial = latest_dispatch(instance_id, arm)
    if initial["focused_tests_passed"]:
        return {
            "instance_id": instance_id,
            "arm": arm,
            "status": "not_needed",
            "focused_tests_passed": True,
        }

    worktree = common.LAB / "worktrees" / common.MODE_DIR[arm] / instance_id
    environment = common.shared_environment(record)
    run_dir = common.RESULTS / arm / instance_id
    failure = (run_dir / "focused-test.log").read_text(encoding="utf-8")
    prompt = (
        "Repair the existing implementation for the repository issue below. A "
        "bounded implementer already edited this isolated checkout, and the "
        "official focused test patch is present. Inspect the current git diff and "
        "failure output, make only the smallest correction needed, run the exact "
        "focused tests, then stop. Do not delegate, add unrelated tests, commit, "
        "read outside this checkout, or search for the upstream solution.\n\n"
        f"Issue:\n{record['problem_statement']}\n\n"
        f"Focused tests: {' '.join(record['FAIL_TO_PASS'])}\n\n"
        f"Failure output:\n{failure[-6000:]}"
    )

    config_dir = (
        common.STATE_HOME
        / "openclaude-config/repair"
        / arm
        / instance_id
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
            "CLAUDE_CODE_USE_OPENAI": "1",
            "OPENAI_BASE_URL": common.LITELLM_URL,
            "OPENAI_API_KEY": common.LITELLM_KEY,
            "OPENAI_MODEL": common.PARENT_ALIAS,
        }
    )
    settings = common.write_settings(
        run_dir, "control", "repair.settings.json"
    )
    offset = common.route_offset()
    command = [
        str(common.NODE),
        str(common.CLI),
        "--print",
        prompt,
        "--provider",
        "openai",
        "--model",
        common.PARENT_ALIAS,
        "--settings",
        str(settings),
        "--allowedTools",
        "Bash",
        "Read",
        "Grep",
        "Glob",
        "Edit",
        "Write",
        "--tools",
        "Bash,Read,Grep,Glob,Edit,Write",
        "--permission-mode",
        "dontAsk",
        "--no-session-persistence",
        "--output-format",
        "stream-json",
        "--max-turns",
        "8",
        "--verbose",
    ]
    started = time.perf_counter()
    process = common.run(command, cwd=worktree, env=env, timeout=600)
    elapsed = time.perf_counter() - started
    (run_dir / "repair.stream.jsonl").write_text(
        process.stdout, encoding="utf-8"
    )
    (run_dir / "repair.stderr.log").write_text(
        process.stderr, encoding="utf-8"
    )
    routes = common.wait_for_routes(
        offset, run_dir / "repair-litellm-routes.jsonl"
    )
    successes = [route for route in routes if route.get("status") == "success"]
    failures = [route for route in routes if route.get("status") == "failure"]
    cloud_routes = [
        route
        for route in successes
        if route.get("requested_model") == common.PARENT_UPSTREAM_MODEL
    ]
    unexpected = [route for route in successes if route not in cloud_routes]
    common.require(cloud_routes, f"{arm}: no GPT-5.4 repair routes")
    common.require(
        not failures and not unexpected,
        f"{arm}: repair failures={len(failures)}, unexpected={len(unexpected)}",
    )

    restored = common.bench.restore_official_test_files(record, worktree)
    common.require(
        common.bench.apply_patch(worktree, record["test_patch"]),
        f"Could not reapply official test patch for {instance_id}",
    )
    test = common.bench.run_tests(record, worktree, environment)
    test_output = (test.stdout + test.stderr)[-12000:]
    (run_dir / "repair-focused-test.log").write_text(
        test_output, encoding="utf-8"
    )
    patch = common.run(
        ["git", "diff", "--binary", record["base_commit"]],
        cwd=worktree,
    ).stdout
    (run_dir / "repaired.patch").write_text(patch, encoding="utf-8")
    result = {
        "instance_id": instance_id,
        "arm": arm,
        "dispatch_session_id": initial["metrics"]["session_id"],
        "status": "complete" if process.returncode == 0 else "repair_failed",
        "repair_exit_code": process.returncode,
        "elapsed_seconds": elapsed,
        "focused_tests_passed": test.returncode == 0,
        "test_exit_code": test.returncode,
        "restored_agent_test_files": restored,
        "repair_metrics": {
            "cloud_model_calls": len(cloud_routes),
            "cloud_usage": common.route_usage(cloud_routes),
            "litellm_failed_routes": len(failures),
        },
    }
    common.append_record(common.REPAIRS_LOG, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(common.MODE_DIR), action="append")
    parser.add_argument("--case", action="append", dest="case_ids")
    args = parser.parse_args()
    common.validate_runtime()
    arms = args.arm or list(common.MODE_DIR)
    for record in common.load_cases(args.case_ids):
        for arm in arms:
            result = repair_case(record, arm)
            print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
