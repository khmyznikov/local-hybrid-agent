"""Compare native baseline/speculative outputs before interpreting speedups."""

import argparse
import json
import statistics
from pathlib import Path


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def compare(baseline: dict, candidate: dict) -> dict:
    if baseline["status"] != "completed" or candidate["status"] != "completed":
        return {"comparable": False, "reason": "At least one run did not complete"}
    differences = [
        key
        for key in ("model", "output_tokens", "batch_size", "prefill_chunk", "kv_gib")
        if baseline["arguments"][key] != candidate["arguments"][key]
    ]
    if baseline["prompt_manifest"] != candidate["prompt_manifest"]:
        differences.append("prompt_manifest")
    if baseline["chat_template_sha256"] != candidate["chat_template_sha256"]:
        differences.append("chat_template_sha256")
    if (
        baseline["environment"]["config_sha256"]
        != candidate["environment"]["config_sha256"]
    ):
        differences.append("model_config")
    if differences:
        return {
            "comparable": False,
            "reason": "Mismatched inputs",
            "differences": differences,
        }
    reference = {
        (r["case"], r["slot"]): r for r in baseline["records"] if r["repeat"] == 0
    }
    checks = []
    for row in candidate["records"]:
        ref = reference[(row["case"], row["slot"])]
        checks.append(
            {
                "case": row["case"],
                "repeat": row["repeat"],
                "slot": row["slot"],
                "exact_token_match": row["output_token_ids"] == ref["output_token_ids"],
                "first_difference_index": first_difference(
                    ref["output_token_ids"], row["output_token_ids"]
                ),
                "candidate_correctness": row["correctness_passed"],
            }
        )
    performance = []
    for case in reference:
        if case[1] != 0:
            continue
        samples = [
            [r for r in report["records"] if r["case"] == case[0] and r["slot"] == 0]
            for report in (baseline, candidate)
        ]
        wall = [
            statistics.median(r["wall_seconds_batch"] for r in rows) for rows in samples
        ]
        ttft = [statistics.median(r["ttft_seconds"] for r in rows) for rows in samples]
        performance.append(
            {
                "case": case[0],
                "baseline_wall_median_seconds": wall[0],
                "candidate_wall_median_seconds": wall[1],
                "wall_speedup": wall[0] / wall[1],
                "baseline_ttft_median_seconds": ttft[0],
                "candidate_ttft_median_seconds": ttft[1],
                "baseline_samples": len(samples[0]),
                "candidate_samples": len(samples[1]),
            }
        )
    accepted = {}
    for stage in ("before", "after"):
        accepted[stage] = {
            m["name"]: m["value"]
            for m in candidate[f"metrics_{stage}"]
            if "spec_decode" in m["name"] and "value" in m
        }
    deltas = {
        key: value - accepted["before"].get(key, 0)
        for key, value in accepted["after"].items()
    }
    drafted = deltas.get("vllm:spec_decode_num_draft_tokens", 0)
    all_exact = all(c["exact_token_match"] for c in checks)
    stable = (
        baseline["summary"]["repeat_exact_match"]
        and candidate["summary"]["repeat_exact_match"]
    )
    correct = (
        baseline["summary"]["correctness_passed"]
        and candidate["summary"]["correctness_passed"]
    )
    return {
        "comparable": True,
        "all_greedy_tokens_match": all_exact,
        "both_runs_repeat_stable": stable,
        "both_runs_pass_canaries": correct,
        "speedup_correctness_qualified": all_exact and stable and correct,
        "checks": checks,
        "performance": performance,
        "speculative_counter_deltas": deltas,
        "draft_acceptance_fraction": deltas.get(
            "vllm:spec_decode_num_accepted_tokens", 0
        )
        / drafted
        if drafted
        else None,
        "note": "Small local canaries, not a full-precision model accuracy evaluation; warm uncached runs.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(
        json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
