import json

import common

CACHE_FRACTIONS = (0.0, 0.5, 0.75, 0.9)
OUTPUT = common.RESULTS / "policy-summary.json"


def latest_records(path, keys):
    latest = {}
    if not path.exists():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        if "metrics" not in record and record.get("status") == "harness_error":
            continue
        latest[tuple(record.get(key) for key in keys)] = record
    return latest


def add_usage(left, right):
    return {
        key: int((left or {}).get(key, 0) or 0)
        + int((right or {}).get(key, 0) or 0)
        for key in ("input_tokens", "cache_read_input_tokens", "output_tokens")
    }


def usage_cost(usage, cached_fraction):
    input_tokens = (
        usage.get("input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
    )
    input_rate = 2.50 * (1 - cached_fraction) + 0.25 * cached_fraction
    return (
        input_tokens * input_rate
        + usage.get("output_tokens", 0) * 15.00
    ) / 1_000_000


def policy_case(arm, instance_id, dispatch, repairs):
    run = dispatch.get((arm, instance_id))
    if run is None or "metrics" not in run:
        return None
    repair = repairs.get((arm, instance_id))
    if repair and repair.get("dispatch_session_id") != run["metrics"]["session_id"]:
        repair = None
    repair_used = (
        not run["focused_tests_passed"]
        and repair is not None
        and repair.get("status") != "not_needed"
    )
    repair_usage = (
        repair.get("repair_metrics", {}).get("cloud_usage")
        if repair_used
        else None
    )
    usage = add_usage(run["metrics"]["cloud_usage"], repair_usage)
    repair_calls = (
        int(repair.get("repair_metrics", {}).get("cloud_model_calls", 0) or 0)
        if repair_used
        else 0
    )
    return {
        "resolved": (
            repair["focused_tests_passed"]
            if repair_used
            else run["focused_tests_passed"]
        ),
        "repair_used": repair_used,
        "cloud_usage": usage,
        "cloud_tokens": sum(usage.values()),
        "cloud_calls": run["metrics"]["cloud_model_calls"] + repair_calls,
        "local_calls": run["metrics"]["local_usage"]["calls"],
        "local_tokens": run["metrics"]["local_usage"]["total_tokens"],
        "local_latency_ms": sum(
            request.get("latencyMs", 0) or 0
            for request in run["metrics"]["local_usage"]["requests"]
        ),
        "litellm_failed_routes": (
            int(run["metrics"].get("litellm_failed_routes", 0) or 0)
            + (
                int(
                    repair.get("repair_metrics", {}).get(
                        "litellm_failed_routes", 0
                    )
                    or 0
                )
                if repair_used
                else 0
            )
        ),
    }


def new_totals():
    return {
        "cases": 0,
        "resolved": 0,
        "repairs": 0,
        "cloud_usage": {
            "input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 0,
        },
        "cloud_tokens": 0,
        "cloud_calls": 0,
        "local_calls": 0,
        "local_tokens": 0,
        "local_latency_ms": 0,
        "litellm_failed_routes": 0,
    }


def accumulate(totals, result):
    totals["cases"] += 1
    totals["resolved"] += int(result["resolved"])
    totals["repairs"] += int(result["repair_used"])
    totals["cloud_usage"] = add_usage(
        totals["cloud_usage"], result["cloud_usage"]
    )
    for key in (
        "cloud_tokens",
        "cloud_calls",
        "local_calls",
        "local_tokens",
        "local_latency_ms",
        "litellm_failed_routes",
    ):
        totals[key] += result[key]


def cost_scenarios(local, cloud):
    electricity = local["local_electricity_usd_at_0_20_per_kwh"]
    scenarios = {}
    for fraction in CACHE_FRACTIONS:
        local_api = usage_cost(local["cloud_usage"], fraction)
        cloud_api = usage_cost(cloud["cloud_usage"], fraction)
        local_net = local_api + electricity
        scenarios[f"{int(fraction * 100)}%_cached_input"] = {
            "local_api_usd": local_api,
            "cloud_api_usd": cloud_api,
            "local_with_electricity_usd": local_net,
            "net_saved_usd": cloud_api - local_net,
            "net_reduction_percent": (
                (cloud_api - local_net) / cloud_api * 100
                if cloud_api
                else None
            ),
        }
    return scenarios


def main():
    selected = [record["instance_id"] for record in common.load_cases()]
    dispatch = latest_records(
        common.RESULTS_LOG, ("arm", "instance_id")
    )
    repairs = latest_records(
        common.REPAIRS_LOG, ("arm", "instance_id")
    )
    rows = []
    cloud = new_totals()
    local = new_totals()
    missing = []
    pending_repairs = []
    for instance_id in selected:
        for arm in common.MODE_DIR:
            run = dispatch.get((arm, instance_id))
            repair = repairs.get((arm, instance_id))
            has_matching_repair = bool(
                repair
                and run
                and "metrics" in run
                and repair.get("dispatch_session_id")
                == run["metrics"]["session_id"]
            )
            if (
                run is not None
                and "metrics" in run
                and not run["focused_tests_passed"]
                and not has_matching_repair
            ):
                pending_repairs.append(
                    {"arm": arm, "instance_id": instance_id}
                )
        cloud_case = policy_case(
            "control", instance_id, dispatch, repairs
        )
        local_case = policy_case("hybrid", instance_id, dispatch, repairs)
        if cloud_case is None or local_case is None:
            missing.append(instance_id)
            continue
        rows.append(
            {
                "instance_id": instance_id,
                "cloud_policy": cloud_case,
                "local_first_policy": local_case,
                "local_cloud_token_reduction_percent": (
                    (cloud_case["cloud_tokens"] - local_case["cloud_tokens"])
                    / cloud_case["cloud_tokens"]
                    * 100
                ),
            }
        )
        accumulate(cloud, cloud_case)
        accumulate(local, local_case)

    local["local_energy_kwh_upper_bound_at_90w"] = (
        local["local_latency_ms"] / 1000 * 90 / 3_600_000
    )
    local["local_electricity_usd_at_0_20_per_kwh"] = (
        local["local_energy_kwh_upper_bound_at_90w"] * 0.20
    )
    token_saving = cloud["cloud_tokens"] - local["cloud_tokens"]
    report = {
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "evaluation": "official FAIL_TO_PASS focused tests on WSL",
        "policy": (
            "Four-turn GPT-5.4 dispatcher, implementation child, and selective "
            "eight-turn GPT-5.4 repair, all pinned through LiteLLM"
        ),
        "complete": not missing and not pending_repairs,
        "missing_cases": missing,
        "pending_repairs": pending_repairs,
        "cases": rows,
        "cloud_implementer_policy": cloud,
        "local_first_policy": local,
        "comparison": {
            "quality_difference": local["resolved"] - cloud["resolved"],
            "cloud_tokens_saved": token_saving,
            "cloud_token_reduction_percent": (
                token_saving / cloud["cloud_tokens"] * 100
                if cloud["cloud_tokens"]
                else None
            ),
            "cloud_call_reduction": (
                cloud["cloud_calls"] - local["cloud_calls"]
            ),
            "cost_scenarios": cost_scenarios(local, cloud),
        },
        "cache_accounting": (
            "LiteLLM reports total prompt tokens but no cached-input breakdown; "
            "cost applies equal cache fractions to both policies."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
