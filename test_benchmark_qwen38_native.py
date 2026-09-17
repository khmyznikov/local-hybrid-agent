"""CPU-only tests for native benchmark checks, prompts, and report comparisons."""

import json
import unittest
from copy import deepcopy

from benchmark_qwen38_native import EXPECTED, TOOLS, check_output, make_cases
from compare_qwen38_native import compare, first_difference


class CorrectnessChecks(unittest.TestCase):
    def test_arithmetic(self):
        self.assertTrue(check_output("arithmetic", "408\n"))
        self.assertFalse(check_output("arithmetic", "The answer is 408."))

    def test_sort(self):
        self.assertTrue(check_output("sort", "[-4,0,2,2,9]"))
        self.assertFalse(check_output("sort", "[-4,0,2,9]"))

    def test_extraction(self):
        self.assertTrue(check_output("extract", json.dumps(EXPECTED)))
        self.assertFalse(
            check_output("extract", json.dumps({**EXPECTED, "database_port": "5432"}))
        )
        self.assertFalse(
            check_output("extract", f"```json\n{json.dumps(EXPECTED)}\n```")
        )

    def test_json_tool(self):
        body = json.dumps({"name": "execute_cli", "arguments": EXPECTED})
        self.assertTrue(check_output("tool", f"<tool_call>{body}</tool_call>"))
        self.assertFalse(
            check_output("tool", f"Explanation\n<tool_call>{body}</tool_call>")
        )

    def test_xml_tool(self):
        parameters = "\n".join(
            f"<parameter={k}>\n{v}\n</parameter>" for k, v in EXPECTED.items()
        )
        text = f"<tool_call>\n<function=execute_cli>\n{parameters}\n</function>\n</tool_call>"
        self.assertTrue(check_output("tool", text))
        self.assertFalse(check_output("tool", text.replace("edge-rtr-7", "edge-rtr-8")))
        self.assertFalse(check_output("tool", text + text))
        self.assertFalse(
            check_output("tool", text.replace("</function>", "extra</function>"))
        )


class FirstDifferenceChecks(unittest.TestCase):
    def test_identical_sequences(self):
        for tokens in ([], [11], [11, 22, 33]):
            with self.subTest(tokens=tokens):
                self.assertIsNone(first_difference(tokens, list(tokens)))

    def test_first_mismatching_token(self):
        for candidate, expected in (
            ([99, 22, 33], 0),
            ([11, 99, 88], 1),
            ([11, 22, 99], 2),
        ):
            with self.subTest(candidate=candidate):
                self.assertEqual(first_difference([11, 22, 33], candidate), expected)
                self.assertEqual(first_difference(candidate, [11, 22, 33]), expected)

    def test_strict_prefixes_in_both_directions(self):
        for prefix in ([], [11], [11, 22]):
            with self.subTest(prefix=prefix):
                self.assertEqual(first_difference(prefix, [11, 22, 33]), len(prefix))
                self.assertEqual(first_difference([11, 22, 33], prefix), len(prefix))


class FakeTokenizer:
    """Reversible character IDs; tokenize=True mimics a batched mapping result."""

    def __init__(self):
        self.template_calls = []
        self.encode_calls = []

    def apply_chat_template(self, messages, **kwargs):
        text = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        if kwargs.get("tools"):
            text += f"\nTools: {json.dumps(kwargs['tools'])}"
        if kwargs.get("add_generation_prompt"):
            text += "\nassistant:"
        self.template_calls.append((text, kwargs))
        if kwargs.get("tokenize", True):
            return {"input_ids": [[ord(character) for character in text]]}
        return text

    def encode(self, text, *, add_special_tokens):
        self.encode_calls.append((text, add_special_tokens))
        return [ord(character) for character in text]

    def decode(self, ids):
        return "".join(chr(token) for token in ids)


class MakeCasesChecks(unittest.TestCase):
    def assert_flat_encoded(self, cases, tokenizer):
        self.assertTrue(tokenizer.template_calls)
        for text, options in tokenizer.template_calls:
            self.assertIs(options["tokenize"], False)
            self.assertIs(options["add_generation_prompt"], True)
            self.assertIs(options["enable_thinking"], False)
            self.assertIn((text, False), tokenizer.encode_calls)
        for _, add_special_tokens in tokenizer.encode_calls:
            self.assertIs(add_special_tokens, False)
        rendered = [text for text, _ in tokenizer.template_calls]
        for case in cases:
            with self.subTest(case=case["name"]):
                self.assertIs(type(case["ids"]), list)
                self.assertTrue(case["ids"])
                self.assertTrue(all(type(token) is int for token in case["ids"]))
                self.assertIn(tokenizer.decode(case["ids"]), rendered)

    def test_smoke_returns_only_flat_arithmetic_ids(self):
        tokenizer = FakeTokenizer()
        cases = make_cases(tokenizer, [2048], True)
        self.assertEqual([case["name"] for case in cases], ["arithmetic"])
        self.assertEqual(cases[0]["kind"], "arithmetic")
        self.assertEqual(len(tokenizer.template_calls), 1)
        self.assertIsNone(tokenizer.template_calls[0][1]["tools"])
        self.assert_flat_encoded(cases, tokenizer)

    def test_no_contexts_keeps_non_tool_cases(self):
        tokenizer = FakeTokenizer()
        cases = make_cases(tokenizer, [], False)
        self.assertEqual(
            [case["name"] for case in cases],
            ["arithmetic", "sort", "extract", "decode"],
        )
        self.assertIsNone(cases[-1]["kind"])
        for _, options in tokenizer.template_calls:
            self.assertIsNone(options["tools"])
        self.assert_flat_encoded(cases, tokenizer)

    def test_context_cases_have_flat_ids_and_tool_schema(self):
        tokenizer = FakeTokenizer()
        contexts = [2048, 4096]
        cases = make_cases(tokenizer, contexts, False)
        self.assertEqual(
            [case["name"] for case in cases],
            ["arithmetic", "sort", "extract", "tool_2048", "tool_4096", "decode"],
        )
        for case, target in zip(cases[3:-1], contexts):
            self.assertEqual(case["kind"], "tool")
            self.assertLessEqual(abs(len(case["ids"]) - target), 8)
        tool_calls = [
            options
            for _, options in tokenizer.template_calls
            if options["tools"] is not None
        ]
        self.assertTrue(tool_calls)
        for options in tool_calls:
            self.assertIs(options["tools"], TOOLS)
        self.assert_flat_encoded(cases, tokenizer)


def completed_report(batch_size=2):
    """Minimal synthetic completed report containing every field compare reads."""
    return {
        "status": "completed",
        "arguments": {
            "model": "synthetic-target",
            "mode": "baseline",
            "graph": "NONE",
            "output_tokens": 16,
            "batch_size": batch_size,
            "prefill_chunk": 1024,
            "kv_gib": 1.0,
        },
        "prompt_manifest": [
            {"name": "synthetic_canary", "tokens": 8, "sha256": "prompt-hash"}
        ],
        "chat_template_sha256": "template-hash",
        "environment": {"config_sha256": "config-hash"},
        "records": [
            {
                "case": "synthetic_canary",
                "repeat": repeat,
                "slot": slot,
                "output_token_ids": [11, 22 + slot, 33],
                "correctness_passed": True,
                "wall_seconds_batch": wall,
                "ttft_seconds": ttft,
            }
            for repeat, (wall, ttft) in enumerate([(2.0, 0.4), (4.0, 0.8), (10.0, 1.2)])
            for slot in range(batch_size)
        ],
        "metrics_before": [],
        "metrics_after": [],
        "summary": {"repeat_exact_match": True, "correctness_passed": True},
    }


class ComparisonChecks(unittest.TestCase):
    def setUp(self):
        self.baseline = completed_report()
        self.candidate = deepcopy(self.baseline)
        self.candidate["arguments"]["mode"] = "dflash"
        self.candidate["arguments"]["graph"] = "FULL_DECODE_ONLY"
        for row in self.candidate["records"]:
            row["wall_seconds_batch"] /= 2
            row["ttft_seconds"] /= 2

    def test_matching_reports_qualify_speedup_without_mutation(self):
        original = deepcopy((self.baseline, self.candidate))
        result = compare(self.baseline, self.candidate)
        self.assertTrue(result["comparable"])
        self.assertTrue(result["all_greedy_tokens_match"])
        self.assertTrue(result["both_runs_repeat_stable"])
        self.assertTrue(result["both_runs_pass_canaries"])
        self.assertTrue(result["speedup_correctness_qualified"])
        self.assertEqual(len(result["checks"]), 6)
        for check in result["checks"]:
            self.assertTrue(check["exact_token_match"])
            self.assertIsNone(check["first_difference_index"])
            self.assertTrue(check["candidate_correctness"])
        self.assertEqual(
            result["performance"],
            [
                {
                    "case": "synthetic_canary",
                    "baseline_wall_median_seconds": 4.0,
                    "candidate_wall_median_seconds": 2.0,
                    "wall_speedup": 2.0,
                    "baseline_ttft_median_seconds": 0.8,
                    "candidate_ttft_median_seconds": 0.4,
                    "baseline_samples": 3,
                    "candidate_samples": 3,
                }
            ],
        )
        self.assertEqual((self.baseline, self.candidate), original)

    def test_incomplete_run_on_either_side_is_not_comparable(self):
        for side in range(2):
            for status in ("starting", "loading", "testing", "error"):
                with self.subTest(side=side, status=status):
                    reports = deepcopy((self.baseline, self.candidate))
                    reports[side]["status"] = status
                    self.assertEqual(
                        compare(*reports),
                        {
                            "comparable": False,
                            "reason": "At least one run did not complete",
                        },
                    )

    def test_mismatched_inputs_on_either_side_are_rejected(self):
        changes = [
            (("arguments", "model"), "other-target", "model"),
            (("arguments", "output_tokens"), 32, "output_tokens"),
            (("arguments", "batch_size"), 1, "batch_size"),
            (("arguments", "prefill_chunk"), 512, "prefill_chunk"),
            (("arguments", "kv_gib"), 2.0, "kv_gib"),
            (("prompt_manifest",), [], "prompt_manifest"),
            (("chat_template_sha256",), "other-template", "chat_template_sha256"),
            (("environment", "config_sha256"), "other-config", "model_config"),
        ]
        for side in range(2):
            for path, value, difference in changes:
                with self.subTest(side=side, difference=difference):
                    reports = deepcopy((self.baseline, self.candidate))
                    container = reports[side]
                    for key in path[:-1]:
                        container = container[key]
                    container[path[-1]] = value
                    self.assertEqual(
                        compare(*reports),
                        {
                            "comparable": False,
                            "reason": "Mismatched inputs",
                            "differences": [difference],
                        },
                    )

    def test_token_mismatch_or_length_drift_disqualifies_speedup(self):
        for tokens, difference in (
            ([11, 99, 33], 1),
            ([11, 23], 2),
            ([11, 23, 33, 44], 3),
        ):
            with self.subTest(tokens=tokens):
                candidate = deepcopy(self.candidate)
                for row in candidate["records"]:
                    if row["slot"] == 1:
                        row["output_token_ids"] = list(tokens)
                result = compare(self.baseline, candidate)
                self.assertTrue(result["comparable"])
                self.assertTrue(result["both_runs_repeat_stable"])
                self.assertTrue(result["both_runs_pass_canaries"])
                self.assertFalse(result["all_greedy_tokens_match"])
                self.assertFalse(result["speedup_correctness_qualified"])
                for check in result["checks"]:
                    self.assertEqual(check["exact_token_match"], check["slot"] == 0)
                    self.assertEqual(
                        check["first_difference_index"],
                        difference if check["slot"] == 1 else None,
                    )

    def test_later_repeats_are_checked_against_baseline_repeat_zero(self):
        for report in (self.baseline, self.candidate):
            report["records"][-1]["output_token_ids"] = [11, 99, 33]
            report["summary"]["repeat_exact_match"] = False
        result = compare(self.baseline, self.candidate)
        last_check = result["checks"][-1]
        self.assertEqual((last_check["repeat"], last_check["slot"]), (2, 1))
        self.assertFalse(last_check["exact_token_match"])
        self.assertEqual(last_check["first_difference_index"], 1)
        self.assertFalse(result["speedup_correctness_qualified"])

    def test_unstable_or_failed_canaries_on_either_side_disqualify_speedup(self):
        for side in range(2):
            for summary_key, result_key in (
                ("repeat_exact_match", "both_runs_repeat_stable"),
                ("correctness_passed", "both_runs_pass_canaries"),
            ):
                with self.subTest(side=side, summary_key=summary_key):
                    reports = deepcopy((self.baseline, self.candidate))
                    reports[side]["summary"][summary_key] = False
                    result = compare(*reports)
                    self.assertTrue(result["comparable"])
                    self.assertTrue(result["all_greedy_tokens_match"])
                    self.assertFalse(result[result_key])
                    self.assertFalse(result["speedup_correctness_qualified"])

    def test_acceptance_uses_counter_deltas_excluding_warmup(self):
        drafted = "vllm:spec_decode_num_draft_tokens"
        accepted = "vllm:spec_decode_num_accepted_tokens"
        drafts = "vllm:spec_decode_num_drafts"
        self.candidate["metrics_before"] = [
            {"name": drafted, "value": 100},
            {"name": accepted, "value": 60},
        ]
        self.candidate["metrics_after"] = [
            {"name": drafted, "value": 120},
            {"name": accepted, "value": 75},
            {"name": drafts, "value": 10},
            {"name": "vllm:unrelated_counter", "value": 999},
            {"name": "vllm:spec_decode_histogram", "buckets": [1, 2]},
        ]
        result = compare(self.baseline, self.candidate)
        self.assertEqual(
            result["speculative_counter_deltas"],
            {drafted: 20, accepted: 15, drafts: 10},
        )
        self.assertEqual(result["draft_acceptance_fraction"], 0.75)

    def test_missing_or_zero_draft_delta_has_no_acceptance_fraction(self):
        drafted = "vllm:spec_decode_num_draft_tokens"
        for metrics in ([], [{"name": drafted, "value": 100}]):
            with self.subTest(metrics=metrics):
                candidate = deepcopy(self.candidate)
                candidate["metrics_before"] = deepcopy(metrics)
                candidate["metrics_after"] = deepcopy(metrics)
                result = compare(self.baseline, candidate)
                self.assertIsNone(result["draft_acceptance_fraction"])
                self.assertEqual(
                    result["speculative_counter_deltas"],
                    {drafted: 0} if metrics else {},
                )


if __name__ == "__main__":
    unittest.main()
