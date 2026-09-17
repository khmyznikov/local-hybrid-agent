"""Diagnose a recorded greedy mismatch with target-only token probabilities.

Compare autoregressive decode with re-prefilling the identical accepted prefix.
This can reveal shape-sensitive target numerics independently of a drafter, but
does not establish full-precision accuracy or prove a particular kernel faulty.
"""

import argparse
import json
import os
from pathlib import Path

from benchmark_qwen38_native import digest, make_cases, save
from compare_qwen38_native import first_difference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    base = next(r for r in baseline["records"] if r["case"] == "decode")
    draft = next(r for r in candidate["records"] if r["case"] == "decode")
    difference = first_difference(base["output_token_ids"], draft["output_token_ids"])
    if difference is None:
        raise ValueError("No greedy divergence to inspect")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        baseline["arguments"]["model"], local_files_only=True
    )
    case = next(c for c in make_cases(tokenizer, [], False) if c["name"] == "decode")
    assert (
        digest(json.dumps(case["ids"]))
        == base["prompt_sha256"]
        == draft["prompt_sha256"]
    )
    common = base["output_token_ids"][:difference]
    assert common == draft["output_token_ids"][:difference]
    kwargs = dict(baseline["engine_kwargs"])
    kwargs["enforce_eager"] = True
    kwargs.pop("compilation_config", None)
    report = {
        "status": "starting",
        "divergence_index": difference,
        "baseline_token": base["output_token_ids"][difference],
        "candidate_token": draft["output_token_ids"][difference],
        "common_prefix_text": tokenizer.decode(common),
        "records": [],
    }
    save(args.output, report)
    llm = LLM(**kwargs)
    try:
        for repeat in range(2):
            for name, ids, count, offset in [
                ("autoregressive", case["ids"], difference + 1, difference),
                ("reprefill_identical_prefix", case["ids"] + common, 1, 0),
            ]:
                result = llm.generate(
                    [{"prompt_token_ids": ids}],
                    SamplingParams(
                        temperature=0, max_tokens=count, ignore_eos=True, logprobs=10
                    ),
                    use_tqdm=False,
                )[0].outputs[0]
                token = result.token_ids[offset]
                record = {
                    "mode": name,
                    "repeat": repeat,
                    "selected_token": token,
                    "selected_text": tokenizer.decode([token]),
                    "top_logprobs": [
                        {
                            "token": k,
                            "text": tokenizer.decode([k]),
                            "logprob": v.logprob,
                            "rank": v.rank,
                        }
                        for k, v in result.logprobs[offset].items()
                    ],
                }
                if name == "autoregressive":
                    record["shared_prefix_exact"] = (
                        list(result.token_ids[:difference]) == common
                    )
                report["records"].append(record)
                print(json.dumps(record), flush=True)
                save(args.output, report)
        report["status"] = "completed"
    finally:
        llm.llm_engine.engine_core.shutdown()
        save(args.output, report)


if __name__ == "__main__":
    main()
