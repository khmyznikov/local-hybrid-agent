import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    baseline = {
        record["name"]: record
        for record in json.loads(args.baseline.read_text(encoding="utf-8"))
    }
    candidate = {
        record["name"]: record
        for record in json.loads(args.candidate.read_text(encoding="utf-8"))
    }

    for name, baseline_record in baseline.items():
        candidate_record = candidate[name]
        baseline_ids = baseline_record["output_token_ids"]
        candidate_ids = candidate_record["output_token_ids"]
        compared_length = min(len(baseline_ids), len(candidate_ids))
        first_divergence = next(
            (
                index
                for index, (baseline_id, candidate_id) in enumerate(
                    zip(baseline_ids, candidate_ids, strict=False)
                )
                if baseline_id != candidate_id
            ),
            compared_length,
        )
        matching_positions = sum(
            baseline_id == candidate_id
            for baseline_id, candidate_id in zip(
                baseline_ids, candidate_ids, strict=False
            )
        )
        print(
            f"name={name} first_divergence={first_divergence} "
            f"matching_positions={matching_positions}/{compared_length} "
            f"acceptance={candidate_record['acceptance_rate']:.4f}"
        )
        observations = candidate_record.get("acceptance_observations", [])
        if first_divergence < compared_length and observations:
            output_offset = 0
            for step, (_, accepted_count) in enumerate(observations):
                step_output_count = accepted_count + 1
                if first_divergence < output_offset + step_output_count:
                    position_in_step = first_divergence - output_offset
                    role = (
                        "accepted_draft"
                        if position_in_step < accepted_count
                        else "target_bonus_or_replacement"
                    )
                    print(
                        f" divergence_step={step} "
                        f"accepted_in_step={accepted_count} "
                        f"position_in_step={position_in_step} role={role}"
                    )
                    break
                output_offset += step_output_count
        if first_divergence < compared_length:
            start = max(0, first_divergence - 5)
            end = min(compared_length, first_divergence + 8)
            print(
                " baseline "
                f"ids={baseline_ids[start:end]} "
                f"text={tokenizer.decode(baseline_ids[start:end])!r}"
            )
            print(
                " candidate "
                f"ids={candidate_ids[start:end]} "
                f"text={tokenizer.decode(candidate_ids[start:end])!r}"
            )


if __name__ == "__main__":
    main()