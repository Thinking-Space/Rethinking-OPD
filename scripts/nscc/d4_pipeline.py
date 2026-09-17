#!/usr/bin/env python3
"""Pure, testable contracts shared by the NSCC OPD D4 scripts."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Iterable, Sequence


DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the answer "
    "to the problem.\n\n"
)
DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'
EVAL_PROMPT_TEMPLATE = "{problem} Please reason step by step, and put your final answer within \\boxed{{}}."


def normalize_problem(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_problem(text: str) -> str:
    text = str(text)
    if text.startswith(DAPO_PREFIX) and text.endswith(DAPO_SUFFIX):
        text = text[len(DAPO_PREFIX) : -len(DAPO_SUFFIX)]
    return normalize_problem(text)


def freeze_indices(
    prompt_texts: Sequence[str],
    prompt_token_lengths: Sequence[int],
    test_problem_texts: Iterable[str],
    *,
    dev_size: int,
    seed: int,
    max_prompt_length: int,
) -> dict:
    """Select a deterministic dev split after filtering and exact de-duplication."""
    if len(prompt_texts) != len(prompt_token_lengths):
        raise ValueError("prompt texts and token lengths must have equal length")
    if dev_size <= 0:
        raise ValueError("dev_size must be positive")

    test_keys = {extract_problem(text) for text in test_problem_texts}
    seen: set[str] = set()
    eligible: list[int] = []
    overlong: list[int] = []
    overlaps: list[int] = []
    duplicates: list[int] = []

    for index, (text, token_length) in enumerate(zip(prompt_texts, prompt_token_lengths)):
        key = extract_problem(text)
        if int(token_length) > max_prompt_length:
            overlong.append(index)
            continue
        if key in test_keys:
            overlaps.append(index)
            continue
        if key in seen:
            duplicates.append(index)
            continue
        seen.add(key)
        eligible.append(index)

    if len(eligible) < dev_size:
        raise ValueError(f"only {len(eligible)} eligible rows for dev_size={dev_size}")

    dev_set = set(random.Random(seed).sample(eligible, dev_size))
    dev_indices = [index for index in eligible if index in dev_set]
    train_indices = [index for index in eligible if index not in dev_set]
    return {
        "source_rows": len(prompt_texts),
        "eligible_rows": len(eligible),
        "train_indices": train_indices,
        "dev_indices": dev_indices,
        "excluded_overlong_indices": overlong,
        "excluded_test_overlap_indices": overlaps,
        "excluded_duplicate_indices": duplicates,
    }


def timeline_evaluation_protocol() -> dict:
    return {
        "temperature": 0.7,
        "top_p": 0.95,
        "prompt_template": EVAL_PROMPT_TEMPLATE,
        "tasks": {
            "dev_256": {"n": 1, "max_tokens": 2048, "expected_problems": 256},
            "MATH-500": {"n": 4, "max_tokens": 7168, "expected_problems": 500},
            "AIME24": {"n": 16, "max_tokens": 7168, "expected_problems": 30},
        },
    }


def summarize_baseline_records(records: Sequence[dict], *, expected_problems: int, expected_n: int) -> dict:
    expected_records = expected_problems * expected_n
    if len(records) != expected_records:
        raise ValueError(f"expected {expected_records} records, found {len(records)}")

    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        grouped[int(record["example_id"])].append(record)
    if len(grouped) != expected_problems:
        raise ValueError(f"expected {expected_problems} problems, found {len(grouped)}")
    for example_id, items in grouped.items():
        sample_ids = {int(item["sample_id"]) for item in items}
        if len(items) != expected_n or sample_ids != set(range(expected_n)):
            raise ValueError(f"problem {example_id} has incomplete or duplicate sampling")

    correct = [bool(record["correct"]) for record in records]
    truncated = [str(record.get("finish_reason", "")) == "length" for record in records]
    missing_box = ["\\boxed" not in str(record.get("response", "")) for record in records]
    empty = [not str(record.get("response", "")).strip() for record in records]
    lengths = [int(record.get("response_tokens", 0)) for record in records]
    per_problem = [sum(bool(item["correct"]) for item in items) / expected_n for items in grouped.values()]
    return {
        "record_count": len(records),
        "problem_count": len(grouped),
        "samples_per_problem": expected_n,
        "pass_at_1_estimate": sum(per_problem) / len(per_problem),
        "correct_rollouts": sum(correct),
        "truncation_rate": sum(truncated) / len(records),
        "missing_box_rate": sum(missing_box) / len(records),
        "empty_response_rate": sum(empty) / len(records),
        "mean_response_tokens": sum(lengths) / len(lengths),
        "max_response_tokens": max(lengths),
    }


def _records_by_sample_key(
    records: Sequence[dict], *, expected_problems: int, expected_n: int
) -> dict[tuple[int, int], dict]:
    expected_records = expected_problems * expected_n
    if len(records) != expected_records:
        raise ValueError(f"expected {expected_records} records, found {len(records)}")
    keyed: dict[tuple[int, int], dict] = {}
    for record in records:
        key = (int(record["example_id"]), int(record["sample_id"]))
        if key in keyed:
            raise ValueError(f"duplicate sample key: {key}")
        keyed[key] = record
    return keyed


def _linear_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap_pass1(
    baseline_records: Sequence[dict],
    final_records: Sequence[dict],
    *,
    expected_problems: int,
    expected_n: int,
    replicates: int = 10_000,
    seed: int = 2026,
) -> dict:
    """Compare pass@1 estimates with paired resampling at the problem level."""
    if expected_problems <= 0 or expected_n <= 0:
        raise ValueError("expected_problems and expected_n must be positive")
    if replicates <= 0:
        raise ValueError("replicates must be positive")

    baseline = _records_by_sample_key(
        baseline_records,
        expected_problems=expected_problems,
        expected_n=expected_n,
    )
    final = _records_by_sample_key(
        final_records,
        expected_problems=expected_problems,
        expected_n=expected_n,
    )
    if set(baseline) != set(final):
        raise ValueError("paired sample keys differ")
    expected_keys = {
        (example_id, sample_id)
        for example_id in range(expected_problems)
        for sample_id in range(expected_n)
    }
    if set(baseline) != expected_keys:
        raise ValueError("records do not contain the expected problem/sample grid")

    for key in baseline:
        for field in ("problem", "answer"):
            if field in baseline[key] and field in final[key]:
                if str(baseline[key][field]) != str(final[key][field]):
                    raise ValueError(f"paired {field} differs for sample key {key}")

    baseline_by_problem: list[float] = []
    final_by_problem: list[float] = []
    differences: list[float] = []
    for example_id in range(expected_problems):
        baseline_score = sum(
            bool(baseline[(example_id, sample_id)]["correct"])
            for sample_id in range(expected_n)
        ) / expected_n
        final_score = sum(
            bool(final[(example_id, sample_id)]["correct"])
            for sample_id in range(expected_n)
        ) / expected_n
        baseline_by_problem.append(baseline_score)
        final_by_problem.append(final_score)
        differences.append(final_score - baseline_score)

    rng = random.Random(seed)
    bootstrapped = []
    for _ in range(replicates):
        bootstrapped.append(
            sum(differences[rng.randrange(expected_problems)] for _ in range(expected_problems))
            / expected_problems
        )
    lower = _linear_quantile(bootstrapped, 0.025)
    upper = _linear_quantile(bootstrapped, 0.975)
    classification = "inconclusive"
    if lower > 0:
        classification = "reliable_gain"
    elif upper < 0:
        classification = "reliable_drop"

    return {
        "baseline_pass_at_1_estimate": sum(baseline_by_problem) / expected_problems,
        "final_pass_at_1_estimate": sum(final_by_problem) / expected_problems,
        "difference": sum(differences) / expected_problems,
        "ci_95": [lower, upper],
        "classification": classification,
        "resampling_unit": "problem",
        "bootstrap_replicates": replicates,
        "seed": seed,
    }


def conclude_independent_evaluation(task_classes: dict[str, str]) -> dict:
    """Apply the timeline stop-loss rule to the two required held-out tasks."""
    required_tasks = ("MATH-500", "AIME24")
    allowed = {"reliable_gain", "reliable_drop", "inconclusive"}
    if set(task_classes) != set(required_tasks):
        raise ValueError(f"task classes must contain exactly {required_tasks}")
    if any(value not in allowed for value in task_classes.values()):
        raise ValueError(f"task classes must be one of {sorted(allowed)}")

    if all(task_classes[task] == "reliable_gain" for task in required_tasks):
        verdict = "REPRODUCED_RELIABLE_GAIN"
        stop_loss_triggered = False
    elif any(task_classes[task] == "reliable_drop" for task in required_tasks):
        verdict = "NOT_REPRODUCED_RELIABLE_DROP"
        stop_loss_triggered = True
    else:
        verdict = "INCONCLUSIVE_NO_JOINT_RELIABLE_GAIN"
        stop_loss_triggered = True
    return {
        "verdict": verdict,
        "stop_loss_triggered": stop_loss_triggered,
        "required_tasks": list(required_tasks),
        "task_classifications": dict(task_classes),
        "rule": "both required tasks must have paired 95% CI lower bound above zero",
    }
