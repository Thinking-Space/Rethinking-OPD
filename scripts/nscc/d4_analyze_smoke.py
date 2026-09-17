#!/usr/bin/env python3
"""Analyze D4 smoke logs against the predeclared Timeline G1 acceptance contract."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re


UPSTREAM_PROBABILITY_METRICS = (
    "training/rollout_probs_diff_valid",
    "training/rollout_probs_diff_max",
    "training/rollout_probs_diff_mean",
    "training/rollout_probs_diff_std",
    "training/rollout_actor_probs_pearson_corr",
)

STUDENT_TEACHER_ALIGNMENT_METRICS = (
    "actor/entropy",
    "teacher/entropy",
    "val-topk/overlap_ratio",
    "val-topk/adv_intersection",
)

RAW_LOG_PROBABILITY_METRICS = {
    "max": "training/rollout_log_probs_diff_max",
    "mean": "training/rollout_log_probs_diff_mean",
    "std": "training/rollout_log_probs_diff_std",
}


def _update_metric_record(record: dict[str, float], payload: str) -> None:
    for part in payload.split(" - "):
        key, separator, raw = part.partition(":")
        if not separator:
            continue
        try:
            record[key.strip()] = float(raw.strip())
        except ValueError:
            continue


def _metrics(text: str) -> list[dict[str, float]]:
    records = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.search(r"(?:^|\s)step:(\d+)\s+-\s+(.+)$", line)
        if not match:
            continue
        record = {"step": float(match.group(1))}
        _update_metric_record(record, match.group(2))
        # Ray's terminal progress output can interrupt one long metrics line and
        # place its suffix on the next TaskRunner line. Recover only adjacent
        # timing/performance fragments, stopping before the next OPD step marker.
        for continuation in lines[index + 1 : index + 5]:
            if re.search(r"(?:^|\s)step:\d+\s+-", continuation):
                break
            if "OPD_ROLLOUT_WEIGHT_SYNC" in continuation:
                break
            if "timing_s/" not in continuation and "perf/" not in continuation:
                continue
            _update_metric_record(record, continuation)
        if "training/global_step" in record:
            records.append(record)
    return records


def _markers(text: str, name: str) -> list[dict]:
    records = []
    for line in text.splitlines():
        position = line.find(name)
        if position < 0:
            continue
        payload = line[position + len(name) :].strip()
        records.append(json.loads(payload))
    return records


def _step_list(records: list[dict[str, float]]) -> list[int]:
    return [int(record["training/global_step"]) for record in records]


def _assert_finite(records: list[dict[str, float]], key: str) -> None:
    missing = [int(record["step"]) for record in records if key not in record]
    if missing:
        raise RuntimeError(f"missing {key} at steps {missing}")
    invalid = [int(record["step"]) for record in records if not math.isfinite(record[key])]
    if invalid:
        raise RuntimeError(f"non-finite {key} at steps {invalid}")


def _diagnostics(
    records: list[dict[str, float]], *, include_top_k: bool = True
) -> dict:
    upstream = []
    alignment = []
    raw_log_probability = []
    for record in records:
        step = int(record["training/global_step"])
        upstream.append(
            {
                "step": step,
                "valid": int(record["training/rollout_probs_diff_valid"]),
                "max": record["training/rollout_probs_diff_max"],
                "mean": record["training/rollout_probs_diff_mean"],
                "std": record["training/rollout_probs_diff_std"],
                "pearson": record["training/rollout_actor_probs_pearson_corr"],
            }
        )
        alignment_record = {
            "step": step,
            "student_entropy": record["actor/entropy"],
            "teacher_entropy": record["teacher/entropy"],
            "student_minus_teacher_entropy": (
                record["actor/entropy"] - record["teacher/entropy"]
            ),
        }
        if include_top_k:
            alignment_record.update(
                {
                    "top_k_overlap_ratio": record["val-topk/overlap_ratio"],
                    "shared_token_advantage": record[
                        "val-topk/adv_intersection"
                    ],
                }
            )
        alignment.append(alignment_record)
        raw_record = {"step": step}
        for label, key in RAW_LOG_PROBABILITY_METRICS.items():
            value = record.get(key)
            if value is not None and math.isfinite(value):
                raw_record[label] = value
        if len(raw_record) > 1:
            raw_log_probability.append(raw_record)
    return {
        "upstream_probability_debug": upstream,
        "student_teacher_alignment": alignment,
        "raw_log_probability_supplemental": raw_log_probability,
    }


def analyze_logs(primary_log: Path, resume_log: Path) -> dict:
    primary_text = primary_log.read_text(encoding="utf-8", errors="replace")
    resume_text = resume_log.read_text(encoding="utf-8", errors="replace")
    primary_metrics = _metrics(primary_text)
    resume_metrics = _metrics(resume_text)
    primary_steps = _step_list(primary_metrics)
    resume_steps = _step_list(resume_metrics)
    if primary_steps != [1, 2, 3, 4, 5]:
        raise RuntimeError(f"primary steps are not exactly 1-5: {primary_steps}")
    if resume_steps != [6, 7, 8]:
        raise RuntimeError(f"resume steps are not exactly 6-8: {resume_steps}")
    resume_match = re.search(r"Setting global step to\s+(\d+)", resume_text)
    if not resume_match or int(resume_match.group(1)) != 5:
        raise RuntimeError("resume did not load global step 5")

    all_metrics = primary_metrics + resume_metrics
    required_finite_metrics = (
        "actor/pg_loss",
        "actor/grad_norm",
        "critic/advantages/mean_overall",
        *UPSTREAM_PROBABILITY_METRICS,
        *STUDENT_TEACHER_ALIGNMENT_METRICS,
    )
    for key in required_finite_metrics:
        _assert_finite(all_metrics, key)
    invalid_debug_steps = [
        int(record["training/global_step"])
        for record in all_metrics
        if record["training/rollout_probs_diff_valid"] != 1.0
    ]
    if invalid_debug_steps:
        raise RuntimeError(
            "upstream response-masked probability comparison is invalid at steps "
            f"{invalid_debug_steps}"
        )

    combined = primary_text + "\n" + resume_text
    mask_checks = _markers(combined, "OPD_RESPONSE_MASK_CHECK")
    teacher_initializations = _markers(combined, "OPD_TEACHER_FROZEN_INIT")
    teacher_checks = _markers(combined, "OPD_TEACHER_FROZEN_CHECK")
    rollout_syncs = _markers(combined, "OPD_ROLLOUT_WEIGHT_SYNC")
    if len(mask_checks) < 8 or not all(
        item.get("passed") is True
        and item.get("prompt_tokens_in_loss") == 0
        and item.get("padding_tokens_in_loss") == 0
        for item in mask_checks
    ):
        raise RuntimeError("response-loss masking evidence is incomplete")
    if len(teacher_initializations) < 2 or not all(item.get("passed") is True for item in teacher_initializations):
        raise RuntimeError("Teacher freeze initialization evidence is incomplete")
    if len(teacher_checks) < 8 or not all(item.get("passed") is True for item in teacher_checks):
        raise RuntimeError("Teacher frozen-state evidence is incomplete")
    if len(rollout_syncs) < 8:
        raise RuntimeError("updated Student rollout synchronization evidence is incomplete")

    return {
        "passed": True,
        "primary_steps": primary_steps,
        "resume_steps": resume_steps,
        "resume_loaded_step": 5,
        "response_mask_checks": len(mask_checks),
        "teacher_frozen_initializations": len(teacher_initializations),
        "teacher_frozen_checks": len(teacher_checks),
        "rollout_weight_syncs": len(rollout_syncs),
        "acceptance_policy": {
            "upstream_probability_debug": (
                "required, response-masked, valid and finite; no upstream numeric threshold"
            ),
            "student_teacher_alignment": (
                "required and finite at every G1 step; no 5-10-step trend threshold"
            ),
            "raw_log_probability": "supplemental diagnostic only; not a G1 numeric gate",
        },
        "diagnostics": _diagnostics(all_metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-log", type=Path, required=True)
    parser.add_argument("--resume-log", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze_logs(args.primary_log, args.resume_log)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
