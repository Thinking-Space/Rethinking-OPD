#!/usr/bin/env python3
"""Verify completion of the frozen Timeline D7-9 full-data formal run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from d4_analyze_smoke import (
    STUDENT_TEACHER_ALIGNMENT_METRICS,
    UPSTREAM_PROBABILITY_METRICS,
    _assert_finite,
    _diagnostics,
    _markers,
    _metrics,
)
from d5_analyze_pilot import (
    ERROR_PATTERNS,
    TIMING_METRICS,
    _effective_answer_report,
    _resource_report,
    _summary,
    _trend,
)


TOTAL_TRAINING_STEPS = 528
MAX_RESPONSE_LENGTH = 7168
ROLLOUT_RECORDS_PER_STEP = 32 * 4
UNIQUE_PROMPTS_PER_STEP = 32
FORMAL_BUDGET_GPU_HOURS = 288
FORMAL_BUDGET_SU = 18_432
SU_PER_GPU_HOUR = 64
WANDB_URL_RE = re.compile(
    r"(https://wandb\.ai/[^\s]+/runs/([A-Za-z0-9_-]+))", re.I
)
WANDB_OFFLINE_RE = re.compile(
    r"wandb[^\n]*\bsyncing\s+is\s+set\s+to\s+[`'\"]?(?:offline|dryrun)\b",
    re.I,
)


def _resume_step(text: str) -> int | None:
    values = [int(value) for value in re.findall(r"Setting global step to\s+(\d+)", text)]
    if not values:
        return None
    if len(set(values)) != 1:
        raise RuntimeError(f"segment has ambiguous resume steps: {values}")
    return values[0]


def _unique_prompt_report(rollouts_root: Path, expected_steps: list[int]) -> dict:
    files: dict[int, Path] = {}
    for path in sorted(
        [*rollouts_root.glob("*.jsonl"), *rollouts_root.glob("*/*.jsonl")]
    ):
        try:
            files[int(path.stem)] = path
        except ValueError:
            continue
    if sorted(files) != expected_steps:
        raise RuntimeError("rollout dumps do not cover all formal-training steps")

    unique_total = 0
    invalid = {}
    for step in expected_steps:
        prompts = set()
        with files[step].open(encoding="utf-8", errors="replace") as rollout_file:
            for line in rollout_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                prompt = record.get("input")
                if not isinstance(prompt, str):
                    raise RuntimeError(f"rollout step {step} has no string input")
                prompts.add(prompt)
        unique_total += len(prompts)
        if len(prompts) != UNIQUE_PROMPTS_PER_STEP:
            invalid[step] = len(prompts)
    if invalid:
        raise RuntimeError(f"formal rollout unique-prompt counts are invalid: {invalid}")
    return {"unique_prompt_records": unique_total}


def analyze_formal(
    training_logs: list[Path],
    resource_csvs: list[Path],
    rollouts_root: Path,
    wandb_run_id_file: Path,
    *,
    gpu_count: int,
    allocated_host_memory_gib: float,
    allocated_walltime_hours_per_segment: float,
    distillation_mode: str = "top_k",
) -> dict:
    """Require one continuous 528-step epoch, full-state resumes, and online W&B."""
    if distillation_mode not in {"top_k", "sampled_token"}:
        raise ValueError("distillation_mode must be top_k or sampled_token")
    if not training_logs or len(training_logs) != len(resource_csvs):
        raise RuntimeError("training logs and resource CSVs must be non-empty and paired")

    texts = [path.read_text(encoding="utf-8", errors="replace") for path in training_logs]
    for text in texts:
        matched_errors = [
            pattern for pattern in ERROR_PATTERNS if re.search(pattern, text, re.I)
        ]
        if matched_errors:
            raise RuntimeError(
                f"formal training log contains OOM/error signature: {matched_errors}"
            )
        if WANDB_OFFLINE_RE.search(text):
            raise RuntimeError("formal training log reports offline W&B")

    resume_steps = [_resume_step(text) for text in texts]
    if resume_steps[0] is not None:
        raise RuntimeError("the first formal segment must start from the initial Student")
    if any(value is None for value in resume_steps[1:]):
        raise RuntimeError("every later formal segment must explicitly load full state")
    loaded_steps = [value for value in resume_steps[1:] if value is not None]
    if loaded_steps != sorted(loaded_steps) or len(set(loaded_steps)) != len(loaded_steps):
        raise RuntimeError(f"formal resume boundaries are not strictly increasing: {loaded_steps}")

    raw_records = [_metrics(text) for text in texts]
    accepted_records = []
    accepted_step_sets = []
    for index, records in enumerate(raw_records):
        lower = 0 if resume_steps[index] is None else int(resume_steps[index])
        upper = (
            int(resume_steps[index + 1])
            if index + 1 < len(resume_steps)
            else TOTAL_TRAINING_STEPS
        )
        selected = [
            record
            for record in records
            if lower < int(record["training/global_step"]) <= upper
        ]
        if not selected:
            raise RuntimeError(f"formal segment {index + 1} has no accepted steps")
        accepted_records.extend(selected)
        accepted_step_sets.append(
            {int(record["training/global_step"]) for record in selected}
        )

    steps = [int(record["training/global_step"]) for record in accepted_records]
    expected_steps = list(range(1, TOTAL_TRAINING_STEPS + 1))
    if steps != expected_steps:
        raise RuntimeError(
            "formal training is not exactly one continuous 528-step epoch: "
            f"observed_count={len(steps)}, first={steps[:3]}, last={steps[-3:]}"
        )

    loader_values = {
        int(value)
        for text in texts
        for value in re.findall(r"Size of train dataloader:\s*(\d+)", text)
    }
    if loader_values != {TOTAL_TRAINING_STEPS}:
        raise RuntimeError(f"formal dataloader size is not exactly 528: {loader_values}")

    alignment_metrics = (
        STUDENT_TEACHER_ALIGNMENT_METRICS
        if distillation_mode == "top_k"
        else ("actor/entropy", "teacher/entropy")
    )
    advantage_mean_metric = (
        "critic/advantages/mean_overall"
        if distillation_mode == "top_k"
        else "critic/advantages/mean"
    )
    sampled_reward_metrics = (
        ()
        if distillation_mode == "top_k"
        else (
            "critic/score/mean",
            "critic/rewards/mean",
            "critic/true_reward/mean",
        )
    )
    required_finite_metrics = (
        "actor/pg_loss",
        "actor/grad_norm",
        advantage_mean_metric,
        "critic/advantages/std",
        "critic/advantages/abs_mean",
        "response_length/mean",
        "response_length/max",
        "response_length/clip_ratio",
        "response/aborted_ratio",
        "perf/max_memory_allocated_gb",
        "perf/max_memory_reserved_gb",
        "perf/cpu_memory_used_gb",
        *TIMING_METRICS.values(),
        *UPSTREAM_PROBABILITY_METRICS,
        *alignment_metrics,
        *sampled_reward_metrics,
    )
    for key in required_finite_metrics:
        _assert_finite(accepted_records, key)
    invalid_probability_steps = [
        int(record["training/global_step"])
        for record in accepted_records
        if record["training/rollout_probs_diff_valid"] != 1.0
    ]
    if invalid_probability_steps:
        raise RuntimeError(
            "upstream response-masked probability comparison is invalid at steps "
            f"{invalid_probability_steps}"
        )

    mask_checks = []
    teacher_checks = 0
    rollout_sync_steps = set()
    teacher_initializations = []
    for segment_index, (text, accepted) in enumerate(
        zip(texts, accepted_step_sets, strict=True), start=1
    ):
        mask_checks.extend(
            item
            for item in _markers(text, "OPD_RESPONSE_MASK_CHECK")
            if int(item["step"]) in accepted
        )
        rollout_syncs = _markers(text, "OPD_ROLLOUT_WEIGHT_SYNC")
        expected_sync_counts = list(range(1, len(accepted) + 1))
        observed_sync_counts = [item.get("sync_count") for item in rollout_syncs]
        if observed_sync_counts != expected_sync_counts or not all(
            item.get("passed") is True and item.get("rank") == 0
            for item in rollout_syncs
        ):
            raise RuntimeError(
                "updated Student rollout synchronization evidence is invalid "
                f"for formal segment {segment_index}"
            )
        rollout_sync_steps.update(accepted)
        teacher_checks += len(_markers(text, "OPD_TEACHER_FROZEN_CHECK"))
        teacher_initializations.extend(_markers(text, "OPD_TEACHER_FROZEN_INIT"))

    checks_by_step = {int(item["step"]): item for item in mask_checks}
    if sorted(checks_by_step) != expected_steps:
        raise RuntimeError("response-loss mask evidence does not cover all 528 steps")
    if not all(
        item.get("passed") is True
        and item.get("prompt_tokens_in_loss") == 0
        and item.get("padding_tokens_in_loss") == 0
        and item.get("batch_size") == ROLLOUT_RECORDS_PER_STEP
        for item in checks_by_step.values()
    ):
        raise RuntimeError("response-loss masking or batch x rollout evidence is invalid")
    response_p95 = [
        float(checks_by_step[step]["response_length_p95"]) for step in expected_steps
    ]
    if not all(
        math.isfinite(value) and 0 <= value <= MAX_RESPONSE_LENGTH
        for value in response_p95
    ):
        raise RuntimeError("response P95 evidence is missing, non-finite, or above 7168")
    if len(teacher_initializations) != len(training_logs) or not all(
        item.get("passed") is True for item in teacher_initializations
    ):
        raise RuntimeError("Teacher freeze initialization evidence is incomplete")
    if teacher_checks < TOTAL_TRAINING_STEPS:
        raise RuntimeError("Teacher frozen-state evidence is incomplete")
    if rollout_sync_steps != set(expected_steps):
        raise RuntimeError("updated Student rollout synchronization evidence is incomplete")

    teacher_abs_signal = [
        record["critic/advantages/abs_mean"] for record in accepted_records
    ]
    teacher_entropy = [record["teacher/entropy"] for record in accepted_records]
    if not any(value > 0 for value in teacher_abs_signal):
        raise RuntimeError("Teacher advantage signal is identically zero")
    if not any(value > 0 for value in teacher_entropy):
        raise RuntimeError("Teacher entropy is identically zero")
    if distillation_mode == "top_k":
        overlap = [
            record["val-topk/overlap_ratio"] for record in accepted_records
        ]
        if all(value in {0.0, 1.0} for value in overlap):
            raise RuntimeError("top-k overlap is saturated at boundary values")

    warmup_discarded_steps = [1, 2, 3]
    warmup_discarded_steps.extend(value + 1 for value in loaded_steps)
    warmup_discarded_steps = sorted(set(warmup_discarded_steps))
    measured_records = [
        record
        for record in accepted_records
        if int(record["training/global_step"]) not in warmup_discarded_steps
    ]
    timings = {
        label: _summary([record[key] for record in measured_records])
        for label, key in TIMING_METRICS.items()
    }

    effective_answers = _effective_answer_report(
        rollouts_root,
        expected_steps,
        expected_records_per_step=ROLLOUT_RECORDS_PER_STEP,
    )
    unique_prompts = _unique_prompt_report(rollouts_root, expected_steps)
    lengths = {
        "configured_max_response_tokens": MAX_RESPONSE_LENGTH,
        "response_mean_tokens": _summary(
            [record["response_length/mean"] for record in accepted_records]
        ),
        "response_p95_tokens": _summary(response_p95),
        "hit_length_limit_ratio": _summary(
            [record["response_length/clip_ratio"] for record in accepted_records]
        ),
        "aborted_ratio": _summary(
            [record["response/aborted_ratio"] for record in accepted_records]
        ),
        **effective_answers,
        **unique_prompts,
    }

    segment_resources = [
        _resource_report(
            path,
            gpu_count=gpu_count,
            allocated_host_memory_gib=allocated_host_memory_gib,
        )
        for path in resource_csvs
    ]
    if any(
        item["peak_concurrently_active_gpus"] != gpu_count
        for item in segment_resources
    ):
        raise RuntimeError("a formal segment never observed all four GPUs active together")
    resources = {
        "segment_count": len(segment_resources),
        "sample_rows": sum(item["sample_rows"] for item in segment_resources),
        "sample_timestamps": sum(
            item["sample_timestamps"] for item in segment_resources
        ),
        "observed_gpu_indices": list(range(gpu_count)),
        "max_gpu_memory_used_mib": max(
            item["max_gpu_memory_used_mib"] for item in segment_resources
        ),
        "min_gpu_memory_headroom_mib": min(
            item["min_gpu_memory_headroom_mib"] for item in segment_resources
        ),
        "max_gpu_utilization_percent": max(
            item["max_gpu_utilization_percent"] for item in segment_resources
        ),
        "peak_concurrently_active_gpus": max(
            item["peak_concurrently_active_gpus"] for item in segment_resources
        ),
        "max_gpu_temperature_c": max(
            item["max_gpu_temperature_c"] for item in segment_resources
        ),
        "max_host_rss_gib": max(
            item["max_host_rss_gib"] for item in segment_resources
        ),
        "allocated_host_memory_gib": allocated_host_memory_gib,
        "max_logged_gpu_memory_allocated_gib": max(
            record["perf/max_memory_allocated_gb"] for record in accepted_records
        ),
        "max_logged_gpu_memory_reserved_gib": max(
            record["perf/max_memory_reserved_gb"] for record in accepted_records
        ),
        "max_logged_cpu_memory_used_gib": max(
            record["perf/cpu_memory_used_gb"] for record in accepted_records
        ),
    }

    segment_count = len(training_logs)
    reserved_gpu_hours = (
        segment_count * gpu_count * allocated_walltime_hours_per_segment
    )
    reserved_su = reserved_gpu_hours * SU_PER_GPU_HOUR
    if reserved_gpu_hours > FORMAL_BUDGET_GPU_HOURS or reserved_su > FORMAL_BUDGET_SU:
        raise RuntimeError("formal PBS segment reservations exceed the Timeline budget")

    run_id = wandb_run_id_file.read_text(encoding="utf-8").strip()
    if not run_id:
        raise RuntimeError("W&B run ID file is empty")
    urls = []
    url_ids = []
    for text in texts:
        matches = WANDB_URL_RE.findall(text)
        if not matches:
            raise RuntimeError("formal segment has no online W&B run URL")
        url, url_id = matches[-1]
        urls.append(url.rstrip(".,)"))
        url_ids.append(url_id)
    if set(url_ids) != {run_id} or len(set(urls)) != 1:
        raise RuntimeError("formal segments did not use one persistent W&B run")

    trend_records = []
    for record in accepted_records:
        trend_record = {
            "student_minus_teacher_entropy": (
                record["actor/entropy"] - record["teacher/entropy"]
            )
        }
        if distillation_mode == "top_k":
            trend_record.update(
                {
                    "top_k_overlap_ratio": record["val-topk/overlap_ratio"],
                    "shared_token_advantage": record[
                        "val-topk/adv_intersection"
                    ],
                }
            )
        trend_records.append(trend_record)
    trend_metrics = {
        key: _trend(trend_records, key, window_size=20) for key in trend_records[0]
    }
    alignment_policy = (
        "report 20-step window trends only; no upstream single-point numeric threshold"
        if distillation_mode == "top_k"
        else (
            "report the entropy-gap trend; top-k metrics are not applicable "
            "when log_prob_top_k=0"
        )
    )

    return {
        "passed": True,
        "distillation_mode": distillation_mode,
        "d7_9_status": "PASS_TRAINING_COMPLETE",
        "g3_status": "NOT_STARTED",
        "full_data_one_epoch": True,
        "steps": steps,
        "resume_loaded_steps": loaded_steps,
        "performance": {
            "warmup_discarded_steps": warmup_discarded_steps,
            "timings_seconds": timings,
        },
        "lengths": lengths,
        "teacher_signal": {
            "advantage_abs_mean": _summary(teacher_abs_signal),
            "teacher_entropy": _summary(teacher_entropy),
        },
        "alignment_trends": {
            "policy": alignment_policy,
            "metrics": trend_metrics,
        },
        "resources": resources,
        "budget": {
            "segment_count": segment_count,
            "gpu_count": gpu_count,
            "allocated_walltime_hours_per_segment": allocated_walltime_hours_per_segment,
            "reserved_gpu_hours_upper_bound": reserved_gpu_hours,
            "reserved_su_upper_bound": reserved_su,
            "formal_gpu_hour_cap": FORMAL_BUDGET_GPU_HOURS,
            "formal_su_cap": FORMAL_BUDGET_SU,
        },
        "wandb": {
            "mode": "online",
            "run_id": run_id,
            "run_url": urls[0],
            "single_run_across_segments": True,
        },
        "acceptance_policy": {
            "hard_failures": (
                "incomplete/non-continuous epoch, non-full-state resume, mask or Teacher "
                "freeze failure, non-finite diagnostics, OOM, uncontrolled response length, "
                "offline/split W&B logging, incomplete resource coverage, or budget overrun"
            ),
            "not_claimed": "independent evaluation score or G3 acceptance",
            "alignment": (
                "finite upstream diagnostics and 20-step window trends; no invented numeric threshold"
            ),
        },
        "diagnostics": _diagnostics(
            accepted_records, include_top_k=distillation_mode == "top_k"
        ),
    }


def render_main_readiness(report: dict, run_root: Path) -> str:
    if not report.get("passed") or report.get("d7_9_status") != "PASS_TRAINING_COMPLETE":
        raise RuntimeError("cannot render formal readiness from a failed report")
    timing = report["performance"]["timings_seconds"]
    lengths = report["lengths"]
    resources = report["resources"]
    budget = report["budget"]
    wandb = report["wandb"]
    sampled_token = report.get("distillation_mode") == "sampled_token"
    title = (
        "# OPD Sampled-token Control Readiness"
        if sampled_token
        else "# OPD D7–9 Formal Training Readiness"
    )
    status = (
        "Sampled-token control status: PASS_TRAINING_COMPLETE"
        if sampled_token
        else "D7–9 status: PASS_TRAINING_COMPLETE"
    )
    return "\n".join(
        [
            title,
            "",
            status,
            "G3 status: NOT_STARTED",
            "",
            f"- Run root: `{run_root}`",
            "- full-data one-epoch run: 528/528 steps",
            f"- Explicit full-state resume boundaries: {report['resume_loaded_steps'] or 'none'}.",
            f"- W&B mode: {wandb['mode']}; run: {wandb['run_url']}",
            f"- Overall step time median/P90: {timing['overall']['median']:.3f} / "
            f"{timing['overall']['p90']:.3f} s after per-segment warmup removal.",
            f"- Response mean: {lengths['response_mean_tokens']['mean']:.1f} tokens; "
            f"per-step P95 max: {lengths['response_p95_tokens']['max']:.1f}; "
            f"mean hit-length-limit ratio: {lengths['hit_length_limit_ratio']['mean']:.4f}.",
            f"- Effective answer presence ratio: "
            f"{lengths['effective_answer_presence_ratio']:.4f} over "
            f"{lengths['rollout_records']} rollouts.",
            f"- Peak monitored GPU memory: {resources['max_gpu_memory_used_mib']:.0f} MiB; "
            f"minimum headroom: {resources['min_gpu_memory_headroom_mib']:.0f} MiB.",
            f"- Peak host RSS: {resources['max_host_rss_gib']:.3f} GiB of "
            f"{resources['allocated_host_memory_gib']:.0f} GiB allocated.",
            f"- Reservation upper bound: {budget['reserved_gpu_hours_upper_bound']:.1f} "
            f"GPU-hours / {budget['reserved_su_upper_bound']:.0f} SU.",
            f"- Alignment policy: {report['alignment_trends']['policy']}.",
            "- Independent evaluation and G3 acceptance have not been started or claimed.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log", type=Path, action="append", required=True)
    parser.add_argument("--resource-csv", type=Path, action="append", required=True)
    parser.add_argument("--rollouts-root", type=Path, required=True)
    parser.add_argument("--wandb-run-id-file", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--allocated-host-memory-gib", type=float, default=220)
    parser.add_argument(
        "--allocated-walltime-hours-per-segment", type=float, default=24
    )
    parser.add_argument(
        "--distillation-mode",
        choices=["top_k", "sampled_token"],
        default="top_k",
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readiness-output", type=Path, required=True)
    args = parser.parse_args()

    report = analyze_formal(
        args.training_log,
        args.resource_csv,
        args.rollouts_root,
        args.wandb_run_id_file,
        gpu_count=args.gpu_count,
        allocated_host_memory_gib=args.allocated_host_memory_gib,
        allocated_walltime_hours_per_segment=(
            args.allocated_walltime_hours_per_segment
        ),
        distillation_mode=args.distillation_mode,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    args.readiness_output.parent.mkdir(parents=True, exist_ok=True)
    args.readiness_output.write_text(
        render_main_readiness(report, args.run_root), encoding="utf-8"
    )
    print(payload)


if __name__ == "__main__":
    main()
