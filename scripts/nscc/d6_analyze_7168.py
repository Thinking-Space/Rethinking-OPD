#!/usr/bin/env python3
"""Close Timeline G2 from the separate 7168-token formal-candidate validation."""

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
    _step_list,
)
from d5_analyze_pilot import (
    ERROR_PATTERNS,
    TIMING_METRICS,
    _effective_answer_report,
    _resource_report,
    _summary,
    _trend,
)


MAX_RESPONSE_LENGTH = 7168
ROLLOUT_RECORDS_PER_STEP = 32 * 4
FORMAL_BUDGET_SU = 18_432
SU_PER_GPU_HOUR = 64
PILOT_GPU_HOUR_CAP = 32


def analyze_7168(
    training_log: Path,
    resource_csv: Path,
    rollouts_root: Path,
    short_report_path: Path,
    *,
    gpu_count: int,
    allocated_host_memory_gib: float,
    allocated_walltime_hours: float,
    short_pilot_actual_gpu_hours: float,
) -> dict:
    """Validate numerical closure, lengths, resources, and affordable D7 scale."""
    text = training_log.read_text(encoding="utf-8", errors="replace")
    matched_errors = [pattern for pattern in ERROR_PATTERNS if re.search(pattern, text, re.I)]
    if matched_errors:
        raise RuntimeError(f"7168 validation log contains OOM/error signature: {matched_errors}")

    short_report = json.loads(short_report_path.read_text(encoding="utf-8"))
    expected_short_state = {
        "passed": True,
        "pilot_status": "PASS_4096",
        "g2_status": "PENDING_7168_VALIDATION",
    }
    if any(short_report.get(key) != value for key, value in expected_short_state.items()):
        raise RuntimeError("4096 Pilot report is not a passed prerequisite for 7168 validation")

    records = _metrics(text)
    steps = _step_list(records)
    expected_steps = list(range(1, 9))
    if steps != expected_steps:
        raise RuntimeError(f"7168 validation steps are not exactly 1-8: {steps}")

    required_finite_metrics = (
        "actor/pg_loss",
        "actor/grad_norm",
        "critic/advantages/mean_overall",
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
        *STUDENT_TEACHER_ALIGNMENT_METRICS,
    )
    for key in required_finite_metrics:
        _assert_finite(records, key)

    invalid_probability_steps = [
        int(record["training/global_step"])
        for record in records
        if record["training/rollout_probs_diff_valid"] != 1.0
    ]
    if invalid_probability_steps:
        raise RuntimeError(
            "upstream response-masked probability comparison is invalid at steps "
            f"{invalid_probability_steps}"
        )

    teacher_abs_signal = [record["critic/advantages/abs_mean"] for record in records]
    teacher_entropy = [record["teacher/entropy"] for record in records]
    overlap = [record["val-topk/overlap_ratio"] for record in records]
    if not any(value > 0 for value in teacher_abs_signal):
        raise RuntimeError("Teacher advantage signal is identically zero")
    if not any(value > 0 for value in teacher_entropy):
        raise RuntimeError("Teacher entropy is identically zero")
    if all(value in {0.0, 1.0} for value in overlap):
        raise RuntimeError("top-k overlap is saturated at boundary values")

    mask_checks = _markers(text, "OPD_RESPONSE_MASK_CHECK")
    checks_by_step = {int(item["step"]): item for item in mask_checks}
    if sorted(checks_by_step) != expected_steps:
        raise RuntimeError("response-loss mask evidence does not cover all eight steps")
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

    teacher_initializations = _markers(text, "OPD_TEACHER_FROZEN_INIT")
    teacher_checks = _markers(text, "OPD_TEACHER_FROZEN_CHECK")
    rollout_syncs = _markers(text, "OPD_ROLLOUT_WEIGHT_SYNC")
    if not teacher_initializations or not all(
        item.get("passed") is True for item in teacher_initializations
    ):
        raise RuntimeError("Teacher freeze initialization evidence is incomplete")
    if len(teacher_checks) < 8 or not all(item.get("passed") is True for item in teacher_checks):
        raise RuntimeError("Teacher frozen-state evidence is incomplete")
    if len(rollout_syncs) < 8:
        raise RuntimeError("updated Student rollout synchronization evidence is incomplete")

    warmup_discarded_steps = [1, 2, 3]
    measured_records = [
        record
        for record in records
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
    lengths = {
        "configured_max_response_tokens": MAX_RESPONSE_LENGTH,
        "response_mean_tokens": _summary(
            [record["response_length/mean"] for record in records]
        ),
        "response_p95_tokens": _summary(response_p95),
        "hit_length_limit_ratio": _summary(
            [record["response_length/clip_ratio"] for record in records]
        ),
        "aborted_ratio": _summary(
            [record["response/aborted_ratio"] for record in records]
        ),
        **effective_answers,
    }

    resources = _resource_report(
        resource_csv,
        gpu_count=gpu_count,
        allocated_host_memory_gib=allocated_host_memory_gib,
    )
    if resources["peak_concurrently_active_gpus"] != gpu_count:
        raise RuntimeError("resource monitor never observed all four GPUs active together")
    resources["max_logged_gpu_memory_allocated_gib"] = max(
        record["perf/max_memory_allocated_gb"] for record in records
    )
    resources["max_logged_gpu_memory_reserved_gib"] = max(
        record["perf/max_memory_reserved_gb"] for record in records
    )
    resources["max_logged_cpu_memory_used_gib"] = max(
        record["perf/cpu_memory_used_gb"] for record in records
    )

    if not math.isfinite(short_pilot_actual_gpu_hours) or short_pilot_actual_gpu_hours <= 0:
        raise RuntimeError("4096 Pilot actual GPU hours must be finite and positive")
    future_reserved_gpu_hours = gpu_count * allocated_walltime_hours
    combined_pilot_gpu_hours_upper_bound = (
        short_pilot_actual_gpu_hours + future_reserved_gpu_hours
    )
    if combined_pilot_gpu_hours_upper_bound > PILOT_GPU_HOUR_CAP:
        raise RuntimeError(
            "combined 4096 actual plus 7168 reservation exceeds the 32 GPU-hour Pilot cap"
        )

    loader_match = re.search(r"Size of train dataloader:\s*(\d+)", text)
    if not loader_match:
        raise RuntimeError("7168 log does not report the actual train dataloader size")
    one_epoch_steps = int(loader_match.group(1))
    p90_step_seconds = timings["overall"]["p90"]
    affordable_steps = math.floor(
        (FORMAL_BUDGET_SU / SU_PER_GPU_HOUR)
        * 3600
        / (p90_step_seconds * gpu_count)
    )
    recommended_training_steps = min(one_epoch_steps, affordable_steps)
    if recommended_training_steps < 1:
        raise RuntimeError("formal budget cannot afford even one observed 7168 step")
    recommended_plan = (
        "one_epoch" if recommended_training_steps == one_epoch_steps else "fixed_steps"
    )

    trend_records = [
        {
            "top_k_overlap_ratio": record["val-topk/overlap_ratio"],
            "shared_token_advantage": record["val-topk/adv_intersection"],
            "student_minus_teacher_entropy": (
                record["actor/entropy"] - record["teacher/entropy"]
            ),
        }
        for record in records
    ]
    trend_metrics = {
        key: _trend(trend_records, key, window_size=3)
        for key in trend_records[0]
    }

    return {
        "passed": True,
        "g2_status": "PASS",
        "selected_config": {
            "max_prompt_length": 1024,
            "max_response_length": 7168,
            "train_batch_size": 32,
            "rollout_n": 4,
            "actor_token_budget_per_gpu": 8192,
        },
        "steps": steps,
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
            "policy": "report window trends only; no upstream single-point numeric threshold",
            "metrics": trend_metrics,
        },
        "resources": resources,
        "budget": {
            "gpu_count": gpu_count,
            "allocated_walltime_hours": allocated_walltime_hours,
            "future_reserved_gpu_hours": future_reserved_gpu_hours,
            "short_pilot_actual_gpu_hours": short_pilot_actual_gpu_hours,
            "combined_pilot_gpu_hours_upper_bound": combined_pilot_gpu_hours_upper_bound,
            "pilot_gpu_hour_cap": PILOT_GPU_HOUR_CAP,
            "actual_train_dataloader_steps": one_epoch_steps,
            "observed_7168_b32_p90_step_seconds": p90_step_seconds,
            "formal_budget_su": FORMAL_BUDGET_SU,
            "affordable_training_steps": affordable_steps,
            "recommended_training_steps": recommended_training_steps,
            "recommended_plan": recommended_plan,
            "estimated_recommended_gpu_hours": (
                recommended_training_steps * p90_step_seconds * gpu_count / 3600
            ),
            "estimated_recommended_su": (
                recommended_training_steps
                * p90_step_seconds
                * gpu_count
                / 3600
                * SU_PER_GPU_HOUR
            ),
        },
        "acceptance_policy": {
            "hard_failures": (
                "non-continuous steps, mask or Teacher freeze failure, non-finite or absent "
                "Teacher signal, OOM, incomplete resource coverage, uncontrolled length, "
                "or a Pilot/formal budget violation"
            ),
            "not_required": "Pilot score improvement or monotonic loss",
            "alignment": (
                "finite upstream diagnostics and window trends; no invented numeric threshold"
            ),
        },
        "diagnostics": _diagnostics(records),
    }


def render_g2_readiness(report: dict, run_root: Path) -> str:
    if not report.get("passed") or report.get("g2_status") != "PASS":
        raise RuntimeError("cannot render G2 readiness from a failed report")
    timing = report["performance"]["timings_seconds"]
    lengths = report["lengths"]
    resources = report["resources"]
    budget = report["budget"]
    selected = report["selected_config"]
    return "\n".join(
        [
            "# OPD D5–6 G2 Readiness",
            "",
            "G2 status: PASS",
            f"selected response length: {selected['max_response_length']}",
            "",
            f"- Run root: `{run_root}`",
            "- Independent 7168-token validation completed steps 1–8 from the initial Student.",
            "- Warmup/compile steps excluded from timing: 1, 2, 3.",
            f"- Overall step time median/P90: {timing['overall']['median']:.3f} / "
            f"{timing['overall']['p90']:.3f} s.",
            f"- Rollout median/P90: {timing['rollout']['median']:.3f} / "
            f"{timing['rollout']['p90']:.3f} s.",
            f"- Teacher scoring median/P90: {timing['teacher_scoring']['median']:.3f} / "
            f"{timing['teacher_scoring']['p90']:.3f} s.",
            f"- Actor update median/P90: {timing['actor_update']['median']:.3f} / "
            f"{timing['actor_update']['p90']:.3f} s.",
            f"- Response mean: {lengths['response_mean_tokens']['mean']:.1f} tokens; "
            f"per-step P95 max: {lengths['response_p95_tokens']['max']:.1f}; "
            f"mean hit-length-limit ratio: {lengths['hit_length_limit_ratio']['mean']:.4f}.",
            f"- Effective answer presence ratio: "
            f"{lengths['effective_answer_presence_ratio']:.4f}.",
            f"- Peak monitored GPU memory: {resources['max_gpu_memory_used_mib']:.0f} MiB; "
            f"minimum headroom: {resources['min_gpu_memory_headroom_mib']:.0f} MiB.",
            f"- Peak host RSS: {resources['max_host_rss_gib']:.3f} GiB of "
            f"{resources['allocated_host_memory_gib']:.0f} GiB allocated.",
            f"- Combined Pilot upper bound: "
            f"{budget['combined_pilot_gpu_hours_upper_bound']:.3f} GPU-hours of "
            f"{budget['pilot_gpu_hour_cap']} allowed.",
            f"- Formal recommendation: {budget['recommended_plan']} / "
            f"{budget['recommended_training_steps']} steps under the "
            f"{budget['formal_budget_su']} SU budget.",
            "- Alignment metrics use three-step window trends; no upstream-absent "
            "single-point threshold was added.",
            "- Pilot score improvement and monotonic loss were not required.",
            "- D7 has not been started.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--resource-csv", type=Path, required=True)
    parser.add_argument("--rollouts-root", type=Path, required=True)
    parser.add_argument("--short-report", type=Path, required=True)
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--allocated-host-memory-gib", type=float, default=220)
    parser.add_argument("--allocated-walltime-hours", type=float, default=3)
    parser.add_argument("--short-pilot-actual-gpu-hours", type=float, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readiness-output", type=Path, required=True)
    args = parser.parse_args()

    report = analyze_7168(
        args.training_log,
        args.resource_csv,
        args.rollouts_root,
        args.short_report,
        gpu_count=args.gpu_count,
        allocated_host_memory_gib=args.allocated_host_memory_gib,
        allocated_walltime_hours=args.allocated_walltime_hours,
        short_pilot_actual_gpu_hours=args.short_pilot_actual_gpu_hours,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    args.readiness_output.parent.mkdir(parents=True, exist_ok=True)
    args.readiness_output.write_text(
        render_g2_readiness(report, args.run_root), encoding="utf-8"
    )
    print(payload)


if __name__ == "__main__":
    main()
