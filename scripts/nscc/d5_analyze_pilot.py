#!/usr/bin/env python3
"""Analyze the Timeline D5 4096-token Pilot without adding local score gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
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


TIMING_METRICS = {
    "overall": "timing_s/step",
    "rollout": "timing_s/gen",
    "teacher_scoring": "timing_s/compute_rm_score",
    "actor_update": "timing_s/update_actor",
}

ERROR_PATTERNS = (
    r"cuda out of memory",
    r"outofmemoryerror",
    r"cudnn_status_alloc_failed",
)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise RuntimeError("cannot summarize an empty metric series")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict:
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("metric series is empty or non-finite")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def _trend(records: list[dict[str, float]], key: str, *, window_size: int = 5) -> dict:
    values = [record[key] for record in records]
    if len(values) < 2 * window_size:
        raise RuntimeError(f"not enough values to report {key} window trend")
    first = statistics.fmean(values[:window_size])
    last = statistics.fmean(values[-window_size:])
    return {
        "window_size": window_size,
        "first_window_mean": first,
        "last_window_mean": last,
        "last_minus_first": last - first,
    }


def _resource_report(resource_csv: Path, *, gpu_count: int, allocated_host_memory_gib: float) -> dict:
    with resource_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("resource monitor has no samples")

    required_fields = {
        "timestamp",
        "gpu_index",
        "memory_used_mib",
        "memory_total_mib",
        "utilization_percent",
        "temperature_c",
        "host_rss_kib",
    }
    if set(rows[0]) != required_fields:
        raise RuntimeError(f"unexpected resource monitor fields: {sorted(rows[0])}")

    parsed = []
    for row in rows:
        try:
            parsed.append(
                {
                    "timestamp": row["timestamp"],
                    "gpu_index": int(row["gpu_index"]),
                    "memory_used_mib": float(row["memory_used_mib"]),
                    "memory_total_mib": float(row["memory_total_mib"]),
                    "utilization_percent": float(row["utilization_percent"]),
                    "temperature_c": float(row["temperature_c"]),
                    "host_rss_kib": float(row["host_rss_kib"]),
                }
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"invalid resource monitor row: {row}") from error
    numeric_values = [
        value
        for row in parsed
        for key, value in row.items()
        if key != "timestamp"
    ]
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise RuntimeError("resource monitor contains non-finite values")

    observed_indices = sorted({row["gpu_index"] for row in parsed})
    if observed_indices != list(range(gpu_count)):
        raise RuntimeError(
            f"resource monitor did not cover all {gpu_count} GPUs: {observed_indices}"
        )
    over_capacity = [
        row for row in parsed if row["memory_used_mib"] > row["memory_total_mib"]
    ]
    if over_capacity:
        raise RuntimeError("GPU monitor reports memory use above device capacity")

    max_host_rss_gib = max(row["host_rss_kib"] for row in parsed) / (1024**2)
    if max_host_rss_gib > allocated_host_memory_gib:
        raise RuntimeError(
            f"host RSS {max_host_rss_gib:.3f} GiB exceeded allocation "
            f"{allocated_host_memory_gib:.3f} GiB"
        )

    utilization_by_timestamp: dict[str, int] = {}
    for row in parsed:
        if row["utilization_percent"] > 0:
            utilization_by_timestamp[row["timestamp"]] = (
                utilization_by_timestamp.get(row["timestamp"], 0) + 1
            )

    max_used = max(row["memory_used_mib"] for row in parsed)
    min_headroom = min(
        row["memory_total_mib"] - row["memory_used_mib"] for row in parsed
    )
    return {
        "sample_rows": len(parsed),
        "sample_timestamps": len({row["timestamp"] for row in parsed}),
        "observed_gpu_indices": observed_indices,
        "max_gpu_memory_used_mib": max_used,
        "min_gpu_memory_headroom_mib": min_headroom,
        "max_gpu_utilization_percent": max(
            row["utilization_percent"] for row in parsed
        ),
        "peak_concurrently_active_gpus": max(utilization_by_timestamp.values(), default=0),
        "max_gpu_temperature_c": max(row["temperature_c"] for row in parsed),
        "max_host_rss_gib": max_host_rss_gib,
        "allocated_host_memory_gib": allocated_host_memory_gib,
    }


def _effective_answer_report(
    rollouts_root: Path,
    expected_steps: list[int],
    *,
    expected_records_per_step: int = 64,
) -> dict:
    files = {}
    rollout_paths = [
        *rollouts_root.glob("*.jsonl"),
        *rollouts_root.glob("*/*.jsonl"),
    ]
    for path in sorted(rollout_paths):
        try:
            step = int(path.stem)
        except ValueError:
            continue
        files[step] = path
    if sorted(files) != expected_steps:
        raise RuntimeError(
            "rollout dumps do not cover continuous Pilot steps: "
            f"found={sorted(files)}, expected={expected_steps}"
        )

    total = 0
    effective = 0
    records_per_step = {}
    answer_pattern = re.compile(r"Answer\s*:\s*\S|\\boxed\s*\{\s*[^}]+\s*\}")
    for step in expected_steps:
        count = 0
        with files[step].open(encoding="utf-8", errors="replace") as rollout_file:
            for line in rollout_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                output = record.get("output")
                if not isinstance(output, str):
                    raise RuntimeError(f"rollout step {step} has no string output")
                total += 1
                count += 1
                effective += bool(answer_pattern.search(output))
        records_per_step[step] = count
    invalid_counts = {
        step: count
        for step, count in records_per_step.items()
        if count != expected_records_per_step
    }
    if invalid_counts:
        raise RuntimeError(
            "Pilot rollout count does not match the configured batch x n: "
            f"{invalid_counts}"
        )
    return {
        "rollout_records": total,
        "effective_answer_records": effective,
        "effective_answer_presence_ratio": effective / total,
    }


def analyze_pilot(
    primary_log: Path,
    resume_log: Path,
    resource_csv: Path,
    rollouts_root: Path,
    *,
    max_response_length: int,
    gpu_count: int,
    allocated_host_memory_gib: float,
    allocated_walltime_hours: float,
) -> dict:
    primary_text = primary_log.read_text(encoding="utf-8", errors="replace")
    resume_text = resume_log.read_text(encoding="utf-8", errors="replace")
    combined_text = primary_text + "\n" + resume_text
    matched_errors = [pattern for pattern in ERROR_PATTERNS if re.search(pattern, combined_text, re.I)]
    if matched_errors:
        raise RuntimeError(f"Pilot log contains OOM/error signature: {matched_errors}")

    primary_metrics = _metrics(primary_text)
    resume_metrics = _metrics(resume_text)
    primary_steps = _step_list(primary_metrics)
    resume_steps = _step_list(resume_metrics)
    if primary_steps != list(range(1, 31)):
        raise RuntimeError(f"primary steps are not exactly 1-30: {primary_steps}")
    if resume_steps != [31, 32]:
        raise RuntimeError(f"resume steps are not exactly 31-32: {resume_steps}")
    resume_match = re.search(r"Setting global step to\s+(\d+)", resume_text)
    if not resume_match or int(resume_match.group(1)) != 30:
        raise RuntimeError("resume did not load global step 30")

    all_metrics = primary_metrics + resume_metrics
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
        _assert_finite(all_metrics, key)

    invalid_probability_steps = [
        int(record["training/global_step"])
        for record in all_metrics
        if record["training/rollout_probs_diff_valid"] != 1.0
    ]
    if invalid_probability_steps:
        raise RuntimeError(
            "upstream response-masked probability comparison is invalid at steps "
            f"{invalid_probability_steps}"
        )

    teacher_abs_signal = [record["critic/advantages/abs_mean"] for record in all_metrics]
    teacher_entropy = [record["teacher/entropy"] for record in all_metrics]
    overlap = [record["val-topk/overlap_ratio"] for record in all_metrics]
    if not any(value > 0 for value in teacher_abs_signal):
        raise RuntimeError("Teacher advantage signal is identically zero")
    if not any(value > 0 for value in teacher_entropy):
        raise RuntimeError("Teacher entropy is identically zero")
    if all(value in {0.0, 1.0} for value in overlap):
        raise RuntimeError("top-k overlap is saturated at boundary values")

    expected_steps = list(range(1, 33))
    mask_checks = _markers(combined_text, "OPD_RESPONSE_MASK_CHECK")
    checks_by_step = {int(item["step"]): item for item in mask_checks}
    if sorted(checks_by_step) != expected_steps:
        raise RuntimeError("response-loss mask evidence does not cover all 32 steps")
    if not all(
        item.get("passed") is True
        and item.get("prompt_tokens_in_loss") == 0
        and item.get("padding_tokens_in_loss") == 0
        and item.get("batch_size") == 64
        for item in checks_by_step.values()
    ):
        raise RuntimeError("response-loss masking or rollout batch evidence is invalid")
    response_p95 = [float(checks_by_step[step]["response_length_p95"]) for step in expected_steps]
    if not all(math.isfinite(value) and 0 <= value <= max_response_length for value in response_p95):
        raise RuntimeError("response P95 evidence is missing, non-finite, or above configured length")

    teacher_initializations = _markers(combined_text, "OPD_TEACHER_FROZEN_INIT")
    teacher_checks = _markers(combined_text, "OPD_TEACHER_FROZEN_CHECK")
    rollout_syncs = _markers(combined_text, "OPD_ROLLOUT_WEIGHT_SYNC")
    if len(teacher_initializations) < 2 or not all(
        item.get("passed") is True for item in teacher_initializations
    ):
        raise RuntimeError("Teacher freeze initialization evidence is incomplete")
    if len(teacher_checks) < 32 or not all(item.get("passed") is True for item in teacher_checks):
        raise RuntimeError("Teacher frozen-state evidence is incomplete")
    if len(rollout_syncs) < 32:
        raise RuntimeError("updated Student rollout synchronization evidence is incomplete")

    warmup_discarded_steps = [1, 2, 3, 31]
    measured_records = [
        record
        for record in all_metrics
        if int(record["training/global_step"]) not in warmup_discarded_steps
    ]
    timings = {
        label: _summary([record[key] for record in measured_records])
        for label, key in TIMING_METRICS.items()
    }

    effective_answers = _effective_answer_report(rollouts_root, expected_steps)
    length_report = {
        "configured_max_response_tokens": max_response_length,
        "response_mean_tokens": _summary(
            [record["response_length/mean"] for record in all_metrics]
        ),
        "response_p95_tokens": _summary(response_p95),
        "hit_length_limit_ratio": _summary(
            [record["response_length/clip_ratio"] for record in all_metrics]
        ),
        "aborted_ratio": _summary(
            [record["response/aborted_ratio"] for record in all_metrics]
        ),
        **effective_answers,
    }

    resources = _resource_report(
        resource_csv,
        gpu_count=gpu_count,
        allocated_host_memory_gib=allocated_host_memory_gib,
    )
    resources["max_logged_gpu_memory_allocated_gib"] = max(
        record["perf/max_memory_allocated_gb"] for record in all_metrics
    )
    resources["max_logged_gpu_memory_reserved_gib"] = max(
        record["perf/max_memory_reserved_gb"] for record in all_metrics
    )
    resources["max_logged_cpu_memory_used_gib"] = max(
        record["perf/cpu_memory_used_gb"] for record in all_metrics
    )

    reserved_gpu_hours = gpu_count * allocated_walltime_hours
    reserved_su = reserved_gpu_hours * 64
    if reserved_su > 2048:
        raise RuntimeError(
            f"Pilot reservation {reserved_su:g} SU exceeds Timeline budget 2048 SU"
        )
    loader_match = re.search(r"Size of train dataloader:\s*(\d+)", primary_text)
    if not loader_match:
        raise RuntimeError("primary log does not report the actual train dataloader size")
    one_epoch_steps = int(loader_match.group(1))
    p90_step_seconds = timings["overall"]["p90"]
    estimated_gpu_hours = one_epoch_steps * p90_step_seconds * gpu_count / 3600

    trend_records = []
    for record in all_metrics:
        trend_records.append(
            {
                "top_k_overlap_ratio": record["val-topk/overlap_ratio"],
                "shared_token_advantage": record["val-topk/adv_intersection"],
                "student_minus_teacher_entropy": (
                    record["actor/entropy"] - record["teacher/entropy"]
                ),
            }
        )
    trend_metrics = {
        key: _trend(trend_records, key)
        for key in trend_records[0]
    }

    return {
        "passed": True,
        "pilot_status": "PASS_4096",
        "g2_status": "PENDING_7168_VALIDATION",
        "primary_steps": primary_steps,
        "resume_steps": resume_steps,
        "resume_loaded_step": 30,
        "performance": {
            "warmup_discarded_steps": warmup_discarded_steps,
            "timings_seconds": timings,
        },
        "lengths": length_report,
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
            "reserved_gpu_hours": reserved_gpu_hours,
            "reserved_su": reserved_su,
            "timeline_pilot_budget_su": 2048,
            "actual_train_dataloader_steps": one_epoch_steps,
            "observed_4096_b16_p90_step_seconds": p90_step_seconds,
            "estimated_4096_b16_one_epoch_steps": one_epoch_steps,
            "estimated_4096_b16_one_epoch_gpu_hours": estimated_gpu_hours,
            "estimated_4096_b16_one_epoch_su": estimated_gpu_hours * 64,
            "estimate_scope": (
                "4096-token batch-16 proxy only; do not reuse as 7168-token batch-32 throughput"
            ),
        },
        "acceptance_policy": {
            "hard_failures": (
                "non-continuous steps, incomplete resume/checkpoint evidence, mask or Teacher "
                "freeze failure, non-finite or absent Teacher signal, OOM, incomplete resource "
                "coverage, or Pilot reservation above the Timeline budget"
            ),
            "not_required": "Pilot score improvement or monotonic loss",
            "next_gate": "separate single-variable 7168-token validation before final G2",
        },
        "diagnostics": _diagnostics(all_metrics),
    }


def render_4096_readiness(report: dict, run_root: Path) -> str:
    if not report.get("passed") or report.get("pilot_status") != "PASS_4096":
        raise RuntimeError("cannot render readiness from a failed 4096 Pilot report")
    timing = report["performance"]["timings_seconds"]
    lengths = report["lengths"]
    resources = report["resources"]
    budget = report["budget"]
    return "\n".join(
        [
            "# OPD D5 4096-token Pilot Readiness",
            "",
            "4096 Pilot status: PASS",
            "G2 status: PENDING 7168 VALIDATION",
            "",
            f"- Run root: `{run_root}`",
            "- Continuous training: steps 1–30; full-state resume: steps 31–32.",
            "- Warmup/compile steps excluded from timing: 1, 2, 3, 31.",
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
            f"- Reservation: {budget['reserved_gpu_hours']:.0f} GPU-hours / "
            f"{budget['reserved_su']:.0f} SU (Timeline Pilot cap: 2048 SU).",
            "- Alignment metrics are reported as five-step window trends; no upstream "
            "single-point threshold was added.",
            "- This report does not claim score improvement and does not freeze D7. "
            "Run a separate 7168-token validation next.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-log", type=Path, required=True)
    parser.add_argument("--resume-log", type=Path, required=True)
    parser.add_argument("--resource-csv", type=Path, required=True)
    parser.add_argument("--rollouts-root", type=Path, required=True)
    parser.add_argument("--max-response-length", type=int, default=4096)
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument("--allocated-host-memory-gib", type=float, default=220)
    parser.add_argument("--allocated-walltime-hours", type=float, default=8)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readiness-output", type=Path, required=True)
    args = parser.parse_args()

    report = analyze_pilot(
        args.primary_log,
        args.resume_log,
        args.resource_csv,
        args.rollouts_root,
        max_response_length=args.max_response_length,
        gpu_count=args.gpu_count,
        allocated_host_memory_gib=args.allocated_host_memory_gib,
        allocated_walltime_hours=args.allocated_walltime_hours,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    args.readiness_output.parent.mkdir(parents=True, exist_ok=True)
    args.readiness_output.write_text(
        render_4096_readiness(report, args.run_root), encoding="utf-8"
    )
    print(payload)


if __name__ == "__main__":
    main()
