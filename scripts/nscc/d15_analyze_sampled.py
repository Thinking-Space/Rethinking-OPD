#!/usr/bin/env python3
"""Compare the sampled-token control with the frozen initial and top-k16 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from d12_analyze_final import (
    build_failure_cases,
    load_json,
    load_jsonl,
    verify_evaluation_contract,
)
from d4_pipeline import (
    conclude_independent_evaluation,
    paired_bootstrap_pass1,
    timeline_evaluation_protocol,
)


REQUIRED_TASKS = ("MATH-500", "AIME24")


def verify_sampled_contract(
    baseline_summary: dict, top_k16_summary: dict, sampled_summary: dict
) -> dict:
    top_k16 = verify_evaluation_contract(baseline_summary, top_k16_summary)
    sampled = verify_evaluation_contract(baseline_summary, sampled_summary)
    top_k16_model = top_k16_summary["plan"]["worker_assignment"]["model"]
    sampled_model = sampled_summary["plan"]["worker_assignment"]["model"]
    if top_k16_model == sampled_model:
        raise ValueError("sampled-token and top-k16 evaluations use the same model path")
    return {
        "protocol_identical": top_k16["protocol_identical"]
        and sampled["protocol_identical"],
        "input_hashes_identical": top_k16["input_hashes_identical"]
        and sampled["input_hashes_identical"],
        "checkpoints_precommitted": top_k16["checkpoint_precommitted"]
        and sampled["checkpoint_precommitted"],
        "dev_used_for_selection": top_k16["dev_used_for_selection"]
        or sampled["dev_used_for_selection"],
        "models_distinct": True,
    }


def compare_task(
    initial_records: Sequence[dict],
    top_k16_records: Sequence[dict],
    sampled_records: Sequence[dict],
    *,
    expected_problems: int,
    expected_n: int,
    replicates: int,
    seed: int,
) -> dict:
    return {
        "sampled_vs_initial": paired_bootstrap_pass1(
            initial_records,
            sampled_records,
            expected_problems=expected_problems,
            expected_n=expected_n,
            replicates=replicates,
            seed=seed,
        ),
        "sampled_vs_top_k16": paired_bootstrap_pass1(
            top_k16_records,
            sampled_records,
            expected_problems=expected_problems,
            expected_n=expected_n,
            replicates=replicates,
            seed=seed,
        ),
    }


def _duration_seconds(value: str) -> int:
    parts = value.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"walltime must use HH:MM:SS, got {value!r}")
    hours, minutes, seconds = (int(part) for part in parts)
    if hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"invalid walltime: {value!r}")
    return hours * 3600 + minutes * 60 + seconds


def _training_compute(
    acceptance: dict,
    *,
    walltime: str,
    gpu_count: int,
    su_per_gpu_hour: float,
) -> dict:
    lengths = acceptance["lengths"]
    rollout_samples = int(lengths["rollout_records"])
    response_mean = float(lengths["response_mean_tokens"]["mean"])
    generated_response_tokens = round(rollout_samples * response_mean)
    walltime_seconds = _duration_seconds(walltime)
    gpu_hours = walltime_seconds * gpu_count / 3600
    return {
        "walltime": walltime,
        "walltime_seconds": walltime_seconds,
        "gpu_count": gpu_count,
        "gpu_hours": gpu_hours,
        "estimated_su": gpu_hours * su_per_gpu_hour,
        "rollout_samples": rollout_samples,
        "generated_response_tokens": generated_response_tokens,
        "mean_response_tokens": response_mean,
        "hit_length_limit_ratio": float(
            lengths["hit_length_limit_ratio"]["mean"]
        ),
        "effective_answer_presence_ratio": float(
            lengths["effective_answer_presence_ratio"]
        ),
    }


def build_compute_comparison(
    top_k16_acceptance: dict,
    sampled_acceptance: dict,
    *,
    top_k16_walltime: str,
    sampled_walltime: str,
    gpu_count: int,
    su_per_gpu_hour: float,
) -> dict:
    top_k16 = _training_compute(
        top_k16_acceptance,
        walltime=top_k16_walltime,
        gpu_count=gpu_count,
        su_per_gpu_hour=su_per_gpu_hour,
    )
    sampled = _training_compute(
        sampled_acceptance,
        walltime=sampled_walltime,
        gpu_count=gpu_count,
        su_per_gpu_hour=su_per_gpu_hour,
    )
    return {
        "top_k16": top_k16,
        "sampled_token": sampled,
        "sampled_over_top_k16": {
            "generated_response_token_ratio": (
                sampled["generated_response_tokens"]
                / top_k16["generated_response_tokens"]
            ),
            "gpu_hour_ratio": sampled["gpu_hours"] / top_k16["gpu_hours"],
            "estimated_su_ratio": sampled["estimated_su"] / top_k16["estimated_su"],
        },
        "accounting_note": (
            "GPU hours and SU use PBS walltime times four assigned A100 GPUs because "
            "NSCC accounting reports resources_used.ngpus=0 for these jobs."
        ),
    }


def _summary_task(summary: dict, task: str) -> dict:
    return summary["models"]["final"]["tasks"][task]


def _top_k16_conclusion(classes: dict[str, str]) -> str:
    values = list(classes.values())
    if all(value == "reliable_gain" for value in values):
        return "SAMPLED_TOKEN_RELIABLE_GAIN_ON_BOTH_TASKS"
    if all(value == "reliable_drop" for value in values):
        return "SAMPLED_TOKEN_RELIABLE_DROP_ON_BOTH_TASKS"
    if any(value in {"reliable_gain", "reliable_drop"} for value in values):
        return "MIXED_TASK_LEVEL_EVIDENCE"
    return "INCONCLUSIVE_NO_RELIABLE_DIFFERENCE"


def render_report(result: dict) -> str:
    lines = [
        "# NSCC OPD sampled-token control report",
        "",
        "- Checkpoint: `global_step_528` (precommitted full-epoch checkpoint)",
        "- Protocol: identical frozen D12 independent evaluation protocol",
        "- Main metric: per-problem mean correctness (pass@1 estimate)",
        "- Uncertainty: paired bootstrap by problem, 10,000 replicates, seed 2026",
        f"- Sampled-token vs initial verdict: `{result['conclusion']['sampled_vs_initial']['verdict']}`",
        f"- Sampled-token vs top-k16 classification: `{result['conclusion']['sampled_vs_top_k16']}`",
        "",
        "## Held-out results",
        "",
        "| Task | Initial | top-k16 OPD | sampled-token | sampled−initial 95% CI | sampled−top-k16 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in REQUIRED_TASKS:
        task_result = result["tasks"][task]
        metrics = task_result["metrics"]
        vs_initial = task_result["sampled_vs_initial"]
        vs_top_k16 = task_result["sampled_vs_top_k16"]
        lines.append(
            f"| {task} | {metrics['initial']['pass_at_1_estimate']:.4f} | "
            f"{metrics['top_k16']['pass_at_1_estimate']:.4f} | "
            f"{metrics['sampled_token']['pass_at_1_estimate']:.4f} | "
            f"{vs_initial['difference']:+.4f} "
            f"[{vs_initial['ci_95'][0]:+.4f}, {vs_initial['ci_95'][1]:+.4f}] | "
            f"{vs_top_k16['difference']:+.4f} "
            f"[{vs_top_k16['ci_95'][0]:+.4f}, {vs_top_k16['ci_95'][1]:+.4f}] |"
        )

    compute = result["training_compute"]
    lines.extend(
        [
            "",
            "## Training-compute comparison",
            "",
            "| Run | Rollouts | Generated response tokens | GPU·h | Estimated SU | Mean response | Hit limit | Effective answer |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, label in (("top_k16", "top-k16 OPD"), ("sampled_token", "sampled-token")):
        item = compute[key]
        lines.append(
            f"| {label} | {item['rollout_samples']:,} | "
            f"{item['generated_response_tokens']:,} | {item['gpu_hours']:.3f} | "
            f"{item['estimated_su']:.2f} | {item['mean_response_tokens']:.1f} | "
            f"{item['hit_length_limit_ratio']:.4f} | "
            f"{item['effective_answer_presence_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            compute["accounting_note"],
            "",
            "The control changes only `log_prob_top_k: 16→0`; it does not establish a new training recipe outside the preregistered boundary.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--top-k16-dir", type=Path, required=True)
    parser.add_argument("--sampled-dir", type=Path, required=True)
    parser.add_argument("--top-k16-training-acceptance", type=Path, required=True)
    parser.add_argument("--sampled-training-acceptance", type=Path, required=True)
    parser.add_argument("--top-k16-walltime", required=True)
    parser.add_argument("--sampled-walltime", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--failure-cases", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    outputs = (args.output_json, args.report_md, args.failure_cases)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite sampled-token outputs: {existing}")

    baseline_summary = load_json(args.baseline_dir / "summary.json")
    top_k16_summary = load_json(args.top_k16_dir / "summary.json")
    sampled_summary = load_json(args.sampled_dir / "summary.json")
    contract = verify_sampled_contract(
        baseline_summary, top_k16_summary, sampled_summary
    )
    acceptances = {
        "baseline": load_json(args.baseline_dir / "acceptance.json"),
        "top_k16_eval": load_json(args.top_k16_dir / "acceptance.json"),
        "sampled_eval": load_json(args.sampled_dir / "acceptance.json"),
        "top_k16_training": load_json(args.top_k16_training_acceptance),
        "sampled_training": load_json(args.sampled_training_acceptance),
    }
    top_training = acceptances["top_k16_training"]
    sampled_training = acceptances["sampled_training"]
    training_gates = (
        top_training.get("passed") is True
        and top_training.get("d7_9_status") == "PASS_TRAINING_COMPLETE"
        and top_training.get("full_data_one_epoch") is True
        and top_training.get("distillation_mode") in {None, "top_k"}
        and sampled_training.get("passed") is True
        and sampled_training.get("d7_9_status") == "PASS_TRAINING_COMPLETE"
        and sampled_training.get("full_data_one_epoch") is True
        and sampled_training.get("distillation_mode") == "sampled_token"
    )
    evaluation_gates = all(
        acceptances[key].get("passed") is True
        for key in ("baseline", "top_k16_eval", "sampled_eval")
    )
    if not training_gates:
        raise ValueError("top-k16 or sampled-token training acceptance did not pass")
    if not evaluation_gates:
        raise ValueError("baseline, top-k16, or sampled-token evaluation did not pass")

    protocol = timeline_evaluation_protocol()
    task_results = {}
    failure_cases = {}
    for task in REQUIRED_TASKS:
        task_contract = protocol["tasks"][task]
        expected_problems = int(task_contract["expected_problems"])
        expected_n = int(task_contract["n"])
        initial_records = load_jsonl(
            args.baseline_dir / "student" / f"{task}.jsonl"
        )
        top_k16_records = load_jsonl(
            args.top_k16_dir / "final" / f"{task}.jsonl"
        )
        sampled_records = load_jsonl(
            args.sampled_dir / "final" / f"{task}.jsonl"
        )
        comparisons = compare_task(
            initial_records,
            top_k16_records,
            sampled_records,
            expected_problems=expected_problems,
            expected_n=expected_n,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        task_results[task] = {
            "metrics": {
                "initial": baseline_summary["models"]["student"]["tasks"][task],
                "teacher": baseline_summary["models"]["teacher"]["tasks"][task],
                "top_k16": _summary_task(top_k16_summary, task),
                "sampled_token": _summary_task(sampled_summary, task),
            },
            **comparisons,
        }
        failure_cases[task] = {
            "sampled_vs_initial": build_failure_cases(
                initial_records, sampled_records, expected_n=expected_n
            ),
            "sampled_vs_top_k16": build_failure_cases(
                top_k16_records, sampled_records, expected_n=expected_n
            ),
        }

    initial_classes = {
        task: task_results[task]["sampled_vs_initial"]["classification"]
        for task in REQUIRED_TASKS
    }
    top_k16_classes = {
        task: task_results[task]["sampled_vs_top_k16"]["classification"]
        for task in REQUIRED_TASKS
    }
    credible = (
        training_gates
        and evaluation_gates
        and contract["protocol_identical"]
        and contract["input_hashes_identical"]
        and contract["checkpoints_precommitted"]
        and not contract["dev_used_for_selection"]
        and contract["models_distinct"]
    )
    result = {
        "passed": credible,
        "evaluation_status": "TECHNICALLY_COMPLETE" if credible else "INVALID_PROCESS",
        "process": {
            **contract,
            "training_gates_passed": training_gates,
            "evaluation_gates_passed": evaluation_gates,
        },
        "protocol": protocol,
        "input_sha256": sampled_summary["input_sha256"],
        "models": {
            "initial": baseline_summary["plan"]["worker_assignments"][0]["model"],
            "top_k16": top_k16_summary["plan"]["worker_assignment"]["model"],
            "sampled_token": sampled_summary["plan"]["worker_assignment"]["model"],
        },
        "tasks": task_results,
        "dev_diagnostic": {
            "initial": baseline_summary["models"]["student"]["tasks"]["dev_256"],
            "top_k16": _summary_task(top_k16_summary, "dev_256"),
            "sampled_token": _summary_task(sampled_summary, "dev_256"),
            "not_used_for_checkpoint_selection": True,
        },
        "training_compute": build_compute_comparison(
            top_training,
            sampled_training,
            top_k16_walltime=args.top_k16_walltime,
            sampled_walltime=args.sampled_walltime,
            gpu_count=4,
            su_per_gpu_hour=64,
        ),
        "conclusion": {
            "sampled_vs_initial": conclude_independent_evaluation(initial_classes),
            "sampled_vs_top_k16": _top_k16_conclusion(top_k16_classes),
            "task_classifications_vs_top_k16": top_k16_classes,
        },
        "wandb": {
            "top_k16": top_training.get("wandb", {}).get("run_url"),
            "sampled_token": sampled_training.get("wandb", {}).get("run_url"),
        },
    }

    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.failure_cases.write_text(
        json.dumps(failure_cases, indent=2, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    args.report_md.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
