#!/usr/bin/env python3
"""Analyze the frozen final Student against independent D4 baselines."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from d4_pipeline import (
    conclude_independent_evaluation,
    paired_bootstrap_pass1,
    timeline_evaluation_protocol,
)


REQUIRED_TASKS = ("MATH-500", "AIME24")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return records


def verify_evaluation_contract(baseline_summary: dict, final_summary: dict) -> dict:
    if baseline_summary.get("passed") is not True or final_summary.get("passed") is not True:
        raise ValueError("baseline and final summaries must both pass")
    if set(baseline_summary.get("models", {})) != {"student", "teacher"}:
        raise ValueError("baseline summary must contain exactly student and teacher")
    if set(final_summary.get("models", {})) != {"final"}:
        raise ValueError("final summary must contain exactly final")

    frozen = timeline_evaluation_protocol()
    baseline_protocol = baseline_summary.get("plan", {}).get("protocol")
    final_protocol = final_summary.get("plan", {}).get("protocol")
    if baseline_protocol != frozen or final_protocol != frozen:
        raise ValueError("frozen evaluation protocol differs from the timeline contract")
    if baseline_summary.get("input_sha256") != final_summary.get("input_sha256"):
        raise ValueError("baseline and final input hashes differ")

    final_plan = final_summary.get("plan", {})
    checkpoint_precommitted = (
        final_plan.get("checkpoint_selection")
        == "precommitted_full_epoch_global_step_528"
        and final_plan.get("dev_role") == "diagnostic_only_no_checkpoint_selection"
    )
    if not checkpoint_precommitted:
        raise ValueError("final checkpoint was not recorded as precommitted")
    return {
        "protocol_identical": True,
        "input_hashes_identical": True,
        "checkpoint_precommitted": True,
        "dev_used_for_selection": False,
    }


def _group_by_problem(records: Sequence[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        grouped[int(record["example_id"])].append(record)
    return grouped


def build_failure_cases(
    baseline_records: Sequence[dict],
    final_records: Sequence[dict],
    *,
    expected_n: int,
    limit: int = 25,
) -> dict:
    baseline = _group_by_problem(baseline_records)
    final = _group_by_problem(final_records)
    if set(baseline) != set(final):
        raise ValueError("baseline and final problem IDs differ")

    rows = []
    for example_id in sorted(baseline):
        baseline_items = baseline[example_id]
        final_items = final[example_id]
        if len(baseline_items) != expected_n or len(final_items) != expected_n:
            raise ValueError(f"problem {example_id} does not have n={expected_n} records")
        problem = str(final_items[0].get("problem", ""))
        if problem != str(baseline_items[0].get("problem", "")):
            raise ValueError(f"problem text differs for example_id={example_id}")
        baseline_score = sum(bool(item["correct"]) for item in baseline_items) / expected_n
        final_score = sum(bool(item["correct"]) for item in final_items) / expected_n
        rows.append(
            {
                "example_id": example_id,
                "problem": problem,
                "baseline_score": baseline_score,
                "final_score": final_score,
                "delta": final_score - baseline_score,
                "baseline_truncation_rate": sum(
                    str(item.get("finish_reason", "")) == "length" for item in baseline_items
                )
                / expected_n,
                "final_truncation_rate": sum(
                    str(item.get("finish_reason", "")) == "length" for item in final_items
                )
                / expected_n,
                "baseline_missing_box_rate": sum(
                    "\\boxed" not in str(item.get("response", "")) for item in baseline_items
                )
                / expected_n,
                "final_missing_box_rate": sum(
                    "\\boxed" not in str(item.get("response", "")) for item in final_items
                )
                / expected_n,
            }
        )
    regressions = sorted(
        (row for row in rows if row["delta"] < 0),
        key=lambda row: (row["delta"], row["example_id"]),
    )[:limit]
    improvements = sorted(
        (row for row in rows if row["delta"] > 0),
        key=lambda row: (-row["delta"], row["example_id"]),
    )[:limit]
    return {
        "regressions": regressions,
        "improvements": improvements,
        "regression_problem_count": sum(row["delta"] < 0 for row in rows),
        "improvement_problem_count": sum(row["delta"] > 0 for row in rows),
        "unchanged_problem_count": sum(row["delta"] == 0 for row in rows),
        "display_limit_per_direction": limit,
    }


def _summary_task(summary: dict, model: str, task: str) -> dict:
    return summary["models"][model]["tasks"][task]


def render_report(result: dict) -> str:
    conclusion = result["conclusion"]
    lines = [
        "# NSCC OPD independent evaluation report",
        "",
        f"- Verdict: `{conclusion['verdict']}`",
        f"- Stop-loss triggered: `{str(conclusion['stop_loss_triggered']).lower()}`",
        "- Final checkpoint: `global_step_528` (precommitted full-epoch checkpoint)",
        "- Dev role: diagnostic only; no checkpoint selection or test-set shopping",
        "- Main metric: per-problem mean correctness (pass@1 estimate)",
        "- Uncertainty: paired bootstrap by problem, 10,000 replicates, seed 2026",
        "",
        "## Held-out results",
        "",
        "| Task | Initial | Final | Teacher | Final−Initial 95% CI | Decision | Trunc. initial→final | Missing-box initial→final |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for task in REQUIRED_TASKS:
        task_result = result["tasks"][task]
        comparison = task_result["final_vs_initial"]
        initial = task_result["metrics"]["initial"]
        final = task_result["metrics"]["final"]
        teacher = task_result["metrics"]["teacher"]
        lines.append(
            f"| {task} | {initial['pass_at_1_estimate']:.4f} | "
            f"{final['pass_at_1_estimate']:.4f} | {teacher['pass_at_1_estimate']:.4f} | "
            f"{comparison['difference']:+.4f} "
            f"[{comparison['ci_95'][0]:+.4f}, {comparison['ci_95'][1]:+.4f}] | "
            f"{comparison['classification']} | "
            f"{initial['truncation_rate']:.3f}→{final['truncation_rate']:.3f} | "
            f"{initial['missing_box_rate']:.3f}→{final['missing_box_rate']:.3f} |"
        )

    process = result["process"]
    lines.extend(
        [
            "",
            "## Required conclusions",
            "",
            f"1. **Did the Student learn?** {result['answers']['did_student_learn']}",
            f"2. **Is the process credible?** {result['answers']['is_process_credible']}",
            f"3. **Is further compute justified?** {result['answers']['is_more_compute_justified']}",
            "",
            "## Process evidence",
            "",
            f"- Training gate passed: `{str(process['training_gate_passed']).lower()}`",
            f"- Baseline/final protocol identical: `{str(process['protocol_identical']).lower()}`",
            f"- Dataset hashes identical: `{str(process['input_hashes_identical']).lower()}`",
            f"- Complete independent sampling and rule grading: `{str(process['evaluation_acceptance_passed']).lower()}`",
            "",
            "The W&B training run is observational evidence only; acceptance comes from the frozen checkpoint and independent raw outputs.",
        ]
    )
    if result.get("wandb_url"):
        lines.extend(["", f"Training curves: {result['wandb_url']}"])
    lines.extend(
        [
            "",
            "## Stop-loss boundary",
            "",
            "Both required held-out tasks must have a paired 95% confidence interval whose lower bound is above zero. Otherwise this reproduction is reported as negative/inconclusive and no second seed or auxiliary control is launched automatically.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--training-acceptance", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--failure-cases", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    outputs = (args.output_json, args.report_md, args.failure_cases)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite analysis outputs: {existing}")

    baseline_summary = load_json(args.baseline_dir / "summary.json")
    final_summary = load_json(args.final_dir / "summary.json")
    baseline_acceptance = load_json(args.baseline_dir / "acceptance.json")
    final_acceptance = load_json(args.final_dir / "acceptance.json")
    training_acceptance = load_json(args.training_acceptance)
    contract = verify_evaluation_contract(baseline_summary, final_summary)
    training_gate_passed = (
        training_acceptance.get("passed") is True
        and training_acceptance.get("d7_9_status") == "PASS_TRAINING_COMPLETE"
        and training_acceptance.get("full_data_one_epoch") is True
    )
    evaluation_acceptance_passed = (
        baseline_acceptance.get("passed") is True
        and final_acceptance.get("passed") is True
    )
    if not training_gate_passed:
        raise ValueError("D7-9 training acceptance did not pass")
    if not evaluation_acceptance_passed:
        raise ValueError("baseline or final evaluation acceptance did not pass")

    protocol = timeline_evaluation_protocol()
    task_results = {}
    failure_cases = {}
    for task in REQUIRED_TASKS:
        task_contract = protocol["tasks"][task]
        expected_problems = int(task_contract["expected_problems"])
        expected_n = int(task_contract["n"])
        initial_records = load_jsonl(args.baseline_dir / "student" / f"{task}.jsonl")
        teacher_records = load_jsonl(args.baseline_dir / "teacher" / f"{task}.jsonl")
        final_records = load_jsonl(args.final_dir / "final" / f"{task}.jsonl")
        final_vs_initial = paired_bootstrap_pass1(
            initial_records,
            final_records,
            expected_problems=expected_problems,
            expected_n=expected_n,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        final_vs_teacher = paired_bootstrap_pass1(
            teacher_records,
            final_records,
            expected_problems=expected_problems,
            expected_n=expected_n,
            replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
        task_results[task] = {
            "metrics": {
                "initial": _summary_task(baseline_summary, "student", task),
                "teacher": _summary_task(baseline_summary, "teacher", task),
                "final": _summary_task(final_summary, "final", task),
            },
            "final_vs_initial": final_vs_initial,
            "final_vs_teacher": final_vs_teacher,
        }
        failure_cases[task] = build_failure_cases(
            initial_records,
            final_records,
            expected_n=expected_n,
        )

    conclusion = conclude_independent_evaluation(
        {
            task: task_results[task]["final_vs_initial"]["classification"]
            for task in REQUIRED_TASKS
        }
    )
    credible = (
        training_gate_passed
        and evaluation_acceptance_passed
        and contract["protocol_identical"]
        and contract["input_hashes_identical"]
        and contract["checkpoint_precommitted"]
        and not contract["dev_used_for_selection"]
    )
    if conclusion["verdict"] == "REPRODUCED_RELIABLE_GAIN":
        learned_answer = "Yes: both required tasks show a reliable paired gain."
        compute_answer = "Potentially yes, but only one predeclared optional expansion is allowed."
    elif conclusion["verdict"] == "NOT_REPRODUCED_RELIABLE_DROP":
        learned_answer = "No: at least one required task shows a reliable drop."
        compute_answer = "No: the stop-loss condition is triggered."
    else:
        learned_answer = "Not established: the two required tasks do not jointly show reliable gains."
        compute_answer = "No: the stop-loss condition is triggered."

    result = {
        "passed": credible,
        "evaluation_status": "TECHNICALLY_COMPLETE" if credible else "INVALID_PROCESS",
        "process": {
            **contract,
            "training_gate_passed": training_gate_passed,
            "evaluation_acceptance_passed": evaluation_acceptance_passed,
        },
        "protocol": protocol,
        "input_sha256": final_summary["input_sha256"],
        "final_model": final_summary["plan"]["worker_assignment"]["model"],
        "tasks": task_results,
        "dev_diagnostic": {
            "initial": _summary_task(baseline_summary, "student", "dev_256"),
            "teacher": _summary_task(baseline_summary, "teacher", "dev_256"),
            "final": _summary_task(final_summary, "final", "dev_256"),
            "not_used_for_checkpoint_selection": True,
        },
        "conclusion": conclusion,
        "answers": {
            "did_student_learn": learned_answer,
            "is_process_credible": (
                "Yes: training and evaluation gates passed with identical protocol and inputs."
                if credible
                else "No: at least one required process gate failed."
            ),
            "is_more_compute_justified": compute_answer,
        },
        "wandb_url": training_acceptance.get("wandb", {}).get("run_url"),
    }

    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.failure_cases.write_text(
        json.dumps(failure_cases, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.report_md.write_text(render_report(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
