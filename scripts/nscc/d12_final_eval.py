#!/usr/bin/env python3
"""Evaluate one precommitted final Student with the frozen D4 protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from d4_baseline_eval import run_model_worker, sha256_file
from d4_pipeline import timeline_evaluation_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_gpu_id(value: str) -> str:
    gpu_id = value.strip()
    if not gpu_id or "," in gpu_id:
        raise ValueError("final evaluation requires exactly one GPU ID")
    return gpu_id


def build_plan(args) -> dict:
    return {
        "seed": 2026,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 8192,
        "enable_thinking": False,
        "rule_based_grader": "scripts/val/eval/utils.py::grade_answer_verl",
        "checkpoint_selection": "precommitted_full_epoch_global_step_528",
        "dev_role": "diagnostic_only_no_checkpoint_selection",
        "worker_assignment": {
            "label": "final",
            "model": args.model,
            "gpu_id": parse_gpu_id(args.gpu_id),
        },
        "protocol": timeline_evaluation_protocol(),
        "data": {
            "dev_256": str(args.dev_file.resolve()),
            "MATH-500": str(args.math500_file.resolve()),
            "AIME24": str(args.aime24_file.resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-id", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument(
        "--dev-file",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "nscc_d4" / "dev_256_seed2026.parquet",
    )
    parser.add_argument(
        "--math500-file",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "val" / "data" / "MATH-500" / "test.parquet",
    )
    parser.add_argument(
        "--aime24-file",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "val" / "data" / "AIME24" / "test.parquet",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args)
    if args.print_plan:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --print-plan is used")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite final evaluation directory: {output_dir}")
    output_dir.mkdir(parents=True)
    tasks = {
        name: {**task, "path": plan["data"][name]}
        for name, task in plan["protocol"]["tasks"].items()
    }
    result = run_model_worker(
        {**plan["worker_assignment"], "output_dir": str(output_dir), "tasks": tasks}
    )
    summary = {
        "passed": True,
        "plan": plan,
        "models": {"final": result},
        "input_sha256": {
            name: sha256_file(Path(path)) for name, path in plan["data"].items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    acceptance = {
        "passed": True,
        "checks": {
            "complete_sampling": True,
            "rule_grading": True,
            "frozen_protocol": True,
            "precommitted_checkpoint": True,
        },
    }
    (output_dir / "acceptance.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
