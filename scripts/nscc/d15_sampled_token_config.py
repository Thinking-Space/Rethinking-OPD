#!/usr/bin/env python3
"""Build and verify the Timeline sampled-token control config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from d5_pilot_config import _replace_override
from d6_7168_config import overrides_to_mapping
from d7_9_formal_config import (
    TOTAL_TRAINING_STEPS,
    build_formal_overrides,
    verify_resolved_config as verify_formal_resolved_config,
)


EXPERIMENT_NAME = "sampled_token_control_opd_seed2026"


def build_sampled_token_overrides(
    *,
    project_root: Path,
    run_root: Path,
    phase: str,
    resume_step: int | None = None,
) -> list[str]:
    """Reuse the formal run and change only top-k plus its run label."""
    overrides = build_formal_overrides(
        project_root=project_root,
        run_root=run_root,
        phase=phase,
        resume_step=resume_step,
    )
    _replace_override(overrides, "actor_rollout_ref.rollout.log_prob_top_k", "0")
    _replace_override(overrides, "trainer.experiment_name", EXPERIMENT_NAME)
    return overrides


def verify_resolved_config(
    path: Path,
    *,
    run_root: Path,
    phase: str,
    resume_step: int | None = None,
) -> dict:
    """Reuse the typed formal contract with sampled-token expectations."""
    result = verify_formal_resolved_config(
        path,
        run_root=run_root,
        phase=phase,
        resume_step=resume_step,
        expected_log_prob_top_k=0,
        expected_experiment_name=EXPERIMENT_NAME,
    )
    result.update(
        {
            "distillation_mode": "sampled_token",
            "total_training_steps": TOTAL_TRAINING_STEPS,
            "log_prob_top_k": 0,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--phase", choices=["primary", "resume"], required=True)
    parser.add_argument("--resume-step", type=int)
    parser.add_argument("--format", choices=["json", "lines"], default="json")
    parser.add_argument("--verify-resolved", type=Path)
    args = parser.parse_args()

    if args.verify_resolved:
        result = verify_resolved_config(
            args.verify_resolved,
            run_root=args.run_root,
            phase=args.phase,
            resume_step=args.resume_step,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    overrides = build_sampled_token_overrides(
        project_root=args.project_root,
        run_root=args.run_root,
        phase=args.phase,
        resume_step=args.resume_step,
    )
    if args.format == "lines":
        print("\n".join(overrides))
    else:
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "resume_step": args.resume_step,
                    "overrides": overrides,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
