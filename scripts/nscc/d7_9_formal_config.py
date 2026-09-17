#!/usr/bin/env python3
"""Build and verify the frozen Timeline D7-9 formal-training config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from d5_pilot_config import _replace_override
from d6_7168_config import build_7168_overrides, overrides_to_mapping


TOTAL_TRAINING_STEPS = 528
FORMAL_EXPERIMENT_NAME = "d7_9_formal_opd_seed2026"
FORMAL_REPLACEMENTS = {
    "trainer.experiment_name": FORMAL_EXPERIMENT_NAME,
    "trainer.total_training_steps": str(TOTAL_TRAINING_STEPS),
    "trainer.total_epochs": "1",
    "trainer.save_freq": "20",
    "trainer.max_actor_ckpt_to_keep": "4",
}


def build_formal_overrides(
    *,
    project_root: Path,
    run_root: Path,
    phase: str,
    resume_step: int | None = None,
) -> list[str]:
    """Reuse G2 exactly, changing only formal run and explicit-resume controls."""
    if phase not in {"primary", "resume"}:
        raise ValueError("phase must be primary or resume")
    if phase == "resume":
        if resume_step is None:
            raise ValueError("resume_step is required for resume")
        if not 1 <= resume_step < TOTAL_TRAINING_STEPS:
            raise ValueError("resume_step must be between 1 and 527")
    elif resume_step is not None:
        raise ValueError("resume_step is only valid for resume")

    project_root = project_root.resolve()
    run_root = run_root.resolve()
    overrides = build_7168_overrides(
        project_root=project_root,
        run_root=run_root,
    )
    for key, value in FORMAL_REPLACEMENTS.items():
        _replace_override(overrides, key, value)

    segment_name = "primary" if phase == "primary" else f"resume_{resume_step}"
    _replace_override(
        overrides,
        "trainer.rollout_data_dir",
        str(run_root / "rollouts" / segment_name),
    )
    _replace_override(
        overrides,
        "trainer.validation_data_dir",
        str(run_root / "validation" / segment_name),
    )
    _replace_override(
        overrides,
        "hydra.run.dir",
        str(run_root / "hydra" / segment_name),
    )

    if phase == "primary":
        _replace_override(overrides, "trainer.resume_mode", "disable")
        overrides = [
            item
            for item in overrides
            if item.partition("=")[0].lstrip("+") != "trainer.resume_from_path"
        ]
    else:
        _replace_override(overrides, "trainer.resume_mode", "resume_path")
        _replace_override(
            overrides,
            "trainer.resume_from_path",
            str(run_root / "checkpoints" / f"global_step_{resume_step}"),
        )
    return overrides


def _get(config: dict, dotted_key: str):
    value = config
    for component in dotted_key.split("."):
        value = value[component]
    return value


def verify_resolved_config(
    path: Path,
    *,
    run_root: Path,
    phase: str,
    resume_step: int | None = None,
    expected_log_prob_top_k: int = 16,
    expected_experiment_name: str = FORMAL_EXPERIMENT_NAME,
) -> dict:
    """Fail before training if Hydra differs from the frozen formal contract."""
    expected_overrides = build_formal_overrides(
        project_root=Path(__file__).resolve().parents[2],
        run_root=run_root,
        phase=phase,
        resume_step=resume_step,
    )
    expected_mapping = overrides_to_mapping(expected_overrides)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    expected_types = {
        "algorithm.adv_estimator": "token_reward_direct",
        "data.train_batch_size": 32,
        "data.max_prompt_length": 1024,
        "data.max_response_length": 7168,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 32,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 8192,
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 8192,
        "actor_rollout_ref.rollout.max_model_len": 8192,
        "actor_rollout_ref.rollout.n": 4,
        "actor_rollout_ref.rollout.enforce_eager": False,
        "actor_rollout_ref.rollout.log_prob_top_k": expected_log_prob_top_k,
        "actor_rollout_ref.rollout.top_k_strategy": "only_stu",
        "actor_rollout_ref.rollout.reward_weight_mode": "student_p",
        "reward_model.micro_batch_size_per_gpu": 1,
        "trainer.logger": ["console", "wandb"],
        "trainer.experiment_name": expected_experiment_name,
        "trainer.n_gpus_per_node": 4,
        "trainer.total_training_steps": TOTAL_TRAINING_STEPS,
        "trainer.total_epochs": 1,
        "trainer.save_freq": 20,
        "trainer.max_actor_ckpt_to_keep": 4,
        "trainer.test_freq": -1,
        "trainer.resume_mode": "disable" if phase == "primary" else "resume_path",
    }
    if phase == "resume":
        expected_types["trainer.resume_from_path"] = expected_mapping[
            "trainer.resume_from_path"
        ]

    mismatches = {}
    for key, expected_value in expected_types.items():
        actual_value = _get(config, key)
        if actual_value != expected_value:
            mismatches[key] = {"expected": expected_value, "actual": actual_value}
    if mismatches:
        raise RuntimeError(f"resolved formal config mismatch: {mismatches}")
    return {
        "passed": True,
        "phase": phase,
        "resume_step": resume_step,
        "checked": expected_types,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--phase", choices=["primary", "resume"], required=True)
    parser.add_argument("--resume-step", type=int)
    parser.add_argument("--format", choices=["json", "lines"], default="json")
    parser.add_argument("--verify-resolved", type=Path)
    args = parser.parse_args()

    if args.run_root is None:
        parser.error("--run-root is required")
    if args.verify_resolved:
        result = verify_resolved_config(
            args.verify_resolved,
            run_root=args.run_root,
            phase=args.phase,
            resume_step=args.resume_step,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    overrides = build_formal_overrides(
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
