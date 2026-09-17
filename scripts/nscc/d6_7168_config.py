#!/usr/bin/env python3
"""Build and verify the independent Timeline D6 7168-token validation config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from d5_pilot_config import (
    _replace_override,
    build_pilot_overrides,
    overrides_to_mapping,
)


LONG_VALIDATION_REPLACEMENTS = {
    "data.train_batch_size": "32",
    "data.max_response_length": "7168",
    "actor_rollout_ref.actor.ppo_mini_batch_size": "32",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "8192",
    "actor_rollout_ref.rollout.max_num_batched_tokens": "8192",
    "actor_rollout_ref.rollout.max_model_len": "8192",
    "actor_rollout_ref.rollout.n": "4",
    "actor_rollout_ref.rollout.enforce_eager": "False",
    "trainer.experiment_name": "d6_validate_7168",
    "trainer.total_training_steps": "8",
    "trainer.save_freq": "8",
    "trainer.max_actor_ckpt_to_keep": "1",
    "trainer.resume_mode": "disable",
}


def build_7168_overrides(*, project_root: Path, run_root: Path) -> list[str]:
    """Change only formal-candidate controls from the G1/D5-validated profile."""
    overrides = build_pilot_overrides(
        project_root=project_root,
        run_root=run_root,
        phase="primary",
    )
    for key, value in LONG_VALIDATION_REPLACEMENTS.items():
        _replace_override(overrides, key, value)
    return overrides


def _get(config: dict, dotted_key: str):
    value = config
    for component in dotted_key.split("."):
        value = value[component]
    return value


def verify_resolved_config(path: Path) -> dict:
    """Fail before training if Hydra differs from the frozen D6 contract."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
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
        "actor_rollout_ref.rollout.log_prob_top_k": 16,
        "actor_rollout_ref.rollout.top_k_strategy": "only_stu",
        "actor_rollout_ref.rollout.reward_weight_mode": "student_p",
        "reward_model.micro_batch_size_per_gpu": 1,
        "trainer.n_gpus_per_node": 4,
        "trainer.save_freq": 8,
        "trainer.test_freq": -1,
        "trainer.total_training_steps": 8,
        "trainer.resume_mode": "disable",
    }
    mismatches = {}
    for key, expected_value in expected.items():
        actual_value = _get(config, key)
        if actual_value != expected_value:
            mismatches[key] = {"expected": expected_value, "actual": actual_value}
    if mismatches:
        raise RuntimeError(f"resolved 7168 validation config mismatch: {mismatches}")
    return {"passed": True, "checked": expected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--format", choices=["json", "lines"], default="json")
    parser.add_argument("--verify-resolved", type=Path)
    args = parser.parse_args()

    if args.verify_resolved:
        print(json.dumps(verify_resolved_config(args.verify_resolved), indent=2))
        return
    if args.run_root is None:
        parser.error("--run-root is required when building overrides")

    overrides = build_7168_overrides(
        project_root=args.project_root,
        run_root=args.run_root,
    )
    if args.format == "lines":
        print("\n".join(overrides))
    else:
        print(json.dumps({"overrides": overrides}, indent=2))


if __name__ == "__main__":
    main()
