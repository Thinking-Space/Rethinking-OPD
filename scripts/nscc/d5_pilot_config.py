#!/usr/bin/env python3
"""Build and verify the Timeline D5 4096-token Pilot overrides."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from d4_smoke_config import build_smoke_overrides, overrides_to_mapping


PILOT_REPLACEMENTS = {
    "data.train_batch_size": "16",
    "data.max_response_length": "4096",
    "actor_rollout_ref.actor.ppo_mini_batch_size": "16",
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "5120",
    "actor_rollout_ref.rollout.max_num_batched_tokens": "5120",
    "actor_rollout_ref.rollout.max_model_len": "5120",
    "actor_rollout_ref.rollout.n": "4",
    "trainer.save_freq": "10",
    "trainer.max_actor_ckpt_to_keep": "4",
}


def _replace_override(overrides: list[str], key: str, value: str) -> None:
    for index, override in enumerate(overrides):
        current_key, separator, _ = override.partition("=")
        if separator and current_key.lstrip("+") == key:
            prefix = "+" if current_key.startswith("+") else ""
            overrides[index] = f"{prefix}{key}={value}"
            return
    overrides.append(f"{key}={value}")


def build_pilot_overrides(*, project_root: Path, run_root: Path, phase: str) -> list[str]:
    """Reuse the G1-validated upstream launcher mapping and change only Pilot controls."""
    if phase not in {"primary", "resume"}:
        raise ValueError("phase must be primary or resume")

    overrides = build_smoke_overrides(
        project_root=project_root,
        run_root=run_root,
        phase=phase,
    )
    for key, value in PILOT_REPLACEMENTS.items():
        _replace_override(overrides, key, value)

    phase_values = {
        "trainer.experiment_name": f"d5_pilot_4096_{phase}",
        "trainer.total_training_steps": "30" if phase == "primary" else "32",
    }
    for key, value in phase_values.items():
        _replace_override(overrides, key, value)

    _replace_override(overrides, "actor_rollout_ref.rollout.enforce_eager", "False")
    if phase == "resume":
        _replace_override(
            overrides,
            "trainer.resume_from_path",
            str(run_root.resolve() / "checkpoints/global_step_30"),
        )
    return overrides


def _get(config: dict, dotted_key: str):
    value = config
    for component in dotted_key.split("."):
        value = value[component]
    return value


def verify_resolved_config(path: Path, *, phase: str) -> dict:
    """Fail before GPU training if Hydra did not resolve to the frozen Pilot contract."""
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "algorithm.adv_estimator": "token_reward_direct",
        "data.train_batch_size": 16,
        "data.max_prompt_length": 1024,
        "data.max_response_length": 4096,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 16,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 5120,
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 5120,
        "actor_rollout_ref.rollout.max_model_len": 5120,
        "actor_rollout_ref.rollout.n": 4,
        "actor_rollout_ref.rollout.enforce_eager": False,
        "actor_rollout_ref.rollout.log_prob_top_k": 16,
        "actor_rollout_ref.rollout.top_k_strategy": "only_stu",
        "actor_rollout_ref.rollout.reward_weight_mode": "student_p",
        "reward_model.micro_batch_size_per_gpu": 1,
        "trainer.n_gpus_per_node": 4,
        "trainer.save_freq": 10,
        "trainer.test_freq": -1,
        "trainer.total_training_steps": 30 if phase == "primary" else 32,
        "trainer.resume_mode": "disable" if phase == "primary" else "resume_path",
    }
    mismatches = {}
    for key, expected_value in expected.items():
        actual_value = _get(config, key)
        if actual_value != expected_value:
            mismatches[key] = {"expected": expected_value, "actual": actual_value}
    if mismatches:
        raise RuntimeError(f"resolved Pilot config mismatch: {mismatches}")
    return {"passed": True, "phase": phase, "checked": expected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--phase", choices=["primary", "resume"], required=True)
    parser.add_argument("--format", choices=["json", "lines"], default="json")
    parser.add_argument("--verify-resolved", type=Path)
    args = parser.parse_args()

    if args.verify_resolved:
        print(json.dumps(verify_resolved_config(args.verify_resolved, phase=args.phase), indent=2))
        return
    if args.run_root is None:
        parser.error("--run-root is required when building overrides")

    overrides = build_pilot_overrides(
        project_root=args.project_root,
        run_root=args.run_root,
        phase=args.phase,
    )
    if args.format == "lines":
        print("\n".join(overrides))
    else:
        print(json.dumps({"phase": args.phase, "overrides": overrides}, indent=2))


if __name__ == "__main__":
    main()
