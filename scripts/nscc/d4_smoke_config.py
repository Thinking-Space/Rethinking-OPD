#!/usr/bin/env python3
"""Build the exact Hydra overrides for the Timeline D4 four-GPU smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_smoke_overrides(*, project_root: Path, run_root: Path, phase: str) -> list[str]:
    if phase not in {"primary", "resume"}:
        raise ValueError("phase must be primary or resume")
    project_root = project_root.resolve()
    run_root = run_root.resolve()
    total_steps = 5 if phase == "primary" else 8
    resume_mode = "disable" if phase == "primary" else "resume_path"
    overrides = [
        "algorithm.adv_estimator=token_reward_direct",
        "algorithm.grpo_outcome_weight=1.0",
        "data.shuffle=False",
        f"data.train_files={project_root / 'datasets/nscc_d4/train_seed2026.parquet'}",
        f"data.val_files={project_root / 'datasets/nscc_d4/dev_256_seed2026.parquet'}",
        "data.train_batch_size=8",
        "data.max_prompt_length=1024",
        "data.max_response_length=2048",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.return_raw_chat=True",
        f"actor_rollout_ref.model.path={project_root / 'model/DeepSeek-R1-Distill-Qwen-1.5B'}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_activation_offload=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=8",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=3072",
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size=1",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        "actor_rollout_ref.actor.fsdp_config.param_offload=True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
        "actor_rollout_ref.actor.fsdp_config.forward_prefetch=True",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=fp32",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.ref.fsdp_config.model_dtype=fp32",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.rollout.max_num_batched_tokens=3072",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.6",
        "actor_rollout_ref.rollout.max_model_len=3072",
        "actor_rollout_ref.rollout.n=1",
        "actor_rollout_ref.rollout.repetition_penalty=1.0",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "+actor_rollout_ref.rollout.log_prob_top_k=16",
        "+actor_rollout_ref.rollout.top_k_strategy=only_stu",
        "+actor_rollout_ref.rollout.reward_weight_mode=student_p",
        "+actor_rollout_ref.rollout.teacher_temperature=1.0",
        "reward_model.enable=True",
        "reward_model.enable_resource_pool=False",
        f"reward_model.model.path={project_root / 'model/JustRL-DeepSeek-1.5B'}",
        "reward_model.model.input_tokenizer=null",
        "reward_model.model.use_remove_padding=True",
        "reward_model.model.fsdp_config.param_offload=True",
        "+reward_model.model.dtype=fp32",
        "reward_model.micro_batch_size_per_gpu=1",
        "+reward_model.reward_kwargs.enable_format_reward=False",
        f"custom_reward_function.path={project_root / 'verl/verl/utils/reward_score/ttrl_math/__init__.py'}",
        "custom_reward_function.name=reward_func",
        "trainer.val_before_train=False",
        "trainer.logger=[console,wandb]",
        "trainer.project_name=OPD-NSCC",
        f"trainer.experiment_name=d4_smoke_{phase}",
        "trainer.n_gpus_per_node=4",
        "trainer.nnodes=1",
        "trainer.save_freq=5",
        "trainer.test_freq=-1",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={total_steps}",
        f"trainer.default_local_dir={run_root / 'checkpoints'}",
        f"trainer.rollout_data_dir={run_root / 'rollouts' / phase}",
        f"trainer.validation_data_dir={run_root / 'validation' / phase}",
        f"trainer.resume_mode={resume_mode}",
        "trainer.is_plot=False",
        "trainer.max_actor_ckpt_to_keep=3",
        f"hydra.run.dir={run_root / 'hydra' / phase}",
        "hydra.job.chdir=False",
    ]
    if phase == "resume":
        overrides.append(f"trainer.resume_from_path={run_root / 'checkpoints/global_step_5'}")
    return overrides


def overrides_to_mapping(overrides: list[str]) -> dict[str, str]:
    mapping = {}
    for override in overrides:
        key, value = override.split("=", 1)
        mapping[key.lstrip("+")] = value
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--phase", choices=["primary", "resume"], required=True)
    parser.add_argument("--format", choices=["json", "lines"], default="json")
    args = parser.parse_args()
    overrides = build_smoke_overrides(
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
