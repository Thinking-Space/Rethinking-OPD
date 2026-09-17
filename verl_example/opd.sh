#!/usr/bin/env bash
# On-policy distillation | text | vLLM rollout | FSDP training | NVIDIA GPUs

set -x

ray stop --force

ray start --head
sleep 5

export WANDB_DIR=${WANDB_DIR:-wandb}
export WANDB_MODE=${WANDB_MODE:-online}
mkdir -p "$WANDB_DIR"

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-DeepSeek-R1-Distill-Qwen-1.5B}
TEACHER_MODEL=${TEACHER_MODEL:-JustRL-DeepSeek-1.5B}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-4}

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1} # k1, forward_kl_topk
use_policy_gradient=${USE_POLICY_GRADIENT:-True} # k1 -> True, forward_kl_topk -> False
distillation_topk=${DISTILLATION_TOPK:-16}

train_batch_size=${TRAIN_BATCH_SIZE:-64}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-7168}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}

actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.7}
teacher_tp=${TEACHER_TP:-1}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.5}

total_epochs=${TOTAL_EPOCHS:-1}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:--1}

project_name=${PROJECT_NAME:-verl_opd}
student_model_name=${STUDENT_MODEL##*/}
teacher_model_name=${TEACHER_MODEL##*/}
timestamp=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
experiment_name=${EXPERIMENT_NAME:-mode-${distillation_loss_mode}-stu-${student_model_name}_tch-${teacher_model_name}_topk-${distillation_topk}-${timestamp}}
ckpt_dir=${CKPT_DIR:-checkpoints/${project_name}/${experiment_name}}

for arg in "$@"; do
    case "$arg" in
        trainer.default_local_dir=*)
            ckpt_dir=${arg#trainer.default_local_dir=}
            ;;
    esac
done
# ---- end user-adjustable ----

# gsm8k_train=$HOME/data/gsm8k/train.parquet
# gsm8k_test=$HOME/data/gsm8k/test.parquet
# math_train=$HOME/data/math/train.parquet
# math_test=$HOME/data/math/test.parquet

# train_files="['$gsm8k_train', '$math_train']"
# val_files="['$gsm8k_test', '$math_test']"

train_data=${TRAIN_DATA:-datasets/DAPO-Math-17k/DAPO-Math.parquet}
val_data=${VAL_DATA:-datasets/test_data/AIME24/test.parquet} # Not matter because test_freq is -1.

train_files="['$train_data']"
val_files="['$val_data']"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
    +data.apply_chat_template_kwargs.enable_thinking=False

)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=4
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.default_local_dir=${ckpt_dir}
)

EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
    reward.custom_reward_function.path=verl/recipe/r1_ascend/deepscaler.py
    reward.custom_reward_function.name=compute_score
)

########################### launch ###########################
REPRO_DIR="${ckpt_dir}/repro"
mkdir -p "${REPRO_DIR}"

script_path=$(readlink -f "${BASH_SOURCE[0]}")
cp "${script_path}" "${REPRO_DIR}/$(basename "${script_path}")"

{
    printf '%q ' "${script_path}"
    printf '%q ' "$@"
    printf '\n'
} > "${REPRO_DIR}/command.txt"

{
    printf 'project_name=%s\n' "${project_name}"
    printf 'experiment_name=%s\n' "${experiment_name}"
    printf 'ckpt_dir=%s\n' "${ckpt_dir}"
    printf 'student_model=%s\n' "${STUDENT_MODEL}"
    printf 'teacher_model=%s\n' "${TEACHER_MODEL}"
    printf 'timestamp=%s\n' "${timestamp}"
} > "${REPRO_DIR}/run_metadata.txt"

git_repo_dir="verl"
printf '%s\n' "${git_repo_dir}" > "${REPRO_DIR}/git_repo.txt"
git -C "${git_repo_dir}" rev-parse HEAD > "${REPRO_DIR}/git_commit.txt" 2>/dev/null || true
git -C "${git_repo_dir}" status --short > "${REPRO_DIR}/git_status.txt" 2>/dev/null || true
git -C "${git_repo_dir}" diff > "${REPRO_DIR}/git_diff.patch" 2>/dev/null || true
(env | grep -v -E '(^|_)(API_)?KEY=|TOKEN=|PASSWORD=|PASS=|SECRET=|CREDENTIAL=|COOKIE=' | sort || true) > "${REPRO_DIR}/env.txt"

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
