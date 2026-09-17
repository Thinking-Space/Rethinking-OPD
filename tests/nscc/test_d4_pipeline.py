import importlib.util
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / "nscc" / name
    assert path.exists(), f"missing D4 implementation: {path}"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_script(name: str, *args: str) -> dict:
    path = PROJECT_ROOT / "scripts" / "nscc" / name
    result = subprocess.run(
        [sys.executable, str(path), *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def wrapped(problem: str) -> str:
    return (
        "Solve the following math problem step by step. The last line of your response "
        "should be of the form Answer: $Answer (without quotes) where $Answer is the answer "
        f"to the problem.\n\n{problem}\n\nRemember to put your answer on its own line after \"Answer:\"."
    )


def test_freeze_indices_excludes_overlong_duplicates_and_test_overlap():
    pipeline = load_script("d4_pipeline.py")
    prompts = [
        wrapped("alpha"),
        wrapped("beta"),
        wrapped("alpha"),
        wrapped("gamma"),
        wrapped("test question"),
        wrapped("delta"),
        wrapped("epsilon"),
    ]
    lengths = [10, 10, 10, 1025, 10, 10, 10]

    result = pipeline.freeze_indices(
        prompts,
        lengths,
        ["test question"],
        dev_size=2,
        seed=2026,
        max_prompt_length=1024,
    )

    assert result["dev_indices"] == [0, 1]
    assert result["train_indices"] == [5, 6]
    assert result["excluded_overlong_indices"] == [3]
    assert result["excluded_test_overlap_indices"] == [4]
    assert result["excluded_duplicate_indices"] == [2]
    assert pipeline.extract_problem(wrapped("  x   +  y  ")) == "x + y"


def test_timeline_protocol_has_exact_baseline_sampling_contract():
    pipeline = load_script("d4_pipeline.py")

    protocol = pipeline.timeline_evaluation_protocol()

    assert protocol == {
        "temperature": 0.7,
        "top_p": 0.95,
        "prompt_template": "{problem} Please reason step by step, and put your final answer within \\boxed{{}}.",
        "tasks": {
            "dev_256": {"n": 1, "max_tokens": 2048, "expected_problems": 256},
            "MATH-500": {"n": 4, "max_tokens": 7168, "expected_problems": 500},
            "AIME24": {"n": 16, "max_tokens": 7168, "expected_problems": 30},
        },
    }


def test_summarize_baseline_reports_pass1_truncation_and_missing_box():
    pipeline = load_script("d4_pipeline.py")
    records = [
        {"example_id": 0, "sample_id": 0, "correct": True, "finish_reason": "stop", "response": "\\boxed{1}", "response_tokens": 5},
        {"example_id": 0, "sample_id": 1, "correct": False, "finish_reason": "length", "response": "thinking", "response_tokens": 10},
        {"example_id": 1, "sample_id": 0, "correct": True, "finish_reason": "stop", "response": "\\boxed{2}", "response_tokens": 4},
        {"example_id": 1, "sample_id": 1, "correct": True, "finish_reason": "stop", "response": "\\boxed{2}", "response_tokens": 6},
    ]

    summary = pipeline.summarize_baseline_records(records, expected_problems=2, expected_n=2)

    assert summary["record_count"] == 4
    assert summary["problem_count"] == 2
    assert summary["pass_at_1_estimate"] == pytest.approx(0.75)
    assert summary["truncation_rate"] == pytest.approx(0.25)
    assert summary["missing_box_rate"] == pytest.approx(0.25)
    assert summary["mean_response_tokens"] == pytest.approx(6.25)


def test_summarize_baseline_rejects_incomplete_sampling():
    pipeline = load_script("d4_pipeline.py")
    records = [
        {"example_id": 0, "sample_id": 0, "correct": True, "finish_reason": "stop", "response": "\\boxed{1}", "response_tokens": 5},
    ]

    with pytest.raises(ValueError, match="expected 4 records"):
        pipeline.summarize_baseline_records(records, expected_problems=2, expected_n=2)


def test_response_mask_report_proves_prompt_and_padding_are_excluded():
    diagnostics = load_script("d4_diagnostics.py")
    attention_mask = torch.tensor([[1, 1, 1, 1, 0]])
    responses = torch.tensor([[7, 8, 0]])
    response_mask = torch.tensor([[1, 1, 0]])

    report = diagnostics.build_response_mask_report(
        attention_mask=attention_mask,
        responses=responses,
        response_mask=response_mask,
        step=1,
    )

    assert report == {
        "passed": True,
        "step": 1,
        "batch_size": 1,
        "input_width": 5,
        "prompt_width": 2,
        "response_width": 3,
        "prompt_tokens_in_loss": 0,
        "padding_tokens_in_loss": 0,
        "response_tokens_in_loss": 2,
        "response_length_mean": 2.0,
        "response_length_p95": 2.0,
    }


def test_response_mask_report_rejects_padding_in_loss():
    diagnostics = load_script("d4_diagnostics.py")
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    responses = torch.tensor([[7, 0]])
    response_mask = torch.tensor([[1, 1]])

    with pytest.raises(RuntimeError, match="padding entered response loss"):
        diagnostics.build_response_mask_report(
            attention_mask=attention_mask,
            responses=responses,
            response_mask=response_mask,
            step=2,
        )


def test_prepare_data_print_plan_freezes_seed_split_and_overlap_sources():
    plan = run_script("d4_prepare_data.py", "--print-plan")

    assert plan["seed"] == 2026
    assert plan["dev_size"] == 256
    assert plan["max_prompt_length"] == 1024
    assert plan["source"].endswith("datasets/dapo-math-17k.parquet")
    assert [Path(path).parent.name for path in plan["test_overlap_sources"]] == ["MATH-500", "AIME24"]


def test_baseline_print_plan_assigns_one_model_per_gpu_and_exact_protocol():
    plan = run_script(
        "d4_baseline_eval.py",
        "--student-model",
        "/models/student",
        "--teacher-model",
        "/models/teacher",
        "--gpu-ids",
        "2,7",
        "--print-plan",
    )

    assert plan["worker_assignments"] == [
        {"label": "student", "model": "/models/student", "gpu_id": "2"},
        {"label": "teacher", "model": "/models/teacher", "gpu_id": "7"},
    ]
    assert plan["protocol"] == load_script("d4_pipeline.py").timeline_evaluation_protocol()


def test_final_eval_print_plan_reuses_frozen_baseline_protocol():
    plan = run_script(
        "d12_final_eval.py",
        "--model",
        "/models/final",
        "--gpu-id",
        "3",
        "--print-plan",
    )

    assert plan["worker_assignment"] == {
        "label": "final",
        "model": "/models/final",
        "gpu_id": "3",
    }
    assert plan["seed"] == 2026
    assert plan["dtype"] == "bfloat16"
    assert plan["tensor_parallel_size"] == 1
    assert plan["gpu_memory_utilization"] == 0.9
    assert plan["max_model_len"] == 8192
    assert plan["enable_thinking"] is False
    assert plan["rule_based_grader"] == "scripts/val/eval/utils.py::grade_answer_verl"
    assert plan["protocol"] == load_script("d4_pipeline.py").timeline_evaluation_protocol()


def test_paired_bootstrap_is_by_problem_and_detects_reliable_gain():
    pipeline = load_script("d4_pipeline.py")
    baseline = []
    final = []
    for example_id in range(3):
        for sample_id in range(2):
            common = {"example_id": example_id, "sample_id": sample_id}
            baseline.append({**common, "correct": False})
            final.append({**common, "correct": sample_id == 0})

    comparison = pipeline.paired_bootstrap_pass1(
        baseline,
        final,
        expected_problems=3,
        expected_n=2,
        replicates=200,
        seed=2026,
    )

    assert comparison["baseline_pass_at_1_estimate"] == pytest.approx(0.0)
    assert comparison["final_pass_at_1_estimate"] == pytest.approx(0.5)
    assert comparison["difference"] == pytest.approx(0.5)
    assert comparison["ci_95"] == pytest.approx([0.5, 0.5])
    assert comparison["classification"] == "reliable_gain"
    assert comparison["resampling_unit"] == "problem"


def test_paired_bootstrap_rejects_unpaired_sample_keys():
    pipeline = load_script("d4_pipeline.py")
    baseline = [{"example_id": 0, "sample_id": 0, "correct": False}]
    final = [{"example_id": 0, "sample_id": 1, "correct": True}]

    with pytest.raises(ValueError, match="paired sample keys differ"):
        pipeline.paired_bootstrap_pass1(
            baseline,
            final,
            expected_problems=1,
            expected_n=1,
            replicates=20,
            seed=2026,
        )


@pytest.mark.parametrize(
    ("classes", "verdict", "stop_loss_triggered"),
    [
        (
            {"MATH-500": "reliable_gain", "AIME24": "reliable_gain"},
            "REPRODUCED_RELIABLE_GAIN",
            False,
        ),
        (
            {"MATH-500": "reliable_gain", "AIME24": "inconclusive"},
            "INCONCLUSIVE_NO_JOINT_RELIABLE_GAIN",
            True,
        ),
        (
            {"MATH-500": "reliable_drop", "AIME24": "inconclusive"},
            "NOT_REPRODUCED_RELIABLE_DROP",
            True,
        ),
    ],
)
def test_independent_eval_conclusion_enforces_stop_loss(
    classes, verdict, stop_loss_triggered
):
    pipeline = load_script("d4_pipeline.py")

    conclusion = pipeline.conclude_independent_evaluation(classes)

    assert conclusion["verdict"] == verdict
    assert conclusion["stop_loss_triggered"] is stop_loss_triggered


def test_final_analyzer_accepts_only_identical_frozen_contract():
    pipeline = load_script("d4_pipeline.py")
    analyzer = load_script("d12_analyze_final.py")
    protocol = pipeline.timeline_evaluation_protocol()
    baseline = {
        "passed": True,
        "plan": {"protocol": protocol},
        "input_sha256": {"dev_256": "a", "MATH-500": "b", "AIME24": "c"},
        "models": {"student": {}, "teacher": {}},
    }
    final = {
        "passed": True,
        "plan": {
            "protocol": copy.deepcopy(protocol),
            "checkpoint_selection": "precommitted_full_epoch_global_step_528",
            "dev_role": "diagnostic_only_no_checkpoint_selection",
        },
        "input_sha256": {"dev_256": "a", "MATH-500": "b", "AIME24": "c"},
        "models": {"final": {}},
    }

    verified = analyzer.verify_evaluation_contract(baseline, final)

    assert verified["protocol_identical"] is True
    assert verified["input_hashes_identical"] is True
    assert verified["checkpoint_precommitted"] is True

    drifted = copy.deepcopy(final)
    drifted["plan"]["protocol"]["top_p"] = 0.9
    with pytest.raises(ValueError, match="frozen evaluation protocol differs"):
        analyzer.verify_evaluation_contract(baseline, drifted)


def test_failure_case_table_is_ranked_by_per_problem_delta():
    analyzer = load_script("d12_analyze_final.py")
    baseline = [
        {"example_id": 0, "sample_id": 0, "problem": "p0", "correct": True, "finish_reason": "stop", "response": "\\boxed{1}"},
        {"example_id": 0, "sample_id": 1, "problem": "p0", "correct": True, "finish_reason": "stop", "response": "\\boxed{1}"},
        {"example_id": 1, "sample_id": 0, "problem": "p1", "correct": False, "finish_reason": "length", "response": "thinking"},
        {"example_id": 1, "sample_id": 1, "problem": "p1", "correct": False, "finish_reason": "length", "response": "thinking"},
    ]
    final = [
        {"example_id": 0, "sample_id": 0, "problem": "p0", "correct": False, "finish_reason": "length", "response": "thinking"},
        {"example_id": 0, "sample_id": 1, "problem": "p0", "correct": True, "finish_reason": "stop", "response": "\\boxed{1}"},
        {"example_id": 1, "sample_id": 0, "problem": "p1", "correct": True, "finish_reason": "stop", "response": "\\boxed{2}"},
        {"example_id": 1, "sample_id": 1, "problem": "p1", "correct": True, "finish_reason": "stop", "response": "\\boxed{2}"},
    ]

    cases = analyzer.build_failure_cases(baseline, final, expected_n=2)

    assert cases["regressions"][0]["example_id"] == 0
    assert cases["regressions"][0]["delta"] == pytest.approx(-0.5)
    assert cases["improvements"][0]["example_id"] == 1
    assert cases["improvements"][0]["delta"] == pytest.approx(1.0)
    assert cases["regressions"][0]["final_truncation_rate"] == pytest.approx(0.5)


def test_smoke_config_primary_and_resume_preserve_timeline_contract():
    config = load_script("d4_smoke_config.py")
    primary = config.build_smoke_overrides(
        project_root=Path("/opd"), run_root=Path("/artifacts/run"), phase="primary"
    )
    resume = config.build_smoke_overrides(
        project_root=Path("/opd"), run_root=Path("/artifacts/run"), phase="resume"
    )
    primary_map = config.overrides_to_mapping(primary)
    resume_map = config.overrides_to_mapping(resume)

    expected_shared = {
        "data.train_batch_size": "8",
        "data.max_prompt_length": "1024",
        "data.max_response_length": "2048",
        "actor_rollout_ref.rollout.n": "1",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "3072",
        "reward_model.micro_batch_size_per_gpu": "1",
        "algorithm.adv_estimator": "token_reward_direct",
        "actor_rollout_ref.rollout.log_prob_top_k": "16",
        "actor_rollout_ref.rollout.top_k_strategy": "only_stu",
        "actor_rollout_ref.rollout.reward_weight_mode": "student_p",
        "actor_rollout_ref.actor.optim.lr": "1e-6",
        "trainer.n_gpus_per_node": "4",
        "trainer.test_freq": "-1",
    }
    for key, value in expected_shared.items():
        assert primary_map[key] == value
        assert resume_map[key] == value
    assert primary_map["trainer.total_training_steps"] == "5"
    assert primary_map["trainer.resume_mode"] == "disable"
    assert resume_map["trainer.total_training_steps"] == "8"
    assert resume_map["trainer.resume_mode"] == "resume_path"
    assert resume_map["trainer.resume_from_path"] == "/artifacts/run/checkpoints/global_step_5"


def test_pilot_config_reuses_smoke_profile_and_freezes_timeline_contract():
    config = load_script("d5_pilot_config.py")
    primary = config.build_pilot_overrides(
        project_root=Path("/opd"), run_root=Path("/artifacts/pilot"), phase="primary"
    )
    resume = config.build_pilot_overrides(
        project_root=Path("/opd"), run_root=Path("/artifacts/pilot"), phase="resume"
    )
    primary_map = config.overrides_to_mapping(primary)
    resume_map = config.overrides_to_mapping(resume)

    expected_shared = {
        "data.train_batch_size": "16",
        "data.max_prompt_length": "1024",
        "data.max_response_length": "4096",
        "actor_rollout_ref.actor.ppo_mini_batch_size": "16",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "1",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "5120",
        "actor_rollout_ref.actor.fsdp_config.param_offload": "True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": "True",
        "actor_rollout_ref.rollout.max_num_batched_tokens": "5120",
        "actor_rollout_ref.rollout.max_model_len": "5120",
        "actor_rollout_ref.rollout.n": "4",
        "actor_rollout_ref.rollout.enforce_eager": "False",
        "reward_model.micro_batch_size_per_gpu": "1",
        "algorithm.adv_estimator": "token_reward_direct",
        "actor_rollout_ref.rollout.log_prob_top_k": "16",
        "actor_rollout_ref.rollout.top_k_strategy": "only_stu",
        "actor_rollout_ref.rollout.reward_weight_mode": "student_p",
        "actor_rollout_ref.actor.optim.lr": "1e-6",
        "trainer.n_gpus_per_node": "4",
        "trainer.save_freq": "10",
        "trainer.test_freq": "-1",
    }
    for key, value in expected_shared.items():
        assert primary_map[key] == value
        assert resume_map[key] == value

    assert primary_map["trainer.total_training_steps"] == "30"
    assert primary_map["trainer.resume_mode"] == "disable"
    assert resume_map["trainer.total_training_steps"] == "32"
    assert resume_map["trainer.resume_mode"] == "resume_path"
    assert resume_map["trainer.resume_from_path"] == (
        "/artifacts/pilot/checkpoints/global_step_30"
    )

    allowed_phase_differences = {
        "trainer.experiment_name",
        "trainer.resume_mode",
        "trainer.resume_from_path",
        "trainer.rollout_data_dir",
        "trainer.validation_data_dir",
        "trainer.total_training_steps",
        "hydra.run.dir",
    }
    for key in set(primary_map) | set(resume_map):
        if key not in allowed_phase_differences:
            assert primary_map.get(key) == resume_map.get(key), key


def test_7168_validation_config_is_independent_and_matches_formal_candidate():
    config = load_script("d6_7168_config.py")
    overrides = config.build_7168_overrides(
        project_root=Path("/opd"), run_root=Path("/artifacts/long")
    )
    mapping = config.overrides_to_mapping(overrides)

    assert mapping["data.train_batch_size"] == "32"
    assert mapping["data.max_prompt_length"] == "1024"
    assert mapping["data.max_response_length"] == "7168"
    assert mapping["actor_rollout_ref.actor.ppo_mini_batch_size"] == "32"
    assert mapping["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] == "1"
    assert mapping["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"] == "8192"
    assert mapping["actor_rollout_ref.rollout.max_num_batched_tokens"] == "8192"
    assert mapping["actor_rollout_ref.rollout.max_model_len"] == "8192"
    assert mapping["actor_rollout_ref.rollout.n"] == "4"
    assert mapping["actor_rollout_ref.rollout.enforce_eager"] == "False"
    assert mapping["reward_model.micro_batch_size_per_gpu"] == "1"
    assert mapping["trainer.total_training_steps"] == "8"
    assert mapping["trainer.resume_mode"] == "disable"
    assert "trainer.resume_from_path" not in mapping
    assert mapping["trainer.save_freq"] == "8"
    assert mapping["trainer.n_gpus_per_node"] == "4"
    assert mapping["algorithm.adv_estimator"] == "token_reward_direct"
    assert mapping["actor_rollout_ref.rollout.log_prob_top_k"] == "16"
    assert mapping["actor_rollout_ref.rollout.top_k_strategy"] == "only_stu"
    assert mapping["actor_rollout_ref.rollout.reward_weight_mode"] == "student_p"


def test_formal_config_reuses_g2_candidate_and_changes_only_run_controls():
    d6 = load_script("d6_7168_config.py")
    formal = load_script("d7_9_formal_config.py")
    project_root = Path("/opd")
    run_root = Path("/artifacts/main")

    g2 = d6.overrides_to_mapping(
        d6.build_7168_overrides(project_root=project_root, run_root=run_root)
    )
    primary = formal.overrides_to_mapping(
        formal.build_formal_overrides(
            project_root=project_root,
            run_root=run_root,
            phase="primary",
        )
    )
    resume = formal.overrides_to_mapping(
        formal.build_formal_overrides(
            project_root=project_root,
            run_root=run_root,
            phase="resume",
            resume_step=300,
        )
    )

    expected_shared = {
        "data.train_batch_size": "32",
        "data.max_prompt_length": "1024",
        "data.max_response_length": "7168",
        "actor_rollout_ref.actor.ppo_mini_batch_size": "32",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "1",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "8192",
        "actor_rollout_ref.actor.fsdp_config.param_offload": "True",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": "True",
        "actor_rollout_ref.rollout.max_num_batched_tokens": "8192",
        "actor_rollout_ref.rollout.max_model_len": "8192",
        "actor_rollout_ref.rollout.n": "4",
        "actor_rollout_ref.rollout.enforce_eager": "False",
        "reward_model.micro_batch_size_per_gpu": "1",
        "algorithm.adv_estimator": "token_reward_direct",
        "actor_rollout_ref.rollout.log_prob_top_k": "16",
        "actor_rollout_ref.rollout.top_k_strategy": "only_stu",
        "actor_rollout_ref.rollout.reward_weight_mode": "student_p",
        "trainer.n_gpus_per_node": "4",
        "trainer.total_training_steps": "528",
        "trainer.total_epochs": "1",
        "trainer.save_freq": "20",
        "trainer.max_actor_ckpt_to_keep": "4",
        "trainer.test_freq": "-1",
    }
    for key, value in expected_shared.items():
        assert primary[key] == value
        assert resume[key] == value

    assert primary["trainer.experiment_name"] == "d7_9_formal_opd_seed2026"
    assert primary["trainer.resume_mode"] == "disable"
    assert "trainer.resume_from_path" not in primary
    assert resume["trainer.resume_mode"] == "resume_path"
    assert resume["trainer.resume_from_path"] == (
        "/artifacts/main/checkpoints/global_step_300"
    )
    assert primary["trainer.rollout_data_dir"] == "/artifacts/main/rollouts/primary"
    assert resume["trainer.rollout_data_dir"] == "/artifacts/main/rollouts/resume_300"

    allowed_d6_differences = {
        "trainer.experiment_name",
        "trainer.total_training_steps",
        "trainer.total_epochs",
        "trainer.save_freq",
        "trainer.max_actor_ckpt_to_keep",
        "trainer.rollout_data_dir",
        "trainer.validation_data_dir",
        "hydra.run.dir",
    }
    for key in set(g2) | set(primary):
        if key not in allowed_d6_differences:
            assert g2.get(key) == primary.get(key), key


def test_sampled_token_control_changes_only_top_k_and_run_controls():
    formal = load_script("d7_9_formal_config.py")
    control = load_script("d15_sampled_token_config.py")
    project_root = Path("/opd")
    run_root = Path("/artifacts/sampled-token")

    formal_map = formal.overrides_to_mapping(
        formal.build_formal_overrides(
            project_root=project_root,
            run_root=run_root,
            phase="primary",
        )
    )
    control_map = control.overrides_to_mapping(
        control.build_sampled_token_overrides(
            project_root=project_root,
            run_root=run_root,
            phase="primary",
        )
    )

    assert formal_map["actor_rollout_ref.rollout.log_prob_top_k"] == "16"
    assert control_map["actor_rollout_ref.rollout.log_prob_top_k"] == "0"
    assert control_map["trainer.experiment_name"] == (
        "sampled_token_control_opd_seed2026"
    )

    allowed_differences = {
        "actor_rollout_ref.rollout.log_prob_top_k",
        "trainer.experiment_name",
    }
    assert set(formal_map) == set(control_map)
    for key in formal_map:
        if key not in allowed_differences:
            assert formal_map[key] == control_map[key], key

    fixed_contract = {
        "algorithm.adv_estimator": "token_reward_direct",
        "data.train_batch_size": "32",
        "data.max_prompt_length": "1024",
        "data.max_response_length": "7168",
        "actor_rollout_ref.rollout.n": "4",
        "actor_rollout_ref.rollout.enforce_eager": "False",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": "8192",
        "actor_rollout_ref.actor.optim.lr": "1e-6",
        "reward_model.micro_batch_size_per_gpu": "1",
        "trainer.total_training_steps": "528",
        "trainer.total_epochs": "1",
        "trainer.resume_mode": "disable",
    }
    for key, expected in fixed_contract.items():
        assert control_map[key] == expected


def test_sampled_token_resolved_job_config_does_not_require_hydra_section(
    tmp_path,
):
    control = load_script("d15_sampled_token_config.py")
    run_root = tmp_path / "sampled-token"
    mapping = control.overrides_to_mapping(
        control.build_sampled_token_overrides(
            project_root=PROJECT_ROOT,
            run_root=run_root,
            phase="primary",
        )
    )
    resolved = {}
    for key, value in mapping.items():
        if key.startswith("hydra."):
            continue
        target = resolved
        components = key.split(".")
        for component in components[:-1]:
            target = target.setdefault(component, {})
        parsed_value = yaml.safe_load(value)
        if key == "actor_rollout_ref.actor.optim.lr":
            parsed_value = float(value)
        target[components[-1]] = parsed_value
    resolved_path = tmp_path / "resolved.yaml"
    resolved_path.write_text(yaml.safe_dump(resolved), encoding="utf-8")

    result = control.verify_resolved_config(
        resolved_path,
        run_root=run_root,
        phase="primary",
    )

    assert result["passed"] is True
    assert result["log_prob_top_k"] == 0


def test_sampled_token_comparison_reuses_frozen_contract_and_problem_bootstrap():
    pipeline = load_script("d4_pipeline.py")
    analyzer = load_script("d15_analyze_sampled.py")
    protocol = pipeline.timeline_evaluation_protocol()
    hashes = {"dev_256": "a", "MATH-500": "b", "AIME24": "c"}
    baseline = {
        "passed": True,
        "plan": {"protocol": protocol},
        "input_sha256": hashes,
        "models": {"student": {}, "teacher": {}},
    }

    def final_summary(model):
        return {
            "passed": True,
            "plan": {
                "protocol": protocol,
                "checkpoint_selection": "precommitted_full_epoch_global_step_528",
                "dev_role": "diagnostic_only_no_checkpoint_selection",
                "worker_assignment": {"model": model},
            },
            "input_sha256": hashes,
            "models": {"final": {}},
        }

    contract = analyzer.verify_sampled_contract(
        baseline,
        final_summary("/models/top-k16"),
        final_summary("/models/sampled-token"),
    )
    assert contract == {
        "protocol_identical": True,
        "input_hashes_identical": True,
        "checkpoints_precommitted": True,
        "dev_used_for_selection": False,
        "models_distinct": True,
    }

    initial = []
    top_k16 = []
    sampled = []
    for example_id in range(4):
        for sample_id in range(2):
            common = {
                "example_id": example_id,
                "sample_id": sample_id,
                "problem": f"p{example_id}",
            }
            initial.append({**common, "correct": False})
            top_k16.append({**common, "correct": sample_id == 0})
            sampled.append({**common, "correct": True})

    comparison = analyzer.compare_task(
        initial,
        top_k16,
        sampled,
        expected_problems=4,
        expected_n=2,
        replicates=200,
        seed=2026,
    )
    assert comparison["sampled_vs_initial"]["difference"] == pytest.approx(1.0)
    assert comparison["sampled_vs_top_k16"]["difference"] == pytest.approx(0.5)
    assert comparison["sampled_vs_initial"]["classification"] == "reliable_gain"
    assert comparison["sampled_vs_top_k16"]["classification"] == "reliable_gain"


def test_sampled_token_compute_comparison_reports_samples_tokens_gpu_hours_and_su():
    analyzer = load_script("d15_analyze_sampled.py")
    top_k16 = {
        "lengths": {
            "rollout_records": 67_584,
            "response_mean_tokens": {"mean": 5_000.0},
            "hit_length_limit_ratio": {"mean": 0.25},
            "effective_answer_presence_ratio": 0.75,
        }
    }
    sampled = {
        "lengths": {
            "rollout_records": 67_584,
            "response_mean_tokens": {"mean": 4_500.0},
            "hit_length_limit_ratio": {"mean": 0.20},
            "effective_answer_presence_ratio": 0.80,
        }
    }

    report = analyzer.build_compute_comparison(
        top_k16,
        sampled,
        top_k16_walltime="15:00:00",
        sampled_walltime="14:00:00",
        gpu_count=4,
        su_per_gpu_hour=64,
    )

    assert report["top_k16"]["rollout_samples"] == 67_584
    assert report["sampled_token"]["generated_response_tokens"] == 304_128_000
    assert report["top_k16"]["gpu_hours"] == pytest.approx(60.0)
    assert report["sampled_token"]["gpu_hours"] == pytest.approx(56.0)
    assert report["sampled_token"]["estimated_su"] == pytest.approx(3_584.0)


def test_sampled_token_closeout_and_eval_are_thin_reuse_wrappers():
    closeout = (
        PROJECT_ROOT / "scripts" / "nscc" / "d15_sampled_token_closeout.pbs"
    ).read_text(encoding="utf-8")
    evaluation = (
        PROJECT_ROOT / "scripts" / "nscc" / "d15_sampled_token_eval.pbs"
    ).read_text(encoding="utf-8")

    assert "ngpus=" not in closeout
    assert "d4_checkpoint_verify.py" in closeout
    assert "d7_9_analyze_formal.py" in closeout
    assert "--distillation-mode sampled_token" in closeout
    assert "python -m verl.trainer.main_ppo" not in closeout

    assert "select=1:ncpus=8:mem=120gb:ngpus=1" in evaluation
    assert "legacy_model_merger.py" in evaluation
    assert "d12_final_eval.py" in evaluation
    assert "d15_analyze_sampled.py" in evaluation
    assert "top_k16_immutable_manifest.txt" in evaluation
    assert 'wc -l < "$TOP_K16_IMMUTABLE_MANIFEST"' in evaluation
    assert "qstat -x" not in evaluation
    assert 'TOP_K16_FINAL_REPORT="$PROJECT_ROOT/reports/NSCC_OPD_FINAL_REPORT.md"' in evaluation
    assert "top_k16_training_evidence.md" in evaluation
    assert 'SAMPLED_RUN_ROOT/reports/training_qstat.txt' in evaluation
    assert "python -m verl.trainer.main_ppo" not in evaluation


def test_formal_config_rejects_invalid_or_ambiguous_resume_requests():
    formal = load_script("d7_9_formal_config.py")

    with pytest.raises(ValueError, match="resume_step is required"):
        formal.build_formal_overrides(
            project_root=Path("/opd"),
            run_root=Path("/run"),
            phase="resume",
        )
    with pytest.raises(ValueError, match="between 1 and 527"):
        formal.build_formal_overrides(
            project_root=Path("/opd"),
            run_root=Path("/run"),
            phase="resume",
            resume_step=528,
        )
    with pytest.raises(ValueError, match="only valid for resume"):
        formal.build_formal_overrides(
            project_root=Path("/opd"),
            run_root=Path("/run"),
            phase="primary",
            resume_step=20,
        )
