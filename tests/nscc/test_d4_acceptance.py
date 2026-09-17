import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from safetensors.torch import save_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_script(name: str):
    path = PROJECT_ROOT / "scripts" / "nscc" / name
    assert path.exists(), f"missing D4 implementation: {path}"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_checkpoint(path: Path, world_size: int = 4) -> None:
    actor = path / "actor"
    actor.mkdir(parents=True)
    torch.save({"step": 5}, path / "data.pt")
    for rank in range(world_size):
        torch.save({"weight": torch.tensor([rank])}, actor / f"model_world_size_{world_size}_rank_{rank}.pt")
        torch.save({"state": {}}, actor / f"optim_world_size_{world_size}_rank_{rank}.pt")
        torch.save({"global_step": 5}, actor / f"extra_state_world_size_{world_size}_rank_{rank}.pt")


def metric_line(step: int, *, include_topk: bool = True) -> str:
    values = {
        "training/global_step": step,
        "actor/pg_loss": 0.1,
        "actor/grad_norm": 1.5,
        "actor/entropy": 0.5,
        "teacher/entropy": 0.7,
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": 0.3,
        "training/rollout_probs_diff_mean": 0.03,
        "training/rollout_probs_diff_std": 0.04,
        "training/rollout_actor_probs_pearson_corr": 0.95,
        "training/rollout_log_probs_diff_max": 0.4,
        "training/rollout_log_probs_diff_mean": 0.05,
        "training/rollout_log_probs_diff_std": 0.06,
    }
    if include_topk:
        values.update(
            {
                "critic/advantages/mean_overall": 0.2,
                "val-topk/overlap_ratio": 0.75,
                "val-topk/adv_intersection": -0.02,
            }
        )
    else:
        values.update(
            {
                "critic/advantages/mean": -0.2,
                "critic/score/mean": -1800.0,
                "critic/rewards/mean": -1800.0,
                "critic/true_reward/mean": 0.4,
            }
        )
    return "step:" + str(step) + " - " + " - ".join(f"{key}:{value}" for key, value in values.items())


def marker(name: str, payload: dict) -> str:
    return f"{name} {json.dumps(payload, sort_keys=True)}"


def test_metric_parser_recovers_ray_progress_interrupted_suffix():
    analyzer = load_script("d4_analyze_smoke.py")
    text = "\n".join(
        [
            "(TaskRunner pid=1) step:347 - actor/pg_loss:0.1 - "
            "training/global_step:347 - timing_s/gen:63.5 - timing_s/comp",
            "(TaskRunner pid=1) \rTraining Progress: 66%| 347/528 [99.2s/it]",
            "(TaskRunner pid=1) ute_log_prob:6.0 - timing_s/compute_rm_score:7.1 - "
            "timing_s/update_actor:23.4 - timing_s/step:100.8 - perf/throughput:2075.9",
            marker(
                "OPD_ROLLOUT_WEIGHT_SYNC",
                {"passed": True, "rank": 0, "sync_count": 348},
            ),
        ]
    )

    records = analyzer._metrics(text)

    assert len(records) == 1
    assert records[0]["training/global_step"] == 347
    assert records[0]["timing_s/compute_rm_score"] == pytest.approx(7.1)
    assert records[0]["timing_s/update_actor"] == pytest.approx(23.4)
    assert records[0]["timing_s/step"] == pytest.approx(100.8)


def make_log(path: Path, steps: range, *, resumed: bool) -> None:
    lines = []
    if resumed:
        lines.append("Setting global step to 5")
    lines.append(marker("OPD_TEACHER_FROZEN_INIT", {"passed": True}))
    for step in steps:
        lines.extend(
            [
                marker("OPD_RESPONSE_MASK_CHECK", {"passed": True, "step": step, "prompt_tokens_in_loss": 0, "padding_tokens_in_loss": 0}),
                marker("OPD_ROLLOUT_WEIGHT_SYNC", {"step": step}),
                marker("OPD_TEACHER_FROZEN_CHECK", {"passed": True}),
                metric_line(step),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pilot_metric_line(step: int, *, include_topk: bool = True) -> str:
    return metric_line(step, include_topk=include_topk) + " - " + " - ".join(
        [
            "critic/advantages/std:0.12",
            "critic/advantages/abs_mean:0.03",
            "response_length/mean:3200",
            "response_length/max:4096",
            "response_length/clip_ratio:0.25",
            "response/aborted_ratio:0",
            f"timing_s/step:{100 + step}",
            f"timing_s/gen:{60 + step}",
            f"timing_s/compute_rm_score:{20 + step}",
            f"timing_s/update_actor:{10 + step}",
            "perf/max_memory_allocated_gb:38",
            "perf/max_memory_reserved_gb:39",
            "perf/cpu_memory_used_gb:120",
        ]
    )


def make_pilot_log(path: Path, steps: range, *, resumed: bool) -> None:
    lines = []
    if not resumed:
        lines.append("Size of train dataloader: 1057, Size of val dataloader: 1")
    if resumed:
        lines.append("Setting global step to 30")
    lines.append(marker("OPD_TEACHER_FROZEN_INIT", {"passed": True}))
    for step in steps:
        lines.extend(
            [
                marker(
                    "OPD_RESPONSE_MASK_CHECK",
                    {
                        "passed": True,
                        "step": step,
                        "batch_size": 64,
                        "prompt_tokens_in_loss": 0,
                        "padding_tokens_in_loss": 0,
                        "response_length_mean": 3200.0,
                        "response_length_p95": 3900.0,
                    },
                ),
                marker("OPD_ROLLOUT_WEIGHT_SYNC", {"step": step}),
                marker("OPD_TEACHER_FROZEN_CHECK", {"passed": True}),
                pilot_metric_line(step),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_resource_csv(path: Path) -> None:
    lines = [
        "timestamp,gpu_index,memory_used_mib,memory_total_mib,"
        "utilization_percent,temperature_c,host_rss_kib"
    ]
    for second in (0, 60):
        for gpu in range(4):
            lines.append(
                f"2026-09-15T00:01:{second:02d}+08:00,{gpu},{40000 + gpu},"
                f"81920,{70 + gpu},60,125829120"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_pilot_rollouts(path: Path) -> None:
    for phase, steps in (("primary", range(1, 31)), ("resume", range(31, 33))):
        phase_dir = path / phase
        phase_dir.mkdir(parents=True)
        for step in steps:
            records = [
                {
                    "input": f"problem {step}-{sample}",
                    "output": "reasoning\nAnswer: 42"
                    if sample % 2 == 0
                    else "reasoning only",
                }
                for sample in range(64)
            ]
            (phase_dir / f"{step}.jsonl").write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )


def make_7168_log(path: Path) -> None:
    lines = [
        "Size of train dataloader: 528, Size of val dataloader: 1",
        marker("OPD_TEACHER_FROZEN_INIT", {"passed": True}),
    ]
    for step in range(1, 9):
        long_metric = (
            pilot_metric_line(step)
            .replace("response_length/mean:3200", "response_length/mean:6100")
            .replace("response_length/max:4096", "response_length/max:7168")
            .replace("response_length/clip_ratio:0.25", "response_length/clip_ratio:0.30")
        )
        lines.extend(
            [
                marker(
                    "OPD_RESPONSE_MASK_CHECK",
                    {
                        "passed": True,
                        "step": step,
                        "batch_size": 128,
                        "prompt_tokens_in_loss": 0,
                        "padding_tokens_in_loss": 0,
                        "response_length_mean": 6100.0,
                        "response_length_p95": 7000.0,
                    },
                ),
                marker("OPD_ROLLOUT_WEIGHT_SYNC", {"step": step}),
                marker("OPD_TEACHER_FROZEN_CHECK", {"passed": True}),
                long_metric,
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_7168_rollouts(path: Path) -> None:
    path.mkdir(parents=True)
    for step in range(1, 9):
        records = [
            {
                "input": f"problem {step}-{sample}",
                "output": "reasoning\nAnswer: 42" if sample % 2 == 0 else "reasoning only",
            }
            for sample in range(128)
        ]
        (path / f"{step}.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )


def make_formal_log(
    path: Path,
    steps: range,
    *,
    resume_step: int | None,
    wandb_run_id: str,
    include_topk: bool = True,
) -> None:
    lines = []
    if resume_step is None:
        lines.append("Size of train dataloader: 528, Size of val dataloader: 1")
    else:
        lines.append(f"Setting global step to {resume_step}")
    lines.extend(
        [
            "wandb: Run `wandb offline` to turn off syncing.",
            f"wandb: View run at https://wandb.ai/example/opd/runs/{wandb_run_id}",
            marker("OPD_TEACHER_FROZEN_INIT", {"passed": True}),
        ]
    )
    for step in steps:
        segment_sync_count = step - (resume_step or 0)
        long_metric = (
            pilot_metric_line(step, include_topk=include_topk)
            .replace("response_length/mean:3200", "response_length/mean:6100")
            .replace("response_length/max:4096", "response_length/max:7168")
            .replace("response_length/clip_ratio:0.25", "response_length/clip_ratio:0.30")
        )
        lines.extend(
            [
                marker(
                    "OPD_RESPONSE_MASK_CHECK",
                    {
                        "passed": True,
                        "step": step,
                        "batch_size": 128,
                        "prompt_tokens_in_loss": 0,
                        "padding_tokens_in_loss": 0,
                        "response_length_mean": 6100.0,
                        "response_length_p95": 7000.0,
                    },
                ),
                marker(
                    "OPD_ROLLOUT_WEIGHT_SYNC",
                    {
                        "passed": True,
                        "rank": 0,
                        "sync_count": segment_sync_count,
                    },
                ),
                marker("OPD_TEACHER_FROZEN_CHECK", {"passed": True}),
                long_metric,
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_formal_rollouts(path: Path) -> None:
    path.mkdir(parents=True)
    for step in range(1, 529):
        records = [
            {
                "input": f"problem {step}-{sample // 4}"
                + ("\u2028continued" if step == 1 and sample // 4 == 0 else ""),
                "output": "reasoning\u2028Answer: 42"
                if step == 1 and sample == 0
                else "reasoning\nAnswer: 42"
                if sample % 2 == 0
                else "reasoning only",
            }
            for sample in range(128)
        ]
        (path / f"{step}.jsonl").write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
            + "\n",
            encoding="utf-8",
        )


def test_checkpoint_verifier_requires_full_four_rank_state_and_detects_change(tmp_path):
    verifier = load_script("d4_checkpoint_verify.py")
    checkpoint = tmp_path / "global_step_5"
    make_checkpoint(checkpoint)
    initial = tmp_path / "initial"
    exported = tmp_path / "exported"
    initial.mkdir()
    exported.mkdir()
    save_file({"model.weight": torch.tensor([1.0, 2.0])}, initial / "model.safetensors")
    save_file({"model.weight": torch.tensor([1.0, 2.5])}, exported / "model.safetensors")

    checkpoint_report = verifier.verify_checkpoint(checkpoint, world_size=4)
    weight_report = verifier.compare_safetensor_models(initial, exported)

    assert checkpoint_report["passed"] is True
    assert checkpoint_report["model_shards"] == 4
    assert checkpoint_report["optimizer_shards"] == 4
    assert checkpoint_report["extra_state_shards"] == 4
    assert weight_report["passed"] is True
    assert weight_report["changed_tensor_count"] == 1
    assert weight_report["max_abs_diff"] == pytest.approx(0.5)


def test_checkpoint_verifier_rejects_missing_optimizer_shard(tmp_path):
    verifier = load_script("d4_checkpoint_verify.py")
    checkpoint = tmp_path / "global_step_5"
    make_checkpoint(checkpoint)
    (checkpoint / "actor" / "optim_world_size_4_rank_3.pt").unlink()

    with pytest.raises(RuntimeError, match="optimizer shards"):
        verifier.verify_checkpoint(checkpoint, world_size=4)


def test_smoke_analyzer_enforces_timeline_g1_contract(tmp_path):
    analyzer = load_script("d4_analyze_smoke.py")
    primary = tmp_path / "primary.log"
    resume = tmp_path / "resume.log"
    make_log(primary, range(1, 6), resumed=False)
    make_log(resume, range(6, 9), resumed=True)

    report = analyzer.analyze_logs(primary, resume)

    assert report["passed"] is True
    assert report["primary_steps"] == [1, 2, 3, 4, 5]
    assert report["resume_steps"] == [6, 7, 8]
    assert report["resume_loaded_step"] == 5
    assert report["response_mask_checks"] == 8
    assert report["teacher_frozen_checks"] == 8
    assert report["rollout_weight_syncs"] == 8
    assert "thresholds" not in report
    assert report["acceptance_policy"]["upstream_probability_debug"] == (
        "required, response-masked, valid and finite; no upstream numeric threshold"
    )
    assert report["acceptance_policy"]["student_teacher_alignment"] == (
        "required and finite at every G1 step; no 5-10-step trend threshold"
    )
    assert len(report["diagnostics"]["upstream_probability_debug"]) == 8
    assert len(report["diagnostics"]["student_teacher_alignment"]) == 8
    assert report["diagnostics"]["student_teacher_alignment"][0][
        "student_minus_teacher_entropy"
    ] == pytest.approx(-0.2)
    assert report["diagnostics"]["raw_log_probability_supplemental"][0][
        "max"
    ] == pytest.approx(0.4)


def test_smoke_analyzer_accepts_debug_values_outside_removed_local_thresholds(tmp_path):
    analyzer = load_script("d4_analyze_smoke.py")
    primary = tmp_path / "primary.log"
    resume = tmp_path / "resume.log"
    make_log(primary, range(1, 6), resumed=False)
    make_log(resume, range(6, 9), resumed=True)

    report = analyzer.analyze_logs(primary, resume)

    assert report["passed"] is True


def test_smoke_analyzer_rejects_nonfinite_required_alignment_metric(tmp_path):
    analyzer = load_script("d4_analyze_smoke.py")
    primary = tmp_path / "primary.log"
    resume = tmp_path / "resume.log"
    make_log(primary, range(1, 6), resumed=False)
    make_log(resume, range(6, 9), resumed=True)
    text = resume.read_text(encoding="utf-8").replace(
        "teacher/entropy:0.7",
        "teacher/entropy:nan",
        1,
    )
    resume.write_text(text, encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-finite teacher/entropy"):
        analyzer.analyze_logs(primary, resume)


def test_pilot_analyzer_enforces_d5_4096_contract_without_invented_trend_thresholds(
    tmp_path,
):
    analyzer = load_script("d5_analyze_pilot.py")
    primary = tmp_path / "primary.log"
    resume = tmp_path / "resume.log"
    resources = tmp_path / "resources.csv"
    rollouts = tmp_path / "rollouts"
    make_pilot_log(primary, range(1, 31), resumed=False)
    make_pilot_log(resume, range(31, 33), resumed=True)
    make_resource_csv(resources)
    make_pilot_rollouts(rollouts)

    report = analyzer.analyze_pilot(
        primary,
        resume,
        resources,
        rollouts,
        max_response_length=4096,
        gpu_count=4,
        allocated_host_memory_gib=220,
        allocated_walltime_hours=8,
    )

    assert report["passed"] is True
    assert report["pilot_status"] == "PASS_4096"
    assert report["g2_status"] == "PENDING_7168_VALIDATION"
    assert report["primary_steps"] == list(range(1, 31))
    assert report["resume_steps"] == [31, 32]
    assert report["resume_loaded_step"] == 30
    assert report["performance"]["warmup_discarded_steps"] == [1, 2, 3, 31]
    assert report["performance"]["timings_seconds"]["overall"]["count"] == 28
    assert report["lengths"]["response_p95_tokens"]["max"] == pytest.approx(3900)
    assert report["lengths"]["hit_length_limit_ratio"]["mean"] == pytest.approx(
        0.25
    )
    assert report["lengths"]["effective_answer_presence_ratio"] == pytest.approx(
        0.5
    )
    assert report["resources"]["observed_gpu_indices"] == [0, 1, 2, 3]
    assert report["resources"]["max_gpu_memory_used_mib"] == 40003
    assert report["budget"]["reserved_gpu_hours"] == 32
    assert report["budget"]["reserved_su"] == 2048
    assert report["budget"]["actual_train_dataloader_steps"] == 1057
    assert report["alignment_trends"]["policy"] == (
        "report window trends only; no upstream single-point numeric threshold"
    )
    assert set(report["alignment_trends"]["metrics"]) == {
        "top_k_overlap_ratio",
        "shared_token_advantage",
        "student_minus_teacher_entropy",
    }

    readiness = analyzer.render_4096_readiness(report, Path("/runs/pilot/123"))
    assert "4096 Pilot status: PASS" in readiness
    assert "G2 status: PENDING 7168 VALIDATION" in readiness
    assert "/runs/pilot/123" in readiness


def test_pilot_analyzer_rejects_oom_even_when_all_metrics_are_finite(tmp_path):
    analyzer = load_script("d5_analyze_pilot.py")
    primary = tmp_path / "primary.log"
    resume = tmp_path / "resume.log"
    resources = tmp_path / "resources.csv"
    rollouts = tmp_path / "rollouts"
    make_pilot_log(primary, range(1, 31), resumed=False)
    make_pilot_log(resume, range(31, 33), resumed=True)
    make_resource_csv(resources)
    make_pilot_rollouts(rollouts)
    resume.write_text(
        resume.read_text(encoding="utf-8") + "CUDA out of memory\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="OOM/error signature"):
        analyzer.analyze_pilot(
            primary,
            resume,
            resources,
            rollouts,
            max_response_length=4096,
            gpu_count=4,
            allocated_host_memory_gib=220,
            allocated_walltime_hours=8,
        )


def test_pilot_pbs_reuses_upstream_entrypoint_and_stops_before_d7():
    script = (PROJECT_ROOT / "scripts" / "nscc" / "d5_pilot_4096.pbs").read_text(
        encoding="utf-8"
    )

    assert "#PBS -l select=1:ncpus=32:mem=220gb:ngpus=4" in script
    assert "#PBS -l walltime=08:00:00" in script
    assert "python -m verl.trainer.main_ppo" in script
    assert "d5_pilot_config.py" in script
    assert "global_step_30" in script
    assert "global_step_32" in script
    assert "d4_checkpoint_verify.py" in script
    assert "d5_analyze_pilot.py" in script
    assert "D5_4096_PILOT_OK" in script
    assert "legacy_model_merger.py" not in script
    assert "d4_export_inference.py" not in script
    assert "7168" not in script


def test_7168_analyzer_closes_g2_using_final_candidate_without_score_gate(tmp_path):
    analyzer = load_script("d6_analyze_7168.py")
    training_log = tmp_path / "training.log"
    resources = tmp_path / "resources.csv"
    rollouts = tmp_path / "rollouts"
    short_report = tmp_path / "pilot_4096.json"
    make_7168_log(training_log)
    make_resource_csv(resources)
    make_7168_rollouts(rollouts)
    short_report.write_text(
        json.dumps(
            {
                "passed": True,
                "pilot_status": "PASS_4096",
                "g2_status": "PENDING_7168_VALIDATION",
            }
        ),
        encoding="utf-8",
    )

    report = analyzer.analyze_7168(
        training_log,
        resources,
        rollouts,
        short_report,
        gpu_count=4,
        allocated_host_memory_gib=220,
        allocated_walltime_hours=3,
        short_pilot_actual_gpu_hours=2.4,
    )

    assert report["passed"] is True
    assert report["g2_status"] == "PASS"
    assert report["selected_config"] == {
        "max_prompt_length": 1024,
        "max_response_length": 7168,
        "train_batch_size": 32,
        "rollout_n": 4,
        "actor_token_budget_per_gpu": 8192,
    }
    assert report["steps"] == list(range(1, 9))
    assert report["performance"]["warmup_discarded_steps"] == [1, 2, 3]
    assert report["performance"]["timings_seconds"]["overall"]["count"] == 5
    assert report["lengths"]["response_p95_tokens"]["max"] == pytest.approx(7000)
    assert report["lengths"]["effective_answer_presence_ratio"] == pytest.approx(0.5)
    assert report["budget"]["actual_train_dataloader_steps"] == 528
    assert report["budget"]["combined_pilot_gpu_hours_upper_bound"] == pytest.approx(
        14.4
    )
    assert report["budget"]["recommended_training_steps"] == 528
    assert report["acceptance_policy"]["not_required"] == (
        "Pilot score improvement or monotonic loss"
    )

    readiness = analyzer.render_g2_readiness(report, Path("/runs/long/123"))
    assert "G2 status: PASS" in readiness
    assert "selected response length: 7168" in readiness
    assert "D7 has not been started" in readiness


def test_7168_pbs_is_a_separate_single_variable_validation_before_d7():
    script = (PROJECT_ROOT / "scripts" / "nscc" / "d6_validate_7168.pbs").read_text(
        encoding="utf-8"
    )

    assert "#PBS -l select=1:ncpus=32:mem=220gb:ngpus=4" in script
    assert "#PBS -l walltime=03:00:00" in script
    assert "python -m verl.trainer.main_ppo" in script
    assert "d6_7168_config.py" in script
    assert "d6_analyze_7168.py" in script
    assert "global_step_8" in script
    assert "G2_PASS" in script
    assert "resume_from_path" not in script
    assert "global_step_30" not in script
    assert "legacy_model_merger.py" not in script
    assert "d4_export_inference.py" not in script


def test_formal_analyzer_accepts_exact_epoch_resume_and_online_wandb(tmp_path):
    analyzer = load_script("d7_9_analyze_formal.py")
    primary = tmp_path / "primary.log"
    resume = tmp_path / "resume_300.log"
    primary_resources = tmp_path / "primary_resources.csv"
    resume_resources = tmp_path / "resume_resources.csv"
    rollouts = tmp_path / "rollouts"
    wandb_run_id = tmp_path / "wandb_run_id.txt"
    make_formal_log(
        primary,
        range(1, 301),
        resume_step=None,
        wandb_run_id="formal123",
    )
    make_formal_log(
        resume,
        range(301, 529),
        resume_step=300,
        wandb_run_id="formal123",
    )
    make_resource_csv(primary_resources)
    make_resource_csv(resume_resources)
    make_formal_rollouts(rollouts)
    wandb_run_id.write_text("formal123\n", encoding="utf-8")

    report = analyzer.analyze_formal(
        [primary, resume],
        [primary_resources, resume_resources],
        rollouts,
        wandb_run_id,
        gpu_count=4,
        allocated_host_memory_gib=220,
        allocated_walltime_hours_per_segment=24,
    )

    assert report["passed"] is True
    assert report["d7_9_status"] == "PASS_TRAINING_COMPLETE"
    assert report["g3_status"] == "NOT_STARTED"
    assert report["steps"] == list(range(1, 529))
    assert report["full_data_one_epoch"] is True
    assert report["resume_loaded_steps"] == [300]
    assert report["performance"]["warmup_discarded_steps"] == [1, 2, 3, 301]
    assert report["performance"]["timings_seconds"]["overall"]["count"] == 524
    assert report["lengths"]["rollout_records"] == 67584
    assert report["lengths"]["unique_prompt_records"] == 16896
    assert report["lengths"]["effective_answer_presence_ratio"] == pytest.approx(0.5)
    assert report["resources"]["segment_count"] == 2
    assert report["budget"]["reserved_gpu_hours_upper_bound"] == 192
    assert report["budget"]["reserved_su_upper_bound"] == 12288
    assert report["wandb"] == {
        "mode": "online",
        "run_id": "formal123",
        "run_url": "https://wandb.ai/example/opd/runs/formal123",
        "single_run_across_segments": True,
    }
    assert report["alignment_trends"]["policy"] == (
        "report 20-step window trends only; no upstream single-point numeric threshold"
    )

    readiness = analyzer.render_main_readiness(report, Path("/runs/main/123"))
    assert "D7–9 status: PASS_TRAINING_COMPLETE" in readiness
    assert "full-data one-epoch run: 528/528 steps" in readiness
    assert "W&B mode: online" in readiness
    assert "G3 status: NOT_STARTED" in readiness

    resume.write_text(
        resume.read_text(encoding="utf-8") + "wandb: W&B syncing is set to `offline`\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="offline W&B"):
        analyzer.analyze_formal(
            [primary, resume],
            [primary_resources, resume_resources],
            rollouts,
            wandb_run_id,
            gpu_count=4,
            allocated_host_memory_gib=220,
            allocated_walltime_hours_per_segment=24,
        )


def test_formal_analyzer_accepts_sampled_token_without_top_k_metrics(tmp_path):
    analyzer = load_script("d7_9_analyze_formal.py")
    training_log = tmp_path / "sampled_token.log"
    resources = tmp_path / "resources.csv"
    rollouts = tmp_path / "rollouts"
    wandb_run_id = tmp_path / "wandb_run_id.txt"
    make_formal_log(
        training_log,
        range(1, 529),
        resume_step=None,
        wandb_run_id="sampled123",
        include_topk=False,
    )
    make_resource_csv(resources)
    make_formal_rollouts(rollouts)
    wandb_run_id.write_text("sampled123\n", encoding="utf-8")

    report = analyzer.analyze_formal(
        [training_log],
        [resources],
        rollouts,
        wandb_run_id,
        gpu_count=4,
        allocated_host_memory_gib=220,
        allocated_walltime_hours_per_segment=24,
        distillation_mode="sampled_token",
    )

    assert report["passed"] is True
    assert report["distillation_mode"] == "sampled_token"
    assert report["teacher_signal"]["advantage_abs_mean"]["mean"] == pytest.approx(
        0.03
    )
    assert set(report["alignment_trends"]["metrics"]) == {
        "student_minus_teacher_entropy"
    }
    assert "top-k metrics are not applicable" in report["alignment_trends"][
        "policy"
    ]
    readiness = analyzer.render_main_readiness(
        report, Path("/runs/sampled-token/123")
    )
    assert "Sampled-token control status: PASS_TRAINING_COMPLETE" in readiness


def test_sampled_token_pbs_is_a_thin_formal_runner_variant():
    script = (
        PROJECT_ROOT / "scripts" / "nscc" / "d15_sampled_token_run.pbs"
    ).read_text(encoding="utf-8")

    assert "#PBS -l select=1:ncpus=32:mem=220gb:ngpus=4" in script
    assert "#PBS -l walltime=24:00:00" in script
    assert "OPD_FORMAL_VARIANT=sampled_token" in script
    assert "d7_9_formal_run.pbs" in script
    assert "python -m verl.trainer.main_ppo" not in script


def test_formal_pbs_uses_frozen_upstream_entrypoint_resume_and_online_wandb():
    script = (PROJECT_ROOT / "scripts" / "nscc" / "d7_9_formal_run.pbs").read_text(
        encoding="utf-8"
    )

    assert "#PBS -l select=1:ncpus=32:mem=220gb:ngpus=4" in script
    assert "#PBS -l walltime=24:00:00" in script
    assert "python -m verl.trainer.main_ppo" in script
    assert "d7_9_formal_config.py" in script
    assert "d7_9_analyze_formal.py" in script
    assert "global_step_528" in script
    assert 'export WANDB_MODE=online' in script
    assert 'export WANDB_RESUME=must' in script
    assert "wandb_run_id.txt" in script
    assert "OPD_MAIN_PHASE" in script
    assert "OPD_MAIN_RUN_ROOT" in script
    assert "OPD_MAIN_RESUME_STEP" in script
    assert "D7_9_FORMAL_TRAINING_OK" in script
    assert "legacy_model_merger.py" not in script
    assert "d4_export_inference.py" not in script


def test_export_inference_print_plan_is_fixed_and_independent(tmp_path):
    script = PROJECT_ROOT / "scripts" / "nscc" / "d4_export_inference.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--model",
            "/exported/model",
            "--dev-file",
            "/data/dev.parquet",
            "--output-dir",
            str(tmp_path / "out"),
            "--print-plan",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan == {
        "model": "/exported/model",
        "dev_file": "/data/dev.parquet",
        "output_dir": str(tmp_path / "out"),
        "example_count": 4,
        "temperature": 0.0,
        "max_tokens": 512,
        "independent_process": True,
    }


def test_closeout_prioritizes_conda_runtime_libraries_before_vllm():
    script = (PROJECT_ROOT / "scripts" / "nscc" / "d4_reanalyze_smoke.pbs").read_text(
        encoding="utf-8"
    )
    activation = 'source "$OPD_SETUP/activate_opd.sh"'
    runtime_fix = (
        'export LD_LIBRARY_PATH="$CONDA_PREFIX/lib'
        '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
    )
    inference = "python scripts/nscc/d4_export_inference.py"

    assert runtime_fix in script
    assert script.index(activation) < script.index(runtime_fix) < script.index(inference)


def test_readiness_is_published_only_from_all_passed_artifacts(tmp_path):
    publisher = load_script("d4_publish_readiness.py")
    paths = {}
    payloads = {
        "data": {"train_rows": 16920, "dev_rows": 256},
        "baseline": {"passed": True},
        "log": {"passed": True, "primary_steps": [1, 2, 3, 4, 5], "resume_steps": [6, 7, 8]},
        "checkpoint": {"passed": True, "student_weight_change": {"passed": True}},
        "export": {"passed": True, "record_count": 4},
    }
    for label, payload in payloads.items():
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[label] = path

    report = publisher.build_readiness(
        data_manifest=paths["data"],
        baseline_summary=paths["baseline"],
        log_report=paths["log"],
        checkpoint_report=paths["checkpoint"],
        export_acceptance=paths["export"],
        smoke_root=Path("/runs/smoke/123"),
    )

    assert "# OPD D4 G1 Readiness" in report
    assert "G1 status: PASS" in report
    assert "primary steps: 1–5" in report
    assert "resume steps: 6–8" in report
    assert "/runs/smoke/123" in report


def test_readiness_rejects_failed_input(tmp_path):
    publisher = load_script("d4_publish_readiness.py")
    paths = []
    for index, passed in enumerate([True, True, False, True, True]):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps({"passed": passed}), encoding="utf-8")
        paths.append(path)
    with pytest.raises(RuntimeError, match="not passed"):
        publisher.build_readiness(
            data_manifest=paths[0],
            baseline_summary=paths[1],
            log_report=paths[2],
            checkpoint_report=paths[3],
            export_acceptance=paths[4],
            smoke_root=Path("/run"),
        )
