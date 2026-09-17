from __future__ import annotations

import functools
import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch


PROJECT_ROOT = Path(
    os.environ.get("OPD_PROJECT_ROOT", "/home/users/nus/xiongjie/scratch/OPD")
)


@functools.lru_cache(maxsize=1)
def load_subject():
    script_path = PROJECT_ROOT / "scripts/nscc/d2_single_gpu_inference.py"
    if not script_path.is_file():
        pytest.fail(f"D2 inference script is not implemented: {script_path}")

    module_name = "opd_d2_single_gpu_inference_under_test"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Cannot create an import spec for D2 inference script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_smoke_cases_are_unique_and_gradeable_by_project_scorer():
    subject = load_subject()

    cases = subject.get_smoke_cases()

    assert len(cases) == 16
    assert len({case["id"] for case in cases}) == 16
    assert all(case["problem"].strip() for case in cases)
    assert all(case["answer"].strip() for case in cases)
    assert all(
        subject.grade_response(
            rf"Reasoning omitted. Final answer: \boxed{{{case['answer']}}}.",
            case["answer"],
            PROJECT_ROOT,
        )
        for case in cases
    )


def test_downloaded_tokenizers_have_exact_opd_token_alignment():
    subject = load_subject()
    student_path = PROJECT_ROOT / "model/DeepSeek-R1-Distill-Qwen-1.5B"
    teacher_path = PROJECT_ROOT / "model/JustRL-DeepSeek-1.5B"

    report = subject.verify_tokenizer_compatibility(student_path, teacher_path)

    assert report["compatible"] is True
    assert report["student_vocab_size"] == 151665
    assert report["teacher_vocab_size"] == 151665
    assert report["vocab_mismatch_count"] == 0
    assert report["special_tokens_equal"] is True
    assert report["special_token_ids_equal"] is True
    assert report["chat_template_equal"] is True
    assert report["smoke_prompt_token_ids_equal"] is True


def test_teacher_freeze_guard_detects_in_place_parameter_mutation():
    subject = load_subject()
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))

    subject.freeze_for_inference(model)
    versions = subject.snapshot_parameter_versions(model)

    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    assert all(parameter.grad is None for parameter in model.parameters())
    subject.assert_parameter_versions_unchanged(model, versions)

    with torch.no_grad():
        model[0].weight.add_(1)

    with pytest.raises(RuntimeError, match="mutated"):
        subject.assert_parameter_versions_unchanged(model, versions)


def test_model_summary_reports_eos_truncation_boxing_and_score():
    subject = load_subject()
    records = [
        {
            "generated_tokens": 11,
            "eos_reached": True,
            "hit_max_new_tokens": False,
            "boxed_answer": "4",
            "correct": True,
            "elapsed_seconds": 0.25,
        },
        {
            "generated_tokens": 20,
            "eos_reached": False,
            "hit_max_new_tokens": True,
            "boxed_answer": None,
            "correct": False,
            "elapsed_seconds": 0.75,
        },
    ]

    summary = subject.build_model_summary(
        "student",
        records,
        load_seconds=3.5,
        peak_allocated_gib=4.0,
        peak_reserved_gib=5.0,
    )

    assert summary == {
        "role": "student",
        "num_prompts": 2,
        "nonempty_responses": 2,
        "eos_count": 1,
        "eos_rate": 0.5,
        "hit_max_new_tokens_count": 1,
        "hit_max_new_tokens_rate": 0.5,
        "boxed_answer_count": 1,
        "boxed_answer_rate": 0.5,
        "correct_count": 1,
        "accuracy": 0.5,
        "mean_generated_tokens": 15.5,
        "max_generated_tokens": 20,
        "generation_seconds": 1.0,
        "load_seconds": 3.5,
        "peak_allocated_gib": 4.0,
        "peak_reserved_gib": 5.0,
    }


def test_acceptance_fails_closed_when_a_model_never_reaches_eos():
    subject = load_subject()
    base_summary = {
        "num_prompts": 16,
        "nonempty_responses": 16,
        "eos_count": 1,
        "boxed_answer_count": 1,
    }
    tokenizer_report = {"compatible": True}
    verl_report = {"inside_project_verl": True}
    teacher_report = {
        "training": False,
        "all_requires_grad_false": True,
        "all_grads_none": True,
        "parameters_unchanged": True,
    }

    passing = subject.build_acceptance(
        {"student": dict(base_summary), "teacher": dict(base_summary)},
        tokenizer_report,
        verl_report,
        teacher_report,
    )
    assert passing["passed"] is True

    no_teacher_eos = dict(base_summary)
    no_teacher_eos["eos_count"] = 0
    failing = subject.build_acceptance(
        {"student": dict(base_summary), "teacher": no_teacher_eos},
        tokenizer_report,
        verl_report,
        teacher_report,
    )
    assert failing["passed"] is False
    assert failing["checks"]["teacher_eos_observed"] is False
