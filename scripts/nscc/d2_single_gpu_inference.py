#!/usr/bin/env python3
"""D2–3 single-GPU inference and compatibility checks for OPD."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROJECT_ROOT = Path("/home/users/nus/xiongjie/scratch/OPD")
STUDENT_RELATIVE_PATH = Path("model/DeepSeek-R1-Distill-Qwen-1.5B")
TEACHER_RELATIVE_PATH = Path("model/JustRL-DeepSeek-1.5B")
EXPECTED_REVISIONS = {
    "student": "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
    "teacher": "0637e4096c789c67f9eecbe8355e0bdeddede1c2",
}
GIB = 1024**3


def get_smoke_cases() -> list[dict[str, str]]:
    """Return fixed, non-benchmark math prompts for environment validation."""
    return [
        {"id": "d2_01", "problem": "Compute 37 + 58.", "answer": "95"},
        {"id": "d2_02", "problem": "Compute 144 divided by 12.", "answer": "12"},
        {"id": "d2_03", "problem": "Solve for x: 3x + 7 = 31.", "answer": "8"},
        {
            "id": "d2_04",
            "problem": "Compute 3/4 + 5/8 and simplify the fraction.",
            "answer": r"\frac{11}{8}",
        },
        {"id": "d2_05", "problem": "What is 15 percent of 240?", "answer": "36"},
        {
            "id": "d2_06",
            "problem": "A rectangle has side lengths 13 and 7. Find its area.",
            "answer": "91",
        },
        {"id": "d2_07", "problem": "Find the positive square root of 196.", "answer": "14"},
        {"id": "d2_08", "problem": "Compute 2 raised to the 10th power.", "answer": "1024"},
        {
            "id": "d2_09",
            "problem": "Find the arithmetic mean of 4, 8, 12, and 16.",
            "answer": "10",
        },
        {"id": "d2_10", "problem": "Compute 5 factorial.", "answer": "120"},
        {
            "id": "d2_11",
            "problem": "If x is positive and x squared equals 81, find x.",
            "answer": "9",
        },
        {
            "id": "d2_12",
            "problem": "A square has side length 6.25. Find its perimeter.",
            "answer": "25",
        },
        {"id": "d2_13", "problem": "Solve for x: 2x - 5 = 9.", "answer": "7"},
        {
            "id": "d2_14",
            "problem": "Find the least common multiple of 6 and 14.",
            "answer": "42",
        },
        {
            "id": "d2_15",
            "problem": "Find the median of 1, 2, 3, 8, and 10.",
            "answer": "3",
        },
        {
            "id": "d2_16",
            "problem": "A cube has edge length 4. Find its volume.",
            "answer": "64",
        },
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4)
def _load_project_grader(project_root_text: str):
    grader_path = Path(project_root_text) / "scripts/val/eval/utils.py"
    spec = importlib.util.spec_from_file_location("opd_justrl_eval_utils", grader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load project grader from {grader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grade_response(response: str, ground_truth: str, project_root: Path) -> bool:
    grader = _load_project_grader(str(project_root.resolve()))
    return bool(grader.grade_answer_verl(response, ground_truth))


def extract_boxed_answer(response: str, project_root: Path) -> str | None:
    grader = _load_project_grader(str(project_root.resolve()))
    return grader.extract_answer(response)


def _special_token_ids(tokenizer) -> dict[str, Any]:
    names = ("bos", "eos", "pad", "unk", "sep", "cls", "mask")
    result = {f"{name}_token_id": getattr(tokenizer, f"{name}_token_id", None) for name in names}
    result["additional_special_tokens_ids"] = list(
        getattr(tokenizer, "additional_special_tokens_ids", [])
    )
    return result


def _smoke_prompt_token_ids(tokenizer) -> list[list[int]]:
    prompts = []
    for case in get_smoke_cases():
        content = (
            f"{case['problem']} Please reason step by step, and put your final "
            r"answer within \boxed{}."
        )
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    return prompts


def verify_tokenizer_compatibility(student_path: Path, teacher_path: Path) -> dict[str, Any]:
    if student_path.resolve() == teacher_path.resolve():
        raise RuntimeError("Student and teacher paths must be distinct")

    student = AutoTokenizer.from_pretrained(student_path, local_files_only=True, use_fast=True)
    teacher = AutoTokenizer.from_pretrained(teacher_path, local_files_only=True, use_fast=True)
    student_vocab = student.get_vocab()
    teacher_vocab = teacher.get_vocab()
    vocab_keys = set(student_vocab) | set(teacher_vocab)
    mismatch_count = sum(student_vocab.get(token) != teacher_vocab.get(token) for token in vocab_keys)
    student_special_ids = _special_token_ids(student)
    teacher_special_ids = _special_token_ids(teacher)
    student_prompt_ids = _smoke_prompt_token_ids(student)
    teacher_prompt_ids = _smoke_prompt_token_ids(teacher)
    student_tokenizer_sha = _sha256(student_path / "tokenizer.json")
    teacher_tokenizer_sha = _sha256(teacher_path / "tokenizer.json")

    report = {
        "student_vocab_size": len(student),
        "teacher_vocab_size": len(teacher),
        "vocab_mismatch_count": mismatch_count,
        "special_tokens_equal": student.special_tokens_map == teacher.special_tokens_map,
        "special_token_ids_equal": student_special_ids == teacher_special_ids,
        "chat_template_equal": student.chat_template == teacher.chat_template,
        "smoke_prompt_token_ids_equal": student_prompt_ids == teacher_prompt_ids,
        "student_tokenizer_json_sha256": student_tokenizer_sha,
        "teacher_tokenizer_json_sha256": teacher_tokenizer_sha,
    }
    report["compatible"] = all(
        [
            report["student_vocab_size"] == report["teacher_vocab_size"],
            mismatch_count == 0,
            report["special_tokens_equal"],
            report["special_token_ids_equal"],
            report["chat_template_equal"],
            report["smoke_prompt_token_ids_equal"],
            student_tokenizer_sha == teacher_tokenizer_sha,
        ]
    )
    return report


def freeze_for_inference(model: torch.nn.Module) -> None:
    model.requires_grad_(False)
    model.eval()


def snapshot_parameter_versions(model: torch.nn.Module) -> dict[str, int]:
    return {name: parameter._version for name, parameter in model.named_parameters()}


def assert_parameter_versions_unchanged(
    model: torch.nn.Module, before: dict[str, int]
) -> None:
    after = snapshot_parameter_versions(model)
    mutated = [name for name, version in before.items() if after.get(name) != version]
    if mutated:
        preview = ", ".join(mutated[:5])
        raise RuntimeError(f"Teacher parameters mutated during inference: {preview}")


def build_model_summary(
    role: str,
    records: list[dict[str, Any]],
    *,
    load_seconds: float,
    peak_allocated_gib: float,
    peak_reserved_gib: float,
) -> dict[str, Any]:
    count = len(records)
    nonempty = sum(
        bool(record.get("response", "")) or int(record.get("generated_tokens", 0)) > 0
        for record in records
    )
    eos_count = sum(bool(record["eos_reached"]) for record in records)
    limit_count = sum(bool(record["hit_max_new_tokens"]) for record in records)
    boxed_count = sum(record.get("boxed_answer") is not None for record in records)
    correct_count = sum(bool(record["correct"]) for record in records)
    token_counts = [int(record["generated_tokens"]) for record in records]
    generation_seconds = sum(float(record["elapsed_seconds"]) for record in records)

    def rate(value: int) -> float:
        return value / count if count else 0.0

    return {
        "role": role,
        "num_prompts": count,
        "nonempty_responses": nonempty,
        "eos_count": eos_count,
        "eos_rate": rate(eos_count),
        "hit_max_new_tokens_count": limit_count,
        "hit_max_new_tokens_rate": rate(limit_count),
        "boxed_answer_count": boxed_count,
        "boxed_answer_rate": rate(boxed_count),
        "correct_count": correct_count,
        "accuracy": rate(correct_count),
        "mean_generated_tokens": statistics.mean(token_counts) if token_counts else 0.0,
        "max_generated_tokens": max(token_counts, default=0),
        "generation_seconds": generation_seconds,
        "load_seconds": load_seconds,
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
    }


def build_acceptance(
    summaries: dict[str, dict[str, Any]],
    tokenizer_report: dict[str, Any],
    verl_report: dict[str, Any],
    teacher_report: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "tokenizers_exactly_compatible": tokenizer_report.get("compatible") is True,
        "project_verl_imported": verl_report.get("inside_project_verl") is True,
        "student_has_16_records": summaries["student"].get("num_prompts") == 16,
        "teacher_has_16_records": summaries["teacher"].get("num_prompts") == 16,
        "student_responses_nonempty": summaries["student"].get("nonempty_responses") == 16,
        "teacher_responses_nonempty": summaries["teacher"].get("nonempty_responses") == 16,
        "student_eos_observed": summaries["student"].get("eos_count", 0) > 0,
        "teacher_eos_observed": summaries["teacher"].get("eos_count", 0) > 0,
        "student_boxed_answer_observed": summaries["student"].get("boxed_answer_count", 0) > 0,
        "teacher_boxed_answer_observed": summaries["teacher"].get("boxed_answer_count", 0) > 0,
        "teacher_eval_mode": teacher_report.get("training") is False,
        "teacher_requires_grad_disabled": teacher_report.get("all_requires_grad_false") is True,
        "teacher_grads_absent": teacher_report.get("all_grads_none") is True,
        "teacher_parameters_unchanged": teacher_report.get("parameters_unchanged") is True,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _normalize_eos_ids(value: int | list[int] | None, fallback: int | None) -> list[int]:
    if value is None:
        value = fallback
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return list(value)


def _format_prompts(tokenizer, cases: list[dict[str, str]]) -> list[str]:
    prompts = []
    for case in cases:
        content = (
            f"{case['problem']} Please reason step by step, and put your final "
            r"answer within \boxed{}."
        )
        prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def run_model(
    *,
    role: str,
    model_path: Path,
    project_root: Path,
    cases: list[dict[str, str]],
    batch_size: int,
    max_new_tokens: int,
    attn_implementation: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    ).to("cuda:0")
    freeze_for_inference(model)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    versions = snapshot_parameter_versions(model) if role == "teacher" else None
    prompts = _format_prompts(tokenizer, cases)
    eos_ids = _normalize_eos_ids(model.generation_config.eos_token_id, tokenizer.eos_token_id)
    generation_eos_value: int | list[int] | None = eos_ids or None
    records: list[dict[str, Any]] = []

    for start in range(0, len(cases), batch_size):
        batch_cases = cases[start : start + batch_size]
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda:0")
        batch_started = time.perf_counter()
        with torch.inference_mode():
            sequences = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=generation_eos_value,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        torch.cuda.synchronize()
        elapsed_per_record = (time.perf_counter() - batch_started) / len(batch_cases)
        prompt_width = inputs["input_ids"].shape[1]

        for case, sequence in zip(batch_cases, sequences):
            generated = sequence[prompt_width:].tolist()
            eos_position = next(
                (index for index, token_id in enumerate(generated) if token_id in eos_ids),
                None,
            )
            eos_reached = eos_position is not None
            if eos_position is not None:
                generated = generated[: eos_position + 1]
            response = tokenizer.decode(generated, skip_special_tokens=True).strip()
            boxed_answer = extract_boxed_answer(response, project_root)
            records.append(
                {
                    "role": role,
                    "case_id": case["id"],
                    "problem": case["problem"],
                    "ground_truth": case["answer"],
                    "formatted_prompt": batch_prompts[len(records) % len(batch_cases)],
                    "response": response,
                    "boxed_answer": boxed_answer,
                    "correct": grade_response(response, case["answer"], project_root),
                    "generated_tokens": len(generated),
                    "eos_reached": eos_reached,
                    "hit_max_new_tokens": not eos_reached and len(generated) >= max_new_tokens,
                    "elapsed_seconds": elapsed_per_record,
                }
            )
        del inputs, sequences

    if role == "teacher":
        assert versions is not None
        assert_parameter_versions_unchanged(model, versions)

    peak_allocated_gib = torch.cuda.max_memory_allocated() / GIB
    peak_reserved_gib = torch.cuda.max_memory_reserved() / GIB
    summary = build_model_summary(
        role,
        records,
        load_seconds=load_seconds,
        peak_allocated_gib=peak_allocated_gib,
        peak_reserved_gib=peak_reserved_gib,
    )
    runtime_report = {
        "training": model.training,
        "all_requires_grad_false": all(not p.requires_grad for p in model.parameters()),
        "all_grads_none": all(p.grad is None for p in model.parameters()),
        "parameters_unchanged": True if role == "teacher" else None,
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "dtype": str(next(model.parameters()).dtype),
        "eos_token_ids": eos_ids,
        "pad_token_id": tokenizer.pad_token_id,
    }

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return records, summary, runtime_report


def _verify_manifest(project_root: Path) -> dict[str, Any]:
    manifest_path = project_root / "model/OPD_MODEL_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_role = {entry["role"]: entry for entry in manifest["models"]}
    for role, expected_revision in EXPECTED_REVISIONS.items():
        if by_role[role]["revision"] != expected_revision:
            raise RuntimeError(
                f"Unexpected {role} revision: {by_role[role]['revision']} != {expected_revision}"
            )
    return {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path),
        "student_revision": by_role["student"]["revision"],
        "teacher_revision": by_role["teacher"]["revision"],
    }


def _verify_verl_import(project_root: Path) -> dict[str, Any]:
    import verl

    actual = Path(verl.__file__).resolve()
    expected_root = (project_root / "verl").resolve()
    try:
        actual.relative_to(expected_root)
        inside = True
    except ValueError:
        inside = False
    return {
        "path": str(actual),
        "expected_root": str(expected_root),
        "inside_project_verl": inside,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("batch-size and max-new-tokens must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("D2 inference requires exactly one visible CUDA GPU")

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    student_path = (project_root / STUDENT_RELATIVE_PATH).resolve()
    teacher_path = (project_root / TEACHER_RELATIVE_PATH).resolve()
    manifest_report = _verify_manifest(project_root)
    verl_report = _verify_verl_import(project_root)
    tokenizer_report = verify_tokenizer_compatibility(student_path, teacher_path)
    cases = get_smoke_cases()
    _write_json(output_dir / "tokenizer_compatibility.json", tokenizer_report)

    summaries: dict[str, dict[str, Any]] = {}
    runtime_reports: dict[str, dict[str, Any]] = {}
    for role, path in (("student", student_path), ("teacher", teacher_path)):
        records, summary, runtime = run_model(
            role=role,
            model_path=path,
            project_root=project_root,
            cases=cases,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            attn_implementation=args.attn_implementation,
        )
        summaries[role] = summary
        runtime_reports[role] = runtime
        _write_jsonl(output_dir / f"{role}_results.jsonl", records)
        _write_json(output_dir / f"{role}_summary.json", summary)

    teacher_report = runtime_reports["teacher"]
    acceptance = build_acceptance(summaries, tokenizer_report, verl_report, teacher_report)
    final_summary = {
        "schema_version": 1,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "seed": 2026,
        "project_root": str(project_root),
        "student_path": str(student_path),
        "teacher_path": str(teacher_path),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "manifest": manifest_report,
        "verl": verl_report,
        "tokenizer": tokenizer_report,
        "models": summaries,
        "runtime": runtime_reports,
        "acceptance": acceptance,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    _write_json(output_dir / "summary.json", final_summary)
    _write_json(output_dir / "acceptance.json", acceptance)
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    if not acceptance["passed"]:
        raise RuntimeError("D2–3 acceptance checks failed; inspect acceptance.json")
    print("D2_SINGLE_GPU_ACCEPTANCE_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"D2_SINGLE_GPU_FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
