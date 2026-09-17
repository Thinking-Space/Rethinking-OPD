#!/usr/bin/env python3
"""Run the D4 Student/Teacher baseline with one model per A100."""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

from d4_pipeline import EVAL_PROMPT_TEMPLATE, extract_problem, summarize_baseline_records, timeline_evaluation_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_UTILS = PROJECT_ROOT / "scripts" / "val" / "eval"
sys.path.insert(0, str(EVAL_UTILS))
from utils import grade_answer_verl  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gpu_ids(value: str) -> list[str]:
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise ValueError("D4 baseline requires exactly two distinct GPU IDs")
    return gpu_ids


def build_plan(args) -> dict:
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    return {
        "seed": 2026,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 8192,
        "enable_thinking": False,
        "rule_based_grader": "scripts/val/eval/utils.py::grade_answer_verl",
        "worker_assignments": [
            {"label": "student", "model": args.student_model, "gpu_id": gpu_ids[0]},
            {"label": "teacher", "model": args.teacher_model, "gpu_id": gpu_ids[1]},
        ],
        "protocol": timeline_evaluation_protocol(),
        "data": {
            "dev_256": str(args.dev_file.resolve()),
            "MATH-500": str(args.math500_file.resolve()),
            "AIME24": str(args.aime24_file.resolve()),
        },
    }


def load_samples(path: str) -> list[dict]:
    import pandas as pd

    frame = pd.read_parquet(path)
    samples = []
    for example_id, row in frame.iterrows():
        raw_prompt = str(row["prompt"][0]["content"]).strip()
        samples.append(
            {
                "example_id": int(example_id),
                "problem": extract_problem(raw_prompt),
                "answer": str(row["reward_model"]["ground_truth"]).strip(),
            }
        )
    return samples


def run_model_worker(payload: dict) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = payload["gpu_id"]
    import torch
    from vllm import LLM, SamplingParams

    label = payload["label"]
    model_path = payload["model"]
    output_dir = Path(payload["output_dir"]) / label
    output_dir.mkdir(parents=True, exist_ok=False)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        dtype="bfloat16",
        seed=2026,
    )
    tokenizer = llm.get_tokenizer()
    stop_token_ids = []
    for token in ["<|im_end|>", "<|endoftext|>"]:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded:
            stop_token_ids.append(encoded[0])

    task_results = {}
    try:
        for task_name, task in payload["tasks"].items():
            samples = load_samples(task["path"])
            if len(samples) != task["expected_problems"]:
                raise ValueError(
                    f"{task_name}: expected {task['expected_problems']} problems, found {len(samples)}"
                )
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": EVAL_PROMPT_TEMPLATE.format(problem=sample["problem"])}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for sample in samples
            ]
            sampling = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=task["max_tokens"],
                stop_token_ids=stop_token_ids or None,
            )
            records = []
            started = time.monotonic()
            for sample_id in range(task["n"]):
                outputs = llm.generate(prompts, sampling, use_tqdm=True)
                for sample, output in zip(samples, outputs):
                    candidate = output.outputs[0]
                    response = candidate.text
                    records.append(
                        {
                            "model": label,
                            "model_path": model_path,
                            "task": task_name,
                            "example_id": sample["example_id"],
                            "sample_id": sample_id,
                            "problem": sample["problem"],
                            "answer": sample["answer"],
                            "response": response,
                            "response_tokens": len(candidate.token_ids),
                            "finish_reason": candidate.finish_reason,
                            "correct": bool(grade_answer_verl(response, sample["answer"])),
                        }
                    )
            out_path = output_dir / f"{task_name}.jsonl"
            with out_path.open("x", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            summary = summarize_baseline_records(
                records,
                expected_problems=task["expected_problems"],
                expected_n=task["n"],
            )
            summary["elapsed_seconds"] = time.monotonic() - started
            summary["output_file"] = str(out_path)
            summary["output_sha256"] = sha256_file(out_path)
            task_results[task_name] = summary
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()
    return {"label": label, "model_path": model_path, "tasks": task_results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--gpu-ids", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"))
    parser.add_argument(
        "--dev-file", type=Path, default=PROJECT_ROOT / "datasets" / "nscc_d4" / "dev_256_seed2026.parquet"
    )
    parser.add_argument(
        "--math500-file", type=Path, default=PROJECT_ROOT / "scripts" / "val" / "data" / "MATH-500" / "test.parquet"
    )
    parser.add_argument(
        "--aime24-file", type=Path, default=PROJECT_ROOT / "scripts" / "val" / "data" / "AIME24" / "test.parquet"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args)
    if args.print_plan:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --print-plan is used")

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite baseline directory: {output_dir}")
    output_dir.mkdir(parents=True)
    protocol_tasks = plan["protocol"]["tasks"]
    tasks = {
        name: {**task, "path": plan["data"][name]}
        for name, task in protocol_tasks.items()
    }
    payloads = [
        {**assignment, "output_dir": str(output_dir), "tasks": tasks}
        for assignment in plan["worker_assignments"]
    ]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=context) as executor:
        results = list(executor.map(run_model_worker, payloads))

    summary = {
        "passed": True,
        "plan": plan,
        "models": {result["label"]: result for result in results},
        "input_sha256": {name: sha256_file(Path(path)) for name, path in plan["data"].items()},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "acceptance.json").write_text(
        json.dumps({"passed": True, "checks": {"complete_sampling": True, "rule_grading": True}}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
