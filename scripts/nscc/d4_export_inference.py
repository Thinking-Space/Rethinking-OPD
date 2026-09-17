#!/usr/bin/env python3
"""Run an independent one-GPU inference check on the exported D4 Student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_plan(args: argparse.Namespace) -> dict:
    return {
        "model": str(args.model),
        "dev_file": str(args.dev_file),
        "output_dir": str(args.output_dir),
        "example_count": 4,
        "temperature": 0.0,
        "max_tokens": 512,
        "independent_process": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args)
    if args.print_plan:
        print(json.dumps(plan, indent=2))
        return
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    frame = pd.read_parquet(args.dev_file)
    if len(frame) < plan["example_count"]:
        raise RuntimeError("dev file has fewer than four examples")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    conversations = frame.iloc[: plan["example_count"]]["prompt"].tolist()
    prompts = [
        tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        for conversation in conversations
    ]
    model = LLM(
        model=str(args.model),
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=1024,
        gpu_memory_utilization=0.8,
        seed=2026,
    )
    outputs = model.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=plan["max_tokens"]),
    )
    records = []
    for index, output in enumerate(outputs):
        candidate = output.outputs[0]
        records.append(
            {
                "example_id": index,
                "prompt": prompts[index],
                "response": candidate.text,
                "response_tokens": len(candidate.token_ids),
                "finish_reason": candidate.finish_reason,
            }
        )
    if len(records) != plan["example_count"] or any(not record["response"].strip() for record in records):
        raise RuntimeError("exported model independent inference produced incomplete or empty responses")
    raw_path = args.output_dir / "raw.jsonl"
    raw_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    acceptance = {
        "passed": True,
        "independent_process": True,
        "record_count": len(records),
        "nonempty_response_count": sum(bool(record["response"].strip()) for record in records),
        "plan": plan,
    }
    (args.output_dir / "acceptance.json").write_text(
        json.dumps(acceptance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(acceptance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
