#!/usr/bin/env python3
"""Freeze the Timeline-defined DAPO train/dev split on a PBS CPU node."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from d4_pipeline import freeze_indices


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "datasets" / "dapo-math-17k.parquet"
DEFAULT_MATH500 = PROJECT_ROOT / "scripts" / "val" / "data" / "MATH-500" / "test.parquet"
DEFAULT_AIME24 = PROJECT_ROOT / "scripts" / "val" / "data" / "AIME24" / "test.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "datasets" / "nscc_d4"
TOKENIZER_PATH = PROJECT_ROOT / "model" / "DeepSeek-R1-Distill-Qwen-1.5B"
SEED = 2026
DEV_SIZE = 256
MAX_PROMPT_LENGTH = 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_content(row) -> str:
    prompt = row["prompt"]
    if len(prompt) != 1 or str(prompt[0]["role"]) != "user":
        raise ValueError("expected each dataset row to contain exactly one user prompt")
    return str(prompt[0]["content"])


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def build_plan(source: Path, math500: Path, aime24: Path, output_dir: Path) -> dict:
    return {
        "source": str(source.resolve()),
        "test_overlap_sources": [str(math500.resolve()), str(aime24.resolve())],
        "output_dir": str(output_dir.resolve()),
        "tokenizer": str(TOKENIZER_PATH.resolve()),
        "seed": SEED,
        "dev_size": DEV_SIZE,
        "max_prompt_length": MAX_PROMPT_LENGTH,
        "pipeline_order": ["filter_overlong", "exclude_test_overlap", "deduplicate_keep_first", "sample_dev"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--math500", type=Path, default=DEFAULT_MATH500)
    parser.add_argument("--aime24", type=Path, default=DEFAULT_AIME24)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--print-plan", action="store_true")
    args = parser.parse_args()

    plan = build_plan(args.source, args.math500, args.aime24, args.output_dir)
    if args.print_plan:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    import pandas as pd
    from transformers import AutoTokenizer

    source = args.source.resolve()
    math500 = args.math500.resolve()
    aime24 = args.aime24.resolve()
    output_dir = args.output_dir.resolve()
    for path in [source, math500, aime24, TOKENIZER_PATH / "tokenizer.json"]:
        if not path.exists():
            raise FileNotFoundError(path)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen data directory: {output_dir}")

    frame = pd.read_parquet(source)
    math_frame = pd.read_parquet(math500)
    aime_frame = pd.read_parquet(aime24)
    texts = [prompt_content(row) for _, row in frame.iterrows()]
    test_texts = [prompt_content(row) for _, row in math_frame.iterrows()]
    test_texts.extend(prompt_content(row) for _, row in aime_frame.iterrows())

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    prompt_lengths = [
        len(
            tokenizer.apply_chat_template(
                [dict(message) for message in prompt],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        for prompt in frame["prompt"]
    ]
    split = freeze_indices(
        texts,
        prompt_lengths,
        test_texts,
        dev_size=DEV_SIZE,
        seed=SEED,
        max_prompt_length=MAX_PROMPT_LENGTH,
    )

    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=".nscc_d4-", dir=parent))
    try:
        train_frame = frame.iloc[split["train_indices"]].copy().reset_index(drop=True)
        dev_frame = frame.iloc[split["dev_indices"]].copy().reset_index(drop=True)
        train_path = temp_dir / "train_seed2026.parquet"
        dev_path = temp_dir / "dev_256_seed2026.parquet"
        train_frame.to_parquet(train_path, index=False)
        dev_frame.to_parquet(dev_path, index=False)

        train_index_bytes = json.dumps(split["train_indices"], separators=(",", ":")).encode()
        manifest = {
            **plan,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_branch": git_value("branch", "--show-current"),
            "source_rows": split["source_rows"],
            "eligible_rows_before_dev_split": split["eligible_rows"],
            "train_rows": len(train_frame),
            "dev_rows": len(dev_frame),
            "excluded_overlong_count": len(split["excluded_overlong_indices"]),
            "excluded_overlong_indices": split["excluded_overlong_indices"],
            "excluded_test_overlap_count": len(split["excluded_test_overlap_indices"]),
            "excluded_test_overlap_indices": split["excluded_test_overlap_indices"],
            "excluded_duplicate_count": len(split["excluded_duplicate_indices"]),
            "excluded_duplicate_indices": split["excluded_duplicate_indices"],
            "dev_source_indices": split["dev_indices"],
            "train_source_indices_sha256": hashlib.sha256(train_index_bytes).hexdigest(),
            "prompt_tokens": {
                "min": min(prompt_lengths),
                "mean": sum(prompt_lengths) / len(prompt_lengths),
                "p95": sorted(prompt_lengths)[int(0.95 * (len(prompt_lengths) - 1))],
                "max": max(prompt_lengths),
            },
            "inputs": {
                str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in [source, math500, aime24, TOKENIZER_PATH / "tokenizer.json"]
            },
            "outputs": {
                train_path.name: {"bytes": train_path.stat().st_size, "sha256": sha256_file(train_path)},
                dev_path.name: {"bytes": dev_path.stat().st_size, "sha256": sha256_file(dev_path)},
            },
        }
        manifest_path = temp_dir / "data_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp_dir, output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
