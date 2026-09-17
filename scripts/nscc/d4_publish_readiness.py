#!/usr/bin/env python3
"""Publish the Timeline D4 readiness report only after every G1 artifact passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_readiness(
    *,
    data_manifest: Path,
    baseline_summary: Path,
    log_report: Path,
    checkpoint_report: Path,
    export_acceptance: Path,
    smoke_root: Path,
) -> str:
    data = _load(data_manifest)
    artifacts = {
        "baseline": _load(baseline_summary),
        "smoke log": _load(log_report),
        "checkpoint and Student weight": _load(checkpoint_report),
        "independent exported-model inference": _load(export_acceptance),
    }
    failed = [label for label, payload in artifacts.items() if payload.get("passed") is not True]
    if failed:
        raise RuntimeError(f"G1 artifact not passed: {failed}")
    if data.get("train_rows") != 16920 or data.get("dev_rows") != 256:
        raise RuntimeError("frozen D4 data counts do not match the accepted manifest")
    log = artifacts["smoke log"]
    if log.get("primary_steps") != [1, 2, 3, 4, 5] or log.get("resume_steps") != [6, 7, 8]:
        raise RuntimeError("smoke step sequence is not 1-5 followed by 6-8")
    return (
        "# OPD D4 G1 Readiness\n\n"
        "- G1 status: PASS\n"
        "- Frozen data: train 16,920; dev 256; seed 2026\n"
        "- Baseline: complete Student/Teacher dev_256, MATH-500 and AIME24 raw outputs\n"
        "- Training: primary steps: 1–5\n"
        "- Resume: resume steps: 6–8 from the full step-5 state\n"
        "- Correctness: response-only loss mask, finite loss/gradient, Teacher frozen, updated Student rollout sync\n"
        "- Checkpoint: four-rank model/optimizer/extra-state shards plus dataloader state\n"
        "- Export: merged step-8 Student changed from initialization and passed independent inference\n"
        f"- Smoke artifact root: {smoke_root.resolve()}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--log-report", type=Path, required=True)
    parser.add_argument("--checkpoint-report", type=Path, required=True)
    parser.add_argument("--export-acceptance", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_readiness(
        data_manifest=args.data_manifest,
        baseline_summary=args.baseline_summary,
        log_report=args.log_report,
        checkpoint_report=args.checkpoint_report,
        export_acceptance=args.export_acceptance,
        smoke_root=args.smoke_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
