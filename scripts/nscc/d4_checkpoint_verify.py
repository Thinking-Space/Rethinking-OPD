#!/usr/bin/env python3
"""Verify D4 full-state checkpoints and prove that exported Student weights changed."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _rank_files(actor_dir: Path, prefix: str, world_size: int) -> list[Path]:
    return [actor_dir / f"{prefix}_world_size_{world_size}_rank_{rank}.pt" for rank in range(world_size)]


def verify_checkpoint(checkpoint_dir: Path, *, world_size: int = 4) -> dict:
    checkpoint_dir = checkpoint_dir.resolve()
    actor_dir = checkpoint_dir / "actor"
    if not actor_dir.is_dir():
        raise RuntimeError(f"missing actor checkpoint directory: {actor_dir}")
    groups = {
        "model": _rank_files(actor_dir, "model", world_size),
        "optimizer": _rank_files(actor_dir, "optim", world_size),
        "extra_state": _rank_files(actor_dir, "extra_state", world_size),
    }
    for label, files in groups.items():
        missing = [str(path) for path in files if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise RuntimeError(f"missing or empty {label} shards: {missing}")
    data_state = checkpoint_dir / "data.pt"
    if not data_state.is_file() or data_state.stat().st_size == 0:
        raise RuntimeError(f"missing or empty dataloader state: {data_state}")
    return {
        "passed": True,
        "checkpoint": str(checkpoint_dir),
        "world_size": world_size,
        "model_shards": len(groups["model"]),
        "optimizer_shards": len(groups["optimizer"]),
        "extra_state_shards": len(groups["extra_state"]),
        "dataloader_state": str(data_state),
    }


def _tensor_files(model_dir: Path) -> dict[str, Path]:
    from safetensors import safe_open

    result: dict[str, Path] = {}
    for path in sorted(model_dir.rglob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in result:
                    raise RuntimeError(f"duplicate tensor key {key!r} in {model_dir}")
                result[key] = path
    if not result:
        raise RuntimeError(f"no safetensors weights found in {model_dir}")
    return result


def compare_safetensor_models(initial_dir: Path, exported_dir: Path) -> dict:
    from safetensors import safe_open

    initial_dir = initial_dir.resolve()
    exported_dir = exported_dir.resolve()
    initial_files = _tensor_files(initial_dir)
    exported_files = _tensor_files(exported_dir)
    if set(initial_files) != set(exported_files):
        only_initial = sorted(set(initial_files) - set(exported_files))[:10]
        only_exported = sorted(set(exported_files) - set(initial_files))[:10]
        raise RuntimeError(f"tensor key mismatch: only_initial={only_initial}, only_exported={only_exported}")

    handles = {}
    changed = 0
    max_abs_diff = 0.0
    changed_examples = []
    try:
        for key in sorted(initial_files):
            initial_path = initial_files[key]
            exported_path = exported_files[key]
            initial_handle = handles.setdefault(
                ("initial", initial_path),
                safe_open(initial_path, framework="pt", device="cpu"),
            )
            exported_handle = handles.setdefault(
                ("exported", exported_path),
                safe_open(exported_path, framework="pt", device="cpu"),
            )
            before = initial_handle.get_tensor(key)
            after = exported_handle.get_tensor(key)
            if before.shape != after.shape:
                raise RuntimeError(f"shape changed for {key}: {before.shape} != {after.shape}")
            if not before.is_floating_point() or not after.is_floating_point():
                different = not before.equal(after)
                tensor_max = 1.0 if different else 0.0
            else:
                difference = (before.float() - after.float()).abs()
                if not bool(difference.isfinite().all()):
                    raise RuntimeError(f"non-finite exported difference for {key}")
                tensor_max = float(difference.max().item()) if difference.numel() else 0.0
                different = tensor_max > 0.0
            if different:
                changed += 1
                max_abs_diff = max(max_abs_diff, tensor_max)
                if len(changed_examples) < 10:
                    changed_examples.append({"key": key, "max_abs_diff": tensor_max})
    finally:
        handles.clear()

    if changed == 0 or not math.isfinite(max_abs_diff):
        raise RuntimeError("exported Student has no finite weight change")
    return {
        "passed": True,
        "initial_model": str(initial_dir),
        "exported_model": str(exported_dir),
        "tensor_count": len(initial_files),
        "changed_tensor_count": changed,
        "max_abs_diff": max_abs_diff,
        "changed_examples": changed_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--exported-model", type=Path)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.initial_model is None) != (args.exported_model is None):
        parser.error("--initial-model and --exported-model must be supplied together")
    report = {
        "passed": True,
        "checkpoints": [verify_checkpoint(path, world_size=args.world_size) for path in args.checkpoint],
    }
    if args.initial_model is not None:
        report["student_weight_change"] = compare_safetensor_models(args.initial_model, args.exported_model)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
