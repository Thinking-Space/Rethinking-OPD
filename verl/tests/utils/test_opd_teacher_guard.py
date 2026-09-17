import importlib
import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_guard_module():
    package_root = PROJECT_ROOT / "verl"
    path = package_root / "verl" / "utils" / "opd_teacher_guard.py"
    assert path.exists(), f"missing OPD teacher guard: {path}"
    sys.path.insert(0, str(package_root))
    return importlib.import_module("verl.utils.opd_teacher_guard")


def test_guard_freezes_teacher_and_accepts_unchanged_parameters():
    guard_module = load_guard_module()
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
    model.train()

    guard = guard_module.freeze_teacher(model)

    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())
    assert guard.assert_unchanged(step=1)["passed"] is True


def test_guard_clears_preexisting_teacher_gradients():
    guard_module = load_guard_module()
    model = torch.nn.Linear(2, 2)
    model(torch.ones(1, 2)).sum().backward()
    assert model.weight.grad is not None

    guard_module.freeze_teacher(model)

    assert all(parameter.grad is None for parameter in model.parameters())


def test_guard_rejects_parameter_mutation():
    guard_module = load_guard_module()
    model = torch.nn.Linear(2, 2)
    guard = guard_module.freeze_teacher(model)
    with torch.no_grad():
        model.weight.add_(1)

    with pytest.raises(RuntimeError, match="teacher parameters changed"):
        guard.assert_unchanged(step=2)


def test_guard_rejects_training_mode_regression():
    guard_module = load_guard_module()
    model = torch.nn.Linear(2, 2)
    guard = guard_module.freeze_teacher(model)
    model.train()

    with pytest.raises(RuntimeError, match="teacher entered training mode"):
        guard.assert_unchanged(step=3)
