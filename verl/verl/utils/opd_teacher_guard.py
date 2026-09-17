# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime guard proving that the OPD teacher stays frozen."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _sampled_checksum(module: torch.nn.Module) -> tuple[float, float]:
    total = 0.0
    squared = 0.0
    with torch.no_grad():
        for parameter in module.parameters():
            flat = parameter.detach().reshape(-1)
            if flat.numel() == 0:
                continue
            stride = max(flat.numel() // 16, 1)
            sample = flat[::stride][:16].to(device="cpu", dtype=torch.float64)
            total += sample.sum().item()
            squared += sample.square().sum().item()
    return total, squared


@dataclass
class TeacherFreezeGuard:
    module: torch.nn.Module
    parameter_ids: tuple[int, ...]
    parameter_versions: tuple[int, ...]
    sampled_checksum: tuple[float, float]

    def assert_unchanged(self, *, step: int | str) -> dict:
        parameters = tuple(self.module.parameters())
        if self.module.training:
            raise RuntimeError(f"teacher entered training mode at step {step}")
        if any(parameter.requires_grad for parameter in parameters):
            raise RuntimeError(f"teacher gradients were enabled at step {step}")
        if tuple(id(parameter) for parameter in parameters) != self.parameter_ids:
            raise RuntimeError(f"teacher parameter objects changed at step {step}")
        versions = tuple(int(parameter._version) for parameter in parameters)
        checksum = _sampled_checksum(self.module)
        if versions != self.parameter_versions or checksum != self.sampled_checksum:
            raise RuntimeError(f"teacher parameters changed at step {step}")
        return {
            "passed": True,
            "step": step,
            "training": self.module.training,
            "requires_grad_count": sum(parameter.requires_grad for parameter in parameters),
            "parameter_count": len(parameters),
            "sampled_checksum": list(checksum),
        }


def freeze_teacher(module: torch.nn.Module) -> TeacherFreezeGuard:
    module.eval()
    module.requires_grad_(False)
    parameters = tuple(module.parameters())
    for parameter in parameters:
        parameter.grad = None
    return TeacherFreezeGuard(
        module=module,
        parameter_ids=tuple(id(parameter) for parameter in parameters),
        parameter_versions=tuple(int(parameter._version) for parameter in parameters),
        sampled_checksum=_sampled_checksum(module),
    )
