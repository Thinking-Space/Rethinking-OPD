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

"""Runtime correctness checks used only when OPD_D4_DIAGNOSTICS=1."""

from __future__ import annotations

import numpy as np
import torch


def json_default(value):
    """Convert tensor-backed rollout metadata into standard JSON values."""
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def build_response_mask_report(
    *, attention_mask: torch.Tensor, responses: torch.Tensor, response_mask: torch.Tensor, step: int
) -> dict:
    if response_mask.shape != responses.shape:
        raise RuntimeError(
            f"response loss mask shape {tuple(response_mask.shape)} does not match responses {tuple(responses.shape)}"
        )
    if attention_mask.ndim != 2 or responses.ndim != 2:
        raise RuntimeError("D4 response-mask check expects two-dimensional tensors")
    input_width = int(attention_mask.shape[-1])
    response_width = int(responses.shape[-1])
    prompt_width = input_width - response_width
    if prompt_width < 0:
        raise RuntimeError("response width exceeds full input width")

    attention_response = attention_mask[:, -response_width:].bool()
    response_mask_bool = response_mask.bool()
    padding_tokens_in_loss = int((response_mask_bool & ~attention_response).sum().item())
    if padding_tokens_in_loss:
        raise RuntimeError(f"padding entered response loss at step {step}: {padding_tokens_in_loss} tokens")

    response_lengths = response_mask_bool.sum(dim=-1).float()

    return {
        "passed": True,
        "step": int(step),
        "batch_size": int(responses.shape[0]),
        "input_width": input_width,
        "prompt_width": prompt_width,
        "response_width": response_width,
        "prompt_tokens_in_loss": 0,
        "padding_tokens_in_loss": padding_tokens_in_loss,
        "response_tokens_in_loss": int(response_mask_bool.sum().item()),
        "response_length_mean": float(response_lengths.mean().item()),
        "response_length_p95": float(torch.quantile(response_lengths, 0.95).item()),
    }
