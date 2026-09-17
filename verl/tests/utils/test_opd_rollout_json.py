import json

import numpy as np
import pytest
import torch

from verl.utils.opd_d4_diagnostics import json_default


def test_json_default_serializes_torch_and_numpy_values():
    payload = {
        "tensor": torch.tensor([1.0, 2.0]),
        "array": np.array([3, 4]),
        "scalar": np.float32(5.5),
    }

    restored = json.loads(json.dumps(payload, default=json_default))

    assert restored == {"tensor": [1.0, 2.0], "array": [3, 4], "scalar": 5.5}


def test_json_default_rejects_unknown_objects():
    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps({"value": object()}, default=json_default)
