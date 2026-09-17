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

import numpy as np
import wandb

from verl.utils.tracking import create_wandb_image


def test_create_wandb_image_returns_native_wandb_image():
    image_data = np.zeros((2, 2, 3), dtype=np.uint8)
    image = create_wandb_image(image_data, caption="OPD diagnostic")

    assert isinstance(image, wandb.Image)
