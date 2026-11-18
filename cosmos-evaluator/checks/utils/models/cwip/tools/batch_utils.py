# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for merging batch outputs from CWIP model inference."""

from typing import Any

import torch


def merge_batch_outputs(outputs: list[dict]) -> dict:
    """
    Merge a list of batch outputs into a single output dictionary.

    Concatenates tensors along the batch dimension and merges lists element-wise.

    Args:
        outputs: List of output dictionaries from model.forward()

    Returns:
        Merged output dictionary with all batches concatenated.
    """
    if not outputs:
        return {}

    merged: dict[Any, Any] = {}
    first = outputs[0]

    for key in first:
        values = [out[key] for out in outputs]

        # Skip None values
        if values[0] is None:
            merged[key] = None
        # Handle lists (e.g., camera_features_list, world_features_list)
        elif isinstance(values[0], list):
            # Merge lists element-wise: concatenate corresponding tensors
            num_elements = len(values[0])
            merged[key] = [torch.cat([v[i] for v in values], dim=0) for i in range(num_elements)]
        # Handle tensors
        elif isinstance(values[0], torch.Tensor):
            merged[key] = torch.cat(values, dim=0)
        else:
            # For other types, just use the first value (shouldn't happen)
            merged[key] = values[0]

    return merged
