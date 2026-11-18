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

"""
CWIP Inference Helper for object processing.

This module provides a helper class that extends CWIPInference with additional functionality needed by the object
processor.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from checks.utils.models.cwip.cwip import CWIPInference
from checks.utils.models.cwip.modules.class_mapping import ObjectTypeClassMapping, get_object_type_class_mapping
from checks.utils.models.cwip.visualization.world_model_utils import ObjectType

logger = logging.getLogger(__name__)

COSMOS_EVALUATOR_OBJECT_TYPE_STRING_TO_ENUMS: dict[str, list[ObjectType]] = {
    "vehicle": [ObjectType.BBOX_CAR, ObjectType.BBOX_TRUCK, ObjectType.BBOX_OTHERS],
    "pedestrian": [ObjectType.BBOX_PEDESTRIAN],
    "motorcycle": [ObjectType.BBOX_CYCLIST],
    "bicycle": [ObjectType.BBOX_CYCLIST],
    "traffic_light": [
        ObjectType.TRAFFIC_LIGHT_RED,
        ObjectType.TRAFFIC_LIGHT_YELLOW,
        ObjectType.TRAFFIC_LIGHT_GREEN,
        ObjectType.TRAFFIC_LIGHT_UNKNOWN,
    ],
    "traffic_sign": [ObjectType.HDMAP_TRAFFIC_SIGNS],
    "lane_line": [
        ObjectType.GEOMETRY_LANELINE_OTHER,
        ObjectType.GEOMETRY_LANELINE_WHITE_SOLID_SINGLE,
        ObjectType.GEOMETRY_LANELINE_WHITE_LONG_DASHED_SINGLE,
        ObjectType.GEOMETRY_LANELINE_WHITE_SHORT_DASHED_SINGLE,
        ObjectType.GEOMETRY_LANELINE_WHITE_DOT_DASHED_SINGLE,
        ObjectType.GEOMETRY_LANELINE_WHITE_OTHER,
        ObjectType.GEOMETRY_LANELINE_WHITE_SOLID_GROUP,
        ObjectType.GEOMETRY_LANELINE_YELLOW_DOT_SOLID_SINGLE,
        ObjectType.GEOMETRY_LANELINE_YELLOW_DOT_SOLID_GROUP,
        ObjectType.GEOMETRY_LANELINE_YELLOW_DASHED_SOLID,
        ObjectType.GEOMETRY_LANELINE_YELLOW_SOLID_DASHED,
        ObjectType.GEOMETRY_LANELINE_YELLOW_SOLID_SINGLE,
        ObjectType.GEOMETRY_LANELINE_YELLOW_SOLID_GROUP,
        ObjectType.GEOMETRY_LANELINE_YELLOW_LONG_DASHED_SINGLE,
        ObjectType.GEOMETRY_LANELINE_YELLOW_OTHER,
    ],
    "wait_line": [ObjectType.HDMAP_WAIT_LINES],
    "crosswalk": [ObjectType.HDMAP_CROSSWALKS],
    "road_boundary": [ObjectType.HDMAP_ROAD_BOUNDARIES],
    "pole": [ObjectType.HDMAP_POLES],
    "road_marking": [ObjectType.HDMAP_ROAD_MARKINGS],
}


def resolve_model_class_indices(
    object_types: list[ObjectType],
    mapping_cls: type[ObjectTypeClassMapping] | None,
) -> list[int]:
    """Convert ObjectType enum values to model class indices using the class mapping.

    When a class mapping is provided (e.g., V3 or V4), model output indices differ
    from ObjectType enum values. This function translates ObjectType enums into the
    indices the model actually predicts, skipping any types marked as IGNORED.

    Args:
        object_types: ObjectType enum values to convert.
        mapping_cls: The model's class mapping, or None for identity mapping.

    Returns:
        Deduplicated list of model class indices to match against.
    """
    if mapping_cls is None:
        return [int(obj_type) for obj_type in object_types]

    indices: set[int] = set()
    for obj_type in object_types:
        try:
            model_idx = mapping_cls.get_model_class(obj_type)
        except (KeyError, ValueError):
            logger.debug("ObjectType %s not found in class mapping %s", obj_type, mapping_cls.__name__)
            continue
        if model_idx == ObjectTypeClassMapping.CLASS_INDEX_IGNORED:
            continue
        indices.add(model_idx)
    return list(indices)


def resize_index_map(
    indices: torch.Tensor,
    target_size: tuple[int, int],
    mode: str = "nearest",
) -> torch.Tensor:
    """Resize a batch of integer index maps via spatial interpolation.

    This handles the common CWIP post-processing pattern of upscaling patch-resolution class-index tensors to a higher
    spatial resolution (e.g. native video or camera-model resolution).

    Args:
        indices: Integer class index tensor of shape ``(N, H, W)``.
        target_size: Target ``(height, width)``.
        mode: Interpolation mode forwarded to
            :func:`torch.nn.functional.interpolate` (default ``"nearest"``).

    Returns:
        Resized ``torch.long`` tensor of shape ``(N, target_h, target_w)``.
    """
    if indices.ndim != 3:
        raise ValueError(f"Expected 3-D tensor (N, H, W), got shape {indices.shape}")
    h, w = indices.shape[1], indices.shape[2]
    if (h, w) == target_size:
        return indices
    return (
        torch.nn.functional.interpolate(
            indices.unsqueeze(1).float(),
            size=target_size,
            mode=mode,
        )
        .squeeze(1)
        .to(torch.long)
    )


class CWIPInferenceHelper(CWIPInference):
    """
    Helper class extending CWIPInference with objects-processing utilities.

    This class adds methods like get_class_mask that are needed by the objects
    checker but are not part of the core CWIP inference functionality.
    """

    def get_class_mask(self, pred: torch.Tensor, object_type_class_name: str | None = None) -> np.ndarray:
        """
        Get the type mask for a specific object class from the predicted class indices for a single frame.

        The model's ``object_type_class_mapping`` (stored in ``self.config``) is used to translate ObjectType enum
        values into model output class indices. When no mapping is configured (legacy models), ObjectType enum values
        are used directly as indices.

        Args:
            pred: Predicted similarity and type class indices stacked into a single
                  tensor of shape (2, H, W).  Layer 0 is a binary similarity mask
                  (0/1), layer 1 contains model-output type class indices.
            object_type_class_name: High-level object class name (e.g. "vehicle").
        Returns:
            The similarity mask filtered by the specified object class (H, W)
        """
        if pred.shape[0] != 2:
            raise ValueError(f"Pred shape must be (2, H, W), but got {pred.shape}")

        mapping_cls = get_object_type_class_mapping(getattr(self.config, "object_type_class_mapping", None))

        all_type_indices = pred[1, :, :]
        type_mask = torch.zeros_like(all_type_indices, dtype=torch.bool)

        if object_type_class_name is not None:
            object_type_list = COSMOS_EVALUATOR_OBJECT_TYPE_STRING_TO_ENUMS.get(object_type_class_name, [])
            if not object_type_list:
                raise ValueError(f"Invalid object type class name: {object_type_class_name}")
            model_indices = resolve_model_class_indices(object_type_list, mapping_cls)
            for model_idx in model_indices:
                type_mask |= all_type_indices == model_idx
        else:
            type_mask = all_type_indices

        return (pred[0, :, :].to(torch.bool) & type_mask).cpu().numpy()
