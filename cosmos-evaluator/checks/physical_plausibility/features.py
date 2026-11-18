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

"""Letterbox preprocessing and fractional token validity weights."""

from __future__ import annotations

import cv2
import numpy as np

IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
RESOLUTION = 384
PATCH_SIZE = 16
TEMPORAL_TOKENS = 32
SPATIAL_TOKENS = RESOLUTION // PATCH_SIZE
EMBEDDING_DIM = 1024


def _to_unit_rgb(frames: np.ndarray) -> np.ndarray:
    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError(f"frames must have shape [F,H,W,3], got {values.shape}")
    unit = values.astype(np.float32)
    if np.issubdtype(values.dtype, np.integer) or float(np.nanmax(unit)) > 1.5:
        unit /= 255.0
    return np.clip(unit, 0.0, 1.0)


def letterbox_frames(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize to a unit RGB canvas and fractional patch weights, before normalization."""
    unit = _to_unit_rgb(frames)
    source_height, source_width = unit.shape[1:3]
    scale = min(RESOLUTION / source_width, RESOLUTION / source_height)
    width = max(1, min(RESOLUTION, round(source_width * scale)))
    height = max(1, min(RESOLUTION, round(source_height * scale)))
    resized = np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in unit]).astype(
        np.float32
    )
    left = (RESOLUTION - width) // 2
    top = (RESOLUTION - height) // 2
    canvas = np.broadcast_to(IMAGE_MEAN, (len(unit), RESOLUTION, RESOLUTION, 3)).copy()
    canvas[:, top : top + height, left : left + width] = resized
    mask = np.zeros((RESOLUTION, RESOLUTION), dtype=np.float32)
    mask[top : top + height, left : left + width] = 1.0
    return canvas, pixel_mask_to_token_weights(mask)


def pixel_mask_to_token_weights(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=np.float32)
    if values.shape != (RESOLUTION, RESOLUTION):
        raise ValueError(f"pixel mask must have shape {(RESOLUTION, RESOLUTION)}")
    return values.reshape(SPATIAL_TOKENS, PATCH_SIZE, SPATIAL_TOKENS, PATCH_SIZE).mean(axis=(1, 3))
