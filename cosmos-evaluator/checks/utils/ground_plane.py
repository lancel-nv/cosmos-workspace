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

"""Back-projection of image pixels onto the world ground plane.

Road markings and boundaries lie on the ground, so a pixel belonging to one can be placed in the world by
intersecting its viewing ray with ``z = 0``. That turns an image-space observation, which slides across the
frame as the ego drives, into a world-space one, which for static geometry does not move at all — the frame in
which a thing seen over many frames is one thing.

The ground assumption is what makes this work and also its only real error source: a pixel that is not on the
ground lands wherever its ray happens to cross the plane, and range resolution degrades sharply toward the
horizon, where a pixel of vertical error is meters of range. ``max_range_m`` bounds both.
"""

import numpy as np
from third_party.cosmos_drive_dreams_toolkits.utils.camera.base import CameraBase


def backproject_to_ground(
    pixels_xy: np.ndarray,
    camera_to_world: np.ndarray,
    camera_model: CameraBase,
    max_range_m: float,
) -> np.ndarray:
    """Intersect each pixel's viewing ray with the world ``z = 0`` plane.

    Args:
        pixels_xy: (N, 2) pixel coordinates as (x, y).
        camera_to_world: 4x4 camera-to-world pose.
        camera_model: Camera model exposing ``pixel2ray_np(pixels) -> (rays, valid)``.
        max_range_m: Drop intersections further than this from the camera.

    Returns:
        (M, 2) world XY points with M <= N, in input order. Points are dropped when the camera model cannot
        unproject the pixel, when the ray does not descend to the plane (at or above the horizon), or when the
        intersection is beyond ``max_range_m``.
    """
    pixels = np.asarray(pixels_xy, dtype=float)
    if pixels.size == 0:
        return np.empty((0, 2), dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"pixels_xy must have shape (N, 2), got {pixels.shape}")
    if camera_to_world.shape != (4, 4):
        raise ValueError("camera_to_world must be 4x4")

    # FTheta's backward polynomial does not invert across the whole sensor, so it flags pixels it cannot
    # unproject rather than raising.
    rays_cam, valid = camera_model.pixel2ray_np(pixels)
    rays_cam = np.asarray(rays_cam, dtype=float)[np.asarray(valid, dtype=bool)]
    if rays_cam.size == 0:
        return np.empty((0, 2), dtype=float)

    # Directions rotate with the pose but must not pick up its translation.
    rays_world = rays_cam @ camera_to_world[:3, :3].T
    origin = camera_to_world[:3, 3]

    descending = rays_world[:, 2] < -1e-6
    if not descending.any():
        return np.empty((0, 2), dtype=float)
    rays_world = rays_world[descending]

    distance = -origin[2] / rays_world[:, 2]
    points = origin[None, :] + distance[:, None] * rays_world
    in_range = (distance > 0) & (np.linalg.norm(points - origin[None, :], axis=1) <= max_range_m)
    return points[in_range][:, :2]
