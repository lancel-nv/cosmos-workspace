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
Coordinate transformation utilities for object processing.

This module provides shared coordinate transformation functions used by
multiple components in the objects system.
"""

from typing import Any

import numpy as np


def rdf_position_to_flu(position_rdf: np.ndarray) -> np.ndarray:
    """Convert a position from RDF (right, down, forward) to FLU (forward, left, up).

    The two conventions this codebase moves between: camera/ego poses are RDF (x right, y down, z forward),
    while every spatial rule that reasons about the vehicle — lane assignment, ego distance, oncoming — is
    stated in FLU (x forward, y left, z up). One implementation, because a sign error here is invisible in a
    symmetric corridor and wrong the moment the bounds differ.

    Args:
        position_rdf: (3,) or (4,) position in RDF coordinates.

    Returns:
        (3,) position in FLU coordinates ``[forward, left, up]``.
    """
    return np.array([position_rdf[2], -position_rdf[0], -position_rdf[1]])


def world_position_to_ego_flu(world_position: np.ndarray, ego_pose: np.ndarray) -> np.ndarray:
    """A world-frame point in the ego's FLU frame as ``[forward, left, up]``.

    Inverts the ego (or camera) pose to land in its RDF frame, then applies :func:`rdf_position_to_flu`. The
    chain behind every ego-relative measurement that is *stated* in FLU — the importance filter's lane
    assignment among them — so callers share it rather than re-deriving it and risking a transposed inverse or
    a flipped axis. (A ``Cuboid``'s
    ``ego_ground_distance_m`` reaches the same ground distance without converting at all — ``hypot(right,
    forward)`` straight off the RDF pose; the same number, one axis relabeling shorter.)

    Args:
        world_position: (3,) or (4,) point in world coordinates.
        ego_pose: 4x4 ego(camera)-to-world pose matrix.

    Returns:
        (3,) position in the ego's FLU coordinates ``[forward, left, up]``.
    """
    homogeneous = np.append(np.asarray(world_position, dtype=float)[:3], 1.0)
    position_rdf = np.linalg.inv(np.asarray(ego_pose, dtype=float)) @ homogeneous
    return rdf_position_to_flu(position_rdf[:3])


def rdf_pose_to_flu(pose_rdf: np.ndarray) -> np.ndarray:
    """Relabel a 4x4 pose's basis from RDF (right, down, forward) to FLU (forward, left, up).

    The pose-level sibling of :func:`rdf_position_to_flu`: the same convention applied to the *columns* of a
    transform, so the result maps FLU body coordinates to the same world frame the input mapped RDF ones to.
    Translation is untouched. Wanted when a FLU-convention pose is the output — an extrinsic handed to code
    that authors positions in FLU, say.

    **Not** the way to put a world point in the ego's frame: composing this with an inverse gives exactly
    :func:`world_position_to_ego_flu`, because the relabeling cancels against the one that function applies
    after inverting. Reach for that instead; two call sites had independently taken the long way round.

    Args:
        pose_rdf: 4x4 pose whose rotation columns are (right, down, forward).

    Returns:
        4x4 pose whose rotation columns are (forward, left, up), with the translation column unchanged.
    """
    return np.concatenate(
        [pose_rdf[:, 2:3], -pose_rdf[:, 0:1], -pose_rdf[:, 1:2], pose_rdf[:, 3:]],
        axis=1,
    )


def get_object_to_camera_pose(tracked_object: Any, camera_pose: np.ndarray) -> np.ndarray:
    """
    Get the transformation matrix from object to camera coordinates.

    Args:
        tracked_object: Object data from world model (must contain 'object_to_world' key)
        camera_pose: Camera pose for this frame (4x4 transformation matrix)


    Returns:
        Object pose in camera coordinates (4x4 transformation matrix)

    Raises:
        KeyError: If tracked_object doesn't contain 'object_to_world'
        ValueError: If matrices have incorrect shapes
    """
    if not hasattr(tracked_object, "object_to_world_pose"):
        raise KeyError("tracked_object must contain 'object_to_world_pose' attribute")

    if camera_pose.shape != (4, 4):
        raise ValueError("camera_pose must be a 4x4 transformation matrix")

    # Get object pose in world coordinates
    object_to_world = tracked_object.object_to_world_pose

    # Transform object pose to ego coordinates
    if object_to_world.shape != (4, 4):
        raise ValueError(f"object_to_world must be 4x4 matrix, got shape {object_to_world.shape}")

    camera_to_world = camera_pose
    world_to_camera = np.linalg.inv(camera_to_world)
    object_to_camera = world_to_camera @ object_to_world

    return object_to_camera


def extract_rpy_in_flu(R_flu_to_rdf: np.ndarray) -> tuple[float, float, float]:
    """
    Extract roll, pitch, yaw defined in forward-left-up coordinates from rotation matrix.

    Args:
        R_flu_to_rdf: Rotation matrix from forward-left-up to right-down-forward coordinates

    Returns:
        Tuple of (roll, pitch, yaw) in radians

    Raises:
        ValueError: If input matrix is not a 3x3 rotation matrix
    """
    if R_flu_to_rdf.shape != (3, 3):
        raise ValueError(f"R_flu_to_rdf must be 3x3 matrix, got shape {R_flu_to_rdf.shape}")

    # Convert RDF→FLU
    R = R_flu_to_rdf.T  # inverse of rotation matrix

    # Extract angles from R using ZYX convention (yaw-pitch-roll)
    if abs(R[2, 0]) != 1:
        pitch = -np.arcsin(R[2, 0])
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
    else:
        # Gimbal lock case
        yaw = 0
        if R[2, 0] == -1:
            pitch = np.pi / 2
            roll = yaw + np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll = -yaw + np.arctan2(-R[0, 1], -R[0, 2])

    return roll, pitch, yaw


def clip_polygon_to_z_planes(
    poly_cam: np.ndarray, near_plane_z: float | None = 1.0e-6, far_plane_z: float | None = None
) -> np.ndarray:
    """
    Clip a 3D polygon in camera coordinates against z-planes.

    The polygon is clipped against the half-space z >= near_plane_z and/or
    the half-space z <= far_plane_z. If both planes are provided, clipping
    is applied in sequence (near then far). If neither plane is provided,
    the original polygon is returned unchanged.

    Args:
        poly_cam: Nx3 array of polygon vertices in camera coordinates.
        near_plane_z: Optional near plane z value; keep z >= near_plane_z.
        far_plane_z: Optional far plane z value; keep z <= far_plane_z.

    Returns:
        Mx3 array of clipped polygon vertices (M can be 0).
    """
    pts = np.asarray(poly_cam, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"poly_cam must have shape (N, 3); got {pts.shape}")
    if pts.shape[0] == 0:
        return pts

    if near_plane_z is None and far_plane_z is None:
        return pts.copy()

    def _clip_against_plane(vertices: np.ndarray, z_plane: float, keep_greater_equal: bool) -> np.ndarray:
        if vertices.shape[0] == 0:
            return vertices

        current = vertices
        prev = np.roll(vertices, 1, axis=0)

        zc = current[:, 2]
        zp = prev[:, 2]
        if keep_greater_equal:
            inside_cur = zc >= z_plane
            inside_prev = zp >= z_plane
        else:
            inside_cur = zc <= z_plane
            inside_prev = zp <= z_plane

        crosses = inside_cur ^ inside_prev

        # Prepare arrays for intersections (placed before the "current" vertex in ordering)
        intersections = np.full_like(current, np.nan, dtype=float)
        if np.any(crosses):
            dz = (zc - zp)[crosses]
            # Avoid division by zero: for true crossings dz must be non-zero
            dz = np.where(np.abs(dz) < 1e-10, 1e-10, dz)
            t = (z_plane - zp[crosses]) / dz
            pc = current[crosses]
            pp = prev[crosses]
            inter = pp + t[:, None] * (pc - pp)
            inter[:, 2] = z_plane  # snap z to the clip plane (avoid FP drift)
            intersections[crosses] = inter

        # Points to include (current vertices that are inside)
        kept_current = np.full_like(current, np.nan, dtype=float)
        kept_current[inside_cur] = current[inside_cur]

        # Interleave [intersection_before_i, current_i_if_kept] preserving order without loops
        interleaved = np.stack([intersections, kept_current], axis=1).reshape(-1, 3)
        out = interleaved[~np.any(np.isnan(interleaved), axis=1)]

        # Remove consecutive duplicates that can arise when a vertex lies exactly on the plane
        if out.shape[0] >= 2:
            dup = np.all(np.isclose(out[1:], out[:-1]), axis=1)
            mask = np.ones(out.shape[0], dtype=bool)
            mask[1:] = ~dup
            out = out[mask]
        return out

    out = pts
    if near_plane_z is not None:
        out = _clip_against_plane(out, float(near_plane_z), keep_greater_equal=True)
    if far_plane_z is not None and out.shape[0] > 0:
        out = _clip_against_plane(out, float(far_plane_z), keep_greater_equal=False)
    return out
