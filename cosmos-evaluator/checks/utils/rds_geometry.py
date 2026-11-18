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
Abstract base class for RDS geometry types with projection capabilities.

This module defines the RDSGeometry abstract base class that unifies
Cuboid, Polyline, and Surface geometry classes by requiring a common
get_projected_mask() interface.
"""

from abc import ABC, abstractmethod

import numpy as np
from third_party.cosmos_drive_dreams_toolkits.utils.camera.base import CameraBase


class RDSGeometry(ABC):
    """
    Abstract base class for 3D geometry objects with projection capabilities.

    All RDS geometry types (Cuboid, Polyline, Surface) inherit from this class
    and must implement the get_projected_mask() method for projecting the
    geometry into image space.

    Subclasses also expose a ``world_distance(other)`` method used by the static
    object tracker to associate per-frame detections with canonical or
    previously-seen tracks. The default implementation returns ``+inf`` so
    cross-type comparisons never match; each concrete subclass overrides
    ``world_distance`` for instances of its own type and falls back to ``+inf``
    otherwise. Each subclass also exposes ``MATCH_GATE_METERS`` as the default
    upper bound for a match in tracking.
    """

    MATCH_GATE_METERS: float = float("inf")

    @abstractmethod
    def get_projected_mask(
        self,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute a binary mask and depth mask of the projected geometry in image space.

        Args:
            camera_to_world_pose: 4x4 camera-to-world pose matrix.
            camera_model: Camera model with .ray2pixel_np(points_cam) method.
            image_width: Output image width.
            image_height: Output image height.

        Returns:
            Tuple of:
                - Binary mask of shape (image_height, image_width), dtype=bool.
                - Depth mask of shape (image_height, image_width), dtype=float32,
                  containing per-pixel depth in camera space (z coordinate).
                  Pixels outside the mask have value inf.
        """
        raise NotImplementedError("Subclasses must implement get_projected_mask()")

    def get_projected_spine_mask(
        self,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
        min_ribbon_width_px: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Compute a binary mask and depth mask of the geometry's projected *spine* — its undilated centerline.

        This is the counterpart of :meth:`get_projected_mask` for geometry whose silhouette is an artefact of a
        chosen thickness rather than a real extent. A polyline's ribbon is 0.4 m wide because
        ``half_width_meters`` says so, not because the painted marking is; scoring against the ribbon therefore
        measures the half-width as much as the world model. The spine carries no such choice, so a correspondence
        metric built on it (``overlap_check.<class>.method: spine``) asks only "does the segmentation label the
        line where the line actually is".

        The default returns ``None``, meaning "this geometry has no meaningful centerline" — true of cuboids and
        surfaces, whose projected extent *is* their real extent. Callers must treat ``None`` as "use an
        area-based metric instead" rather than as an error.

        Args:
            camera_to_world_pose: 4x4 camera-to-world pose matrix.
            camera_model: Camera model with .ray2pixel_np(points_cam) method.
            image_width: Output image width.
            image_height: Output image height.
            min_ribbon_width_px: Drop centerline samples whose local geometry projects narrower than this
                many pixels. Ignored by geometry with no centerline.

        Returns:
            ``None`` for geometry without a centerline, else a tuple of:
                - Binary mask of shape (image_height, image_width), dtype=bool, holding an 8-connected
                  single-pixel-wide trace of the centerline.
                - Depth mask of shape (image_height, image_width), dtype=float32, with per-pixel camera-space
                  depth on the traced pixels and ``inf`` everywhere else.
        """
        return None

    def get_projected_spine_polylines(
        self,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
    ) -> list[np.ndarray]:
        """
        Project the geometry's centerline to image space as one or more open polylines.

        The point-list counterpart of :meth:`get_projected_spine_mask`, for consumers that need the centerline's
        *shape* rather than a rasterization of it — the KPI's delete test attributes a hallucination detection to
        a deleted object by distance to this curve, because IoU against the curve's bounding box cannot express
        "the detection is on the line" for geometry whose box is mostly empty.

        Several polylines rather than one because the range window can cut a line into disjoint visible spans;
        joining them would invent a segment the object does not occupy, and a detection sitting on that invented
        segment would be attributed to the object.

        Unlike :meth:`get_projected_spine_mask` this applies no width cutoff and does no occlusion test: it
        reports the full projected extent, matching :meth:`checks.utils.scene_rasterizer.SceneRasterizer.
        get_projection_bbox`, whose docstring notes a detection may land anywhere on an object's image
        footprint.

        The default returns ``[]`` — the same "no meaningful centerline" contract
        :meth:`get_projected_spine_mask` states with ``None``.

        Args:
            camera_to_world_pose: 4x4 camera-to-world pose matrix.
            camera_model: Camera model with .ray2pixel_np(points_cam) method.

        Returns:
            A list of ``(N, 2)`` float arrays of image-space points, ``N >= 2`` each; empty when the geometry has
            no centerline or none of it falls in the range window.
        """
        return []

    def world_distance(self, other: "RDSGeometry") -> float:
        """
        Compute a scalar distance between two geometries in world coordinates,
        used to associate static detections across frames.

        The default implementation returns ``+inf`` so cross-type comparisons do
        not match. Each concrete subclass overrides this for instances of its
        own type, returning a finite value when the geometries are comparable
        and falling back to ``+inf`` otherwise.

        Args:
            other: Another geometry to compare against.

        Returns:
            A non-negative distance, or ``+inf`` for incompatible types.
        """
        return float("inf")

    def ego_ground_distance_m(self, ego_pose: np.ndarray) -> float | None:
        """Ground-plane (forward/lateral) distance in meters from the ego vehicle to this geometry's centroid.

        The default is ``None`` for geometry without a centroid pose (lanelines, road boundaries, other extended
        map geometry). Concrete types with a centroid (e.g. ``Cuboid``) override this, so callers can ask any
        geometry for its ego distance without checking the object type.

        Args:
            ego_pose: 4x4 ego(camera)-to-world pose matrix.

        Returns:
            Ground distance in meters (rounded), or ``None`` when the geometry has no centroid.
        """
        return None

    def in_frame_fraction(
        self,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
    ) -> float:
        """
        Fraction of the projected geometry's bounding box inside the image frame.

        Returns the share of the projected bounding box that lies within
        ``[0, W-1] x [0, H-1]`` — ``1.0`` when fully in-frame, smaller when truncated by
        the frame edge, ``0.0`` when it does not project. This is a
        projected-bounding-box fraction, not a visible-surface fraction.

        The base implementation returns ``1.0`` so geometries without a meaningful
        frame-edge concept are never gated; only concrete types that can be truncated
        (e.g. :class:`Cuboid`) override it.
        """
        return 1.0
