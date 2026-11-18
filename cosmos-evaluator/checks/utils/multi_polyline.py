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

"""Container geometry for a static track observed as several disjoint polylines in one frame.

RDS-HQ stores extended map geometry (lane lines, road boundaries, wait lines) as many short polyline
fragments rather than as whole lines — 75% of the lane-line fragments in a typical clip are two-vertex stubs
averaging 0.29 m. ``polyline_grouping`` reassembles those fragments into one track per physical line, but a
track's fragments in a single frame are not always contiguous: roughly one (track, frame) cell in five is
split into two or more disjoint pieces by occlusion or by the labeler simply not annotating a stretch.

``geometry_per_frame`` holds exactly one geometry per track per frame, so those pieces have to be carried by a
single object. Concatenating them into one :class:`Polyline` would draw a segment straight across the gap and
project lane paint that does not exist, so this container keeps the pieces separate and composes their masks
instead.
"""

import numpy as np
from third_party.cosmos_drive_dreams_toolkits.utils.camera.base import CameraBase

from checks.utils.polyline import Polyline
from checks.utils.rds_geometry import RDSGeometry


class MultiPolyline(RDSGeometry):
    """Several disjoint :class:`Polyline` pieces that together form one static track in one frame.

    Projection is the union of the members' masks, with per-pixel depth reduced by ``minimum`` so a nearer
    piece wins where two overlap — the same rule :class:`Polyline` applies across its own segments.
    """

    # A track's pieces are the same physical line, so the polyline gate applies unchanged.
    MATCH_GATE_METERS: float = Polyline.MATCH_GATE_METERS

    def __init__(self, polylines: list[Polyline]):
        """Initialize from two or more disjoint pieces.

        Args:
            polylines: The pieces, in no particular order. A single-piece track should use that
                :class:`Polyline` directly rather than wrapping it.

        Raises:
            ValueError: If ``polylines`` is empty.
        """
        if not polylines:
            raise ValueError("MultiPolyline requires at least one Polyline")
        self.polylines = list(polylines)

    @property
    def vertices_world(self) -> np.ndarray:
        """All member vertices stacked into one (N, 3) array.

        Provided so callers that only measure extent or distance keep working against either geometry type.
        **Consecutive rows may belong to different pieces**, so anything that treats the array as a connected
        path — arc length, segment iteration, rendering — must iterate ``polylines`` instead.
        """
        return np.vstack([p.vertices_world for p in self.polylines])

    def get_projected_mask(
        self,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project every piece and combine them.

        Args:
            camera_to_world_pose: 4x4 camera-to-world pose matrix.
            camera_model: Camera model with a ``ray2pixel_np(points_cam)`` method.
            image_width: Output image width.
            image_height: Output image height.

        Returns:
            Tuple of:
                - Binary mask of shape (image_height, image_width), dtype=bool — the union of the pieces.
                - Depth mask of shape (image_height, image_width), dtype=float32, per-pixel camera-space depth,
                  the minimum across pieces. Pixels outside the mask have value inf.
        """
        mask = np.zeros((image_height, image_width), dtype=bool)
        depth = np.full((image_height, image_width), np.inf, dtype=np.float32)
        for piece in self.polylines:
            piece_mask, piece_depth = piece.get_projected_mask(
                camera_to_world_pose, camera_model, image_width, image_height
            )
            mask |= piece_mask
            np.minimum(depth, piece_depth, out=depth)
        return mask, depth

    def get_projected_spine_mask(
        self,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
        min_ribbon_width_px: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Trace every piece's centerline and combine them, mirroring :meth:`get_projected_mask`.

        Overriding this is not optional. Without it a multi-piece track would inherit the ``RDSGeometry``
        default of ``None`` and the ``spine`` correspondence metric would silently fall back to ``ratio`` for
        exactly those frames where a line happens to be split — scoring one physical lane line by two different
        metrics depending on its per-frame fragmentation.

        Args:
            camera_to_world_pose: 4x4 camera-to-world pose matrix.
            camera_model: Camera model with a ``ray2pixel_np(points_cam)`` method.
            image_width: Output image width.
            image_height: Output image height.
            min_ribbon_width_px: Ribbon-width cutoff for the trace, forwarded to each piece.

        Returns:
            Tuple of:
                - Binary mask of shape (image_height, image_width), dtype=bool — the union of the pieces'
                  traced centerlines.
                - Depth mask of shape (image_height, image_width), dtype=float32, the minimum across pieces.
                  Pixels outside the mask have value inf.
        """
        mask = np.zeros((image_height, image_width), dtype=bool)
        depth = np.full((image_height, image_width), np.inf, dtype=np.float32)
        for piece in self.polylines:
            piece_mask, piece_depth = piece.get_projected_spine_mask(
                camera_to_world_pose, camera_model, image_width, image_height, min_ribbon_width_px
            )
            mask |= piece_mask
            np.minimum(depth, piece_depth, out=depth)
        return mask, depth

    def get_projected_spine_polylines(
        self,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
    ) -> list[np.ndarray]:
        """Every piece's projected centerline, concatenated — the point-list analogue of the mask override.

        Overriding is load-bearing for the same reason as :meth:`get_projected_spine_mask`: without it a
        multi-piece track would inherit the base ``[]`` and its consumers would read "this object has no
        centerline" on exactly the frames where the line is fragmented. Pieces stay separate entries rather than
        being joined, since they are already disjoint spans of one line.

        Args:
            camera_to_world_pose: 4x4 camera-to-world pose matrix.
            camera_model: Camera model with a ``ray2pixel_np(points_cam)`` method.

        Returns:
            A list of ``(N, 2)`` float arrays of image-space points, pooled across the member pieces.
        """
        polylines: list[np.ndarray] = []
        for piece in self.polylines:
            polylines.extend(piece.get_projected_spine_polylines(camera_to_world_pose, camera_model))
        return polylines

    def world_distance(self, other: RDSGeometry) -> float:
        """Smallest :meth:`Polyline.world_distance` over the member pieces.

        A track is the same physical line however many pieces it happens to be split into, so the closest piece
        represents it. Accepts :class:`Polyline` and :class:`MultiPolyline`; returns ``+inf`` otherwise, matching
        the base class contract for incompatible types.

        Args:
            other: Another geometry to compare against.

        Returns:
            A non-negative distance, or ``+inf`` for incompatible types.
        """
        others = other.polylines if isinstance(other, MultiPolyline) else [other]
        if not all(isinstance(o, Polyline) for o in others):
            return float("inf")
        return min(
            (mine.world_distance(theirs) for mine in self.polylines for theirs in others),
            default=float("inf"),
        )
