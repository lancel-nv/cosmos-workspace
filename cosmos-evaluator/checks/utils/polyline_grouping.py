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

"""Group world-space polyline pieces into whole objects by BEV rasterization.

RDS-HQ does not store a lane line or road boundary as one object. It stores many short fragments that meet
end-to-end: in a representative clip, 75% of the lane-line fragments are two-vertex stubs averaging 0.29 m,
and 147 of 178 canonical fragments have an endpoint coincident with another's start to a median of 0.00 m.
Feeding those fragments to a per-frame assignment tracker cannot work — neighboring fragments of the *same*
line sit ~0.35 m apart while the association gate is 1.5 m, so which fragment matches which is ambiguous by
construction, and the tracker emits roughly an order of magnitude more tracks than there are physical lines.

Rasterizing every fragment of one object type into a shared bird's-eye grid and taking 8-connected components
sidesteps the ambiguity: fragments of one line touch, so they land in one component, and parallel lines are
meters apart so they do not. On a 100-frame clip this reduced 19,139 lane-line fragments to 13 components on a
straight 12-lane highway, in 0.76 s against 1318 s for the assignment tracker.

The same construction associates anything that arrives as touching pieces of one ground-plane object, not
just annotations: hallucination detections of one lane line, back-projected onto the ground, fragment across
frames for the same reason and group the same way. :func:`label_connected_paths` is that general form and
:func:`label_connected_polylines` the adapter for :class:`Polyline` geometry.

Resolution matters and the safe direction is *fine*. Coarse grids, or thick strokes, dilate each line until
laterally-adjacent lines touch and fuse: at 0.20 m per pixel with a 3-pixel stroke an entire multi-lane road
collapses to one component. The defaults here are deliberately conservative.
"""

import logging

import cv2
import numpy as np

from checks.utils.polyline import Polyline

logger = logging.getLogger(__name__)

# Meters per pixel. Must stay well below the smallest separation between distinct parallel lines (~3 m for
# lane lines) — see the module docstring on why coarsening is the failure direction.
DEFAULT_RESOLUTION_M: float = 0.05

# Stroke width in pixels. 1 keeps the drawn line as thin as the grid allows; anything larger dilates every
# fragment and starts bridging neighboring lines.
DEFAULT_THICKNESS_PX: int = 1

# Padding around the geometry's bounding box so strokes near the edge are not clipped.
_MARGIN_M: float = 5.0

# Upper bound on grid cells (one uint8 each). A clip whose geometry spans kilometers would otherwise allocate
# a grid far larger than any real scene needs; past this the resolution is coarsened to fit and a warning is
# logged, because silently coarsening changes the grouping.
_MAX_GRID_CELLS: int = 64_000_000


def label_connected_polylines(
    geometries: list[Polyline],
    resolution_m: float = DEFAULT_RESOLUTION_M,
    thickness_px: int = DEFAULT_THICKNESS_PX,
) -> list[int]:
    """Assign each polyline a group id, where touching polylines share an id.

    A thin adapter over :func:`label_connected_paths` for world-space :class:`Polyline` geometry; the algorithm
    needs only each polyline's ground-plane path, so the z column is dropped.

    All geometries must be of a single object type; the caller is responsible for partitioning, exactly as the
    assignment tracker partitions before matching. Group ids are consecutive from 0 in order of first
    appearance in ``geometries``, so the result is deterministic and independent of OpenCV's internal labeling.

    Args:
        geometries: Polylines to group, in world coordinates. May be empty.
        resolution_m: Grid resolution in meters per pixel.
        thickness_px: Stroke width used to draw each polyline.

    Returns:
        One group id per input geometry, positionally aligned with ``geometries``.

    Raises:
        ValueError: If ``resolution_m`` is not positive or ``thickness_px`` is less than 1.
    """
    return label_connected_paths(
        [np.asarray(g.vertices_world, dtype=float)[:, :2] for g in geometries],
        resolution_m=resolution_m,
        thickness_px=thickness_px,
    )


def label_connected_paths(
    paths_xy: list[np.ndarray],
    resolution_m: float = DEFAULT_RESOLUTION_M,
    thickness_px: int = DEFAULT_THICKNESS_PX,
) -> list[int]:
    """Assign each ground-plane path a group id, where touching paths share an id.

    The general form of the grouping: it needs only a sequence of world XY points per object, so it serves both
    the RDS-HQ annotation fragments this module was written for and any other set of paths that should be
    associated by touching — hallucination detections back-projected onto the ground, for instance.

    Args:
        paths_xy: One ``(N, 2)`` array of world XY points per object, ``N >= 1``. May be empty.
        resolution_m: Grid resolution in meters per pixel.
        thickness_px: Stroke width used to draw each path. Widening the stroke is how a caller bridges gaps
            between pieces of one object — but it bridges *everything*, so see the module docstring on why
            coarsening is the failure direction.

    Returns:
        One group id per input path, positionally aligned with ``paths_xy``. Ids are consecutive from 0 in
        order of first appearance, so the result does not depend on OpenCV's internal label numbering.

    Raises:
        ValueError: If ``resolution_m`` is not positive or ``thickness_px`` is less than 1.
    """
    if resolution_m <= 0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m}")
    if thickness_px < 1:
        raise ValueError(f"thickness_px must be >= 1, got {thickness_px}")
    if not paths_xy:
        return []

    all_xy = np.vstack([np.asarray(p, dtype=float) for p in paths_xy])
    lower = all_xy.min(axis=0) - _MARGIN_M
    upper = all_xy.max(axis=0) + _MARGIN_M
    resolution_m = _fit_resolution(lower, upper, resolution_m)
    width = int((upper[0] - lower[0]) / resolution_m) + 1
    height = int((upper[1] - lower[1]) / resolution_m) + 1

    grid = np.zeros((height, width), dtype=np.uint8)
    pixels = [np.round((np.asarray(p, dtype=float) - lower) / resolution_m).astype(np.int32) for p in paths_xy]
    for path_px in pixels:
        if len(path_px) == 1:
            # cv2.polylines draws nothing for a single point, which would leave the path on background and
            # split it off into its own group by accident rather than by decision.
            column, row = int(path_px[0, 0]), int(path_px[0, 1])
            if 0 <= row < height and 0 <= column < width:
                grid[row, column] = 255
            if thickness_px > 1:
                cv2.circle(grid, (column, row), thickness_px // 2, 255, -1)
        else:
            cv2.polylines(grid, [path_px], isClosed=False, color=255, thickness=thickness_px)
    _, labeled = cv2.connectedComponents(grid, connectivity=8)

    # Map OpenCV's arbitrary label numbering onto ids ordered by first appearance.
    remap: dict[int, int] = {}
    groups: list[int] = []
    for path_px in pixels:
        raw = _dominant_label(labeled, path_px, width, height)
        groups.append(remap.setdefault(raw, len(remap)))
    return groups


def _fit_resolution(lower: np.ndarray, upper: np.ndarray, resolution_m: float) -> float:
    """Coarsen ``resolution_m`` just enough that the grid fits ``_MAX_GRID_CELLS``."""
    span_x, span_y = float(upper[0] - lower[0]), float(upper[1] - lower[1])
    cells = (span_x / resolution_m + 1) * (span_y / resolution_m + 1)
    if cells <= _MAX_GRID_CELLS:
        return resolution_m
    coarsened = resolution_m * float(np.sqrt(cells / _MAX_GRID_CELLS))
    logger.warning(
        "Polyline extent %.0f x %.0f m needs %.1e cells at %.3f m/px; coarsening to %.3f m/px to fit the "
        "%.1e cell budget. Grouping may merge lines that are closer together than the coarsened resolution.",
        span_x,
        span_y,
        cells,
        resolution_m,
        coarsened,
        float(_MAX_GRID_CELLS),
    )
    return coarsened


def _dominant_label(labeled: np.ndarray, polyline_px: np.ndarray, width: int, height: int) -> int:
    """Return the component label most of a polyline's pixels fall in.

    A vote rather than a single sample because rounding can place one vertex of a fragment on a neighboring
    component's pixel where two lines nearly touch. Zero is the background label and is never returned unless
    the polyline lands entirely on background, which rounding makes vanishingly unlikely but is handled by
    falling back to a unique negative id so the fragment forms its own group.
    """
    cols = np.clip(polyline_px[:, 0], 0, width - 1)
    rows = np.clip(polyline_px[:, 1], 0, height - 1)
    labels = labeled[rows, cols]
    labels = labels[labels != 0]
    if labels.size == 0:
        return -int(rows[0]) * width - int(cols[0]) - 1
    values, counts = np.unique(labels, return_counts=True)
    return int(values[np.argmax(counts)])
