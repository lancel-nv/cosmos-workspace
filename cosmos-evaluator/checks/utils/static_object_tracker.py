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
Static-object tracker.

Assigns clip-stable track IDs to per-frame static-object detections, using a
canonical set of objects (one per physical static object known to the dataset
producer) as the seed for stable enumeration. Per-frame detections that don't
match any canonical produce new synthesized tracks so the clip's full
geometry is still represented even when the canonical undercovers it.

The tracker operates entirely in world coordinates -- per-frame and canonical
geometries are both stored that way already -- so no pose handling is needed.
Matching uses each geometry's ``world_distance`` method, gated by its
``MATCH_GATE_METERS`` class constant.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from checks.utils.multi_polyline import MultiPolyline
from checks.utils.polyline import Polyline
from checks.utils.polyline_grouping import DEFAULT_RESOLUTION_M, DEFAULT_THICKNESS_PX, label_connected_polylines
from checks.utils.rds_geometry import RDSGeometry

logger = logging.getLogger(__name__)

# Finite placeholder substituted for +inf when feeding the cost matrix to
# linear_sum_assignment, which doesn't accept infs. Any value larger than the
# largest plausible gate works -- post-hoc filtering on the original cost
# matrix removes spurious assignments created by this padding.
_INFEASIBLE_COST = 1e9


@dataclass(frozen=True)
class StaticTrackAssignment:
    """What :func:`assign_static_track_ids` resolved for one clip.

    Attributes:
        per_frame: ``{frame index: {track id: detection dict}}``. Every frame present in the input appears
            here, possibly empty.
        absorbed_canonical_ids: Canonical track IDs whose geometry is represented by *another* track, because
            connectivity grouping put several canonical fragments of one physical line into one group and only
            the first owns the ID. These carry no per-frame geometry and can be dropped without losing
            anything. Deliberately distinct from a canonical that is simply never observed: that one is the
            only record of its geometry and must be kept.
        clip_geometry: ``{track id: geometry}`` for grouped polyline tracks — the single observation, canonical
            or per-frame, that covers the most arc length. A clip-level stand-in that is a real annotation
            rather than an average, and the same rule for every polyline track regardless of origin.
    """

    per_frame: dict[int, dict[str, dict[str, Any]]]
    absorbed_canonical_ids: frozenset[str]
    clip_geometry: dict[str, RDSGeometry]


def _coverage_m(geometry: RDSGeometry) -> float:
    """Total ground-plane arc length of a geometry, summed over the pieces of a :class:`MultiPolyline`."""
    total = 0.0
    for piece in getattr(geometry, "polylines", [geometry]):
        vertices = np.asarray(piece.vertices_world, dtype=float)
        if len(vertices) >= 2:
            total += float(np.linalg.norm(np.diff(vertices[:, :2], axis=0), axis=1).sum())
    return total


def assign_static_track_ids(
    canonical_objects: list[dict[str, Any]],
    per_frame_objects: dict[int, list[dict[str, Any]]],
    id_allocator: Callable[[], str],
    resolution_m: float = DEFAULT_RESOLUTION_M,
    thickness_px: int = DEFAULT_THICKNESS_PX,
) -> StaticTrackAssignment:
    """
    Assign clip-stable track IDs to per-frame static-object detections.

    Each input ``object_dict`` (canonical or per-frame) must contain at least:
        - ``object_type``: str -- used to partition matching by type.
        - ``geometry``: an :class:`RDSGeometry` instance whose
          ``world_distance(other)`` and ``MATCH_GATE_METERS`` drive matching.

    Two strategies are used, chosen per object type by geometry:

        - **Polyline types** (lane lines, road boundaries, wait lines) are grouped by connectivity via
          :func:`checks.utils.polyline_grouping.label_connected_polylines` rather than matched frame to frame.
          RDS-HQ stores these as short fragments that meet end-to-end, so neighboring fragments of one line
          sit far inside the association gate and per-frame assignment cannot tell them apart. Where a track
          has several disjoint fragments in one frame they are carried by a
          :class:`checks.utils.multi_polyline.MultiPolyline` so that ``geometry_per_frame`` keeps exactly one
          geometry per track per frame without inventing a segment across the gap.
        - **All other types** (cuboids, surfaces) keep the per-frame Hungarian assignment described below,
          which suits compact objects with stable poses.

    Algorithm (non-polyline types):
        1. Allocate IDs for every canonical object first, in canonical-list
           order, before any per-frame matching begins. The first
           ``len(canonical_objects)`` IDs returned by ``id_allocator`` belong
           to canonical tracks. This invariant lets external references to
           numerical static IDs survive a re-load as long as the canonical
           doesn't change.
        2. Iterate frames in ascending order. For each frame, partition both
           detections and active tracks by ``object_type`` and run a Hungarian
           assignment per type using ``world_distance`` as the cost, gated by
           the detection's ``MATCH_GATE_METERS``.
        3. Detections that don't match any active track create new tracks via
           ``id_allocator``; these tracks join the active set immediately so
           subsequent frames can match against them.
        4. A track's "active geometry" is the most recent matched detection's
           geometry (or the canonical's geometry for canonical tracks that
           haven't yet been matched). World coordinates make this stable
           without a staleness heuristic: a track left behind in the world
           naturally falls outside the gate for later frames.

    Args:
        canonical_objects: Authoritative static-object set defining stable IDs.
            May be empty.
        per_frame_objects: Frame index -> list of per-frame detection dicts.
            Frames may have empty detection lists.
        id_allocator: Zero-arg callable returning a fresh string ID. Called
            ``len(canonical_objects)`` times up front, then once per
            unmatched per-frame detection.

    Returns:
        A :class:`StaticTrackAssignment`. Canonical tracks that were never matched in any frame do not appear
        in ``per_frame`` (their IDs were still allocated); those absorbed into a sibling's group are listed in
        ``absorbed_canonical_ids`` so a caller can tell "represented elsewhere" from "never observed".

    The function is deterministic for fixed inputs (and a deterministic
    allocator). Cross-type matches are forbidden by partitioning, and
    additionally by ``world_distance`` returning ``+inf`` for incompatible
    geometry types.
    """
    # Step 1: Allocate canonical IDs first, in canonical-list order.
    active: dict[str, dict[str, Any]] = {}  # track_id -> {"object_type", "geometry"}
    canonical_track_ids: list[str] = []
    for obj in canonical_objects:
        track_id = id_allocator()
        active[track_id] = {"object_type": obj["object_type"], "geometry": obj["geometry"]}
        canonical_track_ids.append(track_id)

    # Track ID order matters for deterministic Hungarian column assignment.
    active_track_ids: list[str] = list(active.keys())

    # Contract: every frame present in the input appears in the output, even when it has no detections.
    result: dict[int, dict[str, dict[str, Any]]] = {fi: {} for fi in per_frame_objects}

    # Polyline types are grouped by connectivity rather than matched frame to frame -- RDS-HQ splits a lane
    # line into fragments that sit far closer to each other than the association gate, which makes per-frame
    # assignment ambiguous by construction. See ``polyline_grouping``.
    polyline_types = {obj["object_type"] for obj in canonical_objects if isinstance(obj["geometry"], Polyline)} | {
        det["object_type"]
        for dets in per_frame_objects.values()
        for det in dets
        if isinstance(det["geometry"], Polyline)
    }
    absorbed: set[str] = set()
    clip_geometry: dict[str, RDSGeometry] = {}
    for obj_type in sorted(polyline_types):
        _assign_polyline_type(
            obj_type,
            canonical_objects,
            canonical_track_ids,
            per_frame_objects,
            id_allocator,
            result,
            absorbed,
            clip_geometry,
            resolution_m,
            thickness_px,
        )

    # Step 2: Iterate frames in ascending order.
    for frame_idx in sorted(per_frame_objects.keys()):
        detections = per_frame_objects[frame_idx]
        frame_static: dict[str, dict[str, Any]] = result[frame_idx]
        used_dets: set[int] = set()

        # Partition detections and tracks by object_type so cross-type
        # candidates never enter the same cost matrix.
        det_types = {d["object_type"] for d in detections} - polyline_types
        for obj_type in sorted(det_types):
            type_dets = [(i, d) for i, d in enumerate(detections) if d["object_type"] == obj_type]
            type_tracks = [tid for tid in active_track_ids if active[tid]["object_type"] == obj_type]
            if not type_tracks:
                continue  # all type_dets fall through to the new-track loop below

            cost = _build_cost_matrix(type_dets, type_tracks, active)
            cost_for_hungarian = np.where(np.isinf(cost), _INFEASIBLE_COST, cost)
            row_ind, col_ind = linear_sum_assignment(cost_for_hungarian)
            for r, c in zip(row_ind, col_ind):
                if not np.isfinite(cost[r, c]):
                    continue  # original cost was +inf (above gate); not a real match
                det_idx, detection = type_dets[r]
                track_id = type_tracks[c]
                used_dets.add(det_idx)
                frame_static[track_id] = detection
                active[track_id]["geometry"] = detection["geometry"]

        # Step 3: Unmatched detections create new tracks.
        for i, detection in enumerate(detections):
            if i in used_dets or detection["object_type"] in polyline_types:
                continue
            new_tid = id_allocator()
            active[new_tid] = {"object_type": detection["object_type"], "geometry": detection["geometry"]}
            active_track_ids.append(new_tid)
            frame_static[new_tid] = detection

    return StaticTrackAssignment(result, frozenset(absorbed), clip_geometry)


def _assign_polyline_type(
    obj_type: str,
    canonical_objects: list[dict[str, Any]],
    canonical_track_ids: list[str],
    per_frame_objects: dict[int, list[dict[str, Any]]],
    id_allocator: Callable[[], str],
    result: dict[int, dict[str, dict[str, Any]]],
    absorbed: set[str],
    clip_geometry: dict[str, RDSGeometry],
    resolution_m: float,
    thickness_px: int,
) -> None:
    """Group one polyline object type by connectivity and write its tracks into ``result`` in place.

    Args:
        obj_type: The single object type to handle; every other type is ignored.
        canonical_objects: The full canonical list (filtered internally by ``obj_type``).
        canonical_track_ids: IDs already allocated for ``canonical_objects``, positionally aligned with it.
        per_frame_objects: The full per-frame detection map (filtered internally by ``obj_type``).
        id_allocator: Allocator for groups that contain no canonical fragment.
        result: Output map, mutated in place.
        absorbed: Set of canonical track IDs represented by a sibling, extended in place.
        clip_geometry: Map of track ID to greatest-coverage observation, extended in place.
        resolution_m: Grid resolution passed through to the grouper.
        thickness_px: Stroke width passed through to the grouper.
    """
    # A type reaches this path because *some* of its geometry is a Polyline, which does not guarantee all of it
    # is: ``_parse_shape3d_entry`` derives geom_type from the shape's own keys rather than from the folder, so a
    # cuboid or surface shape filed under 3d_lanelines arrives here carrying object_type "LaneLine". Grouping
    # reads polyline-only fields, so such a record would take the whole load down with an AttributeError. Skip it
    # and say so, matching how ``_instantiate_label_geometry`` reports geometry it cannot build.
    skipped = 0
    canonical_positions = []
    geometries: list[Polyline] = []
    for position, obj in enumerate(canonical_objects):
        if obj["object_type"] != obj_type:
            continue
        if not isinstance(obj["geometry"], Polyline):
            skipped += 1
            continue
        canonical_positions.append(position)
        geometries.append(obj["geometry"])
    detections: list[tuple[int, dict[str, Any]]] = []
    for frame_idx in sorted(per_frame_objects):
        for det in per_frame_objects[frame_idx]:
            if det["object_type"] != obj_type:
                continue
            if not isinstance(det["geometry"], Polyline):
                skipped += 1
                continue
            detections.append((frame_idx, det))
            geometries.append(det["geometry"])
    if skipped:
        logger.warning(
            "Skipped %d %s record(s) whose geometry is not a Polyline; connectivity grouping cannot place them.",
            skipped,
            obj_type,
        )
    if not geometries:
        return

    groups = label_connected_polylines(geometries, resolution_m, thickness_px)
    n_canonical = len(canonical_positions)

    # A group containing canonical fragments inherits the ID of its earliest one, so numerical static IDs stay
    # anchored to the canonical list across reloads. Fragments of one physical line collapse into a single
    # track, leaving the other canonical IDs allocated but unused -- the documented behavior for a canonical
    # that never appears in any frame.
    group_to_track: dict[int, str] = {}
    group_members: dict[int, list[str]] = {}
    canonical_by_group: dict[int, list[RDSGeometry]] = {}
    for position, group in zip(canonical_positions, groups[:n_canonical]):
        group_to_track.setdefault(group, canonical_track_ids[position])
        group_members.setdefault(group, []).append(canonical_track_ids[position])
        canonical_by_group.setdefault(group, []).append(canonical_objects[position]["geometry"])

    cells: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for (frame_idx, detection), group in zip(detections, groups[n_canonical:]):
        cells.setdefault((group, frame_idx), []).append(detection)

    # Sorted so IDs for canonical-free groups are allocated in a deterministic order.
    merged_cells: dict[int, list[dict[str, Any]]] = {}
    for group, frame_idx in sorted(cells):
        if group not in group_to_track:
            group_to_track[group] = id_allocator()
        merged = _merge_detections(cells[(group, frame_idx)])
        result[frame_idx][group_to_track[group]] = merged
        merged_cells.setdefault(group, []).append(merged)

    # Only a group that per-frame data actually reaches can spare its extra canonical fragments: in a group
    # with no per-frame detections each canonical is the sole record of its own geometry.
    for group, members in group_members.items():
        if group in merged_cells:
            absorbed.update(members[1:])

    # Clip-level stand-in: the single observation covering the most arc, canonical or per-frame alike. Ties keep
    # the first candidate in canonical-then-ascending-frame order, so the choice is deterministic.
    for group, track_id in group_to_track.items():
        candidates: list[RDSGeometry] = list(canonical_by_group.get(group, []))
        candidates.extend(cell["geometry"] for cell in merged_cells.get(group, []))
        if candidates:
            clip_geometry[track_id] = max(candidates, key=_coverage_m)


def _merge_detections(detections: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine one track's fragments within a single frame into one detection dict.

    Args:
        detections: The fragments assigned to this (track, frame) cell; at least one.

    Returns:
        The sole detection unchanged when there is only one, otherwise a copy of the first carrying a
        :class:`MultiPolyline` geometry and merged attributes.
    """
    if len(detections) == 1:
        return detections[0]
    merged = dict(detections[0])
    merged["geometry"] = MultiPolyline([d["geometry"] for d in detections])
    attributes = _merge_attributes(detections)
    if attributes is not None:
        merged["attributes"] = attributes
    return merged


def _merge_attributes(detections: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Merge fragment attributes so per-vertex lists stay aligned with the merged vertex order.

    RDS-HQ polyline attributes (``colors``, ``styles``) are per-vertex lists, so concatenating them in fragment
    order keeps them aligned with ``MultiPolyline.vertices_world``. A key missing from any fragment would break
    that alignment, so such a key falls back to the first fragment's value rather than producing a list that
    silently mismatches the vertices.

    Args:
        detections: The fragments being merged, in the same order as the ``MultiPolyline`` members.

    Returns:
        The merged attributes, or ``None`` when no fragment carries any.
    """
    if not any("attributes" in d for d in detections):
        return None
    per_fragment: list[dict[str, Any]] = []
    for detection in detections:
        attributes = detection.get("attributes")
        per_fragment.append(attributes if isinstance(attributes, dict) else {})
    merged: dict[str, Any] = {}
    for key in per_fragment[0]:
        values = [attrs.get(key) for attrs in per_fragment]
        lists = [value for value in values if isinstance(value, list)]
        if len(lists) == len(values):
            merged[key] = [item for value in lists for item in value]
        else:
            merged[key] = values[0]
    return merged


def _build_cost_matrix(
    type_dets: list[tuple[int, dict[str, Any]]],
    type_tracks: list[str],
    active: dict[str, dict[str, Any]],
) -> np.ndarray:
    """
    Build the (n_dets, n_tracks) cost matrix for one (frame, object_type)
    cell. Pairs above the detection's gate are marked as ``+inf`` so the
    caller can filter spurious assignments produced by Hungarian padding.
    """
    n_dets = len(type_dets)
    n_tracks = len(type_tracks)
    cost = np.full((n_dets, n_tracks), np.inf, dtype=np.float64)
    for i, (_, det) in enumerate(type_dets):
        det_geom = det["geometry"]
        gate = det_geom.MATCH_GATE_METERS
        for j, tid in enumerate(type_tracks):
            d = det_geom.world_distance(active[tid]["geometry"])
            if d <= gate:
                cost[i, j] = d
    return cost
