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

from collections.abc import Sequence
import itertools
import logging
from typing import Any, cast

import numpy as np

from checks.utils.polyline_grouping import label_connected_paths

logger = logging.getLogger(__name__)

# Monotonic counter for hallucination track IDs (module-level)
_TRACK_ID_COUNTER = 0


def track_hallucinations(
    results: dict[str, Any],
    start_indices: dict[str, int],
    road_mask: np.ndarray | None = None,
    proximity_by_class: dict[str, float] | None = None,
) -> None:
    """
    Associate newly appended hallucination detections from the current frame with existing tracks using a simple
    bounding-box overlap heuristic, or start new tracks when no overlap is found.

    When ``proximity_by_class`` maps a class to a positive pixel distance, association is loosened for that class: a
    detection also joins a track if its bbox center is within that distance of one of the track's last <=2 bboxes,
    even without overlap. This links a fast-moving hallucination whose frame-to-frame boxes no longer overlap (at the
    10fps world-model rate a nearby object can move tens of px per frame). ``proximity_by_class=None`` / a class absent
    from it / a value of 0 preserves the original overlap-only behavior.

    What this function updates
    - Mutates `results["hallucination_tracks"]` in-place by either appending a new track or extending the `detections`
      list of an existing track. If the tracks array is missing, it will be created.

    Inputs
    - results:
        A dict that must contain:
          - `hallucination_detections`: dict[str -> list[dict]] per high-level class
            (e.g., "vehicle", "pedestrian"). Each detection is expected to have at
            least `bbox_xywh: [x, y, w, h]` and `frame_idx`.
          - `hallucination_tracks`: list[track] (optional). Each track is a dict
            `{ "id": int, "class": str, "detections": list[int], "relevancy": int }`.
            The `detections` list stores integer indices into
            `results["hallucination_detections"][track["class"]]`.
    - start_indices:
        dict[class_name -> int] giving, for each class, the starting index (i.e., the length of the class list before
        appending current-frame detections). All detections added in the current frame for a class are assumed to be
        appended contiguously; this function will iterate from `start_indices[class]` to the current end of the list.
    - road_mask:
        Optional boolean numpy array (HxW) indicating road pixels. If provided, used to update `relevancy` for tracks
        whose detections touch road regions. If None, relevancy updates are skipped.

    Association logic
    - For each new detection of a class, compare its bbox against the last up to two
      bboxes of every existing track of the same class (looked up by the stored
      indices in that track). If there is any strict rectangle overlap (intersection
      area > 0), the detection is assigned to that track; otherwise a new track is
      created with `id = len(results["hallucination_tracks"]) + 1` and its `detections`
      initialized with the new detection index.

    Relevancy update
    - For each assignment (new or existing), increment the track's `relevancy` by 1 if the detection bbox region
      contains any pixel labeled as "road" in the provided `road_mask`.

    Assumptions & limitations
    - The `hallucination_detections[class]` lists are append-only, and this function is called exactly once per
      processed frame after appending the current frame's detections.
    - Overlap threshold is binary (> 0 area). There is no IoU gating, motion model, track lifecycle management
      (age/timeouts), or cross-class linking.
    - Indices stored in tracks remain valid for the lifetime of the results object.

    Returns
    - None. All updates are applied directly to `results`.
    """
    if not start_indices:
        return

    # Prepare tracks container
    tracks = results.get("hallucination_tracks")
    if tracks is None:
        tracks = []
        results["hallucination_tracks"] = tracks

    # Ensure module-level counter starts above any pre-existing track IDs
    global _TRACK_ID_COUNTER
    if _TRACK_ID_COUNTER <= 0 and tracks:
        try:
            _TRACK_ID_COUNTER = max(int(t.get("id", 0)) for t in tracks)
        except Exception as exc:  # noqa: BLE001 - malformed track IDs fall back to a reset counter
            logger.debug("Could not compute max track ID; resetting counter: %s", exc)
            _TRACK_ID_COUNTER = 0

    def overlaps(b1: Sequence[float], b2: Sequence[float]) -> bool:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        r1 = (x1, y1, x1 + max(w1, 1) - 1, y1 + max(h1, 1) - 1)
        r2 = (x2, y2, x2 + max(w2, 1) - 1, y2 + max(h2, 1) - 1)
        ix0 = max(r1[0], r2[0])
        iy0 = max(r1[1], r2[1])
        ix1 = min(r1[2], r2[2])
        iy1 = min(r1[3], r2[3])
        return (ix1 >= ix0) and (iy1 >= iy0)

    def within_proximity(b1: Sequence[float], b2: Sequence[float], max_dist: float) -> bool:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        c1x, c1y = x1 + w1 / 2.0, y1 + h1 / 2.0
        c2x, c2y = x2 + w2 / 2.0, y2 + h2 / 2.0
        return ((c1x - c2x) ** 2 + (c1y - c2y) ** 2) <= max_dist * max_dist

    def get_detection_bucket(cls: str) -> list[Any]:
        return (results.get("hallucination_detections", {}) or {}).get(cls, [])

    def get_detection_bbox(cls: str, idx: int) -> Any:
        try:
            det = get_detection_bucket(cls)[idx]
            return det.get("bbox_xywh", None)
        except Exception as exc:  # noqa: BLE001 - missing/invalid detection bbox returns None
            logger.debug("Could not retrieve bbox for %s[%d]: %s", cls, idx, exc)
            return None

    # For each class with new detections in this frame
    for cls_name, base_idx in (start_indices or {}).items():
        bucket = get_detection_bucket(cls_name)
        if not isinstance(bucket, list) or base_idx >= len(bucket):
            continue

        prox = float((proximity_by_class or {}).get(cls_name, 0.0))

        for det_idx in range(int(base_idx), len(bucket)):
            det = bucket[det_idx]
            det_bbox = det.get("bbox_xywh")
            if det_bbox is None:
                continue

            matched_track = None
            for tr in tracks:
                if tr.get("class") != cls_name:
                    continue
                det_indices = tr.get("detections") or []
                for prev_idx in det_indices[-2:][::-1]:
                    prev_bbox = get_detection_bbox(cls_name, int(prev_idx))
                    if prev_bbox is None:
                        continue
                    if overlaps(det_bbox, prev_bbox) or (prox > 0.0 and within_proximity(det_bbox, prev_bbox, prox)):
                        matched_track = tr
                        break
                if matched_track is not None:
                    break

            if matched_track is None:
                _TRACK_ID_COUNTER += 1
                new_id = int(_TRACK_ID_COUNTER)
                track = {
                    "id": new_id,
                    "class": str(cls_name),
                    "detections": [int(det_idx)],
                    "relevancy": 0,
                }
                tracks.append(track)
                target_track = track
            else:
                matched_track.setdefault("detections", []).append(int(det_idx))
                target_track = matched_track

            # Update relevancy if bbox touches road
            try:
                if road_mask is not None:
                    x, y, w, h = [int(v) for v in det_bbox]
                    x1 = max(0, min(x + max(w, 1) - 1, road_mask.shape[1] - 1))
                    y1 = max(0, min(y + max(h, 1) - 1, road_mask.shape[0] - 1))
                    x0 = max(0, min(x, road_mask.shape[1] - 1))
                    y0 = max(0, min(y, road_mask.shape[0] - 1))
                    if x0 <= x1 and y0 <= y1:
                        roi = road_mask[y0 : y1 + 1, x0 : x1 + 1]
                        if roi.any():
                            target_track["relevancy"] = int(cast(int, target_track.get("relevancy", 0)) + 1)
            except Exception as exc:  # noqa: BLE001 - best-effort road-relevancy update
                logger.debug("Ignoring error in road relevancy update: %s", exc)


def _group_by_ground_connectivity(
    indexed: list[tuple[int, dict[str, Any]]],
    resolution_m: float,
    thickness_px: int,
) -> dict[int, list[int]]:
    """``{group id: [detection index, ...]}`` for detections carrying a ground position.

    Delegates to :func:`checks.utils.polyline_grouping.label_connected_paths`, which is the same construction
    already used to collapse RDS-HQ annotation fragments into whole lines — fragments of one hallucinated line
    touch on the ground plane for exactly the reason fragments of one annotated line do. Reusing it also
    inherits its grid-cell budget and its first-appearance id numbering, so a clip spanning kilometers cannot
    allocate an unbounded grid and track ids do not depend on OpenCV's internal labeling.

    Args:
        indexed: ``(detection index, detection)`` pairs, each carrying a non-empty ``ground_xy``.
        resolution_m: Grid resolution, meters per cell.
        thickness_px: Stroke width, which is how the gaps between fragments of one line are bridged.

    Returns:
        Group id to the detection indices assigned to it.
    """
    group_ids = label_connected_paths(
        [np.asarray(detection["ground_xy"], dtype=float) for _, detection in indexed],
        resolution_m=resolution_m,
        thickness_px=thickness_px,
    )
    groups: dict[int, list[int]] = {}
    for (det_idx, _), group_id in zip(indexed, group_ids):
        groups.setdefault(int(group_id), []).append(int(det_idx))
    return groups


def regroup_ground_tracks(
    results: dict[str, Any],
    classes: Sequence[str],
    resolution_m: float = 0.1,
    thickness_px: int = 7,
) -> int:
    """Rebuild the tracks of ``classes`` by grouping their detections in world space. Returns tracks written.

    The alternative to the per-frame bounding-box association in :func:`track_hallucinations`, for classes whose
    detections carry ``ground_xy`` — the detection's centerline placed on the ground plane. Those classes are
    the ones the bbox tracker serves worst: a deleted lane line breaks into many small fragments per frame
    (dashes, occlusions, segmentation gaps), and their boxes shift across the image as the ego drives, so
    overlap against the previous frame links almost nothing. Measured on one clip: 184 detections became 85
    tracks, 62 of them a single frame long.

    In the world those fragments do not move, because the thing they belong to does not. Drawing every
    detection's ground polyline into one shared bird's-eye raster and taking 8-connected components therefore
    collects the fragments of one physical line — across dashes *and* across frames — into one group. It is the
    same construction :func:`checks.utils.polyline_grouping.label_connected_polylines` uses to collapse RDS-HQ
    laneline fragments into whole lines, applied to detections instead of annotations. ``thickness_px`` sets how
    wide a gap is bridged; lanes sit ~3 m apart, so a few tenths of a meter joins dashes without joining lanes.

    Unlike the bbox tracker this cannot run per frame: a component is only settled once every frame has
    contributed, so ids are assigned in one pass at the end of the clip. Detection indices are untouched, so
    they stay valid for anything already holding them.

    Args:
        results: Results container; ``hallucination_detections`` is read and ``hallucination_tracks`` rewritten
            for ``classes`` only. Tracks of every other class are preserved as they are.
        classes: RDS class labels to regroup (e.g. ``["LaneLine", "RoadBoundary"]``).
        resolution_m: Bird's-eye grid resolution, meters per cell.
        thickness_px: Stroke width each detection is drawn with, which is how the gaps between fragments of
            one line are bridged. Lanes sit ~3 m apart, so a stroke a few tenths of a meter wide joins dashes
            without joining lanes.

    Returns:
        Number of tracks written for ``classes``.
    """
    wanted = {str(c) for c in classes}
    if not wanted:
        return 0
    detections_by_class = results.get("hallucination_detections") or {}
    existing = [t for t in (results.get("hallucination_tracks") or []) if t.get("class") not in wanted]

    rebuilt: list[dict[str, Any]] = []
    for class_name in sorted(wanted):
        bucket = detections_by_class.get(class_name) or []
        indexed = [(i, d) for i, d in enumerate(bucket) if d.get("ground_xy")]
        groups = _group_by_ground_connectivity(indexed, resolution_m, thickness_px) if indexed else {}

        global _TRACK_ID_COUNTER
        for group in sorted(groups):
            _TRACK_ID_COUNTER += 1
            rebuilt.append(
                {
                    "id": int(_TRACK_ID_COUNTER),
                    "class": class_name,
                    "detections": sorted(groups[group]),
                    "relevancy": 0,
                }
            )

        # Every detection of a regrouped class must end up in exactly one track. A detection can reach here
        # ungrouped either because back-projection produced no ground position — which the detector treats as
        # losing the world position, not the detection — or because its cells all landed on background. This
        # pass replaced the class's previous tracks wholesale, so without a singleton here those detections
        # would belong to no track at all, and would vanish from any consumer that iterates tracks.
        grouped = {index for members in groups.values() for index in members}
        for det_idx in range(len(bucket)):
            if det_idx in grouped:
                continue
            _TRACK_ID_COUNTER += 1
            rebuilt.append(
                {
                    "id": int(_TRACK_ID_COUNTER),
                    "class": class_name,
                    "detections": [det_idx],
                    "relevancy": 0,
                }
            )

    results["hallucination_tracks"] = existing + rebuilt
    return len(rebuilt)


def _max_consecutive_frames(frame_idxs: Sequence[int]) -> int:
    """Longest run of consecutive frame indices in a (possibly unsorted, duplicated) sequence. 0 if empty.

    Frames are the world-model stream (contiguous integers at 10fps), so a run of length N means the blob was present
    N processed frames in a row (N/10 s).
    """
    nums = sorted({int(f) for f in frame_idxs})
    if not nums:
        return 0
    best = run = 1
    for prev, cur in itertools.pairwise(nums):
        run = run + 1 if cur == prev + 1 else 1
        best = max(best, run)
    return best


def confirm_hallucination_tracks(
    results: dict[str, Any],
    min_frames_by_class: dict[str, int] | None,
    total_frame_classes: Sequence[str] = (),
) -> int:
    """Drop hallucination tracks that don't persist long enough, per class. Returns the number of tracks removed.

    A track is confirmed only if its detections span at least ``min_frames_by_class[class]`` *consecutive* world-model
    frames. Classes absent from the map, or with a threshold <= 1, keep every track (the original behavior) — so this
    is a no-op unless a class is explicitly configured with ``min_track_frames >= 2``. Single-frame detections, which
    at 10fps are usually segmentation flicker rather than a persistent hallucination, are removed for such classes.

    ``total_frame_classes`` counts *distinct* frames instead of the longest consecutive run. A consecutive run is
    the right question for an image-space track, which exists only while its boxes keep overlapping, but the
    wrong one for a world-space track from :func:`regroup_ground_tracks`: that track is a physical thing, and a
    thing can pass behind a vehicle for a few frames and come back still being the same thing. Requiring an
    unbroken run would split exactly the tracks the world-space grouping exists to hold together.

    ``results['hallucination_tracks']`` is filtered in place; ``results['hallucination_detections']`` (the index
    targets) is left untouched so detection indices stored in surviving tracks stay valid.
    """
    if not min_frames_by_class:
        return 0
    tracks = results.get("hallucination_tracks") or []
    detections = results.get("hallucination_detections") or {}
    kept: list[dict[str, Any]] = []
    removed = 0
    for tr in tracks:
        cls = tr.get("class")
        min_frames = int(min_frames_by_class.get(cls, 1)) if cls is not None else 1
        if min_frames <= 1:
            kept.append(tr)
            continue
        bucket = detections.get(cls) or []
        frame_idxs = []
        for det_idx in tr.get("detections") or []:
            try:
                frame_idxs.append(int(bucket[int(det_idx)]["frame_idx"]))
            except (IndexError, KeyError, TypeError, ValueError):
                continue
        span = len(set(frame_idxs)) if cls in set(total_frame_classes) else _max_consecutive_frames(frame_idxs)
        if span >= min_frames:
            kept.append(tr)
        else:
            removed += 1
    if removed:
        results["hallucination_tracks"] = kept
    return removed
