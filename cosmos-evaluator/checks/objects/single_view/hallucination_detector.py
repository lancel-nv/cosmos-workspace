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
Hallucination detection for objects using segmentation masks.

Detects per-frame hallucinated instances: clusters of class-labeled pixels that do not align with any projected GT
object of the same class.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from third_party.cosmos_drive_dreams_toolkits.utils.camera.base import CameraBase

from checks.utils.ground_plane import backproject_to_ground
from checks.utils.object_class_names import class_name_to_rds_label
from checks.utils.rds_data_loader import RdsDataLoader
from checks.utils.scene_rasterizer import SceneRasterizer

# Classes whose GT support is the exact pixels of the intersecting segmentation components, rather than the
# axis-aligned box enclosing them. A box is a fair stand-in for a compact object's extent and fills the gaps
# between, say, a traffic light's separate bulb blobs — but for anything long or diagonal it covers a region far
# larger than the object, and because the per-class masks are OR'd into one shared mask that over-coverage
# suppresses detections of every other class inside it.
_COMPONENT_UNION_CLASSES: frozenset[str] = frozenset({"vehicle", "lane_line", "road_boundary"})

# Classes exempt from the ego-body filter. That filter drops any cluster touching the bottom of the frame to
# reject the ego's own hood, which is the wrong question for ground-plane geometry: a lane line runs from under
# the bumper to the vanishing point, so its nearest — and best resolved — portion is always in the band, and a
# wait line a few meters ahead may lie entirely inside it. Safe to skip because the static path scores the CWIP
# EXTRA mask, and the hood appears in both the camera and world videos, so it is labeled MATCH and never
# reaches the unmatched pool at all. ``wait_line`` is listed for when it is enabled; it has no class entry yet.
_EGO_BODY_FILTER_EXEMPT_CLASSES: frozenset[str] = frozenset({"lane_line", "road_boundary", "wait_line"})

# Classes whose detections carry an image-space centerline (``spine_px``) alongside the bounding box. A box is a
# poor description of a long diagonal shape, and the overlay draws the centerline instead when it is present.
_CENTERLINE_CLASSES: frozenset[str] = frozenset({"lane_line", "road_boundary", "wait_line"})

# Cap on centerline vertices per detection, so the results JSON stays a readable size on a clip that emits
# hundreds of fragments.
_MAX_CENTERLINE_POINTS: int = 16

# Range beyond which a ground-plane back-projection is not trusted. Toward the horizon a pixel of vertical
# error is meters of range, so the far field smears along the viewing ray; this is the same window the polyline
# geometry itself is rasterized over.
_GROUND_MAX_RANGE_M: float = 60.0


@dataclass(frozen=True)
class _Component:
    """One connected component of the unmatched mask, with the label that locates its pixels."""

    label: int
    x: int
    y: int
    width: int
    height: int
    area: int


def component_centerline_px(component_mask: np.ndarray, max_points: int = _MAX_CENTERLINE_POINTS) -> list[list[int]]:
    """Ordered image-space centerline of a thin component, as ``[[x, y], ...]``.

    Scans along whichever image axis the component spans further and takes the median of the other coordinate on
    each scanline. That needs no global axis, so unlike a PCA fit or an oriented box it follows a curve exactly,
    which matters because a lane line is only straight in image space when the road is: a real one bends, and an
    f-theta lens bends it further. A road marking receding from the camera is monotonic in image row, so one x
    per row is well defined; a wait line crossing the view is the transpose, which picking the longer axis
    handles. A shape that doubles back in the chosen axis — a boundary hooking through an intersection — has its
    two branches averaged into a midpoint, which degrades the path rather than corrupting it.

    Taking the median rather than the mean keeps a stray pixel from dragging the line off the mark, and the
    scan axis is picked from a percentile span for the same reason.

    Args:
        component_mask: Boolean mask of a single component.
        max_points: Cap on returned vertices; the polyline is subsampled evenly to fit.

    Returns:
        Vertices ordered along the component, or ``[]`` for an empty mask.
    """
    ys, xs = np.nonzero(component_mask)
    if ys.size == 0:
        return []

    # Choose the scan axis from a percentile span rather than the bounding box: the box is decided by its two
    # most extreme pixels, so a single stray one can flip a near-square component onto the wrong axis and
    # transpose the whole path.
    row_span = float(np.subtract(*np.percentile(ys, [95, 5])))
    col_span = float(np.subtract(*np.percentile(xs, [95, 5])))
    row_major = row_span >= col_span
    keys, values = (ys, xs) if row_major else (xs, ys)
    order = np.argsort(keys, kind="stable")
    keys, values = keys[order], values[order]

    # One median per scanline. ``np.unique`` gives the group boundaries in the already-sorted keys.
    unique_keys, starts = np.unique(keys, return_index=True)
    ends = np.append(starts[1:], len(values))
    medians = [int(np.median(values[start:end])) for start, end in zip(starts, ends)]

    if len(unique_keys) > max_points:
        picks = np.linspace(0, len(unique_keys) - 1, max_points).astype(int)
    else:
        picks = np.arange(len(unique_keys))

    if row_major:
        return [[medians[i], int(unique_keys[i])] for i in picks]
    return [[int(unique_keys[i]), medians[i]] for i in picks]


class _SegHelperLike(Protocol):
    """Structural type for the ``seg_helper`` duck contract this detector consumes.

    The dynamic processor passes ``SegHelper`` (``checks.objects.single_view.seg_helper``) and the
    static processor passes ``CWIPInferenceHelper`` (``checks.objects.single_view.cwip_helper``); the
    two classes share no nominal ancestor but both expose ``get_class_mask(masks, class_name) ->
    np.ndarray`` — the only method this detector calls on ``seg_helper`` (see lines below). A
    Protocol is the type-safe match here without forcing the two helpers into an inheritance graph.
    """

    def get_class_mask(self, masks: Any, class_name: str) -> np.ndarray | None: ...


class HallucinationDetector:
    """
    Detect hallucinated objects by subtracting refined GT support from class masks and clustering the remaining
    unmatched pixels.

    Overview
    --------
    1) Build a per-class GT support mask by intersecting projected 3D cuboids with connected components in the
       corresponding segmentation class mask.
    2) Combine per-class GT masks (OR) into a global support mask.
    3) For each class, compute unmatched regions as (class_mask AND NOT global_GT).
    4) Cluster unmatched regions into hallucination detections.

    Class-specific GT policy:
    - ``_COMPONENT_UNION_CLASSES``: for each projected object, add every intersecting component's exact pixels.
    - every other class: for each projected object, collect all intersecting components and fill the tightest
      axis-aligned rectangle that encloses them (plus a margin) into the GT.

    Note the GT support mask is built only from the *configured* classes, not from every annotated object in the
    frame, so a pixel of a real object whose class is not enabled here has nothing to vouch for it and can be
    reported as a hallucination of whichever class the segmentation assigned it.
    """

    def __init__(
        self,
        config: dict[str, Any],
        seg_helper: _SegHelperLike,
        logger: logging.Logger,
        mask_debug_dir: Path | None = None,
    ) -> None:
        """
        Args:
            config: Dict with keys {enabled, classes (map), max_cluster_per_frame}
            seg_helper: SegHelper instance for class mask extraction
            logger: Logger for diagnostics
            mask_debug_dir: When set, ``detect`` persists the per-frame intermediate masks (the shared GT support
                mask plus, per class, the raw class mask and the unmatched mask) to
                ``<mask_debug_dir>/frame_<idx>.npz``. Off by default (``None``) so production runs pay nothing;
                intended only for offline recall/misclassification analysis of hallucination false negatives.
        """
        self.config = config or {}
        self.seg_helper = seg_helper
        self.logger = logger
        self.mask_debug_dir = mask_debug_dir
        if self.mask_debug_dir:
            self.mask_debug_dir.mkdir(parents=True, exist_ok=True)

        # External config (strict) — all fields must be provided by caller
        if "enabled" not in self.config:
            raise KeyError("hallucination_detector.enabled is required in config")
        if "classes" not in self.config:
            raise KeyError("hallucination_detector.classes is required in config")
        if "max_cluster_per_frame" not in self.config:
            raise KeyError("hallucination_detector.max_cluster_per_frame is required in config")

        self.enabled: bool = bool(self.config["enabled"])
        # classes: mapping class_name -> {min_cluster_area: int}
        classes_val = self.config["classes"]
        if not isinstance(classes_val, dict):
            raise TypeError("hallucination_detector.classes must be a dict mapping class -> config")
        self.classes_cfg = dict(classes_val)
        self.max_clusters_per_frame: int = int(self.config["max_cluster_per_frame"])

    def detect(
        self,
        frame_idx: int,
        resized_masks: Any,
        frame_objects: dict[str, Any],
        camera_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
        scene_rasterizer: SceneRasterizer | None = None,
        min_cluster_area: Mapping[str, int] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Run hallucination detection for a single frame.

        Steps per frame:
        - Build a global GT support mask by OR-ing per-class GT masks
        - For each configured class:
          * Extract the class mask from segmentation
          * Compute unmatched = class_mask AND NOT global_GT
          * Cluster unmatched into components with a per-class min-area threshold
          * Emit detections (bbox and a ratio metric)

        Args:
            frame_idx: Frame index
            resized_masks: Segmentation masks resized to target resolution
            frame_objects: Objects in the current frame
            camera_pose: Camera pose for this frame
            camera_model: Camera model for projection
            image_width: Image width
            image_height: Image height
            scene_rasterizer: Optional pre-computed SceneRasterizer with visibility masks.
                              If provided, uses cached masks for efficient O(1) lookup.
                              If None, computes masks directly.
            min_cluster_area: Optional per-class minimum cluster area (pixels), keyed by class name, that takes
                precedence over the configured ``classes.<name>.min_cluster_area`` for this call only. The configured
                values are left unchanged.

        Returns:
            Dict[str, List[Dict[str, Any]]]: class_name -> list of detections with keys:
                - frame_idx: Frame index of the detection
                - bbox_xywh: Bounding box [x, y, w, h] of the unmatched cluster
                - mask_ratio: Ratio metric for the cluster (implementation-specific)
        """
        detections: dict[str, list[dict[str, Any]]] = {cls: [] for cls in self.classes_cfg}

        if not self.enabled:
            return {class_name_to_rds_label(cls): dets for cls, dets in detections.items()}

        # Build a global refined GT support across all classes once
        gt_mask = np.zeros((image_height, image_width), dtype=bool)
        for class_name in self.classes_cfg:
            # Build refined GT mask for this class
            class_gt_mask = self._build_gt_mask(
                resized_masks=resized_masks,
                frame_objects=frame_objects,
                camera_pose=camera_pose,
                camera_model=camera_model,
                image_width=image_width,
                image_height=image_height,
                class_name=class_name,
                scene_rasterizer=scene_rasterizer,
            )
            if class_gt_mask is not None:
                gt_mask |= class_gt_mask

        # Per-class (class_mask, unmatched_mask) captured only when mask debugging is on (see _dump_debug_masks).
        debug_class_masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for class_name in self.classes_cfg:
            try:
                class_mask = self.seg_helper.get_class_mask(resized_masks, class_name)
            except Exception as exc:  # noqa: BLE001 - unknown/invalid class is skipped
                self.logger.warning("Skipping unknown class %s in hallucination detection: %s", class_name, exc)
                continue

            if class_mask is None:  # get_class_mask returns None when the class has no mask this frame -> nothing to do
                continue

            if class_mask.dtype != bool:
                class_mask = class_mask.astype(bool)

            # Compute unmatched regions: pixels labeled as this class but not supported by GT
            unmatched_mask = class_mask & (~gt_mask)

            # Capture the raw class mask + unmatched mask before any clustering/area/ego filtering, so offline
            # analysis can tell "segmentation never saw this class" from "saw it but GT-subtracted / filtered".
            if self.mask_debug_dir is not None:
                debug_class_masks[class_name] = (class_mask, unmatched_mask)

            if not unmatched_mask.any():
                continue

            # unmatched_mask is already boolean; avoid redundant conversion and checks
            unmatched = unmatched_mask

            # Cluster unmatched regions with a per-class minimum area threshold
            if min_cluster_area is not None and class_name in min_cluster_area:
                min_area = int(min_cluster_area[class_name])
            else:
                min_area = int(self._get_min_cluster_area_for_class(class_name))
            comps, labels = self._cluster_components(unmatched, min_area=min_area)
            if not comps or labels is None:
                continue

            # Cap clusters per frame per class
            comps = comps[: self.max_clusters_per_frame]
            wants_centerline = class_name in _CENTERLINE_CLASSES

            for comp in comps:
                # Ego-body filter: skip clusters intersecting bottom band (see _EGO_BODY_FILTER_EXEMPT_CLASSES).
                if class_name not in _EGO_BODY_FILTER_EXEMPT_CLASSES and self._is_in_ego_body_region(
                    comp.x, comp.y, comp.width, comp.height, image_width, image_height
                ):
                    continue
                detection: dict[str, Any] = {
                    "frame_idx": int(frame_idx),
                    "bbox_xywh": [comp.x, comp.y, comp.width, comp.height],
                    "mask_ratio": float(comp.area / max(comp.width * comp.height, 1)),
                }
                if wants_centerline:
                    # A bounding box says very little about a long diagonal shape; the centerline is what the
                    # overlay draws for these classes and what a reader can compare against a projected line.
                    spine_px = component_centerline_px(labels == comp.label)
                    detection["spine_px"] = spine_px
                    # The same centerline placed on the ground plane. This is the frame the BEV tracker groups
                    # in: a static hallucination does not move in the world, however far the ego has driven.
                    if spine_px:
                        try:
                            ground = backproject_to_ground(
                                np.asarray(spine_px, dtype=float), camera_pose, camera_model, _GROUND_MAX_RANGE_M
                            )
                        except Exception as exc:  # noqa: BLE001 - a camera that cannot unproject loses the
                            # world-space position, not the detection itself.
                            self.logger.debug("Ground back-projection failed for %s: %s", class_name, exc)
                        else:
                            detection["ground_xy"] = [[round(float(px), 2), round(float(py), 2)] for px, py in ground]
                detections[class_name].append(detection)

        if self.mask_debug_dir is not None:
            self._dump_debug_masks(int(frame_idx), image_height, image_width, gt_mask, debug_class_masks)

        return {class_name_to_rds_label(cls): dets for cls, dets in detections.items()}

    def _dump_debug_masks(
        self,
        frame_idx: int,
        image_height: int,
        image_width: int,
        gt_mask: np.ndarray,
        class_masks: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Persist a frame's intermediate hallucination masks for offline recall/misclassification analysis.

        Writes ``<mask_debug_dir>/frame_<idx>.npz`` holding the shared GT support mask and, per class, the raw
        class mask and the unmatched (``class AND NOT GT``) mask. Class keys use the same RDS labels as
        ``detect()``'s returned ``hallucination_detections`` (via ``class_name_to_rds_label``), so a single
        class name matches both these masks and ``static.results.json``. Boolean masks are bit-packed with
        ``np.packbits`` to keep artifacts small; unpack with
        ``np.unpackbits(a)[: h * w].reshape(h, w).astype(bool)`` using the stored ``shape``. A class absent from
        the archive had no segmented pixels that frame. Best-effort: a write failure is logged, never fatal.
        """
        if not self.mask_debug_dir:
            return
        try:
            labels = {name: class_name_to_rds_label(name) for name in class_masks}
            arrays: dict[str, np.ndarray] = {
                "shape": np.asarray([image_height, image_width], dtype=np.int32),
                "classes": np.asarray(list(labels.values())),
                "gt": np.packbits(gt_mask.ravel()),
            }
            for name, (class_mask, unmatched_mask) in class_masks.items():
                label = labels[name]
                arrays[f"cls__{label}"] = np.packbits(class_mask.ravel())
                arrays[f"unm__{label}"] = np.packbits(unmatched_mask.ravel())
            # allow_pickle=False keeps these dumps to plain packed arrays (no pickled objects, safe to load) and, by
            # pinning the bool kwarg explicitly, lets the type checker resolve the remaining ``**arrays`` as archive
            # entries rather than a candidate value for ``allow_pickle``.
            np.savez_compressed(
                self.mask_debug_dir / f"frame_{frame_idx:04d}.npz",
                allow_pickle=False,
                **arrays,
            )
        except Exception as exc:  # noqa: BLE001 - debug dump is best-effort; never abort detection
            self.logger.warning("Failed to dump hallucination debug masks for frame %d: %s", frame_idx, exc)

    def _build_gt_mask(
        self,
        resized_masks: Any,
        frame_objects: dict[str, Any],
        camera_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
        class_name: str,
        scene_rasterizer: SceneRasterizer | None = None,
    ) -> np.ndarray:
        """
        Build a refined GT mask for a single class using segmentation components and 3D projections.

        Algorithm:
        - Obtain the binary class mask from the resized segmentation output.
        - Compute connected components on the class mask once (labels, stats).
        - For each ground-truth object of this class:
            * Project its cuboid to an image-space polygon mask (using SceneRasterizer if available).
            * Intersect the projection with the component labels to find intersecting labels.
            * If none intersect, the cuboid contributes nothing (no fallback).
            * If class == "vehicle": pick the single largest component by area and OR it.
            * Else (non-vehicle): compute a tight AABB enclosing all intersecting components and fill it.
        - Return the per-class union of supports.

        Args:
            resized_masks: Segmentation masks
            frame_objects: Objects in the current frame
            camera_pose: Camera pose
            camera_model: Camera model
            image_width: Image width
            image_height: Image height
            class_name: Class name to build GT mask for
            scene_rasterizer: Optional pre-computed SceneRasterizer for efficient mask lookup
        """
        if resized_masks is None:
            return np.zeros((image_height, image_width), dtype=bool)

        # Get full-image class mask for this high-level class
        try:
            class_mask = self.seg_helper.get_class_mask(resized_masks, class_name)
        except Exception as exc:  # noqa: BLE001 - missing class mask yields an empty GT mask
            self.logger.warning("Failed to get class mask for %s in GT mask build: %s", class_name, exc)
            return np.zeros((image_height, image_width), dtype=bool)
        if class_mask is None:  # get_class_mask returns None when this class has no mask in the frame -> empty GT
            return np.zeros((image_height, image_width), dtype=bool)
        if class_mask.dtype != bool:
            class_mask = class_mask.astype(bool)

        # Connected components on the class mask (label 0 is background)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask.astype(np.uint8), connectivity=8)

        # Objects of this class
        obj_map = RdsDataLoader.get_objects_by_class(frame_objects, class_name)
        if not obj_map:
            return np.zeros((image_height, image_width), dtype=bool)

        # Accumulate GT support for this class here
        class_union = np.zeros((image_height, image_width), dtype=bool)

        for track_id, tracked_object in obj_map.items():
            # Project cuboid to image and rasterize visible faces to a binary mask
            # Use SceneRasterizer if available for occlusion-aware visibility
            try:
                track_id_str = str(track_id)
                if scene_rasterizer is not None and scene_rasterizer.has_object(track_id_str):
                    proj = scene_rasterizer.get_visibility_mask(track_id_str)
                else:
                    geometry = tracked_object["geometry"]
                    proj, _ = geometry.get_projected_mask(camera_pose, camera_model, image_width, image_height)
            except Exception as exc:  # noqa: BLE001 - cuboid projection failure skips the object
                self.logger.warning("Failed to project cuboid for track %s: %s", track_id, exc)
                continue
            if proj is None or proj.sum() == 0:
                continue

            # Find labels (>0) whose pixels intersect the projected cuboid mask
            seed_labels = np.unique(labels[proj.astype(bool)])
            seed_labels = seed_labels[seed_labels > 0]
            if seed_labels.size == 0:
                # No intersecting clusters; skip (no fallback)
                continue

            if class_name in _COMPONENT_UNION_CLASSES:
                # Exactly the intersecting components, with no enclosing box (see _COMPONENT_UNION_CLASSES).
                comp_full = np.isin(labels, seed_labels)  # boolean mask of all selected components
                class_union |= comp_full
            else:
                # Enclose all intersecting components with a tight axis-aligned rectangle (optionally expanded)
                xs = []
                ys = []
                xes = []
                yes = []
                margin = 16
                for lab in seed_labels:
                    lab = int(lab)
                    x = int(stats[lab, cv2.CC_STAT_LEFT])
                    y = int(stats[lab, cv2.CC_STAT_TOP])
                    w = int(stats[lab, cv2.CC_STAT_WIDTH])
                    h = int(stats[lab, cv2.CC_STAT_HEIGHT])
                    xs.append(x)
                    ys.append(y)
                    xes.append(x + max(w, 1) - 1)
                    yes.append(y + max(h, 1) - 1)
                x_min = max(0, min(xs) - margin)
                y_min = max(0, min(ys) - margin)
                x_max = min(image_width - 1, max(xes) + margin)
                y_max = min(image_height - 1, max(yes) + margin)
                if x_min <= x_max and y_min <= y_max:
                    class_union[y_min : y_max + 1, x_min : x_max + 1] = True

        return class_union

    def _cluster_components(self, mask: np.ndarray | None, min_area: int) -> tuple[list[_Component], np.ndarray | None]:
        """
        Cluster a binary mask into connected components.

        Args:
            mask: Binary mask to cluster
            min_area: Minimum area (in pixels) to keep a component

        Returns:
            ``(components, labels)`` where ``components`` are :class:`_Component` records sorted by area
            descending and ``labels`` is the per-pixel label image the caller uses to recover each component's
            pixels (``labels == component.label``). ``labels`` is ``None`` for an empty or absent mask.
        """
        if mask is None or not mask.any():
            return [], None
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        comps: list[_Component] = []
        for lab in range(1, num):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            comps.append(
                _Component(
                    label=lab,
                    x=int(stats[lab, cv2.CC_STAT_LEFT]),
                    y=int(stats[lab, cv2.CC_STAT_TOP]),
                    width=int(stats[lab, cv2.CC_STAT_WIDTH]),
                    height=int(stats[lab, cv2.CC_STAT_HEIGHT]),
                    area=area,
                )
            )
        # Sort largest-first to keep strongest clusters when capping
        comps.sort(key=lambda c: c.area, reverse=True)
        return comps, labels

    def _get_min_cluster_area_for_class(self, class_name: str) -> int:
        """
        Fetch per-class minimum cluster area threshold from configuration.

        Args:
            class_name: High-level class name

        Returns:
            Pixel area threshold (int); defaults to 100 if unspecified/invalid.
        """
        cfg = self.classes_cfg.get(class_name, {}) or {}
        try:
            return int(cfg.get("min_cluster_area", 100))
        except (TypeError, ValueError):
            return 100

    def _is_in_ego_body_region(self, x: int, y: int, w: int, h: int, image_width: int, image_height: int) -> bool:
        """
        Returns True if any part of the bbox intersects the bottom N pixels of the frame.

        Args:
            x, y, w, h: Bounding box in XYWH (image pixel coordinates)
            image_height: Image height in pixels
        """
        bottom_band = int(0.25 * image_height)
        y0 = y
        y1 = y + max(h, 1) - 1
        band_y0 = max(0, image_height - bottom_band)
        # Intersection test between [y0, y1] and [band_y0, image_height - 1]
        return (y1 >= band_y0) and (y0 <= image_height - 1)
