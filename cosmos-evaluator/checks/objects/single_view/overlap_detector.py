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
Overlap detection for objects.

This module owns the per-class processing pipeline that determines how well tracked 3D objects correspond to
segmentation results in each frame.

Responsibilities
----------------
1) Importance filtering: decide if an object should be evaluated (distance, lanes, etc.).
2) Occlusion detection: optionally skip objects considered occluded (currently not implemented)
3) Scoring: compute a per-object correspondence score via one of three methods:
   - "ratio"  : overlap_pixels / projected_cuboid_pixels (identical to legacy implementation).
   - "cluster": find all segmentation connected components intersecting the projected cuboid
                and compute inside / total_cluster_area. This reduces over-scoring caused by
                large components that spill far outside the cuboid.
   - "spine"  : for polylines only — the fraction of the projected, undilated centerline that
                carries the class label. Area metrics normalize by the ribbon that
                ``Polyline.half_width_meters`` invents (0.4 m against ~0.1-0.15 m of real paint),
                so they measure that half-width as much as the world model; the centerline has no
                such free parameter.

Notes
-----
- The class-to-SegFormer label mapping and object-type mapping are centralized in SegHelper.
- For the cluster method, connected components are computed once per class per frame and reused for all objects to
  avoid redundant work.
"""

from contextlib import suppress
import logging
from typing import Any

import cv2
import numpy as np
from third_party.cosmos_drive_dreams_toolkits.utils.camera.base import CameraBase

from checks.objects.single_view.importance_filter import ImportanceFilter
from checks.utils.rds_data_loader import RdsDataLoader
from checks.utils.scene_rasterizer import SceneRasterizer


class OverlapDetector:
    def __init__(
        self,
        oc_config: dict[str, Any],
        seg_model: Any,
        logger: logging.Logger,
        debug_enabled: bool = False,
    ) -> None:
        """Initialize the overlap detector and supporting components.

        Args:
            oc_config: Objects configuration subtree (includes overlap_check, filters, etc.)
            seg_model: SegHelper or CWIPInference instance for mask extraction
            logger: Logger for diagnostics
            debug_enabled: If True, collect extra per-frame debug metadata
        """
        self.seg_model = seg_model
        self.logger = logger
        self.debug_enabled = debug_enabled
        # Store objects config
        self.oc_config = oc_config or {}
        # Build filters from objects config
        self.importance_filter = ImportanceFilter(self.oc_config.get("importance_filter", {}), logger)

    def process_class(
        self,
        class_name: str,
        resized_masks: Any,
        frame_objects: dict[str, Any],
        camera_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
        track_output_agg: dict[int, dict[str, Any]],
        frame_idx: int,
        scene_rasterizer: SceneRasterizer | None = None,
        ego_pose: np.ndarray | None = None,
        is_static: bool = False,
    ) -> tuple[dict[str, float], dict[str, dict[str, Any]], dict[str, Any] | None]:
        """Process a single class for one frame: filter, score, and (optionally) debug.

        Workflow:
            1) Read scoring method for class from config (ratio|cluster)
            2) Extract class mask from segmentation
            3) If cluster method, compute connected components once on class mask
            4) Iterate class objects → importance filter → occlusion filter → score
            5) Aggregate scores, filtered reasons, and optional debug bboxes

        Args:
            class_name: High-level class (vehicle, pedestrian, ...)
            resized_masks: Segmentation masks (resized to target)
            frame_objects: Mapping of track_id → object dict for current frame
            camera_pose: 4x4 camera-to-world pose
            camera_model: Camera model used for projection
            image_width: Target image width
            image_height: Target image height
            track_output_agg: Accumulator for per-track output labels/stats
            frame_idx: Current frame index (object timeline)
            scene_rasterizer: Optional pre-computed SceneRasterizer with visibility masks.
                              If provided, uses cached masks for efficient O(1) lookup.
                              If None, computes masks directly.
            is_static: True when the caller is the static processor. Static tracks still
                       compute and record their importance attributes (so each object with a
                       centroid logs its GT ego-distance for distance-based report breakdowns);
                       only the ``is_important`` filtering and filtered-count logging stay
                       dynamic-only (static objects are always important).

        Returns:
            scores: track_id → score
            filtered_objects: track_id → {reason, type}
            debug_info: Optional visualization info if debug_enabled
        """
        scores: dict[str, float] = {}
        filtered_objects: dict[str, dict[str, Any]] = {}
        debug_info: dict[str, Any] | None = {"frame_boxes": [], "object_categories": {}} if self.debug_enabled else None
        importance_filtered_count = 0
        occlusion_filtered_count = 0
        importance_filter_reasons: dict[str, int] = {}

        # Determine method and per-class thresholds from config. ``min_projection_area`` (pixels²) gates per-frame
        # scoring for static types; ``bottom_occlusion_abstain_frac`` abstains (leaves NaN) on dynamic frames where
        # the cuboid is bottom-occluded beyond that fraction of its projected height; ``min_in_frame_fraction``
        # abstains on frames where the frame edge truncates the projected cuboid. ``spine_match_tolerance_px``
        # and ``spine_min_ribbon_width_px`` apply to the spine method only. Absence (or 0) disables each.
        method = "ratio"
        min_projection_area = 0
        bottom_occlusion_abstain_frac = 0.0
        min_in_frame_fraction = 0.0
        spine_match_tolerance_px = 0
        spine_min_ribbon_width_px = 0.0
        try:
            class_cfg = None
            try:
                overlap_cfg = self.oc_config.get("overlap_check", {}) or {}
                class_cfg = overlap_cfg.get(class_name, {})
            except Exception as exc:  # noqa: BLE001 - missing/invalid config falls back to defaults
                self.logger.debug("Could not read overlap_check config for %s: %s", class_name, exc)
                class_cfg = {}
            if isinstance(class_cfg, dict):
                method = str(class_cfg.get("method", method)).lower()
                with suppress(Exception):
                    min_projection_area = int(class_cfg.get("min_projection_area", 0) or 0)
                with suppress(TypeError, ValueError):
                    bottom_occlusion_abstain_frac = float(class_cfg.get("bottom_occlusion_abstain_frac", 0.0) or 0.0)
                with suppress(TypeError, ValueError):
                    min_in_frame_fraction = float(class_cfg.get("min_in_frame_fraction", 0.0) or 0.0)
                with suppress(TypeError, ValueError):
                    spine_match_tolerance_px = int(class_cfg.get("spine_match_tolerance_px", 0) or 0)
                with suppress(TypeError, ValueError):
                    spine_min_ribbon_width_px = float(class_cfg.get("spine_min_ribbon_width_px", 0.0) or 0.0)
        except Exception as exc:  # noqa: BLE001 - scoring-method resolution falls back to ratio
            self.logger.debug("Could not determine scoring method for %s, defaulting to ratio: %s", class_name, exc)
            method = "ratio"
            min_projection_area = 0
            bottom_occlusion_abstain_frac = 0.0
            min_in_frame_fraction = 0.0
            spine_match_tolerance_px = 0
            spine_min_ribbon_width_px = 0.0

        # Extract binary segmentation mask for the high-level class from SegHelper.
        class_mask = self.seg_model.get_class_mask(resized_masks, class_name)

        # The spine method may match against a dilated copy of the class mask. The segmentation mask is produced at
        # patch resolution and upsampled with nearest-neighbor, so its edges are quantized to a few pixels no
        # matter how precise the underlying prediction is; a strictly one-pixel-wide centerline would be scored
        # against that quantization rather than against the world model. Dilating by a fixed *image-space* radius
        # is the right correction because the error is constant in pixels, while the polyline's world-space
        # half-width shrinks with range. 0 (the default) matches the centerline exactly.
        scoring_mask = class_mask
        if method == "spine" and class_mask is not None and spine_match_tolerance_px > 0:
            scoring_mask = self._dilate_mask(class_mask, spine_match_tolerance_px)

        # If scoring method is cluster, precompute connected components once on the full class mask to obtain
        # per-pixel labels and per-component stats (bbox, area, centroid). This is reused for every object of this
        # class in the current frame.
        labels = None
        stats = None
        if method == "cluster" and class_mask is not None:
            try:
                _, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask.astype(np.uint8), connectivity=8)
            except Exception as exc:  # noqa: BLE001 - connected-components failure disables cluster method
                self.logger.debug("Connected components failed for %s, disabling cluster method: %s", class_name, exc)
                labels, stats = None, None

        # Objects for this class
        class_objects = RdsDataLoader.get_objects_by_class(frame_objects, class_name)
        total_objects = len(class_objects)
        if total_objects == 0:
            return scores, filtered_objects, debug_info

        for track_id, tracked_object in class_objects.items():
            scores[track_id] = np.nan

            # Per-frame projection-area gate (static types only; threshold==0 disables). Skip scoring frames where
            # the visible projection is too small to score reliably — those frames stay NaN and are excluded from
            # per-track means. The processor builds the per-frame projection_area_matrix directly from the
            # rasterizer (using its own local frame index), so no per-frame projection metadata is recorded here.
            # Class-agnostic (kept out of the ``not is_static`` block — static thresholds still depend on it) and
            # left before the importance block so its existing semantics are unchanged.
            if min_projection_area > 0:
                track_id_str = str(track_id)
                visible_area = 0
                if scene_rasterizer is not None and scene_rasterizer.has_object(track_id_str):
                    visible_area = int(scene_rasterizer.get_projection_metadata(track_id_str).get("visible_area_px", 0))
                if visible_area < min_projection_area:
                    continue

            # Compute importance attributes (ego-frame based) for EVERY track — static included — so each object with
            # a physical centroid records its GT ego-distance (consumed by distance-based report breakdowns). Static
            # objects are always important; the ``is_important`` filtering / filtered-count logging below stays
            # dynamic-only. Runs before the in-frame gate so a gated dynamic track still records its attributes.
            importance_pose = ego_pose if ego_pose is not None else camera_pose
            importance_attrs = self.importance_filter.get_importance_attributes(
                tracked_object, importance_pose, track_id
            )

            # Record importance attributes for per-track aggregation (use int key
            # to match _aggregate_output which also uses int(track_id))
            t_id = int(track_id)
            if t_id not in track_output_agg:
                track_output_agg[t_id] = {"object_types": set(), "output_counts": {}}
            track_output_agg[t_id].setdefault("importance_attrs", []).append(importance_attrs)

            if not is_static and not importance_attrs["is_important"]:
                # Still record as filtered for logging, but do NOT skip scoring
                importance_filtered_count += 1
                distance_threshold = self.importance_filter.config.get("distance_threshold_m", float("inf"))
                distance_m = importance_attrs["distance_m"]
                filter_reason = (
                    f"distance_threshold_m ({self.importance_filter.config.get('distance_threshold_m')}m)"
                    if distance_m is not None and distance_m > distance_threshold
                    else "skip_oncoming_obstacles (enabled)"
                    if importance_attrs["is_oncoming"]
                    else f"relevant_lanes ({self.importance_filter.config.get('relevant_lanes', [])})"
                )
                with suppress(Exception):
                    importance_filter_reasons[str(filter_reason)] = (
                        int(importance_filter_reasons.get(str(filter_reason), 0)) + 1
                    )

            # Per-frame in-frame gate (frame-edge truncation), for static and dynamic classes alike. A GT object
            # whose projected bbox is partly outside the frame has too little in-frame signal to score reliably:
            # segmentation cannot match a frame-edge sliver, and a truncated traffic light exposes only some of
            # its bulbs. Scoring either as ~0 would create false "missing" verdicts downstream, so skip such
            # frames (leave score NaN, exactly like outside-FOV). Purely geometric — occlusion by other objects
            # is deliberately NOT considered here (``bottom_occlusion_abstain_frac`` below covers that), because
            # the question is only whether the cuboid crosses the image edge. Threshold comes from
            # ``overlap_check.<class>.min_in_frame_fraction``; 0/absent disables the gate for the class. Strict
            # ``<`` so an object at exactly the threshold is still scored. Only cuboids can be truncated —
            # polylines and surfaces inherit ``RDSGeometry.in_frame_fraction`` == 1.0 and are never gated.
            if min_in_frame_fraction > 0.0 and (
                self._in_frame_fraction(
                    track_id, tracked_object, scene_rasterizer, camera_pose, camera_model, image_width, image_height
                )
                < min_in_frame_fraction
            ):
                # Mandatory aggregate (not just importance): populates object_types / object_type_index and the
                # edge_truncated count, so a track gated in every frame still carries the correct object_type
                # and output counts.
                self._aggregate_output(track_output_agg, track_id, tracked_object, "edge_truncated")
                continue

            # Per-frame top-only-occlusion gate for dynamic objects. When a nearer object occludes the bottom
            # of this cuboid and only its top pokes out (bottom_occluded_frac >= threshold), the visible sliver
            # is too thin to score reliably; scoring it ~0 would create a false "missing" verdict downstream.
            # Abstain on this frame (leave score NaN, exactly like outside-FOV) using the occlusion-aware masks
            # the rasterizer already computed. Requires the rasterizer (occlusion is cuboid-vs-cuboid); when it
            # is absent the object is scored normally. Disabled when the threshold is 0.
            if (
                not is_static
                and bottom_occlusion_abstain_frac > 0.0
                and scene_rasterizer is not None
                and scene_rasterizer.has_object(str(track_id))
                and scene_rasterizer.bottom_occluded_frac(str(track_id)) >= bottom_occlusion_abstain_frac
            ):
                self._aggregate_output(track_output_agg, track_id, tracked_object, "occluded")
                occlusion_filtered_count += 1
                continue

            # Spine scoring needs the object's projected centerline. Compute it once per object per frame so the
            # length gate below and the score itself share the work, mirroring how the cluster method reuses one
            # connected-components pass. ``None`` means this geometry has no centerline (a cuboid or surface in a
            # spine-configured class), which ``_score_object`` handles by falling back to the ratio metric.
            visible_spine: np.ndarray | None = None
            if method == "spine":
                visible_spine = self._visible_spine_mask(
                    track_id,
                    tracked_object,
                    scene_rasterizer,
                    camera_pose,
                    camera_model,
                    image_width,
                    image_height,
                    spine_min_ribbon_width_px,
                )

            # Score each object using the selected method. For cluster, pass the precomputed CC; for spine, the
            # centerline computed above.
            # The dilated mask is the spine metric's own correction for mask quantization, so it must not leak
            # into the ratio fallback an object without a centerline takes: that object would then be scored
            # against a mask no ``method: ratio`` class ever sees, and its score would move with a knob that is
            # documented as spine-only.
            object_mask = scoring_mask if visible_spine is not None else class_mask
            score = self._score_object(
                track_id,
                tracked_object,
                object_mask,
                camera_pose,
                camera_model,
                image_width,
                image_height,
                method,
                labels,
                stats,
                scene_rasterizer,
                visible_spine,
            )
            if score is not None:
                scores[track_id] = score
                self._aggregate_output(track_output_agg, track_id, tracked_object, "scored")
            else:
                # Object not scoreable (no projection, occluded, etc.) — still record in tracks
                self._aggregate_output(track_output_agg, track_id, tracked_object, "occluded")

                # Debug bbox
                if self.debug_enabled and debug_info is not None:
                    try:
                        # Get projected mask from SceneRasterizer if available
                        track_id_str = str(track_id)
                        if scene_rasterizer is not None and scene_rasterizer.has_object(track_id_str):
                            projected_mask = scene_rasterizer.get_visibility_mask(track_id_str)
                        else:
                            geometry = tracked_object["geometry"]
                            projected_mask, _ = geometry.get_projected_mask(
                                camera_pose, camera_model, image_width, image_height
                            )
                        ys, xs = np.where(projected_mask)
                        if xs.size > 0 and ys.size > 0:
                            x_min = int(xs.min())
                            y_min = int(ys.min())
                            x_max = int(xs.max()) + 1
                            y_max = int(ys.max()) + 1
                            x0 = max(0, min(x_min, image_width))
                            y0 = max(0, min(y_min, image_height))
                            x1 = max(0, min(x_max, image_width))
                            y1 = max(0, min(y_max, image_height))
                            w = x1 - x0
                            h = y1 - y0
                            if w > 0 and h > 0:
                                debug_info["object_categories"].setdefault(int(track_id), str(class_name))
                                debug_info["frame_boxes"].append(
                                    {"object_id": int(track_id), "bbox_xywh": [int(x0), int(y0), int(w), int(h)]}
                                )
                    except Exception as exc:  # noqa: BLE001 - best-effort debug bbox build
                        self.logger.debug("Ignoring error building debug bbox for track %s: %s", track_id, exc)

        # Logging summary for this class
        try:
            processed_count = sum(1 for s in scores.values() if not np.isnan(s))
            self.logger.info("Frame %s class=%s filtering summary:", frame_idx, class_name)
            self.logger.info("  - Total objects: %s", total_objects)
            self.logger.info("  - Importance filtered: %s", importance_filtered_count)
            if importance_filter_reasons:
                for reason, count in importance_filter_reasons.items():
                    self.logger.info("    * %s: %s", reason, count)
            self.logger.info("  - Occlusion filtered: %s", occlusion_filtered_count)
            self.logger.info("  - Processed: %s", processed_count)
            if total_objects > 0:
                self.logger.info(
                    "  - Filtering rates: %.1f%% importance, %.1f%% occlusion",
                    importance_filtered_count / total_objects * 100,
                    occlusion_filtered_count / total_objects * 100,
                )
        except Exception as exc:  # noqa: BLE001 - best-effort summary logging
            self.logger.debug("Ignoring error in filtering summary log: %s", exc)

        return scores, filtered_objects, debug_info

    @staticmethod
    def _in_frame_fraction(
        track_id: Any,
        tracked_object: dict[str, Any],
        scene_rasterizer: SceneRasterizer | None,
        camera_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
    ) -> float:
        """Fraction of the object's projected bounding box lying inside the image, ignoring occlusion.

        Prefers the rasterizer, which memoizes the value per object per frame and shares it with the
        traffic-light state check so the cuboid is projected once rather than once per consumer. Falls back
        to projecting the geometry directly when ``process_class`` is called without a rasterizer.

        Args:
            track_id: Track identifier of the object.
            tracked_object: Object dict; its ``geometry`` is used for the fallback path.
            scene_rasterizer: Rasterizer for this frame, or None.
            camera_pose: 4x4 camera-to-world pose (fallback path only).
            camera_model: Camera model used for projection (fallback path only).
            image_width: Target image width.
            image_height: Target image height.

        Returns:
            Fraction in [0, 1]; ``1.0`` for objects carrying no geometry, so they are never gated.
        """
        if scene_rasterizer is not None and scene_rasterizer.has_object(str(track_id)):
            return scene_rasterizer.in_frame_fraction(str(track_id))
        geometry = tracked_object.get("geometry")
        if geometry is None:
            return 1.0
        return float(geometry.in_frame_fraction(camera_pose, camera_model, image_width, image_height))

    @staticmethod
    def _dilate_mask(mask: np.ndarray, radius_px: int) -> np.ndarray:
        """Dilate a binary mask by ``radius_px`` using a disk-shaped structuring element.

        A disk rather than a square so the tolerance is the same in every direction — a polyline crosses the
        image at arbitrary angles, and a square element would grant a diagonal centerline ~1.4x the intended
        slack.

        Args:
            mask: Binary mask to grow.
            radius_px: Dilation radius in pixels; values <= 0 return the mask unchanged.

        Returns:
            The dilated mask as a boolean array (a copy; the input is not modified).
        """
        if radius_px <= 0:
            return mask.astype(bool)
        size = 2 * int(radius_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

    @staticmethod
    def _visible_spine_mask(
        track_id: Any,
        tracked_object: dict[str, Any],
        scene_rasterizer: SceneRasterizer | None,
        camera_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
        min_ribbon_width_px: float = 0.0,
    ) -> np.ndarray | None:
        """Occlusion-aware mask of the object's projected centerline, or ``None`` when it has no centerline.

        Prefers the rasterizer, which hides the centerline where a nearer object covers it and memoizes the
        result. Falls back to projecting the geometry directly when ``process_class`` is called without a
        rasterizer; that fallback is **not** occlusion-aware, matching how :meth:`_score_object` falls back for
        the area metrics.

        Args:
            track_id: Track identifier of the object.
            tracked_object: Object dict; its ``geometry`` is used for the fallback path.
            scene_rasterizer: Rasterizer for this frame, or None.
            camera_pose: 4x4 camera-to-world pose (fallback path only).
            camera_model: Camera model used for projection (fallback path only).
            image_width: Target image width.
            image_height: Target image height.
            min_ribbon_width_px: Drop traced centerline samples narrower than this many pixels.

        Returns:
            Boolean centerline mask, or ``None`` for geometry without a centerline (cuboids, surfaces) and for
            objects carrying no geometry.
        """
        if scene_rasterizer is not None and scene_rasterizer.has_object(str(track_id)):
            return scene_rasterizer.get_visible_spine_mask(str(track_id), min_ribbon_width_px)
        geometry = tracked_object.get("geometry")
        if geometry is None:
            return None
        spine = geometry.get_projected_spine_mask(
            camera_pose, camera_model, image_width, image_height, min_ribbon_width_px
        )
        return None if spine is None else spine[0]

    def _score_object(
        self,
        track_id: Any,
        tracked_object: dict[str, Any],
        class_mask: np.ndarray,
        camera_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
        scoring_method: str,
        labels: np.ndarray | None = None,
        stats: np.ndarray | None = None,
        scene_rasterizer: SceneRasterizer | None = None,
        visible_spine_mask: np.ndarray | None = None,
    ) -> float | None:
        """Score one object using the configured method.

        Args:
            track_id: Track ID of the object
            tracked_object: Object dict with at least object_to_world and object_lwh
            class_mask: Binary mask the object is matched against (the class mask, dilated for the spine method)
            camera_pose: 4x4 camera-to-world pose
            camera_model: Camera model used for projection
            image_width: Image width for projection
            image_height: Image height for projection
            scoring_method: "ratio", "cluster" or "spine"
            labels: Connected component labels for class_mask (if cluster)
            stats: Connected component stats for class_mask (if cluster)
            scene_rasterizer: Optional pre-computed SceneRasterizer for efficient mask lookup
            visible_spine_mask: Occlusion-aware centerline mask (if spine); ``None`` when the geometry has no
                centerline, which falls back to the ratio metric

        Returns:
            Score in [0,1] or None if not scorable (e.g., empty projection)
        """
        try:
            method = scoring_method.lower() if isinstance(scoring_method, str) else "ratio"

            # Spine scoring never looks at the projected area, so it is resolved before the ribbon mask is
            # fetched. That matters beyond saving work: a heavily foreshortened polyline (a wait line, seen
            # nearly edge-on) can rasterize to an empty ribbon while its centerline still traces a usable run of
            # pixels, and the area-based early-out below would abandon exactly those frames.
            if method == "spine" and visible_spine_mask is not None:
                if not visible_spine_mask.any():
                    return None
                return float(self._score_spine(class_mask, visible_spine_mask))
            if method == "spine":
                self.logger.debug(
                    "Track %s has no centerline geometry; scoring with the ratio metric instead of spine", track_id
                )
                method = "ratio"

            # Get projected mask from SceneRasterizer if available, otherwise compute directly
            track_id_str = str(track_id)
            if scene_rasterizer is not None and scene_rasterizer.has_object(track_id_str):
                projected_mask = scene_rasterizer.get_visibility_mask(track_id_str)
            else:
                geometry = tracked_object["geometry"]
                projected_mask, _ = geometry.get_projected_mask(camera_pose, camera_model, image_width, image_height)
            if projected_mask.sum() == 0:
                return None
            if method == "ratio":
                # Legacy ratio metric: overlap / cuboid_area.
                return float(self._score_ratio(class_mask, projected_mask))
            elif method == "cluster":
                # New cluster metric: inside / total_area_of_intersecting_components.
                return float(self._score_cluster(class_mask, projected_mask, labels, stats))
            else:
                self.logger.warning("Invalid scoring method: %s", scoring_method)
                return None
        except Exception as e:  # noqa: BLE001 - per-object scoring failure returns None
            self.logger.warning("Error processing object: %s", e)
            return None

    @staticmethod
    def _score_ratio(class_mask: np.ndarray, projected_cuboid_mask: np.ndarray) -> float:
        """
        Ratio-based score: overlap pixels divided by projected cuboid area.

        This implementation is kept identical to the previous behavior to preserve historical results.
        """
        if projected_cuboid_mask is None or projected_cuboid_mask.sum() == 0:
            return 0.0
        masked = (class_mask.astype(bool) if class_mask is not None else False) & projected_cuboid_mask.astype(bool)
        num_overlap = int(np.count_nonzero(masked))
        denom = int(np.count_nonzero(projected_cuboid_mask))
        if denom <= 0:
            return 0.0
        return float(num_overlap / denom)

    @staticmethod
    def _score_spine(class_mask: np.ndarray | None, visible_spine_mask: np.ndarray) -> float:
        """
        Spine-based score: the fraction of visible centerline pixels the segmentation labels with this class.

        Rationale
        ---------
        ``_score_ratio`` normalizes by the projected ribbon, whose width comes from ``Polyline.half_width_m``
        (0.4 m by default) rather than from the world. Real lane paint is roughly 0.10-0.15 m wide, so even a
        perfectly reproduced, perfectly registered line can only cover a fraction of its own ribbon and the score
        is bounded well below 1 — it reports the half-width as much as the world model. ``_score_cluster`` has
        the opposite failure: near the vanishing point the class mask's connected components merge across
        parallel lines, inflating the denominator by geometry belonging to other tracks.

        Measuring along the centerline removes the free parameter. The denominator is the visible centerline
        (occlusion already subtracted by the rasterizer), the numerator is the part of it the segmentation
        agrees with, and both are properties of where the line actually is.

        Args:
            class_mask: Binary class mask to match against, already dilated by the configured tolerance.
            visible_spine_mask: Occlusion-aware boolean mask of the projected centerline.

        Returns:
            Score in [0, 1]; 0.0 when there is no class mask or no visible centerline.
        """
        total = int(np.count_nonzero(visible_spine_mask))
        if total <= 0 or class_mask is None:
            return 0.0
        matched = int(np.count_nonzero(class_mask.astype(bool) & visible_spine_mask))
        return float(matched / total)

    @staticmethod
    def _score_cluster(
        class_mask: np.ndarray,
        projected_cuboid_mask: np.ndarray,
        labels: np.ndarray | None,
        stats: np.ndarray | None,
    ) -> float:
        """
        Cluster-based score: For components intersecting the cuboid, score = inside / total_cluster_area.

        Rationale
        ---------
        The previous cluster metric was biased by large components that extend outside the cuboid. By normalizing the
        overlap pixels by the true area of the intersecting components (from CC), the score better reflects the
        fraction of relevant segment(s) actually covered by the cuboid.

        Algorithm
        ---------
        - CC is computed once per class mask upstream; this function reuses labels and stats.
        - Identify all labels (>0) that intersect the projected cuboid (via labels[proj]).
        - total_cluster_area = sum(stats[label, AREA]) across these labels (not double-counting overlaps).
        - inside = count of pixels where (labels in those labels) AND projected cuboid.
        - Score = inside / total_cluster_area (0.0 if no intersecting labels or zero area).
        """
        if (
            class_mask is None
            or projected_cuboid_mask is None
            or not projected_cuboid_mask.any()
            or labels is None
            or stats is None
        ):
            return 0.0

        proj_bool = projected_cuboid_mask.astype(bool)
        # Intersecting labels (>0): labels that appear within the projected cuboid region
        try:
            seed_labels = np.unique(labels[proj_bool])
            seed_labels = seed_labels[seed_labels > 0]
        except Exception:  # noqa: BLE001 - degenerate label data scores 0.0
            return 0.0
        if seed_labels.size == 0:
            return 0.0

        # Build mask for union of intersecting components and count pixels inside the cuboid
        union_mask = np.isin(labels, seed_labels)
        inside = int(np.count_nonzero(union_mask & proj_bool))
        # Sum areas from stats to avoid recounting due to overlaps between components
        try:
            total_area = int(sum(int(stats[int(lab), cv2.CC_STAT_AREA]) for lab in seed_labels))
        except Exception:  # noqa: BLE001 - malformed stats fall back to union pixel count
            # Fallback to pixel count of union if stats are malformed
            total_area = int(np.count_nonzero(union_mask))

        if total_area <= 0:
            return 0.0
        return float(inside / total_area)

    def _aggregate_output(
        self,
        track_output_agg: dict[int, dict[str, Any]],
        track_id: int | str,
        tracked_object: dict[str, Any],
        label: str,
    ) -> None:
        """Accumulate per-track metadata for final reporting.

        Increments the count for an output label (e.g., "scored", "occluded", filter reason) and tracks encountered
        object types per track id.
        """
        try:
            t_id = int(track_id)
            entry = track_output_agg.get(t_id)
            if entry is None:
                entry = {"object_types": set(), "output_counts": {}}
                track_output_agg[t_id] = entry
            obj_type = tracked_object.get("object_type")
            if obj_type is not None:
                entry["object_types"].add(str(obj_type))
            obj_type_index = tracked_object.get("object_type_index")
            if obj_type_index is not None:
                entry["object_type_index"] = int(obj_type_index)
            # Carry a synthesized object's stable identity (stamped as the ``synthetic_source_id`` attribute by
            # whoever injected it) through to the result track
            attributes = tracked_object.get("attributes")
            if isinstance(attributes, dict) and attributes.get("synthetic_source_id") is not None:
                entry["synthetic_source_id"] = str(attributes["synthetic_source_id"])
            entry["output_counts"][str(label)] = int(entry["output_counts"].get(str(label), 0)) + 1
        except Exception as e:  # noqa: BLE001 - best-effort per-track aggregation
            self.logger.warning("Failed to aggregate output for track %s: %s", track_id, e)
