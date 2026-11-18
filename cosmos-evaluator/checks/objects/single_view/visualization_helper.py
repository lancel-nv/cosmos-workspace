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
Visualization helper module for object processing.

This module handles the creation of visualization videos and images with segmentation overlays and object tracking
displays.
"""

from contextlib import suppress
import logging
from typing import Any

import cv2
import numpy as np
from third_party.cosmos_drive_dreams_toolkits.utils.camera.base import CameraBase
import torch

from checks.utils.overlay_video_helper import OverlayVideoHelper
from checks.utils.rds_data_loader import RdsDataLoader
from checks.utils.scene_rasterizer import SceneRasterizer
from checks.utils.video import VideoWriter, get_video_encode_args


def hallucination_overlay_label(class_name: str) -> str:
    """Compact overlay tag for a hallucination class, e.g. ``H:TL`` (TrafficLight), ``H:TS`` (TrafficSign).

    The hallucinations dict is keyed by RDS labels (CamelCase, e.g. ``TrafficLight``); older snake_case names are
    also handled. The code is the class's uppercase initials, so any class gets a short, readable tag instead of a
    truncated word.
    """
    caps = [ch for ch in class_name if ch.isupper()]
    if len(caps) >= 2:  # CamelCase RDS label: TrafficLight -> TL, TrafficSign -> TS
        code = "".join(caps)
    elif "_" in class_name:  # snake_case: traffic_light -> TL
        code = "".join(tok[0] for tok in class_name.split("_") if tok).upper()
    else:  # single word: Vehicle -> VE, Pedestrian -> PE
        code = class_name[:2].upper()
    return f"H:{code}"


class VisualizationHelper(OverlayVideoHelper):
    """
    Handles visualization creation for object processing.

    This class manages video writing, frame overlays, and visualization generation with segmentation masks and object
    tracking.
    """

    def __init__(self, clip_id: str, output_dir: str, verbose: bool = False):
        """
        Initialize the visualization helper.

        Args:
            clip_id: Clip ID for file naming
            output_dir: Output directory for videos and images
            verbose: Enable verbose logging
        """
        super().__init__(clip_id, output_dir, verbose=verbose)
        self.logger = logging.getLogger(__name__)

        self.legend_items = [
            # Contour colors for scored objects
            "Green box: Vehicle",
            "Red box: Pedestrian",
            "Magenta box: Motorcycle",
            "Red dot: Distance",
            "Blue circle: Lane",
            "Blue cross: Oncoming",
            "Orange cross: Occluded",
            "Magenta square: Unknown",
            # Static object overlays
            "Yellow box: Traffic light",
            "Orange box: Traffic sign",
            "Cyan strip: Road boundary",
            "Purple strip: Wait line",
            "White strip: Lane line",
            "Pink box: Crosswalk",
        ]

        # Enable debug logging for troubleshooting
        if verbose:
            self.logger.setLevel(logging.DEBUG)

        # Named mask video writers, created on demand via write_mask_frame()
        self._mask_video_writers: dict[str, VideoWriter] = {}
        self._mask_video_paths: dict[str, Any] = {}

        # Unified color map (RGB) for mask rendering
        self.mask_colors = {
            # Movable classes
            "vehicle": (0, 0, 255),  # Blue
            "pedestrian": (255, 0, 0),  # Red
            "motorcycle": (0, 255, 0),  # Green
            "bicycle": (0, 255, 0),  # Green
            # Static classes
            "road": (255, 255, 255),  # White
            "sidewalk": (255, 0, 255),  # Magenta
            "traffic light": (255, 255, 0),  # Yellow
            "traffic sign": (255, 165, 0),  # Orange
            "pole": (128, 0, 128),  # Purple
        }

        # Per-name legend configuration: mask_name -> {"legend_items": ..., "legend_order": ...}
        self._mask_legend_configs: dict[str, dict[str, Any]] = {}

    def create_frame_visualization(
        self,
        frame_idx: int,
        segmentation_masks: Any,
        frame_scores: dict[str, float],
        camera_pose: np.ndarray,
        camera_model: CameraBase,
        frame_objects: dict[str, Any],
        filtered_objects: dict[str, Any],
        hallucinations: dict[str, Any] | None = None,
        scene_rasterizer: SceneRasterizer | None = None,
        traffic_light_state_results: dict[str, Any] | None = None,
    ) -> np.ndarray | None:
        """
        Create visualization for a frame with segmentation overlay and object overlays.

        Args:
            frame_idx: Frame index
            segmentation_masks: Segmentation result
            frame_scores: Dictionary mapping track_id to score
            camera_pose: Camera pose for this frame
            camera_model: Camera model for projection
            frame_objects: Objects in the current frame
            filtered_objects: Objects that were filtered out
            hallucinations: Optional hallucination detections
            scene_rasterizer: Optional pre-computed SceneRasterizer with visibility masks.
                              If provided, uses cached masks for efficient O(1) lookup.
                              If None, visibility masks are computed per object on demand
                              without occlusion caching.
            traffic_light_state_results: Optional mapping from track_id (str) to
                TrafficLightStateResult with traffic_light_state_score.

        Returns:
            Visualization image as numpy array, or None if creation failed
        """
        try:
            # Get the original RGB frame from the video
            base_image = self.get_original_frame(frame_idx)

            # Overlay segmentation masks on the base image
            vis_image = self._overlay_segmentation(base_image, segmentation_masks)

            # Add object overlays with scores
            if frame_objects and frame_scores:
                vis_image = self._add_object_overlays(
                    vis_image,
                    frame_scores,
                    frame_objects,
                    camera_pose,
                    camera_model,
                    filtered_objects,
                    scene_rasterizer=scene_rasterizer,
                )

            # Draw hallucination overlays if provided
            if hallucinations:
                vis_image = self._add_hallucination_overlays(vis_image, hallucinations)

            # Draw traffic light state tags if provided
            if traffic_light_state_results:
                vis_image = self._add_traffic_light_state_tags(
                    vis_image, traffic_light_state_results, frame_objects, camera_pose, camera_model, scene_rasterizer
                )

            # Add frame index to the visualization
            vis_image = self.add_legend(vis_image, frame_idx, self.legend_items)

            return vis_image

        except Exception as e:
            self.logger.error("Error creating frame visualization: %s", e)
            self.logger.debug("Frame visualization failure traceback", exc_info=True)
            return None

    def set_mask_styles(
        self,
        mask_colors: dict[str, tuple[int, int, int]] | None = None,
        legend_order: list[str] | None = None,
        legend_items: list[tuple[str, tuple[int, int, int]]] | None = None,
        mask_name: str = "default",
    ) -> None:
        """Configure mask colors and legend for mask video rendering.

        Args:
            mask_colors: Mapping from class/label name to RGB tuple used for mask coloring.
                         This is a shared attribute (not per-name).
            legend_order: Optional ordering of labels when auto-building the legend from mask_colors.
            legend_items: Optional explicit legend items as (label, RGB) pairs. If provided,
                          this takes precedence over auto-built legend from mask_colors.
            mask_name: Identifier for the mask video this legend config applies to.
        """
        try:
            if mask_colors is not None:
                self.mask_colors = dict(mask_colors)
            config = self._mask_legend_configs.setdefault(mask_name, {})
            config["legend_order"] = list(legend_order) if legend_order else None
            config["legend_items"] = list(legend_items) if legend_items else None
        except (TypeError, ValueError, AttributeError):
            self.logger.exception("Failed to set mask styles")

    def _overlay_segmentation(
        self, base_image: np.ndarray, segmentation_masks: np.ndarray | torch.Tensor
    ) -> np.ndarray:
        """
        Overlay segmentation masks on the base image.

        Args:
            base_image: Original RGB frame
            segmentation result

        Returns:
            Image with segmentation overlay
        """
        # Convert segmentation masks to RGB format
        if isinstance(segmentation_masks, np.ndarray):
            seg_rgb = segmentation_masks
        else:
            seg_rgb = np.transpose(segmentation_masks.numpy(), (1, 2, 0))

        # Ensure segmentation masks match base image dimensions
        if seg_rgb.shape[:2] != base_image.shape[:2]:
            seg_rgb = cv2.resize(seg_rgb, (base_image.shape[1], base_image.shape[0]))

        # Convert segmentation masks to overlay
        seg_overlay = seg_rgb.astype(np.float32) / 255.0

        # Only blend pixels where the segmentation color is non-black (foreground classes).
        # Background / unlabeled pixels are black (0,0,0) in both CWIP (DONT_CARE) and
        # SegFormer (Cityscapes class 0). Blending them unconditionally darkens those regions
        # to (1-alpha) * base, making the background appear dimmer than the original frame.
        alpha = 0.4
        fg_mask = seg_overlay.sum(axis=-1, keepdims=True) > 1e-6  # (H, W, 1) bool
        blended = base_image * (1 - alpha) + seg_overlay * 255 * alpha
        vis_image = np.where(fg_mask, blended, base_image).astype(np.uint8)

        return vis_image

    def _add_hallucination_overlays(self, vis_image: np.ndarray, hallucinations: dict[str, Any]) -> np.ndarray:
        """
        Draw hallucination cluster overlays.

        Args:
            vis_image: Base image (RGB)
            hallucinations: dict class_name -> list of detections with bbox_xywh

        Returns:
            Image with hallucination overlays
        """
        # Use yellow for hallucination bounding boxes
        color = (255, 255, 0)
        thickness = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1

        for class_name, dets in (hallucinations or {}).items():
            if not dets:
                continue
            for det in dets:
                bbox = det.get("bbox_xywh", None)
                if not bbox or len(bbox) != 4:
                    continue
                x, y, w, h = [int(v) for v in bbox]
                x0 = np.clip(x, 0, vis_image.shape[1])
                y0 = np.clip(y, 0, vis_image.shape[0])
                x1 = np.clip(x + w, 0, vis_image.shape[1])
                y1 = np.clip(y + h, 0, vis_image.shape[0])

                # Polyline classes carry a centerline; draw that instead of the box, which for a long diagonal
                # shape covers a large region of unrelated image. Classes without one are unchanged.
                spine = det.get("spine_px")
                if spine and len(spine) >= 2:
                    cv2.polylines(
                        vis_image,
                        [np.asarray(spine, dtype=np.int32)],
                        isClosed=False,
                        color=color,
                        thickness=thickness,
                    )
                else:
                    cv2.rectangle(vis_image, (x0, y0), (x1, y1), color, thickness)

                # Compact label, e.g. H:TL / H:TS (the dict is keyed by RDS labels like "TrafficLight").
                label = hallucination_overlay_label(class_name)

                (tw, th), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
                tx = x0
                ty = y0 - 4
                if ty - th - 2 < 0:
                    ty = y0 + th + 4
                cv2.rectangle(vis_image, (tx, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
                cv2.putText(vis_image, label, (tx + 1, ty), font, font_scale, (255, 255, 255), text_thickness)

        return vis_image

    def _add_traffic_light_state_tags(
        self,
        vis_image: np.ndarray,
        tl_results: dict[str, Any],
        frame_objects: dict[str, Any],
        camera_pose: np.ndarray,
        camera_model: CameraBase,
        scene_rasterizer: SceneRasterizer | None = None,
    ) -> np.ndarray:
        """Draw traffic light state score below each traffic light bbox.

        Args:
            vis_image: Base image (RGB).
            tl_results: Mapping from track_id (str) to an object with
                        traffic_light_state_score (float).
            frame_objects: Objects for the current frame.
            camera_pose: Camera-to-world pose matrix.
            camera_model: Camera intrinsic model.
            scene_rasterizer: Optional SceneRasterizer for visibility masks.

        Returns:
            Image with traffic light state tags drawn.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1

        for track_id_str, tl_result in tl_results.items():
            score = getattr(tl_result, "traffic_light_state_score", None)
            if score is None:
                continue

            confidence = getattr(tl_result, "confidence", None)
            mc_score = getattr(tl_result, "traffic_light_multi_color_score", None)

            # Obtain the bounding box of the projected traffic light
            visibility_mask = None
            if scene_rasterizer is not None and scene_rasterizer.has_object(track_id_str):
                visibility_mask = scene_rasterizer.get_visibility_mask(track_id_str)
            elif track_id_str in frame_objects:
                geometry = frame_objects[track_id_str].get("geometry")
                if geometry is not None:
                    visibility_mask, _ = geometry.get_projected_mask(
                        camera_pose, camera_model, self.frame_width, self.frame_height
                    )

            if visibility_mask is None or not np.any(visibility_mask):
                continue

            ys, xs = np.where(visibility_mask)
            x_min, _ = int(xs.min()), int(xs.max())
            y_max = int(ys.max())

            # Build label lines
            if np.isnan(score):
                state_label = "TLS:n/a"
            else:
                state_label = f"TLS:{score:.2f}"

            mc_label: str | None = f"MC:{mc_score:.2f}" if mc_score is not None and not np.isnan(mc_score) else None

            conf_label: str | None = (
                f"C:{confidence:.2f}" if confidence is not None and not np.isnan(confidence) else None
            )

            tx = x_min
            text_color = (0, 255, 255)
            ty_cursor = y_max + 6  # starting y below the bounding box

            for label in (state_label, mc_label, conf_label):
                if label is None:
                    continue
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                ty = ty_cursor + th
                if ty > vis_image.shape[0] - 4:
                    break
                cv2.rectangle(vis_image, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
                cv2.putText(vis_image, label, (tx, ty), font, font_scale, text_color, thickness)
                ty_cursor = ty + 4

        return vis_image

    def _build_mask_legend(self, vis_image: np.ndarray, frame_idx: int, mask_name: str = "default") -> np.ndarray:
        """Add frame index and class color legend to a mask frame.

        Args:
            vis_image: The mask frame image.
            frame_idx: Current frame index.
            mask_name: Which mask's legend config to use.
        """
        try:
            config = self._mask_legend_configs.get(mask_name, {})
            legend_items_override = config.get("legend_items")

            if legend_items_override:
                return self.add_legend(vis_image, frame_idx, legend_items_override)

            default_order = [
                "vehicle",
                "pedestrian",
                "motorcycle",
                "bicycle",
                "road",
                "sidewalk",
                "traffic light",
                "traffic sign",
                "pole",
            ]
            order = config.get("legend_order") or default_order

            legend_items: list[tuple[str, tuple[int, int, int]]] = []
            seen: set[str] = set()
            for key in order:
                if key in self.mask_colors:
                    legend_items.append((key, self.mask_colors[key]))
                    seen.add(key)
            for key, rgb in self.mask_colors.items():
                if key not in seen:
                    legend_items.append((key, rgb))
            return self.add_legend(vis_image, frame_idx, legend_items)
        except Exception as e:  # noqa: BLE001 - legend overlay is best-effort; return unannotated frame
            self.logger.error("Failed to add mask legend: %s", e)
            return vis_image

    def _add_mask_legend(self, vis_image: np.ndarray, frame_idx: int) -> np.ndarray:
        """Backward-compatible preprocessor for the default mask video."""
        return self._build_mask_legend(vis_image, frame_idx, "default")

    def write_mask_frame(
        self, mask_frame: np.ndarray, frame_idx: int, naming_suffix: str = "", mask_name: str = "default"
    ) -> bool:
        """Write a single mask frame to a named mask video (with legend).

        Args:
            mask_frame: RGB mask image to write.
            frame_idx: Current frame index.
            naming_suffix: Suffix for the output filename (e.g. "static").
            mask_name: Identifier for the mask video. ``"default"`` produces
                       ``{clip}.{suffix}.mask.mp4``; other names produce
                       ``{clip}.{suffix}.{mask_name}_mask.mp4``.
        """
        try:
            if mask_name not in self._mask_video_writers:
                if mask_name == "default":
                    filename = f"{self.clip_id}.{naming_suffix}.mask.mp4"
                else:
                    filename = f"{self.clip_id}.{naming_suffix}.{mask_name}_mask.mp4"
                path = self.output_dir / filename
                self._mask_video_paths[mask_name] = path

                def preprocessor(img: np.ndarray, idx: int, _mn: str = mask_name) -> np.ndarray:
                    return self._build_mask_legend(img, idx, _mn)

                writer = VideoWriter(
                    output_path=path,
                    fps=self.fps,
                    frame_width=self.frame_width,
                    frame_height=self.frame_height,
                    preprocessor=preprocessor,
                    ffmpeg_args=get_video_encode_args(),
                )
                if not writer.open():
                    self.logger.warning("Failed to initialize mask video writer for %s", mask_name)
                    return False
                self._mask_video_writers[mask_name] = writer
            return self._mask_video_writers[mask_name].write_frame(mask_frame, frame_idx)
        except Exception as e:  # noqa: BLE001 - per-frame mask write must not abort the run
            self.logger.error("Error writing mask frame %s (%s): %s", frame_idx, mask_name, e)
            return False

    def release(self) -> None:
        """Release all video resources (overlay + named mask videos)."""
        super().release()
        for writer in self._mask_video_writers.values():
            with suppress(Exception):
                writer.release()

    def _add_object_overlays(
        self,
        vis_image: np.ndarray,
        frame_scores: dict[str, float],
        frame_objects: dict[str, Any],
        camera_pose: np.ndarray,
        camera_model: CameraBase,
        filtered_objects: dict[str, Any],
        scene_rasterizer: SceneRasterizer | None = None,
    ) -> np.ndarray:
        """
        Add object overlays with scores to the visualization image.

        Uses SceneRasterizer to compute occlusion-aware visibility masks, ensuring that object contours reflect only
        the visible portions.

        Args:
            vis_image: Base visualization image
            frame_scores: Dictionary mapping track_id to score
            frame_objects: Objects in the current frame
            camera_pose: Camera pose for this frame
            camera_model: Camera model for projection
            filtered_objects: Objects that were filtered out
            scene_rasterizer: Optional pre-computed SceneRasterizer. If None, visibility masks are computed
                              per object on demand without occlusion caching.

        Returns:
            Image with object overlays
        """

        # Add object overlays with scores for processed objects
        for track_id, score in frame_scores.items():
            if not np.isnan(score) and track_id in frame_objects:
                tracked_object = frame_objects[track_id]

                # Get occlusion-aware visibility mask from scene rasterizer
                track_id_str = str(track_id)
                if scene_rasterizer is not None and scene_rasterizer.has_object(track_id_str):
                    visibility_mask = scene_rasterizer.get_visibility_mask(track_id_str)
                else:
                    # Fallback: compute mask directly if object not in rasterizer
                    geometry = tracked_object.get("geometry")
                    if geometry is None:
                        continue
                    visibility_mask, _ = geometry.get_projected_mask(
                        camera_pose, camera_model, self.frame_width, self.frame_height
                    )

                if np.any(visibility_mask):
                    # Choose contour color by object type
                    obj_type = tracked_object.get("object_type", "")
                    if obj_type in RdsDataLoader.CLASS_TO_OBJECT_TYPES["vehicle"]:
                        contour_color = (0, 255, 0)  # Green (vehicle)
                    elif obj_type in RdsDataLoader.CLASS_TO_OBJECT_TYPES["pedestrian"]:
                        contour_color = (255, 0, 0)  # Red (pedestrian)
                    elif obj_type in RdsDataLoader.CLASS_TO_OBJECT_TYPES["motorcycle"]:
                        contour_color = (255, 0, 255)  # Magenta (motorcycle)
                    elif obj_type in RdsDataLoader.CLASS_TO_OBJECT_TYPES["traffic_light"]:
                        contour_color = (255, 255, 0)  # Yellow (traffic light)
                    elif obj_type in RdsDataLoader.CLASS_TO_OBJECT_TYPES["traffic_sign"]:
                        contour_color = (255, 165, 0)  # Orange (traffic sign)
                    elif obj_type in RdsDataLoader.CLASS_TO_OBJECT_TYPES["road_boundary"]:
                        contour_color = (0, 255, 255)  # Cyan (road boundary)
                    elif obj_type in RdsDataLoader.CLASS_TO_OBJECT_TYPES["wait_line"]:
                        contour_color = (255, 0, 255)  # Purple (wait line)
                    elif obj_type in RdsDataLoader.CLASS_TO_OBJECT_TYPES["lane_line"]:
                        contour_color = (255, 255, 255)  # White (lane line)
                    elif obj_type in RdsDataLoader.CLASS_TO_OBJECT_TYPES["crosswalk"]:
                        contour_color = (255, 192, 203)  # Pink (crosswalk)
                    else:
                        contour_color = (0, 255, 0)  # Default to green

                    vis_image = self._draw_object_overlay(vis_image, visibility_mask, score, track_id, contour_color)

        # Add markers for unprocessed objects (filtered out)
        if filtered_objects:
            for track_id, tracked_object in frame_objects.items():
                if np.isnan(frame_scores.get(track_id, np.nan)):
                    filter_info = filtered_objects.get(track_id, {"reason": "unknown", "type": "unknown"})
                    vis_image = self._draw_filter_marker(
                        vis_image, tracked_object, camera_pose, camera_model, track_id, filter_info
                    )

        return vis_image

    def _draw_object_overlay(
        self,
        vis_image: np.ndarray,
        projected_mask: np.ndarray,
        score: float,
        track_id: int | str,
        contour_color: tuple[int, int, int],
    ) -> np.ndarray:
        """
        Draw object overlay with score on the visualization image.

        Args:
            vis_image: Base visualization image
            projected_mask: Binary mask of projected geometry
            score: Correspondence score
            track_id: Track ID for labeling

        Returns:
            Image with object overlay
        """
        # Find contour of projected mask
        contours, _ = cv2.findContours(projected_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Draw contour overlay using provided color by object type
            cv2.drawContours(vis_image, contours, -1, contour_color, 2)

            # Calculate bounding box of contour for text placement
            x, y, w, h = cv2.boundingRect(contours[0])

            # Add track ID and score text at the top of the box (outside)
            score_text = f"{track_id} ({score:.2f})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # Get text size for background rectangle
            (text_width, text_height), _baseline = cv2.getTextSize(score_text, font, font_scale, thickness)

            # Position text at the top of the bounding box (outside)
            text_x = x + w // 2 - text_width // 2
            text_y = y - 10  # 10 pixels above the top of the box

            # Ensure text doesn't go off the top of the image
            if text_y < text_height + 5:
                text_y = y + h + text_height + 5  # Put below the box instead

            # Draw background rectangle for text
            cv2.rectangle(
                vis_image,
                (text_x - 5, text_y - text_height - 5),
                (text_x + text_width + 5, text_y + 5),
                (255, 255, 255),
                -1,
            )

            # Draw track ID and score text
            cv2.putText(vis_image, score_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

        return vis_image

    def _draw_filter_marker(
        self,
        vis_image: np.ndarray,
        tracked_object: dict[str, Any],
        camera_pose: np.ndarray,
        camera_model: CameraBase,
        track_id: int | str,
        filter_info: dict[str, Any],
    ) -> np.ndarray:
        """
        Draw filter marker for unprocessed objects.

        Args:
            vis_image: Visualization image
            tracked_object: Object data
            camera_pose: Camera pose for this frame
            camera_model: Camera model for projection
            track_id: Track ID for labeling

        Returns:
            Image with filter marker
        """
        # Create cuboid and get projected mask
        geometry = tracked_object["geometry"]

        projected_mask, _ = geometry.get_projected_mask(camera_pose, camera_model, self.frame_width, self.frame_height)

        if np.any(projected_mask):
            # Find center of projected mask
            contours, _ = cv2.findContours(projected_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Calculate center of the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])

                    # Determine marker based on filter reason
                    filter_reason = filter_info.get("reason", "unknown")

                    if "distance_threshold_m" in filter_reason:
                        # Red dot for distance threshold
                        marker_color = (255, 0, 0)  # Red in RGB
                        marker_type = "dot"
                        marker_size = 2  # Smaller size for distance filter
                    elif "skip_oncoming" in filter_reason:
                        # Blue cross for oncoming vehicles
                        marker_color = (0, 0, 255)  # Blue in RGB
                        marker_type = "cross"
                        marker_size = 8
                    elif "relevant_lanes" in filter_reason:
                        # Blue circle for lane filtering
                        marker_color = (0, 0, 255)  # Blue in RGB
                        marker_type = "circle"
                        marker_size = 8
                    elif "occlusion" in filter_reason:
                        # Small orange cross for occlusion
                        marker_color = (255, 165, 0)  # Orange in RGB
                        marker_type = "cross"
                        marker_size = 4  # Smaller size for occlusion
                    else:
                        # Generic marker for unknown reasons
                        marker_color = (255, 0, 255)  # Magenta in RGB
                        marker_type = "square"
                        marker_size = 6

                    marker_thickness = 2

                    # Draw marker based on type
                    if marker_type == "cross":
                        # Draw cross marker
                        cv2.line(
                            vis_image,
                            (center_x - marker_size, center_y - marker_size),
                            (center_x + marker_size, center_y + marker_size),
                            marker_color,
                            marker_thickness,
                        )
                        cv2.line(
                            vis_image,
                            (center_x + marker_size, center_y - marker_size),
                            (center_x - marker_size, center_y + marker_size),
                            marker_color,
                            marker_thickness,
                        )
                    elif marker_type == "dot":
                        # Draw filled circle
                        cv2.circle(vis_image, (center_x, center_y), marker_size, marker_color, -1)
                    elif marker_type == "circle":
                        # Draw circle outline
                        cv2.circle(vis_image, (center_x, center_y), marker_size, marker_color, marker_thickness)
                    elif marker_type == "square":
                        # Draw filled square
                        cv2.rectangle(
                            vis_image,
                            (center_x - marker_size, center_y - marker_size),
                            (center_x + marker_size, center_y + marker_size),
                            marker_color,
                            -1,
                        )

                    # Add track ID text below the marker (only in verbose mode)
                    if self.verbose:
                        track_id_text = f"{track_id}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.4
                        thickness = 1

                        # Get text size for background rectangle
                        (text_width, text_height), _baseline = cv2.getTextSize(
                            track_id_text, font, font_scale, thickness
                        )

                        # Position text below the marker
                        text_x = center_x - text_width // 2
                        text_y = center_y + marker_size + text_height + 5  # 5 pixels below the marker

                        # Ensure text doesn't go off the bottom of the image
                        if text_y + text_height > vis_image.shape[0] - 5:
                            text_y = center_y - marker_size - text_height - 5  # Put above the marker instead

                        # Draw background rectangle for text
                        cv2.rectangle(
                            vis_image,
                            (text_x - 3, text_y - text_height - 3),
                            (text_x + text_width + 3, text_y + 3),
                            (255, 255, 255),
                            -1,
                        )

                        # Draw track ID text
                        cv2.putText(vis_image, track_id_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

        return vis_image
