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
Base class for objects processors.

This class provides shared functionality for objects processors, including logging, data loading, and visualization.
"""

import logging
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from third_party.cosmos_drive_dreams_toolkits.utils.camera.base import CameraBase

from checks.objects.results import compute_signal_group_scores, save_results_to_json
from checks.objects.single_view.importance_filter import ImportanceFilter
from checks.objects.single_view.visualization_helper import VisualizationHelper
from checks.utils.config_manager import ConfigManager, resolve_check_config
from checks.utils.frame_rates import OBJECT_CHECK_FPS
from checks.utils.object_class_names import class_name_to_rds_label
from checks.utils.onnx import configure_onnx_logging
from checks.utils.profiler import Profiler
from checks.utils.rds_data_loader import RdsDataLoader
from checks.utils.scene_rasterizer import SceneRasterizer
from checks.utils.segformer import get_dataloader

logger = logging.getLogger(__name__)


class ObjectProcessorBase:
    """
    Base processor providing shared functionality for object processing.

    Holds common configuration, logging, helpers, profiling, and utilities used by both dynamic (SegFormer-based) and
    static (CWIP-based) processors.
    """

    DYNAMIC_OBJECTS: ClassVar[list[str]] = [
        "bicycle",
        "motorcycle",
        "pedestrian",
        "vehicle",
    ]

    STATIC_OBJECTS: ClassVar[list[str]] = [
        "crosswalk",
        "lane_line",
        "road_boundary",
        "traffic_light",
        "traffic_sign",
        "wait_line",
    ]

    def __init__(
        self,
        clip_id: str,
        output_dir: str,
        config: dict[str, Any] | None = None,
        model_device: str = "cuda",
        verbose: str = "INFO",
    ) -> None:
        """Initialize the base objects processor.

        Args:
            clip_id: Clip identifier used for logging/visualization file names.
            output_dir: Directory where outputs (logs, videos) are written.
            config: Processor configuration dictionary (expects key 'av.objects_single_view'). A non-empty config
                replaces the defaults wholesale, so it must be complete; None or an empty {} uses the defaults.
            model_device: Torch/ONNX device to run models on, e.g. 'cuda' or 'cpu'.
            verbose: Logging verbosity level (e.g. 'DEBUG', 'INFO').

        Raises:
            ValueError: If a non-empty config omits 'av.objects_single_view' or its required 'overlap_check' key.
        """
        self.model_device = model_device
        self.verbose = verbose
        self.clip_id = clip_id
        self.output_dir = output_dir
        self.naming_suffix = "base"

        # 'overlap_check' is required because it also gates which processors run at all (see api.run_object_processors);
        # without it an incomplete config would silently process nothing rather than failing.
        self.oc_config = resolve_check_config(
            config,
            ObjectProcessorBase.get_default_config(),
            section_key="av.objects_single_view",
            required_keys=("overlap_check",),
        )

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize shared components
        self.seg_model: Any = None

        # Initialize filters for tests (and potential external use)
        self.importance_filter: ImportanceFilter | None = None
        try:
            importance_cfg = self.oc_config.get("importance_filter", {})
            self.importance_filter = ImportanceFilter(importance_cfg)
        except Exception as e:  # noqa: BLE001 - filter init is best-effort; log and disable
            self.logger.error("Failed to initialize ImportanceFilter: %s", e)
            self.importance_filter = None

        # Initialize visualization helper
        self.vis_helper: VisualizationHelper | None = None
        if self.oc_config.get("visualization", {}).get("enabled", True):
            vis_verbose = verbose == "DEBUG"
            self.vis_helper = VisualizationHelper(clip_id, output_dir, vis_verbose)
        else:
            self.logger.info("Visualization generation disabled.")

        # DEBUG flag retained for logging verbosity only
        self.debug_enabled = verbose == "DEBUG"

        # Hallucination classes relevant to this processor; overridden by subclasses
        self._hallucination_classes: dict[str, Any] = {}

        # Initialize profiler
        self.profiler = Profiler()

    def _create_data_loader(self, input_data: str, clip_id: str) -> RdsDataLoader:
        """Construct the world model data loader used by ``process_clip``.

        Extension point: subclasses that need a specialized loader (or that want to mutate the loaded world model
        before processing starts) override this rather than duplicating ``process_clip``. The returned loader must
        satisfy the ``RdsDataLoader`` interface.

        Args:
            input_data: Path to the dataset root or run directory.
            clip_id: Clip identifier.

        Returns:
            A loader with the clip's world model data already read.
        """
        return RdsDataLoader(input_data, clip_id)

    def setup_logging(self) -> None:
        """Setup logging handlers and levels.

        Notes:
        - Uses module logger with both console and optional file handlers.
        - File logging is enabled when clip_id and output_dir are provided.
        - In DEBUG mode we also enable verbose ONNX logging.
        - Expected to be called manually after initialization to allow for overriding the log naming suffix.
        """

        # Set logging level based on verbose mode
        log_level = getattr(logging, self.verbose.upper())
        self.logger.setLevel(log_level)

        # Ensure propagation is enabled
        self.logger.propagate = True

        # Clear any existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Create file handler if clip_id and output_dir are provided
        if self.clip_id and self.output_dir:
            log_file_path = Path(self.output_dir) / f"{self.clip_id}.{self.naming_suffix}.objects.log"

            # Clear the log file by opening in write mode (truncates the file)
            file_handler = logging.FileHandler(log_file_path, mode="w")
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            self.logger.info("Logging to file: %s", log_file_path)

            if self.verbose == "DEBUG":
                self.logger.debug("DEBUG logging is enabled")
                configure_onnx_logging(verbose=True)

    def _prepare_video(self, video_path: str, data_loader: RdsDataLoader) -> tuple[Any, float | None]:
        """Prepare the video dataloader and report the source video's frame rate.

        Args:
            video_path (str): Path to the input video file.
            data_loader (RdsDataLoader): Loader providing camera resolution metadata.

        Returns:
            tuple: `(video_dataloader, video_fps)` where `video_fps` is the detected rate, or ``None`` when the
            container does not report a usable one. The caller decides how to proceed; the evaluation rate itself is
            fixed by :class:`checks.utils.frame_rates.ObjectCheckTimeline` and is not derived from the video.
        """
        self.profiler.start("video_loader_init")
        # Prefer seg_helper-provided dataloader in tests/mocks; fallback to default
        try:
            if hasattr(self.seg_model, "get_video_dataloader") and callable(self.seg_model.get_video_dataloader):
                video_dataloader, video_dataset = self.seg_model.get_video_dataloader(video_path)
            else:
                video_dataloader, video_dataset = get_dataloader(video_path)
        except Exception as exc:  # noqa: BLE001 - seg_model may be a mock; fall back to default dataloader
            self.logger.debug("seg_model.get_video_dataloader failed, falling back to get_dataloader: %s", exc)
            video_dataloader, video_dataset = get_dataloader(video_path)

        # Log video dataset information
        video_fps = None
        if hasattr(video_dataset, "number_frames"):
            self.logger.info("Original video frames: %s", video_dataset.number_frames)
            video_fps = getattr(video_dataset, "frames_per_second", None)
            if video_fps is not None:
                try:
                    video_fps = float(video_fps)
                except (ValueError, TypeError) as exc:
                    self.logger.debug("Could not parse video FPS as float: %s", exc)
                    video_fps = None
            self.logger.info("Video FPS: %s", video_fps if video_fps is not None else "unknown")
            if hasattr(video_dataset, "_skip"):
                self.logger.info("Frame skip: %s", video_dataset._skip)
        if video_fps is not None and video_fps <= 0:
            video_fps = None

        self.profiler.end()
        return video_dataloader, video_fps

    def _maybe_init_visualization(self, video_path: str, data_loader: RdsDataLoader) -> None:
        """Initialize the visualization writer when verbose mode is enabled.

        Args:
            video_path (str): Path to the source video used for visualization metadata.
            data_loader (RdsDataLoader): Provides target width/height for visualization frames.

        This is best-effort; failures to initialize the writer don't block processing.
        """
        # Setup output directory (ensure exists for viz)
        # Note: file writing is managed in VisualizationHelper
        if self.verbose and self.vis_helper:
            self.profiler.start("visualization_init")
            if not self.vis_helper.initialize_video_writer(
                video_path,
                naming_suffix=self.naming_suffix + ".objects",
                target_width=data_loader.CAMERA_RESCALED_RESOLUTION_WIDTH,
                target_height=data_loader.CAMERA_RESCALED_RESOLUTION_HEIGHT,
                fps=10.0,
            ):
                self.logger.warning("Failed to initialize video writer, continuing without visualization")
            self.profiler.end()

    def _init_results_container(self, total_video_frames: int) -> dict[str, Any]:
        """Initialize the results structure accumulated over the entire clip.

        Args:
            total_video_frames (int): Total number of frames in the input video.

        Returns:
            Dict[str, Any]: Results dict with keys such as `processed_frames`, `total_video_frames`,
            `total_evaluation_steps`, `evaluation_fps`, `score_matrix`, `track_ids`, `processed_frame_ids`,
            `hallucination_detections`, and `hallucination_tracks`.
        """
        hallucination_detections: dict[str, list[Any]] = {}
        for class_name in self._hallucination_classes:
            hallucination_detections[class_name_to_rds_label(class_name)] = []
        return {
            "processed_frames": 0,
            "total_video_frames": int(total_video_frames),
            # Steps the check could have evaluated, i.e. the denominator ``processed_frames`` belongs over.
            # ``total_video_frames`` stays on the source-video axis, so the two differ whenever the video is
            # denser than the evaluation rate; the processor overwrites this once its timeline is built.
            "total_evaluation_steps": 0,
            # The rate ``processed_frame_ids`` and the score-matrix rows are on. Fixed, but recorded so a
            # consumer can tell what a "frame" in this file means without knowing the checker's internals.
            "evaluation_fps": OBJECT_CHECK_FPS,
            "score_matrix": None,
            "track_ids": set(),
            "processed_frame_ids": [],
            "hallucination_detections": hallucination_detections,
            "hallucination_tracks": [],
        }

    # Dynamic frame processing lives in ObjectProcessorDynamic

    def _release_visualization(self) -> None:
        """Finalize the overlay video and terminate its ffmpeg children. Idempotent; safe on an abort path.

        Called both from the happy path in :meth:`_finalize_and_summarize` and from the frame loop's abort handler,
        so a cancelled run does not leave a decoder reading from a temp dir that is about to be deleted.
        """
        if self.vis_helper is None:
            return
        try:
            self.vis_helper.release()
        except Exception:
            self.logger.warning("Failed to release visualization helper", exc_info=True)

    def _finalize_and_summarize(
        self,
        results: dict[str, Any],
        score_matrix: np.ndarray,
        track_output_agg: dict[int, dict[str, Any]],
        skipped_frames: int,
        total_video_frames: int,
        importance_matrix: np.ndarray | None = None,
        projection_area_matrix: np.ndarray | None = None,
    ) -> None:
        """Finalize results, compute statistics, and release visualization resources.

        Args:
            results (Dict[str, Any]): Results container to be finalized/augmented in-place.
            score_matrix (np.ndarray): Frame-by-track score matrix (may include NaNs).
            track_output_agg (Dict[int, Dict[str, Any]]): Per-track aggregated metadata.
            skipped_frames (int): Number of frames without object data.
            total_video_frames (int): Total frames in the video timeline.
        """
        processed_frames = len(results["processed_frame_ids"])
        results["processed_frames"] = processed_frames

        self.logger.info("Frame processing complete:")
        self.logger.info("  - Evaluation rate: %s fps", OBJECT_CHECK_FPS)
        self.logger.info("  - Total video frames: %s", total_video_frames)
        self.logger.info("  - Evaluation steps available: %s", results["total_evaluation_steps"])
        self.logger.info("  - Steps with object data: %s", processed_frames)
        self.logger.info("  - Steps skipped (no objects): %s", skipped_frames)
        total_steps = results["total_evaluation_steps"]
        self.logger.info(
            "  - Step coverage: %s/%s (%.1f%%); source video frames: %s",
            processed_frames,
            total_steps,
            processed_frames / max(total_steps, 1) * 100,
            total_video_frames,
        )

        # Log filter configuration for reference
        importance_config = self.oc_config.get("importance_filter", {})
        self.logger.info("Filter configuration:")
        self.logger.info(
            "  - Importance filter: distance_threshold=%sm, oncoming_objects=%s, relevant_lanes=%s",
            importance_config.get("distance_threshold_m", "N/A"),
            importance_config.get("oncoming_objects", False),
            importance_config.get("relevant_lanes", []),
        )

        # Hallucination results already aggregated directly into results["hallucination_detections"]

        # Release visualization helper resources
        self._release_visualization()

        # Trim score matrix to processed frames and tracks
        if results["processed_frame_ids"] and results["track_ids"]:
            processed_frame_ids = np.array(results["processed_frame_ids"])
            results["track_ids"] = sorted(results["track_ids"])
            score_matrix = score_matrix[processed_frame_ids][:, results["track_ids"]]
            if projection_area_matrix is not None:
                projection_area_matrix = projection_area_matrix[processed_frame_ids][:, results["track_ids"]]
            if importance_matrix is not None:
                importance_matrix = importance_matrix[processed_frame_ids][:, results["track_ids"]]

        results["score_matrix"] = score_matrix

        # Compute overall statistics from score matrix
        if results["score_matrix"] is not None:
            results["mean_score"] = np.nanmean(results["score_matrix"])
            results["std_score"] = np.nanstd(results["score_matrix"])
            results["min_score"] = np.nanmin(results["score_matrix"])
            results["max_score"] = np.nanmax(results["score_matrix"])
        else:
            results["mean_score"] = 0.0
            results["std_score"] = 0.0
            results["min_score"] = 0.0
            results["max_score"] = 0.0

        # Build unified tracks list
        try:
            tracks_list = []
            for t_id in sorted(track_output_agg.keys()):
                entry = track_output_agg[t_id]
                types_set = entry.get("object_types") or set()
                object_type = None
                if types_set:
                    try:
                        object_type = sorted(types_set)[0]
                    except TypeError as exc:
                        self.logger.debug("Could not sort object types, using first: %s", exc)
                        object_type = next(iter(types_set))
                track_entry = {
                    "track_id": int(t_id),
                    "object_type": object_type,
                }
                if "object_type_index" in entry:
                    track_entry["object_type_index"] = entry["object_type_index"]
                # Surface a synthesized object's stable identity (carried as the ``synthetic_source_id``
                # attribute)
                if "synthetic_source_id" in entry:
                    track_entry["synthetic_source_id"] = entry["synthetic_source_id"]

                tl_state_data = results.get("traffic_light_state", {}).get(str(t_id))
                if tl_state_data:
                    scores = [f["traffic_light_state_score"] for f in tl_state_data]
                    track_entry["traffic_light_state_score"] = float(np.nanmean(scores)) if scores else float("nan")
                    mc_scores = [
                        f["traffic_light_multi_color_score"]
                        for f in tl_state_data
                        if "traffic_light_multi_color_score" in f
                    ]
                    if mc_scores:
                        track_entry["traffic_light_multi_color_score"] = float(np.nanmean(mc_scores))
                    confidences = [
                        f["traffic_light_state_confidence"]
                        for f in tl_state_data
                        if "traffic_light_state_confidence" in f
                    ]
                    if confidences:
                        track_entry["traffic_light_state_confidence"] = float(np.nanmean(confidences))
                    # Expected GT color for this track (stable across frames); a result consumer maps it to the
                    # color that was rendered wrong. Use the most common non-None value.
                    expected_states = [f["expected_state"] for f in tl_state_data if f.get("expected_state")]
                    if expected_states:
                        # Most common non-None value; break count ties by sorted order so the same track always
                        # picks the same state (set iteration order is not stable across runs) and the reported
                        # ``expected_state`` stays deterministic.
                        track_entry["expected_state"] = max(sorted(set(expected_states)), key=expected_states.count)
                    # Per-color mass breakdown, mean over frames (diagnostic: which color CWIP read for this track).
                    sp_frames = [
                        f["traffic_light_state_probs"] for f in tl_state_data if f.get("traffic_light_state_probs")
                    ]
                    if sp_frames:
                        track_entry["traffic_light_state_probs"] = {
                            c: float(np.nanmean([sp.get(c, float("nan")) for sp in sp_frames]))
                            for c in ("red", "yellow", "green", "unknown")
                        }

                imp_attrs_list = entry.get("importance_attrs", [])
                if imp_attrs_list:
                    lane_counts: dict[str, int] = {}
                    oncoming_frames = 0
                    important_frames = 0
                    min_distance = float("inf")
                    for attrs in imp_attrs_list:
                        lane = attrs.get("lane", "unknown")
                        lane_counts[lane] = lane_counts.get(lane, 0) + 1
                        if attrs.get("is_oncoming", False):
                            oncoming_frames += 1
                        if attrs.get("is_important", False):
                            important_frames += 1
                        # distance_m is None for objects without a centroid (lanelines, road boundaries); skip those
                        # so min_distance_m reflects only frames where a real GT distance was computed.
                        d = attrs.get("distance_m")
                        if d is not None:
                            min_distance = min(min_distance, d)
                    track_entry["importance"] = {
                        "lane_counts": lane_counts,
                        "oncoming_frames": oncoming_frames,
                        "min_distance_m": round(min_distance, 2) if min_distance < float("inf") else None,
                        "important_frames": important_frames,
                        "total_frames": len(imp_attrs_list),
                    }

                tracks_list.append(track_entry)
            results["tracks"] = tracks_list
        except Exception as e:  # noqa: BLE001 - track-list build is best-effort; log and emit empty list
            self.logger.warning("Failed to build tracks list: %s", e)
            self.logger.warning("Setting tracks list to empty list")
            results["tracks"] = []

        if self.naming_suffix == "static" and results.get("score_matrix") is not None:
            sorted_track_ids = list(results.get("track_ids") or [])
            results["signal_group_scores"] = compute_signal_group_scores(
                results["score_matrix"],
                sorted_track_ids,
                results.get("tracks") or [],
            )
            for group, group_stats in results["signal_group_scores"].items():
                self.logger.info(
                    "  - %s mean correspondence score: %.3f (%d tracks, %d scored)",
                    group,
                    group_stats["mean_score"],
                    group_stats["num_tracks"],
                    group_stats["num_scored"],
                )

        # Print score matrix in compact format if verbose
        if self.verbose and results["score_matrix"] is not None:
            self._print_score_matrix(results["score_matrix"], results["track_ids"])

        # Save results to JSON file
        try:
            # Use sparse_compact whenever a per-cell auxiliary matrix is provided
            # (importance for dynamic, projection_area_px for static); fall back to
            # plain sparse otherwise. The per-type schema is declared in the
            # ``columns`` field of the saved sparse_compact payload.
            has_auxiliary = importance_matrix is not None or projection_area_matrix is not None
            fmt = "sparse_compact" if has_auxiliary else "sparse"
            output_file = save_results_to_json(
                results,
                self.clip_id,
                self.output_dir,
                matrix_format=fmt,
                output_file_prefix=self.naming_suffix,
                importance_matrix=importance_matrix,
                projection_area_matrix=projection_area_matrix,
            )
            self.logger.info("Results saved to: %s", output_file)
        except Exception as e:  # noqa: BLE001 - JSON serialization failure must not abort processing
            self.logger.warning("Failed to save results to JSON: %s", e)

    def _create_frame_visualization(
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
    ) -> None:
        """Create visualization for a frame using the visualization helper.

        Args:
            frame_idx (int): Video frame index to annotate.
            segmentation_masks (Any): Segmentation masks or mask RGB image for the frame.
            frame_scores (Dict[int, float]): Per-track correspondence scores for the frame.
            camera_pose (np.ndarray): 4x4 pose matrix of the camera in world coordinates.
            camera_model (CameraBase): Camera intrinsics/projection model instance.
            frame_objects (Dict[str, Any]): Raw object data for the frame.
            filtered_objects (Dict[str, Any]): Filtered/relevant objects for visualization.
            hallucinations (Dict[str, Any] | None): Optional hallucination detections for overlays.
            scene_rasterizer (SceneRasterizer | None): Optional pre-computed SceneRasterizer with visibility masks
                for efficient O(1) lookup.
            traffic_light_state_results (Dict[str, Any] | None): Optional per-track traffic light state results
                for overlay tags.
        """
        if self.vis_helper is None:
            return

        # Create visualization using the helper
        vis_image = self.vis_helper.create_frame_visualization(
            frame_idx,
            segmentation_masks,
            frame_scores,
            camera_pose,
            camera_model,
            frame_objects,
            filtered_objects,
            hallucinations,
            scene_rasterizer=scene_rasterizer,
            traffic_light_state_results=traffic_light_state_results,
        )

        if vis_image is not None:
            # Write frame to video
            self.vis_helper.write_frame(vis_image, frame_idx)

    def _print_score_matrix(self, score_matrix: np.ndarray, track_ids: set) -> None:
        """Print score matrix in a compact, human-readable format for debugging.

        Args:
            score_matrix (np.ndarray): Matrix of per-frame, per-track scores.
            track_ids (set): Set of track IDs that were populated.
        """
        if not track_ids:
            self.logger.info("No track IDs found in score matrix")
            return

        self.logger.info("=" * 60)
        self.logger.info("SCORE MATRIX SUMMARY")
        self.logger.info("=" * 60)

        # Print matrix dimensions
        frames, max_track_id = score_matrix.shape
        self.logger.info("Matrix shape: %s frames x %s track IDs", frames, max_track_id)
        self.logger.info("Unique track IDs: %s", len(track_ids))

        # Calculate tracks and frames with valid scores
        tracks_with_valid_scores = np.sum(~np.all(np.isnan(score_matrix), axis=0))
        frames_with_valid_scores = np.sum(~np.all(np.isnan(score_matrix), axis=1))

        self.logger.info("Tracks with valid scores: %s", tracks_with_valid_scores)
        self.logger.info("Frames with valid scores: %s", frames_with_valid_scores)

        # Print statistics
        valid_scores = score_matrix[~np.isnan(score_matrix)]
        nan_count = np.isnan(score_matrix).sum()
        total_elements = score_matrix.size

        self.logger.info("Valid scores: %s", len(valid_scores))
        self.logger.info("NaN entries: %s (%.1f%%)", nan_count, nan_count / total_elements * 100)

        if len(valid_scores) > 0:
            self.logger.info("Score range: %.3f - %.3f", valid_scores.min(), valid_scores.max())
            self.logger.info("Mean score: %.3f", valid_scores.mean())
            self.logger.info("Std score: %.3f", valid_scores.std())

        # Print compact matrix representation (first 10 frames, first 10 track IDs)
        if frames > 0 and max_track_id > 0:
            self.logger.info("\nCompact Matrix (first 10 frames × first 10 track IDs):")

            # Build header row
            header = "Frame\\TrackID"
            for track_id in range(min(10, max_track_id)):
                header += f"  {track_id:3d}"
            self.logger.info(header)

            # Build matrix rows
            for frame_idx in range(min(10, frames)):
                row = f"{frame_idx:4d}"
                for track_id in range(min(10, max_track_id)):
                    score = score_matrix[frame_idx, track_id]
                    if np.isnan(score):
                        row += "  ---"
                    else:
                        row += f" {score:4.2f}"
                self.logger.info(row)

        self.logger.info("=" * 60)

    def get_config_summary(self) -> dict[str, Any]:
        """Get a summary of the current configuration for logging/inspection.

        Returns:
            Dict[str, Any]: Selected configuration subsections and runtime settings.
        """
        summary = {
            "overlap_check": self.oc_config.get("overlap_check", {}),
            "importance_filter": self.oc_config.get("importance_filter", {}),
            "model_device": self.model_device,
        }
        try:
            hall_cfg = self.oc_config.get("hallucination_detector", {}) or {}
            if hall_cfg:
                summary["hallucination_detector"] = {
                    "enabled": bool(hall_cfg.get("enabled", False)),
                    "classes": list(hall_cfg.get("classes", [])),
                    "min_cluster_area": int(hall_cfg.get("min_cluster_area", 100)),
                    "max_cluster_per_frame": int(hall_cfg.get("max_cluster_per_frame", 100)),
                }
        except (KeyError, TypeError, ValueError) as e:
            self.logger.error("Failed to summarize hallucination_detector config: %s", e)
        return summary

    # Static frame processing lives in ObjectProcessorStatic

    @staticmethod
    def get_default_config() -> dict[str, Any]:
        """Load the default configuration from cosmos_evaluator.yaml.

        Returns:
            Dict[str, Any]: Dict with key 'av.objects_single_view' containing objects config.
        """
        try:
            config_manager = ConfigManager()
            config = config_manager.load_config("cosmos_evaluator")
            return {"av.objects_single_view": config["av.objects_single_view"]}
        except Exception as e:
            logger.error("Error loading default configuration: %s", e)
            raise
