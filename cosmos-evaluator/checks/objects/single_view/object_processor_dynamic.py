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
Dynamic objects processor for SegFormer-based video processing.

This module implements the dynamic (SegFormer-based) objects processor, which processes video frames and scores
overlap between dynamic world model objects and segmentation masks to check for type correctness.
"""

import threading
from typing import Any, ClassVar

import numpy as np
import onnxruntime as ort
from third_party.cosmos_drive_dreams_toolkits.utils.camera.base import CameraBase

from checks.objects.single_view.hallucination_detector import HallucinationDetector
from checks.objects.single_view.hallucination_tracking import track_hallucinations
from checks.objects.single_view.object_processor_base import ObjectProcessorBase
from checks.objects.single_view.overlap_detector import OverlapDetector
from checks.objects.single_view.seg_helper import SegHelper
from checks.objects.single_view.seg_mask_cache import SegMaskCache
from checks.utils.coord_transforms import extract_rpy_in_flu
from checks.utils.frame_rates import ObjectCheckTimeline, validate_annotation_axis
from checks.utils.rds_data_loader import RdsDataLoader
from checks.utils.scene_rasterizer import SceneRasterizer
from utils.cancellation import check_cancelled


class ObjectProcessorDynamic(ObjectProcessorBase):
    """Implements dynamic (SegFormer-based) frame processing and scoring."""

    SEGFORMER_STATIC_LABEL_ORDER: ClassVar[list[str]] = [
        "road",
        "sidewalk",
        "traffic light",
        "traffic sign",
        "pole",
    ]

    SEGFORMER_MOVABLE_CLASS_ORDER: ClassVar[list[str]] = [
        "vehicle",
        "pedestrian",
        "motorcycle",
        "bicycle",
    ]

    def __init__(
        self,
        clip_id: str,
        output_dir: str,
        config: dict[str, Any] | None = None,
        model_device: str = "cuda",
        verbose: str = "INFO",
        seg_session: ort.InferenceSession | None = None,
    ):
        """Initialize dynamic (SegFormer-based) processing components.

        Args:
            clip_id: Clip identifier for logging/visualization file names.
            output_dir: Directory where outputs (logs, videos) are written.
            config: Processor configuration dictionary.
            model_device: Device for models, e.g. 'cuda' or 'cpu'.
            verbose: Logging verbosity level.
            seg_session: Pre-built SegFormer inference session to run on, or None to let this processor build its own.
                A caller serving concurrent requests should build one at startup and pass it here; constructing a
                session initializes the CUDA execution provider, which is too expensive and too failure-prone to do
                once per job.
        """
        super().__init__(clip_id, output_dir, config, model_device, verbose)
        self._init_dynamic_components(seg_session)

    def _init_dynamic_components(self, seg_session: ort.InferenceSession | None = None) -> None:
        """Initialize dynamic segmentation, overlap, and hallucination components."""
        self.naming_suffix = "dynamic"
        self.setup_logging()
        self.seg_model = SegHelper(self.model_device, session=seg_session)
        # Overlap detector owns its internal filters
        self.overlap_detector = OverlapDetector(
            self.oc_config, self.seg_model, self.logger, debug_enabled=(self.verbose == "DEBUG")
        )

        # Initialize hallucination detector if enabled, filtered to dynamic object types only
        self.hallucination_detector: HallucinationDetector | None = None
        halluc_cfg = self.oc_config.get("hallucination_detector", {})
        if isinstance(halluc_cfg, dict) and bool(halluc_cfg.get("enabled", False)):
            try:
                filtered_classes = {k: v for k, v in halluc_cfg.get("classes", {}).items() if k in self.DYNAMIC_OBJECTS}
                filtered_cfg = {**halluc_cfg, "classes": filtered_classes}
                self._hallucination_classes = filtered_classes
                self.hallucination_detector = HallucinationDetector(filtered_cfg, self.seg_model, self.logger)
                self.logger.info("Hallucination detector enabled")
                self._hallucination_frames: list[Any] = []
                self._hallucination_summary: dict[str, Any] = {}
            except Exception as e:  # noqa: BLE001 - detector init is best-effort; log and disable
                self.logger.warning("Failed to initialize HallucinationDetector: %s", e)
                self.hallucination_detector = None

    def _process_frames_dynamic(
        self,
        video_dataloader: Any,
        camera_poses: dict[int, np.ndarray],
        camera_model: CameraBase,
        data_loader: RdsDataLoader,
        timeline: ObjectCheckTimeline,
        results: dict[str, Any],
        track_output_agg: dict[int, dict[str, Any]],
        mask_cache: SegMaskCache | None = None,
        ego_poses: dict[int, Any] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Iterate video frames and process each one.

        Args:
            video_dataloader (Any): Iterable of video frames.
            camera_poses (dict[int, np.ndarray]): RDS frame index to 4x4 camera-to-world pose.
            camera_model (CameraBase): Camera intrinsics/projection model.
            data_loader (RdsDataLoader): Loader for per-frame object data and sizes.
            timeline (ObjectCheckTimeline): The object-check timeline; supplies the step count, which camera frames
                carry a step, and the RDS index behind each one.
            results (Dict[str, Any]): Results container to populate.
            track_output_agg (Dict[int, Dict[str, Any]]): Per-track aggregated metadata.
            mask_cache (SegMaskCache | None): Optional cache for resized segmentation masks.
                When provided, cached masks are loaded instead of running SegFormer inference, and newly computed
                masks are saved to the cache.

        Returns:
            tuple: `(score_matrix, importance_matrix, skipped_frames)` where ``score_matrix`` and
            ``importance_matrix`` are ``np.ndarray`` instances shaped ``(num_steps, max_track_id + 1)`` — rows are
            object-check steps, not source video frames — and ``skipped_frames`` is the number of steps without
            object data.
        """
        self.logger.info("Starting frame processing")

        max_track_id = max(data_loader.object_count, 1)
        score_matrix = np.full((timeline.num_steps, max_track_id + 1), np.nan)
        importance_matrix = np.zeros((timeline.num_steps, max_track_id + 1), dtype=np.int32)

        # The dataloader is sequential, so walk it and pick out the camera frames that carry a step. Frames between
        # steps are decoded but never segmented, which is where the saving is: at 30 fps two of every three frames
        # have no object annotation to be scored against and used to run SegFormer for nothing.
        step_by_camera_frame = {timeline.camera_frame(step): step for step in range(timeline.num_steps)}
        last_camera_frame = timeline.camera_frame(timeline.num_steps - 1) if timeline.num_steps else -1

        self.logger.info("Video dataloader length: %s", len(video_dataloader))
        self.logger.info("Camera poses available: %s", len(camera_poses))
        self.logger.info("Evaluating %s steps at %s camera frames apart", timeline.num_steps, timeline.camera_stride)

        skipped_frames = 0
        for frame_idx, frame in enumerate(video_dataloader):
            # Cooperative-cancellation checkpoint: bail out promptly if the owning request was cancelled so this thread
            # stops touching files the service is about to delete.
            check_cancelled(cancel_event)
            if frame_idx > last_camera_frame:
                break
            step = step_by_camera_frame.get(frame_idx)
            if step is None:
                continue

            mapped_idx = timeline.rds_index(step)
            # The timeline is bounded by the pose stream, so a miss here means the stream is sparse rather than short.
            if mapped_idx not in camera_poses:
                self.logger.warning(
                    "Step %s maps to RDS index %s, which the camera pose stream does not carry; stopping",
                    step,
                    mapped_idx,
                )
                break

            camera_pose = camera_poses[mapped_idx]
            num_objects = len(data_loader.get_object_data_for_frame(mapped_idx, include_static=False))
            if num_objects == 0:
                self.logger.debug("Skipping step %s (video frame %s) - no object data available", step, frame_idx)
                skipped_frames += 1
                continue

            self.profiler.start("frame_processing")
            self.logger.debug(
                "Processing step=%s video_idx=%s -> mapped_idx=%s (objects: %s)",
                step,
                frame_idx,
                mapped_idx,
                num_objects,
            )
            ego_pose = ego_poses.get(mapped_idx) if ego_poses else None
            frame_scores = self._process_frame(
                mapped_idx,
                frame,
                data_loader,
                camera_pose,
                camera_model,
                track_output_agg,
                results,
                video_frame_idx=frame_idx,
                mask_cache=mask_cache,
                ego_pose=ego_pose,
            )
            self.profiler.end()

            if frame_scores:
                for track_id, score in frame_scores.items():
                    tid = int(track_id)
                    results["track_ids"].add(tid)
                    score_matrix[step, tid] = score
                    # Set importance flag from track_output_agg (keys may be str or int)
                    agg_entry = track_output_agg.get(tid) or track_output_agg.get(track_id) or {}
                    imp_attrs = agg_entry.get("importance_attrs", [])
                    if imp_attrs and imp_attrs[-1].get("is_important", False):
                        importance_matrix[step, tid] = 1
                results["processed_frame_ids"].append(step)

        return score_matrix, importance_matrix, skipped_frames

    def _process_frame(
        self,
        frame_idx: int,
        frame: Any,
        data_loader: RdsDataLoader,
        camera_pose: np.ndarray,
        camera_model: CameraBase,
        track_output_agg: dict[int, dict[str, Any]],
        results: dict[str, Any],
        video_frame_idx: int | None = None,
        mask_cache: SegMaskCache | None = None,
        ego_pose: np.ndarray | None = None,
    ) -> dict[Any, float] | None:
        """Process a single frame: segmentation → scoring → hallucinations → visualization.

        Args:
            frame_idx (int): Index in the pose/object streams.
            frame (Any): Raw image/frame data for segmentation.
            data_loader (RdsDataLoader): Loader for object data and dimensions.
            camera_pose (np.ndarray): 4x4 camera pose matrix.
            camera_model (CameraBase): Camera intrinsics/projection model.
            track_output_agg (Dict[int, Dict[str, Any]]): Per-track aggregation store (updated in-place).
            results (Dict[str, Any]): Results container for IDs and detections.
            video_frame_idx (int | None): Optional original video index for visualization.
            mask_cache (SegMaskCache | None): Optional cache for resized segmentation masks.

        Returns:
            Optional[Dict[int, float]]: Map of `track_id -> score` for the frame, or None if empty.
        """
        if video_frame_idx is None:
            video_frame_idx = frame_idx
        self.logger.debug(
            "Frame video_idx=%s mapped_idx=%s: ego position in world =(%.1f, %.1f, %.1f)",
            video_frame_idx,
            frame_idx,
            camera_pose[0, 3],
            camera_pose[1, 3],
            camera_pose[2, 3],
        )

        roll, pitch, yaw = extract_rpy_in_flu(camera_pose[:3, :3])
        self.logger.debug(
            "Frame video_idx=%s mapped_idx=%s: camera RPY=(%.1f, %.1f, %.1f)",
            video_frame_idx,
            frame_idx,
            np.degrees(roll),
            np.degrees(pitch),
            np.degrees(yaw),
        )

        # Segmentation (load from cache when available, otherwise run inference and cache)
        self.profiler.start("segmentation")
        cache_key = video_frame_idx if video_frame_idx is not None else frame_idx
        resized_masks = mask_cache.load(cache_key) if mask_cache is not None else None
        if resized_masks is not None:
            self.logger.debug("Frame %s: loaded segmentation masks from cache", cache_key)
        else:
            segmentation_masks = self.seg_model.process_frame(frame)
            resized_masks = self.seg_model.resize_masks(
                segmentation_masks,
                data_loader.CAMERA_RESCALED_RESOLUTION_HEIGHT,
                data_loader.CAMERA_RESCALED_RESOLUTION_WIDTH,
            )
            if mask_cache is not None:
                mask_cache.save(cache_key, resized_masks)
        self.profiler.end()

        # Create SceneRasterizer for this frame - computes visibility masks in constructor
        # This computes occlusion-aware visibility masks that are shared across all components
        self.profiler.start("scene_rasterization")
        scene_rasterizer = SceneRasterizer(
            data_loader.get_object_data_for_frame(frame_idx, include_static=True),
            camera_pose,
            camera_model,
            data_loader.CAMERA_RESCALED_RESOLUTION_WIDTH,
            data_loader.CAMERA_RESCALED_RESOLUTION_HEIGHT,
            logger=self.logger,
        )
        self.profiler.end()

        # Per-class processing
        scores: dict[Any, float] = {}
        filtered_objects: dict[str, Any] = {}

        self.profiler.start("processing")
        frame_objects = data_loader.get_object_data_for_frame(frame_idx, include_static=False)
        for class_name in self.oc_config.get("overlap_check", {}):
            if class_name not in self.DYNAMIC_OBJECTS:
                continue
            cls_scores, cls_filtered, _debug_info = self.overlap_detector.process_class(
                class_name,
                resized_masks,
                frame_objects,
                camera_pose,
                camera_model,
                data_loader.CAMERA_RESCALED_RESOLUTION_WIDTH,
                data_loader.CAMERA_RESCALED_RESOLUTION_HEIGHT,
                track_output_agg,
                frame_idx,
                scene_rasterizer=scene_rasterizer,
                ego_pose=ego_pose,
            )

            scores.update(cls_scores)
            filtered_objects.update(cls_filtered)

        self.profiler.end()

        # Debug mask video
        if self.debug_enabled and self.vis_helper:
            try:
                h = int(data_loader.CAMERA_RESCALED_RESOLUTION_HEIGHT)
                w = int(data_loader.CAMERA_RESCALED_RESOLUTION_WIDTH)
                mask_frame = np.zeros((h, w, 3), dtype=np.uint8)
                seg_rgb = resized_masks.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                for seg_label in self.SEGFORMER_STATIC_LABEL_ORDER:
                    try:
                        target_color = np.array(self.seg_model.get_class_color(seg_label), dtype=np.uint8)
                        mask = np.all(seg_rgb == target_color, axis=-1)
                        rgb = self.vis_helper.mask_colors.get(seg_label)
                        if rgb is not None:
                            mask_frame[mask] = np.array(rgb, dtype=np.uint8)
                    except Exception as e:  # noqa: BLE001 - per-label static mask render must not abort the frame
                        self.logger.debug("Failed to process static label %s: %s", seg_label, e)
                for cls_name in self.SEGFORMER_MOVABLE_CLASS_ORDER:
                    try:
                        cls_mask = self.seg_model.get_class_mask(resized_masks, cls_name)
                        if cls_mask is not None:
                            rgb = self.vis_helper.mask_colors.get(cls_name)
                            if rgb is not None:
                                mask_frame[cls_mask.astype(bool)] = np.array(rgb, dtype=np.uint8)
                    except Exception as e:  # noqa: BLE001 - per-class movable mask render must not abort the frame
                        self.logger.debug("Failed to process movable class %s: %s", cls_name, e)
                self.vis_helper.write_mask_frame(mask_frame, video_frame_idx, self.naming_suffix)
            except Exception as _e:  # noqa: BLE001 - best-effort debug mask frame write
                self.logger.warning("Failed to write debug mask frame for frame %s: %s", frame_idx, _e)

        # Hallucinations
        halluc_det = None
        if self.hallucination_detector is not None:
            self.profiler.start("hallucination_detection")
            try:
                halluc_det = self.hallucination_detector.detect(
                    frame_idx=video_frame_idx,
                    resized_masks=resized_masks,
                    frame_objects=frame_objects,
                    camera_pose=camera_pose,
                    camera_model=camera_model,
                    image_width=data_loader.CAMERA_RESCALED_RESOLUTION_WIDTH,
                    image_height=data_loader.CAMERA_RESCALED_RESOLUTION_HEIGHT,
                    scene_rasterizer=scene_rasterizer,
                )
                if halluc_det:
                    start_indices = {}
                    for cls_name, recs in halluc_det.items():
                        bucket = results.get("hallucination_detections", {}).get(cls_name, None)
                        if bucket is not None and isinstance(recs, list):
                            start_indices[cls_name] = len(bucket)
                            bucket.extend(recs)
                    if any(len(v or []) > 0 for v in halluc_det.values()):
                        try:
                            # Compute road mask from segmentation output
                            road_mask = None
                            try:
                                seg_rgb = resized_masks.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                                road_color_np = SegHelper.get_class_color("road")
                                road_mask = np.all(seg_rgb == road_color_np, axis=-1)
                            except (AttributeError, ValueError):
                                road_mask = None
                            track_hallucinations(results=results, start_indices=start_indices, road_mask=road_mask)
                        except Exception as _te:  # noqa: BLE001 - per-frame tracking must not abort the run
                            self.logger.warning("Hallucination tracking failed for frame %s: %s", frame_idx, _te)
                    counts = {k: len(v or []) for k, v in (halluc_det or {}).items()}
                    self.logger.info("Frame %s: hallucinations %s", frame_idx, counts)
            except Exception as _e:  # noqa: BLE001 - per-frame detection must not abort the run
                self.logger.warning("Hallucination detection failed for frame %s: %s", frame_idx, _e)
            finally:
                self.profiler.end()

        # Visualization frame
        if self.verbose and self.vis_helper:
            self.profiler.start("visualization")
            self._create_frame_visualization(
                video_frame_idx,
                resized_masks,
                scores,
                camera_pose,
                camera_model,
                frame_objects,
                filtered_objects,
                halluc_det,
                scene_rasterizer=scene_rasterizer,
            )
            self.profiler.end()

        return scores if scores else None

    def process_clip(
        self,
        input_data: str,
        clip_id: str,
        camera_name: str,
        video_path: str,
        output_dir: str,
        trial_frames: int | None = None,
        mask_cache_dir: str | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Run the dynamic objects pipeline for a single clip.

        Args:
            input_data (str): Path to the dataset root or run directory.
            clip_id (str): Clip identifier.
            camera_name (str): Camera stream name for poses/intrinsics.
            video_path (str): Path to the camera video file.
            output_dir (str): Output directory for logs/visualizations.
            trial_frames (int | None): If provided, evaluate at most this many object-check steps, i.e. the
                first ``trial_frames / OBJECT_CHECK_FPS`` seconds of the clip.
            mask_cache_dir (str | None): Directory for caching per-frame SegFormer masks.
                When provided, masks are loaded from cache on subsequent runs with the same video, skipping SegFormer
                inference entirely.
            cancel_event (threading.Event | None): Optional cooperative-cancellation event polled at each frame; when
                set, the frame loop raises ``ProcessingCancelled``. ``None`` disables cancellation (local CLI).

        Returns:
            Dict[str, Any]: Aggregated results including score matrix and statistics.
        """
        self.profiler.start("total_processing")

        # Initialization
        data_loader = self._create_data_loader(input_data, clip_id)
        camera_poses, camera_model = data_loader.load_camera_data(camera_name)
        try:
            ego_poses = data_loader.load_ego_poses()
        except Exception:
            self.logger.warning(
                "Could not load ego poses from %s - importance filter will use camera pose",
                RdsDataLoader.EGO_REFERENCE_CAMERA,
                exc_info=True,
            )
            ego_poses = None

        # Cooperative-cancellation checkpoint after data loading. Building the video dataset and initializing
        # visualization below are expensive enough that a request cancelled during the RDS load would otherwise run
        # for a long time before reaching the first per-frame checkpoint.
        check_cancelled(cancel_event)

        video_dataloader, video_fps = self._prepare_video(video_path, data_loader)
        if video_fps is None:
            raise RuntimeError(f"Failed to get video FPS from video path: {video_path}")
        self._maybe_init_visualization(video_path, data_loader)

        # One timeline for the three inputs this check reads: the camera video, the pose stream and the object
        # annotations. No world video is involved — the dynamic check scores the camera video directly.
        annotation_axis = sorted(data_loader.all_dynamic_objects)
        timeline = ObjectCheckTimeline.build(
            camera_video_fps=video_fps,
            camera_frame_count=len(video_dataloader),
            pose_frame_count=(max(camera_poses) + 1) if camera_poses else 0,
            annotated_frame_count=len(annotation_axis),
            trial_steps=trial_frames,
            log=self.logger,
        )
        validate_annotation_axis(annotation_axis, timeline.num_steps, self.logger)

        mask_cache = SegMaskCache(mask_cache_dir, video_path) if mask_cache_dir is not None else None
        if mask_cache is not None:
            self.logger.info("SegFormer mask cache enabled: %s", mask_cache_dir)

        # Dynamic frame processing
        # total_video_frames stays on the source-video axis, which is what the persisted schema documents and
        # what every result consumer reads; the evaluation length is a separate field.
        results = self._init_results_container(len(video_dataloader))
        results["total_evaluation_steps"] = timeline.num_steps
        track_output_agg: dict[int, dict[str, Any]] = {}
        try:
            # Inside the try, not before it: visualization is already initialized by this point, so an abort here has
            # ffmpeg children to release and must reach the handler below.
            check_cancelled(cancel_event)
            score_matrix, importance_matrix, skipped_frames = self._process_frames_dynamic(
                video_dataloader,
                camera_poses,
                camera_model,
                data_loader,
                timeline,
                results,
                track_output_agg,
                mask_cache=mask_cache,
                ego_poses=ego_poses,
                cancel_event=cancel_event,
            )
        except BaseException:
            # On ProcessingCancelled (or any other abort) the happy-path release inside _finalize_and_summarize never
            # runs, leaving the visualization helper's ffmpeg children alive on a temp dir that is about to be deleted.
            self._release_visualization()
            raise
        finally:
            # On every exit, since an aborted job strands its arena regions just as a completed one does. The session
            # is shared, so these regions otherwise accumulate across jobs for the life of the replica.
            self.seg_model.release_arena_memory()

        # Finalization
        self._finalize_and_summarize(
            results,
            score_matrix,
            track_output_agg,
            skipped_frames,
            len(video_dataloader),
            importance_matrix=importance_matrix,
        )

        self.profiler.end()
        self.profiler.print_summary(self.logger)
        self.logger.info("Processing complete. Mean correspondence score: %.3f", results["mean_score"])
        return results
