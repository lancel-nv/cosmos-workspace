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
Visualization utilities.
"""

import csv
from enum import StrEnum
import json
import logging
import os
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from checks.utils.models.cwip.visualization import image_video_utils, world_model_utils

logger = logging.getLogger(__name__)


class TilingOutputMode(StrEnum):
    """
    Enum representing different output modes for tiling functions.

    Attributes:
        IMAGE: Create a static tiled image grid
        VIDEO: Create a video file with tiled frames
    """

    IMAGE = "image"
    VIDEO = "video"


def _make_text_placeholder(
    text: str,
    num_frames: int,
    height: int,
    width: int,
    bg_value: int = 64,
    fg_color: tuple[int, int, int] = (180, 180, 180),
) -> np.ndarray:
    """Create a (num_frames, H, W, 3) uint8 placeholder filled with centered text.

    Used by ``tile_and_save_media`` to render "Feature N/A" tiles when an optional input (backbone features, similarity
    neck, type neck, ...) is missing so the layout still has a tile in that position.
    """
    placeholder = np.full((num_frames, height, width, 3), bg_value, dtype=np.uint8)
    if num_frames == 0:
        return placeholder
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(height, width) / 400.0
    thickness = max(1, int(font_scale * 2))
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    template = placeholder[0].copy()
    cv2.putText(template, text, (text_x, text_y), font, font_scale, fg_color, thickness, cv2.LINE_AA)
    placeholder[:] = template[None, ...]
    return placeholder


def _normalize_video_frames(video_frames: np.ndarray) -> np.ndarray:
    """
    Normalize video frames to [0, 1] range using robust logic.

    Handles [0, 1], [-1, 1], and arbitrary ranges (e.g. ImageNet-normalized) so visualization is never all black due to
    wrong assumed range.

    Non-float64 input is normalized in float32. The frames are float32 model tensors, so float64 adds no precision, and
    at 93 frames of 720x1280x3 each float64 intermediate is a 2 GB allocation made inside a single numpy call.

    Args:
        video_frames: Video frames as numpy array (T, H, W, C)

    Returns:
        Normalized video frames in [0, 1] range, in the input dtype. Always a new array: the input may share storage
        with a caller-owned tensor, and callers write into the result.
    """
    vmin = float(np.min(video_frames))
    vmax = float(np.max(video_frames))

    work_dtype: type[np.floating[Any]] = np.float64 if video_frames.dtype == np.float64 else np.float32

    if vmin >= 0 and vmax <= 1.0:
        # Already in [0, 1], so clipping cannot change a value.
        return video_frames.copy()
    if vmin >= -1.1 and vmax <= 1.1:
        # Likely [-1, 1]
        out = (video_frames.astype(work_dtype) + 1) / 2
        return np.clip(out, 0, 1).astype(video_frames.dtype)

    # Arbitrary range (e.g. ImageNet): min-max scale to [0, 1] so image is visible
    span = vmax - vmin
    if span < 1e-9:
        out = np.full_like(video_frames, 0.5, dtype=work_dtype)
        return np.clip(out, 0, 1).astype(video_frames.dtype)
    out = (video_frames.astype(work_dtype) - vmin) / span
    return np.clip(out, 0, 1).astype(video_frames.dtype)


def tile_and_save_media(
    camera_images: np.ndarray,
    world_images: np.ndarray,
    camera_spatial_features: np.ndarray,
    world_spatial_features: np.ndarray,
    output_path: str | None = None,
    output_mode: TilingOutputMode = TilingOutputMode.IMAGE,
    frame_indices: list[int] | None = None,
    fps: float = 30.0,
    camera_patch_features: np.ndarray | None = None,
    camera_type_features: np.ndarray | None = None,
    object_type_overlay: np.ndarray | None = None,
    object_type_prediction: np.ndarray | None = None,
    object_type_masked_prediction: np.ndarray | None = None,
    similarity_mask: np.ndarray | None = None,
    camera_masked_by_similarity: np.ndarray | None = None,
    spatial_cosine_sim: np.ndarray | None = None,
    camera_backbone_features: np.ndarray | None = None,
    world_backbone_features: np.ndarray | None = None,
    use_gpu_encoding: bool = True,
) -> np.ndarray | None:
    """
    Create a tiled visualization showing camera, world, and spatial features.

    Layout selection:
    - **debug 3x3 format** (preferred when 3x3 is triggered) — the diagnostic layout for inspecting the encoder +
      alignment + PatchFormer pipeline:

      Row 1: input camera | input world | encoder-out cosine similarity (cam vs world)
      Row 2: camera backbone features | world backbone features | similarity neck (or "Patch Similarity Feature N/A")
      Row 3: camera alignment-final features | world alignment-final features | type neck (or "Patch Type Feature N/A")

      ``camera_backbone_features`` / ``world_backbone_features`` populate row 2 when provided (frozen pretrained
      backbone output, e.g. PE-Spatial-B16, ...). When unavailable they are rendered as "Backbone Feature N/A". Row 3 is
      populated from ``camera_spatial_features`` / ``world_spatial_features`` (existing semantics: the alignment-final
      encoder output).

    - 3x2 format: if camera_patch_features OR spatial_cosine_sim is provided (and the 3x3 trigger is not met)
      Row 1: camera image, world image
      Row 2: camera spatial features, world spatial features
      Row 3: patch features (or "N/A" placeholder), spatial cosine similarity heatmap
    - **3x3 with mask trio format** (2x3 + a row-3 mask trio): if the three ``similarity_mask`` /
      ``camera_masked_by_similarity`` / ``object_type_masked_prediction`` arrays are all provided alongside the 2x3
      trigger:

      Row 1: camera w/ text | similarity overlay     | object-type overlay
      Row 2: world w/ text  | similarity prediction  | object-type prediction
      Row 3: camera × mask  | similarity mask        | object-type × mask

      The mask is a per-pixel ``clamp(P(MATCH) + P(EXTRA) + P(WRONG), 0, 1)`` from the similarity head — the union of
      the three "model thinks an object is here" classes (computed in
      ``patch_vis_utils.create_combined_patch_visualization`` via ``CWIPInference.compute_patch_similarity_class_mask``
      with the canonical ``DEFAULT_PATCH_SIMILARITY_MASK_CLASSES``). Row 3 col 3 retains the object-type colormap; the
      mask only modulates its opacity. This layout is mode-agnostic — it activates whenever the row-3 trio is provided,
      regardless of whether the caller is running training, eval, or inference.
    - 2x3 format: if object_type_overlay and object_type_prediction are provided (and the row-3 mask trio is incomplete)
    - 2x2 format: default (neither of the above)

    Can output either a static image grid or a video file.

    Args:
        camera_images: (num_frames, H, W, 3) camera video frames
        world_images: (num_frames, H, W, 3) world video frames
        camera_spatial_features: (num_frames, H', W', 3) camera spatial features. Used by non-3x3 layouts and as the
            row-3 fallback when ``camera_alignment_features`` is None.
        world_spatial_features: same for world.
        output_path: Path to save output file (required for video mode)
        output_mode: TilingOutputMode.IMAGE for static grid, TilingOutputMode.VIDEO for video file
        frame_indices: Specific frame indices to use (if None, uses all frames)
        fps: Frames per second for video output (ignored for image mode)
        camera_patch_features: Optional (num_frames, H'', W'', 3) similarity neck features. In the debug 3x3 layout this
            populates row 2 col 3.
        camera_type_features: Optional (num_frames, H', W', 3) type neck features. In the debug 3x3 layout this
            populates row 3 col 3.
        object_type_overlay: Optional (num_frames, H, W, 3) object type overlay on camera for 2x3 layout
        object_type_prediction: Optional (num_frames, H, W, 3) object type prediction for 2x3 layout
        object_type_masked_prediction: Optional (num_frames, H, W, 3) object-type prediction already multiplied by the
            per-pixel similarity mask. Populates row 3 col 3 of the 3x3 with-mask-trio layout. Required (along with
            ``similarity_mask`` and ``camera_masked_by_similarity``) to trigger the 3x3 with-mask-trio layout from a 2x3
            base.
        similarity_mask: Optional (num_frames, H, W, 3) RGB-broadcast grayscale mask (per-pixel ``clamp(P(MATCH) +
            P(EXTRA) + P(WRONG), 0, 1)`` from the similarity head). Populates row 3 col 2 of the 3x3 with-mask-trio
            layout.
        camera_masked_by_similarity: Optional (num_frames, H, W, 3) camera frame multiplied by the same mask. Populates
            row 3 col 1 of the 3x3 with-mask-trio layout.
        spatial_cosine_sim: Optional (num_frames, H', W', 3) spatial cosine similarity heatmap. In the debug 3x3 layout
            this populates row 1 col 3.
        camera_backbone_features: Optional (num_frames, H', W', 3) camera frozen pretrained backbone features
            (pre-alignment; e.g. PE-Spatial output). When provided, debug 3x3 layout shows them in row 2 col 1.
        world_backbone_features: Same as ``camera_backbone_features`` for world stream.

    Returns:
        For IMAGE mode: tiled_image (H*rows, W*cols*num_frames, 3) array
        For VIDEO mode: None (saves video to output_path)
    """
    # Validate inputs
    if output_mode not in TilingOutputMode:
        raise ValueError(f"output_mode must be one of {list(TilingOutputMode)}, got '{output_mode}'")

    if output_mode == TilingOutputMode.VIDEO and output_path is None:
        raise ValueError("output_path is required for video mode")

    # Determine which frames to visualize
    total_frames = camera_images.shape[0]
    if frame_indices is None:
        # Use all available frames (arrays may already be subsetted)
        frame_indices = list(range(total_frames))

    # Get target size from camera images
    H, W = camera_images.shape[1:3]

    # Convert to uint8 using the global utility function
    camera_images_vis = image_video_utils.to_uint8(camera_images)
    world_images_vis = image_video_utils.to_uint8(world_images)
    camera_spatial_vis = image_video_utils.to_uint8(camera_spatial_features)
    world_spatial_vis = image_video_utils.to_uint8(world_spatial_features)

    # Extract subset of frames based on frame_indices
    # Check if frame_indices are within bounds of the actual arrays
    max_index = max(frame_indices) if frame_indices else 0
    if max_index >= total_frames:
        # If indices are out of bounds, use all available frames
        logger.warning(
            "Frame indices %s out of bounds for arrays with %s frames. Using all available frames.",
            frame_indices,
            total_frames,
        )
        frame_indices = list(range(total_frames))

    num_frames = len(frame_indices)
    camera_images_subset = camera_images_vis[frame_indices]
    world_images_subset = world_images_vis[frame_indices]
    camera_spatial_subset = camera_spatial_vis[frame_indices]
    world_spatial_subset = world_spatial_vis[frame_indices]

    # Determine layout based on provided features
    use_3x3_layout = camera_type_features is not None and (
        camera_patch_features is not None or spatial_cosine_sim is not None
    )
    use_3x2_layout = (not use_3x3_layout) and (camera_patch_features is not None or spatial_cosine_sim is not None)
    # 3x3-with-mask-trio layout = 2x3 trigger + the row-3 mask trio. All three must be present; if any is missing we
    # fall back to plain 2x3 so callers can opt in incrementally. This layout is mode-agnostic — gating is on the input
    # arrays, not on training/eval/inference.
    has_2x3_trigger = object_type_overlay is not None and object_type_prediction is not None
    has_mask_trio_row3 = (
        similarity_mask is not None
        and camera_masked_by_similarity is not None
        and object_type_masked_prediction is not None
    )
    use_mask_trio_3x3_layout = has_2x3_trigger and has_mask_trio_row3
    use_2x3_layout = has_2x3_trigger and not use_mask_trio_3x3_layout

    if (use_3x3_layout or use_3x2_layout) and (use_2x3_layout or use_mask_trio_3x3_layout):
        raise ValueError(
            "Cannot use both 3x2/3x3 layout (camera_patch_features or spatial_cosine_sim) and "
            "2x3 / 3x3-with-mask-trio layout (object_type features) simultaneously"
        )

    # Helper: turn an optional feature array into a uint8 (num_frames, H, W, 3) tile, or render a centered "<label> N/A"
    # placeholder when None.
    def _tile_or_na(feat: np.ndarray | None, label: str) -> np.ndarray:
        if feat is None:
            return _make_text_placeholder(f"{label} N/A", num_frames, H, W)
        feat_vis = image_video_utils.to_uint8(feat)
        return feat_vis[frame_indices]

    if use_3x2_layout:
        # 3x2 layout retains its legacy behavior: camera_patch_features in the bottom-left.
        patch_camera_subset = _tile_or_na(camera_patch_features, "Patch Features")

    if use_3x3_layout:
        # Debug 3x3 layout (rows 1/2/3 specs in the docstring). Build the six conditional tiles up front so the
        # VIDEO/IMAGE branches stay tidy.
        # Row 2 cols 1-2: frozen-backbone features (pre-alignment).
        camera_backbone_subset = _tile_or_na(camera_backbone_features, "Backbone Feature")
        world_backbone_subset = _tile_or_na(world_backbone_features, "Backbone Feature")
        # Row 3 cols 1-2: alignment-final features. We re-use the already-uint8 ``camera_spatial_subset`` /
        # ``world_spatial_subset`` produced above (alignment-final is the existing semantics of ``*_spatial_features``).
        camera_alignment_subset = camera_spatial_subset
        world_alignment_subset = world_spatial_subset
        # Row 2 col 3: similarity neck (camera_patch_features). Row 3 col 3: type neck.
        sim_neck_subset = _tile_or_na(camera_patch_features, "Patch Similarity Feature")
        type_neck_subset = _tile_or_na(camera_type_features, "Patch Type Feature")

    if use_2x3_layout or use_mask_trio_3x3_layout:
        # Both triggers require ``has_2x3_trigger``, which guarantees these two arrays are non-None.
        assert object_type_overlay is not None
        assert object_type_prediction is not None
        object_type_overlay_vis = image_video_utils.to_uint8(object_type_overlay)
        object_type_prediction_vis = image_video_utils.to_uint8(object_type_prediction)
        object_type_overlay_subset = object_type_overlay_vis[frame_indices]
        object_type_prediction_subset = object_type_prediction_vis[frame_indices]

    if use_mask_trio_3x3_layout:
        # Row 3 (left → right):
        #   col 1 = camera × mask, col 2 = mask itself (RGB-broadcast grayscale),
        #   col 3 = object-type prediction × mask.
        # This trigger requires ``has_mask_trio_row3``, which guarantees the trio arrays are non-None.
        assert similarity_mask is not None
        assert camera_masked_by_similarity is not None
        assert object_type_masked_prediction is not None
        similarity_mask_vis = image_video_utils.to_uint8(similarity_mask)
        camera_masked_vis = image_video_utils.to_uint8(camera_masked_by_similarity)
        object_type_masked_prediction_vis = image_video_utils.to_uint8(object_type_masked_prediction)
        similarity_mask_subset = similarity_mask_vis[frame_indices]
        camera_masked_subset = camera_masked_vis[frame_indices]
        object_type_masked_prediction_subset = object_type_masked_prediction_vis[frame_indices]

    # Handle spatial cosine similarity if provided
    if spatial_cosine_sim is not None:
        # Already in uint8 format from feature_vis_utils
        spatial_cosine_sim_subset = spatial_cosine_sim[frame_indices]
    else:
        spatial_cosine_sim_subset = None

    if output_mode == TilingOutputMode.VIDEO:
        # Video mode: Save directly to file using ffmpeg. ``output_path`` is validated as non-None for VIDEO mode above.
        assert output_path is not None
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Ensure all videos have the same number of frames
        if use_3x3_layout:
            num_frames_actual = min(
                camera_images_subset.shape[0],
                world_images_subset.shape[0],
                camera_backbone_subset.shape[0],
                world_backbone_subset.shape[0],
                camera_alignment_subset.shape[0],
                world_alignment_subset.shape[0],
                sim_neck_subset.shape[0],
                type_neck_subset.shape[0],
            )
            out_height = H * 3
            out_width = W * 3

            if spatial_cosine_sim_subset is not None:
                cosine_sim_resized = []
                for i in range(spatial_cosine_sim_subset.shape[0]):
                    resized = cv2.resize(spatial_cosine_sim_subset[i], (W, H), interpolation=cv2.INTER_LINEAR)
                    cosine_sim_resized.append(resized)
                cosine_sim_tile = np.stack(cosine_sim_resized, axis=0)
            else:
                cosine_sim_tile = _make_text_placeholder("Cosine Similarity N/A", num_frames_actual, H, W)

            # Debug 3x3 layout:
            #   Row 1: input camera | input world | encoder-output cosine similarity (cam vs world)
            #   Row 2: camera backbone features | world backbone features | similarity neck (or "N/A")
            #   Row 3: camera alignment-final | world alignment-final | type neck (or "N/A")
            # ``tiles`` is typed to match ``tiled_frames_generator``'s ``list[np.ndarray | None]`` param (list
            # invariance); this declaration covers every later ``tiles`` reassignment in this function.
            tiles: list[np.ndarray | None] = [
                camera_images_subset,
                world_images_subset,
                cosine_sim_tile,
                camera_backbone_subset,
                world_backbone_subset,
                sim_neck_subset,
                camera_alignment_subset,
                world_alignment_subset,
                type_neck_subset,
            ]
            frame_generator = image_video_utils.tiled_frames_generator(
                tiles, 3, 3, num_frames_actual, H, W, use_gpu=use_gpu_encoding
            )

        elif use_3x2_layout:
            num_frames_actual = min(
                camera_images_subset.shape[0],
                world_images_subset.shape[0],
                camera_spatial_subset.shape[0],
                world_spatial_subset.shape[0],
                patch_camera_subset.shape[0],
            )
            out_height = H * 3
            out_width = W * 2

            # Use spatial cosine similarity as 6th tile if provided, otherwise use white placeholder
            if spatial_cosine_sim_subset is not None:
                # Resize cosine sim heatmaps to match tile size (H, W)
                cosine_sim_resized = []
                for i in range(spatial_cosine_sim_subset.shape[0]):
                    resized = cv2.resize(spatial_cosine_sim_subset[i], (W, H), interpolation=cv2.INTER_LINEAR)
                    cosine_sim_resized.append(resized)
                cosine_sim_tile = np.stack(cosine_sim_resized, axis=0)  # (num_frames, H, W, 3)
            else:
                # White placeholder if no cosine sim provided
                cosine_sim_tile = np.ones((num_frames_actual, H, W, 3), dtype=np.uint8) * 255

            # Use 3x2 generator
            tiles = [
                camera_images_subset,
                world_images_subset,
                camera_spatial_subset,
                world_spatial_subset,
                patch_camera_subset,
                cosine_sim_tile,  # Spatial cosine similarity as 6th tile
            ]
            frame_generator = image_video_utils.tiled_frames_generator(
                tiles, 3, 2, num_frames_actual, H, W, use_gpu=use_gpu_encoding
            )
        elif use_mask_trio_3x3_layout:
            num_frames_actual = min(
                camera_images_subset.shape[0],
                world_images_subset.shape[0],
                camera_spatial_subset.shape[0],
                world_spatial_subset.shape[0],
                object_type_overlay_subset.shape[0],
                object_type_prediction_subset.shape[0],
                camera_masked_subset.shape[0],
                similarity_mask_subset.shape[0],
                object_type_masked_prediction_subset.shape[0],
            )
            out_height = H * 3
            out_width = W * 3

            # 3x3 with mask trio layout — 2x3 base plus a mask-gated bottom row.
            # Row 1: camera w/ text     | similarity overlay     | object-type overlay
            # Row 2: world w/ text      | similarity prediction  | object-type prediction
            # Row 3: camera × mask      | similarity mask        | object-type × mask
            tiles = [
                camera_images_subset,
                camera_spatial_subset,
                object_type_overlay_subset,
                world_images_subset,
                world_spatial_subset,
                object_type_prediction_subset,
                camera_masked_subset,
                similarity_mask_subset,
                object_type_masked_prediction_subset,
            ]
            frame_generator = image_video_utils.tiled_frames_generator(
                tiles, 3, 3, num_frames_actual, H, W, use_gpu=use_gpu_encoding
            )
        elif use_2x3_layout:
            num_frames_actual = min(
                camera_images_subset.shape[0],
                world_images_subset.shape[0],
                camera_spatial_subset.shape[0],
                world_spatial_subset.shape[0],
                object_type_overlay_subset.shape[0],
                object_type_prediction_subset.shape[0],
            )
            out_height = H * 2
            out_width = W * 3

            # Use 2x3 generator
            # Row 1: camera with text, similarity overlay, object type overlay
            # Row 2: world with text, similarity prediction, object type prediction
            tiles = [
                camera_images_subset,
                camera_spatial_subset,
                object_type_overlay_subset,
                world_images_subset,
                world_spatial_subset,
                object_type_prediction_subset,
            ]
            frame_generator = image_video_utils.tiled_frames_generator(
                tiles, 2, 3, num_frames_actual, H, W, use_gpu=use_gpu_encoding
            )
        else:
            num_frames_actual = min(
                camera_images_subset.shape[0],
                world_images_subset.shape[0],
                camera_spatial_subset.shape[0],
                world_spatial_subset.shape[0],
            )
            out_height = H * 2
            out_width = W * 2

            # Use 2x2 generator
            tiles = [camera_images_subset, world_images_subset, camera_spatial_subset, world_spatial_subset]
            frame_generator = image_video_utils.tiled_frames_generator(
                tiles, 2, 2, num_frames_actual, H, W, use_gpu=use_gpu_encoding
            )

        # Use the video writing function from image_video_utils
        logger.info(
            "[VIDEO] Tiled frame generator ready (%s frames -> %sx%s @ %sfps); handing off to ffmpeg writer: %s",
            num_frames_actual,
            out_width,
            out_height,
            fps,
            output_path,
        )
        image_video_utils.write_video_with_ffmpeg(frame_generator, output_path, out_width, out_height, fps)

        return None

    else:
        # Image mode: Create tiled frames and arrange in grid
        tiled_frames: list[np.ndarray] = []
        if use_3x3_layout:
            if spatial_cosine_sim_subset is not None:
                cosine_sim_resized = []
                for i in range(spatial_cosine_sim_subset.shape[0]):
                    resized = cv2.resize(spatial_cosine_sim_subset[i], (W, H), interpolation=cv2.INTER_LINEAR)
                    cosine_sim_resized.append(resized)
                cosine_sim_tile = np.stack(cosine_sim_resized, axis=0)
            else:
                cosine_sim_tile = _make_text_placeholder("Cosine Similarity N/A", num_frames, H, W)

            tiles = [
                camera_images_subset,
                world_images_subset,
                cosine_sim_tile,
                camera_backbone_subset,
                world_backbone_subset,
                sim_neck_subset,
                camera_alignment_subset,
                world_alignment_subset,
                type_neck_subset,
            ]
            frame_generator = image_video_utils.tiled_frames_generator(
                tiles, 3, 3, num_frames, H, W, use_gpu=use_gpu_encoding
            )

        elif use_3x2_layout:
            # Use spatial cosine similarity as 6th tile if provided, otherwise use white placeholder
            if spatial_cosine_sim_subset is not None:
                # Resize cosine sim heatmaps to match tile size (H, W)
                cosine_sim_resized = []
                for i in range(spatial_cosine_sim_subset.shape[0]):
                    resized = cv2.resize(spatial_cosine_sim_subset[i], (W, H), interpolation=cv2.INTER_LINEAR)
                    cosine_sim_resized.append(resized)
                cosine_sim_tile = np.stack(cosine_sim_resized, axis=0)  # (num_frames, H, W, 3)
            else:
                # White placeholder if no cosine sim provided
                cosine_sim_tile = np.ones((num_frames, H, W, 3), dtype=np.uint8) * 255

            tiles = [
                camera_images_subset,
                world_images_subset,
                camera_spatial_subset,
                world_spatial_subset,
                patch_camera_subset,
                cosine_sim_tile,  # Spatial cosine similarity as 6th tile
            ]
            frame_generator = image_video_utils.tiled_frames_generator(
                tiles, 3, 2, num_frames, H, W, use_gpu=use_gpu_encoding
            )
        elif use_mask_trio_3x3_layout:
            # 3x3 with mask trio layout — see VIDEO branch above for row-by-row spec.
            tiles = [
                camera_images_subset,
                camera_spatial_subset,
                object_type_overlay_subset,
                world_images_subset,
                world_spatial_subset,
                object_type_prediction_subset,
                camera_masked_subset,
                similarity_mask_subset,
                object_type_masked_prediction_subset,
            ]
            frame_generator = image_video_utils.tiled_frames_generator(
                tiles, 3, 3, num_frames, H, W, use_gpu=use_gpu_encoding
            )
        elif use_2x3_layout:
            # Row 1: camera with text, similarity overlay, object type overlay
            # Row 2: world with text, similarity prediction, object type prediction
            tiles = [
                camera_images_subset,
                camera_spatial_subset,
                object_type_overlay_subset,
                world_images_subset,
                world_spatial_subset,
                object_type_prediction_subset,
            ]
            frame_generator = image_video_utils.tiled_frames_generator(
                tiles, 2, 3, num_frames, H, W, use_gpu=use_gpu_encoding
            )
        else:
            tiles = [camera_images_subset, world_images_subset, camera_spatial_subset, world_spatial_subset]
            frame_generator = image_video_utils.tiled_frames_generator(
                tiles, 2, 2, num_frames, H, W, use_gpu=use_gpu_encoding
            )

        # Process tiled frames
        tiled_frames.extend(frame_generator)

        # Arrange tiled frames in a more square-like layout
        if len(tiled_frames) == 1:
            tiled_image = tiled_frames[0]
        else:
            num_frames_grid = len(tiled_frames)
            best_rows = int(np.floor(np.sqrt(num_frames_grid)))
            best_cols = int(np.ceil(num_frames_grid / best_rows))

            # Arrange frames in grid
            rows = []
            for row_idx in range(best_rows):
                start_idx = row_idx * best_cols
                end_idx = start_idx + best_cols
                row_frames = tiled_frames[start_idx:end_idx]
                if len(row_frames) > 1:
                    row_image = np.concatenate(row_frames, axis=1)
                else:
                    row_image = row_frames[0]
                rows.append(row_image)

            # Ensure all rows have the same width by padding shorter rows
            if len(rows) > 1:
                # Find the maximum width among all rows
                max_width = max(row.shape[1] for row in rows)

                # Pad shorter rows to match the maximum width
                padded_rows = []
                for row in rows:
                    if row.shape[1] < max_width:
                        # Create padding with the same height and channels as the row
                        padding_width = max_width - row.shape[1]
                        padding = np.zeros((row.shape[0], padding_width, row.shape[2]), dtype=row.dtype)
                        padded_row = np.concatenate([row, padding], axis=1)
                        padded_rows.append(padded_row)
                    else:
                        padded_rows.append(row)

                tiled_image = np.concatenate(padded_rows, axis=0)
            else:
                tiled_image = rows[0]

        return tiled_image


def extract_frames_from_data_batch(
    data_batch: dict[str, torch.Tensor],
) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
    """
    Extract frames from data batch.
    """
    # Callers must pass plain ``torch.Tensor`` images (unwrap any nested batch structure before calling).
    camera_tensor = data_batch["camera_image"].detach().cpu()
    world_tensor = data_batch["world_image"].detach().cpu()

    # Convert bfloat16 to float32 for numpy compatibility
    if camera_tensor.dtype == torch.bfloat16:
        camera_tensor = camera_tensor.to(torch.float32)
    if world_tensor.dtype == torch.bfloat16:
        world_tensor = world_tensor.to(torch.float32)

    camera_frames = camera_tensor.numpy()
    world_frames = world_tensor.numpy()

    # Determine video frame rate - try to get from data batch metadata or use default
    camera_fps = 30.0  # Default FPS
    if "fps" in data_batch:
        camera_fps = float(data_batch["fps"])
    elif "camera_fps" in data_batch:
        camera_fps = float(data_batch["camera_fps"])
    else:
        # Try to infer from video metadata if available
        if hasattr(data_batch, "get") and data_batch.get("video_metadata"):
            # ``video_metadata`` is dynamically typed at runtime (the dict-of-Tensor annotation is a lower bound);
            # annotate ``Any`` so the dict-metadata branch stays reachable.
            metadata: Any = data_batch["video_metadata"]
            if isinstance(metadata, dict) and "fps" in metadata:
                camera_fps = float(metadata["fps"])

    logger.info("Using camera FPS: %s for tiled video", camera_fps)

    # Handle image dimensions
    if camera_frames.ndim == 4 and world_frames.ndim == 4:
        if camera_frames.shape[1] == 3 and world_frames.shape[1] == 3:  # (batch, channels, height, width)
            camera_frames = np.transpose(camera_frames, (0, 2, 3, 1))
            world_frames = np.transpose(world_frames, (0, 2, 3, 1))
    else:
        logger.warning(
            "Unexpected camera image dimensions: %s or world image dimensions: %s",
            camera_frames.shape,
            world_frames.shape,
        )
        return None, None, None

    # Normalize images to [0, 1] using the shared utility function
    camera_frames = _normalize_video_frames(camera_frames)
    world_frames = _normalize_video_frames(world_frames)

    # Warn if frames are essentially black (helps diagnose wrong input range or failed load)
    if camera_frames.size > 0 and float(np.max(camera_frames)) < 0.01:
        logger.warning(
            "extract_frames_from_data_batch: camera_image normalized max < 0.01 (nearly black). "
            "Check that data_batch has pre-transform [0,1] or [-1,1] frames, not model-internal normalized tensors."
        )
    if world_frames.size > 0 and float(np.max(world_frames)) < 0.01:
        logger.warning(
            "extract_frames_from_data_batch: world_image normalized max < 0.01 (nearly black). "
            "Check that data_batch has pre-transform [0,1] or [-1,1] frames."
        )

    return camera_frames, world_frames, camera_fps


def save_similarity_matrix_with_frame_indices(
    similarity_matrix: np.ndarray,
    output_path: str,
    camera_start_frame: int | None = 0,
    world_start_frame: int | None = 0,
    similarity_score: float | None = None,
) -> None:
    """
    Save similarity matrix with proper frame indices on axes.

    Args:
        similarity_matrix: (N, N) similarity matrix (can be sparse with -1 off-diagonal)
        output_path: Path to save the image
        camera_start_frame: Starting frame index for camera
        world_start_frame: Starting frame index for world
        similarity_score: Average diagonal similarity score
    """

    # Create figure and axis
    _fig, ax = plt.subplots(figsize=(10, 10))

    # Create the heatmap with coolwarm colormap
    im = ax.imshow(similarity_matrix, cmap="coolwarm", aspect="equal", vmin=-1.0, vmax=1.0)

    # Set title
    if similarity_score is None:
        similarity_score = float(np.mean(np.diag(similarity_matrix)))
    ax.set_title(f"Similarity Matrix\nsimilarity score: {similarity_score:.4f}", fontsize=14, fontweight="bold")

    # Set axis labels
    ax.set_xlabel("Camera Frame Index", fontsize=12)
    ax.set_ylabel("World Frame Index", fontsize=12)

    # Calculate frame indices for ticks
    num_frames = similarity_matrix.shape[0]
    camera_end_frame = camera_start_frame + num_frames - 1
    world_end_frame = world_start_frame + num_frames - 1

    # Set custom tick positions and labels to show start and end frame indices
    # Show ticks at start, middle, and end
    tick_positions = [0, num_frames // 2, num_frames - 1]
    camera_tick_labels = [str(camera_start_frame), str(camera_start_frame + num_frames // 2), str(camera_end_frame)]
    world_tick_labels = [str(world_start_frame), str(world_start_frame + num_frames // 2), str(world_end_frame)]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(camera_tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(world_tick_labels)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Similarity", fontsize=12)

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=72, bbox_inches="tight")
    plt.close()

    logger.info("Similarity matrix saved to: %s", output_path)


def save_enhanced_diagonal_plots(
    similarity_matrix: np.ndarray, output_dir: str, iteration: int, camera_start_frame: int, world_start_frame: int
) -> None:
    """
    Create multiple enhanced visualizations of diagonal similarity scores.

    Args:
        similarity_matrix: (N, N) similarity matrix (can be sparse with -1 off-diagonal)
        output_dir: Directory to save plots
        iteration: Current iteration number
        camera_start_frame: Starting frame index for camera
        world_start_frame: Starting frame index for world
    """

    # Extract diagonal similarity scores
    diagonal_scores = np.diag(similarity_matrix)
    num_frames = len(diagonal_scores)

    # Calculate frame indices
    camera_frame_indices = np.arange(camera_start_frame, camera_start_frame + num_frames)
    world_frame_indices = np.arange(world_start_frame, world_start_frame + num_frames)
    frame_positions = np.arange(num_frames)

    # 1. Gradient-filled area plot
    create_gradient_area_plot(
        diagonal_scores,
        frame_positions,
        camera_frame_indices,
        world_frame_indices,
        os.path.join(output_dir, f"similarity_gradient_iter_{iteration:06d}.png"),
    )

    # 2. Heatmap strip visualization
    create_heatmap_strip(
        diagonal_scores,
        camera_frame_indices,
        world_frame_indices,
        os.path.join(output_dir, f"similarity_heatmap_strip_iter_{iteration:06d}.png"),
    )

    # 3. Circular/radial plot
    create_radial_similarity_plot(
        diagonal_scores,
        camera_frame_indices,
        world_frame_indices,
        os.path.join(output_dir, f"similarity_radial_iter_{iteration:06d}.png"),
    )

    # 4. Multi-panel dashboard
    create_similarity_dashboard(
        similarity_matrix,
        diagonal_scores,
        camera_frame_indices,
        world_frame_indices,
        os.path.join(output_dir, f"similarity_dashboard_iter_{iteration:06d}.png"),
    )


def create_gradient_area_plot(
    scores: np.ndarray,
    positions: np.ndarray,
    camera_indices: np.ndarray,
    world_indices: np.ndarray,
    output_path: str,
) -> None:
    """Create a gradient-filled area plot showing similarity over time."""

    _fig, ax = plt.subplots(figsize=(14, 6))

    # Use consistent coolwarm colormap
    cmap = plt.cm.coolwarm

    # Normalize scores to [0, 1] for colormap
    norm_scores = (scores + 1) / 2

    # Create gradient fill
    for i in range(len(positions) - 1):
        color = cmap(norm_scores[i])
        ax.fill_between([positions[i], positions[i + 1]], [scores[i], scores[i + 1]], alpha=0.7, color=color)

    # Add line on top
    ax.plot(positions, scores, "k-", linewidth=2, alpha=0.8)

    # Styling
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel("Similarity Score", fontsize=12, fontweight="bold")
    ax.set_title("Frame-to-Frame Similarity (Gradient View)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Dual x-axis
    tick_positions = [0, len(positions) // 4, len(positions) // 2, 3 * len(positions) // 4, len(positions) - 1]
    tick_positions = [pos for pos in tick_positions if pos < len(positions)]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(camera_indices[pos]) for pos in tick_positions])
    ax.set_xlabel("Camera Frame Index", fontsize=12)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([str(world_indices[pos]) for pos in tick_positions])
    ax2.set_xlabel("World Frame Index", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_heatmap_strip(
    scores: np.ndarray, camera_indices: np.ndarray, world_indices: np.ndarray, output_path: str
) -> None:
    """Create a horizontal heatmap strip showing similarity."""

    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 5), height_ratios=[1, 3])

    # Top: Heatmap strip
    scores_2d = scores.reshape(1, -1)
    im = ax1.imshow(scores_2d, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax1.set_yticks([])
    ax1.set_title("Similarity Heatmap Strip", fontsize=14, fontweight="bold", pad=15)

    # Set x-axis ticks for heatmap
    tick_positions = [0, len(scores) // 4, len(scores) // 2, 3 * len(scores) // 4, len(scores) - 1]
    tick_positions = [pos for pos in tick_positions if pos < len(scores)]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels([f"C:{camera_indices[pos]}\nW:{world_indices[pos]}" for pos in tick_positions])

    # Bottom: Line plot for reference
    ax2.plot(range(len(scores)), scores, color="#2166ac", linewidth=2)
    ax2.fill_between(range(len(scores)), scores, alpha=0.3, color="#2166ac")
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_ylabel("Similarity", fontsize=12)
    ax2.set_xlabel("Frame Index", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels([str(camera_indices[pos]) for pos in tick_positions])

    # Add colorbar positioned to the far right to avoid overlap
    cbar = plt.colorbar(im, ax=[ax1, ax2], orientation="vertical", pad=0.15, shrink=0.6)
    cbar.set_label("Similarity Score", fontsize=12)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_radial_similarity_plot(
    scores: np.ndarray, camera_indices: np.ndarray, world_indices: np.ndarray, output_path: str
) -> None:
    """Create a circular/radial plot showing similarity."""

    ax: Any
    _fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

    # Convert frame indices to angles
    angles = np.linspace(0, 2 * np.pi, len(scores), endpoint=False)

    # Normalize scores to positive values for radius (0 to 2)
    radii = scores + 1  # Now in range [0, 2]

    # Plot as scatter with varying colors and sizes
    scatter = ax.scatter(angles, radii, c=scores, cmap="coolwarm", s=50, alpha=0.8, vmin=-1, vmax=1)

    # Connect points with lines
    ax.plot(angles, radii, "k-", alpha=0.5, linewidth=1)

    # Fill area
    ax.fill(angles, radii, alpha=0.2, color="#2166ac")

    # Styling
    ax.set_ylim(0, 2)
    ax.set_title("Radial Similarity Plot\n(Distance from center = Similarity)", fontsize=14, fontweight="bold", pad=20)

    # Add radial grid labels
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.set_yticklabels(["-1", "-0.5", "0", "0.5", "1"])
    ax.set_ylabel("Similarity Score", fontsize=12)

    # Set angular tick labels to show frame numbers instead of degrees
    step = max(1, len(scores) // 12)  # Show ~12 labels max
    tick_angles = []
    tick_labels = []
    for i in range(0, len(scores), step):
        tick_angles.append(angles[i])
        tick_labels.append(f"F{i}")  # Frame number

    ax.set_thetagrids(np.degrees(tick_angles), tick_labels, fontsize=10)

    # Add detailed frame indices as annotations (show fewer to avoid clutter)
    annotation_step = max(1, len(scores) // 8)  # Show ~8 annotations max
    for i in range(0, len(scores), annotation_step):
        ax.annotate(
            f"C:{camera_indices[i]}\nW:{world_indices[i]}", xy=(angles[i], 2.2), ha="center", va="center", fontsize=8
        )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.1)
    cbar.set_label("Similarity Score", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _setup_radial_plot_common(
    per_frame_data: dict,
) -> tuple[list[str], list[str], np.ndarray, int, Any, Any]:
    """Common setup for radial plots - returns class names, colors, angles, num_frames, and figure."""
    # Get class names and colors from existing mapping
    class_names = [patch_type.name for patch_type in world_model_utils.PatchSimilarityType]
    class_colors = []

    # Convert RGB colors from PatchSimilarityType2Color to hex for matplotlib
    for i in range(len(class_names)):
        patch_type = world_model_utils.PatchSimilarityType(i)
        rgb_color = world_model_utils.PatchSimilarityType2Color[patch_type]  # (R, G, B) in range [0, 1]
        hex_color = f"#{int(rgb_color[0] * 255):02x}{int(rgb_color[1] * 255):02x}{int(rgb_color[2] * 255):02x}"
        class_colors.append(hex_color)

    # Get number of frames
    num_frames = len(next(iter(per_frame_data.values())))
    angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": "polar"})

    return class_names, class_colors, angles, num_frames, fig, ax


def create_radial_class_percentage_plot(
    per_frame_class_percentages: dict, output_path: str, class_percentages: dict | None = None
) -> None:
    """Create a circular/radial plot showing class percentages over time."""

    # Setup common elements
    class_names, class_colors, angles, num_frames, _fig, ax = _setup_radial_plot_common(per_frame_class_percentages)
    class_averages = []

    # Plot each class as a separate line
    for class_id, percentages in per_frame_class_percentages.items():
        if class_id < len(class_names) and class_id < len(class_colors):
            class_name = class_names[class_id]
            color = class_colors[class_id]

            # Use pre-computed average percentage if available, otherwise calculate it
            if class_percentages and class_id in class_percentages:
                avg_percentage = class_percentages[class_id]
            else:
                avg_percentage = sum(percentages) / len(percentages) if percentages else 0.0
            class_averages.append(avg_percentage)

            # Create label with average value
            label = f"{class_name} (avg: {avg_percentage:.1f}%)"

            # Close the polygon by appending first point to the end
            closed_angles = np.append(angles, angles[0])
            closed_percentages = np.append(percentages, percentages[0])

            # Plot the line
            ax.plot(closed_angles, closed_percentages, color=color, linewidth=2, label=label, alpha=0.8)

            # Fill area under the curve
            ax.fill(closed_angles, closed_percentages, color=color, alpha=0.2)

            # Add scatter points (without the closing point)
            ax.scatter(angles, percentages, color=color, s=30, alpha=0.7)

    # Styling
    ax.set_ylim(0, 100)
    ax.set_title(
        "Radial Class Percentage Plot\n(Distance from center = Percentage)", fontsize=14, fontweight="bold", pad=20
    )

    # Add radial grid labels
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Class Percentage", fontsize=12)

    # Set angular tick labels to show frame numbers
    step = max(1, num_frames // 12)  # Show ~12 labels max
    tick_angles = []
    tick_labels = []
    for i in range(0, num_frames, step):
        tick_angles.append(angles[i])
        tick_labels.append(f"F{i}")

    ax.set_thetagrids(np.degrees(tick_angles), tick_labels, fontsize=10)

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=10)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_radial_class_pixel_count_plot(per_frame_class_pixel_counts: dict, output_path: str) -> None:
    """Create a circular/radial plot showing class pixel counts over time."""

    # Setup common elements
    class_names, class_colors, angles, num_frames, _fig, ax = _setup_radial_plot_common(per_frame_class_pixel_counts)
    class_averages = []

    # Calculate max pixel count for scaling
    max_pixels = 0
    for class_id, pixel_counts in per_frame_class_pixel_counts.items():
        if pixel_counts:
            max_pixels = max(max_pixels, max(pixel_counts))

    # Set a reasonable upper limit for the plot (round up to nearest 100)
    plot_max = ((max_pixels // 100) + 1) * 100 if max_pixels > 0 else 100

    # Calculate total non-dont-care pixel counts for each frame
    total_non_dont_care_pixels = []

    for frame_idx in range(num_frames):
        frame_total = 0
        for class_id, pixel_counts in per_frame_class_pixel_counts.items():
            if class_id > 0 and frame_idx < len(pixel_counts):  # Skip DONT_CARE (class 0)
                frame_total += pixel_counts[frame_idx]
        total_non_dont_care_pixels.append(frame_total)

    # Update max_pixels to include total non-dont-care pixels
    if total_non_dont_care_pixels:
        max_total_pixels = max(total_non_dont_care_pixels)
        max_pixels = max(max_pixels, max_total_pixels)
        # Recalculate plot_max with updated max_pixels
        plot_max = ((max_pixels // 100) + 1) * 100 if max_pixels > 0 else 100

    # Plot each class as a separate line (exclude DONT_CARE)
    for class_id, pixel_counts in per_frame_class_pixel_counts.items():
        if class_id > 0 and class_id < len(class_names) and class_id < len(class_colors):  # Skip DONT_CARE (class 0)
            class_name = class_names[class_id]
            color = class_colors[class_id]

            # Calculate average pixel count
            avg_pixels = sum(pixel_counts) / len(pixel_counts) if pixel_counts else 0.0
            class_averages.append(avg_pixels)

            # Create label with average value
            label = f"{class_name} (avg: {avg_pixels:.0f}px)"

            # Close the polygon by appending first point to the end
            closed_angles = np.append(angles, angles[0])
            closed_pixel_counts = np.append(pixel_counts, pixel_counts[0])

            # Plot the line
            ax.plot(closed_angles, closed_pixel_counts, color=color, linewidth=2, label=label, alpha=0.8)

            # Fill area under the curve
            ax.fill(closed_angles, closed_pixel_counts, color=color, alpha=0.2)

            # Add scatter points (without the closing point)
            ax.scatter(angles, pixel_counts, color=color, s=30, alpha=0.7)

    # Plot total non-dont-care pixel count as a thick black line
    if total_non_dont_care_pixels:
        avg_total_pixels = sum(total_non_dont_care_pixels) / len(total_non_dont_care_pixels)
        total_label = f"All Non-Don't-Care (avg: {avg_total_pixels:.0f}px)"

        # Close the polygon for the total line
        closed_angles_total = np.append(angles, angles[0])
        closed_total_pixels = np.append(total_non_dont_care_pixels, total_non_dont_care_pixels[0])

        # Plot total line with thick black line
        ax.plot(
            closed_angles_total,
            closed_total_pixels,
            color="black",
            linewidth=4,
            label=total_label,
            alpha=0.9,
            linestyle="-",
        )

        # Add scatter points for total (without the closing point)
        ax.scatter(angles, total_non_dont_care_pixels, color="black", s=50, alpha=0.9, marker="o")

    # Styling
    ax.set_ylim(0, plot_max)
    ax.set_title(
        "Radial Class Pixel Count Plot\n(Distance from center = Pixel Count)", fontsize=14, fontweight="bold", pad=20
    )

    # Add radial grid labels
    step = plot_max // 4
    ax.set_yticks([0, step, step * 2, step * 3, plot_max])
    ax.set_yticklabels(["0px", f"{step}px", f"{step * 2}px", f"{step * 3}px", f"{plot_max}px"])
    ax.set_ylabel("Pixel Count", fontsize=12)

    # Set angular tick labels to show frame numbers
    step = max(1, num_frames // 12)  # Show ~12 labels max
    tick_angles = []
    tick_labels = []
    for i in range(0, num_frames, step):
        tick_angles.append(angles[i])
        tick_labels.append(f"F{i}")

    ax.set_thetagrids(np.degrees(tick_angles), tick_labels, fontsize=10)

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=10)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def create_similarity_dashboard(
    similarity_matrix: np.ndarray | None,
    diagonal_scores: np.ndarray,
    camera_indices: np.ndarray,
    world_indices: np.ndarray,
    output_path: str,
) -> None:
    """Create a comprehensive dashboard with multiple views."""

    # Handle None similarity matrix
    if similarity_matrix is None:
        logger.warning("Similarity matrix is None, skipping similarity dashboard")
        return

    fig = plt.figure(figsize=(16, 12))

    # Create grid layout
    gs = fig.add_gridspec(3, 3, height_ratios=[2, 1, 2], width_ratios=[2, 2, 1])

    # 1. Main similarity matrix (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(similarity_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax1.set_title("Full Similarity Matrix", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Camera Frames")
    ax1.set_ylabel("World Frames")

    # 2. Diagonal line plot (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(diagonal_scores, color="#2166ac", linewidth=2, marker="o", markersize=3)
    ax2.fill_between(range(len(diagonal_scores)), diagonal_scores, alpha=0.3, color="#2166ac")
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_title("Diagonal Similarity", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Similarity")
    ax2.grid(True, alpha=0.3)

    # 3. Colorbar (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    cbar = plt.colorbar(im1, cax=ax3)
    cbar.set_label("Similarity Score", fontsize=12)

    # 4. Heatmap strip (middle, spanning all columns)
    ax4 = fig.add_subplot(gs[1, :])
    scores_2d = diagonal_scores.reshape(1, -1)
    ax4.imshow(scores_2d, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax4.set_yticks([])
    ax4.set_title("Similarity Timeline", fontsize=12, fontweight="bold")

    # 5. Statistics panel (bottom-left)
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis("off")

    # Calculate statistics
    mean_sim = np.mean(diagonal_scores)
    std_sim = np.std(diagonal_scores)
    min_sim = np.min(diagonal_scores)
    max_sim = np.max(diagonal_scores)

    stats_text = f"""
    Statistics:
    Mean: {mean_sim:.4f}
    Std:  {std_sim:.4f}
    Min:  {min_sim:.4f}
    Max:  {max_sim:.4f}

    Frame Range:
    Camera: {camera_indices[0]}-{camera_indices[-1]}
    World:  {world_indices[0]}-{world_indices[-1]}
    Total Frames: {len(diagonal_scores)}
    """

    ax5.text(
        0.1,
        0.9,
        stats_text,
        transform=ax5.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.8},
    )

    # 6. Histogram (bottom-middle)
    ax6 = fig.add_subplot(gs[2, 1])
    # Calculate adaptive number of bins based on data range
    data_range = max_sim - min_sim
    num_data_points = len(diagonal_scores)
    # Check if we can create a histogram (need non-zero range and multiple data points)
    if data_range > 1e-10 and num_data_points > 1:
        # Use fewer bins if range is small, but at least 5 and at most 30
        num_bins = min(30, max(5, int(num_data_points / 2)))
        # If range is very small, use even fewer bins
        if data_range < 0.01:
            num_bins = min(10, max(2, num_data_points))
        try:
            ax6.hist(diagonal_scores, bins=num_bins, alpha=0.7, color="#2166ac", edgecolor="black")
        except ValueError:
            # Fallback: if histogram still fails, try with fewer bins or just show text
            try:
                ax6.hist(diagonal_scores, bins=1, alpha=0.7, color="#2166ac", edgecolor="black")
            except ValueError:
                ax6.text(0.5, 0.5, "All values are identical", ha="center", va="center", transform=ax6.transAxes)
    else:
        # All values are the same or only one data point
        ax6.text(0.5, 0.5, "Insufficient data for histogram", ha="center", va="center", transform=ax6.transAxes)
    ax6.set_xlabel("Similarity Score")
    ax6.set_ylabel("Frequency")
    ax6.set_title("Similarity Distribution", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Similarity Score")
    ax6.set_ylabel("Frequency")
    ax6.set_title("Similarity Distribution", fontsize=12, fontweight="bold")
    ax6.axvline(mean_sim, color="red", linestyle="--", label=f"Mean: {mean_sim:.3f}")
    ax6.legend()

    # 7. Moving average (bottom-right)
    ax7 = fig.add_subplot(gs[2, 2])
    window_size = max(5, len(diagonal_scores) // 20)
    if len(diagonal_scores) >= window_size:
        moving_avg = np.convolve(diagonal_scores, np.ones(window_size) / window_size, mode="valid")
        # Create x-axis that matches the length of moving_avg
        x_moving = np.arange(len(moving_avg))
        ax7.plot(x_moving, moving_avg, color="#2166ac", linewidth=2)
        ax7.set_title(f"Moving Avg (w={window_size})", fontsize=10, fontweight="bold")
        ax7.set_ylim(-1.1, 1.1)
        ax7.tick_params(axis="both", which="major", labelsize=8)

    plt.suptitle("Similarity Analysis Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Enhanced diagonal similarity plots saved to: %s", output_path)


def dump_per_frame_patch_scores(
    per_frame_class_percentages: dict,
    output_path: str,
    camera_start_frame: int = 0,
    world_start_frame: int = 0,
    format: str = "csv",  # "csv", "json", "npy"
    patch_results: dict | None = None,  # Additional patch results for pixel counts and dimensions
    data_batch: dict | None = None,  # Data batch for camera/world frame dimensions and video paths
) -> None:
    """
    Dump per-frame patch classification scores to a file for later analysis.

    The patch scores represent the percentage of each patch similarity class
    (e.g., IDENTICAL, SIMILAR, DIFFERENT, OPPOSITE) for each frame.

    Args:
        per_frame_class_percentages: Dict mapping class_id to list of percentages per frame
        output_path: Path to save the patch scores file
        camera_start_frame: Starting frame index for camera
        world_start_frame: Starting frame index for world
        format: Output format - "csv", "json", or "npy"
        patch_results: Additional patch results for pixel counts and dimensions
        data_batch: Data batch containing video paths and frame dimensions
    """
    if not per_frame_class_percentages:
        logger.warning("Per-frame class percentages is empty, cannot dump patch scores")
        return

    # Extract video paths from data_batch
    camera_video_path = None
    world_video_path = None
    if data_batch is not None:
        # Safely extract camera video path
        if data_batch.get("camera_video_paths"):
            camera_video_path = (
                data_batch["camera_video_paths"][0] if len(data_batch["camera_video_paths"]) > 0 else None
            )

        # Safely extract world video path
        if data_batch.get("world_video_paths"):
            world_video_path = data_batch["world_video_paths"][0] if len(data_batch["world_video_paths"]) > 0 else None

    # Log video paths for debugging
    if camera_video_path is not None:
        logger.info("Camera video path: %s", camera_video_path)
    else:
        logger.warning("Camera video path not found in data_batch")

    if world_video_path is not None:
        logger.info("World video path: %s", world_video_path)
    else:
        logger.warning("World video path not found in data_batch")

    # Get number of frames from the first class
    num_frames = len(next(iter(per_frame_class_percentages.values())))

    # Calculate frame indices
    camera_frame_indices = np.arange(camera_start_frame, camera_start_frame + num_frames)
    world_frame_indices = np.arange(world_start_frame, world_start_frame + num_frames)

    # Get class names from PatchSimilarityType enum
    class_names = [patch_type.name for patch_type in world_model_utils.PatchSimilarityType]

    # Get pre-computed per-frame pixel counts from patch_results
    per_frame_class_pixel_counts = {}
    if patch_results is not None and "per_frame_class_pixel_counts" in patch_results:
        per_frame_class_pixel_counts = patch_results["per_frame_class_pixel_counts"]

    # Get dimension information
    camera_frame_height, camera_frame_width = None, None
    world_frame_height, world_frame_width = None, None
    pred_probabilities_height, pred_probabilities_width = None, None

    if data_batch is not None:
        # Get camera frame dimensions - take last two dims as H, W
        if "camera_image" in data_batch:
            camera_frames = data_batch["camera_image"]
            if len(camera_frames.shape) >= 2:
                camera_frame_height, camera_frame_width = camera_frames.shape[-2], camera_frames.shape[-1]

        # Get world frame dimensions - take last two dims as H, W
        if "world_image" in data_batch:
            world_frames = data_batch["world_image"]
            if len(world_frames.shape) >= 2:
                world_frame_height, world_frame_width = world_frames.shape[-2], world_frames.shape[-1]

    if patch_results is not None and "pred_probabilities" in patch_results:
        pred_probabilities = patch_results["pred_probabilities"]  # (B, H, W)
        if len(pred_probabilities.shape) >= 2:
            pred_probabilities_height, pred_probabilities_width = (
                pred_probabilities.shape[-2],
                pred_probabilities.shape[-1],
            )

    # Get all three versions of percentages if available from patch_results
    per_frame_class_percentages_all_classes = {}
    per_frame_class_percentages_no_extra = {}
    if patch_results is not None:
        if "per_frame_class_percentages_all_classes" in patch_results:
            per_frame_class_percentages_all_classes = patch_results["per_frame_class_percentages_all_classes"]
        if "per_frame_class_percentages_no_extra" in patch_results:
            per_frame_class_percentages_no_extra = patch_results["per_frame_class_percentages_no_extra"]

    # Calculate statistics for each class from per-frame data (default version)
    class_statistics = {}
    for class_id, percentages in per_frame_class_percentages.items():
        if percentages:
            class_statistics[class_id] = {
                "mean": float(np.mean(percentages)),
                "std": float(np.std(percentages)),
                "min": float(np.min(percentages)),
                "max": float(np.max(percentages)),
            }
        else:
            class_statistics[class_id] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    # Calculate statistics for all_classes version
    class_statistics_all_classes = {}
    if per_frame_class_percentages_all_classes:
        for class_id, percentages in per_frame_class_percentages_all_classes.items():
            if percentages:
                class_statistics_all_classes[class_id] = {
                    "mean": float(np.mean(percentages)),
                    "std": float(np.std(percentages)),
                    "min": float(np.min(percentages)),
                    "max": float(np.max(percentages)),
                }
            else:
                class_statistics_all_classes[class_id] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    # Calculate statistics for no_extra version
    class_statistics_no_extra = {}
    if per_frame_class_percentages_no_extra:
        for class_id, percentages in per_frame_class_percentages_no_extra.items():
            if percentages:
                class_statistics_no_extra[class_id] = {
                    "mean": float(np.mean(percentages)),
                    "std": float(np.std(percentages)),
                    "min": float(np.min(percentages)),
                    "max": float(np.max(percentages)),
                }
            else:
                class_statistics_no_extra[class_id] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    logger.info("Dumping per-frame patch scores for %s and %s:", camera_video_path, world_video_path)
    logger.info("  - Total frames: %s", num_frames)
    logger.info(
        "  - Frame range: camera %s-%s, world %s-%s",
        camera_start_frame,
        camera_start_frame + num_frames - 1,
        world_start_frame,
        world_start_frame + num_frames - 1,
    )
    logger.info("  - Classes: %s", list(per_frame_class_percentages.keys()))
    for class_id, stats in class_statistics.items():
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        logger.info(
            "  - %s: mean=%.2f%%, std=%.2f%%, min=%.2f%%, max=%.2f%%",
            class_name,
            stats["mean"],
            stats["std"],
            stats["min"],
            stats["max"],
        )

    if format.lower() == "csv":
        # Save as CSV with frame indices, class percentages, and pixel counts
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Create header with all class names for both percentages and pixel counts
            header = ["frame_index", "camera_frame", "world_frame"]
            for class_id in sorted(per_frame_class_percentages.keys()):
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                header.append(f"{class_name}_percentage")
                header.append(f"{class_name}_pixel_count")
            writer.writerow(header)

            # Write data for each frame
            for i, (cam_idx, world_idx) in enumerate(zip(camera_frame_indices, world_frame_indices)):
                row: list[float] = [i, int(cam_idx), int(world_idx)]
                for class_id in sorted(per_frame_class_percentages.keys()):
                    # Add percentage
                    if i < len(per_frame_class_percentages[class_id]):
                        row.append(float(per_frame_class_percentages[class_id][i]))
                    else:
                        row.append(0.0)

                    # Add pixel count
                    if class_id in per_frame_class_pixel_counts and i < len(per_frame_class_pixel_counts[class_id]):
                        row.append(int(per_frame_class_pixel_counts[class_id][i]))
                    else:
                        row.append(0)
                writer.writerow(row)

        logger.info("Patch scores saved to CSV: %s", output_path)

    elif format.lower() == "json":
        # Extract additional information from patch_results
        total_non_dont_care_pixels = None
        total_non_dont_care_pixels_all_classes = None
        total_non_dont_care_pixels_no_extra = None
        total_non_dont_care_pixels_gt_fg = None
        total_pixels = None
        class_counts = None
        class_percentages = None
        class_percentages_all_classes = None
        class_percentages_no_extra = None

        # Per-frame denominators
        per_frame_denominators = None
        per_frame_denominators_all_classes = None
        per_frame_denominators_no_extra = None
        per_frame_denominators_gt_fg = None

        if patch_results is not None:
            total_non_dont_care_pixels = patch_results.get("total_non_dont_care_pixels", None)
            total_non_dont_care_pixels_all_classes = patch_results.get("total_non_dont_care_pixels_all_classes", None)
            total_non_dont_care_pixels_no_extra = patch_results.get("total_non_dont_care_pixels_no_extra", None)
            total_non_dont_care_pixels_gt_fg = patch_results.get("total_non_dont_care_pixels_gt_fg", None)
            total_pixels = patch_results.get("total_pixels", None)

            # Get per-frame denominators
            per_frame_denominators = patch_results.get("per_frame_denominators", None)
            per_frame_denominators_all_classes = patch_results.get("per_frame_denominators_all_classes", None)
            per_frame_denominators_no_extra = patch_results.get("per_frame_denominators_no_extra", None)
            per_frame_denominators_gt_fg = patch_results.get("per_frame_denominators_gt_fg", None)

            # Get class counts (tensor) and convert to dict
            if "class_counts" in patch_results:
                cc = patch_results["class_counts"]
                class_counts = {
                    str(i): int(cc[i].item()) if hasattr(cc[i], "item") else int(cc[i]) for i in range(len(cc))
                }

            # Get overall class percentages (all frames averaged)
            class_percentages = patch_results.get("class_percentages", None)
            class_percentages_all_classes = patch_results.get("class_percentages_all_classes", None)
            class_percentages_no_extra = patch_results.get("class_percentages_no_extra", None)

        # Save as JSON with metadata including all three versions
        data = {
            "camera_video_path": camera_video_path,
            "world_video_path": world_video_path,
            "metadata": {
                "total_frames": int(num_frames),
                "camera_start_frame": int(camera_start_frame),
                "world_start_frame": int(world_start_frame),
                "camera_end_frame": int(camera_start_frame + num_frames - 1),
                "world_end_frame": int(world_start_frame + num_frames - 1),
                "class_statistics": class_statistics,  # Default: GT fg if available, else no_extra
                "class_statistics_all_classes": class_statistics_all_classes if class_statistics_all_classes else None,
                "class_statistics_no_extra": class_statistics_no_extra if class_statistics_no_extra else None,
                "class_names": {
                    str(class_id): class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                    for class_id in per_frame_class_percentages
                },
                "camera_video_path": camera_video_path,
                "world_video_path": world_video_path,
                "dimensions": {
                    "camera_frame_height": camera_frame_height,
                    "camera_frame_width": camera_frame_width,
                    "world_frame_height": world_frame_height,
                    "world_frame_width": world_frame_width,
                    "pred_probabilities_height": pred_probabilities_height,
                    "pred_probabilities_width": pred_probabilities_width,
                },
                "denominators": {
                    "total_pixels": int(total_pixels) if total_pixels is not None else None,
                    "total_non_dont_care_pixels": int(total_non_dont_care_pixels)
                    if total_non_dont_care_pixels is not None
                    else None,
                    "total_non_dont_care_pixels_all_classes": int(total_non_dont_care_pixels_all_classes)
                    if total_non_dont_care_pixels_all_classes is not None
                    else None,
                    "total_non_dont_care_pixels_no_extra": int(total_non_dont_care_pixels_no_extra)
                    if total_non_dont_care_pixels_no_extra is not None
                    else None,
                    "total_non_dont_care_pixels_gt_fg": int(total_non_dont_care_pixels_gt_fg)
                    if total_non_dont_care_pixels_gt_fg is not None
                    else None,
                    "description": {
                        "total_pixels": "Total pixels including DONT_CARE class",
                        "total_non_dont_care_pixels": (
                            "Denominator for default percentages (configurable, defaults to no_extra)"
                        ),
                        "total_non_dont_care_pixels_all_classes": (
                            "Denominator for all_classes percentages (MATCH + WRONG + EXTRA + MISSING)"
                        ),
                        "total_non_dont_care_pixels_no_extra": (
                            "Denominator for no_extra percentages (MATCH + WRONG + MISSING)"
                        ),
                        "total_non_dont_care_pixels_gt_fg": (
                            "Denominator for gt_fg percentages (GT foreground pixel count)"
                        ),
                    },
                },
                "class_counts": class_counts,
                "overall_class_percentages": {
                    "default": class_percentages,
                    "all_classes": class_percentages_all_classes,
                    "no_extra": class_percentages_no_extra,
                    "description": {
                        "default": (
                            "Averaged across all frames using configurable default denominator (defaults to no_extra)"
                        ),
                        "all_classes": "Averaged across all frames using all non-DONT_CARE classes as denominator",
                        "no_extra": "Averaged across all frames excluding EXTRA from denominator",
                    },
                },
            },
            "per_frame_scores": [
                {
                    "frame_index": int(i),
                    "camera_frame": int(cam_idx),
                    "world_frame": int(world_idx),
                    "class_percentages": {
                        str(class_id): float(per_frame_class_percentages[class_id][i])
                        if i < len(per_frame_class_percentages[class_id])
                        else 0.0
                        for class_id in per_frame_class_percentages
                    },
                    "class_percentages_all_classes": {
                        str(class_id): float(per_frame_class_percentages_all_classes[class_id][i])
                        if class_id in per_frame_class_percentages_all_classes
                        and i < len(per_frame_class_percentages_all_classes[class_id])
                        else 0.0
                        for class_id in per_frame_class_percentages
                    }
                    if per_frame_class_percentages_all_classes
                    else None,
                    "class_percentages_no_extra": {
                        str(class_id): float(per_frame_class_percentages_no_extra[class_id][i])
                        if class_id in per_frame_class_percentages_no_extra
                        and i < len(per_frame_class_percentages_no_extra[class_id])
                        else 0.0
                        for class_id in per_frame_class_percentages
                    }
                    if per_frame_class_percentages_no_extra
                    else None,
                    "class_pixel_counts": {
                        str(class_id): int(per_frame_class_pixel_counts[class_id][i])
                        if class_id in per_frame_class_pixel_counts and i < len(per_frame_class_pixel_counts[class_id])
                        else 0
                        for class_id in per_frame_class_percentages
                    },
                    "denominators": {
                        "total_pixels": int(per_frame_denominators[i])
                        if per_frame_denominators is not None and i < len(per_frame_denominators)
                        else None,
                        "total_pixels_all_classes": int(per_frame_denominators_all_classes[i])
                        if per_frame_denominators_all_classes is not None
                        and i < len(per_frame_denominators_all_classes)
                        else None,
                        "total_pixels_no_extra": int(per_frame_denominators_no_extra[i])
                        if per_frame_denominators_no_extra is not None and i < len(per_frame_denominators_no_extra)
                        else None,
                        "total_pixels_gt_fg": int(per_frame_denominators_gt_fg[i])
                        if per_frame_denominators_gt_fg is not None and i < len(per_frame_denominators_gt_fg)
                        else None,
                    },
                }
                for i, (cam_idx, world_idx) in enumerate(zip(camera_frame_indices, world_frame_indices))
            ],
        }

        with open(output_path, "w") as jsonfile:
            json.dump(data, jsonfile, indent=2)

        logger.info("Patch scores saved to JSON: %s", output_path)

    elif format.lower() == "npy":
        # Save as numpy array with metadata in a separate file
        # Create a structured array with all class percentages and pixel counts
        structured_data = []
        for i in range(num_frames):
            frame_data = {
                "frame_index": i,
                "camera_frame": camera_frame_indices[i],
                "world_frame": world_frame_indices[i],
            }
            for class_id in sorted(per_frame_class_percentages.keys()):
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                frame_data[f"{class_name}_percentage"] = (
                    float(per_frame_class_percentages[class_id][i])
                    if i < len(per_frame_class_percentages[class_id])
                    else 0.0
                )
                frame_data[f"{class_name}_pixel_count"] = (
                    int(per_frame_class_pixel_counts[class_id][i])
                    if class_id in per_frame_class_pixel_counts and i < len(per_frame_class_pixel_counts[class_id])
                    else 0
                )
            structured_data.append(frame_data)

        # Save as numpy structured array (object array of per-frame dicts, matching np.save's implicit cast)
        np.save(output_path, np.array(structured_data, dtype=object))

        # Save metadata separately
        metadata_path = output_path.replace(".npy", "_metadata.json")
        metadata = {
            "camera_video_path": camera_video_path,
            "world_video_path": world_video_path,
            "total_frames": int(num_frames),
            "camera_start_frame": int(camera_start_frame),
            "world_start_frame": int(world_start_frame),
            "camera_end_frame": int(camera_start_frame + num_frames - 1),
            "world_end_frame": int(world_start_frame + num_frames - 1),
            "class_statistics": class_statistics,
            "class_names": {
                str(class_id): class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                for class_id in per_frame_class_percentages
            },
            "camera_frame_indices": camera_frame_indices.tolist(),
            "world_frame_indices": world_frame_indices.tolist(),
            "dimensions": {
                "camera_frame_height": camera_frame_height,
                "camera_frame_width": camera_frame_width,
                "world_frame_height": world_frame_height,
                "world_frame_width": world_frame_width,
                "pred_probabilities_height": pred_probabilities_height,
                "pred_probabilities_width": pred_probabilities_width,
            },
        }

        with open(metadata_path, "w") as jsonfile:
            json.dump(metadata, jsonfile, indent=2)

        logger.info("Patch scores saved to NPY: %s", output_path)
        logger.info("Metadata saved to: %s", metadata_path)

    else:
        raise ValueError(f"Unsupported format: {format}. Supported formats: csv, json, npy")


def dump_per_frame_consistency_scores(
    similarity_matrix: np.ndarray | None,
    output_path: str,
    camera_start_frame: int = 0,
    world_start_frame: int = 0,
    format: str = "csv",  # "csv", "json", "npy"
    data_batch: dict | None = None,  # Data batch for camera/world frame dimensions and video paths
) -> None:
    """
    Dump per-frame local consistency scores to a file for later analysis.

    The local consistency score for each frame is defined as the diagonal element
    of the similarity matrix, representing the frame-to-frame similarity between
    camera and world views.

    Args:
        similarity_matrix: (N, N) similarity matrix with diagonal elements as consistency scores
        output_path: Path to save the consistency scores file
        camera_start_frame: Starting frame index for camera
        world_start_frame: Starting frame index for world
        format: Output format - "csv", "json", or "npy"
        data_batch: Data batch for camera/world frame dimensions and video paths
    """
    if similarity_matrix is None:
        logger.warning("Similarity matrix is None, cannot dump consistency scores")
        return

    # Extract video paths from data_batch
    camera_video_path = None
    world_video_path = None
    if data_batch is not None:
        # Safely extract camera video path
        if data_batch.get("camera_video_paths"):
            camera_video_path = (
                data_batch["camera_video_paths"][0] if len(data_batch["camera_video_paths"]) > 0 else None
            )

        # Safely extract world video path
        if data_batch.get("world_video_paths"):
            world_video_path = data_batch["world_video_paths"][0] if len(data_batch["world_video_paths"]) > 0 else None

    # Log video paths for debugging
    if camera_video_path is not None:
        logger.info("Camera video path: %s", camera_video_path)
    else:
        logger.warning("Camera video path not found in data_batch")

    if world_video_path is not None:
        logger.info("World video path: %s", world_video_path)
    else:
        logger.warning("World video path not found in data_batch")

    # Extract diagonal elements as per-frame consistency scores
    consistency_scores = np.diag(similarity_matrix)
    num_frames = len(consistency_scores)

    # Calculate frame indices
    camera_frame_indices = np.arange(camera_start_frame, camera_start_frame + num_frames)
    world_frame_indices = np.arange(world_start_frame, world_start_frame + num_frames)

    # Calculate statistics
    mean_score = np.mean(consistency_scores)
    std_score = np.std(consistency_scores)
    min_score = np.min(consistency_scores)
    max_score = np.max(consistency_scores)

    logger.info("Dumping per-frame consistency scores for %s and %s:", camera_video_path, world_video_path)
    logger.info("  - Total frames: %s", num_frames)
    logger.info(
        "  - Frame range: camera %s-%s, world %s-%s",
        camera_start_frame,
        camera_start_frame + num_frames - 1,
        world_start_frame,
        world_start_frame + num_frames - 1,
    )
    logger.info(
        "  - Score statistics: mean=%.4f, std=%.4f, min=%.4f, max=%.4f", mean_score, std_score, min_score, max_score
    )

    if format.lower() == "csv":
        # Save as CSV with frame indices and scores
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(["frame_index", "camera_frame", "world_frame", "consistency_score"])
            # Write data
            for i, (cam_idx, world_idx, score) in enumerate(
                zip(camera_frame_indices, world_frame_indices, consistency_scores)
            ):
                writer.writerow([i, int(cam_idx), int(world_idx), float(score)])

        logger.info("Consistency scores saved to CSV: %s", output_path)

    elif format.lower() == "json":
        # Save as JSON with metadata
        data = {
            "camera_video_path": camera_video_path,
            "world_video_path": world_video_path,
            "metadata": {
                "total_frames": int(num_frames),
                "camera_start_frame": int(camera_start_frame),
                "world_start_frame": int(world_start_frame),
                "camera_end_frame": int(camera_start_frame + num_frames - 1),
                "world_end_frame": int(world_start_frame + num_frames - 1),
                "statistics": {
                    "mean": float(mean_score),
                    "std": float(std_score),
                    "min": float(min_score),
                    "max": float(max_score),
                },
            },
            "per_frame_scores": [
                {
                    "frame_index": int(i),
                    "camera_frame": int(cam_idx),
                    "world_frame": int(world_idx),
                    "consistency_score": float(score),
                }
                for i, (cam_idx, world_idx, score) in enumerate(
                    zip(camera_frame_indices, world_frame_indices, consistency_scores)
                )
            ],
        }

        with open(output_path, "w") as jsonfile:
            json.dump(data, jsonfile, indent=2)

        logger.info("Consistency scores saved to JSON: %s", output_path)

    elif format.lower() == "npy":
        # Save as numpy array with metadata in a separate file
        np.save(output_path, consistency_scores)

        # Save metadata separately
        metadata_path = output_path.replace(".npy", "_metadata.json")
        metadata = {
            "camera_video_path": camera_video_path,
            "world_video_path": world_video_path,
            "total_frames": int(num_frames),
            "camera_start_frame": int(camera_start_frame),
            "world_start_frame": int(world_start_frame),
            "camera_end_frame": int(camera_start_frame + num_frames - 1),
            "world_end_frame": int(world_start_frame + num_frames - 1),
            "statistics": {
                "mean": float(mean_score),
                "std": float(std_score),
                "min": float(min_score),
                "max": float(max_score),
            },
            "camera_frame_indices": camera_frame_indices.tolist(),
            "world_frame_indices": world_frame_indices.tolist(),
        }

        with open(metadata_path, "w") as jsonfile:
            json.dump(metadata, jsonfile, indent=2)

        logger.info("Consistency scores saved to NPY: %s", output_path)
        logger.info("Metadata saved to: %s", metadata_path)

    else:
        raise ValueError(f"Unsupported format: {format}. Supported formats: csv, json, npy")


def save_patch_predictions(patch_results: dict, output_dir: str, iteration: int) -> None:
    """
    Save pred_class_indices and pred_probabilities as .npy files.

    Args:
        patch_results: Patch prediction results containing pred_class_indices and pred_probabilities
        output_dir: Directory to save the .npy files
        iteration: Iteration number for file naming
    """
    if "pred_class_indices" not in patch_results or "pred_probabilities" not in patch_results:
        logger.warning("pred_class_indices or pred_probabilities not found in patch_results")
        return

    # Extract and convert to numpy
    # Detach first to remove gradients, then move to CPU
    pred_class_indices = patch_results["pred_class_indices"].detach().cpu()  # (B, H, W)
    pred_probabilities = patch_results["pred_probabilities"].detach().cpu()  # (B, H, W)

    # Convert to float32 to avoid BFloat16 issues with numpy
    pred_class_indices = pred_class_indices.float()
    pred_probabilities = pred_probabilities.float()

    # Save pred_class_indices and pred_probabilities as .npy files
    pred_class_indices_path = os.path.join(output_dir, f"pred_class_indices_iter_{iteration:06d}.npy")
    pred_probabilities_path = os.path.join(output_dir, f"pred_probabilities_iter_{iteration:06d}.npy")

    np.save(pred_class_indices_path, pred_class_indices.numpy())
    np.save(pred_probabilities_path, pred_probabilities.numpy())

    logger.info("Saved pred_class_indices to: %s", pred_class_indices_path)
    logger.info("Saved pred_probabilities to: %s", pred_probabilities_path)
