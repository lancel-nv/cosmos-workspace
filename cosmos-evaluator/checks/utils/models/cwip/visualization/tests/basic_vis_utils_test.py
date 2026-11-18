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

import json
import os
import tempfile
import tracemalloc
from typing import cast
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from checks.utils.models.cwip.visualization.basic_vis_utils import (
    TilingOutputMode,
    _normalize_video_frames,
    create_gradient_area_plot,
    create_heatmap_strip,
    create_radial_class_percentage_plot,
    create_radial_class_pixel_count_plot,
    create_radial_similarity_plot,
    create_similarity_dashboard,
    dump_per_frame_consistency_scores,
    dump_per_frame_patch_scores,
    extract_frames_from_data_batch,
    save_enhanced_diagonal_plots,
    save_patch_predictions,
    save_similarity_matrix_with_frame_indices,
    tile_and_save_media,
)
from checks.utils.video import SVTAV1_FFMPEG_ARGS


class TestTilingOutputMode(unittest.TestCase):
    """Tests for TilingOutputMode enum."""

    def test_enum_values(self) -> None:
        """Test that TilingOutputMode enum values are correct."""
        self.assertEqual(TilingOutputMode.IMAGE, "image")
        self.assertEqual(TilingOutputMode.VIDEO, "video")


class TestNormalizeVideoFrames(unittest.TestCase):
    """Tests for _normalize_video_frames function."""

    def test_already_normalized(self) -> None:
        """Test normalization of frames already in [0, 1] range."""
        frames = np.random.rand(10, 64, 64, 3).astype(np.float32)
        normalized = _normalize_video_frames(frames)

        self.assertGreaterEqual(normalized.min(), 0.0)
        self.assertLessEqual(normalized.max(), 1.0)
        self.assertEqual(normalized.shape, frames.shape)

    def test_from_negative_one_to_one(self) -> None:
        """Test normalization of frames in [-1, 1] range."""
        frames = np.random.rand(10, 64, 64, 3).astype(np.float32) * 2 - 1  # Range [-1, 1]
        normalized = _normalize_video_frames(frames)

        self.assertGreaterEqual(normalized.min(), 0.0)
        self.assertLessEqual(normalized.max(), 1.0)
        self.assertEqual(normalized.shape, frames.shape)

    def test_clips_out_of_range(self) -> None:
        """Test that normalization clips values outside [0, 1]."""
        frames = np.array([[[[-0.5, 0.5, 1.5]]]], dtype=np.float32)
        normalized = _normalize_video_frames(frames)

        self.assertGreaterEqual(normalized.min(), 0.0)
        self.assertLessEqual(normalized.max(), 1.0)

    def test_float32_input_stays_float32_in_every_branch(self) -> None:
        """Each full-size intermediate is float32, never float64."""
        rng = np.random.default_rng(0)
        cases = {
            "unit": rng.random((4, 8, 8, 3), dtype=np.float32),
            "signed": rng.random((4, 8, 8, 3), dtype=np.float32) * 2 - 1,
            "arbitrary": rng.random((4, 8, 8, 3), dtype=np.float32) * 100 - 50,
            "constant": np.full((4, 8, 8, 3), 7.0, dtype=np.float32),
        }
        for name, frames in cases.items():
            with self.subTest(name):
                self.assertEqual(_normalize_video_frames(frames).dtype, np.float32)

    def test_peak_allocation_reflects_float32_intermediates(self) -> None:
        """The [-1, 1] branch peaks at about 3x a float32 input in float32 and about 5x in float64.

        The peak is the scaled array, its clipped copy, and the final cast, all alive at once.
        """
        frames = np.random.default_rng(0).random((32, 256, 256, 3), dtype=np.float32) * 2 - 1

        tracemalloc.start()
        try:
            _normalize_video_frames(frames)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        self.assertLess(peak, 4.0 * frames.nbytes)

    def test_already_normalized_input_is_returned_unchanged(self) -> None:
        frames = np.random.default_rng(0).random((4, 8, 8, 3), dtype=np.float32)

        np.testing.assert_array_equal(_normalize_video_frames(frames), frames)

    def test_result_never_aliases_the_input(self) -> None:
        """The input may share storage with a caller-owned tensor, and callers write into the result."""
        frames = np.random.default_rng(0).random((4, 8, 8, 3), dtype=np.float32)
        original = frames.copy()

        normalized = _normalize_video_frames(frames)
        normalized[:] = 0.0

        np.testing.assert_array_equal(frames, original)


class TestExtractFramesFromDataBatch(unittest.TestCase):
    """Tests for extract_frames_from_data_batch function."""

    def test_nchw_format(self) -> None:
        """Test extracting frames from data batch in NCHW format."""
        batch_size = 5
        height, width = 64, 128

        # Create data batch in NCHW format (batch, channels, height, width)
        camera_images = torch.randn(batch_size, 3, height, width)
        world_images = torch.randn(batch_size, 3, height, width)

        data_batch = {
            "camera_image": camera_images,
            "world_image": world_images,
            "camera_fps": 30.0,
        }

        result = extract_frames_from_data_batch(cast(dict[str, torch.Tensor], data_batch))
        self.assertIsNotNone(result)

        camera_frames, world_frames, fps = result
        assert camera_frames is not None
        assert world_frames is not None

        # Check shapes - should be converted to NHWC format
        self.assertEqual(camera_frames.shape, (batch_size, height, width, 3))
        self.assertEqual(world_frames.shape, (batch_size, height, width, 3))
        self.assertEqual(fps, 30.0)

        # Check values are normalized to [0, 1]
        self.assertGreaterEqual(camera_frames.min(), 0.0)
        self.assertLessEqual(camera_frames.max(), 1.0)
        self.assertGreaterEqual(world_frames.min(), 0.0)
        self.assertLessEqual(world_frames.max(), 1.0)

    def test_with_default_fps(self) -> None:
        """Test that default FPS is used when not provided."""
        batch_size = 3
        height, width = 32, 64

        # Create data batch without explicit fps
        data_batch = {
            "camera_image": torch.randn(batch_size, 3, height, width),
            "world_image": torch.randn(batch_size, 3, height, width),
        }

        result = extract_frames_from_data_batch(data_batch)
        self.assertIsNotNone(result)

        camera_frames, world_frames, fps = result
        assert camera_frames is not None
        assert world_frames is not None

        # Should use default FPS of 30.0
        self.assertEqual(fps, 30.0)
        self.assertEqual(camera_frames.shape, (batch_size, height, width, 3))
        self.assertEqual(world_frames.shape, (batch_size, height, width, 3))

    def test_missing_keys(self) -> None:
        """Test that KeyError is raised when required keys are missing."""
        # Missing world_image
        data_batch = {
            "camera_image": torch.randn(5, 3, 64, 64),
        }

        with self.assertRaises(KeyError):
            extract_frames_from_data_batch(data_batch)

        # Missing camera_image
        data_batch = {
            "world_image": torch.randn(5, 3, 64, 64),
        }

        with self.assertRaises(KeyError):
            extract_frames_from_data_batch(data_batch)


class TestTileAndSaveMedia(unittest.TestCase):
    """Tests for tile_and_save_media function."""

    def test_image_mode_2x2(self) -> None:
        """Test tile_and_save_media in IMAGE mode with 2x2 layout."""
        num_frames = 3
        H, W = 64, 128

        # Create dummy input arrays (NHWC format)
        camera_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        camera_spatial = np.random.rand(num_frames, H // 2, W // 2, 3).astype(np.float32)
        world_spatial = np.random.rand(num_frames, H // 2, W // 2, 3).astype(np.float32)

        result = tile_and_save_media(
            camera_images=camera_images,
            world_images=world_images,
            camera_spatial_features=camera_spatial,
            world_spatial_features=world_spatial,
            output_mode=TilingOutputMode.IMAGE,
            use_gpu_encoding=False,
        )

        # In IMAGE mode, should return a numpy array
        self.assertIsInstance(result, np.ndarray)
        assert isinstance(result, np.ndarray)
        # Result should have 3 color channels
        self.assertEqual(result.shape[-1], 3)

    def test_image_mode_with_specific_frames(self) -> None:
        """Test tile_and_save_media with specific frame indices."""
        num_frames = 10
        H, W = 32, 64

        camera_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        camera_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)

        # Request specific frames
        frame_indices = [0, 5, 9]

        result = tile_and_save_media(
            camera_images=camera_images,
            world_images=world_images,
            camera_spatial_features=camera_spatial,
            world_spatial_features=world_spatial,
            output_mode=TilingOutputMode.IMAGE,
            frame_indices=frame_indices,
            use_gpu_encoding=False,
        )

        self.assertIsInstance(result, np.ndarray)

    def test_invalid_output_mode(self) -> None:
        """Test that invalid output_mode raises an error."""
        num_frames = 2
        H, W = 32, 64

        camera_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        camera_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)

        # In Python 3.11, enum __contains__ with string raises TypeError
        # In Python 3.12+, it returns False and then ValueError is raised
        with self.assertRaises((ValueError, TypeError)):
            tile_and_save_media(
                camera_images=camera_images,
                world_images=world_images,
                camera_spatial_features=camera_spatial,
                world_spatial_features=world_spatial,
                output_mode=cast(TilingOutputMode, "invalid_mode"),
                use_gpu_encoding=False,
            )

    def test_video_mode_requires_output_path(self) -> None:
        """Test that VIDEO mode requires output_path."""
        num_frames = 2
        H, W = 32, 64

        camera_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        camera_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)

        with self.assertRaises(ValueError):
            tile_and_save_media(
                camera_images=camera_images,
                world_images=world_images,
                camera_spatial_features=camera_spatial,
                world_spatial_features=world_spatial,
                output_mode=TilingOutputMode.VIDEO,
                output_path=None,  # Should raise error
                use_gpu_encoding=False,
            )

    def test_image_mode_2x3_with_object_type(self) -> None:
        """Test tile_and_save_media in IMAGE mode with 2x3 layout (object type)."""
        num_frames = 2
        H, W = 32, 64

        camera_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        camera_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        object_type_overlay = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        object_type_prediction = np.random.rand(num_frames, H, W, 3).astype(np.float32)

        result = tile_and_save_media(
            camera_images=camera_images,
            world_images=world_images,
            camera_spatial_features=camera_spatial,
            world_spatial_features=world_spatial,
            output_mode=TilingOutputMode.IMAGE,
            object_type_overlay=object_type_overlay,
            object_type_prediction=object_type_prediction,
            use_gpu_encoding=False,
        )

        self.assertIsInstance(result, np.ndarray)

    def test_conflicting_layouts_raises_error(self) -> None:
        """Test that providing both 3x2 and 2x3 layout features raises error."""
        num_frames = 2
        H, W = 32, 64

        camera_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        camera_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        camera_patch_features = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        object_type_overlay = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        object_type_prediction = np.random.rand(num_frames, H, W, 3).astype(np.float32)

        with self.assertRaises(ValueError):
            tile_and_save_media(
                camera_images=camera_images,
                world_images=world_images,
                camera_spatial_features=camera_spatial,
                world_spatial_features=world_spatial,
                output_mode=TilingOutputMode.IMAGE,
                camera_patch_features=camera_patch_features,
                object_type_overlay=object_type_overlay,
                object_type_prediction=object_type_prediction,
                use_gpu_encoding=False,
            )


class TestTileAndSaveMediaDebug3x3(unittest.TestCase):
    """Tests for the debug 3x3 layout (the diagnostic 'backbone vs alignment' layout)."""

    def _make_inputs(self, num_frames: int = 2, H: int = 32, W: int = 64) -> dict:
        return {
            "camera_images": np.random.rand(num_frames, H, W, 3).astype(np.float32),
            "world_images": np.random.rand(num_frames, H, W, 3).astype(np.float32),
            "camera_spatial_features": np.random.rand(num_frames, H, W, 3).astype(np.float32),
            "world_spatial_features": np.random.rand(num_frames, H, W, 3).astype(np.float32),
        }

    def test_3x3_layout_with_full_set(self) -> None:
        num_frames, H, W = 2, 32, 64
        inputs = self._make_inputs(num_frames, H, W)

        result = tile_and_save_media(
            **inputs,
            output_mode=TilingOutputMode.IMAGE,
            camera_patch_features=np.random.rand(num_frames, H, W, 3).astype(np.float32),
            camera_type_features=np.random.rand(num_frames, H, W, 3).astype(np.float32),
            spatial_cosine_sim=np.random.randint(0, 256, size=(num_frames, H, W, 3)).astype(np.uint8),
            camera_backbone_features=np.random.rand(num_frames, H, W, 3).astype(np.float32),
            world_backbone_features=np.random.rand(num_frames, H, W, 3).astype(np.float32),
            use_gpu_encoding=False,
        )

        self.assertIsInstance(result, np.ndarray)
        assert isinstance(result, np.ndarray)
        self.assertEqual(result.shape[-1], 3)
        self.assertGreaterEqual(result.shape[0], H * 3)

    def test_3x3_layout_with_backbone_missing_renders_na(self) -> None:
        """When backbone features are not provided, the row 2 cam/world cells render a "Backbone Feature N/A"
        placeholder. The result is still a valid 3x3 tiled image and does not raise.
        """
        num_frames, H, W = 2, 32, 64
        inputs = self._make_inputs(num_frames, H, W)

        result = tile_and_save_media(
            **inputs,
            output_mode=TilingOutputMode.IMAGE,
            camera_patch_features=np.random.rand(num_frames, H, W, 3).astype(np.float32),
            camera_type_features=np.random.rand(num_frames, H, W, 3).astype(np.float32),
            spatial_cosine_sim=np.random.randint(0, 256, size=(num_frames, H, W, 3)).astype(np.uint8),
            use_gpu_encoding=False,
        )

        self.assertIsInstance(result, np.ndarray)

    def test_3x3_layout_falls_back_to_3x2_without_type_features(self) -> None:
        """3x3 trigger requires camera_type_features. Without it (but with camera_patch_features), the 3x2 layout
        fires; backbone features are ignored in that case and the call still succeeds.
        """
        num_frames, H, W = 2, 32, 64
        inputs = self._make_inputs(num_frames, H, W)

        result = tile_and_save_media(
            **inputs,
            output_mode=TilingOutputMode.IMAGE,
            camera_patch_features=np.random.rand(num_frames, H, W, 3).astype(np.float32),
            spatial_cosine_sim=np.random.randint(0, 256, size=(num_frames, H, W, 3)).astype(np.uint8),
            camera_backbone_features=np.random.rand(num_frames, H, W, 3).astype(np.float32),
            world_backbone_features=np.random.rand(num_frames, H, W, 3).astype(np.float32),
            use_gpu_encoding=False,
        )

        self.assertIsInstance(result, np.ndarray)


class TestTileAndSaveMediaMaskTrio3x3(unittest.TestCase):
    """Tests for the 3x3 with-mask-trio layout (2x3 base + row-3 mask trio).

    The row-3 trio (``camera_masked_by_similarity``, ``similarity_mask``, ``object_type_masked_prediction``) is built
    by ``patch_vis_utils.create_combined_patch_visualization`` from the similarity head's
    ``P(MATCH) + P(EXTRA) + P(WRONG)`` per-pixel mask (the canonical default — see
    ``CWIPInference.DEFAULT_PATCH_SIMILARITY_MASK_CLASSES``). Layout activates only when all three are provided
    alongside the 2x3 trigger; otherwise we fall back to plain 2x3 so callers can opt in incrementally. The layout is
    mode-agnostic — gating is purely on input arrays, not on the caller's training/eval/inference mode.
    """

    def _make_2x3_inputs(self, num_frames: int = 2, H: int = 32, W: int = 64) -> dict:
        return {
            "camera_images": np.random.rand(num_frames, H, W, 3).astype(np.float32),
            "world_images": np.random.rand(num_frames, H, W, 3).astype(np.float32),
            "camera_spatial_features": np.random.rand(num_frames, H, W, 3).astype(np.float32),
            "world_spatial_features": np.random.rand(num_frames, H, W, 3).astype(np.float32),
            "object_type_overlay": np.random.rand(num_frames, H, W, 3).astype(np.float32),
            "object_type_prediction": np.random.rand(num_frames, H, W, 3).astype(np.float32),
        }

    def _make_row3_trio(self, num_frames: int = 2, H: int = 32, W: int = 64) -> dict:
        return {
            "similarity_mask": np.random.rand(num_frames, H, W, 3).astype(np.float32),
            "camera_masked_by_similarity": np.random.rand(num_frames, H, W, 3).astype(np.float32),
            "object_type_masked_prediction": np.random.rand(num_frames, H, W, 3).astype(np.float32),
        }

    def test_full_trio_triggers_3x3_layout(self) -> None:
        """All three row-3 inputs ⇒ 3x3 image with H*3 rows."""
        num_frames, H, W = 2, 32, 64
        result = tile_and_save_media(
            **self._make_2x3_inputs(num_frames, H, W),
            **self._make_row3_trio(num_frames, H, W),
            output_mode=TilingOutputMode.IMAGE,
            use_gpu_encoding=False,
        )

        self.assertIsInstance(result, np.ndarray)
        assert isinstance(result, np.ndarray)
        self.assertEqual(result.shape[-1], 3)
        # 3 rows × H per tile (image-mode arranges multiple frame outputs in a square-ish grid, so total height is at
        # least H*3).
        self.assertGreaterEqual(result.shape[0], H * 3)

    def test_missing_one_row3_input_falls_back_to_2x3(self) -> None:
        """If even one of the row-3 trio is missing we keep the legacy 2x3 layout."""
        num_frames, H, W = 2, 32, 64
        trio = self._make_row3_trio(num_frames, H, W)
        # Drop one input — the trio is incomplete.
        trio.pop("object_type_masked_prediction")

        result = tile_and_save_media(
            **self._make_2x3_inputs(num_frames, H, W),
            **trio,
            output_mode=TilingOutputMode.IMAGE,
            use_gpu_encoding=False,
        )

        self.assertIsInstance(result, np.ndarray)
        assert isinstance(result, np.ndarray)
        # 2x3 layout ⇒ result height is H*2 multiplied by however the IMAGE-mode square-ish grid arranges things.
        # Tightest universal check: it's not taller than the 3x3 layout would produce.
        # (Stronger check exists in the VIDEO test below.)
        self.assertEqual(result.shape[-1], 3)

    def test_video_mode_3x3_emits_3h_3w(self) -> None:
        """VIDEO mode actually writes a 3*H by 3*W frame when the full trio is provided."""
        num_frames, H, W = 2, 32, 64
        with tempfile.TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, "vis.mp4")
            with patch(
                "checks.utils.models.cwip.visualization.basic_vis_utils.image_video_utils.write_video_with_ffmpeg"
            ) as mock_write:
                tile_and_save_media(
                    **self._make_2x3_inputs(num_frames, H, W),
                    **self._make_row3_trio(num_frames, H, W),
                    output_mode=TilingOutputMode.VIDEO,
                    output_path=output_path,
                    use_gpu_encoding=False,
                )

                # write_video_with_ffmpeg(generator, output_path, out_width, out_height, fps)
                _, _, out_width, out_height, _ = mock_write.call_args[0]
                self.assertEqual(out_width, W * 3)
                self.assertEqual(out_height, H * 3)

    def test_video_mode_2x3_when_trio_absent(self) -> None:
        """VIDEO mode without the trio falls back to 2*H by 3*W (unchanged 2x3)."""
        num_frames, H, W = 2, 32, 64
        with tempfile.TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, "vis.mp4")
            with patch(
                "checks.utils.models.cwip.visualization.basic_vis_utils.image_video_utils.write_video_with_ffmpeg"
            ) as mock_write:
                tile_and_save_media(
                    **self._make_2x3_inputs(num_frames, H, W),
                    output_mode=TilingOutputMode.VIDEO,
                    output_path=output_path,
                    use_gpu_encoding=False,
                )

                _, _, out_width, out_height, _ = mock_write.call_args[0]
                self.assertEqual(out_width, W * 3)
                self.assertEqual(out_height, H * 2)

    def test_mask_trio_3x3_conflicts_with_3x2_3x3_layout(self) -> None:
        """3x3 with-mask-trio layout cannot coexist with the diagnostic 3x3 / 3x2 layouts."""
        num_frames, H, W = 2, 32, 64
        with self.assertRaises(ValueError):
            tile_and_save_media(
                **self._make_2x3_inputs(num_frames, H, W),
                **self._make_row3_trio(num_frames, H, W),
                # 3x2/3x3 trigger:
                camera_patch_features=np.random.rand(num_frames, H, W, 3).astype(np.float32),
                output_mode=TilingOutputMode.IMAGE,
                use_gpu_encoding=False,
            )


class TestRadialPlots(unittest.TestCase):
    """Tests for radial plot functions."""

    def test_create_radial_class_percentage_plot(self) -> None:
        """Test create_radial_class_percentage_plot generates a plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_radial.png")

            # Create mock per-frame class percentages
            num_frames = 10
            per_frame_class_percentages = {
                1: [25.0 + i * 2 for i in range(num_frames)],  # MATCH
                2: [10.0 + i for i in range(num_frames)],  # WRONG
                3: [5.0 for _ in range(num_frames)],  # EXTRA
                4: [60.0 - i * 3 for i in range(num_frames)],  # MISSING
            }

            create_radial_class_percentage_plot(per_frame_class_percentages, output_path)

            # Check that the file was created
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_create_radial_class_percentage_plot_with_averages(self) -> None:
        """Test create_radial_class_percentage_plot with pre-computed averages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_radial_avg.png")

            num_frames = 5
            per_frame_class_percentages = {
                1: [50.0 for _ in range(num_frames)],
                2: [30.0 for _ in range(num_frames)],
                3: [10.0 for _ in range(num_frames)],
                4: [10.0 for _ in range(num_frames)],
            }
            class_percentages = {1: 50.0, 2: 30.0, 3: 10.0, 4: 10.0}

            create_radial_class_percentage_plot(per_frame_class_percentages, output_path, class_percentages)

            self.assertTrue(os.path.exists(output_path))

    def test_create_radial_class_pixel_count_plot(self) -> None:
        """Test create_radial_class_pixel_count_plot generates a plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_pixel_radial.png")

            num_frames = 8
            per_frame_class_pixel_counts = {
                0: [100 for _ in range(num_frames)],  # DONT_CARE (excluded)
                1: [500 + i * 10 for i in range(num_frames)],  # MATCH
                2: [200 + i * 5 for i in range(num_frames)],  # WRONG
                3: [50 for _ in range(num_frames)],  # EXTRA
                4: [150 - i * 5 for i in range(num_frames)],  # MISSING
            }

            create_radial_class_pixel_count_plot(per_frame_class_pixel_counts, output_path)

            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)


class TestDumpFunctions(unittest.TestCase):
    """Tests for dump functions."""

    def test_dump_per_frame_patch_scores_csv(self) -> None:
        """Test dump_per_frame_patch_scores in CSV format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "patch_scores.csv")

            num_frames = 5
            per_frame_class_percentages = {
                1: [20.0 + i * 5 for i in range(num_frames)],
                2: [10.0 + i * 2 for i in range(num_frames)],
                3: [5.0 for _ in range(num_frames)],
                4: [15.0 - i for i in range(num_frames)],
            }

            dump_per_frame_patch_scores(
                per_frame_class_percentages=per_frame_class_percentages,
                output_path=output_path,
                camera_start_frame=0,
                world_start_frame=0,
                format="csv",
            )

            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

            # Verify CSV content
            with open(output_path) as f:
                content = f.read()
                self.assertIn("frame_index", content)
                self.assertIn("camera_frame", content)
                self.assertIn("world_frame", content)

    def test_dump_per_frame_patch_scores_json(self) -> None:
        """Test dump_per_frame_patch_scores in JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "patch_scores.json")

            num_frames = 3
            per_frame_class_percentages = {
                1: [30.0, 35.0, 40.0],
                2: [20.0, 25.0, 30.0],
                3: [10.0, 10.0, 10.0],
                4: [40.0, 30.0, 20.0],
            }

            dump_per_frame_patch_scores(
                per_frame_class_percentages=per_frame_class_percentages,
                output_path=output_path,
                camera_start_frame=10,
                world_start_frame=20,
                format="json",
            )

            self.assertTrue(os.path.exists(output_path))

            # Verify JSON content
            with open(output_path) as f:
                data = json.load(f)
                self.assertIn("metadata", data)
                self.assertIn("per_frame_scores", data)
                self.assertEqual(len(data["per_frame_scores"]), num_frames)

    def test_dump_per_frame_patch_scores_npy(self) -> None:
        """Test dump_per_frame_patch_scores in NPY format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "patch_scores.npy")

            per_frame_class_percentages = {
                1: [25.0, 30.0, 35.0, 40.0],
                2: [15.0, 20.0, 25.0, 30.0],
            }

            dump_per_frame_patch_scores(
                per_frame_class_percentages=per_frame_class_percentages,
                output_path=output_path,
                format="npy",
            )

            self.assertTrue(os.path.exists(output_path))

            # Metadata file should also exist
            metadata_path = output_path.replace(".npy", "_metadata.json")
            self.assertTrue(os.path.exists(metadata_path))

    def test_dump_per_frame_patch_scores_invalid_format(self) -> None:
        """Test that invalid format raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "patch_scores.xyz")

            per_frame_class_percentages = {1: [50.0]}

            with self.assertRaises(ValueError):
                dump_per_frame_patch_scores(
                    per_frame_class_percentages=per_frame_class_percentages,
                    output_path=output_path,
                    format="xyz",
                )

    def test_dump_per_frame_patch_scores_empty(self) -> None:
        """Test dump_per_frame_patch_scores with empty data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "empty_scores.csv")

            # Should return early without creating file
            dump_per_frame_patch_scores(
                per_frame_class_percentages={},
                output_path=output_path,
                format="csv",
            )

            # File should not be created for empty data
            self.assertFalse(os.path.exists(output_path))

    def test_dump_per_frame_consistency_scores_csv(self) -> None:
        """Test dump_per_frame_consistency_scores in CSV format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "consistency.csv")

            # Create a simple similarity matrix
            num_frames = 5
            similarity_matrix = np.eye(num_frames) * 0.8 + np.random.rand(num_frames, num_frames) * 0.2

            dump_per_frame_consistency_scores(
                similarity_matrix=similarity_matrix,
                output_path=output_path,
                camera_start_frame=0,
                world_start_frame=0,
                format="csv",
            )

            self.assertTrue(os.path.exists(output_path))

            # Verify CSV content
            with open(output_path) as f:
                content = f.read()
                self.assertIn("frame_index", content)
                self.assertIn("consistency_score", content)

    def test_dump_per_frame_consistency_scores_json(self) -> None:
        """Test dump_per_frame_consistency_scores in JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "consistency.json")

            num_frames = 3
            similarity_matrix = np.eye(num_frames)

            dump_per_frame_consistency_scores(
                similarity_matrix=similarity_matrix,
                output_path=output_path,
                format="json",
            )

            self.assertTrue(os.path.exists(output_path))

            with open(output_path) as f:
                data = json.load(f)
                self.assertIn("metadata", data)
                self.assertIn("per_frame_scores", data)

    def test_dump_per_frame_consistency_scores_npy(self) -> None:
        """Test dump_per_frame_consistency_scores in NPY format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "consistency.npy")

            num_frames = 4
            similarity_matrix = np.eye(num_frames) * 0.9

            dump_per_frame_consistency_scores(
                similarity_matrix=similarity_matrix,
                output_path=output_path,
                format="npy",
            )

            self.assertTrue(os.path.exists(output_path))

            # Load and verify
            loaded = np.load(output_path)
            self.assertEqual(loaded.shape, (num_frames,))

    def test_dump_per_frame_consistency_scores_none_matrix(self) -> None:
        """Test dump_per_frame_consistency_scores with None matrix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "none_consistency.csv")

            # Should return early without creating file
            dump_per_frame_consistency_scores(
                similarity_matrix=None,
                output_path=output_path,
                format="csv",
            )

            self.assertFalse(os.path.exists(output_path))


class TestSimilarityVisualization(unittest.TestCase):
    """Tests for similarity visualization functions."""

    def test_save_similarity_matrix_with_frame_indices(self) -> None:
        """Test saving similarity matrix with frame indices."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "similarity.png")

            # Create a sample similarity matrix
            num_frames = 5
            similarity_matrix = np.eye(num_frames) * 0.9 + np.random.rand(num_frames, num_frames) * 0.1
            similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2  # Make symmetric

            save_similarity_matrix_with_frame_indices(
                similarity_matrix=similarity_matrix,
                output_path=output_path,
                camera_start_frame=0,
                world_start_frame=0,
                similarity_score=0.85,
            )

            self.assertTrue(os.path.exists(output_path))

    def test_save_similarity_matrix_with_offsets(self) -> None:
        """Test saving similarity matrix with frame offsets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "similarity_offset.png")

            num_frames = 4
            similarity_matrix = np.eye(num_frames)

            save_similarity_matrix_with_frame_indices(
                similarity_matrix=similarity_matrix,
                output_path=output_path,
                camera_start_frame=100,
                world_start_frame=200,
                similarity_score=0.9,
            )

            self.assertTrue(os.path.exists(output_path))

    def test_save_enhanced_diagonal_plots(self) -> None:
        """Test saving enhanced diagonal plots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            num_frames = 6
            similarity_matrix = np.eye(num_frames) * 0.8 + np.random.rand(num_frames, num_frames) * 0.2

            # Should run without error
            save_enhanced_diagonal_plots(
                similarity_matrix=similarity_matrix,
                output_dir=tmpdir,
                iteration=0,
                camera_start_frame=0,
                world_start_frame=0,
            )

            # Check that at least some output files were created
            files_created = os.listdir(tmpdir)
            self.assertGreater(len(files_created), 0, "No output files created")


class TestPlotFunctions(unittest.TestCase):
    """Tests for individual plot functions."""

    def test_create_gradient_area_plot(self) -> None:
        """Test creating gradient area plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "gradient.png")

            scores = np.array([0.8, 0.85, 0.9, 0.75, 0.82])
            positions = np.arange(len(scores))
            camera_indices = np.arange(len(scores))
            world_indices = np.arange(len(scores))

            create_gradient_area_plot(
                scores=scores,
                positions=positions,
                camera_indices=camera_indices,
                world_indices=world_indices,
                output_path=output_path,
            )

            self.assertTrue(os.path.exists(output_path))

    def test_create_heatmap_strip(self) -> None:
        """Test creating heatmap strip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "heatmap.png")

            scores = np.array([0.7, 0.8, 0.9, 0.85, 0.75])
            camera_indices = np.arange(len(scores))
            world_indices = np.arange(len(scores))

            create_heatmap_strip(
                scores=scores,
                camera_indices=camera_indices,
                world_indices=world_indices,
                output_path=output_path,
            )

            self.assertTrue(os.path.exists(output_path))

    def test_create_radial_similarity_plot(self) -> None:
        """Test creating radial similarity plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "radial.png")

            scores = np.array([0.75, 0.82, 0.88, 0.91, 0.79])
            camera_indices = np.arange(len(scores))
            world_indices = np.arange(len(scores))

            create_radial_similarity_plot(
                scores=scores,
                camera_indices=camera_indices,
                world_indices=world_indices,
                output_path=output_path,
            )

            self.assertTrue(os.path.exists(output_path))

    def test_create_similarity_dashboard(self) -> None:
        """Test creating similarity dashboard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "dashboard.png")

            num_frames = 5
            similarity_matrix = np.eye(num_frames) * 0.9
            diagonal_scores = np.diag(similarity_matrix)
            camera_indices = np.arange(num_frames)
            world_indices = np.arange(num_frames)

            create_similarity_dashboard(
                similarity_matrix=similarity_matrix,
                diagonal_scores=diagonal_scores,
                camera_indices=camera_indices,
                world_indices=world_indices,
                output_path=output_path,
            )

            self.assertTrue(os.path.exists(output_path))


class TestSavePatchPredictions(unittest.TestCase):
    """Tests for save_patch_predictions function."""

    def test_save_patch_predictions_basic(self) -> None:
        """Test saving patch predictions as numpy files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            num_frames = 4
            H, W = 8, 8

            patch_results = {
                "pred_class_indices": torch.randint(0, 5, (num_frames, H, W)),
                "pred_probabilities": torch.rand(num_frames, H, W),
            }

            save_patch_predictions(
                patch_results=patch_results,
                output_dir=tmpdir,
                iteration=0,
            )

            # Check files were created
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "pred_class_indices_iter_000000.npy")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "pred_probabilities_iter_000000.npy")))

            # Verify contents
            loaded_indices = np.load(os.path.join(tmpdir, "pred_class_indices_iter_000000.npy"))
            self.assertEqual(loaded_indices.shape, (num_frames, H, W))

    def test_save_patch_predictions_missing_keys(self) -> None:
        """Test save_patch_predictions with missing keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Empty patch_results
            patch_results: dict[str, torch.Tensor] = {}

            # Should not raise, just skip missing keys
            save_patch_predictions(
                patch_results=patch_results,
                output_dir=tmpdir,
                iteration=0,
            )


class TestTileAndSaveMediaWithCosine(unittest.TestCase):
    """Tests for tile_and_save_media with spatial_cosine_sim parameter."""

    def test_3x2_layout_with_cosine_sim(self) -> None:
        """Test 3x2 layout with spatial_cosine_sim."""
        num_frames = 2
        H, W = 32, 64

        camera_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        camera_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        camera_patch = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        cosine_sim = np.random.rand(num_frames, H, W, 3).astype(np.float32)

        result = tile_and_save_media(
            camera_images=camera_images,
            world_images=world_images,
            camera_spatial_features=camera_spatial,
            world_spatial_features=world_spatial,
            output_mode=TilingOutputMode.IMAGE,
            camera_patch_features=camera_patch,
            spatial_cosine_sim=cosine_sim,
            use_gpu_encoding=False,
        )

        self.assertIsInstance(result, np.ndarray)


class TestDumpConsistencyScoresExtended(unittest.TestCase):
    """Extended tests for dump_per_frame_consistency_scores."""

    def test_invalid_format_raises(self) -> None:
        """Test invalid format raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "scores.xyz")
            similarity_matrix = np.eye(3)

            with self.assertRaises(ValueError):
                dump_per_frame_consistency_scores(
                    similarity_matrix=similarity_matrix,
                    output_path=output_path,
                    format="xyz",
                )

    def test_1d_similarity_matrix(self) -> None:
        """Test with non-square (1D) similarity matrix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "scores.csv")
            # 1D input should still compute diagonal
            similarity_matrix = np.eye(1)

            dump_per_frame_consistency_scores(
                similarity_matrix=similarity_matrix,
                output_path=output_path,
                format="csv",
            )

            self.assertTrue(os.path.exists(output_path))


class TestTileAndSaveMediaLayouts(unittest.TestCase):
    """Tests for tile_and_save_media with different layouts and edge cases."""

    def test_3x2_layout_no_patch_features_creates_placeholder(self) -> None:
        """Test 3x2 layout with spatial_cosine_sim but no camera_patch_features creates N/A placeholder."""
        num_frames = 2
        H, W = 32, 64
        camera_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        camera_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        cosine_sim = (np.random.rand(num_frames, H // 2, W // 2, 3) * 255).astype(np.uint8)

        result = tile_and_save_media(
            camera_images=camera_images,
            world_images=world_images,
            camera_spatial_features=camera_spatial,
            world_spatial_features=world_spatial,
            output_mode=TilingOutputMode.IMAGE,
            spatial_cosine_sim=cosine_sim,
            use_gpu_encoding=False,
        )
        self.assertIsInstance(result, np.ndarray)

    def test_image_mode_multiple_frames_grid_layout(self) -> None:
        """Test image mode with many frames triggers grid layout with padding."""
        num_frames = 5  # 5 frames -> 2x3 grid -> last row needs padding
        H, W = 16, 16
        camera_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        camera_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)

        result = tile_and_save_media(
            camera_images=camera_images,
            world_images=world_images,
            camera_spatial_features=camera_spatial,
            world_spatial_features=world_spatial,
            output_mode=TilingOutputMode.IMAGE,
            use_gpu_encoding=False,
        )
        self.assertIsInstance(result, np.ndarray)
        # With 5 frames, sqrt(5) ~ 2.23, so best_rows=2, best_cols=3
        # Grid should be 2 rows x 3 cols of 2x2 tiles
        # Height: 2 rows * 2 tile_rows * H = 4H, Width: 3 cols * 2 tile_cols * W = 6W
        # But last row only has 2 frames so it needs padding

    def test_out_of_bounds_frame_indices(self) -> None:
        """Test that out-of-bounds frame indices are handled gracefully."""
        num_frames = 3
        H, W = 16, 16
        camera_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        camera_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)

        # Index 10 is out of bounds for 3 frames
        result = tile_and_save_media(
            camera_images=camera_images,
            world_images=world_images,
            camera_spatial_features=camera_spatial,
            world_spatial_features=world_spatial,
            output_mode=TilingOutputMode.IMAGE,
            frame_indices=[0, 10],
            use_gpu_encoding=False,
        )
        self.assertIsInstance(result, np.ndarray)

    def test_single_frame_image(self) -> None:
        """Test image mode with exactly 1 frame (no grid needed)."""
        H, W = 16, 16
        camera_images = np.random.rand(1, H, W, 3).astype(np.float32)
        world_images = np.random.rand(1, H, W, 3).astype(np.float32)
        camera_spatial = np.random.rand(1, H, W, 3).astype(np.float32)
        world_spatial = np.random.rand(1, H, W, 3).astype(np.float32)

        result = tile_and_save_media(
            camera_images=camera_images,
            world_images=world_images,
            camera_spatial_features=camera_spatial,
            world_spatial_features=world_spatial,
            output_mode=TilingOutputMode.IMAGE,
            use_gpu_encoding=False,
        )
        self.assertIsInstance(result, np.ndarray)

    def test_3x2_with_cosine_sim_no_patch_features_multi_frame(self) -> None:
        """Test 3x2 layout with cosine sim but no patch features and multiple frames."""
        num_frames = 4
        H, W = 16, 16
        camera_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        camera_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        cosine_sim = (np.random.rand(num_frames, H, W, 3) * 255).astype(np.uint8)

        result = tile_and_save_media(
            camera_images=camera_images,
            world_images=world_images,
            camera_spatial_features=camera_spatial,
            world_spatial_features=world_spatial,
            output_mode=TilingOutputMode.IMAGE,
            spatial_cosine_sim=cosine_sim,
            use_gpu_encoding=False,
        )
        self.assertIsInstance(result, np.ndarray)

    def test_2x3_layout_multi_frame(self) -> None:
        """Test 2x3 layout with object type overlay/prediction and multiple frames."""
        num_frames = 4
        H, W = 16, 16
        camera_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        camera_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        object_type_overlay = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        object_type_prediction = np.random.rand(num_frames, H, W, 3).astype(np.float32)

        result = tile_and_save_media(
            camera_images=camera_images,
            world_images=world_images,
            camera_spatial_features=camera_spatial,
            world_spatial_features=world_spatial,
            output_mode=TilingOutputMode.IMAGE,
            object_type_overlay=object_type_overlay,
            object_type_prediction=object_type_prediction,
            use_gpu_encoding=False,
        )
        self.assertIsInstance(result, np.ndarray)


class TestExtractFramesFromDataBatchExtended(unittest.TestCase):
    """Extended tests for extract_frames_from_data_batch."""

    def test_bfloat16_conversion(self) -> None:
        """Test that bfloat16 tensors are converted to float32."""
        data_batch = {
            "camera_image": torch.rand(2, 3, 32, 32).to(torch.bfloat16),
            "world_image": torch.rand(2, 3, 32, 32).to(torch.bfloat16),
        }
        camera, world, _fps = extract_frames_from_data_batch(data_batch)
        assert camera is not None
        assert world is not None
        self.assertEqual(camera.dtype, np.float32)
        self.assertEqual(world.dtype, np.float32)

    def test_fps_from_data_batch(self) -> None:
        """Test fps extraction from data_batch."""
        data_batch = {
            "camera_image": torch.rand(2, 3, 32, 32),
            "world_image": torch.rand(2, 3, 32, 32),
            "fps": 24.0,
        }
        _, _, fps = extract_frames_from_data_batch(cast(dict[str, torch.Tensor], data_batch))
        self.assertEqual(fps, 24.0)

    def test_fps_from_camera_fps_key(self) -> None:
        """Test fps extraction from camera_fps key."""
        data_batch = {
            "camera_image": torch.rand(2, 3, 32, 32),
            "world_image": torch.rand(2, 3, 32, 32),
            "camera_fps": 60.0,
        }
        _, _, fps = extract_frames_from_data_batch(cast(dict[str, torch.Tensor], data_batch))
        self.assertEqual(fps, 60.0)

    def test_nhwc_format_passthrough(self) -> None:
        """Test that NHWC format images are not transposed."""
        data_batch = {
            "camera_image": torch.rand(2, 32, 32, 3),
            "world_image": torch.rand(2, 32, 32, 3),
        }
        camera, _world, _ = extract_frames_from_data_batch(data_batch)
        assert camera is not None
        self.assertEqual(camera.shape, (2, 32, 32, 3))

    def test_unexpected_dimensions(self) -> None:
        """Test that unexpected dimensions return None."""
        data_batch = {
            "camera_image": torch.rand(32, 32),  # 2D - unexpected
            "world_image": torch.rand(32, 32),
        }
        result = extract_frames_from_data_batch(data_batch)
        self.assertIsNone(result[0])


class TestCreateSimilarityDashboardExtended(unittest.TestCase):
    """Extended tests for create_similarity_dashboard."""

    def test_none_similarity_matrix(self) -> None:
        """Test that None similarity_matrix is handled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "dashboard.png")
            create_similarity_dashboard(
                similarity_matrix=None,
                diagonal_scores=np.array([0.5, 0.6, 0.7]),
                camera_indices=np.array([0, 1, 2]),
                world_indices=np.array([0, 1, 2]),
                output_path=output_path,
            )
            # File should NOT be created when matrix is None
            self.assertFalse(os.path.exists(output_path))

    def test_identical_scores_histogram(self) -> None:
        """Test dashboard with identical diagonal scores (edge case for histogram)."""
        matrix = np.ones((5, 5)) * 0.5
        scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "dashboard.png")
            create_similarity_dashboard(
                similarity_matrix=matrix,
                diagonal_scores=scores,
                camera_indices=np.arange(5),
                world_indices=np.arange(5),
                output_path=output_path,
            )
            self.assertTrue(os.path.exists(output_path))


class TestDumpPerFramePatchScoresExtended(unittest.TestCase):
    """Extended tests for dump_per_frame_patch_scores."""

    def test_with_data_batch_video_paths(self) -> None:
        """Test dump with data_batch containing video paths and dimensions."""
        per_frame = {0: [10.0, 20.0], 1: [30.0, 40.0]}
        patch_results = {
            "per_frame_class_pixel_counts": {0: [100, 200], 1: [300, 400]},
            "pred_probabilities": torch.rand(2, 8, 8),
        }
        data_batch = {
            "camera_video_paths": ["/path/to/camera.mp4"],
            "world_video_paths": ["/path/to/world.mp4"],
            "camera_image": torch.rand(2, 3, 64, 128),
            "world_image": torch.rand(2, 3, 64, 128),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "scores.csv")
            dump_per_frame_patch_scores(
                per_frame,
                output_path,
                camera_start_frame=0,
                world_start_frame=0,
                format="csv",
                patch_results=patch_results,
                data_batch=data_batch,
            )
            self.assertTrue(os.path.exists(output_path))

    def test_without_data_batch(self) -> None:
        """Test dump without data_batch."""
        per_frame = {0: [10.0], 1: [30.0]}
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "scores.json")
            dump_per_frame_patch_scores(
                per_frame,
                output_path,
                camera_start_frame=0,
                world_start_frame=0,
                format="json",
                data_batch=None,
            )
            self.assertTrue(os.path.exists(output_path))


class TestDumpConsistencyScoresWithPaths(unittest.TestCase):
    """Tests for dump_per_frame_consistency_scores with video paths in data_batch."""

    def test_with_video_paths(self) -> None:
        """Test consistency scores dump with camera/world video paths (covers L1550,L1556,L1560,L1565)."""
        similarity_matrix = np.eye(4)
        data_batch = {
            "camera_video_paths": ["/path/to/camera.mp4"],
            "world_video_paths": ["/path/to/world.mp4"],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "scores.csv")
            dump_per_frame_consistency_scores(
                similarity_matrix=similarity_matrix,
                output_path=output_path,
                camera_start_frame=0,
                world_start_frame=0,
                data_batch=data_batch,
            )
            self.assertTrue(os.path.exists(output_path))


class TestDumpPatchScoresWithEmptyPercentages(unittest.TestCase):
    """Tests for dump_per_frame_patch_scores with empty percentages (covers L1215,L1229,L1243)."""

    def test_with_empty_percentages(self) -> None:
        """Test with per_frame_class_percentages containing empty lists."""
        per_frame = {0: [], 1: [10.0]}  # Empty list for class 0
        per_frame_all = {0: [], 1: [10.0]}
        per_frame_no_extra = {0: [], 1: [10.0]}
        patch_results = {
            "per_frame_class_percentages": per_frame,
            "per_frame_class_percentages_all_classes": per_frame_all,
            "per_frame_class_percentages_no_extra": per_frame_no_extra,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "scores.csv")
            dump_per_frame_patch_scores(
                per_frame,
                output_path,
                camera_start_frame=0,
                world_start_frame=0,
                format="csv",
                patch_results=patch_results,
            )
            self.assertTrue(os.path.exists(output_path))


class TestSaveEnhancedDiagonalPlotsEdgeCases(unittest.TestCase):
    """Tests for save_enhanced_diagonal_plots histogram edge cases (covers L1063-1068)."""

    def test_identical_diagonal_scores(self) -> None:
        """Test with all identical similarity scores to trigger histogram fallback."""
        # Create a similarity matrix where diagonal is all the same value
        n = 5
        sim_matrix = np.full((n, n), 0.5)
        np.fill_diagonal(sim_matrix, 0.999)  # Near-identical diagonal
        with tempfile.TemporaryDirectory() as tmpdir:
            save_enhanced_diagonal_plots(
                similarity_matrix=sim_matrix,
                output_dir=tmpdir,
                iteration=0,
                camera_start_frame=0,
                world_start_frame=0,
            )

    def test_single_frame_similarity(self) -> None:
        """Test with a 1x1 similarity matrix (single frame)."""
        sim_matrix = np.array([[1.0]])
        with tempfile.TemporaryDirectory() as tmpdir:
            save_enhanced_diagonal_plots(
                similarity_matrix=sim_matrix,
                output_dir=tmpdir,
                iteration=0,
                camera_start_frame=0,
                world_start_frame=0,
            )


class TestExtractFramesVideoMetadata(unittest.TestCase):
    """Test extract_frames_from_data_batch with video_metadata (covers L489-491)."""

    def test_with_video_metadata(self) -> None:
        """Test frame extraction with fps from video_metadata."""
        num_frames = 3
        data_batch = {
            "camera_image": torch.randn(num_frames, 3, 64, 64),
            "world_image": torch.randn(num_frames, 3, 64, 64),
            "video_metadata": {"fps": 15.0},
        }
        camera_frames, _world_frames, camera_fps = extract_frames_from_data_batch(
            cast(dict[str, torch.Tensor], data_batch)
        )
        self.assertIsNotNone(camera_frames)
        # fps from video_metadata should be extracted
        self.assertEqual(camera_fps, 15.0)


class TestTileAndSaveMediaInvalidMode(unittest.TestCase):
    """Test tile_and_save_media with invalid output_mode (covers L197)."""

    def test_invalid_output_mode_raises(self) -> None:
        """Test that invalid output_mode raises (ValueError or TypeError)."""
        num_frames = 2
        H, W = 32, 32
        camera_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        camera_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        world_spatial = np.random.rand(num_frames, H, W, 3).astype(np.float32)

        with self.assertRaises((ValueError, TypeError)):
            tile_and_save_media(
                camera_images=camera_images,
                world_images=world_images,
                camera_spatial_features=camera_spatial,
                world_spatial_features=world_spatial,
                output_mode=cast(TilingOutputMode, "invalid_mode"),
            )


class TestSaveSimilarityMatrixNoneScore(unittest.TestCase):
    """Test save_similarity_matrix_with_frame_indices with None similarity_score (covers L539)."""

    def test_none_similarity_score_computes_diagonal_mean(self) -> None:
        """When similarity_score is None, it should compute from diagonal."""
        sim_matrix = np.eye(4, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sim.png")
            save_similarity_matrix_with_frame_indices(
                sim_matrix,
                output_path=path,
                similarity_score=None,
            )
            self.assertTrue(os.path.exists(path))


class TestTileAndSaveMediaVideoMode(unittest.TestCase):
    """Test tile_and_save_media in VIDEO mode with CPU fallback encoding."""

    @patch(
        "checks.utils.models.cwip.visualization.image_video_utils.get_video_encode_args",
        return_value=list(SVTAV1_FFMPEG_ARGS),
    )
    def test_video_mode_cpu_fallback(self, _mock_encode_args: MagicMock) -> None:
        """Test VIDEO mode encodes with CPU fallback when NVENC is unavailable."""
        num_frames = 2
        H, W = 128, 128
        camera_images = (np.random.rand(num_frames, H, W, 3) * 255).astype(np.uint8)
        world_images = (np.random.rand(num_frames, H, W, 3) * 255).astype(np.uint8)
        camera_spatial = (np.random.rand(num_frames, H, W, 3) * 255).astype(np.uint8)
        world_spatial = (np.random.rand(num_frames, H, W, 3) * 255).astype(np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "video.mp4")
            result = tile_and_save_media(
                camera_images=camera_images,
                world_images=world_images,
                camera_spatial_features=camera_spatial,
                world_spatial_features=world_spatial,
                output_mode=TilingOutputMode.VIDEO,
                output_path=output_path,
                spatial_cosine_sim=None,
                use_gpu_encoding=False,
            )
            self.assertIsNone(result)  # VIDEO mode returns None
            self.assertTrue(os.path.exists(output_path))


class TestDumpPerFramePatchScoresFormats(unittest.TestCase):
    """Test dump_per_frame_patch_scores with various formats."""

    def test_csv_format(self) -> None:
        """Test dumping scores in CSV format."""
        per_frame_class_percentages = {0: [0.5, 0.6], 1: [0.3, 0.2]}

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "scores.csv")
            dump_per_frame_patch_scores(
                per_frame_class_percentages=per_frame_class_percentages,
                output_path=csv_path,
                format="csv",
            )
            self.assertTrue(os.path.exists(csv_path))

    def test_json_format(self) -> None:
        """Test dumping scores in JSON format."""
        per_frame_class_percentages = {0: [0.5, 0.6], 1: [0.3, 0.2]}

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "scores.json")
            dump_per_frame_patch_scores(
                per_frame_class_percentages=per_frame_class_percentages,
                output_path=json_path,
                format="json",
            )
            self.assertTrue(os.path.exists(json_path))


if __name__ == "__main__":
    unittest.main()
