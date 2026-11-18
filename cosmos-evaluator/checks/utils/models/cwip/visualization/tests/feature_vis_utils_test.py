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

import os
import tempfile
import tracemalloc
from typing import Any
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from checks.utils.models.cwip.visualization.basic_vis_utils import TilingOutputMode
from checks.utils.models.cwip.visualization.feature_vis_utils import (
    _get_robust_pca_with_timing,
    _label_frames,
    add_text_overlays,
    compute_spatial_cosine_similarity_heatmaps,
    create_camera_only_tiled_video,
    create_tiled_video,
)


class TestGetRobustPcaWithTiming(unittest.TestCase):
    """Tests for _get_robust_pca_with_timing function."""

    def test_small_float_and_bfloat16(self) -> None:
        torch.manual_seed(0)
        features = torch.randn(20, 5, dtype=torch.float32)
        R, cmin, cmax = _get_robust_pca_with_timing(features, m=2, remove_first_component=False, skip=0)
        self.assertEqual(R.shape[1], 3)
        self.assertEqual(cmin.shape[0], 3)
        self.assertEqual(cmax.shape[0], 3)

        features_bf16 = features.to(torch.bfloat16)
        R2, cmin2, cmax2 = _get_robust_pca_with_timing(features_bf16, m=2)
        self.assertEqual(R2.dtype, torch.bfloat16)
        self.assertEqual(cmin2.dtype, torch.bfloat16)
        self.assertEqual(cmax2.dtype, torch.bfloat16)

    def test_with_skip_parameter(self) -> None:
        """Test PCA with skip parameter to skip initial components."""
        torch.manual_seed(42)
        features = torch.randn(100, 10, dtype=torch.float32)
        R, _cmin, _cmax = _get_robust_pca_with_timing(features, m=2, skip=2)
        self.assertEqual(R.shape[1], 3)

    def test_with_remove_first_component(self) -> None:
        """Test PCA with first component removal."""
        torch.manual_seed(42)
        features = torch.randn(100, 10, dtype=torch.float32)
        R, _cmin, _cmax = _get_robust_pca_with_timing(features, m=2, remove_first_component=True)
        self.assertEqual(R.shape[1], 3)

    def test_with_exactly_three_features(self) -> None:
        """Test when features have exactly 3 dimensions."""
        torch.manual_seed(42)
        features = torch.randn(50, 3, dtype=torch.float32)  # Exactly 3 features
        R, _cmin, _cmax = _get_robust_pca_with_timing(features, m=2)
        self.assertEqual(R.shape[1], 3)

    def test_different_m_values(self) -> None:
        """Test robust statistics with different m values."""
        torch.manual_seed(42)
        features = torch.randn(100, 8, dtype=torch.float32)

        # Smaller m = more aggressive outlier rejection
        R1, _cmin1, _cmax1 = _get_robust_pca_with_timing(features, m=1.5)
        # Larger m = less aggressive outlier rejection
        R2, _cmin2, _cmax2 = _get_robust_pca_with_timing(features, m=3.0)

        self.assertEqual(R1.shape[1], 3)
        self.assertEqual(R2.shape[1], 3)


class TestAddTextOverlays(unittest.TestCase):
    """Tests for add_text_overlays function."""

    def test_preserves_shape_and_range(self) -> None:
        T, H, W = 5, 8, 10
        images = np.random.rand(T, H, W, 3).astype(np.float32)
        sim = torch.eye(T, dtype=torch.float32)
        out = add_text_overlays(images, {"similarity_matrix": sim}, frame_offset=0, view_type="camera")
        self.assertEqual(out.shape, images.shape)
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)

    def test_without_similarity_matrix(self) -> None:
        """Test overlays when similarity matrix is not provided."""
        T, H, W = 3, 64, 64
        images = np.random.rand(T, H, W, 3).astype(np.float32)
        out = add_text_overlays(images, {}, frame_offset=10, view_type="world")
        self.assertEqual(out.shape, images.shape)

    def test_with_none_similarity_matrix(self) -> None:
        """Test overlays when similarity matrix is None."""
        T, H, W = 3, 64, 64
        images = np.random.rand(T, H, W, 3).astype(np.float32)
        none_batch: dict[str, Any] = {"similarity_matrix": None}
        out = add_text_overlays(images, none_batch, frame_offset=5, view_type="camera")
        self.assertEqual(out.shape, images.shape)

    def test_mismatched_similarity_matrix_size(self) -> None:
        """Test overlays when similarity matrix has different size."""
        T, H, W = 5, 64, 64
        images = np.random.rand(T, H, W, 3).astype(np.float32)
        # Similarity matrix is 3x3 but images have 5 frames
        sim = torch.eye(3, dtype=torch.float32)
        out = add_text_overlays(images, {"similarity_matrix": sim}, frame_offset=0, view_type="camera")
        self.assertEqual(out.shape, images.shape)

    def test_with_frame_offset(self) -> None:
        """Test that frame offset is applied correctly."""
        T, H, W = 2, 128, 128
        images = np.random.rand(T, H, W, 3).astype(np.float32)
        sim = torch.eye(T, dtype=torch.float32)
        out = add_text_overlays(images, {"similarity_matrix": sim}, frame_offset=100, view_type="world")
        self.assertEqual(out.shape, images.shape)


class TestCreateTiledVideo(unittest.TestCase):
    """Tests for create_tiled_video function."""

    def test_missing_camera_image(self) -> None:
        """Test that function returns early when camera_image is missing."""
        output_batch: dict[str, Any] = {
            "camera_spatial_features_list": [torch.randn(2, 64, 8, 8)],
            "world_spatial_features_list": [torch.randn(2, 64, 8, 8)],
        }
        data_batch = {
            "world_image": torch.randn(2, 3, 64, 64),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_tiled_video(output_batch, data_batch, 0, tmpdir)
            self.assertIsNone(result)

    def test_missing_world_image(self) -> None:
        """Test that function returns early when world_image is missing."""
        output_batch: dict[str, Any] = {
            "camera_spatial_features_list": [torch.randn(2, 64, 8, 8)],
            "world_spatial_features_list": [torch.randn(2, 64, 8, 8)],
        }
        data_batch = {
            "camera_image": torch.randn(2, 3, 64, 64),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_tiled_video(output_batch, data_batch, 0, tmpdir)
            self.assertIsNone(result)

    def test_missing_camera_features_list(self) -> None:
        """Test that function returns early when camera_spatial_features_list is missing."""
        output_batch: dict[str, Any] = {
            "world_spatial_features_list": [torch.randn(2, 64, 8, 8)],
        }
        data_batch = {
            "camera_image": torch.randn(2, 3, 64, 64),
            "world_image": torch.randn(2, 3, 64, 64),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_tiled_video(output_batch, data_batch, 0, tmpdir)
            self.assertIsNone(result)

    def test_missing_world_features_list(self) -> None:
        """Test that function returns early when world_spatial_features_list is missing."""
        output_batch: dict[str, Any] = {
            "camera_spatial_features_list": [torch.randn(2, 64, 8, 8)],
        }
        data_batch = {
            "camera_image": torch.randn(2, 3, 64, 64),
            "world_image": torch.randn(2, 3, 64, 64),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_tiled_video(output_batch, data_batch, 0, tmpdir)
            self.assertIsNone(result)

    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_extract_frames_returns_none(self, mock_extract: MagicMock) -> None:
        """Test that function returns early when extract_frames returns None."""
        mock_extract.return_value = None

        output_batch: dict[str, Any] = {
            "camera_spatial_features_list": [torch.randn(2, 64, 8, 8)],
            "world_spatial_features_list": [torch.randn(2, 64, 8, 8)],
        }
        data_batch = {
            "camera_image": torch.randn(2, 3, 64, 64),
            "world_image": torch.randn(2, 3, 64, 64),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_tiled_video(output_batch, data_batch, 0, tmpdir)
            self.assertIsNone(result)

    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.tile_and_save_media")
    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_image_mode_basic(self, mock_extract: MagicMock, mock_tile: MagicMock) -> None:
        """Test create_tiled_video in IMAGE mode."""
        num_frames = 4
        H, W = 64, 64
        feat_h, feat_w = 8, 8
        feat_dim = 32

        # Mock extract_frames_from_data_batch
        mock_extract.return_value = (
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            30.0,
        )

        # Mock tile_and_save_media to return a dummy array
        mock_tile.return_value = np.random.rand(H, W * 2, 3).astype(np.float32)

        output_batch: dict[str, Any] = {
            "camera_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
            "world_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
        }
        data_batch = {
            "camera_image": torch.randn(num_frames, 3, H, W),
            "world_image": torch.randn(num_frames, 3, H, W),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_tiled_video(
                output_batch, data_batch, 0, tmpdir, output_mode=TilingOutputMode.IMAGE, num_frames=2
            )
            # Should return an array in IMAGE mode
            self.assertIsNotNone(result)
            mock_tile.assert_called_once()

    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.tile_and_save_media")
    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_backbone_features_forwarded_when_distinct_layers_present(
        self, mock_extract: MagicMock, mock_tile: MagicMock
    ) -> None:
        """When ``spatial_features_list`` carries two distinct layers (frozen-backbone + alignment-final),
        ``create_tiled_video`` extracts the backbone layer and forwards it to ``tile_and_save_media`` as
        ``camera_backbone_features`` / ``world_backbone_features``.
        """
        num_frames = 4
        H, W = 64, 64
        feat_h, feat_w = 8, 8
        feat_dim = 32

        mock_extract.return_value = (
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            30.0,
        )
        mock_tile.return_value = np.random.rand(H, W * 2, 3).astype(np.float32)

        camera_backbone = torch.randn(num_frames, feat_dim, feat_h, feat_w)
        camera_align = torch.randn(num_frames, feat_dim, feat_h, feat_w)
        world_backbone = torch.randn(num_frames, feat_dim, feat_h, feat_w)
        world_align = torch.randn(num_frames, feat_dim, feat_h, feat_w)

        output_batch: dict[str, Any] = {
            "camera_spatial_features_list": [camera_backbone, camera_align],
            "world_spatial_features_list": [world_backbone, world_align],
        }
        data_batch = {
            "camera_image": torch.randn(num_frames, 3, H, W),
            "world_image": torch.randn(num_frames, 3, H, W),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_tiled_video(output_batch, data_batch, 0, tmpdir, output_mode=TilingOutputMode.IMAGE, num_frames=2)

        mock_tile.assert_called_once()
        kwargs = mock_tile.call_args.kwargs
        self.assertIn("camera_backbone_features", kwargs)
        self.assertIn("world_backbone_features", kwargs)
        self.assertIsNotNone(kwargs["camera_backbone_features"])
        self.assertIsNotNone(kwargs["world_backbone_features"])

    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.tile_and_save_media")
    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_backbone_features_none_when_only_alignment_layer_present(
        self, mock_extract: MagicMock, mock_tile: MagicMock
    ) -> None:
        """When the model returns only the alignment-final layer (the ``[-1]``-only path), the backbone row should
        fall back to None so the downstream tile renders a "Backbone Feature N/A" placeholder.
        """
        num_frames = 4
        H, W = 64, 64
        feat_h, feat_w = 8, 8
        feat_dim = 32

        mock_extract.return_value = (
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            30.0,
        )
        mock_tile.return_value = np.random.rand(H, W * 2, 3).astype(np.float32)

        # Single tensor referenced twice mirrors the encoder's behavior for ``layer_idx=[-1, -1]``: the same Python
        # object appended twice.
        same_tensor = torch.randn(num_frames, feat_dim, feat_h, feat_w)
        output_batch: dict[str, Any] = {
            "camera_spatial_features_list": [same_tensor],
            "world_spatial_features_list": [same_tensor, same_tensor],
        }
        data_batch = {
            "camera_image": torch.randn(num_frames, 3, H, W),
            "world_image": torch.randn(num_frames, 3, H, W),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_tiled_video(output_batch, data_batch, 0, tmpdir, output_mode=TilingOutputMode.IMAGE, num_frames=2)

        mock_tile.assert_called_once()
        kwargs = mock_tile.call_args.kwargs
        self.assertIsNone(kwargs.get("camera_backbone_features"))
        self.assertIsNone(kwargs.get("world_backbone_features"))

    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.tile_and_save_media")
    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_video_mode_basic(self, mock_extract: MagicMock, mock_tile: MagicMock) -> None:
        """Test create_tiled_video in VIDEO mode."""
        num_frames = 4
        H, W = 64, 64
        feat_h, feat_w = 8, 8
        feat_dim = 32

        # Mock extract_frames_from_data_batch
        mock_extract.return_value = (
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            30.0,
        )

        output_batch: dict[str, Any] = {
            "camera_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
            "world_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
        }
        data_batch = {
            "camera_image": torch.randn(num_frames, 3, H, W),
            "world_image": torch.randn(num_frames, 3, H, W),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_tiled_video(output_batch, data_batch, 0, tmpdir, output_mode=TilingOutputMode.VIDEO)
            # Should return None in VIDEO mode
            self.assertIsNone(result)
            mock_tile.assert_called_once()

    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.tile_and_save_media")
    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_with_camera_patch_features(self, mock_extract: MagicMock, mock_tile: MagicMock) -> None:
        """Test create_tiled_video with camera_patch_features (3x2 layout)."""
        num_frames = 4
        H, W = 64, 64
        feat_h, feat_w = 8, 8
        feat_dim = 32

        mock_extract.return_value = (
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            30.0,
        )
        mock_tile.return_value = np.random.rand(H * 3, W * 2, 3).astype(np.float32)

        output_batch: dict[str, Any] = {
            "camera_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
            "world_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
            "camera_patch_features": torch.randn(num_frames, feat_dim, feat_h, feat_w),
        }
        data_batch = {
            "camera_image": torch.randn(num_frames, 3, H, W),
            "world_image": torch.randn(num_frames, 3, H, W),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_tiled_video(
                output_batch, data_batch, 0, tmpdir, output_mode=TilingOutputMode.IMAGE, num_frames=2
            )
            self.assertIsNotNone(result)

    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.tile_and_save_media")
    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_with_selected_indices(self, mock_extract: MagicMock, mock_tile: MagicMock) -> None:
        """Test create_tiled_video with selected_indices for frame pair sampling."""
        num_frames = 8
        H, W = 64, 64
        feat_h, feat_w = 8, 8
        feat_dim = 32

        mock_extract.return_value = (
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            30.0,
        )
        mock_tile.return_value = np.random.rand(H, W * 2, 3).astype(np.float32)

        # Create selected_indices (row_idx, col_idx) pairs
        # col_idx (2nd column) is used to index both images and features
        selected_indices = torch.tensor([[0, 0], [1, 2], [2, 4], [3, 6]])

        output_batch: dict[str, Any] = {
            # Features must have same frame count as images since selected_indices indexes both
            "camera_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
            "world_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
            "selected_indices": selected_indices,
        }
        data_batch = {
            "camera_image": torch.randn(num_frames, 3, H, W),
            "world_image": torch.randn(num_frames, 3, H, W),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_tiled_video(
                output_batch, data_batch, 0, tmpdir, output_mode=TilingOutputMode.IMAGE, num_frames=2
            )
            self.assertIsNotNone(result)


class TestGetRobustPcaFallback(unittest.TestCase):
    """Tests for _get_robust_pca_with_timing PCA fallback paths."""

    def test_pca_fallback_few_features(self) -> None:
        """Test PCA fallback when fewer than 3 features are available."""
        # Only 2 features (fewer than 3 required for RGB)
        features = torch.randn(10, 2, dtype=torch.float32)
        with patch("torch.pca_lowrank", side_effect=RuntimeError("ill-conditioned")):
            R, cmin, cmax = _get_robust_pca_with_timing(features, m=2, remove_first_component=False, skip=0)
            self.assertEqual(R.shape[1], 3)
            self.assertEqual(cmin.shape[0], 3)
            self.assertEqual(cmax.shape[0], 3)

    def test_pca_fallback_with_skip(self) -> None:
        """Test PCA fallback with skip > 0 and enough features."""
        features = torch.randn(20, 8, dtype=torch.float32)
        with patch("torch.pca_lowrank", side_effect=RuntimeError("ill-conditioned")):
            R, _cmin, _cmax = _get_robust_pca_with_timing(features, m=2, remove_first_component=False, skip=2)
            self.assertEqual(R.shape[1], 3)

    def test_pca_fallback_exact_3_features(self) -> None:
        """Test PCA fallback when exactly 3 features are available."""
        features = torch.randn(10, 3, dtype=torch.float32)
        with patch("torch.pca_lowrank", side_effect=RuntimeError("ill-conditioned")):
            R, _cmin, _cmax = _get_robust_pca_with_timing(features, m=2, remove_first_component=False, skip=0)
            self.assertEqual(R.shape[1], 3)


class TestComputeSpatialCosineSimilarity(unittest.TestCase):
    """Tests for compute_spatial_cosine_similarity_heatmaps."""

    def test_basic_cosine_similarity(self) -> None:
        """Test basic spatial cosine similarity computation."""
        num_frames = 2
        feat_dim, feat_h, feat_w = 16, 4, 4
        target_size = (8, 8)

        output_batch = {
            "camera_spatial_features": torch.randn(num_frames, feat_dim, feat_h, feat_w),
            "world_spatial_features": torch.randn(num_frames, feat_dim, feat_h, feat_w),
        }

        result = compute_spatial_cosine_similarity_heatmaps(output_batch, None, None, None, target_size=target_size)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.shape[0], num_frames)

    def test_cosine_similarity_missing_features(self) -> None:
        """Test returns None when features are missing."""
        output_batch: dict[str, Any] = {}  # No features
        result = compute_spatial_cosine_similarity_heatmaps(output_batch, None, None, None, target_size=(8, 8))
        self.assertIsNone(result)

    def test_cosine_similarity_wrong_dims(self) -> None:
        """Test returns None when features have wrong dimensions."""
        output_batch = {
            "camera_spatial_features": torch.randn(2, 16),  # 2D instead of 4D
            "world_spatial_features": torch.randn(2, 16),
        }
        result = compute_spatial_cosine_similarity_heatmaps(output_batch, None, None, None, target_size=(8, 8))
        self.assertIsNone(result)

    def test_cosine_similarity_zero_world(self) -> None:
        """Test behavior when all world features are zero."""
        num_frames = 2
        feat_dim, feat_h, feat_w = 16, 4, 4
        output_batch = {
            "camera_spatial_features": torch.randn(num_frames, feat_dim, feat_h, feat_w),
            "world_spatial_features": torch.zeros(num_frames, feat_dim, feat_h, feat_w),
        }
        result = compute_spatial_cosine_similarity_heatmaps(output_batch, None, None, None, target_size=(8, 8))
        # May return None or array with NaN values depending on implementation
        if result is not None:
            self.assertEqual(result.shape[0], num_frames)

    def test_cosine_similarity_with_cached_result(self) -> None:
        """Test uses cached result when available."""
        cached = np.random.rand(2, 8, 8, 3).astype(np.float32)
        output_batch = {
            "camera_spatial_features": torch.randn(2, 16, 4, 4),
            "world_spatial_features": torch.randn(2, 16, 4, 4),
            "_spatial_cosine_sim_rgb_cache_8x8": cached,
        }
        result = compute_spatial_cosine_similarity_heatmaps(output_batch, None, None, None, target_size=(8, 8))
        self.assertIsNotNone(result)
        np.testing.assert_array_equal(result, cached)

    def test_cosine_similarity_with_frame_indices(self) -> None:
        """Test cosine similarity with frame index selection."""
        num_frames = 4
        feat_dim, feat_h, feat_w = 16, 4, 4

        output_batch = {
            "camera_spatial_features": torch.randn(num_frames, feat_dim, feat_h, feat_w),
            "world_spatial_features": torch.randn(num_frames, feat_dim, feat_h, feat_w),
        }

        indices_to_use = torch.tensor([[0, 1], [2, 3]])
        result = compute_spatial_cosine_similarity_heatmaps(
            output_batch,
            indices_to_use,
            [1, 3],
            [0, 2],
            target_size=(8, 8),
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.shape[0], 2)


class TestCreateTiledVideoExtended(unittest.TestCase):
    """Extended tests for create_tiled_video function."""

    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.tile_and_save_media")
    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_with_vis_balanced_indices(self, mock_extract: MagicMock, mock_tile: MagicMock) -> None:
        """Test create_tiled_video with vis_balanced_indices for balanced pair sampling."""
        num_frames = 8
        H, W = 32, 32
        feat_h, feat_w = 4, 4
        feat_dim = 16

        mock_extract.return_value = (
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            30.0,
        )
        mock_tile.return_value = np.random.rand(H, W * 2, 3).astype(np.float32)

        vis_balanced_indices = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        vis_balanced_pair_indices = torch.tensor([0, 3, 5, 7])

        output_batch: dict[str, Any] = {
            "camera_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
            "world_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
            "vis_balanced_indices": vis_balanced_indices,
            "vis_balanced_pair_indices": vis_balanced_pair_indices,
        }
        data_batch = {
            "camera_image": torch.randn(num_frames, 3, H, W),
            "world_image": torch.randn(num_frames, 3, H, W),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_tiled_video(
                output_batch, data_batch, 0, tmpdir, output_mode=TilingOutputMode.IMAGE, num_frames=2
            )
            self.assertIsNotNone(result)

    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.tile_and_save_media")
    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_with_similarity_matrix(self, mock_extract: MagicMock, mock_tile: MagicMock) -> None:
        """Test create_tiled_video with similarity_matrix for text overlays."""
        num_frames = 4
        H, W = 32, 32
        feat_h, feat_w = 4, 4
        feat_dim = 16

        mock_extract.return_value = (
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            30.0,
        )
        mock_tile.return_value = np.random.rand(H, W * 2, 3).astype(np.float32)

        output_batch: dict[str, Any] = {
            "camera_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
            "world_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
            "similarity_matrix": torch.rand(num_frames),
        }
        data_batch = {
            "camera_image": torch.randn(num_frames, 3, H, W),
            "world_image": torch.randn(num_frames, 3, H, W),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_tiled_video(
                output_batch, data_batch, 0, tmpdir, output_mode=TilingOutputMode.IMAGE, num_frames=2
            )
            self.assertIsNotNone(result)

    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.tile_and_save_media")
    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_with_patch_features(self, mock_extract: MagicMock, mock_tile: MagicMock) -> None:
        """Test create_tiled_video with camera_patch_features."""
        num_frames = 4
        H, W = 32, 32
        feat_h, feat_w = 4, 4
        feat_dim = 16

        mock_extract.return_value = (
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            30.0,
        )
        mock_tile.return_value = np.random.rand(H, W * 2, 3).astype(np.float32)

        output_batch: dict[str, Any] = {
            "camera_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
            "world_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
            "camera_patch_features": torch.randn(num_frames, feat_dim, feat_h, feat_w),
        }
        data_batch = {
            "camera_image": torch.randn(num_frames, 3, H, W),
            "world_image": torch.randn(num_frames, 3, H, W),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_tiled_video(
                output_batch, data_batch, 0, tmpdir, output_mode=TilingOutputMode.IMAGE, num_frames=2
            )
            self.assertIsNotNone(result)

    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.tile_and_save_media")
    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_video_mode(self, mock_extract: MagicMock, mock_tile: MagicMock) -> None:
        """Test create_tiled_video with VIDEO output mode."""
        num_frames = 4
        H, W = 32, 32
        feat_h, feat_w = 4, 4
        feat_dim = 16

        mock_extract.return_value = (
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            30.0,
        )
        mock_tile.return_value = np.random.rand(H, W * 2, 3).astype(np.float32)

        output_batch: dict[str, Any] = {
            "camera_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
            "world_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
        }
        data_batch = {
            "camera_image": torch.randn(num_frames, 3, H, W),
            "world_image": torch.randn(num_frames, 3, H, W),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            _ = create_tiled_video(
                output_batch, data_batch, 0, tmpdir, output_mode=TilingOutputMode.VIDEO, num_frames=2
            )


class TestAddTextOverlaysExtended(unittest.TestCase):
    """Extended tests for add_text_overlays function."""

    def test_basic_overlay_with_output_batch(self) -> None:
        """Test basic text overlay on frames using output_batch."""
        frames = (np.random.rand(2, 64, 64, 3)).astype(np.float32)
        output_batch = {"similarity_matrix": torch.tensor([0.5, -0.3])}
        result = add_text_overlays(frames, output_batch=output_batch, frame_offset=0, view_type="camera")
        self.assertEqual(result.shape, frames.shape)

    def test_overlay_no_similarity_in_output_batch(self) -> None:
        """Test text overlay without similarity matrix in output_batch."""
        frames = (np.random.rand(2, 64, 64, 3)).astype(np.float32)
        output_batch: dict[str, Any] = {}
        result = add_text_overlays(frames, output_batch=output_batch, frame_offset=0, view_type="world")
        self.assertEqual(result.shape, frames.shape)

    def test_overlay_with_uint8_range_images(self) -> None:
        """Test overlay when images are in [0, 255] range."""
        frames = (np.random.rand(2, 64, 64, 3) * 255).astype(np.float32)
        output_batch = {"similarity_matrix": torch.tensor([0.8, 0.3])}
        result = add_text_overlays(frames, output_batch=output_batch, frame_offset=0, view_type="camera")
        self.assertEqual(result.shape, frames.shape)

    def test_overlay_with_2d_similarity_matrix(self) -> None:
        """Test overlay with 2D similarity matrix."""
        frames = (np.random.rand(3, 64, 64, 3)).astype(np.float32)
        sim_2d = torch.eye(3, dtype=torch.float32)
        output_batch = {"similarity_matrix": sim_2d}
        result = add_text_overlays(frames, output_batch=output_batch, frame_offset=0, view_type="camera")
        self.assertEqual(result.shape, frames.shape)

    def test_overlay_with_session_ids_and_frame_indices(self) -> None:
        """Test overlay with session IDs and frame indices."""
        frames = (np.random.rand(2, 128, 128, 3)).astype(np.float32)
        output_batch = {"similarity_matrix": torch.tensor([0.5, 0.8])}
        result = add_text_overlays(
            frames,
            output_batch=output_batch,
            frame_offset=0,
            view_type="camera",
            session_ids=["sess1", "sess2"],
            frame_indices=[10, 20],
        )
        self.assertEqual(result.shape, frames.shape)

    def test_overlay_with_similarity_indices(self) -> None:
        """Test overlay with explicit similarity_indices."""
        frames = (np.random.rand(2, 64, 64, 3)).astype(np.float32)
        sim_1d = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        output_batch = {"similarity_matrix": sim_1d}
        result = add_text_overlays(
            frames,
            output_batch=output_batch,
            frame_offset=0,
            view_type="camera",
            similarity_indices=[1, 3],
        )
        self.assertEqual(result.shape, frames.shape)

    def test_overlay_with_2d_similarity_and_indices(self) -> None:
        """Test overlay with 2D similarity matrix and explicit indices."""
        frames = (np.random.rand(2, 64, 64, 3)).astype(np.float32)
        sim_2d = torch.rand(5, 5)
        output_batch = {"similarity_matrix": sim_2d}
        result = add_text_overlays(
            frames,
            output_batch=output_batch,
            frame_offset=0,
            view_type="camera",
            similarity_indices=[1, 3],
        )
        self.assertEqual(result.shape, frames.shape)


class TestCreateTiledVideoWithSessionInfo(unittest.TestCase):
    """Tests for create_tiled_video with session ID/frame index extraction."""

    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.tile_and_save_media")
    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_with_session_ids_and_frame_indices(self, mock_extract: MagicMock, mock_tile: MagicMock) -> None:
        """Test create_tiled_video with session IDs and frame indices in data_batch."""
        num_frames = 4
        H, W = 32, 32
        feat_dim, feat_h, feat_w = 16, 4, 4

        mock_extract.return_value = (
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            30.0,
        )
        mock_tile.return_value = np.random.rand(H, W * 2, 3).astype(np.float32)

        output_batch: dict[str, Any] = {
            "camera_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
            "world_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
            "similarity_matrix": torch.rand(num_frames),
        }
        data_batch: dict[str, Any] = {
            "camera_image": torch.randn(num_frames, 3, H, W),
            "world_image": torch.randn(num_frames, 3, H, W),
            "camera_session_id": ["sess_a", "sess_a", "sess_b", "sess_b"],
            "camera_frame_idx": [10, 20, 30, 40],
            "world_session_id": ["wsess_1", "wsess_1", "wsess_2", "wsess_2"],
            "world_frame_idx": [100, 200, 300, 400],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_tiled_video(
                output_batch, data_batch, 0, tmpdir, output_mode=TilingOutputMode.IMAGE, num_frames=2
            )
            self.assertIsNotNone(result)

    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.tile_and_save_media")
    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_with_session_ids_and_selected_indices(self, mock_extract: MagicMock, mock_tile: MagicMock) -> None:
        """Test create_tiled_video with session IDs when using selected_indices."""
        num_frames = 8
        H, W = 32, 32
        feat_dim, feat_h, feat_w = 16, 4, 4

        mock_extract.return_value = (
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            30.0,
        )
        mock_tile.return_value = np.random.rand(H, W * 2, 3).astype(np.float32)

        selected_indices = torch.tensor([[0, 0], [1, 2], [2, 4], [3, 6]])

        output_batch: dict[str, Any] = {
            "camera_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
            "world_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)],
            "selected_indices": selected_indices,
        }
        data_batch: dict[str, Any] = {
            "camera_image": torch.randn(num_frames, 3, H, W),
            "world_image": torch.randn(num_frames, 3, H, W),
            "camera_session_id": [f"sess_{i}" for i in range(num_frames)],
            "camera_frame_idx": list(range(num_frames)),
            "world_session_id": [f"wsess_{i}" for i in range(num_frames)],
            "world_frame_idx": list(range(100, 100 + num_frames)),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_tiled_video(
                output_batch, data_batch, 0, tmpdir, output_mode=TilingOutputMode.IMAGE, num_frames=2
            )
            self.assertIsNotNone(result)


class TestAddTextOverlaysEdgeCases(unittest.TestCase):
    """Edge case tests for add_text_overlays."""

    def test_1d_similarity_indices_out_of_bounds(self) -> None:
        """Test with out-of-bounds similarity_indices for 1D matrix."""
        frames = (np.random.rand(2, 64, 64, 3)).astype(np.float32)
        sim_1d = torch.tensor([0.1, 0.2])  # Only 2 elements
        output_batch = {"similarity_matrix": sim_1d}
        # Indices [5, 10] are out of bounds
        result = add_text_overlays(
            frames,
            output_batch=output_batch,
            frame_offset=0,
            view_type="camera",
            similarity_indices=[5, 10],
        )
        self.assertEqual(result.shape, frames.shape)

    def test_2d_similarity_indices_out_of_bounds(self) -> None:
        """Test with out-of-bounds similarity_indices for 2D matrix."""
        frames = (np.random.rand(2, 64, 64, 3)).astype(np.float32)
        sim_2d = torch.rand(3, 3)  # Only 3x3
        output_batch = {"similarity_matrix": sim_2d}
        # Indices [5, 10] are out of bounds
        result = add_text_overlays(
            frames,
            output_batch=output_batch,
            frame_offset=0,
            view_type="camera",
            similarity_indices=[5, 10],
        )
        self.assertEqual(result.shape, frames.shape)

    def test_1d_similarity_length_mismatch(self) -> None:
        """Test with 1D similarity matrix that doesn't match image count."""
        frames = (np.random.rand(3, 64, 64, 3)).astype(np.float32)
        sim_1d = torch.tensor([0.1, 0.2])  # 2 scores for 3 frames - mismatch
        output_batch = {"similarity_matrix": sim_1d}
        result = add_text_overlays(frames, output_batch=output_batch, frame_offset=0, view_type="camera")
        self.assertEqual(result.shape, frames.shape)

    def test_2d_similarity_shape_mismatch(self) -> None:
        """Test with 2D similarity matrix that doesn't match image count."""
        frames = (np.random.rand(3, 64, 64, 3)).astype(np.float32)
        sim_2d = torch.rand(5, 5)  # 5x5 for 3 frames
        output_batch = {"similarity_matrix": sim_2d}
        result = add_text_overlays(frames, output_batch=output_batch, frame_offset=0, view_type="camera")
        self.assertEqual(result.shape, frames.shape)

    def test_3d_similarity_matrix(self) -> None:
        """Test with unexpected 3D similarity matrix shape."""
        frames = (np.random.rand(2, 64, 64, 3)).astype(np.float32)
        sim_3d = torch.rand(2, 3, 4)  # 3D - unexpected
        output_batch = {"similarity_matrix": sim_3d}
        result = add_text_overlays(frames, output_batch=output_batch, frame_offset=0, view_type="camera")
        self.assertEqual(result.shape, frames.shape)

    def test_session_ids_without_frame_indices(self) -> None:
        """Test overlay with session IDs but no frame indices."""
        frames = (np.random.rand(2, 64, 64, 3)).astype(np.float32)
        output_batch: dict[str, Any] = {}
        result = add_text_overlays(
            frames,
            output_batch=output_batch,
            frame_offset=5,
            view_type="camera",
            session_ids=["sess1", "sess2"],
        )
        self.assertEqual(result.shape, frames.shape)


def _overlay_frames(count: int = 6, height: int = 48, width: int = 64) -> np.ndarray:
    return np.random.default_rng(0).random((count, height, width, 3)).astype(np.float32)


class TestAddTextOverlaysChunked(unittest.TestCase):
    """The chunked uint8 round trip keeps the output contract and a bounded peak allocation."""

    def test_shape_and_dtype_are_preserved(self) -> None:
        images = _overlay_frames()

        out = add_text_overlays(images, {}, frame_offset=0, view_type="camera")

        self.assertEqual(out.shape, images.shape)
        self.assertEqual(out.dtype, np.float32)

    def test_result_stays_in_unit_range(self) -> None:
        out = add_text_overlays(_overlay_frames(), {"similarity_matrix": torch.eye(6)}, 0, "camera")

        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_every_frame_is_drawn_on(self) -> None:
        """Frames reach OpenCV through a chunk-spanning index, so an off-by-one would leave a chunk unlabeled."""
        images = _overlay_frames(count=9)

        out = add_text_overlays(images, {}, frame_offset=0, view_type="camera")

        for frame in range(images.shape[0]):
            with self.subTest(frame=frame):
                self.assertFalse(np.allclose(out[frame], images[frame]))

    def test_input_is_not_mutated(self) -> None:
        images = _overlay_frames()
        before = images.copy()

        add_text_overlays(images, {}, frame_offset=0, view_type="camera")

        np.testing.assert_array_equal(images, before)

    def test_single_frame(self) -> None:
        out = add_text_overlays(_overlay_frames(count=1), {}, frame_offset=0, view_type="world")

        self.assertEqual(out.shape[0], 1)

    def test_zero_to_255_input_is_rescaled(self) -> None:
        out = add_text_overlays(_overlay_frames() * 255.0, {}, frame_offset=0, view_type="camera")

        self.assertLessEqual(float(out.max()), 1.0)

    def test_frames_spanning_several_chunks_are_all_drawn_on(self) -> None:
        """At 256x256x3 a chunk holds 42 frames, so 100 frames cover three chunks including a partial one."""
        images = _overlay_frames(count=100, height=256, width=256)

        out = add_text_overlays(images, {}, frame_offset=0, view_type="camera")

        for frame in (0, 41, 42, 83, 84, 99):
            with self.subTest(frame=frame):
                self.assertFalse(np.allclose(out[frame], images[frame]))

    def test_peak_allocation_stays_near_one_buffer(self) -> None:
        """One float32 result plus uint8 chunks measures about 1.25x the input; a full-size temporary exceeds 2x."""
        images = _overlay_frames(count=16, height=256, width=256)

        tracemalloc.start()
        try:
            add_text_overlays(images, {}, frame_offset=0, view_type="camera")
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        self.assertLess(peak, 1.6 * images.nbytes)


class TestCreateTiledVideoEmptyFeatures(unittest.TestCase):
    """Test create_tiled_video with empty spatial features (covers L503-505)."""

    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils")
    def test_returns_none_for_empty_features(self, mock_basic_vis: MagicMock) -> None:
        """When no spatial features are available, should return early."""
        output_batch = {
            "camera_spatial_features": torch.zeros(0, 64, 16, 16),  # Empty
            "world_spatial_features": torch.zeros(0, 64, 16, 16),  # Empty
            "camera_global_features": torch.zeros(0, 64),
            "world_global_features": torch.zeros(0, 64),
        }
        data_batch = {
            "camera_image": torch.zeros(0, 3, 32, 32),
            "world_image": torch.zeros(0, 3, 32, 32),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_tiled_video(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                camera_start_frame=0,
                world_start_frame=0,
                output_mode=TilingOutputMode.IMAGE,
            )
            self.assertIsNone(result)


class TestLabelFrames(unittest.TestCase):
    """Tests for the _label_frames helper used by the camera-only tiled video."""

    def test_preserves_shape_and_range(self) -> None:
        frames = np.random.rand(3, 40, 90, 3).astype(np.float32)
        out = _label_frames(frames, "Backbone Feature")
        self.assertEqual(out.shape, frames.shape)
        self.assertEqual(out.dtype, np.float32)
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)

    def test_does_not_mutate_input(self) -> None:
        frames = np.random.rand(2, 32, 64, 3).astype(np.float32)
        original = frames.copy()
        _label_frames(frames, "After-Alignment Feature")
        np.testing.assert_array_equal(frames, original)


class TestCreateCameraOnlyTiledVideo(unittest.TestCase):
    """Tests for create_camera_only_tiled_video (camera-only tiled video)."""

    def test_missing_camera_image(self) -> None:
        output_batch: dict[str, Any] = {"camera_spatial_features_list": [torch.randn(2, 16, 8, 8)]}
        with tempfile.TemporaryDirectory() as tmpdir:
            create_camera_only_tiled_video(output_batch, {}, 0, tmpdir)
            self.assertEqual(os.listdir(tmpdir), [])  # early exit — no file written

    def test_missing_camera_features_list(self) -> None:
        data_batch = {"camera_image": torch.randn(2, 3, 64, 64)}
        with tempfile.TemporaryDirectory() as tmpdir:
            create_camera_only_tiled_video({}, data_batch, 0, tmpdir)
            self.assertEqual(os.listdir(tmpdir), [])

    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_extract_frames_returns_none(self, mock_extract: MagicMock) -> None:
        mock_extract.return_value = None
        output_batch: dict[str, Any] = {"camera_spatial_features_list": [torch.randn(2, 16, 8, 8)]}
        data_batch = {"camera_image": torch.randn(2, 3, 64, 64)}
        with tempfile.TemporaryDirectory() as tmpdir:
            create_camera_only_tiled_video(output_batch, data_batch, 0, tmpdir)
            self.assertEqual(os.listdir(tmpdir), [])

    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.image_video_utils.write_video_with_ffmpeg")
    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.image_video_utils.tiled_frames_generator")
    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_two_layers_produces_2x2_camera_paired(
        self, mock_extract: MagicMock, mock_gen: MagicMock, mock_write: MagicMock
    ) -> None:
        """Two distinct encoder layers (backbone + after-alignment) -> 2x2 with the camera reused per row."""
        num_frames, feat_dim, feat_h, feat_w = 3, 16, 8, 8
        H, W = 72, 128
        mock_extract.return_value = (np.random.rand(num_frames, H, W, 3).astype(np.float32), None, 30.0)
        mock_gen.return_value = iter([])

        backbone = torch.randn(num_frames, feat_dim, feat_h, feat_w)
        aligned = torch.randn(num_frames, feat_dim, feat_h, feat_w)
        output_batch: dict[str, Any] = {"camera_spatial_features_list": [backbone, aligned]}
        data_batch = {"camera_image": torch.randn(num_frames, 3, H, W)}

        with tempfile.TemporaryDirectory() as tmpdir:
            create_camera_only_tiled_video(output_batch, data_batch, 0, tmpdir)

        mock_gen.assert_called_once()
        tiles, rows, cols = mock_gen.call_args.args[0], mock_gen.call_args.args[1], mock_gen.call_args.args[2]
        self.assertEqual((rows, cols), (2, 2))
        self.assertEqual(len(tiles), 4)
        # Every panel is upscaled to the camera resolution *before* labeling (the "eature" clip fix).
        for tile in tiles:
            self.assertEqual(tile.shape[1:3], (H, W))
        # Camera is the per-row reference: row-0 col-0 and row-1 col-0 are the same panel.
        np.testing.assert_array_equal(tiles[0], tiles[2])
        mock_write.assert_called_once()
        _gen, _path, out_w, out_h, _fps = mock_write.call_args.args
        self.assertEqual((out_w, out_h), (W * 2, H * 2))

    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.image_video_utils.write_video_with_ffmpeg")
    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.image_video_utils.tiled_frames_generator")
    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_single_layer_falls_back_to_1x2(
        self, mock_extract: MagicMock, mock_gen: MagicMock, mock_write: MagicMock
    ) -> None:
        """A single encoder layer (no distinct backbone) -> 1x2 [camera | after-alignment]."""
        num_frames, feat_dim, feat_h, feat_w = 2, 16, 8, 8
        H, W = 64, 96
        mock_extract.return_value = (np.random.rand(num_frames, H, W, 3).astype(np.float32), None, 30.0)
        mock_gen.return_value = iter([])

        output_batch: dict[str, Any] = {
            "camera_spatial_features_list": [torch.randn(num_frames, feat_dim, feat_h, feat_w)]
        }
        data_batch = {"camera_image": torch.randn(num_frames, 3, H, W)}

        with tempfile.TemporaryDirectory() as tmpdir:
            create_camera_only_tiled_video(output_batch, data_batch, 0, tmpdir)

        mock_gen.assert_called_once()
        tiles, rows, cols = mock_gen.call_args.args[0], mock_gen.call_args.args[1], mock_gen.call_args.args[2]
        self.assertEqual((rows, cols), (1, 2))
        self.assertEqual(len(tiles), 2)
        for tile in tiles:
            self.assertEqual(tile.shape[1:3], (H, W))
        mock_write.assert_called_once()
        _gen, _path, out_w, out_h, _fps = mock_write.call_args.args
        self.assertEqual((out_w, out_h), (W * 2, H))

    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.image_video_utils.write_video_with_ffmpeg")
    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.image_video_utils.tiled_frames_generator")
    @patch("checks.utils.models.cwip.visualization.feature_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_same_object_twice_is_not_a_backbone(
        self, mock_extract: MagicMock, mock_gen: MagicMock, mock_write: MagicMock
    ) -> None:
        """``layer_idx=[-1]`` returns the same tensor object twice; the ``is`` check must treat that as no backbone."""
        num_frames, feat_dim, feat_h, feat_w = 2, 16, 8, 8
        H, W = 64, 64
        mock_extract.return_value = (np.random.rand(num_frames, H, W, 3).astype(np.float32), None, 30.0)
        mock_gen.return_value = iter([])

        same = torch.randn(num_frames, feat_dim, feat_h, feat_w)
        output_batch: dict[str, Any] = {"camera_spatial_features_list": [same, same]}
        data_batch = {"camera_image": torch.randn(num_frames, 3, H, W)}

        with tempfile.TemporaryDirectory() as tmpdir:
            create_camera_only_tiled_video(output_batch, data_batch, 0, tmpdir)

        rows, cols = mock_gen.call_args.args[1], mock_gen.call_args.args[2]
        self.assertEqual((rows, cols), (1, 2))


if __name__ == "__main__":
    unittest.main()
