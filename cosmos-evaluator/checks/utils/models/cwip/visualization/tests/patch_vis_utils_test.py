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

import tempfile
from typing import Any
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from checks.utils.models.cwip.modules.class_mapping import (
    ObjectTypeClassMappingV3,
    SimilarityClassMappingV1,
)
from checks.utils.models.cwip.visualization.patch_vis_utils import (
    _add_label_overlay,
    _add_patch_text_overlays,
    _add_probability_map_text_overlays,
    _build_type_colors_tensor,
    create_combined_patch_visualization,
    create_detailed_patch_visualization,
    mask_excluded_classes_for_argmax,
    normalize_excluded_classes_for_argmax,
)


class TestAddLabelOverlay(unittest.TestCase):
    """Tests for _add_label_overlay function."""

    def test_top_position(self) -> None:
        """Test adding label overlay to images at top position."""
        num_frames = 5
        H, W = 100, 200

        # Create dummy images in [0, 1] range
        images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        label = "Test Label"

        result = _add_label_overlay(images, label)

        # Check shape is preserved
        self.assertEqual(result.shape, images.shape)

        # Check output is in [0, 1] range
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)

        # Check that the result is different from input (text was added)
        self.assertFalse(np.allclose(result, images))

    def test_preserves_most_pixels(self) -> None:
        """Test that adding label overlay only modifies pixels where text is drawn."""
        num_frames = 3
        H, W = 100, 200

        images = np.random.rand(num_frames, H, W, 3).astype(np.float32)
        label = "Label"

        result = _add_label_overlay(images, label)

        # Most pixels should be unchanged (only text region changes)
        # Check that at least 90% of pixels are very close to original
        diff = np.abs(result - images)
        unchanged_pixels = (diff < 0.01).all(axis=-1)  # Per-pixel check across RGB
        unchanged_ratio = unchanged_pixels.sum() / unchanged_pixels.size

        self.assertGreater(unchanged_ratio, 0.9, f"Expected >90% unchanged pixels, got {unchanged_ratio * 100:.1f}%")


class TestAddProbabilityMapTextOverlays(unittest.TestCase):
    """Tests for _add_probability_map_text_overlays function."""

    def test_similarity_type(self) -> None:
        """Test adding text overlays for similarity type probability maps."""
        num_frames = 5
        H, W = 100, 200
        class_id = 1  # MATCH class
        frame_offset = 0

        # Create dummy probability map images in [0, 1] range
        prob_map_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)

        result = _add_probability_map_text_overlays(
            prob_map_images, class_id, frame_offset, class_mapping=SimilarityClassMappingV1
        )

        # Check shape is preserved
        self.assertEqual(result.shape, prob_map_images.shape)

        # Check output is in [0, 1] range
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)

        # Check that the result is different from input (text was added)
        self.assertFalse(np.allclose(result, prob_map_images))

    def test_object_type(self) -> None:
        """Test adding text overlays for object type probability maps."""
        num_frames = 3
        H, W = 80, 160
        class_id = 0  # First object type class
        frame_offset = 10

        prob_map_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)

        result = _add_probability_map_text_overlays(
            prob_map_images, class_id, frame_offset, class_mapping=ObjectTypeClassMappingV3
        )

        # Check shape is preserved
        self.assertEqual(result.shape, prob_map_images.shape)

        # Check output is in [0, 1] range
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)

    def test_with_offset(self) -> None:
        """Test that frame offset affects frame numbering in overlays."""
        num_frames = 2
        H, W = 64, 128
        class_id = 2

        prob_map_images = np.random.rand(num_frames, H, W, 3).astype(np.float32)

        # Test with different offsets
        result_offset_0 = _add_probability_map_text_overlays(
            prob_map_images, class_id, frame_offset=0, class_mapping=SimilarityClassMappingV1
        )
        result_offset_100 = _add_probability_map_text_overlays(
            prob_map_images, class_id, frame_offset=100, class_mapping=SimilarityClassMappingV1
        )

        # Results should be different (different frame numbers)
        self.assertFalse(np.allclose(result_offset_0, result_offset_100))


class TestAddPatchTextOverlays(unittest.TestCase):
    """Tests for _add_patch_text_overlays function."""

    def test_basic(self) -> None:
        """Test adding patch text overlays to images."""
        num_frames = 5
        H, W = 100, 200

        images = np.random.rand(num_frames, H, W, 3).astype(np.float32)

        # Create mock patch_results
        patch_results: dict[str, Any] = {
            "per_frame_class_percentages": {
                1: [10.0, 15.0, 20.0, 25.0, 30.0],  # MATCH percentages
                2: [5.0, 8.0, 12.0, 15.0, 18.0],  # WRONG percentages
                3: [3.0, 4.0, 5.0, 6.0, 7.0],  # EXTRA percentages
                4: [2.0, 3.0, 4.0, 5.0, 6.0],  # MISSING percentages
            },
            "per_frame_class_pixel_counts": {
                1: [1000, 1500, 2000, 2500, 3000],
                2: [500, 800, 1200, 1500, 1800],
                3: [300, 400, 500, 600, 700],
                4: [200, 300, 400, 500, 600],
            },
        }

        result = _add_patch_text_overlays(images, patch_results, frame_offset=0, view_type="camera")

        # Check shape is preserved
        self.assertEqual(result.shape, images.shape)

        # Check output is in [0, 1] range
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)

        # Check that the result is different from input (text was added)
        self.assertFalse(np.allclose(result, images))

    def test_world_view(self) -> None:
        """Test adding patch text overlays with world view type."""
        num_frames = 3
        H, W = 80, 160

        images = np.random.rand(num_frames, H, W, 3).astype(np.float32)

        patch_results: dict[str, Any] = {
            "per_frame_class_percentages": {
                1: [20.0, 25.0, 30.0],
            },
            "per_frame_class_pixel_counts": {
                1: [2000, 2500, 3000],
            },
        }

        result = _add_patch_text_overlays(images, patch_results, frame_offset=0, view_type="world")

        # Check shape is preserved
        self.assertEqual(result.shape, images.shape)

        # Check output is in [0, 1] range
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)

    def test_empty_results(self) -> None:
        """Test adding patch text overlays with empty results."""
        num_frames = 2
        H, W = 64, 128

        images = np.random.rand(num_frames, H, W, 3).astype(np.float32)

        # Empty patch results
        patch_results: dict[str, Any] = {"per_frame_class_percentages": {}, "per_frame_class_pixel_counts": {}}

        result = _add_patch_text_overlays(images, patch_results, frame_offset=0, view_type="camera")

        # Should still work and add frame number/view type
        self.assertEqual(result.shape, images.shape)
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)

    def test_with_frame_offset(self) -> None:
        """Test that frame offset is correctly applied in patch text overlays."""
        num_frames = 2
        H, W = 64, 128

        images = np.random.rand(num_frames, H, W, 3).astype(np.float32)

        patch_results: dict[str, Any] = {
            "per_frame_class_percentages": {1: [10.0, 15.0]},
            "per_frame_class_pixel_counts": {1: [1000, 1500]},
        }

        # Test with different offsets
        result_offset_0 = _add_patch_text_overlays(images, patch_results, frame_offset=0, view_type="camera")
        result_offset_50 = _add_patch_text_overlays(images, patch_results, frame_offset=50, view_type="camera")

        # Results should be different (different frame numbers)
        self.assertFalse(np.allclose(result_offset_0, result_offset_50))


class TestCreateDetailedPatchVisualization(unittest.TestCase):
    """Tests for create_detailed_patch_visualization function."""

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils._create_detailed_similarity_visualization")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils._create_detailed_type_visualization")
    def test_calls_similarity_visualization_when_available(
        self, mock_type_vis: MagicMock, mock_sim_vis: MagicMock
    ) -> None:
        """Test that similarity visualization is called when patch_similarity_logits is available."""
        output_batch = {
            "patch_similarity_logits": torch.randn(4, 5, 8, 8),
        }
        data_batch = {
            "camera_image": torch.randn(4, 3, 64, 64),
            "world_image": torch.randn(4, 3, 64, 64),
        }
        patch_results: dict[str, Any] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            create_detailed_patch_visualization(output_batch, data_batch, 0, tmpdir, patch_results)

        mock_sim_vis.assert_called_once()
        mock_type_vis.assert_not_called()

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils._create_detailed_similarity_visualization")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils._create_detailed_type_visualization")
    def test_calls_type_visualization_when_available(self, mock_type_vis: MagicMock, mock_sim_vis: MagicMock) -> None:
        """Test that type visualization is called when patch_type_logits is available."""
        output_batch = {
            "patch_type_logits": torch.randn(4, 22, 8, 8),
        }
        data_batch = {
            "camera_image": torch.randn(4, 3, 64, 64),
            "world_image": torch.randn(4, 3, 64, 64),
        }
        patch_results: dict[str, Any] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            create_detailed_patch_visualization(output_batch, data_batch, 0, tmpdir, patch_results)

        mock_sim_vis.assert_not_called()
        mock_type_vis.assert_called_once()

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils._create_detailed_similarity_visualization")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils._create_detailed_type_visualization")
    def test_calls_both_when_both_available(self, mock_type_vis: MagicMock, mock_sim_vis: MagicMock) -> None:
        """Test that both visualizations are called when both logits are available."""
        output_batch = {
            "patch_similarity_logits": torch.randn(4, 5, 8, 8),
            "patch_type_logits": torch.randn(4, 22, 8, 8),
        }
        data_batch = {
            "camera_image": torch.randn(4, 3, 64, 64),
            "world_image": torch.randn(4, 3, 64, 64),
        }
        patch_results: dict[str, Any] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            create_detailed_patch_visualization(output_batch, data_batch, 0, tmpdir, patch_results)

        mock_sim_vis.assert_called_once()
        mock_type_vis.assert_called_once()

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils._create_detailed_similarity_visualization")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils._create_detailed_type_visualization")
    def test_no_visualization_when_no_logits(self, mock_type_vis: MagicMock, mock_sim_vis: MagicMock) -> None:
        """Test that no visualizations are called when no logits are available."""
        output_batch: dict[str, Any] = {}
        data_batch = {
            "camera_image": torch.randn(4, 3, 64, 64),
            "world_image": torch.randn(4, 3, 64, 64),
        }
        patch_results: dict[str, Any] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            create_detailed_patch_visualization(output_batch, data_batch, 0, tmpdir, patch_results)

        mock_sim_vis.assert_not_called()
        mock_type_vis.assert_not_called()


class TestCreateCombinedPatchVisualization(unittest.TestCase):
    """Tests for create_combined_patch_visualization function."""

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_returns_early_when_extract_frames_fails(self, mock_extract: MagicMock) -> None:
        """Test that function returns early when extract_frames returns None."""
        mock_extract.return_value = None
        num_frames = 4

        output_batch = {
            "patch_similarity_logits": torch.randn(num_frames, 5, 8, 8),
        }
        data_batch = {
            "camera_image": torch.randn(num_frames, 3, 64, 64),
            "world_image": torch.randn(num_frames, 3, 64, 64),
        }
        patch_results: dict[str, Any] = {
            "per_frame_class_percentages": {1: [10.0] * num_frames},
            "per_frame_class_pixel_counts": {1: [100] * num_frames},
            "pred_class_indices": torch.ones(num_frames, 8, 8, dtype=torch.long),
            "pred_probabilities": torch.ones(num_frames, 8, 8),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should return without error
            create_combined_patch_visualization(output_batch, data_batch, 0, tmpdir, patch_results)

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.basic_vis_utils.tile_and_save_media")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_creates_visualization_with_similarity_logits(self, mock_extract: MagicMock, mock_tile: MagicMock) -> None:
        """Test that visualization is created with patch_similarity_logits."""
        num_frames = 4
        H, W = 64, 64

        mock_extract.return_value = (
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            30.0,
        )

        output_batch = {
            "patch_similarity_logits": torch.randn(num_frames, 5, 8, 8),
        }
        data_batch = {
            "camera_image": torch.randn(num_frames, 3, H, W),
            "world_image": torch.randn(num_frames, 3, H, W),
        }
        patch_results: dict[str, Any] = {
            "per_frame_class_percentages": {1: [10.0] * num_frames},
            "per_frame_class_pixel_counts": {1: [100] * num_frames},
            "pred_class_indices": torch.ones(num_frames, 8, 8, dtype=torch.long),
            "pred_probabilities": torch.ones(num_frames, 8, 8),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_combined_patch_visualization(output_batch, data_batch, 0, tmpdir, patch_results)

        # Should have called tile_and_save_media
        mock_tile.assert_called_once()

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.basic_vis_utils.tile_and_save_media")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_creates_visualization_with_gpu_encoding_false(self, mock_extract: MagicMock, mock_tile: MagicMock) -> None:
        """Test that visualization is created with use_gpu_encoding=False."""
        num_frames = 4
        H, W = 64, 64

        mock_extract.return_value = (
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            30.0,
        )

        output_batch = {
            "patch_similarity_logits": torch.randn(num_frames, 5, 8, 8),
        }
        data_batch = {
            "camera_image": torch.randn(num_frames, 3, H, W),
            "world_image": torch.randn(num_frames, 3, H, W),
        }
        patch_results: dict[str, Any] = {
            "per_frame_class_percentages": {1: [10.0] * num_frames},
            "per_frame_class_pixel_counts": {1: [100] * num_frames},
            "pred_class_indices": torch.ones(num_frames, 8, 8, dtype=torch.long),
            "pred_probabilities": torch.ones(num_frames, 8, 8),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_combined_patch_visualization(
                output_batch, data_batch, 0, tmpdir, patch_results, use_gpu_encoding=False
            )

        mock_tile.assert_called_once()


class TestNormalizeExcludedClassesForArgmax(unittest.TestCase):
    """Tests for normalize_excluded_classes_for_argmax function."""

    def test_none_input(self) -> None:
        """Test with None input."""
        self.assertIsNone(normalize_excluded_classes_for_argmax(None))

    def test_string_single_int(self) -> None:
        """Test with string '8'."""
        self.assertEqual(normalize_excluded_classes_for_argmax("8"), [8])

    def test_string_list(self) -> None:
        """Test with string '[8]'."""
        self.assertEqual(normalize_excluded_classes_for_argmax("[8]"), [8])

    def test_string_multi_list(self) -> None:
        """Test with string '[8,9]'."""
        self.assertEqual(normalize_excluded_classes_for_argmax("[8,9]"), [8, 9])

    def test_string_invalid(self) -> None:
        """Test with invalid string."""
        self.assertIsNone(normalize_excluded_classes_for_argmax("abc"))

    def test_string_non_list_type(self) -> None:
        """Test with string that evaluates to non-int/non-list."""
        self.assertIsNone(normalize_excluded_classes_for_argmax("{'a': 1}"))

    def test_list_ints(self) -> None:
        """Test with list of ints."""
        self.assertEqual(normalize_excluded_classes_for_argmax([8, 9]), [8, 9])

    def test_list_strings(self) -> None:
        """Test with list of string ints."""
        self.assertEqual(normalize_excluded_classes_for_argmax(["8", "9"]), [8, 9])

    def test_tuple_input(self) -> None:
        """Test with tuple."""
        self.assertEqual(normalize_excluded_classes_for_argmax((8, 9)), [8, 9])

    def test_set_input(self) -> None:
        """Test with set."""
        result = normalize_excluded_classes_for_argmax({8})
        self.assertEqual(result, [8])

    def test_non_iterable(self) -> None:
        """Test with non-iterable (int)."""
        self.assertIsNone(normalize_excluded_classes_for_argmax(42))


class TestMaskExcludedClassesForArgmax(unittest.TestCase):
    """Tests for mask_excluded_classes_for_argmax function."""

    def test_no_exclusion(self) -> None:
        """Test with no excluded classes."""
        logits = torch.randn(2, 5, 8, 8)
        result = mask_excluded_classes_for_argmax(logits, None, 5)
        self.assertTrue(torch.equal(result, logits))

    def test_empty_exclusion(self) -> None:
        """Test with empty excluded classes list."""
        logits = torch.randn(2, 5, 8, 8)
        result = mask_excluded_classes_for_argmax(logits, [], 5)
        self.assertTrue(torch.equal(result, logits))

    def test_exclude_logits(self) -> None:
        """Test masking logits (values can be negative)."""
        logits = torch.randn(2, 5, 8, 8)
        result = mask_excluded_classes_for_argmax(logits, [2], 5)

        # Class 2 should be -inf
        self.assertTrue(torch.all(result[:, 2, :, :] == float("-inf")))
        # Other classes unchanged
        self.assertTrue(torch.all(result[:, 0, :, :] == logits[:, 0, :, :]))
        self.assertTrue(torch.all(result[:, 1, :, :] == logits[:, 1, :, :]))

    def test_exclude_probabilities(self) -> None:
        """Test masking probabilities (all values >= 0)."""
        probs = torch.softmax(torch.randn(2, 5, 8, 8), dim=1)
        result = mask_excluded_classes_for_argmax(probs, [2], 5)

        # Class 2 should be 0
        self.assertTrue(torch.all(result[:, 2, :, :] == 0.0))

    def test_out_of_range_index(self) -> None:
        """Test with excluded class index out of range (should be ignored)."""
        logits = torch.randn(2, 5, 8, 8)
        result = mask_excluded_classes_for_argmax(logits, [10], 5)

        # Nothing should change since index 10 > num_classes
        self.assertTrue(torch.equal(result, logits))

    def test_multiple_exclusions(self) -> None:
        """Test excluding multiple classes."""
        logits = torch.randn(2, 5, 8, 8)
        result = mask_excluded_classes_for_argmax(logits, [0, 3, 4], 5)

        self.assertTrue(torch.all(result[:, 0, :, :] == float("-inf")))
        self.assertTrue(torch.all(result[:, 3, :, :] == float("-inf")))
        self.assertTrue(torch.all(result[:, 4, :, :] == float("-inf")))
        # Classes 1 and 2 unchanged
        self.assertTrue(torch.all(result[:, 1, :, :] == logits[:, 1, :, :]))
        self.assertTrue(torch.all(result[:, 2, :, :] == logits[:, 2, :, :]))


class TestBuildTypeColorsTensor(unittest.TestCase):
    """Tests for _build_type_colors_tensor function."""

    def test_basic(self) -> None:
        """Test basic type colors tensor building."""
        num_classes = 10
        colors = _build_type_colors_tensor(num_classes, ObjectTypeClassMappingV3, device="cpu")
        self.assertEqual(colors.shape, (num_classes, 3))
        self.assertEqual(colors.device.type, "cpu")
        # All values should be in [0, 1]
        self.assertTrue(torch.all(colors >= 0.0))
        self.assertTrue(torch.all(colors <= 1.0))

    def test_without_mapping(self) -> None:
        """Test type colors tensor building without mapping (direct index)."""
        num_classes = 5
        colors = _build_type_colors_tensor(num_classes, None, device="cpu")
        self.assertEqual(colors.shape, (num_classes, 3))

    def test_more_classes_than_types(self) -> None:
        """Test with more classes than mapped ObjectType values."""
        num_classes = 50
        colors = _build_type_colors_tensor(num_classes, ObjectTypeClassMappingV3, device="cpu")
        self.assertEqual(colors.shape, (num_classes, 3))


class TestCreateDetailedPatchVisualizationWithStringMapping(unittest.TestCase):
    """Tests for create_detailed_patch_visualization with string mapping."""

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils._create_detailed_similarity_visualization")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils._create_detailed_type_visualization")
    def test_string_mapping_resolves(self, mock_type_vis: MagicMock, mock_sim_vis: MagicMock) -> None:
        """Test that string mapping 'v3' is resolved to ObjectTypeClassMappingV3."""
        output_batch = {
            "patch_type_logits": torch.randn(4, 17, 8, 8),
        }
        data_batch = {
            "camera_image": torch.randn(4, 3, 64, 64),
            "world_image": torch.randn(4, 3, 64, 64),
        }
        patch_results: dict[str, Any] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            create_detailed_patch_visualization(
                output_batch,
                data_batch,
                0,
                tmpdir,
                patch_results,
                object_type_class_mapping="v3",
            )

        mock_type_vis.assert_called_once()

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils._create_detailed_similarity_visualization")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils._create_detailed_type_visualization")
    def test_excluded_type_passed(self, mock_type_vis: MagicMock, mock_sim_vis: MagicMock) -> None:
        """Test that excluded_model_classes_for_argmax_type is passed through."""
        output_batch: dict[str, Any] = {
            "patch_type_logits": torch.randn(4, 17, 8, 8),
            "excluded_model_classes_for_argmax_type": [0],
        }
        data_batch = {
            "camera_image": torch.randn(4, 3, 64, 64),
            "world_image": torch.randn(4, 3, 64, 64),
        }
        patch_results: dict[str, Any] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            create_detailed_patch_visualization(
                output_batch,
                data_batch,
                0,
                tmpdir,
                patch_results,
            )

        mock_type_vis.assert_called_once()

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils._create_detailed_similarity_visualization")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils._create_detailed_type_visualization")
    def test_both_with_activation(self, mock_type_vis: MagicMock, mock_sim_vis: MagicMock) -> None:
        """Test with both logits and custom activation."""
        output_batch = {
            "patch_similarity_logits": torch.randn(4, 5, 8, 8),
            "patch_type_logits": torch.randn(4, 17, 8, 8),
        }
        data_batch = {
            "camera_image": torch.randn(4, 3, 64, 64),
            "world_image": torch.randn(4, 3, 64, 64),
        }
        patch_results: dict[str, Any] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            create_detailed_patch_visualization(
                output_batch,
                data_batch,
                0,
                tmpdir,
                patch_results,
                activation="softmax",
            )

        mock_sim_vis.assert_called_once()
        mock_type_vis.assert_called_once()


class TestCreateCombinedPatchVisualizationExtended(unittest.TestCase):
    """Extended tests for create_combined_patch_visualization."""

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.basic_vis_utils.tile_and_save_media")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_with_similarity_logits(self, mock_extract: MagicMock, mock_tile: MagicMock) -> None:
        """Test combined visualization with patch_similarity_logits."""
        num_frames = 4
        H, W = 64, 64

        mock_extract.return_value = (
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            np.random.rand(num_frames, H, W, 3).astype(np.float32),
            30.0,
        )

        output_batch = {
            "patch_similarity_logits": torch.randn(num_frames, 5, 8, 8),
        }
        data_batch = {
            "camera_image": torch.randn(num_frames, 3, H, W),
            "world_image": torch.randn(num_frames, 3, H, W),
        }
        patch_results: dict[str, Any] = {
            "per_frame_class_percentages": {1: [10.0] * num_frames},
            "per_frame_class_pixel_counts": {1: [100] * num_frames},
            "pred_class_indices": torch.ones(num_frames, 8, 8, dtype=torch.long),
            "pred_probabilities": torch.ones(num_frames, 8, 8),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_combined_patch_visualization(
                output_batch,
                data_batch,
                0,
                tmpdir,
                patch_results,
            )

        mock_tile.assert_called_once()


class TestBuildTypeColorsTensorExtended(unittest.TestCase):
    """Extended tests for _build_type_colors_tensor."""

    def test_with_v3_mapping(self) -> None:
        """Test type colors with V3 mapping."""
        colors = _build_type_colors_tensor(17, ObjectTypeClassMappingV3, device="cpu")
        self.assertEqual(colors.shape, (17, 3))

    def test_fallback_for_invalid_class_index(self) -> None:
        """Test that fallback color is used for invalid class indices."""
        # Using more classes than mapping supports
        colors = _build_type_colors_tensor(30, ObjectTypeClassMappingV3, device="cpu")
        self.assertEqual(colors.shape, (30, 3))
        # Colors should still be valid
        self.assertTrue(torch.all(colors >= 0.0))
        self.assertTrue(torch.all(colors <= 1.0))


class TestCreateDetailedPatchVisualizationDispatching(unittest.TestCase):
    """Tests for create_detailed_patch_visualization (dispatches to similarity and type)."""

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.image_video_utils.tiled_frames_generator")
    def test_similarity_only(self, mock_tiled: MagicMock) -> None:
        """Test with only similarity logits available."""

        num_frames, H, W = 2, 16, 16
        output_batch = {
            "patch_similarity_logits": torch.randn(num_frames, 5, H, W),
        }
        data_batch = {
            "camera_image": torch.rand(num_frames, H * 2, W * 2, 3),
            "world_image": torch.rand(num_frames, H * 2, W * 2, 3),
        }
        patch_results = {
            "pred_class_indices": torch.randint(0, 5, (num_frames, H, W)),
            "pred_probabilities": torch.rand(num_frames, H, W),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_detailed_patch_visualization(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                patch_results=patch_results,
                target_resolution=(96, 96),
                chunk_size=4,
            )

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.image_video_utils.tiled_frames_generator")
    def test_type_only(self, mock_tiled: MagicMock) -> None:
        """Test with only type logits available."""

        num_frames, H, W = 2, 8, 8
        output_batch = {
            "patch_type_logits": torch.randn(num_frames, 17, H, W),
        }
        data_batch = {
            "camera_image": torch.rand(num_frames, H * 2, W * 2, 3),
            "world_image": torch.rand(num_frames, H * 2, W * 2, 3),
        }
        patch_results = {
            "pred_class_indices": torch.randint(0, 17, (num_frames, H, W)),
            "pred_probabilities": torch.rand(num_frames, H, W),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_detailed_patch_visualization(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                patch_results=patch_results,
                target_resolution=(96, 96),
                chunk_size=4,
                object_type_class_mapping="v3",
            )

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.image_video_utils.tiled_frames_generator")
    def test_both_similarity_and_type(self, mock_tiled: MagicMock) -> None:
        """Test with both similarity and type logits."""

        num_frames, H, W = 2, 8, 8
        output_batch = {
            "patch_similarity_logits": torch.randn(num_frames, 5, H, W),
            "patch_type_logits": torch.randn(num_frames, 17, H, W),
        }
        data_batch = {
            "camera_image": torch.rand(num_frames, H * 2, W * 2, 3),
            "world_image": torch.rand(num_frames, H * 2, W * 2, 3),
        }
        patch_results = {
            "pred_class_indices": torch.randint(0, 5, (num_frames, H, W)),
            "pred_probabilities": torch.rand(num_frames, H, W),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_detailed_patch_visualization(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                patch_results=patch_results,
                target_resolution=(96, 96),
                chunk_size=4,
                object_type_class_mapping="v3",
            )

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.image_video_utils.tiled_frames_generator")
    def test_with_frame_indices(self, mock_tiled: MagicMock) -> None:
        """Test detailed visualization with frame_indices selection."""

        num_frames, H, W = 4, 8, 8
        output_batch = {
            "patch_similarity_logits": torch.randn(num_frames, 5, H, W),
        }
        data_batch = {
            "camera_image": torch.rand(num_frames, H * 2, W * 2, 3),
            "world_image": torch.rand(num_frames, H * 2, W * 2, 3),
        }
        patch_results = {
            "pred_class_indices": torch.randint(0, 5, (num_frames, H, W)),
            "pred_probabilities": torch.rand(num_frames, H, W),
        }
        # frame_indices: (N, 2) tensor of [world_idx, camera_idx] pairs
        frame_indices = torch.tensor([[0, 1], [2, 3]])

        with tempfile.TemporaryDirectory() as tmpdir:
            create_detailed_patch_visualization(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                patch_results=patch_results,
                target_resolution=(96, 96),
                frame_indices=frame_indices,
            )

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.image_video_utils.tiled_frames_generator")
    def test_type_with_excluded_classes(self, mock_tiled: MagicMock) -> None:
        """Test type visualization with excluded model classes."""

        num_frames, H, W = 2, 8, 8
        output_batch: dict[str, Any] = {
            "patch_type_logits": torch.randn(num_frames, 17, H, W),
            "excluded_model_classes_for_argmax_type": [0, 1],
        }
        data_batch = {
            "camera_image": torch.rand(num_frames, H * 2, W * 2, 3),
            "world_image": torch.rand(num_frames, H * 2, W * 2, 3),
        }
        patch_results = {
            "pred_class_indices": torch.randint(0, 17, (num_frames, H, W)),
            "pred_probabilities": torch.rand(num_frames, H, W),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_detailed_patch_visualization(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                patch_results=patch_results,
                target_resolution=(96, 96),
                object_type_class_mapping="v3",
            )

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.image_video_utils.tiled_frames_generator")
    def test_bfloat16_logits(self, mock_tiled: MagicMock) -> None:
        """Test with bfloat16 logits (should be converted to float32)."""

        num_frames, H, W = 2, 8, 8
        output_batch = {
            "patch_similarity_logits": torch.randn(num_frames, 5, H, W).to(torch.bfloat16),
        }
        data_batch = {
            "camera_image": torch.rand(num_frames, H * 2, W * 2, 3),
            "world_image": torch.rand(num_frames, H * 2, W * 2, 3),
        }
        patch_results = {
            "pred_class_indices": torch.randint(0, 5, (num_frames, H, W)),
            "pred_probabilities": torch.rand(num_frames, H, W),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_detailed_patch_visualization(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                patch_results=patch_results,
                target_resolution=(96, 96),
            )


class TestCombinedPatchVisualizationWithFrameIndices(unittest.TestCase):
    """Tests for create_combined_patch_visualization with frame_indices."""

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.image_video_utils.write_video_with_ffmpeg")
    def test_combined_visualization_with_frame_indices(self, mock_write: MagicMock) -> None:
        """Test combined visualization with frame_indices selection."""
        num_frames, num_classes, H, W = 4, 5, 8, 8
        logits = torch.randn(num_frames, num_classes, H, W)
        patch_results = {
            "pred_class_indices": torch.randint(0, num_classes, (num_frames, H, W)),
            "pred_probabilities": torch.rand(num_frames, H, W),
        }
        output_batch = {
            "patch_similarity_logits": logits,
        }
        data_batch = {
            "camera_image": torch.rand(num_frames, H * 2, W * 2, 3),
            "world_image": torch.rand(num_frames, H * 2, W * 2, 3),
        }
        # frame_indices: (N, 2) tensor of [world_idx, camera_idx] pairs
        frame_indices = torch.tensor([[0, 1], [2, 3]])

        with tempfile.TemporaryDirectory() as tmpdir:
            create_combined_patch_visualization(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                patch_results=patch_results,
                frame_indices=frame_indices,
            )

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.image_video_utils.write_video_with_ffmpeg")
    def test_combined_with_string_object_type_mapping(self, mock_write: MagicMock) -> None:
        """Test combined visualization with string object_type_class_mapping."""
        num_frames, num_classes, H, W = 2, 5, 8, 8
        logits = torch.randn(num_frames, num_classes, H, W)
        patch_results = {
            "pred_class_indices": torch.randint(0, num_classes, (num_frames, H, W)),
            "pred_probabilities": torch.rand(num_frames, H, W),
        }
        output_batch = {
            "patch_similarity_logits": logits,
        }
        data_batch = {
            "camera_image": torch.rand(num_frames, H * 2, W * 2, 3),
            "world_image": torch.rand(num_frames, H * 2, W * 2, 3),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_combined_patch_visualization(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                patch_results=patch_results,
                object_type_class_mapping="v3",
            )


class TestBuildTypeColorsTensorEdgeCases(unittest.TestCase):
    """Tests for _build_type_colors_tensor edge cases."""

    def test_no_mapping_exceeds_object_type_range(self) -> None:
        """Test _build_type_colors_tensor without mapping when num_classes exceeds ObjectType range."""
        # ObjectType has ~20 values, try 30 classes to exercise the ValueError fallback (line 228-229)
        result = _build_type_colors_tensor(30, object_type_class_mapping=None, device="cpu")
        self.assertEqual(result.shape, (30, 3))

    def test_with_mapping_keyerror_fallback(self) -> None:
        """Test _build_type_colors_tensor with mapping that raises KeyError."""
        result = _build_type_colors_tensor(5, object_type_class_mapping=ObjectTypeClassMappingV3, device="cpu")
        self.assertEqual(result.shape, (5, 3))

    def test_no_mapping_valid_classes(self) -> None:
        """Test _build_type_colors_tensor with valid ObjectType indices."""
        # Use a small number that's within ObjectType range
        result = _build_type_colors_tensor(5, object_type_class_mapping=None, device="cpu")
        self.assertEqual(result.shape, (5, 3))

    def test_with_v3_mapping(self) -> None:
        """Test _build_type_colors_tensor with V3 mapping."""
        num_classes = ObjectTypeClassMappingV3.get_num_classes()
        result = _build_type_colors_tensor(
            num_classes, object_type_class_mapping=ObjectTypeClassMappingV3, device="cpu"
        )
        self.assertEqual(result.shape, (num_classes, 3))


class TestCombinedVisualizationWithTypeLogits(unittest.TestCase):
    """Tests for create_combined_patch_visualization with object type logits (2x3 layout)."""

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.image_video_utils.write_video_with_ffmpeg")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.torch.cuda.empty_cache")
    def test_combined_with_type_logits_2x3_layout(self, mock_cache: MagicMock, mock_write: MagicMock) -> None:
        """Test combined visualization with both similarity and type logits creates 2x3 layout."""
        num_frames, sim_classes, type_classes, H, W = 2, 5, 5, 8, 8
        sim_logits = torch.randn(num_frames, sim_classes, H, W)
        type_logits = torch.randn(num_frames, type_classes, H, W)
        patch_results: dict[str, Any] = {
            "pred_class_indices": torch.randint(0, sim_classes, (num_frames, H, W)),
            "pred_probabilities": torch.rand(num_frames, H, W),
            "per_frame_class_percentages": {i: [50.0] * num_frames for i in range(sim_classes)},
            "per_frame_class_pixel_counts": {i: [100] * num_frames for i in range(sim_classes)},
        }
        output_batch = {
            "patch_similarity_logits": sim_logits,
            "patch_type_logits": type_logits,  # This triggers the 2x3 layout (lines 1051+)
        }
        data_batch = {
            "camera_image": torch.rand(num_frames, H * 2, W * 2, 3),
            "world_image": torch.rand(num_frames, H * 2, W * 2, 3),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_combined_patch_visualization(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                patch_results=patch_results,
                use_gpu_encoding=False,
            )

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.image_video_utils.write_video_with_ffmpeg")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.torch.cuda.empty_cache")
    def test_combined_with_type_logits_and_excluded_classes(self, mock_cache: MagicMock, mock_write: MagicMock) -> None:
        """Test combined vis with type logits and excluded type classes."""
        num_frames, sim_classes, type_classes, H, W = 2, 5, 5, 8, 8
        sim_logits = torch.randn(num_frames, sim_classes, H, W)
        type_logits = torch.randn(num_frames, type_classes, H, W)
        patch_results: dict[str, Any] = {
            "pred_class_indices": torch.randint(0, sim_classes, (num_frames, H, W)),
            "pred_probabilities": torch.rand(num_frames, H, W),
            "per_frame_class_percentages": {i: [50.0] * num_frames for i in range(sim_classes)},
            "per_frame_class_pixel_counts": {i: [100] * num_frames for i in range(sim_classes)},
        }
        output_batch: dict[str, Any] = {
            "patch_similarity_logits": sim_logits,
            "patch_type_logits": type_logits,
            "excluded_model_classes_for_argmax_type": [0, 2],  # Trigger lines 1062-1067
        }
        data_batch = {
            "camera_image": torch.rand(num_frames, H * 2, W * 2, 3),
            "world_image": torch.rand(num_frames, H * 2, W * 2, 3),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_combined_patch_visualization(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                patch_results=patch_results,
                use_gpu_encoding=False,
            )

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.image_video_utils.write_video_with_ffmpeg")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.torch.cuda.empty_cache")
    def test_combined_with_type_logits_and_object_mapping(self, mock_cache: MagicMock, mock_write: MagicMock) -> None:
        """Test combined vis with type logits and object_type_class_mapping."""
        num_frames, sim_classes, H, W = 2, 5, 8, 8
        type_classes = ObjectTypeClassMappingV3.get_num_classes()
        sim_logits = torch.randn(num_frames, sim_classes, H, W)
        type_logits = torch.randn(num_frames, type_classes, H, W)
        patch_results: dict[str, Any] = {
            "pred_class_indices": torch.randint(0, sim_classes, (num_frames, H, W)),
            "pred_probabilities": torch.rand(num_frames, H, W),
            "per_frame_class_percentages": {i: [50.0] * num_frames for i in range(sim_classes)},
            "per_frame_class_pixel_counts": {i: [100] * num_frames for i in range(sim_classes)},
        }
        output_batch = {
            "patch_similarity_logits": sim_logits,
            "patch_type_logits": type_logits,
        }
        data_batch = {
            "camera_image": torch.rand(num_frames, H * 2, W * 2, 3),
            "world_image": torch.rand(num_frames, H * 2, W * 2, 3),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_combined_patch_visualization(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                patch_results=patch_results,
                use_gpu_encoding=False,
                object_type_class_mapping=ObjectTypeClassMappingV3,
            )

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.image_video_utils.write_video_with_ffmpeg")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.torch.cuda.empty_cache")
    def test_combined_no_similarity_logits_uses_type_logits(self, mock_cache: MagicMock, mock_write: MagicMock) -> None:
        """Test combined vis when patch_similarity_logits is None falls back to patch_type_logits."""
        num_frames, num_classes, H, W = 2, 5, 8, 8
        type_logits = torch.randn(num_frames, num_classes, H, W)
        patch_results: dict[str, Any] = {
            "pred_class_indices": torch.randint(0, num_classes, (num_frames, H, W)),
            "pred_probabilities": torch.rand(num_frames, H, W),
            "per_frame_class_percentages": {i: [50.0] * num_frames for i in range(num_classes)},
            "per_frame_class_pixel_counts": {i: [100] * num_frames for i in range(num_classes)},
        }
        output_batch: dict[str, Any] = {
            "patch_similarity_logits": None,
            "patch_type_logits": type_logits,  # Fallback path (lines 936-937)
        }
        data_batch = {
            "camera_image": torch.rand(num_frames, H * 2, W * 2, 3),
            "world_image": torch.rand(num_frames, H * 2, W * 2, 3),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_combined_patch_visualization(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                patch_results=patch_results,
                use_gpu_encoding=False,
            )


class TestCombinedVisualizationMaskTrio3x3(unittest.TestCase):
    """``create_combined_patch_visualization`` builds a row-3 mask trio and forwards it to ``tile_and_save_media``
    whenever real similarity logits + object-type logits are both available, regardless of training/eval/inference
    mode. The trio is:

    - ``camera_masked_by_similarity``    (row 3 col 1)
    - ``similarity_mask``                 (row 3 col 2,
      ``clamp(P(MATCH) + P(EXTRA) + P(WRONG), 0, 1)`` — the canonical
      default from ``CWIPInference.DEFAULT_PATCH_SIMILARITY_MASK_CLASSES``)
    - ``object_type_masked_prediction``    (row 3 col 3)
    """

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.basic_vis_utils.tile_and_save_media")
    def test_row3_trio_passed_when_both_heads_available(self, mock_tile: MagicMock) -> None:
        num_frames, sim_classes, type_classes, H, W = 2, 5, 5, 8, 8
        sim_logits = torch.randn(num_frames, sim_classes, H, W)
        type_logits = torch.randn(num_frames, type_classes, H, W)
        patch_results: dict[str, Any] = {
            "pred_class_indices": torch.randint(0, sim_classes, (num_frames, H, W)),
            "pred_probabilities": torch.rand(num_frames, H, W),
            "per_frame_class_percentages": {i: [50.0] * num_frames for i in range(sim_classes)},
            "per_frame_class_pixel_counts": {i: [100] * num_frames for i in range(sim_classes)},
        }
        output_batch = {
            "patch_similarity_logits": sim_logits,
            "patch_type_logits": type_logits,
        }
        data_batch = {
            "camera_image": torch.rand(num_frames, H * 2, W * 2, 3),
            "world_image": torch.rand(num_frames, H * 2, W * 2, 3),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_combined_patch_visualization(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                patch_results=patch_results,
                use_gpu_encoding=False,
            )

        mock_tile.assert_called_once()
        kwargs = mock_tile.call_args.kwargs
        for key in (
            "camera_masked_by_similarity",
            "similarity_mask",
            "object_type_masked_prediction",
        ):
            self.assertIsNotNone(kwargs.get(key), f"Expected {key} to be forwarded.")
            arr = kwargs[key]
            self.assertEqual(arr.ndim, 4)
            self.assertEqual(arr.shape[0], num_frames)
            self.assertEqual(arr.shape[-1], 3)

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.basic_vis_utils.tile_and_save_media")
    def test_mask_is_clamped_default_subset_sum(self, mock_tile: MagicMock) -> None:
        """The mask channel of ``similarity_mask`` should equal ``clamp(sum_c P(c), 0, 1)`` over the canonical default
        subset ``MATCH + EXTRA + WRONG`` resized to frame resolution.
        """
        num_frames, sim_classes, type_classes, H, W = 1, 5, 5, 4, 4
        # Saturate every class in the default subset (MATCH=1, WRONG=2, EXTRA=3 in PatchSimilarityType). Each is then ~1
        # post-sigmoid, summing to ~3 ⇒ clamp to 1. The other classes (DONT_CARE=0, MISSING=4) stay at -10 ⇒ negligible
        # contribution. By saturating all three default-subset classes (rather than just MATCH+EXTRA) we observably
        # exercise WRONG's inclusion in the canonical default — a regression where WRONG silently gets dropped from the
        # default would still pass a test that only saturated MATCH+EXTRA.
        sim_logits = torch.full((num_frames, sim_classes, H, W), -10.0)
        sim_logits[:, 1] = 10.0  # MATCH
        sim_logits[:, 2] = 10.0  # WRONG
        sim_logits[:, 3] = 10.0  # EXTRA

        type_logits = torch.randn(num_frames, type_classes, H, W)
        patch_results: dict[str, Any] = {
            "pred_class_indices": torch.zeros(num_frames, H, W, dtype=torch.long),
            "pred_probabilities": torch.ones(num_frames, H, W),
            "per_frame_class_percentages": {i: [0.0] * num_frames for i in range(sim_classes)},
            "per_frame_class_pixel_counts": {i: [0] * num_frames for i in range(sim_classes)},
        }
        output_batch = {
            "patch_similarity_logits": sim_logits,
            "patch_type_logits": type_logits,
        }
        data_batch = {
            "camera_image": torch.ones(num_frames, H * 2, W * 2, 3) * 0.5,
            "world_image": torch.ones(num_frames, H * 2, W * 2, 3) * 0.5,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_combined_patch_visualization(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                patch_results=patch_results,
                use_gpu_encoding=False,
            )

        mask = mock_tile.call_args.kwargs["similarity_mask"]
        # Mask was clamped to 1.0; the central region (away from any text overlay baked in by _add_label_overlay
        # in the top-right) should be ~1.0. Sample a generous central window — robust to overlay placement.
        h, w = mask.shape[1:3]
        center = mask[:, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, :]
        self.assertGreaterEqual(float(center.min()), 0.95)
        self.assertLessEqual(float(center.max()), 1.0 + 1e-6)

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.image_video_utils.write_video_with_ffmpeg")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.image_video_utils.tiled_frames_generator")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.basic_vis_utils.tile_and_save_media")
    def test_no_row3_when_no_real_similarity_logits(
        self, mock_tile: MagicMock, mock_gen: MagicMock, _mock_write: MagicMock
    ) -> None:
        """When only ``patch_type_logits`` is available (camera-only / type-only), there is no similarity head, so the
        combined vis takes the type-only 2x2 path: no similarity column and no row-3 mask trio. That path renders via
        ``tiled_frames_generator`` and never touches ``tile_and_save_media``.
        """
        num_frames, type_classes, H, W = 2, 5, 8, 8
        mock_gen.return_value = iter([])
        type_logits = torch.randn(num_frames, type_classes, H, W)
        patch_results: dict[str, Any] = {
            "pred_class_indices": torch.zeros(num_frames, H, W, dtype=torch.long),
            "pred_probabilities": torch.ones(num_frames, H, W),
            "per_frame_class_percentages": {i: [0.0] * num_frames for i in range(type_classes)},
            "per_frame_class_pixel_counts": {i: [0] * num_frames for i in range(type_classes)},
        }
        output_batch = {
            # No patch_similarity_logits — the type-only 2x2 path is taken.
            "patch_type_logits": type_logits,
        }
        data_batch = {
            "camera_image": torch.rand(num_frames, H * 2, W * 2, 3),
            "world_image": torch.rand(num_frames, H * 2, W * 2, 3),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_combined_patch_visualization(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                patch_results=patch_results,
                use_gpu_encoding=False,
            )

        # Type-only path bypasses the world-paired tiler entirely and renders a 2x2 via tiled_frames_generator.
        mock_tile.assert_not_called()
        mock_gen.assert_called_once()
        rows, cols = mock_gen.call_args.args[1], mock_gen.call_args.args[2]
        self.assertEqual((rows, cols), (2, 2))

    def _run_with_presence(self, mock_tile: MagicMock, presence_mask: Any, **kwargs: Any) -> dict[str, Any]:
        num_frames, sim_classes, type_classes, H, W = 2, 5, 5, 4, 4
        # Saturate MATCH so the soft similarity mask would be ~1 everywhere: any 0 in row 3 must come from the presence
        # mask.
        sim_logits = torch.full((num_frames, sim_classes, H, W), -10.0)
        sim_logits[:, 1] = 10.0
        patch_results: dict[str, Any] = {
            "pred_class_indices": torch.ones(num_frames, H, W, dtype=torch.long),
            "pred_probabilities": torch.ones(num_frames, H, W),
            "per_frame_class_percentages": {i: [0.0] * num_frames for i in range(sim_classes)},
            "per_frame_class_pixel_counts": {i: [0] * num_frames for i in range(sim_classes)},
        }
        output_batch = {
            "patch_similarity_logits": sim_logits,
            "patch_type_logits": torch.randn(num_frames, type_classes, H, W),
        }
        data_batch = {
            "camera_image": torch.ones(num_frames, H * 8, W * 8, 3) * 0.5,
            "world_image": torch.ones(num_frames, H * 8, W * 8, 3) * 0.5,
        }
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch("checks.utils.models.cwip.visualization.patch_vis_utils._add_label_overlay") as mock_label,
        ):
            mock_label.side_effect = lambda images, label: images
            create_combined_patch_visualization(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                patch_results=patch_results,
                use_gpu_encoding=False,
                presence_mask=presence_mask,
                **kwargs,
            )
        labels = [c.args[1] for c in mock_label.call_args_list]
        return {"tile": mock_tile.call_args.kwargs, "labels": labels}

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.basic_vis_utils.tile_and_save_media")
    def test_presence_mask_replaces_similarity_mask_and_is_labeled(self, mock_tile: MagicMock) -> None:
        presence = np.zeros((2, 4, 4), dtype=np.uint8)
        presence[:, :, :2] = 1  # left half present
        out = self._run_with_presence(mock_tile, presence, presence_label="presence_source=union")
        mask = out["tile"]["similarity_mask"]
        w = mask.shape[2]
        self.assertEqual(float(mask[:, :, : w // 2].min()), 1.0)
        self.assertEqual(float(mask[:, :, w // 2 :].max()), 0.0)
        self.assertEqual(float(out["tile"]["camera_masked_by_similarity"][:, :, w // 2 :].max()), 0.0)
        self.assertIn("Presence Mask (presence_source=union)", out["labels"])
        self.assertIn("Cam x Presence", out["labels"])
        self.assertIn("Type x Presence", out["labels"])
        self.assertNotIn("Similarity Mask", out["labels"])

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.basic_vis_utils.tile_and_save_media")
    def test_without_presence_mask_keeps_similarity_labels(self, mock_tile: MagicMock) -> None:
        out = self._run_with_presence(mock_tile, None)
        self.assertIn("Similarity Mask", out["labels"])
        self.assertNotIn("Cam x Presence", out["labels"])

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.basic_vis_utils.tile_and_save_media")
    def test_presence_mask_follows_frame_indices(self, mock_tile: MagicMock) -> None:
        presence = np.zeros((2, 4, 4), dtype=np.uint8)
        presence[1] = 1  # only camera frame 1 present
        out = self._run_with_presence(mock_tile, presence, frame_indices=torch.tensor([[0, 1]]))
        mask = out["tile"]["similarity_mask"]
        self.assertEqual(mask.shape[0], 1)
        self.assertEqual(float(mask.min()), 1.0)

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.basic_vis_utils.tile_and_save_media")
    def test_presence_mask_frame_count_mismatch_raises(self, mock_tile: MagicMock) -> None:
        with self.assertRaises(ValueError):
            self._run_with_presence(mock_tile, np.ones((3, 4, 4), dtype=np.uint8))


class TestBuildTypeColorsTensorInvalidIndex(unittest.TestCase):
    """Test _build_type_colors_tensor with invalid model class indices (covers L228-229)."""

    def test_invalid_model_class_raises_value_error_fallback(self) -> None:
        """Model class index that exceeds ObjectType range should use FALLBACK_COLOR."""
        num_classes = 100  # Far more than ObjectType enum has
        result = _build_type_colors_tensor(
            num_type_classes=num_classes,
            object_type_class_mapping=ObjectTypeClassMappingV3,
            device="cpu",
        )
        # Should have num_classes colors, with fallbacks for out-of-range indices
        self.assertEqual(result.shape, (num_classes, 3))


class TestExtractFramesReturnNone(unittest.TestCase):
    """Test create_combined_patch_visualization when extract_frames returns None (covers L360-361)."""

    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.basic_vis_utils.extract_frames_from_data_batch")
    def test_returns_early_on_none_frames(self, mock_extract: MagicMock) -> None:
        """When extract_frames_from_data_batch returns None, visualization should return early."""
        mock_extract.return_value = None
        num_frames, num_classes, H, W = 2, 5, 8, 8
        output_batch = {
            "patch_similarity_logits": torch.randn(num_frames, num_classes, H, W),
        }
        data_batch = {
            "camera_image": torch.rand(num_frames, H, W, 3),
            "world_image": torch.rand(num_frames, H, W, 3),
        }
        patch_results = {
            "pred_class_indices": torch.randint(0, num_classes, (num_frames, H, W)),
            "pred_probabilities": torch.rand(num_frames, H, W),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            # Should not crash - returns early
            create_combined_patch_visualization(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                patch_results=patch_results,
                use_gpu_encoding=False,
            )


class TestCombinedVisualizationCameraOnly2x2(unittest.TestCase):
    """Camera-only (type-only) combined patch visualization: 2x2 layout, no similarity column, radials skipped."""

    @patch(
        "checks.utils.models.cwip.visualization.patch_vis_utils.basic_vis_utils.create_radial_class_pixel_count_plot"
    )
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.basic_vis_utils.create_radial_class_percentage_plot")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.image_video_utils.write_video_with_ffmpeg")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.image_video_utils.tiled_frames_generator")
    @patch("checks.utils.models.cwip.visualization.patch_vis_utils.torch.cuda.empty_cache")
    def test_type_only_2x2_and_radials_skipped(
        self,
        _mock_cache: MagicMock,
        mock_gen: MagicMock,
        mock_write: MagicMock,
        mock_pct_radial: MagicMock,
        mock_pixel_radial: MagicMock,
    ) -> None:
        """With no similarity logits, render a 2x2 (camera reused per row) and skip both radial plots."""
        num_frames, type_classes, H, W = 2, 5, 8, 8
        mock_gen.return_value = iter([])
        type_logits = torch.randn(num_frames, type_classes, H, W)
        patch_results: dict[str, Any] = {
            "pred_class_indices": torch.randint(0, type_classes, (num_frames, H, W)),
            "pred_probabilities": torch.rand(num_frames, H, W),
            "per_frame_class_percentages": {i: [50.0] * num_frames for i in range(type_classes)},
            "per_frame_class_pixel_counts": {i: [100] * num_frames for i in range(type_classes)},
        }
        output_batch: dict[str, Any] = {
            "patch_similarity_logits": None,
            "patch_type_logits": type_logits,
        }
        data_batch = {
            "camera_image": torch.rand(num_frames, H * 2, W * 2, 3),
            "world_image": torch.rand(num_frames, H * 2, W * 2, 3),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            create_combined_patch_visualization(
                output_batch=output_batch,
                data_batch=data_batch,
                iteration=0,
                output_dir=tmpdir,
                patch_results=patch_results,
                use_gpu_encoding=False,
            )

        # 2x2 with the camera reused as a per-row reference: [camera | overlay] / [camera | type mask].
        mock_gen.assert_called_once()
        tiles, rows, cols = mock_gen.call_args.args[0], mock_gen.call_args.args[1], mock_gen.call_args.args[2]
        self.assertEqual((rows, cols), (2, 2))
        self.assertEqual(len(tiles), 4)
        np.testing.assert_array_equal(tiles[0], tiles[2])
        mock_write.assert_called_once()
        _gen, _path, out_w, out_h, _fps = mock_write.call_args.args
        # Output is a 2x2 grid of the per-panel frames (derive expected size from the tiles themselves so the
        # assertion is robust to whatever resolution extract_frames yields).
        panel_h, panel_w = tiles[0].shape[1], tiles[0].shape[2]
        self.assertEqual((out_w, out_h), (panel_w * 2, panel_h * 2))
        # Similarity-branch radial plots must not be produced in camera-only mode.
        mock_pct_radial.assert_not_called()
        mock_pixel_radial.assert_not_called()


if __name__ == "__main__":
    unittest.main()
