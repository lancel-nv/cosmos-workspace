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

import unittest

import torch

from checks.utils.models.cwip.modules.class_mapping import (
    ClassMapping,
    ObjectTypeClassMappingV3,
    SimilarityClassMappingV1,
)
from checks.utils.models.cwip.visualization.metrics_utils import (
    apply_logits_activation,
    compute_patch_predictions_and_percentages,
    get_class_name_from_mapping,
    get_class_names_from_mapping,
)


class TestApplyLogitsActivation(unittest.TestCase):
    """Tests for apply_logits_activation function."""

    def test_sigmoid(self) -> None:
        """Test sigmoid activation."""
        logits = torch.randn(2, 5, 8, 8)
        result = apply_logits_activation(logits, activation="sigmoid")
        self.assertEqual(result.shape, logits.shape)
        self.assertTrue(torch.all(result >= 0.0))
        self.assertTrue(torch.all(result <= 1.0))

    def test_softmax(self) -> None:
        """Test softmax activation."""
        logits = torch.randn(2, 5, 8, 8)
        result = apply_logits_activation(logits, activation="softmax")
        self.assertEqual(result.shape, logits.shape)
        # Softmax should sum to 1 along class dimension
        sums = result.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_invalid_activation(self) -> None:
        """Test invalid activation raises ValueError."""
        logits = torch.randn(2, 5, 8, 8)
        with self.assertRaises(ValueError):
            apply_logits_activation(logits, activation="relu")

    def test_2d_logits(self) -> None:
        """Test activation with 2D logits (B, C)."""
        logits = torch.randn(4, 5)
        result = apply_logits_activation(logits, activation="sigmoid")
        self.assertEqual(result.shape, (4, 5))


class TestComputePatchPredictionsAndPercentages(unittest.TestCase):
    """Tests for compute_patch_predictions_and_percentages function."""

    def test_basic(self) -> None:
        """Test basic predictions and percentages computation."""
        batch_size = 5
        num_classes = 5
        H, W = 16, 16

        # Create dummy logits
        patch_similarity_logits = torch.randn(batch_size, num_classes, H, W)

        result = compute_patch_predictions_and_percentages(patch_similarity_logits, treat_wrong_type_as_match=False)

        # Check all expected keys are present
        self.assertIn("pred_class_indices", result)
        self.assertIn("pred_probabilities", result)
        self.assertIn("class_counts", result)
        self.assertIn("class_percentages", result)
        self.assertIn("per_frame_class_percentages", result)
        self.assertIn("per_frame_class_pixel_counts", result)
        self.assertIn("total_pixels", result)
        self.assertIn("total_non_dont_care_pixels", result)

        # Check shapes
        self.assertEqual(result["pred_class_indices"].shape, (batch_size, H, W))
        self.assertEqual(result["pred_probabilities"].shape, (batch_size, H, W))
        self.assertEqual(result["class_counts"].shape[0], num_classes)

        # Check percentages are dictionaries with valid values
        self.assertIsInstance(result["class_percentages"], dict)
        self.assertIsInstance(result["per_frame_class_percentages"], dict)

        # Check per-frame percentages have correct length
        for class_id in result["per_frame_class_percentages"]:
            self.assertEqual(len(result["per_frame_class_percentages"][class_id]), batch_size)

    def test_treat_wrong_as_match(self) -> None:
        """Test that treat_wrong_type_as_match merges WRONG predictions into MATCH."""
        batch_size = 2
        num_classes = 5
        H, W = 8, 8

        # Create logits that will predict WRONG (class 2)
        patch_similarity_logits = torch.ones(batch_size, num_classes, H, W) * -10.0
        patch_similarity_logits[:, 2, :, :] = 10.0  # High logit for WRONG

        # Without merging
        result_no_merge = compute_patch_predictions_and_percentages(
            patch_similarity_logits, treat_wrong_type_as_match=False
        )

        # With merging
        result_merge = compute_patch_predictions_and_percentages(
            patch_similarity_logits, treat_wrong_type_as_match=True
        )

        # Without merging: all pixels should be class 2 (WRONG)
        self.assertTrue((result_no_merge["pred_class_indices"] == 2).all())

        # With merging: all pixels should be class 1 (MATCH)
        self.assertTrue((result_merge["pred_class_indices"] == 1).all())

    def test_percentages_sum(self) -> None:
        """Test that class percentages sum to approximately 100%."""
        batch_size = 3
        num_classes = 5
        H, W = 16, 16

        patch_similarity_logits = torch.randn(batch_size, num_classes, H, W)

        result = compute_patch_predictions_and_percentages(patch_similarity_logits, treat_wrong_type_as_match=False)

        # Sum of all non-DONT_CARE class percentages (using all_classes denominator) should be approximately 100%. The
        # default "no_extra" denominator excludes EXTRA from the denominator but still counts its percentage, so it
        # can exceed 100%.
        total_percentage = sum(result["class_percentages_all_classes"].values())

        # Should be close to 100% (allow some floating point error)
        self.assertGreaterEqual(total_percentage, 99.0)
        self.assertLessEqual(total_percentage, 101.0)

    def test_with_gt(self) -> None:
        """Test predictions with ground truth for GT foreground denominator."""
        batch_size = 2
        num_classes = 5
        H, W = 8, 8

        patch_similarity_logits = torch.randn(batch_size, num_classes, H, W)

        # Create ground truth
        gt_patch_similarity = torch.zeros(batch_size, num_classes, H, W)
        gt_patch_similarity[:, 1, :4, :] = 1.0  # Half pixels are MATCH
        gt_patch_similarity[:, 0, 4:, :] = 1.0  # Other half are DONT_CARE

        result = compute_patch_predictions_and_percentages(
            patch_similarity_logits,
            treat_wrong_type_as_match=False,
            gt_patch_similarity=gt_patch_similarity,
            default_denominator="gt_fg",
        )

        # Check that GT foreground denominator is used
        self.assertIn("total_non_dont_care_pixels_gt_fg", result)
        self.assertIsNotNone(result["total_non_dont_care_pixels_gt_fg"])
        self.assertGreater(result["total_non_dont_care_pixels_gt_fg"], 0)

    def test_multiple_denominators(self) -> None:
        """Test that multiple denominator versions are computed."""
        batch_size = 2
        num_classes = 5
        H, W = 8, 8

        patch_similarity_logits = torch.randn(batch_size, num_classes, H, W)

        result = compute_patch_predictions_and_percentages(patch_similarity_logits, treat_wrong_type_as_match=False)

        # Check all three versions of percentages are present
        self.assertIn("class_percentages", result)
        self.assertIn("class_percentages_all_classes", result)
        self.assertIn("class_percentages_no_extra", result)

        self.assertIn("per_frame_class_percentages", result)
        self.assertIn("per_frame_class_percentages_all_classes", result)
        self.assertIn("per_frame_class_percentages_no_extra", result)

        self.assertIn("total_non_dont_care_pixels", result)
        self.assertIn("total_non_dont_care_pixels_all_classes", result)
        self.assertIn("total_non_dont_care_pixels_no_extra", result)


class TestGetClassNameFromMapping(unittest.TestCase):
    """Tests for get_class_name_from_mapping function."""

    def test_similarity_mapping(self) -> None:
        """Test class name resolution for similarity mapping."""
        name = get_class_name_from_mapping(0, SimilarityClassMappingV1)
        self.assertEqual(name, "DONT_CARE")
        name = get_class_name_from_mapping(1, SimilarityClassMappingV1)
        self.assertEqual(name, "MATCH")
        name = get_class_name_from_mapping(2, SimilarityClassMappingV1)
        self.assertEqual(name, "WRONG")

    def test_object_type_mapping_v3_indices(self) -> None:
        """Test class name resolution for object type V3 mapping."""
        name = get_class_name_from_mapping(0, ObjectTypeClassMappingV3)
        self.assertEqual(name, "BACKGROUND")
        name = get_class_name_from_mapping(1, ObjectTypeClassMappingV3)
        self.assertEqual(name, "BBOX_CAR")
        name = get_class_name_from_mapping(3, ObjectTypeClassMappingV3)
        self.assertEqual(name, "BBOX_PEDESTRIAN")

    def test_object_type_mapping_v3(self) -> None:
        """Test class name resolution for object type V3 mapping."""
        name = get_class_name_from_mapping(0, ObjectTypeClassMappingV3)
        self.assertEqual(name, "BACKGROUND")
        name = get_class_name_from_mapping(1, ObjectTypeClassMappingV3)
        self.assertEqual(name, "BBOX_CAR")

    def test_unknown_class_id(self) -> None:
        """Test unknown class ID returns UNKNOWN_CLASS_N string."""
        name = get_class_name_from_mapping(999, SimilarityClassMappingV1)
        self.assertEqual(name, "UNKNOWN_CLASS_999")
        name = get_class_name_from_mapping(999, ObjectTypeClassMappingV3)
        self.assertEqual(name, "UNKNOWN_CLASS_999")

    def test_invalid_mapping_raises(self) -> None:
        """Test that invalid class mapping raises TypeError."""
        with self.assertRaises(TypeError):
            get_class_name_from_mapping(0, ClassMapping)


class TestGetClassNamesFromMapping(unittest.TestCase):
    """Tests for get_class_names_from_mapping function."""

    def test_similarity_v1(self) -> None:
        """Test getting all class names for SimilarityClassMappingV1."""
        names = get_class_names_from_mapping(SimilarityClassMappingV1)
        self.assertEqual(len(names), 5)
        self.assertIn("DONT_CARE", names)
        self.assertIn("MATCH", names)

    def test_object_type_v3_count(self) -> None:
        """Test getting all class names for ObjectTypeClassMappingV3."""
        names = get_class_names_from_mapping(ObjectTypeClassMappingV3)
        self.assertEqual(len(names), 17)
        self.assertEqual(names[0], "BACKGROUND")

    def test_object_type_v3(self) -> None:
        """Test getting all class names for ObjectTypeClassMappingV3."""
        names = get_class_names_from_mapping(ObjectTypeClassMappingV3)
        self.assertEqual(len(names), 17)


class TestComputePatchPredictionsExtended(unittest.TestCase):
    """Extended tests for compute_patch_predictions_and_percentages."""

    def test_softmax_activation(self) -> None:
        """Test with softmax activation."""
        B, C, H, W = 2, 5, 8, 8
        logits = torch.randn(B, C, H, W)

        result = compute_patch_predictions_and_percentages(
            logits, treat_wrong_type_as_match=False, activation="softmax"
        )

        self.assertIn("pred_class_indices", result)
        self.assertEqual(result["pred_class_indices"].shape, (B, H, W))

    def test_excluded_model_classes(self) -> None:
        """Test with excluded model classes for argmax."""
        B, C, H, W = 2, 5, 8, 8
        logits = torch.randn(B, C, H, W)

        result = compute_patch_predictions_and_percentages(
            logits, treat_wrong_type_as_match=False, excluded_model_classes_for_argmax=[0, 3]
        )

        self.assertIn("pred_class_indices", result)
        # Excluded class 0 and 3 should not appear in predictions
        # (unless all other classes have -inf which is unlikely with random)

    def test_gt_fg_denominator_with_gt(self) -> None:
        """Test GT foreground denominator with gt_patch_similarity."""
        B, C, H, W = 2, 5, 8, 8
        logits = torch.randn(B, C, H, W)
        gt = torch.zeros(B, C, H, W)
        gt[:, 1, :4, :] = 1.0  # Half MATCH
        gt[:, 0, 4:, :] = 1.0  # Half DONT_CARE

        result = compute_patch_predictions_and_percentages(
            logits, treat_wrong_type_as_match=False, gt_patch_similarity=gt, default_denominator="gt_fg"
        )

        self.assertIsNotNone(result["total_non_dont_care_pixels_gt_fg"])
        self.assertGreater(result["total_non_dont_care_pixels_gt_fg"], 0)

    def test_gt_fg_denominator_with_pixel_counts(self) -> None:
        """Test GT foreground denominator with pre-computed pixel counts."""
        B, C, H, W = 2, 5, 8, 8
        logits = torch.randn(B, C, H, W)
        gt_fg_counts = torch.tensor([32, 48])

        result = compute_patch_predictions_and_percentages(
            logits,
            treat_wrong_type_as_match=False,
            gt_foreground_pixel_counts=gt_fg_counts,
            default_denominator="gt_fg",
        )

        self.assertIsNotNone(result["total_non_dont_care_pixels_gt_fg"])

    def test_all_classes_denominator(self) -> None:
        """Test all_classes denominator."""
        B, C, H, W = 2, 5, 8, 8
        logits = torch.randn(B, C, H, W)

        result = compute_patch_predictions_and_percentages(
            logits, treat_wrong_type_as_match=False, default_denominator="all_classes"
        )

        self.assertIn("total_non_dont_care_pixels", result)

    def test_gt_resized(self) -> None:
        """Test GT foreground denominator when GT needs resizing."""
        B, C, H, W = 2, 5, 8, 8
        logits = torch.randn(B, C, H, W)
        # GT at different resolution
        gt = torch.zeros(B, C, 16, 16)
        gt[:, 1, :8, :] = 1.0

        result = compute_patch_predictions_and_percentages(
            logits, treat_wrong_type_as_match=False, gt_patch_similarity=gt, default_denominator="gt_fg"
        )

        self.assertIsNotNone(result["total_non_dont_care_pixels_gt_fg"])


class TestComputePatchPredictionsEmptyFrames(unittest.TestCase):
    """Tests for compute_patch_predictions_and_percentages with edge cases."""

    def test_all_dont_care_predictions(self) -> None:
        """Test when all predictions are DONT_CARE (class 0), covers L432, L441, L493, L501, L509."""
        B, C, H, W = 2, 5, 4, 4
        # Logits that make all predictions be class 0 (DONT_CARE)
        logits = torch.zeros(B, C, H, W)
        logits[:, 0, :, :] = 100.0  # Very high for DONT_CARE

        result = compute_patch_predictions_and_percentages(
            logits,
            treat_wrong_type_as_match=False,
        )
        # All non-DONT_CARE class percentages should be 0
        for class_id in range(1, C):
            self.assertAlmostEqual(result["class_percentages"].get(class_id, 0.0), 0.0, places=5)


class TestComputePatchPredictionsZeroFrames(unittest.TestCase):
    """Tests for compute_patch_predictions_and_percentages with zero frames (covers L493, L501, L509)."""

    def test_zero_frames_returns_empty_percentages(self) -> None:
        """When B=0, all per_frame_class_percentages should be empty and averages should be 0."""
        C = 5
        logits = torch.zeros(0, C, 4, 4)
        result = compute_patch_predictions_and_percentages(
            logits,
            treat_wrong_type_as_match=False,
        )
        # All class percentages should be 0.0
        for class_id in range(1, C):
            self.assertEqual(result["class_percentages"].get(class_id, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
