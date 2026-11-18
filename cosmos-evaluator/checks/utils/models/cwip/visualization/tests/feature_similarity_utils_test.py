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

from checks.utils.models.cwip.visualization.feature_similarity_utils import compute_cosine_similarity


class TestComputeCosineSimilarity(unittest.TestCase):
    """Tests for compute_cosine_similarity function."""

    def test_basic_global_features(self) -> None:
        """Test basic cosine similarity with global features (B, D)."""
        feat1 = torch.randn(4, 256)
        feat2 = torch.randn(4, 256)
        sim = compute_cosine_similarity(feat1, feat2)
        self.assertEqual(sim.shape, (4, 1))

    def test_keepdim_false(self) -> None:
        """Test cosine similarity with keepdim=False."""
        feat1 = torch.randn(4, 256)
        feat2 = torch.randn(4, 256)
        sim = compute_cosine_similarity(feat1, feat2, keepdim=False)
        self.assertEqual(sim.shape, (4,))

    def test_spatial_features(self) -> None:
        """Test cosine similarity with spatial features (B, D, H, W)."""
        feat1 = torch.randn(2, 64, 8, 8)
        feat2 = torch.randn(2, 64, 8, 8)
        sim = compute_cosine_similarity(feat1, feat2)
        self.assertEqual(sim.shape, (2, 1, 8, 8))

    def test_identical_features_return_one(self) -> None:
        """Test that identical features have cosine similarity of 1."""
        feat = torch.randn(2, 64)
        sim = compute_cosine_similarity(feat, feat, keepdim=False)
        self.assertTrue(torch.allclose(sim, torch.ones(2), atol=1e-6))

    def test_shape_mismatch_raises_error(self) -> None:
        """Test that shape mismatch raises ValueError (covers L77)."""
        feat1 = torch.randn(4, 256)
        feat2 = torch.randn(4, 128)  # Different feature dim
        with self.assertRaises(ValueError) as ctx:
            compute_cosine_similarity(feat1, feat2)
        self.assertIn("Feature shapes must match", str(ctx.exception))

    def test_negative_dim_conversion(self) -> None:
        """Test that negative dim is correctly converted (covers L80-81)."""
        feat1 = torch.randn(2, 64, 8, 8)
        feat2 = torch.randn(2, 64, 8, 8)
        # dim=-3 should be converted to dim=1 for 4D tensor
        sim = compute_cosine_similarity(feat1, feat2, dim=-3)
        self.assertEqual(sim.shape, (2, 1, 8, 8))

    def test_dim_out_of_range_raises_error(self) -> None:
        """Test that dim out of range raises ValueError (covers L84)."""
        feat1 = torch.randn(4, 256)
        feat2 = torch.randn(4, 256)
        with self.assertRaises(ValueError) as ctx:
            compute_cosine_similarity(feat1, feat2, dim=5)
        self.assertIn("dim=5 is out of range", str(ctx.exception))

    def test_single_feature_vector(self) -> None:
        """Test cosine similarity with single feature vectors (D,)."""
        feat1 = torch.randn(256)
        feat2 = torch.randn(256)
        sim = compute_cosine_similarity(feat1, feat2, dim=0, keepdim=False)
        self.assertEqual(sim.shape, ())

    def test_sequential_features(self) -> None:
        """Test cosine similarity with sequential features (B, T, D)."""
        feat1 = torch.randn(4, 10, 512)
        feat2 = torch.randn(4, 10, 512)
        sim = compute_cosine_similarity(feat1, feat2, dim=2)
        self.assertEqual(sim.shape, (4, 10, 1))


if __name__ == "__main__":
    unittest.main()
