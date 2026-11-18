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

from checks.utils.models.cwip.tools.batch_utils import merge_batch_outputs


class TestMergeBatchOutputs(unittest.TestCase):
    """Tests for merge_batch_outputs function."""

    def test_empty_list(self) -> None:
        result = merge_batch_outputs([])
        self.assertEqual(result, {})

    def test_single_batch(self) -> None:
        batch = {"features": torch.randn(2, 4)}
        result = merge_batch_outputs([batch])
        self.assertTrue(torch.equal(result["features"], batch["features"]))

    def test_merge_tensors(self) -> None:
        b1 = {"features": torch.randn(2, 4)}
        b2 = {"features": torch.randn(3, 4)}
        result = merge_batch_outputs([b1, b2])
        self.assertEqual(result["features"].shape, (5, 4))
        self.assertTrue(torch.equal(result["features"][:2], b1["features"]))
        self.assertTrue(torch.equal(result["features"][2:], b2["features"]))

    def test_merge_three_batches(self) -> None:
        batches = [{"x": torch.randn(1, 8)} for _ in range(3)]
        result = merge_batch_outputs(batches)
        self.assertEqual(result["x"].shape, (3, 8))

    def test_merge_lists_of_tensors(self) -> None:
        b1 = {"features_list": [torch.randn(2, 4), torch.randn(2, 8)]}
        b2 = {"features_list": [torch.randn(3, 4), torch.randn(3, 8)]}
        result = merge_batch_outputs([b1, b2])
        self.assertEqual(len(result["features_list"]), 2)
        self.assertEqual(result["features_list"][0].shape, (5, 4))
        self.assertEqual(result["features_list"][1].shape, (5, 8))

    def test_none_values(self) -> None:
        b1 = {"logits": None}
        b2 = {"logits": None}
        result = merge_batch_outputs([b1, b2])
        self.assertIsNone(result["logits"])

    def test_other_types_use_first(self) -> None:
        b1 = {"metadata": "video_001"}
        b2 = {"metadata": "video_002"}
        result = merge_batch_outputs([b1, b2])
        self.assertEqual(result["metadata"], "video_001")

    def test_mixed_keys(self) -> None:
        b1 = {
            "features": torch.randn(2, 4),
            "features_list": [torch.randn(2, 8)],
            "mask": None,
            "name": "batch_0",
        }
        b2 = {
            "features": torch.randn(3, 4),
            "features_list": [torch.randn(3, 8)],
            "mask": None,
            "name": "batch_1",
        }
        result = merge_batch_outputs([b1, b2])
        self.assertEqual(result["features"].shape, (5, 4))
        self.assertEqual(result["features_list"][0].shape, (5, 8))
        self.assertIsNone(result["mask"])
        self.assertEqual(result["name"], "batch_0")

    def test_preserves_all_keys(self) -> None:
        b1 = {"a": torch.randn(1, 2), "b": torch.randn(1, 3), "c": None}
        result = merge_batch_outputs([b1])
        self.assertEqual(set(result.keys()), {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
