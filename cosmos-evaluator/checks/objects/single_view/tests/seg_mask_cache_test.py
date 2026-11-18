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

"""Unit tests for SegMaskCache."""

import shutil
import tempfile
import unittest

import torch

from checks.objects.single_view.seg_mask_cache import SegMaskCache


class TestSegMaskCache(unittest.TestCase):
    def setUp(self) -> None:
        self.cache_dir = tempfile.mkdtemp()
        self.video_path = "/data/clip/camera_front.mp4"

    def tearDown(self) -> None:
        shutil.rmtree(self.cache_dir)

    def _make_cache(self, video_path: str | None = None) -> SegMaskCache:
        return SegMaskCache(self.cache_dir, video_path or self.video_path)

    def test_has_returns_false_when_empty(self) -> None:
        cache = self._make_cache()
        self.assertFalse(cache.has(0))
        self.assertFalse(cache.has(42))

    def test_load_returns_none_when_empty(self) -> None:
        cache = self._make_cache()
        self.assertIsNone(cache.load(0))

    def test_save_and_load_roundtrip(self) -> None:
        cache = self._make_cache()
        masks = torch.zeros(3, 8, 8, dtype=torch.uint8)
        masks[1, 2, 3] = 255
        cache.save(0, masks)
        loaded = cache.load(0)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertTrue(torch.equal(masks, loaded))

    def test_has_returns_true_after_save(self) -> None:
        cache = self._make_cache()
        cache.save(5, torch.ones(2, 4, 4))
        self.assertTrue(cache.has(5))
        self.assertFalse(cache.has(6))

    def test_different_frame_indices_are_independent(self) -> None:
        cache = self._make_cache()
        a = torch.full((1, 2, 2), 1.0)
        b = torch.full((1, 2, 2), 2.0)
        cache.save(0, a)
        cache.save(1, b)
        loaded_a = cache.load(0)
        loaded_b = cache.load(1)
        assert loaded_a is not None
        assert loaded_b is not None
        self.assertTrue(torch.equal(loaded_a, a))
        self.assertTrue(torch.equal(loaded_b, b))

    def test_different_video_paths_use_different_subdirs(self) -> None:
        cache_a = SegMaskCache(self.cache_dir, "/data/video_a.mp4")
        cache_b = SegMaskCache(self.cache_dir, "/data/video_b.mp4")
        # Write frame 0 only to cache_a
        cache_a.save(0, torch.zeros(1, 2, 2))
        self.assertTrue(cache_a.has(0))
        self.assertFalse(cache_b.has(0))

    def test_same_video_path_uses_same_subdir(self) -> None:
        cache1 = self._make_cache()
        cache2 = self._make_cache()
        masks = torch.ones(1, 2, 2)
        cache1.save(0, masks)
        loaded = cache2.load(0)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertTrue(torch.equal(masks, loaded))

    def test_save_failure_does_not_raise(self) -> None:
        # Point cache at a non-writable path to trigger a silent failure
        cache = SegMaskCache(self.cache_dir, self.video_path)
        # Replace internal dir with a file so writes fail
        cache._dir.rmdir()
        cache._dir.touch()
        # Should not raise
        cache.save(0, torch.zeros(1, 2, 2))

    def test_load_corrupt_file_returns_none(self) -> None:
        cache = self._make_cache()
        path = cache._frame_path(0)
        path.write_bytes(b"not a valid tensor")
        result = cache.load(0)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
