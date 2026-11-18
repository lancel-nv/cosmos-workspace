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

"""Unit tests for CwipInferenceCache."""

import os
import shutil
import tempfile
import unittest

import torch

from checks.objects.single_view.cwip_inference_cache import CwipInferenceCache


class TestCwipInferenceCache(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _make_cache(self, cam: str = "cam.mp4", world: str = "world.mp4") -> CwipInferenceCache:
        return CwipInferenceCache(self.temp_dir, cam, world)

    def test_empty_cache_returns_none(self) -> None:
        cache = self._make_cache()
        self.assertIsNone(cache.load())
        self.assertFalse(cache.has())

    def test_save_and_load_roundtrip(self) -> None:
        cache = self._make_cache()
        summary = {"num_frames": 5, "has_patch_logits": True}
        sim = torch.zeros((5, 4, 4), dtype=torch.long)
        types = torch.ones((5, 4, 4), dtype=torch.long)
        logits = torch.randn(5, 10, 2, 2)

        cache.save(summary, sim, types, logits)
        result = cache.load()

        self.assertIsNotNone(result)
        assert result is not None
        loaded_summary, loaded_sim, loaded_types, loaded_logits = result
        self.assertEqual(loaded_summary["num_frames"], 5)
        assert loaded_sim is not None
        assert loaded_types is not None
        assert loaded_logits is not None
        self.assertTrue(torch.equal(loaded_sim, sim))
        self.assertTrue(torch.equal(loaded_types, types))
        self.assertTrue(torch.equal(loaded_logits, logits))

    def test_has_returns_true_after_save(self) -> None:
        cache = self._make_cache()
        self.assertFalse(cache.has())
        cache.save({"num_frames": 1}, None, None, None)
        self.assertTrue(cache.has())

    def test_load_with_none_tensors(self) -> None:
        cache = self._make_cache()
        cache.save({"num_frames": 2}, None, None, None)
        result = cache.load()
        self.assertIsNotNone(result)
        assert result is not None
        summary, sim, types, logits = result
        self.assertIsNone(sim)
        self.assertIsNone(types)
        self.assertIsNone(logits)
        self.assertEqual(summary["num_frames"], 2)

    def test_different_video_pairs_are_isolated(self) -> None:
        cache_a = self._make_cache(cam="cam_a.mp4", world="world_a.mp4")
        cache_b = self._make_cache(cam="cam_b.mp4", world="world_b.mp4")

        cache_a.save({"num_frames": 10}, None, None, None)
        self.assertTrue(cache_a.has())
        self.assertFalse(cache_b.has())

    def test_same_pair_reuses_cache(self) -> None:
        cache1 = self._make_cache(cam="cam.mp4", world="world.mp4")
        cache1.save({"num_frames": 7}, None, None, None)

        cache2 = self._make_cache(cam="cam.mp4", world="world.mp4")
        self.assertTrue(cache2.has())
        result = cache2.load()
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result[0]["num_frames"], 7)

    def test_swapped_pair_is_different_key(self) -> None:
        """Swapping camera and world paths produces a different cache entry."""
        cache_ab = self._make_cache(cam="cam.mp4", world="world.mp4")
        cache_ba = self._make_cache(cam="world.mp4", world="cam.mp4")

        cache_ab.save({"num_frames": 3}, None, None, None)
        self.assertTrue(cache_ab.has())
        self.assertFalse(cache_ba.has())

    def test_different_sampling_is_isolated(self) -> None:
        """Same video pair but different sampling (extra_key) must not share an entry.

        Guards the regression where a ``--trial N`` run (few frames) poisoned a later full run: the cached
        inference only covers the sampled frames, so different samplings must map to different cache entries.
        """
        cam, world = os.path.join(self.temp_dir, "cam.mp4"), os.path.join(self.temp_dir, "world.mp4")
        trial = CwipInferenceCache(self.temp_dir, cam, world, extra_key="trial=10|sample=first")
        full = CwipInferenceCache(self.temp_dir, cam, world, extra_key="trial=None|sample=uniform")

        trial.save({"num_frames": 3}, None, None, None)
        self.assertTrue(trial.has())
        self.assertFalse(full.has())  # full run does NOT see the 3-frame trial cache

    def test_same_sampling_reuses_cache(self) -> None:
        """Same video pair and same extra_key share the cache entry."""
        cam, world = os.path.join(self.temp_dir, "cam.mp4"), os.path.join(self.temp_dir, "world.mp4")
        a = CwipInferenceCache(self.temp_dir, cam, world, extra_key="trial=None|sample=uniform")
        a.save({"num_frames": 91}, None, None, None)
        b = CwipInferenceCache(self.temp_dir, cam, world, extra_key="trial=None|sample=uniform")
        self.assertTrue(b.has())
        result = b.load()
        assert result is not None
        self.assertEqual(result[0]["num_frames"], 91)

    def test_corrupt_file_returns_none(self) -> None:
        cache = self._make_cache()
        cache.save({"num_frames": 1}, None, None, None)
        # Corrupt the cache file
        cache._path.write_bytes(b"not a valid pt file")
        result = cache.load()
        self.assertIsNone(result)

    def test_write_failure_does_not_raise(self) -> None:
        """If the cache directory is read-only, save() should silently skip."""
        cache = self._make_cache()
        os.chmod(str(cache._path.parent), 0o555)
        try:
            # Should not raise
            cache.save({"num_frames": 1}, None, None, None)
        finally:
            os.chmod(str(cache._path.parent), 0o755)
        self.assertFalse(cache.has())


if __name__ == "__main__":
    unittest.main()
