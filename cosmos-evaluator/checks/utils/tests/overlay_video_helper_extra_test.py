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

"""Extra OverlayVideoHelper coverage (color legend, verbose, init errors) beyond video_helper_test."""

import logging
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from checks.utils.overlay_video_helper import OverlayVideoHelper


class TestOverlayVideoHelperExtra(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.out = self.tmp / "out"
        self.out.mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_verbose_sets_logger_debug(self) -> None:
        h = OverlayVideoHelper(clip_id="c", output_dir=str(self.out), verbose=True)
        self.assertEqual(h.logger.getEffectiveLevel(), logging.DEBUG)

    def test_default_does_not_force_logger_to_debug(self) -> None:
        with patch("checks.utils.overlay_video_helper.logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            OverlayVideoHelper(clip_id="c", output_dir=str(self.out), verbose=False)
            mock_logger.setLevel.assert_not_called()

    def test_add_legend_color_swatch_branch(self) -> None:
        h = OverlayVideoHelper(clip_id="c", output_dir=str(self.out))
        h.frame_width = 120
        h.frame_height = 80
        img = np.zeros((80, 120, 3), dtype=np.uint8)
        legend = [("ego", (255, 0, 0)), ("other", (0, 200, 100))]
        out = h.add_legend(img.copy(), frame_idx=3, legend_items=legend)
        self.assertEqual(out.shape, img.shape)
        self.assertFalse(np.array_equal(out, img))

    def test_add_legend_draws_swatch_colors_in_output(self) -> None:
        """Each legend RGB must appear as exact pixels (swatches are drawn after alpha blend)."""
        h = OverlayVideoHelper(clip_id="c", output_dir=str(self.out))
        h.frame_width = 300
        h.frame_height = 200
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        swatch_rgbs = [(255, 0, 0), (0, 200, 100)]
        legend = [("ego", swatch_rgbs[0]), ("other", swatch_rgbs[1])]
        out = h.add_legend(img.copy(), frame_idx=3, legend_items=legend)
        for rgb in swatch_rgbs:
            matches = np.all(out == np.array(rgb, dtype=np.uint8), axis=-1)
            self.assertTrue(matches.any(), f"Expected swatch color {rgb} to be drawn in legend output")

    @patch("checks.utils.overlay_video_helper.VideoWriter", side_effect=RuntimeError("no writer"))
    @patch("checks.utils.overlay_video_helper.get_video_frame_count", return_value=100)
    def test_initialize_video_writer_handles_exception(self, _mock_count: MagicMock, _mock_vw: MagicMock) -> None:
        h = OverlayVideoHelper(clip_id="c", output_dir=str(self.out))
        self.assertFalse(h.initialize_video_writer(video_path="/fake/in.mp4"))


if __name__ == "__main__":
    unittest.main()
