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

"""Tests for the VideoWriter helper in checks.utils.video."""

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from checks.utils import video


def _mock_proc() -> MagicMock:
    """Build a mock ffmpeg child for the VideoWriter (stdin pipe + clean communicate)."""
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.communicate.return_value = (b"", b"")
    proc.returncode = 0
    return proc


class TestVideoWriter(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch("checks.utils.video._popen_ffmpeg")
    def test_write_frame_resizes_and_releases(self, mock_popen: MagicMock) -> None:
        proc = _mock_proc()
        mock_popen.return_value = proc

        out = self.tmp / "sub" / "out.mp4"
        vw = video.VideoWriter(out, fps=5.0, frame_width=10, frame_height=8, ffmpeg_args=["-c:v", "copy"])
        self.assertTrue(vw.open())
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        self.assertTrue(vw.write_frame(frame, 0))
        vw.release()

        # The frame is resized to (frame_width, frame_height); raw BGR bytes piped to ffmpeg stdin.
        proc.stdin.write.assert_called_once()
        written = proc.stdin.write.call_args.args[0]
        self.assertEqual(len(written), 10 * 8 * 3)
        proc.communicate.assert_called()

    @patch("checks.utils.video._popen_ffmpeg")
    def test_write_frame_none_returns_false(self, mock_popen: MagicMock) -> None:
        mock_popen.return_value = _mock_proc()
        vw = video.VideoWriter(self.tmp / "a.mp4", fps=1.0, frame_width=2, frame_height=2)
        vw.open()
        self.assertFalse(vw.write_frame(None, 0))  # type: ignore[arg-type]

    @patch("checks.utils.video._popen_ffmpeg")
    def test_preprocessor_exception_still_writes(self, mock_popen: MagicMock) -> None:
        proc = _mock_proc()
        mock_popen.return_value = proc

        def bad_pre(_rgb: np.ndarray, _idx: int) -> np.ndarray:
            raise RuntimeError("legend")

        vw = video.VideoWriter(
            self.tmp / "b.mp4",
            fps=1.0,
            frame_width=4,
            frame_height=4,
            preprocessor=bad_pre,
            ffmpeg_args=["-x"],
        )
        vw.open()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        with patch("checks.utils.video.logger"):
            self.assertTrue(vw.write_frame(frame, 0))
        proc.stdin.write.assert_called_once()

    @patch("checks.utils.video._popen_ffmpeg", side_effect=OSError("ffmpeg not found"))
    def test_open_fails_when_ffmpeg_unavailable(self, _mock_popen: MagicMock) -> None:
        vw = video.VideoWriter(self.tmp / "c.mp4", fps=1.0, frame_width=2, frame_height=2)
        self.assertFalse(vw.open())


if __name__ == "__main__":
    unittest.main()
