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
import unittest

import numpy as np

from checks.utils.models.cwip.visualization.image_video_utils import write_video_with_ffmpeg
from checks.utils.video import get_video_codec


class TestWriteVideoWithFfmpegNvencReal(unittest.TestCase):
    """Real (un-mocked) encoding test verifying NVENC GPU encoding produces a valid video."""

    def test_nvenc_produces_valid_video(self) -> None:
        """Write frames through the real ffmpeg subprocess with NVENC hardware encoder."""
        num_frames = 3
        H, W = 256, 256
        frames = [np.random.randint(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(num_frames)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "nvenc.mp4")
            write_video_with_ffmpeg(frames, output_path, W, H, fps=30)

            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)
            # NVENC must actually be used on a GPU host: at 256x256 (<=4096px) the encoder is h264_nvenc, which emits
            # an H.264 stream. If NVENC probing silently fell back to the SVT-AV1 CPU encoder, the codec would be
            # "av1" and this assertion would fail -- catching a regression that a non-empty-file check would miss.
            codec = get_video_codec(output_path)
            self.assertEqual(codec, "h264", f"expected NVENC H.264 output, got codec={codec!r}")


if __name__ == "__main__":
    unittest.main()
