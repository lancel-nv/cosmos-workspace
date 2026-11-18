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

"""Shared encode-then-decode machinery for the tests that measure what an encoder's quality tier actually delivers.

Extracted rather than duplicated because the whole value of these tests is that they drive the *production* argument
lists, and a second copy of the source generator or the round-trip harness would be free to drift into measuring
something subtly different. The two callers are split only by which hardware they need: ``encode_quality_test`` runs on
any GPU runner, and ``av1_nvenc_test`` needs an Ada-class card, so they cannot be one target.
"""

from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from checks.utils.video import DECODE_PIXEL_FORMAT_ARGS, _build_encode_args, get_video_decode_args
from checks.utils.video_codec import VideoQuality

# Small but detailed, and even-sized so the 4:2:0 chroma planes divide cleanly. Flat color would round-trip losslessly
# through almost any encoder and prove nothing, so every plane carries hard edges and gradients.
WIDTH = 320
HEIGHT = 240
FRAMES = 10
FPS = 10

# Bytes per yuv420p frame: a full-resolution luma plane plus two quarter-resolution chroma planes.
FRAME_BYTES = WIDTH * HEIGHT * 3 // 2

ENCODE_TIMEOUT_SECONDS = 300


def source_yuv420p() -> bytes:
    """Build a deterministic, high-detail ``yuv420p`` sequence as raw planar bytes.

    Constructed directly in the encoders' own pixel format so the comparison has no color conversion in it at all.
    """
    chroma_width = WIDTH // 2
    chroma_height = HEIGHT // 2
    luma_columns = np.arange(WIDTH, dtype=np.uint8)
    luma_rows = np.arange(HEIGHT, dtype=np.uint8)
    chroma_columns = np.arange(chroma_width, dtype=np.uint8)
    chroma_rows = np.arange(chroma_height, dtype=np.uint8)

    planes = []
    for index in range(FRAMES):
        luma = ((luma_columns[None, :] * 3 + index) ^ (luma_rows[:, None] * 5)).astype(np.uint8)
        # A moving hard-edged block, so there is high-frequency detail that motion prediction cannot trivially absorb.
        offset = index * 11
        luma[40:120, offset : offset + 60] = 255
        blue = (chroma_columns[None, :] * 7 + index * 3).astype(np.uint8)
        red = (chroma_rows[:, None] * 9 - index * 5).astype(np.uint8)
        planes.extend(
            [
                luma.tobytes(),
                np.broadcast_to(blue, (chroma_height, chroma_width)).tobytes(),
                np.broadcast_to(red, (chroma_height, chroma_width)).tobytes(),
            ]
        )
    return b"".join(planes)


def raw_input_args() -> list[str]:
    """ffmpeg input options describing the raw ``yuv420p`` source on stdin."""
    return [
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
    ]


class EncodeRoundTripTestBase(unittest.TestCase):
    """Shared round-trip machinery. Not discovered on its own -- the name does not start with ``Test``."""

    _source: bytes

    @classmethod
    def setUpClass(cls) -> None:
        cls._source = source_yuv420p()

    def _round_trip(self, encoder: str, quality: VideoQuality, codec: str, suffix: str = ".mp4") -> bytes:
        """Encode the raw source and decode it back, using exactly the arguments production uses in both directions."""
        decode_args = get_video_decode_args(codec)
        # A decode-side regression that stopped naming a decoder would otherwise let this pass via ffmpeg's default.
        self.assertEqual(decode_args[0], "-c:v", f"the ladder named no decoder for {codec}")

        with tempfile.TemporaryDirectory() as work_dir:
            clip = Path(work_dir) / f"round_trip{suffix}"
            encoded = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *raw_input_args()]
                + _build_encode_args(encoder, quality)
                + [str(clip)],
                input=self._source,
                capture_output=True,
                check=False,
                timeout=ENCODE_TIMEOUT_SECONDS,
            )
            self.assertEqual(
                encoded.returncode,
                0,
                f"{encoder} at {quality.value} failed to encode: {encoded.stderr.decode(errors='replace')}",
            )
            decoded = subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error"]
                + decode_args
                + ["-i", str(clip), *DECODE_PIXEL_FORMAT_ARGS, "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                capture_output=True,
                check=False,
                timeout=ENCODE_TIMEOUT_SECONDS,
            )
            self.assertEqual(
                decoded.returncode,
                0,
                f"{codec} failed to decode: {decoded.stderr.decode(errors='replace')}",
            )

        self.assertEqual(
            len(decoded.stdout),
            FRAMES * FRAME_BYTES,
            f"{encoder} round trip returned {len(decoded.stdout)} bytes, expected {FRAMES * FRAME_BYTES}",
        )
        return decoded.stdout

    def _assert_bit_exact(self, encoder: str, codec: str, suffix: str = ".mp4") -> None:
        decoded = self._round_trip(encoder, VideoQuality.HIGHEST, codec, suffix)
        if decoded == self._source:
            return
        before = np.frombuffer(self._source, dtype=np.uint8).astype(np.int16)
        after = np.frombuffer(decoded, dtype=np.uint8).astype(np.int16)
        differing = int(np.count_nonzero(before != after))
        largest = int(np.abs(before - after).max())
        self.fail(
            f"{encoder} at highest quality is documented as bit-exact but {differing} of {before.size} bytes differ, "
            f"largest delta {largest}/255. Either its argument row changed, or the "
            f"{DECODE_PIXEL_FORMAT_ARGS[1]} normalization is no longer applied on the way back."
        )

    def _assert_not_bit_exact(self, encoder: str, codec: str, suffix: str = ".mp4") -> None:
        """Assert the documented *limitation* of the two AV1 encoders, which is as load-bearing as the guarantees.

        A tier table row that silently became lossless would be a pleasant surprise, but it would also mean the
        surrounding documentation -- and the advice that bit-exact output requires pinning ``h264`` or ``vp9`` -- is
        now wrong, so it should fail rather than pass quietly.
        """
        decoded = self._round_trip(encoder, VideoQuality.HIGHEST, codec, suffix)
        self.assertNotEqual(
            decoded,
            self._source,
            f"{encoder} at highest quality round-tripped bit-exactly, contradicting the documented tier table. No "
            f"runtime lossless AV1 encoder is supposed to exist in this build; if {encoder} has gained one, update "
            "the table in checks/utils/video.py and the video_quality docs on SettingsBase rather than relaxing this "
            "assertion.",
        )
