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

"""Tests for the codec and quality selection module."""

import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from checks.utils import video_codec
from checks.utils.video_codec import VideoCodec, VideoQuality


class TestParseVideoCodec(unittest.TestCase):
    """Tests for codec parsing and its fallback."""

    def test_each_member_parses_from_its_own_value(self) -> None:
        for member in VideoCodec:
            self.assertIs(video_codec.parse_video_codec(member.value), member)

    def test_parsing_ignores_case_and_surrounding_whitespace(self) -> None:
        """Env vars arrive as free text, so a stray space or capital must not silently change the codec."""
        self.assertIs(video_codec.parse_video_codec("  H264 "), VideoCodec.H264)

    def test_unset_yields_the_default_without_warning(self) -> None:
        """Not configuring the variable is the normal case, not a misconfiguration to complain about."""
        for value in (None, "", "   "):
            with self.assertNoLogs(video_codec.logger, level="WARNING"):
                self.assertIs(video_codec.parse_video_codec(value), video_codec.DEFAULT_VIDEO_CODEC)

    def test_unrecognized_value_warns_and_falls_back_to_av1(self) -> None:
        """Deliberately not the default: see the fallback constant on why a bad value must not land on H.264."""
        with self.assertLogs(video_codec.logger, level="WARNING") as logs:
            self.assertIs(video_codec.parse_video_codec("h265"), VideoCodec.AV1)
        self.assertIn("h265", logs.output[0])

    def test_the_fallback_is_not_the_default(self) -> None:
        """Pins the asymmetry itself, since collapsing the two would quietly route typos onto the H.264 path."""
        self.assertIsNot(video_codec.UNPARSEABLE_CODEC_FALLBACK, video_codec.DEFAULT_VIDEO_CODEC)

    def test_hevc_is_not_a_member(self) -> None:
        """HEVC has no encoder or decoder in the vendored build, so it must not be nameable as an output codec."""
        with self.assertLogs(video_codec.logger, level="WARNING"):
            self.assertIs(video_codec.parse_video_codec("hevc"), VideoCodec.AV1)


class TestParseVideoQuality(unittest.TestCase):
    """Tests for quality parsing and its fallback."""

    def test_each_member_parses_from_its_own_value(self) -> None:
        for member in VideoQuality:
            self.assertIs(video_codec.parse_video_quality(member.value), member)

    def test_unset_yields_standard(self) -> None:
        for value in (None, "", "   "):
            self.assertIs(video_codec.parse_video_quality(value), VideoQuality.STANDARD)

    def test_unrecognized_value_warns_and_falls_back_to_standard(self) -> None:
        with self.assertLogs(video_codec.logger, level="WARNING") as logs:
            self.assertIs(video_codec.parse_video_quality("lossless"), VideoQuality.STANDARD)
        self.assertIn("lossless", logs.output[0])

    def test_lossless_is_not_a_tier(self) -> None:
        """The tier is named "highest" precisely because two of the four encoders cannot deliver lossless output."""
        self.assertNotIn("lossless", {member.value for member in VideoQuality})


class TestResolveFromEnvironment(unittest.TestCase):
    """Tests for the two sanctioned ``os.environ`` reads."""

    def test_codec_comes_from_the_documented_variable(self) -> None:
        with patch.dict("os.environ", {video_codec.VIDEO_CODEC_ENV_VAR: "vp9"}):
            self.assertIs(video_codec.resolve_video_codec(), VideoCodec.VP9)

    def test_quality_comes_from_the_documented_variable(self) -> None:
        with patch.dict("os.environ", {video_codec.VIDEO_QUALITY_ENV_VAR: "highest"}):
            self.assertIs(video_codec.resolve_video_quality(), VideoQuality.HIGHEST)

    def test_absent_variables_resolve_to_the_defaults(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertIs(video_codec.resolve_video_codec(), video_codec.DEFAULT_VIDEO_CODEC)
            self.assertIs(video_codec.resolve_video_quality(), video_codec.DEFAULT_VIDEO_QUALITY)

    def test_resolution_is_not_memoized(self) -> None:
        """Memoizing would make the value depend on which test imported the module first."""
        with patch.dict("os.environ", {video_codec.VIDEO_CODEC_ENV_VAR: "av1"}):
            self.assertIs(video_codec.resolve_video_codec(), VideoCodec.AV1)
        with patch.dict("os.environ", {video_codec.VIDEO_CODEC_ENV_VAR: "vp9"}):
            self.assertIs(video_codec.resolve_video_codec(), VideoCodec.VP9)


class TestModuleStaysStdlibOnly(unittest.TestCase):
    """The constraint that lets settings and the processor layer share one definition of these values."""

    def test_importing_the_module_pulls_in_neither_pydantic_nor_numpy(self) -> None:
        """A subprocess, not this interpreter: the test runner has already imported both, so in-process checks pass
        vacuously. Importing the module alone must not drag pydantic into the processor layer or numpy into settings.
        """
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import checks.utils.video_codec; "
                "leaked = sorted(m for m in ('pydantic', 'pydantic_settings', 'numpy', 'cv2') if m in sys.modules); "
                "print(','.join(leaked))",
            ],
            capture_output=True,
            text=True,
            check=False,
            # Inherit the runner's environment (the child needs PYTHONPATH to find the module at all) but keep the
            # child from leaving __pycache__ directories in the runfiles tree, per the repo rule for the rare direct
            # interpreter invocation.
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
