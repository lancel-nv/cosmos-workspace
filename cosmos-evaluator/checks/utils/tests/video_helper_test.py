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

"""Unit tests for image module."""

from collections.abc import Generator
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import ClassVar
import unittest
from unittest.mock import MagicMock, call, patch

import numpy as np

from checks.utils import video
from checks.utils.overlay_video_helper import OverlayVideoHelper
from checks.utils.video_codec import VIDEO_CODEC_ENV_VAR, VIDEO_QUALITY_ENV_VAR, VideoQuality
from utils.gpu import GpuUnavailableError


class TestIsNvencAvailable(unittest.TestCase):
    """Tests for the per-codec NVENC probe."""

    def setUp(self) -> None:
        video._NVENC_AVAILABLE_CACHE.clear()

    def tearDown(self) -> None:
        video._NVENC_AVAILABLE_CACHE.clear()

    @patch("checks.utils.video.subprocess.run")
    def test_returns_true_when_nvenc_works(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        self.assertTrue(video.is_nvenc_available())

    @patch("checks.utils.video.subprocess.run")
    def test_returns_false_when_nvenc_fails(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stderr="No capable devices found")
        self.assertFalse(video.is_nvenc_available())

    @patch("checks.utils.video.subprocess.run")
    def test_caches_result(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0)
        video.is_nvenc_available()
        video.is_nvenc_available()
        mock_run.assert_called_once()

    @patch("checks.utils.video.subprocess.run")
    def test_cache_is_per_codec(self, mock_run: MagicMock) -> None:
        """An Ampere host runs h264_nvenc but not av1_nvenc, so one cache entry must not answer for the other."""
        mock_run.side_effect = [MagicMock(returncode=0), MagicMock(returncode=1, stderr="No capable devices found")]
        self.assertTrue(video._is_nvenc_codec_available("h264_nvenc"))
        self.assertFalse(video._is_nvenc_codec_available("av1_nvenc"))
        self.assertEqual(mock_run.call_count, 2)

    @patch("checks.utils.video.subprocess.run")
    def test_cache_is_per_quality_tier(self, mock_run: MagicMock) -> None:
        """The L40 lesson: one encoder can open at one tier and be refused at another, so the tier is part of the key.

        Without it, whichever tier probed first would answer for both, which is exactly how an unsupported tune
        reached the encode instead of being skipped during selection.
        """
        mock_run.side_effect = [MagicMock(returncode=0), MagicMock(returncode=1, stderr="No capable devices found")]
        self.assertTrue(video._is_nvenc_codec_available("h264_nvenc", video.VideoQuality.STANDARD))
        self.assertFalse(video._is_nvenc_codec_available("h264_nvenc", video.VideoQuality.HIGHEST))
        self.assertEqual(mock_run.call_count, 2)

    @patch("checks.utils.video.subprocess.run", return_value=MagicMock(returncode=0))
    def test_probe_uses_the_tier_arguments(self, mock_run: MagicMock) -> None:
        """The probe has to ask ffmpeg the question the encode will ask, or it answers about a different config."""
        video._is_nvenc_codec_available("h264_nvenc", video.VideoQuality.HIGHEST)
        probed = mock_run.call_args[0][0]
        for token in video._QUALITY_ARGS[("h264_nvenc", video.VideoQuality.HIGHEST)]:
            self.assertIn(token, probed)
        # A muxer option in a probe that writes to the null muxer would prove nothing about the encoder.
        self.assertNotIn("-movflags", probed)

    @patch("checks.utils.video.subprocess.run")
    def test_timeout_returns_false(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=5)
        self.assertFalse(video.is_nvenc_available())

    @patch("checks.utils.video.subprocess.run")
    def test_oserror_returns_false(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = OSError("ffmpeg not found")
        self.assertFalse(video.is_nvenc_available())


class TestGetVideoEncodeArgs(unittest.TestCase):
    """Tests for the three-tier encode ladder.

    Several of these call ``get_video_encode_args`` with no codec, or with only a quality, and the omitted argument is
    resolved from the environment. A developer shell or build machine that exports either variable would therefore
    change which encoder the assertions see. Both are unset for the duration of every test here, so "the default" means
    the code's default rather than the host's.
    """

    def setUp(self) -> None:
        video._NVENC_AVAILABLE_CACHE.clear()
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop(VIDEO_CODEC_ENV_VAR, None)
        os.environ.pop(VIDEO_QUALITY_ENV_VAR, None)

    def tearDown(self) -> None:
        video._NVENC_AVAILABLE_CACHE.clear()

    @patch("checks.utils.video._is_nvenc_codec_available", return_value=True)
    def test_prefers_h264_nvenc(self, _: MagicMock) -> None:
        args = video.get_video_encode_args()
        self.assertIn("h264_nvenc", args)
        self.assertNotIn("av1_nvenc", args)
        self.assertNotIn("libsvtav1", args)

    @patch("checks.utils.video._is_nvenc_codec_available", side_effect=lambda enc, _quality: enc == "av1_nvenc")
    def test_falls_back_to_av1_nvenc(self, _: MagicMock) -> None:
        """Tier 2: hardware AV1 when the host has NVENC but cannot encode AVC."""
        args = video.get_video_encode_args()
        self.assertIn("av1_nvenc", args)
        self.assertNotIn("h264_nvenc", args)
        self.assertNotIn("libsvtav1", args)

    @patch("checks.utils.video._is_nvenc_codec_available", return_value=False)
    def test_falls_back_to_svtav1_when_no_hardware(self, _: MagicMock) -> None:
        args = video.get_video_encode_args()
        self.assertIn("libsvtav1", args)
        self.assertNotIn("h264_nvenc", args)
        self.assertNotIn("av1_nvenc", args)
        self.assertNotIn("libopenh264", args)

    @patch("checks.utils.video._is_nvenc_codec_available", return_value=True)
    def test_never_emits_audio_args(self, _: MagicMock) -> None:
        """The vendored ffmpeg ships no audio encoder, so these args must stay video-only."""
        args = video.get_video_encode_args()
        self.assertNotIn("-c:a", args)
        self.assertNotIn("aac", args)

    @patch("checks.utils.video._is_nvenc_codec_available", return_value=True)
    def test_returns_a_copy(self, _: MagicMock) -> None:
        """Callers mutate the returned list, so the module constant must not be handed out by reference."""
        args = video.get_video_encode_args()
        args.append("-sentinel")
        self.assertNotIn("-sentinel", video.NVENC_H264_FFMPEG_ARGS)

    @patch("checks.utils.video._is_nvenc_codec_available", return_value=True)
    def test_standard_tier_reproduces_the_legacy_constants(self, _: MagicMock) -> None:
        """The whole no-op claim for existing deployments rests on this: same tokens, same order, per encoder.

        These three lists were hardcoded before the quality tier existed. Composing them from an encoder base plus a
        quality row has to reproduce them exactly, or every deployment's output silently changes on upgrade.
        """
        self.assertEqual(
            video.NVENC_H264_FFMPEG_ARGS,
            ["-c:v", "h264_nvenc", "-pix_fmt", "yuv420p"]
            + ["-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", "23", "-b:v", "0"]
            + ["-movflags", "+faststart"],
        )
        self.assertEqual(
            video.AV1_NVENC_FFMPEG_ARGS,
            ["-c:v", "av1_nvenc", "-pix_fmt", "yuv420p"]
            + ["-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", "23", "-b:v", "0"]
            + ["-movflags", "+faststart"],
        )
        self.assertEqual(
            video.SVTAV1_FFMPEG_ARGS,
            ["-c:v", "libsvtav1", "-pix_fmt", "yuv420p", "-preset", "8", "-crf", "30", "-movflags", "+faststart"],
        )

    @patch("checks.utils.video._is_nvenc_codec_available", return_value=True)
    def test_default_ladder_matches_the_legacy_h264_first_order(self, _: MagicMock) -> None:
        """With nothing configured, the ladder must still start at h264_nvenc rather than at the AV1 tier."""
        self.assertEqual(video.get_video_encode_args(), video.NVENC_H264_FFMPEG_ARGS)

    @patch("checks.utils.video._is_nvenc_codec_available", return_value=True)
    def test_vp9_selects_libvpx_even_with_hardware_present(self, _: MagicMock) -> None:
        """There is no NVENC VP9, so a VP9 request is a request for the software encoder however good the GPU is."""
        args = video.get_video_encode_args(codec=video.VideoCodec.VP9)
        self.assertEqual(args[:2], ["-c:v", "libvpx-vp9"])

    @patch("checks.utils.video._is_nvenc_codec_available", return_value=False)
    def test_exhausted_h264_ladder_falls_through_to_av1(self, _: MagicMock) -> None:
        """A CPU-only host cannot honor an H.264 request at all, so it must degrade rather than fail."""
        args = video.get_video_encode_args(codec=video.VideoCodec.H264)
        self.assertEqual(args[:2], ["-c:v", "libsvtav1"])

    @patch("checks.utils.video._is_nvenc_codec_available", return_value=False)
    def test_quality_tier_survives_a_codec_fallback(self, _: MagicMock) -> None:
        """The tier is resolved independently of the ladder, so falling back a codec must not reset it to standard.

        This is the point of keying the argument table by encoder: the request was "highest", the host forced a
        different encoder, and the request still has to be honored on whatever encoder answers.
        """
        args = video.get_video_encode_args(codec=video.VideoCodec.H264, quality=video.VideoQuality.HIGHEST)
        self.assertEqual(args[:2], ["-c:v", "libsvtav1"])
        # Compared against the table rather than a duplicated literal, so tuning the row cannot silently break this.
        highest = video._QUALITY_ARGS[("libsvtav1", video.VideoQuality.HIGHEST)]
        self.assertEqual(args[4 : 4 + len(highest)], highest)
        self.assertNotEqual(highest, video._QUALITY_ARGS[("libsvtav1", video.VideoQuality.STANDARD)])

    @patch("checks.utils.video._is_nvenc_codec_available", return_value=True)
    def test_highest_drops_the_rate_control_flags_it_makes_inert(self, _: MagicMock) -> None:
        """``-tune lossless`` makes ``-cq``/``-rc``/``-b:v`` inert; carrying them would invite tuning a dead knob."""
        args = video.get_video_encode_args(quality=video.VideoQuality.HIGHEST)
        self.assertEqual(args[:2], ["-c:v", "h264_nvenc"])
        self.assertIn("lossless", args)
        for inert in ("-cq", "-rc", "-b:v"):
            self.assertNotIn(inert, args)

    # The table is allowed exactly one gap, and it is load-bearing rather than an oversight: NVENC's lossless tune is
    # an H.264/HEVC feature that Ada's AV1 encoder does not implement, so a HIGHEST row for av1_nvenc could only ever
    # fail the encode. Named here so that adding a row, or a second gap, has to be a deliberate edit to this set.
    _EXPECTED_MISSING_ROWS: ClassVar[set[tuple[str, VideoQuality]]] = {("av1_nvenc", VideoQuality.HIGHEST)}

    def test_the_quality_table_has_exactly_the_expected_gaps(self) -> None:
        """A ladder naming an encoder with no row for a tier must be intentional; anything else is a KeyError waiting
        to happen on whichever host reaches that combination.
        """
        laddered = {encoder for ladder in video._ENCODE_LADDERS.values() for encoder in ladder}
        expected = {
            (encoder, tier) for encoder in laddered for tier in video.VideoQuality
        } - self._EXPECTED_MISSING_ROWS
        self.assertEqual(set(video._QUALITY_ARGS), expected)

    def test_every_codec_can_still_be_encoded_at_every_tier(self) -> None:
        """What the gap must not cost: some encoder, somewhere in the fallthrough, has to offer each tier.

        A gap is only acceptable because the ladder routes around it. If one ever left a codec with no encoder at all
        for a tier, that request would hard-fail on every host rather than degrade, which is the outcome the whole
        fallthrough design exists to avoid.
        """
        for codec, ladder in video._ENCODE_LADDERS.items():
            reachable = tuple(ladder) + video._ENCODE_LADDERS[video.VideoCodec.AV1]
            for tier in video.VideoQuality:
                self.assertTrue(
                    any((encoder, tier) in video._QUALITY_ARGS for encoder in reachable),
                    f"no encoder reachable from {codec.value} offers the {tier.value} tier",
                )

    @patch("checks.utils.video._is_nvenc_codec_available", return_value=True)
    def test_av1_at_highest_skips_hardware_even_when_it_is_available(self, _: MagicMock) -> None:
        """The gap's whole point, and the bug that motivated it.

        Requesting AV1 at HIGHEST on a host whose av1_nvenc probes fine must still land on libsvtav1, because the
        hardware encoder has no lossless mode and asking it for one aborts the encode with "No capable devices found".
        Before the tier took part in selection, this combination reached av1_nvenc and failed there.
        """
        args = video.get_video_encode_args(codec=video.VideoCodec.AV1, quality=video.VideoQuality.HIGHEST)
        self.assertEqual(args[:2], ["-c:v", "libsvtav1"])

    @patch("checks.utils.video._is_nvenc_codec_available", return_value=True)
    def test_av1_at_standard_still_prefers_hardware(self, _: MagicMock) -> None:
        """The other half: the gap is per-tier, so it must not push the ordinary AV1 request onto the CPU."""
        args = video.get_video_encode_args(codec=video.VideoCodec.AV1, quality=video.VideoQuality.STANDARD)
        self.assertEqual(args[:2], ["-c:v", "av1_nvenc"])

    def test_every_ladder_terminates_in_a_software_encoder(self) -> None:
        """Either directly or via the AV1 fallthrough; otherwise a host could end up with no encoder at all."""
        for codec, ladder in video._ENCODE_LADDERS.items():
            reachable = set(ladder) | set(video._ENCODE_LADDERS[video.VideoCodec.AV1])
            self.assertTrue(
                reachable - video._HARDWARE_ENCODERS,
                f"ladder for {codec.value} can be exhausted with no software tier to fall back to",
            )


class TestGetVideoDecodeArgs(unittest.TestCase):
    """Tests for the decode ladder."""

    def setUp(self) -> None:
        video._NVDEC_PROBE = None

    def tearDown(self) -> None:
        video._NVDEC_PROBE = None

    @staticmethod
    def _nvdec(*, av1: bool = False, stack: bool = True) -> MagicMock:
        """Return a stub ``_probe_nvdec`` reporting the given host capabilities."""
        return MagicMock(return_value=video._NvdecProbe(av1_cuvid_works=av1, nvdec_stack_loads=stack))

    def test_h264_selects_cuvid_when_the_nvdec_stack_loads(self) -> None:
        """There is no software AVC decoder in the vendored build, so cuvid is the only option -- and it must be used.

        H.264 is not probed directly the way AV1 is, because producing an H.264 sample to probe with would itself need
        the hardware in question. A loadable stack is taken as sufficient.
        """
        with patch("checks.utils.video._probe_nvdec", self._nvdec(stack=True)):
            self.assertEqual(video.get_video_decode_args("h264"), ["-c:v", "h264_cuvid"])

    def test_h264_raises_when_the_nvdec_stack_is_missing(self) -> None:
        """The hardware-only branch: no software tier to fall back to, so this host genuinely cannot decode H.264.

        The message has to name the likely cause, because the overwhelmingly common one is a fixable deployment
        mistake -- a container started without ``video`` in ``NVIDIA_DRIVER_CAPABILITIES`` -- rather than a CPU-only
        host, and the two are indistinguishable from ffmpeg's own error.
        """
        with (
            patch("checks.utils.video._probe_nvdec", self._nvdec(stack=False)),
            self.assertRaises(video.VideoDecodeError) as caught,
        ):
            video.get_video_decode_args("h264")
        self.assertIn("NVIDIA_DRIVER_CAPABILITIES", str(caught.exception))

    def test_hevc_raises_as_unsupported(self) -> None:
        """HEVC is unsupported in both tiers, not merely un-accelerated.

        The vendored build carries neither a software HEVC decoder (royalty exposure comparable to AVC) nor
        ``hevc_cuvid`` (no caller to justify it), so this is a policy decision and the error should read as one rather
        than as ffmpeg's generic "no decoder found".
        """
        with self.assertRaises(video.VideoDecodeError) as caught:
            video.get_video_decode_args("hevc")
        self.assertIn("unsupported codec hevc", str(caught.exception))

    def test_vp9_always_selects_the_software_decoder(self) -> None:
        """VP9 is the mirror image of H.264: software-only, because ``vp9_cuvid`` mistags colorspace as RGB.

        No probe and no host dependence -- VP9 decode is identical on every machine in the fleet.
        """
        self.assertEqual(video.get_video_decode_args("vp9"), ["-c:v", "vp9"])

    @patch("checks.utils.video._is_av1_cuvid_available", return_value=True)
    def test_av1_prefers_cuvid(self, _: MagicMock) -> None:
        self.assertEqual(video.get_video_decode_args("av1"), ["-c:v", "av1_cuvid"])

    @patch("checks.utils.video._is_av1_cuvid_available", return_value=False)
    def test_av1_falls_back_to_dav1d(self, _: MagicMock) -> None:
        """AV1 is the one codec with a software tier, so CPU-only hosts stay able to decode our own output."""
        self.assertEqual(video.get_video_decode_args("av1"), ["-c:v", "libdav1d"])

    def test_unsupported_codec_error_lists_what_is_supported(self) -> None:
        """MPEG-4 stands in for any codec outside the allow-list; the message should say what would have worked."""
        with self.assertRaises(video.VideoDecodeError) as caught:
            video.get_video_decode_args("mpeg4")
        message = str(caught.exception)
        self.assertIn("unsupported codec mpeg4", message)
        for supported in ("av1", "h264", "vp9"):
            self.assertIn(supported, message)

    def test_codec_name_is_case_insensitive(self) -> None:
        with patch("checks.utils.video._probe_nvdec", self._nvdec(stack=True)):
            self.assertEqual(video.get_video_decode_args("H264"), ["-c:v", "h264_cuvid"])

    def test_pixel_format_normalization_is_yuv420p(self) -> None:
        """The determinism guard: cuvid emits NV12 and libdav1d YUV420P, which convert to BGR differently."""
        self.assertEqual(video.DECODE_PIXEL_FORMAT_ARGS, ["-vf", "format=yuv420p"])

    def test_pixel_format_args_are_built_from_the_filter(self) -> None:
        """Call sites that already pass -vf concatenate the bare filter, so the two must not drift apart."""
        self.assertEqual(video.DECODE_PIXEL_FORMAT_ARGS, ["-vf", video.DECODE_PIXEL_FORMAT_FILTER])

    @patch("checks.utils.video.get_video_codec", return_value="av1")
    @patch("checks.utils.video._is_av1_cuvid_available", return_value=True)
    def test_input_args_resolve_the_probed_codec(self, _: MagicMock, mock_codec: MagicMock) -> None:
        self.assertEqual(video.get_video_decode_input_args("/tmp/clip.mp4"), ["-c:v", "av1_cuvid"])
        mock_codec.assert_called_once_with("/tmp/clip.mp4")

    @patch("checks.utils.video.get_video_codec", return_value=None)
    def test_input_args_empty_when_probe_fails(self, _: MagicMock) -> None:
        """A failed probe must degrade to ffmpeg's own choice rather than fail the decode."""
        self.assertEqual(video.get_video_decode_input_args("/tmp/clip.mp4"), [])


def _no_frames(*_: object, **__: object) -> Generator[np.ndarray]:
    """Stand in for a decode that yields nothing.

    A real generator rather than ``iter([])`` because both readers call ``.close()`` on the result in a ``finally``,
    which a list iterator does not have.
    """
    yield from ()


class TestDecodeCallSites(unittest.TestCase):
    """The readers must select a decoder before -i and normalize pixels on the way out."""

    def setUp(self) -> None:
        dims_patcher = patch.object(video, "get_video_dimensions", return_value=(4, 2))
        dims_patcher.start()
        self.addCleanup(dims_patcher.stop)

        decode_patcher = patch.object(video, "get_video_decode_input_args", return_value=["-c:v", "h264_cuvid"])
        decode_patcher.start()
        self.addCleanup(decode_patcher.stop)

        raw_patcher = patch.object(video, "_iter_raw_bgr_frames", side_effect=_no_frames)
        self.mock_raw = raw_patcher.start()
        self.addCleanup(raw_patcher.stop)

    def _captured_args(self) -> list[str]:
        return list(self.mock_raw.call_args.args[0])

    def test_iter_video_frames_places_decoder_before_input(self) -> None:
        list(video.iter_video_frames("/tmp/clip.mp4"))
        args = self._captured_args()
        self.assertEqual(args[: args.index("-i")], ["-c:v", "h264_cuvid"])

    def test_iter_video_frames_normalizes_pixel_format(self) -> None:
        list(video.iter_video_frames("/tmp/clip.mp4"))
        args = self._captured_args()
        self.assertEqual(args[args.index("-vf") + 1], video.DECODE_PIXEL_FORMAT_FILTER)

    def test_read_video_frame_places_decoder_before_input(self) -> None:
        video.read_video_frame("/tmp/clip.mp4", 7)
        args = self._captured_args()
        self.assertEqual(args[: args.index("-i")], ["-c:v", "h264_cuvid"])

    def test_read_video_frame_appends_normalization_to_the_select_filter(self) -> None:
        """A second -vf would silently drop the frame selection, so both must share one chain."""
        video.read_video_frame("/tmp/clip.mp4", 7)
        args = self._captured_args()
        self.assertEqual(args.count("-vf"), 1)
        self.assertEqual(args[args.index("-vf") + 1], f"select=eq(n\\,7),{video.DECODE_PIXEL_FORMAT_FILTER}")


class TestConvertVideoInputArgs(unittest.TestCase):
    """convert_video_ffmpeg must put input options before -i, where ffmpeg requires them."""

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.test_dir, True)
        self.input_video = self.test_dir / "in.mp4"
        self.input_video.write_text("dummy")
        self.output_video = self.test_dir / "out.mp4"

    def _run_and_capture(self, **kwargs: object) -> list[str]:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock()
            self.output_video.write_text("converted")
            video.convert_video_ffmpeg(
                input_video_path=self.input_video,
                ffmpeg_args=["-c:v", "libsvtav1"],
                output_video_path=self.output_video,
                **kwargs,  # type: ignore[arg-type]
            )
            return list(mock_run.call_args.args[0])

    def test_explicit_input_args_precede_input(self) -> None:
        cmd = self._run_and_capture(input_args=["-c:v", "av1_cuvid"])
        self.assertEqual(cmd[: cmd.index("-i")], ["ffmpeg", "-y", "-c:v", "av1_cuvid"])

    def test_empty_input_args_suppress_decoder_selection(self) -> None:
        cmd = self._run_and_capture(input_args=[])
        self.assertEqual(cmd[: cmd.index("-i")], ["ffmpeg", "-y"])

    def test_default_input_args_come_from_the_probe(self) -> None:
        with patch.object(video, "get_video_decode_input_args", return_value=["-c:v", "h264_cuvid"]) as mock_args:
            cmd = self._run_and_capture()
        self.assertEqual(cmd[: cmd.index("-i")], ["ffmpeg", "-y", "-c:v", "h264_cuvid"])
        mock_args.assert_called_once_with(str(self.input_video))


class TestConvertVideoFFmpeg(unittest.TestCase):
    """Test cases for convert_video_ffmpeg function."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.test_dir = Path(tempfile.mkdtemp())
        self.input_video = self.test_dir / "test_input.mp4"
        self.output_video = self.test_dir / "test_output.mp4"

        # Create a dummy input video file
        self.input_video.write_text("dummy video content")

        # Sample ffmpeg arguments
        self.sample_ffmpeg_args = ["-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-preset", "p4"]

        # convert_video_ffmpeg probes the input to choose a decoder, which would otherwise put a second
        # subprocess.run call in front of every assertion here. Stub it to no decoder so these tests stay about the
        # conversion command; the decoder args themselves are covered by TestConvertVideoInputArgs.
        decode_patcher = patch.object(video, "get_video_decode_input_args", return_value=[])
        decode_patcher.start()
        self.addCleanup(decode_patcher.stop)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        # Remove temporary directory and all contents
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_convert_video_with_explicit_output_path_success(self) -> None:
        """Test successful video conversion with explicit output path."""
        with patch("subprocess.run") as mock_run:
            # Mock successful subprocess execution
            mock_run.return_value = MagicMock()

            # Create output file to simulate ffmpeg success
            self.output_video.write_text("converted video content")

            result = video.convert_video_ffmpeg(
                input_video_path=self.input_video,
                ffmpeg_args=self.sample_ffmpeg_args,
                output_video_path=self.output_video,
            )

            # Verify result
            self.assertEqual(result, self.output_video)

            # Verify subprocess was called with correct arguments
            expected_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(self.input_video),
                "-an",
                "-c:v",
                "h264_nvenc",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "p4",
                str(self.output_video),
            ]
            mock_run.assert_called_once_with(expected_cmd, capture_output=False, text=True, check=True)

    @patch("tempfile.mkdtemp")
    def test_convert_video_replace_original_success(self, mock_mkdtemp: MagicMock) -> None:
        """Test successful video conversion that replaces original file."""
        # Setup mock temp directory
        temp_dir = self.test_dir / "temp_conversion"
        temp_dir.mkdir()
        mock_mkdtemp.return_value = str(temp_dir)

        temp_output = temp_dir / "test_input.compatible.mp4"

        with patch("subprocess.run") as mock_run:
            # Mock successful subprocess execution
            mock_run.return_value = MagicMock()

            # Create temp output file to simulate ffmpeg success
            temp_output.write_text("converted video content")

            result = video.convert_video_ffmpeg(
                input_video_path=self.input_video, ffmpeg_args=self.sample_ffmpeg_args, output_video_path=None
            )

            # Verify result - should return the original path (after replacement)
            self.assertEqual(result, self.input_video)

            # Verify temp directory was created in parent of input
            mock_mkdtemp.assert_called_once_with(dir=self.input_video.parent)

            # Verify subprocess was called with correct temp path
            expected_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(self.input_video),
                "-an",
                "-c:v",
                "h264_nvenc",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "p4",
                str(temp_output),
            ]
            mock_run.assert_called_once_with(expected_cmd, capture_output=False, text=True, check=True)

    def test_convert_video_input_not_exists(self) -> None:
        """Test conversion with non-existent input file."""
        non_existent_input = self.test_dir / "non_existent.mp4"

        with patch("checks.utils.video.logger") as mock_logger:
            result = video.convert_video_ffmpeg(
                input_video_path=non_existent_input,
                ffmpeg_args=self.sample_ffmpeg_args,
                output_video_path=self.output_video,
            )

            # Verify function returns None
            self.assertIsNone(result)

            # Verify error was logged
            mock_logger.error.assert_called_once_with(f"Input video not found: {non_existent_input}")

    def test_convert_video_subprocess_error(self) -> None:
        """Test conversion when subprocess fails."""
        with patch("subprocess.run") as mock_run:
            # Mock subprocess failure
            error = subprocess.CalledProcessError(returncode=1, cmd="ffmpeg", stderr="ffmpeg error message")
            mock_run.side_effect = error

            with patch("checks.utils.video.logger") as mock_logger:
                result = video.convert_video_ffmpeg(
                    input_video_path=self.input_video,
                    ffmpeg_args=self.sample_ffmpeg_args,
                    output_video_path=self.output_video,
                )

                # Verify function returns None
                self.assertIsNone(result)

                # Verify error was logged
                mock_logger.error.assert_called_once_with("ffmpeg conversion failed: %s", "ffmpeg error message")

    def test_convert_video_output_file_not_created(self) -> None:
        """Test conversion when ffmpeg succeeds but output file isn't created."""
        with patch("subprocess.run") as mock_run:
            # Mock successful subprocess execution
            mock_run.return_value = MagicMock()
            # Don't create output file to simulate this edge case

            with patch("checks.utils.video.logger") as mock_logger:
                result = video.convert_video_ffmpeg(
                    input_video_path=self.input_video,
                    ffmpeg_args=self.sample_ffmpeg_args,
                    output_video_path=self.output_video,
                )

                # Verify function returns None
                self.assertIsNone(result)

                # Verify error was logged
                mock_logger.error.assert_called_once_with("ffmpeg conversion completed but output file not found")

    def test_convert_video_general_exception(self) -> None:
        """Test conversion when an unexpected exception occurs."""
        with patch("subprocess.run") as mock_run:
            # Mock unexpected exception
            mock_run.side_effect = OSError("Unexpected OS error")

            with patch("checks.utils.video.logger") as mock_logger:
                result = video.convert_video_ffmpeg(
                    input_video_path=self.input_video,
                    ffmpeg_args=self.sample_ffmpeg_args,
                    output_video_path=self.output_video,
                )

                # Verify function returns None
                self.assertIsNone(result)

                # Verify error was logged (lazy formatting: format string + exception arg)
                mock_logger.error.assert_called_once()
                error_args = mock_logger.error.call_args.args
                self.assertEqual(error_args[0], "Error during video conversion: %s")
                self.assertIsInstance(error_args[1], OSError)
                self.assertEqual(str(error_args[1]), "Unexpected OS error")

    def test_convert_video_empty_ffmpeg_args(self) -> None:
        """Test conversion with empty ffmpeg arguments."""
        with patch("subprocess.run") as mock_run:
            # Mock successful subprocess execution
            mock_run.return_value = MagicMock()

            # Create output file to simulate ffmpeg success
            self.output_video.write_text("converted video content")

            result = video.convert_video_ffmpeg(
                input_video_path=self.input_video,
                ffmpeg_args=[],  # Empty args
                output_video_path=self.output_video,
            )

            # Verify result
            self.assertEqual(result, self.output_video)

            # Verify subprocess was called with minimal arguments (-an is always injected: video-only output)
            expected_cmd = ["ffmpeg", "-y", "-i", str(self.input_video), "-an", str(self.output_video)]
            mock_run.assert_called_once_with(expected_cmd, capture_output=False, text=True, check=True)

    def test_convert_video_complex_ffmpeg_args(self) -> None:
        """Test conversion with complex ffmpeg arguments."""
        complex_args = [
            "-c:v",
            "h264_nvenc",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "p4",
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            "23",
            "-b:v",
            "0",
            "-g",
            "30",
            "-bf",
            "2",
            "-movflags",
            "+faststart",
        ]

        with patch("subprocess.run") as mock_run:
            # Mock successful subprocess execution
            mock_run.return_value = MagicMock()

            # Create output file to simulate ffmpeg success
            self.output_video.write_text("converted video content")

            result = video.convert_video_ffmpeg(
                input_video_path=self.input_video, ffmpeg_args=complex_args, output_video_path=self.output_video
            )

            # Verify result
            self.assertEqual(result, self.output_video)

            # Verify subprocess was called with all arguments
            expected_cmd = ["ffmpeg", "-y", "-i", str(self.input_video), "-an", *complex_args, str(self.output_video)]
            mock_run.assert_called_once_with(expected_cmd, capture_output=False, text=True, check=True)

    @patch("checks.utils.video.logger")
    def test_convert_video_logging_success_with_output_path(self, mock_logger: MagicMock) -> None:
        """Test that success logging works correctly with explicit output path."""
        with patch("subprocess.run") as mock_run:
            # Mock successful subprocess execution
            mock_run.return_value = MagicMock()

            # Create output file to simulate ffmpeg success
            self.output_video.write_text("converted video content")

            video.convert_video_ffmpeg(
                input_video_path=self.input_video,
                ffmpeg_args=self.sample_ffmpeg_args,
                output_video_path=self.output_video,
            )

            # Verify logging calls
            expected_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(self.input_video),
                "-an",
                "-c:v",
                "h264_nvenc",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "p4",
                str(self.output_video),
            ]
            expected_calls = [
                call("Converting video with ffmpeg: %s", expected_cmd),
                call("Successfully converted video: %s", self.output_video),
            ]
            mock_logger.info.assert_has_calls(expected_calls)

    @patch("tempfile.mkdtemp")
    @patch("checks.utils.video.logger")
    def test_convert_video_logging_success_replace_original(
        self, mock_logger: MagicMock, mock_mkdtemp: MagicMock
    ) -> None:
        """Test that success logging works correctly when replacing original."""
        # Setup mock temp directory
        temp_dir = self.test_dir / "temp_conversion"
        temp_dir.mkdir()
        mock_mkdtemp.return_value = str(temp_dir)

        temp_output = temp_dir / "test_input.compatible.mp4"

        with patch("subprocess.run") as mock_run:
            # Mock successful subprocess execution
            mock_run.return_value = MagicMock()

            # Create temp output file to simulate ffmpeg success
            temp_output.write_text("converted video content")

            video.convert_video_ffmpeg(
                input_video_path=self.input_video, ffmpeg_args=self.sample_ffmpeg_args, output_video_path=None
            )

            # Verify logging calls
            expected_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(self.input_video),
                "-an",
                "-c:v",
                "h264_nvenc",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "p4",
                str(temp_output),
            ]
            expected_calls = [
                call("Converting video with ffmpeg: %s", expected_cmd),
                call("Successfully converted video: %s", self.input_video),
            ]
            mock_logger.info.assert_has_calls(expected_calls)

    def test_convert_video_hide_output_true(self) -> None:
        """Test that hide_output=True sets capture_output=True."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock()
            self.output_video.write_text("converted video content")

            video.convert_video_ffmpeg(
                input_video_path=self.input_video,
                ffmpeg_args=self.sample_ffmpeg_args,
                output_video_path=self.output_video,
                hide_output=True,
            )

            # Verify capture_output=True when hide_output=True
            mock_run.assert_called_once()
            _args, kwargs = mock_run.call_args
            self.assertTrue(kwargs["capture_output"])


def _ffprobe_result(stdout: str, returncode: int = 0, stderr: str = "") -> MagicMock:
    """Build a mock subprocess.run result mimicking an ffprobe JSON invocation."""
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


class TestGetVideoFPS(unittest.TestCase):
    def setUp(self) -> None:
        # The probe helpers fail fast on a missing file; these tests exercise probe parsing with fake paths, so treat
        # the input as existing and let the mocked ffprobe drive the outcome.
        exists_patcher = patch("checks.utils.video.os.path.exists", return_value=True)
        exists_patcher.start()
        self.addCleanup(exists_patcher.stop)

    def test_missing_file_raises(self) -> None:
        with patch("checks.utils.video.os.path.exists", return_value=False), self.assertRaises(video.VideoDecodeError):
            video.get_video_fps("/does/not/exist.mp4")

    @patch("checks.utils.video.subprocess.run")
    def test_success_returns_fps(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _ffprobe_result('{"streams": [{"r_frame_rate": "30000/1001"}]}')

        fps = video.get_video_fps("/path/to/video.mp4")

        assert fps is not None
        self.assertAlmostEqual(fps, 30000.0 / 1001.0, places=4)
        # ffprobe (not cv2) is used for metadata now.
        self.assertEqual(mock_run.call_args.args[0][0], "ffprobe")

    @patch("checks.utils.video.subprocess.run")
    def test_integer_rate_returns_fps(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _ffprobe_result('{"streams": [{"r_frame_rate": "24/1"}]}')
        self.assertEqual(video.get_video_fps("/v.mp4"), 24.0)

    @patch("checks.utils.video.subprocess.run")
    def test_returns_none_when_fps_zero(self, mock_run: MagicMock) -> None:
        """Test that get_video_fps returns None when FPS is zero or invalid."""
        mock_run.return_value = _ffprobe_result('{"streams": [{"r_frame_rate": "0/0"}]}')
        self.assertIsNone(video.get_video_fps("/v.mp4"))

    @patch("checks.utils.video.subprocess.run")
    def test_failed_probe_returns_none_and_logs(self, mock_run: MagicMock) -> None:
        """Test that get_video_fps returns None and logs error when ffprobe fails."""
        mock_run.return_value = _ffprobe_result("", returncode=1, stderr="bad input")

        with patch("checks.utils.video.logger") as mock_logger:
            fps = video.get_video_fps("/bad/video.mp4")
            self.assertIsNone(fps)
            mock_logger.error.assert_called_once()


class TestGetVideoFrameCount(unittest.TestCase):
    def setUp(self) -> None:
        exists_patcher = patch("checks.utils.video.os.path.exists", return_value=True)
        exists_patcher.start()
        self.addCleanup(exists_patcher.stop)

    def test_missing_file_raises(self) -> None:
        with patch("checks.utils.video.os.path.exists", return_value=False), self.assertRaises(video.VideoDecodeError):
            video.get_video_frame_count("/does/not/exist.mp4")

    @patch("checks.utils.video.subprocess.run")
    def test_uses_nb_frames_fast_path(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _ffprobe_result('{"streams": [{"nb_frames": "120"}]}')
        self.assertEqual(video.get_video_frame_count("/v.mp4"), 120)
        mock_run.assert_called_once()

    @patch("checks.utils.video.subprocess.run")
    def test_falls_back_to_count_frames(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = [
            _ffprobe_result('{"streams": [{}]}'),  # nb_frames missing
            _ffprobe_result('{"streams": [{"nb_read_frames": "97"}]}'),  # exact count
        ]
        self.assertEqual(video.get_video_frame_count("/v.mp4"), 97)
        self.assertEqual(mock_run.call_count, 2)

    @patch("checks.utils.video.subprocess.run")
    def test_returns_none_when_unprobeable(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _ffprobe_result("", returncode=1, stderr="bad")
        self.assertIsNone(video.get_video_frame_count("/v.mp4"))


# The stderr an ffprobe run produces while the MPS server is still coming up. Reproduced verbatim from a checker
# replica, because the whole point of the classification below is matching what the driver actually prints.
_MPS_NOT_READY_STDERR = (
    "[CUDA @ 0x57bde4876040] cu->cuInit(0) failed -> CUDA_ERROR_MPS_SERVER_NOT_READY: "
    "MPS server is not ready to accept new MPS client requests"
)


class TestProbeGpuFailureHandling(unittest.TestCase):
    """A probe that fails because CUDA would not start must not be reported as a bad file.

    Reading a video's dimensions looks like pure container metadata, but the allow-list build carries no software
    H.264 decoder, so ffprobe opens ``h264_cuvid`` to fill in stream info and the probe is only as available as the
    GPU. When MPS is mid-creation that probe exits non-zero, which is indistinguishable from a corrupt container
    unless the stderr is classified -- and readable videos were being failed as unprobeable because of it.
    """

    def setUp(self) -> None:
        video._DIMENSION_CACHE.clear()
        self.addCleanup(video._DIMENSION_CACHE.clear)
        identity_patcher = patch.object(video, "_file_identity", return_value=("/v.mp4", 1, 1))
        identity_patcher.start()
        self.addCleanup(identity_patcher.stop)
        # The backoff is real wall-clock time; the schedule is asserted instead of waited out.
        sleep_patcher = patch("checks.utils.video.time.sleep")
        self.mock_sleep = sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

    @patch("checks.utils.video.subprocess.run")
    def test_transient_mps_failure_is_retried_until_it_clears(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = [
            _ffprobe_result("", returncode=1, stderr=_MPS_NOT_READY_STDERR),
            _ffprobe_result('{"streams": [{"width": 1920, "height": 1080}]}'),
        ]

        self.assertEqual(video.get_video_dimensions("/v.mp4"), (1920, 1080))
        self.assertEqual(mock_run.call_count, 2)
        self.mock_sleep.assert_called_once_with(video._GPU_PROBE_BACKOFF_BASE_SECONDS)

    @patch("checks.utils.video.subprocess.run")
    def test_persistent_mps_failure_raises_gpu_unavailable(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _ffprobe_result("", returncode=1, stderr=_MPS_NOT_READY_STDERR)

        with self.assertRaises(video.GpuUnavailableDecodeError) as context:
            video.get_video_dimensions("/v.mp4")

        self.assertEqual(mock_run.call_count, video._GPU_PROBE_MAX_ATTEMPTS)
        # The message has to name the GPU and carry ffprobe's own words: "could not determine dimensions" is what
        # sent the previous investigation looking for a corrupt download.
        self.assertIn("could not initialize CUDA", str(context.exception))
        self.assertIn("MPS server is not ready", str(context.exception))

    @patch("checks.utils.video.subprocess.run")
    def test_gpu_unavailable_is_catchable_as_either_kind_of_failure(self, mock_run: MagicMock) -> None:
        """Both parents are load-bearing, and for different readers.

        ``VideoDecodeError`` so every existing decode handler keeps catching the GPU case unchanged.
        ``GpuUnavailableError`` so the checker framework -- which knows nothing about video and reacts to an unusable
        GPU by retiring the replica -- recognizes this the same way it recognizes an ONNX session that would not
        start.
        """
        mock_run.return_value = _ffprobe_result("", returncode=1, stderr=_MPS_NOT_READY_STDERR)

        with self.assertRaises(video.VideoDecodeError):
            video.get_video_dimensions("/v.mp4")

        with self.assertRaises(GpuUnavailableError):
            video.get_video_dimensions("/v.mp4")

    @patch("checks.utils.video.subprocess.run")
    def test_permanent_cuda_failure_is_not_retried(self, mock_run: MagicMock) -> None:
        """A host with no GPU will not grow one, so waiting to say so would only hold the job slot longer."""
        mock_run.return_value = _ffprobe_result(
            "", returncode=1, stderr="[AVHWDeviceContext @ 0x1] cu->cuInit(0) failed -> CUDA_ERROR_NO_DEVICE"
        )

        with self.assertRaises(video.GpuUnavailableDecodeError):
            video.get_video_dimensions("/v.mp4")

        mock_run.assert_called_once()
        self.mock_sleep.assert_not_called()

    @patch("checks.utils.video.subprocess.run")
    def test_unreadable_file_keeps_returning_none(self, mock_run: MagicMock) -> None:
        """The genuinely corrupt case keeps its old contract: one attempt, no exception, None."""
        mock_run.return_value = _ffprobe_result("", returncode=1, stderr="moov atom not found")

        self.assertIsNone(video.get_video_dimensions("/v.mp4"))
        mock_run.assert_called_once()
        self.mock_sleep.assert_not_called()

    @patch("checks.utils.video.subprocess.run")
    def test_frame_reader_surfaces_the_gpu_error(self, mock_run: MagicMock) -> None:
        """iter_video_frames must let the GPU error through rather than replacing it with its own message."""
        mock_run.return_value = _ffprobe_result("", returncode=1, stderr=_MPS_NOT_READY_STDERR)

        with self.assertRaises(video.GpuUnavailableDecodeError):
            next(video.iter_video_frames("/v.mp4"))


class TestVideoOverlayHelper(unittest.TestCase):
    """Tests for VideoOverlayHelper class."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.output_dir = self.temp_dir / "out"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("checks.utils.overlay_video_helper.VideoWriter")
    @patch("checks.utils.overlay_video_helper.get_video_frame_count", return_value=100)
    def test_initialize_video_writer_success(self, _mock_count: MagicMock, mock_writer_cls: MagicMock) -> None:
        writer = MagicMock()
        writer.open.return_value = True
        mock_writer_cls.return_value = writer

        helper = OverlayVideoHelper(clip_id="clip1", output_dir=str(self.output_dir))
        ok = helper.initialize_video_writer(
            video_path="/path/to/input.mp4",
            naming_suffix="renamed_suffix",
            target_width=320,
            target_height=240,
            fps=12.5,
        )

        self.assertTrue(ok)
        self.assertEqual(helper.frame_width, 320)
        self.assertEqual(helper.frame_height, 240)
        self.assertEqual(helper.fps, 12.5)
        self.assertEqual(helper.original_video_path, "/path/to/input.mp4")
        self.assertEqual(helper.get_frame_count(), 100)
        self.assertTrue(str(helper.video_output_path).endswith("clip1.renamed_suffix.mp4"))
        mock_writer_cls.assert_called_once()
        writer.open.assert_called_once()

    @patch("checks.utils.overlay_video_helper.VideoWriter")
    @patch("checks.utils.overlay_video_helper.get_video_frame_count", return_value=0)
    def test_initialize_video_writer_fails_opening_video(
        self, _mock_count: MagicMock, mock_writer_cls: MagicMock
    ) -> None:
        helper = OverlayVideoHelper(clip_id="clip1", output_dir=str(self.output_dir))
        ok = helper.initialize_video_writer(video_path="/bad/input.mp4")
        self.assertFalse(ok)
        mock_writer_cls.assert_not_called()

    @patch("checks.utils.overlay_video_helper.VideoWriter")
    @patch("checks.utils.overlay_video_helper.get_video_frame_count", return_value=100)
    def test_initialize_video_writer_writer_open_fails(
        self, _mock_count: MagicMock, mock_writer_cls: MagicMock
    ) -> None:
        # Writer that never opens
        writer = MagicMock()
        writer.open.return_value = False
        mock_writer_cls.return_value = writer

        helper = OverlayVideoHelper(clip_id="clip1", output_dir=str(self.output_dir))
        ok = helper.initialize_video_writer(video_path="/path/to/input.mp4")
        self.assertFalse(ok)

    def test_get_original_frame_success(self) -> None:
        helper = OverlayVideoHelper(clip_id="c", output_dir=str(self.output_dir))
        helper.frame_width = 64
        helper.frame_height = 48

        # Create a blue-ish BGR image 20x30
        bgr = np.zeros((30, 20, 3), dtype=np.uint8)
        bgr[..., 0] = 255
        reader = MagicMock()
        reader.read.return_value = bgr
        helper.frame_reader = reader

        rgb = helper.get_original_frame(frame_idx=5)
        self.assertEqual(rgb.shape, (48, 64, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        reader.read.assert_called_once_with(5)

    def test_get_original_frame_fallback_gray(self) -> None:
        helper = OverlayVideoHelper(clip_id="c", output_dir=str(self.output_dir))
        helper.frame_width = 32
        helper.frame_height = 24
        reader = MagicMock()
        reader.read.return_value = None
        helper.frame_reader = reader

        img = helper.get_original_frame(frame_idx=0)
        self.assertEqual(img.shape, (24, 32, 3))
        self.assertTrue(np.all(img == 128))

    def test_release_closes_the_frame_reader(self) -> None:
        """The decoder's ffmpeg child must die with the helper, not linger on a temp dir about to be deleted."""
        helper = OverlayVideoHelper(clip_id="c", output_dir=str(self.output_dir))
        reader = MagicMock()
        helper.frame_reader = reader
        helper.video_writer = MagicMock()

        helper.release()

        reader.close.assert_called_once_with()
        self.assertIsNone(helper.frame_reader)
        # A second release must not blow up: abort paths call it alongside the happy path.
        helper.release()

    def test_write_frame_delegates_to_writer(self) -> None:
        helper = OverlayVideoHelper(clip_id="c", output_dir=str(self.output_dir))
        helper.frame_width = 50
        helper.frame_height = 40

        frame = np.random.rand(10, 20, 3).astype(np.float32) * 255.0

        # Our VideoWriter handles color/resize internally; the helper just delegates.
        writer = MagicMock()
        writer.write_frame.return_value = True
        helper.video_writer = writer

        ok = helper.write_frame(frame=frame, frame_idx=1)
        self.assertTrue(ok)
        writer.write_frame.assert_called_once_with(frame, 1)

    def test_write_frame_when_writer_missing(self) -> None:
        helper = OverlayVideoHelper(clip_id="c", output_dir=str(self.output_dir))
        helper.frame_width = 10
        helper.frame_height = 10
        helper.video_writer = None

        with patch.object(helper, "logger") as mock_logger:
            ok = helper.write_frame(frame=np.zeros((10, 10, 3), dtype=np.uint8), frame_idx=2)
            self.assertFalse(ok)
            mock_logger.warning.assert_called()

    @patch("cv2.imwrite")
    def test_save_frame_image_success(self, mock_imwrite: MagicMock) -> None:
        mock_imwrite.return_value = True
        helper = OverlayVideoHelper(clip_id="c", output_dir=str(self.output_dir))
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        ok = helper.save_frame_image(frame=frame, frame_idx=3)
        self.assertTrue(ok)
        expected_path = self.output_dir / "overlay_frame_0003.png"
        args, _ = mock_imwrite.call_args
        self.assertEqual(Path(args[0]), expected_path)

    @patch("cv2.imwrite")
    def test_save_frame_image_exception(self, mock_imwrite: MagicMock) -> None:
        mock_imwrite.side_effect = RuntimeError("disk error")
        helper = OverlayVideoHelper(clip_id="c", output_dir=str(self.output_dir))
        with patch.object(helper, "logger") as mock_logger:
            ok = helper.save_frame_image(frame=np.zeros((4, 4, 3), dtype=np.uint8), frame_idx=1)
            self.assertFalse(ok)
            mock_logger.error.assert_called()

    def test_release_releases_writer(self) -> None:
        helper = OverlayVideoHelper(clip_id="clipX", output_dir=str(self.output_dir))
        helper.video_output_path = self.output_dir / "clipX.overlay.mp4"

        writer = MagicMock()
        helper.video_writer = writer

        helper.release()

        # The ffmpeg-backed VideoWriter encodes directly; release just finalizes it (no post-conversion pass).
        writer.release.assert_called_once()
        self.assertEqual(helper.video_output_path, self.output_dir / "clipX.overlay.mp4")

    def test_release_without_writer_is_noop(self) -> None:
        helper = OverlayVideoHelper(clip_id="clipY", output_dir=str(self.output_dir))
        helper.video_writer = None
        # Should not raise.
        helper.release()

    def test_add_legend_draws_expected_elements(self) -> None:
        helper = OverlayVideoHelper(clip_id="c", output_dir=str(self.output_dir))
        helper.frame_width = 200
        helper.frame_height = 120

        vis = np.zeros((120, 200, 3), dtype=np.uint8)
        legend = ["Item A", "Item B", "Item C"]

        # Mock OpenCV text sizing and drawing to assert calls/positions
        with (
            patch("cv2.getTextSize") as mock_get_text_size,
            patch("cv2.rectangle") as mock_rectangle,
            patch("cv2.addWeighted") as mock_add_weighted,
            patch("cv2.putText") as mock_put_text,
        ):
            # Return fixed size for any text queried
            mock_get_text_size.return_value = ((100, 12), 3)

            # Make addWeighted return a sentinel object to verify the return value
            blended_sentinel = object()
            mock_add_weighted.return_value = blended_sentinel

            result = helper.add_legend(vis.copy(), frame_idx=7, legend_items=legend)

            # Validate addWeighted called with overlay copy and original image
            self.assertTrue(mock_add_weighted.called)
            aw_args, _aw_kwargs = mock_add_weighted.call_args
            overlay_param = aw_args[0]
            base_param = aw_args[2]
            self.assertIsInstance(overlay_param, np.ndarray)
            self.assertIsNot(overlay_param, vis)
            self.assertEqual(overlay_param.shape, vis.shape)
            self.assertIs(base_param, mock_add_weighted.call_args[0][2])
            self.assertEqual(aw_args[1], 0.65)
            self.assertEqual(aw_args[3], 0.35)
            self.assertEqual(aw_args[4], 0)

            # Rectangle should be drawn around the legend area with expected coords
            # Using fixed text sizes: frame_h=12, margin=14, pad=12, legend_height=3*(12+4)=48
            expected_tl = (14 - 12, (12 + 44) - 12 - 12)  # (2, 32)
            expected_br = (14 + max(100, 100) + 12, (12 + 44) + 48 + 12)  # (126, 116)
            mock_rectangle.assert_called_once_with(overlay_param, expected_tl, expected_br, (24, 28, 32), -1)

            # putText should be called for frame text + each legend item at expected positions
            expected_calls = []
            # Frame text at (14, 56)
            expected_calls.append(
                call(blended_sentinel, "Frame: 7", (14, 56), video.cv2.FONT_HERSHEY_SIMPLEX, 0.6, (250, 240, 230), 2)
            )
            # Legend items starting at y = 56 + 12 + 10 = 78, step = 16
            legend_start_y = 56 + 12 + 10
            for i, item in enumerate(legend):
                expected_calls.append(
                    call(
                        blended_sentinel,
                        item,
                        (14, legend_start_y + i * (12 + 4)),
                        video.cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (250, 240, 230),
                        2,
                    )
                )

            # putText is called in order; verify subset and order
            mock_put_text.assert_has_calls(expected_calls, any_order=False)

            # The returned image is whatever addWeighted returned (then drawn on)
            self.assertIs(result, blended_sentinel)


class TestWriterReaderParity(unittest.TestCase):
    """Round-trip integration: VideoWriter -> ffprobe/ffmpeg readers, using the vendored LGPL ffmpeg binary.

    This is the decode/encode parity guard for the no-bundled-ffmpeg OpenCV cutover: it exercises real
    ffmpeg encode + ffprobe count + ffmpeg single-frame and streaming decode (no mocks).
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "roundtrip.mp4"
        # NVENC-safe resolution: h264_nvenc rejects tiny frames ("Frame Dimension less than minimum").
        self.width, self.height, self.num_frames = 256, 144, 5
        # Distinct solid colors with wide separation so they survive H.264 quantization.
        self.colors = [
            (200, 20, 20),
            (20, 200, 20),
            (20, 20, 200),
            (200, 200, 20),
            (20, 200, 200),
        ]
        writer = video.VideoWriter(self.path, fps=10.0, frame_width=self.width, frame_height=self.height)
        self.assertTrue(writer.open())
        for i, color in enumerate(self.colors):
            frame_rgb = np.full((self.height, self.width, 3), color, dtype=np.uint8)
            self.assertTrue(writer.write_frame(frame_rgb, i))
        writer.release()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writer_produces_decodable_file(self) -> None:
        self.assertTrue(self.path.exists())
        self.assertGreater(self.path.stat().st_size, 0)

    def test_frame_count_round_trips(self) -> None:
        self.assertEqual(video.get_video_frame_count(self.path), self.num_frames)

    def test_iter_video_frames_yields_all_frames(self) -> None:
        frames = list(video.iter_video_frames(str(self.path)))
        self.assertEqual(len(frames), self.num_frames)
        for frame in frames:
            self.assertEqual(frame.shape, (self.height, self.width, 3))

    def test_read_video_frame_is_frame_accurate(self) -> None:
        # read_video_frame returns BGR; compare against the BGR-reversed source color for the same index.
        for idx in (0, 2, 4):
            frame = video.read_video_frame(str(self.path), idx)
            self.assertIsNotNone(frame)
            assert frame is not None
            self.assertEqual(frame.shape, (self.height, self.width, 3))
            expected_bgr = np.array(self.colors[idx][::-1], dtype=np.float64)
            mean_bgr = frame.reshape(-1, 3).mean(axis=0)
            # The requested frame must be far closer to its own color than to any other frame's color.
            own_dist = float(np.linalg.norm(mean_bgr - expected_bgr))
            other_dists = [
                float(np.linalg.norm(mean_bgr - np.array(c[::-1], dtype=np.float64)))
                for j, c in enumerate(self.colors)
                if j != idx
            ]
            self.assertLess(own_dist, min(other_dists))

    def _assert_is_frame(self, frame: np.ndarray | None, idx: int) -> None:
        """Assert ``frame`` is the clip's frame ``idx``, by nearest-color match against the known source colors."""
        self.assertIsNotNone(frame, f"expected a frame at index {idx}")
        assert frame is not None
        self.assertEqual(frame.shape, (self.height, self.width, 3))
        mean_bgr = frame.reshape(-1, 3).mean(axis=0)
        distances = [float(np.linalg.norm(mean_bgr - np.array(c[::-1], dtype=np.float64))) for c in self.colors]
        self.assertEqual(int(np.argmin(distances)), idx, f"decoded frame does not match source frame {idx}")

    def test_sequential_reader_reads_frames_in_order(self) -> None:
        with video.SequentialFrameReader(str(self.path)) as reader:
            for idx in range(self.num_frames):
                self._assert_is_frame(reader.read(idx), idx)

    def test_sequential_reader_spawns_one_decode_per_forward_pass(self) -> None:
        """The whole point of the reader: an in-order walk costs one ffmpeg, not one per frame."""
        with video.SequentialFrameReader(str(self.path)) as reader:
            for idx in range(self.num_frames):
                reader.read(idx)
            self.assertEqual(reader.decode_passes, 1)

    def test_sequential_reader_skips_ahead_without_restarting(self) -> None:
        with video.SequentialFrameReader(str(self.path)) as reader:
            self._assert_is_frame(reader.read(1), 1)
            self._assert_is_frame(reader.read(4), 4)
            self.assertEqual(reader.decode_passes, 1)

    def test_sequential_reader_restarts_on_backward_seek(self) -> None:
        """Random access still works — it just costs a fresh decode, same as ``read_video_frame`` always did."""
        with video.SequentialFrameReader(str(self.path)) as reader:
            self._assert_is_frame(reader.read(3), 3)
            self._assert_is_frame(reader.read(1), 1)
            self.assertEqual(reader.decode_passes, 2)

    def test_sequential_reader_rereads_the_current_index_via_restart(self) -> None:
        with video.SequentialFrameReader(str(self.path)) as reader:
            self._assert_is_frame(reader.read(2), 2)
            self._assert_is_frame(reader.read(2), 2)

    def test_sequential_reader_returns_none_past_eof(self) -> None:
        """Matches ``read_video_frame``'s contract so callers keep their existing "no frame" branch."""
        with video.SequentialFrameReader(str(self.path)) as reader:
            self.assertIsNone(reader.read(self.num_frames + 10))

    def test_sequential_reader_returns_none_for_negative_index(self) -> None:
        with video.SequentialFrameReader(str(self.path)) as reader:
            self.assertIsNone(reader.read(-1))

    def test_sequential_reader_close_is_idempotent(self) -> None:
        reader = video.SequentialFrameReader(str(self.path))
        self._assert_is_frame(reader.read(0), 0)
        reader.close()
        reader.close()

    def test_sequential_reader_reopens_after_close(self) -> None:
        reader = video.SequentialFrameReader(str(self.path))
        self._assert_is_frame(reader.read(1), 1)
        reader.close()
        # A read after close restarts from scratch rather than raising on the torn-down generator.
        self._assert_is_frame(reader.read(1), 1)
        reader.close()

    def test_dimension_probe_is_cached_across_calls(self) -> None:
        """One ffprobe per file identity, not one per read — this is what made the per-frame reader so expensive."""
        video._DIMENSION_CACHE.clear()
        with patch.object(video.subprocess, "run", wraps=video.subprocess.run) as spy:
            first = video.get_video_dimensions(str(self.path))
            second = video.get_video_dimensions(str(self.path))
        self.assertEqual(first, (self.width, self.height))
        self.assertEqual(second, first)
        self.assertEqual(spy.call_count, 1)

    def test_dimension_cache_is_invalidated_when_the_file_changes(self) -> None:
        """Keying on (path, size, mtime) means a file replaced in place is re-probed, not served stale."""
        video._DIMENSION_CACHE.clear()
        self.assertEqual(video.get_video_dimensions(str(self.path)), (self.width, self.height))

        replacement = self.tmp / "replacement.mp4"
        writer = video.VideoWriter(replacement, fps=10.0, frame_width=320, frame_height=192)
        self.assertTrue(writer.open())
        writer.write_frame(np.full((192, 320, 3), 90, dtype=np.uint8), 0)
        writer.release()
        shutil.copy2(replacement, self.path)

        self.assertEqual(video.get_video_dimensions(str(self.path)), (320, 192))

    def test_dimension_probe_raises_on_missing_file(self) -> None:
        with self.assertRaises(video.VideoDecodeError):
            video.get_video_dimensions(str(self.tmp / "does_not_exist.mp4"))


if __name__ == "__main__":
    unittest.main()
