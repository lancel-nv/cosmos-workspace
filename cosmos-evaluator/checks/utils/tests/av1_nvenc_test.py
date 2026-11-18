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

"""Exercise ``av1_nvenc`` on real hardware, the one encoder in the ladder that no other test can reach.

``av1_nvenc`` needs Ada or newer (RTX 40, L40/L40S). It is the only ladder entry with a hardware floor above "any
NVENC", so it is the only one that cannot be covered by a plain ``gpu``-tagged target: a ``gpu`` test lands on whichever
GPU runner is free, and an Ampere A40 cannot open this encoder at all. Hence the narrower ``gpu_ada`` tag, which routes
the target to Ada hardware, rather than a runtime skip -- which the repo forbids, and which would in any case have made
the gap invisible.

The gap is worth closing because this encoder is what ``COSMOS_EVALUATOR_VIDEO_CODEC=av1`` runs on an Ada host, so on a
meaningful slice of the fleet it is what actually executes. Its first run proved the point: the tier table gave
``av1_nvenc`` a ``highest`` row of ``-preset p7 -tune lossless``, copied from the ``h264_nvenc`` row above it and
documented in three places as merely "not bit-exact". On a real L40 that configuration does not encode at all. NVENC's
lossless tune is an H.264/HEVC feature, Ada's AV1 encoder has no lossless mode, and the rejection surfaces as the
thoroughly misleading "No capable devices found" -- on a card that is, in fact, a perfectly capable AV1 encoder. None
of the measurements behind that row could have been taken, since the only GPU available at the time was an Ampere
A6000 that cannot open this encoder under any settings.

So ``av1_nvenc`` now has no ``highest`` row, an AV1 request at that tier goes to ``libsvtav1``, and what is asserted
here is deliberately modest: that the encoder opens, that its one tier produces a stream the vendored ffmpeg decodes
back to the right number of frames, that the ladder reaches it at ``standard``, and that it is skipped at ``highest``.

Losslessness is not asserted as an aspiration anywhere here: no runtime lossless AV1 encoder exists in this build. See
``encode_quality_test`` for the two rows that are bit-exact, and ``_QUALITY_ARGS`` in ``checks/utils/video.py`` for the
table both files pin.
"""

import subprocess
import unittest

from checks.utils.tests.encode_round_trip_test_base import EncodeRoundTripTestBase
from checks.utils.video import _QUALITY_ARGS, _is_nvenc_codec_available, get_video_encode_args
from checks.utils.video_codec import VideoCodec, VideoQuality

_ENCODER = "av1_nvenc"


class TestAv1Nvenc(EncodeRoundTripTestBase):
    def setUp(self) -> None:
        # A hard failure, not a skip: this target is tagged gpu_ada precisely so that it only runs where the encoder
        # exists, and a host that cannot open it means the tag routing is wrong rather than that the test is untestable.
        self.assertTrue(
            _is_nvenc_codec_available(_ENCODER),
            f"{_ENCODER} cannot open on this host. This target is tagged gpu_ada and must be routed to an Ada-class "
            "runner (L40/L40S); Ampere cards such as the A40 have no AV1 encode block.",
        )

    def test_the_av1_ladder_prefers_hardware_here(self) -> None:
        """The assertion only an Ada host can make: that requesting AV1 reaches hardware rather than falling through.

        The ladder's ordering is covered on any host with the probe patched out, but only real Ada hardware proves the
        probe says yes to the encoder the ordering depends on.
        """
        args = get_video_encode_args(VideoCodec.AV1, VideoQuality.STANDARD)
        self.assertEqual(
            args[:2],
            ["-c:v", _ENCODER],
            f"an Ada host that can open {_ENCODER} should not be falling back to a software AV1 encoder",
        )

    def test_every_offered_tier_produces_a_decodable_av1_stream(self) -> None:
        """The core of what this target buys: the encoder runs at every tier the table offers it, and output decodes.

        Driven off the table rather than off ``VideoQuality`` so that restoring an ``av1_nvenc`` HIGHEST row -- with a
        lossy configuration Ada can actually honor -- brings it under test here automatically.

        Frame count is checked by the round trip itself, so a stream that decodes to a truncated sequence fails here
        rather than silently reducing the comparison to whatever both sides happened to produce.
        """
        offered = [tier for tier in VideoQuality if (_ENCODER, tier) in _QUALITY_ARGS]
        # Guards against the loop passing vacuously if the encoder ever loses its rows entirely. Which tiers are
        # offered is pinned in video_helper_test, so it deliberately is not restated here.
        self.assertTrue(offered, f"{_ENCODER} has no quality rows at all, so this target asserts nothing")
        for tier in offered:
            with self.subTest(quality=tier.value):
                self._round_trip(_ENCODER, tier, "av1")

    def test_highest_skips_this_encoder_on_hardware_that_could_otherwise_run_it(self) -> None:
        """The regression this target exists to hold: an AV1 ``highest`` request must not reach ``av1_nvenc``.

        Only an Ada host can make this assertion meaningfully. Everywhere else the encoder is unavailable anyway, so
        the fallthrough would happen for the wrong reason and prove nothing about the tier being what skipped it.
        """
        args = get_video_encode_args(VideoCodec.AV1, VideoQuality.HIGHEST)
        self.assertEqual(
            args[:2],
            ["-c:v", "libsvtav1"],
            f"{_ENCODER} has no lossless mode, so a highest-tier AV1 request must fall through to the software "
            "encoder rather than fail the encode with 'No capable devices found'",
        )

    def test_the_lossless_tune_is_still_rejected_by_this_hardware(self) -> None:
        """Why the row is absent rather than merely deprioritized: the configuration genuinely cannot run.

        Asserted against ffmpeg directly rather than through the availability probe, which reports any tier with no
        table row as unavailable and would therefore pass on the strength of the row's absence alone -- restating the
        table instead of testing the hardware. This is the one fact the whole arrangement rests on, so if a driver or
        ffmpeg update ever brings AV1 lossless support, this fails and prompts the table to be revisited rather than
        leaving an unnecessary fallthrough to the CPU in place indefinitely.
        """
        self.assertNotIn(
            (_ENCODER, VideoQuality.HIGHEST),
            _QUALITY_ARGS,
            "a highest row for av1_nvenc needs to be a lossy configuration; -tune lossless is not supported here",
        )
        attempt = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=256x256:d=0.1",
                "-c:v",
                _ENCODER,
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "p7",
                "-tune",
                "lossless",
                "-f",
                "null",
                "-",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertNotEqual(
            attempt.returncode,
            0,
            f"{_ENCODER} accepted -tune lossless on this host. If NVENC has gained AV1 lossless support, restore its "
            "_QUALITY_ARGS row and the tier tables in checks/utils/video.py and services/settings_base.py.",
        )


if __name__ == "__main__":
    unittest.main()
