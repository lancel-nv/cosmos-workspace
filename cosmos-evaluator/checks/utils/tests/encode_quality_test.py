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

"""Pin what the ``highest`` quality tier actually guarantees, encoder by encoder, by round-tripping real pixels.

The tier is named "highest" rather than "lossless" because it is mathematically lossless for only two of the four
encoders. That asymmetry is documented in two places and is the thing most likely to be quietly broken by an edit to a
single argument row, so it is asserted here rather than trusted:

* ``h264_nvenc -tune lossless`` and ``libvpx-vp9 -lossless 1`` must round-trip bit-exactly.
* ``libsvtav1`` at ``highest`` must produce a *decodable* stream, and is expected **not** to be bit-exact. Asserting
  losslessness there would fail, and the failure would read as a bug rather than as a codec limitation.

Everything is measured in ``yuv420p``, which is the only color space in which the lossless claim is even meaningful.
Every encoder in the ladder is configured ``-pix_fmt yuv420p``, so a round trip starting from full-resolution RGB or BGR
would subsample chroma on the way in and could not be bit-exact for *any* encoder, lossless mode or not. Comparing in
the encoder's own space isolates what the tier controls.

The ``libsvtav1`` case also guards a trap worth a test of its own: ffmpeg's wrapper treats ``-crf 0`` as "unset" and
falls back to SVT's internal default of 35, which is coarser than the ``standard`` tier. A ``highest`` row that quietly
became the worst row in the table would be invisible without measuring reconstruction error, so this asserts that
``highest`` really does beat ``standard`` on the same source.

The fourth encoder, ``av1_nvenc``, is covered by ``av1_nvenc_test`` instead: it needs an Ada-class card, so it is routed
to a narrower runner and cannot share this target.
"""

from pathlib import Path
import re
import subprocess
import tempfile
import unittest

from checks.utils.tests.encode_round_trip_test_base import (
    ENCODE_TIMEOUT_SECONDS,
    EncodeRoundTripTestBase,
    raw_input_args,
)
from checks.utils.video import _build_encode_args, is_nvenc_available
from checks.utils.video_codec import VideoQuality


class TestLosslessRows(EncodeRoundTripTestBase):
    """The only two bit-exactness guarantees the ``highest`` tier makes."""

    def test_h264_nvenc_highest_is_bit_exact(self) -> None:
        """The reason anything needing reproducible re-encodes must pin ``video_codec=h264`` and ``highest``."""
        self.assertTrue(
            is_nvenc_available(),
            "h264_nvenc is unavailable on this host, but this target is tagged gpu and H.264 has no software encoder",
        )
        self._assert_bit_exact("h264_nvenc", "h264")

    def test_libvpx_vp9_highest_is_bit_exact(self) -> None:
        """VP9's lossless mode needs no hardware at either end, so this is the codec to pin on a CPU-only host."""
        self._assert_bit_exact("libvpx-vp9", "vp9", suffix=".webm")


class TestLossyRows(EncodeRoundTripTestBase):
    """The AV1 half, where ``highest`` means "best available mode" and explicitly not "lossless"."""

    def test_libsvtav1_highest_is_decodable_but_not_bit_exact(self) -> None:
        """Encodes the documented limitation rather than an aspiration: no runtime lossless AV1 encoder is shipped."""
        self._assert_not_bit_exact("libsvtav1", "av1")

    def test_libsvtav1_uses_the_crf_it_was_given(self) -> None:
        """Guard the ``-crf 0`` trap at its source: ffmpeg's libsvtav1 wrapper declares ``crf`` with a default of 0 and
        treats 0 as "unset", so ``-crf 0`` silently leaves SVT at its internal default of 35 -- coarser than the
        ``standard`` row's explicit 30, which would make a request for the best mode the worst row in the table.

        Asserted against SVT's own startup banner rather than by comparing reconstruction error to ``standard``. That
        comparison looks like the obvious test but has no teeth: the two rows differ in preset as well as CRF, and
        preset 2 at CRF 35 still reconstructs better than preset 8 at CRF 30, so the trap passes it unnoticed. Reading
        back the rate factor SVT actually adopted is the only assertion that isolates the variable.
        """
        for tier in VideoQuality:
            args = _build_encode_args("libsvtav1", tier)
            requested = int(args[args.index("-crf") + 1])
            with tempfile.TemporaryDirectory() as work_dir:
                clip = Path(work_dir) / "banner.mp4"
                # Info level, unlike every other encode here: the rate factor SVT settled on is only in the banner.
                result = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-loglevel", "info", "-y", *raw_input_args(), *args, str(clip)],
                    input=self._source,
                    capture_output=True,
                    check=False,
                    timeout=ENCODE_TIMEOUT_SECONDS,
                )
            report = result.stderr.decode(errors="replace")
            self.assertEqual(result.returncode, 0, report)
            banner = re.search(r"BRC mode / rate factor\s*:\s*(\S+)\s*/\s*(\d+)", report)
            if banner is None:
                self.fail(f"could not find SVT's rate factor banner for the {tier.value} row in:\n{report}")
            self.assertEqual(
                (banner.group(1), int(banner.group(2))),
                ("CRF", requested),
                f"the libsvtav1 {tier.value} row asks for -crf {requested} but SVT adopted "
                f"{banner.group(1)}/{banner.group(2)}. A CRF that ffmpeg's wrapper treats as unset (notably 0) falls "
                "back to SVT's internal default of 35.",
            )


if __name__ == "__main__":
    unittest.main()
