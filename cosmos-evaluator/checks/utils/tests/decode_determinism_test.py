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

"""Pin the cross-tier decode invariant: the hardware and software AV1 decoders must return identical pixels.

Which tier a clip decodes on is a property of the host, not of the request -- ``get_video_decode_args`` picks
``av1_cuvid`` where NVDEC supports AV1 and ``libdav1d`` everywhere else, and a GA100 falls to the software tier where an
L40S uses hardware. Any divergence between the two would therefore surface as scores that differ by which host in the
fleet happened to claim the job, with nothing in the output to distinguish them.

Exact equality is the correct assertion, not a tolerance. Both decoders are bit-exact by specification, and the
conversion to ``bgr24`` runs on CPU swscale in both tiers because the ladder names a standalone cuvid decoder instead of
``-hwaccel cuda`` (which would convert on-device via NPP, with different coefficients and rounding). So a failure here
means either NVDEC is non-conformant on this stream or something has reintroduced an on-device color conversion, and
both are worth failing on.

This is the regression that made CWIP outputs irreproducible under torchcodec, in a new form, so it is worth a test of
its own rather than a comment.
"""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from checks.utils.video import (
    DECODE_PIXEL_FORMAT_FILTER,
    _is_av1_cuvid_available,
    get_video_codec,
    iter_video_frames,
)
from utils.bazel import get_runfiles_path

# Any AV1 content fixture works; this one is 1280x704, small enough to keep the comparison cheap.
_FIXTURE = "checks/sample_data/augmented.mp4"

# Enough frames to cover an inter-coded run rather than just the keyframe, where the two decoders have the least room to
# disagree. The whole clip would be 300.
_FRAMES = 8


def _decode_with(decoder: str, video_path: str) -> list[np.ndarray]:
    """Decode the first ``_FRAMES`` frames with ``decoder`` forced, bypassing the host-dependent ladder."""
    with patch("checks.utils.video.get_video_decode_input_args", return_value=["-c:v", decoder]):
        return list(iter_video_frames(video_path, max_frames=_FRAMES))


class TestCrossTierDecodeDeterminism(unittest.TestCase):
    _fixture: str

    @classmethod
    def setUpClass(cls) -> None:
        fixture = get_runfiles_path(_FIXTURE)
        assert fixture is not None, f"Could not locate {_FIXTURE} in runfiles"
        assert Path(fixture).exists(), f"Fixture does not exist: {fixture}"
        cls._fixture = fixture

    def setUp(self) -> None:
        # A hard failure rather than a skip, and worth stating separately from the comparison below. AV1 NVDEC needs
        # GA10x or newer (RTX 30, A40, L40) -- a lower bar than the gpu_ada targets, but still not implied by the
        # plain `gpu` tag, since a GA100 has no AV1 decoder at all. Without this, a misrouted runner reports the
        # determinism invariant as broken when the real problem is that one of the two tiers does not exist here.
        self.assertTrue(
            _is_av1_cuvid_available(),
            "av1_cuvid cannot decode on this host, so there is no hardware tier to compare the software one against. "
            "This target needs a GPU with AV1 NVDEC (GA10x or newer: RTX 30, A40, L40); a GA100 has none.",
        )

    def test_fixture_is_av1(self) -> None:
        """Both decoders under test are AV1-only, so a re-encoded fixture would make this suite silently vacuous."""
        self.assertEqual(get_video_codec(self._fixture), "av1")

    def test_av1_cuvid_and_libdav1d_return_identical_frames(self) -> None:
        hardware = _decode_with("av1_cuvid", self._fixture)
        software = _decode_with("libdav1d", self._fixture)

        # Guards the comparison itself: a decoder that produced nothing would otherwise make the loop below trivially
        # pass on two empty lists.
        self.assertEqual(len(hardware), _FRAMES, "av1_cuvid did not decode the requested frames")
        self.assertEqual(len(software), _FRAMES, "libdav1d did not decode the requested frames")

        for index, (hardware_frame, software_frame) in enumerate(zip(hardware, software, strict=True)):
            self.assertEqual(hardware_frame.shape, software_frame.shape, f"frame {index} shape differs")
            if np.array_equal(hardware_frame, software_frame):
                continue
            differing = int(np.count_nonzero(hardware_frame != software_frame))
            largest = int(np.abs(hardware_frame.astype(np.int16) - software_frame.astype(np.int16)).max())
            self.fail(
                f"av1_cuvid and libdav1d disagree on frame {index}: {differing} of {hardware_frame.size} bytes "
                f"differ, largest delta {largest}/255. Either NVDEC is non-conformant on this stream or the "
                f"'{DECODE_PIXEL_FORMAT_FILTER}' normalization is no longer being applied to both tiers."
            )


if __name__ == "__main__":
    unittest.main()
