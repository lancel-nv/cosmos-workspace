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

"""Guard the codec set of the *vendored* LGPL FFmpeg tarball (the artifact actually shipped in the checker images).

Unlike the encoder assertions in the per-service container ``image_test`` targets (which are ``gpu``-tagged and run a
loaded OCI image), this test operates directly on ``//third_party/ffmpeg:ffmpeg_lgpl_tar`` and needs no GPU: it extracts
the tarball and runs its own ``ffmpeg -encoders`` / ``ffmpeg -decoders``. Listing codecs reports compiled-in codecs and
does not dlopen the NVENC runtime, so ``h264_nvenc`` shows up without any NVIDIA hardware/driver present.

The build is an allow-list (``--disable-encoders --disable-decoders`` then explicit ``--enable-encoder``/
``--enable-decoder``, see ../tar.sh), which makes the enabled set small and knowable -- so this pins it by **exact set
equality** rather than by membership. Drift in either direction fails: a rebuild that loses SVT-AV1 or NVENC, that drops
the ``libdav1d`` software AV1 decoder (without which SVT-AV1 CPU output is undecodable on GPU-less hosts), or that
re-admits a patent-encumbered software H.264 implementation. Set equality is what makes the last case impossible to
miss; a forbidden-list only catches the implementations someone thought to name.
"""

import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

from utils.bazel import get_runfiles_path

# Must match the tarball wired into //third_party/ffmpeg:ffmpeg_lgpl_tar (and .dazelrc / the dazel image).
_TARBALL = "third_party/ffmpeg/ffmpeg-9.0.1-lgpl-nvenc-svtav1-linux-amd64.tar.gz"

# The complete encoder set the allow-list build produces.
#
# - h264_nvenc: NVIDIA's licensed hardware AVC encoder, the only H.264 encoding we do at all -- egomotion re-encodes
#   non-H.264 input for the NDAS VO binary. There is deliberately no software AVC counterpart, at any quality, because
#   that is what carries the AVC royalty.
# - av1_nvenc: hardware AV1, first fallback when the host cannot do H.264 (Ada and newer).
# - libsvtav1: royalty-free CPU AV1, the final fallback that always works.
# - libvpx-vp9: royalty-free CPU VP9, and one of only two encoders here that can be mathematically lossless. Note the
#   hyphen: configure takes `--enable-encoder=libvpx_vp9` but the codec is named `libvpx-vp9`, so this is the one entry
#   whose spelling does not match the build flag that produced it.
# - mjpeg, rawvideo, wrapped_avframe: intra-only/passthrough codecs that ffmpeg's own filters and image sinks need.
_EXPECTED_ENCODERS = frozenset(
    {
        "av1_nvenc",
        "h264_nvenc",
        "libsvtav1",
        "libvpx-vp9",
        "mjpeg",
        "rawvideo",
        "wrapped_avframe",
    }
)

# The complete decoder set, one row per supported codec per tier. H.264 is hardware-only by design: decoding carries the
# same royalty exposure as encoding, so no software AVC is enabled and H.264 input is undecodable without NVDEC. HEVC is
# absent in both tiers for the same royalty reason plus the absence of any caller. AV1 has both tiers, because libsvtav1
# output must stay decodable on GPU-less hosts. VP9 is software-only, which is the reverse asymmetry and is deliberate:
# vp9_cuvid mistags every frame's colorspace as RGB, which breaks the `format=yuv420p` determinism normalization -- see
# ../tar.sh. Being single-tier makes VP9 decode host-independent by construction.
_EXPECTED_DECODERS = frozenset(
    {
        "av1_cuvid",
        "h264_cuvid",
        "libdav1d",
        "mjpeg",
        "rawvideo",
        "vp9",
        "wrapped_avframe",
    }
)


def _parse_codec_names(listing: str) -> frozenset[str]:
    """Extract the exact codec-name tokens from an ``ffmpeg -encoders``/``-decoders`` listing.

    Parsing the name column is what lets the assertions above compare exact tokens. Substring matching against the raw
    listing cannot express set equality and is actively misleading here: ``h264_cuvid`` contains ``h264``, so a
    ``assertNotIn("h264", decoders)`` check would fail on a build that is in fact correct.
    """
    names: set[str] = set()
    in_table = False
    for line in listing.splitlines():
        if not in_table:
            # ffmpeg prints a flag legend, then a "------" rule, then one row per codec: " FLAGS name  description".
            in_table = line.strip().startswith("---")
            continue
        columns = line.split()
        if len(columns) >= 2:
            names.add(columns[1])
    return frozenset(names)


class TestVendoredFfmpegCodecs(unittest.TestCase):
    """Assert the vendored LGPL FFmpeg binary exposes exactly the expected encoder and decoder sets."""

    _tmpdir: tempfile.TemporaryDirectory[str]
    _encoders: frozenset[str]
    _decoders: frozenset[str]

    @classmethod
    def setUpClass(cls) -> None:
        """Extract the tarball once and read both codec listings out of the binary it ships."""
        tarball = get_runfiles_path(_TARBALL)
        assert tarball is not None, f"Could not locate vendored ffmpeg tarball in runfiles: {_TARBALL}"
        assert os.path.exists(tarball), f"Vendored ffmpeg tarball does not exist: {tarball}"

        cls._tmpdir = tempfile.TemporaryDirectory()
        with tarfile.open(tarball, "r:gz") as tf:
            # filter="data" (Python 3.12+) rejects unsafe members (absolute paths, parent traversal, links escaping
            # the destination), constraining extraction to the temporary directory.
            tf.extractall(cls._tmpdir.name, filter="data")

        root = Path(cls._tmpdir.name) / "usr" / "local"
        ffmpeg_bin = root / "bin" / "ffmpeg"
        assert ffmpeg_bin.exists(), f"ffmpeg binary missing from tarball at {ffmpeg_bin}"

        # The tarball ships its own libav*/libSvtAv1Enc/libdav1d shared objects; point the loader at them so the test
        # is independent of any ffmpeg installed on the host/PATH.
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(root / "lib"), env.get("LD_LIBRARY_PATH", "")]))

        cls._encoders = _parse_codec_names(cls._list_codecs(ffmpeg_bin, "-encoders", env))
        cls._decoders = _parse_codec_names(cls._list_codecs(ffmpeg_bin, "-decoders", env))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    @staticmethod
    def _list_codecs(ffmpeg_bin: Path, list_flag: str, env: dict[str, str]) -> str:
        result = subprocess.run(
            [str(ffmpeg_bin), "-hide_banner", list_flag],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg {list_flag} failed (rc={result.returncode}): {result.stderr.strip()}")
        return result.stdout

    def test_parser_finds_a_plausible_codec_table(self) -> None:
        """Guard the parser itself, so a parsing regression cannot make the set assertions vacuously pass.

        An empty or near-empty parse would otherwise fail the equality checks with a confusing diff rather than
        pointing at the real cause.
        """
        for label, parsed in (("encoders", self._encoders), ("decoders", self._decoders)):
            self.assertGreater(len(parsed), 1, f"parsed no plausible {label} table: {parsed}")
            self.assertNotIn("=", " ".join(parsed), f"flag-legend lines leaked into the {label} set: {parsed}")

    def test_vendored_ffmpeg_ships_exactly_the_expected_encoders(self) -> None:
        self.assertEqual(self._encoders, _EXPECTED_ENCODERS)

    def test_vendored_ffmpeg_ships_exactly_the_expected_decoders(self) -> None:
        self.assertEqual(self._decoders, _EXPECTED_DECODERS)

    def test_no_software_avc_or_hevc_in_either_direction(self) -> None:
        """Spell out the patent-driven half of the policy, so a failure names the reason rather than a set diff.

        Redundant with the equality assertions by construction, and deliberately so: this is the assertion whose
        failure message explains *why* the set is what it is.
        """
        # The bare names are FFmpeg's native (software) implementations; the rest are the external libraries that would
        # bring one in. Anything hardware is spelled with an _nvenc/_cuvid suffix and so cannot match.
        forbidden = {"h264", "hevc", "libopenh264", "libx264", "libx265"}
        for label, parsed in (("encoders", self._encoders), ("decoders", self._decoders)):
            software_avc_hevc = parsed & forbidden
            self.assertEqual(
                software_avc_hevc,
                set(),
                f"vendored ffmpeg ships software AVC/HEVC {label} {software_avc_hevc}; H.264 is restricted to "
                "NVIDIA's licensed hardware paths and HEVC is unsupported outright, both for patent reasons",
            )


if __name__ == "__main__":
    unittest.main()
