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

"""Guard the lockfile against packages that bring a codec policy of their own.

The service image tests catch a bundled FFmpeg once it is in an image. This catches it a step earlier and far more
cheaply, at the pin, and it also covers the case those probes cannot see: a wheel that links *our* libav* but chooses
its own decoder inside it. See
``third_party/ffmpeg/bundled_ffmpeg_audit.py`` for what qualifies and why.
"""

from pathlib import Path
import unittest

from third_party.ffmpeg.bundled_ffmpeg_audit import find_foreign_codec_pins, pinned_distributions

from utils.bazel import get_runfiles_path

_REQUIREMENTS = "requirements.txt"

# cosmos_eval pins none of them, and there is no reason it ever should: every image built from this tree gets its
# codecs from //third_party/ffmpeg.
_ACCEPTED: frozenset[str] = frozenset()


class TestLockfileCodecs(unittest.TestCase):
    _text: str

    @classmethod
    def setUpClass(cls) -> None:
        requirements = get_runfiles_path(_REQUIREMENTS)
        assert requirements is not None, f"Could not locate {_REQUIREMENTS} in runfiles"
        cls._text = Path(requirements).read_text(encoding="utf-8")

    def test_lockfile_parses_as_a_pinned_requirements_file(self) -> None:
        """Guard the parser, so a format change cannot turn the assertion below into a no-op."""
        self.assertGreater(len(pinned_distributions(self._text)), 10, "found almost no pins; is this a lockfile?")

    def test_no_unexpected_foreign_codec_packages(self) -> None:
        found = find_foreign_codec_pins(self._text)
        self.assertEqual(
            frozenset(found),
            _ACCEPTED,
            "requirements.txt pins packages that bypass the vendored FFmpeg: "
            + "; ".join(f"{name} ({reason})" for name, reason in sorted(found.items())),
        )


if __name__ == "__main__":
    unittest.main()
