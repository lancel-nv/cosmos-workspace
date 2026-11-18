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

"""Unit tests for deriving the OpenCV wheel's third-party license file, plus an audit of the file the wheel ships."""

from pathlib import Path
import re
import tempfile
import unittest

from third_party.opencv.curate_third_party_licenses import (
    COPYRIGHT_HEADING,
    DROPPED_SECTIONS,
    KEPT_SECTIONS,
    SEPARATOR,
    AddedSection,
    build_dependencies,
    copyright_section,
    curate,
    installed_license_sections,
    split_sections,
)
from third_party.vendored_python import dist_info_dir, parse_facts

from utils.bazel import get_runfiles_path

_PREAMBLE = "OpenCV library is redistributed within opencv-python package.\n\nApache License text.\n\n"
_KEPT = next(iter(KEPT_SECTIONS))
_DROPPED = sorted(DROPPED_SECTIONS)[0]
_THIRD_PARTY_COPYRIGHTS = "Third party copyrights are property of their respective owners."
_COPYRIGHT_TEXT = f"Copyright (C) 2000, Example.\n\n{_THIRD_PARTY_COPYRIGHTS}\n"
_COPYRIGHT = AddedSection(COPYRIGHT_HEADING, _COPYRIGHT_TEXT, "COPYRIGHT")
_INSTALLED = AddedSection("ade-LICENSE, which OpenCV installs.", "\nADE license text.\n", "ade-LICENSE")
_INSTALLED_SUFFIX = ", which OpenCV installs for third-party code compiled into this build."


def _section(heading: str, body: str = "License text.", blank_before_heading: bool = False) -> str:
    gap = "\n" if blank_before_heading else ""
    return f"{SEPARATOR}\n{gap}{heading}\n\n{body}\n\n"


def _kept_sections() -> str:
    return "".join(_section(heading, f"{heading} license.") for heading in KEPT_SECTIONS)


def _upstream(*extra_sections: str) -> str:
    """Return an opencv-python-style file carrying every kept section, one dropped section, and ``extra_sections``."""
    return _PREAMBLE + _kept_sections() + _section(_DROPPED, "Dropped license.") + "".join(extra_sections)


class SplitSectionsTest(unittest.TestCase):
    """Covers splitting opencv-python's file into a preamble and headed sections."""

    def test_returns_the_text_before_the_first_separator_as_the_preamble(self) -> None:
        preamble, sections = split_sections(_PREAMBLE + _section(_KEPT))
        self.assertEqual(preamble, _PREAMBLE)
        self.assertEqual([section.text for section in sections], [_section(_KEPT)])

    def test_takes_the_first_non_blank_line_as_the_heading(self) -> None:
        _, sections = split_sections(_section(_KEPT, blank_before_heading=True))
        self.assertEqual([section.heading for section in sections], [_KEPT])

    def test_folds_files_blocks_into_the_preceding_section(self) -> None:
        files_block = _section("Files: src/example.c", "Per-file license.", blank_before_heading=True)
        _, sections = split_sections(_section(_KEPT) + files_block)
        self.assertEqual([section.heading for section in sections], [_KEPT])
        self.assertEqual(sections[0].text, _section(_KEPT) + files_block)

    def test_rejects_a_section_without_a_heading(self) -> None:
        with self.assertRaisesRegex(ValueError, "has no heading"):
            split_sections(f"{SEPARATOR}\n\n\n")

    def test_rejects_a_files_block_with_no_section_to_continue(self) -> None:
        with self.assertRaisesRegex(ValueError, "continues no section"):
            split_sections(_section("Files: src/example.c"))


class CurateTest(unittest.TestCase):
    """Covers selecting, ordering and extending the sections of the derived file."""

    def test_keeps_the_preamble_and_kept_sections_verbatim_and_drops_the_rest(self) -> None:
        curated = curate(_upstream(), ["libpng"], _COPYRIGHT, [])
        self.assertEqual(curated, (_PREAMBLE + _COPYRIGHT.render() + _kept_sections()).rstrip("\n") + "\n")
        self.assertNotIn(_DROPPED, curated)

    def test_places_the_copyright_notice_first_and_installed_licenses_last(self) -> None:
        _, sections = split_sections(curate(_upstream(), ["ade"], _COPYRIGHT, [_INSTALLED]))
        self.assertEqual(
            [section.heading for section in sections], [COPYRIGHT_HEADING, *KEPT_SECTIONS, _INSTALLED.heading]
        )
        self.assertEqual(sections[-1].text, f"{SEPARATOR}\n{_INSTALLED.heading}\n\nADE license text.\n")

    def test_rejects_a_section_that_is_neither_kept_nor_dropped(self) -> None:
        unknown = "Unknown library is redistributed within all opencv-python packages."
        with self.assertRaisesRegex(ValueError, rf"neither kept nor dropped.*\['{re.escape(unknown)}'\]"):
            curate(_upstream(_section(unknown)), [], _COPYRIGHT, [])

    def test_rejects_an_upstream_file_that_lacks_a_kept_section(self) -> None:
        upstream = _upstream().replace(_section(_KEPT, f"{_KEPT} license."), "")
        with self.assertRaisesRegex(ValueError, "lacks kept sections"):
            curate(upstream, [], _COPYRIGHT, [])

    def test_rejects_a_build_dependency_that_nothing_attributes(self) -> None:
        with self.assertRaisesRegex(ValueError, r"neither a kept section nor an installed license.*\['zlib'\]"):
            curate(_upstream(), ["libpng", "zlib"], _COPYRIGHT, [])

    def test_accepts_a_build_dependency_attributed_by_an_installed_license(self) -> None:
        self.assertIn(_INSTALLED.heading, curate(_upstream(), ["ade"], _COPYRIGHT, [_INSTALLED]))
        with self.assertRaisesRegex(ValueError, r"\['ade'\]"):
            curate(_upstream(), ["ade"], _COPYRIGHT, [])


class BuildDependenciesTest(unittest.TestCase):
    """Covers reading the third-party libraries out of the build configuration embedded in the cv2 extension."""

    def test_reads_the_libraries_of_the_build_configuration_line(self) -> None:
        extension = b"\x00\x7fELF...\n    3rdparty dependencies:       libprotobuf ade libpng\n  OpenCV modules:\n"
        self.assertEqual(build_dependencies(extension), ["libprotobuf", "ade", "libpng"])

    def test_rejects_an_extension_without_the_line(self) -> None:
        with self.assertRaisesRegex(ValueError, "no '3rdparty dependencies'"):
            build_dependencies(b"\x00\x7fELF")


class AddedSectionsTest(unittest.TestCase):
    """Covers reading OpenCV's copyright notice and the license files OpenCV installs."""

    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.licenses = self.root / "licenses"
        self.licenses.mkdir()

    def _install(self, name: str, text: str) -> None:
        (self.licenses / name).write_text(text, encoding="utf-8")

    def test_reads_the_copyright_notice(self) -> None:
        copyright_file = self.root / "COPYRIGHT"
        copyright_file.write_text(_COPYRIGHT_TEXT, encoding="utf-8")
        section = copyright_section(copyright_file)
        self.assertEqual(section.heading, COPYRIGHT_HEADING)
        self.assertIn(_THIRD_PARTY_COPYRIGHTS, section.body)

    def test_adds_each_installed_license_not_covered_by_a_kept_section(self) -> None:
        self._install("ittnotify-BSD-3-Clause.txt", "ITT BSD license.\n")
        self._install("SoftFloat-COPYING.txt", "SoftFloat license.\n")
        self._install("libpng-LICENSE", "libpng license.\n")
        self._install("ade-LICENSE", "ADE license.\n")

        sections = installed_license_sections(self.licenses)
        self.assertEqual(
            [section.source for section in sections],
            ["ade-LICENSE", "ittnotify-BSD-3-Clause.txt", "SoftFloat-COPYING.txt"],
        )
        self.assertEqual(sections[0].heading, f"ade-LICENSE{_INSTALLED_SUFFIX}")
        self.assertEqual(sections[0].body, "ADE license.\n")
        self.assertTrue(sections[1].attributes("ittnotify"))
        self.assertFalse(sections[1].attributes("itt"))

    def test_rejects_an_empty_licenses_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "installed no third-party license files"):
            installed_license_sections(self.licenses)


class ShippedLicenseFileTest(unittest.TestCase):
    """Audits the LICENSE-3RD-PARTY.txt in the .dist-info of the committed wheel."""

    def test_carries_the_copyright_notice_kept_sections_and_installed_licenses_only(self) -> None:
        facts = get_runfiles_path("vendored_python_repos.txt")
        assert facts is not None, "Could not locate vendored_python_repos.txt in runfiles"
        repo = next(repo for repo in parse_facts(Path(facts).read_text(encoding="utf-8")) if repo.kind == "wheel")
        # source_repo="" is the main module's repo mapping, under which the wheel's repo carries its apparent name.
        shipped = get_runfiles_path(
            f"{dist_info_dir(repo)}/LICENSE-3RD-PARTY.txt", external_repo=repo.name, source_repo=""
        )
        assert shipped is not None, f"Could not locate the LICENSE-3RD-PARTY.txt of {repo.name} in runfiles"

        text = Path(shipped).read_text(encoding="utf-8")
        headings = [section.heading for section in split_sections(text)[1]]
        kept_end = 1 + len(KEPT_SECTIONS)
        self.assertEqual(headings[:kept_end], [COPYRIGHT_HEADING, *KEPT_SECTIONS])
        self.assertGreater(len(headings), kept_end)
        self.assertEqual([heading for heading in headings[kept_end:] if not heading.endswith(_INSTALLED_SUFFIX)], [])
        self.assertIn(_THIRD_PARTY_COPYRIGHTS, text)


if __name__ == "__main__":
    unittest.main()
