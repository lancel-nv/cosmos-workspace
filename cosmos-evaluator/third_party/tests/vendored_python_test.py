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

"""Unit tests for the vendored Python metadata audit, plus the audit of this module's own vendored repos."""

from pathlib import Path
import tempfile
from typing import ClassVar
import unittest

from third_party.vendored_python import (
    DistInfo,
    VendoredRepo,
    audit_metadata,
    audit_overlays,
    dist_info_dir,
    escaped_name,
    expected_distribution,
    normalized_name,
    overlay_path,
    parse_facts,
    read_dist_info,
)

from utils.bazel import get_runfiles_path

_FACTS_FILE = "vendored_python_repos.txt"

_ARCHIVE = VendoredRepo(
    kind="archive",
    name="detectron2",
    src=None,
    build_file="//third_party/detectron2:BUILD.detectron2",
    dist_info_name="detectron2",
    dist_info_root="third_party",
    dist_info_version="0.6+a1ce2f95",
)

_CLEAN_DIST_INFO = DistInfo(
    name="detectron2",
    version="0.6+a1ce2f95",
    license_expression="Apache-2.0",
    files=frozenset({"METADATA", "licenses/LICENSE"}),
)


def _overlay(*rules: str) -> str:
    return "\n\n".join(rules) + "\n"


def _py_library(name: str, *, carries_dist_info: bool) -> str:
    data = '    data = [":dist_info"],\n' if carries_dist_info else ""
    return f'py_library(\n    name = "{name}",\n    srcs = ["a.py"],\n{data})'


class NameNormalizationTest(unittest.TestCase):
    """Covers the two distinct name spellings a distribution has."""

    def test_normalized_name_folds_separators_and_case(self) -> None:
        self.assertEqual(normalized_name("Cosmos_Drive.Dreams--Toolkits"), "cosmos-drive-dreams-toolkits")

    def test_escaped_name_matches_the_dist_info_directory_spelling(self) -> None:
        self.assertEqual(escaped_name("cosmos-drive-dreams-toolkits"), "cosmos_drive_dreams_toolkits")

    def test_escaped_name_folds_case_and_dots_like_the_extension_does(self) -> None:
        # http_archive.bzl lowercases and collapses runs of -_. to one underscore. Escaping a declared name any other
        # way points the audit at a directory the extension never wrote.
        self.assertEqual(escaped_name("Zope.Interface"), "zope_interface")
        self.assertEqual(escaped_name("cosmos--drive._dreams"), "cosmos_drive_dreams")


class ParseFactsTest(unittest.TestCase):
    """Covers parsing the genrule output into declarations."""

    def test_parses_an_archive_and_a_wheel(self) -> None:
        facts = (
            "archive\tname\tdetectron2\n"
            "archive\tbuild_file\t//third_party/detectron2:BUILD.detectron2\n"
            "archive\tdist_info_name\tdetectron2\n"
            "archive\tdist_info_root\tthird_party\n"
            "archive\tdist_info_version\t0.6+a1ce2f95\n"
            "wheel\tname\topencv_python_headless\n"
            "wheel\tsrc\t//third_party/opencv:opencv_python_headless-4.13.0.92+noffmpeg-cp313-cp313-linux.whl\n"
            "wheel\tbuild_file\t//third_party/opencv:BUILD.opencv\n"
        )
        repos = parse_facts(facts)
        self.assertEqual([repo.name for repo in repos], ["detectron2", "opencv_python_headless"])
        self.assertEqual(repos[0], _ARCHIVE)
        self.assertEqual(repos[1].kind, "wheel")
        self.assertIsNone(repos[1].dist_info_version)

    def test_parses_an_archive_that_declares_no_metadata(self) -> None:
        repos = parse_facts("archive\tname\tcosmos_metrics\n")
        self.assertEqual(len(repos), 1)
        self.assertIsNone(repos[0].dist_info_name)

    def test_ignores_blank_lines(self) -> None:
        self.assertEqual(len(parse_facts("\narchive\tname\tdetectron2\n\n")), 1)

    def test_rejects_a_line_that_is_not_three_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "line 1 .* is not"):
            parse_facts("archive\tname\n")

    def test_rejects_an_unexpected_attribute(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected attribute 'integrity'"):
            parse_facts("archive\tname\tdetectron2\narchive\tintegrity\tsha256-abc\n")


class DeclaredLocationTest(unittest.TestCase):
    """Covers deriving the paths a declaration implies."""

    def test_archive_distribution_comes_from_the_dist_info_attributes(self) -> None:
        self.assertEqual(expected_distribution(_ARCHIVE), ("detectron2", "0.6+a1ce2f95"))

    def test_wheel_distribution_comes_from_the_filename(self) -> None:
        wheel = VendoredRepo(
            kind="wheel",
            name="opencv_python_headless",
            src="//third_party/opencv:opencv_python_headless-4.13.0.92+noffmpeg-cp313-cp313-linux.whl",
            build_file="//third_party/opencv:BUILD.opencv",
            dist_info_name=None,
            dist_info_root=None,
            dist_info_version=None,
        )
        self.assertEqual(expected_distribution(wheel), ("opencv_python_headless", "4.13.0.92+noffmpeg"))
        self.assertEqual(dist_info_dir(wheel), "opencv_python_headless-4.13.0.92+noffmpeg.dist-info")

    def test_archive_without_metadata_declares_no_distribution(self) -> None:
        bare = VendoredRepo("archive", "cosmos_metrics", None, "//third_party/x:BUILD.x", None, None, None)
        self.assertIsNone(expected_distribution(bare))
        self.assertIsNone(dist_info_dir(bare))

    def test_dist_info_dir_escapes_the_name_and_honors_the_root(self) -> None:
        toolkits = VendoredRepo(
            kind="archive",
            name="cosmos_drive_dreams_toolkits",
            src=None,
            build_file="//third_party/cosmos_drive_dreams_toolkits:BUILD.cosmos_drive_dreams_toolkits",
            dist_info_name="cosmos-drive-dreams-toolkits",
            dist_info_root="third_party/cosmos_drive_dreams_toolkits",
            dist_info_version="0+908ba8f6",
        )
        self.assertEqual(
            dist_info_dir(toolkits),
            "third_party/cosmos_drive_dreams_toolkits/cosmos_drive_dreams_toolkits-0+908ba8f6.dist-info",
        )

    def test_overlay_path_turns_a_label_into_a_workspace_path(self) -> None:
        self.assertEqual(overlay_path(_ARCHIVE), "third_party/detectron2/BUILD.detectron2")


class ReadDistInfoTest(unittest.TestCase):
    """Covers reading a .dist-info off disk."""

    def test_reads_headers_and_lists_the_files_beside_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dist_info = Path(directory, "detectron2-0.6.dist-info")
            dist_info.joinpath("licenses").mkdir(parents=True)
            dist_info.joinpath("METADATA").write_text(
                "Metadata-Version: 2.4\nName: detectron2\nVersion: 0.6\nLicense-Expression: Apache-2.0\n\nbody\n",
                encoding="utf-8",
            )
            dist_info.joinpath("licenses", "LICENSE").write_text("Apache License\n", encoding="utf-8")

            parsed = read_dist_info(dist_info)

        self.assertEqual(parsed.name, "detectron2")
        self.assertEqual(parsed.version, "0.6")
        self.assertEqual(parsed.license_expression, "Apache-2.0")
        self.assertEqual(parsed.files, frozenset({"METADATA", "licenses/LICENSE"}))

    def test_falls_back_to_the_legacy_license_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dist_info = Path(directory, "opencv_python_headless-4.13.0.92.dist-info")
            dist_info.mkdir()
            dist_info.joinpath("METADATA").write_text(
                "Metadata-Version: 2.1\nName: opencv-python-headless\nVersion: 4.13.0.92\nLicense: Apache 2.0\n",
                encoding="utf-8",
            )

            self.assertEqual(read_dist_info(dist_info).license_expression, "Apache 2.0")


class AuditMetadataTest(unittest.TestCase):
    """Covers the metadata invariants."""

    def test_accepts_a_declaration_matched_by_its_shipped_metadata(self) -> None:
        self.assertEqual(audit_metadata([_ARCHIVE], {"detectron2": _CLEAN_DIST_INFO}, {}), [])

    def test_flags_an_archive_that_declares_no_metadata(self) -> None:
        bare = VendoredRepo("archive", "cosmos_metrics", None, "//third_party/x:BUILD.x", None, None, None)
        violations = audit_metadata([bare], {}, {})
        self.assertEqual(len(violations), 1)
        self.assertIn("declares no Python metadata", violations[0])

    def test_accepts_an_exempted_archive_that_declares_no_metadata(self) -> None:
        bare = VendoredRepo("archive", "cosmos_metrics", None, "//third_party/x:BUILD.x", None, None, None)
        self.assertEqual(audit_metadata([bare], {}, {"cosmos_metrics": "upstream declares no license"}), [])

    def test_flags_an_exemption_that_is_no_longer_needed(self) -> None:
        violations = audit_metadata([_ARCHIVE], {"detectron2": _CLEAN_DIST_INFO}, {"detectron2": "historical"})
        self.assertEqual(len(violations), 1)
        self.assertIn("now declares metadata", violations[0])

    def test_flags_an_exemption_for_a_repo_that_is_gone(self) -> None:
        violations = audit_metadata([], {}, {"removed_repo": "some reason"})
        self.assertEqual(len(violations), 1)
        self.assertIn("no longer declared in MODULE.bazel", violations[0])

    def test_flags_an_exemption_without_a_reason(self) -> None:
        bare = VendoredRepo("archive", "cosmos_metrics", None, "//third_party/x:BUILD.x", None, None, None)
        violations = audit_metadata([bare], {}, {"cosmos_metrics": "   "})
        self.assertEqual(len(violations), 1)
        self.assertIn("without a reason", violations[0])

    def test_flags_metadata_that_is_not_where_the_declaration_says(self) -> None:
        violations = audit_metadata([_ARCHIVE], {"detectron2": None}, {})
        self.assertEqual(len(violations), 1)
        self.assertIn("no readable METADATA at @detectron2//third_party/detectron2-0.6+a1ce2f95", violations[0])

    def test_flags_a_name_that_disagrees_with_the_declaration(self) -> None:
        shipped = DistInfo("detectron", "0.6+a1ce2f95", "Apache-2.0", frozenset({"licenses/LICENSE"}))
        violations = audit_metadata([_ARCHIVE], {"detectron2": shipped}, {})
        self.assertEqual(len(violations), 1)
        self.assertIn("reports the name 'detectron'", violations[0])

    def test_tolerates_a_name_that_differs_only_in_normalization(self) -> None:
        # The escaped spelling a .dist-info directory uses is not the canonical one the declaration carries.
        toolkits = VendoredRepo(
            kind="archive",
            name="cosmos_drive_dreams_toolkits",
            src=None,
            build_file="//third_party/x:BUILD.x",
            dist_info_name="cosmos-drive-dreams-toolkits",
            dist_info_root=None,
            dist_info_version="0+908ba8f6",
        )
        shipped = DistInfo("cosmos_drive_dreams_toolkits", "0+908ba8f6", "Apache-2.0", frozenset({"LICENSE"}))
        self.assertEqual(audit_metadata([toolkits], {"cosmos_drive_dreams_toolkits": shipped}, {}), [])

    def test_flags_a_version_that_disagrees_with_the_declaration(self) -> None:
        shipped = DistInfo("detectron2", "0.6", "Apache-2.0", frozenset({"licenses/LICENSE"}))
        violations = audit_metadata([_ARCHIVE], {"detectron2": shipped}, {})
        self.assertEqual(len(violations), 1)
        self.assertIn("reports version '0.6' but the declaration says '0.6+a1ce2f95'", violations[0])

    def test_flags_metadata_with_no_license(self) -> None:
        shipped = DistInfo("detectron2", "0.6+a1ce2f95", None, frozenset({"licenses/LICENSE"}))
        violations = audit_metadata([_ARCHIVE], {"detectron2": shipped}, {})
        self.assertEqual(len(violations), 1)
        self.assertIn("states no license", violations[0])

    def test_flags_metadata_with_no_license_text(self) -> None:
        shipped = DistInfo("detectron2", "0.6+a1ce2f95", "Apache-2.0", frozenset({"METADATA", "RECORD"}))
        violations = audit_metadata([_ARCHIVE], {"detectron2": shipped}, {})
        self.assertEqual(len(violations), 1)
        self.assertIn("carries no license text", violations[0])

    def test_accepts_license_text_at_the_dist_info_root(self) -> None:
        shipped = DistInfo("detectron2", "0.6+a1ce2f95", "Apache-2.0", frozenset({"LICENSE-3RD-PARTY.txt"}))
        self.assertEqual(audit_metadata([_ARCHIVE], {"detectron2": shipped}, {}), [])


class AuditOverlaysTest(unittest.TestCase):
    """Covers the requirement that overlay targets carry the metadata into runfiles."""

    def test_accepts_an_overlay_whose_libraries_all_carry_the_metadata(self) -> None:
        overlay = _overlay(
            'filegroup(\n    name = "dist_info",\n    srcs = glob(["*.dist-info/**"]),\n)',
            _py_library("config", carries_dist_info=True),
            _py_library("registry", carries_dist_info=True),
        )
        self.assertEqual(audit_overlays([_ARCHIVE], {"detectron2": overlay}), [])

    def test_flags_a_library_that_does_not_carry_the_metadata(self) -> None:
        overlay = _overlay(
            _py_library("config", carries_dist_info=True),
            _py_library("registry", carries_dist_info=False),
        )
        violations = audit_overlays([_ARCHIVE], {"detectron2": overlay})
        self.assertEqual(len(violations), 1)
        self.assertIn("py_library 'registry'", violations[0])

    def test_flags_an_overlay_missing_from_runfiles(self) -> None:
        violations = audit_overlays([_ARCHIVE], {"detectron2": None})
        self.assertEqual(len(violations), 1)
        self.assertIn("third_party/detectron2/BUILD.detectron2 is not in runfiles", violations[0])

    def test_skips_a_repo_that_declares_no_metadata(self) -> None:
        bare = VendoredRepo("archive", "cosmos_metrics", None, "//third_party/x:BUILD.x", None, None, None)
        self.assertEqual(audit_overlays([bare], {}), [])

    def test_flags_a_declaration_with_no_overlay_to_carry_it(self) -> None:
        no_overlay = VendoredRepo("archive", "detectron2", None, None, "detectron2", None, "0.6")
        violations = audit_overlays([no_overlay], {})
        self.assertEqual(len(violations), 1)
        self.assertIn("no build_file", violations[0])


class CosmosEvalVendoredPythonTest(unittest.TestCase):
    """Audits the vendored Python repos this module declares."""

    # Repos that legitimately ship no Python metadata, keyed to the reason. Every entry keeps an unattributable library
    # in a shipped image, so it needs a reason that says what has to change before it can be dropped.
    _EXEMPTIONS: ClassVar[dict[str, str]] = {}

    def setUp(self) -> None:
        facts = get_runfiles_path(_FACTS_FILE)
        assert facts is not None, f"Could not locate {_FACTS_FILE} in runfiles"
        self.repos = parse_facts(Path(facts).read_text(encoding="utf-8"))

    def test_declares_the_repos_this_module_vendors(self) -> None:
        self.assertEqual(
            sorted(repo.name for repo in self.repos),
            ["cosmos_drive_dreams_toolkits", "detectron2", "opencv_python_headless", "perception_models"],
        )

    def test_every_vendored_repo_ships_metadata_matching_its_declaration(self) -> None:
        dist_infos = {}
        for repo in self.repos:
            location = dist_info_dir(repo)
            if location is None:
                continue
            # source_repo="" is the main module's repo mapping, which is this module's: these tests run from the
            # workspace root, so the apparent repo names below are the ones its own MODULE.bazel use_repo'd.
            resolved = get_runfiles_path(f"{location}/METADATA", external_repo=repo.name, source_repo="")
            dist_infos[repo.name] = read_dist_info(Path(resolved).parent) if resolved else None

        self.assertEqual(audit_metadata(self.repos, dist_infos, self._EXEMPTIONS), [])

    def test_every_overlay_library_carries_the_metadata(self) -> None:
        overlays = {}
        for repo in self.repos:
            path = overlay_path(repo)
            resolved = get_runfiles_path(path) if path else None
            overlays[repo.name] = Path(resolved).read_text(encoding="utf-8") if resolved else None

        self.assertEqual(audit_overlays(self.repos, overlays), [])


if __name__ == "__main__":
    unittest.main()
