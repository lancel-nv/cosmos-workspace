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

"""Tests for services.apt_pins, plus the live audit of this module's apt manifests."""

from pathlib import Path
import unittest

from services.apt_pins import audit_manifests, declared_manifest_labels, manifest_label, parse_floor, snapshot_pins
from utils.bazel import get_runfiles_path

# The rules_distroless apt manifests audited below, which must match what MODULE.bazel declares. Both this tuple and the
# target's data attribute have to list a new service's manifest, so the coverage test compares them against the labels
# the //:apt_pin_facts genrule reads back out of MODULE.bazel.
_MANIFESTS = (
    "services/attribute_verification/packages.yaml",
    "services/objects/single_view/packages.yaml",
    "services/physical_plausibility/packages.yaml",
)

# Both emitted by //:apt_pin_facts.
_FLOOR_FILE = "ubuntu_base_snapshot_floor.txt"
_LABELS_FILE = "apt_manifest_labels.txt"


def _read(relative_path: str) -> str:
    resolved = get_runfiles_path(relative_path)
    assert resolved is not None, f"Could not locate {relative_path} in runfiles"
    return Path(resolved).read_text(encoding="utf-8")


class SnapshotPinsTest(unittest.TestCase):
    """Unit tests for the manifest parsing helpers."""

    def test_snapshot_pins_collects_every_channel(self) -> None:
        manifest = """
sources:
  - channel: jammy main universe
    url: https://snapshot.ubuntu.com/ubuntu/20260818T000000Z
  - channel: jammy-updates main universe
    url: https://snapshot.ubuntu.com/ubuntu/20260818T000000Z
"""
        self.assertEqual(snapshot_pins(manifest), {"20260818T000000Z"})

    def test_snapshot_pins_reports_disagreeing_channels(self) -> None:
        manifest = """
  - url: https://snapshot.ubuntu.com/ubuntu/20260818T000000Z
  - url: https://snapshot.ubuntu.com/ubuntu/20260401T000000Z
"""
        self.assertEqual(snapshot_pins(manifest), {"20260818T000000Z", "20260401T000000Z"})

    def test_snapshot_pins_empty_without_snapshot_urls(self) -> None:
        self.assertEqual(snapshot_pins("  - url: http://archive.ubuntu.com/ubuntu\n"), set())

    def test_parse_floor_accepts_a_genrule_stamp(self) -> None:
        self.assertEqual(parse_floor("20260627T000000Z\n"), "20260627T000000Z")

    def test_parse_floor_rejects_a_bare_date(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected a snapshot timestamp"):
            parse_floor("20260627\n")

    def test_declared_manifest_labels_sorts_and_drops_blank_lines(self) -> None:
        labels = "//services/b:packages.yaml\n\n//services/a:packages.yaml\n"
        self.assertEqual(
            declared_manifest_labels(labels),
            ["//services/a:packages.yaml", "//services/b:packages.yaml"],
        )

    def test_manifest_label_matches_module_bazel_spelling(self) -> None:
        self.assertEqual(
            manifest_label(Path("services/objects/single_view/packages.yaml")),
            "//services/objects/single_view:packages.yaml",
        )


class AuditManifestsTest(unittest.TestCase):
    """Unit tests for the violation messages produced by audit_manifests."""

    def test_clean_manifests_produce_no_violations(self) -> None:
        manifests = {
            Path("a/packages.yaml"): "url: https://snapshot.ubuntu.com/ubuntu/20260818T000000Z",
            Path("b/packages.yaml"): "url: https://snapshot.ubuntu.com/ubuntu/20260818T000000Z",
        }
        self.assertEqual(audit_manifests(manifests, "20260627T000000Z"), [])

    def test_snapshot_older_than_base_image_is_flagged(self) -> None:
        manifests = {Path("a/packages.yaml"): "url: https://snapshot.ubuntu.com/ubuntu/20260401T000000Z"}
        violations = audit_manifests(manifests, "20260627T000000Z")
        self.assertEqual(len(violations), 1)
        self.assertIn("predates the Ubuntu base image build date", violations[0])

    def test_snapshot_equal_to_base_image_date_is_allowed(self) -> None:
        manifests = {Path("a/packages.yaml"): "url: https://snapshot.ubuntu.com/ubuntu/20260627T000000Z"}
        self.assertEqual(audit_manifests(manifests, "20260627T000000Z"), [])

    def test_missing_snapshot_url_is_flagged(self) -> None:
        manifests = {Path("a/packages.yaml"): "url: http://archive.ubuntu.com/ubuntu"}
        violations = audit_manifests(manifests, "20260627T000000Z")
        self.assertEqual(len(violations), 1)
        self.assertIn("no snapshot.ubuntu.com pin found", violations[0])

    def test_channels_disagreeing_within_one_manifest_is_flagged(self) -> None:
        text = (
            "url: https://snapshot.ubuntu.com/ubuntu/20260818T000000Z\n"
            "url: https://snapshot.ubuntu.com/ubuntu/20260728T000000Z\n"
        )
        violations = audit_manifests({Path("a/packages.yaml"): text}, "20260627T000000Z")
        self.assertEqual(len(violations), 1)
        self.assertIn("channels disagree", violations[0])

    def test_manifests_drifting_apart_is_flagged(self) -> None:
        manifests = {
            Path("a/packages.yaml"): "url: https://snapshot.ubuntu.com/ubuntu/20260818T000000Z",
            Path("b/packages.yaml"): "url: https://snapshot.ubuntu.com/ubuntu/20260728T000000Z",
        }
        violations = audit_manifests(manifests, "20260627T000000Z")
        self.assertEqual(len(violations), 1)
        self.assertIn("pin different snapshots", violations[0])


class CosmosEvalAptManifestsTest(unittest.TestCase):
    """Audits the apt manifests this module actually ships."""

    def test_every_apt_install_manifest_is_audited(self) -> None:
        self.assertEqual(
            declared_manifest_labels(_read(_LABELS_FILE)),
            sorted(manifest_label(Path(relative)) for relative in _MANIFESTS),
            "the set of apt.install() manifests in MODULE.bazel and the set audited by this test have diverged; add "
            "the new service's packages.yaml to _MANIFESTS and to this target's data attribute so it is audited too.",
        )

    def test_manifests_pin_a_snapshot_no_older_than_the_base_image(self) -> None:
        floor = parse_floor(_read(_FLOOR_FILE))
        manifests = {Path(relative): _read(relative) for relative in _MANIFESTS}
        self.assertEqual(audit_manifests(manifests, floor), [])


if __name__ == "__main__":
    unittest.main()
