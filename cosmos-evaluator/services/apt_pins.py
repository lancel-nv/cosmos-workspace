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

"""Audit the snapshot.ubuntu.com pins in the services' rules_distroless apt manifests.

An apt layer exists to *upgrade* packages over the ones baked into the base image. When its snapshot pin predates the
base image's own build date, the overlay ships **older** packages than the base already had, and because the overlay's
.deb files and dpkg stanzas win at image-assembly time, that regression is what actually runs.

The checks here are deliberately metadata-only. Comparing resolved versions against the base image would mean pulling
multi-gigabyte OCI layouts into a test's runfiles, so instead this pins the invariants that make such a regression
impossible in the first place: one snapshot per manifest, agreement across a module's manifests, and a floor at the
base image's build date. The floor and the list of manifests a module declares both come from the ``//:apt_pin_facts``
genrule, which extracts them from MODULE.bazel in the package that owns it.
"""

from __future__ import annotations

from pathlib import Path
import re

# The snapshot service URL as written in the manifests' `sources[].url` fields, e.g.
# https://snapshot.ubuntu.com/ubuntu/20260818T000000Z
_SNAPSHOT_URL_RE = re.compile(r"https://snapshot\.ubuntu\.com/ubuntu/(\d{8}T\d{6}Z)")

# A snapshot timestamp is ordered lexicographically (YYYYMMDDThhmmssZ), so plain string comparison is chronological.
SnapshotStamp = str


def snapshot_pins(manifest_text: str) -> set[SnapshotStamp]:
    """Return the distinct snapshot timestamps a single apt manifest pins across its channels."""
    return set(_SNAPSHOT_URL_RE.findall(manifest_text))


def parse_floor(floor_text: str) -> SnapshotStamp:
    """Validate and return the snapshot floor emitted by the ``apt_pin_facts`` genrule."""
    floor = floor_text.strip()
    if not re.fullmatch(r"\d{8}T\d{6}Z", floor):
        msg = f"expected a snapshot timestamp like 20260627T000000Z from the base image floor genrule, got {floor!r}"
        raise ValueError(msg)
    return floor


def declared_manifest_labels(labels_text: str) -> list[str]:
    """Return the sorted apt.install() manifest labels emitted by the ``apt_pin_facts`` genrule."""
    return sorted(line.strip() for line in labels_text.splitlines() if line.strip())


def manifest_label(manifest_path: Path) -> str:
    """Return the Bazel label of a manifest given its module-relative path, matching MODULE.bazel's spelling."""
    return f"//{manifest_path.parent.as_posix()}:{manifest_path.name}"


def audit_manifests(manifests: dict[Path, str], floor: SnapshotStamp) -> list[str]:
    """Check a module's apt manifests against the pin invariants, returning one message per violation.

    Enforces that every manifest pins exactly one snapshot, that all manifests in the module agree on it, and that it
    is no older than ``floor``. Returning messages rather than raising lets a caller report every offending manifest in
    one failure instead of stopping at the first.
    """
    violations: list[str] = []
    pins_by_manifest: dict[Path, SnapshotStamp] = {}

    for path, text in sorted(manifests.items()):
        pins = snapshot_pins(text)
        if not pins:
            violations.append(
                f"{path}: no snapshot.ubuntu.com pin found. Reproducible builds require snapshot URLs; if the source "
                f"scheme changed deliberately, update _SNAPSHOT_URL_RE in apt_pins.py to match."
            )
            continue
        if len(pins) > 1:
            violations.append(
                f"{path}: channels disagree on the snapshot timestamp ({', '.join(sorted(pins))}). Pin every channel "
                f"to the same snapshot so the resolved set is internally consistent."
            )
            continue
        pins_by_manifest[path] = pins.pop()

    for path, pin in pins_by_manifest.items():
        if pin < floor:
            violations.append(
                f"{path}: snapshot {pin} predates the Ubuntu base image build date {floor}, so this apt layer can "
                f"resolve packages to versions OLDER than the base image already ships, and the overlay wins at image "
                f"assembly. Bump the snapshot to {floor} or later and regenerate the lockfile."
            )

    distinct = set(pins_by_manifest.values())
    if len(distinct) > 1:
        listed = ", ".join(f"{path} -> {pin}" for path, pin in sorted(pins_by_manifest.items()))
        violations.append(
            f"apt manifests in this module pin different snapshots ({listed}). Keep them on a single snapshot so no "
            f"service silently lags behind on security updates."
        )

    return violations
