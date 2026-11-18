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

"""Stamp a PEP 440 local version label into a built wheel, in place.

opencv-python derives its version entirely from the pinned sources (find_version.py -> cv2/version.py) and offers no
hook for a local label, so the finished wheel has to be relabeled. Without this the wheel's own METADATA reports the
plain upstream version, and since that is what ``importlib.metadata`` and every SBOM scanner read, our custom
FFmpeg-free build would be indistinguishable from the PyPI build that bundles FFmpeg -- the exact confusion the custom
build exists to avoid.

The same pass replaces both copies of LICENSE-3RD-PARTY.txt, the one in ``cv2/`` and the one in the ``.dist-info``, with
the file that curate_third_party_licenses.py derives for this build.

The wheel is edited entry by entry rather than through ``wheel unpack``/``wheel pack``, which rebuilds the filename from
the WHEEL compatibility tags and would therefore rename the artifact. Everything auditwheel produced -- payload,
timestamps, compatibility tags, filename -- is preserved; only the version stamp and the third-party license file change.

Usage (the env var keeps bytecode out of the directory the script sits in, which is a bind-mounted checkout):
    PYTHONDONTWRITEBYTECODE=1 python3 stamp_local_version.py <source> <destination> <current version> \
        <labeled version> <third-party licenses>
"""

import base64
import csv
import hashlib
import io
from pathlib import Path
import sys
import zipfile

_DISTRIBUTION = "opencv_python_headless"
_THIRD_PARTY_LICENSES = "LICENSE-3RD-PARTY.txt"


def third_party_license_entries(dist_info: str) -> set[str]:
    """Return the archive names of the wheel's two copies of the third-party license file."""
    return {f"cv2/{_THIRD_PARTY_LICENSES}", f"{dist_info}/{_THIRD_PARTY_LICENSES}"}


def record_hash(payload: bytes) -> str:
    """Return the RECORD digest of an entry: urlsafe base64 of its sha256, stripped of padding (PEP 427)."""
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode("ascii").rstrip("=")


def stamped_metadata(payload: bytes, current_version: str, labeled_version: str) -> bytes:
    """Return METADATA with its Version header rewritten to the labeled version."""
    lines = payload.decode("utf-8").splitlines(keepends=True)
    expected = f"Version: {current_version}\n"
    matches = [index for index, line in enumerate(lines) if line == expected]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {expected.strip()!r} line in METADATA, found {len(matches)}")
    lines[matches[0]] = f"Version: {labeled_version}\n"
    return "".join(lines).encode("utf-8")


def stamp(
    source: Path, destination: Path, current_version: str, labeled_version: str, third_party_licenses: bytes
) -> None:
    """Write ``source`` to ``destination`` relabeled to ``labeled_version``, with ``third_party_licenses`` swapped in."""
    old_dist_info = f"{_DISTRIBUTION}-{current_version}.dist-info"
    new_dist_info = f"{_DISTRIBUTION}-{labeled_version}.dist-info"
    record_name = f"{new_dist_info}/RECORD"
    license_entries = third_party_license_entries(new_dist_info)
    replaced: set[str] = set()

    with zipfile.ZipFile(source) as archive:
        entries = [(info, archive.read(info.filename)) for info in archive.infolist()]

    records: list[tuple[str, str, str | int]] = []
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as output:
        for info, original_payload in entries:
            name = info.filename
            payload = original_payload
            if name.startswith(f"{old_dist_info}/"):
                name = f"{new_dist_info}{name[len(old_dist_info) :]}"
            if name == f"{new_dist_info}/METADATA":
                payload = stamped_metadata(payload, current_version, labeled_version)
            if name in license_entries:
                payload = third_party_licenses
                replaced.add(name)
            # RECORD lists the digest of every other entry, so it can only be written once the rest are known.
            if name == record_name:
                continue
            stamped = zipfile.ZipInfo(name, date_time=info.date_time)
            stamped.compress_type = info.compress_type
            stamped.external_attr = info.external_attr
            output.writestr(stamped, payload)
            records.append((name, record_hash(payload), len(payload)))

        # RECORD cannot record its own digest, so PEP 427 has it list itself with both fields empty.
        records.append((record_name, "", ""))
        rows = io.StringIO()
        csv.writer(rows, lineterminator="\n").writerows(sorted(records))
        output.writestr(record_name, rows.getvalue())

    if replaced != license_entries:
        raise SystemExit(f"wheel lacks third-party license entries: {sorted(license_entries - replaced)}")


def verify(wheel_path: Path, labeled_version: str, third_party_licenses: bytes) -> None:
    """Fail unless the wheel is stamped as ``stamp`` leaves it.

    The wheel must report the labeled version, carry no unlabeled metadata, ship ``third_party_licenses`` in both
    locations, and have an intact RECORD.
    """
    dist_info = f"{_DISTRIBUTION}-{labeled_version}.dist-info"

    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())
        metadata = archive.read(f"{dist_info}/METADATA").decode("utf-8")
        record = archive.read(f"{dist_info}/RECORD").decode("utf-8")
        stale_licenses = sorted(
            name for name in third_party_license_entries(dist_info) if archive.read(name) != third_party_licenses
        )

    if f"Version: {labeled_version}\n" not in metadata:
        raise SystemExit(f"METADATA does not report version {labeled_version}")
    stale = sorted(name for name in names if ".dist-info" in name and not name.startswith(f"{dist_info}/"))
    if stale:
        raise SystemExit(f"wheel still carries unlabeled .dist-info entries: {stale}")
    if stale_licenses:
        raise SystemExit(f"wheel ships a third-party license file other than the derived one: {stale_licenses}")
    recorded = {row.split(",")[0] for row in record.splitlines() if row}
    if recorded != names:
        raise SystemExit(f"RECORD and the zip disagree on: {sorted(recorded ^ names)}")


def main() -> None:
    """Stamp the wheel named on the command line and verify the result."""
    if len(sys.argv) != 6:
        raise SystemExit(
            f"usage: {Path(sys.argv[0]).name} <source> <destination> <current version> <labeled version> "
            "<third-party licenses>"
        )

    source, destination, current_version, labeled_version, licenses_path = sys.argv[1:6]
    third_party_licenses = Path(licenses_path).read_bytes()
    stamp(Path(source), Path(destination), current_version, labeled_version, third_party_licenses)
    verify(Path(destination), labeled_version, third_party_licenses)
    print(
        f"OK: {destination} reports version {labeled_version}, ships the derived {_THIRD_PARTY_LICENSES}, and RECORD "
        "covers every entry."
    )


if __name__ == "__main__":
    main()
