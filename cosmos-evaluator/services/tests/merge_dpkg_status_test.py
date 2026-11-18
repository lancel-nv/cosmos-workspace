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

"""Unit tests for services.merge_dpkg_status."""

from collections.abc import Mapping, Sequence
import dataclasses
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest

from services.merge_dpkg_status import LockedPackage, merge

# A data-layer member: its header and, for a regular file, its contents.
LayerMember = tuple[tarfile.TarInfo, bytes | None]

_REPO = "test_packages"
_INFO_DIR = "./var/lib/dpkg/info"

_BASEPKG = "Package: basepkg\nStatus: install ok installed\nVersion: 1.0\nArchitecture: amd64"
_UPGRADED_BASE = "Package: upgraded\nStatus: install ok installed\nVersion: 1.0\nArchitecture: amd64"
_UPGRADED = "Package: upgraded\nStatus: install ok installed\nVersion: 2.0\nArchitecture: amd64"
_NEWTOOL = "Package: newtool\nStatus: install ok installed\nVersion: 3.0-1\nArchitecture: amd64"
_LIBNEW = "Package: libnew\nStatus: install ok installed\nVersion: 1:4.2\nArchitecture: amd64\nMulti-Arch: same"

_OVERLAY = (_UPGRADED, _NEWTOOL, _LIBNEW)
_LOCKED = (
    LockedPackage(name="upgraded", version="2.0", key="upgraded_2.0_amd64"),
    LockedPackage(name="newtool", version="3.0-1", key="newtool_3.0-1_amd64"),
    LockedPackage(name="libnew", version="1:4.2", key="libnew_1-4.2_amd64"),
)

_TOOL_BYTES = b"tool binary"
_CONF_BYTES = b"setting = 1\n"
_LIB_BYTES = b"\x7fELF library"


def _dir(name: str) -> LayerMember:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    return info, None


def _file(name: str, data: bytes) -> LayerMember:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    return info, data


def _symlink(name: str, target: str) -> LayerMember:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    return info, None


def _hardlink(name: str, target: str) -> LayerMember:
    info = tarfile.TarInfo(name)
    info.type = tarfile.LNKTYPE
    info.linkname = target
    return info, None


def _layers() -> dict[str, tuple[LayerMember, ...]]:
    """Data layers of the ``_OVERLAY`` packages, keyed by lockfile key."""
    return {
        "upgraded_2.0_amd64": (_dir("./"), _dir("./etc/"), _file("./etc/upgraded.conf", _CONF_BYTES)),
        "newtool_3.0-1_amd64": (
            _dir("./"),
            _dir("./usr/"),
            _dir("./usr/bin/"),
            _file("./usr/bin/tool", _TOOL_BYTES),
            _symlink("./usr/bin/tool-alias", "tool"),
            _hardlink("./usr/bin/tool-hardlink", "./usr/bin/tool"),
        ),
        "libnew_1-4.2_amd64": (
            _dir("./"),
            _dir("./usr/"),
            _dir("./usr/lib/"),
            _file("./usr/lib/libnew.so.1", _LIB_BYTES),
        ),
    }


def _md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _tar_bytes(members: Sequence[LayerMember], compress: bool = False) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz" if compress else "w") as tar:
        for info, data in members:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    return buffer.getvalue()


def _status_text(stanzas: Sequence[str]) -> str:
    return "\n\n".join(stanzas) + "\n"


def _write_blob(blobs: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    (blobs / digest).write_bytes(data)
    return digest


def _write_oci_layout(root: Path, status: str) -> Path:
    """Write a single-layer OCI layout whose layer holds ``status`` as /var/lib/dpkg/status."""
    layout = root / "layout"
    blobs = layout / "blobs" / "sha256"
    blobs.mkdir(parents=True)
    layer_digest = _write_blob(blobs, _tar_bytes([_file("./var/lib/dpkg/status", status.encode())], compress=True))
    manifest = {
        "layers": [{"digest": f"sha256:{layer_digest}", "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip"}]
    }
    manifest_digest = _write_blob(blobs, json.dumps(manifest).encode())
    (layout / "index.json").write_text(json.dumps({"manifests": [{"digest": f"sha256:{manifest_digest}"}]}))
    return layout


def _read_tar(path: Path) -> dict[str, bytes]:
    contents: dict[str, bytes] = {}
    with tarfile.open(path) as tar:
        for member in tar:
            extracted = tar.extractfile(member)
            assert extracted is not None, f"{member.name} is not a regular file"
            contents[member.name] = extracted.read()
    return contents


class MergeDpkgStatusTest(unittest.TestCase):
    """Tests for merge(), driven through synthetic rules_distroless inputs."""

    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self._tmp)

    def _merge(
        self,
        overlay: Sequence[str] = _OVERLAY,
        locked: Sequence[LockedPackage] = _LOCKED,
        layers: Mapping[str, Sequence[LayerMember]] | None = None,
        compress_layers: bool = True,
    ) -> dict[str, bytes]:
        """Run merge() over a base image holding basepkg and upgraded 1.0, and return the output tar's members.

        ``layers`` maps a lockfile key to the members of the data layer stored in that key's repository directory.
        """
        root = Path(tempfile.mkdtemp(dir=self._tmp))
        layout = _write_oci_layout(root, _status_text([_BASEPKG, _UPGRADED_BASE]))
        overlay_tar = root / "dpkg_status.tar"
        overlay_tar.write_bytes(_tar_bytes([_file("./var/lib/dpkg/status", _status_text(overlay).encode())]))
        lock = root / "packages.lock.json"
        lock.write_text(json.dumps({"packages": [dataclasses.asdict(package) for package in locked], "version": 1}))

        data_layers = []
        for key, members in (_layers() if layers is None else layers).items():
            layer = root / f"rules_distroless++apt+{_REPO}_{key}" / "layer.tar.gz"
            layer.parent.mkdir()
            layer.write_bytes(_tar_bytes(members, compress=compress_layers))
            data_layers.append(layer)

        output = root / "dpkg_status_merged.tar"
        merge(layout, overlay_tar, output, _REPO, lock, data_layers)
        return _read_tar(output)

    def test_status_applies_overlay_stanzas_over_base(self) -> None:
        contents = self._merge()
        self.assertEqual(
            contents["./var/lib/dpkg/status"].decode(),
            _status_text([_BASEPKG, _LIBNEW, _NEWTOOL, _UPGRADED]),
        )

    def test_records_files_of_package_missing_from_base(self) -> None:
        contents = self._merge()
        self.assertEqual(
            contents[f"{_INFO_DIR}/newtool.list"].decode(),
            "/.\n/usr\n/usr/bin\n/usr/bin/tool\n/usr/bin/tool-alias\n/usr/bin/tool-hardlink\n",
        )
        self.assertEqual(
            contents[f"{_INFO_DIR}/newtool.md5sums"].decode(),
            f"{_md5(_TOOL_BYTES)}  usr/bin/tool\n{_md5(_TOOL_BYTES)}  usr/bin/tool-hardlink\n",
        )

    def test_replaces_file_lists_of_upgraded_package_only(self) -> None:
        contents = self._merge()
        self.assertEqual(contents[f"{_INFO_DIR}/upgraded.list"].decode(), "/.\n/etc\n/etc/upgraded.conf\n")
        self.assertEqual(
            contents[f"{_INFO_DIR}/upgraded.md5sums"].decode(), f"{_md5(_CONF_BYTES)}  etc/upgraded.conf\n"
        )
        self.assertNotIn(f"{_INFO_DIR}/basepkg.list", contents)
        self.assertNotIn(f"{_INFO_DIR}/basepkg.md5sums", contents)

    def test_multi_arch_same_package_uses_architecture_qualified_names(self) -> None:
        contents = self._merge()
        self.assertEqual(
            contents[f"{_INFO_DIR}/libnew:amd64.list"].decode(), "/.\n/usr\n/usr/lib\n/usr/lib/libnew.so.1\n"
        )
        self.assertEqual(
            contents[f"{_INFO_DIR}/libnew:amd64.md5sums"].decode(), f"{_md5(_LIB_BYTES)}  usr/lib/libnew.so.1\n"
        )
        self.assertNotIn(f"{_INFO_DIR}/libnew.list", contents)

    def test_package_without_regular_files_has_no_md5sums(self) -> None:
        contents = self._merge(
            overlay=[_NEWTOOL],
            locked=[_LOCKED[1]],
            layers={"newtool_3.0-1_amd64": (_dir("./"), _dir("./usr/"), _symlink("./usr/tool-alias", "tool"))},
        )
        self.assertEqual(contents[f"{_INFO_DIR}/newtool.list"].decode(), "/.\n/usr\n/usr/tool-alias\n")
        self.assertNotIn(f"{_INFO_DIR}/newtool.md5sums", contents)

    def test_overlay_package_without_data_layer_raises(self) -> None:
        layers = _layers()
        del layers["libnew_1-4.2_amd64"]
        with self.assertRaisesRegex(ValueError, "no data layer for overlay packages: libnew"):
            self._merge(layers=layers)

    def test_data_layer_matching_no_lockfile_key_raises(self) -> None:
        layers = {**_layers(), "unknown_1.0_amd64": (_dir("./"),)}
        with self.assertRaisesRegex(ValueError, "matches 0 entries"):
            self._merge(layers=layers)

    def test_data_layer_of_package_absent_from_overlay_raises(self) -> None:
        locked = [*_LOCKED, LockedPackage(name="extra", version="1.0", key="extra_1.0_amd64")]
        layers = {**_layers(), "extra_1.0_amd64": (_dir("./"),)}
        with self.assertRaisesRegex(ValueError, "absent from the overlay status: extra"):
            self._merge(locked=locked, layers=layers)

    def test_lockfile_version_differing_from_status_raises(self) -> None:
        locked = [_LOCKED[0], dataclasses.replace(_LOCKED[1], version="3.0-2"), _LOCKED[2]]
        with self.assertRaisesRegex(
            ValueError, r"disagree on package versions: newtool \(status 3.0-1, lockfile 3.0-2\)"
        ):
            self._merge(locked=locked)

    def test_data_layer_without_gzip_compression_raises(self) -> None:
        with self.assertRaises(gzip.BadGzipFile):
            self._merge(compress_layers=False)

    def test_unsafe_member_name_raises(self) -> None:
        for name in ("../etc/passwd", "./usr/../../etc/passwd", "/etc/passwd", "./usr/bin/new\nline"):
            with self.subTest(name=name):
                layers = {**_layers(), "newtool_3.0-1_amd64": (_dir("./"), _file(name, b"x"))}
                with self.assertRaisesRegex(ValueError, "refusing unsafe member name"):
                    self._merge(layers=layers)


if __name__ == "__main__":
    unittest.main()
