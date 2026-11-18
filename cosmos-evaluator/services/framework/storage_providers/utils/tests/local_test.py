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

"""Unit tests for the local-URL helpers.

Both LocalStorageProvider.download_from_url and S3StorageProvider.download_from_url
delegate to ``symlink_local_url`` when ``is_local_url`` returns True, so the two
helpers are validated together here.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from services.framework.storage_providers.utils.local import is_local_url, symlink_local_url


class TestIsLocalUrl(unittest.TestCase):
    """Detection rule: file:// URIs and plain (schemeless) paths are local."""

    def test_file_uri_absolute(self) -> None:
        self.assertTrue(is_local_url("file:///abs/path/foo.mp4"))

    def test_file_uri_with_host(self) -> None:
        # file://localhost/abs/path is also a valid file URI.
        self.assertTrue(is_local_url("file://localhost/abs/path/foo.mp4"))

    def test_absolute_path(self) -> None:
        self.assertTrue(is_local_url("/abs/path/foo.mp4"))

    def test_relative_path(self) -> None:
        self.assertTrue(is_local_url("rel/path/foo.mp4"))
        self.assertTrue(is_local_url("./foo.mp4"))
        self.assertTrue(is_local_url("foo.mp4"))

    def test_s3_url_is_not_local(self) -> None:
        self.assertFalse(is_local_url("s3://bucket/key"))

    def test_https_url_is_not_local(self) -> None:
        self.assertFalse(is_local_url("https://example.com/foo.mp4"))

    def test_http_url_is_not_local(self) -> None:
        self.assertFalse(is_local_url("http://example.com/foo.mp4"))

    def test_gs_url_is_not_local(self) -> None:
        self.assertFalse(is_local_url("gs://bucket/key"))


class TestSymlinkLocalUrl(unittest.TestCase):
    """Materialization: symlink ``url`` to ``destination``, never copy."""

    def test_symlinks_absolute_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "src.bin"
            src.write_bytes(b"hello")
            dest = tmp_path / "out" / "dest.bin"

            result = symlink_local_url(str(src), dest)

            self.assertEqual(result, dest)
            self.assertTrue(dest.is_symlink())
            self.assertEqual(dest.read_bytes(), b"hello")
            self.assertEqual(os.readlink(dest), str(src.resolve()))

    def test_symlinks_file_uri(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "src.bin"
            src.write_bytes(b"file-uri")
            dest = tmp_path / "dest.bin"

            symlink_local_url(f"file://{src}", dest)

            self.assertTrue(dest.is_symlink())
            self.assertEqual(dest.read_bytes(), b"file-uri")

    def test_symlinks_file_uri_with_localhost_host(self) -> None:
        # ``file://localhost/abs/path`` is RFC 8089-equivalent to
        # ``file:///abs/path``; both must resolve to the same source.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "src.bin"
            src.write_bytes(b"file-uri-localhost")
            dest_no_host = tmp_path / "out_no_host" / "dest.bin"
            dest_localhost = tmp_path / "out_localhost" / "dest.bin"

            symlink_local_url(f"file://{src}", dest_no_host)
            symlink_local_url(f"file://localhost{src}", dest_localhost)

            self.assertTrue(dest_no_host.is_symlink())
            self.assertTrue(dest_localhost.is_symlink())
            self.assertEqual(os.readlink(dest_no_host), os.readlink(dest_localhost))
            self.assertEqual(dest_localhost.read_bytes(), b"file-uri-localhost")

    def test_creates_destination_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "src.bin"
            src.write_bytes(b"x")
            dest = tmp_path / "deeply" / "nested" / "dir" / "dest.bin"

            symlink_local_url(str(src), dest)

            self.assertTrue(dest.parent.is_dir())
            self.assertTrue(dest.is_symlink())

    def test_replaces_existing_destination_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            old_src = tmp_path / "old.bin"
            old_src.write_bytes(b"old")
            new_src = tmp_path / "new.bin"
            new_src.write_bytes(b"new")
            dest = tmp_path / "dest.bin"

            symlink_local_url(str(old_src), dest)
            self.assertEqual(dest.read_bytes(), b"old")

            symlink_local_url(str(new_src), dest)
            self.assertEqual(dest.read_bytes(), b"new")
            self.assertEqual(os.readlink(dest), str(new_src.resolve()))

    def test_replaces_existing_destination_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "src.bin"
            src.write_bytes(b"src")
            dest = tmp_path / "dest.bin"
            dest.write_bytes(b"existing")

            symlink_local_url(str(src), dest)

            self.assertTrue(dest.is_symlink())
            self.assertEqual(dest.read_bytes(), b"src")

    def test_missing_source_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dest = tmp_path / "dest.bin"

            with self.assertRaises(FileNotFoundError):
                symlink_local_url(str(tmp_path / "does-not-exist.bin"), dest)

    def test_missing_source_via_file_uri_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dest = tmp_path / "dest.bin"

            with self.assertRaises(FileNotFoundError):
                symlink_local_url(f"file://{tmp_path}/does-not-exist.bin", dest)


if __name__ == "__main__":
    unittest.main()
