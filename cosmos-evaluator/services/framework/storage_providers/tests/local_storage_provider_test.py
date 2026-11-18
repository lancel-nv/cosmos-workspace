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

"""Unit tests for LocalStorageProvider."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from services.framework.protocols.storage_provider import StorageProvider
from services.framework.storage_providers.local_storage_provider import LocalStorageProvider


class TestLocalStorageProviderProtocol(unittest.TestCase):
    """Verify LocalStorageProvider satisfies the StorageProvider protocol."""

    def test_satisfies_protocol(self) -> None:
        provider = LocalStorageProvider(base_path="")
        self.assertIsInstance(provider, StorageProvider)


class TestLocalStorageProviderMakeKeyPrefix(unittest.TestCase):
    """Tests for LocalStorageProvider.make_key_prefix.

    The local backend uses pathlib semantics: the first non-empty part is taken as-is so an absolute
    path anchors at filesystem root, while tail parts have leading slashes stripped to prevent
    accidental re-anchoring via pathlib's "absolute RHS wins" rule.
    """

    def setUp(self) -> None:
        self.provider = LocalStorageProvider(base_path="")

    def test_no_parts_returns_empty(self) -> None:
        self.assertEqual(self.provider.make_key_prefix(), "")

    def test_all_none_returns_empty(self) -> None:
        self.assertEqual(self.provider.make_key_prefix(None, None), "")

    def test_all_empty_strings_returns_empty(self) -> None:
        self.assertEqual(self.provider.make_key_prefix("", ""), "")

    def test_absolute_first_part_is_preserved(self) -> None:
        self.assertEqual(
            self.provider.make_key_prefix("/tmp/output", "egomotion", "cam_front"),
            "/tmp/output/egomotion/cam_front/",
        )

    def test_absolute_first_part_with_trailing_slash_normalized(self) -> None:
        self.assertEqual(
            self.provider.make_key_prefix("/tmp/output/", "egomotion", "cam_front"),
            "/tmp/output/egomotion/cam_front/",
        )

    def test_relative_first_part_resolves_against_cwd(self) -> None:
        self.assertEqual(
            self.provider.make_key_prefix("my-team/run-42", "egomotion", "cam_front"),
            "my-team/run-42/egomotion/cam_front/",
        )

    def test_empty_first_part_skipped(self) -> None:
        self.assertEqual(
            self.provider.make_key_prefix("", "egomotion", "cam_front"),
            "egomotion/cam_front/",
        )

    def test_root_only(self) -> None:
        self.assertEqual(self.provider.make_key_prefix("/"), "/")

    def test_none_in_middle_is_skipped(self) -> None:
        self.assertEqual(
            self.provider.make_key_prefix(None, "a", None, "b"),
            "a/b/",
        )

    def test_tail_part_leading_slash_does_not_re_anchor(self) -> None:
        """A leading / on a tail part must not override the first part's anchor."""
        self.assertEqual(
            self.provider.make_key_prefix("/abs", "/should-not-anchor"),
            "/abs/should-not-anchor/",
        )

    def test_tail_part_only_slashes_is_skipped(self) -> None:
        self.assertEqual(
            self.provider.make_key_prefix("foo", "/", "//"),
            "foo/",
        )

    def test_inner_slashes_in_segment_are_preserved(self) -> None:
        """A part may carry multi-level structure; only outer slashes are stripped on tail parts."""
        self.assertEqual(
            self.provider.make_key_prefix("my-team/run-42", "objects_single_view", "camera_front_wide_120fov"),
            "my-team/run-42/objects_single_view/camera_front_wide_120fov/",
        )

    def test_single_relative_segment_gets_trailing_slash(self) -> None:
        self.assertEqual(self.provider.make_key_prefix("foo"), "foo/")

    def test_single_absolute_segment_gets_trailing_slash(self) -> None:
        self.assertEqual(self.provider.make_key_prefix("/foo"), "/foo/")


class TestLocalStorageProviderStoreFile(unittest.IsolatedAsyncioTestCase):
    """Tests that combine make_key_prefix output with store_file to verify destination paths."""

    def setUp(self) -> None:
        self.tmp_root = Path(tempfile.mkdtemp())
        self.src = self.tmp_root / "src.txt"
        self.src.write_text("payload")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    async def test_absolute_prefix_writes_to_absolute_path(self) -> None:
        """make_key_prefix with an absolute first part lands at that absolute path on disk."""
        provider = LocalStorageProvider(base_path="")
        target_root = self.tmp_root / "abs_root"
        prefix = provider.make_key_prefix(str(target_root), "egomotion", "cam_front")

        urls = await provider.store_file(self.src, f"{prefix}foo.json")

        expected = target_root / "egomotion" / "cam_front" / "foo.json"
        self.assertEqual(urls.raw, str(expected))
        self.assertTrue(expected.exists())
        self.assertEqual(expected.read_text(), "payload")

    async def test_relative_prefix_writes_relative_to_base_path(self) -> None:
        """A relative first part lands under the provider's base_path."""
        provider = LocalStorageProvider(base_path=str(self.tmp_root))
        prefix = provider.make_key_prefix("rel-prefix", "egomotion", "cam_front")

        urls = await provider.store_file(self.src, f"{prefix}foo.json")

        expected = self.tmp_root / "rel-prefix" / "egomotion" / "cam_front" / "foo.json"
        self.assertEqual(urls.raw, str(expected))
        self.assertTrue(expected.exists())


if __name__ == "__main__":
    unittest.main()
