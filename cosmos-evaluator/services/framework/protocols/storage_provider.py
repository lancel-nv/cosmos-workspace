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

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self, runtime_checkable


@dataclass
class StorageUrls:
    """Cloud-agnostic container for storage URLs."""

    raw: str | None = None
    presigned: str | None = None


@runtime_checkable
class StorageProvider(Protocol):
    """Protocol for storage operations."""

    async def store(self, data: Any, key: str, metadata: dict[str, Any] | None = None) -> str:
        """Stores data.

        Args:
            data: Data to store (bytes, string, or serializable object)
            key: Storage key/path
            metadata: Optional metadata to attach

        Returns:
            Storage reference/URL for retrieval
        """
        ...

    async def retrieve(self, key: str) -> Any:
        """Retrieves data.

        Args:
            key: Storage key/path

        Returns:
            Retrieved data
        """
        ...

    async def delete(self, key: str) -> bool:
        """Deletes data.

        Args:
            key: Storage key/path

        Returns:
            True if deletion successful
        """
        ...

    async def exists(self, key: str) -> bool:
        """Checks if data exists.

        Args:
            key: Storage key/path

        Returns:
            True if data exists
        """
        ...

    def list_keys(self, prefix: str = "") -> AsyncIterator[str]:
        """Lists storage keys with optional prefix.

        Args:
            prefix: Optional prefix filter

        Yields:
            Storage keys matching the prefix
        """
        ...

    async def store_file(
        self,
        file_path: Path,
        key: str,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StorageUrls:
        """Stores a file from disk and returns both raw and temporary access URLs.

        Args:
            file_path: Local path to the file to upload
            key: Storage key/path
            content_type: MIME content type (auto-detected if not provided)
            metadata: Optional metadata to attach

        Returns:
            StorageUrls with raw and presigned/temporary access URLs
        """
        ...

    def make_key_prefix(self, *parts: str | None) -> str:
        """Combines path-like parts into a key prefix for this backend.

        Each backend interprets the parts according to its own semantics:

        - Object stores (S3, future Azure/GCS) treat parts as relative key segments. Leading and trailing slashes are
          stripped from every part; empty/None parts are dropped; the surviving parts are joined with ``/`` and the
          result ends with ``/``.

        - Filesystems use pathlib semantics: the first non-empty part is taken as-is so an absolute path anchors at the
          filesystem root, while a relative path is resolved against the process CWD. Subsequent parts have leading
          slashes stripped (preventing accidental re-anchoring) and are joined via ``Path.__truediv__``. The result ends
           with ``/`` for caller convenience so callers can concatenate filenames directly: ``f"{prefix}{filename}"``.

        Returns ``""`` when all parts are empty/``None`` so callers can write to the bucket root or service CWD with no
        prefix.

        Args:
            *parts: Path-like segments, in order from outer-most to inner-most. ``None`` and empty strings are skipped.

        Returns:
            A normalized key prefix (e.g. ``"foo/bar/"`` for object stores or ``"/tmp/output/foo/bar/"`` for an
            absolute filesystem target), or ``""`` if all parts were empty.
        """
        ...

    async def generate_presigned_url(self, key: str, operation: str = "get_object", expiration: int = 3600) -> str:
        """Generates a temporary access URL for a stored object.

        Args:
            key: Storage key/path
            operation: Storage operation (e.g. get_object, put_object)
            expiration: URL expiration time in seconds

        Returns:
            Temporary access URL
        """
        ...

    async def download_from_url(self, url: str, destination: Path) -> Path:
        """Downloads a file from a URL (presigned or raw) to a local path.

        HTTP(S) URLs are downloaded directly, making this compatible with presigned URLs from any cloud provider.

        Args:
            url: HTTP(S) URL or provider-specific URI
            destination: Local file path to write to

        Returns:
            Path to the downloaded file
        """
        ...

    async def download_file(self, key: str, destination: Path) -> Path:
        """Downloads an object by storage key to a local path.

        Unlike ``download_from_url`` this fetches directly through the backend (e.g. an authenticated
        S3 ``get_object``) rather than presigning + HTTP, so it works against backends/endpoints that
        a URL fetch can't reach (e.g. internal S3-compatible clusters behind private addresses).

        Args:
            key: Storage key/path of the object to download.
            destination: Local file path to write to.

        Returns:
            Path to the downloaded file.
        """
        ...

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        ...
