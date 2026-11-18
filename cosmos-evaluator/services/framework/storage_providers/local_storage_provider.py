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

"""Local filesystem storage provider implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator
import json
import logging
from pathlib import Path
import shutil
from types import TracebackType
from typing import Any, Self

from services.framework.protocols.storage_provider import StorageUrls
from services.framework.storage_providers.utils.local import symlink_local_url

logger = logging.getLogger(__name__)


class LocalStorageProvider:
    """Local filesystem storage provider.

    Implements the StorageProvider protocol using the local filesystem. All keys are resolved relative to
    ``base_path``.
    """

    def __init__(self, base_path: str) -> None:
        self._base_path = Path(base_path)

    def _full_path(self, key: str) -> Path:
        return self._base_path / key

    def make_key_prefix(self, *parts: str | None) -> str:
        """Joins parts as filesystem path segments using pathlib semantics.

        The first non-empty part is taken as-is so an absolute path (``/tmp/output``) anchors at the filesystem root,
        while a relative path resolves against the CWD when used as a key. Subsequent parts have their leading slashes
        stripped to prevent accidental re-anchoring via pathlib's "absolute RHS wins" rule. The result always ends with
        ``/`` so callers can concatenate filenames.
        """
        non_empty = [p for p in parts if p]
        if not non_empty:
            return ""

        head, *tail = non_empty
        joined = Path(head)
        for segment in tail:
            cleaned = segment.strip("/")
            if cleaned:
                joined = joined / cleaned

        out = str(joined)
        return out if out.endswith("/") else out + "/"

    async def store(self, data: Any, key: str, metadata: dict[str, Any] | None = None) -> str:
        dest = self._full_path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, bytes):
            raw = data
        elif isinstance(data, str):
            raw = data.encode("utf-8")
        else:
            raw = json.dumps(data).encode("utf-8")

        dest.write_bytes(raw)
        logger.info("Stored %d bytes to %s", len(raw), dest)
        return str(dest)

    async def retrieve(self, key: str) -> Any:
        src = self._full_path(key)
        raw = src.read_bytes()
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return raw

    async def delete(self, key: str) -> bool:
        target = self._full_path(key)
        if target.is_file():
            target.unlink()
            return True
        if target.is_dir():
            shutil.rmtree(target)
            return True
        return False

    async def exists(self, key: str) -> bool:
        return self._full_path(key).exists()

    async def list_keys(self, prefix: str = "") -> AsyncIterator[str]:
        search_dir = self._full_path(prefix) if prefix else self._base_path
        if not search_dir.is_dir():
            search_dir = search_dir.parent

        base = str(self._base_path)
        for path in search_dir.rglob("*"):
            if path.is_file():
                rel = str(path.relative_to(base))
                if not prefix or rel.startswith(prefix):
                    yield rel

    async def store_file(
        self,
        file_path: Path,
        key: str,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StorageUrls:
        dest = self._full_path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest)
        logger.info("Copied %s -> %s", file_path, dest)
        return StorageUrls(raw=str(dest), presigned=None)

    async def generate_presigned_url(self, key: str, operation: str = "get_object", expiration: int = 3600) -> str:
        return str(self._full_path(key))

    async def download_from_url(self, url: str, destination: Path) -> Path:
        return symlink_local_url(url, destination)

    async def download_file(self, key: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._full_path(key), destination)
        return destination

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass
