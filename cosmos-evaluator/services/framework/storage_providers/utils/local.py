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

"""Local-URL detection and materialization helpers.

Used by both :class:`LocalStorageProvider` (as the entire body of ``download_from_url``) and :class:`S3StorageProvider`
(as a fast path that runs before any boto3 / aiohttp call). Centralising the rule for "what counts as a local URL"
keeps the two providers in lock-step: a future tweak to the symlink semantics (e.g. switching to ``shutil.copy``) only
needs to land here.

Sync helpers; the only I/O is ``Path.exists``, ``Path.mkdir``, ``Path.unlink`` and ``os.symlink``, all of which are
fast syscalls. The async wrapping in each provider's ``download_from_url`` exists to satisfy the protocol, not because
the work is genuinely async.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_FILE_URI_PREFIX = "file://"


def is_local_url(url: str) -> bool:
    """Returns True iff ``url`` should be served from the local filesystem.

    Recognizes:

    - ``file://`` URIs (explicit opt-in via standard URI scheme).
    - Plain path strings with no URL scheme - absolute (``/foo``) or relative (``rel/foo``). This matches the de-facto
      behavior of :meth:`LocalStorageProvider.download_from_url` before this helper existed (it ``Path(url)``-ed any
      string and existence-checked it).

    Schemes ``s3://``, ``http://``, ``https://``, ``gs://`` etc. are *not* local; callers should fall through to their
    remote download code path.
    """
    if url.startswith(_FILE_URI_PREFIX):
        return True
    return not urlparse(url).scheme


def symlink_local_url(url: str, destination: Path) -> Path:
    """Symlinks ``url`` (a local path or ``file://`` URI) to ``destination``.

    Mirrors the contract of ``StorageProvider.download_from_url``: after this returns, ``destination`` resolves to the
    file referenced by ``url``. The mechanism is a *symlink*, not a copy, matching the historical
    :class:`LocalStorageProvider` behavior - code that mutates the file at ``destination`` will mutate the source.
    Callers that need an independent copy should ``shutil.copy2`` instead.

    If ``destination`` already exists (or is a dangling symlink), it is unlinked first so ``os.symlink`` does not raise
    ``FileExistsError``.

    Args:
        url: Either a ``file:///abs/path`` URI or a plain path string (absolute or relative).
        destination: Target path; its parent directory is created if missing.

    Returns:
        ``destination`` for fluent chaining.

    Raises:
        FileNotFoundError: If the resolved source path does not exist.
    """
    if url.startswith(_FILE_URI_PREFIX):
        # Parse to handle both ``file:///abs/path`` (empty authority) and ``file://localhost/abs/path`` (RFC 8089's
        # ``localhost`` form). A naive ``url[len("file://"):]`` strip would mis-treat the latter as the relative path
        # ``localhost/abs/path``.
        src_str = urlparse(url).path
    else:
        src_str = url
    src = Path(src_str)
    if not src.exists():
        raise FileNotFoundError(f"Local file not found: {url}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    os.symlink(src.resolve(), destination)
    logger.info("Symlinked %s -> %s", url, destination)
    return destination
