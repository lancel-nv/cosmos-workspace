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

"""Utility functions for services."""

from __future__ import annotations

import logging
from pathlib import Path
import zipfile

from utils.bazel import get_runfiles_path

logger = logging.getLogger(__name__)


def add_uuid_to_filename(name: str, uid: str) -> str:
    """Insert ``uid`` before the file extension of ``name``.

    The function preserves multi-dot stems by treating only the final
    extension (``Path.suffix``) as the extension. For example::

        add_uuid_to_filename("clip.dynamic.objects.results.json", "abc")
        # -> "clip.dynamic.objects.results.abc.json"

        add_uuid_to_filename("clip.hallucination.mp4", "abc")
        # -> "clip.hallucination.abc.mp4"

    Args:
        name: The original filename (no directory component expected).
        uid: The unique id to inject before the extension.

    Returns:
        The new filename with ``uid`` inserted before the extension. If the
        input has no extension, ``uid`` is appended (e.g. ``"foo"`` ->
        ``"foo.abc"``).
    """
    p = Path(name)
    suffix = p.suffix
    stem = p.stem
    return f"{stem}.{uid}{suffix}" if suffix else f"{stem}.{uid}"


def extract_clip_id(rds_hq_dir_path: Path) -> str:
    """Extract clip ID from the tar files in the extracted RDS HQ directory.

    An RDS-HQ archive stores one ``<clip_id>.tar`` per layer subdirectory, so a single clip typically appears as
    several tar files that all share the same stem. This function collapses them to the set of distinct clip stems.

    Args:
        rds_hq_dir_path: Path to the extracted RDS HQ directory

    Returns:
        Clip ID (the shared stem of the ``.tar`` files)

    Raises:
        ValueError: If no ``.tar`` files are found, or if the archive contains more than one distinct clip (this path
            expects a single-clip archive and refuses to grade an arbitrary one).
    """
    # Look for .tar files in any subdirectory. Sort for deterministic ordering, since rglob's traversal order is
    # filesystem-dependent and we must not silently depend on it.
    tar_files = sorted(rds_hq_dir_path.rglob("*.tar"))

    if not tar_files:
        raise ValueError(f"No .tar files found in extracted RDS HQ directory: {rds_hq_dir_path}")

    # Collapse to distinct clip stems. Format: 1408ad50-5d9c-463d-a6c9-9e042c7cbcb3_17280960000.tar
    clip_ids = sorted({tar_file.stem for tar_file in tar_files})

    if len(clip_ids) > 1:
        raise ValueError(
            f"Multiple distinct clips found in RDS HQ directory {rds_hq_dir_path}: {clip_ids}. This path expects a "
            f"single-clip archive; provide one clip at a time."
        )

    clip_id = clip_ids[0]
    logger.info("Extracted clip_id: %s", clip_id)
    return clip_id


def extract_rds_hq_zip(zip_path: Path, dest: Path) -> None:
    """Safely extract an RDS-HQ zip archive to ``dest``.

    This is a blocking, CPU+I/O bound operation; callers on an asyncio event loop MUST run it off the loop (e.g. via
    ``asyncio.to_thread``) so lease renewals are not starved during extraction. The member scan rejects absolute paths
    and ``..`` traversal before any bytes are written to guard against zip-slip.

    Args:
        zip_path: Path to the downloaded RDS-HQ ``.zip`` archive.
        dest: Directory to extract into (created by ``extractall`` if missing).

    Raises:
        ValueError: If the archive contains a member that resolves outside ``dest`` (zip-slip / absolute path).
    """
    dest_resolved = dest.resolve()
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for info in zip_ref.infolist():
            # Resolve each member against the destination and require it to stay inside the destination tree.
            # This rejects ``..`` traversal and absolute paths (joining an absolute member resets to that root)
            # more robustly than substring checks.
            target = (dest_resolved / info.filename).resolve()
            if target != dest_resolved and not target.is_relative_to(dest_resolved):
                raise ValueError(f"Zip file contains unsafe path: {info.filename}")
        zip_ref.extractall(dest)


def get_contents_from_file(file_path: str) -> str:
    """Read the contents of a file.

    Args:
        file_path: Path to the file

    Returns:
        Contents of the file

    Raises:
        ValueError: If the file is empty
    """
    with open(file_path, "r") as file:
        contents = file.read().strip()

    if not contents:
        raise ValueError(f"File is empty: {file_path}")

    return contents


def get_contents_from_runfile(repo_path: str) -> str:
    """Read the contents of a file in the Bazel runfiles directory.

    Args:
        repo_path: Path to the file in the repository (e.g. "services/objects/single_view/version.txt")

    Returns:
        Contents of the file

    Raises:
        ValueError: If repo_path is empty
        FileNotFoundError: If the file is not found
    """
    if not repo_path:
        raise ValueError("repo_path must not be empty")
    file_path = get_runfiles_path(repo_path)
    if not file_path:
        raise FileNotFoundError(f"File not found: {repo_path}")
    return get_contents_from_file(file_path)


def get_git_sha() -> str | None:
    """Get the current Git commit SHA.

    Tries multiple sources in order:
    1. Container path (/app/git_sha.txt) - available in Docker containers via git_sha_layer
    2. Bazel runfiles - available when //utils:git_sha is in data deps (local dev, tests)

    Returns:
        Git SHA string, or None if not available (e.g. image_base test images).
    """
    # Container path (available for :rest_api_deployment via git_sha_layer)
    container_path = "/app/git_sha.txt"
    if Path(container_path).exists():
        return get_contents_from_file(container_path).split("=")[1].strip()

    # Bazel runfiles
    runfile_path = get_runfiles_path("utils/git_sha.txt", external_repo="cosmos_eval", source_repo="")
    if not runfile_path:
        return None
    try:
        contents = get_contents_from_file(runfile_path)
    except (FileNotFoundError, ValueError):
        # FileNotFoundError: runfile missing from sandbox.
        # ValueError: get_contents_from_file raises on empty file.
        return None
    parts = contents.split("=", 1)
    if len(parts) != 2:
        return None
    sha = parts[1].strip()
    return sha or None
