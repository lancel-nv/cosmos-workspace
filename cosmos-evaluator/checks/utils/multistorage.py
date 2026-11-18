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

"""
Module for setting up the multistorage client configuration.
"""

import json
import logging
import os
import tempfile

import multistorageclient as msc

logger = logging.getLogger(__name__)
DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024


def setup_msc_config(config: str | None = None) -> None:
    """
    Write MSC config to a temp file and set MSC_CONFIG env var.

    This allows msc.open(), msc.is_file(), etc. to work directly with
    any URL (s3://, msc://, local paths) using path_mapping from config.
    """
    if config is None or config == "":
        logger.warning("Multistorage client configuration is not set")
        return

    try:
        config = json.loads(config)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in multistorage config: %s", e)
        raise ValueError(f"Invalid multistorage configuration: {e}") from e

    # Write to a persistent temp file (not auto-deleted)
    fd, config_path = tempfile.mkstemp(suffix=".json", prefix="msc_config_")
    os.close(fd)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    os.environ["MSC_CONFIG"] = config_path
    logger.debug("MSC config written to: %s", config_path)


def is_remote_path(path: str) -> bool:
    """
    Check if the path is a remote path.
    """
    return path.startswith(("s3://", "gs://", "azure://", "msc://", "https://"))


def download_if_remote(path: str) -> str:
    """
    Download video to temporary file if it's a remote path.

    Args:
        path: Video path (local or remote)

    Returns:
        Local path to video file
    """
    # Check if path is remote (starts with s3://, gs://, etc.)
    if is_remote_path(path):
        logger.debug("Downloading remote video: %s", path)
        # Create a temp file that outlives this function (the caller owns/cleans it up), so a context
        # manager is not appropriate here; mkstemp gives us a path without leaving an open handle.
        suffix = os.path.splitext(path)[1] or ".mp4"
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

        # Download using multistorageclient. Track success so a failed download doesn't leave the pre-created
        # temp file behind; the original exception still propagates through the finally block.
        download_ok = False
        try:
            with msc.open(path, "rb") as src, open(temp_path, "wb") as dst:
                while True:
                    chunk = src.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    dst.write(chunk)
            download_ok = True
        finally:
            if not download_ok and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    logger.warning("Failed to remove temp file after download error: %s", temp_path)

        logger.debug("Downloaded to: %s", temp_path)
        return temp_path
    else:
        logger.debug("Path is local: %s", path)
        return path


def validate_uri(path: str, is_file: bool = True) -> bool:
    """
    Validate if the path is a valid URI.
    """
    try:
        if is_file:
            return msc.is_file(path)
        else:
            return not msc.is_empty(path)
    except Exception:
        logger.exception("Error validating URI %s", path)
        return False
