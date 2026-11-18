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

"""Load the images a container-based test runs into the local Docker daemon from the test's runfiles.

Every image comes from an ``oci_load`` target, either built by Bazel or pulled by digest and cached by Bazel's
downloader, so no test asks the Docker daemon for an image it would have to pull from a registry at run time.
"""

import functools
import logging
import re
import subprocess

from utils.bazel import get_runfiles_path

logger = logging.getLogger(__name__)

_LOADED_IMAGE_PATTERN = re.compile(r"Loaded image:\s*(\S+)")
_RYUK_LOADER = "utils/ryuk_image_load.sh"


def load_image(loader_runfiles_path: str) -> str:
    """Load the image behind an ``oci_load`` script into the local Docker daemon and return its tag.

    The first call in a process also loads Ryuk (see :func:`load_ryuk_image`), because testcontainers starts Ryuk
    before the first container of a session and would otherwise pull it.

    Args:
        loader_runfiles_path: Runfiles-relative path of the ``oci_load`` script, for example
            ``"services/arbitrator/tests/test_image_load.sh"``. The test must list that ``oci_load`` target in its
            ``data``.

    Raises:
        FileNotFoundError: The loader script is not in the test's runfiles.
        RuntimeError: The loader script fails, or its output names no image.
    """
    load_ryuk_image()
    return _run_loader(loader_runfiles_path)


@functools.cache
def load_ryuk_image() -> str:
    """Load Ryuk, the reaper testcontainers runs to remove a session's containers, and return its tag.

    The image is tagged with testcontainers' default Ryuk image name, so testcontainers finds it locally. Later calls
    in the same process return the tag without reloading.
    """
    return _run_loader(_RYUK_LOADER)


def _run_loader(loader_runfiles_path: str) -> str:
    loader = get_runfiles_path(loader_runfiles_path)
    if not loader:
        raise FileNotFoundError(
            f"'{loader_runfiles_path}' is not in the test's runfiles; list its oci_load target in the test's data"
        )
    logger.info("Loading image with %s", loader)
    result = subprocess.run(["bash", loader], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"'{loader_runfiles_path}' exited with status {result.returncode}: {result.stderr}")
    match = _LOADED_IMAGE_PATTERN.search(result.stdout)
    if not match:
        raise RuntimeError(
            f"Could not parse the image tag from the output of '{loader_runfiles_path}': {result.stdout}"
        )
    image_tag = match.group(1)
    logger.info("Loaded image %s", image_tag)
    return image_tag
