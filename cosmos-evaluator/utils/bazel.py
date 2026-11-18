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

import logging
from os import path

from python.runfiles import runfiles

logger = logging.getLogger(__name__)

"""Bazel utility functions."""


def get_runfiles_path(repo_path: str, external_repo: str = "_main", source_repo: str | None = None) -> str | None:
    """Get the path to a file using Bazel runfiles.

    Resolution happens in two stages so cross-module lookups work identically whether cosmos_eval is the root
    module or is consumed as a ``bazel_dep`` (where its canonical repo name is ``cosmos_eval+`` rather than
    ``_main``):

    1. The explicitly requested repo (``external_repo``, optionally resolved through ``source_repo``'s repo
       mapping). This honors callers that target a specific repo such as the main module (``"_main"``).
    2. If that yields no existing file, fall back to the repository that contains *this helper* (i.e. the
       cosmos_eval module), looked up by its **canonical** name via ``CurrentRepository()``. Canonical names
       pass through ``Rlocation`` without repo mapping, which avoids the apparent-name pitfall: an apparent
       name like ``"cosmos_eval"`` is absent from cosmos_eval's own runfiles repo mapping and is only present
       in the *root* module's mapping, so it resolves standalone (root == cosmos_eval) but not when a
       cosmos_eval-owned target runs from a consumer root. Most files looked up here (configs, ``version.txt``,
       ``git_sha``) live alongside cosmos_eval code, so this fallback resolves them in both layouts.

    Args:
        repo_path: Path to the file from the root of the repository (e.g., "checks/cosmos_evaluator.yaml")
        external_repo: Name of the external repository to use without the leading @ (default: "_main"). For
            cross-module lookups, pass the apparent name (e.g., "cosmos_eval") and set ``source_repo`` so the runfiles
            library resolves it through the appropriate repo mapping; if that fails, the stage-2 fallback above
            still resolves files that live in the cosmos_eval module.
        source_repo: Canonical name of the repository whose repo_mapping resolves apparent-to-canonical names in
            ``external_repo``. Pass ``""`` to use the main module's mapping. Leave ``None`` (default) when
            ``external_repo`` is already a canonical name like ``"_main"``, since canonical names pass through any
            mapping unchanged.

    Returns:
        Runtime path of a file in Bazel runfiles if found, None otherwise
    """
    try:
        r = runfiles.Create()

        if not r:
            logger.debug("Runfiles.Create() failed")
            return None

        clean_repo_path = repo_path.lstrip(path.sep)

        # Stage 1: the explicitly requested repo.
        requested = path.join(external_repo, clean_repo_path)
        primary = r.Rlocation(requested, source_repo=source_repo) if source_repo is not None else r.Rlocation(requested)
        if primary is not None and path.exists(primary):
            return primary

        # Stage 2: the repository that owns this helper (resolved by canonical name, so no repo mapping is
        # needed). CurrentRepository() returns "" for the main module; map that to the "_main" runfiles dir.
        own_repo = r.CurrentRepository()
        own_base = own_repo if own_repo else "_main"
        fallback_relative_path = path.join(own_base, clean_repo_path)
        fallback = r.Rlocation(fallback_relative_path)
        if fallback is not None and path.exists(fallback):
            return fallback

        logger.error(
            "Runfiles lookup failed for repo_path=%r (external_repo=%r, source_repo=%r, own_repo=%r)",
            repo_path,
            external_repo,
            source_repo,
            own_repo,
        )
        return None
    except (OSError, TypeError, ValueError) as e:
        logger.error("Exception while getting runfiles path: %s", e)
        return None
