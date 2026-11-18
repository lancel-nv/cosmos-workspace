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

"""Local database configuration loader for Cosmos Evaluator.

Reads a section from ``database/config.yaml`` and resolves it into the fully-populated ``db_config`` mapping consumed by
the Database REST API service (``services/database/service.py``): ``host`` / ``port`` / ``database`` / ``user`` /
``password`` plus any passthrough knobs such as ``connect_timeout``. Credentials are never stored in the YAML - the
section names the env vars (``user_env_var`` / ``password_env_var``) that hold them, and this loader resolves those from
the process environment.

This is intentionally local-only: there is no proxy/tunnel handling, because the only section is ``local`` (a direct
localhost connection to the cosmos-eval-local-db container).
"""

import logging
import os
from typing import Any

import yaml

from utils import bazel

logger = logging.getLogger(__name__)

# Runfiles-relative path of the config file (see the ``config_yaml`` filegroup in database/BUILD).
CONFIG_RUNFILE_PATH = "database/config.yaml"


def load_db_config(section: str = "local", config_path: str = CONFIG_RUNFILE_PATH) -> dict[str, Any] | None:
    """Load and resolve a database config section into a connection mapping.

    Args:
        section: The top-level YAML section to load (default: ``"local"``).
        config_path: Runfiles-relative path to the config YAML (default: ``database/config.yaml``).

    Returns:
        A ``db_config`` mapping with ``user`` / ``password`` resolved from the environment, or ``None`` when the file,
        the section, or the required credentials are missing. Returning ``None`` (rather than raising) lets the caller
        fall back to the service's unconfigured "no-DB" mode instead of crashing local startup.
    """
    resolved_path = bazel.get_runfiles_path(config_path)
    if not resolved_path:
        logger.warning("Database config %r not found in runfiles; leaving db_config unset.", config_path)
        return None

    with open(resolved_path, "r") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        logger.warning("Database config %r is not a YAML mapping; leaving db_config unset.", config_path)
        return None

    db_config = config.get(section)
    if not isinstance(db_config, dict):
        logger.warning("Section %r missing from %r; leaving db_config unset.", section, config_path)
        return None

    # Copy so the returned mapping is independent of the parsed YAML, then swap the *_env_var indirection for the
    # resolved credentials the service actually consumes.
    resolved = dict(db_config)
    user_env_var = resolved.pop("user_env_var", None)
    password_env_var = resolved.pop("password_env_var", None)
    user = os.environ.get(user_env_var) if user_env_var else None
    password = os.environ.get(password_env_var) if password_env_var else None
    if not user or not password:
        logger.warning(
            "Database credentials for section %r not set (expected env vars %s / %s); leaving db_config unset.",
            section,
            user_env_var,
            password_env_var,
        )
        return None

    resolved["user"] = user
    resolved["password"] = password
    return resolved
