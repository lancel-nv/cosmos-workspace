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

from collections.abc import Iterable
import logging
from pathlib import Path
from typing import Any

import yaml

from utils import bazel

logger = logging.getLogger(__name__)


def resolve_check_config(
    config: dict[str, Any] | None,
    default_config: dict[str, Any],
    section_key: str | None = None,
    required_keys: Iterable[str] = (),
) -> dict[str, Any]:
    """Resolve the effective configuration section for a check, applying the shared AV config semantics.

    Every AV checker resolves a caller-supplied config the same way:

    * An omitted (``None``) or empty (``{}``) config resolves to ``default_config``. An empty dict almost always
      means "just use the defaults", not "run with no configuration".
    * A non-empty config **replaces** the defaults wholesale — it is never merged over them — so each run is
      self-describing with no hidden inherited defaults. It must therefore be complete, which is enforced here so
      an incomplete config fails immediately with a clear message instead of surfacing later as a missing-key
      crash or, worse, a silently skipped processing stage.

    Args:
        config: Caller-supplied configuration, or ``None``.
        default_config: The image defaults, shaped identically to ``config``.
        section_key: When set, ``config`` and ``default_config`` are dicts keyed by this section name (e.g.
            ``"av.objects_single_view"``) and the section beneath it is returned. When ``None``, the dicts *are* the
            section.
        required_keys: Keys a non-empty caller-supplied section must define. Keys with a documented inline
            fallback should be omitted; list the ones whose absence would crash or silently disable work.

    Returns:
        The effective configuration section.

    Raises:
        ValueError: If a non-empty ``config`` omits ``section_key``, is not a mapping, or omits a required key.
    """
    label = section_key or "config"
    if not config:
        section = default_config if section_key is None else default_config.get(section_key, {})
        return dict(section)

    if section_key is None:
        section = config
    else:
        if section_key not in config:
            raise ValueError(
                f"Configuration is missing the required '{section_key}' section. A non-empty config replaces the "
                f"defaults instead of being merged over them, so it must be complete. Omit the config entirely "
                f"(or send an empty {{}}) to use the defaults."
            )
        section = config[section_key]

    if not isinstance(section, dict):
        # ValueError (not TypeError) is deliberate: this is caller-supplied request data, and the service layer maps
        # ValueError to HTTP 422. Raising TypeError here would surface a malformed config as a 500 instead.
        raise ValueError(  # noqa: TRY004
            f"Configuration section '{label}' must be a mapping, got {type(section).__name__}."
        )

    missing = [key for key in required_keys if key not in section]
    if missing:
        raise ValueError(
            f"Configuration section '{label}' is missing required key(s): {', '.join(sorted(missing))}. A non-empty "
            f"config replaces the defaults instead of being merged over them, so it must be complete. Omit the "
            f"config entirely (or send an empty {{}}) to use the defaults."
        )
    return dict(section)


class ConfigManager:
    """
    A configuration manager that loads and manages YAML configuration files.

    This class provides easy access to configuration parameters from YAML files.
    """

    def __init__(self, config_dir: str | None = None) -> None:
        """
        Initialize the config manager.

        Args:
            config_dir: Path to the configuration directory. If None, each config is resolved by name from Bazel
                       runfiles at load time (see ``_resolve_config_path``), so no module-specific config filename is
                       hardcoded here and the manager works standalone in any module.
        """
        self._config_dir: Path | None = Path(config_dir) if config_dir else None
        self._configs: dict[str, Any] = {}

    def _resolve_config_path(self, config_name: str) -> Path:
        """
        Resolve the path to ``<config_name>.yaml``.

        If an explicit ``config_dir`` was provided, look there. Otherwise resolve ``checks/<config_name>.yaml``
        from Bazel runfiles: the two-stage runfiles lookup finds the file in whichever module bundles it (the
        consuming root module or the cosmos_eval module itself), so no module-specific filename is hardcoded.

        Raises:
            FileNotFoundError: If the config file cannot be located in runfiles.
        """
        if self._config_dir is not None:
            return self._config_dir / f"{config_name}.yaml"

        runfiles_path = bazel.get_runfiles_path(f"checks/{config_name}.yaml")
        if not runfiles_path:
            raise FileNotFoundError(f"Configuration file not found: checks/{config_name}.yaml")
        return Path(runfiles_path)

    def load_config(self, config_name: str) -> dict[str, Any]:
        """
        Load a configuration file by name.

        Args:
            config_name: Name of the config file (without .yaml extension)

        Returns:
            Dictionary containing the configuration data

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
        """
        # Validate config_name to prevent path traversal
        if "/" in config_name or "\\" in config_name or ".." in config_name:
            raise ValueError(f"Invalid config name: {config_name}")

        if config_name in self._configs:
            return self._configs[config_name]

        config_path = self._resolve_config_path(config_name)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            self._configs[config_name] = config_data
            return config_data

        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {config_path}: {e}") from e

    def reload_config(self, config_name: str) -> dict[str, Any]:
        """
        Force reload a configuration file, even if it was previously loaded.

        Args:
            config_name: Name of the config file to reload

        Returns:
            Dictionary containing the configuration data
        """
        if config_name in self._configs:
            del self._configs[config_name]

        return self.load_config(config_name)

    def list_configs(self) -> list[str]:
        """
        List all available configuration files.

        Only meaningful when an explicit ``config_dir`` was provided; without one, configs are resolved by name from
        runfiles and there is no single directory to enumerate, so an empty list is returned.

        Returns:
            List of configuration file names (without .yaml extension)
        """
        if self._config_dir is None:
            return []
        return [file_path.stem for file_path in self._config_dir.glob("*.yaml")]
