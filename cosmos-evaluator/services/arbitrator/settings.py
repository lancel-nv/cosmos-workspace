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
Configuration module for the client-agnostic Arbitrator service.

This module resolves checker endpoints from a required ``COSMOS_EVALUATOR_CHECKER_ENDPOINTS`` JSON environment variable,
resolves the optional Database REST API endpoint from the environment, and builds its checker dependency graph from
``checks/cosmos_evaluator.yaml``.
"""

from collections.abc import Mapping
from functools import lru_cache
import json
import logging
from types import MappingProxyType
from urllib.parse import urlsplit

from pydantic import AliasChoices, Field, PrivateAttr
from pydantic_settings import SettingsConfigDict
import yaml

from services.arbitrator.dependency_graph import DependencyGraph
from services.settings_base import SettingsBase
from services.utils import get_runfiles_path
from utils.environment import Environment

logger = logging.getLogger(__name__)


def _redact_url_for_log(url: str) -> str:
    """Reduce ``url`` to ``scheme://host[:port]/path`` for logging, stripping any userinfo and query string."""
    split = urlsplit(url)
    netloc = split.hostname or ""
    try:
        port = split.port
    except ValueError:
        # A malformed/out-of-range port must never break settings init from this logging path; omit it.
        port = None
    if port:
        netloc = f"{netloc}:{port}"
    return f"{split.scheme}://{netloc}{split.path}"


class Settings(SettingsBase):
    """Settings for the client-agnostic Arbitrator service."""

    model_config = SettingsConfigDict(
        strict=True, extra="ignore", env_prefix="ARBITRATOR_", case_sensitive=False, frozen=True, populate_by_name=True
    )

    db_env: Environment = Field(
        default=Environment.LOCAL,
        validation_alias=AliasChoices("COSMOS_EVALUATOR_ENV"),
    )

    database_api_key: str = Field(
        default="",
        description="API key for the Database REST API service. Set via ARBITRATOR_DATABASE_API_KEY.",
    )

    queue_capacity: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional cap on the number of admitted requests the WaitingRoom FIFO will hold concurrently (queued + "
            "in-flight). When unset (the default), the queue grows unbounded. Set via ARBITRATOR_QUEUE_CAPACITY."
        ),
    )

    checker_endpoints_env: str | None = Field(
        default=None,
        validation_alias=AliasChoices("COSMOS_EVALUATOR_CHECKER_ENDPOINTS"),
        description=(
            "Required JSON object string mapping checker base name -> endpoint URL, e.g. "
            '\'{"av.objects_single_view": "http://localhost:8082"}\'. Set via COSMOS_EVALUATOR_CHECKER_ENDPOINTS. '
            "Required on the client-agnostic base; a subclass may relax this."
        ),
    )

    database_api_url_env: str = Field(
        default="",
        validation_alias=AliasChoices("COSMOS_EVALUATOR_DATABASE_API_URL"),
        description=(
            "Optional Database REST API endpoint URL. When empty the arbitrator runs without orchestration recording "
            "(client tracking disabled). Set via COSMOS_EVALUATOR_DATABASE_API_URL."
        ),
    )

    _checker_endpoints: dict[str, str] = PrivateAttr(default_factory=dict)
    _database_api_url: str = PrivateAttr(default="")
    _dependency_graph: DependencyGraph = PrivateAttr()

    def model_post_init(self, context: dict[str, object] | None, /) -> None:
        """Post-initialization hook: resolve endpoints and build the dependency graph."""
        self._checker_endpoints = self._resolve_checker_endpoints()
        self._database_api_url = self._resolve_database_api_url()
        self._dependency_graph = self._load_dependency_graph()

    @property
    def checker_endpoints(self) -> Mapping[str, str]:
        """Read-only view of the resolved checker endpoints (base name -> URL); the frozen settings are not mutable."""
        return MappingProxyType(self._checker_endpoints)

    @property
    def database_api_url(self) -> str:
        """Database REST API endpoint URL (empty string when recording is disabled)."""
        return self._database_api_url

    @property
    def dependency_graph(self) -> DependencyGraph:
        """Dependency graph parsed and validated from the checker config."""
        return self._dependency_graph

    def _resolve_checker_endpoints(self) -> dict[str, str]:
        """Resolve the checker endpoints from the required ``COSMOS_EVALUATOR_CHECKER_ENDPOINTS`` JSON env var.

        Raises:
            ValueError: If the env var is unset, is not a non-empty JSON object, or has a non-string value. A
                subclass may relax the "required" contract by overriding this method.
        """
        if not self.checker_endpoints_env:
            raise ValueError(
                "COSMOS_EVALUATOR_CHECKER_ENDPOINTS is required: set it to a JSON object string mapping checker base "
                'name -> endpoint URL, e.g. \'{"av.objects_single_view": "http://localhost:8082"}\'.'
            )
        try:
            parsed = json.loads(self.checker_endpoints_env)
        except json.JSONDecodeError as exc:
            raise ValueError(f"COSMOS_EVALUATOR_CHECKER_ENDPOINTS must be a JSON object string: {exc}") from exc

        if not isinstance(parsed, dict) or not parsed:
            raise ValueError("COSMOS_EVALUATOR_CHECKER_ENDPOINTS must be a non-empty JSON object")

        for name, url in parsed.items():
            if not isinstance(url, str):
                # ValueError (not TypeError) is the deliberate config-validation contract here, matching every sibling
                # raise in this method and the documented behavior asserted by the unit tests.
                raise ValueError(  # noqa: TRY004
                    f"Checker endpoint '{name}' must be a string URL, got {type(url).__name__}"
                )

        logger.info("Loaded checker endpoints: %s", list(parsed.keys()))
        return dict(parsed)

    def _resolve_database_api_url(self) -> str:
        """Resolve the Database REST API endpoint URL from the environment.

        Returns the configured URL, or an empty string when none is set (recording disabled). A subclass may override
        this to resolve the URL from per-environment configuration instead.
        """
        if self.database_api_url_env:
            logger.info("Loaded database config: api_url=%s", _redact_url_for_log(self.database_api_url_env))
        else:
            logger.warning(
                "No database_api_url configured (COSMOS_EVALUATOR_DATABASE_API_URL unset); client tracking disabled"
            )
        return self.database_api_url_env

    def _dependency_graph_class(self) -> type[DependencyGraph]:
        """Return the dependency-graph class to instantiate. Subclasses return one with extra checker logic."""
        return DependencyGraph

    def _dependency_graph_config(self) -> dict:
        """Load and return the raw checker config dict that backs the dependency graph.

        The modulereads ``checks/cosmos_evaluator.yaml`` from runfiles.

        Raises:
            FileNotFoundError: If the config file cannot be found in runfiles.
            ValueError: If the config is empty or not a mapping.
        """
        config_path = get_runfiles_path("checks/cosmos_evaluator.yaml")
        if not config_path:
            raise FileNotFoundError("cosmos_evaluator.yaml not found in runfiles: checks/cosmos_evaluator.yaml")

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not config or not isinstance(config, dict):
            raise ValueError(f"cosmos_evaluator.yaml must be a non-empty mapping: {config_path}")
        return config

    def _load_dependency_graph(self) -> DependencyGraph:
        """Build and validate the dependency graph from the checker config."""
        config = self._dependency_graph_config()
        graph = self._dependency_graph_class()(config)
        graph.validate()
        logger.info("Loaded dependency graph with %d checkers", len(config))
        return graph


@lru_cache
def get_settings() -> Settings:
    """Gets cached settings instance.

    This function uses lru_cache to ensure only one Settings instance is created throughout the application lifecycle.
    """
    return Settings(_env_file=SettingsBase.get_env_files())
