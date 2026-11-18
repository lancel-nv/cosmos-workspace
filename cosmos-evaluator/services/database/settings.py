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

"""Client-agnostic configuration for the Database REST API service.

The base ``Settings`` here carries the transport/auth knobs shared by every Database REST API deployment plus the
``db_config`` PostgreSQL connection mapping. ``db_config`` is a public field, so a standalone/OSS client can run the
service by injecting a fully-resolved mapping directly (e.g. via ``DATABASE_DB_CONFIG`` as JSON) without pulling in any
deployment-specific config-loading machinery.
"""

from functools import lru_cache
import json
import logging
from typing import Annotated, Any

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import NoDecode, SettingsConfigDict

from services.settings_base import SettingsBase
from utils.environment import Environment

logger = logging.getLogger(__name__)


class Settings(SettingsBase):
    """Client-agnostic settings for the Database REST API service."""

    model_config = SettingsConfigDict(
        strict=True, extra="ignore", env_prefix="DATABASE_", case_sensitive=False, frozen=True
    )

    db_env: Environment = Field(
        default=Environment.LOCAL,
        validation_alias=AliasChoices("DATABASE_DB_ENV", "COSMOS_EVALUATOR_ENV", "db_env"),
    )

    # Fully-resolved PostgreSQL connection mapping consumed by the service: ``user`` / ``password`` / ``host`` /
    # ``port`` / ``database``, plus the optional ``connect_timeout`` (default 10s) and ``echo`` (default False) knobs.
    # Injecting it here - e.g. ``DATABASE_DB_CONFIG='{"host":"localhost","port":5432,"database":"staging",
    # "user":"u","password":"p"}'`` (pydantic-settings parses the dict field from env as JSON) - bypasses any
    # ``*_env_var`` indirection or tunneling a subclass would otherwise perform, because the connection bootstrap only
    # runs when this mapping is left empty.
    db_config: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("DATABASE_DB_CONFIG", "db_config"),
    )

    read_key: str = Field(default="")
    admin_key: str = Field(default="")

    # Keys for checker pods coordinating their own job rows. Scoped separately from read/admin because the two
    # existing secrets are shared across every caller and would let any checker write anywhere. Accepts either a
    # JSON array or a comma-separated list so one key can be rotated in alongside another without a restart window.
    # ``NoDecode`` suppresses the JSON decode that pydantic-settings applies to collection-typed fields inside the env
    # source; without it a comma-separated ``DATABASE_WORKER_KEYS`` raises before the validator below ever runs.
    worker_keys: Annotated[list[str], NoDecode] = Field(default_factory=list)

    @field_validator("worker_keys", mode="before")
    @classmethod
    def _parse_worker_keys(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if text.startswith("["):
            return json.loads(text)
        return [key.strip() for key in text.split(",") if key.strip()]

    @model_validator(mode="after")
    def _validate_api_keys_in_production(self) -> "Settings":
        if self.db_env in (Environment.STAGING, Environment.PROD):
            if not self.read_key:
                raise ValueError("DATABASE_READ_KEY is required for staging/production environments")
            if not self.admin_key:
                raise ValueError("DATABASE_ADMIN_KEY is required for staging/production environments")
        return self


@lru_cache
def get_settings() -> Settings:
    """Gets the cached client-agnostic settings instance."""
    return Settings(_env_file=SettingsBase.get_env_files())
