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

from functools import lru_cache
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict

from services.framework.worker_settings import WorkerDatabaseSettings
from services.settings_base import SettingsBase


class Settings(WorkerDatabaseSettings):
    process_concurrency_limit: int = Field(default=4, ge=1)
    max_batch_size: int = Field(default=4, ge=1)
    max_batch_wait_ms: float = Field(default=10.0, ge=0)
    checkpoint_uri: str | None = Field(
        default=None,
        description="Optional attention bundle URI. None uses the checker YAML configuration.",
    )
    config_dir: str | None = Field(
        default=None,
        description="Directory holding the checker's config files. None lets the processor resolve its own default.",
    )

    model_config = SettingsConfigDict(
        strict=True,
        extra="ignore",
        env_prefix="PHYSICAL_PLAUSIBILITY_",
        case_sensitive=False,
        frozen=True,
    )

    @model_validator(mode="after")
    def validate_batch_capacity(self) -> Self:
        if self.max_batch_size > self.process_concurrency_limit:
            raise ValueError("max_batch_size cannot exceed process_concurrency_limit")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings(_env_file=SettingsBase.get_env_files())
