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
Configuration module for the Objects service.

This module provides environment-specific configuration management.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from services.framework.worker_settings import WorkerDatabaseSettings
from services.settings_base import SettingsBase


class Settings(WorkerDatabaseSettings):
    """Settings for the Objects service."""

    process_concurrency_limit: int = Field(default=3, ge=1)

    # Process-level device for model inference. This is authoritative for the REST/lifespan path: the
    # service loads/runs CWIP and SegFormer on this device, and lifespan fails fast when it is "cuda"
    # but the GPU stack is unavailable (see rest_api_common._verify_model_devices). Set
    # OBJECTS_MODEL_DEVICE=cpu for CPU-only dev deployments.
    model_device: Literal["cuda", "cpu"] = Field(default="cuda", description="Device for model inference")

    model_config = SettingsConfigDict(
        strict=True, extra="ignore", env_prefix="OBJECTS_", case_sensitive=False, frozen=True
    )


@lru_cache
def get_settings() -> Settings:
    """Get the settings instance."""
    return Settings(_env_file=SettingsBase.get_env_files())
