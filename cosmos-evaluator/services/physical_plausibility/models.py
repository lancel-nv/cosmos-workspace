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

"""Physical-plausibility service request models."""

from typing import Any

from pydantic import BaseModel, Field, field_validator


class PhysicalPlausibilityRequestBase(BaseModel):
    clip_id: str = Field(..., description="Identifier for the clip being processed (echoed in the result).")
    config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Per-request processing overrides. bad_threshold is the only field consumed by this checker; unrelated "
            "fields are accepted for compatibility with shared Arbitrator configuration. Model checkpoint and device "
            "settings are fixed when the long-lived service starts."
        ),
    )
    output_storage_prefix: str | None = Field(
        default=None,
        description=(
            "Optional caller-supplied prefix for output storage keys. Accepted for parity with other checkers; "
            "physical plausibility produces only metrics, so it is currently a no-op."
        ),
    )

    @field_validator("config")
    @classmethod
    def validate_request_config(cls, config: dict[str, Any] | None) -> dict[str, Any] | None:
        if config is None:
            return None
        threshold = config.get("bad_threshold")
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
                raise ValueError("bad_threshold must be a number in [0, 1]")
            if not 0 <= float(threshold) <= 1:
                raise ValueError("bad_threshold must be in [0, 1]")
        return config


class PhysicalPlausibilityRequest(PhysicalPlausibilityRequestBase):
    video_path: str = Field(..., description="Path or URI to the video being evaluated")


class PhysicalPlausibilityCloudRequest(PhysicalPlausibilityRequestBase):
    video_url: str = Field(..., description="URL or path to the video being evaluated")
