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
Arbitrator request/result contract.

The arbitrator's own inbound request shape (``CheckerInputs`` / ``ArbitratorRequest``) and the aggregated result
(``ArbitratorResult``) it returns to clients. Kept free of FastAPI / httpx so this contract can be imported by
lightweight callers (clients, tests, subclasses) without pulling the transport layer.
"""

from typing import Any

from pydantic import BaseModel, Field

CheckerName = str


class CheckerInputs(BaseModel):
    """Inputs for a checker.

    The ``augmented_video_urls`` dict drives camera fan-out. Keys are full RDS camera stream ids (e.g.
    ``camera_front_wide_120fov``); values are the per-camera augmented video URLs. Every per-view checker is fanned out
    into one parameterized job per entry.

    ``world_model_video_urls`` parallels ``augmented_video_urls``: a per-camera dict of rendered world-model (control)
    video URLs, consumed only by the single-view (CWIP) static check. There is no shared fallback, so when single-view
    is requested a camera without an entry has its single-view job skipped (the rest of the request still runs).

    ``original_video_urls`` likewise parallels ``augmented_video_urls``: a per-camera dict of original/reference video
    URLs, consumed only by the ``metropolis.hallucination`` check (its motion baseline). There is no shared fallback,
    so when hallucination is requested a camera without an entry has its hallucination job skipped (the rest of the
    request still runs).
    """

    augmented_video_urls: dict[str, str] = Field(
        ...,
        min_length=1,
        description=(
            "Mapping of full RDS camera stream id -> augmented video URL. Keys drive per-view fan-out. "
            "Example: {'camera_front_wide_120fov': 's3://.../front_wide.mp4', "
            "'camera_cross_right_120fov': 's3://.../cross_right.mp4'}"
        ),
    )
    rds_hq_url: str | None = Field(
        default=None,
        description=(
            "URL or path to the RDS-HQ dataset zip (shared across cameras). Required only when the "
            "``av.objects_single_view`` checker is requested; Metropolis checks (hallucination, "
            "attribute_verification) do not consume it, so it may be omitted for Metropolis-only requests."
        ),
    )
    world_model_video_urls: dict[str, str] | None = Field(
        default=None,
        description=(
            "Mapping of full RDS camera stream id -> per-camera rendered world-model (control) video URL. "
            "Parallels ``augmented_video_urls``: the authoritative per-camera source for the single-view static "
            "(CWIP) check. Optional overall; there is no shared fallback (a top-down BEV would collapse static scores "
            "to ~0), so when single-view is requested a camera without an entry has its single-view job skipped (the "
            "rest of the request still runs). "
            "Example: {'camera_front_wide_120fov': 's3://.../camera_front_wide_120fov_..._0.mp4'}"
        ),
    )
    original_video_urls: dict[str, str] | None = Field(
        default=None,
        description=(
            "Mapping of full RDS camera stream id -> per-camera original/reference video URL. Parallels "
            "``augmented_video_urls``: the authoritative per-camera source for the ``metropolis.hallucination`` "
            "check's motion baseline. Optional overall; there is no shared fallback, so when hallucination is "
            "requested a camera without an entry has its hallucination job skipped (the rest of the request still "
            "runs). "
            "Example: {'camera_front_wide_120fov': 's3://.../front_wide_original.mp4'}"
        ),
    )

    config: dict[str, Any] | None = Field(default=None, description="Configuration dictionary for processing")


class ArbitratorRequest(BaseModel):
    """Request model for the client-agnostic Arbitrator service."""

    inputs: CheckerInputs = Field(..., description="Inputs for the checks")

    checks: list[CheckerName] = Field(default=[], description="List of checkers to execute")
    output_storage_prefix: str | None = Field(
        default=None,
        description=(
            "Optional caller-supplied prefix for checker output storage keys (e.g. 'my-team/run-42/'). Passed through "
            "unchanged to each individual checker; each checker then appends its own "
            "'<checker_name>/<camera_name>/' (and a per-invocation uuid) to guarantee per-camera uniqueness within "
            "this request. Cross-request isolation (e.g. keeping history of multiple runs of the same clip) is the "
            "caller's responsibility: pass a unique prefix per run/experiment/analysis. If omitted or empty, checker "
            "outputs land at the bucket root with no caller-specific grouping."
        ),
    )


class ArbitratorResult(BaseModel):
    """Result model for the client-agnostic Arbitrator service."""

    checker_results: dict[CheckerName, Any] = Field(default_factory=dict, description="Results of the checks")
