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
Objects Request and Response Models.
"""

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_serializer


# Base request model
class ObjectsRequestBase(BaseModel):
    """Base request model for object processing."""

    # Required fields
    camera_name: str = Field(..., description="Camera name for processing")

    config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Full processing configuration. A non-empty config REPLACES the image defaults (it is not merged), so it "
            "must be complete — each run is self-describing with no assumed defaults. Omitting it, or sending null or "
            "an empty {}, is treated identically and uses the image defaults; GET /config returns them as a starting "
            "point."
        ),
    )
    trial_frames: int | None = Field(default=None, description="Limit number of frames for testing")
    verbose: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO", description="Logging level")

    # Output storage
    output_storage_prefix: str | None = Field(
        default=None,
        description=(
            "Optional caller-supplied prefix for output storage keys "
            "(e.g., 'my-team/run-42/'). The service appends "
            "'objects_single_view/<camera_name>/' and a per-invocation "
            "uuid to guarantee per-camera uniqueness within a single "
            "request. Cross-request isolation (e.g. preserving history of "
            "multiple runs of the same clip) is the caller's "
            "responsibility: pass a unique prefix per "
            "run/experiment/analysis. If omitted or empty, no caller "
            "prefix is applied and uploads land at "
            "'objects_single_view/<camera_name>/...' at the bucket root."
        ),
    )


# Request model that accepts local paths
class ObjectsRequest(ObjectsRequestBase):
    """Request model for object processing."""

    # Required fields
    clip_id: str = Field(..., description="Clip ID to process")
    input_data_path: str = Field(..., description="Path to input data directory")
    video_path: str = Field(..., description="Path to video file")
    world_video_path: str | None = Field(
        default=None,
        description="Path to the world video file for two-stream static processing. Optional: omitting it runs the "
        "static check camera-only.",
    )

    # Device is local-only: it is set by the local CLI/processor callers and, on the REST path, populated
    # from Settings.model_device by the transport layer. It is intentionally absent from
    # ObjectsCloudRequest so REST callers cannot pick the device - the deployment does.
    model_device: Literal["cuda", "cpu"] = Field(default="cuda", description="Device to run models on")


# Request model that accepts cloud storage URLs or local file paths
class ObjectsCloudRequest(ObjectsRequestBase):
    """Request model for object processing with cloud storage URLs or local file paths."""

    # Required fields
    rds_hq_url: str = Field(..., description="Presigned URL of RDS HQ zip file")
    augmented_video_url: str = Field(..., description="URL or path to the augmented video MP4 file")
    world_model_video_url: str | None = Field(
        default=None,
        description="URL or path to the world model video MP4 file for two-stream static processing. Optional: "
        "omitting it runs the static check camera-only.",
    )


class ObjectsResult(BaseModel):
    """Result model for local object processing."""

    # Processing statistics
    processed_frames: int = Field(..., description="Number of frames processed")
    total_video_frames: int = Field(..., description="Total frames in video")

    # Score statistics
    mean_score: float = Field(..., description="Mean correspondence score")
    std_score: float = Field(..., description="Standard deviation of scores")
    min_score: float = Field(..., description="Minimum correspondence score")
    max_score: float = Field(..., description="Maximum correspondence score")

    # Track information
    unique_track_ids: list[int] = Field(..., description="List of unique track IDs found")
    processed_frame_ids: list[int] = Field(..., description="List of frame IDs that were processed")

    # Configuration summary
    config_summary: dict[str, Any] = Field(..., description="Summary of processing configuration")

    # Paths and metadata
    clip_id: str = Field(..., description="Processed clip ID")
    output_dir: str = Field(..., description="Output directory used")

    # Configure JSON serialization to replace NaN and infinite values with null
    # Required due to a bug in Pydantic 2.11.7
    # https://github.com/pydantic/pydantic/issues/10037
    @field_serializer("mean_score", "std_score", "min_score", "max_score", when_used="json")
    def serialize_float_values(self, value: float) -> float | None:
        """Convert NaN and infinity values to null when serializing to JSON.

        Args:
            value: The float value to serialize.

        Returns:
            The serialized float value, or None if the value is NaN or infinity.
        """
        if math.isnan(value) or math.isinf(value):
            return None
        return value


class ObjectsCloudResult(ObjectsResult):
    """Result model for object processing with storage URLs."""

    results_json_presigned_urls: dict[str, str] = Field(
        default={}, description="Presigned URLs of generated results JSON files"
    )
    results_json_urls: dict[str, str] = Field(default={}, description="Storage URLs of generated results JSON files")
    visualizations_presigned_urls: dict[str, str] = Field(
        default={}, description="Presigned URLs of generated visualizations"
    )
    visualizations_urls: dict[str, str] = Field(default={}, description="Storage URLs of generated visualizations")
    debug_artifacts_presigned_url: str | None = Field(
        default=None,
        description="Presigned URL of the opt-in debug artifacts tar (raw CWIP + hallucination masks), when debug_dump "
        "is enabled; None otherwise",
    )
    debug_artifacts_url: str | None = Field(
        default=None, description="Storage URL of the opt-in debug artifacts tar, when debug_dump is enabled"
    )
