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
Generic response-envelope and error machinery shared across all Arbitrator endpoints.

Every endpoint returns the standard envelope produced by ``JsonResponseFormatter`` (``{success, data, timestamp,
[metadata]}`` on success; ``{success: false, error: {...}, timestamp}`` on error). The endpoints construct
``JSONResponse`` directly, so these models do not perform runtime validation of responses; they exist to populate
the OpenAPI schema rendered at ``/docs`` so callers see the actual response shape per endpoint. Also hosts the
``ExpiredInputUrlError`` exception, kept next to its typed error envelope models.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ApiSuccessEnvelope[DataT](BaseModel):
    """Standard success envelope returned by every endpoint."""

    success: Literal[True] = Field(default=True, description="Always ``true`` for success responses.")
    data: DataT = Field(..., description="Endpoint-specific success payload.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of when the response was produced.")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional response metadata (omitted by default)."
    )


class ApiErrorDetails(BaseModel):
    """Body of the ``error`` field on error envelopes."""

    type: str = Field(..., description="Exception class name (e.g. ``ValueError``).")
    message: str = Field(..., description="Human-readable error message.")
    code: str | None = Field(default=None, description="Optional error code (only set when raised exception has one).")
    details: Any | None = Field(default=None, description="Optional structured error details.")
    field: str | None = Field(default=None, description="Optional field name for validation-style errors.")


class ApiErrorEnvelope(BaseModel):
    """Standard error envelope returned for non-2xx responses."""

    success: Literal[False] = Field(default=False, description="Always ``false`` for error responses.")
    error: ApiErrorDetails = Field(..., description="Error type and message.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of when the error response was produced.")


class ExpiredUrlDetail(BaseModel):
    """Per-URL diagnostic for a rejected expired-signature input."""

    field: str = Field(
        ...,
        description=(
            "JSON-path label of the offending request field, "
            "e.g. ``inputs.augmented_video_urls[camera_front_wide_120fov]``."
        ),
    )
    scheme: str = Field(
        ...,
        description=(
            "Detected signature scheme: one of ``aws_sigv4``, ``aws_sigv2``, ``azure_sas``, ``gcs_v4``, or ``unknown``."
        ),
    )
    expires_at: str = Field(
        ...,
        description=(
            "ISO 8601 UTC expiry timestamp parsed from the signature, or ``unknown`` when the "
            "scheme was detected but its expiry parameters could not be read."
        ),
    )
    detail: str = Field(
        ...,
        description="Human-readable rejection reason, e.g. ``expired 312s ago``.",
    )


class ExpiredInputUrlsErrorDetails(BaseModel):
    """Structured ``error.details`` payload for ``ExpiredInputUrlError``."""

    request_id: int = Field(
        ...,
        description=(
            "Minted ``arbitrator_executions.request_id`` for the rejected request. Clients can "
            "poll ``GET /executions/{request_id}`` to inspect the persisted ``fail`` row."
        ),
    )
    expired_urls: list[ExpiredUrlDetail] = Field(
        ...,
        description="Per-URL breakdown of every input URL whose presigned signature has expired.",
    )


class ExpiredInputUrlsError(BaseModel):
    """Body of the ``error`` field on the 422 expired-input-URL envelope."""

    type: Literal["ExpiredInputUrlError"] = Field(
        default="ExpiredInputUrlError",
        description="Always ``ExpiredInputUrlError``; clients can dispatch on this discriminator.",
    )
    message: str = Field(..., description="Human-readable error message.")
    details: ExpiredInputUrlsErrorDetails = Field(
        ...,
        description="Typed details payload identifying the rejected request and every expired URL.",
    )


class ExpiredInputUrlsErrorEnvelope(BaseModel):
    """Error envelope returned with HTTP 422 when one or more input URLs have expired signatures."""

    success: Literal[False] = Field(default=False, description="Always ``false`` for error responses.")
    error: ExpiredInputUrlsError = Field(..., description="Typed error payload for expired-input-URL rejections.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of when the error response was produced.")


class ExpiredInputUrlError(ValueError):
    """One or more input URLs carry a presigned signature that has already expired.

    Attaches ``details`` so :meth:`JsonResponseFormatter.format_error` surfaces both the minted ``request_id`` (clients
    can poll ``GET /executions/{request_id}`` to inspect the persisted rejection row) and the per-URL diagnostic
    payload alongside the standard error envelope.
    """

    def __init__(self, request_id: int, expired_urls: list[dict[str, str]]) -> None:
        fields = [entry.get("field", "?") for entry in expired_urls]
        super().__init__(f"One or more input URLs are expired: {fields}")
        self.details = {
            "request_id": request_id,
            "expired_urls": expired_urls,
        }
