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

from datetime import UTC, date, datetime
from ipaddress import IPv4Address, IPv4Network, IPv6Address, IPv6Network
import json
from typing import Any, Literal

from fastapi import status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict


class SuccessEnvelope[DataT](BaseModel):
    """Schema of every body :meth:`JsonResponseFormatter.format_success` produces.

    Declared here, beside the method that builds it, so the two cannot drift apart unnoticed. Routes name it as their
    ``response_model`` parameterized by their own payload type, which is what puts the payload shape in the OpenAPI
    document and gives contract tests something to validate a real response against.

    Note that declaring this does not make FastAPI validate anything: these routes return a ``JSONResponse`` built by
    the formatter, and FastAPI skips response validation whenever a handler returns a ``Response`` directly. That is
    deliberate -- the formatter owns serialization, and routing it through Pydantic instead would change the wire
    format (tz-aware datetimes render as ``...Z`` rather than ``...+00:00``, among other differences). Enforcement
    comes from the contract tests, which validate real response bodies against these models.

    ``timestamp`` is present whenever the formatter was built with ``include_timestamp`` (the default). ``metadata`` is
    omitted entirely rather than set to null when a route passes none.
    """

    model_config = ConfigDict(extra="forbid")

    success: Literal[True]
    data: DataT
    timestamp: datetime
    metadata: dict[str, Any] | None = None


class ErrorDetail(BaseModel):
    """The ``error`` object inside an error response.

    ``code``, ``details`` and ``field`` appear only when the raised exception carried the correspondingly named
    attribute; ``validation_errors`` only on bodies from :meth:`JsonResponseFormatter.format_validation_error`.
    """

    model_config = ConfigDict(extra="forbid")

    type: str
    message: str
    code: str | None = None
    details: Any = None
    field: str | None = None
    validation_errors: list[Any] | None = None


class ErrorEnvelope(BaseModel):
    """Schema of every body :meth:`JsonResponseFormatter.format_error` produces.

    Named once at router level so all of a router's endpoints document their failure shape without repeating it on
    every route.
    """

    model_config = ConfigDict(extra="forbid")

    success: Literal[False]
    error: ErrorDetail
    timestamp: datetime
    metadata: dict[str, Any] | None = None


class _SafeEncoder(json.JSONEncoder):
    """JSON encoder that handles types commonly returned by asyncpg/PostgreSQL."""

    def default(self, o: Any) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, date):
            return o.isoformat()
        if isinstance(o, (IPv4Address, IPv6Address, IPv4Network, IPv6Network)):
            return str(o)
        return super().default(o)


class _SafeJSONResponse(JSONResponse):
    """JSONResponse subclass that handles non-standard types (datetime, IP addresses, etc.)."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            cls=_SafeEncoder,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


class JsonResponseFormatter:
    """JSON response formatter implementation."""

    def __init__(self, include_timestamp: bool = True) -> None:
        """Initializes JSON response formatter.

        Args:
            include_timestamp: Whether to include timestamp in responses
        """
        self.include_timestamp = include_timestamp

    async def format_success(
        self, data: Any, metadata: dict[str, Any] | None = None, status_code: int = status.HTTP_200_OK
    ) -> JSONResponse:
        """Formats successful JSON response.

        Args:
            data: Response data
            metadata: Optional response metadata
            status_code: Success status to return. Override it for outcomes that succeeded without completing, such as
                the 202 a checker returns when it has accepted a job it has not yet run.

        Returns:
            JSONResponse with formatted data
        """
        response_data = {"success": True, "data": data}

        if self.include_timestamp:
            response_data["timestamp"] = datetime.now(UTC).isoformat()

        if metadata:
            response_data["metadata"] = metadata

        return _SafeJSONResponse(
            content=response_data, status_code=status_code, headers={"Content-Type": "application/json"}
        )

    async def format_error(
        self, error: Exception, status_code: int = 500, metadata: dict[str, Any] | None = None
    ) -> JSONResponse:
        """Formats error JSON response.

        Args:
            error: Exception that occurred
            status_code: HTTP status code
            metadata: Optional response metadata

        Returns:
            JSONResponse with error information
        """
        error_data: dict[str, Any] = {
            "success": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }

        if self.include_timestamp:
            error_data["timestamp"] = datetime.now(UTC).isoformat()

        # Add additional error details if available
        if hasattr(error, "error_code"):
            error_data["error"]["code"] = error.error_code
        if hasattr(error, "details"):
            error_data["error"]["details"] = error.details
        if hasattr(error, "field"):
            error_data["error"]["field"] = error.field

        if metadata:
            error_data["metadata"] = metadata

        return _SafeJSONResponse(
            content=error_data, status_code=status_code, headers={"Content-Type": "application/json"}
        )

    async def format_progress(self, progress: float, message: str = "") -> JSONResponse:
        """Formats progress JSON response.

        Args:
            progress: Progress percentage (0-100)
            message: Optional progress message

        Returns:
            JSONResponse with progress information
        """
        response_data = {
            "success": True,
            "progress": {
                "percentage": min(100.0, max(0.0, progress)),
                "message": message,
                "complete": progress >= 100.0,
            },
        }

        if self.include_timestamp:
            response_data["timestamp"] = datetime.now(UTC).isoformat()

        return _SafeJSONResponse(
            content=response_data, status_code=status.HTTP_200_OK, headers={"Content-Type": "application/json"}
        )

    def get_content_type(self) -> str:
        """Gets content type for JSON responses."""
        return "application/json"

    def format_validation_error(self, validation_errors: list, status_code: int = 422) -> JSONResponse:
        """Formats validation error response.

        Args:
            validation_errors: List of validation errors
            status_code: HTTP status code

        Returns:
            JSONResponse with validation error details
        """
        error_data = {
            "success": False,
            "error": {
                "type": "ValidationError",
                "message": "Request validation failed",
                "validation_errors": validation_errors,
            },
        }

        if self.include_timestamp:
            error_data["timestamp"] = datetime.now(UTC).isoformat()

        return _SafeJSONResponse(
            content=error_data, status_code=status_code, headers={"Content-Type": "application/json"}
        )

    def format_paginated_response(
        self, items: list, total: int, page: int, page_size: int, metadata: dict[str, Any] | None = None
    ) -> JSONResponse:
        """Formats paginated response.

        Args:
            items: List of items for current page
            total: Total number of items
            page: Current page number (1-based)
            page_size: Number of items per page
            metadata: Optional additional metadata

        Returns:
            JSONResponse with paginated data
        """
        # Validate parameters
        if page_size <= 0:
            raise ValueError("Number of items per page (page_size) must be positive")
        if page < 1:
            raise ValueError("Current page number (page) must be at least 1")
        if total < 0:
            raise ValueError("Total number of items (total) cannot be negative")

        total_pages = (total + page_size - 1) // page_size
        has_next = page < total_pages
        has_prev = page > 1

        response_data = {
            "success": True,
            "data": items,
            "pagination": {
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev,
            },
        }

        if self.include_timestamp:
            response_data["timestamp"] = datetime.now(UTC).isoformat()

        if metadata:
            response_data["metadata"] = metadata

        return _SafeJSONResponse(
            content=response_data, status_code=status.HTTP_200_OK, headers={"Content-Type": "application/json"}
        )
