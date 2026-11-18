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
Per-endpoint response payload models plus their ``ApiSuccessEnvelope[...]`` aliases.

Each ``*Data`` model is the ``data`` field of a specific endpoint's success envelope; the module-level ``*Response``
aliases bind the generic :class:`~services.arbitrator.models.envelopes.ApiSuccessEnvelope` to each payload so the
endpoints can advertise the concrete response shape in their OpenAPI schema.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from services.arbitrator.models.contract import ArbitratorResult
from services.arbitrator.models.envelopes import ApiSuccessEnvelope


class CheckersData(BaseModel):
    """Payload of ``GET /checkers``."""

    checkers: list[str] = Field(
        ...,
        description="Names of checkers whose ``/health`` endpoint reported healthy at request time.",
    )


class DependencyGraphCheckerEntry(BaseModel):
    """One entry in the dependency graph map."""

    deps: list[str] = Field(..., description="Names of checkers this checker depends on (must finish first).")
    priority: int = Field(
        ...,
        description=(
            "Topological priority used by the wave scheduler. Lower values run earlier; "
            "checkers with the same priority can run in parallel."
        ),
    )


class DependencyGraphData(BaseModel):
    """Payload of ``GET /dependency-graph``."""

    checkers: dict[str, DependencyGraphCheckerEntry] = Field(
        ...,
        description="Map of checker name to its dependency-graph entry (deps + priority).",
    )


class ConfigData(BaseModel):
    """Payload of ``GET /config``."""

    config: dict[str, Any] = Field(
        ...,
        description=(
            "Map of checker name to that checker's default configuration object as returned by the "
            "checker's own ``/config`` endpoint. Entries whose checker did not respond contain ``{'error': ...}``."
        ),
    )


class HealthData(BaseModel):
    """Payload of ``GET /health``."""

    service: str = Field(..., description="FastAPI app title (``Arbitrator Service API``).")
    status: Literal["healthy"] = Field(..., description="Always ``healthy`` if this endpoint returns 200.")
    version: str = Field(..., description="Service version string.")
    git_sha: str = Field(..., description="Git commit SHA the service was built from.")
    environment: str = Field(..., description="Environment the service is running in (``local``/``dev``/...).")
    database: Literal["ok", "unavailable"] = Field(
        "unavailable",
        description=(
            "Reachability of the Database REST API backing ``POST /process/async``. ``ok`` means the DB answered a "
            "health probe; ``unavailable`` means it is unconfigured or unreachable, so async grading will 503. This "
            "field is additive — the top-level ``status`` stays ``healthy`` regardless so existing liveness consumers "
            "are unaffected."
        ),
    )
    draining: bool = Field(
        False,
        description=(
            "Whether this replica has been asked to shut down and is finishing its admitted requests. While ``true`` "
            "it rejects ``POST /process/async`` with 503 but still answers observation endpoints for the requests it "
            "is completing. Additive, like ``database``: the top-level ``status`` stays ``healthy`` so a liveness "
            "probe does not read a graceful drain as a crash."
        ),
    )


class ProcessAsyncData(BaseModel):
    """Payload of ``POST /process/async``."""

    request_id: int = Field(
        ...,
        description=(
            "Database-issued id for the newly created arbitrator execution. "
            "Pass to ``GET /executions/{request_id}`` to observe progress."
        ),
    )


class ExecutionRow(BaseModel):
    """Parent ``arbitrator_executions`` row (with ``is_terminal`` derived)."""

    request_id: int = Field(..., description="Database id of this arbitrator execution.")
    status: str = Field(
        ...,
        description=(
            "Lifecycle status. One of ``queued`` (admitted to the waiting room; no pod has claimed any of "
            "this request's jobs yet), ``in_progress`` (at least one pod has claimed a job from this "
            "request), ``success`` (terminal), or ``fail`` (terminal)."
        ),
    )
    is_terminal: bool = Field(
        ...,
        description=(
            "Convenience flag derived from ``status``. ``true`` iff the execution has reached a terminal "
            "state (``success`` or ``fail``) and clients should stop polling."
        ),
    )
    timestamp: str | None = Field(default=None, description="ISO 8601 timestamp the execution row was created.")
    duration_ms: int | None = Field(
        default=None,
        description="End-to-end duration in milliseconds. ``0`` while in-progress; final on success/fail.",
    )
    version: str | None = Field(default=None, description="Service version that handled the run.")
    commit_sha: str | None = Field(default=None, description="Git commit SHA the service was running.")
    response_json: ArbitratorResult | dict[str, Any] | None = Field(
        default=None,
        description=(
            "Final aggregated arbitrator response (``ArbitratorResult`` shape) **only** when "
            "``is_terminal`` is ``true``. ``null`` while pending/in-progress."
        ),
    )


class CheckerExecutionRow(BaseModel):
    """One projection of a ``checker_executions`` row for a given ``request_id``.

    Mirrors most of the DB columns, but **omits ``response_json``**. The raw per-checker payload is redundant with
    ``execution.response_json`` once the run is terminal (which carries the same data unwrapped + nested by base_name
    → camera_name) and is otherwise reachable by querying the ``checker_executions`` table directly. Excluding it
    from the snapshot keeps poll responses small even on multi-camera 7-stream runs.

    Fields not yet populated by the wave loop come back ``null``.
    """

    execution_id: int = Field(..., description="Database id of this checker_execution row.")
    request_id: int = Field(..., description="Parent arbitrator execution id this row belongs to.")
    checker_name: str = Field(..., description="Logical checker name (e.g. ``av.objects_single_view``).")
    camera_name: str | None = Field(default=None, description="RDS camera stream id this checker invocation ran on.")
    status: str = Field(
        ...,
        description=(
            "Per-checker status. One of ``queued`` (waiting on dependencies, or on the queue for a pod to "
            "claim), ``in_progress`` (a pod claimed the job and is processing it), ``success`` (terminal), "
            "``fail`` (terminal — checker error, or the job exhausted its attempts), ``timeout`` (terminal — "
            "the job outlived the arbitrator's patience for it), or ``skipped`` (terminal — a dependency hit "
            "a non-success terminal state, so this job was never queued for real)."
        ),
    )
    start_time: str | None = Field(default=None, description="ISO 8601 timestamp the checker invocation started.")
    end_time: str | None = Field(default=None, description="ISO 8601 timestamp the checker invocation ended.")
    duration_ms: int | None = Field(default=None, description="Per-checker duration in milliseconds.")
    error_message: str | None = Field(default=None, description="Error message when ``status == 'fail'``.")
    version: str | None = Field(default=None, description="Checker service version.")
    commit_sha: str | None = Field(default=None, description="Checker service commit SHA.")
    output_metadata: dict[str, Any] | None = Field(
        default=None, description="Per-checker output metadata (e.g. result URLs)."
    )


class CheckersGroup(BaseModel):
    """Aggregate progress info plus the underlying checker_execution rows."""

    completed: int = Field(
        ...,
        description=(
            "Count of child rows whose status is terminal (``success``, ``fail``, ``timeout`` or ``skipped``). The "
            "last two are terminal for a checker but never surface as a parent status, so counting them here is what "
            "keeps a timed-out or skipped checker from reading as still running."
        ),
    )
    total: int = Field(..., description="Total number of child checker_execution rows for this request.")
    items: list[CheckerExecutionRow] = Field(
        ..., description="The underlying ``checker_executions`` rows ordered by ``start_time``."
    )


class ExecutionSnapshotData(BaseModel):
    """Payload of ``GET /executions/{request_id}``."""

    execution: ExecutionRow = Field(..., description="Parent execution row with ``is_terminal`` derived.")
    checkers: CheckersGroup = Field(..., description="Children rows plus aggregate progress counts.")


CheckersResponse = ApiSuccessEnvelope[CheckersData]
DependencyGraphResponse = ApiSuccessEnvelope[DependencyGraphData]
ConfigResponse = ApiSuccessEnvelope[ConfigData]
HealthResponse = ApiSuccessEnvelope[HealthData]
ProcessAsyncResponse = ApiSuccessEnvelope[ProcessAsyncData]
ExecutionSnapshotResponse = ApiSuccessEnvelope[ExecutionSnapshotData]
