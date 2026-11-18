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

"""Endpoints for the arbitrator database tables.

Covers ``arbitrator_executions``, ``checker_executions``, and ``clients``.

Read endpoints require a read key (or admin key).
Write endpoints require the admin key.
"""

from datetime import datetime
import logging
from typing import Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from database.api_models import (
    AdoptedExecutions,
    ArbitratorExecutionRow,
    CheckerExecutionRow,
    CheckerExecutionStateRow,
    ClientRow,
    CreatedCheckerExecution,
    CreatedExecution,
    ExecutionLeaseRenewal,
    RefusedCheckerExecutions,
    UpdatedCheckerExecution,
    UpdatedExecution,
)
from services.database.auth import require_admin_key
from services.database.service import DatabaseService, PoolUnavailableError
from services.framework.response_formatters.json_response_formatter import (
    ErrorEnvelope,
    JsonResponseFormatter,
    SuccessEnvelope,
)

logger = logging.getLogger(__name__)
# Every endpoint here can answer with the formatter's error envelope, so it is declared once for the whole router
# rather than repeated per route.
router = APIRouter(
    tags=["arbitrator"],
    responses={"4XX": {"model": ErrorEnvelope}, "5XX": {"model": ErrorEnvelope}},
)
json_formatter = JsonResponseFormatter()


def _get_service(request: Request) -> DatabaseService:
    return request.app.state.service


@router.get("/arbitrator-executions/{request_id}", response_model=SuccessEnvelope[ArbitratorExecutionRow])
async def get_execution(request_id: int, request: Request) -> JSONResponse:
    """Get a single arbitrator execution by request_id."""
    service = _get_service(request)
    result = await service.get_execution(request_id)
    if not result:
        return await json_formatter.format_error(ValueError(f"Execution {request_id} not found"), status_code=404)
    return await json_formatter.format_success(result)


@router.get(
    "/arbitrator-executions/{request_id}/checker-executions",
    response_model=SuccessEnvelope[list[CheckerExecutionRow]],
)
async def get_checker_executions_for_request(request_id: int, request: Request) -> JSONResponse:
    """List all checker_executions rows for a given arbitrator request_id.

    Used by the arbitrator's ``GET /executions/{request_id}`` snapshot endpoint
    to assemble the per-checker progress array. Rows are ordered by start_time.
    """
    service = _get_service(request)
    results = await service.get_checker_executions_for_request(request_id)
    return await json_formatter.format_success(results)


@router.get(
    "/arbitrator-executions/{request_id}/checker-states",
    response_model=SuccessEnvelope[list[CheckerExecutionStateRow]],
)
async def get_checker_execution_states_for_request(request_id: int, request: Request) -> JSONResponse:
    """List a request's checker_executions rows with their coordination columns.

    The wider sibling of ``/checker-executions`` above, read by a replica rebuilding a request whose original replica
    stopped renewing its claim.
    """
    service = _get_service(request)
    try:
        results = await service.get_checker_execution_states_for_request(request_id)
    except PoolUnavailableError as exc:
        return await json_formatter.format_error(exc, status_code=503)
    return await json_formatter.format_success(results)


# ── write endpoints (require admin key) ───────────────────────────
#
# Every ``lease_seconds`` below is constrained to be positive. ``lease_expires = NOW() + 0`` is a claim that expired
# before the row was written, so the request is adoptable immediately and its owner loses it to a sweep without a
# single call having failed. Rejected here as a 422 rather than persisted; the checker-work endpoints bound their own
# leases the same way, with an explicit floor and ceiling.


class RecordClientRequest(BaseModel):
    ip_address: str
    name: str | None = None
    description: str | None = None


class CreateExecutionRequest(BaseModel):
    client_id: int
    request_json: dict
    version: str
    commit_sha: str
    metadata: dict | None = None
    # Identity of the arbitrator replica that will orchestrate this request, and how long its claim holds before
    # another replica may adopt it. Omitted by callers that are not arbitrator replicas; such rows are never adopted.
    claimed_by: str | None = None
    lease_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _ownership_is_all_or_nothing(self) -> "CreateExecutionRequest":
        """Reject a half-specified claim, which would otherwise insert a row nothing can ever adopt."""
        if (self.claimed_by is None) != (self.lease_seconds is None):
            raise ValueError("claimed_by and lease_seconds must be provided together, or neither")
        return self


class UpdateExecutionRequest(BaseModel):
    """Update payload for ``arbitrator_executions``.

    ``status`` is one of the four lifecycle values ``queued`` / ``in_progress`` / ``success`` / ``fail`` (see
    ``database/schemas/orchestration.sql``). ``response_json`` and ``duration_ms`` are populated together on the
    terminal transitions written by ``ExecutionRecord.complete()`` / ``fail()``; the ``in_progress`` flip written
    when the request's first job is claimed carries ``status`` only.

    ``claimed_by`` fences the write to the replica that owns the request. A 404 then means either "no such row" or
    "you were adopted while you were working", and both leave the caller with nothing more to do.
    """

    status: Literal["queued", "in_progress", "success", "fail"]
    response_json: dict | None = None
    duration_ms: int | None = None
    claimed_by: str | None = None


class HeartbeatExecutionsRequest(BaseModel):
    """Lease renewal from a replica for every request it is orchestrating."""

    claimed_by: str
    lease_seconds: float = Field(gt=0)
    request_ids: list[int]


class AdoptExecutionsRequest(BaseModel):
    """Sweep for requests whose orchestrating replica went away."""

    claimed_by: str
    lease_seconds: float = Field(gt=0)
    limit: int = 10


class CreateCheckerExecutionRequest(BaseModel):
    request_id: int
    checker_name: str
    request_json: dict
    version: str = ""
    commit_sha: str = ""
    # Full RDS camera stream id for per-camera fan-out checkers (e.g. single-view). NULL for non-parameterized
    # checkers, such as cross-camera aggregators.
    camera_name: str | None = None


class UpdateCheckerExecutionRequest(BaseModel):
    """Update payload for ``checker_executions``.

    All fields are optional so callers can refine just the columns they actually need to change. Two distinct shapes are
    supported by the underlying service:

    - **Terminal update** (``record.complete()`` / ``fail()`` / ``timeout()`` / ``skipped()``) — ``status`` together
      with at least one of ``response_json`` / ``duration_ms`` / ``error_message`` / ``output_metadata``. Stamps
      ``end_time = NOW()`` server-side.
    - **Partial update** (``record.enqueue()``) — any subset of ``status``, ``request_json`` and/or ``start_time``
      without the terminal-payload columns. Touches only the requested columns; ``end_time`` / ``response_json`` /
      ``duration_ms`` are left intact.

    ``status`` accepts the six lifecycle values ``queued`` / ``in_progress`` / ``success`` / ``fail`` / ``timeout`` /
    ``skipped``. ``start_time`` refines the coarse row-insertion placeholder to the moment a pod claimed the job, so
    SQL postmortems can compute end-to-end timing accurately; the arbitrator makes that refinement in batches
    through ``POST /checker-executions/in-progress`` rather than here, since it observes claims a polling pass at a
    time. ``request_json`` is rewritten by ``record.enqueue()`` because the row is what the claiming pod reads its
    payload from, and the payload of a dependent job is only knowable once its dependencies have produced results.

    Either shape is refused with 409 on a row that has already reached a terminal status, whatever the caller asked
    for; a repeat of the status the row already holds is accepted as the no-op it is. Finishing a job is the owning
    pod's business, through ``POST /checker-executions/{id}/complete``, not this endpoint's.
    """

    status: Literal["queued", "in_progress", "success", "fail", "timeout", "skipped"] | None = None
    request_json: dict | None = None
    response_json: dict | None = None
    duration_ms: int | None = None
    error_message: str | None = None
    output_metadata: dict | None = None
    start_time: datetime | None = None


@router.post(
    "/clients",
    dependencies=[Depends(require_admin_key)],
    status_code=201,
    response_model=SuccessEnvelope[ClientRow],
)
async def record_client(body: RecordClientRequest, request: Request) -> JSONResponse:
    """Upsert a client record by IP address."""
    service = _get_service(request)
    result = await service.record_client(body.ip_address, name=body.name, description=body.description)
    if not result:
        return await json_formatter.format_error(ValueError("Database pool not available"), status_code=503)
    response = await json_formatter.format_success(result)
    response.status_code = 201
    return response


@router.post(
    "/arbitrator-executions",
    dependencies=[Depends(require_admin_key)],
    status_code=201,
    response_model=SuccessEnvelope[CreatedExecution],
)
async def create_execution(body: CreateExecutionRequest, request: Request) -> JSONResponse:
    """Create a new arbitrator execution record with 'queued' status."""
    service = _get_service(request)
    request_id = await service.create_execution(
        client_id=body.client_id,
        request_json=body.request_json,
        version=body.version,
        commit_sha=body.commit_sha,
        metadata=body.metadata,
        claimed_by=body.claimed_by,
        lease_seconds=body.lease_seconds,
    )
    if request_id is None:
        return await json_formatter.format_error(ValueError("Database pool not available"), status_code=503)
    response = await json_formatter.format_success({"request_id": request_id})
    response.status_code = 201
    return response


@router.put(
    "/arbitrator-executions/{request_id}",
    dependencies=[Depends(require_admin_key)],
    response_model=SuccessEnvelope[UpdatedExecution],
)
async def update_execution(request_id: int, body: UpdateExecutionRequest, request: Request) -> JSONResponse:
    """Update an arbitrator execution's status, response, and duration."""
    service = _get_service(request)
    try:
        updated = await service.update_execution(
            request_id=request_id,
            status=body.status,
            response_json=body.response_json,
            duration_ms=body.duration_ms,
            claimed_by=body.claimed_by,
        )
    except PoolUnavailableError as exc:
        return await json_formatter.format_error(exc, status_code=503)
    if not updated:
        return await json_formatter.format_error(ValueError(f"Execution {request_id} not found"), status_code=404)
    return await json_formatter.format_success({"request_id": request_id, "status": body.status})


@router.post(
    "/arbitrator-executions/heartbeat",
    dependencies=[Depends(require_admin_key)],
    response_model=SuccessEnvelope[ExecutionLeaseRenewal],
)
async def heartbeat_executions(body: HeartbeatExecutionsRequest, request: Request) -> JSONResponse:
    """Renew a replica's claim on every request it is still orchestrating, and report the ones it has lost.

    Succeeds even when nothing renewed. A request that could not be renewed is returned under ``lost`` together with the
    row state that explains why -- another replica's ``claimed_by``, or a terminal ``status`` -- because the caller has
    to log and act on those two cases differently, and a bare 404 cannot tell them apart.
    """
    service = _get_service(request)
    try:
        renewal = await service.heartbeat_executions(
            request_ids=body.request_ids, claimed_by=body.claimed_by, lease_seconds=body.lease_seconds
        )
    except PoolUnavailableError as exc:
        return await json_formatter.format_error(exc, status_code=503)
    return await json_formatter.format_success({"renewed": renewal.renewed, "lost": renewal.lost})


@router.post(
    "/arbitrator-executions/adopt",
    dependencies=[Depends(require_admin_key)],
    response_model=SuccessEnvelope[AdoptedExecutions],
)
async def adopt_executions(body: AdoptExecutionsRequest, request: Request) -> JSONResponse:
    """Claim up to ``limit`` requests whose orchestrating replica stopped renewing, and return them to be rebuilt.

    Safe to call from every replica concurrently: the claim uses ``FOR UPDATE SKIP LOCKED``, so each caller gets a
    disjoint set of rows.
    """
    service = _get_service(request)
    adopted = await service.adopt_executions(
        claimed_by=body.claimed_by, lease_seconds=body.lease_seconds, limit=body.limit
    )
    return await json_formatter.format_success({"executions": adopted})


@router.post(
    "/checker-executions",
    dependencies=[Depends(require_admin_key)],
    status_code=201,
    response_model=SuccessEnvelope[CreatedCheckerExecution],
)
async def create_checker_execution(body: CreateCheckerExecutionRequest, request: Request) -> JSONResponse:
    """Create a new checker execution record with 'queued' status."""
    service = _get_service(request)
    execution_id = await service.create_checker_execution(
        request_id=body.request_id,
        checker_name=body.checker_name,
        request_json=body.request_json,
        version=body.version,
        commit_sha=body.commit_sha,
        camera_name=body.camera_name,
    )
    if execution_id is None:
        return await json_formatter.format_error(ValueError("Database pool not available"), status_code=503)
    response = await json_formatter.format_success({"execution_id": execution_id})
    response.status_code = 201
    return response


@router.put(
    "/checker-executions/{execution_id}",
    dependencies=[Depends(require_admin_key)],
    response_model=SuccessEnvelope[UpdatedCheckerExecution],
)
async def update_checker_execution(
    execution_id: int, body: UpdateCheckerExecutionRequest, request: Request
) -> JSONResponse:
    """Update a checker execution, unless it has already finished.

    All body fields are optional so a single call can refine just one column. The arbitrator sends
    ``{"status": "in_progress", "start_time": "..."}`` once it sees a job claimed, overwriting the row-insertion
    placeholder with the moment a pod actually picked the job up.

    409 rather than 404 for an already-terminal row, because the two mean opposite things to the arbitrator: a
    missing row is a job it can no longer account for, while a terminal one is a job that finished without it
    noticing yet -- something to read on the next pass, not to react to. The status the row holds is returned in the
    error metadata so the caller can tell which.
    """
    service = _get_service(request)
    try:
        outcome = await service.update_checker_execution(
            execution_id=execution_id,
            status=body.status,
            request_json=body.request_json,
            response_json=body.response_json,
            duration_ms=body.duration_ms,
            error_message=body.error_message,
            output_metadata=body.output_metadata,
            start_time=body.start_time,
        )
    except PoolUnavailableError as exc:
        return await json_formatter.format_error(exc, status_code=503)
    if not outcome.found:
        return await json_formatter.format_error(
            ValueError(f"Checker execution {execution_id} not found"), status_code=404
        )
    if not outcome.updated:
        return await json_formatter.format_error(
            ValueError(
                f"Checker execution {execution_id} already reached terminal status "
                f"'{outcome.existing_status}'; refusing to write '{body.status}' over it"
            ),
            status_code=409,
            metadata={"existing_status": outcome.existing_status},
        )
    return await json_formatter.format_success({"execution_id": execution_id, "status": body.status})


class MarkCheckerExecutionsInProgressRequest(BaseModel):
    """The claims one arbitrator polling pass observed, written back together."""

    execution_ids: list[int]
    # One timestamp for the whole batch: the moment the pass saw these claims, which is what ``start_time`` is being
    # refined to mean. Supplied by the caller rather than defaulted server-side so a replica whose write is delayed
    # still records when it observed the claim, not when the database got around to it.
    start_time: datetime


@router.post(
    "/checker-executions/in-progress",
    dependencies=[Depends(require_admin_key)],
    response_model=SuccessEnvelope[RefusedCheckerExecutions],
)
async def mark_checker_executions_in_progress(
    body: MarkCheckerExecutionsInProgressRequest, request: Request
) -> JSONResponse:
    """Record observed claims on many jobs at once, reporting the ones that had already finished.

    The batched form of the ``{"status": "in_progress", "start_time": ...}`` write above, and a POST for the same
    reason ``/checker-executions/states`` is: the arbitrator writes back every job it just polled, and that
    identifier list is unbounded in a way a query string is not.

    200 even when rows are refused, unlike the single-row PUT's 409. A batch has no single outcome to encode in a
    status code, and a refusal here is not an error to begin with -- it is the expected answer for a short job that
    was claimed and finished inside one polling interval. The refusals are the response body, and an empty list is
    the healthy case.
    """
    service = _get_service(request)
    try:
        refused = await service.mark_checker_executions_in_progress(body.execution_ids, body.start_time)
    except PoolUnavailableError as exc:
        return await json_formatter.format_error(exc, status_code=503)
    if refused:
        logger.info(
            "%d of %d observed claim(s) were not recorded because the job had already finished: %s",
            len(refused),
            len(body.execution_ids),
            [row["execution_id"] for row in refused],
        )
    return await json_formatter.format_success({"refused": refused})
