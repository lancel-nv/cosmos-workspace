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

"""Job-coordination endpoints for checker pods.

A checker pod sits behind a gateway and cannot be addressed directly, so its job row is the only channel it shares
with the arbitrator. These endpoints are that channel: a pod takes work, refreshes its claim while it runs, and writes
its own terminal state, while the arbitrator reads the row and can leave a cancellation request on it.

A pod takes work by asking for whatever is waiting for its checker (``/claim-next``) rather than being handed a
particular job. That is what lets a pod take work only when it has room for it, so capacity needs no fleet-wide
accounting: a busy pod simply does not ask.

Claim, heartbeat and complete require a worker key: they are the only database surface checker pods are given, and
they are scoped to jobs the caller owns. Cancellation, the batched state read and the reaper are arbitrator concerns
and keep the admin and read keys respectively.

HTTP 409 has one meaning throughout: the caller does not own this job. Its lease expired and its work has been taken
over, so it must stop rather than retry.
"""

import asyncio
import logging
import random
import time
from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from database.api_models import (
    CancelRequestedCheckerJob,
    CheckerExecutionStateRow,
    CheckerJobState,
    ClaimedCheckerJobs,
    CompletedCheckerJob,
    RequeuedCheckerJobs,
)
from services.database.auth import require_admin_key, require_read_key, require_worker_key
from services.database.service import DatabaseService, PoolUnavailableError
from services.framework.response_formatters.json_response_formatter import (
    ErrorEnvelope,
    JsonResponseFormatter,
    SuccessEnvelope,
)

logger = logging.getLogger(__name__)
# 409 (the caller no longer owns the job) and 503 (no pool) are reachable from most of these endpoints, so the error
# envelope is declared once for the router rather than per route.
router = APIRouter(
    tags=["checker-work"],
    responses={"4XX": {"model": ErrorEnvelope}, "5XX": {"model": ErrorEnvelope}},
)
json_formatter = JsonResponseFormatter()

# Bounds on a caller-supplied lease. Too short and an ordinary garbage-collection pause looks like a dead pod and the
# job gets taken over while it is still running; too long and a genuinely dead pod's work sits unrecoverable for that
# whole window. The range is wide because checkers differ by orders of magnitude in step duration.
_MIN_LEASE_SECONDS = 5.0
_MAX_LEASE_SECONDS = 3600.0

# How long ``/claim-next`` may hold a request open waiting for work to appear. Comfortably inside the gateway's idle
# window, and long enough that an empty queue costs one request every twenty seconds per idle pod rather than one per
# second. The ceiling is what keeps the answer inside the gateway's patience; the caller may ask for less.
_MAX_CLAIM_WAIT_SECONDS = 20.0

# How often the long-poll re-checks the queue while it waits, and how much of that is jittered away. The interval is
# short because it costs one indexed query against a partial index; the jitter is what stops a fleet of idle pods from
# converging into a synchronized burst the moment work appears.
_CLAIM_RECHECK_INTERVAL = 1.0
_CLAIM_RECHECK_JITTER = 0.5

# Most jobs one claim call will hand over, and most rows one reaper sweep will requeue. Both bound the blast radius of
# a single call rather than expressing a policy: a pod asks for as many as it has slots for, and the reaper is called
# often enough that a backlog clears over a few sweeps.
_MAX_CLAIM_BATCH = 16
_MAX_REAP_BATCH = 100


def _get_service(request: Request) -> DatabaseService:
    return request.app.state.service


class ClaimNextJobsRequest(BaseModel):
    """Request for whatever work is waiting for one checker.

    ``limit`` is how many jobs the caller has room to start right now, so one request fills every free slot.
    ``wait_seconds`` is how long it may be held open if the queue is empty; zero makes it a plain poll.
    ``version`` and ``commit_sha`` are the caller's own build, recorded on every job it claims; omitting them leaves
    the rows' values unchanged.
    """

    checker_name: str = Field(min_length=1)
    claimed_by: str = Field(min_length=1)
    lease_seconds: float = Field(ge=_MIN_LEASE_SECONDS, le=_MAX_LEASE_SECONDS)
    limit: int = Field(default=1, ge=1, le=_MAX_CLAIM_BATCH)
    wait_seconds: float = Field(default=_MAX_CLAIM_WAIT_SECONDS, ge=0.0, le=_MAX_CLAIM_WAIT_SECONDS)
    version: str | None = None
    commit_sha: str | None = Field(default=None, max_length=64)


class ReapExpiredJobsRequest(BaseModel):
    """Sweep for claims whose owner stopped renewing."""

    limit: int = Field(default=_MAX_REAP_BATCH, ge=1, le=_MAX_REAP_BATCH)


class HeartbeatJobRequest(BaseModel):
    """Request to extend an existing claim."""

    claimed_by: str = Field(min_length=1)
    lease_seconds: float = Field(ge=_MIN_LEASE_SECONDS, le=_MAX_LEASE_SECONDS)


class CompleteJobRequest(BaseModel):
    """Terminal write from the pod that owns a checker job.

    The status vocabulary is the terminal subset of the seven values ``checker_executions.status`` accepts: an owner
    reports how its own attempt ended and has no reason to move a job back into a pending state.
    """

    claimed_by: str = Field(min_length=1)
    status: Literal["success", "fail", "timeout"]
    response_json: dict[str, Any] | None = None
    duration_ms: int | None = None
    error_message: str | None = None
    output_metadata: dict[str, Any] | None = None


class CheckerExecutionStatesRequest(BaseModel):
    """Batched read of the coordination state of many jobs."""

    execution_ids: list[int]


@router.post(
    "/checker-executions/claim-next",
    dependencies=[Depends(require_worker_key)],
    response_model=SuccessEnvelope[ClaimedCheckerJobs],
)
async def claim_next_checker_executions(body: ClaimNextJobsRequest, request: Request) -> JSONResponse:
    """Hand the caller up to ``limit`` of the oldest waiting jobs for its checker, waiting briefly for one to appear.

    An empty list is an ordinary answer, not an error: it means nothing was waiting for the whole window.

    The wait is a loop of short queries rather than one long-running one, and that distinction is the reason this
    endpoint is affordable. Holding a database connection open for the whole window would cost one connection per
    waiting pod, which at fleet size is most of the instance's connection budget spent on doing nothing -- and would
    starve every other caller, including the arbitrator's ability to admit requests at all. Polling instead costs a few
    milliseconds of one connection per recheck.
    """
    service = _get_service(request)
    deadline = time.monotonic() + body.wait_seconds
    while True:
        try:
            jobs = await service.claim_next_checker_executions(
                checker_name=body.checker_name,
                claimed_by=body.claimed_by,
                lease_seconds=body.lease_seconds,
                limit=body.limit,
                version=body.version,
                commit_sha=body.commit_sha,
            )
        except PoolUnavailableError as exc:
            return await json_formatter.format_error(exc, status_code=503)
        if jobs:
            return await json_formatter.format_success({"jobs": jobs})
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return await json_formatter.format_success({"jobs": []})
        # Jittered so a fleet of idle pods does not synchronize onto the same instant. FOR UPDATE SKIP LOCKED copes
        # with a herd correctly, but there is no reason to create one.
        interval = _CLAIM_RECHECK_INTERVAL + random.uniform(0.0, _CLAIM_RECHECK_JITTER)
        await asyncio.sleep(min(interval, remaining))


@router.post(
    "/checker-executions/reap",
    dependencies=[Depends(require_admin_key)],
    response_model=SuccessEnvelope[RequeuedCheckerJobs],
)
async def reap_expired_checker_executions(body: ReapExpiredJobsRequest, request: Request) -> JSONResponse:
    """Requeue jobs whose owner stopped renewing its claim, so another pod can pick them up.

    Idempotent by construction -- a single conditional UPDATE -- so every replica can call this on a jittered interval
    without electing a leader. Whichever gets there first requeues the row and the rest match nothing.
    """
    service = _get_service(request)
    requeued = await service.requeue_expired_checker_executions(limit=body.limit)
    if requeued:
        logger.warning(
            "Requeued %d checker job(s) abandoned by their owner: %s",
            len(requeued),
            [row["execution_id"] for row in requeued],
        )
    return await json_formatter.format_success({"requeued": requeued})


@router.post(
    "/checker-executions/{execution_id}/heartbeat",
    dependencies=[Depends(require_worker_key)],
    response_model=SuccessEnvelope[CheckerJobState],
)
async def heartbeat_checker_execution(execution_id: int, body: HeartbeatJobRequest, request: Request) -> JSONResponse:
    """Extend the caller's lease, returning whether a cancellation has been requested."""
    service = _get_service(request)
    try:
        state = await service.heartbeat_checker_execution(
            execution_id=execution_id,
            claimed_by=body.claimed_by,
            lease_seconds=body.lease_seconds,
        )
    except PoolUnavailableError as exc:
        return await json_formatter.format_error(exc, status_code=503)
    if state is None:
        return await json_formatter.format_error(
            ValueError(f"Checker execution {execution_id} is no longer claimed by {body.claimed_by}"), status_code=409
        )
    return await json_formatter.format_success(state)


@router.post(
    "/checker-executions/{execution_id}/complete",
    dependencies=[Depends(require_worker_key)],
    response_model=SuccessEnvelope[CompletedCheckerJob],
)
async def complete_checker_execution(execution_id: int, body: CompleteJobRequest, request: Request) -> JSONResponse:
    """Write the caller's terminal state for a job it owns."""
    service = _get_service(request)
    try:
        written = await service.complete_checker_execution(
            execution_id=execution_id,
            claimed_by=body.claimed_by,
            status=body.status,
            response_json=body.response_json,
            duration_ms=body.duration_ms,
            error_message=body.error_message,
            output_metadata=body.output_metadata,
        )
    except PoolUnavailableError as exc:
        return await json_formatter.format_error(exc, status_code=503)
    if not written:
        return await json_formatter.format_error(
            ValueError(f"Checker execution {execution_id} is no longer claimed by {body.claimed_by}"), status_code=409
        )
    return await json_formatter.format_success({"execution_id": execution_id, "status": body.status})


@router.post(
    "/checker-executions/{execution_id}/cancel",
    dependencies=[Depends(require_admin_key)],
    response_model=SuccessEnvelope[CancelRequestedCheckerJob],
)
async def cancel_checker_execution(execution_id: int, request: Request) -> JSONResponse:
    """Ask the owner of a job to stop.

    Answering 404 for an already-terminal job keeps the endpoint honest: there is nothing left to cancel, and
    reporting success would suggest the arbitrator had stopped work that had in fact already finished.
    """
    service = _get_service(request)
    try:
        requested = await service.request_checker_execution_cancel(execution_id)
    except PoolUnavailableError as exc:
        return await json_formatter.format_error(exc, status_code=503)
    if not requested:
        return await json_formatter.format_error(
            ValueError(f"Checker execution {execution_id} is not running"), status_code=404
        )
    return await json_formatter.format_success({"execution_id": execution_id, "cancel_requested": True})


@router.post(
    "/checker-executions/states",
    dependencies=[Depends(require_read_key)],
    response_model=SuccessEnvelope[list[CheckerExecutionStateRow]],
)
async def get_checker_execution_states(body: CheckerExecutionStatesRequest, request: Request) -> JSONResponse:
    """Read the coordination state of many jobs at once.

    A POST for what is logically a read, because the arbitrator polls every job it owns and that identifier list is
    unbounded in a way a query string is not.
    """
    service = _get_service(request)
    try:
        states = await service.get_checker_execution_states(body.execution_ids)
    except PoolUnavailableError as exc:
        return await json_formatter.format_error(exc, status_code=503)
    return await json_formatter.format_success(states)
