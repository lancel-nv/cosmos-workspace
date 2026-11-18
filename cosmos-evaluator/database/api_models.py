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

"""Pydantic response models for the orchestration subset of the Database REST API.

These model the payloads returned inside the ``{"success", "data", ...}`` envelope by the Database REST API's
orchestration endpoints (clients, arbitrator_executions, checker_executions): the row shapes its reads return, and
the small echo and wrapper payloads its writes return. They are deliberately kept dependency-light (pure Pydantic,
no FastAPI / asyncpg / boto3) so the client library can import them without pulling in the server's dependency graph.

**Both sides of the wire use this module, and that is the point.** ``ApiClient`` parses every response through these
models, and the server names them as each route's ``response_model``, so the contract has one definition rather than
one per end. A field added on one side and not the other is a change to this file, which is reviewable, instead of a
mismatch that surfaces as a consumer failing in production. Adding an endpoint means adding its payload here.

Every model uses ``extra="allow"`` so a column added server-side survives a parse -> ``model_dump()`` round-trip
rather than being silently dropped; this matters because the arbitrator forwards whole rows over the wire. The
consequence is that these models pin the fields they declare but do not reject undeclared ones, so they document and
type the contract without freezing it. Dynamic JSONB payloads (``response_json``, ``output_metadata``) stay loosely
typed as ``dict`` because their inner shape is checker-specific; the Database REST API's asyncpg pool registers a
``json``/``jsonb`` codec, so these always arrive decoded as objects (never raw JSON strings). Temporal columns are
kept as ISO ``str``.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict


class ClientRow(BaseModel):
    """A ``clients`` row as returned by ``ApiClient.record_client`` (client tracking upsert)."""

    model_config = ConfigDict(extra="allow")

    client_id: int
    ip_address: str
    name: str | None = None
    description: str | None = None


class ArbitratorExecutionRow(BaseModel):
    """An ``arbitrator_executions`` row as returned by ``ApiClient.get_execution``."""

    model_config = ConfigDict(extra="allow")

    request_id: int
    status: str
    response_json: dict[str, Any] | None = None
    timestamp: str | None = None
    duration_ms: int | None = None
    version: str | None = None
    commit_sha: str | None = None


class CheckerExecutionRow(BaseModel):
    """A ``checker_executions`` row as returned by ``ApiClient.get_checker_executions_for_request``.

    ``response_json`` and ``output_metadata`` carry the checker's own (checker-specific, alias-heavy) payload and are
    left opaque as ``dict``; temporal columns are kept as ISO strings to match what consumers currently sort/format on.
    The Database REST API's asyncpg pool registers a ``json``/``jsonb`` codec, so ``response_json`` always arrives
    decoded as an object rather than a raw JSON string.
    """

    model_config = ConfigDict(extra="allow")

    execution_id: int
    request_id: int | None = None
    checker_name: str | None = None
    camera_name: str | None = None
    status: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    duration_ms: int | None = None
    error_message: str | None = None
    version: str | None = None
    commit_sha: str | None = None
    output_metadata: dict[str, Any] | None = None
    response_json: dict[str, Any] | None = None
    output_uuid: str | None = None
    created_at: str | None = None


class CheckerJobClaim(BaseModel):
    """The outcome of a successful claim on a ``checker_executions`` row.

    ``claimed_by`` is echoed back rather than assumed, because it is the fencing token every later write must carry
    and reading it from the server's response keeps the caller from inventing one that never reached the row.

    ``checker_name`` / ``camera_name`` / ``request_id`` identify work the claimer chose rather than was handed: a pod
    claiming whatever is waiting for its checker learns what it got from the claim itself.
    """

    model_config = ConfigDict(extra="allow")

    execution_id: int
    claimed_by: str
    attempt: int
    lease_expires: str | None = None
    cancel_requested: bool = False
    request_json: dict[str, Any] | None = None
    checker_name: str | None = None
    camera_name: str | None = None
    request_id: int | None = None


class RequeuedCheckerJob(BaseModel):
    """One ``checker_executions`` row the reaper returned to the queue after its owner stopped renewing."""

    model_config = ConfigDict(extra="allow")

    execution_id: int
    checker_name: str | None = None
    attempt: int | None = None


class CheckerJobState(BaseModel):
    """The control state of a claimed job, as returned by a heartbeat."""

    model_config = ConfigDict(extra="allow")

    execution_id: int
    lease_expires: str | None = None
    cancel_requested: bool = False
    attempt: int | None = None


class CheckerExecutionStateRow(CheckerExecutionRow):
    """A ``checker_executions`` row with its coordination columns, as polled by the arbitrator.

    A superset of the per-request projection: the arbitrator needs the result columns to assemble its response and
    the coordination columns to tell a job that is progressing from one whose owning pod has died.
    """

    claimed_by: str | None = None
    lease_expires: str | None = None
    attempt: int | None = None
    cancel_requested: bool = False


class AdoptedExecutionRow(BaseModel):
    """An ``arbitrator_executions`` row this replica has just taken over, with what it needs to rebuild the request.

    ``request_json`` is the original submission, which is the only thing that cannot be reconstructed from anywhere
    else. Everything that has happened since lives in the request's ``checker_executions`` rows.
    """

    model_config = ConfigDict(extra="allow")

    request_id: int
    client_id: int
    status: str
    request_json: dict[str, Any]
    claimed_by: str
    lease_expires: str | None = None
    version: str | None = None
    commit_sha: str | None = None


class LostExecutionLease(BaseModel):
    """A request whose lease a replica could not renew, with the row state that says why.

    ``claimed_by`` naming a different replica means the request was adopted and its outcome belongs to that replica now.
    ``claimed_by`` still naming the caller means the row went terminal underneath it, so the outcome is already
    recorded. Both fields are ``None`` if the row no longer exists at all.
    """

    model_config = ConfigDict(extra="allow")

    request_id: int
    status: str | None = None
    claimed_by: str | None = None


class ExecutionLeaseRenewal(BaseModel):
    """Which of a replica's claims survived a renewal pass, and why the rest did not."""

    model_config = ConfigDict(extra="allow")

    renewed: list[int] = []
    lost: list[LostExecutionLease] = []


class CheckerExecutionByVideoRow(CheckerExecutionRow):
    """A ``checker_executions`` row from ``ApiClient.get_checker_executions_by_video``.

    The by-video listing reaches a checker execution through the video UUID it was run for, so it is a superset of the
    per-request projection: the video UUID that was queried, plus the orchestrating arbitrator's provenance. Note that
    ``version`` and ``commit_sha`` remain the *checker's*; the arbitrator's are the ``arbitrator_``-prefixed pair.

    ``video_uuid`` is declared because the endpoint does return it, but is optional rather than required, and consumers
    should not reach for it: the query filters on it, so it only ever echoes back the caller's own argument, and callers
    that hold an arbitrator request id take the request-scoped path instead, which yields the plain
    :class:`CheckerExecutionRow`. Both shapes flow through the same downstream code, so anything that depended on the
    UUID being present would break on the path that is now the primary one.

    Reaching a row by video UUID requires a mapping from request to video, which the bundled schema has no table for.
    The endpoint behind this method is therefore served only by a deployment that adds such a table and a router over
    it; the model is declared here so that the client method and any such router agree on one shape.
    """

    video_uuid: str | None = None
    arbitrator_timestamp: str | None = None
    arbitrator_version: str | None = None
    arbitrator_commit_sha: str | None = None


# =========================================================================
# Write echoes
# =========================================================================
# What a write returns rather than what a row contains: the identifier the caller now has to refer to, and the state
# the server agreed to record. Small, but modelled for the same reason as the rows -- an endpoint whose response is
# undeclared is an endpoint whose response can change without anyone noticing.


class CreatedExecution(BaseModel):
    """Identifier of the ``arbitrator_executions`` row a create call inserted."""

    model_config = ConfigDict(extra="allow")

    request_id: int


class UpdatedExecution(BaseModel):
    """Echo of an accepted ``arbitrator_executions`` update."""

    model_config = ConfigDict(extra="allow")

    request_id: int
    status: str


class CreatedCheckerExecution(BaseModel):
    """Identifier of the ``checker_executions`` row a create call inserted."""

    model_config = ConfigDict(extra="allow")

    execution_id: int


class UpdatedCheckerExecution(BaseModel):
    """Echo of an accepted ``checker_executions`` update.

    ``status`` is null when the call changed only other columns, which is what the arbitrator's ``start_time``-only
    write does once it sees a job claimed.
    """

    model_config = ConfigDict(extra="allow")

    execution_id: int
    status: str | None = None


class CompletedCheckerJob(BaseModel):
    """Echo of the terminal write from the pod that owned a job."""

    model_config = ConfigDict(extra="allow")

    execution_id: int
    status: str


class CancelRequestedCheckerJob(BaseModel):
    """Echo of a cancellation left on a running job's row. ``cancel_requested`` is always true here."""

    model_config = ConfigDict(extra="allow")

    execution_id: int
    cancel_requested: bool


# =========================================================================
# List wrappers
# =========================================================================
# Endpoints returning a collection wrap it in a single-key object rather than a bare array, so the payload has
# somewhere to grow. These name that wrapper.


class AdoptedExecutions(BaseModel):
    """Requests one adoption sweep took over."""

    model_config = ConfigDict(extra="allow")

    executions: list[AdoptedExecutionRow] = []


class ClaimedCheckerJobs(BaseModel):
    """Jobs handed over by one claim call. An empty list is an ordinary answer: nothing was waiting."""

    model_config = ConfigDict(extra="allow")

    jobs: list[CheckerJobClaim] = []


class RequeuedCheckerJobs(BaseModel):
    """Jobs one reaper sweep returned to the queue."""

    model_config = ConfigDict(extra="allow")

    requeued: list[RequeuedCheckerJob] = []


class RefusedCheckerExecution(BaseModel):
    """One job whose observed claim was not recorded because the row had already finished."""

    model_config = ConfigDict(extra="allow")

    execution_id: int
    existing_status: str | None = None


class RefusedCheckerExecutions(BaseModel):
    """The stale claims out of one batch. Empty is the healthy answer: every claim was recorded."""

    model_config = ConfigDict(extra="allow")

    refused: list[RefusedCheckerExecution] = []
