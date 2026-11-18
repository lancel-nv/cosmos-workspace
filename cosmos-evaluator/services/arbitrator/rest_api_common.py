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
Client-agnostic Arbitrator REST API.

This API orchestrates the execution of one or more checks and returns the final aggregated result to the client. The
client-agnostic base supports ``av.objects_single_view``, ``metropolis.hallucination``,
``metropolis.attribute_verification``, and ``metropolis.physical_plausibility``, and records orchestration rows
through the :class:`~database.api_client.ApiClient`. A deployment that needs more subclasses :class:`ArbitratorApp`
(overriding the ``init_api_client`` / ``request_model`` / ``build_checker_request`` / report / finalize seams) to add
further checkers and deployment-specific behavior without forking this module.

The scheduling engine (the per-replica :class:`~services.arbitrator.waiting_room_queue.WaitingRoom`) is fully generic:
it is wired with this app's ``build_checker_request`` and ``_store_checker_result`` callables, so overriding those
instance methods is sufficient to change checker behavior.

The arbitrator never calls a checker. Every job's payload is written to its ``checker_executions`` row, and checker
replicas claim the rows naming them whenever they have room; results, liveness and cancellation all travel back through
the same row. Checker endpoints are still configured, but only so the arbitrator can health-probe a checker at
admission and proxy its ``/config`` — no work is ever sent to one.

NOTE: This module intentionally does NOT use ``from __future__ import annotations``. The route handlers in
:meth:`ArbitratorApp.create_app` use the instance's ``request_model`` (a class bound to a local variable) as a real
function annotation so FastAPI parses each request body against the concrete model; string annotations would break that.
"""

import argparse
import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import os
import random
import signal
import socket
from typing import Any, ClassVar
import uuid

from fastapi import FastAPI, Path, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse
import httpx
import uvicorn

from database.api_client import ApiClient
from database.api_models import (
    AdoptedExecutionRow,
    CheckerExecutionStateRow,
    LostExecutionLease,
    RequeuedCheckerJob,
)
from database.statuses import CHECKER_TERMINAL_STATUSES, EXECUTION_TERMINAL_STATUSES
from services import utils
from services.arbitrator.checker_queue import CheckerJob, CheckerQueue, JobState
from services.arbitrator.models.contract import ArbitratorRequest, ArbitratorResult, CheckerName
from services.arbitrator.models.endpoints import (
    CheckersResponse,
    ConfigResponse,
    DependencyGraphResponse,
    ExecutionSnapshotResponse,
    HealthResponse,
    ProcessAsyncResponse,
)
from services.arbitrator.models.envelopes import (
    ApiErrorEnvelope,
    ExpiredInputUrlError,
    ExpiredInputUrlsErrorEnvelope,
)
from services.arbitrator.records.checker_execution_record import CheckerExecutionRecord
from services.arbitrator.records.execution_record import ExecutionRecord
from services.arbitrator.settings import Settings, get_settings
from services.arbitrator.waiting_room_queue import (
    LIVE_JOB_STATES,
    ActiveRequest,
    WaitingRoom,
    WaitingRoomMetrics,
)
from services.attribute_verification.models import AttributeVerificationCloudRequest
from services.framework.auth import install_api_key_auth
from services.framework.response_formatters.json_response_formatter import JsonResponseFormatter
from services.framework.url.expiry import inspect_signed_url
from services.hallucination.models import HallucinationCloudRequest
from services.objects.single_view.models import ObjectsCloudRequest
from services.physical_plausibility.models import PhysicalPlausibilityCloudRequest
from services.settings_base import LOG_FORMAT, LogLevel, build_log_config
from utils.environment import Environment

logger = logging.getLogger(__name__)

# Base name of the per-camera single-view check.
_SINGLE_VIEW_CHECKER = "av.objects_single_view"
# Base name of the per-camera hallucination check (consumes a per-camera original/reference video).
_HALLUCINATION_CHECKER = "metropolis.hallucination"
# Base name of the per-camera attribute-verification check (consumes only the per-camera augmented video).
_ATTRIBUTE_VERIFICATION_CHECKER = "metropolis.attribute_verification"
# Base name of the per-camera physical-plausibility check (consumes only the augmented video).
_PHYSICAL_PLAUSIBILITY_CHECKER = "metropolis.physical_plausibility"

# How many jobs one reap pass may return to the queue. Every replica reaps on its own jittered interval, so this is a
# per-pass ceiling rather than a per-cycle one, and it exists to keep a large backlog of dead claims from turning into
# one enormous transaction.
JOB_REAP_LIMIT = 50

# Prometheus exposition format v0.0.4 -- the version most Prometheus scrapers expect when negotiating ``text/plain``.
# The version token in the content type lets scrapers detect the textual format (rather than the protobuf format)
# without parsing the body.
_PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

_TAG_DISCOVERY = "Discovery"
_TAG_SUBMISSION = "Submission"
_TAG_OBSERVATION = "Observation"
_TAG_HEALTH = "Health"

_APP_DESCRIPTION = """
The Arbitrator orchestrates per-checker REST services and aggregates their results.

## Request lifecycle

The recommended pattern is **submit + poll**:

1. ``POST /process/async`` — submit a request. Returns ``202 Accepted`` with
   ``{"request_id": <int>}`` immediately and runs the work in the background.
2. ``GET /executions/{request_id}`` — poll for progress. The snapshot includes
   parent + per-checker status; clients should stop polling once
   ``execution.is_terminal`` is ``true``.

## Response envelope

Every endpoint wraps its payload in the standard envelope produced by
``services/framework`` :

```json
{ "success": true, "data": <endpoint-specific>, "timestamp": "..." }
```

Errors use the same shape with ``"success": false`` and an ``error`` object
(``type``, ``message``, etc.).

## Metrics

``GET /metrics`` is the one exception to the envelope contract. It returns
**Prometheus text exposition format** (``text/plain; version=0.0.4``) so it can be scraped directly by a Prometheus
server. Gauges are computed from this replica's in-memory waiting-room state -- each replica reports only its own
admitted requests; cross-replica aggregation is the scraper's responsibility.
""".strip()


@dataclass
class Client:
    """Client information corresponding to the clients table schema."""

    client_id: int | None = None  # Auto-generated by database
    ip_address: str = ""  # Required: IPv4 or IPv6 address
    name: str | None = None  # Optional name/nickname
    description: str | None = None  # Optional description


@dataclass
class CheckerHealth:
    """Health status of a checker."""

    name: CheckerName
    is_healthy: bool
    version: str = ""
    commit_sha: str = ""


@dataclass(frozen=True)
class _ValidatedAdmission:
    """Cached health-probe + queue-build result, threaded from validation into finalization."""

    checkers_to_run: list[CheckerName]
    checker_health_info: dict[str, CheckerHealth]
    queue: CheckerQueue


async def check_service_health(
    client: httpx.AsyncClient, name: str, url: str, max_retries: int = 5, retry_delay: float = 0.5
) -> CheckerHealth:
    """Check if a service is healthy."""
    health_url = f"{url.rstrip('/')}/health"

    for attempt in range(max_retries + 1):
        try:
            response = await client.get(health_url)

            if response.status_code == status.HTTP_200_OK:
                data = response.json().get("data", {})
                is_healthy = data.get("status") == "healthy"
                version = data.get("version", "")
                commit_sha = data.get("git_sha", "")
                if not is_healthy:
                    logger.warning("Checker %s at %s returned unhealthy status: %s", name, health_url, data)
                return CheckerHealth(name=name, is_healthy=is_healthy, version=version, commit_sha=commit_sha)
            elif response.status_code >= 500 and response.status_code < 600 and attempt < max_retries:
                logger.warning(
                    "Checker %s at %s returned %s, retrying (%s/%s)...",
                    name,
                    health_url,
                    response.status_code,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(retry_delay * (2**attempt))
                continue
            else:
                logger.warning("Checker %s at %s returned status code %s", name, health_url, response.status_code)
                return CheckerHealth(name=name, is_healthy=False)
        except Exception as e:  # noqa: BLE001 - a health probe must treat *any* failure as "unhealthy", never escape.
            logger.warning("Failed to connect to checker %s at %s: %s", name, url, e)
            return CheckerHealth(name=name, is_healthy=False)

    return CheckerHealth(name=name, is_healthy=False)


def _store_checker_result(
    results: dict[str, Any],
    job: CheckerJob,
    value: Any,
) -> None:
    """Write ``value`` into the Arbitrator's aggregated ``checker_results`` dict.

    Per-camera (parameterized) jobs are nested under their base checker name, keyed by full camera name:

        {"av.objects_single_view": {"camera_front_wide_120fov": <value>, ...}}

    Non-parameterized jobs are stored flat under their ``job.name``.
    """
    camera_name = job.params.get("camera_name") if job.params else None
    if camera_name:
        base_bucket = results.setdefault(job.base_name, {})
        if not isinstance(base_bucket, dict):
            # Defensive: if a non-parameterized value ever collided with a base name, preserve it under a sentinel so
            # it isn't silently lost.
            logger.warning(
                "Existing non-dict value for base checker %s; replacing with per-camera bucket",
                job.base_name,
            )
            base_bucket = {}
            results[job.base_name] = base_bucket
        base_bucket[camera_name] = value
    else:
        results[job.name] = value


def _match_job(by_name: dict[str, CheckerJob], row: CheckerExecutionStateRow) -> CheckerJob | None:
    """Find the rebuilt queue's job for a persisted row, or ``None`` if the graph no longer produces one.

    A row records the checker and (for fan-out checkers) the camera, which is exactly the pair the queue derives a job
    name from -- so the match is by identity rather than by position, and a graph that has gained or lost jobs since
    admission cannot silently shift results onto the wrong job.
    """
    for job in by_name.values():
        if job.base_name != row.checker_name:
            continue
        if (job.params.get("camera_name") if job.params else None) == row.camera_name:
            return job
    return None


def _restore_job(
    queue: CheckerQueue,
    job: CheckerJob,
    row: CheckerExecutionStateRow,
    aggregated_results: dict[str, Any],
    db_terminalized: set[str],
    in_flight: dict[str, int],
) -> None:
    """Put one job back into the state its row says it reached, and record what that implies for the rebuild.

    The mapping from row status to job state:

    - ``success`` / ``fail`` / ``timeout`` / ``skipped`` — terminal. The result or error is folded into the aggregation
      and the job is marked as already having its terminal row, so nothing rewrites it.
    - ``in_progress`` with an owner — a pod is working on it. Restored as ``IN_PROGRESS`` and reported as in flight, so
      the monitor watches the row instead of the scheduler publishing a second copy.
    - anything else — a ``pending`` row that was never offered, or a ``queued`` row that nobody has claimed yet — left
      as the freshly-built queue has it, so the scheduler publishes it in the ordinary way. Both are safe to re-offer:
      ``enqueue`` rewrites the payload and the status together, so publishing a row that is already ``queued`` is an
      idempotent restatement rather than a second copy of the job.
    """
    status_value = (row.status or "").lower()

    if status_value == "success":
        envelope = row.response_json or {}
        data = envelope.get("data", envelope) if isinstance(envelope, dict) else {}
        queue.restore(job.name, JobState.COMPLETED, result=data, attempt=row.attempt or 0)
        _store_checker_result(aggregated_results, job, data)
        db_terminalized.add(job.name)
        return

    if status_value in CHECKER_TERMINAL_STATUSES:
        error = row.error_message or f"Checker reported status '{status_value}'"
        if status_value == "skipped":
            queue.restore(job.name, JobState.SKIPPED, error=error, attempt=row.attempt or 0)
        else:
            queue.restore(job.name, JobState.FAILED, error=error, attempt=row.attempt or 0)
        _store_checker_result(aggregated_results, job, {"error": error})
        db_terminalized.add(job.name)
        return

    if status_value == "in_progress" and row.claimed_by:
        queue.restore(job.name, JobState.IN_PROGRESS, attempt=row.attempt or 0)
        in_flight[job.name] = row.execution_id


def _render_metrics(metrics: WaitingRoomMetrics) -> str:
    """Render a :class:`WaitingRoomMetrics` snapshot as Prometheus v0.0.4 text exposition.

    Per-checker lines are emitted in sorted ``(checker, state)`` order so the body is byte-stable across scrapes for
    a given snapshot (helps with diffs in test failures and avoids any accidental ordering coupling on scrapers).
    """
    lines: list[str] = [
        (
            "# HELP arbitrator_requests_in_progress Number of admitted requests on this replica where at least "
            "one pod has claimed a job."
        ),
        "# TYPE arbitrator_requests_in_progress gauge",
        f"arbitrator_requests_in_progress {metrics.in_progress}",
        (
            "# HELP arbitrator_requests_queued Number of admitted requests on this replica waiting for the first "
            "pod to claim a job."
        ),
        "# TYPE arbitrator_requests_queued gauge",
        f"arbitrator_requests_queued {metrics.queued}",
        (
            "# HELP arbitrator_checker_jobs Per-checker count of jobs in each live state (pending, ready, queued, "
            "in_progress) across all admitted requests on this replica. The queued and in_progress states carry the "
            "same names as checker_executions.status, so a gauge can be read against the rows behind it."
        ),
        "# TYPE arbitrator_checker_jobs gauge",
    ]
    for checker in sorted(metrics.per_checker_state):
        for state in sorted(metrics.per_checker_state[checker]):
            count = metrics.per_checker_state[checker][state]
            lines.append(f'arbitrator_checker_jobs{{checker="{checker}",state="{state}"}} {count}')
    return "\n".join(lines) + "\n"


class ArbitratorApp:
    """Client-agnostic Arbitrator application.

    Holds the per-replica orchestration state (Database API client, waiting-room scheduler, scheduler task) and builds
    the FastAPI app via :meth:`create_app`. Subclasses override the seam methods (``init_api_client`` /
    ``build_checker_request`` / ``_build_result`` / ``_on_finalize``) plus the ``request_model`` class attribute to add
    checkers and deployment-specific behavior.
    """

    # Request body model parsed by ``POST /process/async``. Subclasses set a model adding their own input fields.
    request_model: type[ArbitratorRequest] = ArbitratorRequest

    # How long a replica keeps working after SIGTERM before it lets the process exit regardless. A replica holds each
    # admitted request's aggregation in its own memory, so one killed mid-flight strands those executions at
    # ``in_progress`` with nobody left to finish them even though the checker work itself survives. Rollouts cause
    # nearly all of these and rollouts are planned, which is what makes draining first worth so much: it turns the
    # common case into no loss at all. Must stay under the platform's termination grace period, since past that the
    # process is killed wherever it happens to be.
    SHUTDOWN_DRAIN_TIMEOUT: ClassVar[float] = 120.0

    # How long an admitting replica's claim on a request holds without renewal, and how often it renews. The lease
    # survives two consecutive missed renewals, so a database blip or a busy event loop does not hand a live request to
    # somebody else. The gap between them is also the recovery latency: a crashed replica's requests sit untouched for
    # up to the lease before another replica can see them as adoptable.
    EXECUTION_LEASE_SECONDS: ClassVar[float] = 90.0
    OWNERSHIP_INTERVAL: ClassVar[float] = 30.0
    OWNERSHIP_JITTER: ClassVar[float] = 0.3

    # Most requests a single sweep will take on. Bounded so a fleet-wide restart is absorbed across several sweeps and
    # replicas instead of landing entirely on whichever replica swept first.
    ADOPTION_BATCH_LIMIT: ClassVar[int] = 5

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings: Settings = settings if settings is not None else self._build_settings()
        self._json_formatter = JsonResponseFormatter()
        self._api_client: ApiClient | None = None
        self._waiting_room: WaitingRoom | None = None
        self._scheduler_task: asyncio.Task[None] | None = None
        self._job_monitor_task: asyncio.Task[None] | None = None
        self._ownership_task: asyncio.Task[None] | None = None
        self._drain_task: asyncio.Task[None] | None = None
        self._app: FastAPI | None = None
        self._version: str = ""
        # Identity this replica claims its requests under. A fresh uuid per process rather than hostname alone: a
        # restarted pod keeps its hostname, and must not be able to renew the lease its dead predecessor took.
        self._replica_id: str = f"{socket.gethostname()}-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # Seams (overridden by subclasses)
    # ------------------------------------------------------------------

    def _build_settings(self) -> Settings:
        """Return the settings instance. Subclasses return their own :class:`Settings` subclass."""
        return get_settings()

    def init_api_client(self) -> ApiClient | None:
        """Initialize the Database API client from settings, or ``None`` when no URL is configured.

        Subclasses return an :class:`ApiClient` subclass carrying extra deployment-specific endpoints.
        """
        if not self._settings.database_api_url:
            logger.info("No database_api_url configured, skipping API client initialization")
            return None
        client = ApiClient(
            base_url=self._settings.database_api_url,
            api_key=self._settings.database_api_key,
        )
        logger.info("Database API client initialized (url=%s)", self._settings.database_api_url)
        return client

    def build_checker_request(
        self,
        job: CheckerJob,
        request: ArbitratorRequest,
        queue: CheckerQueue | None = None,
    ) -> dict:
        """Build the checker-specific request JSON from the arbitrator request and job.

        The client-agnostic base knows how to build the ``av.objects_single_view``, ``metropolis.hallucination``,
        ``metropolis.attribute_verification``, and ``metropolis.physical_plausibility`` payloads. Subclasses override
        this (calling ``super()``) to add their own checkers.

        Raises:
            NotImplementedError: If the checker type is not supported by this app.
        """
        base_name = job.base_name
        if base_name == _SINGLE_VIEW_CHECKER:
            return self._build_single_view_request(job, request).model_dump(mode="json")
        if base_name == _HALLUCINATION_CHECKER:
            return self._build_hallucination_request(job, request).model_dump(mode="json")
        if base_name == _ATTRIBUTE_VERIFICATION_CHECKER:
            return self._build_attribute_verification_request(job, request).model_dump(mode="json")
        if base_name == _PHYSICAL_PLAUSIBILITY_CHECKER:
            return self._build_physical_plausibility_request(job, request).model_dump(mode="json")
        raise NotImplementedError(f"Checker {base_name} not yet implemented")

    def _build_single_view_request(self, job: CheckerJob, request: ArbitratorRequest) -> ObjectsCloudRequest:
        """Build the single-view cloud request from a per-camera job and the arbitrator request."""
        camera_name = job.params["camera_name"]
        augmented_video_url = job.params["augmented_video_url"]
        # The static (CWIP) check needs the per-camera *perspective* control video. The dependency graph resolves it
        # per camera from ``world_model_video_urls`` during fan-out and bakes it into job.params — symmetric with
        # ``augmented_video_url`` — so it is present for two-stream single-view jobs. In camera-only mode there is no
        # world-model video, so the param is absent.
        world_model_video_url = job.params.get("world_model_video_url")
        return ObjectsCloudRequest(
            camera_name=camera_name,
            rds_hq_url=request.inputs.rds_hq_url,
            augmented_video_url=augmented_video_url,
            world_model_video_url=world_model_video_url,
            output_storage_prefix=request.output_storage_prefix,
            config=request.inputs.config,
        )

    def _build_hallucination_request(self, job: CheckerJob, request: ArbitratorRequest) -> HallucinationCloudRequest:
        """Build the hallucination cloud request from a per-camera job and the arbitrator request.

        The hallucination checker has no client-supplied clip identifier on the arbitrator request, so a fresh uuid4 is
        generated per job for result identity / log naming. The per-camera original/reference video is resolved during
        fan-out from ``original_video_urls`` and baked into ``job.params`` (symmetric with ``augmented_video_url``), so
        it is always present for hallucination jobs (cameras without one are SKIPPED at admission and never reach here).
        """
        return HallucinationCloudRequest(
            clip_id=str(uuid.uuid4()),
            original_video_url=job.params["original_video_url"],
            augmented_video_url=job.params["augmented_video_url"],
            output_storage_prefix=request.output_storage_prefix,
            config=request.inputs.config,
        )

    def _build_attribute_verification_request(
        self, job: CheckerJob, request: ArbitratorRequest
    ) -> AttributeVerificationCloudRequest:
        """Build the attribute-verification cloud request from a per-camera job and the arbitrator request.

        Like hallucination, there is no client-supplied clip identifier, so a fresh uuid4 is generated per job. This
        check consumes only the per-camera augmented video.
        """
        return AttributeVerificationCloudRequest(
            clip_id=str(uuid.uuid4()),
            augmented_video_url=job.params["augmented_video_url"],
            output_storage_prefix=request.output_storage_prefix,
            config=request.inputs.config,
        )

    def _build_physical_plausibility_request(
        self, job: CheckerJob, request: ArbitratorRequest
    ) -> PhysicalPlausibilityCloudRequest:
        """Build a per-camera physical-plausibility request from its augmented video."""
        return PhysicalPlausibilityCloudRequest(
            clip_id=str(uuid.uuid4()),
            video_url=job.params["augmented_video_url"],
            output_storage_prefix=request.output_storage_prefix,
            config=request.inputs.config,
        )

    def _rds_hq_required_by(self) -> set[CheckerName]:
        """Base names of the checkers this app cannot run without ``inputs.rds_hq_url``.

        Which checkers those are is a property of the checker set an app builds requests for, not of the request shape,
        so it belongs beside ``build_checker_request``: a subclass that adds a checker consuming the RDS-HQ dataset adds
        it here in the same edit. Anything absent from this set may be requested with ``rds_hq_url`` omitted, which is
        what lets a Metropolis-only request -- or an AV request for a checker that only reads the video -- skip
        producing a dataset zip it would never open.
        """
        return {_SINGLE_VIEW_CHECKER}

    def _build_result(self, aggregated_results: dict[str, Any], request: ArbitratorRequest) -> ArbitratorResult:
        """Build the arbitrator result model from the aggregated checker results. Subclasses add a display widget."""
        return ArbitratorResult(checker_results=aggregated_results)

    async def _on_finalize(self, active: ActiveRequest) -> None:
        """Hook run after the execution row is marked terminal. No-op in the base; subclasses record extra rows."""

    def _read_version(self) -> str:
        """Read the service version string from runfiles (resolved against the build-root ``_main`` repo)."""
        return utils.get_contents_from_runfile("services/arbitrator/version.txt")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _checker_headers(self) -> dict[str, str]:
        """Default headers for outbound checker requests.

        Attaches the per-environment ``X-API-Key`` when one is configured (staging/prod) so checker apps protected by
        the framework auth middleware accept the request. Returns an empty mapping when no key is active (local/dev).
        """
        key = self._settings.active_api_key
        return {"X-API-Key": key} if key else {}

    async def _close_api_client(self) -> None:
        """Close the Database API client (awaits the underlying httpx.AsyncClient pool shutdown)."""
        if self._api_client:
            await self._api_client.aclose()
            self._api_client = None
            logger.info("Database API client closed")

    async def record_client(self, request: Request) -> Client | None:
        """Record client information to the database if not already present."""
        if not self._api_client:
            logger.warning("Database API client not initialized, skipping client recording")
            return None

        client_ip = None
        if request.client:
            client_ip = request.client.host

        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()

        if not client_ip:
            logger.warning("Could not determine client IP address")
            return None

        client_name = request.headers.get("X-Client-Name")
        client_description = request.headers.get("X-Client-Description")

        try:
            result = await self._api_client.record_client(
                ip_address=client_ip,
                name=client_name,
                description=client_description,
            )
            if not result:
                return None

            logger.debug("Recorded client: %s (id: %s)", client_ip, result.client_id)
            return Client(
                client_id=result.client_id,
                ip_address=str(result.ip_address or ""),
                name=result.name,
                description=result.description,
            )
        except Exception:
            logger.exception("Failed to record client")
            return None

    async def _create_execution(self, request: ArbitratorRequest, http_request: Request) -> ExecutionRecord | None:
        """Record the client and create an ExecutionRecord for the request.

        Returns ``None`` if no client could be recorded, the database is not available, or the initial
        ``ExecutionRecord.save()`` failed (in which case the record has no ``request_id`` and is not safe to hand back
        to callers).
        """
        recorded_client = await self.record_client(http_request)
        if not (recorded_client and recorded_client.client_id and self._api_client):
            return None

        execution = ExecutionRecord(
            api_client=self._api_client,
            client_id=recorded_client.client_id,
            request_json=request.model_dump(mode="json"),
            version=self._version,
            commit_sha=utils.get_git_sha() or "",
            claimed_by=self._replica_id,
            lease_seconds=self.EXECUTION_LEASE_SECONDS,
        )
        if not await execution.save():
            return None
        return execution

    # ------------------------------------------------------------------
    # Request ownership: leases, adoption
    # ------------------------------------------------------------------

    async def ownership_loop(self) -> None:
        """Renew this replica's claim on the requests it is running, and take over requests nobody is running.

        The counterpart, one level up, to the durable job monitor: that loop makes checker *work* survive a pod loss,
        this one makes the *orchestration* survive it. A request's aggregation lives in one replica's memory, so a
        replica lost mid-request leaves an execution that no longer has anybody assembling its answer even though the
        checker jobs themselves carry on and finish. The lease is how that becomes visible from the outside, and the
        sweep is how another replica picks the request back up from its rows.

        Renewal and sweeping share one loop and one interval on purpose: they are both cheap single round trips, and
        keeping them in step means one obvious knob relating recovery latency to database load. A pass that raises is
        logged and retried on the next one -- the lease is long enough to survive several failures.

        Public because tests drive a single pass directly rather than waiting out the sleep.
        """
        logger.info("Execution ownership loop started as %s", self._replica_id)
        while True:
            spread = self.OWNERSHIP_INTERVAL * self.OWNERSHIP_JITTER
            await asyncio.sleep(self.OWNERSHIP_INTERVAL + random.uniform(-spread, spread))
            try:
                await self.renew_ownership()
                await self.sweep_adoptable()
            except Exception:
                logger.exception("Execution ownership pass raised; continuing")

    async def renew_ownership(self) -> None:
        """Push forward the claim on every request this replica is still running, in a single round trip.

        One call for the whole set rather than one per request, so the time a pass takes does not grow with how many
        requests this replica is orchestrating. A pass that scaled with load would be slowest exactly when there is most
        to lose, and long enough to let the leases it is renewing expire while it works through them.

        A failure raises out of here rather than being absorbed per request, so the enclosing loop logs it and tries
        again on the next pass. That is the conservative reading: an unreachable database is not evidence that anything
        was adopted, and the lease is sized to outlive several passes.
        """
        if self._waiting_room is None or self._api_client is None:
            return
        active_by_id = {
            request_id: active
            for active in list(self._waiting_room.queue)
            if (request_id := active.execution.request_id) is not None
        }
        if not active_by_id:
            return

        renewal = await self._api_client.heartbeat_executions(
            request_ids=list(active_by_id),
            claimed_by=self._replica_id,
            lease_seconds=self.EXECUTION_LEASE_SECONDS,
        )
        for lost in renewal.lost:
            active = active_by_id.get(lost.request_id)
            if active is None:
                continue
            # This copy keeps running whichever reason applies -- stopping it mid-dispatch is not something the
            # scheduler supports, and it is not what protects the outcome anyway. Standing the record down is: every
            # write it could still make is refused locally, so its stale aggregation cannot land on top of the real one.
            active.execution.stand_down()
            self._log_lost_claim(lost)

    def _log_lost_claim(self, lost: LostExecutionLease) -> None:
        """Report one failed renewal as the specific thing that happened to it."""
        if lost.claimed_by is None and lost.status is None:
            logger.error(
                "Execution row for request_id=%s no longer exists; nothing here can be recorded", lost.request_id
            )
        elif lost.claimed_by != self._replica_id:
            # The ordinary case this machinery exists for: this replica stopped renewing long enough that another one
            # took the request over and is rebuilding it from the rows. The cost is duplicated compute until this copy
            # finishes, which is why the lease is long enough that it takes a real outage to get here.
            logger.error(
                "Lost the claim on request_id=%s to replica %s; its work here is now redundant",
                lost.request_id,
                lost.claimed_by,
            )
        else:
            # Still claimed here, so nobody took it: the row reached a terminal status while this replica was still
            # orchestrating it. That means two copies of one request were running here, which adoption is not
            # supposed to be able to produce -- see the self-exclusion in ``adopt_executions``.
            logger.error(
                "request_id=%s is already terminal (status=%s) but was still being orchestrated here; this replica was "
                "running a duplicate copy of it",
                lost.request_id,
                lost.status,
            )

    async def _reap_expired_jobs(self) -> list[RequeuedCheckerJob]:
        """Return checker jobs whose owning pod stopped renewing to the queue.

        Thin adapter so the waiting room can run the reap on its own schedule without knowing about the API client or
        the batch size. Every replica does this, unconditionally: the jobs that most need reaping belong to requests
        whose own replica is gone, so a reaper that only touched its own jobs would never reach them.
        """
        if self._api_client is None:
            return []
        return await self._api_client.requeue_expired_checker_jobs(limit=JOB_REAP_LIMIT)

    async def sweep_adoptable(self) -> None:
        """Claim requests whose orchestrating replica stopped renewing, and rebuild them here."""
        if self._api_client is None or self._waiting_room is None:
            return
        free = self._waiting_room.free_capacity()
        limit = self.ADOPTION_BATCH_LIMIT if free is None else min(self.ADOPTION_BATCH_LIMIT, free)
        if limit <= 0:
            return

        rows = await self._api_client.adopt_executions(
            claimed_by=self._replica_id, lease_seconds=self.EXECUTION_LEASE_SECONDS, limit=limit
        )
        for row in rows:
            try:
                await self._adopt(row)
            except Exception:
                logger.exception("Failed to adopt request_id=%s; its lease will expire again", row.request_id)

    async def _adopt(self, row: AdoptedExecutionRow) -> None:
        """Rebuild one adopted request and hand it to the scheduler."""
        assert self._waiting_room is not None, "_adopt requires a running waiting_room"
        # A request this replica is already running must never be rebuilt on top of itself: two copies would duplicate
        # every dispatch, and the first to finish would strand the other on a terminal row. ``adopt_executions``
        # excludes this replica's own rows so the ordinary path cannot get here, but a row adopted away and later left
        # stranded by its new owner can come back, and the check is cheap next to the cost of getting it wrong. The
        # lease this sweep just took is left to expire, at which point the request is swept again.
        if any(active.execution.request_id == row.request_id for active in self._waiting_room.queue):
            logger.warning(
                "Adopted request_id=%s but it is already being orchestrated here; not rebuilding a second copy",
                row.request_id,
            )
            return
        if not self._waiting_room.try_admit():
            logger.warning("Adopted request_id=%s but have no room for it; leaving it to expire again", row.request_id)
            return
        admitted = False
        try:
            active, in_flight = await self._rebuild_active_request(row)
            self._waiting_room.adopt(active, in_flight)
            admitted = True
        finally:
            if not admitted:
                self._waiting_room.release()
        logger.info(
            "Adopted request_id=%s: %d job(s) already in flight, %d rebuilt from rows",
            row.request_id,
            len(in_flight),
            len(active.aggregated_results),
        )

    async def _rebuild_active_request(self, row: AdoptedExecutionRow) -> tuple[ActiveRequest, dict[str, int]]:
        """Reconstruct an in-flight request from its own row plus its ``checker_executions`` rows.

        Everything needed is already persisted, which is the whole payoff of running work off rows: the submission is on
        the parent row, and each job's state, result and current owner are on its own row. Nothing is re-probed and no
        health check is repeated -- the set of checkers to run was decided at admission and is read back off the rows
        rather than recomputed, so a checker that has since gone unhealthy does not silently drop out of a request that
        was already admitted with it.

        Returns the rebuilt request and the job-name-to-execution_id map of jobs a pod is still working on.
        """
        assert self._api_client is not None, "_rebuild_active_request requires a Database API client"
        request = self.request_model.model_validate(row.request_json)
        job_rows = await self._api_client.get_checker_execution_states_for_request(row.request_id)

        queue = self._settings.dependency_graph.build_queue(
            sorted({str(job_row.checker_name) for job_row in job_rows if job_row.checker_name}),
            request.inputs.augmented_video_urls,
            world_model_video_urls=request.inputs.world_model_video_urls,
            original_video_urls=request.inputs.original_video_urls,
        )

        execution = ExecutionRecord(
            api_client=self._api_client,
            client_id=row.client_id,
            request_json=row.request_json,
            version=row.version or self._version,
            commit_sha=row.commit_sha or (utils.get_git_sha() or ""),
            claimed_by=self._replica_id,
            lease_seconds=self.EXECUTION_LEASE_SECONDS,
        )
        execution.request_id = row.request_id

        by_name = {job.name: job for job in queue.iter_jobs()}
        checker_executions: dict[str, CheckerExecutionRecord] = {}
        aggregated_results: dict[str, Any] = {}
        db_terminalized: set[str] = set()
        in_flight: dict[str, int] = {}

        for job_row in job_rows:
            job = _match_job(by_name, job_row)
            if job is None:
                # A row whose job the graph no longer produces (the dependency graph or camera config changed under a
                # live request). Nothing can be done with it here; the rebuilt queue simply does not contain it.
                logger.warning(
                    "Adopted request_id=%s has a row for %s/%s that the rebuilt queue has no job for",
                    row.request_id,
                    job_row.checker_name,
                    job_row.camera_name,
                )
                continue
            checker_executions[job.name] = CheckerExecutionRecord.attach(
                api_client=self._api_client,
                request_id=row.request_id,
                execution_id=job_row.execution_id,
                checker_name=str(job_row.checker_name or job.base_name),
                camera_name=job_row.camera_name,
            )
            _restore_job(queue, job, job_row, aggregated_results, db_terminalized, in_flight)

        return (
            ActiveRequest(
                execution=execution,
                request=request,
                queue=queue,
                checker_executions=checker_executions,
                health_info={},
                aggregated_results=aggregated_results,
                completion_future=asyncio.get_running_loop().create_future(),
                finalize=self._finalize_request,
                # The parent row is past ``queued`` if any job ever got as far as a pod, so re-flipping it would be a
                # no-op write. Taken from the row rather than assumed, so a request adopted before its first dispatch
                # still gets the flip when one lands.
                in_progress_marked=row.status == "in_progress",
                db_terminalized=db_terminalized,
            ),
            in_flight,
        )

    async def _gather_healthy_checkers(
        self,
        request: ArbitratorRequest,
    ) -> tuple[list[CheckerName], dict[str, CheckerHealth]]:
        """Probe each candidate checker's ``/health`` endpoint in parallel.

        If ``request.checks`` is empty, every configured checker is probed. Otherwise only the requested checkers are
        probed (and a warning is logged for any names that aren't in the configured endpoints).
        """
        checkers_to_run: list[CheckerName] = []
        checker_health_info: dict[str, CheckerHealth] = {}
        # Health-probe budget per checker is intentionally tight: a single retry with a 0.2 s delay caps total
        # per-call wait at ~0.4 s. Under burst admissions, the 5-retry exponential-backoff default (~15.5 s
        # worst-case) amplifies upstream checker pressure into per-request 503s. Failing fast here lets the client
        # retry quickly against the arbitrator's next probe.
        async with httpx.AsyncClient(timeout=5.0) as client:
            if not request.checks:
                tasks = []
                for name, url in self._settings.checker_endpoints.items():
                    tasks.append(check_service_health(client, name, url, max_retries=2, retry_delay=0.2))
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if not isinstance(result, BaseException) and result.is_healthy:
                        checkers_to_run.append(result.name)
                        checker_health_info[result.name] = result
            else:
                tasks = []
                for name in request.checks:
                    if name in self._settings.checker_endpoints:
                        tasks.append(
                            check_service_health(
                                client, name, self._settings.checker_endpoints[name], max_retries=2, retry_delay=0.2
                            )
                        )
                    else:
                        logger.warning("Requested checker %s not found in configuration", name)
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException):
                        continue
                    if result.is_healthy:
                        checkers_to_run.append(result.name)
                        checker_health_info[result.name] = result
                    else:
                        logger.warning("Requested checker %s is not healthy", result.name)
        return checkers_to_run, checker_health_info

    async def _create_checker_execution_records(
        self,
        queue: CheckerQueue,
        request: ArbitratorRequest,
        execution: ExecutionRecord,
    ) -> dict[str, CheckerExecutionRecord]:
        """Eagerly insert one ``checker_executions`` row per non-terminal job, all born ``pending``.

        ``pending`` rather than ``queued`` because these rows exist to be *found* -- by the snapshot endpoint, and by a
        replica rebuilding this request after an adoption -- not yet to be run. The scheduler offers each job
        separately, via ``record.enqueue``, once it can supply the payload.
        """
        records: dict[str, CheckerExecutionRecord] = {}
        if not (execution.request_id and self._api_client):
            return records

        for job in queue.iter_jobs():
            if job.state in (JobState.SKIPPED, JobState.FAILED, JobState.COMPLETED):
                continue
            if job.state == JobState.PENDING:
                # Dependency-blocked, so its payload is a function of results that do not exist yet. Asking for one
                # raises by construction -- and an aggregator's builder raises ``ValueError`` naming the dependencies
                # it is missing, which the catch-all below would report as an unexpected failure, with a traceback, on
                # every request that includes such a checker. There is simply nothing to record here: ``enqueue``
                # supplies the real payload once the dependencies have produced results.
                request_json = {}
            else:
                try:
                    request_json = self.build_checker_request(job, request, queue)
                except (KeyError, NotImplementedError):
                    # Still expected for a job whose deps are met: a dep-resolved param may be absent (KeyError), or
                    # the checker is built by a subclass that cannot assemble the payload yet (NotImplementedError).
                    request_json = {}
                except Exception:
                    # The row's ``request_json`` is purely informational and the scheduler rebuilds the real payload on
                    # dispatch, so any other construction failure must never break admission — log it and carry on.
                    logger.exception("Unexpected error building informational request_json for %s", job.name)
                    request_json = {}

            camera_name = job.params.get("camera_name") if job.params else None
            record = CheckerExecutionRecord(
                api_client=self._api_client,
                request_id=execution.request_id,
                checker_name=job.base_name,
                request_json=request_json,
                camera_name=camera_name,
            )
            if await record.save():
                records[job.name] = record
            else:
                logger.warning(
                    "Failed to create checker execution row for %s, request_id=%s",
                    job.name,
                    execution.request_id,
                )
        return records

    async def _finalize_request(self, active: ActiveRequest) -> JSONResponse:
        """Build the final JSONResponse and update ``arbitrator_executions`` to its terminal state.

        Invoked by :class:`WaitingRoom` once every job in the request's queue has reached a terminal state. Aggregates
        SKIPPED-job result entries into ``active.aggregated_results``, builds the result model (subclasses attach a
        display widget), finalizes the execution row, and runs the :meth:`_on_finalize` hook (subclasses record extra
        rows there).
        """
        request = active.request
        queue = active.queue
        aggregated_results = active.aggregated_results
        execution = active.execution

        # Synthesize result entries for SKIPPED jobs that the scheduler didn't already record. Per-camera fan-out
        # checkers nest under ``base_name``; flat-key jobs use their job name. Don't overwrite real results.
        for name, job_info in queue.get_results().items():
            if job_info["state"] != "skipped":
                continue
            skipped_job = queue.get_job(name)
            camera_name = skipped_job.params.get("camera_name") if skipped_job.params else None
            if camera_name:
                existing = aggregated_results.get(skipped_job.base_name)
                if isinstance(existing, dict) and camera_name in existing:
                    continue
            elif name in aggregated_results:
                continue
            _store_checker_result(
                aggregated_results,
                skipped_job,
                {"error": f"Skipped: {job_info['error']}"},
            )

        arbitrator_result = self._build_result(aggregated_results, request)
        response_data = arbitrator_result.model_dump(mode="json", exclude_none=True)
        response_metadata: dict[str, Any] | None = None
        try:
            duration_ms = await execution.complete(response_data)
            response_metadata = {"duration_ms": duration_ms}
            await self._on_finalize(active)
        except Exception:
            logger.exception("Failed to finalize execution row for request_id=%s", execution.request_id)

        return await self._json_formatter.format_success(response_data, metadata=response_metadata)

    async def _validate_admission(
        self,
        request: ArbitratorRequest,
    ) -> tuple[_ValidatedAdmission | None, JSONResponse | None]:
        """Run all pre-DB admission gates: probe checker health and build the queue."""
        logger.info("Checker endpoints: %s", list(self._settings.checker_endpoints.keys()))
        logger.info("Checking service health for checkers...")
        checkers_to_run, checker_health_info = await self._gather_healthy_checkers(request)

        if not checkers_to_run:
            error_response = await self._json_formatter.format_error(
                RuntimeError("No healthy checkers available to run"),
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
            return None, error_response

        logger.info("Running checkers: %s", checkers_to_run)

        # A single-view camera missing its per-camera world-model (control) video — or a hallucination camera missing
        # its per-camera original/reference video — does not fail admission: ``build_queue`` inserts that camera's job
        # already in the SKIPPED terminal state, so the remaining cameras/checkers still run.
        queue = self._settings.dependency_graph.build_queue(
            checkers_to_run,
            request.inputs.augmented_video_urls,
            world_model_video_urls=request.inputs.world_model_video_urls,
            original_video_urls=request.inputs.original_video_urls,
        )

        # ``rds_hq_url`` is consumed by some checkers and not others, so it is validated against the checkers this
        # request actually resolved to. The *resolved* queue rather than the pre-resolution ``checkers_to_run``, so
        # transitively auto-included jobs (e.g. single-view pulled in as an aggregator's dependency) are covered too.
        requested = {job.base_name for job in queue.iter_jobs()} & self._rds_hq_required_by()
        if not request.inputs.rds_hq_url and requested:
            error_response = await self._json_formatter.format_error(
                ValueError(f"rds_hq_url is required when these checkers are requested: {sorted(requested)}"),
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            )
            return None, error_response

        return _ValidatedAdmission(checkers_to_run, checker_health_info, queue), None

    async def _reject_for_expired_urls(
        self,
        execution: ExecutionRecord,
        expired_urls: list[dict[str, str]],
    ) -> JSONResponse:
        """Fail the just-minted execution row and return the 422 envelope to the client."""
        assert execution.request_id is not None, "_reject_for_expired_urls requires a minted execution row"
        logger.warning(
            "Rejecting request_id=%s for expired input URLs: %s",
            execution.request_id,
            [entry["field"] for entry in expired_urls],
        )
        try:
            await execution.fail(
                {
                    "rejected": True,
                    "reason": "expired_input_urls",
                    "expired_urls": expired_urls,
                }
            )
        except Exception as exc:
            logger.exception(
                "Failed to mark execution failed for expired-URL rejection (request_id=%s)",
                execution.request_id,
            )
            return await self._json_formatter.format_error(
                exc,
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return await self._json_formatter.format_error(
            ExpiredInputUrlError(execution.request_id, expired_urls),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    async def _finalize_admission(
        self,
        request: ArbitratorRequest,
        execution: ExecutionRecord,
        validated: _ValidatedAdmission,
    ) -> ActiveRequest:
        """Create ``checker_executions`` rows, build the :class:`ActiveRequest`, and consume the reserved slot."""
        assert self._waiting_room is not None, "_finalize_admission requires a running waiting_room"
        checker_executions = await self._create_checker_execution_records(validated.queue, request, execution)
        active = ActiveRequest(
            execution=execution,
            request=request,
            queue=validated.queue,
            checker_executions=checker_executions,
            health_info=dict(validated.checker_health_info),
            aggregated_results={},
            completion_future=asyncio.get_running_loop().create_future(),
            finalize=self._finalize_request,
        )
        self._waiting_room.admit(active)
        return active

    async def _format_queue_full_response(self) -> JSONResponse:
        """503 envelope returned when the waiting-room queue is at capacity."""
        return await self._json_formatter.format_error(
            asyncio.QueueFull("Arbitrator queue is at capacity; retry later."),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    async def _format_draining_response(self) -> JSONResponse:
        """503 envelope returned while this replica is shutting down and no longer taking work."""
        return await self._json_formatter.format_error(
            RuntimeError("Arbitrator replica is shutting down; retry to reach another replica."),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    async def _format_scheduler_not_initialized_response(self) -> JSONResponse:
        """503 envelope returned when the FastAPI lifespan never populated the waiting room."""
        return await self._json_formatter.format_error(
            RuntimeError("Arbitrator scheduler not initialized"),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    # ------------------------------------------------------------------
    # Endpoint implementations (registered by create_app)
    # ------------------------------------------------------------------

    async def _get_checkers(self, http_request: Request) -> JSONResponse:
        await self.record_client(http_request)
        try:
            healthy_checkers = []
            async with httpx.AsyncClient(timeout=5.0, headers=self._checker_headers()) as client:
                tasks = []
                for name, url in self._settings.checker_endpoints.items():
                    tasks.append(check_service_health(client, name, url))
                results = await asyncio.gather(*tasks, return_exceptions=True)
                healthy_checkers = [
                    result.name for result in results if not isinstance(result, BaseException) and result.is_healthy
                ]
            return await self._json_formatter.format_success({"checkers": healthy_checkers})
        except Exception as e:
            logger.exception("Error checking health of endpoints")
            return await self._json_formatter.format_error(e)

    async def _get_dependency_graph(self) -> JSONResponse:
        try:
            all_deps = self._settings.dependency_graph.get_all_deps()
            checkers = {
                name: {
                    "deps": deps,
                    "priority": self._settings.dependency_graph.compute_priority(name),
                }
                for name, deps in all_deps.items()
            }
            return await self._json_formatter.format_success({"checkers": checkers})
        except Exception as e:
            logger.exception("Error getting dependency graph")
            return await self._json_formatter.format_error(e)

    async def _get_config(self, http_request: Request) -> JSONResponse:
        await self.record_client(http_request)
        try:
            aggregated_config = {}
            async with httpx.AsyncClient(timeout=5.0, headers=self._checker_headers()) as client:
                tasks = []
                checker_names = []
                for name, url in self._settings.checker_endpoints.items():
                    config_url = f"{url.rstrip('/')}/config"
                    tasks.append(client.get(config_url))
                    checker_names.append(name)
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for name, result in zip(checker_names, results, strict=True):
                    if isinstance(result, Exception):
                        logger.warning("Failed to get config from checker %s: %s", name, result)
                        aggregated_config[name] = {"error": str(result)}
                    elif isinstance(result, httpx.Response):
                        if result.status_code == status.HTTP_200_OK:
                            try:
                                response_data = result.json()
                                data = response_data.get("data", {}).get("default_config", {}).get(name, {})
                                aggregated_config[name] = data
                            except (ValueError, AttributeError) as e:
                                # ValueError: non-JSON body; AttributeError: JSON payload that is not a dict.
                                logger.warning("Failed to parse config response from checker %s: %s", name, e)
                                aggregated_config[name] = {"error": f"Failed to parse response: {e}"}
                        else:
                            logger.warning("Checker %s returned status code %s", name, result.status_code)
                            aggregated_config[name] = {"error": f"Status {result.status_code}"}
            return await self._json_formatter.format_success({"config": aggregated_config})
        except Exception as e:
            logger.exception("Error fetching checker configurations")
            return await self._json_formatter.format_error(e)

    async def _health_check(self) -> JSONResponse:
        assert self._app is not None
        # Additive DB-reachability signal: a caller can see the database isn't wired *before* submitting an async
        # request that would otherwise 503 at gate 3. The top-level ``status`` stays ``healthy`` so this endpoint's
        # liveness semantics are unchanged. ``ping`` is best-effort with a short timeout and never raises.
        database_status = "unavailable"
        if self._api_client is not None and await self._api_client.ping():
            database_status = "ok"
        return await self._json_formatter.format_success(
            {
                "service": self._app.title,
                "status": "healthy",
                "version": self._app.version,
                "git_sha": utils.get_git_sha(),
                "environment": self._settings.env.value,
                "database": database_status,
                "draining": self._waiting_room is not None and self._waiting_room.is_draining,
            }
        )

    async def _get_metrics(self) -> PlainTextResponse:
        checker_names = list(self._settings.checker_endpoints)
        if self._waiting_room is None:
            per_checker_state = {name: {state.value: 0 for state in LIVE_JOB_STATES} for name in checker_names}
            metrics = WaitingRoomMetrics(in_progress=0, queued=0, per_checker_state=per_checker_state)
        else:
            metrics = self._waiting_room.snapshot_metrics(checker_names)
        return PlainTextResponse(content=_render_metrics(metrics), media_type=_PROMETHEUS_CONTENT_TYPE)

    async def _process_async(self, request: ArbitratorRequest, http_request: Request) -> JSONResponse:
        """Submit an arbitration request and return immediately with ``request_id``."""
        logger.info("Submitting async request: %s", request)

        if self._waiting_room is None:
            return await self._format_scheduler_not_initialized_response()

        # Gate 1: this replica is still taking work, and has room for it. No DB writes if rejected.
        if not self._waiting_room.try_admit():
            if self._waiting_room.is_draining:
                logger.info("Replica is draining; rejecting async request so it lands on another one")
                return await self._format_draining_response()
            logger.warning(
                "Arbitrator queue at capacity (%s); rejecting async request",
                self._waiting_room.capacity,
            )
            return await self._format_queue_full_response()

        reservation_consumed = False
        execution: ExecutionRecord | None = None
        try:
            try:
                # Gate 2: healthy-checker validation + queue build. No DB writes if rejected.
                validated, validation_error = await self._validate_admission(request)
                if validated is None:
                    assert validation_error is not None
                    return validation_error

                # Gate 3: mint the request_id.
                execution = await self._create_execution(request, http_request)
                if execution is None or execution.request_id is None:
                    return await self._json_formatter.format_error(
                        RuntimeError(
                            "Database is not available; cannot submit async request because no request_id can be "
                            "issued. Check arbitrator/database connectivity."
                        ),
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    )

                # Gate 4: reject expired presigned input URLs (auditable as a terminal ``fail`` row).
                expired_urls = _find_expired_urls(request)
                if expired_urls:
                    return await self._reject_for_expired_urls(execution, expired_urls)

                # All gates passed — create child rows and admit.
                try:
                    await self._finalize_admission(request, execution, validated)
                    reservation_consumed = True
                except Exception as exc:
                    logger.exception("Error finalizing admission for request_id=%s", execution.request_id)
                    try:
                        await execution.fail({"error": str(exc)})
                    except Exception:
                        logger.exception("Failed to mark execution failed after finalization error")
                    return await self._json_formatter.format_error(exc)

                response = await self._json_formatter.format_success({"request_id": execution.request_id})
                response.status_code = status.HTTP_202_ACCEPTED
                return response
            except Exception as exc:
                logger.exception("Unexpected error in /process/async admission pipeline")
                if execution is not None and execution.request_id is not None and not reservation_consumed:
                    try:
                        await execution.fail({"error": str(exc)})
                    except Exception:
                        logger.exception(
                            "Failed to mark execution failed after backstop in /process/async admission pipeline"
                        )
                return await self._json_formatter.format_error(exc)
        finally:
            if not reservation_consumed:
                self._waiting_room.release()

    async def _get_execution_snapshot(self, request_id: int) -> JSONResponse:
        """Return a progress snapshot of an arbitration execution by ``request_id``."""
        if not self._api_client:
            return await self._json_formatter.format_error(
                ValueError("Database is not available; cannot fetch execution snapshot"),
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        try:
            execution = await self._api_client.get_execution(request_id)
        except Exception as e:
            logger.exception("Failed to fetch execution %s", request_id)
            return await self._json_formatter.format_error(e, status_code=status.HTTP_502_BAD_GATEWAY)

        if execution is None:
            return await self._json_formatter.format_error(
                LookupError(f"Execution {request_id} not found"),
                status_code=status.HTTP_404_NOT_FOUND,
            )

        try:
            checker_rows = await self._api_client.get_checker_executions_for_request(request_id)
        except Exception as e:
            logger.exception("Failed to fetch checker executions for %s", request_id)
            return await self._json_formatter.format_error(e, status_code=status.HTTP_502_BAD_GATEWAY)

        parent_status = str(execution.status or "pending")
        is_terminal = parent_status in EXECUTION_TERMINAL_STATUSES

        completed = sum(1 for row in checker_rows if str(row.status or "") in CHECKER_TERMINAL_STATUSES)
        total = len(checker_rows)
        response_json: Any = execution.response_json if is_terminal else None
        projected_checker_rows = [row.model_dump(exclude={"response_json"}) for row in checker_rows]

        snapshot = {
            "execution": {
                "request_id": execution.request_id,
                "status": parent_status,
                "is_terminal": is_terminal,
                "timestamp": execution.timestamp,
                "duration_ms": execution.duration_ms,
                "version": execution.version,
                "commit_sha": execution.commit_sha,
                "response_json": response_json,
            },
            "checkers": {
                "completed": completed,
                "total": total,
                "items": projected_checker_rows,
            },
        }

        return await self._json_formatter.format_success(snapshot)

    # ------------------------------------------------------------------
    # Lifespan + app assembly
    # ------------------------------------------------------------------

    @staticmethod
    async def _stop_task(task: asyncio.Task[None] | None, label: str) -> None:
        """Cancel one of the replica's long-running loops and wait for it to unwind."""
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("%s task raised during shutdown", label)

    def _install_drain_on_sigterm(self) -> Callable[[], None]:
        """Take over SIGTERM so a rollout drains this replica before the server stops listening.

        Returns a callable that undoes the installation, so the process is left as it was found. Draining here rather
        than from the lifespan's teardown is the whole point: by the time teardown runs the listener is already closed,
        so a client polling ``GET /executions/{id}`` for a request this replica is still finishing would be refused for
        the length of the drain. Once the drain is done the signal goes on to whoever held it before us, which under
        uvicorn is ``Server.handle_exit`` -- so the ordinary shutdown then proceeds exactly as it would have.

        A second SIGTERM means somebody wants this over with now, so it passes straight through.
        """
        loop = asyncio.get_running_loop()
        previous = signal.getsignal(signal.SIGTERM)
        forwarded = False

        def _forward() -> None:
            """Hand the shutdown on, exactly once however many signals or drains arrive at this."""
            nonlocal forwarded
            if forwarded:
                return
            forwarded = True
            if callable(previous):
                previous(signal.SIGTERM, None)
            else:
                # Nothing to defer to (SIG_DFL / SIG_IGN): re-raise against the default disposition so the process
                # still dies from its own SIGTERM rather than sitting there having swallowed it.
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                signal.raise_signal(signal.SIGTERM)

        def _on_sigterm() -> None:
            if self._drain_task is not None:
                logger.warning("Second SIGTERM during drain; exiting without finishing the remaining requests")
                self._drain_task.cancel()
                _forward()
                return
            logger.info("SIGTERM received; draining before shutdown")
            self._drain_task = loop.create_task(self._drain_then(_forward))

        def _restore() -> None:
            loop.remove_signal_handler(signal.SIGTERM)
            if callable(previous) or isinstance(previous, signal.Handlers):
                signal.signal(signal.SIGTERM, previous)

        try:
            # Not ``signal.signal``: this routes the signal through asyncio's wakeup pipe, so the callback runs as an
            # ordinary loop callback rather than between bytecodes, where starting a task would not be safe.
            loop.add_signal_handler(signal.SIGTERM, _on_sigterm)
        except (NotImplementedError, RuntimeError, ValueError):
            # No signal handling available here (not the main thread, or a platform without it). Nothing to take over,
            # and the lifespan's own bounded drain remains as the safety net.
            logger.info("SIGTERM draining is unavailable in this process; relying on the lifespan drain")
            return lambda: None
        return _restore

    async def _drain_then(self, resume_shutdown: Callable[[], None]) -> None:
        """Stop admitting, wait out the in-flight requests, then let the interrupted shutdown proceed."""
        try:
            await self._drain()
        finally:
            resume_shutdown()

    async def _drain(self) -> None:
        """Stop admitting and give everything already admitted a bounded chance to finish. Idempotent."""
        room = self._waiting_room
        if room is None or room.is_draining:
            return
        room.begin_draining()
        if not room.has_work_in_flight():
            return
        logger.info("Waiting up to %.0fs for in-flight requests to finish", self.SHUTDOWN_DRAIN_TIMEOUT)
        if await room.wait_until_drained(self.SHUTDOWN_DRAIN_TIMEOUT):
            logger.info("All in-flight requests finished; shutting down cleanly")
        else:
            logger.error(
                "Drain budget of %.0fs expired with requests still in flight; their executions will be left at "
                "in_progress with no replica finishing them",
                self.SHUTDOWN_DRAIN_TIMEOUT,
            )

    @asynccontextmanager
    async def _lifespan(self, _: FastAPI) -> AsyncGenerator[None]:
        """Application lifespan manager for service initialization and cleanup."""
        logger.info("Initializing ArbitratorService for %s environment...", self._settings.env.value)

        try:
            self._api_client = self.init_api_client()
        except Exception:
            logger.exception("Failed to initialize Database API client. Client tracking will be disabled.")
            self._api_client = None

        # Without a reachable Database REST API the arbitrator still serves /health and the checker /process paths, but
        # POST /process/async 503s at gate 3 because no request_id can be minted.
        if self._api_client is None:
            logger.warning(
                "Database API client is not configured (COSMOS_EVALUATOR_DATABASE_API_URL is unset or client init "
                "failed). POST /process/async will fail with 503 'Database is not available' until the database is "
                "reachable."
            )
        elif not await self._api_client.ping():
            logger.warning(
                "Database REST API at %s failed a startup health probe; it may not have finished initializing. While "
                "unreachable, POST /process/async will return 503 'Database is not available'.",
                self._settings.database_api_url,
            )

        raw_capacity = getattr(self._settings, "queue_capacity", None)
        capacity = raw_capacity if isinstance(raw_capacity, int) else None
        self._waiting_room = WaitingRoom(
            build_request=self.build_checker_request,
            store_result=_store_checker_result,
            capacity=capacity,
            read_job_states=self._api_client.get_checker_execution_states if self._api_client else None,
            request_job_cancel=self._api_client.request_checker_cancel if self._api_client else None,
            requeue_expired_jobs=self._reap_expired_jobs if self._api_client else None,
            record_claims=self._api_client.mark_checker_executions_in_progress if self._api_client else None,
        )
        self._scheduler_task = asyncio.create_task(self._waiting_room.scheduler_loop())
        self._job_monitor_task = asyncio.create_task(self._waiting_room.job_monitor_loop())
        if self._api_client is not None:
            self._ownership_task = asyncio.create_task(self.ownership_loop())
        restore_sigterm = self._install_drain_on_sigterm()
        logger.info("Service initialized successfully")

        try:
            yield
        finally:
            logger.info("Shutting down service...")
            restore_sigterm()
            # Teardown that did not come through SIGTERM (SIGINT, an embedded app, a test client) stops admitting but
            # does not wait. The drain budget is sized to fit inside the platform's termination grace period, and
            # nothing off the signal path is bounded by that period -- blocking a Ctrl-C or a test client for two
            # minutes would be a bug, not politeness.
            if self._waiting_room is not None:
                self._waiting_room.begin_draining()
            await self._stop_task(self._drain_task, "Drain")
            self._drain_task = None
            # Stopped after the drain, not before it: a draining replica has to keep renewing its claims or the
            # requests it is in the middle of finishing look abandoned and get adopted out from under it. Adoption is
            # already off while draining, since ``free_capacity`` reports nothing free. Once the drain is over anything
            # still unfinished *should* lapse, so another replica picks it up.
            await self._stop_task(self._ownership_task, "Execution ownership")
            self._ownership_task = None
            await self._stop_task(self._scheduler_task, "Scheduler")
            self._scheduler_task = None
            await self._stop_task(self._job_monitor_task, "Job monitor")
            self._job_monitor_task = None
            self._waiting_room = None
        await self._close_api_client()
        logger.info("Service shutdown complete")

    def create_app(self) -> FastAPI:
        """Build and return the FastAPI application with all routes registered."""
        self._version = self._read_version()
        app = FastAPI(
            title="Arbitrator Service API",
            description=_APP_DESCRIPTION,
            version=self._version,
            lifespan=self._lifespan,
            debug=self._settings.env == Environment.LOCAL,
            openapi_tags=[
                {"name": _TAG_SUBMISSION, "description": "Submit arbitration requests for execution."},
                {"name": _TAG_OBSERVATION, "description": "Observe in-flight or completed executions."},
                {
                    "name": _TAG_DISCOVERY,
                    "description": (
                        "Inspect available checkers, their dependency graph, and aggregated default configuration."
                    ),
                },
                {"name": _TAG_HEALTH, "description": "Service health and version metadata."},
            ],
        )
        self._app = app

        # Enforce the per-environment API key on inbound requests (no-op unless settings.active_api_key is set).
        install_api_key_auth(app, self._settings.active_api_key)

        # The request body model is resolved from the instance so a subclass can supply a richer model. The
        # ``process_async`` handler is annotated with the base ``ArbitratorRequest`` for the static type checker, then
        # its runtime ``__annotations__`` are rebound to the concrete model and the route is registered explicitly (not
        # via the decorator) so FastAPI parses each request body against the subclass's model while mypy stays happy.
        request_model = self.request_model

        @app.get(
            "/checkers",
            tags=[_TAG_DISCOVERY],
            summary="List healthy checkers",
            response_model=CheckersResponse,
            responses={500: {"model": ApiErrorEnvelope, "description": "Internal error while polling checker health."}},
        )
        async def get_checkers(http_request: Request) -> JSONResponse:
            return await self._get_checkers(http_request)

        @app.get(
            "/dependency-graph",
            tags=[_TAG_DISCOVERY],
            summary="Get checker dependency graph",
            response_model=DependencyGraphResponse,
            responses={500: {"model": ApiErrorEnvelope, "description": "Internal error while computing the graph."}},
        )
        async def get_dependency_graph() -> JSONResponse:
            return await self._get_dependency_graph()

        @app.get(
            "/config",
            tags=[_TAG_DISCOVERY],
            summary="Get aggregated checker default configurations",
            response_model=ConfigResponse,
            responses={
                500: {"model": ApiErrorEnvelope, "description": "Internal error while aggregating configurations."}
            },
        )
        async def get_config(http_request: Request) -> JSONResponse:
            return await self._get_config(http_request)

        @app.get(
            "/health",
            tags=[_TAG_HEALTH],
            summary="Service health & version",
            response_model=HealthResponse,
        )
        async def health_check() -> JSONResponse:
            return await self._health_check()

        @app.get(
            "/metrics",
            tags=[_TAG_HEALTH],
            summary="Prometheus metrics (per-replica)",
            response_class=PlainTextResponse,
            responses={
                200: {
                    "description": (
                        "Prometheus exposition format (v0.0.4). Gauges reflect this replica's in-memory "
                        "waiting-room state."
                    ),
                    "content": {"text/plain": {"schema": {"type": "string"}}},
                }
            },
        )
        async def get_metrics() -> PlainTextResponse:
            return await self._get_metrics()

        async def process_async(request: ArbitratorRequest, http_request: Request) -> JSONResponse:
            return await self._process_async(request, http_request)

        # Rebind the runtime annotation to the concrete model the instance supplied so FastAPI parses the full body
        # and then register the route explicitly.
        process_async.__annotations__["request"] = request_model
        app.post(
            "/process/async",
            status_code=status.HTTP_202_ACCEPTED,
            tags=[_TAG_SUBMISSION],
            summary="Submit an arbitration request asynchronously",
            response_model=ProcessAsyncResponse,
            responses={
                202: {
                    "model": ProcessAsyncResponse,
                    "description": "Request accepted and is now running in the background.",
                },
                422: {
                    "model": ExpiredInputUrlsErrorEnvelope,
                    "description": (
                        "One or more input URLs carry a presigned signature that has already expired. A terminal "
                        "``fail`` row IS written to ``arbitrator_executions`` and the response includes the "
                        "``request_id`` plus a per-URL breakdown under ``error.details``."
                    ),
                },
                503: {
                    "model": ApiErrorEnvelope,
                    "description": (
                        "Admission was rejected: the waiting room is at capacity, no healthy checkers are currently "
                        "available, or the database is unreachable. No ``arbitrator_executions`` row is written."
                    ),
                },
            },
        )(process_async)

        @app.get(
            "/executions/{request_id}",
            tags=[_TAG_OBSERVATION],
            summary="Get an execution progress snapshot",
            response_model=ExecutionSnapshotResponse,
            responses={
                404: {"model": ApiErrorEnvelope, "description": "No execution exists with the given ``request_id``."},
                502: {
                    "model": ApiErrorEnvelope,
                    "description": "Upstream Database REST API returned an unexpected error.",
                },
                503: {
                    "model": ApiErrorEnvelope,
                    "description": "Database is not available; snapshot cannot be fetched.",
                },
            },
        )
        async def get_execution_snapshot(
            request_id: int = Path(
                ...,
                description=(
                    "Database id of the arbitrator execution to observe. Issued by ``POST /process/async`` at submit "
                    "time."
                ),
                ge=1,
                examples=[42],
            ),
        ) -> JSONResponse:
            return await self._get_execution_snapshot(request_id)

        return app


def _collect_input_urls(request: ArbitratorRequest) -> list[tuple[str, str]]:
    """Return ``[(field_label, url), ...]`` for every fetchable URL on the request.

    Field labels mirror the request body's JSON path so error responses point clients directly at the input that needs
    to be regenerated.
    """
    urls: list[tuple[str, str]] = []
    if request.inputs.rds_hq_url:
        urls.append(("inputs.rds_hq_url", request.inputs.rds_hq_url))
    for camera_name, video_url in request.inputs.augmented_video_urls.items():
        urls.append((f"inputs.augmented_video_urls[{camera_name}]", video_url))
    for camera_name, video_url in (request.inputs.world_model_video_urls or {}).items():
        urls.append((f"inputs.world_model_video_urls[{camera_name}]", video_url))
    for camera_name, video_url in (request.inputs.original_video_urls or {}).items():
        urls.append((f"inputs.original_video_urls[{camera_name}]", video_url))
    return urls


def _find_expired_urls(request: ArbitratorRequest) -> list[dict[str, str]]:
    """Return a structured list of input URLs whose presigned signatures have expired.

    Unsigned URLs (plain ``https://`` without recognized signature query params, or ``s3://`` URIs) are ignored here
    — the storage provider re-presigns those at fetch time. URLs recognized as signed but with unparseable expiry
    parameters are *not* flagged: the rejection is reserved for definitively-expired cases so a malformed-but-fresh
    URL still has a chance to succeed downstream.
    """
    expired: list[dict[str, str]] = []
    for field_label, url in _collect_input_urls(request):
        info = inspect_signed_url(url)
        if info.is_signed and info.is_expired:
            expired.append(
                {
                    "field": field_label,
                    "scheme": info.signature_scheme or "unknown",
                    "expires_at": info.expires_at.isoformat() if info.expires_at else "unknown",
                    "detail": info.reason or "expired",
                }
            )
    return expired


def main() -> None:
    """Build the client-agnostic Arbitrator app and run it under uvicorn.

    Shared entry point for both the local (:rest_api) and container (:rest_api_deployment) binaries; the two differ
    only in whether //utils:git_sha is on the binary's runfiles, controlled in the BUILD file. The app is assembled
    here rather than at module import so that code importing :class:`ArbitratorApp` from this module to subclass it
    does not build a base app as a side effect.
    """
    settings = get_settings()
    app = ArbitratorApp(settings=settings).create_app()

    parser = argparse.ArgumentParser(
        description="Run the Arbitrator API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=os.getenv("UVICORN_HOST", "0.0.0.0"), help="Host to bind to")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("UVICORN_PORT", "8080")),
        choices=range(1, 65536),
        help="Port to bind to",
        metavar="[1-65535]",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level.value,
        choices=[log_level.value for log_level in LogLevel],
        help="Log level",
    )

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT)
    logger.info("Starting Arbitrator API on %s:%s", args.host, args.port)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        log_config=build_log_config(args.log_level),
    )
