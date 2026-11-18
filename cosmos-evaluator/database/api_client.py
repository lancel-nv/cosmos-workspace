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
Async HTTP client for the Database REST API service (orchestration subset).

Provides access to the orchestration tables (clients, arbitrator executions, checker executions) over HTTP via the
Database REST API, plus the job-coordination surface checker pods use to claim, report on and finish their work.

Coordination writes go through ``CheckerJobHandle`` rather than plain client methods. A checker pod behind a gateway
can be superseded while it is still running, so every write it makes has to prove it still owns the job; the handle
carries that proof so no call site has to remember to.

All public methods are ``async def`` and must be awaited. Synchronous CLI entry points wrap their main loops in
``asyncio.run(...)``; FastAPI / async services call the methods directly from inside their event loops, which keeps the
loop unblocked while a database round-trip is in flight.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
import dataclasses
from datetime import datetime
import logging
from typing import Any, Self, TypeVar

import httpx

from database.api_models import (
    AdoptedExecutionRow,
    ArbitratorExecutionRow,
    CheckerExecutionByVideoRow,
    CheckerExecutionRow,
    CheckerExecutionStateRow,
    CheckerJobClaim,
    CheckerJobState,
    ClientRow,
    ExecutionLeaseRenewal,
    RefusedCheckerExecution,
    RequeuedCheckerJob,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class CheckerExecutionWrite:
    """Outcome of an arbitrator-side write to a ``checker_executions`` row.

    Attributes:
        written: Whether the update landed.
        existing_status: When it did not, the terminal status the row already held. Carried because the caller's
            reaction depends on it: a job that finished on its own needs no further action, whereas a write that
            failed for any other reason does.
    """

    written: bool
    existing_status: str | None


def _error_metadata(exc: httpx.HTTPStatusError) -> dict[str, Any]:
    """Best-effort read of the ``metadata`` block from an error envelope.

    Tolerant by design: this is only ever used to enrich handling of a response whose status code has already been
    matched, so a body that is missing, truncated or not JSON at all should degrade to "no detail" rather than
    replace the caller's real outcome with a parse error.
    """
    try:
        body = exc.response.json()
    except ValueError:
        return {}
    metadata = body.get("metadata") if isinstance(body, dict) else None
    return metadata if isinstance(metadata, dict) else {}


# Connect timeout covers TCP connect + TLS handshake.
_DEFAULT_CONNECT_TIMEOUT = 30.0
_DEFAULT_READ_TIMEOUT = 30.0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE = 1.0
_DEFAULT_BACKOFF_FACTOR = 2.0

# Keep idle TLS connections in the pool well beyond the default 5 s so bursty traffic (e.g. the arbitrator writing
# several rows per request) reuses one TLS session instead of re-handshaking each time.
#
# The ceiling is whatever idle timeout the proxy fronting the API enforces: an expiry above it hands out pooled sockets
# the server has already closed, which surfaces as a ``RemoteProtocolError`` on first use rather than as anything
# recognizable. This value must stay low enough that the client is always the side that retires a connection first, so
# raising it is not a free knob.
_DEFAULT_KEEPALIVE_EXPIRY = 120.0
_DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 10
_DEFAULT_MAX_CONNECTIONS = 20

# How long a claim long-poll waits by default, and how much slack its read deadline gets on top. The wait matches the
# server's ceiling; the margin is there so the client gives up only after the server has answered, making a timeout on
# this call mean something is genuinely wrong rather than that the queue was empty.
_DEFAULT_CLAIM_WAIT_SECONDS = 20.0
_CLAIM_TIMEOUT_MARGIN = 10.0

# Connection-level errors that warrant a retry (server never received the request).
_RETRYABLE_CONNECT_EXCEPTIONS = (httpx.ConnectError, httpx.ConnectTimeout)

# HTTP status codes from upstream proxies / cold-starting pods where the request never reached a stable application
# handler. A single replica rolling to a new version answers with these for the seconds its old pod is draining, so
# every deploy produces a burst of them. Retried only for calls a replay cannot change the outcome of.
_RETRYABLE_HTTP_STATUSES = frozenset({502, 503, 504, 530})

_T = TypeVar("_T")


class ApiClient:
    """Async HTTP client for the orchestration subset of the Database REST API.

    Base interface for the arbitrator and checker containers. Provides the HTTP plumbing (connection pooling, retries,
    payload unwrapping) plus the orchestration-table operations (clients, arbitrator_executions, checker_executions).

    Connection-level failures (SSL handshake timeouts, DNS resolution errors) are retried with exponential backoff on
    every call. Transient upstream 5xx responses (502/503/504/530) are additionally retried for calls a replay cannot
    change the outcome of; 4xx and non-transient 5xx propagate immediately. That is a property of the call rather than
    of its HTTP method: reads qualify, and so do the coordination writes, which are conditional updates to explicit
    values or upserts on a natural key. What does not qualify is a call that would acquire something a second time --
    inserting a row whose id the server generates, or claiming and adopting, where a replay whose predecessor actually
    landed takes a *different* set of rows and leaves the first set owned by a caller that never learned about it.

    The underlying ``httpx.AsyncClient`` speaks HTTP/1.1 with a long keep-alive expiry, so idle TLS connections survive
    between bursts of requests without a per-connection stream ceiling to run out of.

    Synchronous callers (CLIs that don't otherwise need an event loop) should wrap their entry points in
    ``asyncio.run(main())`` and ``await`` the methods normally; this is dramatically simpler than the previous sync
    client because the per-call overhead is identical and the lifecycle moves into a single ``async with`` block.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float | httpx.Timeout | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_base: float = _DEFAULT_BACKOFF_BASE,
        backoff_factor: float = _DEFAULT_BACKOFF_FACTOR,
    ) -> None:
        if not isinstance(max_retries, int) or max_retries < 1:
            raise ValueError(f"max_retries must be an int >= 1, got {max_retries!r}")
        if timeout is None:
            timeout = httpx.Timeout(_DEFAULT_READ_TIMEOUT, connect=_DEFAULT_CONNECT_TIMEOUT)
        limits = httpx.Limits(
            max_connections=_DEFAULT_MAX_CONNECTIONS,
            max_keepalive_connections=_DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
            keepalive_expiry=_DEFAULT_KEEPALIVE_EXPIRY,
        )
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"X-API-Key": api_key},
            timeout=timeout,
            limits=limits,
            # HTTP/1.1 deliberately. A gateway retiring an HTTP/2 connection sends GOAWAY carrying the last stream id
            # it will process; httpx retries the streams above that number on a fresh connection by itself, but raises
            # the ones at or below it as RemoteProtocolError. Those cannot be retried here -- the request may already
            # have taken effect -- so on HTTP/2 a pod is guaranteed to lose a request every time it exhausts a
            # connection's stream space. Multiplexing wins nothing back at our request rate, and the TLS session reuse
            # this client actually needs comes from the keepalive expiry above, which HTTP/1.1 honors just as well.
            http2=False,
        )
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_factor = backoff_factor

    async def aclose(self) -> None:
        """Close the underlying HTTP connection pool."""
        await self._client.aclose()

    async def ping(self, timeout: float = 3.0) -> bool:
        """Best-effort reachability probe against the Database REST API ``/health`` endpoint.

        Returns True if the service answers ``200``, False on any connection error, timeout, or non-200 status. Never
        raises and never retries — it is meant to be called from a caller's own ``/health`` handler to surface DB
        reachability cheaply, so a DB outage must not stall or crash the probe. Uses a short timeout that overrides the
        client's default so an unreachable DB fails fast instead of blocking the caller's health response.
        """
        try:
            response = await self._client.get("/health", timeout=timeout)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    async def _request_with_retry(self, fn: Callable[[], Awaitable[_T]], *, retry_safe: bool = False) -> _T:
        """Execute *fn* (an awaitable factory) with retries on transient failures.

        Connection-level failures (SSL handshake timeout, DNS errors, TCP refused) are retried for every call. Transient
        upstream HTTP responses (502/503/504/530) are retried only when ``retry_safe`` is True, meaning a replay of this
        particular call cannot change the outcome. All other HTTP errors (4xx, non-transient 5xx) propagate immediately.
        """
        last_exc: BaseException | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return await fn()
            except _RETRYABLE_CONNECT_EXCEPTIONS as exc:
                last_exc = exc
                if attempt == self._max_retries:
                    break
                delay = self._backoff_base * (self._backoff_factor ** (attempt - 1))
                logger.warning(
                    "Connection failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt,
                    self._max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            except httpx.HTTPStatusError as exc:
                if not retry_safe or exc.response.status_code not in _RETRYABLE_HTTP_STATUSES:
                    raise
                last_exc = exc
                if attempt == self._max_retries:
                    break
                delay = self._backoff_base * (self._backoff_factor ** (attempt - 1))
                logger.warning(
                    "HTTP %d from upstream (attempt %d/%d) — retrying in %.1fs",
                    exc.response.status_code,
                    attempt,
                    self._max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _unwrap(self, response: httpx.Response) -> Any:
        response.raise_for_status()
        return response.json().get("data")

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """Issue a GET request and return the unwrapped ``data`` payload."""

        async def _do() -> Any:
            return self._unwrap(await self._client.get(path, params=params))

        return await self._request_with_retry(_do, retry_safe=True)

    async def _post(
        self, path: str, json_body: dict[str, Any], timeout: float | None = None, retry_safe: bool = False
    ) -> Any:
        """Issue a POST request and return the unwrapped ``data`` payload.

        ``timeout`` overrides the client-wide read deadline for the one call, which an endpoint that deliberately holds
        the request open (a long-poll) needs: the default deadline is shorter than the window such an endpoint waits.

        ``retry_safe`` defaults to False because a POST is the shape most likely to create something, and a caller that
        has not thought about replay should get the conservative behavior.
        """

        async def _do() -> Any:
            if timeout is None:
                return self._unwrap(await self._client.post(path, json=json_body))
            return self._unwrap(await self._client.post(path, json=json_body, timeout=timeout))

        return await self._request_with_retry(_do, retry_safe=retry_safe)

    async def _put(self, path: str, json_body: dict[str, Any], retry_safe: bool = False) -> Any:
        """Issue a PUT request and return the unwrapped ``data`` payload."""

        async def _do() -> Any:
            return self._unwrap(await self._client.put(path, json=json_body))

        return await self._request_with_retry(_do, retry_safe=retry_safe)

    async def _delete(self, path: str) -> Any:
        """Issue a DELETE request and return the unwrapped ``data`` payload."""

        async def _do() -> Any:
            return self._unwrap(await self._client.delete(path))

        return await self._request_with_retry(_do)

    # ------------------------------------------------------------------
    # arbitrator tables: reads
    # ------------------------------------------------------------------

    async def get_execution(self, request_id: int) -> ArbitratorExecutionRow | None:
        """Get a single arbitrator_executions row by request_id (None on 404)."""
        try:
            data = await self._get(f"/arbitrator-executions/{request_id}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        return ArbitratorExecutionRow.model_validate(data) if data else None

    async def get_checker_executions_for_request(self, request_id: int) -> list[CheckerExecutionRow]:
        """List checker_executions rows for an arbitrator request_id.

        Returns rows ordered by start_time. Used by the arbitrator's snapshot endpoint to assemble the per-checker
        progress array.
        """
        data = await self._get(f"/arbitrator-executions/{request_id}/checker-executions")
        return [CheckerExecutionRow.model_validate(row) for row in (data or [])]

    async def get_checker_execution_states_for_request(self, request_id: int) -> list[CheckerExecutionStateRow]:
        """List a request's checker_executions rows with their coordination columns.

        Read once by a replica adopting an orphaned request: the results say what has already been decided and the
        coordination columns say which jobs still have a pod working on them.
        """
        data = await self._get(f"/arbitrator-executions/{request_id}/checker-states")
        return [CheckerExecutionStateRow.model_validate(row) for row in (data or [])]

    async def get_checker_executions_by_video(
        self,
        video_uuid: str,
        checker_name: str | None = None,
        camera_name: str | None = None,
        status: str = "success",
    ) -> list[CheckerExecutionByVideoRow]:
        """Get checker execution results for a video UUID.

        ``camera_name`` filters to a specific full RDS camera stream id; pass ``"null"`` to match rows where
        camera_name is NULL (non-parameterized checkers). Rows are the by-video projection (joined arbitrator
        provenance columns).
        """
        params: dict[str, Any] = {"status": status}
        if checker_name:
            params["checker_name"] = checker_name
        if camera_name is not None:
            params["camera_name"] = camera_name
        data = await self._get(f"/checker-executions/by-video/{video_uuid}", params=params)
        return [CheckerExecutionByVideoRow.model_validate(row) for row in (data or [])]

    # ------------------------------------------------------------------
    # arbitrator tables: writes
    # ------------------------------------------------------------------

    async def record_client(
        self, ip_address: str, name: str | None = None, description: str | None = None
    ) -> ClientRow | None:
        """Upsert a client record by IP address."""
        payload: dict[str, Any] = {"ip_address": ip_address}
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        # Upsert on the client's address, so a replay lands on the same row and returns the same id.
        data = await self._post("/clients", payload, retry_safe=True)
        return ClientRow.model_validate(data) if data else None

    async def create_execution(
        self,
        client_id: int,
        request_json: dict[str, Any],
        version: str,
        commit_sha: str,
        metadata: dict[str, Any] | None = None,
        claimed_by: str | None = None,
        lease_seconds: float | None = None,
    ) -> int | None:
        """Create an arbitrator execution record. Returns the request_id.

        ``claimed_by`` and ``lease_seconds`` record which arbitrator replica is orchestrating the request and for how
        long its claim holds. Omit them and the row is unowned, which no sweeper will adopt.

        They have to arrive together, and passing one alone raises rather than being quietly dropped: a lease with no
        owner would otherwise vanish here and the caller would believe it had claimed a request it does not own.
        """
        if (claimed_by is None) != (lease_seconds is None):
            raise ValueError("create_execution requires claimed_by and lease_seconds together, or neither")

        payload: dict[str, Any] = {
            "client_id": client_id,
            "request_json": request_json,
            "version": version,
            "commit_sha": commit_sha,
        }
        if metadata is not None:
            payload["metadata"] = metadata
        if claimed_by is not None:
            payload["claimed_by"] = claimed_by
            payload["lease_seconds"] = lease_seconds
        # Not retry-safe: the server generates ``request_id``, so a replay whose predecessor landed inserts a second row
        # that nothing will ever renew, and an unrenewed row with an owner is exactly what the sweeper adopts.
        data = await self._post("/arbitrator-executions", payload)
        return data.get("request_id") if data else None

    async def update_execution(
        self,
        request_id: int,
        status: str,
        response_json: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        claimed_by: str | None = None,
    ) -> bool:
        """Update an arbitrator execution's status, response, and duration.

        Passing ``claimed_by`` fences the write to the replica that owns the request; ``False`` then means the request
        was adopted while this replica was working on it, and its aggregation is stale.
        """
        payload: dict[str, Any] = {"status": status}
        if response_json is not None:
            payload["response_json"] = response_json
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        if claimed_by is not None:
            payload["claimed_by"] = claimed_by
        try:
            await self._put(f"/arbitrator-executions/{request_id}", payload, retry_safe=True)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404 and claimed_by is not None:
                return False
            raise
        return True

    async def heartbeat_executions(
        self, request_ids: Sequence[int], claimed_by: str, lease_seconds: float
    ) -> ExecutionLeaseRenewal:
        """Renew this replica's claim on every request it is orchestrating, in one round trip.

        Anything reported under ``lost`` is no longer this replica's to finish: it either belongs to whoever adopted it
        or has already been recorded as terminal.
        """
        # Retry-safe: a replay renews the same leases to the same values and reports the same rows as lost.
        data = await self._post(
            "/arbitrator-executions/heartbeat",
            {"request_ids": list(request_ids), "claimed_by": claimed_by, "lease_seconds": lease_seconds},
            retry_safe=True,
        )
        return ExecutionLeaseRenewal.model_validate(data or {})

    async def adopt_executions(
        self, claimed_by: str, lease_seconds: float, limit: int = 10
    ) -> list[AdoptedExecutionRow]:
        """Claim requests whose orchestrating replica stopped renewing, so this one can rebuild them.

        Safe to call from every replica at once; each caller gets a disjoint set of rows.
        """
        # Not retry-safe, for the same reason as claiming a job: a replay adopts a different set and orphans the first.
        data = await self._post(
            "/arbitrator-executions/adopt",
            {"claimed_by": claimed_by, "lease_seconds": lease_seconds, "limit": limit},
        )
        return [AdoptedExecutionRow.model_validate(row) for row in (data or {}).get("executions", [])]

    async def create_checker_execution(
        self,
        request_id: int,
        checker_name: str,
        request_json: dict[str, Any],
        version: str = "",
        commit_sha: str = "",
        camera_name: str | None = None,
    ) -> int | None:
        """Create a checker execution record. Returns the execution_id.

        ``camera_name`` is the full RDS camera stream id (e.g. ``camera_front_wide_120fov``) for per-camera fan-out
        checkers, or ``None`` for non-parameterized checkers.
        """
        payload: dict[str, Any] = {
            "request_id": request_id,
            "checker_name": checker_name,
            "request_json": request_json,
            "version": version,
            "commit_sha": commit_sha,
        }
        if camera_name is not None:
            payload["camera_name"] = camera_name
        # Not retry-safe, for the same reason as the request row: ``execution_id`` comes from the server.
        data = await self._post("/checker-executions", payload)
        return data.get("execution_id") if data else None

    async def update_checker_execution(
        self,
        execution_id: int,
        status: str | None = None,
        request_json: dict[str, Any] | None = None,
        response_json: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        error_message: str | None = None,
        output_metadata: dict[str, Any] | None = None,
        start_time: datetime | None = None,
    ) -> "CheckerExecutionWrite":
        """Update a checker execution, unless it has already finished.

        All fields are optional so a single call can refine just one column without clobbering the others.
        ``record.enqueue()`` uses this to write the payload the claiming pod reads back out of ``request_json``, and
        the terminal writes use it to record an outcome. The ``in_progress`` refinement has its own batched
        endpoint (:meth:`mark_checker_executions_in_progress`), since the arbitrator observes claims a pass at a time.

        A row that has already reached a terminal status is left alone and reported as a refusal rather than raised
        as an error, because on the arbitrator's write path it is an ordinary and expected outcome: it means the job
        finished between the poll that decided to write and the write itself. Callers branch on it (see
        :class:`CheckerExecutionWrite`), which an exception would push them into using for control flow.

        A missing row still raises, since that is a genuine inconsistency rather than a race the caller can absorb.

        Datetimes are serialized as ISO-8601 strings so the JSON payload is encoder-friendly; the database endpoint
        converts them back via Pydantic.
        """
        payload: dict[str, Any] = {}
        if status is not None:
            payload["status"] = status
        if request_json is not None:
            payload["request_json"] = request_json
        if response_json is not None:
            payload["response_json"] = response_json
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        if error_message is not None:
            payload["error_message"] = error_message
        if output_metadata is not None:
            payload["output_metadata"] = output_metadata
        if start_time is not None:
            payload["start_time"] = start_time.isoformat()
        try:
            await self._put(f"/checker-executions/{execution_id}", payload, retry_safe=True)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 409:
                raise
            return CheckerExecutionWrite(written=False, existing_status=_error_metadata(exc).get("existing_status"))
        return CheckerExecutionWrite(written=True, existing_status=None)

    # ------------------------------------------------------------------
    # checker job coordination
    # ------------------------------------------------------------------

    async def claim_next_checker_jobs(
        self,
        checker_name: str,
        claimed_by: str,
        lease_seconds: float,
        limit: int = 1,
        wait_seconds: float = _DEFAULT_CLAIM_WAIT_SECONDS,
        version: str | None = None,
        commit_sha: str | None = None,
    ) -> list["CheckerJobHandle"]:
        """Ask for up to ``limit`` jobs waiting for ``checker_name``, waiting up to ``wait_seconds`` for one to appear.

        Each returned handle carries the writes that ownership of that job permits. An empty list means nothing was
        queued for the whole window — the normal answer on an idle fleet, and the caller's cue to ask again. ``limit``
        should be the number of jobs the caller can actually start right now, so one round trip fills every free slot;
        a caller with no free slot should not be calling at all.

        ``version`` and ``commit_sha`` are the caller's own build. The server records them on every job claimed, so a
        job's row names the build that ran it; a field left as ``None`` is not sent and leaves the row's value as is.

        Overrides the client's read deadline for this call, since the server holds the request open on purpose. That
        is also why ``wait_seconds`` is checked here rather than left to the server: it sizes the local read deadline
        before the request is sent, so a negative value would time the call out on this side no matter what the server
        would have answered. The upper bounds on all three are the server's to enforce -- restating them here would
        just be a second copy to keep in sync, and the server's 422 already names the field and the limit.
        """
        if limit < 1:
            raise ValueError(f"claim_next_checker_jobs needs a positive limit, got {limit}")
        if wait_seconds < 0:
            raise ValueError(f"claim_next_checker_jobs needs a non-negative wait_seconds, got {wait_seconds}")
        if lease_seconds <= 0:
            raise ValueError(f"claim_next_checker_jobs needs a positive lease_seconds, got {lease_seconds}")

        payload: dict[str, Any] = {
            "checker_name": checker_name,
            "claimed_by": claimed_by,
            "lease_seconds": lease_seconds,
            "limit": limit,
            "wait_seconds": wait_seconds,
        }
        if version is not None:
            payload["version"] = version
        if commit_sha is not None:
            payload["commit_sha"] = commit_sha
        # Not retry-safe: a replay takes a second, disjoint batch and abandons the first, whose rows stay claimed by a
        # caller that never saw them until their lease lapses. Losing a poll costs nothing -- the loop asks again.
        data = await self._post("/checker-executions/claim-next", payload, timeout=wait_seconds + _CLAIM_TIMEOUT_MARGIN)
        rows = (data or {}).get("jobs") or []
        return [CheckerJobHandle(self, CheckerJobClaim.model_validate(row)) for row in rows]

    async def requeue_expired_checker_jobs(self, limit: int = 100) -> list[RequeuedCheckerJob]:
        """Return jobs whose owner stopped renewing its claim to the queue, and report which ones moved.

        Safe to call from every replica without coordination: the sweep is a single conditional UPDATE, so the first
        caller moves the row and the rest match nothing.
        """
        data = await self._post("/checker-executions/reap", {"limit": limit}, retry_safe=True)
        rows = (data or {}).get("requeued") or []
        return [RequeuedCheckerJob.model_validate(row) for row in rows]

    async def request_checker_cancel(self, execution_id: int) -> bool:
        """Ask the pod that owns a job to stop. ``False`` means the job had already finished."""
        try:
            await self._post(f"/checker-executions/{execution_id}/cancel", {}, retry_safe=True)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return False
            raise
        return True

    async def get_checker_execution_states(self, execution_ids: list[int]) -> list[CheckerExecutionStateRow]:
        """Read the coordination state of many jobs in one round trip.

        Batched because the caller is polling everything it owns; one request per replica rather than one per job
        keeps the load proportional to the number of pollers instead of the width of the fan-out.
        """
        if not execution_ids:
            return []
        data = await self._post("/checker-executions/states", {"execution_ids": execution_ids}, retry_safe=True)
        return [CheckerExecutionStateRow.model_validate(row) for row in (data or [])]

    async def mark_checker_executions_in_progress(
        self, execution_ids: list[int], start_time: datetime
    ) -> dict[int, str | None]:
        """Record observed claims on many jobs in one round trip, returning the ones that had already finished.

        The batched counterpart to the ``in_progress`` write in :meth:`update_checker_execution`, paired with
        :meth:`get_checker_execution_states` so a polling pass costs one read and one write regardless of how many
        jobs it is watching.

        Maps each refused ``execution_id`` to the status that won. An id absent from the result was accepted, which
        keeps the caller's rule the same as the single-row path: only a *known* terminal status means the job has
        stopped running, so a row this call never mentions is still running.
        """
        if not execution_ids:
            return {}
        data = await self._post(
            "/checker-executions/in-progress",
            {"execution_ids": execution_ids, "start_time": start_time.isoformat()},
            retry_safe=True,
        )
        rows = (data or {}).get("refused") or []
        refused = [RefusedCheckerExecution.model_validate(row) for row in rows]
        return {row.execution_id: row.existing_status for row in refused}


class JobOwnershipLost(RuntimeError):
    """Raised when a write is rejected because the job has been taken over by another pod.

    The pod that sees this has been superseded: something already decided it was dead and gave its job to somebody
    else. It must abandon the work and clean up rather than retry, because retrying is exactly the behavior the
    fencing rule exists to prevent.
    """


class CheckerJobHandle:
    """Ownership of one checker job, and the only way to write to that job's row.

    Every write a job owner makes has to carry ``claimed_by`` as a predicate. Without it, a pod that was merely slow
    rather than dead comes back after its job has been reassigned and overwrites the replacement's result — with no
    error raised anywhere, because from the database's point of view it is a perfectly ordinary UPDATE. The failure
    is silent, arrives long after the mistake, and looks like a checker producing wrong answers.

    That invariant cannot survive as a convention repeated at every call site, so it is not expressed as one. The
    handle holds the token, injects it into every request, and offers no unfenced write; a caller that has not
    claimed the job has nothing to call.
    """

    def __init__(self, client: ApiClient, claim: CheckerJobClaim) -> None:
        self._client = client
        self._claim = claim

    @property
    def execution_id(self) -> int:
        return self._claim.execution_id

    @property
    def claimed_by(self) -> str:
        return self._claim.claimed_by

    @property
    def attempt(self) -> int:
        """How many times this job has been claimed, including this claim.

        Lets a caller give up on a job that keeps killing whichever pod picks it up, instead of letting it cycle
        through the fleet.
        """
        return self._claim.attempt

    @property
    def request_json(self) -> dict[str, Any] | None:
        """The job's payload as recorded on the row."""
        return self._claim.request_json

    @property
    def checker_name(self) -> str | None:
        """Which checker the job is for. Set when the caller claimed queued work rather than a named job."""
        return self._claim.checker_name

    @property
    def camera_name(self) -> str | None:
        """The camera the job covers, for checkers that fan out per camera."""
        return self._claim.camera_name

    @property
    def request_id(self) -> int | None:
        """The arbitrator request this job belongs to, for correlating logs across the fan-out."""
        return self._claim.request_id

    @property
    def cancel_requested(self) -> bool:
        """Whether a cancellation was already outstanding when this job was claimed.

        A cancel can land while a job is still queued, and the claim does not clear it. Checking this before starting
        the pipeline is what stops a pod from running work that was called off before anybody picked it up.
        """
        return self._claim.cancel_requested

    async def heartbeat(self, lease_seconds: float) -> CheckerJobState:
        """Extend the claim and report whether a cancellation has been requested.

        Raises ``JobOwnershipLost`` if the claim has already expired and the job was taken over.
        """
        payload: dict[str, Any] = {"claimed_by": self.claimed_by, "lease_seconds": lease_seconds}
        data = await self._post_owned("heartbeat", payload)
        return CheckerJobState.model_validate(data)

    async def complete(
        self,
        response_json: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        output_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record this job as successful. Raises ``JobOwnershipLost`` if the job was taken over."""
        await self._terminal("success", response_json, duration_ms, None, output_metadata)

    async def fail(
        self,
        error_message: str,
        duration_ms: int | None = None,
        response_json: dict[str, Any] | None = None,
        output_metadata: dict[str, Any] | None = None,
        timed_out: bool = False,
    ) -> None:
        """Record this job as failed. Raises ``JobOwnershipLost`` if the job was taken over."""
        status = "timeout" if timed_out else "fail"
        await self._terminal(status, response_json, duration_ms, error_message, output_metadata)

    async def _terminal(
        self,
        status: str,
        response_json: dict[str, Any] | None,
        duration_ms: int | None,
        error_message: str | None,
        output_metadata: dict[str, Any] | None,
    ) -> None:
        payload: dict[str, Any] = {"claimed_by": self.claimed_by, "status": status}
        if response_json is not None:
            payload["response_json"] = response_json
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        if error_message is not None:
            payload["error_message"] = error_message
        if output_metadata is not None:
            payload["output_metadata"] = output_metadata
        await self._post_owned("complete", payload)

    async def _post_owned(self, action: str, payload: dict[str, Any]) -> Any:
        """POST a job action, translating the server's "not yours" answer into ``JobOwnershipLost``.

        Both actions are conditional updates fenced on ``claimed_by``, so they are retried through a transient upstream
        failure. That matters most for ``complete``: a lost terminal write leaves a finished job looking abandoned, and
        the row is requeued and the whole check run a second time on another pod once the lease expires.
        """
        try:
            return await self._client._post(
                f"/checker-executions/{self.execution_id}/{action}", payload, retry_safe=True
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                raise JobOwnershipLost(
                    f"Checker execution {self.execution_id} is no longer claimed by {self.claimed_by}"
                ) from exc
            raise
