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

"""Client-agnostic database service: async read/write access to the orchestration tables.

This is the base data-access layer shared by every deployment of the Database REST API. It owns the asyncpg
connection pool and decides what an unavailable pool means for each caller; the statements themselves live in
:mod:`services.database.queries.orchestration`. No HTTP concepts live here; it is pure data access driven by a
Settings instance.
"""

from collections.abc import Sequence
from datetime import datetime
import json
import logging
from typing import Any
from urllib.parse import quote_plus

import asyncpg

from services.database.queries import orchestration
from services.database.settings import Settings

logger = logging.getLogger(__name__)

# Pool bounds when db_config does not set them. One pool serves every domain, so max_size is the replica's whole
# connection footprint; min_size keeps a couple warm so the first request after an idle period skips the handshake.
DEFAULT_POOL_MIN_SIZE = 2
DEFAULT_POOL_MAX_SIZE = 16


class PoolUnavailableError(RuntimeError):
    """Raised when a write needs the asyncpg pool but it is not configured/available.

    Lets the transport layer tell infrastructure unavailability (HTTP 503) apart from a well-formed
    request that simply matched no row (HTTP 404).
    """


def _build_postgres_url(db_config: dict[str, Any]) -> str:
    """Assemble a ``postgresql://`` DSN from a db_config mapping, URL-escaping the credentials."""
    return (
        f"postgresql://{quote_plus(db_config['user'])}:{quote_plus(db_config['password'])}"
        f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )


def _json_encoder(value: Any) -> str:
    """Serialize a value for a ``json``/``jsonb`` bind parameter.

    Tolerant on purpose: a ``dict``/``list`` is serialized with ``json.dumps``, while a value that is already a
    JSON ``str`` is passed through unchanged. This keeps the codec idempotent so a stray pre-serialized write can
    never double-encode a column (there are no bare-string JSON values in this schema, so passthrough is safe).
    """
    return value if isinstance(value, str) else json.dumps(value)


async def _init_json_codecs(conn: asyncpg.Connection) -> None:
    """Register ``json``/``jsonb`` text codecs so columns decode to Python objects and dicts encode transparently.

    Without this, asyncpg hands back ``json``/``jsonb`` columns as raw text, forcing every caller to ``json.loads``
    defensively and to ``json.dumps`` on writes. Installing it on every pooled connection makes the DB layer's
    contract dict-in / dict-out.
    """
    for type_name in ("json", "jsonb"):
        await conn.set_type_codec(
            type_name,
            schema="pg_catalog",
            encoder=_json_encoder,
            decoder=json.loads,
            format="text",
        )


class DatabaseService:
    """Async data-access layer for the orchestration database tables.

    Holds the asyncpg pool and exposes the ``arbitrator_executions``, ``checker_executions``, and ``clients``
    queries. Each method acquires a connection, hands it to the matching function in
    :mod:`services.database.queries.orchestration`, and applies this layer's own policy for an unavailable pool:
    reads and best-effort writes degrade to empty/``None``, while anything a caller must not silently lose raises
    :class:`PoolUnavailableError` so the transport can answer 503 rather than 404.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: asyncpg.Pool | None = None
        self._db_url: str | None = None

    async def init(self) -> None:
        """Initialize the asyncpg connection pool from ``settings.db_config`` (no-op when unconfigured).

        ``pool_min_size`` and ``pool_max_size`` are read from ``db_config`` so an environment sharing a crowded
        instance can be given a smaller ceiling without a code change. See the connection budget in
        ``database/MIGRATIONS.md`` before raising either: the ceiling is spent per replica.
        """
        db_config = self._settings.db_config
        if not db_config:
            logger.warning("No database configured, service will operate without DB")
            return

        min_size = int(db_config.get("pool_min_size", DEFAULT_POOL_MIN_SIZE))
        max_size = int(db_config.get("pool_max_size", DEFAULT_POOL_MAX_SIZE))
        if min_size > max_size:
            raise ValueError(f"pool_min_size ({min_size}) exceeds pool_max_size ({max_size})")

        self._db_url = _build_postgres_url(db_config)
        self._pool = await asyncpg.create_pool(
            self._db_url,
            min_size=min_size,
            max_size=max_size,
            command_timeout=60,
            timeout=db_config.get("connect_timeout", 10),
            init=_init_json_codecs,
        )
        logger.info("asyncpg connection pool initialized (min_size=%d, max_size=%d)", min_size, max_size)

    async def close(self) -> None:
        """Close the asyncpg connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("asyncpg pool closed")

    # =========================================================================
    # Arbitrator tables — reads
    # =========================================================================

    async def get_execution(self, request_id: int) -> dict[str, Any] | None:
        """Get a single arbitrator execution by request_id."""
        if not self._pool:
            logger.debug("asyncpg pool not available, skipping get_execution")
            return None

        async with self._pool.acquire() as conn:
            return await orchestration.get_execution(conn, request_id)

    async def get_checker_executions_for_request(self, request_id: int) -> list[dict[str, Any]]:
        """Get all checker_executions rows for a given arbitrator request_id, ordered by ``start_time``."""
        if not self._pool:
            logger.debug("asyncpg pool not available, skipping get_checker_executions_for_request")
            return []

        async with self._pool.acquire() as conn:
            return await orchestration.get_checker_executions_for_request(conn, request_id)

    async def get_checker_execution_states_for_request(self, request_id: int) -> list[dict[str, Any]]:
        """Get every checker_executions row for a request, coordination columns included.

        Raises ``PoolUnavailableError`` when the pool is unavailable, so an empty result always means the request has
        no checker rows rather than that we failed to look.
        """
        if not self._pool:
            raise PoolUnavailableError("asyncpg pool not available for get_checker_execution_states_for_request")

        async with self._pool.acquire() as conn:
            return await orchestration.get_checker_execution_states_for_request(conn, request_id)

    # =========================================================================
    # Arbitrator tables — writes
    # =========================================================================

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
        """Insert a new arbitrator execution with 'queued' status. Returns request_id.

        ``claimed_by`` and ``lease_seconds`` have to arrive together. An owner with no lease is a row nothing can
        ever adopt, because adoption is driven entirely by expiry; a lease with no owner is a value with nothing to
        fence. Both are silent -- the row inserts and the request only fails to make progress much later -- so they
        are rejected here instead.
        """
        if (claimed_by is None) != (lease_seconds is None):
            raise ValueError("create_execution requires claimed_by and lease_seconds together, or neither")
        lease = (
            orchestration.lease_interval(lease_seconds, caller="create_execution")
            if lease_seconds is not None
            else None
        )

        if not self._pool:
            logger.debug("asyncpg pool not available, skipping create_execution")
            return None

        async with self._pool.acquire() as conn:
            return await orchestration.create_execution(
                conn,
                client_id,
                request_json,
                version,
                commit_sha,
                metadata=metadata,
                claimed_by=claimed_by,
                lease=lease,
            )

    async def update_execution(
        self,
        request_id: int,
        status: str,
        response_json: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        claimed_by: str | None = None,
    ) -> bool:
        """Update an arbitrator execution's status, response, and duration.

        Returns ``True`` when a row was updated, ``False`` when no ``arbitrator_executions`` row matched
        ``request_id`` (so the transport layer can answer 404). Raises ``PoolUnavailableError`` when the
        asyncpg pool is unavailable (transport answers 503).
        """
        if not self._pool:
            raise PoolUnavailableError("asyncpg pool not available for update_execution")

        async with self._pool.acquire() as conn:
            return await orchestration.update_execution(
                conn,
                request_id,
                status,
                response_json=response_json,
                duration_ms=duration_ms,
                claimed_by=claimed_by,
            )

    async def heartbeat_executions(
        self, request_ids: Sequence[int], claimed_by: str, lease_seconds: float
    ) -> orchestration.LeaseRenewal:
        """Push the owning replica's claim forward on every request it is still orchestrating."""
        lease = orchestration.lease_interval(lease_seconds, caller="heartbeat_executions")
        if not request_ids:
            return orchestration.LeaseRenewal(renewed=[], lost=[])
        if not self._pool:
            raise PoolUnavailableError("asyncpg pool not available for heartbeat_executions")

        async with self._pool.acquire() as conn:
            return await orchestration.heartbeat_executions(conn, request_ids, claimed_by, lease)

    async def adopt_executions(self, claimed_by: str, lease_seconds: float, limit: int) -> list[dict[str, Any]]:
        """Take over requests whose orchestrating replica stopped renewing its claim."""
        lease = orchestration.lease_interval(lease_seconds, caller="adopt_executions")
        if not self._pool:
            logger.debug("asyncpg pool not available, skipping adopt_executions")
            return []

        async with self._pool.acquire() as conn:
            return await orchestration.adopt_executions(conn, claimed_by, lease, limit)

    async def create_checker_execution(
        self,
        request_id: int,
        checker_name: str,
        request_json: dict[str, Any],
        version: str = "",
        commit_sha: str = "",
        camera_name: str | None = None,
    ) -> int | None:
        """Insert a new checker execution with 'pending' status. Returns execution_id."""
        if not self._pool:
            logger.debug("asyncpg pool not available, skipping create_checker_execution")
            return None

        async with self._pool.acquire() as conn:
            return await orchestration.create_checker_execution(
                conn,
                request_id,
                checker_name,
                request_json,
                version=version,
                commit_sha=commit_sha,
                camera_name=camera_name,
            )

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
    ) -> orchestration.CheckerExecutionUpdate:
        """Update one or more columns on a checker_executions row, unless the row has already finished.

        The returned outcome distinguishes the two ways a write can fail to land, because the transport layer owes
        them different answers: no row matched ``execution_id`` (404), or the row is already terminal and was left
        that way (409, carrying the status it holds). Raises ``PoolUnavailableError`` when the asyncpg pool is
        unavailable (transport answers 503).
        """
        if not self._pool:
            raise PoolUnavailableError("asyncpg pool not available for update_checker_execution")

        async with self._pool.acquire() as conn:
            return await orchestration.update_checker_execution(
                conn,
                execution_id,
                status=status,
                request_json=request_json,
                response_json=response_json,
                duration_ms=duration_ms,
                error_message=error_message,
                output_metadata=output_metadata,
                start_time=start_time,
            )

    async def mark_checker_executions_in_progress(
        self, execution_ids: list[int], start_time: datetime
    ) -> list[dict[str, Any]]:
        """Record observed claims on many jobs in one write, returning the ones that had already finished.

        Raises ``PoolUnavailableError`` when the pool is unavailable, so an empty result always means every claim was
        recorded rather than that we failed to write. That distinction is the whole point here: the caller treats an
        absent id as still running, so a silent failure would have it start a budget clock on a finished job.
        """
        if not self._pool:
            raise PoolUnavailableError("asyncpg pool not available for mark_checker_executions_in_progress")
        if not execution_ids:
            return []

        async with self._pool.acquire() as conn:
            return await orchestration.mark_checker_executions_in_progress(conn, execution_ids, start_time)

    # =========================================================================
    # Checker job coordination
    # =========================================================================

    async def claim_next_checker_executions(
        self,
        checker_name: str,
        claimed_by: str,
        lease_seconds: float,
        limit: int,
        version: str | None = None,
        commit_sha: str | None = None,
    ) -> list[dict[str, Any]]:
        """Claim up to ``limit`` of the oldest unstarted jobs for one checker. Empty means there is nothing waiting.

        ``version`` / ``commit_sha`` are the claiming pod's build, stamped onto each claimed row; ``None`` leaves the
        row's value unchanged.

        Raises ``PoolUnavailableError`` when the pool is unavailable, so an empty result always means an empty queue.
        """
        if not self._pool:
            raise PoolUnavailableError("asyncpg pool not available for claim_next_checker_executions")
        if limit <= 0:
            return []

        lease = orchestration.lease_interval(lease_seconds, caller="claim_next_checker_executions")
        async with self._pool.acquire() as conn:
            return await orchestration.claim_next_checker_executions(
                conn, checker_name, claimed_by, lease, limit, version=version, commit_sha=commit_sha
            )

    async def requeue_expired_checker_executions(self, limit: int) -> list[dict[str, Any]]:
        """Return jobs whose owner stopped renewing to the queue, so somebody else can pick them up."""
        if not self._pool:
            logger.debug("asyncpg pool not available, skipping requeue_expired_checker_executions")
            return []
        if limit <= 0:
            return []

        async with self._pool.acquire() as conn:
            return await orchestration.requeue_expired_checker_executions(conn, limit)

    async def heartbeat_checker_execution(
        self,
        execution_id: int,
        claimed_by: str,
        lease_seconds: float,
    ) -> dict[str, Any] | None:
        """Extend the owner's lease, returning the job's current control state.

        Returns ``None`` when the caller no longer owns the row, which is the only signal a superseded pod gets that
        it should abandon its work. Callers must treat that as terminal for their attempt rather than retrying.
        """
        if not self._pool:
            raise PoolUnavailableError("asyncpg pool not available for heartbeat_checker_execution")

        lease = orchestration.lease_interval(lease_seconds, caller="heartbeat_checker_execution")
        async with self._pool.acquire() as conn:
            return await orchestration.heartbeat_checker_execution(conn, execution_id, claimed_by, lease)

    async def complete_checker_execution(
        self,
        execution_id: int,
        claimed_by: str,
        status: str,
        response_json: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        error_message: str | None = None,
        output_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Write the owner's terminal state for a job. Returns ``False`` when the caller no longer owns the row."""
        if not self._pool:
            raise PoolUnavailableError("asyncpg pool not available for complete_checker_execution")

        async with self._pool.acquire() as conn:
            return await orchestration.complete_checker_execution(
                conn,
                execution_id,
                claimed_by,
                status,
                response_json=response_json,
                duration_ms=duration_ms,
                error_message=error_message,
                output_metadata=output_metadata,
            )

    async def request_checker_execution_cancel(self, execution_id: int) -> bool:
        """Ask whichever pod owns this job to stop. Returns ``False`` when the job already reached a terminal state."""
        if not self._pool:
            raise PoolUnavailableError("asyncpg pool not available for request_checker_execution_cancel")

        async with self._pool.acquire() as conn:
            return await orchestration.request_checker_execution_cancel(conn, execution_id)

    async def get_checker_execution_states(self, execution_ids: list[int]) -> list[dict[str, Any]]:
        """Read the coordination state of many jobs in one query.

        Raises ``PoolUnavailableError`` when the pool is unavailable, so an empty result always means the ids matched
        no rows rather than that we failed to look.
        """
        if not self._pool:
            raise PoolUnavailableError("asyncpg pool not available for get_checker_execution_states")
        if not execution_ids:
            return []

        async with self._pool.acquire() as conn:
            return await orchestration.get_checker_execution_states(conn, execution_ids)

    async def record_client(
        self, ip_address: str, name: str | None = None, description: str | None = None
    ) -> dict[str, Any] | None:
        """Upsert a client record by IP address."""
        if not self._pool:
            logger.debug("asyncpg pool not available, skipping record_client")
            return None

        async with self._pool.acquire() as conn:
            return await orchestration.record_client(conn, ip_address, name=name, description=description)
