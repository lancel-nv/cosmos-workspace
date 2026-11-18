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

"""SQL for the orchestration tables: ``arbitrator_executions``, ``checker_executions``, and ``clients``.

Every function here takes an already-acquired ``asyncpg.Connection`` and returns plain rows. Deciding whether a
connection can be had at all -- and what an unavailable pool means for each caller -- belongs to
:class:`services.database.service.DatabaseService`, which owns the pool; this module assumes it has one and
concerns itself only with the statement and the shape of what comes back.

Taking a connection rather than the pool is what lets a caller compose several of these into one transaction, and
lets the multi-statement flows below (lease renewal, the claim sweeps) keep their statements on a single session.
"""

from collections.abc import Sequence
import dataclasses
from datetime import datetime, timedelta
from typing import Any

import asyncpg

from database.statuses import CHECKER_TERMINAL_STATUSES
from services.database.queries.sql import PartialUpdate


@dataclasses.dataclass(frozen=True)
class LeaseRenewal:
    """Outcome of renewing one replica's claim on a batch of requests.

    Attributes:
        renewed: Requests whose lease was pushed forward.
        lost: One entry per request that could not be renewed, carrying that row's observed ``status`` and
            ``claimed_by`` (both ``None`` if the row is gone). The state is returned rather than just the id because
            renewal fails for two unrelated reasons -- another replica adopted the request, or the request already
            reached a terminal status -- and a caller told only *that* it failed can do nothing but guess which.
    """

    renewed: list[int]
    lost: list[dict[str, Any]]


@dataclasses.dataclass(frozen=True)
class CheckerExecutionUpdate:
    """Outcome of one arbitrator-side write to a ``checker_executions`` row.

    Attributes:
        found: Whether a row with that ``execution_id`` exists at all.
        updated: Whether the write landed. ``found`` without ``updated`` is the terminal guard refusing it.
        existing_status: The status the row held when the write was attempted, or ``None`` if there was no row.
            Carried for the same reason as :attr:`LeaseRenewal.lost`: a caller told only that its write did not
            land cannot tell a finished job from a missing one, and the two call for opposite reactions.
    """

    found: bool
    updated: bool
    existing_status: str | None


def command_tag_rowcount(tag: str) -> int:
    """Return the affected-row count from an asyncpg command tag such as ``"UPDATE 1"`` (0 if unparseable)."""
    try:
        return int(tag.rsplit(" ", 1)[-1])
    except (AttributeError, IndexError, ValueError):
        return 0


def lease_interval(lease_seconds: float, *, caller: str) -> timedelta:
    """Turn a lease duration into the interval a row's ``lease_expires`` is computed from.

    A non-positive lease is rejected rather than stored. ``NOW() + 0`` is a claim that has already expired, so the row
    is adoptable the instant it is written and its owner loses the work to a sweep it never had a chance to beat --
    silently, because every write succeeded. The API boundary bounds this too; this is the guard for anything that
    reaches the service directly.
    """
    if lease_seconds <= 0:
        raise ValueError(f"{caller} requires a positive lease_seconds, got {lease_seconds}")
    return timedelta(seconds=lease_seconds)


def with_isoformat_times(row: dict[str, Any], *columns: str) -> dict[str, Any]:
    """Render the named timestamp columns as ISO strings in place, leaving NULLs alone.

    Rows cross the wire as JSON, and a ``datetime`` is not JSON-serializable. Doing this per named column rather
    than by scanning for datetimes keeps the conversion explicit about which columns are part of the response
    contract.
    """
    for column in columns:
        value = row.get(column)
        if isinstance(value, datetime):
            row[column] = value.isoformat()
    return row


def build_update_execution_sql(
    *,
    request_id: int,
    status: str,
    response_json: dict[str, Any] | None,
    duration_ms: int | None,
    claimed_by: str | None,
) -> tuple[str, list[Any]]:
    """Build the ``arbitrator_executions`` update and its bind parameters.

    Split out from :func:`update_execution` so the generated SQL can be handed to a real Postgres and checked for
    parse errors. The statement is assembled from the arguments that were supplied, so the shape varies per call and
    a mocked pool proves nothing about whether Postgres will accept any given variant.

    Both uses of ``$2`` carry the same explicit ``::text`` cast. Without it Postgres deduces ``character varying``
    from the ``status`` column and ``text`` from the untyped literals in the ``IN`` list, then rejects the statement
    with ``AmbiguousParameterError`` before it ever reaches a row.
    """
    # Terminal updates clear the lease: the request is over, and a row left with an expired lease would keep looking
    # adoptable to every sweeper forever.
    sets = [
        "status = $2::text",
        "lease_expires = CASE WHEN $2::text IN ('success', 'fail') THEN NULL ELSE lease_expires END",
    ]
    params: list[Any] = [request_id, status]
    if response_json is not None:
        sets.append(f"response_json = ${len(params) + 1}")
        params.append(response_json)
        sets.append(f"duration_ms = ${len(params) + 1}")
        params.append(duration_ms or 0)
    elif duration_ms is not None:
        sets.append(f"duration_ms = ${len(params) + 1}")
        params.append(duration_ms)

    where = "request_id = $1"
    if claimed_by is not None:
        where += f" AND claimed_by = ${len(params) + 1}"
        params.append(claimed_by)

    return f"UPDATE arbitrator_executions SET {', '.join(sets)} WHERE {where}", params


def build_update_checker_execution_sql(
    *,
    execution_id: int,
    status: str | None = None,
    request_json: dict[str, Any] | None = None,
    response_json: dict[str, Any] | None = None,
    duration_ms: int | None = None,
    error_message: str | None = None,
    output_metadata: dict[str, Any] | None = None,
    start_time: datetime | None = None,
) -> tuple[str, list[Any]] | None:
    """Build the guarded ``checker_executions`` update and its bind parameters, or ``None`` if nothing is to be written.

    Split out from :func:`update_checker_execution` for the same reason as :func:`build_update_execution_sql`: the
    statement's shape varies with which arguments were supplied, so it has to be handed to a real Postgres and
    checked for parse errors rather than exercised through a mock that would accept anything.

    Two shapes, distinguished by whether a terminal payload came with the status:

    - **Terminal update** — ``status`` plus at least one of ``response_json`` / ``duration_ms`` / ``error_message`` /
      ``output_metadata``, from ``record.complete()`` / ``fail()`` / ``timeout()`` / ``skipped()``. Replaces the row's
      whole outcome, so every payload column is written even when the caller passed nothing for it, and ``end_time``
      is stamped to now. A requeued job's second attempt must not report success while still carrying the first
      attempt's ``error_message``.
    - **Partial update** — any other subset, from ``record.enqueue()`` (status + ``request_json``). Touches only the
      supplied columns, leaving the terminal ones for whoever actually finishes the job. The ``status`` +
      ``start_time`` form is still supported here, but the arbitrator issues it through
      :func:`mark_checker_executions_in_progress` instead, a batch at a time.

    Both shapes carry the same guard (see :func:`_guard_terminal_status`).
    """
    is_terminal_update = status is not None and (
        response_json is not None or duration_ms is not None or error_message is not None or output_metadata is not None
    )

    if is_terminal_update:
        # ``end_time`` is unconditional: the row's close time is the moment the outcome landed, which is now.
        update = PartialUpdate("end_time = NOW()")
        update.set("status", status)
        # Supported alongside a terminal update for defensive callers; today none exercises it.
        update.set("start_time", start_time)
        update.set("duration_ms", duration_ms or 0)
        update.set("response_json", response_json if response_json is not None else {})
        # Cleared rather than left alone when absent -- see :meth:`PartialUpdate.set_always`.
        update.set_always("error_message", error_message)
        update.set_always("output_metadata", output_metadata)
    else:
        update = PartialUpdate()
        update.set_all(
            [
                ("status", status),
                ("start_time", start_time),
                ("request_json", request_json),
                ("duration_ms", duration_ms),
                ("response_json", response_json),
                ("error_message", error_message),
                ("output_metadata", output_metadata),
            ]
        )
        if not update.has_columns():
            return None

    assignments, params = update.parts()
    return _guard_terminal_status(assignments, params, execution_id=execution_id, incoming_status=status)


def _guard_terminal_status(
    assignments: str, params: list[Any], *, execution_id: int, incoming_status: str | None
) -> tuple[str, list[Any]]:
    """Wrap a ``checker_executions`` ``SET`` list in the guard that refuses to write a finished row backwards.

    The arbitrator decides what to write from a snapshot of the row taken at the top of a polling pass, and a pass
    over many jobs takes long enough that the row can reach its terminal state in between. The write it then issues
    is correct for the row it read and wrong for the row that exists, and the damage is not a lost update but a
    stuck one: a job walked back to ``in_progress`` has already had its result discarded, keeps a NULL lease that
    puts it out of the reaper's reach, and never reappears as terminal for the monitor still waiting on it.

    So the row's own state decides, not the caller: a terminal row accepts no further writes through this path. Two
    consequences worth being explicit about, because both look like the guard misfiring:

    - **A repeat of the same status is allowed through**, since the client sends this update as retry-safe and a
      retry that lands after a successful write must report success rather than a refusal it would then have to
      interpret.
    - **A refusal is the right answer even when the caller had good reason to write.** The one flow that hits this
      legitimately is the arbitrator's job budget expiring at the moment the pod writes its own outcome; the pod's
      diagnosis is the better record, which is what the guard keeps.

    The pod's own write path (:func:`complete_checker_execution`) does not come through here. It is fenced on
    ``claimed_by`` instead, which is the right test for it: a pod is authoritative about the job it still owns.

    The predicate tests ``c.status`` -- the row the UPDATE is holding -- rather than ``current.status`` from the CTE,
    which would reintroduce the same defect in miniature. The CTE is fixed at the statement's snapshot, so a pod
    committing between that snapshot and this statement taking the row lock would be overwritten anyway: under READ
    COMMITTED the UPDATE re-checks its qual against the new row version, and a qual reading a materialized CTE has
    nothing to re-check. ``current`` is kept only to report what the row held and to tell a missing row from a
    refused one.

    ``$N::text`` casts are explicit for the reason :func:`build_update_execution_sql` documents -- Postgres would
    otherwise deduce ``character varying`` from the column and ``text`` from the array, and reject the statement as
    ambiguous before reaching a row.
    """
    key = len(params) + 1
    terminal = len(params) + 2
    incoming = len(params) + 3
    statement = f"""
        WITH current AS (
            SELECT execution_id, status FROM checker_executions WHERE execution_id = ${key}
        ), updated AS (
            UPDATE checker_executions AS c
            SET {assignments}
            FROM current
            WHERE c.execution_id = current.execution_id
              AND (c.status <> ALL(${terminal}::text[]) OR c.status = ${incoming}::text)
            RETURNING c.execution_id
        )
        SELECT current.status AS existing_status, EXISTS (SELECT 1 FROM updated) AS updated
        FROM current
    """
    # Sorted so the statement is stable across runs, which keeps it comparable in logs and in the parse test.
    return statement, [*params, execution_id, sorted(CHECKER_TERMINAL_STATUSES), incoming_status]


# Module-level so the image test can hand it to a real Postgres, for the reason
# :func:`build_update_checker_execution_sql` is a builder rather than an inline string: a statement only a mocked
# connection ever sees is a statement nothing has checked for parse errors. Fixed shape, so a constant suffices.
MARK_CHECKER_EXECUTIONS_IN_PROGRESS_SQL = """
    WITH current AS (
        SELECT execution_id, status FROM checker_executions WHERE execution_id = ANY($1::int[])
    ), updated AS (
        UPDATE checker_executions AS c
        SET status = 'in_progress',
            start_time = $2
        FROM current
        WHERE c.execution_id = current.execution_id
          AND (c.status <> ALL($3::text[]) OR c.status = 'in_progress')
        RETURNING c.execution_id
    )
    SELECT current.execution_id, current.status AS existing_status
    FROM current
    WHERE NOT EXISTS (SELECT 1 FROM updated WHERE updated.execution_id = current.execution_id)
    ORDER BY current.execution_id
"""


async def mark_checker_executions_in_progress(
    conn: asyncpg.Connection, execution_ids: list[int], start_time: datetime
) -> list[dict[str, Any]]:
    """Record observed claims on many jobs at once, and report the ones already finished.

    The batched sibling of :func:`update_checker_execution` for the one write the arbitrator issues per job per
    polling pass. It exists because the read it pairs with is already batched
    (:func:`get_checker_execution_states`), and a pass that reads every tracked job in one query and then writes
    them back one at a time scales with fan-out width after all -- the very property the batched read was built to
    avoid. One replica holding a large batch turned that into passes measured in minutes, which is the window the
    guard below then has to defend.

    Returns one entry per **refused** row rather than per requested id: the caller's question is which of its
    observed claims turned out to be stale, and on a healthy pass the answer is none. A row that matched no id is
    absent from the result and so reads as accepted, matching :meth:`CheckerExecutionRecord.in_progress`'s rule that
    only a *known* terminal status means the job has stopped running.

    ``start_time`` is one value for the whole batch, not one per row. It is the moment the pass observed the claims,
    which is what this column is refined to mean; per-row timestamps would imply the arbitrator knows when each pod
    picked its job up, and it does not.

    Carries the same guard as the single-row path, for the same reason and with the same same-status allowance -- a
    job requeued and re-claimed is legitimately walked to ``in_progress`` again, which re-refines its start time to
    the latest claim. See :func:`_guard_terminal_status`.
    """
    rows = await conn.fetch(
        MARK_CHECKER_EXECUTIONS_IN_PROGRESS_SQL,
        execution_ids,
        start_time,
        sorted(CHECKER_TERMINAL_STATUSES),
    )
    return [dict(row) for row in rows]


# =========================================================================
# arbitrator_executions — reads
# =========================================================================


async def get_execution(conn: asyncpg.Connection, request_id: int) -> dict[str, Any] | None:
    """Get a single arbitrator execution by request_id."""
    row = await conn.fetchrow("SELECT * FROM arbitrator_executions WHERE request_id = $1", request_id)
    if not row:
        return None
    return with_isoformat_times(dict(row), "timestamp")


async def get_checker_executions_for_request(conn: asyncpg.Connection, request_id: int) -> list[dict[str, Any]]:
    """Get all checker_executions rows for a given arbitrator request_id.

    Returns rows ordered by ``start_time`` so the caller can render per-checker progress in the order the jobs were
    claimed. Used by the arbitrator's ``GET /executions/{request_id}`` snapshot endpoint to assemble the full
    execution-state response.
    """
    rows = await conn.fetch(
        """
        SELECT execution_id, request_id, checker_name, camera_name, status,
               start_time, end_time, duration_ms, error_message, version, commit_sha,
               output_metadata, response_json
        FROM checker_executions
        WHERE request_id = $1
        ORDER BY start_time ASC, execution_id ASC
        """,
        request_id,
    )
    return [with_isoformat_times(dict(row), "start_time", "end_time") for row in rows]


async def get_checker_execution_states_for_request(conn: asyncpg.Connection, request_id: int) -> list[dict[str, Any]]:
    """Get every checker_executions row for a request, coordination columns included.

    The wider sibling of :func:`get_checker_executions_for_request`: a replica adopting an orphaned request rebuilds
    its per-job state from these rows and needs both the results (to reassemble what has already been decided) and
    the coordination columns (to tell a job whose pod is still working from one that nobody owns).
    """
    rows = await conn.fetch(
        """
        SELECT execution_id, request_id, checker_name, camera_name, status, claimed_by, lease_expires,
               attempt, cancel_requested, start_time, end_time, duration_ms, error_message,
               output_metadata, response_json
        FROM checker_executions
        WHERE request_id = $1
        ORDER BY execution_id
        """,
        request_id,
    )
    return [with_isoformat_times(dict(row), "lease_expires", "start_time", "end_time") for row in rows]


# =========================================================================
# arbitrator_executions — writes
# =========================================================================


async def create_execution(
    conn: asyncpg.Connection,
    client_id: int,
    request_json: dict[str, Any],
    version: str,
    commit_sha: str,
    metadata: dict[str, Any] | None = None,
    claimed_by: str | None = None,
    lease: timedelta | None = None,
) -> int | None:
    """Insert a new arbitrator execution with 'queued' status. Returns request_id.

    ``claimed_by`` records the arbitrator replica that will orchestrate this request, and ``lease`` how long that
    claim is good for before another replica may adopt it. Omitting both creates an unowned row, which no sweeper
    will ever adopt -- correct for a caller that is not an arbitrator replica.
    """
    row = await conn.fetchrow(
        """
        INSERT INTO arbitrator_executions
            (client_id, timestamp, status, request_json, response_json, duration_ms, metadata,
             version, commit_sha, claimed_by, lease_expires)
        VALUES ($1, NOW(), 'queued', $2, '{}'::json, 0, $3, $4, $5, $6,
                CASE WHEN $7::interval IS NULL THEN NULL ELSE NOW() + $7::interval END)
        RETURNING request_id
        """,
        client_id,
        request_json,
        metadata,
        version,
        commit_sha,
        claimed_by,
        lease,
    )
    return row["request_id"] if row else None


async def update_execution(
    conn: asyncpg.Connection,
    request_id: int,
    status: str,
    response_json: dict[str, Any] | None = None,
    duration_ms: int | None = None,
    claimed_by: str | None = None,
) -> bool:
    """Update an arbitrator execution's status, response, and duration. False when no row matched.

    Passing ``claimed_by`` fences the write to the replica that owns the request, so ``False`` also covers "you were
    adopted while you were working". A replica that has lost the row must stop rather than finish, or its stale
    aggregation overwrites the adopter's.
    """
    sql, params = build_update_execution_sql(
        request_id=request_id,
        status=status,
        response_json=response_json,
        duration_ms=duration_ms,
        claimed_by=claimed_by,
    )
    result = await conn.execute(sql, *params)
    return command_tag_rowcount(result) > 0


async def heartbeat_executions(
    conn: asyncpg.Connection, request_ids: Sequence[int], claimed_by: str, lease: timedelta
) -> LeaseRenewal:
    """Push the owning replica's claim forward on every request it is still orchestrating.

    Fenced for the same reason the terminal update is: two replicas aggregating one request would each write a
    partial answer over the other's.

    Takes the whole batch in one statement because renewal is what keeps a replica's work alive, and a round trip
    per request makes the time to renew scale with how many requests the replica is running. That is backwards: the
    busiest replica has the most claims to lose and takes the longest to renew them, and past a few dozen requests
    on a slow link a single pass can take longer than the lease it is renewing -- so the earliest rows in the batch
    expire while the pass is still working through the rest, and the replica's own adoption sweep then finds them.
    """
    # Deduplicated so a caller that somehow holds one request twice gets one entry per id back, and the ``lost``
    # bookkeeping below cannot report the same row repeatedly.
    ids = list(dict.fromkeys(request_ids))
    rows = await conn.fetch(
        """
        UPDATE arbitrator_executions
        SET lease_expires = NOW() + $3::interval
        WHERE request_id = ANY($1::int[]) AND claimed_by = $2 AND status IN ('queued', 'in_progress')
        RETURNING request_id
        """,
        ids,
        claimed_by,
        lease,
    )
    renewed = [row["request_id"] for row in rows]
    renewed_ids = set(renewed)
    missed = [request_id for request_id in ids if request_id not in renewed_ids]
    lost: list[dict[str, Any]] = []
    if missed:
        # Only when something failed, which is the rare path: the common pass renews everything and stops here.
        observed = await conn.fetch(
            "SELECT request_id, status, claimed_by FROM arbitrator_executions WHERE request_id = ANY($1::int[])",
            missed,
        )
        by_id = {row["request_id"]: row for row in observed}
        lost = [
            {
                "request_id": request_id,
                "status": by_id[request_id]["status"] if request_id in by_id else None,
                "claimed_by": by_id[request_id]["claimed_by"] if request_id in by_id else None,
            }
            for request_id in missed
        ]

    return LeaseRenewal(renewed=renewed, lost=lost)


async def adopt_executions(
    conn: asyncpg.Connection, claimed_by: str, lease: timedelta, limit: int
) -> list[dict[str, Any]]:
    """Take over requests whose orchestrating replica stopped renewing its claim.

    ``FOR UPDATE SKIP LOCKED`` is what makes it safe to run this on every replica at once: each sweeper takes a
    disjoint set of rows and none of them block on each other. ``limit`` bounds how much one replica absorbs in a
    single sweep, so a fleet-wide restart spreads the recovery instead of landing it all on whoever swept first.

    A replica never adopts its own rows. Without that exclusion a replica whose renewal pass ran late would sweep up
    requests it is still running and hand itself a second copy to orchestrate: the claim would not change hands, so
    nothing would look wrong, and the two copies would duplicate every checker dispatch until whichever finished
    first wrote the terminal status -- at which point the other's next renewal would fail and look like a takeover
    by a replica that does not exist. ``IS DISTINCT FROM`` rather than ``<>`` so the comparison still holds when
    ``claimed_by`` is NULL; such rows carry no lease and are excluded above regardless.

    Returns enough of each row to rebuild the orchestration: ``request_json`` is the original submission, and the
    request's ``checker_executions`` rows supply everything that has happened since.
    """
    async with conn.transaction():
        rows = await conn.fetch(
            """
            WITH stale AS (
                SELECT request_id
                FROM arbitrator_executions
                WHERE status IN ('queued', 'in_progress')
                  AND lease_expires IS NOT NULL
                  AND lease_expires < NOW()
                  AND claimed_by IS DISTINCT FROM $1
                ORDER BY lease_expires
                LIMIT $3
                FOR UPDATE SKIP LOCKED
            )
            UPDATE arbitrator_executions AS a
            SET claimed_by = $1,
                lease_expires = NOW() + $2::interval
            FROM stale
            WHERE a.request_id = stale.request_id
            RETURNING a.request_id, a.client_id, a.status, a.request_json, a.claimed_by, a.lease_expires,
                      a.version, a.commit_sha
            """,
            claimed_by,
            lease,
            limit,
        )

    return [with_isoformat_times(dict(row), "lease_expires") for row in rows]


# =========================================================================
# checker_executions — writes
# =========================================================================


async def create_checker_execution(
    conn: asyncpg.Connection,
    request_id: int,
    checker_name: str,
    request_json: dict[str, Any],
    version: str = "",
    commit_sha: str = "",
    camera_name: str | None = None,
) -> int | None:
    """Insert a new checker execution with 'pending' status. Returns execution_id.

    Deliberately *not* 'queued': the row is inserted at admission carrying a placeholder payload, and 'queued' is the
    status :func:`claim_next_checker_executions` selects on. Inserting straight into 'queued' would offer every job to
    the fleet the instant its row existed -- before the arbitrator has written the payload the job is supposed to run.
    A pod claiming that row either rejects a payload that is still empty (a dependent job, whose inputs are not
    knowable until its dependencies finish) or starts work from a payload that is about to be rewritten under it.
    'pending' is the state where a row exists but has not been offered, so ``CheckerExecutionRecord.enqueue`` writing
    the payload is what makes the job claimable.

    ``camera_name`` is the full RDS camera stream id (e.g. ``camera_front_wide_120fov``) for per-camera fan-out
    checkers (e.g. ``av.objects_single_view``). It is ``None`` for non-parameterized checkers, such as cross-camera
    aggregators.
    """
    row = await conn.fetchrow(
        """
        INSERT INTO checker_executions
            (request_id, checker_name, camera_name, start_time, end_time, duration_ms,
             status, request_json, response_json, version, commit_sha)
        VALUES ($1, $2, $3, NOW(), NOW(), 0, 'pending', $4, '{}'::json, $5, $6)
        RETURNING execution_id
        """,
        request_id,
        checker_name,
        camera_name,
        request_json,
        version,
        commit_sha,
    )
    return row["execution_id"] if row else None


async def update_checker_execution(
    conn: asyncpg.Connection,
    execution_id: int,
    status: str | None = None,
    request_json: dict[str, Any] | None = None,
    response_json: dict[str, Any] | None = None,
    duration_ms: int | None = None,
    error_message: str | None = None,
    output_metadata: dict[str, Any] | None = None,
    start_time: datetime | None = None,
) -> CheckerExecutionUpdate:
    """Update one or more columns on a checker_executions row, unless the row has already finished.

    See :func:`build_update_checker_execution_sql` for the two update shapes and
    :func:`_guard_terminal_status` for why a finished row is left alone.

    The three outcomes are distinct because the callers react to them differently: the row is missing
    (``found=False``), the row is finished and kept that way (``found=True, updated=False``, with the status it
    holds), or the write landed. The no-op case -- no columns supplied -- reports success on an existing row, since
    there was nothing to refuse, and costs only an existence lookup rather than an update.
    """
    built = build_update_checker_execution_sql(
        execution_id=execution_id,
        status=status,
        request_json=request_json,
        response_json=response_json,
        duration_ms=duration_ms,
        error_message=error_message,
        output_metadata=output_metadata,
        start_time=start_time,
    )
    if built is None:
        # Nothing to write means nothing to refuse, but the row still has to exist for this to be a no-op rather
        # than a write to a job that is not there. ``found`` means "a row with that execution_id exists at all", so
        # answering True without looking would report a successful write for a missing row -- a 200 where every
        # other shape of this call gives the caller its documented 404.
        exists = await conn.fetchval("SELECT true FROM checker_executions WHERE execution_id = $1", execution_id)
        return CheckerExecutionUpdate(found=bool(exists), updated=bool(exists), existing_status=None)

    query, params = built
    row = await conn.fetchrow(query, *params)
    if row is None:
        return CheckerExecutionUpdate(found=False, updated=False, existing_status=None)
    return CheckerExecutionUpdate(found=True, updated=bool(row["updated"]), existing_status=row["existing_status"])


# =========================================================================
# checker_executions — job coordination
# =========================================================================
#
# A checker pod is not addressable through its gateway, so its job row is the only channel it shares with the
# arbitrator. The pod takes ownership of the row, refreshes its claim as it works, and writes its own terminal
# state; the arbitrator reads the row rather than holding a connection open to the pod.
#
# ``claimed_by`` is a fencing token, not a label. A pod whose lease expires can be replaced, so every write after
# the claim carries the writer's identity as a predicate and a zero-row result means "you have been superseded,
# stop". Dropping that predicate would let a merely-slow pod wake up and overwrite its replacement's result with
# no error raised anywhere, which is why the coordination writes are separate functions rather than options on
# ``update_checker_execution``: there is no way to call one of these without being fenced.


# Module-level so the image test can hand it to a real Postgres, for the reason given above
# MARK_CHECKER_EXECUTIONS_IN_PROGRESS_SQL.
CLAIM_NEXT_CHECKER_EXECUTIONS_SQL = """
    WITH claimable AS (
        SELECT execution_id
        FROM checker_executions
        WHERE checker_name = $1 AND status = 'queued'
        ORDER BY execution_id
        LIMIT $4
        FOR UPDATE SKIP LOCKED
    )
    UPDATE checker_executions AS c
    SET status = 'in_progress',
        claimed_by = $2,
        lease_expires = NOW() + $3::interval,
        attempt = attempt + 1,
        start_time = NOW(),
        version = COALESCE($5, c.version),
        commit_sha = COALESCE($6, c.commit_sha)
    FROM claimable
    WHERE c.execution_id = claimable.execution_id
    RETURNING c.execution_id, c.claimed_by, c.attempt, c.lease_expires, c.cancel_requested,
              c.request_json, c.checker_name, c.camera_name, c.request_id
"""


async def claim_next_checker_executions(
    conn: asyncpg.Connection,
    checker_name: str,
    claimed_by: str,
    lease: timedelta,
    limit: int,
    version: str | None = None,
    commit_sha: str | None = None,
) -> list[dict[str, Any]]:
    """Claim up to ``limit`` of the oldest unstarted jobs for one checker. Empty means there is nothing waiting.

    The worker chooses the row rather than being handed one, and that inversion is the point: a pod takes work only
    when it has room for it, so capacity is expressed by not claiming and nothing has to know the fleet's
    concurrency. ``FOR UPDATE SKIP LOCKED`` is what makes the whole fleet able to run this query at once -- each
    caller passes over rows another is claiming rather than blocking behind it.

    Only ``queued`` rows are taken, and that status carries a promise this query relies on: the row's payload is the
    one the job should run. A row that merely exists is ``pending`` (see :func:`create_checker_execution`), so there
    is no window in which this query can hand a pod a job whose payload has not been written yet. A job whose owner
    died comes back here by way of :func:`requeue_expired_checker_executions`, which is what keeps recovery in one
    place rather than making every claim also a takeover.

    Batched because a pod with several free slots should fill them in one round trip; ``ORDER BY execution_id``
    makes the queue FIFO within a checker, and is served directly by the partial claimable index.

    ``cancel_requested`` is deliberately carried through the claim rather than cleared: a cancel that landed while
    the job sat in the queue still applies to it, and the flag comes back in the returned row so the new owner can
    abandon the job before doing any work rather than running something nobody is waiting for.

    ``version`` and ``commit_sha`` identify the build of the claiming pod and replace whatever the row carried, so a
    row names the pod that ran it; a requeued job is stamped again by its next owner, the only one whose outcome the
    row can still record. ``None`` leaves the column as it is, which is what a caller that does not report its build
    gets.
    """
    async with conn.transaction():
        rows = await conn.fetch(
            CLAIM_NEXT_CHECKER_EXECUTIONS_SQL, checker_name, claimed_by, lease, limit, version, commit_sha
        )

    return [with_isoformat_times(dict(row), "lease_expires") for row in rows]


async def requeue_expired_checker_executions(conn: asyncpg.Connection, limit: int) -> list[dict[str, Any]]:
    """Return jobs whose owner stopped renewing to the queue, so somebody else can pick them up.

    A single conditional UPDATE, which makes it idempotent and therefore safe to run from every replica on a
    jittered interval with no leader to elect: whichever one gets there first requeues the row and the rest match
    nothing. ``claimed_by`` is cleared as part of the requeue, so the dead owner's writes are fenced out even if it
    comes back to life afterwards.

    ``attempt`` is deliberately left alone -- it was already incremented by the claim being reaped, so the count
    reflects how many pods have tried this job, which is what a cap on retries needs to know.

    The requeue targets ``queued`` rather than ``pending`` because a reaped job has already been enqueued once: its
    payload is on the row, so it is immediately runnable by whoever takes it next. ``pending`` is only ever the state
    of a row that has not been offered yet, and sending a reaped job back there would strand it, since nothing
    revisits a row the arbitrator believes it has already handed over.
    """
    async with conn.transaction():
        rows = await conn.fetch(
            """
            WITH expired AS (
                SELECT execution_id
                FROM checker_executions
                WHERE status = 'in_progress'
                  AND lease_expires IS NOT NULL
                  AND lease_expires < NOW()
                ORDER BY lease_expires
                LIMIT $1
                FOR UPDATE SKIP LOCKED
            )
            UPDATE checker_executions AS c
            SET status = 'queued',
                claimed_by = NULL,
                lease_expires = NULL
            FROM expired
            WHERE c.execution_id = expired.execution_id
            RETURNING c.execution_id, c.checker_name, c.attempt
            """,
            limit,
        )

    return [dict(row) for row in rows]


async def heartbeat_checker_execution(
    conn: asyncpg.Connection, execution_id: int, claimed_by: str, lease: timedelta
) -> dict[str, Any] | None:
    """Extend the owner's lease, returning the job's current control state.

    Returns ``None`` when the caller no longer owns the row, which is the only signal a superseded pod gets that
    it should abandon its work. Callers must treat that as terminal for their attempt rather than retrying.

    The returned ``cancel_requested`` rides along on the same round trip because the owner has to poll for
    cancellation anyway: there is no way to push a cancel to a specific pod behind a gateway, so the arbitrator
    leaves the request on the row and the owner picks it up the next time it checks in.
    """
    row = await conn.fetchrow(
        """
        UPDATE checker_executions
        SET lease_expires = NOW() + $3::interval
        WHERE execution_id = $1 AND claimed_by = $2
        RETURNING execution_id, lease_expires, cancel_requested, attempt
        """,
        execution_id,
        claimed_by,
        lease,
    )
    if not row:
        return None
    return with_isoformat_times(dict(row), "lease_expires")


async def complete_checker_execution(
    conn: asyncpg.Connection,
    execution_id: int,
    claimed_by: str,
    status: str,
    response_json: dict[str, Any] | None = None,
    duration_ms: int | None = None,
    error_message: str | None = None,
    output_metadata: dict[str, Any] | None = None,
) -> bool:
    """Write the owner's terminal state for a job. Returns ``False`` when the caller no longer owns the row.

    This is the write the fencing rule exists for. Everything else a superseded pod does is recoverable noise;
    an unfenced terminal write silently replaces a good result with a stale one.

    ``lease_expires`` is cleared so the row cannot later look like an abandoned claim to a takeover query, and
    ``claimed_by`` is deliberately left in place as a record of which pod produced the result.
    """
    result = await conn.execute(
        """
        UPDATE checker_executions
        SET status = $3,
            end_time = NOW(),
            lease_expires = NULL,
            duration_ms = $4,
            response_json = $5,
            error_message = $6,
            output_metadata = $7
        WHERE execution_id = $1 AND claimed_by = $2
        """,
        execution_id,
        claimed_by,
        status,
        duration_ms or 0,
        response_json if response_json is not None else {},
        error_message,
        output_metadata,
    )
    return command_tag_rowcount(result) > 0


async def request_checker_execution_cancel(conn: asyncpg.Connection, execution_id: int) -> bool:
    """Ask whichever pod owns this job to stop. Returns ``False`` when the job already reached a terminal state.

    Cancellation is a flag on the row rather than a call to the owner because the owner is not reachable: an
    endpoint would route to a random pod. The owner observes the flag on its next heartbeat, so cancellation is
    asynchronous by construction and a job that is not yet claimed is cancelled before it ever starts.

    Every non-terminal status is cancellable, ``pending`` included. A job whose payload has not been written yet is
    the *earliest* thing a caller can call off, so excluding it would silently drop exactly the cancels that arrive in
    time to save the most work -- and the row would go on to be enqueued and run anyway, because the flag rides
    through the claim rather than being re-checked against the status.
    """
    result = await conn.execute(
        """
        UPDATE checker_executions
        SET cancel_requested = true
        WHERE execution_id = $1 AND status IN ('pending', 'queued', 'in_progress')
        """,
        execution_id,
    )
    return command_tag_rowcount(result) > 0


async def get_checker_execution_states(conn: asyncpg.Connection, execution_ids: list[int]) -> list[dict[str, Any]]:
    """Read the coordination state of many jobs in one query.

    The arbitrator polls every job it owns, so this is deliberately a batch read: one query per polling replica
    rather than one per in-flight job keeps the query rate proportional to replicas instead of to fan-out width.

    The projection excludes ``request_json``, which the poller already has and which dwarfs everything else on
    the row.
    """
    rows = await conn.fetch(
        """
        SELECT execution_id, request_id, checker_name, camera_name, status, claimed_by, lease_expires,
               attempt, cancel_requested, start_time, end_time, duration_ms, error_message,
               output_metadata, response_json
        FROM checker_executions
        WHERE execution_id = ANY($1::int[])
        ORDER BY execution_id
        """,
        execution_ids,
    )
    return [with_isoformat_times(dict(row), "lease_expires", "start_time", "end_time") for row in rows]


# =========================================================================
# clients
# =========================================================================


async def record_client(
    conn: asyncpg.Connection, ip_address: str, name: str | None = None, description: str | None = None
) -> dict[str, Any] | None:
    """Upsert a client record by IP address."""
    row = await conn.fetchrow(
        """
        INSERT INTO clients (ip_address, name, description)
        VALUES ($1::inet, $2, $3)
        ON CONFLICT (ip_address) DO UPDATE SET
            name = COALESCE(EXCLUDED.name, clients.name),
            description = COALESCE(EXCLUDED.description, clients.description)
        RETURNING client_id, ip_address, name, description
        """,
        ip_address,
        name,
        description,
    )
    return dict(row) if row else None
