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

"""Unit tests for the client-agnostic services.database.service.DatabaseService base.

Covers the arbitrator database table access with a mocked asyncpg pool.
"""

import contextlib
from datetime import UTC, datetime, timedelta
from ipaddress import IPv4Address
import json
from typing import Any
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from services.database.queries.orchestration import CheckerExecutionUpdate
from services.database.service import (
    DEFAULT_POOL_MAX_SIZE,
    DEFAULT_POOL_MIN_SIZE,
    DatabaseService,
    PoolUnavailableError,
    _build_postgres_url,
    _init_json_codecs,
    _json_encoder,
)
from services.database.settings import Settings


def _make_service(pool: Any = None) -> DatabaseService:
    """Create a base DatabaseService with an injected mock pool."""
    service = DatabaseService(Settings())
    service._pool = pool
    return service


def _mock_pool(conn: Any) -> MagicMock:
    """Create a mock asyncpg pool whose acquire() works as an async context manager."""
    pool = MagicMock()

    @contextlib.asynccontextmanager
    async def _acquire() -> Any:
        yield conn

    pool.acquire = _acquire
    return pool


def _conn_in_a_transaction(fetch: list[dict[str, Any]]) -> Any:
    """A mock connection whose ``transaction()`` works as an async context manager.

    ``AsyncMock`` cannot stand in for one: calling ``conn.transaction()`` on it yields a coroutine rather than
    something ``async with`` accepts.
    """
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=fetch)

    @contextlib.asynccontextmanager
    async def _transaction() -> Any:
        yield None

    conn.transaction = _transaction
    return conn


def _conn_answering_checker_update(*, existing_status: str, updated: bool) -> Any:
    """A mock connection whose ``fetchrow`` returns what the guarded checker_executions update selects.

    The statement ends in a SELECT rather than an UPDATE, so a mock that only answers ``execute`` would let the
    service read nothing and report the wrong outcome.
    """
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(return_value={"existing_status": existing_status, "updated": updated})
    return conn


class TestPoolUnavailable(unittest.IsolatedAsyncioTestCase):
    """When the asyncpg pool is None, read/write methods return empty/None/False.

    The exceptions are ``update_execution`` and ``update_checker_execution``, which raise
    ``PoolUnavailableError`` so the transport layer can answer 503 (pool down) distinctly from 404
    (no matching row).
    """

    def setUp(self) -> None:
        self.service = _make_service(pool=None)

    async def test_get_execution_returns_none(self) -> None:
        self.assertIsNone(await self.service.get_execution(1))

    async def test_get_checker_executions_for_request_returns_empty(self) -> None:
        self.assertEqual(await self.service.get_checker_executions_for_request(1), [])

    async def test_create_execution_returns_none(self) -> None:
        self.assertIsNone(await self.service.create_execution(1, {}, "v", "sha"))

    async def test_update_execution_raises_when_pool_unavailable(self) -> None:
        with self.assertRaises(PoolUnavailableError):
            await self.service.update_execution(1, "success")

    async def test_create_checker_execution_returns_none(self) -> None:
        self.assertIsNone(await self.service.create_checker_execution(1, "chk", {}))

    async def test_update_checker_execution_raises_when_pool_unavailable(self) -> None:
        with self.assertRaises(PoolUnavailableError):
            await self.service.update_checker_execution(1, "success")

    async def test_record_client_returns_none(self) -> None:
        self.assertIsNone(await self.service.record_client("1.2.3.4"))


class TestGetExecution(unittest.IsolatedAsyncioTestCase):
    """Tests for get_execution with a mocked pool."""

    async def test_returns_dict_when_found(self) -> None:
        row = {"request_id": 1, "status": "success", "timestamp": None}
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=row)
        service = _make_service(pool=_mock_pool(conn))
        result = await service.get_execution(1)
        self.assertEqual(result, {"request_id": 1, "status": "success", "timestamp": None})

    async def test_returns_none_when_not_found(self) -> None:
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        service = _make_service(pool=_mock_pool(conn))
        self.assertIsNone(await service.get_execution(999))

    async def test_converts_timestamp_to_iso(self) -> None:
        ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"request_id": 1, "timestamp": ts})
        service = _make_service(pool=_mock_pool(conn))
        result = await service.get_execution(1)
        assert result is not None
        self.assertEqual(result["timestamp"], ts.isoformat())


class TestGetCheckerExecutionsForRequest(unittest.IsolatedAsyncioTestCase):
    """Tests for get_checker_executions_for_request with a mocked pool."""

    async def test_returns_rows_with_iso_timestamps(self) -> None:
        start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        conn = AsyncMock()
        conn.fetch = AsyncMock(
            return_value=[{"execution_id": 1, "status": "success", "start_time": start, "end_time": None}]
        )
        service = _make_service(pool=_mock_pool(conn))
        out = await service.get_checker_executions_for_request(7)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["start_time"], start.isoformat())
        self.assertIsNone(out[0]["end_time"])


class TestCreateExecution(unittest.IsolatedAsyncioTestCase):
    """Tests for create_execution with a mocked pool."""

    async def test_returns_request_id(self) -> None:
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"request_id": 42})
        service = _make_service(pool=_mock_pool(conn))
        request_id = await service.create_execution(1, {"foo": "bar"}, "1.0.0", "abc123")
        self.assertEqual(request_id, 42)

    async def test_metadata_none_passes_null_json(self) -> None:
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"request_id": 1})
        service = _make_service(pool=_mock_pool(conn))
        await service.create_execution(1, {}, "v", "s", metadata=None)
        args = conn.fetchrow.call_args[0]
        self.assertIsNone(args[3])


class TestUpdateExecutionBranches(unittest.IsolatedAsyncioTestCase):
    """Tests for the three update_execution SQL branches."""

    async def test_with_response_json(self) -> None:
        conn = AsyncMock()
        conn.execute.return_value = "UPDATE 1"
        service = _make_service(pool=_mock_pool(conn))
        self.assertTrue(await service.update_execution(1, "done", response_json={"a": 1}, duration_ms=12))
        conn.execute.assert_awaited_once()

    async def test_status_and_duration_only(self) -> None:
        conn = AsyncMock()
        conn.execute.return_value = "UPDATE 1"
        service = _make_service(pool=_mock_pool(conn))
        self.assertTrue(await service.update_execution(2, "done", response_json=None, duration_ms=5))
        self.assertIn("duration_ms", conn.execute.call_args[0][0])

    async def test_status_only(self) -> None:
        conn = AsyncMock()
        conn.execute.return_value = "UPDATE 1"
        service = _make_service(pool=_mock_pool(conn))
        self.assertTrue(await service.update_execution(3, "pending", response_json=None, duration_ms=None))
        self.assertIn("SET status", conn.execute.call_args[0][0])

    async def test_returns_false_when_no_row_matched(self) -> None:
        conn = AsyncMock()
        conn.execute.return_value = "UPDATE 0"
        service = _make_service(pool=_mock_pool(conn))
        self.assertFalse(await service.update_execution(4, "done"))


class TestExecutionOwnership(unittest.IsolatedAsyncioTestCase):
    """Claiming, renewing and adopting ``arbitrator_executions`` rows."""

    async def test_a_claimed_row_is_created_with_a_lease(self) -> None:
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"request_id": 42})
        service = _make_service(pool=_mock_pool(conn))

        await service.create_execution(1, {}, "v", "s", claimed_by="replica-a", lease_seconds=90)

        args = conn.fetchrow.call_args[0]
        self.assertEqual(args[6], "replica-a")
        self.assertEqual(args[7], timedelta(seconds=90))

    async def test_an_unowned_row_gets_no_lease(self) -> None:
        """A NULL lease is what keeps a row nobody claimed from looking adoptable forever."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"request_id": 42})
        service = _make_service(pool=_mock_pool(conn))

        await service.create_execution(1, {}, "v", "s")

        args = conn.fetchrow.call_args[0]
        self.assertIsNone(args[6])
        self.assertIsNone(args[7])

    async def test_an_owned_update_is_fenced_on_the_claim(self) -> None:
        conn = AsyncMock()
        conn.execute.return_value = "UPDATE 1"
        service = _make_service(pool=_mock_pool(conn))

        self.assertTrue(await service.update_execution(1, "success", claimed_by="replica-a"))

        query = conn.execute.call_args[0][0]
        self.assertIn("claimed_by = $", query)
        self.assertIn("replica-a", conn.execute.call_args[0][1:])

    async def test_an_update_from_a_superseded_replica_matches_nothing(self) -> None:
        conn = AsyncMock()
        conn.execute.return_value = "UPDATE 0"
        service = _make_service(pool=_mock_pool(conn))

        self.assertFalse(await service.update_execution(1, "success", claimed_by="replica-a"))

    async def test_a_terminal_update_drops_the_lease(self) -> None:
        """A finished request with a lease still on it would look adoptable to every sweeper from then on."""
        conn = AsyncMock()
        conn.execute.return_value = "UPDATE 1"
        service = _make_service(pool=_mock_pool(conn))

        await service.update_execution(1, "success", claimed_by="replica-a")

        self.assertIn("lease_expires = CASE WHEN", conn.execute.call_args[0][0])

    async def test_a_heartbeat_renews_the_whole_batch_in_one_statement(self) -> None:
        """One round trip regardless of how many requests: a pass that scaled with load could outlast its own lease."""
        conn = AsyncMock()
        conn.fetch.return_value = [{"request_id": 1}, {"request_id": 2}]
        service = _make_service(pool=_mock_pool(conn))

        renewal = await service.heartbeat_executions([1, 2], "replica-a", 90)

        self.assertEqual((renewal.renewed, renewal.lost), ([1, 2], []))
        conn.fetch.assert_awaited_once()
        query, ids, claimed_by, lease = conn.fetch.call_args[0]
        self.assertIn("claimed_by = $2", query)
        self.assertEqual((ids, claimed_by, lease), ([1, 2], "replica-a", timedelta(seconds=90)))

    async def test_a_request_taken_over_comes_back_named_with_its_new_owner(self) -> None:
        """The reason matters: the caller logs an adoption and a finished row very differently."""
        conn = AsyncMock()
        conn.fetch.side_effect = [
            [{"request_id": 1}],
            [{"request_id": 2, "status": "in_progress", "claimed_by": "replica-b"}],
        ]
        service = _make_service(pool=_mock_pool(conn))

        renewal = await service.heartbeat_executions([1, 2], "replica-a", 90)

        self.assertEqual(renewal.renewed, [1])
        self.assertEqual(renewal.lost, [{"request_id": 2, "status": "in_progress", "claimed_by": "replica-b"}])

    async def test_a_request_that_already_finished_comes_back_still_owned(self) -> None:
        """Same failure, opposite cause: nobody took it, its outcome is simply already recorded."""
        conn = AsyncMock()
        conn.fetch.side_effect = [[], [{"request_id": 5, "status": "success", "claimed_by": "replica-a"}]]
        service = _make_service(pool=_mock_pool(conn))

        renewal = await service.heartbeat_executions([5], "replica-a", 90)

        self.assertEqual(renewal.lost, [{"request_id": 5, "status": "success", "claimed_by": "replica-a"}])

    async def test_a_vanished_row_is_reported_without_a_state(self) -> None:
        conn = AsyncMock()
        conn.fetch.side_effect = [[], []]
        service = _make_service(pool=_mock_pool(conn))

        renewal = await service.heartbeat_executions([9], "replica-a", 90)

        self.assertEqual(renewal.lost, [{"request_id": 9, "status": None, "claimed_by": None}])

    async def test_an_empty_batch_does_not_touch_the_database(self) -> None:
        conn = AsyncMock()
        service = _make_service(pool=_mock_pool(conn))

        renewal = await service.heartbeat_executions([], "replica-a", 90)

        self.assertEqual((renewal.renewed, renewal.lost), ([], []))
        conn.fetch.assert_not_awaited()

    async def test_a_heartbeat_without_a_pool_raises_rather_than_reporting_loss(self) -> None:
        """A reported loss means "somebody took your request"; a database outage must not be mistaken for that."""
        with self.assertRaises(PoolUnavailableError):
            await _make_service(pool=None).heartbeat_executions([1], "replica-a", 90)

    async def test_adoption_takes_stale_rows_and_returns_what_rebuilds_them(self) -> None:
        expires = datetime(2026, 1, 1, tzinfo=UTC)
        conn = _conn_in_a_transaction(
            fetch=[
                {
                    "request_id": 7,
                    "client_id": 3,
                    "status": "in_progress",
                    "request_json": {"inputs": {}},
                    "claimed_by": "replica-b",
                    "lease_expires": expires,
                    "version": "1.0.0",
                    "commit_sha": "abc",
                }
            ]
        )
        service = _make_service(pool=_mock_pool(conn))

        rows = await service.adopt_executions("replica-b", 90, limit=5)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["request_json"], {"inputs": {}})
        self.assertEqual(rows[0]["lease_expires"], expires.isoformat())
        query, claimed_by, lease, limit = conn.fetch.call_args[0]
        self.assertIn("FOR UPDATE SKIP LOCKED", query, "N sweepers must not be able to adopt the same row")
        self.assertEqual((claimed_by, lease, limit), ("replica-b", timedelta(seconds=90), 5))

    async def test_adoption_only_considers_rows_whose_lease_has_actually_lapsed(self) -> None:
        conn = _conn_in_a_transaction(fetch=[])
        service = _make_service(pool=_mock_pool(conn))

        await service.adopt_executions("replica-b", 90, limit=5)

        query = conn.fetch.call_args[0][0]
        self.assertIn("lease_expires < NOW()", query)
        self.assertIn("status IN ('queued', 'in_progress')", query)

    async def test_a_replica_never_adopts_a_request_it_already_owns(self) -> None:
        """Self-adoption hands a replica a second copy of work it is already running, and the claim never changes hands
        -- so nothing looks wrong until one copy finishes and strands the other on a terminal row."""
        conn = _conn_in_a_transaction(fetch=[])
        service = _make_service(pool=_mock_pool(conn))

        await service.adopt_executions("replica-b", 90, limit=5)

        query = conn.fetch.call_args[0][0]
        self.assertIn("claimed_by IS DISTINCT FROM $1", query)

    async def test_adoption_without_a_pool_finds_nothing(self) -> None:
        self.assertEqual(await _make_service(pool=None).adopt_executions("replica-b", 90, 5), [])


class TestGetCheckerExecutionStatesForRequest(unittest.IsolatedAsyncioTestCase):
    """The wider per-request read an adopting replica rebuilds from."""

    async def test_the_projection_carries_both_results_and_ownership(self) -> None:
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        service = _make_service(pool=_mock_pool(conn))

        await service.get_checker_execution_states_for_request(7)

        query = conn.fetch.call_args[0][0]
        for column in ("response_json", "error_message", "claimed_by", "lease_expires", "attempt"):
            self.assertIn(column, query)
        self.assertEqual(conn.fetch.call_args[0][1], 7)

    async def test_timestamps_are_rendered_for_json(self) -> None:
        expires = datetime(2026, 1, 1, tzinfo=UTC)
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[{"execution_id": 1, "lease_expires": expires, "start_time": None}])
        service = _make_service(pool=_mock_pool(conn))

        rows = await service.get_checker_execution_states_for_request(7)

        self.assertEqual(rows[0]["lease_expires"], expires.isoformat())

    async def test_without_a_pool_the_caller_is_told_rather_than_handed_an_empty_request(self) -> None:
        """An adopting replica must not read "no pool" as "this request has no jobs" and finalize it empty."""
        with self.assertRaises(PoolUnavailableError):
            await _make_service(pool=None).get_checker_execution_states_for_request(7)


class TestCheckerExecutionWrites(unittest.IsolatedAsyncioTestCase):
    """Tests for create/update checker execution with a mocked pool."""

    async def test_create_checker_execution_returns_id(self) -> None:
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"execution_id": 99})
        service = _make_service(pool=_mock_pool(conn))
        eid = await service.create_checker_execution(1, "chk", {}, camera_name="cam")
        self.assertEqual(eid, 99)

    async def test_a_new_row_is_not_claimable(self) -> None:
        """Inserting into 'queued' would offer the job before its payload was written.

        The row is born carrying a placeholder, so a pod claiming it either rejects an empty payload or starts work
        from one that is about to be rewritten under it. ``enqueue`` is what promotes the row.
        """
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"execution_id": 99})
        service = _make_service(pool=_mock_pool(conn))

        await service.create_checker_execution(1, "chk", {})

        sql = conn.fetchrow.await_args.args[0]
        self.assertIn("'pending'", sql)
        self.assertNotIn("'queued'", sql)

    async def test_update_checker_execution_terminal(self) -> None:
        conn = _conn_answering_checker_update(existing_status="in_progress", updated=True)
        service = _make_service(pool=_mock_pool(conn))
        outcome = await service.update_checker_execution(
            5, "success", response_json={"x": 1}, duration_ms=10, error_message=None, output_metadata={"o": 1}
        )
        self.assertEqual(outcome, CheckerExecutionUpdate(found=True, updated=True, existing_status="in_progress"))
        conn.fetchrow.assert_awaited_once()

    async def test_update_checker_execution_partial_status_only(self) -> None:
        conn = _conn_answering_checker_update(existing_status="pending", updated=True)
        service = _make_service(pool=_mock_pool(conn))
        outcome = await service.update_checker_execution(5, "queued")
        self.assertIs(outcome.updated, True)
        self.assertIn("status = $1", conn.fetchrow.call_args[0][0])

    async def test_update_checker_execution_rewrites_the_job_payload(self) -> None:
        """The payload lives on the row, because that is where the claiming pod reads it."""
        conn = _conn_answering_checker_update(existing_status="pending", updated=True)
        service = _make_service(pool=_mock_pool(conn))

        outcome = await service.update_checker_execution(5, "queued", request_json={"video_url": "s3://clip.mp4"})

        self.assertIs(outcome.updated, True)
        query = conn.fetchrow.call_args[0][0]
        self.assertIn("request_json = $2", query)
        self.assertNotIn("end_time", query, "writing a payload must not close the row out")
        self.assertEqual(conn.fetchrow.call_args[0][2], {"video_url": "s3://clip.mp4"})

    async def test_update_checker_execution_reports_not_found_when_no_row_exists(self) -> None:
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        service = _make_service(pool=_mock_pool(conn))
        self.assertEqual(
            await service.update_checker_execution(5, "queued"),
            CheckerExecutionUpdate(found=False, updated=False, existing_status=None),
        )

    async def test_an_update_with_no_columns_succeeds_without_writing(self) -> None:
        """Nothing to write means nothing to refuse, so an existing row reports success and is left untouched."""
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=True)
        service = _make_service(pool=_mock_pool(conn))

        self.assertEqual(
            await service.update_checker_execution(5),
            CheckerExecutionUpdate(found=True, updated=True, existing_status=None),
        )
        conn.fetchrow.assert_not_awaited()

    async def test_an_update_with_no_columns_still_reports_a_missing_row(self) -> None:
        """``found`` means the row exists, so a no-op write must not report success for a job that is not there.

        Every other shape of this call answers a missing execution_id with 404, and the client raises on it rather
        than absorbing it; reporting success here would hand that caller a 200 for the same inconsistency.
        """
        conn = AsyncMock()
        conn.fetchval = AsyncMock(return_value=None)
        service = _make_service(pool=_mock_pool(conn))

        self.assertEqual(
            await service.update_checker_execution(5),
            CheckerExecutionUpdate(found=False, updated=False, existing_status=None),
        )

    async def test_update_checker_execution_reports_refusal_on_a_terminal_row(self) -> None:
        """A row that has finished keeps its status, and the caller is told which one it holds.

        Distinct from the not-found case above because the arbitrator reacts to them differently: a finished job is
        one whose result is waiting to be read, a missing one is a job it can no longer account for.
        """
        conn = _conn_answering_checker_update(existing_status="success", updated=False)
        service = _make_service(pool=_mock_pool(conn))
        self.assertEqual(
            await service.update_checker_execution(5, "in_progress"),
            CheckerExecutionUpdate(found=True, updated=False, existing_status="success"),
        )

    async def test_the_batched_claim_write_returns_only_the_refusals(self) -> None:
        """The healthy answer is an empty list, so what comes back is the exceptions rather than a row per id."""
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[{"execution_id": 7, "existing_status": "success"}])
        service = _make_service(pool=_mock_pool(conn))

        refused = await service.mark_checker_executions_in_progress([7, 8, 9], datetime.now(UTC))

        self.assertEqual(refused, [{"execution_id": 7, "existing_status": "success"}])
        self.assertEqual(conn.fetch.await_args.args[1], [7, 8, 9])

    async def test_the_batched_claim_write_skips_the_query_for_an_empty_batch(self) -> None:
        """A pass that observed no new claims must not spend a round trip saying so."""
        conn = AsyncMock()
        service = _make_service(pool=_mock_pool(conn))

        self.assertEqual(await service.mark_checker_executions_in_progress([], datetime.now(UTC)), [])
        conn.fetch.assert_not_awaited()

    async def test_the_batched_claim_write_raises_when_the_pool_is_down(self) -> None:
        """An empty result has to mean every claim was recorded, so a failure to look cannot read as one.

        The caller treats an id it hears nothing about as still running and starts its budget clock, which for a
        finished job is exactly the state this whole guard exists to prevent.
        """
        service = _make_service(pool=None)
        with self.assertRaises(PoolUnavailableError):
            await service.mark_checker_executions_in_progress([7], datetime.now(UTC))


class TestRecordClient(unittest.IsolatedAsyncioTestCase):
    """Tests for record_client with a mocked pool."""

    async def test_returns_dict_on_success(self) -> None:
        row = {"client_id": 1, "ip_address": IPv4Address("1.2.3.4"), "name": "test", "description": None}
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=row)
        service = _make_service(pool=_mock_pool(conn))
        result = await service.record_client("1.2.3.4", name="test")
        assert result is not None
        self.assertEqual(result["client_id"], 1)

    async def test_returns_none_when_fetchrow_returns_none(self) -> None:
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        service = _make_service(pool=_mock_pool(conn))
        self.assertIsNone(await service.record_client("1.2.3.4"))


class TestBuildPostgresUrl(unittest.TestCase):
    """Tests for the _build_postgres_url helper."""

    def test_builds_dsn_with_escaped_credentials(self) -> None:
        url = _build_postgres_url(
            {"user": "u", "password": "p@ss word", "host": "db", "port": 5432, "database": "staging"}
        )
        self.assertEqual(url, "postgresql://u:p%40ss+word@db:5432/staging")


class TestClaimNextCheckerExecutions(unittest.IsolatedAsyncioTestCase):
    """Tests for the claim a pod makes when it has room."""

    async def test_the_claim_carries_everything_needed_to_run_the_job(self) -> None:
        # A pod that chose its own work knows nothing about it beforehand, so the claim has to identify the job as well
        # as authorize it: without checker_name/camera_name/request_json the claimer has a lease on a mystery.
        lease = datetime(2026, 3, 1, 12, 0, 30, tzinfo=UTC)
        conn = _conn_in_a_transaction(
            fetch=[
                {
                    "execution_id": 7,
                    "claimed_by": "pod-a",
                    "attempt": 1,
                    "lease_expires": lease,
                    "cancel_requested": False,
                    "request_json": {"clip_id": "c"},
                    "checker_name": "av.egomotion",
                    "camera_name": "camera_front_wide_120fov",
                    "request_id": 3,
                }
            ]
        )
        service = _make_service(pool=_mock_pool(conn))

        jobs = await service.claim_next_checker_executions(
            checker_name="av.egomotion", claimed_by="pod-a", lease_seconds=30.0, limit=2
        )

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["checker_name"], "av.egomotion")
        self.assertEqual(jobs[0]["request_json"], {"clip_id": "c"})
        self.assertEqual(jobs[0]["lease_expires"], lease.isoformat())

    async def test_an_empty_queue_is_an_empty_list(self) -> None:
        conn = _conn_in_a_transaction(fetch=[])
        service = _make_service(pool=_mock_pool(conn))
        self.assertEqual(
            await service.claim_next_checker_executions(
                checker_name="av.vlm", claimed_by="pod-a", lease_seconds=30.0, limit=1
            ),
            [],
        )

    async def test_the_whole_fleet_can_run_this_query_at_once(self) -> None:
        # Without SKIP LOCKED every pod queues behind the first one's row lock, which turns the pull model into a
        # serialized queue with the whole fleet waiting on one transaction.
        conn = _conn_in_a_transaction(fetch=[])
        service = _make_service(pool=_mock_pool(conn))

        await service.claim_next_checker_executions(
            checker_name="av.vlm", claimed_by="pod-a", lease_seconds=30.0, limit=4
        )

        query, checker, claimed_by, lease, limit, version, commit_sha = conn.fetch.call_args[0]
        self.assertIn("FOR UPDATE SKIP LOCKED", query)
        self.assertIn("status = 'queued'", query)
        self.assertIn("ORDER BY execution_id", query, "the queue must be FIFO within a checker")
        self.assertEqual((checker, claimed_by, lease, limit), ("av.vlm", "pod-a", timedelta(seconds=30.0), 4))
        self.assertEqual((version, commit_sha), (None, None), "a caller that reports no build leaves the row alone")

    async def test_the_claimer_s_build_is_stamped_on_the_row(self) -> None:
        """The row names the pod that ran it, and a NULL parameter keeps whatever the row already carried."""
        conn = _conn_in_a_transaction(fetch=[])
        service = _make_service(pool=_mock_pool(conn))

        await service.claim_next_checker_executions(
            checker_name="av.vlm", claimed_by="pod-a", lease_seconds=30.0, limit=1, version="1.44.0", commit_sha="abc"
        )

        query, *_, version, commit_sha = conn.fetch.call_args[0]
        self.assertIn("version = COALESCE($5, c.version)", query)
        self.assertIn("commit_sha = COALESCE($6, c.commit_sha)", query)
        self.assertEqual((version, commit_sha), ("1.44.0", "abc"))

    async def test_a_job_that_was_never_offered_is_not_taken(self) -> None:
        """'queued' promises the row's payload is the one to run, and only that status may be claimed.

        Admitting 'pending' here would restore the defect the two statuses exist to separate: a pod would claim a
        dependent job minutes before its inputs existed, and an independent one moments before the scheduler
        republished its row, running it twice.
        """
        conn = _conn_in_a_transaction(fetch=[])
        service = _make_service(pool=_mock_pool(conn))

        await service.claim_next_checker_executions(
            checker_name="av.objects_multi_view", claimed_by="pod-a", lease_seconds=30.0, limit=1
        )

        self.assertNotIn("pending", conn.fetch.call_args[0][0])

    async def test_claiming_counts_as_an_attempt(self) -> None:
        # The count is how a job that kills whichever pod claims it can eventually be given up on instead of cycling
        # through the fleet forever.
        conn = _conn_in_a_transaction(fetch=[])
        service = _make_service(pool=_mock_pool(conn))
        await service.claim_next_checker_executions(
            checker_name="av.vlm", claimed_by="pod-a", lease_seconds=30.0, limit=1
        )
        self.assertIn("attempt = attempt + 1", conn.fetch.call_args[0][0])

    async def test_a_cancel_that_landed_while_queued_survives_the_claim(self) -> None:
        """Clearing the flag on claim would hand the new owner a job the arbitrator has already called off."""
        conn = _conn_in_a_transaction(fetch=[])
        service = _make_service(pool=_mock_pool(conn))
        await service.claim_next_checker_executions(
            checker_name="av.egomotion", claimed_by="pod-a", lease_seconds=30.0, limit=1
        )
        sql = conn.fetch.call_args[0][0]
        self.assertNotIn("cancel_requested = false", sql)
        self.assertIn("c.cancel_requested", sql)

    async def test_asking_for_nothing_does_not_touch_the_database(self) -> None:
        conn = _conn_in_a_transaction(fetch=[])
        service = _make_service(pool=_mock_pool(conn))
        self.assertEqual(
            await service.claim_next_checker_executions(
                checker_name="av.vlm", claimed_by="pod-a", lease_seconds=30.0, limit=0
            ),
            [],
        )
        conn.fetch.assert_not_awaited()

    async def test_a_down_pool_is_distinguishable_from_an_empty_queue(self) -> None:
        # These have to be different answers: an empty queue means "wait and ask again", a down pool means the caller
        # cannot know whether there is work at all.
        service = _make_service(pool=None)
        with self.assertRaises(PoolUnavailableError):
            await service.claim_next_checker_executions(
                checker_name="av.vlm", claimed_by="pod-a", lease_seconds=30.0, limit=1
            )


class TestRequeueExpiredCheckerExecutions(unittest.IsolatedAsyncioTestCase):
    """Tests for the sweep that returns a dead owner's work to the queue."""

    async def test_the_sweep_reports_what_it_moved(self) -> None:
        conn = _conn_in_a_transaction(fetch=[{"execution_id": 7, "checker_name": "av.egomotion", "attempt": 2}])
        service = _make_service(pool=_mock_pool(conn))

        requeued = await service.requeue_expired_checker_executions(limit=50)

        self.assertEqual(requeued, [{"execution_id": 7, "checker_name": "av.egomotion", "attempt": 2}])
        self.assertEqual(conn.fetch.call_args[0][1], 50)

    async def test_only_lapsed_claims_are_taken_away(self) -> None:
        conn = _conn_in_a_transaction(fetch=[])
        service = _make_service(pool=_mock_pool(conn))

        await service.requeue_expired_checker_executions(limit=10)

        query = conn.fetch.call_args[0][0]
        self.assertIn("status = 'in_progress'", query)
        self.assertIn("lease_expires < NOW()", query)
        self.assertIn("FOR UPDATE SKIP LOCKED", query, "every replica sweeps; they must not fight over rows")

    async def test_requeueing_fences_out_the_owner_it_took_the_job_from(self) -> None:
        # A pod that was merely paused comes back believing it still owns the job. Clearing claimed_by is what makes
        # its next write match nothing instead of overwriting the replacement's result.
        conn = _conn_in_a_transaction(fetch=[])
        service = _make_service(pool=_mock_pool(conn))

        await service.requeue_expired_checker_executions(limit=10)

        query = conn.fetch.call_args[0][0]
        self.assertIn("claimed_by = NULL", query)
        self.assertIn("status = 'queued'", query)

    async def test_the_attempt_count_survives_the_requeue(self) -> None:
        # It counts pods that have tried the job, which is what a retry cap needs; resetting it here would make a job
        # that kills its owner immortal.
        conn = _conn_in_a_transaction(fetch=[])
        service = _make_service(pool=_mock_pool(conn))
        await service.requeue_expired_checker_executions(limit=10)
        self.assertNotIn("attempt", conn.fetch.call_args[0][0].split("RETURNING")[0])

    async def test_a_down_pool_leaves_the_sweep_a_no_op(self) -> None:
        self.assertEqual(await _make_service(pool=None).requeue_expired_checker_executions(limit=10), [])


class TestHeartbeatCheckerExecution(unittest.IsolatedAsyncioTestCase):
    """Tests for the fenced lease refresh that also carries the cancellation answer back."""

    async def test_returns_control_state(self) -> None:
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(
            return_value={"execution_id": 7, "lease_expires": None, "cancel_requested": True, "attempt": 2}
        )
        service = _make_service(pool=_mock_pool(conn))
        state = await service.heartbeat_checker_execution(7, "pod-a", 30.0)
        assert state is not None
        self.assertTrue(state["cancel_requested"])

    async def test_returns_none_when_superseded(self) -> None:
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        service = _make_service(pool=_mock_pool(conn))
        self.assertIsNone(await service.heartbeat_checker_execution(7, "pod-a", 30.0))

    async def test_write_is_fenced_on_claimed_by(self) -> None:
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        service = _make_service(pool=_mock_pool(conn))
        await service.heartbeat_checker_execution(7, "pod-a", 30.0)
        self.assertIn("claimed_by = $2", conn.fetchrow.await_args.args[0])

    async def test_heartbeat_only_advances_the_lease(self) -> None:
        # A heartbeat says nothing but "still mine". How long a job has been running is NOW() - start_time, so there is
        # no elapsed time to report and nothing else on the row a renewal has any business touching.
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value=None)
        service = _make_service(pool=_mock_pool(conn))
        await service.heartbeat_checker_execution(7, "pod-a", 30.0)
        set_clause = conn.fetchrow.await_args.args[0].split("SET ", 1)[1].split("WHERE", 1)[0]
        self.assertEqual(set_clause.split("=")[0].strip(), "lease_expires")
        self.assertNotIn(",", set_clause)

    async def test_raises_when_pool_unavailable(self) -> None:
        service = _make_service(pool=None)
        with self.assertRaises(PoolUnavailableError):
            await service.heartbeat_checker_execution(7, "pod-a", 30.0)


class TestNonPositiveLeases(unittest.IsolatedAsyncioTestCase):
    """Every path that writes a lease refuses a non-positive one, on the way in rather than into the row.

    ``NOW() + 0`` is a claim that expired as it was written: the owner is adoptable before it starts, and loses the
    work to a sweep it never had a chance to beat, with every write having succeeded. The API boundary bounds this
    too, so this is the guard for a caller reaching the service directly.
    """

    async def test_the_lease_writes_reject_a_non_positive_lease(self) -> None:
        conn = _conn_in_a_transaction(fetch=[])
        conn.fetchrow = AsyncMock(return_value=None)
        service = _make_service(pool=_mock_pool(conn))

        for lease_seconds in (0.0, -30.0):
            with self.subTest(lease_seconds=lease_seconds):
                with self.assertRaises(ValueError):
                    await service.create_execution(1, {}, "v", "s", claimed_by="replica-a", lease_seconds=lease_seconds)
                with self.assertRaises(ValueError):
                    await service.heartbeat_executions([1], "replica-a", lease_seconds)
                with self.assertRaises(ValueError):
                    await service.adopt_executions("replica-a", lease_seconds, limit=1)
                with self.assertRaises(ValueError):
                    await service.claim_next_checker_executions("av.vlm", "pod-a", lease_seconds, limit=1)
                with self.assertRaises(ValueError):
                    await service.heartbeat_checker_execution(7, "pod-a", lease_seconds)


class TestCompleteCheckerExecution(unittest.IsolatedAsyncioTestCase):
    """Tests for the terminal write, which is the one the fencing rule exists to protect."""

    async def test_returns_true_when_written(self) -> None:
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")
        service = _make_service(pool=_mock_pool(conn))
        self.assertTrue(await service.complete_checker_execution(7, "pod-a", "success", {"score": 1}, 500))

    async def test_returns_false_when_superseded(self) -> None:
        # A pod that was merely slow must be told its result was rejected rather than silently overwriting the
        # replacement's result.
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 0")
        service = _make_service(pool=_mock_pool(conn))
        self.assertFalse(await service.complete_checker_execution(7, "pod-a", "success"))

    async def test_write_is_fenced_and_clears_the_lease(self) -> None:
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")
        service = _make_service(pool=_mock_pool(conn))
        await service.complete_checker_execution(7, "pod-a", "fail", error_message="boom")
        sql = conn.execute.await_args.args[0]
        self.assertIn("claimed_by = $2", sql)
        # A terminal row with a live lease could be mistaken for an abandoned claim by a takeover query.
        self.assertIn("lease_expires = NULL", sql)

    async def test_raises_when_pool_unavailable(self) -> None:
        service = _make_service(pool=None)
        with self.assertRaises(PoolUnavailableError):
            await service.complete_checker_execution(7, "pod-a", "success")


class TestRequestCheckerExecutionCancel(unittest.IsolatedAsyncioTestCase):
    """Tests for leaving a cancellation request on a job whose owner is not addressable."""

    async def test_returns_true_when_flagged(self) -> None:
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")
        service = _make_service(pool=_mock_pool(conn))
        self.assertTrue(await service.request_checker_execution_cancel(7))

    async def test_returns_false_for_an_already_terminal_job(self) -> None:
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 0")
        service = _make_service(pool=_mock_pool(conn))
        self.assertFalse(await service.request_checker_execution_cancel(7))

    async def test_a_job_can_be_called_off_before_it_is_ever_offered(self) -> None:
        """A 'pending' job is the earliest thing a caller can cancel, so it must be cancellable.

        Omitting it would drop exactly the cancels that arrive in time to save the most work, and silently: the row
        would go on to be enqueued and run, because the flag rides through the claim rather than being re-checked
        against the status.
        """
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value="UPDATE 1")
        service = _make_service(pool=_mock_pool(conn))

        self.assertTrue(await service.request_checker_execution_cancel(7))

        sql = conn.execute.await_args.args[0]
        for status in ("pending", "queued", "in_progress"):
            self.assertIn(f"'{status}'", sql, f"a {status} job must still be cancellable")

    async def test_raises_when_pool_unavailable(self) -> None:
        service = _make_service(pool=None)
        with self.assertRaises(PoolUnavailableError):
            await service.request_checker_execution_cancel(7)


class TestGetCheckerExecutionStates(unittest.IsolatedAsyncioTestCase):
    """Tests for the batched coordination-state read the arbitrator polls with."""

    async def test_returns_rows_with_iso_timestamps(self) -> None:
        lease = datetime(2026, 3, 1, 12, 0, 30, tzinfo=UTC)
        start = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
        conn = AsyncMock()
        conn.fetch = AsyncMock(
            return_value=[
                {
                    "execution_id": 7,
                    "status": "in_progress",
                    "lease_expires": lease,
                    "start_time": start,
                    "end_time": None,
                }
            ]
        )
        service = _make_service(pool=_mock_pool(conn))
        rows = await service.get_checker_execution_states([7])
        self.assertEqual(rows[0]["lease_expires"], lease.isoformat())
        self.assertEqual(rows[0]["start_time"], start.isoformat())
        self.assertIsNone(rows[0]["end_time"])

    async def test_empty_id_list_issues_no_query(self) -> None:
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        service = _make_service(pool=_mock_pool(conn))
        self.assertEqual(await service.get_checker_execution_states([]), [])
        conn.fetch.assert_not_awaited()

    async def test_reads_every_id_in_one_query(self) -> None:
        # One query per poller rather than per job is the whole point of the batch read; a per-id loop would make
        # query volume scale with fan-out width instead of with the number of arbitrator replicas.
        conn = AsyncMock()
        conn.fetch = AsyncMock(return_value=[])
        service = _make_service(pool=_mock_pool(conn))
        await service.get_checker_execution_states([1, 2, 3])
        conn.fetch.assert_awaited_once()
        self.assertEqual(conn.fetch.await_args.args[1], [1, 2, 3])

    async def test_raises_when_pool_unavailable(self) -> None:
        """The poller must not read "no pool" as "these jobs made no progress" and start reaping live work."""
        with self.assertRaises(PoolUnavailableError):
            await _make_service(pool=None).get_checker_execution_states([1])


class TestJsonCodec(unittest.IsolatedAsyncioTestCase):
    """Tests for the json/jsonb type codec registered on every pooled connection."""

    def test_encoder_serializes_dict_and_list(self) -> None:
        self.assertEqual(_json_encoder({"a": 1}), '{"a": 1}')
        self.assertEqual(_json_encoder([1, 2]), "[1, 2]")

    def test_encoder_passes_strings_through(self) -> None:
        # Idempotency: an already-serialized JSON string is not double-encoded.
        self.assertEqual(_json_encoder('{"a": 1}'), '{"a": 1}')

    async def test_init_registers_json_and_jsonb_codecs(self) -> None:
        conn = AsyncMock()
        await _init_json_codecs(conn)
        registered = {call.args[0] for call in conn.set_type_codec.await_args_list}
        self.assertEqual(registered, {"json", "jsonb"})
        for call in conn.set_type_codec.await_args_list:
            self.assertEqual(call.kwargs["decoder"], json.loads)
            self.assertEqual(call.kwargs["encoder"], _json_encoder)
            self.assertEqual(call.kwargs["format"], "text")


class TestLifecycle(unittest.IsolatedAsyncioTestCase):
    """Tests for init/close when no database is configured."""

    async def test_init_with_no_db_config_skips_pool(self) -> None:
        service = DatabaseService(Settings())
        await service.init()
        self.assertIsNone(service._pool)

    async def test_close_with_pool(self) -> None:
        pool = AsyncMock()
        service = _make_service(pool=pool)
        await service.close()
        pool.close.assert_awaited_once()
        self.assertIsNone(service._pool)


_CONNECTION_CONFIG = {"user": "u", "password": "p", "host": "h", "port": 5432, "database": "d"}


class TestPoolBounds(unittest.IsolatedAsyncioTestCase):
    """db_config supplies the pool bounds; the module defaults apply when it does not."""

    async def _created_pool_kwargs(self, **overrides: object) -> dict[str, Any]:
        service = DatabaseService(Settings(db_config={**_CONNECTION_CONFIG, **overrides}))
        with patch("services.database.service.asyncpg.create_pool", new=AsyncMock()) as create_pool:
            await service.init()
        create_pool.assert_awaited_once()
        assert create_pool.await_args is not None
        return dict(create_pool.await_args.kwargs)

    async def test_defaults_apply_when_db_config_is_silent(self) -> None:
        kwargs = await self._created_pool_kwargs()
        self.assertEqual(kwargs["min_size"], DEFAULT_POOL_MIN_SIZE)
        self.assertEqual(kwargs["max_size"], DEFAULT_POOL_MAX_SIZE)

    async def test_db_config_overrides_both_bounds(self) -> None:
        kwargs = await self._created_pool_kwargs(pool_min_size=3, pool_max_size=7)
        self.assertEqual(kwargs["min_size"], 3)
        self.assertEqual(kwargs["max_size"], 7)

    async def test_string_bounds_are_coerced(self) -> None:
        # A mapping injected as JSON via DATABASE_DB_CONFIG can carry these as strings.
        kwargs = await self._created_pool_kwargs(pool_min_size="1", pool_max_size="4")
        self.assertEqual(kwargs["min_size"], 1)
        self.assertEqual(kwargs["max_size"], 4)

    async def test_inverted_bounds_raise(self) -> None:
        with self.assertRaises(ValueError):
            await self._created_pool_kwargs(pool_min_size=9, pool_max_size=4)


if __name__ == "__main__":
    unittest.main()
