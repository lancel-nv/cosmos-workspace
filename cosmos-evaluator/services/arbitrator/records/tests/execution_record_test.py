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

"""Tests for the ExecutionRecord dataclass."""

import logging
import time
import unittest
from unittest.mock import AsyncMock, patch

from services.arbitrator.records import execution_record as execution_record_module
from services.arbitrator.records.execution_record import ExecutionRecord

# Suppress logging during tests
logging.getLogger("services.arbitrator.records.execution_record").setLevel(logging.CRITICAL)


class TestExecutionRecord(unittest.IsolatedAsyncioTestCase):
    """Tests for ExecutionRecord dataclass."""

    def test_execution_record_initialization(self) -> None:
        """Test ExecutionRecord initializes with correct default values."""
        mock_client = AsyncMock()
        record = ExecutionRecord(
            api_client=mock_client,
            client_id=42,
            request_json={"test": "data"},
            version="1.0.0",
            commit_sha="abc123",
        )

        self.assertEqual(record.api_client, mock_client)
        self.assertEqual(record.client_id, 42)
        self.assertEqual(record.request_json, {"test": "data"})
        self.assertEqual(record.version, "1.0.0")
        self.assertEqual(record.commit_sha, "abc123")
        self.assertIsNone(record.metadata)
        self.assertIsNone(record.request_id)
        self.assertIsNotNone(record._start_time)

    def test_execution_record_with_metadata(self) -> None:
        """Test ExecutionRecord with optional metadata."""
        mock_client = AsyncMock()
        record = ExecutionRecord(
            api_client=mock_client,
            client_id=42,
            request_json={"test": "data"},
            version="1.0.0",
            commit_sha="abc123",
            metadata={"extra": "info"},
        )

        self.assertEqual(record.metadata, {"extra": "info"})

    def test_execution_record_duration_ms(self) -> None:
        """Test ExecutionRecord._duration_ms calculates correctly."""
        mock_client = AsyncMock()
        record = ExecutionRecord(
            api_client=mock_client, client_id=42, request_json={}, version="1.0.0", commit_sha="abc123"
        )
        time.sleep(0.01)  # Sleep 10ms
        duration = record._duration_ms()

        self.assertGreaterEqual(duration, 10)

    async def test_execution_record_save_success(self) -> None:
        """Test ExecutionRecord.save calls create_execution and sets request_id."""
        mock_client = AsyncMock()
        mock_client.create_execution.return_value = 123

        record = ExecutionRecord(
            api_client=mock_client, client_id=42, request_json={"test": "data"}, version="1.0.0", commit_sha="abc123"
        )
        result = await record.save()

        self.assertTrue(result)
        self.assertEqual(record.request_id, 123)
        mock_client.create_execution.assert_awaited_once_with(
            client_id=42,
            request_json={"test": "data"},
            version="1.0.0",
            commit_sha="abc123",
            metadata=None,
            claimed_by=None,
            lease_seconds=None,
        )

    async def test_execution_record_save_with_metadata(self) -> None:
        """Test ExecutionRecord.save includes metadata when provided."""
        mock_client = AsyncMock()
        mock_client.create_execution.return_value = 123

        record = ExecutionRecord(
            api_client=mock_client,
            client_id=42,
            request_json={"test": "data"},
            version="1.0.0",
            commit_sha="abc123",
            metadata={"source": "test"},
        )
        result = await record.save()

        self.assertTrue(result)
        mock_client.create_execution.assert_awaited_once_with(
            client_id=42,
            request_json={"test": "data"},
            version="1.0.0",
            commit_sha="abc123",
            metadata={"source": "test"},
            claimed_by=None,
            lease_seconds=None,
        )

    async def test_execution_record_save_returns_false_when_none(self) -> None:
        """Test ExecutionRecord.save returns False when API returns None."""
        mock_client = AsyncMock()
        mock_client.create_execution.return_value = None

        record = ExecutionRecord(
            api_client=mock_client, client_id=42, request_json={"test": "data"}, version="1.0.0", commit_sha="abc123"
        )
        result = await record.save()

        self.assertFalse(result)
        self.assertIsNone(record.request_id)

    async def test_execution_record_save_api_error(self) -> None:
        """Test ExecutionRecord.save handles API errors gracefully."""
        mock_client = AsyncMock()
        mock_client.create_execution.side_effect = Exception("API error")

        record = ExecutionRecord(
            api_client=mock_client, client_id=42, request_json={"test": "data"}, version="1.0.0", commit_sha="abc123"
        )
        result = await record.save()

        self.assertFalse(result)
        self.assertIsNone(record.request_id)

    async def test_execution_record_in_progress(self) -> None:
        """Test ExecutionRecord.in_progress flips the row from 'queued' to 'in_progress'.

        The parent row is born ``queued`` at admission and only flips to ``in_progress`` once a pod is seen holding
        one of its checker jobs. This is a single status-only update — ``response_json``/``duration_ms`` come later
        from ``complete()``/``fail()``.
        """
        mock_client = AsyncMock()
        record = ExecutionRecord(
            api_client=mock_client, client_id=42, request_json={"test": "data"}, version="1.0.0", commit_sha="abc123"
        )
        record.request_id = 123

        await record.in_progress()

        mock_client.update_execution.assert_awaited_once_with(
            request_id=123, status="in_progress", response_json=None, duration_ms=None, claimed_by=None
        )

    async def test_execution_record_complete(self) -> None:
        """Test ExecutionRecord.complete updates status with response and returns duration_ms."""
        mock_client = AsyncMock()
        record = ExecutionRecord(
            api_client=mock_client, client_id=42, request_json={"test": "data"}, version="1.0.0", commit_sha="abc123"
        )
        record.request_id = 123

        returned_duration_ms = await record.complete({"result": "success"})

        call_kwargs = mock_client.update_execution.call_args[1]
        self.assertEqual(call_kwargs["request_id"], 123)
        self.assertEqual(call_kwargs["status"], "success")
        self.assertEqual(call_kwargs["response_json"], {"result": "success"})
        self.assertIsNotNone(call_kwargs["duration_ms"])
        # The value returned to the caller must match the value written to the DB
        # and must be a plain int so callers can safely include it in their HTTP response metadata.
        self.assertIs(type(returned_duration_ms), int)
        self.assertEqual(returned_duration_ms, call_kwargs["duration_ms"])

    async def test_execution_record_fail(self) -> None:
        """Test ExecutionRecord.fail updates status with error and returns duration_ms."""
        mock_client = AsyncMock()
        record = ExecutionRecord(
            api_client=mock_client, client_id=42, request_json={"test": "data"}, version="1.0.0", commit_sha="abc123"
        )
        record.request_id = 123

        returned_duration_ms = await record.fail({"error": "Something went wrong"})

        call_kwargs = mock_client.update_execution.call_args[1]
        self.assertEqual(call_kwargs["status"], "fail")
        self.assertEqual(call_kwargs["response_json"], {"error": "Something went wrong"})
        self.assertIs(type(returned_duration_ms), int)
        self.assertEqual(returned_duration_ms, call_kwargs["duration_ms"])

    async def test_execution_record_duration_ms_coerces_float_to_int(self) -> None:
        """duration_ms must be an int even when the wall-clock math yields a float.

        ``(time.monotonic() - start) * 1000`` is a float in the general case;
        the record is contractually required to write (and return) a plain int
        so the DB REST API (Optional[int]) and the HTTP response metadata stay
        consistent.
        """
        mock_client = AsyncMock()
        record = ExecutionRecord(
            api_client=mock_client,
            client_id=42,
            request_json={"test": "data"},
            version="1.0.0",
            commit_sha="abc123",
        )
        record.request_id = 123
        record._start_time = 100.0

        # 100.123456 - 100.0 = 0.123456 seconds -> 123.456 ms (a float) -> int 123.
        # ``_start_time`` is populated by a ``default_factory`` bound at class
        # definition time so we set it explicitly; we only need to drive
        # ``time.monotonic()`` inside ``_duration_ms``.
        with patch.object(execution_record_module.time, "monotonic", return_value=100.123456):
            returned_duration_ms = await record.complete({"result": "success"})

        self.assertIs(type(returned_duration_ms), int)
        self.assertEqual(returned_duration_ms, 123)

        call_kwargs = mock_client.update_execution.call_args[1]
        self.assertIs(type(call_kwargs["duration_ms"]), int)
        self.assertEqual(call_kwargs["duration_ms"], 123)

    async def test_execution_record_update_no_request_id(self) -> None:
        """Test ExecutionRecord update methods do nothing when request_id is None."""
        mock_client = AsyncMock()
        record = ExecutionRecord(
            api_client=mock_client, client_id=42, request_json={"test": "data"}, version="1.0.0", commit_sha="abc123"
        )

        await record.in_progress()
        await record.complete({"result": "ok"})
        await record.fail({"error": "test"})

        mock_client.update_execution.assert_not_awaited()

    async def test_execution_record_update_api_error(self) -> None:
        """Test ExecutionRecord update methods handle API errors gracefully."""
        mock_client = AsyncMock()
        mock_client.update_execution.side_effect = Exception("API error")

        record = ExecutionRecord(
            api_client=mock_client, client_id=42, request_json={"test": "data"}, version="1.0.0", commit_sha="abc123"
        )
        record.request_id = 123

        # Should not raise
        await record.in_progress()
        await record.complete({"result": "ok"})
        await record.fail({"error": "test"})


class TestOwnership(unittest.IsolatedAsyncioTestCase):
    """Holding, renewing and losing the claim on a request."""

    def _owned(self, client: AsyncMock, request_id: int | None = 123) -> ExecutionRecord:
        record = ExecutionRecord(
            api_client=client,
            client_id=42,
            request_json={},
            version="1.0.0",
            commit_sha="abc123",
            claimed_by="replica-a",
            lease_seconds=90.0,
        )
        record.request_id = request_id
        return record

    async def test_a_claim_is_taken_at_insert(self) -> None:
        client = AsyncMock()
        client.create_execution.return_value = 123

        await self._owned(client, request_id=None).save()

        kwargs = client.create_execution.call_args[1]
        self.assertEqual((kwargs["claimed_by"], kwargs["lease_seconds"]), ("replica-a", 90.0))

    async def test_every_write_carries_the_claim(self) -> None:
        client = AsyncMock()
        record = self._owned(client)

        await record.complete({"result": "ok"})

        self.assertEqual(client.update_execution.call_args[1]["claimed_by"], "replica-a")

    async def test_a_superseded_record_writes_nothing_further(self) -> None:
        """Its aggregation is stale; writing it would overwrite the adopter's answer with an older one."""
        client = AsyncMock()
        record = self._owned(client)
        record.stand_down()

        await record.complete({"result": "stale"})

        client.update_execution.assert_not_awaited()

    def test_standing_down_needs_no_round_trip(self) -> None:
        """Whoever holds the row now is the only one entitled to write it, so giving it up is purely local."""
        client = AsyncMock()
        record = self._owned(client)

        record.stand_down()
        record.stand_down()

        self.assertTrue(record.adopted_away)
        client.update_execution.assert_not_awaited()

    async def test_a_rejected_write_supersedes_the_record_too(self) -> None:
        """The loss can surface on the terminal write itself, between two renewals."""
        client = AsyncMock()
        client.update_execution.return_value = False
        record = self._owned(client)

        await record.complete({"result": "stale"})

        self.assertTrue(record.adopted_away)


if __name__ == "__main__":
    unittest.main()
