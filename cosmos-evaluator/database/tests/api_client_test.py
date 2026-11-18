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

"""Unit tests for the orchestration subset of the async ApiClient.

Covers the shared HTTP plumbing (connection handling, retry/idempotent semantics) and the orchestration-table
operations (clients, arbitrator_executions, checker_executions).

These tests assert the contract by patching the underlying ``httpx.AsyncClient`` calls with ``AsyncMock`` and awaiting
from ``IsolatedAsyncioTestCase`` test methods.
"""

from datetime import UTC, datetime
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from database.api_client import ApiClient, CheckerExecutionWrite, CheckerJobHandle, JobOwnershipLost
from database.api_models import (
    ArbitratorExecutionRow,
    CheckerExecutionByVideoRow,
    CheckerExecutionRow,
    CheckerExecutionStateRow,
    ClientRow,
)


def _ok_response(payload: object) -> MagicMock:
    """Build a 200-status mock httpx.Response whose JSON envelope is ``{"data": payload}``."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = 200
    response.raise_for_status = MagicMock()
    response.json.return_value = {"data": payload}
    return response


def _http_error_response(status_code: int, body: object | None = None) -> MagicMock:
    """Build a mock httpx.Response that raises HTTPStatusError on ``raise_for_status``.

    ``body`` is what ``.json()`` plays back, for the handful of errors the client reads detail out of rather than
    just re-raising. Left unset it raises ``ValueError`` the way httpx does on a body that is not JSON, which is
    what proves the client's parsing is tolerant of one.
    """
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    if body is None:
        response.json.side_effect = ValueError("not json")
    else:
        response.json.return_value = body
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        f"{status_code}", request=MagicMock(), response=response
    )
    return response


class _BaseClientTest(unittest.IsolatedAsyncioTestCase):
    """Shared setup for ApiClient tests.

    Uses ``IsolatedAsyncioTestCase`` so each test gets its own event loop; the client opens an
    ``httpx.AsyncClient`` in __init__ which would otherwise leak across tests.
    """

    async def asyncSetUp(self) -> None:
        self.client = ApiClient(base_url="https://db.example.com", api_key="test-key")
        self.mock_response = _ok_response(None)

    async def asyncTearDown(self) -> None:
        await self.client.aclose()


class TestGetExecution(_BaseClientTest):
    """get_execution returns a parsed model, None on 404, raises on other 4xx/5xx."""

    async def test_returns_model_on_200(self) -> None:
        self.mock_response.json.return_value = {
            "data": {"request_id": 7, "status": "in_progress", "timestamp": "2026-01-01T00:00:00"}
        }
        with patch.object(self.client._client, "get", new=AsyncMock(return_value=self.mock_response)) as mock_get:
            result = await self.client.get_execution(7)

        mock_get.assert_called_once_with("/arbitrator-executions/7", params=None)
        assert result is not None
        self.assertIsInstance(result, ArbitratorExecutionRow)
        self.assertEqual(result.request_id, 7)
        self.assertEqual(result.status, "in_progress")

    async def test_returns_none_on_404(self) -> None:
        """404 means the execution simply does not exist; the snapshot endpoint maps this to its own 404."""
        with patch.object(self.client._client, "get", new=AsyncMock(return_value=_http_error_response(404))):
            self.assertIsNone(await self.client.get_execution(9999))

    async def test_raises_on_500(self) -> None:
        """A DB outage should propagate; the snapshot endpoint maps this to 502."""
        with (
            patch.object(self.client._client, "get", new=AsyncMock(return_value=_http_error_response(500))),
            self.assertRaises(httpx.HTTPStatusError),
        ):
            await self.client.get_execution(7)


class TestGetCheckerExecutionsForRequest(_BaseClientTest):
    """get_checker_executions_for_request hits the per-request listing route."""

    async def test_returns_list(self) -> None:
        self.mock_response.json.return_value = {
            "data": [
                {"execution_id": 1, "checker_name": "av.egomotion", "status": "success"},
                {"execution_id": 2, "checker_name": "av.objects_single_view", "status": "in_progress"},
            ]
        }
        with patch.object(self.client._client, "get", new=AsyncMock(return_value=self.mock_response)) as mock_get:
            result = await self.client.get_checker_executions_for_request(7)

        mock_get.assert_called_once_with("/arbitrator-executions/7/checker-executions", params=None)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], CheckerExecutionRow)
        self.assertEqual(result[0].execution_id, 1)

    async def test_returns_empty_list_when_data_is_none(self) -> None:
        """A just-submitted run with zero checker rows is a valid state, not an error."""
        self.mock_response.json.return_value = {"data": None}
        with patch.object(self.client._client, "get", new=AsyncMock(return_value=self.mock_response)):
            result = await self.client.get_checker_executions_for_request(7)
        self.assertEqual(result, [])


class TestGetCheckerExecutionsByVideo(_BaseClientTest):
    async def test_returns_list_with_filters(self) -> None:
        self.mock_response.json.return_value = {"data": [{"execution_id": 1, "video_uuid": "vid-123"}]}
        with patch.object(self.client._client, "get", new=AsyncMock(return_value=self.mock_response)) as mock_get:
            result = await self.client.get_checker_executions_by_video("vid-123", checker_name="av.objects_single_view")

        mock_get.assert_called_once_with(
            "/checker-executions/by-video/vid-123",
            params={"status": "success", "checker_name": "av.objects_single_view"},
        )
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], CheckerExecutionByVideoRow)
        self.assertEqual(result[0].execution_id, 1)

    async def test_passes_camera_name_filter(self) -> None:
        """camera_name filter is forwarded as a query param."""
        self.mock_response.json.return_value = {"data": []}
        with patch.object(self.client._client, "get", new=AsyncMock(return_value=self.mock_response)) as mock_get:
            await self.client.get_checker_executions_by_video(
                "vid-123",
                checker_name="av.egomotion",
                camera_name="camera_front_wide_120fov",
            )
        params = mock_get.call_args[1]["params"]
        self.assertEqual(params["camera_name"], "camera_front_wide_120fov")
        self.assertEqual(params["checker_name"], "av.egomotion")

    async def test_returns_empty_list_when_no_data(self) -> None:
        self.mock_response.json.return_value = {"data": None}
        with patch.object(self.client._client, "get", new=AsyncMock(return_value=self.mock_response)):
            result = await self.client.get_checker_executions_by_video("vid-123")
        self.assertEqual(result, [])


class TestRecordClient(_BaseClientTest):
    async def test_posts_client_record(self) -> None:
        self.mock_response.json.return_value = {
            "data": {"client_id": 1, "ip_address": "10.0.0.1", "name": "test", "description": None}
        }
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=self.mock_response)) as mock_post:
            result = await self.client.record_client("10.0.0.1", name="test")
        mock_post.assert_called_once_with("/clients", json={"ip_address": "10.0.0.1", "name": "test"})
        assert result is not None
        self.assertIsInstance(result, ClientRow)
        self.assertEqual(result.client_id, 1)

    async def test_omits_none_fields(self) -> None:
        self.mock_response.json.return_value = {"data": {"client_id": 1, "ip_address": "10.0.0.1"}}
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=self.mock_response)) as mock_post:
            await self.client.record_client("10.0.0.1")
        mock_post.assert_called_once_with("/clients", json={"ip_address": "10.0.0.1"})


class TestCreateExecution(_BaseClientTest):
    async def test_returns_request_id(self) -> None:
        self.mock_response.json.return_value = {"data": {"request_id": 42}}
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=self.mock_response)) as mock_post:
            result = await self.client.create_execution(
                client_id=1, request_json={"test": True}, version="1.0", commit_sha="abc"
            )
        self.assertEqual(result, 42)
        call_json = mock_post.call_args[1]["json"]
        self.assertEqual(call_json["client_id"], 1)
        self.assertEqual(call_json["version"], "1.0")

    async def test_includes_metadata_when_provided(self) -> None:
        self.mock_response.json.return_value = {"data": {"request_id": 42}}
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=self.mock_response)) as mock_post:
            await self.client.create_execution(
                client_id=1, request_json={}, version="1.0", commit_sha="abc", metadata={"env": "test"}
            )
        call_json = mock_post.call_args[1]["json"]
        self.assertEqual(call_json["metadata"], {"env": "test"})

    async def test_returns_none_when_data_is_none(self) -> None:
        self.mock_response.json.return_value = {"data": None}
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=self.mock_response)):
            result = await self.client.create_execution(client_id=1, request_json={}, version="1.0", commit_sha="abc")
        self.assertIsNone(result)

    async def test_half_an_ownership_claim_never_reaches_the_wire(self) -> None:
        """A lease with no owner would be dropped on the way out, leaving the caller sure it had claimed the row."""
        for overrides in ({"claimed_by": "pod-a"}, {"lease_seconds": 60.0}):
            with self.subTest(**overrides):
                post = AsyncMock(return_value=self.mock_response)
                with patch.object(self.client._client, "post", new=post), self.assertRaises(ValueError):
                    await self.client.create_execution(
                        client_id=1, request_json={}, version="1.0", commit_sha="abc", **overrides
                    )
                post.assert_not_awaited()

    async def test_a_whole_ownership_claim_is_sent_as_given(self) -> None:
        self.mock_response.json.return_value = {"data": {"request_id": 42}}
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=self.mock_response)) as mock_post:
            await self.client.create_execution(
                client_id=1,
                request_json={},
                version="1.0",
                commit_sha="abc",
                claimed_by="pod-a",
                lease_seconds=60.0,
            )
        call_json = mock_post.call_args[1]["json"]
        self.assertEqual(call_json["claimed_by"], "pod-a")
        self.assertEqual(call_json["lease_seconds"], 60.0)


class TestUpdateExecution(_BaseClientTest):
    async def test_puts_status_update(self) -> None:
        self.mock_response.json.return_value = {"data": {"request_id": 42, "status": "success"}}
        with patch.object(self.client._client, "put", new=AsyncMock(return_value=self.mock_response)) as mock_put:
            result = await self.client.update_execution(42, "success", response_json={"ok": True}, duration_ms=150)
        self.assertTrue(result)
        mock_put.assert_called_once_with(
            "/arbitrator-executions/42",
            json={"status": "success", "response_json": {"ok": True}, "duration_ms": 150},
        )

    async def test_omits_none_optional_fields(self) -> None:
        self.mock_response.json.return_value = {"data": {"request_id": 42}}
        with patch.object(self.client._client, "put", new=AsyncMock(return_value=self.mock_response)) as mock_put:
            await self.client.update_execution(42, "running")
        mock_put.assert_called_once_with("/arbitrator-executions/42", json={"status": "running"})

    async def test_sends_the_fencing_token_when_the_caller_owns_the_request(self) -> None:
        self.mock_response.json.return_value = {"data": {"request_id": 42}}
        with patch.object(self.client._client, "put", new=AsyncMock(return_value=self.mock_response)) as mock_put:
            await self.client.update_execution(42, "success", claimed_by="replica-a")
        self.assertEqual(mock_put.call_args[1]["json"]["claimed_by"], "replica-a")

    async def test_reports_a_fenced_write_that_matched_nothing(self) -> None:
        """The request was adopted mid-flight; the caller has to know its aggregation is now stale."""
        with patch.object(self.client._client, "put", new=AsyncMock(return_value=_http_error_response(404))):
            self.assertFalse(await self.client.update_execution(42, "success", claimed_by="replica-a"))

    async def test_an_unfenced_404_still_raises(self) -> None:
        """Without a claim there is no "you were superseded" to report, so a missing row is an ordinary error."""
        with (
            patch.object(self.client._client, "put", new=AsyncMock(return_value=_http_error_response(404))),
            self.assertRaises(httpx.HTTPStatusError),
        ):
            await self.client.update_execution(42, "success")


class TestExecutionOwnership(_BaseClientTest):
    """Renewing and adopting the claim on an arbitrator request."""

    async def test_heartbeat_renews_every_claim_in_one_call(self) -> None:
        self.mock_response.json.return_value = {"data": {"renewed": [42, 43], "lost": []}}
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=self.mock_response)) as mock_post:
            renewal = await self.client.heartbeat_executions([42, 43], "replica-a", 90)
        self.assertEqual(renewal.renewed, [42, 43])
        mock_post.assert_called_once_with(
            "/arbitrator-executions/heartbeat",
            json={"request_ids": [42, 43], "claimed_by": "replica-a", "lease_seconds": 90},
        )

    async def test_heartbeat_carries_back_why_each_claim_was_lost(self) -> None:
        """Adopted and already-finished are both losses, and the caller has to tell them apart."""
        self.mock_response.json.return_value = {
            "data": {
                "renewed": [],
                "lost": [
                    {"request_id": 42, "status": "in_progress", "claimed_by": "replica-b"},
                    {"request_id": 43, "status": "success", "claimed_by": "replica-a"},
                ],
            }
        }
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=self.mock_response)):
            renewal = await self.client.heartbeat_executions([42, 43], "replica-a", 90)

        self.assertEqual([lost.claimed_by for lost in renewal.lost], ["replica-b", "replica-a"])
        self.assertEqual([lost.status for lost in renewal.lost], ["in_progress", "success"])

    async def test_heartbeat_raises_on_an_outage(self) -> None:
        """A 503 must not be read as loss, or a replica abandons a request nobody else has taken."""
        with (
            patch.object(self.client._client, "post", new=AsyncMock(return_value=_http_error_response(503))),
            self.assertRaises(httpx.HTTPStatusError),
        ):
            await self.client.heartbeat_executions([42], "replica-a", 90)

    async def test_adopt_returns_parsed_rows(self) -> None:
        self.mock_response.json.return_value = {
            "data": {
                "executions": [
                    {
                        "request_id": 7,
                        "client_id": 3,
                        "status": "in_progress",
                        "request_json": {"inputs": {}},
                        "claimed_by": "replica-b",
                    }
                ]
            }
        }
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=self.mock_response)) as mock_post:
            rows = await self.client.adopt_executions("replica-b", 90, limit=5)
        mock_post.assert_called_once_with(
            "/arbitrator-executions/adopt", json={"claimed_by": "replica-b", "lease_seconds": 90, "limit": 5}
        )
        self.assertEqual([row.request_id for row in rows], [7])
        self.assertEqual(rows[0].request_json, {"inputs": {}})

    async def test_adopt_returns_empty_when_nothing_was_stale(self) -> None:
        self.mock_response.json.return_value = {"data": {"executions": []}}
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=self.mock_response)):
            self.assertEqual(await self.client.adopt_executions("replica-b", 90), [])

    async def test_checker_states_for_a_request_carry_the_coordination_columns(self) -> None:
        self.mock_response.json.return_value = {
            "data": [{"execution_id": 3, "status": "in_progress", "claimed_by": "pod-a", "attempt": 2}]
        }
        with patch.object(self.client._client, "get", new=AsyncMock(return_value=self.mock_response)) as mock_get:
            rows = await self.client.get_checker_execution_states_for_request(7)
        mock_get.assert_called_once_with("/arbitrator-executions/7/checker-states", params=None)
        self.assertEqual((rows[0].claimed_by, rows[0].attempt), ("pod-a", 2))


class TestCreateCheckerExecution(_BaseClientTest):
    async def test_returns_execution_id(self) -> None:
        self.mock_response.json.return_value = {"data": {"execution_id": 99}}
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=self.mock_response)) as mock_post:
            result = await self.client.create_checker_execution(
                request_id=42, checker_name="av.objects_single_view", request_json={"camera": "front"}
            )
        self.assertEqual(result, 99)
        call_json = mock_post.call_args[1]["json"]
        self.assertEqual(call_json["request_id"], 42)
        self.assertEqual(call_json["checker_name"], "av.objects_single_view")
        self.assertNotIn("camera_name", call_json)

    async def test_returns_none_when_data_is_none(self) -> None:
        self.mock_response.json.return_value = {"data": None}
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=self.mock_response)):
            result = await self.client.create_checker_execution(
                request_id=42, checker_name="av.objects_single_view", request_json={}
            )
        self.assertIsNone(result)

    async def test_includes_camera_name_in_payload(self) -> None:
        """Per-camera fan-out: camera_name is passed in the POST body."""
        self.mock_response.json.return_value = {"data": {"execution_id": 100}}
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=self.mock_response)) as mock_post:
            await self.client.create_checker_execution(
                request_id=42,
                checker_name="av.egomotion",
                request_json={},
                camera_name="camera_front_wide_120fov",
            )
        call_json = mock_post.call_args[1]["json"]
        self.assertEqual(call_json["camera_name"], "camera_front_wide_120fov")
        self.assertEqual(call_json["checker_name"], "av.egomotion")


class TestUpdateCheckerExecution(_BaseClientTest):
    async def test_puts_final_state(self) -> None:
        self.mock_response.json.return_value = {"data": {"execution_id": 99, "status": "success"}}
        with patch.object(self.client._client, "put", new=AsyncMock(return_value=self.mock_response)) as mock_put:
            result = await self.client.update_checker_execution(
                99, "success", response_json={"score": 0.95}, duration_ms=3200, output_metadata={"frames": 100}
            )
        self.assertEqual(result, CheckerExecutionWrite(written=True, existing_status=None))
        mock_put.assert_called_once_with(
            "/checker-executions/99",
            json={
                "status": "success",
                "response_json": {"score": 0.95},
                "duration_ms": 3200,
                "output_metadata": {"frames": 100},
            },
        )

    async def test_includes_error_message(self) -> None:
        self.mock_response.json.return_value = {"data": {"execution_id": 99}}
        with patch.object(self.client._client, "put", new=AsyncMock(return_value=self.mock_response)) as mock_put:
            await self.client.update_checker_execution(99, "fail", error_message="Timeout after 600s")
        call_json = mock_put.call_args[1]["json"]
        self.assertEqual(call_json["error_message"], "Timeout after 600s")

    async def test_puts_the_job_payload(self) -> None:
        """The payload is written to the row, because that is where the claiming pod reads it."""
        self.mock_response.json.return_value = {"data": {"execution_id": 99}}
        with patch.object(self.client._client, "put", new=AsyncMock(return_value=self.mock_response)) as mock_put:
            await self.client.update_checker_execution(99, "queued", request_json={"clip_id": "c"})
        mock_put.assert_called_once_with(
            "/checker-executions/99", json={"status": "queued", "request_json": {"clip_id": "c"}}
        )

    async def test_a_refused_write_is_a_value_not_an_exception(self) -> None:
        """409 means the row already finished, which on this path is an expected race rather than an error.

        Returned as a value because callers branch on it; raising would push them into using exceptions for
        control flow on the ordinary case of a job finishing between a poll and the write it decided on.
        """
        refused = _http_error_response(409, body={"metadata": {"existing_status": "success"}})
        with patch.object(self.client._client, "put", new=AsyncMock(return_value=refused)):
            result = await self.client.update_checker_execution(99, "in_progress")

        self.assertEqual(result, CheckerExecutionWrite(written=False, existing_status="success"))

    async def test_a_refused_write_survives_a_body_it_cannot_parse(self) -> None:
        """The status code already settled the outcome, so an unreadable body must not replace it with an error."""
        refused = _http_error_response(409)
        with patch.object(self.client._client, "put", new=AsyncMock(return_value=refused)):
            result = await self.client.update_checker_execution(99, "in_progress")

        self.assertEqual(result, CheckerExecutionWrite(written=False, existing_status=None))

    async def test_a_missing_row_still_raises(self) -> None:
        """A row that is gone is a real inconsistency, not a race the caller can absorb by carrying on."""
        with (
            patch.object(self.client._client, "put", new=AsyncMock(return_value=_http_error_response(404))),
            self.assertRaises(httpx.HTTPStatusError),
        ):
            await self.client.update_checker_execution(99, "in_progress")


class TestApiKeyHeader(_BaseClientTest):
    async def test_sets_api_key_header(self) -> None:
        self.assertEqual(self.client._client.headers.get("X-API-Key"), "test-key")

    async def test_strips_trailing_slash_from_base_url(self) -> None:
        client = ApiClient(base_url="https://db.example.com/", api_key="k")
        try:
            self.assertEqual(str(client._client.base_url), "https://db.example.com")
        finally:
            await client.aclose()


class TestAclose(_BaseClientTest):
    async def test_aclose_calls_client_aclose(self) -> None:
        with patch.object(self.client._client, "aclose", new=AsyncMock()) as mock_aclose:
            await self.client.aclose()
        mock_aclose.assert_awaited_once()


class TestRetryOnConnectionError(unittest.IsolatedAsyncioTestCase):
    """Verify that connection-level errors trigger retries with backoff.

    This suite uses ``backoff_base=0.0`` so retries don't actually sleep; the
    ``test_backoff_delays`` case below switches to non-zero values and patches
    ``asyncio.sleep`` to assert the requested delay schedule.
    """

    async def asyncSetUp(self) -> None:
        self.client = ApiClient(
            base_url="https://db.example.com",
            api_key="test-key",
            max_retries=3,
            backoff_base=0.0,
        )
        self.mock_response = _ok_response([{"execution_id": 1}])

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def test_succeeds_after_transient_connect_error(self) -> None:
        with patch.object(
            self.client._client,
            "get",
            new=AsyncMock(side_effect=[httpx.ConnectError("SSL handshake timed out"), self.mock_response]),
        ):
            result = await self.client.get_checker_executions_for_request(7)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].execution_id, 1)

    async def test_succeeds_after_transient_connect_timeout(self) -> None:
        with patch.object(
            self.client._client,
            "get",
            new=AsyncMock(side_effect=[httpx.ConnectTimeout("connection timed out"), self.mock_response]),
        ):
            result = await self.client.get_checker_executions_for_request(7)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].execution_id, 1)

    async def test_raises_after_max_retries_exhausted(self) -> None:
        with (
            patch.object(
                self.client._client,
                "get",
                new=AsyncMock(side_effect=httpx.ConnectError("SSL handshake timed out")),
            ),
            self.assertRaises(httpx.ConnectError),
        ):
            await self.client.get_checker_executions_for_request(7)

    async def test_does_not_retry_http_status_errors(self) -> None:
        error_response = _http_error_response(500)
        with (
            patch.object(self.client._client, "get", new=AsyncMock(return_value=error_response)) as mock_get,
            self.assertRaises(httpx.HTTPStatusError),
        ):
            await self.client.get_checker_executions_for_request(7)
        mock_get.assert_called_once()

    async def test_retry_works_for_post(self) -> None:
        self.mock_response.json.return_value = {"data": {"request_id": 42}}
        with patch.object(
            self.client._client,
            "post",
            new=AsyncMock(side_effect=[httpx.ConnectError("refused"), self.mock_response]),
        ):
            result = await self.client.create_execution(client_id=1, request_json={}, version="1.0", commit_sha="abc")
        self.assertEqual(result, 42)

    async def test_retry_works_for_put(self) -> None:
        self.mock_response.json.return_value = {"data": {"request_id": 42}}
        with patch.object(
            self.client._client,
            "put",
            new=AsyncMock(side_effect=[httpx.ConnectTimeout("timeout"), self.mock_response]),
        ):
            result = await self.client.update_execution(42, "success")
        self.assertTrue(result)

    async def test_retry_works_for_delete(self) -> None:
        self.mock_response.json.return_value = {"data": {"deleted": True}}
        with patch.object(
            self.client._client,
            "delete",
            new=AsyncMock(side_effect=[httpx.ConnectError("reset"), self.mock_response]),
        ):
            result = await self.client._delete("/some-resource/1")
        self.assertEqual(result, {"deleted": True})

    async def test_backoff_delays(self) -> None:
        """Verify exponential backoff timing between retries."""
        client = ApiClient(
            base_url="https://db.example.com",
            api_key="test-key",
            max_retries=3,
            backoff_base=1.0,
            backoff_factor=2.0,
        )
        try:
            with (
                patch.object(
                    client._client,
                    "get",
                    new=AsyncMock(side_effect=httpx.ConnectError("fail")),
                ),
                patch("database.api_client.asyncio.sleep", new=AsyncMock()) as mock_sleep,
                self.assertRaises(httpx.ConnectError),
            ):
                await client.get_checker_executions_for_request(7)
            # 3 retries → 2 sleeps (after attempt 1 and 2, not after final failure).
            self.assertEqual(mock_sleep.call_count, 2)
            mock_sleep.assert_any_await(1.0)  # 1.0 * 2^0
            mock_sleep.assert_any_await(2.0)  # 1.0 * 2^1
        finally:
            await client.aclose()


class TestResponseModels(unittest.TestCase):
    """Contract tests for the api_models row shapes returned by the read methods."""

    def test_unknown_columns_survive_round_trip(self) -> None:
        """extra="allow": a column the client doesn't declare is preserved through parse -> model_dump so the
        arbitrator can forward whole rows over the wire without silently dropping new server columns."""
        row = CheckerExecutionRow.model_validate({"execution_id": 5, "status": "success", "some_new_column": "keep-me"})
        dumped = row.model_dump()
        self.assertEqual(dumped["some_new_column"], "keep-me")
        self.assertEqual(dumped["execution_id"], 5)

    def test_project_out_response_json(self) -> None:
        """The snapshot projection drops response_json while keeping every other (including extra) column."""
        row = CheckerExecutionRow.model_validate(
            {"execution_id": 5, "status": "success", "response_json": {"big": "blob"}, "extra_col": 1}
        )
        projected = row.model_dump(exclude={"response_json"})
        self.assertNotIn("response_json", projected)
        self.assertEqual(projected["extra_col"], 1)

    def test_nullable_columns_default_to_none(self) -> None:
        """Only the primary key is required on a checker row; the rest default to None."""
        row = CheckerExecutionRow.model_validate({"execution_id": 9})
        self.assertIsNone(row.status)
        self.assertIsNone(row.response_json)
        self.assertIsNone(row.start_time)

    def test_by_video_subclass_carries_arbitrator_columns(self) -> None:
        """The by-video projection is a superset that adds the video UUID and the joined arbitrator provenance."""
        row = CheckerExecutionByVideoRow.model_validate(
            {
                "execution_id": 1,
                "video_uuid": "vid-123",
                "version": "9.9.9",
                "commit_sha": "checker-sha",
                "arbitrator_version": "1.2.3",
                "arbitrator_commit_sha": "abc123",
                "arbitrator_timestamp": "2026-01-01T00:00:30",
            }
        )
        self.assertEqual(row.video_uuid, "vid-123")
        self.assertEqual(row.arbitrator_version, "1.2.3")
        self.assertEqual(row.arbitrator_commit_sha, "abc123")
        self.assertEqual(row.arbitrator_timestamp, "2026-01-01T00:00:30")
        # The unprefixed pair stays the checker's own, which is the distinction the prefix exists to draw.
        self.assertEqual(row.version, "9.9.9")
        self.assertEqual(row.commit_sha, "checker-sha")

    def test_by_video_tolerates_a_row_without_the_video_uuid(self) -> None:
        """The UUID only echoes the query argument, and the request-scoped sibling omits it, so it is not required."""
        row = CheckerExecutionByVideoRow.model_validate({"execution_id": 1})
        self.assertIsNone(row.video_uuid)


class TestClaimNextCheckerJobs(_BaseClientTest):
    """claim_next_checker_jobs turns whatever is queued for a checker into handles the caller can act on."""

    async def test_each_claimed_row_becomes_an_owned_handle(self) -> None:
        response = _ok_response(
            {
                "jobs": [
                    {
                        "execution_id": 7,
                        "claimed_by": "pod-a",
                        "attempt": 1,
                        "checker_name": "av.egomotion",
                        "camera_name": "camera_front_wide_120fov",
                        "request_id": 3,
                        "request_json": {"clip_id": "c"},
                    },
                    {"execution_id": 8, "claimed_by": "pod-a", "attempt": 1, "checker_name": "av.egomotion"},
                ]
            }
        )
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=response)):
            handles = await self.client.claim_next_checker_jobs("av.egomotion", "pod-a", 30.0, limit=2)

        self.assertEqual([h.execution_id for h in handles], [7, 8])
        self.assertEqual(handles[0].checker_name, "av.egomotion")
        self.assertEqual(handles[0].camera_name, "camera_front_wide_120fov")
        self.assertEqual(handles[0].request_id, 3)
        self.assertEqual(handles[0].request_json, {"clip_id": "c"})

    async def test_an_empty_queue_is_an_empty_list(self) -> None:
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=_ok_response({"jobs": []}))):
            self.assertEqual(await self.client.claim_next_checker_jobs("av.vlm", "pod-a", 30.0), [])

    async def test_the_request_asks_for_exactly_what_the_caller_can_start(self) -> None:
        with patch.object(
            self.client._client, "post", new=AsyncMock(return_value=_ok_response({"jobs": []}))
        ) as mock_post:
            await self.client.claim_next_checker_jobs("av.vlm", "pod-a", 30.0, limit=3, wait_seconds=5.0)

        self.assertEqual(mock_post.call_args.args[0], "/checker-executions/claim-next")
        self.assertEqual(
            mock_post.call_args.kwargs["json"],
            {
                "checker_name": "av.vlm",
                "claimed_by": "pod-a",
                "lease_seconds": 30.0,
                "limit": 3,
                "wait_seconds": 5.0,
            },
        )

    async def test_the_claimer_s_build_travels_with_the_request(self) -> None:
        with patch.object(
            self.client._client, "post", new=AsyncMock(return_value=_ok_response({"jobs": []}))
        ) as mock_post:
            await self.client.claim_next_checker_jobs(
                "av.vlm", "pod-a", 30.0, wait_seconds=0.0, version="1.44.0", commit_sha="abc"
            )

        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual((payload["version"], payload["commit_sha"]), ("1.44.0", "abc"))

    async def test_the_read_deadline_outlives_the_wait_the_server_was_asked_for(self) -> None:
        # The client's default read deadline is shorter than the longest wait the server will hold. Without an override
        # the client would time out on its own request and read an idle queue as a database failure.
        with patch.object(
            self.client._client, "post", new=AsyncMock(return_value=_ok_response({"jobs": []}))
        ) as mock_post:
            await self.client.claim_next_checker_jobs("av.vlm", "pod-a", 30.0, wait_seconds=20.0)
        self.assertGreater(mock_post.call_args.kwargs["timeout"], 20.0)

    async def test_nonsensical_polling_arguments_never_reach_the_wire(self) -> None:
        """These sit in the local timeout and slot arithmetic, so a bad one misconfigures the loop before any reply."""
        cases: list[tuple[str, float, int, float]] = [
            # label, lease_seconds, limit, wait_seconds
            ("no room asked for", 30.0, 0, 5.0),
            ("negative limit", 30.0, -1, 5.0),
            ("negative wait", 30.0, 1, -1.0),
            ("lease already expired", 0.0, 1, 5.0),
            ("negative lease", -30.0, 1, 5.0),
        ]
        for label, lease_seconds, limit, wait_seconds in cases:
            with self.subTest(label):
                post = AsyncMock(return_value=_ok_response({"jobs": []}))
                with patch.object(self.client._client, "post", new=post), self.assertRaises(ValueError):
                    await self.client.claim_next_checker_jobs(
                        "av.vlm", "pod-a", lease_seconds, limit=limit, wait_seconds=wait_seconds
                    )
                post.assert_not_awaited()


class TestRequeueExpiredCheckerJobs(_BaseClientTest):
    """requeue_expired_checker_jobs reports which abandoned jobs the sweep returned to the queue."""

    async def test_the_requeued_rows_come_back(self) -> None:
        response = _ok_response({"requeued": [{"execution_id": 7, "checker_name": "av.vlm", "attempt": 2}]})
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=response)) as mock_post:
            requeued = await self.client.requeue_expired_checker_jobs(limit=50)

        mock_post.assert_called_once_with("/checker-executions/reap", json={"limit": 50})
        self.assertEqual(requeued[0].execution_id, 7)
        self.assertEqual(requeued[0].attempt, 2)

    async def test_a_sweep_that_found_nothing_is_an_empty_list(self) -> None:
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=_ok_response({"requeued": []}))):
            self.assertEqual(await self.client.requeue_expired_checker_jobs(), [])


class TestCheckerJobHandle(_BaseClientTest):
    """The handle carries the fencing token into every write and refuses to keep going once it is superseded."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        claim = _ok_response({"jobs": [{"execution_id": 7, "claimed_by": "pod-a", "attempt": 1}]})
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=claim)):
            handles = await self.client.claim_next_checker_jobs("av.egomotion", "pod-a", 30.0)
        self.handle = handles[0]

    async def test_heartbeat_sends_the_token_and_returns_state(self) -> None:
        response = _ok_response({"execution_id": 7, "cancel_requested": True})
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=response)) as mock_post:
            state = await self.handle.heartbeat(30.0)

        mock_post.assert_called_once_with(
            "/checker-executions/7/heartbeat",
            json={"claimed_by": "pod-a", "lease_seconds": 30.0},
        )
        self.assertTrue(state.cancel_requested)

    async def test_complete_sends_the_token_and_a_success_status(self) -> None:
        response = _ok_response({"execution_id": 7, "status": "success"})
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=response)) as mock_post:
            await self.handle.complete({"score": 1}, duration_ms=900)

        mock_post.assert_called_once_with(
            "/checker-executions/7/complete",
            json={
                "claimed_by": "pod-a",
                "status": "success",
                "response_json": {"score": 1},
                "duration_ms": 900,
            },
        )

    async def test_fail_reports_the_error(self) -> None:
        response = _ok_response({"execution_id": 7, "status": "fail"})
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=response)) as mock_post:
            await self.handle.fail("boom", duration_ms=100)

        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["status"], "fail")
        self.assertEqual(payload["error_message"], "boom")

    async def test_fail_can_report_a_timeout(self) -> None:
        response = _ok_response({"execution_id": 7, "status": "timeout"})
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=response)) as mock_post:
            await self.handle.fail("too slow", timed_out=True)
        self.assertEqual(mock_post.call_args.kwargs["json"]["status"], "timeout")

    async def test_heartbeat_raises_ownership_lost_on_409(self) -> None:
        with (
            patch.object(self.client._client, "post", new=AsyncMock(return_value=_http_error_response(409))),
            self.assertRaises(JobOwnershipLost),
        ):
            await self.handle.heartbeat(30.0)

    async def test_complete_raises_ownership_lost_on_409(self) -> None:
        # This is the case the whole fencing design exists for: the pod is still running but its job was reassigned,
        # so its result must be refused rather than allowed to overwrite the replacement's.
        with (
            patch.object(self.client._client, "post", new=AsyncMock(return_value=_http_error_response(409))),
            self.assertRaises(JobOwnershipLost),
        ):
            await self.handle.complete({"score": 1})

    async def test_other_errors_are_not_mistaken_for_ownership_loss(self) -> None:
        with (
            patch.object(self.client._client, "post", new=AsyncMock(return_value=_http_error_response(500))),
            self.assertRaises(httpx.HTTPStatusError),
        ):
            await self.handle.heartbeat(30.0)


class TestRequestCheckerCancel(_BaseClientTest):
    """request_checker_cancel reports whether there was still anything to cancel."""

    async def test_returns_true_when_flagged(self) -> None:
        response = _ok_response({"execution_id": 7, "cancel_requested": True})
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=response)) as mock_post:
            self.assertTrue(await self.client.request_checker_cancel(7))
        mock_post.assert_called_once_with("/checker-executions/7/cancel", json={})

    async def test_returns_false_on_404(self) -> None:
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=_http_error_response(404))):
            self.assertFalse(await self.client.request_checker_cancel(7))


class TestGetCheckerExecutionStates(_BaseClientTest):
    """get_checker_execution_states batches the poll into one request."""

    async def test_returns_parsed_rows(self) -> None:
        response = _ok_response([{"execution_id": 7, "status": "in_progress", "claimed_by": "pod-a", "attempt": 1}])
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=response)) as mock_post:
            rows = await self.client.get_checker_execution_states([7, 8])

        mock_post.assert_called_once_with("/checker-executions/states", json={"execution_ids": [7, 8]})
        self.assertIsInstance(rows[0], CheckerExecutionStateRow)
        self.assertEqual(rows[0].claimed_by, "pod-a")
        self.assertEqual(rows[0].attempt, 1)

    async def test_empty_id_list_skips_the_request(self) -> None:
        with patch.object(self.client._client, "post", new=AsyncMock()) as mock_post:
            self.assertEqual(await self.client.get_checker_execution_states([]), [])
        mock_post.assert_not_called()


class TestMarkCheckerExecutionsInProgress(_BaseClientTest):
    """The write half of a polling pass, batched to match the read."""

    async def test_refusals_come_back_mapped_to_the_status_that_won(self) -> None:
        response = _ok_response({"refused": [{"execution_id": 7, "existing_status": "success"}]})
        start = datetime(2026, 9, 3, 12, 0, tzinfo=UTC)
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=response)) as mock_post:
            refused = await self.client.mark_checker_executions_in_progress([7, 8], start)

        mock_post.assert_called_once_with(
            "/checker-executions/in-progress",
            json={"execution_ids": [7, 8], "start_time": "2026-09-03T12:00:00+00:00"},
        )
        self.assertEqual(refused, {7: "success"})

    async def test_an_accepted_batch_reports_nothing(self) -> None:
        """Absence is the signal: an id the answer does not mention was written, so its job is running."""
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=_ok_response({"refused": []}))):
            refused = await self.client.mark_checker_executions_in_progress([7, 8], datetime.now(UTC))

        self.assertEqual(refused, {})

    async def test_empty_id_list_skips_the_request(self) -> None:
        with patch.object(self.client._client, "post", new=AsyncMock()) as mock_post:
            self.assertEqual(await self.client.mark_checker_executions_in_progress([], datetime.now(UTC)), {})
        mock_post.assert_not_called()

    async def test_a_failure_is_raised_rather_than_read_as_an_accepted_batch(self) -> None:
        """Silently answering "nothing refused" would have the caller start budget clocks on finished jobs.

        The arbitrator's poll catches this and carries on, which is the right reaction -- but it has to be its
        decision, made knowing the write did not land, not one forced by a client that cannot tell it.
        """
        with (
            patch.object(self.client._client, "post", new=AsyncMock(return_value=_http_error_response(503))),
            self.assertRaises(httpx.HTTPStatusError),
        ):
            await self.client.mark_checker_executions_in_progress([7], datetime.now(UTC))


class TestRetryOnTransientUpstreamStatus(unittest.IsolatedAsyncioTestCase):
    """What a call does with a transient 502 decides what a routine deploy of the database service costs.

    The service runs a single replica, so every rollout leaves a window where the gateway answers 502 for requests
    aimed at the draining pod. Both sides of the split are pinned here because both failures are silent: a dropped
    terminal write re-runs a check that already succeeded, and a replayed insert leaves behind a row nothing finishes.

    ``backoff_base=0.0`` keeps the retries from actually sleeping.
    """

    async def asyncSetUp(self) -> None:
        self.client = ApiClient(base_url="https://db.example.com", api_key="test-key", max_retries=3, backoff_base=0.0)

    async def asyncTearDown(self) -> None:
        await self.client.aclose()

    async def _claim_one_job(self) -> CheckerJobHandle:
        claim = _ok_response({"jobs": [{"execution_id": 7, "claimed_by": "pod-a", "attempt": 1}]})
        with patch.object(self.client._client, "post", new=AsyncMock(return_value=claim)):
            handles = await self.client.claim_next_checker_jobs("av.egomotion", "pod-a", 30.0)
        return handles[0]

    async def test_a_terminal_write_survives_the_gateway_dropping_it(self) -> None:
        """The write worth retrying: without this the job looks abandoned and its whole check runs again."""
        handle = await self._claim_one_job()
        with patch.object(
            self.client._client,
            "post",
            new=AsyncMock(side_effect=[_http_error_response(502), _ok_response(None)]),
        ) as mock_post:
            await handle.complete(response_json={"score": 1}, duration_ms=5)
        self.assertEqual(mock_post.await_count, 2)

    async def test_a_lease_renewal_survives_it_too(self) -> None:
        handle = await self._claim_one_job()
        with patch.object(
            self.client._client,
            "post",
            new=AsyncMock(side_effect=[_http_error_response(503), _ok_response({"execution_id": 7})]),
        ) as mock_post:
            await handle.heartbeat(120.0)
        self.assertEqual(mock_post.await_count, 2)

    async def test_the_reaper_sweep_survives_it(self) -> None:
        with patch.object(
            self.client._client,
            "post",
            new=AsyncMock(side_effect=[_http_error_response(502), _ok_response({"requeued": []})]),
        ) as mock_post:
            self.assertEqual(await self.client.requeue_expired_checker_jobs(), [])
        self.assertEqual(mock_post.await_count, 2)

    async def test_creating_a_request_row_is_not_replayed(self) -> None:
        """The server generates the id, so a second insert is a second row that nothing will ever finish."""
        with (
            patch.object(self.client._client, "post", new=AsyncMock(return_value=_http_error_response(502))) as post,
            self.assertRaises(httpx.HTTPStatusError),
        ):
            await self.client.create_execution(client_id=1, request_json={}, version="1.0", commit_sha="abc")
        post.assert_awaited_once()

    async def test_claiming_work_is_not_replayed(self) -> None:
        """A replay would take a different batch and strand the first one until its lease lapsed."""
        with (
            patch.object(self.client._client, "post", new=AsyncMock(return_value=_http_error_response(503))) as post,
            self.assertRaises(httpx.HTTPStatusError),
        ):
            await self.client.claim_next_checker_jobs("av.egomotion", "pod-a", 30.0)
        post.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
