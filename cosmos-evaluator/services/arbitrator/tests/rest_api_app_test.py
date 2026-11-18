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

"""End-to-end tests for the client-agnostic Arbitrator app (``rest_api_common.ArbitratorApp``).

Drives the async submit-and-poll lifecycle (``POST /process/async`` + ``GET /executions/{id}``) through FastAPI's
``TestClient`` with the lifespan (scheduler and job monitor) running. The health-probe clients are a patched
``httpx.AsyncClient`` mock, and the Database REST API is a stateful in-memory ``ApiClient`` mock that also stands in
for the checker fleet: anything the arbitrator writes to the job table comes back finished on the next poll. So the
whole admission -> enqueue -> claim -> finalize path runs in-process without a network or a database.
"""

from datetime import datetime
import time
from typing import Any
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import status
from fastapi.testclient import TestClient
import httpx

from database.api_client import ApiClient, CheckerExecutionWrite
from database.api_models import (
    ArbitratorExecutionRow,
    CheckerExecutionRow,
    CheckerExecutionStateRow,
    ClientRow,
)
from database.statuses import CHECKER_TERMINAL_STATUSES
from services.arbitrator.dependency_graph import DependencyGraph
from services.arbitrator.rest_api_common import ArbitratorApp, check_service_health
from services.arbitrator.settings import Settings
from services.arbitrator.waiting_room_queue import WaitingRoom

_CONFIG = {"av.objects_single_view": {"enabled": True, "deps": []}}
_ENDPOINTS = {"av.objects_single_view": "http://objects"}

# ``//utils:git_sha`` is not in this test's runfiles; stub it for the whole process. The patch is intentionally never
# stopped (mirrors the production behavior of a fixed commit sha per replica).
_git_sha_patcher = patch("services.utils.get_git_sha", return_value="test_sha")
_git_sha_patcher.start()


def _build_settings() -> Settings:
    """Build a real base ``Settings`` with YAML/runfiles resolution stubbed out for tests."""
    graph = DependencyGraph(_CONFIG)
    graph.validate()
    with (
        patch.object(Settings, "_resolve_checker_endpoints", return_value=dict(_ENDPOINTS)),
        patch.object(Settings, "_resolve_database_api_url", return_value=""),
        patch.object(Settings, "_load_dependency_graph", return_value=graph),
        patch("services.settings_base.SettingsBase.get_env_files", return_value=[]),
    ):
        return Settings(_env_file=[])


class _TestArbitratorApp(ArbitratorApp):
    """Base app with an injected mock DB client."""

    def __init__(self, settings: Settings, api_client: Any) -> None:
        super().__init__(settings=settings)
        self._injected_client = api_client

    def init_api_client(self) -> Any:
        return self._injected_client


def make_mock_api_client(request_id: int = 100, checker_result: dict[str, Any] | None = None) -> AsyncMock:
    """Build a minimally-stateful ``ApiClient`` mock so ``post_and_await`` can observe finalization.

    ``create_execution`` mints a fresh request_id; ``update_execution`` records the latest status/response/duration
    keyed by request_id and ``get_execution`` plays it back; checker rows are tracked similarly.

    ``get_checker_execution_states`` doubles as the checker fleet: every job the arbitrator has written to the table
    reads back claimed and finished, carrying ``checker_result`` as its payload. That is the only signal the
    arbitrator ever gets that a checker ran, so it is all a test needs to stand one up.
    """
    mock = AsyncMock(spec=ApiClient)
    mock.record_client.return_value = ClientRow(client_id=1, ip_address="127.0.0.1", name=None, description=None)
    mock.requeue_expired_checker_jobs.return_value = []

    execution_states: dict[int, dict[str, Any]] = {}
    checker_states: dict[int, dict[int, dict[str, Any]]] = {}
    next_checker_id = [99]
    next_request_id = [request_id]

    def _default_state(req_id: int) -> dict[str, Any]:
        return {
            "request_id": req_id,
            "status": "queued",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "duration_ms": 0,
            "version": "test",
            "commit_sha": "test_sha",
            "response_json": {},
        }

    async def _create_execution(**_: Any) -> int:
        new_id = next_request_id[0]
        next_request_id[0] += 1
        execution_states[new_id] = _default_state(new_id)
        checker_states[new_id] = {}
        return new_id

    async def _update_execution(**kwargs: Any) -> bool:
        req_id = kwargs.get("request_id")
        if req_id is None:
            return True
        state = execution_states.setdefault(req_id, _default_state(req_id))
        for key in ("status", "response_json", "duration_ms"):
            if key in kwargs:
                state[key] = kwargs[key]
        return True

    async def _get_execution(req_id: int) -> ArbitratorExecutionRow | None:
        state = execution_states.get(req_id)
        return ArbitratorExecutionRow.model_validate(state) if state else None

    async def _create_checker_execution(**kwargs: Any) -> int:
        execution_id = next_checker_id[0]
        next_checker_id[0] += 1
        req_id = kwargs.get("request_id", request_id)
        checker_states.setdefault(req_id, {})[execution_id] = {
            "execution_id": execution_id,
            "request_id": req_id,
            "checker_name": kwargs.get("checker_name"),
            "camera_name": kwargs.get("camera_name"),
            "status": kwargs.get("status", "queued"),
            "response_json": {},
        }
        return execution_id

    async def _update_checker_execution(**kwargs: Any) -> CheckerExecutionWrite:
        execution_id = kwargs.get("execution_id")
        status = kwargs.get("status")
        for rows in checker_states.values():
            if execution_id not in rows:
                continue
            # Same terminal guard the real endpoint applies, so a test cannot pass here on a write the server
            # would have refused.
            existing = rows[execution_id].get("status")
            if existing in CHECKER_TERMINAL_STATUSES and existing != status:
                return CheckerExecutionWrite(written=False, existing_status=existing)
            for key, value in kwargs.items():
                if key != "execution_id":
                    # Mirror ApiClient.update_checker_execution, which serializes datetimes to ISO strings; the
                    # real GET plays back JSON strings, so store strings here too (not raw datetime objects).
                    rows[execution_id][key] = value.isoformat() if isinstance(value, datetime) else value
            break
        return CheckerExecutionWrite(written=True, existing_status=None)

    async def _get_checker_executions_for_request(req_id: int) -> list[CheckerExecutionRow]:
        return [CheckerExecutionRow.model_validate(row) for row in checker_states.get(req_id, {}).values()]

    async def _get_checker_execution_states(execution_ids: list[int]) -> list[CheckerExecutionStateRow]:
        wanted = set(execution_ids)
        states = []
        for rows in checker_states.values():
            for execution_id, row in rows.items():
                if execution_id not in wanted:
                    continue
                row["status"] = "success"
                row["response_json"] = {"success": True, "data": checker_result or {}, "metadata": {}}
                states.append(
                    CheckerExecutionStateRow(
                        execution_id=execution_id,
                        status="success",
                        response_json=row["response_json"],
                        error_message=None,
                        claimed_by="checker-pod",
                        attempt=1,
                    )
                )
        return states

    mock.create_execution.side_effect = _create_execution
    mock.update_execution.side_effect = _update_execution
    mock.get_execution.side_effect = _get_execution
    mock.create_checker_execution.side_effect = _create_checker_execution
    mock.update_checker_execution.side_effect = _update_checker_execution
    mock.get_checker_executions_for_request.side_effect = _get_checker_executions_for_request
    mock.get_checker_execution_states.side_effect = _get_checker_execution_states
    return mock


def _health_response(*, healthy: bool = True) -> MagicMock:
    response = MagicMock()
    response.status_code = status.HTTP_200_OK
    response.json.return_value = {
        "data": {"status": "healthy" if healthy else "unhealthy", "version": "v1", "git_sha": "sha"}
    }
    return response


def _single_view_body() -> dict:
    return {
        "checks": ["av.objects_single_view"],
        "inputs": {
            "augmented_video_urls": {"camera_front_wide_120fov": "s3://b/aug.mp4"},
            "world_model_video_urls": {"camera_front_wide_120fov": "s3://b/wm.mp4"},
            "rds_hq_url": "s3://b/rds.zip",
        },
    }


def post_and_await(client: TestClient, body: dict, *, poll_interval: float = 0.01, timeout: float = 30.0) -> Any:
    """Submit to ``/process/async`` and poll ``/executions/{id}`` until terminal; return the final httpx response."""
    submit = client.post("/process/async", json=body)
    if submit.status_code != status.HTTP_202_ACCEPTED:
        return submit
    request_id = submit.json()["data"]["request_id"]
    deadline = time.monotonic() + timeout
    while True:
        snapshot = client.get(f"/executions/{request_id}")
        execution = (snapshot.json().get("data") or {}).get("execution") or {}
        if execution.get("is_terminal"):
            return snapshot
        if time.monotonic() >= deadline:
            raise AssertionError(f"Execution {request_id} did not terminate within {timeout:.1f}s")
        time.sleep(poll_interval)


class TestSubmitAndPoll(unittest.TestCase):
    """The async submit-and-poll lifecycle over the client-agnostic base app."""

    @patch.object(WaitingRoom, "JOB_POLL_JITTER", 0.0)
    @patch.object(WaitingRoom, "JOB_POLL_INTERVAL", 0.01)
    @patch("services.arbitrator.rest_api_common.httpx.AsyncClient")
    def test_happy_path_runs_single_view_to_success(self, mock_client_cls: MagicMock) -> None:
        mock_http = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_http
        mock_http.get.return_value = _health_response(healthy=True)

        api_client = make_mock_api_client(checker_result={"score": 0.95})
        app = _TestArbitratorApp(_build_settings(), api_client).create_app()
        with TestClient(app) as client:
            response = post_and_await(client, _single_view_body())

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        execution = response.json()["data"]["execution"]
        self.assertEqual(execution["status"], "success")
        results = execution["response_json"]["checker_results"]
        self.assertEqual(results["av.objects_single_view"]["camera_front_wide_120fov"]["score"], 0.95)
        api_client.create_execution.assert_called_once()
        api_client.create_checker_execution.assert_called_once()

    @patch("services.arbitrator.rest_api_common.httpx.AsyncClient")
    def test_no_healthy_checkers_returns_503(self, mock_client_cls: MagicMock) -> None:
        mock_http = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_http
        mock_http.get.return_value = _health_response(healthy=False)

        app = _TestArbitratorApp(_build_settings(), make_mock_api_client()).create_app()
        with TestClient(app) as client:
            submit = client.post("/process/async", json=_single_view_body())

        self.assertEqual(submit.status_code, status.HTTP_503_SERVICE_UNAVAILABLE)


class TestDiscoveryEndpoints(unittest.TestCase):
    """``/checkers`` and ``/config`` aggregate over the configured checker endpoints."""

    @patch("services.arbitrator.rest_api_common.httpx.AsyncClient")
    def test_checkers_lists_healthy(self, mock_client_cls: MagicMock) -> None:
        mock_http = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_http
        mock_http.get.return_value = _health_response(healthy=True)

        app = _TestArbitratorApp(_build_settings(), make_mock_api_client()).create_app()
        with TestClient(app) as client:
            response = client.get("/checkers")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json()["data"]["checkers"], ["av.objects_single_view"])

    @patch("services.arbitrator.rest_api_common.httpx.AsyncClient")
    def test_config_aggregates_default_config(self, mock_client_cls: MagicMock) -> None:
        mock_http = AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_http
        # ``_get_config`` gates on ``isinstance(result, httpx.Response)``, so the mock must carry that spec.
        config_response = MagicMock(spec=httpx.Response)
        config_response.status_code = status.HTTP_200_OK
        config_response.json.return_value = {"data": {"default_config": {"av.objects_single_view": {"fps": 30}}}}
        mock_http.get.return_value = config_response

        app = _TestArbitratorApp(_build_settings(), make_mock_api_client()).create_app()
        with TestClient(app) as client:
            response = client.get("/config")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.json()["data"]["config"]["av.objects_single_view"], {"fps": 30})


class TestCheckServiceHealth(unittest.IsolatedAsyncioTestCase):
    """``check_service_health`` maps probe outcomes onto a ``CheckerHealth`` verdict."""

    async def test_healthy_response(self) -> None:
        client = AsyncMock()
        client.get.return_value = _health_response(healthy=True)
        health = await check_service_health(client, "av.objects_single_view", "http://objects")
        self.assertTrue(health.is_healthy)
        self.assertEqual(health.version, "v1")

    async def test_unhealthy_status_in_body(self) -> None:
        client = AsyncMock()
        client.get.return_value = _health_response(healthy=False)
        health = await check_service_health(client, "av.objects_single_view", "http://objects")
        self.assertFalse(health.is_healthy)

    async def test_non_200_is_unhealthy(self) -> None:
        client = AsyncMock()
        not_found = MagicMock()
        not_found.status_code = status.HTTP_404_NOT_FOUND
        client.get.return_value = not_found
        health = await check_service_health(client, "av.objects_single_view", "http://objects")
        self.assertFalse(health.is_healthy)

    async def test_connection_error_is_unhealthy(self) -> None:
        client = AsyncMock()
        client.get.side_effect = RuntimeError("connection refused")
        health = await check_service_health(client, "av.objects_single_view", "http://objects")
        self.assertFalse(health.is_healthy)


if __name__ == "__main__":
    unittest.main()
