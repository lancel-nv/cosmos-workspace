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

"""Contract tests for the Database REST API's response shapes.

These routes hand back a ``JSONResponse`` built by ``JsonResponseFormatter``, and FastAPI skips response validation
whenever a handler returns a ``Response`` directly. The ``response_model`` declarations therefore document the
contract without enforcing it, and this module is the enforcement: it drives each endpoint with a representative
service return value and validates the body that actually comes back against the model the route declares.

The models live in ``database.api_models``, which ``ApiClient`` also parses responses with, so what is checked here is
that the two ends still agree. They use ``extra="allow"`` by deliberate design -- the arbitrator forwards whole rows,
so unknown columns have to survive a round trip -- which means this catches a declared field going missing or changing
type, not a new field appearing. ``TestValidationActuallyBites`` keeps that guarantee honest by showing a truncated
response failing.
"""

from datetime import UTC, datetime
from typing import Any
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

with patch("services.settings_base.SettingsBase.get_env_files", return_value=[]):
    from services.database.settings import get_settings

    get_settings.cache_clear()
    from database.api_models import ArbitratorExecutionRow
    from services.database.queries.orchestration import CheckerExecutionUpdate, LeaseRenewal
    from services.database.rest_api_common import app
    from services.framework.response_formatters.json_response_formatter import ErrorEnvelope

# Intentionally never stopped: this module-level patch persists for the entire test process.
_git_sha_patcher = patch("services.utils.get_git_sha", return_value="test_sha")
_git_sha_patcher.start()

_NOW = datetime(2026, 8, 21, 12, 0, 0, tzinfo=UTC).isoformat()

# One representative payload per projection, matching what the service hands the formatter. Nullable columns are
# populated rather than left null so that a field wrongly declared required is still exercised.
_ARBITRATOR_EXECUTION_ROW: dict[str, Any] = {
    "request_id": 1,
    "client_id": 2,
    "timestamp": _NOW,
    "status": "success",
    "request_json": {"video_uuid": "v1"},
    "response_json": {"checker_results": {}},
    "duration_ms": 1234,
    "metadata": {"source": "test"},
    "version": "1.0.0",
    "commit_sha": "abc123",
    "claimed_by": "replica-a",
    "lease_expires": _NOW,
}

_CHECKER_EXECUTION_ROW: dict[str, Any] = {
    "execution_id": 10,
    "request_id": 1,
    "checker_name": "objects_single_view",
    "camera_name": "camera_front_wide_120fov",
    "status": "success",
    "start_time": _NOW,
    "end_time": _NOW,
    "duration_ms": 900,
    "error_message": None,
    "version": "1.0.0",
    "commit_sha": "abc123",
    "output_metadata": {"results_url": "s3://bucket/key"},
    "response_json": {"score": 1.0},
}

_CHECKER_EXECUTION_STATE_ROW: dict[str, Any] = {
    **_CHECKER_EXECUTION_ROW,
    "status": "in_progress",
    "claimed_by": "pod-a",
    "lease_expires": _NOW,
    "attempt": 1,
    "cancel_requested": False,
}

_CLAIMED_JOB_ROW: dict[str, Any] = {
    "execution_id": 10,
    "claimed_by": "pod-a",
    "attempt": 1,
    "lease_expires": _NOW,
    "cancel_requested": False,
    "request_json": {"video_uuid": "v1"},
    "checker_name": "objects_single_view",
    "camera_name": "camera_front_wide_120fov",
    "request_id": 1,
}

_CLIENT_ROW: dict[str, Any] = {
    "client_id": 3,
    "ip_address": "10.0.0.1",
    "name": "batch-tool",
    "description": "nightly run",
}

_ADOPTED_EXECUTION_ROW: dict[str, Any] = {
    "request_id": 4,
    "client_id": 2,
    "status": "in_progress",
    "request_json": {"video_uuid": "v1"},
    "claimed_by": "replica-b",
    "lease_expires": _NOW,
    "version": "1.0.0",
    "commit_sha": "abc123",
}

_ADMIN_HEADERS = {"X-API-Key": "test-admin-key"}
_WORKER_HEADERS = {"X-API-Key": "test-worker-key"}
_READ_HEADERS = {"X-API-Key": "test-read-key"}


def _api_routes() -> list[APIRoute]:
    """Every ``APIRoute`` reachable from the app, descending into included routers.

    As of FastAPI 0.141 ``include_router`` no longer flattens a router's routes onto ``app.routes``; the app holds an
    opaque ``_IncludedRouter`` that keeps the real routes on its ``original_router``. Both shapes are followed here so
    this keeps working either way, and the callers assert a floor on the count -- without that, a future change to how
    routers are mounted would silently reduce this to ``/health`` and every assertion built on it would still pass.
    """
    found: list[APIRoute] = []
    pending = list(app.routes)
    seen: set[int] = set()
    while pending:
        route = pending.pop()
        if id(route) in seen:
            continue
        seen.add(id(route))
        if isinstance(route, APIRoute):
            found.append(route)
        included = getattr(route, "original_router", None)
        nested = getattr(route, "routes", None) or getattr(included, "routes", None)
        if nested:
            pending.extend(nested)
    return found


def _mock_service() -> MagicMock:
    """Attach a service to ``app.state`` whose methods return the representative payloads above."""
    service = MagicMock()
    service.get_execution = AsyncMock(return_value=dict(_ARBITRATOR_EXECUTION_ROW))
    service.get_checker_executions_for_request = AsyncMock(return_value=[dict(_CHECKER_EXECUTION_ROW)])
    service.get_checker_execution_states_for_request = AsyncMock(return_value=[dict(_CHECKER_EXECUTION_STATE_ROW)])
    service.get_checker_execution_states = AsyncMock(return_value=[dict(_CHECKER_EXECUTION_STATE_ROW)])
    service.mark_checker_executions_in_progress = AsyncMock(
        return_value=[{"execution_id": 10, "existing_status": "success"}]
    )
    service.record_client = AsyncMock(return_value=dict(_CLIENT_ROW))
    service.create_execution = AsyncMock(return_value=1)
    service.update_execution = AsyncMock(return_value=True)
    service.heartbeat_executions = AsyncMock(
        return_value=LeaseRenewal(renewed=[1], lost=[{"request_id": 2, "status": "success", "claimed_by": "other"}])
    )
    service.adopt_executions = AsyncMock(return_value=[dict(_ADOPTED_EXECUTION_ROW)])
    service.create_checker_execution = AsyncMock(return_value=10)
    service.update_checker_execution = AsyncMock(
        return_value=CheckerExecutionUpdate(found=True, updated=True, existing_status="in_progress")
    )
    service.claim_next_checker_executions = AsyncMock(return_value=[dict(_CLAIMED_JOB_ROW)])
    service.requeue_expired_checker_executions = AsyncMock(
        return_value=[{"execution_id": 10, "checker_name": "vlm", "attempt": 2}]
    )
    service.heartbeat_checker_execution = AsyncMock(
        return_value={"execution_id": 10, "lease_expires": _NOW, "cancel_requested": False, "attempt": 1}
    )
    service.complete_checker_execution = AsyncMock(return_value=True)
    service.request_checker_execution_cancel = AsyncMock(return_value=True)
    app.state.service = service
    return service


class TestEveryRouteDeclaresAResponseModel(unittest.TestCase):
    """A new endpoint must not be able to ship without declaring what it returns.

    This is what keeps the contract complete. Without it the coverage below extends only to the endpoints somebody
    remembered, and the next endpoint added is undeclared and untested by construction -- which is how every route on
    this service came to be unpinned in the first place.
    """

    def test_no_route_is_missing_a_response_model(self) -> None:
        routes = _api_routes()
        # One path from each of the three ways a route reaches this app: declared on the app, mounted under the
        # read-key guard, and mounted as a self-guarded router. If the walk stops reaching any of them the list below
        # goes quiet and this test starts passing for the wrong reason.
        discovered = {route.path for route in routes}
        self.assertLessEqual(
            {"/health", "/clients", "/checker-executions/claim-next"},
            discovered,
            f"Route discovery is not reaching the whole mounted surface; found only {sorted(discovered)}",
        )
        undeclared = sorted(
            f"{','.join(sorted(route.methods or set()))} {route.path}"
            for route in routes
            if route.response_model is None
        )
        self.assertEqual(
            undeclared,
            [],
            "Every route must declare response_model=SuccessEnvelope[...] naming a model from database.api_models. "
            "Undeclared: " + ", ".join(undeclared),
        )

    def test_openapi_schema_builds(self) -> None:
        """The generic envelope has to survive schema generation, or ``/docs`` and ``/openapi.json`` break."""
        schemas = app.openapi()["components"]["schemas"]
        self.assertIn("SuccessEnvelope_ArbitratorExecutionRow_", schemas)
        self.assertIn("ErrorEnvelope", schemas)


class _ContractTestCase(unittest.TestCase):
    """Base for the per-endpoint contract checks."""

    def setUp(self) -> None:
        self.service = _mock_service()
        self.client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def assert_matches_declared_model(self, method: str, path: str, response_json: dict[str, Any]) -> None:
        """Validate a response body against the model the route itself declares.

        The model is looked up from the app rather than named in the test, so the assertion cannot drift from the
        declaration: change a route's ``response_model`` and this validates against the new one.
        """
        model = self._declared_model(method, path)
        model.model_validate(response_json)

    def _declared_model(self, method: str, path: str) -> type[BaseModel]:
        for route in _api_routes():
            if route.path == path and method in (route.methods or set()):
                self.assertIsNotNone(route.response_model, f"{method} {path} declares no response_model")
                return route.response_model
        routes = _api_routes()
        self.assertGreater(len(routes), 1, "Route discovery is broken: expected the whole mounted surface")
        registered = sorted(f"{','.join(sorted(r.methods or set()))} {r.path}" for r in routes)
        self.fail(f"No route registered for {method} {path}. Registered: " + "; ".join(registered))


class TestArbitratorResponses(_ContractTestCase):
    """Every arbitrator endpoint's real response body validates against its declared model."""

    def test_get_execution(self) -> None:
        response = self.client.get("/arbitrator-executions/1", headers=_READ_HEADERS)
        self.assertEqual(response.status_code, 200)
        self.assert_matches_declared_model("GET", "/arbitrator-executions/{request_id}", response.json())

    def test_get_checker_executions_for_request(self) -> None:
        response = self.client.get("/arbitrator-executions/1/checker-executions", headers=_READ_HEADERS)
        self.assertEqual(response.status_code, 200)
        self.assert_matches_declared_model(
            "GET", "/arbitrator-executions/{request_id}/checker-executions", response.json()
        )

    def test_get_checker_states_for_request(self) -> None:
        response = self.client.get("/arbitrator-executions/1/checker-states", headers=_READ_HEADERS)
        self.assertEqual(response.status_code, 200)
        self.assert_matches_declared_model("GET", "/arbitrator-executions/{request_id}/checker-states", response.json())

    def test_record_client(self) -> None:
        response = self.client.post("/clients", json={"ip_address": "10.0.0.1"}, headers=_ADMIN_HEADERS)
        self.assertEqual(response.status_code, 201)
        self.assert_matches_declared_model("POST", "/clients", response.json())

    def test_create_execution(self) -> None:
        response = self.client.post(
            "/arbitrator-executions",
            json={"client_id": 1, "request_json": {}, "version": "1.0.0", "commit_sha": "abc"},
            headers=_ADMIN_HEADERS,
        )
        self.assertEqual(response.status_code, 201)
        self.assert_matches_declared_model("POST", "/arbitrator-executions", response.json())

    def test_update_execution(self) -> None:
        response = self.client.put("/arbitrator-executions/1", json={"status": "success"}, headers=_ADMIN_HEADERS)
        self.assertEqual(response.status_code, 200)
        self.assert_matches_declared_model("PUT", "/arbitrator-executions/{request_id}", response.json())

    def test_heartbeat_executions(self) -> None:
        response = self.client.post(
            "/arbitrator-executions/heartbeat",
            json={"claimed_by": "replica-a", "lease_seconds": 30, "request_ids": [1, 2]},
            headers=_ADMIN_HEADERS,
        )
        self.assertEqual(response.status_code, 200)
        self.assert_matches_declared_model("POST", "/arbitrator-executions/heartbeat", response.json())

    def test_adopt_executions(self) -> None:
        response = self.client.post(
            "/arbitrator-executions/adopt",
            json={"claimed_by": "replica-b", "lease_seconds": 30},
            headers=_ADMIN_HEADERS,
        )
        self.assertEqual(response.status_code, 200)
        self.assert_matches_declared_model("POST", "/arbitrator-executions/adopt", response.json())

    def test_create_checker_execution(self) -> None:
        response = self.client.post(
            "/checker-executions",
            json={"request_id": 1, "checker_name": "vlm", "request_json": {}},
            headers=_ADMIN_HEADERS,
        )
        self.assertEqual(response.status_code, 201)
        self.assert_matches_declared_model("POST", "/checker-executions", response.json())

    def test_update_checker_execution(self) -> None:
        response = self.client.put("/checker-executions/10", json={"status": "success"}, headers=_ADMIN_HEADERS)
        self.assertEqual(response.status_code, 200)
        self.assert_matches_declared_model("PUT", "/checker-executions/{execution_id}", response.json())

    def test_update_checker_execution_without_status(self) -> None:
        """The arbitrator's ``start_time``-only write leaves ``status`` null, which the model has to allow."""
        response = self.client.put("/checker-executions/10", json={"start_time": _NOW}, headers=_ADMIN_HEADERS)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["data"]["status"])
        self.assert_matches_declared_model("PUT", "/checker-executions/{execution_id}", response.json())


class TestCheckerWorkResponses(_ContractTestCase):
    """Every checker-work endpoint's real response body validates against its declared model."""

    def test_claim_next(self) -> None:
        response = self.client.post(
            "/checker-executions/claim-next",
            json={"checker_name": "vlm", "claimed_by": "pod-a", "lease_seconds": 30, "wait_seconds": 0},
            headers=_WORKER_HEADERS,
        )
        self.assertEqual(response.status_code, 200)
        self.assert_matches_declared_model("POST", "/checker-executions/claim-next", response.json())

    def test_claim_next_empty_queue(self) -> None:
        """An empty queue is an ordinary answer and has to satisfy the same model."""
        self.service.claim_next_checker_executions = AsyncMock(return_value=[])
        response = self.client.post(
            "/checker-executions/claim-next",
            json={"checker_name": "vlm", "claimed_by": "pod-a", "lease_seconds": 30, "wait_seconds": 0},
            headers=_WORKER_HEADERS,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["jobs"], [])
        self.assert_matches_declared_model("POST", "/checker-executions/claim-next", response.json())

    def test_reap(self) -> None:
        response = self.client.post("/checker-executions/reap", json={}, headers=_ADMIN_HEADERS)
        self.assertEqual(response.status_code, 200)
        self.assert_matches_declared_model("POST", "/checker-executions/reap", response.json())

    def test_heartbeat(self) -> None:
        response = self.client.post(
            "/checker-executions/10/heartbeat",
            json={"claimed_by": "pod-a", "lease_seconds": 30},
            headers=_WORKER_HEADERS,
        )
        self.assertEqual(response.status_code, 200)
        self.assert_matches_declared_model("POST", "/checker-executions/{execution_id}/heartbeat", response.json())

    def test_complete(self) -> None:
        response = self.client.post(
            "/checker-executions/10/complete",
            json={"claimed_by": "pod-a", "status": "success"},
            headers=_WORKER_HEADERS,
        )
        self.assertEqual(response.status_code, 200)
        self.assert_matches_declared_model("POST", "/checker-executions/{execution_id}/complete", response.json())

    def test_cancel(self) -> None:
        response = self.client.post("/checker-executions/10/cancel", headers=_ADMIN_HEADERS)
        self.assertEqual(response.status_code, 200)
        self.assert_matches_declared_model("POST", "/checker-executions/{execution_id}/cancel", response.json())

    def test_states(self) -> None:
        response = self.client.post("/checker-executions/states", json={"execution_ids": [10]}, headers=_READ_HEADERS)
        self.assertEqual(response.status_code, 200)
        self.assert_matches_declared_model("POST", "/checker-executions/states", response.json())

    def test_mark_in_progress(self) -> None:
        response = self.client.post(
            "/checker-executions/in-progress",
            json={"execution_ids": [10], "start_time": "2026-09-03T12:00:00+00:00"},
            headers=_ADMIN_HEADERS,
        )
        self.assertEqual(response.status_code, 200)
        self.assert_matches_declared_model("POST", "/checker-executions/in-progress", response.json())


class TestHealthResponse(_ContractTestCase):
    def test_health(self) -> None:
        response = self.client.get("/health", headers=_READ_HEADERS)
        self.assertEqual(response.status_code, 200)
        self.assert_matches_declared_model("GET", "/health", response.json())


class TestErrorEnvelope(_ContractTestCase):
    """Failures share one shape, and the declared error model has to accept it."""

    def test_not_found_matches_error_envelope(self) -> None:
        self.service.get_execution = AsyncMock(return_value=None)
        response = self.client.get("/arbitrator-executions/999", headers=_READ_HEADERS)
        self.assertEqual(response.status_code, 404)
        ErrorEnvelope.model_validate(response.json())

    def test_validation_failure_matches_error_envelope(self) -> None:
        """A rejected request body is FastAPI's own 422, not the formatter's envelope, and is asserted as such.

        Worth pinning: a caller handling errors uniformly has to know this one endpoint of the contract differs.
        """
        response = self.client.post(
            "/arbitrator-executions/heartbeat",
            json={"claimed_by": "replica-a", "lease_seconds": -1, "request_ids": [1]},
            headers=_ADMIN_HEADERS,
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("detail", response.json())


class TestValidationActuallyBites(unittest.TestCase):
    """Guard against the contract checks above passing vacuously.

    The payload models allow unknown fields, so it is worth demonstrating what they still reject. If this stops
    failing, every assertion above has quietly become a no-op.
    """

    def test_missing_required_field_is_rejected(self) -> None:
        truncated = {key: value for key, value in _ARBITRATOR_EXECUTION_ROW.items() if key != "request_id"}
        with self.assertRaises(ValidationError):
            ArbitratorExecutionRow.model_validate(truncated)

    def test_wrong_type_is_rejected(self) -> None:
        mistyped = {**_ARBITRATOR_EXECUTION_ROW, "request_id": "not-an-int"}
        with self.assertRaises(ValidationError):
            ArbitratorExecutionRow.model_validate(mistyped)

    def test_unknown_field_is_kept_not_rejected(self) -> None:
        """``extra="allow"`` is load-bearing: the arbitrator forwards whole rows, so a new column must survive."""
        widened = {**_ARBITRATOR_EXECUTION_ROW, "column_added_later": "value"}
        parsed = ArbitratorExecutionRow.model_validate(widened)
        self.assertEqual(parsed.model_dump()["column_added_later"], "value")


if __name__ == "__main__":
    unittest.main()
