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

"""Unit tests for the Database REST API app.

Exercises the base ``DatabaseRestApi`` app built by ``services.database.rest_api_common`` to verify the container
serves /health and the core endpoints, and enforces RBAC.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from services.utils import get_contents_from_runfile

with patch("services.settings_base.SettingsBase.get_env_files", return_value=[]):
    from services.database.settings import get_settings

    get_settings.cache_clear()
    from services.database.app import DatabaseRestApi
    from services.database.queries.orchestration import LeaseRenewal
    from services.database.rest_api import main
    from services.database.rest_api_common import app
    from services.database.service import PoolUnavailableError

# Intentionally never stopped: this module-level patch persists for the entire test process.
_git_sha_patcher = patch("services.utils.get_git_sha", return_value="test_sha")
_git_sha_patcher.start()


def _mock_service() -> MagicMock:
    """Create a mock DatabaseService and attach it to app.state."""
    service = MagicMock()
    service.get_execution = AsyncMock()
    service.get_checker_executions_for_request = AsyncMock()
    service.get_checker_execution_states_for_request = AsyncMock()
    service.record_client = AsyncMock()
    service.create_execution = AsyncMock()
    service.update_execution = AsyncMock()
    service.heartbeat_executions = AsyncMock()
    service.adopt_executions = AsyncMock()
    service.create_checker_execution = AsyncMock()
    service.update_checker_execution = AsyncMock()
    service.claim_next_checker_executions = AsyncMock(return_value=[])
    service.requeue_expired_checker_executions = AsyncMock(return_value=[])
    service.heartbeat_checker_execution = AsyncMock()
    service.complete_checker_execution = AsyncMock()
    service.request_checker_execution_cancel = AsyncMock()
    service.get_checker_execution_states = AsyncMock()
    service.mark_checker_executions_in_progress = AsyncMock(return_value=[])
    app.state.service = service
    return service


class TestLifespan(unittest.TestCase):
    """Exercise FastAPI lifespan startup/shutdown on the base app builder."""

    def test_lifespan_init_and_close_called(self) -> None:
        mock_svc = MagicMock()
        mock_svc.init = AsyncMock()
        mock_svc.close = AsyncMock()
        with (
            patch.object(DatabaseRestApi, "service_class", MagicMock(return_value=mock_svc)),
            TestClient(app) as client,
        ):
            self.assertEqual(client.get("/health").status_code, 200)
        mock_svc.init.assert_awaited_once()
        mock_svc.close.assert_awaited_once()

    def test_lifespan_init_failure_still_attaches_service(self) -> None:
        mock_svc = MagicMock()
        mock_svc.init = AsyncMock(side_effect=ConnectionError("db down"))
        mock_svc.close = AsyncMock()
        with (
            patch.object(DatabaseRestApi, "service_class", MagicMock(return_value=mock_svc)),
            TestClient(app) as client,
        ):
            self.assertEqual(client.get("/health").status_code, 200)
        mock_svc.init.assert_awaited_once()
        self.assertIs(app.state.service, mock_svc)


class TestHealthEndpoint(unittest.TestCase):
    """Test cases for /health endpoint."""

    def setUp(self) -> None:
        _mock_service()
        self.client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def test_health_returns_200(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["data"]["status"], "healthy")
        self.assertEqual(data["data"]["service"], "Database REST API")
        self.assertEqual(data["data"]["version"], get_contents_from_runfile("services/database/version.txt"))
        self.assertEqual(data["data"]["git_sha"], "test_sha")


class TestArbitratorEndpoints(unittest.TestCase):
    """Test cases for the arbitrator endpoints served by the standalone app."""

    def setUp(self) -> None:
        self.service = _mock_service()
        self.client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def test_get_execution_returns_200_when_found(self) -> None:
        self.service.get_execution.return_value = {"request_id": 1, "status": "success"}
        response = self.client.get("/arbitrator-executions/1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["request_id"], 1)

    def test_get_execution_returns_404_when_missing(self) -> None:
        self.service.get_execution.return_value = None
        self.assertEqual(self.client.get("/arbitrator-executions/999").status_code, 404)

    def test_create_execution_returns_201(self) -> None:
        self.service.create_execution.return_value = 42
        response = self.client.post(
            "/arbitrator-executions",
            json={"client_id": 1, "request_json": {}, "version": "1.0", "commit_sha": "abc"},
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["data"]["request_id"], 42)

    def test_create_execution_forwards_the_claim(self) -> None:
        self.service.create_execution.return_value = 42

        self.client.post(
            "/arbitrator-executions",
            json={
                "client_id": 1,
                "request_json": {},
                "version": "1.0",
                "commit_sha": "abc",
                "claimed_by": "replica-a",
                "lease_seconds": 90,
            },
        )

        kwargs = self.service.create_execution.call_args.kwargs
        self.assertEqual((kwargs["claimed_by"], kwargs["lease_seconds"]), ("replica-a", 90))

    def test_update_execution_forwards_the_fencing_token(self) -> None:
        self.service.update_execution.return_value = True

        response = self.client.put("/arbitrator-executions/42", json={"status": "success", "claimed_by": "replica-a"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.service.update_execution.call_args.kwargs["claimed_by"], "replica-a")

    def test_heartbeat_renews_the_batch_it_was_given(self) -> None:
        self.service.heartbeat_executions.return_value = LeaseRenewal(renewed=[42, 43], lost=[])

        response = self.client.post(
            "/arbitrator-executions/heartbeat",
            json={"request_ids": [42, 43], "claimed_by": "replica-a", "lease_seconds": 90},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["renewed"], [42, 43])
        self.assertEqual(self.service.heartbeat_executions.call_args.kwargs["request_ids"], [42, 43])

    def test_a_lost_claim_is_reported_with_its_reason_rather_than_a_404(self) -> None:
        """A 404 would say only "gone", leaving the caller to guess between adoption and an already-finished row."""
        self.service.heartbeat_executions.return_value = LeaseRenewal(
            renewed=[], lost=[{"request_id": 42, "status": "in_progress", "claimed_by": "replica-b"}]
        )

        response = self.client.post(
            "/arbitrator-executions/heartbeat",
            json={"request_ids": [42], "claimed_by": "replica-a", "lease_seconds": 90},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["lost"][0]["claimed_by"], "replica-b")

    def test_heartbeat_returns_503_when_the_pool_is_down(self) -> None:
        """Renewal lost to an outage must not read as adoption, or a live replica abandons a request it still owns."""
        self.service.heartbeat_executions.side_effect = PoolUnavailableError("pool down")

        response = self.client.post(
            "/arbitrator-executions/heartbeat",
            json={"request_ids": [42], "claimed_by": "replica-a", "lease_seconds": 90},
        )

        self.assertEqual(response.status_code, 503)

    def test_adopt_returns_the_claimed_rows(self) -> None:
        self.service.adopt_executions.return_value = [{"request_id": 7, "client_id": 1, "status": "in_progress"}]

        response = self.client.post(
            "/arbitrator-executions/adopt", json={"claimed_by": "replica-b", "lease_seconds": 90, "limit": 5}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["executions"][0]["request_id"], 7)

    def test_adopt_is_not_shadowed_by_the_by_id_read(self) -> None:
        """``/adopt`` sits where a ``{request_id}`` would; a route collision would send it to the wrong handler."""
        self.service.adopt_executions.return_value = []

        self.client.post("/arbitrator-executions/adopt", json={"claimed_by": "replica-b", "lease_seconds": 90})

        self.service.adopt_executions.assert_called_once()

    def test_checker_states_for_a_request_are_served(self) -> None:
        self.service.get_checker_execution_states_for_request.return_value = [
            {"execution_id": 3, "claimed_by": "pod-a", "status": "in_progress"}
        ]

        response = self.client.get("/arbitrator-executions/7/checker-states")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["claimed_by"], "pod-a")

    def test_record_client_returns_201(self) -> None:
        self.service.record_client.return_value = {"client_id": 1, "ip_address": "10.0.0.1", "name": None}
        response = self.client.post("/clients", json={"ip_address": "10.0.0.1"})
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["data"]["client_id"], 1)


class TestCheckerWorkEndpoints(unittest.TestCase):
    """Test cases for the checker job-coordination endpoints."""

    def setUp(self) -> None:
        self.service = _mock_service()
        self.client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def test_claim_next_rejects_an_out_of_range_lease(self) -> None:
        response = self.client.post(
            "/checker-executions/claim-next",
            json={"checker_name": "av.egomotion", "claimed_by": "pod-a", "lease_seconds": 0.5},
        )
        self.assertEqual(response.status_code, 422)

    def test_claim_next_hands_over_the_waiting_jobs(self) -> None:
        self.service.claim_next_checker_executions.return_value = [
            {"execution_id": 7, "claimed_by": "pod-a", "attempt": 1, "checker_name": "av.egomotion"}
        ]
        response = self.client.post(
            "/checker-executions/claim-next",
            json={"checker_name": "av.egomotion", "claimed_by": "pod-a", "lease_seconds": 30, "limit": 2},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["jobs"][0]["execution_id"], 7)
        self.assertEqual(self.service.claim_next_checker_executions.await_args.kwargs["limit"], 2)
        kwargs = self.service.claim_next_checker_executions.await_args.kwargs
        self.assertEqual((kwargs["version"], kwargs["commit_sha"]), (None, None), "an older pod reports no build")

    def test_claim_next_forwards_the_claimer_s_build(self) -> None:
        self.service.claim_next_checker_executions.return_value = []
        response = self.client.post(
            "/checker-executions/claim-next",
            json={
                "checker_name": "av.egomotion",
                "claimed_by": "pod-a",
                "lease_seconds": 30,
                "wait_seconds": 0,
                "version": "1.44.0",
                "commit_sha": "a" * 40,
            },
        )
        self.assertEqual(response.status_code, 200)
        kwargs = self.service.claim_next_checker_executions.await_args.kwargs
        self.assertEqual((kwargs["version"], kwargs["commit_sha"]), ("1.44.0", "a" * 40))

    def test_claim_next_rejects_a_commit_sha_the_column_cannot_hold(self) -> None:
        # checker_executions.commit_sha is VARCHAR(64); a longer value would fail the UPDATE and lose the claim.
        response = self.client.post(
            "/checker-executions/claim-next",
            json={"checker_name": "av.egomotion", "claimed_by": "pod-a", "lease_seconds": 30, "commit_sha": "a" * 65},
        )
        self.assertEqual(response.status_code, 422)

    def test_an_empty_queue_answers_200_with_no_jobs(self) -> None:
        # The caller's normal steady state on an idle fleet. Reporting it as an error would make an idle system look
        # broken and would push callers into treating a real failure as routine.
        self.service.claim_next_checker_executions.return_value = []
        response = self.client.post(
            "/checker-executions/claim-next",
            json={"checker_name": "av.vlm", "claimed_by": "pod-a", "lease_seconds": 30, "wait_seconds": 0},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["jobs"], [])

    def test_the_wait_rechecks_until_work_shows_up(self) -> None:
        self.service.claim_next_checker_executions.side_effect = [[], [], [{"execution_id": 9, "claimed_by": "pod-a"}]]
        with patch("services.database.routers.checker_work.asyncio.sleep", AsyncMock()) as sleep:
            response = self.client.post(
                "/checker-executions/claim-next",
                json={"checker_name": "av.vlm", "claimed_by": "pod-a", "lease_seconds": 30, "wait_seconds": 20},
            )
        self.assertEqual(response.json()["data"]["jobs"][0]["execution_id"], 9)
        self.assertEqual(self.service.claim_next_checker_executions.await_count, 3)
        self.assertEqual(sleep.await_count, 2, "each empty answer must cost a sleep, not a spin")

    def test_the_wait_never_sleeps_past_its_own_deadline(self) -> None:
        # Overshooting would hold the request open past the gateway's patience, and the caller would see a dropped
        # connection instead of an empty answer.
        self.service.claim_next_checker_executions.return_value = []
        with patch("services.database.routers.checker_work.asyncio.sleep", AsyncMock()) as sleep:
            self.client.post(
                "/checker-executions/claim-next",
                json={"checker_name": "av.vlm", "claimed_by": "pod-a", "lease_seconds": 30, "wait_seconds": 0.2},
            )
        for call in sleep.await_args_list:
            self.assertLessEqual(call.args[0], 0.2)

    def test_claim_next_returns_503_when_the_pool_is_down(self) -> None:
        self.service.claim_next_checker_executions.side_effect = PoolUnavailableError("pool down")
        response = self.client.post(
            "/checker-executions/claim-next",
            json={"checker_name": "av.vlm", "claimed_by": "pod-a", "lease_seconds": 30},
        )
        self.assertEqual(response.status_code, 503)

    def test_claim_next_caps_the_batch_size(self) -> None:
        response = self.client.post(
            "/checker-executions/claim-next",
            json={"checker_name": "av.vlm", "claimed_by": "pod-a", "lease_seconds": 30, "limit": 1000},
        )
        self.assertEqual(response.status_code, 422)

    def test_claim_next_caps_how_long_it_may_be_held_open(self) -> None:
        response = self.client.post(
            "/checker-executions/claim-next",
            json={"checker_name": "av.vlm", "claimed_by": "pod-a", "lease_seconds": 30, "wait_seconds": 600},
        )
        self.assertEqual(response.status_code, 422)

    def test_the_reaper_reports_what_it_requeued(self) -> None:
        self.service.requeue_expired_checker_executions.return_value = [{"execution_id": 7, "attempt": 2}]
        response = self.client.post("/checker-executions/reap", json={"limit": 50})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["requeued"][0]["execution_id"], 7)
        self.assertEqual(self.service.requeue_expired_checker_executions.await_args.kwargs["limit"], 50)

    def test_the_reaper_is_callable_without_a_body(self) -> None:
        response = self.client.post("/checker-executions/reap", json={})
        self.assertEqual(response.status_code, 200)

    def test_heartbeat_returns_the_cancellation_answer(self) -> None:
        self.service.heartbeat_checker_execution.return_value = {"execution_id": 7, "cancel_requested": True}
        response = self.client.post(
            "/checker-executions/7/heartbeat",
            json={"claimed_by": "pod-a", "lease_seconds": 30},
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["data"]["cancel_requested"])

    def test_heartbeat_returns_409_when_superseded(self) -> None:
        self.service.heartbeat_checker_execution.return_value = None
        response = self.client.post(
            "/checker-executions/7/heartbeat", json={"claimed_by": "pod-a", "lease_seconds": 30}
        )
        self.assertEqual(response.status_code, 409)

    def test_complete_returns_200(self) -> None:
        self.service.complete_checker_execution.return_value = True
        response = self.client.post(
            "/checker-executions/7/complete",
            json={"claimed_by": "pod-a", "status": "success", "response_json": {"score": 1}, "duration_ms": 900},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["status"], "success")

    def test_complete_returns_409_when_superseded(self) -> None:
        self.service.complete_checker_execution.return_value = False
        response = self.client.post("/checker-executions/7/complete", json={"claimed_by": "pod-a", "status": "success"})
        self.assertEqual(response.status_code, 409)

    def test_complete_rejects_a_non_terminal_status(self) -> None:
        # An owner reports how its own attempt ended; it has no business moving a job back to a pending state.
        response = self.client.post("/checker-executions/7/complete", json={"claimed_by": "pod-a", "status": "queued"})
        self.assertEqual(response.status_code, 422)

    def test_cancel_returns_200(self) -> None:
        self.service.request_checker_execution_cancel.return_value = True
        response = self.client.post("/checker-executions/7/cancel")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["data"]["cancel_requested"])

    def test_cancel_returns_404_for_an_already_terminal_job(self) -> None:
        self.service.request_checker_execution_cancel.return_value = False
        self.assertEqual(self.client.post("/checker-executions/7/cancel").status_code, 404)

    def test_states_returns_the_batch(self) -> None:
        self.service.get_checker_execution_states.return_value = [{"execution_id": 7, "status": "in_progress"}]
        response = self.client.post("/checker-executions/states", json={"execution_ids": [7]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["execution_id"], 7)
        self.service.get_checker_execution_states.assert_awaited_once_with([7])

    def test_the_batched_claim_write_reports_the_refusals(self) -> None:
        self.service.mark_checker_executions_in_progress.return_value = [
            {"execution_id": 7, "existing_status": "success"}
        ]
        response = self.client.post(
            "/checker-executions/in-progress",
            json={"execution_ids": [7, 8], "start_time": "2026-09-03T12:00:00+00:00"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["refused"], [{"execution_id": 7, "existing_status": "success"}])

    def test_the_batched_claim_write_answers_200_when_nothing_was_refused(self) -> None:
        """The healthy case, and not a 404: no refusals is the answer, not the absence of one.

        Deliberately unlike the single-row PUT's 409. A batch has no one status code to carry, and a refusal here is
        an expected outcome rather than an error -- a job claimed and finished inside one polling interval.
        """
        self.service.mark_checker_executions_in_progress.return_value = []
        response = self.client.post(
            "/checker-executions/in-progress",
            json={"execution_ids": [7, 8], "start_time": "2026-09-03T12:00:00+00:00"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["refused"], [])

    def test_the_batched_claim_write_returns_503_when_the_pool_is_down(self) -> None:
        self.service.mark_checker_executions_in_progress.side_effect = PoolUnavailableError("pool down")
        response = self.client.post(
            "/checker-executions/in-progress",
            json={"execution_ids": [7], "start_time": "2026-09-03T12:00:00+00:00"},
        )
        self.assertEqual(response.status_code, 503)

    def test_the_batched_claim_write_requires_a_start_time(self) -> None:
        """Without one the column would be refined to whenever the write happened to land, not when it was observed."""
        response = self.client.post("/checker-executions/in-progress", json={"execution_ids": [7]})
        self.assertEqual(response.status_code, 422)


# =========================================================================
# RBAC (API key authentication)
# =========================================================================

READ_KEY = "test-read-key-123"
ADMIN_KEY = "test-admin-key-456"
WORKER_KEY = "test-worker-key-789"


class TestRbac(unittest.TestCase):
    """When keys are configured, endpoints require the appropriate key."""

    def setUp(self) -> None:
        _mock_service()
        self._orig_settings = app.state.settings
        app.state.settings = MagicMock(read_key=READ_KEY, admin_key=ADMIN_KEY, worker_keys=[WORKER_KEY])
        self.client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def tearDown(self) -> None:
        app.state.settings = self._orig_settings

    def test_health_rejects_no_key(self) -> None:
        self.assertEqual(self.client.get("/health").status_code, 401)

    def test_health_accepts_read_key(self) -> None:
        self.assertEqual(self.client.get("/health", headers={"X-API-Key": READ_KEY}).status_code, 200)

    def test_write_endpoint_rejects_read_key(self) -> None:
        response = self.client.post("/clients", json={"ip_address": "10.0.0.1"}, headers={"X-API-Key": READ_KEY})
        self.assertEqual(response.status_code, 403)

    def test_write_endpoint_accepts_admin_key(self) -> None:
        self.service = _mock_service()
        self.service.record_client.return_value = {"client_id": 1, "ip_address": "10.0.0.1", "name": None}
        app.state.settings = MagicMock(read_key=READ_KEY, admin_key=ADMIN_KEY)
        response = self.client.post("/clients", json={"ip_address": "10.0.0.1"}, headers={"X-API-Key": ADMIN_KEY})
        self.assertEqual(response.status_code, 201)

    def test_openapi_accessible_without_key(self) -> None:
        self.assertEqual(self.client.get("/openapi.json").status_code, 200)

    def test_claim_next_accepts_a_worker_key(self) -> None:
        # A checker pod holds only a worker key, so the checker-work router must not sit behind the blanket
        # read-key guard the other routers inherit. Mounting it the usual way locks every pod out.
        service = _mock_service()
        service.claim_next_checker_executions.return_value = []
        app.state.settings = MagicMock(read_key=READ_KEY, admin_key=ADMIN_KEY, worker_keys=[WORKER_KEY])
        response = self.client.post(
            "/checker-executions/claim-next",
            json={"checker_name": "av.egomotion", "claimed_by": "pod-a", "lease_seconds": 30, "wait_seconds": 0},
            headers={"X-API-Key": WORKER_KEY},
        )
        self.assertEqual(response.status_code, 200)

    def test_claim_next_rejects_a_read_key(self) -> None:
        response = self.client.post(
            "/checker-executions/claim-next",
            json={"checker_name": "av.egomotion", "claimed_by": "pod-a", "lease_seconds": 30},
            headers={"X-API-Key": READ_KEY},
        )
        self.assertEqual(response.status_code, 403)

    def test_claim_next_rejects_no_key(self) -> None:
        response = self.client.post(
            "/checker-executions/claim-next",
            json={"checker_name": "av.egomotion", "claimed_by": "pod-a", "lease_seconds": 30},
        )
        self.assertEqual(response.status_code, 401)

    def test_the_reaper_rejects_a_worker_key(self) -> None:
        # Taking a job away from its owner is a fleet-wide decision. A pod holding a worker key must not be able to
        # requeue work it does not own.
        response = self.client.post("/checker-executions/reap", json={}, headers={"X-API-Key": WORKER_KEY})
        self.assertEqual(response.status_code, 401)

    def test_cancel_rejects_a_worker_key(self) -> None:
        # Cancellation is the arbitrator's decision, not something a pod can request for another pod's job.
        response = self.client.post("/checker-executions/7/cancel", headers={"X-API-Key": WORKER_KEY})
        self.assertEqual(response.status_code, 401)

    def test_states_accepts_a_read_key(self) -> None:
        service = _mock_service()
        service.get_checker_execution_states.return_value = []
        app.state.settings = MagicMock(read_key=READ_KEY, admin_key=ADMIN_KEY, worker_keys=[WORKER_KEY])
        response = self.client.post(
            "/checker-executions/states", json={"execution_ids": [7]}, headers={"X-API-Key": READ_KEY}
        )
        self.assertEqual(response.status_code, 200)

    def test_states_rejects_no_key(self) -> None:
        # The checker-work router forgoes the inherited read-key guard, so every one of its routes has to declare
        # its own. A route that forgets is wide open, and this is the guard against that.
        self.assertEqual(self.client.post("/checker-executions/states", json={"execution_ids": [7]}).status_code, 401)


class TestMainEntrypoint(unittest.TestCase):
    """The dev (:rest_api) and container (:rest_api_deployment) binaries both delegate to ``main``.

    ``main`` parses the uvicorn CLI args and serves the module-level ``app`` singleton. uvicorn is patched so nothing
    binds a socket; the test asserts the parsed host/port reach ``uvicorn.run`` and that the served object is ``app``.
    """

    def test_main_serves_app_with_uvicorn(self) -> None:
        with (
            patch("services.database.rest_api_common.uvicorn") as mock_uvicorn,
            patch("sys.argv", ["rest_api.py", "--host", "127.0.0.1", "--port", "9999"]),
        ):
            main()

        mock_uvicorn.run.assert_called_once()
        run_call = mock_uvicorn.run.call_args
        self.assertIs(run_call.args[0], app)
        self.assertEqual(run_call.kwargs["host"], "127.0.0.1")
        self.assertEqual(run_call.kwargs["port"], 9999)


if __name__ == "__main__":
    unittest.main()
