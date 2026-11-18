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

"""Unit tests for the client-agnostic Arbitrator REST app (``rest_api_common.ArbitratorApp``).

Covers the pure result/URL helpers plus the synchronous discovery/health/observation endpoints driven through
FastAPI's ``TestClient`` (lifespan running, no Database REST API client installed). The submit-and-poll scheduler
happy path is intentionally out of scope here.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from fastapi.testclient import TestClient

from services.arbitrator.checker_queue import CheckerJob, CheckerQueue
from services.arbitrator.dependency_graph import DependencyGraph
from services.arbitrator.models.contract import (
    ArbitratorRequest,
    CheckerInputs,
)
from services.arbitrator.records.execution_record import ExecutionRecord
from services.arbitrator.rest_api_common import (
    ArbitratorApp,
    _collect_input_urls,
    _find_expired_urls,
    _store_checker_result,
)
from services.arbitrator.settings import Settings

_CONFIG = {
    "av.objects_single_view": {"enabled": True, "deps": []},
    "metropolis.hallucination": {"enabled": True, "deps": []},
    "metropolis.attribute_verification": {"enabled": True, "deps": []},
}

_ENDPOINTS = {
    "av.objects_single_view": "http://objects",
    "metropolis.hallucination": "http://hallucination",
    "metropolis.attribute_verification": "http://attribute_verification",
}


def _build_settings() -> Settings:
    """Build a real base ``Settings`` with YAML/runfiles resolution stubbed out for tests."""
    graph = DependencyGraph(_CONFIG)
    graph.validate()
    with (
        patch.object(Settings, "_resolve_checker_endpoints", return_value=_ENDPOINTS),
        patch.object(Settings, "_resolve_database_api_url", return_value=""),
        patch.object(Settings, "_load_dependency_graph", return_value=graph),
        patch("services.settings_base.SettingsBase.get_env_files", return_value=[]),
    ):
        return Settings(_env_file=[])


class TestResultHelpers(unittest.TestCase):
    """Pure helpers for aggregating per-job results into the checker-results map."""

    def test_store_nests_parameterized_job_under_camera(self) -> None:
        results: dict = {}
        job = CheckerJob(
            name="av.objects_single_view[camera_front_wide_120fov]",
            base_name="av.objects_single_view",
            priority=0,
            deps=set(),
            params={"camera_name": "camera_front_wide_120fov"},
        )
        _store_checker_result(results, job, {"score": 1.0})
        self.assertEqual(results, {"av.objects_single_view": {"camera_front_wide_120fov": {"score": 1.0}}})

    def test_store_writes_plain_job_flat(self) -> None:
        results: dict = {}
        _store_checker_result(results, CheckerJob(name="x", base_name="x", priority=0, deps=set()), {"ok": True})
        self.assertEqual(results, {"x": {"ok": True}})


class TestExpiredUrlHelpers(unittest.TestCase):
    """``_collect_input_urls`` / ``_find_expired_urls`` enumerate inputs and flag only definitively-expired URLs."""

    @staticmethod
    def _request() -> ArbitratorRequest:
        return ArbitratorRequest(
            inputs=CheckerInputs(
                augmented_video_urls={"camera_front_wide_120fov": "s3://b/aug.mp4"},
                rds_hq_url="s3://b/rds.zip",
                world_model_video_urls={"camera_front_wide_120fov": "s3://b/wm.mp4"},
                original_video_urls={"camera_front_wide_120fov": "s3://b/orig.mp4"},
            )
        )

    def test_collect_returns_all_input_urls(self) -> None:
        labels = {label for label, _ in _collect_input_urls(self._request())}
        self.assertEqual(
            labels,
            {
                "inputs.rds_hq_url",
                "inputs.augmented_video_urls[camera_front_wide_120fov]",
                "inputs.world_model_video_urls[camera_front_wide_120fov]",
                "inputs.original_video_urls[camera_front_wide_120fov]",
            },
        )

    def test_collect_omits_rds_hq_url_when_absent(self) -> None:
        # Metropolis-only request: rds_hq_url is optional and absent, so it must not appear in the collected URLs.
        request = ArbitratorRequest(
            inputs=CheckerInputs(
                augmented_video_urls={"camera_front_wide_120fov": "s3://b/aug.mp4"},
                original_video_urls={"camera_front_wide_120fov": "s3://b/orig.mp4"},
            )
        )
        labels = {label for label, _ in _collect_input_urls(request)}
        self.assertEqual(
            labels,
            {
                "inputs.augmented_video_urls[camera_front_wide_120fov]",
                "inputs.original_video_urls[camera_front_wide_120fov]",
            },
        )

    def test_unsigned_urls_are_not_flagged_expired(self) -> None:
        self.assertEqual(_find_expired_urls(self._request()), [])


class TestBuildCheckerRequest(unittest.TestCase):
    """``build_checker_request`` builds the correct per-checker cloud payload from a job + arbitrator request."""

    def setUp(self) -> None:
        self._app = ArbitratorApp(settings=_build_settings())

    @staticmethod
    def _request() -> ArbitratorRequest:
        return ArbitratorRequest(
            inputs=CheckerInputs(
                augmented_video_urls={"camera_front_wide_120fov": "s3://b/aug.mp4"},
                rds_hq_url="s3://b/rds.zip",
                world_model_video_urls={"camera_front_wide_120fov": "s3://b/wm.mp4"},
                original_video_urls={"camera_front_wide_120fov": "s3://b/orig.mp4"},
                config={"threshold": 0.5},
            ),
            output_storage_prefix="run-42/",
        )

    def test_single_view_request(self) -> None:
        job = CheckerJob(
            name="av.objects_single_view[camera_front_wide_120fov]",
            base_name="av.objects_single_view",
            priority=0,
            deps=set(),
            params={
                "camera_name": "camera_front_wide_120fov",
                "augmented_video_url": "s3://b/aug.mp4",
                "world_model_video_url": "s3://b/wm.mp4",
            },
        )
        payload = self._app.build_checker_request(job, self._request())
        self.assertEqual(payload["camera_name"], "camera_front_wide_120fov")
        self.assertEqual(payload["rds_hq_url"], "s3://b/rds.zip")
        self.assertEqual(payload["world_model_video_url"], "s3://b/wm.mp4")

    def test_hallucination_request_maps_videos_and_generates_clip_id(self) -> None:
        job = CheckerJob(
            name="metropolis.hallucination[camera_front_wide_120fov]",
            base_name="metropolis.hallucination",
            priority=0,
            deps=set(),
            params={
                "camera_name": "camera_front_wide_120fov",
                "augmented_video_url": "s3://b/aug.mp4",
                "original_video_url": "s3://b/orig.mp4",
            },
        )
        payload = self._app.build_checker_request(job, self._request())
        self.assertEqual(payload["augmented_video_url"], "s3://b/aug.mp4")
        self.assertEqual(payload["original_video_url"], "s3://b/orig.mp4")
        self.assertEqual(payload["config"], {"threshold": 0.5})
        self.assertEqual(payload["output_storage_prefix"], "run-42/")
        # clip_id is a generated uuid4 (no client-supplied id on the arbitrator request).
        self.assertEqual(str(uuid.UUID(payload["clip_id"])), payload["clip_id"])

    def test_attribute_verification_request_maps_video_and_generates_clip_id(self) -> None:
        job = CheckerJob(
            name="metropolis.attribute_verification[camera_front_wide_120fov]",
            base_name="metropolis.attribute_verification",
            priority=0,
            deps=set(),
            params={
                "camera_name": "camera_front_wide_120fov",
                "augmented_video_url": "s3://b/aug.mp4",
            },
        )
        payload = self._app.build_checker_request(job, self._request())
        self.assertEqual(payload["augmented_video_url"], "s3://b/aug.mp4")
        self.assertNotIn("original_video_url", payload)
        self.assertEqual(str(uuid.UUID(payload["clip_id"])), payload["clip_id"])

    def test_physical_plausibility_request_accepts_shared_config_and_generates_clip_id(self) -> None:
        job = CheckerJob(
            name="metropolis.physical_plausibility[camera_front_wide_120fov]",
            base_name="metropolis.physical_plausibility",
            priority=0,
            deps=set(),
            params={
                "camera_name": "camera_front_wide_120fov",
                "augmented_video_url": "s3://b/aug.mp4",
            },
        )
        request = self._request()
        request.inputs.config = {
            "selected_variables": {"anomaly_type": "person_falling"},
            "question_generation": {"llm": {"model": "test-llm"}},
            "vlm_verification": {"vlm": {"model": "test-vlm"}},
        }
        payload = self._app.build_checker_request(job, request)
        self.assertEqual(payload["video_url"], "s3://b/aug.mp4")
        self.assertEqual(payload["config"], request.inputs.config)
        self.assertEqual(payload["output_storage_prefix"], "run-42/")
        self.assertEqual(str(uuid.UUID(payload["clip_id"])), payload["clip_id"])

    def test_unknown_checker_raises(self) -> None:
        job = CheckerJob(name="unknown.checker", base_name="unknown.checker", priority=0, deps=set())
        with self.assertRaises(NotImplementedError):
            self._app.build_checker_request(job, self._request())


class TestAdmissionPayloads(unittest.IsolatedAsyncioTestCase):
    """What the eagerly-inserted ``checker_executions`` rows carry, and what admission declines to ask for."""

    def setUp(self) -> None:
        self._app = ArbitratorApp(settings=_build_settings())
        self._api_client = AsyncMock()
        self._api_client.create_checker_execution.return_value = 101
        self._app._api_client = self._api_client
        self._execution = ExecutionRecord(
            api_client=self._api_client, client_id=1, request_json={}, version="v1", commit_sha="sha"
        )
        self._execution.request_id = 7

    @staticmethod
    def _queue() -> CheckerQueue:
        """One independent job and one that depends on it, which ``add_jobs`` lands in READY and PENDING."""
        return CheckerQueue(
            [
                CheckerJob(name="leaf", base_name="leaf", priority=0, deps=set()),
                CheckerJob(name="aggregator", base_name="aggregator", priority=1, deps={"leaf"}),
            ]
        )

    @staticmethod
    def _request() -> ArbitratorRequest:
        return ArbitratorRequest(inputs=CheckerInputs(augmented_video_urls={"camera_front_wide_120fov": "s3://b/a"}))

    async def test_a_dependency_blocked_job_is_never_asked_for_a_payload(self) -> None:
        """Its payload is a function of results that do not exist, so asking can only raise.

        An aggregator's builder reports missing dependencies as ``ValueError``, which admission cannot tell from a
        real defect -- so asking at all meant a stack trace in the logs for every request carrying such a checker.
        """
        queue = self._queue()
        builder = MagicMock(side_effect=lambda job, *_: {"job": job.name})

        with patch.object(ArbitratorApp, "build_checker_request", builder):
            records = await self._app._create_checker_execution_records(queue, self._request(), self._execution)

        self.assertEqual([call.args[0].name for call in builder.call_args_list], ["leaf"])
        self.assertEqual(set(records), {"leaf", "aggregator"}, "both jobs still get a row to be found by")

    async def test_a_dependency_blocked_job_still_gets_its_row(self) -> None:
        """The row exists to be found -- by the snapshot endpoint, and by a replica rebuilding after an adoption."""
        with patch.object(ArbitratorApp, "build_checker_request", MagicMock(return_value={})):
            await self._app._create_checker_execution_records(self._queue(), self._request(), self._execution)

        payloads = {
            call.kwargs["checker_name"]: call.kwargs["request_json"]
            for call in self._api_client.create_checker_execution.await_args_list
        }
        self.assertEqual(payloads["aggregator"], {}, "nothing is knowable about it yet")

    async def test_admission_records_no_checker_version(self) -> None:
        """The row's version is the claiming pod's to write, so admission leaves it to the server's empty default.

        Writing the health probe's answer here would name whichever replica answered the probe, not the one that ends
        up running the job.
        """
        with patch.object(ArbitratorApp, "build_checker_request", MagicMock(return_value={})):
            await self._app._create_checker_execution_records(self._queue(), self._request(), self._execution)

        for call in self._api_client.create_checker_execution.await_args_list:
            self.assertNotIn("version", call.kwargs)
            self.assertNotIn("commit_sha", call.kwargs)

    async def test_an_unexpected_build_failure_still_does_not_break_admission(self) -> None:
        """The payload here is informational and the scheduler rebuilds the real one, so a raise must not be fatal."""
        builder = MagicMock(side_effect=RuntimeError("boom"))

        with patch.object(ArbitratorApp, "build_checker_request", builder), self.assertLogs(level="ERROR") as logs:
            records = await self._app._create_checker_execution_records(self._queue(), self._request(), self._execution)

        self.assertEqual(set(records), {"leaf", "aggregator"})
        self.assertTrue(any("Unexpected error building informational request_json" in line for line in logs.output))


class TestAdmissionValidation(unittest.IsolatedAsyncioTestCase):
    """``_validate_admission`` requires rds_hq_url only when single-view is among the checkers to run."""

    def setUp(self) -> None:
        self._app = ArbitratorApp(settings=_build_settings())

    async def test_single_view_requires_rds_hq_url(self) -> None:
        request = ArbitratorRequest(
            inputs=CheckerInputs(
                augmented_video_urls={"cam": "s3://b/aug.mp4"},
                world_model_video_urls={"cam": "s3://b/wm.mp4"},
            )
        )
        with patch.object(
            self._app, "_gather_healthy_checkers", new=AsyncMock(return_value=(["av.objects_single_view"], {}))
        ):
            validated, error = await self._app._validate_admission(request)
        self.assertIsNone(validated)
        self.assertIsNotNone(error)
        assert error is not None
        self.assertEqual(error.status_code, 422)

    async def test_metropolis_only_does_not_require_rds_hq_url(self) -> None:
        request = ArbitratorRequest(
            inputs=CheckerInputs(
                augmented_video_urls={"cam": "s3://b/aug.mp4"},
                original_video_urls={"cam": "s3://b/orig.mp4"},
            )
        )
        with patch.object(
            self._app, "_gather_healthy_checkers", new=AsyncMock(return_value=(["metropolis.hallucination"], {}))
        ):
            validated, error = await self._app._validate_admission(request)
        self.assertIsNone(error)
        self.assertIsNotNone(validated)

    async def test_transitively_included_single_view_requires_rds_hq_url(self) -> None:
        # A checker that transitively pulls in single-view as a dependency must also trip the rds_hq_url gate, even
        # though single-view was never directly among the requested ``checkers_to_run``. Validation runs against the
        # resolved queue, not the pre-resolution request, so the auto-included single-view job is covered.
        config = {
            "av.objects_single_view": {"enabled": True, "deps": []},
            "agg.needs_single_view": {"enabled": True, "deps": ["av.objects_single_view"]},
        }
        endpoints = {"av.objects_single_view": "http://objects", "agg.needs_single_view": "http://agg"}
        graph = DependencyGraph(config)
        graph.validate()
        with (
            patch.object(Settings, "_resolve_checker_endpoints", return_value=endpoints),
            patch.object(Settings, "_resolve_database_api_url", return_value=""),
            patch.object(Settings, "_load_dependency_graph", return_value=graph),
            patch("services.settings_base.SettingsBase.get_env_files", return_value=[]),
        ):
            app = ArbitratorApp(settings=Settings(_env_file=[]))
        request = ArbitratorRequest(
            inputs=CheckerInputs(
                augmented_video_urls={"cam": "s3://b/aug.mp4"},
                world_model_video_urls={"cam": "s3://b/wm.mp4"},
            )
        )
        with patch.object(app, "_gather_healthy_checkers", new=AsyncMock(return_value=(["agg.needs_single_view"], {}))):
            validated, error = await app._validate_admission(request)
        self.assertIsNone(validated)
        assert error is not None
        self.assertEqual(error.status_code, 422)


class TestRestApiEndpoints(unittest.TestCase):
    """Synchronous discovery/health/observation endpoints over the base app (no Database REST API client)."""

    def setUp(self) -> None:
        # ``//utils:git_sha`` is not in this test's runfiles, so the health endpoint's get_git_sha() call is stubbed for
        # the whole test.
        self._git_patcher = patch("services.utils.get_git_sha", return_value="test_sha")
        self._git_patcher.start()
        self.addCleanup(self._git_patcher.stop)
        self._fastapi_app = ArbitratorApp(settings=_build_settings()).create_app()

    def test_health_reports_healthy(self) -> None:
        with TestClient(self._fastapi_app) as client:
            response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(data["status"], "healthy")
        # No database_api_url is configured in _build_settings, so the additive DB signal reports "unavailable" while
        # the top-level status stays "healthy".
        self.assertEqual(data["database"], "unavailable")

    def test_health_reports_database_ok_when_reachable(self) -> None:
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        app = ArbitratorApp(settings=_build_settings())
        with patch.object(app, "init_api_client", return_value=mock_client):
            fastapi_app = app.create_app()
            with TestClient(fastapi_app) as client:
                response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["database"], "ok")
        mock_client.ping.assert_awaited()

    def test_metrics_exposes_per_checker_gauges(self) -> None:
        with TestClient(self._fastapi_app) as client:
            response = client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("arbitrator_checker_jobs", response.text)
        self.assertIn("av.objects_single_view", response.text)

    def test_dependency_graph_lists_checkers(self) -> None:
        with TestClient(self._fastapi_app) as client:
            response = client.get("/dependency-graph")
        self.assertEqual(response.status_code, 200)
        self.assertIn("av.objects_single_view", response.json()["data"]["checkers"])

    def test_execution_snapshot_returns_503_without_database(self) -> None:
        with TestClient(self._fastapi_app) as client:
            response = client.get("/executions/1")
        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
