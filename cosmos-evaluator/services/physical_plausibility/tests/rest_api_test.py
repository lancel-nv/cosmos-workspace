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

"""Contract tests for the physical-plausibility REST API."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from checks.physical_plausibility.processor import PhysicalPlausibilityResult
from services.physical_plausibility.models import PhysicalPlausibilityCloudRequest
from services.physical_plausibility.rest_api import main
from services.physical_plausibility.rest_api_common import _build_response_data, _process_request, _verify_startup, app

_VALID_BODY = {"clip_id": "clip-1", "video_url": "s3://videos/clip-1.mp4"}


def _result(clip_id: str = "clip-1") -> PhysicalPlausibilityResult:
    return PhysicalPlausibilityResult(
        clip_id=clip_id,
        passed=True,
        bad_score=0.2,
        plausibility_score=0.8,
        bad_threshold=0.4,
        prediction="good",
        fingerprint="test-fingerprint",
    )


class TestPhysicalPlausibilityApp(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.service = MagicMock()
        self.service.get_default_config = AsyncMock(return_value={"bad_threshold": 0.4})
        app.state.service = self.service
        app.state.input_storage_provider = MagicMock()
        app.state.output_storage_provider = MagicMock()
        self.addCleanup(self._reset_app_state)

    @staticmethod
    def _reset_app_state() -> None:
        app.state.service = None
        app.state.input_storage_provider = None
        app.state.output_storage_provider = None

    def test_health_and_metrics_identify_this_checker(self) -> None:
        health = self.client.get("/health").json()["data"]
        metrics = self.client.get("/metrics")

        self.assertEqual(health["service"], "Physical Plausibility Service API")
        self.assertEqual(health["checker"], "metropolis.physical_plausibility")
        self.assertEqual(health["slots"], {"active": 0, "limit": app.state.slots.limit})
        self.assertIn('checker_slots_limit{checker="metropolis.physical_plausibility"}', metrics.text)

    def test_config_exposes_service_defaults(self) -> None:
        data = self.client.get("/config").json()["data"]

        self.assertEqual(data["default_config"], {"bad_threshold": 0.4})
        self.assertIn("description", data)

    def test_process_maps_the_cloud_request_and_returns_metrics(self) -> None:
        self.service.validate_input = AsyncMock(return_value=True)
        self.service.process = AsyncMock(return_value=_result())
        self.service.cleanup = AsyncMock()

        response = self.client.post("/process", json=_VALID_BODY)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["plausibility_score"], 0.8)
        self.assertEqual(self.service.process.await_args.args[0].video_path, _VALID_BODY["video_url"])
        self.service.cleanup.assert_not_awaited()

    def test_invalid_and_failed_processing_return_expected_errors(self) -> None:
        self.service.validate_input = AsyncMock(side_effect=ValueError("bad video"))
        self.assertEqual(self.client.post("/process", json=_VALID_BODY).status_code, 422)

        self.service.validate_input = AsyncMock(return_value=True)
        self.service.process = AsyncMock(side_effect=RuntimeError("processor failed"))
        self.assertEqual(self.client.post("/process", json=_VALID_BODY).status_code, 500)
        self.assertEqual(self.client.post("/process", content=b"not json").status_code, 422)

    def test_process_accepts_unrelated_shared_config_fields(self) -> None:
        config = {
            "selected_variables": {"anomaly_type": "person_falling"},
            "question_generation": {"llm": {"model": "test-llm"}},
            "vlm_verification": {"vlm": {"model": "test-vlm"}},
        }
        self.service.validate_input = AsyncMock(return_value=True)
        self.service.process = AsyncMock(return_value=_result())

        response = self.client.post("/process", json=_VALID_BODY | {"config": config})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.service.process.await_args.args[0].config, config)

    def test_bad_threshold_remains_strictly_validated(self) -> None:
        invalid_threshold = self.client.post("/process", json=_VALID_BODY | {"config": {"bad_threshold": 1.1}})

        self.assertEqual(invalid_threshold.status_code, 422)


class TestCheckerSpecificMappings(unittest.IsolatedAsyncioTestCase):
    async def test_request_preparation_preserves_all_checker_fields(self) -> None:
        provider = MagicMock()
        request = PhysicalPlausibilityCloudRequest(
            clip_id="clip-2",
            video_url="msc://videos/clip-2.mp4",
            config={"bad_threshold": 0.3},
            output_storage_prefix="runs/42/",
        )

        local = await _process_request(request, provider, Path(tempfile.gettempdir()))

        self.assertEqual(local.clip_id, "clip-2")
        self.assertEqual(local.video_path, request.video_url)
        self.assertEqual(local.config, request.config)
        self.assertEqual(local.output_storage_prefix, "runs/42/")
        provider.download_from_url.assert_not_called()

    async def test_response_returns_metrics_without_uploading_artifacts(self) -> None:
        provider = MagicMock()
        request = PhysicalPlausibilityCloudRequest(**_VALID_BODY)

        data = await _build_response_data(request, _result(), provider)

        self.assertEqual(data["fingerprint"], "test-fingerprint")
        self.assertNotIn("output_dir", data)
        provider.store_file.assert_not_called()


class TestEntryPoint(unittest.TestCase):
    def test_verify_startup_configures_multistorage(self) -> None:
        with (
            patch("services.physical_plausibility.rest_api_common.setup_msc_config") as setup_msc_config,
            patch.dict(os.environ, {"MULTISTORAGECLIENT_CONFIGURATION": "{}"}, clear=True),
        ):
            _verify_startup()

        setup_msc_config.assert_called_once_with("{}")

    def test_main_passes_parsed_arguments_to_uvicorn(self) -> None:
        with (
            patch("services.physical_plausibility.rest_api_common.uvicorn.run") as run,
            patch("sys.argv", ["rest_api.py", "--host", "127.0.0.1", "--port", "9000", "--log-level", "DEBUG"]),
        ):
            main()

        self.assertEqual(run.call_args.kwargs["host"], "127.0.0.1")
        self.assertEqual(run.call_args.kwargs["port"], 9000)
        self.assertEqual(run.call_args.kwargs["log_level"], "debug")

    def test_main_defaults_to_physical_plausibility_port(self) -> None:
        with (
            patch("services.physical_plausibility.rest_api_common.uvicorn.run") as run,
            patch.dict(os.environ, {}, clear=True),
            patch("sys.argv", ["rest_api.py"]),
        ):
            main()

        self.assertEqual(run.call_args.kwargs["port"], 8088)


if __name__ == "__main__":
    unittest.main()
