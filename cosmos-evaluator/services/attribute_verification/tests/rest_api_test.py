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

"""Unit tests for the Attribute Verification REST API.

Scoped to what this service contributes: its identity, the way it maps a cloud request onto a local one, and the fact
that it uploads nothing. The transport around them -- status mapping, slot accounting, the temp-directory lifecycle,
the job protocol -- belongs to ``services.framework.checker_app`` and is tested there once rather than per checker.
"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from checks.attribute_verification.processor import (
    AttributeVerificationCheck,
    AttributeVerificationResult,
    AttributeVerificationSummary,
)
from services.attribute_verification.models import AttributeVerificationCloudRequest
from services.attribute_verification.rest_api import main
from services.attribute_verification.rest_api_common import _build_response_data, _process_request, app
from services.utils import get_contents_from_runfile

# Mock get_git_sha because //utils:git_sha is not in this test's runfiles. Stopped via
# ``addModuleCleanup`` so the patch does not leak into other modules in the same test run.
_git_sha_patcher = patch("services.utils.get_git_sha", return_value="test_sha")
_git_sha_patcher.start()
unittest.addModuleCleanup(_git_sha_patcher.stop)

_VALID_BODY = {"clip_id": "test_clip_id", "augmented_video_url": "s3://b/test_augmented.mp4"}


def _result(clip_id: str = "test_clip_id") -> AttributeVerificationResult:
    return AttributeVerificationResult(
        clip_id=clip_id,
        passed=True,
        summary=AttributeVerificationSummary(total_checks=1, passed_checks=1, failed_checks=0),
        checks=[
            AttributeVerificationCheck(
                variable="weather",
                value="rain",
                question="What is the weather?",
                options={"A": "rain", "B": "sun"},
                expected_answer="A",
                vlm_answer="A",
                passed=True,
            )
        ],
    )


class TestAttributeVerificationApp(unittest.TestCase):
    """The app is assembled from this service's spec, with its own identity and its own service behind it.

    ``TestClient`` is used without its context manager: entering it would run the real lifespan, which builds the real
    service and real storage clients. The collaborators go onto ``app.state`` directly, which is where the transport
    reads them from.
    """

    def setUp(self) -> None:
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        self.service = MagicMock()
        self.service.get_default_config = AsyncMock(return_value={})
        app.state.service = self.service
        app.state.input_storage_provider = MagicMock()
        app.state.output_storage_provider = MagicMock()
        self.addCleanup(self._reset_state)

    def _reset_state(self) -> None:
        app.state.service = None
        app.state.input_storage_provider = None
        app.state.output_storage_provider = None

    def test_health_reports_this_services_identity(self) -> None:
        data = self.client.get("/health").json()["data"]

        self.assertEqual(data["service"], "Attribute Verification Service API")
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["version"], get_contents_from_runfile("services/attribute_verification/version.txt"))
        self.assertEqual(data["git_sha"], "test_sha")
        self.assertEqual(data["checker"], "metropolis.attribute_verification")
        self.assertEqual(data["slots"], {"active": 0, "limit": app.state.slots.limit})

    def test_metrics_are_labeled_for_this_checker(self) -> None:
        response = self.client.get("/metrics")

        self.assertIn("version=0.0.4", response.headers["content-type"])
        self.assertIn(
            f'checker_slots_limit{{checker="metropolis.attribute_verification"}} {app.state.slots.limit}',
            response.text,
        )

    def test_config_exposes_the_attribute_verification_defaults(self) -> None:
        self.service.get_default_config = AsyncMock(return_value={"enabled": True, "question_generation": {}})

        data = self.client.get("/config").json()["data"]

        self.assertIn("question_generation", data["default_config"])
        self.assertIn("description", data)

    def test_a_request_runs_this_services_pipeline_end_to_end(self) -> None:
        self.service.validate_input = AsyncMock(return_value=True)
        self.service.process = AsyncMock(return_value=_result())
        self.service.cleanup = AsyncMock()

        response = self.client.post("/process", json=_VALID_BODY)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["data"]["clip_id"], "test_clip_id")
        self.assertTrue(body["data"]["passed"])
        self.assertIn("duration_ms", body["metadata"])
        # The request the service saw came from this checker's own prepare step, not from the test.
        local_request = self.service.process.await_args.args[0]
        self.assertEqual(local_request.augmented_video_path, "s3://b/test_augmented.mp4")

    def test_nothing_is_cleaned_up_because_nothing_was_written(self) -> None:
        """This check reports verdicts only, so the transport has no output directory to hand back to the service."""
        self.service.validate_input = AsyncMock(return_value=True)
        self.service.process = AsyncMock(return_value=_result())
        self.service.cleanup = AsyncMock()

        self.client.post("/process", json=_VALID_BODY)

        self.service.cleanup.assert_not_awaited()

    def test_unusable_input_is_the_callers_fault(self) -> None:
        self.service.validate_input = AsyncMock(side_effect=ValueError("bad input"))

        response = self.client.post("/process", json=_VALID_BODY)

        self.assertEqual(response.status_code, 422)
        self.assertIn("bad input", response.json()["error"]["message"])

    def test_a_processing_failure_is_ours(self) -> None:
        self.service.validate_input = AsyncMock(return_value=True)
        self.service.process = AsyncMock(side_effect=RuntimeError("some error occurred"))

        response = self.client.post("/process", json=_VALID_BODY)

        self.assertEqual(response.status_code, 500)
        self.assertIn("some error occurred", response.json()["error"]["message"])

    def test_endpoint_rejects_a_malformed_body(self) -> None:
        self.assertEqual(self.client.post("/process", content=b"not json").status_code, 422)
        self.assertEqual(self.client.post("/process", json={}).status_code, 422)


class TestRequestPreparation(unittest.IsolatedAsyncioTestCase):
    """The processor resolves the remote video itself, so preparation is a mapping rather than a download."""

    async def test_the_url_flows_straight_onto_the_local_path(self) -> None:
        provider = MagicMock()
        temp_dir = Path(tempfile.gettempdir())
        request = AttributeVerificationCloudRequest(
            clip_id="clip_1",
            augmented_video_url="s3://b/aug.mp4",
            config={"question_generation": {}},
            verbose="DEBUG",
            output_storage_prefix="my-team/run-42/",
        )

        local_request = await _process_request(request, provider, temp_dir)

        self.assertEqual(local_request.clip_id, "clip_1")
        self.assertEqual(local_request.augmented_video_path, "s3://b/aug.mp4")
        self.assertEqual(local_request.config, {"question_generation": {}})
        self.assertEqual(local_request.verbose, "DEBUG")
        self.assertEqual(local_request.output_storage_prefix, "my-team/run-42/")
        provider.download_from_url.assert_not_called()


class TestResponse(unittest.IsolatedAsyncioTestCase):
    """The verdicts reach the caller unchanged, and no artifact is uploaded on the way."""

    async def test_verdicts_are_returned_without_an_upload(self) -> None:
        provider = MagicMock()
        request = AttributeVerificationCloudRequest(clip_id="clip_1", augmented_video_url="s3://b/aug.mp4")

        data = await _build_response_data(request, _result("clip_1"), provider)

        self.assertEqual(data["clip_id"], "clip_1")
        self.assertEqual(data["summary"]["passed_checks"], 1)
        self.assertNotIn("output_dir", data)
        provider.store_file.assert_not_called()


class TestEntryPoint(unittest.TestCase):
    def test_main_entrypoint_runs_uvicorn(self) -> None:
        """main() assembles the CLI args and hands the app to uvicorn without starting a server."""
        with (
            patch("services.attribute_verification.rest_api_common.uvicorn") as fake_uvicorn,
            patch("sys.argv", ["rest_api.py"]),
        ):
            main()

        fake_uvicorn.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
