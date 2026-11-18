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

"""Tests for the CheckerExecutionRecord dataclass."""

import logging
import unittest
from unittest.mock import AsyncMock

from database.api_client import CheckerExecutionWrite
from services.arbitrator.records.checker_execution_record import CheckerExecutionRecord

# Suppress logging during tests
logging.getLogger("services.arbitrator.records.checker_execution_record").setLevel(logging.CRITICAL)

_WRITTEN = CheckerExecutionWrite(written=True, existing_status=None)


def _api_client(write: CheckerExecutionWrite = _WRITTEN) -> AsyncMock:
    """A client double whose ``update_checker_execution`` answers with a real outcome object.

    Worth a helper rather than a bare ``AsyncMock``: that returns a truthy stand-in for every attribute, so a record
    branching on ``write.written`` takes the success path whatever the server would have answered. The branch then
    has no test covering it while every test still passes.
    """
    client = AsyncMock()
    client.update_checker_execution.return_value = write
    return client


class TestCheckerExecutionRecord(unittest.IsolatedAsyncioTestCase):
    """Tests for CheckerExecutionRecord dataclass."""

    def test_checker_execution_record_initialization(self) -> None:
        """Test CheckerExecutionRecord initializes with correct default values."""
        mock_client = _api_client()
        record = CheckerExecutionRecord(
            api_client=mock_client,
            request_id=42,
            checker_name="av.objects_single_view",
            request_json={"camera_name": "front"},
        )

        self.assertEqual(record.api_client, mock_client)
        self.assertEqual(record.request_id, 42)
        self.assertEqual(record.checker_name, "av.objects_single_view")
        self.assertEqual(record.request_json, {"camera_name": "front"})
        self.assertIsNone(record.execution_id)
        self.assertIsNone(record.camera_name)

    async def test_checker_execution_record_save_success(self) -> None:
        """save() inserts the row and keeps its id, sending no version: the pod that claims the job stamps its own."""
        mock_client = _api_client()
        mock_client.create_checker_execution.return_value = 456

        record = CheckerExecutionRecord(
            api_client=mock_client,
            request_id=42,
            checker_name="av.objects_single_view",
            request_json={"camera_name": "front"},
        )
        result = await record.save()

        self.assertTrue(result)
        self.assertEqual(record.execution_id, 456)
        mock_client.create_checker_execution.assert_awaited_once_with(
            request_id=42,
            checker_name="av.objects_single_view",
            request_json={"camera_name": "front"},
            camera_name=None,
        )

    async def test_checker_execution_record_save_returns_false_when_none(self) -> None:
        """Test CheckerExecutionRecord.save returns False when API returns None."""
        mock_client = _api_client()
        mock_client.create_checker_execution.return_value = None

        record = CheckerExecutionRecord(
            api_client=mock_client,
            request_id=42,
            checker_name="av.objects_single_view",
            request_json={"test": "data"},
        )
        result = await record.save()

        self.assertFalse(result)
        self.assertIsNone(record.execution_id)

    async def test_checker_execution_record_save_forwards_camera_name(self) -> None:
        """Per-camera fan-out: camera_name is passed through to the API client."""
        mock_client = _api_client()
        mock_client.create_checker_execution.return_value = 789

        record = CheckerExecutionRecord(
            api_client=mock_client,
            request_id=42,
            checker_name="av.egomotion",
            request_json={"video_url": "s3://v.mp4"},
            camera_name="camera_front_wide_120fov",
        )
        result = await record.save()

        self.assertTrue(result)
        self.assertEqual(record.camera_name, "camera_front_wide_120fov")
        self.assertEqual(record.execution_id, 789)
        mock_client.create_checker_execution.assert_awaited_once_with(
            request_id=42,
            checker_name="av.egomotion",
            request_json={"video_url": "s3://v.mp4"},
            camera_name="camera_front_wide_120fov",
        )

    async def test_checker_execution_record_save_api_error(self) -> None:
        """Test CheckerExecutionRecord.save handles API errors gracefully."""
        mock_client = _api_client()
        mock_client.create_checker_execution.side_effect = Exception("API error")

        record = CheckerExecutionRecord(
            api_client=mock_client,
            request_id=42,
            checker_name="av.objects_single_view",
            request_json={"test": "data"},
        )
        result = await record.save()

        self.assertFalse(result)
        self.assertIsNone(record.execution_id)

    async def test_checker_execution_record_enqueue_writes_the_payload_the_pod_will_run(self) -> None:
        """The row's payload replaces the admission-time one, since the pod runs whatever the row holds."""
        mock_client = _api_client()
        record = CheckerExecutionRecord(
            api_client=mock_client,
            request_id=42,
            checker_name="av.objects_single_view",
            request_json={"placeholder": True},
        )
        record.execution_id = 456

        result = await record.enqueue({"video_path": "s3://bucket/clip.mp4"})

        self.assertTrue(result)
        mock_client.update_checker_execution.assert_awaited_once_with(
            execution_id=456, status="queued", request_json={"video_path": "s3://bucket/clip.mp4"}
        )

    async def test_checker_execution_record_enqueue_reports_a_failed_write(self) -> None:
        """The caller has to know: a job whose payload never landed cannot be run by the pod that claims it."""
        mock_client = _api_client()
        mock_client.update_checker_execution.side_effect = RuntimeError("database unreachable")
        record = CheckerExecutionRecord(
            api_client=mock_client,
            request_id=42,
            checker_name="av.objects_single_view",
            request_json={"test": "data"},
        )
        record.execution_id = 456

        self.assertFalse(await record.enqueue({"clip_id": "abc"}))

    async def test_checker_execution_record_enqueue_without_a_row_reports_failure(self) -> None:
        """There is nothing for a pod to claim, so the caller must fail the job rather than wait for one."""
        record = CheckerExecutionRecord(
            api_client=AsyncMock(),
            request_id=42,
            checker_name="av.objects_single_view",
            request_json={"test": "data"},
        )

        self.assertFalse(await record.enqueue({"clip_id": "abc"}))

    async def test_checker_execution_record_enqueue_on_a_finished_row_is_not_a_failure(self) -> None:
        """An already-finished job needs no payload, which is the opposite of a job nobody can run.

        Happens when a replica adopts a request and re-offers a job that a pod claimed and finished in the meantime.
        Reporting it as a failed write would have the caller fail a job that had in fact succeeded.
        """
        mock_client = _api_client(CheckerExecutionWrite(written=False, existing_status="success"))
        record = CheckerExecutionRecord(
            api_client=mock_client,
            request_id=42,
            checker_name="av.objects_single_view",
            request_json={"placeholder": True},
        )
        record.execution_id = 456

        self.assertIs(await record.enqueue({"clip_id": "abc"}), True)

    async def test_checker_execution_record_skipped(self) -> None:
        """``record.skipped()`` finalizes the row with status 'skipped'."""
        mock_client = _api_client()
        record = CheckerExecutionRecord(
            api_client=mock_client,
            request_id=42,
            checker_name="av.objects_single_view",
            request_json={"test": "data"},
        )
        record.execution_id = 456

        await record.skipped(error_message="Dependency failed")

        call_kwargs = mock_client.update_checker_execution.call_args.kwargs
        self.assertEqual(call_kwargs["status"], "skipped")
        self.assertEqual(call_kwargs["error_message"], "Dependency failed")

    async def test_checker_execution_record_complete(self) -> None:
        """Test CheckerExecutionRecord.complete updates with response and duration_ms."""
        mock_client = _api_client()
        record = CheckerExecutionRecord(
            api_client=mock_client,
            request_id=42,
            checker_name="av.objects_single_view",
            request_json={"test": "data"},
        )
        record.execution_id = 456

        await record.complete(
            response_json={"success": True, "data": {"score": 0.95}},
            duration_ms=1234,
            output_metadata={"frames_processed": 100},
        )

        mock_client.update_checker_execution.assert_awaited_once_with(
            execution_id=456,
            status="success",
            response_json={"success": True, "data": {"score": 0.95}},
            duration_ms=1234,
            error_message=None,
            output_metadata={"frames_processed": 100},
        )

    async def test_checker_execution_record_complete_without_metadata(self) -> None:
        """Test CheckerExecutionRecord.complete works without output_metadata."""
        mock_client = _api_client()
        record = CheckerExecutionRecord(
            api_client=mock_client,
            request_id=42,
            checker_name="av.objects_single_view",
            request_json={"test": "data"},
        )
        record.execution_id = 456

        await record.complete(response_json={"success": True}, duration_ms=500)

        call_kwargs = mock_client.update_checker_execution.call_args[1]
        self.assertEqual(call_kwargs["status"], "success")
        self.assertEqual(call_kwargs["duration_ms"], 500)
        self.assertIsNone(call_kwargs["output_metadata"])

    async def test_checker_execution_record_fail(self) -> None:
        """Test CheckerExecutionRecord.fail updates with error message."""
        mock_client = _api_client()
        record = CheckerExecutionRecord(
            api_client=mock_client,
            request_id=42,
            checker_name="av.objects_single_view",
            request_json={"test": "data"},
        )
        record.execution_id = 456

        await record.fail(
            response_json={"success": False, "error": "Processing failed"},
            duration_ms=100,
            error_message="Status 500: Internal Server Error",
        )

        call_kwargs = mock_client.update_checker_execution.call_args[1]
        self.assertEqual(call_kwargs["status"], "fail")
        self.assertEqual(call_kwargs["duration_ms"], 100)
        self.assertEqual(call_kwargs["error_message"], "Status 500: Internal Server Error")

    async def test_checker_execution_record_timeout(self) -> None:
        """Test CheckerExecutionRecord.timeout updates with timeout status."""
        mock_client = _api_client()
        record = CheckerExecutionRecord(
            api_client=mock_client,
            request_id=42,
            checker_name="av.objects_single_view",
            request_json={"test": "data"},
        )
        record.execution_id = 456

        await record.timeout(error_message="Request timeout after 600s")

        mock_client.update_checker_execution.assert_awaited_once_with(
            execution_id=456,
            status="timeout",
            response_json={},
            duration_ms=0,
            error_message="Request timeout after 600s",
            output_metadata=None,
        )

    async def test_checker_execution_record_final_write_refused_on_a_finished_row(self) -> None:
        """The pod's own account of a job wins over the arbitrator's outside verdict.

        Reported as not written so the caller does not record the row as carrying the status it asked for, since it
        carries the pod's instead.
        """
        mock_client = _api_client(CheckerExecutionWrite(written=False, existing_status="fail"))
        record = CheckerExecutionRecord(
            api_client=mock_client,
            request_id=42,
            checker_name="av.egomotion",
            request_json={},
        )
        record.execution_id = 456

        self.assertIs(await record.timeout(error_message="Checker exceeded its 5400s budget"), False)

    async def test_checker_execution_record_update_no_execution_id(self) -> None:
        """Test CheckerExecutionRecord update methods do nothing when execution_id is None."""
        mock_client = _api_client()
        record = CheckerExecutionRecord(
            api_client=mock_client,
            request_id=42,
            checker_name="av.objects_single_view",
            request_json={"test": "data"},
        )

        await record.complete({"result": "ok"}, duration_ms=100)
        await record.fail({"error": "test"}, duration_ms=50)
        await record.timeout("timeout error")

        mock_client.update_checker_execution.assert_not_awaited()

    async def test_checker_execution_record_complete_reports_a_landed_write(self) -> None:
        """The caller records the row as terminal only on the strength of this, so it must reflect the real answer."""
        record = CheckerExecutionRecord(
            api_client=_api_client(),
            request_id=42,
            checker_name="av.objects_single_view",
            request_json={"test": "data"},
        )
        record.execution_id = 456

        self.assertIs(await record.complete({"result": "ok"}, duration_ms=100), True)

    async def test_checker_execution_record_update_api_error(self) -> None:
        """Test CheckerExecutionRecord update methods handle API errors gracefully."""
        mock_client = _api_client()
        mock_client.update_checker_execution.side_effect = Exception("API error")

        record = CheckerExecutionRecord(
            api_client=mock_client,
            request_id=42,
            checker_name="av.objects_single_view",
            request_json={"test": "data"},
        )
        record.execution_id = 456

        # Should not raise
        await record.complete({"result": "ok"}, duration_ms=100)
        await record.fail({"error": "test"}, duration_ms=50)
        await record.timeout("timeout error")


class TestAttach(unittest.IsolatedAsyncioTestCase):
    """Binding a record to a row that already exists."""

    async def test_an_attached_record_writes_to_the_existing_row(self) -> None:
        client = _api_client()
        record = CheckerExecutionRecord.attach(
            api_client=client,
            request_id=42,
            execution_id=456,
            checker_name="av.egomotion",
            camera_name="camera_front_wide_120fov",
        )

        await record.enqueue({"clip_id": "abc"})

        self.assertEqual(record.execution_id, 456)
        client.create_checker_execution.assert_not_awaited()
        self.assertEqual(client.update_checker_execution.call_args[1]["execution_id"], 456)

    async def test_the_row_s_own_identity_is_preserved(self) -> None:
        """A rebuilt record has to keep pointing at the same checker and camera, or its later writes go elsewhere."""
        record = CheckerExecutionRecord.attach(
            api_client=AsyncMock(),
            request_id=42,
            execution_id=456,
            checker_name="av.egomotion",
            camera_name="camera_front_wide_120fov",
        )

        self.assertEqual(record.checker_name, "av.egomotion")
        self.assertEqual(record.camera_name, "camera_front_wide_120fov")


if __name__ == "__main__":
    unittest.main()
