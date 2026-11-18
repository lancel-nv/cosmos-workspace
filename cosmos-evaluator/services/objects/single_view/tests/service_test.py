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

"""Unit tests for ObjectsService."""

import asyncio
import contextlib
import math
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from checks.objects.single_view.object_processor_base import ObjectProcessorBase
from checks.utils.video import GpuUnavailableDecodeError
from services.objects.single_view.models import (
    ObjectsCloudRequest,
    ObjectsRequest,
    ObjectsResult,
)
from services.objects.single_view.service import ObjectsService


class TestObjectsService(unittest.IsolatedAsyncioTestCase):
    """Test cases for ObjectsService."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.service = ObjectsService()
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.input_dir.mkdir()

        self.video_file = Path(self.temp_dir) / "test_video.mp4"
        self.video_file.write_text("mock video content")

        self.world_video_file = Path(self.temp_dir) / "test_world_video.mp4"
        self.world_video_file.write_text("mock world video content")

        # Validation now probes both videos with ffprobe. The fixtures above are text files, and the container's
        # ffmpeg cannot decode H.264 without a GPU anyway, so the probe is stubbed here and driven explicitly by the
        # tests that care what it returns.
        probe_patcher = patch("services.objects.single_view.service.get_video_dimensions", return_value=(1920, 1080))
        self.mock_probe = probe_patcher.start()
        self.addCleanup(probe_patcher.stop)

        self.valid_config = {
            "av.objects_single_view": {
                "overlap_check": {"vehicle": {"method": "ratio"}},
                "importance_filter": {
                    "distance_threshold_m": 50.0,
                    "oncoming_objects": False,
                    "relevant_lanes": ["ego"],
                },
            }
        }

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(self.temp_dir)

    async def test_service_validation_success(self) -> None:
        """Test input validation with valid request."""
        valid_request = ObjectsRequest(
            input_data_path=str(self.input_dir),
            clip_id="test_clip",
            camera_name="front_camera",
            video_path=str(self.video_file),
            world_video_path=str(self.world_video_file),
            config=self.valid_config,
            model_device="cpu",
            verbose="INFO",
        )

        is_valid = await self.service.validate_input(valid_request)
        self.assertTrue(is_valid, "Valid request should pass validation")

    async def test_service_validation_missing_input_path(self) -> None:
        """Test input validation with missing input path."""
        invalid_request = ObjectsRequest(
            input_data_path="/nonexistent/path",
            clip_id="test_clip",
            camera_name="front_camera",
            video_path=str(self.video_file),
            world_video_path=str(self.world_video_file),
            config=self.valid_config,
            model_device="cpu",
            verbose="INFO",
        )

        with self.assertRaises(ValueError) as context:
            await self.service.validate_input(invalid_request)

        self.assertIn("Input data directory does not exist", str(context.exception))

    async def test_service_validation_missing_video(self) -> None:
        """Test input validation with missing video file."""
        invalid_request = ObjectsRequest(
            input_data_path=str(self.input_dir),
            clip_id="test_clip",
            camera_name="front_camera",
            video_path="/nonexistent/video.mp4",
            world_video_path=str(self.world_video_file),
            config=self.valid_config,
            model_device="cpu",
            verbose="INFO",
        )

        with self.assertRaises(ValueError) as context:
            await self.service.validate_input(invalid_request)

        self.assertIn("Video file does not exist", str(context.exception))

    async def test_service_validation_missing_world_video(self) -> None:
        """A missing world-model video is rejected here rather than by the static processor mid-run."""
        invalid_request = ObjectsRequest(
            input_data_path=str(self.input_dir),
            clip_id="test_clip",
            camera_name="front_camera",
            video_path=str(self.video_file),
            world_video_path="/nonexistent/world_video.mp4",
            config=self.valid_config,
            model_device="cpu",
            verbose="INFO",
        )

        with self.assertRaises(ValueError) as context:
            await self.service.validate_input(invalid_request)

        self.assertIn("World model video file does not exist", str(context.exception))

    async def test_service_validation_rejects_undecodable_video(self) -> None:
        """A file that downloaded but has no video stream is the caller's problem, so it fails as a ValueError (422).

        Existence alone let a zero-byte or truncated download through, and the run then died deep inside the frame
        loop after paying for the RDS zip and the model load.
        """
        self.mock_probe.return_value = None
        request = ObjectsRequest(
            input_data_path=str(self.input_dir),
            clip_id="test_clip",
            camera_name="front_camera",
            video_path=str(self.video_file),
            world_video_path=str(self.world_video_file),
            config=self.valid_config,
            model_device="cpu",
            verbose="INFO",
        )

        with self.assertRaises(ValueError) as context:
            await self.service.validate_input(request)

        self.assertIn("not a readable video", str(context.exception))

    async def test_service_validation_propagates_gpu_unavailable(self) -> None:
        """An unusable GPU must not be laundered into a 422: the input is fine and the replica is not.

        ``GpuUnavailableDecodeError`` is a ``VideoDecodeError`` rather than a ``ValueError``, so it reaches the
        transport as a 500 and the job's recorded error names the GPU instead of blaming the video.
        """
        self.mock_probe.side_effect = GpuUnavailableDecodeError("ffprobe could not initialize CUDA")
        request = ObjectsRequest(
            input_data_path=str(self.input_dir),
            clip_id="test_clip",
            camera_name="front_camera",
            video_path=str(self.video_file),
            world_video_path=str(self.world_video_file),
            config=self.valid_config,
            model_device="cpu",
            verbose="INFO",
        )

        with self.assertRaises(GpuUnavailableDecodeError):
            await self.service.validate_input(request)

    @patch("services.objects.single_view.service.api.run_object_processors")
    async def test_service_mocked_processing(self, mock_run: MagicMock) -> None:
        """Test service processing with mocked processor."""
        request = ObjectsRequest(
            input_data_path=str(self.input_dir),
            clip_id="test_clip",
            camera_name="front_camera",
            video_path=str(self.video_file),
            world_video_path=str(self.world_video_file),
            config=self.valid_config,
            model_device="cpu",
            verbose="INFO",
            trial_frames=5,
        )

        # Mock the run function to avoid actual processing
        mock_results = {
            "dynamic": {
                "processed_frames": 5,
                "total_video_frames": 100,
                "score_matrix": None,
                "track_ids": {1, 2, 3},
                "processed_frame_ids": [0, 1, 2, 3, 4],
                "mean_score": 0.75,
                "std_score": 0.15,
                "min_score": 0.5,
                "max_score": 0.9,
            }
        }

        mock_run.return_value = mock_results

        # Process the request
        result = await self.service.process(request)

        # Verify result
        self.assertIsInstance(result, ObjectsResult)
        self.assertEqual(result.clip_id, "test_clip")
        self.assertEqual(result.processed_frames, 5)
        self.assertEqual(result.mean_score, 0.75)
        self.assertEqual(set(result.unique_track_ids), {1, 2, 3})

    async def test_output_storage_prefix_default(self) -> None:
        """Test that output_storage_prefix defaults to None."""
        request = ObjectsRequest(
            input_data_path=str(self.input_dir),
            clip_id="test_clip",
            camera_name="front_camera",
            video_path=str(self.video_file),
            world_video_path=str(self.world_video_file),
            config=self.valid_config,
            model_device="cpu",
            verbose="INFO",
        )

        self.assertIsNone(request.output_storage_prefix)

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.is_dir")
    @patch("shutil.rmtree")
    async def test_cleanup_removes_any_directory(
        self, mock_rmtree: MagicMock, mock_is_dir: MagicMock, mock_exists: MagicMock
    ) -> None:
        """Test that cleanup removes any output directory when it exists."""
        mock_exists.return_value = True
        mock_is_dir.return_value = True

        # Test with temporary directory - should clean up
        temp_output_dir = Path("/tmp/objects/test123")
        await self.service.cleanup(temp_output_dir)
        mock_rmtree.assert_called_once_with(temp_output_dir)

        # Reset mock
        mock_rmtree.reset_mock()

        # Test with custom directory - should also clean up now
        custom_output_dir = Path("/home/user/custom_output")
        await self.service.cleanup(custom_output_dir)
        mock_rmtree.assert_called_once_with(custom_output_dir)

    @patch("services.objects.single_view.service.api.run_object_processors")
    async def test_concurrent_requests_no_output_dir_collision(self, mock_run: MagicMock) -> None:
        """Test that concurrent requests get unique directories."""
        request1 = ObjectsRequest(
            input_data_path=str(self.input_dir),
            clip_id="test_clip_1",
            camera_name="front_camera",
            video_path=str(self.video_file),
            world_video_path=str(self.world_video_file),
            config=self.valid_config,
            model_device="cpu",
            verbose="INFO",
        )

        request2 = ObjectsRequest(
            input_data_path=str(self.input_dir),
            clip_id="test_clip_2",
            camera_name="front_camera",
            video_path=str(self.video_file),
            world_video_path=str(self.world_video_file),
            config=self.valid_config,
            model_device="cpu",
            verbose="INFO",
        )

        # Mock the run function
        mock_results = {
            "dynamic": {
                "processed_frames": 5,
                "total_video_frames": 100,
                "score_matrix": None,
                "track_ids": {1, 2, 3},
                "processed_frame_ids": [0, 1, 2, 3, 4],
                "mean_score": 0.75,
                "std_score": 0.15,
                "min_score": 0.5,
                "max_score": 0.9,
            }
        }

        mock_run.return_value = mock_results

        # Process both requests
        result1 = await self.service.process(request1)
        result2 = await self.service.process(request2)

        # Verify both got unique output directories named with the secure "objects_" prefix
        self.assertNotEqual(result1.output_dir, result2.output_dir)
        self.assertTrue(Path(result1.output_dir).name.startswith("objects_"))
        self.assertTrue(Path(result2.output_dir).name.startswith("objects_"))
        self.assertIsInstance(result1, ObjectsResult)
        self.assertIsInstance(result2, ObjectsResult)

    @patch("tempfile.mkdtemp")
    @patch("services.objects.single_view.service.api.run_object_processors")
    async def test_tempfile_mkdtemp_called_for_no_output_dir(
        self, mock_run: MagicMock, mock_mkdtemp: MagicMock
    ) -> None:
        """Test that tempfile.mkdtemp is called to create the output directory."""
        mock_mkdtemp.return_value = "/tmp/objects_mock_unique_dir"

        request = ObjectsRequest(
            input_data_path=str(self.input_dir),
            clip_id="test_clip",
            camera_name="front_camera",
            video_path=str(self.video_file),
            world_video_path=str(self.world_video_file),
            config=self.valid_config,
            model_device="cpu",
            verbose="INFO",
        )

        # Mock the run function
        mock_results = {
            "dynamic": {
                "processed_frames": 5,
                "total_video_frames": 100,
                "score_matrix": None,
                "track_ids": {1, 2, 3},
                "processed_frame_ids": [0, 1, 2, 3, 4],
                "mean_score": 0.75,
                "std_score": 0.15,
                "min_score": 0.5,
                "max_score": 0.9,
            }
        }

        mock_run.return_value = mock_results

        # Process the request
        result = await self.service.process(request)

        # Verify tempfile.mkdtemp was called to create a securely-named directory in the
        # system-managed temp location (no predictable, world-writable parent path).
        mock_mkdtemp.assert_called_once_with(prefix="objects_")
        self.assertEqual(result.output_dir, "/tmp/objects_mock_unique_dir")

    async def test_service_no_longer_uses_shared_state(self) -> None:
        """Test that the service no longer relies on shared state for processing."""
        # This test ensures we eliminated the race condition by not using instance variables

        # Verify that there's no self.output_dir instance variable being set during processing
        # (This was the source of the race condition)
        self.assertFalse(hasattr(self.service, "output_dir"))

        # Verify that there's no self.default_output_dir anymore either
        self.assertFalse(hasattr(self.service, "default_output_dir"))

    async def test_validation_invalid_device(self) -> None:
        """Test validation with invalid model device."""
        # Pydantic validation should fail during object creation
        with self.assertRaises(ValidationError):
            ObjectsRequest(
                input_data_path=str(self.input_dir),
                clip_id="test_clip",
                camera_name="front_camera",
                video_path=str(self.video_file),
                world_video_path=str(self.world_video_file),
                config=self.valid_config,
                model_device="invalid_device",  # Invalid device
                verbose="INFO",
            )

    def test_validation_invalid_verbose_level(self) -> None:
        """Test validation with invalid verbose level."""
        with self.assertRaises(ValidationError) as context:
            ObjectsRequest(
                input_data_path=str(self.input_dir),
                clip_id="test_clip",
                camera_name="front_camera",
                video_path=str(self.video_file),
                world_video_path=str(self.world_video_file),
                config=self.valid_config,
                model_device="cpu",
                verbose="INVALID_LEVEL",  # Invalid verbose level
            )

        # Pydantic will raise a validation error for invalid literal values
        self.assertIn("Input should be", str(context.exception))

    async def test_validation_verbose_valid_values(self) -> None:
        """Test that verbose level validation accepts valid uppercase values."""
        for verbose_level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            request = ObjectsRequest(
                input_data_path=str(self.input_dir),
                clip_id="test_clip",
                camera_name="front_camera",
                video_path=str(self.video_file),
                world_video_path=str(self.world_video_file),
                config=self.valid_config,
                model_device="cpu",
                verbose=verbose_level,
            )

            is_valid = await self.service.validate_input(request)
            self.assertTrue(is_valid)
            self.assertEqual(request.verbose, verbose_level)

    async def test_validation_negative_trial_frames(self) -> None:
        """Test validation with negative trial_frames."""
        invalid_request = ObjectsRequest(
            input_data_path=str(self.input_dir),
            clip_id="test_clip",
            camera_name="front_camera",
            video_path=str(self.video_file),
            world_video_path=str(self.world_video_file),
            config=self.valid_config,
            model_device="cpu",
            verbose="INFO",
            trial_frames=-5,  # Negative value
        )

        with self.assertRaises(ValueError) as context:
            await self.service.validate_input(invalid_request)

        self.assertIn("trial_frames must be positive", str(context.exception))

    async def test_validation_zero_trial_frames(self) -> None:
        """Test validation with zero trial_frames."""
        invalid_request = ObjectsRequest(
            input_data_path=str(self.input_dir),
            clip_id="test_clip",
            camera_name="front_camera",
            video_path=str(self.video_file),
            world_video_path=str(self.world_video_file),
            config=self.valid_config,
            model_device="cpu",
            verbose="INFO",
            trial_frames=0,  # Zero value
        )

        with self.assertRaises(ValueError) as context:
            await self.service.validate_input(invalid_request)

        self.assertIn("trial_frames must be positive", str(context.exception))

    def test_validation_non_dict_config(self) -> None:
        """Test validation with non-dictionary config fails at Pydantic level."""
        # Test that Pydantic prevents creation of request with non-dict config
        with self.assertRaises(Exception) as context:
            ObjectsRequest(
                input_data_path=str(self.input_dir),
                clip_id="test_clip",
                camera_name="front_camera",
                video_path=str(self.video_file),
                world_video_path=str(self.world_video_file),
                config="not_a_dict",  # String instead of dict
                model_device="cpu",
                verbose="INFO",
            )

        # Should be a Pydantic validation error
        self.assertIn("Input should be a valid dictionary", str(context.exception))

    def test_service_initialization(self) -> None:
        """Test service initialization."""
        service = ObjectsService()
        self.assertIsNotNone(service.logger)

    @patch("services.objects.single_view.service.api.run_object_processors")
    async def test_process_error_handling(self, mock_run: MagicMock) -> None:
        """Test that process method handles processor errors gracefully."""
        request = ObjectsRequest(
            input_data_path=str(self.input_dir),
            clip_id="test_clip",
            camera_name="front_camera",
            video_path=str(self.video_file),
            world_video_path=str(self.world_video_file),
            config=self.valid_config,
            model_device="cpu",
            verbose="INFO",
        )

        # Mock the run function to raise an exception
        mock_run.side_effect = RuntimeError("Processing failed")

        with self.assertRaises(RuntimeError) as context:
            await self.service.process(request)

        self.assertIn("Processing failed", str(context.exception))

    async def test_cleanup_nonexistent_directory(self) -> None:
        """Test cleanup with nonexistent directory."""
        with patch("pathlib.Path.exists", return_value=False):
            nonexistent_dir = Path("/tmp/nonexistent_dir")
            # Should not raise any exception
            await self.service.cleanup(nonexistent_dir)

    @patch("services.objects.single_view.service.api.run_object_processors")
    async def test_process_cancellation_sets_event_and_drains_worker(self, mock_run: MagicMock) -> None:
        """Cancelling ``process`` must set ``cancel_event`` and wait for the executor thread to unwind.

        This is the zombie-thread guard: cancelling the coroutine (client disconnect / arbitrator read-timeout)
        does not by itself stop the ``run_in_executor`` thread, so ``process`` flips the cooperative flag and drains
        the worker before re-raising ``CancelledError`` — guaranteeing the pipeline is no longer touching files by
        the time the caller deletes the temp dir.
        """
        request = ObjectsRequest(
            input_data_path=str(self.input_dir),
            clip_id="test_clip",
            camera_name="front_camera",
            video_path=str(self.video_file),
            world_video_path=str(self.world_video_file),
            config=self.valid_config,
            model_device="cpu",
            verbose="INFO",
        )

        started = threading.Event()
        drained = threading.Event()

        def fake_run(**kwargs: object) -> dict:
            cancel_event = kwargs["cancel_event"]
            assert isinstance(cancel_event, threading.Event)
            started.set()
            # Block like a hot per-frame loop would, unwinding only once cancellation is signalled.
            if not cancel_event.wait(timeout=5.0):
                raise AssertionError("cancel_event was never set by process()")
            drained.set()
            return {
                "dynamic": {
                    "processed_frames": 0,
                    "total_video_frames": 0,
                    "track_ids": set(),
                    "processed_frame_ids": [],
                }
            }

        mock_run.side_effect = fake_run

        cancel_event = threading.Event()
        task = asyncio.create_task(self.service.process(request, cancel_event=cancel_event))

        # Wait until the worker thread has actually started before cancelling.
        for _ in range(500):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        self.assertTrue(started.is_set(), "worker thread did not start")

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(cancel_event.is_set(), "process() must set cancel_event on cancellation")
        self.assertTrue(drained.wait(timeout=5.0), "worker thread must be drained before cancellation propagates")

    # Tests for _setup_output_dir method
    def test_setup_output_dir_creates_temp_dir(self) -> None:
        """Test _setup_output_dir creates a temp directory under /tmp/objects/."""
        output_dir = self.service._setup_output_dir()

        self.assertTrue(str(output_dir).startswith("/tmp/objects"))
        self.assertTrue(output_dir.exists())
        self.assertTrue(output_dir.is_dir())

        # Clean up
        shutil.rmtree(output_dir)

    @patch("tempfile.mkdtemp")
    def test_setup_output_dir_temp_parent_creation_failure(self, mock_mkdtemp: MagicMock) -> None:
        """Test _setup_output_dir handles failure to create temp directory."""
        mock_mkdtemp.side_effect = OSError("No space left on device")

        with self.assertRaises(OSError):
            self.service._setup_output_dir()

    def test_setup_output_dir_multiple_calls_unique_dirs(self) -> None:
        """Test that multiple calls to _setup_output_dir create unique directories."""
        output_dir1 = self.service._setup_output_dir()
        output_dir2 = self.service._setup_output_dir()
        output_dir3 = self.service._setup_output_dir()

        self.assertNotEqual(str(output_dir1), str(output_dir2))
        self.assertNotEqual(str(output_dir2), str(output_dir3))
        self.assertNotEqual(str(output_dir1), str(output_dir3))

        for output_dir in [output_dir1, output_dir2, output_dir3]:
            self.assertTrue(output_dir.exists())
            self.assertTrue(output_dir.is_dir())
            self.assertTrue(str(output_dir).startswith("/tmp/objects"))

        # Clean up
        for output_dir in [output_dir1, output_dir2, output_dir3]:
            shutil.rmtree(output_dir)

    @patch("tempfile.mkdtemp")
    def test_setup_output_dir_mkdtemp_failure(self, mock_mkdtemp: MagicMock) -> None:
        """Test _setup_output_dir handles tempfile.mkdtemp failure."""
        mock_mkdtemp.side_effect = OSError("Failed to create temporary directory")

        with self.assertRaises(OSError):
            self.service._setup_output_dir()

    @patch("services.objects.single_view.service.api.run_object_processors")
    async def test_default_config_used_when_config_is_none(self, mock_run: MagicMock) -> None:
        """Test that the default config is used when request's config is None."""
        request = ObjectsRequest(
            input_data_path=str(self.input_dir),
            clip_id="test_clip",
            camera_name="front_camera",
            video_path=str(self.video_file),
            world_video_path=str(self.world_video_file),
            config=None,  # Explicitly None
            model_device="cpu",
            verbose="INFO",
        )

        mock_results = {
            "dynamic": {
                "processed_frames": 5,
                "total_video_frames": 100,
                "score_matrix": None,
                "track_ids": {1, 2, 3},
                "processed_frame_ids": [0, 1, 2, 3, 4],
                "mean_score": 0.75,
                "std_score": 0.15,
                "min_score": 0.5,
                "max_score": 0.9,
            }
        }
        mock_run.return_value = mock_results

        result = await self.service.process(request)

        # Verify that run_object_processors was called with default config
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        expected_default_config = ObjectProcessorBase.get_default_config()
        self.assertEqual(call_kwargs["config"], expected_default_config)
        # The result reports the effective config actually used (the resolved defaults), not the raw None request.
        self.assertEqual(result.config_summary, expected_default_config)

    @patch("services.objects.single_view.service.api.run_object_processors")
    async def test_default_config_used_when_config_is_empty(self, mock_run: MagicMock) -> None:
        """Test that the default config is used when request's config is empty dict."""
        request = ObjectsRequest(
            input_data_path=str(self.input_dir),
            clip_id="test_clip",
            camera_name="front_camera",
            video_path=str(self.video_file),
            world_video_path=str(self.world_video_file),
            config={},  # Empty dict
            model_device="cpu",
            verbose="INFO",
        )

        mock_results = {
            "dynamic": {
                "processed_frames": 5,
                "total_video_frames": 100,
                "score_matrix": None,
                "track_ids": {1, 2, 3},
                "processed_frame_ids": [0, 1, 2, 3, 4],
                "mean_score": 0.75,
                "std_score": 0.15,
                "min_score": 0.5,
                "max_score": 0.9,
            }
        }
        mock_run.return_value = mock_results

        result = await self.service.process(request)

        # Verify that run_object_processors was called with default config
        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        expected_default_config = ObjectProcessorBase.get_default_config()
        self.assertEqual(call_kwargs["config"], expected_default_config)
        # An empty {} request is treated identically to None: the result reports the resolved defaults, not {}.
        self.assertEqual(result.config_summary, expected_default_config)

    @patch("services.objects.single_view.service.api.run_object_processors")
    async def test_request_config_used_verbatim_not_merged(self, mock_run: MagicMock) -> None:
        """A provided request config is the FULL config for the run — passed through as-is, NOT merged over the
        image defaults (the design requires a complete config so each run is self-describing)."""
        request_config = {"av.objects_single_view": {"overlap_check": {"traffic_light": {"method": "ratio"}}}}
        request = ObjectsRequest(
            input_data_path=str(self.input_dir),
            clip_id="test_clip",
            camera_name="front_camera",
            video_path=str(self.video_file),
            world_video_path=str(self.world_video_file),
            config=request_config,
            model_device="cpu",
            verbose="INFO",
        )
        mock_run.return_value = {
            "dynamic": {
                "processed_frames": 5,
                "total_video_frames": 100,
                "score_matrix": None,
                "track_ids": {1, 2, 3},
                "processed_frame_ids": [0, 1, 2, 3, 4],
                "mean_score": 0.75,
                "std_score": 0.15,
                "min_score": 0.5,
                "max_score": 0.9,
            }
        }

        result = await self.service.process(request)

        call_kwargs = mock_run.call_args[1]
        # Used exactly as sent — no default keys (e.g. debug_dump / hallucination_detector) merged in.
        self.assertEqual(call_kwargs["config"], request_config)
        self.assertNotIn("hallucination_detector", call_kwargs["config"]["av.objects_single_view"])
        # A non-empty config is reported verbatim in the result (it is the full config used for the run).
        self.assertEqual(result.config_summary, request_config)


class TestObjectsCloudRequestService(unittest.IsolatedAsyncioTestCase):
    """Test cases for ObjectsCloudRequest with ObjectsService."""

    def setUp(self) -> None:
        """Set up test fixtures for S3 request tests."""
        self.service = ObjectsService()
        self.valid_config = {
            "av.objects_single_view": {
                "overlap_check": {"vehicle": {"method": "ratio"}},
                "importance_filter": {
                    "distance_threshold_m": 50.0,
                    "oncoming_objects": False,
                    "relevant_lanes": ["ego"],
                },
            }
        }

    async def test_s3_request_validation_success(self) -> None:
        """Test that S3 request validation succeeds with valid data."""
        s3_request = ObjectsCloudRequest(
            rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip?presigned",
            augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4?presigned",
            world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4?presigned",
            camera_name="front_camera",
            config=self.valid_config,
            verbose="INFO",
        )

        # Validate that all fields are set correctly
        self.assertEqual(s3_request.rds_hq_url, "https://s3.amazonaws.com/bucket/rds_hq.zip?presigned")
        self.assertEqual(s3_request.augmented_video_url, "https://s3.amazonaws.com/bucket/video.mp4?presigned")
        self.assertEqual(s3_request.camera_name, "front_camera")
        self.assertEqual(s3_request.verbose, "INFO")
        self.assertEqual(s3_request.config, self.valid_config)

    async def test_s3_request_output_storage_prefix_default(self) -> None:
        """Test that S3 request output_storage_prefix defaults to None."""
        s3_request = ObjectsCloudRequest(
            rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip?presigned",
            augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4?presigned",
            world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4?presigned",
            camera_name="front_camera",
            config=self.valid_config,
            verbose="INFO",
        )

        self.assertIsNone(s3_request.output_storage_prefix)

    async def test_s3_request_missing_required_fields(self) -> None:
        """Test that S3 request validation fails with missing required fields."""
        # Test missing rds_hq_url
        with self.assertRaises(Exception) as context:
            ObjectsCloudRequest(  # type: ignore[call-arg]  # deliberately omits required rds_hq_url
                augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4",
                world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4",
                camera_name="front_camera",
                config=self.valid_config,
            )
        self.assertIn("rds_hq_url", str(context.exception))

        # Test missing augmented_video_url
        with self.assertRaises(Exception) as context:
            ObjectsCloudRequest(  # type: ignore[call-arg]  # deliberately omits required augmented_video_url
                rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip",
                world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4",
                camera_name="front_camera",
                config=self.valid_config,
            )
        self.assertIn("augmented_video_url", str(context.exception))

        # world_model_video_url is optional (omitted in camera-only mode): a request without it is valid.
        ObjectsCloudRequest(
            rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip",
            augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4",
            camera_name="front_camera",
            camera_only=True,
            config=self.valid_config,
        )

        # Test missing camera_name
        with self.assertRaises(Exception) as context:
            ObjectsCloudRequest(  # type: ignore[call-arg]  # deliberately omits required camera_name
                rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip",
                augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4",
                world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4",
                config=self.valid_config,
            )
        self.assertIn("camera_name", str(context.exception))

    async def test_s3_request_optional_fields(self) -> None:
        """Test that S3 request handles optional fields correctly."""
        s3_request = ObjectsCloudRequest(
            rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip",
            augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4",
            world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4",
            camera_name="front_camera",
            config=self.valid_config,
            verbose="DEBUG",
            trial_frames=100,
            output_storage_prefix="/custom/output/path",
        )

        self.assertEqual(s3_request.verbose, "DEBUG")
        self.assertEqual(s3_request.trial_frames, 100)
        self.assertEqual(s3_request.output_storage_prefix, "/custom/output/path")

    async def test_s3_request_invalid_verbose_level(self) -> None:
        """Test S3 request validation with invalid verbose level."""
        with self.assertRaises(Exception) as context:
            ObjectsCloudRequest(
                rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip",
                augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4",
                world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4",
                camera_name="front_camera",
                config=self.valid_config,
                verbose="INVALID_LEVEL",
            )
        # Should contain validation error about invalid choice
        self.assertIn("INVALID_LEVEL", str(context.exception))

    async def test_s3_request_negative_trial_frames(self) -> None:
        """Test S3 request with negative trial_frames (should be allowed)."""
        # Based on the regular request tests, negative trial_frames appear to be allowed
        s3_request = ObjectsCloudRequest(
            rds_hq_url="https://s3.amazonaws.com/bucket/rds_hq.zip",
            augmented_video_url="https://s3.amazonaws.com/bucket/video.mp4",
            world_model_video_url="https://s3.amazonaws.com/bucket/world_video.mp4",
            camera_name="front_camera",
            config=self.valid_config,
            trial_frames=-5,
        )

        self.assertEqual(s3_request.trial_frames, -5)


class TestObjectsResult(unittest.TestCase):
    """Test cases for ObjectsResult model."""

    def test_nan_and_inf_serialization_to_null_in_json_mode(self) -> None:
        """Test that the custom field serializer converts NaN and infinity values to the string 'null' when
        model_dump(mode='json') is called."""

        # Create a result object with NaN and infinity values in all float fields
        result = ObjectsResult(
            processed_frames=10,
            total_video_frames=100,
            mean_score=float("nan"),  # NaN value
            std_score=float("inf"),  # Positive infinity
            min_score=float("-inf"),  # Negative infinity
            max_score=float("nan"),  # Another NaN value
            unique_track_ids=[1, 2, 3],
            processed_frame_ids=[10, 20, 30],
            config_summary={"test": "config"},
            clip_id="test_clip_123",
            output_dir="/tmp/test_output",
        )

        # Test that model_dump(mode="json") converts NaN and infinity to null
        json_data = result.model_dump(mode="json")

        # Verify that all NaN and infinity values are converted to null
        self.assertIsNone(json_data["mean_score"], "mean_score should be null when NaN is serialized")
        self.assertIsNone(json_data["std_score"], "std_score should be null when inf is serialized")
        self.assertIsNone(json_data["min_score"], "min_score should be null when -inf is serialized")
        self.assertIsNone(json_data["max_score"], "max_score should be null when NaN is serialized")

        # Verify that other fields remain unchanged
        self.assertEqual(json_data["processed_frames"], 10)
        self.assertEqual(json_data["total_video_frames"], 100)
        self.assertEqual(json_data["unique_track_ids"], [1, 2, 3])
        self.assertEqual(json_data["processed_frame_ids"], [10, 20, 30])
        self.assertEqual(json_data["config_summary"], {"test": "config"})
        self.assertEqual(json_data["clip_id"], "test_clip_123")
        self.assertEqual(json_data["output_dir"], "/tmp/test_output")

    def test_normal_float_values_unchanged_in_json_mode(self) -> None:
        """Test that normal float values are not affected by the custom field serializer."""
        result = ObjectsResult(
            processed_frames=10,
            total_video_frames=100,
            mean_score=0.75,  # Normal float
            std_score=0.123,  # Normal float
            min_score=0.0,  # Zero
            max_score=1.0,  # Normal float
            unique_track_ids=[1, 2, 3],
            processed_frame_ids=[10, 20, 30],
            config_summary={"test": "config"},
            clip_id="test_clip_789",
            output_dir="/tmp/test_output",
        )

        # Test that model_dump(mode="json") preserves normal values
        json_data = result.model_dump(mode="json")

        # Verify that normal values remain unchanged
        self.assertEqual(json_data["mean_score"], 0.75)
        self.assertEqual(json_data["std_score"], 0.123)
        self.assertEqual(json_data["min_score"], 0.0)
        self.assertEqual(json_data["max_score"], 1.0)

    def test_field_serializer_only_applies_in_json_mode(self) -> None:
        """Test that the field serializer only applies when mode='json', not in regular model_dump()."""
        result = ObjectsResult(
            processed_frames=10,
            total_video_frames=100,
            mean_score=float("nan"),  # NaN value
            std_score=float("inf"),  # Positive infinity
            min_score=float("-inf"),  # Negative infinity
            max_score=float("nan"),  # Another NaN value
            unique_track_ids=[1, 2, 3],
            processed_frame_ids=[10, 20, 30],
            config_summary={"test": "config"},
            clip_id="test_clip_456",
            output_dir="/tmp/test_output",
        )

        # Test that regular model_dump() preserves NaN and infinity values
        regular_data = result.model_dump()

        # Verify that NaN and infinity values are preserved in regular mode
        self.assertTrue(math.isnan(regular_data["mean_score"]), "mean_score should remain NaN in regular mode")
        self.assertTrue(math.isinf(regular_data["std_score"]), "std_score should remain inf in regular mode")
        self.assertTrue(math.isinf(regular_data["min_score"]), "min_score should remain -inf in regular mode")
        self.assertTrue(math.isnan(regular_data["max_score"]), "max_score should remain NaN in regular mode")


if __name__ == "__main__":
    unittest.main()
