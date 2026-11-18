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

"""Unit tests for seg_helper module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from checks.objects.single_view.seg_helper import SegHelper


class TestSegHelper(unittest.TestCase):
    """Test cases for SegHelper class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Mock the model setup to avoid loading actual ONNX model
        with (
            patch("checks.objects.single_view.seg_helper.setup_model") as mock_setup,
            patch("checks.objects.single_view.seg_helper.create_cityscapes_label_colormap") as mock_colormap,
        ):
            # Mock setup_model return values
            mock_session = MagicMock()
            mock_io_binding = MagicMock()
            mock_setup.return_value = (mock_session, mock_io_binding)

            # Mock input/output names
            mock_input = MagicMock()
            mock_input.name = "input_tensor"
            mock_output = MagicMock()
            mock_output.name = "output_tensor"
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]

            # Mock colormap
            mock_colormap.return_value = (
                {0: (128, 64, 128), 13: (0, 0, 142)},  # road and car colors
                {"road": 0, "car": 13},
            )

            self.seg_helper = SegHelper("cuda")

    def test_init(self) -> None:
        """Test SegHelper initialization."""
        self.assertEqual(self.seg_helper.model_device, "cuda")
        self.assertIsNotNone(self.seg_helper.ort_session)
        self.assertIsNotNone(self.seg_helper.io_binding)
        self.assertIsNotNone(self.seg_helper.colormap)
        self.assertIsNotNone(self.seg_helper.class_label_to_idx)

    def test_init_cpu_device(self) -> None:
        """Test SegHelper initialization with CPU device."""
        with (
            patch("checks.objects.single_view.seg_helper.setup_model") as mock_setup,
            patch("checks.objects.single_view.seg_helper.create_cityscapes_label_colormap") as mock_colormap,
        ):
            mock_session = MagicMock()
            mock_io_binding = MagicMock()
            mock_setup.return_value = (mock_session, mock_io_binding)

            mock_input = MagicMock()
            mock_input.name = "input"
            mock_output = MagicMock()
            mock_output.name = "output"
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]

            mock_colormap.return_value = ({}, {})

            seg_helper_cpu = SegHelper("cpu")

            self.assertEqual(seg_helper_cpu.model_device, "cpu")

    def test_get_class_color_valid_class(self) -> None:
        """Test getting color for a valid class."""
        color = self.seg_helper.get_class_color("car")
        expected_color = np.array((0, 0, 142), dtype=np.uint8)
        self.assertTrue(np.array_equal(color, expected_color))

    def test_get_class_color_invalid_class(self) -> None:
        """Test getting color for an invalid class."""
        with self.assertRaises(ValueError):
            self.seg_helper.get_class_color("nonexistent_class")

    @patch("checks.objects.single_view.seg_helper.get_masks")
    def test_process_frame(self, mock_get_masks: MagicMock) -> None:
        """Test processing a single frame."""
        # Create mock frame
        frame = torch.zeros(3, 480, 640)  # C, H, W format

        # Mock ONNX inference outputs
        mock_output = [np.random.randint(0, 19, (1, 480, 640))]
        self.seg_helper.io_binding.copy_outputs_to_cpu.return_value = mock_output

        # Mock get_masks return value
        mock_masks = torch.zeros(3, 480, 640)
        mock_get_masks.return_value = mock_masks

        result = self.seg_helper.process_frame(frame)

        # Check that inference was called
        self.seg_helper.io_binding.bind_cpu_input.assert_called_once()
        self.seg_helper.ort_session.run_with_iobinding.assert_called_once()
        self.seg_helper.io_binding.copy_outputs_to_cpu.assert_called_once()

        # Check that get_masks was called with correct parameters
        mock_get_masks.assert_called_once_with(mock_output[0], frame.shape[1:])

        self.assertEqual(result.shape, mock_masks.shape)

    def test_get_class_mask_single_class(self) -> None:
        """Test extracting mask for a single high-level class (vehicle contains car)."""
        # Create mock segmentation masks with known colors
        seg_masks = torch.zeros(3, 100, 100)  # RGB format

        # Set some pixels to car color (0, 0, 142)
        seg_masks[2, 10:20, 10:20] = 142  # Blue channel

        # Mock get_class_color to return car color (used for vehicle mapping)
        with patch.object(self.seg_helper, "get_class_color", return_value=(0, 0, 142)):
            mask = self.seg_helper.get_class_mask(seg_masks, "vehicle")

        # Check that mask has correct shape and type
        self.assertEqual(mask.shape, (100, 100))
        self.assertEqual(mask.dtype, bool)

        # Check that the car pixels are True
        self.assertTrue(np.any(mask[10:20, 10:20]))

    def test_get_class_mask_vehicle_group(self) -> None:
        """Test extracting mask for vehicle group."""
        # Create mock segmentation masks
        seg_masks = torch.zeros(3, 100, 100)

        # Set pixels for different vehicle types
        seg_masks[2, 10:20, 10:20] = 142  # Car color (blue channel)
        seg_masks[1, 30:40, 30:40] = 128  # Truck color (green channel)

        # Mock get_class_color for different vehicle classes
        def mock_get_color(class_name: str) -> tuple[int, int, int]:
            colors = {"car": (0, 0, 142), "truck": (0, 128, 0), "bus": (0, 60, 100)}
            if class_name in colors:
                return colors[class_name]
            raise ValueError(f"Unknown class: {class_name}")

        with patch.object(self.seg_helper, "get_class_color", side_effect=mock_get_color):
            mask = self.seg_helper.get_class_mask(seg_masks, "vehicle")

        # Check that both car and truck pixels are included
        self.assertTrue(np.any(mask[10:20, 10:20]))  # Car pixels
        self.assertTrue(np.any(mask[30:40, 30:40]))  # Truck pixels

    def test_get_class_mask_unknown_class(self) -> None:
        """Test extracting mask for unknown class."""
        seg_masks = torch.zeros(3, 100, 100)

        # Mock get_class_color to raise ValueError for unknown classes
        with patch.object(self.seg_helper, "get_class_color", side_effect=ValueError):
            mask = self.seg_helper.get_class_mask(seg_masks, "unknown_class")

        # Should return all-zeros mask
        self.assertEqual(mask.shape, (100, 100))
        self.assertFalse(np.any(mask))

    def test_resize_masks(self) -> None:
        """Test resizing segmentation masks."""
        # Create input masks
        input_masks = torch.zeros(3, 100, 100)
        target_height, target_width = 200, 300

        with patch("torch.nn.functional.interpolate") as mock_interpolate:
            # Mock interpolate to return correct shape
            mock_result = torch.zeros(1, 3, 200, 300)
            mock_interpolate.return_value = mock_result

            result = self.seg_helper.resize_masks(input_masks, target_height, target_width)

            # Check that interpolate was called with correct parameters
            mock_interpolate.assert_called_once()
            call_args = mock_interpolate.call_args
            self.assertEqual(call_args[1]["size"], (target_height, target_width))
            self.assertEqual(call_args[1]["mode"], "nearest")

            # Check result shape after squeeze
            self.assertEqual(result.shape, (3, 200, 300))

    # get_video_dataloader is no longer part of SegHelper; handled via processor

    def test_class_to_segmentation_mapping(self) -> None:
        """Test the class to segmentation classes mapping."""
        # Test that the mapping is correctly defined
        expected_mappings = {
            "vehicle": ["car", "truck", "bus"],
            "pedestrian": ["person"],
            "motorcycle": ["motorcycle", "rider"],
            "bicycle": ["bicycle", "rider"],
        }

        # This is tested indirectly through get_class_mask
        seg_masks = torch.zeros(3, 100, 100)

        with patch.object(self.seg_helper, "get_class_color") as mock_get_color:
            mock_get_color.side_effect = ValueError  # Simulate unknown classes

            # Test each mapping
            for class_name, expected_classes in expected_mappings.items():
                self.seg_helper.get_class_mask(seg_masks, class_name)

                # Should call get_class_color for each expected segmentation class
                expected_calls = len(expected_classes)
                actual_calls = mock_get_color.call_count

                # Check that the correct number of calls were made
                self.assertEqual(actual_calls, expected_calls)

                # Reset mock for next iteration
                mock_get_color.reset_mock()

    def test_setup_model_calls(self) -> None:
        """Test that setup_model is called with correct parameters."""
        with (
            patch("checks.objects.single_view.seg_helper.setup_model") as mock_setup,
            patch("checks.objects.single_view.seg_helper.create_cityscapes_label_colormap") as mock_colormap,
        ):
            mock_session = MagicMock()
            mock_io_binding = MagicMock()
            mock_setup.return_value = (mock_session, mock_io_binding)

            mock_input = MagicMock()
            mock_input.name = "input"
            mock_output = MagicMock()
            mock_output.name = "output"
            mock_session.get_inputs.return_value = [mock_input]
            mock_session.get_outputs.return_value = [mock_output]

            mock_colormap.return_value = ({}, {})

            SegHelper("cuda")

            # Check that setup_model was called with verbose=False
            mock_setup.assert_called_once_with(model_device="cuda", verbose=False)


class TestSegHelperWithSuppliedSession(unittest.TestCase):
    """Reusing one ONNX session across jobs, which is how a replica avoids re-initializing CUDA per request.

    Building a session initializes the CUDA execution provider, allocating a cuBLAS handle and a device arena. Doing
    that once per job meant a pod claiming a batch ran several of them at once, which is how a healthy replica came
    to fail ``cublasCreate``. These tests pin the two halves of the arrangement that make sharing correct.
    """

    @staticmethod
    def _session() -> MagicMock:
        session = MagicMock()
        session.get_inputs.return_value = [MagicMock(name="input")]
        session.get_outputs.return_value = [MagicMock(name="output")]
        # A fresh binding per call, as onnxruntime does, so the test can tell sharing from copying.
        session.io_binding.side_effect = lambda: MagicMock()
        return session

    def test_a_supplied_session_is_used_instead_of_building_one(self) -> None:
        """No session built means no CUDA context established, which is the entire point of passing one in."""
        session = self._session()

        with (
            patch("checks.objects.single_view.seg_helper.setup_model") as mock_setup,
            patch("checks.objects.single_view.seg_helper.create_cityscapes_label_colormap", return_value=({}, {})),
        ):
            helper = SegHelper("cuda", session=session)

        mock_setup.assert_not_called()
        self.assertIs(helper.ort_session, session)

    def test_each_helper_gets_its_own_binding(self) -> None:
        """The session is shareable and the binding is not, and this is the line between them.

        ``process_frame`` binds its inputs and then reads the outputs off the binding. Two jobs sharing one would
        overwrite each other's inputs between the bind and the read, and the failure mode is not a crash -- it is
        each job silently returning masks computed from the other's frame.
        """
        session = self._session()

        with (
            patch("checks.objects.single_view.seg_helper.setup_model"),
            patch("checks.objects.single_view.seg_helper.create_cityscapes_label_colormap", return_value=({}, {})),
        ):
            first = SegHelper("cuda", session=session)
            second = SegHelper("cuda", session=session)

        self.assertIs(first.ort_session, second.ort_session)
        self.assertIsNot(first.io_binding, second.io_binding)

    def test_no_session_supplied_still_builds_one(self) -> None:
        """The CLI path, which owns its session outright because it runs one clip in a process that then exits."""
        with (
            patch("checks.objects.single_view.seg_helper.setup_model") as mock_setup,
            patch("checks.objects.single_view.seg_helper.create_cityscapes_label_colormap", return_value=({}, {})),
        ):
            mock_setup.return_value = (self._session(), MagicMock())
            SegHelper("cuda")

        mock_setup.assert_called_once_with(model_device="cuda", verbose=False)


class TestSegHelperReleaseArenaMemory(unittest.TestCase):
    """Returning the shared session's unused CUDA arena regions to the driver once a job has finished with them."""

    def setUp(self) -> None:
        self.session = MagicMock()
        model_input = MagicMock()
        model_input.name = "input"
        model_output = MagicMock()
        model_output.name = "output"
        self.session.get_inputs.return_value = [model_input]
        self.session.get_outputs.return_value = [model_output]
        self.session.get_provider_options.return_value = {
            "CUDAExecutionProvider": {"device_id": "1"},
            "CPUExecutionProvider": {},
        }
        self.session.io_binding.side_effect = lambda: MagicMock()

        with patch("checks.objects.single_view.seg_helper.create_cityscapes_label_colormap", return_value=({}, {})):
            self.helper = SegHelper("cuda", session=self.session)

    def _run_one_frame(self) -> None:
        self.helper.io_binding.copy_outputs_to_cpu.return_value = [np.zeros((1, 4, 6))]
        with patch("checks.objects.single_view.seg_helper.get_masks"):
            self.helper.process_frame(torch.zeros(3, 4, 6))

    def test_release_runs_one_shrinking_inference_on_the_sessions_cuda_device(self) -> None:
        """The arena shrinks only at the end of a run that asks for it, on the device the CUDA provider is bound to."""
        self._run_one_frame()

        self.helper.release_arena_memory()

        self.session.run.assert_called_once()
        output_names, feed, run_options = self.session.run.call_args.args
        self.assertEqual(output_names, ["output"])
        self.assertEqual(list(feed), ["input"])
        np.testing.assert_array_equal(feed["input"], np.zeros((3, 4, 6), dtype=np.float32))
        self.assertEqual(feed["input"].dtype, np.float32)
        self.assertEqual(run_options.get_run_config_entry("memory.enable_memory_arena_shrinkage"), "gpu:1")

    def test_release_clears_this_helpers_bindings(self) -> None:
        """The last frame's output stays bound on the device until cleared, which would keep its region reserved."""
        self._run_one_frame()
        binding = self.helper.io_binding
        binding.clear_binding_outputs.reset_mock()
        binding.clear_binding_inputs.reset_mock()

        self.helper.release_arena_memory()

        binding.clear_binding_outputs.assert_called_once_with()
        binding.clear_binding_inputs.assert_called_once_with()

    def test_release_without_an_inference_runs_nothing(self) -> None:
        """A job whose masks all came from the cache allocated nothing in the arena, so there is nothing to release."""
        self.helper.release_arena_memory()

        self.session.run.assert_not_called()

    def test_second_release_runs_nothing(self) -> None:
        """The release inference frees its own working memory, so a repeat call without new frames has nothing to do."""
        self._run_one_frame()

        self.helper.release_arena_memory()
        self.helper.release_arena_memory()

        self.session.run.assert_called_once()

    def test_release_on_a_session_without_the_cuda_provider_runs_nothing(self) -> None:
        """A CPU session has no CUDA arena, and naming a GPU arena that does not exist would fail the run."""
        self.session.get_provider_options.return_value = {"CPUExecutionProvider": {}}
        self._run_one_frame()

        self.helper.release_arena_memory()

        self.session.run.assert_not_called()

    def test_failed_release_is_logged_and_not_raised(self) -> None:
        """The release runs on the job's error path too, where raising would replace the job's own exception."""
        self.session.run.side_effect = RuntimeError("CUDA failure 2: out of memory")
        self._run_one_frame()

        with self.assertLogs("checks.objects.single_view.seg_helper", level="WARNING") as logs:
            self.helper.release_arena_memory()

        self.assertIn("Failed to release unused SegFormer arena memory", logs.output[0])


if __name__ == "__main__":
    unittest.main()
