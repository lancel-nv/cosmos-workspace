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

"""Unit tests for segformer module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torchvision.transforms as T

from checks.utils.segformer import (
    create_cityscapes_label_colormap,
    get_dataloader,
    get_masks,
    get_model_input_shape,
    get_normalization_constants,
    get_transforms_fn,
    reached_end_of_segment,
    setup_model,
)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def test_get_model_input_shape(self) -> None:
        """Test get_model_input_shape function."""
        shape = get_model_input_shape()
        expected_shape = (1, 3, 1024, 1820)
        self.assertEqual(shape, expected_shape)

    @patch("checks.utils.segformer.get_inference_session")
    def test_setup_model(self, mock_get_inference_session: MagicMock) -> None:
        """Test setup_model function."""
        # Mock inference session
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_output = MagicMock()
        mock_output.name = "output"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]

        mock_io_binding = MagicMock()
        mock_session.io_binding.return_value = mock_io_binding

        mock_get_inference_session.return_value = mock_session

        with patch("checks.utils.onnx.configure_onnx_logging"):
            sess, io_binding = setup_model(verbose=False)

        self.assertEqual(sess, mock_session)
        self.assertEqual(io_binding, mock_io_binding)

        # Verify warm-up run was performed
        mock_io_binding.bind_cpu_input.assert_called_once()
        mock_io_binding.bind_output.assert_called_once_with("output", device_type="cuda")
        mock_session.run_with_iobinding.assert_called_once_with(mock_io_binding)
        mock_io_binding.copy_outputs_to_cpu.assert_called_once()

    def test_create_cityscapes_label_colormap(self) -> None:
        """Test create_cityscapes_label_colormap function."""
        colormap, class_idx2label = create_cityscapes_label_colormap()

        # Check that we have the expected number of classes
        self.assertEqual(len(colormap), 19)
        self.assertEqual(len(class_idx2label), 19)

        # Check some specific mappings
        self.assertIn("road", class_idx2label)
        self.assertIn("car", class_idx2label)
        self.assertIn("person", class_idx2label)

        # Check that all colors are RGB tuples
        for color in colormap.values():
            self.assertEqual(len(color), 3)
            for channel in color:
                self.assertIsInstance(channel, int)
                self.assertGreaterEqual(channel, 0)
                self.assertLessEqual(channel, 255)

    def test_get_normalization_constants(self) -> None:
        """Test get_normalization_constants function."""
        mean, std = get_normalization_constants()

        expected_mean = [123.675, 116.28, 103.53]
        expected_std = [58.395, 57.12, 57.375]

        self.assertEqual(mean, expected_mean)
        self.assertEqual(std, expected_std)

    def test_get_transforms_fn(self) -> None:
        """Test get_transforms_fn function."""
        transforms = get_transforms_fn()

        self.assertIsInstance(transforms, T.Compose)
        # Should have 2 transforms: Resize and Normalize
        self.assertEqual(len(transforms.transforms), 2)

    @patch("checks.utils.segformer.DataLoader")
    @patch("checks.utils.segformer.VideoDataset")
    def test_get_dataloader(self, mock_video_dataset: MagicMock, mock_dataloader: MagicMock) -> None:
        """Test get_dataloader function."""
        video_path = "/path/to/video.mp4"
        start_frame = 10

        mock_dataset = MagicMock()
        mock_video_dataset.return_value = mock_dataset

        mock_loader = MagicMock()
        mock_dataloader.return_value = mock_loader

        dataloader, video_dataset = get_dataloader(video_path, start_frame)

        # Check that VideoDataset was called with correct parameters
        mock_video_dataset.assert_called_once()
        call_args = mock_video_dataset.call_args
        self.assertEqual(call_args[0][0], video_path)
        self.assertEqual(call_args[1]["start_frame"], start_frame)

        # Check that DataLoader was called with correct parameters
        mock_dataloader.assert_called_once_with(mock_dataset, batch_size=1, shuffle=False, num_workers=0)

        self.assertEqual(dataloader, mock_loader)
        self.assertEqual(video_dataset, mock_dataset)

    def test_get_masks(self) -> None:
        """Test get_masks function."""
        # Create sample mask prediction
        mask_prediction = np.array([[[0, 1, 2], [3, 4, 5]]])  # 1x2x3
        image_shape = (3, 2, 3)  # C, H, W

        result = get_masks(mask_prediction, image_shape)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, image_shape)

    def test_reached_end_of_segment(self) -> None:
        """Test reached_end_of_segment function."""
        # Test cases: (current_frame, end_frame, expected_result)
        test_cases = [
            (5, 10, False),  # Not reached end
            (10, 10, False),  # At end frame
            (15, 10, True),  # Past end frame
            (5, -1, False),  # No end frame specified
            (0, 0, False),  # At frame 0 with end frame 0
        ]

        for current_frame, end_frame, expected in test_cases:
            result = reached_end_of_segment(current_frame, end_frame)
            self.assertEqual(result, expected, f"Failed for current_frame={current_frame}, end_frame={end_frame}")

    def test_reached_end_of_segment_invalid_inputs(self) -> None:
        """Test reached_end_of_segment with invalid inputs."""
        with self.assertRaises(AssertionError):
            reached_end_of_segment(-1, 10)  # Negative current frame

        with self.assertRaises(AssertionError):
            reached_end_of_segment(5, -2)  # Invalid end frame


if __name__ == "__main__":
    unittest.main()
