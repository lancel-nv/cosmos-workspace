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

"""Unit tests for visualization_helper module."""

import logging
from pathlib import Path
import shutil
import tempfile
from typing import Any
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from checks.objects.single_view.visualization_helper import VisualizationHelper, hallucination_overlay_label
from checks.utils.rds_geometry import RDSGeometry
from checks.utils.scene_rasterizer import SceneRasterizer


class TestVisualizationHelper(unittest.TestCase):
    """Test cases for VisualizationHelper class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.clip_id = "test_clip_123"
        self.output_dir = self.temp_dir

        self.vis_helper = VisualizationHelper(self.clip_id, self.output_dir, verbose=False)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_init(self) -> None:
        """Test VisualizationHelper initialization."""
        self.assertEqual(self.vis_helper.clip_id, self.clip_id)
        self.assertEqual(self.vis_helper.output_dir, Path(self.output_dir))
        self.assertFalse(self.vis_helper.verbose)
        self.assertIsNone(self.vis_helper.video_writer)
        self.assertIsNone(self.vis_helper.original_video_path)

    def test_init_verbose(self) -> None:
        """Test VisualizationHelper initialization with verbose mode."""
        vis_helper_verbose = VisualizationHelper(self.clip_id, self.output_dir, verbose=True)
        self.assertTrue(vis_helper_verbose.verbose)

    def test_init_creates_output_dir(self) -> None:
        """Test that initialization creates output directory."""
        new_output_dir = Path(self.temp_dir) / "new_subdir"
        self.assertFalse(new_output_dir.exists())

        VisualizationHelper(self.clip_id, str(new_output_dir))

        self.assertTrue(new_output_dir.exists())

    @patch("checks.utils.overlay_video_helper.VideoWriter")
    @patch("checks.utils.overlay_video_helper.get_video_frame_count", return_value=100)
    def test_initialize_video_writer_success(self, _mock_count: MagicMock, mock_video_writer: MagicMock) -> None:
        """Test successful video writer initialization."""
        mock_writer = MagicMock()
        mock_writer.open.return_value = True
        mock_video_writer.return_value = mock_writer

        video_path = "/path/to/test_video.mp4"
        result = self.vis_helper.initialize_video_writer(video_path)

        self.assertTrue(result)
        self.assertEqual(self.vis_helper.frame_width, 1280)
        self.assertEqual(self.vis_helper.frame_height, 720)
        self.assertIsNotNone(self.vis_helper.video_output_path)

    @patch("checks.utils.overlay_video_helper.get_video_frame_count", return_value=0)
    def test_initialize_video_writer_video_open_fails(self, _mock_count: MagicMock) -> None:
        """Test video writer initialization when the source video cannot be probed."""
        video_path = "/path/to/nonexistent_video.mp4"
        result = self.vis_helper.initialize_video_writer(video_path)

        self.assertFalse(result)

    @patch("checks.utils.overlay_video_helper.VideoWriter")
    @patch("checks.utils.overlay_video_helper.get_video_frame_count", return_value=100)
    def test_initialize_video_writer_records_frame_count(
        self, _mock_count: MagicMock, mock_video_writer: MagicMock
    ) -> None:
        """Test that initialization records the source frame count for later use."""
        mock_writer = MagicMock()
        mock_writer.open.return_value = True
        mock_video_writer.return_value = mock_writer

        video_path = "/path/to/test_video.mp4"
        result = self.vis_helper.initialize_video_writer(video_path)

        self.assertTrue(result)
        self.assertEqual(self.vis_helper.get_frame_count(), 100)
        mock_video_writer.assert_called_once()

    @patch("checks.utils.overlay_video_helper.VideoWriter")
    @patch("checks.utils.overlay_video_helper.get_video_frame_count", return_value=100)
    def test_initialize_video_writer_writer_open_fails(
        self, _mock_count: MagicMock, mock_video_writer: MagicMock
    ) -> None:
        """Test video writer initialization when the ffmpeg encoder fails to start."""
        mock_writer = MagicMock()
        mock_writer.open.return_value = False
        mock_video_writer.return_value = mock_writer

        video_path = "/path/to/test_video.mp4"
        result = self.vis_helper.initialize_video_writer(video_path)

        self.assertFalse(result)

    def test_get_original_frame_no_capture(self) -> None:
        """Test getting original frame when no video capture is available."""
        # Set frame dimensions for this test
        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720

        frame = self.vis_helper.get_original_frame(frame_idx=5)

        # Should return gray background
        expected_shape = (720, 1280, 3)  # Default height, width, channels
        self.assertEqual(frame.shape, expected_shape)
        self.assertTrue(np.all(frame == 128))  # Gray background

    def test_get_original_frame_with_capture(self) -> None:
        """Test getting original frame via the ffmpeg-backed frame reader."""
        mock_read = MagicMock(return_value=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
        self.vis_helper.frame_reader = MagicMock(read=mock_read)
        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720

        with patch("cv2.cvtColor") as mock_cvtcolor, patch("cv2.resize") as mock_resize:
            expected_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            mock_cvtcolor.return_value = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            mock_resize.return_value = expected_frame

            frame = self.vis_helper.get_original_frame(frame_idx=5)

            # Verify the returned frame
            np.testing.assert_array_equal(frame, expected_frame)

        # Should read the requested frame index and convert/resize it
        mock_read.assert_called_once_with(5)
        mock_cvtcolor.assert_called_once()
        mock_resize.assert_called_once()

    def test_get_original_frame_read_fails(self) -> None:
        """Test getting original frame when frame read fails."""
        self.vis_helper.frame_reader = MagicMock(read=MagicMock(return_value=None))
        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720

        frame = self.vis_helper.get_original_frame(frame_idx=5)

        # Should return gray background when read fails
        expected_shape = (720, 1280, 3)
        self.assertEqual(frame.shape, expected_shape)
        self.assertTrue(np.all(frame == 128))

    def test_overlay_segmentation(self) -> None:
        """Test overlaying segmentation masks."""
        base_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        seg_masks = MagicMock()
        seg_masks.numpy.return_value = np.random.randint(0, 255, (3, 720, 1280), dtype=np.uint8)

        with patch("cv2.resize") as mock_resize:
            mock_resize.return_value = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

            result = self.vis_helper._overlay_segmentation(base_image, seg_masks)

        self.assertEqual(result.shape, base_image.shape)
        self.assertEqual(result.dtype, np.uint8)

    def test_overlay_segmentation_resize_needed(self) -> None:
        """Test overlaying segmentation when resizing is needed."""
        base_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        seg_masks = MagicMock()
        # Different size than base image
        seg_masks.numpy.return_value = np.random.randint(0, 255, (3, 480, 640), dtype=np.uint8)

        with patch("cv2.resize") as mock_resize:
            mock_resize.return_value = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

            result = self.vis_helper._overlay_segmentation(base_image, seg_masks)

            # Should call resize to match base image dimensions
            mock_resize.assert_called_once_with(unittest.mock.ANY, (1280, 720))

            # Verify the result
            self.assertEqual(result.shape, base_image.shape)
            self.assertEqual(result.dtype, np.uint8)

    def test_add_object_overlays(self) -> None:
        """Test adding object overlays."""
        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_scores = {"123": 0.8}
        camera_pose = np.eye(4)
        camera_model = MagicMock()
        filtered_objects: dict[str, Any] = {}

        # Mock cuboid
        mock_cuboid = MagicMock()
        mock_projected_mask = np.zeros((720, 1280), dtype=bool)
        mock_projected_mask[100:200, 100:200] = True
        mock_depth_mask = np.zeros((720, 1280), dtype=np.float32)
        mock_depth_mask[100:200, 100:200] = 1.0
        # Some area
        mock_cuboid.get_projected_mask.return_value = mock_projected_mask, mock_depth_mask
        frame_objects = {"123": {"geometry": mock_cuboid}}
        with patch.object(self.vis_helper, "_draw_object_overlay") as mock_draw_overlay:
            mock_draw_overlay.return_value = vis_image

            result = self.vis_helper._add_object_overlays(
                vis_image, frame_scores, frame_objects, camera_pose, camera_model, filtered_objects
            )

        # Should create cuboid and draw overlay
        mock_cuboid.get_projected_mask.assert_called_once()
        mock_draw_overlay.assert_called_once()

        # Verify the result is the expected image
        self.assertIs(result, vis_image)

    def test_add_object_overlays_with_scene_rasterizer(self) -> None:
        """Test adding object overlays with pre-computed SceneRasterizer."""
        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_scores = {"1234": 0.8}
        camera_pose = np.eye(4)
        camera_model = MagicMock()
        filtered_objects: dict[str, Any] = {}

        # Mock cuboid — spec=RDSGeometry so SceneRasterizer accepts it as a valid geometry
        # and pre-computes a real visibility mask (otherwise it short-circuits to an empty mask).
        mock_cuboid = MagicMock(spec=RDSGeometry)
        mock_projected_mask = np.zeros((720, 1280), dtype=bool)
        mock_projected_mask[100:200, 100:200] = True
        mock_depth_mask = np.full((720, 1280), np.inf, dtype=np.float32)
        mock_depth_mask[100:200, 100:200] = 10.0
        mock_cuboid.get_projected_mask.return_value = mock_projected_mask, mock_depth_mask
        frame_objects = {"1234": {"geometry": mock_cuboid}}

        # Create a SceneRasterizer (computes visibility masks in constructor)
        scene_rasterizer = SceneRasterizer(
            frame_objects, camera_pose, camera_model, 1280, 720, logger=logging.getLogger("test")
        )

        with patch.object(self.vis_helper, "_draw_object_overlay") as mock_draw_overlay:
            mock_draw_overlay.return_value = vis_image

            result = self.vis_helper._add_object_overlays(
                vis_image,
                frame_scores,
                frame_objects,
                camera_pose,
                camera_model,
                filtered_objects,
                scene_rasterizer=scene_rasterizer,
            )

        # When scene_rasterizer is provided, it should use cached masks (no additional projection calls)
        # The cuboid's get_projected_mask was called during SceneRasterizer construction,
        # not during _add_object_overlays
        mock_draw_overlay.assert_called_once()
        self.assertIs(result, vis_image)

    def test_add_object_overlays_traffic_light_color(self) -> None:
        """Traffic light objects should render with yellow contour color."""
        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_scores = {"1": 0.9}
        camera_pose = np.eye(4)
        camera_model = MagicMock()
        filtered_objects: dict[str, Any] = {}

        mock_cuboid = MagicMock()
        mask = np.zeros((720, 1280), dtype=bool)
        mask[10:20, 10:20] = True
        mock_depth_mask = np.zeros((720, 1280), dtype=np.float32)
        mock_depth_mask[10:20, 10:20] = 1.0
        mock_cuboid.get_projected_mask.return_value = mask, mock_depth_mask
        frame_objects = {
            "1": {
                "geometry": mock_cuboid,
                "object_type": "TrafficLight",
            }
        }

        with patch.object(self.vis_helper, "_draw_object_overlay") as mock_draw_overlay:
            mock_draw_overlay.return_value = vis_image
            _ = self.vis_helper._add_object_overlays(
                vis_image, frame_scores, frame_objects, camera_pose, camera_model, filtered_objects
            )

        # contour_color is the 5th positional argument
        contour_color = mock_draw_overlay.call_args[0][4]
        self.assertEqual(contour_color, (255, 255, 0))  # Yellow

    def test_add_object_overlays_traffic_sign_color(self) -> None:
        """Traffic sign objects should render with orange contour color."""
        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_scores = {"2": 0.9}
        camera_pose = np.eye(4)
        camera_model = MagicMock()
        filtered_objects: dict[str, Any] = {}

        mock_cuboid = MagicMock()
        mask = np.zeros((720, 1280), dtype=bool)
        mask[10:20, 10:20] = True
        mock_depth_mask = np.zeros((720, 1280), dtype=np.float32)
        mock_depth_mask[10:20, 10:20] = 1.0
        mock_cuboid.get_projected_mask.return_value = mask, mock_depth_mask
        frame_objects = {
            "2": {
                "geometry": mock_cuboid,
                "object_type": "TrafficSign",
            }
        }
        with patch.object(self.vis_helper, "_draw_object_overlay") as mock_draw_overlay:
            mock_draw_overlay.return_value = vis_image
            _ = self.vis_helper._add_object_overlays(
                vis_image, frame_scores, frame_objects, camera_pose, camera_model, filtered_objects
            )

        contour_color = mock_draw_overlay.call_args[0][4]
        self.assertEqual(contour_color, (255, 165, 0))  # Orange

    @patch("cv2.findContours")
    @patch("cv2.drawContours")
    @patch("cv2.boundingRect")
    @patch("cv2.getTextSize")
    @patch("cv2.rectangle")
    @patch("cv2.putText")
    def test_draw_object_overlay(
        self,
        mock_puttext: MagicMock,
        mock_rectangle: MagicMock,
        mock_textsize: MagicMock,
        mock_boundingrect: MagicMock,
        mock_drawcontours: MagicMock,
        mock_findcontours: MagicMock,
    ) -> None:
        """Test drawing object overlay."""
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        projected_mask = np.zeros((720, 1280), dtype=bool)
        projected_mask[100:200, 100:200] = True
        score = 0.85
        track_id = 123

        # Mock CV2 functions
        mock_contour = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
        mock_findcontours.return_value = ([mock_contour], None)
        mock_boundingrect.return_value = (100, 100, 100, 100)  # x, y, w, h
        mock_textsize.return_value = ((30, 15), 3)  # (width, height), baseline

        contour_color = (255, 0, 255)  # magenta sample color
        result = self.vis_helper._draw_object_overlay(vis_image, projected_mask, score, track_id, contour_color)

        # Should find contours and draw overlay
        mock_findcontours.assert_called_once()
        mock_drawcontours.assert_called_once_with(vis_image, [mock_contour], -1, contour_color, 2)
        mock_boundingrect.assert_called_once()
        mock_textsize.assert_called()
        mock_rectangle.assert_called()
        mock_puttext.assert_called()

        # Verify the result
        self.assertIs(result, vis_image)

    def test_mask_legend_override_items(self) -> None:
        """Explicit legend items override should be passed to add_legend as-is."""
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_idx = 3
        override_items = [("drivable", (1, 2, 3)), ("vehicle", (9, 8, 7))]

        self.vis_helper.set_mask_styles(legend_items=override_items)

        with patch.object(self.vis_helper, "add_legend") as mock_add_legend:
            mock_add_legend.return_value = vis_image
            result = self.vis_helper._add_mask_legend(vis_image, frame_idx)

        mock_add_legend.assert_called_once()
        called_items = mock_add_legend.call_args[0][2]
        self.assertEqual(called_items, override_items)
        self.assertIs(result, vis_image)

        # reset override for other tests
        self.vis_helper.set_mask_styles(legend_items=None)

    def test_mask_legend_ordering(self) -> None:
        """Legend should honor provided legend_order, then append remaining colors."""
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_idx = 4
        colors = {"b": (2, 2, 2), "a": (1, 1, 1), "c": (3, 3, 3)}
        order = ["a", "c"]

        self.vis_helper.set_mask_styles(mask_colors=colors, legend_order=order, legend_items=None)

        with patch.object(self.vis_helper, "add_legend") as mock_add_legend:
            mock_add_legend.return_value = vis_image
            result = self.vis_helper._add_mask_legend(vis_image, frame_idx)

        called_items = mock_add_legend.call_args[0][2]
        expected = [("a", (1, 1, 1)), ("c", (3, 3, 3)), ("b", (2, 2, 2))]
        self.assertEqual(called_items, expected)
        self.assertIs(result, vis_image)

    def test_named_mask_legend_independent(self) -> None:
        """Named mask legends should be independent of the default legend."""
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_idx = 5
        default_items = [("MATCH", (0, 255, 0))]
        type_items = [("BBOX_CAR", (200, 50, 50)), ("BBOX_TRUCK", (100, 100, 200))]

        self.vis_helper.set_mask_styles(legend_items=default_items)
        self.vis_helper.set_mask_styles(legend_items=type_items, mask_name="type")

        with patch.object(self.vis_helper, "add_legend") as mock_add_legend:
            mock_add_legend.return_value = vis_image

            self.vis_helper._build_mask_legend(vis_image, frame_idx, "default")
            default_call_items = mock_add_legend.call_args[0][2]
            self.assertEqual(default_call_items, default_items)

            self.vis_helper._build_mask_legend(vis_image, frame_idx, "type")
            type_call_items = mock_add_legend.call_args[0][2]
            self.assertEqual(type_call_items, type_items)

    def test_write_mask_frame_creates_named_files(self) -> None:
        """write_mask_frame with different mask_name values should create separate files."""
        self.vis_helper.frame_width = 64
        self.vis_helper.frame_height = 48
        self.vis_helper.fps = 10

        frame = np.zeros((48, 64, 3), dtype=np.uint8)

        with patch("checks.objects.single_view.visualization_helper.VideoWriter") as mock_writer_cls:
            mock_instance = MagicMock()
            mock_instance.open.return_value = True
            mock_instance.write_frame.return_value = True
            mock_writer_cls.return_value = mock_instance

            self.vis_helper.write_mask_frame(frame, 0, "static")
            self.vis_helper.write_mask_frame(frame, 0, "static", mask_name="type")

        # Two separate writers should have been created
        self.assertEqual(mock_writer_cls.call_count, 2)
        paths = [call.kwargs["output_path"] for call in mock_writer_cls.call_args_list]
        self.assertTrue(str(paths[0]).endswith(".static.mask.mp4"))
        self.assertTrue(str(paths[1]).endswith(".static.type_mask.mp4"))

    def test_release_closes_all_mask_writers(self) -> None:
        """release() should close overlay writer and all named mask video writers."""
        mock_writer_a = MagicMock()
        mock_writer_b = MagicMock()
        self.vis_helper._mask_video_writers = {"default": mock_writer_a, "type": mock_writer_b}

        self.vis_helper.release()

        mock_writer_a.release.assert_called_once()
        mock_writer_b.release.assert_called_once()

    @patch("cv2.findContours")
    @patch("cv2.moments")
    def test_draw_filter_marker_distance(self, mock_moments: MagicMock, mock_findcontours: MagicMock) -> None:
        """Test drawing filter marker for distance filter."""
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        camera_pose = np.eye(4)
        camera_model = MagicMock()
        track_id = 123
        filter_info = {"reason": "distance_threshold_m (50.0m)", "type": "importance"}

        # Mock cuboid and contours
        mock_cuboid = MagicMock()
        mock_projected_mask = np.zeros((720, 1280), dtype=bool)
        mock_projected_mask[100:200, 100:200] = True
        mock_depth_mask = np.zeros((720, 1280), dtype=np.float32)
        mock_depth_mask[100:200, 100:200] = 1.0
        mock_cuboid.get_projected_mask.return_value = mock_projected_mask, mock_depth_mask
        tracked_object = {"geometry": mock_cuboid}

        mock_contour = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
        mock_findcontours.return_value = ([mock_contour], None)
        mock_moments.return_value = {"m00": 10000, "m10": 1500000, "m01": 1500000}  # Center at (150, 150)

        with patch("cv2.circle") as mock_circle:
            result = self.vis_helper._draw_filter_marker(
                vis_image, tracked_object, camera_pose, camera_model, track_id, filter_info
            )

        # Should draw red dot for distance filter
        mock_circle.assert_called_once_with(vis_image, (150, 150), 2, (255, 0, 0), -1)

        # Verify the result
        self.assertIs(result, vis_image)

    @patch("cv2.findContours")
    @patch("cv2.moments")
    def test_draw_filter_marker_oncoming(self, mock_moments: MagicMock, mock_findcontours: MagicMock) -> None:
        """Test drawing filter marker for oncoming filter."""
        vis_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        camera_pose = np.eye(4)
        camera_model = MagicMock()
        track_id = 123
        filter_info = {"reason": "skip_oncoming_obstacles (enabled)", "type": "importance"}

        # Mock cuboid and contours
        mock_cuboid = MagicMock()
        mock_projected_mask = np.zeros((720, 1280), dtype=bool)
        mock_projected_mask[100:200, 100:200] = True
        mock_depth_mask = np.zeros((720, 1280), dtype=np.float32)
        mock_depth_mask[100:200, 100:200] = 1.0
        mock_cuboid.get_projected_mask.return_value = mock_projected_mask, mock_depth_mask
        tracked_object = {"geometry": mock_cuboid}

        mock_contour = np.array([[100, 100], [200, 100], [200, 200], [100, 200]])
        mock_findcontours.return_value = ([mock_contour], None)
        mock_moments.return_value = {"m00": 10000, "m10": 1500000, "m01": 1500000}  # Center at (150, 150)

        with patch("cv2.line") as mock_line:
            result = self.vis_helper._draw_filter_marker(
                vis_image, tracked_object, camera_pose, camera_model, track_id, filter_info
            )

        # Should draw blue cross for oncoming filter
        self.assertEqual(mock_line.call_count, 2)  # Two lines for cross

        # Verify the result
        self.assertIs(result, vis_image)

    def test_write_frame_success(self) -> None:
        """Test successful frame writing (delegated to the VideoWriter)."""
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_idx = 5

        # Our VideoWriter handles color/resize internally; the helper just delegates.
        mock_writer = MagicMock()
        mock_writer.write_frame.return_value = True
        self.vis_helper.video_writer = mock_writer
        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720

        result = self.vis_helper.write_frame(frame, frame_idx)

        self.assertTrue(result)
        mock_writer.write_frame.assert_called_once_with(frame, frame_idx)

    def test_write_frame_no_writer(self) -> None:
        """Test frame writing when no video writer is available."""
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_idx = 5

        result = self.vis_helper.write_frame(frame, frame_idx)

        self.assertFalse(result)

    def test_write_frame_resize_delegated(self) -> None:
        """Test that a wrong-sized frame is delegated as-is (VideoWriter resizes internally)."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Wrong size
        frame_idx = 5

        mock_writer = MagicMock()
        mock_writer.write_frame.return_value = True
        self.vis_helper.video_writer = mock_writer
        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720

        result = self.vis_helper.write_frame(frame, frame_idx)

        self.assertTrue(result)
        mock_writer.write_frame.assert_called_once_with(frame, frame_idx)

    def test_save_frame_image(self) -> None:
        """Test saving frame as individual image."""
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        frame_idx = 5

        with patch("cv2.imwrite") as mock_imwrite, patch("cv2.cvtColor") as mock_cvtcolor:
            mock_cvtcolor.return_value = frame
            mock_imwrite.return_value = True

            result = self.vis_helper.save_frame_image(frame, frame_idx)

        self.assertTrue(result)
        mock_imwrite.assert_called_once()

        # Check that filename contains frame index
        call_args = mock_imwrite.call_args[0]
        filename = call_args[0]
        self.assertIn("0005", filename)  # Frame index formatted with zero padding

    def test_release(self) -> None:
        """Test releasing video resources."""
        # The ffmpeg-backed reader needs no handle to release; only the writer is finalized.
        mock_writer = MagicMock()
        self.vis_helper.video_writer = mock_writer

        self.vis_helper.release()

        mock_writer.release.assert_called_once()

    def test_release_no_resources(self) -> None:
        """Test releasing when no resources are allocated."""
        # Should not raise an exception
        self.vis_helper.release()

    def test_create_frame_visualization_integration(self) -> None:
        """Test the complete frame visualization pipeline."""
        frame_idx = 10
        segmentation_masks = MagicMock()
        segmentation_masks.numpy.return_value = np.random.randint(0, 255, (3, 720, 1280), dtype=np.uint8)

        frame_scores = {"123": 0.8}
        camera_pose = np.eye(4)
        camera_model = MagicMock()
        mock_cuboid = MagicMock()
        mock_cuboid.get_projected_mask.return_value = (
            np.zeros((720, 1280), dtype=bool),
            np.full((720, 1280), np.inf, dtype=np.float32),
        )
        frame_objects = {"123": {"geometry": mock_cuboid}}
        filtered_objects: dict[str, Any] = {}

        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720

        with (
            patch.object(self.vis_helper, "get_original_frame") as mock_get_frame,
            patch.object(self.vis_helper, "_overlay_segmentation") as mock_overlay,
            patch.object(self.vis_helper, "_add_object_overlays") as mock_add_overlays,
            patch.object(self.vis_helper, "add_legend") as mock_add_legend,
        ):
            mock_base_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            mock_get_frame.return_value = mock_base_image
            mock_overlay.return_value = mock_base_image
            mock_add_overlays.return_value = mock_base_image
            mock_add_legend.return_value = mock_base_image

            result = self.vis_helper.create_frame_visualization(
                frame_idx, segmentation_masks, frame_scores, camera_pose, camera_model, frame_objects, filtered_objects
            )

        # Should call all steps in the pipeline
        mock_get_frame.assert_called_once_with(frame_idx)
        mock_overlay.assert_called_once()
        mock_add_overlays.assert_called_once()
        mock_add_legend.assert_called_once_with(mock_base_image, frame_idx, self.vis_helper.legend_items)

        self.assertIsNotNone(result)

    def test_create_frame_visualization_with_scene_rasterizer(self) -> None:
        """Test frame visualization pipeline with pre-computed SceneRasterizer."""
        frame_idx = 10
        segmentation_masks = MagicMock()
        segmentation_masks.numpy.return_value = np.random.randint(0, 255, (3, 720, 1280), dtype=np.uint8)

        frame_scores = {"123": 0.8}
        camera_pose = np.eye(4)
        camera_model = MagicMock()
        mock_cuboid = MagicMock()
        mock_projected_mask = np.zeros((720, 1280), dtype=bool)
        mock_projected_mask[100:200, 100:200] = True
        mock_depth_mask = np.full((720, 1280), np.inf, dtype=np.float32)
        mock_depth_mask[100:200, 100:200] = 5.0
        mock_cuboid.get_projected_mask.return_value = mock_projected_mask, mock_depth_mask
        frame_objects = {"123": {"geometry": mock_cuboid}}
        filtered_objects: dict[str, Any] = {}

        self.vis_helper.frame_width = 1280
        self.vis_helper.frame_height = 720

        # Create a SceneRasterizer (computes visibility masks in constructor)
        scene_rasterizer = SceneRasterizer(
            frame_objects, camera_pose, camera_model, 1280, 720, logger=logging.getLogger("test")
        )

        with (
            patch.object(self.vis_helper, "get_original_frame") as mock_get_frame,
            patch.object(self.vis_helper, "_overlay_segmentation") as mock_overlay,
            patch.object(self.vis_helper, "_add_object_overlays") as mock_add_overlays,
            patch.object(self.vis_helper, "add_legend") as mock_add_legend,
        ):
            mock_base_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            mock_get_frame.return_value = mock_base_image
            mock_overlay.return_value = mock_base_image
            mock_add_overlays.return_value = mock_base_image
            mock_add_legend.return_value = mock_base_image

            result = self.vis_helper.create_frame_visualization(
                frame_idx,
                segmentation_masks,
                frame_scores,
                camera_pose,
                camera_model,
                frame_objects,
                filtered_objects,
                scene_rasterizer=scene_rasterizer,
            )

        # Should pass scene_rasterizer to _add_object_overlays
        mock_add_overlays.assert_called_once()
        call_kwargs = mock_add_overlays.call_args[1]
        self.assertIn("scene_rasterizer", call_kwargs)
        self.assertIs(call_kwargs["scene_rasterizer"], scene_rasterizer)
        self.assertIsNotNone(result)


class TestAddTrafficLightStateTags(unittest.TestCase):
    """Test cases for VisualizationHelper._add_traffic_light_state_tags."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.vis_helper = VisualizationHelper("clip_tl", self.temp_dir, verbose=False)
        self.vis_helper.frame_width = 200
        self.vis_helper.frame_height = 200
        self.vis_image = np.zeros((200, 200, 3), dtype=np.uint8)
        self.camera_pose = np.eye(4)
        self.camera_model = MagicMock()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    @staticmethod
    def _make_result(
        score: float | None = 0.85, mc_score: float | None = 0.2, confidence: float | None = 0.7
    ) -> MagicMock:
        """Build a lightweight object mimicking TrafficLightStateResult."""
        return MagicMock(
            traffic_light_state_score=score,
            traffic_light_multi_color_score=mc_score,
            confidence=confidence,
        )

    @staticmethod
    def _make_rasterizer_with_mask(
        track_id_str: str, shape: tuple[int, int], region: tuple[int, int, int, int]
    ) -> MagicMock:
        """Return a mock SceneRasterizer that has a visibility mask for *track_id_str*."""
        mask = np.zeros(shape, dtype=bool)
        r0, r1, c0, c1 = region
        mask[r0:r1, c0:c1] = True
        rasterizer = MagicMock(spec=SceneRasterizer)
        rasterizer.has_object.side_effect = lambda tid: tid == track_id_str
        rasterizer.get_visibility_mask.return_value = mask
        return rasterizer

    def test_all_labels_drawn(self) -> None:
        """All three tag lines (TLS, MC, C) are drawn when all fields are valid."""
        rasterizer = self._make_rasterizer_with_mask("42", (200, 200), (50, 100, 60, 120))
        tl_results = {"42": self._make_result(score=0.91, mc_score=0.30, confidence=0.75)}

        with (
            patch("cv2.getTextSize", return_value=((40, 10), 2)),
            patch("cv2.rectangle"),
            patch("cv2.putText") as mock_put,
        ):
            result = self.vis_helper._add_traffic_light_state_tags(
                self.vis_image, tl_results, {}, self.camera_pose, self.camera_model, rasterizer
            )

        labels = [call[0][1] for call in mock_put.call_args_list]
        self.assertEqual(labels, ["TLS:0.91", "MC:0.30", "C:0.75"])
        self.assertIs(result, self.vis_image)

    def test_nan_score_shows_na(self) -> None:
        """NaN state score renders as 'TLS:n/a'."""
        rasterizer = self._make_rasterizer_with_mask("1", (200, 200), (10, 30, 10, 30))
        tl_results = {"1": self._make_result(score=float("nan"), mc_score=float("nan"), confidence=float("nan"))}

        with (
            patch("cv2.getTextSize", return_value=((40, 10), 2)),
            patch("cv2.rectangle"),
            patch("cv2.putText") as mock_put,
        ):
            self.vis_helper._add_traffic_light_state_tags(
                self.vis_image, tl_results, {}, self.camera_pose, self.camera_model, rasterizer
            )

        labels = [call[0][1] for call in mock_put.call_args_list]
        self.assertEqual(labels, ["TLS:n/a"])

    def test_none_score_skips_track(self) -> None:
        """Track is skipped entirely when traffic_light_state_score is None."""
        rasterizer = self._make_rasterizer_with_mask("9", (200, 200), (10, 30, 10, 30))
        tl_results = {"9": self._make_result(score=None)}

        with patch("cv2.putText") as mock_put:
            self.vis_helper._add_traffic_light_state_tags(
                self.vis_image, tl_results, {}, self.camera_pose, self.camera_model, rasterizer
            )

        mock_put.assert_not_called()

    def test_empty_visibility_mask_skips_track(self) -> None:
        """Track is skipped when its visibility mask is all-False."""
        rasterizer = MagicMock(spec=SceneRasterizer)
        rasterizer.has_object.return_value = True
        rasterizer.get_visibility_mask.return_value = np.zeros((200, 200), dtype=bool)

        tl_results = {"5": self._make_result()}

        with patch("cv2.putText") as mock_put:
            self.vis_helper._add_traffic_light_state_tags(
                self.vis_image, tl_results, {}, self.camera_pose, self.camera_model, rasterizer
            )

        mock_put.assert_not_called()

    def test_geometry_fallback_when_no_rasterizer(self) -> None:
        """Falls back to frame_objects geometry when scene_rasterizer is None."""
        mask = np.zeros((200, 200), dtype=bool)
        mask[80:100, 80:100] = True
        mock_geometry = MagicMock()
        mock_geometry.get_projected_mask.return_value = (mask, None)
        frame_objects = {"7": {"geometry": mock_geometry}}

        tl_results = {"7": self._make_result(score=0.60, mc_score=None, confidence=None)}

        with (
            patch("cv2.getTextSize", return_value=((30, 10), 2)),
            patch("cv2.rectangle"),
            patch("cv2.putText") as mock_put,
        ):
            self.vis_helper._add_traffic_light_state_tags(
                self.vis_image, tl_results, frame_objects, self.camera_pose, self.camera_model, None
            )

        mock_geometry.get_projected_mask.assert_called_once_with(self.camera_pose, self.camera_model, 200, 200)
        labels = [call[0][1] for call in mock_put.call_args_list]
        self.assertEqual(labels, ["TLS:0.60"])

    def test_partial_fields_only_score(self) -> None:
        """Only TLS label is drawn when mc_score and confidence are NaN."""
        rasterizer = self._make_rasterizer_with_mask("3", (200, 200), (40, 60, 40, 60))
        tl_results = {"3": self._make_result(score=0.55, mc_score=float("nan"), confidence=float("nan"))}

        with (
            patch("cv2.getTextSize", return_value=((30, 10), 2)),
            patch("cv2.rectangle"),
            patch("cv2.putText") as mock_put,
        ):
            self.vis_helper._add_traffic_light_state_tags(
                self.vis_image, tl_results, {}, self.camera_pose, self.camera_model, rasterizer
            )

        labels = [call[0][1] for call in mock_put.call_args_list]
        self.assertEqual(labels, ["TLS:0.55"])

    def test_track_not_in_rasterizer_or_objects_skipped(self) -> None:
        """Track is skipped when not in rasterizer and not in frame_objects."""
        rasterizer = MagicMock(spec=SceneRasterizer)
        rasterizer.has_object.return_value = False

        tl_results = {"99": self._make_result()}

        with patch("cv2.putText") as mock_put:
            self.vis_helper._add_traffic_light_state_tags(
                self.vis_image, tl_results, {}, self.camera_pose, self.camera_model, rasterizer
            )

        mock_put.assert_not_called()

    def test_label_clipped_at_bottom_edge(self) -> None:
        """Later labels that would overflow the image bottom are not drawn."""
        mask = np.zeros((200, 200), dtype=bool)
        mask[170:176, 50:60] = True
        rasterizer = MagicMock(spec=SceneRasterizer)
        rasterizer.has_object.return_value = True
        rasterizer.get_visibility_mask.return_value = mask

        tl_results = {"6": self._make_result(score=0.8, mc_score=0.1, confidence=0.9)}

        with (
            patch("cv2.getTextSize", return_value=((40, 10), 2)),
            patch("cv2.rectangle"),
            patch("cv2.putText") as mock_put,
        ):
            self.vis_helper._add_traffic_light_state_tags(
                self.vis_image, tl_results, {}, self.camera_pose, self.camera_model, rasterizer
            )

        drawn_labels = [call[0][1] for call in mock_put.call_args_list]
        self.assertGreater(len(drawn_labels), 0)
        self.assertLess(len(drawn_labels), 3)
        self.assertEqual(drawn_labels[0], "TLS:0.80")

    def test_multiple_tracks(self) -> None:
        """Tags are drawn independently for each track in tl_results."""
        mask_a = np.zeros((200, 200), dtype=bool)
        mask_a[10:30, 10:30] = True
        mask_b = np.zeros((200, 200), dtype=bool)
        mask_b[60:80, 60:80] = True

        rasterizer = MagicMock(spec=SceneRasterizer)
        rasterizer.has_object.return_value = True
        rasterizer.get_visibility_mask.side_effect = lambda tid: mask_a if tid == "A" else mask_b

        tl_results = {
            "A": self._make_result(score=0.40, mc_score=None, confidence=None),
            "B": self._make_result(score=0.95, mc_score=None, confidence=None),
        }

        with (
            patch("cv2.getTextSize", return_value=((30, 10), 2)),
            patch("cv2.rectangle"),
            patch("cv2.putText") as mock_put,
        ):
            self.vis_helper._add_traffic_light_state_tags(
                self.vis_image, tl_results, {}, self.camera_pose, self.camera_model, rasterizer
            )

        labels = [call[0][1] for call in mock_put.call_args_list]
        self.assertIn("TLS:0.40", labels)
        self.assertIn("TLS:0.95", labels)


class TestHallucinationOverlayLabel(unittest.TestCase):
    """The hallucination overlay tag is compact and derived from the class's initials."""

    def test_rds_camelcase_labels(self) -> None:
        # The hallucinations dict is keyed by RDS labels; these are what actually appear in the overlay.
        self.assertEqual(hallucination_overlay_label("TrafficLight"), "H:TL")
        self.assertEqual(hallucination_overlay_label("TrafficSign"), "H:TS")

    def test_snake_case_labels(self) -> None:
        self.assertEqual(hallucination_overlay_label("traffic_light"), "H:TL")
        self.assertEqual(hallucination_overlay_label("traffic_sign"), "H:TS")

    def test_single_word_fallback(self) -> None:
        self.assertEqual(hallucination_overlay_label("Vehicle"), "H:VE")
        self.assertEqual(hallucination_overlay_label("Pedestrian"), "H:PE")


if __name__ == "__main__":
    unittest.main()
