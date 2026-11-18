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

"""Unit tests for polyline module."""

import unittest
import unittest.mock
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from checks.utils.polyline import Polyline, _clip_segment_to_z_range
from checks.utils.surface import Surface


class TestPolyline(unittest.TestCase):
    """Test cases for Polyline class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Sample polyline vertices
        self.sample_vertices = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 5.0, 3.0]])
        self.polyline = Polyline(self.sample_vertices, half_width_meters=0.5)

    def test_init(self) -> None:
        """Test Polyline initialization with valid inputs."""
        self.assertTrue(np.allclose(self.polyline.vertices_world, self.sample_vertices))
        self.assertEqual(self.polyline.half_width_m, 0.5)
        self.assertEqual(self.polyline.min_cutoff_distance, 3.0)
        self.assertEqual(self.polyline.max_cutoff_distance, 50.0)

    def test_init_with_custom_parameters(self) -> None:
        """Test Polyline initialization with custom parameters."""
        polyline = Polyline(
            self.sample_vertices, half_width_meters=1.0, min_cutoff_distance=5.0, max_cutoff_distance=100.0
        )
        self.assertEqual(polyline.half_width_m, 1.0)
        self.assertEqual(polyline.min_cutoff_distance, 5.0)
        self.assertEqual(polyline.max_cutoff_distance, 100.0)

    def test_init_with_list_input(self) -> None:
        """Test Polyline initialization with list input."""
        vertices_list = self.sample_vertices.tolist()
        polyline = Polyline(vertices_list)
        assert_array_almost_equal(polyline.vertices_world, self.sample_vertices)

    def test_init_invalid_shape_1d(self) -> None:
        """Test Polyline initialization with 1D array."""
        with self.assertRaises(ValueError):
            Polyline(np.array([1.0, 2.0, 3.0]))

    def test_init_invalid_shape_wrong_columns(self) -> None:
        """Test Polyline initialization with wrong number of columns."""
        with self.assertRaises(ValueError):
            Polyline(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_init_invalid_shape_single_point(self) -> None:
        """Test Polyline initialization with single point."""
        with self.assertRaises(ValueError):
            Polyline(np.array([[1.0, 2.0, 3.0]]))

    def test_init_negative_half_width(self) -> None:
        """Test Polyline initialization with negative half_width."""
        polyline = Polyline(self.sample_vertices, half_width_meters=-1.0)
        # Should clamp to 0.0
        self.assertEqual(polyline.half_width_m, 0.0)

    def test_init_invalid_cutoff_distances(self) -> None:
        """Test Polyline initialization with min >= max cutoff distance."""
        with self.assertRaises(ValueError):
            Polyline(self.sample_vertices, min_cutoff_distance=10.0, max_cutoff_distance=5.0)

    def test_init_equal_cutoff_distances(self) -> None:
        """Test Polyline initialization with equal cutoff distances."""
        with self.assertRaises(ValueError):
            Polyline(self.sample_vertices, min_cutoff_distance=10.0, max_cutoff_distance=10.0)

    @patch("checks.utils.polyline.CameraBase.transform_points_np")
    def test_get_projected_mask_basic(self, mock_transform: MagicMock) -> None:
        """Test get_projected_mask with basic polyline."""

        # Mock camera transformation: transform points to camera coordinates
        # For a simple case, return points with positive z (in front of camera)
        def transform_side_effect(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
            # Simple identity-like transform: just return points with z=10
            result = np.asarray(points).copy()
            result[:, 2] = 10.0  # All points at z=10 in camera frame
            return result

        mock_transform.side_effect = transform_side_effect

        # Mock camera model: return x,y from 3D points
        mock_camera_model = MagicMock()
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.asarray(pts)[:, :2] * 10 + np.array([320, 240])

        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        with patch("checks.utils.rasterization.cv2.fillPoly") as mock_fillpoly:
            mask, depth_mask = self.polyline.get_projected_mask(
                camera_to_world_pose, mock_camera_model, image_width, image_height
            )

        # Check that mask has correct shape and type
        self.assertEqual(mask.shape, (image_height, image_width))
        self.assertEqual(mask.dtype, bool)

        # Check that depth_mask has correct shape and type
        self.assertEqual(depth_mask.shape, (image_height, image_width))
        self.assertEqual(depth_mask.dtype, np.float32)

        # Check that transform_points_np was called (at least once per segment)
        self.assertGreaterEqual(mock_transform.call_count, 1)

        # Check that ray2pixel_np was called
        self.assertGreaterEqual(mock_camera_model.ray2pixel_np.call_count, 1)

        # Check that fillPoly was called
        self.assertGreaterEqual(mock_fillpoly.call_count, 1)

    @patch("checks.utils.polyline.CameraBase.transform_points_np")
    def test_get_projected_mask_behind_camera(self, mock_transform: MagicMock) -> None:
        """Test get_projected_mask when polyline is behind camera."""

        # Mock camera transformation with negative z (behind camera)
        def transform_side_effect(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
            result = np.asarray(points).copy()
            result[:, 2] = -10.0  # All points behind camera
            return result

        mock_transform.side_effect = transform_side_effect

        mock_camera_model = MagicMock()
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.asarray(pts)[:, :2]

        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        mask, depth_mask = self.polyline.get_projected_mask(
            camera_to_world_pose, mock_camera_model, image_width, image_height
        )

        # Should return all-zeros mask (clipped away)
        expected_mask = np.zeros((image_height, image_width), dtype=bool)
        assert_array_equal(mask, expected_mask)

        # Depth mask should be all inf (no valid pixels)
        self.assertTrue(np.all(np.isinf(depth_mask)))

    @patch("checks.utils.polyline.CameraBase.transform_points_np")
    def test_get_projected_mask_zero_length_segment(self, mock_transform: MagicMock) -> None:
        """Test get_projected_mask with zero-length segment (duplicate points)."""
        # Polyline with duplicate consecutive points
        vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        polyline = Polyline(vertices)

        def transform_side_effect(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
            result = np.asarray(points).copy()
            result[:, 2] = 10.0
            return result

        mock_transform.side_effect = transform_side_effect

        mock_camera_model = MagicMock()
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.asarray(pts)[:, :2]

        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        mask, depth_mask = polyline.get_projected_mask(
            camera_to_world_pose, mock_camera_model, image_width, image_height
        )

        # Zero-length segment should be skipped, but other segments should still project
        self.assertEqual(mask.shape, (image_height, image_width))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(depth_mask.shape, (image_height, image_width))
        self.assertEqual(depth_mask.dtype, np.float32)

    def test_get_projected_mask_invalid_pose_shape(self) -> None:
        """Test get_projected_mask with invalid pose shape."""
        mock_camera_model = MagicMock()
        invalid_pose = np.eye(3)  # 3x3 instead of 4x4

        with self.assertRaises(ValueError):
            self.polyline.get_projected_mask(invalid_pose, mock_camera_model, 640, 480)

    def test_get_projected_mask_invalid_image_dimensions(self) -> None:
        """Test get_projected_mask with invalid image dimensions."""
        mock_camera_model = MagicMock()
        camera_to_world_pose = np.eye(4)

        with self.assertRaises(ValueError):
            self.polyline.get_projected_mask(camera_to_world_pose, mock_camera_model, 0, 480)

        with self.assertRaises(ValueError):
            self.polyline.get_projected_mask(camera_to_world_pose, mock_camera_model, 640, -1)

    @patch("checks.utils.polyline.CameraBase.transform_points_np")
    def test_get_projected_mask_with_rotation(self, mock_transform: MagicMock) -> None:
        """Test get_projected_mask with rotated camera pose."""
        # Create rotation around Z-axis by 90 degrees
        angle = np.pi / 2
        rotation_matrix = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0, 0],
                [np.sin(angle), np.cos(angle), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        def transform_side_effect(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
            result = np.asarray(points).copy()
            result[:, 2] = 10.0
            return result

        mock_transform.side_effect = transform_side_effect

        mock_camera_model = MagicMock()
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.asarray(pts)[:, :2] * 10 + np.array([320, 240])

        image_width, image_height = 640, 480

        with patch("checks.utils.rasterization.cv2.fillPoly"):
            mask, depth_mask = self.polyline.get_projected_mask(
                rotation_matrix, mock_camera_model, image_width, image_height
            )

        self.assertEqual(mask.shape, (image_height, image_width))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(depth_mask.shape, (image_height, image_width))
        self.assertEqual(depth_mask.dtype, np.float32)

    @patch("checks.utils.polyline.CameraBase.transform_points_np")
    def test_get_projected_mask_multiple_segments(self, mock_transform: MagicMock) -> None:
        """Test get_projected_mask with polyline having multiple segments."""
        # Polyline with 4 points = 3 segments
        vertices = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0], [0.0, 5.0, 0.0]])
        polyline = Polyline(vertices, half_width_meters=0.5)

        def transform_side_effect(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
            result = np.asarray(points).copy()
            result[:, 2] = 10.0
            return result

        mock_transform.side_effect = transform_side_effect

        mock_camera_model = MagicMock()
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.asarray(pts)[:, :2] * 10 + np.array([320, 240])

        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        with patch("checks.utils.rasterization.cv2.fillPoly") as mock_fillpoly:
            mask, depth_mask = polyline.get_projected_mask(
                camera_to_world_pose, mock_camera_model, image_width, image_height
            )

        self.assertEqual(mask.shape, (image_height, image_width))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(depth_mask.shape, (image_height, image_width))
        self.assertEqual(depth_mask.dtype, np.float32)
        # Should have called fillPoly for each segment (3 segments)
        self.assertGreaterEqual(mock_fillpoly.call_count, 1)

    @patch("checks.utils.polyline.CameraBase.transform_points_np")
    def test_get_projected_mask_cutoff_distances(self, mock_transform: MagicMock) -> None:
        """Test get_projected_mask respects cutoff distances."""
        # Create polyline with custom cutoff distances
        polyline = Polyline(self.sample_vertices, min_cutoff_distance=5.0, max_cutoff_distance=20.0)

        def transform_side_effect(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
            result = np.asarray(points).copy()
            # Set z values that are outside the cutoff range
            result[:, 2] = 2.0  # Less than min_cutoff_distance
            return result

        mock_transform.side_effect = transform_side_effect

        mock_camera_model = MagicMock()
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.asarray(pts)[:, :2]

        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        mask, depth_mask = polyline.get_projected_mask(
            camera_to_world_pose, mock_camera_model, image_width, image_height
        )

        # Points outside cutoff range should be clipped away
        expected_mask = np.zeros((image_height, image_width), dtype=bool)
        assert_array_equal(mask, expected_mask)

        # Depth mask should be all inf
        self.assertTrue(np.all(np.isinf(depth_mask)))

    @patch("checks.utils.polyline.CameraBase.transform_points_np")
    def test_get_projected_mask_camera_model_exception(self, mock_transform: MagicMock) -> None:
        """Test get_projected_mask handles camera model exceptions gracefully."""

        def transform_side_effect(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
            result = np.asarray(points).copy()
            result[:, 2] = 10.0
            return result

        mock_transform.side_effect = transform_side_effect

        # Mock camera model that raises exception
        mock_camera_model = MagicMock()
        mock_camera_model.ray2pixel_np.side_effect = Exception("Camera projection failed")

        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        # Should not raise exception, but return empty mask
        mask, depth_mask = self.polyline.get_projected_mask(
            camera_to_world_pose, mock_camera_model, image_width, image_height
        )

        self.assertEqual(mask.shape, (image_height, image_width))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(depth_mask.shape, (image_height, image_width))
        self.assertEqual(depth_mask.dtype, np.float32)

    def test_vertices_world_immutability(self) -> None:
        """Test that modifying input array doesn't affect polyline."""
        vertices = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        polyline = Polyline(vertices)

        # Modify original array
        vertices[0, 0] = 999.0

        # Polyline should have its own copy
        self.assertNotEqual(polyline.vertices_world[0, 0], 999.0)
        self.assertEqual(polyline.vertices_world[0, 0], 0.0)


class TestPolylineWorldDistance(unittest.TestCase):
    """Tracker-facing world_distance metric."""

    def test_match_gate_default(self) -> None:
        """Default gate is 1.5 m and is exposed both as class attr and instance attr."""
        self.assertEqual(Polyline.MATCH_GATE_METERS, 1.5)
        polyline = Polyline(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
        self.assertEqual(polyline.MATCH_GATE_METERS, 1.5)

    def test_identical_geometry_zero_distance(self) -> None:
        """world_distance to self (or a copy) is exactly 0."""
        verts = np.array([[0.0, 0.0, 0.0], [5.0, 1.0, 0.0], [10.0, 2.0, 0.0]])
        a = Polyline(verts)
        b = Polyline(np.array(verts))
        self.assertEqual(a.world_distance(a), 0.0)
        self.assertEqual(a.world_distance(b), 0.0)

    def test_constant_offset_recovers_offset(self) -> None:
        """A uniformly translated copy yields a distance equal to the offset magnitude."""
        verts = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        a = Polyline(verts)
        b = Polyline(verts + np.array([0.0, 0.4, 0.0]))
        self.assertAlmostEqual(a.world_distance(b), 0.4, places=6)

    def test_partial_overlap_uses_median_not_mean(self) -> None:
        """
        Asymmetric median tolerates a tail of unmatched ``self`` vertices: when
        most of ``self`` lies on top of ``other`` and a few extra vertices
        extend beyond, the median nearest-neighbor distance stays small.
        """
        # `other` covers x in [0, 10]; `self` extends to x=100 with one tail vertex.
        other = Polyline(np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
        self_verts = np.array([[i, 0.0, 0.0] for i in (0.0, 2.5, 5.0, 7.5, 10.0, 100.0)])
        a = Polyline(self_verts)
        # 5/6 self-vertices have nearest distance 0; the 6th has distance 90.
        # Median of [0, 0, 0, 0, 0, 90] = 0.
        self.assertEqual(a.world_distance(other), 0.0)

    def test_asymmetry_is_intentional(self) -> None:
        """
        ``self.world_distance(other)`` and ``other.world_distance(self)`` may
        differ when polyline extents differ. The longer polyline is the one
        that can be far from the shorter on average.
        """
        short = Polyline(np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))
        long = Polyline(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [50.0, 0.0, 0.0], [100.0, 0.0, 0.0]]))
        # short -> long: every short vertex is on the long polyline
        self.assertEqual(short.world_distance(long), 0.0)
        # long -> short: half the long-vertices are far from short
        self.assertGreater(long.world_distance(short), 10.0)

    def test_distinct_polylines_exceed_typical_gate(self) -> None:
        """Two laterally separated lanes (3 m apart) score well above the 1.5 m gate."""
        a = Polyline(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
        b = Polyline(np.array([[0.0, 3.0, 0.0], [10.0, 3.0, 0.0]]))
        self.assertGreater(a.world_distance(b), Polyline.MATCH_GATE_METERS)

    def test_cross_type_returns_inf(self) -> None:
        """A non-Polyline ``other`` yields +inf so cross-type matches never bind."""
        a = Polyline(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
        s = Surface(np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 5.0, 0.0]]))
        self.assertEqual(a.world_distance(s), float("inf"))


class _PinholeStub:
    """Minimal pinhole projector standing in for a ``CameraBase`` model.

    The spine tests need projection to actually behave like a camera — foreshortening, depth-dependent scale —
    because what they assert (a ribbon collapsing to sub-pixel height while its centerline still traces cleanly)
    only exists under a real perspective divide. A MagicMock returning fixed pixels cannot express that.
    """

    def __init__(self, focal_px: float = 500.0, cx: float = 640.0, cy: float = 360.0) -> None:
        self.focal_px = focal_px
        self.cx = cx
        self.cy = cy

    def ray2pixel_np(self, points_cam: np.ndarray) -> np.ndarray:
        pts = np.asarray(points_cam, dtype=float)
        z = np.maximum(pts[:, 2], 1e-6)
        return np.stack([pts[:, 0] / z * self.focal_px + self.cx, pts[:, 1] / z * self.focal_px + self.cy], axis=1)


# Camera-to-world pose for a forward-facing camera 1.5 m above the ground in an FLU world (X forward, Y left,
# Z up). Camera axes are RDF, so camera +x is world -Y, camera +y is world -Z, and camera +z is world +X.
_CAMERA_TO_WORLD = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 1.5],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
_IMAGE_WIDTH = 1280
_IMAGE_HEIGHT = 720


class TestPolylineSpineMask(unittest.TestCase):
    """The undilated centerline used by the ``spine`` correspondence metric."""

    def setUp(self) -> None:
        self.camera = _PinholeStub()
        # A lane line running straight ahead from 5 m to 40 m, directly in front of the camera.
        self.ahead = Polyline(np.array([[5.0, 0.0, 0.0], [40.0, 0.0, 0.0]]))

    def _spine(self, polyline: Polyline) -> tuple[np.ndarray, np.ndarray]:
        result = polyline.get_projected_spine_mask(_CAMERA_TO_WORLD, self.camera, _IMAGE_WIDTH, _IMAGE_HEIGHT)
        self.assertIsNotNone(result)
        assert result is not None  # narrow for the type checker
        return result

    def _ribbon(self, polyline: Polyline) -> np.ndarray:
        mask, _depth = polyline.get_projected_mask(_CAMERA_TO_WORLD, self.camera, _IMAGE_WIDTH, _IMAGE_HEIGHT)
        return mask

    def test_returns_full_size_mask_and_depth(self) -> None:
        mask, depth = self._spine(self.ahead)
        self.assertEqual(mask.shape, (_IMAGE_HEIGHT, _IMAGE_WIDTH))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(depth.shape, (_IMAGE_HEIGHT, _IMAGE_WIDTH))
        self.assertEqual(depth.dtype, np.float32)

    def test_is_one_pixel_wide(self) -> None:
        """A line running straight ahead projects to a single image column — the metric's denominator is a
        length, not an area."""
        mask, _depth = self._spine(self.ahead)
        columns = np.unique(np.nonzero(mask)[1])
        self.assertEqual(columns.size, 1)
        self.assertEqual(int(columns[0]), int(self.camera.cx))

    def test_trace_is_connected(self) -> None:
        """Consecutive traced rows leave no gaps, so the pixel count really is the visible arc length."""
        mask, _depth = self._spine(self.ahead)
        rows = np.unique(np.nonzero(mask)[0])
        self.assertGreater(rows.size, 50)
        assert_array_equal(np.diff(rows), np.ones(rows.size - 1, dtype=rows.dtype))

    def test_far_thinner_than_the_ribbon(self) -> None:
        """The whole point of the metric: the ribbon's area is dominated by a half-width nobody measured."""
        spine, _depth = self._spine(self.ahead)
        ribbon = self._ribbon(self.ahead)
        self.assertGreater(int(ribbon.sum()), 5 * int(spine.sum()))

    def test_lies_inside_the_ribbon(self) -> None:
        """The centerline is the ribbon's axis, so it must fall within it up to rasterization rounding."""
        spine, _depth = self._spine(self.ahead)
        ribbon = self._ribbon(self.ahead)
        inside = int(np.count_nonzero(spine & ribbon))
        self.assertGreater(inside / int(spine.sum()), 0.95)

    def test_depth_spans_the_polyline(self) -> None:
        mask, depth = self._spine(self.ahead)
        traced = depth[mask]
        self.assertGreater(traced.size, 0)
        self.assertAlmostEqual(float(traced.min()), 5.0, delta=0.2)
        self.assertAlmostEqual(float(traced.max()), 40.0, delta=0.2)

    def test_untraced_pixels_keep_infinite_depth(self) -> None:
        mask, depth = self._spine(self.ahead)
        self.assertTrue(np.isinf(depth[~mask]).all())

    def test_clipped_to_the_cutoff_window(self) -> None:
        """A line reaching from 1 m to 100 m is traced only over [min_cutoff, max_cutoff], the same window the
        ribbon uses, so the two representations agree on what is in range."""
        long_line = Polyline(np.array([[1.0, 0.0, 0.0], [100.0, 0.0, 0.0]]))
        mask, depth = self._spine(long_line)
        traced = depth[mask]
        self.assertGreater(traced.size, 0)
        self.assertGreaterEqual(float(traced.min()), long_line.min_cutoff_distance - 1e-3)
        self.assertLessEqual(float(traced.max()), long_line.max_cutoff_distance + 1e-3)

    def test_empty_when_behind_the_camera(self) -> None:
        behind = Polyline(np.array([[-20.0, 0.0, 0.0], [-5.0, 0.0, 0.0]]))
        mask, depth = self._spine(behind)
        self.assertFalse(mask.any())
        self.assertTrue(np.isinf(depth).all())

    def test_traces_a_lateral_line_the_ribbon_foreshortens_away(self) -> None:
        """A wait line crosses the view, so its 0.4 m ribbon is foreshortened to ~half a pixel of image height
        while the centerline still traces its full width. This is the case an area metric cannot score."""
        wait_line = Polyline(np.array([[25.0, -5.0, 0.0], [25.0, 5.0, 0.0]]))
        spine, _depth = self._spine(wait_line)
        ribbon = self._ribbon(wait_line)

        spine_rows = np.unique(np.nonzero(spine)[0])
        spine_cols = np.nonzero(spine)[1]
        self.assertEqual(spine_rows.size, 1, "a line at constant depth traces exactly one image row")
        self.assertGreater(int(spine_cols.max() - spine_cols.min()), 150)

        ribbon_rows = np.unique(np.nonzero(ribbon)[0])
        self.assertLessEqual(ribbon_rows.size, 2, "the ribbon has almost no image height to measure area over")

    def test_width_cutoff_trims_the_far_end(self) -> None:
        """The far end of a line is where the ribbon becomes too thin for a segmentation mask to resolve, so a
        width cutoff must remove the far samples and keep the near ones."""
        full, _depth = self._spine(self.ahead)
        trimmed = self.ahead.get_projected_spine_mask(
            _CAMERA_TO_WORLD, self.camera, _IMAGE_WIDTH, _IMAGE_HEIGHT, min_ribbon_width_px=8.0
        )
        self.assertIsNotNone(trimmed)
        assert trimmed is not None
        trimmed_mask, trimmed_depth = trimmed

        self.assertLess(int(trimmed_mask.sum()), int(full.sum()))
        self.assertGreater(int(trimmed_mask.sum()), 0)
        # Trimming removes far samples, so the surviving trace is strictly nearer than the full one.
        self.assertLess(float(trimmed_depth[trimmed_mask].max()), float(_depth[full].max()))
        # And it is a subset: nothing new appears.
        self.assertFalse((trimmed_mask & ~full).any())

    def test_width_cutoff_matches_the_geometric_prediction(self) -> None:
        """A 0.4 m ribbon at depth d projects to ``focal * 0.4 / d`` px under this pinhole, so a cutoff of
        ``w`` px should trim at ``focal * 0.4 / w`` m. Pinning this is what makes the knob mean the same thing
        on a camera with a different focal length."""
        cutoff_px = 8.0
        expected_max_depth = self.camera.focal_px * 0.4 / cutoff_px
        result = self.ahead.get_projected_spine_mask(
            _CAMERA_TO_WORLD, self.camera, _IMAGE_WIDTH, _IMAGE_HEIGHT, min_ribbon_width_px=cutoff_px
        )
        assert result is not None
        mask, depth = result
        self.assertAlmostEqual(float(depth[mask].max()), expected_max_depth, delta=1.0)

    def test_width_cutoff_of_zero_traces_everything(self) -> None:
        full, _d = self._spine(self.ahead)
        result = self.ahead.get_projected_spine_mask(
            _CAMERA_TO_WORLD, self.camera, _IMAGE_WIDTH, _IMAGE_HEIGHT, min_ribbon_width_px=0.0
        )
        assert result is not None
        assert_array_equal(result[0], full)

    def test_width_cutoff_ignored_for_a_zero_width_ribbon(self) -> None:
        """A ribbon with no width has no width to measure; the cutoff must not silently erase the whole line."""
        flat = Polyline(np.array([[5.0, 0.0, 0.0], [40.0, 0.0, 0.0]]), half_width_meters=0.0)
        result = flat.get_projected_spine_mask(
            _CAMERA_TO_WORLD, self.camera, _IMAGE_WIDTH, _IMAGE_HEIGHT, min_ribbon_width_px=8.0
        )
        assert result is not None
        self.assertTrue(result[0].any())

    def test_skips_zero_length_segments(self) -> None:
        """Duplicate vertices contribute nothing rather than raising."""
        degenerate = Polyline(np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))
        mask, _depth = self._spine(degenerate)
        self.assertTrue(mask.any())

    def test_a_duplicate_vertex_adds_no_pixel(self) -> None:
        """The ribbon skips a zero-length segment, so the spine must too. Tracing one would add a pixel with no
        counterpart in the area mask, inflating the denominator the spine score divides by."""
        clean = Polyline(np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))
        duplicated = Polyline(np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))
        clean_mask, _d1 = self._spine(clean)
        duplicated_mask, _d2 = self._spine(duplicated)
        assert_array_equal(duplicated_mask, clean_mask)

    def test_a_duplicate_vertex_adds_no_pixel_under_a_width_cutoff(self) -> None:
        """Same invariant on the other code path, which measures the ribbon's projected width."""
        clean = Polyline(np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))
        duplicated = Polyline(np.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))
        args = (_CAMERA_TO_WORLD, self.camera, _IMAGE_WIDTH, _IMAGE_HEIGHT)
        clean_result = clean.get_projected_spine_mask(*args, min_ribbon_width_px=4.0)
        duplicated_result = duplicated.get_projected_spine_mask(*args, min_ribbon_width_px=4.0)
        assert clean_result is not None and duplicated_result is not None
        assert_array_equal(duplicated_result[0], clean_result[0])

    def test_rejects_invalid_pose_and_image_size(self) -> None:
        with self.assertRaises(ValueError):
            self.ahead.get_projected_spine_mask(np.eye(3), self.camera, _IMAGE_WIDTH, _IMAGE_HEIGHT)
        with self.assertRaises(ValueError):
            self.ahead.get_projected_spine_mask(_CAMERA_TO_WORLD, self.camera, 0, _IMAGE_HEIGHT)
        with self.assertRaises(ValueError):
            self.ahead.get_projected_spine_mask(_CAMERA_TO_WORLD, self.camera, _IMAGE_WIDTH, -1)

    def test_surface_has_no_spine(self) -> None:
        """Only geometry whose silhouette is an invented thickness gets a centerline; a surface's extent is real."""
        surface = Surface(np.array([[10.0, -2.0, 0.0], [20.0, -2.0, 0.0], [20.0, 2.0, 0.0], [10.0, 2.0, 0.0]]))
        self.assertIsNone(surface.get_projected_spine_mask(_CAMERA_TO_WORLD, self.camera, _IMAGE_WIDTH, _IMAGE_HEIGHT))


class TestClipSegmentToZRange(unittest.TestCase):
    """Segment clipping against the near/far depth slab."""

    def test_fully_inside_is_unchanged(self) -> None:
        p0 = np.array([0.0, 0.0, 10.0])
        p1 = np.array([1.0, 2.0, 20.0])
        clipped = _clip_segment_to_z_range(p0, p1, 3.0, 50.0)
        self.assertIsNotNone(clipped)
        assert clipped is not None
        assert_array_almost_equal(clipped, np.stack([p0, p1]))

    def test_near_plane_trims_the_start(self) -> None:
        clipped = _clip_segment_to_z_range(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 21.0]), 3.0, 50.0)
        self.assertIsNotNone(clipped)
        assert clipped is not None
        self.assertAlmostEqual(float(clipped[0, 2]), 3.0)
        self.assertAlmostEqual(float(clipped[1, 2]), 21.0)

    def test_far_plane_trims_the_end(self) -> None:
        clipped = _clip_segment_to_z_range(np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, 90.0]), 3.0, 50.0)
        self.assertIsNotNone(clipped)
        assert clipped is not None
        self.assertAlmostEqual(float(clipped[0, 2]), 10.0)
        self.assertAlmostEqual(float(clipped[1, 2]), 50.0)

    def test_receding_segment_clips_on_both_ends(self) -> None:
        """Direction must not matter: a segment running from far to near clips to the same span."""
        clipped = _clip_segment_to_z_range(np.array([0.0, 0.0, 90.0]), np.array([0.0, 0.0, 1.0]), 3.0, 50.0)
        self.assertIsNotNone(clipped)
        assert clipped is not None
        self.assertAlmostEqual(float(clipped[0, 2]), 50.0)
        self.assertAlmostEqual(float(clipped[1, 2]), 3.0)

    def test_entirely_behind_is_rejected(self) -> None:
        self.assertIsNone(_clip_segment_to_z_range(np.array([0.0, 0.0, -5.0]), np.array([0.0, 0.0, 1.0]), 3.0, 50.0))

    def test_entirely_beyond_far_is_rejected(self) -> None:
        self.assertIsNone(_clip_segment_to_z_range(np.array([0.0, 0.0, 60.0]), np.array([0.0, 0.0, 90.0]), 3.0, 50.0))

    def test_constant_depth_inside_is_kept(self) -> None:
        """A segment parallel to the image plane has no parameter to narrow; it is kept or dropped whole."""
        p0 = np.array([-5.0, 0.0, 25.0])
        p1 = np.array([5.0, 0.0, 25.0])
        clipped = _clip_segment_to_z_range(p0, p1, 3.0, 50.0)
        self.assertIsNotNone(clipped)
        assert clipped is not None
        assert_array_almost_equal(clipped, np.stack([p0, p1]))

    def test_constant_depth_outside_is_rejected(self) -> None:
        p0 = np.array([-5.0, 0.0, 70.0])
        p1 = np.array([5.0, 0.0, 70.0])
        self.assertIsNone(_clip_segment_to_z_range(p0, p1, 3.0, 50.0))


class TestPolylineSpinePolylines(unittest.TestCase):
    """The point-list centerline the KPI's delete test measures detection distance against."""

    def setUp(self) -> None:
        self.camera = _PinholeStub()
        self.ahead = Polyline(np.array([[5.0, 0.0, 0.0], [40.0, 0.0, 0.0]]))

    def test_points_lie_on_the_traced_mask(self) -> None:
        # The two representations must agree: a point-list that wandered off the rasterized centerline would
        # attribute detections to a curve the object does not occupy.
        polylines = self.ahead.get_projected_spine_polylines(_CAMERA_TO_WORLD, self.camera)
        self.assertEqual(len(polylines), 1)
        mask, _depth = self.ahead.get_projected_spine_mask(_CAMERA_TO_WORLD, self.camera, _IMAGE_WIDTH, _IMAGE_HEIGHT)
        for x, y in polylines[0]:
            self.assertTrue(mask[round(y), round(x)], f"({x}, {y}) is not on the traced centerline")

    def test_spans_the_visible_extent(self) -> None:
        # A straight line ahead runs from its near end toward the vanishing point, so the projected points must
        # differ: collapsing them would make every detection equidistant and attribution meaningless.
        (points,) = self.ahead.get_projected_spine_polylines(_CAMERA_TO_WORLD, self.camera)
        self.assertGreaterEqual(len(points), 2)
        self.assertGreater(abs(float(points[0][1]) - float(points[-1][1])), 1.0)

    def test_segments_outside_the_range_window_are_dropped(self) -> None:
        # Entirely beyond max_cutoff_distance: nothing is in range, so there is no centerline to match against.
        far = Polyline(np.array([[500.0, 0.0, 0.0], [600.0, 0.0, 0.0]]))
        self.assertEqual(far.get_projected_spine_polylines(_CAMERA_TO_WORLD, self.camera), [])

    def test_disjoint_visible_spans_stay_separate(self) -> None:
        # A line that leaves the range window and re-enters it must not be reported as one polyline: joining the
        # spans would invent a segment across the gap, and a detection sitting on that invention would be
        # credited to this object.
        # Leaves the window at one lateral offset and re-enters at another, so the two clip points differ. A
        # line that doubled back along itself would re-enter where it left and is genuinely continuous.
        out_and_back = Polyline(np.array([[5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [300.0, 150.0, 0.0], [20.0, -4.0, 0.0]]))
        polylines = out_and_back.get_projected_spine_polylines(_CAMERA_TO_WORLD, self.camera)
        self.assertGreater(len(polylines), 1)

    def test_unprojectable_camera_yields_no_polylines(self) -> None:
        camera = unittest.mock.MagicMock()
        camera.ray2pixel_np.side_effect = Exception("Camera projection failed")
        self.assertEqual(self.ahead.get_projected_spine_polylines(_CAMERA_TO_WORLD, camera), [])

    def test_rejects_a_malformed_pose(self) -> None:
        with self.assertRaises(ValueError):
            self.ahead.get_projected_spine_polylines(np.eye(3), self.camera)


if __name__ == "__main__":
    unittest.main()
