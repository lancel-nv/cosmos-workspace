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

"""Unit tests for surface module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from checks.utils.cuboid import Cuboid
from checks.utils.surface import Surface


class TestSurface(unittest.TestCase):
    """Test cases for Surface class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Sample surface boundary vertices (triangle)
        self.sample_vertices = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 5.0, 3.0]])
        self.surface = Surface(self.sample_vertices)

    def test_init(self) -> None:
        """Test Surface initialization with valid inputs."""
        self.assertTrue(np.allclose(self.surface.vertices_world, self.sample_vertices))
        self.assertEqual(self.surface.min_cutoff_distance, 3.0)
        self.assertEqual(self.surface.max_cutoff_distance, 50.0)

    def test_init_with_custom_parameters(self) -> None:
        """Test Surface initialization with custom parameters."""
        surface = Surface(self.sample_vertices, min_cutoff_distance=5.0, max_cutoff_distance=100.0)
        self.assertEqual(surface.min_cutoff_distance, 5.0)
        self.assertEqual(surface.max_cutoff_distance, 100.0)

    def test_init_with_list_input(self) -> None:
        """Test Surface initialization with list input."""
        vertices_list = self.sample_vertices.tolist()
        surface = Surface(vertices_list)
        assert_array_almost_equal(surface.vertices_world, self.sample_vertices)

    def test_init_invalid_shape_1d(self) -> None:
        """Test Surface initialization with 1D array."""
        with self.assertRaises(ValueError):
            Surface(np.array([1.0, 2.0, 3.0]))

    def test_init_invalid_shape_wrong_columns(self) -> None:
        """Test Surface initialization with wrong number of columns."""
        with self.assertRaises(ValueError):
            Surface(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

    def test_init_invalid_shape_too_few_points(self) -> None:
        """Test Surface initialization with too few points (less than 3)."""
        with self.assertRaises(ValueError):
            Surface(np.array([[1.0, 2.0, 3.0]]))

        with self.assertRaises(ValueError):
            Surface(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_init_invalid_cutoff_distances(self) -> None:
        """Test Surface initialization with min >= max cutoff distance."""
        with self.assertRaises(ValueError):
            Surface(self.sample_vertices, min_cutoff_distance=10.0, max_cutoff_distance=5.0)

    def test_init_equal_cutoff_distances(self) -> None:
        """Test Surface initialization with equal cutoff distances."""
        with self.assertRaises(ValueError):
            Surface(self.sample_vertices, min_cutoff_distance=10.0, max_cutoff_distance=10.0)

    @patch("checks.utils.surface.CameraBase.transform_points_np")
    def test_get_projected_mask_basic(self, mock_transform: MagicMock) -> None:
        """Test get_projected_mask with basic surface."""

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
            mask, depth_mask = self.surface.get_projected_mask(
                camera_to_world_pose, mock_camera_model, image_width, image_height
            )

        # Check that mask has correct shape and type
        self.assertEqual(mask.shape, (image_height, image_width))
        self.assertEqual(mask.dtype, bool)

        # Check that depth_mask has correct shape and type
        self.assertEqual(depth_mask.shape, (image_height, image_width))
        self.assertEqual(depth_mask.dtype, np.float32)

        # Check that transform_points_np was called
        self.assertEqual(mock_transform.call_count, 1)

        # Check that ray2pixel_np was called
        self.assertGreaterEqual(mock_camera_model.ray2pixel_np.call_count, 1)

        # Check that fillPoly was called
        self.assertGreaterEqual(mock_fillpoly.call_count, 1)

    @patch("checks.utils.surface.CameraBase.transform_points_np")
    def test_get_projected_mask_behind_camera(self, mock_transform: MagicMock) -> None:
        """Test get_projected_mask when surface is behind camera."""

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

        mask, depth_mask = self.surface.get_projected_mask(
            camera_to_world_pose, mock_camera_model, image_width, image_height
        )

        # Should return all-zeros mask (clipped away)
        expected_mask = np.zeros((image_height, image_width), dtype=bool)
        assert_array_equal(mask, expected_mask)

        # Depth mask should be all inf
        self.assertTrue(np.all(np.isinf(depth_mask)))

    def test_get_projected_mask_invalid_pose_shape(self) -> None:
        """Test get_projected_mask with invalid pose shape."""
        mock_camera_model = MagicMock()
        invalid_pose = np.eye(3)  # 3x3 instead of 4x4

        with self.assertRaises(ValueError):
            self.surface.get_projected_mask(invalid_pose, mock_camera_model, 640, 480)

    def test_get_projected_mask_invalid_image_dimensions(self) -> None:
        """Test get_projected_mask with invalid image dimensions."""
        mock_camera_model = MagicMock()
        camera_to_world_pose = np.eye(4)

        with self.assertRaises(ValueError):
            self.surface.get_projected_mask(camera_to_world_pose, mock_camera_model, 0, 480)

        with self.assertRaises(ValueError):
            self.surface.get_projected_mask(camera_to_world_pose, mock_camera_model, 640, -1)

    @patch("checks.utils.surface.CameraBase.transform_points_np")
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
            mask, depth_mask = self.surface.get_projected_mask(
                rotation_matrix, mock_camera_model, image_width, image_height
            )

        self.assertEqual(mask.shape, (image_height, image_width))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(depth_mask.shape, (image_height, image_width))
        self.assertEqual(depth_mask.dtype, np.float32)

    @patch("checks.utils.surface.CameraBase.transform_points_np")
    def test_get_projected_mask_multiple_vertices(self, mock_transform: MagicMock) -> None:
        """Test get_projected_mask with surface having multiple vertices."""
        # Surface with 4 vertices (quad)
        vertices = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0], [0.0, 5.0, 0.0]])
        surface = Surface(vertices)

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
            mask, depth_mask = surface.get_projected_mask(
                camera_to_world_pose, mock_camera_model, image_width, image_height
            )

        self.assertEqual(mask.shape, (image_height, image_width))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(depth_mask.shape, (image_height, image_width))
        self.assertEqual(depth_mask.dtype, np.float32)
        # Should have called fillPoly once for the entire polygon
        self.assertGreaterEqual(mock_fillpoly.call_count, 1)

    @patch("checks.utils.surface.CameraBase.transform_points_np")
    def test_get_projected_mask_cutoff_distances(self, mock_transform: MagicMock) -> None:
        """Test get_projected_mask respects cutoff distances."""
        # Create surface with custom cutoff distances
        surface = Surface(self.sample_vertices, min_cutoff_distance=5.0, max_cutoff_distance=20.0)

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

        mask, depth_mask = surface.get_projected_mask(
            camera_to_world_pose, mock_camera_model, image_width, image_height
        )

        # Points outside cutoff range should be clipped away
        expected_mask = np.zeros((image_height, image_width), dtype=bool)
        assert_array_equal(mask, expected_mask)

        # Depth mask should be all inf
        self.assertTrue(np.all(np.isinf(depth_mask)))

    @patch("checks.utils.surface.CameraBase.transform_points_np")
    def test_get_projected_mask_cutoff_distances_far_plane(self, mock_transform: MagicMock) -> None:
        """Test get_projected_mask respects far plane cutoff distance."""
        # Create surface with custom cutoff distances
        surface = Surface(self.sample_vertices, min_cutoff_distance=5.0, max_cutoff_distance=20.0)

        def transform_side_effect(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
            result = np.asarray(points).copy()
            # Set z values that are beyond the far plane
            result[:, 2] = 25.0  # Greater than max_cutoff_distance
            return result

        mock_transform.side_effect = transform_side_effect

        mock_camera_model = MagicMock()
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.asarray(pts)[:, :2]

        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        mask, depth_mask = surface.get_projected_mask(
            camera_to_world_pose, mock_camera_model, image_width, image_height
        )

        # Points beyond far plane should be clipped away
        expected_mask = np.zeros((image_height, image_width), dtype=bool)
        assert_array_equal(mask, expected_mask)

        # Depth mask should be all inf
        self.assertTrue(np.all(np.isinf(depth_mask)))

    @patch("checks.utils.surface.CameraBase.transform_points_np")
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
        mask, depth_mask = self.surface.get_projected_mask(
            camera_to_world_pose, mock_camera_model, image_width, image_height
        )

        self.assertEqual(mask.shape, (image_height, image_width))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(depth_mask.shape, (image_height, image_width))
        self.assertEqual(depth_mask.dtype, np.float32)

    @patch("checks.utils.surface.CameraBase.transform_points_np")
    def test_get_projected_mask_clipped_too_few_points(self, mock_transform: MagicMock) -> None:
        """Test get_projected_mask when clipping results in too few points."""

        def transform_side_effect(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
            result = np.asarray(points).copy()
            # Set z values that will result in clipping to < 3 points
            result[:, 2] = 1.0  # Less than min_cutoff_distance, will be clipped
            return result

        mock_transform.side_effect = transform_side_effect

        mock_camera_model = MagicMock()
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.asarray(pts)[:, :2]

        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        mask, depth_mask = self.surface.get_projected_mask(
            camera_to_world_pose, mock_camera_model, image_width, image_height
        )

        # Should return empty mask when clipped polygon has < 3 points
        expected_mask = np.zeros((image_height, image_width), dtype=bool)
        assert_array_equal(mask, expected_mask)

        # Depth mask should be all inf
        self.assertTrue(np.all(np.isinf(depth_mask)))

    def test_vertices_world_immutability(self) -> None:
        """Test that modifying input array doesn't affect surface."""
        vertices = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 5.0, 3.0]])
        surface = Surface(vertices)

        # Modify original array
        vertices[0, 0] = 999.0

        # Surface should have its own copy
        self.assertNotEqual(surface.vertices_world[0, 0], 999.0)
        self.assertEqual(surface.vertices_world[0, 0], 0.0)

    @patch("checks.utils.surface.CameraBase.transform_points_np")
    def test_get_projected_mask_invalid_pixel_coordinates(self, mock_transform: MagicMock) -> None:
        """Test get_projected_mask handles invalid pixel coordinates."""

        def transform_side_effect(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
            result = np.asarray(points).copy()
            result[:, 2] = 10.0
            return result

        mock_transform.side_effect = transform_side_effect

        mock_camera_model = MagicMock()
        # Return invalid pixel coordinates (wrong shape or too few points)
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.array([[100, 200]])  # Only 1 point

        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        mask, depth_mask = self.surface.get_projected_mask(
            camera_to_world_pose, mock_camera_model, image_width, image_height
        )

        # Should return empty mask when projection has < 3 points
        expected_mask = np.zeros((image_height, image_width), dtype=bool)
        assert_array_equal(mask, expected_mask)

        # Depth mask should be all inf
        self.assertTrue(np.all(np.isinf(depth_mask)))

    @patch("checks.utils.surface.CameraBase.transform_points_np")
    def test_get_projected_mask_coordinate_clipping(self, mock_transform: MagicMock) -> None:
        """Test that projected coordinates are clipped to image boundaries."""

        def transform_side_effect(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
            result = np.asarray(points).copy()
            result[:, 2] = 10.0
            return result

        mock_transform.side_effect = transform_side_effect

        mock_camera_model = MagicMock()
        # Return coordinates outside image bounds
        mock_camera_model.ray2pixel_np.side_effect = lambda pts: np.array(
            [[-100, -200], [1000, 1000], [320, 240]]  # Some outside, some inside
        )

        camera_to_world_pose = np.eye(4)
        image_width, image_height = 640, 480

        with patch("checks.utils.rasterization.cv2.fillPoly") as mock_fillpoly:
            mask, depth_mask = self.surface.get_projected_mask(
                camera_to_world_pose, mock_camera_model, image_width, image_height
            )

        # Should still execute without error (clipping happens internally)
        self.assertEqual(mask.shape, (image_height, image_width))
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(depth_mask.shape, (image_height, image_width))
        self.assertEqual(depth_mask.dtype, np.float32)
        # Check that fillPoly was called with clipped coordinates
        if mock_fillpoly.called:
            call_args = mock_fillpoly.call_args
            if call_args and len(call_args[0]) > 1:
                pts_int = call_args[0][1][0]  # Second arg is [pts_int], so [1][0] gets pts_int
                # Coordinates should be clipped to [0, width-1] and [0, height-1]
                self.assertTrue(np.all(pts_int[:, 0] >= 0))
                self.assertTrue(np.all(pts_int[:, 0] < image_width))
                self.assertTrue(np.all(pts_int[:, 1] >= 0))
                self.assertTrue(np.all(pts_int[:, 1] < image_height))


class TestSurfaceWorldDistance(unittest.TestCase):
    """Tracker-facing world_distance metric (XY-projected ``1 - IoU``)."""

    def test_match_gate_default(self) -> None:
        """Default gate is 0.5 (require IoU >= 0.5) and is exposed both as class and instance attr."""
        self.assertEqual(Surface.MATCH_GATE_METERS, 0.5)
        surface = Surface(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 4.0, 0.0], [0.0, 4.0, 0.0]]))
        self.assertEqual(surface.MATCH_GATE_METERS, 0.5)

    def test_identical_geometry_zero_distance(self) -> None:
        """world_distance to a copy is exactly 0 (IoU == 1)."""
        verts = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 4.0, 0.0], [0.0, 4.0, 0.0]])
        a = Surface(verts)
        b = Surface(np.array(verts))
        self.assertEqual(a.world_distance(b), 0.0)

    def test_partial_overlap_yields_one_minus_iou(self) -> None:
        """Two 4x4 squares offset 0.6 m in y. Intersection 4x3.4=13.6, union 16+16-13.6=18.4 ->
        IoU~=0.7391, cost~=0.2609."""
        verts = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 4.0, 0.0], [0.0, 4.0, 0.0]])
        a = Surface(verts)
        offset = np.array([0.0, 0.6, 0.0])
        b = Surface(verts + offset)
        self.assertAlmostEqual(a.world_distance(b), 1.0 - (13.6 / 18.4), places=6)

    def test_disjoint_polygons_score_one(self) -> None:
        """Polygons with zero overlap return cost 1.0 (IoU == 0)."""
        a = Surface(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))
        b = Surface(np.array([[10.0, 10.0, 0.0], [11.0, 10.0, 0.0], [11.0, 11.0, 0.0], [10.0, 11.0, 0.0]]))
        self.assertAlmostEqual(a.world_distance(b), 1.0, places=6)

    def test_concentric_size_mismatch_uses_iou(self) -> None:
        """
        Same centroid but different areas: IoU = small_area / large_area when the
        small polygon is fully inside the large one.
        """
        small = Surface(np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]))
        large = Surface(np.array([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]]))
        # Intersection = small area (4); union = large area (16); IoU = 4/16 = 0.25; cost = 0.75.
        self.assertAlmostEqual(small.world_distance(large), 0.75, places=6)

    def test_z_is_ignored(self) -> None:
        """Polygons identical in XY but at different Z still score 0 (IoU is XY-projected)."""
        verts_xy = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
        a = Surface(np.column_stack([verts_xy, np.zeros(4)]))
        b = Surface(np.column_stack([verts_xy, np.ones(4) * 5.0]))
        self.assertEqual(a.world_distance(b), 0.0)

    def test_cross_type_returns_inf(self) -> None:
        """A non-Surface ``other`` yields +inf so cross-type matches never bind."""
        s = Surface(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 2.0, 0.0]]))
        c = Cuboid(np.eye(4), np.array([1.0, 1.0, 1.0]))
        self.assertEqual(s.world_distance(c), float("inf"))

    def test_self_intersecting_polygon_is_repaired(self) -> None:
        """A bowtie polygon (self-intersecting quad) is repaired by ``buffer(0)``
        and yields a finite cost rather than crashing or scoring +inf.

        Vertices in order ``A->B->C->D = (0,0)->(4,4)->(4,0)->(0,4)`` produce a
        bowtie because edges A->B and C->D cross.  Shapely's ``buffer(0)``
        splits this into two triangles whose union has positive area.
        """
        bowtie = Surface(np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]]))
        valid_quad = Surface(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 4.0, 0.0], [0.0, 4.0, 0.0]]))
        d = bowtie.world_distance(valid_quad)
        self.assertTrue(np.isfinite(d))
        self.assertGreaterEqual(d, 0.0)
        self.assertLessEqual(d, 1.0)

    def test_degenerate_polygon_returns_inf(self) -> None:
        """A zero-area polygon (three collinear points) makes the metric
        return +inf -- there's no meaningful overlap to compare against.

        Same outcome regardless of which side is degenerate, so we test
        both directions."""
        collinear = Surface(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
        valid_quad = Surface(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 4.0, 0.0], [0.0, 4.0, 0.0]]))
        self.assertEqual(collinear.world_distance(valid_quad), float("inf"))
        self.assertEqual(valid_quad.world_distance(collinear), float("inf"))

    def test_metric_is_symmetric(self) -> None:
        """``a.world_distance(b) == b.world_distance(a)`` -- IoU is symmetric
        by construction; this guards against an accidental asymmetric
        re-implementation."""
        a = Surface(np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 4.0, 0.0], [0.0, 4.0, 0.0]]))
        b = Surface(np.array([[2.0, 1.0, 0.0], [6.0, 1.0, 0.0], [6.0, 5.0, 0.0], [2.0, 5.0, 0.0]]))
        self.assertAlmostEqual(a.world_distance(b), b.world_distance(a), places=10)

    def test_rotated_square_iou_uses_true_polygon_overlap(self) -> None:
        """Unit square vs same-area square rotated 45 degrees around the same center.

        The intersection is an irregular octagon with area ``2*sqrt(2) - 2`` and
        the union has area ``4 - 2*sqrt(2)``, so IoU = (2*sqrt(2) - 2) / (4 - 2*sqrt(2)) =
        sqrt(2)/2 ~= 0.7071. ``1 - IoU ~= 0.2929``. An AABB-based overlap metric
        would incorrectly score 1.0 here (axis-aligned bounding boxes
        coincide), so this test guarantees we're using true polygon overlap.
        """
        aligned = Surface(np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]]))
        half_sqrt2 = float(np.sqrt(2.0) / 2.0)
        rotated = Surface(
            np.array(
                [
                    [half_sqrt2, 0.0, 0.0],
                    [0.0, half_sqrt2, 0.0],
                    [-half_sqrt2, 0.0, 0.0],
                    [0.0, -half_sqrt2, 0.0],
                ]
            )
        )
        expected_cost = 1.0 - half_sqrt2  # IoU = sqrt(2)/2
        self.assertAlmostEqual(aligned.world_distance(rotated), expected_cost, places=6)


if __name__ == "__main__":
    unittest.main()
