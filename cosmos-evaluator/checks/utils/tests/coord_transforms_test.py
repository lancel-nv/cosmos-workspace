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

"""Unit tests for coord_transforms module."""

import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from checks.utils.coord_transforms import (
    clip_polygon_to_z_planes,
    extract_rpy_in_flu,
    get_object_to_camera_pose,
    rdf_pose_to_flu,
    rdf_position_to_flu,
    world_position_to_ego_flu,
)
from checks.utils.cuboid import Cuboid


class TestCoordTransforms(unittest.TestCase):
    """Test cases for coordinate transformation utilities."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Sample 4x4 transformation matrices
        self.sample_object_to_world = np.array(
            [[1.0, 0.0, 0.0, 5.0], [0.0, 1.0, 0.0, 10.0], [0.0, 0.0, 1.0, 2.0], [0.0, 0.0, 0.0, 1.0]]
        )

        self.sample_camera_pose = np.array(
            [[0.0, -1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 1.0, 1.5], [0.0, 0.0, 0.0, 1.0]]
        )

    def test_get_object_to_camera_pose_basic(self) -> None:
        """Test basic object to camera pose transformation."""
        tracked_object = Cuboid(self.sample_object_to_world, np.array([1.0, 1.0, 1.0]))

        result = get_object_to_camera_pose(tracked_object, self.sample_camera_pose)

        # Verify result is a 4x4 matrix
        self.assertEqual(result.shape, (4, 4))

        # Verify it's a valid transformation matrix (last row should be [0,0,0,1])
        assert_array_almost_equal(result[3, :], [0, 0, 0, 1])

    def test_get_object_to_camera_pose_identity(self) -> None:
        """Test object to camera pose with identity matrices."""
        identity = np.eye(4)
        tracked_object = Cuboid(identity, np.array([1.0, 1.0, 1.0]))

        result = get_object_to_camera_pose(tracked_object, identity)

        # With identity matrices, result should be identity
        assert_array_almost_equal(result, identity)

    def test_get_object_to_camera_pose_translation_only(self) -> None:
        """Test object to camera pose with translation only."""
        object_translation = np.eye(4)
        object_translation[:3, 3] = [1, 2, 3]

        camera_translation = np.eye(4)
        camera_translation[:3, 3] = [0.5, 1.0, 1.5]

        tracked_object = Cuboid(object_translation, np.array([1.0, 1.0, 1.0]))

        result = get_object_to_camera_pose(tracked_object, camera_translation)

        # Expected translation should be object - camera
        expected_translation = [0.5, 1.0, 1.5]  # [1-0.5, 2-1.0, 3-1.5]
        assert_array_almost_equal(result[:3, 3], expected_translation)

    def test_extract_rpy_in_flu_identity(self) -> None:
        """Test RPY extraction from identity matrix."""
        identity = np.eye(3)
        roll, pitch, yaw = extract_rpy_in_flu(identity)

        # Identity matrix should give zero rotations
        self.assertAlmostEqual(roll, 0.0, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)
        self.assertAlmostEqual(yaw, 0.0, places=6)

    def test_extract_rpy_in_flu_rotation_x(self) -> None:
        """Test RPY extraction from rotation around X-axis (roll)."""
        angle = np.pi / 4  # 45 degrees
        R_x = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])

        roll, pitch, yaw = extract_rpy_in_flu(R_x.T)  # Input is R_flu_to_rdf, so we transpose

        self.assertAlmostEqual(roll, angle, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)
        self.assertAlmostEqual(yaw, 0.0, places=6)

    def test_extract_rpy_in_flu_rotation_y(self) -> None:
        """Test RPY extraction from rotation around Y-axis (pitch)."""
        angle = np.pi / 6  # 30 degrees
        R_y = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])

        roll, pitch, yaw = extract_rpy_in_flu(R_y.T)

        self.assertAlmostEqual(roll, 0.0, places=6)
        self.assertAlmostEqual(pitch, angle, places=6)
        self.assertAlmostEqual(yaw, 0.0, places=6)

    def test_extract_rpy_in_flu_rotation_z(self) -> None:
        """Test RPY extraction from rotation around Z-axis (yaw)."""
        angle = np.pi / 3  # 60 degrees
        R_z = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

        roll, pitch, yaw = extract_rpy_in_flu(R_z.T)

        self.assertAlmostEqual(roll, 0.0, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)
        self.assertAlmostEqual(yaw, angle, places=6)

    def test_extract_rpy_in_flu_combined_rotation(self) -> None:
        """Test RPY extraction from combined rotations."""
        roll_angle = np.pi / 6  # 30 degrees
        pitch_angle = np.pi / 8  # 22.5 degrees
        yaw_angle = np.pi / 4  # 45 degrees

        # Create individual rotation matrices
        R_x = np.array(
            [[1, 0, 0], [0, np.cos(roll_angle), -np.sin(roll_angle)], [0, np.sin(roll_angle), np.cos(roll_angle)]]
        )

        R_y = np.array(
            [[np.cos(pitch_angle), 0, np.sin(pitch_angle)], [0, 1, 0], [-np.sin(pitch_angle), 0, np.cos(pitch_angle)]]
        )

        R_z = np.array(
            [[np.cos(yaw_angle), -np.sin(yaw_angle), 0], [np.sin(yaw_angle), np.cos(yaw_angle), 0], [0, 0, 1]]
        )

        # Combined rotation (ZYX order)
        R_combined = R_z @ R_y @ R_x

        roll, pitch, yaw = extract_rpy_in_flu(R_combined.T)

        # Should approximately match original angles
        self.assertAlmostEqual(roll, roll_angle, places=5)
        self.assertAlmostEqual(pitch, pitch_angle, places=5)
        self.assertAlmostEqual(yaw, yaw_angle, places=5)

    def test_extract_rpy_in_flu_gimbal_lock_positive(self) -> None:
        """Test RPY extraction in positive gimbal lock condition."""
        # Create a rotation matrix with pitch = 90 degrees (gimbal lock)
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

        _roll, pitch, yaw = extract_rpy_in_flu(R.T)

        # In gimbal lock, pitch should be ±π/2, yaw should be 0
        self.assertAlmostEqual(pitch, np.pi / 2, places=6)
        self.assertAlmostEqual(yaw, 0.0, places=6)

    def test_extract_rpy_in_flu_gimbal_lock_negative(self) -> None:
        """Test RPY extraction in negative gimbal lock condition."""
        # Create a rotation matrix with pitch = -90 degrees (gimbal lock)
        R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])

        _roll, pitch, yaw = extract_rpy_in_flu(R.T)

        # In gimbal lock, pitch should be -π/2, yaw should be 0
        self.assertAlmostEqual(pitch, -np.pi / 2, places=6)
        self.assertAlmostEqual(yaw, 0.0, places=6)

    def test_get_object_to_camera_pose_with_numpy_array(self) -> None:
        """Test object to camera pose when object_to_world is already numpy array."""
        tracked_object = Cuboid(self.sample_object_to_world, np.array([1.0, 1.0, 1.0]))  # numpy array provided

        result = get_object_to_camera_pose(tracked_object, self.sample_camera_pose)

        # Should work with both numpy arrays and lists
        self.assertEqual(result.shape, (4, 4))
        assert_array_almost_equal(result[3, :], [0, 0, 0, 1])

    def test_extract_rpy_large_angles(self) -> None:
        """Test RPY extraction with large angles."""
        # Test with angles greater than π
        large_angle = 3 * np.pi / 2
        R_z = np.array(
            [[np.cos(large_angle), -np.sin(large_angle), 0], [np.sin(large_angle), np.cos(large_angle), 0], [0, 0, 1]]
        )

        _roll, _pitch, yaw = extract_rpy_in_flu(R_z.T)

        # The extracted angle should be within [-π, π]
        self.assertGreaterEqual(yaw, -np.pi)
        self.assertLessEqual(yaw, np.pi)

        # The rotation should be equivalent
        expected_angle = large_angle - 2 * np.pi  # Normalize to [-π, π]
        self.assertAlmostEqual(yaw, expected_angle, places=6)

    # -------------------------------
    # clip_polygon_to_z_planes tests
    # -------------------------------
    def test_clip_polygon_no_planes_returns_copy(self) -> None:
        """Polygon should be unchanged when both planes are None."""
        poly = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
        out = clip_polygon_to_z_planes(poly, near_plane_z=None, far_plane_z=None)
        # Should return a copy, not the same object
        self.assertIsNot(poly, out)
        assert_array_almost_equal(out, poly)

    def test_clip_polygon_empty_returns_empty(self) -> None:
        """Empty input should return empty output."""
        poly = np.zeros((0, 3))
        out = clip_polygon_to_z_planes(poly)
        self.assertEqual(out.shape, (0, 3))

    def test_clip_polygon_invalid_shape_raises(self) -> None:
        """Invalid shapes should raise ValueError."""
        with self.assertRaises(ValueError):
            clip_polygon_to_z_planes(np.array([1.0, 2.0, 3.0]))
        with self.assertRaises(ValueError):
            clip_polygon_to_z_planes(np.array([[0.0, 0.0], [1.0, 1.0]]))

    def test_clip_polygon_all_inside_near(self) -> None:
        """Polygon fully in front of near plane should remain unchanged."""
        poly = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        out = clip_polygon_to_z_planes(poly, near_plane_z=1.0, far_plane_z=None)
        assert_array_almost_equal(out, poly)

    def test_clip_polygon_all_outside_near(self) -> None:
        """Polygon fully behind near plane should be clipped away."""
        poly = np.array([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [1.0, 1.0, 0.5], [0.0, 1.0, 0.5]])
        out = clip_polygon_to_z_planes(poly, near_plane_z=1.0, far_plane_z=None)
        self.assertEqual(out.shape, (0, 3))

    def test_clip_polygon_near_plane_intersection(self) -> None:
        """Polygon partially behind near plane should be clipped with correct intersections."""
        # Two vertices behind near plane z=1.0 (z=0.5) and two in front (z=2.0)
        p0 = np.array([0.0, 0.0, 0.5])
        p1 = np.array([1.0, 0.0, 0.5])
        p2 = np.array([1.0, 1.0, 2.0])
        p3 = np.array([0.0, 1.0, 2.0])
        poly = np.stack([p0, p1, p2, p3], axis=0)

        out = clip_polygon_to_z_planes(poly, near_plane_z=1.0, far_plane_z=None)
        # Expect 4 vertices after clipping:
        # intersections on edges (p3->p0) and (p1->p2), plus kept p2, p3
        self.assertEqual(out.shape[1], 3)
        self.assertEqual(out.shape[0], 4)
        # All points must satisfy z >= 1.0
        self.assertTrue(np.all(out[:, 2] >= 1.0 - 1e-9))

        # Expected intersection points at z=1
        inter_a = np.array([0.0, 1.0 / 3.0, 1.0])  # on edge p3->p0
        inter_b = np.array([1.0, 1.0 / 3.0, 1.0])  # on edge p1->p2

        # Check that p2 and p3 are present, and intersections present (order-independent)
        def contains_point(points: np.ndarray, pt: np.ndarray, tol: float = 1e-6) -> np.bool_:
            return np.any(np.all(np.isclose(points, pt, atol=tol), axis=1))

        self.assertTrue(contains_point(out, p2))
        self.assertTrue(contains_point(out, p3))
        self.assertTrue(contains_point(out, inter_a))
        self.assertTrue(contains_point(out, inter_b))

    def test_clip_polygon_far_plane_intersection(self) -> None:
        """Polygon partially beyond far plane should be clipped with correct intersections."""
        # Two vertices beyond far plane z=1.0 (z=2.0) and two inside (z=0.5)
        p0 = np.array([0.0, 0.0, 2.0])
        p1 = np.array([1.0, 0.0, 2.0])
        p2 = np.array([1.0, 1.0, 0.5])
        p3 = np.array([0.0, 1.0, 0.5])
        poly = np.stack([p0, p1, p2, p3], axis=0)

        out = clip_polygon_to_z_planes(poly, near_plane_z=None, far_plane_z=1.0)
        self.assertEqual(out.shape[1], 3)
        self.assertEqual(out.shape[0], 4)
        # All points must satisfy z <= 1.0
        self.assertTrue(np.all(out[:, 2] <= 1.0 + 1e-9))

        # Expected intersection points at z=1 on edges (p0->p3) and (p1->p2)
        inter_a = np.array([0.0, 2.0 / 3.0, 1.0])
        inter_b = np.array([1.0, 2.0 / 3.0, 1.0])

        def contains_point(points: np.ndarray, pt: np.ndarray, tol: float = 1e-6) -> np.bool_:
            return np.any(np.all(np.isclose(points, pt, atol=tol), axis=1))

        self.assertTrue(contains_point(out, p2))
        self.assertTrue(contains_point(out, p3))
        self.assertTrue(contains_point(out, inter_a))
        self.assertTrue(contains_point(out, inter_b))

    def test_clip_polygon_both_planes(self) -> None:
        """Clipping against both planes should keep only points within the slab."""
        # Polygon with z values spanning from -1 to 3
        poly = np.array(
            [
                [-1.0, 0.0, -1.0],
                [1.0, 0.0, 0.5],
                [1.0, 1.0, 2.0],
                [-1.0, 1.0, 3.0],
            ]
        )
        out = clip_polygon_to_z_planes(poly, near_plane_z=0.0, far_plane_z=2.0)
        # All z must be within [0, 2]
        self.assertTrue(np.all(out[:, 2] >= -1e-9))
        self.assertTrue(np.all(out[:, 2] <= 2.0 + 1e-9))
        # Should have at least 3 vertices to form a polygon
        self.assertGreaterEqual(out.shape[0], 3)

    def test_clip_polygon_vertex_on_plane_no_duplicates(self) -> None:
        """Vertices exactly on plane should be handled without duplicate consecutive points."""
        # One vertex exactly on near plane z=1.0, others in front
        poly = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.5], [1.0, 1.0, 1.5], [0.0, 1.0, 1.5]])
        out = clip_polygon_to_z_planes(poly, near_plane_z=1.0, far_plane_z=None)
        # Should retain a valid polygon with no consecutive duplicate rows
        self.assertGreaterEqual(out.shape[0], 3)
        if out.shape[0] >= 2:
            diffs = np.linalg.norm(out[1:] - out[:-1], axis=1)
            self.assertTrue(np.all(diffs > 1e-12))


class TestRdfPositionToFlu(unittest.TestCase):
    """Test RDF to FLU coordinate conversion.

    Moved here with the function itself, which several call sites — the importance filter among them — each
    used to carry their own copy of.
    """

    def test_identity(self) -> None:
        """RDF (0,0,0) -> FLU (0,0,0)."""
        self.assertTrue(np.allclose(rdf_position_to_flu(np.array([0, 0, 0])), [0, 0, 0]))

    def test_forward(self) -> None:
        """RDF z=10 (forward) -> FLU x=10 (forward)."""
        flu = rdf_position_to_flu(np.array([0.0, 0.0, 10.0]))
        self.assertAlmostEqual(flu[0], 10.0)  # forward
        self.assertAlmostEqual(flu[1], 0.0)  # left
        self.assertAlmostEqual(flu[2], 0.0)  # up

    def test_right(self) -> None:
        """RDF x=5 (right) -> FLU y=-5 (right = negative left)."""
        flu = rdf_position_to_flu(np.array([5.0, 0.0, 0.0]))
        self.assertAlmostEqual(flu[0], 0.0)  # forward
        self.assertAlmostEqual(flu[1], -5.0)  # left (negative = right)
        self.assertAlmostEqual(flu[2], 0.0)  # up

    def test_down(self) -> None:
        """RDF y=3 (down) -> FLU z=-3 (down = negative up)."""
        flu = rdf_position_to_flu(np.array([0.0, 3.0, 0.0]))
        self.assertAlmostEqual(flu[0], 0.0)
        self.assertAlmostEqual(flu[1], 0.0)
        self.assertAlmostEqual(flu[2], -3.0)

    def test_combined(self) -> None:
        """RDF (2, 1, 10) -> FLU (10, -2, -1)."""
        flu = rdf_position_to_flu(np.array([2.0, 1.0, 10.0]))
        np.testing.assert_allclose(flu, [10.0, -2.0, -1.0])


class TestWorldPositionToEgoFlu(unittest.TestCase):
    """The world -> ego-FLU chain every ego-relative measurement is built on."""

    @staticmethod
    def _ego_pose(x: float = 0.0, y: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
        """Camera-to-world pose at ``(x, y)`` facing ``yaw_deg``, in RDF (right, down, forward) column order."""
        angle = np.radians(yaw_deg)
        forward = np.array([np.cos(angle), np.sin(angle), 0.0])
        right = np.array([np.sin(angle), -np.cos(angle), 0.0])
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = [0.0, 0.0, -1.0]
        pose[:3, 2] = forward
        pose[:3, 3] = [x, y, 0.0]
        return pose

    def test_a_point_ahead_is_pure_forward(self) -> None:
        flu = world_position_to_ego_flu(np.array([10.0, 0.0, 0.0]), self._ego_pose())
        assert_array_almost_equal(flu, [10.0, 0.0, 0.0])

    def test_left_of_the_ego_is_positive_lateral(self) -> None:
        # The sign that decides every left/right lane verdict downstream.
        flu = world_position_to_ego_flu(np.array([10.0, 3.5, 0.0]), self._ego_pose())
        assert_array_almost_equal(flu, [10.0, 3.5, 0.0])

    def test_the_result_is_relative_to_a_rotated_ego(self) -> None:
        # World axes are not ego axes: 10 m along +Y is straight ahead of an ego facing +Y, not beside it.
        flu = world_position_to_ego_flu(np.array([0.0, 10.0, 0.0]), self._ego_pose(yaw_deg=90.0))
        assert_array_almost_equal(flu, [10.0, 0.0, 0.0])

    def test_the_result_is_relative_to_a_translated_ego(self) -> None:
        flu = world_position_to_ego_flu(np.array([30.0, 0.0, 0.0]), self._ego_pose(x=20.0))
        assert_array_almost_equal(flu, [10.0, 0.0, 0.0])

    def test_height_reads_as_up(self) -> None:
        flu = world_position_to_ego_flu(np.array([10.0, 0.0, 2.0]), self._ego_pose())
        assert_array_almost_equal(flu, [10.0, 0.0, 2.0])

    def test_a_point_behind_the_ego_reads_negative_forward(self) -> None:
        flu = world_position_to_ego_flu(np.array([-10.0, 0.0, 0.0]), self._ego_pose())
        assert_array_almost_equal(flu, [-10.0, 0.0, 0.0])

    def test_a_homogeneous_input_is_accepted(self) -> None:
        flu = world_position_to_ego_flu(np.array([10.0, 0.0, 0.0, 1.0]), self._ego_pose())
        assert_array_almost_equal(flu, [10.0, 0.0, 0.0])


class TestRdfPoseToFlu(unittest.TestCase):
    """The pose-level sibling: the same convention applied to a transform's columns."""

    def test_columns_are_relabeled_and_translation_kept(self) -> None:
        pose = np.array(
            [
                [1.0, 2.0, 3.0, 10.0],
                [4.0, 5.0, 6.0, 20.0],
                [7.0, 8.0, 9.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        expected = np.array(
            [
                [3.0, -1.0, -2.0, 10.0],
                [6.0, -4.0, -5.0, 20.0],
                [9.0, -7.0, -8.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        assert_array_almost_equal(rdf_pose_to_flu(pose), expected)

    def test_an_identity_rdf_pose_becomes_the_flu_basis(self) -> None:
        # RDF identity means right=+x, down=+y, forward=+z; as FLU that is forward=+z, left=-x, up=-y.
        flu = rdf_pose_to_flu(np.eye(4))
        assert_array_almost_equal(flu[:3, 0], [0.0, 0.0, 1.0])
        assert_array_almost_equal(flu[:3, 1], [-1.0, 0.0, 0.0])
        assert_array_almost_equal(flu[:3, 2], [0.0, -1.0, 0.0])

    def test_relabeling_preserves_a_rotation(self) -> None:
        # Column swaps and sign flips chosen so the result stays right-handed: a relabeled pose is still a
        # pose, and code that inverts it would otherwise be inverting a reflection.
        angle = np.radians(37.0)
        pose = np.eye(4)
        pose[:3, :3] = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rotation = rdf_pose_to_flu(pose)[:3, :3]
        assert_array_almost_equal(rotation @ rotation.T, np.eye(3))
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)

    def test_inverting_a_relabeled_pose_is_the_world_to_ego_chain(self) -> None:
        # Why this is not the way to put a world point in the ego frame: relabeling then inverting cancels
        # into world_position_to_ego_flu exactly. Two call sites had taken that long way round, so the
        # equivalence is pinned here rather than left as a comment.
        angle = np.radians(-52.0)
        pose = np.eye(4)
        pose[:3, 0] = [np.sin(angle), -np.cos(angle), 0.0]
        pose[:3, 1] = [0.0, 0.0, -1.0]
        pose[:3, 2] = [np.cos(angle), np.sin(angle), 0.0]
        pose[:3, 3] = [12.0, -7.0, 1.5]
        point = np.array([31.0, 4.0, 2.5])

        the_long_way = np.linalg.inv(rdf_pose_to_flu(pose)) @ np.append(point, 1.0)
        assert_array_almost_equal(the_long_way[:3], world_position_to_ego_flu(point, pose))


if __name__ == "__main__":
    unittest.main()
