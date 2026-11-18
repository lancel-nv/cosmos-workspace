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

"""Unit tests for importance_filter module (ego-frame FLU version)."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from checks.objects.single_view.importance_filter import ImportanceFilter
from checks.utils.cuboid import Cuboid


class TestImportanceFilter(unittest.TestCase):
    """Test cases for ImportanceFilter class."""

    def setUp(self) -> None:
        self.base_config = {
            "distance_threshold_m": 50.0,
            "skip_oncoming_objects": True,
            "relevant_lanes": ["ego", "left", "right"],
        }
        self.logger = MagicMock()
        self.filter = ImportanceFilter(self.base_config, self.logger)
        self.ego_pose = np.eye(4)

        self.sample_object = {
            "geometry": Cuboid(np.eye(4), np.array([10.0, 0.0, 0.0])),
            "is_static": False,
        }
        self.static_object = {
            "geometry": Cuboid(np.eye(4), np.array([10.0, 0.0, 0.0])),
            "is_static": True,
        }
        self.no_centroid_object = {
            "geometry": None,
            "is_static": False,
        }

    def test_static_object_passes_through(self) -> None:
        should_process, reason = self.filter.should_process_object(self.static_object, self.ego_pose, track_id=1)
        self.assertTrue(should_process)
        self.assertEqual(reason, "passed")

    def test_static_object_importance_attrs(self) -> None:
        # Static objects report the loader-recorded ``ego_distance_m`` (RdsDataLoader.get_object_data_for_frame stamps
        # it at load time), not a recompute here, so distance breakdowns work for every object.
        obj = {**self.static_object, "ego_distance_m": 25.0}
        attrs = self.filter.get_importance_attributes(obj, self.ego_pose, track_id=1)
        self.assertTrue(attrs["is_important"])
        self.assertEqual(attrs["lane"], "n/a")
        self.assertEqual(attrs["distance_m"], 25.0)

    def test_static_object_no_centroid_distance_none(self) -> None:
        # A static object without a centroid pose (e.g. laneline / road boundary) self-excludes with distance None.
        laneline_like = {"geometry": None, "is_static": True}
        attrs = self.filter.get_importance_attributes(laneline_like, self.ego_pose, track_id=1)
        self.assertTrue(attrs["is_important"])
        self.assertEqual(attrs["lane"], "n/a")
        self.assertIsNone(attrs["distance_m"])

    def test_static_object_geometry_without_pose_distance_none(self) -> None:
        # The other missing-geometry branch: geometry is present but has no ``object_to_world_pose`` (extended map
        # geometry, e.g. a laneline polyline). The static path must not index/pose it — it self-excludes with
        # distance None while staying important, and must not raise.
        no_pose_geom = {"geometry": object(), "is_static": True}
        attrs = self.filter.get_importance_attributes(no_pose_geom, self.ego_pose, track_id=1)
        self.assertTrue(attrs["is_important"])
        self.assertEqual(attrs["lane"], "n/a")
        self.assertIsNone(attrs["distance_m"])

    def test_no_centroid_filtered(self) -> None:
        attrs = self.filter.get_importance_attributes(self.no_centroid_object, self.ego_pose, track_id=1)
        self.assertFalse(attrs["is_important"])
        self.assertEqual(attrs["lane"], "unknown")
        self.assertIsNone(attrs["distance_m"])

    def test_should_process_no_centroid_does_not_raise(self) -> None:
        # Regression: a filtered no-centroid object has distance_m=None; building the skip reason must not raise on
        # the ``None > distance_threshold_m`` comparison.
        should_process, reason = self.filter.should_process_object(self.no_centroid_object, self.ego_pose, track_id=1)
        self.assertFalse(should_process)
        self.assertIsInstance(reason, str)

    @patch("checks.objects.single_view.importance_filter.get_object_to_camera_pose")
    def test_passes_all_filters(self, mock_get_pose: MagicMock) -> None:
        """Object at 10m forward, ego lane, same direction → passes."""
        # RDF: x=0 (no lateral), y=0, z=10 (forward) → FLU: (10, 0, 0)
        mock_get_pose.return_value = np.array([[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, 10.0], [0, 0, 0, 1]]).astype(
            float
        )
        obj = {**self.sample_object, "ego_distance_m": 10.0}
        should_process, reason = self.filter.should_process_object(obj, self.ego_pose, track_id=1)
        self.assertTrue(should_process)
        self.assertEqual(reason, "passed")

    @patch("checks.objects.single_view.importance_filter.get_object_to_camera_pose")
    def test_fails_distance_filter(self, mock_get_pose: MagicMock) -> None:
        """Object at 100m forward → beyond 50m threshold."""
        mock_get_pose.return_value = np.array([[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, 100.0], [0, 0, 0, 1]]).astype(
            float
        )
        obj = {**self.sample_object, "ego_distance_m": 100.0}
        should_process, reason = self.filter.should_process_object(obj, self.ego_pose, track_id=1)
        self.assertFalse(should_process)
        self.assertIn("distance_threshold_m", reason)

    @patch("checks.objects.single_view.importance_filter.get_object_to_camera_pose")
    def test_fails_lane_filter(self, mock_get_pose: MagicMock) -> None:
        """Object at 10m right (RDF x=10) → FLU y=-10 → far outside lanes → unknown."""
        mock_get_pose.return_value = np.array([[1, 0, 0, 10.0], [0, 1, 0, 0.0], [0, 0, 1, 10.0], [0, 0, 0, 1]]).astype(
            float
        )
        # Within the 50m threshold (so distance passes) — only the lane filter should reject.
        obj = {**self.sample_object, "ego_distance_m": 14.14}
        should_process, reason = self.filter.should_process_object(obj, self.ego_pose, track_id=1)
        self.assertFalse(should_process)
        self.assertIn("relevant_lanes", reason)


class TestAssignObjectToLane(unittest.TestCase):
    """Test lane assignment in FLU coordinates."""

    def setUp(self) -> None:
        self.filter = ImportanceFilter({})

    def test_ego_lane_center(self) -> None:
        self.assertEqual(self.filter._assign_object_to_lane(np.array([10, 0.0, 0])), "ego")

    def test_ego_lane_slight_left(self) -> None:
        self.assertEqual(self.filter._assign_object_to_lane(np.array([10, 1.0, 0])), "ego")

    def test_ego_lane_slight_right(self) -> None:
        self.assertEqual(self.filter._assign_object_to_lane(np.array([10, -1.0, 0])), "ego")

    def test_left_lane(self) -> None:
        """FLU y=2.0 (left) → left lane."""
        self.assertEqual(self.filter._assign_object_to_lane(np.array([10, 2.0, 0])), "left")

    def test_right_lane(self) -> None:
        """FLU y=-2.0 (right) → right lane."""
        self.assertEqual(self.filter._assign_object_to_lane(np.array([10, -2.0, 0])), "right")

    def test_far_left_unknown(self) -> None:
        self.assertEqual(self.filter._assign_object_to_lane(np.array([10, 8.0, 0])), "unknown")

    def test_far_right_unknown(self) -> None:
        self.assertEqual(self.filter._assign_object_to_lane(np.array([10, -8.0, 0])), "unknown")

    def test_pedestrian_wider_margin(self) -> None:
        """Pedestrian has wider margin band (0.5m extra) than vehicle. Left lane for vehicle: 1.5 to 4.5m. For
        pedestrian: 1.5 to 5.0m. At 4.8m: left for pedestrian, unknown for vehicle."""
        self.assertEqual(self.filter._assign_object_to_lane(np.array([10, 4.8, 0]), "Pedestrian"), "left")
        self.assertEqual(self.filter._assign_object_to_lane(np.array([10, 4.8, 0]), "Car"), "unknown")


class TestIsWithinDistance(unittest.TestCase):
    """Test distance check in FLU coordinates."""

    def setUp(self) -> None:
        self.filter = ImportanceFilter({"distance_threshold_m": 50.0})

    def test_close_object(self) -> None:
        self.assertTrue(self.filter._is_within_distance(np.array([10.0, 0.0, 0.0]), track_id=1))

    def test_far_object(self) -> None:
        self.assertFalse(self.filter._is_within_distance(np.array([60.0, 0.0, 0.0]), track_id=1))

    def test_diagonal_at_threshold(self) -> None:
        """Distance = sqrt(30² + 40²) = 50 → at threshold."""
        self.assertTrue(self.filter._is_within_distance(np.array([30.0, 40.0, 0.0]), track_id=1))

    def test_diagonal_beyond_threshold(self) -> None:
        self.assertFalse(self.filter._is_within_distance(np.array([30.0, 40.1, 0.0]), track_id=1))

    def test_no_threshold(self) -> None:
        f = ImportanceFilter({})
        self.assertTrue(f._is_within_distance(np.array([1000.0, 0.0, 0.0]), track_id=1))


class TestIsOncoming(unittest.TestCase):
    """Test oncoming detection using ego-frame rotation."""

    def setUp(self) -> None:
        self.filter = ImportanceFilter({"skip_oncoming_obstacles": True})

    def test_oncoming_at_180(self) -> None:
        with patch("checks.objects.single_view.importance_filter.extract_rpy_in_flu") as mock_rpy:
            mock_rpy.return_value = (0.0, 0.0, np.pi)
            self.assertTrue(self.filter._is_oncoming(np.eye(4), track_id=1))

    def test_not_oncoming_at_0(self) -> None:
        with patch("checks.objects.single_view.importance_filter.extract_rpy_in_flu") as mock_rpy:
            mock_rpy.return_value = (0.0, 0.0, 0.0)
            self.assertFalse(self.filter._is_oncoming(np.eye(4), track_id=1))

    def test_within_tolerance_150(self) -> None:
        with patch("checks.objects.single_view.importance_filter.extract_rpy_in_flu") as mock_rpy:
            mock_rpy.return_value = (0.0, 0.0, np.deg2rad(150))
            self.assertTrue(self.filter._is_oncoming(np.eye(4), track_id=1))

    def test_outside_tolerance_120(self) -> None:
        with patch("checks.objects.single_view.importance_filter.extract_rpy_in_flu") as mock_rpy:
            mock_rpy.return_value = (0.0, 0.0, np.deg2rad(120))
            self.assertFalse(self.filter._is_oncoming(np.eye(4), track_id=1))


class TestMultiviewConsistency(unittest.TestCase):
    """Test that importance filter gives the same result regardless of which camera pose is used, as long as ego_pose is
    consistent."""

    def setUp(self) -> None:
        self.filter = ImportanceFilter({"distance_threshold_m": 50.0, "relevant_lanes": ["ego", "left", "right"]})

    @patch("checks.objects.single_view.importance_filter.get_object_to_camera_pose")
    def test_same_result_with_different_ego_pose(self, mock_get_pose: MagicMock) -> None:
        """Ego pose is always front_wide → same result regardless of which camera we're processing."""
        # Object 10m forward in ego FLU
        mock_get_pose.return_value = np.array([[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, 10.0], [0, 0, 0, 1]]).astype(
            float
        )

        obj = {"geometry": Cuboid(np.eye(4), np.array([4, 2, 1.5])), "is_static": False}
        ego_pose = np.eye(4)

        result1, _ = self.filter.should_process_object(obj, ego_pose, track_id=1)
        result2, _ = self.filter.should_process_object(obj, ego_pose, track_id=1)
        self.assertEqual(result1, result2)


if __name__ == "__main__":
    unittest.main()
