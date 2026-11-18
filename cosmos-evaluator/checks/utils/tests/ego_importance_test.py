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

"""Tests for the shared ego-importance rules.

These pin the definition every caller agrees on, not just the object check's ``ImportanceFilter``, so they are
stricter than any one caller's own tests need: a change here changes which objects the check reports as
important, and silently changes the verdict anything predicting it arrives at too.
"""

import unittest

from checks.utils.ego_importance import (
    EGO_CORRIDOR_LANES,
    HALF_LANE_WIDTH_M,
    LANE_WIDTH_M,
    ONCOMING_TOLERANCE_DEG,
    PEDESTRIAN_MARGIN_BAND_M,
    assign_object_to_lane,
    is_ego_important,
    is_oncoming,
)


class AssignObjectToLaneTest(unittest.TestCase):
    def test_the_ego_lane_is_half_a_lane_either_side(self) -> None:
        self.assertEqual(assign_object_to_lane(0.0), "ego")
        self.assertEqual(assign_object_to_lane(HALF_LANE_WIDTH_M - 0.01), "ego")
        self.assertEqual(assign_object_to_lane(-(HALF_LANE_WIDTH_M - 0.01)), "ego")

    def test_positive_lateral_is_the_left_lane(self) -> None:
        # Positive lateral is left in FLU; mirroring this would swap every left/right verdict the check
        # reports, invisibly, since the corridor accepts both.
        self.assertEqual(assign_object_to_lane(2.0), "left")
        self.assertEqual(assign_object_to_lane(-2.0), "right")

    def test_beyond_the_adjacent_lane_is_unknown(self) -> None:
        self.assertEqual(assign_object_to_lane(8.0), "unknown")
        self.assertEqual(assign_object_to_lane(-8.0), "unknown")

    def test_the_bands_are_strict_on_both_sides(self) -> None:
        # An object exactly on a boundary is "unknown", not the lane inside or outside it.
        self.assertEqual(assign_object_to_lane(HALF_LANE_WIDTH_M), "unknown")
        self.assertEqual(assign_object_to_lane(LANE_WIDTH_M + HALF_LANE_WIDTH_M), "unknown")

    def test_pedestrians_get_a_wider_outer_band(self) -> None:
        # So someone on the pavement beside the left lane still reads as being in it. A vehicle at the same
        # offset does not: that really is outside the corridor.
        just_inside = LANE_WIDTH_M + HALF_LANE_WIDTH_M + PEDESTRIAN_MARGIN_BAND_M - 0.2
        self.assertEqual(assign_object_to_lane(just_inside, "Pedestrian"), "left")
        self.assertEqual(assign_object_to_lane(just_inside, "Person"), "left")
        self.assertEqual(assign_object_to_lane(just_inside, "Car"), "unknown")

    def test_the_ego_corridor_is_exactly_the_named_lanes(self) -> None:
        # What ``relevant_lanes`` means, stated once: the union of the three named bands and nothing else.
        self.assertEqual(EGO_CORRIDOR_LANES, {"ego", "left", "right"})
        for lateral in (0.0, 3.0, -3.0, 4.4, -4.4):
            with self.subTest(lateral=lateral):
                self.assertIn(assign_object_to_lane(lateral), EGO_CORRIDOR_LANES)
        for lateral in (4.6, -4.6, 20.0):
            with self.subTest(lateral=lateral):
                self.assertNotIn(assign_object_to_lane(lateral), EGO_CORRIDOR_LANES)


class IsOncomingTest(unittest.TestCase):
    def test_head_on_is_oncoming_and_following_is_not(self) -> None:
        self.assertIs(is_oncoming(180.0), True)
        self.assertIs(is_oncoming(0.0), False)

    def test_the_tolerance_bounds_are_inclusive(self) -> None:
        self.assertIs(is_oncoming(180.0 - ONCOMING_TOLERANCE_DEG), True)
        self.assertIs(is_oncoming(180.0 + ONCOMING_TOLERANCE_DEG), True)
        self.assertIs(is_oncoming(180.0 - ONCOMING_TOLERANCE_DEG - 0.1), False)

    def test_crossing_traffic_is_not_oncoming(self) -> None:
        self.assertIs(is_oncoming(90.0), False)
        self.assertIs(is_oncoming(270.0), False)

    def test_both_yaw_conventions_agree(self) -> None:
        # One caller arrives from an extracted Euler yaw in [0, 360), the other from an atan2 in (-180, 180].
        # The two must not disagree about the same heading, so the predicate normalizes before comparing.
        for yaw_deg in (170.0, 190.0, 350.0, 10.0):
            with self.subTest(yaw_deg=yaw_deg):
                self.assertIs(is_oncoming(yaw_deg), is_oncoming(yaw_deg - 360.0))

    def test_a_wider_tolerance_can_be_asked_for(self) -> None:
        self.assertIs(is_oncoming(120.0), False)
        self.assertIs(is_oncoming(120.0, tolerance_deg=60.0), True)


class IsEgoImportantTest(unittest.TestCase):
    def _important(self, **overrides: object) -> bool:
        kwargs: dict[str, object] = {
            "distance_m": 20.0,
            "distance_threshold_m": 100.0,
            "lane": "ego",
            "relevant_lanes": EGO_CORRIDOR_LANES,
            "oncoming": False,
        }
        kwargs.update(overrides)
        return is_ego_important(**kwargs)  # type: ignore[arg-type]  # kwargs dict erases the per-key types

    def test_all_three_conditions_met(self) -> None:
        self.assertIs(self._important(), True)

    def test_each_condition_alone_is_enough_to_fail(self) -> None:
        self.assertIs(self._important(distance_m=200.0), False)
        self.assertIs(self._important(lane="unknown"), False)
        self.assertIs(self._important(oncoming=True), False)

    def test_the_distance_bound_is_inclusive(self) -> None:
        self.assertIs(self._important(distance_m=100.0), True)
        self.assertIs(self._important(distance_m=100.01), False)

    def test_an_unknown_distance_fails_closed(self) -> None:
        # Objects with no centroid (lanelines, road boundaries) carry no distance. A missing measurement must
        # not be waved through as "within range" - it is not evidence of anything.
        self.assertIs(self._important(distance_m=None), False)
        self.assertIs(self._important(distance_m=None, distance_threshold_m=float("inf")), False)

    def test_empty_relevant_lanes_means_every_lane(self) -> None:
        # How a config that omits the key disables the lane test, rather than rejecting everything.
        self.assertIs(self._important(lane="unknown", relevant_lanes=[]), True)


if __name__ == "__main__":
    unittest.main()
