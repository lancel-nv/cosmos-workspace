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

import unittest

from checks.utils.object_class_names import OBJECT_CLASS_TO_RDS_TYPES
from kpi.automotive.taxonomy import (
    KINDS,
    Group,
    Signal,
    class_names_in_group,
    defines_signal,
    dynamic_object_types,
    hallucination_label_for,
    kind_of,
    relevancy_is_meaningful,
    types_in_group,
)

# The registry's type sets, pinned explicitly. Membership decides which objects get scored at all, so a
# change here silently moves every metric this library produces; spelling the sets out means such a change
# has to be made deliberately and shows up in review as a diff to an assertion.
FOUR_WHEELER_TYPES = {"Car", "Truck", "Bus", "Vehicle", "Automobile", "Train"}
VRU_TYPES = {"Pedestrian", "Person", "Cyclist", "Rider", "Bicycle", "Motorcycle"}
LANE_GRAPH_TYPES = {"LaneLine", "RoadBoundary"}
FOUR_WHEELER_CLASS_NAMES = {"vehicle"}
VRU_CLASS_NAMES = {"pedestrian", "bicycle", "motorcycle"}


class TestRegistryContents(unittest.TestCase):
    def test_group_membership(self) -> None:
        self.assertEqual(set(types_in_group(Group.FOUR_WHEELER)), FOUR_WHEELER_TYPES)
        self.assertEqual(set(types_in_group(Group.VRU)), VRU_TYPES)
        self.assertEqual(set(types_in_group(Group.LANE_GRAPH)), LANE_GRAPH_TYPES)

    def test_hallucination_class_names_are_derived_from_the_type_sets(self) -> None:
        self.assertEqual(set(class_names_in_group(Group.FOUR_WHEELER)), FOUR_WHEELER_CLASS_NAMES)
        self.assertEqual(set(class_names_in_group(Group.VRU)), VRU_CLASS_NAMES)

    def test_registry_covers_every_dynamic_type_in_the_canonical_class_map(self) -> None:
        """A dynamic type in the canonical map but absent here would be silently dropped from every metric."""
        dynamic_classes = ("vehicle", "pedestrian", "motorcycle", "bicycle")
        mapped_types = {
            rds_type for class_name in dynamic_classes for rds_type in OBJECT_CLASS_TO_RDS_TYPES[class_name]
        }
        self.assertEqual(dynamic_object_types(), frozenset(mapped_types))


class TestHallucinationLabels(unittest.TestCase):
    def test_dynamic_types_do_not_round_trip_to_their_own_name(self) -> None:
        self.assertEqual(hallucination_label_for("Car"), "Vehicle")
        self.assertEqual(hallucination_label_for("Truck"), "Vehicle")
        self.assertEqual(hallucination_label_for("Person"), "Pedestrian")
        self.assertEqual(hallucination_label_for("Bicycle"), "Bicycle")

    def test_ambiguous_types_fall_back_to_the_rds_type(self) -> None:
        """Rider and Cyclist belong to both the motorcycle and bicycle classes; guessing one would misattribute."""
        for rds_type in ("Rider", "Cyclist"):
            self.assertIsNone(KINDS[rds_type].hallucination_label)
            self.assertEqual(hallucination_label_for(rds_type), rds_type)

    def test_lane_graph_types_are_their_own_label(self) -> None:
        self.assertEqual(hallucination_label_for("LaneLine"), "LaneLine")
        self.assertEqual(hallucination_label_for("RoadBoundary"), "RoadBoundary")

    def test_unregistered_type_falls_back_to_itself(self) -> None:
        self.assertEqual(hallucination_label_for("WaitLine"), "WaitLine")


class TestLookups(unittest.TestCase):
    def test_kind_of_returns_none_for_unregistered_and_missing_types(self) -> None:
        self.assertIsNone(kind_of(None))
        self.assertIsNone(kind_of("WaitLine"), "unregistered static types are not silently scored")
        self.assertIsNone(kind_of(""))
        self.assertIsNotNone(kind_of("Car"))

    def test_relevancy_is_meaningful_for_every_dynamic_type(self) -> None:
        for rds_type in dynamic_object_types():
            self.assertTrue(relevancy_is_meaningful(rds_type), rds_type)
        for rds_type in types_in_group(Group.LANE_GRAPH):
            self.assertFalse(relevancy_is_meaningful(rds_type), rds_type)

    def test_four_wheelers_define_the_correspondence_signals_only(self) -> None:
        """Color state belongs to traffic lights; a vehicle has no color to be wrong about."""
        for rds_type in types_in_group(Group.FOUR_WHEELER):
            for signal in (Signal.MISSING, Signal.MISALIGNED, Signal.EXTRA):
                self.assertTrue(defines_signal(rds_type, signal), f"{rds_type} {signal}")
            for signal in (Signal.WRONG_STATE, Signal.CONFLICTING):
                self.assertFalse(defines_signal(rds_type, signal), f"{rds_type} {signal}")

    def test_traffic_lights_define_color_state_but_not_misalignment(self) -> None:
        for signal in (Signal.MISSING, Signal.EXTRA, Signal.WRONG_STATE, Signal.CONFLICTING):
            self.assertTrue(defines_signal("TrafficLight", signal), signal)
        self.assertFalse(defines_signal("TrafficLight", Signal.MISALIGNED))

    def test_vrus_do_not_define_misalignment(self) -> None:
        """Misalignment is n/a for VRUs rather than zero, so nothing may report a rate for it."""
        for rds_type in types_in_group(Group.VRU):
            self.assertTrue(defines_signal(rds_type, Signal.MISSING), rds_type)
            self.assertTrue(defines_signal(rds_type, Signal.EXTRA), rds_type)
            self.assertFalse(defines_signal(rds_type, Signal.MISALIGNED), rds_type)

    def test_lane_graph_defines_existence_and_extra_only(self) -> None:
        """A polyline has no box to be partly outside of, and no color state."""
        for rds_type in types_in_group(Group.LANE_GRAPH):
            self.assertEqual(KINDS[rds_type].mode, "static", rds_type)
            for signal in Signal:
                expected = signal in (Signal.MISSING, Signal.EXTRA)
                self.assertIs(defines_signal(rds_type, signal), expected, f"{rds_type} {signal}")

    def test_unregistered_type_defines_nothing(self) -> None:
        for signal in Signal:
            self.assertFalse(defines_signal("WaitLine", signal), signal)
        self.assertFalse(defines_signal(None, Signal.MISSING))

    def test_every_kind_carries_a_class_name_and_the_mode_that_emits_it(self) -> None:
        for rds_type, kind in KINDS.items():
            self.assertTrue(kind.class_name, rds_type)
            self.assertIn(kind.mode, ("dynamic", "static"), rds_type)
            self.assertEqual(kind.rds_type, rds_type)

    def test_traffic_lights_come_from_the_static_processor(self) -> None:
        """Which processor emits a type decides whether its relevancy field means anything."""
        self.assertEqual(KINDS["TrafficLight"].mode, "static")
        self.assertFalse(relevancy_is_meaningful("TrafficLight"), "the static tracker runs with no road mask")
        self.assertTrue(relevancy_is_meaningful("Car"))


if __name__ == "__main__":
    unittest.main()
