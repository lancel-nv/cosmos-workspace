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

import dataclasses
import json
import unittest

from kpi.automotive.policy import DASHED_LANE_LINE, default_policy
from kpi.automotive.result_view import ImportancePolicy
from kpi.automotive.rules import DrawnEmpty, MeanBelow, PercentileBetween, RelevancyGate
from kpi.automotive.taxonomy import Group, dynamic_object_types, types_in_group


class TestDefaultDynamicPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = default_policy(ImportancePolicy.EGO_RELEVANT)

    def test_every_dynamic_type_uses_the_drawn_empty_existence_rule(self) -> None:
        """Existence is "the box was drawn empty", not a tunable threshold, for every dynamic type."""
        for rds_type in dynamic_object_types():
            self.assertEqual(self.policy.existence_rule(rds_type), DrawnEmpty(atol=1e-8), rds_type)

    def test_four_wheelers_carry_the_misalignment_rule(self) -> None:
        for rds_type in types_in_group(Group.FOUR_WHEELER):
            rule = self.policy.misalignment_rule(rds_type)
            self.assertEqual(rule, PercentileBetween(percentile=85.0, lower=0.0, upper=0.35), rds_type)

    def test_vrus_have_no_misalignment_rule(self) -> None:
        for rds_type in types_in_group(Group.VRU):
            self.assertIsNone(self.policy.misalignment_rule(rds_type), rds_type)

    def test_every_dynamic_type_gates_hallucinations_on_one_on_road_frame(self) -> None:
        for rds_type in dynamic_object_types():
            self.assertEqual(self.policy.hallucination_rule(rds_type), RelevancyGate(min_relevancy=1), rds_type)

    def test_unregistered_and_missing_types_resolve_to_no_rule(self) -> None:
        for object_type in ("WaitLine", None):
            self.assertIsNone(self.policy.existence_rule(object_type))
            self.assertIsNone(self.policy.misalignment_rule(object_type))
            self.assertIsNone(self.policy.hallucination_rule(object_type))

    def test_the_importance_policy_is_carried_not_defaulted(self) -> None:
        self.assertIs(self.policy.importance, ImportancePolicy.EGO_RELEVANT)
        self.assertIs(default_policy(ImportancePolicy.ALL_SCORED).importance, ImportancePolicy.ALL_SCORED)


class TestDefaultLaneGraphPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = default_policy(ImportancePolicy.ALL_SCORED)

    def test_lane_graph_existence_is_a_mean_cutoff(self) -> None:
        for rds_type in types_in_group(Group.LANE_GRAPH):
            self.assertEqual(self.policy.existence_rule(rds_type), MeanBelow(threshold=0.1), rds_type)

    def test_a_dashed_lane_line_takes_the_dashed_cutoff(self) -> None:
        self.assertEqual(self.policy.existence_rule("LaneLine", "LONG_DASHED_SINGLE"), MeanBelow(threshold=0.002))
        self.assertEqual(self.policy.existence_rule("LaneLine", "short_dashed_single"), MeanBelow(threshold=0.002))
        for style in ("SOLID_SINGLE", "SOLID_DASHED", "DASHED_SOLID", "OTHER", None):
            self.assertEqual(self.policy.existence_rule("LaneLine", style), MeanBelow(threshold=0.1), style)

    def test_style_moves_no_other_type(self) -> None:
        self.assertEqual(self.policy.existence_rule("RoadBoundary", "LONG_DASHED_SINGLE"), MeanBelow(threshold=0.1))
        self.assertEqual(self.policy.existence_rule("Car", "LONG_DASHED_SINGLE"), DrawnEmpty(atol=1e-8))

    def test_without_a_dashed_rule_a_dashed_line_takes_the_lane_line_rule(self) -> None:
        stripped = dataclasses.replace(
            self.policy, existence={k: v for k, v in self.policy.existence.items() if k != DASHED_LANE_LINE}
        )
        self.assertEqual(stripped.existence_rule("LaneLine", "LONG_DASHED_SINGLE"), MeanBelow(threshold=0.1))

    def test_lane_graph_has_no_misalignment_rule_and_keeps_every_hallucination(self) -> None:
        for rds_type in types_in_group(Group.LANE_GRAPH):
            self.assertIsNone(self.policy.misalignment_rule(rds_type), rds_type)
            self.assertEqual(self.policy.hallucination_rule(rds_type), RelevancyGate(min_relevancy=1), rds_type)

    def test_as_dict_records_both_lane_line_cutoffs(self) -> None:
        payload = self.policy.as_dict()
        self.assertEqual(payload["existence"]["LaneLine"], {"rule": "mean_below", "threshold": 0.1})
        self.assertEqual(payload["existence"][DASHED_LANE_LINE], {"rule": "mean_below", "threshold": 0.002})
        self.assertEqual(payload["existence"]["RoadBoundary"], {"rule": "mean_below", "threshold": 0.1})


class TestAsDict(unittest.TestCase):
    def test_record_is_json_serializable_and_names_each_rule_shape(self) -> None:
        payload = default_policy(ImportancePolicy.ALL_SCORED).as_dict()
        json.dumps(payload)
        self.assertEqual(payload["name"], "default")
        self.assertEqual(payload["importance"], "all_scored")
        self.assertEqual(payload["existence"]["Car"], {"rule": "drawn_empty", "atol": 1e-8})
        self.assertEqual(
            payload["misalignment"]["Car"],
            {"rule": "percentile_between", "percentile": 85.0, "lower": 0.0, "upper": 0.35},
        )
        self.assertEqual(payload["hallucination"]["Car"], {"rule": "relevancy_gate", "min_relevancy": 1})

    def test_a_type_without_a_rule_is_absent_rather_than_recorded_as_none(self) -> None:
        payload = default_policy(ImportancePolicy.EGO_RELEVANT).as_dict()
        self.assertIn("Car", payload["misalignment"])
        self.assertNotIn("Pedestrian", payload["misalignment"])

    def test_a_caller_defined_rule_stamps_its_own_parameters(self) -> None:
        """The provenance record must describe a rule this library did not define, not flatten it."""
        base = default_policy(ImportancePolicy.EGO_RELEVANT)
        swept = dataclasses.replace(
            base,
            name="sweep-0.5",
            misalignment={**base.misalignment, "Car": PercentileBetween(percentile=85.0, lower=0.0, upper=0.5)},
        )
        payload = swept.as_dict()
        self.assertEqual(payload["name"], "sweep-0.5")
        self.assertEqual(payload["misalignment"]["Car"]["upper"], 0.5)
        self.assertEqual(payload["misalignment"]["Truck"]["upper"], 0.35, "the variant changed one type only")


if __name__ == "__main__":
    unittest.main()
