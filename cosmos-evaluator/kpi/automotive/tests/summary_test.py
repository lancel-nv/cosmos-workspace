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

from kpi.automotive.policy import default_policy
from kpi.automotive.result_view import ImportancePolicy, read_result
from kpi.automotive.summary import ObjectSignals, merge_summaries, signal_summary
from kpi.automotive.tests.fixtures import load_dynamic_result


def _summary(importance: ImportancePolicy) -> dict[str, ObjectSignals]:
    view = read_result(load_dynamic_result(), importance=importance)
    return signal_summary(view, default_policy(importance))


class TestSignalSummary(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = _summary(ImportancePolicy.EGO_RELEVANT)

    def test_counts_present_missing_and_unscored_separately(self) -> None:
        car = self.rows["Car"]
        self.assertEqual((car.observed, car.present, car.missing), (2, 1, 1))
        self.assertEqual(car.unscored, 1, "the never-important car is coverage, not a missing object")

    def test_unscored_tracks_are_excluded_from_the_rate_denominator(self) -> None:
        car = self.rows["Car"]
        self.assertEqual(car.observed, car.present + car.missing)
        self.assertAlmostEqual(car.rates()["missing_rate"] or 0.0, 0.5)

    def test_misalignment_is_counted_where_a_question_governs_it(self) -> None:
        self.assertEqual(self.rows["Truck"].misaligned, 1)
        self.assertEqual(self.rows["Car"].misaligned, 0)

    def test_misalignment_is_none_for_vrus_rather_than_zero(self) -> None:
        pedestrian = self.rows["Pedestrian"]
        self.assertIsNone(pedestrian.misaligned, "S2 asks no misalignment question, so nothing was measured")
        self.assertIsNone(pedestrian.rates()["misaligned_rate"])

    def test_a_type_seen_only_as_unscored_reports_no_rates(self) -> None:
        bus = self.rows["Bus"]
        self.assertEqual((bus.observed, bus.unscored), (0, 1))
        self.assertIsNone(bus.rates()["missing_rate"], "a rate over zero observations is unmeasured, not 0.0")

    def test_hallucinations_land_under_the_detector_class_label(self) -> None:
        """A hallucination has no ground-truth object to take an RDS type from, so it keys by class label."""
        self.assertEqual(self.rows["Vehicle"].hallucinated, 1)
        self.assertEqual(self.rows["Pedestrian"].hallucinated, 1)
        self.assertEqual(self.rows["Car"].hallucinated, 0)

    def test_a_hallucination_only_row_reports_no_correspondence_counts(self) -> None:
        vehicle = self.rows["Vehicle"]
        self.assertEqual((vehicle.observed, vehicle.unscored, vehicle.hallucinated), (0, 0, 1))

    def test_unregistered_types_are_skipped_entirely(self) -> None:
        result = load_dynamic_result()
        result["tracks"].append({"track_id": 8, "object_type": "WaitLine"})
        view = read_result(result, importance=ImportancePolicy.EGO_RELEVANT)
        rows = signal_summary(view, default_policy(ImportancePolicy.EGO_RELEVANT))
        self.assertNotIn("WaitLine", rows, "a static type has no rule here and must not be scored by proxy")

    def test_as_dict_is_serializable_and_carries_the_rates(self) -> None:
        payload = self.rows["Truck"].as_dict()
        self.assertEqual(payload["object_type"], "Truck")
        self.assertEqual(payload["misaligned"], 1)
        self.assertAlmostEqual(payload["present_rate"], 1.0)


class TestImportancePolicyChangesTheTable(unittest.TestCase):
    def test_the_never_important_car_moves_from_unscored_to_present(self) -> None:
        """The same file, the same rules, a different population - which is why the policy is never defaulted."""
        ego = _summary(ImportancePolicy.EGO_RELEVANT)["Car"]
        every = _summary(ImportancePolicy.ALL_SCORED)["Car"]
        self.assertEqual((ego.observed, ego.present, ego.unscored), (2, 1, 1))
        self.assertEqual((every.observed, every.present, every.unscored), (3, 2, 0))
        self.assertAlmostEqual(ego.rates()["missing_rate"] or 0.0, 0.5)
        self.assertAlmostEqual(every.rates()["missing_rate"] or 0.0, 1 / 3)


class TestMergeSummaries(unittest.TestCase):
    def test_counts_accumulate_across_clips(self) -> None:
        target = dict(_summary(ImportancePolicy.EGO_RELEVANT))
        merge_summaries(target, _summary(ImportancePolicy.EGO_RELEVANT))
        self.assertEqual(target["Car"].observed, 4)
        self.assertEqual(target["Car"].missing, 2)
        self.assertEqual(target["Vehicle"].hallucinated, 2)

    def test_a_type_absent_from_the_target_is_adopted(self) -> None:
        target: dict[str, ObjectSignals] = {}
        merge_summaries(target, _summary(ImportancePolicy.EGO_RELEVANT))
        self.assertEqual(target["Truck"].misaligned, 1)

    def test_unmeasured_misalignment_stays_unmeasured(self) -> None:
        target = {"Pedestrian": ObjectSignals(object_type="Pedestrian", observed=1, misaligned=None)}
        merge_summaries(target, {"Pedestrian": ObjectSignals(object_type="Pedestrian", observed=1, misaligned=None)})
        self.assertIsNone(target["Pedestrian"].misaligned)

    def test_a_measured_side_makes_the_merged_row_measured(self) -> None:
        target = {"Car": ObjectSignals(object_type="Car", observed=1, misaligned=None)}
        merge_summaries(target, {"Car": ObjectSignals(object_type="Car", observed=1, misaligned=2)})
        self.assertEqual(target["Car"].misaligned, 2)


if __name__ == "__main__":
    unittest.main()
