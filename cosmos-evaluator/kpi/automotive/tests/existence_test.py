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

import math
import unittest

from kpi.automotive.existence import (
    governing_object_type,
    hallucination_matches,
    is_misaligned,
    is_missing,
    is_missing_over,
    is_present,
    is_unscored,
    relevant_hallucinations,
)
from kpi.automotive.policy import SignalPolicy, default_policy
from kpi.automotive.result_view import ImportancePolicy, ScoredFrame, TrackView, read_result
from kpi.automotive.rules import DrawnEmpty, PercentileBelow, RelevancyGate
from kpi.automotive.taxonomy import Group, types_in_group
from kpi.automotive.tests.fixtures import load_dynamic_result

EGO = default_policy(ImportancePolicy.EGO_RELEVANT)


def _frames(*scores: float) -> list[ScoredFrame]:
    """Scores as frames on consecutive ordinals, for the slice-taking predicates."""
    return [ScoredFrame(ordinal=i, frame_idx=i, score=score) for i, score in enumerate(scores)]


def _track(object_type: str, scores: tuple[float, ...]) -> TrackView:
    """A ``TrackView`` carrying exactly ``scores``, on consecutive frame ordinals."""
    frame_scores = tuple(ScoredFrame(ordinal=i, frame_idx=i, score=s) for i, s in enumerate(scores))
    mean_score = math.fsum(scores) / len(scores) if scores else math.nan
    return TrackView(
        track_id=1,
        object_type=object_type,
        frame_scores=frame_scores,
        mean_score=mean_score,
        synthetic_source_id=None,
    )


class TestIsUnscored(unittest.TestCase):
    def test_track_without_scores_is_unscored(self) -> None:
        self.assertIs(is_unscored(_track("Car", ())), True)

    def test_track_with_scores_is_scored_even_when_they_are_all_zero(self) -> None:
        self.assertIs(is_unscored(_track("Car", (0.0, 0.0))), False)


class TestExistence(unittest.TestCase):
    def test_all_zero_scores_mean_missing(self) -> None:
        track = _track("Car", (0.0, 0.0, 0.0))
        self.assertIs(is_missing(track, EGO), True)
        self.assertIs(is_present(track, EGO), False)

    def test_a_weakly_corresponding_object_is_present_not_missing(self) -> None:
        """The zero-score rule exists so a badly rendered car is not conflated with an absent one."""
        track = _track("Car", (0.08, 0.05, 0.11))
        self.assertIs(is_missing(track, EGO), False)
        self.assertIs(is_present(track, EGO), True)

    def test_unscored_track_yields_none_rather_than_missing(self) -> None:
        track = _track("Car", ())
        self.assertIsNone(is_missing(track, EGO))
        self.assertIsNone(is_present(track, EGO))

    def test_unregistered_type_has_no_rule_and_yields_none(self) -> None:
        track = _track("WaitLine", (0.0, 0.0))
        self.assertIsNone(is_missing(track, EGO), "a type with no governing question gets no substitute rule")

    def test_scores_within_tolerance_of_zero_still_count_as_missing(self) -> None:
        self.assertIs(is_missing(_track("Car", (1e-12, 0.0)), EGO), True)
        self.assertIs(is_missing(_track("Car", (1e-4, 0.0)), EGO), False)

    def test_is_missing_over_applies_the_rule_to_an_arbitrary_slice(self) -> None:
        self.assertIs(is_missing_over(_frames(0.0, 0.0), "Car", EGO), True)
        self.assertIs(is_missing_over(_frames(0.0, 0.4), "Car", EGO), False)
        self.assertIsNone(is_missing_over([], "Car", EGO))
        self.assertIsNone(is_missing_over(_frames(0.0), "WaitLine", EGO))

    def test_a_slice_keeps_the_ordinals_it_had_in_the_parent_track(self) -> None:
        """A band is a subset of frames, not a fresh sequence - a contiguity rule must see the real gaps."""
        track = _track("Car", (0.9, 0.0, 0.0, 0.9))
        band = [frame for frame in track.frame_scores if frame.score == 0.0]
        self.assertEqual([frame.ordinal for frame in band], [1, 2])
        self.assertIs(is_missing_over(band, "Car", EGO), True)

    def test_a_percentile_existence_rule_can_be_installed_for_one_type(self) -> None:
        policy = SignalPolicy(
            name="test",
            importance=ImportancePolicy.EGO_RELEVANT,
            existence={"Car": PercentileBelow(percentile=85.0, threshold=0.3)},
            misalignment={},
            hallucination={},
            wrong_state={},
            conflicting={},
        )
        self.assertIs(is_missing(_track("Car", (0.1, 0.2, 0.3)), policy), True)
        self.assertIs(is_missing(_track("Car", (0.8, 0.85, 0.9)), policy), False)


class TestMisalignment(unittest.TestCase):
    def test_low_but_non_zero_percentile_is_misaligned(self) -> None:
        self.assertIs(is_misaligned(_track("Truck", (0.1, 0.2, 0.3)), EGO), True)

    def test_well_corresponded_object_is_not_misaligned(self) -> None:
        self.assertIs(is_misaligned(_track("Car", (0.8, 0.85, 0.9)), EGO), False)

    def test_a_missing_object_is_not_also_misaligned(self) -> None:
        """Score 0 is the missing question's answer; misalignment is strictly the non-zero low band."""
        self.assertIs(is_misaligned(_track("Car", (0.0, 0.0, 0.0)), EGO), False)

    def test_a_missing_object_is_never_also_misaligned(self) -> None:
        """Scores within the zero tolerance but not exactly zero would satisfy both rules' bounds."""
        track = _track("Car", (0.0, 0.0, 1e-12))
        self.assertIs(is_missing(track, EGO), True)
        self.assertIs(is_misaligned(track, EGO), False)

    def test_vrus_have_no_misalignment_rule(self) -> None:
        self.assertIsNone(is_misaligned(_track("Pedestrian", (0.1, 0.2, 0.15)), EGO))

    def test_unscored_track_yields_none(self) -> None:
        self.assertIsNone(is_misaligned(_track("Car", ()), EGO))

    def test_absent_rule_yields_none(self) -> None:
        policy = SignalPolicy(
            name="test",
            importance=ImportancePolicy.EGO_RELEVANT,
            existence={"Car": DrawnEmpty()},
            misalignment={},
            hallucination={},
            wrong_state={},
            conflicting={},
        )
        self.assertIsNone(is_misaligned(_track("Car", (0.1, 0.2)), policy))


class TestHallucinations(unittest.TestCase):
    def setUp(self) -> None:
        self.view = read_result(load_dynamic_result(), importance=ImportancePolicy.EGO_RELEVANT)

    def test_four_wheeler_hallucinations_are_relevancy_gated(self) -> None:
        found = relevant_hallucinations(self.view, Group.FOUR_WHEELER, EGO)
        self.assertEqual([hallucination.id for hallucination in found], [1])

    def test_vru_hallucinations_are_selected_by_class(self) -> None:
        found = relevant_hallucinations(self.view, Group.VRU, EGO)
        self.assertEqual([hallucination.id for hallucination in found], [3])

    def test_min_relevancy_zero_disables_the_gate(self) -> None:
        policy = SignalPolicy(
            name="test",
            importance=ImportancePolicy.EGO_RELEVANT,
            existence={},
            misalignment={},
            hallucination=dict.fromkeys(types_in_group(Group.FOUR_WHEELER), RelevancyGate(min_relevancy=0)),
            wrong_state={},
            conflicting={},
        )
        found = relevant_hallucinations(self.view, Group.FOUR_WHEELER, policy)
        self.assertEqual([hallucination.id for hallucination in found], [1, 2])

    def test_class_matching_is_case_and_separator_insensitive(self) -> None:
        """Labels are matched through the shared alias table rather than by equality."""
        result = load_dynamic_result()
        result["hallucination_tracks"][0]["class"] = "car"
        view = read_result(result, importance=ImportancePolicy.EGO_RELEVANT)
        found = relevant_hallucinations(view, Group.FOUR_WHEELER, EGO)
        self.assertEqual([hallucination.id for hallucination in found], [1])

    def test_governing_object_type_separates_no_rule_from_rule_says_no(self) -> None:
        """A class no type claims can never be scored; a relevancy-0 one was scored and rejected."""
        self.assertEqual(governing_object_type("Vehicle"), "Automobile")
        self.assertIsNotNone(governing_object_type("Pedestrian"))
        self.assertIsNone(governing_object_type("WaitLine"), "no registered type claims this label")
        self.assertIsNone(governing_object_type(""), "a hallucination with no class is governed by nothing")

    def test_a_governed_class_stays_governed_even_when_its_rule_rejects_it(self) -> None:
        rejected = self.view.hallucinations[1]  # Vehicle, relevancy 0
        self.assertEqual(rejected.relevancy, 0)
        self.assertNotIn(rejected.id, [h.id for h in relevant_hallucinations(self.view, Group.FOUR_WHEELER, EGO)])
        self.assertIsNotNone(governing_object_type(rejected.class_label))

    def test_hallucination_matches_uses_the_detector_class_label(self) -> None:
        vehicle = self.view.hallucinations[0]
        self.assertIs(hallucination_matches(vehicle, "Car"), True)
        self.assertIs(hallucination_matches(vehicle, "Pedestrian"), False)


if __name__ == "__main__":
    unittest.main()
