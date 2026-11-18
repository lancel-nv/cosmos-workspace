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

"""Traffic-light scoring: existence, color state, conflicting signals, and the frame space they depend on."""

from typing import Any
import unittest

from checks.utils.tl_state import conflicting_run
from kpi.automotive.existence import (
    is_conflicting,
    is_conflicting_over,
    is_missing,
    is_unscored,
    is_wrong_state,
    is_wrong_state_over,
    relevant_hallucinations,
)
from kpi.automotive.policy import default_policy
from kpi.automotive.result_view import ImportancePolicy, ResultView, read_result, state_frames_by_track
from kpi.automotive.summary import signal_summary
from kpi.automotive.taxonomy import Group
from kpi.automotive.tests.fixtures import load_dynamic_result, load_traffic_light_result

POLICY = default_policy(ImportancePolicy.ALL_SCORED)


def _view(result: dict[str, Any] | None = None) -> ResultView:
    """The traffic-light fixture as a view, or ``result`` when a test has modified it."""
    return read_result(result or load_traffic_light_result(), importance=ImportancePolicy.ALL_SCORED)


class TestFrameSpace(unittest.TestCase):
    def test_state_frames_are_normalized_onto_processed_frame_ordinals(self) -> None:
        """The fixture's rows sit on video frames 0, 3, 6; the predicates need a dense axis to see adjacency."""
        track = _view().tracks[10]
        self.assertEqual([f.ordinal for f in track.state_frames], [0, 1, 2])

    def test_a_row_naming_an_unprocessed_frame_is_dropped(self) -> None:
        """Its position on the dense axis is undefined; inventing one would fabricate adjacency."""
        result = load_traffic_light_result()
        result["traffic_light_state"]["10"].append(
            {
                "frame_idx": 999,
                "expected_state": "green",
                "traffic_light_state_score": 0.1,
                "traffic_light_multi_color_score": 0.0,
                "traffic_light_state_confidence": 0.5,
            }
        )
        self.assertEqual([f.ordinal for f in _view(result).tracks[10].state_frames], [0, 1, 2])

    def test_without_normalization_the_conflict_rule_would_not_fire(self) -> None:
        """Guards the guard: proves the fixture actually depends on the normalization, not just tolerates it."""
        raw = load_traffic_light_result()["traffic_light_state"]["13"]
        self.assertEqual(conflicting_run(raw, 0.5)[0], 1, "raw ids spaced 3 apart give a max run of 1")
        normalized = [f.as_row() for f in _view().tracks[13].state_frames]
        self.assertEqual(conflicting_run(normalized, 0.5)[0], 2, "on ordinals the two frames are adjacent")


class TestTrafficLightSignals(unittest.TestCase):
    def setUp(self) -> None:
        self.view = _view()

    def test_a_clean_light_trips_nothing(self) -> None:
        track = self.view.tracks[10]
        self.assertIs(is_missing(track, POLICY), False)
        self.assertIs(is_wrong_state(track, POLICY), False)
        self.assertIs(is_conflicting(track, POLICY), False)

    def test_an_empty_box_is_missing(self) -> None:
        self.assertIs(is_missing(self.view.tracks[11], POLICY), True)

    def test_a_light_with_no_state_frames_reports_state_as_unmeasured(self) -> None:
        """Track 11 was scored for correspondence but never sampled for color: null, not 'correct'."""
        track = self.view.tracks[11]
        self.assertEqual(track.state_frames, ())
        self.assertIsNone(is_wrong_state(track, POLICY))
        self.assertIsNone(is_conflicting(track, POLICY))

    def test_a_light_outside_the_importance_scope_reports_no_state_verdict(self) -> None:
        """State rows bypass the importance filter, so the predicates gate on the track being scored at all.

        Track 10 keeps its three state frames either way. Under ``EGO_RELEVANT`` none of its score cells count,
        which puts it outside the selected population - answering the color question for it anyway would draw
        a rate's numerator from a wider population than its denominator.
        """
        result = load_traffic_light_result()
        for row in result["score_matrix"]["data"]:
            if row[1] == 0:  # track_idx of track 10
                row[2] = 0  # importance
        unscoped = read_result(result, importance=ImportancePolicy.EGO_RELEVANT).tracks[10]
        self.assertIs(is_unscored(unscoped), True)
        self.assertEqual(len(unscoped.state_frames), 3)
        self.assertIsNone(is_wrong_state(unscoped, default_policy(ImportancePolicy.EGO_RELEVANT)))
        self.assertIsNone(is_conflicting(unscoped, default_policy(ImportancePolicy.EGO_RELEVANT)))

        scoped = read_result(result, importance=ImportancePolicy.ALL_SCORED).tracks[10]
        self.assertIs(is_wrong_state(scoped, POLICY), False)
        self.assertIs(is_conflicting(scoped, POLICY), False)

    def test_wrong_color_is_attributed_per_episode(self) -> None:
        """Track 12 is right in its green phase and wrong in its red one; a per-track mean would hide it."""
        track = self.view.tracks[12]
        self.assertIs(is_wrong_state(track, POLICY), True)
        self.assertIs(is_conflicting(track, POLICY), False)

    def test_a_light_naming_no_ground_truth_color_declines_the_wrong_color_question(self) -> None:
        """Files written before the checker emitted ``expected_state`` carry scores and no colors.

        Without the guard the frames form one episode of color ``""`` split only on contiguity, and track 12
        - whose scores really do dip - comes back a confident ``True`` with nothing to have been wrong about.
        """
        result = load_traffic_light_result()
        for row in result["traffic_light_state"]["12"]:
            del row["expected_state"]
        track = _view(result).tracks[12]
        self.assertEqual(len(track.state_frames), 3, "the state rows are still read; only the color is gone")
        self.assertIsNone(is_wrong_state(track, POLICY))
        self.assertIs(is_conflicting(track, POLICY), False, "conflict needs no ground-truth color")

    def test_a_sustained_two_color_run_is_conflicting_but_not_wrong_state(self) -> None:
        track = self.view.tracks[13]
        self.assertIs(is_conflicting(track, POLICY), True)
        self.assertIs(is_wrong_state(track, POLICY), False)

    def test_static_hallucinations_are_not_dropped_by_the_relevancy_gate(self) -> None:
        """The static tracker runs without a road mask, so every static hallucination carries relevancy 0."""
        found = relevant_hallucinations(self.view, Group.TRAFFIC_LIGHT, POLICY)
        self.assertEqual([h.id for h in found], [1])
        self.assertEqual(found[0].relevancy, 0)


class TestTrafficLightSummary(unittest.TestCase):
    def test_the_row_reports_state_counts_and_no_misalignment(self) -> None:
        row = signal_summary(_view(), POLICY)["TrafficLight"]
        self.assertEqual((row.observed, row.present, row.missing), (4, 3, 1))
        self.assertEqual((row.wrong_state, row.conflicting), (1, 1))
        self.assertIsNone(row.misaligned, "a light has no box-containment signal")
        self.assertEqual(row.hallucinated, 1)

    def test_color_rates_use_the_color_denominator_not_the_geometry_one(self) -> None:
        """The two signal families come from different detectors, so they do not share a denominator.

        The fixture has four lights the overlap check scored and three the state check measured, so a color
        rate over ``observed`` would divide by the wrong population - which on real files roughly doubles it.
        """
        row = signal_summary(_view(), POLICY)["TrafficLight"]
        self.assertEqual((row.observed, row.color_observed), (4, 3))
        rates = row.rates()
        self.assertEqual(rates["wrong_state_rate"], 1 / 3)
        self.assertEqual(rates["conflicting_rate"], 1 / 3)
        self.assertEqual(rates["missing_rate"], 0.25, "existence still divides by the geometry population")
        self.assertIsNone(rates["misaligned_rate"], "unmeasured stays null rather than becoming 0.0")

    def test_a_type_with_no_color_rule_gets_no_row_from_the_color_pass(self) -> None:
        """An all-zero row reads as "measured and clean", which is the substitute verdict this library refuses.

        The existence pass skips unregistered types outright; the color pass must not reintroduce them by
        creating a counter before discovering that no rule governs the type.
        """
        result = load_traffic_light_result()
        for track in result["tracks"]:
            if track["track_id"] == 12:
                track["object_type"] = "Crosswalk"
        rows = signal_summary(_view(result), POLICY)
        self.assertNotIn("Crosswalk", rows)

    def test_a_light_absent_from_tracks_still_counts_toward_color(self) -> None:
        """Color coverage must not inherit the overlap check's roster - see ``state_frames_by_track``."""
        result = load_traffic_light_result()
        result["tracks"] = [track for track in result["tracks"] if track["track_id"] != 13]
        row = signal_summary(_view(result), POLICY)["TrafficLight"]
        self.assertEqual(row.observed, 3, "the light is gone from the geometry population")
        self.assertEqual(row.color_observed, 3, "but it is still measured for color")
        self.assertEqual(row.conflicting, 1, "and its conflicting verdict still counts")

    def test_a_file_naming_no_colors_reports_wrong_state_as_unmeasured_not_zero(self) -> None:
        """The count must not collapse to 0, which would read as 'no light was the wrong color'."""
        result = load_traffic_light_result()
        for rows in result["traffic_light_state"].values():
            for row in rows:
                del row["expected_state"]
        row = signal_summary(_view(result), POLICY)["TrafficLight"]
        self.assertIsNone(row.wrong_state)
        self.assertIsNone(row.rates()["wrong_state_rate"])
        self.assertEqual(row.conflicting, 1, "the conflict signal is unaffected")

    def test_dynamic_types_report_state_as_not_applicable(self) -> None:
        rows = signal_summary(read_result(load_dynamic_result(), importance=ImportancePolicy.ALL_SCORED), POLICY)
        self.assertIsNone(rows["Car"].wrong_state)
        self.assertIsNone(rows["Car"].conflicting)
        self.assertIsNone(rows["Car"].rates()["wrong_state_rate"])


class TestColorPredicatesOverFrames(unittest.TestCase):
    """The frame-taking forms: color graded without going through a correspondence-scored track."""

    def test_they_grade_a_light_that_has_no_tracks_entry(self) -> None:
        """The reason these exist - most real lights are measured for color but absent from ``tracks``."""
        result = load_traffic_light_result()
        result["tracks"] = [track for track in result["tracks"] if track["track_id"] != 13]
        view = read_result(result, importance=ImportancePolicy.ALL_SCORED)
        self.assertNotIn(13, view.tracks, "precondition: no track view for this light")

        frames = state_frames_by_track(result)[13]
        self.assertIs(is_conflicting_over(frames, "TrafficLight", POLICY), True)

    def test_they_agree_with_the_track_form_where_both_apply(self) -> None:
        view = _view()
        frames_by_track = state_frames_by_track(load_traffic_light_result())
        for track_id, track in view.tracks.items():
            if not track.state_frames:
                continue
            frames = frames_by_track[track_id]
            self.assertEqual(is_wrong_state_over(frames, "TrafficLight", POLICY), is_wrong_state(track, POLICY))
            self.assertEqual(is_conflicting_over(frames, "TrafficLight", POLICY), is_conflicting(track, POLICY))

    def test_a_slice_is_judged_on_its_own_frames(self) -> None:
        """Track 12 is right in its green phase and wrong in its red one; the slice names which."""
        frames = state_frames_by_track(load_traffic_light_result())[12]
        green = tuple(frame for frame in frames if frame.expected_state == "green")
        red = tuple(frame for frame in frames if frame.expected_state == "red")
        self.assertIs(is_wrong_state_over(green, "TrafficLight", POLICY), False)
        self.assertIs(is_wrong_state_over(red, "TrafficLight", POLICY), True)

    def test_no_rule_and_no_frames_both_report_unmeasured(self) -> None:
        frames = state_frames_by_track(load_traffic_light_result())[10]
        self.assertIsNone(is_conflicting_over(frames, "Car", POLICY), "no color rule governs a car")
        self.assertIsNone(is_conflicting_over((), "TrafficLight", POLICY))


if __name__ == "__main__":
    unittest.main()
