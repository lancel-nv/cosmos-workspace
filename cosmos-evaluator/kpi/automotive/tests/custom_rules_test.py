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

"""The extension path: a rule this library did not define, evaluated and stamped like a built-in.

If these fail, a downstream caller cannot supply their own rule without forking the package - which is the
property the protocol-based design exists to provide.
"""

from collections.abc import Sequence
import dataclasses
import json
import math
import statistics
from typing import Any
import unittest

from kpi.automotive.existence import is_misaligned, is_missing, relevant_hallucinations
from kpi.automotive.policy import default_policy
from kpi.automotive.result_view import HallucinationView, ImportancePolicy, ScoredFrame, TrackView, read_result
from kpi.automotive.summary import signal_summary
from kpi.automotive.taxonomy import Group, types_in_group
from kpi.automotive.tests.fixtures import load_dynamic_result


@dataclasses.dataclass(frozen=True)
class MedianBelow:
    """A caller-defined existence rule: missing when the median score is below a threshold."""

    threshold: float

    def holds(self, frames: Sequence[ScoredFrame]) -> bool:
        return statistics.median(frame.score for frame in frames) < self.threshold

    def as_dict(self) -> dict[str, Any]:
        return {"rule": "median_below", "threshold": self.threshold}


@dataclasses.dataclass(frozen=True)
class MinimumDetections:
    """A caller-defined hallucination rule: counts only tracks seen in enough frames."""

    min_detections: int

    def holds(self, hallucination: HallucinationView, *, relevancy_meaningful: bool) -> bool:
        return len(hallucination.detections) >= self.min_detections

    def as_dict(self) -> dict[str, Any]:
        return {"rule": "minimum_detections", "min_detections": self.min_detections}


def _track_with(scores: Sequence[float]) -> TrackView:
    """A Car track carrying ``scores`` on consecutive ordinals."""
    return _track_from(tuple(ScoredFrame(ordinal=i, frame_idx=i * 3, score=v) for i, v in enumerate(scores)))


def _track_from(frames: tuple[ScoredFrame, ...]) -> TrackView:
    """A Car track carrying exactly ``frames``, ordinals and all."""
    return TrackView(
        track_id=1,
        object_type="Car",
        frame_scores=frames,
        mean_score=math.fsum(f.score for f in frames) / len(frames),
        synthetic_source_id=None,
    )


@dataclasses.dataclass(frozen=True)
class SustainedDropout:
    """A frame-aware existence rule: missing when the render drew nothing for a contiguous stretch.

    The reason :meth:`TrackRule.holds` takes frames rather than bare scores. An object drawn faithfully for
    most of a clip and absent for a run of frames averages well above zero, so the default ``DrawnEmpty`` rule
    calls it present; only the frames can tell that stretch from the same zeros scattered at random.
    """

    min_missing_frames: int
    atol: float = 1e-8

    def holds(self, frames: Sequence[ScoredFrame]) -> bool:
        longest = run = 0
        previous: int | None = None
        for frame in frames:
            if abs(frame.score) > self.atol:
                run = 0
            elif previous is not None and frame.ordinal == previous + 1:
                run += 1
            else:
                run = 1
            previous, longest = frame.ordinal, max(longest, run)
        return longest >= self.min_missing_frames

    def as_dict(self) -> dict[str, Any]:
        return {"rule": "sustained_dropout", "min_missing_frames": self.min_missing_frames, "atol": self.atol}


def _view() -> Any:
    return read_result(load_dynamic_result(), importance=ImportancePolicy.EGO_RELEVANT)


class TestCustomExistenceRule(unittest.TestCase):
    def setUp(self) -> None:
        base = default_policy(ImportancePolicy.EGO_RELEVANT)
        self.base = base
        self.policy = dataclasses.replace(
            base, name="median-experiment", existence={**base.existence, "Car": MedianBelow(0.9)}
        )

    def test_the_custom_rule_decides_for_the_type_it_was_installed_for(self) -> None:
        """Track 2 scores 0.8/0.85/0.9 - present by default, missing under a median-below-0.9 rule."""
        car = _view().tracks[2]
        self.assertIs(is_missing(car, self.base), False)
        self.assertIs(is_missing(car, self.policy), True)

    def test_untouched_types_keep_the_default_rule(self) -> None:
        pedestrian = _view().tracks[3]
        self.assertIs(is_missing(pedestrian, self.policy), is_missing(pedestrian, self.base))

    def test_the_custom_rule_flows_through_the_summary(self) -> None:
        rows = signal_summary(_view(), self.policy)
        self.assertEqual((rows["Car"].observed, rows["Car"].missing), (2, 2))
        self.assertEqual(rows["Truck"].missing, 0, "a type the variant did not touch is unaffected")

    def test_the_custom_rule_stamps_its_own_parameters(self) -> None:
        payload = self.policy.as_dict()
        json.dumps(payload)
        self.assertEqual(payload["existence"]["Car"], {"rule": "median_below", "threshold": 0.9})
        self.assertEqual(payload["existence"]["Truck"], {"rule": "drawn_empty", "atol": 1e-8})


class TestCustomHallucinationRule(unittest.TestCase):
    def test_a_caller_rule_replaces_the_relevancy_gate(self) -> None:
        """Hallucination 2 has one detection and relevancy 0: kept by detection count, dropped by relevancy."""
        base = default_policy(ImportancePolicy.EGO_RELEVANT)
        policy = dataclasses.replace(
            base,
            name="detection-count",
            hallucination=dict.fromkeys(types_in_group(Group.FOUR_WHEELER), MinimumDetections(1)),
        )
        view = _view()
        self.assertEqual([h.id for h in relevant_hallucinations(view, Group.FOUR_WHEELER, base)], [1])
        self.assertEqual([h.id for h in relevant_hallucinations(view, Group.FOUR_WHEELER, policy)], [1, 2])


class TestCustomMisalignmentRule(unittest.TestCase):
    def test_a_type_with_no_misalignment_rule_gains_one(self) -> None:
        """VRUs have no misalignment rule by default; installing one makes the signal reportable."""
        base = default_policy(ImportancePolicy.EGO_RELEVANT)
        pedestrian = _view().tracks[3]
        self.assertIsNone(is_misaligned(pedestrian, base))
        policy = dataclasses.replace(
            base, name="vru-misalignment", misalignment={**base.misalignment, "Pedestrian": MedianBelow(0.9)}
        )
        self.assertIs(is_misaligned(pedestrian, policy), True)
        self.assertEqual(signal_summary(_view(), policy)["Pedestrian"].misaligned, 1)


class TestFrameAwareExistenceRule(unittest.TestCase):
    """A rule that could not be written at all against a bare sequence of scores."""

    def setUp(self) -> None:
        self.base = default_policy(ImportancePolicy.ALL_SCORED)
        self.policy = dataclasses.replace(
            self.base, name="dropout", existence={**self.base.existence, "Car": SustainedDropout(2)}
        )

    def _car(self, scores: Sequence[float]) -> Any:
        return _track_with(scores)

    def test_a_contiguous_dropout_is_missing_where_the_mean_rule_says_present(self) -> None:
        track = _track_with((0.9, 0.0, 0.0, 0.9))
        self.assertIs(is_missing(track, self.base), False, "the mean is well above zero")
        self.assertIs(is_missing(track, self.policy), True)

    def test_the_same_zeros_scattered_do_not_trip_it(self) -> None:
        """Identical score multiset, different arrangement - the verdict must differ."""
        track = _track_with((0.0, 0.9, 0.0, 0.9))
        self.assertIs(is_missing(track, self.policy), False)

    def test_a_gap_in_the_ordinals_breaks_the_run(self) -> None:
        """Frames 0 and 2 are both empty but were not consecutive measurements."""
        frames = (
            ScoredFrame(ordinal=0, frame_idx=0, score=0.0),
            ScoredFrame(ordinal=2, frame_idx=6, score=0.0),
            ScoredFrame(ordinal=3, frame_idx=9, score=0.9),
        )
        self.assertIs(is_missing(_track_from(frames), self.policy), False)

    def test_it_stamps_its_own_parameters(self) -> None:
        payload = self.policy.as_dict()
        json.dumps(payload)
        self.assertEqual(payload["existence"]["Car"]["rule"], "sustained_dropout")
        self.assertEqual(payload["existence"]["Car"]["min_missing_frames"], 2)


if __name__ == "__main__":
    unittest.main()
