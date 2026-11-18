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

"""Lane lines and road boundaries, from a static result file through to the ``object_metrics`` block."""

from typing import Any
import unittest

from kpi.automotive.augment import build_metrics
from kpi.automotive.existence import is_misaligned, is_missing, is_missing_over, is_unscored
from kpi.automotive.policy import default_policy
from kpi.automotive.result_view import ImportancePolicy, read_result
from kpi.automotive.summary import signal_summary

POLICY = default_policy(ImportancePolicy.ALL_SCORED)

DASHED = "LONG_DASHED_SINGLE"
SOLID = "SOLID_SINGLE"

# track id -> (object type, recorded style, per-frame scores)
_TRACKS: dict[int, tuple[str, str | None, list[float]]] = {
    1: ("LaneLine", SOLID, [0.05, 0.05]),  # below the solid cutoff: missing
    2: ("LaneLine", DASHED, [0.05, 0.05]),  # above the dashed cutoff: present
    3: ("LaneLine", DASHED, [0.001, 0.0]),  # below the dashed cutoff: missing
    4: ("LaneLine", None, [0.05, 0.05]),  # no recorded style takes the solid cutoff: missing
    5: ("RoadBoundary", None, [0.2, 0.3]),  # present
    6: ("RoadBoundary", None, [0.0999, 0.0999]),  # just below the cutoff: missing
    7: ("LaneLine", SOLID, []),  # never scored
}


def _result() -> dict[str, Any]:
    """A static result in the on-disk ``sparse_compact`` shape, with two lane-graph hallucination tracks."""
    track_ids = sorted(_TRACKS)
    rows = [
        [frame, column, score, 120.0]
        for column, track_id in enumerate(track_ids)
        for frame, score in enumerate(_TRACKS[track_id][2])
    ]
    tracks = []
    for track_id in track_ids:
        object_type, style, _scores = _TRACKS[track_id]
        entry: dict[str, Any] = {"track_id": track_id, "object_type": object_type}
        if style is not None:
            entry["lane_style"] = style
        tracks.append(entry)
    return {
        "tracks": tracks,
        "track_ids": track_ids,
        "processed_frame_ids": [0, 1],
        "score_matrix": {
            "format": "sparse_compact",
            "shape": [2, len(track_ids)],
            "columns": ["frame_idx", "track_idx", "score", "projection_area_px"],
            "data": rows,
        },
        "hallucination_detections": {
            "LaneLine": [{"frame_idx": 0, "bbox_xywh": [10, 20, 30, 4]}],
            "RoadBoundary": [{"frame_idx": 1, "bbox_xywh": [50, 60, 40, 6]}],
        },
        # The static tracker runs without a road mask, so relevancy is 0; both must still count.
        "hallucination_tracks": [
            {"id": 11, "class": "LaneLine", "detections": [0], "relevancy": 0},
            {"id": 12, "class": "RoadBoundary", "detections": [0], "relevancy": 0},
        ],
    }


class TestLaneGraphVerdicts(unittest.TestCase):
    def setUp(self) -> None:
        self.view = read_result(_result(), importance=ImportancePolicy.ALL_SCORED)

    def test_the_recorded_style_is_read_off_the_track(self) -> None:
        self.assertEqual(self.view.tracks[2].lane_style, DASHED)
        self.assertIsNone(self.view.tracks[4].lane_style)
        self.assertIsNone(self.view.tracks[5].lane_style)

    def test_each_object_is_judged_by_the_cutoff_for_its_type_and_style(self) -> None:
        verdicts = {track_id: is_missing(track, POLICY) for track_id, track in self.view.tracks.items()}
        self.assertEqual(verdicts, {1: True, 2: False, 3: True, 4: True, 5: False, 6: True, 7: None})

    def test_a_slice_takes_the_style_the_caller_supplies(self) -> None:
        frames = self.view.tracks[1].frame_scores
        self.assertIs(is_missing_over(frames, "LaneLine", POLICY), True)
        self.assertIs(is_missing_over(frames, "LaneLine", POLICY, lane_style=DASHED), False)

    def test_no_misalignment_verdict_is_defined(self) -> None:
        for track in self.view.tracks.values():
            if not is_unscored(track):
                self.assertIsNone(is_misaligned(track, POLICY), track.track_id)

    def test_summary_counts_lane_lines_and_road_boundaries(self) -> None:
        rows = signal_summary(self.view, POLICY)
        lane, boundary = rows["LaneLine"], rows["RoadBoundary"]
        self.assertEqual((lane.observed, lane.unscored, lane.present, lane.missing), (4, 1, 1, 3))
        self.assertEqual((boundary.observed, boundary.present, boundary.missing), (2, 1, 1))
        self.assertIsNone(lane.misaligned)
        self.assertEqual((lane.hallucinated, boundary.hallucinated), (1, 1))


class TestLaneGraphMetricsBlock(unittest.TestCase):
    def setUp(self) -> None:
        self.block = build_metrics(_result(), POLICY)

    def test_lane_graph_types_are_evaluated(self) -> None:
        self.assertIn("LaneLine", self.block["by_type"])
        self.assertIn("RoadBoundary", self.block["by_type"])
        self.assertEqual(self.block["unevaluated_object_types"], [])
        self.assertEqual(self.block["unevaluated_hallucination_classes"], [])

    def test_each_track_records_the_style_that_selected_its_cutoff(self) -> None:
        styles = {entry["track_id"]: entry["lane_style"] for entry in self.block["tracks"]}
        self.assertEqual(styles[2], DASHED)
        self.assertIsNone(styles[5])

    def test_lane_graph_hallucinations_are_extra_objects(self) -> None:
        extras = {extra["id"]: extra["group"] for extra in self.block["extra_objects"]}
        self.assertEqual(extras, {11: "lane_graph", 12: "lane_graph"})

    def test_the_policy_records_both_lane_line_cutoffs(self) -> None:
        existence = self.block["policy"]["existence"]
        self.assertEqual(existence["LaneLine"], {"rule": "mean_below", "threshold": 0.1})
        self.assertEqual(existence["LaneLine:dashed"], {"rule": "mean_below", "threshold": 0.002})


if __name__ == "__main__":
    unittest.main()
