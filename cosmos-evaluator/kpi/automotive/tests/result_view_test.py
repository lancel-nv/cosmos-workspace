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
from typing import Any
import unittest

from kpi.automotive.result_view import (
    LEGACY_COMPACT_COLUMNS,
    UNKNOWN_FRAME_ORDINAL,
    ImportancePolicy,
    ScoreCell,
    ScoredFrame,
    cell_is_counted,
    hallucination_views,
    iter_score_cells,
    read_result,
    state_frames_by_track,
)
from kpi.automotive.tests.fixtures import load_dynamic_result, load_traffic_light_result


def _compact(data: list[Any], columns: list[str] | None = None, shape: list[int] | None = None) -> dict[str, Any]:
    """A minimal ``sparse_compact`` result carrying two track ids and the given rows."""
    matrix: dict[str, Any] = {"format": "sparse_compact", "shape": shape or [4, 2], "data": data}
    if columns is not None:
        matrix["columns"] = columns
    return {
        "track_ids": [11, 22],
        "tracks": [{"track_id": 11, "object_type": "Car"}, {"track_id": 22, "object_type": "Car"}],
        "score_matrix": matrix,
    }


class TestIterScoreCells(unittest.TestCase):
    def test_resolves_positional_track_idx_to_track_id(self) -> None:
        cells = list(iter_score_cells(_compact([[0, 0, 1, 0.5], [1, 1, 1, 0.25]])))
        self.assertEqual([cell.track_id for cell in cells], [11, 22])
        self.assertEqual([cell.score for cell in cells], [0.5, 0.25])
        self.assertEqual([cell.frame_ordinal for cell in cells], [0, 1])
        self.assertEqual([cell.important for cell in cells], [True, True])

    def test_falls_back_to_legacy_columns_when_undeclared(self) -> None:
        """Older files declare no ``columns``; reading them positionally is the only way to recover them."""
        self.assertEqual(LEGACY_COMPACT_COLUMNS, ("frame_idx", "track_idx", "importance", "score"))
        cells = list(iter_score_cells(_compact([[2, 1, 0, 0.75]], columns=None)))
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0], ScoreCell(frame_ordinal=2, track_id=22, score=0.75, important=False))

    def test_honors_declared_column_order(self) -> None:
        result = _compact([[0.75, 1, 0], ["x", 0, 1]], columns=["score", "track_idx", "frame_idx"])
        cells = list(iter_score_cells(result))
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0].score, 0.75)
        self.assertEqual(cells[0].track_id, 22)
        self.assertIsNone(cells[0].important, "a file with no importance column reports no flag")

    def test_yields_nothing_without_a_score_column(self) -> None:
        self.assertEqual(list(iter_score_cells(_compact([[0, 1]], columns=["frame_idx", "track_idx"]))), [])

    def test_skips_rows_that_are_short_or_unparseable(self) -> None:
        rows = [[0, 0, 1, 0.5], [0, 1], [0, 1, 1, "not-a-score"], "not-a-row", [0, "nan-track", 1, 0.3]]
        cells = list(iter_score_cells(_compact(rows)))
        self.assertEqual([cell.track_id for cell in cells], [11])

    def test_drops_out_of_range_and_negative_track_indexes(self) -> None:
        """A negative index would wrap to the far end of the axis in a dense reader and corrupt another track."""
        cells = list(iter_score_cells(_compact([[0, -1, 1, 0.5], [0, 7, 1, 0.5], [0, 1, 1, 0.5]])))
        self.assertEqual([cell.track_id for cell in cells], [22])

    def test_keeps_the_score_when_the_frame_index_is_unusable(self) -> None:
        rows = [[99, 0, 1, 0.5], [-3, 1, 1, 0.25], ["x", 1, 1, 0.125]]
        cells = list(iter_score_cells(_compact(rows, shape=[4, 2])))
        self.assertEqual([cell.score for cell in cells], [0.5, 0.25, 0.125])
        for cell in cells:
            self.assertEqual(cell.frame_ordinal, UNKNOWN_FRAME_ORDINAL)

    def test_an_unusable_track_id_does_not_shift_later_positions(self) -> None:
        """``track_idx`` addresses ``track_ids`` positionally, so a bad entry must not renumber the rest."""
        result = {
            "track_ids": [10, "not-an-id", 20],
            "tracks": [{"track_id": 10, "object_type": "Car"}, {"track_id": 20, "object_type": "Car"}],
            "score_matrix": {
                "format": "sparse",
                "shape": [1, 3],
                "data": [
                    {"frame_idx": 0, "track_idx": 0, "score": 0.5},
                    {"frame_idx": 0, "track_idx": 1, "score": 0.9},
                    {"frame_idx": 0, "track_idx": 2, "score": 0.1},
                ],
            },
        }
        cells = list(iter_score_cells(result))
        self.assertEqual(
            [(cell.track_id, cell.score) for cell in cells],
            [(10, 0.5), (20, 0.1)],
            "the unusable slot is skipped; index 2 still resolves to track 20, not to a shifted neighbor",
        )

    def test_an_unusable_track_id_does_not_shift_positions_in_the_compact_format(self) -> None:
        result = _compact([[0, 0, 1, 0.5], [0, 1, 1, 0.9], [0, 2, 1, 0.1]], shape=[1, 3])
        result["track_ids"] = [10, None, 20]
        cells = list(iter_score_cells(result))
        self.assertEqual([(cell.track_id, cell.score) for cell in cells], [(10, 0.5), (20, 0.1)])

    def test_processed_frame_ids_keep_positions_for_unusable_entries(self) -> None:
        """A cell's frame ordinal indexes into this array, so a bad entry stays in place as ``None``."""
        result = load_dynamic_result()
        result["processed_frame_ids"] = [0, "x", 6, 9]
        view = read_result(result, importance=ImportancePolicy.EGO_RELEVANT)
        self.assertEqual(view.processed_frame_ids, (0, None, 6, 9))

    def test_reads_the_sparse_dict_format(self) -> None:
        result = {
            "track_ids": [11, 22],
            "score_matrix": {
                "format": "sparse",
                "shape": [4, 2],
                "data": [
                    {"frame_idx": 1, "track_idx": 1, "score": 0.5},
                    {"track_idx": 0, "score": 0.25},
                    {"track_idx": 0},
                    "not-an-entry",
                ],
            },
        }
        cells = list(iter_score_cells(result))
        self.assertEqual([(cell.track_id, cell.score) for cell in cells], [(22, 0.5), (11, 0.25)])
        self.assertEqual(cells[1].frame_ordinal, UNKNOWN_FRAME_ORDINAL)
        self.assertIsNone(cells[0].important, "the sparse format carries no importance flag")

    def test_returns_nothing_for_an_absent_or_unknown_matrix(self) -> None:
        self.assertEqual(list(iter_score_cells({})), [])
        self.assertEqual(list(iter_score_cells({"score_matrix": {"format": "summary"}})), [])
        self.assertEqual(list(iter_score_cells({"score_matrix": {"format": "sparse", "data": "x"}})), [])


class TestCellIsCounted(unittest.TestCase):
    def test_ego_relevant_drops_only_explicitly_unimportant_cells(self) -> None:
        policy = ImportancePolicy.EGO_RELEVANT
        self.assertIs(cell_is_counted(ScoreCell(0, 1, 0.5, important=True), policy), True)
        self.assertIs(cell_is_counted(ScoreCell(0, 1, 0.5, important=False), policy), False)
        self.assertIs(
            cell_is_counted(ScoreCell(0, 1, 0.5, important=None), policy),
            True,
            "a file with no importance column has nothing to filter on, so nothing may be dropped",
        )

    def test_all_scored_counts_every_cell(self) -> None:
        policy = ImportancePolicy.ALL_SCORED
        for flag in (True, False, None):
            self.assertIs(cell_is_counted(ScoreCell(0, 1, 0.5, important=flag), policy), True)


class TestReadResult(unittest.TestCase):
    def test_importance_policy_changes_which_tracks_are_scored(self) -> None:
        """Track 5 is scored only in non-important frames, which is the whole reason the policy is required."""
        result = load_dynamic_result()
        ego = read_result(result, importance=ImportancePolicy.EGO_RELEVANT)
        every = read_result(result, importance=ImportancePolicy.ALL_SCORED)

        self.assertEqual(ego.tracks[5].frame_scores, ())
        self.assertTrue(math.isnan(ego.tracks[5].mean_score))
        self.assertEqual(every.tracks[5].scores(), (0.7, 0.6))
        self.assertAlmostEqual(every.tracks[5].mean_score, 0.65)
        self.assertIs(ego.importance_policy, ImportancePolicy.EGO_RELEVANT)
        self.assertIs(every.importance_policy, ImportancePolicy.ALL_SCORED)

    def test_track_views_carry_type_scores_and_importance(self) -> None:
        view = read_result(load_dynamic_result(), importance=ImportancePolicy.EGO_RELEVANT)
        car = view.tracks[2]
        self.assertEqual(car.object_type, "Car")
        self.assertEqual(car.scores(), (0.9, 0.8, 0.85))
        self.assertEqual(
            car.frame_scores,
            (
                ScoredFrame(ordinal=0, frame_idx=0, score=0.9),
                ScoredFrame(ordinal=1, frame_idx=3, score=0.8),
                ScoredFrame(ordinal=2, frame_idx=6, score=0.85),
            ),
        )
        self.assertAlmostEqual(car.mean_score, 0.85)
        self.assertEqual(car.importance["min_distance_m"], 14.0)
        self.assertIsNone(car.synthetic_source_id)

    def test_track_with_no_cells_at_all_is_unscored_under_both_policies(self) -> None:
        result = load_dynamic_result()
        for policy in ImportancePolicy:
            view = read_result(result, importance=policy)
            self.assertEqual(view.tracks[6].frame_scores, (), f"track 6 under {policy}")
            self.assertTrue(math.isnan(view.tracks[6].mean_score), f"track 6 under {policy}")

    def test_ids_without_a_tracks_entry_are_reported_separately(self) -> None:
        """Track 7 has scores but no type, so it cannot be classified and must not become an 'unknown' object."""
        view = read_result(load_dynamic_result(), importance=ImportancePolicy.EGO_RELEVANT)
        self.assertNotIn(7, view.tracks)
        self.assertEqual(view.untyped_track_ids, frozenset({7}))

    def test_processed_frame_ids_are_preserved(self) -> None:
        view = read_result(load_dynamic_result(), importance=ImportancePolicy.EGO_RELEVANT)
        self.assertEqual(view.processed_frame_ids, (0, 3, 6, 9))

    def test_synthetic_source_id_is_surfaced_for_injected_tracks(self) -> None:
        result = load_dynamic_result()
        result["tracks"][0]["synthetic_source_id"] = "kpi_add_0"
        result["tracks"][1]["synthetic_source_id"] = ""
        view = read_result(result, importance=ImportancePolicy.EGO_RELEVANT)
        self.assertEqual(view.tracks[1].synthetic_source_id, "kpi_add_0")
        self.assertIsNone(view.tracks[2].synthetic_source_id, "an empty marker is not an identity")


class TestHallucinationViews(unittest.TestCase):
    def test_resolves_detection_indices_into_boxes(self) -> None:
        views = hallucination_views(load_dynamic_result())
        by_id = {view.id: view for view in views}
        self.assertEqual(sorted(by_id), [1, 2, 3])
        self.assertEqual(by_id[1].class_label, "Vehicle")
        self.assertEqual(by_id[1].relevancy, 2)
        self.assertEqual([detection.frame_idx for detection in by_id[1].detections], [0, 3])
        self.assertEqual(by_id[1].detections[0].bbox_xywh, (10, 12, 40, 30))
        self.assertEqual(by_id[1].detections[0].class_label, "Vehicle")

    def test_detection_frame_idx_is_a_video_frame_not_a_processed_ordinal(self) -> None:
        """The pedestrian hallucination sits on video frame 6, i.e. a *value* in processed_frame_ids."""
        by_id = {view.id: view for view in hallucination_views(load_dynamic_result())}
        self.assertEqual([detection.frame_idx for detection in by_id[3].detections], [6])

    def test_drops_unusable_detection_entries_but_keeps_the_track(self) -> None:
        result = load_dynamic_result()
        result["hallucination_tracks"][0]["detections"] = [0, 99, -1, "1", True, None]
        result["hallucination_detections"]["Vehicle"][1] = {"frame_idx": 3}
        by_id = {view.id: view for view in hallucination_views(result)}
        self.assertIn(1, by_id, "a track that loses some detections is still a real hallucination")
        self.assertEqual([detection.frame_idx for detection in by_id[1].detections], [0])

    def test_accepts_the_bbox_alias_and_rejects_malformed_boxes(self) -> None:
        result = load_dynamic_result()
        result["hallucination_detections"]["Vehicle"] = [
            {"frame_idx": 0, "bbox": [1, 2, 3, 4]},
            {"frame_idx": 1, "bbox_xywh": [1, 2, 3]},
        ]
        result["hallucination_tracks"][0]["detections"] = [0, 1]
        by_id = {view.id: view for view in hallucination_views(result)}
        self.assertEqual([detection.bbox_xywh for detection in by_id[1].detections], [(1, 2, 3, 4)])

    def test_track_whose_class_has_no_detection_list_survives_with_no_detections(self) -> None:
        result = load_dynamic_result()
        result["hallucination_detections"] = {}
        views = hallucination_views(result)
        self.assertEqual(len(views), 3)
        self.assertTrue(all(view.detections == () for view in views))


class TestStateFramesByTrack(unittest.TestCase):
    """The seam for color state, keyed off the ``traffic_light_state`` block rather than ``tracks``."""

    def test_reads_a_light_that_has_no_tracks_entry(self) -> None:
        """The reason this exists: most real files measure lights that never appear in ``tracks``.

        ``TrackView.state_frames`` can only reach a light with a ``tracks`` entry, so a consumer reading state
        that way silently drops the rest of the population.
        """
        result = load_traffic_light_result()
        result["tracks"] = [track for track in result["tracks"] if track["track_id"] != 10]
        view = read_result(result, importance=ImportancePolicy.ALL_SCORED)
        self.assertNotIn(10, view.tracks, "precondition: the light is absent from the track views")
        self.assertIn(10, state_frames_by_track(result), "but its color state is still measured")

    def test_frames_arrive_on_the_scored_frame_axis(self) -> None:
        """Same normalization ``read_result`` applies - the fixture's rows sit on video frames 0, 3, 6."""
        frames = state_frames_by_track(load_traffic_light_result())[10]
        self.assertEqual([frame.ordinal for frame in frames], [0, 1, 2])
        self.assertEqual([frame.frame_idx for frame in frames], [0, 3, 6])

    def test_agrees_with_the_track_view_where_both_can_see_the_light(self) -> None:
        result = load_traffic_light_result()
        view = read_result(result, importance=ImportancePolicy.ALL_SCORED)
        by_track = state_frames_by_track(result)
        for track_id, track in view.tracks.items():
            if track.state_frames:
                self.assertEqual(by_track[track_id], track.state_frames, track_id)

    def test_a_file_that_cannot_place_its_rows_yields_nothing(self) -> None:
        """Without ``processed_frame_ids`` there is no way to tell a skipped frame from the normal stride."""
        result = load_traffic_light_result()
        del result["processed_frame_ids"]
        self.assertEqual(state_frames_by_track(result), {})


if __name__ == "__main__":
    unittest.main()
