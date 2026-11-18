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

"""Unit tests for hallucination_tracking module."""

from typing import Any
import unittest

import numpy as np

import checks.objects.single_view.hallucination_tracking
from checks.objects.single_view.hallucination_tracking import (
    _max_consecutive_frames,
    confirm_hallucination_tracks,
    regroup_ground_tracks,
    track_hallucinations,
)


class TestHallucinationTracking(unittest.TestCase):
    """Test cases for hallucination tracking functionality."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Reset global counter for consistent testing
        checks.objects.single_view.hallucination_tracking._TRACK_ID_COUNTER = 0

        # Sample results structure
        self.base_results: dict[str, Any] = {
            "hallucination_detections": {
                "vehicle": [
                    {"frame_idx": 0, "bbox_xywh": [100, 100, 50, 50], "mask_ratio": 0.8},
                    {"frame_idx": 0, "bbox_xywh": [200, 200, 60, 40], "mask_ratio": 0.9},
                ],
                "pedestrian": [{"frame_idx": 0, "bbox_xywh": [300, 300, 30, 60], "mask_ratio": 0.7}],
            },
            "hallucination_tracks": [],
        }

    def test_track_hallucinations_empty_start_indices(self) -> None:
        """Test track_hallucinations with empty start_indices."""
        results = self.base_results.copy()
        track_hallucinations(results, {})

        # Should not modify results when start_indices is empty
        self.assertEqual(len(results["hallucination_tracks"]), 0)

    def test_track_hallucinations_no_existing_tracks(self) -> None:
        """Test track_hallucinations with no existing tracks."""
        results = self.base_results.copy()
        start_indices = {"vehicle": 0, "pedestrian": 0}

        track_hallucinations(results, start_indices)

        # Should create new tracks for all detections
        tracks = results["hallucination_tracks"]
        self.assertEqual(len(tracks), 3)  # 2 vehicles + 1 pedestrian

        # Check first vehicle track
        vehicle_tracks = [t for t in tracks if t["class"] == "vehicle"]
        self.assertEqual(len(vehicle_tracks), 2)

        # Check pedestrian track
        pedestrian_tracks = [t for t in tracks if t["class"] == "pedestrian"]
        self.assertEqual(len(pedestrian_tracks), 1)

    def test_track_hallucinations_with_existing_tracks(self) -> None:
        """Test track_hallucinations with existing tracks."""
        results = self.base_results.copy()

        # Add existing tracks
        results["hallucination_tracks"] = [
            {
                "id": 1,
                "class": "vehicle",
                "detections": [0],  # First vehicle detection
                "relevancy": 0,
            }
        ]

        # Add new detections for current frame
        results["hallucination_detections"]["vehicle"].extend(
            [
                {"frame_idx": 1, "bbox_xywh": [105, 105, 55, 55], "mask_ratio": 0.8},  # Overlaps with first detection
                {"frame_idx": 1, "bbox_xywh": [400, 400, 50, 50], "mask_ratio": 0.85},  # New detection
            ]
        )

        start_indices = {"vehicle": 2}  # New detections start from index 2

        track_hallucinations(results, start_indices)

        tracks = results["hallucination_tracks"]

        # Should have 2 tracks total: extended existing track + new track
        self.assertEqual(len(tracks), 2)

        # First track should be extended
        track1 = tracks[0]
        self.assertEqual(track1["id"], 1)
        self.assertEqual(len(track1["detections"]), 2)  # Original + new overlapping detection
        self.assertIn(2, track1["detections"])  # Should include new overlapping detection

        # Second track should be new
        track2 = tracks[1]
        self.assertEqual(track2["id"], 2)
        self.assertEqual(len(track2["detections"]), 1)
        self.assertIn(3, track2["detections"])  # Should include new non-overlapping detection

    def test_track_hallucinations_overlap_detection(self) -> None:
        """Test overlap detection between bounding boxes."""
        # Start with detection from frame 0
        results: dict[str, Any] = {
            "hallucination_detections": {
                "vehicle": [
                    {"frame_idx": 0, "bbox_xywh": [100, 100, 50, 50], "mask_ratio": 0.8}  # Detection 0
                ]
            },
            "hallucination_tracks": [],
        }

        # Frame 0: Process initial detection
        track_hallucinations(results, {"vehicle": 0})

        # Frame 1: Add new detections and process them
        results["hallucination_detections"]["vehicle"].extend(
            [
                {
                    "frame_idx": 1,
                    "bbox_xywh": [110, 110, 50, 50],
                    "mask_ratio": 0.8,
                },  # Detection 1 - overlaps with 0
                {
                    "frame_idx": 1,
                    "bbox_xywh": [200, 200, 50, 50],
                    "mask_ratio": 0.8,
                },  # Detection 2 - non-overlapping
            ]
        )
        track_hallucinations(results, {"vehicle": 1})

        tracks = results["hallucination_tracks"]

        # Should have 2 tracks: one extended, one new
        self.assertEqual(len(tracks), 2)

        # Find tracks by detection count
        extended_track = next(t for t in tracks if len(t["detections"]) == 2)
        new_track = next(t for t in tracks if len(t["detections"]) == 1 and t["id"] != extended_track["id"])

        # Extended track should contain overlapping detections
        self.assertIn(0, extended_track["detections"])
        self.assertIn(1, extended_track["detections"])

        # New track should contain non-overlapping detection
        self.assertIn(2, new_track["detections"])

    def test_track_hallucinations_no_overlap(self) -> None:
        """Test track_hallucinations when no overlap is detected."""
        # Start with detection from frame 0
        results: dict[str, Any] = {
            "hallucination_detections": {
                "vehicle": [
                    {"frame_idx": 0, "bbox_xywh": [100, 100, 50, 50], "mask_ratio": 0.8}  # Detection 0
                ]
            },
            "hallucination_tracks": [],
        }

        # Frame 0: Process initial detection
        track_hallucinations(results, {"vehicle": 0})

        # Frame 1: Add new non-overlapping detection and process it
        results["hallucination_detections"]["vehicle"].append(
            {
                "frame_idx": 1,
                "bbox_xywh": [300, 300, 50, 50],
                "mask_ratio": 0.8,
            }  # Detection 1 - far away, no overlap
        )
        track_hallucinations(results, {"vehicle": 1})

        tracks = results["hallucination_tracks"]

        # Should have 2 separate tracks
        self.assertEqual(len(tracks), 2)
        self.assertEqual(len(tracks[0]["detections"]), 1)
        self.assertEqual(len(tracks[1]["detections"]), 1)

    def test_track_hallucinations_relevancy_update_with_road(self) -> None:
        """Test relevancy update when detection overlaps with road pixels."""
        results = self.base_results.copy()
        start_indices = {"vehicle": 0}

        # Create road mask with road pixels at detection bbox location (100-150, 100-150)
        road_mask = np.zeros((200, 200), dtype=bool)
        road_mask[100:150, 100:150] = True

        track_hallucinations(results, start_indices, road_mask=road_mask)

        tracks = results["hallucination_tracks"]
        vehicle_track = next(t for t in tracks if t["class"] == "vehicle")

        # Should have increased relevancy due to road overlap
        self.assertGreater(vehicle_track["relevancy"], 0)

    def test_track_hallucinations_relevancy_update_no_road(self) -> None:
        """Test relevancy update when detection doesn't overlap with road pixels."""
        results = self.base_results.copy()
        start_indices = {"vehicle": 0}

        # Create road mask with no road pixels at detection locations
        road_mask = np.zeros((200, 200), dtype=bool)

        track_hallucinations(results, start_indices, road_mask=road_mask)

        tracks = results["hallucination_tracks"]
        vehicle_track = next(t for t in tracks if t["class"] == "vehicle")

        # Should have zero relevancy due to no road overlap
        self.assertEqual(vehicle_track["relevancy"], 0)

    def test_track_hallucinations_missing_bbox(self) -> None:
        """Test track_hallucinations with detection missing bbox_xywh."""
        results = {
            "hallucination_detections": {
                "vehicle": [
                    {"frame_idx": 0, "mask_ratio": 0.8}  # Missing bbox_xywh
                ]
            },
            "hallucination_tracks": [],
        }

        start_indices = {"vehicle": 0}

        track_hallucinations(results, start_indices)

        # Should not create any tracks due to missing bbox
        self.assertEqual(len(results["hallucination_tracks"]), 0)

    def test_track_hallucinations_invalid_detection_bucket(self) -> None:
        """Test track_hallucinations with invalid detection bucket."""
        results = {
            "hallucination_detections": {
                "vehicle": "not_a_list"  # Invalid type
            },
            "hallucination_tracks": [],
        }

        start_indices = {"vehicle": 0}

        track_hallucinations(results, start_indices)

        # Should handle gracefully and not create tracks
        self.assertEqual(len(results["hallucination_tracks"]), 0)

    def test_track_hallucinations_out_of_bounds_start_index(self) -> None:
        """Test track_hallucinations with start_index beyond detection list length."""
        results = self.base_results.copy()
        start_indices = {"vehicle": 10}  # Beyond list length

        track_hallucinations(results, start_indices)

        # Should not process any detections or create tracks
        self.assertEqual(len(results["hallucination_tracks"]), 0)

    def test_track_hallucinations_missing_tracks_key(self) -> None:
        """Test track_hallucinations when results doesn't have hallucination_tracks key."""
        results = {
            "hallucination_detections": {
                "vehicle": [{"frame_idx": 0, "bbox_xywh": [100, 100, 50, 50], "mask_ratio": 0.8}]
            }
            # Missing hallucination_tracks key
        }

        start_indices = {"vehicle": 0}

        track_hallucinations(results, start_indices)

        # Should create tracks key and add tracks
        self.assertIn("hallucination_tracks", results)
        self.assertEqual(len(results["hallucination_tracks"]), 1)

    def test_track_hallucinations_track_id_counter_initialization(self) -> None:
        """Test that track ID counter is properly initialized from existing tracks."""
        results: dict[str, Any] = {
            "hallucination_detections": {
                "vehicle": [{"frame_idx": 0, "bbox_xywh": [100, 100, 50, 50], "mask_ratio": 0.8}]
            },
            "hallucination_tracks": [{"id": 5, "class": "vehicle", "detections": [], "relevancy": 0}],
        }

        start_indices = {"vehicle": 0}

        track_hallucinations(results, start_indices)

        # New track should have ID > 5
        new_track = next(t for t in results["hallucination_tracks"] if t["id"] > 5)
        self.assertEqual(new_track["id"], 6)

    def test_track_hallucinations_multiple_previous_detections(self) -> None:
        """Test tracking when comparing against multiple previous detections."""
        results: dict[str, Any] = {
            "hallucination_detections": {
                "vehicle": [
                    {"frame_idx": 0, "bbox_xywh": [100, 100, 50, 50], "mask_ratio": 0.8},
                    {"frame_idx": 1, "bbox_xywh": [110, 110, 50, 50], "mask_ratio": 0.8},
                    {"frame_idx": 2, "bbox_xywh": [105, 105, 50, 50], "mask_ratio": 0.8},  # Overlaps with both previous
                ]
            },
            "hallucination_tracks": [
                {
                    "id": 1,
                    "class": "vehicle",
                    "detections": [0, 1],  # Track with 2 previous detections
                    "relevancy": 0,
                }
            ],
        }

        start_indices = {"vehicle": 2}  # Process only the new detection

        track_hallucinations(results, start_indices)

        tracks = results["hallucination_tracks"]

        # Should still have only 1 track, extended with new detection
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0]["detections"]), 3)
        self.assertIn(2, tracks[0]["detections"])

    def test_track_hallucinations_with_none_road_mask(self) -> None:
        """Test track_hallucinations when road_mask is None."""
        results = self.base_results.copy()
        start_indices = {"vehicle": 0}

        # Pass None for road_mask (should skip relevancy updates)
        track_hallucinations(results, start_indices, road_mask=None)

        # Should still create tracks
        self.assertGreater(len(results["hallucination_tracks"]), 0)

        # Relevancy should remain 0 since no road mask was provided
        vehicle_track = next(t for t in results["hallucination_tracks"] if t["class"] == "vehicle")
        self.assertEqual(vehicle_track["relevancy"], 0)


class TestProximityAssociation(unittest.TestCase):
    """Proximity association links a moving blob whose frame-to-frame boxes no longer overlap."""

    def _track_all(self, proximity_by_class: dict[str, float] | None) -> dict[str, Any]:
        # A 20x20 traffic light jumping 30px/frame: consecutive boxes never overlap. Detections are appended one per
        # frame and tracked immediately, mirroring the per-frame call in object_processor_static.
        results: dict[str, Any] = {"hallucination_detections": {"TrafficLight": []}, "hallucination_tracks": []}
        bucket = results["hallucination_detections"]["TrafficLight"]
        for frame in range(3):
            start = len(bucket)
            bucket.append({"frame_idx": frame, "bbox_xywh": [frame * 30, 100, 20, 20], "mask_ratio": 1.0})
            track_hallucinations(results, {"TrafficLight": start}, proximity_by_class=proximity_by_class)
        return results

    def test_overlap_only_splits_moving_blob_into_separate_tracks(self) -> None:
        results = self._track_all(proximity_by_class=None)  # shipping behavior: overlap only
        self.assertEqual(len(results["hallucination_tracks"]), 3)  # no box overlaps -> three length-1 tracks

    def test_proximity_links_moving_blob_into_one_track(self) -> None:
        results = self._track_all(proximity_by_class={"TrafficLight": 30.0})  # centers 30px apart -> linked
        self.assertEqual(len(results["hallucination_tracks"]), 1)
        self.assertEqual(len(results["hallucination_tracks"][0]["detections"]), 3)

    def test_proximity_zero_or_absent_class_is_overlap_only(self) -> None:
        results = self._track_all(proximity_by_class={"Vehicle": 30.0})  # TrafficLight absent -> unchanged
        self.assertEqual(len(results["hallucination_tracks"]), 3)


class TestConfirmHallucinationTracks(unittest.TestCase):
    """Persistence gate drops tracks that don't span min_track_frames consecutive frames, per class."""

    def _results(self) -> dict[str, Any]:
        return {
            "hallucination_detections": {
                "TrafficLight": [
                    {"frame_idx": 5, "bbox_xywh": [0, 0, 10, 10]},  # single-frame track
                    {"frame_idx": 10, "bbox_xywh": [0, 0, 10, 10]},  # 2 consecutive frames
                    {"frame_idx": 11, "bbox_xywh": [0, 0, 10, 10]},
                ]
            },
            "hallucination_tracks": [
                {"id": 1, "class": "TrafficLight", "detections": [0], "relevancy": 0},
                {"id": 2, "class": "TrafficLight", "detections": [1, 2], "relevancy": 0},
            ],
        }

    def test_min_frames_two_drops_single_frame_track(self) -> None:
        results = self._results()
        removed = confirm_hallucination_tracks(results, {"TrafficLight": 2})
        self.assertEqual(removed, 1)
        ids = [t["id"] for t in results["hallucination_tracks"]]
        self.assertEqual(ids, [2])  # only the 2-consecutive-frame track survives

    def test_default_keeps_all_tracks(self) -> None:
        results = self._results()
        self.assertEqual(confirm_hallucination_tracks(results, None), 0)
        self.assertEqual(len(results["hallucination_tracks"]), 2)
        self.assertEqual(confirm_hallucination_tracks(results, {"TrafficLight": 1}), 0)
        self.assertEqual(len(results["hallucination_tracks"]), 2)

    def test_other_class_unaffected(self) -> None:
        results = self._results()
        # A gate on a different class leaves TrafficLight tracks (which default to 1) untouched.
        self.assertEqual(confirm_hallucination_tracks(results, {"Vehicle": 5}), 0)
        self.assertEqual(len(results["hallucination_tracks"]), 2)

    def test_max_consecutive_frames(self) -> None:
        self.assertEqual(_max_consecutive_frames([]), 0)
        self.assertEqual(_max_consecutive_frames([5]), 1)
        self.assertEqual(_max_consecutive_frames([10, 11, 12, 20]), 3)
        self.assertEqual(_max_consecutive_frames([3, 3, 4]), 2)  # dedups then counts consecutive


def _ground_detection(frame_idx: int, points: list[list[float]]) -> dict[str, Any]:
    return {"frame_idx": frame_idx, "bbox_xywh": [0, 0, 2, 2], "mask_ratio": 1.0, "ground_xy": points}


class RegroupGroundTracksTest(unittest.TestCase):
    """World-space grouping: the batch alternative to per-frame bbox association."""

    def setUp(self) -> None:
        checks.objects.single_view.hallucination_tracking._TRACK_ID_COUNTER = 0

    def test_fragments_of_one_line_across_frames_become_one_track(self) -> None:
        """The case the bbox tracker cannot serve: a dashed line seen over many frames arrives as many small
        fragments, whose image positions move as the ego drives but whose world positions do not."""
        results: dict[str, Any] = {
            "hallucination_detections": {
                "LaneLine": [
                    _ground_detection(0, [[0.0, 0.0], [2.0, 0.0]]),
                    _ground_detection(1, [[2.2, 0.0], [4.0, 0.0]]),
                    _ground_detection(2, [[4.2, 0.0], [6.0, 0.0]]),
                ]
            },
            "hallucination_tracks": [],
        }
        written = regroup_ground_tracks(results, ["LaneLine"], resolution_m=0.1, thickness_px=7)
        self.assertEqual(written, 1)
        self.assertEqual(results["hallucination_tracks"][0]["detections"], [0, 1, 2])
        self.assertEqual(results["hallucination_tracks"][0]["class"], "LaneLine")

    def test_parallel_lanes_stay_separate(self) -> None:
        """Dilation has to bridge dashes without reaching the next lane, ~3 m away."""
        results: dict[str, Any] = {
            "hallucination_detections": {
                "LaneLine": [
                    _ground_detection(0, [[0.0, 0.0], [5.0, 0.0]]),
                    _ground_detection(0, [[0.0, 3.0], [5.0, 3.0]]),
                ]
            },
            "hallucination_tracks": [],
        }
        self.assertEqual(regroup_ground_tracks(results, ["LaneLine"], resolution_m=0.1, thickness_px=7), 2)

    def test_other_classes_are_left_alone(self) -> None:
        """Only the configured classes are regrouped; everything else keeps the tracks it already had."""
        results: dict[str, Any] = {
            "hallucination_detections": {"LaneLine": [_ground_detection(0, [[0.0, 0.0], [1.0, 0.0]])]},
            "hallucination_tracks": [{"id": 99, "class": "TrafficLight", "detections": [0, 1], "relevancy": 3}],
        }
        regroup_ground_tracks(results, ["LaneLine"], resolution_m=0.1, thickness_px=1)
        kept = [t for t in results["hallucination_tracks"] if t["class"] == "TrafficLight"]
        self.assertEqual(kept, [{"id": 99, "class": "TrafficLight", "detections": [0, 1], "relevancy": 3}])

    def test_a_detection_without_ground_points_still_gets_a_track(self) -> None:
        """The detector keeps a detection whose back-projection failed, treating it as losing the world
        position rather than the detection. Regrouping replaces the class's tracks wholesale, so it has to
        carry that detection too or it belongs to no track at all and disappears from any track-based reader."""
        results: dict[str, Any] = {
            "hallucination_detections": {"LaneLine": [{"frame_idx": 0, "bbox_xywh": [0, 0, 2, 2]}]},
            "hallucination_tracks": [],
        }
        self.assertEqual(regroup_ground_tracks(results, ["LaneLine"], resolution_m=0.1, thickness_px=1), 1)
        self.assertEqual(results["hallucination_tracks"][0]["detections"], [0])

    def test_every_detection_lands_in_exactly_one_track(self) -> None:
        """Mixed bucket: two groupable fragments of one line, plus one with no ground position."""
        results: dict[str, Any] = {
            "hallucination_detections": {
                "LaneLine": [
                    _ground_detection(0, [[0.0, 0.0], [2.0, 0.0]]),
                    {"frame_idx": 1, "bbox_xywh": [0, 0, 2, 2]},
                    _ground_detection(2, [[2.2, 0.0], [4.0, 0.0]]),
                ]
            },
            "hallucination_tracks": [],
        }
        regroup_ground_tracks(results, ["LaneLine"], resolution_m=0.1, thickness_px=7)
        assigned = [i for t in results["hallucination_tracks"] for i in t["detections"]]
        self.assertEqual(sorted(assigned), [0, 1, 2])
        self.assertEqual(len(assigned), len(set(assigned)), "a detection must not be in two tracks")

    def test_no_classes_is_a_noop(self) -> None:
        results: dict[str, Any] = {"hallucination_detections": {}, "hallucination_tracks": [{"id": 1}]}
        self.assertEqual(regroup_ground_tracks(results, []), 0)
        self.assertEqual(results["hallucination_tracks"], [{"id": 1}])

    def test_track_ids_are_unique_across_classes(self) -> None:
        results: dict[str, Any] = {
            "hallucination_detections": {
                "LaneLine": [_ground_detection(0, [[0.0, 0.0], [1.0, 0.0]])],
                "RoadBoundary": [_ground_detection(0, [[50.0, 50.0], [51.0, 50.0]])],
            },
            "hallucination_tracks": [],
        }
        regroup_ground_tracks(results, ["LaneLine", "RoadBoundary"], resolution_m=0.1, thickness_px=1)
        ids = [t["id"] for t in results["hallucination_tracks"]]
        self.assertEqual(len(ids), len(set(ids)))


class ConfirmTracksTotalFrameModeTest(unittest.TestCase):
    """A world-space track may vanish behind a vehicle and come back; it is still one track."""

    @staticmethod
    def _results_with_gap() -> dict[str, Any]:
        return {
            "hallucination_detections": {"LaneLine": [{"frame_idx": f, "bbox_xywh": [0, 0, 2, 2]} for f in (0, 5, 9)]},
            "hallucination_tracks": [{"id": 1, "class": "LaneLine", "detections": [0, 1, 2], "relevancy": 0}],
        }

    def test_consecutive_mode_drops_a_gapped_track(self) -> None:
        results = self._results_with_gap()
        self.assertEqual(confirm_hallucination_tracks(results, {"LaneLine": 3}), 1)
        self.assertEqual(results["hallucination_tracks"], [])

    def test_total_frame_mode_keeps_it(self) -> None:
        results = self._results_with_gap()
        removed = confirm_hallucination_tracks(results, {"LaneLine": 3}, total_frame_classes=["LaneLine"])
        self.assertEqual(removed, 0)
        self.assertEqual(len(results["hallucination_tracks"]), 1)

    def test_total_frame_mode_still_drops_a_short_track(self) -> None:
        results = self._results_with_gap()
        results["hallucination_tracks"][0]["detections"] = [0]
        removed = confirm_hallucination_tracks(results, {"LaneLine": 3}, total_frame_classes=["LaneLine"])
        self.assertEqual(removed, 1)


if __name__ == "__main__":
    unittest.main()
