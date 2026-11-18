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

"""Unit tests for the static-object tracker."""

from typing import Any
import unittest

import numpy as np

from checks.utils.cuboid import Cuboid
from checks.utils.multi_polyline import MultiPolyline
from checks.utils.polyline import Polyline
from checks.utils.static_object_tracker import assign_static_track_ids


def _line(x_start: float, x_end: float, y: float = 0.0, n: int = 5) -> Polyline:
    """Build a horizontal polyline at constant y, sampled n times along x."""
    xs = np.linspace(x_start, x_end, n)
    verts = np.stack([xs, np.full(n, y), np.zeros(n)], axis=1)
    return Polyline(verts)


def _cuboid_at(x: float, y: float = 0.0, z: float = 0.0, lwh: tuple = (1.0, 1.0, 1.0)) -> Cuboid:
    """Build a unit-pose cuboid centered at (x, y, z)."""
    pose = np.eye(4)
    pose[0, 3] = x
    pose[1, 3] = y
    pose[2, 3] = z
    return Cuboid(pose, np.array(lwh))


def _laneline(x_start: float, x_end: float, y: float = 0.0, n: int = 5) -> dict[str, Any]:
    return {"object_type": "LaneLine", "geometry": _line(x_start, x_end, y, n)}


def _trafficlight(x: float, y: float = 0.0, z: float = 0.0) -> dict[str, Any]:
    return {"object_type": "TrafficLight", "geometry": _cuboid_at(x, y, z)}


class _CountingAllocator:
    """Deterministic ID allocator: returns "0", "1", "2", ..."""

    def __init__(self, start: int = 0) -> None:
        self._next = start
        self.calls: list[str] = []

    def __call__(self) -> str:
        sid = str(self._next)
        self._next += 1
        self.calls.append(sid)
        return sid


class TestAssignStaticTrackIds(unittest.TestCase):
    """Coverage for assign_static_track_ids."""

    def test_empty_inputs_return_empty(self) -> None:
        """No canonicals and no per-frame data -> empty result, no IDs allocated."""
        alloc = _CountingAllocator()
        result = assign_static_track_ids([], {}, alloc).per_frame
        self.assertEqual(result, {})
        self.assertEqual(alloc.calls, [])

    def test_canonical_only_no_per_frame(self) -> None:
        """Canonicals consume IDs but produce no per-frame entries."""
        alloc = _CountingAllocator()
        result = assign_static_track_ids([_laneline(0, 10)], {}, alloc).per_frame
        self.assertEqual(result, {})
        self.assertEqual(alloc.calls, ["0"])  # canonical ID still allocated

    def test_canonical_ids_allocated_first_in_order(self) -> None:
        """
        Load-bearing invariant: canonicals get IDs 0..N-1 in canonical-list order,
        before any per-frame match runs. Detections that don't match a canonical
        receive IDs N onward.
        """
        alloc = _CountingAllocator()
        canonical = [_laneline(0, 10), _laneline(0, 10, y=3.0), _laneline(0, 10, y=6.0)]
        # Frame 0 has a detection nowhere near the canonicals -> new track at ID "3"
        per_frame = {0: [_laneline(100, 110, y=50.0)]}
        result = assign_static_track_ids(canonical, per_frame, alloc).per_frame
        self.assertEqual(alloc.calls[:3], ["0", "1", "2"])  # canonicals first
        self.assertEqual(alloc.calls[3], "3")  # new synthesized track
        # The synthesized track is the only entry in frame 0 (canonicals don't match).
        self.assertEqual(set(result[0].keys()), {"3"})

    def test_identical_canonical_and_per_frame_match(self) -> None:
        """A per-frame detection identical to a canonical attaches under canonical's ID."""
        alloc = _CountingAllocator()
        canonical = [_laneline(0, 10)]
        per_frame = {0: [_laneline(0, 10)]}
        result = assign_static_track_ids(canonical, per_frame, alloc).per_frame
        # Canonical got ID "0"; per-frame matched, no new ID.
        self.assertEqual(alloc.calls, ["0"])
        self.assertEqual(set(result[0].keys()), {"0"})

    def test_unmatched_detection_creates_new_track(self) -> None:
        """
        When a per-frame detection is geometrically far from every canonical,
        a new track is allocated. The canonical ID is unaffected.
        """
        alloc = _CountingAllocator()
        canonical = [_laneline(0, 10, y=0.0)]  # near origin
        per_frame = {0: [_laneline(100, 110, y=50.0)]}  # far away
        result = assign_static_track_ids(canonical, per_frame, alloc).per_frame
        # Canonical "0" + synthesized "1"; per-frame has only the synthesized.
        self.assertEqual(alloc.calls, ["0", "1"])
        self.assertEqual(set(result[0].keys()), {"1"})

    def test_synthesized_track_persists_across_frames(self) -> None:
        """A synthesized track from frame 0 keeps its ID when re-observed at frame 3."""
        alloc = _CountingAllocator()
        canonical: list[dict[str, Any]] = []  # no canonicals
        per_frame = {
            0: [_laneline(100, 110, y=50.0)],
            3: [_laneline(100, 110, y=50.0)],  # same world geometry
        }
        result = assign_static_track_ids(canonical, per_frame, alloc).per_frame
        # Only one synthesized track ever; same ID in both frames.
        self.assertEqual(alloc.calls, ["0"])
        self.assertEqual(set(result[0].keys()), {"0"})
        self.assertEqual(set(result[3].keys()), {"0"})

    def test_dropout_and_reappearance_keeps_track(self) -> None:
        """
        World-coord matching naturally bridges dropouts: a detection at frame 0,
        absent at frame 3, reappearing at frame 6, gets the same ID throughout.
        """
        alloc = _CountingAllocator()
        canonical = [_laneline(0, 10)]
        per_frame = {
            0: [_laneline(0, 10)],
            3: [],  # dropout
            6: [_laneline(0, 10)],  # reappear at the same world position
        }
        result = assign_static_track_ids(canonical, per_frame, alloc).per_frame
        self.assertEqual(set(result[0].keys()), {"0"})
        self.assertEqual(result[3], {})
        self.assertEqual(set(result[6].keys()), {"0"})

    def test_two_cuboid_detections_compete_for_one_canonical(self) -> None:
        """
        When two per-frame cuboid detections both lie near the same canonical, only
        one binds via Hungarian; the other becomes a new synthesized track.
        """
        alloc = _CountingAllocator()
        canonical = [_trafficlight(0.0)]
        # Two detections, both within the cuboid gate of the canonical
        per_frame = {0: [_trafficlight(0.05), _trafficlight(-0.05)]}
        result = assign_static_track_ids(canonical, per_frame, alloc).per_frame
        # Canonical "0" matched once; one detection forced into a new track "1".
        self.assertEqual(alloc.calls, ["0", "1"])
        self.assertEqual(set(result[0].keys()), {"0", "1"})

    def test_overlapping_polyline_detections_collapse_into_one_track(self) -> None:
        """
        Two observations of the same physical line 10 cm apart are that one line, so grouping
        binds both to the canonical's ID instead of inventing a second track. This is the case
        per-frame assignment got wrong: RDS-HQ jitter routinely puts two observations of one lane
        inside the gate, and the loser used to mint a spurious track.
        """
        alloc = _CountingAllocator()
        canonical = [_laneline(0, 10, y=0.0)]
        per_frame = {0: [_laneline(0, 10, y=0.05), _laneline(0, 10, y=-0.05)]}
        result = assign_static_track_ids(canonical, per_frame, alloc).per_frame
        self.assertEqual(alloc.calls, ["0"])
        self.assertEqual(set(result[0].keys()), {"0"})

    def test_disjoint_fragments_in_one_frame_become_a_multi_polyline(self) -> None:
        """
        A line broken mid-frame (occlusion, or an unannotated stretch) keeps its pieces separate
        rather than being concatenated across the gap, which would project lane paint that is not
        there. Both pieces still belong to the one track.
        """
        alloc = _CountingAllocator()
        canonical = [_laneline(0, 30, y=0.0)]
        per_frame = {0: [_laneline(0, 8, y=0.0), _laneline(20, 30, y=0.0)]}
        result = assign_static_track_ids(canonical, per_frame, alloc).per_frame
        self.assertEqual(set(result[0].keys()), {"0"})
        geometry = result[0]["0"]["geometry"]
        self.assertIsInstance(geometry, MultiPolyline)
        self.assertEqual(len(geometry.polylines), 2)

    def test_fragments_of_one_line_share_a_track_across_frames(self) -> None:
        """Fragments are grouped by connectivity in world space, so the track spans every frame."""
        alloc = _CountingAllocator()
        canonical = [_laneline(0, 10, y=0.0)]
        per_frame = {
            0: [_laneline(0, 5, y=0.0), _laneline(5, 10, y=0.0)],
            1: [_laneline(0, 5, y=0.0), _laneline(5, 10, y=0.0)],
        }
        result = assign_static_track_ids(canonical, per_frame, alloc).per_frame
        self.assertEqual(set(result[0].keys()), {"0"})
        self.assertEqual(set(result[1].keys()), {"0"})

    def test_parallel_lanes_keep_distinct_tracks(self) -> None:
        """Grouping must not fuse laterally adjacent lanes -- the risk a coarse grid would create."""
        alloc = _CountingAllocator()
        canonical = [_laneline(0, 10, y=0.0), _laneline(0, 10, y=3.0)]
        per_frame = {0: [_laneline(0, 10, y=0.02), _laneline(0, 10, y=3.02)]}
        result = assign_static_track_ids(canonical, per_frame, alloc).per_frame
        self.assertEqual(set(result[0].keys()), {"0", "1"})

    def test_cross_type_isolation(self) -> None:
        """
        A Cuboid detection at the same world position as a Polyline canonical
        must NOT match: world_distance returns +inf for incompatible types.
        """
        alloc = _CountingAllocator()
        canonical = [_laneline(0, 10, y=0.0)]
        # Cuboid centered at (5, 0) -- spatially "in" the polyline but wrong type
        per_frame = {0: [_trafficlight(5.0, 0.0, 0.0)]}
        result = assign_static_track_ids(canonical, per_frame, alloc).per_frame
        # Canonical polyline got "0"; cuboid forced into a new track "1".
        self.assertEqual(alloc.calls, ["0", "1"])
        self.assertEqual(set(result[0].keys()), {"1"})

    def test_determinism_repeated_runs_match(self) -> None:
        """
        Repeated calls with the same inputs (and a deterministic allocator)
        yield identical track-ID assignments.
        """
        canonical = [_laneline(0, 10, y=0.0), _laneline(0, 10, y=3.0)]
        per_frame = {
            0: [_laneline(0, 10, y=0.1), _laneline(0, 10, y=3.1)],
            1: [_laneline(0, 10, y=3.05), _laneline(0, 10, y=0.05)],  # reordered
        }
        run1 = assign_static_track_ids(canonical, per_frame, _CountingAllocator()).per_frame
        run2 = assign_static_track_ids(canonical, per_frame, _CountingAllocator()).per_frame
        # Same ID assignments for each frame.
        self.assertEqual(set(run1[0].keys()), set(run2[0].keys()))
        self.assertEqual(set(run1[1].keys()), set(run2[1].keys()))

    def test_three_detections_three_canonicals_all_match(self) -> None:
        """When detections clearly map 1-to-1 onto canonicals, Hungarian binds them all."""
        alloc = _CountingAllocator()
        canonical = [_laneline(0, 10, y=0.0), _laneline(0, 10, y=3.0), _laneline(0, 10, y=6.0)]
        per_frame = {
            0: [
                _laneline(0, 10, y=6.05),  # nearest canonical[2]
                _laneline(0, 10, y=0.05),  # nearest canonical[0]
                _laneline(0, 10, y=3.05),  # nearest canonical[1]
            ]
        }
        result = assign_static_track_ids(canonical, per_frame, alloc).per_frame
        self.assertEqual(alloc.calls, ["0", "1", "2"])  # only canonicals; no new tracks
        self.assertEqual(set(result[0].keys()), {"0", "1", "2"})

    def test_canonical_with_no_match_does_not_appear_in_frames(self) -> None:
        """
        A canonical that's never observed in any per-frame data still consumes
        an ID, but doesn't appear in any frame's result dict.
        """
        alloc = _CountingAllocator()
        canonical = [_laneline(0, 10, y=0.0), _laneline(0, 10, y=100.0)]  # 2nd canonical never seen
        per_frame = {0: [_laneline(0, 10, y=0.0)]}
        result = assign_static_track_ids(canonical, per_frame, alloc).per_frame
        # Both canonicals allocated; only canonical[0] appears in frame 0.
        self.assertEqual(alloc.calls, ["0", "1"])
        self.assertEqual(set(result[0].keys()), {"0"})

    def test_per_frame_detection_dict_is_preserved_in_output(self) -> None:
        """The output stores the original per-frame detection dict unchanged."""
        alloc = _CountingAllocator()
        canonical = [_laneline(0, 10)]
        det = _laneline(0, 10)
        det["attributes"] = {"colors": ["WHITE"]}
        per_frame = {0: [det]}
        result = assign_static_track_ids(canonical, per_frame, alloc).per_frame
        # The matched track holds a reference to the detection dict.
        self.assertIs(result[0]["0"], det)
        self.assertEqual(result[0]["0"]["attributes"], {"colors": ["WHITE"]})

    def test_absorbed_canonicals_are_reported_so_the_loader_can_drop_them(self) -> None:
        """Fragments of one line that lose the ID are reported as represented elsewhere, not as unobserved."""
        alloc = _CountingAllocator()
        canonical = [_laneline(0, 5, y=0.0), _laneline(5, 10, y=0.0)]
        per_frame = {0: [_laneline(0, 10, y=0.0)]}
        assignment = assign_static_track_ids(canonical, per_frame, alloc)
        self.assertEqual(set(assignment.per_frame[0]), {"0"})
        self.assertEqual(assignment.absorbed_canonical_ids, frozenset({"1"}))

    def test_unobserved_canonical_is_not_reported_as_absorbed(self) -> None:
        """A canonical no per-frame data reaches is the only record of its geometry, so it must be kept."""
        alloc = _CountingAllocator()
        canonical = [_laneline(0, 10, y=0.0), _laneline(500, 510, y=0.0)]
        per_frame = {0: [_laneline(0, 10, y=0.0)]}
        assignment = assign_static_track_ids(canonical, per_frame, alloc)
        self.assertEqual(assignment.absorbed_canonical_ids, frozenset())

    def test_clip_geometry_is_the_greatest_coverage_observation(self) -> None:
        """``geometry`` should represent the whole line, not whichever fragment happened to come first."""
        alloc = _CountingAllocator()
        canonical = [_laneline(0, 2, y=0.0), _laneline(2, 4, y=0.0)]
        per_frame = {0: [_laneline(0, 30, y=0.0)]}
        assignment = assign_static_track_ids(canonical, per_frame, alloc)
        geometry = assignment.clip_geometry["0"]
        self.assertIsInstance(geometry, Polyline)
        assert isinstance(geometry, Polyline)  # narrows RDSGeometry for the type checker
        self.assertAlmostEqual(float(np.ptp(geometry.vertices_world[:, 0])), 30.0, places=3)

    def test_clip_geometry_can_come_from_the_canonical_when_it_covers_most(self) -> None:
        """Canonical and per-frame compete on equal terms; the longer one wins."""
        alloc = _CountingAllocator()
        canonical = [_laneline(0, 40, y=0.0)]
        per_frame = {0: [_laneline(0, 6, y=0.0)]}
        assignment = assign_static_track_ids(canonical, per_frame, alloc)
        geometry = assignment.clip_geometry["0"]
        self.assertIsInstance(geometry, Polyline)
        assert isinstance(geometry, Polyline)  # narrows RDSGeometry for the type checker
        self.assertAlmostEqual(float(np.ptp(geometry.vertices_world[:, 0])), 40.0, places=3)

    def test_cuboid_types_get_no_clip_geometry_entry(self) -> None:
        """Only polyline types are grouped, so nothing else acquires a clip-level stand-in."""
        alloc = _CountingAllocator()
        assignment = assign_static_track_ids([_trafficlight(0.0)], {0: [_trafficlight(0.0)]}, alloc)
        self.assertEqual(assignment.clip_geometry, {})
        self.assertEqual(assignment.absorbed_canonical_ids, frozenset())

    def test_non_polyline_geometry_under_a_polyline_type_is_skipped_not_fatal(self) -> None:
        """RDS-HQ derives geom_type from the shape, not the folder, so a cuboid can arrive typed "LaneLine".

        Grouping reads polyline-only fields, so such a record used to take the whole load down. It should be
        skipped with a warning while the genuine polylines of that type still group.
        """
        alloc = _CountingAllocator()
        canonical = [_laneline(0, 10, y=0.0), {"object_type": "LaneLine", "geometry": _cuboid_at(50.0)}]
        per_frame = {0: [_laneline(0, 10, y=0.0), {"object_type": "LaneLine", "geometry": _cuboid_at(50.0)}]}
        with self.assertLogs("checks.utils.static_object_tracker", level="WARNING") as logs:
            assignment = assign_static_track_ids(canonical, per_frame, alloc)
        self.assertEqual(set(assignment.per_frame[0]), {"0"})
        self.assertIn("not a Polyline", logs.output[0])


if __name__ == "__main__":
    unittest.main()
