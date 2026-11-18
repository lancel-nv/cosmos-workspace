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

"""Unit tests for connectivity-based polyline grouping."""

import unittest

import numpy as np

from checks.utils.polyline import Polyline
from checks.utils.polyline_grouping import label_connected_polylines


def _segment(x_start: float, x_end: float, y: float = 0.0) -> Polyline:
    """A two-vertex horizontal fragment, the shape RDS-HQ actually stores."""
    return Polyline(np.array([[x_start, y, 0.0], [x_end, y, 0.0]]))


class PolylineGroupingTest(unittest.TestCase):
    def test_fragments_meeting_end_to_end_share_a_group(self) -> None:
        """The core case: a lane stored as touching fragments must come back as one line."""
        fragments = [_segment(0.0, 1.0), _segment(1.0, 2.0), _segment(2.0, 3.0)]
        self.assertEqual(label_connected_polylines(fragments), [0, 0, 0])

    def test_fragments_join_regardless_of_input_order(self) -> None:
        """Connectivity is transitive, so the middle fragment can arrive last."""
        fragments = [_segment(0.0, 1.0), _segment(2.0, 3.0), _segment(1.0, 2.0)]
        groups = label_connected_polylines(fragments)
        self.assertEqual(len(set(groups)), 1)

    def test_parallel_lines_stay_separate(self) -> None:
        """Lines a lane-width apart must not fuse -- the failure mode coarse grids cause."""
        fragments = [_segment(0.0, 5.0, y=0.0), _segment(0.0, 5.0, y=3.0)]
        self.assertEqual(label_connected_polylines(fragments), [0, 1])

    def test_collinear_but_separated_fragments_stay_separate(self) -> None:
        """A gap much larger than the resolution is a real break, not a join."""
        fragments = [_segment(0.0, 1.0), _segment(20.0, 21.0)]
        self.assertEqual(label_connected_polylines(fragments), [0, 1])

    def test_group_ids_follow_first_appearance(self) -> None:
        """Ids are positional, not OpenCV's internal labels, so callers get a stable ordering."""
        fragments = [_segment(0.0, 1.0, y=6.0), _segment(0.0, 1.0, y=0.0), _segment(1.0, 2.0, y=6.0)]
        self.assertEqual(label_connected_polylines(fragments), [0, 1, 0])

    def test_empty_input_returns_empty(self) -> None:
        self.assertEqual(label_connected_polylines([]), [])

    def test_result_is_positionally_aligned_with_input(self) -> None:
        fragments = [_segment(float(i), float(i + 1), y=3.0 * i) for i in range(6)]
        self.assertEqual(len(label_connected_polylines(fragments)), len(fragments))

    def test_rejects_non_positive_resolution(self) -> None:
        with self.assertRaises(ValueError):
            label_connected_polylines([_segment(0.0, 1.0)], resolution_m=0.0)

    def test_rejects_thickness_below_one(self) -> None:
        with self.assertRaises(ValueError):
            label_connected_polylines([_segment(0.0, 1.0)], thickness_px=0)

    def test_huge_extent_is_coarsened_rather_than_allocating_an_enormous_grid(self) -> None:
        """A kilometer-scale clip must still return a labeling instead of exhausting memory."""
        fragments = [_segment(0.0, 1.0), _segment(50_000.0, 50_001.0)]
        with self.assertLogs("checks.utils.polyline_grouping", level="WARNING") as logs:
            groups = label_connected_polylines(fragments)
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(set(groups)), 2)
        self.assertIn("coarsening", logs.output[0])

    def test_thicker_stroke_bridges_a_gap_that_a_thin_one_does_not(self) -> None:
        """Documents why the default stroke is 1: dilation is what fuses neighboring lines."""
        fragments = [_segment(0.0, 1.0, y=0.0), _segment(0.0, 1.0, y=0.4)]
        self.assertEqual(len(set(label_connected_polylines(fragments, thickness_px=1))), 2)
        self.assertEqual(len(set(label_connected_polylines(fragments, thickness_px=15))), 1)


if __name__ == "__main__":
    unittest.main()
