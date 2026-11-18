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

"""Unit tests for the multi-piece polyline container."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from checks.utils.cuboid import Cuboid
from checks.utils.multi_polyline import MultiPolyline
from checks.utils.polyline import Polyline


def _segment(x_start: float, x_end: float, y: float = 0.0) -> Polyline:
    return Polyline(np.array([[x_start, y, 0.0], [x_end, y, 0.0]]))


class MultiPolylineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pieces = [_segment(0.0, 1.0), _segment(5.0, 6.0)]
        self.geometry = MultiPolyline(self.pieces)

    def test_requires_at_least_one_piece(self) -> None:
        with self.assertRaises(ValueError):
            MultiPolyline([])

    def test_keeps_pieces_separate(self) -> None:
        self.assertEqual(len(self.geometry.polylines), 2)

    def test_vertices_world_concatenates_members_in_order(self) -> None:
        expected = np.vstack([p.vertices_world for p in self.pieces])
        np.testing.assert_allclose(self.geometry.vertices_world, expected)

    def test_match_gate_matches_polyline(self) -> None:
        self.assertEqual(MultiPolyline.MATCH_GATE_METERS, Polyline.MATCH_GATE_METERS)

    def test_world_distance_uses_the_closest_piece(self) -> None:
        """A track is one physical line however it is split, so the nearest piece represents it."""
        probe = _segment(5.0, 6.0, y=0.2)
        self.assertAlmostEqual(self.geometry.world_distance(probe), 0.2, places=6)

    def test_world_distance_is_symmetric_between_containers(self) -> None:
        other = MultiPolyline([_segment(5.0, 6.0, y=0.25)])
        self.assertAlmostEqual(self.geometry.world_distance(other), 0.25, places=6)

    def test_world_distance_infinite_for_incompatible_geometry(self) -> None:
        cuboid = Cuboid(np.eye(4), np.array([1.0, 1.0, 1.0]))
        self.assertEqual(self.geometry.world_distance(cuboid), float("inf"))

    def test_projected_mask_unions_pieces_and_takes_nearest_depth(self) -> None:
        """The container must compose member masks, not re-implement projection."""
        first = np.zeros((4, 4), dtype=bool)
        first[0, 0] = True
        first[1, 1] = True
        first_depth = np.full((4, 4), np.inf, dtype=np.float32)
        first_depth[0, 0] = 5.0
        first_depth[1, 1] = 9.0

        second = np.zeros((4, 4), dtype=bool)
        second[1, 1] = True
        second[2, 2] = True
        second_depth = np.full((4, 4), np.inf, dtype=np.float32)
        second_depth[1, 1] = 3.0
        second_depth[2, 2] = 7.0

        with (
            patch.object(self.pieces[0], "get_projected_mask", return_value=(first, first_depth)),
            patch.object(self.pieces[1], "get_projected_mask", return_value=(second, second_depth)),
        ):
            mask, depth = self.geometry.get_projected_mask(np.eye(4), MagicMock(), 4, 4)

        np.testing.assert_array_equal(mask, first | second)
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(depth.dtype, np.float32)
        # Where the pieces overlap the nearer one wins.
        self.assertAlmostEqual(float(depth[1, 1]), 3.0)
        self.assertAlmostEqual(float(depth[0, 0]), 5.0)
        self.assertAlmostEqual(float(depth[2, 2]), 7.0)
        self.assertTrue(np.isinf(depth[3, 3]))

    def test_projected_mask_has_requested_shape(self) -> None:
        empty = np.zeros((3, 5), dtype=bool)
        empty_depth = np.full((3, 5), np.inf, dtype=np.float32)
        with (
            patch.object(self.pieces[0], "get_projected_mask", return_value=(empty, empty_depth)),
            patch.object(self.pieces[1], "get_projected_mask", return_value=(empty, empty_depth)),
        ):
            mask, depth = self.geometry.get_projected_mask(np.eye(4), MagicMock(), 5, 3)
        self.assertEqual(mask.shape, (3, 5))
        self.assertEqual(depth.shape, (3, 5))

    def test_in_frame_fraction_defaults_to_one_like_polyline(self) -> None:
        """Extended map geometry is never gated on frame-edge truncation."""
        self.assertEqual(self.geometry.in_frame_fraction(np.eye(4), MagicMock(), 640, 480), 1.0)

    def test_ego_ground_distance_is_none_like_polyline(self) -> None:
        """No centroid, so the loader records no ego distance -- same as a plain Polyline."""
        self.assertIsNone(self.geometry.ego_ground_distance_m(np.eye(4)))

    def test_projected_spine_unions_pieces_and_keeps_nearest_depth(self) -> None:
        """Centerlines combine exactly like the area masks, so a split track scores as one line."""
        first = np.zeros((4, 4), dtype=bool)
        first[0, 0] = True
        first[1, 1] = True
        first_depth = np.full((4, 4), np.inf, dtype=np.float32)
        first_depth[0, 0] = 5.0
        first_depth[1, 1] = 9.0

        second = np.zeros((4, 4), dtype=bool)
        second[1, 1] = True
        second[2, 2] = True
        second_depth = np.full((4, 4), np.inf, dtype=np.float32)
        second_depth[1, 1] = 3.0
        second_depth[2, 2] = 7.0

        with (
            patch.object(self.pieces[0], "get_projected_spine_mask", return_value=(first, first_depth)),
            patch.object(self.pieces[1], "get_projected_spine_mask", return_value=(second, second_depth)),
        ):
            mask, depth = self.geometry.get_projected_spine_mask(np.eye(4), MagicMock(), 4, 4)

        np.testing.assert_array_equal(mask, first | second)
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(depth.dtype, np.float32)
        self.assertAlmostEqual(float(depth[1, 1]), 3.0)
        self.assertAlmostEqual(float(depth[0, 0]), 5.0)
        self.assertAlmostEqual(float(depth[2, 2]), 7.0)
        self.assertTrue(np.isinf(depth[3, 3]))

    def test_projected_spine_polylines_pool_across_pieces(self) -> None:
        """Every piece's centerline is reported, kept as separate spans of the one line."""
        first = [np.array([[0.0, 0.0], [1.0, 1.0]])]
        second = [np.array([[5.0, 5.0], [6.0, 6.0]])]
        with (
            patch.object(self.pieces[0], "get_projected_spine_polylines", return_value=first),
            patch.object(self.pieces[1], "get_projected_spine_polylines", return_value=second),
        ):
            polylines = self.geometry.get_projected_spine_polylines(np.eye(4), MagicMock())
        self.assertEqual(len(polylines), 2)
        np.testing.assert_array_equal(polylines[0], first[0])
        np.testing.assert_array_equal(polylines[1], second[0])

    def test_projected_spine_polylines_are_never_empty_when_a_piece_has_one(self) -> None:
        """The override is load-bearing: the base class reports "no centerline", which for the KPI's delete test
        means falling back to box IoU on exactly the frames where a line is fragmented."""
        with patch.object(self.pieces[0], "get_projected_spine_polylines", return_value=[np.zeros((2, 2))]):
            self.assertNotEqual(self.geometry.get_projected_spine_polylines(np.eye(4), MagicMock()), [])

    def test_projected_spine_is_never_none(self) -> None:
        """Regression guard: inheriting the ``RDSGeometry`` default would silently drop multi-piece frames from
        the spine metric back to the ratio metric, scoring one lane line two different ways depending on how it
        happened to fragment that frame."""
        result = self.geometry.get_projected_spine_mask(np.eye(4), MagicMock(), 8, 8)
        self.assertIsNotNone(result)

    def test_projected_spine_has_requested_shape(self) -> None:
        empty = np.zeros((3, 5), dtype=bool)
        empty_depth = np.full((3, 5), np.inf, dtype=np.float32)
        with (
            patch.object(self.pieces[0], "get_projected_spine_mask", return_value=(empty, empty_depth)),
            patch.object(self.pieces[1], "get_projected_spine_mask", return_value=(empty, empty_depth)),
        ):
            mask, depth = self.geometry.get_projected_spine_mask(np.eye(4), MagicMock(), 5, 3)
        self.assertEqual(mask.shape, (3, 5))
        self.assertEqual(depth.shape, (3, 5))


if __name__ == "__main__":
    unittest.main()
