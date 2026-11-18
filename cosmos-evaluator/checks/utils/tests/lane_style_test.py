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

"""Tests for the lane-line style summary."""

from typing import Any
import unittest

from checks.utils.lane_style import annotate_lane_styles, is_dashed_style, modal_lane_style


class TestModalLaneStyle(unittest.TestCase):
    def test_counts_vertices(self) -> None:
        self.assertEqual(
            modal_lane_style(["SOLID_SINGLE", "LONG_DASHED_SINGLE", "LONG_DASHED_SINGLE"]), "LONG_DASHED_SINGLE"
        )

    def test_ties_resolve_to_first_seen(self) -> None:
        self.assertEqual(modal_lane_style(["SOLID_SINGLE", "DASHED", "DASHED", "SOLID_SINGLE"]), "SOLID_SINGLE")

    def test_ignores_empty_and_non_string_values(self) -> None:
        self.assertEqual(modal_lane_style(["", None, "DASHED", 3]), "DASHED")

    def test_nothing_usable_is_none(self) -> None:
        for styles in (None, [], ["", ""], "DASHED", {"styles": ["DASHED"]}):
            with self.subTest(styles=styles):
                self.assertIsNone(modal_lane_style(styles))


class TestIsDashedStyle(unittest.TestCase):
    def test_dash_substring_in_any_case(self) -> None:
        for style in ("LONG_DASHED_SINGLE", "short_dashed_single", "DOT_DASHED_SINGLE", "Dash"):
            with self.subTest(style=style):
                self.assertIs(is_dashed_style(style), True)

    def test_hybrid_solid_and_dashed_lines_are_not_dashed(self) -> None:
        for style in ("SOLID_DASHED", "DASHED_SOLID", "solid_dashed"):
            with self.subTest(style=style):
                self.assertIs(is_dashed_style(style), False)

    def test_other_styles(self) -> None:
        for style in ("SOLID_SINGLE", "SOLID_GROUP", "DOT_SOLID_SINGLE", "OTHER", "", None):
            with self.subTest(style=style):
                self.assertIs(is_dashed_style(style), False)


class TestAnnotateLaneStyles(unittest.TestCase):
    def test_records_the_clip_level_style_on_lane_lines_only(self) -> None:
        tracks: list[dict[str, Any]] = [
            {"track_id": 1, "object_type": "LaneLine"},
            {"track_id": 2, "object_type": "RoadBoundary"},
            {"track_id": 3, "object_type": "LaneLine"},
            {"track_id": 4, "object_type": "LaneLine"},
        ]
        static_objects: dict[str, dict[str, Any]] = {
            "1": {"attributes": {"styles": ["LONG_DASHED_SINGLE", "LONG_DASHED_SINGLE"]}},
            "2": {"attributes": {"styles": ["SOLID_SINGLE"]}},
            "3": {"attributes": {}},
        }
        self.assertEqual(annotate_lane_styles(tracks, static_objects), 1)
        self.assertEqual(tracks[0]["lane_style"], "LONG_DASHED_SINGLE")
        for entry in tracks[1:]:
            self.assertNotIn("lane_style", entry)


if __name__ == "__main__":
    unittest.main()
