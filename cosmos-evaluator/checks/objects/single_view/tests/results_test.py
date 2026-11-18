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

import unittest

import numpy as np

from checks.objects.results import compute_signal_group_scores


class TestComputeSignalGroupScores(unittest.TestCase):
    def test_groups_static_object_types(self) -> None:
        score_matrix = np.array(
            [
                [0.80, 0.90, 0.70, np.nan],
                [0.85, np.nan, np.nan, 0.60],
            ]
        )
        tracks = [
            {"track_id": 10, "object_type": "LaneLine"},
            {"track_id": 20, "object_type": "TrafficSign"},
            {
                "track_id": 30,
                "object_type": "TrafficLight",
                "traffic_light_state_score": 0.92,
                "traffic_light_multi_color_score": 0.03,
                "traffic_light_state_confidence": 0.65,
            },
            {"track_id": 40, "object_type": "RoadBoundary"},
        ]
        grouped = compute_signal_group_scores(score_matrix, [10, 20, 30, 40], tracks)

        self.assertAlmostEqual(float(grouped["lane"]["mean_score"]), float(np.nanmean([0.80, 0.85])))
        self.assertEqual(grouped["lane"]["num_tracks"], 1)
        self.assertAlmostEqual(grouped["traffic_sign"]["mean_score"], 0.90)
        self.assertAlmostEqual(grouped["traffic_light"]["mean_score"], 0.70)
        self.assertAlmostEqual(grouped["traffic_light"]["traffic_light_state_score"], 0.92)
        self.assertAlmostEqual(grouped["boundary"]["mean_score"], 0.60)
        self.assertNotIn("crosswalk", grouped)

    def test_omits_groups_with_no_scores(self) -> None:
        score_matrix = np.full((2, 1), np.nan)
        grouped = compute_signal_group_scores(score_matrix, [10], [{"track_id": 10, "object_type": "LaneLine"}])
        self.assertEqual(grouped, {})


if __name__ == "__main__":
    unittest.main()
