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

from checks.utils.object_class_names import (
    OBJECT_CLASS_TO_RDS_TYPES,
    class_name_to_rds_label,
    normalize_object_class_name,
    object_class_matches,
)


class TestObjectClassNames(unittest.TestCase):
    def test_class_name_to_rds_label(self) -> None:
        self.assertEqual(class_name_to_rds_label("traffic_light"), "TrafficLight")
        self.assertEqual(class_name_to_rds_label("traffic_sign"), "TrafficSign")
        self.assertEqual(class_name_to_rds_label("road_boundary"), "RoadBoundary")

    def test_normalize_object_class_name(self) -> None:
        self.assertEqual(normalize_object_class_name("traffic_light"), "trafficlight")
        self.assertEqual(normalize_object_class_name("TrafficLight"), "trafficlight")
        self.assertEqual(normalize_object_class_name("traffic-light"), "trafficlight")

    def test_object_class_matches_covers_snake_and_rds_labels(self) -> None:
        for class_name, rds_types in OBJECT_CLASS_TO_RDS_TYPES.items():
            self.assertTrue(object_class_matches(class_name, class_name))
            self.assertTrue(object_class_matches(class_name_to_rds_label(class_name), class_name))
            for rds_type in rds_types:
                self.assertTrue(object_class_matches(rds_type, class_name))

    def test_object_class_matches_rejects_different_classes(self) -> None:
        self.assertFalse(object_class_matches("TrafficLight", "traffic_sign"))
        self.assertFalse(object_class_matches("TrafficSign", "traffic_light"))
        self.assertFalse(object_class_matches(None, "traffic_light"))


if __name__ == "__main__":
    unittest.main()
