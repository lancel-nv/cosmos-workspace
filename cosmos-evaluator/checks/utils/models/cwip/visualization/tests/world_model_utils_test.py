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

from checks.utils.models.cwip.visualization.world_model_utils import (
    ObjectType,
    PatchSimilarityType,
    PatchSimilarityType2Color,
    get_object_type_colors,
    get_similarity_type_colors,
    object_type_to_category_and_name_in_json,
)


class TestPatchSimilarityType(unittest.TestCase):
    """Tests for PatchSimilarityType enum."""

    def test_enum_values(self) -> None:
        """Test that PatchSimilarityType enum has correct values."""
        self.assertEqual(PatchSimilarityType.DONT_CARE, 0)
        self.assertEqual(PatchSimilarityType.MATCH, 1)
        self.assertEqual(PatchSimilarityType.WRONG, 2)
        self.assertEqual(PatchSimilarityType.EXTRA, 3)
        self.assertEqual(PatchSimilarityType.MISSING, 4)

        # Test enum has 5 members
        self.assertEqual(len(PatchSimilarityType), 5)

    def test_color_mapping(self) -> None:
        """Test that color mapping contains all patch similarity types."""
        # Check all types have colors
        for patch_type in PatchSimilarityType:
            self.assertIn(patch_type, PatchSimilarityType2Color)

        # Check colors are RGB tuples with 3 values
        for color in PatchSimilarityType2Color.values():
            self.assertEqual(len(color), 3)
            self.assertTrue(all(0.0 <= c <= 1.0 for c in color))

    def test_specific_colors(self) -> None:
        """Test specific color values for patch similarity types."""
        # Check specific known colors
        self.assertEqual(PatchSimilarityType2Color[PatchSimilarityType.DONT_CARE], (0, 0, 0))  # black
        self.assertEqual(PatchSimilarityType2Color[PatchSimilarityType.MATCH], (0, 1, 0))  # green
        self.assertEqual(PatchSimilarityType2Color[PatchSimilarityType.WRONG], (1, 0, 0))  # red
        self.assertEqual(PatchSimilarityType2Color[PatchSimilarityType.EXTRA], (0, 1, 1))  # cyan
        self.assertEqual(PatchSimilarityType2Color[PatchSimilarityType.MISSING], (1, 1, 0))  # yellow


class TestObjectType(unittest.TestCase):
    """Tests for ObjectType enum."""

    def test_enum_values(self) -> None:
        """Test that ObjectType enum has expected values."""
        # Background / Special
        self.assertEqual(ObjectType.BACKGROUND, 0)
        self.assertEqual(ObjectType.EDGE, 1)
        self.assertEqual(ObjectType.AMBIGUOUS, 2)

        # Bounding boxes
        self.assertEqual(ObjectType.BBOX_CAR, 3)
        self.assertEqual(ObjectType.BBOX_TRUCK, 4)
        self.assertEqual(ObjectType.BBOX_PEDESTRIAN, 5)
        self.assertEqual(ObjectType.BBOX_CYCLIST, 6)
        self.assertEqual(ObjectType.BBOX_OTHERS, 7)

        # Traffic lights
        self.assertEqual(ObjectType.TRAFFIC_LIGHT_RED, 8)
        self.assertEqual(ObjectType.TRAFFIC_LIGHT_YELLOW, 9)
        self.assertEqual(ObjectType.TRAFFIC_LIGHT_GREEN, 10)
        self.assertEqual(ObjectType.TRAFFIC_LIGHT_UNKNOWN, 11)

        # HD Map
        self.assertEqual(ObjectType.HDMAP_CROSSWALKS, 12)
        self.assertEqual(ObjectType.HDMAP_ROAD_BOUNDARIES, 13)
        self.assertEqual(ObjectType.HDMAP_WAIT_LINES, 14)
        self.assertEqual(ObjectType.HDMAP_POLES, 15)
        self.assertEqual(ObjectType.HDMAP_TRAFFIC_SIGNS, 16)
        self.assertEqual(ObjectType.HDMAP_ROAD_MARKINGS, 17)

        # Geometry lanelines (18-32)
        self.assertEqual(ObjectType.GEOMETRY_LANELINE_WHITE_SOLID_SINGLE, 18)
        self.assertEqual(ObjectType.GEOMETRY_LANELINE_WHITE_LONG_DASHED_SINGLE, 19)
        self.assertEqual(ObjectType.GEOMETRY_LANELINE_WHITE_SHORT_DASHED_SINGLE, 20)
        self.assertEqual(ObjectType.GEOMETRY_LANELINE_WHITE_DOT_DASHED_SINGLE, 21)
        self.assertEqual(ObjectType.GEOMETRY_LANELINE_WHITE_OTHER, 22)
        self.assertEqual(ObjectType.GEOMETRY_LANELINE_WHITE_SOLID_GROUP, 23)
        self.assertEqual(ObjectType.GEOMETRY_LANELINE_YELLOW_DOT_SOLID_SINGLE, 24)
        self.assertEqual(ObjectType.GEOMETRY_LANELINE_YELLOW_DOT_SOLID_GROUP, 25)
        self.assertEqual(ObjectType.GEOMETRY_LANELINE_YELLOW_DASHED_SOLID, 26)
        self.assertEqual(ObjectType.GEOMETRY_LANELINE_YELLOW_SOLID_DASHED, 27)
        self.assertEqual(ObjectType.GEOMETRY_LANELINE_YELLOW_SOLID_SINGLE, 28)
        self.assertEqual(ObjectType.GEOMETRY_LANELINE_YELLOW_SOLID_GROUP, 29)
        self.assertEqual(ObjectType.GEOMETRY_LANELINE_YELLOW_LONG_DASHED_SINGLE, 30)
        self.assertEqual(ObjectType.GEOMETRY_LANELINE_YELLOW_OTHER, 31)
        self.assertEqual(ObjectType.GEOMETRY_LANELINE_OTHER, 32)

        # Test enum has 33 members (0-32)
        self.assertEqual(len(ObjectType), 33)


class TestGetObjectTypeColors(unittest.TestCase):
    """Tests for get_object_type_colors function."""

    def test_returns_dict(self) -> None:
        """Test that get_object_type_colors returns a valid color dictionary."""
        colors = get_object_type_colors()

        # Should return a dictionary
        self.assertIsInstance(colors, dict)

        # Should have entries for all object types (at least 22)
        self.assertGreaterEqual(len(colors), 22)

        # All keys should be integers (type indices)
        self.assertTrue(all(isinstance(k, int) for k in colors))

        # All values should be RGB lists with 3 floats in [0, 1]
        for rgb in colors.values():
            self.assertIsInstance(rgb, list)
            self.assertEqual(len(rgb), 3)
            self.assertTrue(all(isinstance(c, float) for c in rgb))
            self.assertTrue(all(0.0 <= c <= 1.0 for c in rgb))

    def test_specific_colors(self) -> None:
        """Test specific color values for known object types."""
        colors = get_object_type_colors()

        # Background should be black
        self.assertEqual(colors[ObjectType.BACKGROUND], [0.0, 0.0, 0.0])

        # Edge should be gray
        self.assertEqual(colors[ObjectType.EDGE], [0.78, 0.78, 0.78])

        # Ambiguous should be bright lime green
        self.assertEqual(colors[ObjectType.AMBIGUOUS], [0.0, 1.0, 0.0])


class TestGetSimilarityTypeColors(unittest.TestCase):
    """Tests for get_similarity_type_colors function."""

    def test_default(self) -> None:
        """Test default behavior with include_ignored=True."""
        colors = get_similarity_type_colors()

        # Should have 5 base classes + 1 ignored
        self.assertEqual(len(colors), 6)

        # Check base classes are present
        for i in range(5):
            self.assertIn(i, colors)
            self.assertEqual(len(colors[i]), 3)

        # Ignored should be at index 5 (num_classes)
        self.assertIn(5, colors)

    def test_without_ignored(self) -> None:
        """Test with include_ignored=False."""
        colors = get_similarity_type_colors(include_ignored=False)

        # Should have only 5 base classes
        self.assertEqual(len(colors), 5)
        self.assertNotIn(5, colors)

    def test_custom_num_classes(self) -> None:
        """Test with custom num_classes."""
        colors = get_similarity_type_colors(include_ignored=True, num_classes=10)

        # Ignored should be at index 10
        self.assertIn(10, colors)

    def test_color_values_in_range(self) -> None:
        """Test all color values are in [0, 1]."""
        colors = get_similarity_type_colors()
        for color in colors.values():
            for c in color:
                self.assertGreaterEqual(c, 0.0)
                self.assertLessEqual(c, 1.0)


class TestObjectTypeToCategoryAndNameInJson(unittest.TestCase):
    """Tests for object_type_to_category_and_name_in_json function."""

    def test_bbox_types(self) -> None:
        """Test bbox type conversion."""
        category, name = object_type_to_category_and_name_in_json(ObjectType.BBOX_CAR)
        self.assertEqual(category, "bbox")
        self.assertEqual(name, "Car")

        category, name = object_type_to_category_and_name_in_json(ObjectType.BBOX_TRUCK)
        self.assertEqual(category, "bbox")
        self.assertEqual(name, "Truck")

        category, name = object_type_to_category_and_name_in_json(ObjectType.BBOX_PEDESTRIAN)
        self.assertEqual(category, "bbox")
        self.assertEqual(name, "Pedestrian")

        category, name = object_type_to_category_and_name_in_json(ObjectType.BBOX_CYCLIST)
        self.assertEqual(category, "bbox")
        self.assertEqual(name, "Cyclist")

        category, name = object_type_to_category_and_name_in_json(ObjectType.BBOX_OTHERS)
        self.assertEqual(category, "bbox")
        self.assertEqual(name, "Others")

    def test_traffic_light_types(self) -> None:
        """Test traffic light type conversion."""
        category, name = object_type_to_category_and_name_in_json(ObjectType.TRAFFIC_LIGHT_RED)
        self.assertEqual(category, "traffic_light")
        self.assertEqual(name, "red")

        category, name = object_type_to_category_and_name_in_json(ObjectType.TRAFFIC_LIGHT_YELLOW)
        self.assertEqual(category, "traffic_light")
        self.assertEqual(name, "yellow")

        category, name = object_type_to_category_and_name_in_json(ObjectType.TRAFFIC_LIGHT_GREEN)
        self.assertEqual(category, "traffic_light")
        self.assertEqual(name, "green")

        category, name = object_type_to_category_and_name_in_json(ObjectType.TRAFFIC_LIGHT_UNKNOWN)
        self.assertEqual(category, "traffic_light")
        self.assertEqual(name, "unknown")

    def test_hdmap_types(self) -> None:
        """Test hdmap type conversion."""
        category, name = object_type_to_category_and_name_in_json(ObjectType.HDMAP_CROSSWALKS)
        self.assertEqual(category, "hdmap")
        self.assertEqual(name, "crosswalks")

        category, name = object_type_to_category_and_name_in_json(ObjectType.HDMAP_ROAD_BOUNDARIES)
        self.assertEqual(category, "hdmap")
        self.assertEqual(name, "road_boundaries")

        category, name = object_type_to_category_and_name_in_json(ObjectType.HDMAP_POLES)
        self.assertEqual(category, "hdmap")
        self.assertEqual(name, "poles")

    def test_geometry_laneline_types(self) -> None:
        """Test geometry laneline type conversion."""
        category, name = object_type_to_category_and_name_in_json(ObjectType.GEOMETRY_LANELINE_WHITE_SOLID_SINGLE)
        self.assertEqual(category, "geometry_laneline")
        self.assertEqual(name, "WHITE SOLID_SINGLE")

        category, name = object_type_to_category_and_name_in_json(ObjectType.GEOMETRY_LANELINE_YELLOW_OTHER)
        self.assertEqual(category, "geometry_laneline")
        self.assertEqual(name, "YELLOW OTHER")

    def test_special_types_raise(self) -> None:
        """Test that special types raise ValueError."""
        with self.assertRaises(ValueError):
            object_type_to_category_and_name_in_json(ObjectType.BACKGROUND)
        with self.assertRaises(ValueError):
            object_type_to_category_and_name_in_json(ObjectType.EDGE)
        with self.assertRaises(ValueError):
            object_type_to_category_and_name_in_json(ObjectType.AMBIGUOUS)


class TestGetObjectTypeColorsAdditional(unittest.TestCase):
    """Additional tests for get_object_type_colors function."""

    def test_returns_dict(self) -> None:
        """Test that get_object_type_colors returns a dictionary."""
        colors = get_object_type_colors()
        self.assertIsInstance(colors, dict)

    def test_has_all_object_types(self) -> None:
        """Test that all ObjectType values have colors."""
        colors = get_object_type_colors()
        for obj_type in ObjectType:
            self.assertIn(obj_type, colors, f"Missing color for {obj_type}")

    def test_colors_in_valid_range(self) -> None:
        """Test that all color values are in [0, 1] range."""
        colors = get_object_type_colors()
        for obj_type, rgb in colors.items():
            self.assertEqual(len(rgb), 3, f"Color for {obj_type} should have 3 channels")
            for c in rgb:
                self.assertGreaterEqual(c, 0.0, f"Color channel for {obj_type} should be >= 0")
                self.assertLessEqual(c, 1.0, f"Color channel for {obj_type} should be <= 1")

    def test_special_types_have_expected_colors(self) -> None:
        """Test that special types have expected colors."""
        colors = get_object_type_colors()
        # BACKGROUND should be black
        self.assertEqual(colors[ObjectType.BACKGROUND], [0.0, 0.0, 0.0])
        # EDGE should be gray
        self.assertAlmostEqual(colors[ObjectType.EDGE][0], 0.78, places=1)
        # AMBIGUOUS should be green
        self.assertEqual(colors[ObjectType.AMBIGUOUS], [0.0, 1.0, 0.0])


class TestGetSimilarityTypeColorsAdditional(unittest.TestCase):
    """Additional tests for get_similarity_type_colors function."""

    def test_returns_dict(self) -> None:
        """Test that get_similarity_type_colors returns a dictionary."""
        colors = get_similarity_type_colors()
        self.assertIsInstance(colors, dict)

    def test_has_all_similarity_types(self) -> None:
        """Test that all PatchSimilarityType values have colors."""
        colors = get_similarity_type_colors()
        for sim_type in PatchSimilarityType:
            self.assertIn(sim_type, colors, f"Missing color for {sim_type}")


class TestGetObjectTypeColorsEdgeCases(unittest.TestCase):
    """Tests for edge cases in get_object_type_colors."""

    def test_get_object_type_colors_basic(self) -> None:
        """Test basic get_object_type_colors returns all types."""
        colors = get_object_type_colors()
        self.assertIsInstance(colors, dict)
        # Should contain BACKGROUND, EDGE, AMBIGUOUS at minimum
        self.assertIn(ObjectType.BACKGROUND, colors)
        self.assertIn(ObjectType.EDGE, colors)
        self.assertIn(ObjectType.AMBIGUOUS, colors)
        # Colors should be [r, g, b] in [0, 1]
        for color in colors.values():
            self.assertEqual(len(color), 3)
            for c in color:
                self.assertGreaterEqual(c, 0.0)
                self.assertLessEqual(c, 1.0)


class TestGetSimilarityTypeColorsNumClassesNone(unittest.TestCase):
    """Test get_similarity_type_colors with num_classes=None (covers L87)."""

    def test_default_num_classes(self) -> None:
        """When num_classes=None, should use len(PatchSimilarityType)."""
        colors = get_similarity_type_colors(num_classes=None, include_ignored=False)
        self.assertIn(0, colors)
        self.assertEqual(len(colors), len(PatchSimilarityType))

    def test_with_ignored(self) -> None:
        """When include_ignored=True, should have extra color."""
        colors = get_similarity_type_colors(num_classes=None, include_ignored=True)
        self.assertEqual(len(colors), len(PatchSimilarityType) + 1)


class TestObjectTypeToCategoryEdgeCases(unittest.TestCase):
    """Test object_type_to_category_and_name_in_json edge cases (covers L750-751)."""

    def test_special_types_raise_value_error(self) -> None:
        """BACKGROUND, EDGE, AMBIGUOUS are not mapped to JSON categories."""
        for obj_type in [ObjectType.BACKGROUND, ObjectType.EDGE, ObjectType.AMBIGUOUS]:
            with self.assertRaises(ValueError):
                object_type_to_category_and_name_in_json(obj_type)


class TestGetObjectTypeColorsSpecialTypes(unittest.TestCase):
    """Test get_object_type_colors special types (covers L1406-1407, 1426-1428, 1442)."""

    def test_returns_colors_for_all_types(self) -> None:
        """get_object_type_colors should return a color for every ObjectType."""
        colors = get_object_type_colors()
        for obj_type in ObjectType:
            self.assertIn(obj_type, colors)
            self.assertEqual(len(colors[obj_type]), 3)

    def test_special_types_have_correct_colors(self) -> None:
        """Background, Edge, Ambiguous should have known fixed colors."""
        colors = get_object_type_colors()
        self.assertAlmostEqual(colors[ObjectType.BACKGROUND][0], 0.0)
        self.assertAlmostEqual(colors[ObjectType.BACKGROUND][1], 0.0)
        self.assertAlmostEqual(colors[ObjectType.BACKGROUND][2], 0.0)
        # Edge should be grayish
        self.assertGreater(colors[ObjectType.EDGE][0], 0.5)
        # Ambiguous should be green
        self.assertAlmostEqual(colors[ObjectType.AMBIGUOUS][1], 1.0)


if __name__ == "__main__":
    unittest.main()
