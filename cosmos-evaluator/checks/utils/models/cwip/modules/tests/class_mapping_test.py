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

from checks.utils.models.cwip.modules.class_mapping import (
    OBJECT_TYPE_CLASS_MAPPING_REGISTRY,
    SIMILARITY_CLASS_MAPPING_REGISTRY,
    ClassMapping,
    ObjectTypeClassMapping,
    ObjectTypeClassMappingV3,
    ObjectTypeClassMappingV4,
    PatchClassificationType,
    SimilarityClassMapping,
    SimilarityClassMappingV1,
    get_object_type_class_mapping,
    get_similarity_class_mapping,
)
from checks.utils.models.cwip.visualization.world_model_utils import ObjectType, PatchSimilarityType


class TestClassMappingBase(unittest.TestCase):
    """Tests for ClassMapping base class."""

    def test_get_num_classes_without_ignored(self) -> None:
        self.assertEqual(SimilarityClassMappingV1.get_num_classes(include_ignored_class=False), 5)

    def test_get_num_classes_with_ignored(self) -> None:
        self.assertEqual(SimilarityClassMappingV1.get_num_classes(include_ignored_class=True), 6)

    def test_class_index_ignored(self) -> None:
        self.assertEqual(ClassMapping.CLASS_INDEX_IGNORED, -1)


class TestPatchClassificationType(unittest.TestCase):
    """Tests for PatchClassificationType enum."""

    def test_patch_similarity_value(self) -> None:
        self.assertEqual(PatchClassificationType.PATCH_SIMILARITY, "patch_similarity")

    def test_patch_type_value(self) -> None:
        self.assertEqual(PatchClassificationType.PATCH_TYPE, "patch_type")

    def test_is_str_enum(self) -> None:
        self.assertIsInstance(PatchClassificationType.PATCH_SIMILARITY, str)
        self.assertIsInstance(PatchClassificationType.PATCH_TYPE, str)

    def test_member_count(self) -> None:
        self.assertEqual(len(PatchClassificationType), 2)


class TestSimilarityClassMappingV1(unittest.TestCase):
    """Tests for SimilarityClassMappingV1."""

    def test_num_classes(self) -> None:
        self.assertEqual(SimilarityClassMappingV1.NUM_CLASSES, 5)

    def test_class_constants(self) -> None:
        self.assertEqual(SimilarityClassMappingV1.DONT_CARE, 0)
        self.assertEqual(SimilarityClassMappingV1.MATCH, 1)
        self.assertEqual(SimilarityClassMappingV1.WRONG, 2)
        self.assertEqual(SimilarityClassMappingV1.EXTRA, 3)
        self.assertEqual(SimilarityClassMappingV1.MISSING, 4)
        self.assertEqual(SimilarityClassMappingV1.IGNORED, ClassMapping.CLASS_INDEX_IGNORED)

    def test_get_model_class(self) -> None:
        self.assertEqual(SimilarityClassMappingV1.get_model_class(PatchSimilarityType.DONT_CARE), 0)
        self.assertEqual(SimilarityClassMappingV1.get_model_class(PatchSimilarityType.MATCH), 1)
        self.assertEqual(SimilarityClassMappingV1.get_model_class(PatchSimilarityType.WRONG), 2)
        self.assertEqual(SimilarityClassMappingV1.get_model_class(PatchSimilarityType.EXTRA), 3)
        self.assertEqual(SimilarityClassMappingV1.get_model_class(PatchSimilarityType.MISSING), 4)

    def test_get_similarity_type(self) -> None:
        self.assertEqual(SimilarityClassMappingV1.get_similarity_type(0), PatchSimilarityType.DONT_CARE)
        self.assertEqual(SimilarityClassMappingV1.get_similarity_type(1), PatchSimilarityType.MATCH)
        self.assertEqual(SimilarityClassMappingV1.get_similarity_type(2), PatchSimilarityType.WRONG)
        self.assertEqual(SimilarityClassMappingV1.get_similarity_type(3), PatchSimilarityType.EXTRA)
        self.assertEqual(SimilarityClassMappingV1.get_similarity_type(4), PatchSimilarityType.MISSING)

    def test_get_ignored_similarity_types_empty(self) -> None:
        """SimilarityClassMappingV1 has no ignored types."""
        ignored = SimilarityClassMappingV1.get_ignored_similarity_types()
        self.assertEqual(len(ignored), 0)

    def test_roundtrip(self) -> None:
        """Every similarity type roundtrips through model class and back."""
        for sim_type in PatchSimilarityType:
            model_class = SimilarityClassMappingV1.get_model_class(sim_type)
            if model_class != ClassMapping.CLASS_INDEX_IGNORED:
                back = SimilarityClassMappingV1.get_similarity_type(model_class)
                self.assertEqual(back, sim_type)

    def test_is_subclass_of_similarity_class_mapping(self) -> None:
        self.assertTrue(issubclass(SimilarityClassMappingV1, SimilarityClassMapping))
        self.assertTrue(issubclass(SimilarityClassMappingV1, ClassMapping))


class TestObjectTypeClassMappingV3(unittest.TestCase):
    """Tests for ObjectTypeClassMappingV3 (color-distinguishable types)."""

    def test_num_classes(self) -> None:
        self.assertEqual(ObjectTypeClassMappingV3.NUM_CLASSES, 17)

    def test_ignored_types(self) -> None:
        ignored = ObjectTypeClassMappingV3.get_ignored_object_types()
        self.assertIn(ObjectType.EDGE, ignored)
        self.assertIn(ObjectType.AMBIGUOUS, ignored)
        self.assertIn(ObjectType.GEOMETRY_LANELINE_OTHER, ignored)
        self.assertIn(ObjectType.BBOX_OTHERS, ignored)
        self.assertEqual(len(ignored), 4)

    def test_background_mapping(self) -> None:
        self.assertEqual(ObjectTypeClassMappingV3.get_model_class(ObjectType.BACKGROUND), 0)

    def test_bbox_mappings(self) -> None:
        self.assertEqual(ObjectTypeClassMappingV3.get_model_class(ObjectType.BBOX_CAR), 1)
        self.assertEqual(ObjectTypeClassMappingV3.get_model_class(ObjectType.BBOX_TRUCK), 2)
        self.assertEqual(ObjectTypeClassMappingV3.get_model_class(ObjectType.BBOX_PEDESTRIAN), 3)
        self.assertEqual(ObjectTypeClassMappingV3.get_model_class(ObjectType.BBOX_CYCLIST), 4)

    def test_traffic_light_mappings(self) -> None:
        self.assertEqual(ObjectTypeClassMappingV3.get_model_class(ObjectType.TRAFFIC_LIGHT_RED), 5)
        self.assertEqual(ObjectTypeClassMappingV3.get_model_class(ObjectType.TRAFFIC_LIGHT_YELLOW), 6)
        self.assertEqual(ObjectTypeClassMappingV3.get_model_class(ObjectType.TRAFFIC_LIGHT_GREEN), 7)
        self.assertEqual(ObjectTypeClassMappingV3.get_model_class(ObjectType.TRAFFIC_LIGHT_UNKNOWN), 8)

    def test_hdmap_mappings(self) -> None:
        self.assertEqual(ObjectTypeClassMappingV3.get_model_class(ObjectType.HDMAP_CROSSWALKS), 9)
        self.assertEqual(ObjectTypeClassMappingV3.get_model_class(ObjectType.HDMAP_ROAD_BOUNDARIES), 10)
        self.assertEqual(ObjectTypeClassMappingV3.get_model_class(ObjectType.HDMAP_WAIT_LINES), 11)
        self.assertEqual(ObjectTypeClassMappingV3.get_model_class(ObjectType.HDMAP_POLES), 12)
        self.assertEqual(ObjectTypeClassMappingV3.get_model_class(ObjectType.HDMAP_TRAFFIC_SIGNS), 13)
        self.assertEqual(ObjectTypeClassMappingV3.get_model_class(ObjectType.HDMAP_ROAD_MARKINGS), 14)

    def test_white_lanelines_map_to_same_class(self) -> None:
        """All white laneline types map to class 15."""
        white_types = [
            ObjectType.GEOMETRY_LANELINE_WHITE_SOLID_SINGLE,
            ObjectType.GEOMETRY_LANELINE_WHITE_LONG_DASHED_SINGLE,
            ObjectType.GEOMETRY_LANELINE_WHITE_SHORT_DASHED_SINGLE,
            ObjectType.GEOMETRY_LANELINE_WHITE_DOT_DASHED_SINGLE,
            ObjectType.GEOMETRY_LANELINE_WHITE_OTHER,
            ObjectType.GEOMETRY_LANELINE_WHITE_SOLID_GROUP,
        ]
        for obj_type in white_types:
            self.assertEqual(ObjectTypeClassMappingV3.get_model_class(obj_type), 15)

    def test_yellow_lanelines_map_to_same_class(self) -> None:
        """All yellow laneline types map to class 16."""
        yellow_types = [
            ObjectType.GEOMETRY_LANELINE_YELLOW_DOT_SOLID_SINGLE,
            ObjectType.GEOMETRY_LANELINE_YELLOW_DOT_SOLID_GROUP,
            ObjectType.GEOMETRY_LANELINE_YELLOW_DASHED_SOLID,
            ObjectType.GEOMETRY_LANELINE_YELLOW_SOLID_DASHED,
            ObjectType.GEOMETRY_LANELINE_YELLOW_SOLID_SINGLE,
            ObjectType.GEOMETRY_LANELINE_YELLOW_SOLID_GROUP,
            ObjectType.GEOMETRY_LANELINE_YELLOW_LONG_DASHED_SINGLE,
            ObjectType.GEOMETRY_LANELINE_YELLOW_OTHER,
        ]
        for obj_type in yellow_types:
            self.assertEqual(ObjectTypeClassMappingV3.get_model_class(obj_type), 16)

    def test_reverse_mapping_all_classes(self) -> None:
        """Check reverse mapping for all 17 class indices."""
        expected = {
            0: ObjectType.BACKGROUND,
            1: ObjectType.BBOX_CAR,
            2: ObjectType.BBOX_TRUCK,
            3: ObjectType.BBOX_PEDESTRIAN,
            4: ObjectType.BBOX_CYCLIST,
            5: ObjectType.TRAFFIC_LIGHT_RED,
            6: ObjectType.TRAFFIC_LIGHT_YELLOW,
            7: ObjectType.TRAFFIC_LIGHT_GREEN,
            8: ObjectType.TRAFFIC_LIGHT_UNKNOWN,
            9: ObjectType.HDMAP_CROSSWALKS,
            10: ObjectType.HDMAP_ROAD_BOUNDARIES,
            11: ObjectType.HDMAP_WAIT_LINES,
            12: ObjectType.HDMAP_POLES,
            13: ObjectType.HDMAP_TRAFFIC_SIGNS,
            14: ObjectType.HDMAP_ROAD_MARKINGS,
            15: ObjectType.GEOMETRY_LANELINE_WHITE_OTHER,
            16: ObjectType.GEOMETRY_LANELINE_YELLOW_OTHER,
        }
        for class_id, expected_type in expected.items():
            self.assertEqual(ObjectTypeClassMappingV3.get_object_type(class_id), expected_type)

    def test_is_subclass_of_object_type_class_mapping(self) -> None:
        self.assertTrue(issubclass(ObjectTypeClassMappingV3, ObjectTypeClassMapping))
        self.assertTrue(issubclass(ObjectTypeClassMappingV3, ClassMapping))

    def test_all_object_types_are_mapped(self) -> None:
        """Every ObjectType enum value should have a mapping."""
        for obj_type in ObjectType:
            self.assertIn(obj_type, ObjectTypeClassMappingV3.OBJECT_TYPE_TO_MODEL_CLASS)


class TestRegistries(unittest.TestCase):
    """Tests for class mapping registries."""

    def test_object_type_registry_has_v3(self) -> None:
        self.assertIn("v3", OBJECT_TYPE_CLASS_MAPPING_REGISTRY)
        self.assertIs(OBJECT_TYPE_CLASS_MAPPING_REGISTRY["v3"], ObjectTypeClassMappingV3)

    def test_object_type_registry_has_default(self) -> None:
        self.assertIn("default", OBJECT_TYPE_CLASS_MAPPING_REGISTRY)
        self.assertIs(OBJECT_TYPE_CLASS_MAPPING_REGISTRY["default"], ObjectTypeClassMappingV3)

    def test_similarity_registry_has_v1(self) -> None:
        self.assertIn("v1", SIMILARITY_CLASS_MAPPING_REGISTRY)
        self.assertIs(SIMILARITY_CLASS_MAPPING_REGISTRY["v1"], SimilarityClassMappingV1)

    def test_similarity_registry_has_default(self) -> None:
        self.assertIn("default", SIMILARITY_CLASS_MAPPING_REGISTRY)
        self.assertIs(SIMILARITY_CLASS_MAPPING_REGISTRY["default"], SimilarityClassMappingV1)


class TestGetObjectTypeClassMapping(unittest.TestCase):
    """Tests for get_object_type_class_mapping function."""

    def test_none_returns_default(self) -> None:
        result = get_object_type_class_mapping(None)
        self.assertIs(result, ObjectTypeClassMappingV3)

    def test_no_args_returns_default(self) -> None:
        result = get_object_type_class_mapping()
        self.assertIs(result, ObjectTypeClassMappingV3)

    def test_string_v3(self) -> None:
        self.assertIs(get_object_type_class_mapping("v3"), ObjectTypeClassMappingV3)

    def test_string_default(self) -> None:
        self.assertIs(get_object_type_class_mapping("default"), ObjectTypeClassMappingV3)

    def test_string_v4(self) -> None:
        self.assertIs(get_object_type_class_mapping("v4"), ObjectTypeClassMappingV4)

    def test_class_passthrough(self) -> None:
        self.assertIs(get_object_type_class_mapping(ObjectTypeClassMappingV3), ObjectTypeClassMappingV3)

    def test_invalid_string_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_object_type_class_mapping("v99")


class TestGetSimilarityClassMapping(unittest.TestCase):
    """Tests for get_similarity_class_mapping function."""

    def test_none_returns_default(self) -> None:
        result = get_similarity_class_mapping(None)
        self.assertIs(result, SimilarityClassMappingV1)

    def test_no_args_returns_default(self) -> None:
        result = get_similarity_class_mapping()
        self.assertIs(result, SimilarityClassMappingV1)

    def test_string_v1(self) -> None:
        self.assertIs(get_similarity_class_mapping("v1"), SimilarityClassMappingV1)

    def test_string_default(self) -> None:
        self.assertIs(get_similarity_class_mapping("default"), SimilarityClassMappingV1)

    def test_class_passthrough(self) -> None:
        self.assertIs(get_similarity_class_mapping(SimilarityClassMappingV1), SimilarityClassMappingV1)

    def test_invalid_string_raises(self) -> None:
        with self.assertRaises(ValueError):
            get_similarity_class_mapping("v99")


if __name__ == "__main__":
    unittest.main()
