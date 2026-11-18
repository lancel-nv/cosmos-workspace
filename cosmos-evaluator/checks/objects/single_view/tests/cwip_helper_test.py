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
from unittest.mock import MagicMock

import numpy as np
import torch

from checks.objects.single_view.cwip_helper import (
    COSMOS_EVALUATOR_OBJECT_TYPE_STRING_TO_ENUMS,
    CWIPInferenceHelper,
    resize_index_map,
    resolve_model_class_indices,
)
from checks.utils.models.cwip.modules.class_mapping import (
    OBJECT_TYPE_CLASS_MAPPING_REGISTRY,
    ObjectTypeClassMappingV3,
)
from checks.utils.models.cwip.visualization.world_model_utils import ObjectType


def _make_helper_with_mapping(mapping_name: str | None) -> CWIPInferenceHelper:
    """Create a CWIPInferenceHelper stub with the given object_type_class_mapping."""
    helper = object.__new__(CWIPInferenceHelper)
    helper.config = MagicMock()
    helper.config.object_type_class_mapping = mapping_name
    return helper


class TestResolveModelClassIndices(unittest.TestCase):
    """Tests for _resolve_model_class_indices helper."""

    def test_none_mapping_returns_enum_values(self) -> None:
        types = [ObjectType.BBOX_CAR, ObjectType.BBOX_TRUCK]
        result = resolve_model_class_indices(types, None)
        self.assertEqual(sorted(result), sorted([int(ObjectType.BBOX_CAR), int(ObjectType.BBOX_TRUCK)]))

    def test_v3_mapping_remaps_indices(self) -> None:
        types = [ObjectType.BBOX_CAR, ObjectType.BBOX_TRUCK]
        result = resolve_model_class_indices(types, ObjectTypeClassMappingV3)
        expected = [
            ObjectTypeClassMappingV3.get_model_class(ObjectType.BBOX_CAR),
            ObjectTypeClassMappingV3.get_model_class(ObjectType.BBOX_TRUCK),
        ]
        self.assertEqual(sorted(result), sorted(expected))

    def test_v3_skips_ignored_types(self) -> None:
        types = [ObjectType.GEOMETRY_LANELINE_OTHER, ObjectType.GEOMETRY_LANELINE_WHITE_SOLID_GROUP]
        result = resolve_model_class_indices(types, ObjectTypeClassMappingV3)
        # GEOMETRY_LANELINE_OTHER is IGNORED in V3, only WHITE_SOLID_GROUP should remain
        expected_idx = ObjectTypeClassMappingV3.get_model_class(ObjectType.GEOMETRY_LANELINE_WHITE_SOLID_GROUP)
        self.assertEqual(result, [expected_idx])

    def test_v3_deduplicates_same_model_class(self) -> None:
        types = [
            ObjectType.GEOMETRY_LANELINE_WHITE_SOLID_GROUP,
            ObjectType.GEOMETRY_LANELINE_WHITE_SOLID_SINGLE,
        ]
        result = resolve_model_class_indices(types, ObjectTypeClassMappingV3)
        # Both map to the same class 15 in V3
        self.assertEqual(len(result), 1)


class TestGetClassMask(unittest.TestCase):
    """Tests for CWIPInferenceHelper.get_class_mask."""

    def test_no_mapping_uses_default_v3(self) -> None:
        """When no class mapping is configured, the default (V3) mapping is used."""
        helper = _make_helper_with_mapping(None)

        car_idx = ObjectTypeClassMappingV3.get_model_class(ObjectType.BBOX_CAR)
        truck_idx = ObjectTypeClassMappingV3.get_model_class(ObjectType.BBOX_TRUCK)
        ped_idx = ObjectTypeClassMappingV3.get_model_class(ObjectType.BBOX_PEDESTRIAN)

        similarity_layer = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.long)
        type_layer = torch.tensor(
            [
                [car_idx, ped_idx, car_idx],
                [car_idx, truck_idx, ped_idx],
            ],
            dtype=torch.long,
        )
        pred = torch.stack([similarity_layer, type_layer], dim=0)

        mask = helper.get_class_mask(pred, object_type_class_name="vehicle")

        # sim=1 AND (type=BBOX_CAR OR type=BBOX_TRUCK)
        expected = np.array([[True, False, False], [False, True, False]])
        np.testing.assert_array_equal(mask, expected)

    def test_v3_mapping_uses_remapped_indices(self) -> None:
        """V3 mapping: model outputs different indices than ObjectType enum values."""
        helper = _make_helper_with_mapping("v3")

        car_idx = ObjectTypeClassMappingV3.get_model_class(ObjectType.BBOX_CAR)
        truck_idx = ObjectTypeClassMappingV3.get_model_class(ObjectType.BBOX_TRUCK)
        ped_idx = ObjectTypeClassMappingV3.get_model_class(ObjectType.BBOX_PEDESTRIAN)

        similarity_layer = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.long)
        type_layer = torch.tensor(
            [
                [car_idx, ped_idx, car_idx],
                [car_idx, truck_idx, ped_idx],
            ],
            dtype=torch.long,
        )
        pred = torch.stack([similarity_layer, type_layer], dim=0)

        mask = helper.get_class_mask(pred, object_type_class_name="vehicle")

        expected = np.array([[True, False, False], [False, True, False]])
        np.testing.assert_array_equal(mask, expected)

    def test_invalid_class_name_raises(self) -> None:
        helper = _make_helper_with_mapping(None)
        pred = torch.zeros((2, 2, 2), dtype=torch.long)
        with self.assertRaises(ValueError, msg="Invalid object type class name"):
            helper.get_class_mask(pred, object_type_class_name="nonexistent_class")

    def test_invalid_pred_shape_raises(self) -> None:
        helper = _make_helper_with_mapping(None)
        pred = torch.zeros((3, 2, 2), dtype=torch.long)
        with self.assertRaises(ValueError, msg="Pred shape must be"):
            helper.get_class_mask(pred, object_type_class_name="vehicle")

    def test_none_class_name_returns_unfiltered(self) -> None:
        """When object_type_class_name is None, return similarity & type indices."""
        helper = _make_helper_with_mapping(None)
        similarity_layer = torch.tensor([[1, 0], [1, 1]], dtype=torch.long)
        type_layer = torch.tensor([[5, 3], [0, 7]], dtype=torch.long)
        pred = torch.stack([similarity_layer, type_layer], dim=0)

        mask = helper.get_class_mask(pred)

        # sim & type_indices: (1&5, 0&3, 1&0, 1&7) → bool of (5, 0, 0, 7)
        expected = np.array([[True, False], [False, True]])
        np.testing.assert_array_equal(mask, expected)


class TestClassMappingContract(unittest.TestCase):
    """Contract tests that fail when the CWIP model's class mapping API diverges from what the objects processor
    expects.

    These iterate over *all* registered mapping versions so that adding a new version (e.g., V5) that silently drops
    support for a class will be caught immediately rather than surfacing as a silent scoring failure at runtime.
    """

    def test_all_object_types_are_valid_enum_members(self) -> None:
        """Every ObjectType referenced in COSMOS_EVALUATOR_OBJECT_TYPE_STRING_TO_ENUMS must be a current member of the
        ObjectType enum."""
        all_enum_members = set(ObjectType)
        for class_name, obj_types in COSMOS_EVALUATOR_OBJECT_TYPE_STRING_TO_ENUMS.items():
            for obj_type in obj_types:
                self.assertIn(
                    obj_type,
                    all_enum_members,
                    f"COSMOS_EVALUATOR_OBJECT_TYPE_STRING_TO_ENUMS['{class_name}'] references "
                    f"{obj_type!r} which is not in the ObjectType enum",
                )

    def test_every_class_resolves_for_all_registered_mappings(self) -> None:
        """For every registered class mapping version, every high-level class in
        COSMOS_EVALUATOR_OBJECT_TYPE_STRING_TO_ENUMS must resolve to at least one non-IGNORED model class index.

        If a new mapping version makes an entire class invisible (all types IGNORED or missing), the objects processor
        would silently produce zero scores for that class.
        """
        for version, mapping_cls in OBJECT_TYPE_CLASS_MAPPING_REGISTRY.items():
            for class_name, obj_types in COSMOS_EVALUATOR_OBJECT_TYPE_STRING_TO_ENUMS.items():
                resolved = resolve_model_class_indices(obj_types, mapping_cls)
                self.assertGreater(
                    len(resolved),
                    0,
                    f"Class '{class_name}' resolves to zero model indices under "
                    f"mapping version '{version}' ({mapping_cls.__name__}). "
                    f"All ObjectTypes {[t.name for t in obj_types]} are either "
                    f"IGNORED or missing from the mapping.",
                )

    def test_all_object_types_present_in_comprehensive_mappings(self) -> None:
        """Every ObjectType referenced by the processor should have an explicit entry (even if IGNORED) in each
        comprehensive mapping.  A missing key indicates the mapping author forgot to account for a type the processor
        relies on.

        Legacy/partial mappings (e.g., V1 which predates the full laneline enum) are skipped here; their class-level
        coverage is still verified by test_every_class_resolves_for_all_registered_mappings.
        """
        all_object_types = set(ObjectType)
        for version, mapping_cls in OBJECT_TYPE_CLASS_MAPPING_REGISTRY.items():
            mapping_domain = set(mapping_cls.OBJECT_TYPE_TO_MODEL_CLASS.keys())
            if mapping_domain < all_object_types:
                continue
            for class_name, obj_types in COSMOS_EVALUATOR_OBJECT_TYPE_STRING_TO_ENUMS.items():
                for obj_type in obj_types:
                    self.assertIn(
                        obj_type,
                        mapping_domain,
                        f"ObjectType {obj_type.name} (used by class '{class_name}') "
                        f"has no entry in mapping version '{version}' "
                        f"({mapping_cls.__name__}.OBJECT_TYPE_TO_MODEL_CLASS). "
                        f"Add an explicit mapping (or CLASS_INDEX_IGNORED) for it.",
                    )

    def test_resolved_indices_within_num_classes(self) -> None:
        """Model class indices returned by _resolve_model_class_indices must be in [0, NUM_CLASSES) for the mapping.  An
        out-of-range index would cause the argmax-produced type tensor to never match."""
        for version, mapping_cls in OBJECT_TYPE_CLASS_MAPPING_REGISTRY.items():
            for class_name, obj_types in COSMOS_EVALUATOR_OBJECT_TYPE_STRING_TO_ENUMS.items():
                resolved = resolve_model_class_indices(obj_types, mapping_cls)
                for idx in resolved:
                    self.assertGreaterEqual(
                        idx,
                        0,
                        f"Negative index {idx} for class '{class_name}' under mapping '{version}'",
                    )
                    self.assertLess(
                        idx,
                        mapping_cls.NUM_CLASSES,
                        f"Index {idx} >= NUM_CLASSES ({mapping_cls.NUM_CLASSES}) "
                        f"for class '{class_name}' under mapping '{version}'",
                    )


class TestResizeIndexMap(unittest.TestCase):
    """Tests for resize_index_map helper."""

    def test_upscale_doubles_spatial_dimensions(self) -> None:
        indices = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)
        result = resize_index_map(indices, (4, 4))
        self.assertEqual(result.shape, (1, 4, 4))
        self.assertEqual(result.dtype, torch.long)
        # Nearest-neighbor: each original pixel fills a 2x2 block
        expected = torch.tensor(
            [[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]],
            dtype=torch.long,
        )
        torch.testing.assert_close(result, expected)

    def test_downscale_halves_spatial_dimensions(self) -> None:
        indices = torch.tensor(
            [[[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]],
            dtype=torch.long,
        )
        result = resize_index_map(indices, (2, 2))
        self.assertEqual(result.shape, (1, 2, 2))
        expected = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)
        torch.testing.assert_close(result, expected)

    def test_noop_when_already_target_size(self) -> None:
        indices = torch.tensor([[[5, 6], [7, 8]]], dtype=torch.long)
        result = resize_index_map(indices, (2, 2))
        self.assertIs(result, indices)

    def test_preserves_class_values_across_batch(self) -> None:
        """Verify that class indices (not just 0/1) survive the float round-trip."""
        frame0 = torch.full((4, 4), 17, dtype=torch.long)
        frame1 = torch.full((4, 4), 42, dtype=torch.long)
        indices = torch.stack([frame0, frame1])
        result = resize_index_map(indices, (8, 8))
        self.assertTrue((result[0] == 17).all())
        self.assertTrue((result[1] == 42).all())

    def test_non_square_resize(self) -> None:
        indices = torch.tensor([[[1, 2, 3]]], dtype=torch.long)  # (1, 1, 3)
        result = resize_index_map(indices, (2, 6))
        self.assertEqual(result.shape, (1, 2, 6))
        self.assertEqual(result.dtype, torch.long)

    def test_rejects_2d_input(self) -> None:
        indices = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        with self.assertRaises(ValueError, msg="Expected 3-D tensor"):
            resize_index_map(indices, (4, 4))

    def test_rejects_4d_input(self) -> None:
        indices = torch.zeros((1, 1, 2, 2), dtype=torch.long)
        with self.assertRaises(ValueError, msg="Expected 3-D tensor"):
            resize_index_map(indices, (4, 4))

    def test_multi_frame_batch(self) -> None:
        batch = torch.zeros((5, 8, 16), dtype=torch.long)
        result = resize_index_map(batch, (4, 8))
        self.assertEqual(result.shape, (5, 4, 8))
        self.assertEqual(result.dtype, torch.long)


if __name__ == "__main__":
    unittest.main()
