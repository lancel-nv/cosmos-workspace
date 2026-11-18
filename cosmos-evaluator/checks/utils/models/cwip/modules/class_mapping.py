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

from abc import ABC
import enum
from typing import ClassVar

from checks.utils.models.cwip.visualization.world_model_utils import ObjectType, PatchSimilarityType


class PatchClassificationType(enum.StrEnum):
    PATCH_SIMILARITY = "patch_similarity"
    PATCH_TYPE = "patch_type"


class ClassMapping(ABC):
    """Abstract base class for all class mappings (similarity and object type).

    Provides common constants and methods for class index management.
    """

    # Class index for ignored types in loss computation
    CLASS_INDEX_IGNORED: int = -1

    # Subclasses must define this class attribute
    NUM_CLASSES: int

    @classmethod
    def get_num_classes(cls, include_ignored_class: bool = False) -> int:
        """Get the number of classes for this mapping.

        Args:
            include_ignored_class: If True, adds 1 to account for CLASS_INDEX_IGNORED
                                   as a separate class (e.g., for tensor allocation).
                                   If False (default), returns only valid model output classes.

        Returns:
            The number of classes (optionally +1 for the ignored class)
        """
        if include_ignored_class:
            return cls.NUM_CLASSES + 1
        return cls.NUM_CLASSES


class SimilarityClassMapping(ClassMapping):
    """Abstract base class for mapping PatchSimilarityType enum values to model output class indices.

    Subclasses define specific mappings for different versions.
    """

    # Subclasses must define these class attributes
    SIMILARITY_TYPE_TO_MODEL_CLASS: ClassVar[dict[PatchSimilarityType, int]]
    MODEL_CLASS_TO_SIMILARITY_TYPE: ClassVar[dict[int, PatchSimilarityType]]

    @classmethod
    def get_model_class(cls, sim_type: PatchSimilarityType) -> int:
        """Get the model output class index for a PatchSimilarityType."""
        return cls.SIMILARITY_TYPE_TO_MODEL_CLASS[sim_type]

    @classmethod
    def get_similarity_type(cls, model_class: int) -> PatchSimilarityType:
        """Get the PatchSimilarityType for a model output class index."""
        return cls.MODEL_CLASS_TO_SIMILARITY_TYPE[model_class]

    @classmethod
    def get_ignored_similarity_types(cls) -> list[PatchSimilarityType]:
        """Get all PatchSimilarityTypes that are mapped to CLASS_INDEX_IGNORED.

        Returns:
            List of PatchSimilarityType enum values that are ignored in this mapping
        """
        return [
            sim_type
            for sim_type, model_class in cls.SIMILARITY_TYPE_TO_MODEL_CLASS.items()
            if model_class == cls.CLASS_INDEX_IGNORED
        ]


class SimilarityClassMappingV1(SimilarityClassMapping):
    """V1 similarity class mapping with 5 prediction classes.

    Maps PatchSimilarityType enum values to model output class indices.
    The model predicts 5 classes (DONT_CARE, MATCH, WRONG, EXTRA, MISSING).

    For loss masking, use IGNORED (CLASS_INDEX_IGNORED = -1).
    For GT visualization with IGNORED, use get_similarity_type_colors(include_ignored=True).
    """

    NUM_CLASSES: int = 5  # DONT_CARE, MATCH, WRONG, EXTRA, MISSING

    # Class indices (matching PatchSimilarityType enum values 0-4)
    DONT_CARE: int = 0
    MATCH: int = 1
    WRONG: int = 2
    EXTRA: int = 3
    MISSING: int = 4
    IGNORED: int = ClassMapping.CLASS_INDEX_IGNORED  # -1, used for loss masking

    # Mapping from PatchSimilarityType to model class index
    SIMILARITY_TYPE_TO_MODEL_CLASS: ClassVar[dict[PatchSimilarityType, int]] = {
        PatchSimilarityType.DONT_CARE: 0,
        PatchSimilarityType.MATCH: 1,
        PatchSimilarityType.WRONG: 2,
        PatchSimilarityType.EXTRA: 3,
        PatchSimilarityType.MISSING: 4,
    }

    MODEL_CLASS_TO_SIMILARITY_TYPE: ClassVar[dict[int, PatchSimilarityType]] = {
        0: PatchSimilarityType.DONT_CARE,
        1: PatchSimilarityType.MATCH,
        2: PatchSimilarityType.WRONG,
        3: PatchSimilarityType.EXTRA,
        4: PatchSimilarityType.MISSING,
    }


class ObjectTypeClassMapping(ClassMapping):
    """Abstract base class for mapping ObjectType enum values to model output class indices.

    Subclasses define specific mappings for different model versions.
    """

    # Subclasses must define these class attributes
    OBJECT_TYPE_TO_MODEL_CLASS: ClassVar[dict[ObjectType, int]]
    MODEL_CLASS_TO_OBJECT_TYPE: ClassVar[dict[int, ObjectType]]

    @classmethod
    def get_model_class(cls, obj_type: ObjectType) -> int:
        """Get the model output class index for an ObjectType.

        Args:
            obj_type: The ObjectType enum value

        Returns:
            The model output class index
        """
        return cls.OBJECT_TYPE_TO_MODEL_CLASS[obj_type]

    @classmethod
    def get_object_type(cls, model_class: int) -> ObjectType:
        """Get the ObjectType for a model output class index.

        Args:
            model_class: The model output class index

        Returns:
            The ObjectType enum value
        """
        return cls.MODEL_CLASS_TO_OBJECT_TYPE[model_class]

    @classmethod
    def get_ignored_object_types(cls) -> list[ObjectType]:
        """Get all ObjectTypes that are mapped to CLASS_INDEX_IGNORED.

        Returns:
            List of ObjectType enum values that are ignored in this mapping
        """
        return [
            obj_type
            for obj_type, model_class in cls.OBJECT_TYPE_TO_MODEL_CLASS.items()
            if model_class == cls.CLASS_INDEX_IGNORED
        ]


class ObjectTypeClassMappingV3(ObjectTypeClassMapping):
    """V3 class mapping with 17 classes (indices 0-16).

    Maps to only color-distinguishable types for GT extraction.
    """

    NUM_CLASSES: int = 17

    OBJECT_TYPE_TO_MODEL_CLASS: ClassVar[dict[ObjectType, int]] = {
        # Ignored types
        ObjectType.EDGE: ObjectTypeClassMapping.CLASS_INDEX_IGNORED,
        ObjectType.AMBIGUOUS: ObjectTypeClassMapping.CLASS_INDEX_IGNORED,
        ObjectType.GEOMETRY_LANELINE_OTHER: ObjectTypeClassMapping.CLASS_INDEX_IGNORED,
        ObjectType.BBOX_OTHERS: ObjectTypeClassMapping.CLASS_INDEX_IGNORED,
        # Background (0)
        ObjectType.BACKGROUND: 0,
        # Bounding boxes (1-4)
        ObjectType.BBOX_CAR: 1,
        ObjectType.BBOX_TRUCK: 2,
        ObjectType.BBOX_PEDESTRIAN: 3,
        ObjectType.BBOX_CYCLIST: 4,
        # Traffic lights (5-8)
        ObjectType.TRAFFIC_LIGHT_RED: 5,
        ObjectType.TRAFFIC_LIGHT_YELLOW: 6,
        ObjectType.TRAFFIC_LIGHT_GREEN: 7,
        ObjectType.TRAFFIC_LIGHT_UNKNOWN: 8,
        # HD Map (9-14)
        ObjectType.HDMAP_CROSSWALKS: 9,
        ObjectType.HDMAP_ROAD_BOUNDARIES: 10,
        ObjectType.HDMAP_WAIT_LINES: 11,
        ObjectType.HDMAP_POLES: 12,
        ObjectType.HDMAP_TRAFFIC_SIGNS: 13,
        ObjectType.HDMAP_ROAD_MARKINGS: 14,
        # Geometry lanelines (15-16)
        # All white laneline types map to class 15, all yellow to class 16.
        # We ignore GEOMETRY_LANELINE_OTHER as they are mostly virtual lane lines.
        # White lanelines -> class 15
        ObjectType.GEOMETRY_LANELINE_WHITE_SOLID_SINGLE: 15,
        ObjectType.GEOMETRY_LANELINE_WHITE_LONG_DASHED_SINGLE: 15,
        ObjectType.GEOMETRY_LANELINE_WHITE_SHORT_DASHED_SINGLE: 15,
        ObjectType.GEOMETRY_LANELINE_WHITE_DOT_DASHED_SINGLE: 15,
        ObjectType.GEOMETRY_LANELINE_WHITE_OTHER: 15,
        ObjectType.GEOMETRY_LANELINE_WHITE_SOLID_GROUP: 15,
        # Yellow lanelines -> class 16
        ObjectType.GEOMETRY_LANELINE_YELLOW_DOT_SOLID_SINGLE: 16,
        ObjectType.GEOMETRY_LANELINE_YELLOW_DOT_SOLID_GROUP: 16,
        ObjectType.GEOMETRY_LANELINE_YELLOW_DASHED_SOLID: 16,
        ObjectType.GEOMETRY_LANELINE_YELLOW_SOLID_DASHED: 16,
        ObjectType.GEOMETRY_LANELINE_YELLOW_SOLID_SINGLE: 16,
        ObjectType.GEOMETRY_LANELINE_YELLOW_SOLID_GROUP: 16,
        ObjectType.GEOMETRY_LANELINE_YELLOW_LONG_DASHED_SINGLE: 16,
        ObjectType.GEOMETRY_LANELINE_YELLOW_OTHER: 16,
    }

    # Reverse mapping uses representative types for classes with multiple ObjectTypes
    MODEL_CLASS_TO_OBJECT_TYPE: ClassVar[dict[int, ObjectType]] = {
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
        15: ObjectType.GEOMETRY_LANELINE_WHITE_OTHER,  # Representative for all white lanelines
        16: ObjectType.GEOMETRY_LANELINE_YELLOW_OTHER,  # Representative for all yellow lanelines
    }


class ObjectTypeClassMappingV4(ObjectTypeClassMapping):
    """V4 class mapping with 18 classes (indices 0-17).

    Same as V3 but with GEOMETRY_LANELINE_OTHER as its own class (17)
    instead of being ignored.
    """

    NUM_CLASSES: int = 18

    OBJECT_TYPE_TO_MODEL_CLASS: ClassVar[dict[ObjectType, int]] = {
        # Ignored types
        ObjectType.EDGE: ObjectTypeClassMapping.CLASS_INDEX_IGNORED,
        ObjectType.AMBIGUOUS: ObjectTypeClassMapping.CLASS_INDEX_IGNORED,
        ObjectType.BBOX_OTHERS: ObjectTypeClassMapping.CLASS_INDEX_IGNORED,
        # Background (0)
        ObjectType.BACKGROUND: 0,
        # Bounding boxes (1-4)
        ObjectType.BBOX_CAR: 1,
        ObjectType.BBOX_TRUCK: 2,
        ObjectType.BBOX_PEDESTRIAN: 3,
        ObjectType.BBOX_CYCLIST: 4,
        # Traffic lights (5-8)
        ObjectType.TRAFFIC_LIGHT_RED: 5,
        ObjectType.TRAFFIC_LIGHT_YELLOW: 6,
        ObjectType.TRAFFIC_LIGHT_GREEN: 7,
        ObjectType.TRAFFIC_LIGHT_UNKNOWN: 8,
        # HD Map (9-14)
        ObjectType.HDMAP_CROSSWALKS: 9,
        ObjectType.HDMAP_ROAD_BOUNDARIES: 10,
        ObjectType.HDMAP_WAIT_LINES: 11,
        ObjectType.HDMAP_POLES: 12,
        ObjectType.HDMAP_TRAFFIC_SIGNS: 13,
        ObjectType.HDMAP_ROAD_MARKINGS: 14,
        # Geometry lanelines (15-17)
        # White lanelines -> class 15
        ObjectType.GEOMETRY_LANELINE_WHITE_SOLID_SINGLE: 15,
        ObjectType.GEOMETRY_LANELINE_WHITE_LONG_DASHED_SINGLE: 15,
        ObjectType.GEOMETRY_LANELINE_WHITE_SHORT_DASHED_SINGLE: 15,
        ObjectType.GEOMETRY_LANELINE_WHITE_DOT_DASHED_SINGLE: 15,
        ObjectType.GEOMETRY_LANELINE_WHITE_OTHER: 15,
        ObjectType.GEOMETRY_LANELINE_WHITE_SOLID_GROUP: 15,
        # Yellow lanelines -> class 16
        ObjectType.GEOMETRY_LANELINE_YELLOW_DOT_SOLID_SINGLE: 16,
        ObjectType.GEOMETRY_LANELINE_YELLOW_DOT_SOLID_GROUP: 16,
        ObjectType.GEOMETRY_LANELINE_YELLOW_DASHED_SOLID: 16,
        ObjectType.GEOMETRY_LANELINE_YELLOW_SOLID_DASHED: 16,
        ObjectType.GEOMETRY_LANELINE_YELLOW_SOLID_SINGLE: 16,
        ObjectType.GEOMETRY_LANELINE_YELLOW_SOLID_GROUP: 16,
        ObjectType.GEOMETRY_LANELINE_YELLOW_LONG_DASHED_SINGLE: 16,
        ObjectType.GEOMETRY_LANELINE_YELLOW_OTHER: 16,
        # Other lanelines -> class 17 (V4 only)
        ObjectType.GEOMETRY_LANELINE_OTHER: 17,
    }

    MODEL_CLASS_TO_OBJECT_TYPE: ClassVar[dict[int, ObjectType]] = {
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
        15: ObjectType.GEOMETRY_LANELINE_WHITE_OTHER,  # Representative for all white lanelines
        16: ObjectType.GEOMETRY_LANELINE_YELLOW_OTHER,  # Representative for all yellow lanelines
        17: ObjectType.GEOMETRY_LANELINE_OTHER,
    }


# Registry for looking up class mappings by name (string) or returning the class directly
OBJECT_TYPE_CLASS_MAPPING_REGISTRY: dict[str, type[ObjectTypeClassMapping]] = {
    "v3": ObjectTypeClassMappingV3,
    "v4": ObjectTypeClassMappingV4,
    "default": ObjectTypeClassMappingV3,
}


def get_model_class_index_for_object_type(
    object_type: ObjectType | str,
    mapping: type[ObjectTypeClassMapping] | None,
) -> int | None:
    """Return the model output class index for the given object type if it is a real class (not ignored).

    Used e.g. in inference/validation to get the class index to mask out from type argmax
    (e.g. GEOMETRY_LANELINE_OTHER in v4). Separate from configurable excluded_model_classes_for_argmax_type.

    Args:
        object_type: ObjectType enum value or name string (e.g. "GEOMETRY_LANELINE_OTHER").
        mapping: Object type class mapping (e.g. ObjectTypeClassMappingV4). If None, returns None.

    Returns:
        The model class index for that object type, or None if not in mapping or ignored.
    """
    if mapping is None:
        return None
    if isinstance(object_type, str):
        try:
            object_type = ObjectType[object_type]
        except KeyError:
            return None
    idx = mapping.OBJECT_TYPE_TO_MODEL_CLASS.get(object_type)
    if idx is None or idx == ObjectTypeClassMapping.CLASS_INDEX_IGNORED:
        return None
    return idx


# Registry for similarity class mappings
SIMILARITY_CLASS_MAPPING_REGISTRY: dict[str, type[SimilarityClassMapping]] = {
    "v1": SimilarityClassMappingV1,
    "default": SimilarityClassMappingV1,
}


def get_object_type_class_mapping(
    mapping: str | type[ObjectTypeClassMapping] | None = None,
) -> type[ObjectTypeClassMapping]:
    """Get an ObjectTypeClassMapping class by name or return the class directly.

    This helper allows config files to use either string identifiers (for OmegaConf
    compatibility) or direct class references.

    Args:
        mapping: Either a string ("v3", "v4"), a class reference, or None (returns default).

    Returns:
        The corresponding ObjectTypeClassMapping class, or the default if input is None.

    Raises:
        ValueError: If the string identifier is not recognized.
    """
    if mapping is None:
        return OBJECT_TYPE_CLASS_MAPPING_REGISTRY["default"]
    if isinstance(mapping, str):
        if mapping not in OBJECT_TYPE_CLASS_MAPPING_REGISTRY:
            raise ValueError(
                f"Unknown object_type_class_mapping: {mapping!r}. "
                f"Valid options are: {list(OBJECT_TYPE_CLASS_MAPPING_REGISTRY.keys())}"
            )
        return OBJECT_TYPE_CLASS_MAPPING_REGISTRY[mapping]
    return mapping


def get_similarity_class_mapping(
    mapping: str | type[SimilarityClassMapping] | None = None,
) -> type[SimilarityClassMapping]:
    """Get a SimilarityClassMapping class by name or return the class directly.

    This helper allows config files to use either string identifiers (for OmegaConf
    compatibility) or direct class references.

    Args:
        mapping: Either a string ("v1"), a class reference, or None (returns default).

    Returns:
        The corresponding SimilarityClassMapping class, or the default if input is None.

    Raises:
        ValueError: If the string identifier is not recognized.
    """
    if mapping is None:
        return SIMILARITY_CLASS_MAPPING_REGISTRY["default"]
    if isinstance(mapping, str):
        if mapping not in SIMILARITY_CLASS_MAPPING_REGISTRY:
            raise ValueError(
                f"Unknown similarity_class_mapping: {mapping!r}. "
                f"Valid options are: {list(SIMILARITY_CLASS_MAPPING_REGISTRY.keys())}"
            )
        return SIMILARITY_CLASS_MAPPING_REGISTRY[mapping]
    return mapping
