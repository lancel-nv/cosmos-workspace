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

"""
Core enums and color mappings for patch similarity types and object types.
"""

import enum
from importlib.resources import as_file, files
import json

# =============================================================================
# Visualization Colors for Ignored/Unmapped Classes
# =============================================================================

# Light gray: midpoint between ``traffic_lights.unknown`` [100, 100, 100] and white [255, 255, 255] in
# ``world_scenario_color_config.json``. Used for IGNORED in both similarity and object-type GT viz.
_tl_unknown_rgb01 = 100 / 255.0
_IGNORED_LIGHT_GRAY = (_tl_unknown_rgb01 + 1.0) / 2.0
SIMILARITY_IGNORED_COLOR: tuple[float, float, float] = (
    _IGNORED_LIGHT_GRAY,
    _IGNORED_LIGHT_GRAY,
    _IGNORED_LIGHT_GRAY,
)
OBJECT_TYPE_IGNORED_COLOR: tuple[float, float, float] = SIMILARITY_IGNORED_COLOR

# Fallback color for unmapped or unknown classes (brown)
# Used when color lookup fails for any reason
FALLBACK_COLOR: tuple[float, float, float] = (0.6, 0.3, 0.1)  # brown


# =============================================================================
# Patch Similarity Types
# =============================================================================


class PatchSimilarityType(enum.IntEnum):
    """Patch similarity type - model prediction classes.

    Values 0-4 are model prediction classes. For IGNORED masking in loss computation, use
    SimilarityClassMapping.CLASS_INDEX_IGNORED (-1). For IGNORED visualization, use
    get_similarity_type_colors(include_ignored=True).
    """

    DONT_CARE = 0
    MATCH = 1
    WRONG = 2
    EXTRA = 3
    MISSING = 4


# Color mapping for prediction classes (use get_similarity_type_colors() for IGNORED)
PatchSimilarityType2Color: dict[PatchSimilarityType, tuple[float, float, float]] = {
    PatchSimilarityType.DONT_CARE: (0.0, 0.0, 0.0),  # black
    PatchSimilarityType.MATCH: (0.0, 1.0, 0.0),  # green
    PatchSimilarityType.WRONG: (1.0, 0.0, 0.0),  # red
    PatchSimilarityType.EXTRA: (0.0, 1.0, 1.0),  # cyan
    PatchSimilarityType.MISSING: (1.0, 1.0, 0.0),  # yellow
}


def get_similarity_type_colors(
    include_ignored: bool = True,
    num_classes: int | None = None,
) -> dict[int, tuple[float, float, float]]:
    """
    Get color mapping for similarity types as RGB tuples in range [0, 1].

    Args:
        include_ignored: If True, includes IGNORED color at index num_classes.
        num_classes: Number of prediction classes. If None, uses len(PatchSimilarityType). When include_ignored=True,
                     IGNORED color is placed at this index.

    Returns:
        Dictionary mapping class index (int) to RGB color tuple (r, g, b) in range [0, 1].
    """
    if num_classes is None:
        num_classes = len(PatchSimilarityType)

    # Start with base colors (cast enum keys to int)
    colors: dict[int, tuple[float, float, float]] = {int(k): v for k, v in PatchSimilarityType2Color.items()}

    if include_ignored:
        colors[num_classes] = SIMILARITY_IGNORED_COLOR  # gray at index num_classes

    return colors


# Object types in world scenario rendering
class ObjectType(enum.IntEnum):
    """Object types available in world scenario rendering.

    Continuous indices for model output classes (0-32).
    """

    # Background / Special (0-2)
    BACKGROUND = 0
    EDGE = 1
    AMBIGUOUS = 2

    # Bounding boxes (bbox category) (3-7)
    BBOX_CAR = 3
    BBOX_TRUCK = 4
    BBOX_PEDESTRIAN = 5
    BBOX_CYCLIST = 6
    BBOX_OTHERS = 7

    # Traffic lights (traffic_light category, v2 colors) (8-11)
    TRAFFIC_LIGHT_RED = 8
    TRAFFIC_LIGHT_YELLOW = 9
    TRAFFIC_LIGHT_GREEN = 10
    TRAFFIC_LIGHT_UNKNOWN = 11

    # HD Map (hdmap category) (12-17)
    # Note: hdmap.lanelines is excluded from v3 rendering (geometry_laneline types used instead)
    HDMAP_CROSSWALKS = 12
    HDMAP_ROAD_BOUNDARIES = 13
    HDMAP_WAIT_LINES = 14
    HDMAP_POLES = 15
    HDMAP_TRAFFIC_SIGNS = 16
    HDMAP_ROAD_MARKINGS = 17

    # Geometry lanelines (geometry_laneline category) (18-32)
    # Order follows config_color_geometry_laneline.json
    GEOMETRY_LANELINE_WHITE_SOLID_SINGLE = 18
    GEOMETRY_LANELINE_WHITE_LONG_DASHED_SINGLE = 19
    GEOMETRY_LANELINE_WHITE_SHORT_DASHED_SINGLE = 20
    GEOMETRY_LANELINE_WHITE_DOT_DASHED_SINGLE = 21
    GEOMETRY_LANELINE_WHITE_OTHER = 22
    GEOMETRY_LANELINE_WHITE_SOLID_GROUP = 23
    GEOMETRY_LANELINE_YELLOW_DOT_SOLID_SINGLE = 24
    GEOMETRY_LANELINE_YELLOW_DOT_SOLID_GROUP = 25
    GEOMETRY_LANELINE_YELLOW_DASHED_SOLID = 26
    GEOMETRY_LANELINE_YELLOW_SOLID_DASHED = 27
    GEOMETRY_LANELINE_YELLOW_SOLID_SINGLE = 28
    GEOMETRY_LANELINE_YELLOW_SOLID_GROUP = 29
    GEOMETRY_LANELINE_YELLOW_LONG_DASHED_SINGLE = 30
    GEOMETRY_LANELINE_YELLOW_OTHER = 31
    GEOMETRY_LANELINE_OTHER = 32


def object_type_to_category_and_name_in_json(obj_type: ObjectType) -> tuple[str, str]:
    """Convert ObjectType enum to category and object type name for color scheme lookup.

    Args:
        obj_type: ObjectType enum value (must be a type that exists in color scheme JSON files)

    Returns:
        Tuple of (category, obj_type_name) for looking up colors in color scheme configs.
        E.g., ObjectType.BBOX_CAR -> ("bbox", "Car")
              ObjectType.GEOMETRY_LANELINE_WHITE_SOLID_SINGLE -> ("geometry_laneline", "WHITE SOLID_SINGLE")

    Raises:
        ValueError: If obj_type is BACKGROUND, EDGE, AMBIGUOUS, or unknown (these types don't have entries in color
                   scheme JSON files)
    """
    name = obj_type.name  # e.g., "BBOX_CAR", "GEOMETRY_LANELINE_WHITE_SOLID_SINGLE"

    # Parse category from prefix
    if name.startswith("BBOX_"):
        category = "bbox"
        obj_name = name[5:]  # Remove "BBOX_"
        # Convert to title case: "CAR" → "Car", "OTHERS" → "Others"
        obj_name = obj_name.title()
    elif name.startswith("TRAFFIC_LIGHT_"):
        category = "traffic_light"
        obj_name = name[14:].lower()  # Remove "TRAFFIC_LIGHT_", lowercase: "RED" → "red"
    elif name.startswith("HDMAP_"):
        category = "hdmap"
        obj_name = name[6:].lower()  # Remove "HDMAP_", lowercase: "CROSSWALKS" → "crosswalks"
    elif name.startswith("GEOMETRY_LANELINE_"):
        category = "geometry_laneline"
        obj_name = name[18:]  # Remove "GEOMETRY_LANELINE_"
        # Replace only the FIRST underscore with space: "WHITE_SOLID_SINGLE" → "WHITE SOLID_SINGLE"
        # JSON keys have format "WHITE SOLID_SINGLE" (space after color, underscore elsewhere)
        obj_name = obj_name.replace("_", " ", 1)
    else:
        raise ValueError(
            f"ObjectType {obj_type} does not have a color scheme entry. "
            f"BACKGROUND, EDGE, and AMBIGUOUS are handled separately."
        )

    return (category, obj_name)


def load_color_scheme(object_category: str) -> dict[str, list | list[list]]:
    """Load color scheme for a given object category.

    Config structure (public repo cosmos_drive_dreams_toolkits):
      - world_scenario_color_config.json: bbox, hdmap, traffic_lights
      - world_scenario_laneline_config.json: geometry laneline configs

    Args:
        object_category: One of "bbox", "hdmap", "geometry_laneline", "traffic_light"

    Returns:
        Dictionary mapping object type name to RGB color list [r, g, b] (0-255) or gradient [[r1,g1,b1], [r2,g2,b2]].

    Raises:
        ValueError: If object_category is not a valid category
    """
    if object_category == "geometry_laneline":
        config_path = (
            files("third_party.cosmos_drive_dreams_toolkits") / "config" / "world_scenario_laneline_config.json"
        )
        with as_file(config_path) as p, open(p, "r") as f:
            config = json.load(f)
        return {laneline_type: props["color"] for laneline_type, props in config.items()}

    if object_category not in ("bbox", "hdmap", "traffic_light"):
        raise ValueError(
            f"Unknown object category: {object_category}. "
            f"Supported categories: bbox, hdmap, geometry_laneline, traffic_light"
        )

    config_path = files("third_party.cosmos_drive_dreams_toolkits") / "config" / "world_scenario_color_config.json"
    category_key_map = {"bbox": "bbox", "hdmap": "hdmap", "traffic_light": "traffic_lights"}

    with as_file(config_path) as p, open(p, "r") as f:
        config = json.load(f)

    key = category_key_map.get(object_category)
    if key is None or key not in config:
        raise ValueError(
            f"Category '{object_category}' not found in world_scenario_color_config.json. "
            f"Available keys: {list(config.keys())}"
        )

    return config[key]


def get_object_type_colors() -> dict[ObjectType, list[float]]:
    """
    Get color mapping for all object types (ObjectType) as [r, g, b] in range [0, 1].

    Returns:
        Dictionary mapping ObjectType to RGB color tuple [r, g, b] in range [0, 1]
    """
    # Load all color schemes
    color_schemes: dict[str, dict[str, list]] = {}
    for category in ["bbox", "traffic_light", "hdmap", "geometry_laneline"]:
        try:
            color_schemes[category] = load_color_scheme(category)
        except Exception as e:  # noqa: BLE001 - missing color scheme is non-fatal; warn and continue
            print(f"WARNING: Failed to load color scheme for {category}: {e}")

    # Build color mapping for all ObjectType values
    colors: dict[ObjectType, list[float]] = {}
    for obj_type in ObjectType:
        # Special types
        if obj_type == ObjectType.BACKGROUND:
            colors[obj_type] = [0.0, 0.0, 0.0]  # black
            continue
        if obj_type == ObjectType.EDGE:
            colors[obj_type] = [0.78, 0.78, 0.78]  # gray (200/255)
            continue
        if obj_type == ObjectType.AMBIGUOUS:
            colors[obj_type] = [0.0, 1.0, 0.0]  # bright lime green
            continue

        # Get category and object name for color lookup
        try:
            category, obj_name = object_type_to_category_and_name_in_json(obj_type)
        except ValueError:
            colors[obj_type] = [0.5, 0.5, 0.5]  # gray fallback
            continue

        # Look up color from color scheme
        if category in color_schemes and obj_name in color_schemes[category]:
            color_spec = color_schemes[category][obj_name]
            # Handle gradient vs uniform color
            if isinstance(color_spec[0], list):
                # Gradient: use middle color
                rgb = [(color_spec[0][i] + color_spec[1][i]) / 2.0 / 255.0 for i in range(3)]
            else:
                # Uniform color
                rgb = [c / 255.0 for c in color_spec]
            colors[obj_type] = rgb
        else:
            colors[obj_type] = [0.5, 0.5, 0.5]  # gray fallback

    return colors
