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

"""Canonical object class names, and matching helpers, shared by the object checks and the object-level metrics
library in :mod:`kpi.automotive`.

The matching helpers exist because the same object travels under several spellings: the RDS annotation calls it
``"TrafficLight"``, the checker config calls it ``"traffic_light"``, and a hallucination track's ``class`` field
carries whichever label the detector emitted. :func:`object_class_matches` collapses all of them onto one
comparison so callers never spell out an alias list.
"""

from typing import Any

OBJECT_CLASS_TO_RDS_TYPES: dict[str, tuple[str, ...]] = {
    "vehicle": ("Automobile", "Car", "Truck", "Vehicle", "Bus", "Train"),
    "pedestrian": ("Pedestrian", "Person"),
    "motorcycle": ("Motorcycle", "Rider", "Cyclist"),
    "bicycle": ("Bicycle", "Rider", "Cyclist"),
    "traffic_light": ("TrafficLight",),
    "traffic_sign": ("TrafficSign",),
    "lane_line": ("LaneLine",),
    "road_boundary": ("RoadBoundary",),
    "wait_line": ("WaitLine",),
    "crosswalk": ("Crosswalk",),
}


def class_name_to_rds_label(class_name: str) -> str:
    """Convert a snake_case check class name to its canonical PascalCase RDS label."""
    return "".join(word.capitalize() for word in class_name.split("_"))


def normalize_object_class_name(class_name: Any) -> str:
    """Normalize snake_case, kebab-case, and PascalCase object class names for matching."""
    return "".join(ch for ch in str(class_name or "").lower() if ch.isalnum())


def object_class_aliases(class_name: str) -> frozenset[str]:
    """Return known class-name aliases for a high-level check class."""
    aliases = {class_name, class_name_to_rds_label(class_name)}
    aliases.update(OBJECT_CLASS_TO_RDS_TYPES.get(class_name, ()))
    return frozenset(aliases)


def object_class_matches(class_name: Any, expected_class_name: str) -> bool:
    """Return whether ``class_name`` matches ``expected_class_name`` across AG/RDS naming styles."""
    normalized_class = normalize_object_class_name(class_name)
    if not normalized_class:
        return False
    return normalized_class in {
        normalize_object_class_name(alias) for alias in object_class_aliases(expected_class_name)
    }
