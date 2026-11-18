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

"""The object-type registry: which RDS object types this library scores, how they group, and what is defined
for each.

This is the row axis of the object-level metrics. It exists so that facts about an object type - which group
it is scored under, what label a hallucination of it carries, whether its relevancy field means anything -
live in one place with one spelling, instead of as loose type-set constants restated at every call site.

Scope: dynamic object types, traffic lights, and the lane graph (lane lines and road boundaries). Traffic signs,
crosswalks and wait lines are not registered yet.

:func:`kind_of` returns ``None`` for anything unregistered, which callers must treat as "no verdict available"
rather than falling back to a substitute rule.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
import enum

from checks.utils.object_class_names import OBJECT_CLASS_TO_RDS_TYPES, class_name_to_rds_label


class Group(enum.StrEnum):
    """The coarse bucket an object type is scored under.

    A group bundles types scored identically. Four-wheelers and vulnerable road users are separated because
    misalignment is meaningful for a vehicle whose cuboid is large and well-determined, and is not scored for
    a VRU, whose projected box is small enough that partial containment is not distinguishable from noise.
    Traffic lights are their own group because they are the only type carrying per-frame color state, so
    ``wrong_state`` and ``conflicting`` are defined for nothing else. Lane lines and road boundaries form the lane
    graph: polylines rather than boxes, whose existence is judged by a mean-score cutoff, and for which no
    misalignment signal is defined.
    """

    FOUR_WHEELER = "four_wheeler"
    VRU = "vru"
    TRAFFIC_LIGHT = "traffic_light"
    LANE_GRAPH = "lane_graph"


class Signal(enum.StrEnum):
    """The per-object determinations this library makes."""

    MISSING = "missing"  # the world model had the object; the render did not draw it
    MISALIGNED = "misaligned"  # drawn, but only partly inside its box
    EXTRA = "extra"  # drawn without any world-model object behind it (a hallucination)
    WRONG_STATE = "wrong_state"  # drawn in a color the ground truth did not call for
    CONFLICTING = "conflicting"  # drawn showing two colors at once, so the signal is ambiguous


@dataclass(frozen=True)
class ObjectKind:
    """Everything the metrics layer needs to know about one RDS object type."""

    rds_type: str  # as it appears in a result JSON's ``tracks[].object_type``: "Car"
    class_name: str  # the checker's snake_case class: "vehicle"
    group: Group
    mode: str  # which processor emits it: "dynamic" | "static"
    # The label a hallucination of this kind carries in ``hallucination_tracks[].class``. ``None`` where the
    # RDS type maps onto more than one detector class and picking either would be a guess - see
    # :func:`hallucination_label_for`.
    hallucination_label: str | None
    # Whether ``hallucination_tracks[].relevancy`` is populated for this kind's mode. The static processor
    # runs its tracker with ``road_mask=None``, so every static hallucination - a lane line on the drivable
    # surface included - carries ``relevancy == 0`` and a relevancy gate would silently discard all of them.
    relevancy_meaningful: bool
    # Which signals are defined for this kind. A signal absent here is ``n/a`` for the type - not zero - and
    # every predicate returns ``None`` for it rather than a verdict.
    signals: frozenset[Signal] = field(default_factory=frozenset)


# The dynamic type sets. Two quirks are deliberate and should not be "tidied" without a measured before/after,
# because both change which objects get scored:
#
#   * ``Train`` and ``Automobile`` are registered as four-wheelers even though no RDS clip is known to emit
#     them. Registering a type that never appears costs nothing; failing to register one that does appear
#     silently drops it from every metric.
#   * ``OBJECT_CLASS_TO_RDS_TYPES`` maps ``Cyclist`` and ``Rider`` into *both* ``motorcycle`` and ``bicycle``,
#     so neither has a single detector class. That ambiguity survives as ``hallucination_label = None``.
_FOUR_WHEELER_TYPES: tuple[str, ...] = ("Car", "Truck", "Bus", "Vehicle", "Automobile", "Train")
_VRU_TYPES: tuple[str, ...] = ("Pedestrian", "Person", "Cyclist", "Rider", "Bicycle", "Motorcycle")
_TRAFFIC_LIGHT_TYPES: tuple[str, ...] = ("TrafficLight",)
_LANE_GRAPH_TYPES: tuple[str, ...] = ("LaneLine", "RoadBoundary")

# Misalignment is scored for four-wheelers only, and color state for traffic lights only; see :class:`Group`.
_FOUR_WHEELER_SIGNALS = frozenset({Signal.MISSING, Signal.MISALIGNED, Signal.EXTRA})
_VRU_SIGNALS = frozenset({Signal.MISSING, Signal.EXTRA})
_TRAFFIC_LIGHT_SIGNALS = frozenset({Signal.MISSING, Signal.EXTRA, Signal.WRONG_STATE, Signal.CONFLICTING})
_LANE_GRAPH_SIGNALS = frozenset({Signal.MISSING, Signal.EXTRA})


def _labels_by_rds_type() -> dict[str, set[str]]:
    """``rds_type -> {detector class labels that claim it}``, inverted from the canonical class mapping."""
    labels: dict[str, set[str]] = {}
    for class_name, rds_types in OBJECT_CLASS_TO_RDS_TYPES.items():
        for rds_type in rds_types:
            labels.setdefault(rds_type, set()).add(class_name_to_rds_label(class_name))
    return labels


_LABELS_BY_RDS_TYPE: Mapping[str, set[str]] = _labels_by_rds_type()


def _class_name_for(rds_type: str) -> str:
    """The checker class owning ``rds_type``; the first match wins for the deliberately ambiguous types."""
    for class_name, rds_types in OBJECT_CLASS_TO_RDS_TYPES.items():
        if rds_type in rds_types:
            return class_name
    return ""


def _hallucination_label_for(rds_type: str) -> str | None:
    """The detector class label a hallucination of ``rds_type`` carries, or ``None`` when ambiguous.

    The check keys hallucinations by the high-level class's canonical RDS label ("vehicle" -> "Vehicle"), not
    by the object's own RDS type - so ``Car`` and ``Person`` do not round-trip. Measured on real dynamic
    results, the labels actually emitted are Vehicle, Pedestrian, Bicycle and Motorcycle.
    """
    labels = _LABELS_BY_RDS_TYPE.get(rds_type, set())
    if len(labels) != 1:
        return None
    return next(iter(labels))


def _build_kinds() -> dict[str, ObjectKind]:
    """The registry, derived from the type sets above so a type cannot be listed in one place and not another."""
    kinds: dict[str, ObjectKind] = {}
    for group, rds_types, signals, mode, relevancy in (
        (Group.FOUR_WHEELER, _FOUR_WHEELER_TYPES, _FOUR_WHEELER_SIGNALS, "dynamic", True),
        (Group.VRU, _VRU_TYPES, _VRU_SIGNALS, "dynamic", True),
        # The static tracker runs with no road mask, so every static hallucination carries relevancy 0.
        # Gating on it would discard all of them rather than filtering any.
        (Group.TRAFFIC_LIGHT, _TRAFFIC_LIGHT_TYPES, _TRAFFIC_LIGHT_SIGNALS, "static", False),
        (Group.LANE_GRAPH, _LANE_GRAPH_TYPES, _LANE_GRAPH_SIGNALS, "static", False),
    ):
        for rds_type in rds_types:
            kinds[rds_type] = ObjectKind(
                rds_type=rds_type,
                class_name=_class_name_for(rds_type),
                group=group,
                mode=mode,
                hallucination_label=_hallucination_label_for(rds_type),
                relevancy_meaningful=relevancy,
                signals=signals,
            )
    return kinds


KINDS: Mapping[str, ObjectKind] = _build_kinds()


def state_block_object_type() -> str:
    """The object type a ``traffic_light_state`` entry belongs to.

    A result file's state block carries no object type - it is keyed by track id alone - and the objects in it
    routinely have no ``tracks`` entry to take a type from. Membership is nonetheless conclusive rather than a
    guess: the block is written only by the checker's traffic-light state stage, which returns early when the
    frame holds no traffic lights, and a traffic light is the only object type that has a state at all.

    This is a fact about the file format, stated once here, rather than a type inferred from data - which is
    what :func:`kind_of` returning ``None`` for an unregistered type continues to refuse to do.
    """
    return _TRAFFIC_LIGHT_TYPES[0]


def kind_of(object_type: str | None) -> ObjectKind | None:
    """The registry entry for ``object_type``, or ``None`` when it is unregistered.

    ``None`` is a real answer and callers must handle it: a type this library does not score has no verdict
    available, and must be reported as such rather than scored by a substitute rule.
    """
    if object_type is None:
        return None
    return KINDS.get(object_type)


def types_in_group(group: Group) -> frozenset[str]:
    """Every RDS type scored under ``group``."""
    return frozenset(kind.rds_type for kind in KINDS.values() if kind.group is group)


def class_names_in_group(group: Group) -> frozenset[str]:
    """The checker class names covering ``group``, i.e. what a hallucination of that group is matched against.

    Derived from the registry rather than restated, so it cannot drift from the type sets it is meant to
    describe: four-wheelers resolve to ``{"vehicle"}`` and VRUs to ``{"pedestrian", "bicycle", "motorcycle"}``.
    """
    return frozenset(kind.class_name for kind in KINDS.values() if kind.group is group and kind.class_name)


def dynamic_object_types() -> frozenset[str]:
    """Every registered dynamic RDS type, across groups."""
    return frozenset(kind.rds_type for kind in KINDS.values() if kind.mode == "dynamic")


def hallucination_label_for(object_type: str) -> str:
    """The class label a hallucination of ``object_type`` is filed under.

    Falls back to the RDS type itself where the mapping is ambiguous (``Cyclist``, ``Rider``). The fallback is
    deliberate: guessing ``Bicycle`` over ``Motorcycle`` would attribute counts to a class the detector may
    never have emitted.
    """
    kind = kind_of(object_type)
    if kind is None or kind.hallucination_label is None:
        return object_type
    return kind.hallucination_label


def relevancy_is_meaningful(object_type: str) -> bool:
    """Whether ``hallucination_tracks[].relevancy`` carries information for ``object_type``'s mode."""
    kind = kind_of(object_type)
    return kind is not None and kind.relevancy_meaningful


def defines_signal(object_type: str | None, signal: Signal) -> bool:
    """Whether ``signal`` is defined for ``object_type``.

    ``False`` means the signal is ``n/a`` for this type - a VRU has no misalignment verdict - which is a
    different fact from "the count is zero" and must survive into whatever this feeds.
    """
    kind = kind_of(object_type)
    return kind is not None and signal in kind.signals
