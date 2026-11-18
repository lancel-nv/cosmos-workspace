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

"""What makes an object *ego-important*: lane assignment, the oncoming test, and how they combine.

The single home for the rules behind the object check's ``is_important``, extracted so that the check and the
things that predict its verdict share one definition instead of each carrying a copy.

**"Important" here is a narrow, cheap heuristic** — within a distance threshold, inside a three-lane band
measured off the ego's lateral axis, not facing the ego — and it is emphatically *not* an answer to "would this
object affect the ego's trajectory". Answering that needs real lane-line geometry, lane-change tracking and the
ego's actual path, none of which this module has. The two are easy to confuse and give different answers on
purpose: a parked car one lane over is *important* here and would not affect the ego's path at all. The word
choice keeps them apart — "importance" is this heuristic (``ImportanceFilter``, ``is_important``,
``importance_matrix``), and a trajectory-impact analysis should not borrow it.

``checks.objects.single_view.importance_filter.ImportanceFilter`` is the caller inside this package: it
computes the attributes the check reports per track (``lane``, ``is_oncoming``, ``is_important``) and the
``importance_matrix``.

The rules live here rather than in the filter because a caller may need to *predict* the verdict without being
able to call the filter — reasoning from world-frame poses rather than loader objects, or asking about an
object that has no loader entry at all. Such a caller must still agree with the filter, so re-tuning the
numbers below has to move both at once; a private copy would let the two silently diverge.

Everything here is scalar geometry: no numpy, no loader types, no config objects. Callers do their own frame
transforms, hand in the results, and keep their own plumbing — which is what lets two callers with completely
different input shapes share the part that actually encodes the policy.
"""

from collections.abc import Collection

# The lane the ego occupies is one lane wide, centered on it; a lane either side is one more lane out. Objects
# beyond that are "unknown" rather than a named lane — deliberately not "far left"/"far right", because the
# check has no use for a direction it does not act on.
LANE_WIDTH_M = 3.0
HALF_LANE_WIDTH_M = LANE_WIDTH_M / 2.0
# Pedestrians and riders get a wider outer band, so that someone on the pavement beside the left lane still
# reads as being in it rather than as an unplaced "unknown". Vehicles get no such slack: a car 4.8 m off the
# ego's axis really is outside the three-lane corridor.
PEDESTRIAN_MARGIN_BAND_M = 0.5
PEDESTRIAN_TYPES: frozenset[str] = frozenset({"pedestrian", "person"})
# How far from a 180 deg relative yaw still counts as oncoming. Wide, because a vehicle approaching on a curve
# or mid-turn is oncoming in every sense that matters while being some way off a head-on heading.
ONCOMING_TOLERANCE_DEG = 30.0
# The lanes the ego has to reason about, and so the checker's configured ``relevant_lanes``: its own and one
# either side. Named here because a caller predicting the verdict needs the same set, not a re-derived one.
EGO_CORRIDOR_LANES: frozenset[str] = frozenset({"ego", "left", "right"})


def assign_object_to_lane(lateral_m: float, object_type: str = "unknown") -> str:
    """Which lane a lateral offset puts an object in: ``ego``, ``left``, ``right`` or ``unknown``.

    ``lateral_m`` is the object's offset in the ego's ground plane, positive to the **left** (the FLU
    convention). Takes the scalar rather than a position vector because callers hold it in different shapes —
    a 3-vector ``[forward, left, up]`` in the check, a 2-vector ``[forward, lateral]`` elsewhere —
    and neither should have to reshape to ask this question.

    The bands are strict on both sides, so an object exactly on a boundary lands in ``unknown``. That is the
    long-standing behavior and is left alone: at the meter scale these bands work at, exact equality is a
    float coincidence rather than a case worth naming.
    """
    margin_band_m = PEDESTRIAN_MARGIN_BAND_M if object_type.lower() in PEDESTRIAN_TYPES else 0.0
    outer_bound_m = LANE_WIDTH_M + HALF_LANE_WIDTH_M + margin_band_m
    if abs(lateral_m) < HALF_LANE_WIDTH_M:
        return "ego"
    if HALF_LANE_WIDTH_M < lateral_m < outer_bound_m:
        return "left"
    if -outer_bound_m < lateral_m < -HALF_LANE_WIDTH_M:
        return "right"
    return "unknown"


def is_oncoming(relative_yaw_deg: float, tolerance_deg: float = ONCOMING_TOLERANCE_DEG) -> bool:
    """Is an object at ``relative_yaw_deg`` in the ego's frame driving towards the ego?

    ``relative_yaw_deg`` is the object's heading measured from the ego's forward axis; oncoming is ~180 deg.
    Normalized into ``[0, 360)`` first, so it accepts either convention a caller might arrive with — the
    ``(-180, 180]`` of an ``atan2`` or the ``[0, 360)`` of an extracted Euler yaw — and agrees on both.
    """
    yaw_deg = float(relative_yaw_deg) % 360.0
    return abs(yaw_deg - 180.0) <= tolerance_deg


def is_ego_important(
    distance_m: float | None,
    distance_threshold_m: float,
    lane: str,
    relevant_lanes: Collection[str],
    oncoming: bool,
) -> bool:
    """Does an object clear all three importance conditions: close enough, in a relevant lane, not oncoming?

    Takes the sub-verdicts rather than computing them, because the check reports ``lane`` and ``is_oncoming``
    alongside the composite and would otherwise compute each twice.

    Two conventions worth stating, since both are load-bearing and neither is obvious: an **unknown distance**
    (an object with no centroid — a laneline, a road boundary) is *not* important, so a missing measurement
    fails closed rather than being waved through; and an **empty** ``relevant_lanes`` means "every lane", which
    is how a config that omits the key disables the lane test instead of rejecting everything.
    """
    within_distance = distance_m is not None and distance_m <= distance_threshold_m
    in_relevant_lane = (not relevant_lanes) or (lane in relevant_lanes)
    return within_distance and (not oncoming) and in_relevant_lane
