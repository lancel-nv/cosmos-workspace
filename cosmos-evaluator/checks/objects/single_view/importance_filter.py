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
Importance filter module for object processing.

This module implements filtering logic to determine which objects are important
enough to process based on configuration rules.

All spatial reasoning (distance, lane assignment, oncoming detection) operates in the ego-vehicle coordinate frame
(FLU: Forward-Left-Up) derived from the ego pose.  This produces correct results for all camera orientations in
multiview processing.  The ego pose is typically the front_wide camera pose (camera-to-vehicle extrinsic is
approximately identity).
"""

import logging
from typing import Any

import numpy as np

from checks.utils.coord_transforms import extract_rpy_in_flu, get_object_to_camera_pose, rdf_position_to_flu
from checks.utils.ego_importance import ONCOMING_TOLERANCE_DEG, assign_object_to_lane, is_ego_important, is_oncoming


class ImportanceFilter:
    """
    Filters objects based on importance criteria defined in configuration.

    This class applies various filters to determine if an object should be processed for correspondence analysis.
    All spatial checks use the ego-vehicle coordinate frame (FLU) for correct multiview behavior.
    """

    def __init__(self, filter_config: dict[str, Any], logger: logging.Logger | None = None) -> None:
        """
        Initialize the importance filter.

        Args:
            filter_config: Configuration dictionary for importance filtering
            logger: Logger instance to use (optional)
        """
        self.config = filter_config
        self.logger = logger

    def get_importance_attributes(
        self, tracked_object: dict[str, Any], ego_pose: np.ndarray, track_id: int | str
    ) -> dict[str, Any]:
        """Compute importance attributes for an object.

        Returns a dict with lane, is_oncoming, distance_m, and is_important. This is called for every visible object;
        is_important indicates whether the object passes all configured importance filters. ``lane`` and
        ``is_oncoming`` are dynamic-object attributes only ("n/a"/False for static objects, which are uniformly kept).

        Args:
            tracked_object: Object data from world model.
            ego_pose: Ego vehicle pose (4x4 camera-to-world matrix).
            track_id: Track ID (used for logging).

        Returns:
            Dict with keys: lane, is_oncoming, distance_m, is_important.
        """
        # Static objects: always important (allow_all_static_objects) and no lane/oncoming reasoning — but still report
        # the GT ego ground-plane distance so distance-based report breakdowns work for every object. The distance is
        # the loader-recorded ``ego_distance_m`` (RdsDataLoader.get_object_data_for_frame), so it is computed once at
        # load time rather than here; objects without a centroid pose (lanelines, road boundaries) carry ``None``.
        if tracked_object.get("is_static", True) and self.config.get("allow_all_static_objects", True):
            return {
                "lane": "n/a",
                "is_oncoming": False,
                "distance_m": tracked_object.get("ego_distance_m"),
                "is_important": True,
            }

        # Objects without centroid: can't compute spatial attributes (distance None -> excluded from distance stats)
        if not hasattr(tracked_object["geometry"], "object_to_world_pose"):
            return {"lane": "unknown", "is_oncoming": False, "distance_m": None, "is_important": False}

        # Compute position in ego FLU frame (used for lane assignment + oncoming reasoning). The ego ground-plane
        # distance itself is the loader-recorded ``ego_distance_m`` — the same value this FLU position would give
        # (sqrt(forward^2 + lateral^2) == the geometry's ego_ground_distance_m) — so distance has one source of truth.
        object_to_ego_rdf = get_object_to_camera_pose(tracked_object["geometry"], ego_pose)
        ego_position_flu = rdf_position_to_flu(object_to_ego_rdf[:3, 3])

        distance_m = tracked_object.get("ego_distance_m")

        object_type = str(tracked_object.get("object_type", "unknown"))
        lane = self._assign_object_to_lane(ego_position_flu, object_type)

        # Named ``oncoming`` so it does not shadow the shared ``is_oncoming`` predicate imported above.
        oncoming = False
        if self.config.get("skip_oncoming_obstacles", True):
            oncoming = self._is_oncoming(object_to_ego_rdf, track_id)

        # Composed in checks.utils.ego_importance, which is also what anything predicting this verdict without
        # being able to call the filter reads — see that module for why the rules live outside this class.
        is_important = is_ego_important(
            distance_m,
            self.config.get("distance_threshold_m", float("inf")),
            lane,
            self.config.get("relevant_lanes", []),
            oncoming,
        )

        return {
            "lane": lane,
            "is_oncoming": oncoming,
            "distance_m": distance_m,
            "is_important": is_important,
        }

    def should_process_object(
        self, tracked_object: dict[str, Any], ego_pose: np.ndarray, track_id: int | str
    ) -> tuple[bool, str]:
        """Determine if an object should be processed based on importance filters.

        This is the legacy API used for backward compatibility. It returns (is_important, reason) matching the old
        filter-based contract.

        Args:
            tracked_object: Object data from world model.
            ego_pose: Ego vehicle pose (4x4 camera-to-world matrix).
            track_id: Track ID (used for logging).

        Returns:
            Tuple of (should_process, reason).
        """
        attrs = self.get_importance_attributes(tracked_object, ego_pose, track_id)

        if not attrs["is_important"]:
            # Build reason string matching old format. distance_m is None for objects without a centroid pose
            # (lanelines, road boundaries); guard the numeric comparison so those fall through to the other reasons
            # instead of raising on ``None > float``.
            distance_m = attrs["distance_m"]
            if distance_m is not None and distance_m > self.config.get("distance_threshold_m", float("inf")):
                reason = f"distance_threshold_m ({self.config.get('distance_threshold_m')}m)"
            elif attrs["is_oncoming"]:
                reason = "skip_oncoming_obstacles (enabled)"
            else:
                reason = f"relevant_lanes ({self.config.get('relevant_lanes', [])})"
            return False, reason

        return True, "passed"

    def _is_oncoming(self, object_to_ego_rdf: np.ndarray, track_id: int | str) -> bool:
        """
        Check if object is oncoming relative to the ego vehicle.

        Uses the ego-frame rotation to extract yaw in FLU coordinates.
        An oncoming object has yaw ~180° (facing opposite to ego forward).

        Args:
            object_to_ego_rdf: Object pose in ego RDF coordinates (4x4 matrix).
            track_id: Track ID for logging.

        Returns:
            True if object is oncoming, False otherwise.
        """
        # Extract yaw in FLU from the ego-frame rotation, then apply the shared oncoming test
        # (checks.utils.ego_importance), which normalizes the angle and owns the tolerance.
        _, _, yaw = extract_rpy_in_flu(object_to_ego_rdf[:3, :3])
        yaw_degrees = (yaw * 180 / np.pi + 360) % 360

        oncoming = is_oncoming(yaw_degrees)

        if self.logger:
            self.logger.debug(
                "Oncoming check: Track %s, yaw=%.1f°, tolerance=±%s°, is_oncoming=%s",
                track_id,
                yaw_degrees,
                ONCOMING_TOLERANCE_DEG,
                oncoming,
            )

        return oncoming

    def _is_in_relevant_lanes(
        self, ego_position_flu: np.ndarray, tracked_object: dict[str, Any], track_id: int | str
    ) -> bool:
        """
        Check if object is in one of the relevant lanes.

        Uses the lateral (left) component of the ego-frame FLU position.

        Args:
            ego_position_flu: Object position in ego FLU coordinates [forward, left, up].
            tracked_object: Object data from world model.
            track_id: Track ID for logging.

        Returns:
            True if object is in a relevant lane.
        """
        relevant_lanes = self.config.get("relevant_lanes", [])
        if not relevant_lanes:
            return True

        object_type = str(tracked_object.get("object_type", "unknown"))
        lane = self._assign_object_to_lane(ego_position_flu, object_type)

        is_in_relevant = lane in relevant_lanes

        if self.logger:
            self.logger.debug(
                "Lane check: Track %s, lateral=%.1fm, lane=%s, relevant_lanes=%s, in_relevant=%s",
                track_id,
                ego_position_flu[1],
                lane,
                relevant_lanes,
                is_in_relevant,
            )

        return is_in_relevant

    def _assign_object_to_lane(self, ego_position_flu: np.ndarray, object_type: str = "unknown") -> str:
        """
        Determine lane assignment based on lateral position in ego FLU frame.

        In FLU: y-axis = left (positive = left of ego).
        - |y| < half_lane → ego lane
        - y > half_lane → left lane
        - y < -half_lane → right lane

        Args:
            ego_position_flu: Object position in ego FLU coordinates [forward, left, up].
            object_type: World-model object type string (e.g., "Car", "Pedestrian").

        Returns:
            Lane assignment: "ego", "left", "right", or "unknown".
        """
        # Lateral position: positive = left of ego in FLU. The bands themselves (and the wider one pedestrians
        # and riders get, so a sidewalk does not read as a lane) live in checks.utils.ego_importance, shared so
        # that a caller predicting this verdict cannot drift from the assignment made here.
        return assign_object_to_lane(float(ego_position_flu[1]), object_type)

    def _is_within_distance(self, ego_position_flu: np.ndarray, track_id: int | str) -> bool:
        """
        Check if object is within the distance threshold.

        Uses the forward and lateral components of the ego-frame FLU position
        to compute 2D ground-plane distance.

        Args:
            ego_position_flu: Object position in ego FLU coordinates [forward, left, up].
            track_id: Track ID for logging.

        Returns:
            True if object is within distance threshold.
        """
        distance_threshold = self.config.get("distance_threshold_m", float("inf"))
        if distance_threshold == float("inf"):
            return True

        # 2D ground-plane distance in FLU: forward (x) and left (y)
        forward = ego_position_flu[0]
        lateral = ego_position_flu[1]
        distance = float(np.sqrt(forward**2 + lateral**2))

        is_within = distance <= distance_threshold

        if self.logger:
            self.logger.debug(
                "Distance check: Track %s, position_flu=(%.1f, %.1f, %.1f), distance=%.1fm, threshold=%sm, within=%s",
                track_id,
                ego_position_flu[0],
                ego_position_flu[1],
                ego_position_flu[2],
                distance,
                distance_threshold,
                is_within,
            )

        return is_within
