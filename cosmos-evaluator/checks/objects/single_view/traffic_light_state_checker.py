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
Traffic light state checker for the static objects processor.

Validates that CWIP-predicted traffic light colors within each projected cuboid match the expected state from RDS
ground-truth data.  Scoring uses the full sigmoid probability distribution over the four TL color classes rather than
hard argmax pixel counts, correctly handling ambiguous predictions (e.g. traffic lights with multiple bulb colors
turned on).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, ClassVar

import numpy as np
import torch

from checks.objects.single_view.cwip_helper import CWIPInferenceHelper, resolve_model_class_indices
from checks.utils.models.cwip.modules.class_mapping import get_object_type_class_mapping
from checks.utils.models.cwip.visualization.world_model_utils import ObjectType, PatchSimilarityType
from checks.utils.rds_data_loader import RdsDataLoader
from checks.utils.scene_rasterizer import SceneRasterizer

RDS_STATE_TO_OBJECT_TYPE: dict[str, ObjectType] = {
    "red": ObjectType.TRAFFIC_LIGHT_RED,
    "yellow": ObjectType.TRAFFIC_LIGHT_YELLOW,
    "green": ObjectType.TRAFFIC_LIGHT_GREEN,
    "unknown": ObjectType.TRAFFIC_LIGHT_UNKNOWN,
}

TRAFFIC_LIGHT_OBJECT_TYPES: list[ObjectType] = [
    ObjectType.TRAFFIC_LIGHT_RED,
    ObjectType.TRAFFIC_LIGHT_YELLOW,
    ObjectType.TRAFFIC_LIGHT_GREEN,
    ObjectType.TRAFFIC_LIGHT_UNKNOWN,
]


@dataclass
class TrafficLightStateResult:
    """Per-object result from the traffic light state check."""

    traffic_light_state_score: float
    traffic_light_multi_color_score: float
    confidence: float
    # Expected GT traffic-light color for this object/frame ("red"/"yellow"/"green"/"unknown"/None). Surfaced so a
    # result consumer can report which color was wrong (e.g. GT green + low state score -> the green was missed).
    expected_state: str | None = None
    # Per-color proportion of the CWIP TL probability mass over the visible cuboid pixels, one entry per class in
    # {"red", "yellow", "green", "unknown"} (each in [0, 1], summing to ~1; "unknown" is the gray/off/unlit class).
    # ``traffic_light_state_score`` is exactly ``state_probs[expected_state]``; the full breakdown is surfaced so a
    # low-score light can be diagnosed (e.g. GT green but read mostly red -> rendered red; mostly unknown -> gray/off).
    # None when scores are NaN (below the minimum TL probability mass).
    state_probs: dict[str, float] | None = None


class TrafficLightStateChecker:
    """Checks traffic light color state agreement between CWIP predictions and RDS ground truth.

    Scoring uses the full sigmoid probability distribution over the four TL
    color classes.  For each visible pixel *i* in the projected cuboid the total
    TL probability mass is ``w_i = Σ_c p(c, i)`` over
    ``c ∈ {red, yellow, green, unknown}``.  The state score is then::

        score = Σ_i p(expected, i) / Σ_i w_i

    This correctly down-weights ambiguous pixels whose probability mass is spread
    across multiple colors (e.g. a traffic light with both red and yellow bulbs).

    A per-object **confidence** score quantifies robustness to a single CWIP
    patch being reclassified.  Because CWIP operates at 8× reduced resolution,
    all native-resolution pixels within an 8×8 patch share a single model
    prediction.  Visible pixels are grouped by patch and the total TL mass per
    patch ``W_k = Σ_{i ∈ patch_k} w_i`` is computed.  The confidence is::

        confidence = 1 − max_k(W_k) / Σ_i w_i

    Confidence approaches 1 when TL probability mass is spread across many
    patches and is 0 when a single patch contains all the mass.

    A **multi-color score** detects whether the cuboid contains multiple lit
    bulb colors.  For each color ``c ∈ {red, yellow, green}`` (excluding
    *unknown*) the proportion of total TL probability mass is computed.  The
    score is the ratio of the second-largest to the largest proportion::

        multi_color_score = proportion_2nd / proportion_1st

    The score is 1 when at least two colors occupy equal proportions and
    approaches 0 when one color dominates.

    Requires both a CWIPInferenceHelper (for model class mapping) and a
    SceneRasterizer (for occlusion-aware visibility masks).
    """

    _MIN_TL_PROBABILITY_MASS = 0.5
    _MULTI_COLOR_STATES: ClassVar[list[str]] = ["red", "yellow", "green"]

    def __init__(
        self,
        cwip_helper: CWIPInferenceHelper,
        scene_rasterizer: SceneRasterizer,
        logger: logging.Logger | None = None,
        min_in_frame_fraction: float = 0.0,
    ) -> None:
        """Initialize the traffic light state checker.

        Args:
            cwip_helper: CWIP inference helper, used to resolve model class
                indices for the traffic light object types.
            scene_rasterizer: Occlusion-aware rasterizer that provides
                per-object visibility masks for projected cuboids.
            logger: Optional logger instance.  Defaults to module-level logger.
            min_in_frame_fraction: Abstain on frames where the frame edge truncates
                the projected cuboid below this fraction (see :meth:`check_frame`).
                ``0`` (the default) disables the gate.  Sourced from
                ``overlap_check.traffic_light.state_min_in_frame_fraction``, which is
                separate from the ``min_in_frame_fraction`` gating correspondence in
                ``OverlapDetector``: color reading is far more sensitive to a clipped
                cuboid than existence is, and correspondence doubles as the existence
                verdict downstream, so the two are tuned independently.
        """
        self._cwip_helper = cwip_helper
        self._scene_rasterizer = scene_rasterizer
        self._logger = logger or logging.getLogger(__name__)
        self._patch_size = cwip_helper.output_downsample_factor
        self._min_in_frame_fraction = min_in_frame_fraction

        mapping_cls = get_object_type_class_mapping(getattr(cwip_helper.config, "object_type_class_mapping", None))

        self._tl_model_indices = resolve_model_class_indices(TRAFFIC_LIGHT_OBJECT_TYPES, mapping_cls)

        self._state_to_model_indices: dict[str, list[int]] = {}
        for state_str, obj_type in RDS_STATE_TO_OBJECT_TYPE.items():
            self._state_to_model_indices[state_str] = resolve_model_class_indices([obj_type], mapping_cls)

    def check_frame(
        self,
        frame_type_probs: torch.Tensor,
        frame_objects: dict[str, Any],
        frame_idx: int,
        frame_similarity: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, TrafficLightStateResult]:
        """Run the traffic light state check for a single frame.

        Args:
            frame_type_probs: Per-class sigmoid probabilities, shape (C, H, W).
            frame_objects: All objects for the current frame (from RdsDataLoader).
            frame_idx: RDS timeline frame index, used to look up expected state.
            frame_similarity: CWIP similarity prediction indices, shape (H, W).
                When provided, only pixels classified as MATCH are considered
                alongside the visibility mask.

        Returns:
            Mapping from track_id (str) to TrafficLightStateResult.
        """
        traffic_lights = RdsDataLoader.get_objects_by_class(frame_objects, "traffic_light")
        if not traffic_lights:
            return {}

        probs_np = frame_type_probs.detach().numpy() if isinstance(frame_type_probs, torch.Tensor) else frame_type_probs
        tl_probs = probs_np[self._tl_model_indices]  # (num_tl_classes, H, W)
        w_all = tl_probs.sum(axis=0)  # (H, W)

        if frame_similarity is not None:
            sim_np = frame_similarity.numpy() if isinstance(frame_similarity, torch.Tensor) else frame_similarity
            match_mask = sim_np == PatchSimilarityType.MATCH.value
        else:
            match_mask = None

        img_h, img_w = w_all.shape
        patches_per_row = -(-img_w // self._patch_size)  # ceil division
        patch_indices = (np.arange(img_h)[:, None] // self._patch_size) * patches_per_row + (
            np.arange(img_w)[None, :] // self._patch_size
        )

        results: dict[str, TrafficLightStateResult] = {}
        for track_id, obj in traffic_lights.items():
            track_id_str = str(track_id)

            if not self._scene_rasterizer.has_object(track_id_str):
                continue

            # Frame-edge truncation gate.  A cuboid clipped by the image edge shows only part of the
            # light's housing, so the color mass ratios below are computed over an arbitrary crop —
            # a red bulb cut off at the frame edge can read as green simply because the red pixels
            # fell outside the image.  Abstain on the frame entirely (record nothing, so the frame is
            # excluded from the per-track nanmean) rather than emit a score from a partial crop.
            # Purely geometric and independent of occlusion, which the visibility mask already
            # handles.  Shares the mechanism with the correspondence gate in
            # ``OverlapDetector.process_class`` but is tuned separately (see ``__init__``).
            if (
                self._min_in_frame_fraction > 0.0
                and self._scene_rasterizer.in_frame_fraction(track_id_str) < self._min_in_frame_fraction
            ):
                self._logger.debug(
                    "Track %s frame %d: projected cuboid truncated by the frame edge (in-frame fraction "
                    "below %.2f), skipping state check",
                    track_id_str,
                    frame_idx,
                    self._min_in_frame_fraction,
                )
                continue

            vis_mask = self._scene_rasterizer.get_visibility_mask(track_id_str)
            if match_mask is not None:
                vis_mask = vis_mask & match_mask
            if not np.any(vis_mask):
                continue

            w_vis = w_all[vis_mask]
            total_mass = float(w_vis.sum())
            expected_state = self._get_expected_state(obj, frame_idx)

            if total_mass < self._MIN_TL_PROBABILITY_MASS:
                self._logger.debug(
                    "Track %s frame %d: total TL probability mass %.3f is below "
                    "threshold %.1f (%d visible pixels), all scores NaN",
                    track_id_str,
                    frame_idx,
                    total_mass,
                    self._MIN_TL_PROBABILITY_MASS,
                    int(vis_mask.sum()),
                )
                results[track_id_str] = TrafficLightStateResult(
                    traffic_light_state_score=float("nan"),
                    traffic_light_multi_color_score=float("nan"),
                    confidence=float("nan"),
                    expected_state=expected_state,
                )
                continue

            vis_patch_ids = patch_indices[vis_mask]
            patch_masses = np.bincount(vis_patch_ids.ravel(), weights=w_vis.ravel())
            confidence = 1.0 - float(patch_masses.max()) / total_mass

            # State score (depends on expected state)
            state_score = self._compute_state_score(
                probs_np,
                vis_mask,
                total_mass,
                obj,
                frame_idx,
                track_id_str,
            )

            # Multi-color score (independent of expected state)
            multi_color_score = self._compute_multi_color_score(probs_np, vis_mask)

            # Full per-color mass breakdown (diagnostic; state_score == state_probs[expected_state]).
            state_probs = self._compute_state_probs(probs_np, vis_mask, total_mass)

            results[track_id_str] = TrafficLightStateResult(
                traffic_light_state_score=state_score,
                traffic_light_multi_color_score=multi_color_score,
                confidence=confidence,
                expected_state=expected_state,
                state_probs=state_probs,
            )

        return results

    def _compute_state_probs(
        self,
        probs_np: np.ndarray,
        vis_mask: np.ndarray,
        total_mass: float,
    ) -> dict[str, float]:
        """Per-color proportion of TL probability mass over visible pixels, for every TL color class.

        Returns ``{"red", "yellow", "green", "unknown"}`` -> proportion in [0, 1] (summing to ~1). Each is the same
        quantity ``_compute_state_score`` returns for the expected color, computed for every class so a low-score
        light can be attributed (which color CWIP actually read). ``unknown`` is the gray / off / unlit class.
        """
        out: dict[str, float] = {}
        for state_str in ("red", "yellow", "green", "unknown"):
            indices = self._state_to_model_indices.get(state_str, [])
            mass = float(probs_np[indices].sum(axis=0)[vis_mask].sum()) if indices else 0.0
            out[state_str] = mass / total_mass if total_mass > 0 else float("nan")
        return out

    def _compute_state_score(
        self,
        probs_np: np.ndarray,
        vis_mask: np.ndarray,
        total_mass: float,
        obj: dict[str, Any],
        frame_idx: int,
        track_id_str: str,
    ) -> float:
        """Compute the state score for a single traffic light.

        Args:
            probs_np: Per-class sigmoid probabilities, shape (C, H, W).
            vis_mask: Boolean visibility mask, shape (H, W).
            total_mass: Pre-computed total TL probability mass over visible
                pixels, used as the denominator for normalization.
            obj: Traffic light object dict from RDS ground-truth data.
            frame_idx: RDS timeline frame index, used to look up expected state.
            track_id_str: Object track ID string, used for debug logging.

        Returns:
            Score in [0, 1], or NaN when the expected state is unavailable, is ``"unknown"``, or has no model class
            indices.
        """
        expected_state = self._get_expected_state(obj, frame_idx)

        if expected_state is None:
            self._logger.debug(
                "Track %s frame %d: no expected state available, state score NaN",
                track_id_str,
                frame_idx,
            )
            return float("nan")

        if expected_state == "unknown":
            self._logger.debug(
                "Track %s frame %d: expected state is 'unknown', state score NaN",
                track_id_str,
                frame_idx,
            )
            return float("nan")

        expected_indices = self._state_to_model_indices.get(expected_state, [])
        if not expected_indices:
            self._logger.debug(
                "Expected state '%s' has no model class indices, state score NaN",
                expected_state,
            )
            return float("nan")

        p_expected_vis = probs_np[expected_indices].sum(axis=0)[vis_mask]
        return float(p_expected_vis.sum()) / total_mass

    def _compute_multi_color_score(
        self,
        probs_np: np.ndarray,
        vis_mask: np.ndarray,
    ) -> float:
        """Compute the multi-color score for a single traffic light.

        Measures whether at least two bulb colors (red, yellow, green) occupy
        similar proportions of the projected cuboid.  Returns the ratio of the
        second-largest per-color probability mass to the largest.

        Args:
            probs_np: Per-class sigmoid probabilities, shape (C, H, W).
            vis_mask: Boolean visibility mask, shape (H, W).

        Returns:
            Score in [0, 1].  1 means two colors have equal mass; 0 means one color completely dominates.  NaN if the
            dominant color has zero mass.
        """
        color_masses: list[float] = []
        for state_str in self._MULTI_COLOR_STATES:
            indices = self._state_to_model_indices.get(state_str, [])
            if indices:
                mass = float(probs_np[indices].sum(axis=0)[vis_mask].sum())
            else:
                mass = 0.0
            color_masses.append(mass)

        color_masses.sort(reverse=True)
        if color_masses[0] <= 0:
            return float("nan")
        return color_masses[1] / color_masses[0]

    @staticmethod
    def _get_expected_state(obj: dict[str, Any], frame_idx: int) -> str | None:
        """Look up the expected traffic light state for a given frame.

        Args:
            obj: Traffic light object dict, may contain "state_per_frame".
            frame_idx: RDS frame index.

        Returns:
            Lowercase state string ("red", "yellow", "green", "unknown") or None.
        """
        states = obj.get("state_per_frame")
        if states is None or not isinstance(states, list):
            return None
        if frame_idx < 0 or frame_idx >= len(states):
            return None
        state = states[frame_idx]
        if isinstance(state, str):
            return state.lower()
        return None
