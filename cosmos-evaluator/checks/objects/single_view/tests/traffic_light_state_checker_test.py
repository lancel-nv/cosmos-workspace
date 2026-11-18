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

"""Unit tests for the TrafficLightStateChecker module."""

import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from checks.objects.single_view.traffic_light_state_checker import (
    TrafficLightStateChecker,
    TrafficLightStateResult,
)
from checks.utils.models.cwip.modules.class_mapping import ObjectTypeClassMappingV3
from checks.utils.models.cwip.visualization.world_model_utils import ObjectType, PatchSimilarityType

# V3 model class indices for traffic light colors
RED_IDX = ObjectTypeClassMappingV3.get_model_class(ObjectType.TRAFFIC_LIGHT_RED)
YELLOW_IDX = ObjectTypeClassMappingV3.get_model_class(ObjectType.TRAFFIC_LIGHT_YELLOW)
GREEN_IDX = ObjectTypeClassMappingV3.get_model_class(ObjectType.TRAFFIC_LIGHT_GREEN)
UNKNOWN_IDX = ObjectTypeClassMappingV3.get_model_class(ObjectType.TRAFFIC_LIGHT_UNKNOWN)
BACKGROUND_IDX = ObjectTypeClassMappingV3.get_model_class(ObjectType.BACKGROUND)
NUM_CLASSES_V3 = ObjectTypeClassMappingV3.NUM_CLASSES


def _make_cwip_helper(mapping_name: str = "v3", output_downsample_factor: int = 8) -> MagicMock:
    """Create a mock CWIPInferenceHelper with a given class mapping."""
    helper = MagicMock()
    helper.config = MagicMock()
    helper.config.object_type_class_mapping = mapping_name
    helper.output_downsample_factor = output_downsample_factor
    return helper


def _make_scene_rasterizer(
    visibility_masks: dict[str, np.ndarray],
    in_frame_fractions: dict[str, float] | None = None,
) -> MagicMock:
    """Create a mock SceneRasterizer with pre-set visibility masks and in-frame fractions.

    Objects absent from ``in_frame_fractions`` report 1.0 (fully in-frame, never gated).
    """
    fractions = in_frame_fractions or {}

    rasterizer = MagicMock()

    def has_object(obj_id: str) -> bool:
        return obj_id in visibility_masks

    def get_visibility_mask(obj_id: str) -> np.ndarray:
        if obj_id not in visibility_masks:
            raise KeyError(f"Object ID '{obj_id}' not found")
        return visibility_masks[obj_id]

    def in_frame_fraction(obj_id: str) -> float:
        if obj_id not in visibility_masks:
            raise KeyError(f"Object ID '{obj_id}' not found")
        return fractions.get(obj_id, 1.0)

    rasterizer.has_object = MagicMock(side_effect=has_object)
    rasterizer.get_visibility_mask = MagicMock(side_effect=get_visibility_mask)
    rasterizer.in_frame_fraction = MagicMock(side_effect=in_frame_fraction)
    return rasterizer


def _make_traffic_light_obj(
    state_per_frame: list[str] | None = None,
    obj_type_idx: int = 0,
) -> dict:
    """Create a traffic light object dict matching the RdsDataLoader format."""
    obj = {
        "object_type": "TrafficLight",
        "geometry": MagicMock(),
        "is_static": True,
        "object_type_index": obj_type_idx,
    }
    if state_per_frame is not None:
        obj["state_per_frame"] = state_per_frame
    return obj


def _make_uniform_probs(H: int, W: int, class_idx: int, prob: float = 0.95) -> torch.Tensor:
    """Create a (NUM_CLASSES_V3, H, W) probability tensor with one dominant class."""
    probs = torch.full((NUM_CLASSES_V3, H, W), 0.01)
    probs[class_idx] = prob
    return probs


class TestTrafficLightStateCheckerInit(unittest.TestCase):
    """Test initialization of TrafficLightStateChecker."""

    def test_init_resolves_model_indices(self) -> None:
        helper = _make_cwip_helper("v3")
        rasterizer = _make_scene_rasterizer({})
        checker = TrafficLightStateChecker(helper, rasterizer)

        self.assertEqual(sorted(checker._tl_model_indices), [RED_IDX, YELLOW_IDX, GREEN_IDX, UNKNOWN_IDX])

    def test_init_state_to_model_indices(self) -> None:
        helper = _make_cwip_helper("v3")
        rasterizer = _make_scene_rasterizer({})
        checker = TrafficLightStateChecker(helper, rasterizer)

        self.assertEqual(checker._state_to_model_indices["red"], [RED_IDX])
        self.assertEqual(checker._state_to_model_indices["yellow"], [YELLOW_IDX])
        self.assertEqual(checker._state_to_model_indices["green"], [GREEN_IDX])
        self.assertEqual(checker._state_to_model_indices["unknown"], [UNKNOWN_IDX])

    def test_init_patch_size_from_cwip_helper(self) -> None:
        helper = _make_cwip_helper("v3", output_downsample_factor=8)
        rasterizer = _make_scene_rasterizer({})
        checker = TrafficLightStateChecker(helper, rasterizer)
        self.assertEqual(checker._patch_size, 8)

    def test_init_patch_size_custom_value(self) -> None:
        helper = _make_cwip_helper("v3", output_downsample_factor=16)
        rasterizer = _make_scene_rasterizer({})
        checker = TrafficLightStateChecker(helper, rasterizer)
        self.assertEqual(checker._patch_size, 16)


class TestTrafficLightStateCheckerCheckFrame(unittest.TestCase):
    """Test check_frame method of TrafficLightStateChecker."""

    def _make_checker(self, visibility_masks: dict[str, np.ndarray]) -> TrafficLightStateChecker:
        helper = _make_cwip_helper("v3")
        rasterizer = _make_scene_rasterizer(visibility_masks)
        return TrafficLightStateChecker(helper, rasterizer)

    def test_no_traffic_lights_returns_empty(self) -> None:
        """No traffic light objects → empty result."""
        checker = self._make_checker({})
        frame_objects = {"1": {"object_type": "Vehicle", "geometry": MagicMock()}}
        probs = _make_uniform_probs(4, 4, RED_IDX)
        result = checker.check_frame(probs, frame_objects, 0)
        self.assertEqual(result, {})

    def test_perfect_score_all_red(self) -> None:
        """High red probability everywhere, expected state is red."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red", "green"])}
        probs = _make_uniform_probs(H, W, RED_IDX, prob=0.95)

        result = checker.check_frame(probs, frame_objects, 0)
        self.assertIn("10", result)
        per_pixel_mass = 0.95 + 0.01 + 0.01 + 0.01
        self.assertAlmostEqual(result["10"].traffic_light_state_score, 0.95 / per_pixel_mass, places=5)
        # All pixels within one 8x8 patch → confidence = 0.0
        self.assertAlmostEqual(result["10"].confidence, 0.0)
        # Single dominant color → multi-color score near 0
        self.assertAlmostEqual(result["10"].traffic_light_multi_color_score, 0.01 / 0.95, places=5)

    def test_low_score_wrong_state(self) -> None:
        """High red probability everywhere, expected state is green → low score."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["green", "red"])}
        probs = _make_uniform_probs(H, W, RED_IDX, prob=0.95)

        result = checker.check_frame(probs, frame_objects, 0)
        per_pixel_mass = 0.95 + 0.01 + 0.01 + 0.01
        self.assertAlmostEqual(result["10"].traffic_light_state_score, 0.01 / per_pixel_mass, places=5)
        self.assertAlmostEqual(result["10"].confidence, 0.0)
        self.assertAlmostEqual(result["10"].traffic_light_multi_color_score, 0.01 / 0.95, places=5)

    def test_state_probs_breakdown(self) -> None:
        """The full per-color mass breakdown is surfaced: it has all four TL classes, ``state_probs[expected]`` equals
        the state score, and the four proportions cover the whole TL mass (sum ~1)."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["green", "red"])}  # expected green
        probs = _make_uniform_probs(H, W, RED_IDX, prob=0.95)  # but CWIP reads red

        sp = checker.check_frame(probs, frame_objects, 0)["10"].state_probs
        assert sp is not None  # narrow for mypy; a NaN-mass light would give None
        self.assertEqual(set(sp), {"red", "yellow", "green", "unknown"})
        per_pixel_mass = 0.95 + 0.01 + 0.01 + 0.01
        # CWIP read red (0.95 mass) but expected green → the red component is dominant, green is tiny.
        self.assertAlmostEqual(sp["red"], 0.95 / per_pixel_mass, places=5)
        self.assertAlmostEqual(sp["green"], 0.01 / per_pixel_mass, places=5)
        # state score == the expected color's proportion.
        result_green = checker.check_frame(probs, frame_objects, 0)["10"]
        self.assertAlmostEqual(sp["green"], result_green.traffic_light_state_score, places=5)
        # The four class proportions cover the whole TL mass.
        self.assertAlmostEqual(sum(sp.values()), 1.0, places=5)

    def test_partial_score_mixed_pixels(self) -> None:
        """Half pixels strongly red, half strongly green → partial score for red."""
        H, W = 2, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}

        probs = torch.full((NUM_CLASSES_V3, H, W), 0.01)
        probs[RED_IDX, 0, :] = 0.95  # top row: strong red
        probs[GREEN_IDX, 1, :] = 0.95  # bottom row: strong green

        result = checker.check_frame(probs, frame_objects, 0)

        w_red_pixel = 0.95 + 0.01 + 0.01 + 0.01  # 0.98
        w_green_pixel = 0.01 + 0.01 + 0.95 + 0.01  # 0.98
        total_mass = 4 * w_red_pixel + 4 * w_green_pixel
        numerator = 4 * 0.95 + 4 * 0.01
        self.assertAlmostEqual(result["10"].traffic_light_state_score, numerator / total_mass, places=5)

    def test_low_tl_mass_background_gives_nan(self) -> None:
        """Pixels with negligible TL probability → NaN score."""
        H, W = 2, 2
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}

        probs = torch.full((NUM_CLASSES_V3, H, W), 0.01)
        probs[BACKGROUND_IDX] = 0.95

        result = checker.check_frame(probs, frame_objects, 0)
        self.assertTrue(np.isnan(result["10"].traffic_light_state_score))
        self.assertTrue(np.isnan(result["10"].confidence))
        self.assertTrue(np.isnan(result["10"].traffic_light_multi_color_score))

    def test_missing_state_per_frame_gives_nan_state_but_valid_multi_color(self) -> None:
        """If the object has no state_per_frame, state is NaN but multi-color is still computed."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=None)}
        probs = _make_uniform_probs(H, W, RED_IDX)

        result = checker.check_frame(probs, frame_objects, 0)
        self.assertTrue(np.isnan(result["10"].traffic_light_state_score))
        self.assertAlmostEqual(result["10"].traffic_light_multi_color_score, 0.01 / 0.95, places=5)
        self.assertFalse(np.isnan(result["10"].confidence))

    def test_unknown_expected_state_gives_nan_state_but_valid_multi_color(self) -> None:
        """If the expected state from RDS is 'unknown', state is NaN but multi-color is computed."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["unknown"])}
        probs = _make_uniform_probs(H, W, UNKNOWN_IDX)

        result = checker.check_frame(probs, frame_objects, 0)
        self.assertTrue(np.isnan(result["10"].traffic_light_state_score))
        # Unknown class is dominant but excluded from multi-color; all 3 colors have 0.01
        self.assertAlmostEqual(result["10"].traffic_light_multi_color_score, 1.0, places=5)
        self.assertFalse(np.isnan(result["10"].confidence))

    def test_frame_idx_out_of_range_gives_nan_state_but_valid_multi_color(self) -> None:
        """Frame index beyond state_per_frame length → state is NaN but multi-color computed."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX)

        result = checker.check_frame(probs, frame_objects, 5)
        self.assertTrue(np.isnan(result["10"].traffic_light_state_score))
        self.assertAlmostEqual(result["10"].traffic_light_multi_color_score, 0.01 / 0.95, places=5)
        self.assertFalse(np.isnan(result["10"].confidence))

    def test_visibility_mask_limits_region(self) -> None:
        """Only pixels within the visibility mask contribute."""
        H, W = 4, 4
        vis_mask = np.zeros((H, W), dtype=bool)
        vis_mask[0, 0] = True
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}

        probs = torch.full((NUM_CLASSES_V3, H, W), 0.01)
        probs[RED_IDX, 0, 0] = 0.95  # visible pixel: high red
        probs[GREEN_IDX, :, :] = 0.95  # everything else: high green
        probs[GREEN_IDX, 0, 0] = 0.01  # but visible pixel is low green

        result = checker.check_frame(probs, frame_objects, 0)
        w = 0.95 + 0.01 + 0.01 + 0.01
        self.assertAlmostEqual(result["10"].traffic_light_state_score, 0.95 / w, places=5)
        # 1 pixel → confidence = 0.0
        self.assertAlmostEqual(result["10"].confidence, 0.0)

    def test_object_not_in_rasterizer_skipped(self) -> None:
        """If the object isn't in the rasterizer, it's not scored."""
        H, W = 4, 4
        checker = self._make_checker({})  # empty rasterizer

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX)

        result = checker.check_frame(probs, frame_objects, 0)
        self.assertEqual(result, {})

    def test_empty_visibility_mask_skipped(self) -> None:
        """If visibility mask is all-False, object is skipped."""
        H, W = 4, 4
        vis_mask = np.zeros((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX)

        result = checker.check_frame(probs, frame_objects, 0)
        self.assertEqual(result, {})


class TestInFrameFractionGate(unittest.TestCase):
    """Frame-edge truncation gate: a traffic light the image edge clips exposes only part of its bulbs, so
    the color-mass ratios would be computed over an arbitrary crop. Such frames abstain entirely (no result
    recorded) rather than contribute a misleading state score."""

    def _make_checker(
        self,
        visibility_masks: dict[str, np.ndarray],
        in_frame_fractions: dict[str, float] | None = None,
        min_in_frame_fraction: float = 1.0,
    ) -> TrafficLightStateChecker:
        helper = _make_cwip_helper("v3")
        rasterizer = _make_scene_rasterizer(visibility_masks, in_frame_fractions)
        return TrafficLightStateChecker(helper, rasterizer, min_in_frame_fraction=min_in_frame_fraction)

    @staticmethod
    def _full_mask(h: int, w: int) -> np.ndarray:
        return np.ones((h, w), dtype=bool)

    def test_truncated_light_is_skipped(self) -> None:
        """Below the threshold: no entry at all, so the frame is excluded from the per-track nanmean."""
        H, W = 4, 4
        checker = self._make_checker({"10": self._full_mask(H, W)}, {"10": 0.6})
        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX)

        self.assertEqual(checker.check_frame(probs, frame_objects, 0), {})

    def test_fully_in_frame_light_is_scored(self) -> None:
        H, W = 4, 4
        checker = self._make_checker({"10": self._full_mask(H, W)}, {"10": 1.0})
        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX)

        result = checker.check_frame(probs, frame_objects, 0)
        self.assertIn("10", result)
        self.assertGreater(result["10"].traffic_light_state_score, 0.9)

    def test_exactly_at_threshold_is_scored(self) -> None:
        """Strict ``<``, matching the correspondence gate in OverlapDetector."""
        H, W = 4, 4
        checker = self._make_checker({"10": self._full_mask(H, W)}, {"10": 0.5}, min_in_frame_fraction=0.5)
        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX)

        self.assertIn("10", checker.check_frame(probs, frame_objects, 0))

    def test_gate_disabled_by_default(self) -> None:
        """Default 0.0 disables the gate: a badly truncated light is still scored, so the behavior is
        opt-in via config."""
        H, W = 4, 4
        checker = self._make_checker({"10": self._full_mask(H, W)}, {"10": 0.01}, min_in_frame_fraction=0.0)
        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX)

        self.assertIn("10", checker.check_frame(probs, frame_objects, 0))

    def test_gate_is_per_object(self) -> None:
        """One truncated light does not suppress a second, fully in-frame one in the same frame."""
        H, W = 4, 8
        vis_a = np.zeros((H, W), dtype=bool)
        vis_a[:, :4] = True
        vis_b = np.zeros((H, W), dtype=bool)
        vis_b[:, 4:] = True
        checker = self._make_checker({"10": vis_a, "20": vis_b}, {"10": 0.3, "20": 1.0})
        frame_objects = {
            "10": _make_traffic_light_obj(state_per_frame=["red"]),
            "20": _make_traffic_light_obj(state_per_frame=["red"]),
        }
        probs = _make_uniform_probs(H, W, RED_IDX)

        result = checker.check_frame(probs, frame_objects, 0)
        self.assertNotIn("10", result)
        self.assertIn("20", result)


class TestMultipleObjects(unittest.TestCase):
    """Test check_frame with multiple traffic light objects."""

    def test_two_traffic_lights_independent_scores(self) -> None:
        H, W = 4, 8
        vis_a = np.zeros((H, W), dtype=bool)
        vis_a[:, :4] = True  # left half
        vis_b = np.zeros((H, W), dtype=bool)
        vis_b[:, 4:] = True  # right half

        helper = _make_cwip_helper("v3")
        rasterizer = _make_scene_rasterizer({"20": vis_a, "21": vis_b})
        checker = TrafficLightStateChecker(helper, rasterizer)

        frame_objects = {
            "20": _make_traffic_light_obj(state_per_frame=["red"]),
            "21": _make_traffic_light_obj(state_per_frame=["green"], obj_type_idx=1),
        }

        probs = torch.full((NUM_CLASSES_V3, H, W), 0.01)
        probs[RED_IDX, :, :4] = 0.95  # left half: strong red
        probs[GREEN_IDX, :, 4:] = 0.95  # right half: strong green

        result = checker.check_frame(probs, frame_objects, 0)

        per_pixel_mass = 0.95 + 0.01 + 0.01 + 0.01
        self.assertAlmostEqual(result["20"].traffic_light_state_score, 0.95 / per_pixel_mass, places=5)
        self.assertAlmostEqual(result["21"].traffic_light_state_score, 0.95 / per_pixel_mass, places=5)
        # Both objects fit in a single 8x8 patch → confidence = 0.0
        self.assertAlmostEqual(result["20"].confidence, 0.0)
        self.assertAlmostEqual(result["21"].confidence, 0.0)
        self.assertAlmostEqual(result["20"].traffic_light_multi_color_score, 0.01 / 0.95, places=5)
        self.assertAlmostEqual(result["21"].traffic_light_multi_color_score, 0.01 / 0.95, places=5)


class TestGetExpectedState(unittest.TestCase):
    """Test the static _get_expected_state helper."""

    def test_returns_state_at_index(self) -> None:
        obj = {"state_per_frame": ["red", "green", "yellow"]}
        self.assertEqual(TrafficLightStateChecker._get_expected_state(obj, 0), "red")
        self.assertEqual(TrafficLightStateChecker._get_expected_state(obj, 1), "green")
        self.assertEqual(TrafficLightStateChecker._get_expected_state(obj, 2), "yellow")

    def test_returns_none_for_missing_key(self) -> None:
        obj: dict[str, list] = {}
        self.assertIsNone(TrafficLightStateChecker._get_expected_state(obj, 0))

    def test_returns_none_for_out_of_range(self) -> None:
        obj = {"state_per_frame": ["red"]}
        self.assertIsNone(TrafficLightStateChecker._get_expected_state(obj, 5))

    def test_returns_none_for_negative_index(self) -> None:
        obj = {"state_per_frame": ["red", "green"]}
        self.assertIsNone(TrafficLightStateChecker._get_expected_state(obj, -1))

    def test_lowercases_state(self) -> None:
        obj = {"state_per_frame": ["RED", "Green"]}
        self.assertEqual(TrafficLightStateChecker._get_expected_state(obj, 0), "red")
        self.assertEqual(TrafficLightStateChecker._get_expected_state(obj, 1), "green")

    def test_returns_none_for_non_list(self) -> None:
        obj = {"state_per_frame": "not_a_list"}
        self.assertIsNone(TrafficLightStateChecker._get_expected_state(obj, 0))


class TestTrafficLightStateResult(unittest.TestCase):
    """Test the TrafficLightStateResult dataclass."""

    def test_dataclass_fields(self) -> None:
        result = TrafficLightStateResult(
            traffic_light_state_score=0.75,
            traffic_light_multi_color_score=0.3,
            confidence=0.9,
        )
        self.assertAlmostEqual(result.traffic_light_state_score, 0.75)
        self.assertAlmostEqual(result.traffic_light_multi_color_score, 0.3)
        self.assertAlmostEqual(result.confidence, 0.9)


class TestSoftScoring(unittest.TestCase):
    """Test soft probability-based scoring in check_frame."""

    def _make_checker(self, visibility_masks: dict[str, np.ndarray]) -> TrafficLightStateChecker:
        helper = _make_cwip_helper("v3")
        rasterizer = _make_scene_rasterizer(visibility_masks)
        return TrafficLightStateChecker(helper, rasterizer)

    def test_soft_perfect_score_single_patch(self) -> None:
        """High red probability everywhere within one patch → score ≈ 1, confidence = 0."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX, prob=0.95)

        result = checker.check_frame(probs, frame_objects, 0)

        per_pixel_mass = 0.95 + 0.01 + 0.01 + 0.01
        expected_score = 0.95 / per_pixel_mass
        self.assertAlmostEqual(result["10"].traffic_light_state_score, expected_score, places=5)

        # All pixels in one patch → confidence = 0.0
        self.assertAlmostEqual(result["10"].confidence, 0.0)
        self.assertAlmostEqual(result["10"].traffic_light_multi_color_score, 0.01 / 0.95, places=5)

    def test_soft_zero_score_wrong_state(self) -> None:
        """All probability on red, expected green → score ≈ 0."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["green"])}
        probs = _make_uniform_probs(H, W, RED_IDX, prob=0.95)

        result = checker.check_frame(probs, frame_objects, 0)

        expected_score = 0.01 / (0.95 + 0.01 + 0.01 + 0.01)
        self.assertAlmostEqual(result["10"].traffic_light_state_score, expected_score, places=5)
        self.assertAlmostEqual(result["10"].traffic_light_multi_color_score, 0.01 / 0.95, places=5)

    def test_soft_ambiguous_split_reduces_score(self) -> None:
        """Equal red and green probability → score ≈ 0.5 for either expected state."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}

        probs = torch.full((NUM_CLASSES_V3, H, W), 0.01)
        probs[RED_IDX] = 0.45
        probs[GREEN_IDX] = 0.45

        result = checker.check_frame(probs, frame_objects, 0)

        total_tl_mass = 0.45 + 0.45 + 0.01 + 0.01
        expected_score = 0.45 / total_tl_mass
        self.assertAlmostEqual(result["10"].traffic_light_state_score, expected_score, places=5)
        # Equal red and green → multi-color score = 1.0
        self.assertAlmostEqual(result["10"].traffic_light_multi_color_score, 1.0, places=5)

    def test_soft_heterogeneous_pixels(self) -> None:
        """Different pixels have different TL probabilities → weighted average."""
        H, W = 1, 2
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}

        probs = torch.full((NUM_CLASSES_V3, H, W), 0.01)
        # Pixel (0,0): strong red
        probs[RED_IDX, 0, 0] = 0.9
        probs[GREEN_IDX, 0, 0] = 0.05
        probs[YELLOW_IDX, 0, 0] = 0.02
        probs[UNKNOWN_IDX, 0, 0] = 0.01
        # Pixel (0,1): ambiguous red/green
        probs[RED_IDX, 0, 1] = 0.4
        probs[GREEN_IDX, 0, 1] = 0.35
        probs[YELLOW_IDX, 0, 1] = 0.15
        probs[UNKNOWN_IDX, 0, 1] = 0.05

        result = checker.check_frame(probs, frame_objects, 0)

        w0 = 0.9 + 0.05 + 0.02 + 0.01  # 0.98
        w1 = 0.4 + 0.35 + 0.15 + 0.05  # 0.95
        expected_score = (0.9 + 0.4) / (w0 + w1)
        self.assertAlmostEqual(result["10"].traffic_light_state_score, expected_score, places=5)

        # Both pixels in one patch → confidence = 0
        self.assertAlmostEqual(result["10"].confidence, 0.0)

    def test_soft_single_patch_zero_confidence(self) -> None:
        """All visible pixels in one patch → confidence = 0."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX, prob=0.9)

        result = checker.check_frame(probs, frame_objects, 0)
        self.assertAlmostEqual(result["10"].confidence, 0.0)

    def test_soft_two_equal_patches_half_confidence(self) -> None:
        """Visible pixels equally split across two patches → confidence = 0.5."""
        H, W = 8, 16
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX, prob=0.95)

        result = checker.check_frame(probs, frame_objects, 0)
        # 2 patches with equal mass → confidence = 1 - 0.5 = 0.5
        self.assertAlmostEqual(result["10"].confidence, 0.5, places=5)

    def test_soft_four_equal_patches_confidence(self) -> None:
        """Visible pixels equally split across four patches → confidence = 0.75."""
        H, W = 16, 16
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX, prob=0.95)

        result = checker.check_frame(probs, frame_objects, 0)
        # 4 patches with equal mass → confidence = 1 - 0.25 = 0.75
        self.assertAlmostEqual(result["10"].confidence, 0.75, places=5)

    def test_soft_low_tl_mass_gives_nan(self) -> None:
        """Total TL probability mass below threshold → NaN score."""
        H, W = 2, 2
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}

        probs = torch.full((NUM_CLASSES_V3, H, W), 0.01)
        probs[BACKGROUND_IDX] = 0.95

        result = checker.check_frame(probs, frame_objects, 0)

        # Total TL mass = 4 * (0.01*4) = 0.16 < _MIN_TL_PROBABILITY_MASS
        self.assertTrue(np.isnan(result["10"].traffic_light_state_score))
        self.assertTrue(np.isnan(result["10"].confidence))
        self.assertTrue(np.isnan(result["10"].traffic_light_multi_color_score))

    def test_soft_unknown_state_gives_nan_state_but_valid_multi_color(self) -> None:
        """Expected state 'unknown' → state NaN but multi-color is still computed."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["unknown"])}
        probs = _make_uniform_probs(H, W, RED_IDX)

        result = checker.check_frame(probs, frame_objects, 0)
        self.assertTrue(np.isnan(result["10"].traffic_light_state_score))
        self.assertAlmostEqual(result["10"].traffic_light_multi_color_score, 0.01 / 0.95, places=5)
        self.assertFalse(np.isnan(result["10"].confidence))

    def test_soft_visibility_mask_limits_region(self) -> None:
        """Only visible pixels contribute to soft score."""
        H, W = 2, 2
        vis_mask = np.zeros((H, W), dtype=bool)
        vis_mask[0, 0] = True
        vis_mask[0, 1] = True
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}

        probs = torch.full((NUM_CLASSES_V3, H, W), 0.01)
        # Visible pixels: high red
        probs[RED_IDX, 0, 0] = 0.9
        probs[RED_IDX, 0, 1] = 0.9
        # Non-visible pixels: high green (should be ignored)
        probs[GREEN_IDX, 1, 0] = 0.9
        probs[GREEN_IDX, 1, 1] = 0.9

        result = checker.check_frame(probs, frame_objects, 0)

        w_per_vis_pixel = 0.9 + 0.01 + 0.01 + 0.01  # 0.93
        expected_score = (0.9 * 2) / (w_per_vis_pixel * 2)
        self.assertAlmostEqual(result["10"].traffic_light_state_score, expected_score, places=5)


class TestMultiColorScore(unittest.TestCase):
    """Test multi-color score computation."""

    def _make_checker(self, visibility_masks: dict[str, np.ndarray]) -> TrafficLightStateChecker:
        helper = _make_cwip_helper("v3")
        rasterizer = _make_scene_rasterizer(visibility_masks)
        return TrafficLightStateChecker(helper, rasterizer)

    def test_single_dominant_color_low_score(self) -> None:
        """One color dominates → multi-color score near 0."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX, prob=0.95)

        result = checker.check_frame(probs, frame_objects, 0)
        self.assertAlmostEqual(result["10"].traffic_light_multi_color_score, 0.01 / 0.95, places=5)

    def test_two_equal_colors_score_one(self) -> None:
        """Two colors equally probable → multi-color score = 1.0."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}

        probs = torch.full((NUM_CLASSES_V3, H, W), 0.01)
        probs[RED_IDX] = 0.45
        probs[GREEN_IDX] = 0.45

        result = checker.check_frame(probs, frame_objects, 0)
        self.assertAlmostEqual(result["10"].traffic_light_multi_color_score, 1.0, places=5)

    def test_three_equal_colors_score_one(self) -> None:
        """Three colors equally probable → multi-color score = 1.0."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}

        probs = torch.full((NUM_CLASSES_V3, H, W), 0.01)
        probs[RED_IDX] = 0.30
        probs[YELLOW_IDX] = 0.30
        probs[GREEN_IDX] = 0.30

        result = checker.check_frame(probs, frame_objects, 0)
        self.assertAlmostEqual(result["10"].traffic_light_multi_color_score, 1.0, places=5)

    def test_spatial_split_two_colors(self) -> None:
        """Half pixels red, half green → multi-color score = 1.0."""
        H, W = 2, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}

        probs = torch.full((NUM_CLASSES_V3, H, W), 0.01)
        probs[RED_IDX, 0, :] = 0.95
        probs[GREEN_IDX, 1, :] = 0.95

        result = checker.check_frame(probs, frame_objects, 0)
        # red mass = 4*0.95 + 4*0.01, green mass = 4*0.01 + 4*0.95
        red_mass = 4 * 0.95 + 4 * 0.01
        green_mass = 4 * 0.01 + 4 * 0.95
        self.assertAlmostEqual(
            result["10"].traffic_light_multi_color_score,
            min(red_mass, green_mass) / max(red_mass, green_mass),
            places=5,
        )

    def test_unknown_class_excluded(self) -> None:
        """High 'unknown' probability does not affect multi-color score."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["unknown"])}

        probs = torch.full((NUM_CLASSES_V3, H, W), 0.01)
        probs[UNKNOWN_IDX] = 0.95
        probs[RED_IDX] = 0.30
        probs[GREEN_IDX] = 0.15

        result = checker.check_frame(probs, frame_objects, 0)
        # Only red, yellow, green considered: masses = 16*0.30, 16*0.01, 16*0.15
        red_mass = 16 * 0.30
        yellow_mass = 16 * 0.01
        green_mass = 16 * 0.15
        sorted_masses = sorted([red_mass, yellow_mass, green_mass], reverse=True)
        self.assertAlmostEqual(
            result["10"].traffic_light_multi_color_score,
            sorted_masses[1] / sorted_masses[0],
            places=5,
        )

    def test_multi_color_single_patch_zero_confidence(self) -> None:
        """All visible pixels in one patch → confidence = 0."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX, prob=0.9)

        result = checker.check_frame(probs, frame_objects, 0)
        self.assertAlmostEqual(result["10"].confidence, 0.0)


class TestSimilarityFilter(unittest.TestCase):
    """Test that frame_similarity filters pixels to MATCH-only."""

    def _make_checker(self, visibility_masks: dict[str, np.ndarray]) -> TrafficLightStateChecker:
        helper = _make_cwip_helper("v3")
        rasterizer = _make_scene_rasterizer(visibility_masks)
        return TrafficLightStateChecker(helper, rasterizer)

    def test_none_similarity_uses_full_visibility_mask(self) -> None:
        """frame_similarity=None preserves the original behavior."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX, prob=0.95)

        result_without = checker.check_frame(probs, frame_objects, 0, frame_similarity=None)
        self.assertIn("10", result_without)

        sim_all_match = np.full((H, W), PatchSimilarityType.MATCH.value, dtype=np.int64)
        result_with = checker.check_frame(probs, frame_objects, 0, frame_similarity=sim_all_match)
        self.assertAlmostEqual(
            result_without["10"].traffic_light_state_score,
            result_with["10"].traffic_light_state_score,
            places=5,
        )
        self.assertAlmostEqual(result_without["10"].confidence, result_with["10"].confidence, places=5)
        self.assertAlmostEqual(
            result_without["10"].traffic_light_multi_color_score,
            result_with["10"].traffic_light_multi_color_score,
            places=5,
        )

    def test_all_match_same_as_no_filter(self) -> None:
        """All pixels classified as MATCH → identical to no similarity filter."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX, prob=0.95)

        sim = np.full((H, W), PatchSimilarityType.MATCH.value, dtype=np.int64)
        result = checker.check_frame(probs, frame_objects, 0, frame_similarity=sim)

        per_pixel_mass = 0.95 + 0.01 + 0.01 + 0.01
        self.assertAlmostEqual(result["10"].traffic_light_state_score, 0.95 / per_pixel_mass, places=5)

    def test_no_match_pixels_skips_object(self) -> None:
        """All pixels classified as non-MATCH → effective mask is empty, object skipped."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX, prob=0.95)

        sim = np.full((H, W), PatchSimilarityType.EXTRA.value, dtype=np.int64)
        result = checker.check_frame(probs, frame_objects, 0, frame_similarity=sim)
        self.assertEqual(result, {})

    def test_partial_match_restricts_pixels(self) -> None:
        """Only MATCH pixels contribute; non-MATCH pixels are excluded even if visible."""
        H, W = 2, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}

        probs = torch.full((NUM_CLASSES_V3, H, W), 0.01)
        probs[RED_IDX, 0, :] = 0.95  # top row: strong red
        probs[GREEN_IDX, 1, :] = 0.95  # bottom row: strong green

        sim = np.full((H, W), PatchSimilarityType.EXTRA.value, dtype=np.int64)
        sim[0, :] = PatchSimilarityType.MATCH.value  # only top row is MATCH

        result = checker.check_frame(probs, frame_objects, 0, frame_similarity=sim)

        w_per_pixel = 0.95 + 0.01 + 0.01 + 0.01
        total_mass = 4 * w_per_pixel  # only 4 top-row pixels
        expected_score = (4 * 0.95) / total_mass
        self.assertAlmostEqual(result["10"].traffic_light_state_score, expected_score, places=5)

    def test_similarity_intersects_with_visibility_mask(self) -> None:
        """Effective mask is vis_mask AND match_mask; requires both conditions."""
        H, W = 4, 4
        vis_mask = np.zeros((H, W), dtype=bool)
        vis_mask[0, :] = True  # only top row visible
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX, prob=0.95)

        sim = np.full((H, W), PatchSimilarityType.EXTRA.value, dtype=np.int64)
        sim[:, 0:2] = PatchSimilarityType.MATCH.value  # left two columns are MATCH

        result = checker.check_frame(probs, frame_objects, 0, frame_similarity=sim)

        # Effective: top row AND left two columns → pixels (0,0) and (0,1)
        w_per_pixel = 0.95 + 0.01 + 0.01 + 0.01
        total_mass = 2 * w_per_pixel
        expected_score = (2 * 0.95) / total_mass
        self.assertAlmostEqual(result["10"].traffic_light_state_score, expected_score, places=5)

    def test_similarity_filter_causes_low_mass_nan(self) -> None:
        """If similarity filter leaves too few high-TL-mass pixels, result is NaN."""
        H, W = 2, 2
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}

        probs = torch.full((NUM_CLASSES_V3, H, W), 0.01)
        probs[RED_IDX, 0, 0] = 0.95  # only one pixel has meaningful TL mass

        sim = np.full((H, W), PatchSimilarityType.EXTRA.value, dtype=np.int64)
        sim[0, 0] = PatchSimilarityType.MATCH.value  # only one MATCH pixel

        result = checker.check_frame(probs, frame_objects, 0, frame_similarity=sim)

        # Total TL mass from single pixel: 0.95 + 0.01 + 0.01 + 0.01 = 0.98 > 0.5 threshold
        self.assertIn("10", result)
        self.assertFalse(np.isnan(result["10"].traffic_light_state_score))

    def test_similarity_as_torch_tensor(self) -> None:
        """frame_similarity works when passed as a torch.Tensor."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX, prob=0.95)

        sim = torch.full((H, W), PatchSimilarityType.MATCH.value, dtype=torch.long)
        result = checker.check_frame(probs, frame_objects, 0, frame_similarity=sim)

        per_pixel_mass = 0.95 + 0.01 + 0.01 + 0.01
        self.assertAlmostEqual(result["10"].traffic_light_state_score, 0.95 / per_pixel_mass, places=5)

    def test_dont_care_pixels_excluded(self) -> None:
        """DONT_CARE similarity pixels are excluded by the MATCH filter."""
        H, W = 4, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}
        probs = _make_uniform_probs(H, W, RED_IDX, prob=0.95)

        sim = np.full((H, W), PatchSimilarityType.DONT_CARE.value, dtype=np.int64)
        result = checker.check_frame(probs, frame_objects, 0, frame_similarity=sim)
        self.assertEqual(result, {})

    def test_multi_color_respects_similarity_filter(self) -> None:
        """Multi-color score only considers MATCH pixels."""
        H, W = 2, 4
        vis_mask = np.ones((H, W), dtype=bool)
        checker = self._make_checker({"10": vis_mask})

        frame_objects = {"10": _make_traffic_light_obj(state_per_frame=["red"])}

        probs = torch.full((NUM_CLASSES_V3, H, W), 0.01)
        probs[RED_IDX, 0, :] = 0.95  # top row: strong red
        probs[GREEN_IDX, 1, :] = 0.95  # bottom row: strong green

        sim = np.full((H, W), PatchSimilarityType.EXTRA.value, dtype=np.int64)
        sim[0, :] = PatchSimilarityType.MATCH.value  # only top row is MATCH

        result = checker.check_frame(probs, frame_objects, 0, frame_similarity=sim)

        # Only red-dominant top row pixels are considered → low multi-color
        self.assertAlmostEqual(result["10"].traffic_light_multi_color_score, 0.01 / 0.95, places=5)


if __name__ == "__main__":
    unittest.main()
