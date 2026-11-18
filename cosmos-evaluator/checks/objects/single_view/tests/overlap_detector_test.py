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

"""Unit tests for checks.objects.single_view.overlap_detector."""

import logging
from typing import Any, ClassVar, cast
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from numpy.testing import assert_array_equal

from checks.objects.single_view.overlap_detector import OverlapDetector
from checks.utils.config_manager import ConfigManager


class TestScoreRatio(unittest.TestCase):
    def test_full_overlap(self) -> None:
        m = np.ones((3, 3), dtype=bool)
        self.assertEqual(OverlapDetector._score_ratio(m, m), 1.0)

    def test_half_overlap(self) -> None:
        class_m = np.zeros((4, 4), dtype=bool)
        class_m[:2, :] = True
        proj = np.ones((4, 4), dtype=bool)
        score = OverlapDetector._score_ratio(class_m, proj)
        self.assertAlmostEqual(score, 0.5, places=5)

    def test_none_projected(self) -> None:
        self.assertEqual(OverlapDetector._score_ratio(np.ones((2, 2), dtype=bool), cast(np.ndarray, None)), 0.0)

    def test_empty_projected(self) -> None:
        z = np.zeros((2, 2), dtype=bool)
        self.assertEqual(OverlapDetector._score_ratio(np.ones((2, 2), dtype=bool), z), 0.0)


class TestScoreCluster(unittest.TestCase):
    def test_returns_zero_without_labels(self) -> None:
        proj = np.ones((2, 2), dtype=bool)
        self.assertEqual(OverlapDetector._score_cluster(np.ones((2, 2), dtype=bool), proj, None, None), 0.0)

    def test_happy_path(self) -> None:
        labels = np.zeros((4, 4), dtype=np.int32)
        labels[1:3, 1:3] = 1
        proj = np.zeros((4, 4), dtype=bool)
        proj[1:3, 1:3] = True
        stats = np.zeros((2, 5), dtype=np.int32)
        stats[1, cv2.CC_STAT_AREA] = 4
        score = OverlapDetector._score_cluster(np.ones((4, 4), dtype=bool), proj, labels, stats)
        self.assertEqual(score, 1.0)

    def test_no_seed_labels(self) -> None:
        labels = np.zeros((3, 3), dtype=np.int32)
        proj = np.eye(3, dtype=bool)
        stats = np.zeros((1, 5), dtype=np.int32)
        self.assertEqual(OverlapDetector._score_cluster(np.ones((3, 3), dtype=bool), proj, labels, stats), 0.0)


class TestScoreObject(unittest.TestCase):
    log: ClassVar[logging.Logger]

    @classmethod
    def setUpClass(cls) -> None:
        cls.log = logging.getLogger("overlap_test")
        if not any(isinstance(h, logging.NullHandler) for h in cls.log.handlers):
            cls.log.addHandler(logging.NullHandler())

    def setUp(self) -> None:
        self.seg = MagicMock()
        self.det = OverlapDetector(oc_config={}, seg_model=self.seg, logger=self.log)
        self.det.importance_filter = MagicMock()
        self.det.importance_filter.should_process_object.return_value = (True, "")

    def test_invalid_method_logs_and_returns_none(self) -> None:
        geom = MagicMock()
        geom.get_projected_mask.return_value = (np.ones((2, 2), dtype=bool), np.zeros((2, 2), dtype=np.float32))
        obj = {"geometry": geom, "object_type": "Car"}
        score = self.det._score_object(
            1,
            obj,
            np.ones((2, 2), dtype=np.uint8),
            np.eye(4),
            MagicMock(),
            2,
            2,
            "not_a_method",
        )
        self.assertIsNone(score)

    def test_ratio_branch(self) -> None:
        geom = MagicMock()
        geom.get_projected_mask.return_value = (np.ones((2, 2), dtype=bool), np.zeros((2, 2), dtype=np.float32))
        obj = {"geometry": geom, "object_type": "Car"}
        mask = np.ones((2, 2), dtype=np.uint8) * 255
        s = self.det._score_object(1, obj, mask, np.eye(4), MagicMock(), 2, 2, "ratio")
        self.assertIsNotNone(s)
        assert s is not None
        self.assertAlmostEqual(s, 1.0, places=5)


class TestProcessClass(unittest.TestCase):
    def setUp(self) -> None:
        self.log = logging.getLogger("overlap_proc")
        self.log.addHandler(logging.NullHandler())

    @patch("checks.objects.single_view.overlap_detector.RdsDataLoader.get_objects_by_class", return_value={})
    def test_no_objects_returns_empty(self, _mock_gobc: MagicMock) -> None:
        det = OverlapDetector({}, MagicMock(), self.log)
        scores, filtered, dbg = det.process_class(
            "vehicle",
            None,
            {},
            np.eye(4),
            MagicMock(),
            10,
            10,
            {},
            0,
        )
        self.assertEqual(scores, {})
        self.assertEqual(filtered, {})
        self.assertIsNone(dbg)

    @patch("checks.objects.single_view.overlap_detector.OverlapDetector._score_object", return_value=0.75)
    @patch("checks.objects.single_view.overlap_detector.RdsDataLoader.get_objects_by_class")
    def test_unimportant_still_scored(self, mock_gobc: MagicMock, _mock_score: MagicMock) -> None:
        """Importance is tracked for logging; unimportant objects are still scored (not in filtered)."""
        geom = MagicMock()
        geom.in_frame_fraction.return_value = 1.0  # fully in-frame: not gated by the in-frame gate
        mock_gobc.return_value = {1: {"geometry": geom, "object_type": "Car"}}
        det = OverlapDetector({}, MagicMock(), self.log)
        det.importance_filter = MagicMock()
        det.importance_filter.config = {"distance_threshold_m": 100.0, "relevant_lanes": []}
        det.importance_filter.get_importance_attributes.return_value = {
            "lane": "n/a",
            "is_oncoming": False,
            "distance_m": 999.0,
            "is_important": False,
        }
        det.seg_model.get_class_mask.return_value = np.ones((4, 4), dtype=np.uint8)
        scores, filtered, _ = det.process_class(
            "vehicle",
            MagicMock(),
            cast(dict[str, Any], {1: {}}),
            np.eye(4),
            MagicMock(),
            4,
            4,
            {},
            0,
        )
        # Scoring still runs; filtered_objects is not used for importance in current workflow
        self.assertNotIn(1, filtered)
        self.assertEqual(cast(dict[int, float], scores)[1], 0.75)
        det.importance_filter.get_importance_attributes.assert_called()

    @patch("checks.objects.single_view.overlap_detector.OverlapDetector._score_object", return_value=None)
    @patch("checks.objects.single_view.overlap_detector.RdsDataLoader.get_objects_by_class")
    def test_debug_bbox_populated(self, mock_gobc: MagicMock, _mock_score: MagicMock) -> None:
        """Debug bboxes are recorded when scoring fails (e.g. occluded) and debug is enabled."""
        geom = MagicMock()
        geom.in_frame_fraction.return_value = 1.0  # fully in-frame: gate doesn't fire before the (mocked) None score
        mask = np.zeros((5, 5), dtype=bool)
        mask[1:4, 1:4] = True
        geom.get_projected_mask.return_value = (mask, np.full((5, 5), 1.0, dtype=np.float32))
        mock_gobc.return_value = {9: {"geometry": geom, "object_type": "Car", "object_type_index": 3}}
        det = OverlapDetector({}, MagicMock(), self.log, debug_enabled=True)
        det.importance_filter = MagicMock()
        det.importance_filter.config = {"distance_threshold_m": 100.0, "relevant_lanes": []}
        det.importance_filter.get_importance_attributes.return_value = {
            "lane": "n/a",
            "is_oncoming": False,
            "distance_m": 0.0,
            "is_important": True,
        }
        det.seg_model.get_class_mask.return_value = np.ones((5, 5), dtype=np.uint8)
        scores, _f, dbg = det.process_class(
            "vehicle",
            MagicMock(),
            cast(dict[str, Any], {9: {}}),
            np.eye(4),
            MagicMock(),
            5,
            5,
            {},
            2,
        )
        self.assertIsNotNone(dbg)
        assert dbg is not None
        self.assertGreater(len(dbg["frame_boxes"]), 0)
        self.assertTrue(np.isnan(cast(dict[int, float], scores)[9]))


class TestProjectionAreaGate(unittest.TestCase):
    """Per-frame projection-area gate skips scoring for tracks whose visible projection is below ``min_projection_area``
    for that frame."""

    def setUp(self) -> None:
        self.log = logging.getLogger("overlap_gate")
        self.log.addHandler(logging.NullHandler())

    def _make_rasterizer(self, area_by_id: dict) -> MagicMock:
        ras = MagicMock()
        ras.has_object.side_effect = lambda tid: tid in area_by_id
        ras.get_projection_metadata.side_effect = lambda tid: {
            "bbox_width": 0,
            "bbox_height": 0,
            "visible_area_px": int(area_by_id[tid]),
        }
        ras.get_visibility_mask.side_effect = lambda tid: np.ones((4, 4), dtype=bool)
        return ras

    @patch("checks.objects.single_view.overlap_detector.OverlapDetector._score_object", return_value=0.5)
    @patch("checks.objects.single_view.overlap_detector.RdsDataLoader.get_objects_by_class")
    def test_below_threshold_skips_scoring(self, mock_gobc: MagicMock, _mock_score: MagicMock) -> None:
        """Track with visible_area_px below threshold yields NaN score and no projection_attrs."""
        mock_gobc.return_value = {7: {"geometry": MagicMock(), "object_type": "TrafficLight"}}
        cfg = {"overlap_check": {"traffic_light": {"method": "ratio", "min_projection_area": 200}}}
        det = OverlapDetector(cfg, MagicMock(), self.log)
        det.importance_filter = MagicMock()
        det.importance_filter.get_importance_attributes.return_value = {
            "lane": "n/a",
            "is_oncoming": False,
            "distance_m": 30.0,
            "is_important": True,
        }
        det.seg_model.get_class_mask.return_value = np.ones((4, 4), dtype=np.uint8)
        track_agg: dict = {}
        ras = self._make_rasterizer({"7": 50})  # below 200
        scores, _, _ = det.process_class(
            "traffic_light",
            MagicMock(),
            cast(dict[str, Any], {7: {}}),
            np.eye(4),
            MagicMock(),
            4,
            4,
            track_agg,
            5,
            scene_rasterizer=ras,
        )
        self.assertTrue(np.isnan(cast(dict[int, float], scores)[7]))
        self.assertNotIn(7, track_agg)  # gate happened before any aggregation

    @patch("checks.objects.single_view.overlap_detector.OverlapDetector._score_object", return_value=0.9)
    @patch("checks.objects.single_view.overlap_detector.RdsDataLoader.get_objects_by_class")
    def test_above_threshold_is_scored(self, mock_gobc: MagicMock, _mock_score: MagicMock) -> None:
        """Track passing threshold is scored normally."""
        mock_gobc.return_value = {7: {"geometry": MagicMock(), "object_type": "TrafficLight"}}
        cfg = {"overlap_check": {"traffic_light": {"method": "ratio", "min_projection_area": 200}}}
        det = OverlapDetector(cfg, MagicMock(), self.log)
        det.importance_filter = MagicMock()
        det.importance_filter.get_importance_attributes.return_value = {
            "lane": "n/a",
            "is_oncoming": False,
            "distance_m": 30.0,
            "is_important": True,
        }
        det.seg_model.get_class_mask.return_value = np.ones((4, 4), dtype=np.uint8)
        track_agg: dict = {}
        ras = self._make_rasterizer({"7": 350})  # above 200
        scores, _, _ = det.process_class(
            "traffic_light",
            MagicMock(),
            cast(dict[str, Any], {7: {}}),
            np.eye(4),
            MagicMock(),
            4,
            4,
            track_agg,
            5,
            scene_rasterizer=ras,
        )
        self.assertEqual(cast(dict[int, float], scores)[7], 0.9)
        # Track was aggregated normally (importance attrs recorded).
        self.assertIn(7, track_agg)
        self.assertEqual(len(track_agg[7].get("importance_attrs", [])), 1)

    @patch("checks.objects.single_view.overlap_detector.OverlapDetector._score_object", return_value=0.6)
    @patch("checks.objects.single_view.overlap_detector.RdsDataLoader.get_objects_by_class")
    def test_no_threshold_does_not_gate(self, mock_gobc: MagicMock, _mock_score: MagicMock) -> None:
        """Classes without ``min_projection_area`` (dynamic types) are not gated."""
        geom = MagicMock()
        geom.in_frame_fraction.return_value = 1.0  # fully in-frame: not gated by the in-frame gate
        mock_gobc.return_value = {3: {"geometry": geom, "object_type": "Car"}}
        cfg = {"overlap_check": {"vehicle": {"method": "ratio"}}}  # no min_projection_area
        det = OverlapDetector(cfg, MagicMock(), self.log)
        det.importance_filter = MagicMock()
        det.importance_filter.get_importance_attributes.return_value = {
            "lane": "ego",
            "is_oncoming": False,
            "distance_m": 10.0,
            "is_important": True,
        }
        det.seg_model.get_class_mask.return_value = np.ones((4, 4), dtype=np.uint8)
        track_agg: dict = {}
        ras = self._make_rasterizer({"3": 5})  # tiny — would be gated if threshold applied
        scores, _, _ = det.process_class(
            "vehicle",
            MagicMock(),
            cast(dict[str, Any], {3: {}}),
            np.eye(4),
            MagicMock(),
            4,
            4,
            track_agg,
            0,
            scene_rasterizer=ras,
        )
        self.assertEqual(cast(dict[int, float], scores)[3], 0.6)


class TestProcessClassStaticRecordsImportance(unittest.TestCase):
    """``is_static=True`` now records importance attributes so every static object carries its GT ego-distance.

    Static objects are always important, so the ``is_important`` filtering / filtered-count logging stays dynamic-only;
    but each static track records per-frame ``importance_attrs`` (with the real ``distance_m``) so ``min_distance_m``
    is available for distance-based report breakdowns.
    """

    def setUp(self) -> None:
        self.log = logging.getLogger("overlap_static")
        self.log.addHandler(logging.NullHandler())

    @patch("checks.objects.single_view.overlap_detector.OverlapDetector._score_object", return_value=0.7)
    @patch("checks.objects.single_view.overlap_detector.RdsDataLoader.get_objects_by_class")
    def test_static_records_importance_attrs(self, mock_gobc: MagicMock, _mock_score: MagicMock) -> None:
        mock_gobc.return_value = {5: {"geometry": MagicMock(), "object_type": "TrafficLight"}}
        det = OverlapDetector({}, MagicMock(), self.log)
        det.importance_filter = MagicMock()
        det.importance_filter.get_importance_attributes.return_value = {
            "lane": "n/a",
            "is_oncoming": False,
            "distance_m": 42.0,
            "is_important": True,
        }
        det.seg_model.get_class_mask.return_value = np.ones((4, 4), dtype=np.uint8)
        track_agg: dict = {}

        scores, _filtered, _dbg = det.process_class(
            "traffic_light",
            MagicMock(),
            cast(dict[str, Any], {5: {}}),
            np.eye(4),
            MagicMock(),
            4,
            4,
            track_agg,
            0,
            is_static=True,
        )

        self.assertEqual(cast(dict[int, float], scores)[5], 0.7)
        # Importance IS now computed + recorded for the static track (for the GT distance).
        det.importance_filter.get_importance_attributes.assert_called()
        self.assertIn(5, track_agg)
        self.assertIn("importance_attrs", track_agg[5])
        self.assertEqual(track_agg[5]["importance_attrs"][0]["distance_m"], 42.0)

    @patch("checks.objects.single_view.overlap_detector.OverlapDetector._score_object", return_value=0.7)
    @patch("checks.objects.single_view.overlap_detector.RdsDataLoader.get_objects_by_class")
    def test_dynamic_default_records_importance(self, mock_gobc: MagicMock, _mock_score: MagicMock) -> None:
        """Default ``is_static=False`` (dynamic) still records importance attrs."""
        geom = MagicMock()
        geom.in_frame_fraction.return_value = 1.0  # fully in-frame: not gated by the in-frame gate
        mock_gobc.return_value = {3: {"geometry": geom, "object_type": "Car"}}
        det = OverlapDetector({}, MagicMock(), self.log)
        det.importance_filter = MagicMock()
        det.importance_filter.config = {"distance_threshold_m": 100.0, "relevant_lanes": []}
        det.importance_filter.get_importance_attributes.return_value = {
            "lane": "ego",
            "is_oncoming": False,
            "distance_m": 10.0,
            "is_important": True,
        }
        det.seg_model.get_class_mask.return_value = np.ones((4, 4), dtype=np.uint8)
        track_agg: dict = {}

        det.process_class(
            "vehicle",
            MagicMock(),
            cast(dict[str, Any], {3: {}}),
            np.eye(4),
            MagicMock(),
            4,
            4,
            track_agg,
            0,
        )

        det.importance_filter.get_importance_attributes.assert_called_once()
        self.assertEqual(len(track_agg[3].get("importance_attrs", [])), 1)


class TestInFrameGate(unittest.TestCase):
    """In-frame gate: a GT object the frame edge truncates below ``overlap_check.<class>.min_in_frame_fraction``
    is left NaN (treated like outside-FOV) so frame-edge slivers cannot create false 'missing' verdicts. The
    threshold is per-class config, applies to static and dynamic classes alike, and is disabled when absent.
    Strict ``<`` so an object at exactly the threshold is still scored."""

    def setUp(self) -> None:
        self.log = logging.getLogger("overlap_inframe")
        self.log.addHandler(logging.NullHandler())
        # Geometry mock from the most recent _run_gate call, so a test can assert whether the
        # geometry fallback was exercised.
        self.last_geometry = MagicMock()

    def _run_gate(
        self,
        class_name: str,
        object_type: str,
        fraction: float,
        is_static: bool = False,
        track_id: int = 11,
        score_value: float = 0.8,
        min_in_frame_fraction: float | None = 1.0,
        scene_rasterizer: Any = None,
    ) -> tuple[dict, dict]:
        """Run process_class for one object whose ``in_frame_fraction`` is mocked to ``fraction``;
        ``_score_object`` is mocked to ``score_value`` so a non-gated object is scored. The class is configured
        with ``min_in_frame_fraction`` unless that is None (key omitted -> gate disabled). Returns
        ``(scores, track_output_agg)``."""
        class_cfg: dict[str, Any] = {"method": "ratio"}
        if min_in_frame_fraction is not None:
            class_cfg["min_in_frame_fraction"] = min_in_frame_fraction
        det = OverlapDetector({"overlap_check": {class_name: class_cfg}}, MagicMock(), self.log)
        det.importance_filter = MagicMock()
        det.importance_filter.config = {"distance_threshold_m": 100.0, "relevant_lanes": []}
        det.importance_filter.get_importance_attributes.return_value = {
            "lane": "ego",
            "is_oncoming": False,
            "distance_m": 10.0,
            "is_important": True,
        }
        det.seg_model.get_class_mask.return_value = np.ones((4, 4), dtype=np.uint8)
        geom = MagicMock()
        geom.in_frame_fraction.return_value = fraction
        self.last_geometry = geom
        track_agg: dict = {}
        with (
            patch.object(OverlapDetector, "_score_object", return_value=score_value),
            patch(
                "checks.objects.single_view.overlap_detector.RdsDataLoader.get_objects_by_class",
                return_value={track_id: {"geometry": geom, "object_type": object_type}},
            ),
        ):
            scores, _filtered, _dbg = det.process_class(
                class_name,
                MagicMock(),
                cast(dict[str, Any], {track_id: {}}),
                np.eye(4),
                MagicMock(),
                4,
                4,
                track_agg,
                0,
                scene_rasterizer=scene_rasterizer,
                is_static=is_static,
            )
        return scores, track_agg

    def test_vehicle_below_half_gated_and_aggregated(self) -> None:
        """Vehicle < 0.5 in-frame: score NaN, with edge_truncated count, object_type, and
        importance attrs preserved (track stays in results, just not scored)."""
        scores, agg = self._run_gate("vehicle", "Car", 0.4, track_id=11, min_in_frame_fraction=0.5)
        self.assertTrue(np.isnan(scores[11]))
        self.assertEqual(agg[11]["output_counts"].get("edge_truncated"), 1)
        self.assertIn("Car", agg[11]["object_types"])
        self.assertEqual(len(agg[11].get("importance_attrs", [])), 1)

    def test_vehicle_exactly_half_is_scored(self) -> None:
        """Strict ``<``: a vehicle at exactly 0.5 is scored (not gated)."""
        scores, agg = self._run_gate("vehicle", "Car", 0.5, track_id=11, min_in_frame_fraction=0.5)
        self.assertEqual(scores[11], 0.8)
        self.assertNotIn("edge_truncated", agg[11]["output_counts"])

    def test_vehicle_just_below_half_gated(self) -> None:
        scores, agg = self._run_gate("vehicle", "Car", 0.5 - 1e-6, track_id=11, min_in_frame_fraction=0.5)
        self.assertTrue(np.isnan(scores[11]))
        self.assertEqual(agg[11]["output_counts"].get("edge_truncated"), 1)

    def test_pedestrian_fully_in_is_scored(self) -> None:
        scores, agg = self._run_gate("pedestrian", "Pedestrian", 1.0, track_id=11)
        self.assertEqual(scores[11], 0.8)
        self.assertNotIn("edge_truncated", agg[11]["output_counts"])

    def test_pedestrian_just_below_one_gated(self) -> None:
        """A threshold of 1.0 requires full containment: any clamping (< 1.0) gates the frame."""
        scores, agg = self._run_gate("pedestrian", "Pedestrian", 1.0 - 1e-6, track_id=11)
        self.assertTrue(np.isnan(scores[11]))
        self.assertEqual(agg[11]["output_counts"].get("edge_truncated"), 1)

    def test_motorcycle_requires_full_containment(self) -> None:
        scores, _agg = self._run_gate("motorcycle", "Motorcycle", 0.95, track_id=11)
        self.assertTrue(np.isnan(scores[11]))

    def test_static_traffic_light_is_gated(self) -> None:
        """The gate is not dynamic-only: a truncated static traffic light abstains just like a dynamic
        object, so its clipped projection cannot produce a spurious correspondence score."""
        scores, agg = self._run_gate("traffic_light", "TrafficLight", 0.6, is_static=True, track_id=11)
        self.assertTrue(np.isnan(scores[11]))
        self.assertEqual(agg[11]["output_counts"].get("edge_truncated"), 1)
        self.assertIn("TrafficLight", agg[11]["object_types"])

    def test_static_traffic_light_fully_in_frame_is_scored(self) -> None:
        scores, agg = self._run_gate("traffic_light", "TrafficLight", 1.0, is_static=True, track_id=11)
        self.assertEqual(scores[11], 0.8)
        self.assertNotIn("edge_truncated", agg[11]["output_counts"])

    def test_class_without_config_key_is_never_gated(self) -> None:
        """Classes with no ``min_in_frame_fraction`` (e.g. traffic_sign today) are never gated, even when
        the projection is almost entirely off-frame."""
        scores, agg = self._run_gate(
            "traffic_sign", "TrafficSign", 0.01, is_static=True, track_id=11, min_in_frame_fraction=None
        )
        self.assertEqual(scores[11], 0.8)
        self.assertNotIn("edge_truncated", agg[11]["output_counts"])

    def test_zero_threshold_disables_gate(self) -> None:
        """An explicit 0 disables the gate, matching the other per-class abstain knobs."""
        scores, _agg = self._run_gate("vehicle", "Car", 0.01, track_id=11, min_in_frame_fraction=0.0)
        self.assertEqual(scores[11], 0.8)

    def test_rasterizer_fraction_preferred_over_geometry(self) -> None:
        """When a rasterizer carries the object, its memoized fraction is used and the geometry is not
        re-projected — the two consumers of this value share one projection per object per frame."""
        rasterizer = MagicMock()
        rasterizer.has_object.return_value = True
        rasterizer.in_frame_fraction.return_value = 0.2
        rasterizer.get_projection_metadata.return_value = {"visible_area_px": 10_000}
        scores, agg = self._run_gate(
            "traffic_light",
            "TrafficLight",
            1.0,  # geometry would say "fully in frame"; the rasterizer's 0.2 must win
            is_static=True,
            track_id=11,
            scene_rasterizer=rasterizer,
        )
        self.assertTrue(np.isnan(scores[11]))
        self.assertEqual(agg[11]["output_counts"].get("edge_truncated"), 1)
        rasterizer.in_frame_fraction.assert_called_with("11")
        # The point of the rasterizer path is that the cuboid is projected once per frame and the
        # result shared, so the geometry fallback must stay cold.
        self.last_geometry.in_frame_fraction.assert_not_called()


class TestAggregateOutputSyntheticId(unittest.TestCase):
    """_aggregate_output carries a synthesized object's synthetic_source_id into the agg entry."""

    def setUp(self) -> None:
        self.det = OverlapDetector({}, MagicMock(), logging.getLogger("test"))

    def test_synthetic_source_id_propagated_from_attributes(self) -> None:
        agg: dict[int, dict[str, Any]] = {}
        tracked = {
            "object_type": "TrafficLight",
            "object_type_index": 8,
            "attributes": {"feature_id": "1388", "synthetic_source_id": "tl_add_between_pair"},
        }
        self.det._aggregate_output(agg, 7, tracked, "scored")
        self.assertEqual(agg[7]["synthetic_source_id"], "tl_add_between_pair")
        self.assertEqual(agg[7]["object_type_index"], 8)

    def test_real_object_has_no_synthetic_source_id(self) -> None:
        agg: dict[int, dict[str, Any]] = {}
        real = {"object_type": "TrafficLight", "attributes": {"feature_id": "42"}}
        self.det._aggregate_output(agg, 3, real, "scored")
        self.assertNotIn("synthetic_source_id", agg[3])


class TestBottomOcclusionGate(unittest.TestCase):
    """Per-frame top-only-occlusion gate: a dynamic track occluded from the bottom up beyond the per-class
    ``overlap_check.<type>.bottom_occlusion_abstain_frac`` is abstained (left NaN) instead of scored ~0."""

    def setUp(self) -> None:
        self.log = logging.getLogger("overlap_occ")
        self.log.addHandler(logging.NullHandler())

    def _make_det(self, frac: float) -> OverlapDetector:
        cfg = {"overlap_check": {"vehicle": {"method": "ratio", "bottom_occlusion_abstain_frac": frac}}}
        det = OverlapDetector(cfg, MagicMock(), self.log)
        det.importance_filter = MagicMock()
        det.importance_filter.get_importance_attributes.return_value = {
            "lane": "ego",
            "is_oncoming": False,
            "distance_m": 10.0,
            "is_important": True,
        }
        det.seg_model.get_class_mask.return_value = np.ones((4, 4), dtype=np.uint8)
        return det

    def _make_rasterizer(self, occ_by_id: dict) -> MagicMock:
        ras = MagicMock()
        ras.has_object.side_effect = lambda tid: tid in occ_by_id
        ras.bottom_occluded_frac.side_effect = lambda tid: float(occ_by_id[tid])
        ras.get_visibility_mask.side_effect = lambda tid: np.ones((4, 4), dtype=bool)
        return ras

    @staticmethod
    def _geom() -> MagicMock:
        geom = MagicMock()
        geom.in_frame_fraction.return_value = 1.0  # fully in-frame: in-frame gate does not fire first
        return geom

    def _run(self, det: OverlapDetector, ras: MagicMock, track_agg: dict) -> dict:
        scores, _, _ = det.process_class(
            "vehicle",
            MagicMock(),
            cast(dict[str, Any], {3: {}}),
            np.eye(4),
            MagicMock(),
            4,
            4,
            track_agg,
            5,
            scene_rasterizer=ras,
        )
        return cast(dict[int, float], scores)

    @patch("checks.objects.single_view.overlap_detector.OverlapDetector._score_object", return_value=0.7)
    @patch("checks.objects.single_view.overlap_detector.RdsDataLoader.get_objects_by_class")
    def test_top_only_occluded_abstains(self, mock_gobc: MagicMock, mock_score: MagicMock) -> None:
        """Bottom-occlusion >= threshold yields NaN and never reaches scoring."""
        mock_gobc.return_value = {3: {"geometry": self._geom(), "object_type": "Car"}}
        track_agg: dict = {}
        scores = self._run(self._make_det(0.5), self._make_rasterizer({"3": 0.8}), track_agg)
        self.assertTrue(np.isnan(scores[3]))
        mock_score.assert_not_called()
        self.assertIn(3, track_agg)  # importance recorded before the gate; track still present

    @patch("checks.objects.single_view.overlap_detector.OverlapDetector._score_object", return_value=0.7)
    @patch("checks.objects.single_view.overlap_detector.RdsDataLoader.get_objects_by_class")
    def test_partly_occluded_is_scored(self, mock_gobc: MagicMock, mock_score: MagicMock) -> None:
        """Bottom-occlusion below threshold is scored normally."""
        mock_gobc.return_value = {3: {"geometry": self._geom(), "object_type": "Car"}}
        scores = self._run(self._make_det(0.5), self._make_rasterizer({"3": 0.2}), {})
        self.assertEqual(scores[3], 0.7)
        mock_score.assert_called_once()

    @patch("checks.objects.single_view.overlap_detector.OverlapDetector._score_object", return_value=0.7)
    @patch("checks.objects.single_view.overlap_detector.RdsDataLoader.get_objects_by_class")
    def test_threshold_zero_disables_gate(self, mock_gobc: MagicMock, mock_score: MagicMock) -> None:
        """Threshold 0 disables the gate: even a fully bottom-occluded track is scored."""
        mock_gobc.return_value = {3: {"geometry": self._geom(), "object_type": "Car"}}
        scores = self._run(self._make_det(0.0), self._make_rasterizer({"3": 0.95}), {})
        self.assertEqual(scores[3], 0.7)
        mock_score.assert_called_once()


class TestScoreSpine(unittest.TestCase):
    """The spine metric: fraction of the visible centerline the segmentation labels with this class."""

    def test_fully_labeled_spine(self) -> None:
        spine = np.zeros((4, 4), dtype=bool)
        spine[2, :] = True
        self.assertEqual(OverlapDetector._score_spine(np.ones((4, 4), dtype=bool), spine), 1.0)

    def test_partially_labeled_spine(self) -> None:
        spine = np.zeros((4, 4), dtype=bool)
        spine[2, :] = True
        class_mask = np.zeros((4, 4), dtype=bool)
        class_mask[2, :2] = True
        self.assertAlmostEqual(OverlapDetector._score_spine(class_mask, spine), 0.5, places=5)

    def test_unlabeled_spine_scores_zero(self) -> None:
        spine = np.zeros((4, 4), dtype=bool)
        spine[2, :] = True
        self.assertEqual(OverlapDetector._score_spine(np.zeros((4, 4), dtype=bool), spine), 0.0)

    def test_denominator_ignores_class_pixels_off_the_spine(self) -> None:
        """Unlike the ratio metric, segmentation that spills either side of the line neither helps nor hurts —
        only coverage along the line counts."""
        spine = np.zeros((6, 6), dtype=bool)
        spine[3, :] = True
        class_mask = np.zeros((6, 6), dtype=bool)
        class_mask[1:6, :] = True
        self.assertEqual(OverlapDetector._score_spine(class_mask, spine), 1.0)

    def test_empty_spine_scores_zero(self) -> None:
        self.assertEqual(OverlapDetector._score_spine(np.ones((3, 3), dtype=bool), np.zeros((3, 3), dtype=bool)), 0.0)

    def test_missing_class_mask_scores_zero(self) -> None:
        spine = np.zeros((3, 3), dtype=bool)
        spine[1, :] = True
        self.assertEqual(OverlapDetector._score_spine(None, spine), 0.0)

    def test_is_independent_of_ribbon_width(self) -> None:
        """The reason the metric exists: a perfectly reproduced line scores 1.0 however wide its projected
        ribbon would have been, where ``_score_ratio`` is bounded by paint width / ribbon width."""
        spine = np.zeros((9, 9), dtype=bool)
        spine[4, :] = True
        class_mask = np.zeros((9, 9), dtype=bool)
        class_mask[4, :] = True
        ribbon = np.zeros((9, 9), dtype=bool)
        ribbon[1:8, :] = True

        self.assertEqual(OverlapDetector._score_spine(class_mask, spine), 1.0)
        self.assertAlmostEqual(OverlapDetector._score_ratio(class_mask, ribbon), 1.0 / 7.0, places=5)


class TestDilateMask(unittest.TestCase):
    """Image-space tolerance applied to the class mask before spine matching."""

    def test_zero_radius_is_a_no_op(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        assert_array_equal(OverlapDetector._dilate_mask(mask, 0), mask)

    def test_grows_by_the_requested_radius(self) -> None:
        mask = np.zeros((7, 7), dtype=bool)
        mask[3, 3] = True
        grown = OverlapDetector._dilate_mask(mask, 2)
        self.assertTrue(grown[3, 1])
        self.assertTrue(grown[1, 3])
        self.assertFalse(grown[3, 0])
        self.assertEqual(grown.dtype, bool)

    def test_does_not_modify_the_input(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        OverlapDetector._dilate_mask(mask, 1)
        self.assertEqual(int(mask.sum()), 1)


class TestSpineScoringPath(unittest.TestCase):
    """``method: spine`` end to end through process_class / _score_object."""

    def setUp(self) -> None:
        self.log = logging.getLogger("overlap_spine")
        self.log.addHandler(logging.NullHandler())

    @staticmethod
    def _spine_mask(width: int = 8) -> np.ndarray:
        spine = np.zeros((width, width), dtype=bool)
        spine[width // 2, :] = True
        return spine

    def _detector(self, class_cfg: dict[str, Any]) -> OverlapDetector:
        det = OverlapDetector({"overlap_check": {"lane_line": class_cfg}}, MagicMock(), self.log)
        det.importance_filter = MagicMock()
        det.importance_filter.config = {"distance_threshold_m": 100.0, "relevant_lanes": []}
        det.importance_filter.get_importance_attributes.return_value = {
            "lane": "n/a",
            "is_oncoming": False,
            "distance_m": None,
            "is_important": True,
        }
        return det

    def _run(
        self,
        class_cfg: dict[str, Any],
        spine: np.ndarray | None,
        class_mask: np.ndarray,
        track_id: int = 7,
    ) -> tuple[dict, dict]:
        """Score one LaneLine track whose visible spine is ``spine`` (``None`` = geometry with no centerline)."""
        det = self._detector(class_cfg)
        det.seg_model.get_class_mask.return_value = class_mask

        rasterizer = MagicMock()
        rasterizer.has_object.return_value = True
        rasterizer.get_visible_spine_mask.return_value = spine
        rasterizer.get_projection_metadata.return_value = {"visible_area_px": 10_000}
        rasterizer.get_visibility_mask.return_value = np.ones_like(class_mask, dtype=bool)

        size = class_mask.shape[0]
        track_agg: dict = {}
        with patch(
            "checks.objects.single_view.overlap_detector.RdsDataLoader.get_objects_by_class",
            return_value={track_id: {"geometry": MagicMock(), "object_type": "LaneLine", "is_static": True}},
        ):
            scores, _filtered, _dbg = det.process_class(
                "lane_line",
                MagicMock(),
                cast(dict[str, Any], {track_id: {}}),
                np.eye(4),
                MagicMock(),
                size,
                size,
                track_agg,
                0,
                scene_rasterizer=rasterizer,
                is_static=True,
            )
        return scores, track_agg

    def test_scores_along_the_centerline(self) -> None:
        class_mask = np.zeros((8, 8), dtype=bool)
        class_mask[4, :4] = True
        scores, agg = self._run({"method": "spine"}, self._spine_mask(), class_mask)
        self.assertAlmostEqual(scores[7], 0.5, places=5)
        self.assertEqual(agg[7]["output_counts"]["scored"], 1)

    def test_ignores_class_pixels_beside_the_line(self) -> None:
        """A wide segmented stripe scores 1.0 rather than being penalised by the ribbon it does not fill."""
        class_mask = np.zeros((8, 8), dtype=bool)
        class_mask[2:7, :] = True
        scores, _agg = self._run({"method": "spine"}, self._spine_mask(), class_mask)
        self.assertEqual(scores[7], 1.0)

    def test_empty_spine_is_not_scored(self) -> None:
        scores, agg = self._run({"method": "spine"}, np.zeros((8, 8), dtype=bool), np.ones((8, 8), dtype=bool))
        self.assertTrue(np.isnan(scores[7]))
        self.assertEqual(agg[7]["output_counts"].get("occluded"), 1)

    def test_geometry_without_a_spine_falls_back_to_ratio(self) -> None:
        """A cuboid in a spine-configured class is still scored, by the area metric."""
        class_mask = np.zeros((8, 8), dtype=bool)
        class_mask[:4, :] = True
        scores, _agg = self._run({"method": "spine"}, None, class_mask)
        self.assertAlmostEqual(scores[7], 0.5, places=5)

    def test_tolerance_dilates_the_class_mask(self) -> None:
        """A centerline one pixel off the segmented stripe scores 0 exactly, and 1 with a 1 px tolerance."""
        class_mask = np.zeros((8, 8), dtype=bool)
        class_mask[3, :] = True  # one row above the spine at row 4

        strict, _agg = self._run({"method": "spine"}, self._spine_mask(), class_mask)
        self.assertEqual(strict[7], 0.0)

        tolerant, _agg = self._run({"method": "spine", "spine_match_tolerance_px": 1}, self._spine_mask(), class_mask)
        self.assertEqual(tolerant[7], 1.0)

    def test_area_methods_never_request_a_spine(self) -> None:
        """Classes left on ratio/cluster must not pay for centerline projection."""
        det = self._detector({"method": "ratio"})
        det.seg_model.get_class_mask.return_value = np.ones((8, 8), dtype=bool)
        rasterizer = MagicMock()
        rasterizer.has_object.return_value = True
        rasterizer.get_visibility_mask.return_value = np.ones((8, 8), dtype=bool)
        with patch(
            "checks.objects.single_view.overlap_detector.RdsDataLoader.get_objects_by_class",
            return_value={7: {"geometry": MagicMock(), "object_type": "LaneLine", "is_static": True}},
        ):
            det.process_class(
                "lane_line",
                MagicMock(),
                cast(dict[str, Any], {7: {}}),
                np.eye(4),
                MagicMock(),
                8,
                8,
                {},
                0,
                scene_rasterizer=rasterizer,
                is_static=True,
            )
        rasterizer.get_visible_spine_mask.assert_not_called()


class TestSpineConfiguredForPolylineClasses(unittest.TestCase):
    """The shipped config routes every polyline type through the spine metric and leaves the rest on area."""

    def test_default_config_methods(self) -> None:
        config = ConfigManager().load_config("cosmos_evaluator")
        overlap_cfg = config["av.objects_single_view"]["overlap_check"]
        for class_name in ("lane_line", "road_boundary", "wait_line"):
            self.assertEqual(overlap_cfg[class_name]["method"], "spine", class_name)
        # Crosswalk is a Surface, whose projected extent is real; it stays on an area metric.
        self.assertEqual(overlap_cfg["crosswalk"]["method"], "ratio")
        self.assertEqual(overlap_cfg["traffic_light"]["method"], "ratio")

    def test_the_spine_tuning_knobs_ship_disabled(self) -> None:
        """Both are mechanisms without a justified value: swept on clip bb4f394b, each raises the mean while
        either failing to convert zeros (tolerance) or costing more nonzero evidence than it gains (width). A
        value has to come from the perturbation KPI, which can measure whether a deleted line's score drops.
        Pinned so turning one on is a deliberate act with evidence behind it, not a silent default change."""
        overlap_cfg = ConfigManager().load_config("cosmos_evaluator")["av.objects_single_view"]["overlap_check"]
        for class_name in ("lane_line", "road_boundary", "wait_line"):
            self.assertEqual(overlap_cfg[class_name]["spine_match_tolerance_px"], 0, class_name)
            self.assertEqual(overlap_cfg[class_name]["spine_min_ribbon_width_px"], 0, class_name)


if __name__ == "__main__":
    unittest.main()
