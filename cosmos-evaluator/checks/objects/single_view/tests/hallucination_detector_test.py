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

"""Unit tests for hallucination_detector module."""

from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from checks.objects.single_view.hallucination_detector import HallucinationDetector, component_centerline_px


class TestHallucinationDetector(unittest.TestCase):
    """Test cases for HallucinationDetector class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Mock configuration
        self.config = {
            "enabled": True,
            "classes": {"vehicle": {"min_cluster_area": 100}, "pedestrian": {"min_cluster_area": 50}},
            "max_cluster_per_frame": 10,
        }

        # Mock seg_helper
        self.mock_seg_helper = MagicMock()

        # Mock logger
        self.mock_logger = MagicMock()

        # Create detector instance
        self.detector = HallucinationDetector(
            config=self.config, seg_helper=self.mock_seg_helper, logger=self.mock_logger
        )

    def test_init_valid_config(self) -> None:
        """Test HallucinationDetector initialization with valid config."""
        self.assertTrue(self.detector.enabled)
        self.assertEqual(self.detector.classes_cfg, self.config["classes"])
        self.assertEqual(self.detector.max_clusters_per_frame, 10)

    def test_init_missing_enabled_key(self) -> None:
        """Test HallucinationDetector initialization with missing enabled key."""
        config = {"classes": {}, "max_cluster_per_frame": 5}
        with self.assertRaises(KeyError) as context:
            HallucinationDetector(config, self.mock_seg_helper, self.mock_logger)
        self.assertIn("enabled", str(context.exception))

    def test_init_missing_classes_key(self) -> None:
        """Test HallucinationDetector initialization with missing classes key."""
        config = {"enabled": True, "max_cluster_per_frame": 5}
        with self.assertRaises(KeyError) as context:
            HallucinationDetector(config, self.mock_seg_helper, self.mock_logger)
        self.assertIn("classes", str(context.exception))

    def test_init_missing_max_cluster_key(self) -> None:
        """Test HallucinationDetector initialization with missing max_cluster_per_frame key."""
        config = {"enabled": True, "classes": {}}
        with self.assertRaises(KeyError) as context:
            HallucinationDetector(config, self.mock_seg_helper, self.mock_logger)
        self.assertIn("max_cluster_per_frame", str(context.exception))

    def test_init_invalid_classes_type(self) -> None:
        """Test HallucinationDetector initialization with invalid classes type."""
        config = {"enabled": True, "classes": "invalid", "max_cluster_per_frame": 5}
        with self.assertRaises(TypeError) as context:
            HallucinationDetector(config, self.mock_seg_helper, self.mock_logger)
        self.assertIn("must be a dict", str(context.exception))

    def test_detect_disabled(self) -> None:
        """Test detect method when detector is disabled."""
        self.detector.enabled = False
        result = self.detector.detect(
            frame_idx=0,
            resized_masks=None,
            frame_objects={},
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            image_width=1920,
            image_height=1080,
        )

        expected: dict[str, list] = {"Vehicle": [], "Pedestrian": []}
        self.assertEqual(result, expected)

    @patch("checks.objects.single_view.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_detect_no_masks(self, mock_cv2_stats: MagicMock) -> None:
        """Test detect method when segmentation yields an empty class mask (no class pixels)."""
        # get_class_mask is contractually -> np.ndarray; an all-False mask represents "no class pixels".
        self.mock_seg_helper.get_class_mask.return_value = np.zeros((1080, 1920), dtype=bool)

        # _build_gt_mask always runs connected components on the class mask; return a background-only result.
        mock_cv2_stats.return_value = (
            1,  # num components (just background)
            np.zeros((1080, 1920), dtype=np.int32),  # labels
            np.array([[0, 0, 0, 0, 0]]),  # stats (background)
            np.array([[0, 0]]),  # centroids (background)
        )

        result = self.detector.detect(
            frame_idx=0,
            resized_masks=MagicMock(),
            frame_objects={},
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            image_width=1920,
            image_height=1080,
        )

        expected: dict[str, list] = {"Vehicle": [], "Pedestrian": []}
        self.assertEqual(result, expected)

    @patch("checks.objects.single_view.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_detect_with_hallucinations(self, mock_cv2_stats: MagicMock) -> None:
        """Test detect method with hallucination detections."""
        # Setup mock segmentation masks
        vehicle_mask = np.zeros((1080, 1920), dtype=bool)
        vehicle_mask[100:200, 100:300] = True  # Create a vehicle mask region

        self.mock_seg_helper.get_class_mask.side_effect = lambda _masks, class_name: {
            "vehicle": vehicle_mask,
            "pedestrian": np.zeros((1080, 1920), dtype=bool),
        }[class_name]

        # Mock connected components for clustering
        # Simulate one connected component for the unmatched region
        mock_cv2_stats.return_value = (
            2,  # num components (including background)
            np.array([[0, 0, 0, 0, 0], [100, 100, 200, 100, 20000]]),  # labels (mock)
            np.array([[0, 0, 0, 0, 0], [100, 100, 200, 100, 20000]]),  # stats
            np.array([[0, 0], [200, 150]]),  # centroids
        )

        result = self.detector.detect(
            frame_idx=5,
            resized_masks=MagicMock(),
            frame_objects={},
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            image_width=1920,
            image_height=1080,
        )

        # Should detect one hallucination for vehicle class
        self.assertEqual(len(result["Vehicle"]), 1)
        self.assertEqual(len(result["Pedestrian"]), 0)

        detection = result["Vehicle"][0]
        self.assertEqual(detection["frame_idx"], 5)
        self.assertEqual(detection["bbox_xywh"], [100, 100, 200, 100])
        self.assertAlmostEqual(detection["mask_ratio"], 1.0, places=2)

    def test_min_cluster_area_argument_overrides_the_configured_area_for_one_call(self) -> None:
        """A per-call ``min_cluster_area`` applies to its classes only and leaves the configured areas unchanged."""
        mask = np.zeros((120, 160), dtype=bool)
        mask[10:20, 10:40] = True
        self.mock_seg_helper.get_class_mask.side_effect = lambda _masks, _class_name: mask

        def _areas(**kwargs: Any) -> dict[str, int]:
            with patch.object(self.detector, "_cluster_components", return_value=([], None)) as mock_cluster:
                self.detector.detect(
                    frame_idx=0,
                    resized_masks=MagicMock(),
                    frame_objects={},
                    camera_pose=np.eye(4),
                    camera_model=MagicMock(),
                    image_width=160,
                    image_height=120,
                    **kwargs,
                )
            # One clustering call per class, in configured class order (vehicle, pedestrian).
            return dict(zip(["vehicle", "pedestrian"], [c.kwargs["min_area"] for c in mock_cluster.call_args_list]))

        self.assertEqual(_areas(min_cluster_area={"vehicle": 300}), {"vehicle": 300, "pedestrian": 50})
        self.assertEqual(_areas(), {"vehicle": 100, "pedestrian": 50})
        self.assertEqual(self.detector.classes_cfg["vehicle"]["min_cluster_area"], 100)

    @patch("checks.objects.single_view.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_mask_debug_dir_dumps_intermediate_masks(self, mock_cv2_stats: MagicMock) -> None:
        """When mask_debug_dir is set, detect persists per-frame packed masks that unpack to the originals."""
        vehicle_mask = np.zeros((120, 160), dtype=bool)
        vehicle_mask[10:20, 10:40] = True
        self.mock_seg_helper.get_class_mask.side_effect = lambda _masks, class_name: {
            "vehicle": vehicle_mask,
            "pedestrian": None,
        }.get(class_name)
        mock_cv2_stats.return_value = (
            2,
            np.array([[0, 0, 0, 0, 0], [10, 10, 30, 10, 300]]),
            np.array([[0, 0, 0, 0, 0], [10, 10, 30, 10, 300]]),
            np.array([[0, 0], [25, 15]]),
        )
        with tempfile.TemporaryDirectory() as tmp:
            detector = HallucinationDetector(
                config=self.config, seg_helper=self.mock_seg_helper, logger=self.mock_logger, mask_debug_dir=Path(tmp)
            )
            detector.detect(
                frame_idx=7,
                resized_masks=MagicMock(),
                frame_objects={},
                camera_pose=np.eye(4),
                camera_model=MagicMock(),
                image_width=160,
                image_height=120,
            )
            path = Path(tmp) / "frame_0007.npz"
            self.assertTrue(path.exists())
            data = np.load(path, allow_pickle=False)
            h, w = int(data["shape"][0]), int(data["shape"][1])
            self.assertEqual((h, w), (120, 160))
            # Keys use RDS labels (class_name_to_rds_label), matching hallucination_detections: vehicle -> Vehicle.
            self.assertIn("Vehicle", [str(c) for c in data["classes"]])
            cls = np.unpackbits(data["cls__Vehicle"])[: h * w].reshape(h, w).astype(bool)
            np.testing.assert_array_equal(cls, vehicle_mask)
            # No frame objects -> empty GT -> unmatched equals the raw class mask; pedestrian (None) is absent.
            unm = np.unpackbits(data["unm__Vehicle"])[: h * w].reshape(h, w).astype(bool)
            np.testing.assert_array_equal(unm, vehicle_mask)
            gt = np.unpackbits(data["gt"])[: h * w].reshape(h, w).astype(bool)
            self.assertFalse(gt.any())
            self.assertNotIn("cls__Pedestrian", data.files)

    def test_mask_debug_dir_none_writes_nothing(self) -> None:
        """Default (mask_debug_dir=None) persists no artifacts."""
        with tempfile.TemporaryDirectory() as tmp:
            self.mock_seg_helper.get_class_mask.return_value = None
            self.detector.detect(
                frame_idx=0,
                resized_masks=MagicMock(),
                frame_objects={},
                camera_pose=np.eye(4),
                camera_model=MagicMock(),
                image_width=160,
                image_height=120,
            )
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_cluster_components_empty_mask(self) -> None:
        """Test _cluster_components with empty mask."""
        empty_mask = np.zeros((100, 100), dtype=bool)
        comps, labels = self.detector._cluster_components(empty_mask, min_area=50)
        self.assertEqual(comps, [])
        self.assertIsNone(labels)

    def test_cluster_components_none_mask(self) -> None:
        """Test _cluster_components with None mask."""
        comps, labels = self.detector._cluster_components(None, min_area=50)
        self.assertEqual(comps, [])
        self.assertIsNone(labels)

    @patch("checks.objects.single_view.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_cluster_components_with_components(self, mock_cv2_stats: MagicMock) -> None:
        """Test _cluster_components with valid components."""
        mask = np.ones((100, 100), dtype=bool)

        # Mock connected components output
        mock_cv2_stats.return_value = (
            3,  # num components (including background)
            np.zeros((100, 100)),  # labels (not used in this test)
            np.array(
                [
                    [0, 0, 0, 0, 0],  # background component
                    [10, 10, 20, 20, 400],  # first component (area 400)
                    [50, 50, 15, 15, 225],  # second component (area 225)
                ]
            ),
            np.array([[0, 0], [20, 20], [57.5, 57.5]]),  # centroids
        )

        comps, labels = self.detector._cluster_components(mask, min_area=200)

        # Should return both components (both above min_area=200)
        self.assertEqual(len(comps), 2)
        self.assertIsNotNone(labels)

        # Should be sorted by area descending, and carry the label that locates each component's pixels
        self.assertEqual((comps[0].label, comps[0].x, comps[0].y, comps[0].width, comps[0].height), (1, 10, 10, 20, 20))
        self.assertEqual(comps[0].area, 400)
        self.assertEqual((comps[1].label, comps[1].x, comps[1].y, comps[1].width, comps[1].height), (2, 50, 50, 15, 15))
        self.assertEqual(comps[1].area, 225)

    @patch("checks.objects.single_view.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_cluster_components_min_area_filter(self, mock_cv2_stats: MagicMock) -> None:
        """Test _cluster_components with min_area filtering."""
        mask = np.ones((100, 100), dtype=bool)

        # Mock connected components output
        mock_cv2_stats.return_value = (
            3,  # num components
            np.zeros((100, 100)),  # labels
            np.array(
                [
                    [0, 0, 0, 0, 0],  # background
                    [10, 10, 20, 20, 100],  # small component (area 100)
                    [50, 50, 15, 15, 500],  # large component (area 500)
                ]
            ),
            np.array([[0, 0], [20, 20], [57.5, 57.5]]),
        )

        comps, _labels = self.detector._cluster_components(mask, min_area=200)

        # Should only return large component
        self.assertEqual(len(comps), 1)
        self.assertEqual((comps[0].label, comps[0].x, comps[0].y, comps[0].width, comps[0].height), (2, 50, 50, 15, 15))
        self.assertEqual(comps[0].area, 500)

    def test_get_min_cluster_area_for_class_configured(self) -> None:
        """Test _get_min_cluster_area_for_class with configured class."""
        area = self.detector._get_min_cluster_area_for_class("vehicle")
        self.assertEqual(area, 100)

        area = self.detector._get_min_cluster_area_for_class("pedestrian")
        self.assertEqual(area, 50)

    def test_get_min_cluster_area_for_class_unconfigured(self) -> None:
        """Test _get_min_cluster_area_for_class with unconfigured class."""
        area = self.detector._get_min_cluster_area_for_class("unknown_class")
        self.assertEqual(area, 100)  # default value

    def test_get_min_cluster_area_for_class_invalid_config(self) -> None:
        """Test _get_min_cluster_area_for_class with invalid config."""
        # Test with invalid min_cluster_area value
        self.detector.classes_cfg["vehicle"] = {"min_cluster_area": "invalid"}
        area = self.detector._get_min_cluster_area_for_class("vehicle")
        self.assertEqual(area, 100)  # should fallback to default

    def test_is_in_ego_body_region_inside(self) -> None:
        """Test _is_in_ego_body_region for bbox inside ego body region."""
        # Bbox in bottom 25% of image
        image_width, image_height = 1920, 1080
        band_y0 = image_height - int(0.25 * image_height)  # 810 for 1080p

        # Small bbox fully inside ego body region
        result = self.detector._is_in_ego_body_region(
            x=960, y=band_y0 + 10, w=50, h=50, image_width=image_width, image_height=image_height
        )
        self.assertTrue(result)

    def test_is_in_ego_body_region_outside(self) -> None:
        """Test _is_in_ego_body_region for bbox outside ego body region."""
        # Bbox in top part of image, above the bottom 25% band
        result = self.detector._is_in_ego_body_region(x=960, y=100, w=50, h=50, image_width=1920, image_height=1080)
        self.assertFalse(result)

    def test_is_in_ego_body_region_partial_overlap(self) -> None:
        """Test _is_in_ego_body_region for bbox that partially overlaps the bottom band."""
        image_width, image_height = 1920, 1080
        band_y0 = image_height - int(0.25 * image_height)  # 810 for 1080p

        # Bbox spans from above band into band (e.g. y=780, h=100 -> y1=879, intersects [810,1079])
        result = self.detector._is_in_ego_body_region(
            x=960, y=band_y0 - 30, w=100, h=100, image_width=image_width, image_height=image_height
        )
        self.assertTrue(result)

    @patch("checks.objects.single_view.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_detect_with_scene_rasterizer(self, mock_cv2_stats: MagicMock) -> None:
        """Test detect method with pre-computed SceneRasterizer."""
        # Setup mock segmentation masks
        vehicle_mask = np.zeros((1080, 1920), dtype=bool)
        vehicle_mask[100:200, 100:300] = True  # Create a vehicle mask region

        self.mock_seg_helper.get_class_mask.side_effect = lambda masks, class_name: {
            "vehicle": vehicle_mask,
            "pedestrian": np.zeros((1080, 1920), dtype=bool),
        }[class_name]

        # Mock connected components for clustering
        mock_cv2_stats.return_value = (
            2,  # num components (including background)
            np.array([[0, 0, 0, 0, 0], [100, 100, 200, 100, 20000]]),  # labels (mock)
            np.array([[0, 0, 0, 0, 0], [100, 100, 200, 100, 20000]]),  # stats
            np.array([[0, 0], [200, 150]]),  # centroids
        )

        # Create a mock SceneRasterizer
        mock_rasterizer = MagicMock()
        mock_rasterizer.has_object.return_value = False  # No objects in rasterizer

        result = self.detector.detect(
            frame_idx=5,
            resized_masks=MagicMock(),
            frame_objects={},
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            image_width=1920,
            image_height=1080,
            scene_rasterizer=mock_rasterizer,
        )

        # Should still detect hallucinations (scene_rasterizer has no objects to provide masks for)
        self.assertEqual(len(result["Vehicle"]), 1)

    @patch("checks.objects.single_view.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_build_gt_mask_no_masks(self, mock_cv2_stats: MagicMock) -> None:
        """Test _build_gt_mask with no segmentation masks."""
        result = self.detector._build_gt_mask(
            resized_masks=None,
            frame_objects={},
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            image_width=1920,
            image_height=1080,
            class_name="vehicle",
        )

        expected = np.zeros((1080, 1920), dtype=bool)
        np.testing.assert_array_equal(result, expected)

    @patch("checks.objects.single_view.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_build_gt_mask_no_objects(self, mock_cv2_stats: MagicMock) -> None:
        """Test _build_gt_mask with no frame objects."""
        # Mock class mask
        class_mask = np.ones((1080, 1920), dtype=bool)
        self.mock_seg_helper.get_class_mask.return_value = class_mask

        # Mock connected components with proper return values
        mock_cv2_stats.return_value = (
            1,  # num components (just background)
            np.zeros((1080, 1920), dtype=np.int32),  # labels array with correct shape
            np.array([[0, 0, 0, 0, 0]]),  # stats for background only
            np.array([[0, 0]]),  # centroids for background only
        )

        result = self.detector._build_gt_mask(
            resized_masks=MagicMock(),
            frame_objects={},
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            image_width=1920,
            image_height=1080,
            class_name="vehicle",
        )

        expected = np.zeros((1080, 1920), dtype=bool)
        np.testing.assert_array_equal(result, expected)

    @patch("checks.objects.single_view.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_build_gt_mask_vehicle_class(self, mock_cv2_stats: MagicMock) -> None:
        """Test _build_gt_mask for vehicle class."""
        # Setup mock class mask and objects
        class_mask = np.ones((1080, 1920), dtype=bool)
        self.mock_seg_helper.get_class_mask.return_value = class_mask

        # Mock connected components
        labels = np.zeros((1080, 1920), dtype=np.int32)
        labels[100:300, 100:300] = 1  # component 1 region
        labels[500:700, 500:800] = 2  # component 2 region

        mock_cv2_stats.return_value = (
            3,  # num components
            labels,  # labels array with correct 2D shape
            np.array(
                [
                    [0, 0, 0, 0, 0],  # background
                    [100, 100, 200, 100, 20000],  # component 1
                    [500, 500, 300, 200, 60000],  # component 2
                ]
            ),
            np.array([[0, 0], [200, 150], [650, 600]]),
        )

        # Mock cuboid projection
        mock_cuboid = MagicMock()
        mock_projection = np.zeros((1080, 1920), dtype=bool)
        mock_projection[100:200, 100:300] = True  # Intersects with component 1
        mock_depth_mask = np.zeros((1080, 1920), dtype=np.float32)
        mock_depth_mask[100:200, 100:300] = 1.0
        mock_cuboid.get_projected_mask.return_value = mock_projection, mock_depth_mask

        # Mock frame objects
        frame_objects = {"track_1": {"object_type": "Car", "geometry": mock_cuboid}}

        result = self.detector._build_gt_mask(
            resized_masks=MagicMock(),
            frame_objects=frame_objects,
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            image_width=1920,
            image_height=1080,
            class_name="vehicle",
        )

        # Should have GT mask where component 1 is located
        self.assertTrue(result.any())
        mock_cuboid.get_projected_mask.assert_called_once()

    @patch("checks.objects.single_view.hallucination_detector.cv2.connectedComponentsWithStats")
    def test_build_gt_mask_non_vehicle_class(self, mock_cv2_stats: MagicMock) -> None:
        """Test _build_gt_mask for non-vehicle class (pedestrian)."""
        # Setup mock class mask and objects
        class_mask = np.ones((1080, 1920), dtype=bool)
        self.mock_seg_helper.get_class_mask.return_value = class_mask

        # Mock connected components
        labels = np.zeros((1080, 1920), dtype=np.int32)
        labels[100:150, 100:150] = 1  # component 1 region
        labels[200:240, 200:230] = 2  # component 2 region

        mock_cv2_stats.return_value = (
            3,  # num components
            labels,  # labels array with correct 2D shape
            np.array(
                [
                    [0, 0, 0, 0, 0],  # background
                    [100, 100, 50, 50, 2500],  # component 1
                    [200, 200, 30, 40, 1200],  # component 2
                ]
            ),
            np.array([[0, 0], [125, 125], [215, 220]]),
        )

        # Mock cuboid projection that intersects both components
        mock_cuboid = MagicMock()
        mock_projection = np.zeros((1080, 1920), dtype=bool)
        mock_projection[100:250, 100:250] = True  # Intersects with both components
        mock_depth_mask = np.zeros((1080, 1920), dtype=np.float32)
        mock_depth_mask[100:250, 100:250] = 1.0
        mock_cuboid.get_projected_mask.return_value = mock_projection, mock_depth_mask

        # Mock frame objects
        frame_objects = {"track_1": {"object_type": "Pedestrian", "geometry": mock_cuboid}}

        result = self.detector._build_gt_mask(
            resized_masks=MagicMock(),
            frame_objects=frame_objects,
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            image_width=1920,
            image_height=1080,
            class_name="pedestrian",
        )

        # Should have GT mask as expanded bounding box around intersecting components
        self.assertTrue(result.any())
        mock_cuboid.get_projected_mask.assert_called_once()


class _PinholeStub:
    """Minimal pinhole camera exposing the ``pixel2ray_np`` contract (rays, valid) the real models use."""

    def __init__(self, focal_px: float = 20.0, cx: float = 20.0, cy: float = 20.0) -> None:
        self.focal_px, self.cx, self.cy = focal_px, cx, cy

    def pixel2ray_np(self, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pixels = np.asarray(pixels, dtype=float)
        rays = np.stack(
            [
                (pixels[:, 0] - self.cx) / self.focal_px,
                (pixels[:, 1] - self.cy) / self.focal_px,
                np.ones(len(pixels)),
            ],
            axis=1,
        )
        return rays, np.ones(len(pixels), dtype=bool)


class TestPolylineGtPolicy(unittest.TestCase):
    """GT support and the ego-body filter for the ground-plane polyline classes."""

    def setUp(self) -> None:
        self.logger = MagicMock()
        self.height, self.width = 40, 40
        # Forward-facing camera 1.5 m above the ground in an FLU world (X forward, Y left, Z up).
        self.camera_pose = np.array(
            [[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.5], [0.0, 0.0, 0.0, 1.0]]
        )
        self.camera_model = _PinholeStub()

    def _detector(self, class_name: str) -> HallucinationDetector:
        """Build a detector for one class, keeping the seg-helper mock on ``self.seg_helper``.

        The mock is held here rather than reached through ``detector.seg_helper``, which is typed as the
        ``_SegHelperLike`` protocol and so has no ``return_value`` to set.
        """
        self.seg_helper = MagicMock()
        return HallucinationDetector(
            config={
                "enabled": True,
                "classes": {class_name: {"min_cluster_area": 1}},
                "max_cluster_per_frame": 50,
            },
            seg_helper=self.seg_helper,
            logger=self.logger,
        )

    def _diagonal_class_mask(self) -> np.ndarray:
        """A thin diagonal stripe, i.e. how a lane line actually appears in image space."""
        mask = np.zeros((self.height, self.width), dtype=bool)
        for i in range(5, 35):
            mask[i, i] = True
            mask[i, i + 1] = True
        return mask

    def _gt_mask_for(self, class_name: str, class_mask: np.ndarray) -> np.ndarray:
        """Run _build_gt_mask with the projection covering the whole stripe."""
        detector = self._detector(class_name)
        self.seg_helper.get_class_mask.return_value = class_mask
        rasterizer = MagicMock()
        rasterizer.has_object.return_value = True
        rasterizer.get_visibility_mask.return_value = class_mask.copy()
        with patch(
            "checks.objects.single_view.hallucination_detector.RdsDataLoader.get_objects_by_class",
            return_value={"7": {"geometry": MagicMock(), "object_type": "LaneLine"}},
        ):
            return detector._build_gt_mask(
                resized_masks=MagicMock(),
                frame_objects={"7": {}},
                camera_pose=np.eye(4),
                camera_model=MagicMock(),
                image_width=self.width,
                image_height=self.height,
                class_name=class_name,
                scene_rasterizer=rasterizer,
            )

    def test_polyline_gt_is_the_component_not_its_bounding_box(self) -> None:
        """A diagonal stripe's AABB is a slab of the image; since the per-class masks are OR'd into one shared
        mask, that slab would suppress every other class's detections inside it."""
        stripe = self._diagonal_class_mask()
        for class_name in ("lane_line", "road_boundary"):
            gt = self._gt_mask_for(class_name, stripe)
            self.assertEqual(int(gt.sum()), int(stripe.sum()), class_name)
            self.assertFalse((gt & ~stripe).any(), f"{class_name} covered pixels outside the component")

    def test_a_compact_class_still_uses_the_bounding_box(self) -> None:
        """The change is scoped: traffic lights keep the box that fills the gaps between their bulb blobs."""
        stripe = self._diagonal_class_mask()
        gt = self._gt_mask_for("traffic_light", stripe)
        self.assertGreater(int(gt.sum()), int(stripe.sum()))

    def _detect_bottom_cluster(self, class_name: str) -> int:
        """Detections for one blob sitting in the bottom quarter of the frame, with no GT support."""
        detector = self._detector(class_name)
        blob = np.zeros((self.height, self.width), dtype=bool)
        blob[34:39, 10:20] = True  # bottom 25% is rows 30..39
        self.seg_helper.get_class_mask.return_value = blob
        with patch(
            "checks.objects.single_view.hallucination_detector.RdsDataLoader.get_objects_by_class",
            return_value={},
        ):
            result = detector.detect(
                frame_idx=0,
                resized_masks=MagicMock(),
                frame_objects={},
                camera_pose=self.camera_pose,
                camera_model=self.camera_model,
                image_width=self.width,
                image_height=self.height,
            )
        self.last_detections = result
        return sum(len(v) for v in result.values())

    def test_ego_body_filter_skipped_for_ground_plane_classes(self) -> None:
        """A lane line runs from under the bumper to the vanishing point, so its nearest and best-resolved part
        is always inside the bottom band. Filtering on that discards the detection entirely."""
        for class_name in ("lane_line", "road_boundary"):
            self.assertEqual(self._detect_bottom_cluster(class_name), 1, class_name)

    def test_ego_body_filter_still_applies_to_other_classes(self) -> None:
        """Scoped, not removed: the dynamic path reads raw segmentation masks and still needs it."""
        for class_name in ("vehicle", "traffic_sign"):
            self.assertEqual(self._detect_bottom_cluster(class_name), 0, class_name)

    def test_ground_plane_detections_carry_a_spine_and_world_position(self) -> None:
        """The two fields the overlay and the world-space tracker consume."""
        self.assertEqual(self._detect_bottom_cluster("lane_line"), 1)
        detection = self.last_detections["LaneLine"][0]
        self.assertGreaterEqual(len(detection["spine_px"]), 2)
        self.assertGreaterEqual(len(detection["ground_xy"]), 1)
        for x, y in detection["ground_xy"]:
            self.assertIsInstance(x, float)
            self.assertIsInstance(y, float)

    def test_compact_classes_carry_neither(self) -> None:
        """Unchanged output for everything that was already shipping."""
        detector = self._detector("traffic_light")
        blob = np.zeros((self.height, self.width), dtype=bool)
        blob[10:15, 10:20] = True
        self.seg_helper.get_class_mask.return_value = blob
        with patch(
            "checks.objects.single_view.hallucination_detector.RdsDataLoader.get_objects_by_class",
            return_value={},
        ):
            result = detector.detect(
                frame_idx=0,
                resized_masks=MagicMock(),
                frame_objects={},
                camera_pose=self.camera_pose,
                camera_model=self.camera_model,
                image_width=self.width,
                image_height=self.height,
            )
        detection = result["TrafficLight"][0]
        self.assertNotIn("spine_px", detection)
        self.assertNotIn("ground_xy", detection)

    def test_a_camera_that_cannot_unproject_still_yields_the_detection(self) -> None:
        """Losing the world position must not lose the detection."""
        detector = self._detector("lane_line")
        blob = np.zeros((self.height, self.width), dtype=bool)
        blob[34:39, 10:20] = True
        self.seg_helper.get_class_mask.return_value = blob
        broken = MagicMock()
        broken.pixel2ray_np.side_effect = RuntimeError("no inverse")
        with patch(
            "checks.objects.single_view.hallucination_detector.RdsDataLoader.get_objects_by_class",
            return_value={},
        ):
            result = detector.detect(
                frame_idx=0,
                resized_masks=MagicMock(),
                frame_objects={},
                camera_pose=self.camera_pose,
                camera_model=broken,
                image_width=self.width,
                image_height=self.height,
            )
        self.assertEqual(len(result["LaneLine"]), 1)
        self.assertNotIn("ground_xy", result["LaneLine"][0])
        self.assertIn("spine_px", result["LaneLine"][0])


class TestComponentCenterline(unittest.TestCase):
    """Image-space centerline extraction for the polyline classes."""

    def test_follows_a_diagonal(self) -> None:
        mask = np.zeros((40, 40), dtype=bool)
        for i in range(5, 35):
            mask[i, i] = True
            mask[i, i + 1] = True
        spine = component_centerline_px(mask)
        self.assertGreaterEqual(len(spine), 2)
        for x, y in spine:
            self.assertLessEqual(abs(x - y), 1, f"({x},{y}) is off the diagonal")

    def test_follows_a_curve_a_straight_fit_would_miss(self) -> None:
        """The reason for scanline medians over PCA or an oriented box: a real lane line bends."""
        mask = np.zeros((60, 60), dtype=bool)
        expected = {}
        for row in range(5, 55):
            col = int(30 + 20 * np.sin((row - 5) / 50 * np.pi))
            mask[row, col - 1 : col + 2] = True
            expected[row] = col
        for x, y in component_centerline_px(mask, max_points=64):
            self.assertLessEqual(abs(x - expected[y]), 1)

    def test_orders_along_the_longer_axis(self) -> None:
        """A wait line crosses the view, so it is parameterized by column rather than row."""
        mask = np.zeros((40, 60), dtype=bool)
        mask[20:22, 5:55] = True
        spine = component_centerline_px(mask)
        xs = [p[0] for p in spine]
        self.assertEqual(xs, sorted(xs))
        self.assertGreater(xs[-1] - xs[0], 30)

    def test_respects_the_point_cap(self) -> None:
        mask = np.zeros((200, 200), dtype=bool)
        for i in range(5, 195):
            mask[i, i] = True
        self.assertLessEqual(len(component_centerline_px(mask, max_points=8)), 8)

    def test_median_ignores_a_stray_pixel(self) -> None:
        mask = np.zeros((30, 30), dtype=bool)
        mask[10:20, 14:17] = True
        mask[15, 28] = True  # outlier on one scanline
        spine = {y: x for x, y in component_centerline_px(mask, max_points=64)}
        self.assertLessEqual(abs(spine[15] - 15), 1)

    def test_empty_mask(self) -> None:
        self.assertEqual(component_centerline_px(np.zeros((10, 10), dtype=bool)), [])


if __name__ == "__main__":
    unittest.main()
