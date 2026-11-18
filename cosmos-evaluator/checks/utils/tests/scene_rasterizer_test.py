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

"""Unit tests for scene_rasterizer module."""

import logging
from typing import Any
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.testing import assert_array_equal
from third_party.cosmos_drive_dreams_toolkits.utils.camera.base import CameraBase

from checks.utils.rds_geometry import RDSGeometry
from checks.utils.scene_rasterizer import SceneRasterizer


class MockGeometry(RDSGeometry):
    """Mock geometry for testing with configurable mask and depth."""

    def __init__(self, mask: np.ndarray, depth_mask: np.ndarray):
        """Initialize mock geometry with predefined masks."""
        self._mask = mask
        self._depth_mask = depth_mask

    def get_projected_mask(
        self,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._mask.copy(), self._depth_mask.copy()


class TestSceneRasterizer(unittest.TestCase):
    """Test cases for SceneRasterizer class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.image_width = 100
        self.image_height = 80
        self.camera_pose = np.eye(4)
        self.camera_model = MagicMock()
        self.logger = logging.getLogger("test")

    def _create_rasterizer(self, objects: dict[str, dict[str, Any]], depth_tolerance: float = 0.05) -> SceneRasterizer:
        """Helper to create a SceneRasterizer with standard parameters."""
        return SceneRasterizer(
            objects,
            self.camera_pose,
            self.camera_model,
            self.image_width,
            self.image_height,
            depth_tolerance=depth_tolerance,
            logger=self.logger,
        )

    def test_init_empty_objects(self) -> None:
        """Test initialization with empty objects dict."""
        rasterizer = self._create_rasterizer({})
        self.assertEqual(len(rasterizer.objects), 0)

    def test_init_with_objects(self) -> None:
        """Test initialization with objects."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        objects = {
            "obj1": {"geometry": MockGeometry(mask, depth)},
            "obj2": {"geometry": MockGeometry(mask, depth)},
        }
        rasterizer = self._create_rasterizer(objects)
        self.assertEqual(len(rasterizer.objects), 2)
        self.assertTrue(rasterizer.has_object("obj1"))
        self.assertTrue(rasterizer.has_object("obj2"))

    def test_image_dimensions_properties(self) -> None:
        """Test that image dimensions are accessible via properties."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        objects = {"obj1": {"geometry": MockGeometry(mask, depth)}}
        rasterizer = self._create_rasterizer(objects)
        self.assertEqual(rasterizer.image_width, self.image_width)
        self.assertEqual(rasterizer.image_height, self.image_height)

    def test_single_object_no_occlusion(self) -> None:
        """Test projection of a single object (no occlusion possible)."""
        # Create a mask with some pixels set (24x24, above min_projected_size)
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask[10:34, 10:34] = True

        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth[10:34, 10:34] = 5.0

        objects = {"obj1": {"geometry": MockGeometry(mask, depth)}}
        rasterizer = self._create_rasterizer(objects)

        # Visibility should equal projected mask (no occlusion)
        vis_mask = rasterizer.get_visibility_mask("obj1")
        assert_array_equal(vis_mask, mask)

        # Check visibility ratio
        self.assertEqual(rasterizer.get_visibility_ratio("obj1"), 1.0)

    def test_bottom_occluded_frac_fully_visible(self) -> None:
        """A fully-visible object has its projected bottom visible -> bottom_occluded_frac == 0."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask[10:50, 20:40] = True
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth[10:50, 20:40] = 10.0
        rasterizer = self._create_rasterizer({"obj": {"geometry": MockGeometry(mask, depth)}})
        self.assertEqual(rasterizer.bottom_occluded_frac("obj"), 0.0)

    def test_bottom_occluded_frac_top_only(self) -> None:
        """A nearer object occluding the bottom half lifts the visible bottom -> ~0.5 (top-only)."""
        far = np.zeros((self.image_height, self.image_width), dtype=bool)
        far[10:50, 20:40] = True  # projected rows 10..49 (height 40)
        far_depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        far_depth[10:50, 20:40] = 10.0
        near = np.zeros((self.image_height, self.image_width), dtype=bool)
        near[30:50, 20:40] = True  # nearer; occludes far's bottom half (rows 30..49)
        near_depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        near_depth[30:50, 20:40] = 5.0
        rasterizer = self._create_rasterizer(
            {"far": {"geometry": MockGeometry(far, far_depth)}, "near": {"geometry": MockGeometry(near, near_depth)}}
        )
        # Only rows 10..29 of `far` stay visible -> visible bottom 29, (49 - 29) / 40 = 0.5.
        self.assertAlmostEqual(rasterizer.bottom_occluded_frac("far"), 0.5, places=2)
        self.assertEqual(rasterizer.bottom_occluded_frac("near"), 0.0)

    def test_bottom_occluded_frac_fully_occluded(self) -> None:
        """A fully-occluded object (no visible pixels) -> bottom_occluded_frac == 1.0."""
        far = np.zeros((self.image_height, self.image_width), dtype=bool)
        far[10:50, 20:40] = True
        far_depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        far_depth[10:50, 20:40] = 10.0
        near = np.zeros((self.image_height, self.image_width), dtype=bool)
        near[10:50, 20:40] = True  # nearer; fully covers `far`
        near_depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        near_depth[10:50, 20:40] = 5.0
        rasterizer = self._create_rasterizer(
            {"far": {"geometry": MockGeometry(far, far_depth)}, "near": {"geometry": MockGeometry(near, near_depth)}}
        )
        self.assertEqual(rasterizer.bottom_occluded_frac("far"), 1.0)

    def test_two_objects_no_overlap(self) -> None:
        """Test two non-overlapping objects."""
        mask1 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask1[10:34, 10:34] = True
        depth1 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth1[10:34, 10:34] = 5.0

        mask2 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask2[40:64, 50:80] = True  # Non-overlapping region
        depth2 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth2[40:64, 50:80] = 10.0

        objects = {
            "obj1": {"geometry": MockGeometry(mask1, depth1)},
            "obj2": {"geometry": MockGeometry(mask2, depth2)},
        }
        rasterizer = self._create_rasterizer(objects)

        # Both should be fully visible
        assert_array_equal(rasterizer.get_visibility_mask("obj1"), mask1)
        assert_array_equal(rasterizer.get_visibility_mask("obj2"), mask2)
        self.assertEqual(rasterizer.get_visibility_ratio("obj1"), 1.0)
        self.assertEqual(rasterizer.get_visibility_ratio("obj2"), 1.0)

    def test_two_objects_full_occlusion(self) -> None:
        """Test two overlapping objects where one fully occludes the other."""
        # Both objects cover the same region
        mask1 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask1[20:44, 20:60] = True
        depth1 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth1[20:44, 20:60] = 5.0  # Closer

        mask2 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask2[20:44, 20:60] = True  # Same region
        depth2 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth2[20:44, 20:60] = 10.0  # Further away

        objects = {
            "obj1": {"geometry": MockGeometry(mask1, depth1)},
            "obj2": {"geometry": MockGeometry(mask2, depth2)},
        }
        rasterizer = self._create_rasterizer(objects)

        # obj1 should be fully visible (closer)
        assert_array_equal(rasterizer.get_visibility_mask("obj1"), mask1)
        self.assertEqual(rasterizer.get_visibility_ratio("obj1"), 1.0)

        # obj2 should be fully occluded (further away)
        expected_obj2_vis = np.zeros((self.image_height, self.image_width), dtype=bool)
        assert_array_equal(rasterizer.get_visibility_mask("obj2"), expected_obj2_vis)
        self.assertEqual(rasterizer.get_visibility_ratio("obj2"), 0.0)

    def test_two_objects_partial_occlusion(self) -> None:
        """Test two overlapping objects with partial occlusion."""
        # obj1 covers region [20:44, 20:60]
        mask1 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask1[20:44, 20:60] = True
        depth1 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth1[20:44, 20:60] = 5.0

        # obj2 covers region [30:54, 40:80] - partial overlap at [30:44, 40:60]
        mask2 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask2[30:54, 40:80] = True
        depth2 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth2[30:54, 40:80] = 10.0  # Further away

        objects = {
            "obj1": {"geometry": MockGeometry(mask1, depth1)},
            "obj2": {"geometry": MockGeometry(mask2, depth2)},
        }
        rasterizer = self._create_rasterizer(objects)

        # obj1 should be fully visible
        assert_array_equal(rasterizer.get_visibility_mask("obj1"), mask1)
        self.assertEqual(rasterizer.get_visibility_ratio("obj1"), 1.0)

        # obj2 should be partially visible (overlap region occluded)
        vis2 = rasterizer.get_visibility_mask("obj2")
        # The overlap region [30:44, 40:60] should be occluded
        overlap_region = mask1 & mask2
        expected_vis2 = mask2 & ~overlap_region
        assert_array_equal(vis2, expected_vis2)

        # Visibility ratio should be < 1.0
        self.assertLess(rasterizer.get_visibility_ratio("obj2"), 1.0)
        self.assertGreater(rasterizer.get_visibility_ratio("obj2"), 0.0)

    def test_depth_tolerance(self) -> None:
        """Test that depth tolerance allows nearly-equal depths to be visible."""
        # Two objects at nearly the same depth
        mask1 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask1[20:44, 20:60] = True
        depth1 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth1[20:44, 20:60] = 5.0

        mask2 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask2[20:44, 20:60] = True
        depth2 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth2[20:44, 20:60] = 5.02  # Within default tolerance of 0.05

        objects = {
            "obj1": {"geometry": MockGeometry(mask1, depth1)},
            "obj2": {"geometry": MockGeometry(mask2, depth2)},
        }
        rasterizer = self._create_rasterizer(objects, depth_tolerance=0.05)

        # Both should be visible due to tolerance
        assert_array_equal(rasterizer.get_visibility_mask("obj1"), mask1)
        assert_array_equal(rasterizer.get_visibility_mask("obj2"), mask2)

    def test_depth_tolerance_exceeded(self) -> None:
        """Test that objects beyond depth tolerance are occluded."""
        mask1 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask1[20:44, 20:60] = True
        depth1 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth1[20:44, 20:60] = 5.0

        mask2 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask2[20:44, 20:60] = True
        depth2 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth2[20:44, 20:60] = 5.1  # Beyond tolerance of 0.05

        objects = {
            "obj1": {"geometry": MockGeometry(mask1, depth1)},
            "obj2": {"geometry": MockGeometry(mask2, depth2)},
        }
        rasterizer = self._create_rasterizer(objects, depth_tolerance=0.05)

        # obj1 visible, obj2 occluded
        assert_array_equal(rasterizer.get_visibility_mask("obj1"), mask1)
        self.assertEqual(rasterizer.get_visibility_ratio("obj2"), 0.0)

    def test_object_without_geometry(self) -> None:
        """Test handling of objects without geometry key."""
        objects = {
            "obj_no_geom": {"other_key": "value"},  # No geometry key
        }
        rasterizer = self._create_rasterizer(objects)

        # Should create empty masks for objects without geometry
        vis_mask = rasterizer.get_visibility_mask("obj_no_geom")
        self.assertEqual(vis_mask.shape, (self.image_height, self.image_width))
        self.assertFalse(np.any(vis_mask))

    def test_get_all_visibility_masks(self) -> None:
        """Test getting all visibility masks at once."""
        mask1 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask1[10:34, 10:34] = True
        depth1 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth1[10:34, 10:34] = 5.0

        mask2 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask2[40:64, 50:80] = True
        depth2 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth2[40:64, 50:80] = 10.0

        objects = {
            "obj1": {"geometry": MockGeometry(mask1, depth1)},
            "obj2": {"geometry": MockGeometry(mask2, depth2)},
        }
        rasterizer = self._create_rasterizer(objects)

        all_masks = rasterizer.get_all_visibility_masks()
        self.assertIn("obj1", all_masks)
        self.assertIn("obj2", all_masks)
        assert_array_equal(all_masks["obj1"], mask1)
        assert_array_equal(all_masks["obj2"], mask2)

    def test_get_scene_depth_buffer(self) -> None:
        """Test getting the composited scene depth buffer."""
        mask1 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask1[20:40, 20:60] = True
        depth1 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth1[20:40, 20:60] = 5.0

        mask2 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask2[30:50, 40:80] = True
        depth2 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth2[30:50, 40:80] = 10.0

        objects = {
            "obj1": {"geometry": MockGeometry(mask1, depth1)},
            "obj2": {"geometry": MockGeometry(mask2, depth2)},
        }
        rasterizer = self._create_rasterizer(objects)

        depth_buffer = rasterizer.get_scene_depth_buffer()

        # Check that minimum depth is used at each pixel
        # obj1-only region [20:30, 20:40] should have depth 5.0
        self.assertTrue(np.all(depth_buffer[20:30, 20:40] == 5.0))

        # obj2-only region [40:50, 60:80] should have depth 10.0
        self.assertTrue(np.all(depth_buffer[40:50, 60:80] == 10.0))

        # Overlap region [30:40, 40:60] should have depth 5.0 (minimum)
        self.assertTrue(np.all(depth_buffer[30:40, 40:60] == 5.0))

        # Uncovered region should be inf
        self.assertTrue(np.isinf(depth_buffer[0, 0]))

    def test_get_visible_pixel_count(self) -> None:
        """Test getting visible pixel count."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask[10:34, 10:34] = True  # 24x24 = 576 pixels
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth[10:34, 10:34] = 5.0

        objects = {"obj1": {"geometry": MockGeometry(mask, depth)}}
        rasterizer = self._create_rasterizer(objects)

        self.assertEqual(rasterizer.get_visible_pixel_count("obj1"), 576)

    def test_invalid_object_id(self) -> None:
        """Test error handling for invalid object IDs."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        objects = {"obj1": {"geometry": MockGeometry(mask, depth)}}
        rasterizer = self._create_rasterizer(objects)

        with self.assertRaises(KeyError):
            rasterizer.get_visibility_mask("nonexistent")

        with self.assertRaises(KeyError):
            rasterizer.get_projected_mask("nonexistent")

        with self.assertRaises(KeyError):
            rasterizer.get_depth_mask("nonexistent")

    def test_empty_mask_visibility_ratio(self) -> None:
        """Test visibility ratio for object with no projected pixels."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)

        objects = {"obj1": {"geometry": MockGeometry(mask, depth)}}
        rasterizer = self._create_rasterizer(objects)

        # Visibility ratio should be 0.0 when no pixels are projected
        self.assertEqual(rasterizer.get_visibility_ratio("obj1"), 0.0)

    def test_three_objects_layered_occlusion(self) -> None:
        """Test three objects at different depths with layered occlusion."""
        # All three cover the same region
        mask1 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask1[20:50, 20:70] = True
        depth1 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth1[20:50, 20:70] = 5.0  # Front

        mask2 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask2[20:50, 20:70] = True
        depth2 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth2[20:50, 20:70] = 10.0  # Middle

        mask3 = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask3[20:50, 20:70] = True
        depth3 = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth3[20:50, 20:70] = 15.0  # Back

        objects = {
            "front": {"geometry": MockGeometry(mask1, depth1)},
            "middle": {"geometry": MockGeometry(mask2, depth2)},
            "back": {"geometry": MockGeometry(mask3, depth3)},
        }
        rasterizer = self._create_rasterizer(objects)

        # Only front object should be visible
        self.assertEqual(rasterizer.get_visibility_ratio("front"), 1.0)
        self.assertEqual(rasterizer.get_visibility_ratio("middle"), 0.0)
        self.assertEqual(rasterizer.get_visibility_ratio("back"), 0.0)


class TestProjectionMetadata(unittest.TestCase):
    """Tests for get_projection_metadata and small-projection handling."""

    def setUp(self) -> None:
        self.image_width = 100
        self.image_height = 80
        self.camera_pose = np.eye(4)
        self.camera_model = MagicMock()
        self.logger = logging.getLogger("test")

    def _create_rasterizer(self, objects: dict[str, dict[str, Any]]) -> SceneRasterizer:
        return SceneRasterizer(
            objects,
            self.camera_pose,
            self.camera_model,
            self.image_width,
            self.image_height,
            logger=self.logger,
        )

    def test_small_projection_object_is_scored(self) -> None:
        """Small-projection objects (previously filtered) now have non-empty visibility masks."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask[10:15, 10:16] = True  # 5h x 6w: would have been filtered under old threshold=16
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth[10:15, 10:16] = 5.0

        rasterizer = self._create_rasterizer({"obj1": {"geometry": MockGeometry(mask, depth)}})
        assert_array_equal(rasterizer.get_visibility_mask("obj1"), mask)

    def test_projection_metadata_reports_correct_bbox(self) -> None:
        """get_projection_metadata returns the exact visibility bbox dimensions and pixel count."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask[20:30, 40:55] = True  # 10h x 15w = 150px
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth[20:30, 40:55] = 5.0

        rasterizer = self._create_rasterizer({"obj1": {"geometry": MockGeometry(mask, depth)}})
        meta = rasterizer.get_projection_metadata("obj1")
        self.assertEqual(meta["bbox_width"], 15)
        self.assertEqual(meta["bbox_height"], 10)
        self.assertEqual(meta["visible_area_px"], 150)

    def test_projection_metadata_zero_for_fully_occluded_object(self) -> None:
        """An object fully occluded by a closer object has zero-area projection metadata."""
        mask_front = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask_front[20:50, 20:50] = True
        depth_front = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth_front[20:50, 20:50] = 1.0

        mask_back = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask_back[25:45, 25:45] = True  # entirely inside front
        depth_back = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth_back[25:45, 25:45] = 10.0

        rasterizer = self._create_rasterizer(
            {
                "front": {"geometry": MockGeometry(mask_front, depth_front)},
                "back": {"geometry": MockGeometry(mask_back, depth_back)},
            }
        )
        meta = rasterizer.get_projection_metadata("back")
        self.assertEqual(meta["bbox_width"], 0)
        self.assertEqual(meta["bbox_height"], 0)
        self.assertEqual(meta["visible_area_px"], 0)

    def test_projection_metadata_for_object_with_no_geometry(self) -> None:
        """Objects without valid geometry still have a metadata entry with zero values."""
        rasterizer = self._create_rasterizer({"obj_noeg": {"geometry": None}})
        meta = rasterizer.get_projection_metadata("obj_noeg")
        self.assertEqual(meta, {"bbox_width": 0, "bbox_height": 0, "visible_area_px": 0})

    def test_projection_metadata_key_error_for_unknown_object(self) -> None:
        """Querying an unknown object id raises KeyError."""
        rasterizer = self._create_rasterizer({})
        with self.assertRaises(KeyError):
            rasterizer.get_projection_metadata("does_not_exist")


class TruncatableMockGeometry(MockGeometry):
    """MockGeometry that reports a caller-supplied in-frame fraction and counts how often it is asked."""

    def __init__(self, mask: np.ndarray, depth_mask: np.ndarray, fraction: float) -> None:
        """Initialize with predefined masks and the in-frame fraction to report."""
        super().__init__(mask, depth_mask)
        self._fraction = fraction
        self.in_frame_fraction_calls = 0

    def in_frame_fraction(
        self,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
    ) -> float:
        self.in_frame_fraction_calls += 1
        return self._fraction


class TestInFrameFraction(unittest.TestCase):
    """Tests for SceneRasterizer.in_frame_fraction — the one occlusion-free accessor on the class."""

    def setUp(self) -> None:
        self.image_width = 100
        self.image_height = 80
        self.camera_pose = np.eye(4)
        self.camera_model = MagicMock()
        self.logger = logging.getLogger("test")

    def _create_rasterizer(self, objects: dict[str, dict[str, Any]]) -> SceneRasterizer:
        return SceneRasterizer(
            objects,
            self.camera_pose,
            self.camera_model,
            self.image_width,
            self.image_height,
            logger=self.logger,
        )

    def _make_geometry(self, rows: slice, cols: slice, depth: float, fraction: float) -> TruncatableMockGeometry:
        """Build a TruncatableMockGeometry covering ``rows`` x ``cols`` at a constant ``depth``."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask[rows, cols] = True
        depth_mask = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth_mask[rows, cols] = depth
        return TruncatableMockGeometry(mask, depth_mask, fraction)

    def test_delegates_to_geometry(self) -> None:
        """The rasterizer reports whatever the object's own geometry computes."""
        geom = self._make_geometry(slice(20, 40), slice(20, 60), 5.0, 0.35)
        rasterizer = self._create_rasterizer({"obj1": {"geometry": geom}})
        self.assertAlmostEqual(rasterizer.in_frame_fraction("obj1"), 0.35)

    def test_ignores_occlusion(self) -> None:
        """Occlusion must NOT influence the in-frame fraction: it answers 'does this cuboid cross the image
        edge?', not 'how much of it can be seen?'. A fully occluded object with zero visible pixels still
        reports its geometric fraction, so a traffic light hidden behind a truck is not mistaken for a
        frame-edge-truncated one."""
        front = self._make_geometry(slice(20, 50), slice(20, 50), 1.0, 1.0)
        back = self._make_geometry(slice(25, 45), slice(25, 45), 10.0, 1.0)
        rasterizer = self._create_rasterizer({"front": {"geometry": front}, "back": {"geometry": back}})

        # Sanity: "back" really is fully occluded by the depth buffer.
        self.assertEqual(rasterizer.get_visibility_ratio("back"), 0.0)
        self.assertEqual(rasterizer.get_projection_metadata("back")["visible_area_px"], 0)
        # ...yet it is fully in-frame.
        self.assertEqual(rasterizer.in_frame_fraction("back"), 1.0)

    def test_computed_lazily_and_memoized(self) -> None:
        """Never computed during construction (callers that do not gate pay nothing), and computed exactly
        once thereafter so the correspondence and traffic-light-state gates share one projection."""
        geom = self._make_geometry(slice(20, 40), slice(20, 60), 5.0, 0.5)
        rasterizer = self._create_rasterizer({"obj1": {"geometry": geom}})
        self.assertEqual(geom.in_frame_fraction_calls, 0)

        for _ in range(3):
            self.assertAlmostEqual(rasterizer.in_frame_fraction("obj1"), 0.5)
        self.assertEqual(geom.in_frame_fraction_calls, 1)

    def test_memoizes_zero_fraction(self) -> None:
        """A 0.0 result is cached like any other (a falsy value must not defeat the memo)."""
        geom = self._make_geometry(slice(20, 40), slice(20, 60), 5.0, 0.0)
        rasterizer = self._create_rasterizer({"obj1": {"geometry": geom}})
        self.assertEqual(rasterizer.in_frame_fraction("obj1"), 0.0)
        self.assertEqual(rasterizer.in_frame_fraction("obj1"), 0.0)
        self.assertEqual(geom.in_frame_fraction_calls, 1)

    def test_geometry_without_override_is_never_gated(self) -> None:
        """Geometries that inherit the RDSGeometry default (polylines, surfaces) report 1.0, so a
        min_in_frame_fraction configured for such a class can never gate."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask[10:20, 10:20] = True
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth[10:20, 10:20] = 5.0
        rasterizer = self._create_rasterizer({"obj1": {"geometry": MockGeometry(mask, depth)}})
        self.assertEqual(rasterizer.in_frame_fraction("obj1"), 1.0)

    def test_object_without_geometry_reports_one(self) -> None:
        """Objects carrying no geometry report 1.0 so they are never gated."""
        rasterizer = self._create_rasterizer({"obj_nogeo": {"geometry": None}})
        self.assertEqual(rasterizer.in_frame_fraction("obj_nogeo"), 1.0)

    def test_key_error_for_unknown_object(self) -> None:
        """Querying an unknown object id raises KeyError, matching the other accessors."""
        rasterizer = self._create_rasterizer({})
        with self.assertRaises(KeyError):
            rasterizer.in_frame_fraction("does_not_exist")


class SpineMockGeometry(MockGeometry):
    """Mock geometry that also exposes a centerline, standing in for a Polyline."""

    def __init__(
        self, mask: np.ndarray, depth_mask: np.ndarray, spine_mask: np.ndarray, spine_depth: np.ndarray
    ) -> None:
        super().__init__(mask, depth_mask)
        self._spine_mask = spine_mask
        self._spine_depth = spine_depth

    def get_projected_spine_mask(
        self,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
        min_ribbon_width_px: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Stands in for a real polyline's width cutoff: this mock's ribbon is 240 px wide at 1 m, so a sample
        at depth d measures 240 / d and is dropped when that falls below the threshold."""
        mask, depth = self._spine_mask.copy(), self._spine_depth.copy()
        if min_ribbon_width_px > 0.0:
            too_narrow = np.zeros_like(mask)
            traced = mask & np.isfinite(depth)
            too_narrow[traced] = (240.0 / depth[traced]) < min_ribbon_width_px
            mask &= ~too_narrow
            depth[too_narrow] = np.inf
        return mask, depth


class TestVisibleSpineMask(unittest.TestCase):
    """Occlusion-aware centerline lookup backing the ``spine`` correspondence metric."""

    def setUp(self) -> None:
        self.image_width = 20
        self.image_height = 10
        self.camera_pose = np.eye(4)
        self.camera_model = MagicMock()
        self.logger = logging.getLogger("test")

    def _rasterizer(self, objects: dict[str, dict[str, Any]]) -> SceneRasterizer:
        return SceneRasterizer(
            objects,
            self.camera_pose,
            self.camera_model,
            self.image_width,
            self.image_height,
            logger=self.logger,
        )

    def _visible_spine(self, rasterizer: SceneRasterizer, object_id: str, width_px: float = 0.0) -> np.ndarray:
        """``get_visible_spine_mask`` narrowed to a real mask — these cases all use spine-bearing geometry."""
        mask = rasterizer.get_visible_spine_mask(object_id, width_px)
        self.assertIsNotNone(mask)
        assert mask is not None
        return mask

    def _line_geometry(self, depth_value: float, spine_depth_value: float | None = None) -> SpineMockGeometry:
        """A horizontal band 3 px tall at a fixed depth, whose centerline is its middle row.

        ``spine_depth_value`` defaults to ``depth_value``; passing a different value models the rounding
        disagreement between two rasterizations of one surface.
        """
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask[4:7, :] = True
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth[mask] = depth_value

        spine = np.zeros((self.image_height, self.image_width), dtype=bool)
        spine[5, :] = True
        spine_depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        spine_depth[spine] = depth_value if spine_depth_value is None else spine_depth_value
        return SpineMockGeometry(mask, depth, spine, spine_depth)

    def test_unoccluded_spine_is_returned_whole(self) -> None:
        rasterizer = self._rasterizer({"line": {"geometry": self._line_geometry(10.0)}})
        spine = rasterizer.get_visible_spine_mask("line")
        self.assertIsNotNone(spine)
        assert spine is not None
        self.assertEqual(int(spine.sum()), self.image_width)
        assert_array_equal(np.unique(np.nonzero(spine)[0]), np.array([5]))

    def test_nearer_object_hides_part_of_the_spine(self) -> None:
        """A vehicle standing on the line removes those centerline pixels from the denominator rather than
        scoring them as missing."""
        occluder_mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        occluder_mask[0:10, 5:10] = True
        occluder_depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        occluder_depth[occluder_mask] = 2.0

        rasterizer = self._rasterizer(
            {
                "line": {"geometry": self._line_geometry(10.0)},
                "car": {"geometry": MockGeometry(occluder_mask, occluder_depth)},
            }
        )
        spine = rasterizer.get_visible_spine_mask("line")
        assert spine is not None
        self.assertEqual(int(spine.sum()), self.image_width - 5)
        self.assertFalse(spine[5, 5:10].any())
        self.assertTrue(spine[5, 0:5].all())
        self.assertTrue(spine[5, 10:].all())

    def test_own_ribbon_never_occludes_its_spine(self) -> None:
        """The ribbon and the spine rasterize the same surface, and the ribbon is in the scene depth buffer.
        A spine pixel a hair behind its own ribbon must still count."""
        # Centerline half a meter behind its own ribbon, well past the 0.05 m visibility tolerance.
        geometry = self._line_geometry(10.0, spine_depth_value=10.5)
        rasterizer = self._rasterizer({"line": {"geometry": geometry}})
        spine = rasterizer.get_visible_spine_mask("line")
        assert spine is not None
        self.assertEqual(int(spine.sum()), self.image_width)

    def test_fully_occluded_spine_is_empty(self) -> None:
        occluder_mask = np.ones((self.image_height, self.image_width), dtype=bool)
        occluder_depth = np.full((self.image_height, self.image_width), 1.0, dtype=np.float32)
        rasterizer = self._rasterizer(
            {
                "line": {"geometry": self._line_geometry(10.0)},
                "wall": {"geometry": MockGeometry(occluder_mask, occluder_depth)},
            }
        )
        spine = rasterizer.get_visible_spine_mask("line")
        assert spine is not None
        self.assertFalse(spine.any())

    def test_geometry_without_a_spine_returns_none(self) -> None:
        """Cuboids and surfaces have no centerline; callers use an area metric for them."""
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        mask[2:4, 2:4] = True
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        depth[mask] = 5.0
        rasterizer = self._rasterizer({"box": {"geometry": MockGeometry(mask, depth)}})
        self.assertIsNone(rasterizer.get_visible_spine_mask("box"))

    def test_object_without_geometry_returns_none(self) -> None:
        rasterizer = self._rasterizer({"ghost": {}})
        self.assertIsNone(rasterizer.get_visible_spine_mask("ghost"))

    def test_computed_lazily_and_memoized(self) -> None:
        """A frame that scores no polyline class must not pay for centerline projection."""
        geometry = self._line_geometry(10.0)
        rasterizer = self._rasterizer({"line": {"geometry": geometry}})
        with patch.object(geometry, "get_projected_spine_mask", wraps=geometry.get_projected_spine_mask) as traced:
            first = rasterizer.get_visible_spine_mask("line")
            second = rasterizer.get_visible_spine_mask("line")
        self.assertEqual(traced.call_count, 1)
        self.assertIs(first, second)

    def test_memoizes_the_absence_of_a_spine(self) -> None:
        mask = np.zeros((self.image_height, self.image_width), dtype=bool)
        depth = np.full((self.image_height, self.image_width), np.inf, dtype=np.float32)
        geometry = MockGeometry(mask, depth)
        rasterizer = self._rasterizer({"box": {"geometry": geometry}})
        with patch.object(geometry, "get_projected_spine_mask", wraps=geometry.get_projected_spine_mask) as traced:
            self.assertIsNone(rasterizer.get_visible_spine_mask("box"))
            self.assertIsNone(rasterizer.get_visible_spine_mask("box"))
        self.assertEqual(traced.call_count, 1)

    def test_key_error_for_unknown_object(self) -> None:
        rasterizer = self._rasterizer({})
        with self.assertRaises(KeyError):
            rasterizer.get_visible_spine_mask("does_not_exist")

    def test_width_cutoff_truncates_the_traced_centerline(self) -> None:
        """Only the scored centerline shortens; the object's area mask and its depth-buffer contribution are
        untouched, so it still occludes whatever it did before. At 30 m the mock's ribbon measures 8 px."""
        rasterizer = self._rasterizer({"line": {"geometry": self._line_geometry(30.0)}})
        self.assertEqual(int(self._visible_spine(rasterizer, "line").sum()), self.image_width)
        self.assertFalse(self._visible_spine(rasterizer, "line", width_px=12.0).any())
        self.assertEqual(int(self._visible_spine(rasterizer, "line", width_px=6.0).sum()), self.image_width)

    def test_width_cutoff_is_part_of_the_memo_key(self) -> None:
        """Two cutoffs must not collide in the cache — a run that asks for both has to get both answers."""
        geometry = self._line_geometry(30.0)
        rasterizer = self._rasterizer({"line": {"geometry": geometry}})
        with patch.object(geometry, "get_projected_spine_mask", wraps=geometry.get_projected_spine_mask) as traced:
            full = self._visible_spine(rasterizer, "line")
            clipped = self._visible_spine(rasterizer, "line", width_px=12.0)
            again = self._visible_spine(rasterizer, "line", width_px=12.0)
        self.assertEqual(traced.call_count, 2, "each distinct cutoff projects once, then memoizes")
        self.assertIs(clipped, again)
        self.assertTrue(full.any())
        self.assertFalse(clipped.any())


if __name__ == "__main__":
    unittest.main()
