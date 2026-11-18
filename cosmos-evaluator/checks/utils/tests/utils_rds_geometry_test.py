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

"""Unit tests for checks.utils.rds_geometry."""

import unittest
from unittest.mock import MagicMock

import numpy as np
from third_party.cosmos_drive_dreams_toolkits.utils.camera.base import CameraBase

from checks.utils.rds_geometry import RDSGeometry


class _DummyGeometry(RDSGeometry):
    """Concrete geometry for exercising the RDSGeometry ABC."""

    def get_projected_mask(
        self,
        camera_to_world_pose: np.ndarray,
        camera_model: CameraBase,
        image_width: int,
        image_height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = np.zeros((image_height, image_width), dtype=bool)
        depth = np.full((image_height, image_width), np.inf, dtype=np.float32)
        if image_width > 0 and image_height > 0:
            mask[0, 0] = True
            depth[0, 0] = 1.0
        return mask, depth


class TestRDSGeometry(unittest.TestCase):
    def test_cannot_instantiate_abstract_base(self) -> None:
        """RDSGeometry is abstract; instantiating it directly must raise TypeError."""
        with self.assertRaises(TypeError):
            RDSGeometry()  # type: ignore[abstract]

    def test_subclass_implements_interface(self) -> None:
        """A concrete subclass returns mask/depth in (height, width) order with matching values."""
        g = _DummyGeometry()
        pose = np.eye(4, dtype=np.float32)
        cam = MagicMock()
        m, d = g.get_projected_mask(pose, cam, 4, 5)
        self.assertEqual(m.shape, (5, 4))
        self.assertEqual(d.shape, (5, 4))
        self.assertTrue(m[0, 0])
        self.assertEqual(float(d[0, 0]), 1.0)

    def test_subclass_handles_various_image_dimensions(self) -> None:
        """Mask/depth shapes track (image_height, image_width) across edge sizes."""
        g = _DummyGeometry()
        pose = np.eye(4, dtype=np.float32)
        cam = MagicMock()
        cases = [
            ("1x1", 1, 1),
            ("non-square wider than tall", 16, 4),
            ("non-square taller than wide", 4, 16),
        ]
        for name, width, height in cases:
            with self.subTest(case=name):
                m, d = g.get_projected_mask(pose, cam, width, height)
                self.assertEqual(m.shape, (height, width))
                self.assertEqual(d.shape, (height, width))
                self.assertEqual(m.dtype, np.bool_)
                self.assertEqual(d.dtype, np.float32)

    def test_subclass_zero_dimensions_produce_empty_arrays(self) -> None:
        """Zero width or height yields empty mask/depth without raising."""
        g = _DummyGeometry()
        pose = np.eye(4, dtype=np.float32)
        cam = MagicMock()
        cases = [
            ("zero width", 0, 5),
            ("zero height", 4, 0),
            ("zero both", 0, 0),
        ]
        for name, width, height in cases:
            with self.subTest(case=name):
                m, d = g.get_projected_mask(pose, cam, width, height)
                self.assertEqual(m.shape, (height, width))
                self.assertEqual(d.shape, (height, width))
                self.assertEqual(m.size, 0)
                self.assertEqual(d.size, 0)

    def test_subclass_must_implement_get_projected_mask(self) -> None:
        """Subclasses that omit get_projected_mask cannot be instantiated."""

        class _Incomplete(RDSGeometry):
            pass

        with self.assertRaises(TypeError):
            _Incomplete()  # type: ignore[abstract]

    def test_world_distance_default_is_inf(self) -> None:
        """
        The base class ``world_distance`` returns +inf so subclasses that don't
        override it (or comparisons across unrelated subclasses) cannot
        accidentally bind during tracking.
        """
        a = _DummyGeometry()
        b = _DummyGeometry()
        self.assertEqual(a.world_distance(b), float("inf"))

    def test_match_gate_default_is_inf(self) -> None:
        """The base class default gate is +inf; subclasses set finite values."""
        self.assertEqual(RDSGeometry.MATCH_GATE_METERS, float("inf"))

    def test_in_frame_fraction_default_is_one(self) -> None:
        """
        The base ``in_frame_fraction`` returns 1.0 so geometries without a meaningful
        frame-edge concept are never gated by the dynamic in-frame check; only concrete
        types that can be truncated (e.g. Cuboid) override it.
        """
        g = _DummyGeometry()
        cam = MagicMock()
        self.assertEqual(g.in_frame_fraction(np.eye(4, dtype=np.float32), cam, 8, 6), 1.0)


if __name__ == "__main__":
    unittest.main()
