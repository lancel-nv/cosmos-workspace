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

"""Unit tests for rds_data_loader module."""

import copy
from pathlib import Path
import shutil
import tempfile
from typing import Any
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from numpy.testing import assert_array_almost_equal

from checks.utils.rds_data_loader import RdsDataLoader
from checks.utils.rds_geometry import RDSGeometry


class TestRdsDataLoader(unittest.TestCase):
    """Test cases for RdsDataLoader class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_dir = Path(self.temp_dir)
        self.clip_id = "uuid_1234567890"
        self.session_uuid = "uuid"

        # Create directory structure
        (self.dataset_dir / "all_object_info").mkdir(parents=True)
        (self.dataset_dir / "pose").mkdir(parents=True)
        (self.dataset_dir / "3d_traffic_lights").mkdir(parents=True)
        (self.dataset_dir / "3d_traffic_signs").mkdir(parents=True)
        (self.dataset_dir / "3d_lanelines").mkdir(parents=True)
        (self.dataset_dir / "3d_road_boundaries").mkdir(parents=True)
        (self.dataset_dir / "3d_wait_lines").mkdir(parents=True)
        (self.dataset_dir / "3d_crosswalks").mkdir(parents=True)
        (self.dataset_dir / "pinhole_intrinsic").mkdir(parents=True)
        (self.dataset_dir / "ftheta_intrinsic").mkdir(parents=True)

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_webdataset_sample(self, data: dict) -> dict:
        """Helper to create a mock WebDataset sample."""
        return data

    def _create_mock_object_data(self, num_frames: int = 3) -> dict:
        """Create mock object data for testing."""
        object_data = {}
        for frame_idx in range(num_frames):
            frame_objects = {
                "0": {
                    "object_type": "Car",
                    "object_to_world": np.eye(4),
                    "object_lwh": np.array([4.0, 2.0, 1.5]),
                },
                "1": {
                    "object_type": "Pedestrian",
                    "object_to_world": np.eye(4),
                    "object_lwh": np.array([0.5, 0.5, 1.7]),
                },
            }
            object_data[f"{frame_idx}.all_object_info.json"] = frame_objects
        return object_data

    def _create_mock_pose_data(self, num_frames: int = 5) -> dict:
        """Create mock pose data for testing."""
        pose_data = {}
        for frame_idx in range(num_frames):
            pose = np.eye(4)
            pose[:3, 3] = [frame_idx * 0.1, 0.0, 0.0]
            pose_data[f"{frame_idx}.pose.camera_front.npy"] = pose
        return pose_data

    def _create_mock_intrinsics_data(self, camera_name: str, model_type: str) -> dict:
        """Create mock intrinsics data for testing."""
        if model_type == "pinhole":
            intrinsic = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
        else:  # ftheta
            intrinsic = np.random.rand(10, 10)  # Mock ftheta intrinsics
        return {f"{model_type}_intrinsic.{camera_name}.npy": intrinsic}

    def _create_mock_label_data(self) -> dict:
        """Create mock label data for testing."""
        return {
            "labels.json": {
                "labels": [
                    {
                        "labelData": {
                            "shape3d": {
                                "polyline3d": {
                                    "vertices": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]],
                                },
                                "attributes": [
                                    {
                                        "name": "colors",
                                        "enumsList": {"enumsList": ["red", "yellow", "white"]},
                                    }
                                ],
                            }
                        }
                    },
                    {
                        "labelData": {
                            "shape3d": {
                                "cuboid3d": {
                                    "vertices": [
                                        [0.0, 0.0, 0.0],
                                        [1.0, 0.0, 0.0],
                                        [1.0, 1.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                        [0.0, 0.0, 1.0],
                                        [1.0, 0.0, 1.0],
                                        [1.0, 1.0, 1.0],
                                        [0.0, 1.0, 1.0],
                                    ],
                                },
                                "attributes": [{"name": "type", "text": "solid"}],
                            }
                        }
                    },
                ]
            }
        }

    @patch("checks.utils.rds_data_loader.WebDataset")
    def test_get_sample_static_method(self, mock_webdataset: MagicMock) -> None:
        """Test static get_sample method."""
        mock_sample = {"key1": "value1", "key2": np.array([1, 2, 3])}
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([mock_sample]))
        mock_webdataset.return_value.decode.return_value = mock_dataset

        result = RdsDataLoader.get_sample("test.tar")

        self.assertEqual(result, mock_sample)
        mock_webdataset.assert_called_once_with("test.tar", shardshuffle=False)

    @patch("checks.utils.rds_data_loader.WebDataset")
    def test_get_sample_with_path_object(self, mock_webdataset: MagicMock) -> None:
        """Test get_sample with Path object."""
        mock_sample = {"key": "value"}
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([mock_sample]))
        mock_webdataset.return_value.decode.return_value = mock_dataset

        result = RdsDataLoader.get_sample(Path("test.tar"))

        self.assertEqual(result, mock_sample)
        mock_webdataset.assert_called_once_with("test.tar", shardshuffle=False)

    @patch("checks.utils.rds_data_loader.WebDataset")
    def test_get_sample_closes_dataset(self, mock_webdataset: MagicMock) -> None:
        """Test get_sample closes dataset if it has close method."""
        mock_sample = {"key": "value"}
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter([mock_sample]))
        mock_dataset.close = Mock()
        mock_webdataset.return_value.decode.return_value = mock_dataset

        result = RdsDataLoader.get_sample("test.tar")

        self.assertEqual(result, mock_sample)
        mock_dataset.close.assert_called_once()

    @patch.object(RdsDataLoader, "get_sample")
    @patch.object(RdsDataLoader, "_get_dynamic_object_data_all_frames")
    @patch.object(RdsDataLoader, "_get_static_objects")
    def test_init(self, mock_get_static: MagicMock, mock_get_dynamic: MagicMock, mock_get_sample: MagicMock) -> None:
        """Test RdsDataLoader initialization."""
        mock_get_dynamic.return_value = ({}, 0)
        mock_get_static.return_value = {}

        loader = RdsDataLoader(self.dataset_dir, self.clip_id)

        self.assertEqual(loader.dataset_dir, self.dataset_dir)
        self.assertEqual(loader.clip_id, self.clip_id)
        self.assertEqual(loader.session_uuid, self.session_uuid)
        self.assertEqual(loader.CAMERA_RESCALED_RESOLUTION_WIDTH, 1280)
        self.assertEqual(loader.CAMERA_RESCALED_RESOLUTION_HEIGHT, 720)

    @patch.object(RdsDataLoader, "get_sample")
    @patch.object(RdsDataLoader, "_get_static_objects")
    @patch.object(RdsDataLoader, "_get_dynamic_object_data_all_frames")
    def test_get_object_data_for_frame_stamps_ego_distance(
        self, mock_get_dynamic: MagicMock, mock_get_static: MagicMock, mock_get_sample: MagicMock
    ) -> None:
        """get_object_data_for_frame stamps the loader-recorded ego_distance_m on each per-frame view (None for
        centroid-less geometry / no ego pose), and get_object_ego_distances returns the full per-frame map."""
        mock_get_dynamic.return_value = ({}, 0)
        mock_get_static.return_value = {}
        loader = RdsDataLoader(self.dataset_dir, self.clip_id)

        def _geom(dist: float | None) -> MagicMock:
            g = MagicMock(spec=RDSGeometry)  # spec so isinstance(g, RDSGeometry) holds
            g.ego_ground_distance_m.return_value = dist
            return g

        # Dynamic obj 1 (30m at frame 0, 20m at frame 1); obj 2 centroid-less (None); obj 3 has no geometry key.
        loader.all_dynamic_objects = {
            0: {"1": {"geometry": _geom(30.0)}, "2": {"geometry": _geom(None)}},
            1: {"1": {"geometry": _geom(20.0)}, "3": {"geometry": None}},
        }
        with patch.object(loader, "load_ego_poses", return_value={0: np.eye(4), 1: np.eye(4)}):
            frame0 = loader.get_object_data_for_frame(0)
            self.assertEqual(frame0["1"]["ego_distance_m"], 30.0)
            self.assertIsNone(frame0["2"]["ego_distance_m"])  # centroid-less -> None
            self.assertIsNone(loader.get_object_data_for_frame(1)["3"]["ego_distance_m"])  # no geometry -> None

            dists = loader.get_object_ego_distances()
            self.assertEqual(dists[0]["1"], 30.0)
            self.assertEqual(dists[1]["1"], 20.0)  # per-frame, not min-aggregated
            self.assertIsNone(dists[0]["2"])

    @patch.object(RdsDataLoader, "get_sample")
    @patch.object(RdsDataLoader, "_get_static_objects")
    @patch.object(RdsDataLoader, "_get_dynamic_object_data_all_frames")
    def test_stamp_ego_distance_none_when_no_ego_pose(
        self, mock_get_dynamic: MagicMock, mock_get_static: MagicMock, mock_get_sample: MagicMock
    ) -> None:
        """A frame with no ego pose stamps ego_distance_m=None (rather than raising or recomputing)."""
        mock_get_dynamic.return_value = ({}, 0)
        mock_get_static.return_value = {}
        loader = RdsDataLoader(self.dataset_dir, self.clip_id)
        geom = MagicMock(spec=RDSGeometry)
        geom.ego_ground_distance_m.return_value = 12.0
        loader.all_dynamic_objects = {5: {"1": {"geometry": geom}}}
        with patch.object(loader, "load_ego_poses", return_value={}):  # frame 5 absent
            frame5 = loader.get_object_data_for_frame(5)
            self.assertIsNone(frame5["1"]["ego_distance_m"])
            geom.ego_ground_distance_m.assert_not_called()

    @patch.object(RdsDataLoader, "get_sample")
    @patch.object(RdsDataLoader, "_get_static_objects")
    @patch.object(RdsDataLoader, "_get_dynamic_object_data_all_frames")
    def test_get_object_ego_distances_static_only_frame(
        self, mock_get_dynamic: MagicMock, mock_get_static: MagicMock, mock_get_sample: MagicMock
    ) -> None:
        """A static object on a frame with no dynamic objects is still visited, stamped, and in the ego-distance map."""
        mock_get_dynamic.return_value = ({}, 0)
        mock_get_static.return_value = {}
        loader = RdsDataLoader(self.dataset_dir, self.clip_id)
        geom = MagicMock(spec=RDSGeometry)  # spec so isinstance(geom, RDSGeometry) holds after the per-frame deepcopy
        geom.ego_ground_distance_m.return_value = 42.0
        loader.all_dynamic_objects = {}  # no dynamic frames at all -> frame 7 comes only from the static object
        loader.all_static_objects = {"tl7": {"is_static": True, "geometry_per_frame": {7: geom}}}
        with patch.object(loader, "load_ego_poses", return_value={7: np.eye(4)}):
            dists = loader.get_object_ego_distances()
            self.assertIn(7, dists)  # static-only frame present in the map (union of dynamic + static frames)
            self.assertEqual(dists[7]["tl7"], 42.0)
            self.assertEqual(loader.get_object_data_for_frame(7, include_static=True)["tl7"]["ego_distance_m"], 42.0)

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_camera_intrinsics_array_pinhole(self, mock_get_sample: MagicMock) -> None:
        """Test _get_camera_intrinsics_array for pinhole model."""
        camera_name = "camera_front"
        intrinsics_data = self._create_mock_intrinsics_data(camera_name, "pinhole")
        mock_get_sample.return_value = intrinsics_data

        # Create tar file
        tar_path = self.dataset_dir / "pinhole_intrinsic" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.PINHOLE_INTRINSICS_FOLDER = "pinhole_intrinsic"
        loader.FTHETA_INTRINSICS_FOLDER = "ftheta_intrinsic"

        result = loader._get_camera_intrinsics_array(camera_name, "pinhole")

        self.assertIsInstance(result, np.ndarray)
        mock_get_sample.assert_called_once()

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_camera_intrinsics_array_ftheta(self, mock_get_sample: MagicMock) -> None:
        """Test _get_camera_intrinsics_array for ftheta model."""
        camera_name = "camera_front"
        intrinsics_data = self._create_mock_intrinsics_data(camera_name, "ftheta")
        mock_get_sample.return_value = intrinsics_data

        tar_path = self.dataset_dir / "ftheta_intrinsic" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.PINHOLE_INTRINSICS_FOLDER = "pinhole_intrinsic"
        loader.FTHETA_INTRINSICS_FOLDER = "ftheta_intrinsic"

        result = loader._get_camera_intrinsics_array(camera_name, "ftheta")

        self.assertIsInstance(result, np.ndarray)

    def test_get_camera_intrinsics_array_invalid_model_type(self) -> None:
        """Test _get_camera_intrinsics_array with invalid model type."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        with self.assertRaises(ValueError) as context:
            loader._get_camera_intrinsics_array("camera_front", "invalid")  # type: ignore[arg-type]
        self.assertIn("Invalid model_type", str(context.exception))

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_camera_intrinsics_array_file_not_found(self, mock_get_sample: MagicMock) -> None:
        """Test _get_camera_intrinsics_array when file doesn't exist."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.PINHOLE_INTRINSICS_FOLDER = "pinhole_intrinsic"
        loader.FTHETA_INTRINSICS_FOLDER = "ftheta_intrinsic"

        with self.assertRaises(FileNotFoundError):
            loader._get_camera_intrinsics_array("camera_front", "pinhole")

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_camera_intrinsics_array_missing_camera(self, mock_get_sample: MagicMock) -> None:
        """Test _get_camera_intrinsics_array when camera not found."""
        mock_get_sample.return_value = {"other_camera.npy": np.array([1, 2, 3])}

        tar_path = self.dataset_dir / "pinhole_intrinsic" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.PINHOLE_INTRINSICS_FOLDER = "pinhole_intrinsic"
        loader.FTHETA_INTRINSICS_FOLDER = "ftheta_intrinsic"

        with self.assertRaises(KeyError) as context:
            loader._get_camera_intrinsics_array("camera_front", "pinhole")
        self.assertIn("Camera intrinsics not found", str(context.exception))

    @patch("checks.utils.rds_data_loader.FThetaCamera")
    @patch.object(RdsDataLoader, "_get_camera_intrinsics_array")
    def test_get_camera_intrinsics_ftheta(self, mock_get_array: MagicMock, mock_ftheta_camera: MagicMock) -> None:
        """Test get_camera_intrinsics for ftheta model."""
        mock_intrinsic = np.random.rand(10, 10)
        mock_get_array.return_value = mock_intrinsic

        mock_camera_instance = Mock()
        mock_camera_instance.height = 1440
        mock_ftheta_camera.from_numpy.return_value = mock_camera_instance

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.CAMERA_RESCALED_RESOLUTION_HEIGHT = 720

        result = loader.get_camera_intrinsics("camera_front", "ftheta", rescaled=True)

        self.assertEqual(result, mock_camera_instance)
        mock_ftheta_camera.from_numpy.assert_called_once_with(mock_intrinsic, device="cpu")
        mock_camera_instance.rescale.assert_called_once()

    @patch("checks.utils.rds_data_loader.FThetaCamera")
    @patch.object(RdsDataLoader, "_get_camera_intrinsics_array")
    def test_get_camera_intrinsics_ftheta_not_rescaled(
        self, mock_get_array: MagicMock, mock_ftheta_camera: MagicMock
    ) -> None:
        """Test get_camera_intrinsics for ftheta model without rescaling."""
        mock_intrinsic = np.random.rand(10, 10)
        mock_get_array.return_value = mock_intrinsic

        mock_camera_instance = Mock()
        mock_ftheta_camera.from_numpy.return_value = mock_camera_instance

        loader = RdsDataLoader.__new__(RdsDataLoader)

        result = loader.get_camera_intrinsics("camera_front", "ftheta", rescaled=False)

        self.assertEqual(result, mock_camera_instance)
        mock_camera_instance.rescale.assert_not_called()

    @patch.object(RdsDataLoader, "_get_camera_intrinsics_array")
    def test_get_camera_intrinsics_pinhole(self, mock_get_array: MagicMock) -> None:
        """Test get_camera_intrinsics for pinhole model returns None."""
        mock_intrinsic = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
        mock_get_array.return_value = mock_intrinsic

        loader = RdsDataLoader.__new__(RdsDataLoader)

        result = loader.get_camera_intrinsics("camera_front", "pinhole")

        self.assertIsNone(result)

    @patch("checks.utils.rds_data_loader.FThetaCamera")
    @patch.object(RdsDataLoader, "_get_camera_intrinsics_array")
    def test_get_camera_intrinsics_ftheta_runtime_error(
        self, mock_get_array: MagicMock, mock_ftheta_camera: MagicMock
    ) -> None:
        """Test get_camera_intrinsics raises RuntimeError when camera creation fails."""
        mock_intrinsic = np.random.rand(10, 10)
        mock_get_array.return_value = mock_intrinsic

        mock_ftheta_camera.from_numpy.side_effect = Exception("Camera creation failed")

        loader = RdsDataLoader.__new__(RdsDataLoader)

        with self.assertRaises(RuntimeError) as context:
            loader.get_camera_intrinsics("camera_front", "ftheta")
        self.assertIn("Failed to create FTheta camera model", str(context.exception))

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_camera_poses(self, mock_get_sample: MagicMock) -> None:
        """Test get_camera_poses method."""
        camera_name = "camera_front"
        pose_data = self._create_mock_pose_data(3)
        mock_get_sample.return_value = pose_data

        tar_path = self.dataset_dir / "pose" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.POSE_FOLDER = "pose"

        result = loader.get_camera_poses(camera_name)

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 3)
        for frame_idx in range(3):
            self.assertIn(frame_idx, result)
            self.assertEqual(result[frame_idx].shape, (4, 4))

    @patch.object(RdsDataLoader, "get_camera_intrinsics")
    @patch.object(RdsDataLoader, "get_camera_poses")
    def test_load_camera_data(self, mock_get_poses: MagicMock, mock_get_intrinsics: MagicMock) -> None:
        """Test load_camera_data returns poses and the ftheta intrinsics model, delegating to the getters."""
        mock_poses = {0: np.eye(4)}
        mock_model = Mock()
        mock_get_poses.return_value = mock_poses
        mock_get_intrinsics.return_value = mock_model

        loader = RdsDataLoader.__new__(RdsDataLoader)

        camera_poses, camera_model = loader.load_camera_data("camera_front")

        self.assertIs(camera_poses, mock_poses)
        self.assertIs(camera_model, mock_model)
        mock_get_poses.assert_called_once_with("camera_front")
        mock_get_intrinsics.assert_called_once_with("camera_front", "ftheta")

    @patch.object(RdsDataLoader, "get_camera_poses")
    def test_load_ego_poses(self, mock_get_poses: MagicMock) -> None:
        """Test load_ego_poses delegates to get_camera_poses using the ego reference camera."""
        mock_poses = {0: np.eye(4), 1: np.eye(4)}
        mock_get_poses.return_value = mock_poses

        loader = RdsDataLoader.__new__(RdsDataLoader)

        result = loader.load_ego_poses()

        self.assertIs(result, mock_poses)
        mock_get_poses.assert_called_once_with(RdsDataLoader.EGO_REFERENCE_CAMERA)

    @patch.object(RdsDataLoader, "get_camera_poses")
    def test_load_ego_poses_propagates_error(self, mock_get_poses: MagicMock) -> None:
        """Test load_ego_poses propagates errors from get_camera_poses (no graceful fallback in the loader)."""
        mock_get_poses.side_effect = FileNotFoundError("pose file missing")

        loader = RdsDataLoader.__new__(RdsDataLoader)

        with self.assertRaises(FileNotFoundError):
            loader.load_ego_poses()

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_dynamic_object_data_all_frames(self, mock_get_sample: MagicMock) -> None:
        """Test get_dynamic_object_data_all_frames method."""
        object_data = self._create_mock_object_data(2)
        mock_get_sample.return_value = object_data

        tar_path = self.dataset_dir / "all_object_info" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.OBJECT_FOLDER = "all_object_info"

        result, next_object_id = loader._get_dynamic_object_data_all_frames()

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        # One *past* the highest dynamic ID (objects are "0" and "1"), not the highest itself: callers use
        # this as the next free ID, so returning the maximum handed the first allocated static an ID that a
        # dynamic object already owned. Stated as the invariant as well as the constant, because it is the
        # invariant every ID allocator downstream depends on.
        self.assertEqual(next_object_id, 2)
        highest_dynamic = max(int(tid) for frame in result.values() for tid in frame)
        self.assertGreater(next_object_id, highest_dynamic)
        for frame_idx in range(2):
            self.assertIn(frame_idx, result)
            for obj_info in result[frame_idx].values():
                self.assertIn("geometry", obj_info)
                self.assertFalse(obj_info["is_static"])

    @patch.object(RdsDataLoader, "get_sample")
    def test_clip_with_no_dynamic_objects_starts_the_id_space_at_zero(self, mock_get_sample: MagicMock) -> None:
        """A static-only clip has nothing to clear, so the first free ID is 0, not 1.

        The seed is "one past the highest dynamic ID", and with no dynamic objects there is no highest.
        Starting at 1 would leave 0 held by nobody and shift every static in the clip for no reason.
        """
        mock_get_sample.return_value = {f"{i}.all_object_info.json": {} for i in range(2)}
        tar_path = self.dataset_dir / "all_object_info" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.OBJECT_FOLDER = "all_object_info"

        result, next_object_id = loader._get_dynamic_object_data_all_frames()

        self.assertEqual(len(result), 2)  # the frames exist
        self.assertFalse(any(result.values()))  # ...and carry no objects
        self.assertEqual(next_object_id, 0)

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_dynamic_object_data_missing_fields(self, mock_get_sample: MagicMock) -> None:
        """Test get_dynamic_object_data_all_frames with missing fields."""
        object_data = {
            "0.all_object_info.json": {
                "0": {"object_type": "Car"},  # Missing object_to_world and object_lwh
                "1": {
                    "object_type": "Pedestrian",
                    "object_to_world": np.eye(4),
                    "object_lwh": np.array([0.5, 0.5, 1.7]),
                },
            }
        }
        mock_get_sample.return_value = object_data

        tar_path = self.dataset_dir / "all_object_info" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.OBJECT_FOLDER = "all_object_info"

        result, _max_object_id = loader._get_dynamic_object_data_all_frames()

        # Object 0 should be skipped, only object 1 should have geometry
        self.assertIn(0, result)
        self.assertIn("1", result[0])
        self.assertIn("geometry", result[0]["1"])

    def test_get_objects_by_class_vehicle(self) -> None:
        """Test get_objects_by_class for vehicle class."""
        frame_objects = {
            "0": {"object_type": "Car"},
            "1": {"object_type": "Pedestrian"},
            "2": {"object_type": "Truck"},
            "3": {"object_type": "Motorcycle"},
        }

        result = RdsDataLoader.get_objects_by_class(frame_objects, "vehicle")

        self.assertEqual(len(result), 2)
        self.assertIn("0", result)
        self.assertIn("2", result)
        self.assertNotIn("1", result)
        self.assertNotIn("3", result)

    def test_get_objects_by_class_pedestrian(self) -> None:
        """Test get_objects_by_class for pedestrian class."""
        frame_objects = {
            "0": {"object_type": "Car"},
            "1": {"object_type": "Pedestrian"},
            "2": {"object_type": "Person"},
        }

        result = RdsDataLoader.get_objects_by_class(frame_objects, "pedestrian")

        self.assertEqual(len(result), 2)
        self.assertIn("1", result)
        self.assertIn("2", result)

    def test_get_objects_by_class_unknown_class(self) -> None:
        """Test get_objects_by_class for unknown class."""
        frame_objects = {
            "0": {"object_type": "Car"},
            "1": {"object_type": "UnknownType"},
        }

        result = RdsDataLoader.get_objects_by_class(frame_objects, "unknown_class")

        # Should filter by exact match when class not in mapping
        self.assertEqual(len(result), 0)

    def test_get_all_traffic_light_status_returns_empty_when_tar_missing(self) -> None:
        """Test _get_all_traffic_light_status returns empty dict when tar does not exist."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.TRAFFIC_LIGHT_STATUS_FOLDER = "3d_traffic_lights_status"

        result = loader._get_all_traffic_light_status()

        self.assertEqual(result, {})
        self.assertIsInstance(result, dict)

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_all_traffic_light_status_returns_empty_when_no_json(self, mock_get_sample: MagicMock) -> None:
        """Test _get_all_traffic_light_status returns empty dict when sample has no .json key."""
        mock_get_sample.return_value = {"other_file.npy": np.array([1, 2, 3])}

        (self.dataset_dir / "3d_traffic_lights_status").mkdir(parents=True)
        tar_path = self.dataset_dir / "3d_traffic_lights_status" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.TRAFFIC_LIGHT_STATUS_FOLDER = "3d_traffic_lights_status"

        result = loader._get_all_traffic_light_status()

        self.assertEqual(result, {})
        mock_get_sample.assert_called_once()

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_all_traffic_light_status_returns_states_when_present(self, mock_get_sample: MagicMock) -> None:
        """Test _get_all_traffic_light_status returns traffic_light_states from sample."""
        traffic_light_states = {"0": {"state": ["red", "green"]}, "1": {"state": ["yellow"]}}
        mock_get_sample.return_value = {"traffic_light_states.json": {"traffic_light_states": traffic_light_states}}

        (self.dataset_dir / "3d_traffic_lights_status").mkdir(parents=True)
        tar_path = self.dataset_dir / "3d_traffic_lights_status" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.TRAFFIC_LIGHT_STATUS_FOLDER = "3d_traffic_lights_status"

        result = loader._get_all_traffic_light_status()

        self.assertIsInstance(result, dict)
        self.assertEqual(result, traffic_light_states)
        self.assertEqual(result["0"]["state"], ["red", "green"])
        self.assertEqual(result["1"]["state"], ["yellow"])
        mock_get_sample.assert_called_once()

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_all_traffic_light_status_returns_empty_when_key_missing(self, mock_get_sample: MagicMock) -> None:
        """Test _get_all_traffic_light_status returns empty dict when traffic_light_states key is absent."""
        mock_get_sample.return_value = {"traffic_light_states.json": {"other_key": "other_value"}}

        (self.dataset_dir / "3d_traffic_lights_status").mkdir(parents=True)
        tar_path = self.dataset_dir / "3d_traffic_lights_status" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id
        loader.TRAFFIC_LIGHT_STATUS_FOLDER = "3d_traffic_lights_status"

        result = loader._get_all_traffic_light_status()

        self.assertEqual(result, {})
        mock_get_sample.assert_called_once()

    @patch.object(RdsDataLoader, "get_labels")
    @patch.object(RdsDataLoader, "get_per_frame_labels", return_value={})
    @patch.object(RdsDataLoader, "_get_all_traffic_light_status", return_value={})
    def test_get_static_geometry(
        self,
        mock_get_traffic_light_status: MagicMock,
        mock_get_per_frame_labels: MagicMock,
        mock_get_labels: MagicMock,
    ) -> None:
        """Test get_static_geometry method."""
        mock_get_labels.side_effect = [
            [  # LaneLine
                {
                    "vertices": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
                    "geom_type": "polyline",
                }
            ],
            [  # RoadBoundary
                {
                    "vertices": np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]),
                    "geom_type": "polyline",
                }
            ],
            [],  # WaitLine
            [],  # TrafficLight
            [],  # TrafficSign
            [],  # Crosswalk
        ]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"
        loader._cached_canonical_static_frame_axis = [0]

        result = loader._get_static_objects()

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)  # Two polylines
        for obj_info in result.values():
            self.assertTrue(obj_info["is_static"])
            self.assertIn("object_type", obj_info)
            # Canonical and per-frame entries point at the same instance
            # in Phase 1 (canonical replicated across the frame axis).
            self.assertIn("geometry", obj_info)
            self.assertIn("geometry_per_frame", obj_info)
            self.assertIn(0, obj_info["geometry_per_frame"])
            self.assertIs(obj_info["geometry_per_frame"][0], obj_info["geometry"])

    @patch.object(RdsDataLoader, "get_labels")
    @patch.object(RdsDataLoader, "get_per_frame_labels", return_value={})
    @patch.object(RdsDataLoader, "_get_all_traffic_light_status")
    def test_get_static_objects_traffic_light_with_state(
        self,
        mock_get_traffic_light_status: MagicMock,
        mock_get_per_frame_labels: MagicMock,
        mock_get_labels: MagicMock,
    ) -> None:
        """Test _get_static_objects assigns state_per_frame when traffic light status has state."""
        cuboid_vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1.0]]
        )
        mock_get_traffic_light_status.return_value = {"0": {"state": ["red", "green", "red"]}}
        mock_get_labels.side_effect = [
            [],  # LaneLine
            [],  # RoadBoundary
            [],  # WaitLine
            [{"vertices": cuboid_vertices, "geom_type": "cuboid"}],  # TrafficLight
            [],  # TrafficSign
            [],  # Crosswalk
        ]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"
        loader._cached_canonical_static_frame_axis = [0]

        result = loader._get_static_objects()

        obj = next(iter(result.values()))
        self.assertEqual(obj["object_type"], "TrafficLight")
        self.assertEqual(obj["state_per_frame"], ["red", "green", "red"])

    @patch.object(RdsDataLoader, "get_labels")
    @patch.object(RdsDataLoader, "get_per_frame_labels", return_value={})
    @patch.object(RdsDataLoader, "_get_all_traffic_light_status")
    def test_get_static_objects_traffic_light_missing_state_key(
        self,
        mock_get_traffic_light_status: MagicMock,
        mock_get_per_frame_labels: MagicMock,
        mock_get_labels: MagicMock,
    ) -> None:
        """Test _get_static_objects omits state_per_frame when status entry lacks state key."""
        cuboid_vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1.0]]
        )
        mock_get_traffic_light_status.return_value = {"0": {"other_field": "value"}}
        mock_get_labels.side_effect = [
            [],  # LaneLine
            [],  # RoadBoundary
            [],  # WaitLine
            [{"vertices": cuboid_vertices, "geom_type": "cuboid"}],  # TrafficLight
            [],  # TrafficSign
            [],  # Crosswalk
        ]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"
        loader._cached_canonical_static_frame_axis = [0]

        result = loader._get_static_objects()

        obj = next(iter(result.values()))
        self.assertEqual(obj["object_type"], "TrafficLight")
        self.assertNotIn("state_per_frame", obj)

    def test_state_per_frame_from_labels_holds_each_label_forward(self) -> None:
        """Each 10 Hz label fills its own three RDS indices; an unlabeled slot and the uncovered tail read unknown."""
        obj = {
            "geometry_per_frame": {0: Mock(), 3: Mock(), 9: Mock()},
            "attributes_per_frame": {0: {"state": "red"}, 3: {"state": "green"}, 9: {"state": "red"}},
        }

        state_per_frame = RdsDataLoader._state_per_frame_from_labels(obj, 11)

        self.assertEqual(
            state_per_frame,
            ["red"] * 3 + ["green"] * 3 + ["unknown"] * 3 + ["red"] * 2,
        )

    def test_state_per_frame_from_labels_rejects_unusable_observations(self) -> None:
        """A light with no observations, or with an observation whose state is missing or empty, gets no list."""
        cases: dict[str, dict[str, Any]] = {
            "no observations": {"geometry_per_frame": {}, "attributes_per_frame": {}},
            "observation without attributes": {
                "geometry_per_frame": {0: Mock(), 3: Mock()},
                "attributes_per_frame": {0: {"state": "red"}},
            },
            "observation without state": {
                "geometry_per_frame": {0: Mock(), 3: Mock()},
                "attributes_per_frame": {0: {"state": "red"}, 3: {"feature_id": "7"}},
            },
            "empty state": {
                "geometry_per_frame": {0: Mock()},
                "attributes_per_frame": {0: {"state": " "}},
            },
        }
        for name, obj in cases.items():
            with self.subTest(name):
                self.assertIsNone(RdsDataLoader._state_per_frame_from_labels(obj, 6))

    def _label_state_loader(self, label_frames: list[int]) -> RdsDataLoader:
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.clip_id = self.clip_id
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader._cached_per_frame_labels = {"3d_traffic_lights": {frame: [] for frame in label_frames}}
        return loader

    def test_label_state_num_frames_spans_the_pose_axis(self) -> None:
        loader = self._label_state_loader([0, 3, 6])

        self.assertEqual(loader._label_state_num_frames({i: np.eye(4) for i in range(8)}), 8)

    def test_label_state_num_frames_without_poses_ends_after_last_hold(self) -> None:
        loader = self._label_state_loader([0, 3, 6])

        self.assertEqual(loader._label_state_num_frames({}), 9)

    def test_label_state_num_frames_none_without_per_frame_labels(self) -> None:
        loader = self._label_state_loader([])

        self.assertIsNone(loader._label_state_num_frames({0: np.eye(4)}))

    def test_label_state_num_frames_none_when_labels_are_off_grid(self) -> None:
        loader = self._label_state_loader([0, 3, 5])

        with self.assertLogs("checks.utils.rds_data_loader", level="WARNING") as logs:
            self.assertIsNone(loader._label_state_num_frames({0: np.eye(4)}))
        self.assertIn("off the every-3 RDS index grid", logs.output[0])

    def _per_frame_traffic_light_loader(self) -> RdsDataLoader:
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.clip_id = self.clip_id
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"
        loader._cached_canonical_static_frame_axis = [0, 3]
        return loader

    @staticmethod
    def _traffic_light_label(offset: float, state: str) -> dict[str, Any]:
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1.0]],
            dtype=np.float32,
        )
        return {"vertices": vertices + offset, "geom_type": "cuboid", "attributes": {"state": state}}

    def _fake_traffic_light_labels(self) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
        """One canonical light observed at frames 0 and 3, plus a light at frame 3 that matches no canonical."""
        canonical = [self._traffic_light_label(0.0, "red")]
        per_frame = {
            0: [self._traffic_light_label(0.0, "red")],
            3: [self._traffic_light_label(0.0, "green"), self._traffic_light_label(500.0, "red")],
        }
        return canonical, per_frame

    @patch.object(RdsDataLoader, "_ego_poses_for_synthesized_canonical")
    @patch.object(RdsDataLoader, "get_per_frame_labels")
    @patch.object(RdsDataLoader, "get_labels")
    @patch.object(RdsDataLoader, "_get_all_traffic_light_status", return_value={})
    def test_get_static_objects_builds_traffic_light_state_from_per_frame_labels(
        self,
        _mock_status: MagicMock,
        mock_get_labels: MagicMock,
        mock_get_per_frame_labels: MagicMock,
        mock_ego_poses: MagicMock,
    ) -> None:
        """Without a status entry, canonical and synthesized lights both take their state from per-frame labels."""
        canonical, per_frame = self._fake_traffic_light_labels()
        mock_get_labels.side_effect = lambda folder: canonical if folder == "3d_traffic_lights" else []
        mock_get_per_frame_labels.side_effect = lambda folder: per_frame if folder == "3d_traffic_lights" else {}
        mock_ego_poses.return_value = {i: np.eye(4) for i in range(8)}

        result = self._per_frame_traffic_light_loader()._get_static_objects()

        self.assertEqual(set(result), {"0", "1"})
        self.assertEqual(result["0"]["state_per_frame"], ["red"] * 3 + ["green"] * 3 + ["unknown"] * 2)
        self.assertIsNone(result["1"]["object_type_index"])
        self.assertEqual(result["1"]["state_per_frame"], ["unknown"] * 3 + ["red"] * 3 + ["unknown"] * 2)

    @patch.object(RdsDataLoader, "_ego_poses_for_synthesized_canonical")
    @patch.object(RdsDataLoader, "get_per_frame_labels")
    @patch.object(RdsDataLoader, "get_labels")
    @patch.object(RdsDataLoader, "_get_all_traffic_light_status")
    def test_get_static_objects_prefers_traffic_light_status_over_per_frame_labels(
        self,
        mock_status: MagicMock,
        mock_get_labels: MagicMock,
        mock_get_per_frame_labels: MagicMock,
        mock_ego_poses: MagicMock,
    ) -> None:
        canonical, per_frame = self._fake_traffic_light_labels()
        mock_status.return_value = {"0": {"state": ["yellow"] * 8}}
        mock_get_labels.side_effect = lambda folder: canonical if folder == "3d_traffic_lights" else []
        mock_get_per_frame_labels.side_effect = lambda folder: per_frame if folder == "3d_traffic_lights" else {}
        mock_ego_poses.return_value = {i: np.eye(4) for i in range(8)}

        result = self._per_frame_traffic_light_loader()._get_static_objects()

        self.assertEqual(result["0"]["state_per_frame"], ["yellow"] * 8)

    @patch.object(RdsDataLoader, "get_labels")
    @patch.object(RdsDataLoader, "get_per_frame_labels", return_value={})
    @patch.object(RdsDataLoader, "_get_all_traffic_light_status", return_value={})
    def test_get_static_geometry_with_cuboid(
        self,
        mock_get_traffic_light_status: MagicMock,
        mock_get_per_frame_labels: MagicMock,
        mock_get_labels: MagicMock,
    ) -> None:
        """Test get_static_geometry with cuboid geometry."""
        # Create cuboid vertices (8 corners)
        cuboid_vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )

        mock_get_labels.side_effect = [
            [],  # LaneLine
            [],  # RoadBoundary
            [],  # WaitLine
            [  # TrafficLight
                {
                    "vertices": cuboid_vertices,
                    "geom_type": "cuboid",
                }
            ],
            [],  # TrafficSign
            [],  # Crosswalk
        ]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"
        loader._cached_canonical_static_frame_axis = [0]

        result = loader._get_static_objects()

        self.assertEqual(len(result), 1)
        obj_info = next(iter(result.values()))
        self.assertEqual(obj_info["object_type"], "TrafficLight")
        self.assertIsNotNone(obj_info["geometry"])
        self.assertIn("geometry_per_frame", obj_info)
        self.assertIs(obj_info["geometry_per_frame"][0], obj_info["geometry"])

    @patch.object(RdsDataLoader, "get_labels")
    @patch.object(RdsDataLoader, "get_per_frame_labels", return_value={})
    @patch.object(RdsDataLoader, "_get_all_traffic_light_status", return_value={})
    def test_get_static_geometry_skips_invalid_labels(
        self,
        mock_get_traffic_light_status: MagicMock,
        mock_get_per_frame_labels: MagicMock,
        mock_get_labels: MagicMock,
    ) -> None:
        """Test get_static_geometry skips labels without vertices or geom_type."""
        mock_get_labels.side_effect = [
            [
                {"vertices": np.array([[0, 0, 0], [1, 1, 0]]), "geom_type": "polyline"},  # Valid
                {"vertices": np.array([[0, 0, 0]])},  # Missing geom_type
                {"geom_type": "polyline"},  # Missing vertices
                {},  # Missing both
            ],
            [],  # RoadBoundary
            [],  # WaitLine
            [],  # TrafficLight
            [],  # TrafficSign
            [],  # Crosswalk
        ]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"
        loader._cached_canonical_static_frame_axis = [0]

        result = loader._get_static_objects()

        # Should only include the valid label
        self.assertEqual(len(result), 1)

    @patch.object(RdsDataLoader, "get_labels")
    @patch.object(RdsDataLoader, "get_per_frame_labels", return_value={})
    @patch.object(RdsDataLoader, "_get_all_traffic_light_status", return_value={})
    def test_get_static_geometry_with_attributes(
        self,
        mock_get_traffic_light_status: MagicMock,
        mock_get_per_frame_labels: MagicMock,
        mock_get_labels: MagicMock,
    ) -> None:
        """Test get_static_geometry preserves attributes."""
        mock_get_labels.side_effect = [
            [
                {
                    "vertices": np.array([[0, 0, 0], [1, 1, 0]]),
                    "geom_type": "polyline",
                    "attributes": {"color": "yellow"},
                }
            ],
            [],
            [],
            [],
            [],
            [],
        ]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"
        loader._cached_canonical_static_frame_axis = [0]

        result = loader._get_static_objects()

        obj_info = next(iter(result.values()))
        # Canonical attributes (frame-independent) and per-frame attributes
        # both reflect the source label.
        self.assertEqual(obj_info["attributes"]["color"], "yellow")
        self.assertIs(obj_info["attributes_per_frame"][0], obj_info["attributes"])

    def test_get_object_data_for_frame(self) -> None:
        """Test get_object_data_for_frame method.

        Static objects use per-frame indexed dicts; the per-frame view at
        ``frame_id`` plucks geometry/attributes for that frame from
        ``geometry_per_frame`` / ``attributes_per_frame`` (not from the
        canonical clip-level ``geometry`` / ``attributes`` fields), and
        omits static objects whose ``geometry_per_frame`` does not include
        ``frame_id``.
        """
        # Use distinct sentinel values for canonical vs per-frame so the
        # test can verify that the per-frame view sources from
        # geometry_per_frame, not from the clip-level canonical.
        canonical_geometry = ("polyline", "canonical")
        per_frame_geometry = ("polyline", "frame-0")
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader._object_ego_distances = {}
        loader.all_dynamic_objects = {0: {"0": {"object_type": "Car"}}, 1: {"1": {"object_type": "Pedestrian"}}}
        loader.all_static_objects = {
            "100": {
                "object_type": "LaneLine",
                "is_static": True,
                "geometry": canonical_geometry,
                "geometry_per_frame": {0: per_frame_geometry},
            }
        }

        # Test without static
        result = loader.get_object_data_for_frame(0, include_static=False)
        self.assertEqual(len(result), 1)
        self.assertIn("0", result)

        # Test with static — frame 0 has the static object surfaced.
        result = loader.get_object_data_for_frame(0, include_static=True)
        self.assertEqual(len(result), 2)
        self.assertIn("0", result)
        self.assertIn("100", result)
        # Geometry was plucked from geometry_per_frame[0] (not the canonical);
        # per-frame indexed dicts and the canonical fields are stripped from
        # the per-frame view.
        self.assertEqual(result["100"]["geometry"], per_frame_geometry)
        self.assertNotEqual(result["100"]["geometry"], canonical_geometry)
        self.assertNotIn("geometry_per_frame", result["100"])

        # Frame 1 has dynamic but not static (geometry_per_frame omits it).
        result = loader.get_object_data_for_frame(1, include_static=True)
        self.assertEqual(len(result), 1)
        self.assertIn("1", result)
        self.assertNotIn("100", result)

        # Test missing frame
        result = loader.get_object_data_for_frame(999, include_static=False)
        self.assertEqual(len(result), 0)

    def test_static_object_sharing_a_dynamic_id_is_reported(self) -> None:
        """A static whose ID collides with a dynamic one must not disappear silently.

        Dynamic and static objects share one key space in the merged view, so a duplicate ID replaces a real
        dynamic object and every occlusion-aware consumer (``SceneRasterizer`` and everything reading through
        it) then measures the wrong geometry. That is always an ID-allocation bug upstream, so the merge is
        required to say so rather than corrupt results quietly.
        """
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader._object_ego_distances = {}
        loader.all_dynamic_objects = {0: {"7": {"object_type": "Car"}}}
        loader.all_static_objects = {
            "7": {"object_type": "LaneLine", "is_static": True, "geometry_per_frame": {0: ("polyline", "f0")}}
        }

        with self.assertLogs("checks.utils.rds_data_loader", level="WARNING") as captured:
            result = loader.get_object_data_for_frame(0, include_static=True)

        self.assertEqual(len(result), 1)
        self.assertIn("shadows a dynamic object", "\n".join(captured.output))
        self.assertIn("LaneLine", "\n".join(captured.output))
        self.assertIn("Car", "\n".join(captured.output))

    def test_get_fps_object(self) -> None:
        """Test get_fps for object data type."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        result = loader.get_fps("object")
        self.assertEqual(result, 10.0)

    def test_get_fps_pose(self) -> None:
        """Test get_fps for pose data type."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        result = loader.get_fps("pose")
        self.assertEqual(result, 30.0)

    def test_get_fps_invalid(self) -> None:
        """Test get_fps with invalid data type."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        with self.assertRaises(ValueError) as context:
            loader.get_fps("invalid")
        self.assertIn("Invalid data_type", str(context.exception))

    def test_get_time_range(self) -> None:
        """Test get_time_range method."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.clip_id = "uuid_1000000"
        loader.all_dynamic_objects = {0: {}, 1: {}, 2: {}}  # 3 frames

        start_us, end_us = loader.get_time_range()

        self.assertEqual(start_us, 1000000)
        # 3 frames at 10 FPS = 0.3 seconds = 300000 microseconds
        self.assertEqual(end_us, 1000000 + 300000)

    def test_get_time_range_invalid_clip_id(self) -> None:
        """Test get_time_range with invalid clip_id format."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.clip_id = "invalid"
        loader.all_dynamic_objects = {}

        with self.assertRaises(ValueError) as context:
            loader.get_time_range()
        self.assertIn("Invalid clip_id format", str(context.exception))

    def test_get_time_range_invalid_timestamp(self) -> None:
        """Test get_time_range with non-numeric timestamp."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.clip_id = "test_uuid_not_a_number"
        loader.all_dynamic_objects = {}

        with self.assertRaises(ValueError) as context:
            loader.get_time_range()
        self.assertIn("Failed to parse timestamp", str(context.exception))

    def test_extract_shape_attributes_with_enums(self) -> None:
        """Test _extract_shape_attributes with enumsList."""
        shape3d = {
            "attributes": [
                {
                    "name": "colors",
                    "enumsList": {"enumsList": ["red", "yellow", "white"]},
                },
                {"name": "styles", "enumsList": {"enumsList": ["solid", "dashed"]}},
            ]
        }

        result = RdsDataLoader._extract_shape_attributes(shape3d)

        self.assertEqual(result["colors"], ["red", "yellow", "white"])
        self.assertEqual(result["styles"], ["solid", "dashed"])

    def test_extract_shape_attributes_with_text(self) -> None:
        """Test _extract_shape_attributes with text values."""
        shape3d = {
            "attributes": [
                {"name": "type", "text": "solid"},
                {"name": "material", "text": "concrete"},
            ]
        }

        result = RdsDataLoader._extract_shape_attributes(shape3d)

        self.assertEqual(result["type"], "solid")
        self.assertEqual(result["material"], "concrete")

    def test_extract_shape_attributes_empty(self) -> None:
        """Test _extract_shape_attributes with empty attributes."""
        shape3d: dict[str, Any] = {"attributes": []}
        result = RdsDataLoader._extract_shape_attributes(shape3d)
        self.assertEqual(result, {})

    def test_extract_shape_attributes_missing_name(self) -> None:
        """Test _extract_shape_attributes with missing name."""
        shape3d = {"attributes": [{"text": "value"}]}
        result = RdsDataLoader._extract_shape_attributes(shape3d)
        self.assertEqual(result, {})

    def test_extract_shape_attributes_none_text(self) -> None:
        """Test _extract_shape_attributes with None text."""
        shape3d = {"attributes": [{"name": "attr", "text": None}]}
        result = RdsDataLoader._extract_shape_attributes(shape3d)
        self.assertEqual(result, {})

    def test_parse_shape3d_entry_polyline(self) -> None:
        """Test _parse_shape3d_entry with polyline3d."""
        shape3d = {
            "polyline3d": {
                "vertices": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]],
            }
        }

        result = RdsDataLoader._parse_shape3d_entry(shape3d)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["geom_type"], "polyline")
        assert_array_almost_equal(result["vertices"], np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]]))

    def test_parse_shape3d_entry_cuboid(self) -> None:
        """Test _parse_shape3d_entry with cuboid3d."""
        shape3d = {
            "cuboid3d": {
                "vertices": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ],
            }
        }

        result = RdsDataLoader._parse_shape3d_entry(shape3d)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["geom_type"], "cuboid")
        self.assertEqual(len(result["vertices"]), 8)

    def test_parse_shape3d_entry_surface(self) -> None:
        """Test _parse_shape3d_entry with surface."""
        shape3d = {"surface": {"vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]}}

        result = RdsDataLoader._parse_shape3d_entry(shape3d)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["geom_type"], "surface")

    def test_parse_shape3d_entry_invalid(self) -> None:
        """Test _parse_shape3d_entry with invalid input."""
        self.assertIsNone(RdsDataLoader._parse_shape3d_entry(None))
        self.assertIsNone(RdsDataLoader._parse_shape3d_entry("not a dict"))
        self.assertIsNone(RdsDataLoader._parse_shape3d_entry({}))

    def test_parse_shape3d_entry_empty_vertices(self) -> None:
        """Test _parse_shape3d_entry with empty vertices."""
        shape3d: dict[str, Any] = {"polyline3d": {"vertices": []}}
        result = RdsDataLoader._parse_shape3d_entry(shape3d)
        self.assertIsNone(result)

    def test_parse_shape3d_entry_with_attributes(self) -> None:
        """Test _parse_shape3d_entry with attributes."""
        shape3d = {
            "polyline3d": {
                "vertices": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            },
            "attributes": [{"name": "color", "text": "yellow"}],
        }

        result = RdsDataLoader._parse_shape3d_entry(shape3d)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("attributes", result)
        self.assertEqual(result["attributes"]["color"], "yellow")

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_labels(self, mock_get_sample: MagicMock) -> None:
        """Test get_labels method."""
        label_data = self._create_mock_label_data()
        mock_get_sample.return_value = label_data

        tar_path = self.dataset_dir / "3d_lanelines" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_labels("3d_lanelines")

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["geom_type"], "polyline")
        self.assertEqual(result[1]["geom_type"], "cuboid")

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_labels_picks_canonical_when_per_frame_keys_present(self, mock_get_sample: MagicMock) -> None:
        """Regression: when a label tar bundles canonical and per-frame entries,
        ``get_labels`` must read the canonical even when per-frame keys come
        first in the sample dict.

        WebDataset groups every entry of a static label tar into one sample
        keyed by the file's filename suffix.  A typical tar-order iteration
        puts ``000000.<type>.json`` (the per-frame entry for frame 0) before
        ``<type>.json`` (the clip-level canonical), so a naive "first json
        key" pick would silently surface frame-0 data as if it were the
        canonical.
        """
        mock_get_sample.return_value = {
            "000000.crosswalks.json": {
                "labels": [
                    {
                        "labelData": {
                            "shape3d": {"surface": {"vertices": [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0]]}}
                        }
                    },
                ]
            },
            "000003.crosswalks.json": {
                "labels": [
                    {
                        "labelData": {
                            "shape3d": {"surface": {"vertices": [[3.0, 3.0, 0.0], [4.0, 3.0, 0.0], [4.0, 4.0, 0.0]]}}
                        }
                    },
                ]
            },
            "crosswalks.json": {
                "labels": [
                    {
                        "labelData": {
                            "shape3d": {
                                "surface": {
                                    "vertices": [
                                        [10.0, 10.0, 0.0],
                                        [11.0, 10.0, 0.0],
                                        [11.0, 11.0, 0.0],
                                        [10.0, 11.0, 0.0],
                                    ]
                                }
                            }
                        }
                    },
                ]
            },
        }

        tar_path = self.dataset_dir / "3d_crosswalks" / f"{self.clip_id}.tar"
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_labels("3d_crosswalks")

        # The canonical entry has 4 vertices; the per-frame entries have 3.
        # Asserting on vertex count proves we read the canonical, not a
        # per-frame entry that happened to come first in iteration order.
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["geom_type"], "surface")
        self.assertEqual(result[0]["vertices"].shape, (4, 3))

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_labels_returns_empty_when_only_per_frame_keys(self, mock_get_sample: MagicMock) -> None:
        """A label sample that contains only per-frame entries (no canonical)
        yields an empty list.  Phase 2 will add a separate per-frame reader
        so callers can opt in explicitly."""
        mock_get_sample.return_value = {
            "000000.crosswalks.json": {
                "labels": [
                    {
                        "labelData": {
                            "shape3d": {"surface": {"vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]}}
                        }
                    }
                ]
            },
        }
        tar_path = self.dataset_dir / "3d_crosswalks" / f"{self.clip_id}.tar"
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        self.assertEqual(loader.get_labels("3d_crosswalks"), [])

    def test_find_canonical_label_key(self) -> None:
        """``_find_canonical_label_key`` returns the non-frame-numbered json key."""
        sample = {
            "000000.crosswalks.json": {},
            "000003.crosswalks.json": {},
            "crosswalks.json": {},
            "__key__": "ignored",
            "__url__": "ignored.tar",
        }
        self.assertEqual(RdsDataLoader._find_canonical_label_key(sample), "crosswalks.json")

        # Per-frame-only sample yields None — the caller treats that as "no canonical present".
        self.assertIsNone(
            RdsDataLoader._find_canonical_label_key(
                {
                    "000000.crosswalks.json": {},
                    "000003.crosswalks.json": {},
                }
            )
        )

        # Empty sample yields None.
        self.assertIsNone(RdsDataLoader._find_canonical_label_key({}))

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_per_frame_labels_parses_indexed_entries(self, mock_get_sample: MagicMock) -> None:
        """``get_per_frame_labels`` returns parsed labels keyed by frame index.

        It must skip the canonical (non-frame-numbered) entry, only consume
        per-frame entries, and surface their parsed shape3d data the same
        way ``get_labels`` does for the canonical.
        """
        mock_get_sample.return_value = {
            "crosswalks.json": {
                "labels": [
                    {
                        "labelData": {
                            "shape3d": {
                                "surface": {
                                    "vertices": [
                                        [0.0, 0.0, 0.0],
                                        [1.0, 0.0, 0.0],
                                        [1.0, 1.0, 0.0],
                                        [0.0, 1.0, 0.0],
                                    ]
                                }
                            }
                        }
                    }
                ]
            },
            "000000.crosswalks.json": {
                "labels": [
                    {
                        "labelData": {
                            "shape3d": {
                                "surface": {
                                    "vertices": [
                                        [0.0, 0.0, 0.0],
                                        [2.0, 0.0, 0.0],
                                        [2.0, 2.0, 0.0],
                                    ]
                                }
                            }
                        }
                    }
                ]
            },
            "000003.crosswalks.json": {
                "labels": [
                    {
                        "labelData": {
                            "shape3d": {
                                "surface": {
                                    "vertices": [
                                        [10.0, 10.0, 0.0],
                                        [11.0, 10.0, 0.0],
                                        [11.0, 11.0, 0.0],
                                    ]
                                }
                            }
                        }
                    }
                ]
            },
        }
        tar_path = self.dataset_dir / "3d_crosswalks" / f"{self.clip_id}.tar"
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_per_frame_labels("3d_crosswalks")

        self.assertEqual(set(result), {0, 3})
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(result[0][0]["geom_type"], "surface")
        self.assertEqual(result[0][0]["vertices"].shape, (3, 3))
        self.assertEqual(result[3][0]["vertices"].shape, (3, 3))

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_per_frame_labels_empty_when_only_canonical(self, mock_get_sample: MagicMock) -> None:
        """Tars containing only a canonical entry yield ``{}`` from
        ``get_per_frame_labels`` — Phase 2 then falls back to canonical
        replication for the type."""
        mock_get_sample.return_value = {
            "traffic_lights.json": {"labels": []},
        }
        tar_path = self.dataset_dir / "3d_traffic_lights" / f"{self.clip_id}.tar"
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        self.assertEqual(loader.get_per_frame_labels("3d_traffic_lights"), {})

    @patch.object(RdsDataLoader, "get_per_frame_labels")
    def test_frame_indices_from_per_frame_static_unions_across_folders(
        self, mock_get_per_frame_labels: MagicMock
    ) -> None:
        """``_frame_indices_from_per_frame_static`` returns the sorted union
        of frame indices observed across all six static label tars.  Types
        without per-frame data simply contribute nothing."""

        def fake_per_frame(folder: str) -> dict[int, list[dict[str, Any]]]:
            # Different folders contribute different (overlapping) frame sets.
            if folder == "3d_lanelines":
                return {0: [], 3: [], 6: []}
            if folder == "3d_road_boundaries":
                return {3: [], 9: []}
            if folder == "3d_crosswalks":
                return {0: [], 12: []}
            return {}

        mock_get_per_frame_labels.side_effect = fake_per_frame

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"

        self.assertEqual(loader._frame_indices_from_per_frame_static(), [0, 3, 6, 9, 12])

    @patch.object(RdsDataLoader, "get_per_frame_labels")
    @patch.object(RdsDataLoader, "get_labels")
    @patch.object(RdsDataLoader, "_get_all_traffic_light_status", return_value={})
    def test_get_static_objects_uses_per_frame_for_typed_with_per_frame_files(
        self,
        _mock_status: MagicMock,
        mock_get_labels: MagicMock,
        mock_get_per_frame_labels: MagicMock,
    ) -> None:
        """Per-frame parsing path: when a type's tar carries per-frame entries,
        the static-object tracker matches them to the canonical and the
        resulting ``geometry_per_frame`` carries per-frame instances (not
        the canonical replicated)."""

        # Single canonical lane line, plus two per-frame detections that
        # geometrically match the canonical (within MATCH_GATE_METERS) and
        # one that doesn't (giving a synthesized track).
        canonical_vertices = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
        per_frame_match_0 = np.array([[0.1, 0.1, 0.0], [10.1, 0.1, 0.0]], dtype=np.float32)
        per_frame_match_3 = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)
        per_frame_far_3 = np.array([[500.0, 500.0, 0.0], [510.0, 500.0, 0.0]], dtype=np.float32)

        def fake_get_labels(folder: str) -> list[dict[str, Any]]:
            if folder == "3d_lanelines":
                return [{"vertices": canonical_vertices, "geom_type": "polyline"}]
            return []

        def fake_get_per_frame_labels(folder: str) -> dict[int, list[dict[str, Any]]]:
            if folder == "3d_lanelines":
                return {
                    0: [{"vertices": per_frame_match_0, "geom_type": "polyline"}],
                    3: [
                        {"vertices": per_frame_match_3, "geom_type": "polyline"},
                        {"vertices": per_frame_far_3, "geom_type": "polyline"},
                    ],
                }
            return {}

        mock_get_labels.side_effect = fake_get_labels
        mock_get_per_frame_labels.side_effect = fake_get_per_frame_labels

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"
        loader._cached_canonical_static_frame_axis = [0, 3]

        result = loader._get_static_objects()

        # Canonical track keeps id "0"; the unmatched detection at frame 3
        # spawns a synthesized track at id "1".
        self.assertEqual(set(result), {"0", "1"})

        canonical_track = result["0"]
        self.assertEqual(canonical_track["object_type"], "LaneLine")
        # Canonical geometry is preserved (frame-independent).
        self.assertIsNotNone(canonical_track["geometry"])
        # Per-frame entries are the per-frame instances, NOT the canonical replicated across frames. Compare
        # vertices rather than identity: ``geometry`` is now the greatest-coverage observation, so it may
        # legitimately *be* one of the per-frame instances rather than the canonical label's geometry.
        self.assertIn(0, canonical_track["geometry_per_frame"])
        self.assertIn(3, canonical_track["geometry_per_frame"])
        np.testing.assert_allclose(
            canonical_track["geometry_per_frame"][0].vertices_world, per_frame_match_0, rtol=1e-6
        )
        np.testing.assert_allclose(
            canonical_track["geometry_per_frame"][3].vertices_world, per_frame_match_3, rtol=1e-6
        )
        self.assertIsNot(canonical_track["geometry_per_frame"][0], canonical_track["geometry_per_frame"][3])

        # Synthesized track exists at frame 3 only and adopts that frame's
        # geometry as its canonical.
        synthesized = result["1"]
        self.assertEqual(synthesized["object_type"], "LaneLine")
        self.assertIsNone(synthesized["object_type_index"])
        self.assertEqual(set(synthesized["geometry_per_frame"]), {3})
        self.assertIs(synthesized["geometry"], synthesized["geometry_per_frame"][3])

    @patch.object(RdsDataLoader, "get_per_frame_labels", return_value={})
    @patch.object(RdsDataLoader, "get_labels")
    @patch.object(RdsDataLoader, "_get_all_traffic_light_status", return_value={})
    def test_get_static_objects_replicates_canonical_when_no_per_frame(
        self,
        _mock_status: MagicMock,
        mock_get_labels: MagicMock,
        _mock_per_frame: MagicMock,
    ) -> None:
        """Canonical-only fallback: when ``get_per_frame_labels`` returns
        ``{}`` for every folder, each canonical object's ``geometry_per_frame``
        is the canonical instance replicated across the structural axis
        (preserves Phase 1 behavior)."""
        canonical_vertices = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)

        def fake_get_labels(folder: str) -> list[dict[str, Any]]:
            if folder == "3d_lanelines":
                return [{"vertices": canonical_vertices, "geom_type": "polyline"}]
            return []

        mock_get_labels.side_effect = fake_get_labels

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"
        loader._cached_canonical_static_frame_axis = [0, 3, 6]

        result = loader._get_static_objects()

        self.assertEqual(set(result), {"0"})
        track = result["0"]
        self.assertEqual(set(track["geometry_per_frame"]), {0, 3, 6})
        # Every per-frame entry is the canonical instance (Phase 1 fallback).
        for fi in (0, 3, 6):
            self.assertIs(track["geometry_per_frame"][fi], track["geometry"])

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_labels_file_not_found(self, mock_get_sample: MagicMock) -> None:
        """Test get_labels when tar file doesn't exist."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_labels("nonexistent_folder")

        self.assertEqual(result, [])
        mock_get_sample.assert_not_called()

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_labels_no_json_key(self, mock_get_sample: MagicMock) -> None:
        """Test get_labels when no JSON key is found."""
        mock_get_sample.return_value = {"other_file.npy": np.array([1, 2, 3])}

        tar_path = self.dataset_dir / "3d_lanelines" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_labels("3d_lanelines")

        self.assertEqual(result, [])

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_labels_invalid_data_structure(self, mock_get_sample: MagicMock) -> None:
        """Test get_labels with invalid data structure."""
        mock_get_sample.return_value = {"labels.json": "not a dict"}

        tar_path = self.dataset_dir / "3d_lanelines" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_labels("3d_lanelines")

        self.assertEqual(result, [])

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_labels_missing_label_data(self, mock_get_sample: MagicMock) -> None:
        """Test get_labels with missing labelData."""
        mock_get_sample.return_value = {
            "labels.json": {
                "labels": [
                    {"other_field": "value"},  # Missing labelData
                    {
                        "labelData": {
                            "shape3d": {
                                "polyline3d": {
                                    "vertices": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                                }
                            }
                        }
                    },
                ]
            }
        }

        tar_path = self.dataset_dir / "3d_lanelines" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_labels("3d_lanelines")

        # Should only include the label with labelData
        self.assertEqual(len(result), 1)

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_labels_missing_shape3d(self, mock_get_sample: MagicMock) -> None:
        """Test get_labels with missing shape3d."""
        mock_get_sample.return_value = {
            "labels.json": {
                "labels": [
                    {
                        "labelData": {
                            "other_field": "value"  # Missing shape3d
                        }
                    },
                ]
            }
        }

        tar_path = self.dataset_dir / "3d_lanelines" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_labels("3d_lanelines")

        self.assertEqual(result, [])

    @patch.object(RdsDataLoader, "get_sample")
    def test_get_labels_skips_invalid_parsed_entries(self, mock_get_sample: MagicMock) -> None:
        """Test get_labels skips entries that fail to parse."""
        mock_get_sample.return_value = {
            "labels.json": {
                "labels": [
                    {
                        "labelData": {
                            "shape3d": {
                                "polyline3d": {
                                    "vertices": [],  # Empty vertices should return None
                                }
                            }
                        }
                    },
                    {
                        "labelData": {
                            "shape3d": {
                                "polyline3d": {
                                    "vertices": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                                }
                            }
                        }
                    },
                ]
            }
        }

        tar_path = self.dataset_dir / "3d_lanelines" / f"{self.clip_id}.tar"
        tar_path.touch()

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_labels("3d_lanelines")

        # Should only include the valid entry
        self.assertEqual(len(result), 1)

    @patch.object(RdsDataLoader, "get_labels")
    def test_get_lanelines(self, mock_get_labels: MagicMock) -> None:
        """Test get_lanelines method."""
        mock_get_labels.return_value = [{"vertices": np.array([[0, 0, 0]]), "geom_type": "polyline"}]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_lanelines()

        mock_get_labels.assert_called_once_with("3d_lanelines")
        self.assertEqual(len(result), 1)

    @patch.object(RdsDataLoader, "get_labels")
    def test_get_road_boundaries(self, mock_get_labels: MagicMock) -> None:
        """Test get_road_boundaries method."""
        mock_get_labels.return_value = []

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        result = loader.get_road_boundaries()

        mock_get_labels.assert_called_once_with("3d_road_boundaries")
        self.assertEqual(result, [])

    @patch.object(RdsDataLoader, "get_labels")
    def test_get_wait_lines(self, mock_get_labels: MagicMock) -> None:
        """Test get_wait_lines method."""
        mock_get_labels.return_value = []

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        loader.get_wait_lines()

        mock_get_labels.assert_called_once_with("3d_wait_lines")

    @patch.object(RdsDataLoader, "get_labels")
    def test_get_crosswalks(self, mock_get_labels: MagicMock) -> None:
        """Test get_crosswalks method."""
        mock_get_labels.return_value = []

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.CROSSWALK_FOLDER = "3d_crosswalks"
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        loader.get_crosswalks()

        mock_get_labels.assert_called_once_with("3d_crosswalks")

    @patch.object(RdsDataLoader, "get_labels")
    def test_get_traffic_lights(self, mock_get_labels: MagicMock) -> None:
        """Test get_traffic_lights method."""
        mock_get_labels.return_value = []

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        loader.get_traffic_lights()

        mock_get_labels.assert_called_once_with("3d_traffic_lights")

    @patch.object(RdsDataLoader, "get_labels")
    def test_get_traffic_signs(self, mock_get_labels: MagicMock) -> None:
        """Test get_traffic_signs method."""
        mock_get_labels.return_value = []

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        loader.get_traffic_signs()

        mock_get_labels.assert_called_once_with("3d_traffic_signs")

    @patch("numpy.savetxt")
    @patch.object(RdsDataLoader, "_get_camera_intrinsics_array")
    def test_dump_camera_intrinsics_to_csv_success(self, mock_get_array: MagicMock, mock_savetxt: MagicMock) -> None:
        """Test dump_camera_intrinsics_to_csv with successful dump."""
        mock_intrinsic = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
        mock_get_array.return_value = mock_intrinsic

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        output_dir = Path(self.temp_dir) / "output"
        success = loader.dump_camera_intrinsics_to_csv("camera_front", str(output_dir))

        self.assertTrue(success["pinhole"])
        self.assertTrue(success["ftheta"])
        self.assertEqual(mock_savetxt.call_count, 2)

    @patch("numpy.savetxt")
    @patch.object(RdsDataLoader, "_get_camera_intrinsics_array")
    def test_dump_camera_intrinsics_to_csv_failure(self, mock_get_array: MagicMock, mock_savetxt: MagicMock) -> None:
        """Test dump_camera_intrinsics_to_csv with failure."""
        mock_get_array.side_effect = [np.array([[1, 2], [3, 4]]), FileNotFoundError("Not found")]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        output_dir = Path(self.temp_dir) / "output"
        success = loader.dump_camera_intrinsics_to_csv("camera_front", str(output_dir))

        self.assertTrue(success["pinhole"])
        self.assertFalse(success["ftheta"])

    @patch("numpy.savetxt")
    @patch.object(RdsDataLoader, "get_camera_poses")
    def test_dump_camera_frame_egomotion_to_csv(self, mock_get_poses: MagicMock, mock_savetxt: MagicMock) -> None:
        """Test dump_camera_frame_egomotion_to_csv method."""
        pose_data = {}
        for frame_idx in range(6):  # 6 frames at 30 FPS
            pose = np.eye(4)
            pose[:3, 3] = [frame_idx * 0.1, 0.0, 0.0]
            pose_data[frame_idx] = pose
        mock_get_poses.return_value = pose_data

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        output_dir = Path(self.temp_dir) / "output"
        loader.dump_camera_frame_egomotion_to_csv("camera_front", 10.0, str(output_dir))

        # Should write CSV with frames sampled at 10 FPS (every 3rd frame from 30 FPS)
        mock_savetxt.assert_called_once()
        call_args = mock_savetxt.call_args
        rows = call_args[0][1]  # Second argument is the data
        self.assertEqual(len(rows), 2)  # 6 frames / 3 = 2 frames at 10 FPS

    def test_dump_camera_frame_egomotion_to_csv_invalid_fps_too_high(self) -> None:
        """Test dump_camera_frame_egomotion_to_csv with FPS > 30."""
        loader = RdsDataLoader.__new__(RdsDataLoader)

        with self.assertRaises(ValueError) as context:
            loader.dump_camera_frame_egomotion_to_csv("camera_front", 31.0, self.temp_dir)
        self.assertIn("Invalid video FPS", str(context.exception))

    def test_dump_camera_frame_egomotion_to_csv_invalid_fps_zero(self) -> None:
        """Test dump_camera_frame_egomotion_to_csv with FPS <= 0."""
        loader = RdsDataLoader.__new__(RdsDataLoader)

        with self.assertRaises(ValueError) as context:
            loader.dump_camera_frame_egomotion_to_csv("camera_front", 0.0, self.temp_dir)
        self.assertIn("Invalid video FPS", str(context.exception))

    def test_dump_camera_frame_egomotion_to_csv_invalid_fps_not_divisible(self) -> None:
        """Test dump_camera_frame_egomotion_to_csv with FPS that doesn't divide 30."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.dataset_dir = self.dataset_dir
        loader.clip_id = self.clip_id

        with patch.object(RdsDataLoader, "get_camera_poses") as mock_get_poses:
            mock_get_poses.return_value = {0: np.eye(4)}

            with self.assertRaises(ValueError) as context:
                loader.dump_camera_frame_egomotion_to_csv("camera_front", 7.0, self.temp_dir)
            self.assertIn("Invalid video FPS", str(context.exception))

    @patch("checks.utils.rds_data_loader.logger")
    @patch.object(RdsDataLoader, "get_labels")
    @patch.object(RdsDataLoader, "get_per_frame_labels", return_value={})
    @patch.object(RdsDataLoader, "_get_all_traffic_light_status", return_value={})
    def test_get_static_objects_handles_geometry_instantiation_exception(
        self,
        mock_get_traffic_light_status: MagicMock,
        mock_get_per_frame_labels: MagicMock,
        mock_get_labels: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Test get_static_objects catches and logs exceptions during geometry instantiation."""
        # First label has invalid vertices that will cause Polyline to raise
        # Second label is valid and should still be processed
        mock_get_labels.side_effect = [
            [
                {
                    "vertices": np.array([[0.0, 0.0]]),  # Invalid: only 2D points, may cause issue
                    "geom_type": "polyline",
                },
                {
                    "vertices": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
                    "geom_type": "polyline",
                },
            ],
            [],  # RoadBoundary
            [],  # WaitLine
            [],  # TrafficLight
            [],  # TrafficSign
            [],  # Crosswalk
        ]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"
        loader._cached_canonical_static_frame_axis = [0]

        # Mock Polyline to raise on first call (invalid vertices) but succeed on second
        with patch("checks.utils.rds_data_loader.Polyline") as mock_polyline:
            mock_polyline.side_effect = [ValueError("Invalid vertices"), Mock()]

            result = loader._get_static_objects()

            # Should have logged a warning for the failed instantiation
            mock_logger.warning.assert_called_once()
            warning_args = mock_logger.warning.call_args.args
            self.assertIn("Failed to instantiate", warning_args[0])
            self.assertEqual(warning_args[1], "polyline")
            self.assertEqual(warning_args[2], "LaneLine")

            # Should still have processed the second valid label
            self.assertEqual(len(result), 1)
            self.assertEqual(next(iter(result.values()))["object_type"], "LaneLine")

    @patch("checks.utils.rds_data_loader.logger")
    @patch.object(RdsDataLoader, "get_labels")
    @patch.object(RdsDataLoader, "get_per_frame_labels", return_value={})
    @patch.object(RdsDataLoader, "_get_all_traffic_light_status", return_value={})
    def test_get_static_objects_handles_cuboid_exception(
        self,
        mock_get_traffic_light_status: MagicMock,
        mock_get_per_frame_labels: MagicMock,
        mock_get_labels: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Test get_static_objects catches and logs exceptions during Cuboid instantiation."""
        # Invalid cuboid vertices that will cause compute_pose_and_lwh_from_corners to fail
        invalid_cuboid_vertices = np.array([[0.0, 0.0, 0.0]])  # Too few vertices for a cuboid

        mock_get_labels.side_effect = [
            [],  # LaneLine
            [],  # RoadBoundary
            [],  # WaitLine
            [
                {
                    "vertices": invalid_cuboid_vertices,
                    "geom_type": "cuboid",
                },
            ],  # TrafficLight
            [],  # TrafficSign
            [],  # Crosswalk
        ]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"
        loader._cached_canonical_static_frame_axis = [0]

        with patch("checks.utils.rds_data_loader.Cuboid") as mock_cuboid:
            mock_cuboid.compute_pose_and_lwh_from_corners.side_effect = ValueError("Invalid cuboid vertices")

            result = loader._get_static_objects()

            # Should have logged a warning
            mock_logger.warning.assert_called_once()
            warning_args = mock_logger.warning.call_args.args
            self.assertIn("Failed to instantiate", warning_args[0])
            self.assertEqual(warning_args[1], "cuboid")
            self.assertEqual(warning_args[2], "TrafficLight")

            # Should return empty since the only label failed
            self.assertEqual(len(result), 0)

    @patch("checks.utils.rds_data_loader.logger")
    @patch.object(RdsDataLoader, "get_labels")
    @patch.object(RdsDataLoader, "get_per_frame_labels", return_value={})
    @patch.object(RdsDataLoader, "_get_all_traffic_light_status", return_value={})
    def test_get_static_objects_handles_surface_exception(
        self,
        mock_get_traffic_light_status: MagicMock,
        mock_get_per_frame_labels: MagicMock,
        mock_get_labels: MagicMock,
        mock_logger: MagicMock,
    ) -> None:
        """Test get_static_objects catches and logs exceptions during Surface instantiation."""
        mock_get_labels.side_effect = [
            [],  # LaneLine
            [],  # RoadBoundary
            [],  # WaitLine
            [],  # TrafficLight
            [],  # TrafficSign
            [
                {
                    "vertices": np.array([[0.0, 0.0, 0.0]]),  # Single point, invalid for surface
                    "geom_type": "surface",
                },
            ],  # Crosswalk
        ]

        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader.object_count = 0
        loader.LANE_LINE_FOLDER = "3d_lanelines"
        loader.ROAD_BOUNDARY_FOLDER = "3d_road_boundaries"
        loader.WAIT_LINE_FOLDER = "3d_wait_lines"
        loader.TRAFFIC_LIGHT_FOLDER = "3d_traffic_lights"
        loader.TRAFFIC_SIGN_FOLDER = "3d_traffic_signs"
        loader.CROSSWALK_FOLDER = "3d_crosswalks"
        loader._cached_canonical_static_frame_axis = [0]

        with patch("checks.utils.rds_data_loader.Surface") as mock_surface:
            mock_surface.side_effect = RuntimeError("Cannot create surface from single point")

            result = loader._get_static_objects()

            # Should have logged a warning
            mock_logger.warning.assert_called_once()
            warning_args = mock_logger.warning.call_args.args
            self.assertIn("Failed to instantiate", warning_args[0])
            self.assertEqual(warning_args[1], "surface")
            self.assertEqual(warning_args[2], "Crosswalk")

            # Should return empty since the only label failed
            self.assertEqual(len(result), 0)

    def test_get_object_data_for_frame_does_not_modify_class_members(self) -> None:
        """Test that get_object_data_for_frame does not expose internal state to mutation."""
        loader = RdsDataLoader.__new__(RdsDataLoader)
        loader._object_ego_distances = {}
        loader.all_dynamic_objects = {
            0: {
                "0": {"object_type": "Car", "score": 0.9},
                "1": {"object_type": "Pedestrian", "score": 0.8},
            },
        }
        loader.all_static_objects = {
            "100": {
                "object_type": "LaneLine",
                "is_static": True,
                "tags": ["a", "b"],
                "geometry_per_frame": {0: ("polyline-a",)},
            },
            "101": {
                "object_type": "RoadBoundary",
                "is_static": True,
                "tags": ["c"],
                "geometry_per_frame": {0: ("polyline-b",)},
            },
        }

        # Snapshot the original state for later comparison
        original_dynamic = copy.deepcopy(loader.all_dynamic_objects)
        original_static = copy.deepcopy(loader.all_static_objects)

        # --- dynamic-only call ---
        result_dynamic = loader.get_object_data_for_frame(0, include_static=False)

        # Mutate the returned dict: add a key, modify a nested value, delete a key
        result_dynamic["new_key"] = {"object_type": "Truck"}
        result_dynamic["0"]["score"] = 0.0
        result_dynamic["0"]["extra"] = "injected"
        del result_dynamic["1"]

        # The underlying class members must be unchanged
        self.assertEqual(loader.all_dynamic_objects, original_dynamic)
        self.assertEqual(loader.all_static_objects, original_static)

        # --- include_static=True call ---
        result_with_static = loader.get_object_data_for_frame(0, include_static=True)

        # Mutate the returned static entries
        result_with_static["100"]["object_type"] = "MUTATED"
        result_with_static["100"]["tags"].append("MUTATED")
        result_with_static["101"]["new_field"] = 42
        del result_with_static["0"]

        # The underlying class members must still be unchanged
        self.assertEqual(loader.all_dynamic_objects, original_dynamic)
        self.assertEqual(loader.all_static_objects, original_static)

    def test_class_to_object_types_mapping(self) -> None:
        """Test CLASS_TO_OBJECT_TYPES class variable."""
        self.assertIn("vehicle", RdsDataLoader.CLASS_TO_OBJECT_TYPES)
        self.assertIn("pedestrian", RdsDataLoader.CLASS_TO_OBJECT_TYPES)
        self.assertIn("motorcycle", RdsDataLoader.CLASS_TO_OBJECT_TYPES)
        self.assertIn("bicycle", RdsDataLoader.CLASS_TO_OBJECT_TYPES)
        self.assertIn("traffic_light", RdsDataLoader.CLASS_TO_OBJECT_TYPES)
        self.assertIn("traffic_sign", RdsDataLoader.CLASS_TO_OBJECT_TYPES)
        self.assertIn("lane_line", RdsDataLoader.CLASS_TO_OBJECT_TYPES)
        self.assertIn("road_boundary", RdsDataLoader.CLASS_TO_OBJECT_TYPES)
        self.assertIn("wait_line", RdsDataLoader.CLASS_TO_OBJECT_TYPES)
        self.assertIn("crosswalk", RdsDataLoader.CLASS_TO_OBJECT_TYPES)

        # Verify vehicle types
        vehicle_types = RdsDataLoader.CLASS_TO_OBJECT_TYPES["vehicle"]
        self.assertIn("Car", vehicle_types)
        self.assertIn("Truck", vehicle_types)


if __name__ == "__main__":
    unittest.main()
