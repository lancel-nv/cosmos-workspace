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

"""Unit tests for dynamic objects processor: ObjectProcessorDynamic."""

import shutil
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from checks.objects.single_view.object_processor_dynamic import ObjectProcessorDynamic
from checks.objects.single_view.seg_mask_cache import SegMaskCache
from checks.utils.frame_rates import OBJECT_CHECK_FPS, ObjectCheckTimeline
from utils.cancellation import ProcessingCancelled


class TestObjectProcessorDynamic(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.clip_id = "clip_dyn"
        self.config = {
            "av.objects_single_view": {
                "overlap_check": {"vehicle": {}, "pedestrian": {}},
                "importance_filter": {"distance_threshold_m": 50.0},
                "hallucination_detector": {
                    "enabled": True,
                    "classes": {"vehicle": {"min_cluster_area": 5000}, "pedestrian": {"min_cluster_area": 1000}},
                    "max_cluster_per_frame": 100,
                },
            }
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    @patch("checks.objects.single_view.object_processor_dynamic.SegHelper")
    @patch("checks.objects.single_view.object_processor_dynamic.OverlapDetector")
    @patch("checks.objects.single_view.object_processor_dynamic.HallucinationDetector")
    def test_init_components(self, mock_hall: MagicMock, mock_overlap: MagicMock, mock_seg: MagicMock) -> None:
        p = ObjectProcessorDynamic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu", verbose="INFO"
        )
        mock_seg.assert_called_once()
        mock_overlap.assert_called_once()
        mock_hall.assert_called_once()
        self.assertIsNotNone(getattr(p, "hallucination_detector", None))

    @patch("checks.objects.single_view.object_processor_dynamic.SceneRasterizer")
    @patch("checks.objects.single_view.object_processor_dynamic.extract_rpy_in_flu", return_value=(0.0, 0.0, 0.0))
    def test_process_frame_calls_overlap_and_hallucination(
        self, _mock_rpy: MagicMock, _mock_rasterizer: MagicMock
    ) -> None:
        p = ObjectProcessorDynamic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu"
        )
        # Plug in mocks
        p.seg_model = MagicMock()
        p.seg_model.process_frame.return_value = MagicMock()
        p.seg_model.resize_masks.return_value = MagicMock()
        p.overlap_detector = MagicMock()
        # Each class returns its own scores
        p.overlap_detector.process_class.side_effect = [({1: 0.6}, {}, {}), ({3: 0.9}, {}, {})]

        p.hallucination_detector = MagicMock()
        with patch("checks.objects.single_view.object_processor_dynamic.track_hallucinations") as mock_track:
            res = {
                "hallucination_detections": {
                    "vehicle": [{"frame_idx": 0, "bbox_xywh": [100, 100, 50, 50], "mask_ratio": 0.8}],
                    "pedestrian": [],
                    "motorcycle": [],
                    "bicycle": [],
                }
            }
            p.hallucination_detector.detect.return_value = res["hallucination_detections"]
            data_loader = MagicMock(CAMERA_RESCALED_RESOLUTION_HEIGHT=480, CAMERA_RESCALED_RESOLUTION_WIDTH=640)
            data_loader.get_object_data_for_frame.return_value = {"t1": {}}
            scores = p._process_frame(
                frame_idx=0,
                frame=MagicMock(),
                data_loader=data_loader,
                camera_pose=np.eye(4),
                camera_model=MagicMock(),
                track_output_agg={},
                results=res,
            )
            self.assertEqual(scores, {1: 0.6, 3: 0.9})
            mock_track.assert_called()

    @patch("checks.objects.single_view.object_processor_dynamic.SceneRasterizer")
    def test_process_frames_dynamic_iteration_and_mapping(self, _mock_rasterizer: MagicMock) -> None:
        p = ObjectProcessorDynamic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu"
        )
        p.overlap_detector = MagicMock()
        p.overlap_detector.process_class.return_value = ({2: 0.5}, {}, {})
        p.seg_model = MagicMock()
        p.seg_model.process_frame.return_value = MagicMock()
        p.seg_model.resize_masks.return_value = MagicMock()

        # A 10 fps camera video: one video frame per object-check step, so 3 frames give 3 steps addressing RDS
        # indices 0, 3 and 6.
        video_dataloader = [MagicMock(), MagicMock(), MagicMock()]
        camera_poses = {i: np.eye(4) for i in range(7)}
        data_loader = MagicMock()
        data_loader.CAMERA_RESCALED_RESOLUTION_HEIGHT = 480
        data_loader.CAMERA_RESCALED_RESOLUTION_WIDTH = 640
        data_loader.object_count = 10

        # Objects exist at RDS indices 0 and 6 (steps 0 and 2); step 1 (index 3) has none and must be skipped.
        def get_object_data_side_effect(idx: int, include_static: bool = False) -> dict:
            objects_by_frame: dict[int, dict] = {0: {"t": {}}, 6: {"t": {}}}
            return objects_by_frame.get(idx, {})

        data_loader.get_object_data_for_frame.side_effect = get_object_data_side_effect

        timeline = ObjectCheckTimeline.build(
            camera_video_fps=10.0, camera_frame_count=len(video_dataloader), pose_frame_count=7
        )
        self.assertEqual(timeline.num_steps, 3)

        results = p._init_results_container(total_video_frames=timeline.num_steps)
        track_output_agg: dict[int, dict] = {}
        score_matrix, _importance_matrix, skipped = p._process_frames_dynamic(
            video_dataloader=video_dataloader,
            camera_poses=camera_poses,
            camera_model=MagicMock(),
            data_loader=data_loader,
            timeline=timeline,
            results=results,
            track_output_agg=track_output_agg,
        )
        self.assertEqual(skipped, 1)
        # Rows and processed ids are object-check steps, not source video frames.
        self.assertIn(0, results["processed_frame_ids"])
        self.assertIn(2, results["processed_frame_ids"])
        self.assertNotIn(1, results["processed_frame_ids"])
        self.assertFalse(np.isnan(score_matrix[0, 2]))
        self.assertTrue(np.isnan(score_matrix[1, 2]))

    @patch("checks.objects.single_view.object_processor_dynamic.SceneRasterizer")
    @patch("checks.objects.single_view.object_processor_base.RdsDataLoader")
    def test_process_clip_end_to_end_mocked(self, mock_loader_cls: MagicMock, _mock_rasterizer: MagicMock) -> None:
        p = ObjectProcessorDynamic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu"
        )

        # Mock loader
        loader = MagicMock()
        camera_poses = {i: np.eye(4) for i in range(4)}
        loader.load_camera_data.return_value = (camera_poses, MagicMock())
        loader.load_ego_poses.return_value = camera_poses
        loader.all_dynamic_objects = {0: {}, 3: {}}
        loader.CAMERA_RESCALED_RESOLUTION_HEIGHT = 480
        loader.CAMERA_RESCALED_RESOLUTION_WIDTH = 640
        loader.object_count = 10
        mock_loader_cls.return_value = loader

        # Mock seg model dataloader
        p.seg_model = MagicMock()
        mock_video_dataloader = [MagicMock(), MagicMock()]
        mock_dataset = MagicMock()
        mock_dataset.number_frames = len(mock_video_dataloader)
        mock_dataset.frames_per_second = 10.0
        p.seg_model.get_video_dataloader.return_value = (mock_video_dataloader, mock_dataset)

        # Ensure processing returns a score for first frame only
        p.overlap_detector = MagicMock()
        p.overlap_detector.process_class.return_value = ({5: 0.8}, {}, {})
        loader.get_object_data_for_frame.side_effect = lambda idx, include_static=False: {"t": {}} if idx == 0 else {}

        p.profiler = MagicMock()
        res = p.process_clip(
            input_data="/data",
            clip_id="c1",
            camera_name="front",
            video_path="/vid.mp4",
            output_dir=self.temp_dir,
        )
        self.assertIn("processed_frames", res)
        self.assertIn("score_matrix", res)
        # total_video_frames is the source-video axis (what the persisted schema documents and what every result
        # consumer reads); the evaluation length is reported separately.
        self.assertEqual(res["total_video_frames"], len(mock_video_dataloader))
        self.assertEqual(res["total_evaluation_steps"], 2)
        self.assertEqual(res["evaluation_fps"], OBJECT_CHECK_FPS)
        p.seg_model.release_arena_memory.assert_called_once_with()

    @patch("checks.objects.single_view.object_processor_dynamic.SegHelper")
    @patch("checks.objects.single_view.object_processor_base.RdsDataLoader")
    def test_process_clip_releases_arena_memory_when_the_frame_loop_fails(
        self, mock_loader_cls: MagicMock, _mock_seg: MagicMock
    ) -> None:
        """A job that aborts mid-clip strands its arena regions just as a finished one does, so it releases them too."""
        p = ObjectProcessorDynamic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu"
        )

        loader = MagicMock()
        camera_poses = {i: np.eye(4) for i in range(4)}
        loader.load_camera_data.return_value = (camera_poses, MagicMock())
        loader.load_ego_poses.return_value = camera_poses
        loader.all_dynamic_objects = {0: {}, 3: {}}
        loader.CAMERA_RESCALED_RESOLUTION_HEIGHT = 480
        loader.CAMERA_RESCALED_RESOLUTION_WIDTH = 640
        mock_loader_cls.return_value = loader

        p.seg_model = MagicMock()
        mock_dataset = MagicMock(number_frames=2, frames_per_second=10.0)
        p.seg_model.get_video_dataloader.return_value = ([MagicMock(), MagicMock()], mock_dataset)
        p.profiler = MagicMock()

        with (
            patch.object(p, "_process_frames_dynamic", side_effect=RuntimeError("Available memory of 16920064")),
            self.assertRaisesRegex(RuntimeError, "Available memory of 16920064"),
        ):
            p.process_clip(
                input_data="/data",
                clip_id="c1",
                camera_name="front",
                video_path="/vid.mp4",
                output_dir=self.temp_dir,
            )

        p.seg_model.release_arena_memory.assert_called_once_with()

    @patch("checks.objects.single_view.object_processor_dynamic.SceneRasterizer")
    @patch("checks.objects.single_view.object_processor_base.RdsDataLoader")
    def test_process_clip_cancels_before_video_dataset_build(
        self, mock_loader_cls: MagicMock, _mock_rasterizer: MagicMock
    ) -> None:
        """An already-cancelled request aborts right after data loading, before the prologue's expensive stages.

        Asserting only that ``ProcessingCancelled`` is raised would still pass if the checkpoint sat at the first
        per-frame loop, so pin the position: building the video dataset must never happen.
        """
        p = ObjectProcessorDynamic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu"
        )

        loader = MagicMock()
        loader.load_camera_data.return_value = ({0: np.eye(4)}, MagicMock())
        loader.load_ego_poses.return_value = {0: np.eye(4)}
        mock_loader_cls.return_value = loader

        p.seg_model = MagicMock()
        p.profiler = MagicMock()

        cancel_event = threading.Event()
        cancel_event.set()

        with self.assertRaises(ProcessingCancelled):
            p.process_clip(
                input_data="/data",
                clip_id="c1",
                camera_name="front",
                video_path="/vid.mp4",
                output_dir=self.temp_dir,
                cancel_event=cancel_event,
            )

        p.seg_model.get_video_dataloader.assert_not_called()

    @patch("checks.objects.single_view.object_processor_dynamic.SceneRasterizer")
    @patch("checks.objects.single_view.object_processor_dynamic.extract_rpy_in_flu", return_value=(0.0, 0.0, 0.0))
    def test_process_frame_uses_cached_masks(self, _mock_rpy: MagicMock, _mock_rasterizer: MagicMock) -> None:
        """When a cache hit occurs, seg_model.process_frame must not be called."""
        p = ObjectProcessorDynamic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu"
        )
        p.seg_model = MagicMock()
        p.overlap_detector = MagicMock()
        p.overlap_detector.process_class.return_value = ({1: 0.5}, {}, {})
        p.hallucination_detector = None

        cached_masks = torch.zeros(3, 8, 8, dtype=torch.uint8)
        mock_cache = MagicMock(spec=SegMaskCache)
        mock_cache.load.return_value = cached_masks

        data_loader = MagicMock(CAMERA_RESCALED_RESOLUTION_HEIGHT=8, CAMERA_RESCALED_RESOLUTION_WIDTH=8)
        data_loader.get_object_data_for_frame.return_value = {"t1": {}}

        p._process_frame(
            frame_idx=0,
            frame=MagicMock(),
            data_loader=data_loader,
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            track_output_agg={},
            results={"hallucination_detections": {}},
            mask_cache=mock_cache,
        )

        # SegFormer inference must be bypassed entirely
        p.seg_model.process_frame.assert_not_called()
        p.seg_model.resize_masks.assert_not_called()
        mock_cache.load.assert_called_once_with(0)

    @patch("checks.objects.single_view.object_processor_dynamic.SceneRasterizer")
    @patch("checks.objects.single_view.object_processor_dynamic.extract_rpy_in_flu", return_value=(0.0, 0.0, 0.0))
    def test_process_frame_saves_masks_on_cache_miss(self, _mock_rpy: MagicMock, _mock_rasterizer: MagicMock) -> None:
        """On a cache miss the computed masks must be saved to the cache."""
        p = ObjectProcessorDynamic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu"
        )
        computed_masks = torch.ones(3, 8, 8, dtype=torch.uint8)
        p.seg_model = MagicMock()
        p.seg_model.process_frame.return_value = MagicMock()
        p.seg_model.resize_masks.return_value = computed_masks
        p.overlap_detector = MagicMock()
        p.overlap_detector.process_class.return_value = ({1: 0.5}, {}, {})
        p.hallucination_detector = None

        mock_cache = MagicMock(spec=SegMaskCache)
        mock_cache.load.return_value = None  # cache miss

        data_loader = MagicMock(CAMERA_RESCALED_RESOLUTION_HEIGHT=8, CAMERA_RESCALED_RESOLUTION_WIDTH=8)
        data_loader.get_object_data_for_frame.return_value = {"t1": {}}

        p._process_frame(
            frame_idx=0,
            frame=MagicMock(),
            data_loader=data_loader,
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            track_output_agg={},
            results={"hallucination_detections": {}},
            mask_cache=mock_cache,
        )

        p.seg_model.process_frame.assert_called_once()
        mock_cache.save.assert_called_once_with(0, computed_masks)


if __name__ == "__main__":
    unittest.main()
