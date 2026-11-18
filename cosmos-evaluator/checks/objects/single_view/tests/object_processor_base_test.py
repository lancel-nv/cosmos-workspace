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

"""Unit tests for base objects processor: ObjectProcessorBase."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from checks.objects.single_view.object_processor_base import ObjectProcessorBase


class TestObjectProcessorBase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.clip_id = "clip_001"
        self.basic_config = {
            "av.objects_single_view": {
                "overlap_check": {"vehicle": {}},
                "importance_filter": {"distance_threshold_m": 30.0},
            }
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_logging_console_only_no_file(self) -> None:
        # With required clip_id/output_dir, file handler is created but we can verify console handler exists
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu", verbose="INFO"
        )
        proc.setup_logging()
        # Verify logger has handlers (console handler should exist)
        self.assertGreater(len(proc.logger.handlers), 0)

    @patch("checks.objects.single_view.object_processor_base.configure_onnx_logging")
    def test_logging_with_file_and_debug(self, mock_onnx: MagicMock) -> None:
        proc = ObjectProcessorBase(
            clip_id=self.clip_id,
            output_dir=self.temp_dir,
            config=self.basic_config,
            model_device="cpu",
            verbose="DEBUG",
        )
        proc.setup_logging()
        log_path = os.path.join(self.temp_dir, f"{self.clip_id}.base.objects.log")
        self.assertTrue(os.path.exists(log_path))
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertNotEqual(content.strip(), "")
        mock_onnx.assert_called_once_with(verbose=True)

    def test_init_results_container_structure(self) -> None:
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu"
        )
        container = proc._init_results_container(total_video_frames=42)
        self.assertEqual(container["processed_frames"], 0)
        self.assertEqual(container["total_video_frames"], 42)
        self.assertIsNone(container["score_matrix"])
        self.assertIsInstance(container["track_ids"], set)
        self.assertIsInstance(container["processed_frame_ids"], list)
        self.assertEqual(container["hallucination_detections"], {})  # base class has no hallucination classes

    def test_finalize_and_summarize_trims_and_stats(self) -> None:
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu"
        )

        # Build results as during processing
        results = proc._init_results_container(total_video_frames=5)
        results["processed_frame_ids"] = [0, 2]
        results["track_ids"] = {1, 3}

        # Score matrix before trimming (5 frames X 6 possible IDs)
        score_matrix = np.full((5, 6), np.nan)
        score_matrix[0, 1] = 0.5
        score_matrix[2, 3] = 0.9

        # Aggregate per-track outputs (dynamic objects without object_type_index)
        track_output_agg = {
            1: {"object_types": {"vehicle"}, "output_counts": {"scored": 2, "occluded": 1, "extra": 7}},
            3: {"object_types": {"pedestrian"}, "output_counts": {"extra": 4, "scored": 3}},
        }

        proc._finalize_and_summarize(
            results=results,
            score_matrix=score_matrix,
            track_output_agg=track_output_agg,
            skipped_frames=1,
            total_video_frames=5,
        )

        # Check trimming
        trimmed = results["score_matrix"]
        self.assertEqual(trimmed.shape, (2, 2))  # frames [0,2] X track IDs 1 and 3
        self.assertTrue(np.isnan(trimmed[0, 1]))  # column 1 corresponds to track_id=3; row 0 has NaN
        self.assertAlmostEqual(trimmed[1, 1], 0.9)

        # Check statistics exist
        self.assertIn("mean_score", results)
        self.assertIn("std_score", results)

        # processor_output_labels removed — importance is now per-track attributes
        self.assertIn("tracks", results)
        self.assertEqual(len(results["tracks"]), 2)

        # Dynamic objects should not have object_type_index
        for track in results["tracks"]:
            self.assertNotIn("object_type_index", track)

    def test_finalize_and_summarize_with_object_type_index(self) -> None:
        """Test that object_type_index is included in tracks for static objects."""
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu"
        )

        # Build results as during processing
        results = proc._init_results_container(total_video_frames=3)
        results["processed_frame_ids"] = [0, 1, 2]
        results["track_ids"] = {100, 101}

        # Score matrix (3 frames X 102 possible IDs to accommodate track IDs 100, 101)
        score_matrix = np.full((3, 102), np.nan)
        score_matrix[0, 100] = 0.8
        score_matrix[1, 100] = 0.85
        score_matrix[2, 101] = 0.9

        # Aggregate per-track outputs (static objects with object_type_index)
        track_output_agg = {
            100: {
                "object_types": {"LaneLine"},
                "object_type_index": 19,  # GEOMETRY_LANELINE_WHITE_SOLID_GROUP
                "output_counts": {"scored": 2},
            },
            101: {
                "object_types": {"TrafficLight"},
                "object_type_index": 8,  # TRAFFIC_LIGHT_RED
                "output_counts": {"scored": 1},
            },
        }

        proc._finalize_and_summarize(
            results=results,
            score_matrix=score_matrix,
            track_output_agg=track_output_agg,
            skipped_frames=0,
            total_video_frames=3,
        )

        # Check tracks have object_type_index
        self.assertIn("tracks", results)
        self.assertEqual(len(results["tracks"]), 2)

        # Find tracks by track_id
        track_100 = next(t for t in results["tracks"] if t["track_id"] == 100)
        track_101 = next(t for t in results["tracks"] if t["track_id"] == 101)

        # Verify object_type_index is present and correct
        self.assertIn("object_type_index", track_100)
        self.assertEqual(track_100["object_type_index"], 19)
        self.assertEqual(track_100["object_type"], "LaneLine")

        self.assertIn("object_type_index", track_101)
        self.assertEqual(track_101["object_type_index"], 8)
        self.assertEqual(track_101["object_type"], "TrafficLight")

    def test_finalize_surfaces_synthetic_source_id(self) -> None:
        """A synthesized track's synthetic_source_id marker is surfaced on the result track."""
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu"
        )
        results = proc._init_results_container(total_video_frames=2)
        results["processed_frame_ids"] = [0, 1]
        results["track_ids"] = {100, 101}
        score_matrix = np.full((2, 102), np.nan)
        score_matrix[0, 100] = 0.8
        score_matrix[1, 101] = 0.9

        # Track 101 is the injected object (carries the synthetic_source_id marker); 100 is a real object.
        track_output_agg = {
            100: {"object_types": {"TrafficLight"}, "object_type_index": 8, "output_counts": {"scored": 1}},
            101: {
                "object_types": {"TrafficLight"},
                "object_type_index": 8,
                "synthetic_source_id": "tl_add_between_pair",
                "output_counts": {"scored": 1},
            },
        }

        proc._finalize_and_summarize(
            results=results,
            score_matrix=score_matrix,
            track_output_agg=track_output_agg,
            skipped_frames=0,
            total_video_frames=2,
        )

        track_100 = next(t for t in results["tracks"] if t["track_id"] == 100)
        track_101 = next(t for t in results["tracks"] if t["track_id"] == 101)
        self.assertNotIn("synthetic_source_id", track_100)
        self.assertEqual(track_101["synthetic_source_id"], "tl_add_between_pair")

    def test_finalize_traffic_light_state_aggregation(self) -> None:
        """Traffic light state/multi-color/confidence are averaged across frames per track."""
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu"
        )

        results = proc._init_results_container(total_video_frames=3)
        results["processed_frame_ids"] = [0, 1, 2]
        results["track_ids"] = {10}

        score_matrix = np.full((3, 11), np.nan)
        score_matrix[0, 10] = 0.7
        score_matrix[1, 10] = 0.8
        score_matrix[2, 10] = 0.9

        results["traffic_light_state"] = {
            "10": [
                {
                    "traffic_light_state_score": 0.9,
                    "traffic_light_multi_color_score": 0.1,
                    "traffic_light_state_confidence": 0.8,
                },
                {
                    "traffic_light_state_score": 0.7,
                    "traffic_light_multi_color_score": 0.3,
                    "traffic_light_state_confidence": 0.6,
                },
            ],
        }

        track_output_agg = {
            10: {"object_types": {"TrafficLight"}, "object_type_index": 8, "output_counts": {"scored": 3}},
        }

        proc._finalize_and_summarize(
            results=results,
            score_matrix=score_matrix,
            track_output_agg=track_output_agg,
            skipped_frames=0,
            total_video_frames=3,
        )

        track = results["tracks"][0]
        self.assertAlmostEqual(track["traffic_light_state_score"], np.nanmean([0.9, 0.7]))
        self.assertAlmostEqual(track["traffic_light_multi_color_score"], np.nanmean([0.1, 0.3]))
        self.assertAlmostEqual(track["traffic_light_state_confidence"], np.nanmean([0.8, 0.6]))

    def test_finalize_static_signal_group_scores(self) -> None:
        """Static processors emit per-signal-group score breakdowns."""
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu"
        )
        proc.naming_suffix = "static"

        results = proc._init_results_container(total_video_frames=3)
        results["processed_frame_ids"] = [0, 1, 2]
        results["track_ids"] = {100, 101, 102}

        score_matrix = np.full((3, 103), np.nan)
        score_matrix[0, 100] = 0.80
        score_matrix[1, 100] = 0.90
        score_matrix[2, 101] = 0.70
        score_matrix[0, 102] = 0.60

        track_output_agg = {
            100: {"object_types": {"LaneLine"}, "object_type_index": 19, "output_counts": {"scored": 2}},
            101: {"object_types": {"TrafficLight"}, "object_type_index": 8, "output_counts": {"scored": 1}},
            102: {"object_types": {"RoadBoundary"}, "object_type_index": 1, "output_counts": {"scored": 1}},
        }

        proc._finalize_and_summarize(
            results=results,
            score_matrix=score_matrix,
            track_output_agg=track_output_agg,
            skipped_frames=0,
            total_video_frames=3,
        )

        grouped = results["signal_group_scores"]
        self.assertAlmostEqual(grouped["lane"]["mean_score"], np.nanmean([0.80, 0.90]))
        self.assertAlmostEqual(grouped["traffic_light"]["mean_score"], 0.70)
        self.assertAlmostEqual(grouped["boundary"]["mean_score"], 0.60)
        self.assertNotIn("traffic_sign", grouped)

    def test_finalize_traffic_light_state_with_nan_scores(self) -> None:
        """NaN values in per-frame TL scores are handled by nanmean."""
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu"
        )

        results = proc._init_results_container(total_video_frames=2)
        results["processed_frame_ids"] = [0, 1]
        results["track_ids"] = {5}

        score_matrix = np.full((2, 6), np.nan)
        score_matrix[0, 5] = 0.5

        results["traffic_light_state"] = {
            "5": [
                {
                    "traffic_light_state_score": float("nan"),
                    "traffic_light_multi_color_score": 0.4,
                    "traffic_light_state_confidence": 0.9,
                },
                {
                    "traffic_light_state_score": 0.6,
                    "traffic_light_multi_color_score": float("nan"),
                    "traffic_light_state_confidence": 0.7,
                },
            ],
        }

        track_output_agg = {
            5: {"object_types": {"TrafficLight"}, "output_counts": {"scored": 1}},
        }

        proc._finalize_and_summarize(
            results=results,
            score_matrix=score_matrix,
            track_output_agg=track_output_agg,
            skipped_frames=0,
            total_video_frames=2,
        )

        track = results["tracks"][0]
        self.assertAlmostEqual(track["traffic_light_state_score"], 0.6)
        self.assertAlmostEqual(track["traffic_light_multi_color_score"], 0.4)
        self.assertAlmostEqual(track["traffic_light_state_confidence"], 0.8)

    def test_finalize_traffic_light_state_partial_fields(self) -> None:
        """Only fields present in per-frame dicts appear on the aggregated track entry."""
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu"
        )

        results = proc._init_results_container(total_video_frames=1)
        results["processed_frame_ids"] = [0]
        results["track_ids"] = {7}

        score_matrix = np.full((1, 8), np.nan)
        score_matrix[0, 7] = 0.6

        results["traffic_light_state"] = {
            "7": [
                {"traffic_light_state_score": 0.75},
            ],
        }

        track_output_agg = {
            7: {"object_types": {"TrafficLight"}, "output_counts": {"scored": 1}},
        }

        proc._finalize_and_summarize(
            results=results,
            score_matrix=score_matrix,
            track_output_agg=track_output_agg,
            skipped_frames=0,
            total_video_frames=1,
        )

        track = results["tracks"][0]
        self.assertAlmostEqual(track["traffic_light_state_score"], 0.75)
        self.assertNotIn("traffic_light_multi_color_score", track)
        self.assertNotIn("traffic_light_state_confidence", track)

    def test_finalize_no_traffic_light_state_in_results(self) -> None:
        """Tracks have no TL fields when results lack traffic_light_state entirely."""
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu"
        )

        results = proc._init_results_container(total_video_frames=1)
        results["processed_frame_ids"] = [0]
        results["track_ids"] = {2}

        score_matrix = np.full((1, 3), np.nan)
        score_matrix[0, 2] = 0.5

        track_output_agg = {
            2: {"object_types": {"vehicle"}, "output_counts": {"scored": 1}},
        }

        proc._finalize_and_summarize(
            results=results,
            score_matrix=score_matrix,
            track_output_agg=track_output_agg,
            skipped_frames=0,
            total_video_frames=1,
        )

        track = results["tracks"][0]
        self.assertNotIn("traffic_light_state_score", track)
        self.assertNotIn("traffic_light_multi_color_score", track)
        self.assertNotIn("traffic_light_state_confidence", track)

    def test_maybe_init_visualization_calls_helper(self) -> None:
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu", verbose="INFO"
        )
        proc.vis_helper = MagicMock()
        dl = MagicMock()
        dl.CAMERA_RESCALED_RESOLUTION_WIDTH = 640
        dl.CAMERA_RESCALED_RESOLUTION_HEIGHT = 480
        proc._maybe_init_visualization("test_fixtures/video.mp4", dl)
        proc.vis_helper.initialize_video_writer.assert_called()

    @patch("checks.objects.single_view.object_processor_base.get_dataloader")
    def test_prepare_video_prefers_seg_helper(self, mock_get_dataloader: MagicMock) -> None:
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu"
        )
        # Mock seg_model with custom get_video_dataloader
        seg = MagicMock()
        mock_dl = MagicMock()
        mock_ds = MagicMock()
        mock_ds.number_frames = 10
        mock_ds.frames_per_second = 20
        mock_dl.__len__ = MagicMock(return_value=10)
        seg.get_video_dataloader.return_value = (mock_dl, mock_ds)
        proc.seg_model = seg

        dataloader, video_fps = proc._prepare_video("test_fixtures/video.mp4", MagicMock())
        self.assertIs(dataloader, mock_dl)
        self.assertEqual(video_fps, 20.0)
        mock_get_dataloader.assert_not_called()

    @patch("checks.objects.single_view.object_processor_base.get_dataloader")
    def test_prepare_video_fallback_when_no_helper(self, mock_get_dataloader: MagicMock) -> None:
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu"
        )
        proc.seg_model = MagicMock()
        proc.seg_model.get_video_dataloader = None
        del proc.seg_model.get_video_dataloader

        mock_dl = MagicMock()
        mock_ds = MagicMock()
        mock_ds.number_frames = 5
        mock_ds.frames_per_second = None
        mock_dl.__len__ = MagicMock(return_value=5)
        mock_get_dataloader.return_value = (mock_dl, mock_ds)

        dataloader, video_fps = proc._prepare_video("test_fixtures/video.mp4", MagicMock())
        self.assertIs(dataloader, mock_dl)
        # The container reports no usable rate, so the caller is told so rather than handed a made-up one.
        self.assertIsNone(video_fps)

    def test_create_frame_visualization_delegates(self) -> None:
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu", verbose="INFO"
        )
        proc.vis_helper = MagicMock()
        proc._create_frame_visualization(
            frame_idx=0,
            segmentation_masks=MagicMock(),
            frame_scores={"track1": 0.7},
            camera_pose=np.eye(4),
            camera_model=MagicMock(),
            frame_objects={"track1": {}},
            filtered_objects={},
            hallucinations=None,
        )
        proc.vis_helper.create_frame_visualization.assert_called()
        proc.vis_helper.write_frame.assert_called()

    def test_get_config_summary_includes_hallucination(self) -> None:
        cfg = {
            "av.objects_single_view": {
                "overlap_check": {},
                "importance_filter": {},
                "hallucination_detector": {
                    "enabled": True,
                    "classes": ["vehicle"],
                    "min_cluster_area": 50,
                    "max_cluster_per_frame": 5,
                },
            }
        }
        proc = ObjectProcessorBase(clip_id=self.clip_id, output_dir=self.temp_dir, config=cfg, model_device="cpu")
        summary = proc.get_config_summary()
        self.assertIn("hallucination_detector", summary)

    def test_finalize_passes_projection_area_matrix_to_save(self) -> None:
        """Per-frame projection_area_matrix is plumbed to save_results_to_json and triggers the static sparse_compact
        schema."""
        proc = ObjectProcessorBase(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.basic_config, model_device="cpu"
        )

        results = proc._init_results_container(total_video_frames=2)
        results["processed_frame_ids"] = [0, 1]
        results["track_ids"] = {3}

        score_matrix = np.full((2, 4), np.nan)
        score_matrix[0, 3] = 0.4
        score_matrix[1, 3] = 0.6

        proj_area = np.full((2, 4), np.nan)
        proj_area[0, 3] = 250.0
        proj_area[1, 3] = 320.0

        track_output_agg = {
            3: {"object_types": {"TrafficLight"}, "output_counts": {"scored": 2}},
        }

        with patch("checks.objects.single_view.object_processor_base.save_results_to_json") as mock_save:
            mock_save.return_value = "/tmp/x.json"
            proc._finalize_and_summarize(
                results=results,
                score_matrix=score_matrix,
                track_output_agg=track_output_agg,
                skipped_frames=0,
                total_video_frames=2,
                projection_area_matrix=proj_area,
            )

        # save_results_to_json got the matrix and the sparse_compact format
        kwargs = mock_save.call_args.kwargs
        self.assertEqual(kwargs["matrix_format"], "sparse_compact")
        self.assertIs(kwargs["projection_area_matrix"] is None, False)
        # No track-level projection dict on track entries (per-frame matrix supersedes it).
        track = results["tracks"][0]
        self.assertNotIn("projection", track)


class TestGetDefaultConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_cosmos_evaluator_config = {
            "av.objects_single_view": {
                "enabled": True,
                "overlap_check": {"pedestrian": {}, "vehicle": {}},
                "importance_filter": {
                    "distance_threshold_m": 100,
                    "skip_oncoming_objects": True,
                    "relevant_lanes": ["ego", "left", "right"],
                },
            },
            "av.vlm": {"enabled": False},
        }

    @patch("checks.objects.single_view.object_processor_base.ConfigManager")
    def test_get_default_config_success(self, mock_config_manager_class: MagicMock) -> None:
        mock_config_manager = MagicMock()
        mock_config_manager.load_config.return_value = self.sample_cosmos_evaluator_config
        mock_config_manager_class.return_value = mock_config_manager

        result = ObjectProcessorBase.get_default_config()
        expected_config = {"av.objects_single_view": self.sample_cosmos_evaluator_config["av.objects_single_view"]}
        self.assertEqual(result, expected_config)

    @patch("checks.objects.single_view.object_processor_base.ConfigManager")
    def test_get_default_config_file_not_found(self, mock_config_manager_class: MagicMock) -> None:
        mock_config_manager = MagicMock()
        mock_config_manager.load_config.side_effect = FileNotFoundError("Configuration file not found")
        mock_config_manager_class.return_value = mock_config_manager
        with self.assertRaises(FileNotFoundError):
            ObjectProcessorBase.get_default_config()

    @patch("checks.objects.single_view.object_processor_base.ConfigManager")
    @patch("checks.objects.single_view.object_processor_base.logger")
    def test_get_default_config_generic_exception(
        self, mock_logger: MagicMock, mock_config_manager_class: MagicMock
    ) -> None:
        mock_config_manager = MagicMock()
        mock_config_manager.load_config.side_effect = Exception("Unexpected error")
        mock_config_manager_class.return_value = mock_config_manager
        with self.assertRaises(Exception) as context:
            ObjectProcessorBase.get_default_config()
        self.assertEqual(str(context.exception), "Unexpected error")
        mock_logger.error.assert_called_once_with(
            "Error loading default configuration: %s", mock_config_manager.load_config.side_effect
        )


if __name__ == "__main__":
    unittest.main()
