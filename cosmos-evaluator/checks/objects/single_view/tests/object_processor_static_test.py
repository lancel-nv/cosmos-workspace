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

"""Unit tests for static objects processor: ObjectProcessorStatic."""

import os
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
from typing import Any, ClassVar, cast
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from checks.objects.single_view.cwip_helper import (
    COSMOS_EVALUATOR_OBJECT_TYPE_STRING_TO_ENUMS,
    resolve_model_class_indices,
)
from checks.objects.single_view.cwip_inference_cache import CwipInferenceCache
from checks.objects.single_view.object_processor_static import (
    HallucinationGates,
    ObjectProcessorStatic,
    summarize_cwip_output,
)
from checks.utils.frame_rates import RDS_INDEX_STRIDE, ObjectCheckTimeline
from checks.utils.models.cwip.modules.class_mapping import ObjectTypeClassMappingV3, get_object_type_class_mapping
from checks.utils.models.cwip.modules.presence import PresenceSource, TypeMaskGate
from checks.utils.models.cwip.visualization.world_model_utils import PatchSimilarityType
from checks.utils.object_class_names import class_name_to_rds_label
from utils.cancellation import ProcessingCancelled

PRESENCE_SOURCES = [s.value for s in PresenceSource]


class TestObjectProcessorStatic(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.clip_id = "clip_static"
        self.config = {
            "av.objects_single_view": {
                "overlap_check": {"traffic_light": {}, "traffic_sign": {}},
                "importance_filter": {},
                "cwip": {"sample_method": "uniform"},
            }
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    @patch("checks.objects.single_view.object_processor_static.get_object_type_colors", return_value={})
    @patch(
        "checks.objects.single_view.object_processor_static.get_object_type_class_mapping",
        return_value=ObjectTypeClassMappingV3,
    )
    @patch(
        "checks.objects.single_view.object_processor_static.get_runfiles_path",
        return_value=os.path.join(os.getcwd(), "cwip", "model.safetensors"),
    )
    @patch("checks.objects.single_view.object_processor_static.CWIPInferenceHelper")
    @patch("checks.objects.single_view.object_processor_static.OverlapDetector")
    def test_presence_config_validation(self, *_mocks: MagicMock) -> None:
        """A malformed ``presence`` block is rejected at construction; a valid one sets the presence source."""

        def build(presence_cfg: Any, **extra: Any) -> ObjectProcessorStatic:
            cfg = {
                "av.objects_single_view": {
                    **self.config["av.objects_single_view"],
                    # traffic_sign/traffic_light have a hallucination-detector entry; crosswalk does not. vehicle has
                    # one but is a dynamic class, which the static detector drops.
                    "hallucination_detector": {"classes": {"traffic_sign": {}, "traffic_light": {}, "vehicle": {}}},
                    "presence": presence_cfg,
                    **extra,
                }
            }
            return ObjectProcessorStatic(
                clip_id=self.clip_id, output_dir=self.temp_dir, config=cfg, model_device="cpu", verbose="INFO"
            )

        with self.assertRaisesRegex(TypeError, "presence must be a mapping"):
            build("union")
        with self.assertRaisesRegex(ValueError, "presence has unknown key"):
            build({"presence_source": "union"})
        with self.assertRaisesRegex(ValueError, "presence.source"):
            build({"source": "argmax"})  # a type-mask gate, not a presence source
        with self.assertRaisesRegex(ValueError, "presence.source"):
            build({"source": "Similarity"})
        # Presence settings under a mode-named block would be ignored, so those keys are rejected.
        with self.assertRaisesRegex(ValueError, "camera_only"):
            build({}, camera_only={"type_confidence_threshold": 0.3})
        with self.assertRaisesRegex(ValueError, "two_stream"):
            build({}, two_stream={"presence_source": "similarity"})

        # type_mask
        with self.assertRaisesRegex(TypeError, "presence.type_mask must be a mapping"):
            build({"type_mask": "confidence"})
        with self.assertRaisesRegex(ValueError, "presence.type_mask has unknown key"):
            build({"type_mask": {"threshold": 0.5}})
        with self.assertRaisesRegex(ValueError, "presence.type_mask.gate"):
            build({"type_mask": {"gate": "type_confidence"}})
        # Out-of-range probabilities are rejected (would mark every patch absent/present).
        with self.assertRaisesRegex(ValueError, "confidence_threshold"):
            build({"type_mask": {"confidence_threshold": 1.5}})
        with self.assertRaisesRegex(ValueError, "per_class_thresholds"):
            build({"type_mask": {"per_class_thresholds": {"traffic_sign": 2.0}}})
        # Unknown per_class_thresholds class.
        with self.assertRaisesRegex(ValueError, "per_class_thresholds"):
            build({"type_mask": {"per_class_thresholds": {"traffic_lite": 0.5}}})
        # Falsy non-mappings are rejected too, not read as "no overrides".
        for bad in ([0.5], [], "", 0):
            with self.subTest(per_class_thresholds=bad), self.assertRaisesRegex(TypeError, "per_class_thresholds must"):
                build({"type_mask": {"per_class_thresholds": bad}})

        # hallucination_overrides key must be a static class with a detector entry: a typo, crosswalk (no detector
        # entry), and vehicle (detector entry, but dynamic) are all rejected.
        for name in ("traffic_signs", "crosswalk", "vehicle"):
            with self.subTest(override_class=name), self.assertRaisesRegex(ValueError, "hallucination_overrides"):
                build({"hallucination_overrides": {name: {"min_cluster_area": 300}}})
        # Non-mapping overrides are rejected rather than silently ignored or failing with AttributeError.
        with self.assertRaisesRegex(TypeError, "presence.hallucination_overrides must be a mapping"):
            build({"hallucination_overrides": ["traffic_sign"]})
        with self.assertRaisesRegex(TypeError, "presence.hallucination_overrides must be a mapping"):
            build({"hallucination_overrides": "traffic_sign"})
        with self.assertRaisesRegex(TypeError, r"hallucination_overrides\['traffic_sign'\] must be a mapping"):
            build({"hallucination_overrides": {"traffic_sign": 300}})
        # Out-of-range override values.
        with self.assertRaisesRegex(ValueError, "hallucination_overrides.*min_cluster_area"):
            build({"hallucination_overrides": {"traffic_sign": {"min_cluster_area": -1}}})
        with self.assertRaisesRegex(ValueError, "hallucination_overrides.*min_track_frames"):
            build({"hallucination_overrides": {"traffic_sign": {"min_track_frames": 0}}})
        # A non-mapping hallucination_detector block is a config error, not an AttributeError.
        with self.assertRaisesRegex(TypeError, "hallucination_detector must be a mapping"):
            build({"hallucination_overrides": {"traffic_sign": {"min_cluster_area": 300}}}, hallucination_detector="on")
        # An empty YAML block or class entry (None) means no override.
        build({"hallucination_overrides": None})
        build({"hallucination_overrides": {"traffic_sign": None}})

        # A well-formed block constructs and selects its source.
        valid = build(
            {
                "source": "intersection",
                "type_mask": {
                    "gate": "confidence",
                    "confidence_threshold": 0.5,
                    "per_class_thresholds": {"traffic_sign": 0.5},
                },
                "hallucination_overrides": {"traffic_sign": {"min_cluster_area": 300, "min_track_frames": 5}},
            }
        )
        self.assertIs(valid._presence.source, PresenceSource.INTERSECTION)
        self.assertEqual(valid._presence.type_mask.confidence_threshold, 0.5)
        # Per-class thresholds are resolved once, at construction, to every model class index of the object class.
        sign_indices = resolve_model_class_indices(
            COSMOS_EVALUATOR_OBJECT_TYPE_STRING_TO_ENUMS["traffic_sign"], ObjectTypeClassMappingV3
        )
        self.assertTrue(sign_indices)
        self.assertEqual(dict(valid._presence.type_mask.class_thresholds), dict.fromkeys(sign_indices, 0.5))
        self.assertEqual(valid._halluc_overrides, {"traffic_sign": {"min_cluster_area": 300, "min_track_frames": 5}})
        for source in PRESENCE_SOURCES:
            with self.subTest(source=source):
                self.assertEqual(build({"source": source})._presence.source, source)
        self.assertIs(build({"type_mask": {"gate": "argmax"}})._presence.type_mask.gate, TypeMaskGate.ARGMAX)
        build({"type_mask": None})
        # An empty or absent block selects the defaults: the union source and the confidence gate at 0.5.
        self.assertIs(build({})._presence.source, PresenceSource.UNION)
        self.assertIs(build(None)._presence.source, PresenceSource.UNION)
        no_block = ObjectProcessorStatic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu", verbose="INFO"
        )
        self.assertIs(no_block._presence.source, PresenceSource.UNION)
        self.assertIs(no_block._presence.type_mask.gate, TypeMaskGate.CONFIDENCE)
        self.assertEqual(no_block._presence.type_mask.confidence_threshold, 0.5)

    @patch("checks.objects.single_view.object_processor_static.get_object_type_colors", return_value={})
    @patch(
        "checks.objects.single_view.object_processor_static.get_object_type_class_mapping",
        return_value=ObjectTypeClassMappingV3,
    )
    @patch(
        "checks.objects.single_view.object_processor_static.get_runfiles_path",
        return_value=os.path.join(os.getcwd(), "cwip", "model.safetensors"),
    )
    @patch("checks.objects.single_view.object_processor_static.CWIPInferenceHelper")
    @patch("checks.objects.single_view.object_processor_static.OverlapDetector")
    @patch("checks.objects.single_view.object_processor_static.get_video_frame_count", return_value=None)
    @patch("checks.objects.single_view.object_processor_static.get_video_fps", return_value=30.0)
    @patch("checks.objects.single_view.object_processor_base.RdsDataLoader")
    def test_undeterminable_frame_count_raises(
        self,
        mock_loader_cls: MagicMock,
        _mock_fps: MagicMock,
        _mock_frame_count: MagicMock,
        _mock_overlap: MagicMock,
        _mock_cwip: MagicMock,
        _mock_runfiles: MagicMock,
        _mock_mapping: MagicMock,
        _mock_colors: MagicMock,
    ) -> None:
        """A frame count ffprobe cannot determine must fail loudly rather than bound the timeline at zero steps.

        Coercing ``None`` to ``0`` would leave ``num_steps == 0`` and produce an empty result for the whole clip,
        even though the FPS probes above succeeded.
        """
        loader = MagicMock()
        loader.CAMERA_RESCALED_RESOLUTION_WIDTH = 640
        loader.CAMERA_RESCALED_RESOLUTION_HEIGHT = 480
        loader.object_count = 10
        loader.load_camera_data.return_value = ({i: np.eye(4) for i in range(9)}, MagicMock())
        loader.load_ego_poses.return_value = {i: np.eye(4) for i in range(9)}
        loader.get_canonical_static_frame_axis.return_value = [0, 3, 6]
        mock_loader_cls.return_value = loader

        processor = ObjectProcessorStatic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu", verbose="INFO"
        )
        with self.assertRaisesRegex(RuntimeError, "frame count"):
            processor.process_clip(
                input_data="/data",
                clip_id=self.clip_id,
                camera_name="front",
                camera_video_path="/cam.mp4",
                world_video_path="/world.mp4",
                output_dir=self.temp_dir,
            )

    @patch("checks.objects.single_view.object_processor_static.get_object_type_colors", return_value={})
    @patch(
        "checks.objects.single_view.object_processor_static.get_object_type_class_mapping",
        return_value=ObjectTypeClassMappingV3,
    )
    @patch(
        "checks.objects.single_view.object_processor_static.get_runfiles_path",
        return_value=os.path.join(os.getcwd(), "cwip", "model.safetensors"),
    )
    @patch("checks.objects.single_view.object_processor_static.CWIPInferenceHelper")
    @patch("checks.objects.single_view.object_processor_static.OverlapDetector")
    def test_static_init_components(
        self,
        mock_overlap: MagicMock,
        mock_cwip: MagicMock,
        _mock_runfiles: MagicMock,
        _mock_mapping: MagicMock,
        _mock_colors: MagicMock,
    ) -> None:
        p = ObjectProcessorStatic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu", verbose="INFO"
        )
        mock_cwip.from_hf_format.assert_called()
        mock_overlap.assert_called()
        assert p.vis_helper is not None
        # set_mask_styles should be called with similarity legend
        self.assertTrue(p.vis_helper.mask_colors.get("MATCH") is not None)
        # type mask legend config should also be set
        self.assertIn("type", p.vis_helper._mask_legend_configs)

    @patch("checks.objects.single_view.object_processor_static.get_runfiles_path", return_value=None)
    def test_static_init_missing_model_path(self, _mock_runfiles: MagicMock) -> None:
        with self.assertRaises(RuntimeError):
            ObjectProcessorStatic(
                clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu", verbose="INFO"
            )

    @patch("checks.objects.single_view.object_processor_static.get_object_type_colors", return_value={})
    @patch(
        "checks.objects.single_view.object_processor_static.get_object_type_class_mapping",
        return_value=ObjectTypeClassMappingV3,
    )
    @patch(
        "checks.objects.single_view.object_processor_static.get_runfiles_path",
        return_value=os.path.join(os.getcwd(), "cwip", "model.safetensors"),
    )
    @patch("checks.objects.single_view.object_processor_static.CWIPInferenceHelper")
    @patch("checks.objects.single_view.object_processor_static.OverlapDetector")
    def test_state_gate_reads_its_own_key_not_the_correspondence_one(
        self,
        _mock_overlap: MagicMock,
        _mock_cwip: MagicMock,
        _mock_runfiles: MagicMock,
        _mock_mapping: MagicMock,
        _mock_colors: MagicMock,
    ) -> None:
        """The two thresholds are independent: correspondence gating must not leak into the state check.

        They are tuned apart because correspondence doubles as the existence verdict downstream, so a value
        chosen to protect color reading would otherwise silently move which lights count as present.
        """
        config = {
            "av.objects_single_view": {
                "overlap_check": {
                    "traffic_light": {"min_in_frame_fraction": 0.4, "state_min_in_frame_fraction": 1.0},
                },
                "importance_filter": {},
                "cwip": {"sample_method": "uniform"},
            }
        }
        p = ObjectProcessorStatic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=config, model_device="cpu", verbose="INFO"
        )
        self.assertEqual(p._tl_state_min_in_frame_fraction, 1.0)

    @patch("checks.objects.single_view.object_processor_static.get_object_type_colors", return_value={})
    @patch(
        "checks.objects.single_view.object_processor_static.get_object_type_class_mapping",
        return_value=ObjectTypeClassMappingV3,
    )
    @patch(
        "checks.objects.single_view.object_processor_static.get_runfiles_path",
        return_value=os.path.join(os.getcwd(), "cwip", "model.safetensors"),
    )
    @patch("checks.objects.single_view.object_processor_static.CWIPInferenceHelper")
    @patch("checks.objects.single_view.object_processor_static.OverlapDetector")
    def test_state_gate_does_not_fall_back_to_the_correspondence_key(
        self,
        _mock_overlap: MagicMock,
        _mock_cwip: MagicMock,
        _mock_runfiles: MagicMock,
        _mock_mapping: MagicMock,
        _mock_colors: MagicMock,
    ) -> None:
        """A config setting only ``min_in_frame_fraction`` leaves the state check ungated, by design."""
        config = {
            "av.objects_single_view": {
                "overlap_check": {"traffic_light": {"min_in_frame_fraction": 1.0}},
                "importance_filter": {},
                "cwip": {"sample_method": "uniform"},
            }
        }
        p = ObjectProcessorStatic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=config, model_device="cpu", verbose="INFO"
        )
        self.assertEqual(p._tl_state_min_in_frame_fraction, 0.0)

    @patch("checks.objects.single_view.object_processor_static.get_object_type_colors", return_value={})
    @patch(
        "checks.objects.single_view.object_processor_static.get_object_type_class_mapping",
        return_value=ObjectTypeClassMappingV3,
    )
    @patch(
        "checks.objects.single_view.object_processor_static.get_runfiles_path",
        return_value=os.path.join(os.getcwd(), "cwip", "model.safetensors"),
    )
    @patch("checks.objects.single_view.object_processor_static.CWIPInferenceHelper")
    @patch("checks.objects.single_view.object_processor_static.OverlapDetector")
    @patch("checks.objects.single_view.object_processor_static.get_video_frame_count", return_value=9)
    @patch("checks.objects.single_view.object_processor_static.get_video_fps", return_value=30.0)
    @patch("checks.objects.single_view.object_processor_static.ObjectProcessorStatic._run_cwip_inference_on_videos")
    @patch("checks.objects.single_view.object_processor_base.RdsDataLoader")
    def test_process_frames_static_and_clip(
        self,
        mock_loader_cls: MagicMock,
        mock_run_cwip: MagicMock,
        _mock_fps: MagicMock,
        _mock_frame_count: MagicMock,
        _mock_overlap: MagicMock,
        _mock_cwip: MagicMock,
        _mock_runfiles: MagicMock,
        _mock_mapping: MagicMock,
        _mock_colors: MagicMock,
    ) -> None:
        result = self._run_mocked_clip(mock_loader_cls, mock_run_cwip, self.config)
        self.assertIn("processed_frames", result)
        self.assertGreaterEqual(result["processed_frames"], 1)
        # The run mode is recorded on the static results; this is a two-stream run (world video provided).
        self.assertIs(result["camera_only"], False)
        # Without a presence block the default source applies, and the result says so.
        self.assertEqual(result["presence_source"], "union")

    def _run_mocked_clip(
        self,
        mock_loader_cls: MagicMock,
        mock_run_cwip: MagicMock,
        config: dict[str, Any],
        world_video_path: str | None = "/world.mp4",
    ) -> dict[str, Any]:
        """Run ``process_clip`` over mocked CWIP output and data; without a world video the run is camera-only."""
        # Mock CWIP outputs: 3 frames of class indices + raw type logits (no similarity head without a world video).
        pred_similarity = torch.zeros((3, 480, 640), dtype=torch.long) if world_video_path is not None else None
        pred_type = torch.zeros((3, 480, 640), dtype=torch.long)
        raw_type_logits = torch.zeros((3, 20, 30, 40), dtype=torch.float)
        mock_run_cwip.return_value = ({"num_frames": 3}, pred_similarity, pred_type, raw_type_logits)

        # Mock loader
        loader = MagicMock()
        loader.CAMERA_RESCALED_RESOLUTION_WIDTH = 640
        loader.CAMERA_RESCALED_RESOLUTION_HEIGHT = 480
        loader.object_count = 10
        camera_poses = {i: np.eye(4) for i in range(9)}
        loader.load_camera_data.return_value = (camera_poses, MagicMock())
        loader.load_ego_poses.return_value = camera_poses
        loader.get_canonical_static_frame_axis.return_value = [0, 3, 6]
        loader.traffic_cuboids_all_frames = [{}, {}, {}]
        mock_loader_cls.return_value = loader

        processor = ObjectProcessorStatic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=config, model_device="cpu", verbose="INFO"
        )
        processor.vis_helper = MagicMock()
        processor.vis_helper.frame_width = 640
        processor.vis_helper.frame_height = 480
        cast(MagicMock, processor.overlap_detector.process_class).side_effect = [
            ({1: 0.2}, {}, {}),
            ({1: 0.3}, {}, {}),
        ] * 3
        return processor.process_clip(
            input_data="/data",
            clip_id=self.clip_id,
            camera_name="front",
            camera_video_path="/cam.mp4",
            world_video_path=world_video_path,
            camera_only=world_video_path is None,
        )

    @patch("checks.objects.single_view.object_processor_static.get_object_type_colors", return_value={})
    @patch(
        "checks.objects.single_view.object_processor_static.get_object_type_class_mapping",
        return_value=ObjectTypeClassMappingV3,
    )
    @patch(
        "checks.objects.single_view.object_processor_static.get_runfiles_path",
        return_value=os.path.join(os.getcwd(), "cwip", "model.safetensors"),
    )
    @patch("checks.objects.single_view.object_processor_static.CWIPInferenceHelper")
    @patch("checks.objects.single_view.object_processor_static.OverlapDetector")
    @patch("checks.objects.single_view.object_processor_static.get_video_frame_count", return_value=9)
    @patch("checks.objects.single_view.object_processor_static.get_video_fps", return_value=30.0)
    @patch("checks.objects.single_view.object_processor_static.ObjectProcessorStatic._run_cwip_inference_on_videos")
    @patch("checks.objects.single_view.object_processor_base.RdsDataLoader")
    def test_process_clip_records_configured_presence_source(
        self, mock_loader_cls: MagicMock, mock_run_cwip: MagicMock, *_mocks: MagicMock
    ) -> None:
        """A non-default source runs end to end over the frame loop and is recorded on the static results."""
        config = {
            "av.objects_single_view": {
                **self.config["av.objects_single_view"],
                "presence": {"source": "intersection"},
            }
        }
        result = self._run_mocked_clip(mock_loader_cls, mock_run_cwip, config)
        self.assertGreaterEqual(result["processed_frames"], 1)
        self.assertEqual(result["presence_source"], "intersection")

    @patch("checks.objects.single_view.object_processor_static.get_object_type_colors", return_value={})
    @patch(
        "checks.objects.single_view.object_processor_static.get_object_type_class_mapping",
        return_value=ObjectTypeClassMappingV3,
    )
    @patch(
        "checks.objects.single_view.object_processor_static.get_runfiles_path",
        return_value=os.path.join(os.getcwd(), "cwip", "model.safetensors"),
    )
    @patch("checks.objects.single_view.object_processor_static.CWIPInferenceHelper")
    @patch("checks.objects.single_view.object_processor_static.OverlapDetector")
    @patch("checks.objects.single_view.object_processor_static.get_video_frame_count", return_value=9)
    @patch("checks.objects.single_view.object_processor_static.get_video_fps", return_value=30.0)
    @patch("checks.objects.single_view.object_processor_static.ObjectProcessorStatic._run_cwip_inference_on_videos")
    @patch("checks.objects.single_view.object_processor_base.RdsDataLoader")
    def test_type_filtered_view_follows_the_presence_layers(
        self, mock_loader_cls: MagicMock, mock_run_cwip: MagicMock, *_mocks: MagicMock
    ) -> None:
        """The filtered type-mask view is built from the scored presence layers (correspondence OR hallucination)."""
        match = torch.zeros((480, 640), dtype=torch.long)
        match[0, 0] = 1
        halluc = torch.zeros((480, 640), dtype=torch.long)
        halluc[1, 1] = 1
        base = self.config["av.objects_single_view"]
        config = {"av.objects_single_view": {**base, "presence": {"source": "union"}}}
        with (
            patch.object(ObjectProcessorStatic, "_presence_layers", return_value=(match, halluc)),
            patch.object(
                ObjectProcessorStatic,
                "_build_type_mask_frame",
                autospec=True,
                return_value=np.zeros((480, 640, 3), dtype=np.uint8),
            ) as mock_build,
        ):
            self._run_mocked_clip(mock_loader_cls, mock_run_cwip, config)
        filtered = [c for c in mock_build.call_args_list if len(c.args) > 3 and c.args[3] is not None]
        self.assertTrue(filtered, "the filtered type view was never built")
        for call in filtered:
            presence = call.args[3]
            self.assertEqual(int(presence.sum()), 2)
            self.assertEqual((int(presence[0, 0]), int(presence[1, 1])), (1, 1))

    @patch("checks.objects.single_view.object_processor_static.get_object_type_colors", return_value={})
    @patch(
        "checks.objects.single_view.object_processor_static.get_object_type_class_mapping",
        return_value=ObjectTypeClassMappingV3,
    )
    @patch(
        "checks.objects.single_view.object_processor_static.get_runfiles_path",
        return_value=os.path.join(os.getcwd(), "cwip", "model.safetensors"),
    )
    @patch("checks.objects.single_view.object_processor_static.CWIPInferenceHelper")
    @patch("checks.objects.single_view.object_processor_static.OverlapDetector")
    @patch("checks.objects.single_view.object_processor_static.get_video_frame_count", return_value=9)
    @patch("checks.objects.single_view.object_processor_static.get_video_fps", return_value=30.0)
    @patch("checks.objects.single_view.object_processor_static.ObjectProcessorStatic._run_cwip_inference_on_videos")
    @patch("checks.objects.single_view.object_processor_base.RdsDataLoader")
    def test_process_clip_applies_the_overrides_for_its_mode(
        self, mock_loader_cls: MagicMock, mock_run_cwip: MagicMock, *_mocks: MagicMock
    ) -> None:
        """process_clip applies, and records, the overrides of its mode and presence source."""
        overrides = {"traffic_sign": {"min_cluster_area": 300, "min_track_frames": 5}}
        cases = [
            ("union", "/world.mp4", overrides),
            ("type", "/world.mp4", overrides),
            ("similarity", "/world.mp4", {}),
            ("intersection", "/world.mp4", {}),
            ("similarity", None, overrides),  # camera-only: the type mask whatever the two-stream source
        ]
        for source, world_video_path, expected in cases:
            with (
                self.subTest(source=source, camera_only=world_video_path is None),
                patch.object(
                    ObjectProcessorStatic,
                    "_process_frames_static",
                    autospec=True,
                    side_effect=ObjectProcessorStatic._process_frames_static,
                ) as mock_frames,
                patch("checks.objects.single_view.object_processor_static.HallucinationDetector") as mock_detector_cls,
            ):
                mock_detector_cls.return_value.detect.return_value = {}
                config = {
                    "av.objects_single_view": {
                        **self.config["av.objects_single_view"],
                        "hallucination_detector": {"enabled": True, "classes": {"traffic_sign": {}}},
                        "presence": {"source": source, "hallucination_overrides": overrides},
                    }
                }
                result = self._run_mocked_clip(
                    mock_loader_cls, mock_run_cwip, config, world_video_path=world_video_path
                )
                self.assertEqual(result["presence"]["hallucination_overrides"], expected)
                gates = mock_frames.call_args.kwargs["halluc_gates"]
                self.assertEqual(dict(gates.min_cluster_area), {"traffic_sign": 300} if expected else {})
                self.assertEqual(gates.min_track_frames[class_name_to_rds_label("traffic_sign")], 5 if expected else 1)
                # The detector receives the run's cluster areas on every call; its configured state is never touched.
                detect_calls = mock_detector_cls.return_value.detect.call_args_list
                self.assertTrue(detect_calls)
                for call in detect_calls:
                    self.assertEqual(dict(call.kwargs["min_cluster_area"]), dict(gates.min_cluster_area))

    @patch("checks.objects.single_view.object_processor_static.get_object_type_colors", return_value={})
    @patch(
        "checks.objects.single_view.object_processor_static.get_object_type_class_mapping",
        return_value=ObjectTypeClassMappingV3,
    )
    @patch(
        "checks.objects.single_view.object_processor_static.get_runfiles_path",
        return_value=os.path.join(os.getcwd(), "cwip", "model.safetensors"),
    )
    @patch("checks.objects.single_view.object_processor_static.CWIPInferenceHelper")
    @patch("checks.objects.single_view.object_processor_static.OverlapDetector")
    @patch("checks.objects.single_view.object_processor_static.get_video_frame_count", return_value=9)
    @patch("checks.objects.single_view.object_processor_static.get_video_fps", return_value=30.0)
    @patch("checks.objects.single_view.object_processor_static.ObjectProcessorStatic._run_cwip_inference_on_videos")
    @patch("checks.objects.single_view.object_processor_base.RdsDataLoader")
    def test_camera_only_result_records_the_type_presence_source(
        self, mock_loader_cls: MagicMock, mock_run_cwip: MagicMock, *_mocks: MagicMock
    ) -> None:
        """A camera-only run has no similarity head, so it records ``type`` whatever ``presence.source`` says."""
        config = {
            "av.objects_single_view": {
                **self.config["av.objects_single_view"],
                "presence": {"source": "intersection"},
            }
        }
        result = self._run_mocked_clip(mock_loader_cls, mock_run_cwip, config, world_video_path=None)
        self.assertIs(result["camera_only"], True)
        self.assertEqual(result["presence_source"], "type")

    @patch("checks.objects.single_view.object_processor_static.get_video_fps", return_value=None)
    @patch("checks.objects.single_view.object_processor_base.RdsDataLoader")
    def test_process_clip_static_raises_when_no_fps(self, mock_loader: MagicMock, _mock_fps: MagicMock) -> None:
        loader = mock_loader.return_value
        loader.load_camera_data.return_value = (MagicMock(), MagicMock())
        loader.load_ego_poses.return_value = MagicMock()
        with self.assertRaises(RuntimeError):
            ObjectProcessorStatic(
                clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu", verbose="INFO"
            ).process_clip(
                input_data="/data",
                clip_id="cS",
                camera_name="front",
                camera_video_path="/cam.mp4",
                world_video_path="/world.mp4",
                output_dir=self.temp_dir,
            )

    @patch("checks.objects.single_view.object_processor_static.get_video_fps", return_value=30.0)
    @patch("checks.objects.single_view.object_processor_base.RdsDataLoader")
    def test_process_clip_static_cancels_before_fps_probes(self, mock_loader: MagicMock, mock_fps: MagicMock) -> None:
        """An already-cancelled request aborts right after data loading, before the prologue's expensive stages.

        Asserting only that ``ProcessingCancelled`` is raised would still pass if the checkpoint sat at the first
        per-frame loop, so pin the position: the FPS probes that follow it must never run.
        """
        loader = mock_loader.return_value
        loader.load_camera_data.return_value = (MagicMock(), MagicMock())
        loader.load_ego_poses.return_value = MagicMock()

        cancel_event = threading.Event()
        cancel_event.set()

        with self.assertRaises(ProcessingCancelled):
            ObjectProcessorStatic(
                clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu", verbose="INFO"
            ).process_clip(
                input_data="/data",
                clip_id="cS",
                camera_name="front",
                camera_video_path="/cam.mp4",
                world_video_path="/world.mp4",
                output_dir=self.temp_dir,
                cancel_event=cancel_event,
            )

        mock_fps.assert_not_called()


class TestObjectProcessorStaticCache(unittest.TestCase):
    """Tests for CWIP inference caching integration."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.clip_id = "clip_cache"
        self.config = {
            "av.objects_single_view": {
                "overlap_check": {"traffic_light": {}, "traffic_sign": {}},
                "importance_filter": {},
                "cwip": {"sample_method": "uniform"},
            }
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _make_processor(self) -> ObjectProcessorStatic:
        with (
            patch("checks.objects.single_view.object_processor_static.get_object_type_colors", return_value={}),
            patch(
                "checks.objects.single_view.object_processor_static.get_object_type_class_mapping",
                return_value=ObjectTypeClassMappingV3,
            ),
            patch(
                "checks.objects.single_view.object_processor_static.get_runfiles_path",
                return_value=os.path.join(os.getcwd(), "cwip", "model.safetensors"),
            ),
            patch("checks.objects.single_view.object_processor_static.CWIPInferenceHelper"),
            patch("checks.objects.single_view.object_processor_static.OverlapDetector"),
        ):
            return ObjectProcessorStatic(
                clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu", verbose="INFO"
            )

    def test_process_frames_uses_cached_cwip_result(self) -> None:
        """When a cache hit occurs, _run_cwip_inference_on_videos must not be called."""
        processor = self._make_processor()
        processor.vis_helper = None
        cast(MagicMock, processor.overlap_detector.process_class).return_value = ({}, {}, {})

        pred_similarity = torch.zeros((2, 4, 4), dtype=torch.long)
        pred_type = torch.zeros((2, 4, 4), dtype=torch.long)
        raw_logits = torch.zeros((2, 5, 2, 2), dtype=torch.float)
        cached_summary = {"num_frames": 2}

        mock_cache = MagicMock(spec=CwipInferenceCache)
        mock_cache.load.return_value = (cached_summary, pred_similarity, pred_type, raw_logits)

        loader = MagicMock()
        loader.CAMERA_RESCALED_RESOLUTION_WIDTH = 4
        loader.CAMERA_RESCALED_RESOLUTION_HEIGHT = 4
        loader.object_count = 10
        loader.all_static_objects = {}
        # Per-frame fetch returns dynamic + static for each frame; this test exercises a no-static, no-dynamic path so
        # an empty dict suffices.
        loader.get_object_data_for_frame.return_value = {}

        with patch.object(processor, "_run_cwip_inference_on_videos") as mock_run:
            processor._process_frames_static(
                camera_poses={i: np.eye(4) for i in range(4)},
                camera_model=MagicMock(),
                data_loader=loader,
                timeline=ObjectCheckTimeline.build(
                    camera_video_fps=30.0,
                    camera_frame_count=6,
                    world_video_fps=30.0,
                    world_frame_count=6,
                    pose_frame_count=4,
                ),
                camera_video_path="/cam.mp4",
                world_video_path="/world.mp4",
                trial_frames=None,
                results={"processed_frame_ids": [], "track_ids": set(), "hallucination_detections": {}},
                track_output_agg={},
                cwip_cache=mock_cache,
            )
            mock_run.assert_not_called()
        mock_cache.load.assert_called_once()

    def test_debug_dump_replays_on_the_scored_grid(self) -> None:
        """The dumped type-head masks are on the grid the frame loop scores, so a replay of the dump runs and matches.

        The CWIP output (4x4) differs from the camera resolution (6x8), as with a 1280x704 control video against a
        1280x720 camera; the frame loop resizes the index maps before scoring, and the dumped masks must follow.
        """
        self.config["av.objects_single_view"]["debug_dump"] = True
        processor = self._make_processor()
        processor.vis_helper = None
        cast(MagicMock, processor.overlap_detector.process_class).return_value = ({}, {}, {})
        pred_similarity = torch.zeros((2, 4, 4), dtype=torch.long)
        pred_type = torch.full((2, 4, 4), 3, dtype=torch.long)
        raw_logits = torch.full((2, 6, 2, 2), -20.0)
        raw_logits[:, 3, 0] = 10.0
        mock_cache = MagicMock(spec=CwipInferenceCache)
        mock_cache.load.return_value = ({"num_frames": 2}, pred_similarity, pred_type, raw_logits)
        loader = MagicMock()
        loader.CAMERA_RESCALED_RESOLUTION_WIDTH = 8
        loader.CAMERA_RESCALED_RESOLUTION_HEIGHT = 6
        loader.object_count = 10
        loader.all_static_objects = {}
        loader.get_object_data_for_frame.return_value = {}
        timeline = ObjectCheckTimeline.build(
            camera_video_fps=30.0, camera_frame_count=6, world_video_fps=30.0, world_frame_count=6, pose_frame_count=4
        )

        def run(proc: ObjectProcessorStatic, cache: CwipInferenceCache | None) -> None:
            proc._process_frames_static(
                camera_poses={i: np.eye(4) for i in range(4)},
                camera_model=MagicMock(),
                data_loader=loader,
                timeline=timeline,
                camera_video_path="/cam.mp4",
                world_video_path="/world.mp4",
                trial_frames=None,
                results={"processed_frame_ids": [], "track_ids": set(), "hallucination_detections": {}},
                track_output_agg={},
                cwip_cache=cache,
            )

        run(processor, mock_cache)
        debug_dir = Path(self.temp_dir) / "debug"
        replay_masks = processor._load_type_presence_from_dump(str(debug_dir))
        assert replay_masks is not None
        self.assertEqual(tuple(replay_masks.shape), (2, 6, 8))

        self.config["av.objects_single_view"]["debug_dump"] = False
        self.config["av.objects_single_view"]["cwip_debug_dump_dir"] = str(debug_dir)
        replayer = self._make_processor()
        replayer.vis_helper = None
        cast(MagicMock, replayer.overlap_detector.process_class).return_value = ({}, {}, {})
        replayer._check_replay_supports_presence(camera_only=False)
        with patch.object(replayer, "_presence_layers", wraps=replayer._presence_layers) as spy:
            run(replayer, None)
        self.assertTrue(spy.call_args_list)
        for call in spy.call_args_list:
            frame_idx, replayed = call.args[0], call.args[4]
            self.assertIsNone(call.args[3])  # no logits in a compact-dump replay
            self.assertTrue(torch.equal(replayed[frame_idx], replay_masks[frame_idx]))

    def test_process_frames_saves_on_cache_miss(self) -> None:
        """On a cache miss, _run_cwip_inference_on_videos result must be saved."""
        processor = self._make_processor()
        processor.vis_helper = None
        cast(MagicMock, processor.overlap_detector.process_class).return_value = ({}, {}, {})

        pred_similarity = torch.zeros((2, 4, 4), dtype=torch.long)
        pred_type = torch.zeros((2, 4, 4), dtype=torch.long)
        raw_logits = torch.zeros((2, 5, 2, 2), dtype=torch.float)
        inferred_summary = {"num_frames": 2}

        mock_cache = MagicMock(spec=CwipInferenceCache)
        mock_cache.load.return_value = None  # cache miss

        loader = MagicMock()
        loader.CAMERA_RESCALED_RESOLUTION_WIDTH = 4
        loader.CAMERA_RESCALED_RESOLUTION_HEIGHT = 4
        loader.object_count = 10
        loader.all_static_objects = {}
        # Per-frame fetch returns dynamic + static for each frame; this test exercises a no-static, no-dynamic path so
        # an empty dict suffices.
        loader.get_object_data_for_frame.return_value = {}

        with patch.object(
            processor,
            "_run_cwip_inference_on_videos",
            return_value=(inferred_summary, pred_similarity, pred_type, raw_logits),
        ) as mock_run:
            processor._process_frames_static(
                camera_poses={i: np.eye(4) for i in range(4)},
                camera_model=MagicMock(),
                data_loader=loader,
                timeline=ObjectCheckTimeline.build(
                    camera_video_fps=30.0,
                    camera_frame_count=6,
                    world_video_fps=30.0,
                    world_frame_count=6,
                    pose_frame_count=4,
                ),
                camera_video_path="/cam.mp4",
                world_video_path="/world.mp4",
                trial_frames=None,
                results={"processed_frame_ids": [], "track_ids": set(), "hallucination_detections": {}},
                track_output_agg={},
                cwip_cache=mock_cache,
            )
            mock_run.assert_called_once()
        mock_cache.save.assert_called_once_with(inferred_summary, pred_similarity, pred_type, raw_logits)


class TestObjectProcessorStaticTrialMode(unittest.TestCase):
    """Tests for ``trial_frames`` semantics and per-frame index plumbing.

    These tests cover two related fixes that keep the static check's overlay in sync with the camera image:

    * ``trial_frames`` is interpreted as a count of *camera* frames (matching the dynamic check), so the static check
      inspects the same wall-time slice from the start of the video.
    * The per-output ``frame_idx`` is translated through the ``world_frame_indices`` / ``camera_frame_indices`` returned
      by ``preprocess_world_camera_video_pair`` before computing object/pose lookups and visualization indices.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.clip_id = "clip_trial"
        self.config = {
            "av.objects_single_view": {
                "overlap_check": {"traffic_light": {}, "traffic_sign": {}},
                "importance_filter": {},
                "cwip": {"sample_method": "uniform"},
            }
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def _make_processor(self) -> ObjectProcessorStatic:
        with (
            patch("checks.objects.single_view.object_processor_static.get_object_type_colors", return_value={}),
            patch(
                "checks.objects.single_view.object_processor_static.get_object_type_class_mapping",
                return_value=ObjectTypeClassMappingV3,
            ),
            patch(
                "checks.objects.single_view.object_processor_static.get_runfiles_path",
                return_value=os.path.join(os.getcwd(), "cwip", "model.safetensors"),
            ),
            patch("checks.objects.single_view.object_processor_static.CWIPInferenceHelper"),
            patch("checks.objects.single_view.object_processor_static.OverlapDetector"),
        ):
            return ObjectProcessorStatic(
                clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu", verbose="INFO"
            )

    def _make_loader(self, n_camera_poses: int = 300) -> MagicMock:
        loader = MagicMock()
        loader.CAMERA_RESCALED_RESOLUTION_WIDTH = 4
        loader.CAMERA_RESCALED_RESOLUTION_HEIGHT = 4
        loader.object_count = 10
        loader.all_static_objects = {}
        loader.get_object_data_for_frame.return_value = {}
        return loader

    def _run_with_trial(
        self,
        *,
        trial_frames: int,
        video_fps: float,
        camera_video_fps: float | None,
        cwip_num_frames: int,
        summary_extra: dict[str, Any] | None = None,
    ) -> tuple[MagicMock, MagicMock]:
        """Run ``_process_frames_static`` in trial mode and return the captured CWIP call args.

        Patches ``_run_cwip_inference_on_videos`` so we can assert what ``num_frames`` and ``sample_method`` were
        forwarded to the underlying preprocessing helper without actually decoding any video. ``video_fps`` and
        ``camera_video_fps`` build the timeline; they no longer reach the frame loop directly.
        """
        processor = self._make_processor()
        processor.vis_helper = None
        cast(MagicMock, processor.overlap_detector.process_class).return_value = ({}, {}, {})

        pred_similarity = torch.zeros((cwip_num_frames, 4, 4), dtype=torch.long)
        pred_type = torch.zeros((cwip_num_frames, 4, 4), dtype=torch.long)
        raw_logits = torch.zeros((cwip_num_frames, 5, 2, 2), dtype=torch.float)
        summary = {"num_frames": cwip_num_frames}
        if summary_extra:
            summary.update(summary_extra)

        loader = self._make_loader()

        # Provide enough poses to cover any plausible mapped_idx the loop might emit for the indices under test.
        camera_poses = {i: np.eye(4) for i in range(300)}

        timeline = ObjectCheckTimeline.build(
            camera_video_fps=camera_video_fps if camera_video_fps else video_fps,
            camera_frame_count=600,
            world_video_fps=video_fps,
            world_frame_count=600,
            trial_steps=trial_frames,
        )

        with patch.object(
            processor,
            "_run_cwip_inference_on_videos",
            return_value=(summary, pred_similarity, pred_type, raw_logits),
        ) as mock_run:
            processor._process_frames_static(
                camera_poses=camera_poses,
                camera_model=MagicMock(),
                data_loader=loader,
                timeline=timeline,
                camera_video_path="/cam.mp4",
                world_video_path="/world.mp4",
                trial_frames=trial_frames,
                results={"processed_frame_ids": [], "track_ids": set(), "hallucination_detections": {}},
                track_output_agg={},
                cwip_cache=None,
            )
        return mock_run, loader

    def test_trial_uses_first_sampling(self) -> None:
        """``--trial N`` should always force contiguous-from-start sampling."""
        mock_run, _ = self._run_with_trial(trial_frames=10, video_fps=10.0, camera_video_fps=10.0, cwip_num_frames=10)
        kwargs = mock_run.call_args.kwargs
        self.assertEqual(kwargs["sample_method"], "first")

    def test_trial_passes_step_count_through(self) -> None:
        """``--trial N`` counts evaluation steps, so N reaches CWIP unscaled."""
        mock_run, _ = self._run_with_trial(trial_frames=10, video_fps=10.0, camera_video_fps=10.0, cwip_num_frames=10)
        self.assertEqual(mock_run.call_args.kwargs["num_frames"], 10)

    def test_trial_step_count_is_independent_of_camera_rate(self) -> None:
        """A faster camera video no longer shrinks the window: N steps is N steps whatever the sources are encoded at.

        Previously ``--trial 10`` against a 30 fps camera and a 10 fps world video collapsed to 3 logical samples.
        """
        mock_run, _ = self._run_with_trial(trial_frames=10, video_fps=10.0, camera_video_fps=30.0, cwip_num_frames=10)
        self.assertEqual(mock_run.call_args.kwargs["num_frames"], 10)

    def test_trial_step_count_is_independent_of_world_rate(self) -> None:
        """The mirror case: a world video faster than the camera also leaves the step count at N."""
        mock_run, _ = self._run_with_trial(trial_frames=10, video_fps=30.0, camera_video_fps=10.0, cwip_num_frames=10)
        self.assertEqual(mock_run.call_args.kwargs["num_frames"], 10)

    def test_trial_falls_back_when_camera_fps_missing(self) -> None:
        """If the camera FPS is unavailable (``None``), assume it matches the world FPS rather than raising."""
        mock_run, _ = self._run_with_trial(
            trial_frames=7,
            video_fps=10.0,
            camera_video_fps=None,
            cwip_num_frames=7,
        )
        self.assertEqual(mock_run.call_args.kwargs["num_frames"], 7)

    def test_per_frame_indices_use_summary_when_present(self) -> None:
        """When the summary carries explicit ``logical_frame_indices``, the per-frame loop must use them to compute
        ``mapped_idx`` rather than assuming the CWIP output index is the step index.
        """
        # Simulate uniform sampling that picks non-contiguous steps (a full run over a long video).
        _mock_run, loader = self._run_with_trial(
            trial_frames=4,
            video_fps=10.0,
            camera_video_fps=30.0,
            cwip_num_frames=4,
            summary_extra={
                "logical_frame_indices": [0, 9, 18, 27],
                "world_frame_indices": [0, 9, 18, 27],
                "camera_frame_indices": [0, 27, 54, 81],
            },
        )
        # mapped_idx = rds_index(step) = step * 3.
        expected_mapped_idx = [0, 27, 54, 81]
        actual_mapped_idx = [call.args[0] for call in loader.get_object_data_for_frame.call_args_list]
        self.assertEqual(actual_mapped_idx, expected_mapped_idx)

    def test_per_frame_indices_fall_back_to_contiguous_when_missing(self) -> None:
        """Cached results from older runs carry no ``logical_frame_indices``; the loop then treats CWIP output index
        ``i`` as step ``i``, which is what a contiguous run produces anyway.
        """
        _mock_run, loader = self._run_with_trial(
            trial_frames=4,
            video_fps=10.0,
            camera_video_fps=30.0,
            cwip_num_frames=4,
        )
        expected_mapped_idx = [0, 3, 6, 9]
        actual_mapped_idx = [call.args[0] for call in loader.get_object_data_for_frame.call_args_list]
        self.assertEqual(actual_mapped_idx, expected_mapped_idx)

    def test_every_mapped_index_is_on_the_annotation_grid(self) -> None:
        """The property the fixed rate exists to guarantee: no step may address an unannotated RDS index."""
        _mock_run, loader = self._run_with_trial(
            trial_frames=8, video_fps=30.0, camera_video_fps=30.0, cwip_num_frames=8
        )
        mapped = [call.args[0] for call in loader.get_object_data_for_frame.call_args_list]
        self.assertEqual(len(mapped), 8)
        self.assertTrue(all(index % RDS_INDEX_STRIDE == 0 for index in mapped), mapped)


class TestObjectProcessorStaticVisualizationDisabled(unittest.TestCase):
    """Tests for ObjectProcessorStatic when visualization is disabled."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.clip_id = "clip_static_no_vis"
        self.config = {
            "av.objects_single_view": {
                "overlap_check": {"traffic_light": {}, "traffic_sign": {}},
                "importance_filter": {},
                "cwip": {"sample_method": "uniform"},
                "visualization": {"enabled": False},
            }
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    @patch("checks.objects.single_view.object_processor_static.get_object_type_colors", return_value={})
    @patch(
        "checks.objects.single_view.object_processor_static.get_object_type_class_mapping",
        return_value=ObjectTypeClassMappingV3,
    )
    @patch(
        "checks.objects.single_view.object_processor_static.get_runfiles_path",
        return_value=os.path.join(os.getcwd(), "cwip", "model.safetensors"),
    )
    @patch("checks.objects.single_view.object_processor_static.CWIPInferenceHelper")
    @patch("checks.objects.single_view.object_processor_static.OverlapDetector")
    def test_init_succeeds_when_visualization_disabled(
        self,
        mock_overlap: MagicMock,
        mock_cwip: MagicMock,
        _mock_runfiles: MagicMock,
        _mock_mapping: MagicMock,
        _mock_colors: MagicMock,
    ) -> None:
        """Test that ObjectProcessorStatic initializes without error when visualization is disabled.

        This tests the fix for the 'NoneType' object has no attribute 'set_mask_styles' error that occurred when
        visualization was disabled but set_mask_styles was called unconditionally.
        """
        p = ObjectProcessorStatic(
            clip_id=self.clip_id, output_dir=self.temp_dir, config=self.config, model_device="cpu", verbose="INFO"
        )

        self.assertIsNone(p.vis_helper)

        mock_cwip.from_hf_format.assert_called()
        mock_overlap.assert_called()


class TestValidateWorldCameraAspectRatio(unittest.TestCase):
    """Unit tests for the world/camera aspect-ratio guard (rejects e.g. a BEV world video)."""

    def setUp(self) -> None:
        # Bypass __init__ (which would load CWIP/SegFormer models) — the guard only reads the class-level tolerance
        # constant.
        self.processor = object.__new__(ObjectProcessorStatic)

    def test_matching_aspect_ratio_passes(self) -> None:
        """Identical 16:9 camera/world dimensions are accepted."""
        self.processor._validate_world_camera_aspect_ratio(camera_hw=(720, 1280), world_hw=(720, 1280))

    def test_control_vs_camera_scaling_tolerated(self) -> None:
        """The benign 1280x704 control vs 1280x720 camera difference (~2%) is within tolerance."""
        self.processor._validate_world_camera_aspect_ratio(camera_hw=(720, 1280), world_hw=(704, 1280))

    def test_bev_square_world_video_rejected(self) -> None:
        """A square (1:1) BEV world video against a 16:9 camera is rejected with a clear error."""
        with self.assertRaises(ValueError) as ctx:
            self.processor._validate_world_camera_aspect_ratio(camera_hw=(720, 1280), world_hw=(480, 480))
        self.assertIn("aspect ratio", str(ctx.exception))

    def test_zero_dimension_rejected(self) -> None:
        """Non-positive dimensions raise rather than divide-by-zero."""
        with self.assertRaises(ValueError):
            self.processor._validate_world_camera_aspect_ratio(camera_hw=(0, 1280), world_hw=(720, 1280))


class TestSummarizeCwipOutput(unittest.TestCase):
    """Tests for the compact CWIP debug summary (top-k type probs + uint8 index maps)."""

    def test_top_k_probs_and_indices(self) -> None:
        # 1 frame, 4 type classes, 2x2 patch grid. Build logits whose sigmoid ranks are known per patch.
        logits = torch.full((1, 4, 2, 2), -10.0)  # sigmoid ~ 0
        # patch (0,0): class 2 highest, then 1, then 3
        logits[0, 2, 0, 0], logits[0, 1, 0, 0], logits[0, 3, 0, 0] = 5.0, 0.0, -1.0
        pred_type = torch.tensor([[[2, 0], [0, 0]]])  # native-res argmax map (values < 256)
        pred_sim = torch.tensor([[[1, 0], [0, 0]]])
        out = summarize_cwip_output(pred_sim, pred_type, logits, top_k=3)

        self.assertEqual(out["type_top_idx"].dtype, np.uint8)
        self.assertEqual(out["type_top_prob_pct"].dtype, np.uint8)
        self.assertEqual(out["type_top_idx"].shape, (1, 3, 2, 2))  # (frames, k, Hp, Wp)
        # Winner at patch (0,0) is class 2 with sigmoid(5) ~ 0.993 -> 99 pct; runner-up class 1 -> 50 pct.
        self.assertEqual(out["type_top_idx"][0, 0, 0, 0], 2)
        self.assertEqual(out["type_top_idx"][0, 1, 0, 0], 1)
        self.assertEqual(int(out["type_top_prob_pct"][0, 0, 0, 0]), 99)
        self.assertEqual(int(out["type_top_prob_pct"][0, 1, 0, 0]), 50)
        # Index maps preserved as uint8.
        self.assertEqual(out["pred_type_indices"].dtype, np.uint8)
        np.testing.assert_array_equal(out["pred_type_indices"], np.array([[[2, 0], [0, 0]]], dtype=np.uint8))
        np.testing.assert_array_equal(out["pred_similarity_indices"], np.array([[[1, 0], [0, 0]]], dtype=np.uint8))

    def test_k_capped_at_num_classes_and_none_inputs(self) -> None:
        logits = torch.randn(2, 2, 3, 3)  # only 2 classes -> top_k must cap at 2
        out = summarize_cwip_output(None, None, logits, top_k=3)
        self.assertEqual(out["type_top_idx"].shape, (2, 2, 3, 3))
        self.assertNotIn("pred_type_indices", out)
        self.assertNotIn("pred_similarity_indices", out)
        self.assertEqual(summarize_cwip_output(None, None, None), {})


class TestTypePresenceMask(unittest.TestCase):
    """Unit tests for the type-head presence mask (camera-only presence, and the two-stream type-head option).

    The processor is built via ``__new__`` (bypassing model load) and parses only its ``presence`` block.
    """

    @staticmethod
    def _make_proc(type_mask_cfg: dict[str, Any], mapping: Any = None) -> ObjectProcessorStatic:
        return _proc_with_presence({"type_mask": type_mask_cfg}, mapping)

    def test_argmax_gate_is_all_present(self) -> None:
        # threshold-free: every patch is present, even background type indices; the class filter selects the type.
        proc = self._make_proc({"gate": "argmax"})
        pred_type_indices = torch.zeros((1, 2, 2), dtype=torch.long)
        raw_type_logits = torch.randn((1, 6, 2, 2))
        presence = proc._type_presence_mask(0, pred_type_indices, raw_type_logits)
        self.assertEqual(presence.dtype, torch.long)
        self.assertTrue(torch.equal(presence, torch.ones((2, 2), dtype=torch.long)))

    def test_confidence_gate_without_logits_raises(self) -> None:
        # The confidence gate is per-patch probability, so missing logits (a compact-dump replay) must fail loudly
        # rather than silently mark every patch present, which would inflate correspondence/hallucination.
        proc = self._make_proc({"gate": "confidence", "confidence_threshold": 0.5})
        pred_type_indices = torch.full((1, 2, 2), 3, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "type logits"):
            proc._type_presence_mask(0, pred_type_indices, None)

    def test_argmax_gate_without_logits_is_all_present(self) -> None:
        # The argmax gate is threshold-free and needs no logits, so a compact-dump replay stays all-present.
        proc = self._make_proc({"gate": "argmax"})
        pred_type_indices = torch.full((1, 2, 2), 3, dtype=torch.long)
        presence = proc._type_presence_mask(0, pred_type_indices, None)
        self.assertTrue(torch.equal(presence, torch.ones((2, 2), dtype=torch.long)))

    def test_confidence_threshold_gates_per_patch(self) -> None:
        proc = self._make_proc({"gate": "confidence", "confidence_threshold": 0.5})
        pred_type_indices = torch.full((1, 2, 2), 3, dtype=torch.long)
        # Winning-class (channel 3) logits: +10 -> sigmoid ~1 (present), -10 -> sigmoid ~0 (absent). Other channels
        # are strongly negative so amax picks channel 3.
        raw_type_logits = torch.full((1, 6, 2, 2), -20.0)
        raw_type_logits[0, 3] = torch.tensor([[10.0, -10.0], [10.0, -10.0]])
        presence = proc._type_presence_mask(0, pred_type_indices, raw_type_logits)
        self.assertTrue(torch.equal(presence, torch.tensor([[1, 0], [1, 0]], dtype=torch.long)))

    def test_per_class_threshold_override(self) -> None:
        mapping_cls = get_object_type_class_mapping(None)
        tl_idx = resolve_model_class_indices(
            COSMOS_EVALUATOR_OBJECT_TYPE_STRING_TO_ENUMS["traffic_light"], mapping_cls
        )[0]
        num_classes = tl_idx + 2
        pred_type_indices = torch.full((1, 2, 2), tl_idx, dtype=torch.long)
        # Winning class sits at 0.0 logit -> sigmoid 0.5 everywhere; other channels strongly negative.
        raw_type_logits = torch.full((1, num_classes, 2, 2), -20.0)
        raw_type_logits[0, tl_idx] = 0.0

        # Global threshold 0.9 rejects 0.5 everywhere...
        strict = self._make_proc({"gate": "confidence", "confidence_threshold": 0.9})
        self.assertTrue(
            torch.equal(
                strict._type_presence_mask(0, pred_type_indices, raw_type_logits),
                torch.zeros((2, 2), dtype=torch.long),
            )
        )
        # ...but a per-class override of 0.1 for traffic_light admits every traffic-light patch.
        override = self._make_proc(
            {
                "gate": "confidence",
                "confidence_threshold": 0.9,
                "per_class_thresholds": {"traffic_light": 0.1},
            }
        )
        self.assertTrue(
            torch.equal(
                override._type_presence_mask(0, pred_type_indices, raw_type_logits),
                torch.ones((2, 2), dtype=torch.long),
            )
        )


class TestPresenceLayers(unittest.TestCase):
    """Layer 0 of the correspondence and hallucination class-mask tensors, for every presence source.

    One 2x3 frame. Similarity labels: MATCH, EXTRA, MISSING across the top row and MATCH, EXTRA, MISSING again across
    the bottom. Type-head probability: confident (present) in the top row, unconfident (absent) in the bottom row. So
    each cell pairs one similarity label with one type-head verdict, and every combination appears once.
    """

    MATCH = PatchSimilarityType.MATCH.value
    EXTRA = PatchSimilarityType.EXTRA.value
    MISSING = PatchSimilarityType.MISSING.value

    def setUp(self) -> None:
        self.pred_similarity = torch.tensor([[[self.MATCH, self.EXTRA, self.MISSING]] * 2], dtype=torch.long)
        self.pred_type = torch.full((1, 2, 3), 3, dtype=torch.long)
        self.raw_type_logits = torch.full((1, 6, 2, 3), -20.0)
        self.raw_type_logits[0, 3] = torch.tensor([[10.0, 10.0, 10.0], [-10.0, -10.0, -10.0]])
        self.match = torch.tensor([[1, 0, 0], [1, 0, 0]], dtype=torch.long)
        self.extra = torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.long)
        self.presence = torch.tensor([[1, 1, 1], [0, 0, 0]], dtype=torch.long)

    @staticmethod
    def _make_proc(source: str) -> ObjectProcessorStatic:
        return _proc_with_presence({"source": source, "type_mask": {"confidence_threshold": 0.5}})

    def _layers(self, source: str, *, two_stream: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        similarity = self.pred_similarity if two_stream else None
        return self._make_proc(source)._presence_layers(0, similarity, self.pred_type, self.raw_type_logits)

    def _assert_layers(
        self, layers: tuple[torch.Tensor, torch.Tensor], match: torch.Tensor, halluc: torch.Tensor
    ) -> None:
        self.assertEqual(layers[0].dtype, torch.long)
        self.assertEqual(layers[1].dtype, torch.long)
        self.assertTrue(torch.equal(layers[0], match), f"correspondence layer {layers[0].tolist()}")
        self.assertTrue(torch.equal(layers[1], halluc), f"hallucination layer {layers[1].tolist()}")

    def test_similarity_is_the_match_and_extra_verdict(self) -> None:
        self._assert_layers(self._layers("similarity"), self.match, self.extra)

    def test_type_uses_the_type_head_mask_for_both(self) -> None:
        self._assert_layers(self._layers("type"), self.presence, self.presence)

    def test_union_ors_each_similarity_label_with_the_type_head_mask(self) -> None:
        self._assert_layers(
            self._layers("union"),
            torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.long),
            torch.tensor([[1, 1, 1], [0, 1, 0]], dtype=torch.long),
        )

    def test_intersection_ands_each_similarity_label_with_the_type_head_mask(self) -> None:
        self._assert_layers(
            self._layers("intersection"),
            torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.long),
            torch.tensor([[0, 1, 0], [0, 0, 0]], dtype=torch.long),
        )

    def test_camera_only_ignores_the_configured_source(self) -> None:
        # Without a similarity head the type-head mask is the presence layer whatever presence.source says.
        for source in PRESENCE_SOURCES:
            with self.subTest(source=source):
                self._assert_layers(self._layers(source, two_stream=False), self.presence, self.presence)

    def test_similarity_source_needs_no_type_logits(self) -> None:
        # The default two-stream path must keep working on a compact debug-dump replay, which carries no logits.
        layers = self._make_proc("similarity")._presence_layers(0, self.pred_similarity, self.pred_type, None)
        self._assert_layers(layers, self.match, self.extra)

    def test_type_head_sources_without_logits_fail_loudly(self) -> None:
        # Gating on probability without logits must not silently fall back to an all-present or all-absent mask.
        for source in ("type", "union", "intersection"):
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, "type logits"):
                self._make_proc(source)._presence_layers(0, self.pred_similarity, self.pred_type, None)

    def test_type_head_thresholds_come_from_the_type_mask_block(self) -> None:
        # A stricter type_mask threshold shrinks the type-head mask the two-stream union draws on.
        proc = _proc_with_presence({"source": "union", "type_mask": {"confidence_threshold": 1.0}})
        layers = proc._presence_layers(0, self.pred_similarity, self.pred_type, self.raw_type_logits)
        self._assert_layers(layers, self.match, self.extra)

    def test_replayed_type_presence_stands_in_for_the_logits(self) -> None:
        # A compact-dump replay has no logits; the dumped type-head masks reproduce the run's layers exactly.
        for source in ("type", "union", "intersection"):
            with self.subTest(source=source):
                proc = self._make_proc(source)
                dumped = proc._type_presence_arrays(self.pred_type, self.raw_type_logits)
                replay = torch.from_numpy(
                    np.unpackbits(dumped["type_presence_packed"], axis=-1, count=3).astype(np.int64)
                )
                expected = proc._presence_layers(0, self.pred_similarity, self.pred_type, self.raw_type_logits)
                self._assert_layers(
                    proc._presence_layers(0, self.pred_similarity, self.pred_type, None, replay), *expected
                )


class TestBuildTypeMaskFrame(unittest.TestCase):
    """The type-mask frame colors every patch by its type, or only the patches of a given presence mask."""

    def setUp(self) -> None:
        self.proc = ObjectProcessorStatic.__new__(ObjectProcessorStatic)
        self.proc.type_color_lut = np.array([[0, 0, 0], [10, 20, 30], [40, 50, 60]], dtype=np.uint8)
        self.pred_type = torch.tensor([[[1, 2], [2, 1]]])

    def test_without_presence_colors_every_patch(self) -> None:
        rgb = self.proc._build_type_mask_frame(self.pred_type, 0)
        self.assertEqual(rgb.tolist(), [[[10, 20, 30], [40, 50, 60]], [[40, 50, 60], [10, 20, 30]]])

    def test_presence_keeps_only_present_patches(self) -> None:
        presence = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
        rgb = self.proc._build_type_mask_frame(self.pred_type, 0, presence)
        self.assertEqual(rgb.tolist(), [[[10, 20, 30], [0, 0, 0]], [[0, 0, 0], [10, 20, 30]]])

    def test_lut_is_not_modified(self) -> None:
        self.proc._build_type_mask_frame(self.pred_type, 0, torch.zeros((2, 2), dtype=torch.long))
        self.assertEqual(self.proc.type_color_lut[1].tolist(), [10, 20, 30])


class TestHallucinationGates(unittest.TestCase):
    """Per-run hallucination-tracker gates: the configured gates with the run's overrides on top, as a value.

    ``presence.hallucination_overrides`` applies whenever the hallucination layer can contain type-head-only patches:
    camera-only runs, and two-stream runs whose source is ``type`` or ``union``. Other sources take none.
    """

    OVERRIDES: ClassVar[dict[str, Any]] = {"traffic_sign": {"min_cluster_area": 300, "min_track_frames": 5}}
    CONFIGURED_FRAMES: ClassVar[dict[str, int]] = {
        class_name_to_rds_label("traffic_sign"): 1,
        class_name_to_rds_label("traffic_light"): 2,
    }

    @classmethod
    def _make_proc(cls, overrides: dict[str, Any] | None = None, source: str = "union") -> ObjectProcessorStatic:
        proc = _proc_with_presence(
            {"source": source, "hallucination_overrides": overrides},
            halluc_classes={"traffic_sign": {"min_cluster_area": 150}, "traffic_light": {"min_cluster_area": 32}},
        )
        proc._hallucination_classes = {"traffic_sign": {}, "traffic_light": {}}
        proc.hallucination_detector = cast(Any, SimpleNamespace())
        proc._hallucination_min_frames_by_class = dict(cls.CONFIGURED_FRAMES)
        return proc

    def _sign_gates(self, gates: HallucinationGates) -> tuple[int | None, int]:
        return gates.min_cluster_area.get("traffic_sign"), gates.min_track_frames[
            class_name_to_rds_label("traffic_sign")
        ]

    def test_overrides_applied(self) -> None:
        proc = self._make_proc(
            {
                "traffic_sign": {"min_cluster_area": 300, "min_track_frames": 2},
                "traffic_light": {"min_cluster_area": 300, "min_track_frames": 3},
            }
        )
        gates = proc._halluc_gates_for_run(camera_only=True)
        self.assertEqual(dict(gates.min_cluster_area), {"traffic_sign": 300, "traffic_light": 300})
        self.assertEqual(
            dict(gates.min_track_frames),
            {class_name_to_rds_label("traffic_sign"): 2, class_name_to_rds_label("traffic_light"): 3},
        )

    def test_no_overrides_gives_the_configured_gates(self) -> None:
        gates = self._make_proc({})._halluc_gates_for_run(camera_only=True)
        self.assertEqual(dict(gates.min_cluster_area), {})
        self.assertEqual(dict(gates.min_track_frames), self.CONFIGURED_FRAMES)

    def test_partial_override_only_touches_given_keys(self) -> None:
        gates = self._make_proc({"traffic_sign": {"min_track_frames": 4}})._halluc_gates_for_run(camera_only=True)
        # min_cluster_area left to the detector's configured value, min_track_frames overridden.
        self.assertEqual(self._sign_gates(gates), (None, 4))

    def test_camera_only_takes_the_overrides_whatever_the_source(self) -> None:
        for source in PRESENCE_SOURCES:
            with self.subTest(source=source):
                proc = self._make_proc(self.OVERRIDES, source)
                self.assertEqual(proc._halluc_overrides_for_run(camera_only=True), self.OVERRIDES)
                self.assertEqual(self._sign_gates(proc._halluc_gates_for_run(camera_only=True)), (300, 5))

    def test_two_stream_takes_the_overrides_only_for_type_mask_sources(self) -> None:
        expected = {"similarity": {}, "type": self.OVERRIDES, "union": self.OVERRIDES, "intersection": {}}
        self.assertEqual(set(expected), set(PRESENCE_SOURCES))
        for source, overrides in expected.items():
            with self.subTest(source=source):
                proc = self._make_proc(self.OVERRIDES, source)
                self.assertEqual(proc._halluc_overrides_for_run(camera_only=False), overrides)
                gates = proc._halluc_gates_for_run(camera_only=False)
                self.assertEqual(self._sign_gates(gates), (300, 5) if overrides else (None, 1))

    def test_runs_never_modify_the_configured_gates(self) -> None:
        # A camera-only run tightens traffic_sign; a following similarity run (no overrides) sees the configured gates.
        proc = self._make_proc(self.OVERRIDES, "similarity")
        self.assertEqual(self._sign_gates(proc._halluc_gates_for_run(camera_only=True)), (300, 5))
        self.assertEqual(self._sign_gates(proc._halluc_gates_for_run(camera_only=False)), (None, 1))
        self.assertEqual(proc._hallucination_min_frames_by_class, self.CONFIGURED_FRAMES)

    def test_missing_block_or_disabled_detector_gives_no_overrides(self) -> None:
        proc = self._make_proc()
        self.assertEqual(proc._halluc_overrides_for_run(camera_only=True), {})
        self.assertEqual(proc._halluc_overrides_for_run(camera_only=False), {})
        proc.hallucination_detector = None
        gates = proc._halluc_gates_for_run(camera_only=True)
        self.assertEqual((dict(gates.min_cluster_area), dict(gates.min_track_frames)), ({}, {}))


class TestPresenceRecordAndReplay(unittest.TestCase):
    """The results' presence record, and the start-of-run check that a debug-dump replay can reproduce the run."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.pred_type = torch.full((2, 2, 3), 3, dtype=torch.long)
        self.raw_type_logits = torch.full((2, 6, 2, 3), -20.0)
        self.raw_type_logits[:, 3, 0] = 10.0

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def _write_dump(self, proc: ObjectProcessorStatic | None) -> str:
        if proc is not None:
            proc._dump_type_presence(Path(self.tmp), self.pred_type, self.raw_type_logits)
        return self.tmp

    def test_record_lists_every_setting_the_score_depends_on(self) -> None:
        proc = _proc_with_presence(
            {
                "source": "union",
                "type_mask": {"confidence_threshold": 0.6, "per_class_thresholds": {"traffic_light": 0.7}},
                "hallucination_overrides": {"traffic_sign": {"min_cluster_area": 300}},
            },
            halluc_classes={"traffic_sign": {}},
        )
        proc.hallucination_detector = cast(Any, SimpleNamespace())
        type_mask = {"gate": "confidence", "confidence_threshold": 0.6, "per_class_thresholds": {"traffic_light": 0.7}}
        overrides = {"traffic_sign": {"min_cluster_area": 300}}
        self.assertEqual(
            proc._presence_record(camera_only=False),
            {"source": "union", "type_mask": type_mask, "hallucination_overrides": overrides},
        )
        self.assertEqual(
            proc._presence_record(camera_only=True),
            {"source": "type", "type_mask": type_mask, "hallucination_overrides": overrides},
        )
        # With hallucination detection disabled no tracker gate runs, so no override is recorded as applied.
        proc.hallucination_detector = None
        self.assertEqual(proc._presence_record(camera_only=False)["hallucination_overrides"], {})
        similarity = _proc_with_presence({"source": "similarity"}, halluc_classes={"traffic_sign": {}})
        self.assertEqual(
            similarity._presence_record(camera_only=False),
            {"source": "similarity", "type_mask": None, "hallucination_overrides": {}},
        )

    def test_replay_round_trip(self) -> None:
        proc = _proc_with_presence({"source": "union"})
        proc._cwip_debug_dump_dir = self._write_dump(proc)
        proc._check_replay_supports_presence(camera_only=False)
        proc._check_replay_supports_presence(camera_only=True)
        replay = proc._load_type_presence_from_dump(proc._cwip_debug_dump_dir)
        assert replay is not None
        for i in range(2):
            self.assertTrue(torch.equal(replay[i], proc._type_presence_mask(i, self.pred_type, self.raw_type_logits)))

    def test_replay_without_dumped_masks_is_rejected_up_front(self) -> None:
        proc = _proc_with_presence({"source": "union"})
        proc._cwip_debug_dump_dir = self._write_dump(None)
        self.assertIsNone(proc._load_type_presence_from_dump(proc._cwip_debug_dump_dir))
        for camera_only in (False, True):
            with self.subTest(camera_only=camera_only), self.assertRaises(ValueError) as ctx:
                proc._check_replay_supports_presence(camera_only=camera_only)
            # The hint names the remedy for each mode (the argmax gate would make two-stream union all-present).
            self.assertIn("presence.source=similarity", str(ctx.exception))
            self.assertIn("presence.type_mask.gate=argmax", str(ctx.exception))
        # Sources and gates that do not need the logits replay without the masks.
        _with_dump(_proc_with_presence({"source": "similarity"}), self.tmp)._check_replay_supports_presence(False)
        argmax = _proc_with_presence({"source": "union", "type_mask": {"gate": "argmax"}})
        _with_dump(argmax, self.tmp)._check_replay_supports_presence(False)
        _with_dump(argmax, self.tmp)._check_replay_supports_presence(True)

    def test_replay_with_different_type_mask_settings_is_rejected(self) -> None:
        dumped = _proc_with_presence({"source": "union"})
        self._write_dump(dumped)
        stricter = _with_dump(
            _proc_with_presence({"source": "union", "type_mask": {"confidence_threshold": 0.9}}), self.tmp
        )
        with self.assertRaisesRegex(ValueError, "type_mask settings"):
            stricter._check_replay_supports_presence(camera_only=False)
        # The source is not part of the dumped mask, so a different type-mask-using source replays fine.
        _with_dump(_proc_with_presence({"source": "intersection"}), self.tmp)._check_replay_supports_presence(False)

    def test_no_replay_dir_is_always_accepted(self) -> None:
        proc = _proc_with_presence({"source": "union"})
        proc._cwip_debug_dump_dir = None
        proc._check_replay_supports_presence(camera_only=False)


def _proc_with_presence(
    presence_cfg: dict[str, Any], mapping: Any = None, halluc_classes: dict[str, Any] | None = None
) -> ObjectProcessorStatic:
    """A processor built via ``__new__`` (no model load) whose ``presence`` block has been parsed."""
    proc = ObjectProcessorStatic.__new__(ObjectProcessorStatic)
    proc.oc_config = {"presence": presence_cfg, "hallucination_detector": {"classes": halluc_classes or {}}}
    proc.seg_model = SimpleNamespace(config=SimpleNamespace(object_type_class_mapping=mapping))
    proc.hallucination_detector = None
    proc._parse_presence_config()
    return proc


def _with_dump(proc: ObjectProcessorStatic, dump_dir: str) -> ObjectProcessorStatic:
    proc._cwip_debug_dump_dir = dump_dir
    return proc


if __name__ == "__main__":
    unittest.main()
