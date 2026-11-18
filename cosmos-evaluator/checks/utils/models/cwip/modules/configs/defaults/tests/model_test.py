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

import unittest

import attrs
from omegaconf import DictConfig as LazyDict

from checks.utils.models.cwip.modules.configs.defaults.model import (
    InferenceConfig,
    ModelConfig,
)


class TestInferenceConfig(unittest.TestCase):
    """Tests for InferenceConfig attrs class."""

    def test_default_values(self) -> None:
        cfg = InferenceConfig()
        self.assertFalse(cfg.mode)
        self.assertIsNone(cfg.output_dir)
        self.assertIsNone(cfg.start_iteration)
        self.assertEqual(cfg.chunk_threshold, 128)
        self.assertEqual(cfg.camera_start_frame, 0)
        self.assertEqual(cfg.world_start_frame, 0)
        self.assertTrue(cfg.save_detailed_patch_vis)
        self.assertTrue(cfg.use_gpu_video_encoding)
        self.assertEqual(cfg.detailed_vis_subplot_resolution, (640, 360))
        self.assertEqual(cfg.patch_classifier_activation, "sigmoid")
        self.assertFalse(cfg.blacken_input_world_frames)
        self.assertFalse(cfg.camera_only_mode)

    def test_custom_values(self) -> None:
        cfg = InferenceConfig(
            mode=True,
            output_dir="/tmp/output",
            start_iteration=100,
            chunk_threshold=256,
            camera_start_frame=10,
            world_start_frame=5,
            save_detailed_patch_vis=False,
            use_gpu_video_encoding=False,
            detailed_vis_subplot_resolution=(1280, 720),
            patch_classifier_activation="softmax",
            blacken_input_world_frames=True,
            camera_only_mode=True,
        )
        self.assertTrue(cfg.mode)
        self.assertEqual(cfg.output_dir, "/tmp/output")
        self.assertEqual(cfg.start_iteration, 100)
        self.assertEqual(cfg.chunk_threshold, 256)
        self.assertEqual(cfg.camera_start_frame, 10)
        self.assertEqual(cfg.world_start_frame, 5)
        self.assertFalse(cfg.save_detailed_patch_vis)
        self.assertFalse(cfg.use_gpu_video_encoding)
        self.assertEqual(cfg.detailed_vis_subplot_resolution, (1280, 720))
        self.assertEqual(cfg.patch_classifier_activation, "softmax")
        self.assertTrue(cfg.blacken_input_world_frames)
        self.assertTrue(cfg.camera_only_mode)

    def test_separate_instances(self) -> None:
        cfg1 = InferenceConfig()
        cfg2 = InferenceConfig()
        self.assertIsNot(cfg1, cfg2)

    def test_asdict_serializes_fields(self) -> None:
        """attrs.asdict exposes the attrs-generated fields, so custom values survive serialization."""
        cfg = InferenceConfig(mode=True, chunk_threshold=256, blacken_input_world_frames=True, camera_only_mode=True)
        as_dict = attrs.asdict(cfg)
        self.assertTrue(as_dict["mode"])
        self.assertEqual(as_dict["chunk_threshold"], 256)
        self.assertTrue(as_dict["blacken_input_world_frames"])
        self.assertTrue(as_dict["camera_only_mode"])


class TestModelConfig(unittest.TestCase):
    """Tests for ModelConfig attrs class."""

    def test_defaults_with_required_field(self) -> None:
        cam_enc = LazyDict({"encoder": "test"})
        cfg = ModelConfig(camera_encoder=cam_enc)
        self.assertIs(cfg.camera_encoder, cam_enc)
        self.assertIsNone(cfg.world_encoder)
        self.assertIsNone(cfg.patch_classifier)
        self.assertEqual(cfg.precision, "bfloat16")
        self.assertIsNone(cfg.checkpoint_path)
        self.assertIsInstance(cfg.inference, InferenceConfig)
        self.assertIsNone(cfg.object_type_class_mapping)
        # Backbone-feature viz capture is off by default; flipping it on requests the pre-alignment layer in addition to
        # one, populating row 2 of the debug 3x3 tile.
        self.assertFalse(cfg.viz_capture_backbone_features)

    def test_inference_factory_creates_new_instances(self) -> None:
        cam_enc = LazyDict({"encoder": "test"})
        cfg1 = ModelConfig(camera_encoder=cam_enc)
        cfg2 = ModelConfig(camera_encoder=cam_enc)
        self.assertIsNot(cfg1.inference, cfg2.inference)

    def test_custom_values(self) -> None:
        cam_enc = LazyDict({"encoder": "camera"})
        world_enc = LazyDict({"encoder": "world"})
        patch_cls = LazyDict({"classifier": "patch"})
        inference_cfg = InferenceConfig(mode=True, output_dir="/tmp")
        cfg = ModelConfig(
            camera_encoder=cam_enc,
            world_encoder=world_enc,
            patch_classifier=patch_cls,
            precision="float32",
            checkpoint_path="/path/to/ckpt",
            inference=inference_cfg,
            object_type_class_mapping="v3",
        )
        self.assertIs(cfg.camera_encoder, cam_enc)
        self.assertIs(cfg.world_encoder, world_enc)
        self.assertIs(cfg.patch_classifier, patch_cls)
        self.assertEqual(cfg.precision, "float32")
        self.assertEqual(cfg.checkpoint_path, "/path/to/ckpt")
        self.assertIs(cfg.inference, inference_cfg)
        self.assertEqual(cfg.object_type_class_mapping, "v3")

    def test_inference_defaults_are_correct(self) -> None:
        cam_enc = LazyDict({"encoder": "test"})
        cfg = ModelConfig(camera_encoder=cam_enc)
        self.assertFalse(cfg.inference.mode)
        self.assertEqual(cfg.inference.chunk_threshold, 128)
        self.assertEqual(cfg.inference.patch_classifier_activation, "sigmoid")


if __name__ == "__main__":
    unittest.main()
