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

"""Smoke test: importing inference must not raise.

Tagged ``gpu`` because the transitive CWIP model stack (parallelism / CWIPInference) is GPU-gated at import time,
mirroring the sibling ``cwip_test``.
"""

import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any, ClassVar
import unittest
from unittest import mock

import torch

from checks.utils.models.cwip.modules.presence import PresenceSource, TypeMaskGate
from checks.utils.models.cwip.tools import inference


def _fake_output_batch() -> dict:
    """A minimal camera-only forward output accepted by ``_print_inference_success_summary``."""
    return {
        "camera_global_features": torch.zeros(1, 4),
        "world_global_features": None,
        "camera_spatial_features_list": [torch.zeros(1, 4, 2, 2)],
        "world_spatial_features_list": None,
        "patch_type_logits": torch.zeros(1, 3, 2, 2),
        "patch_similarity_logits": None,
    }


class TestInferenceImport(unittest.TestCase):
    def test_module_imports(self) -> None:
        self.assertIsNotNone(inference)


class TestRunLoadedModelInferenceCameraOnly(unittest.TestCase):
    """CLI-wiring tests for ``run_loaded_model_inference`` camera-only behavior (no real model / GPU needed)."""

    def _make_model(self, camera_only_mode: bool = False) -> mock.MagicMock:
        model = mock.MagicMock()
        # ``run_loaded_model_inference`` reads the device off the first parameter.
        model.parameters.return_value = iter([torch.zeros(1)])
        model.image_size = (8, 8)
        model.precision = torch.float32
        model.forward.return_value = _fake_output_batch()
        # Set explicitly so the sync logic at run_loaded_model_inference entry gets a real bool,
        # not a truthy MagicMock, and the two-branch / camera-only paths are distinguishable.
        model.config.inference.camera_only_mode = camera_only_mode
        return model

    def test_test_inference_camera_only_forwards_world_none(self) -> None:
        model = self._make_model()
        inference.run_loaded_model_inference(model, test_inference=True, camera_only=True, fp_size=1)
        model.forward.assert_called_once()
        _, kwargs = model.forward.call_args
        self.assertIsNone(kwargs["world_image"])
        self.assertIsNotNone(kwargs["camera_image"])

    def test_test_inference_two_branch_forwards_world_tensor(self) -> None:
        # Control: without camera-only, a dummy world tensor is still built and forwarded.
        model = self._make_model()
        model.forward.return_value = {
            **_fake_output_batch(),
            "world_global_features": torch.zeros(1, 4),
            "world_spatial_features_list": [torch.zeros(1, 4, 2, 2)],
        }
        inference.run_loaded_model_inference(model, test_inference=True, camera_only=False, fp_size=1)
        _, kwargs = model.forward.call_args
        self.assertIsNotNone(kwargs["world_image"])

    def test_video_camera_only_skips_world(self) -> None:
        model = self._make_model()
        fake_video = {
            "camera_video": torch.zeros(2, 3, 8, 8),
            "world_video": torch.zeros(2, 3, 8, 8),
            "num_frames": 2,
        }
        with (
            mock.patch.object(
                inference.image_video_utils,
                "preprocess_world_camera_video_pair",
                return_value=fake_video,
            ) as mock_preprocess,
            mock.patch.object(inference, "merge_batch_outputs", return_value=_fake_output_batch()),
        ):
            inference.run_loaded_model_inference(
                model,
                camera_video_path="cam.mp4",
                world_video_path=None,
                output_dir=None,
                batch_size=4,
                camera_only=True,
                fp_size=1,
            )

        # Preprocessing decodes camera-only with a synthesized black world.
        _, pkwargs = mock_preprocess.call_args
        self.assertEqual(pkwargs["world_video_path"], "")
        self.assertTrue(pkwargs["create_black_world_if_missing"])

        # Forward runs with no world image, and visualization still runs.
        model.forward.assert_called_once()
        _, fkwargs = model.forward.call_args
        self.assertIsNone(fkwargs["world_image"])
        model.create_inference_visualizations.assert_called_once()


def _similarity_logits(labels: list[list[int]]) -> torch.Tensor:
    """``(1, 5, h, w)`` similarity logits whose argmax is ``labels`` (PatchSimilarityType values)."""
    one_hot = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=5).permute(2, 0, 1).float()
    return (one_hot * 10.0 - 5.0).unsqueeze(0)


def _type_logits(top_probs: list[list[float]], num_classes: int = 3) -> torch.Tensor:
    """``(1, num_classes, h, w)`` type logits: class 1 wins every patch with probability ``top_probs``."""
    top = torch.tensor(top_probs)
    probs = torch.full((num_classes, *top.shape), 0.01)
    probs[1] = top
    return torch.log(probs / (1 - probs)).unsqueeze(0)


class TestRunPresence(unittest.TestCase):
    """``run_loaded_model_inference`` with ``presence`` set: what it computes and hands to the presence writers."""

    MATCH, EXTRA, MISSING = 1, 3, 4

    def setUp(self) -> None:
        # Columns: MATCH / EXTRA / MISSING similarity labels; type-head top probability 0.9 / 0.9 / 0.3.
        self.sim = _similarity_logits([[self.MATCH, self.EXTRA, self.MISSING]])
        self.typ = _type_logits([[0.9, 0.9, 0.3]])

    def test_run_similarity_only_model(self) -> None:
        # A similarity-only checkpoint (no patch_type_logits) exports the similarity source without a comparison.
        output = {**_fake_output_batch(), "patch_similarity_logits": self.sim, "patch_type_logits": None}
        fake_video = {"camera_video": torch.zeros(1, 3, 8, 8), "world_video": torch.zeros(1, 3, 8, 8), "num_frames": 1}
        model = mock.MagicMock()
        model.parameters.return_value = iter([torch.zeros(1)])
        model.config.inference.camera_only_mode = False
        model.config.object_type_class_mapping = None
        model.forward.return_value = _fake_output_batch()
        with (
            mock.patch.object(
                inference.image_video_utils, "preprocess_world_camera_video_pair", return_value=fake_video
            ),
            mock.patch.object(inference, "merge_batch_outputs", return_value=dict(output)),
            mock.patch.object(inference, "write_presence_outputs") as mock_write,
        ):
            inference.run_loaded_model_inference(
                model,
                camera_video_path="cam.mp4",
                world_video_path="world.mp4",
                output_dir="/out",
                presence=inference.PresenceOptions(presence_source=PresenceSource.SIMILARITY),
            )
        args, kwargs = mock_write.call_args
        self.assertEqual(args[2].tolist(), [[[1, 0, 0]]])
        self.assertIsNone(kwargs["compared"])

    def test_run_writes_presence_outputs_only_when_requested(self) -> None:
        output = {**_fake_output_batch(), "patch_similarity_logits": self.sim, "patch_type_logits": self.typ}
        fake_video = {
            "camera_video": torch.zeros(1, 3, 8, 8),
            "world_video": torch.zeros(1, 3, 8, 8),
            "num_frames": 1,
            "camera_fps": 12.0,
        }
        for presence in (None, inference.PresenceOptions(presence_source=PresenceSource.INTERSECTION)):
            model = mock.MagicMock()
            model.parameters.return_value = iter([torch.zeros(1)])
            model.config.inference.camera_only_mode = False
            model.config.object_type_class_mapping = None
            model.forward.return_value = _fake_output_batch()
            with (
                mock.patch.object(
                    inference.image_video_utils, "preprocess_world_camera_video_pair", return_value=fake_video
                ),
                mock.patch.object(inference, "merge_batch_outputs", return_value=dict(output)),
                mock.patch.object(inference, "write_presence_outputs") as mock_write,
            ):
                inference.run_loaded_model_inference(
                    model,
                    camera_video_path="cam.mp4",
                    world_video_path="world.mp4",
                    output_dir="/out",
                    presence=presence,
                )
            vis_kwargs = model.create_inference_visualizations.call_args.kwargs
            if presence is None:
                mock_write.assert_not_called()
                self.assertIsNone(vis_kwargs["presence_mask"])
                self.assertIsNone(vis_kwargs["presence_label"])
                continue
            # combined_patch_vis row 3 gets the union of both checks' masks, labeled with the source.
            self.assertEqual(vis_kwargs["presence_mask"].tolist(), [[[1, 1, 0]]])
            self.assertEqual(vis_kwargs["presence_label"], "presence_source=intersection, type_confidence>=0.5")
            mock_write.assert_called_once()
            args, kwargs = mock_write.call_args
            self.assertEqual(args[0], Path("/out"))
            self.assertEqual(args[2].tolist(), [[[1, 0, 0]]])
            self.assertEqual(args[3].tolist(), [[[0, 1, 0]]])
            self.assertIs(args[4], presence)
            self.assertEqual((kwargs["camera_only"], kwargs["fps"]), (False, 12.0))
            # The similarity mask and the type-head mask are built for the comparison.
            first, second = kwargs["compared"]
            self.assertEqual((first.key, second.key), ("similarity", "type"))

    def test_run_camera_only_compares_with_default_type_threshold(self) -> None:
        output = {**_fake_output_batch(), "patch_type_logits": self.typ}
        fake_video = {
            "camera_video": torch.zeros(1, 3, 8, 8),
            "world_video": torch.zeros(1, 3, 8, 8),
            "num_frames": 1,
        }
        model = mock.MagicMock()
        model.parameters.return_value = iter([torch.zeros(1)])
        model.config.inference.camera_only_mode = True
        model.config.object_type_class_mapping = None
        model.forward.return_value = _fake_output_batch()
        with (
            mock.patch.object(
                inference.image_video_utils, "preprocess_world_camera_video_pair", return_value=fake_video
            ),
            mock.patch.object(inference, "merge_batch_outputs", return_value=dict(output)),
            mock.patch.object(inference, "write_presence_outputs") as mock_write,
        ):
            inference.run_loaded_model_inference(
                model,
                camera_video_path="cam.mp4",
                output_dir="/out",
                camera_only=True,
                presence=inference.PresenceOptions(type_confidence_threshold=0.95),
            )
        args, kwargs = mock_write.call_args
        # Type-head top probabilities are 0.9 / 0.9 / 0.3: the default 0.5 threshold keeps two patches, 0.95 none.
        self.assertEqual(args[2].tolist(), [[[0, 0, 0]]])
        first, second = kwargs["compared"]
        self.assertEqual((first.key, first.mask.tolist()), ("default_type", [[[1, 1, 0]]]))
        self.assertEqual((second.key, second.mask.tolist()), ("type", [[[0, 0, 0]]]))
        self.assertEqual((kwargs["camera_only"], kwargs["fps"]), (True, 30.0))

    def test_presence_without_output_dir_is_rejected_up_front(self) -> None:
        model = mock.MagicMock()
        with self.assertRaisesRegex(ValueError, "output_dir"):
            inference.run_loaded_model_inference(
                model, camera_video_path="cam.mp4", world_video_path="world.mp4", presence=inference.PresenceOptions()
            )
        model.forward.assert_not_called()


class TestPresenceCli(unittest.TestCase):
    """Validation of the presence-mask flags in ``main``."""

    VIDEO_ARGS: ClassVar[list[str]] = [
        "--camera_video_path",
        "cam.mp4",
        "--world_video_path",
        "world.mp4",
        "--output_dir",
        "/out",
    ]

    def _run(self, argv: list[str]) -> mock.MagicMock:
        with (
            mock.patch("sys.argv", ["inference.py", *argv]),
            mock.patch.object(inference, "load_from_hf_format"),
            mock.patch.object(inference, "run_loaded_model_inference") as mock_run,
        ):
            inference.main()
        return mock_run

    def _assert_rejected(self, argv: list[str]) -> None:
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            self._run(argv)

    def test_default_passes_no_presence(self) -> None:
        self.assertIsNone(self._run(self.VIDEO_ARGS).call_args.kwargs["presence"])

    def test_options_parsed(self) -> None:
        mock_run = self._run(
            [
                *self.VIDEO_ARGS,
                "--save_presence_masks",
                "--presence_source",
                "union",
                "--type_mask_gate",
                "argmax",
                "--type_confidence_threshold",
                "0.4",
                "--per_class_thresholds",
                "traffic_light_red=0.7, HDMAP_TRAFFIC_SIGNS=0.6",
            ]
        )
        self.assertEqual(
            mock_run.call_args.kwargs["presence"],
            inference.PresenceOptions(
                presence_source=PresenceSource.UNION,
                type_mask_gate=TypeMaskGate.ARGMAX,
                type_confidence_threshold=0.4,
                per_class_thresholds={"TRAFFIC_LIGHT_RED": 0.7, "HDMAP_TRAFFIC_SIGNS": 0.6},
            ),
        )

    def test_camera_only_needs_no_world_video(self) -> None:
        mock_run = self._run(
            ["--camera_video_path", "cam.mp4", "--output_dir", "/out", "--camera_only_mode", "--save_presence_masks"]
        )
        self.assertEqual(mock_run.call_args.kwargs["presence"], inference.PresenceOptions())

    def test_rejected_flags(self) -> None:
        cases = {
            "flag without --save_presence_masks": [*self.VIDEO_ARGS, "--presence_source", "similarity"],
            "threshold without --save_presence_masks": [*self.VIDEO_ARGS, "--type_confidence_threshold", "0.3"],
            "no output dir": ["--camera_video_path", "c.mp4", "--world_video_path", "w.mp4", "--save_presence_masks"],
            "no world video, two-stream": [
                "--camera_video_path",
                "c.mp4",
                "--output_dir",
                "/o",
                "--save_presence_masks",
            ],
            "test inference only": ["--test_inference", "--save_presence_masks"],
            "unknown source": [*self.VIDEO_ARGS, "--save_presence_masks", "--presence_source", "type_confidence"],
            "unknown gate": [*self.VIDEO_ARGS, "--save_presence_masks", "--type_mask_gate", "type_argmax"],
            "threshold above 1": [*self.VIDEO_ARGS, "--save_presence_masks", "--type_confidence_threshold", "1.5"],
            "per-class out of range": [*self.VIDEO_ARGS, "--save_presence_masks", "--per_class_thresholds", "EDGE=2"],
            "per-class not a number": [*self.VIDEO_ARGS, "--save_presence_masks", "--per_class_thresholds", "EDGE=x"],
            "per-class missing value": [*self.VIDEO_ARGS, "--save_presence_masks", "--per_class_thresholds", "EDGE"],
            "per-class unknown type": [*self.VIDEO_ARGS, "--save_presence_masks", "--per_class_thresholds", "CAR=0.5"],
        }
        for name, argv in cases.items():
            with self.subTest(name):
                self._assert_rejected(argv)


def _fake_model(precision: torch.dtype = torch.float32, image_size: tuple[int, int] | int = (8, 8)) -> mock.MagicMock:
    """A loaded-model stand-in exposing the attributes ``load_from_hf_format`` reads/writes after construction."""
    model = mock.MagicMock()
    model.precision = precision
    model.image_size = image_size
    model.camera.output_dim = 4
    model.world.output_dim = 4
    model.patch_classifier = None
    model.parameters.return_value = iter([torch.zeros(1)])
    model.to.return_value = model
    model.half.return_value = model
    return model


class _StubCwipFromHfOnly:
    """Stand-in for ``CWIPInference`` exposing only ``from_hf_format`` (no ``from_pretrained``)."""

    last_from_hf_format_kwargs: dict | None = None

    @classmethod
    def from_hf_format(cls, model_path: str, device: str) -> mock.MagicMock:
        cls.last_from_hf_format_kwargs = {"model_path": model_path, "device": device}
        return _fake_model()


class _StubCwipWithFromPretrained(_StubCwipFromHfOnly):
    """Stand-in for ``CWIPInference`` that also exposes ``from_pretrained`` (``transformers`` present)."""

    last_from_pretrained_args: tuple | None = None

    @classmethod
    def from_pretrained(cls, model_path: str, low_cpu_mem_usage: bool) -> mock.MagicMock:
        cls.last_from_pretrained_args = (model_path, low_cpu_mem_usage)
        return _fake_model()


class LoadFromHfFormatTest(unittest.TestCase):
    """``load_from_hf_format``: path resolution, true_cp_size, parallelism guards, loader selection, and post-load
    device/precision handling -- with ``CWIPInference``, ``dist``, and ``parallelism_module`` mocked so no real
    checkpoint, CUDA device, or process group is required."""

    def setUp(self) -> None:
        patchers: list[mock._patch[Any]] = [
            mock.patch.object(inference, "CWIPInference", _StubCwipFromHfOnly),
            mock.patch.object(inference.dist, "is_initialized", return_value=False),
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch.object(inference, "summary", return_value="model summary"),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_fp_size_below_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            inference.load_from_hf_format(model_path="/abs/path", fp_size=0)

    def test_none_model_path_without_runfiles_helper_raises(self) -> None:
        with mock.patch.object(inference, "get_runfiles_path", None), self.assertRaises(FileNotFoundError):
            inference.load_from_hf_format(model_path=None)

    def test_none_model_path_runfiles_miss_raises(self) -> None:
        with (
            mock.patch.object(inference, "get_runfiles_path", return_value=None),
            self.assertRaises(FileNotFoundError),
        ):
            inference.load_from_hf_format(model_path=None)

    def test_none_model_path_resolves_checkpoint_parent_via_runfiles(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            weights = os.path.join(d, "model.safetensors")
            Path(weights).write_bytes(b"")
            with mock.patch.object(inference, "get_runfiles_path", return_value=weights):
                inference.load_from_hf_format(model_path=None, device="cpu")
            kwargs = _StubCwipFromHfOnly.last_from_hf_format_kwargs
            assert kwargs is not None
            self.assertEqual(kwargs["model_path"], d)

    def test_absolute_model_path_used_as_is(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inference.load_from_hf_format(model_path=d, device="cpu")
        kwargs = _StubCwipFromHfOnly.last_from_hf_format_kwargs
        assert kwargs is not None
        self.assertEqual(kwargs["model_path"], d)

    def test_relative_model_path_resolved_via_runfiles(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(inference, "get_runfiles_path", return_value=d):
                inference.load_from_hf_format(model_path="relative/checkpoint", device="cpu")
            kwargs = _StubCwipFromHfOnly.last_from_hf_format_kwargs
            assert kwargs is not None
            self.assertEqual(kwargs["model_path"], d)

    def test_relative_model_path_falls_back_to_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "sub", "checkpoint")
            os.makedirs(sub)
            with (
                mock.patch.object(inference, "get_runfiles_path", return_value=None),
                mock.patch.dict(os.environ, {"BUILD_WORKSPACE_DIRECTORY": d}),
            ):
                inference.load_from_hf_format(model_path="sub/checkpoint", device="cpu")
            kwargs = _StubCwipFromHfOnly.last_from_hf_format_kwargs
            assert kwargs is not None
            self.assertEqual(kwargs["model_path"], sub)

    def test_relative_model_path_workspace_root_set_but_missing_falls_back_to_cwd(self) -> None:
        # BUILD_WORKSPACE_DIRECTORY is set, but the joined path doesn't exist under it -> Strategy 3
        # (resolve relative to cwd), distinct from the no-workspace-root Strategy 4 fallback.
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.object(inference, "get_runfiles_path", return_value=None),
            mock.patch.dict(os.environ, {"BUILD_WORKSPACE_DIRECTORY": d}),
        ):
            inference.load_from_hf_format(model_path="does/not/exist", device="cpu")
        expected = str(Path("does/not/exist").resolve())
        kwargs = _StubCwipFromHfOnly.last_from_hf_format_kwargs
        assert kwargs is not None
        self.assertEqual(kwargs["model_path"], expected)

    def test_relative_model_path_falls_back_to_cwd_resolve(self) -> None:
        with (
            mock.patch.object(inference, "get_runfiles_path", return_value=None),
            mock.patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("BUILD_WORKSPACE_DIRECTORY", None)
            inference.load_from_hf_format(model_path="unresolvable/checkpoint", device="cpu")
        expected = str(Path("unresolvable/checkpoint").resolve())
        kwargs = _StubCwipFromHfOnly.last_from_hf_format_kwargs
        assert kwargs is not None
        self.assertEqual(kwargs["model_path"], expected)

    def test_relative_model_path_runfiles_exception_falls_back(self) -> None:
        with (
            mock.patch.object(inference, "get_runfiles_path", side_effect=OSError("boom")),
            self.assertLogs(inference.logger, level=logging.WARNING),
        ):
            inference.load_from_hf_format(model_path="unresolvable/checkpoint", device="cpu")
        expected = str(Path("unresolvable/checkpoint").resolve())
        kwargs = _StubCwipFromHfOnly.last_from_hf_format_kwargs
        assert kwargs is not None
        self.assertEqual(kwargs["model_path"], expected)

    def test_relative_model_path_runfiles_exception_uses_existing_workspace_path(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "checkpoint")
            os.makedirs(sub)
            with (
                mock.patch.object(inference, "get_runfiles_path", side_effect=TypeError("boom")),
                mock.patch.dict(os.environ, {"BUILD_WORKSPACE_DIRECTORY": d}),
                self.assertLogs(inference.logger, level=logging.WARNING),
            ):
                inference.load_from_hf_format(model_path="checkpoint", device="cpu")
            kwargs = _StubCwipFromHfOnly.last_from_hf_format_kwargs
            assert kwargs is not None
            self.assertEqual(kwargs["model_path"], sub)

    def test_true_cp_size_read_from_config_json(self) -> None:
        # true_context_parallel_size=2 makes megatron_cp_size > 1, which routes through the multi-GPU init
        # path below -- exercise that path too rather than asserting only on the config read.
        with tempfile.TemporaryDirectory() as d:
            Path(d, "config.json").write_text(json.dumps({"true_context_parallel_size": 2}))
            with (
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch("torch.cuda.set_device"),
                mock.patch.object(inference.dist, "init_process_group"),
                mock.patch.object(inference.dist, "get_rank", return_value=0),
                mock.patch.object(inference, "initialize_model_parallel") as mock_init_parallel,
                mock.patch.object(inference.parallelism_module, "sync_from_megatron"),
            ):
                inference.load_from_hf_format(model_path=d, device="cuda")
            self.assertEqual(mock_init_parallel.call_args.kwargs["context_parallel_size"], 2)

    def test_true_cp_size_defaults_when_config_key_absent(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            Path(d, "config.json").write_text(json.dumps({}))
            inference.load_from_hf_format(model_path=d, device="cpu")
        # No exception -> true_cp_size defaulted to 1 and megatron_cp_size computed without error.

    def test_true_cp_size_defaults_when_no_config_json(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inference.load_from_hf_format(model_path=d, device="cpu")
        # No config.json present -> true_cp_size stays None then is defaulted to 1.

    def test_multi_gpu_requested_on_cpu_device_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d, self.assertRaises(RuntimeError):
            inference.load_from_hf_format(model_path=d, device="cpu", tp_size=2)

    def test_multi_gpu_requested_without_cuda_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d, self.assertRaises(RuntimeError):
            inference.load_from_hf_format(model_path=d, device="cuda", fp_size=2)

    def test_multi_gpu_init_initializes_process_group_and_parallel_state(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with (
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch("torch.cuda.set_device"),
                mock.patch.object(inference.dist, "init_process_group") as mock_init_pg,
                mock.patch.object(inference.dist, "get_rank", return_value=0),
                mock.patch.object(inference, "initialize_model_parallel") as mock_init_parallel,
                mock.patch.object(inference.parallelism_module, "sync_from_megatron") as mock_sync,
                mock.patch.dict(os.environ, {"LOCAL_RANK": "0"}),
            ):
                inference.load_from_hf_format(model_path=d, device="cuda", tp_size=2)
            mock_init_pg.assert_called_once_with(backend="nccl")
            mock_init_parallel.assert_called_once()
            mock_sync.assert_called_once()

    def test_use_hf_without_from_pretrained_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d, self.assertRaises(RuntimeError):
            inference.load_from_hf_format(model_path=d, device="cpu", use_hf=True)

    def test_use_hf_calls_from_pretrained(self) -> None:
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.object(inference, "CWIPInference", _StubCwipWithFromPretrained),
        ):
            inference.load_from_hf_format(model_path=d, device="cpu", use_hf=True)
        self.assertEqual(_StubCwipWithFromPretrained.last_from_pretrained_args, (d, False))

    def test_bfloat16_model_converted_to_half_when_unsupported(self) -> None:
        stub_model = _fake_model(precision=torch.bfloat16)
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.object(_StubCwipFromHfOnly, "from_hf_format", return_value=stub_model),
            mock.patch("torch.cuda.is_bf16_supported", return_value=False),
        ):
            inference.load_from_hf_format(model_path=d, device="cuda")
        stub_model.half.assert_called_once()

    def test_channels_last_skipped_on_non_cuda_device(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inference.load_from_hf_format(model_path=d, device="cpu", channels_last=True)
        # No exception, and the CUDA-only memory-format conversion is skipped (nothing to assert on a stub model).

    def test_channels_last_applied_on_cuda_device(self) -> None:
        stub_model = _fake_model()
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.object(_StubCwipFromHfOnly, "from_hf_format", return_value=stub_model),
        ):
            inference.load_from_hf_format(model_path=d, device="cuda", channels_last=True)
        stub_model.to.assert_any_call(memory_format=torch.channels_last)
        self.assertTrue(stub_model._use_channels_last)

    def test_camera_only_mode_sets_model_config_flag(self) -> None:
        stub_model = _fake_model()
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.object(_StubCwipFromHfOnly, "from_hf_format", return_value=stub_model),
        ):
            inference.load_from_hf_format(model_path=d, device="cpu", camera_only_mode=True)
        self.assertTrue(stub_model.config.inference.camera_only_mode)

    def test_image_size_as_int_is_treated_as_square(self) -> None:
        stub_model = _fake_model(image_size=16)
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.object(_StubCwipFromHfOnly, "from_hf_format", return_value=stub_model),
        ):
            inference.load_from_hf_format(model_path=d, device="cpu")
        # No exception -> the int branch of the image_size tuple/list check ran without error.

    def test_patch_classifier_present_is_reported(self) -> None:
        stub_model = _fake_model()
        stub_model.patch_classifier = mock.MagicMock()
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.object(_StubCwipFromHfOnly, "from_hf_format", return_value=stub_model),
        ):
            inference.load_from_hf_format(model_path=d, device="cpu")

    def test_model_summary_exception_is_caught(self) -> None:
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.object(inference, "summary", side_effect=RuntimeError("boom")),
        ):
            inference.load_from_hf_format(model_path=d, device="cpu")
        # No exception propagates -> the try/except around torchinfo.summary swallowed it.

    def test_model_summary_skipped_when_distributed(self) -> None:
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch.object(inference.dist, "is_initialized", return_value=True),
            mock.patch.object(inference.dist, "get_world_size", return_value=2),
        ):
            inference.load_from_hf_format(model_path=d, device="cpu")
        # No exception -> the distributed branch (skip torchinfo, print param count) ran.


class PrintGpuMemoryUsageTest(unittest.TestCase):
    def test_no_op_when_cuda_unavailable(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=False):
            inference._print_gpu_memory_usage()  # No exception; returns before touching CUDA memory APIs.

    def test_logs_memory_stats_when_cuda_available(self) -> None:
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.synchronize"),
            mock.patch("torch.cuda.memory_allocated", return_value=1024**2),
            mock.patch("torch.cuda.memory_reserved", return_value=2 * 1024**2),
            mock.patch("torch.cuda.max_memory_allocated", return_value=3 * 1024**2),
        ):
            inference._print_gpu_memory_usage("after test")


class MoveBatchToDeviceTest(unittest.TestCase):
    def test_moves_tensors_inside_list_values(self) -> None:
        batch = {"features_list": [torch.zeros(2), torch.ones(2)], "flag": True}
        moved = inference._move_batch_to_device(batch, "cpu")
        self.assertEqual(len(moved["features_list"]), 2)
        self.assertTrue(torch.equal(moved["features_list"][1], torch.ones(2)))
        self.assertTrue(moved["flag"])

    def test_skip_keys_left_untouched(self) -> None:
        sentinel = torch.zeros(3)
        moved = inference._move_batch_to_device({"lazy": sentinel}, "cpu", skip_keys=("lazy",))
        self.assertIs(moved["lazy"], sentinel)


class PrintInferenceSuccessSummaryTest(unittest.TestCase):
    def test_patch_similarity_logits_branch_is_logged(self) -> None:
        output_batch = {
            "camera_global_features": torch.zeros(1, 4),
            "world_global_features": None,
            "camera_spatial_features_list": [torch.zeros(1, 4, 2, 2)],
            "world_spatial_features_list": None,
            "patch_similarity_logits": torch.zeros(1, 2, 2, 2),
            "patch_type_logits": None,
        }
        inference._print_inference_success_summary(output_batch, label="Inference")
        # No exception -> the patch_similarity_logits (as opposed to patch_type_logits) branch ran.


class RunLoadedModelInferenceEdgeCasesTest(unittest.TestCase):
    """Remaining ``run_loaded_model_inference`` branches not covered by the camera-only CLI-wiring tests:
    the fp_size guard, the config/caller camera-only mismatch warning, and the frame-parallel split path."""

    def _make_model(self, camera_only_mode: bool = False) -> mock.MagicMock:
        model = mock.MagicMock()
        model.parameters.return_value = iter([torch.zeros(1)])
        model.image_size = (8, 8)
        model.precision = torch.float32
        model.config.inference.camera_only_mode = camera_only_mode
        return model

    def test_fp_size_below_one_raises(self) -> None:
        model = self._make_model()
        with self.assertRaises(ValueError):
            inference.run_loaded_model_inference(model, fp_size=0)

    def test_config_camera_only_overrides_caller_flag_with_warning(self) -> None:
        # model.config already has camera_only_mode=True; caller passes camera_only=False -> the runner
        # follows the model config and forwards world_image=None, not a dummy world tensor.
        model = self._make_model(camera_only_mode=True)
        model.forward.return_value = _fake_output_batch()
        with self.assertLogs(inference.logger, level="WARNING"):
            inference.run_loaded_model_inference(model, test_inference=True, camera_only=False, fp_size=1)
        _, kwargs = model.forward.call_args
        self.assertIsNone(kwargs["world_image"])

    def test_frame_parallel_test_inference_splits_and_gathers(self) -> None:
        model = self._make_model()
        model.forward.return_value = {
            **_fake_output_batch(),
            "world_global_features": torch.zeros(1, 4),
            "world_spatial_features_list": [torch.zeros(1, 4, 2, 2)],
        }
        with (
            mock.patch.object(inference.parallelism_module, "get_frame_parallel_world_size", return_value=2),
            mock.patch.object(inference.parallelism_module, "get_frame_parallel_rank", return_value=0),
            mock.patch.object(
                inference.parallelism_module, "split_tensor_for_fp", side_effect=lambda t, **_: t
            ) as mock_split,
            mock.patch.object(
                inference.parallelism_module, "gather_features_from_fp", side_effect=lambda b: b
            ) as mock_gather,
        ):
            inference.run_loaded_model_inference(model, test_inference=True, camera_only=False, fp_size=2)
        self.assertGreaterEqual(mock_split.call_count, 2)  # camera and world dummy tensors both split.
        mock_gather.assert_called_once()

    def test_frame_parallel_group_lookup_failure_falls_back_to_single_rank(self) -> None:
        # get_frame_parallel_world_size()/rank() raising means the FP process group was never
        # initialized; the runner must fall back to fp_world=1, fp_rank=0 rather than propagate.
        model = self._make_model()
        model.forward.return_value = _fake_output_batch()
        with mock.patch.object(
            inference.parallelism_module, "get_frame_parallel_world_size", side_effect=RuntimeError("boom")
        ):
            inference.run_loaded_model_inference(model, test_inference=True, camera_only=True, fp_size=1)
        model.forward.assert_called_once()

    def test_video_frame_parallel_splits_each_chunk(self) -> None:
        model = self._make_model()
        model.forward.return_value = _fake_output_batch()
        fake_video = {
            "camera_video": torch.zeros(2, 3, 8, 8),
            "world_video": torch.zeros(2, 3, 8, 8),
            "num_frames": 2,
        }
        with (
            mock.patch.object(
                inference.image_video_utils, "preprocess_world_camera_video_pair", return_value=fake_video
            ),
            mock.patch.object(inference, "merge_batch_outputs", return_value=_fake_output_batch()),
            mock.patch.object(inference.parallelism_module, "get_frame_parallel_world_size", return_value=2),
            mock.patch.object(inference.parallelism_module, "get_frame_parallel_rank", return_value=1),
            mock.patch.object(
                inference.parallelism_module,
                "split_tensor_for_fp",
                side_effect=lambda t, return_indices=False: (t, 0, t.shape[0]) if return_indices else t,
            ),
            mock.patch.object(inference.parallelism_module, "gather_features_from_fp", side_effect=lambda b: b),
        ):
            inference.run_loaded_model_inference(
                model,
                camera_video_path="cam.mp4",
                world_video_path="world.mp4",
                output_dir=None,
                batch_size=2,
                camera_only=False,
                fp_size=1,
            )
        model.forward.assert_called_once()

    def test_video_frame_parallel_rank_with_no_real_frames_reports_padding_only(self) -> None:
        # split_tensor_for_fp reporting real_start >= real_end means this rank's slice of the chunk
        # is pure FP padding -- the "FP padding only on this rank" branch, not the real-indices one.
        model = self._make_model()
        model.forward.return_value = _fake_output_batch()
        fake_video = {
            "camera_video": torch.zeros(1, 3, 8, 8),
            "world_video": torch.zeros(1, 3, 8, 8),
            "num_frames": 1,
        }
        with (
            mock.patch.object(
                inference.image_video_utils, "preprocess_world_camera_video_pair", return_value=fake_video
            ),
            mock.patch.object(inference, "merge_batch_outputs", return_value=_fake_output_batch()),
            mock.patch.object(inference.parallelism_module, "get_frame_parallel_world_size", return_value=2),
            mock.patch.object(inference.parallelism_module, "get_frame_parallel_rank", return_value=1),
            mock.patch.object(
                inference.parallelism_module,
                "split_tensor_for_fp",
                side_effect=lambda t, return_indices=False: (t, 0, 0) if return_indices else t,
            ),
            mock.patch.object(inference.parallelism_module, "gather_features_from_fp", side_effect=lambda b: b),
        ):
            inference.run_loaded_model_inference(
                model,
                camera_video_path="cam.mp4",
                world_video_path="world.mp4",
                output_dir=None,
                batch_size=1,
                camera_only=False,
                fp_size=1,
            )
        model.forward.assert_called_once()

    def test_video_frame_parallel_rank_with_real_frames_reports_indices(self) -> None:
        # split_tensor_for_fp reporting real_start < real_end means this rank's slice of the chunk
        # holds real (non-padding) frames -- the "real global frame indices" branch, not the padding one.
        model = self._make_model()
        model.forward.return_value = _fake_output_batch()
        fake_video = {
            "camera_video": torch.zeros(2, 3, 8, 8),
            "world_video": torch.zeros(2, 3, 8, 8),
            "num_frames": 2,
        }
        with (
            mock.patch.object(
                inference.image_video_utils, "preprocess_world_camera_video_pair", return_value=fake_video
            ),
            mock.patch.object(inference, "merge_batch_outputs", return_value=_fake_output_batch()),
            mock.patch.object(inference.parallelism_module, "get_frame_parallel_world_size", return_value=2),
            mock.patch.object(inference.parallelism_module, "get_frame_parallel_rank", return_value=0),
            mock.patch.object(
                inference.parallelism_module,
                "split_tensor_for_fp",
                side_effect=lambda t, return_indices=False: (t, 0, t.shape[0]) if return_indices else t,
            ),
            mock.patch.object(inference.parallelism_module, "gather_features_from_fp", side_effect=lambda b: b),
        ):
            inference.run_loaded_model_inference(
                model,
                camera_video_path="cam.mp4",
                world_video_path="world.mp4",
                output_dir=None,
                batch_size=2,
                camera_only=False,
                fp_size=1,
            )
        model.forward.assert_called_once()

    def test_video_two_branch_mode_prints_world_video_and_resizes(self) -> None:
        model = self._make_model()
        model.forward.return_value = {
            **_fake_output_batch(),
            "world_global_features": torch.zeros(1, 4),
            "world_spatial_features_list": [torch.zeros(1, 4, 2, 2)],
        }
        fake_video = {
            "camera_video": torch.zeros(2, 3, 8, 8),
            "world_video": torch.zeros(2, 3, 8, 8),
            "num_frames": 2,
        }
        with (
            mock.patch.object(
                inference.image_video_utils, "preprocess_world_camera_video_pair", return_value=fake_video
            ),
            mock.patch.object(inference, "merge_batch_outputs", return_value=model.forward.return_value),
        ):
            inference.run_loaded_model_inference(
                model,
                camera_video_path="cam.mp4",
                world_video_path="world.mp4",
                output_dir=None,
                batch_size=4,
                camera_only=False,
                image_size=(4, 4),
                fp_size=1,
            )
        model.forward.assert_called_once()
        _, fkwargs = model.forward.call_args
        self.assertIsNotNone(fkwargs["world_image"])
        self.assertEqual(tuple(fkwargs["world_image"].shape[-2:]), (4, 4))

    def test_video_mode_barriers_when_distributed(self) -> None:
        model = self._make_model()
        model.forward.return_value = _fake_output_batch()
        fake_video = {
            "camera_video": torch.zeros(1, 3, 8, 8),
            "world_video": torch.zeros(1, 3, 8, 8),
            "num_frames": 1,
        }
        with (
            mock.patch.object(
                inference.image_video_utils, "preprocess_world_camera_video_pair", return_value=fake_video
            ),
            mock.patch.object(inference, "merge_batch_outputs", return_value=_fake_output_batch()),
            mock.patch.object(inference.dist, "is_initialized", return_value=True),
            mock.patch.object(inference.dist, "get_world_size", return_value=2),
            mock.patch.object(inference.dist, "get_rank", return_value=0),
            mock.patch.object(inference.dist, "barrier") as mock_barrier,
        ):
            inference.run_loaded_model_inference(
                model,
                camera_video_path="cam.mp4",
                world_video_path=None,
                output_dir=None,
                batch_size=4,
                camera_only=True,
                fp_size=1,
            )
        self.assertEqual(mock_barrier.call_count, 2)


class MainCliTest(unittest.TestCase):
    """``main()``: argument validation, --image_size parsing, and the load -> run orchestration wired together with
    ``load_from_hf_format`` / ``run_loaded_model_inference`` mocked at their module-level call sites."""

    def _argv(self, *extra: str) -> list[str]:
        return ["inference", "--test_inference", *extra]

    def test_fp_size_below_one_exits(self) -> None:
        with mock.patch("sys.argv", self._argv("--fp_size", "0")), self.assertRaises(SystemExit):
            inference.main()

    def test_image_size_single_int_parsed_as_square(self) -> None:
        with (
            mock.patch("sys.argv", self._argv("--image_size", "64")),
            mock.patch.object(inference, "load_from_hf_format", return_value=mock.MagicMock()),
            mock.patch.object(inference, "run_loaded_model_inference") as mock_run,
        ):
            inference.main()
        self.assertEqual(mock_run.call_args.kwargs["image_size"], (64, 64))

    def test_image_size_non_numeric_exits(self) -> None:
        with (
            mock.patch("sys.argv", self._argv("--image_size", "not_a_number")),
            self.assertRaises(SystemExit),
        ):
            inference.main()

    def test_image_size_wrong_arity_exits(self) -> None:
        with mock.patch("sys.argv", self._argv("--image_size", "1,2,3")), self.assertRaises(SystemExit):
            inference.main()

    def test_image_size_non_positive_exits(self) -> None:
        with mock.patch("sys.argv", self._argv("--image_size", "0,64")), self.assertRaises(SystemExit):
            inference.main()

    def test_successful_run_calls_load_then_run(self) -> None:
        loaded_model = mock.MagicMock()
        with (
            mock.patch("sys.argv", self._argv()),
            mock.patch.object(inference, "load_from_hf_format", return_value=loaded_model) as mock_load,
            mock.patch.object(inference, "run_loaded_model_inference") as mock_run,
        ):
            inference.main()
        mock_load.assert_called_once()
        mock_run.assert_called_once()
        self.assertIs(mock_run.call_args.args[0], loaded_model)

    def test_exception_prints_traceback_and_exits_nonzero(self) -> None:
        with (
            mock.patch("sys.argv", self._argv()),
            mock.patch.object(inference, "load_from_hf_format", side_effect=RuntimeError("boom")),
            self.assertRaises(SystemExit) as ctx,
        ):
            inference.main()
        self.assertEqual(ctx.exception.code, 1)

    def test_destroys_process_group_in_finally_when_distributed(self) -> None:
        with (
            mock.patch("sys.argv", self._argv()),
            mock.patch.object(inference, "load_from_hf_format", return_value=mock.MagicMock()),
            mock.patch.object(inference, "run_loaded_model_inference"),
            mock.patch.object(inference.dist, "is_initialized", return_value=True),
            mock.patch.object(inference.dist, "destroy_process_group") as mock_destroy,
        ):
            inference.main()
        mock_destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
