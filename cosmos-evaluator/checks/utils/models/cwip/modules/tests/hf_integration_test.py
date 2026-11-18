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

"""Tests for :mod:`checks.utils.models.cwip.modules.hf_integration`."""

import json
import os
import tempfile
import unittest
from unittest import mock

from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from omegaconf import DictConfig
from third_party.perception_models.core.vision_encoder.rope import Rope2D
import torch
from torch import nn
from transformers import PreTrainedModel

from checks.utils.models.cwip.modules import hf_integration as hf_integration_module
from checks.utils.models.cwip.modules.configs.defaults.model import ModelConfig


def _make_minimal_model_config() -> ModelConfig:
    """Build a minimal :class:`ModelConfig` that satisfies attrs' required field."""
    return ModelConfig(camera_encoder=DictConfig({"encoder_cfg_name": "dummy"}))  # type: ignore[call-arg]


class _DummyPreTrainedModel(hf_integration_module.CWIPPreTrainedModel):
    """Concrete :class:`CWIPPreTrainedModel` subclass for testing the base hooks."""

    @staticmethod
    def _reconstruct_config_from_dict(saved_config_dict: dict) -> ModelConfig:
        cfg = _make_minimal_model_config()
        cfg.precision = saved_config_dict.get("precision", cfg.precision)
        return cfg

    def _instantiate_encoders_and_decoders(self) -> None:
        # Trivial submodule so `self.modules()` walks something.
        self.body = nn.Linear(4, 4)

    def _initialize_remaining_parameters(self) -> None:
        self.scale = nn.Parameter(torch.ones(1))


class TestResolveToLocalDir(unittest.TestCase):
    """Tests for :func:`_resolve_to_local_dir`."""

    def test_existing_dir_returned_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(hf_integration_module, "snapshot_download") as m:
                resolved = hf_integration_module._resolve_to_local_dir(tmp)
            self.assertEqual(resolved, tmp)
            m.assert_not_called()

    def test_local_files_only_with_missing_path_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            hf_integration_module._resolve_to_local_dir("/nonexistent/path/for/test", local_files_only=True)

    def test_repo_id_calls_snapshot_download(self) -> None:
        with mock.patch.object(
            hf_integration_module,
            "snapshot_download",
            return_value="/tmp/fake_cache_dir",
        ) as m:
            resolved = hf_integration_module._resolve_to_local_dir(
                "test-org/test-model",
                revision="v1.0.0",
                cache_dir="/tmp/cache",
                token="dummy-token",
                local_files_only=False,
            )
        self.assertEqual(resolved, "/tmp/fake_cache_dir")
        m.assert_called_once()
        call_kwargs = m.call_args.kwargs
        self.assertEqual(call_kwargs["repo_id"], "test-org/test-model")
        self.assertEqual(call_kwargs["revision"], "v1.0.0")
        self.assertEqual(call_kwargs["cache_dir"], "/tmp/cache")
        self.assertEqual(call_kwargs["token"], "dummy-token")
        self.assertFalse(call_kwargs["local_files_only"])
        # Allow-list pinned to known checkpoint files.
        self.assertIn("config.json", call_kwargs["allow_patterns"])
        self.assertIn("model.safetensors", call_kwargs["allow_patterns"])

    def _make_fake_hub_response(self) -> mock.MagicMock:
        """Build a stand-in for ``requests.Response`` accepted by the HF errors.

        ``huggingface_hub.errors.HfHubHTTPError`` (and its :class:`RepositoryNotFoundError` subclass) read
        ``response.headers`` and ``response.text`` in ``__init__``, so we provide a mock with both.
        """
        resp = mock.MagicMock()
        resp.headers.get.return_value = None
        resp.text = ""
        return resp

    def test_repository_not_found_translated_to_file_not_found(self) -> None:
        err = RepositoryNotFoundError("no such repo", response=self._make_fake_hub_response())
        with (
            mock.patch.object(
                hf_integration_module,
                "snapshot_download",
                side_effect=err,
            ),
            self.assertRaises(FileNotFoundError) as ctx,
        ):
            hf_integration_module._resolve_to_local_dir("org/missing-repo")
        self.assertIn("neither a local directory", str(ctx.exception))

    def test_hub_http_error_translated_to_file_not_found(self) -> None:
        err = HfHubHTTPError("503 service unavailable", response=self._make_fake_hub_response())
        with (
            mock.patch.object(
                hf_integration_module,
                "snapshot_download",
                side_effect=err,
            ),
            self.assertRaises(FileNotFoundError) as ctx,
        ):
            hf_integration_module._resolve_to_local_dir("org/some-repo")
        self.assertIn("Failed to download", str(ctx.exception))


class TestHFModelConfig(unittest.TestCase):
    """Tests for :class:`HFModelConfig`."""

    def test_init_with_dict_stores_verbatim(self) -> None:
        raw = {"camera_encoder": {"encoder_cfg_name": "foo"}, "precision": "bfloat16"}
        config = hf_integration_module.HFModelConfig(cwip_config_dict=raw)
        self.assertEqual(config.cwip_config_dict, raw)
        self.assertEqual(config.model_type, "cwip")

    def test_init_with_none_yields_empty_dict(self) -> None:
        config = hf_integration_module.HFModelConfig(cwip_config_dict=None)
        self.assertEqual(config.cwip_config_dict, {})

    def test_init_defaults_to_empty_dict(self) -> None:
        config = hf_integration_module.HFModelConfig()
        self.assertEqual(config.cwip_config_dict, {})

    def test_from_pretrained_loads_local_config_json(self) -> None:
        raw = {"precision": "float32", "camera_encoder": {"name": "X"}}
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump(raw, f)
            config = hf_integration_module.HFModelConfig.from_pretrained(tmp)
        self.assertIsInstance(config, hf_integration_module.HFModelConfig)
        assert isinstance(config, hf_integration_module.HFModelConfig)
        self.assertEqual(config.cwip_config_dict, raw)

    def test_from_pretrained_return_unused_kwargs(self) -> None:
        raw = {"foo": "bar"}
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump(raw, f)
            result = hf_integration_module.HFModelConfig.from_pretrained(
                tmp, return_unused_kwargs=True, my_extra="keep me", trust_remote_code=True
            )
        self.assertIsInstance(result, tuple)
        config, unused = result
        self.assertIsInstance(config, hf_integration_module.HFModelConfig)
        # `trust_remote_code` is a recognized HF kwarg and must be consumed.
        self.assertNotIn("trust_remote_code", unused)
        # Unknown extra kwargs must be passed through.
        self.assertEqual(unused.get("my_extra"), "keep me")

    def test_from_pretrained_missing_config_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError):
            hf_integration_module.HFModelConfig.from_pretrained(tmp)

    def test_from_pretrained_pops_internal_kwargs(self) -> None:
        raw = {"foo": "bar"}
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w") as f:
                json.dump(raw, f)
            # All listed kwargs are stripped by from_pretrained before forwarding.
            config = hf_integration_module.HFModelConfig.from_pretrained(
                tmp,
                cache_dir=None,
                force_download=False,
                resume_download=None,
                proxies=None,
                token=None,
                local_files_only=False,
                revision=None,
                trust_remote_code=False,
                subfolder="",
                _from_pipeline=None,
                _from_auto=False,
                _commit_hash=None,
            )
        self.assertIsInstance(config, hf_integration_module.HFModelConfig)


class TestCWIPPreTrainedModelInit(unittest.TestCase):
    """Tests for the two-branch :meth:`CWIPPreTrainedModel.__init__`."""

    def test_direct_construction_with_model_config(self) -> None:
        cfg = _make_minimal_model_config()
        model = _DummyPreTrainedModel(cfg)
        self.assertIs(model.config, cfg)
        self.assertIsNone(model.hf_config)
        # Subclass hooks ran:
        self.assertIsInstance(model.body, nn.Linear)
        self.assertTrue(hasattr(model, "scale"))

    def test_hf_construction_swaps_config_and_stamps_fields(self) -> None:
        raw = {"precision": "float16"}
        hf_cfg = hf_integration_module.HFModelConfig(cwip_config_dict=raw)
        model = _DummyPreTrainedModel(hf_cfg)
        self.assertIs(model.hf_config, hf_cfg)
        self.assertIsInstance(model.config, ModelConfig)
        self.assertEqual(model.config.precision, "float16")
        # HF runtime stamps:
        self.assertEqual(model.config.model_type, "cwip")
        self.assertIsNone(model.config.base_model_tp_plan)
        self.assertIsNone(model.config.base_model_ep_plan)
        self.assertIsNone(model.config.base_model_pp_plan)
        self.assertIsNone(model.config.base_model_fsdp_plan)

    def test_abstract_reconstruct_raises_not_implemented(self) -> None:
        # Calling the unimplemented base hook directly must raise.
        with self.assertRaises(NotImplementedError):
            hf_integration_module.CWIPPreTrainedModel._reconstruct_config_from_dict({})

    def test_abstract_instantiate_encoders_raises_not_implemented(self) -> None:
        cfg = _make_minimal_model_config()
        # Build a partial instance that does NOT run the abstract hooks; we need a freshly-allocated CWIPPreTrainedModel
        # with `self.config` set.
        partial = hf_integration_module.CWIPPreTrainedModel.__new__(hf_integration_module.CWIPPreTrainedModel)
        nn.Module.__init__(partial)
        partial.config = cfg  # type: ignore[assignment]
        with self.assertRaises(NotImplementedError):
            partial._instantiate_encoders_and_decoders()

    def test_abstract_initialize_remaining_parameters_raises_not_implemented(self) -> None:
        partial = hf_integration_module.CWIPPreTrainedModel.__new__(hf_integration_module.CWIPPreTrainedModel)
        nn.Module.__init__(partial)
        with self.assertRaises(NotImplementedError):
            partial._initialize_remaining_parameters()


class TestCWIPPreTrainedModelFromPretrained(unittest.TestCase):
    """Tests for the :meth:`CWIPPreTrainedModel.from_pretrained` post-load hooks."""

    def _make_loaded_model(self) -> _DummyPreTrainedModel:
        cfg = _make_minimal_model_config()
        return _DummyPreTrainedModel(cfg)

    def test_rope2d_re_materialization_called(self) -> None:
        loaded = self._make_loaded_model()
        # Attach a Rope2D via __dict__ (so it's not registered as a child module), matching the real layout where Rope2D
        # is a plain Python container.
        rope_a = Rope2D(dim=32)
        rope_b = Rope2D(dim=32)
        loaded.__dict__["rope"] = rope_a
        loaded.body.__dict__["rope"] = rope_b
        # Pre-condition: Rope2D.init_tensors hasn't been called.
        self.assertFalse(hasattr(rope_a, "rope"))
        self.assertFalse(hasattr(rope_b, "rope"))

        with mock.patch.object(
            PreTrainedModel,
            "from_pretrained",
            return_value=loaded,
        ) as m:
            returned = _DummyPreTrainedModel.from_pretrained("/some/path")

        m.assert_called_once()
        self.assertIs(returned, loaded)
        # Post-condition: init_tensors invoked on every Rope2D.
        self.assertTrue(hasattr(rope_a, "rope"))
        self.assertTrue(hasattr(rope_b, "rope"))

    def test_from_pretrained_with_no_rope2d_runs_clean(self) -> None:
        loaded = self._make_loaded_model()
        with mock.patch.object(
            PreTrainedModel,
            "from_pretrained",
            return_value=loaded,
        ):
            returned = _DummyPreTrainedModel.from_pretrained("/some/path")
        self.assertIs(returned, loaded)


if __name__ == "__main__":
    unittest.main()
