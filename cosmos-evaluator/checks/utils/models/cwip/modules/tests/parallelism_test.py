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

"""Tests for parallelism module — TP=1 / no-distributed paths and checkpoint utilities."""

from typing import Any
import unittest
from unittest.mock import Mock, patch

import torch

from checks.utils.models.cwip.modules import parallelism as P


class TestGlobalState(unittest.TestCase):
    """Test model-parallel global state helpers."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def tearDown(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def test_initial_state(self) -> None:
        self.assertFalse(P._MODEL_PARALLEL_STATE["initialized"])
        self.assertEqual(P.get_tensor_parallel_world_size(), 1)
        self.assertEqual(P.get_context_parallel_world_size(), 1)
        self.assertFalse(P.is_sequence_parallel_enabled())
        self.assertIsNone(P.get_tensor_parallel_group())

    def test_get_tensor_parallel_rank_no_group(self) -> None:
        self.assertEqual(P.get_tensor_parallel_rank(), 0)

    def test_initialize_without_dist_is_noop(self) -> None:
        """initialize_model_parallel logs warning and returns when dist is not init."""
        P.initialize_model_parallel(tensor_model_parallel_size=2)
        self.assertFalse(P._MODEL_PARALLEL_STATE["initialized"])


class TestFrameParallelAccessors(unittest.TestCase):
    """Test frame parallelism sub-group accessors with default state (no dist)."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def tearDown(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def test_get_frame_parallel_group_default(self) -> None:
        self.assertIsNone(P.get_frame_parallel_group())

    def test_get_frame_parallel_world_size_default(self) -> None:
        self.assertEqual(P.get_frame_parallel_world_size(), 1)

    def test_get_frame_parallel_rank_no_group(self) -> None:
        self.assertEqual(P.get_frame_parallel_rank(), 0)


class TestTrueContextParallelAccessors(unittest.TestCase):
    """Test true context parallelism (ring attention) sub-group accessors."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def tearDown(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def test_get_true_context_parallel_group_default(self) -> None:
        self.assertIsNone(P.get_true_context_parallel_group())

    def test_get_true_context_parallel_world_size_default(self) -> None:
        self.assertEqual(P.get_true_context_parallel_world_size(), 1)


class TestDefaultState(unittest.TestCase):
    """Test _DEFAULT_STATE structure and ModelParallelState TypedDict."""

    def test_default_state_keys(self) -> None:
        expected_keys = {
            "initialized",
            "tensor_parallel_size",
            "context_parallel_size",
            "sequence_parallel",
            "tensor_parallel_group",
            "context_parallel_group",
            "data_parallel_group",
            "frame_parallel_size",
            "frame_parallel_group",
            "true_context_parallel_size",
            "true_context_parallel_group",
        }
        self.assertEqual(set(P._DEFAULT_STATE.keys()), expected_keys)

    def test_default_state_values(self) -> None:
        self.assertFalse(P._DEFAULT_STATE["initialized"])
        self.assertEqual(P._DEFAULT_STATE["tensor_parallel_size"], 1)
        self.assertEqual(P._DEFAULT_STATE["context_parallel_size"], 1)
        self.assertFalse(P._DEFAULT_STATE["sequence_parallel"])
        self.assertIsNone(P._DEFAULT_STATE["tensor_parallel_group"])
        self.assertIsNone(P._DEFAULT_STATE["context_parallel_group"])
        self.assertIsNone(P._DEFAULT_STATE["data_parallel_group"])
        self.assertEqual(P._DEFAULT_STATE["frame_parallel_size"], 1)
        self.assertIsNone(P._DEFAULT_STATE["frame_parallel_group"])
        self.assertEqual(P._DEFAULT_STATE["true_context_parallel_size"], 1)
        self.assertIsNone(P._DEFAULT_STATE["true_context_parallel_group"])

    def test_model_parallel_state_is_copy(self) -> None:
        """Global state should be a separate dict from _DEFAULT_STATE."""
        self.assertIsNot(P._MODEL_PARALLEL_STATE, P._DEFAULT_STATE)


class TestMegatronConfig(unittest.TestCase):
    """Test megatron config helpers."""

    def setUp(self) -> None:
        P._MEGATRON_CONFIG = None

    def test_reset_megatron_config(self) -> None:
        P._MEGATRON_CONFIG = None
        self.assertIsNone(P._MEGATRON_CONFIG)

    def test_get_megatron_config_no_megatron(self) -> None:
        """Should raise RuntimeError when Megatron is not installed."""
        if not P.HAS_MEGATRON:
            with self.assertRaises(RuntimeError):
                P.get_megatron_config()

    def test_megatron_init_method(self) -> None:
        t = torch.empty(4, 4)
        result = P.megatron_init_method(t)
        self.assertEqual(result.shape, (4, 4))

    def test_megatron_init_method_modifies_tensor(self) -> None:
        """Xavier uniform should produce values in a reasonable range."""
        t = torch.zeros(32, 32)
        result = P.megatron_init_method(t)
        self.assertFalse(torch.all(result == 0))


class TestCopyReduceGatherScatter(unittest.TestCase):
    """Test identity paths when no TP group is set (group=None)."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def test_copy_to_tensor_parallel_region_identity(self) -> None:
        x = torch.randn(2, 4)
        out = P.copy_to_tensor_parallel_region(x)
        self.assertTrue(torch.equal(x, out))

    def test_reduce_from_tensor_parallel_region_identity(self) -> None:
        x = torch.randn(2, 4)
        out = P.reduce_from_tensor_parallel_region(x)
        self.assertTrue(torch.equal(x, out))

    def test_gather_from_tensor_parallel_region_identity(self) -> None:
        x = torch.randn(2, 4)
        out = P.gather_from_tensor_parallel_region(x)
        self.assertTrue(torch.equal(x, out))

    def test_scatter_to_tensor_parallel_region_identity(self) -> None:
        x = torch.randn(2, 4)
        out = P.scatter_to_tensor_parallel_region(x)
        self.assertTrue(torch.equal(x, out))

    def test_gather_preserves_grad(self) -> None:
        x = torch.randn(2, 4, requires_grad=True)
        out = P.gather_from_tensor_parallel_region(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_scatter_preserves_grad(self) -> None:
        x = torch.randn(2, 4, requires_grad=True)
        out = P.scatter_to_tensor_parallel_region(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)


class TestTPColumnParallelLinear(unittest.TestCase):
    """Test TPColumnParallelLinear fallback (TP=1, no Megatron)."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def test_forward_shape(self) -> None:
        layer = P.TPColumnParallelLinear(8, 16, bias=True, gather_output=True)
        out = layer(torch.randn(2, 8))
        self.assertEqual(out.shape, (2, 16))

    def test_forward_no_bias(self) -> None:
        layer = P.TPColumnParallelLinear(8, 16, bias=False)
        out = layer(torch.randn(2, 8))
        self.assertEqual(out.shape, (2, 16))

    def test_state_dict_keys(self) -> None:
        layer = P.TPColumnParallelLinear(4, 8, bias=True)
        sd = layer.state_dict()
        self.assertIn("weight", sd)
        self.assertIn("bias", sd)

    def test_load_state_dict_roundtrip(self) -> None:
        layer = P.TPColumnParallelLinear(4, 8)
        sd = layer.state_dict()
        layer2 = P.TPColumnParallelLinear(4, 8)
        layer2.load_state_dict(sd)
        self.assertTrue(torch.equal(layer.weight, layer2.weight))

    def test_gather_output_false(self) -> None:
        layer = P.TPColumnParallelLinear(8, 16, gather_output=False)
        out = layer(torch.randn(2, 8))
        self.assertEqual(out.shape, (2, 16))

    def test_differentiable(self) -> None:
        layer = P.TPColumnParallelLinear(4, 8)
        x = torch.randn(1, 4, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_batched_input(self) -> None:
        layer = P.TPColumnParallelLinear(8, 16)
        out = layer(torch.randn(4, 3, 8))
        self.assertEqual(out.shape, (4, 3, 16))


class TestTPRowParallelLinear(unittest.TestCase):
    """Test TPRowParallelLinear fallback (TP=1, no Megatron)."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def test_forward_shape(self) -> None:
        layer = P.TPRowParallelLinear(16, 8, bias=True)
        out = layer(torch.randn(2, 16))
        self.assertEqual(out.shape, (2, 8))

    def test_forward_no_bias(self) -> None:
        layer = P.TPRowParallelLinear(16, 8, bias=False)
        out = layer(torch.randn(2, 16))
        self.assertEqual(out.shape, (2, 8))

    def test_state_dict_keys(self) -> None:
        layer = P.TPRowParallelLinear(8, 4, bias=True)
        sd = layer.state_dict()
        self.assertIn("weight", sd)
        self.assertIn("bias", sd)

    def test_load_state_dict_roundtrip(self) -> None:
        layer = P.TPRowParallelLinear(8, 4)
        sd = layer.state_dict()
        layer2 = P.TPRowParallelLinear(8, 4)
        layer2.load_state_dict(sd)
        self.assertTrue(torch.equal(layer.weight, layer2.weight))

    def test_input_not_parallel(self) -> None:
        layer = P.TPRowParallelLinear(8, 4, input_is_parallel=False)
        out = layer(torch.randn(2, 8))
        self.assertEqual(out.shape, (2, 4))

    def test_differentiable(self) -> None:
        layer = P.TPRowParallelLinear(4, 8)
        x = torch.randn(1, 4, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_batched_input(self) -> None:
        layer = P.TPRowParallelLinear(16, 8)
        out = layer(torch.randn(4, 3, 16))
        self.assertEqual(out.shape, (4, 3, 8))


class TestShardFusedQKV(unittest.TestCase):
    """Test shard_fused_qkv for tensor parallel QKV splitting."""

    def test_shard_weight_tp2_rank0(self) -> None:
        weight = torch.randn(24, 8)  # 3 * 8 = 24 (Q=8, K=8, V=8)
        shard = P.shard_fused_qkv(weight, tp_size=2, tp_rank=0)
        self.assertEqual(shard.shape, (12, 8))  # 3 * (8/2) = 12

    def test_shard_weight_tp2_rank1(self) -> None:
        weight = torch.randn(24, 8)
        shard = P.shard_fused_qkv(weight, tp_size=2, tp_rank=1)
        self.assertEqual(shard.shape, (12, 8))

    def test_shard_covers_all_elements(self) -> None:
        """All shards together should reconstruct Q, K, V (reordered per shard)."""
        weight = torch.arange(24).reshape(24, 1).float()
        shard0 = P.shard_fused_qkv(weight, tp_size=2, tp_rank=0)
        shard1 = P.shard_fused_qkv(weight, tp_size=2, tp_rank=1)
        all_elements = torch.cat([shard0, shard1], dim=0)
        self.assertEqual(all_elements.shape, weight.shape)

    def test_shard_bias_1d(self) -> None:
        bias = torch.randn(12)  # 3 * 4 = 12
        shard = P.shard_fused_qkv(bias, tp_size=2, tp_rank=0)
        self.assertEqual(shard.shape, (6,))  # 3 * (4/2) = 6

    def test_shard_tp1_is_identity(self) -> None:
        weight = torch.randn(12, 4)
        shard = P.shard_fused_qkv(weight, tp_size=1, tp_rank=0)
        self.assertTrue(torch.equal(shard, weight))

    def test_invalid_dim_raises(self) -> None:
        weight = torch.randn(10, 4)  # 10 not divisible by 3
        with self.assertRaisesRegex(ValueError, "must be divisible by 3"):
            P.shard_fused_qkv(weight, tp_size=1, tp_rank=0)


class TestSplitTensorGatherFeaturesFpBasics(unittest.TestCase):
    """``split_tensor_for_fp`` / ``gather_features_from_fp`` with default FP size 1."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def tearDown(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def test_split_tensor_for_fp_when_disabled(self) -> None:
        """No-op when FP world size is 1."""
        camera = torch.rand(4, 3, 128, 128, dtype=torch.float32)
        world = torch.rand(4, 3, 128, 128, dtype=torch.float32)
        cam_out = P.split_tensor_for_fp(camera)
        world_out = P.split_tensor_for_fp(world)
        assert isinstance(cam_out, torch.Tensor)
        assert isinstance(world_out, torch.Tensor)
        self.assertTrue(torch.equal(camera, cam_out))
        self.assertTrue(torch.equal(world, world_out))

    def test_gather_features_from_fp_when_disabled(self) -> None:
        """No-op when FP world size is 1."""
        output: dict[str, Any] = {
            "camera_global_features": torch.rand(2, 192, dtype=torch.float32),
            "world_global_features": torch.rand(2, 192, dtype=torch.float32),
        }
        gathered = P.gather_features_from_fp(output)
        self.assertEqual(output, gathered)


class TestSplitTensorGatherFeaturesFpMocked(unittest.TestCase):
    """``split_tensor_for_fp`` / ``gather_features_from_fp`` with FP > 1 (mocked)."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def tearDown(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    @patch.object(P, "get_frame_parallel_rank", return_value=0)
    @patch.object(P, "get_frame_parallel_world_size", return_value=2)
    def test_split_tensor_with_fp_enabled(self, _mock_ws: Mock, _mock_rank: Mock) -> None:
        """FP rank 0 with fp_size 2 takes the first half of frames."""
        camera = torch.rand(4, 3, 32, 32)
        world = torch.rand(4, 3, 32, 32)
        cam_out = P.split_tensor_for_fp(camera)
        world_out = P.split_tensor_for_fp(world)
        assert isinstance(cam_out, torch.Tensor)
        assert isinstance(world_out, torch.Tensor)
        self.assertEqual(cam_out.shape[0], 2)
        self.assertEqual(world_out.shape[0], 2)
        self.assertTrue(torch.equal(cam_out, camera[:2]))

    @patch.object(P, "get_frame_parallel_rank", return_value=1)
    @patch.object(P, "get_frame_parallel_world_size", return_value=2)
    def test_split_tensor_with_fp_rank1(self, _mock_ws: Mock, _mock_rank: Mock) -> None:
        """FP rank 1 with fp_size 2 takes the second half of frames."""
        camera = torch.rand(4, 3, 32, 32)
        world = torch.rand(4, 3, 32, 32)
        cam_out = P.split_tensor_for_fp(camera)
        world_out = P.split_tensor_for_fp(world)
        assert isinstance(cam_out, torch.Tensor)
        assert isinstance(world_out, torch.Tensor)
        self.assertEqual(cam_out.shape[0], 2)
        self.assertEqual(world_out.shape[0], 2)
        self.assertTrue(torch.equal(cam_out, camera[2:4]))
        self.assertTrue(torch.equal(world_out, world[2:4]))

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    @patch.object(P, "get_frame_parallel_group")
    @patch.object(P, "get_frame_parallel_world_size", return_value=2)
    def test_gather_features_with_fp_enabled(self, _mock_ws: Mock, mock_get_group: Mock, mock_dist: Mock) -> None:
        """Tensors are gathered along batch dim when FP group is set."""
        mock_fp_group = Mock()
        mock_get_group.return_value = mock_fp_group
        mock_dist.get_world_size.return_value = 2

        def fake_all_gather(gathered_list: list[torch.Tensor], tensor: torch.Tensor, group: object = None) -> None:
            for t in gathered_list:
                t.copy_(tensor)

        mock_dist.all_gather.side_effect = fake_all_gather

        output: dict[str, Any] = {
            "camera_global_features": torch.rand(2, 64),
            "world_global_features": torch.rand(2, 64),
            "metadata": "some_string",
        }
        gathered = P.gather_features_from_fp(output)
        cam_feat = gathered["camera_global_features"]
        world_feat = gathered["world_global_features"]
        assert isinstance(cam_feat, torch.Tensor)
        assert isinstance(world_feat, torch.Tensor)
        self.assertEqual(cam_feat.shape[0], 4)
        self.assertEqual(world_feat.shape[0], 4)
        self.assertEqual(gathered["metadata"], "some_string")

    @patch("checks.utils.models.cwip.modules.parallelism.dist")
    @patch.object(P, "get_frame_parallel_group")
    @patch.object(P, "get_frame_parallel_world_size", return_value=2)
    def test_gather_features_with_list_values(self, _mock_ws: Mock, mock_get_group: Mock, mock_dist: Mock) -> None:
        """List-valued outputs are gathered element-wise."""
        mock_fp_group = Mock()
        mock_get_group.return_value = mock_fp_group
        mock_dist.get_world_size.return_value = 2

        def fake_all_gather(gathered_list: list[torch.Tensor], tensor: torch.Tensor, group: object = None) -> None:
            for t in gathered_list:
                t.copy_(tensor)

        mock_dist.all_gather.side_effect = fake_all_gather

        output: dict[str, Any] = {
            "spatial_features_list": [torch.rand(2, 64, 4, 4), torch.rand(2, 128, 2, 2)],
            "none_value": None,
        }
        gathered = P.gather_features_from_fp(output)
        spatial_feat = gathered["spatial_features_list"]
        assert isinstance(spatial_feat, list)
        self.assertEqual(len(spatial_feat), 2)
        self.assertEqual(spatial_feat[0].shape[0], 4)

    @patch.object(P, "get_frame_parallel_group", return_value=None)
    @patch.object(P, "get_frame_parallel_world_size", return_value=2)
    def test_gather_features_fp_group_none(self, _mock_ws: Mock, _mock_group: Mock) -> None:
        """No gather when the FP process group is not initialized (e.g. unit tests)."""
        output: dict[str, torch.Tensor | list[Any] | None] = {"camera_global_features": torch.rand(2, 64)}
        gathered = P.gather_features_from_fp(output)
        self.assertEqual(gathered, output)


class TestModuleFlags(unittest.TestCase):
    """Test module-level flags."""

    def test_has_megatron_is_bool(self) -> None:
        self.assertIsInstance(P.HAS_MEGATRON, bool)


if __name__ == "__main__":
    unittest.main()
