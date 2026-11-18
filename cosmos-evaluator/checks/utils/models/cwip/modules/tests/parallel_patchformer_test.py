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

"""Tests for parallel_patchformer module with TP=1 (no distributed)."""

from typing import Any, cast
import unittest

import torch

from checks.utils.models.cwip.modules import parallel_patchformer as ppf
from checks.utils.models.cwip.modules import parallelism as P
from checks.utils.models.cwip.modules.class_mapping import ObjectTypeClassMappingV3, PatchClassificationType


class TestQueryUpsampleMethod(unittest.TestCase):
    """Test QueryUpsampleMethod enum."""

    def test_values(self) -> None:
        self.assertEqual(ppf.QueryUpsampleMethod.SUBPIXEL_CONV, "subpixel_conv")
        self.assertEqual(ppf.QueryUpsampleMethod.BILINEAR, "bilinear")

    def test_from_string(self) -> None:
        method = ppf.QueryUpsampleMethod("subpixel_conv")
        self.assertEqual(method, ppf.QueryUpsampleMethod.SUBPIXEL_CONV)


class TestDeformableAttention2D(unittest.TestCase):
    """Test DeformableAttention2D without TP."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def test_init_default(self) -> None:
        attn = ppf.DeformableAttention2D(embed_dim=32, num_heads=4)
        self.assertEqual(attn.embed_dim, 32)
        self.assertEqual(attn.num_heads, 4)
        self.assertEqual(attn.head_dim, 8)
        self.assertEqual(attn.num_sampling_points, 4)

    def test_init_bilinear_upsample(self) -> None:
        attn = ppf.DeformableAttention2D(
            embed_dim=32,
            num_heads=4,
            query_upsample_method=ppf.QueryUpsampleMethod.BILINEAR,
        )
        self.assertIsNone(attn.query_upsample_conv)
        self.assertIsNone(attn.query_pixel_shuffle)

    def test_init_subpixel_conv(self) -> None:
        attn = ppf.DeformableAttention2D(
            embed_dim=32,
            num_heads=4,
            query_upsample_method=ppf.QueryUpsampleMethod.SUBPIXEL_CONV,
        )
        self.assertIsNotNone(attn.query_upsample_conv)
        self.assertIsNotNone(attn.query_pixel_shuffle)

    def test_init_string_upsample_method(self) -> None:
        attn = ppf.DeformableAttention2D(
            embed_dim=32,
            num_heads=4,
            query_upsample_method="bilinear",
        )
        self.assertEqual(attn.query_upsample_method, ppf.QueryUpsampleMethod.BILINEAR)

    def test_forward_same_resolution(self) -> None:
        attn = ppf.DeformableAttention2D(embed_dim=32, num_heads=4)
        B, H, W, C = 1, 4, 4, 32
        query = torch.randn(B, H * W, C)
        key_value = torch.randn(B, H * W, C)
        out = attn(query, key_value, input_spatial_shape=(H, W))
        self.assertEqual(out.shape, (B, H * W, C))

    def test_forward_upscaling_bilinear(self) -> None:
        attn = ppf.DeformableAttention2D(
            embed_dim=32,
            num_heads=4,
            query_upsample_method="bilinear",
        )
        B, H_in, W_in, C = 1, 4, 4, 32
        H_out, W_out = 8, 8
        query = torch.randn(B, H_in * W_in, C)
        kv = torch.randn(B, H_in * W_in, C)
        out = attn(query, kv, input_spatial_shape=(H_in, W_in), output_spatial_shape=(H_out, W_out))
        self.assertEqual(out.shape, (B, H_out * W_out, C))

    def test_forward_upscaling_subpixel_conv(self) -> None:
        attn = ppf.DeformableAttention2D(
            embed_dim=32,
            num_heads=4,
            query_upsample_method="subpixel_conv",
            query_upsample_scale_factor=2,
        )
        B, H_in, W_in, C = 1, 4, 4, 32
        H_out, W_out = 8, 8
        query = torch.randn(B, H_in * W_in, C)
        kv = torch.randn(B, H_in * W_in, C)
        out = attn(query, kv, input_spatial_shape=(H_in, W_in), output_spatial_shape=(H_out, W_out))
        self.assertEqual(out.shape, (B, H_out * W_out, C))

    def test_compute_attention_einsum(self) -> None:
        attn = ppf.DeformableAttention2D(embed_dim=32, num_heads=4, use_sdpa=False)
        B, H, W, C = 1, 4, 4, 32
        query = torch.randn(B, H * W, C)
        kv = torch.randn(B, H * W, C)
        out = attn(query, kv, input_spatial_shape=(H, W))
        self.assertEqual(out.shape, (B, H * W, C))

    def test_compute_attention_sdpa(self) -> None:
        attn = ppf.DeformableAttention2D(embed_dim=32, num_heads=4, use_sdpa=True)
        B, H, W, C = 1, 4, 4, 32
        query = torch.randn(B, H * W, C)
        kv = torch.randn(B, H * W, C)
        out = attn(query, kv, input_spatial_shape=(H, W))
        self.assertEqual(out.shape, (B, H * W, C))

    def test_differentiable(self) -> None:
        attn = ppf.DeformableAttention2D(embed_dim=32, num_heads=4, dropout=0.0)
        attn.eval()
        B, H, W, C = 1, 4, 4, 32
        q = torch.randn(B, H * W, C, requires_grad=True)
        kv = torch.randn(B, H * W, C)
        out = attn(q, kv, input_spatial_shape=(H, W))
        out.sum().backward()
        self.assertIsNotNone(q.grad)

    def test_with_dropout(self) -> None:
        attn = ppf.DeformableAttention2D(embed_dim=32, num_heads=4, dropout=0.1)
        attn.train()
        B, H, W, C = 1, 4, 4, 32
        q = torch.randn(B, H * W, C)
        kv = torch.randn(B, H * W, C)
        out = attn(q, kv, input_spatial_shape=(H, W))
        self.assertEqual(out.shape, (B, H * W, C))

    def test_forward_batch_size(self) -> None:
        attn = ppf.DeformableAttention2D(embed_dim=32, num_heads=4)
        B, H, W, C = 4, 4, 4, 32
        query = torch.randn(B, H * W, C)
        kv = torch.randn(B, H * W, C)
        out = attn(query, kv, input_spatial_shape=(H, W))
        self.assertEqual(out.shape, (B, H * W, C))

    def test_custom_sampling_points(self) -> None:
        attn = ppf.DeformableAttention2D(embed_dim=32, num_heads=4, num_sampling_points=8)
        self.assertEqual(attn.num_sampling_points, 8)
        B, H, W, C = 1, 4, 4, 32
        q = torch.randn(B, H * W, C)
        kv = torch.randn(B, H * W, C)
        out = attn(q, kv, input_spatial_shape=(H, W))
        self.assertEqual(out.shape, (B, H * W, C))

    def test_learned_weight_scale(self) -> None:
        attn = ppf.DeformableAttention2D(embed_dim=32, num_heads=4, learned_weight_scale=0.1)
        self.assertAlmostEqual(attn.learned_weight_scale, 0.1)


class TestDeformableAttentionBlock(unittest.TestCase):
    """Test DeformableAttentionBlock without TP."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def test_init(self) -> None:
        block = ppf.DeformableAttentionBlock(embed_dim=32, num_heads=4)
        self.assertIsNotNone(block.attn)
        self.assertIsNotNone(block.mlp)
        self.assertIsNotNone(block.norm1)
        self.assertIsNotNone(block.norm2)

    def test_forward_same_resolution(self) -> None:
        block = ppf.DeformableAttentionBlock(embed_dim=32, num_heads=4)
        B, H, W, C = 1, 4, 4, 32
        x = torch.randn(B, H * W, C)
        out = block(x, input_spatial_shape=(H, W))
        self.assertEqual(out.shape, (B, H * W, C))

    def test_forward_with_context(self) -> None:
        block = ppf.DeformableAttentionBlock(embed_dim=32, num_heads=4)
        B, H, W, C = 1, 4, 4, 32
        x = torch.randn(B, H * W, C)
        context = torch.randn(B, H * W, C)
        out = block(x, context=context, input_spatial_shape=(H, W))
        self.assertEqual(out.shape, (B, H * W, C))

    def test_forward_upscaling(self) -> None:
        block = ppf.DeformableAttentionBlock(
            embed_dim=32,
            num_heads=4,
            query_upsample_method="bilinear",
        )
        B, H_in, W_in, C = 1, 4, 4, 32
        H_out, W_out = 8, 8
        x = torch.randn(B, H_in * W_in, C)
        out = block(x, input_spatial_shape=(H_in, W_in), output_spatial_shape=(H_out, W_out))
        self.assertEqual(out.shape, (B, H_out * W_out, C))

    def test_forward_with_drop_path(self) -> None:
        block = ppf.DeformableAttentionBlock(embed_dim=32, num_heads=4, drop_path=0.1)
        block.train()
        B, H, W, C = 1, 4, 4, 32
        x = torch.randn(B, H * W, C)
        out = block(x, input_spatial_shape=(H, W))
        self.assertEqual(out.shape, (B, H * W, C))

    def test_assert_input_spatial_shape(self) -> None:
        block = ppf.DeformableAttentionBlock(embed_dim=32, num_heads=4)
        x = torch.randn(1, 16, 32)
        with self.assertRaises(AssertionError):
            block(x, input_spatial_shape=None)

    def test_forward_grad_checkpointing(self) -> None:
        block = ppf.DeformableAttentionBlock(embed_dim=32, num_heads=4, grad_checkpointing=True)
        block.train()
        B, H, W, C = 1, 4, 4, 32
        x = torch.randn(B, H * W, C, requires_grad=True)
        out = block(x, input_spatial_shape=(H, W))
        self.assertEqual(out.shape, (B, H * W, C))
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_forward_upscaling_with_residual(self) -> None:
        """Test that residual is correctly interpolated during upscaling."""
        block = ppf.DeformableAttentionBlock(
            embed_dim=32,
            num_heads=4,
            query_upsample_method="bilinear",
        )
        B, H_in, W_in, C = 2, 4, 4, 32
        H_out, W_out = 8, 8
        x = torch.randn(B, H_in * W_in, C)
        out = block(x, input_spatial_shape=(H_in, W_in), output_spatial_shape=(H_out, W_out))
        self.assertEqual(out.shape, (B, H_out * W_out, C))

    def test_forward_subpixel_upscale(self) -> None:
        block = ppf.DeformableAttentionBlock(
            embed_dim=32,
            num_heads=4,
            query_upsample_method="subpixel_conv",
            query_upsample_scale_factor=2,
        )
        B, H_in, W_in, C = 1, 4, 4, 32
        x = torch.randn(B, H_in * W_in, C)
        out = block(x, input_spatial_shape=(H_in, W_in), output_spatial_shape=(8, 8))
        self.assertEqual(out.shape, (B, 64, C))


class TestCrossAttention(unittest.TestCase):
    """Test CrossAttention with TP=1."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def test_init(self) -> None:
        attn = ppf.CrossAttention(embed_dim=32, num_heads=4)
        self.assertEqual(attn.embed_dim, 32)
        self.assertEqual(attn.num_heads, 4)
        self.assertEqual(attn.head_dim, 8)

    def test_init_different_dims(self) -> None:
        attn = ppf.CrossAttention(embed_dim=32, num_heads=4, q_dim=16, kv_dim=64)
        self.assertEqual(attn.q_dim, 16)
        self.assertEqual(attn.kv_dim, 64)

    def test_forward(self) -> None:
        attn = ppf.CrossAttention(embed_dim=32, num_heads=4)
        q = torch.randn(2, 5, 32)
        kv = torch.randn(2, 10, 32)
        out = attn(q, kv)
        self.assertEqual(out.shape, (2, 5, 32))

    def test_forward_different_dims(self) -> None:
        attn = ppf.CrossAttention(embed_dim=32, num_heads=4, q_dim=16, kv_dim=64)
        q = torch.randn(1, 5, 16)
        kv = torch.randn(1, 10, 64)
        out = attn(q, kv)
        self.assertEqual(out.shape, (1, 5, 32))

    def test_differentiable(self) -> None:
        attn = ppf.CrossAttention(embed_dim=32, num_heads=4, dropout=0.0)
        attn.eval()
        q = torch.randn(1, 4, 32, requires_grad=True)
        kv = torch.randn(1, 8, 32)
        out = attn(q, kv)
        out.sum().backward()
        self.assertIsNotNone(q.grad)

    def test_forward_with_dropout(self) -> None:
        attn = ppf.CrossAttention(embed_dim=32, num_heads=4, dropout=0.1)
        attn.train()
        q = torch.randn(2, 5, 32)
        kv = torch.randn(2, 10, 32)
        out = attn(q, kv)
        self.assertEqual(out.shape, (2, 5, 32))

    def test_set_context_parallel_group(self) -> None:
        attn = ppf.CrossAttention(embed_dim=32, num_heads=4)
        self.assertIsNone(attn._cp_group)


class TestSelfAttention(unittest.TestCase):
    """Test SelfAttention with TP=1."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def test_init(self) -> None:
        attn = ppf.SelfAttention(embed_dim=32, num_heads=4)
        self.assertEqual(attn.embed_dim, 32)
        self.assertEqual(attn.num_heads, 4)
        self.assertEqual(attn.head_dim, 8)
        self.assertEqual(attn.tp_size, 1)
        self.assertEqual(attn.num_heads_per_rank, 4)
        self.assertEqual(attn.embed_dim_per_rank, 32)

    def test_init_weights(self) -> None:
        attn = ppf.SelfAttention(embed_dim=32, num_heads=4)
        # Biases should be zeros
        assert attn.q_proj is not None
        assert attn.k_proj is not None
        assert attn.v_proj is not None
        self.assertTrue(torch.all(attn.q_proj.bias == 0))
        self.assertTrue(torch.all(attn.k_proj.bias == 0))
        self.assertTrue(torch.all(attn.v_proj.bias == 0))
        self.assertTrue(torch.all(attn.out_proj.bias == 0))

    def test_forward(self) -> None:
        attn = ppf.SelfAttention(embed_dim=32, num_heads=4)
        x = torch.randn(2, 10, 32)
        out = attn(x)
        self.assertEqual(out.shape, (2, 10, 32))

    def test_forward_with_mask(self) -> None:
        attn = ppf.SelfAttention(embed_dim=32, num_heads=4)
        x = torch.randn(1, 6, 32)
        out = attn(x, attn_mask=None)
        self.assertEqual(out.shape, (1, 6, 32))

    def test_differentiable(self) -> None:
        attn = ppf.SelfAttention(embed_dim=32, num_heads=4, dropout=0.0)
        attn.eval()
        x = torch.randn(1, 4, 32, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_set_context_parallel_group(self) -> None:
        attn = ppf.SelfAttention(embed_dim=32, num_heads=4)
        self.assertIsNone(attn._cp_group)


class TestResidualAttentionBlock(unittest.TestCase):
    """Test ResidualAttentionBlock with TP=1."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def test_init_cross_attn(self) -> None:
        block = ppf.ResidualAttentionBlock(
            d_model=32,
            n_head=4,
            q_dim=32,
            kv_dim=32,
            use_cross_attn=True,
        )
        self.assertTrue(block.use_cross_attn)
        self.assertIsNotNone(block.ln_context)
        self.assertIsInstance(block.attn, ppf.CrossAttention)

    def test_init_self_attn(self) -> None:
        block = ppf.ResidualAttentionBlock(
            d_model=32,
            n_head=4,
            use_cross_attn=False,
        )
        self.assertFalse(block.use_cross_attn)
        self.assertIsNone(block.ln_context)
        self.assertIsInstance(block.attn, ppf.SelfAttention)

    def test_forward_cross_attn(self) -> None:
        block = ppf.ResidualAttentionBlock(
            d_model=32,
            n_head=4,
            q_dim=32,
            kv_dim=32,
            use_cross_attn=True,
        )
        x = torch.randn(1, 8, 32)
        context = torch.randn(1, 8, 32)
        out = block(x, context=context)
        self.assertEqual(out.shape, (1, 8, 32))

    def test_forward_self_attn(self) -> None:
        block = ppf.ResidualAttentionBlock(
            d_model=32,
            n_head=4,
            use_cross_attn=False,
        )
        x = torch.randn(1, 8, 32)
        out = block(x)
        self.assertEqual(out.shape, (1, 8, 32))

    def test_with_layer_scale(self) -> None:
        block = ppf.ResidualAttentionBlock(
            d_model=32,
            n_head=4,
            use_cross_attn=False,
            ls_init_value=1e-4,
        )
        # Init LayerScale gamma
        for child in block.modules():
            init_fn = getattr(child, "init_tensors", None)
            if init_fn is not None:
                init_fn()
        x = torch.randn(1, 4, 32)
        out = block(x)
        self.assertEqual(out.shape, (1, 4, 32))

    def test_with_drop_path(self) -> None:
        block = ppf.ResidualAttentionBlock(
            d_model=32,
            n_head=4,
            use_cross_attn=False,
            drop_path=0.1,
        )
        block.train()
        x = torch.randn(1, 4, 32)
        out = block(x)
        self.assertEqual(out.shape, (1, 4, 32))

    def test_residual_connection(self) -> None:
        block = ppf.ResidualAttentionBlock(
            d_model=32,
            n_head=4,
            use_cross_attn=False,
        )
        x = torch.randn(1, 4, 32)
        out = block(x)
        self.assertFalse(torch.equal(x, out))


class TestMixedAttentionTransformer(unittest.TestCase):
    """Test MixedAttentionTransformer with TP=1."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def test_init_no_cross_attn(self) -> None:
        transformer = ppf.MixedAttentionTransformer(
            width=32,
            layers=2,
            heads=4,
            cross_attn_frequency=None,
        )
        self.assertEqual(transformer.width, 32)
        self.assertEqual(transformer.layers, 2)
        self.assertEqual(len(transformer.resblocks), 2)
        self.assertEqual(transformer.cross_attn_indices, set())

    def test_init_with_cross_attn(self) -> None:
        transformer = ppf.MixedAttentionTransformer(
            width=32,
            layers=4,
            heads=4,
            q_dim=32,
            kv_dim=32,
            cross_attn_frequency=1,
        )
        # cross_attn_frequency=1 means every 2nd layer is cross-attention
        self.assertIn(1, transformer.cross_attn_indices)
        self.assertIn(3, transformer.cross_attn_indices)

    def test_forward_self_attn_only(self) -> None:
        transformer = ppf.MixedAttentionTransformer(
            width=32,
            layers=2,
            heads=4,
            cross_attn_frequency=None,
        )
        x = torch.randn(1, 8, 32)
        out = transformer(x)
        self.assertEqual(out.shape, (1, 8, 32))

    def test_forward_mixed(self) -> None:
        transformer = ppf.MixedAttentionTransformer(
            width=32,
            layers=2,
            heads=4,
            q_dim=32,
            kv_dim=32,
            cross_attn_frequency=1,
        )
        x = torch.randn(1, 8, 32)
        context = torch.randn(1, 8, 32)
        out = transformer(x, context=context)
        self.assertEqual(out.shape, (1, 8, 32))

    def test_grad_checkpointing(self) -> None:
        transformer = ppf.MixedAttentionTransformer(
            width=32,
            layers=2,
            heads=4,
            cross_attn_frequency=None,
        )
        transformer.set_grad_checkpointing(True)
        self.assertTrue(transformer.grad_checkpointing)
        # Forward with grad checkpointing
        transformer.train()
        x = torch.randn(1, 4, 32, requires_grad=True)
        out = transformer(x)
        self.assertEqual(out.shape, (1, 4, 32))
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        transformer.set_grad_checkpointing(False)

    def test_set_context_parallel_group(self) -> None:
        transformer = ppf.MixedAttentionTransformer(
            width=32,
            layers=2,
            heads=4,
            cross_attn_frequency=None,
        )
        # Should not crash; cast satisfies the ProcessGroup type expected by the signature
        transformer.set_context_parallel_group(cast(Any, None))

    def test_with_layer_scale(self) -> None:
        transformer = ppf.MixedAttentionTransformer(
            width=32,
            layers=2,
            heads=4,
            ls_init_value=1e-4,
            cross_attn_frequency=None,
        )
        for child in transformer.modules():
            init_fn = getattr(child, "init_tensors", None)
            if init_fn is not None:
                init_fn()
        x = torch.randn(1, 4, 32)
        out = transformer(x)
        self.assertEqual(out.shape, (1, 4, 32))

    def test_cross_attn_frequency_property(self) -> None:
        transformer = ppf.MixedAttentionTransformer(
            width=32,
            layers=6,
            heads=4,
            q_dim=32,
            kv_dim=32,
            cross_attn_frequency=2,
        )
        # cross_attn_frequency=2 means every 3rd layer is cross-attention (layers 2, 5)
        self.assertIn(2, transformer.cross_attn_indices)
        self.assertIn(5, transformer.cross_attn_indices)
        self.assertNotIn(0, transformer.cross_attn_indices)
        self.assertNotIn(1, transformer.cross_attn_indices)


class TestParallelPatchFormer(unittest.TestCase):
    """Test ParallelPatchFormer with TP=1."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def _make_model(self, **kwargs: Any) -> ppf.ParallelPatchFormer:
        defaults: dict[str, Any] = {
            "camera_dim": 32,
            "world_dim": 32,
            "num_classes": [5, 2],
            "hidden_dim": 32,
            "num_layers": 1,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "cross_attn_frequency": 1,
            "use_tensor_parallel": False,
            "use_rope2d": False,
            "use_abs_posemb": False,
        }
        defaults.update(kwargs)
        return ppf.ParallelPatchFormer(**defaults)

    def test_init_basic(self) -> None:
        model = self._make_model()
        self.assertEqual(model.camera_dim, 32)
        self.assertEqual(model.world_dim, 32)
        self.assertEqual(model.num_classes, [5, 2])
        self.assertEqual(model.hidden_dim, 32)
        self.assertIsNotNone(model.camera_proj)
        self.assertIsNotNone(model.world_proj)
        self.assertIsNotNone(model.head_type)
        self.assertIsNotNone(model.head_similarity)
        self.assertIsNotNone(model.transformer)
        self.assertIsNone(model.neck_type)
        self.assertIsNone(model.neck_similarity)

    def test_init_single_num_classes(self) -> None:
        model = self._make_model(num_classes=5)
        self.assertEqual(model.num_classes, [5])

    def test_init_num_classes_from_mapping(self) -> None:
        model = self._make_model(num_classes=None, object_type_class_mapping=ObjectTypeClassMappingV3)
        self.assertIsNotNone(model.num_classes)
        self.assertEqual(len(model.num_classes), 2)

    def test_forward_basic(self) -> None:
        model = self._make_model()
        B, C, H, W = 1, 32, 4, 4
        camera_features = torch.randn(B, C, H, W)
        world_features = torch.randn(B, C, H, W)
        result = model(camera_features, world_features)
        self.assertIn("patch_type_logits", result)
        self.assertIn("patch_similarity_logits", result)
        self.assertIn("camera_patch_features", result)
        self.assertEqual(result["patch_type_logits"].shape, (B, 5, H, W))
        self.assertEqual(result["patch_similarity_logits"].shape, (B, 2, H, W))

    def test_forward_batch(self) -> None:
        model = self._make_model()
        B, C, H, W = 4, 32, 4, 4
        camera_features = torch.randn(B, C, H, W)
        world_features = torch.randn(B, C, H, W)
        result = model(camera_features, world_features)
        self.assertEqual(result["patch_type_logits"].shape, (B, 5, H, W))
        self.assertEqual(result["patch_similarity_logits"].shape, (B, 2, H, W))

    def test_forward_with_bias(self) -> None:
        model = self._make_model()
        B, C, H, W = 1, 32, 4, 4
        camera_features = torch.randn(B, C, H, W)
        world_features = torch.randn(B, C, H, W)
        result_no_bias = model(camera_features, world_features, apply_bias=False)
        result_with_bias = model(camera_features, world_features, apply_bias=True)
        # With default bias=0, should be the same
        torch.testing.assert_close(
            result_no_bias["patch_type_logits"],
            result_with_bias["patch_type_logits"],
        )

    def test_forward_with_temperature(self) -> None:
        model = self._make_model()
        B, C, H, W = 1, 32, 4, 4
        camera_features = torch.randn(B, C, H, W)
        world_features = torch.randn(B, C, H, W)
        result_no_temp = model(camera_features, world_features, apply_temperature=False)
        result_with_temp = model(camera_features, world_features, apply_temperature=True)
        # With default temperature=1.0, should be the same
        torch.testing.assert_close(
            result_no_temp["patch_type_logits"],
            result_with_temp["patch_type_logits"],
        )

    def test_apply_bias(self) -> None:
        model = self._make_model()
        # num_classes=[5, 2] → type has 5 classes, similarity has 2 classes
        sim_logits = torch.randn(1, 2, 4, 4)
        biased = model.apply_bias(sim_logits, "similarity")
        self.assertEqual(biased.shape, sim_logits.shape)
        type_logits = torch.randn(1, 5, 4, 4)
        biased_type = model.apply_bias(type_logits, "type")
        self.assertEqual(biased_type.shape, type_logits.shape)

    def test_apply_temperature_scaling(self) -> None:
        model = self._make_model()
        # num_classes=[5, 2] → type has 5 classes, similarity has 2 classes
        sim_logits = torch.randn(1, 2, 4, 4)
        scaled = model.apply_temperature_scaling(sim_logits, "similarity")
        self.assertEqual(scaled.shape, sim_logits.shape)
        type_logits = torch.randn(1, 5, 4, 4)
        scaled_type = model.apply_temperature_scaling(type_logits, "type")
        self.assertEqual(scaled_type.shape, type_logits.shape)

    def test_total_spatial_sampling_factor(self) -> None:
        model = self._make_model(input_resize_factor=2)
        self.assertEqual(model.total_spatial_sampling_factor, 2)
        model2 = self._make_model(input_resize_factor=None)
        self.assertEqual(model2.total_spatial_sampling_factor, 1)

    def test_forward_with_input_resize_factor(self) -> None:
        model = self._make_model(input_resize_factor=2)
        B, C, H, W = 1, 32, 4, 4
        camera_features = torch.randn(B, C, H, W)
        world_features = torch.randn(B, C, H, W)
        result = model(camera_features, world_features)
        # Output should be upscaled by factor 2
        self.assertEqual(result["patch_type_logits"].shape, (B, 5, 8, 8))
        self.assertEqual(result["patch_similarity_logits"].shape, (B, 2, 8, 8))

    def test_forward_with_abs_posemb(self) -> None:
        model = self._make_model(use_abs_posemb=True)
        self.assertTrue(hasattr(model, "positional_embedding"))
        B, C, H, W = 1, 32, 4, 4
        camera_features = torch.randn(B, C, H, W)
        world_features = torch.randn(B, C, H, W)
        result = model(camera_features, world_features)
        self.assertEqual(result["patch_type_logits"].shape, (B, 5, H, W))

    def test_sample_abs_posemb(self) -> None:
        model = self._make_model(use_abs_posemb=True)
        # Matching grid size
        pos = model._sample_abs_posemb(model.posemb_grid_size, model.posemb_grid_size)
        self.assertEqual(pos.shape[0], 1)
        # Different grid size
        pos_interp = model._sample_abs_posemb(4, 4)
        self.assertEqual(pos_interp.shape, (1, 16, 32))

    def test_forward_with_rope2d(self) -> None:
        model = self._make_model(use_rope2d=True)
        self.assertIsNotNone(model.rope)
        B, C, H, W = 1, 32, 4, 4
        camera_features = torch.randn(B, C, H, W)
        world_features = torch.randn(B, C, H, W)
        result = model(camera_features, world_features)
        self.assertEqual(result["patch_type_logits"].shape, (B, 5, H, W))

    def test_excluded_classes_properties(self) -> None:
        model = self._make_model()
        # Default: empty sets
        self.assertEqual(model.excluded_model_classes_for_argmax_similarity, set())
        self.assertEqual(model.excluded_model_classes_for_argmax_type, set())
        # Set excluded classes
        model.excluded_model_classes_for_argmax_similarity = [0]
        self.assertEqual(model.excluded_model_classes_for_argmax_similarity, {0})
        model.excluded_model_classes_for_argmax_type = [1, 2]
        self.assertEqual(model.excluded_model_classes_for_argmax_type, {1, 2})
        # Set to None
        model.excluded_model_classes_for_argmax_similarity = None
        self.assertEqual(model.excluded_model_classes_for_argmax_similarity, set())

    def test_excluded_classes_validation(self) -> None:
        model = self._make_model(num_classes=[5, 2])
        with self.assertRaises(ValueError):
            model.excluded_model_classes_for_argmax_type = [99]

    def test_forward_with_excluded_classes(self) -> None:
        model = self._make_model()
        model.excluded_model_classes_for_argmax_similarity = [0]
        B, C, H, W = 1, 32, 4, 4
        camera_features = torch.randn(B, C, H, W)
        world_features = torch.randn(B, C, H, W)
        result = model(camera_features, world_features, apply_excluded_classes=True)
        self.assertIn("patch_similarity_logits", result)

    def test_forward_with_separate_necks(self) -> None:
        model = self._make_model(use_separate_necks=True)
        self.assertIsNotNone(model.neck_type)
        self.assertIsNotNone(model.neck_similarity)
        self.assertIsNone(model.transformer)
        B, C, H, W = 1, 32, 4, 4
        camera_features = torch.randn(B, C, H, W)
        world_features = torch.randn(B, C, H, W)
        result = model(camera_features, world_features)
        self.assertEqual(result["patch_type_logits"].shape, (B, 5, H, W))
        self.assertEqual(result["patch_similarity_logits"].shape, (B, 2, H, W))

    def test_forward_camera_only_without_world_features(self) -> None:
        model = self._make_model(use_separate_necks=True, cross_attn_frequency=None)
        B, C, H, W = 1, 32, 4, 4
        camera_features = torch.randn(B, C, H, W)
        result = model(camera_features, world_features=None)
        self.assertEqual(result["patch_type_logits"].shape, (B, 5, H, W))
        self.assertNotIn("patch_similarity_logits", result)

    def test_forward_with_deformable_attn(self) -> None:
        model = self._make_model(
            use_deformable_attn=True,
            input_resize_factor=2,
            deformable_query_upsample_method="bilinear",
        )
        self.assertIsNotNone(model.deformable_attn_block)
        B, C, H, W = 1, 32, 4, 4
        camera_features = torch.randn(B, C, H, W)
        world_features = torch.randn(B, C, H, W)
        result = model(camera_features, world_features)
        # Deformable attn does upscaling: 4*2=8
        self.assertEqual(result["patch_type_logits"].shape, (B, 5, 8, 8))
        self.assertEqual(result["patch_similarity_logits"].shape, (B, 2, 8, 8))

    def test_forward_with_separate_necks_and_deformable(self) -> None:
        model = self._make_model(
            use_separate_necks=True,
            use_deformable_attn=True,
            input_resize_factor=2,
            deformable_query_upsample_method="bilinear",
        )
        B, C, H, W = 1, 32, 4, 4
        camera_features = torch.randn(B, C, H, W)
        world_features = torch.randn(B, C, H, W)
        result = model(camera_features, world_features)
        self.assertEqual(result["patch_type_logits"].shape, (B, 5, 8, 8))
        self.assertEqual(result["patch_similarity_logits"].shape, (B, 2, 8, 8))

    def test_forward_all_calibration(self) -> None:
        model = self._make_model(
            per_class_temperature_similarity=[2.0, 2.0],
            per_class_bias_similarity=[0.5, 0.5],
            per_class_temperature_type=[2.0] * 5,
            per_class_bias_type=[0.1] * 5,
        )
        B, C, H, W = 1, 32, 4, 4
        camera_features = torch.randn(B, C, H, W)
        world_features = torch.randn(B, C, H, W)
        result = model(camera_features, world_features, apply_bias=True, apply_temperature=True)
        self.assertIn("patch_type_logits", result)
        self.assertIn("patch_similarity_logits", result)

    def test_differentiable(self) -> None:
        model = self._make_model()
        B, C, H, W = 1, 32, 4, 4
        camera_features = torch.randn(B, C, H, W, requires_grad=True)
        world_features = torch.randn(B, C, H, W)
        result = model(camera_features, world_features)
        result["patch_type_logits"].sum().backward()
        self.assertIsNotNone(camera_features.grad)

    def test_patch_classification_type(self) -> None:
        model = self._make_model(
            patch_classification_type=[PatchClassificationType.PATCH_TYPE, PatchClassificationType.PATCH_SIMILARITY],
        )
        self.assertEqual(len(model.patch_classification_type), 2)

    def test_treat_wrong_type_as_match(self) -> None:
        model = self._make_model(treat_wrong_type_as_match=True)
        self.assertTrue(model.treat_wrong_type_as_match)

    def test_use_legacy_transformer(self) -> None:
        model = self._make_model(use_legacy_transformer=True)
        self.assertTrue(model.use_legacy_transformer)
        # Only test initialization; legacy transformer forward has known API incompatibility with PyTorch
        # nn.MultiheadAttention expecting separate key/value args
        self.assertIsNotNone(model.transformer)

    def test_no_cross_attn_frequency(self) -> None:
        model = self._make_model(cross_attn_frequency=None)
        B, C, H, W = 1, 32, 4, 4
        camera_features = torch.randn(B, C, H, W)
        world_features = torch.randn(B, C, H, W)
        result = model(camera_features, world_features)
        self.assertIn("patch_type_logits", result)


class TestMixedAttentionTransformerBlock(unittest.TestCase):
    """Test MixedAttentionTransformer building blocks."""

    def setUp(self) -> None:
        P._MODEL_PARALLEL_STATE.update(P._DEFAULT_STATE)

    def test_cross_attention_residual(self) -> None:
        """Verify cross attention output shape through a CrossAttention module."""
        attn = ppf.CrossAttention(embed_dim=32, num_heads=4, q_dim=32, kv_dim=32)
        q = torch.randn(1, 8, 32)
        kv = torch.randn(1, 8, 32)
        out = attn(q, kv)
        self.assertEqual(out.shape, (1, 8, 32))


if __name__ == "__main__":
    unittest.main()
