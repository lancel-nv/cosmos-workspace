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

"""Tests for deployment.modules.attention (TP=1, no distributed)."""

import unittest

import torch

from checks.utils.models.cwip.modules import attention as attention_module
from checks.utils.models.cwip.modules import parallelism as parallelism_module


class _TP1Mixin:
    def setUp(self) -> None:
        parallelism_module._MODEL_PARALLEL_STATE.update(parallelism_module._DEFAULT_STATE)

    def tearDown(self) -> None:
        parallelism_module._MODEL_PARALLEL_STATE.update(parallelism_module._DEFAULT_STATE)


class TestCrossAttention(_TP1Mixin, unittest.TestCase):
    """CrossAttention with use_tensor_parallel=False."""

    def test_forward_shape(self) -> None:
        attn = attention_module.CrossAttention(embed_dim=64, num_heads=4, use_tensor_parallel=False, dropout=0.0)
        q = torch.randn(2, 8, 64)
        ctx = torch.randn(2, 16, 64)
        out = attn(q, ctx)
        self.assertEqual(out.shape, (2, 8, 64))

    def test_different_q_kv_dims(self) -> None:
        attn = attention_module.CrossAttention(
            embed_dim=64,
            num_heads=4,
            q_dim=32,
            kv_dim=48,
            use_tensor_parallel=False,
        )
        q = torch.randn(2, 8, 32)
        ctx = torch.randn(2, 12, 48)
        out = attn(q, ctx)
        self.assertEqual(out.shape, (2, 8, 64))

    def test_set_context_parallel_group(self) -> None:
        m = attention_module.CrossAttention(embed_dim=32, num_heads=4, use_tensor_parallel=False)
        self.assertIsNone(m._cp_group)


class TestSelfAttention(_TP1Mixin, unittest.TestCase):
    """SelfAttention split vs fused QKV."""

    def test_split_forward(self) -> None:
        m = attention_module.SelfAttention(embed_dim=32, num_heads=4, use_tensor_parallel=False)
        x = torch.randn(2, 10, 32)
        out = m(x)
        self.assertEqual(out.shape, (2, 10, 32))
        self.assertIsNotNone(m.q_proj)

    def test_fused_forward(self) -> None:
        m = attention_module.SelfAttention(embed_dim=32, num_heads=4, use_tensor_parallel=False, use_fused_qkv=True)
        x = torch.randn(2, 10, 32)
        out = m(x)
        self.assertEqual(out.shape, (2, 10, 32))
        self.assertIsNotNone(m.in_proj_weight)

    def test_differentiable(self) -> None:
        m = attention_module.SelfAttention(embed_dim=32, num_heads=4, use_tensor_parallel=False, dropout=0.0)
        m.eval()
        x = torch.randn(1, 4, 32, requires_grad=True)
        m(x).sum().backward()
        self.assertIsNotNone(x.grad)


class TestResidualAttentionBlock(_TP1Mixin, unittest.TestCase):
    def test_cross_attn_forward(self) -> None:
        block = attention_module.ResidualAttentionBlock(
            d_model=32,
            n_head=4,
            q_dim=32,
            kv_dim=32,
            use_cross_attn=True,
            use_tensor_parallel=False,
        )
        x = torch.randn(1, 8, 32)
        ctx = torch.randn(1, 8, 32)
        out = block(x, context=ctx)
        self.assertEqual(out.shape, (1, 8, 32))

    def test_self_attn_forward(self) -> None:
        block = attention_module.ResidualAttentionBlock(
            d_model=32, n_head=4, use_cross_attn=False, use_tensor_parallel=False
        )
        x = torch.randn(1, 8, 32)
        out = block(x, attn_mask=None)
        self.assertEqual(out.shape, (1, 8, 32))

    def test_fused_self_attn(self) -> None:
        block = attention_module.ResidualAttentionBlock(
            d_model=32,
            n_head=4,
            use_cross_attn=False,
            use_tensor_parallel=False,
            use_fused_qkv=True,
        )
        x = torch.randn(1, 8, 32)
        self.assertEqual(block(x).shape, (1, 8, 32))

    def test_fused_with_cross_raises(self) -> None:
        with self.assertRaises(ValueError):
            attention_module.ResidualAttentionBlock(
                d_model=32,
                n_head=4,
                use_cross_attn=True,
                use_tensor_parallel=False,
                use_fused_qkv=True,
            )


class TestQueryUpsampleMethod(unittest.TestCase):
    def test_enum_values(self) -> None:
        self.assertEqual(attention_module.QueryUpsampleMethod.SUBPIXEL_CONV, "subpixel_conv")
        self.assertEqual(attention_module.QueryUpsampleMethod.BILINEAR, "bilinear")

    def test_from_string(self) -> None:
        self.assertEqual(
            attention_module.QueryUpsampleMethod("bilinear"),
            attention_module.QueryUpsampleMethod.BILINEAR,
        )


class TestDeformableAttention2D(_TP1Mixin, unittest.TestCase):
    def test_forward_same_resolution(self) -> None:
        m = attention_module.DeformableAttention2D(embed_dim=32, num_heads=4)
        B, H, W, C = 1, 4, 4, 32
        q = torch.randn(B, H * W, C)
        kv = torch.randn(B, H * W, C)
        out = m(q, kv, input_spatial_shape=(H, W))
        self.assertEqual(out.shape, (B, H * W, C))

    def test_forward_upscale_bilinear(self) -> None:
        m = attention_module.DeformableAttention2D(
            embed_dim=32,
            num_heads=4,
            query_upsample_method=attention_module.QueryUpsampleMethod.BILINEAR,
        )
        B, Hi, Wi, C = 1, 4, 4, 32
        Ho, Wo = 8, 8
        q = torch.randn(B, Hi * Wi, C)
        kv = torch.randn(B, Hi * Wi, C)
        out = m(
            q,
            kv,
            input_spatial_shape=(Hi, Wi),
            output_spatial_shape=(Ho, Wo),
        )
        self.assertEqual(out.shape, (B, Ho * Wo, C))


class TestDeformableAttentionBlock(_TP1Mixin, unittest.TestCase):
    def test_forward_same_resolution(self) -> None:
        block = attention_module.DeformableAttentionBlock(embed_dim=32, num_heads=4)
        B, H, W, C = 1, 4, 4, 32
        x = torch.randn(B, H * W, C)
        out = block(x, input_spatial_shape=(H, W))
        self.assertEqual(out.shape, (B, H * W, C))


if __name__ == "__main__":
    unittest.main()
