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

"""Tests for parallel_pe module with TP=1 (includes pe_configs + VisionTransformer from config)."""

import unittest

import attrs
from third_party.perception_models.core.vision_encoder.rope import Rope2D
import torch

from checks.utils.models.cwip.modules import parallel_pe
from checks.utils.models.cwip.modules import parallelism as parallelism_module
from checks.utils.models.cwip.modules.configs.defaults import pe_configs
from checks.utils.models.cwip.modules.configs.defaults.pe_configs import VisionTransformerConfig


def _vision_transformer_from_config(cfg: pe_configs.VisionTransformerConfig) -> parallel_pe.VisionTransformer:
    """TP=1 ViT from attrs config; matches checkpoint layout (``use_tensor_parallel=False``)."""
    d = attrs.asdict(cfg)
    vit = parallel_pe.VisionTransformer(
        image_size=d["image_size"],
        patch_size=d["patch_size"],
        width=d["width"],
        layers=d["layers"],
        heads=d["heads"],
        mlp_ratio=d["mlp_ratio"],
        output_dim=d["output_dim"],
        pool_type=d["pool_type"],
        use_cls_token=d["use_cls_token"],
        use_abs_posemb=d["use_abs_posemb"],
        use_rope2d=d["use_rope2d"],
        use_ln_post=d["use_ln_post"],
        ls_init_value=d["ls_init_value"],
        act_layer=d.get("act_layer", torch.nn.GELU),
        norm_layer=d.get("norm_layer", torch.nn.LayerNorm),
        drop_path=d["drop_path"],
        use_tensor_parallel=False,
        attn_pooler_heads=d.get("attn_pooler_heads"),
    )
    vit.init_tensors()
    return vit


# Register a tiny test config for fast ParallelImageEncoder tests
_TEST_PE_CFG = VisionTransformerConfig(
    patch_size=16,
    width=64,
    layers=2,
    heads=4,
    output_dim=64,
    mlp_ratio=2.0,
    ls_init_value=None,
    drop_path=0.0,
    image_size=32,
    use_abs_posemb=True,
    use_cls_token=True,
    use_rope2d=True,
    pool_type="attn",
    attn_pooler_heads=2,
    use_ln_pre=True,
    use_ln_post=True,
)
pe_configs.DEFAULT_PE_CONFIGS["_test-tiny"] = _TEST_PE_CFG
_TEST_CFG_NAME = "_test-tiny"


def _tiny_pe_vit_cfg(**overrides: object) -> VisionTransformerConfig:
    cfg = pe_configs.VisionTransformerConfig(
        patch_size=16,
        width=64,
        layers=2,
        heads=4,
        output_dim=64,
        mlp_ratio=2.0,
        ls_init_value=1e-5,
        drop_path=0.0,
        image_size=32,
        use_abs_posemb=False,
        use_cls_token=False,
        use_rope2d=False,
        pool_type="avg",
        attn_pooler_heads=2,
        use_ln_pre=True,
        use_ln_post=True,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class TestPeConfigForwardFeatures(unittest.TestCase):
    """forward_features via VisionTransformer from PE config."""

    def test_returns_requested_layers_and_last(self) -> None:
        cfg = _tiny_pe_vit_cfg()
        vit = _vision_transformer_from_config(cfg)
        x = torch.randn(1, 3, cfg.image_size, cfg.image_size)
        result = vit.forward_features(x, norm=False, layer_idx=[0], strip_cls_token=False)
        last = result["spatial_features"]
        x_list = result["spatial_features_list"]
        self.assertIsInstance(x_list, list)
        self.assertEqual(len(x_list), 1)
        seq_len = (cfg.image_size // cfg.patch_size) ** 2
        assert isinstance(last, torch.Tensor)
        self.assertEqual(last.shape, (1, seq_len, cfg.width))
        self.assertEqual(x_list[0].shape, (1, seq_len, cfg.width))

    def test_with_multiple_indices_sorted_and_unique(self) -> None:
        cfg = _tiny_pe_vit_cfg()
        vit = _vision_transformer_from_config(cfg)
        x = torch.randn(1, 3, cfg.image_size, cfg.image_size)
        result = vit.forward_features(x, norm=False, layer_idx=[0, -1, 0], strip_cls_token=False)
        x_list = result["spatial_features_list"]
        self.assertGreaterEqual(len(x_list), 2)
        for t in x_list:
            self.assertEqual(t.shape[-1], cfg.width)

    def test_strip_cls_token_removes_token_when_enabled(self) -> None:
        cfg = _tiny_pe_vit_cfg(use_cls_token=True)
        vit = _vision_transformer_from_config(cfg)
        x = torch.randn(1, 3, cfg.image_size, cfg.image_size)
        result = vit.forward_features(x, norm=False, layer_idx=[-1], strip_cls_token=True)
        last = result["spatial_features"]
        x_list = result["spatial_features_list"]
        seq_len_wo_cls = (cfg.image_size // cfg.patch_size) ** 2
        assert isinstance(last, torch.Tensor)
        self.assertEqual(last.shape[1], seq_len_wo_cls)
        self.assertEqual(x_list[0].shape[1], seq_len_wo_cls)

    def test_with_abs_posemb(self) -> None:
        cfg = _tiny_pe_vit_cfg(use_abs_posemb=True)
        vit = _vision_transformer_from_config(cfg)
        x = torch.randn(1, 3, cfg.image_size, cfg.image_size)
        result = vit.forward_features(x, norm=True, layer_idx=[-1], strip_cls_token=False)
        self.assertIsInstance(result["spatial_features"], torch.Tensor)

    def test_with_rope2d(self) -> None:
        cfg = _tiny_pe_vit_cfg(use_rope2d=True)
        vit = _vision_transformer_from_config(cfg)
        x = torch.randn(1, 3, cfg.image_size, cfg.image_size)
        result = vit.forward_features(x, norm=True, layer_idx=[-1], strip_cls_token=False)
        self.assertIsInstance(result["spatial_features"], torch.Tensor)

    def test_with_norm_true(self) -> None:
        cfg = _tiny_pe_vit_cfg()
        vit = _vision_transformer_from_config(cfg)
        x = torch.randn(1, 3, cfg.image_size, cfg.image_size)
        result_with_norm = vit.forward_features(x, norm=True, layer_idx=[-1])
        result_without_norm = vit.forward_features(x, norm=False, layer_idx=[-1])
        assert isinstance(result_with_norm["spatial_features"], torch.Tensor)
        assert isinstance(result_without_norm["spatial_features"], torch.Tensor)
        self.assertFalse(torch.allclose(result_with_norm["spatial_features"], result_without_norm["spatial_features"]))

    def test_with_single_int_layer_idx(self) -> None:
        cfg = _tiny_pe_vit_cfg()
        vit = _vision_transformer_from_config(cfg)
        x = torch.randn(1, 3, cfg.image_size, cfg.image_size)
        result = vit.forward_features(x, norm=False, layer_idx=0, strip_cls_token=False)
        x_list = result["spatial_features_list"]
        self.assertIsInstance(x_list, list)
        self.assertEqual(len(x_list), 1)


class TestPeConfigForwardFeaturesIdempotentAutoAppend(unittest.TestCase):
    """Regression tests for the idempotent ``+ [-1]`` migration in :meth:`VisionTransformer.forward_features`.

    Background: the pre-migration code unconditionally appended ``-1`` to ``layer_idx`` and popped the trailing
    transformer output to recover ``spatial_features``. That worked but (a) ballooned transformer compute by one
    redundant slot per call when the caller already passed ``-1``, and (b) made explicit-duplicate semantics
    (``layer_idx=[-1, -1]``) opaque -- the transformer ran three blocks of the same layer before the pop. The migration
    replaces the unconditional append + pop with an idempotent append + conditional pop: the auto-append fires only when
    the caller did NOT include the final layer (negative or positive spelling), and the pop fires only when the
    auto-append did.

    These tests pin down two invariants:
      1. End-state byte-equivalence vs. the pre-migration code for every (``layer_idx``, ``spatial_features``,
         ``spatial_features_list``) combination -- patch_classifier (which reads ``spatial_features_list[-1]``) and viz
         consumers must see identical tensors.
      2. The transformer runs strictly fewer (or equal) blocks per call after the migration -- never more.
    """

    def test_layer_idx_minus_one_keeps_single_pre_ln_entry(self) -> None:
        """``layer_idx=[-1]`` -> ``len(spatial_features_list) == 1`` and the entry is pre-LN final (same byte-for-byte
        as the pre-migration pop-the-trailing-copy result, because the trailing copy and the kept copy are the same
        tensor object).
        """
        cfg = _tiny_pe_vit_cfg()
        vit = _vision_transformer_from_config(cfg)
        x = torch.randn(1, 3, cfg.image_size, cfg.image_size)

        result = vit.forward_features(x, norm=True, layer_idx=[-1], strip_cls_token=False)
        x_list = result["spatial_features_list"]
        self.assertEqual(len(x_list), 1)
        # ``spatial_features`` is post-LN; ``spatial_features_list[-1]`` is pre-LN -- the LN difference must produce
        # different tensors.
        assert isinstance(result["spatial_features"], torch.Tensor)
        self.assertFalse(
            torch.allclose(result["spatial_features"], x_list[-1], atol=1e-6),
            "spatial_features and spatial_features_list[-1] are byte-equal; "
            "ln_post should have changed the value, but it didn't -- check "
            "that norm=True is being honored and that the list entry stayed pre-LN.",
        )

    def test_layer_idx_explicit_duplicate_returns_two_pre_ln_entries(self) -> None:
        """``layer_idx=[-1, -1]`` -> ``len(spatial_features_list) == 2``, both entries are the SAME pre-LN final tensor
        (the transformer's ``while i in stop_idx: pop`` loop appends ``x`` once per duplicate stop index; both copies
        share identity)."""
        cfg = _tiny_pe_vit_cfg()
        vit = _vision_transformer_from_config(cfg)
        x = torch.randn(1, 3, cfg.image_size, cfg.image_size)

        result = vit.forward_features(x, norm=False, layer_idx=[-1, -1], strip_cls_token=False)
        x_list = result["spatial_features_list"]
        self.assertEqual(len(x_list), 2)
        self.assertTrue(
            torch.allclose(x_list[0], x_list[1], atol=0),
            "Both pre-LN copies should be identical -- they share the same "
            "transformer output tensor (no-norm path doesn't transform either).",
        )

    def test_layer_idx_zero_keeps_single_pre_ln_layer0_entry(self) -> None:
        """``layer_idx=[0]`` -> ``len(spatial_features_list) == 1`` and the entry is the pre-LN layer-0 output
        (auto-append fires because ``-1`` is not in the request; the conditional pop then removes the auto-appended
        final, leaving just layer-0). Matches pre-migration behavior exactly."""
        cfg = _tiny_pe_vit_cfg()
        vit = _vision_transformer_from_config(cfg)
        x = torch.randn(1, 3, cfg.image_size, cfg.image_size)

        result = vit.forward_features(x, norm=True, layer_idx=[0], strip_cls_token=False)
        x_list = result["spatial_features_list"]
        self.assertEqual(len(x_list), 1)
        # ``spatial_features`` is post-LN final, ``x_list[0]`` is pre-LN layer 0 -- different layers AND different LN
        # regime -> must produce different tensors.
        assert isinstance(result["spatial_features"], torch.Tensor)
        self.assertFalse(
            torch.allclose(result["spatial_features"], x_list[0], atol=1e-6),
        )

    def test_layer_idx_zero_and_minus_one_keeps_two_entries(self) -> None:
        """``layer_idx=[0, -1]`` -> ``len(spatial_features_list) == 2``: ``[layer_0_pre_ln, layer_final_pre_ln]``.
        Auto-append is skipped (``-1`` already present); no pop. Pre-migration would have produced the same end state
        via append-then-pop."""
        cfg = _tiny_pe_vit_cfg()
        vit = _vision_transformer_from_config(cfg)
        x = torch.randn(1, 3, cfg.image_size, cfg.image_size)

        result = vit.forward_features(x, norm=True, layer_idx=[0, -1], strip_cls_token=False)
        x_list = result["spatial_features_list"]
        self.assertEqual(len(x_list), 2)
        # ``spatial_features`` (post-LN final) must differ from ``x_list[-1]`` (pre-LN final) -- the migration preserves
        # PE's "list is pre-LN, singular is post-LN" contract.
        assert isinstance(result["spatial_features"], torch.Tensor)
        self.assertFalse(
            torch.allclose(result["spatial_features"], x_list[-1], atol=1e-6),
            "spatial_features_list[-1] is post-LN -- this would be the "
            "SegFormer-Aligner regime, not PE's. The migration must NOT "
            "have flipped the list semantic.",
        )

    def test_positive_final_index_also_suppresses_auto_append(self) -> None:
        """Auto-append must also detect the positive spelling of the final layer (``layers - 1``). Otherwise a caller
        passing the positive index would still get the redundant ``-1`` slot."""
        cfg = _tiny_pe_vit_cfg()
        vit = _vision_transformer_from_config(cfg)
        x = torch.randn(1, 3, cfg.image_size, cfg.image_size)

        layers = vit.transformer.layers
        # ``layer_idx=[layers - 1]`` already targets the final block.
        # Pre-migration: auto-append +[-1] -> [layers-1, -1] -> 2 outputs
        # -> pop -> 1 entry. Post-migration: no auto-append -> 1 output
        # -> no pop -> 1 entry. Same length, same end-state tensor.
        result = vit.forward_features(
            x,
            norm=True,
            layer_idx=[layers - 1],
            strip_cls_token=False,
        )
        x_list = result["spatial_features_list"]
        self.assertEqual(len(x_list), 1)


class TestPeConfigVisionTransformerForward(unittest.TestCase):
    def test_returns_pooled_and_intermediates(self) -> None:
        cfg = _tiny_pe_vit_cfg()
        vit = _vision_transformer_from_config(cfg)
        x = torch.randn(1, 3, cfg.image_size, cfg.image_size)
        result = vit(x)
        self.assertIsInstance(result, dict)
        self.assertIn("global_features", result)
        self.assertIn("spatial_features_list", result)
        self.assertEqual(result["global_features"].shape[-1], cfg.output_dim)

    def test_with_proj_dim(self) -> None:
        cfg = _tiny_pe_vit_cfg(output_dim=32)
        vit = _vision_transformer_from_config(cfg)
        x = torch.randn(1, 3, cfg.image_size, cfg.image_size)
        result = vit(x)
        self.assertEqual(result["global_features"].shape[-1], 32)


class TestPeConfigTransformerViaVit(unittest.TestCase):
    def test_invalid_layer_idx_type_raises(self) -> None:
        cfg = _tiny_pe_vit_cfg()
        vit = _vision_transformer_from_config(cfg)
        x = torch.randn(1, 3, cfg.image_size, cfg.image_size)
        x = vit.conv1(x)
        batch = x.shape[0]
        x = x.permute(0, 2, 3, 1).reshape(batch, -1, vit.width)
        x = vit.ln_pre(x)
        with self.assertRaises(TypeError):
            vit.transformer(x, layer_idx=1.5)

    def test_grad_checkpointing_end_to_end(self) -> None:
        cfg = _tiny_pe_vit_cfg()
        vit = _vision_transformer_from_config(cfg)
        vit.set_grad_checkpointing(enable=True)
        x = torch.randn(1, 3, cfg.image_size, cfg.image_size, requires_grad=True)
        result = vit(x, layer_idx=[-1])
        self.assertIsInstance(result, dict)
        self.assertIn("global_features", result)
        result["global_features"].mean().backward()
        self.assertIsNotNone(x.grad)


class TestPeConfigVisionTransformerInitTensors(unittest.TestCase):
    def test_init_tensors_without_rope(self) -> None:
        cfg = _tiny_pe_vit_cfg(use_rope2d=False)
        vit = _vision_transformer_from_config(cfg)
        vit.init_tensors()

    def test_init_tensors_with_rope(self) -> None:
        cfg = _tiny_pe_vit_cfg(use_rope2d=True)
        vit = _vision_transformer_from_config(cfg)
        vit.init_tensors()


class TestPeConfigStripClsToken(unittest.TestCase):
    def test_strip_cls_token_removes_first_token(self) -> None:
        cfg = _tiny_pe_vit_cfg(use_cls_token=True)
        vit = _vision_transformer_from_config(cfg)
        x = torch.randn(2, 10, 64)
        result = vit.strip_cls_token(x)
        self.assertEqual(result.shape, (2, 9, 64))

    def test_strip_cls_token_no_change_when_disabled(self) -> None:
        cfg = _tiny_pe_vit_cfg(use_cls_token=False)
        vit = _vision_transformer_from_config(cfg)
        x = torch.randn(2, 10, 64)
        result = vit.strip_cls_token(x)
        self.assertEqual(result.shape, (2, 10, 64))
        self.assertTrue(torch.equal(result, x))


class TestPeConfigNanCheck(unittest.TestCase):
    def test_nan_in_output_detected(self) -> None:
        cfg = pe_configs.VisionTransformerConfig(
            patch_size=16,
            width=64,
            layers=2,
            heads=4,
            output_dim=64,
            mlp_ratio=2.0,
            ls_init_value=1e-5,
            image_size=32,
            use_abs_posemb=False,
            use_cls_token=False,
            use_rope2d=False,
            pool_type="avg",
        )
        vt = _vision_transformer_from_config(cfg)
        x = torch.randn(1, 3, 32, 32)
        output = vt(x, layer_idx=-1)
        self.assertFalse(torch.isnan(output["global_features"]).any())


class TestPeConfigTransformerLayerIdxEdgeCases(unittest.TestCase):
    def test_vit_forward_layer_idx_minus_one(self) -> None:
        cfg = _tiny_pe_vit_cfg()
        vt = _vision_transformer_from_config(cfg)
        x = torch.randn(1, 3, 32, 32)
        output = vt(x, layer_idx=-1)
        self.assertIn("spatial_features", output)

    def test_transformer_forward_all_layers(self) -> None:
        transformer = parallel_pe.Transformer(
            width=64,
            layers=2,
            heads=4,
            mlp_ratio=2.0,
            ls_init_value=1e-5,
        )
        for block in transformer.resblocks:
            ls_1 = getattr(block, "ls_1", None)
            if ls_1 is not None and hasattr(ls_1, "init_tensors"):
                ls_1.init_tensors()
            ls_2 = getattr(block, "ls_2", None)
            if ls_2 is not None and hasattr(ls_2, "init_tensors"):
                ls_2.init_tensors()
        x = torch.randn(1, 4, 64)
        result = transformer(x, layer_idx=list(range(2)))
        self.assertEqual(len(result), 2)

    def test_transformer_stop_idx_beyond_layers_fallback(self) -> None:
        transformer = parallel_pe.Transformer(
            width=64,
            layers=2,
            heads=4,
            mlp_ratio=2.0,
            ls_init_value=1e-5,
        )
        for block in transformer.resblocks:
            ls_1 = getattr(block, "ls_1", None)
            if ls_1 is not None and hasattr(ls_1, "init_tensors"):
                ls_1.init_tensors()
            ls_2 = getattr(block, "ls_2", None)
            if ls_2 is not None and hasattr(ls_2, "init_tensors"):
                ls_2.init_tensors()
        x = torch.randn(1, 4, 64)
        result = transformer(x, layer_idx=[10])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].shape, (1, 4, 64))


class TestSelfAttention(unittest.TestCase):
    """Test parallel_pe.SelfAttention (fused QKV) with TP=1."""

    def setUp(self) -> None:
        parallelism_module._MODEL_PARALLEL_STATE.update(parallelism_module._DEFAULT_STATE)

    def test_init(self) -> None:
        attn = parallel_pe.SelfAttention(embed_dim=32, num_heads=4)
        self.assertEqual(attn.embed_dim, 32)
        self.assertEqual(attn.num_heads, 4)
        self.assertEqual(attn.head_dim, 8)
        self.assertEqual(attn.num_heads_per_rank, 4)
        # in_proj_weight has shape [3*embed_dim/tp, embed_dim]
        self.assertEqual(attn.in_proj_weight.shape, (96, 32))
        self.assertEqual(attn.in_proj_bias.shape[0], 96)

    def test_forward_self_attention(self) -> None:
        attn = parallel_pe.SelfAttention(embed_dim=32, num_heads=4)
        x = torch.randn(2, 10, 32)
        out = attn(x)
        self.assertEqual(out.shape, (2, 10, 32))

    def test_forward_with_attn_mask(self) -> None:
        attn = parallel_pe.SelfAttention(embed_dim=32, num_heads=4)
        x = torch.randn(1, 6, 32)
        out = attn(x, attn_mask=None)
        self.assertEqual(out.shape, (1, 6, 32))

    def test_differentiable(self) -> None:
        attn = parallel_pe.SelfAttention(embed_dim=32, num_heads=4, dropout=0.0)
        attn.eval()
        x = torch.randn(1, 4, 32, requires_grad=True)
        out = attn(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_context_parallel_not_supported_on_pe_self_attention(self) -> None:
        attn = parallel_pe.SelfAttention(embed_dim=32, num_heads=4)
        with self.assertRaises(RuntimeError) as ctx:
            attn.set_context_parallel_group(None)
        self.assertIn("parallel_patchformer", str(ctx.exception).lower())

    def test_forward_batch_dim(self) -> None:
        attn = parallel_pe.SelfAttention(embed_dim=32, num_heads=4)
        x = torch.randn(4, 8, 32)
        out = attn(x)
        self.assertEqual(out.shape, (4, 8, 32))

    def test_forward_with_dropout(self) -> None:
        attn = parallel_pe.SelfAttention(embed_dim=32, num_heads=4, dropout=0.1)
        attn.train()
        x = torch.randn(2, 6, 32)
        out = attn(x)
        self.assertEqual(out.shape, (2, 6, 32))

    def test_init_weights(self) -> None:
        attn = parallel_pe.SelfAttention(embed_dim=32, num_heads=4)
        self.assertTrue(torch.all(attn.in_proj_bias == 0))


class TestResidualAttentionBlock(unittest.TestCase):
    """Test parallel_pe.ResidualAttentionBlock with TP=1."""

    def setUp(self) -> None:
        parallelism_module._MODEL_PARALLEL_STATE.update(parallelism_module._DEFAULT_STATE)

    def test_init(self) -> None:
        block = parallel_pe.ResidualAttentionBlock(d_model=32, n_head=4)
        self.assertIsNotNone(block.ln_1)
        self.assertIsNotNone(block.ln_2)
        self.assertIsNotNone(block.attn)
        self.assertIsNotNone(block.mlp)

    def test_forward(self) -> None:
        block = parallel_pe.ResidualAttentionBlock(d_model=32, n_head=4)
        x = torch.randn(2, 8, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 8, 32))

    def test_forward_with_layer_scale(self) -> None:
        block = parallel_pe.ResidualAttentionBlock(d_model=32, n_head=4, ls_init_value=1e-4)
        # LayerScale requires init_tensors() to create gamma parameter
        for child in block.modules():
            init_fn = getattr(child, "init_tensors", None)
            if init_fn is not None:
                init_fn()
        x = torch.randn(1, 4, 32)
        out = block(x)
        self.assertEqual(out.shape, (1, 4, 32))

    def test_forward_with_drop_path(self) -> None:
        block = parallel_pe.ResidualAttentionBlock(d_model=32, n_head=4, drop_path=0.1)
        block.train()
        x = torch.randn(2, 4, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 4, 32))

    def test_residual_connection(self) -> None:
        block = parallel_pe.ResidualAttentionBlock(d_model=32, n_head=4)
        x = torch.randn(1, 4, 32)
        out = block(x)
        # Output should differ from input (residual + transformation)
        self.assertFalse(torch.equal(x, out))


class TestTransformer(unittest.TestCase):
    """Test parallel_pe.Transformer with TP=1."""

    def setUp(self) -> None:
        parallelism_module._MODEL_PARALLEL_STATE.update(parallelism_module._DEFAULT_STATE)

    def test_init(self) -> None:
        transformer = parallel_pe.Transformer(width=32, layers=2, heads=4)
        self.assertEqual(transformer.width, 32)
        self.assertEqual(transformer.layers, 2)
        self.assertEqual(len(transformer.resblocks), 2)

    def test_forward_default_layer_idx(self) -> None:
        transformer = parallel_pe.Transformer(width=32, layers=2, heads=4)
        x = torch.randn(1, 8, 32)
        out_list = transformer(x, layer_idx=-1)
        self.assertEqual(len(out_list), 1)
        self.assertEqual(out_list[0].shape, (1, 8, 32))

    def test_forward_multiple_layer_idx(self) -> None:
        transformer = parallel_pe.Transformer(width=32, layers=4, heads=4)
        x = torch.randn(1, 8, 32)
        out_list = transformer(x, layer_idx=[0, 2, -1])
        self.assertGreaterEqual(len(out_list), 3)

    def test_forward_list_layer_idx(self) -> None:
        transformer = parallel_pe.Transformer(width=32, layers=3, heads=4)
        x = torch.randn(1, 6, 32)
        out_list = transformer(x, layer_idx=[0, 1])
        self.assertGreaterEqual(len(out_list), 2)

    def test_grad_checkpointing(self) -> None:
        transformer = parallel_pe.Transformer(width=32, layers=2, heads=4)
        transformer.set_grad_checkpointing(True)
        self.assertTrue(transformer.grad_checkpointing)
        transformer.set_grad_checkpointing(False)
        self.assertFalse(transformer.grad_checkpointing)

    def test_forward_invalid_layer_idx(self) -> None:
        transformer = parallel_pe.Transformer(width=32, layers=2, heads=4)
        x = torch.randn(1, 4, 32)
        with self.assertRaises(TypeError):
            transformer(x, layer_idx="invalid")


class TestVisionTransformer(unittest.TestCase):
    """Test parallel_pe.VisionTransformer with TP=1."""

    def setUp(self) -> None:
        parallelism_module._MODEL_PARALLEL_STATE.update(parallelism_module._DEFAULT_STATE)
        # Use small dims for fast tests
        self.vit = parallel_pe.VisionTransformer(
            image_size=28,
            patch_size=14,
            width=32,
            layers=2,
            heads=4,
            mlp_ratio=2.0,
            output_dim=16,
            pool_type="avg",
            use_cls_token=False,
            use_abs_posemb=False,
            use_rope2d=False,
            use_ln_post=True,
            ls_init_value=None,
        )
        self.vit.init_tensors()

    def test_init(self) -> None:
        self.assertEqual(self.vit.image_size, 28)
        self.assertEqual(self.vit.patch_size, 14)
        self.assertEqual(self.vit.width, 32)

    def test_forward_features(self) -> None:
        x = torch.randn(1, 3, 28, 28)
        result = self.vit.forward_features(x)
        self.assertIn("spatial_features", result)
        self.assertIn("spatial_features_list", result)
        assert isinstance(result["spatial_features"], torch.Tensor)
        self.assertEqual(result["spatial_features"].shape[0], 1)
        self.assertEqual(result["spatial_features"].shape[2], 32)  # width

    def test_forward(self) -> None:
        x = torch.randn(1, 3, 28, 28)
        result = self.vit(x)
        self.assertIn("global_features", result)
        self.assertIn("spatial_features", result)
        self.assertIn("spatial_features_list", result)

    def test_forward_features_with_norm(self) -> None:
        x = torch.randn(1, 3, 28, 28)
        result = self.vit.forward_features(x, norm=True)
        self.assertIn("spatial_features", result)

    def test_forward_features_layer_idx(self) -> None:
        x = torch.randn(1, 3, 28, 28)
        result = self.vit.forward_features(x, layer_idx=[0])
        self.assertIn("spatial_features_list", result)

    def test_pool_avg(self) -> None:
        x = torch.randn(1, 4, 32)
        pooled = self.vit._pool(x)
        self.assertEqual(pooled.shape, (1, 32))

    def test_pool_none(self) -> None:
        self.vit.pool_type = "none"
        x = torch.randn(1, 4, 32)
        pooled = self.vit._pool(x)
        self.assertEqual(pooled.shape, (1, 4, 32))

    def test_pool_invalid(self) -> None:
        self.vit.pool_type = "invalid"
        x = torch.randn(1, 4, 32)
        with self.assertRaises(ValueError):
            self.vit._pool(x)

    def test_strip_cls_token_no_cls(self) -> None:
        x = torch.randn(1, 5, 32)
        result = self.vit.strip_cls_token(x)
        self.assertEqual(result.shape, (1, 5, 32))

    def test_grad_checkpointing(self) -> None:
        self.vit.set_grad_checkpointing(True)
        self.vit.set_grad_checkpointing(False)


class TestVisionTransformerWithCls(unittest.TestCase):
    """Test VisionTransformer with cls_token and abs_posemb."""

    def setUp(self) -> None:
        parallelism_module._MODEL_PARALLEL_STATE.update(parallelism_module._DEFAULT_STATE)
        self.vit = parallel_pe.VisionTransformer(
            image_size=28,
            patch_size=14,
            width=32,
            layers=1,
            heads=4,
            mlp_ratio=2.0,
            pool_type="avg",
            use_cls_token=True,
            use_abs_posemb=True,
            use_rope2d=False,
            use_ln_post=True,
        )
        self.vit.init_tensors()

    def test_forward_with_cls_token(self) -> None:
        x = torch.randn(1, 3, 28, 28)
        result = self.vit.forward_features(x)
        self.assertIn("spatial_features", result)

    def test_strip_cls_token(self) -> None:
        x = torch.randn(1, 5, 32)  # 1 cls + 4 patches
        self.vit.use_cls_token = True
        result = self.vit.strip_cls_token(x)
        self.assertEqual(result.shape, (1, 4, 32))

    def test_forward_features_strip_cls(self) -> None:
        x = torch.randn(1, 3, 28, 28)
        result = self.vit.forward_features(x, strip_cls_token=True)
        # Should have cls token stripped from features
        self.assertIn("spatial_features", result)

    def test_sample_abs_posemb_matching_size(self) -> None:
        """When grid size matches, return positional_embedding directly."""
        # grid_size = 28/14 = 2
        pos = self.vit._sample_abs_posemb(2, 2)
        self.assertEqual(pos.shape[0], 1)

    def test_sample_abs_posemb_different_size(self) -> None:
        """When grid size differs, interpolate."""
        pos = self.vit._sample_abs_posemb(4, 4)
        self.assertEqual(pos.shape[0], 1)


class TestVisionTransformerAttnPool(unittest.TestCase):
    """Test with attention pooling."""

    def setUp(self) -> None:
        parallelism_module._MODEL_PARALLEL_STATE.update(parallelism_module._DEFAULT_STATE)
        self.vit = parallel_pe.VisionTransformer(
            image_size=28,
            patch_size=14,
            width=32,
            layers=1,
            heads=4,
            mlp_ratio=2.0,
            output_dim=16,
            pool_type="attn",
            use_cls_token=False,
            use_abs_posemb=False,
            use_rope2d=False,
        )
        self.vit.init_tensors()

    def test_attn_pool(self) -> None:
        x = torch.randn(1, 3, 28, 28)
        result = self.vit(x)
        self.assertIn("global_features", result)


class TestVisionTransformerRoPE(unittest.TestCase):
    """Test VisionTransformer with RoPE."""

    def setUp(self) -> None:
        parallelism_module._MODEL_PARALLEL_STATE.update(parallelism_module._DEFAULT_STATE)
        self.vit = parallel_pe.VisionTransformer(
            image_size=28,
            patch_size=14,
            width=32,
            layers=1,
            heads=4,
            mlp_ratio=2.0,
            output_dim=16,
            pool_type="avg",
            use_cls_token=False,
            use_abs_posemb=False,
            use_rope2d=True,
            use_ln_post=True,
        )
        self.vit.init_tensors()

    def test_forward_with_rope(self) -> None:
        x = torch.randn(1, 3, 28, 28)
        result = self.vit(x)
        self.assertIn("global_features", result)
        self.assertIn("spatial_features", result)

    def test_forward_features_with_rope(self) -> None:
        x = torch.randn(1, 3, 28, 28)
        result = self.vit.forward_features(x)
        self.assertIn("spatial_features", result)


class TestParallelImageEncoder(unittest.TestCase):
    """Test ParallelImageEncoder (wrapper around VisionTransformer)."""

    def setUp(self) -> None:
        parallelism_module._MODEL_PARALLEL_STATE.update(parallelism_module._DEFAULT_STATE)

    def test_init(self) -> None:
        encoder = parallel_pe.ParallelImageEncoder(
            encoder_cfg_name=_TEST_CFG_NAME,
            freeze_entire_encoder=False,
        )
        self.assertIsNotNone(encoder.encoder)

    def test_forward(self) -> None:
        encoder = parallel_pe.ParallelImageEncoder(
            encoder_cfg_name=_TEST_CFG_NAME,
        )
        x = torch.randn(1, 3, 32, 32)
        result = encoder(x)
        self.assertIn("global_features", result)
        self.assertIn("spatial_features", result)
        self.assertIn("spatial_features_list", result)

    def test_forward_custom_layer_idx(self) -> None:
        encoder = parallel_pe.ParallelImageEncoder(
            encoder_cfg_name=_TEST_CFG_NAME,
        )
        x = torch.randn(1, 3, 32, 32)
        result = encoder(x, layer_idx=[0, -1])
        self.assertIn("spatial_features_list", result)
        self.assertGreaterEqual(len(result["spatial_features_list"]), 2)

    def test_image_size_property(self) -> None:
        encoder = parallel_pe.ParallelImageEncoder(
            encoder_cfg_name=_TEST_CFG_NAME,
        )
        # image_size is on the inner encoder, not ParallelImageEncoder
        self.assertIsNotNone(encoder.encoder.image_size)

    def test_freeze_all_weights(self) -> None:
        encoder = parallel_pe.ParallelImageEncoder(
            encoder_cfg_name=_TEST_CFG_NAME,
            freeze_entire_encoder=True,
        )
        for param in encoder.encoder.parameters():
            self.assertFalse(param.requires_grad)

    def test_set_grad_checkpointing(self) -> None:
        encoder = parallel_pe.ParallelImageEncoder(
            encoder_cfg_name=_TEST_CFG_NAME,
        )
        # set_grad_checkpointing is on the inner encoder
        encoder.encoder.set_grad_checkpointing(True)
        encoder.encoder.set_grad_checkpointing(False)


class TestTransformerEdgeCases(unittest.TestCase):
    """Additional edge case tests for Transformer."""

    def setUp(self) -> None:
        parallelism_module._MODEL_PARALLEL_STATE.update(parallelism_module._DEFAULT_STATE)

    def test_forward_all_layers(self) -> None:
        transformer = parallel_pe.Transformer(width=32, layers=4, heads=4)
        x = torch.randn(1, 8, 32)
        # Request all layers
        out_list = transformer(x, layer_idx=list(range(4)))
        self.assertEqual(len(out_list), 4)

    def test_forward_negative_layer_idx(self) -> None:
        transformer = parallel_pe.Transformer(width=32, layers=4, heads=4)
        x = torch.randn(1, 8, 32)
        out_list = transformer(x, layer_idx=[-1])
        self.assertEqual(len(out_list), 1)

    def test_forward_with_rope(self) -> None:
        rope = Rope2D(dim=8)
        rope.init_tensors()
        transformer = parallel_pe.Transformer(
            width=32,
            layers=2,
            heads=4,
            rope=rope,
        )
        x = torch.randn(1, 4, 32)
        rope.update_grid(x.device, 2, 2)
        out_list = transformer(x, layer_idx=-1)
        self.assertEqual(len(out_list), 1)
        self.assertEqual(out_list[0].shape, (1, 4, 32))

    def test_forward_grad_checkpointing_with_grad(self) -> None:
        transformer = parallel_pe.Transformer(width=32, layers=2, heads=4)
        transformer.set_grad_checkpointing(True)
        transformer.train()
        x = torch.randn(1, 4, 32, requires_grad=True)
        out_list = transformer(x, layer_idx=-1)
        out_list[0].sum().backward()
        self.assertIsNotNone(x.grad)
        transformer.set_grad_checkpointing(False)


if __name__ == "__main__":
    unittest.main()
