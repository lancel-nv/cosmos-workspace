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

from typing import Any
import unittest

import attrs
import torch
from torch import nn

from checks.utils.models.cwip.modules.configs.defaults.pe_configs import (
    DEFAULT_PE_CONFIGS,
    VisionTransformerConfig,
    get_pe_config,
)
from checks.utils.models.cwip.modules.parallel_pe import VisionTransformer


def _vit_from_pe_config(cfg: VisionTransformerConfig) -> VisionTransformer:
    d = attrs.asdict(cfg)
    vit = VisionTransformer(
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
        act_layer=d.get("act_layer", nn.GELU),
        norm_layer=d.get("norm_layer", nn.LayerNorm),
        drop_path=d["drop_path"],
        use_tensor_parallel=False,
        attn_pooler_heads=d.get("attn_pooler_heads"),
    )
    vit.init_tensors()
    return vit


class TestVisionTransformerConfig(unittest.TestCase):
    """Tests for VisionTransformerConfig attrs class."""

    def test_required_fields(self) -> None:
        cfg = VisionTransformerConfig(
            patch_size=16,
            width=64,
            layers=2,
            heads=4,
            output_dim=64,
        )
        self.assertEqual(cfg.patch_size, 16)
        self.assertEqual(cfg.width, 64)
        self.assertEqual(cfg.layers, 2)
        self.assertEqual(cfg.heads, 4)
        self.assertEqual(cfg.output_dim, 64)

    def test_default_values(self) -> None:
        cfg = VisionTransformerConfig(
            patch_size=16,
            width=64,
            layers=2,
            heads=4,
            output_dim=64,
        )
        self.assertEqual(cfg.mlp_ratio, 4.0)
        self.assertIsNone(cfg.ls_init_value)
        self.assertEqual(cfg.drop_path, 0.0)
        self.assertEqual(cfg.image_size, 224)
        self.assertTrue(cfg.use_abs_posemb)
        self.assertTrue(cfg.use_cls_token)
        self.assertFalse(cfg.use_rope2d)
        self.assertEqual(cfg.pool_type, "cls")
        self.assertEqual(cfg.attn_pooler_heads, 8)
        self.assertTrue(cfg.use_ln_pre)
        self.assertTrue(cfg.use_ln_post)
        self.assertEqual(cfg.act_layer, nn.GELU)
        self.assertEqual(cfg.norm_layer, nn.LayerNorm)

    def test_custom_values(self) -> None:
        cfg = VisionTransformerConfig(
            patch_size=14,
            width=1024,
            layers=24,
            heads=16,
            output_dim=768,
            mlp_ratio=5.0,
            ls_init_value=1e-5,
            drop_path=0.1,
            image_size=512,
            use_abs_posemb=False,
            use_cls_token=False,
            use_rope2d=True,
            pool_type="attn",
            attn_pooler_heads=4,
            use_ln_pre=False,
            use_ln_post=False,
        )
        self.assertEqual(cfg.mlp_ratio, 5.0)
        self.assertEqual(cfg.ls_init_value, 1e-5)
        self.assertEqual(cfg.drop_path, 0.1)
        self.assertEqual(cfg.image_size, 512)
        self.assertFalse(cfg.use_abs_posemb)
        self.assertFalse(cfg.use_cls_token)
        self.assertTrue(cfg.use_rope2d)
        self.assertEqual(cfg.pool_type, "attn")
        self.assertEqual(cfg.attn_pooler_heads, 4)


class TestDefaultPEConfigs(unittest.TestCase):
    """Tests for DEFAULT_PE_CONFIGS dictionary."""

    def test_all_entries_are_valid_configs(self) -> None:
        for name, config in DEFAULT_PE_CONFIGS.items():
            with self.subTest(name=name):
                self.assertIsInstance(config, VisionTransformerConfig)

    def test_all_configs_have_positive_dimensions(self) -> None:
        for name, config in DEFAULT_PE_CONFIGS.items():
            with self.subTest(name=name):
                self.assertGreater(config.patch_size, 0)
                self.assertGreater(config.width, 0)
                self.assertGreater(config.layers, 0)
                self.assertGreater(config.heads, 0)
                self.assertGreater(config.output_dim, 0)
                self.assertGreater(config.image_size, 0)

    def test_width_divisible_by_heads(self) -> None:
        for name, config in DEFAULT_PE_CONFIGS.items():
            with self.subTest(name=name):
                self.assertEqual(config.width % config.heads, 0)

    def test_config_families_present(self) -> None:
        core_names = [n for n in DEFAULT_PE_CONFIGS if n.startswith("PE-Core")]
        lang_names = [n for n in DEFAULT_PE_CONFIGS if n.startswith("PE-Lang")]
        spatial_names = [n for n in DEFAULT_PE_CONFIGS if n.startswith("PE-Spatial")]
        self.assertGreater(len(core_names), 0)
        self.assertGreater(len(lang_names), 0)
        self.assertGreater(len(spatial_names), 0)

    def test_known_configs_exist(self) -> None:
        self.assertIn("PE-Spatial-B16-512", DEFAULT_PE_CONFIGS)
        self.assertIn("PE-Spatial-T16-512", DEFAULT_PE_CONFIGS)
        self.assertIn("PE-Core-G14-448", DEFAULT_PE_CONFIGS)

    def test_spatial_b16_512_specifics(self) -> None:
        cfg = DEFAULT_PE_CONFIGS["PE-Spatial-B16-512"]
        self.assertEqual(cfg.patch_size, 16)
        self.assertEqual(cfg.width, 768)
        self.assertEqual(cfg.layers, 12)
        self.assertEqual(cfg.heads, 12)
        self.assertEqual(cfg.image_size, 512)

    def test_spatial_t16_512_specifics(self) -> None:
        cfg = DEFAULT_PE_CONFIGS["PE-Spatial-T16-512"]
        self.assertEqual(cfg.patch_size, 16)
        self.assertEqual(cfg.width, 192)
        self.assertEqual(cfg.layers, 12)
        self.assertEqual(cfg.heads, 3)
        self.assertEqual(cfg.image_size, 512)


class TestVisionTransformerFromPeConfig(unittest.TestCase):
    """Build :class:`VisionTransformer` from :class:`VisionTransformerConfig` (TP off)."""

    def test_initializes_layerscale(self) -> None:
        cfg = VisionTransformerConfig(
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

        vit = _vit_from_pe_config(cfg)

        self.assertTrue(hasattr(vit, "transformer"))
        self.assertTrue(hasattr(vit.transformer, "resblocks"))
        self.assertEqual(len(vit.transformer.resblocks), cfg.layers)

        for block in vit.transformer.resblocks:
            if hasattr(block, "ls_1") and hasattr(block.ls_1, "init_tensors"):
                ls_1: Any = block.ls_1
                self.assertTrue(hasattr(ls_1, "gamma"))
                self.assertIsNotNone(ls_1.gamma)
                self.assertEqual(ls_1.gamma.shape[0], cfg.width)
            if hasattr(block, "ls_2") and hasattr(block.ls_2, "init_tensors"):
                ls_2: Any = block.ls_2
                self.assertTrue(hasattr(ls_2, "gamma"))
                self.assertIsNotNone(ls_2.gamma)
                self.assertEqual(ls_2.gamma.shape[0], cfg.width)

        x = torch.randn(1, 3, cfg.image_size, cfg.image_size)
        out = vit(x)
        self.assertIsInstance(out, dict)
        self.assertIn("global_features", out)
        self.assertIn("spatial_features", out)
        self.assertIn("spatial_features_list", out)
        self.assertEqual(out["global_features"].shape[-1], cfg.output_dim)

    def test_without_layerscale(self) -> None:
        cfg = VisionTransformerConfig(
            patch_size=16,
            width=64,
            layers=1,
            heads=4,
            output_dim=64,
            mlp_ratio=2.0,
            ls_init_value=None,
            image_size=32,
            use_abs_posemb=False,
            use_cls_token=False,
            use_rope2d=False,
            pool_type="avg",
            attn_pooler_heads=2,
        )
        vit = _vit_from_pe_config(cfg)
        x = torch.randn(1, 3, cfg.image_size, cfg.image_size)
        out = vit(x)
        self.assertIn("global_features", out)


class TestGetPeConfig(unittest.TestCase):
    """Tests for get_pe_config function."""

    def test_returns_known_config(self) -> None:
        config = get_pe_config("PE-Spatial-B16-512")
        self.assertIsInstance(config, VisionTransformerConfig)
        self.assertEqual(config.patch_size, 16)
        self.assertEqual(config.width, 768)
        self.assertEqual(config.image_size, 512)

    def test_raises_on_unknown_config(self) -> None:
        with self.assertRaises(ValueError) as context:
            get_pe_config("Unknown-Model")
        self.assertIn("Unknown PE config", str(context.exception))
        self.assertIn("Unknown-Model", str(context.exception))

    def test_all_default_configs_accessible(self) -> None:
        for name in DEFAULT_PE_CONFIGS:
            config = get_pe_config(name)
            self.assertIsInstance(config, VisionTransformerConfig)

    def test_returns_same_object_as_registry(self) -> None:
        config = get_pe_config("PE-Spatial-T16-512")
        self.assertIs(config, DEFAULT_PE_CONFIGS["PE-Spatial-T16-512"])


if __name__ == "__main__":
    unittest.main()
