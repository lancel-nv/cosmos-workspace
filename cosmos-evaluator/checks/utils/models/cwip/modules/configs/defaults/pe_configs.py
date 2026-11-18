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

"""
Configuration classes for Perception Encoder (PE) components.
"""

from collections.abc import Callable

import attrs
from torch import nn


@attrs.define(slots=False)
class VisionTransformerConfig:
    """Config for VisionTransformer module."""

    patch_size: int
    width: int
    layers: int
    heads: int
    output_dim: int
    mlp_ratio: float = 4.0
    ls_init_value: float | None = None
    drop_path: float = 0.0
    image_size: int = 224
    use_abs_posemb: bool = True
    use_cls_token: bool = True
    use_rope2d: bool = False
    pool_type: str = "cls"
    attn_pooler_heads: int = 8
    use_ln_pre: bool = True
    use_ln_post: bool = True
    act_layer: Callable = nn.GELU
    norm_layer: Callable = nn.LayerNorm


# Default configurations based on PE_VISION_CONFIG entries
DEFAULT_PE_CONFIGS: dict[str, VisionTransformerConfig] = {
    # PE-Core models
    "PE-Core-G14-448": VisionTransformerConfig(
        patch_size=14,
        width=1536,
        layers=50,
        heads=16,
        output_dim=1280,
        mlp_ratio=5.833333333333333,
        ls_init_value=None,
        drop_path=0.0,
        image_size=448,
        use_abs_posemb=True,
        use_cls_token=False,
        use_rope2d=True,
        pool_type="attn",
        attn_pooler_heads=8,
        use_ln_pre=True,
        use_ln_post=True,
    ),
    "PE-Core-L14-336": VisionTransformerConfig(
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        output_dim=1024,
        mlp_ratio=4.0,
        ls_init_value=None,
        drop_path=0.0,
        image_size=336,
        use_abs_posemb=True,
        use_cls_token=True,
        use_rope2d=True,
        pool_type="attn",
        attn_pooler_heads=8,
        use_ln_pre=True,
        use_ln_post=True,
    ),
    "PE-Core-B16-224": VisionTransformerConfig(
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        output_dim=1024,
        mlp_ratio=4.0,
        ls_init_value=None,
        drop_path=0.0,
        image_size=224,
        use_abs_posemb=True,
        use_cls_token=True,
        use_rope2d=True,
        pool_type="attn",
        attn_pooler_heads=8,
        use_ln_pre=True,
        use_ln_post=True,
    ),
    "PE-Core-S16-384": VisionTransformerConfig(
        patch_size=16,
        width=384,
        layers=12,
        heads=6,
        output_dim=512,
        mlp_ratio=4.0,
        ls_init_value=None,
        drop_path=0.0,
        image_size=384,
        use_abs_posemb=True,
        use_cls_token=True,
        use_rope2d=True,
        pool_type="attn",
        attn_pooler_heads=8,
        use_ln_pre=True,
        use_ln_post=True,
    ),
    "PE-Core-T16-384": VisionTransformerConfig(
        patch_size=16,
        width=192,
        layers=12,
        heads=3,
        output_dim=512,
        mlp_ratio=4.0,
        ls_init_value=None,
        drop_path=0.0,
        image_size=384,
        use_abs_posemb=True,
        use_cls_token=True,
        use_rope2d=True,
        pool_type="attn",
        attn_pooler_heads=8,
        use_ln_pre=True,
        use_ln_post=True,
    ),
    # PE-Lang models (note: output_dim=None in original configs, setting to reasonable values)
    "PE-Lang-G14-448": VisionTransformerConfig(
        patch_size=14,
        width=1536,
        layers=47,
        heads=16,
        output_dim=1536,  # Using width as output_dim since original is None
        mlp_ratio=5.833333333333333,
        ls_init_value=None,
        drop_path=0.0,
        image_size=448,
        use_abs_posemb=True,
        use_cls_token=False,
        use_rope2d=True,
        pool_type="none",
        attn_pooler_heads=8,
        use_ln_pre=True,
        use_ln_post=False,
    ),
    "PE-Lang-L14-448": VisionTransformerConfig(
        patch_size=14,
        width=1024,
        layers=23,
        heads=16,
        output_dim=1024,  # Using width as output_dim since original is None
        mlp_ratio=4.0,
        ls_init_value=None,
        drop_path=0.0,
        image_size=448,
        use_abs_posemb=True,
        use_cls_token=True,
        use_rope2d=True,
        pool_type="none",
        attn_pooler_heads=8,
        use_ln_pre=True,
        use_ln_post=False,
    ),
    "PE-Lang-G14-448-Tiling": VisionTransformerConfig(
        patch_size=14,
        width=1536,
        layers=47,
        heads=16,
        output_dim=1536,  # Using width as output_dim since original is None
        mlp_ratio=5.833333333333333,
        ls_init_value=None,
        drop_path=0.0,
        image_size=448,
        use_abs_posemb=True,
        use_cls_token=False,
        use_rope2d=True,
        pool_type="none",
        attn_pooler_heads=8,
        use_ln_pre=True,
        use_ln_post=False,
    ),
    "PE-Lang-L14-448-Tiling": VisionTransformerConfig(
        patch_size=14,
        width=1024,
        layers=23,
        heads=16,
        output_dim=1024,  # Using width as output_dim since original is None
        mlp_ratio=4.0,
        ls_init_value=None,
        drop_path=0.0,
        image_size=448,
        use_abs_posemb=True,
        use_cls_token=True,
        use_rope2d=True,
        pool_type="none",
        attn_pooler_heads=8,
        use_ln_pre=True,
        use_ln_post=False,
    ),
    # PE-Spatial models (note: output_dim=None in original configs, setting to width values)
    "PE-Spatial-G14-448": VisionTransformerConfig(
        patch_size=14,
        width=1536,
        layers=50,
        heads=16,
        output_dim=1536,  # Using width as output_dim since original is None
        mlp_ratio=5.833333333333333,
        ls_init_value=None,
        drop_path=0.0,
        image_size=448,
        use_abs_posemb=True,
        use_cls_token=False,
        use_rope2d=True,
        pool_type="none",
        attn_pooler_heads=8,
        use_ln_pre=True,
        use_ln_post=False,
    ),
    "PE-Spatial-L14-448": VisionTransformerConfig(
        patch_size=14,
        width=1024,
        layers=24,
        heads=16,
        output_dim=1024,  # Using width as output_dim since original is None
        mlp_ratio=4.0,
        ls_init_value=None,
        image_size=448,
        use_abs_posemb=True,
        use_cls_token=True,
        use_rope2d=True,
        pool_type="none",
        attn_pooler_heads=8,
        use_ln_pre=True,
        use_ln_post=False,
    ),
    "PE-Spatial-B16-512": VisionTransformerConfig(
        patch_size=16,
        width=768,
        layers=12,
        heads=12,
        output_dim=768,  # Using width as output_dim since original is None
        mlp_ratio=4.0,
        ls_init_value=None,
        drop_path=0.0,
        image_size=512,
        use_abs_posemb=True,
        use_cls_token=True,
        use_rope2d=True,
        pool_type="none",
        attn_pooler_heads=8,
        use_ln_pre=True,
        use_ln_post=False,
    ),
    "PE-Spatial-S16-512": VisionTransformerConfig(
        patch_size=16,
        width=384,
        layers=12,
        heads=6,
        output_dim=384,  # Using width as output_dim since original is None
        mlp_ratio=4.0,
        ls_init_value=None,
        drop_path=0.0,
        image_size=512,
        use_abs_posemb=True,
        use_cls_token=True,
        use_rope2d=True,
        pool_type="none",
        attn_pooler_heads=8,
        use_ln_pre=True,
        use_ln_post=False,
    ),
    "PE-Spatial-T16-512": VisionTransformerConfig(
        patch_size=16,
        width=192,
        layers=12,
        heads=3,
        output_dim=192,  # Using width as output_dim since original is None
        mlp_ratio=4.0,
        ls_init_value=None,
        drop_path=0.0,
        image_size=512,
        use_abs_posemb=True,
        use_cls_token=True,
        use_rope2d=True,
        pool_type="none",
        attn_pooler_heads=8,
        use_ln_pre=True,
        use_ln_post=False,
    ),
}


def get_pe_config(config_name: str) -> VisionTransformerConfig:
    """Get a PE configuration by name."""
    if config_name not in DEFAULT_PE_CONFIGS:
        raise ValueError(f"Unknown PE config: {config_name}. Available: {list(DEFAULT_PE_CONFIGS.keys())}")
    return DEFAULT_PE_CONFIGS[config_name]
