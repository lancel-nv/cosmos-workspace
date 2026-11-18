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
#
# Portions of the VisionTransformer class are derived from core/vision_encoder/pe.py in
# https://github.com/facebookresearch/perception_models (commit 430a0121bd30e92df635bd59f724b2c52aa5bdd7),
# Copyright (c) Meta Platforms, Inc. and affiliates, licensed under the Apache License, Version 2.0, and have been
# modified by NVIDIA.

"""
Parallel PE (Perception Encoder): ViT-compatible stack with optional **tensor parallelism**.

**Tensor parallelism (TP)** — :class:`VisionTransformer` takes ``use_tensor_parallel`` (default ``True``) and passes it
to :class:`Transformer` → :class:`ResidualAttentionBlock``. When that flag is ``True`` and the tensor-parallel world
size is > 1, attention and MLP in each block are sharded and all-reduced. :class:`ParallelImageEncoder` sets the flag on
the inner ViT only when ``use_tensor_parallel`` and ``tp_size > 1``. Patch embed, pre/post norms, pooling, and
projection stay **replicated**.

**Context parallelism (CP)** — Not supported in this module (no sequence-splitting CP on the PE ViT). Use
:mod:`parallel_patchformer` for CP.

**Sequence parallelism** — Not implemented in this module.

Layer names match the original PE module for checkpoint loading.

Usage:
    from checks.utils.models.cwip.modules.parallel_pe import ParallelImageEncoder

    encoder = ParallelImageEncoder(
        encoder_cfg_name="pe_core_spatial_3b",
        use_tensor_parallel=True,
    )
"""

from __future__ import annotations

from collections.abc import Callable
import copy
import logging
from typing import Any

from third_party.perception_models.core.vision_encoder import pe
from third_party.perception_models.core.vision_encoder.rope import Rope2D
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from checks.utils.models.cwip.modules import attention as attention_module
from checks.utils.models.cwip.modules import parallelism as parallelism_module
from checks.utils.models.cwip.modules.configs.defaults import pe_configs

logger = logging.getLogger(__name__)


class SelfAttention(attention_module.SelfAttention):
    """
    PE-style self-attention: fused ``in_proj_*`` (checkpoint-compatible).

    Same as ``attention_module.SelfAttention`` with ``use_fused_qkv=True``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rope: nn.Module | None = None,
        dropout: float = 0.0,
        use_tensor_parallel: bool = True,
    ):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            rope=rope,
            dropout=dropout,
            use_tensor_parallel=use_tensor_parallel,
            use_fused_qkv=True,
        )

    def set_context_parallel_group(self, group: object) -> None:
        """PE attention does not support context parallelism; use :mod:`parallel_patchformer` for CP."""
        raise RuntimeError(
            "Context parallelism is not supported on the PE encoder path. "
            "Use checks.utils.models.cwip.modules.parallel_patchformer for CP."
        )


class ResidualAttentionBlock(attention_module.ResidualAttentionBlock):
    """ViT residual block with fused QKV self-attention (same names as ``pe``)."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float | None = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: nn.Module | None = None,
        use_tensor_parallel: bool = True,
    ):
        super().__init__(
            d_model=d_model,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path=drop_path,
            rope=rope,
            use_cross_attn=False,
            use_tensor_parallel=use_tensor_parallel,
            use_fused_qkv=True,
        )


# =============================================================================
# Parallel Transformer (maintains same interface as pe.Transformer)
# =============================================================================


class Transformer(nn.Module):
    """
    Transformer with tensor parallelism.

    Maintains the same interface and layer names as ``pe.Transformer`` for checkpoint
    compatibility. Context parallelism is not supported on this path (see module docstring).
    """

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float | None = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: nn.Module | None = None,
        use_tensor_parallel: bool = True,
    ):
        super().__init__()

        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.use_tensor_parallel = use_tensor_parallel

        # Create parallel (TP) residual blocks.
        # IMPORTANT: Keep the name "resblocks" for checkpoint compatibility
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    d_model=width,
                    n_head=heads,
                    mlp_ratio=mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop_path=drop_path,
                    rope=rope,
                    use_tensor_parallel=use_tensor_parallel,
                )
                for _ in range(layers)
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        layer_idx: int | list[int] = -1,
    ) -> list[torch.Tensor]:
        """
        Forward the transformer.

        Args:
            x: (B, S, D)
            attn_mask: Optional attention mask
            layer_idx: Layer index(es) to return features from

        Returns:
            List of features from specified layers
        """
        # Get the stop indices for layers to return
        if isinstance(layer_idx, int):
            stop_idx = [(self.layers + layer_idx) % self.layers]
        elif isinstance(layer_idx, list):
            stop_idx = [(self.layers + i) % self.layers for i in layer_idx]
            stop_idx = sorted(stop_idx)
        else:
            raise TypeError(f"Invalid layer_idx: {layer_idx}")

        x_list = []
        for i, block in enumerate(self.resblocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x, attn_mask, use_reentrant=False)
            else:
                x = block(x, attn_mask=attn_mask)

            # Collect features at specified layers
            while i in stop_idx:
                x_list.append(x)
                if len(stop_idx) == 1:
                    break
                stop_idx.pop(0)

        if len(x_list) == 0:
            x_list = [x]

        return x_list


# =============================================================================
# Parallel Vision Transformer
# =============================================================================


class VisionTransformer(nn.Module):
    """
    Vision Transformer with optional tensor parallelism in the transformer trunk.

    Maintains checkpoint compatibility with pe.VisionTransformer by keeping
    the same layer names and structure.

    ``use_tensor_parallel`` is passed to :class:`Transformer` and each
    :class:`ResidualAttentionBlock` (attention + MLP sharding when True and TP world
    size > 1). Patch embed, pooling, and projection stay replicated.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        width: int = 1024,
        layers: int = 24,
        heads: int = 16,
        mlp_ratio: float = 4.0,
        output_dim: int | None = None,
        pool_type: str = "attn",
        use_cls_token: bool = False,
        use_abs_posemb: bool = False,
        use_rope2d: bool = True,
        use_ln_post: bool = True,
        ls_init_value: float | None = 1e-6,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        use_tensor_parallel: bool = True,
        attn_pooler_heads: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.output_dim = output_dim
        self.pool_type = pool_type
        self.use_cls_token = use_cls_token
        self.use_abs_posemb = use_abs_posemb
        self.use_rope2d = use_rope2d
        self.use_ln_post = use_ln_post
        self.use_tensor_parallel = use_tensor_parallel
        self._attn_pooler_heads = attn_pooler_heads if attn_pooler_heads is not None else heads

        self.proj_dim = output_dim

        # Patch embedding (not parallelized, same on all ranks)
        # IMPORTANT: Keep name "conv1" for checkpoint compatibility
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # RoPE 2D for positional encoding
        # RoPE2D is used in each multi-head attention block
        # For each head, the first half of feature dimension is applied x-direction RoPE,
        # and the second half is applied y-direction RoPE
        # See https://github.com/limefax/rope-nd for visual example
        self.rope = Rope2D(dim=width // heads, use_cls_token=self.use_cls_token) if use_rope2d else None

        # Pre-transformer layer norm
        # IMPORTANT: Keep name "ln_pre" for checkpoint compatibility
        self.ln_pre = norm_layer(width)

        # Parallel transformer (same class for TP=1 and TP>1; checkpoint-compatible)
        # IMPORTANT: Keep name "transformer" for checkpoint compatibility
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path=drop_path,
            rope=self.rope,
            use_tensor_parallel=use_tensor_parallel,
        )

        # NOTE: We do NOT set true context parallelism (ring attention) on the PE encoder. Frame-level distribution is
        # handled by frame parallelism (FP): each FP rank encodes a subset of frames independently. The PE encoder's
        # attention is over spatial tokens within a single frame. Applying true CP (sequence-splitting) here would
        # require that sequence length (e.g. 1025 with CLS) be evenly divisible by cp_size, which generally doesn't
        # hold. True CP is used in PatchFormer only; PE stays FP/TP only.

        # Post-transformer layer norm
        # IMPORTANT: Keep name "ln_post" for checkpoint compatibility
        self.ln_post = norm_layer(width) if use_ln_post else nn.Identity()

        # Pooling (attn_pooler_heads can differ from heads for checkpoint compatibility)
        if pool_type == "attn":
            self.attn_pool = pe.AttentionPooling(embed_dim=width, num_heads=self._attn_pooler_heads)
        else:
            self.attn_pool = None

        # Store grid size for positional embeddings
        self.posemb_grid_size = image_size // patch_size

    def init_tensors(self) -> None:
        """Initialize tensors for the vision transformer."""

        def init_submodule_tensors(module: Any) -> None:
            for name, child in module.named_children():
                if hasattr(child, "init_tensors"):
                    logger.debug("Initializing tensors for submodule: %s", name)
                    child.init_tensors()
                init_submodule_tensors(child)

        init_submodule_tensors(self)

        if hasattr(self, "rope") and self.rope is not None:
            self.rope.init_tensors()

        # Initialize class embedding and positional embedding
        init_scale = self.width**-0.5

        if self.use_cls_token:
            self.class_embedding = nn.Parameter(init_scale * torch.randn(self.width))

        if self.use_abs_posemb:
            self.positional_embedding = nn.Parameter(
                init_scale * torch.randn(int(self.use_cls_token) + self.posemb_grid_size**2, self.width)
            )

        if self.proj_dim is not None:
            self.proj = nn.Parameter(init_scale * torch.randn(self.width, self.proj_dim))

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.transformer.set_grad_checkpointing(enable)

    def _sample_abs_posemb(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """Sample absolute positional embedding for given grid size."""
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]

        pos_embed: torch.Tensor = self.positional_embedding

        if self.use_cls_token:
            cls_embed = pos_embed[:1, :]
            pos_embed = pos_embed[1:, :]
        else:
            cls_embed = None

        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1).permute(0, 3, 1, 2).contiguous()
        )

        pos_embed = F.interpolate(pos_embed, size=(grid_h, grid_w), mode="bilinear", align_corners=False)

        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, -1, self.width).contiguous()

        if cls_embed is not None:
            pos_embed = torch.cat([cls_embed[None, ...], pos_embed], dim=1)

        return pos_embed

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """Pool spatial features to global features.

        AttentionPooling returns (B, num_probe, D) with num_probe=1 by default. Squeeze the probe dimension so callers
        get a clean (B, D) tensor.
        """
        if self.pool_type == "attn":
            out = self.attn_pool(x)
            return out.squeeze(1) if out.dim() == 3 and out.shape[1] == 1 else out
        elif self.pool_type == "avg":
            return x.mean(dim=1)
        elif self.pool_type == "none":
            return x
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")

    def strip_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """Strip the class token from features."""
        if self.use_cls_token:
            x = x[:, 1:, :]
        return x

    def forward_features(
        self,
        x: torch.Tensor,
        norm: bool = False,
        layer_idx: int | list[int] = -1,
        strip_cls_token: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Forward pass returning spatial features.

        Args:
            x: Input tensor [B, 3, H, W]
            norm: Whether to apply layer norm to output
            layer_idx: Layer indices to return features from
            strip_cls_token: Whether to strip class token from output

        Returns:
            Dictionary with spatial_features and spatial_features_list
        """
        batch, _, h, w = x.shape
        grid_h, grid_w = h // self.patch_size, w // self.patch_size

        # Patch embedding
        x = self.conv1(x)

        # Reshape: [B, C, H/P, W/P] -> [B, H/P * W/P, C]
        x = x.permute(0, 2, 3, 1).reshape(batch, -1, self.width)

        # Add class token if used
        if self.use_cls_token:
            x = torch.cat(
                [self.class_embedding.view(1, 1, -1).expand(batch, -1, -1), x],
                dim=1,
            )

        # Add positional embedding
        if self.use_abs_posemb:
            x = x + self._sample_abs_posemb(grid_h, grid_w)

        # Update RoPE grid
        if self.use_rope2d and self.rope is not None:
            self.rope.update_grid(x.device, grid_h, grid_w)

        # Pre-transformer norm
        x = self.ln_pre(x)

        # NOTE: Frame-level CP is now handled by CWIP.forward() which splits frames across CP ranks BEFORE calling the
        # encoder. Patch-level CP splitting here would cause double-splitting issues and is disabled. The ring
        # attention in transformer blocks still provides cross-frame attention when needed, but the primary memory
        # reduction comes from frame-level CP.

        # Transformer
        if isinstance(layer_idx, int):
            layer_idx = [layer_idx]
        # Ensure the final layer is in the transformer output, but don't auto-add it if the caller already asked for it
        # (explicitly or via the equivalent positive index ``self.transformer.layers - 1``). Explicit duplicates of
        # ``-1`` in ``layer_idx`` are preserved 1:1 in ``x_list`` (the transformer's ``while i in stop_idx: pop`` loop
        # appends one copy per duplicate). The conditional pop below removes the auto-appended entry iff we added it,
        # leaving the user-requested layers untouched in ``x_list``.
        if -1 not in layer_idx and (self.transformer.layers - 1) not in layer_idx:
            layer_idx = layer_idx + [-1]
            _auto_appended = True
        else:
            _auto_appended = False
        x_list = self.transformer(x, layer_idx=layer_idx)

        if _auto_appended:
            x = x_list.pop()
        else:
            # ``x_list[-1]`` is the user's last requested layer, which is the final block whenever ``-1``
            # (or ``layers - 1``) was in the request -- guaranteed because we only skip the auto-append in exactly that
            # case.
            x = x_list[-1]

        # Post-transformer norm
        if norm:
            x = self.ln_post(x)

        # Strip class token if requested
        if strip_cls_token:
            x = self.strip_cls_token(x)
            x_list = [self.strip_cls_token(xi) for xi in x_list]

        return {
            "spatial_features": x,
            "spatial_features_list": x_list,
        }

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int | list[int] = -1,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Forward pass returning global and spatial features.

        Args:
            x: Input tensor [B, 3, H, W]
            layer_idx: Layer indices to return features from
            **kwargs: Additional arguments

        Returns:
            Dictionary with global_features, spatial_features, spatial_features_list
        """
        image_features = self.forward_features(x, norm=True, layer_idx=layer_idx, **kwargs)

        spatial = image_features["spatial_features"]
        assert isinstance(spatial, torch.Tensor)
        global_features = self._pool(spatial)
        if hasattr(self, "proj") and self.proj_dim is not None:
            global_features = global_features @ self.proj

        return {
            "global_features": global_features,
            "spatial_features": image_features["spatial_features"],
            "spatial_features_list": image_features["spatial_features_list"],
        }


# =============================================================================
# Parallel Image Encoder (drop-in replacement for ImageEncoder)
# Single composed ViT: layers = base_layers + num_alignment_layers, with ln_post + attn_pool + proj.
# Checkpoint layout: encoder.transformer.resblocks.0..(L-1), encoder.ln_post, encoder.attn_pool, encoder.proj.
# =============================================================================


class ParallelImageEncoder(nn.Module):
    """
    Image encoder with tensor parallelism (PE path).

    This is a drop-in replacement for the original ImageEncoder. Frame-level parallelism is handled outside this
    module. Context parallelism is not supported here (see module docstring).

    Maintains checkpoint compatibility with the non-parallel encoder layout.

    Usage:
        encoder = ParallelImageEncoder(
            encoder_cfg_name="pe_core_spatial_3b",
            use_tensor_parallel=True,
        )
    """

    def __init__(
        self,
        encoder_cfg_name: str,
        freeze_entire_encoder: bool = True,
        output_dim: int | None = 1024,
        num_alignment_layers: int = 0,
        use_tensor_parallel: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the parallel image encoder.

        Args:
            encoder_cfg_name: Name of the encoder configuration
            freeze_entire_encoder: Whether to freeze all encoder weights (default True for deployment)
            output_dim: Output dimension
            num_alignment_layers: Number of alignment layers to add
            use_tensor_parallel: Whether to enable Tensor Parallelism
        """
        super().__init__()

        encoder_cfg = copy.deepcopy(pe_configs.DEFAULT_PE_CONFIGS[encoder_cfg_name])

        self.num_spatial_layers = encoder_cfg.layers
        self.num_alignment_layers = num_alignment_layers
        self.num_layers = self.num_spatial_layers + self.num_alignment_layers
        self.output_dim = output_dim
        self.use_tensor_parallel = use_tensor_parallel

        tp_size = parallelism_module.get_tensor_parallel_world_size() if use_tensor_parallel else 1
        fp_size = parallelism_module.get_frame_parallel_world_size()

        # Single composed ViT: layers = base + alignment, with ln_post + attn_pool + proj
        if self.output_dim is not None:
            encoder_cfg.output_dim = self.output_dim
        encoder_cfg.pool_type = "attn"
        encoder_cfg.use_ln_post = True
        encoder_cfg.layers = self.num_layers  # base layers + num_alignment_layers

        msg = (
            f"ParallelImageEncoder: {encoder_cfg_name}, "
            f"layers={encoder_cfg.layers}, width={encoder_cfg.width}, "
            f"TP={tp_size}, FP={fp_size} (PE does not use true CP sequence-splitting attention)"
        )
        logger.info(msg)
        print(msg)

        # Always use VisionTransformer (TP=1 or TP>1; checkpoint-compatible)
        self.encoder = VisionTransformer(
            image_size=encoder_cfg.image_size,
            patch_size=encoder_cfg.patch_size,
            width=encoder_cfg.width,
            layers=encoder_cfg.layers,
            heads=encoder_cfg.heads,
            mlp_ratio=encoder_cfg.mlp_ratio,
            output_dim=encoder_cfg.output_dim,
            pool_type=encoder_cfg.pool_type,
            use_cls_token=encoder_cfg.use_cls_token,
            use_abs_posemb=encoder_cfg.use_abs_posemb,
            use_rope2d=encoder_cfg.use_rope2d,
            use_ln_post=encoder_cfg.use_ln_post,
            ls_init_value=encoder_cfg.ls_init_value,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            drop_path=encoder_cfg.drop_path,
            use_tensor_parallel=use_tensor_parallel and tp_size > 1,
            attn_pooler_heads=getattr(encoder_cfg, "attn_pooler_heads", None),
        )
        self.encoder.init_tensors()

        if freeze_entire_encoder:
            self.freeze_all_weights()

    def freeze_all_weights(self, verbose: bool = False) -> None:
        """Freeze all encoder weights."""
        total_params = 0
        for name, param in self.encoder.named_parameters():
            if verbose:
                logger.info("Freezing %s", name)
            param.requires_grad = False
            total_params += param.numel()

        self.encoder.eval()
        logger.info("Froze entire encoder: %d parameters", total_params)

    def forward(
        self,
        x: torch.Tensor,
        layer_idx: int | list[int] = -1,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Forward pass through the encoder.

        Args:
            x: Input tensor [B, C, H, W]
            layer_idx: Layer indices to return features from
            **kwargs: Additional arguments

        Returns:
            Dictionary with features from the encoder
        """
        if not any(p.requires_grad for p in self.encoder.parameters()):
            with torch.no_grad():
                out = self.encoder.forward(x, layer_idx=layer_idx, **kwargs)
            return out
        return self.encoder.forward(x, layer_idx=layer_idx, **kwargs)
