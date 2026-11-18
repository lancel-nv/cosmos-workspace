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
Tensor-parallel and context-parallel PatchFormer (patch-token attention).

``ParallelPatchFormer`` mirrors the PatchFormer constructor surface for drop-in use. **Tensor-parallel (TP)** and **true
context-parallel (CP)** world sizes are determined by the active distributed process groups at runtime, not by arguments
to this class. ``use_tensor_parallel`` only selects whether attention and MLP weights are sharded when the TP world size
is greater than one.

Use cases include large activations (long frame stacks, high resolution) and long patch sequences (patch tokens may be
sharded for gather-style CP).

Example (architecture only; in multi-GPU runs, initialize the appropriate parallel groups before ``forward``):

    from checks.utils.models.cwip.modules.parallel_patchformer import (
        ParallelPatchFormer,
        MixedAttentionTransformer,
    )

    model = ParallelPatchFormer(
        camera_dim=768,
        world_dim=768,
        num_classes=[22, 5],
        hidden_dim=768,
        num_layers=4,
        num_heads=12,
        use_tensor_parallel=True,
    )
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from typing import Any

from einops import rearrange
from third_party.perception_models.core.vision_encoder.rope import Rope2D
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from checks.utils.models.cwip.modules import attention as attention_module
from checks.utils.models.cwip.modules import parallelism as parallelism_module
from checks.utils.models.cwip.modules.class_mapping import ObjectTypeClassMapping, PatchClassificationType
from checks.utils.models.cwip.visualization import patch_vis_utils, world_model_utils

# Re-export for callers using ``parallel_patchformer.QueryUpsampleMethod`` etc. (tests).
CrossAttention = attention_module.CrossAttention
FusedKVCameraWorldCrossAttention = attention_module.FusedKVCameraWorldCrossAttention
DeformableAttention2D = attention_module.DeformableAttention2D
DeformableAttentionBlock = attention_module.DeformableAttentionBlock
QueryUpsampleMethod = attention_module.QueryUpsampleMethod
ResidualAttentionBlock = attention_module.ResidualAttentionBlock
SelfAttention = attention_module.SelfAttention

logger = logging.getLogger(__name__)


class MixedAttentionTransformer(nn.Module):
    """
    Mixed attention transformer with TP and CP support.

    Interleaves self-attention and cross-attention blocks.
    """

    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        q_dim: int | None = None,
        kv_dim: int | None = None,
        mlp_ratio: float = 4.0,
        ls_init_value: float | None = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: nn.Module | None = None,
        cross_attn_frequency: int | None = None,
        use_tensor_parallel: bool = True,
        fused_kv_camera_world_cross_attn: bool = False,
    ):
        """
        Initialize MixedAttentionTransformer.

        Args:
            width: Model embedding dimension.
            layers: Number of transformer layers.
            heads: Number of attention heads. Must be divisible by TP size.
            q_dim: Query dimension for cross-attention. Defaults to width if None.
            kv_dim: Key/Value dimension for cross-attention. Defaults to width if None.
            mlp_ratio: Ratio of MLP hidden dim to width.
            ls_init_value: Initial value for LayerScale. None disables LayerScale.
            act_layer: Activation layer class for MLP.
            norm_layer: Normalization layer class.
            drop_path: Drop path probability for stochastic depth.
            rope: Optional RoPE module for positional encoding.
            cross_attn_frequency: Layer ``i`` is cross-attention when ``(i + 1) % (N + 1) == 0`` for integer ``N >= 0``.
                ``None`` means all self-attention layers. ``N == 0`` makes every layer cross-attention.
                See :class:`ParallelPatchFormer` for interaction with fused K/V.
            use_tensor_parallel: Whether to shard weights across TP ranks.
            fused_kv_camera_world_cross_attn: If True, cross-attention layers use ``FusedKVCameraWorldCrossAttention``
                (Q from camera, K/V from camera and world concatenated on the sequence axis, with per-branch RoPE on
                keys). Ignored when there are no cross-attention layers (``cross_attn_frequency`` is ``None``).
        """
        super().__init__()

        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.cross_attn_frequency = cross_attn_frequency
        self.fused_kv_camera_world_cross_attn = fused_kv_camera_world_cross_attn

        # Determine which layers use cross attention
        if cross_attn_frequency is not None:
            self.cross_attn_indices = {i for i in range(layers) if (i + 1) % (cross_attn_frequency + 1) == 0}
        else:
            self.cross_attn_indices = set()

        # Create blocks
        self.resblocks = nn.ModuleList()
        for i in range(layers):
            use_cross_attn = i in self.cross_attn_indices
            block = attention_module.ResidualAttentionBlock(
                d_model=width,
                n_head=heads,
                q_dim=q_dim if use_cross_attn else None,
                kv_dim=kv_dim if use_cross_attn else None,
                mlp_ratio=mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_path=drop_path,
                rope=rope,
                use_cross_attn=use_cross_attn,
                use_tensor_parallel=use_tensor_parallel,
                use_fused_kv_camera_world_cross_attn=(fused_kv_camera_world_cross_attn and use_cross_attn),
            )
            self.resblocks.append(block)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """
        Enable or disable gradient checkpointing for memory optimization.

        Args:
            enable: If True, enable gradient checkpointing to trade compute for memory.
        """
        self.grad_checkpointing = enable

    def set_context_parallel_group(self, group: dist.ProcessGroup) -> None:
        """
        Set context parallel group for all attention modules.

        Args:
            group: The distributed process group for context parallelism.
        """
        for block in self.resblocks:
            set_cp_group = getattr(block.attn, "set_context_parallel_group", None)
            if set_cp_group is not None:
                set_cp_group(group)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through all transformer layers.

        Args:
            x: Input tensor of shape [B, S, width].
            context: Optional context tensor for cross-attention of shape [B, S_ctx, kv_dim].
            attn_mask: Optional attention mask.

        Returns:
            Output tensor of shape [B, S, width].
        """
        for block in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x, context, attn_mask, use_reentrant=False)
            else:
                x = block(x, context=context, attn_mask=attn_mask)
        return x


class ParallelPatchFormer(nn.Module):
    """
    PatchFormer with Tensor Parallelism and Context Parallelism support.

    This is the production-scale version designed for:
    - 600+ frames at 1K resolution
    - TP=4, CP=8 on 32-64 GPUs (A5 config)

    Memory optimization through:
    - TP: Split attention heads and MLP across GPUs (4x activation reduction)
    - CP: Split sequences across GPUs with gather + SDPA attention (sequence sharding)
    - SP: Sequence parallel for non-attention ops (combined with TP)
    """

    # Registered-buffer annotations: mypy uses these instead of nn.Module.__getattr__'s
    # ``Tensor | Module`` union, so ``.view(...)`` / ``/ tensor`` etc. are valid on them.
    per_class_temperature_similarity: torch.Tensor
    per_class_bias_similarity: torch.Tensor
    per_class_temperature_type: torch.Tensor
    per_class_bias_type: torch.Tensor

    def __init__(
        self,
        camera_dim: int,
        world_dim: int,
        num_classes: int | list[int] | None = None,
        hidden_dim: int = 768,
        num_layers: int = 4,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        cross_attn_frequency: int | None = 1,
        ls_init_value: float | None = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        use_abs_posemb: bool = False,
        use_rope2d: bool = True,
        input_resize_factor: int | None = None,
        grad_checkpointing: bool = False,
        patch_classification_type: list | None = None,
        treat_wrong_type_as_match: bool = False,
        object_type_class_mapping: type[ObjectTypeClassMapping] | None = None,
        use_tensor_parallel: bool = True,
        per_class_temperature_similarity: list[float] | None = None,
        per_class_bias_similarity: list[float] | None = None,
        per_class_temperature_type: list[float] | None = None,
        per_class_bias_type: list[float] | None = None,
        excluded_model_classes_for_argmax_similarity: list[int] | None = None,
        excluded_model_classes_for_argmax_type: list[int] | None = None,
        use_separate_necks: bool = False,
        neck_type_config: dict | None = None,
        neck_similarity_config: dict | None = None,
        use_deformable_attn: bool = False,
        use_deformable_attn_type: bool = False,
        deformable_num_sampling_points: int = 4,
        deformable_query_upsample_method: attention_module.QueryUpsampleMethod | str = (
            attention_module.QueryUpsampleMethod.BILINEAR
        ),
        deformable_query_upsample_kernel_size: int = 3,
        use_legacy_transformer: bool = False,
        fused_kv_camera_world_cross_attn: bool = False,
        pre_upscale_when_deformable_off: bool = False,
        use_separate_projs: bool = False,
        **kwargs: Any,  # Accept extra params from Hydra for compatibility with PatchFormer config
    ):
        """
        Initialize ParallelPatchFormer.

        This signature matches PatchFormer for seamless config swap.

        Args:
            camera_dim: Input dimension for camera features.
            world_dim: Input dimension for world features.
            num_classes: Number of output classes. Can be a single int or list [type_classes, similarity_classes].
                If None, determined automatically from object_type_class_mapping.
            hidden_dim: Hidden dimension for transformer layers.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads. Must be divisible by TP size when using TP.
            mlp_ratio: Ratio of MLP hidden dim to hidden_dim.
            cross_attn_frequency: Chooses which neck layers are cross-attention (camera queries, world or fused memory)
                versus self-attention (camera tokens only). With integer ``N >= 0``, layer index ``i`` is
                cross-attention when ``(i + 1) % (N + 1) == 0``.
                If ``None``, every block is self-attention and there are no cross-attention layers (so
                ``fused_kv_camera_world_cross_attn`` never applies). If ``N == 0``, the modulus is ``1``, so every layer
                is cross-attention; with ``fused_kv_camera_world_cross_attn=True``, every layer is then fused-style
                cross-attention only, with no pure self-attention layers. The common default ``N == 1`` alternates
                self-attention and cross-attention.
            ls_init_value: Initial value for LayerScale. None disables LayerScale.
            act_layer: Activation layer class for MLP.
            norm_layer: Normalization layer class.
            drop_path: Drop path probability for stochastic depth.
            use_abs_posemb: If True, use learnable absolute positional embeddings.
            use_rope2d: If True, use 2D Rotary Position Embeddings.
            input_resize_factor: Factor to resize input features. None or 1 means no resize.
            grad_checkpointing: If True, enable gradient checkpointing to save memory.
            patch_classification_type: List of PatchClassificationType values specifying which classification heads to
                use.
            treat_wrong_type_as_match: If True, treat wrong type predictions as matches.
            object_type_class_mapping: Class mapping for object types (determines num_classes if not specified).
            use_tensor_parallel: If True, enable Tensor Parallelism.
            per_class_temperature_similarity: Per-class temperature scaling for similarity logits.
            per_class_bias_similarity: Per-class bias for similarity logits.
            per_class_temperature_type: Per-class temperature scaling for type logits.
            per_class_bias_type: Per-class bias for type logits.
            excluded_model_classes_for_argmax_similarity: List of similarity model class indices to exclude from argmax.
            excluded_model_classes_for_argmax_type: List of type model class indices to exclude from argmax.
                When computing predictions, these classes will be masked out (set to -inf) before argmax. This is useful
                for excluding classes like TRAFFIC_LIGHT_UNKNOWN from predictions. Can also be achieved by setting a
                very negative per_class_bias for those classes.
            use_separate_necks: If True, use separate transformer necks for type and similarity heads. Each head gets
                its own transformer layers. By default necks use the same configuration as the main transformer
                (num_layers, num_heads, etc.). Use neck_type_config / neck_similarity_config to override per neck.
                When False, a single shared transformer is used for both heads.
            neck_type_config: Optional dict of transformer kwargs to override for the type neck. Only used when
                use_separate_necks=True. Keys can include num_layers, num_heads, cross_attn_frequency, mlp_ratio,
                num_heads, cross_attn_frequency, mlp_ratio, fused_kv_camera_world_cross_attn, etc.
            neck_similarity_config: Optional dict of transformer kwargs to override for the similarity neck. Only used
                when use_separate_necks=True. Keys can include fused_kv_camera_world_cross_attn, num_layers, num_heads,
                cross_attn_frequency, etc.
            use_deformable_attn: If True, use deformable attention as the final layer. Deformable attention is more
                memory-efficient for high-resolution outputs as it samples only a sparse set of positions instead of
                full attention. Gates the shared / similarity-side block; use ``use_deformable_attn_type`` to
                independently enable a dedicated type-side block.
            use_deformable_attn_type: If True, build a dedicated type-side deformable block
                (``deformable_attn_block_type`` + ``deformable_norm_type``) so the type head uses learned deformable
                upsampling from ``(H_proc, W_proc)`` to ``(H_out, W_out)`` instead of bilinear interpolation on the
                logits. Independent of ``use_deformable_attn`` -- can be enabled on type only. Only meaningful with
                ``use_separate_necks=True`` (ignored with a warning otherwise, since the shared block already serves
                both heads). Default ``False`` so checkpoints without the type-side weights load unchanged.
            deformable_num_sampling_points: Number of sampling points per head for deformable attention. More
            points = more capacity but more compute. Default 4.
            deformable_query_upsample_method: Method for upsampling queries in deformable attention.
                ``QueryUpsampleMethod.SUBPIXEL_CONV``: learnable sub-pixel conv (default).
                ``QueryUpsampleMethod.BILINEAR``: bilinear interpolation.
            deformable_query_upsample_kernel_size: Kernel size for subpixel conv. Default 3.
                Note: kernel_size=1 is equivalent to Patch Expand (Swin-style).
            use_legacy_transformer: Deprecated and ignored. We always use MixedAttentionTransformer (TP format). Legacy
                checkpoints (combined kv_proj or in_proj) are converted on load by the checkpointer (dcp.py).
            fused_kv_camera_world_cross_attn: Use :class:`FusedKVCameraWorldCrossAttention` on cross-attention layers
                when True (ignored when ``cross_attn_frequency`` is ``None``, since there are no cross-attention
                layers). With ``use_separate_necks=False``, this applies to the shared neck. With
                ``use_separate_necks=True``, it is the default for both necks; set ``fused_kv_camera_world_cross_attn``
                inside ``neck_type_config`` or ``neck_similarity_config`` to override per neck.
            pre_upscale_when_deformable_off: When ``input_resize_factor > 1`` and the branch running on a given input
                has no deformable block to handle upsampling, choose between two equivalent-output strategies:

                * ``True``: pre-upscale the input bilinearly so the transformer / neck runs at ``(H_out, W_out)``; heads
                  then emit logits directly at output resolution. Heavier compute (transformer sees ``factor**2`` more
                  tokens) but the transformer attends over high-resolution spatial structure.
                * ``False`` (default): keep the input at ``(H_in, W_in)``, run the transformer at low resolution,
                  bilinear-upsample logits / features after the head. Cheaper compute; numerically differs because the
                  attention pattern was computed at lower resolution.

                Only applies when neither ``use_deformable_attn`` nor a branch's deformable block is active for the run
                (with ``use_deformable_attn=True`` the input is never pre-upscaled regardless). Existing v0.4.x
                checkpoints all set ``use_deformable_attn=True``, so the default ``False`` is a no-op for them; set
                ``True`` only when porting a non-deformable, ``input_resize_factor>1`` checkpoint whose transformer was
                trained on pre-upscaled features.
            use_separate_projs: If True, build per-branch input projections
                ``camera_proj_{type,similarity}`` / ``world_proj_{type,similarity}`` instead of the shared
                ``camera_proj`` / ``world_proj``, so each branch owns its full forward path (projections + neck + norm +
                deformable + head). Requires ``use_separate_necks=True``; the abs pos-emb add (when
                ``use_abs_posemb=True``) is deferred into each branch's sub-forward so it sees the post-projection
                ``hidden_dim`` seq.
            **kwargs: Additional keyword arguments for compatibility with PatchFormer config.

        Note:
            Branch-level freezing (``freeze_similarity_branch`` / ``freeze_type_branch``) is a training-only concern (it
            mutates ``requires_grad`` and calls ``.eval()`` on submodules, both of which are no-ops at inference).

        """
        super().__init__()

        # Handle mutable defaults (match PatchFormer behavior)
        if patch_classification_type is None:
            patch_classification_type = [PatchClassificationType.PATCH_TYPE, PatchClassificationType.PATCH_SIMILARITY]

        # Store object_type_class_mapping for type classification (match PatchFormer)
        # Note: object_type_class_mapping should already be a class (not a string) when passed from cwip.py
        self.object_type_class_mapping = object_type_class_mapping

        # Determine num_classes (match PatchFormer behavior)
        if num_classes is None:
            if object_type_class_mapping is not None:
                type_num_classes = object_type_class_mapping.get_num_classes()
            else:
                type_num_classes = 22  # Default fallback
            similarity_num_classes = len(world_model_utils.PatchSimilarityType)
            num_classes = [type_num_classes, similarity_num_classes]

        self.camera_dim = camera_dim
        self.world_dim = world_dim
        self.num_classes = [num_classes] if isinstance(num_classes, int) else num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_tensor_parallel = use_tensor_parallel
        self.input_resize_factor = max(int(input_resize_factor), 1) if input_resize_factor else 1
        self.use_abs_posemb = use_abs_posemb
        self.treat_wrong_type_as_match = treat_wrong_type_as_match
        self.patch_classification_type = (
            [patch_classification_type]
            if not isinstance(patch_classification_type, list)
            else patch_classification_type
        )
        self.use_separate_necks = use_separate_necks
        self.use_legacy_transformer = use_legacy_transformer
        self.fused_kv_camera_world_cross_attn = fused_kv_camera_world_cross_attn
        self.use_separate_projs = use_separate_projs

        if use_separate_projs and not use_separate_necks:
            raise ValueError(
                "use_separate_projs=True requires use_separate_necks=True. The shared-neck "
                "path has no per-branch fork, so per-branch input projections have no "
                "consumer. Set use_separate_necks=True or leave use_separate_projs=False."
            )

        tp_size = parallelism_module.get_tensor_parallel_world_size() if use_tensor_parallel else 1
        cp_size = parallelism_module.get_true_context_parallel_world_size()

        if cp_size > 1:
            cp_note = f"True CP world size={cp_size} (patch tokens sharded; gather attention in necks)"
        else:
            cp_note = "True CP=1"
        msg = (
            f"ParallelPatchFormer: hidden_dim={hidden_dim}, num_layers={num_layers}, "
            f"num_heads={num_heads}, TP={tp_size}, {cp_note}"
        )
        logger.info(msg)
        print(msg)

        # Input projections. With ``use_separate_projs=True`` we build a pair per branch
        # (``camera_proj_{type,similarity}`` / ``world_proj_{type,similarity}``) so each branch fully owns its forward
        # path; otherwise we keep the legacy shared pair. See the ``use_separate_projs`` docstring entry above for
        # rationale.
        def _build_input_proj(in_dim: int, out_dim: int) -> nn.Module:
            if tp_size > 1:
                # Column-parallel with ``gather_output=True`` -> output is full
                # ``hidden_dim`` on every TP rank; downstream code is unchanged.
                return parallelism_module.TPColumnParallelLinear(in_dim, out_dim, gather_output=True)
            return nn.Linear(in_dim, out_dim)

        # Declare all projection attributes as Optional so both branches of the
        # if/else below are compatible regardless of which branch assigns None.
        self.camera_proj: nn.Module | None = None
        self.world_proj: nn.Module | None = None
        self.camera_proj_type: nn.Module | None = None
        self.camera_proj_similarity: nn.Module | None = None
        self.world_proj_type: nn.Module | None = None
        self.world_proj_similarity: nn.Module | None = None

        if use_separate_projs:
            self.camera_proj_type = _build_input_proj(camera_dim, hidden_dim)
            self.camera_proj_similarity = _build_input_proj(camera_dim, hidden_dim)
            self.world_proj_type = _build_input_proj(world_dim, hidden_dim)
            self.world_proj_similarity = _build_input_proj(world_dim, hidden_dim)
        else:
            self.camera_proj = _build_input_proj(camera_dim, hidden_dim)
            self.world_proj = _build_input_proj(world_dim, hidden_dim)

        # RoPE 2D
        self.rope = Rope2D(dim=hidden_dim // num_heads) if use_rope2d else None
        if self.rope is not None:
            self.rope.init_tensors()

        # Absolute positional encoding (learnable) - match PatchFormer
        self.posemb_grid_size = 32
        if use_abs_posemb:
            init_scale = hidden_dim**-0.5
            self.positional_embedding = nn.Parameter(init_scale * torch.randn(self.posemb_grid_size**2, hidden_dim))

        # Transformer config for _create_transformer calls.
        # Typed as dict[str, Any] so the **-unpack into _create_transformer (which has
        # individually-typed parameters) doesn't produce arg-type errors for every key.
        transformer_kwargs: dict[str, Any] = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "ls_init_value": ls_init_value,
            "act_layer": act_layer,
            "norm_layer": norm_layer,
            "drop_path": drop_path,
            "cross_attn_frequency": cross_attn_frequency,
            "fused_kv_camera_world_cross_attn": fused_kv_camera_world_cross_attn,
        }

        # Declare transformer/neck/norm attributes as Optional so both branches of the
        # if/else below are compatible regardless of which branch assigns None.
        self.transformer: MixedAttentionTransformer | None = None
        self.norm: nn.Module | None = None
        self.neck_type: MixedAttentionTransformer | None = None
        self.neck_similarity: MixedAttentionTransformer | None = None

        # Create transformers
        # When use_separate_necks=True: create separate necks, transformer=None
        # When use_separate_necks=False: create shared transformer, necks=None
        if use_separate_necks:
            # Build per-neck kwargs: start with shared defaults, override with per-neck config
            type_kwargs = {**transformer_kwargs, "cross_attn_frequency": None}
            if neck_type_config:
                type_kwargs.update(neck_type_config)
            sim_kwargs = {**transformer_kwargs}
            if neck_similarity_config:
                sim_kwargs.update(neck_similarity_config)

            logger.info(
                "ParallelPatchFormer: Using separate necks — "
                "type: num_layers=%s, cross_attn_frequency=%s, fused_kv_camera_world_cross_attn=%s | "
                "similarity: num_layers=%s, cross_attn_frequency=%s, fused_kv_camera_world_cross_attn=%s",
                type_kwargs["num_layers"],
                type_kwargs.get("cross_attn_frequency"),
                type_kwargs.get("fused_kv_camera_world_cross_attn"),
                sim_kwargs["num_layers"],
                sim_kwargs.get("cross_attn_frequency"),
                sim_kwargs.get("fused_kv_camera_world_cross_attn"),
            )
            self.neck_type = self._create_transformer(**type_kwargs)
            self.neck_similarity = self._create_transformer(**sim_kwargs)
            self._setup_transformer(self.neck_type, cp_size, grad_checkpointing)
            self._setup_transformer(self.neck_similarity, cp_size, grad_checkpointing)

            # Separate norms for each neck
            self.norm_type = norm_layer(hidden_dim)
            self.norm_similarity = norm_layer(hidden_dim)

            # No shared transformer when using separate necks
            self.transformer = None
            self.norm = None
        else:
            # Create shared transformer
            self.transformer = self._create_transformer(**transformer_kwargs)
            self._setup_transformer(self.transformer, cp_size, grad_checkpointing)
            self.norm = norm_layer(hidden_dim)

            # No separate necks when using shared transformer
            self.neck_type = None
            self.neck_similarity = None
            self.norm_type = None
            self.norm_similarity = None

        # Deformable attention final layer (optional, for high-resolution output).
        # ``use_deformable_attn`` (shared / similarity-side block) and
        # ``use_deformable_attn_type`` (dedicated type-side block, requires separate
        # necks) are independent gates.
        self.use_deformable_attn = use_deformable_attn
        self.use_deformable_attn_type = use_deformable_attn_type
        self.pre_upscale_when_deformable_off = pre_upscale_when_deformable_off

        # Without separate necks the shared block already serves both heads, so
        # ``use_deformable_attn_type=True`` has no pathway -- warn and no-op.
        if use_deformable_attn_type and not use_separate_necks:
            logger.warning(
                "ParallelPatchFormer: use_deformable_attn_type=True is being ignored "
                "because use_separate_necks=False -- the shared deformable_attn_block "
                "(gated by use_deformable_attn) already serves both heads in this "
                "configuration. Set use_separate_necks=True to enable a dedicated "
                "type-side deformable block."
            )

        def _build_deformable_block() -> attention_module.DeformableAttentionBlock:
            return attention_module.DeformableAttentionBlock(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                num_sampling_points=deformable_num_sampling_points,
                mlp_ratio=mlp_ratio,
                dropout=0.0,
                drop_path=drop_path,
                use_tensor_parallel=use_tensor_parallel,
                grad_checkpointing=grad_checkpointing,
                query_upsample_method=deformable_query_upsample_method,
                query_upsample_scale_factor=self.input_resize_factor,
                query_upsample_kernel_size=deformable_query_upsample_kernel_size,
            )

        if use_deformable_attn or (use_separate_necks and use_deformable_attn_type):
            logger.info(
                "ParallelPatchFormer: Using deformable attention final layer with "
                "num_sampling_points=%s, TP=%s, query_upsample=%s(scale=%s, kernel=%s), "
                "similarity_block=%s, type_block=%s",
                deformable_num_sampling_points,
                use_tensor_parallel,
                deformable_query_upsample_method,
                self.input_resize_factor,
                deformable_query_upsample_kernel_size,
                use_deformable_attn,
                use_separate_necks and use_deformable_attn_type,
            )

        # Declare deformable-block attributes as Optional so both branches are compatible.
        self.deformable_attn_block: DeformableAttentionBlock | None = None
        self.deformable_norm: nn.Module | None = None
        self.deformable_attn_block_type: DeformableAttentionBlock | None = None
        self.deformable_norm_type: nn.Module | None = None

        if use_deformable_attn:
            self.deformable_attn_block = _build_deformable_block()
            self.deformable_norm = norm_layer(hidden_dim)

        if use_separate_necks and use_deformable_attn_type:
            self.deformable_attn_block_type = _build_deformable_block()
            self.deformable_norm_type = norm_layer(hidden_dim)

        # Classification heads: full nn.Linear (same weights replicated on every TP rank).
        # The neck (ResidualAttentionBlock / deformable block) ends each sublayer with row-parallel
        # matmul + all_reduce, so hidden states are full [B, S, hidden_dim] on every rank before
        # these heads (not hidden-dim-sharded); a plain Linear therefore sees the full feature width.
        self.head_type = nn.Linear(hidden_dim, self.num_classes[0]) if len(self.num_classes) > 0 else None
        self.head_similarity = (
            nn.Linear(hidden_dim, self.num_classes[1] if len(self.num_classes) > 1 else self.num_classes[0])
            if len(self.num_classes) > 0
            else None
        )

        # Per-class calibration parameters (for inference)
        # Similarity head calibration
        similarity_num_classes = self.num_classes[1] if len(self.num_classes) > 1 else self.num_classes[0]
        if per_class_temperature_similarity is None:
            per_class_temperature_similarity = [1.0] * similarity_num_classes
        if per_class_bias_similarity is None:
            per_class_bias_similarity = [0.0] * similarity_num_classes
        self.register_buffer(
            "per_class_temperature_similarity", torch.tensor(per_class_temperature_similarity, dtype=torch.float32)
        )
        self.register_buffer("per_class_bias_similarity", torch.tensor(per_class_bias_similarity, dtype=torch.float32))

        # Type head calibration
        type_num_classes = self.num_classes[0] if len(self.num_classes) > 0 else 0
        if per_class_temperature_type is None:
            per_class_temperature_type = [1.0] * type_num_classes
        if per_class_bias_type is None:
            per_class_bias_type = [0.0] * type_num_classes
        self.register_buffer(
            "per_class_temperature_type", torch.tensor(per_class_temperature_type, dtype=torch.float32)
        )
        self.register_buffer("per_class_bias_type", torch.tensor(per_class_bias_type, dtype=torch.float32))

        # Store num_classes for property setters
        self._similarity_num_classes = similarity_num_classes
        self._type_num_classes = type_num_classes

        # Excluded classes for argmax - use property setters to normalize and validate
        self._excluded_model_classes_for_argmax_similarity: set[int] = set()
        self._excluded_model_classes_for_argmax_type: set[int] = set()
        self.excluded_model_classes_for_argmax_similarity = excluded_model_classes_for_argmax_similarity
        self.excluded_model_classes_for_argmax_type = excluded_model_classes_for_argmax_type

    def _create_transformer(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float,
        ls_init_value: float | None,
        act_layer: Callable,
        norm_layer: Callable,
        drop_path: float,
        cross_attn_frequency: int | None,
        fused_kv_camera_world_cross_attn: bool = False,
    ) -> MixedAttentionTransformer:
        """
        Create a transformer module (TP or non-TP) based on configuration.

        Args:
            hidden_dim: Hidden dimension for transformer layers.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of MLP hidden dim to hidden_dim.
            ls_init_value: Initial value for LayerScale. None disables LayerScale.
            act_layer: Activation layer class for MLP.
            norm_layer: Normalization layer class.
            drop_path: Drop path probability for stochastic depth.
            cross_attn_frequency: Frequency of cross-attention layers.
            fused_kv_camera_world_cross_attn: Use fused camera+world K/V cross-attention.

        Returns:
            MixedAttentionTransformer (TP version). Legacy checkpoints (kv_proj or in_proj format)
            are converted on load by the checkpointer (dcp.py) to separate q_proj, k_proj, v_proj.
        """
        # Always use TP MixedAttentionTransformer. Legacy checkpoints are converted on load by
        # the checkpointer (dcp.py) to separate q_proj, k_proj, v_proj. Legacy formats:
        # - in_proj: single combined Q+K+V projection (in_proj_weight/bias, shape 3*embed_dim);
        # - kv_proj: separate q_proj plus combined kv_proj (q_proj + kv_proj_weight/bias).
        return MixedAttentionTransformer(
            width=hidden_dim,
            layers=num_layers,
            heads=num_heads,
            q_dim=hidden_dim,
            kv_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path=drop_path,
            rope=self.rope,
            cross_attn_frequency=cross_attn_frequency,
            use_tensor_parallel=self.use_tensor_parallel,
            fused_kv_camera_world_cross_attn=fused_kv_camera_world_cross_attn,
        )

    def _setup_transformer(
        self,
        transformer: MixedAttentionTransformer,
        cp_size: int,
        grad_checkpointing: bool,
    ) -> None:
        """
        Set up context parallelism and gradient checkpointing for a transformer.

        When true CP > 1, :meth:`forward` shards patch tokens per rank and wires the true-CP process group into
        attention so :func:`~parallelism.sdpa_or_true_cp_attention` runs gather + SDPA over K/V shards.

        Args:
            transformer: The transformer module to configure.
            cp_size: True context parallel world size.
            grad_checkpointing: Whether to enable gradient checkpointing.
        """
        if cp_size > 1:
            cp_group = parallelism_module.get_true_context_parallel_group()
            if cp_group is not None:
                transformer.set_context_parallel_group(cp_group)
        if grad_checkpointing:
            transformer.set_grad_checkpointing(True)

    @property
    def total_spatial_sampling_factor(self) -> int:
        """
        Total spatial sampling factor applied to input features.

        Returns:
            The input resize factor (1 if no resizing is applied).
        """
        return self.input_resize_factor

    @property
    def excluded_model_classes_for_argmax_similarity(self) -> set[int]:
        """
        Get the set of excluded similarity model class indices for argmax.

        Returns:
            Set of similarity class indices that are excluded from argmax operations.
        """
        return self._excluded_model_classes_for_argmax_similarity

    @excluded_model_classes_for_argmax_similarity.setter
    def excluded_model_classes_for_argmax_similarity(self, value: Any) -> None:
        """
        Set excluded similarity model class indices for argmax with normalization and validation.

        Args:
            value: Input value that can be None, a string, list, tuple, or other iterable.
                  Will be normalized and validated before being stored as a set.

        Raises:
            ValueError: If any class index is outside the valid range [0, similarity_num_classes).
        """
        self._set_excluded_classes_for_argmax(value, "similarity")

    @property
    def excluded_model_classes_for_argmax_type(self) -> set[int]:
        """
        Get the set of excluded type model class indices for argmax.

        Returns:
            Set of type class indices that are excluded from argmax operations.
        """
        return self._excluded_model_classes_for_argmax_type

    @excluded_model_classes_for_argmax_type.setter
    def excluded_model_classes_for_argmax_type(self, value: Any) -> None:
        """
        Set excluded type model class indices for argmax with normalization and validation.

        Args:
            value: Input value that can be None, a string, list, tuple, or other iterable.
                  Will be normalized and validated before being stored as a set.

        Raises:
            ValueError: If any class index is outside the valid range [0, type_num_classes).
        """
        self._set_excluded_classes_for_argmax(value, "type")

    def _set_excluded_classes_for_argmax(
        self,
        value: Any,
        logit_type: str,
    ) -> None:
        """
        Set excluded model class indices for argmax with normalization and validation.

        This is a helper method used by both property setters to avoid code duplication.

        Args:
            value: Input value that can be None, a string, list, tuple, or other iterable.
                  Will be normalized and validated before being stored as a set.
            logit_type: "similarity" or "type" to determine which attribute to set and which num_classes to use.

        Raises:
            ValueError: If any class index is outside the valid range [0, num_classes).
        """
        # Normalize from config (handles both string and list inputs)
        normalized = patch_vis_utils.normalize_excluded_classes_for_argmax(value)
        if normalized is None:
            normalized = []

        # Get the appropriate num_classes based on logit_type
        if logit_type == "similarity":
            num_classes = self._similarity_num_classes
            attr_name = "_excluded_model_classes_for_argmax_similarity"
        else:
            num_classes = self._type_num_classes
            attr_name = "_excluded_model_classes_for_argmax_type"

        # Validate excluded classes are within valid range
        for class_idx in normalized:
            if not (0 <= class_idx < num_classes):
                raise ValueError(
                    f"excluded_model_classes_for_argmax_{logit_type} contains invalid class index {class_idx}. "
                    f"Must be in range [0, {num_classes})"
                )

        setattr(self, attr_name, set(normalized))

    def _sample_abs_posemb(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """
        Sample/interpolate absolute positional embedding to target grid size.

        Args:
            grid_h: Target grid height
            grid_w: Target grid width

        Returns:
            Positional embedding of shape (1, grid_h * grid_w, hidden_dim)
        """
        if self.posemb_grid_size == grid_h and self.posemb_grid_size == grid_w:
            return self.positional_embedding[None, ...]

        pos_embed: torch.Tensor = self.positional_embedding

        # Reshape to 2D grid: (grid_size^2, hidden_dim) -> (1, grid_size, grid_size, hidden_dim)
        pos_embed = (
            pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1).permute(0, 3, 1, 2).contiguous()
        )

        # Interpolate to target size
        pos_embed = F.interpolate(
            pos_embed,
            size=(grid_h, grid_w),
            mode="bicubic",
            align_corners=False,
        )

        # Reshape back to sequence: (1, hidden_dim, h, w) -> (1, h*w, hidden_dim)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)

        return pos_embed

    def apply_bias(self, logits: torch.Tensor, logit_type: str = "similarity") -> torch.Tensor:
        """
        Apply per-class logit bias to shift decision boundaries.

        Args:
            logits: Logits tensor of shape (B, num_classes, H, W)
            logit_type: "similarity" or "type"

        Returns:
            Bias-adjusted logits of shape (B, num_classes, H, W)
        """
        if logit_type == "similarity":
            bias = self.per_class_bias_similarity.view(1, -1, 1, 1)
        else:
            bias = self.per_class_bias_type.view(1, -1, 1, 1)
        return logits + bias

    def apply_temperature_scaling(self, logits: torch.Tensor, logit_type: str = "similarity") -> torch.Tensor:
        """
        Apply per-class temperature scaling to logits.

        Args:
            logits: Logits tensor of shape (B, num_classes, H, W)
            logit_type: "similarity" or "type"

        Returns:
            Temperature-scaled logits of shape (B, num_classes, H, W)
        """
        if logit_type == "similarity":
            temperature = self.per_class_temperature_similarity.view(1, -1, 1, 1)
        else:
            temperature = self.per_class_temperature_type.view(1, -1, 1, 1)
        return logits / temperature

    def _emit_logits_and_features(
        self,
        *,
        out: torch.Tensor,
        head: nn.Linear | None,
        H_seq: int,
        W_seq: int,
        H_out: int,
        W_out: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        Unified post-neck output for both heads: rearrange ``out`` (token sequence) to ``BCHW`` at ``(H_seq, W_seq)``
        and bilinear-upsample to ``(H_out, W_out)`` when the two differ. Used by every branch (type, similarity, shared)
        so the four ``(deformable on/off) x (pre-upscale on/off)`` combinations all go through the same post-head logic.

        Args:
            out: Token-sequence features ``(B, S, hidden_dim)`` where ``S = H_seq * W_seq``. After an upstream
                deformable block, ``S = H_out * W_out``; otherwise ``S = H_proc * W_proc``.
            head: Classification head, or ``None`` to skip logits and emit only features.
            H_seq, W_seq: Spatial shape of ``out`` interpreted as a 2-D grid.
            H_out, W_out: Target output resolution; bilinear upsampling kicks in when
                ``(H_seq, W_seq) != (H_out, W_out)``.

        Returns:
            ``(logits, features)``, both at ``(B, ?, H_out, W_out)``. ``logits`` is ``None`` when ``head`` is ``None``.
        """
        needs_upsample = (H_seq, W_seq) != (H_out, W_out)

        logits = head(out) if head is not None else None
        if logits is not None:
            logits = rearrange(logits, "b (h w) c -> b c h w", h=H_seq, w=W_seq)
            if needs_upsample:
                logits = F.interpolate(logits, size=[H_out, W_out], mode="bilinear", align_corners=True)

        features = rearrange(out, "b (h w) c -> b c h w", h=H_seq, w=W_seq)
        if needs_upsample:
            features = F.interpolate(features, size=[H_out, W_out], mode="bilinear", align_corners=True)

        return logits, features

    def _apply_branch_posemb(
        self,
        camera_seq: torch.Tensor,
        world_seq: torch.Tensor | None,
        world_seq_full: torch.Tensor | None,
        H_proc: int,
        W_proc: int,
        cp_meta: dict | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Apply abs pos-emb inside a branch sub-forward (``use_separate_projs=True``).

        ``forward()`` skips the pos-emb add for the split-proj path because the seqs there are still at raw
        ``camera_dim`` / ``world_dim``. After the per-branch projection inside the sub-forward they're at ``hidden_dim``
        and match ``positional_embedding``'s channel dim, so the add runs here: the sharded inputs receive a sliced
        pos-emb (via ``cp_meta``) and the gathered ``world_seq_full`` receives the full-grid pos-emb. This is
        numerically equivalent to the non-split path's pre-shard add (the gather of pos-emb-laden sharded tensors ==
        full-grid pos-emb on the gathered tensor, because pos-emb is content-independent).
        """
        pos_embed = self._sample_abs_posemb(H_proc, W_proc)  # (1, H*W, hidden_dim)
        if cp_meta is not None:
            s_lo, s_hi = cp_meta["s_lo"], cp_meta["s_hi"]
            pos_embed_shard = pos_embed[:, s_lo:s_hi, :]
        else:
            pos_embed_shard = pos_embed
        camera_seq = camera_seq + pos_embed_shard
        if world_seq is not None:
            world_seq = world_seq + pos_embed_shard
        if world_seq_full is not None:
            world_seq_full = world_seq_full + pos_embed
        return camera_seq, world_seq, world_seq_full

    def _forward_type_branch_separate(
        self,
        *,
        camera_seq: torch.Tensor,
        world_seq: torch.Tensor | None,
        world_seq_full: torch.Tensor | None,
        cp_meta: dict | None,
        H_proc: int,
        W_proc: int,
        H_out: int,
        W_out: int,
        output_spatial_shape: tuple[int, int] | None,
        type_neck_uses_cross_attn: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        Type sub-pass when ``use_separate_necks=True``: ``neck_type -> norm_type -> (optional type-side deformable
        block) -> head_type``. Extracted so the training subclass can wrap the entire sub-pass in ``torch.no_grad()``
        when the type branch is frozen (avoids retaining activations for a backward that will never run).

        With ``use_separate_projs=True`` the inbound ``camera_seq`` / ``world_seq`` (and gathered ``world_seq_full``)
        are still at the raw ``camera_dim`` / ``world_dim`` -- this method applies ``camera_proj_type`` /
        ``world_proj_type`` to them before the neck.
        """
        if self.use_separate_projs:
            assert self.camera_proj_type is not None
            assert self.world_proj_type is not None
            camera_seq = self.camera_proj_type(camera_seq)
            if world_seq is not None:
                world_seq = self.world_proj_type(world_seq)
                if world_seq_full is not None:
                    world_seq_full = self.world_proj_type(world_seq_full)
            if self.use_abs_posemb and hasattr(self, "positional_embedding"):
                camera_seq, world_seq, world_seq_full = self._apply_branch_posemb(
                    camera_seq, world_seq, world_seq_full, H_proc, W_proc, cp_meta
                )
        assert self.neck_type is not None
        type_context = world_seq if (type_neck_uses_cross_attn and world_seq is not None) else None
        out_type = self.neck_type(
            x=camera_seq,
            context=type_context,
        )
        out_type = self.norm_type(out_type)
        out_type = parallelism_module.gather_trim_true_cp_sequence(out_type, cp_meta)

        if self.deformable_attn_block_type is not None:
            # Dedicated type-side deformable block: learned upsample from (H_proc, W_proc)
            # to (H_out, W_out). ``context=None`` keeps the block self-attention when the
            # type neck is self-attention only. After this, ``out_type`` is at (H_out, W_out).
            out_type = self.deformable_attn_block_type(
                x=out_type,
                context=world_seq_full if (type_neck_uses_cross_attn and world_seq_full is not None) else None,
                input_spatial_shape=(H_proc, W_proc),
                output_spatial_shape=output_spatial_shape,
            )
            assert self.deformable_norm_type is not None
            out_type = self.deformable_norm_type(out_type)
            H_seq, W_seq = H_out, W_out
        else:
            H_seq, W_seq = H_proc, W_proc

        return self._emit_logits_and_features(
            out=out_type, head=self.head_type, H_seq=H_seq, W_seq=W_seq, H_out=H_out, W_out=W_out
        )

    def _forward_similarity_branch_separate(
        self,
        *,
        camera_seq: torch.Tensor,
        world_seq: torch.Tensor,
        world_seq_full: torch.Tensor | None,
        cp_meta: dict | None,
        H_proc: int,
        W_proc: int,
        H_out: int,
        W_out: int,
        output_spatial_shape: tuple[int, int] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """
        Similarity sub-pass when ``use_separate_necks=True``: ``neck_similarity -> norm_similarity -> (optional
        similarity-side deformable block) -> head_similarity``. Extracted for the same reason as
        ``_forward_type_branch_separate`` -- lets the training subclass wrap this in ``torch.no_grad()`` when the
        similarity branch is frozen.

        With ``use_separate_projs=True`` the inbound ``camera_seq`` / ``world_seq`` (and gathered ``world_seq_full``)
        are still at the raw ``camera_dim`` / ``world_dim`` -- this method applies ``camera_proj_similarity`` /
        ``world_proj_similarity`` to them before the neck.
        """
        if self.use_separate_projs:
            assert self.camera_proj_similarity is not None
            assert self.world_proj_similarity is not None
            camera_seq = self.camera_proj_similarity(camera_seq)
            world_seq = self.world_proj_similarity(world_seq)
            if world_seq_full is not None:
                world_seq_full = self.world_proj_similarity(world_seq_full)
            if self.use_abs_posemb and hasattr(self, "positional_embedding"):
                camera_seq, _world_seq_opt, world_seq_full = self._apply_branch_posemb(
                    camera_seq, world_seq, world_seq_full, H_proc, W_proc, cp_meta
                )
                # _apply_branch_posemb returns Tensor | None for world_seq, but the similarity branch
                # is never called when world_seq is None, so the result is always non-None here.
                assert _world_seq_opt is not None
                world_seq = _world_seq_opt
        assert self.neck_similarity is not None
        out_similarity = self.neck_similarity(
            x=camera_seq,
            context=world_seq if self.neck_similarity.cross_attn_frequency is not None else None,
        )
        out_similarity = self.norm_similarity(out_similarity)
        out_similarity = parallelism_module.gather_trim_true_cp_sequence(out_similarity, cp_meta)

        if self.use_deformable_attn and self.deformable_attn_block is not None:
            # Similarity-side deformable block: learned upsample from (H_proc, W_proc)
            # to (H_out, W_out). After this, ``out_similarity`` is at (H_out, W_out).
            out_similarity = self.deformable_attn_block(
                x=out_similarity,
                context=world_seq_full,
                input_spatial_shape=(H_proc, W_proc),
                output_spatial_shape=output_spatial_shape,
            )
            assert self.deformable_norm is not None
            out_similarity = self.deformable_norm(out_similarity)
            H_seq, W_seq = H_out, W_out
        else:
            H_seq, W_seq = H_proc, W_proc

        return self._emit_logits_and_features(
            out=out_similarity, head=self.head_similarity, H_seq=H_seq, W_seq=W_seq, H_out=H_out, W_out=W_out
        )

    def _forward_shared(
        self,
        *,
        camera_seq: torch.Tensor,
        world_seq: torch.Tensor | None,
        cp_meta: dict | None,
        H_proc: int,
        W_proc: int,
        H_out: int,
        W_out: int,
        output_spatial_shape: tuple[int, int] | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """
        Shared-transformer sub-pass when ``use_separate_necks=False``: a single transformer feeds both ``head_type`` and
        ``head_similarity``. Not split per-branch because the backbone activations are reused across both heads;
        freezing one head doesn't free any activations from the shared path.
        """
        assert self.transformer is not None
        assert self.norm is not None
        shared_context = (
            world_seq if (self.transformer.cross_attn_frequency is not None and world_seq is not None) else None
        )
        out_shared = self.transformer(
            x=camera_seq,
            context=shared_context,
        )
        out_shared = self.norm(out_shared)
        out_shared = parallelism_module.gather_trim_true_cp_sequence(out_shared, cp_meta)
        world_seq_full = (
            parallelism_module.gather_trim_true_cp_sequence(world_seq, cp_meta) if world_seq is not None else None
        )

        if self.use_deformable_attn and self.deformable_attn_block is not None:
            # Shared deformable block: learned upsample from (H_proc, W_proc) to (H_out, W_out). After this,
            # ``out_shared`` is at (H_out, W_out).
            out_shared = self.deformable_attn_block(
                x=out_shared,
                context=world_seq_full,
                input_spatial_shape=(H_proc, W_proc),
                output_spatial_shape=output_spatial_shape,
            )
            assert self.deformable_norm is not None
            out_shared = self.deformable_norm(out_shared)
            H_seq, W_seq = H_out, W_out
        else:
            H_seq, W_seq = H_proc, W_proc

        # Both heads consume the same shared features. ``camera_patch_features`` is computed once via the type-head call
        # (head=None would still produce the features tensor), but we want both logits explicitly, so call the helper
        # twice -- the rearrange / F.interpolate on ``out_shared`` is cheap relative to the head matmuls.
        patch_type_logits, camera_patch_features = self._emit_logits_and_features(
            out=out_shared, head=self.head_type, H_seq=H_seq, W_seq=W_seq, H_out=H_out, W_out=W_out
        )
        if world_seq is None:
            patch_similarity_logits = None
        else:
            patch_similarity_logits, _ = self._emit_logits_and_features(
                out=out_shared, head=self.head_similarity, H_seq=H_seq, W_seq=W_seq, H_out=H_out, W_out=W_out
            )
        return patch_type_logits, patch_similarity_logits, camera_patch_features

    def forward(
        self,
        camera_features: torch.Tensor,
        world_features: torch.Tensor | None = None,
        apply_bias: bool = False,
        apply_temperature: bool = False,
        apply_excluded_classes: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """
        Forward pass with TP and CP.

        Args:
            camera_features: [B, C_cam, H, W]
            world_features: [B, C_world, H, W], or ``None`` to run the camera-only
                type branch (skip world tokens and the similarity head).
            apply_bias: If True, apply per-class bias calibration
            apply_temperature: If True, apply per-class temperature scaling
            apply_excluded_classes: If True, mask excluded classes in logits (set to -inf)

        Returns:
            Dictionary with logits and features
        """
        _B, _, H_in, W_in = camera_features.shape
        camera_only = world_features is None
        if camera_only:
            active_type_neck = self.neck_type if self.use_separate_necks else self.transformer
            if active_type_neck is not None and active_type_neck.cross_attn_frequency is not None:
                raise ValueError(
                    "Camera-only inference requires a self-attention-only type path (cross_attn_frequency=None)."
                )

        # Compute output resolution
        H_out = H_in * self.input_resize_factor
        W_out = W_in * self.input_resize_factor

        # Upscaling strategy. Two paths produce equivalent output shapes:
        #
        # 1. Pre-upscale here, transformer at (H_out, W_out), plain rearrange after head.
        # 2. Skip pre-upscale, transformer at (H_in, W_in), bilinear-upsample after head.
        #
        # Path 2 is cheaper but numerically different (attention computed at lower resolution). Pre-upscaling is opt-in
        # via ``pre_upscale_when_deformable_off`` and only meaningful when no deformable block will handle the
        # upsampling. Whenever any deformable block exists, the block does the (H_proc, W_proc) -> (H_out, W_out)
        # upsample inside its attention and pre-upscaling here would defeat its purpose, so we always skip in that case
        # regardless of the flag.
        any_deformable_upsamples = self.deformable_attn_block is not None or self.deformable_attn_block_type is not None
        if self.input_resize_factor > 1 and not any_deformable_upsamples and self.pre_upscale_when_deformable_off:
            camera_features = F.interpolate(camera_features, size=[H_out, W_out], mode="bilinear", align_corners=True)
            if not camera_only:
                assert world_features is not None
                world_features = F.interpolate(world_features, size=[H_out, W_out], mode="bilinear", align_corners=True)
            H_proc, W_proc = H_out, W_out  # Processing resolution
        else:
            # Deformable attention handles upscaling, or no upscaling needed
            H_proc, W_proc = H_in, W_in  # Processing resolution

        # Flatten to sequences. With ``use_separate_projs=True`` we defer the ``camera_proj`` / ``world_proj`` to each
        # branch's sub-forward, so ``camera_seq`` here is at ``camera_dim`` (not ``hidden_dim``) and likewise for
        # ``world_seq``. Anything below this point that assumes ``hidden_dim`` is also guarded: the abs pos-emb add is
        # deferred into each branch's sub-forward (post per-branch projection) via :meth:`_apply_branch_posemb`, and
        # RoPE acts inside each neck's attention block (already operating on the projected seq).
        camera_seq = rearrange(camera_features, "b c h w -> b (h w) c")
        world_seq = None if camera_only else rearrange(world_features, "b c h w -> b (h w) c")

        if not self.use_separate_projs:
            assert self.camera_proj is not None
            assert self.world_proj is not None
            camera_seq = self.camera_proj(camera_seq)
            if world_seq is not None:
                world_seq = self.world_proj(world_seq)

        # Add absolute positional encoding if enabled. With ``use_separate_projs=True`` the seqs here are still at raw
        # ``camera_dim`` / ``world_dim`` (the per-branch projection happens inside each branch sub-forward), so the
        # pos-emb add is deferred into the branch sub-forwards via :meth:`_apply_branch_posemb` -- it runs
        # post-projection there, with the ``positional_embedding`` tensor (sized ``hidden_dim``) matched to the
        # projected seq's channel dim.
        if self.use_abs_posemb and hasattr(self, "positional_embedding") and not self.use_separate_projs:
            pos_embed = self._sample_abs_posemb(H_proc, W_proc)  # (1, H*W, hidden_dim)
            camera_seq = camera_seq + pos_embed
            if world_seq is not None:
                world_seq = world_seq + pos_embed

        # Update RoPE for full grid, then optionally narrow to this rank's token slice
        rope_backup = None
        if self.rope is not None:
            self.rope.update_grid(camera_seq.device, H_proc, W_proc)

        cp_meta: dict | None = None
        if camera_only:
            camera_seq, _, cp_meta = parallelism_module.prepare_true_cp_sequence_shard(
                camera_seq, camera_seq, H_proc, W_proc
            )
        else:
            assert world_seq is not None  # camera_only=False guarantees world_seq was set from world_features
            camera_seq, world_seq, cp_meta = parallelism_module.prepare_true_cp_sequence_shard(
                camera_seq, world_seq, H_proc, W_proc
            )
        if cp_meta is not None and self.rope is not None and self.rope.freq is not None:
            rope_backup = self.rope.freq
            s_lo, s_hi = cp_meta["s_lo"], cp_meta["s_hi"]
            self.rope.freq = parallelism_module.slice_rope_freq_along_sequence(rope_backup, s_lo, s_hi)

        # Determine output spatial shape for deformable attention. Either the similarity-side block
        # (``use_deformable_attn``) or the type-side block (``use_deformable_attn_type`` with separate necks) being
        # active is enough to need it.
        any_deformable_runs = self.use_deformable_attn or self.deformable_attn_block_type is not None
        output_spatial_shape = (H_out, W_out) if any_deformable_runs else None

        camera_type_features: torch.Tensor | None = None
        try:
            if self.use_separate_necks:
                # Separate necks for type and similarity heads.
                # Memory optimization: process the type sub-pass first, return its outputs, then call empty_cache before
                # the similarity sub-pass (~20GB peak savings).
                #
                # World tokens are gathered into a single full-sequence tensor when at least one deformable block that
                # consumes context will run:
                #   - similarity-side block: always consumes ``world_seq_full`` (existing behavior).
                #   - type-side block: only when the type neck itself is configured for cross-attn.
                assert self.neck_type is not None
                type_neck_uses_cross_attn = self.neck_type.cross_attn_frequency is not None
                world_seq_full: torch.Tensor | None = None
                if self.use_deformable_attn or (
                    self.deformable_attn_block_type is not None and type_neck_uses_cross_attn
                ):
                    world_seq_full = (
                        parallelism_module.gather_trim_true_cp_sequence(world_seq, cp_meta)
                        if world_seq is not None
                        else None
                    )

                # Branches are factored into sub-methods so the training-side subclass can wrap them in
                # ``torch.no_grad()`` when the corresponding branch is frozen (see ``modules/parallel_patchformer.py``).
                # Keep them as methods (not inline) so the override surface is stable.
                patch_type_logits, camera_type_features = self._forward_type_branch_separate(
                    camera_seq=camera_seq,
                    world_seq=world_seq,
                    world_seq_full=world_seq_full,
                    cp_meta=cp_meta,
                    H_proc=H_proc,
                    W_proc=W_proc,
                    H_out=H_out,
                    W_out=W_out,
                    output_spatial_shape=output_spatial_shape,
                    type_neck_uses_cross_attn=type_neck_uses_cross_attn,
                )
                if camera_only:
                    patch_similarity_logits = None
                    camera_patch_features = None
                else:
                    torch.cuda.empty_cache()
                    assert world_seq is not None  # similarity branch only runs when not camera_only
                    patch_similarity_logits, camera_patch_features = self._forward_similarity_branch_separate(
                        camera_seq=camera_seq,
                        world_seq=world_seq,
                        world_seq_full=world_seq_full,
                        cp_meta=cp_meta,
                        H_proc=H_proc,
                        W_proc=W_proc,
                        H_out=H_out,
                        W_out=W_out,
                        output_spatial_shape=output_spatial_shape,
                    )
            else:
                patch_type_logits, patch_similarity_logits, camera_patch_features = self._forward_shared(
                    camera_seq=camera_seq,
                    world_seq=world_seq,
                    cp_meta=cp_meta,
                    H_proc=H_proc,
                    W_proc=W_proc,
                    H_out=H_out,
                    W_out=W_out,
                    output_spatial_shape=output_spatial_shape,
                )

            # Apply calibration (bias first, then temperature)
            if apply_bias:
                if patch_type_logits is not None:
                    patch_type_logits = self.apply_bias(patch_type_logits, "type")
                if patch_similarity_logits is not None:
                    patch_similarity_logits = self.apply_bias(patch_similarity_logits, "similarity")
            if apply_temperature:
                if patch_type_logits is not None:
                    patch_type_logits = self.apply_temperature_scaling(patch_type_logits, "type")
                if patch_similarity_logits is not None:
                    patch_similarity_logits = self.apply_temperature_scaling(patch_similarity_logits, "similarity")

            if apply_excluded_classes:
                if patch_type_logits is not None:
                    type_num_classes = self.num_classes[0] if len(self.num_classes) > 0 else 0
                    excluded_type = (
                        list(self.excluded_model_classes_for_argmax_type)
                        if self.excluded_model_classes_for_argmax_type
                        else None
                    )
                    patch_type_logits = patch_vis_utils.mask_excluded_classes_for_argmax(
                        logits_or_probs=patch_type_logits,
                        excluded_classes=excluded_type,
                        num_classes=type_num_classes,
                    )
                if patch_similarity_logits is not None:
                    similarity_num_classes = self.num_classes[1] if len(self.num_classes) > 1 else self.num_classes[0]
                    excluded_similarity = (
                        list(self.excluded_model_classes_for_argmax_similarity)
                        if self.excluded_model_classes_for_argmax_similarity
                        else None
                    )
                    patch_similarity_logits = patch_vis_utils.mask_excluded_classes_for_argmax(
                        logits_or_probs=patch_similarity_logits,
                        excluded_classes=excluded_similarity,
                        num_classes=similarity_num_classes,
                    )

            result = {"camera_patch_features": camera_patch_features}
            if self.use_separate_necks:
                assert camera_type_features is not None
                result["camera_type_features"] = camera_type_features
            if patch_type_logits is not None:
                result["patch_type_logits"] = patch_type_logits
            if patch_similarity_logits is not None:
                result["patch_similarity_logits"] = patch_similarity_logits
        finally:
            if rope_backup is not None and self.rope is not None:
                self.rope.freq = rope_backup

        return result
