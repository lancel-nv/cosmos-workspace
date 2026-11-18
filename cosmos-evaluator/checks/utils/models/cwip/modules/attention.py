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
Attention and residual blocks for Cosmos Evaluator deployment.

**Tensor-parallel / CP (PatchFormer):** :class:`CrossAttention`, :class:`FusedKVCameraWorldCrossAttention`,
:class:`SelfAttention`, :class:`ResidualAttentionBlock`.

**TP linears:** When ``tp_size > 1``, Q/K/V and MLP fc1 use
:class:`~parallelism.TPColumnParallelLinear` (``gather_output=False``); ``out_proj`` and MLP fc2 use
:class:`~parallelism.TPRowParallelLinear` (``input_is_parallel=True``), via ``parallelism.py``
(Megatron or fallback). Fused QKV (``use_fused_qkv=True``) keeps ``in_proj_*`` for PE checkpoints.

**Deformable (2D):** :class:`QueryUpsampleMethod`, :class:`DeformableAttention2D`, :class:`DeformableAttentionBlock`
(optional TP; PatchFormer final layer).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from enum import Enum

from einops import rearrange
from third_party.perception_models.core.vision_encoder import pe
from timm.layers import DropPath
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from checks.utils.models.cwip.modules import parallelism as parallelism_module

# =============================================================================
# TP + CP: Cross-attention, self-attention, residual block (PatchFormer)
# =============================================================================


class CrossAttention(nn.Module):
    """
    Cross attention with Tensor Parallelism (TP) and Context Parallelism (CP).

    **TP:** Splits attention heads across TP ranks; Q/K/V column-parallel; output row-parallel with all-reduce.

    **CP:** Uses :func:`parallelism.sdpa_or_true_cp_attention` (gather + SDPA) when true CP size > 1.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        q_dim: int | None = None,
        kv_dim: int | None = None,
        rope: nn.Module | None = None,
        dropout: float = 0.0,
        use_tensor_parallel: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.q_dim = q_dim if q_dim is not None else embed_dim
        self.kv_dim = kv_dim if kv_dim is not None else embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.tp_size = parallelism_module.get_tensor_parallel_world_size() if use_tensor_parallel else 1
        if num_heads % self.tp_size != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by tp_size ({self.tp_size})")

        self.num_heads_per_rank = num_heads // self.tp_size
        self.embed_dim_per_rank = embed_dim // self.tp_size

        self.q_proj: parallelism_module.TPColumnParallelLinear | nn.Linear
        self.k_proj: parallelism_module.TPColumnParallelLinear | nn.Linear
        self.v_proj: parallelism_module.TPColumnParallelLinear | nn.Linear
        self.out_proj: parallelism_module.TPRowParallelLinear | nn.Linear
        if self.tp_size > 1:
            self.q_proj = parallelism_module.TPColumnParallelLinear(
                self.q_dim, embed_dim, bias=True, gather_output=False
            )
            self.k_proj = parallelism_module.TPColumnParallelLinear(
                self.kv_dim, embed_dim, bias=True, gather_output=False
            )
            self.v_proj = parallelism_module.TPColumnParallelLinear(
                self.kv_dim, embed_dim, bias=True, gather_output=False
            )
            self.out_proj = parallelism_module.TPRowParallelLinear(
                embed_dim, embed_dim, bias=True, input_is_parallel=True
            )
        else:
            self.q_proj = nn.Linear(self.q_dim, self.embed_dim_per_rank, bias=True)
            self.k_proj = nn.Linear(self.kv_dim, self.embed_dim_per_rank, bias=True)
            self.v_proj = nn.Linear(self.kv_dim, self.embed_dim_per_rank, bias=True)
            self.out_proj = nn.Linear(self.embed_dim_per_rank, embed_dim, bias=True)

        self.rope = rope
        self.dropout = nn.Dropout(dropout)

        self._cp_group: dist.ProcessGroup | None = None

        self._init_weights()

    def _init_weights(self) -> None:
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_context_parallel_group(
        self,
        group: dist.ProcessGroup,
    ) -> None:
        self._cp_group = group

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.q_proj(query)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads_per_rank)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads_per_rank)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads_per_rank)

        if self.rope is not None:
            q, k = self.rope(q, k)

        attn_out = parallelism_module.sdpa_or_true_cp_attention(
            q,
            k,
            v,
            self.scale,
            attn_mask,
            self.dropout.p if self.training else 0.0,
            self._cp_group,
        )

        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")
        return self.out_proj(attn_out)


class FusedKVCameraWorldCrossAttention(nn.Module):
    """
    Cross-attention where queries are camera tokens and keys/values attend over camera and world tokens concatenated on
    the sequence axis.

    **RoPE:** Applies the shared 2D RoPE table separately to query/camera-keys and to world keys (same spatial layout),
    then concatenates rotated keys. Values are not rotated (same as :class:`CrossAttention`).

    **TP / CP:** Same projection layout as :class:`CrossAttention`. Under true context parallel, camera and world K/V
    shards are each all-gathered in rank order, then concatenated as ``[K_cam_full, K_world_full]`` (and likewise for V)
    so positions stay consistent with full-sequence attention.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        q_dim: int | None = None,
        kv_dim: int | None = None,
        rope: nn.Module | None = None,
        dropout: float = 0.0,
        use_tensor_parallel: bool = True,
    ) -> None:
        """
        Build Q/K/V projections, output projection, and optional dropout.

        Args:
            embed_dim: Model width; output channels of each projection before ``out_proj``.
            num_heads: Number of attention heads (must divide ``embed_dim`` and TP world size).
            q_dim: Input feature size for queries. Defaults to ``embed_dim`` if ``None``.
            kv_dim: Input feature size for keys and values (camera and world branches). Defaults
                to ``embed_dim`` if ``None``.
            rope: Optional 2D RoPE module applied to Q and to each key branch before concatenation.
            dropout: Dropout probability passed to scaled dot-product attention.
            use_tensor_parallel: If ``True``, use column-parallel Q/K/V and row-parallel ``out_proj``
                when the tensor-parallel world size is greater than one.

        Returns:
            None

        Raises:
            ValueError: If ``embed_dim`` is not divisible by ``num_heads``, or ``num_heads`` is not
                divisible by the tensor-parallel world size.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.q_dim = q_dim if q_dim is not None else embed_dim
        self.kv_dim = kv_dim if kv_dim is not None else embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        if self.head_dim * num_heads != embed_dim:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.tp_size = parallelism_module.get_tensor_parallel_world_size() if use_tensor_parallel else 1
        if num_heads % self.tp_size != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by tp_size ({self.tp_size})")

        self.num_heads_per_rank = num_heads // self.tp_size
        self.embed_dim_per_rank = embed_dim // self.tp_size

        self.q_proj: parallelism_module.TPColumnParallelLinear | nn.Linear
        self.k_proj: parallelism_module.TPColumnParallelLinear | nn.Linear
        self.v_proj: parallelism_module.TPColumnParallelLinear | nn.Linear
        self.out_proj: parallelism_module.TPRowParallelLinear | nn.Linear
        if self.tp_size > 1:
            self.q_proj = parallelism_module.TPColumnParallelLinear(
                self.q_dim, embed_dim, bias=True, gather_output=False
            )
            self.k_proj = parallelism_module.TPColumnParallelLinear(
                self.kv_dim, embed_dim, bias=True, gather_output=False
            )
            self.v_proj = parallelism_module.TPColumnParallelLinear(
                self.kv_dim, embed_dim, bias=True, gather_output=False
            )
            self.out_proj = parallelism_module.TPRowParallelLinear(
                embed_dim, embed_dim, bias=True, input_is_parallel=True
            )
        else:
            self.q_proj = nn.Linear(self.q_dim, self.embed_dim_per_rank, bias=True)
            self.k_proj = nn.Linear(self.kv_dim, self.embed_dim_per_rank, bias=True)
            self.v_proj = nn.Linear(self.kv_dim, self.embed_dim_per_rank, bias=True)
            self.out_proj = nn.Linear(self.embed_dim_per_rank, embed_dim, bias=True)

        self.rope = rope
        self.dropout = nn.Dropout(dropout)

        self._cp_group: dist.ProcessGroup | None = None

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Initialize ``nn.Linear`` projection weights (Xavier uniform) and biases (zeros).

        Tensor-parallel linear types are skipped; their loaders or builders apply init elsewhere.

        Returns:
            None
        """
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def set_context_parallel_group(
        self,
        group: dist.ProcessGroup,
    ) -> None:
        """
        Store the true context-parallel process group for attention.

        Args:
            group: Process group used to all-gather K/V shards along the sequence dimension
                when the group's world size is greater than one.

        Returns:
            None
        """
        self._cp_group = group

    def _sdpa_maybe_cp(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Run scaled dot-product attention with optional true context parallelism.

        When CP is active, ``k`` and ``v`` are split into camera and world halves along sequence dim 2; each half is
        all-gathered, then concatenated as ``[K_cam_full, K_world_full]`` before SDPA. Otherwise delegates to
        :func:`parallelism.sdpa_or_true_cp_attention` with ``cp_group=None``.

        Args:
            q: Query tensor of shape ``(B, H, S_q, D)`` where ``H`` is heads per rank.
            k: Key tensor of shape ``(B, H, S_kv, D)`` with ``S_kv`` even; first half is camera,
                second half is world (same layout as ``v``).
            v: Value tensor of shape ``(B, H, S_kv, D)``, matching ``k`` along dim 2.
            attn_mask: Optional mask for the non-CP path; ignored when true CP size is greater than one.

        Returns:
            Attention output of shape ``(B, H, S_q, D)``.
        """
        if self._cp_group is None or dist.get_world_size(self._cp_group) <= 1:
            return parallelism_module.sdpa_or_true_cp_attention(
                q,
                k,
                v,
                self.scale,
                attn_mask,
                self.dropout.p if self.training else 0.0,
                None,
            )

        cp_size = dist.get_world_size(self._cp_group)

        def gather_heads_seq(t: torch.Tensor) -> torch.Tensor:
            """
            All-gather a tensor from each rank in ``self._cp_group`` and concatenate along dim 2.

            Args:
                t: Tensor of shape ``(B, H, S_local, D)`` on this rank.

            Returns:
                Tensor of shape ``(B, H, S_local * cp_size, D)`` with shards ordered by rank.
            """
            chunks = [torch.empty_like(t) for _ in range(cp_size)]
            dist.all_gather(chunks, t.contiguous(), group=self._cp_group)
            return torch.cat(chunks, dim=2)

        half = k.shape[2] // 2
        k_cam, k_world = k[:, :, :half, :], k[:, :, half:, :]
        v_cam, v_world = v[:, :, :half, :], v[:, :, half:, :]
        k_full = torch.cat([gather_heads_seq(k_cam), gather_heads_seq(k_world)], dim=2)
        v_full = torch.cat([gather_heads_seq(v_cam), gather_heads_seq(v_world)], dim=2)
        return F.scaled_dot_product_attention(
            q,
            k_full,
            v_full,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )

    def forward(
        self,
        query: torch.Tensor,
        kv_camera: torch.Tensor,
        kv_world: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Project queries from camera tokens; project keys/values from camera and world separately,
        apply RoPE to keys (and queries), concatenate K/V along the sequence axis, then SDPA and output projection.

        Args:
            query: Camera token features of shape ``(B, S, q_dim)`` (queries only).
            kv_camera: Camera-side key/value features of shape ``(B, S, kv_dim)`` (same ``S`` as ``query``).
            kv_world: World-side key/value features of shape ``(B, S, kv_dim)``.
            attn_mask: Optional attention mask for SDPA when context parallel is disabled or size one.

        Returns:
            Tensor of shape ``(B, S, embed_dim)`` after attention and ``out_proj``.
        """
        q = self.q_proj(query)
        k_cam = self.k_proj(kv_camera)
        k_world = self.k_proj(kv_world)
        v_cam = self.v_proj(kv_camera)
        v_world = self.v_proj(kv_world)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads_per_rank)
        k_cam = rearrange(k_cam, "b s (h d) -> b h s d", h=self.num_heads_per_rank)
        k_world = rearrange(k_world, "b s (h d) -> b h s d", h=self.num_heads_per_rank)
        v_cam = rearrange(v_cam, "b s (h d) -> b h s d", h=self.num_heads_per_rank)
        v_world = rearrange(v_world, "b s (h d) -> b h s d", h=self.num_heads_per_rank)

        if self.rope is not None:
            q, k_cam = self.rope(q, k_cam)
            _, k_world = self.rope(k_world, k_world)

        k = torch.cat([k_cam, k_world], dim=2)
        v = torch.cat([v_cam, v_world], dim=2)

        attn_out = self._sdpa_maybe_cp(q, k, v, attn_mask)
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")
        return self.out_proj(attn_out)


class SelfAttention(nn.Module):
    """
    Self-attention with TP and optional CP (gather + SDPA via :func:`parallelism.sdpa_or_true_cp_attention`).

    Args:
        use_fused_qkv: If True, single ``in_proj_weight`` / ``in_proj_bias`` (PE checkpoint layout).
            If False, separate ``q_proj`` / ``k_proj`` / ``v_proj`` (PatchFormer default).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rope: nn.Module | None = None,
        dropout: float = 0.0,
        use_tensor_parallel: bool = True,
        use_fused_qkv: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_fused_qkv = use_fused_qkv

        self.tp_size = parallelism_module.get_tensor_parallel_world_size() if use_tensor_parallel else 1
        if num_heads % self.tp_size != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by tp_size ({self.tp_size})")

        self.num_heads_per_rank = num_heads // self.tp_size
        self.embed_dim_per_rank = embed_dim // self.tp_size

        self.q_proj: parallelism_module.TPColumnParallelLinear | nn.Linear | None
        self.k_proj: parallelism_module.TPColumnParallelLinear | nn.Linear | None
        self.v_proj: parallelism_module.TPColumnParallelLinear | nn.Linear | None
        if use_fused_qkv:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim // self.tp_size, embed_dim))
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim // self.tp_size))
            self.q_proj = self.k_proj = self.v_proj = None
        else:
            self.register_parameter("in_proj_weight", None)
            self.register_parameter("in_proj_bias", None)
            if self.tp_size > 1:
                self.q_proj = parallelism_module.TPColumnParallelLinear(
                    embed_dim, embed_dim, bias=True, gather_output=False
                )
                self.k_proj = parallelism_module.TPColumnParallelLinear(
                    embed_dim, embed_dim, bias=True, gather_output=False
                )
                self.v_proj = parallelism_module.TPColumnParallelLinear(
                    embed_dim, embed_dim, bias=True, gather_output=False
                )
            else:
                self.q_proj = nn.Linear(embed_dim, self.embed_dim_per_rank, bias=True)
                self.k_proj = nn.Linear(embed_dim, self.embed_dim_per_rank, bias=True)
                self.v_proj = nn.Linear(embed_dim, self.embed_dim_per_rank, bias=True)

        self.out_proj: parallelism_module.TPRowParallelLinear | nn.Linear
        if self.tp_size > 1:
            self.out_proj = parallelism_module.TPRowParallelLinear(
                embed_dim, embed_dim, bias=True, input_is_parallel=True
            )
        else:
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.rope = rope
        self.dropout = nn.Dropout(dropout)

        self._cp_group: dist.ProcessGroup | None = None

        self._init_weights()

    def _init_weights(self) -> None:
        if self.use_fused_qkv:
            nn.init.xavier_uniform_(self.in_proj_weight)
            nn.init.zeros_(self.in_proj_bias)
        else:
            assert self.q_proj is not None
            for m in (self.q_proj, self.k_proj, self.v_proj):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            if isinstance(self.out_proj, nn.Linear):
                nn.init.xavier_uniform_(self.out_proj.weight)
                if self.out_proj.bias is not None:
                    nn.init.zeros_(self.out_proj.bias)

    def set_context_parallel_group(self, group: dist.ProcessGroup) -> None:
        self._cp_group = group

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_fused_qkv:
            qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
            qkv = rearrange(qkv, "b s (n h d) -> n b h s d", n=3, h=self.num_heads_per_rank)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            assert self.q_proj is not None
            assert self.k_proj is not None
            assert self.v_proj is not None
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads_per_rank)
            k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads_per_rank)
            v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads_per_rank)

        if self.rope is not None:
            q, k = self.rope(q, k)

        attn_out = parallelism_module.sdpa_or_true_cp_attention(
            q,
            k,
            v,
            self.scale,
            attn_mask,
            self.dropout.p if self.training else 0.0,
            self._cp_group,
        )

        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")
        return self.out_proj(attn_out)


class ResidualAttentionBlock(nn.Module):
    """
    Residual block with TP-enabled self- or cross-attention and sharded MLP.

    For self-attention only, ``use_fused_qkv=True`` uses PE-style ``in_proj_*``;
    default ``False`` uses separate ``q_proj`` / ``k_proj`` / ``v_proj`` (PatchFormer).

    Layer names align with checkpoint conventions: ``attn``, ``ln_1``, ``ln_2``,
    ``ln_context`` (cross), ``mlp.c_fc``, ``mlp.c_proj``, ``ls_1``, ``ls_2``.

    When ``use_fused_kv_camera_world_cross_attn`` is True (cross-attention only), attention is
    :class:`FusedKVCameraWorldCrossAttention`: queries use ``ln_1(x)``; keys/values use the same
    tensor for the camera branch and ``ln_context(context)`` for the world branch, with
    per-branch RoPE on keys before concatenation.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        q_dim: int | None = None,
        kv_dim: int | None = None,
        mlp_ratio: float = 4.0,
        ls_init_value: float | None = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        drop_path: float = 0.0,
        rope: nn.Module | None = None,
        use_cross_attn: bool = True,
        use_tensor_parallel: bool = True,
        use_fused_qkv: bool = False,
        use_fused_kv_camera_world_cross_attn: bool = False,
    ):
        super().__init__()

        self.use_cross_attn = use_cross_attn
        self.use_fused_kv_camera_world_cross_attn = use_fused_kv_camera_world_cross_attn
        self.tp_size = parallelism_module.get_tensor_parallel_world_size() if use_tensor_parallel else 1

        if use_fused_qkv and use_cross_attn:
            raise ValueError("use_fused_qkv applies only to self-attention; set use_cross_attn=False.")
        if use_fused_kv_camera_world_cross_attn and not use_cross_attn:
            raise ValueError("use_fused_kv_camera_world_cross_attn requires use_cross_attn=True.")

        self.attn: nn.Module
        if use_cross_attn:
            if use_fused_kv_camera_world_cross_attn:
                self.attn = FusedKVCameraWorldCrossAttention(
                    embed_dim=d_model,
                    num_heads=n_head,
                    q_dim=q_dim,
                    kv_dim=kv_dim,
                    rope=rope,
                    use_tensor_parallel=use_tensor_parallel,
                )
            else:
                self.attn = CrossAttention(
                    embed_dim=d_model,
                    num_heads=n_head,
                    q_dim=q_dim,
                    kv_dim=kv_dim,
                    rope=rope,
                    use_tensor_parallel=use_tensor_parallel,
                )
        else:
            self.attn = SelfAttention(
                embed_dim=d_model,
                num_heads=n_head,
                rope=rope,
                use_tensor_parallel=use_tensor_parallel,
                use_fused_qkv=use_fused_qkv,
            )

        self.ln_1 = norm_layer(d_model)
        self.ln_context = norm_layer(kv_dim if kv_dim else d_model) if use_cross_attn else None
        self.ln_2 = norm_layer(d_model)

        mlp_hidden = int(d_model * mlp_ratio)

        if self.tp_size > 1:
            self.mlp = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "c_fc",
                            parallelism_module.TPColumnParallelLinear(
                                d_model, mlp_hidden, bias=True, gather_output=False
                            ),
                        ),
                        ("gelu", act_layer()),
                        (
                            "c_proj",
                            parallelism_module.TPRowParallelLinear(
                                mlp_hidden, d_model, bias=True, input_is_parallel=True
                            ),
                        ),
                    ]
                )
            )
        else:
            self.mlp = nn.Sequential(
                OrderedDict(
                    [
                        ("c_fc", nn.Linear(d_model, mlp_hidden)),
                        ("gelu", act_layer()),
                        ("c_proj", nn.Linear(mlp_hidden, d_model)),
                    ]
                )
            )

        if ls_init_value is not None:
            self.ls_1 = pe.LayerScale(d_model, ls_init_value)
            self.ls_2 = pe.LayerScale(d_model, ls_init_value)
        else:
            self.ls_1 = nn.Identity()
            self.ls_2 = nn.Identity()

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_norm = self.ln_1(x)

        if self.use_cross_attn:
            if context is None:
                raise ValueError("context is required when use_cross_attn=True")
            assert self.ln_context is not None
            ctx_norm = self.ln_context(context)
            if self.use_fused_kv_camera_world_cross_attn:
                attn_out = self.attn(q_norm, q_norm, ctx_norm, attn_mask)
            else:
                attn_out = self.attn(q_norm, ctx_norm, attn_mask)
        else:
            attn_out = self.attn(q_norm, attn_mask)

        attn_out = self.ls_1(attn_out)
        x = x + self.drop_path1(attn_out)

        x2 = self.ln_2(x)
        mlp_out = self.mlp(x2)

        mlp_out = self.ls_2(mlp_out)
        x = x + self.drop_path2(mlp_out)

        return x


# =============================================================================
# Deformable attention (2D spatial, optional TP)
# =============================================================================


class QueryUpsampleMethod(str, Enum):
    """
    Method for upsampling queries to output resolution in deformable attention.

    SUBPIXEL_CONV: Learnable upsampling via Sub-pixel Convolution (Conv2d + PixelShuffle).
        From ESPCN paper: "Real-Time Single Image and Video Super-Resolution Using an
        Efficient Sub-Pixel Convolutional Neural Network" (Shi et al., 2016).
        Captures local spatial patterns through convolution before rearranging channels to space.
        Setting kernel_size=1 is equivalent to Patch Expand (Swin-style).
    BILINEAR: Bilinear interpolation (non-learnable).
        Simple and fast, but no learnable parameters.
    """

    SUBPIXEL_CONV = "subpixel_conv"
    BILINEAR = "bilinear"


class DeformableAttention2D(nn.Module):
    """
    Deformable Attention for 2D spatial features with Tensor Parallelism support.

    This is a hybrid implementation combining:
    - **Content-based attention**: Scaled dot-product attention (Q·K) from
      "Attention Is All You Need" (Vaswani et al., 2017) https://arxiv.org/abs/1706.03762
    - **Position-based attention**: Learned attention weights from
      "Deformable DETR" (Zhu et al., 2020) https://arxiv.org/abs/2010.04159

    The hybrid combination (content + position) is our custom design, not from a paper.

    References:
    - Attention Is All You Need (content-based): https://arxiv.org/abs/1706.03762
    - Deformable DETR (position-based + offsets): https://arxiv.org/abs/2010.04159
    - DAT (Deformable Attention Transformer): https://arxiv.org/abs/2201.00520

    Algorithm Steps & Equations
    ---------------------------

    **Notation:**
    - B = batch size, H = num_heads, D = head_dim, K = num_sampling_points
    - N_in = H_in × W_in (input spatial positions)
    - N_out = H_out × W_out (output spatial positions, may differ for upscaling)
    - x ∈ R^(B × N_in × C) = input features

    **Step 1: Project K, V from input features**
        K = x · W_K,  V = x · W_V
        K, V ∈ R^(B × N_in × C)  →  reshape to R^(B × H × D × H_in × W_in)

    **Step 2: Create queries at output resolution (for upscaling)**
        If upscaling (N_out > N_in):
            q_features = bilinear_interpolate(x, (H_out, W_out))
        Else:
            q_features = x
        Q = q_features · W_Q,  Q ∈ R^(B × N_out × C)

    **Step 3: Predict sampling offsets from queries (Deformable DETR)**
        Δp = tanh(offset_net(q_features)) × 0.5
        Δp ∈ R^(B × N_out × H × K × 2)  (x, y offsets per head, per sampling point)

    **Step 4: Predict learned attention weights (Deformable DETR style)**
        A_learned = softmax(attn_weights_net(q_features), dim=-1)
        A_learned ∈ R^(B × N_out × H × K)

    **Step 5: Compute sampling grid**
        p_ref = meshgrid(H_out, W_out)  (reference positions in [-1, 1])
        p_sample = clamp(p_ref + 2 × Δp, -1, 1)
        p_sample ∈ R^(B × N_out × H × K × 2)

    **Step 6: Sample K, V at offset positions via bilinear interpolation**
        K_sampled = grid_sample(K, p_sample)
        V_sampled = grid_sample(V, p_sample)
        K_sampled, V_sampled ∈ R^(B × H × N_out × K × D)

    **Step 7a: Content-based attention (Vaswani et al., 2017)**
        Standard scaled dot-product attention over sampled positions:
        attn_scores = (Q · K_sampled^T) / √D
        A_content = softmax(attn_scores, dim=-1)
        out_content = A_content · V_sampled
        out_content ∈ R^(B × H × N_out × D)

    **Step 7b: Position-based attention (Zhu et al., 2020 - Deformable DETR)**
        Directly predicted attention weights (no Q·K computation):
        out_learned = A_learned · V_sampled
        out_learned ∈ R^(B × H × N_out × D)

    **Step 8: Combine and project (our hybrid design)**
        out = out_content + learned_weight_scale × out_learned
        output = out · W_O
        output ∈ R^(B × N_out × C)

        Note: learned_weight_scale (default 0.1) is our design choice.
        Content attention is primary; position attention is a small correction.
        The original Deformable DETR uses only position-based attention.

    Complexity
    ----------
    - Standard attention: O(N² × D)
    - Deformable attention: O(N × K × D) where K << N

    Tensor Parallelism
    ------------------
    - Attention heads are sharded across TP ranks (H_local = H // tp_size)
    - Q, K, V projections output embed_dim // tp_size
    - Output projection: row-parallel matmul, all-reduce, then add bias once
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_sampling_points: int = 4,
        dropout: float = 0.0,
        use_tensor_parallel: bool = False,
        learned_weight_scale: float = 0.1,
        query_upsample_method: QueryUpsampleMethod | str = QueryUpsampleMethod.SUBPIXEL_CONV,
        query_upsample_scale_factor: int = 2,
        query_upsample_kernel_size: int = 3,
        use_sdpa: bool = False,
    ):
        """
        Initialize DeformableAttention2D.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads. Must be divisible by TP size when using TP.
            num_sampling_points: Number of sampling points per head per query.
                More points = more capacity but more compute. Default 4.
            dropout: Dropout probability for attention weights.
            use_tensor_parallel: If True, enable Tensor Parallelism for attention heads.
            learned_weight_scale: Scale factor for the learned attention weights (position-based)
                when combining with content-based attention. Default 0.1.
                This is a design choice for this hybrid implementation (not from a paper).
                The hybrid combines: out = content_attn + scale * position_attn
                Small values (0.1) make content-based attention dominant with position-based
                as a small correction. Set to 1.0 for equal weighting, or 0.0 to disable.
            query_upsample_method: Method for upsampling queries to output resolution.
                QueryUpsampleMethod.SUBPIXEL_CONV: Learnable upsampling via Sub-pixel Convolution (default).
                QueryUpsampleMethod.BILINEAR: Bilinear interpolation (non-learnable).
            query_upsample_scale_factor: Scale factor for subpixel conv upsampling. Default 2.
                Must match the actual upscale factor at runtime. Falls back to bilinear if mismatch.
            query_upsample_kernel_size: Kernel size for the Conv2d in subpixel conv. Default 3.
                Note: Setting kernel_size=1 is equivalent to Patch Expand (Swin-style),
                which uses only point-wise operations without spatial context.
            use_sdpa: If True, use scaled dot-product attention for content scores; if False (default),
                use the einsum path (more robust for very large ``N_out``).
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_sampling_points = num_sampling_points
        self.scale = self.head_dim**-0.5
        self.use_tensor_parallel = use_tensor_parallel
        # Scale for combining content-based (Q·K) and position-based (learned weights) attention.
        # Not from a paper - this hybrid design uses content attention as primary signal
        # with learned position attention as a small residual correction.
        self.learned_weight_scale = learned_weight_scale

        # Query upsampling configuration
        # Convert string to enum if needed (for backward compatibility with configs)
        if isinstance(query_upsample_method, str):
            query_upsample_method = QueryUpsampleMethod(query_upsample_method)
        self.query_upsample_method = query_upsample_method
        self.query_upsample_scale_factor = query_upsample_scale_factor
        self.query_upsample_kernel_size = query_upsample_kernel_size

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        # Tensor Parallelism setup
        self.tp_size = parallelism_module.get_tensor_parallel_world_size() if use_tensor_parallel else 1
        if use_tensor_parallel and num_heads % self.tp_size != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by tp_size ({self.tp_size})")

        self.num_heads_per_rank = num_heads // self.tp_size
        self.embed_dim_per_rank = embed_dim // self.tp_size

        # Q, K, V projections (column parallel: output sharded across TP ranks)
        self.q_proj: parallelism_module.TPColumnParallelLinear | nn.Linear
        self.k_proj: parallelism_module.TPColumnParallelLinear | nn.Linear
        self.v_proj: parallelism_module.TPColumnParallelLinear | nn.Linear
        if self.tp_size > 1:
            self.q_proj = parallelism_module.TPColumnParallelLinear(
                embed_dim, embed_dim, bias=True, gather_output=False
            )
            self.k_proj = parallelism_module.TPColumnParallelLinear(
                embed_dim, embed_dim, bias=True, gather_output=False
            )
            self.v_proj = parallelism_module.TPColumnParallelLinear(
                embed_dim, embed_dim, bias=True, gather_output=False
            )
        else:
            self.q_proj = nn.Linear(embed_dim, self.embed_dim_per_rank)
            self.k_proj = nn.Linear(embed_dim, self.embed_dim_per_rank)
            self.v_proj = nn.Linear(embed_dim, self.embed_dim_per_rank)

        # Offset prediction network: predicts (x, y) offsets for each LOCAL head and sampling point
        # Output: (num_heads_per_rank * num_sampling_points * 2) for x, y offsets
        self.offset_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, self.num_heads_per_rank * num_sampling_points * 2),
        )

        # Attention weight modulation (for local heads only)
        self.attn_weights_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, self.num_heads_per_rank * num_sampling_points),
        )

        # Output projection (row parallel: input sharded, output all-reduced)
        self.out_proj: parallelism_module.TPRowParallelLinear | nn.Linear
        if self.tp_size > 1:
            self.out_proj = parallelism_module.TPRowParallelLinear(
                embed_dim, embed_dim, bias=True, input_is_parallel=True
            )
        else:
            self.out_proj = nn.Linear(self.embed_dim_per_rank, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.use_sdpa = use_sdpa

        # Sub-pixel convolution upsampling module (Conv2d + PixelShuffle)
        # This is a learnable alternative to bilinear interpolation for query upsampling.
        # Note: Setting query_upsample_kernel_size=1 is equivalent to Patch Expand (Swin-style),
        # which uses only point-wise linear projection without spatial context from neighbors.
        self.query_upsample_conv: nn.Conv2d | None
        self.query_pixel_shuffle: nn.PixelShuffle | None
        if self.query_upsample_method == QueryUpsampleMethod.SUBPIXEL_CONV:
            r = query_upsample_scale_factor
            padding = (query_upsample_kernel_size - 1) // 2
            self.query_upsample_conv = nn.Conv2d(
                embed_dim,
                embed_dim * (r**2),  # Expand channels by r^2 for pixel shuffle
                kernel_size=query_upsample_kernel_size,
                padding=padding,
            )
            self.query_pixel_shuffle = nn.PixelShuffle(r)
        else:
            self.query_upsample_conv = None
            self.query_pixel_shuffle = None

        self._init_weights()

    def _compute_attention_einsum(
        self,
        q: torch.Tensor,
        k_sampled: torch.Tensor,
        v_sampled: torch.Tensor,
        attn_weights: torch.Tensor,
        B: int,
        N_out: int,
        num_heads_local: int,
    ) -> torch.Tensor:
        """
        Compute attention using einsum operations.

        This is more memory-efficient for large N_out values as it avoids
        creating massive batch dimensions that can exceed CUDA kernel limits.

        Args:
            q: Query tensor [B, N_out, embed_dim_per_rank]
            k_sampled: Sampled keys [B, H_local, D, N_out, K]
            v_sampled: Sampled values [B, H_local, D, N_out, K]
            attn_weights: Learned attention weights [B, N_out, H_local, K]
            B: Batch size
            N_out: Number of output positions
            num_heads_local: Number of local heads (per TP rank)

        Returns:
            Output tensor [B, N_out, embed_dim_per_rank]
        """
        # Q: [B, N_out, embed_dim_per_rank] -> [B, H_local, N_out, D]
        q = q.view(B, N_out, num_heads_local, self.head_dim).permute(0, 2, 1, 3)

        # k_sampled, v_sampled: [B, H_local, D, N_out, K] -> [B, H_local, N_out, K, D]
        k_sampled = k_sampled.permute(0, 1, 3, 4, 2)
        v_sampled = v_sampled.permute(0, 1, 3, 4, 2)

        # Compute scaled dot-product attention scores: Q @ K^T
        # q: [B, H_local, N_out, D], k_sampled: [B, H_local, N_out, K, D]
        # -> attn_scores: [B, H_local, N_out, K]
        attn_scores = torch.einsum("bhnd,bhnkd->bhnk", q, k_sampled) * self.scale

        # Softmax over sampling points
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply dropout when the module is in train mode
        if self.training and self.dropout.p > 0:
            attn_probs = self.dropout(attn_probs)

        # Weighted sum of values: attn_probs @ V
        # attn_probs: [B, H_local, N_out, K], v_sampled: [B, H_local, N_out, K, D]
        # -> attn_out: [B, H_local, N_out, D]
        attn_out = torch.einsum("bhnk,bhnkd->bhnd", attn_probs, v_sampled)

        # Apply learned attention weight modulation
        # attn_weights: [B, N_out, H_local, K] -> [B, H_local, N_out, K]
        attn_weights_reshaped = attn_weights.permute(0, 2, 1, 3)
        v_weighted = torch.einsum("bhnk,bhnkd->bhnd", attn_weights_reshaped, v_sampled)

        # Combine content-based attention with position-based (learned weights) attention.
        # This hybrid design is not from a paper - content attention is primary,
        # learned position attention provides a small residual correction.
        out = attn_out + self.learned_weight_scale * v_weighted

        # [B, H_local, N_out, D] -> [B, N_out, embed_dim_per_rank]
        out = out.permute(0, 2, 1, 3).reshape(B, N_out, self.embed_dim_per_rank)
        return out

    def _compute_attention_sdpa(
        self,
        q: torch.Tensor,
        k_sampled: torch.Tensor,
        v_sampled: torch.Tensor,
        attn_weights: torch.Tensor,
        B: int,
        N_out: int,
        num_heads_local: int,
    ) -> torch.Tensor:
        """
        Compute attention using PyTorch's scaled_dot_product_attention (SDPA).

        WARNING: This can fail with "CUDA error: invalid configuration argument"
        when N_out is large (e.g., > 10000) because it creates a batch dimension
        of B * N_out which may exceed CUDA kernel limits.

        Use for smaller output resolutions or when Flash Attention is needed.

        Args:
            q: Query tensor [B, N_out, embed_dim_per_rank]
            k_sampled: Sampled keys [B, H_local, D, N_out, K]
            v_sampled: Sampled values [B, H_local, D, N_out, K]
            attn_weights: Learned attention weights [B, N_out, H_local, K]
            B: Batch size
            N_out: Number of output positions
            num_heads_local: Number of local heads (per TP rank)

        Returns:
            Output tensor [B, N_out, embed_dim_per_rank]
        """
        # Reshape Q: [B, N_out, embed_dim_per_rank] -> [B * N_out, H_local, 1, D]
        q = q.view(B, N_out, num_heads_local, self.head_dim)
        q = q.view(B * N_out, num_heads_local, 1, self.head_dim)

        # k_sampled, v_sampled: [B, H_local, D, N_out, K] -> [B * N_out, H_local, K, D]
        k_sampled = k_sampled.permute(0, 3, 1, 4, 2).reshape(
            B * N_out, num_heads_local, self.num_sampling_points, self.head_dim
        )
        v_sampled = v_sampled.permute(0, 3, 1, 4, 2).reshape(
            B * N_out, num_heads_local, self.num_sampling_points, self.head_dim
        )

        # Use scaled_dot_product_attention (auto-dispatches to Flash Attention on GPU)
        dropout_p = self.dropout.p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q,
            k_sampled,
            v_sampled,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=False,
            scale=self.scale,
        )

        # Apply learned attention weight modulation
        attn_weights_reshaped = attn_weights.view(B * N_out, num_heads_local, 1, self.num_sampling_points)
        v_weighted = torch.einsum("bhok,bhkd->bhod", attn_weights_reshaped, v_sampled)

        # Combine content-based attention with position-based (learned weights) attention.
        # This hybrid design is not from a paper - content attention is primary,
        # learned position attention provides a small residual correction.
        out = attn_out + self.learned_weight_scale * v_weighted

        # Reshape output: [B * N_out, H_local, 1, D] -> [B, N_out, embed_dim_per_rank]
        out = out.view(B, N_out, num_heads_local, self.head_dim)
        out = out.reshape(B, N_out, self.embed_dim_per_rank)
        return out

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize offsets to small values (start near reference points)
        last_offset_layer = self.offset_net[-1]
        assert isinstance(last_offset_layer, nn.Linear)
        nn.init.zeros_(last_offset_layer.weight)
        if last_offset_layer.bias is not None:
            nn.init.zeros_(last_offset_layer.bias)

        # Initialize attention weights uniformly
        last_attn_layer = self.attn_weights_net[-1]
        assert isinstance(last_attn_layer, nn.Linear)
        nn.init.zeros_(last_attn_layer.weight)
        if last_attn_layer.bias is not None:
            nn.init.zeros_(last_attn_layer.bias)

        # Initialize pixel shuffle conv if present
        if self.query_upsample_conv is not None:
            nn.init.xavier_uniform_(self.query_upsample_conv.weight)
            if self.query_upsample_conv.bias is not None:
                nn.init.zeros_(self.query_upsample_conv.bias)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        input_spatial_shape: tuple[int, int],
        output_spatial_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with deformable attention and optional Tensor Parallelism.

        Supports upscaling: queries are created at output_spatial_shape resolution,
        while K/V are sampled from input_spatial_shape resolution. This enables
        memory-efficient upscaling without processing high-res features through
        the entire transformer.

        Args:
            query: Query tensor [B, N_q, C] where N_q = H_in * W_in
            key_value: Key/Value tensor [B, N_kv, C] where N_kv = H_in * W_in
            input_spatial_shape: (H_in, W_in) input spatial dimensions for K/V
            output_spatial_shape: (H_out, W_out) output spatial dimensions. If None,
                same as input_spatial_shape (no upscaling).

        Returns:
            Output tensor [B, N_out, C] where N_out = H_out * W_out
        """
        B, _N_kv, C = key_value.shape
        H_in, W_in = input_spatial_shape

        # Output shape defaults to input shape (no upscaling)
        if output_spatial_shape is None:
            output_spatial_shape = input_spatial_shape
        H_out, W_out = output_spatial_shape
        N_out = H_out * W_out

        # Use local heads (sharded when TP is enabled)
        num_heads_local = self.num_heads_per_rank

        # Project K, V from input (output is sharded across TP ranks when TP enabled)
        k = self.k_proj(key_value)  # [B, N_kv, embed_dim_per_rank]
        v = self.v_proj(key_value)  # [B, N_kv, embed_dim_per_rank]

        # Reshape K, V to spatial format for grid_sample (using local heads)
        # Need contiguous() after permute for view() to work
        k_spatial = (
            k.view(B, H_in, W_in, num_heads_local, self.head_dim).permute(0, 3, 4, 1, 2).contiguous()
        )  # [B, H_local, D, H_in, W_in]
        v_spatial = (
            v.view(B, H_in, W_in, num_heads_local, self.head_dim).permute(0, 3, 4, 1, 2).contiguous()
        )  # [B, H_local, D, H_in, W_in]

        # Create query at OUTPUT resolution by upsampling input query
        # This is the key to upscaling: queries are at high-res, K/V at low-res
        if H_out != H_in or W_out != W_in:
            # Reshape query to spatial format
            # Use .contiguous() to force NCHW memory layout - NHWC has INT_MAX element limit
            query_spatial = query.view(B, H_in, W_in, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H_in, W_in]

            # Check if subpixel conv can be used (scale factor must match)
            scale_h, scale_w = H_out // H_in, W_out // W_in
            use_subpixel_conv = (
                self.query_upsample_method == QueryUpsampleMethod.SUBPIXEL_CONV
                and self.query_upsample_conv is not None
                and scale_h == scale_w == self.query_upsample_scale_factor
                and H_out == H_in * scale_h  # Exact integer scaling
                and W_out == W_in * scale_w
            )

            if use_subpixel_conv:
                # Sub-pixel Convolution: Conv2d expands channels by r^2, then PixelShuffle rearranges to spatial
                # This is a learnable upsampling method that captures local spatial patterns.
                # Note: Using kernel_size=1 is equivalent to Patch Expand (Swin-style upsampling),
                # which performs point-wise linear projection without spatial context from neighbors.
                assert self.query_upsample_conv is not None
                assert self.query_pixel_shuffle is not None
                query_spatial = self.query_upsample_conv(query_spatial)  # [B, C*r^2, H_in, W_in]
                query_spatial = self.query_pixel_shuffle(query_spatial)  # [B, C, H_out, W_out]
            else:
                # Bilinear interpolation: non-learnable fallback for arbitrary scale factors
                query_spatial = F.interpolate(query_spatial, size=(H_out, W_out), mode="bilinear", align_corners=True)

            query_out = query_spatial.permute(0, 2, 3, 1).reshape(B, N_out, C)  # [B, N_out, C]
        else:
            query_out = query

        # Project Q at output resolution
        q = self.q_proj(query_out)  # [B, N_out, embed_dim_per_rank]

        # Predict offsets from queries at OUTPUT resolution
        # Each output position predicts where to sample from input
        offsets = self.offset_net(query_out)
        offsets = offsets.view(B, N_out, num_heads_local, self.num_sampling_points, 2)  # [B, N_out, H_local, K, 2]
        offsets = torch.tanh(offsets) * 0.5  # Limit offsets to [-0.5, 0.5] of spatial extent

        # Predict attention weights for LOCAL heads at output resolution
        attn_weights = self.attn_weights_net(query_out)
        attn_weights = attn_weights.view(B, N_out, num_heads_local, self.num_sampling_points)  # [B, N_out, H_local, K]
        attn_weights = F.softmax(attn_weights, dim=-1)  # Normalize over sampling points

        # Create reference grid at OUTPUT resolution (normalized coordinates [-1, 1])
        # These are the base positions in input space that each output query samples from
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H_out, device=query.device),
            torch.linspace(-1, 1, W_out, device=query.device),
            indexing="ij",
        )
        ref_grid = torch.stack([grid_x, grid_y], dim=-1)  # [H_out, W_out, 2]
        ref_grid = ref_grid.view(1, N_out, 1, 1, 2).expand(B, -1, num_heads_local, self.num_sampling_points, -1)

        # Add offsets to reference positions
        # offsets are in [-0.5, 0.5], scale to grid coordinates
        sampling_grid = ref_grid + offsets * 2  # Scale offsets to [-1, 1] range
        sampling_grid = sampling_grid.clamp(-1, 1)  # Clamp to valid range

        # Sample K and V from INPUT resolution at offset positions
        # Reshape for grid_sample: [B * H_local, D, H_in, W_in] and grid [B * H_local, N_out * K, 1, 2]
        k_spatial_flat = k_spatial.view(B * num_heads_local, self.head_dim, H_in, W_in)
        v_spatial_flat = v_spatial.view(B * num_heads_local, self.head_dim, H_in, W_in)

        # Reshape sampling grid: [B, N_out, H_local, K, 2] -> [B * H_local, N_out * K, 1, 2]
        # Cast to same dtype as input for grid_sample (handles mixed precision)
        sampling_grid_flat = sampling_grid.permute(0, 2, 1, 3, 4).reshape(
            B * num_heads_local, N_out * self.num_sampling_points, 1, 2
        )
        sampling_grid_flat = sampling_grid_flat.to(dtype=k_spatial_flat.dtype)

        # Sample using grid_sample from INPUT resolution
        # Output: [B * H_local, D, N_out * K, 1]
        k_sampled = F.grid_sample(
            k_spatial_flat, sampling_grid_flat, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        v_sampled = F.grid_sample(
            v_spatial_flat, sampling_grid_flat, mode="bilinear", padding_mode="zeros", align_corners=True
        )

        # Reshape back: [B, H_local, D, N_out, K]
        k_sampled = k_sampled.view(B, num_heads_local, self.head_dim, N_out, self.num_sampling_points)
        v_sampled = v_sampled.view(B, num_heads_local, self.head_dim, N_out, self.num_sampling_points)

        # Compute attention and project output using the selected method
        if self.use_sdpa:
            out = self._compute_attention_sdpa(q, k_sampled, v_sampled, attn_weights, B, N_out, num_heads_local)
        else:
            out = self._compute_attention_einsum(q, k_sampled, v_sampled, attn_weights, B, N_out, num_heads_local)

        return self.out_proj(out)


class DeformableAttentionBlock(nn.Module):
    """
    Residual block with deformable attention and MLP, with Tensor Parallelism support.

    This block replaces standard attention with deformable attention for
    the final layer(s) of PatchFormer when processing high-resolution outputs.

    When TP is enabled:
    - Attention heads are sharded across TP ranks
    - MLP: column-parallel fc1, row-parallel fc2 (matmul → all_reduce → bias)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_sampling_points: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_tensor_parallel: bool = False,
        grad_checkpointing: bool = False,
        query_upsample_method: QueryUpsampleMethod | str = QueryUpsampleMethod.SUBPIXEL_CONV,
        query_upsample_scale_factor: int = 2,
        query_upsample_kernel_size: int = 3,
    ):
        """
        Initialize DeformableAttentionBlock.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads. Must be divisible by TP size when using TP.
            num_sampling_points: Number of sampling points per head.
            mlp_ratio: MLP hidden dim ratio.
            dropout: Dropout probability.
            drop_path: Drop path probability for stochastic depth.
            use_tensor_parallel: If True, enable Tensor Parallelism.
            grad_checkpointing: If True, use gradient checkpointing to save memory.
            query_upsample_method: Method for upsampling queries (QueryUpsampleMethod enum).
            query_upsample_scale_factor: Scale factor for subpixel conv upsampling.
            query_upsample_kernel_size: Kernel size for subpixel conv.
        """
        super().__init__()

        self.use_tensor_parallel = use_tensor_parallel
        self.grad_checkpointing = grad_checkpointing
        self.tp_size = parallelism_module.get_tensor_parallel_world_size() if use_tensor_parallel else 1

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = DeformableAttention2D(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_sampling_points=num_sampling_points,
            dropout=dropout,
            use_tensor_parallel=use_tensor_parallel,
            query_upsample_method=query_upsample_method,
            query_upsample_scale_factor=query_upsample_scale_factor,
            query_upsample_kernel_size=query_upsample_kernel_size,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        # MLP with TP: TPColumnParallelLinear (fc1) + TPRowParallelLinear (fc2)
        mlp_hidden = int(embed_dim * mlp_ratio)
        mlp_hidden_per_rank = mlp_hidden // self.tp_size

        if self.tp_size > 1:
            self.mlp = nn.Sequential(
                parallelism_module.TPColumnParallelLinear(embed_dim, mlp_hidden, bias=True, gather_output=False),
                nn.GELU(),
                nn.Dropout(dropout),
                parallelism_module.TPRowParallelLinear(mlp_hidden, embed_dim, bias=True, input_is_parallel=True),
                nn.Dropout(dropout),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(embed_dim, mlp_hidden_per_rank),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden_per_rank, embed_dim),
                nn.Dropout(dropout),
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _attn_forward(
        self,
        x_normed: torch.Tensor,
        kv_normed: torch.Tensor,
        input_spatial_shape: tuple[int, int],
        output_spatial_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Helper for gradient checkpointing of attention."""
        return self.attn(
            x_normed,
            kv_normed,
            input_spatial_shape=input_spatial_shape,
            output_spatial_shape=output_spatial_shape,
        )

    def _mlp_forward(self, x: torch.Tensor) -> torch.Tensor:
        """MLP forward (TP collectives inside TPColumnParallelLinear / TPRowParallelLinear)."""
        x2 = self.norm2(x)
        return self.mlp(x2)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        input_spatial_shape: tuple[int, int] | None = None,
        output_spatial_shape: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional Tensor Parallelism and upscaling.

        When output_spatial_shape differs from input_spatial_shape, the deformable
        attention upscales the output. In this case, the residual connection
        interpolates x to match the output resolution.

        Args:
            x: Input tensor [B, N_in, C] where N_in = H_in * W_in
            context: Context tensor for cross-attention [B, N_in, C]. If None, uses self-attention.
            input_spatial_shape: (H_in, W_in) input spatial shape. Required.
            output_spatial_shape: (H_out, W_out) output spatial shape. If None, same as input.

        Returns:
            Output tensor [B, N_out, C] where N_out = H_out * W_out
        """
        assert input_spatial_shape is not None, "input_spatial_shape required for DeformableAttentionBlock"

        H_in, W_in = input_spatial_shape
        if output_spatial_shape is None:
            output_spatial_shape = input_spatial_shape
        H_out, W_out = output_spatial_shape
        is_upscaling = H_out != H_in or W_out != W_in

        # Self-attention or cross-attention
        kv = context if context is not None else x

        # Deformable attention (handles upscaling internally)
        if self.grad_checkpointing and self.training:
            attn_out = checkpoint(
                self._attn_forward,
                self.norm1(x),
                self.norm1(kv),
                input_spatial_shape,
                output_spatial_shape,
                use_reentrant=False,
            )
        else:
            attn_out = self.attn(
                self.norm1(x),
                self.norm1(kv),
                input_spatial_shape=input_spatial_shape,
                output_spatial_shape=output_spatial_shape,
            )

        # Residual connection: interpolate x if upscaling
        if is_upscaling:
            B, _, C = x.shape
            # Use .contiguous() to force NCHW memory layout - NHWC has INT_MAX element limit
            x_spatial = x.view(B, H_in, W_in, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H_in, W_in]
            x_spatial = F.interpolate(x_spatial, size=(H_out, W_out), mode="bilinear", align_corners=True)
            x = x_spatial.permute(0, 2, 3, 1).reshape(B, H_out * W_out, C)  # [B, N_out, C]

        x = x + self.drop_path(attn_out)

        # MLP with residual (at output resolution; TP inside self.mlp)
        if self.grad_checkpointing and self.training:
            mlp_out = checkpoint(self._mlp_forward, x, use_reentrant=False)
        else:
            mlp_out = self._mlp_forward(x)

        x = x + self.drop_path(mlp_out)

        return x
