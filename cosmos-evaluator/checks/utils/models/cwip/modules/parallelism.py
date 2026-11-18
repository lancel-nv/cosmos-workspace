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
Model Parallelism utilities for CWIP.

This module provides infrastructure for:
- Tensor Parallelism (TP): Split layer computations across GPUs
- Frame Parallelism (FP): Splits video frames across GPUs for CWIP's encoder. Each FP rank independently encodes a
  subset of frames; features are gathered after encoding.  Uses FP sub-groups derived from Megatron's CP infrastructure.
- Context Parallelism (CP): True context parallelism in PatchFormer (token sharding + ``all_gather`` of K/V plus SDPA).
  Splits spatial tokens within each image pair across CP ranks.  Uses CP sub-groups orthogonal to FP.
- Sequence Parallelism (SP): Split activation sequences for memory efficiency
- Megatron-backed TP linear layers with checkpoint compatibility

Megatron's ``context_parallel_size`` is set to ``fp_size * cp_size``.  This module creates FP and CP sub-groups from
the overall Megatron CP group.
"""

from __future__ import annotations

from collections import OrderedDict
import logging
from typing import Any, TypedDict, cast

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Megatron (optional): TP linear layers
# =============================================================================

# megatron is an optional dependency used for multi-GPU tensor/context/frame parallelism (both training and the
# multi-GPU inference paths in tools/inference.py); it is guarded by try/except (see HAS_MEGATRON below) and is not
# pinned in requirements.in, so tell Gazelle not to resolve these imports to a build dependency.
# gazelle:ignore megatron,megatron.core,megatron.core.model_parallel_config,megatron.core.tensor_parallel
# gazelle:ignore megatron.core.tensor_parallel.random
try:
    from megatron.core import parallel_state
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.core.tensor_parallel import (
        ColumnParallelLinear as MegatronColumnParallelLinear,
    )
    from megatron.core.tensor_parallel import (
        RowParallelLinear as MegatronRowParallelLinear,
    )

    HAS_MEGATRON = True
except ImportError:
    HAS_MEGATRON = False
    parallel_state = None
    MegatronColumnParallelLinear = None
    MegatronRowParallelLinear = None
    ModelParallelConfig = None

try:
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
except ImportError:
    model_parallel_cuda_manual_seed = None


class ModelParallelState(TypedDict):
    """Type hints for model parallelism global state."""

    initialized: bool
    tensor_parallel_size: int
    context_parallel_size: int  # Megatron CP size (= fp_size * cp_size)
    sequence_parallel: bool
    tensor_parallel_group: dist.ProcessGroup | None
    context_parallel_group: dist.ProcessGroup | None  # Megatron CP group
    data_parallel_group: dist.ProcessGroup | None
    frame_parallel_size: int
    frame_parallel_group: dist.ProcessGroup | None
    true_context_parallel_size: int
    true_context_parallel_group: dist.ProcessGroup | None


_DEFAULT_STATE: ModelParallelState = {
    "initialized": False,
    "tensor_parallel_size": 1,
    "context_parallel_size": 1,
    "sequence_parallel": False,
    "tensor_parallel_group": None,
    "context_parallel_group": None,
    "data_parallel_group": None,
    "frame_parallel_size": 1,
    "frame_parallel_group": None,
    "true_context_parallel_size": 1,
    "true_context_parallel_group": None,
}

# Global state for model parallelism
_MODEL_PARALLEL_STATE: ModelParallelState = cast(ModelParallelState, dict(_DEFAULT_STATE))


# =============================================================================
# Megatron Configuration (shared across parallel modules)
# =============================================================================

_MEGATRON_CONFIG: ModelParallelConfig | None = None


def get_megatron_config() -> ModelParallelConfig:
    """
    Get or create the Megatron ModelParallelConfig.

    This config is shared across all parallel layers and must be initialized after parallel state is set up.

    Returns:
        ModelParallelConfig instance for Megatron-LM layers

    Raises:
        RuntimeError: If Megatron-LM is not installed
    """
    global _MEGATRON_CONFIG

    if not HAS_MEGATRON:
        raise RuntimeError("Megatron-LM is required but not installed")

    if _MEGATRON_CONFIG is None:
        _MEGATRON_CONFIG = ModelParallelConfig(
            tensor_model_parallel_size=get_tensor_parallel_world_size(),
            pipeline_model_parallel_size=1,
            context_parallel_size=get_context_parallel_world_size(),
            sequence_parallel=is_sequence_parallel_enabled(),
        )

    return _MEGATRON_CONFIG


def megatron_init_method(tensor: torch.Tensor) -> torch.Tensor:
    """Xavier uniform initialization for Megatron layers."""
    return nn.init.xavier_uniform_(tensor)


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
) -> None:
    """
    Initialize model parallel groups.

    This function sets up the process groups for different parallelism strategies. Must be called after
    torch.distributed.init_process_group().

    Args:
        tensor_model_parallel_size: Number of GPUs for tensor parallelism
        context_parallel_size: Number of GPUs for context parallelism
        sequence_parallel: Whether to enable sequence parallelism (requires TP)

    Example:
        # 32 GPUs with TP=4, CP=8 → 1 DP replica
        initialize_model_parallel(tensor_model_parallel_size=4, context_parallel_size=8)

        # 64 GPUs with TP=4, CP=8 → 2 DP replicas
        initialize_model_parallel(tensor_model_parallel_size=4, context_parallel_size=8)
    """
    if not dist.is_initialized():
        logger.warning("torch.distributed not initialized, skipping model parallel initialization")
        return

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Validate configuration
    model_parallel_size = tensor_model_parallel_size * context_parallel_size
    if world_size % model_parallel_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by "
            f"tensor_parallel_size ({tensor_model_parallel_size}) x "
            f"context_parallel_size ({context_parallel_size}) = {model_parallel_size}"
        )

    data_parallel_size = world_size // model_parallel_size

    # Use Megatron's parallel_state when available; otherwise fall back to custom process groups.
    # parallel_state is imported (guarded) at module top level, so no per-call import is needed.
    if HAS_MEGATRON:
        _ps = parallel_state

        # Destroy any existing state
        if _MODEL_PARALLEL_STATE["initialized"]:
            _ps.destroy_model_parallel()

        # Initialize Megatron's model parallel
        _ps.initialize_model_parallel(
            tensor_model_parallel_size=tensor_model_parallel_size,
            context_parallel_size=context_parallel_size,
        )

        _MODEL_PARALLEL_STATE["tensor_parallel_group"] = _ps.get_tensor_model_parallel_group()
        _MODEL_PARALLEL_STATE["context_parallel_group"] = _ps.get_context_parallel_group()
        # Align with Imaginaire DDP (`parallel_model_wrapper`): use Megatron's "DP with CP" group.
        # Example rank layout (TP=PP=1, CP=2, 6 GPUs):
        #   Rank  dp_idx  cp_idx  Role
        #      0       0       0  batch replica 0, sequence shard 0
        #      1       0       1  batch replica 0, sequence shard 1
        #      2       1       0  batch replica 1, sequence shard 0
        #      3       1       1  batch replica 1, sequence shard 1
        #      4       2       0  batch replica 2, sequence shard 0
        #      5       2       1  batch replica 2, sequence shard 1
        # `with_context_parallel=False` yields DP groups {0,2,4} and {1,3,5} (same CP shard
        # index, different batches); `True` yields {0,1}, {2,3}, {4,5} (same batch replica, both CP
        # ranks that share weights). Intuition: True = fold CP into the same grad-sync team as
        # DDP/optimizer; False = DP only within a fixed CP shard.
        _MODEL_PARALLEL_STATE["data_parallel_group"] = _ps.get_data_parallel_group(with_context_parallel=True)

        # Set sequence parallel flag in Megatron
        if sequence_parallel:
            _ps.sequence_parallel = True

        # Initialize the CUDA RNG tracker required by Megatron's TP layers
        # for weight initialization (get_cuda_rng_tracker().fork()).
        if tensor_model_parallel_size > 1 and model_parallel_cuda_manual_seed is not None:
            model_parallel_cuda_manual_seed(42)

        logger.info("Using Megatron parallel_state for model parallelism")

        # Map Megatron CP group to frame-parallel scatter/gather until sync_from_megatron()
        # applies FP x true_CP subgroup layout. Without this, initialize_model_parallel() could
        # set initialized=True while frame_parallel_group stayed None and sync_from_megatron()
        # would no-op on its early return — breaking encoder FP gather/split.
        _MODEL_PARALLEL_STATE["frame_parallel_size"] = context_parallel_size
        _MODEL_PARALLEL_STATE["frame_parallel_group"] = _ps.get_context_parallel_group()
        _MODEL_PARALLEL_STATE["true_context_parallel_size"] = 1
        _MODEL_PARALLEL_STATE["true_context_parallel_group"] = None

    else:
        logger.info("Megatron not available, using custom parallel groups")

        # Create custom process groups
        # TP groups: ranks that share the same layer computation
        # Each TP group has tensor_model_parallel_size ranks
        tp_groups = []
        for i in range(world_size // tensor_model_parallel_size):
            ranks = list(range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                _MODEL_PARALLEL_STATE["tensor_parallel_group"] = group
            tp_groups.append(group)

        # CP groups: ranks that share the same sequence
        # Each CP group has context_parallel_size ranks
        cp_groups = []
        for i in range(world_size // context_parallel_size):
            ranks = list(range(i * context_parallel_size, (i + 1) * context_parallel_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                _MODEL_PARALLEL_STATE["context_parallel_group"] = group
            cp_groups.append(group)

        # Same as Megatron branch: CP group serves as FP group when true_CP is not split out.
        _MODEL_PARALLEL_STATE["frame_parallel_size"] = context_parallel_size
        _MODEL_PARALLEL_STATE["frame_parallel_group"] = _MODEL_PARALLEL_STATE["context_parallel_group"]
        _MODEL_PARALLEL_STATE["true_context_parallel_size"] = 1
        _MODEL_PARALLEL_STATE["true_context_parallel_group"] = None

        # DP groups: ranks that process different data
        dp_groups = []
        for i in range(model_parallel_size):
            ranks = list(range(i, world_size, model_parallel_size))
            group = dist.new_group(ranks)
            if rank in ranks:
                _MODEL_PARALLEL_STATE["data_parallel_group"] = group
            dp_groups.append(group)

    _MODEL_PARALLEL_STATE["initialized"] = True
    _MODEL_PARALLEL_STATE["tensor_parallel_size"] = tensor_model_parallel_size
    _MODEL_PARALLEL_STATE["context_parallel_size"] = context_parallel_size
    _MODEL_PARALLEL_STATE["sequence_parallel"] = sequence_parallel

    logger.info(
        "Model parallel initialized: TP=%s, CP=%s, SP=%s, DP=%s, world_size=%s",
        tensor_model_parallel_size,
        context_parallel_size,
        sequence_parallel,
        data_parallel_size,
        world_size,
    )


def sync_from_megatron(
    frame_parallel_size: int = 0,
    true_context_parallel_size: int = 1,
) -> None:
    """Sync parallelism state from Megatron's ``parallel_state`` into this module.

    Populates ``_MODEL_PARALLEL_STATE`` with TP, FP, CP, and the data-parallel process group
    (``get_data_parallel_group(with_context_parallel=True)``, same as Imaginaire DDP) so that the rest of the
    Cosmos Evaluator codebase has a single source of truth via the ``get_*`` helpers.

    Safe to call multiple times: refreshes TP/CP/DP from Megatron and (re)assigns frame-parallel groups.
    ``_create_fp_cp_subgroups`` runs at most once per process (when true context parallel size > 1).

    The idempotent refresh mainly fixes ``deployment/tools/inference.py`` ordering (``initialize_model_parallel`` then
    sync). Other top-level launchers may initialize Megatron before model construction; encoder FP edge cases are a
    separate concern.

    Args:
        frame_parallel_size: Number of GPUs for frame parallelism.
            0 (default) means "use the entire Megatron CP group as FP".
        true_context_parallel_size: Number of GPUs for true context parallelism
            (true CP / gather attention in PatchFormer).  Defaults to 1 (disabled).
    """
    if not HAS_MEGATRON:
        return
    # parallel_state is imported (guarded) at module top level, so no per-call import is needed.
    _ps = parallel_state

    if not _ps.is_initialized():
        return

    _MODEL_PARALLEL_STATE["tensor_parallel_size"] = _ps.get_tensor_model_parallel_world_size()
    _MODEL_PARALLEL_STATE["tensor_parallel_group"] = _ps.get_tensor_model_parallel_group()
    megatron_cp_size = _ps.get_context_parallel_world_size()
    megatron_cp_group = _ps.get_context_parallel_group()
    _MODEL_PARALLEL_STATE["context_parallel_size"] = megatron_cp_size
    _MODEL_PARALLEL_STATE["context_parallel_group"] = megatron_cp_group
    _MODEL_PARALLEL_STATE["sequence_parallel"] = getattr(_ps, "sequence_parallel", False)
    _MODEL_PARALLEL_STATE["data_parallel_group"] = _ps.get_data_parallel_group(with_context_parallel=True)

    fp_size = frame_parallel_size if frame_parallel_size > 0 else megatron_cp_size
    cp_size = true_context_parallel_size

    if fp_size * cp_size != megatron_cp_size:
        raise ValueError(
            f"frame_parallel_size ({fp_size}) x context_parallel_size ({cp_size}) "
            f"= {fp_size * cp_size} but Megatron CP size = {megatron_cp_size}"
        )

    _MODEL_PARALLEL_STATE["frame_parallel_size"] = fp_size
    _MODEL_PARALLEL_STATE["true_context_parallel_size"] = cp_size

    if cp_size == 1:
        # FP group IS the Megatron CP group; no separate CP sub-group needed
        _MODEL_PARALLEL_STATE["frame_parallel_group"] = megatron_cp_group
        _MODEL_PARALLEL_STATE["true_context_parallel_group"] = None
    elif _MODEL_PARALLEL_STATE.get("true_context_parallel_group") is None:
        _create_fp_cp_subgroups(fp_size, cp_size)
    # else: subgroups already created; keep existing FP/CP process groups

    _MODEL_PARALLEL_STATE["initialized"] = True

    # Ensure Megatron CUDA RNG tracker is seeded (needed by TP layers)
    if _MODEL_PARALLEL_STATE["tensor_parallel_size"] > 1 and model_parallel_cuda_manual_seed is not None:
        model_parallel_cuda_manual_seed(42)

    logger.info(
        "Synced from Megatron parallel_state: TP=%s, FP=%s, CP=%s (Megatron CP=%s)",
        _MODEL_PARALLEL_STATE["tensor_parallel_size"],
        fp_size,
        cp_size,
        megatron_cp_size,
    )


def _create_fp_cp_subgroups(fp_size: int, cp_size: int) -> None:
    """Create FP and CP sub-groups from the Megatron CP group.

    Within the Megatron CP group of size ``fp_size * cp_size``, ranks are
    indexed as ``megatron_cp_rank = fp_rank * cp_size + cp_rank``.

    - FP sub-group (size ``fp_size``): ranks sharing the same ``cp_rank``
    - CP sub-group (size ``cp_size``): ranks sharing the same ``fp_rank``

    Must be called collectively by all ranks (``dist.new_group`` is collective).
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    tp_size = _MODEL_PARALLEL_STATE["tensor_parallel_size"]
    megatron_cp_size = fp_size * cp_size

    num_cp_supergroups = world_size // (tp_size * megatron_cp_size)

    my_fp_group = None
    my_cp_group = None

    # FP sub-groups: ranks with the same cp_rank within each Megatron CP group
    for sg in range(num_cp_supergroups):
        for tp_r in range(tp_size):
            for cp_r in range(cp_size):
                fp_ranks = [(sg * megatron_cp_size + fp_r * cp_size + cp_r) * tp_size + tp_r for fp_r in range(fp_size)]
                group = dist.new_group(fp_ranks)
                if rank in fp_ranks:
                    my_fp_group = group

    # CP sub-groups: ranks with the same fp_rank within each Megatron CP group
    for sg in range(num_cp_supergroups):
        for tp_r in range(tp_size):
            for fp_r in range(fp_size):
                cp_ranks = [(sg * megatron_cp_size + fp_r * cp_size + cp_r) * tp_size + tp_r for cp_r in range(cp_size)]
                group = dist.new_group(cp_ranks)
                if rank in cp_ranks:
                    my_cp_group = group

    _MODEL_PARALLEL_STATE["frame_parallel_group"] = my_fp_group
    _MODEL_PARALLEL_STATE["true_context_parallel_group"] = my_cp_group

    logger.info(
        "Created FP/CP sub-groups: FP=%s, CP=%s, rank=%s, fp_group=%s, cp_group=%s",
        fp_size,
        cp_size,
        rank,
        my_fp_group,
        my_cp_group,
    )


def get_tensor_parallel_group() -> dist.ProcessGroup | None:
    """Get the tensor parallel process group."""
    return _MODEL_PARALLEL_STATE["tensor_parallel_group"]


def get_tensor_parallel_world_size() -> int:
    """Get the tensor parallel world size."""
    return _MODEL_PARALLEL_STATE["tensor_parallel_size"]


def get_context_parallel_world_size() -> int:
    """Get the Megatron context parallel world size (= fp * cp)."""
    return _MODEL_PARALLEL_STATE["context_parallel_size"]


# ---- Frame Parallelism (FP) sub-group accessors ----


def get_frame_parallel_group() -> dist.ProcessGroup | None:
    """Get the frame parallel sub-group (for frame scatter/gather)."""
    return _MODEL_PARALLEL_STATE["frame_parallel_group"]


def get_frame_parallel_world_size() -> int:
    """Get the frame parallel world size."""
    return _MODEL_PARALLEL_STATE["frame_parallel_size"]


def get_frame_parallel_rank() -> int:
    """Get rank within the frame parallel sub-group."""
    group = get_frame_parallel_group()
    if group is None:
        return 0
    return dist.get_rank(group)


class _AllGatherCatBatchDim(torch.autograd.Function):
    """All-gather each rank's batch slice and ``torch.cat`` on dim 0 with autograd.

    A plain ``dist.all_gather`` into fresh buffers plus ``torch.cat`` is not linked to the input in autograd, so
    upstream gradients would not flow correctly after encoder FP without this. Backward returns this rank's contiguous
    slice of ``grad_output`` (same layout as forward split in :func:`split_tensor_for_fp`).
    """

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        world_size = dist.get_world_size(group)
        if world_size <= 1:
            return input_
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(tensor_list, input_.contiguous(), group=group)
        return torch.cat(tensor_list, dim=0)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        world_size = dist.get_world_size(ctx.group)
        if world_size <= 1:
            return grad_output, None
        rank = dist.get_rank(ctx.group)
        dim_size = grad_output.size(0)
        chunk_size = dim_size // world_size
        chunks = torch.split(grad_output, chunk_size, dim=0)
        return chunks[rank].contiguous(), None


def all_gather_cat_batch_dim_autograd(tensor: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """All-gather on batch dimension (dim 0) with autograd support."""
    if group is None:
        return tensor
    return _AllGatherCatBatchDim.apply(tensor, group)


def gather_optional_tensor(tensor: torch.Tensor | None, group: dist.ProcessGroup | None) -> torch.Tensor | None:
    """All-gather a tensor along dim 0, preserving ``None``."""
    if tensor is None:
        return None
    return all_gather_cat_batch_dim_autograd(tensor.contiguous(), group)


def gather_optional_tensor_list(
    tensor_list: list[torch.Tensor | None] | None, group: dist.ProcessGroup | None
) -> list[torch.Tensor | None] | None:
    """All-gather each tensor in ``tensor_list`` along dim 0, preserving ``None`` values."""
    if tensor_list is None:
        return None
    return [gather_optional_tensor(t, group) for t in tensor_list]


def split_tensor_for_fp(
    t: torch.Tensor,
    fp_size: int | None = None,
    *,
    return_indices: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, int, int]:
    """Split frame batches across FP ranks, right-padding batch dim when needed.

    Args:
        t: Full batch along dim 0 (frames / pairs).
        fp_size: Frame-parallel world size. If ``None``, uses :func:`get_frame_parallel_world_size`.
        return_indices: If True, also return half-open ``[real_start, real_end)`` indices into the **original** batch
            (before right-padding). Use this to map each rank's shard back to global frame indices. If the shard is
            padding-only, ``real_start == real_end``.

    Returns:
        The local shard ``(N_local, ...)``. If ``return_indices`` is True, returns ``(shard, real_start, real_end)``
        with ``real_*`` in original-batch coordinates. If ``fp_size <= 1``, returns ``t`` or ``(t, 0, n)`` with
        ``n = t.shape[0]``.
    """
    if fp_size is None:
        fp_size = get_frame_parallel_world_size()
    if fp_size <= 1:
        n = int(t.shape[0])
        if return_indices:
            return t, 0, n
        return t

    fp_rank = get_frame_parallel_rank()
    original_batch_size = int(t.shape[0])
    if original_batch_size % fp_size != 0:
        pad_size = fp_size - (original_batch_size % fp_size)
        # Rank-agnostic batch padding (``F.pad`` 8-tuple is only valid for 4D (N,C,H,W) inputs).
        pad_shape = (pad_size, *t.shape[1:])
        pad_tensor = torch.zeros(pad_shape, dtype=t.dtype, device=t.device)
        t = torch.cat([t, pad_tensor], dim=0)

    padded_batch_size = int(t.shape[0])
    frames_per_rank = padded_batch_size // fp_size
    start_idx = fp_rank * frames_per_rank
    end_idx = start_idx + frames_per_rank
    out = t[start_idx:end_idx]
    real_start = start_idx
    real_end = min(end_idx, original_batch_size)
    if return_indices:
        return out, real_start, real_end
    return out


def gather_features_from_fp(
    output: dict[str, torch.Tensor | list | None],
    fp_size: int | None = None,
) -> dict[str, torch.Tensor | list | None]:
    """Gather encoded outputs across FP ranks, preserving non-tensor entries.

    Args:
        output: Per-rank encoder / head outputs to all-gather along batch dim 0.
        fp_size: Expected FP world size for logging consistency. If ``None``, uses
            :func:`get_frame_parallel_world_size`.
    """
    if fp_size is None:
        fp_size = get_frame_parallel_world_size()
    if fp_size <= 1:
        return output

    fp_group = get_frame_parallel_group()
    if fp_group is None:
        return output

    fp_group_world = dist.get_world_size(group=fp_group)
    if fp_group_world != fp_size:
        logger.warning(
            "[CWIP] frame_parallel_group world size (%s) != fp_size (%s); using group world size "
            "for all_gather (check sync_from_megatron / initialize_model_parallel order).",
            fp_group_world,
            fp_size,
        )

    gathered_output: dict[str, torch.Tensor | list | None] = {}
    for key, value in output.items():
        if isinstance(value, list):
            gathered_output[key] = gather_optional_tensor_list(value, fp_group)
        elif isinstance(value, torch.Tensor):
            gathered_output[key] = gather_optional_tensor(value, fp_group)
        else:
            gathered_output[key] = value
    return gathered_output


# ---- True Context Parallelism sub-group accessors ----


def get_true_context_parallel_group() -> dist.ProcessGroup | None:
    """Get the true CP sub-group (PatchFormer token sharding + gather attention)."""
    return _MODEL_PARALLEL_STATE["true_context_parallel_group"]


def get_true_context_parallel_world_size() -> int:
    """Get the true context parallel world size."""
    return _MODEL_PARALLEL_STATE["true_context_parallel_size"]


def get_true_context_parallel_rank() -> int:
    """Rank within the true context-parallel sub-group."""
    group = get_true_context_parallel_group()
    if group is None:
        return 0
    return dist.get_rank(group)


# =============================================================================
# True CP sequence sharding + attention flow (PatchFormer)
#
# Use these helpers together, in order:
#   1) prepare_true_cp_sequence_shard(seq_a, seq_b, grid_h, grid_w)
#      -> local shards + metadata for this CP rank.
#   2) slice_rope_freq_along_sequence(freq, meta["s_lo"], meta["s_hi"])
#      -> make RoPE positions match the local shard.
#   3) sdpa_or_true_cp_attention(q_local, k_local, v_local, ..., cp_group)
#      -> gather K/V across CP ranks and run SDPA for local queries.
#   4) gather_trim_true_cp_sequence(x_local, meta)
#      -> reconstruct full sequence and trim right-padding.
# =============================================================================


def prepare_true_cp_sequence_shard(
    seq_a: torch.Tensor,
    seq_b: torch.Tensor,
    grid_h: int,
    grid_w: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
    """
    Shard two matching token sequences along dim 1 for true context parallelism.

    Right-pads the sequence length to a multiple of the true-CP world size with zeros, then each rank keeps a
    contiguous slice ``[cp_rank * S_local : (cp_rank + 1) * S_local]``. Typical use: two aligned spatial token rows
    ``(B, H*W, C)`` (e.g. PatchFormer camera / world) with ``grid_h * grid_w == S``.

    Args:
        seq_a: First token tensor, shape ``(B, S, C_a)``.
        seq_b: Second token tensor, shape ``(B, S, C_b)``. Must match ``seq_a`` in
            batch ``B`` and sequence length ``S``.
        grid_h: Spatial height in tokens; with ``grid_w``, used only to validate ``S == grid_h * grid_w``.
        grid_w: Spatial width in tokens; with ``grid_h``, used only to validate ``S == grid_h * grid_w``.

    Returns:
        Tuple ``(shard_a, shard_b, meta)``.

        * ``shard_a``: ``(B, S_local, C_a)`` local slice of ``seq_a`` when sharding is
          active; unchanged ``seq_a`` when inactive.
        * ``shard_b``: ``(B, S_local, C_b)`` local slice of ``seq_b`` when sharding is
          active; unchanged ``seq_b`` when inactive.
        * ``meta``: ``None`` if sharding is inactive (distributed not initialized,
          true-CP world size ``<= 1``, or no true-CP process group). Otherwise a dict
          suitable for :func:`gather_trim_true_cp_sequence` and RoPE slicing, with keys:

          - ``"active"``: ``True``
          - ``"cp_group"``: true-CP :class:`~torch.distributed.ProcessGroup`
          - ``"cp_size"``: int, world size of that group
          - ``"s_unpadded"``: original ``S`` before padding
          - ``"s_padded"``: ``S`` after right-padding
          - ``"s_lo"``, ``"s_hi"``: index range of this rank's slice in the padded sequence
          - ``"s_local"``: ``s_hi - s_lo`` (``S_pad // cp_size``)

    Raises:
        ValueError: If ``seq_a`` and ``seq_b`` differ in ``B`` or ``S``, or if ``S != grid_h * grid_w``.
    """
    if not dist.is_initialized():
        return seq_a, seq_b, None

    cp_size = get_true_context_parallel_world_size()
    cp_group = get_true_context_parallel_group()
    if cp_size <= 1 or cp_group is None:
        return seq_a, seq_b, None

    B, S, _ = seq_a.shape
    if seq_b.shape[0] != B or seq_b.shape[1] != S:
        raise ValueError(f"True CP shard: seq_a {seq_a.shape} and seq_b {seq_b.shape} must match in batch and length")
    S_spatial = grid_h * grid_w
    if S != S_spatial:
        raise ValueError(f"True CP shard: sequence length {S} must equal grid_h*grid_w={S_spatial}")

    pad_tokens = (cp_size - (S % cp_size)) % cp_size
    if pad_tokens:
        z_a = seq_a.new_zeros(B, pad_tokens, seq_a.shape[-1])
        z_b = seq_b.new_zeros(B, pad_tokens, seq_b.shape[-1])
        seq_a = torch.cat([seq_a, z_a], dim=1)
        seq_b = torch.cat([seq_b, z_b], dim=1)

    S_pad = seq_a.shape[1]
    s_local = S_pad // cp_size
    cp_rank = get_true_context_parallel_rank()
    lo = cp_rank * s_local
    hi = lo + s_local

    meta: dict[str, Any] = {
        "active": True,
        "cp_group": cp_group,
        "cp_size": cp_size,
        "s_unpadded": S,
        "s_padded": S_pad,
        "s_lo": lo,
        "s_hi": hi,
        "s_local": s_local,
    }
    return seq_a[:, lo:hi, :].contiguous(), seq_b[:, lo:hi, :].contiguous(), meta


def gather_trim_true_cp_sequence(x: torch.Tensor, meta: dict[str, Any] | None) -> torch.Tensor:
    """
    All-gather sequence shards along dim 1 and trim true-CP padding tokens.

    Inverts :func:`prepare_true_cp_sequence_shard` on the sequence axis: collects each rank's local ``(B, S_local, C)``
    shard into ``(B, S_padded, C)``, then drops padding so the result has length ``s_unpadded`` from ``meta``.

    Args:
        x: Local tensor, shape ``(B, S_local, C)`` when ``meta`` is not ``None``;
            arbitrary shape when ``meta`` is ``None`` (returned unchanged).
        meta: Output metadata from :func:`prepare_true_cp_sequence_shard`, or ``None``
            for a no-op (single-GPU / no true CP).

    Returns:
        Tensor of shape ``(B, s_unpadded, C)`` when ``meta`` is set; otherwise ``x``
        unchanged.
    """
    if meta is None:
        return x
    cp_size = meta["cp_size"]
    cp_group = meta["cp_group"]
    gathered = [torch.empty_like(x) for _ in range(cp_size)]
    dist.all_gather(gathered, x.contiguous(), group=cp_group)
    full = torch.cat(gathered, dim=1)
    su = meta["s_unpadded"]
    return full[:, :su, :].contiguous()


def slice_rope_freq_along_sequence(freq: torch.Tensor, s_lo: int, s_hi: int) -> torch.Tensor:
    """
    Slice RoPE frequency cache along the token (sequence) dimension.

    Use the same ``s_lo`` / ``s_hi`` as in the metadata from :func:`prepare_true_cp_sequence_shard` so rotary
    positions match the local token shard. Different ``Rope2D`` builds use different layouts; this helper supports the
    common shapes before :meth:`apply_rotary_emb` expands dimensions.

    Args:
        freq: Rotary frequency buffer, one of:

            - ``ndim == 4``: ``(1, 1, seq, rot_dim)``
            - ``ndim == 3``: ``(1, seq, rot_dim)``
            - ``ndim == 2``: ``(seq, rot_dim)``
        s_lo: Start index along the sequence axis (inclusive).
        s_hi: End index along the sequence axis (exclusive).

    Returns:
        Sliced ``freq``, same ``ndim`` as the input, contiguous, with sequence length
        ``s_hi - s_lo``.

    Raises:
        ValueError: If ``freq.ndim`` is not 2, 3, or 4.
    """
    if freq.ndim == 4:
        return freq[:, :, s_lo:s_hi, :].contiguous()
    if freq.ndim == 3:
        return freq[:, s_lo:s_hi, :].contiguous()
    if freq.ndim == 2:
        return freq[s_lo:s_hi, :].contiguous()
    raise ValueError(f"RoPE freq has unsupported ndim {freq.ndim}, shape {tuple(freq.shape)}")


def sdpa_or_true_cp_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    cp_group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """
    SDPA, or true context-parallel attention when ``cp_group`` is set and ``dist.get_world_size(cp_group) > 1``
    (group size is the only gate; not global state).

    With an active true-CP group, all-gathers K/V on ``cp_group`` (concat along seq dim 2 in rank order), then runs SDPA
    with ``attn_mask=None``. True CP does not apply ``attn_mask``; if a mask is passed while true CP is active, it is
    ignored.

    Args:
        q: Queries, shape ``(B, H, S_q, D)``.
        k: Keys, shape ``(B, H, S_kv, D)``. Under true CP, ``S_kv`` is the local shard length.
        v: Values, shape ``(B, H, S_kv, D)``, matching ``k``.
        scale: Attention scale passed to SDPA (typically ``1 / sqrt(D)``).
        attn_mask: Used only for the non-true-CP ``F.scaled_dot_product_attention`` branch.
        dropout_p: Dropout probability for SDPA (often ``0.0`` at inference).
        cp_group: True context-parallel process group, or ``None`` to skip CP branches.

    Returns:
        Attention output, shape ``(B, H, S_q, D)``.
    """
    if cp_group is None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale)
    cp_size = dist.get_world_size(cp_group)
    if cp_size <= 1:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=scale)

    def gather_heads_seq(t: torch.Tensor) -> torch.Tensor:
        chunks = [torch.empty_like(t) for _ in range(cp_size)]
        dist.all_gather(chunks, t.contiguous(), group=cp_group)
        return torch.cat(chunks, dim=2)

    k_full = gather_heads_seq(k)
    v_full = gather_heads_seq(v)
    return F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=None, dropout_p=dropout_p, scale=scale)


def get_tensor_parallel_rank() -> int:
    """Get the tensor parallel rank."""
    group = get_tensor_parallel_group()
    if group is None:
        return 0
    return dist.get_rank(group)


def get_tensor_parallel_world_size_and_rank_for_inference_viz() -> tuple[int, int]:
    """Return ``(tensor_parallel_world_size, tensor_parallel_rank)`` for inference visualization gating.

    Prefer Megatron's live ``parallel_state`` when initialized. :func:`sync_from_megatron` may run from the model
    ``__init__`` before ``parallel_state`` is ready; it then returns early and leaves ``_MODEL_PARALLEL_STATE`` at
    defaults (TP size 1). Both ranks would incorrectly run duplicate PCA/NVENC. Reading Megatron here matches the real
    TP group used by the forward pass.

    Returns:
        ``(tp_world_size, tp_rank)`` within the tensor-parallel group.
    """
    if parallel_state is not None:
        try:
            if parallel_state.is_initialized():
                return (
                    parallel_state.get_tensor_model_parallel_world_size(),
                    parallel_state.get_tensor_model_parallel_rank(),
                )
        except RuntimeError as e:
            # Megatron often raises RuntimeError when parallel groups are not fully built yet.
            # Do not catch AttributeError/KeyError/etc. here — those likely indicate API misuse or bugs.
            logger.warning(
                "Megatron parallel_state query failed for inference viz TP gating; using local TP state: %s",
                e,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
    return (get_tensor_parallel_world_size(), get_tensor_parallel_rank())


def is_sequence_parallel_enabled() -> bool:
    """Check if sequence parallelism is enabled."""
    return _MODEL_PARALLEL_STATE["sequence_parallel"]


# =============================================================================
# Tensor Parallel Linear Layers
# =============================================================================


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        # All-reduce gradients across tensor parallel group
        dist.all_reduce(grad_output, group=ctx.group)
        return grad_output, None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        dist.all_reduce(input_, group=group)
        return input_

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, dim: int, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.dim = dim
        ctx.group = group

        world_size = dist.get_world_size(group)
        if world_size == 1:
            return input_

        # Gather from all ranks
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        dist.all_gather(tensor_list, input_, group=group)
        output = torch.cat(tensor_list, dim=dim)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        world_size = dist.get_world_size(ctx.group)
        if world_size == 1:
            return grad_output, None, None

        rank = dist.get_rank(ctx.group)
        dim_size = grad_output.size(ctx.dim)
        chunk_size = dim_size // world_size

        # Split and return own chunk
        chunks = torch.split(grad_output, chunk_size, dim=ctx.dim)
        return chunks[rank].contiguous(), None, None


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and scatter to model parallel region."""

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, dim: int, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.dim = dim
        ctx.group = group

        world_size = dist.get_world_size(group)
        if world_size == 1:
            return input_

        rank = dist.get_rank(group)
        dim_size = input_.size(dim)
        chunk_size = dim_size // world_size

        # Split and return own chunk
        chunks = torch.split(input_, chunk_size, dim=dim)
        return chunks[rank].contiguous()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        world_size = dist.get_world_size(ctx.group)
        if world_size == 1:
            return grad_output, None, None

        # Gather gradients from all ranks
        tensor_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(tensor_list, grad_output, group=ctx.group)
        output = torch.cat(tensor_list, dim=ctx.dim)
        return output, None, None


def copy_to_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """Copy input to tensor parallel region (identity forward, all-reduce backward)."""
    group = get_tensor_parallel_group()
    if group is None:
        return input_
    return _CopyToModelParallelRegion.apply(input_, group)


def reduce_from_tensor_parallel_region(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce input from tensor parallel region."""
    group = get_tensor_parallel_group()
    if group is None:
        return input_
    return _ReduceFromModelParallelRegion.apply(input_, group)


def gather_from_tensor_parallel_region(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Gather input from tensor parallel region."""
    group = get_tensor_parallel_group()
    if group is None:
        return input_
    return _GatherFromModelParallelRegion.apply(input_, dim, group)


def scatter_to_tensor_parallel_region(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Scatter input to tensor parallel region."""
    group = get_tensor_parallel_group()
    if group is None:
        return input_
    return _ScatterToModelParallelRegion.apply(input_, dim, group)


# =============================================================================
# Megatron-backed TP Linear Layers (with fallback)
# =============================================================================


class TPColumnParallelLinear(nn.Module):
    """
    Column-parallel Linear layer using Megatron-LM.

    Uses Megatron's optimized ColumnParallelLinear when available, falls back to custom implementation otherwise.

    Implements custom state_dict/load_state_dict for checkpoint compatibility:
    - Saves/loads with `weight` and `bias` keys regardless of internal implementation
    - Megatron layers internally use `_megatron_layer.weight` but expose as `weight`
    - Auto-shards full (unsharded) weights on load when TP > 1

    Supported checkpoint transitions:
    - TP=1 → TP=N:  Supported. _load_from_state_dict auto-shards full weights.
    - TP=N → TP=N:  Supported. Shapes match, direct copy.
    - TP=N → TP=1:  NOT supported. Each rank saves its shard under the flat key;
                     loading with TP=1 would get only one shard. To support this,
                     add an all-gather in _load_from_state_dict (when incoming
                     weight is smaller than expected), or expose weights as DTensor
                     in state_dict so DCP can reshard automatically.

    This is the shared implementation used by both parallel_pe.py and parallel_patchformer.py.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self._has_bias = bias

        tp_size = get_tensor_parallel_world_size()

        if out_features % tp_size != 0:
            raise ValueError(f"out_features ({out_features}) must be divisible by tp_size ({tp_size})")

        self.out_features_per_rank = out_features // tp_size

        if HAS_MEGATRON and tp_size > 1:
            # Use Megatron's optimized implementation
            config = get_megatron_config()
            self._megatron_layer = MegatronColumnParallelLinear(
                in_features,
                out_features,
                config=config,
                init_method=megatron_init_method,
                bias=bias,
                gather_output=gather_output,
            )
            self._use_megatron = True
            logger.debug("Using Megatron ColumnParallelLinear: %s -> %s", in_features, out_features)
        else:
            # Fallback: custom implementation with standard names
            self._use_megatron = False
            self.weight = nn.Parameter(torch.empty(self.out_features_per_rank, in_features))
            if bias:
                self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))
            else:
                self.register_parameter("bias", None)
            self._init_weights()

    def _init_weights(self) -> None:
        if not self._use_megatron:
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __getattr__(self, name: str) -> Any:
        """Proxy ``weight`` and ``bias`` attribute access to the Megatron sub-layer.

        DCP's ``get_model_state_dict`` resolves fully-qualified names via ``getattr``.  When Megatron is active the
        real parameters live inside ``_megatron_layer``, so we redirect here to keep the flat attribute namespace that
        DCP and ``state_dict()`` expect.

        Args:
            name: Attribute name to look up.

        Returns:
            The ``weight`` or ``bias`` parameter from ``_megatron_layer`` when
            Megatron is active, otherwise falls back to ``nn.Module.__getattr__``.
        """
        if name in ("weight", "bias") and self.__dict__.get("_use_megatron", False):
            megatron = self._modules.get("_megatron_layer")
            if megatron is not None:
                return getattr(megatron, name)
        return super().__getattr__(name)

    def _save_to_state_dict(self, destination: dict[str, torch.Tensor], prefix: str, keep_vars: bool) -> None:
        """Save Megatron weights under flat names for checkpoint compatibility.

        Maps ``_megatron_layer.weight`` → ``{prefix}weight`` so checkpoints always use the same key format regardless
        of whether TP is enabled.

        Args:
            destination: Dict to write state entries into.
            prefix: Key prefix for this module (e.g. ``"encoder.resblocks.0.mlp.c_fc."``).
            keep_vars: If True, return ``nn.Parameter`` objects; otherwise return ``.data``.
        """
        if self._use_megatron:
            destination[prefix + "weight"] = (
                self._megatron_layer.weight if keep_vars else self._megatron_layer.weight.data
            )
            if self._megatron_layer.bias is not None:
                destination[prefix + "bias"] = (
                    self._megatron_layer.bias if keep_vars else self._megatron_layer.bias.data
                )
        else:
            super()._save_to_state_dict(destination, prefix, keep_vars)

    def state_dict(  # type: ignore[override]
        self, *args: Any, destination: OrderedDict | None = None, prefix: str = "", keep_vars: bool = False
    ) -> OrderedDict:
        """Return state dict with flat key names, skipping ``_megatron_layer`` child traversal.

        When Megatron is active, ``_save_to_state_dict`` already emits ``weight``/``bias`` from the Megatron sub-layer.
        We skip the default child recursion to prevent duplicate ``_megatron_layer.weight`` keys from appearing.

        Args:
            destination: Optional pre-existing dict to write into.
            prefix: Key prefix for this module.
            keep_vars: If True, return ``nn.Parameter`` objects; otherwise return ``.data``.

        Returns:
            OrderedDict mapping flat parameter names to tensors.
        """
        if destination is None:
            destination = OrderedDict()
        self._save_to_state_dict(destination, prefix, keep_vars)
        if not self._use_megatron:
            for name, module in self._modules.items():
                if module is not None:
                    module.state_dict(*args, destination=destination, prefix=prefix + name + ".", keep_vars=keep_vars)
        return destination

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Load checkpoint tensors, auto-sharding full weights for ColumnParallel (dim 0).

        When the incoming weight is larger than the expected sharded shape, it is automatically sliced along dim 0 for
        the current TP rank.  After loading, megatron-prefixed keys are injected into ``state_dict`` so the child
        ``_megatron_layer`` does not report them as missing.

        Args:
            state_dict: Checkpoint state dict (may be mutated with injected keys).
            prefix: Key prefix for this module.
            local_metadata: Unused; required by PyTorch API.
            strict: If True, missing keys are reported.
            missing_keys: List to append missing key names to.
            unexpected_keys: List to append unexpected key names to.
            error_msgs: List to append error messages to.
        """
        if self._use_megatron:
            weight_key = prefix + "weight"
            bias_key = prefix + "bias"

            if weight_key in state_dict:
                w = state_dict[weight_key]
                expected = self._megatron_layer.weight.data.shape
                if w.shape != expected:
                    tp_rank = get_tensor_parallel_rank()
                    per_rank = expected[0]
                    w = w[tp_rank * per_rank : (tp_rank + 1) * per_rank]
                self._megatron_layer.weight.data.copy_(w)
            elif strict:
                missing_keys.append(weight_key)

            if self._has_bias:
                if bias_key in state_dict:
                    b = state_dict[bias_key]
                    expected_b = self._megatron_layer.bias.data.shape
                    if b.shape != expected_b:
                        tp_rank = get_tensor_parallel_rank()
                        per_rank = expected_b[0]
                        b = b[tp_rank * per_rank : (tp_rank + 1) * per_rank]
                    self._megatron_layer.bias.data.copy_(b)
                elif strict:
                    missing_keys.append(bias_key)

            # Inject megatron-prefixed keys so the child's _load_from_state_dict
            # finds them and doesn't report missing keys.
            megatron_prefix = prefix + "_megatron_layer."
            state_dict[megatron_prefix + "weight"] = self._megatron_layer.weight.data
            if self._megatron_layer.bias is not None:
                state_dict[megatron_prefix + "bias"] = self._megatron_layer.bias.data
        else:
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_megatron:
            output, _ = self._megatron_layer(x)
            return output
        else:
            # Copy to TP region (identity forward, all-reduce backward)
            x = copy_to_tensor_parallel_region(x)

            # Local computation
            output = F.linear(x, self.weight, self.bias)

            # Optionally gather outputs from all ranks
            if self.gather_output:
                output = gather_from_tensor_parallel_region(output, dim=-1)

            return output


class TPRowParallelLinear(nn.Module):
    """
    Row-parallel Linear layer using Megatron-LM.

    Uses Megatron's optimized RowParallelLinear when available, falls back to custom implementation otherwise.

    Implements custom state_dict/load_state_dict for checkpoint compatibility:
    - Saves/loads with `weight` and `bias` keys regardless of internal implementation
    - Auto-shards full (unsharded) weights on load when TP > 1

    See TPColumnParallelLinear docstring for supported checkpoint transitions.
    - Megatron layers internally use `_megatron_layer.weight` but expose as `weight`

    This is the shared implementation used by both parallel_pe.py and parallel_patchformer.py.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self._has_bias = bias

        tp_size = get_tensor_parallel_world_size()

        if in_features % tp_size != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by tp_size ({tp_size})")

        self.in_features_per_rank = in_features // tp_size

        if HAS_MEGATRON and tp_size > 1:
            # Use Megatron's optimized implementation
            config = get_megatron_config()
            self._megatron_layer = MegatronRowParallelLinear(
                in_features,
                out_features,
                config=config,
                init_method=megatron_init_method,
                bias=bias,
                input_is_parallel=input_is_parallel,
                skip_bias_add=False,
            )
            self._use_megatron = True
            logger.debug("Using Megatron RowParallelLinear: %s -> %s", in_features, out_features)
        else:
            # Fallback: custom implementation with standard names
            self._use_megatron = False
            self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_rank))
            if bias:
                self.bias = nn.Parameter(torch.empty(out_features))
            else:
                self.register_parameter("bias", None)
            self._init_weights()

    def _init_weights(self) -> None:
        if not self._use_megatron:
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def __getattr__(self, name: str) -> Any:
        """Proxy ``weight`` and ``bias`` attribute access to the Megatron sub-layer.

        See :meth:`TPColumnParallelLinear.__getattr__` for rationale.

        Args:
            name: Attribute name to look up.

        Returns:
            The parameter from ``_megatron_layer`` or the default ``nn.Module`` lookup.
        """
        if name in ("weight", "bias") and self.__dict__.get("_use_megatron", False):
            megatron = self._modules.get("_megatron_layer")
            if megatron is not None:
                return getattr(megatron, name)
        return super().__getattr__(name)

    def _save_to_state_dict(self, destination: dict[str, torch.Tensor], prefix: str, keep_vars: bool) -> None:
        """Save Megatron weights under flat names for checkpoint compatibility.

        See :meth:`TPColumnParallelLinear._save_to_state_dict` for details.

        Args:
            destination: Dict to write state entries into.
            prefix: Key prefix for this module.
            keep_vars: If True, return ``nn.Parameter`` objects; otherwise return ``.data``.
        """
        if self._use_megatron:
            destination[prefix + "weight"] = (
                self._megatron_layer.weight if keep_vars else self._megatron_layer.weight.data
            )
            if self._megatron_layer.bias is not None:
                destination[prefix + "bias"] = (
                    self._megatron_layer.bias if keep_vars else self._megatron_layer.bias.data
                )
        else:
            super()._save_to_state_dict(destination, prefix, keep_vars)

    def state_dict(  # type: ignore[override]
        self, *args: Any, destination: OrderedDict | None = None, prefix: str = "", keep_vars: bool = False
    ) -> OrderedDict:
        """Return state dict with flat key names, skipping ``_megatron_layer`` child traversal.

        See :meth:`TPColumnParallelLinear.state_dict` for details.

        Args:
            destination: Optional pre-existing dict to write into.
            prefix: Key prefix for this module.
            keep_vars: If True, return ``nn.Parameter`` objects; otherwise return ``.data``.

        Returns:
            OrderedDict mapping flat parameter names to tensors.
        """
        if destination is None:
            destination = OrderedDict()
        self._save_to_state_dict(destination, prefix, keep_vars)
        if not self._use_megatron:
            for name, module in self._modules.items():
                if module is not None:
                    module.state_dict(*args, destination=destination, prefix=prefix + name + ".", keep_vars=keep_vars)
        return destination

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Load checkpoint tensors, auto-sharding full weights for RowParallel (dim 1).

        When the incoming weight is larger than the expected sharded shape, it is automatically sliced along dim 1 for
        the current TP rank.  Bias is NOT sharded for row-parallel layers (each rank holds the full bias).

        Args:
            state_dict: Checkpoint state dict (may be mutated with injected keys).
            prefix: Key prefix for this module.
            local_metadata: Unused; required by PyTorch API.
            strict: If True, missing keys are reported.
            missing_keys: List to append missing key names to.
            unexpected_keys: List to append unexpected key names to.
            error_msgs: List to append error messages to.
        """
        if self._use_megatron:
            weight_key = prefix + "weight"
            bias_key = prefix + "bias"

            if weight_key in state_dict:
                w = state_dict[weight_key]
                expected = self._megatron_layer.weight.data.shape
                if w.shape != expected:
                    tp_rank = get_tensor_parallel_rank()
                    per_rank = expected[1]
                    w = w[:, tp_rank * per_rank : (tp_rank + 1) * per_rank]
                self._megatron_layer.weight.data.copy_(w)
            elif strict:
                missing_keys.append(weight_key)

            if self._has_bias:
                if bias_key in state_dict:
                    self._megatron_layer.bias.data.copy_(state_dict[bias_key])
                elif strict:
                    missing_keys.append(bias_key)

            # Inject megatron-prefixed keys so the child's _load_from_state_dict
            # finds them and doesn't report missing keys.
            megatron_prefix = prefix + "_megatron_layer."
            state_dict[megatron_prefix + "weight"] = self._megatron_layer.weight.data
            if self._megatron_layer.bias is not None:
                state_dict[megatron_prefix + "bias"] = self._megatron_layer.bias.data
        else:
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_megatron:
            output, _ = self._megatron_layer(x)
            return output
        else:
            # Scatter input if not already parallel
            if not self.input_is_parallel:
                x = scatter_to_tensor_parallel_region(x, dim=-1)

            # Local computation (no bias yet)
            output = F.linear(x, self.weight)

            # All-reduce to combine partial results
            output = reduce_from_tensor_parallel_region(output)

            # Add bias after reduction
            if self.bias is not None:
                output = output + self.bias

            return output


# =============================================================================
# Utility Functions
# =============================================================================


def shard_fused_qkv(value: torch.Tensor, tp_size: int, tp_rank: int) -> torch.Tensor:
    """Shard a fused QKV weight or bias for the given TP rank.

    The fused layout is [Q | K | V] along dim 0. We split into three equal parts, shard each by head, and recombine so
    that each rank gets the correct subset of heads from all three of Q, K, V.
    """
    total = value.shape[0]
    if total % 3 != 0:
        raise ValueError(f"Fused QKV dim 0 ({total}) must be divisible by 3")
    d = total // 3
    q, k, v = value[:d], value[d : 2 * d], value[2 * d :]

    shard = d // tp_size
    q_shard = q[tp_rank * shard : (tp_rank + 1) * shard]
    k_shard = k[tp_rank * shard : (tp_rank + 1) * shard]
    v_shard = v[tp_rank * shard : (tp_rank + 1) * shard]
    return torch.cat([q_shard, k_shard, v_shard], dim=0)
