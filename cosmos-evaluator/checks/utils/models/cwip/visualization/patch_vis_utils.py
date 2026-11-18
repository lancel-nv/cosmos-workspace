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
Patch visualization utilities for generating videos during inference.
"""

import ast
from collections.abc import Iterable
import logging
import os
from typing import Any, cast

import cv2
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from checks.utils.models.cwip.modules import class_mapping as class_mapping_module
from checks.utils.models.cwip.visualization import (
    basic_vis_utils,
    image_video_utils,
    metrics_utils,
    world_model_utils,
)

logger = logging.getLogger(__name__)

# Import common visualization colors from world_model_utils
FALLBACK_COLOR = world_model_utils.FALLBACK_COLOR
SIMILARITY_IGNORED_COLOR = world_model_utils.SIMILARITY_IGNORED_COLOR
OBJECT_TYPE_IGNORED_COLOR = world_model_utils.OBJECT_TYPE_IGNORED_COLOR


def _get_similarity_colors_tensor(
    num_classes: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """RGB rows (num_classes, 3) in [0, 1] from :func:`world_model_utils.get_similarity_type_colors`."""
    palette = world_model_utils.get_similarity_type_colors(include_ignored=False, num_classes=num_classes)
    rows = []
    for i in range(num_classes):
        rgb = palette.get(i, FALLBACK_COLOR)
        rows.append([float(rgb[0]), float(rgb[1]), float(rgb[2])])
    t = torch.tensor(rows, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


def normalize_excluded_classes_for_argmax(value: Any) -> list[int] | None:
    """Convert excluded classes to a list of integers, handling both list and string inputs.

    Handles:
    - String inputs: "[8]", "8", "[8,9]" -> parsed and converted to [8] or [8, 9]
    - List inputs: [8], ['8'], [8, 9] -> all elements converted to int
    - Other iterables: converted to list of ints

    Args:
        value: Input value that can be None, a string, list, tuple, or other iterable

    Returns:
        list[int] | None: List of integer class indices, or None if invalid
    """
    if value is None:
        return None
    if isinstance(value, str):
        # Try to parse string like "[8]" or "8" or "[8,9]"
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, int):
                return [parsed]  # Already an int
            elif isinstance(parsed, list):
                # Convert each element to int (handles both int and str elements)
                return [int(x) for x in parsed]
            else:
                return None
        except (ValueError, SyntaxError):
            return None
    elif isinstance(value, (list, tuple)):
        # Convert each element to int (handles both int and str elements like [8] or ['8'])
        # int('8') -> 8, int(8) -> 8, so this works for both cases
        return [int(x) for x in value]
    else:
        # Try to convert to list (e.g., if it's a set or other iterable)
        try:
            return [int(x) for x in value]
        except (TypeError, ValueError):
            return None


def mask_excluded_classes_for_argmax(
    logits_or_probs: torch.Tensor,
    excluded_classes: list[int] | None,
    num_classes: int,
) -> torch.Tensor:
    """
    Mask excluded classes by setting them to -inf (for logits) or 0 (for probabilities).
    This ensures they are never selected by argmax.

    This is a consolidated standalone function that can be used in both module methods and utility functions without
    requiring access to the model instance.

    Args:
        logits_or_probs: Logits or probabilities tensor of shape (B, num_classes, H, W)
        excluded_classes: List of class indices to exclude from argmax, or None
        num_classes: Total number of classes in the tensor

    Returns:
        Masked tensor of same shape with excluded classes set to -inf (if logits) or 0 (if probs)
    """
    if not excluded_classes:
        return logits_or_probs

    # Create a mask: True for excluded classes, False otherwise
    # Shape: (num_classes,)
    mask = torch.zeros(num_classes, dtype=torch.bool, device=logits_or_probs.device)
    for excluded_idx in excluded_classes:
        if 0 <= excluded_idx < num_classes:
            mask[excluded_idx] = True

    # Reshape mask to (1, num_classes, 1, 1) for broadcasting
    mask = mask.view(1, -1, 1, 1)

    # Determine if input is probabilities (all values >= 0) or logits
    is_probabilities = logits_or_probs.min() >= 0.0

    if is_probabilities:
        # For probabilities: set excluded classes to 0
        masked = logits_or_probs.clone()
        masked[mask.expand_as(masked)] = 0.0
    else:
        # For logits: set excluded classes to -inf
        masked = logits_or_probs.clone()
        masked[mask.expand_as(masked)] = float("-inf")

    return masked


# Canonical default: ``MATCH + EXTRA + WRONG`` — the three classes that all say "the model thinks an object is here".
# ``MATCH`` ("object that's also in the world frame") and ``EXTRA`` ("object the model detected that's NOT in the world
# frame") are the two presence classes for ground-truth-vs-detection alignment; ``WRONG`` ("model detected an object
# that aligns with the world frame, but assigned the wrong type") is also a model-presence signal — the type head was
# confused, not the similarity head. ``DONT_CARE`` (background) and ``MISSING`` ("model thinks NOTHING is here, but the
# world says something should be") are deliberately excluded — they are the two non-presence classes.
#
# Lives here (not on ``CWIPInference``) so the visualization module — which is the primary consumer — can reference it
# without dragging in the heavy ``cwip.py`` import (which itself depends on this module, creating a runtime circular
# import when invoked from unit tests that don't have ``cwip`` in their runfiles). ``CWIPInference`` re-exports the
# symbol as a class attribute for back-compat with existing callers.
DEFAULT_PATCH_SIMILARITY_MASK_CLASSES: tuple[world_model_utils.PatchSimilarityType, ...] = (
    world_model_utils.PatchSimilarityType.MATCH,
    world_model_utils.PatchSimilarityType.EXTRA,
    world_model_utils.PatchSimilarityType.WRONG,
)


def compute_patch_similarity_class_mask(
    patch_similarity_logits: torch.Tensor,
    patch_type_logits: torch.Tensor | None = None,
    *,
    similarity_classes: Iterable[world_model_utils.PatchSimilarityType | int] = DEFAULT_PATCH_SIMILARITY_MASK_CLASSES,
    activation: str = "sigmoid",
    mask_mode: str = "max",
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel similarity-head confidence mask over a configurable class subset.

    Builds a per-pixel mask over the user-selected ``similarity_classes``. The default — ``MATCH + EXTRA + WRONG`` (see
    :data:`DEFAULT_PATCH_SIMILARITY_MASK_CLASSES`) — is the canonical "the model thinks an object is here" presence mask
    used by the 3x3-with-mask-trio visualization row-3 and most downstream Cosmos Evaluator checks. All three classes
    are model-presence signals:

    * ``MATCH`` (index 1) — model detected an object that's also in the world frame.
    * ``EXTRA`` (index 3) — model detected an object that is NOT in the world frame.
    * ``WRONG`` (index 2) — model detected an object that aligns with the world frame, but assigned the wrong type. The
      similarity head was confident an object was there; only the type head was wrong.

    ``DONT_CARE`` (0, background) and ``MISSING`` (4, "model thinks NOTHING is here, but the world says something should
    be") are deliberately excluded — they are the two non-presence classes.

    Other useful subsets (passed via ``similarity_classes``):

    * ``(MATCH, EXTRA)`` — legacy strict-presence mask: drops ``WRONG`` so model-detected objects with wrong type don't
      contribute. Useful when downstream consumers want to gate on type-confident detections only.
    * ``(MATCH,)`` alone — "model detected the right object at the right place" mask; drops both ``EXTRA`` and
      ``WRONG``.
    * ``(EXTRA,)`` alone — "model thinks something is there that isn't in the world frame".
    * ``(WRONG, MISSING)`` — "model is confused" mask: union of the two failure modes; useful for surfacing
      low-confidence regions in offline analysis.
    * Any other subset of :class:`world_model_utils.PatchSimilarityType` for ablation / debugging.

    The five class semantics (see :class:`world_model_utils.PatchSimilarityType`): ``DONT_CARE`` (0), ``MATCH`` (1),
    ``WRONG`` (2), ``EXTRA`` (3), ``MISSING`` (4).

    Used by :func:`create_combined_patch_visualization` for the 3x3 with-mask-trio layout's row-3 (mask itself, camera x
    mask, type prediction x mask) — the layout is mode-agnostic and works in training, eval, and inference. Exposed as a
    top-level utility so downstream checks (e.g. the static-object Cosmos Evaluator checks under
    ``checks/objects/single_view/``) can gate their type-head consumers by similarity-head confidence without
    re-deriving the indices.

    The function lives in ``patch_vis_utils`` (not ``cwip.py``) so that visualization code — the primary consumer — can
    call it without dragging in the heavy ``cwip`` module (which itself imports this module, creating a runtime circular
    import when invoked from unit tests that don't carry ``cwip`` in their runfiles).
    :class:`~checks.utils.models.cwip.cwip.CWIPInference` re-exports this function as a ``@staticmethod`` (and the
    constant as a class attribute) for back-compat with existing callers that import via
    ``CWIPInference.compute_patch_similarity_class_mask``.

    Optional masked type probabilities
    ----------------------------------
    When ``patch_type_logits`` is provided, the function also returns the type-head probabilities gated by the same
    mask. This is the canonical "type prediction with similarity-head confidence as opacity" used by the
    3x3-with-mask-trio row-3 col-3 panel and is exactly what most downstream Cosmos Evaluator checks want — it lets a
    check do a single forward pass and consume already-gated per-class probabilities instead of having to re-derive the
    indices and replicate the broadcast multiply.

    Args:
        patch_similarity_logits: ``(B, num_sim_classes, H, W)`` raw logits from the similarity head. ``num_sim_classes``
            must cover every index in ``similarity_classes``.
        patch_type_logits: Optional ``(B, num_type_classes, H, W)`` raw logits from the type head. Must share batch and
            spatial dimensions with ``patch_similarity_logits`` (the channel dim is independent — type and similarity
            heads have different class counts). When provided, the function returns the ``(mask,
            masked_type_probabilities)`` tuple described below.
        similarity_classes: Keyword-only iterable of similarity-head class indices to include in the mask. Accepts
            :class:`PatchSimilarityType` enum members, raw ``int`` indices, or a mix. Empty iterables raise (an all-zero
            mask is almost always a bug); duplicates raise (they would silently inflate the ``"sum"`` mask before the
            ``clamp(0, 1)`` and almost always indicate a programming error). Defaults to
            :data:`DEFAULT_PATCH_SIMILARITY_MASK_CLASSES` for the legacy "object-presence" semantics.
        activation: Activation passed to :func:`metrics_utils.apply_logits_activation`. Applied to BOTH heads for
            consistency.
            ``"sigmoid"`` (default) treats the heads as multi-label (the standard CWIP setting); ``"softmax"`` treats
            them as single-label. Ignored when ``mask_mode='max'`` because ``argmax`` is invariant to monotonic
            per-class activations.
        mask_mode: How the mask is built from ``patch_similarity_logits``:

            * ``"max"`` (default): hard binary mask. Take ``argmax`` over the full class dim of
              ``patch_similarity_logits``; the mask is 1 where the winning class is in ``similarity_classes`` and 0
              otherwise. Pixels are kept in full or dropped — there is no soft opacity. ``activation`` is irrelevant in
              this branch (``argmax`` is invariant to monotonic per-class transforms), so it is ignored. This is the
              canonical "model believes class is in this subset" gate that downstream consumers (binary masking of
              camera frames, type-probability gating) want — multiplication by a 0/1 mask is equivalent to selecting /
              zeroing out, matching the "non-zero ⇒ keep as 1" semantics.
            * ``"sum"`` (legacy): soft mask ``clamp(sum_{c in similarity_classes} P(c), 0, 1)``. Preserves per-pixel
              similarity-head confidence as a ``[0, 1]`` opacity. Sigmoid sums can exceed 1 trivially; the clamp keeps
              the result a valid mask. ``activation`` selects sigmoid (multi-label) vs softmax (single-label) for
              ``P(c)``. Use this when a visualization wants per-pixel opacity tied to similarity confidence rather than
              a strict 0/1 select.

    Returns:
        - When ``patch_type_logits`` is ``None`` (default): the mask alone, a ``(B, H, W)`` float tensor in ``[0, 1]``,
          same dtype/device as the input logits. ``"sum"`` mode yields soft values in ``[0, 1]``; ``"max"`` yields
          binary 0/1 values in the same dtype.
        - When ``patch_type_logits`` is provided: a tuple ``(mask, masked_type_probabilities)``. ``mask`` is as above;
          ``masked_type_probabilities`` is ``activation(patch_type_logits) * mask.unsqueeze(1)`` — ``(B,
          num_type_classes, H, W)`` in ``[0, 1]``. With ``mask_mode='max'`` the multiplication is effectively a 0/1
          select on each pixel.

    .. warning::
        **NumPy interop with bfloat16/float16.** The returned tensor preserves the input dtype. Validation / inference
        frequently runs in mixed precision (``bfloat16``), so the mask comes back as ``bfloat16`` — and
        ``Tensor.numpy()`` rejects ``bfloat16`` with ``TypeError: Got unsupported ScalarType BFloat16``. Downstream
        consumers (visualization, the Cosmos Evaluator checks under ``checks/objects/single_view/``, anything else
        that needs a NumPy view) MUST upcast first::

            mask.detach().float().cpu().numpy()   # safe for any input dtype

        ``.float()`` is a no-op when the input is already ``float32``, so the upcast is free in fp32 paths and mandatory
        in bf16/fp16 paths. We deliberately do NOT upcast inside this function so torch consumers that want to keep the
        mask in the model's native precision (e.g. a downstream bf16 multiply with ``patch_type_logits``) don't pay an
        unnecessary copy.
    """
    if mask_mode not in ("sum", "max"):
        raise ValueError(f"mask_mode must be 'sum' or 'max', got '{mask_mode}'")

    indices = [int(c) for c in similarity_classes]
    if not indices:
        raise ValueError(
            "similarity_classes must contain at least one entry; an "
            "empty subset would yield an all-zero mask, which is almost "
            "always a programming error rather than the intended "
            "'no classes contribute' semantics."
        )
    seen: set[int] = set()
    dups: set[int] = set()
    for i in indices:
        if i in seen:
            dups.add(i)
        seen.add(i)
    if dups:
        raise ValueError(
            f"similarity_classes contains duplicate indices: {sorted(dups)}. Each class index may appear at most once."
        )
    num_classes = patch_similarity_logits.shape[1]
    out_of_range = sorted({i for i in indices if not (0 <= i < num_classes)})
    if out_of_range:
        raise ValueError(
            f"similarity_classes index/indices {out_of_range} are out of "
            f"range for patch_similarity_logits with {num_classes} classes "
            f"(valid range: [0, {num_classes}))."
        )

    if mask_mode == "sum":
        sim_probabilities = metrics_utils.apply_logits_activation(patch_similarity_logits, activation=activation)
        mask = sim_probabilities[:, indices].sum(dim=1).clamp(0.0, 1.0)
    else:  # "max"
        # ``argmax`` is invariant to monotonic per-class activations (sigmoid / softmax don't reorder per-pixel class
        # rankings), so we skip the activation pass and ``argmax`` directly on logits — saves an elementwise op over the
        # full ``(B, num_classes, H, W)`` tensor.
        argmax_indices = patch_similarity_logits.argmax(dim=1)  # (B, H, W) int64
        indices_tensor = torch.tensor(indices, device=patch_similarity_logits.device, dtype=argmax_indices.dtype)
        # Binary 0/1 mask in the input dtype — preserves the "preserves input dtype" contract that ``"sum"`` mode and
        # downstream consumers rely on (e.g. bf16 training paths multiply this against bf16 type logits).
        mask = torch.isin(argmax_indices, indices_tensor).to(patch_similarity_logits.dtype)

    if patch_type_logits is None:
        return mask

    if patch_type_logits.shape[0] != patch_similarity_logits.shape[0]:
        raise ValueError(
            f"patch_type_logits batch size {patch_type_logits.shape[0]} does not match "
            f"patch_similarity_logits batch size {patch_similarity_logits.shape[0]}."
        )
    if patch_type_logits.shape[2:] != patch_similarity_logits.shape[2:]:
        raise ValueError(
            f"patch_type_logits spatial shape {tuple(patch_type_logits.shape[2:])} does not "
            f"match patch_similarity_logits spatial shape "
            f"{tuple(patch_similarity_logits.shape[2:])}."
        )
    type_probabilities = metrics_utils.apply_logits_activation(patch_type_logits, activation=activation)
    # ``"sum"``: scalar opacity in ``[0, 1]`` -> soft scaling.
    # ``"max"``: 0/1 binary mask -> multiplication selects / zeros pixels
    # (matches the "non-zero ⇒ keep as 1" gating semantics requested by
    # downstream consumers that mask camera frames / type probabilities).
    masked_type_probabilities = type_probabilities * mask.unsqueeze(1)
    return mask, masked_type_probabilities


def _build_type_colors_tensor(
    num_type_classes: int,
    object_type_class_mapping: type[class_mapping_module.ObjectTypeClassMapping] | None = None,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Build a tensor of RGB colors for each model class index.

    When object_type_class_mapping is provided, uses the mapping to convert model class indices to ObjectType values
    before looking up colors. This is necessary because model class indices may not directly correspond to ObjectType
    enum values (e.g., with v3 mapping).

    Args:
        num_type_classes: Number of object type classes in the model output
        object_type_class_mapping: Optional class mapping to convert model class indices to ObjectType values
        device: Device to place the tensor on

    Returns:
        Tensor of shape (num_type_classes, 3) with RGB colors in [0, 1]
    """
    type_colors = world_model_utils.get_object_type_colors()

    colors_list = []
    for model_class_idx in range(num_type_classes):
        if object_type_class_mapping is not None:
            # Use the mapping to get the ObjectType for this model class index
            try:
                obj_type = object_type_class_mapping.get_object_type(model_class_idx)
                color = type_colors.get(obj_type, FALLBACK_COLOR)
            except (KeyError, ValueError):
                color = FALLBACK_COLOR
        else:
            # Fallback: assume direct mapping (model class index == ObjectType value)
            try:
                obj_type = world_model_utils.ObjectType(model_class_idx)
                color = type_colors.get(obj_type, FALLBACK_COLOR)
            except ValueError:
                color = FALLBACK_COLOR
        colors_list.append([color[0], color[1], color[2]])

    return torch.tensor(colors_list, dtype=torch.float32, device=device)


def create_detailed_patch_visualization(
    output_batch: dict[str, torch.Tensor],
    data_batch: dict[str, torch.Tensor],
    iteration: int,
    output_dir: str,
    patch_results: dict[str, torch.Tensor],
    target_resolution: tuple[int, int] = (1280, 720),
    chunk_size: int = 256,
    frame_indices: torch.Tensor | None = None,
    object_type_class_mapping: type[class_mapping_module.ObjectTypeClassMapping] | str | None = None,
    activation: str = "sigmoid",
) -> None:
    """
    Create detailed patch visualization videos with probability maps for each class.

    Creates separate visualizations for similarity and object type predictions:
    - detailed_patch_similarity_vis: 3x3 layout with 5 similarity classes
    - detailed_patch_type_vis: 5x6 layout with 22 object type classes

    Args:
        output_batch: Model output containing patch predictions
        data_batch: Data batch containing camera and world frames
        iteration: Iteration number
        output_dir: Directory to save the visualization
        patch_results: Pre-computed patch predictions and percentages from CWIP model (should be computed with
                      treat_wrong_type_as_match=False)
        target_resolution: Target resolution (width, height) for individual subplots
                          Default: (1280, 720) for high quality
                          Use (640, 360) to fit 5x6 layout within NVENC limits
        frame_indices: Optional (N, 2) tensor of [world_idx, camera_idx] pairs to visualize. If None, visualizes all
                      frames. If provided, only visualizes selected pairs.
        object_type_class_mapping: Class mapping for object type visualization. Can be a string ("v1", "v2", "v3") or
                                   mapping class. Required for correct color rendering when model class indices don't
                                   match ObjectType values.
        activation: Activation function for converting logits to probabilities ('sigmoid' or 'softmax').
    """
    object_type_class_mapping = class_mapping_module.get_object_type_class_mapping(object_type_class_mapping)
    # Check what types of predictions are available
    patch_similarity_logits = output_batch.get("patch_similarity_logits")
    patch_type_logits = output_batch.get("patch_type_logits")

    # Create similarity visualization if available
    if patch_similarity_logits is not None:
        _create_detailed_similarity_visualization(
            output_batch,
            data_batch,
            iteration,
            output_dir,
            patch_results,
            target_resolution,
            chunk_size,
            frame_indices,
            activation=activation,
        )

    # Create object type visualization if available
    if patch_type_logits is not None:
        excluded_type = output_batch.get("excluded_model_classes_for_argmax_type", None)
        _create_detailed_type_visualization(
            output_batch,
            data_batch,
            iteration,
            output_dir,
            target_resolution,
            chunk_size,
            frame_indices,
            object_type_class_mapping,
            activation=activation,
            excluded_model_classes_for_argmax=cast(list[int] | None, excluded_type),
        )


def _create_detailed_similarity_visualization(
    output_batch: dict[str, torch.Tensor],
    data_batch: dict[str, torch.Tensor],
    iteration: int,
    output_dir: str,
    patch_results: dict[str, torch.Tensor],
    target_resolution: tuple[int, int] = (1280, 720),
    chunk_size: int = 256,
    frame_indices: torch.Tensor | None = None,
    activation: str = "sigmoid",
) -> None:
    """
    Create a detailed 3x3 patch similarity visualization video with the following layout:

    Row 1: Input camera video | Input world video | Overlay visualization (MATCH, WRONG, EXTRA, MISSING on camera)
    Row 2: DONT_CARE prob map | MATCH prob map | WRONG prob map
    Row 3: EXTRA prob map | MISSING prob map | Combined argmax visualization

    Note: The top-right (overlay) and bottom-right (argmax) visualizations use strict mode
          (treat_wrong_type_as_match=False), showing WRONG and MATCH as separate classes.

    Args:
        output_batch: Model output containing patch predictions
        data_batch: Data batch containing camera and world frames
        iteration: Iteration number
        output_dir: Directory to save the visualization
        patch_results: Pre-computed patch predictions and percentages from CWIP model (should be computed with
                      treat_wrong_type_as_match=False)
        activation: Activation function for converting logits to probabilities ('sigmoid' or 'softmax').
    """
    logger.info("Creating detailed 3x3 patch similarity visualization...")

    # Extract pre-computed results from CWIP model - keep on same device as input
    pred_class_indices = patch_results["pred_class_indices"]  # (B, H, W)
    pred_probabilities = patch_results["pred_probabilities"]  # (B, H, W)

    # Get similarity logits for probability map visualization - KEEP ON GPU
    patch_similarity_logits = output_batch.get("patch_similarity_logits")  # GPU
    assert patch_similarity_logits is not None, "patch_similarity_logits missing from output_batch"

    # Convert bfloat16 to float32 for computation
    if patch_similarity_logits.dtype == torch.bfloat16:
        patch_similarity_logits = patch_similarity_logits.to(torch.float32)

    probabilities = metrics_utils.apply_logits_activation(
        patch_similarity_logits, activation=activation
    )  # (B, num_classes, H, W) - GPU

    # Get camera and world frames (convert to numpy immediately to save GPU memory)
    result = basic_vis_utils.extract_frames_from_data_batch(data_batch)
    if result is None or result[0] is None:
        logger.error("Failed to extract frames from data batch")
        return
    camera_frames, world_frames, camera_fps = result
    assert camera_frames is not None and world_frames is not None and camera_fps is not None

    # Apply frame selection if indices are provided
    if frame_indices is not None:
        world_frame_indices = frame_indices[:, 0].cpu().tolist()
        camera_frame_indices = frame_indices[:, 1].cpu().tolist()
        logger.info("  - Using balanced frame indices for detailed similarity vis: %s pairs", len(camera_frame_indices))

        # Select frames
        camera_frames = camera_frames[camera_frame_indices]
        world_frames = world_frames[world_frame_indices]

        # Select corresponding predictions
        probabilities = probabilities[camera_frame_indices]
        pred_class_indices = pred_class_indices[camera_frame_indices]
        pred_probabilities = pred_probabilities[camera_frame_indices]

        # Also update patch_results to only include selected frames
        for key, val in patch_results.items():
            if isinstance(val, torch.Tensor) and val.shape[0] == patch_similarity_logits.shape[0]:
                patch_results[key] = val[camera_frame_indices]

    # Resize frames to target resolution if needed (CPU to avoid OOM - GPU already full from model)
    original_height, original_width = camera_frames.shape[1], camera_frames.shape[2]
    target_width, target_height = target_resolution
    if (original_height, original_width) != (target_height, target_width):
        logger.info(
            "Resizing frames from %sx%s to %sx%s for detailed similarity visualization",
            original_width,
            original_height,
            target_width,
            target_height,
        )
        resized_camera = np.zeros((camera_frames.shape[0], target_height, target_width, 3), dtype=camera_frames.dtype)
        resized_world = np.zeros((world_frames.shape[0], target_height, target_width, 3), dtype=world_frames.dtype)
        for i in range(camera_frames.shape[0]):
            resized_camera[i] = cv2.resize(
                camera_frames[i], (target_width, target_height), interpolation=cv2.INTER_LINEAR
            )
            resized_world[i] = cv2.resize(
                world_frames[i], (target_width, target_height), interpolation=cv2.INTER_LINEAR
            )
        camera_frames = resized_camera
        world_frames = resized_world

    # Add text overlays showing class percentages (same as combined_patch_vis)
    camera_frames_with_text = _add_patch_text_overlays(camera_frames, patch_results, 0, "camera")
    world_frames_with_text = _add_patch_text_overlays(world_frames, patch_results, 0, "world")

    num_frames = camera_frames.shape[0]
    frame_height, frame_width = camera_frames.shape[1], camera_frames.shape[2]
    num_classes = patch_similarity_logits.shape[1]

    device = pred_probabilities.device
    class_colors_tensor = _get_similarity_colors_tensor(num_classes, device=device)

    # Pre-allocate output buffers on CPU as numpy arrays. The previous implementation kept these as GPU float32 RGB
    # tensors, which cost ~6.3 GiB for the type-vis path on a 5-second video at 1280x720 (22 classes x T x H x W x 3 x 4
    # bytes). That was enough to OOM the H100 after the model weights were resident, with the buffers serving only as a
    # staging area before being immediately .cpu()'d for ffmpeg. Prob maps are stored single-channel and broadcast to
    # RGB lazily right before the video encoder consumes them.
    overlay_video = np.zeros((num_frames, frame_height, frame_width, 3), dtype=np.float32)
    argmax_video = np.zeros((num_frames, frame_height, frame_width, 3), dtype=np.float32)
    prob_maps = np.zeros((num_classes, num_frames, frame_height, frame_width), dtype=np.float32)

    # Convert camera frames to tensor for blending (use same device)
    camera_tensor = torch.from_numpy(camera_frames_with_text).to(device)  # (T, H, W, 3)

    # Process frames in chunks (GPU batch operations)
    for chunk_start in tqdm(
        range(0, num_frames, chunk_size), desc="Processing detailed similarity visualization chunks (GPU)"
    ):
        chunk_end = min(chunk_start + chunk_size, num_frames)

        chunk_camera = camera_tensor[chunk_start:chunk_end]  # (chunk_frames, H, W, 3) - GPU
        chunk_pred_class_indices = pred_class_indices[chunk_start:chunk_end]  # GPU
        chunk_pred_probabilities = pred_probabilities[chunk_start:chunk_end]  # GPU
        chunk_probabilities = probabilities[chunk_start:chunk_end]  # (chunk_frames, num_classes, H, W) - GPU

        # Create overlay visualization (similar to top-right of combined_patch_vis)
        # Using strict mode: WRONG and MATCH are shown as separate classes
        # RGB colors with intensity = probability
        rgb_colors = class_colors_tensor[chunk_pred_class_indices]  # (chunk_frames, H, W, 3) - GPU
        # Ensure pred_probabilities is on the same device as rgb_colors (should be GPU, but handle CPU case)
        chunk_pred_probabilities = chunk_pred_probabilities.to(rgb_colors.device)
        pred_probabilities_expanded = chunk_pred_probabilities.unsqueeze(-1).expand(-1, -1, -1, 3)
        pred_combined_rgb = rgb_colors * pred_probabilities_expanded  # GPU

        # Create argmax visualization (similar to bottom-right of combined_patch_vis)
        # Using strict mode: WRONG and MATCH are shown as separate classes
        # RGB colors with intensity = probability (same as overlay base, before blending with camera)
        argmax_rgb = rgb_colors * pred_probabilities_expanded  # (chunk_frames, H, W, 3) - GPU

        # GPU batch resize for predictions (T, H, W, C) -> (T, C, H, W) -> resize -> (T, H, W, C)
        pred_rgb_resized = torch.nn.functional.interpolate(
            pred_combined_rgb.permute(0, 3, 1, 2),  # (T, C, H, W)
            size=(frame_height, frame_width),
            mode="nearest",
        ).permute(0, 2, 3, 1)  # (T, H, W, C)

        argmax_resized = torch.nn.functional.interpolate(
            argmax_rgb.permute(0, 3, 1, 2), size=(frame_height, frame_width), mode="nearest"
        ).permute(0, 2, 3, 1)

        # Create overlay: blend prediction over camera only for foreground pixels (non-black color).
        # DONT_CARE (background) is black (0,0,0); blending it unconditionally would darken
        # background regions to 0.7×camera. Show camera unchanged where prediction color is zero.
        fg_mask = pred_rgb_resized.sum(dim=-1, keepdim=True) > 1e-6  # (T, H, W, 1)
        blended = torch.clamp(0.7 * chunk_camera + 0.3 * pred_rgb_resized, 0, 1)
        overlay = torch.where(fg_mask, blended, chunk_camera)

        # Stream chunk back to CPU buffers immediately so the GPU allocator can reuse this memory for the next chunk.
        overlay_video[chunk_start:chunk_end] = overlay.detach().cpu().numpy()
        argmax_video[chunk_start:chunk_end] = argmax_resized.detach().cpu().numpy()

        # Create probability maps for each class - SINGLE-CHANNEL (GPU batch ops)
        for class_id in range(num_classes):
            prob_map = chunk_probabilities[:, class_id, :, :]  # (chunk_frames, H, W) - GPU

            prob_map_resized = torch.nn.functional.interpolate(
                prob_map.unsqueeze(1),  # (chunk_frames, 1, H, W)
                size=(frame_height, frame_width),
                mode="nearest",
            ).squeeze(1)  # (chunk_frames, H, W)

            prob_maps[class_id, chunk_start:chunk_end] = prob_map_resized.detach().cpu().numpy()

    # Free GPU tensors we no longer need before kicking off ffmpeg / NVENC, which itself allocates a non-trivial CUDA
    # workspace.
    del camera_tensor, probabilities, pred_class_indices, pred_probabilities
    del patch_similarity_logits, class_colors_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create the 3x3 tiled video
    video_path = os.path.join(output_dir, f"detailed_patch_similarity_vis_iter_{iteration:06d}.mp4")

    # Add text overlays to probability maps using SimilarityClassMappingV1 for correct labels. Broadcast the
    # single-channel prob maps to RGB lazily; np.broadcast_to is zero-copy so the 3x memory blow-up only happens inside
    # _add_probability_map_text_overlays if it actually mutates a copy.
    prob_maps_with_text = {}
    for class_id in range(num_classes):
        prob_maps_with_text[class_id] = _add_probability_map_text_overlays(
            np.broadcast_to(prob_maps[class_id, ..., None], (num_frames, frame_height, frame_width, 3)),
            class_id,
            0,  # frame_offset
            class_mapping=class_mapping_module.SimilarityClassMappingV1,
        )

    # Prepare the 9 tiles in order:
    # Row 1: camera (with text), world (with text), overlay
    # Row 2: DONT_CARE prob, MATCH prob, WRONG prob (all with text)
    # Row 3: EXTRA prob, MISSING prob (with text), argmax
    tiles = [
        camera_frames_with_text,  # Top-left
        world_frames_with_text,  # Top-center
        overlay_video,  # Top-right
        prob_maps_with_text[world_model_utils.PatchSimilarityType.DONT_CARE.value],  # Mid-left
        prob_maps_with_text[world_model_utils.PatchSimilarityType.MATCH.value],  # Mid-center
        prob_maps_with_text[world_model_utils.PatchSimilarityType.WRONG.value],  # Mid-right
        prob_maps_with_text[world_model_utils.PatchSimilarityType.EXTRA.value],  # Bot-left
        prob_maps_with_text[world_model_utils.PatchSimilarityType.MISSING.value],  # Bot-center
        argmax_video,  # Bot-right
    ]

    # Convert all tiles to uint8
    tiles_uint8 = [image_video_utils.to_uint8(tile) for tile in tiles]

    # Determine tile dimensions
    tile_height = frame_height
    tile_width = frame_width

    # Create output directory if needed
    if not os.path.exists(os.path.dirname(video_path)):
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Write video using ffmpeg (NVENC when available, else CPU)
    out_height = tile_height * 3
    out_width = tile_width * 3

    logger.info("Starting 3x3 video encoding: %s frames at %sx%s", num_frames, out_width, out_height)

    frame_gen = image_video_utils.tiled_frames_generator(
        tiles_uint8,
        3,
        3,
        num_frames,
        tile_height,
        tile_width,
        use_gpu=True,  # Use GPU for frame resizing
    )
    image_video_utils.write_video_with_ffmpeg(frame_gen, video_path, out_width, out_height, camera_fps)

    logger.info("Detailed 3x3 patch similarity visualization video saved to: %s", video_path)

    # Clear GPU memory after video creation
    torch.cuda.empty_cache()
    logger.info("GPU memory cleared after detailed patch similarity visualization video creation")


def _create_detailed_type_visualization(
    output_batch: dict[str, torch.Tensor],
    data_batch: dict[str, torch.Tensor],
    iteration: int,
    output_dir: str,
    target_resolution: tuple[int, int] = (1280, 720),
    chunk_size: int = 256,
    frame_indices: torch.Tensor | None = None,
    object_type_class_mapping: type[class_mapping_module.ObjectTypeClassMapping] | None = None,
    activation: str = "sigmoid",
    excluded_model_classes_for_argmax: list[int] | None = None,
) -> None:
    """
    Create a detailed 5x6 patch object type visualization video.

    Layout (5 rows x 6 columns = 30 tiles, 4 empty):
    Row 1: Camera | World | Overlay | Class 0 | Class 1 | Class 2
    Rows 2-5: Class 3-22 probability maps + argmax + empty slots

    Args:
        output_batch: Model output containing patch type predictions
        data_batch: Data batch containing camera and world frames
        iteration: Iteration number
        output_dir: Directory to save the visualization
        object_type_class_mapping: Class mapping for converting model class indices to ObjectType values for color
                                   lookup
        activation: Activation function for converting logits to probabilities ('sigmoid' or 'softmax').
    """
    logger.info("Creating detailed 5x6 patch object type visualization...")

    # Get object type logits - KEEP ON GPU
    patch_type_logits = output_batch.get("patch_type_logits")  # GPU
    assert patch_type_logits is not None, "patch_type_logits missing from output_batch"

    # Convert bfloat16 to float32 for computation
    if patch_type_logits.dtype == torch.bfloat16:
        patch_type_logits = patch_type_logits.to(torch.float32)

    type_probabilities = metrics_utils.apply_logits_activation(
        patch_type_logits, activation=activation
    )  # (B, num_classes, H, W) - GPU

    # Mask excluded classes before argmax (for combined visualization only)
    # Individual class visualizations show all classes, but combined argmax visualization excludes these
    if excluded_model_classes_for_argmax:
        masked_type_probs = type_probabilities.clone()
        for excluded_idx in excluded_model_classes_for_argmax:
            if 0 <= excluded_idx < type_probabilities.shape[1]:
                masked_type_probs[:, excluded_idx, :, :] = 0.0
        type_probabilities = masked_type_probs

    # Compute predictions - KEEP ON GPU
    type_pred_class_indices = type_probabilities.argmax(dim=1)  # (B, H, W) - GPU
    type_pred_probabilities = type_probabilities.max(dim=1)[0]  # (B, H, W) - GPU

    # Get camera and world frames (convert to numpy immediately to save GPU memory)
    result = basic_vis_utils.extract_frames_from_data_batch(data_batch)
    if result is None or result[0] is None:
        logger.error("Failed to extract frames from data batch")
        return
    camera_frames, world_frames, camera_fps = result
    assert camera_frames is not None and world_frames is not None and camera_fps is not None

    # Apply frame selection if indices are provided
    if frame_indices is not None:
        world_frame_indices = frame_indices[:, 0].cpu().tolist()
        camera_frame_indices = frame_indices[:, 1].cpu().tolist()
        logger.info("  - Using balanced frame indices for detailed type vis: %s pairs", len(camera_frame_indices))

        # Select frames
        camera_frames = camera_frames[camera_frame_indices]
        world_frames = world_frames[world_frame_indices]

        # Select corresponding predictions
        type_probabilities = type_probabilities[camera_frame_indices]
        type_pred_class_indices = type_pred_class_indices[camera_frame_indices]
        type_pred_probabilities = type_pred_probabilities[camera_frame_indices]

    # Resize frames to target resolution if needed (CPU to avoid OOM - GPU already full from model)
    original_height, original_width = camera_frames.shape[1], camera_frames.shape[2]
    target_width, target_height = target_resolution
    if (original_height, original_width) != (target_height, target_width):
        logger.info(
            "Resizing frames from %sx%s to %sx%s for detailed type visualization",
            original_width,
            original_height,
            target_width,
            target_height,
        )
        resized_camera = np.zeros((camera_frames.shape[0], target_height, target_width, 3), dtype=camera_frames.dtype)
        resized_world = np.zeros((world_frames.shape[0], target_height, target_width, 3), dtype=world_frames.dtype)
        for i in range(camera_frames.shape[0]):
            resized_camera[i] = cv2.resize(
                camera_frames[i], (target_width, target_height), interpolation=cv2.INTER_LINEAR
            )
            resized_world[i] = cv2.resize(
                world_frames[i], (target_width, target_height), interpolation=cv2.INTER_LINEAR
            )
        camera_frames = resized_camera
        world_frames = resized_world

    num_frames = camera_frames.shape[0]
    frame_height, frame_width = camera_frames.shape[1], camera_frames.shape[2]
    num_type_classes = patch_type_logits.shape[1]

    # Get object type colors using the class mapping for correct color lookup
    # Determine device from type_probabilities
    device = type_probabilities.device
    type_colors_tensor = _build_type_colors_tensor(num_type_classes, object_type_class_mapping, device=str(device))

    # Pre-allocate output buffers on CPU as numpy arrays. See _create_detailed_similarity_visualization for the OOM
    # rationale; the type-vis path is even more expensive because num_type_classes can be 22.
    overlay_video = np.zeros((num_frames, frame_height, frame_width, 3), dtype=np.float32)
    argmax_video = np.zeros((num_frames, frame_height, frame_width, 3), dtype=np.float32)
    prob_maps = np.zeros((num_type_classes, num_frames, frame_height, frame_width), dtype=np.float32)

    # Convert camera frames to tensor for blending (use same device)
    camera_tensor = torch.from_numpy(camera_frames).to(device)  # (T, H, W, 3)

    # Process frames in chunks (GPU batch operations)
    for chunk_start in tqdm(
        range(0, num_frames, chunk_size), desc="Processing detailed type visualization chunks (GPU)"
    ):
        chunk_end = min(chunk_start + chunk_size, num_frames)

        chunk_camera = camera_tensor[chunk_start:chunk_end]  # (chunk_frames, H, W, 3) - GPU
        chunk_type_pred_class_indices = type_pred_class_indices[chunk_start:chunk_end]  # GPU
        chunk_type_pred_probabilities = type_pred_probabilities[chunk_start:chunk_end]  # GPU
        chunk_type_probabilities = type_probabilities[chunk_start:chunk_end]  # (chunk_frames, num_classes, H, W) - GPU

        # Create overlay and argmax visualizations (GPU)
        type_rgb_colors = type_colors_tensor[chunk_type_pred_class_indices]  # (chunk_frames, H, W, 3) - GPU
        # Ensure chunk_type_pred_probabilities is on the same device as type_rgb_colors
        chunk_type_pred_probabilities = chunk_type_pred_probabilities.to(type_rgb_colors.device)
        type_pred_probabilities_expanded = chunk_type_pred_probabilities.unsqueeze(-1).expand(-1, -1, -1, 3)
        type_pred_combined_rgb = type_rgb_colors * type_pred_probabilities_expanded  # GPU
        argmax_rgb = type_rgb_colors * type_pred_probabilities_expanded  # GPU

        # GPU batch resize for predictions (T, H, W, C) -> (T, C, H, W) -> resize -> (T, H, W, C)
        type_rgb_resized = torch.nn.functional.interpolate(
            type_pred_combined_rgb.permute(0, 3, 1, 2),  # (T, C, H, W)
            size=(frame_height, frame_width),
            mode="nearest",
        ).permute(0, 2, 3, 1)  # (T, H, W, C)

        argmax_resized = torch.nn.functional.interpolate(
            argmax_rgb.permute(0, 3, 1, 2), size=(frame_height, frame_width), mode="nearest"
        ).permute(0, 2, 3, 1)

        # Blend only foreground pixels (non-black color). BACKGROUND is black (0,0,0);
        # blending it unconditionally would darken background regions to 0.7×camera.
        fg_mask = type_rgb_resized.sum(dim=-1, keepdim=True) > 1e-6  # (T, H, W, 1)
        blended = torch.clamp(0.7 * chunk_camera + 0.3 * type_rgb_resized, 0, 1)
        overlay = torch.where(fg_mask, blended, chunk_camera)

        # Stream chunk back to CPU buffers immediately to release GPU memory.
        overlay_video[chunk_start:chunk_end] = overlay.detach().cpu().numpy()
        argmax_video[chunk_start:chunk_end] = argmax_resized.detach().cpu().numpy()

        # Create probability maps for each class - SINGLE-CHANNEL (GPU batch ops)
        for class_id in range(num_type_classes):
            prob_map = chunk_type_probabilities[:, class_id, :, :]  # (chunk_frames, H, W) - GPU

            prob_map_resized = torch.nn.functional.interpolate(
                prob_map.unsqueeze(1),  # (chunk_frames, 1, H, W)
                size=(frame_height, frame_width),
                mode="nearest",
            ).squeeze(1)  # (chunk_frames, H, W)

            prob_maps[class_id, chunk_start:chunk_end] = prob_map_resized.detach().cpu().numpy()

    # Free GPU tensors we no longer need before kicking off ffmpeg / NVENC, which itself allocates a non-trivial CUDA
    # workspace.
    del camera_tensor, type_probabilities, type_pred_class_indices, type_pred_probabilities
    del patch_type_logits, type_colors_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create the 5x6 tiled video
    video_path = os.path.join(output_dir, f"detailed_patch_type_vis_iter_{iteration:06d}.mp4")

    # Add text overlays to probability maps using the class mapping for correct labels. Broadcast single-channel prob
    # maps to RGB lazily (zero-copy view).
    assert object_type_class_mapping is not None, "object_type_class_mapping required for type visualization"
    prob_maps_with_text = {}
    for class_id in range(num_type_classes):
        prob_maps_with_text[class_id] = _add_probability_map_text_overlays(
            np.broadcast_to(prob_maps[class_id, ..., None], (num_frames, frame_height, frame_width, 3)),
            class_id,
            0,  # frame_offset
            class_mapping=object_type_class_mapping,
        )

    # Prepare the 30 tiles in 5x6 layout (row-major order):
    # Expected: 22 object type classes + 1 argmax + 1 camera + 1 world + 1 overlay = 26 tiles
    # Row 1 (6 tiles): camera, world, overlay, class 0-2 prob maps
    # Rows 2-4 (18 tiles): class 3-20 prob maps
    # Row 5 (6 tiles): class 21 prob map, argmax, 4 empty
    tiles = [
        camera_frames,  # Row 1, Col 1
        world_frames,  # Row 1, Col 2
        overlay_video,  # Row 1, Col 3
    ]

    # Add first 3 probability maps to complete row 1
    for class_id in range(min(3, num_type_classes)):
        tiles.append(prob_maps_with_text[class_id])

    # Add remaining probability maps (classes 3-21)
    for class_id in range(3, num_type_classes):
        tiles.append(prob_maps_with_text[class_id])

    # Add argmax
    tiles.append(argmax_video)

    # Add empty tiles to fill out the 5x6 grid (30 tiles total)
    num_tiles_so_far = 3 + num_type_classes + 1  # camera, world, overlay, all prob maps, argmax
    num_empty = 30 - num_tiles_so_far
    empty_tile = np.zeros((num_frames, frame_height, frame_width, 3), dtype=np.float32)
    for _ in range(num_empty):
        tiles.append(empty_tile)

    # Convert all tiles to uint8
    tiles_uint8 = [image_video_utils.to_uint8(tile) for tile in tiles]

    # Create output directory if needed
    if not os.path.exists(os.path.dirname(video_path)):
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Write video using ffmpeg (5 rows x 6 columns; NVENC when available, else CPU)
    out_height = frame_height * 5
    out_width = frame_width * 6

    logger.info("Starting 5x6 video encoding: %s frames at %sx%s", num_frames, out_width, out_height)

    frame_gen = image_video_utils.tiled_frames_generator(
        tiles_uint8,
        5,  # rows
        6,  # cols
        num_frames,
        frame_height,
        frame_width,
        use_gpu=True,  # Use GPU for frame resizing
    )
    image_video_utils.write_video_with_ffmpeg(frame_gen, video_path, out_width, out_height, camera_fps)

    logger.info("Detailed 5x6 patch object type visualization video saved to: %s", video_path)

    # Clear GPU memory after video creation
    torch.cuda.empty_cache()
    logger.info("GPU memory cleared after detailed patch type visualization video creation")


def _render_class_overlay_video(
    camera_frames: np.ndarray,
    pred_class_indices: torch.Tensor,
    pred_probabilities: torch.Tensor,
    class_colors_tensor: torch.Tensor,
    num_frames: int,
    frame_width: int,
    frame_height: int,
    chunk_size: int,
    *,
    overlay_label: str,
    progress_desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Colorize per-pixel class predictions and blend them over the camera frames, in memory-bounded chunks.

    Shared by the similarity and object-type columns of :func:`create_combined_patch_visualization`, which run the
    identical pipeline: look up a per-class RGB color, scale it by the per-pixel confidence, nearest-resize to the
    camera resolution, and alpha-blend ``0.7 * camera + 0.3 * prediction``.

    Args:
        camera_frames: ``(>=num_frames, H, W, 3)`` float ``[0, 1]`` camera frames at display resolution.
        pred_class_indices: ``(>=num_frames, H', W')`` integer class indices (argmax output), on any device.
        pred_probabilities: ``(>=num_frames, H', W')`` per-pixel confidence in ``[0, 1]`` for the predicted class.
        class_colors_tensor: ``(num_classes, 3)`` RGB colors indexed by class id.
        num_frames: Number of leading frames to render. The prediction/camera tensors may be longer (e.g. when
            ``pred_class_indices`` was not subset alongside the frame selection); only the first ``num_frames`` are
            used, matching the legacy behavior that bounded the loop by ``probabilities.shape[0]``.
        frame_width: Target width (camera resolution) for the resized prediction.
        frame_height: Target height (camera resolution) for the resized prediction.
        chunk_size: Number of frames to colorize per chunk (bounds peak memory at native resolution).
        overlay_label: Top-right text label drawn on the returned overlay video (e.g. ``"Similarity"``).
        progress_desc: tqdm description for the per-chunk progress bar (rank-0 only under distributed).

    Returns:
        ``(prediction_video, overlay_video)`` — both ``(num_frames, frame_height, frame_width, 3)`` float ``[0, 1]``.
        ``prediction_video`` is the colorized prediction; ``overlay_video`` is the camera/prediction blend with
        ``overlay_label`` drawn on it.
    """
    prediction_video = np.zeros((num_frames, frame_height, frame_width, 3), dtype=np.float32)
    overlay_video = np.zeros((num_frames, frame_height, frame_width, 3), dtype=np.float32)

    chunk_ranges = list(range(0, num_frames, chunk_size))
    show_progress = (not dist.is_initialized()) or dist.get_rank() == 0
    progress_bar = tqdm(chunk_ranges, desc=progress_desc, unit="chunk") if show_progress else None

    for chunk_start in chunk_ranges:
        chunk_end = min(chunk_start + chunk_size, num_frames)
        chunk_frames = chunk_end - chunk_start
        if progress_bar is not None:
            progress_bar.set_description(f"{progress_desc} {chunk_start}-{chunk_end}")
            progress_bar.update(1)

        chunk_camera = camera_frames[chunk_start:chunk_end]
        chunk_indices = pred_class_indices[chunk_start:chunk_end]  # (chunk, H', W')
        chunk_probs = pred_probabilities[chunk_start:chunk_end]  # (chunk, H', W')

        # (chunk, H', W', 3) per-pixel color, scaled by confidence, on the color tensor's device.
        rgb_colors = class_colors_tensor[chunk_indices]
        probs_expanded = chunk_probs.unsqueeze(-1).expand(-1, -1, -1, 3).to(rgb_colors.device)
        combined_rgb = rgb_colors * probs_expanded

        for i in range(chunk_frames):
            rgb_prediction = combined_rgb[i].detach().cpu().numpy()
            rgb_prediction_resized = cv2.resize(
                rgb_prediction, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST
            )
            # Only blend the prediction over pixels with non-black color (non-background classes).
            # DONT_CARE (background) is mapped to black (0,0,0); blending it in would darken those
            # regions to 0.7×camera, making background appear darker than the original frame.
            fg_mask = rgb_prediction_resized.sum(axis=-1, keepdims=True) > 1e-6  # (H, W, 1)
            blended = np.clip(0.7 * chunk_camera[i] + 0.3 * rgb_prediction_resized, 0, 1)
            overlay = np.where(fg_mask, blended, chunk_camera[i])
            frame_idx = chunk_start + i
            prediction_video[frame_idx] = rgb_prediction_resized
            overlay_video[frame_idx] = overlay

    if progress_bar is not None:
        progress_bar.close()

    overlay_video = _add_label_overlay(overlay_video, overlay_label)
    return prediction_video, overlay_video


def create_combined_patch_visualization(
    output_batch: dict[str, torch.Tensor],
    data_batch: dict[str, torch.Tensor],
    iteration: int,
    output_dir: str,
    patch_results: dict[str, torch.Tensor],
    use_gpu_encoding: bool = True,
    frame_indices: torch.Tensor | None = None,
    object_type_class_mapping: type[class_mapping_module.ObjectTypeClassMapping] | str | None = None,
    activation: str = "sigmoid",
    presence_mask: torch.Tensor | np.ndarray | None = None,
    presence_label: str | None = None,
) -> None:
    """
    Create a combined patch visualization tiled video.

    Layout depends on which model heads produced output:

    2x2 layout (if only similarity predictions):
    - Top-left: camera video with text overlay (class percentages)
    - Top-right: camera with model similarity prediction overlay (labeled "Similarity")
    - Bottom-left: world video with text overlay (class percentages)
    - Bottom-right: model similarity prediction (RGB colorized with intensity = probability)

    2x3 layout (if object type predictions are available but no real similarity logits):
    - Top-left: camera video with text overlay
    - Top-middle: camera with similarity prediction overlay (labeled "Similarity")
    - Top-right: camera with object type prediction overlay (labeled "Object Type")
    - Bottom-left: world video with text overlay
    - Bottom-middle: similarity prediction (RGB colorized)
    - Bottom-right: object type prediction (RGB colorized)

    3x3 layout with mask trio (object type predictions AND real similarity logits available):
    Same as 2x3 above for rows 1-2, plus a row-3 mask trio. The trio is built in any mode (training/eval/inference) —
    gating is purely on whether the necessary predictions are available, not on which mode the caller is in.
    - Row 3 col 1: camera frame multiplied by the per-pixel match+extra mask
    - Row 3 col 2: the mask itself, ``clamp(P(MATCH) + P(EXTRA), 0, 1)``, broadcast to RGB as grayscale.
    - Row 3 col 3: object-type prediction multiplied by the same mask. The object-type colormap is preserved; the mask
                   only modulates opacity, so col 3 reads as "what type is each match/extra pixel (with confidence as
                   opacity)".

    Args:
        output_batch: Model output containing patch predictions
        data_batch: Data batch containing camera and world frames
        iteration: Iteration number
        output_dir: Directory to save the visualization
        patch_results: Pre-computed patch predictions and percentages from CWIP model
        frame_indices: Optional (N, 2) tensor of [world_idx, camera_idx] pairs to visualize. If None, visualizes all
                      frames. If provided, only visualizes selected pairs.
        object_type_class_mapping: Class mapping for object type visualization. Can be a string ("v1", "v2", "v3") or
                                   mapping class. Required for correct color rendering when model class indices don't
                                   match ObjectType values.
        activation: Activation function for converting logits to probabilities ('sigmoid' or 'softmax').
        presence_mask: Optional ``(N, h, w)`` 0/1 mask (one per frame, before any ``frame_indices`` selection) to use
                       for the row-3 trio instead of the soft similarity mask, e.g. the static object check's presence
                       mask for a given presence source. The trio's labels then read "Presence Mask (<presence_label>)",
                       "Cam x Presence", and "Type x Presence".
        presence_label: Text naming ``presence_mask`` in the row-3 labels (e.g. ``"presence_source=union"``).
    """
    # Resolve string mapping to class
    if isinstance(object_type_class_mapping, str):
        object_type_class_mapping = class_mapping_module.get_object_type_class_mapping(object_type_class_mapping)
    # Track whether we ran with real similarity logits (vs falling back to type logits at
    # line ~870 below). The row-3 mask trio is only meaningful for real similarity heads —
    # ``MATCH`` and ``EXTRA`` are not class indices in any other head.
    similarity_logits_present = output_batch.get("patch_similarity_logits") is not None
    # Camera-only / any type-only call has no real similarity head. Without it the "Similarity"
    # column below is just the type logits rendered with the wrong (similarity) colormap under a
    # misleading label (and the mask trio is already skipped). In that case render a compact
    # type-only layout (object-type overlay + prediction) and drop both the bogus similarity
    # column and the meaningless all-black world row.
    type_only_layout = (not similarity_logits_present) and (output_batch.get("patch_type_logits") is not None)
    # Extract pre-computed results from CWIP model (already selected above if frame_indices provided)
    pred_class_indices = patch_results["pred_class_indices"].cpu()  # (B, H, W)
    pred_probabilities = patch_results["pred_probabilities"].cpu()  # (B, H, W)

    # Get camera and world frames - use same logic as create_tiled_video
    result = basic_vis_utils.extract_frames_from_data_batch(data_batch)
    if result is None or result[0] is None:
        logger.error("Failed to extract frames from data batch")
        return
    camera_frames, world_frames, camera_fps = result
    assert camera_frames is not None and world_frames is not None and camera_fps is not None

    # Apply frame selection if indices are provided
    if frame_indices is not None:
        world_frame_indices = frame_indices[:, 0].cpu().tolist()
        camera_frame_indices = frame_indices[:, 1].cpu().tolist()
        logger.info("  - Using balanced frame indices: %s pairs", len(camera_frame_indices))

        # Select frames
        camera_frames = camera_frames[camera_frame_indices]
        world_frames = world_frames[world_frame_indices]

        # Select corresponding predictions (probabilities and logits are per-frame)
        patch_similarity_logits_full = output_batch.get("patch_similarity_logits")
        if patch_similarity_logits_full is None:
            patch_similarity_logits_full = output_batch.get("patch_type_logits")
        assert patch_similarity_logits_full is not None, "no patch logits available in output_batch"
        patch_similarity_logits = patch_similarity_logits_full[camera_frame_indices].cpu()

        # Also update patch_results to only include selected frames
        for key, val in patch_results.items():
            if isinstance(val, torch.Tensor) and val.shape[0] == patch_similarity_logits_full.shape[0]:
                patch_results[key] = val[camera_frame_indices]
    else:
        # Use all frames
        patch_similarity_logits_full = output_batch.get("patch_similarity_logits")
        if patch_similarity_logits_full is None:
            patch_similarity_logits_full = output_batch.get("patch_type_logits")
        assert patch_similarity_logits_full is not None, "no patch logits available in output_batch"
        patch_similarity_logits = patch_similarity_logits_full.cpu()

    # Add text overlays showing class percentages
    camera_frames_with_text = _add_patch_text_overlays(camera_frames, patch_results, 0, "camera")
    world_frames_with_text = _add_patch_text_overlays(world_frames, patch_results, 0, "world")

    num_classes = patch_similarity_logits.shape[1]

    # Process frames in chunks to avoid memory issues
    num_frames = patch_similarity_logits.shape[0]
    chunk_size = 64  # Process 64 frames at a time

    # Pre-allocate arrays for the final results using original frame dimensions
    frame_height, frame_width = camera_frames.shape[1], camera_frames.shape[2]
    # The similarity column is skipped entirely in type-only mode (camera-only), so don't render or even allocate its
    # (potentially multi-GB at native resolution) per-frame buffers.
    rgb_predictions_video = None
    overlays_video = None
    if not type_only_layout:
        rgb_predictions_video, overlays_video = _render_class_overlay_video(
            camera_frames,
            pred_class_indices,
            pred_probabilities,
            _get_similarity_colors_tensor(num_classes),
            num_frames,
            frame_width,
            frame_height,
            chunk_size,
            overlay_label="Similarity",
            progress_desc="Processing video chunks",
        )

    # Overall class percentages are already computed and logged above using the CWIP model method

    # Check if object type predictions are available
    patch_type_logits = output_batch.get("patch_type_logits")
    object_type_overlay_video = None
    object_type_prediction_video = None

    if patch_type_logits is not None:
        logger.info("Object type predictions available, creating 2x3 layout with object type column")
        patch_type_logits = patch_type_logits.cpu()  # (B, num_type_classes, H, W)
        type_probabilities = metrics_utils.apply_logits_activation(patch_type_logits, activation=activation)

        # Get object type colors using the canonical colormap for this class mapping
        num_type_classes = patch_type_logits.shape[1]

        # Mask excluded classes before argmax (for combined visualization only)
        # Individual class visualizations show all classes, but combined argmax visualization excludes these
        excluded_type = output_batch.get("excluded_model_classes_for_argmax_type", None)
        if excluded_type:
            masked_type_probs = type_probabilities.clone()
            for excluded_idx in excluded_type:
                if 0 <= excluded_idx < type_probabilities.shape[1]:
                    masked_type_probs[:, excluded_idx, :, :] = 0.0
            type_probabilities = masked_type_probs

        # Compute object type predictions
        type_pred_class_indices = type_probabilities.argmax(dim=1)  # (B, H, W)
        type_pred_probabilities = type_probabilities.max(dim=1)[0]  # (B, H, W)

        # Convert to tensor for efficient processing, using class mapping for correct color lookup
        type_colors_tensor = _build_type_colors_tensor(num_type_classes, object_type_class_mapping, device="cpu")

        object_type_prediction_video, object_type_overlay_video = _render_class_overlay_video(
            camera_frames,
            type_pred_class_indices,
            type_pred_probabilities,
            type_colors_tensor,
            num_frames,
            frame_width,
            frame_height,
            chunk_size,
            overlay_label="Object Type",
            progress_desc="Object type overlay",
        )
        logger.info(
            "Object type visualizations created: overlay shape=%s, prediction shape=%s",
            object_type_overlay_video.shape,
            object_type_prediction_video.shape,
        )

    # ------------------------------------------------------------------
    # Row 3: per-pixel similarity mask + masked previews.
    #
    # The 3x3 layout (2x3 base + this row) is mode-agnostic — it only depends on the input tensors and is built whenever
    # the gating conditions below are met, regardless of whether the caller is running training, eval, or inference.
    #
    # Mask is the union of the canonical "model thinks an object is here" similarity classes (MATCH + EXTRA + WRONG by
    # default — see ``CWIPInference.DEFAULT_PATCH_SIMILARITY_MASK_CLASSES``), clamped to ``[0, 1]``. The full row 3
    # trio:
    #   col 1: camera frame x mask
    #   col 2: the mask itself (RGB-broadcast grayscale)
    #   col 3: object-type prediction x mask (object-type colormap retained,
    #          opacity gated by mask)
    # We only build it when (a) we have real similarity logits (so the similarity-class indices are meaningful) and (b)
    # we have object-type predictions to populate col 3. Otherwise downstream falls back to plain 2x3 / 2x2.
    # ------------------------------------------------------------------
    similarity_mask_video = None
    camera_masked_video = None
    object_type_masked_prediction_video = None

    # Derive the required class-index range from the canonical default so this gate stays in lockstep with the actual
    # mask computation. Hard-coding ``max(MATCH, EXTRA)`` here was the bug shape that made it possible for the default
    # to expand (to include WRONG) without anything in this gating check noticing. The constant and helper live in this
    # module (re-exported on ``CWIPInference``) so we can reference them directly without the lazy-import-from-cwip
    # dance that used to be required to break a runtime cycle.
    default_class_indices = [int(c) for c in DEFAULT_PATCH_SIMILARITY_MASK_CLASSES]
    can_build_row3 = (
        similarity_logits_present
        and object_type_prediction_video is not None
        and all(0 <= idx < num_classes for idx in default_class_indices)
    )
    if can_build_row3:
        # ``object_type_prediction_video`` is guaranteed non-None by the ``can_build_row3`` gate above.
        assert object_type_prediction_video is not None
        # ``patch_similarity_logits`` is already on CPU per the assignment above, so the returned mask is on CPU too. We
        # pay one extra activation pass over the whole (T, num_classes, H, W) tensor here (the function applies the
        # activation internally), but this is a cheap elementwise op compared to the model forward pass.
        # We use the canonical default ``similarity_classes`` — ``MATCH + EXTRA + WRONG`` (the three "model thinks an
        # object is here" classes) — implicitly rather than re-stating it, so the visualization tracks any future change
        # to the canonical default in lockstep.
        if presence_mask is not None:
            # Caller-supplied presence mask (e.g. the static check's, for a chosen presence source) replaces the soft
            # similarity mask; select the same camera frames as the rest of the layout.
            mask_patch = torch.as_tensor(presence_mask)
            if frame_indices is not None:
                mask_patch = mask_patch[frame_indices[:, 1].cpu()]
            if mask_patch.shape[0] != num_frames:
                raise ValueError(f"presence_mask has {mask_patch.shape[0]} frames, expected {num_frames}")
        else:
            similarity_mask = compute_patch_similarity_class_mask(patch_similarity_logits, activation=activation)
            # No ``patch_type_logits`` passed, so a single mask tensor is returned (not the mask/probs tuple).
            assert isinstance(similarity_mask, torch.Tensor)
            mask_patch = similarity_mask
        # ``compute_patch_similarity_class_mask`` preserves the input dtype (validated by its unit tests). Validation
        # runs in mixed precision (bfloat16), so the mask comes back as bfloat16 — and ``Tensor.numpy()`` rejects
        # bfloat16 with ``TypeError: Got unsupported ScalarType BFloat16``. The other ``.numpy()`` calls in this
        # function (lines ~962, ~1043) accidentally avoid this because they multiply by a float32 ``colors_tensor``
        # first (FP32 x BF16 promotes to FP32). We don't have that promotion here, so cast explicitly. ``.float()`` is a
        # no-op when the input is already float32, so this is safe across all dtypes.
        mask_patch_np = mask_patch.detach().float().cpu().numpy()  # (T, H_patch, W_patch) float32

        similarity_mask_video = np.zeros((num_frames, frame_height, frame_width, 3), dtype=np.float32)
        camera_masked_video = np.zeros_like(similarity_mask_video)
        object_type_masked_prediction_video = np.zeros_like(similarity_mask_video)

        for chunk_start in range(0, num_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_frames)
            for i in range(chunk_start, chunk_end):
                # Resize mask to frame resolution. Nearest-neighbor matches the rgb_predictions /
                # object_type_prediction resize above so the row-3 panels align pixel-perfectly with rows 1 & 2.
                mask_resized = cv2.resize(
                    mask_patch_np[i],
                    (frame_width, frame_height),
                    interpolation=cv2.INTER_NEAREST,
                )  # (H, W) float32 in [0, 1]
                mask_3ch = np.stack([mask_resized] * 3, axis=-1)  # (H, W, 3)

                similarity_mask_video[i] = mask_3ch
                camera_masked_video[i] = np.clip(camera_frames[i] * mask_3ch, 0.0, 1.0)
                object_type_masked_prediction_video[i] = np.clip(object_type_prediction_video[i] * mask_3ch, 0.0, 1.0)

        if presence_mask is not None:
            mask_label = f"Presence Mask ({presence_label})" if presence_label else "Presence Mask"
            cam_label, type_label = "Cam x Presence", "Type x Presence"
        else:
            mask_label, cam_label, type_label = "Similarity Mask", "Cam x Mask", "Type x Mask"
        similarity_mask_video = _add_label_overlay(similarity_mask_video, mask_label)
        camera_masked_video = _add_label_overlay(camera_masked_video, cam_label)
        object_type_masked_prediction_video = _add_label_overlay(object_type_masked_prediction_video, type_label)
        logger.info("Row-3 mask trio built — similarity mask, camera x mask, object-type x mask")
    elif object_type_prediction_video is not None and not similarity_logits_present:
        logger.info(
            "Skipping 3x3 row-3 mask trio: only patch_type_logits available "
            "(no patch_similarity_logits) — similarity-class indices would be undefined."
        )

    # Create tiled video using the same approach as create_tiled_video
    # Create output path
    video_path = os.path.join(output_dir, f"combined_patch_vis_iter_{iteration:06d}.mp4")

    # Call the consolidated function to create tiled video
    if type_only_layout and object_type_overlay_video is not None:
        # Camera-only / type-only: no similarity head, so drop both the bogus similarity column and the
        # all-black world row. Render a 2x2 with the camera as a per-row reference (16:9-friendly on screen):
        #   Row 1: camera | camera+object-type overlay
        #   Row 2: camera | object-type prediction (type mask)
        assert object_type_prediction_video is not None
        logger.info("Creating 2x2 type-only combined_patch_vis (camera-only: no similarity column)")
        logger.info(
            "[PATCH_VIS] Encoding combined_patch_vis 2x2 type-only (%s frames) -> %s",
            num_frames,
            video_path,
        )
        camera_labeled = image_video_utils.to_uint8(_add_label_overlay(camera_frames, "Camera"))
        overlay_u8 = image_video_utils.to_uint8(object_type_overlay_video)
        type_mask_u8 = image_video_utils.to_uint8(_add_label_overlay(object_type_prediction_video, "Type Mask"))
        type_only_tiles = [camera_labeled, overlay_u8, camera_labeled, type_mask_u8]
        image_video_utils.write_tiled_video(
            type_only_tiles,
            2,
            2,
            num_frames,
            frame_height,
            frame_width,
            video_path,
            camera_fps,
            use_gpu=use_gpu_encoding,
        )
    elif object_type_overlay_video is not None:
        # 2x3 (or 3x3 with mask trio, when row-3 was built) layout with object type predictions.
        # Row 1: camera w/ text   | similarity overlay     | object-type overlay
        # Row 2: world w/ text    | similarity prediction  | object-type prediction
        # Row 3 (when present):   camera x mask | similarity mask | object-type x mask
        # Reached only when not type_only_layout, so the similarity column was rendered above.
        assert overlays_video is not None and rgb_predictions_video is not None
        layout_label = "3x3 with mask trio" if similarity_mask_video is not None else "2x3"
        logger.info("Creating %s tiled video with object type predictions", layout_label)
        logger.info(
            "[PATCH_VIS] Encoding combined_patch_vis %s (%s frames) -> %s",
            layout_label,
            num_frames,
            video_path,
        )
        basic_vis_utils.tile_and_save_media(
            camera_images=camera_frames_with_text,  # Row 1 col 1: camera with text overlays
            world_images=world_frames_with_text,  # Row 2 col 1: world with text overlays
            camera_spatial_features=overlays_video,  # Row 1 col 2: camera with similarity overlay
            world_spatial_features=rgb_predictions_video,  # Row 2 col 2: similarity prediction
            object_type_overlay=object_type_overlay_video,  # Row 1 col 3: camera with object type overlay
            object_type_prediction=object_type_prediction_video,  # Row 2 col 3: object type prediction
            # Row 3 trio (None when we couldn't build it; downstream falls back to 2x3):
            camera_masked_by_similarity=camera_masked_video,
            similarity_mask=similarity_mask_video,
            object_type_masked_prediction=object_type_masked_prediction_video,
            output_path=video_path,
            output_mode=basic_vis_utils.TilingOutputMode.VIDEO,
            fps=camera_fps,
            use_gpu_encoding=use_gpu_encoding,
        )
    else:
        # 2x2 layout without object type predictions (original layout)
        # Row 1: camera with text, similarity overlay
        # Row 2: world with text, similarity prediction
        # Reached only when not type_only_layout, so the similarity column was rendered above.
        assert overlays_video is not None and rgb_predictions_video is not None
        logger.info("Creating 2x2 tiled video (no object type predictions)")
        logger.info(
            "[PATCH_VIS] Encoding combined_patch_vis 2x2 (%s frames) -> %s",
            num_frames,
            video_path,
        )
        basic_vis_utils.tile_and_save_media(
            camera_images=camera_frames_with_text,  # Top-left: camera with text overlays
            world_images=overlays_video,  # Top-right: camera with model similarity prediction overlay
            camera_spatial_features=world_frames_with_text,  # Bottom-left: world with text overlays
            world_spatial_features=rgb_predictions_video,  # Bottom-right: model similarity prediction
            output_path=video_path,
            output_mode=basic_vis_utils.TilingOutputMode.VIDEO,
            fps=camera_fps,
            use_gpu_encoding=use_gpu_encoding,
        )

    logger.info("Combined patch visualization video saved to: %s", video_path)

    # The radial class-percentage / pixel-count plots summarize the similarity-branch class
    # distribution. In camera-only (type-only) mode there is no world input and the similarity
    # branch never ran, so these plots would be derived from the type logits under similarity
    # semantics — misleading. Skip them entirely; the object-type distribution is covered by the
    # detailed object-type visualization.
    if type_only_layout:
        logger.info("Skipping radial class-percentage / pixel-count plots (camera-only: no similarity branch)")
    else:
        # Create radial class percentage plot
        if "per_frame_class_percentages" in patch_results:
            radial_plot_path = os.path.join(output_dir, f"class_percentage_radial_iter_{iteration:06d}.png")
            basic_vis_utils.create_radial_class_percentage_plot(
                cast(dict[Any, Any], patch_results["per_frame_class_percentages"]),
                radial_plot_path,
                cast(dict[Any, Any] | None, patch_results.get("class_percentages", None)),
            )
            logger.info("Radial class percentage plot saved to: %s", radial_plot_path)

        # Create radial class pixel count plot
        if "per_frame_class_pixel_counts" in patch_results:
            pixel_radial_plot_path = os.path.join(output_dir, f"class_pixel_count_radial_iter_{iteration:06d}.png")
            basic_vis_utils.create_radial_class_pixel_count_plot(
                cast(dict[Any, Any], patch_results["per_frame_class_pixel_counts"]), pixel_radial_plot_path
            )
            logger.info("Radial class pixel count plot saved to: %s", pixel_radial_plot_path)


def _add_probability_map_text_overlays(
    prob_map_images: np.ndarray,
    class_id: int,
    frame_offset: int,
    class_mapping: type[class_mapping_module.ObjectTypeClassMapping]
    | type[class_mapping_module.SimilarityClassMapping],
) -> np.ndarray:
    """
    Add class name and frame number as text overlays to probability map visualizations.

    Args:
        prob_map_images: Input probability map images array (num_frames, H, W, 3) in range [0, 1]
        class_id: The class ID for the probability map (model class index)
        frame_offset: Frame offset for numbering
        class_mapping: Class mapping to use for name resolution. Uses metrics_utils.get_class_name_from_mapping for
                       correct name lookup. Should be class_mapping_module.ObjectTypeClassMapping for type
                       visualization, or class_mapping_module.SimilarityClassMapping for similarity visualization.

    Returns:
        images_with_text: Images with text overlays
    """
    # Make a copy to avoid modifying the original
    images_with_text = prob_map_images.copy()

    # Convert to uint8 for OpenCV
    images_uint8 = (images_with_text * 255).astype(np.uint8)

    # Get class name using the class mapping
    class_name = metrics_utils.get_class_name_from_mapping(class_id, class_mapping)

    # Add class name + frame number to each frame's top-right corner.
    for frame_idx in range(len(images_uint8)):
        frame_text = f"Frame {frame_offset + frame_idx}"
        class_text = f"{class_name} Probability"
        image_video_utils.draw_topright_lines(
            images_uint8[frame_idx], [class_text, frame_text], font_scale=0.7, y_start=20, y_spacing=20
        )

    # Convert back to float [0, 1]
    images_with_text = images_uint8.astype(np.float32) / 255.0

    return images_with_text


def _add_label_overlay(
    images: np.ndarray,
    label: str,
) -> np.ndarray:
    """
    Add a simple text label overlay to images in the top-right corner.

    Args:
        images: Input images array (num_frames, H, W, 3) in range [0, 1]
        label: Text label to display

    Returns:
        images_with_label: Images with label overlay
    """
    # Convert to uint8 for cv2
    images_uint8 = (images * 255).astype(np.uint8)

    # Single top-right label per frame (matches the _add_patch_text_overlays styling).
    for i in range(images_uint8.shape[0]):
        image_video_utils.draw_topright_lines(images_uint8[i], [label], font_scale=0.7, y_start=20, y_spacing=20)

    # Convert back to float [0, 1]
    images_with_label = images_uint8.astype(np.float32) / 255.0

    return images_with_label


def _add_patch_text_overlays(
    images: np.ndarray, patch_results: dict[str, torch.Tensor], frame_offset: int, view_type: str
) -> np.ndarray:
    """
    Add frame numbers and class percentages as text overlays in the top right corner.

    Args:
        images: Input images array (num_frames, H, W, 3) in range [0, 1]
        patch_results: Patch prediction results containing per-frame class percentages
        frame_offset: Frame offset for numbering
        view_type: "camera" or "world" for labeling

    Returns:
        images_with_text: Images with text overlays
    """
    # Make a copy to avoid modifying the original
    images_with_text = images.copy()

    # Convert to uint8 for OpenCV
    images_uint8 = (images_with_text * 255).astype(np.uint8)

    # Get per-frame class percentages (heterogeneous dict value, not a Tensor despite the dict annotation)
    per_frame_class_percentages = cast(dict[int, Any], patch_results.get("per_frame_class_percentages", {}))

    # Class names and colors for display - use brighter colors for better readability
    # Get class names from world_model_utils.PatchSimilarityType enum
    class_names = [patch_type.name for patch_type in world_model_utils.PatchSimilarityType]
    similarity_palette = world_model_utils.get_similarity_type_colors(
        include_ignored=False, num_classes=len(class_names)
    )
    class_colors = []
    for i in range(len(class_names)):
        rgb_color = similarity_palette[i]  # (R, G, B) in range [0, 1]; keys 0..len-1 always present
        rgb_color_255 = (int(rgb_color[0] * 255), int(rgb_color[1] * 255), int(rgb_color[2] * 255))
        class_colors.append(rgb_color_255)

    # Add text to each frame
    for frame_idx in range(len(images_uint8)):
        frame = images_uint8[frame_idx]

        # Frame number (with offset)
        frame_num = frame_offset + frame_idx
        frame_text = f"Frame {frame_num}"

        # View type label
        view_text = f"{view_type.capitalize()}"

        # Get image dimensions
        _h, w = frame.shape[:2]

        # Position view type + frame number in the top right corner.
        y_start = 20
        image_video_utils.draw_topright_lines(
            frame, [view_text, frame_text], font_scale=0.7, y_start=y_start, y_spacing=20
        )

        # Add class percentages and pixel counts. These are drawn per-class in the class's own color (not the shared
        # white top-right style), at non-contiguous y positions keyed by class_id, so they stay a custom loop rather
        # than using image_video_utils.draw_topright_lines.
        y_start += 60  # Start below frame info
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (255, 255, 255)  # White text (used for the Total line)
        outline_color = (0, 0, 0)  # Black outline
        y_spacing = 20

        # Get per-frame pixel counts if available (heterogeneous dict value, not a Tensor)
        per_frame_class_pixel_counts = cast(dict[int, Any], patch_results.get("per_frame_class_pixel_counts", {}))

        for class_id, percentages in per_frame_class_percentages.items():
            if class_id < len(class_names) and frame_idx < len(percentages):
                class_name = class_names[class_id]
                percentage = percentages[frame_idx]

                # Get pixel count for this class and frame
                pixel_count = 0
                if class_id in per_frame_class_pixel_counts and frame_idx < len(per_frame_class_pixel_counts[class_id]):
                    pixel_count = per_frame_class_pixel_counts[class_id][frame_idx]

                # class_id is 1,2,3,4 but class_colors is indexed 0,1,2,3,4
                class_color = class_colors[class_id] if class_id < len(class_colors) else (255, 255, 255)

                # Format percentage and pixel count text
                percentage_text = f"{class_name}: {percentage:.1f}% ({pixel_count}px)"

                # Get text size for positioning
                (text_w, _), _baseline = cv2.getTextSize(percentage_text, font, font_scale, thickness)

                # Position in top right corner
                x = w - text_w - 10
                y = y_start + (class_id - 1) * y_spacing  # class_id - 1 because we skip DONT_CARE

                # Draw text outline (black) - thicker outline for better contrast
                cv2.putText(frame, percentage_text, (x, y), font, font_scale, outline_color, thickness + 2)
                # Draw text with class color
                cv2.putText(frame, percentage_text, (x, y), font, font_scale, class_color, thickness)

        # Add total non-dont-care pixel count
        total_non_dont_care_pixels = 0
        for class_id in per_frame_class_pixel_counts:
            if class_id > 0 and frame_idx < len(per_frame_class_pixel_counts[class_id]):  # Skip DONT_CARE (class 0)
                total_non_dont_care_pixels += per_frame_class_pixel_counts[class_id][frame_idx]

        if total_non_dont_care_pixels > 0:
            total_text = f"Total: {total_non_dont_care_pixels}px"
            (text_w, _text_h), _baseline = cv2.getTextSize(total_text, font, font_scale, thickness)
            x = w - text_w - 10
            y = y_start + 4 * y_spacing  # Position below all class percentages

            # Draw total text with white color
            cv2.putText(frame, total_text, (x, y), font, font_scale, outline_color, thickness + 2)
            cv2.putText(frame, total_text, (x, y), font, font_scale, color, thickness)

    # Convert back to float [0, 1]
    images_with_text = images_uint8.astype(np.float32) / 255.0

    return images_with_text
