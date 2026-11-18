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

"""Utility functions for computing patch classification metrics."""

from typing import Any

import torch
import torch.nn.functional as F

from checks.utils.models.cwip.modules import class_mapping as class_mapping_module
from checks.utils.models.cwip.visualization import world_model_utils


def apply_logits_activation(logits: torch.Tensor, activation: str = "sigmoid") -> torch.Tensor:
    """
    Apply activation function to convert logits to probabilities.

    Args:
        logits: Raw logits tensor of shape (B, C, H, W) or (B, C).
        activation: Activation function to apply:
            - 'sigmoid': Independent per-class probabilities (multi-label)
            - 'softmax': Mutually exclusive class probabilities (single-label)

    Returns:
        Probabilities tensor of same shape as logits.
    """
    if activation == "sigmoid":
        return torch.sigmoid(logits)
    elif activation == "softmax":
        return torch.softmax(logits, dim=1)
    else:
        raise ValueError(f"activation must be 'sigmoid' or 'softmax', got '{activation}'")


def get_class_name_from_mapping(
    class_id: int,
    class_mapping: type[class_mapping_module.ClassMapping],
) -> str:
    """
    Get the class name for a given class index using the provided class mapping.

    This is the single source of truth for resolving model class indices to human-readable names. It works for both
    similarity classification and object type classification.

    Args:
        class_id: The model class index (0-based)
        class_mapping: Class mapping to use for name resolution. Must be one of:
            - ObjectTypeClassMapping subclass: Uses MODEL_CLASS_TO_OBJECT_TYPE for type names
            - SimilarityClassMapping subclass: Uses MODEL_CLASS_TO_SIMILARITY_TYPE for similarity names

    Returns:
        str: Human-readable class name (e.g., "BACKGROUND", "BBOX_CAR", "MATCH", etc.)

    Raises:
        ValueError: If class_mapping is not a valid ClassMapping subclass
    """
    if issubclass(class_mapping, class_mapping_module.SimilarityClassMapping):
        # Similarity classification: use MODEL_CLASS_TO_SIMILARITY_TYPE
        if class_id in class_mapping.MODEL_CLASS_TO_SIMILARITY_TYPE:
            return class_mapping.MODEL_CLASS_TO_SIMILARITY_TYPE[class_id].name
        else:
            return f"UNKNOWN_CLASS_{class_id}"
    elif issubclass(class_mapping, class_mapping_module.ObjectTypeClassMapping):
        # Object type classification: use MODEL_CLASS_TO_OBJECT_TYPE
        # This correctly maps V3 class indices (e.g., 15 -> WHITE laneline, 16 -> YELLOW laneline)
        if class_id in class_mapping.MODEL_CLASS_TO_OBJECT_TYPE:
            return class_mapping.MODEL_CLASS_TO_OBJECT_TYPE[class_id].name
        else:
            return f"UNKNOWN_CLASS_{class_id}"
    else:
        raise TypeError(
            f"class_mapping must be a subclass of ObjectTypeClassMapping or SimilarityClassMapping, "
            f"got {type(class_mapping).__name__}"
        )


def get_class_names_from_mapping(
    class_mapping: type[class_mapping_module.ClassMapping],
) -> list[str]:
    """
    Get all class names for a given class mapping.

    This is a convenience function that returns class names for all class indices (0 to NUM_CLASSES-1).

    Args:
        class_mapping: Class mapping to use for name resolution. Must be one of:
            - ObjectTypeClassMapping subclass: Uses MODEL_CLASS_TO_OBJECT_TYPE for type names
            - SimilarityClassMapping subclass: Uses MODEL_CLASS_TO_SIMILARITY_TYPE for similarity names

    Returns:
        list[str]: List of human-readable class names in order of class index

    Raises:
        ValueError: If class_mapping is not a valid ClassMapping subclass
    """
    num_classes = class_mapping.NUM_CLASSES
    return [get_class_name_from_mapping(class_id, class_mapping) for class_id in range(num_classes)]


def compute_patch_predictions_and_percentages(
    patch_similarity_logits: torch.Tensor,
    treat_wrong_type_as_match: bool = False,
    gt_patch_similarity: torch.Tensor | None = None,
    gt_foreground_pixel_counts: torch.Tensor | None = None,
    default_denominator: str = "no_extra",
    activation: str = "sigmoid",
    excluded_model_classes_for_argmax: list[int] | None = None,
) -> dict[str, Any]:
    """
    Compute per-frame per-pixel class indices, probabilities, and percentages from patch logits.

    This function computes THREE versions of percentages:
    1. 'class_percentages': Default denominator (configurable via default_denominator arg)
    2. 'class_percentages_all_classes': All classes (denominator = MATCH + WRONG + EXTRA + MISSING)
    3. 'class_percentages_no_extra': Exclude EXTRA (denominator = MATCH + WRONG + MISSING)

    Args:
        patch_similarity_logits: Patch similarity logits tensor of shape (B, num_classes, H, W)
        treat_wrong_type_as_match: Whether to treat wrong type objects as match
        gt_patch_similarity: Optional ground truth patch similarity tensor of shape (B, num_classes, H, W)
                 (one-hot encoded)
                 When provided, enables pair-wise ground-truth–aware denominators
        gt_foreground_pixel_counts: Optional tensor of shape (B,) containing foreground pixel counts per frame
                 Used for computing GT foreground metrics
        default_denominator: Which denominator to use as default. Options:
            - "no_extra": MATCH + WRONG + MISSING (default)
            - "all_classes": MATCH + WRONG + EXTRA + MISSING
            - "gt_fg": GT foreground pixel count (requires gt_foreground_pixel_counts or gt_patch_similarity)
        activation: Activation function for converting logits to probabilities ('sigmoid' or 'softmax').

    Returns:
        Dictionary containing:
            - 'pred_class_indices': Predicted class indices (B, H, W)
            - 'pred_probabilities': Predicted probabilities (B, H, W)
            - 'class_counts': Count of pixels per class (num_classes,)
            - 'class_percentages': Percentages using default denominator
            - 'class_percentages_all_classes': Percentages using all non-DONT_CARE as denominator
            - 'class_percentages_no_extra': Percentages excluding EXTRA from denominator
            - 'per_frame_class_percentages': Per-frame percentages using default denominator
            - 'per_frame_class_percentages_all_classes': Per-frame all classes percentages
            - 'per_frame_class_percentages_no_extra': Per-frame no EXTRA percentages
            - 'per_frame_class_pixel_counts': Per-frame pixel counts
            - 'total_pixels': Total number of pixels
            - 'total_non_dont_care_pixels': Total pixels using default denominator
    """
    # Convert logits to probabilities
    probabilities = apply_logits_activation(patch_similarity_logits, activation=activation)  # (B, num_classes, H, W)

    # Mask excluded classes before argmax (for combined visualization only)
    # Individual class visualizations show all classes, but combined argmax visualization excludes these
    if excluded_model_classes_for_argmax:
        masked_probs = probabilities.clone()
        for excluded_idx in excluded_model_classes_for_argmax:
            if 0 <= excluded_idx < probabilities.shape[1]:
                masked_probs[:, excluded_idx, :, :] = 0.0
        probabilities = masked_probs

    # Get predicted class indices and probabilities for all frames
    pred_class_indices = probabilities.argmax(dim=1)  # (B, H, W)
    pred_probabilities = probabilities.max(dim=1)[0]  # (B, H, W)

    # Apply wrong type treatment if requested
    if treat_wrong_type_as_match:
        pred_class_indices = pred_class_indices.clone()
        pred_class_indices[pred_class_indices == world_model_utils.PatchSimilarityType.WRONG.value] = (
            world_model_utils.PatchSimilarityType.MATCH.value
        )

    # Compute class percentages (excluding DONT_CARE=0)
    # Count pixels for each class
    class_counts = torch.bincount(pred_class_indices.flatten(), minlength=probabilities.shape[1])

    # Compute all three denominators
    # 1. All classes: all non-DONT_CARE classes (1, 2, 3, 4)
    total_all_classes = class_counts[1:].sum().item()

    # 2. Exclude EXTRA: only classes 1, 2, 4
    total_no_extra = (class_counts[1] + class_counts[2] + class_counts[4]).item()

    # 3. GT foreground: from gt_foreground_pixel_counts when set, else derived from gt_patch_similarity when set
    total_gt_fg = None
    if gt_foreground_pixel_counts is not None:
        # Use pre-computed foreground pixel counts (e.g. from metadata)
        total_gt_fg = gt_foreground_pixel_counts.sum().item()
    elif gt_patch_similarity is not None:
        # Compute from gt_patch_similarity when per-frame counts are absent
        # Resize gt_patch_similarity to match patch_similarity_logits size if needed
        gt_h, gt_w = gt_patch_similarity.shape[-2:]
        pred_h, pred_w = patch_similarity_logits.shape[-2:]

        if (gt_h, gt_w) != (pred_h, pred_w):
            gt_patch_resized = F.interpolate(gt_patch_similarity.float(), size=(pred_h, pred_w), mode="nearest")
        else:
            gt_patch_resized = gt_patch_similarity.float()

        # Get GT class indices (gt_patch_similarity is one-hot encoded)
        gt_class_indices = gt_patch_resized.argmax(dim=1)  # (B, H, W)
        gt_foreground_mask = gt_class_indices > 0
        total_gt_fg = gt_foreground_mask.sum().item()

    total_pixels = class_counts.sum().item()

    # Compute per-frame class percentages and pixel counts
    num_frames = pred_class_indices.shape[0]

    # Initialize dictionaries for all three versions
    per_frame_class_percentages: dict[int, list[float]] = {}  # Default (GT fg if available, else all_classes)
    per_frame_class_percentages_all_classes: dict[int, list[float]] = {}  # All non-DONT_CARE
    per_frame_class_percentages_no_extra: dict[int, list[float]] = {}  # Exclude EXTRA
    per_frame_class_pixel_counts: dict[int, list[float]] = {}

    for class_id in range(1, probabilities.shape[1]):  # Skip DONT_CARE (class 0)
        per_frame_class_percentages[class_id] = []
        per_frame_class_percentages_all_classes[class_id] = []
        per_frame_class_percentages_no_extra[class_id] = []
        per_frame_class_pixel_counts[class_id] = []

    # Collect per-frame denominators
    per_frame_denominators = []  # Default: GT fg if available, else no_extra
    per_frame_denominators_all_classes = []
    per_frame_denominators_no_extra = []
    per_frame_denominators_gt_fg = []

    for frame_idx in range(num_frames):
        frame_class_indices = pred_class_indices[frame_idx]  # (H, W)
        frame_class_counts = torch.bincount(frame_class_indices.flatten(), minlength=probabilities.shape[1])

        # Compute all three denominators for this frame
        frame_total_all_classes = frame_class_counts[1:].sum().item()
        frame_total_no_extra = (frame_class_counts[1] + frame_class_counts[2] + frame_class_counts[4]).item()

        # Get GT foreground count for this frame
        frame_total_gt_fg = None
        if gt_foreground_pixel_counts is not None:
            # Use pre-computed foreground pixel counts (e.g. from metadata)
            frame_total_gt_fg = gt_foreground_pixel_counts[frame_idx].item()
        elif gt_patch_similarity is not None:
            # Compute from gt_patch_similarity when per-frame counts are absent - gt_class_indices was computed earlier
            gt_class_indices = gt_patch_resized.argmax(dim=1)  # (B, H, W)
            frame_gt_class_indices = gt_class_indices[frame_idx]
            frame_gt_foreground_mask = frame_gt_class_indices > 0
            frame_total_gt_fg = frame_gt_foreground_mask.sum().item()

        # Compute percentages for each class using all three denominators
        for class_id in range(1, probabilities.shape[1]):  # Skip DONT_CARE (class 0)
            frame_class_count = frame_class_counts[class_id].item()

            # Version 1: All classes denominator
            if frame_total_all_classes > 0:
                pct_all_classes = (frame_class_count / frame_total_all_classes) * 100.0
            else:
                pct_all_classes = 0.0
            per_frame_class_percentages_all_classes[class_id].append(pct_all_classes)

            # Version 2: No EXTRA denominator (EXTRA excluded from denominator but still gets a percentage)
            if frame_total_no_extra > 0:
                pct_no_extra = (frame_class_count / frame_total_no_extra) * 100.0
            else:
                pct_no_extra = 0.0
            per_frame_class_percentages_no_extra[class_id].append(pct_no_extra)

            # Version 3: GT foreground denominator
            if frame_total_gt_fg is not None and frame_total_gt_fg > 0:
                pct_gt_fg = (frame_class_count / frame_total_gt_fg) * 100.0
            else:
                pct_gt_fg = 0.0

            # Determine default based on default_denominator argument
            if default_denominator == "all_classes":
                pct_default = pct_all_classes
            elif default_denominator == "gt_fg":
                pct_default = pct_gt_fg if frame_total_gt_fg is not None and frame_total_gt_fg > 0 else pct_no_extra
            else:  # default_denominator == "no_extra"
                pct_default = pct_no_extra

            per_frame_class_percentages[class_id].append(pct_default)

            # Pixel count (same for all versions)
            per_frame_class_pixel_counts[class_id].append(frame_class_count)

        # Store denominators for this frame
        per_frame_denominators_all_classes.append(frame_total_all_classes)
        per_frame_denominators_no_extra.append(frame_total_no_extra)
        per_frame_denominators_gt_fg.append(frame_total_gt_fg if frame_total_gt_fg is not None else 0)

        # Determine default denominator based on default_denominator argument
        if default_denominator == "all_classes":
            frame_default_denominator = frame_total_all_classes
        elif default_denominator == "gt_fg":
            frame_default_denominator = frame_total_gt_fg if frame_total_gt_fg is not None else frame_total_no_extra
        else:  # default_denominator == "no_extra"
            frame_default_denominator = frame_total_no_extra

        per_frame_denominators.append(frame_default_denominator)

    # Calculate average class percentages from per-frame percentages for all three versions
    class_percentages = {}
    class_percentages_all_classes = {}
    class_percentages_no_extra = {}

    for class_id in range(1, probabilities.shape[1]):  # Skip DONT_CARE (class 0)
        # Default version (GT fg if available, else all_classes)
        if per_frame_class_percentages[class_id]:
            class_percentages[class_id] = sum(per_frame_class_percentages[class_id]) / len(
                per_frame_class_percentages[class_id]
            )
        else:
            class_percentages[class_id] = 0.0

        # All classes version
        if per_frame_class_percentages_all_classes[class_id]:
            class_percentages_all_classes[class_id] = sum(per_frame_class_percentages_all_classes[class_id]) / len(
                per_frame_class_percentages_all_classes[class_id]
            )
        else:
            class_percentages_all_classes[class_id] = 0.0

        # No EXTRA version
        if per_frame_class_percentages_no_extra[class_id]:
            class_percentages_no_extra[class_id] = sum(per_frame_class_percentages_no_extra[class_id]) / len(
                per_frame_class_percentages_no_extra[class_id]
            )
        else:
            class_percentages_no_extra[class_id] = 0.0

    # Determine total default denominator based on default_denominator argument
    if default_denominator == "all_classes":
        total_default = total_all_classes
    elif default_denominator == "gt_fg":
        total_default = total_gt_fg if total_gt_fg is not None else total_no_extra
    else:  # default_denominator == "no_extra"
        total_default = total_no_extra

    return {
        "pred_class_indices": pred_class_indices,
        "pred_probabilities": pred_probabilities,
        "class_counts": class_counts,
        # Three versions of class percentages
        "class_percentages": class_percentages,  # Default: GT fg if available, else no_extra
        "class_percentages_all_classes": class_percentages_all_classes,
        "class_percentages_no_extra": class_percentages_no_extra,
        # Three versions of per-frame percentages
        "per_frame_class_percentages": per_frame_class_percentages,  # Default: GT fg if available, else no_extra
        "per_frame_class_percentages_all_classes": per_frame_class_percentages_all_classes,
        "per_frame_class_percentages_no_extra": per_frame_class_percentages_no_extra,
        # Other data
        "per_frame_class_pixel_counts": per_frame_class_pixel_counts,
        "total_pixels": total_pixels,
        # Three versions of total non-DONT_CARE pixels (denominators used)
        "total_non_dont_care_pixels": total_default,  # GT fg if available, else no_extra
        "total_non_dont_care_pixels_all_classes": total_all_classes,
        "total_non_dont_care_pixels_no_extra": total_no_extra,
        "total_non_dont_care_pixels_gt_fg": total_gt_fg,
        # Per-frame denominators
        "per_frame_denominators": per_frame_denominators,  # Default: GT fg if available, else no_extra
        "per_frame_denominators_all_classes": per_frame_denominators_all_classes,
        "per_frame_denominators_no_extra": per_frame_denominators_no_extra,
        "per_frame_denominators_gt_fg": per_frame_denominators_gt_fg,
    }
