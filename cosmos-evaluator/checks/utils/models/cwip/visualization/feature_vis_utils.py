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
Feature visualization utilities.
"""

import logging
import os
import time
from typing import Any, cast

import cv2
import einops
from matplotlib import colormaps
import numpy as np
import torch
import torch.nn.functional as F

from checks.utils.models.cwip.visualization import basic_vis_utils, feature_similarity_utils, image_video_utils

logger = logging.getLogger(__name__)


def _pair_rgb_bchw_to_numpy_hwc_batch(t: torch.Tensor) -> np.ndarray:
    """Convert ``(N, C, H, W)`` RGB tensor (typically [-1, 1]) to ``(N, H, W, C)`` float in ``[0, 1]``."""
    x = t.detach().cpu().float()
    x = x.permute(0, 2, 3, 1)
    if float(x.min()) < 0:
        x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
    return x.numpy()


def _get_robust_pca_with_timing(
    features: torch.Tensor,
    m: float = 2,
    remove_first_component: bool = False,
    skip: int = 0,
    max_samples: int = 100_000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom robust PCA implementation with proper BFloat16 handling and detailed timing.

    This function performs PCA on high-dimensional features to reduce them to 3 RGB color channels for visualization. It
    includes robust statistics to handle outliers.

    Args:
        features: Input features tensor of shape (N, C) where N is number of spatial locations
        m: Robust statistics parameter - controls outlier rejection (default: 2.0)
        remove_first_component: Whether to remove the first principal component (default: False)
        skip: Number of initial components to skip (default: 0)
        max_samples: Upper bound on rows used for PCA fitting. PCA on the full unrolled spatial-feature grid (e.g. 16M+
            rows x D=1024) materializes a multi-GB float32 tensor on the GPU when ``features.dtype`` is bfloat16; the
            float32 copy alone OOMs on a single H100 after model weights are resident. We subsample BEFORE the
            bf16->fp32 conversion using a deterministic CPU Generator so visualization PNGs are reproducible across
            runs. Returned reduction matrix is identical in shape (D, 3) and applies unchanged to the full feature
            tensor.

    Returns:
        tuple: (reduction_matrix, color_min, color_max) for PCA transformation
    """
    pca_total_start = time.time()
    logger.info("    Starting robust PCA computation on features shape: %s", features.shape)

    assert len(features.shape) == 2, f"Features should be (N, C), got {features.shape}"

    # Subsample BEFORE the bf16->fp32 conversion. Doing it after would double the peak GPU memory footprint of
    # `features` and is exactly what caused the original OOM (a 2.06 GiB float32 copy on top of the resident bf16
    # tensor).
    if features.shape[0] > max_samples:
        sampling_start = time.time()
        # CPU generator with a fixed seed so re-running visualization yields the same selected rows -> same PNG bytes
        # for diff-based regression.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        perm = torch.randperm(features.shape[0], generator=gen)[:max_samples]
        features = features[perm.to(features.device)]
        sampling_time = time.time() - sampling_start
        logger.info(
            "    Subsampled features to %s rows (cap=max_samples=%s) in %.3fs to keep PCA off the OOM path",
            f"{features.shape[0]:,}",
            f"{max_samples:,}",
            sampling_time,
        )

    # Step 1: Convert to float32 for PCA computation (BFloat16 not supported)
    dtype_conversion_start = time.time()
    original_dtype = features.dtype
    if features.dtype == torch.bfloat16:
        logger.info("    Converting features from %s to float32 for PCA", original_dtype)
        features = features.float()
    dtype_conversion_time = time.time() - dtype_conversion_start
    logger.info("    Dtype conversion: %.3fs", dtype_conversion_time)

    # Step 2: Compute initial PCA using SVD
    initial_pca_start = time.time()
    logger.info("    Computing initial PCA with q=%s components...", 3 + skip)
    try:
        reduction_mat = torch.pca_lowrank(features, q=3 + skip, niter=20)[2]
        reduction_mat = reduction_mat[:, skip:]  # Skip initial components if requested
        logger.info("    PCA reduction matrix shape: %s", reduction_mat.shape)
    except RuntimeError as e:
        logger.warning("    PCA failed (likely ill-conditioned matrix): %s", e)
        logger.warning("    Falling back to simple feature selection for visualization")
        # Fallback: use first 3 feature dimensions (or pad if fewer)
        num_features = features.shape[1]
        if num_features >= 3 + skip:
            reduction_mat = torch.eye(num_features, 3, device=features.device, dtype=features.dtype)
            if skip > 0:
                reduction_mat = torch.cat(
                    [torch.zeros(num_features, skip, device=features.device, dtype=features.dtype), reduction_mat],
                    dim=1,
                )[:, skip:]
        else:
            # Not enough features, pad with zeros
            reduction_mat = torch.eye(num_features, device=features.device, dtype=features.dtype)
            if num_features < 3:
                padding = torch.zeros(num_features, 3 - num_features, device=features.device, dtype=features.dtype)
                reduction_mat = torch.cat([reduction_mat, padding], dim=1)
    initial_pca_time = time.time() - initial_pca_start
    logger.info("    Initial PCA computation: %.3fs", initial_pca_time)

    # Step 3: Project features onto principal components
    projection_start = time.time()
    logger.info("    Projecting features onto %s principal components...", reduction_mat.shape[1])
    colors = features @ reduction_mat
    logger.info("    Projected colors shape: %s", colors.shape)
    projection_time = time.time() - projection_start
    logger.info("    Feature projection: %.3fs", projection_time)

    # Step 4: Handle first component removal if requested
    if remove_first_component:
        logger.info("    Removing first component and recomputing PCA...")
        first_comp_start = time.time()

        # Normalize colors to find foreground mask
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2  # Foreground mask based on first component

        # Recompute PCA on foreground features only
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat

        first_comp_time = time.time() - first_comp_start
        logger.info("    First component removal: %.3fs", first_comp_time)
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()

    # Step 5: Compute robust statistics to handle outliers
    robust_stats_start = time.time()
    logger.info("    Computing robust statistics with m=%s...", m)

    # Compute median and median absolute deviation for robust statistics
    median_colors = torch.median(colors[fg_mask], dim=0).values
    d = torch.abs(colors[fg_mask] - median_colors)
    mdev = torch.median(d, dim=0).values
    s = d / mdev  # Scaled deviations

    logger.info("    Median colors: %s", median_colors)
    logger.info("    Median absolute deviations: %s", mdev)

    robust_stats_time = time.time() - robust_stats_start
    logger.info("    Robust statistics computation: %.3fs", robust_stats_time)

    # Step 6: Compute color ranges using robust statistics
    color_range_start = time.time()
    try:
        # Find inliers for each color channel based on robust statistics
        rins = colors[fg_mask][s[:, 0] < m, 0]  # Red channel inliers
        gins = colors[fg_mask][s[:, 1] < m, 1]  # Green channel inliers
        bins = colors[fg_mask][s[:, 2] < m, 2]  # Blue channel inliers

        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()], device=features.device)
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()], device=features.device)

        logger.info("    Robust color ranges - Min: %s, Max: %s", rgb_min, rgb_max)
        logger.info("    Inlier counts - R: %s, G: %s, B: %s", len(rins), len(gins), len(bins))

    except Exception as e:  # noqa: BLE001 - robust-stats math can fail many ways; fall back to all colors
        logger.warning("    Failed to compute robust statistics, using all colors. Error: %s", e)
        # Fallback to using all colors if robust statistics fail
        rgb_min = colors.min(dim=0).values
        rgb_max = colors.max(dim=0).values
        logger.info("    Fallback color ranges - Min: %s, Max: %s", rgb_min, rgb_max)

    color_range_time = time.time() - color_range_start
    logger.info("    Color range computation: %.3fs", color_range_time)

    # Step 7: Convert results back to original dtype if needed
    final_conversion_start = time.time()
    rgb_min = rgb_min.to(reduction_mat.device)
    rgb_max = rgb_max.to(reduction_mat.device)

    if original_dtype == torch.bfloat16:
        logger.info("    Converting results back to %s", original_dtype)
        reduction_mat = reduction_mat.to(original_dtype)
        rgb_min = rgb_min.to(original_dtype)
        rgb_max = rgb_max.to(original_dtype)

    final_conversion_time = time.time() - final_conversion_start
    logger.info("    Final conversion: %.3fs", final_conversion_time)

    pca_total_time = time.time() - pca_total_start
    logger.info("    Total robust PCA computation: %.3fs", pca_total_time)

    return reduction_mat, rgb_min, rgb_max


def _apply_pca_chunked(
    feats: torch.Tensor,
    reduct_mat: torch.Tensor,
    color_min: torch.Tensor | None = None,
    color_max: torch.Tensor | None = None,
    chunk_size: int = 4,
) -> torch.Tensor:
    """Project ``(B, D, H, W)`` features through a PCA reduction matrix in chunks.

    Returns a CPU tensor of shape ``(B, H, W, 3)`` with values clamped to ``[0, 1]`` (after normalization with
    ``color_min``/``color_max``).

    Rationale: the naive ``einops.rearrange("b d h w -> (b h w) d")`` allocates a full contiguous ``(B*H*W, D)`` copy
    because of the permutation. For ``(100, 768, 180, 320)`` bf16 inputs that copy alone is ~8.85 GiB, which OOMs the
    H100 once accumulated features are resident. Chunking along ``B`` bounds the peak working set to ``(chunk * H * W,
    D)`` plus a tiny ``(chunk * H * W, 3)`` PCA output, and streaming the result back to CPU after every chunk lets the
    allocator reuse memory across iterations.

    When ``color_min``/``color_max`` are both supplied, normalization is applied inline. Otherwise the function does two
    passes over ``feats``: one to compute streaming per-channel min/max, one to project and normalize.

    Args:
        feats: ``(B, D, H, W)`` tensor on a CUDA device.
        reduct_mat: ``(D, 3)`` PCA matrix on the same device as ``feats``.
        color_min, color_max: Optional ``(3,)`` per-channel normalization bounds. Both must be supplied for the one-pass
            fast path.
        chunk_size: Number of frames to project per chunk. ``4`` is the sweet spot for ``D=768, H=180, W=320`` features
            (~350 MiB intermediate copy per chunk).

    Returns:
        CPU ``torch.Tensor`` of shape ``(B, H, W, 3)`` and dtype ``float32``. Float32 (not the input dtype) is
        intentional: downstream consumers do ``F.interpolate(..., mode='bilinear')`` and ``.float().numpy()``, and bf16
        bilinear interpolation is not universally supported on CPU.
    """
    if feats.dim() != 4:
        raise ValueError(f"_apply_pca_chunked expects (B, D, H, W); got {tuple(feats.shape)}")

    B, D, H, W = feats.shape

    # Two-pass path: compute streaming global min/max on the projected output without ever materializing the full
    # (B*H*W, D) intermediate.
    if color_min is None or color_max is None:
        gmin: torch.Tensor | None = None
        gmax: torch.Tensor | None = None
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk_flat = feats[start:end].permute(0, 2, 3, 1).contiguous().view(-1, D)
            proj = chunk_flat @ reduct_mat
            cmin = proj.min(dim=0).values
            cmax = proj.max(dim=0).values
            gmin = cmin if gmin is None else torch.minimum(gmin, cmin)
            gmax = cmax if gmax is None else torch.maximum(gmax, cmax)
            del chunk_flat, proj
        color_min, color_max = gmin, gmax

    assert color_min is not None and color_max is not None
    span = (color_max - color_min) + 1e-8

    # Stage output buffer on CPU as fp32; chunks transfer immediately so each iteration reuses the same GPU allocator
    # blocks rather than accumulating.
    out = torch.empty((B, H, W, 3), dtype=torch.float32)
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk_flat = feats[start:end].permute(0, 2, 3, 1).contiguous().view(-1, D)
        proj = chunk_flat @ reduct_mat
        proj = ((proj - color_min) / span).clamp(0, 1)
        out[start:end] = proj.view(end - start, H, W, 3).detach().float().cpu()
        del chunk_flat, proj
    return out


def compute_spatial_cosine_similarity_heatmaps(
    output_batch: dict[str, Any],
    indices_to_use: torch.Tensor | None,
    camera_frame_indices: list[int] | None,
    world_frame_indices: list[int] | None,
    target_size: tuple[int, int],
) -> np.ndarray | None:
    """
    Compute spatial cosine similarity heatmaps between camera and world spatial features.

    Args:
        output_batch: Dictionary containing camera_spatial_features and world_spatial_features
        indices_to_use: Frame indices to use (if None, uses all frames)
        camera_frame_indices: List of camera frame indices to select
        world_frame_indices: List of world frame indices to select
        target_size: Target size (H, W) for resizing the heatmaps

    Returns:
        Spatial cosine similarity RGB heatmaps of shape (N, H, W, 3) or None if computation fails
    """
    cosine_sim_start = time.time()

    # Check if this exact computation was already done and cached
    cache_key = f"_spatial_cosine_sim_rgb_cache_{target_size[0]}x{target_size[1]}"
    if cache_key in output_batch:
        cached_result = output_batch[cache_key]
        # Verify cache is valid for current selection
        expected_num_frames = (
            len(camera_frame_indices)
            if camera_frame_indices is not None
            else output_batch.get("camera_spatial_features", torch.empty(0)).shape[0]
        )
        if cached_result is not None and len(cached_result) == expected_num_frames:
            logger.info("  - Using cached spatial cosine similarity RGB heatmaps")
            return cached_result

    # Get spatial features from output_batch (now in (B, C, H, W) format after rearrangement)
    camera_spatial_orig = output_batch.get("camera_spatial_features", None)
    world_spatial_orig = output_batch.get("world_spatial_features", None)

    if camera_spatial_orig is None or world_spatial_orig is None:
        logger.warning(
            "  - camera_spatial_features or world_spatial_features not found in output_batch, "
            "skipping cosine similarity"
        )
        return None

    # Check if spatial features have correct dimensions (B, C, H, W)
    if len(camera_spatial_orig.shape) != 4:
        logger.warning(
            "  - camera_spatial_features has unexpected shape %s, expected 4D (B, C, H, W)", camera_spatial_orig.shape
        )
        return None

    # Select the same frames we're visualizing
    if indices_to_use is not None and camera_frame_indices is not None and world_frame_indices is not None:
        camera_spatial_for_sim = camera_spatial_orig[camera_frame_indices]
        world_spatial_for_sim = world_spatial_orig[world_frame_indices]
    else:
        camera_spatial_for_sim = camera_spatial_orig
        world_spatial_for_sim = world_spatial_orig

    # Check for zero/invalid features
    world_norms = torch.norm(world_spatial_for_sim, p=2, dim=1)  # (N, H, W)
    zero_world = (world_norms == 0).sum().item()
    total_pixels = camera_spatial_for_sim.shape[0] * camera_spatial_for_sim.shape[2] * camera_spatial_for_sim.shape[3]

    # If all world features are zero, skip cosine similarity computation
    if zero_world == total_pixels:
        logger.warning("  - All world features are zero! Skipping spatial cosine similarity computation.")
        return None

    # Compute cosine similarity using utility function
    # Use keepdim=False to get (N, H, W) directly
    spatial_cosine_sim = feature_similarity_utils.compute_cosine_similarity(
        camera_spatial_for_sim, world_spatial_for_sim, dim=1, keepdim=False
    )  # (N, H, W)

    # Convert to numpy and handle BFloat16
    spatial_cosine_sim_np = spatial_cosine_sim.detach().cpu().float().numpy()  # (N, H, W)

    # Resize to match the target size
    resized_maps: list[np.ndarray] = []
    target_h, target_w = target_size
    for i in range(spatial_cosine_sim_np.shape[0]):
        sim_map = spatial_cosine_sim_np[i]  # (H, W)

        # Resize using cv2
        sim_map_resized = cv2.resize(sim_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        resized_maps.append(sim_map_resized)

    spatial_cosine_sim_resized = np.stack(resized_maps, axis=0)  # (N, H, W)

    # Convert to RGB heatmap using colormap
    # Use 'PRGn' (Purple-Green) colormap to match global similarity visualization
    # Map [-1, 1] to [0, 1] for colormap (PRGn expects values in [0, 1])
    spatial_cosine_sim_normalized = (spatial_cosine_sim_resized + 1) / 2  # [-1, 1] -> [0, 1]

    # Clip to ensure values are in valid range
    spatial_cosine_sim_normalized = np.clip(spatial_cosine_sim_normalized, 0, 1)

    # Apply PRGn colormap (same as global similarity)
    cmap = colormaps["PRGn"]
    rgb_frames: list[np.ndarray] = []
    for i in range(spatial_cosine_sim_normalized.shape[0]):
        colored = cmap(spatial_cosine_sim_normalized[i])[:, :, :3]  # (H, W, 3) in [0, 1]
        colored_uint8 = (colored * 255).astype(np.uint8)
        rgb_frames.append(colored_uint8)

    spatial_cosine_sim_rgb = np.stack(rgb_frames, axis=0)  # (N, H, W, 3)

    cosine_sim_time = time.time() - cosine_sim_start
    logger.info("  - Spatial cosine similarity computation: %.3fs", cosine_sim_time)

    # Cache the result for reuse (with underscore prefix to indicate internal use)
    cache_key = f"_spatial_cosine_sim_rgb_cache_{target_size[0]}x{target_size[1]}"
    output_batch[cache_key] = spatial_cosine_sim_rgb

    # Also cache the raw similarity values for standalone visualization
    output_batch["_spatial_cosine_sim_cache"] = spatial_cosine_sim

    return spatial_cosine_sim_rgb


def _label_frames(frames: np.ndarray, label: str) -> np.ndarray:
    """Draw a small top-right text label on every frame of a ``(N, H, W, 3)`` float ``[0, 1]`` array.

    Returns a new float ``[0, 1]`` array (input is not mutated). Mirrors the labeling style used by the patch
    visualizations so the camera-only tiled video panels are self-describing.
    """
    frames_uint8 = (np.clip(frames, 0.0, 1.0) * 255).astype(np.uint8)
    for i in range(frames_uint8.shape[0]):
        image_video_utils.draw_topright_lines(frames_uint8[i], [label], font_scale=0.7, y_start=20, y_spacing=20)
    return frames_uint8.astype(np.float32) / 255.0


def _maybe_backbone_layer(feature_list: list[torch.Tensor]) -> torch.Tensor | None:
    """Return the detached frozen-backbone (pre-alignment) feature map, or ``None`` when unavailable.

    The alignment-final features are always the last list entry. A distinct pre-alignment layer exists only when the
    encoder fetched it separately; when ``layer_idx == [-1]`` the same tensor object is returned twice, so the identity
    (``is``) check disambiguates a genuine two-layer fetch from a single-layer one.
    """
    if len(feature_list) >= 2 and feature_list[0] is not feature_list[-1]:
        return feature_list[0].detach()
    return None


def _fit_joint_pca(
    feature_tensors: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Fit a single robust PCA basis jointly over several ``(B, D, H, W)`` feature maps.

    Concatenates the maps along the batch dim, flattens to ``(N, D)``, and fits :func:`_get_robust_pca_with_timing`.
    Fitting jointly means every projected panel (camera/world, backbone/aligned) shares one color basis and is directly
    comparable across rows.

    Args:
        feature_tensors: Feature maps to pool for the fit; each ``(B, D, H, W)`` on the same device.

    Returns:
        ``(reduction_matrix, color_min, color_max)``, or ``None`` when the pool is empty (no rows to fit).
    """
    if not feature_tensors:
        return None
    pooled = torch.cat(feature_tensors, dim=0)
    if pooled.shape[0] == 0:
        return None
    flat = einops.rearrange(pooled, "b d h w -> (b h w) d")
    return _get_robust_pca_with_timing(flat, skip=0)


def _project_optional_feature(
    feats: torch.Tensor,
    reduct_mat: torch.Tensor,
    target_hw: tuple[int, int],
    output_batch: dict[str, Any],
    output_batch_key: str,
) -> torch.Tensor:
    """Project one optional ``(B, D, H, W)`` feature map to RGB and resize it to the camera/world spatial grid.

    Shared by the optional patch- and type-camera feature maps in :func:`create_tiled_video`. Projects through the
    joint ``reduct_mat`` when the channel dim matches, otherwise fits a fresh robust PCA on a small frame subset
    (self-normalized). Each projection runs through :func:`_apply_pca_chunked` so the ``(B*H*W, D)`` intermediate never
    materializes (that copy alone is ~8.85 GiB for ``(100, 768, 180, 320)`` bf16 features and OOMs the H100). The heavy
    source tensor is freed immediately -- and popped from ``output_batch`` too, since ``.detach()`` on entry only shares
    storage so both refs must drop for the storage to free -- to give the next projection and downstream tiling
    headroom.

    Args:
        feats: ``(B, D, H, W)`` feature map (typically a ``.detach()`` of an ``output_batch`` entry).
        reduct_mat: Joint ``(D, 3)`` PCA matrix to reuse when channel dims match.
        target_hw: ``(H, W)`` of the camera/world spatial RGB grid to bilinear-resize the projection to.
        output_batch: Model output dict; ``output_batch_key`` is popped from it after projection.
        output_batch_key: Key in ``output_batch`` holding the source tensor (e.g. ``"camera_patch_features"``).

    Returns:
        CPU float32 ``(B, target_h, target_w, 3)`` tensor.
    """
    feats = feats.to(reduct_mat.device)
    if feats.shape[1] != reduct_mat.shape[0]:
        # Dimension mismatch -> fit a fresh PCA on a sub-batch. Sampling frames BEFORE rearrange keeps the fit input
        # small.
        n_fit_frames = min(8, feats.shape[0])
        idx = torch.randperm(feats.shape[0], device=feats.device)[:n_fit_frames]
        fit_subset = einops.rearrange(feats[idx], "b d h w -> (b h w) d")
        fit_mat, _, _ = _get_robust_pca_with_timing(fit_subset, skip=0)
        del fit_subset
        rgb = _apply_pca_chunked(feats, fit_mat)
    else:
        rgb = _apply_pca_chunked(feats, reduct_mat)

    del feats
    output_batch.pop(output_batch_key, None)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target_h, target_w = target_hw
    if rgb.shape[1] != target_h or rgb.shape[2] != target_w:
        # rgb: [B, H, W, 3] -> [B, 3, H, W] for interpolate -> [B, 3, target] -> [B, target, 3]
        rgb_nchw = rgb.permute(0, 3, 1, 2)
        rgb_nchw = F.interpolate(rgb_nchw, size=(target_h, target_w), mode="bilinear", align_corners=False)
        rgb = rgb_nchw.permute(0, 2, 3, 1)
        logger.info("  - Resized optional feature '%s' to (%s, %s)", output_batch_key, target_h, target_w)
    return rgb


def _extract_per_frame_meta(
    data_batch: dict[str, Any],
    key: str,
    indices_to_use: torch.Tensor | None,
    selection_column: int,
) -> list[Any] | None:
    """Pull a per-frame metadata list (session ids or frame numbers) from ``data_batch``, honoring frame selection.

    Used for the camera/world session-id and frame-index text overlays in :func:`create_tiled_video`, which extracted
    these four lists with near-identical blocks.

    Args:
        data_batch: Data batch that may contain ``key``.
        key: Metadata key, e.g. ``"camera_session_id"`` or ``"world_frame_idx"``.
        indices_to_use: ``(K, 2)`` selection tensor, or ``None`` to return every entry.
        selection_column: Column of ``indices_to_use`` to select on (camera overlays use ``1``, world overlays ``0``).

    Returns:
        A list of scalars (``.item()`` applied to tensor scalars) for the selected frames, or ``None`` when ``key`` is
        absent from ``data_batch``.
    """
    if key not in data_batch:
        return None
    values = data_batch[key]
    selection = indices_to_use[:, selection_column].cpu().tolist() if indices_to_use is not None else range(len(values))
    return [values[i] if isinstance(values, list) else values[i].item() for i in selection]


def create_camera_only_tiled_video(
    output_batch: dict[str, torch.Tensor],
    data_batch: dict[str, torch.Tensor],
    iteration: int,
    output_dir: str,
    camera_start_frame: int = 0,
    use_gpu_encoding: bool = True,
) -> None:
    """Camera-only tiled video: ``[camera | backbone feature | aligned feature]`` as a single 1xN row.

    :func:`create_tiled_video` is built around camera+world pairs (joint camera/world PCA, 2x2/3x2/3x3 layouts) and
    requires ``world_spatial_features_list``. In camera-only mode there is no world stream, so this renders just the
    camera column: the input frame plus PCA-colorized frozen-backbone (pre-alignment) and alignment-final
    (post-alignment) encoder features. The backbone panel is omitted when the encoder returned a single layer
    (``aligned is backbone``), yielding a 1x2 ``[camera | aligned feature]`` video instead.

    Args:
        output_batch: Model output. Must contain ``camera_spatial_features_list`` (list of ``(B, D, H, W)`` tensors,
            backbone-first / alignment-final-last).
        data_batch: Data batch containing ``camera_image``.
        iteration: Iteration number (used in the output filename).
        output_dir: Output directory.
        camera_start_frame: Frame offset for camera frame numbering in the text overlay.
        use_gpu_encoding: Use NVIDIA NVENC if available.
    """
    if "camera_image" not in data_batch:
        logger.warning("Camera images not available for camera-only tiled video creation")
        return
    camera_features = output_batch.get("camera_spatial_features_list")
    if camera_features is None:
        logger.warning("camera_spatial_features_list not available for camera-only tiled video creation")
        return
    camera_spatial_features_list = cast(list[torch.Tensor], camera_features)

    result = basic_vis_utils.extract_frames_from_data_batch(data_batch)
    if result is None or result[0] is None:
        logger.error("Failed to extract frames from data batch")
        return
    camera_images, _world_images, camera_fps = result
    assert camera_images is not None
    if camera_images.shape[0] == 0:
        logger.error("No camera frames to visualize for camera-only tiled video")
        return

    # Alignment-final (post-alignment) features are always the last layer; the frozen-backbone (pre-alignment) layer is
    # only distinct when the encoder fetched it separately (see :func:`_maybe_backbone_layer`).
    camera_aligned_full = camera_spatial_features_list[-1].detach()
    camera_backbone_full = _maybe_backbone_layer(camera_spatial_features_list)

    # Joint PCA fit over the available camera feature maps so the backbone and aligned panels share a color basis and
    # are directly comparable. Subsample frames BEFORE concatenating: torch.cat + rearrange on the full clip
    # materializes a second copy of every feature map (~8.85 GiB for 100 bf16 frames at 768×180×320), which
    # re-introduces OOM risk on long videos. 8 evenly-spaced frames are sufficient to capture the color basis.
    _max_pca_frames = 8
    pca_sample = [t[:_max_pca_frames] if t.shape[0] > _max_pca_frames else t for t in [camera_aligned_full]]
    if camera_backbone_full is not None:
        bb = camera_backbone_full
        pca_sample.append(bb[:_max_pca_frames] if bb.shape[0] > _max_pca_frames else bb)
    pca = _fit_joint_pca(pca_sample)
    if pca is None:
        logger.warning("No camera spatial features available for camera-only tiled video")
        return
    reduct_mat, color_min, color_max = pca

    aligned_rgb = _apply_pca_chunked(camera_aligned_full, reduct_mat, color_min, color_max).float().numpy()
    backbone_rgb = (
        _apply_pca_chunked(camera_backbone_full, reduct_mat, color_min, color_max).float().numpy()
        if camera_backbone_full is not None
        else None
    )

    # Frame-number text overlay on the camera panel (gracefully degrades to plain frame numbers when no
    # similarity_matrix is present, which is always the case in camera-only mode).
    try:
        camera_panel = add_text_overlays(camera_images, output_batch, camera_start_frame, "camera")
    except Exception:
        logger.exception("Failed to add camera text overlays; using raw frames")
        camera_panel = camera_images

    tile_h, tile_w = camera_images.shape[1], camera_images.shape[2]

    def _resize_to_frame(rgb: np.ndarray) -> np.ndarray:
        """Upscale a ``(N, H', W', 3)`` PCA feature panel to the camera frame size ``(N, tile_h, tile_w, 3)``.

        Labels are drawn *after* this so the fixed-size text isn't clipped: the raw feature grid is tiny (e.g. ~80x45
        at image_size 720x1280 with patch size 16), so a label drawn on it would run off the left edge (the "eature"
        bug). Nearest-neighbor keeps the PCA color blocks crisp and matches the resize the tiler used previously.
        """
        out = np.empty((rgb.shape[0], tile_h, tile_w, 3), dtype=np.float32)
        for i in range(rgb.shape[0]):
            out[i] = cv2.resize(rgb[i], (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
        return out

    aligned_panel = _label_frames(_resize_to_frame(aligned_rgb), "After-Alignment Feature")
    backbone_panel = (
        _label_frames(_resize_to_frame(backbone_rgb), "Backbone Feature") if backbone_rgb is not None else None
    )

    num_frames = min(camera_panel.shape[0], aligned_rgb.shape[0])
    camera_u8 = image_video_utils.to_uint8(camera_panel)
    aligned_u8 = image_video_utils.to_uint8(aligned_panel)
    if backbone_panel is not None:
        # 2x2 with the camera as a per-row reference (16:9-friendly on screen):
        #   Row 1: camera | backbone feature
        #   Row 2: camera | after-alignment feature
        backbone_u8 = image_video_utils.to_uint8(backbone_panel)
        tiles = [camera_u8, backbone_u8, camera_u8, aligned_u8]
        rows, cols = 2, 2
        layout_desc = "2x2 (camera|backbone / camera|after-alignment)"
    else:
        # Only a single encoder layer available -> camera | after-alignment.
        tiles = [camera_u8, aligned_u8]
        rows, cols = 1, 2
        layout_desc = "1x2 (camera | after-alignment)"

    fps = camera_fps if camera_fps is not None else 30.0
    video_path = os.path.join(output_dir, f"tiled_video_iter_{iteration:06d}.mp4")

    logger.info("Creating camera-only tiled video %s -> %s", layout_desc, video_path)
    image_video_utils.write_tiled_video(
        tiles, rows, cols, num_frames, tile_h, tile_w, video_path, fps, use_gpu=use_gpu_encoding
    )
    logger.info("Created camera-only tiled video: %s (%s, %s frames)", video_path, layout_desc, num_frames)


def create_tiled_video(
    output_batch: dict[str, torch.Tensor],
    data_batch: dict[str, torch.Tensor],
    iteration: int,
    output_dir: str,
    camera_start_frame: int = 0,
    world_start_frame: int = 0,
    output_mode: basic_vis_utils.TilingOutputMode = basic_vis_utils.TilingOutputMode.VIDEO,
    num_frames: int | None = None,  # Number of frames to process (None = all frames)
    frame_indices: list[Any] | None = None,  # Specific frame indices (None = evenly spaced)
    use_gpu_encoding: bool = True,  # Use NVIDIA NVENC if available
) -> np.ndarray | None:
    """Create 2x2 or 3x2 tiled video (.mp4) or frame visualization.

    Uses 2x2 layout by default (camera, world images + spatial features).
    If camera_patch_features is available in output_batch, uses 3x2 layout with
    camera_patch_features in row 3, position 0.

    Args:
        output_batch: Model output containing camera and world features
          - output_batch["camera_spatial_features_list"]: List of camera features
          - output_batch["world_spatial_features_list"]: List of world features
          - output_batch["camera_patch_features"]: Optional patch camera features
        data_batch: Data batch containing camera and world images
          - data_batch["camera_image"]: Camera images
          - data_batch["world_image"]: World images
        iteration: Iteration number
        output_dir: Output directory
        camera_start_frame: Frame offset for camera frame numbering
        world_start_frame: Frame offset for world frame numbering
        output_mode: basic_vis_utils.TilingOutputMode.VIDEO to save as MP4,
            basic_vis_utils.TilingOutputMode.IMAGE to return tiled image array
        num_frames: Number of frames to process (None = all frames for video, required for frames mode)
        frame_indices: Specific frame indices (None = evenly spaced)

    Returns:
        If output_mode=basic_vis_utils.TilingOutputMode.VIDEO: None
        If output_mode=basic_vis_utils.TilingOutputMode.IMAGE: np.ndarray of tiled frames
    """

    tiled_video_start_time = time.time()

    has_camera = "camera_image" in data_batch
    has_world = "world_image" in data_batch

    if not has_camera or not has_world:
        logger.warning("Camera or world images not available for tiled video creation")
        return None

    # Check if we have spatial features
    if "camera_spatial_features_list" not in output_batch or "world_spatial_features_list" not in output_batch:
        logger.warning("Spatial features not available for tiled video creation")
        return None

    # Data preparation
    data_prep_start_time = time.time()
    logger.info("  - Preparing data for tiled video...")

    # Get images and spatial features (callers apply any frame offsets before this step)
    result = basic_vis_utils.extract_frames_from_data_batch(data_batch)
    if result is None or result[0] is None:
        logger.error("Failed to extract frames from data batch")
        return None
    camera_images, world_images, camera_fps = result
    assert camera_images is not None and world_images is not None
    if camera_images.shape[0] == 0 or world_images.shape[0] == 0:
        logger.error(
            "No frames to visualize: camera_images.shape[0]=%s, world_images.shape[0]=%s. "
            "Check data_batch and that selected_indices/vis_balanced_indices are in range.",
            camera_images.shape[0],
            world_images.shape[0],
        )
        return None
    camera_spatial_features_list = cast(list[torch.Tensor], output_batch["camera_spatial_features_list"])
    world_spatial_features_list = cast(list[torch.Tensor], output_batch["world_spatial_features_list"])

    data_prep_time = time.time() - data_prep_start_time
    logger.info("  - Data preparation: %.3fs", data_prep_time)

    # Last-layer features (alignment-final, post-alignment encoder output).
    # KEEP ON GPU for faster PCA computation.
    camera_spatial_features_full = camera_spatial_features_list[-1].detach()
    world_spatial_features_full = world_spatial_features_list[-1].detach()

    # Frozen-backbone features (pre-alignment, layer just before alignment block) — only when the model fetched them as
    # a distinct intermediate layer (see :func:`_maybe_backbone_layer`).
    camera_backbone_features_full = _maybe_backbone_layer(camera_spatial_features_list)
    world_backbone_features_full = _maybe_backbone_layer(world_spatial_features_list)
    if camera_backbone_features_full is not None:
        logger.info("  - Found camera backbone features: %s", tuple(camera_backbone_features_full.shape))
    if world_backbone_features_full is not None:
        logger.info("  - Found world backbone features: %s", tuple(world_backbone_features_full.shape))

    # Check for camera_patch_features (optional, for 3x2 or 3x3 layout)
    camera_patch_features = output_batch.get("camera_patch_features", None)
    if camera_patch_features is not None:
        camera_patch_features = camera_patch_features.detach()  # Keep on GPU
        logger.info("  - Found camera_patch_features: %s", camera_patch_features.shape)

    # Check for camera_type_features (optional, separate necks only)
    camera_type_features = output_batch.get("camera_type_features", None)
    if camera_type_features is not None:
        camera_type_features = camera_type_features.detach()  # Keep on GPU
        logger.info("  - Found camera_type_features: %s", camera_type_features.shape)

    # Check if we have pre-selected balanced pairs from visualization callback
    # These take priority over regular selected_indices for consistent visualization
    vis_balanced_indices = output_batch.get("vis_balanced_indices", None)
    vis_balanced_pair_indices = output_batch.get("vis_balanced_pair_indices", None)
    selected_indices = output_batch.get("selected_indices", None)

    # Use balanced indices if available (from visualization callback), otherwise fall back to selected_indices
    # If we have patch features, we need BOTH vis_balanced_indices and vis_balanced_pair_indices for correct alignment
    if camera_patch_features is not None:
        # Need both indices for patch feature alignment
        indices_to_use = (
            vis_balanced_indices
            if (vis_balanced_indices is not None and vis_balanced_pair_indices is not None)
            else selected_indices
        )
        if vis_balanced_indices is not None and vis_balanced_pair_indices is None:
            logger.warning(
                "  - vis_balanced_indices available but vis_balanced_pair_indices is None. "
                "Falling back to selected_indices for patch feature alignment."
            )
    else:
        # No patch features, only need vis_balanced_indices for image/spatial feature alignment
        indices_to_use = vis_balanced_indices if vis_balanced_indices is not None else selected_indices

    # Track indices into similarity_matrix for text overlays
    similarity_indices = None

    if indices_to_use is not None:
        # indices_to_use is (K, 2): batch row indices or explicit pair-row identities for overlays.
        world_frame_indices: list[Any] | None = indices_to_use[:, 0].cpu().tolist()
        camera_frame_indices: list[Any] | None = indices_to_use[:, 1].cpu().tolist()

        assert camera_frame_indices is not None and world_frame_indices is not None
        if vis_balanced_indices is not None:
            logger.info("  - Using vis_balanced_indices (balanced pos/neg pairs): %s pairs", len(camera_frame_indices))
        else:
            logger.info("  - Using selected_indices to align features: %s sampled pairs", len(camera_frame_indices))

        # Pair-row ids into output_batch tensors (spatial / patch / pair RGB), length K.
        pair_row_indices: list[int] | None
        if vis_balanced_pair_indices is not None:
            pair_row_indices = vis_balanced_pair_indices.cpu().tolist()
        elif vis_balanced_indices is not None:
            vi = vis_balanced_indices
            pair_row_indices = vi[:, 0].cpu().tolist() if isinstance(vi, torch.Tensor) else [int(x[0]) for x in vi]
        elif selected_indices is not None:
            pair_row_indices = list(range(len(world_frame_indices)))
        else:
            pair_row_indices = None

        # RGB tiles: use pair-aligned tensors from the model when present (same shard as spatial features).
        pair_cam = output_batch.get("camera_other_frames")
        pair_wref = output_batch.get("world_ref_frames")
        use_pair_rgb = (
            pair_row_indices is not None
            and pair_cam is not None
            and pair_wref is not None
            and isinstance(pair_cam, torch.Tensor)
            and isinstance(pair_wref, torch.Tensor)
        )
        if use_pair_rgb:
            assert pair_cam is not None and pair_wref is not None
            pr = torch.tensor(pair_row_indices, dtype=torch.long, device=pair_cam.device)
            camera_images = _pair_rgb_bchw_to_numpy_hwc_batch(pair_cam[pr])
            world_images = _pair_rgb_bchw_to_numpy_hwc_batch(pair_wref[pr])
            logger.info("  - RGB tiles from output_batch camera_other_frames / world_ref_frames (pair rows)")
        else:
            # Flattened batch layout: index by dim-0 frame rows.
            camera_images = camera_images[camera_frame_indices]
            world_images = world_images[world_frame_indices]

        # Select feature/logit rows by pair-row ids (not raw frame ids). output_batch spatial/patch tensors are ordered
        # by selected pair rows.
        if pair_row_indices is not None:
            camera_spatial_features = camera_spatial_features_full[pair_row_indices]
            world_spatial_features = world_spatial_features_full[pair_row_indices]
            camera_backbone_features = (
                camera_backbone_features_full[pair_row_indices] if camera_backbone_features_full is not None else None
            )
            world_backbone_features = (
                world_backbone_features_full[pair_row_indices] if world_backbone_features_full is not None else None
            )
        else:
            # Conservative fallback: do not assume frame ids map to pair rows.
            camera_spatial_features = camera_spatial_features_full
            world_spatial_features = world_spatial_features_full
            camera_backbone_features = camera_backbone_features_full
            world_backbone_features = world_backbone_features_full

        # Track similarity_indices for text overlays (two-stage selection process):
        # Stage 1: Select balanced pairs from full batch
        #   - Full batch has M pairs with similarity_matrix of shape (M,)
        #   - vis_balanced_pair_indices selects a subset (e.g., 8 pairs from 256)
        #   - similarity_indices = [12, 45, 78, ...] maps to entries in similarity_matrix
        # Stage 2 (in IMAGE mode below): Further subset for visualization
        #   - frame_indices selects which of the balanced pairs to display (e.g., 4 of 8)
        #   - similarity_indices is further subsetted to match displayed frames
        if pair_row_indices is not None:
            similarity_indices = pair_row_indices
            logger.info("  - Tracking similarity_indices from pair-row indices: %s indices", len(similarity_indices))

        # Select patch features using the correct indices
        # Patch features are computed only for selected pairs, so we need indices INTO selected_indices
        if camera_patch_features is not None:
            if pair_row_indices is not None:
                camera_patch_features = camera_patch_features[pair_row_indices]
                logger.info("  - Selected patch features using pair-row indices: %s pairs", len(pair_row_indices))
            else:
                logger.warning(
                    "  - Could not infer pair-row indices for patch features; using full patch feature tensor."
                )
        if camera_type_features is not None:
            if pair_row_indices is not None:
                camera_type_features = camera_type_features[pair_row_indices]
            else:
                logger.warning(
                    "  - Could not infer pair-row indices for type features; using full type feature tensor."
                )
    else:
        # No selection, use all frames
        camera_spatial_features = camera_spatial_features_full
        world_spatial_features = world_spatial_features_full
        camera_backbone_features = camera_backbone_features_full
        world_backbone_features = world_backbone_features_full

    # Determine frame selection based on output mode BEFORE processing spatial features
    total_frames = len(camera_images)
    if total_frames == 0:
        logger.error(
            "No frames selected for tiled video (camera_images has 0 frames after subset). "
            "Check vis_balanced_indices/selected_indices and that they are in range [0, %s).",
            len(camera_spatial_features_full),
        )
        return None

    if output_mode == basic_vis_utils.TilingOutputMode.IMAGE:
        if frame_indices is None:
            if num_frames is None or num_frames <= 0 or num_frames > total_frames:
                num_frames = total_frames
            # Select evenly spaced frames
            step = total_frames // num_frames
            frame_indices = [i * step for i in range(num_frames)]
        else:
            num_frames = len(frame_indices)

        camera_images = camera_images[frame_indices]
        world_images = world_images[frame_indices]
        camera_spatial_features = camera_spatial_features[frame_indices]
        world_spatial_features = world_spatial_features[frame_indices]
        if camera_backbone_features is not None:
            camera_backbone_features = camera_backbone_features[frame_indices]
        if world_backbone_features is not None:
            world_backbone_features = world_backbone_features[frame_indices]
        if camera_patch_features is not None:
            camera_patch_features = camera_patch_features[frame_indices]
        if camera_type_features is not None:
            camera_type_features = camera_type_features[frame_indices]
        # Stage 2 of similarity_indices selection (see Stage 1 comment above):
        # Further subset similarity_indices to match the frames selected for visualization
        # e.g., if frame_indices=[0,2,4,6] and similarity_indices=[12,45,78,102,156,189,210,245],
        # result is [12,78,156,210] - the 4 indices into similarity_matrix for the 4 displayed frames
        # Here frame_indices index into the *current* camera_images list (K or N rows), not global time.
        if similarity_indices is not None:
            similarity_indices = [similarity_indices[i] for i in frame_indices]
            logger.info("  - Subsetted similarity_indices for IMAGE mode: %s indices", len(similarity_indices))

        # No vis_balanced / selected_indices: similarity_matrix is full (N, N) or length N for N batch frames, but IMAGE
        # mode shows only len(frame_indices) tiles. Stage 2 above does not run (similarity_indices was None). Map each
        # displayed tile to its matrix row = global frame index in the pre-subset batch — the same frame_indices used
        # for image subsetting.
        if similarity_indices is None:
            sim_mat = output_batch.get("similarity_matrix")
            if sim_mat is not None and hasattr(sim_mat, "shape") and len(sim_mat.shape) >= 1:
                n_mat = int(sim_mat.shape[0])
                if n_mat > 0 and len(frame_indices) > 0 and n_mat != len(frame_indices):
                    similarity_indices = [int(i) for i in frame_indices]
                    logger.info(
                        "  - Inferred similarity_indices from frame_indices for tiled IMAGE mode "
                        "(matrix size %s, displayed %s): %s",
                        n_mat,
                        len(frame_indices),
                        similarity_indices,
                    )
    else:  # video mode
        num_frames = total_frames
        frame_indices = list(range(total_frames))

    # Apply PCA to spatial features using faster embedder PCA implementation
    # Use the last camera layer and last world layer (matching reference script approach)
    # In the reference script: camera_spatial_features_list[-1] and world_spatial_features_list[idx]

    pca_start_time = time.time()

    # Compute PCA on ALL frames (not just selected ones) to get consistent color space. Joint PCA fit includes
    # alignment-final features and (when present) frozen-backbone features for both streams, so colors are directly
    # comparable across rows 2 and 3 in the debug 3x3 layout.
    pca_pool = [camera_spatial_features_full, world_spatial_features_full]
    if camera_backbone_features_full is not None:
        pca_pool.append(camera_backbone_features_full)
    if world_backbone_features_full is not None:
        pca_pool.append(world_backbone_features_full)
    logger.info(
        "  - Computing PCA jointly on backbone features "
        "(streams included: %s = camera_alignment + world_alignment%s%s)",
        len(pca_pool),
        " + camera_backbone" if camera_backbone_features_full is not None else "",
        " + world_backbone" if world_backbone_features_full is not None else "",
    )

    pca = _fit_joint_pca(pca_pool)
    if pca is None:
        logger.warning("No spatial features available for tiled video generation")
        return None
    reduct_mat, color_min, color_max = pca

    # Apply PCA transformation to all three feature types separately. Each projection runs through `_apply_pca_chunked`
    # so the (B*H*W, D) intermediate copy never materializes; for (100, 768, 180, 320) bf16 features that copy alone
    # would be ~8.85 GiB and OOM the H100.
    pca_transform_start = time.time()
    logger.info("  - Applying PCA transformation to all features...")

    # All RGB results live on CPU from here on (downstream `torch.cat`/`F.interpolate` work fine on CPU, and the very
    # next step is `.cpu().float().numpy()` anyway).
    camera_rgb = _apply_pca_chunked(camera_spatial_features, reduct_mat, color_min, color_max)
    world_rgb = _apply_pca_chunked(world_spatial_features, reduct_mat, color_min, color_max)

    # Transform frozen-backbone features (pre-alignment) using the same joint PCA basis so backbone-row colors are
    # directly comparable to alignment-row colors in row 2 vs 3 of the debug 3x3 layout.
    camera_backbone_rgb = (
        _apply_pca_chunked(camera_backbone_features, reduct_mat, color_min, color_max)
        if camera_backbone_features is not None
        else None
    )
    world_backbone_rgb = (
        _apply_pca_chunked(world_backbone_features, reduct_mat, color_min, color_max)
        if world_backbone_features is not None
        else None
    )

    # Transform camera_patch_features if available (joint or self-fit PCA basis). ``_project_optional_feature`` runs the
    # chunked projection (never materializing the ~8.85 GiB (B*H*W, D) intermediate), frees the source tensor, and
    # resizes to the camera/world spatial grid. Track the "we have patch RGB" signal explicitly since the source is
    # dropped.
    target_hw = (camera_rgb.shape[1], camera_rgb.shape[2])
    have_patch_rgb = False
    if camera_patch_features is not None:
        patch_rgb = _project_optional_feature(
            camera_patch_features, reduct_mat, target_hw, output_batch, "camera_patch_features"
        )
        camera_patch_features = None
        have_patch_rgb = True
        # Concatenate camera, world, and patch (all CPU)
        spatial_features_pca = torch.cat([camera_rgb, world_rgb, patch_rgb], dim=0).float().numpy()
    else:
        # Concatenate only camera and world
        spatial_features_pca = torch.cat([camera_rgb, world_rgb], dim=0).float().numpy()

    # Transform camera_type_features if available (separate necks). Same projection/free/resize path as the patch
    # features above.
    type_rgb_numpy = None
    if camera_type_features is not None:
        type_rgb = _project_optional_feature(
            camera_type_features, reduct_mat, target_hw, output_batch, "camera_type_features"
        )
        camera_type_features = None
        type_rgb_numpy = type_rgb.float().numpy()

    pca_transform_time = time.time() - pca_transform_start
    logger.info("  - PCA transformation and reshaping: %.3fs", pca_transform_time)
    logger.info("  - Final PCA features shape: %s", spatial_features_pca.shape)

    pca_time = time.time() - pca_start_time
    logger.info("PCA computation time: %.3fs", pca_time)

    # Skip LCH conversion - our robust PCA already outputs RGB-like features in [0,1] range
    color_conversion_start = time.time()
    logger.info("  - Skipping LCH conversion (PCA features already in RGB format)")

    # PCA features are already in [0,1] range and represent RGB-like colors
    spatial_features_rgb = spatial_features_pca  # No conversion needed

    color_conversion_time = time.time() - color_conversion_start
    logger.info("  - Color space handling: %.3fs", color_conversion_time)

    # Split back into camera, world, and optionally mixed features
    spatial_frames = camera_spatial_features.shape[0]  # Should equal world_spatial_features.shape[0]
    camera_spatial_rgb = spatial_features_rgb[:spatial_frames]
    world_spatial_rgb = spatial_features_rgb[spatial_frames : 2 * spatial_frames]

    # Backbone features as numpy (uint8 conversion happens later in tile_and_save_media).
    # camera_backbone_rgb / world_backbone_rgb are already CPU tensors from _apply_pca_chunked.
    camera_backbone_numpy = camera_backbone_rgb.float().numpy() if camera_backbone_rgb is not None else None
    world_backbone_numpy = world_backbone_rgb.float().numpy() if world_backbone_rgb is not None else None

    patch_camera_rgb = None
    type_camera_rgb = type_rgb_numpy
    if have_patch_rgb:
        patch_camera_rgb = spatial_features_rgb[2 * spatial_frames : 3 * spatial_frames]
        logger.info(
            "Split spatial features: camera %s, world %s, patch %s",
            camera_spatial_rgb.shape,
            world_spatial_rgb.shape,
            patch_camera_rgb.shape,
        )
    else:
        logger.info("Split spatial features: camera %s, world %s", camera_spatial_rgb.shape, world_spatial_rgb.shape)

    # Compute spatial cosine similarity heatmap using original features from output_batch
    target_size = (camera_spatial_rgb.shape[1], camera_spatial_rgb.shape[2])  # (H, W)
    spatial_cosine_sim_rgb = compute_spatial_cosine_similarity_heatmaps(
        output_batch=output_batch,
        indices_to_use=indices_to_use,
        camera_frame_indices=camera_frame_indices if indices_to_use is not None else None,
        world_frame_indices=world_frame_indices if indices_to_use is not None else None,
        target_size=target_size,
    )

    # Frame selection has already been done before PCA processing

    # Use original output_batch for text overlays (no trimming needed)
    output_batch_for_text = output_batch

    # Extract session IDs and frame indices from data_batch if available (camera overlays select on column 1, world on
    # column 0 of indices_to_use).
    camera_session_ids = _extract_per_frame_meta(data_batch, "camera_session_id", indices_to_use, 1)
    camera_frame_indices = _extract_per_frame_meta(data_batch, "camera_frame_idx", indices_to_use, 1)
    world_session_ids = _extract_per_frame_meta(data_batch, "world_session_id", indices_to_use, 0)
    world_frame_indices = _extract_per_frame_meta(data_batch, "world_frame_idx", indices_to_use, 0)

    # Add frame numbers and similarity scores as text overlays
    try:
        camera_images_with_text = add_text_overlays(
            camera_images,
            output_batch_for_text,
            camera_start_frame,
            "camera",
            similarity_indices=similarity_indices,
            session_ids=camera_session_ids,
            frame_indices=camera_frame_indices,
        )
        world_images_with_text = add_text_overlays(
            world_images,
            output_batch_for_text,
            world_start_frame,
            "world",
            similarity_indices=similarity_indices,
            session_ids=world_session_ids,
            frame_indices=world_frame_indices,
        )
        logger.info("  - Successfully added text overlays")
    except Exception:
        logger.exception("Failed to add text overlays")
        camera_images_with_text = camera_images
        world_images_with_text = world_images

    if output_mode == basic_vis_utils.TilingOutputMode.VIDEO:
        # Video creation
        video_creation_start_time = time.time()
        logger.info("  - Creating and saving tiled video...")

        # Create output path
        video_path = os.path.join(output_dir, f"tiled_video_iter_{iteration:06d}.mp4")

        # Call the consolidated function directly
        basic_vis_utils.tile_and_save_media(
            camera_images=camera_images_with_text,  # (num_frames, H, W, 3)
            world_images=world_images_with_text,  # (num_frames, H, W, 3)
            camera_spatial_features=camera_spatial_rgb,  # (num_frames, H', W', 3) — alignment-final (row 3)
            world_spatial_features=world_spatial_rgb,  # (num_frames, H', W', 3)
            output_path=video_path,
            output_mode=basic_vis_utils.TilingOutputMode.VIDEO,
            fps=camera_fps if camera_fps is not None else 30.0,
            camera_patch_features=patch_camera_rgb,
            camera_type_features=type_camera_rgb,
            spatial_cosine_sim=spatial_cosine_sim_rgb,
            # Optional (num_frames, H', W', 3) — frozen pretrained backbone (row 2)
            camera_backbone_features=camera_backbone_numpy,
            world_backbone_features=world_backbone_numpy,
            use_gpu_encoding=use_gpu_encoding,
        )

        video_creation_time = time.time() - video_creation_start_time
        tiled_video_total_time = time.time() - tiled_video_start_time

        logger.info("  - Video creation: %.3fs", video_creation_time)
        logger.info(
            "Created tiled video: %s with %s frames (total time: %.3fs)", video_path, num_frames, tiled_video_total_time
        )
        return None

    else:  # basic_vis_utils.TilingOutputMode.IMAGE mode
        # Create tiled frame visualization
        frame_creation_start_time = time.time()
        logger.info("  - Creating tiled frame visualization...")

        tiled_image = basic_vis_utils.tile_and_save_media(
            camera_images_with_text,
            world_images_with_text,
            camera_spatial_rgb,
            world_spatial_rgb,
            output_mode=basic_vis_utils.TilingOutputMode.IMAGE,
            frame_indices=None,
            camera_patch_features=patch_camera_rgb,
            camera_type_features=type_camera_rgb,
            spatial_cosine_sim=spatial_cosine_sim_rgb,
            camera_backbone_features=camera_backbone_numpy,
            world_backbone_features=world_backbone_numpy,
        )

        frame_creation_time = time.time() - frame_creation_start_time
        tiled_video_total_time = time.time() - tiled_video_start_time

        logger.info("  - Frame creation: %.3fs", frame_creation_time)
        logger.info("Created tiled frames with %s frames (total time: %.3fs)", num_frames, tiled_video_total_time)
        return tiled_image


class _ChunkedFrames:
    """Frame-indexed view over a list of uint8 chunks, so a frame-wise loop needs no chunk arithmetic."""

    def __init__(self, chunks: list[np.ndarray], bounds: list[tuple[int, int]], total: int) -> None:
        self._chunks = chunks
        self._bounds = bounds
        self._total = total

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, frame_idx: int) -> np.ndarray:
        """Return the frame as a view into its chunk, so drawing on it mutates the chunk."""
        for chunk, (start, stop) in zip(self._chunks, self._bounds, strict=True):
            if start <= frame_idx < stop:
                return chunk[frame_idx - start]
        raise IndexError(f"frame {frame_idx} out of range for {self._total} frames")


def add_text_overlays(
    images: np.ndarray,
    output_batch: dict[str, torch.Tensor],
    frame_offset: int,
    view_type: str,
    similarity_indices: list[int] | None = None,
    session_ids: list | None = None,
    frame_indices: list | None = None,
) -> np.ndarray:
    """
    Add frame numbers and similarity scores as text overlays in the top right corner.

    Args:
        images: Input images array (num_frames, H, W, 3) in range [0, 1]
        output_batch: Model output containing similarity matrix
        frame_offset: Frame offset for numbering
        view_type: "camera" or "world" for labeling
        similarity_indices: Indices into similarity_matrix corresponding to each image. If None, assumes 1:1
                           correspondence (images[i] -> similarity_matrix[i]).
        session_ids: Optional list of session IDs for each frame
        frame_indices: Optional list of frame indices for each frame

    Returns:
        images_with_text: Images with text overlays
    """
    logger.info(
        "  - Adding text overlays to %s images: %s, dtype: %s, range: [%.3f, %.3f]",
        view_type,
        images.shape,
        images.dtype,
        images.min(),
        images.max(),
    )

    scale = 255.0
    if images.max() > 1.0:
        logger.warning("  - Images appear to be in [0, 255] range, normalizing to [0, 1]")
        scale = 1.0

    # The only full-size allocation is the float32 result, which doubles as scratch for the uint8 conversion. Frames
    # are converted to uint8 in chunks of about 8 MB, drawn on, and divided back into the result chunk by chunk.
    images_with_text = np.empty(images.shape, dtype=np.float32)
    chunk_frames = max(1, int(8 * 1024 * 1024 // max(int(np.prod(images.shape[1:])), 1)))
    chunk_bounds = [
        (start, min(start + chunk_frames, images.shape[0])) for start in range(0, images.shape[0], chunk_frames)
    ]
    uint8_chunks = []
    for start, stop in chunk_bounds:
        window = images_with_text[start:stop]
        np.multiply(images[start:stop], scale, out=window)
        np.clip(window, 0, 255, out=window)
        uint8_chunks.append(window.astype(np.uint8))
    images_uint8 = _ChunkedFrames(uint8_chunks, chunk_bounds, images.shape[0])
    logger.info("  - Converted to uint8 in %s chunk(s)", len(chunk_bounds))

    # Get similarity scores (diagonal of similarity matrix for frame-to-frame similarity)
    similarity_scores = None
    if "similarity_matrix" in output_batch and output_batch["similarity_matrix"] is not None:
        similarity_matrix = output_batch["similarity_matrix"].cpu().float().numpy()
        logger.info(
            "  - Similarity matrix shape: %s, images: %s, similarity_indices: %s",
            similarity_matrix.shape,
            len(images),
            similarity_indices,
        )

        # Handle both 1D (N,) and 2D (N, N) similarity matrices
        if len(similarity_matrix.shape) == 1:
            # 1D similarity scores
            if similarity_indices is not None:
                # Use provided indices to select corresponding similarity scores
                if max(similarity_indices) < similarity_matrix.shape[0]:
                    similarity_scores = similarity_matrix[similarity_indices]
                    logger.info(
                        "  - Using indexed 1D similarity scores: %s (from indices %s)",
                        similarity_scores.shape,
                        similarity_indices,
                    )
                else:
                    logger.warning(
                        "  - similarity_indices out of bounds: max index %s >= %s",
                        max(similarity_indices),
                        similarity_matrix.shape[0],
                    )
            elif similarity_matrix.shape[0] == len(images):
                # Direct 1:1 correspondence
                similarity_scores = similarity_matrix
                logger.info("  - Using 1D similarity scores directly: %s", similarity_scores.shape)
            else:
                logger.warning(
                    "  - 1D similarity scores length mismatch: %s vs %s images. Pass similarity_indices to fix.",
                    similarity_matrix.shape[0],
                    len(images),
                )
        elif len(similarity_matrix.shape) == 2:
            # 2D similarity matrix - extract diagonal
            if similarity_indices is not None:
                # Use provided indices to select corresponding diagonal elements
                if max(similarity_indices) < similarity_matrix.shape[0]:
                    similarity_scores = np.array([similarity_matrix[i, i] for i in similarity_indices])
                    logger.info("  - Using indexed diagonal similarity scores: %s", similarity_scores.shape)
                else:
                    logger.warning(
                        "  - similarity_indices out of bounds: max index %s >= %s",
                        max(similarity_indices),
                        similarity_matrix.shape[0],
                    )
            elif similarity_matrix.shape[0] == similarity_matrix.shape[1] == len(images):
                similarity_scores = np.diag(similarity_matrix)
                logger.info("  - Using diagonal similarity scores from 2D matrix: %s", similarity_scores.shape)
            else:
                logger.warning(
                    "  - 2D similarity matrix shape mismatch: %s vs %s images. Pass similarity_indices to fix.",
                    similarity_matrix.shape,
                    len(images),
                )
        else:
            logger.warning("  - Unexpected similarity matrix dimensions: %s", similarity_matrix.shape)
    else:
        logger.info("  - No similarity matrix available in output_batch")

    # Add text to each frame
    for frame_idx in range(len(images_uint8)):
        frame = images_uint8[frame_idx]

        # Session ID and frame index if available
        if session_ids is not None and frame_idx < len(session_ids):
            sid = session_ids[frame_idx]
            if frame_indices is not None and frame_idx < len(frame_indices):
                fnum = frame_indices[frame_idx]
                frame_text = f"S:{sid} F:{fnum}"
            else:
                frame_text = f"S:{sid} F:{frame_offset + frame_idx}"
        else:
            # Frame number (with offset) - fallback
            frame_num = frame_offset + frame_idx
            frame_text = f"Frame {frame_num}"

        # Similarity score if available
        if similarity_scores is not None:
            sim_score = similarity_scores[frame_idx]
            sim_text = f"Sim: {sim_score:.3f}"
        else:
            sim_text = "Sim: N/A"

        # View type label
        view_text = f"{view_type.capitalize()}"

        # Position frame/similarity/view text in the top-right corner.
        image_video_utils.draw_topright_lines(frame, [view_text, frame_text, sim_text], font_scale=0.6)

    for chunk, (start, stop) in zip(uint8_chunks, chunk_bounds, strict=True):
        np.divide(chunk, 255.0, out=images_with_text[start:stop])

    logger.info("  - Successfully added text overlays to %s frames", len(images_uint8))
    return images_with_text
