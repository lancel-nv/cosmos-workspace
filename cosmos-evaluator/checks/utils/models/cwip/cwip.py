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

"""Contrastive World-Image Pre-training - Deployment/Inference Version."""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from typing import Any, Literal

import einops
import numpy as np
from omegaconf import open_dict
from safetensors.torch import load_file
from third_party.detectron2.config.instantiate import instantiate
from third_party.detectron2.config.lazy import LazyCall as L
import torch
from torch import nn
import torch.nn.functional as F

from checks.utils.models.cwip.modules import parallel_pe as pe_module
from checks.utils.models.cwip.modules import parallelism as parallelism_module
from checks.utils.models.cwip.modules.class_mapping import (
    get_model_class_index_for_object_type,
    get_object_type_class_mapping,
)
from checks.utils.models.cwip.modules.configs.defaults.model import InferenceConfig, ModelConfig
from checks.utils.models.cwip.modules.hf_integration import CWIPPreTrainedModel
from checks.utils.models.cwip.modules.parallel_patchformer import ParallelPatchFormer
from checks.utils.models.cwip.visualization import (
    basic_vis_utils,
    feature_vis_utils,
    image_video_utils,
    metrics_utils,
    patch_vis_utils,
    world_model_utils,
)

# Set up logger
logger = logging.getLogger(__name__)


def _log_safe(value: object) -> str:
    """
    Render ``value`` for logging with CR/LF escaped.

    Checkpoint paths, ``config.json`` fields, tensor names, and backend exception text all originate outside this
    process, and the logger treats a newline as a record separator: an embedded ``\\n`` would let that data forge
    additional, attacker-authored log entries. Escaping keeps the payload visible on a single record instead of
    dropping it.
    """
    return str(value).replace("\r", "\\r").replace("\n", "\\n")


# Valid values for ``CWIPInference.create_inference_visualizations``'s ``similarity_plots`` argument. The implementation
# still has a friendly warn-and-fall-back-to-``"all"`` branch for unknown values, so the runtime behavior is unchanged
# — this alias just lets IDEs and mypy catch typos at the call site (the HF inference tool passes ``"radial"``
# literally).
SimilarityPlotMode = Literal["all", "radial", "none"]


class CWIPInference(CWIPPreTrainedModel):
    """
    CWIP Inference Model - Minimal deployment version.

    This class contains only the essential inference functionality:

    - Model initialization (encoders, decoders, learnable parameters)
    - Forward pass
    - Encoding images
    - HuggingFace format conversion and loading

    Config attribute semantics:

    * :attr:`config` -- the canonical
      :class:`~checks.utils.models.cwip.modules.configs.defaults.model.ModelConfig`.
      Set in **every** code path (direct construction or HF) so that all internal CWIP code and any subclasses can read
      ``self.config.camera_encoder`` etc. unconditionally.

    * :attr:`hf_config` -- the :class:`HFModelConfig`
      (:class:`transformers.PretrainedConfig`) companion, only set when the model is constructed from a
      :class:`HFModelConfig` (i.e. the HF path). ``None`` otherwise.
    """

    def _instantiate_encoders_and_decoders(self) -> None:
        """Instantiate camera and world encoders."""
        # Get object_type_class_mapping from config (if available)
        # Resolve string identifier (e.g., "v3") to actual class if needed
        object_type_class_mapping = get_object_type_class_mapping(
            getattr(self.config, "object_type_class_mapping", None)
        )

        camera_encoder_config = self.config.camera_encoder
        world_encoder_config = self.config.world_encoder
        patch_classifier_config = self.config.patch_classifier

        # Backbone encoders
        self.camera = instantiate(camera_encoder_config)
        self.world = instantiate(world_encoder_config) if world_encoder_config is not None else None

        # Patch classification heads. ``object_type_class_mapping`` is set on ``ModelConfig`` (not on
        # ``patch_classifier``), so we inject it onto a deepcopy of the patch_classifier sub-config before
        # instantiation. The ``open_dict`` is needed because in the imaginaire4 / Hydra path the sub-config is in struct
        # mode (Hydra ``compose()``) and ``object_type_class_mapping`` is not in the schema -- our training-side
        # ``ParallelPatchFormer`` subclass uses ``*args, **kwargs`` forwarding, which hides deployment-base defaults
        # from ``LazyCall``'s constructor introspection. In the cosmos_evaluator / detectron2 path the sub-config is
        # already non-struct (its ``LazyCall`` doesn't introspect, and its ``instantiate(cfg)`` doesn't accept override
        # kwargs), so ``open_dict`` is a no-op there. Same source therefore works in both runtimes.
        if patch_classifier_config is not None:
            if object_type_class_mapping is not None:
                patch_classifier_config = copy.deepcopy(patch_classifier_config)
                with open_dict(patch_classifier_config):
                    patch_classifier_config.object_type_class_mapping = object_type_class_mapping
            self.patch_classifier = instantiate(patch_classifier_config)
        else:
            self.patch_classifier = None

    def _initialize_remaining_parameters(self) -> None:
        """Initialize remaining parameters."""
        # Set up precision for inference
        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.config.precision]

        logger.info("CWIP Inference Model: precision %s", self.precision)

        # Initialize image_size from encoder (will be updated dynamically based on input images)
        # encoder.image_size might be a single int, list, or tuple - always convert to tuple
        encoder_size = self.camera.encoder.image_size
        if isinstance(encoder_size, list):
            # If it's a list like [512, 512], convert to tuple
            self.image_size = tuple(encoder_size)
        elif isinstance(encoder_size, tuple):
            # Already a tuple, use as-is
            self.image_size = encoder_size
        else:
            # Single int, create (H, W) tuple
            self.image_size = (encoder_size, encoder_size)
        # inplace=False so the normalize transform doesn't mutate the caller's input tensors, which may still be
        # referenced for visualization.
        self.image_transform = image_video_utils.get_pe_image_transform(
            image_size=self.image_size,
            normalize=True,
            inplace=False,
        )

        # Convert model to specified precision
        if self.precision != torch.float32:
            self.to(dtype=self.precision)
            logger.info("Converted model to %s", self.precision)

        # Opt-in memory-layout flag stamped by callers that also run ``self.to(memory_format=torch.channels_last)`` on
        # the loaded model (e.g. ``tools/inference.py --channels_last``). When True, ``forward`` converts the 4-D image
        # inputs to channels_last after the image transform (torchvision transforms force contiguous NCHW, so the
        # conversion must happen here, not at the caller).
        self._use_channels_last: bool = False

    @property
    def output_downsample_factor(self) -> int:
        """
        Downsample factor from input resolution to the model's final output (patch logits).

        Input (H, W) is reduced to (H // factor, W // factor) at the patch classifier output (similarity/type logits).
        Equals encoder patch_size divided by the patch classifier's input_resize_factor (e.g. 16 / 2 = 8 when
        patch_size=16 and input_resize_factor=2). If there is no patch classifier, returns encoder patch_size.
        """
        patch_size = self.camera.encoder.patch_size
        if self.patch_classifier is not None:
            up = getattr(self.patch_classifier, "input_resize_factor", 1) or 1
            up = max(int(up), 1)
            return patch_size // up
        return patch_size

    def trim_output_to_num_frames(
        self,
        output: dict[str, torch.Tensor | list | None],
        num_frames: int,
    ) -> None:
        """
        Drop trailing batch rows from encode/classify outputs (FP padding).

        Frame parallelism pads each forward to a multiple of the frame-parallel world size; after
        :func:`~checks.utils.models.cwip.modules.parallelism.gather_features_from_fp` the leading dimension can exceed
        the number of real frames in the chunk. Chunked inference must trim each chunk to the actual slice length so
        concatenated logits match ``world_image.shape[0]`` and ``gt_foreground_pixel_counts``.
        """
        if num_frames <= 0:
            return

        def trim_tensor(t: torch.Tensor | None) -> torch.Tensor | None:
            if t is None or t.shape[0] <= num_frames:
                return t
            return t[:num_frames].contiguous()

        tensor_keys = (
            "camera_global_features",
            "world_global_features",
            "camera_spatial_features",
            "world_spatial_features",
            "patch_similarity_logits",
            "patch_type_logits",
            "camera_patch_features",
        )
        for key in tensor_keys:
            v = output.get(key)
            if isinstance(v, torch.Tensor):
                output[key] = trim_tensor(v)

        for key in ("camera_spatial_features_list", "world_spatial_features_list"):
            lst = output.get(key)
            if isinstance(lst, list):
                output[key] = [trim_tensor(t) if isinstance(t, torch.Tensor) else t for t in lst]

    def encode_image(
        self,
        image: torch.Tensor,
        normalize: bool = False,
        input_type: str = "camera",
        layer_idx: int | list[int] = -1,
        rearrange_feature_list_to_spatial: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Encode images into feature vectors.

        Args:
            image: (B, C, H, W)
            normalize: bool, whether to normalize the feature vector to unit length
            input_type: str, "camera" or "world"
            layer_idx: the layer index(es) to return the features from
            rearrange_feature_list_to_spatial: whether to rearrange the feature list to spatial features

        Returns:
            output: dict containing the global features, spatial features, and spatial features list
                "global_features": (B, D), the global features
                "spatial_features": (B, (H/P) * (W/P), D), the spatial features
                "spatial_features_list": list of (B, (H/P) * (W/P), D), where D is the dimension of the
                    features from the specified layers in @layer_idx
                if @rearrange_feature_list_to_spatial is True: list of (B, H/P, W/P, D)
        """
        strip_cls_token = rearrange_feature_list_to_spatial
        grid_h, grid_w = (
            image.shape[2] // self.camera.encoder.patch_size,
            image.shape[3] // self.camera.encoder.patch_size,
        )

        # No separate global embedding heads for camera and world in deployment version
        if input_type == "camera":
            image_features = self.camera(image, layer_idx=layer_idx, strip_cls_token=strip_cls_token)
        elif input_type == "world":
            if self.world is None:
                raise ValueError("World encoder is not initialized")
            image_features = self.world(image, layer_idx=layer_idx, strip_cls_token=strip_cls_token)
        else:
            raise ValueError(f"Invalid input type: {input_type}")

        # (B, D)
        global_features = image_features["global_features"]
        # (B, (H/P) * (W/P), D)
        spatial_features = image_features["spatial_features"]
        spatial_features_list = image_features["spatial_features_list"]

        # Apply normalization BEFORE rearrangement (normalize along D dimension, dim=-1)
        if normalize:
            global_features = F.normalize(global_features, dim=-1)
            spatial_features = F.normalize(spatial_features, dim=-1)  # (B, num_patches, D) - normalizes along D

        if rearrange_feature_list_to_spatial:
            # (B, (H/P) * (W/P), D) -> (B, D, H/P, W/P)
            spatial_features = einops.rearrange(spatial_features, "b (h w) d -> b d h w", h=grid_h, w=grid_w)
            spatial_features_list = [
                einops.rearrange(feats, "b (h w) d -> b d h w", h=grid_h, w=grid_w) for feats in spatial_features_list
            ]

        output = {
            "global_features": global_features,
            "spatial_features": spatial_features,
            "spatial_features_list": spatial_features_list,
        }
        return output

    def _encode_and_classify(
        self,
        camera_image: torch.Tensor,
        world_image: torch.Tensor | None,
        camera_layer_idx: list[int],
        world_layer_idx: list[int],
        camera_only: bool | None = None,
    ) -> dict[str, torch.Tensor | list | None]:
        """
        Encode camera and world images and apply patch classifier if available.

        Frame-parallel split/gather is intentionally handled by the caller, not inside this method.

        Args:
            camera_image: (B * N, C, H, W) - Camera images (already preprocessed and on correct device/dtype)
            world_image: (B * N, C, H, W) - World images (already preprocessed and on correct device/dtype)
            camera_layer_idx: Layer indices to extract camera features from
            world_layer_idx: Layer indices to extract world features from

        Returns:
            Dictionary containing:
                - camera_global_features: (B * N, D) - Global camera features
                - world_global_features: (B * N, D) - Global world features
                - camera_spatial_features: (B * N, D, H/P, W/P) - Spatial camera features
                - world_spatial_features: (B * N, D, H/P, W/P) - Spatial world features
                - camera_spatial_features_list: list of (B * N, D, H/P, W/P) - Spatial camera features
                    at different layers
                - world_spatial_features_list: list of (B * N, D, H/P, W/P) - Spatial world features
                    at different layers
                - patch_similarity_logits: (B * N, num_classes, H/P, W/P) if a patch classifier is configured
                - patch_type_logits: (B * N, num_classes, H/P, W/P) if PatchFormer is used
                - camera_patch_features: Camera patch features if PatchFormer is used

        Per-class bias and temperature are applied only when ``config.inference.mode`` is True.

        ``camera_only`` defaults to ``config.inference.camera_only_mode`` when not passed; :meth:`forward` passes the
        value it resolved so the mode is consistent across the pass without a second config read.
        """
        if camera_only is None:
            camera_only = getattr(self.config.inference, "camera_only_mode", False)
        if not camera_only and world_image is None:
            raise ValueError("world_image is required unless camera_only_mode=True")

        # Encode images
        camera_features = self.encode_image(
            camera_image,
            normalize=True,
            layer_idx=camera_layer_idx,
            input_type="camera",
            rearrange_feature_list_to_spatial=True,
        )
        world_features: dict[str, torch.Tensor | list[torch.Tensor]] | None = None
        if not camera_only:
            assert world_image is not None  # validated by early raise in this method
            world_features = self.encode_image(
                world_image,
                normalize=True,
                layer_idx=world_layer_idx,
                input_type="world",
                rearrange_feature_list_to_spatial=True,
            )

        # Generic extension point for post-encoder transforms. ``CWIPInference`` itself defines this as a no-op;
        # subclasses can override it to apply transforms to ``spatial_features`` and every entry of
        # ``spatial_features_list`` before they reach the patch classifier and the visualization tensors.
        if world_features is not None and world_image is not None:
            self._post_encode_hook(
                camera_features=camera_features,
                world_features=world_features,
                world_image=world_image,
            )

        # Apply patch classifier when present; bias/temperature only when ``inference.mode``.
        if self.patch_classifier is not None:
            world_spatial = world_features["spatial_features_list"][-1] if world_features is not None else None
            patch_classifier_output = self.patch_classifier(
                camera_features["spatial_features_list"][-1],
                world_spatial,
                apply_bias=self.config.inference.mode,
                apply_temperature=self.config.inference.mode,
            )
            # Extract logits from dict output (key depends on classifier type)
            # PatchClassifier returns "patch_similarity_logits", PatchFormer returns "patch_type_logits"
            patch_similarity_logits = patch_classifier_output.get("patch_similarity_logits")
            patch_type_logits = patch_classifier_output.get("patch_type_logits")
            # For PatchFormer, patch_features are also available in the dict
            camera_patch_features = patch_classifier_output.get("camera_patch_features", None)
            camera_type_features = patch_classifier_output.get("camera_type_features", None)
            # ``camera_type_features`` / ``camera_patch_features`` are high-resolution
            # (B, D, H_out, W_out) maps consumed *only* by the tiled-feature video
            # (``feature_vis_utils.create_tiled_video``), which is skipped in camera-only
            # mode. Retaining them there is pure dead weight -- at native/high resolution
            # a single one is ~90 MB/frame, so a full clip is tens of GB that never gets
            # used. Drop them from the camera-only output so they don't accumulate across
            # the chunked inference loop or blow up visualization.
            if camera_only:
                camera_type_features = None
                camera_patch_features = None
        else:
            patch_similarity_logits = None
            patch_type_logits = None
            camera_patch_features = None
            camera_type_features = None

        output = {
            "camera_global_features": camera_features["global_features"],
            "world_global_features": world_features["global_features"] if world_features is not None else None,
            "camera_spatial_features": camera_features["spatial_features"],
            "world_spatial_features": world_features["spatial_features"] if world_features is not None else None,
            "camera_spatial_features_list": camera_features["spatial_features_list"],
            "world_spatial_features_list": (
                world_features["spatial_features_list"] if world_features is not None else None
            ),
            "patch_similarity_logits": patch_similarity_logits,
            "patch_type_logits": patch_type_logits,
            "camera_patch_features": camera_patch_features,
            "camera_type_features": camera_type_features,
        }

        return output

    def _post_encode_hook(
        self,
        camera_features: dict[str, torch.Tensor | list[torch.Tensor]],
        world_features: dict[str, torch.Tensor | list[torch.Tensor]],
        world_image: torch.Tensor,
    ) -> None:
        """No-op extension point for post-encoder transforms.

        Called from :meth:`_encode_and_classify` after both streams have been encoded and rearranged to ``(B, D, H',
        W')`` and *before* the patch classifier runs. Subclasses may override this to mutate ``camera_features`` /
        ``world_features`` in place.

        Args:
            camera_features: Output of ``encode_image`` for the camera stream (mutated in place by overrides).
            world_features: Output of ``encode_image`` for the world stream (mutated in place by overrides).
            world_image: World input tensor; available so overrides can derive input-pixel-based masks (e.g. don't-care
                patches).
        """
        del camera_features, world_features, world_image  # unused in the no-op base

    def forward(
        self,
        camera_image: torch.Tensor,
        world_image: torch.Tensor | None = None,
        camera_only_mode: bool | None = None,
    ) -> dict[str, torch.Tensor | list | None]:
        """
        Forward the camera and world images through the model.

        Args:
            camera_image: (B * N, C, H, W)
            world_image: (B * N, C, H, W)
            camera_only_mode: Per-call override of the camera-only branch. When ``None`` (default), the mode is read
                from ``config.inference.camera_only_mode``. Passing it explicitly lets a caller that shares one model
                across concurrent clips select the mode per forward without mutating shared config, so it needs no
                outer lock to keep the mode stable against a concurrent clip.

        Returns:
            Dictionary containing:
                - camera_global_features: (B * N, D) - Global camera features
                - world_global_features: (B * N, D) - Global world features
                - camera_spatial_features: (B * N, D, H/P, W/P) - Spatial camera features
                - world_spatial_features: (B * N, D, H/P, W/P) - Spatial world features
                - camera_spatial_features_list: list of (B * N, D, H/P, W/P) - Spatial camera features
                    at different layers
                - world_spatial_features_list: list of (B * N, D, H/P, W/P) - Spatial world features
                    at different layers
                - patch_similarity_logits: (B * N, num_classes, H/P, W/P) if a patch classifier is configured
                - patch_type_logits: (B * N, num_classes, H/P, W/P) if PatchFormer is used
                - camera_patch_features: Camera patch features if PatchFormer is used
        """
        camera_only = (
            camera_only_mode
            if camera_only_mode is not None
            else getattr(self.config.inference, "camera_only_mode", False)
        )

        # Held for the whole pass: the resolution-dependent state below and the Rope2D grids inside the encoders are
        # instance state, and a checker service shares one model across the jobs it runs concurrently. See
        # ``CWIPPreTrainedModel.__init__``.
        with self.inference_lock:
            # Move input tensors to model device (keep float32 for transform)
            tensor_device = next(self.parameters()).device
            camera_image = camera_image.to(device=tensor_device)
            if not camera_only:
                if world_image is None:
                    raise ValueError("world_image is required unless camera_only_mode=True")
                world_image = world_image.to(device=tensor_device)

            # Ensure camera and world images have the same shape
            if not camera_only:
                assert world_image is not None  # validated by earlier raise
                if camera_image.shape != world_image.shape:
                    logger.info(
                        "Camera image shape %s does not match world image shape %s. Resize to world image shape.",
                        camera_image.shape,
                        world_image.shape,
                    )

            # Get image dimensions and update transform if needed
            if camera_only:
                _, _, image_height, image_width = camera_image.shape
            else:
                assert world_image is not None  # validated by earlier raise
                _, _, image_height, image_width = world_image.shape

            if image_height != self.image_size[0] or image_width != self.image_size[1]:
                self.image_size = (image_height, image_width)
                # inplace=False: see comment in __init__
                self.image_transform = image_video_utils.get_pe_image_transform(
                    image_size=self.image_size,
                    normalize=True,
                    inplace=False,
                )
                logger.info("Updated image size to %s", self.image_size)

            # Apply image transforms
            camera_image = self.image_transform(camera_image)
            if not camera_only:
                world_image = self.image_transform(world_image)

            # Convert to model's precision
            camera_image = camera_image.to(dtype=self.precision)
            if not camera_only:
                assert world_image is not None  # validated by earlier raise
                world_image = world_image.to(dtype=self.precision)

            # Match the model's memory layout when channels_last was opted in by the loader (see
            # ``self._use_channels_last`` in ``__init__``). The image transform above re-materializes contiguous NCHW
            # tensors, so the layout swap has to happen *after* it.
            if self._use_channels_last:
                camera_image = camera_image.contiguous(memory_format=torch.channels_last)
                if not camera_only:
                    assert world_image is not None  # validated by earlier raise
                    world_image = world_image.contiguous(memory_format=torch.channels_last)

            # Encode images and apply patch classifier. When ``viz_capture_backbone_features`` is set, also retain the
            # pre-alignment (frozen-backbone) layer for both streams so the debug 3x3 tile can populate row 2 during
            # deployment-time inference. Adds one ``(B, D, H', W')`` tensor per stream; no extra compute.
            if self.config.viz_capture_backbone_features:
                camera_layer_idx = (
                    [-self.camera.num_alignment_layers - 1, -1] if self.camera.num_alignment_layers > 0 else [-1]
                )
                world_layer_idx = (
                    [-self.world.num_alignment_layers - 1, -1]
                    if (self.world is not None and self.world.num_alignment_layers > 0)
                    else [-1]
                )
            else:
                camera_layer_idx = [-1]
                world_layer_idx = [-1]
            return self._encode_and_classify(
                camera_image=camera_image,
                world_image=world_image,
                camera_layer_idx=camera_layer_idx,
                world_layer_idx=world_layer_idx,
                camera_only=camera_only,
            )

    # Canonical default: ``MATCH + EXTRA + WRONG`` — the three classes that all say "the model thinks an object is
    # here". ``MATCH`` ("object that's also in the world frame") and ``EXTRA`` ("object the model detected that's NOT in
    # the world frame") are the two presence classes for ground-truth-vs-detection alignment; ``WRONG`` ("model detected
    # an object that aligns with the world frame, but assigned the wrong type") is also a model-presence signal — the
    # type head was confused, not the similarity head. ``DONT_CARE`` (background) and ``MISSING`` ("model thinks NOTHING
    # is here, but the world says something should be") are deliberately excluded — they are the two non-presence
    # classes.
    #
    # The constant itself lives in :mod:`checks.utils.models.cwip.visualization.patch_vis_utils` so the visualization
    # module — the primary consumer — can use it without a runtime circular import on ``cwip.py``. Re-exported here as a
    # class attribute for back-compat with existing callers/tests that reference
    # ``CWIPInference.DEFAULT_PATCH_SIMILARITY_MASK_CLASSES``.
    DEFAULT_PATCH_SIMILARITY_MASK_CLASSES: tuple[world_model_utils.PatchSimilarityType, ...] = (
        patch_vis_utils.DEFAULT_PATCH_SIMILARITY_MASK_CLASSES
    )

    # The implementation and full docstring live in :func:`patch_vis_utils.compute_patch_similarity_class_mask` — moved
    # there to break a runtime circular import (visualization is the primary consumer and ``cwip.py`` imports
    # ``patch_vis_utils`` at top, so visualization unit tests that didn't carry ``cwip`` in their runfiles couldn't
    # lazy-import ``CWIPInference``). Re-exported as a ``@staticmethod`` here so existing callers that reach for
    # ``CWIPInference.compute_patch_similarity_class_mask`` keep working unchanged. ``help()`` / IDE hover surface the
    # underlying function's docstring transparently.
    compute_patch_similarity_class_mask = staticmethod(patch_vis_utils.compute_patch_similarity_class_mask)

    def create_inference_visualizations(
        self,
        output_batch: dict[str, torch.Tensor],
        data_batch: dict[str, torch.Tensor],
        iteration: int,
        output_dir: str,
        similarity_plots: SimilarityPlotMode = "all",
        save_patch_data_dumps: bool = True,
        presence_mask: torch.Tensor | np.ndarray | None = None,
        presence_label: str | None = None,
    ) -> None:
        """
        Create and save comprehensive inference visualizations.

        This method generates multiple types of visualizations for model analysis:
        1. Similarity matrix visualization (heatmap with frame indices)
        2. Enhanced diagonal similarity plots
        3. Tiled video showing camera, world, and feature maps
        4. Combined patch visualization video (if patch classifier present)
        5. Detailed patch visualizations (similarity and/or type predictions)
        6. Per-frame patch scores (CSV and JSON formats)
        7. Per-frame consistency scores
        8. Patch prediction arrays (.npy files)

        Video encoding automatically uses GPU (NVENC) if available, falls back to CPU. Memory is cleared after each
        visualization step to prevent GPU OOM.

        Args:
            output_batch: Model outputs containing:
                - similarity_matrix: Frame-to-frame similarity (N, N). When this key is missing but
                  ``camera_global_features`` and ``world_global_features`` are present, a similarity matrix is derived
                  on the fly so HF inference (which only persists features, not the similarity matrix) can still produce
                  the radial diagonal plot.
                - camera_features: Camera feature embeddings
                - world_features: World feature embeddings
                - patch_similarity_logits: Patch-level predictions (optional)
                - patch_type_logits: Object type predictions (optional)
            data_batch: Input data containing:
                - camera_image: Camera video frames (N, C, H, W)
                - world_image: World video frames (N, C, H, W)
            iteration: Iteration number for file naming
            output_dir: Directory to save all visualizations
            similarity_plots: Which similarity-matrix plots to render in Step 1.
                ``"all"`` renders the full set (matrix heatmap + gradient + heatmap strip + radial + dashboard).
                ``"radial"`` only writes ``similarity_radial_iter_*.png`` -- this is what the HF inference CLI uses so
                we don't spam the output dir.
                ``"none"`` skips Step 1 plots entirely.
            save_patch_data_dumps: When ``True`` dumps per-frame patch scores (CSV/JSON), per-frame consistency scores
                (CSV/JSON), and ``.npy`` prediction arrays. The HF inference CLI passes ``False`` because those are
                offline-analysis artifacts, not visualizations.
            presence_mask: Optional ``(N, h, w)`` 0/1 mask shown in the combined patch visualization's row-3 trio in
                place of the soft similarity mask (see
                :func:`~checks.utils.models.cwip.visualization.patch_vis_utils.create_combined_patch_visualization`).
            presence_label: Text naming ``presence_mask`` in that trio's labels.

        Returns:
            None. All visualizations are saved to disk at output_dir.

        Note:
            - GPU memory is cleared after each step using torch.cuda.empty_cache()
            - Frame offsets are read from self.config.inference.camera_start_frame
              and self.config.inference.world_start_frame
            - Visualization resolution for detailed views is controlled by
              self.config.inference.detailed_vis_subplot_resolution
        """
        total_viz_start_time = time.time()
        logger.info("Starting inference visualizations...")

        # Ensure output_dir exists. Several downstream `plt.savefig(...)` calls (e.g. `create_radial_similarity_plot`,
        # `save_similarity_matrix_with_frame_indices`) do not create their parent directory, and the HF inference CLI
        # does not pre-create it either. Without this, Step 1 fails immediately with `FileNotFoundError` on a fresh
        # output path.
        os.makedirs(output_dir, exist_ok=True)

        # HF inference does not persist `similarity_matrix` (it only saves the per-frame global features). Derive it on
        # the fly from camera/world global features so Step 1 can still emit the radial diagonal plot.
        if (
            ("similarity_matrix" not in output_batch or output_batch.get("similarity_matrix") is None)
            and "camera_global_features" in output_batch
            and "world_global_features" in output_batch
            and output_batch["camera_global_features"] is not None
            and output_batch["world_global_features"] is not None
        ):
            cam = output_batch["camera_global_features"].float()
            world = output_batch["world_global_features"].float()
            cam = F.normalize(cam, dim=-1)
            world = F.normalize(world, dim=-1)
            output_batch["similarity_matrix"] = cam @ world.t()
            logger.info(
                "Derived similarity_matrix from global features (shape=%s) for visualization",
                tuple(output_batch["similarity_matrix"].shape),
            )

        # 1. Create enhanced similarity matrix visualization using utility function
        if "similarity_matrix" in output_batch and output_batch["similarity_matrix"] is not None:
            step_start_time = time.time()
            logger.info("Step 1: Creating similarity matrix visualization...")
            similarity = output_batch["similarity_matrix"].cpu().float().numpy()
            similarity_score = float(similarity.diagonal().mean())

            # Calculate frame indices accounting for offsets
            camera_start = getattr(self.config.inference, "camera_start_frame", 0)
            world_start = getattr(self.config.inference, "world_start_frame", 0)

            # Step 1 PNG plots -- gated by `similarity_plots` so HF inference can narrow this to just
            # `similarity_radial_iter_*.png` (the rest spam the output directory and are redundant for deployment).
            if similarity_plots == "radial":
                # Replicate just the radial slice of save_enhanced_diagonal_plots without producing gradient / heatmap
                # strip / dashboard PNGs.
                diagonal_scores = np.diag(similarity)
                num_frames = len(diagonal_scores)
                camera_frame_indices = np.arange(camera_start, camera_start + num_frames)
                world_frame_indices = np.arange(world_start, world_start + num_frames)
                basic_vis_utils.create_radial_similarity_plot(
                    diagonal_scores,
                    camera_frame_indices,
                    world_frame_indices,
                    os.path.join(output_dir, f"similarity_radial_iter_{iteration:06d}.png"),
                )
            elif similarity_plots == "none":
                logger.info("Step 1: skipping similarity matrix plots (similarity_plots='none')")
            else:
                # "all" -- the full set of Step 1 PNGs. Also the safe fallback for any unexpected value.
                basic_vis_utils.save_similarity_matrix_with_frame_indices(
                    similarity,
                    os.path.join(output_dir, f"similarity_iter_{iteration:06d}.png"),
                    camera_start,
                    world_start,
                    similarity_score,
                )
                basic_vis_utils.save_enhanced_diagonal_plots(
                    similarity, output_dir, iteration, camera_start, world_start
                )

            # Dump per-frame patch scores for later analysis (if available and enabled).
            # Check for either patch_similarity_logits (PatchClassifier) or patch_type_logits (PatchFormer)
            patch_logits_key = None
            if "patch_similarity_logits" in output_batch and output_batch["patch_similarity_logits"] is not None:
                patch_logits_key = "patch_similarity_logits"
            elif "patch_type_logits" in output_batch and output_batch["patch_type_logits"] is not None:
                patch_logits_key = "patch_type_logits"

            if patch_logits_key is not None and save_patch_data_dumps:
                # Get activation function from config (default to 'sigmoid' for backward compatibility)
                activation = getattr(self.config.inference, "patch_classifier_activation", "sigmoid")

                # Compute patch results to get per_frame_class_percentages (reuse existing utility function)
                gt_patch_similarity = output_batch.get("gt_patch_similarity", None)
                gt_foreground_pixel_counts = output_batch.get("gt_foreground_pixel_counts", None)
                patch_results = metrics_utils.compute_patch_predictions_and_percentages(
                    output_batch[patch_logits_key],
                    treat_wrong_type_as_match=self.patch_classifier.treat_wrong_type_as_match,
                    gt_patch_similarity=gt_patch_similarity,
                    gt_foreground_pixel_counts=gt_foreground_pixel_counts,
                    activation=activation,
                )
                # Use default percentages for backward compatibility
                per_frame_class_percentages = patch_results["per_frame_class_percentages"]

                # Dump in multiple formats for flexibility
                patch_csv_path = os.path.join(output_dir, f"patch_scores_iter_{iteration:06d}.csv")
                patch_json_path = os.path.join(output_dir, f"patch_scores_iter_{iteration:06d}.json")

                basic_vis_utils.dump_per_frame_patch_scores(
                    per_frame_class_percentages,
                    patch_csv_path,
                    camera_start,
                    world_start,
                    format="csv",
                    patch_results=patch_results,
                    data_batch=data_batch,
                )

                basic_vis_utils.dump_per_frame_patch_scores(
                    per_frame_class_percentages,
                    patch_json_path,
                    camera_start,
                    world_start,
                    format="json",
                    patch_results=patch_results,
                    data_batch=data_batch,
                )

                # Save pred_class_indices and pred_probabilities as .npy files
                basic_vis_utils.save_patch_predictions(
                    patch_results=patch_results, output_dir=output_dir, iteration=iteration
                )

                # Dump per-frame consistency scores (diagonal of similarity matrix)
                consistency_csv_path = os.path.join(output_dir, f"consistency_scores_iter_{iteration:06d}.csv")
                consistency_json_path = os.path.join(output_dir, f"consistency_scores_iter_{iteration:06d}.json")

                basic_vis_utils.dump_per_frame_consistency_scores(
                    similarity, consistency_csv_path, camera_start, world_start, format="csv", data_batch=data_batch
                )

                basic_vis_utils.dump_per_frame_consistency_scores(
                    similarity, consistency_json_path, camera_start, world_start, format="json", data_batch=data_batch
                )
            elif patch_logits_key is None:
                logger.info("Patch logits not available, skipping patch scores dumping")
            else:
                logger.info("save_patch_data_dumps=False, skipping CSV/JSON/.npy dumps")

            step_time = time.time() - step_start_time
            logger.info("Step 1 (similarity matrix) completed in %.3fs", step_time)

            # Clear memory after similarity matrix
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared after similarity matrix")
        else:
            logger.info("Step 1: Skipping similarity matrix visualization (not available)")

        # 2. Create tiled video visualization
        step_start_time = time.time()
        camera_only = getattr(self.config.inference, "camera_only_mode", False)
        logger.info("Step 2: Creating tiled video visualization...")
        # Get frame offsets from config
        camera_start_frame = getattr(self.config.inference, "camera_start_frame", 0)
        world_start_frame = getattr(self.config.inference, "world_start_frame", 0)

        # Auto-detect GPU availability for video encoding
        use_gpu_encoding = getattr(self.config.inference, "use_gpu_video_encoding", True)
        if use_gpu_encoding and not torch.cuda.is_available():
            use_gpu_encoding = False
            logger.info("CUDA not available, using CPU video encoding")
        elif use_gpu_encoding:
            logger.info("Using GPU (NVENC) video encoding")
        else:
            logger.info("Using CPU video encoding")

        if camera_only or output_batch.get("world_spatial_features_list") is None:
            # No world stream — render the camera-only tiled video (camera | backbone | aligned) instead of the
            # world-paired 2x2/3x2/3x3 layout, which requires world_spatial_features_list.
            logger.info("Step 2: Creating camera-only tiled feature video (no world features)")
            feature_vis_utils.create_camera_only_tiled_video(
                output_batch,
                data_batch,
                iteration,
                output_dir,
                camera_start_frame,
                use_gpu_encoding=use_gpu_encoding,
            )
        else:
            feature_vis_utils.create_tiled_video(
                output_batch,
                data_batch,
                iteration,
                output_dir,
                camera_start_frame,
                world_start_frame,
                output_mode=basic_vis_utils.TilingOutputMode.VIDEO,
                use_gpu_encoding=use_gpu_encoding,
            )
        step_time = time.time() - step_start_time
        logger.info("Step 2 (tiled video) completed in %.3fs", step_time)
        # PCA intermediates and the tiled video buffer can stay cached on the GPU allocator after Step 2; release them
        # so Step 3 / Step 4 don't OOM when they allocate their own large per-class probability tensors.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "[INFERENCE_VIZ] Step 2 done (tiled video, %.1fs). "
            "Next: combined patch visualization if patch classifier is active...",
            step_time,
        )

        # 3. Create combined patch visualization, if patch classifier is used
        if self.patch_classifier is not None:
            # Get the correct logits key
            patch_logits_key = None
            if "patch_similarity_logits" in output_batch and output_batch["patch_similarity_logits"] is not None:
                patch_logits_key = "patch_similarity_logits"
            elif "patch_type_logits" in output_batch and output_batch["patch_type_logits"] is not None:
                patch_logits_key = "patch_type_logits"

            if patch_logits_key is not None:
                step_start_time = time.time()
                logger.info("Step 3: Creating combined patch visualization...")

                # Get activation function from config (default to 'sigmoid' for backward compatibility)
                activation = getattr(self.config.inference, "patch_classifier_activation", "sigmoid")

                # Compute excluded classes for argmax (shared by combined and detailed visualizations)
                excluded_similarity = None
                excluded_type = None
                if self.patch_classifier is not None:
                    if hasattr(self.patch_classifier, "excluded_model_classes_for_argmax_similarity"):
                        excluded_similarity = patch_vis_utils.normalize_excluded_classes_for_argmax(
                            self.patch_classifier.excluded_model_classes_for_argmax_similarity
                        )
                    if hasattr(self.patch_classifier, "excluded_model_classes_for_argmax_type"):
                        excluded_type = patch_vis_utils.normalize_excluded_classes_for_argmax(
                            self.patch_classifier.excluded_model_classes_for_argmax_type
                        )
                    if hasattr(self.patch_classifier, "excluded_model_classes_for_argmax"):
                        excluded_similarity = patch_vis_utils.normalize_excluded_classes_for_argmax(
                            self.patch_classifier.excluded_model_classes_for_argmax
                        )
                object_type_mapping_cls = get_object_type_class_mapping(
                    getattr(self.config, "object_type_class_mapping", None)
                )
                laneline_other_idx = get_model_class_index_for_object_type(
                    "GEOMETRY_LANELINE_OTHER", object_type_mapping_cls
                )
                if laneline_other_idx is not None:
                    excluded_type = list(excluded_type) if excluded_type else []
                    if laneline_other_idx not in excluded_type:
                        excluded_type.append(laneline_other_idx)

                output_batch_with_excluded: dict[str, Any] = dict(output_batch)
                output_batch_with_excluded["excluded_model_classes_for_argmax_similarity"] = excluded_similarity
                output_batch_with_excluded["excluded_model_classes_for_argmax_type"] = excluded_type

                # Compute patch predictions and percentages using the utility function
                gt_patch_similarity = output_batch.get("gt_patch_similarity", None)
                gt_foreground_pixel_counts = output_batch.get("gt_foreground_pixel_counts", None)
                patch_results = metrics_utils.compute_patch_predictions_and_percentages(
                    output_batch[patch_logits_key],
                    treat_wrong_type_as_match=self.patch_classifier.treat_wrong_type_as_match,
                    gt_patch_similarity=gt_patch_similarity,
                    gt_foreground_pixel_counts=gt_foreground_pixel_counts,
                    activation=activation,
                )

                patch_vis_utils.create_combined_patch_visualization(
                    output_batch_with_excluded,
                    data_batch,
                    iteration,
                    output_dir,
                    patch_results,
                    use_gpu_encoding=use_gpu_encoding,
                    object_type_class_mapping=getattr(self.config, "object_type_class_mapping", None),
                    activation=activation,
                    presence_mask=presence_mask,
                    presence_label=presence_label,
                )
                step_time = time.time() - step_start_time
                logger.info("Step 3 (combined patch visualization) completed in %.3fs", step_time)
                # Step 4 (detailed similarity + type vis) is by far the most
                # memory-hungry stage; drop Step 3's intermediates before it runs.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("GPU memory cleared after combined patch visualization")

                # 4. Create detailed patch visualizations (if enabled)
                save_detailed = getattr(self.config.inference, "save_detailed_patch_vis", True)
                if save_detailed:
                    step_start_time = time.time()
                    logger.info("Step 4: Creating detailed patch visualizations (similarity and/or type)...")

                    # Compute patch results with treat_wrong_type_as_match=False for strict mode
                    # (overlay and argmax visualizations will show WRONG and MATCH as separate classes)
                    patch_results_strict = metrics_utils.compute_patch_predictions_and_percentages(
                        output_batch[patch_logits_key],
                        treat_wrong_type_as_match=False,  # Use strict mode for detailed visualization
                        gt_patch_similarity=gt_patch_similarity,
                        gt_foreground_pixel_counts=gt_foreground_pixel_counts,
                        activation=activation,
                        excluded_model_classes_for_argmax=excluded_similarity,
                    )

                    # Get target resolution for detailed visualizations from config
                    target_resolution = getattr(self.config.inference, "detailed_vis_subplot_resolution", (1280, 720))

                    # Get chunk size from config (used for frame batching during visualization)
                    chunk_size = getattr(self.config.inference, "chunk_threshold", 128)

                    patch_vis_utils.create_detailed_patch_visualization(
                        output_batch_with_excluded,
                        data_batch,
                        iteration,
                        output_dir,
                        patch_results_strict,
                        target_resolution=target_resolution,
                        chunk_size=chunk_size,
                        object_type_class_mapping=getattr(self.config, "object_type_class_mapping", None),
                        activation=activation,
                    )
                    step_time = time.time() - step_start_time
                    logger.info("Step 4 (detailed patch visualizations) completed in %.3fs", step_time)

                    # Clear memory after detailed patch visualization
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.info("GPU memory cleared after detailed patch visualization")
                else:
                    logger.info("Step 4: Skipping detailed patch visualizations (save_detailed_patch_vis=False)")

        total_viz_time = time.time() - total_viz_start_time
        logger.info("All visualizations completed in %.3fs (saved to %s)", total_viz_time, _log_safe(output_dir))

    @classmethod
    def from_hf_format(
        cls,
        model_path: str,
        config: ModelConfig | None = None,
        strict: bool = False,
        device: str = "cuda",
    ) -> CWIPInference:
        """
        Load a CWIP model from HuggingFace safetensors format.

        This method loads weights saved with convert_to_hf_format() back into a CWIP model.
        If config is not provided, it will automatically reconstruct a minimal ModelConfig
        from config.json for inference (no pretrained PE checkpoints needed).

        Args:
            model_path: Directory path containing the safetensors file(s) and config.json
            config: ModelConfig to initialize the model architecture. If None, will automatically
                   reconstruct from config.json with minimal inference settings. The reconstructed
                   config has use_pretrained_pe_ckpt=False since all weights are in safetensors.
            strict: Whether to strictly enforce that the keys in state_dict match the model
            device: Device to load the model onto ("cuda", "cpu", or specific device like "cuda:0")

        Returns:
            model: Initialized CWIPInference model with loaded weights

        Example:
            >>> # Simple: Load directly from saved model (auto-reconstructs config from JSON)
            >>> model = CWIPInference.from_hf_format("./my_model")
            >>>
            >>> # Advanced: Provide custom config if needed
            >>> from checks.utils.models.cwip.modules.configs.defaults.model import ModelConfig
            >>> config = ModelConfig(...)
            >>> model = CWIPInference.from_hf_format("./my_model", config)
            >>>
            >>> # Load to CPU for inference
            >>> model = CWIPInference.from_hf_format("./my_model", device="cpu")
        """
        logger.info("Loading CWIP model from HuggingFace format at %s", _log_safe(model_path))

        # Load and parse saved config
        config_path = os.path.join(model_path, "config.json")
        saved_config_dict = None
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                saved_config_dict = json.load(f)
            logger.info("Found saved config.json")

        # Handle config parameter - reconstruct from JSON if not provided
        if config is None:
            if saved_config_dict is not None:
                logger.info("Reconstructing ModelConfig from saved config.json for inference...")
                config = cls._reconstruct_config_from_dict(saved_config_dict)
                logger.info("Successfully reconstructed ModelConfig from saved config.json")
            else:
                raise ValueError(
                    f"No config provided and config.json not found at {model_path}. "
                    f"Please either provide a ModelConfig or ensure config.json exists."
                )

        # Check if we have a single file or sharded files
        single_file = os.path.join(model_path, "model.safetensors")
        index_file = os.path.join(model_path, "model.safetensors.index.json")

        if os.path.exists(single_file):
            # Single file model
            logger.info("Loading single safetensors file: %s", _log_safe(single_file))
            state_dict = load_file(single_file, device=device)
        elif os.path.exists(index_file):
            # Sharded model - load all shards
            logger.info("Loading sharded model using index: %s", _log_safe(index_file))
            with open(index_file, "r") as f:
                index = json.load(f)

            # Get unique shard filenames
            shard_files = sorted(set(index["weight_map"].values()))
            logger.info("Found %s shards to load", len(shard_files))

            # Load all shards into a single state dict
            state_dict = {}
            model_dir = os.path.normpath(model_path)
            for shard_file in shard_files:
                # ``weight_map`` ships inside the checkpoint, so its shard names are untrusted: an absolute name
                # silently discards ``model_dir`` in ``join``, and ``..`` segments walk out of it. The containment
                # test is deliberately lexical rather than ``realpath``-based -- Hub checkpoints are symlink farms
                # pointing into the HF cache's sibling ``blobs/`` directory, which resolving would reject.
                shard_path = os.path.normpath(os.path.join(model_dir, shard_file))
                if not shard_path.startswith(model_dir + os.sep):
                    raise ValueError(
                        f"Shard {shard_file!r} listed in {index_file} escapes the model directory; refusing to load."
                    )
                logger.info("Loading shard: %s", _log_safe(shard_file))
                shard_state_dict = load_file(shard_path, device=device)
                state_dict.update(shard_state_dict)
        else:
            raise FileNotFoundError(
                f"Could not find model files at {model_path}. Expected either {single_file} or {index_file}"
            )

        # Log loaded state dict info
        dtypes = {v.dtype for v in state_dict.values()}
        total_params = sum(v.numel() for v in state_dict.values())
        logger.info("Loaded state dict with %s keys, %s parameters", len(state_dict), f"{total_params:,}")
        logger.info("Loaded dtypes: %s", dtypes)

        state_dict, config = cls._prepare_for_loading(state_dict, config)

        logger.info("Initializing CWIP inference model...")
        assert config is not None, "Config must be set before model initialization"
        model = cls(config=config)

        # Shard checkpoint weights for Tensor Parallelism if needed
        tp_size = parallelism_module.get_tensor_parallel_world_size()
        if tp_size > 1:
            tp_rank = parallelism_module.get_tensor_parallel_rank()
            state_dict = cls._shard_state_dict_for_tp(model, state_dict, tp_size, tp_rank)

        logger.info("Loading state dict into model...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

        if missing_keys:
            logger.warning("Missing keys when loading state dict: %s", _log_safe(missing_keys))
        if unexpected_keys:
            logger.warning("Unexpected keys when loading state dict: %s", _log_safe(unexpected_keys))

        if not missing_keys and not unexpected_keys:
            logger.info("Successfully loaded all weights!")

        if device != "cpu" and not next(model.parameters()).is_cuda:
            model = model.to(device)
            logger.info("Moved model to %s", _log_safe(device))

        logger.info("Model loaded successfully from %s", _log_safe(model_path))
        return model

    @classmethod
    def _prepare_for_loading(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: ModelConfig | None,
    ) -> tuple[dict[str, torch.Tensor], ModelConfig | None]:
        """Normalize legacy checkpoint formats and apply runtime parallelism overrides.

        1. State dict normalization:
           - Patch classifier: kv_proj_weight/in_proj_weight → separate q_proj/k_proj/v_proj
           - Encoders: separate q_proj/k_proj/v_proj → fused in_proj_weight
        2. Config overrides:
           - Sets use_legacy_transformer=False (state dict is now in canonical format)
           - Overrides use_tensor_parallel on camera_encoder, world_encoder, and
             patch_classifier from runtime TP (``--tp_size``), so inference matches
             the initialized process groups. Hybrid PE-TP + non-TP PatchFormer is not
             supported.
           - Context parallelism (FP/true CP) is handled entirely by the
             parallelism module at runtime, not by config flags.
        """
        state_dict = cls._normalize_state_dict(state_dict)

        if config is not None:
            # Override TP flag based on runtime parallelism state. The config records what the checkpoint was saved
            # with, but the user may request different TP at load time via CLI args.
            tp_size = parallelism_module.get_tensor_parallel_world_size()
            use_tp = tp_size > 1

            for section_name in ("camera_encoder", "world_encoder", "patch_classifier"):
                section = getattr(config, section_name, None)
                if section is None:
                    continue
                cls._set_config_field(section, "use_tensor_parallel", use_tp)

            if hasattr(config, "patch_classifier") and config.patch_classifier is not None:
                cls._set_config_field(config.patch_classifier, "use_legacy_transformer", False)

        return state_dict, config

    @staticmethod
    def _set_config_field(section: object, field: str, value: object) -> None:
        """Set a field on a config section (supports both attr and dict access)."""
        try:
            if hasattr(section, field):
                setattr(section, field, value)
            elif isinstance(section, dict):
                section[field] = value
        except Exception as exc:  # noqa: BLE001 - config backend raises its own error types; best-effort set
            logger.debug("Ignoring error setting config field %s: %s", _log_safe(field), _log_safe(exc))

    @staticmethod
    def _is_legacy_patchformer_checkpoint(state_dict: dict[str, torch.Tensor]) -> bool:
        """Detect if patch_classifier uses legacy kv_proj/in_proj format."""
        for key in state_dict:
            if "patch_classifier" in key:
                if "kv_proj_weight" in key or "kv_proj_bias" in key:
                    return True
                if "in_proj_weight" in key or "in_proj_bias" in key:
                    return True
                if "q_proj_weight" in key or "q_proj_bias" in key:
                    layer_key = key.replace("q_proj_weight", "").replace("q_proj_bias", "")
                    has_k = any("k_proj" in k and layer_key in k for k in state_dict)
                    has_v = any("v_proj" in k and layer_key in k for k in state_dict)
                    if not (has_k and has_v):
                        return True
        return False

    @staticmethod
    def _convert_legacy_kv_proj_to_separate(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert legacy patch_classifier kv_proj/in_proj → separate q/k/v."""
        new_state_dict = {}
        converted_count = 0

        for key, value in state_dict.items():
            if "patch_classifier" in key and "in_proj_weight" in key:
                embed_dim = value.size(0) // 3
                base_key = key.replace("in_proj_weight", "")
                new_state_dict[f"{base_key}q_proj.weight"] = value[:embed_dim, :]
                new_state_dict[f"{base_key}k_proj.weight"] = value[embed_dim : 2 * embed_dim, :]
                new_state_dict[f"{base_key}v_proj.weight"] = value[2 * embed_dim :, :]
                converted_count += 1

            elif "patch_classifier" in key and "in_proj_bias" in key:
                embed_dim = value.size(0) // 3
                base_key = key.replace("in_proj_bias", "")
                new_state_dict[f"{base_key}q_proj.bias"] = value[:embed_dim]
                new_state_dict[f"{base_key}k_proj.bias"] = value[embed_dim : 2 * embed_dim]
                new_state_dict[f"{base_key}v_proj.bias"] = value[2 * embed_dim :]
                converted_count += 1

            elif "patch_classifier" in key and "kv_proj_weight" in key:
                embed_dim = value.size(0) // 2
                new_state_dict[key.replace("kv_proj_weight", "k_proj.weight")] = value[:embed_dim, :]
                new_state_dict[key.replace("kv_proj_weight", "v_proj.weight")] = value[embed_dim:, :]
                converted_count += 1

            elif "patch_classifier" in key and "kv_proj_bias" in key:
                embed_dim = value.size(0) // 2
                new_state_dict[key.replace("kv_proj_bias", "k_proj.bias")] = value[:embed_dim]
                new_state_dict[key.replace("kv_proj_bias", "v_proj.bias")] = value[embed_dim:]
                converted_count += 1

            elif "patch_classifier" in key and ("q_proj_weight" in key or "q_proj_bias" in key):
                if "q_proj_weight" in key:
                    new_state_dict[key.replace("q_proj_weight", "q_proj.weight")] = value
                else:
                    new_state_dict[key.replace("q_proj_bias", "q_proj.bias")] = value

            else:
                new_state_dict[key] = value

        if converted_count > 0:
            logger.info("Converted %s legacy patch_classifier parameters to separate q/k/v format", converted_count)
        return new_state_dict

    @staticmethod
    def _convert_separate_to_in_proj_for_encoders(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert encoder separate q/k/v → fused in_proj (fixes buggy checkpoints)."""
        new_state_dict = {}
        converted_count = 0
        processed_keys: set[str] = set()

        for key, value in state_dict.items():
            if ("camera" in key or "world" in key) and "encoder" in key:
                if ".q_proj.weight" in key:
                    base_key = key.replace(".q_proj.weight", "")
                    k_key = base_key + ".k_proj.weight"
                    v_key = base_key + ".v_proj.weight"
                    if k_key in state_dict and v_key in state_dict:
                        new_state_dict[base_key + ".in_proj_weight"] = torch.cat(
                            [value, state_dict[k_key], state_dict[v_key]], dim=0
                        )
                        processed_keys.update({k_key, v_key})
                        converted_count += 1
                        continue

                elif ".q_proj.bias" in key:
                    base_key = key.replace(".q_proj.bias", "")
                    k_key = base_key + ".k_proj.bias"
                    v_key = base_key + ".v_proj.bias"
                    if k_key in state_dict and v_key in state_dict:
                        new_state_dict[base_key + ".in_proj_bias"] = torch.cat(
                            [value, state_dict[k_key], state_dict[v_key]], dim=0
                        )
                        processed_keys.update({k_key, v_key})
                        converted_count += 1
                        continue

            if key in processed_keys:
                continue
            new_state_dict[key] = value

        if converted_count > 0:
            logger.info("Converted %s encoder parameters from separate q/k/v to in_proj format", converted_count)
        return new_state_dict

    @classmethod
    def _normalize_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Normalize state dict to canonical format for current model architecture."""
        if cls._is_legacy_patchformer_checkpoint(state_dict):
            logger.info("Detected legacy patch_classifier format, converting...")
            state_dict = cls._convert_legacy_kv_proj_to_separate(state_dict)

        has_encoder_qkv = any(
            ("camera" in k or "world" in k) and "encoder" in k and ".q_proj.weight" in k for k in state_dict
        )
        if has_encoder_qkv:
            logger.info("Detected separate q/k/v for encoders, converting to in_proj...")
            state_dict = cls._convert_separate_to_in_proj_for_encoders(state_dict)

        return state_dict

    @classmethod
    def _shard_state_dict_for_tp(
        cls,
        model: nn.Module,
        state_dict: dict[str, torch.Tensor],
        tp_size: int,
        tp_rank: int,
    ) -> dict[str, torch.Tensor]:
        """Shard a full (unsharded) checkpoint state dict for the current TP rank.

        The HF checkpoint stores full weights. When the model is created with TP>1,
        parameters are already TP-sharded (smaller). This method slices each checkpoint
        tensor to match the model's expected shape for this rank.

        Sharding rules (standard Megatron-LM column/row parallel convention):
        - 2-D weight, checkpoint dim 0 > model dim 0: column-parallel → slice dim 0
        - 2-D weight, checkpoint dim 1 > model dim 1: row-parallel   → slice dim 1
        - 1-D bias,   checkpoint dim 0 > model dim 0: column-parallel bias → slice dim 0
        - Matching shapes: pass through unchanged

        Special handling for fused QKV projections (in_proj_weight / in_proj_bias):
        The weight layout is [Q | K | V] stacked along dim 0. A naive contiguous
        slice would scramble Q/K/V across ranks. Instead, we split into Q, K, V,
        shard each independently, and recombine.
        """
        model_sd = model.state_dict()
        sharded: dict[str, torch.Tensor] = {}
        num_sharded = 0

        for key, ckpt_val in state_dict.items():
            if key not in model_sd:
                sharded[key] = ckpt_val
                continue

            expected_shape = model_sd[key].shape
            ckpt_shape = ckpt_val.shape

            if ckpt_shape == expected_shape:
                sharded[key] = ckpt_val
            elif cls._is_fused_qkv_key(key):
                sharded[key] = parallelism_module.shard_fused_qkv(ckpt_val, tp_size, tp_rank)
                num_sharded += 1
            elif len(ckpt_shape) == 2:
                if ckpt_shape[0] != expected_shape[0] and ckpt_shape[1] == expected_shape[1]:
                    # Column parallel: shard output dimension (dim 0)
                    chunk = expected_shape[0]
                    sharded[key] = ckpt_val[tp_rank * chunk : (tp_rank + 1) * chunk]
                    num_sharded += 1
                elif ckpt_shape[0] == expected_shape[0] and ckpt_shape[1] != expected_shape[1]:
                    # Row parallel: shard input dimension (dim 1)
                    chunk = expected_shape[1]
                    sharded[key] = ckpt_val[:, tp_rank * chunk : (tp_rank + 1) * chunk]
                    num_sharded += 1
                else:
                    raise ValueError(
                        f"Cannot determine TP sharding for '{key}': "
                        f"checkpoint {list(ckpt_shape)} vs model {list(expected_shape)}"
                    )
            elif len(ckpt_shape) == 1 and ckpt_shape[0] != expected_shape[0]:
                if cls._is_fused_qkv_key(key):
                    sharded[key] = parallelism_module.shard_fused_qkv(ckpt_val, tp_size, tp_rank)
                else:
                    # Bias for column-parallel layer
                    chunk = expected_shape[0]
                    sharded[key] = ckpt_val[tp_rank * chunk : (tp_rank + 1) * chunk]
                num_sharded += 1
            else:
                sharded[key] = ckpt_val

        logger.info(
            "[TP rank %s] Sharded %s/%s checkpoint tensors for TP=%s",
            tp_rank,
            num_sharded,
            len(state_dict),
            tp_size,
        )
        return sharded

    @staticmethod
    def _is_fused_qkv_key(key: str) -> bool:
        """Check if a state dict key is a fused QKV projection (in_proj_weight/bias)."""
        return key.endswith(("in_proj_weight", "in_proj_bias"))

    @staticmethod
    def _parse_encoder_config(encoder_info: dict) -> L:
        """
        Parse encoder configuration from saved config dict.

        Args:
            encoder_info: Dictionary containing encoder configuration

        Returns:
            LazyCall for ParallelImageEncoder
        """
        return L(pe_module.ParallelImageEncoder)(
            encoder_cfg_name=encoder_info.get("encoder_cfg_name", "PE-Spatial-B16-512"),
            num_alignment_layers=encoder_info.get("num_alignment_layers", 0),
            output_dim=encoder_info.get("output_dim", 1024),
            use_tensor_parallel=encoder_info.get("use_tensor_parallel", False),
        )

    @staticmethod
    def _parse_classifier_dims(
        patch_classifier_info: dict, camera_info: dict, world_info: dict, split_in_channels: bool = False
    ) -> tuple[int | None, int | None]:
        """
        Parse camera_dim and world_dim from classifier config, supporting old format.

        Args:
            patch_classifier_info: Dictionary containing patch classifier configuration
            camera_info: Dictionary containing camera encoder configuration
            world_info: Dictionary containing world encoder configuration
            split_in_channels: If True, split ``in_channels`` equally between camera/world; if False, use
                full width for both

        Returns:
            Tuple of (camera_dim, world_dim)
        """
        camera_dim = patch_classifier_info.get("camera_dim")
        world_dim = patch_classifier_info.get("world_dim")

        # If old format (in_channels), compute camera_dim and world_dim
        if camera_dim is None or world_dim is None:
            in_channels = patch_classifier_info.get("in_channels")
            if in_channels:
                if split_in_channels:
                    # Equal split of fused in_channels into camera vs world
                    camera_dim = in_channels // 2
                    world_dim = in_channels // 2
                else:
                    # Full fused width for both paths
                    camera_dim = in_channels
                    world_dim = in_channels
            else:
                # Use output_dim from encoders as default
                camera_dim = camera_info.get("output_dim", 1024)
                world_dim = world_info.get("output_dim", 1024)

        return camera_dim, world_dim

    @staticmethod
    def _parse_patch_classifier_config(
        patch_classifier_info: dict,
        camera_info: dict,
        world_info: dict,
        split_in_channels: bool = False,
        object_type_class_mapping: str | None = None,
    ) -> L | None:
        """
        Parse patch classifier configuration from saved config dict.
        Supports both minimal and full saved-config parameter sets.

        Args:
            patch_classifier_info: Dictionary containing patch classifier configuration
            camera_info: Dictionary containing camera encoder configuration
            world_info: Dictionary containing world encoder configuration
            split_in_channels: If True, apply the split-in-channels rule for legacy ``in_channels`` format

        Returns:
            LazyCall for ParallelPatchFormer or None if not configured
        """
        if not patch_classifier_info:
            return None

        classifier_type = patch_classifier_info.get("type", "ParallelPatchFormer")

        # Parse dimensions
        camera_dim, world_dim = CWIPInference._parse_classifier_dims(
            patch_classifier_info, camera_info, world_info, split_in_channels
        )

        # Support both new and old parameter names
        hidden_dim = patch_classifier_info.get("hidden_dim", patch_classifier_info.get("hidden_channels", 256))
        num_layers = patch_classifier_info.get("num_layers", patch_classifier_info.get("layers", 2))

        # Convert object_type_class_mapping string to class if needed
        object_type_class_mapping_cls = None
        if object_type_class_mapping is not None:
            try:
                object_type_class_mapping_cls = get_object_type_class_mapping(object_type_class_mapping)
            except ValueError as e:
                logger.warning(
                    "Could not resolve object_type_class_mapping=%s: %s",
                    _log_safe(object_type_class_mapping),
                    _log_safe(e),
                )

        # Get num_classes from config if provided
        # For PatchFormer, num_classes should be a list [type_num_classes, similarity_num_classes] If not provided,
        # ParallelPatchFormer will derive it from object_type_class_mapping
        num_classes = patch_classifier_info.get("num_classes")

        # Build classifier based on type
        # Both "PatchFormer" and "ParallelPatchFormer" use ParallelPatchFormer (with TP/CP disabled for inference)
        if classifier_type in ("PatchFormer", "ParallelPatchFormer"):
            # PatchFormer: transformer-based classifier
            # Use ParallelPatchFormer with TP/CP disabled for inference
            # Detect legacy format from saved config, but default to False (new format) The actual format will be
            # detected from state dict keys when loading weights
            use_legacy_transformer = patch_classifier_info.get("use_legacy_transformer")
            if use_legacy_transformer is None:
                # Default to False (new format) - most checkpoints after Jan 2026 use new format
                use_legacy_transformer = False
            return L(ParallelPatchFormer)(
                camera_dim=camera_dim,
                world_dim=world_dim,
                num_classes=num_classes,
                object_type_class_mapping=object_type_class_mapping_cls,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=patch_classifier_info.get("num_heads", 12),
                mlp_ratio=patch_classifier_info.get("mlp_ratio", 4.0),
                cross_attn_frequency=patch_classifier_info.get("cross_attn_frequency", 1),
                ls_init_value=patch_classifier_info.get("ls_init_value"),
                drop_path=patch_classifier_info.get("drop_path", 0.0),
                use_abs_posemb=patch_classifier_info.get("use_abs_posemb", False),
                use_rope2d=patch_classifier_info.get("use_rope2d", True),
                input_resize_factor=patch_classifier_info.get("input_resize_factor"),
                treat_wrong_type_as_match=patch_classifier_info.get("treat_wrong_type_as_match", False),
                use_tensor_parallel=patch_classifier_info.get("use_tensor_parallel", False),
                use_legacy_transformer=use_legacy_transformer,
                # Pass through additional PatchFormer-specific params
                per_class_temperature_similarity=patch_classifier_info.get("per_class_temperature_similarity"),
                per_class_bias_similarity=patch_classifier_info.get("per_class_bias_similarity"),
                per_class_temperature_type=patch_classifier_info.get("per_class_temperature_type"),
                per_class_bias_type=patch_classifier_info.get("per_class_bias_type"),
                excluded_model_classes_for_argmax_similarity=patch_classifier_info.get(
                    "excluded_model_classes_for_argmax_similarity"
                ),
                excluded_model_classes_for_argmax_type=patch_classifier_info.get(
                    "excluded_model_classes_for_argmax_type"
                ),
                use_deformable_attn=patch_classifier_info.get("use_deformable_attn", False),
                # Independent of ``use_deformable_attn``: gates the dedicated type-side ``deformable_attn_block_type`` +
                # ``deformable_norm_type`` (only meaningful with ``use_separate_necks=True``). Default ``False`` keeps
                # legacy checkpoints (which don't write this key) loading unchanged.
                use_deformable_attn_type=patch_classifier_info.get("use_deformable_attn_type", False),
                deformable_num_sampling_points=patch_classifier_info.get("deformable_num_sampling_points", 4),
                deformable_query_upsample_method=patch_classifier_info.get(
                    "deformable_query_upsample_method", "bilinear"
                ),
                deformable_query_upsample_kernel_size=patch_classifier_info.get(
                    "deformable_query_upsample_kernel_size", 3
                ),
                patch_classification_type=patch_classifier_info.get("patch_classification_type"),
                # Neck-topology fields: these change parameter naming so they MUST be restored from saved config or
                # load_state_dict will mismatch.
                use_separate_necks=patch_classifier_info.get("use_separate_necks", False),
                neck_type_config=patch_classifier_info.get("neck_type_config"),
                neck_similarity_config=patch_classifier_info.get("neck_similarity_config"),
                fused_kv_camera_world_cross_attn=patch_classifier_info.get("fused_kv_camera_world_cross_attn", False),
                # Per-branch input projections (``camera_proj_{type,similarity}`` / ``world_proj_{type,similarity}``).
                # Parameter-bearing -- mismatch surfaces as state_dict key mismatches at load time.
                use_separate_projs=patch_classifier_info.get("use_separate_projs", False),
                # Forward-path-only knob (see save side in ``convert_cwip_to_hf.py``). Default ``False`` matches the
                # neck-then-bilinear-upsample default for non-deformable configs; explicit ``True`` in ``config.json``
                # restores the legacy pre-upscale-then-neck path.
                pre_upscale_when_deformable_off=patch_classifier_info.get("pre_upscale_when_deformable_off", False),
            )
        else:
            raise ValueError(
                f"Unknown patch classifier type: {classifier_type!r}. Expected 'PatchFormer' or 'ParallelPatchFormer'."
            )

    @staticmethod
    def _reconstruct_config_from_dict(saved_config_dict: dict) -> ModelConfig:
        """
        Reconstruct a minimal ModelConfig from saved config.json for inference.

        This creates a config with just enough information to initialize the model architecture.
        All weights will be loaded from safetensors, so we don't need pretrained checkpoints.

        Args:
            saved_config_dict: Dictionary loaded from config.json

        Returns:
            ModelConfig: Minimal config for inference
        """

        # Extract precision string (e.g., "torch.bfloat16" -> "bfloat16")
        precision_str = saved_config_dict.get("precision", "bfloat16")
        if "." in precision_str:
            precision_str = precision_str.split(".")[-1]  # "torch.bfloat16" -> "bfloat16"

        # Create encoder configs using helper
        camera_info = saved_config_dict.get("camera_encoder", {})
        camera_encoder = CWIPInference._parse_encoder_config(camera_info)

        world_info = saved_config_dict.get("world_encoder", {})
        world_encoder = CWIPInference._parse_encoder_config(world_info)

        # Create patch classifier if it was present using helper
        patch_classifier = None
        if saved_config_dict.get("patch_classifier"):
            object_type_class_mapping = saved_config_dict.get("object_type_class_mapping")
            patch_classifier = CWIPInference._parse_patch_classifier_config(
                saved_config_dict["patch_classifier"],
                camera_info,
                world_info,
                split_in_channels=False,  # Deployment uses symmetric dims
                object_type_class_mapping=object_type_class_mapping,
            )

        # Create inference config
        inference_info = saved_config_dict.get("inference", {})
        inference_config = InferenceConfig(  # type: ignore[call-arg]
            mode=inference_info.get("mode", False),
            output_dir=inference_info.get("output_dir", None),
            chunk_threshold=inference_info.get("chunk_threshold", 128),
            blacken_input_world_frames=inference_info.get("blacken_input_world_frames", False),
            camera_only_mode=inference_info.get("camera_only_mode", False),
        )

        # Create ModelConfig (image_size will be initialized from encoder and updated dynamically)
        config = ModelConfig(  # type: ignore[call-arg]  # attrs.define init not resolved by mypy in build sandbox
            camera_encoder=camera_encoder,
            world_encoder=world_encoder,
            patch_classifier=patch_classifier,
            precision=precision_str,
            checkpoint_path=saved_config_dict.get("checkpoint_path", None),
            inference=inference_config,
            object_type_class_mapping=saved_config_dict.get("object_type_class_mapping", None),
        )

        return config
