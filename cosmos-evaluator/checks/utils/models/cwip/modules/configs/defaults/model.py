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

from __future__ import annotations

import attrs
from omegaconf import DictConfig as LazyDict


@attrs.define(slots=False)
class InferenceConfig:
    """Configuration for inference-specific settings."""

    # Whether the model is in inference mode. When True, applies patch logit
    # calibration (bias/temperature) and inference-oriented outputs where configured.
    mode: bool = False
    # Directory to save inference outputs (tensors, visualizations).
    # If None, defaults to '/tmp/inference_output'
    output_dir: str | None = None
    # Starting iteration for inference/validation resume functionality (default: None for auto-resume)
    start_iteration: int | None = None
    # Chunk processing threshold for inference visualization / long-sequence handling
    # Controls: (1) Model forward pass chunking for long videos, (2) Visualization frame batch size
    # Default 128 is conservative; use 256+ for cluster GPUs with 80GB memory for faster processing
    chunk_threshold: int = 128
    # Starting frame offset for camera video during inference (0-based indexing)
    camera_start_frame: int = 0
    # Starting frame offset for world video during inference (0-based indexing)
    world_start_frame: int = 0
    # Whether to save detailed patch visualization videos (detailed_patch_similarity_vis and detailed_patch_type_vis)
    # Set to False to save time and disk space if only combined visualization is needed
    save_detailed_patch_vis: bool = True
    # Whether to use GPU hardware video encoding (NVIDIA NVENC) if available (default: True)
    # Provides 5-10x faster video encoding compared to CPU-based encoding
    # Automatically falls back to CPU encoding if NVENC is not available
    use_gpu_video_encoding: bool = True
    # Resolution for individual subplots in detailed patch visualizations (width, height)
    # Default 640x360 for 5x6 layouts fits within NVENC hardware limits (3840x1800 total < 4096x2160 max)
    # Using 1280x720 would produce 7680x3600 total which exceeds NVENC limits and forces CPU fallback
    # Smaller resolutions enable GPU-accelerated encoding (10x faster) instead of CPU fallback
    detailed_vis_subplot_resolution: tuple[int, int] = (640, 360)
    # Activation function for converting patch classifier logits to probabilities in visualization and metrics
    # Options: 'sigmoid' (multi-label), 'softmax' (single-label/mutually exclusive)
    # Should match how logits were calibrated when the checkpoint was produced
    patch_classifier_activation: str = "sigmoid"
    # If True, replace input world frames with all black (for ablation / camera-only evaluation)
    blacken_input_world_frames: bool = False
    # If True, load the full two-branch checkpoint but skip world encoding and the
    # similarity head at forward time. Only the camera encoder/aligner, camera global
    # embedding, and object-type prediction path run. Unlike ``blacken_input_world_frames``,
    # the world branch is not executed at all (no world FLOPs / activations).
    camera_only_mode: bool = False


@attrs.define(slots=False)
class ModelConfig:
    """
    Config for model.

    See this file for configuration examples.
    """

    camera_encoder: LazyDict  # LazyCall for camera encoder
    world_encoder: LazyDict | None = None  # LazyCall for world encoder
    patch_classifier: LazyDict | None = None  # LazyCall for patch classifier
    # Model compute precision (e.g. bfloat16 for inference)
    # Options: "float32", "float16", "bfloat16"
    precision: str = "bfloat16"
    # File path to load the checkpoint for the whole model from,
    # It overrides pretrained PE checkpoints.
    checkpoint_path: str | None = None
    # Inference configuration
    inference: InferenceConfig = attrs.Factory(InferenceConfig)
    # Object type class mapping for GT generation and type classification
    # Controls how ObjectType enums map to model class indices
    # Use string identifier ("v3") - resolved to class by get_object_type_class_mapping()
    object_type_class_mapping: str | None = None
    # When True, the encoder forward pass returns the **pre-alignment** (frozen-backbone) feature in addition to the
    # alignment-final feature for every stream that has ``num_alignment_layers > 0``. This populates row 2 of the
    # debug 3x3 tiled feature visualization (camera/world backbone features). Costs zero extra compute (the
    # pre-alignment activation is computed unconditionally; this only retains a reference to it) and one extra
    # ``(B, D, H', W')`` tensor of memory per stream (small, but during training it stays in the autograd graph).
    # Off by default to preserve existing memory profiles; flip on when you want backbone tiles in viz for
    # training, validation, or deployment-time inference. Independent of ``use_mutual_feature_map_loss`` — the
    # mutual-loss path always retains both layers regardless of this flag.
    viz_capture_backbone_features: bool = False
