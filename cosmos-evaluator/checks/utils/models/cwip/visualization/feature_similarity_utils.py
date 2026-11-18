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

import torch
import torch.nn.functional as F


def compute_cosine_similarity(
    features1: torch.Tensor, features2: torch.Tensor, dim: int = 1, eps: float = 1e-8, keepdim: bool = True
) -> torch.Tensor:
    """
    Compute cosine similarity between two feature tensors.

    Supports arbitrary tensor dimensions including:
    - Global features: (B, D)
    - Spatial features: (B, D, H, W)
    - Video/temporal features: (B, D, T, H, W) or (B, T, D)
    - Single feature vectors: (D,)
    - Any other tensor shapes where features are aligned along a specific dimension

    Args:
        features1: First feature tensor of any shape
        features2: Second feature tensor with matching shape
        dim: Dimension along which to compute cosine similarity (default: 1 for channel dimension in (B, C, ...) format)
        eps: Small epsilon value for numerical stability (default: 1e-8)
        keepdim: Whether to keep the dimension along which similarity is computed (default: True)

    Returns:
        Cosine similarity tensor with similarity computed along the specified dimension:
            - If keepdim=True: dimension at 'dim' becomes size 1
            - If keepdim=False: dimension at 'dim' is removed

        Examples:
            - (B, D) with dim=1, keepdim=True -> (B, 1)
            - (B, D) with dim=1, keepdim=False -> (B,)
            - (B, D, H, W) with dim=1, keepdim=True -> (B, 1, H, W)
            - (B, D, H, W) with dim=1, keepdim=False -> (B, H, W)
            - (B, T, D) with dim=2, keepdim=True -> (B, T, 1)
            - (B, D, T, H, W) with dim=1, keepdim=True -> (B, 1, T, H, W)

    Examples:
        >>> # Global features (B, D)
        >>> feat1 = torch.randn(4, 256)
        >>> feat2 = torch.randn(4, 256)
        >>> sim = compute_cosine_similarity(feat1, feat2)  # (4, 1)
        >>> sim_no_keepdim = compute_cosine_similarity(feat1, feat2, keepdim=False)  # (4,)

        >>> # Spatial features (B, D, H, W)
        >>> feat1 = torch.randn(4, 256, 16, 16)
        >>> feat2 = torch.randn(4, 256, 16, 16)
        >>> sim = compute_cosine_similarity(feat1, feat2)  # (4, 1, 16, 16)
        >>> sim_no_keepdim = compute_cosine_similarity(feat1, feat2, keepdim=False)  # (4, 16, 16)

        >>> # Video/temporal features (B, D, T, H, W)
        >>> feat1 = torch.randn(2, 128, 8, 16, 16)
        >>> feat2 = torch.randn(2, 128, 8, 16, 16)
        >>> sim = compute_cosine_similarity(feat1, feat2)  # (2, 1, 8, 16, 16)

        >>> # Sequential features (B, T, D)
        >>> feat1 = torch.randn(4, 10, 512)
        >>> feat2 = torch.randn(4, 10, 512)
        >>> sim = compute_cosine_similarity(feat1, feat2, dim=2)  # (4, 10, 1)

        >>> # Single feature vectors (D,)
        >>> feat1 = torch.randn(256)
        >>> feat2 = torch.randn(256)
        >>> sim = compute_cosine_similarity(feat1, feat2, dim=0)  # (1,) if keepdim=True, () if keepdim=False
    """
    # Validate input shapes match
    if features1.shape != features2.shape:
        raise ValueError(f"Feature shapes must match: got {features1.shape} and {features2.shape}")

    # Validate dim is within valid range
    if dim < 0:
        dim = len(features1.shape) + dim  # Convert negative index to positive

    if dim >= len(features1.shape):
        raise ValueError(f"dim={dim} is out of range for tensor with {len(features1.shape)} dimensions")

    # Normalize features along the specified dimension
    features1_norm = F.normalize(features1, p=2, dim=dim, eps=eps)
    features2_norm = F.normalize(features2, p=2, dim=dim, eps=eps)

    # Compute cosine similarity: element-wise product and sum along specified dim
    cosine_sim = (features1_norm * features2_norm).sum(dim=dim, keepdim=keepdim)

    return cosine_sim
