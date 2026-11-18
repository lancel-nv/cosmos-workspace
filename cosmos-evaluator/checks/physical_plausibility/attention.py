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

"""Portable attention + linear heads; logits internally, probability scores at the API.
The frozen token contract and state keys match the verified FineVideo attention + linear bundle.
Normalization is fitted only for new heads and remains fixed during fine-tuning.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

SCHEMA = "checker-attention-bundle-v1"
CONTRACT = {
    "schema": "checker-token-space-v1",
    "backbone": "vjepa21",
    "checkpoint_sha256": "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6",
    "source_commit": "204698b45b3712590f06245fbfba32d3be539812",
    "sampling": "uniform-duration-64-v1",
    "spatial": "letterbox384",
    "normalization": "imagenet; normalized-zero padding",
    "encoder": "bf16-autocast-fp32-weights",
    "storage": "float16",
    "shape": [32, 24, 24, 1024],
    "padding": "remove-zero-validity-only; retain-fractional-weights",
}


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def contract_fingerprint() -> str:
    return digest(CONTRACT)


def atomic_save(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class AttentionHead(nn.Module):
    mean: torch.Tensor
    scale: torch.Tensor

    def __init__(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean.float().clone())
        self.register_buffer("scale", scale.float().clone())
        self.query = nn.Parameter(torch.zeros(len(mean)))
        self.head = nn.Linear(len(mean), 1)

    def pool(self, tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        z = (tokens.float() - self.mean) / self.scale
        attention = (z @ self.query / len(self.mean) ** 0.5 + weights.log()).softmax(-1)
        return (attention[..., None] * z).sum(1)

    def forward(self, tokens: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(tokens, weights)).squeeze(-1)

    def regularization(self, head_l2: float, query_l2: float) -> torch.Tensor:
        return (head_l2 * self.head.weight.square().sum() + query_l2 * self.query.square().sum()) / 2


def validate_bundle(bundle: dict[str, Any]) -> None:
    if not isinstance(bundle, dict) or bundle.get("schema") != SCHEMA:
        raise ValueError("expected a portable attention + linear bundle")
    if bundle.get("contract") != CONTRACT or bundle.get("score_convention") != "higher-is-bad; label1-good":
        raise ValueError("attention feature contract or score convention mismatch")
    config = bundle.get("config", {})
    if config.get("head") != "linear" or config.get("pooling") != "attention":
        raise ValueError("expected attention + linear architecture")
    expected = {
        "mean": (1024,),
        "scale": (1024,),
        "query": (1024,),
        "head.weight": (1, 1024),
        "head.bias": (1,),
    }
    state = bundle.get("state")
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("invalid attention state keys")
    for name, shape in expected.items():
        tensor = state[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != shape
            or tensor.dtype != torch.float32
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"invalid attention tensor {name}")
    if (state["scale"] <= 0).any():
        raise ValueError("normalization scale must be positive")
    cuts = bundle.get("thresholds")
    if not isinstance(cuts, dict) or not cuts:
        raise ValueError("missing attention thresholds")
    for key, value in cuts.items():
        if (
            not isinstance(key, str)
            or (key != "val_f1" and not key.startswith("val_good_rejection_cap_"))
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError("invalid attention thresholds")
        if key != "val_f1" and not 0 <= float(key.removeprefix("val_good_rejection_cap_")) < 1:
            raise ValueError("invalid rejection cap")
    hashes = bundle.get("training_and_selection_video_sha256")
    if not isinstance(hashes, list) or any(
        not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes
    ):
        raise ValueError("invalid training/selection provenance")


def load_bundle(path: str | Path, *, device: str = "cpu") -> tuple[AttentionHead, dict[str, Any]]:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    validate_bundle(bundle)
    model = AttentionHead(bundle["state"]["mean"], bundle["state"]["scale"])
    model.load_state_dict(bundle["state"], strict=True)
    return model.requires_grad_(False).eval().to(device), bundle


def threshold_logit(bundle: dict[str, Any], cap: float = 0.3) -> float:
    key = f"val_good_rejection_cap_{cap:g}"
    if key not in bundle["thresholds"]:
        raise ValueError(f"no threshold for cap {cap:g}; available: {list(bundle['thresholds'])}")
    return float(bundle["thresholds"][key])


def probability(logits: Any) -> torch.Tensor:
    # Double precision avoids float32 sigmoid saturation for ordinary model logits.
    return torch.sigmoid(torch.as_tensor(logits, dtype=torch.float64))


def select_threshold(labels: Any, logits: Any, cap: float) -> float:
    """Maximize bad recall under a validation good-rejection cap, ties conservative."""
    y = torch.as_tensor(labels).cpu()
    scores = torch.as_tensor(logits, dtype=torch.float64).cpu()
    if (
        y.ndim != 1
        or y.shape != scores.shape
        or set(y.tolist()) != {0, 1}
        or not torch.isfinite(scores).all()
        or not 0 <= cap < 1
    ):
        raise ValueError("threshold selection needs finite aligned scores, both classes, cap in [0,1)")
    candidates = torch.cat(
        [
            torch.nextafter(scores.max(), torch.tensor(float("inf")))[None],
            scores.unique(sorted=True).flip(0),
        ]
    )
    best, best_recall = float(candidates[0]), -1
    for threshold in candidates:
        rejected = scores >= threshold
        recall = int(rejected[y == 0].sum())
        if float(rejected[y == 1].double().mean()) <= cap + 1e-12 and recall > best_recall:
            best, best_recall = float(threshold), recall
    return best
