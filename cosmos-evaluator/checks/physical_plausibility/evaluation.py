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

"""Held-out evaluation and threshold-only calibration of a frozen attention head."""

from pathlib import Path
from typing import Any

import numpy as np

from checks.physical_plausibility.attention import (
    atomic_save,
    load_bundle,
    select_threshold,
    threshold_logit,
    validate_bundle,
)
from checks.physical_plausibility.metrics import binary_metrics
from checks.physical_plausibility.model import file_sha256
from checks.physical_plausibility.tokens import TokenStore, predict


def held_out_scores(
    store: TokenStore, checkpoint: Path, *, device: str = "cuda", batch_size: int = 8
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model, bundle = load_bundle(checkpoint, device=device)
    hashes = [row["video_sha256"] for row in store.rows]
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate videos in evaluation/calibration")
    if set(hashes) & set(bundle["training_and_selection_video_sha256"]):
        raise ValueError("evaluation/calibration overlaps model training or selection data")
    labels = np.array([row["label"] for row in store.rows])
    if set(labels) != {0, 1}:
        raise ValueError("evaluation/calibration requires both classes")
    scores = predict(model, store, np.arange(len(labels)), device=device, batch_size=batch_size)
    return bundle, labels, scores


def evaluate(store: TokenStore, checkpoint: Path, *, cap: float = 0.3, device: str = "cuda") -> dict[str, Any]:
    bundle, labels, scores = held_out_scores(store, checkpoint, device=device)
    cutoff = threshold_logit(bundle, cap)
    return {
        "model_sha256": file_sha256(checkpoint),
        "cache_fingerprint": store.fingerprint,
        "n_videos": len(labels),
        "cap": cap,
        "bad_logit_threshold": cutoff,
        **binary_metrics(labels, scores, cutoff),
    }


def calibrate(
    store: TokenStore, checkpoint: Path, output: Path, *, cap: float = 0.3, device: str = "cuda"
) -> dict[str, Any]:
    """Update the requested operating point; retain learned weights and other thresholds."""
    if output.exists():
        raise FileExistsError(output)
    bundle, labels, scores = held_out_scores(store, checkpoint, device=device)
    calibrated = {
        **bundle,
        "thresholds": {
            **bundle["thresholds"],
            f"val_good_rejection_cap_{cap:g}": select_threshold(labels, scores, cap),
        },
        "training_and_selection_video_sha256": sorted(
            set(bundle["training_and_selection_video_sha256"]) | {row["video_sha256"] for row in store.rows}
        ),
        "calibration": {
            "parent_model_sha256": file_sha256(checkpoint),
            "cache_fingerprint": store.fingerprint,
            "cap": cap,
            "n_good": int((labels == 1).sum()),
            "n_bad": int((labels == 0).sum()),
        },
    }
    # Keep fit_identity as provenance of the unchanged weights; the calibrated bundle is not resumable.
    calibrated.pop("fit_fingerprint", None)
    validate_bundle(calibrated)
    atomic_save(output, calibrated)
    return calibrated
