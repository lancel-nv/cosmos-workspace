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

"""Binary metrics with bad (label 0) as positive; tied scores share one operating point."""

import numpy as np
import torch


def binary_metrics(
    labels_good: np.ndarray | torch.Tensor,
    bad_scores: np.ndarray | torch.Tensor,
    threshold: float,
) -> dict[str, float | int]:
    labels = torch.as_tensor(labels_good, device="cpu")
    scores = torch.as_tensor(bad_scores, dtype=torch.float64, device="cpu")
    if labels.ndim != 1 or labels.shape != scores.shape or not len(labels):
        raise ValueError("labels and scores must be non-empty aligned vectors")
    if not torch.all((labels == 0) | (labels == 1)) or not torch.isfinite(scores).all():
        raise ValueError("labels must be binary and scores finite")
    bad, good = labels == 0, labels == 1
    predicted = scores >= threshold
    tp, fn = int((predicted & bad).sum()), int((~predicted & bad).sum())
    fp, tn = int((predicted & good).sum()), int((~predicted & good).sum())
    recall, specificity = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    auc, ap = float("nan"), float("nan")
    if bad.any() and good.any():
        order = scores.argsort(descending=True)
        sorted_scores = scores[order]
        ends = torch.cat([sorted_scores[:-1] != sorted_scores[1:], torch.tensor([True])])
        true_positive = bad[order].double().cumsum(0)[ends]
        false_positive = good[order].double().cumsum(0)[ends]
        tpr = torch.cat([torch.zeros(1), true_positive / bad.sum()])
        fpr = torch.cat([torch.zeros(1), false_positive / good.sum()])
        auc = float(torch.trapezoid(tpr, fpr))
        ap = float((tpr.diff() * true_positive / (true_positive + false_positive)).sum())
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "auroc_bad": auc,
        "average_precision_bad": ap,
        "recall_bad": recall,
        "precision_bad": precision,
        "f1_bad": 2 * precision * recall / max(precision + recall, 1e-12),
        "specificity_good": specificity,
        "good_false_rejection_rate": fp / max(fp + tn, 1),
        "balanced_accuracy": 0.5 * (recall + specificity),
    }
