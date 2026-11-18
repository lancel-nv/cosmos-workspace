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

"""Train a new attention head or continue from pretrained query/classifier weights.
Encoder tokens never require gradients. Fine-tuning preserves source normalization.
Balanced BCE plus explicit L2 updates the selected parameters; validation picks
both the best checkpoint (including step zero) and the rejection thresholds.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from filelock import FileLock
import numpy as np
import torch
from torch.nn import functional as F

from checks.physical_plausibility.attention import (
    CONTRACT,
    SCHEMA,
    AttentionHead,
    atomic_save,
    digest,
    load_bundle,
    select_threshold,
    validate_bundle,
)
from checks.physical_plausibility.metrics import binary_metrics
from checks.physical_plausibility.model import file_sha256, make_deterministic
from checks.physical_plausibility.tokens import TokenStore, batch, predict, training_roles

LOGGER = logging.getLogger(__name__)


def settings(*, finetune: bool = False, **overrides: Any) -> dict[str, Any]:
    c: dict[str, Any] = {
        "optimizer": "adam",
        "lr": 0.0001 if finetune else 0.003,
        "head_l2": 0.001,
        "query_l2": 0.001,
        "batch_size": 8,
        "steps": 3600,
        "epochs": 10,
        "validation_every": 100,
        "seed": 20260915,
        "caps": [0.1, 0.2, 0.3],
        "scope": "all",
    }
    if set(overrides) - set(c):
        raise ValueError(f"unknown settings: {set(overrides) - set(c)}")
    c.update(overrides)
    for key in ("batch_size", "steps", "epochs", "validation_every"):
        if type(c[key]) is not int or c[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if (
        type(c["seed"]) is not int
        or c["seed"] < 0
        or c["optimizer"] not in ("adam", "sgd")
        or c["scope"] not in ("all", "linear")
    ):
        raise ValueError("invalid seed, optimizer or scope")
    if not finetune and c["scope"] != "all":
        raise ValueError("linear-only scope requires a pretrained attention head")
    if c["lr"] <= 0 or min(c["head_l2"], c["query_l2"]) < 0 or not c["caps"] or any(not 0 <= v < 1 for v in c["caps"]):
        raise ValueError("invalid learning rate, regularization or caps")
    digest(c)
    return c


@torch.no_grad()
def scaler(store: TokenStore, indices: np.ndarray, device: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    means = []
    for offset in range(0, len(indices), batch_size):
        t, w = batch(store, indices[offset : offset + batch_size], device)
        means.append(((t.float() * w[..., None]).sum(1) / w.sum(1, keepdim=True)).double())
    values = torch.cat(means)
    return values.mean(0).float(), values.std(0, correction=0).clamp_min(1e-12).float()


def initialize(
    store: TokenStore, tr: np.ndarray, *, source: Path | None, scope: str, device: str, batch_size: int
) -> tuple[AttentionHead, dict[str, Any] | None]:
    if source:
        model, bundle = load_bundle(source, device=device)
    else:
        model = AttentionHead(*scaler(store, tr, device, batch_size)).to(device)
        bundle = None
    model.requires_grad_(True)
    model.query.requires_grad_(scope == "all")
    return model, bundle


def train(
    store: TokenStore,
    output: Path,
    *,
    source: Path | None = None,
    config: dict[str, Any] | None = None,
    device: str = "cuda",
    group_field: str | None = None,
    label_definition: str | None = None,
) -> dict[str, Any]:
    output = Path(output)
    c = settings(finetune=source is not None, **(config or {}))
    tr, va = training_roles(store.rows, group_field)
    labels = np.array([r["label"] for r in store.rows])
    source_sha = file_sha256(source) if source else None
    identity = {
        "cache": store.fingerprint,
        "config": c,
        "source_sha256": source_sha,
        "group_field": group_field,
        "label_definition": label_definition,
        "implementation": {
            p.name: file_sha256(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("attention.py"),
                Path(__file__).with_name("tokens.py"),
                Path(__file__).with_name("metrics.py"),
            ]
        },
    }
    fingerprint = digest(identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock", timeout=0):
        if output.exists():
            _, existing = load_bundle(output, device="cpu")
            if existing.get("fit_fingerprint") != fingerprint:
                raise ValueError("output already exists with another fit identity; choose a new path")
            return existing
        make_deterministic()
        torch.manual_seed(c["seed"])
        model, source_bundle = initialize(
            store, tr, source=source, scope=c["scope"], device=device, batch_size=c["batch_size"]
        )
        source_seen = set(source_bundle["training_and_selection_video_sha256"]) if source_bundle else set()
        if source_seen & {store.rows[i]["video_sha256"] for i in va}:
            raise ValueError("target validation overlaps source training/selection data")
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = (
            torch.optim.Adam(parameters, lr=c["lr"], weight_decay=0)
            if c["optimizer"] == "adam"
            else torch.optim.SGD(parameters, lr=c["lr"], momentum=0.9, weight_decay=0)
        )
        steps = max(c["steps"], math.ceil(c["epochs"] * len(tr) / c["batch_size"]))
        resume = output.with_name(output.name + ".resume.pt")
        step, best_step, best_auc = 0, 0, -1.0
        best: dict[str, torch.Tensor] | None = None
        history: list[dict[str, Any]] = []
        initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if resume.exists():
            saved = torch.load(resume, map_location="cpu", weights_only=True)
            if saved.get("fingerprint") != fingerprint:
                raise ValueError("resume identity mismatch; choose a new output")
            model.load_state_dict(saved["state"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            step, best_step, best_auc, best, history = (
                saved[k] for k in ("step", "best_step", "best_auc", "best", "history")
            )

        def validate() -> None:
            nonlocal best, best_step, best_auc
            scores = predict(model, store, va, device=device, batch_size=c["batch_size"])
            auc = float(binary_metrics(labels[va], scores, 0.0)["auroc_bad"])
            history.append({"step": step, "val_auc": auc})
            if auc > best_auc + 1e-12:
                best_auc, best_step = auc, step
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            atomic_save(
                resume,
                {
                    "fingerprint": fingerprint,
                    "state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_step": best_step,
                    "best_auc": best_auc,
                    "best": best,
                    "history": history,
                },
            )
            LOGGER.info("attention step=%d val_auc=%.6f best_step=%d", step, auc, best_step)

        if best is None:
            validate()
        epoch, offset = divmod(step * c["batch_size"], len(tr))
        order = np.random.default_rng(c["seed"] + epoch).permutation(tr)
        n_bad = int((labels[tr] == 0).sum())
        while step < steps:
            indices: list[int] = []
            while len(indices) < c["batch_size"]:
                take = min(c["batch_size"] - len(indices), len(tr) - offset)
                indices.extend(order[offset : offset + take].tolist())
                offset += take
                if offset == len(tr):
                    epoch, offset = epoch + 1, 0
                    order = np.random.default_rng(c["seed"] + epoch).permutation(tr)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch(store, indices, device))
            y = torch.as_tensor(1 - labels[indices], dtype=torch.float32, device=device)
            weights = torch.where(y == 1, len(tr) / (2 * n_bad), len(tr) / (2 * (len(tr) - n_bad)))
            loss = (
                weights * F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            ).mean() + model.regularization(c["head_l2"], c["query_l2"])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("nonfinite gradient")
            optimizer.step()
            step += 1
            if step % c["validation_every"] == 0 or step == steps:
                validate()
        last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        assert best is not None
        model.load_state_dict(best, strict=True)
        scores = predict(model, store, va, device=device, batch_size=c["batch_size"])
        result = {
            "schema": SCHEMA,
            "contract": CONTRACT,
            "score_convention": "higher-is-bad; label1-good",
            "config": {
                **{k: v for k, v in c.items() if k not in ("epochs", "scope")},
                "head": "linear",
                "pooling": "attention",
            },
            "state": best,
            "thresholds": {
                f"val_good_rejection_cap_{cap:g}": select_threshold(labels[va], scores, cap) for cap in c["caps"]
            },
            "selected_step": best_step,
            "label_definition": label_definition
            or (source_bundle or {}).get("label_definition", (source_bundle or {}).get("dataset", {})),
            "training_and_selection_video_sha256": sorted(source_seen | {r["video_sha256"] for r in store.rows}),
            "fit_fingerprint": fingerprint,
            "fit_identity": identity,
            "training": {
                "mode": "finetune" if source else "scratch",
                "scope": c["scope"],
                "source_sha256": source_sha,
                "val_auc": best_auc,
                "history": history,
                "n_train": len(tr),
                "n_val": len(va),
                "last_parameter_delta_l2": {k: float((last_state[k] - v).norm()) for k, v in initial_state.items()},
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "device": str(device),
            },
        }
        validate_bundle(result)
        atomic_save(output, result)
        return result
