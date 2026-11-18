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

"""Physical-plausibility checker processor."""

from __future__ import annotations

from collections.abc import Sequence
import logging
import math
import os
from typing import Any

from huggingface_hub import hf_hub_download, parse_hf_uri
from pydantic import BaseModel
import torch

from checks.physical_plausibility.attention import contract_fingerprint, load_bundle, probability, threshold_logit
from checks.physical_plausibility.model import VjepaTokenEncoder
from checks.utils.config_manager import ConfigManager
from checks.utils.multistorage import download_if_remote

logger = logging.getLogger(__name__)


class PhysicalPlausibilityResult(BaseModel):
    clip_id: str
    passed: bool
    bad_score: float
    plausibility_score: float
    bad_threshold: float
    prediction: str
    fingerprint: str
    bad_logit: float | None = None
    bad_logit_threshold: float | None = None

    @property
    def output_dir(self) -> None:
        """This checker returns metrics only; there are no artifacts to clean up."""
        return None


class PhysicalPlausibilityProcessor:
    def __init__(
        self,
        params: dict[str, Any] | None = None,
        *,
        encoder: VjepaTokenEncoder | None = None,
        checkpoint_uri: str | None = None,
        config_dir: str | None = None,
    ) -> None:
        config = self.get_default_config(config_dir).get("metropolis.physical_plausibility", {})
        if params:
            config.update(params)
        source = checkpoint_uri or config.get("checkpoint_uri")
        if not source:
            raise ValueError("set checkpoint_uri to an attention bundle")
        source = str(source)
        is_hf = source.startswith("hf://")
        if is_hf:
            uri = parse_hf_uri(source)
            if uri.type != "model" or not uri.path_in_repo:
                raise ValueError("checkpoint_uri must reference a file in a Hugging Face model repository")
            checkpoint = hf_hub_download(repo_id=uri.id, filename=uri.path_in_repo, revision=uri.revision)
        else:
            checkpoint = download_if_remote(source)
        try:
            self.head, self.attention_bundle = load_bundle(checkpoint)
        finally:
            if not is_hf and checkpoint != source:
                try:
                    os.remove(checkpoint)
                except OSError:
                    logger.warning("Could not remove downloaded checkpoint path=%s", checkpoint, exc_info=True)
        self.fingerprint = contract_fingerprint()
        self.cap = float(config.get("cap", 0.3))
        threshold_logit(self.attention_bundle, self.cap)
        device = str(config.get("device", "cuda"))
        self.head = self.head.to(device)
        self.encoder = encoder or VjepaTokenEncoder(device)
        configured_threshold = config.get("bad_threshold")
        self.bad_threshold = float(configured_threshold) if configured_threshold is not None else None
        if self.bad_threshold is not None and not 0 <= self.bad_threshold <= 1:
            raise ValueError("bad_threshold must be in [0, 1]")

    def process(
        self,
        clip_id: str,
        video_path: str,
        *,
        bad_threshold: float | None = None,
    ) -> PhysicalPlausibilityResult:
        return self.process_batch([(clip_id, video_path, bad_threshold)])[0]

    def process_batch(
        self,
        items: Sequence[tuple[str, str, float | None]],
    ) -> list[PhysicalPlausibilityResult]:
        if not items:
            return []
        local_paths: list[str] = []
        try:
            for _, video_path, _ in items:
                local_paths.append(download_if_remote(video_path))
            logits = []
            with torch.inference_mode():
                for path in local_paths:
                    tokens, weights = self.encoder.encode(path)
                    logits.append(float(self.head(tokens[None], weights[None])[0]))
            if not all(math.isfinite(value) for value in logits):
                raise ValueError("nonfinite attention scores")
            bad_scores = probability(logits).tolist()
        finally:
            for local_path, (_, source_path, _) in zip(local_paths, items):
                if local_path != source_path:
                    try:
                        os.remove(local_path)
                    except OSError:
                        logger.warning("Could not remove downloaded video path=%s", local_path, exc_info=True)

        cutoff = threshold_logit(self.attention_bundle, self.cap)
        default_threshold = float(probability(cutoff))
        results = []
        for index, ((clip_id, _, override), score) in enumerate(zip(items, bad_scores)):
            threshold = default_threshold
            if self.bad_threshold is not None:
                threshold = self.bad_threshold
            if override is not None:
                threshold = float(override)
            if not 0 <= threshold <= 1:
                raise ValueError("bad_threshold must be in [0, 1]")
            bad_score = float(score)
            is_bad = (
                logits[index] >= cutoff if self.bad_threshold is None and override is None else bad_score >= threshold
            )
            results.append(
                PhysicalPlausibilityResult(
                    clip_id=clip_id,
                    passed=not is_bad,
                    bad_score=bad_score,
                    plausibility_score=1.0 - bad_score,
                    bad_threshold=threshold,
                    prediction="bad" if is_bad else "good",
                    fingerprint=self.fingerprint,
                    bad_logit=logits[index],
                    bad_logit_threshold=cutoff if self.bad_threshold is None and override is None else None,
                )
            )
        logger.info("Completed physical-plausibility batch batch_size=%d", len(results))
        return results

    @staticmethod
    def get_default_config(config_dir: str | None = None) -> dict[str, Any]:
        config = ConfigManager(config_dir).load_config("cosmos_evaluator")
        return {"metropolis.physical_plausibility": config["metropolis.physical_plausibility"]}
