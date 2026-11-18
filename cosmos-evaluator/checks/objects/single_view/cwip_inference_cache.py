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
On-disk cache for CWIP inference outputs.

Stores the full CWIP inference result for a camera/world video pair so that it can be reused across multiple runs that
differ only in world model metadata.

Unlike the per-frame SegFormer cache, CWIP processes all frames in a single batch call, so the cache is a single ``.pt``
file per video pair.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class CwipInferenceCache:
    """Cache for CWIP inference outputs keyed on a camera/world video pair.

    Cache layout on disk::

        <cache_dir>/<pair_hash>/inference.pt

    **Cache key design**

    The per-pair subdirectory name is derived from the *resolved absolute paths* of both the camera and world video
    files plus the frame-sampling parameters, specifically the first 16 hex characters of
    ``SHA-256(resolved_camera_path + "|" + resolved_world_path + "|" + extra_key)``. This means:

    - Two runs pointing at the *same pair of paths* **and the same sampling** (``extra_key``) share a cache entry.
    - If either path changes (e.g. a re-download to a different location or a different symlink), the hash changes and
      the result is a cache miss.
    - **Different frame sampling produces a different entry.** The cached inference covers only the frames that were
      sampled, so a partial run (e.g. ``--trial N``) must not be reused by a full run. Callers pass the sampling
      params (trial count, sample method) as ``extra_key`` so the two never collide.

    Usage::

        cache = CwipInferenceCache(cache_dir, camera_path, world_path, extra_key="trial=None|sample=uniform")
        result = cache.load()
        if result is None:
            summary, sim, types, logits = run_cwip(...)
            cache.save(summary, sim, types, logits)
    """

    _FILENAME = "inference.pt"

    def __init__(self, cache_dir: str, camera_video_path: str, world_video_path: str, extra_key: str = "") -> None:
        """Initialize the cache for a specific camera/world video pair and frame sampling.

        Args:
            cache_dir: Root directory under which per-pair subdirectories are created.
            camera_video_path: Absolute or relative path to the camera video file.
            world_video_path: Absolute or relative path to the world video file.
                Together with *camera_video_path*, used to derive a stable per-pair
                subdirectory name via SHA-256 hash of the resolved paths.
            extra_key: Extra discriminator folded into the hash so runs that sample a *different set of frames*
                (e.g. ``--trial N`` vs a full run, or different sample methods) never share a cache entry. The
                cached inference only covers the sampled frames, so reusing it across samplings truncates results.
        """
        cam_resolved = str(Path(camera_video_path).resolve())
        world_resolved = str(Path(world_video_path).resolve())
        pair_key = cam_resolved + "|" + world_resolved + "|" + extra_key
        pair_hash = hashlib.sha256(pair_key.encode()).hexdigest()[:16]
        self._path = Path(cache_dir) / pair_hash / self._FILENAME
        self._path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(
            "CwipInferenceCache initialized at %s for pair (%r, %r)",
            self._path,
            camera_video_path,
            world_video_path,
        )

    def has(self) -> bool:
        """Return True if a cached inference result exists for this video pair."""
        return self._path.exists()

    def load(
        self,
    ) -> tuple[dict[str, Any], torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None:
        """Load and return the cached inference result, or None if not cached.

        Returns:
            ``(summary, pred_similarity_indices, pred_type_indices, raw_type_logits)``
            matching the return signature of
            ``ObjectProcessorStatic._run_cwip_inference_on_videos``, or ``None``
            on a cache miss or load error.
        """
        if not self._path.exists():
            return None
        try:
            data = torch.load(self._path, weights_only=True)
            return (
                data["summary"],
                data.get("pred_similarity_indices"),
                data.get("pred_type_indices"),
                data.get("raw_type_logits"),
            )
        except Exception as exc:  # noqa: BLE001 - torch.load failure surface is broad; cache miss must not abort
            logger.warning("Failed to load cached CWIP inference from %s: %s", self._path, exc)
            return None

    def save(
        self,
        summary: dict[str, Any],
        pred_similarity_indices: torch.Tensor | None,
        pred_type_indices: torch.Tensor | None,
        raw_type_logits: torch.Tensor | None,
    ) -> None:
        """Persist the CWIP inference result to disk.

        Silently skips the write if an error occurs so that a cache write failure never aborts processing.

        Args:
            summary: Summary dict returned by ``_run_cwip_inference_on_videos``.
            pred_similarity_indices: Patch similarity class indices tensor (N, H, W), or None.
            pred_type_indices: Patch type class indices tensor (N, H, W), or None.
            raw_type_logits: Raw type logits tensor (N, C, Hp, Wp) on CPU, or None.
        """
        data = {
            "summary": summary,
            "pred_similarity_indices": pred_similarity_indices,
            "pred_type_indices": pred_type_indices,
            "raw_type_logits": raw_type_logits,
        }
        try:
            torch.save(data, self._path)
        except Exception as exc:  # noqa: BLE001 - cache write failure must not abort processing
            logger.warning("Failed to save CWIP inference to %s: %s", self._path, exc)
