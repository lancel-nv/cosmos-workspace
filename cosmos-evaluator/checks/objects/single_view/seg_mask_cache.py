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
On-disk cache for SegFormer segmentation masks.

Stores per-frame resized segmentation masks so that they can be reused across multiple runs that differ only in world
model data. Each frame is saved as an individual .pt file so that frames can be loaded on-demand without reading the
entire cache into memory.
"""

import hashlib
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class SegMaskCache:
    """Per-video, per-frame cache for resized SegFormer segmentation masks.

    Cache layout on disk::

        <cache_dir>/<video_hash>/frame_<frame_idx>.pt

    **Cache key design**

    The per-video subdirectory name is derived from the *resolved absolute path*
    of the video file, not from its contents.  Specifically it is the first 16
    hex characters of ``SHA-256(resolved_absolute_path_string)``.  This means:

    - Two runs that point at the *same path* always share a cache.
    - The same video file accessed via a *different path* (e.g. a symlink, a re-download to a different location, or a
      different mount point) will produce a *different* subdirectory and result in a cache miss.

    The per-frame filename is ``frame_<video_frame_idx>.pt``, where ``video_frame_idx`` is the 0-based index into the
    video dataloader (i.e. the raw frame position in the video file, before any FPS mapping to the pose timeline).

    Usage::

        cache = SegMaskCache("/tmp/seg_cache", "/path/to/video.mp4")
        masks = cache.load(frame_idx)
        if masks is None:
            masks = seg_model.process_frame(...)
            cache.save(frame_idx, masks)
    """

    def __init__(self, cache_dir: str, video_path: str) -> None:
        """Initialize the cache for a specific video.

        Args:
            cache_dir: Root directory under which per-video subdirectories are created.
            video_path: Absolute or relative path to the video file. Used to derive a
                stable per-video subdirectory name via SHA-256 hash.
        """
        video_hash = hashlib.sha256(str(Path(video_path).resolve()).encode()).hexdigest()[:16]
        self._dir = Path(cache_dir) / video_hash
        self._dir.mkdir(parents=True, exist_ok=True)
        logger.debug("SegMaskCache initialized at %s for video %r", self._dir, video_path)

    def _frame_path(self, frame_idx: int) -> Path:
        return self._dir / f"frame_{frame_idx}.pt"

    def has(self, frame_idx: int) -> bool:
        """Return True if a cached mask exists for *frame_idx*."""
        return self._frame_path(frame_idx).exists()

    def load(self, frame_idx: int) -> torch.Tensor | None:
        """Load and return the cached mask for *frame_idx*, or None if not cached."""
        path = self._frame_path(frame_idx)
        if not path.exists():
            return None
        try:
            return torch.load(path, weights_only=True)
        except Exception as exc:  # noqa: BLE001 - torch.load failure surface is broad; cache miss returns None
            logger.warning("Failed to load cached mask for frame %s from %s: %s", frame_idx, path, exc)
            return None

    def save(self, frame_idx: int, masks: torch.Tensor) -> None:
        """Persist *masks* for *frame_idx* to disk.

        Silently skips the write if an error occurs so that a cache write failure
        never aborts processing.
        """
        path = self._frame_path(frame_idx)
        try:
            torch.save(masks, path)
        except Exception as exc:  # noqa: BLE001 - cache write failure must not abort processing
            logger.warning("Failed to save mask for frame %s to %s: %s", frame_idx, path, exc)
