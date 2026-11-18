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

"""Pinned V-JEPA 2.1 loading and GPU token extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from typing import Any
import zipfile

from filelock import FileLock
import numpy as np
import torch
from torch import nn

from checks.physical_plausibility.features import (
    EMBEDDING_DIM,
    IMAGE_MEAN,
    IMAGE_STD,
    SPATIAL_TOKENS,
    TEMPORAL_TOKENS,
    letterbox_frames,
)
from checks.utils.video import get_video_frame_count, iter_video_frames

SOURCE_URL = "https://github.com/facebookresearch/vjepa2"
MODEL_REVISION = "204698b45b3712590f06245fbfba32d3be539812"
SOURCE_ARCHIVE_FILENAME = f"vjepa2-{MODEL_REVISION}.zip"
SOURCE_ARCHIVE_URL = f"{SOURCE_URL}/archive/{MODEL_REVISION}.zip"
SOURCE_ARCHIVE_SHA256 = "7e07b634ce33d743efbc81f2caae5a140170c90299de6266e14f8a0123e8ba76"
SOURCE_ARCHIVE_ROOT = f"vjepa2-{MODEL_REVISION}"
MODEL_FACTORY = "vjepa2_1_vit_large_384"
MODEL_ID = "official/vjepa2_1_vit_large_384"
CHECKPOINT_FILENAME = "vjepa2_1_vitl_dist_vitG_384.pt"
CHECKPOINT_URL = f"https://dl.fbaipublicfiles.com/vjepa2/{CHECKPOINT_FILENAME}"
CHECKPOINT_SHA256 = "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6"
CHECKPOINT_STATE_KEY = "ema_encoder"
NUM_FRAMES = 64


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def duration_spanning_indices(n_frames: int, count: int = NUM_FRAMES) -> np.ndarray:
    if n_frames < count:
        raise ValueError(f"V-JEPA 2.1 requires at least {count} real frames; got {n_frames}")
    indices = np.rint(np.linspace(0, n_frames - 1, count)).astype(np.int64)
    if len(np.unique(indices)) != count or indices[0] != 0 or indices[-1] != n_frames - 1:
        raise RuntimeError("uniform sampler failed its unique endpoint contract")
    return indices


def sample_frames_uniform(video_path: str | Path) -> np.ndarray:
    """Sample 64 unique RGB frames, including endpoints, with hermetic FFmpeg."""
    total = get_video_frame_count(video_path)
    if total is None:
        raise RuntimeError(f"could not determine frame count for {video_path}")
    wanted_indices = duration_spanning_indices(total)
    wanted = set(wanted_indices.tolist())
    selected: dict[int, np.ndarray] = {}
    decoded = 0
    stream = iter_video_frames(str(video_path))
    try:
        for index, bgr in enumerate(stream):
            decoded = index + 1
            if index in wanted:
                selected[index] = bgr[:, :, ::-1].copy()
    finally:
        stream.close()
    if decoded != total or len(selected) != NUM_FRAMES:
        raise RuntimeError(f"video frame-count/decode mismatch for {video_path}: probed={total}, decoded={decoded}")
    return np.stack([selected[int(index)] for index in wanted_indices])


def _checkpoint_path() -> Path:
    path = Path(torch.hub.get_dir()) / "checkpoints" / CHECKPOINT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_verified_encoder_checkpoint() -> Path:
    destination = _checkpoint_path()
    with FileLock(f"{destination}.lock"):
        if destination.exists():
            actual = file_sha256(destination)
            if actual != CHECKPOINT_SHA256:
                raise ValueError(f"existing V-JEPA checkpoint hash mismatch: {actual}")
            return destination
        temporary = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
        try:
            torch.hub.download_url_to_file(CHECKPOINT_URL, str(temporary), progress=True)
            actual = file_sha256(temporary)
            if actual != CHECKPOINT_SHA256:
                raise ValueError(f"downloaded V-JEPA checkpoint hash mismatch: {actual}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def _source_archive_path() -> Path:
    path = Path(torch.hub.get_dir()) / "sources" / SOURCE_ARCHIVE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_verified_encoder_source_archive() -> Path:
    """Cache the pinned V-JEPA source archive only after digest verification."""
    destination = _source_archive_path()
    with FileLock(f"{destination}.lock"):
        if destination.exists():
            actual = file_sha256(destination)
            if actual != SOURCE_ARCHIVE_SHA256:
                raise ValueError(f"existing V-JEPA source archive hash mismatch: {actual}")
            return destination
        temporary = destination.with_name(f".{destination.name}.partial.{os.getpid()}")
        try:
            torch.hub.download_url_to_file(SOURCE_ARCHIVE_URL, str(temporary), progress=True)
            actual = file_sha256(temporary)
            if actual != SOURCE_ARCHIVE_SHA256:
                raise ValueError(f"downloaded V-JEPA source archive hash mismatch: {actual}")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return destination


def _validated_source_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = []
    has_hubconf = False
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != SOURCE_ARCHIVE_ROOT:
            raise ValueError(f"unsafe path in V-JEPA source archive: {member.filename!r}")
        mode = (member.external_attr >> 16) & 0o170000
        if mode not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError(f"unsupported entry in V-JEPA source archive: {member.filename!r}")
        has_hubconf |= path == PurePosixPath(SOURCE_ARCHIVE_ROOT, "hubconf.py")
        members.append(member)
    if not has_hubconf:
        raise ValueError("V-JEPA source archive is missing hubconf.py")
    return members


def load_verified_encoder_model() -> nn.Module:
    """Load V-JEPA code locally after verifying the complete source archive."""
    archive_path = ensure_verified_encoder_source_archive()
    try:
        with (
            tempfile.TemporaryDirectory(prefix="vjepa2-source-") as directory,
            zipfile.ZipFile(archive_path) as archive,
        ):
            members = _validated_source_members(archive)
            archive.extractall(directory, members=members)
            source_path = Path(directory) / SOURCE_ARCHIVE_ROOT
            loaded = torch.hub.load(str(source_path), MODEL_FACTORY, source="local", pretrained=False)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid V-JEPA source archive {archive_path}") from exc
    return loaded[0] if isinstance(loaded, tuple) else loaded


def _clean_encoder_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for original_key, value in state.items():
        key = str(original_key)
        for prefix in ("module.", "backbone."):
            key = key.removeprefix(prefix)
        if key in cleaned:
            raise ValueError(f"checkpoint keys collide after prefix removal: {original_key!r}")
        cleaned[key] = value
    return cleaned


def _token_candidates(output: Any) -> list[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, Sequence) and not isinstance(output, (str, bytes, bytearray)):
        result: list[torch.Tensor] = []
        for value in output:
            result.extend(_token_candidates(value))
        return result
    return []


def _normalize_output(output: Any, batch_size: int) -> torch.Tensor:
    expected = (
        batch_size,
        TEMPORAL_TOKENS * SPATIAL_TOKENS * SPATIAL_TOKENS,
        EMBEDDING_DIM,
    )
    matches = [value for value in _token_candidates(output) if value.shape == expected]
    if not matches:
        raise RuntimeError(
            f"V-JEPA token shape mismatch: expected {expected}, "
            f"observed {[tuple(value.shape) for value in _token_candidates(output)]}"
        )
    return matches[-1].reshape(batch_size, TEMPORAL_TOKENS, SPATIAL_TOKENS, SPATIAL_TOKENS, EMBEDDING_DIM)


def make_deterministic() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VjepaTokenEncoder:
    """GPU attention tokens; process videos individually to preserve encoder layout."""

    def __init__(self, device: str = "cuda") -> None:
        make_deterministic()
        if torch.device(device).type != "cuda":
            raise ValueError("live attention extraction requires CUDA")
        self.device = device
        self.model = load_verified_encoder_model()
        checkpoint_path = ensure_verified_encoder_checkpoint()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
        state = checkpoint.get(CHECKPOINT_STATE_KEY) if isinstance(checkpoint, Mapping) else None
        if not isinstance(state, Mapping):
            raise TypeError(f"V-JEPA checkpoint is missing {CHECKPOINT_STATE_KEY!r}")
        self.model.load_state_dict(_clean_encoder_state_dict(state), strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.to(device)

    @torch.inference_mode()
    def encode(self, video_path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
        frames = sample_frames_uniform(video_path)
        canvas, spatial_weights = letterbox_frames(frames)
        x = torch.as_tensor(canvas, device=self.device).float().clamp(0, 1)
        mean = torch.as_tensor(IMAGE_MEAN, device=self.device)
        std = torch.as_tensor(IMAGE_STD, device=self.device)
        tensor = ((x - mean) / std).permute(3, 0, 1, 2).unsqueeze(0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = self.model(tensor, training=False)
        grid = _normalize_output(output, 1)[0].reshape(-1, 1024)
        weights = torch.as_tensor(np.broadcast_to(spatial_weights, (32, 24, 24)).reshape(-1).copy(), device=self.device)
        keep = weights > 0
        tokens = grid[keep].half()
        if not torch.isfinite(tokens).all():
            raise ValueError("nonfinite encoder tokens")
        return tokens, weights[keep]
