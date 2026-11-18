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

"""Atomic per-video token caches, shared by training, adaptation and evaluation.
Only a batch is loaded into GPU memory; encoder tokens are stored as FP16.
A manifest binds cache rows to video hashes, labels, groups and the token contract.
"""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
from typing import Any

from filelock import FileLock
import numpy as np
import torch

from checks.physical_plausibility.attention import CONTRACT, digest
from checks.physical_plausibility.model import VjepaTokenEncoder, file_sha256


def atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def inventory(manifest: Path, *, labeled: bool = True) -> list[dict[str, Any]]:
    manifest = Path(manifest).resolve()
    rows = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get("video_path"):
            raise ValueError("each manifest row needs video_path")
        path = (manifest.parent / row["video_path"]).resolve()
        row = {**row, "video_path": str(path), "sample_id": str(row.get("sample_id", path))}
        if not row["sample_id"]:
            raise ValueError("empty sample ID")
        if labeled and (type(row.get("label")) is not int or row["label"] not in (0, 1)):
            raise ValueError("labels must be integer 1=good or 0=bad")
        sha = file_sha256(path)
        if row.get("video_sha256", sha) != sha:
            raise ValueError(f"video hash mismatch: {path}")
        rows.append({**row, "video_sha256": sha})
    if not rows or len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("empty manifest or duplicate sample IDs")
    return rows


class TokenStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.index = json.loads((self.root / "index.json").read_text())
        body = {k: v for k, v in self.index.items() if k != "fingerprint"}
        if body.get("contract") != CONTRACT or digest(body) != self.index.get("fingerprint"):
            raise ValueError("token cache identity mismatch")
        self.rows = body["rows"]
        self.fingerprint = self.index["fingerprint"]
        self.verified: set[int] = set()

    def path(self, i: int) -> Path:
        return self.root / "rows" / f"{i:06d}.npz"

    def read(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        path = self.path(i)
        record = json.loads(path.with_suffix(".json").read_text())
        row = self.rows[i]
        if record.get("fingerprint") != self.fingerprint or any(
            record.get(k) != row[k] for k in ("sample_id", "video_sha256")
        ):
            raise ValueError("token row identity mismatch")
        if i not in self.verified:
            if file_sha256(path) != record.get("tokens_sha256"):
                raise ValueError("token checksum mismatch")
            self.verified.add(i)
        with np.load(path, allow_pickle=False) as data:
            t, w = data["tokens"], data["weights"]
        if (
            t.dtype != np.float16
            or t.ndim != 2
            or t.shape[1] != 1024
            or not 0 < len(t) <= 18432
            or w.shape != (len(t),)
            or not np.isfinite(t).all()
            or not np.isfinite(w).all()
            or np.any(w <= 0)
            or np.any(w > 1)
        ):
            raise ValueError("invalid token row")
        return t, w

    def write(self, i: int, tokens: np.ndarray, weights: np.ndarray, metadata: dict[str, Any]) -> None:
        path = self.path(i)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with temp.open("wb") as f:
                np.savez(f, tokens=tokens, weights=weights)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
        atomic_json(
            path.with_suffix(".json"),
            {
                "fingerprint": self.fingerprint,
                "sample_id": self.rows[i]["sample_id"],
                "video_sha256": self.rows[i]["video_sha256"],
                "tokens_sha256": file_sha256(path),
                "metadata": metadata,
            },
        )


def extract(manifest: Path | None, cache: Path, *, device: str = "cuda") -> TokenStore:
    cache = Path(cache)
    if manifest is None:
        return TokenStore(cache)
    rows = inventory(manifest)
    cache.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache / ".extract.lock"), timeout=0):
        body = {"contract": CONTRACT, "rows": rows}
        index = {**body, "fingerprint": digest(body)}
        if (cache / "index.json").exists():
            if json.loads((cache / "index.json").read_text()) != index:
                raise ValueError("cache inputs changed; use a new token cache directory")
        else:
            atomic_json(cache / "index.json", index)
        store, encoder = TokenStore(cache), None
        for i, row in enumerate(rows):
            if store.path(i).with_suffix(".json").exists():
                store.read(i)
                continue
            if encoder is None:
                encoder = VjepaTokenEncoder(device=device)
            t, w = encoder.encode(row["video_path"])
            if file_sha256(row["video_path"]) != row["video_sha256"]:
                raise ValueError("video changed during extraction")
            store.write(i, t.cpu().numpy(), w.cpu().numpy(), {"decoder": "ffmpeg"})
            store.read(i)
    return store


def batch(store: TokenStore, indices: Sequence[int] | np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    arrays = [store.read(int(i)) for i in indices]
    width = max(len(t) for t, _ in arrays)
    tokens = torch.zeros((len(arrays), width, 1024), dtype=torch.float16)
    weights = torch.zeros((len(arrays), width), dtype=torch.float32)
    for i, (t, w) in enumerate(arrays):
        tokens[i, : len(t)] = torch.from_numpy(t)
        weights[i, : len(w)] = torch.from_numpy(w)
    return tokens.to(device), weights.to(device)


@torch.no_grad()
def predict(
    model: torch.nn.Module, store: TokenStore, indices: Sequence[int] | np.ndarray, *, device: str, batch_size: int = 8
) -> np.ndarray:
    model.eval()
    return torch.cat(
        [model(*batch(store, indices[i : i + batch_size], device)).cpu() for i in range(0, len(indices), batch_size)]
    ).numpy()


def training_roles(rows: list[dict[str, Any]], group_field: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    if any(type(r.get("label")) is not int or r["label"] not in (0, 1) for r in rows):
        raise ValueError("training requires binary integer labels")
    if any(r.get("split") not in ("train", "val") for r in rows):
        raise ValueError("training manifest must contain only train and val; keep test data separate")
    tr, va = (np.array([i for i, r in enumerate(rows) if r["split"] == role]) for role in ("train", "val"))
    for indices in (tr, va):
        if {rows[i]["label"] for i in indices} != {0, 1}:
            raise ValueError("train and val must each contain both classes")
    keys = ["video_sha256"]
    field = group_field or (
        "group" if any("group" in r for r in rows) else "group_id" if any("group_id" in r for r in rows) else None
    )
    if field:
        keys.append(field)
    for key in keys:
        if any(r.get(key) is None or str(r[key]) == "" for r in rows):
            raise ValueError(f"missing {key}")
        if {str(rows[i][key]) for i in tr} & {str(rows[i][key]) for i in va}:
            raise ValueError(f"{key} crosses train/val")
    return tr, va
