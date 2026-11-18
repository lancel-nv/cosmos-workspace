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

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
import torch

from checks.physical_plausibility import attention, evaluation, tokens, training
from checks.physical_plausibility.metrics import binary_metrics
from checks.physical_plausibility.model import file_sha256
from checks.physical_plausibility.tokens import TokenStore, atomic_json, training_roles


def make_bundle(path: Path, seed: int = 17) -> tuple[attention.AttentionHead, dict[str, Any]]:
    torch.manual_seed(seed)
    model = attention.AttentionHead(torch.randn(1024) * 0.05, torch.rand(1024) + 0.5)
    model.query.data.normal_(std=0.03)
    bundle = {
        "schema": attention.SCHEMA,
        "contract": attention.CONTRACT,
        "score_convention": "higher-is-bad; label1-good",
        "config": {"head": "linear", "pooling": "attention"},
        "state": model.state_dict(),
        "thresholds": {"val_good_rejection_cap_0.3": 0.1, "val_f1": 0.2},
        "training_and_selection_video_sha256": ["a" * 64],
    }
    attention.atomic_save(path, bundle)
    return model, bundle


def make_store(tmp_path: Path) -> TokenStore:
    rows = [
        {
            "sample_id": str(i),
            "video_path": f"/data/{i}.mp4",
            "video_sha256": f"{i:064x}",
            "label": i % 2,
            "split": "train" if i < 8 else "val",
            "group": str(i),
        }
        for i in range(12)
    ]
    body = {"contract": attention.CONTRACT, "rows": rows}
    atomic_json(tmp_path / "index.json", {**body, "fingerprint": attention.digest(body)})
    store = TokenStore(tmp_path)
    rng = np.random.default_rng(42)
    for i in range(12):
        t = rng.normal(size=(3 + i % 3, 1024)).astype(np.float16)
        t[:, 0] += 2 * (1 - 2 * (i % 2))
        store.write(i, t, np.linspace(0.25, 1, len(t), dtype=np.float32), {})
    return store


def test_pooling_matches_explicit_weighted_query_and_padding(tmp_path: Path) -> None:
    m, _ = make_bundle(tmp_path / "m.pt")
    t = torch.randn(2, 7, 1024).half()
    w = torch.tensor([[1, 0.5, 1, 1, 0, 0, 0], [1, 1, 0.25, 1, 1, 1, 1]])
    z = (t.float() - m.mean) / m.scale
    a = (z @ m.query / 32 + w.log()).softmax(-1)
    expected = ((a[..., None] * z).sum(1) @ m.head.weight.T + m.head.bias).squeeze(-1)
    torch.testing.assert_close(m(t, w), expected, rtol=0, atol=0)
    torch.testing.assert_close(m(t[:1], w[:1]), m(t[:1, :4], w[:1, :4]))


@pytest.mark.parametrize("scope", ["all", "linear"])
def test_finetune_reuses_weights_and_updates_only_selected_parameters(tmp_path: Path, scope: str) -> None:
    store = make_store(tmp_path / "cache")
    path = tmp_path / "source.pt"
    _, source = make_bundle(path)
    tr, _va = training_roles(store.rows)
    model, _ = training.initialize(store, tr, source=path, scope=scope, device="cpu", batch_size=4)
    for key, value in source["state"].items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert model.query.requires_grad == (scope == "all")
    output = tmp_path / "adapted.pt"
    c = {"steps": 4, "epochs": 1, "batch_size": 4, "validation_every": 1, "scope": scope}
    result = training.train(store, output, source=path, config=c, device="cpu")
    delta = result["training"]["last_parameter_delta_l2"]
    assert delta["mean"] == delta["scale"] == 0
    assert delta["head.weight"] > 0 and delta["head.bias"] > 0
    assert (delta["query"] > 0) == (scope == "all")
    assert result["training"]["source_sha256"] == file_sha256(path)
    original = output.read_bytes()
    training.train(store, output, source=path, config=c, device="cpu")
    assert output.read_bytes() == original
    with pytest.raises(ValueError, match="another fit identity"):
        training.train(store, output, source=path, config={**c, "lr": 0.01}, device="cpu")


def test_scaler_uses_training_videos_only_and_video_weighted_means(tmp_path: Path) -> None:
    store = make_store(tmp_path / "cache")
    tr, va = training_roles(store.rows)
    mean, scale = training.scaler(store, tr, "cpu", 4)
    means = []
    for i in tr:
        t, w = store.read(i)
        tensor = torch.from_numpy(t).float()
        weights = torch.from_numpy(w)
        means.append((tensor * weights[:, None]).sum(0) / weights.sum())
    values = torch.stack(means).double()
    torch.testing.assert_close(mean, values.mean(0).float())
    torch.testing.assert_close(scale, values.std(0, correction=0).clamp_min(1e-12).float())
    for i in va:
        t, w = store.read(i)
        store.write(i, t * 100, w, {})
    a, b = training.scaler(store, tr, "cpu", 4)
    torch.testing.assert_close(a, mean, rtol=0, atol=0)
    torch.testing.assert_close(b, scale, rtol=0, atol=0)


def test_training_rejects_source_seen_validation(tmp_path: Path) -> None:
    store = make_store(tmp_path / "cache")
    p = tmp_path / "source.pt"
    _, b = make_bundle(p)
    b["training_and_selection_video_sha256"] = [store.rows[8]["video_sha256"]]
    attention.atomic_save(p, b)
    with pytest.raises(ValueError, match="target validation overlaps"):
        training.train(store, tmp_path / "new.pt", source=p, config={"steps": 1}, device="cpu")


def test_bundle_rejects_contract_and_tensor_corruption(tmp_path: Path) -> None:
    p = tmp_path / "m.pt"
    _, b = make_bundle(p)
    b["contract"] = {**b["contract"], "storage": "float32"}
    attention.atomic_save(p, b)
    with pytest.raises(ValueError, match="contract"):
        attention.load_bundle(p)
    _, b = make_bundle(p)
    b["state"]["scale"][0] = 0
    attention.atomic_save(p, b)
    with pytest.raises(ValueError, match="scale"):
        attention.load_bundle(p)


def test_threshold_ties_keep_rejection_cap() -> None:
    y = np.array([1, 1, 0, 0])
    s = np.array([0.5, 0.4, 0.5, 0.6])
    cut = attention.select_threshold(y, s, 0)
    assert cut == 0.6
    assert attention.select_threshold(y, s, 0.5) == 0.5


def test_cache_checksum_and_split_leakage(tmp_path: Path) -> None:
    store = make_store(tmp_path / "cache")
    store.path(0).write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        store.read(0)
    store.rows[8]["group"] = store.rows[0]["group"]
    with pytest.raises(ValueError, match="crosses"):
        training_roles(store.rows)


def test_scratch_training_and_interrupted_resume_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = make_store(tmp_path / "cache")
    c = {"steps": 6, "epochs": 1, "batch_size": 4, "validation_every": 2}
    reference = training.train(store, tmp_path / "reference.pt", config=c, device="cpu")
    original = training.predict
    calls = 0

    def interrupted(*a: Any, **kw: Any) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption after saved step 2")
        return original(*a, **kw)

    monkeypatch.setattr(training, "predict", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        training.train(store, tmp_path / "resumed.pt", config=c, device="cpu")
    monkeypatch.setattr(training, "predict", original)
    resumed = training.train(store, tmp_path / "resumed.pt", config=c, device="cpu")
    assert resumed["selected_step"] == reference["selected_step"]
    assert resumed["thresholds"] == reference["thresholds"]
    assert resumed["training"]["history"] == reference["training"]["history"]
    for key in reference["state"]:
        torch.testing.assert_close(resumed["state"][key], reference["state"][key], rtol=0, atol=0)


def test_token_extraction_resumes_and_detects_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"original")
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({"video_path": "video.mp4", "label": 1}))
    calls = []

    class Encoder:
        def __init__(self, device: str) -> None:
            pass

        def encode(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
            calls.append(path)
            return torch.zeros(4, 1024, dtype=torch.float16), torch.ones(4)

    monkeypatch.setattr(tokens, "VjepaTokenEncoder", Encoder)
    tokens.extract(manifest, tmp_path / "cache", device="cpu")
    tokens.extract(manifest, tmp_path / "cache", device="cpu")
    assert len(calls) == 1
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="inputs changed"):
        tokens.extract(manifest, tmp_path / "cache", device="cpu")


@pytest.mark.parametrize("cap", [0.3, 0.45])
def test_calibration_preserves_weights_and_other_caps(tmp_path: Path, cap: float) -> None:
    store = make_store(tmp_path / "cache")
    source, output = tmp_path / "source.pt", tmp_path / "calibrated.pt"
    _, original = make_bundle(source)
    original["fit_identity"] = {"config": {"seed": 13}}
    original["fit_fingerprint"] = attention.digest(original["fit_identity"])
    attention.atomic_save(source, original)
    source_sha = file_sha256(source)
    evaluation.calibrate(store, source, output, cap=cap, device="cpu")
    _, calibrated = attention.load_bundle(output)
    assert calibrated["fit_identity"] == original["fit_identity"]
    assert "fit_fingerprint" not in calibrated
    key = f"val_good_rejection_cap_{cap:g}"
    assert set(calibrated["thresholds"]) == set(original["thresholds"]) | {key}
    for name, value in original["thresholds"].items():
        if name != key:
            assert calibrated["thresholds"][name] == value
    for name, value in original["state"].items():
        torch.testing.assert_close(calibrated["state"][name], value, rtol=0, atol=0)
    assert file_sha256(source) == source_sha
    assert evaluation.evaluate(store, source, device="cpu")["n_videos"] == len(store.rows)
    with pytest.raises(ValueError, match="overlaps"):
        evaluation.evaluate(store, output, cap=cap, device="cpu")
    with pytest.raises(FileExistsError):
        evaluation.calibrate(store, source, output, cap=cap, device="cpu")


def test_auc_matches_pairwise_ranking_with_ties() -> None:
    rng = np.random.default_rng(13)
    labels = np.array([0, 1] * 16)
    for _ in range(12):
        scores = rng.integers(0, 6, len(labels)).astype(float)
        bad, good = scores[labels == 0, None], scores[None, labels == 1]
        expected = float(((bad > good) + 0.5 * (bad == good)).mean())
        assert abs(binary_metrics(labels, scores, 2)["auroc_bad"] - expected) < 1e-12
    report = binary_metrics(np.array([1, 1, 0, 0]), np.array([0.5, 0.4, 0.5, 0.6]), 0.6)
    assert report["auroc_bad"] == 0.875
    assert abs(report["average_precision_bad"] - 5 / 6) < 1e-12
    assert report["recall_bad"] == 0.5 and report["good_false_rejection_rate"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
