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

"""Execute all notebook workflows with synthetic tokens; no model download or GPU required."""

import json
from pathlib import Path
import runpy
import sys
from typing import Any

import pytest
import torch

from checks.physical_plausibility import model, tokens
from checks.physical_plausibility.attention import CONTRACT, SCHEMA, AttentionHead, atomic_save
from utils import bazel


class SyntheticEncoder:
    def __init__(self, device: str) -> None:
        pass

    def encode(self, path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        seed = int(Path(path).read_text())
        generator = torch.Generator().manual_seed(seed)
        return torch.randn(4, 1024, generator=generator).half(), torch.tensor([1.0, 0.5, 0.25, 1.0])


@pytest.mark.parametrize("mode", ["scratch", "finetune", "calibrate"])
def test_notebook_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    notebook_path = bazel.get_runfiles_path("docs/tutorials/train_physical_plausibility_checker.ipynb")
    assert notebook_path is not None
    notebook = json.loads(Path(notebook_path).read_text())
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(not cell["outputs"] and cell["execution_count"] is None for cell in cells)
    for name, roles in (
        ("train-val", ["train"] * 4 + ["val"] * 4),
        ("calibration", ["calibration"] * 4),
        ("test", ["test"] * 4),
    ):
        rows = []
        for i, role in enumerate(roles):
            path = tmp_path / f"{name}-{i}.mp4"
            # Distinct file contents ensure source hashing tests real overlap checks.
            path.write_text(str({"train-val": 0, "calibration": 100, "test": 200}[name] + i))
            rows.append({"video_path": str(path), "label": i % 2, "split": role, "group_id": path.stem})
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    source = tmp_path / "source.pt"
    head = AttentionHead(torch.zeros(1024), torch.ones(1024))
    atomic_save(
        source,
        {
            "schema": SCHEMA,
            "contract": CONTRACT,
            "score_convention": "higher-is-bad; label1-good",
            "config": {"head": "linear", "pooling": "attention"},
            "state": head.state_dict(),
            "thresholds": {"val_good_rejection_cap_0.3": 0.0},
            "training_and_selection_video_sha256": [],
        },
    )
    monkeypatch.setattr(tokens, "VjepaTokenEncoder", SyntheticEncoder)
    monkeypatch.setattr(model, "VjepaTokenEncoder", SyntheticEncoder)
    # Import discovery starts from the notebook's runfiles module, as from a source checkout.
    monkeypatch.chdir(Path(notebook_path).parents[2])
    namespace: dict[str, Any] = {}
    for cell in cells:
        code = "".join(cell["source"])
        if cell["metadata"].get("workflow", mode) != mode:
            continue
        code = code.replace('DEVICE = "cuda"', 'DEVICE = "cpu"')
        for old, new in {
            "/models/finevideo-attention-linear-v1/model.pt": source,
            "/data/train-val.jsonl": tmp_path / "train-val.jsonl",
            "/data/calibration.jsonl": tmp_path / "calibration.jsonl",
            "/data/test.jsonl": tmp_path / "test.jsonl",
            "/results/my-checker": tmp_path / "output",
        }.items():
            code = code.replace(old, str(new))
        if "CONFIG = settings(" in code:
            code += "\nCONFIG.update(steps=2, epochs=1, batch_size=4, validation_every=1)\n"
        code = code.replace("/data/new-video.mp4", str(tmp_path / "test-0.mp4"))
        script = tmp_path / "notebook_cell.py"
        script.write_text(code)
        if "CUDA is required for extraction" in code:
            with monkeypatch.context() as setup:
                setup.setattr(torch.cuda, "is_available", lambda: True)
                namespace = runpy.run_path(str(script), init_globals=namespace)
        else:
            namespace = runpy.run_path(str(script), init_globals=namespace)
    assert namespace["CHECKPOINT"].is_file()
    assert namespace["report"]["n_videos"] == 4
    assert torch.isfinite(torch.tensor(namespace["logit"]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
