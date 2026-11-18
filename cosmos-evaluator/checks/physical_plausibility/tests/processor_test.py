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

from pathlib import Path
import tempfile
from typing import cast
import unittest
from unittest.mock import MagicMock, patch

import torch

from checks.physical_plausibility.attention import CONTRACT, SCHEMA, AttentionHead, atomic_save
from checks.physical_plausibility.model import VjepaTokenEncoder
from checks.physical_plausibility.processor import PhysicalPlausibilityProcessor


class CheckpointUriTest(unittest.TestCase):
    @patch("checks.physical_plausibility.processor.VjepaTokenEncoder")
    def test_missing_checkpoint_fails_before_encoder(self, encoder: MagicMock) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint_uri"):
            PhysicalPlausibilityProcessor(params={"checkpoint_uri": None})
        encoder.assert_not_called()

    @patch(
        "checks.physical_plausibility.processor.load_bundle",
        return_value=(MagicMock(), {"thresholds": {"val_good_rejection_cap_0.3": 0.0}}),
    )
    @patch("checks.physical_plausibility.processor.download_if_remote", side_effect=lambda path: path)
    def test_explicit_checkpoint_overrides_configuration(self, download: MagicMock, _load: MagicMock) -> None:
        PhysicalPlausibilityProcessor(
            params={"checkpoint_uri": "/models/configured.pt"},
            checkpoint_uri="/models/attention.pt",
            encoder=cast(VjepaTokenEncoder, object()),
        )
        download.assert_called_once_with("/models/attention.pt")

    @patch(
        "checks.physical_plausibility.processor.load_bundle",
        return_value=(MagicMock(), {"thresholds": {"val_good_rejection_cap_0.3": 0.0}}),
    )
    @patch("checks.physical_plausibility.processor.hf_hub_download", return_value="/cache/model.pt")
    def test_hugging_face_revision_is_optional(self, download: MagicMock, _load: MagicMock) -> None:
        for source, revision in (
            ("hf://nvidia/evaluator/model.pt", None),
            ("hf://nvidia/evaluator@abc123/model.pt", "abc123"),
        ):
            with self.subTest(source=source):
                PhysicalPlausibilityProcessor(checkpoint_uri=source, encoder=cast(VjepaTokenEncoder, object()))
                download.assert_called_with(repo_id="nvidia/evaluator", filename="model.pt", revision=revision)

    @patch("checks.physical_plausibility.processor.os.remove")
    @patch(
        "checks.physical_plausibility.processor.load_bundle",
        return_value=(MagicMock(), {"thresholds": {"val_good_rejection_cap_0.3": 0.0}}),
    )
    @patch("checks.physical_plausibility.processor.download_if_remote", return_value="/tmp/model.pt")
    def test_remote_checkpoint_is_removed_after_loading(
        self, _download: MagicMock, _load: MagicMock, remove: MagicMock
    ) -> None:
        PhysicalPlausibilityProcessor(checkpoint_uri="s3://models/model.pt", encoder=cast(VjepaTokenEncoder, object()))
        remove.assert_called_once_with("/tmp/model.pt")


class AttentionProcessorTest(unittest.TestCase):
    @patch("checks.physical_plausibility.processor.download_if_remote", side_effect=lambda path: path)
    def test_attention_checkpoint_dispatch_scores_and_threshold_override(self, _download: MagicMock) -> None:
        model = AttentionHead(torch.zeros(1024), torch.ones(1024))
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.fill_(100.0)
        bundle = {
            "schema": SCHEMA,
            "contract": CONTRACT,
            "config": {"head": "linear", "pooling": "attention"},
            "state": model.state_dict(),
            "score_convention": "higher-is-bad; label1-good",
            "thresholds": {"val_good_rejection_cap_0.3": 101.0},
            "training_and_selection_video_sha256": [],
        }

        class Encoder:
            def encode(self, path: str) -> tuple[torch.Tensor, torch.Tensor]:
                return torch.zeros(3, 1024, dtype=torch.float16), torch.ones(3)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "head.pt"
            atomic_save(checkpoint, bundle)
            processor = PhysicalPlausibilityProcessor(
                params={"device": "cpu"}, encoder=cast(VjepaTokenEncoder, Encoder()), checkpoint_uri=str(checkpoint)
            )
            result = processor.process("clip", "video.mp4")
            self.assertEqual(result.bad_logit, 100.0)
            self.assertEqual(result.bad_logit_threshold, 101.0)
            self.assertTrue(result.passed)  # Saturated probabilities must not change saved-cut decisions.
            override = processor.process("clip", "video.mp4", bad_threshold=0.5)
            self.assertFalse(override.passed)
            self.assertIsNone(override.bad_logit_threshold)
            self.assertEqual(override.bad_threshold, 0.5)


if __name__ == "__main__":
    unittest.main()
