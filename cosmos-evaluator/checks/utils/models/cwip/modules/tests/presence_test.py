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

"""Tests for the shared static-check presence masks."""

import math
from typing import cast
import unittest

import torch

from checks.utils.models.cwip.modules.presence import (
    CAMERA_ONLY_PRESENCE_SOURCE,
    PresenceConfig,
    PresenceSource,
    TypeMaskConfig,
    TypeMaskGate,
    presence_layers,
    type_presence_mask,
)
from checks.utils.models.cwip.visualization.world_model_utils import PatchSimilarityType

MATCH = PatchSimilarityType.MATCH.value
EXTRA = PatchSimilarityType.EXTRA.value
MISSING = PatchSimilarityType.MISSING.value


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class TestTypePresenceMask(unittest.TestCase):
    def setUp(self) -> None:
        # Three patches, two classes. Top probabilities: 0.9 (class 0), 0.6 (class 1), 0.3 (class 1).
        self.type_pred = torch.tensor([[0, 1, 1]])
        probs = torch.tensor([[[0.9, 0.2, 0.1]], [[0.1, 0.6, 0.3]]])
        self.type_logits = torch.log(probs / (1 - probs))

    def test_threshold_gates_on_top_probability(self) -> None:
        mask = type_presence_mask(self.type_pred, self.type_logits, TypeMaskConfig(confidence_threshold=0.5))
        self.assertEqual(mask.tolist(), [[1, 1, 0]])
        self.assertEqual(mask.dtype, torch.long)

    def test_threshold_is_inclusive(self) -> None:
        mask = type_presence_mask(self.type_pred, self.type_logits, TypeMaskConfig(confidence_threshold=0.6 - 1e-6))
        self.assertEqual(mask.tolist(), [[1, 1, 0]])

    def test_class_threshold_overrides_only_its_class(self) -> None:
        mask = type_presence_mask(
            self.type_pred, self.type_logits, TypeMaskConfig(confidence_threshold=0.5, class_thresholds={1: 0.25})
        )
        self.assertEqual(mask.tolist(), [[1, 1, 1]])
        mask = type_presence_mask(
            self.type_pred, self.type_logits, TypeMaskConfig(confidence_threshold=0.5, class_thresholds={0: 0.95})
        )
        self.assertEqual(mask.tolist(), [[0, 1, 0]])

    def test_argmax_gate_marks_every_patch_and_needs_no_logits(self) -> None:
        mask = type_presence_mask(self.type_pred, None, TypeMaskConfig(gate=TypeMaskGate.ARGMAX))
        self.assertEqual(mask.tolist(), [[1, 1, 1]])

    def test_confidence_gate_without_logits_raises(self) -> None:
        with self.assertRaises(ValueError):
            type_presence_mask(self.type_pred, None, TypeMaskConfig())

    def test_logits_resized_to_prediction_grid(self) -> None:
        type_pred = torch.zeros(4, 6, dtype=torch.long)
        type_logits = torch.full((2, 2, 3), _logit(0.8))
        type_logits[1] = _logit(0.1)
        self.assertEqual(
            type_presence_mask(type_pred, type_logits, TypeMaskConfig(confidence_threshold=0.5)).shape, (4, 6)
        )
        self.assertTrue(
            bool(type_presence_mask(type_pred, type_logits, TypeMaskConfig(confidence_threshold=0.5)).all())
        )
        self.assertFalse(
            bool(type_presence_mask(type_pred, type_logits, TypeMaskConfig(confidence_threshold=0.9)).any())
        )


class TestPresenceLayers(unittest.TestCase):
    """Each similarity label (MATCH / EXTRA / MISSING) meets each type-head verdict (present / absent)."""

    def setUp(self) -> None:
        self.similarity = torch.tensor([[MATCH, EXTRA, MISSING], [MATCH, EXTRA, MISSING]])
        self.type_mask = torch.tensor([[1, 1, 1], [0, 0, 0]])
        self.calls = 0

    def _type_presence(self) -> torch.Tensor:
        self.calls += 1
        return self.type_mask

    def _layers(self, source: PresenceSource) -> tuple[list, list]:
        match, extra = presence_layers(self.similarity, self._type_presence, source)
        return match.tolist(), extra.tolist()

    def test_similarity_ignores_type_head(self) -> None:
        self.assertEqual(self._layers(PresenceSource.SIMILARITY), ([[1, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0]]))
        self.assertEqual(self.calls, 0)

    def test_type_uses_type_mask_for_both(self) -> None:
        self.assertEqual(self._layers(PresenceSource.TYPE), ([[1, 1, 1], [0, 0, 0]], [[1, 1, 1], [0, 0, 0]]))

    def test_union(self) -> None:
        self.assertEqual(self._layers(PresenceSource.UNION), ([[1, 1, 1], [1, 0, 0]], [[1, 1, 1], [0, 1, 0]]))

    def test_intersection(self) -> None:
        self.assertEqual(self._layers(PresenceSource.INTERSECTION), ([[1, 0, 0], [0, 0, 0]], [[0, 1, 0], [0, 0, 0]]))

    def test_camera_only_uses_type_mask_whatever_the_source(self) -> None:
        for source in PresenceSource:
            match, extra = presence_layers(None, self._type_presence, source)
            self.assertEqual(match.tolist(), self.type_mask.tolist(), source)
            self.assertEqual(extra.tolist(), self.type_mask.tolist(), source)

    def test_layers_are_long(self) -> None:
        for source in PresenceSource:
            match, extra = presence_layers(self.similarity, self._type_presence, source)
            self.assertEqual((match.dtype, extra.dtype), (torch.long, torch.long), source)

    def test_unknown_source_raises(self) -> None:
        with self.assertRaises(ValueError):
            presence_layers(self.similarity, self._type_presence, cast(PresenceSource, "argmax"))


class TestPresenceSourceAndConfig(unittest.TestCase):
    def test_type_mask_dependence(self) -> None:
        self.assertEqual(
            {s: s.uses_type_mask for s in PresenceSource},
            {
                PresenceSource.SIMILARITY: False,
                PresenceSource.TYPE: True,
                PresenceSource.UNION: True,
                PresenceSource.INTERSECTION: True,
            },
        )
        # Only sources whose hallucination mask can hold patches the similarity head does not mark.
        self.assertEqual(
            {s for s in PresenceSource if s.includes_type_only_patches}, {PresenceSource.TYPE, PresenceSource.UNION}
        )

    def test_only_the_confidence_gate_needs_logits(self) -> None:
        self.assertTrue(TypeMaskConfig(gate=TypeMaskGate.CONFIDENCE).needs_logits)
        self.assertFalse(TypeMaskConfig(gate=TypeMaskGate.ARGMAX).needs_logits)

    def test_source_for_run(self) -> None:
        config = PresenceConfig(source=PresenceSource.INTERSECTION)
        self.assertIs(config.source_for_run(camera_only=False), PresenceSource.INTERSECTION)
        self.assertIs(config.source_for_run(camera_only=True), CAMERA_ONLY_PRESENCE_SOURCE)
        self.assertIs(CAMERA_ONLY_PRESENCE_SOURCE, PresenceSource.TYPE)
        self.assertIs(PresenceConfig().source, PresenceSource.UNION)


if __name__ == "__main__":
    unittest.main()
