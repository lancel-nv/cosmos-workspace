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

"""Per-patch presence masks derived from CWIP outputs.

The static object check reads CWIP through a binary presence mask per frame: correspondence asks whether a class is
present where the ground truth projects it, hallucination asks whether it is present where nothing projects. Two
sources can supply that mask:

- the similarity head (two-stream only): MATCH patches for correspondence, EXTRA patches for hallucination;
- the type head: a patch is present when its top object-type probability clears a confidence threshold (gate
  ``confidence``), or unconditionally (gate ``argmax``, leaving the class filter to select the type).

A two-stream run picks one :class:`PresenceSource`; a camera-only run, which has no similarity head, always uses the
type-head mask (:data:`CAMERA_ONLY_PRESENCE_SOURCE`). The static checker and the CWIP inference tool build their masks
with these functions, so they apply the same source and gate logic. They do not produce identical masks: the checker
gates logits bilinearly resized to its frame grid and keeps only the queried class, whereas the tool gates at the
model's patch resolution and clears background patches instead.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import torch

from checks.utils.models.cwip.visualization.world_model_utils import PatchSimilarityType


class PresenceSource(StrEnum):
    """How a two-stream run builds its presence masks, applied to both checks."""

    SIMILARITY = "similarity"
    """The similarity verdict: MATCH for correspondence, EXTRA for hallucination."""
    TYPE = "type"
    """The type-head mask for both checks, as in a camera-only run."""
    UNION = "union"
    """Each check's similarity label OR the type-head mask."""
    INTERSECTION = "intersection"
    """Each check's similarity label AND the type-head mask."""

    @property
    def uses_type_mask(self) -> bool:
        """Whether the masks depend on the type head (and so on its logits under the ``confidence`` gate)."""
        return self is not PresenceSource.SIMILARITY

    @property
    def includes_type_only_patches(self) -> bool:
        """Whether the hallucination mask can contain patches only the type-head mask marks present.

        Those patches are the small, short-lived clusters the type-head mask over-produces relative to the similarity
        mask; ``similarity`` never contains them and ``intersection`` keeps only patches the similarity head also
        marks.
        """
        return self in (PresenceSource.TYPE, PresenceSource.UNION)


class TypeMaskGate(StrEnum):
    """How the type-head presence mask is built."""

    CONFIDENCE = "confidence"
    """A patch is present when its top type probability is at least the (per-class) confidence threshold."""
    ARGMAX = "argmax"
    """Every patch is present; the downstream class filter alone selects the type. Needs no logits."""


# The presence source of a two-stream run unless configured otherwise.
DEFAULT_TWO_STREAM_PRESENCE_SOURCE = PresenceSource.UNION

# The effective presence source of a camera-only run, which has no similarity head.
CAMERA_ONLY_PRESENCE_SOURCE = PresenceSource.TYPE

DEFAULT_TYPE_CONFIDENCE_THRESHOLD = 0.5


@dataclass(frozen=True)
class TypeMaskConfig:
    """Settings of the type-head presence mask.

    Attributes:
        gate: How the mask is built.
        confidence_threshold: Minimum top type probability for a patch to be present under the ``confidence`` gate.
        class_thresholds: Per-class overrides of ``confidence_threshold``, keyed by model class index and applied to
            the patches whose predicted class is that index.
    """

    gate: TypeMaskGate = TypeMaskGate.CONFIDENCE
    confidence_threshold: float = DEFAULT_TYPE_CONFIDENCE_THRESHOLD
    class_thresholds: Mapping[int, float] = field(default_factory=dict)

    @property
    def needs_logits(self) -> bool:
        return self.gate is TypeMaskGate.CONFIDENCE


@dataclass(frozen=True)
class PresenceConfig:
    """The parsed presence settings of a run: the two-stream source and the type-head mask settings."""

    source: PresenceSource = DEFAULT_TWO_STREAM_PRESENCE_SOURCE
    type_mask: TypeMaskConfig = field(default_factory=TypeMaskConfig)

    def source_for_run(self, camera_only: bool) -> PresenceSource:
        """The source a run uses: ``source`` for two-stream, :data:`CAMERA_ONLY_PRESENCE_SOURCE` for camera-only."""
        return CAMERA_ONLY_PRESENCE_SOURCE if camera_only else self.source


def type_presence_mask(
    type_pred: torch.Tensor, type_logits: torch.Tensor | None, config: TypeMaskConfig
) -> torch.Tensor:
    """Build the type-head presence mask for one frame.

    Args:
        type_pred: ``(H, W)`` model class index per patch, the argmax of the type probabilities.
        type_logits: ``(C, h, w)`` raw type logits for the same frame, bilinearly resized to ``(H, W)``. Required by
            the ``confidence`` gate; unused by ``argmax``.
        config: The type-head mask settings.

    Returns:
        ``(H, W)`` long tensor of 0/1.

    Raises:
        ValueError: on the ``confidence`` gate without ``type_logits``.
    """
    if config.gate is TypeMaskGate.ARGMAX:
        return torch.ones_like(type_pred, dtype=torch.long)
    if type_logits is None:
        raise ValueError("the confidence gate requires the type logits")

    target_h, target_w = int(type_pred.shape[0]), int(type_pred.shape[1])
    frame_logits = torch.nn.functional.interpolate(
        type_logits.unsqueeze(0).float(), size=(target_h, target_w), mode="bilinear", align_corners=False
    )
    # amax over the class channel is the probability of the winning class, which is exactly type_pred when type_pred is
    # the argmax of the same sigmoid probabilities.
    top_prob = torch.sigmoid(frame_logits).squeeze(0).amax(dim=0)

    thr_map = torch.full(
        (target_h, target_w), float(config.confidence_threshold), dtype=top_prob.dtype, device=top_prob.device
    )
    for model_idx, class_thr in config.class_thresholds.items():
        thr_map[type_pred.to(top_prob.device) == model_idx] = float(class_thr)
    return (top_prob >= thr_map).to(torch.long).to(type_pred.device)


def presence_layers(
    similarity_pred: torch.Tensor | None,
    type_presence: Callable[[], torch.Tensor],
    source: PresenceSource,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``(correspondence, hallucination)`` presence masks for one frame.

    Args:
        similarity_pred: ``(H, W)`` :class:`PatchSimilarityType` value per patch, or ``None`` in a camera-only run.
        type_presence: Zero-argument callable returning the ``(H, W)`` type-head presence mask (e.g. a closure over
            :func:`type_presence_mask`). Called only when the mask is needed, so ``similarity`` never needs the type
            logits.
        source: The two-stream presence source; ignored in a camera-only run, where both checks use the type-head
            mask.

    Returns:
        Two ``(H, W)`` long tensors of 0/1, the correspondence and the hallucination mask. Camera-only: the type-head
        mask, twice. Two-stream: per ``source``, from the similarity label MATCH (correspondence) and EXTRA
        (hallucination), the type-head mask (both), or each label combined with the type-head mask by OR (``union``)
        or AND (``intersection``).

    Raises:
        ValueError: on a ``source`` that is not a :class:`PresenceSource` value.
    """
    source = PresenceSource(source)
    if similarity_pred is None:
        presence = type_presence()
        return presence, presence
    match = (similarity_pred == PatchSimilarityType.MATCH.value).to(torch.long)
    extra = (similarity_pred == PatchSimilarityType.EXTRA.value).to(torch.long)
    if source is PresenceSource.SIMILARITY:
        return match, extra
    presence = type_presence()
    if source is PresenceSource.TYPE:
        return presence, presence
    if source is PresenceSource.UNION:
        return match | presence, extra | presence
    return match & presence, extra & presence
