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

"""Static-check presence masks computed from CWIP outputs, and their visualization.

:func:`compute_presence_masks` builds the per-frame correspondence and hallucination presence masks with the shared
presence logic in :mod:`checks.utils.models.cwip.modules.presence`; :func:`write_presence_outputs` writes them, a
coverage summary, and an overlay video, so a presence setting can be inspected on a clip before it is used in the check.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from checks.utils.models.cwip.modules.class_mapping import (
    ObjectTypeClassMapping,
    get_model_class_index_for_object_type,
)
from checks.utils.models.cwip.modules.presence import (
    CAMERA_ONLY_PRESENCE_SOURCE,
    DEFAULT_TWO_STREAM_PRESENCE_SOURCE,
    DEFAULT_TYPE_CONFIDENCE_THRESHOLD,
    PresenceConfig,
    PresenceSource,
    TypeMaskConfig,
    TypeMaskGate,
    presence_layers,
    type_presence_mask,
)
from checks.utils.models.cwip.visualization import image_video_utils
from checks.utils.models.cwip.visualization.metrics_utils import compute_patch_predictions_and_percentages
from checks.utils.models.cwip.visualization.world_model_utils import ObjectType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PresenceOptions:
    """How :func:`compute_presence_masks` builds the static-check presence masks (``--save_presence_masks``).

    The fields mirror the static object check's ``presence`` config block, so a setting tried here carries over to the
    checker: ``presence_source`` is ``presence.source``; the remaining fields are the ``presence.type_mask`` settings
    (``gate``, ``confidence_threshold``, ``per_class_thresholds``), except that ``per_class_thresholds`` is keyed by
    ``ObjectType`` name (e.g. ``TRAFFIC_LIGHT_RED``) rather than by the checker's object class name.
    """

    presence_source: PresenceSource = DEFAULT_TWO_STREAM_PRESENCE_SOURCE
    type_mask_gate: TypeMaskGate = TypeMaskGate.CONFIDENCE
    type_confidence_threshold: float = DEFAULT_TYPE_CONFIDENCE_THRESHOLD
    per_class_thresholds: Mapping[str, float] = field(default_factory=dict)

    def to_config(self, mapping_cls: type[ObjectTypeClassMapping] | None) -> PresenceConfig:
        """The :class:`PresenceConfig` these options select, per-class thresholds resolved to model class indices."""
        return PresenceConfig(
            source=PresenceSource(self.presence_source),
            type_mask=TypeMaskConfig(
                gate=TypeMaskGate(self.type_mask_gate),
                confidence_threshold=self.type_confidence_threshold,
                class_thresholds=resolve_class_thresholds(self.per_class_thresholds, mapping_cls),
            ),
        )


def type_mask_label(options: PresenceOptions) -> str:
    """Describe the type-head mask settings, e.g. ``type_confidence>=0.6 (+2 per-class)`` or ``type_argmax``."""
    if options.type_mask_gate is TypeMaskGate.ARGMAX:
        return "type_argmax"
    label = f"type_confidence>={options.type_confidence_threshold:g}"
    if options.per_class_thresholds:
        label += f" (+{len(options.per_class_thresholds)} per-class)"
    return label


def presence_label(options: PresenceOptions, *, camera_only: bool) -> str:
    """Describe the presence settings in effect, for visualization labels.

    Camera-only always uses the type-head mask, so only its settings are named. Two-stream names the presence source,
    plus the type-head settings when the source uses the type-head mask.
    """
    type_desc = type_mask_label(options)
    if camera_only:
        return type_desc
    if options.presence_source is PresenceSource.SIMILARITY:
        return "presence_source=similarity"
    return f"presence_source={options.presence_source}, {type_desc}"


def resolve_class_thresholds(
    per_class_thresholds: Mapping[str, float],
    mapping_cls: type[ObjectTypeClassMapping] | None,
) -> dict[int, float]:
    """Map ``ObjectType``-name threshold overrides to model class indices.

    Raises:
        ValueError: if a name is not an ``ObjectType`` or the model's class mapping ignores it (the model never
            predicts it, so the override could never apply).
    """
    resolved: dict[int, float] = {}
    for name, threshold in per_class_thresholds.items():
        if name not in ObjectType.__members__:
            raise ValueError(f"unknown ObjectType {name!r} in per-class thresholds")
        if mapping_cls is None:
            model_idx: int | None = int(ObjectType[name])
        else:
            model_idx = get_model_class_index_for_object_type(name, mapping_cls)
        if model_idx is None:
            mapping_name = getattr(mapping_cls, "__name__", "the identity mapping")
            raise ValueError(f"ObjectType {name!r} is not a class the model predicts under {mapping_name}")
        resolved[model_idx] = float(threshold)
    return resolved


def compute_presence_masks(
    patch_similarity_logits: torch.Tensor | None,
    patch_type_logits: torch.Tensor | None,
    options: PresenceOptions,
    mapping_cls: type[ObjectTypeClassMapping] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the per-frame correspondence and hallucination presence masks the static object check scores against.

    Uses the same similarity labels (WRONG folded into MATCH), presence source, and type-head gate as the checker, at
    the model's patch resolution. The masks approximate the checker's rather than reproduce them: the checker gates the
    type logits after bilinearly resizing them to the video frame size, and keys per-class thresholds by object class
    name where ``options`` keys them by ``ObjectType``. Without similarity logits (camera-only) both masks are the
    type-head mask and ``options.presence_source`` is ignored, as in the checker.

    When the type head ran, patches it predicts as background are cleared. The checker keeps only the patches whose
    predicted type is the class being checked, so a background patch never counts, but the type-head mask marks
    confident background (sky, road surface) present; left in, it would cover most of the frame.

    Args:
        patch_similarity_logits: ``(N, C_sim, h, w)`` similarity-head logits, or ``None`` in camera-only mode.
        patch_type_logits: ``(N, C_type, h, w)`` type-head logits, or ``None`` when the model has no type head.
        options: Presence settings.
        mapping_cls: The model's object-type class mapping, used to resolve ``options.per_class_thresholds``.

    Returns:
        ``(correspondence, hallucination)``, each an ``(N, h, w)`` ``uint8`` array of 0/1.

    Raises:
        ValueError: if neither head produced logits, or the selected source needs the type head and it is absent.
    """
    if patch_similarity_logits is None and patch_type_logits is None:
        raise ValueError("presence masks need the similarity or the type head output; the model produced neither")
    needs_type = patch_similarity_logits is None or PresenceSource(options.presence_source).uses_type_mask
    if needs_type and patch_type_logits is None:
        raise ValueError(
            f"presence_source={options.presence_source!r} needs the type head, which this model does not have"
        )

    type_pred: torch.Tensor | None = None
    if patch_type_logits is not None:
        type_pred = torch.sigmoid(patch_type_logits.float()).argmax(dim=1)
    similarity_pred: torch.Tensor | None = None
    if patch_similarity_logits is not None:
        similarity_pred = compute_patch_predictions_and_percentages(
            patch_similarity_logits, treat_wrong_type_as_match=True
        )["pred_class_indices"]
        if type_pred is not None and similarity_pred.shape[-2:] != type_pred.shape[-2:]:
            similarity_pred = (
                torch.nn.functional.interpolate(
                    similarity_pred.unsqueeze(1).float(), size=tuple(type_pred.shape[-2:]), mode="nearest"
                )
                .squeeze(1)
                .long()
            )

    config = options.to_config(mapping_cls)
    background_idx = (
        int(ObjectType.BACKGROUND)
        if mapping_cls is None
        else get_model_class_index_for_object_type(ObjectType.BACKGROUND, mapping_cls)
    )
    reference = similarity_pred if similarity_pred is not None else type_pred
    assert reference is not None
    num_frames = int(reference.shape[0])
    correspondence, hallucination = [], []
    for i in range(num_frames):

        def _type_presence(i: int = i) -> torch.Tensor:
            assert type_pred is not None and patch_type_logits is not None
            return type_presence_mask(type_pred[i], patch_type_logits[i], config.type_mask)

        match_mask, extra_mask = presence_layers(
            similarity_pred[i] if similarity_pred is not None else None, _type_presence, config.source
        )
        if type_pred is not None and background_idx is not None:
            foreground = (type_pred[i] != background_idx).to(torch.long)
            match_mask, extra_mask = match_mask & foreground, extra_mask & foreground
        correspondence.append(match_mask.cpu().numpy().astype(np.uint8))
        hallucination.append(extra_mask.cpu().numpy().astype(np.uint8))
    return np.stack(correspondence), np.stack(hallucination)


_GREEN = (0, 255, 0)
_RED = (255, 0, 0)
_CYAN = (0, 200, 255)


def _draw_legend(panel: np.ndarray, title: str, entries: list[tuple[tuple[int, int, int], str]]) -> None:
    """Draw a legend (a title, then one color swatch + label per entry) in the panel's top-left corner, in place."""
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    pad, line_h, swatch = 8, 20, 12
    widths = [cv2.getTextSize(title, font, scale, thickness)[0][0]]
    widths += [cv2.getTextSize(text, font, scale, thickness)[0][0] + swatch + pad for _, text in entries]
    box_w = min(max(widths) + 2 * pad, panel.shape[1])
    box_h = min(2 * pad + line_h * (1 + len(entries)), panel.shape[0])
    # Darken the box so the legend reads over any frame content.
    panel[:box_h, :box_w] = (panel[:box_h, :box_w] * 0.35).astype(np.uint8)
    cv2.putText(panel, title, (pad, pad + 13), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    for k, (color, text) in enumerate(entries, start=1):
        y = pad + line_h * k
        cv2.rectangle(panel, (pad, y + 2), (pad + swatch, y + 2 + swatch), color, thickness=-1)
        cv2.putText(panel, text, (2 * pad + swatch, y + 13), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def _tint(frame: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    """Blend ``color`` 50/50 into the ``mask`` pixels of ``frame``, in place."""
    frame[mask] = (0.5 * frame[mask] + 0.5 * np.array(color)).astype(np.uint8)


@dataclass(frozen=True)
class ComparedMask:
    """One of the two presence masks compared in the presence visualization's right column."""

    key: str  # Stem of its ``.npz`` / summary keys.
    label: str  # Legend text.
    mask: np.ndarray  # (N, h, w) uint8 0/1.


def compare_masks(
    patch_similarity_logits: torch.Tensor | None,
    patch_type_logits: torch.Tensor | None,
    options: PresenceOptions,
    mapping_cls: type[ObjectTypeClassMapping] | None,
) -> tuple[ComparedMask, ComparedMask] | None:
    """Build the two presence masks the presence visualization contrasts, or ``None`` without a type head.

    Two-stream: the similarity mask (the ``similarity`` source) against the type-head mask under ``options``' type-head
    settings, i.e. the two inputs the ``union`` / ``intersection`` sources combine. Camera-only, which has no similarity
    head: the type-head mask at the default settings against the one under ``options``, i.e. what the type-head flags
    change. Both are presence masks (correspondence OR hallucination) with background cleared, as in
    :func:`compute_presence_masks`. A model without a type head (similarity logits only) has no type-head mask to
    contrast, so the result is ``None``.
    """
    if patch_type_logits is None:
        return None

    def _presence(opts: PresenceOptions) -> np.ndarray:
        return np.maximum(*compute_presence_masks(patch_similarity_logits, patch_type_logits, opts, mapping_cls))

    type_options = replace(options, presence_source=PresenceSource.TYPE)
    selected_type = ComparedMask("type", type_mask_label(options), _presence(type_options))
    if patch_similarity_logits is None:
        default = PresenceOptions()
        return ComparedMask("default_type", f"{type_mask_label(default)} (default)", _presence(default)), selected_type
    similarity = _presence(PresenceOptions(presence_source=PresenceSource.SIMILARITY))
    return ComparedMask("similarity", "similarity", similarity), selected_type


_MAGENTA = (255, 0, 255)
_YELLOW = (255, 220, 0)


def _presence_overlay_frames(
    camera_video: torch.Tensor,
    presence: np.ndarray,
    compared: tuple[ComparedMask, ComparedMask] | None,
    panel_width: int,
    presence_desc: str,
) -> list[np.ndarray]:
    """Render the presence visualization as a 2x2 grid, one RGB ``uint8`` frame per video frame.

    Left column: the presence mask under the selected settings (cyan), tinted onto the camera (top) and alone on black
    (bottom). Right column: the two :class:`ComparedMask` masks, green where only the first is present, magenta where
    only the second is, and yellow where both are, tinted onto the camera (top) and alone on black (bottom). Without
    ``compared`` (no type head) only the left column is rendered. Each panel carries a legend in its top-left corner.
    """
    _, _, cam_h, cam_w = camera_video.shape
    panel_w = min(panel_width, cam_w) // 2 * 2
    panel_h = round(cam_h * panel_w / cam_w) // 2 * 2

    def _upsample(mask: np.ndarray) -> np.ndarray:
        resized = torch.nn.functional.interpolate(
            torch.from_numpy(mask)[None, None].float(), size=(panel_h, panel_w), mode="nearest"
        )
        return resized[0, 0].numpy().astype(bool)

    presence_entries = [(_CYAN, "present")]
    frames = []
    for i in range(camera_video.shape[0]):
        cam = torch.nn.functional.interpolate(
            camera_video[i : i + 1].float(), size=(panel_h, panel_w), mode="bilinear", align_corners=False
        )
        cam_np = image_video_utils.to_uint8(cam.squeeze(0).permute(1, 2, 0).numpy())
        selected = _upsample(presence[i])
        presence_overlay = cam_np.copy()
        _tint(presence_overlay, selected, _CYAN)
        _draw_legend(presence_overlay, f"Presence: {presence_desc}", presence_entries)
        presence_panel = np.zeros_like(cam_np)
        presence_panel[selected] = _CYAN
        _draw_legend(presence_panel, "Presence mask", presence_entries)
        if compared is None:
            frames.append(np.concatenate([presence_overlay, presence_panel], axis=0))
            continue

        first, second = compared
        diff_entries = [(_GREEN, f"only {first.label}"), (_MAGENTA, f"only {second.label}"), (_YELLOW, "both")]
        first_mask, second_mask = _upsample(first.mask[i]), _upsample(second.mask[i])
        first_only, second_only = first_mask & ~second_mask, second_mask & ~first_mask
        both = first_mask & second_mask
        diff_overlay = cam_np.copy()
        _tint(diff_overlay, first_only, _GREEN)
        _tint(diff_overlay, second_only, _MAGENTA)
        _tint(diff_overlay, both, _YELLOW)
        _draw_legend(diff_overlay, f"{first.label} vs {second.label}", diff_entries)
        diff_panel = np.zeros_like(cam_np)
        diff_panel[first_only] = _GREEN
        diff_panel[second_only] = _MAGENTA
        diff_panel[both] = _YELLOW
        _draw_legend(diff_panel, f"{first.label} vs {second.label} mask", diff_entries)

        top = np.concatenate([presence_overlay, diff_overlay], axis=1)
        bottom = np.concatenate([presence_panel, diff_panel], axis=1)
        frames.append(np.concatenate([top, bottom], axis=0))
    return frames


def _fractions(mask: np.ndarray) -> np.ndarray:
    """Per-frame fraction of patches set in an ``(N, h, w)`` 0/1 mask."""
    return mask.reshape(len(mask), -1).mean(axis=1) if len(mask) else np.zeros(0)


def write_presence_outputs(
    output_dir: Path,
    camera_video: torch.Tensor,
    correspondence: np.ndarray,
    hallucination: np.ndarray,
    options: PresenceOptions,
    *,
    compared: tuple[ComparedMask, ComparedMask] | None,
    camera_only: bool,
    fps: float,
    iteration: int = 0,
    panel_width: int = 960,
) -> None:
    """Write the presence masks, a coverage summary, and a visualization video to ``output_dir``.

    The presence mask is the union of the correspondence and hallucination masks. ``compared`` are the two masks the
    visualization contrasts (see :func:`compare_masks`); ``None`` (no type head) omits them from every output.

    Files (``NNNNNN`` = ``iteration``):

    - ``presence_masks_iter_NNNNNN.npz``: ``(N, h, w)`` ``uint8`` masks: ``presence``, ``correspondence``,
      ``hallucination``, and ``<key>_presence`` for each compared mask.
    - ``presence_summary_iter_NNNNNN.json``: the settings used and the fraction of patches, per frame and on average, in
      ``presence``, each compared mask (``<key>_presence``), each compared mask alone (``<key>_only``), and both
      (``both``).
    - ``presence_mask_vis_iter_NNNNNN.mp4``: see :func:`_presence_overlay_frames`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"iter_{iteration:06d}"
    presence = np.maximum(correspondence, hallucination)
    arrays = {"presence": presence, "correspondence": correspondence, "hallucination": hallucination}
    series = {"presence": _fractions(presence)}
    compared_labels: dict[str, str] | None = None
    if compared is not None:
        first, second = compared
        arrays |= {f"{first.key}_presence": first.mask, f"{second.key}_presence": second.mask}
        series |= {
            f"{first.key}_presence": _fractions(first.mask),
            f"{second.key}_presence": _fractions(second.mask),
            f"{first.key}_only": _fractions(first.mask & (1 - second.mask)),
            f"{second.key}_only": _fractions(second.mask & (1 - first.mask)),
            "both": _fractions(first.mask & second.mask),
        }
        compared_labels = {first.key: first.label, second.key: second.label}
    np.savez_compressed(output_dir / f"presence_masks_{suffix}.npz", **arrays)

    presence_desc = presence_label(options, camera_only=camera_only)
    summary_json = {
        "mode": "camera_only" if camera_only else "two_stream",
        # Camera-only has no similarity head: the mask is the type-head mask regardless of presence_source.
        "presence_source": str(CAMERA_ONLY_PRESENCE_SOURCE if camera_only else options.presence_source),
        "presence": presence_desc,
        "compared": compared_labels,
        "type_mask_gate": str(options.type_mask_gate),
        "type_confidence_threshold": options.type_confidence_threshold,
        "per_class_thresholds": dict(options.per_class_thresholds),
        "mask_shape": list(presence.shape[1:]),
        "mean_fraction_present": {k: float(v.mean()) if len(v) else 0.0 for k, v in series.items()},
        "per_frame_fraction_present": {k: [round(float(x), 6) for x in v] for k, v in series.items()},
    }
    (output_dir / f"presence_summary_{suffix}.json").write_text(json.dumps(summary_json, indent=2))

    frames = _presence_overlay_frames(camera_video, presence, compared, panel_width, presence_desc)
    if frames:
        out_h, out_w = frames[0].shape[:2]
        image_video_utils.write_video_with_ffmpeg(
            frames, str(output_dir / f"presence_mask_vis_{suffix}.mp4"), out_w, out_h, fps
        )
    logger.info("  Presence masks written to %s (%s)", output_dir, suffix)
