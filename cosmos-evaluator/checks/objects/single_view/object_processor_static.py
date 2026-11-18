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
Static objects processor for CWIP-based video-pair processing.

This module implements the static (CWIP-based) objects processor, which processes produces segmentation masks from the
CWIP model and checks the results against projected static world model objects.
"""

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
from third_party.cosmos_drive_dreams_toolkits.utils.camera.base import CameraBase
import torch

from checks.objects.single_view.cwip_helper import (
    COSMOS_EVALUATOR_OBJECT_TYPE_STRING_TO_ENUMS,
    CWIPInferenceHelper,
    resize_index_map,
    resolve_model_class_indices,
)
from checks.objects.single_view.cwip_inference_cache import CwipInferenceCache
from checks.objects.single_view.hallucination_detector import HallucinationDetector
from checks.objects.single_view.hallucination_tracking import (
    confirm_hallucination_tracks,
    regroup_ground_tracks,
    track_hallucinations,
)
from checks.objects.single_view.object_processor_base import ObjectProcessorBase
from checks.objects.single_view.overlap_detector import OverlapDetector
from checks.objects.single_view.traffic_light_state_checker import TrafficLightStateChecker
from checks.utils.frame_rates import OBJECT_CHECK_FPS, ObjectCheckTimeline, validate_annotation_axis
from checks.utils.lane_style import annotate_lane_styles
from checks.utils.models.cwip.modules.class_mapping import get_object_type_class_mapping
from checks.utils.models.cwip.modules.presence import (
    DEFAULT_TWO_STREAM_PRESENCE_SOURCE,
    DEFAULT_TYPE_CONFIDENCE_THRESHOLD,
    PresenceConfig,
    PresenceSource,
    TypeMaskConfig,
    TypeMaskGate,
    presence_layers,
    type_presence_mask,
)
from checks.utils.models.cwip.tools.batch_utils import merge_batch_outputs
from checks.utils.models.cwip.visualization import image_video_utils as cwip_image_video
from checks.utils.models.cwip.visualization.metrics_utils import compute_patch_predictions_and_percentages
from checks.utils.models.cwip.visualization.world_model_utils import (
    ObjectType,
    PatchSimilarityType,
    PatchSimilarityType2Color,
    get_object_type_colors,
)
from checks.utils.object_class_names import class_name_to_rds_label
from checks.utils.onnx import clear_gpu_memory
from checks.utils.rds_data_loader import RdsDataLoader
from checks.utils.scene_rasterizer import SceneRasterizer
from checks.utils.video import get_video_fps, get_video_frame_count
from utils.bazel import get_runfiles_path
from utils.cancellation import check_cancelled


def load_cwip_model(device: str) -> CWIPInferenceHelper:
    """Load the bundled CWIP model onto ``device``.

    Exposed as a module-level function so a caller can load the model **once** and inject the returned instance into
    ``ObjectProcessorStatic(seg_model=...)`` to reuse it across clips instead of reloading it on every instance.
    """
    cwip_model_path = get_runfiles_path("checks/utils/models/cwip/checkpoints/model.safetensors")
    if cwip_model_path is None:
        raise RuntimeError("CWIP model path not found")
    return CWIPInferenceHelper.from_hf_format(model_path=str(Path(cwip_model_path).parent), device=device)


def summarize_cwip_output(
    pred_similarity_indices: torch.Tensor | None,
    pred_type_indices: torch.Tensor | None,
    raw_type_logits: torch.Tensor | None,
    top_k: int = 3,
) -> dict[str, np.ndarray]:
    """Compact, debug-oriented summary of the raw CWIP output (shrinks the ~GB/clip float dump ~50-100x).

    Keeps what offline debugging needs while dropping the bulk of the float logits:
    - Index maps (argmax class per pixel, both heads) downcast to uint8 — both heads have < 256 classes, so this
      is lossless for the segmentation the checks consume.
    - The type head's per-patch distribution reduced to its ``top_k`` classes as parallel
      ``type_top_idx`` (class index, uint8) and ``type_top_prob_pct`` (sigmoid probability * 100, uint8) arrays,
      shaped ``(frames, top_k, Hp, Wp)``. Probabilities use ``sigmoid`` (matching how the head is read:
      ``sigmoid(logits).argmax``) rounded to whole percent — the "2 decimal places" resolution. This preserves the
      winner, runners-up, and confidence (e.g. a TrafficLight@0.47 vs TrafficSign@0.45 near-tie) at a fraction of
      the size of the full ``(frames, num_classes, Hp, Wp)`` float logits.
    """
    out: dict[str, np.ndarray] = {}
    if pred_similarity_indices is not None:
        out["pred_similarity_indices"] = pred_similarity_indices.to(torch.uint8).cpu().numpy()
    if pred_type_indices is not None:
        out["pred_type_indices"] = pred_type_indices.to(torch.uint8).cpu().numpy()
    if raw_type_logits is not None and raw_type_logits.dim() >= 2:
        # sigmoid is monotonic, so take topk on the raw logits first and sigmoid only the selected top-k values —
        # this avoids materializing a second full (frames, num_classes, Hp, Wp) float volume for long clips.
        k = min(top_k, raw_type_logits.shape[1])
        top_logit, top_idx = raw_type_logits.topk(k, dim=1)
        top_prob = torch.sigmoid(top_logit.float())
        out["type_top_idx"] = top_idx.to(torch.uint8).cpu().numpy()
        out["type_top_prob_pct"] = (top_prob * 100).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    return out


# The only CWIP ``forward`` outputs the static pipeline consumes downstream are the patch-resolution logits. The
# encoder spatial-feature maps and per-layer feature lists (``camera/world_spatial_features[_list]``, 768-channel
# tensors over the full patch grid for every frame) are also returned but unused here; accumulating them across all
# batches of a long clip is tens of GB and is what OOMs CPU memory. Keep only these keys per batch.
_CWIP_OUTPUT_KEYS_USED = ("patch_similarity_logits", "patch_type_logits")


# Debug-dump file holding the run's type-head presence masks, which stand in for the type logits the compact dump omits.
TYPE_PRESENCE_DUMP = "type_presence.npz"


@dataclass(frozen=True)
class HallucinationGates:
    """Per-class hallucination-tracker gates in effect for one run.

    Attributes:
        min_cluster_area: Minimum cluster area (pixels) by object-config class name, for the classes whose area differs
            from the hallucination detector's configured value.
        min_track_frames: Persistence gate (frames) by RDS label, for every tracked class.
    """

    min_cluster_area: Mapping[str, int]
    min_track_frames: Mapping[str, int]


def extract_used_cwip_outputs(output_batch: dict[str, Any]) -> dict[str, Any]:
    """Keep only the downstream-consumed CWIP outputs, moved to CPU.

    The static pipeline reads only the patch-resolution similarity/type logits (``_CWIP_OUTPUT_KEYS_USED``). The
    encoder spatial-feature maps and per-layer feature lists returned by ``forward`` are large (768-channel tensors
    at the full patch grid, for every frame) and unused here; accumulating them across every batch of a long clip
    exhausts CPU memory. Dropping them keeps only a few MB/frame of logits. Keys are preserved (with their original
    value, including ``None``) so downstream membership and ``.get(...)`` checks behave identically to the full
    output.

    Args:
        output_batch: A single batch's ``forward`` output dict.

    Returns:
        A new dict containing only the used keys, with tensors moved to CPU.
    """
    pruned: dict[str, Any] = {}
    for key in _CWIP_OUTPUT_KEYS_USED:
        value = output_batch.get(key)
        pruned[key] = value.cpu() if isinstance(value, torch.Tensor) else value
    return pruned


def dump_cwip_raw(
    debug_dir: Path,
    *,
    dump_full_logits: bool,
    pred_similarity_indices: torch.Tensor | None,
    pred_type_indices: torch.Tensor | None,
    raw_type_logits: torch.Tensor | None,
    summary: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Persist the CWIP model output under ``debug_dir`` for offline debugging of any static check.

    Every static check (correspondence, missing-object, traffic-light-state, hallucination) is derived from the CWIP
    output, so this artifact is check-agnostic. Two tiers, selected by ``dump_full_logits``:

    - ``dump_full_logits`` false: a compact ``cwip_compact.npz`` from ``summarize_cwip_output`` — uint8 index maps +
      the type head's top-3 (class, sigmoid-prob%) per patch — plus ``cwip_summary.json``. ~tens of MB/clip.
    - ``dump_full_logits`` true: the full-fidelity ``cwip_inference.pt`` (same schema as ``CwipInferenceCache``:
      index maps + raw float type logits + summary). Order GB/clip; use only when patch-resolution top-k
      probabilities are not enough.

    Callers must invoke this only when a debug dir is configured; ``debug_dir`` is a resolved path, never ``None``.
    Best-effort: a write failure is logged, never fatal.
    """
    try:
        if dump_full_logits:
            torch.save(
                {
                    "summary": summary,
                    "pred_similarity_indices": pred_similarity_indices,
                    "pred_type_indices": pred_type_indices,
                    "raw_type_logits": raw_type_logits,
                },
                debug_dir / "cwip_inference.pt",
            )
            return
        arrays = summarize_cwip_output(pred_similarity_indices, pred_type_indices, raw_type_logits)
        # allow_pickle=False keeps these dumps to plain numeric arrays (no pickled objects, safe to load) and, by
        # pinning the bool kwarg explicitly, lets the type checker resolve the remaining ``**arrays`` as archive
        # entries rather than a candidate value for ``allow_pickle``.
        np.savez_compressed(
            debug_dir / "cwip_compact.npz",
            allow_pickle=False,
            **arrays,
        )
        with open(debug_dir / "cwip_summary.json", "w") as fh:
            json.dump(summary, fh, default=str)
    except Exception as exc:  # noqa: BLE001 - debug dump is best-effort; never abort processing
        logger.warning("Failed to dump CWIP output: %s", exc)


class ObjectProcessorStatic(ObjectProcessorBase):
    """Implements static (CWIP-based) video-pair processing."""

    # Maximum allowed relative difference between the camera and world (control) video aspect ratios. The legitimate
    # control video matches the camera view (e.g. 1280x704 ~ 1.82 vs camera 1280x720 ~ 1.78, a ~2% difference), while
    # a top-down BEV is square (1.0, a ~44% difference), so a 10% tolerance cleanly separates a valid control video
    # from a wrong-view one.
    _MAX_WORLD_CAMERA_ASPECT_RATIO_DIFF = 0.10

    def __init__(
        self,
        clip_id: str,
        output_dir: str,
        config: dict[str, Any] | None = None,
        model_device: str = "cuda",
        verbose: str = "INFO",
        seg_model: Any = None,
    ) -> None:
        """Initialize static (CWIP-based) processing components.

        Args:
            clip_id: Clip identifier for logging/visualization file names.
            output_dir: Directory where outputs (logs, videos) are written.
            config: Processor configuration dictionary.
            model_device: Device for models, e.g. 'cuda' or 'cpu'.
            verbose: Logging verbosity level.
            seg_model: Optional pre-loaded CWIP model (see ``load_cwip_model``) to reuse instead of loading one per
                instance. When ``None`` (the default), the model is loaded from the bundled checkpoint.
        """
        super().__init__(clip_id, output_dir, config, model_device, verbose)
        self._injected_seg_model = seg_model
        self._init_static_components()

    def _init_static_components(self) -> None:
        """Initialize CWIP model, overlap detector, hallucination detector, and visualization styles."""
        self.naming_suffix = "static"
        self.setup_logging()
        if self._injected_seg_model is not None:
            # Reuse the injected model directly — do not touch the runfiles/checkpoint path at all.
            self.seg_model = self._injected_seg_model
        else:
            self.seg_model = load_cwip_model(self.model_device)
        self.logger.info(
            "Loaded CWIP model: precision=%s, image_size=%sx%s",
            getattr(self.seg_model, "precision", "unknown"),
            getattr(self.seg_model, "image_size", ("?", "?"))[0],
            getattr(self.seg_model, "image_size", ("?", "?"))[1],
        )
        self.overlap_detector = OverlapDetector(
            self.oc_config, self.seg_model, self.logger, debug_enabled=(self.verbose == "DEBUG")
        )

        # Frame-edge truncation gate for the traffic-light STATE check, read from its own
        # ``overlap_check.traffic_light.state_min_in_frame_fraction`` key. Deliberately separate from the
        # class-agnostic ``min_in_frame_fraction`` that gates correspondence: truncation is far more damaging to
        # reading bulb colors (a red bulb cut off at the frame edge can read as green) than to deciding whether a
        # box is present at all, and correspondence *is* the existence verdict downstream, so gating the two on
        # one number would move existence in service of the state check. Resolved once here rather than per frame
        # (the checker is constructed per frame). Absent or 0 disables, as elsewhere; no fallback to the
        # correspondence key, so the effective threshold is always readable from the config.
        tl_cfg = (self.oc_config.get("overlap_check", {}) or {}).get("traffic_light", {}) or {}
        self._tl_state_min_in_frame_fraction = 0.0
        if isinstance(tl_cfg, dict):
            with suppress(TypeError, ValueError):
                self._tl_state_min_in_frame_fraction = float(tl_cfg.get("state_min_in_frame_fraction", 0.0) or 0.0)

        # Opt-in offline debug dump: when the objects ``debug_dump`` flag is set, persist raw + post-processing
        # artifacts under ``<output_dir>/debug`` (uploaded to S3 in service mode) so ANY static check —
        # correspondence, missing-object, traffic-light-state, hallucination — can be debugged offline. Off by
        # default: production runs pay nothing.
        self._debug_dir: Path | None = (
            Path(self.output_dir) / "debug" if bool(self.oc_config.get("debug_dump", False)) else None
        )
        if self._debug_dir:
            self._debug_dir.mkdir(parents=True, exist_ok=True)

        # Replay CWIP output from a prior debug dump instead of running inference. Loads the similarity +
        # type index maps from ``<dir>/cwip_compact.npz`` (+ ``cwip_summary.json``), plus the type-head presence masks
        # the dumped run computed, which stand in for the logits under the confidence gate. raw_type_logits is not
        # dumped compactly, so it is returned as None (TL-state, which needs it, is skipped in replay rather than fed
        # misleading zero logits).
        self._cwip_debug_dump_dir: str | None = self.oc_config.get("cwip_debug_dump_dir") or None

        self._parse_presence_config()

        # Initialize hallucination detector if enabled, filtered to static object types only
        self.hallucination_detector = None
        halluc_cfg = self.oc_config.get("hallucination_detector", {})
        if isinstance(halluc_cfg, dict) and bool(halluc_cfg.get("enabled", False)):
            try:
                filtered_classes = {k: v for k, v in halluc_cfg.get("classes", {}).items() if k in self.STATIC_OBJECTS}
                filtered_cfg = {**halluc_cfg, "classes": filtered_classes}
                self._hallucination_classes = filtered_classes
                # Per-class tracker knobs, keyed by the RDS label used on tracks (e.g. "TrafficLight"). Defaults keep
                # the shipping behavior: no proximity association (0) and no persistence gate (1 frame).
                self._hallucination_proximity_by_class = {
                    class_name_to_rds_label(name): float((cfg or {}).get("association_proximity_px", 0.0))
                    for name, cfg in filtered_classes.items()
                }
                self._hallucination_min_frames_by_class = {
                    class_name_to_rds_label(name): int((cfg or {}).get("min_track_frames", 1))
                    for name, cfg in filtered_classes.items()
                }
                # Classes tracked by grouping their ground-plane detections instead of per-frame bbox overlap.
                # A batch pass, so it runs once the clip is done rather than per frame.
                self._ground_tracked_classes = [
                    class_name_to_rds_label(name)
                    for name, cfg in filtered_classes.items()
                    if str((cfg or {}).get("tracker", "")).lower() == "bev_ground"
                ]
                self._ground_tracker_resolution_m = float(halluc_cfg.get("bev_resolution_m", 0.1))
                if self._ground_tracked_classes and self._ground_tracker_resolution_m <= 0:
                    # Caught here rather than in the regrouping pass, which runs after the whole frame loop —
                    # a bad grid resolution would otherwise fail the run only once the expensive part is done.
                    raise ValueError(
                        f"hallucination_detector.bev_resolution_m must be > 0, got {self._ground_tracker_resolution_m}"
                    )
                self._ground_tracker_thickness_px = int(halluc_cfg.get("bev_thickness_px", 7))
                # Hallucination masks (gt / class / unmatched) are one component of the debug dump.
                mask_debug_dir = self._debug_dir / "hallucination_masks" if self._debug_dir else None
                self.hallucination_detector = HallucinationDetector(
                    filtered_cfg, self.seg_model, self.logger, mask_debug_dir=mask_debug_dir
                )
                self.logger.info("Hallucination detector enabled")
                self._hallucination_summary: dict[str, Any] = {}
            except Exception as e:  # noqa: BLE001 - detector init is best-effort; log and disable
                self.logger.warning("Failed to initialize HallucinationDetector: %s", e)
                self.hallucination_detector = None

        self.class_colors = []
        for similarity_type in PatchSimilarityType:
            rgb = PatchSimilarityType2Color[similarity_type]
            self.class_colors.append((int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))
        class_names = [similarity_type.name for similarity_type in PatchSimilarityType]
        color_map = {name: color for name, color in zip(class_names, self.class_colors)}
        legend_items = list(zip(class_names, self.class_colors))
        if self.vis_helper:
            self.vis_helper.set_mask_styles(mask_colors=color_map, legend_items=legend_items, mask_name="similarity")

        self._init_type_mask_colors()

    def _init_type_mask_colors(self) -> None:
        """Build a color LUT for type mask visualization from the model's class mapping."""
        mapping_cls = get_object_type_class_mapping(getattr(self.seg_model.config, "object_type_class_mapping", None))
        obj_type_colors_float = get_object_type_colors()

        type_class_colors: dict[int, tuple[int, int, int]] = {}
        type_legend_items: list[tuple[str, tuple[int, int, int]]] = []

        # get_object_type_class_mapping always returns a class (defaults for None, raises on unknown), so mapping_cls is
        # never None here.
        for model_idx, obj_type in sorted(mapping_cls.MODEL_CLASS_TO_OBJECT_TYPE.items()):
            rgb_f = obj_type_colors_float.get(obj_type, [0.5, 0.5, 0.5])
            rgb = (int(rgb_f[0] * 255), int(rgb_f[1] * 255), int(rgb_f[2] * 255))
            type_class_colors[model_idx] = rgb
            if obj_type != ObjectType.BACKGROUND:
                type_legend_items.append((obj_type.name, rgb))

        max_idx = max(type_class_colors.keys(), default=0) + 1
        self.type_color_lut = np.zeros((max_idx, 3), dtype=np.uint8)
        for idx, color in type_class_colors.items():
            self.type_color_lut[idx] = color

        if self.vis_helper:
            self.vis_helper.set_mask_styles(legend_items=type_legend_items, mask_name="type")
            self.vis_helper.set_mask_styles(legend_items=type_legend_items, mask_name="type_filtered")

    def _build_type_mask_frame(
        self,
        pred_type_indices: torch.Tensor,
        frame_idx: int,
        presence: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Render a single frame of the type mask as an RGB image using the precomputed LUT.

        When *presence* (an (H, W) 0/1 mask) is provided, only its pixels are colored and all other pixels stay black.
        The caller passes the presence layer the static checks score against (the correspondence OR hallucination
        layer under the configured presence source), so the filtered view shows what is scored.
        """
        type_indices_np = pred_type_indices[frame_idx].numpy()
        clamped = np.clip(type_indices_np, 0, self.type_color_lut.shape[0] - 1)
        rgb = self.type_color_lut[clamped]

        if presence is not None:
            rgb[~presence.to(torch.bool).numpy()] = 0

        return rgb

    def _validate_world_camera_aspect_ratio(self, camera_hw: tuple[int, int], world_hw: tuple[int, int]) -> None:
        """Reject a world-model (control) video whose aspect ratio doesn't match the camera.

        CWIP compares the camera and world videos as the same perspective view; a mismatched-view world video
        (most notably the top-down BEV) silently yields no MATCH and ~0 static scores. Raises ``ValueError`` when
        the relative aspect-ratio difference exceeds :attr:`_MAX_WORLD_CAMERA_ASPECT_RATIO_DIFF` (REST maps this to
        422, CLI exits 1).

        Args:
            camera_hw: ``(height, width)`` of the camera (augmented) video.
            world_hw: ``(height, width)`` of the world-model (control) video.
        """
        cam_h, cam_w = camera_hw
        world_h, world_w = world_hw
        if min(cam_h, cam_w, world_h, world_w) <= 0:
            raise ValueError(f"Invalid video dimensions: camera={camera_hw}, world={world_hw}")
        cam_ar = cam_w / cam_h
        world_ar = world_w / world_h
        if abs(cam_ar - world_ar) / cam_ar > self._MAX_WORLD_CAMERA_ASPECT_RATIO_DIFF:
            raise ValueError(
                f"World/control video aspect ratio {world_ar:.3f} ({world_w}x{world_h}) does not match camera aspect "
                f"ratio {cam_ar:.3f} ({cam_w}x{cam_h}). The world-model (control) video must be the same perspective "
                f"view as the camera; this looks like a top-down BEV or otherwise wrong video, which would cause CWIP "
                f"to find no match and collapse all static scores to ~0."
            )

    def _load_cwip_from_dump(
        self, dump_dir: str
    ) -> tuple[dict[str, Any], torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Load CWIP similarity/type index maps from a debug dump instead of running inference.

        Reads ``cwip_compact.npz`` (``pred_similarity_indices``, ``pred_type_indices``, ``type_top_idx``) and
        ``cwip_summary.json``. ``raw_type_logits`` is not stored compactly, so it is returned as ``None``: the
        correspondence and hallucination checks consume only the index maps and stay exact, while TL-state (which
        needs the logits) is skipped explicitly downstream rather than being fed zero logits that would mislead.

        A camera-only dump has no similarity head, so ``pred_similarity_indices`` is absent from the archive and is
        returned as ``None``; the frame count then comes from the type-head map, which is present in both modes.
        """
        with open(os.path.join(dump_dir, "cwip_summary.json")) as fh:
            summary: dict[str, Any] = json.load(fh)
        with np.load(os.path.join(dump_dir, "cwip_compact.npz")) as cw:
            pred_similarity_indices = (
                torch.from_numpy(cw["pred_similarity_indices"]).long() if "pred_similarity_indices" in cw else None
            )
            pred_type_indices = torch.from_numpy(cw["pred_type_indices"]).long()
        # Frame count comes from whichever index map is present (both carry the same frame dim); the type-head map is
        # always present, so it is the reliable source in camera-only dumps that lack the similarity map.
        num_frames = int(pred_type_indices.shape[0])
        self.logger.info("Replay: loaded %d frames of CWIP indices from %s", num_frames, dump_dir)
        return summary, pred_similarity_indices, pred_type_indices, None

    def _run_cwip_inference_on_videos(
        self,
        camera_video_path: str,
        world_video_path: str | None,
        num_frames: int = -1,
        sample_method: str = "uniform",
        cancel_event: threading.Event | None = None,
        camera_only: bool = False,
    ) -> tuple[dict[str, Any], torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Run CWIP inference on a camera/world video pair and summarize outputs.

        Args:
            camera_video_path (str): Path to the camera-facing video.
            world_video_path (str | None): Path to the outward/world video. ``None`` only when ``camera_only`` is set.
            num_frames (int): Number of frames to sample. Defaults to -1 (all frames).
            sample_method (str): Frame sampling method (e.g. 'uniform').
            cancel_event (threading.Event | None): Optional cooperative-cancellation event polled between GPU batches
                so a cancelled request stops before the next forward pass.
            camera_only (bool): When set, run CWIP on the camera stream alone. The world encoder is skipped and the
                similarity head is not produced (``pred_similarity_indices`` is ``None``); a black world tensor of
                matching shape is synthesized purely so the summary carries a native resolution.

        Returns:
            Tuple of ``(summary, pred_similarity_indices, pred_type_indices, raw_type_logits)``.  ``raw_type_logits``
            is the patch-resolution type logits tensor (N, C, Hp, Wp) on CPU, or None.
        """
        self.seg_model.eval()

        self.logger.info("Preprocessing world/camera videos for CWIP inference (all frames)...")
        video_data = cwip_image_video.preprocess_world_camera_video_pair(
            world_video_path=("" if camera_only else world_video_path) or "",
            camera_video_path=camera_video_path,
            num_frames=num_frames,
            transform=None,
            sample_method=sample_method,
            verbose=(self.verbose == "DEBUG"),
            # Both modes sample onto the object-check timeline via ref_fps, so every frame CWIP sees has RDS object
            # geometry behind it. In camera-only the black-world path resamples the camera stream at this rate too, so
            # the timeline is identical whether or not a world video is present.
            ref_fps=OBJECT_CHECK_FPS,
            # In camera-only mode the world file is absent. Take the camera-only decode path, but do not materialize a
            # full black world tensor: it is never fed to the forward pass (world_image=None below), so allocating a
            # second copy of the decoded clip would only waste memory. The world shape downstream bookkeeping needs
            # equals the camera shape here.
            create_black_world_if_missing=camera_only,
            materialize_black_world=not camera_only,
        )

        camera_video = video_data["camera_video"]
        world_video = video_data["world_video"]
        num_frames = int(video_data["num_frames"])
        # In camera-only the black world is not materialized (world_video is None); the world shape equals the camera's.
        world_video_shape = tuple((world_video if world_video is not None else camera_video).shape)

        self.logger.info(
            "CWIP input prepared: camera=%s, world=%s, frames=%d",
            tuple(camera_video.shape),
            world_video_shape,
            num_frames,
        )

        # Get batch size from config, default to 4 to avoid OOM on large videos
        cwip_cfg = self.oc_config.get("cwip", {}) if isinstance(self.oc_config, dict) else {}
        batch_size = int(cwip_cfg.get("batch_size", 4))
        if batch_size <= 0:
            self.logger.warning("Invalid CWIP batch_size=%d; defaulting to 4", batch_size)
            batch_size = 4

        # camera_only drives the model to skip the world encoder and emit no similarity head, matching the
        # ``world_image=None`` forwards below. It is passed per forward call rather than set on the shared model config,
        # so a concurrent clip on the same replica (the service reuses one model at OBJECTS_PROCESS_CONCURRENCY_LIMIT >
        # 1) selects its own mode without contending on shared state. forward() holds its own inference_lock around the
        # resolution-dependent instance state, so no outer lock is needed here.
        with torch.no_grad():
            start = time.time()

            # Process frames in batches to avoid GPU OOM
            if num_frames <= batch_size:
                # Small enough to process in one batch. Checkpoint first so a cancelled request skips the (expensive)
                # single forward pass instead of running it on a doomed temp dir.
                check_cancelled(cancel_event)
                # world_video is None exactly in camera-only mode (the black world is not materialized), so this also
                # narrows the Optional for the type checker.
                data_batch = {
                    "camera_image": camera_video.to(self.model_device),
                    "world_image": None if world_video is None else world_video.to(self.model_device),
                }
                output_batch = self.seg_model.forward(**data_batch, camera_only_mode=camera_only)
            else:
                # Process in smaller batches and merge results
                self.logger.info("Processing %s frames in batches of %s...", num_frames, batch_size)
                all_outputs = []
                for i in range(0, num_frames, batch_size):
                    # Cooperative-cancellation checkpoint between GPU batches so a cancelled request stops before the
                    # next (expensive) forward pass instead of running to completion on a doomed temp dir.
                    check_cancelled(cancel_event)
                    end_idx = min(i + batch_size, num_frames)
                    mini_batch = {
                        "camera_image": camera_video[i:end_idx].to(self.model_device),
                        "world_image": (None if world_video is None else world_video[i:end_idx].to(self.model_device)),
                    }
                    batch_output = self.seg_model.forward(**mini_batch, camera_only_mode=camera_only)
                    # Keep only the logits consumed downstream and move them to CPU. The large encoder spatial-feature
                    # maps/lists are unused here; accumulating them across every batch is what OOMs CPU memory on long
                    # clips (see extract_used_cwip_outputs).
                    batch_output = extract_used_cwip_outputs(batch_output)
                    all_outputs.append(batch_output)

                    # Clear GPU memory after each batch
                    del mini_batch
                    clear_gpu_memory()

                # Merge all batch outputs
                output_batch = merge_batch_outputs(all_outputs)

            elapsed = time.time() - start
            self.logger.info(
                "CWIP inference completed in %.3fs (%.3f ms/frame)", elapsed, (elapsed / max(num_frames, 1)) * 1000.0
            )

        # Capture the actual frame indices that CWIP consumed, so that the caller can map per-output frame_idx back
        # to the underlying world/camera video frames. This is required for trial-mode runs (where N frames are
        # *uniformly sampled* across the entire video) and for any pair where camera_fps != world_fps (where world i
        # and camera i refer to different source frames). Without this, downstream visualization fetches the wrong
        # base camera frame and per-frame object/pose lookups index the wrong row, causing world-model overlays to
        # drift relative to the camera image.
        summary = {
            "num_frames": num_frames,
            "camera_video_shape": tuple(camera_video.shape),
            "world_video_shape": world_video_shape,
            "has_patch_logits": bool("patch_similarity_logits" in (output_batch or {})),
            "has_patch_type_logits": bool("patch_type_logits" in (output_batch or {})),
            "camera_frame_indices": list(video_data.get("camera_frame_indices", [])),
            "world_frame_indices": list(video_data.get("world_frame_indices", [])),
            "logical_frame_indices": list(video_data.get("logical_frame_indices", [])),
            "world_fps": video_data.get("world_fps"),
            "camera_fps": video_data.get("camera_fps"),
        }
        # Initialize before the try: the similarity head is absent in camera-only mode (patch_similarity_logits is
        # None), so pred_similarity_indices stays None; both are also reset to None if logit conversion raises.
        pred_similarity_indices: torch.Tensor | None = None
        pred_type_indices: torch.Tensor | None = None
        raw_type_logits: torch.Tensor | None = None
        try:
            native_size = (world_video_shape[2], world_video_shape[3])

            logits_patch = output_batch.get("patch_similarity_logits", None)
            if logits_patch is not None:
                patch_results = compute_patch_predictions_and_percentages(logits_patch, treat_wrong_type_as_match=True)
                pred_similarity_indices = resize_index_map(patch_results["pred_class_indices"], native_size).cpu()

            type_logits = output_batch.get("patch_type_logits", None)
            if type_logits is not None:
                pred_type_indices = torch.sigmoid(type_logits).argmax(dim=1)
                pred_type_indices = resize_index_map(pred_type_indices, native_size).cpu()
                raw_type_logits = type_logits.float().cpu()
        except Exception as e:  # noqa: BLE001 - CWIP logit conversion failure surface is broad; log and skip
            self.logger.warning("Exception getting logits_patch and converting to pred_class_indices: %s", e)
            pred_similarity_indices = None
            pred_type_indices = None
        return summary, pred_similarity_indices, pred_type_indices, raw_type_logits

    def _parse_presence_config(self) -> None:
        """Parse and validate the ``presence`` config block once, at construction.

        Sets ``self._presence`` (the parsed :class:`PresenceConfig`, per-class thresholds resolved to model class
        indices), ``self._type_mask_settings`` (the type-mask settings as configured, keyed by object class name, as
        recorded in the results and the debug dump), and ``self._halluc_overrides`` (the per-class tracker overrides,
        empty class entries dropped).

        A malformed block fails the job fast instead of silently taking no effect: an unknown key would be ignored; an
        unrecognized ``source`` or ``type_mask.gate`` would fall back to a default; a probability outside ``[0, 1]``
        would mark every patch absent (> 1) or present (< 0); an unknown ``type_mask.per_class_thresholds`` key, or a
        ``hallucination_overrides`` key that is not a static object class with a hallucination-detector entry, would be
        ignored. The objects-config keys ``camera_only`` and ``two_stream`` are rejected, since presence settings
        placed there would otherwise be ignored.

        Raises ``ValueError`` / ``TypeError`` (never ``RuntimeError``): processor construction wraps this call, and a
        ``RuntimeError`` there is caught and the static check skipped with only a warning, which would let the job
        succeed with no static results on a bad config.
        """
        misplaced = sorted(k for k in ("camera_only", "two_stream") if k in self.oc_config)
        if misplaced:
            raise ValueError(
                f"objects config key(s) {misplaced} are not recognized; the presence settings live under 'presence' "
                f"(source, type_mask, hallucination_overrides)"
            )
        cfg = self.oc_config.get("presence")
        if cfg is None:
            cfg = {}
        if not isinstance(cfg, dict):
            raise TypeError(f"presence must be a mapping, got {type(cfg).__name__}")
        unknown_keys = sorted(set(cfg) - {"source", "type_mask", "hallucination_overrides"})
        if unknown_keys:
            raise ValueError(
                f"presence has unknown key(s) {unknown_keys}; valid keys: source, type_mask, hallucination_overrides"
            )
        source = cfg.get("source", DEFAULT_TWO_STREAM_PRESENCE_SOURCE)
        if source not in set(PresenceSource):
            raise ValueError(f"presence.source must be one of {[s.value for s in PresenceSource]}, got {source!r}")

        type_mask = cfg.get("type_mask")
        if type_mask is None:
            type_mask = {}
        if not isinstance(type_mask, dict):
            raise TypeError(f"presence.type_mask must be a mapping, got {type(type_mask).__name__}")
        unknown_keys = sorted(set(type_mask) - {"gate", "confidence_threshold", "per_class_thresholds"})
        if unknown_keys:
            raise ValueError(
                f"presence.type_mask has unknown key(s) {unknown_keys}; "
                f"valid keys: gate, confidence_threshold, per_class_thresholds"
            )
        gate = type_mask.get("gate", TypeMaskGate.CONFIDENCE)
        if gate not in set(TypeMaskGate):
            raise ValueError(f"presence.type_mask.gate must be one of {[g.value for g in TypeMaskGate]}, got {gate!r}")

        def _probability(name: str, value: object) -> float:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"presence.type_mask.{name} must be a probability in [0, 1], got {value!r}")
            return float(value)

        threshold = _probability(
            "confidence_threshold", type_mask.get("confidence_threshold", DEFAULT_TYPE_CONFIDENCE_THRESHOLD)
        )
        static_classes = set(self.STATIC_OBJECTS)
        # per_class_thresholds gates the presence mask for any static class: keys must be static, values in [0, 1].
        per_class = type_mask.get("per_class_thresholds")
        if per_class is None:
            per_class = {}
        if not isinstance(per_class, dict):
            raise TypeError(
                f"presence.type_mask.per_class_thresholds must be a mapping, got {type(per_class).__name__}"
            )
        unknown = sorted(k for k in per_class if k not in static_classes)
        if unknown:
            raise ValueError(
                f"presence.type_mask.per_class_thresholds has unknown object class(es) {unknown}; "
                f"valid classes: {sorted(static_classes)}"
            )
        per_class_by_name = {
            name: _probability(f"per_class_thresholds[{name!r}]", value) for name, value in sorted(per_class.items())
        }
        class_thresholds: dict[int, float] = {}
        if per_class_by_name:
            mapping_cls = get_object_type_class_mapping(
                getattr(self.seg_model.config, "object_type_class_mapping", None)
            )
            for class_name, class_thr in per_class_by_name.items():
                object_type_list = COSMOS_EVALUATOR_OBJECT_TYPE_STRING_TO_ENUMS.get(class_name, [])
                for model_idx in resolve_model_class_indices(object_type_list, mapping_cls):
                    class_thresholds[model_idx] = class_thr

        self._halluc_overrides = self._parse_halluc_overrides(cfg.get("hallucination_overrides"))
        self._presence = PresenceConfig(
            source=PresenceSource(source),
            type_mask=TypeMaskConfig(
                gate=TypeMaskGate(gate), confidence_threshold=threshold, class_thresholds=class_thresholds
            ),
        )
        self._type_mask_settings: dict[str, Any] = {
            "gate": str(TypeMaskGate(gate)),
            "confidence_threshold": threshold,
            "per_class_thresholds": per_class_by_name,
        }

    def _parse_halluc_overrides(self, overrides: Any) -> dict[str, dict[str, int]]:
        """Validate ``presence.hallucination_overrides`` and return it with empty class entries dropped.

        ``None`` (the block or a class entry left empty in YAML) means no override. Overrides only take effect for
        static classes that have a hallucination-detector entry: the static detector drops every non-static class, so
        an override for a dynamic class (e.g. vehicle) or for a static class without a detector entry (e.g. crosswalk /
        wait_line) would be silently ignored. Raises ``ValueError`` / ``TypeError``, never ``RuntimeError``.
        """
        if overrides is None:
            return {}
        block = "presence.hallucination_overrides"
        if not isinstance(overrides, dict):
            raise TypeError(f"{block} must be a mapping, got {type(overrides).__name__}")
        static_classes = set(self.STATIC_OBJECTS)
        detector_cfg = self.oc_config.get("hallucination_detector")
        if detector_cfg is None:
            detector_cfg = {}
        if not isinstance(detector_cfg, dict):
            raise TypeError(
                f"hallucination_detector must be a mapping when {block} is set, got {type(detector_cfg).__name__}"
            )
        detector_classes = set(detector_cfg.get("classes") or {}) & static_classes
        unknown_ov = sorted(k for k in overrides if k not in detector_classes)
        if unknown_ov:
            raise ValueError(
                f"{block} has class(es) {unknown_ov} that are not static object classes with a hallucination-detector "
                f"entry; valid classes: {sorted(detector_classes)}"
            )
        parsed: dict[str, dict[str, int]] = {}
        for class_name, override in overrides.items():
            if override is None:
                continue
            if not isinstance(override, dict):
                raise TypeError(f"{block}[{class_name!r}] must be a mapping, got {type(override).__name__}")
            entry: dict[str, int] = {}
            area = override.get("min_cluster_area")
            if area is not None:
                if not isinstance(area, (int, float)) or isinstance(area, bool) or area < 0:
                    raise ValueError(f"{block}[{class_name!r}].min_cluster_area must be >= 0, got {area!r}")
                entry["min_cluster_area"] = int(area)
            frames = override.get("min_track_frames")
            if frames is not None:
                if not isinstance(frames, int) or isinstance(frames, bool) or frames < 1:
                    raise ValueError(
                        f"{block}[{class_name!r}].min_track_frames must be an integer >= 1, got {frames!r}"
                    )
                entry["min_track_frames"] = frames
            if entry:
                parsed[class_name] = entry
        return parsed

    def _halluc_overrides_for_run(self, camera_only: bool) -> dict[str, dict[str, int]]:
        """Return the per-class hallucination-tracker overrides that apply to this run.

        The type-head presence mask over-produces small, short-lived clusters relative to the similarity mask, so a run
        whose hallucination layer can contain type-head-only patches (:attr:`PresenceSource.includes_type_only_patches`:
        camera-only, and two-stream ``type`` / ``union``) needs a tighter tracker than the shared
        ``hallucination_detector.classes`` config tuned on the similarity mask. Other sources take none.
        """
        if not self._presence.source_for_run(camera_only).includes_type_only_patches:
            return {}
        return self._halluc_overrides

    def _halluc_gates_for_run(self, camera_only: bool) -> HallucinationGates:
        """The per-class tracker gates of one run: the detector's configured gates with this run's overrides on top.

        Computed as a value per run and passed to the detector and the persistence gate, so no configured state is
        modified and a later run on the same instance starts from the configured gates.
        """
        if self.hallucination_detector is None:
            return HallucinationGates(min_cluster_area={}, min_track_frames={})
        overrides = self._halluc_overrides_for_run(camera_only)
        min_cluster_area = {
            name: ov["min_cluster_area"]
            for name, ov in overrides.items()
            if "min_cluster_area" in ov and name in self._hallucination_classes
        }
        min_track_frames = dict(self._hallucination_min_frames_by_class)
        for name, ov in overrides.items():
            if "min_track_frames" in ov:
                min_track_frames[class_name_to_rds_label(name)] = ov["min_track_frames"]
        return HallucinationGates(min_cluster_area=min_cluster_area, min_track_frames=min_track_frames)

    def _presence_record(self, camera_only: bool) -> dict[str, Any]:
        """The effective presence settings of a run, as recorded on the static results.

        ``type_mask`` is ``None`` when the source does not use the type-head mask (``similarity``), and
        ``hallucination_overrides`` lists the overrides actually applied: empty when the source takes none, or when
        hallucination detection is disabled (or failed to initialize), since no tracker gate then runs.
        """
        source = self._presence.source_for_run(camera_only)
        applied = self._halluc_overrides_for_run(camera_only) if self.hallucination_detector is not None else {}
        return {
            "source": str(source),
            "type_mask": dict(self._type_mask_settings) if source.uses_type_mask else None,
            "hallucination_overrides": {k: dict(v) for k, v in applied.items()},
        }

    def _type_mask_settings_json(self) -> str:
        """The configured type-mask settings as a canonical JSON string, stored with dumped type-head presence masks."""
        return json.dumps(self._type_mask_settings, sort_keys=True)

    def _check_replay_supports_presence(self, camera_only: bool) -> None:
        """Reject, before any data is loaded, a debug-dump replay that cannot reproduce this run's presence layers.

        A compact dump does not store the type logits the ``confidence`` gate needs. It is accompanied by the
        type-head presence masks the dumped run computed instead (:data:`TYPE_PRESENCE_DUMP`, with the type-mask
        settings that produced them), so a replay that uses the type-head mask under the ``confidence`` gate needs
        those masks, computed under the same settings.
        """
        if self._cwip_debug_dump_dir is None:
            return
        source = self._presence.source_for_run(camera_only)
        if not (source.uses_type_mask and self._presence.type_mask.needs_logits):
            return
        dump_path = Path(self._cwip_debug_dump_dir) / TYPE_PRESENCE_DUMP
        dumped_settings = None
        if dump_path.is_file():
            with np.load(dump_path) as dumped:
                dumped_settings = str(dumped["type_presence_settings"])
        if dumped_settings is None:
            raise ValueError(
                f"debug dump {dump_path} has no type-head presence masks, which presence source {source.value!r} "
                f"with the confidence gate needs; replay with presence.type_mask.gate=argmax (camera-only) or "
                f"presence.source=similarity (two-stream), or re-run inference"
            )
        if dumped_settings != self._type_mask_settings_json():
            raise ValueError(
                f"debug dump {dump_path} holds type-head presence masks computed with type_mask settings "
                f"{dumped_settings}, but this run is configured with {self._type_mask_settings_json()}; replay with "
                f"the dumped settings or re-run inference"
            )

    @staticmethod
    def _load_type_presence_from_dump(dump_dir: str) -> torch.Tensor | None:
        """Load the dumped ``(frames, H, W)`` long type-head presence masks, or ``None`` if the dump has none."""
        path = Path(dump_dir) / TYPE_PRESENCE_DUMP
        if not path.is_file():
            return None
        with np.load(path) as dumped:
            shape = tuple(int(x) for x in dumped["type_presence_shape"])
            bits = np.unpackbits(dumped["type_presence_packed"], axis=-1, count=shape[-1])
        return torch.from_numpy(bits.reshape(shape).astype(np.int64))

    def _type_presence_mask(
        self,
        frame_idx: int,
        pred_type_indices: torch.Tensor,
        raw_type_logits: torch.Tensor | None,
        replay_type_presence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the type-head presence mask (H, W long 0/1) for one frame.

        The presence layer of a camera-only run, which has no similarity head, and an alternative or complement to the
        similarity verdict in a two-stream run (``presence.source``). The result feeds layer 0 of the (2, H, W)
        class-mask tensor; the downstream class filter ANDs it with (type == queried class), so background patches are
        excluded regardless of the gate. ``replay_type_presence`` holds the masks a dumped run computed, used in place
        of the logits when replaying a compact debug dump (see :meth:`_check_replay_supports_presence`).
        """
        if replay_type_presence is not None and self._presence.type_mask.needs_logits and raw_type_logits is None:
            return replay_type_presence[frame_idx]
        return type_presence_mask(
            pred_type_indices[frame_idx],
            raw_type_logits[frame_idx] if raw_type_logits is not None else None,
            self._presence.type_mask,
        )

    def _presence_layers(
        self,
        frame_idx: int,
        pred_similarity_indices: torch.Tensor | None,
        pred_type_indices: torch.Tensor,
        raw_type_logits: torch.Tensor | None,
        replay_type_presence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Layer 0 (H, W long 0/1) of the correspondence and hallucination class-mask tensors for one frame.

        Camera-only (no similarity head): both checks share the type-head presence mask. Two-stream: ``presence.source``
        selects ``similarity`` (MATCH for correspondence, EXTRA for hallucination), ``type`` (the type-head mask for
        both, as in camera-only), or the ``union`` / ``intersection`` of the similarity verdict with the type-head mask,
        applied to each check's similarity label.
        """
        return presence_layers(
            pred_similarity_indices[frame_idx] if pred_similarity_indices is not None else None,
            lambda: self._type_presence_mask(frame_idx, pred_type_indices, raw_type_logits, replay_type_presence),
            self._presence.source,
        )

    def _dump_type_presence(
        self, debug_dir: Path, pred_type_indices: torch.Tensor | None, raw_type_logits: torch.Tensor | None
    ) -> None:
        """Write the run's type-head presence masks, bit-packed, to ``<debug_dir>/type_presence.npz``.

        Written only under the ``confidence`` gate (``argmax`` needs no logits to recompute) and when the logits are
        available. One bit per pixel of the grid the frame loop scores, with the type-mask settings that produced them,
        so a replay of the compact dump reproduces the run's presence layers exactly without the omitted logits.
        Best-effort, like the rest of the debug dump: a failure is logged, never fatal.
        """
        if pred_type_indices is None or raw_type_logits is None or not self._presence.type_mask.needs_logits:
            return
        try:
            np.savez_compressed(
                debug_dir / TYPE_PRESENCE_DUMP,
                allow_pickle=False,
                **self._type_presence_arrays(pred_type_indices, raw_type_logits),
            )
        except Exception as exc:  # noqa: BLE001 - debug dump is best-effort; never abort processing
            self.logger.warning("Failed to dump type-head presence masks: %s", exc)

    def _type_presence_arrays(
        self, pred_type_indices: torch.Tensor, raw_type_logits: torch.Tensor
    ) -> dict[str, np.ndarray]:
        """The arrays of :data:`TYPE_PRESENCE_DUMP`: packed masks, their shape, and the settings JSON."""
        masks = np.stack(
            [
                self._type_presence_mask(i, pred_type_indices, raw_type_logits).to(torch.uint8).cpu().numpy()
                for i in range(int(pred_type_indices.shape[0]))
            ]
        )
        return {
            "type_presence_packed": np.packbits(masks, axis=-1),
            "type_presence_shape": np.asarray(masks.shape, dtype=np.int64),
            "type_presence_settings": np.asarray(self._type_mask_settings_json()),
        }

    def _process_frames_static(
        self,
        camera_poses: dict[int, np.ndarray],
        camera_model: CameraBase,
        data_loader: RdsDataLoader,
        timeline: ObjectCheckTimeline,
        camera_video_path: str,
        world_video_path: str | None,
        trial_frames: int | None,
        results: dict[str, Any],
        track_output_agg: dict[int, dict[str, Any]],
        cwip_cache: CwipInferenceCache | None = None,
        ego_poses: dict[int, Any] | None = None,
        cancel_event: threading.Event | None = None,
        camera_only: bool = False,
        halluc_gates: HallucinationGates | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Process frames using CWIP outputs to compute overlap scores and visualization.

        Args:
            timeline: The object-check timeline. Supplies the step count and every step-to-source index mapping, so
                neither the video rates nor the RDS pose rate are consulted here.
            trial_frames: When set, only selects contiguous-from-start sampling; the step count itself already
                reflects the trial cap via ``timeline.num_steps``.
            halluc_gates: The run's per-class hallucination-tracker gates; ``None`` uses the detector's configured
                cluster areas.

        Returns:
            Tuple[np.ndarray, np.ndarray, int]: ``(score_matrix, projection_area_matrix, skipped_frames)``.
            ``projection_area_matrix`` is a (frames, tracks) float matrix of per-frame visible projection area in
            pixels², NaN where the track was not scored.
        """
        cwip_cfg = self.oc_config.get("cwip", {}) if isinstance(self.oc_config, dict) else {}
        sample_method = cwip_cfg.get("sample_method", "uniform")

        num_frames = timeline.num_steps
        if trial_frames is not None:
            # Anchor the shortened window at the start of the clip so the evaluated steps are exactly the first N,
            # matching what the dynamic check iterates. ``--trial N`` counts evaluation steps, not source frames, so
            # no rate conversion is needed: both checks cover the same ``N / OBJECT_CHECK_FPS`` seconds.
            sample_method = "first"

        # CWIP inference (replay from a debug dump when set, else load from cache, else run inference)
        self.profiler.start("segmentation")
        cached_result = cwip_cache.load() if cwip_cache is not None else None
        replay_type_presence: torch.Tensor | None = None
        if self._cwip_debug_dump_dir is not None:
            self.logger.info("CWIP inference: REPLAY from dump %s, skipping model inference", self._cwip_debug_dump_dir)
            summary, pred_similarity_indices, pred_type_indices, raw_type_logits = self._load_cwip_from_dump(
                self._cwip_debug_dump_dir
            )
            replay_type_presence = self._load_type_presence_from_dump(self._cwip_debug_dump_dir)
        elif cached_result is not None:
            self.logger.info("CWIP inference: loaded from cache, skipping model inference")
            summary, pred_similarity_indices, pred_type_indices, raw_type_logits = cached_result
        else:
            summary, pred_similarity_indices, pred_type_indices, raw_type_logits = self._run_cwip_inference_on_videos(
                camera_video_path=camera_video_path,
                world_video_path=world_video_path,
                num_frames=num_frames,
                sample_method=sample_method,
                cancel_event=cancel_event,
                camera_only=camera_only,
            )
            if cwip_cache is not None:
                cwip_cache.save(summary, pred_similarity_indices, pred_type_indices, raw_type_logits)
        self.profiler.end()

        # Raw CWIP output is the shared upstream for every static check, so dumping it once enables offline
        # debugging of all of them (reconstruct any class mask, re-derive correspondence, inspect TL-state logits).
        dump_full_logits = bool(self.oc_config.get("debug_dump_full_logits", False))
        if self._debug_dir is not None:
            dump_cwip_raw(
                self._debug_dir,
                dump_full_logits=dump_full_logits,
                pred_similarity_indices=pred_similarity_indices,
                pred_type_indices=pred_type_indices,
                raw_type_logits=raw_type_logits,
                summary=summary,
                logger=self.logger,
            )

        # Guard against a mismatched world-model (control) video, on both the cache-hit and fresh-inference paths (which
        # converge here on ``summary``). CWIP is resolution-flexible:  on a shape mismatch it silently resizes the
        # camera image to the world image's shape and adapts its positional embeddings
        # (see checks/utils/models/cwip/cwip.py). Feeding it a top-down BEV world video for a forward-facing camera
        # therefore produces no MATCH and collapses every static score to ~0 with no error. The world (control) video
        # must be the same *perspective* view as the camera, so require matching aspect ratios; this rejects the BEV
        # (square 1:1) while tolerating the benign control-vs-camera scaling (1280x704 vs 1280x720).
        camera_shape = summary.get("camera_video_shape")
        world_shape = summary.get("world_video_shape")
        if camera_shape and world_shape:
            self._validate_world_camera_aspect_ratio(
                camera_hw=(int(camera_shape[2]), int(camera_shape[3])),
                world_hw=(int(world_shape[2]), int(world_shape[3])),
            )

        num_frames = int(summary.get("num_frames", 0))
        # The similarity head is absent in camera-only mode; the type head then serves as the per-frame index map and
        # the source of shape/resolution for the loop below.
        shape_ref = pred_similarity_indices if pred_similarity_indices is not None else pred_type_indices
        if num_frames <= 0 or shape_ref is None:
            self.logger.warning("CWIP returned zero frames; producing empty score matrix")
            # Match the 3-tuple shape the caller unpacks (score_matrix, projection_area_matrix, skipped_frames). Both
            # matrices mirror the normal-path dtype/ndim (float64 2-D) so downstream code can index them safely;
            # ``skipped_frames`` is 0 because we never entered the per-frame loop.
            empty_matrix = np.full((0, 0), np.nan)
            return empty_matrix, empty_matrix, 0

        # Per-output frame -> source video frame index. ``preprocess_world_camera_video_pair`` may sample frames
        # non-contiguously (uniformly spread across the full video in trial mode, or at different cadences for camera vs
        # world when the FPS differ). Without these explicit lookups, downstream calls would assume CWIP output
        # index == video frame index and silently desynchronize the world-model overlay from the camera frame. Cached
        # results from older runs (and some test fixtures) may not carry these lists; in that case fall back to the
        # legacy contiguous assumption where output index i corresponds to camera frame
        # ``i * camera_fps / world_fps`` and world frame ``i``.
        world_frame_indices = list(summary.get("world_frame_indices") or [])
        camera_frame_indices = list(summary.get("camera_frame_indices") or [])
        logical_frame_indices = list(summary.get("logical_frame_indices") or [])
        if len(logical_frame_indices) < num_frames:
            logical_frame_indices = list(range(num_frames))
        if len(world_frame_indices) < num_frames or len(camera_frame_indices) < num_frames:
            world_frame_indices = [timeline.world_frame(step) for step in logical_frame_indices]
            camera_frame_indices = [timeline.camera_frame(step) for step in logical_frame_indices]

        # The camera model is calibrated for CAMERA_RESCALED_RESOLUTION (typically 1280x720), but the world model
        # control video is sometimes at a different, scaled resolution (e.g. 1280x704). The world model control video
        # resolution determines the CWIP mask output resolution, so we need to resize the CWIP masks to the camera model
        # resolution so projected masks align correctly.
        cam_h = data_loader.CAMERA_RESCALED_RESOLUTION_HEIGHT
        cam_w = data_loader.CAMERA_RESCALED_RESOLUTION_WIDTH
        cam_size = (cam_h, cam_w)
        cwip_h, cwip_w = shape_ref.shape[1], shape_ref.shape[2]
        if cwip_h != cam_h or cwip_w != cam_w:
            self.logger.info(
                "Resizing CWIP output from %dx%d to camera resolution %dx%d",
                cwip_w,
                cwip_h,
                cam_w,
                cam_h,
            )
            if pred_similarity_indices is not None:
                pred_similarity_indices = resize_index_map(pred_similarity_indices, cam_size)
            if pred_type_indices is not None:
                pred_type_indices = resize_index_map(pred_type_indices, cam_size)
            shape_ref = pred_similarity_indices if pred_similarity_indices is not None else pred_type_indices

        # The compact debug dump omits the type logits, so it is accompanied by the type-head presence masks a replay
        # uses in their place. They are computed here, on the grid the frame loop scores (after the resize above).
        if self._debug_dir is not None and not dump_full_logits:
            self._dump_type_presence(self._debug_dir, pred_type_indices, raw_type_logits)

        # The resize branch reassigns shape_ref from Optional tensors; at least one head is always present past the
        # early return above, so pin it non-None for the per-frame loop below.
        assert shape_ref is not None

        max_track_id = max(data_loader.object_count, 1)
        score_matrix = np.full((num_frames, max_track_id + 1), np.nan)
        score_matrix[:, 0] = 0.0
        # Per-frame visible projection area (pixels²) for each scored (frame, track). Parallel to score_matrix;
        # NaN where the track was not scored that frame. Used downstream by the cross-view aggregator for
        # evidence-strength weighting.
        projection_area_matrix = np.full((num_frames, max_track_id + 1), np.nan)

        num_vis_frames = min(num_frames, shape_ref.shape[0])
        for frame_idx in range(num_vis_frames):
            # Cooperative-cancellation checkpoint: stop the per-frame scoring/visualization loop promptly if the
            # owning request was cancelled.
            check_cancelled(cancel_event)
            # Translate this CWIP output index back to the actual source-video frame indices (see derivation of
            # *_frame_indices above) before mapping to the object/pose timeline or fetching the camera image.
            camera_frame_idx = camera_frame_indices[frame_idx]
            # Exact by construction: the timeline only ever addresses RDS indices that carry object annotations, so
            # this can no longer land on an unannotated frame the way the old rate-ratio rounding could.
            mapped_idx = timeline.rds_index(logical_frame_indices[frame_idx])

            mask_frame = np.zeros((shape_ref.shape[1], shape_ref.shape[2], 3), dtype=np.uint8)
            # The similarity-class overlay only exists in two-stream mode; camera-only leaves it black.
            if pred_similarity_indices is not None:
                for cls_id, _ in enumerate(PatchSimilarityType):
                    cls_mask = pred_similarity_indices[frame_idx] == cls_id
                    mask_frame[cls_mask] = np.array(self.class_colors[cls_id], dtype=np.uint8)

            # Per-frame fetch: static objects' geometry/attributes are now selected for ``mapped_idx`` rather than
            # coming from a clip-level snapshot. Reuse the same dict for the SceneRasterizer and the downstream
            # class-by-class scoring loop so they see a consistent view of this frame.
            frame_objects = data_loader.get_object_data_for_frame(mapped_idx, include_static=True)

            # Create SceneRasterizer for this frame - computes visibility masks in constructor
            self.profiler.start("scene_rasterization")
            scene_rasterizer = SceneRasterizer(
                frame_objects,
                camera_poses[mapped_idx],
                camera_model,
                shape_ref.shape[2],
                shape_ref.shape[1],
                logger=self.logger,
            )
            self.profiler.end()

            frame_scores = {}

            if pred_type_indices is None:
                raise RuntimeError(f"pred_type_indices is None while scoring static frame_idx={frame_idx}")
            frame_type_pred = pred_type_indices[frame_idx]
            match_class_indices = torch.zeros((2, frame_type_pred.shape[0], frame_type_pred.shape[1]), dtype=torch.long)
            hallucination_class_indices = match_class_indices.clone()
            # get_class_mask ANDs layer 0 with (type == queried class); correspondence scores "class present inside the
            # projected GT" and hallucination "class present outside it".
            frame_similarity_pred = pred_similarity_indices[frame_idx] if pred_similarity_indices is not None else None
            match_layer0, halluc_layer0 = self._presence_layers(
                frame_idx, pred_similarity_indices, pred_type_indices, raw_type_logits, replay_type_presence
            )
            match_class_indices[0, :, :] = match_layer0
            match_class_indices[1, :, :] = frame_type_pred
            hallucination_class_indices[0, :, :] = halluc_layer0
            hallucination_class_indices[1, :, :] = frame_type_pred

            self.profiler.start("processing")
            for class_name in self.oc_config.get("overlap_check", {}):
                if class_name not in self.STATIC_OBJECTS:
                    continue
                ego_pose = ego_poses[mapped_idx] if ego_poses and mapped_idx < len(ego_poses) else None
                cls_scores, _cls_filtered, _debug_info = self.overlap_detector.process_class(
                    class_name,
                    match_class_indices,
                    frame_objects,
                    camera_poses[mapped_idx],
                    camera_model,
                    match_class_indices.shape[2],
                    match_class_indices.shape[1],
                    track_output_agg,
                    mapped_idx,
                    scene_rasterizer=scene_rasterizer,
                    ego_pose=ego_pose,
                    is_static=True,
                )
                frame_scores.update(cls_scores)
            self.profiler.end()

            # Traffic light state check (requires per-class type probabilities)
            tl_results = {}
            if raw_type_logits is not None:
                self.profiler.start("traffic_light_state")
                try:
                    if pred_type_indices is None:
                        raise RuntimeError(
                            f"pred_type_indices is None during traffic-light check for frame_idx={frame_idx}"
                        )
                    native_h, native_w = pred_type_indices.shape[1], pred_type_indices.shape[2]
                    frame_logits = raw_type_logits[frame_idx : frame_idx + 1]
                    frame_logits_native = torch.nn.functional.interpolate(
                        frame_logits, size=(native_h, native_w), mode="bilinear", align_corners=False
                    )
                    frame_type_probs = torch.sigmoid(frame_logits_native).squeeze(0)
                    tl_checker = TrafficLightStateChecker(
                        self.seg_model,
                        scene_rasterizer,
                        logger=self.logger,
                        min_in_frame_fraction=self._tl_state_min_in_frame_fraction,
                    )
                    tl_results = tl_checker.check_frame(
                        frame_type_probs,
                        frame_objects,
                        mapped_idx,
                        frame_similarity=frame_similarity_pred,
                    )
                    for track_id_str, tl_result in tl_results.items():
                        tl_agg = results.setdefault("traffic_light_state", {})
                        tl_track = tl_agg.setdefault(track_id_str, [])
                        tl_track.append(
                            {
                                "frame_idx": frame_idx,
                                "traffic_light_state_score": tl_result.traffic_light_state_score,
                                "traffic_light_multi_color_score": tl_result.traffic_light_multi_color_score,
                                "traffic_light_state_confidence": tl_result.confidence,
                                "expected_state": tl_result.expected_state,
                                "traffic_light_state_probs": tl_result.state_probs,
                            }
                        )
                except Exception as e:  # noqa: BLE001 - per-frame traffic-light check must not abort the run
                    self.logger.warning("Traffic light state check failed for frame %d: %s", frame_idx, e)
                self.profiler.end()

            # Hallucination detection
            halluc_det = None
            if self.hallucination_detector is not None:
                self.profiler.start("hallucination_detection")
                try:
                    halluc_det = self.hallucination_detector.detect(
                        frame_idx=frame_idx,
                        resized_masks=hallucination_class_indices,
                        frame_objects=frame_objects,
                        camera_pose=camera_poses[mapped_idx],
                        camera_model=camera_model,
                        image_width=hallucination_class_indices.shape[2],
                        image_height=hallucination_class_indices.shape[1],
                        min_cluster_area=halluc_gates.min_cluster_area if halluc_gates is not None else None,
                    )
                    if halluc_det:
                        start_indices = {}
                        for cls_name, recs in halluc_det.items():
                            bucket = results.get("hallucination_detections", {}).get(cls_name, None)
                            if bucket is not None and isinstance(recs, list):
                                start_indices[cls_name] = len(bucket)
                                bucket.extend(recs)
                        if any(len(v or []) > 0 for v in halluc_det.values()):
                            try:
                                track_hallucinations(
                                    results=results,
                                    start_indices=start_indices,
                                    road_mask=None,
                                    proximity_by_class=self._hallucination_proximity_by_class,
                                )
                            except Exception as _te:  # noqa: BLE001 - per-frame tracking must not abort the run
                                self.logger.warning("Hallucination tracking failed for frame %s: %s", frame_idx, _te)
                        counts = {k: len(v or []) for k, v in (halluc_det or {}).items()}
                        self.logger.info("Frame %s: hallucinations %s", frame_idx, counts)
                except Exception as _e:  # noqa: BLE001 - per-frame detection must not abort the run
                    self.logger.warning("Hallucination detection failed for frame %s: %s", frame_idx, _e)
                finally:
                    self.profiler.end()

            # Mirror the dynamic processor's filter: only count this frame as "processed" - and only render its
            # visualization - when at least one static-object track actually scored here. Without this gate,
            # ``--trial N`` produces a visualization that includes every CWIP output frame regardless of whether any
            # world-model geometry projected into it, which makes the static viz video much longer than the dynamic
            # one and not aligned with processed_frame_ids.
            # EXCEPTION: a frame that has a hallucination detection but no GT score is still rendered (so a
            # pure-hallucination clip — a hallucinated TL has no GT object to score — still yields an overlay video);
            # it is NOT added to processed_frame_ids/score_matrix so metrics semantics are unchanged.
            frame_has_halluc = isinstance(halluc_det, dict) and any(len(v or []) > 0 for v in halluc_det.values())
            if not frame_scores and not frame_has_halluc:
                continue

            for track_id, score in frame_scores.items():
                track_id_int = int(track_id)
                results["track_ids"].add(track_id_int)
                score_matrix[frame_idx, track_id_int] = score
                if not np.isnan(score) and scene_rasterizer.has_object(str(track_id_int)):
                    projection_area_matrix[frame_idx, track_id_int] = float(
                        scene_rasterizer.get_projection_metadata(str(track_id_int)).get("visible_area_px", 0)
                    )
            if frame_scores:
                results["processed_frame_ids"].append(frame_idx)

            if self.vis_helper:
                self.profiler.start("visualization")
                # Filter the type view by the presence layers the checks scored this frame, so the rendered mask follows
                # the configured presence source (camera-only: the type-head mask), not the raw similarity labels.
                type_mask_frame_filtered = (
                    self._build_type_mask_frame(pred_type_indices, frame_idx, match_layer0 | halluc_layer0)
                    if pred_type_indices is not None
                    else None
                )
                type_mask_frame_full = (
                    self._build_type_mask_frame(pred_type_indices, frame_idx)
                    if pred_type_indices is not None and self.debug_enabled
                    else None
                )
                if self.debug_enabled:
                    self.vis_helper.write_mask_frame(mask_frame, frame_idx, self.naming_suffix, mask_name="similarity")
                    if type_mask_frame_filtered is not None:
                        self.vis_helper.write_mask_frame(
                            type_mask_frame_filtered, frame_idx, self.naming_suffix, mask_name="type_filtered"
                        )
                    if type_mask_frame_full is not None:
                        self.vis_helper.write_mask_frame(
                            type_mask_frame_full, frame_idx, self.naming_suffix, mask_name="type"
                        )
                if self.verbose:
                    vis_mask = type_mask_frame_filtered if type_mask_frame_filtered is not None else mask_frame
                    self._create_frame_visualization(
                        camera_frame_idx,
                        vis_mask,
                        frame_scores,
                        camera_poses[mapped_idx],
                        camera_model,
                        frame_objects,
                        {},
                        halluc_det,
                        scene_rasterizer=scene_rasterizer,
                        traffic_light_state_results=tl_results or None,
                    )
                self.profiler.end()

        skipped_frames = 0
        return score_matrix, projection_area_matrix, skipped_frames

    def process_clip(
        self,
        input_data: str,
        clip_id: str,
        camera_name: str,
        camera_video_path: str,
        world_video_path: str | None,
        output_dir: str | None = None,
        trial_frames: int | None = None,
        mask_cache_dir: str | None = None,
        cancel_event: threading.Event | None = None,
        camera_only: bool = False,
    ) -> dict[str, Any]:
        """Run the static (CWIP-based) objects pipeline for a clip.

        Args:
            input_data (str): Path to the dataset root or run directory.
            clip_id (str): Clip identifier.
            camera_name (str): Camera stream name for intrinsics.
            camera_video_path (str): Path to the camera video.
            world_video_path (str): Path to the world video.
            output_dir (str | None): Optional output directory for artifacts.
            trial_frames (int | None): If provided, evaluate at most this many object-check steps, i.e. the
                first ``trial_frames / OBJECT_CHECK_FPS`` seconds of the clip.
            mask_cache_dir (str | None): Directory for caching CWIP inference outputs.
                When provided, the full CWIP inference result (similarity indices, type indices, and raw type logits)
                is written on the first run and reloaded on subsequent runs with the same camera/world video pair,
                skipping model inference entirely.
            cancel_event (threading.Event | None): Optional cooperative-cancellation event polled between CWIP batches
                and per frame; when set, processing raises ``ProcessingCancelled``. ``None`` disables cancellation.

        Returns:
            Dict[str, Any]: Aggregated results including score matrix and statistics.
        """
        self.profiler.start("total_processing")

        if not camera_only and world_video_path is None:
            raise ValueError("world_video_path is required for two-stream static processing (or set camera_only)")

        self._check_replay_supports_presence(camera_only)
        halluc_gates = self._halluc_gates_for_run(camera_only)

        data_loader = self._create_data_loader(input_data, clip_id)
        camera_poses, camera_model = data_loader.load_camera_data(camera_name)
        try:
            ego_poses = data_loader.load_ego_poses()
        except Exception:
            self.logger.warning(
                "Could not load ego poses from %s - importance filter will use camera pose",
                RdsDataLoader.EGO_REFERENCE_CAMERA,
                exc_info=True,
            )
            ego_poses = None

        # Cooperative-cancellation checkpoint after data loading. The prologue below (FPS probes, visualization init)
        # is expensive enough that a request cancelled during the RDS load would otherwise run for a long time before
        # reaching the first per-frame checkpoint.
        check_cancelled(cancel_event)

        camera_video_fps = get_video_fps(camera_video_path)
        # In camera-only mode there is no world video; the timeline's "world" axis is driven by the camera stream.
        # world_video_path is guaranteed non-None here for two-stream by the precondition at the top of this method.
        video_fps = camera_video_fps if camera_only else get_video_fps(world_video_path or "")
        if video_fps is None:
            source_name = "camera" if camera_only else "world"
            source_path = camera_video_path if camera_only else world_video_path
            raise RuntimeError(f"Failed to get video FPS from {source_name} video path: {source_path}")
        if not camera_only and camera_video_fps is not None and camera_video_fps != video_fps:
            self.logger.info(
                "Camera video FPS (%.1f) differs from world video FPS (%.1f); "
                "visualization will sample every %d-th camera frame",
                camera_video_fps,
                video_fps,
                round(camera_video_fps / video_fps),
            )

        self._maybe_init_visualization(camera_video_path, data_loader)

        # One timeline for all four inputs. The annotation axis and the pose stream bound it alongside the two
        # videos, so the loop below cannot address a frame that lacks poses or object geometry.
        annotation_axis = data_loader.get_canonical_static_frame_axis()
        # An undeterminable frame count must not reach the timeline: it would bound the step count at zero and drop
        # the clip silently, having already passed the FPS probes above. Fail the same way a missing FPS does.
        camera_frame_count = get_video_frame_count(camera_video_path)
        world_frame_count = camera_frame_count if camera_only else get_video_frame_count(world_video_path or "")
        if camera_frame_count is None or world_frame_count is None:
            raise RuntimeError(
                f"Failed to get frame count from camera video path: {camera_video_path} ({camera_frame_count}) "
                f"or world video path: {world_video_path} ({world_frame_count})"
            )
        timeline = ObjectCheckTimeline.build(
            camera_video_fps=camera_video_fps if camera_video_fps and camera_video_fps > 0 else video_fps,
            camera_frame_count=camera_frame_count,
            world_video_fps=video_fps,
            world_frame_count=world_frame_count,
            pose_frame_count=(max(camera_poses) + 1) if camera_poses else 0,
            annotated_frame_count=len(annotation_axis),
            trial_steps=trial_frames,
            log=self.logger,
        )
        validate_annotation_axis(annotation_axis, timeline.num_steps, self.logger)

        # Fold the frame-sampling params into the cache key: the cached CWIP inference only covers the sampled
        # frames, so a partial ``--trial N`` run must never be reused by a full run (or vice versa). ``sample_method``
        # is included because a different sampling strategy selects different frames, and the evaluation rate because
        # a cache written before the timeline was pinned holds frames sampled at the video's own rate.
        cwip_sample_method = (self.oc_config.get("cwip", {}) or {}).get("sample_method", "uniform")
        cwip_cache = (
            CwipInferenceCache(
                mask_cache_dir,
                camera_video_path,
                world_video_path or "",
                extra_key=(
                    # sampling=v2: camera-only now resamples the camera stream at OBJECT_CHECK_FPS (ref_fps) like the
                    # two-stream path. Bumping the token misses any pre-fix camera-only cache entries, which for a
                    # non-OBJECT_CHECK_FPS camera were sampled without that resampling.
                    f"trial={trial_frames}|sample={cwip_sample_method}|fps={OBJECT_CHECK_FPS}"
                    f"|camera_only={int(camera_only)}|sampling=v2"
                ),
            )
            if mask_cache_dir is not None
            else None
        )

        # Prepare results and run static processing
        results = self._init_results_container(0)
        # Record the run mode so downstream readers can distinguish a camera-only static result (presence from the
        # type head, no world reference) from a two-stream one — they are otherwise indistinguishable now that an
        # absent world video selects camera-only automatically.
        results["camera_only"] = camera_only
        # How the presence layer was built (``type`` for camera-only, which has no similarity head), so comparability
        # checks read one field in both modes, plus every presence setting the scores depend on.
        results["presence_source"] = str(self._presence.source_for_run(camera_only))
        results["presence"] = self._presence_record(camera_only)
        results["total_evaluation_steps"] = timeline.num_steps
        track_output_agg: dict[int, dict[str, Any]] = {}
        try:
            # Inside the try, not before it: visualization is already initialized by this point, so an abort here has
            # ffmpeg children to release and must reach the handler below.
            check_cancelled(cancel_event)
            score_matrix, projection_area_matrix, skipped_frames = self._process_frames_static(
                camera_poses=camera_poses,
                camera_model=camera_model,
                data_loader=data_loader,
                camera_video_path=camera_video_path,
                world_video_path=world_video_path,
                timeline=timeline,
                trial_frames=trial_frames,
                results=results,
                track_output_agg=track_output_agg,
                cwip_cache=cwip_cache,
                ego_poses=ego_poses,
                cancel_event=cancel_event,
                camera_only=camera_only,
                halluc_gates=halluc_gates,
            )
        except BaseException:
            # On ProcessingCancelled (or any other abort) the happy-path release inside _finalize_and_summarize never
            # runs, leaving the visualization helper's ffmpeg children alive on a temp dir that is about to be deleted.
            self._release_visualization()
            raise

        # Regroup the ground-plane classes in world space, replacing the per-frame bbox tracks the frame loop
        # built for them. Batch by necessity: a connected component is only settled once every frame has
        # contributed to the shared bird's-eye map.
        if self.hallucination_detector is not None and self._ground_tracked_classes:
            written = regroup_ground_tracks(
                results,
                self._ground_tracked_classes,
                resolution_m=self._ground_tracker_resolution_m,
                thickness_px=self._ground_tracker_thickness_px,
            )
            self.logger.info(
                "Ground-plane regrouping wrote %d track(s) for %s", written, ", ".join(self._ground_tracked_classes)
            )

        # Confirm hallucination tracks by temporal persistence before summarizing: drop tracks that don't span the
        # per-class ``min_track_frames`` frames (no-op unless a class is configured with a value >= 2). Ground
        # tracked classes count distinct frames rather than a consecutive run; see confirm_hallucination_tracks.
        if self.hallucination_detector is not None:
            removed = confirm_hallucination_tracks(
                results, dict(halluc_gates.min_track_frames), total_frame_classes=self._ground_tracked_classes
            )
            if removed:
                self.logger.info("Hallucination tracks dropped by persistence gate: %d", removed)

        # Finalization (no data_loader context required for summary)
        total_frames = int(score_matrix.shape[0]) if isinstance(score_matrix, np.ndarray) else 0
        self._finalize_and_summarize(
            results,
            score_matrix,
            track_output_agg=track_output_agg,
            skipped_frames=skipped_frames,
            total_video_frames=total_frames,
            projection_area_matrix=projection_area_matrix,
        )
        # Which existence cutoff a lane line is judged by depends on whether it is dashed, which only the RDS-HQ
        # attributes record; carry it on the track so a reader of the result does not need the RDS-HQ data.
        annotate_lane_styles(results.get("tracks") or [], data_loader.all_static_objects)

        self.profiler.end()
        self.profiler.print_summary(self.logger)
        self.logger.info("Static processing complete. Frames: %s", results["processed_frames"])
        return results


# End of module
