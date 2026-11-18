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
Unified entry point for object processing.

This module provides a single `run()` function that executes both dynamic (SegFormer-based) and static (CWIP-based)
object processors sequentially, combining their results.
"""

import logging
import threading
from typing import Any

from checks.objects.single_view.object_processor_base import ObjectProcessorBase
from checks.objects.single_view.object_processor_dynamic import ObjectProcessorDynamic
from checks.objects.single_view.object_processor_static import ObjectProcessorStatic
from checks.utils.config_manager import resolve_check_config
from checks.utils.onnx import clear_gpu_memory

logger = logging.getLogger(__name__)


class ObjectCheckRunner:
    """Runs the dynamic and static object processors in sequence, managing GPU memory between them.

    Processor construction goes through :meth:`create_dynamic_processor` and :meth:`create_static_processor` so
    subclasses can substitute specialized processors without reimplementing the run sequence (which owns the
    config-driven gating, the inter-processor GPU cleanup, and the processor teardown ordering).
    """

    def create_dynamic_processor(
        self,
        clip_id: str,
        output_dir: str,
        config: dict[str, Any],
        model_device: str,
        verbose: str,
        seg_session: Any = None,
    ) -> ObjectProcessorDynamic:
        """Construct the dynamic (SegFormer-based) processor. Extension point for subclasses."""
        return ObjectProcessorDynamic(
            clip_id=clip_id,
            output_dir=output_dir,
            config=config,
            model_device=model_device,
            verbose=verbose,
            seg_session=seg_session,
        )

    def create_static_processor(
        self,
        clip_id: str,
        output_dir: str,
        config: dict[str, Any],
        model_device: str,
        verbose: str,
        seg_model: Any,
    ) -> ObjectProcessorStatic:
        """Construct the static (CWIP-based) processor. Extension point for subclasses.

        Raises:
            RuntimeError: If the CWIP model cannot be initialized. The caller treats this as "skip static".
        """
        return ObjectProcessorStatic(
            clip_id=clip_id,
            output_dir=output_dir,
            config=config,
            model_device=model_device,
            verbose=verbose,
            seg_model=seg_model,
        )

    def run(
        self,
        config: dict[str, Any],
        input_data: str,
        clip_id: str,
        camera_name: str,
        video_path: str,
        output_dir: str,
        world_video_path: str | None = None,
        model_device: str = "cuda",
        verbose: str = "INFO",
        trial_frames: int | None = None,
        mask_cache_dir: str | None = None,
        cancel_event: threading.Event | None = None,
        static_seg_model: Any = None,
        dynamic_seg_session: Any = None,
    ) -> dict[str, Any]:
        """Run both dynamic and static objects processors sequentially.

        See :func:`run_object_processors` for the argument and return documentation.
        """
        # An absent world-model video is a supported camera-only request, not an error: without a reference the
        # static (CWIP) check can only run camera-only, so select it from the world video alone. This is the single
        # place the rule lives, so every transport (CLI, service) inherits it; to run camera-only, omit the world
        # video.
        camera_only = world_video_path is None

        results: dict[str, Any] = {}

        # Extract overlap_check config to determine which processors to run. Resolve through the shared helper so this
        # gate sees exactly the same effective section the processors will use, and so an incomplete config raises here
        # instead of quietly selecting no processors at all.
        oc_config = resolve_check_config(
            config,
            ObjectProcessorBase.get_default_config(),
            section_key="av.objects_single_view",
            required_keys=("overlap_check",),
        )
        overlap_check = oc_config.get("overlap_check", {})
        configured_objects = set(overlap_check.keys())

        # Check if any dynamic objects are configured
        has_dynamic_objects = bool(configured_objects & set(ObjectProcessorBase.DYNAMIC_OBJECTS))

        # Check if any static objects are configured
        has_static_objects = bool(configured_objects & set(ObjectProcessorBase.STATIC_OBJECTS))

        # Run dynamic processor if any dynamic objects are configured
        if has_dynamic_objects:
            dynamic_processor = self.create_dynamic_processor(
                clip_id=clip_id,
                output_dir=output_dir,
                config=config,
                model_device=model_device,
                verbose=verbose,
                seg_session=dynamic_seg_session,
            )
            results["dynamic"] = dynamic_processor.process_clip(
                input_data=input_data,
                clip_id=clip_id,
                camera_name=camera_name,
                video_path=video_path,
                output_dir=output_dir,
                trial_frames=trial_frames,
                mask_cache_dir=mask_cache_dir,
                cancel_event=cancel_event,
            )

            # Delete the dynamic processor to release its per-clip state and GPU memory before running CWIP, which
            # avoids GPU OOM. When ``dynamic_seg_session`` was supplied the session itself outlives this -- it belongs
            # to the caller, which reuses it across jobs -- and only the processor's own binding and caches go.
            del dynamic_processor

        # Clear GPU memory between processors to reduce fragmentation and prevent OOM errors
        # This is especially important when switching between different models (SegFormer -> CWIP)
        if has_dynamic_objects and has_static_objects:
            clear_gpu_memory()

        # Run the static processor if any static objects are configured. It runs two-stream when a world video was
        # provided and camera-only otherwise (camera_only derived above).
        if has_static_objects:
            try:
                static_processor = self.create_static_processor(
                    clip_id=clip_id,
                    output_dir=output_dir,
                    config=config,
                    model_device=model_device,
                    verbose=verbose,
                    seg_model=static_seg_model,
                )
            except RuntimeError:
                logger.warning(
                    "Failed to initialize static object processor, skipping static object check", exc_info=True
                )
            else:
                try:
                    results["static"] = static_processor.process_clip(
                        input_data=input_data,
                        clip_id=clip_id,
                        camera_name=camera_name,
                        camera_video_path=video_path,
                        world_video_path=world_video_path,
                        output_dir=output_dir,
                        trial_frames=trial_frames,
                        mask_cache_dir=mask_cache_dir,
                        cancel_event=cancel_event,
                        camera_only=camera_only,
                    )
                finally:
                    # Release CWIP model + caches so back-to-back callers
                    # don't accumulate GPU memory across invocations.
                    del static_processor
                    clear_gpu_memory()

        return results


def run_object_processors(
    config: dict[str, Any],
    input_data: str,
    clip_id: str,
    camera_name: str,
    video_path: str,
    output_dir: str,
    world_video_path: str | None = None,
    model_device: str = "cuda",
    verbose: str = "INFO",
    trial_frames: int | None = None,
    mask_cache_dir: str | None = None,
    cancel_event: threading.Event | None = None,
    static_seg_model: Any = None,
    dynamic_seg_session: Any = None,
) -> dict[str, Any]:
    """Run both dynamic and static objects processors sequentially.

    This function acts as the single entry point for object processing. It conditionally runs the dynamic processor (if
    any dynamic objects are configured in overlap_check) and the static processor (if any static objects are
    configured). The static check runs two-stream when a ``world_video_path`` is provided and camera-only otherwise:
    an absent world video selects camera-only rather than skipping the static check. To run camera-only, omit the
    world video.

    Args:
        config: Configuration dictionary (expects key 'av.objects_single_view'). A non-empty config replaces the
            defaults wholesale, so it must be complete; None or an empty {} uses the defaults.
        input_data: Path to the dataset root or run directory.
        clip_id: Clip identifier.
        camera_name: Camera stream name for poses/intrinsics.
        video_path: Path to the camera video file.
        output_dir: Output directory for results, logs, and visualizations.
        world_video_path: Path to the world video file for two-stream static processing. If None, the static check
            runs camera-only (the camera stream alone, no world-model reference) rather than being skipped. In
            camera-only mode the similarity head is unavailable and the per-class presence mask is derived from the
            type head.
        model_device: Device to run models on ('cuda' or 'cpu'). Defaults to 'cuda'.
        verbose: Logging verbosity level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
            Defaults to 'INFO'.
        trial_frames: If provided, evaluate at most this many steps of the fixed 10 fps object-check
            timeline (useful for debugging).
        mask_cache_dir: Directory for caching inference outputs. When provided:
            - **Dynamic processor**: per-frame SegFormer masks are cached under
              ``<mask_cache_dir>/<video_hash>/frame_<N>.pt`` and reused on subsequent
              runs with the same camera video path.
            - **Static processor**: the full CWIP inference result (similarity indices,
              type indices, and raw type logits) is cached under
              ``<mask_cache_dir>/<pair_hash>/inference.pt`` and reused on subsequent
              runs with the same camera/world video path pair.
            Useful for evaluation runs that re-process the same video repeatedly. Has no effect if the
            corresponding processor is not configured.
        cancel_event: Optional cooperative-cancellation event. When set (by the service layer on client disconnect /
            read-timeout), the per-frame processing loops raise ``ProcessingCancelled`` so this thread unwinds promptly
            instead of running on against a temp dir that is about to be deleted. ``None`` for the local CLI.
        static_seg_model: Pre-loaded segmentation model handed to the static processor, or None to let it load its own.
        dynamic_seg_session: Pre-built SegFormer inference session handed to the dynamic processor, or None to let it
            build its own. Building one initializes the CUDA execution provider, so a caller running jobs concurrently
            should build a single session at startup and pass it here rather than paying that cost per job.

    Returns:
        Dict[str, Any]: Combined results dictionary with keys:
            - 'dynamic': Results from the dynamic (SegFormer) processor (if run).
            - 'static': Results from the static (CWIP) processor (if run).
    """
    return ObjectCheckRunner().run(
        config=config,
        input_data=input_data,
        clip_id=clip_id,
        camera_name=camera_name,
        video_path=video_path,
        output_dir=output_dir,
        world_video_path=world_video_path,
        model_device=model_device,
        verbose=verbose,
        trial_frames=trial_frames,
        mask_cache_dir=mask_cache_dir,
        cancel_event=cancel_event,
        static_seg_model=static_seg_model,
        dynamic_seg_session=dynamic_seg_session,
    )


def get_default_config() -> dict[str, Any]:
    """Load the default configuration from cosmos_evaluator.yaml.

    Returns:
        Dict[str, Any]: Dict with key 'av.objects_single_view' containing objects config.
    """
    return ObjectProcessorBase.get_default_config()
