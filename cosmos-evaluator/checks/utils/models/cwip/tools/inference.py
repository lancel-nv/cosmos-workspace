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
CWIP Model Inference Script

This script loads a CWIP model from HuggingFace format and runs inference on either
dummy data or real camera/world video pairs.

Example Usage:

1. Load model and run test inference with dummy data:
   python checks/utils/models/cwip/tools/inference.py \
       --model_path /path/to/hf_model \
       --device cuda \
       --test_inference

2. Run inference on actual camera and world videos:
   python checks/utils/models/cwip/tools/inference.py \
       --model_path /path/to/hf_model \
       --device cuda \
       --camera_video_path /path/to/camera_video.mp4 \
       --world_video_path /path/to/world_video.mp4 \
       --output_dir /path/to/output

3. Use default model path from Bazel runfiles (in-tree Git LFS checkpoint):
   python checks/utils/models/cwip/tools/inference.py \
       --device cuda \
       --test_inference

4. Load model on CPU:
   python checks/utils/models/cwip/tools/inference.py \
       --model_path /path/to/hf_model \
       --device cpu \
       --test_inference

5. Multi-GPU inference with tensor parallelism (TP=2):
   torchrun --nproc_per_node=2 checks/utils/models/cwip/tools/inference.py \
       --model_path /path/to/hf_model \
       --tp_size 2 \
       --test_inference

6. Multi-GPU inference with TP=4 and frame parallelism (FP=2):
   torchrun --nproc_per_node=8 checks/utils/models/cwip/tools/inference.py \
       --model_path /path/to/hf_model \
       --tp_size 4 \
       --fp_size 2 \
       --test_inference

7. Multi-GPU inference with FP=4 and true CP=2 (Megatron CP = FP x CP; PatchFormer shards patches + gather attention):
   torchrun --nproc_per_node=8 checks/utils/models/cwip/tools/inference.py \
       --model_path /path/to/hf_model \
       --fp_size 4 \
       --true_cp_size 2 \
       --test_inference

8. Camera-only mode (skip world encoder + similarity head; only camera embedding + object-type path):
   dazel run //checks/utils/models/cwip/tools:inference -- \
       --model_path /path/to/hf_model \
       --device cuda \
       --camera_video_path /path/to/camera_video.mp4 \
       --output_dir /path/to/output \
       --camera_only_mode \
       --image_size 720,1280

   # NOTE: unlike the two-branch path (which resizes the camera down to the world frame size),
   # camera-only has no world frame and otherwise runs at the camera's native resolution. Use
   # --image_size to bound patch-former memory (or lower --batch_size) for native-res input.

   # or with dummy data:
   dazel run //checks/utils/models/cwip/tools:inference -- \
       --model_path /path/to/hf_model \
       --device cuda \
       --test_inference \
       --camera_only_mode

9. Save the static-check presence masks (the per-patch masks correspondence and hallucination score against), here the
   union of the similarity verdict and the type-head mask with a stricter traffic-light threshold:
   dazel run //checks/utils/models/cwip/tools:inference -- \
       --camera_video_path /path/to/camera_video.mp4 \
       --world_video_path /path/to/world_video.mp4 \
       --output_dir /path/to/output \
       --save_presence_masks \
       --presence_source union \
       --type_confidence_threshold 0.5 \
       --per_class_thresholds TRAFFIC_LIGHT_RED=0.7,TRAFFIC_LIGHT_GREEN=0.7
"""

import argparse
from collections.abc import Callable
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torchinfo import summary
from torchvision.transforms import v2

from checks.utils.models.cwip.cwip import CWIPInference
from checks.utils.models.cwip.modules import parallelism as parallelism_module
from checks.utils.models.cwip.modules.class_mapping import get_object_type_class_mapping
from checks.utils.models.cwip.modules.parallelism import initialize_model_parallel
from checks.utils.models.cwip.modules.presence import (
    DEFAULT_TWO_STREAM_PRESENCE_SOURCE,
    DEFAULT_TYPE_CONFIDENCE_THRESHOLD,
    PresenceSource,
    TypeMaskGate,
)
from checks.utils.models.cwip.tools.batch_utils import merge_batch_outputs
from checks.utils.models.cwip.visualization import image_video_utils
from checks.utils.models.cwip.visualization.presence_vis import (
    PresenceOptions,
    compare_masks,
    compute_presence_masks,
    presence_label,
    write_presence_outputs,
)
from checks.utils.models.cwip.visualization.world_model_utils import ObjectType

_get_runfiles_path: Callable[..., Any] | None = None
try:
    from utils.bazel import get_runfiles_path as _get_runfiles_path
except ImportError:
    pass

get_runfiles_path = _get_runfiles_path

logger = logging.getLogger(__name__)


def _print_inference_success_summary(output_batch: dict, *, label: str) -> None:
    """Log a success banner and key tensor shapes from a forward output dictionary.

    Args:
        output_batch: Mapping from :meth:`~checks.utils.models.cwip.cwip.CWIPInference.forward` (after any batch merge).
            Must include ``camera_global_features``, ``world_global_features``, ``camera_spatial_features_list``, and
            ``world_spatial_features_list``. Optionally includes ``patch_similarity_logits`` or ``patch_type_logits``
            when present.
        label: Text inserted after the checkmark (e.g. ``\"Inference\"`` or ``\"Test inference\"``).

    Returns:
        ``None``. Side effect: writes INFO lines via :mod:`logging`.
    """
    logger.info("\n✅ %s successful!", label)
    logger.info("  Camera global features shape: %s", output_batch["camera_global_features"].shape)
    if output_batch.get("world_global_features") is not None:
        logger.info("  World global features shape: %s", output_batch["world_global_features"].shape)
    logger.info(
        "  Camera spatial list last element shape: %s",
        output_batch["camera_spatial_features_list"][-1].shape,
    )
    if output_batch.get("world_spatial_features_list") is not None:
        logger.info(
            "  World spatial list last element shape: %s",
            output_batch["world_spatial_features_list"][-1].shape,
        )
    if output_batch.get("patch_similarity_logits") is not None:
        logger.info("  Patch logits shape: %s", output_batch["patch_similarity_logits"].shape)
    elif output_batch.get("patch_type_logits") is not None:
        logger.info("  Patch logits shape: %s", output_batch["patch_type_logits"].shape)


def _print_gpu_memory_usage(label: str = "after first batch") -> None:
    """Log GPU memory stats if CUDA is available.

    Args:
        label: Descriptive context string inserted into the output header (e.g. ``\"after first batch\"``).

    Returns:
        ``None``. Side effect: INFO logs for allocated, reserved, and max allocated memory (MB)
        when ``torch.cuda.is_available()``; otherwise no-op.
    """
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    allocated_mb = torch.cuda.memory_allocated() / 1024**2
    reserved_mb = torch.cuda.memory_reserved() / 1024**2
    max_allocated_mb = torch.cuda.max_memory_allocated() / 1024**2
    logger.info("\nGPU Memory Usage (%s):", label)
    logger.info("  Allocated: %.2f MB", allocated_mb)
    logger.info("  Reserved:  %.2f MB", reserved_mb)
    logger.info("  Max Allocated: %.2f MB", max_allocated_mb)


# Large per-frame feature maps (B, D, H_out, W_out) that ``create_tiled_video`` moves to the GPU
# itself, one tensor at a time, freeing each chunk-wise before touching the next. Excluding them
# from the bulk move-back avoids a ~2x peak spike (both maps resident on GPU at once) for the
# two-branch path; the viz helper still pulls them from CPU on demand.
_VIZ_LAZY_MOVE_KEYS = ("camera_patch_features", "camera_type_features")


def _move_batch_to_device(batch: dict, device: str | torch.device, *, skip_keys: tuple[str, ...] = ()) -> dict:
    """Move every tensor (and tensors inside list values) in ``batch`` to ``device``.

    Used to offload per-batch forward outputs to CPU during the chunked inference loop so the
    accumulated results don't grow unbounded on the GPU (the large ``camera_type_features`` /
    ``camera_patch_features`` maps are ~tens of MB per frame at high output resolution), then to
    move the merged result back to the compute device before visualization. Keys in ``skip_keys``
    are left untouched. Non-tensor values (e.g. the ``viz_*_aligner_present`` booleans) pass
    through unchanged.
    """
    moved: dict = {}
    for key, value in batch.items():
        if key in skip_keys:
            moved[key] = value
        elif isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, list):
            moved[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
        else:
            moved[key] = value
    return moved


def run_loaded_model_inference(
    model: CWIPInference,
    *,
    test_inference: bool = False,
    camera_video_path: str | None = None,
    world_video_path: str | None = None,
    output_dir: str | None = None,
    batch_size: int = 4,
    fp_size: int = 1,
    camera_only: bool = False,
    image_size: tuple[int, int] | None = None,
    presence: PresenceOptions | None = None,
) -> None:
    """Run inference on real video pairs or on synthetic tensors (test mode).

    Uses the device of ``model`` parameters (e.g. ``cuda:{local_rank}`` under ``torchrun``). When frame parallelism is
    active (``fp_world > 1``), each batch splits inputs with
    :func:`~checks.utils.models.cwip.modules.parallelism.split_tensor_for_fp`, runs
    :meth:`~checks.utils.models.cwip.cwip.CWIPInference.forward`, gathers with
    :func:`~checks.utils.models.cwip.modules.parallelism.gather_features_from_fp`, then trims FP padding. Video mode
    runs rank-0 visualization when distributed.

    Args:
        model: Loaded :class:`~checks.utils.models.cwip.cwip.CWIPInference` instance.
        test_inference: If True and no video paths are set, run one forward on random tensors sized to
            ``model.image_size``.
        camera_video_path: Path to the camera RGB (or equivalent) video; paired with ``world_video_path`` for video
            mode. Both must be set to enable video mode.
        world_video_path: Path to the world/scene video.
        output_dir: Directory for :meth:`~checks.utils.models.cwip.cwip.CWIPInference.create_inference_visualizations`
            (video mode, rank 0).
        batch_size: Number of frames per forward in video mode (chunked to limit memory).
        fp_size: CLI frame-parallel degree (must be ``>= 1``); used as dummy batch size in test mode (at least one frame
            per FP rank when ``fp_world > 1``).
        camera_only: If True, run the camera-only path. Only ``camera_video_path`` is required (``world_video_path`` is
            ignored); ``world_image=None`` is passed to ``forward`` so the world encoder and similarity head are
            skipped. In test mode, no dummy world tensor is built.
        image_size: Optional ``(H, W)`` to resize decoded frames to before the forward pass.
            :meth:`CWIPInference.forward` derives its processing resolution from the *input* shape, so this directly
            sets the patch-former resolution (and thus peak memory). Chiefly useful in ``camera_only`` mode: with a
            real world video the two-branch path resizes the camera down to the world frame size, but camera-only has
            no world frame and would otherwise run at the camera's native resolution. Applied to both camera and
            (black) world tensors.
        presence: If set (video mode), also write the static-check presence masks to ``output_dir`` on rank 0; see
            :func:`~checks.utils.models.cwip.visualization.presence_vis.compute_presence_masks` and
            :func:`~checks.utils.models.cwip.visualization.presence_vis.write_presence_outputs`. Requires
            ``output_dir``.

    Returns:
        ``None``. Side effects: prints logs; may write files under ``output_dir`` on rank 0.
    """
    if fp_size < 1:
        raise ValueError(f"fp_size must be >= 1, got {fp_size}")
    if presence is not None and not output_dir:
        raise ValueError("presence masks are written to output_dir, which is not set")

    # Sync the runner's camera_only flag with model.config.inference.camera_only_mode so that
    # the preprocessing decision (world_image=None vs. real tensor) and the model's own branch
    # gate are always in agreement.  Two cases can diverge without this:
    #   1. caller passes camera_only=True but model.config.inference.camera_only_mode=False →
    #      runner omits the world tensor, model.forward() hits ValueError
    #   2. model.config.inference.camera_only_mode=True but caller passes camera_only=False →
    #      runner builds a world tensor the model silently discards
    # Resolution: the param wins if it is True (matching what load_from_hf_format does when
    # --camera_only_mode is set); otherwise the model config is the source of truth.
    config_camera_only: bool = getattr(model.config.inference, "camera_only_mode", False)
    if camera_only and not config_camera_only:
        model.config.inference.camera_only_mode = True
        config_camera_only = True
    elif config_camera_only and not camera_only:
        logger.warning(
            "model.config.inference.camera_only_mode=True but camera_only=False was passed; "
            "using camera-only mode from model config."
        )
        camera_only = True

    device = str(next(model.parameters()).device)
    is_distributed = dist.is_initialized() and dist.get_world_size() > 1
    try:
        fp_world = parallelism_module.get_frame_parallel_world_size()
        fp_rank = parallelism_module.get_frame_parallel_rank()
    except (RuntimeError, AttributeError):
        fp_world, fp_rank = 1, 0

    if camera_video_path and (world_video_path or camera_only):
        print("\nRunning inference on provided videos...")
        print(f"  Camera video: {camera_video_path}")
        if camera_only:
            logger.info("  World video: (skipped — camera-only mode)")
        else:
            print(f"  World video: {world_video_path}")
        model.eval()

        # Preprocess videos
        print("\nPreprocessing videos...")
        # In camera-only mode, decode only the camera stream and synthesize a black world tensor of matching shape. The
        # black world is used solely so that visualization tiling has a world frame to reference; it is never fed to the
        # forward pass (world_image=None below), so no world FLOPs are incurred.
        video_data = image_video_utils.preprocess_world_camera_video_pair(
            world_video_path="" if camera_only else (world_video_path or ""),
            camera_video_path=camera_video_path,
            num_frames=-1,  # Use all frames
            transform=None,  # Model will apply its own transform
            sample_method="uniform",
            verbose=True,
            create_black_world_if_missing=camera_only,
        )

        # Keep the full decoded video on the CPU. ``batch_size`` bounds the forward, and
        # :meth:`CWIPInference.forward` moves each chunk to the device itself, so the whole clip
        # never needs to sit on the GPU. Staging all frames on the GPU up front (especially at
        # native resolution) can exhaust memory before inference even starts. Frames are moved to
        # the device per-chunk in the loop below and, for the rank-0 visualization, staged just
        # before rendering.
        camera_video = video_data["camera_video"]  # (N, C, H, W) on CPU
        world_video = video_data["world_video"]  # (N, C, H, W) on CPU
        # This tool materializes the black world (the default): camera-only still references its shape for the shape
        # print and the visualization tile, so it is always a tensor here, never None.
        assert world_video is not None
        num_frames = video_data["num_frames"]

        # Optional explicit resize, done on the CPU before anything reaches the GPU. ``forward``
        # sets its processing resolution from the input shape, so this bounds patch-former memory.
        # In camera-only mode this is the knob that replaces the two-branch "resize camera down to
        # world frame size" behavior.
        if image_size is not None:
            orig_hw = tuple(camera_video.shape[-2:])
            camera_video = v2.functional.resize(
                camera_video, list(image_size), interpolation=v2.InterpolationMode.BILINEAR
            )
            world_video = v2.functional.resize(
                world_video, list(image_size), interpolation=v2.InterpolationMode.BILINEAR
            )
            logger.info("  Resized frames %s -> %s (--image_size)", orig_hw, tuple(image_size))

        # CPU-resident full video; per-chunk slices are moved to the device inside the loop, and
        # the whole thing is staged to the device only for rank-0 visualization.
        data_batch = {
            "camera_image": camera_video,
            "world_image": world_video,
        }

        print(f"\nProcessed {num_frames} frames")
        print(f"  Camera video shape: {camera_video.shape}")
        world_note = " (synthesized black; camera-only)" if camera_only else ""
        print(f"  World video shape: {world_video.shape}{world_note}")

        # Run inference in batches to avoid OOM
        print(f"\nRunning inference on {num_frames} frames (batch_size={batch_size})...")
        if fp_world > 1:
            print(
                f"  Frame parallelism: FP world_size={fp_world}, this FP rank={fp_rank}. "
                f"Each rank runs the same loop but forwards only a 1/{fp_world} slice per batch "
                f"(see 'local forward' below); duplicate lines are one per GPU process."
            )

        with torch.no_grad():
            inference_start_time = time.time()

            # Process frames in batches
            all_outputs = []
            for i in range(0, num_frames, batch_size):
                end_idx = min(i + batch_size, num_frames)
                chunk_n = end_idx - i
                cam_chunk = camera_video[i:end_idx]
                world_chunk = None if camera_only else world_video[i:end_idx]
                if fp_world > 1:
                    cam_chunk, real_start, real_end = parallelism_module.split_tensor_for_fp(
                        cam_chunk, return_indices=True
                    )
                    if not camera_only:
                        assert world_chunk is not None
                        world_chunk_split = parallelism_module.split_tensor_for_fp(world_chunk)
                        assert isinstance(world_chunk_split, torch.Tensor)
                        world_chunk = world_chunk_split
                batch_output = model.forward(camera_image=cam_chunk, world_image=world_chunk)
                if fp_world > 1:
                    batch_output = parallelism_module.gather_features_from_fp(batch_output)
                model.trim_output_to_num_frames(batch_output, chunk_n)
                # Offload this chunk's outputs to CPU so the accumulated results stay bounded on
                # the GPU. Otherwise every chunk's full-resolution feature maps pile up in device
                # memory for the whole video and, combined with the (constant) per-forward
                # deformable-attention transient, OOM part-way through long clips.
                all_outputs.append(_move_batch_to_device(batch_output, "cpu"))
                del batch_output
                local_n = int(cam_chunk.shape[0])
                if fp_world > 1:
                    if real_start < real_end:
                        g0, g1 = i + real_start, i + real_end - 1
                        idx_part = f"real global frame indices on this rank: {g0}-{g1}"
                    else:
                        idx_part = "FP padding only on this rank (no real frames)"
                    print(
                        f"  [FP rank {fp_rank}/{fp_world}] window {i}-{end_idx - 1} ({chunk_n} real frames); "
                        f"forward batch rows: {local_n}; {idx_part}"
                    )
                else:
                    print(f"  Processed frames {i}-{end_idx - 1} ({chunk_n} frames)")

                # Log GPU memory usage after first batch (no-op if CUDA unavailable)
                if i == 0:
                    _print_gpu_memory_usage("after first batch")

            # Merge all batch outputs from this rank (currently on CPU after per-chunk offload).
            output_batch = merge_batch_outputs(all_outputs)
            del all_outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            inference_time = time.time() - inference_start_time
            print(f"Model inference time: {inference_time:.3f}s")

            # Sync before rank-0-only work: avoids NCCL peers diverging while one rank runs long CPU/GPU visualization,
            # and ensures non-zero ranks do not exit and tear down the process group early.
            if is_distributed:
                dist.barrier()

            # Only rank 0 creates visualizations.
            # HF inference is a deployment artifact; we only emit the 7 outputs
            # the user opted into:
            #   - similarity_radial_iter_*.png            (Step 1, radial-only)
            #   - tiled_video_iter_*.mp4                  (Step 2)
            #   - combined_patch_vis_iter_*.mp4           (Step 3)
            #   - class_percentage_radial_iter_*.png      (Step 3)
            #   - class_pixel_count_radial_iter_*.png     (Step 3)
            #   - detailed_patch_similarity_vis_iter_*.mp4 (Step 4)
            #   - detailed_patch_type_vis_iter_*.mp4       (Step 4)
            if not is_distributed or dist.get_rank() == 0:
                presence_mask: np.ndarray | None = None
                presence_mask_label: str | None = None
                if presence is not None:
                    mapping_cls = get_object_type_class_mapping(
                        getattr(model.config, "object_type_class_mapping", None)
                    )
                    similarity_logits = output_batch.get("patch_similarity_logits")
                    type_logits = output_batch.get("patch_type_logits")
                    correspondence, hallucination = compute_presence_masks(
                        similarity_logits, type_logits, presence, mapping_cls
                    )
                    write_presence_outputs(
                        Path(output_dir or ""),
                        camera_video,
                        correspondence,
                        hallucination,
                        presence,
                        compared=compare_masks(similarity_logits, type_logits, presence, mapping_cls),
                        camera_only=camera_only,
                        fps=float(video_data.get("camera_fps") or 30.0),
                    )
                    # combined_patch_vis row 3 then shows what the check scores (either check's mask) instead of the
                    # soft similarity mask, labeled with the presence source.
                    presence_mask = np.maximum(correspondence, hallucination)
                    presence_mask_label = presence_label(presence, camera_only=camera_only)
                # Visualization operates on device tensors; stage the CPU-resident frames and the
                # CPU-offloaded outputs to the device just before rendering. The per-step
                # empty_cache inside create_inference_visualizations keeps this bounded. The large
                # tiled-video feature maps are left on CPU (create_tiled_video moves them one at a
                # time) to avoid a double-size GPU spike.
                data_batch = _move_batch_to_device(data_batch, device)
                output_batch = _move_batch_to_device(output_batch, device, skip_keys=_VIZ_LAZY_MOVE_KEYS)
                model.create_inference_visualizations(
                    output_batch=output_batch,
                    data_batch=data_batch,
                    output_dir=output_dir or "",
                    iteration=0,
                    similarity_plots="radial",
                    save_patch_data_dumps=False,
                    presence_mask=presence_mask,
                    presence_label=presence_mask_label,
                )

            if is_distributed:
                dist.barrier()

        _print_inference_success_summary(output_batch, label="Inference")

    elif test_inference:
        print("\nRunning test inference on dummy data...")
        model.eval()

        # Create dummy inputs — use fp_size frames so each rank gets at least 1
        test_batch_size = fp_size
        # model.image_size is always normalized to an (H, W) tuple in CWIPInference.__init__;
        # an explicit --image_size wins.
        img_h, img_w = image_size if image_size is not None else model.image_size

        dummy_camera = torch.randn(test_batch_size, 3, img_h, img_w, dtype=model.precision, device=device)
        # Camera-only: no world tensor is built or forwarded (world_image=None).
        dummy_world = (
            None if camera_only else torch.randn(test_batch_size, 3, img_h, img_w, dtype=model.precision, device=device)
        )
        if fp_world > 1:
            dummy_camera_split = parallelism_module.split_tensor_for_fp(dummy_camera)
            assert isinstance(dummy_camera_split, torch.Tensor)
            dummy_camera = dummy_camera_split
            if not camera_only:
                assert dummy_world is not None
                dummy_world_split = parallelism_module.split_tensor_for_fp(dummy_world)
                assert isinstance(dummy_world_split, torch.Tensor)
                dummy_world = dummy_world_split

        with torch.no_grad():
            inference_start_time = time.time()
            output_batch = model.forward(camera_image=dummy_camera, world_image=dummy_world)
            if fp_world > 1:
                output_batch = parallelism_module.gather_features_from_fp(output_batch)
            model.trim_output_to_num_frames(output_batch, test_batch_size)

            inference_time = time.time() - inference_start_time
            print(f"Model inference time: {inference_time:.3f}s")

            # Log GPU memory usage after first batch (no-op if CUDA unavailable)
            _print_gpu_memory_usage("after first batch")

        _print_inference_success_summary(output_batch, label="Test inference")


def load_from_hf_format(
    model_path: str | Path | None = None,
    device: str = "cuda",
    tp_size: int = 1,
    fp_size: int = 1,
    true_cp_size: int | None = None,
    sequence_parallel: bool = False,
    channels_last: bool = False,
    use_hf: bool = False,
    camera_only_mode: bool = False,
) -> CWIPInference:
    """Load a :class:`~checks.utils.models.cwip.cwip.CWIPInference` checkpoint from HuggingFace layout.

    Portable path: reads ``config.json`` and ``model.safetensors`` (or sharded index) under ``model_path`` without
    Hydra. When ``tp_size``, ``fp_size``, or ``true_cp_size`` imply multi-GPU use, initializes ``torch.distributed`` and
    Megatron parallel state before loading.

    Always-on accelerator: TF32 fast path for residual fp32 matmuls is enabled via
    ``torch.set_float32_matmul_precision("high")`` -- shape-independent and lossless for the bf16 model. The
    ``channels_last`` flag is opt-in and also shape-independent (no recompile cost when the input ``(H, W)`` changes).

    Args:
        model_path: Directory containing ``config.json`` and weights. If ``None`` or blank, resolves the in-tree
            Git LFS checkpoint from Bazel runfiles (requires Bazel test utils helper).
        device: Requested device (``\"cuda\"``, ``\"cpu\"``, or ``\"cuda:N\"``). Under multi-GPU init, overridden to
            ``cuda:{LOCAL_RANK}``.
        tp_size: Tensor-parallel world size (Megatron TP). Checkpoint tensors are sharded per rank when ``tp_size > 1``.
        fp_size: Frame-parallel size (scatter/gather frames along the FP process group).
        true_cp_size: True context-parallel degree (PatchFormer patch-token sharding). Megatron
            ``context_parallel_size`` is ``fp_size * true_cp_size``. If ``None``, reads ``true_context_parallel_size``
            from ``config.json`` (default ``1``).
        sequence_parallel: Enables Megatron sequence parallelism when ``tp_size > 1``.
        channels_last: If True, convert the loaded model to ``torch.channels_last`` memory format and have ``forward``
            reshape its 4-D image inputs to match. Typically helps conv-heavy image-encoder stems on Ampere+ by 5--15%.
            Works with any input shape.
        use_hf: If True, load via :meth:`CWIPInference.from_pretrained` -- the thin wrapper around
            :meth:`transformers.PreTrainedModel.from_pretrained` (HF Hub repo IDs, ``revision``, ``cache_dir``,
            ``device_map``, etc. all supported). If False (default), use the legacy :meth:`CWIPInference.from_hf_format`
            custom loader, which does not require ``transformers`` to be installed. Both loaders consume the same
            on-disk layout (``config.json`` + ``model.safetensors``).
        camera_only_mode: If True, override the loaded config to run only the camera branch (encoder/aligner, camera
            global embedding, and object-type prediction). The world encoder and similarity head are skipped at forward
            time; ``world_image`` may be ``None``.

    Returns:
        A :class:`~checks.utils.models.cwip.cwip.CWIPInference` module with weights loaded on the resolved device. Also
        prints memory, architecture info, and an optional ``torchinfo`` summary on single-process runs.
    """
    if fp_size < 1:
        raise ValueError(f"fp_size must be >= 1, got {fp_size}")

    # TF32 is shape-independent and lossless for bf16/fp16 inference. Apply unconditionally before any matmul-heavy work
    # (model construction below touches Linear init / Xavier init, which run in fp32).
    torch.set_float32_matmul_precision("high")

    # Resolve the model path first, before reading ``config.json`` below, so that an unset ``--model_path`` (which
    # falls back to the in-tree Git LFS checkpoint) still contributes ``true_context_parallel_size`` to the parallelism
    # setup instead of being silently forced to 1.
    if model_path is None or not str(model_path).strip():
        if get_runfiles_path is None:
            raise FileNotFoundError(
                "No --model_path provided and Bazel runfiles helper not available. Please pass --model_path explicitly."
            )
        # The CWIP model weights live in-tree at ``checks/utils/models/cwip/checkpoints/`` (tracked via Git LFS) and
        # are exposed through the ``//checks/utils/models/cwip/checkpoints:cwip_model`` filegroup, mirroring the
        # SegFormer pattern. This matches the sibling lookup in
        # ``checks/objects/single_view/object_processor_static.py``.
        cwip_model = get_runfiles_path("checks/utils/models/cwip/checkpoints/model.safetensors")
        if cwip_model is None:
            raise FileNotFoundError(
                "Could not resolve model.safetensors in Bazel runfiles. Please pass --model_path explicitly."
            )

        model_path = Path(cwip_model).parent
    else:
        # Resolve relative paths - try multiple strategies
        model_path_str = str(model_path)
        if not os.path.isabs(model_path_str):
            # Strategy 1: Try resolving through runfiles
            try:
                resolved = get_runfiles_path(model_path_str) if get_runfiles_path else None
                if resolved and os.path.exists(resolved):
                    model_path = Path(resolved)
                else:
                    # Strategy 2: Try relative to workspace root (BUILD_WORKSPACE_DIRECTORY env var)
                    workspace_root = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
                    if workspace_root:
                        abs_path = os.path.join(workspace_root, model_path_str)
                        if os.path.exists(abs_path):
                            model_path = Path(abs_path)
                        else:
                            # Strategy 3: Try as absolute path from current working directory
                            model_path = Path(model_path_str).resolve()
                    else:
                        # Strategy 4: Try as absolute path from current working directory
                        model_path = Path(model_path_str).resolve()
            except (OSError, TypeError, ValueError) as exc:
                # Fallback: resolve relative to current directory (path / runfiles edge cases)
                logger.warning(
                    "Relative model path resolution failed; using workspace/cwd fallback: %s",
                    exc,
                )
                workspace_root = os.environ.get("BUILD_WORKSPACE_DIRECTORY") or os.getcwd()
                abs_path = os.path.join(workspace_root, model_path_str)
                if os.path.exists(abs_path):
                    model_path = Path(abs_path)
                else:
                    model_path = Path(model_path_str).resolve()
        else:
            model_path = Path(model_path_str)

    # Resolve true_cp_size from config.json if not explicitly provided
    if true_cp_size is None and model_path is not None:
        config_json_path = Path(model_path) / "config.json"
        if config_json_path.exists():
            with open(config_json_path, "r") as f:
                saved_config = json.load(f)
            true_cp_size = saved_config.get("true_context_parallel_size", 1)
    if true_cp_size is None:
        true_cp_size = 1

    # Megatron's context_parallel_size = FP x true_CP
    megatron_cp_size = fp_size * true_cp_size

    # Initialize multi-GPU parallelism if requested
    if tp_size > 1 or fp_size > 1 or true_cp_size > 1 or sequence_parallel:
        # Fail fast with a clear message when parallelism is requested but CUDA is unavailable or the user asked for
        # CPU. ``dist.init_process_group(backend="nccl")`` and ``torch.cuda.set_device(...)`` below would otherwise
        # raise opaque NCCL/CUDA errors.
        if device == "cpu" or not torch.cuda.is_available():
            raise RuntimeError(
                "Multi-GPU parallelism was requested "
                f"(tp_size={tp_size}, fp_size={fp_size}, true_cp_size={true_cp_size}, "
                f"sequence_parallel={sequence_parallel}) but "
                + ("--device=cpu was specified" if device == "cpu" else "CUDA is not available")
                + ". Enable CUDA or set all parallelism sizes to 1 (and sequence_parallel=False)."
            )
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        global_rank = dist.get_rank()
        # Prefer torchrun's LOCAL_RANK so each process binds to the correct GPU when global rank ordering differs from
        # local device index (multi-node / some launchers).
        local_rank = int(os.environ.get("LOCAL_RANK", str(global_rank)))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

        initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            context_parallel_size=megatron_cp_size,
            sequence_parallel=sequence_parallel,
        )
        parallelism_module.sync_from_megatron(
            frame_parallel_size=fp_size,
            true_context_parallel_size=true_cp_size,
        )
        print(
            f"[Rank {global_rank} local {local_rank}] Initialized parallelism: TP={tp_size}, FP={fp_size}, "
            f"True CP={true_cp_size} (Megatron CP; PatchFormer token shard + gather), SP={sequence_parallel}"
        )

    if use_hf:
        if not hasattr(CWIPInference, "from_pretrained"):
            raise RuntimeError(
                "--use_hf requires the `transformers` library to be importable. "
                "Run via Bazel (the cwip py_library lists transformers as a dep) "
                "or `pip install transformers` in your environment."
            )
        print(f"Loading CWIP model via transformers.PreTrainedModel.from_pretrained: {model_path}")
        # `low_cpu_mem_usage=False` so HF instantiates real (non-meta) modules under the active Megatron parallel state.
        # `channels_last` is applied below, after `.to(device)`, so we don't pass it to `from_pretrained`.
        model = CWIPInference.from_pretrained(str(model_path), low_cpu_mem_usage=False)
        model = model.to(device)
    else:
        print(f"Loading CWIP model from HuggingFace format: {model_path}")
        # Custom loader: reads config.json + safetensors directly; no `transformers` import.
        model = CWIPInference.from_hf_format(model_path=str(model_path), device=device)

    # Check if model is bfloat16 and GPU doesn't support it - convert to float16
    if device.startswith("cuda") and model.precision == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("⚠️  Model is bfloat16 but GPU does not support BF16. Converting to float16...")
        model = model.half()
        print(f"  Converted model precision: {model.precision}")

    # Opt-in accelerator: channels_last is a pure memory-layout change on the existing parameter tensors.
    # Shape-independent; safe under dynamic shapes.
    if channels_last:
        if not device.startswith("cuda"):
            print("⚠️  --channels_last requested but device is not CUDA; skipping.")
        else:
            model.to(memory_format=torch.channels_last)
            model._use_channels_last = True

    # Camera-only override: load the full two-branch checkpoint but skip the world encoder and similarity head at
    # forward time. Applied after load so the weights (which still include the world branch) are read normally.
    if camera_only_mode:
        model.config.inference.camera_only_mode = True
        logger.info("  Camera-only mode: ENABLED (world encoder + similarity head skipped at forward)")

    # This is the visualization CLI, so always retain the frozen-backbone (pre-alignment) encoder layer in addition to
    # the alignment-final layer. Without it the encoder only fetches ``[-1]`` and the tiled video has a single feature
    # panel (camera-only 1x2, or no backbone row in the two-branch layout); with it the tiled video shows both the
    # backbone and after-alignment feature maps. Costs one extra layer's activations per forward.
    model.config.viz_capture_backbone_features = True
    print("  Backbone-feature capture: ENABLED (tiled video shows backbone + after-alignment feature maps)")

    print(f"✅ Successfully loaded model from: {model_path}")

    # Print GPU memory usage after loading
    if device.startswith("cuda"):
        _print_gpu_memory_usage("after model load")

    # Print model info
    print("\nModel information:")
    print(f"  Precision: {model.precision}")

    # Handle image_size as either int or tuple/list
    if isinstance(model.image_size, (tuple, list)):
        img_h, img_w = model.image_size
    else:
        img_h = img_w = model.image_size
    print(f"  Image size: {img_h}x{img_w}")

    print(f"  Camera encoder output dim: {model.camera.output_dim}")
    print(f"  World encoder output dim: {model.world.output_dim}")
    if model.patch_classifier:
        print("  Patch classifier: Present")

    # Print model summary (skip for multi-GPU — torchinfo runs a forward pass that conflicts with distributed process
    # groups)
    is_distributed = dist.is_initialized() and dist.get_world_size() > 1
    if not is_distributed:
        print("\nModel Summary:")
        print("-" * 80)
        # In camera-only mode the world branch is skipped, so feed a single camera input (``world_image`` defaults to
        # ``None`` in ``forward``).
        summary_input_data = [torch.randn(1, 3, img_h, img_w, dtype=model.precision, device=device)]
        if not camera_only_mode:
            summary_input_data.append(torch.randn(1, 3, img_h, img_w, dtype=model.precision, device=device))
        try:
            print(
                summary(
                    model,
                    input_data=summary_input_data,
                    depth=5,
                    col_names=["output_size", "num_params", "trainable"],
                    row_settings=["var_names"],
                    verbose=0,
                )
            )
        except Exception as e:  # noqa: BLE001 - torchinfo summary is cosmetic; log and continue
            print(f"Could not print model summary: {e}")
    else:
        print("\nModel Summary: skipped (multi-GPU mode)")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """``argparse`` help formatter: show argument defaults and keep the epilog (examples) verbatim."""


def _parse_presence_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> PresenceOptions | None:
    """Validate the presence-mask flags; return :class:`PresenceOptions` when ``--save_presence_masks`` is set."""
    customized = (
        args.presence_source != DEFAULT_TWO_STREAM_PRESENCE_SOURCE
        or args.type_mask_gate != TypeMaskGate.CONFIDENCE
        or args.type_confidence_threshold != DEFAULT_TYPE_CONFIDENCE_THRESHOLD
        or args.per_class_thresholds
    )
    if not args.save_presence_masks:
        if customized:
            parser.error("the presence-mask flags take effect only with --save_presence_masks")
        return None
    if not (args.camera_video_path and args.output_dir and (args.world_video_path or args.camera_only_mode)):
        parser.error(
            "--save_presence_masks needs video mode: --camera_video_path, --output_dir, and --world_video_path or "
            "--camera_only_mode"
        )
    if not 0.0 <= args.type_confidence_threshold <= 1.0:
        parser.error(f"--type_confidence_threshold must be in [0, 1], got {args.type_confidence_threshold}")
    per_class: dict[str, float] = {}
    for item in (args.per_class_thresholds or "").split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, value = item.partition("=")
        name = name.strip().upper()
        try:
            threshold = float(value) if sep else float("nan")
        except ValueError:
            threshold = float("nan")
        if not 0.0 <= threshold <= 1.0:
            parser.error(f"--per_class_thresholds entries must be ObjectType=threshold in [0, 1], got {item!r}")
        if name not in ObjectType.__members__:
            parser.error(f"--per_class_thresholds: unknown ObjectType {name!r}; valid: {list(ObjectType.__members__)}")
        per_class[name] = threshold
    if args.camera_only_mode and args.presence_source != DEFAULT_TWO_STREAM_PRESENCE_SOURCE:
        logger.warning("--presence_source is ignored with --camera_only_mode: both masks use the type-head mask")
    return PresenceOptions(
        presence_source=args.presence_source,
        type_mask_gate=args.type_mask_gate,
        type_confidence_threshold=args.type_confidence_threshold,
        per_class_thresholds=per_class,
    )


def main() -> None:
    """CLI entrypoint: parse arguments, call :func:`load_from_hf_format`, then :func:`run_loaded_model_inference`.

    On uncaught exceptions, prints the traceback and exits with status 1. Destroys the process group in ``finally`` when
    distributed was initialized.

    Returns:
        ``None`` (process exit code conveys success or failure).
    """
    if not logging.root.handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Loads CWIP model from HuggingFace format and runs inference",
        formatter_class=CustomFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default=None,
        help="Path to HuggingFace format model directory (uses in-tree Git LFS checkpoint runfiles if not provided)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to load model on")
    parser.add_argument(
        "--test_inference",
        action="store_true",
        help="Run test inference after loading (uses dummy data if no videos provided)",
    )
    parser.add_argument("--camera_video_path", type=str, default=None, help="Path to camera video file")
    parser.add_argument("--world_video_path", type=str, default=None, help="Path to world video file")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to output directory")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size for inference to avoid OOM (reduce if running out of GPU memory)",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="Tensor parallelism size (use with torchrun --nproc_per_node=N)",
    )
    parser.add_argument(
        "--fp_size",
        type=int,
        default=1,
        help="Frame parallelism size — scatter/gather frames across GPUs (use with torchrun --nproc_per_node=N)",
    )
    parser.add_argument(
        "--true_cp_size",
        type=int,
        default=None,
        help=(
            "True context-parallel degree (Megatron CP = fp_size x this). "
            "PatchFormer shards patch tokens and uses gather+SDPA attention. "
            "If not set, reads from config.json."
        ),
    )
    parser.add_argument(
        "--sequence_parallel",
        action="store_true",
        help="Enable sequence parallelism (requires TP > 1)",
    )
    parser.add_argument(
        "--channels_last",
        action="store_true",
        help=(
            "Convert model and image inputs to channels_last memory format. "
            "Helps conv-heavy image-encoder stems on Ampere+ GPUs (~5-15%); "
            "shape-independent and safe under CWIP's dynamic-shape regime."
        ),
    )
    parser.add_argument(
        "--use_hf",
        action="store_true",
        help=(
            "Load the model via transformers.PreTrainedModel.from_pretrained instead of "
            "the legacy custom loader. Requires `transformers` to be installed. The two "
            "paths consume the same on-disk format (config.json + model.safetensors)."
        ),
    )
    parser.add_argument(
        "--camera_only_mode",
        action="store_true",
        help=(
            "Run only the camera branch (camera encoder/aligner, camera global embedding, and "
            "object-type prediction). The world encoder and similarity head are skipped at forward "
            "time. In video mode only --camera_video_path is required (--world_video_path is ignored); "
            "in --test_inference no dummy world tensor is used."
        ),
    )
    parser.add_argument(
        "--image_size",
        type=str,
        default=None,
        help=(
            "Optional resize target as 'H,W' (or a single int for a square) applied to decoded frames "
            "before the forward pass. Because forward derives its processing resolution from the input "
            "shape, this directly bounds patch-former memory. Recommended for --camera_only_mode at "
            "native camera resolution: the two-branch path resizes the camera down to the world frame "
            "size, but camera-only has no world frame and otherwise runs at full camera resolution."
        ),
    )

    parser.add_argument(
        "--save_presence_masks",
        action="store_true",
        help=(
            "Video mode: also write the static object check's presence masks (the per-patch masks correspondence and "
            "hallucination score against) as an .npz, a coverage summary .json, and an overlay video to --output_dir."
        ),
    )
    parser.add_argument(
        "--presence_source",
        type=PresenceSource,
        choices=list(PresenceSource),
        default=DEFAULT_TWO_STREAM_PRESENCE_SOURCE,
        help=(
            "With --save_presence_masks, two-stream only (as the checker's presence.source): similarity = MATCH / "
            "EXTRA similarity labels; type = the type-head mask for both; union / intersection = the similarity label "
            "OR / AND the type-head mask. Camera-only always uses the type-head mask."
        ),
    )
    parser.add_argument(
        "--type_mask_gate",
        type=TypeMaskGate,
        choices=list(TypeMaskGate),
        default=TypeMaskGate.CONFIDENCE,
        help=(
            "With --save_presence_masks, how the type-head mask is built (as the checker's presence.type_mask.gate): "
            "confidence gates on the top type probability; argmax marks every patch present."
        ),
    )
    parser.add_argument(
        "--type_confidence_threshold",
        type=float,
        default=DEFAULT_TYPE_CONFIDENCE_THRESHOLD,
        help=(
            "With --save_presence_masks, the minimum top type probability for the type-head mask (in [0, 1]; as the "
            "checker's presence.type_mask.confidence_threshold)."
        ),
    )
    parser.add_argument(
        "--per_class_thresholds",
        type=str,
        default=None,
        help=(
            "With --save_presence_masks, per-class overrides of --type_confidence_threshold as comma-separated "
            "ObjectType=threshold pairs, e.g. 'TRAFFIC_LIGHT_RED=0.7,HDMAP_TRAFFIC_SIGNS=0.6'."
        ),
    )

    args = parser.parse_args()

    if args.fp_size < 1:
        parser.error(f"--fp_size must be >= 1, got {args.fp_size}")

    image_size: tuple[int, int] | None = None
    if args.image_size:
        try:
            parts = [int(p) for p in args.image_size.replace(" ", "").split(",") if p]
        except ValueError:
            parser.error(f"--image_size must be 'H,W' or a single int, got {args.image_size!r}")
        if len(parts) == 1:
            image_size = (parts[0], parts[0])
        elif len(parts) == 2:
            image_size = (parts[0], parts[1])
        else:
            parser.error(f"--image_size must be 'H,W' or a single int, got {args.image_size!r}")
        if any(v <= 0 for v in image_size):
            parser.error(f"--image_size values must be positive, got {image_size}")

    presence = _parse_presence_options(parser, args)

    try:
        model = load_from_hf_format(
            model_path=args.model_path,
            device=args.device,
            tp_size=args.tp_size,
            fp_size=args.fp_size,
            true_cp_size=args.true_cp_size,
            sequence_parallel=args.sequence_parallel,
            channels_last=args.channels_last,
            use_hf=args.use_hf,
            camera_only_mode=args.camera_only_mode,
        )
        run_loaded_model_inference(
            model,
            test_inference=args.test_inference,
            camera_video_path=args.camera_video_path,
            world_video_path=args.world_video_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            fp_size=args.fp_size,
            camera_only=args.camera_only_mode,
            image_size=image_size,
            presence=presence,
        )
    except Exception as e:  # noqa: BLE001 - CLI top-level handler prints traceback and exits non-zero
        print(f"❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
