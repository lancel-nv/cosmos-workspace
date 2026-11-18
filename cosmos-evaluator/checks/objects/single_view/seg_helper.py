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
Segmentation helper module for object processing.

Owns the segmentation model setup and offers utilities to:
- Process video frames to segmentation outputs
- Extract per-class binary masks, using a centralized mapping from our high-level class names to SegFormer label names
- Filter frame objects by our high-level classes, using a centralized mapping from high-level classes to dataset
  object types
"""

import logging
from typing import Any, ClassVar

import numpy as np
import onnxruntime as ort
import torch

from checks.utils.segformer import (
    create_cityscapes_label_colormap,
    get_masks,
    setup_model,
)

logger = logging.getLogger(__name__)

# Run-configuration key that makes ONNX Runtime return an arena's unused regions to the driver at the end of the run.
_ARENA_SHRINKAGE_CONFIG_KEY = "memory.enable_memory_arena_shrinkage"


class SegHelper:
    """
    Handles instance segmentation processing for objects.

    This class manages the segmentation model, processes video frames,
    and generates segmentation masks.
    """

    # Mapping from our high-level classes to SegFormer class labels
    CLASS_TO_SEGFORMER_CLASSES: ClassVar[dict[str, list[str]]] = {
        "vehicle": ["car", "truck", "bus"],
        "pedestrian": ["person"],
        "motorcycle": ["motorcycle", "rider"],
        "bicycle": ["bicycle", "rider"],
    }

    # Cached colormap data (lazy initialization)
    _cached_colormap = None
    _cached_class_label_to_idx = None

    def __init__(self, model_device: str = "cuda", session: ort.InferenceSession | None = None):
        """
        Initialize the segmentation processor.

        Args:
            model_device: Device to run the model on ("cuda" or "cpu")
            session: An inference session built elsewhere (see :func:`checks.utils.segformer.load_segformer_session`).
                When provided, this helper creates only its own ``IOBinding`` and initializes no CUDA context, which
                is what lets several concurrent jobs in one process share a single execution provider. When ``None``,
                the helper builds and owns a session, which is right for a one-clip-per-process caller such as the CLI.
        """
        self.model_device = model_device
        self.ort_session: Any = None
        self.io_binding: Any = None
        self.input_name = None
        self.output_name = None
        self.colormap: dict[int, tuple[int, int, int]] | None = None
        self.class_label_to_idx: dict[str, int] | None = None
        self._last_input_shape: tuple[int, ...] | None = None
        self._last_input_dtype: np.dtype[Any] | None = None

        # Initialize the model
        self._setup_model(session)

    def _setup_model(self, session: ort.InferenceSession | None = None) -> None:
        """Set up the segmentation model and related components."""
        if session is None:
            # Setup ONNX model with reduced logging
            self.ort_session, self.io_binding = setup_model(model_device=self.model_device, verbose=False)
        else:
            # A binding of our own is required rather than merely cheaper: ``process_frame`` rebinds and reads it, so
            # two jobs sharing one binding would overwrite each other's inputs and silently return the wrong masks.
            # Creating one allocates no device memory and initializes no CUDA context -- the session already did both.
            self.ort_session = session
            self.io_binding = session.io_binding()
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

        logger.info("SegHelper initialized with device=%s", self.model_device)

        # Setup colormap for visualization
        self.colormap, self.class_label_to_idx = create_cityscapes_label_colormap()

    @staticmethod
    def get_class_color(class_name: str) -> np.ndarray:
        """
        Get the RGB color for a SegFormer class label.

        This static method resolves colors using the canonical Cityscapes colormap and label index mapping, without
        requiring an instance of SegHelper.

        Args:
            class_name: Exact SegFormer class label (e.g., "road", "person", "car").

        Returns:
            RGB color as numpy array
        """
        # Lazy initialization of cached colormap
        if SegHelper._cached_colormap is None:
            SegHelper._cached_colormap, SegHelper._cached_class_label_to_idx = create_cityscapes_label_colormap()

        class_label_to_idx = SegHelper._cached_class_label_to_idx

        if class_name not in class_label_to_idx:
            raise ValueError(f"Unknown class name: {class_name}")
        class_idx = class_label_to_idx[class_name]
        return np.array(SegHelper._cached_colormap[class_idx], dtype=np.uint8)

    def process_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Process a single frame through the segmentation model.

        Args:
            frame: Input frame as torch tensor

        Returns:
            segmentation_masks: Full segmentation result as torch tensor
        """
        # Convert frame to numpy for ONNX inference
        image_np_array = frame.cpu().numpy()
        self._last_input_shape = image_np_array.shape
        self._last_input_dtype = image_np_array.dtype

        # Clear previous bindings to release GPU memory from prior inference
        self.io_binding.clear_binding_inputs()
        self.io_binding.clear_binding_outputs()

        # Bind inputs and outputs fresh for this frame
        self.io_binding.bind_cpu_input(self.input_name, image_np_array)
        self.io_binding.bind_output(self.output_name, device_type=self.model_device)

        # Run inference
        self.ort_session.run_with_iobinding(self.io_binding)

        # Synchronize to ensure GPU operations complete before copying
        self.io_binding.synchronize_outputs()
        output = self.io_binding.copy_outputs_to_cpu()

        # Get segmentation masks
        return get_masks(output[0], frame.shape[1:])

    def release_arena_memory(self) -> None:
        """Return the CUDA arena regions that no inference is using to the driver.

        The arena of the CUDA execution provider grows but does not shrink of its own accord: a region freed by a
        finished inference stays reserved for the life of the process and is reused only by a request that fits within
        it. When concurrent jobs share a session their requests interleave, the free space fragments across regions, and
        the arena eventually reaches ``gpu_mem_limit`` with no region large enough for the next request. ONNX Runtime
        releases unused regions only at the end of a run that requests it, so this runs one inference on a zero-valued
        input of the most recent input's shape and dtype with that request attached. Regions holding memory of another
        job's in-flight inference are retained.

        Intended to be called once per job, after its last inference. This helper's bindings are cleared first so that
        the last output does not pin its region. Nothing runs if the session has no CUDA execution provider or if no
        inference has run since the previous release. A failure is logged rather than raised: it leaves the arena as it
        was, and it must not replace the outcome of the job that called it.
        """
        self.io_binding.clear_binding_inputs()
        self.io_binding.clear_binding_outputs()

        input_shape = self._last_input_shape
        self._last_input_shape = None
        if input_shape is None:
            return

        cuda_options = self.ort_session.get_provider_options().get("CUDAExecutionProvider")
        if cuda_options is None:
            return

        run_options = ort.RunOptions()
        run_options.add_run_config_entry(_ARENA_SHRINKAGE_CONFIG_KEY, f"gpu:{cuda_options['device_id']}")
        zero_input = np.zeros(input_shape, dtype=self._last_input_dtype)
        try:
            self.ort_session.run([self.output_name], {self.input_name: zero_input}, run_options)
        except Exception:
            logger.warning("Failed to release unused SegFormer arena memory", exc_info=True)

    def get_class_mask(self, segmentation_masks: torch.Tensor, class_name: str) -> np.ndarray:
        """
        Extract binary mask for a specific high-level class from segmentation result.

        Args:
            segmentation_masks: Segmentation result from process_frame
            class_name: High-level class name (e.g., "vehicle", "pedestrian")

        Returns:
            Binary mask for the specified class or class group
        """
        # Convert segmentation masks to numpy (H, W, C) for color comparisons
        seg_mask_np = segmentation_masks.permute(1, 2, 0).cpu().numpy()

        # Initialize combined mask
        combined_mask = np.zeros(seg_mask_np.shape[:2], dtype=bool)

        # Add each SegFormer class corresponding to the high-level class
        segformer_classes = SegHelper.CLASS_TO_SEGFORMER_CLASSES.get(class_name, [])
        for seg_class in segformer_classes:
            try:
                target_color = np.array(self.get_class_color(seg_class), dtype=np.uint8)
                class_mask = np.all(seg_mask_np == target_color, axis=-1)
                combined_mask = combined_mask | class_mask
            except ValueError:
                # Skip if segmentation class doesn't exist
                continue

        return combined_mask

    def resize_masks(self, masks: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """
        Resize segmentation masks to target dimensions.

        Args:
            masks: Input masks as torch tensor
            target_height: Target height
            target_width: Target width

        Returns:
            Resized masks
        """
        resized_masks = (
            torch.nn.functional.interpolate(
                masks.unsqueeze(0).float(), size=(target_height, target_width), mode="nearest"
            )
            .squeeze(0)
            .type(masks.dtype)
        )

        return resized_masks

    # selection helpers removed; scoring and hallucination refined union now own their logic
