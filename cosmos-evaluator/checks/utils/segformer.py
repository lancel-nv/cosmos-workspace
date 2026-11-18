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

"""Utility functions specific to the SegFormer model."""

import os
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import draw_segmentation_masks

from checks.utils.cv2_video_dataset import VideoDataset
from checks.utils.onnx import configure_onnx_logging, get_inference_session
from utils.bazel import get_runfiles_path


def get_model_input_shape() -> tuple[int, int, int, int]:
    """Return the NCHW input shape required by the model."""
    return (1, 3, 1024, 1820)


def _resolve_segformer_model_path() -> str:
    """Locate the SegFormer ONNX file, preferring runfiles and falling back to the source tree.

    Returns:
        Filesystem path to the ONNX model.

    Raises:
        FileNotFoundError: If the model is in neither location.
    """
    model_path = get_runfiles_path("checks/utils/models/segformer/cityformer_final_1024_dynamic_input_v11_sim.onnx")
    if model_path is not None and os.path.exists(model_path):
        return model_path

    current_dir = Path(__file__).parent / "models" / "segformer"
    fallback_path = current_dir / "cityformer_final_1024_dynamic_input_v11_sim.onnx"
    if not fallback_path.exists():
        raise FileNotFoundError(f"ONNX model file not found at: {fallback_path}")
    return str(fallback_path)


def load_segformer_session(model_device: str = "cuda", verbose: bool = False) -> ort.InferenceSession:
    """Build a SegFormer inference session and warm it up.

    Separated from binding creation because this is the expensive and failure-prone half: it initializes the CUDA
    execution provider, and with it a cuBLAS handle and a device memory arena. That initialization takes seconds and
    is the step that fails when the GPU stack is unusable, so a service handling concurrent work constructs one
    session at startup and shares it across jobs rather than incurring the cost, and the risk, per request.

    The session is safe to share, as ONNX Runtime permits concurrent ``run_with_iobinding`` calls on a single
    session. The ``IOBinding`` is not, since it holds the input and output state of one call, so each caller owns its
    own (see :class:`~checks.objects.single_view.seg_helper.SegHelper`).

    Args:
        model_device: Device on which to run the model, either "cuda" or "cpu".
        verbose: Whether to enable verbose logging.

    Returns:
        A warmed-up inference session.
    """
    configure_onnx_logging(verbose)

    # ``model_device`` must be passed explicitly, as the session otherwise defaults to CPU. CPU fallback is likewise
    # disallowed: a caller that requested CUDA and received a CPU session would execute the check at a small fraction
    # of the expected rate while appearing to succeed, leaving a replica serving GPU work it cannot perform.
    sess = get_inference_session(
        _resolve_segformer_model_path(), model_device=model_device, verbose=verbose, allow_cpu_fallback=False
    )

    # Warm up on a temporary binding so that lazy kernel and workspace initialization is not charged to the first
    # frame of real work.
    io_binding = sess.io_binding()
    io_binding.bind_cpu_input(sess.get_inputs()[0].name, np.zeros(get_model_input_shape(), dtype=np.float32))
    io_binding.bind_output(sess.get_outputs()[0].name, device_type=model_device)
    sess.run_with_iobinding(io_binding)
    io_binding.synchronize_outputs()
    _ = io_binding.copy_outputs_to_cpu()

    # Release the GPU memory references held by the warm-up binding.
    io_binding.clear_binding_inputs()
    io_binding.clear_binding_outputs()

    return sess


def setup_model(model_device: str = "cuda", verbose: bool = False) -> tuple[ort.InferenceSession, ort.IOBinding]:
    """Construct a runtime session together with an I/O binding for inference.

    Intended for callers that own their session outright, such as the CLI and tests, which process a single clip in a
    process that then exits. Service code handling concurrent requests should instead call
    :func:`load_segformer_session` once at startup and create a per-job binding from the result.

    Args:
        model_device: Device on which to run the model, either "cuda" or "cpu".
        verbose: Whether to enable verbose logging.

    Returns:
        The inference session and a new I/O binding on that session.
    """
    sess = load_segformer_session(model_device=model_device, verbose=verbose)
    return sess, sess.io_binding()


def create_cityscapes_label_colormap() -> tuple[dict[int, tuple[int, int, int]], dict[str, int]]:
    """Create the label colormap used by the Cityscapes segmentation benchmark.

    See https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/citysemsegformer.

    Returns:
        The class-index-to-RGB colormap used to visualize segmentation results, and the inverse mapping from class
        label to class index.
    """
    colormap = {}
    colormap[0] = (128, 64, 128)
    colormap[1] = (244, 35, 232)
    colormap[2] = (70, 70, 70)
    colormap[3] = (102, 102, 156)
    colormap[4] = (190, 153, 153)
    colormap[5] = (153, 153, 153)
    colormap[6] = (250, 170, 30)
    colormap[7] = (220, 220, 0)
    colormap[8] = (107, 142, 35)
    colormap[9] = (152, 251, 152)
    colormap[10] = (70, 130, 180)
    # person = red
    colormap[11] = (255, 0, 0)
    # rider = yellow
    colormap[12] = (255, 255, 0)
    colormap[13] = (0, 0, 142)
    colormap[14] = (0, 0, 70)
    colormap[15] = (0, 60, 100)
    colormap[16] = (0, 80, 100)
    # motorcycle = orange
    colormap[17] = (255, 165, 0)
    # bike = green
    colormap[18] = (0, 255, 0)
    class_labels = [
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]
    assert len(class_labels) == len(colormap), "Incorrect mapping for colors!"
    class_label_to_idx = dict(zip(class_labels, range(len(colormap))))
    return colormap, class_label_to_idx


def get_normalization_constants() -> tuple[list[float], list[float]]:
    """Return the per-channel mean and standard deviation used to normalize image input."""
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]
    return (mean, std)


def get_transforms_fn() -> T.Compose:
    """Build the preprocessing pipeline that prepares an image for inference."""
    return T.Compose(
        [
            T.Resize(get_model_input_shape()[-2:]),
            T.Normalize(*get_normalization_constants()),
        ]
    )


def get_dataloader(video_path: str, start_frame: int = 0) -> tuple[DataLoader[Any], VideoDataset]:
    """Build a single-frame dataloader over a video, returned alongside the dataset it reads from."""
    transforms_fn = get_transforms_fn()

    video_dataset = VideoDataset(video_path, transforms_fn=[transforms_fn], start_frame=start_frame)

    dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False, num_workers=0)
    return dataloader, video_dataset


def get_masks(mask_prediction: np.ndarray, image_shape: tuple[int, ...], save_path: str | None = None) -> torch.Tensor:
    """Render the per-class segmentation masks into a single colorized tensor."""
    color_map, _ = create_cityscapes_label_colormap()
    num_classes = len(color_map)
    colors: list[str | tuple[int, int, int]] = [color_map[c] for c in range(num_classes)]

    mask_np = np.squeeze(mask_prediction)
    mask_tensor = torch.from_numpy(mask_np)
    masks_list = []
    for i in range(num_classes):
        masks_list.append(mask_tensor.eq(i))
    all_masks_tensor = torch.stack(masks_list)

    # At an alpha of 1.0 the masks fully replace whatever they are drawn over, so an empty canvas serves in place of
    # the original image.
    canvas = torch.zeros(image_shape, dtype=torch.uint8)
    masks_combined = draw_segmentation_masks(canvas, all_masks_tensor, colors=colors, alpha=1.0)
    if save_path is not None:
        # Persist the rendered mask for use by postprocessing.
        img = T.ToPILImage()(masks_combined)
        img.save(save_path)
    return masks_combined


def reached_end_of_segment(current_frame: int, end_frame: int) -> bool:
    """Report whether the current frame lies beyond the end of the segment.

    An ``end_frame`` of -1 denotes an unbounded segment, whose end is never reached.
    """
    assert current_frame >= 0
    assert end_frame >= -1, f"Invalid end of segment ({end_frame}) specified."
    return False if end_frame == -1 else current_frame > end_frame
