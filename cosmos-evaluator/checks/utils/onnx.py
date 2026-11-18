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

"""Utilities for configuring ONNX Runtime and constructing inference sessions."""

import gc
import logging
import warnings

import onnxruntime as ort

from utils.gpu import GpuUnavailableError, is_gpu_init_failure

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def clear_gpu_memory() -> None:
    """Clear GPU memory caches to reduce fragmentation and release unused memory.

    Intended to be called between heavy GPU operations, such as between runs of different models, to consolidate GPU
    memory and avoid out-of-memory errors arising from fragmentation. Three steps are performed: Python garbage
    collection to release unreferenced objects, a flush of PyTorch's CUDA memory cache where PyTorch is available, and
    a CUDA stream synchronization to confirm that outstanding operations have completed.
    """
    gc.collect()

    if torch is None or not hasattr(torch, "cuda"):
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared PyTorch CUDA cache")
    except Exception as e:  # noqa: BLE001 - the cache clear is best-effort; log the failure and continue
        logger.warning("Failed to clear CUDA cache during clear_gpu_memory: %s", e)


def configure_onnx_logging(verbose: bool = False) -> None:
    """Configure ONNX Runtime logging so that warnings are suppressed unless verbose output is requested.

    Args:
        verbose: Whether to enable verbose logging.
    """
    if not verbose:
        ort.set_default_logger_severity(3)  # 0=verbose, 1=info, 2=warning, 3=error, 4=fatal
        warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")


def get_inference_session(
    model_path: str,
    model_device: str = "cuda",
    verbose: bool = False,
    allow_cpu_fallback: bool = False,
) -> ort.InferenceSession:
    """Construct an ONNX Runtime inference session for the model at the given path.

    Args:
        model_path: Path to the ONNX file.
        model_device: Device on which to run the model, either "cuda" or "cpu".
        verbose: Whether to enable verbose logging.
        allow_cpu_fallback: Whether a session may be built on the CPU provider after the CUDA provider fails to
            initialize. Defaults to False, so that a replica which requested a GPU and did not obtain one reports the
            failure rather than continuing. Enable it only where a slow result is preferable to no result, such as a
            local CLI run on a machine without a GPU, and never in a service, where the fallback yields a replica that
            appears healthy, accepts GPU work, and takes orders of magnitude longer per job than the queue was sized
            for.

    Returns:
        An ONNX Runtime inference session.

    Raises:
        GpuUnavailableError: If ``model_device`` is ``"cuda"``, the failure is recognizable as a CUDA initialization
            failure, and ``allow_cpu_fallback`` is False. This is distinguished from an ordinary failure because the
            remedy differs: the model and the requested work are valid and this process's GPU is not.
        Exception: Whatever ONNX Runtime raised, if CPU was the requested device or the failure was not a CUDA
            initialization failure. A missing model file is not evidence of a broken GPU, so it propagates unchanged.
    """
    # Providers are tried in list order, so the CPU entry serves as the terminal fallback and the CUDA provider is
    # prepended ahead of it when a GPU is requested.
    providers: list[str | tuple[str, dict[str, object]]] = ["CPUExecutionProvider"]

    if model_device == "cuda":
        cuda_providers: list[str | tuple[str, dict[str, object]]] = [
            (
                "CUDAExecutionProvider",
                {
                    # Allocate exactly the amount requested, avoiding the power-of-two growth that can leave the
                    # arena fixed at an oversized allocation.
                    "arena_extend_strategy": "kSameAsRequested",
                    # Ceiling on arena growth, sized for an 80 GB GPU. A session can be shared across concurrent
                    # inferences, so this bounds one pool serving all of them rather than a single job, and the
                    # arena returns memory to the driver only at the end of a run that requests arena shrinkage -
                    # until then, whatever it claims stays unavailable to other allocators on the device. Set it too
                    # low and inference fails while the device still has room; too high and it starves everything
                    # sharing the GPU.
                    "gpu_mem_limit": 32 * 1024 * 1024 * 1024,  # 32 GB
                    # HEURISTIC rather than EXHAUSTIVE, to limit memory pressure from the cuDNN workspace.
                    "cudnn_conv_algo_search": "HEURISTIC",
                    "cudnn_conv1d_pad_to_nc1d": True,  # Optimize 1D convolutions
                    "do_copy_in_default_stream": True,  # Improve memory management
                    "enable_cuda_graph": False,  # Disabled to avoid conflicts with shape adjustment
                    # Limit cuDNN workspace memory usage.
                    "cudnn_conv_use_max_workspace": False,
                },
            )
        ]
        providers = cuda_providers + providers

    sess_options = ort.SessionOptions()

    # Reduce logging verbosity unless it is explicitly requested.
    if verbose:
        sess_options.log_severity_level = 0  # Verbose logging
    else:
        sess_options.log_severity_level = 2  # Only warnings and errors

    # Performance-oriented session settings.
    sess_options.enable_cpu_mem_arena = True  # Enable CPU memory arena for better memory management
    sess_options.enable_mem_pattern = True
    sess_options.intra_op_num_threads = 0  # Use all available CPU cores (0 = auto-detect)
    sess_options.inter_op_num_threads = 0  # Use all available CPU cores for parallel ops
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.enable_profiling = False  # Disable profiling for better performance

    try:
        ort.preload_dlls()
        ort_session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)

        active_providers = ort_session.get_providers()
        gpu_mem_limit = providers[0][1].get("gpu_mem_limit", "N/A") if isinstance(providers[0], tuple) else "N/A"
        logger.info("ONNX Runtime using providers: %s, GPU mem limit: %s", active_providers, gpu_mem_limit)

        if verbose:
            logger.info("ORT device: %s", ort.get_device())

            model_inputs = ort_session.get_inputs()
            for i, model_input in enumerate(model_inputs):
                input_name = model_input.name
                input_shape = model_input.shape
                logger.info("Model input%s name: %s, shape: %s", i, input_name, input_shape)

            model_outputs = ort_session.get_outputs()
            for i, model_output in enumerate(model_outputs):
                output_name = model_output.name
                output_shape = model_output.shape
                logger.info("Model output%s name: %s, shape: %s", i, output_name, output_shape)

        return ort_session

    except Exception as e:
        if model_device != "cuda":
            # CPU execution was requested, so no fallback applies and the GPU is not implicated; the fault lies in
            # the model file or in the runtime itself. Propagate the original error untranslated.
            raise
        if not is_gpu_init_failure(str(e)):
            # A request for CUDA does not make every failure a CUDA failure. A missing model file, an unsupported
            # opset, or a corrupt proto fails here as well, and neither classification as a GPU failure nor a retry
            # on the CPU is appropriate: the former would cost a replica over a packaging fault, and the latter would
            # fail identically.
            raise

        logger.error("Failed to create ONNX inference session on CUDA: %s", e)
        if not allow_cpu_fallback:
            # With fallback disallowed, the failure is reported as a GPU failure rather than degraded silently. A
            # silent CPU fallback returns a working session, which leaves the process reporting itself healthy and
            # continuing to claim GPU work it can no longer serve at the expected rate.
            raise GpuUnavailableError(f"Could not initialize the CUDA execution provider for {model_path}: {e}") from e

        logger.warning("Falling back to CPU-only execution; inference will be far slower than on a GPU")
        providers = ["CPUExecutionProvider"]
        sess_options.log_severity_level = 2  # Warnings and errors only on the fallback path
        return ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
