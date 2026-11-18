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

"""The vocabulary for "this process cannot use the GPU", shared by everything that talks to CUDA.

Several unrelated subsystems reach CUDA by different routes -- ffmpeg and ffprobe through NVDEC, ONNX Runtime through
its CUDA execution provider, PyTorch through its own runtime -- and each of them can fail for the same underlying
reason. Keeping the error type and the classification here means a caller that has to react to an unusable GPU
(stop claiming work, report itself unhealthy, ask to be replaced) can do so without knowing which library noticed.

Deliberately dependency-free: importable from a pure check processor, a service, or the checker framework alike.
"""


class GpuUnavailableError(Exception):
    """Raised when the GPU is unusable by this process, though the work it was asked to do is perfectly valid.

    The distinction this type carries is about *whose fault it is*, because that decides what the caller should do.
    An ordinary failure says the input is bad and should be rejected, and retrying it anywhere would fail the same
    way. This one says the input is fine and the machine is not, so the work belongs on a different replica -- or on
    this one once its GPU works again -- rather than being recorded as a bad result.

    Raised both where a GPU failure is detected directly (an ONNX session whose CUDA provider will not initialize)
    and, via subclasses, where it surfaces as something more specific (see
    ``checks.utils.video.GpuUnavailableDecodeError``, which is also a ``VideoDecodeError`` so existing decode
    handlers keep working).
    """


# Emitted by ffmpeg when the NVDEC stack fails to load, as distinct from loading successfully but not supporting the
# codec requested of it. The distinction separates a container configuration fault (started without
# ``NVIDIA_DRIVER_CAPABILITIES=video``), which is remediable, from the absence of an NVDEC block for that codec on
# the device, which is not.
NVDEC_MISSING_SIGNATURE = "Cannot load libnvcuvid"

# Substrings emitted when the CUDA driver API fails to initialize for the process. The classification applies to a
# probe as well as to a decode: an ffmpeg build without a software H.264 decoder opens ``h264_cuvid`` during
# ``avformat_find_stream_info``, so reading the dimensions of an H.264 MP4 invokes ``cuInit``. Left unclassified, the
# resulting non-zero exit is indistinguishable from a corrupt container.
#
# The later entries originate in libraries layered above the driver, which fail on their own terms rather than at
# ``cuInit``; ``cublasCreate`` fails while ONNX Runtime constructs its CUDA execution provider and terminates the
# session with it. Substring matching is the only available signal, as ONNX Runtime reports the failure as a generic
# error carrying the C++ call site. A missed match fails safe: the original error propagates and a single job fails,
# rather than a missing model file or an unsupported opset being misclassified as a broken GPU at the cost of a
# replica.
GPU_INIT_FAILURE_SIGNATURES: tuple[str, ...] = (
    "cuInit",
    NVDEC_MISSING_SIGNATURE,
    "CUBLAS failure",
    "cublasCreate",
    "CUDA failure",
    "CUDNN failure",
    "cudaErrorInitializationError",
)

# The subset for which a retry is warranted. Every entry denotes a transient state rather than a host capability: MPS
# creates its per-user server lazily on the first client connection, and clients that arrive during that interval, or
# during the re-creation following a server exit, are refused until it becomes available. A process that starts the
# control daemon and begins claiming work concurrently opens exactly that interval. The remaining ``cuInit`` failures
# (no device, no driver, absent libnvcuvid) are properties of the host that waiting will not change, and therefore
# fail on the first attempt.
TRANSIENT_GPU_INIT_SIGNATURES: tuple[str, ...] = (
    "CUDA_ERROR_MPS_SERVER_NOT_READY",
    "CUDA_ERROR_MPS_CONNECTION_FAILED",
    "CUDA_ERROR_MPS_RPC_FAILURE",
    "CUDA_ERROR_MPS_CLIENT_TERMINATED",
    "CUDA_ERROR_MPS_MAX_CLIENTS_REACHED",
    "CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED",
    "CUDA_ERROR_SYSTEM_NOT_READY",
)


def is_gpu_init_failure(message: str) -> bool:
    """Reports whether a failure was CUDA refusing to initialize rather than anything about the work requested.

    Args:
        message: Diagnostic text from a CUDA client -- ffmpeg or ffprobe stderr, an ONNX Runtime error, or a driver
            error string.
    """
    return any(signature in message for signature in GPU_INIT_FAILURE_SIGNATURES)


def is_transient_gpu_init_failure(message: str) -> bool:
    """Reports whether a CUDA initialization failure is one that may clear on its own.

    See :data:`TRANSIENT_GPU_INIT_SIGNATURES` for which states qualify and why the rest do not.

    Args:
        message: Diagnostic text from a CUDA client -- ffmpeg or ffprobe stderr, or a driver error string.
    """
    return any(signature in message for signature in TRANSIENT_GPU_INIT_SIGNATURES)
