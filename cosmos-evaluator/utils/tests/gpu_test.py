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

"""Tests for the shared "this process cannot use the GPU" vocabulary.

Every case below uses text as the libraries actually emit it. The classifiers match on substrings, so a test written
against paraphrased messages would pass while the thing it stands for went on failing in production.
"""

import unittest

from utils.gpu import GpuUnavailableError, is_gpu_init_failure, is_transient_gpu_init_failure


class TestIsGpuInitFailure(unittest.TestCase):
    """Telling "the GPU would not start" apart from "the work was bad"."""

    def test_an_ffprobe_driver_failure_is_recognized(self) -> None:
        """The failure that made a good video read as corrupt.

        The vendored ffmpeg carries no software H.264 decoder, so it opens ``h264_cuvid`` while merely reading stream
        parameters -- which means probing the width of an ordinary MP4 calls ``cuInit`` and can fail on the GPU.
        """
        self.assertIs(
            is_gpu_init_failure(
                "[CUDA @ 0x57bde4876040] cu->cuInit(0) failed -> CUDA_ERROR_MPS_SERVER_NOT_READY: "
                "MPS server is not ready to accept new MPS client requests"
            ),
            True,
        )

    def test_a_missing_nvdec_library_is_recognized(self) -> None:
        """Distinct from a codec this GPU has no decode block for: this one is a fixable container misconfiguration."""
        self.assertIs(is_gpu_init_failure("Cannot load libnvcuvid.so.1"), True)

    def test_the_onnx_cublas_failure_is_recognized(self) -> None:
        """The production failure this classification exists for, verbatim from the pod that died of it."""
        self.assertIs(
            is_gpu_init_failure(
                "[ONNXRuntimeError] : 1 : FAIL : Exception during initialization: "
                "/onnxruntime_src/onnxruntime/core/providers/cuda/cuda_call.cc:122 CUBLAS failure 14: "
                "an internal operation failed ; GPU=0 ; expr=cublasCreate(&cublas_handle_);"
            ),
            True,
        )

    def test_a_corrupt_container_is_not_a_gpu_failure(self) -> None:
        self.assertIs(is_gpu_init_failure("moov atom not found"), False)

    def test_a_bad_model_file_is_not_a_gpu_failure(self) -> None:
        """Misclassifying this would let a packaging mistake retire a replica, which is worse than the failed job."""
        self.assertIs(is_gpu_init_failure("Load model failed: invalid protobuf"), False)

    def test_an_unsupported_codec_is_not_a_gpu_failure(self) -> None:
        self.assertIs(is_gpu_init_failure("Decoder (codec hevc) not found for input stream"), False)


class TestIsTransientGpuInitFailure(unittest.TestCase):
    """Telling a state worth waiting out from a fact about the host that waiting will not change."""

    def test_an_mps_server_still_starting_is_worth_retrying(self) -> None:
        """MPS creates its per-user server lazily, so a pod that starts the daemon and claims work in the same breath
        opens a window where its own clients are refused."""
        self.assertIs(
            is_transient_gpu_init_failure(
                "cu->cuInit(0) failed -> CUDA_ERROR_MPS_SERVER_NOT_READY: "
                "MPS server is not ready to accept new MPS client requests"
            ),
            True,
        )

    def test_a_server_that_went_away_is_worth_retrying(self) -> None:
        self.assertIs(is_transient_gpu_init_failure("CUDA_ERROR_MPS_CLIENT_TERMINATED"), True)

    def test_a_host_with_no_gpu_should_fail_on_the_first_attempt(self) -> None:
        """Retrying only holds a job slot open for the several seconds it takes to conclude what is already known."""
        self.assertIs(is_transient_gpu_init_failure("cu->cuInit(0) failed -> CUDA_ERROR_NO_DEVICE"), False)

    def test_a_missing_driver_should_fail_on_the_first_attempt(self) -> None:
        self.assertIs(is_transient_gpu_init_failure("cu->cuInit(0) failed -> CUDA_ERROR_INVALID_DEVICE"), False)

    def test_a_cublas_failure_is_not_waited_out(self) -> None:
        """It is a GPU failure but not a transient one: the context is already poisoned and will stay that way."""
        self.assertIs(is_transient_gpu_init_failure("CUBLAS failure 14: an internal operation failed"), False)


class TestGpuUnavailableError(unittest.TestCase):
    """The type exists so callers can act on "the machine is at fault" without knowing which library noticed."""

    def test_it_is_an_ordinary_exception(self) -> None:
        with self.assertRaises(GpuUnavailableError):
            raise GpuUnavailableError("cublasCreate failed")

    def test_the_message_survives(self) -> None:
        """The reason reaches ``/health`` and the shutdown log, and is usually the only evidence left of which of the
        several ways a CUDA context can die actually happened."""
        self.assertEqual(str(GpuUnavailableError("cublasCreate failed")), "cublasCreate failed")


if __name__ == "__main__":
    unittest.main()
