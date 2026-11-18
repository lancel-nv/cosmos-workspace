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

"""Tests for the GPU health latch."""

import unittest

from services.framework.gpu_health import GpuHealth


class TestGpuHealth(unittest.TestCase):
    """The latch's whole contract: it starts optimistic, trips once, and never goes back."""

    def test_a_replica_starts_out_believing_its_gpu_works(self) -> None:
        health = GpuHealth()

        self.assertIs(health.is_healthy, True)
        self.assertIsNone(health.failure)

    def test_tripping_records_both_the_verdict_and_the_reason(self) -> None:
        health = GpuHealth()

        self.assertIs(health.trip("CUBLAS failure 14"), True)
        self.assertIs(health.is_healthy, False)
        self.assertEqual(health.failure, "CUBLAS failure 14")

    def test_only_the_first_trip_claims_the_verdict(self) -> None:
        """The return value is what lets every failing job call this while retirement still happens once.

        A replica running three jobs concurrently will see all three fail once the context dies, and draining and
        terminating three times over is not a thing that can be made to work.
        """
        health = GpuHealth()

        self.assertIs(health.trip("first"), True)
        self.assertIs(health.trip("second"), False)
        self.assertIs(health.trip("third"), False)

    def test_the_first_reason_is_the_one_kept(self) -> None:
        """Later failures are consequences of the first, so the first is the one with diagnostic value."""
        health = GpuHealth()

        health.trip("cublasCreate failed")
        health.trip("CUDA error: unspecified launch failure")

        self.assertEqual(health.failure, "cublasCreate failed")

    def test_there_is_no_way_back(self) -> None:
        """Asserted because it is a design decision rather than an oversight: a dead context has no in-process repair.

        If this ever grows a reset, the claim loop and the retirement path both become racy -- they are written on
        the assumption that a replica which has stopped claiming stays stopped.
        """
        health = GpuHealth()
        health.trip("CUBLAS failure 14")

        self.assertFalse(any(name.startswith(("reset", "clear", "recover")) for name in dir(health)))


if __name__ == "__main__":
    unittest.main()
