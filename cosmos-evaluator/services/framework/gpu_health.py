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

"""Record of whether a replica's GPU remains usable, and of the decision to stop advertising that it is.

The failure mode addressed here is replica-scoped rather than job-scoped. A poisoned CUDA context (a cuBLAS handle
that fails to allocate, an MPS server that has terminated, or a device that has fallen off the bus) remains poisoned
for the lifetime of the process. Every subsequent job on that replica fails identically, and because the process
continues to answer health checks and to claim work, it continues to remove jobs from the queue solely in order to
fail them. A faulty replica is therefore more damaging to a fleet than a missing one, since it acts as a sink.

A latch is used rather than a gauge because no in-process recovery exists for a dead context. Recovery requires a
new process, which the orchestrator supplies once this one ceases to advertise itself as usable.
"""

import logging

logger = logging.getLogger(__name__)


class GpuHealth:
    """One-way latch recording that this replica's GPU has become unusable, together with the reason.

    The scope is deliberately limited to state: the instance holds a verdict and a reason and takes no action on
    either. Draining, refusal of new work, and requesting replacement remain the caller's responsibility, since the
    framework's claim loop and lifespan already implement each; reimplementing any of that here would leave a replica
    with two independent and potentially contradictory views of whether it is shutting down.

    Reads and trips are safe from any coroutine on the event loop: a trip is a single attribute assignment containing
    no await point, so concurrent coroutines cannot interleave within it and discard one another's verdict. The type
    is not thread-safe and does not need to be, because GPU failures are raised on whichever coroutine awaited the
    work rather than on the worker thread that executed it.
    """

    def __init__(self) -> None:
        # The reason also serves as the verdict: None denotes a usable GPU. A single field is used in place of a
        # boolean paired with a message so that the two cannot disagree about whether a failure has occurred.
        self._failure: str | None = None

    @property
    def is_healthy(self) -> bool:
        """Whether this replica currently considers its GPU usable for work."""
        return self._failure is None

    @property
    def failure(self) -> str | None:
        """Why the GPU was declared unusable, or None while it remains healthy.

        Intended to be surfaced verbatim in ``/health`` and in the log line preceding shutdown. The affected replica
        is typically no longer running by the time the failure is examined, which leaves this reason as the only
        record of which mode of CUDA context failure occurred.
        """
        return self._failure

    def trip(self, reason: str) -> bool:
        """Latch the GPU as unusable and report whether this call performed the transition.

        The return value permits every failing job to invoke this method rather than only the first. A replica running
        several jobs concurrently observes all of them fail once the context is lost, whereas the response to that
        failure (logging, draining, termination) must occur exactly once. Callers should therefore condition that
        response on the return value rather than on a preceding ``is_healthy`` check, which would race across an
        await point.

        Args:
            reason: Human-readable description of the failure, retained for ``/health`` and the shutdown log.

        Returns:
            True if the latch transitioned on this call, False if it had already been tripped.
        """
        if self._failure is not None:
            logger.warning("GPU already unusable (%s); additional failure: %s", self._failure, reason)
            return False

        self._failure = reason
        logger.error("GPU declared unusable; this replica will stop claiming work and exit. Reason: %s", reason)
        return True
