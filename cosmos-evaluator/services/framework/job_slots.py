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

"""How much work one checker replica will run at once.

A checker pod sizes this against its own memory and GPU footprint, and it is the only place that ceiling is enforced:
a pod asks the queue for as many jobs as it has room for and asks for nothing when it is full. Occupancy therefore
needs no fleet-wide accounting and nothing outside the pod has to guess at it.

The distinction from a rejecting limiter is the whole point. A caller told "full" has learned only that one
gateway-selected pod was busy, which is not a fact about the fleet and not something it can act on. Here there is no
caller to tell: work waits in the queue until some pod has room, and ``free`` is a number the pod uses to size its own
next request rather than an error it reports to somebody else.
"""

import asyncio


def _check_count(count: int, operation: str) -> None:
    """Reject a negative slot count before it is applied.

    Every method here is signed arithmetic on one counter, so a negative count silently inverts the operation: a
    negative ``release`` reserves slots and a negative ``commit`` pushes ``free`` above ``limit``, letting the replica
    take on more work than it was configured for. Neither is recoverable after the fact and neither looks like a bug
    from outside, so it fails at the call site the way ``limit`` does in the constructor.
    """
    if count < 0:
        raise ValueError(f"{operation} count must not be negative, got {count}")


class JobSlots:
    """A count of simultaneous jobs this replica will run, and a way to wait for one to come free.

    Not a semaphore, because the interesting operation is not "wait for a permit" but "how many can I start right now"
    -- a pod claiming work in batches has to size the batch before it asks, and hand back the slots for jobs the queue
    turned out not to have.

    Slots are reserved before the work exists and released when it ends, so a reservation the caller does not use has
    to be returned explicitly. Every method is safe from any coroutine on the loop: reservations and releases are
    synchronous and indivisible, so two coroutines cannot both see the same last free slot.
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        self._limit = limit
        self._active = 0
        # Set exactly when at least one slot is free, so a waiter can block without polling.
        self._room = asyncio.Event()
        self._room.set()

    @property
    def limit(self) -> int:
        """Simultaneous jobs this replica is configured to run."""
        return self._limit

    @property
    def active(self) -> int:
        """Slots currently reserved."""
        return self._active

    @property
    def free(self) -> int:
        """Slots available right now. The natural batch size for a claim request."""
        return max(0, self._limit - self._active)

    def commit(self, count: int = 1) -> None:
        """Reserve slots for work that already exists, whether or not the replica is under its limit.

        For jobs a pod has claimed: the lease is already running, so declining one would not free the pod of it -- it
        would just mean nobody works on that job until the lease expires. Overshooting the limit slightly is the better
        of those two outcomes, and it can only happen when a direct ``/process`` call took a slot between this pod
        sizing its claim and the claim coming back.
        """
        _check_count(count, "commit")
        self._active += count
        if self.free == 0:
            self._room.clear()

    def take(self, count: int = 1) -> int:
        """Reserve up to ``count`` slots and report how many were actually reserved.

        Partial success is the expected case rather than a degraded one: a pod with two free slots asking for three
        should start two jobs, not wait for a third slot it does not need.
        """
        _check_count(count, "take")
        taken = max(0, min(count, self.free))
        self._active += taken
        if self.free == 0:
            self._room.clear()
        return taken

    def release(self, count: int = 1) -> None:
        """Return ``count`` reserved slots.

        Clamped at zero rather than raising. This is called from cleanup paths where the alternative to a double
        release is a slot leaked for the lifetime of the process, and a replica quietly running below its limit is far
        harder to notice than a count that is briefly generous.

        A *negative* count is a different matter and still raises: clamping it would reserve slots through the method
        that exists to give them back, which no cleanup path can have meant.
        """
        _check_count(count, "release")
        self._active = max(0, self._active - count)
        if self.free > 0:
            self._room.set()

    async def acquire(self) -> None:
        """Wait for one slot and reserve it.

        For a caller holding a connection that has nowhere else to go. Waiting is right there and rejecting is not:
        the caller cannot usefully be told to try a different pod, since it cannot address one.
        """
        while self.take() == 0:
            await self._room.wait()

    async def wait_for_room(self) -> None:
        """Wait until at least one slot is free, without reserving it.

        For a caller that wants to size a batch against ``free`` once it wakes, rather than commit to a single slot.
        """
        while self.free == 0:
            await self._room.wait()
