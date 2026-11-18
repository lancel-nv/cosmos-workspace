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

"""Tests for the per-replica job slot count."""

import asyncio
import unittest

from services.framework.job_slots import JobSlots


class TestSizing(unittest.TestCase):
    """``free`` is the number a pod puts in its next claim request, so it has to mean exactly what it says."""

    def test_a_fresh_replica_offers_its_whole_capacity(self) -> None:
        slots = JobSlots(4)

        self.assertEqual((slots.limit, slots.active, slots.free), (4, 0, 4))

    def test_a_limit_below_one_is_refused(self) -> None:
        """A replica that can run nothing would claim nothing forever, silently."""
        with self.assertRaises(ValueError):
            JobSlots(0)

    def test_taking_more_than_is_free_reserves_what_there_is(self) -> None:
        """Partial success is the expected case: a pod with two slots free should start two jobs, not wait for three."""
        slots = JobSlots(2)

        self.assertEqual(slots.take(5), 2)
        self.assertEqual(slots.free, 0)

    def test_taking_from_a_full_replica_reserves_nothing(self) -> None:
        slots = JobSlots(1)
        slots.take()

        self.assertEqual(slots.take(), 0)
        self.assertEqual(slots.active, 1)

    def test_released_slots_come_back(self) -> None:
        slots = JobSlots(2)
        slots.take(2)

        slots.release()

        self.assertEqual(slots.free, 1)

    def test_a_double_release_cannot_manufacture_capacity(self) -> None:
        """Cleanup paths release defensively, and a replica quietly running over its limit is hard to notice."""
        slots = JobSlots(2)
        slots.take()

        slots.release()
        slots.release()

        self.assertEqual((slots.active, slots.free), (0, 2))

    def test_work_already_claimed_is_accounted_for_even_over_the_limit(self) -> None:
        """Declining a claimed job would not free the pod of it; it would just leave the job idle until its lease ran
        out."""
        slots = JobSlots(1)
        slots.take()

        slots.commit()

        self.assertEqual(slots.active, 2)
        self.assertEqual(slots.free, 0, "an over-committed replica must not advertise room it does not have")


class TestWaiting(unittest.IsolatedAsyncioTestCase):
    """The two ways a caller waits: for a slot of its own, or for a batch size worth asking about."""

    async def test_acquire_waits_for_a_slot_rather_than_refusing(self) -> None:
        """The caller reached this pod through a gateway, so "try another replica" is not advice it can take."""
        slots = JobSlots(1)
        await slots.acquire()

        waiter = asyncio.create_task(slots.acquire())
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())

        slots.release()
        await asyncio.wait_for(waiter, 1.0)

        self.assertEqual(slots.active, 1)

    async def test_only_one_waiter_gets_the_slot_that_came_free(self) -> None:
        slots = JobSlots(1)
        await slots.acquire()

        first = asyncio.create_task(slots.acquire())
        second = asyncio.create_task(slots.acquire())
        await asyncio.sleep(0)
        slots.release()
        await asyncio.sleep(0)

        self.assertEqual([first.done(), second.done()], [True, False])
        self.assertEqual(slots.active, 1)
        second.cancel()

    async def test_wait_for_room_returns_without_reserving_anything(self) -> None:
        slots = JobSlots(2)
        slots.take(2)

        waiter = asyncio.create_task(slots.wait_for_room())
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())

        slots.release()
        await asyncio.wait_for(waiter, 1.0)

        self.assertEqual(slots.active, 1, "wait_for_room must leave the batch size to the caller")


if __name__ == "__main__":
    unittest.main()
