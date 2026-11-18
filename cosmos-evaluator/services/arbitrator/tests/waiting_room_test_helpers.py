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

"""Fixtures shared by the waiting-room test modules.

Both the scheduling tests and the durable-dispatch tests need a waiting room, an admitted request with stubbed database
records, and a way to shut a loop down at the end, so those live here rather than being written twice.
"""

import asyncio
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from services.arbitrator.checker_queue import CheckerJob, CheckerQueue
from services.arbitrator.waiting_room_queue import ActiveRequest, WaitingRoom


def make_waiting_room(
    *,
    build_request: Callable | None = None,
    capacity: int | None = None,
    **kwargs: Any,
) -> WaitingRoom:
    """A waiting room with stub callbacks, for tests that care about scheduling rather than payloads.

    ``store_result`` files each job's value under its own name so a test can read back exactly what one job produced.
    """
    return WaitingRoom(
        build_request=build_request or (lambda job, request, queue: {"name": job.name}),
        store_result=lambda agg, job, data: agg.__setitem__(job.name, data),
        capacity=capacity,
        **kwargs,
    )


def make_active_request(
    job_names: list[str],
    request_id: int = 1,
    finalize: Callable | None = None,
    execution_ids: dict[str, int] | None = None,
) -> ActiveRequest:
    """Build an ActiveRequest with a flat queue of independent jobs (no deps).

    ``execution_ids`` assigns each job's stub record an ``execution_id``, which a queued job requires: without one there
    is no row for a checker to claim.
    """
    jobs = [CheckerJob(name=n, base_name=n, priority=0, deps=set()) for n in job_names]
    queue = CheckerQueue(jobs)

    execution = MagicMock()
    execution.request_id = request_id
    execution.in_progress = AsyncMock()
    execution.complete = AsyncMock(return_value=0)
    execution.fail = AsyncMock(return_value=0)

    checker_executions: dict[str, Any] = {}
    for index, name in enumerate(job_names):
        record = MagicMock()
        record.execution_id = (execution_ids or {}).get(name, 1000 + index)
        # Return values are explicit rather than left as truthy mocks: the waiting room branches on all of these,
        # so a bare AsyncMock would silently pin every branch to the success side.
        record.enqueue = AsyncMock(return_value=True)
        record.in_progress = AsyncMock(return_value=True)
        record.complete = AsyncMock(return_value=True)
        record.fail = AsyncMock(return_value=True)
        record.skipped = AsyncMock(return_value=True)
        record.timeout = AsyncMock(return_value=True)
        checker_executions[name] = record

    async def _default_finalize(_active: ActiveRequest) -> dict:
        return {"results": dict(_active.aggregated_results)}

    return ActiveRequest(
        execution=execution,
        request=MagicMock(),
        queue=queue,
        checker_executions=cast(Any, checker_executions),
        health_info={},
        aggregated_results={},
        completion_future=asyncio.get_event_loop().create_future(),
        finalize=finalize or _default_finalize,
    )


async def cancel_scheduler(scheduler: asyncio.Task) -> None:
    """Cancel a waiting-room loop task and await it, suppressing CancelledError."""
    scheduler.cancel()
    try:
        await scheduler
    except asyncio.CancelledError:
        pass


def reserve_and_admit(room: WaitingRoom, req: ActiveRequest) -> None:
    """Reserve a capacity slot and admit ``req``, mirroring the REST layer's contract."""
    assert room.try_admit(), "expected an unbounded test room to accept the reservation"
    room.admit(req)
