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

"""Behavioral tests for the waiting room's queued jobs and its row monitor.

A queued job is split across two coroutines that never speak to each other: the scheduler writes the job to a row and
stops there, and the monitor finds out later what happened by reading that row. These tests drive both halves directly
-- the scheduler by running it, the monitor by calling ``poll_durable_jobs`` rather than waiting out its sleep -- and
script the states a real ``checker_executions`` row can present: waiting, claimed, finished, failed, handed back by the
reaper, and wedged.
"""

import asyncio
from datetime import datetime
import sys
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from database.api_models import CheckerExecutionStateRow, RequeuedCheckerJob
from services.arbitrator.checker_queue import JobState
from services.arbitrator.tests.waiting_room_test_helpers import (
    cancel_scheduler,
    make_active_request,
    reserve_and_admit,
)
from services.arbitrator.waiting_room_queue import ActiveRequest, WaitingRoom, _DurableJob

_EXECUTION_ID = 4242


def _row(
    status: str,
    *,
    execution_id: int = _EXECUTION_ID,
    response_json: dict[str, Any] | None = None,
    error_message: str | None = None,
    claimed_by: str | None = "pod-a/abc",
    attempt: int = 1,
) -> CheckerExecutionStateRow:
    """Build one ``checker_executions`` state row the way the database API returns it."""
    return CheckerExecutionStateRow(
        execution_id=execution_id,
        status=status,
        response_json=response_json,
        error_message=error_message,
        claimed_by=claimed_by,
        attempt=attempt,
    )


def _record(req: ActiveRequest, name: str = "check_a") -> MagicMock:
    """The stub ``CheckerExecutionRecord`` behind ``name``, typed so its awaits can be asserted on."""
    return cast(MagicMock, req.checker_executions[name])


class _Harness:
    """A waiting room wired for a queued checker, with the database scripted."""

    def __init__(self) -> None:
        self.rows: list[CheckerExecutionStateRow] = []
        self.cancelled: list[int] = []
        self.requeued: list[RequeuedCheckerJob] = []
        # Execution ids the database will refuse to record a claim for, mapped to the status that won. Standing in for
        # a row that reached a terminal state between the pass reading it and the pass writing it back.
        self.refused: dict[int, str | None] = {}
        # One entry per batched claim write, so a test can assert on how many round trips a pass cost rather than
        # only on its outcome.
        self.claim_writes: list[list[int]] = []

        async def _read_job_states(execution_ids: list[int]) -> list[CheckerExecutionStateRow]:
            return [row for row in self.rows if row.execution_id in execution_ids]

        async def _request_cancel(execution_id: int) -> bool:
            self.cancelled.append(execution_id)
            return True

        async def _requeue_expired() -> list[RequeuedCheckerJob]:
            return self.requeued

        async def _record_claims(execution_ids: list[int], start_time: datetime) -> dict[int, str | None]:
            self.claim_writes.append(list(execution_ids))
            return {eid: self.refused[eid] for eid in execution_ids if eid in self.refused}

        self.room = WaitingRoom(
            build_request=lambda job, request, queue: {"clip": job.name},
            store_result=lambda agg, job, data: agg.__setitem__(job.base_name, data),
            read_job_states=_read_job_states,
            request_job_cancel=_request_cancel,
            requeue_expired_jobs=_requeue_expired,
            record_claims=_record_claims,
        )

    def new_request(self) -> ActiveRequest:
        """One admitted-shaped request for a single queued checker."""
        return make_active_request(["check_a"], execution_ids={"check_a": _EXECUTION_ID})

    async def enqueue(self, req: ActiveRequest) -> None:
        """Run the scheduler just long enough for it to write ``req``'s job to its row (or decide not to)."""
        scheduler = asyncio.create_task(self.room.scheduler_loop())
        try:
            reserve_and_admit(self.room, req)
            for _ in range(200):
                if self.room._durable_jobs or req.completion_future.done():
                    break
                await asyncio.sleep(0)
        finally:
            await cancel_scheduler(scheduler)

    async def enqueue_new(self) -> ActiveRequest:
        """The common setup: admit a request and get its job onto the queue."""
        req = self.new_request()
        await self.enqueue(req)
        return req

    async def claim(self, req: ActiveRequest) -> None:
        """Let the monitor see a pod holding the job, the way a real claim becomes visible."""
        self.rows = [_row("in_progress")]
        await self.room.poll_durable_jobs()


# ---------------------------------------------------------------------------
# Putting the job on the queue
# ---------------------------------------------------------------------------


class TestEnqueue:
    """What the scheduler does with a queued checker, and what it deliberately does not do."""

    @pytest.mark.asyncio
    async def test_the_payload_is_written_to_the_row_and_nothing_is_sent(self) -> None:
        harness = _Harness()
        req = await harness.enqueue_new()

        _record(req).enqueue.assert_awaited_once_with({"clip": "check_a"})
        assert _EXECUTION_ID in harness.room._durable_jobs

    @pytest.mark.asyncio
    async def test_a_queued_job_is_outstanding_rather_than_running(self) -> None:
        """Nobody has taken it yet, so it is neither finished nor something the request can be finalized without."""
        harness = _Harness()
        req = await harness.enqueue_new()

        assert req.queue.get_job("check_a").state == JobState.QUEUED
        assert not req.completion_future.done(), "the result has not arrived yet; it comes from the row"

    @pytest.mark.asyncio
    async def test_a_second_sweep_does_not_write_the_same_job_again(self) -> None:
        """Without the synchronous state flip the scheduler would re-queue every job on every wake."""
        harness = _Harness()
        req = await harness.enqueue_new()

        harness.room._tick()
        await asyncio.sleep(0)

        assert _record(req).enqueue.await_count == 1

    @pytest.mark.asyncio
    async def test_every_eligible_job_goes_at_once_rather_than_one_at_a_time(self) -> None:
        """There is no capacity to respect here: the fleet takes work at its own rate."""
        harness = _Harness()
        first = make_active_request(["check_a"], request_id=1, execution_ids={"check_a": 1})
        second = make_active_request(["check_a"], request_id=2, execution_ids={"check_a": 2})

        scheduler = asyncio.create_task(harness.room.scheduler_loop())
        try:
            reserve_and_admit(harness.room, first)
            reserve_and_admit(harness.room, second)
            for _ in range(200):
                if len(harness.room._durable_jobs) == 2:
                    break
                await asyncio.sleep(0)
        finally:
            await cancel_scheduler(scheduler)

        assert sorted(harness.room._durable_jobs) == [1, 2]

    @pytest.mark.asyncio
    async def test_a_job_with_no_row_fails_instead_of_being_queued(self) -> None:
        harness = _Harness()
        req = harness.new_request()
        req.checker_executions["check_a"].execution_id = None

        await harness.enqueue(req)

        assert harness.room._durable_jobs == {}, "there would be nothing for a pod to claim"
        assert "No checker_executions row" in req.aggregated_results["check_a"]["error"]

    @pytest.mark.asyncio
    async def test_a_payload_that_cannot_be_recorded_fails_the_job(self) -> None:
        """A queued row without its payload is a job no pod can run, so leaving it there strands the request."""
        harness = _Harness()
        req = harness.new_request()
        _record(req).enqueue = AsyncMock(return_value=False)

        await harness.enqueue(req)

        assert harness.room._durable_jobs == {}
        assert "Could not record the payload" in req.aggregated_results["check_a"]["error"]


# ---------------------------------------------------------------------------
# Learning the outcome
# ---------------------------------------------------------------------------


class TestMonitor:
    """What the monitor makes of each state a row can be in."""

    @pytest.mark.asyncio
    async def test_a_claim_is_how_in_progress_becomes_visible(self) -> None:
        """Nothing reports in: the pod that took the job never contacts the arbitrator at all."""
        harness = _Harness()
        req = await harness.enqueue_new()

        await harness.claim(req)

        assert req.queue.get_job("check_a").state == JobState.IN_PROGRESS
        assert harness.claim_writes == [[_EXECUTION_ID]]
        cast(MagicMock, req.execution).in_progress.assert_awaited()

    @pytest.mark.asyncio
    async def test_seeing_the_same_claim_again_changes_nothing(self) -> None:
        harness = _Harness()
        req = await harness.enqueue_new()

        await harness.claim(req)
        await harness.room.poll_durable_jobs()

        assert harness.claim_writes == [[_EXECUTION_ID]], (
            "re-recording a claim every pass would keep pushing a running job's start_time forward"
        )
        assert req.queue.get_job("check_a").state == JobState.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_a_pass_costs_one_claim_write_however_many_jobs_it_is_watching(self) -> None:
        """The property this batching exists for.

        A pass that wrote per job scaled its own duration with fan-out width, and every decision it carried was made
        against a snapshot taken before the first write -- so the wider the batch, the staler the writes at the end
        of it. One replica holding a thousand-odd jobs is what turned that into passes measured in minutes.
        """
        harness = _Harness()
        requests = [make_active_request(["check_a"], request_id=n, execution_ids={"check_a": n}) for n in range(1, 26)]
        scheduler = asyncio.create_task(harness.room.scheduler_loop())
        try:
            for req in requests:
                reserve_and_admit(harness.room, req)
            for _ in range(2000):
                if len(harness.room._durable_jobs) == len(requests):
                    break
                await asyncio.sleep(0)
        finally:
            await cancel_scheduler(scheduler)
        harness.rows = [_row("in_progress", execution_id=n) for n in range(1, 26)]

        await harness.room.poll_durable_jobs()

        assert harness.claim_writes == [list(range(1, 26))], "25 observed claims, one round trip"
        assert all(req.queue.get_job("check_a").state == JobState.IN_PROGRESS for req in requests)

    @pytest.mark.asyncio
    async def test_only_the_newly_claimed_jobs_are_written(self) -> None:
        """A pass writes what it learned, not everything it is holding."""
        harness = _Harness()
        first = make_active_request(["check_a"], request_id=1, execution_ids={"check_a": 1})
        second = make_active_request(["check_a"], request_id=2, execution_ids={"check_a": 2})
        scheduler = asyncio.create_task(harness.room.scheduler_loop())
        try:
            reserve_and_admit(harness.room, first)
            reserve_and_admit(harness.room, second)
            for _ in range(2000):
                if len(harness.room._durable_jobs) == 2:
                    break
                await asyncio.sleep(0)
        finally:
            await cancel_scheduler(scheduler)

        harness.rows = [_row("in_progress", execution_id=1), _row("queued", execution_id=2, claimed_by=None)]
        await harness.room.poll_durable_jobs()
        harness.rows = [_row("in_progress", execution_id=1), _row("in_progress", execution_id=2)]
        await harness.room.poll_durable_jobs()

        assert harness.claim_writes == [[1], [2]], "job 1 is already recorded by the second pass; only job 2 is new"

    @pytest.mark.asyncio
    async def test_a_successful_row_completes_the_request(self) -> None:
        harness = _Harness()
        req = await harness.enqueue_new()
        harness.rows = [_row("success", response_json={"data": {"score": 0.9}, "metadata": {"duration_ms": 10}})]

        await harness.room.poll_durable_jobs()

        assert req.completion_future.done()
        assert req.aggregated_results["check_a"] == {"score": 0.9}
        assert harness.room._durable_jobs == {}

    @pytest.mark.asyncio
    async def test_a_successful_row_is_not_written_back_to(self) -> None:
        """The pod already recorded the result; re-recording it from here could only lose detail."""
        harness = _Harness()
        req = await harness.enqueue_new()
        harness.rows = [_row("success", response_json={"data": {}, "metadata": {}})]

        await harness.room.poll_durable_jobs()

        _record(req).complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_failed_row_fails_the_job_with_the_pod_s_own_error(self) -> None:
        harness = _Harness()
        req = await harness.enqueue_new()
        harness.rows = [_row("fail", error_message="ffprobe found no frames")]

        await harness.room.poll_durable_jobs()

        assert req.completion_future.done()
        assert req.aggregated_results["check_a"] == {"error": "ffprobe found no frames"}
        _record(req).fail.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_job_still_waiting_for_a_pod_is_left_alone(self) -> None:
        harness = _Harness()
        req = await harness.enqueue_new()
        harness.rows = [_row("queued", claimed_by=None)]

        await harness.room.poll_durable_jobs()

        assert not req.completion_future.done()
        assert _EXECUTION_ID in harness.room._durable_jobs

    @pytest.mark.asyncio
    async def test_rows_that_do_not_come_back_are_left_alone(self) -> None:
        """An unreachable database reads as zero rows; failing everything over that would be far worse."""
        harness = _Harness()
        req = await harness.enqueue_new()
        harness.rows = []

        await harness.room.poll_durable_jobs()

        assert not req.completion_future.done()
        assert _EXECUTION_ID in harness.room._durable_jobs

    @pytest.mark.asyncio
    async def test_polling_with_nothing_in_flight_does_not_query(self) -> None:
        harness = _Harness()
        reader = AsyncMock()
        harness.room._read_job_states = reader

        await harness.room.poll_durable_jobs()

        reader.assert_not_awaited()


class TestStaleClaimRace:
    """A claim seen on a stale snapshot must not restart a job that has since finished.

    The polling pass reads every tracked row once and then walks them, so a long pass acts on state that has moved
    on. This is the case that broke: the row went terminal mid-walk, the arbitrator wrote ``in_progress`` over it
    from the snapshot, and the job was left running forever in a state nothing recovers from -- its result already
    discarded, its lease NULL so the reaper skips it, and no terminal status left for the monitor to wait for.

    The row is what settles it, so these drive the refusal the database now answers with.
    """

    @pytest.mark.asyncio
    async def test_a_job_that_finished_mid_pass_is_not_started(self) -> None:
        harness = _Harness()
        req = await harness.enqueue_new()
        # The snapshot says claimed; the write is refused because the row is terminal by the time it lands.
        harness.rows = [_row("in_progress")]
        harness.refused = {_EXECUTION_ID: "success"}

        await harness.room.poll_durable_jobs()

        assert req.queue.get_job("check_a").state != JobState.IN_PROGRESS
        assert not req.completion_future.done(), "the outcome comes from the row, on the next pass"
        assert _EXECUTION_ID in harness.room._durable_jobs, "the job stays tracked so its row is read again"

    @pytest.mark.asyncio
    async def test_no_budget_clock_starts_for_a_job_that_already_finished(self) -> None:
        """Starting one would give a finished job a deadline, and time it out on the row's real result."""
        harness = _Harness()
        await harness.enqueue_new()
        harness.rows = [_row("in_progress")]
        harness.refused = {_EXECUTION_ID: "success"}

        await harness.room.poll_durable_jobs()

        assert harness.room._durable_jobs[_EXECUTION_ID].started_at is None

    @pytest.mark.asyncio
    async def test_the_next_pass_settles_it_from_the_row(self) -> None:
        """The refusal defers the decision rather than dropping it: the row still has the answer."""
        harness = _Harness()
        req = await harness.enqueue_new()
        harness.rows = [_row("in_progress")]
        harness.refused = {_EXECUTION_ID: "success"}
        await harness.room.poll_durable_jobs()

        harness.refused = {}
        harness.rows = [_row("success", response_json={"data": {"score": 0.7}, "metadata": {}})]
        await harness.room.poll_durable_jobs()

        assert req.completion_future.done()
        assert req.aggregated_results["check_a"] == {"score": 0.7}
        assert harness.room._durable_jobs == {}

    @pytest.mark.asyncio
    async def test_a_write_that_merely_failed_still_leaves_the_job_running(self) -> None:
        """Only a known-terminal row stops the transition.

        A write that failed says nothing about whether the job finished, and the job still needs a budget clock and
        a place in the queue's in-progress count -- otherwise a database blip turns a running job into one nothing
        is watching.
        """
        harness = _Harness()
        req = await harness.enqueue_new()
        harness.rows = [_row("in_progress")]
        harness.room._record_claims = AsyncMock(side_effect=RuntimeError("database is unreachable"))

        await harness.room.poll_durable_jobs()

        assert req.queue.get_job("check_a").state == JobState.IN_PROGRESS
        assert harness.room._durable_jobs[_EXECUTION_ID].started_at is not None

    @pytest.mark.asyncio
    async def test_a_budget_expiry_racing_the_pod_leaves_the_pod_s_verdict(self) -> None:
        """The pod that ran the work wrote a better account than the arbitrator's outside guess.

        ``_fail_job`` reports the row as terminalized only if its own write landed, so a refused one must not have
        the arbitrator believe it recorded a timeout it did not.
        """
        harness = _Harness()
        req = await harness.enqueue_new()
        await harness.claim(req)
        entry = harness.room._durable_jobs[_EXECUTION_ID]
        entry.started_at = -harness.room.JOB_BUDGET_SECONDS
        _record(req).timeout = AsyncMock(return_value=False)
        harness.rows = []

        await harness.room.poll_durable_jobs()

        _record(req).timeout.assert_awaited_once()
        assert "check_a" not in req.db_terminalized


class TestDeadPodTakeover:
    """A pod that stops renewing is the failure a connection-bound dispatch could not see at all."""

    @pytest.mark.asyncio
    async def test_a_requeued_job_waits_for_another_pod_rather_than_being_re_sent(self) -> None:
        """The row going back on the queue *is* the redelivery; there is nothing for the arbitrator to do."""
        harness = _Harness()
        req = await harness.enqueue_new()
        await harness.claim(req)

        harness.rows = [_row("queued", claimed_by=None, attempt=1)]
        await harness.room.poll_durable_jobs()

        assert _EXECUTION_ID in harness.room._durable_jobs
        assert not req.completion_future.done()

    @pytest.mark.asyncio
    async def test_the_next_pod_to_claim_it_is_noticed_as_a_fresh_claim(self) -> None:
        harness = _Harness()
        req = await harness.enqueue_new()
        await harness.claim(req)
        harness.rows = [_row("queued", claimed_by=None, attempt=1)]
        await harness.room.poll_durable_jobs()

        harness.rows = [_row("in_progress", claimed_by="pod-b/def", attempt=2)]
        await harness.room.poll_durable_jobs()

        assert harness.claim_writes == [[_EXECUTION_ID], [_EXECUTION_ID]], (
            "the second pod's claim is a fresh observation and has to re-refine start_time"
        )
        assert req.queue.get_job("check_a").state == JobState.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_a_job_that_keeps_killing_pods_eventually_fails(self) -> None:
        harness = _Harness()
        req = await harness.enqueue_new()
        await harness.claim(req)

        harness.rows = [_row("queued", claimed_by=None, attempt=WaitingRoom.JOB_MAX_ATTEMPTS)]
        await harness.room.poll_durable_jobs()

        assert req.completion_future.done()
        assert "stopped responding" in req.aggregated_results["check_a"]["error"]
        # Nobody else will write this row's terminal state: the pods that owned it are gone.
        _record(req).fail.assert_awaited_once()


class TestReaper:
    """Returning dead claims to the queue, which every replica does for the whole fleet."""

    @pytest.mark.asyncio
    async def test_reaping_leaves_this_replicas_own_bookkeeping_alone(self) -> None:
        """A requeued job is still outstanding here: its result will arrive on the same row it always would have."""
        harness = _Harness()
        await harness.enqueue_new()
        harness.requeued = [RequeuedCheckerJob(execution_id=_EXECUTION_ID, checker_name="check_a", attempt=1)]

        await harness.room.reap_expired_jobs()

        assert _EXECUTION_ID in harness.room._durable_jobs

    @pytest.mark.asyncio
    async def test_the_reap_covers_jobs_this_replica_knows_nothing_about(self) -> None:
        """The jobs that most need reaping belong to requests whose own replica is gone."""
        harness = _Harness()
        reaper = AsyncMock(return_value=[])
        harness.room._requeue_expired_jobs = reaper

        await harness.room.reap_expired_jobs()

        reaper.assert_awaited_once_with()

    @pytest.mark.asyncio
    async def test_a_replica_with_no_database_does_not_try_to_reap(self) -> None:
        harness = _Harness()
        harness.room._requeue_expired_jobs = None

        # Returns rather than raising, so the monitor loop of a database-less replica keeps running.
        await harness.room.reap_expired_jobs()


class TestBudget:
    """The backstop for a job that hangs while still heartbeating, which no lease can catch.

    The clock runs from the claim, not from the enqueue: a job still waiting its turn cannot be hung, and charging it
    for the backlog ahead of it would fail work that was never given a chance to run.
    """

    @pytest.mark.asyncio
    async def test_an_overrunning_job_is_cancelled_and_timed_out(self) -> None:
        harness = _Harness()
        req = await harness.enqueue_new()
        await harness.claim(req)
        entry = harness.room._durable_jobs[_EXECUTION_ID]
        assert entry.started_at is not None, "the claim is what starts the clock"
        entry.started_at -= WaitingRoom.JOB_BUDGET_SECONDS + 1
        harness.rows = [_row("in_progress")]

        await harness.room.poll_durable_jobs()

        assert harness.cancelled == [_EXECUTION_ID], "the owner can only be reached through the row"
        assert req.completion_future.done()
        _record(req).timeout.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_job_inside_its_budget_is_untouched(self) -> None:
        harness = _Harness()
        req = await harness.enqueue_new()
        harness.rows = [_row("in_progress")]

        await harness.room.poll_durable_jobs()

        assert harness.cancelled == []
        assert not req.completion_future.done()

    @pytest.mark.asyncio
    async def test_a_job_no_pod_has_claimed_never_times_out(self) -> None:
        """Backlog is not a hung job. However long the queue makes it wait, an unclaimed job keeps its turn."""
        harness = _Harness()
        req = await harness.enqueue_new()
        entry = harness.room._durable_jobs[_EXECUTION_ID]
        assert entry.started_at is None, "nothing has claimed it, so its clock has not started"
        entry.enqueued_at -= WaitingRoom.JOB_BUDGET_SECONDS * 10
        harness.rows = [_row("queued", claimed_by=None)]

        await harness.room.poll_durable_jobs()

        assert harness.cancelled == []
        assert not req.completion_future.done()

    @pytest.mark.asyncio
    async def test_a_requeued_job_gets_its_budget_back(self) -> None:
        """The lapsed attempt is charged to the pod that dropped it, not to the next one to pick the job up."""
        harness = _Harness()
        req = await harness.enqueue_new()
        await harness.claim(req)
        entry = harness.room._durable_jobs[_EXECUTION_ID]
        assert entry.started_at is not None, "the claim is what starts the clock"
        entry.started_at -= WaitingRoom.JOB_BUDGET_SECONDS - 1

        # The reaper handed the job back: the row is queued again, unclaimed, with the attempt count bumped.
        harness.rows = [_row("queued", claimed_by=None, attempt=1)]
        await harness.room.poll_durable_jobs()

        requeued = harness.room._durable_jobs[_EXECUTION_ID]
        assert requeued.started_at is None, "nobody holds it, so nothing should be accruing against its budget"
        assert harness.cancelled == []
        assert not req.completion_future.done()


class TestMonitorLoop:
    """The loop around the poll."""

    @pytest.mark.asyncio
    async def test_a_failing_poll_does_not_stop_the_next_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        harness = _Harness()
        calls = {"n": 0}

        async def _flaky_poll() -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("database blip")

        monkeypatch.setattr(WaitingRoom, "JOB_POLL_INTERVAL", 0.01)
        monkeypatch.setattr(harness.room, "poll_durable_jobs", _flaky_poll)
        monitor = asyncio.create_task(harness.room.job_monitor_loop())
        try:
            for _ in range(500):
                if calls["n"] >= 2:
                    break
                await asyncio.sleep(0.01)
        finally:
            await cancel_scheduler(monitor)

        assert calls["n"] >= 2

    @pytest.mark.asyncio
    async def test_a_failing_reap_does_not_stop_the_next_poll(self, monkeypatch: pytest.MonkeyPatch) -> None:
        harness = _Harness()
        polls = {"n": 0}

        async def _count_poll() -> None:
            polls["n"] += 1

        async def _broken_reap() -> None:
            raise RuntimeError("database blip")

        monkeypatch.setattr(WaitingRoom, "JOB_POLL_INTERVAL", 0.01)
        monkeypatch.setattr(harness.room, "poll_durable_jobs", _count_poll)
        monkeypatch.setattr(harness.room, "reap_expired_jobs", _broken_reap)
        monitor = asyncio.create_task(harness.room.job_monitor_loop())
        try:
            for _ in range(500):
                if polls["n"] >= 2:
                    break
                await asyncio.sleep(0.01)
        finally:
            await cancel_scheduler(monitor)

        assert polls["n"] >= 2

    @pytest.mark.asyncio
    async def test_the_monitor_does_not_start_without_a_reader(self) -> None:
        harness = _Harness()
        harness.room._read_job_states = None

        # It returns rather than looping, so awaiting it directly must not hang.
        await asyncio.wait_for(harness.room.job_monitor_loop(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_the_poll_delay_stays_inside_its_jitter_window(self) -> None:
        room = _Harness().room
        spread = room.JOB_POLL_INTERVAL * room.JOB_POLL_JITTER
        delays = [room._poll_delay() for _ in range(50)]

        assert all(room.JOB_POLL_INTERVAL - spread <= delay <= room.JOB_POLL_INTERVAL + spread for delay in delays)
        assert len(set(delays)) > 1, "a fixed interval would let replicas synchronize into a stampede"


class TestFinalizeCleanup:
    """A finalized request must not leave anything behind for the monitor to keep polling."""

    @pytest.mark.asyncio
    async def test_finalizing_drops_leftover_in_flight_entries(self) -> None:
        harness = _Harness()
        req = await harness.enqueue_new()
        harness.room._durable_jobs[9999] = _DurableJob(
            req=req, job=req.queue.get_job("check_a"), execution_id=9999, enqueued_at=0.0
        )
        harness.rows = [_row("success", response_json={"data": {}, "metadata": {}})]

        await harness.room.poll_durable_jobs()

        assert harness.room._durable_jobs == {}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
