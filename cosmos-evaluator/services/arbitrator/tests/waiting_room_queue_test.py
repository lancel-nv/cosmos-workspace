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

"""Behavioral tests for :class:`services.arbitrator.waiting_room_queue.WaitingRoom`.

These cover everything the waiting room does around a job rather than to it: the admission and capacity contract,
draining, the metrics snapshot, finalization, and the error branches on the way to writing a job to its row. Each test
runs against a real ``WaitingRoom`` with stub callbacks, so the scheduler logic is exercised in isolation from the
FastAPI app and the database.

The job lifecycle itself -- written to a row, claimed, finished, failed, requeued, wedged -- lives in
``durable_dispatch_test``, which scripts the row states a checker fleet can present.
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
import sys
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from services.arbitrator.checker_queue import CheckerJob, CheckerQueue, JobState
from services.arbitrator.tests.waiting_room_test_helpers import (
    cancel_scheduler as _cancel_scheduler,
)
from services.arbitrator.tests.waiting_room_test_helpers import (
    make_active_request as _make_active_request,
)
from services.arbitrator.tests.waiting_room_test_helpers import (
    make_waiting_room as _room,
)
from services.arbitrator.tests.waiting_room_test_helpers import (
    reserve_and_admit as _reserve_and_admit,
)
from services.arbitrator.waiting_room_queue import (
    LIVE_JOB_STATES,
    ActiveRequest,
    WaitingRoom,
    WaitingRoomMetrics,
    WaitingRoomStateError,
    _request_id,
)

# ---------------------------------------------------------------------------
# Test stubs and helpers
# ---------------------------------------------------------------------------


def _single_job(req: ActiveRequest, name: str = "check_a") -> CheckerJob:
    """Return the named job from ``req``'s queue (asserts it exists)."""
    for job in req.queue.iter_jobs():
        if job.name == name:
            return job
    raise AssertionError(f"job {name} not found")


def _record(req: ActiveRequest, name: str = "check_a") -> MagicMock:
    """The stub ``CheckerExecutionRecord`` behind ``name``, typed so its awaits can be asserted on."""
    return cast(MagicMock, req.checker_executions[name])


def _taken_job(req: ActiveRequest, name: str = "check_a") -> CheckerJob:
    """The named job, flipped out of the ready set the way a sweep leaves it before calling ``_enqueue``."""
    req.queue.mark_queued(name)
    return _single_job(req, name)


def _set_job_states(req: ActiveRequest, states: dict[str, JobState]) -> None:
    """Override the state of named jobs on ``req.queue`` (mutates in place)."""
    for job in req.queue.iter_jobs():
        if job.name in states:
            job.state = states[job.name]


def _make_dep_request() -> ActiveRequest:
    """Build an ActiveRequest with ``dependent`` depending on ``dep`` so failing ``dep`` cascades a SKIP."""
    jobs = [
        CheckerJob(name="dep", base_name="dep", priority=0, deps=set()),
        CheckerJob(name="dependent", base_name="dependent", priority=0, deps={"dep"}),
    ]
    queue = CheckerQueue(jobs)

    execution = MagicMock()
    execution.request_id = 1
    execution.in_progress = AsyncMock()
    execution.complete = AsyncMock(return_value=0)
    execution.fail = AsyncMock(return_value=0)

    records: dict[str, Any] = {}
    for name in ("dep", "dependent"):
        record = MagicMock()
        record.enqueue = AsyncMock(return_value=True)
        record.in_progress = AsyncMock()
        record.complete = AsyncMock()
        record.fail = AsyncMock()
        record.skipped = AsyncMock()
        record.timeout = AsyncMock()
        records[name] = record

    async def _finalize(_active: ActiveRequest) -> dict:
        return {"ok": True}

    return ActiveRequest(
        execution=execution,
        request=MagicMock(),
        queue=queue,
        checker_executions=cast(Any, records),
        health_info={},
        aggregated_results={},
        completion_future=asyncio.get_event_loop().create_future(),
        finalize=_finalize,
    )


# ---------------------------------------------------------------------------
# Terminal-at-admission
# ---------------------------------------------------------------------------


class TestTerminalAtAdmission:
    """A request whose CheckerQueue is born terminal (no READY jobs) must still have its ``completion_future``
    resolved by :meth:`WaitingRoom.admit`. Reproduces the production scenario where the only requested checker has
    every one of its jobs SKIPPED at admission, so the queue is terminal on arrival and the sweep would otherwise
    find nothing to do.
    """

    @pytest.mark.asyncio
    async def test_all_skipped_queue_finalizes_on_admit(self) -> None:
        """A queue with only SKIPPED jobs resolves ``completion_future`` without anything being enqueued."""
        room = _room()
        scheduler = asyncio.create_task(room.scheduler_loop())
        try:
            req = _make_active_request(["skipped_check"])
            for job in req.queue.iter_jobs():
                job.state = JobState.SKIPPED
                job.error = "no matching config at admission"
            assert req.queue.is_done(), "test setup invariant: queue should be terminal pre-admit"
            _reserve_and_admit(room, req)
            result = await asyncio.wait_for(req.completion_future, timeout=2.0)
            assert "results" in result
            assert req not in room.queue, "terminal request must be evicted from the FIFO"
        finally:
            await _cancel_scheduler(scheduler)

    @pytest.mark.asyncio
    async def test_empty_queue_finalizes_on_admit(self) -> None:
        """
        An empty CheckerQueue (e.g. per-camera checkers with empty ``augmented_video_urls``) finalizes immediately.
        """
        room = _room()
        scheduler = asyncio.create_task(room.scheduler_loop())
        try:
            req = _make_active_request([])  # zero jobs => is_done() is vacuously True
            assert req.queue.is_done(), "test setup invariant: empty queue should be terminal pre-admit"
            _reserve_and_admit(room, req)
            await asyncio.wait_for(req.completion_future, timeout=2.0)
            assert req not in room.queue
        finally:
            await _cancel_scheduler(scheduler)


# ---------------------------------------------------------------------------
# Metrics snapshot
# ---------------------------------------------------------------------------


_LIVE_STATE_VALUES: tuple[str, ...] = tuple(state.value for state in LIVE_JOB_STATES)


def _zero_states() -> dict[str, int]:
    """Return the all-zero per-checker state map that ``snapshot_metrics`` pre-seeds for a known checker."""
    return {state: 0 for state in _LIVE_STATE_VALUES}


class TestSnapshotMetrics:
    """Tests for :meth:`WaitingRoom.snapshot_metrics`.

    The metrics endpoint is a thin renderer over this method, so these tests carry the bulk of the behavioral coverage
    for the gauge definitions (in-progress vs. queued request bookkeeping, per-checker per-state counting across all
    four live :class:`JobState` values, and the seeded ``known_checkers`` contract).
    """

    @pytest.mark.asyncio
    async def test_empty_room_emits_zero_for_seeded_checkers(self) -> None:
        """An empty room returns zero counts; ``known_checkers`` seeds the per-checker, per-state map."""
        room = _room()

        snapshot = room.snapshot_metrics(["av.egomotion", "av.objects_single_view"])

        assert snapshot == WaitingRoomMetrics(
            in_progress=0,
            queued=0,
            per_checker_state={
                "av.egomotion": _zero_states(),
                "av.objects_single_view": _zero_states(),
            },
        )

    @pytest.mark.asyncio
    async def test_in_progress_marked_separates_in_progress_from_queued(self) -> None:
        """``ActiveRequest.in_progress_marked`` is the boundary between the two request gauges."""
        room = _room()
        for _ in range(3):
            req = _make_active_request(["check_a"])
            req.in_progress_marked = True
            room.queue.append(req)
        for _ in range(5):
            req = _make_active_request(["check_a"])
            room.queue.append(req)  # in_progress_marked defaults to False

        snapshot = room.snapshot_metrics(["check_a"])

        assert snapshot.in_progress == 3
        assert snapshot.queued == 5

    @pytest.mark.asyncio
    async def test_per_checker_counts_each_live_state_separately(self) -> None:
        """All four live ``JobState`` values count into their own bucket; terminal states are excluded."""
        room = _room()

        # Request A: one job each in READY / QUEUED -- both should show up under their own state.
        req_a = _make_active_request(["check_a_ready", "check_a_queued"], request_id=1)
        req_a.in_progress_marked = True
        _set_job_states(
            req_a,
            {
                "check_a_queued": JobState.QUEUED,
                # check_a_ready already READY by default
            },
        )
        room.queue.append(req_a)

        # Request B: two jobs for different checkers -- READY and IN_PROGRESS -- both count, in their own buckets.
        req_b = _make_active_request(["check_b_one", "check_b_two"], request_id=2)
        req_b.in_progress_marked = True
        _set_job_states(req_b, {"check_b_two": JobState.IN_PROGRESS})
        room.queue.append(req_b)

        # Request C: one PENDING job (e.g. waiting on a dep) -- counts.
        req_c = _make_active_request(["check_c"], request_id=3)
        _set_job_states(req_c, {"check_c": JobState.PENDING})
        room.queue.append(req_c)

        # Request D: one COMPLETED job -- excluded (terminal).
        req_d = _make_active_request(["check_d"], request_id=4)
        req_d.in_progress_marked = True
        _set_job_states(req_d, {"check_d": JobState.COMPLETED})
        room.queue.append(req_d)

        snapshot = room.snapshot_metrics(
            ["check_a_ready", "check_a_queued", "check_b_one", "check_b_two", "check_c", "check_d"]
        )

        # in-progress: A, B, D = 3; queued: C = 1.
        assert snapshot.in_progress == 3
        assert snapshot.queued == 1

        # Per-checker, per-state counts.
        assert snapshot.per_checker_state["check_a_ready"] == {
            "pending": 0,
            "ready": 1,
            "queued": 0,
            "in_progress": 0,
        }
        assert snapshot.per_checker_state["check_a_queued"] == {
            "pending": 0,
            "ready": 0,
            "queued": 1,
            "in_progress": 0,
        }
        assert snapshot.per_checker_state["check_b_one"] == {
            "pending": 0,
            "ready": 1,
            "queued": 0,
            "in_progress": 0,
        }
        assert snapshot.per_checker_state["check_b_two"] == {
            "pending": 0,
            "ready": 0,
            "queued": 0,
            "in_progress": 1,
        }
        assert snapshot.per_checker_state["check_c"] == {
            "pending": 1,
            "ready": 0,
            "queued": 0,
            "in_progress": 0,
        }
        # check_d's only job is COMPLETED (terminal) -- not counted, but the seeded zeros remain.
        assert snapshot.per_checker_state["check_d"] == _zero_states()

    @pytest.mark.asyncio
    async def test_unknown_checker_in_request_is_still_counted(self) -> None:
        """Checkers that appear in a request but were not in ``known_checkers`` are still counted.

        ``known_checkers`` controls the *seed* set (zeros for known-idle checkers); it does not gate counting. This
        matches the operational intent: if work for an unconfigured checker shows up in a request, it should be
        visible in the metric. The per-state map is lazily extended with all live states for the new checker so the
        rendered Prometheus output is structurally identical to a seeded entry.
        """
        room = _room()
        req = _make_active_request(["surprise_checker"])  # READY by default
        room.queue.append(req)

        snapshot = room.snapshot_metrics(["known_checker"])

        assert snapshot.per_checker_state == {
            "known_checker": _zero_states(),
            "surprise_checker": {"pending": 0, "ready": 1, "queued": 0, "in_progress": 0},
        }

    @pytest.mark.asyncio
    async def test_returned_metrics_are_immutable(self) -> None:
        """The returned dataclass is frozen so callers cannot mutate it accidentally."""
        room = _room()
        snapshot = room.snapshot_metrics(["check_a"])

        with pytest.raises(FrozenInstanceError):
            snapshot.in_progress = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Capacity / reservation contract
# ---------------------------------------------------------------------------


class TestCapacityReservation:
    """Tests for the ``capacity`` constructor parameter and the reservation API.

    These exercise :meth:`WaitingRoom.try_admit` / :meth:`WaitingRoom.admit` / :meth:`WaitingRoom.release` directly
    without running the scheduler, since the scheduler is irrelevant to the capacity-gating decision.
    """

    @pytest.mark.asyncio
    async def test_unbounded_room_accepts_any_number_of_reservations(self) -> None:
        """``capacity=None`` (the default) lets ``try_admit`` succeed indefinitely."""
        room = _room()
        for _ in range(50):
            assert room.try_admit() is True
        assert room.is_at_capacity() is False

    @pytest.mark.asyncio
    async def test_bounded_room_fills_then_rejects(self) -> None:
        """``try_admit`` succeeds exactly ``capacity`` times back-to-back, then rejects."""
        room = _room(capacity=2)
        assert room.try_admit() is True
        assert room.try_admit() is True
        assert room.is_at_capacity() is True
        assert room.try_admit() is False
        assert room.try_admit() is False

    @pytest.mark.asyncio
    async def test_admit_consumes_reservation_and_appends_to_queue(self) -> None:
        """A successful ``admit`` pops the reservation but leaves the queue +1 — capacity holds."""
        room = _room(capacity=1)
        req = _make_active_request(["check_a"])

        assert room.try_admit() is True
        assert room.is_at_capacity() is True  # reservation alone fills the room
        room.admit(req)
        assert len(room.queue) == 1
        assert room.is_at_capacity() is True  # now the queue itself is full
        assert room.try_admit() is False

    @pytest.mark.asyncio
    async def test_release_returns_a_slot_to_the_pool(self) -> None:
        """``release`` after ``try_admit`` lets a subsequent ``try_admit`` succeed."""
        room = _room(capacity=1)
        assert room.try_admit() is True
        assert room.try_admit() is False
        room.release()
        assert room.try_admit() is True

    @pytest.mark.asyncio
    async def test_admit_without_reservation_raises_state_error(self) -> None:
        """An unbalanced ``admit`` is a caller bug and raises ``WaitingRoomStateError``."""
        room = _room()
        req = _make_active_request(["check_a"])
        with pytest.raises(WaitingRoomStateError):
            room.admit(req)

    @pytest.mark.asyncio
    async def test_release_without_reservation_is_a_warning_noop(self, caplog: pytest.LogCaptureFixture) -> None:
        """A spurious ``release`` (e.g. cancellation race) logs but does not raise or underflow."""
        room = _room()
        with caplog.at_level("WARNING", logger="services.arbitrator.waiting_room_queue"):
            room.release()
        assert any("release() called with no outstanding reservations" in r.message for r in caplog.records)
        # Counter never went negative.
        assert room.try_admit() is True

    @pytest.mark.asyncio
    async def test_concurrent_try_admit_does_not_oversubscribe(self) -> None:
        """``capacity+1`` concurrent reservation attempts on the same event loop never exceed capacity.

        Since ``try_admit`` is synchronous (no ``await``), a single event loop guarantees the check and reservation
        increment run as one atomic unit. This test pins that property down with an explicit burst.
        """
        room = _room(capacity=3)

        async def _attempt() -> bool:
            return room.try_admit()

        results = await asyncio.gather(*[_attempt() for _ in range(10)])
        assert sum(results) == 3
        assert sum(1 for r in results if not r) == 7


class TestDraining:
    """Shutting a replica down without stranding what it has already taken on."""

    @pytest.mark.asyncio
    async def test_a_draining_room_refuses_admissions_it_has_room_for(self) -> None:
        room = _room()
        assert room.try_admit() is True
        room.release()

        room.begin_draining()

        assert room.is_draining is True
        assert room.try_admit() is False, "room to spare is not the question once the process is leaving"

    @pytest.mark.asyncio
    async def test_draining_twice_is_harmless(self) -> None:
        room = _room()
        room.begin_draining()
        room.begin_draining()
        assert room.is_draining is True

    @pytest.mark.asyncio
    async def test_an_empty_room_has_nothing_to_wait_for(self) -> None:
        room = _room()
        assert room.has_work_in_flight() is False
        assert await room.wait_until_drained(timeout=0.0) is True

    @pytest.mark.asyncio
    async def test_an_uncommitted_reservation_counts_as_work(self) -> None:
        """The submission holding it is mid-flight; dropping it now would answer the client with nothing."""
        room = _room()
        assert room.try_admit() is True

        assert room.has_work_in_flight() is True
        room.release()
        assert room.has_work_in_flight() is False

    @pytest.mark.asyncio
    async def test_the_wait_ends_when_the_last_request_finishes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A request whose jobs are all terminal at admission drains without anything having to be claimed."""
        monkeypatch.setattr(WaitingRoom, "DRAIN_POLL_INTERVAL", 0.01)
        room = _room()
        scheduler = asyncio.create_task(room.scheduler_loop())
        try:
            req = _make_active_request([])
            _reserve_and_admit(room, req)
            room.begin_draining()

            assert await room.wait_until_drained(timeout=3.0) is True
            assert req.completion_future.done(), "drained means finished, not abandoned"
        finally:
            await _cancel_scheduler(scheduler)

    @pytest.mark.asyncio
    async def test_the_wait_gives_up_on_a_request_that_never_finishes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(WaitingRoom, "DRAIN_POLL_INTERVAL", 0.01)
        room = _room()
        room.queue.append(_make_active_request(["check_a"]))

        assert await room.wait_until_drained(timeout=0.05) is False

    @pytest.mark.asyncio
    async def test_a_request_being_finalized_is_still_work(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``_maybe_finalize`` leaves the FIFO before it writes the terminal row, so the FIFO alone is not the test."""
        monkeypatch.setattr(WaitingRoom, "DRAIN_POLL_INTERVAL", 0.01)
        room = _room()
        inside_finalize = asyncio.Event()
        let_finalize_finish = asyncio.Event()

        async def _slow_finalize(_active: ActiveRequest) -> dict:
            inside_finalize.set()
            await let_finalize_finish.wait()
            return {}

        req = _make_active_request(["check_a"], finalize=_slow_finalize)
        req.queue.get_job("check_a").state = JobState.COMPLETED
        room.queue.append(req)
        finalizing = asyncio.create_task(room._maybe_finalize(req))
        await asyncio.wait_for(inside_finalize.wait(), timeout=1.0)

        assert not room.queue, "the request is out of the FIFO by now"
        assert room.has_work_in_flight() is True
        assert await room.wait_until_drained(timeout=0.05) is False

        let_finalize_finish.set()
        await asyncio.wait_for(finalizing, timeout=1.0)
        assert room.has_work_in_flight() is False

    @pytest.mark.asyncio
    async def test_a_finalize_that_raises_still_releases_the_drain(self) -> None:
        room = _room()

        async def _boom_finalize(_active: ActiveRequest) -> dict:
            raise RuntimeError("report generation failed")

        req = _make_active_request(["check_a"], finalize=_boom_finalize)
        req.queue.get_job("check_a").state = JobState.COMPLETED
        room.queue.append(req)

        await room._maybe_finalize(req)

        assert room.has_work_in_flight() is False, "a failed finalize would otherwise wedge every future drain"


# ---------------------------------------------------------------------------
# Everything that can go wrong on the way to writing a job to its row
# ---------------------------------------------------------------------------


class TestEnqueueFailureBranches:
    """A job that never reaches its row must end up terminal, not silently outstanding.

    ``durable_dispatch_test`` covers the successful write and everything after it; these are the branches where the
    handover never happens, driven by awaiting :meth:`WaitingRoom._enqueue` directly.
    """

    @pytest.mark.asyncio
    async def test_a_checker_with_no_implementation_completes_with_a_placeholder(self) -> None:
        def _raise_not_implemented(job: CheckerJob, request: Any, queue: CheckerQueue) -> dict[str, Any]:
            raise NotImplementedError

        room = _room(build_request=_raise_not_implemented)
        req = _make_active_request(["check_a"])
        await room._enqueue(req, _taken_job(req))

        assert req.queue.is_done()
        _record(req).complete.assert_awaited()
        _record(req).enqueue.assert_not_awaited()
        assert req.in_progress_marked is True

    @pytest.mark.asyncio
    async def test_a_payload_that_cannot_be_built_fails_the_job(self) -> None:
        def _boom(job: CheckerJob, request: Any, queue: CheckerQueue) -> dict[str, Any]:
            raise ValueError("cannot build")

        room = _room(build_request=_boom)
        req = _make_active_request(["check_a"])
        await room._enqueue(req, _taken_job(req))

        assert "cannot build" in req.aggregated_results["check_a"]["error"]
        _record(req).fail.assert_awaited()

    @pytest.mark.asyncio
    async def test_a_job_with_no_row_fails_rather_than_waiting_for_a_claim_that_cannot_come(self) -> None:
        room = _room()
        req = _make_active_request(["check_a"])
        _record(req).execution_id = None

        await room._enqueue(req, _taken_job(req))

        assert "No checker_executions row" in req.aggregated_results["check_a"]["error"]
        assert not room._durable_jobs

    @pytest.mark.asyncio
    async def test_a_payload_that_could_not_be_written_fails_the_job(self) -> None:
        """Leaving the row queued would hand some pod a job it has no payload to run."""
        room = _room()
        req = _make_active_request(["check_a"])
        _record(req).enqueue = AsyncMock(return_value=False)

        await room._enqueue(req, _taken_job(req))

        assert "Could not record the payload" in req.aggregated_results["check_a"]["error"]
        _record(req).fail.assert_awaited()
        assert not room._durable_jobs

    @pytest.mark.asyncio
    async def test_an_unexpected_error_still_terminalizes_the_job(self) -> None:
        room = _room()
        req = _make_active_request(["check_a"])
        _record(req).enqueue = AsyncMock(side_effect=RuntimeError("kaboom"))

        await room._enqueue(req, _taken_job(req))

        assert req.aggregated_results["check_a"]["error"] == "Internal scheduler error"
        assert req.completion_future.done(), "the finally clause always finalizes"

    @pytest.mark.asyncio
    async def test_an_already_abandoned_request_is_not_enqueued_at_all(self) -> None:
        """One sweep spawns every eligible job at once, so an abandoned request can still have tasks behind it."""
        room = _room()
        req = _make_active_request(["check_a"])
        room._abandon(req, RuntimeError("queue would not terminalize"))

        await room._enqueue(req, _taken_job(req))

        _record(req).enqueue.assert_not_awaited()
        assert not room._durable_jobs
        assert isinstance(req.completion_future.exception(), RuntimeError)

    @pytest.mark.asyncio
    async def test_a_job_abandoned_mid_write_is_not_handed_to_the_monitor(self) -> None:
        """``_abandon`` drops the request's durable jobs, so one registered after it would be polled to no purpose."""
        room = _room()
        req = _make_active_request(["check_a"])

        async def _abandon_then_report_written(_payload: dict[str, Any]) -> bool:
            room._abandon(req, RuntimeError("queue would not terminalize"))
            return True

        _record(req).enqueue = AsyncMock(side_effect=_abandon_then_report_written)

        await room._enqueue(req, _taken_job(req))

        assert not room._durable_jobs
        assert isinstance(req.completion_future.exception(), RuntimeError)

    @pytest.mark.asyncio
    async def test_even_a_broken_recovery_path_does_not_escape_the_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        room = _room()
        req = _make_active_request(["check_a"])
        _record(req).enqueue = AsyncMock(side_effect=RuntimeError("kaboom"))
        monkeypatch.setattr(req.queue, "mark_failed", MagicMock(side_effect=RuntimeError("queue broke")))

        await room._enqueue(req, _taken_job(req))  # must not raise


class TestRecordWriteFailuresAreSwallowed:
    """A ``checker_executions`` write that fails must not take the in-memory run down with it."""

    @pytest.mark.asyncio
    async def test_a_failed_placeholder_write_still_completes_the_job(self) -> None:
        def _raise_not_implemented(job: CheckerJob, request: Any, queue: CheckerQueue) -> dict[str, Any]:
            raise NotImplementedError

        room = _room(build_request=_raise_not_implemented)
        req = _make_active_request(["check_a"])
        _record(req).complete.side_effect = RuntimeError("db down")

        await room._enqueue(req, _taken_job(req))

        assert req.queue.is_done()

    @pytest.mark.asyncio
    async def test_a_failed_fail_write_still_records_the_error_in_memory(self) -> None:
        def _boom(job: CheckerJob, request: Any, queue: CheckerQueue) -> dict[str, Any]:
            raise ValueError("cannot build")

        room = _room(build_request=_boom)
        req = _make_active_request(["check_a"])
        _record(req).fail.side_effect = RuntimeError("db down")

        await room._enqueue(req, _taken_job(req))

        assert "cannot build" in req.aggregated_results["check_a"]["error"]


class TestLifecycleHelpers:
    """Direct coverage for :meth:`_mark_in_progress_if_needed` and :meth:`_apply_skipped_cascade`."""

    @pytest.mark.asyncio
    async def test_mark_in_progress_swallows_execution_error(self) -> None:
        room = _room()
        req = _make_active_request(["check_a"])
        cast(AsyncMock, req.execution.in_progress).side_effect = RuntimeError("db down")
        await room._mark_in_progress_if_needed(req)
        assert req.in_progress_marked is True  # flag flips before the await and stays set

    @pytest.mark.asyncio
    async def test_mark_in_progress_is_idempotent(self) -> None:
        room = _room()
        req = _make_active_request(["check_a"])
        req.in_progress_marked = True
        await room._mark_in_progress_if_needed(req)
        cast(AsyncMock, req.execution.in_progress).assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skipped_cascade_swallows_record_error(self) -> None:
        room = _room()
        req = _make_dep_request()
        req.queue.mark_queued("dep")
        req.queue.mark_failed("dep", "boom")  # cascades 'dependent' to SKIPPED
        cast(AsyncMock, req.checker_executions["dependent"].skipped).side_effect = RuntimeError("db down")
        await room._apply_skipped_cascade(req)
        assert "dependent" not in req.db_terminalized  # the failed write must not be recorded as terminalized

    @pytest.mark.asyncio
    async def test_skipped_cascade_ignores_job_without_record(self) -> None:
        room = _room()
        req = _make_dep_request()
        del req.checker_executions["dependent"]  # no row to update -> the cascade must skip it
        req.queue.mark_queued("dep")
        req.queue.mark_failed("dep", "boom")
        await room._apply_skipped_cascade(req)  # must not raise


class TestMaybeFinalize:
    """:meth:`WaitingRoom._maybe_finalize` guards and error handling."""

    @pytest.mark.asyncio
    async def test_noop_when_not_done(self) -> None:
        room = _room()
        req = _make_active_request(["check_a"])  # READY -> not terminal
        await room._maybe_finalize(req)
        assert not req.completion_future.done()

    @pytest.mark.asyncio
    async def test_noop_when_future_already_done(self) -> None:
        room = _room()
        req = _make_active_request([])  # empty queue is vacuously done
        req.completion_future.set_result({"already": True})
        await room._maybe_finalize(req)
        assert req.completion_future.result() == {"already": True}

    @pytest.mark.asyncio
    async def test_handles_already_removed_request(self) -> None:
        room = _room()
        req = _make_active_request([])  # done, and never appended to room.queue
        await room._maybe_finalize(req)  # queue.remove raises ValueError -> swallowed
        assert req.completion_future.done()

    @pytest.mark.asyncio
    async def test_sets_exception_when_finalize_raises(self) -> None:
        async def _boom_finalize(_active: ActiveRequest) -> dict:
            raise RuntimeError("finalize failed")

        room = _room()
        req = _make_active_request([], finalize=_boom_finalize)
        await room._maybe_finalize(req)
        assert req.completion_future.done()
        with pytest.raises(RuntimeError, match="finalize failed"):
            req.completion_future.result()


class TestSchedulerLoop:
    """The sweep's wake sources and its tolerance for a bad tick."""

    @pytest.mark.asyncio
    async def test_an_idle_scheduler_wakes_on_admit(self) -> None:
        """With nothing admitted the loop blocks indefinitely, so an admission has to be what rouses it."""
        room = _room()
        scheduler = asyncio.create_task(room.scheduler_loop())
        try:
            await asyncio.sleep(0.05)  # let it settle into its wait
            req = _make_active_request(["check_a"])
            _reserve_and_admit(room, req)

            for _ in range(200):
                if _single_job(req).state == JobState.QUEUED:
                    break
                await asyncio.sleep(0)
            assert _single_job(req).state == JobState.QUEUED
        finally:
            await _cancel_scheduler(scheduler)

    @pytest.mark.asyncio
    async def test_scheduler_survives_tick_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        room = _room()
        real_tick = room._tick
        calls = {"n": 0}

        def _flaky_tick() -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")  # first sweep blows up; the loop must log and carry on
            real_tick()

        monkeypatch.setattr(room, "_tick", _flaky_tick)
        scheduler = asyncio.create_task(room.scheduler_loop())
        try:
            # Let the scheduler run its first (failing) tick and settle into its wait before admitting, so the admit's
            # wakeup drives a *second*, healthy tick rather than being consumed by the failing one.
            await asyncio.sleep(0.05)
            assert calls["n"] >= 1
            req = _make_active_request([])
            _reserve_and_admit(room, req)
            await asyncio.wait_for(req.completion_future, timeout=2.0)
            assert calls["n"] >= 2
        finally:
            await _cancel_scheduler(scheduler)


class TestRequestIdHelper:
    """The best-effort ``_request_id`` log helper."""

    @pytest.mark.asyncio
    async def test_returns_request_id(self) -> None:
        req = _make_active_request(["check_a"], request_id=42)
        assert _request_id(req) == 42

    @pytest.mark.asyncio
    async def test_returns_none_on_attribute_error(self) -> None:
        class _NoRequestId:
            @property
            def request_id(self) -> int:
                raise AttributeError("gone")

        req = _make_active_request(["check_a"])
        req.execution = cast(Any, _NoRequestId())
        assert _request_id(req) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
