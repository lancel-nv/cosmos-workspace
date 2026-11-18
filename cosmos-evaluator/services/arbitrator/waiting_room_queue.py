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

"""Per-replica waiting-room queue and event-driven scheduler for the Arbitrator.

Two coroutines run per replica. :meth:`WaitingRoom.scheduler_loop` decides what work is eligible to start and
:meth:`WaitingRoom.job_monitor_loop` learns how the work that started turned out.

Work reaches a checker one way: the arbitrator writes the job's payload to its ``checker_executions`` row and leaves it
there. A checker pod claims it when it has room, and liveness, cancellation and the result all travel through that row.
Nothing is sent to a checker and no connection is held, so neither a gateway recycle nor a pod dying loses the work: the
monitor sees the outcome on the row, and an abandoned claim goes back on the queue for somebody else.

That is why this scheduler has no notion of pacing. There is no capacity to respect at dispatch time, because writing a
row cannot overload anything, and no order to impose, because whichever pod is free next decides what runs. All the
waiting room does is maintain a FIFO of admitted requests (``WaitingRoom.queue``) and, on each sweep, write out every
job whose dependencies are satisfied.

The scheduler is event-driven rather than polled, and sleeps until :meth:`WaitingRoom.admit` appends a request or a
settled job wakes it. When fully idle it blocks indefinitely.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
import logging
import random
import time
from typing import Any, ClassVar

from database.api_models import CheckerExecutionStateRow
from database.statuses import CHECKER_TERMINAL_STATUSES
from services.arbitrator.checker_queue import CheckerJob, CheckerQueue, JobState
from services.arbitrator.records.checker_execution_record import CheckerExecutionRecord
from services.arbitrator.records.execution_record import ExecutionRecord

logger = logging.getLogger(__name__)


class WaitingRoomStateError(RuntimeError):
    """A :class:`WaitingRoom` invariant was violated by a caller.

    Raised by :meth:`WaitingRoom.admit` when no reservation is outstanding (i.e. ``admit()`` was called without a
    paired prior ``try_admit()``). Defined as a typed exception rather than enforced via ``assert`` so the contract
    survives ``python -O`` / ``PYTHONOPTIMIZE=1`` and surfaces in tracebacks and tests.
    """


# Type aliases for the scheduler's helper callbacks. Defined here rather than imported from rest_api_common to avoid a
# circular import.
BuildRequestFn = Callable[[CheckerJob, Any, CheckerQueue], dict[str, Any]]
StoreResultFn = Callable[[dict[str, Any], CheckerJob, Any], None]
FinalizeFn = Callable[["ActiveRequest"], Awaitable[Any]]
ReadJobStatesFn = Callable[[list[int]], Awaitable[list[CheckerExecutionStateRow]]]
RequestJobCancelFn = Callable[[int], Awaitable[bool]]
RequeueExpiredJobsFn = Callable[[], Awaitable[list[Any]]]
# Records observed claims on many jobs at once, answering with the refused ids mapped to the status that won.
RecordClaimsFn = Callable[[list[int], datetime], Awaitable[Mapping[int, str | None]]]


class _RowAction(Enum):
    """What one polled ``checker_executions`` row asks the monitor to do.

    Deciding what a row needs is named separately from acting on it so that a pass can classify every row before it
    writes anything, which is what lets the claim writes be batched. It also leaves planning the batch and applying
    it reading one answer per row, rather than two copies of the same logic free to drift apart.
    """

    # Terminal on the row: adopt the result, or the pod's own diagnosis of its failure.
    COMPLETED = "completed"
    FAILED = "failed"
    # Claimed, and this pass is the first to see it. The only action that owes the row a write.
    CLAIM_OBSERVED = "claim_observed"
    # Was claimed, is not now: the reaper handed it back while nobody was looking.
    REQUEUED = "requeued"
    # Progressing as expected, or waiting to be claimed. Nothing to do but read it again next pass.
    NOTHING = "nothing"


# Live (non-terminal) :class:`JobState` values exposed per checker on ``/metrics``. Terminal states
# (``COMPLETED`` / ``FAILED`` / ``SKIPPED``) are dropped from the waiting-room queue on finalize, so they would always
# be zero from a gauge's perspective and are intentionally omitted. Module-level so callers (e.g.
# ``rest_api_common.get_metrics`` constructing a zero-snapshot before the lifespan has run) can pre-seed the
# per-checker maps without reaching into a private attribute on :class:`WaitingRoom`.
LIVE_JOB_STATES: tuple[JobState, ...] = (
    JobState.PENDING,
    JobState.READY,
    JobState.QUEUED,
    JobState.IN_PROGRESS,
)


@dataclass(frozen=True)
class WaitingRoomMetrics:
    """Point-in-time snapshot of this replica's :class:`WaitingRoom` for metrics emission.

    Counts:

    - ``in_progress`` -- number of admitted requests whose :attr:`ActiveRequest.in_progress_marked` is ``True``
      (i.e. at least one pod has claimed a job for the request).
    - ``queued`` -- admitted requests none of whose jobs has been claimed yet.
    - ``per_checker_state`` -- per checker base name, a nested map from live :class:`JobState` value
      (``"pending" | "ready" | "queued" | "in_progress"`` -- the values of :data:`LIVE_JOB_STATES`) to the count of
      jobs in that state across all admitted requests. Terminal states (``COMPLETED`` / ``FAILED`` / ``SKIPPED``) are
      omitted because finalized requests are removed from the waiting-room queue, so those counts would always be
      zero from a gauge's perspective.

    The outer map is keyed by checker base name and pre-seeded by :meth:`WaitingRoom.snapshot_metrics` with every
    caller-supplied known checker (zero across every live state when idle) so Prometheus ``rate()`` / ``increase()``
    queries do not gap when a checker happens to have no work in a particular state.
    """

    in_progress: int
    queued: int
    per_checker_state: dict[str, dict[str, int]]


@dataclass
class ActiveRequest:
    """One admitted request and the local state the scheduler needs to run it.

    ``aggregated_results`` is filled in by the scheduler as jobs complete; the finalize callback reads it to assemble
    the response. ``in_progress_marked`` ensures the parent execution's ``queued -> in_progress`` flip fires exactly
    once across the whole request's lifetime.
    """

    execution: ExecutionRecord
    request: Any
    queue: CheckerQueue
    checker_executions: dict[str, CheckerExecutionRecord]
    health_info: dict[str, Any]
    aggregated_results: dict[str, Any]
    completion_future: asyncio.Future[Any]
    finalize: FinalizeFn
    in_progress_marked: bool = False
    # Names of jobs whose checker_executions row has already been moved to a terminal status. Lets
    # ``_apply_skipped_cascade`` skip jobs we already finalized on a different code path.
    db_terminalized: set[str] = field(default_factory=set)


@dataclass
class _DurableJob:
    """One job whose outcome will arrive on its row, as the monitor needs to see it.

    Held per replica rather than looked up from the database, because the row says who is working on a job but not
    which in-memory request is waiting for it. The monitor needs both to turn a terminal row into a completed job.

    ``in_progress`` records that some pod has been seen holding this job, so the request's first-claim bookkeeping
    fires once no matter how many polls observe the same claim.

    ``started_at`` is the monotonic stamp of that same first claim and is what :meth:`WaitingRoom._enforce_job_budget`
    measures against; ``None`` until a pod picks the job up. It mirrors the ``start_time``
    :meth:`WaitingRoom._record_observed_claims` writes to the row, so the budget and the recorded duration agree on
    when the job began. Cleared again on requeue, so a redelivered job is judged on its new attempt rather than on
    time charged to a pod that stopped responding.
    """

    req: ActiveRequest
    job: CheckerJob
    execution_id: int
    enqueued_at: float
    cancel_sent: bool = False
    in_progress: bool = False
    started_at: float | None = None


class WaitingRoom:
    """Per-replica scheduler that drains all admitted requests.

    Construct once at process start, register the helper callbacks via :meth:`__init__`, then start
    :meth:`scheduler_loop` and :meth:`job_monitor_loop` as asyncio tasks.
    """

    # How often the monitor reads the rows of the jobs it is waiting on, and how much of that interval is jittered
    # away so several replicas do not converge on the same instant. One batched query per poll, so this is a per-replica
    # query rate rather than a per-job one.
    JOB_POLL_INTERVAL: ClassVar[float] = 10.0
    JOB_POLL_JITTER: ClassVar[float] = 0.3

    # How many pods may take a run at one job before the arbitrator stops letting it go back on the queue and fails it.
    # Work that reliably kills whichever pod picks it up must fail loudly instead of cycling through the fleet forever.
    JOB_MAX_ATTEMPTS: ClassVar[int] = 3

    # Wall-clock ceiling on one job, measured from the moment a pod actually claimed it. A pod that heartbeats forever
    # keeps its lease forever, so the lease alone cannot bound a hung job; this can. Generous, because a large clip
    # legitimately takes tens of minutes and a job killed at its ceiling has to be re-run from the start.
    #
    # Deliberately NOT measured from enqueue: queue wait is backlog, not a hung job, and charging it here fails work
    # that was only ever waiting its turn. A backlog is bounded by ``capacity`` (admission returns 503 so the caller
    # retries elsewhere), not by killing jobs that have already been admitted. An unclaimed job is therefore never
    # budget-eligible, which is what makes this a liveness bound rather than a queueing policy.
    JOB_BUDGET_SECONDS: ClassVar[float] = 5400.0

    # How often :meth:`wait_until_drained` re-examines what is left. Polling rather than an event, because every path
    # that can empty the room -- a finalize, a reservation that never became an admission, a dispatch task unwinding --
    # would otherwise have to remember to signal, and one forgotten signal means a replica that waits out its whole
    # drain budget with nothing left to do.
    DRAIN_POLL_INTERVAL: ClassVar[float] = 0.25

    def __init__(
        self,
        build_request: BuildRequestFn,
        store_result: StoreResultFn,
        capacity: int | None = None,
        read_job_states: ReadJobStatesFn | None = None,
        request_job_cancel: RequestJobCancelFn | None = None,
        requeue_expired_jobs: RequeueExpiredJobsFn | None = None,
        record_claims: RecordClaimsFn | None = None,
    ) -> None:
        self._build_request = build_request
        self._store_result = store_result
        self._read_job_states = read_job_states
        self._request_job_cancel = request_job_cancel
        self._requeue_expired_jobs = requeue_expired_jobs
        # Left unset, a pass still tracks claims locally but stops refining ``start_time`` on the row, so a job's
        # recorded duration includes its queue wait. Degraded reporting rather than broken orchestration.
        self._record_claims = record_claims

        self.capacity = capacity
        self.queue: deque[ActiveRequest] = deque()
        self._wakeup: asyncio.Event = asyncio.Event()
        self._dispatch_tasks: set[asyncio.Task[None]] = set()
        # Published jobs this replica is waiting on, keyed by ``checker_executions.execution_id``.
        self._durable_jobs: dict[int, _DurableJob] = {}
        # Slots reserved by ``try_admit`` but not yet consumed by ``admit`` / released by ``release``. Counted toward
        # fullness so concurrent admissions can't oversubscribe across the ``await`` between reserve and commit.
        self._pending_admissions: int = 0
        # Requests removed from the FIFO but whose terminal row is still being written. See :meth:`has_work_in_flight`.
        self._finalizing: int = 0
        self._draining: bool = False

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def is_at_capacity(self) -> bool:
        """True iff a capacity is configured and the FIFO + outstanding reservations are full."""
        return self.capacity is not None and len(self.queue) + self._pending_admissions >= self.capacity

    def try_admit(self) -> bool:
        """Atomically check capacity and reserve a slot.

        Returns ``True`` if a slot was reserved — the caller MUST eventually call :meth:`admit` (commit) or
        :meth:`release` (rollback). Returns ``False`` if the queue is full or this replica is draining; no slot was
        reserved and the caller must NOT call ``admit()`` or ``release()``.

        Synchronous and free of ``await``s on purpose: the check and the reservation increment must be inseparable so
        concurrent coroutines on the same event loop cannot both pass the check when only one slot remains.
        """
        if self._draining or self.is_at_capacity():
            return False
        self._pending_admissions += 1
        return True

    def admit(self, req: ActiveRequest) -> None:
        """Consume a previously-reserved slot by appending the request to the FIFO.

        Must be paired with a prior successful :meth:`try_admit`. Wakes the scheduler so the next sweep picks up the
        request. If ``req.queue`` is born already terminal (e.g. the only requested checker has every one of its jobs
        SKIPPED at admission for want of a required per-camera input; also covers empty queues), schedules
        :meth:`_maybe_finalize` directly — ``_tick`` only iterates checkers with
        ``READY`` / ``QUEUED`` jobs, so without this admit-path call nothing would ever be enqueued,
        ``_maybe_finalize`` would never run, and ``req.completion_future`` would hang indefinitely.

        Raises:
            WaitingRoomStateError: if no reservation is outstanding. ``admit`` is the half of the contract that
                genuinely should never fire without a paired ``try_admit`` — unlike :meth:`release`, there is no
                asyncio-cancellation race that could legitimately produce an unbalanced call here.
        """
        if self._pending_admissions <= 0:
            raise WaitingRoomStateError("admit() called without a prior try_admit()")
        self._pending_admissions -= 1
        self.queue.append(req)
        self._wakeup.set()
        if req.queue.is_done():
            task = asyncio.create_task(self._maybe_finalize(req))
            self._dispatch_tasks.add(task)
            task.add_done_callback(self._dispatch_tasks.discard)

    def adopt(self, req: ActiveRequest, in_flight: Mapping[str, int]) -> None:
        """Admit a request rebuilt from another replica's rows, and resume watching the jobs it had already published.

        Same reservation contract as :meth:`admit`. ``in_flight`` maps job name to ``checker_executions.execution_id``
        for the jobs a checker pod had claimed before the takeover; registering them hands them to the monitor, which
        either sees them finish or sees their lease go stale and dispatches them again. Without that the scheduler would
        see jobs it has no record of dispatching and start second copies of work already running.

        Each adopted job's wall-clock budget restarts here, because this replica cannot know when the original dispatch
        happened. The consequence is that a job can outlive one budget per adoption, which is the right trade: the
        alternative is killing work that is progressing fine on the strength of a timestamp we do not have.

        ``started_at`` is stamped rather than left unset because everything in ``in_flight`` was already claimed: these
        are running jobs, not waiting ones, so they are budget-eligible from the moment they are adopted. Waiting for a
        later poll to observe the claim would leave an adopted job unbounded for as long as its row stays unreadable,
        which is exactly the case the budget is the last line of defense for.
        """
        now = time.monotonic()
        for name, execution_id in in_flight.items():
            job = req.queue.get_job(name)
            self._durable_jobs[execution_id] = _DurableJob(
                req=req, job=job, execution_id=execution_id, enqueued_at=now, started_at=now
            )
        self.admit(req)

    def free_capacity(self) -> int | None:
        """Admissions this replica could still take, or ``None`` when uncapped. Zero while draining.

        Read before claiming work that has to be claimed to be seen -- an adoption sweep that took more than it can
        hold would leave the surplus claimed by a replica that never runs it, recoverable only once the lease expires
        all over again.
        """
        if self._draining:
            return 0
        if self.capacity is None:
            return None
        return max(0, self.capacity - len(self.queue) - self._pending_admissions)

    def release(self) -> None:
        """Return a reserved slot without admitting.

        Use on any error path between :meth:`try_admit` and :meth:`admit`. Defensively a no-op when no reservation is
        outstanding: an ``asyncio.CancelledError`` (e.g. a client disconnecting mid-request) can fire between
        ``admit()``'s synchronous decrement and the caller's ``reservation_consumed`` flag flip, leaving the
        ``finally`` clause to call ``release()`` after the slot was already consumed.
        """
        if self._pending_admissions <= 0:
            logger.warning("release() called with no outstanding reservations; ignoring")
            return
        self._pending_admissions -= 1

    # ------------------------------------------------------------------
    # Draining
    # ------------------------------------------------------------------

    @property
    def is_draining(self) -> bool:
        """True once :meth:`begin_draining` has been called. There is no way back — the process is on its way out."""
        return self._draining

    def begin_draining(self) -> None:
        """Refuse further admissions while letting everything already admitted run to completion. Idempotent."""
        if self._draining:
            return
        self._draining = True
        logger.info("Draining: no further admissions; %d request(s) still admitted", len(self.queue))

    def has_work_in_flight(self) -> bool:
        """True while this replica still owns something whose loss would strand an execution.

        Four things count, and the last two are the reason this is not just ``bool(self.queue)``: the FIFO itself;
        reservations taken by a submission that has not committed yet; the enqueue tasks the scheduler spawns; and
        finalization, which removes a request from the FIFO *before* writing its terminal row.
        """
        return bool(self.queue) or self._pending_admissions > 0 or bool(self._dispatch_tasks) or self._finalizing > 0

    async def wait_until_drained(self, timeout: float) -> bool:
        """Wait for everything this replica owns to reach a recorded conclusion. Returns whether it got there.

        ``False`` means the budget ran out with work still in flight. Those requests are the ones another replica's
        adoption sweep would recover; without one they are stranded, so the caller should say so loudly.
        """
        deadline = time.monotonic() + timeout
        while self.has_work_in_flight():
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(self.DRAIN_POLL_INTERVAL)
        return True

    def snapshot_metrics(self, known_checkers: Iterable[str] = ()) -> WaitingRoomMetrics:
        """Return a point-in-time :class:`WaitingRoomMetrics` snapshot of this replica.

        Must be called from the same event loop as :meth:`scheduler_loop`. Iteration of ``self.queue`` and each
        request's :class:`CheckerQueue` is safe without locking because all mutations are single-event-loop. The deque
        is snapshotted via ``list(...)`` so the caller does not observe mid-mutation if a dispatch callback removes a
        finished request concurrently.

        ``known_checkers`` seeds the per-checker map so every known checker emits a gauge value for every live state
        in :data:`LIVE_JOB_STATES` (zero when idle), letting downstream Prometheus rate / increase queries avoid gaps.
        Pass the full configured checker set (e.g. ``settings.checker_endpoints``); checkers not in the set are still
        counted if they appear in an admitted request's queue (the per-state map is lazily extended for them as jobs
        are encountered).
        """
        in_progress = 0
        queued = 0
        per_checker_state: dict[str, dict[str, int]] = {
            name: {state.value: 0 for state in LIVE_JOB_STATES} for name in known_checkers
        }
        for req in list(self.queue):
            if req.in_progress_marked:
                in_progress += 1
            else:
                queued += 1
            for job in req.queue.iter_jobs():
                if job.state not in LIVE_JOB_STATES:
                    continue
                checker_states = per_checker_state.setdefault(
                    job.base_name, {state.value: 0 for state in LIVE_JOB_STATES}
                )
                checker_states[job.state.value] += 1
        return WaitingRoomMetrics(
            in_progress=in_progress,
            queued=queued,
            per_checker_state=per_checker_state,
        )

    async def scheduler_loop(self) -> None:
        """Single per-replica scheduler coroutine. Started in lifespan.

        Runs forever. Per iteration: clear the wakeup event, sweep every checker with eligible work, then sleep until
        something wakes it. Nothing here is time-based — a job becomes eligible only when a request is admitted or an
        earlier job settles, and both of those set the wakeup event.
        """
        logger.info("Waiting-room scheduler started")
        while True:
            self._wakeup.clear()
            try:
                self._tick()
            except Exception:
                # A single buggy tick must not kill the entire replica's scheduling. Per-job exceptions are handled
                # inside _enqueue.
                logger.exception("Scheduler tick raised; continuing")
            await self._wakeup.wait()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """One sweep: write out every job whose dependencies are satisfied, oldest request first.

        Everything eligible goes at once. There is no capacity to respect, because writing a row cannot overload
        anything, and no order to impose, because the checker fleet takes work at its own rate and whichever pod is
        free next decides what runs.

        Requests whose CheckerQueue is already terminal at admission (e.g. the only requested checker had every job
        SKIPPED for want of a required per-camera input) are skipped implicitly:
        they have no READY job to find. ``admit()`` schedules ``_maybe_finalize`` directly for those, so their
        ``completion_future`` is resolved without anything ever being enqueued.

        The state flip to QUEUED happens here, synchronously, before the write is even started. That is what stops
        the next sweep from writing the same job twice.
        """
        for req in list(self.queue):
            for job in list(req.queue.iter_jobs()):
                if job.state != JobState.READY:
                    continue
                req.queue.mark_queued(job.name)
                self._spawn(self._enqueue(req, job))

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        """Run an enqueue coroutine detached, keeping a strong reference until it finishes."""
        task = asyncio.create_task(coro)
        self._dispatch_tasks.add(task)
        task.add_done_callback(self._dispatch_tasks.discard)

    async def _force_fail(self, req: ActiveRequest, job: CheckerJob) -> None:
        """Last-resort terminalization for a job the scheduler itself failed to handle.

        Deliberately does not touch the database or the skipped cascade: this runs when the code that normally does
        those things has already thrown, and a request that finalizes with an error beats one that never finalizes.
        """
        try:
            if job.state in (JobState.QUEUED, JobState.IN_PROGRESS):
                req.queue.mark_failed(job.name, "Internal scheduler error")
            req.aggregated_results.setdefault(job.name, {"error": "Internal scheduler error"})
        except Exception as exc:
            logger.exception("Failed to terminalize job %s after scheduler error", job.name)
            self._abandon(req, exc)

    def _abandon(self, req: ActiveRequest, exc: Exception) -> None:
        """Give up on a request whose queue could not be made terminal, instead of leaving it waiting forever.

        One rung below :meth:`_force_fail`, for when even that could not make the queue terminal. Finalization only
        happens for a queue that reports itself done, so a queue that refuses to move has no other way out: the request
        would hold its waiting-room slot, its caller would wait on a future nothing will ever resolve, and a drain would
        spend its whole budget on it. An error is a worse answer than a result and a better one than no answer at all.
        """
        if not req.completion_future.done():
            req.completion_future.set_exception(exc)
        try:
            self.queue.remove(req)
        except ValueError:
            # Already gone; the point was only that it must not still be here.
            pass
        for execution_id, entry in list(self._durable_jobs.items()):
            if entry.req is req:
                self._durable_jobs.pop(execution_id, None)

    async def _enqueue(self, req: ActiveRequest, job: CheckerJob) -> None:
        """Write one job to the table a checker fleet claims from.

        The job's payload landing on the row *is* the handover, so a payload that could not be written is terminal
        rather than something to retry: leaving the row queued would hand a pod a job it cannot run, and rewriting it
        later is exactly what the next sweep would have to be told not to do.

        A request that :meth:`_abandon` has already failed is left alone. One sweep spawns every eligible job of a
        request at once, so a sibling abandoning the request can land before this task first runs or while it is
        mid-write; going ahead anyway would queue work for a caller that has already been handed its exception.
        """
        if req.completion_future.done():
            logger.info("Not enqueueing %s: its request is already resolved", job.name)
            return
        record = req.checker_executions.get(job.name)
        try:
            request_json = await self._build_payload(req, job, record)
            if request_json is None:
                return
            if record is None or record.execution_id is None:
                await self._fail_job(req, job, f"No checker_executions row for '{job.name}' to enqueue", None)
                return
            if not await record.enqueue(request_json):
                await self._fail_job(req, job, f"Could not record the payload for '{job.name}'", record)
                return
            if req.completion_future.done():
                # Abandoned while the payload was being written. ``_abandon`` has already dropped this request's
                # durable jobs, so registering one now would leave an entry the monitor polls until its budget
                # expires, on behalf of a request nobody is waiting for.
                logger.warning(
                    "Job %s reached checker execution %s after its request was abandoned; not tracking it",
                    job.name,
                    record.execution_id,
                )
                return
            # Registered before anything else can observe the row, so a job claimed and finished between this write
            # and the next poll is still one the monitor knows to look for.
            self._durable_jobs[record.execution_id] = _DurableJob(
                req=req, job=job, execution_id=record.execution_id, enqueued_at=time.monotonic()
            )
            logger.info("Job %s is queued as checker execution %s", job.name, record.execution_id)
        except Exception:
            logger.exception("Unexpected error enqueueing %s; force-failing job", job.name)
            await self._force_fail(req, job)
        finally:
            self._wakeup.set()
            await self._maybe_finalize(req)

    async def _build_payload(
        self, req: ActiveRequest, job: CheckerJob, record: CheckerExecutionRecord | None
    ) -> dict[str, Any] | None:
        """Build a job's payload, or terminalize the job and return ``None``.

        Two things can go wrong before a checker is ever involved: there is no implementation behind the checker, or
        the payload cannot be assembled from what the job's dependencies produced.
        """
        try:
            return self._build_request(job, req.request, req.queue)
        except NotImplementedError:
            await self._complete_unimplemented(req, job, record)
            return None
        except Exception as exc:
            logger.exception("Failed to construct request for %s", job.name)
            await self._fail_job(req, job, str(exc), record)
            return None

    async def _complete_unimplemented(
        self, req: ActiveRequest, job: CheckerJob, record: CheckerExecutionRecord | None
    ) -> None:
        """Close out a job whose checker has no implementation yet, as a success carrying a placeholder result."""
        logger.warning("Checker %s not yet implemented", job.name)
        req.queue.mark_in_progress(job.name)
        req.queue.mark_completed(job.name, "Not yet implemented")
        self._store_result(req.aggregated_results, job, "Not yet implemented")
        if record is not None:
            try:
                if await record.complete(response_json={"data": "Not yet implemented", "metadata": {}}, duration_ms=0):
                    req.db_terminalized.add(job.name)
            except Exception:
                logger.exception("Failed to mark checker_execution %s complete (synthetic)", job.name)
        await self._mark_in_progress_if_needed(req)

    # ------------------------------------------------------------------
    # Durable job monitoring
    # ------------------------------------------------------------------

    async def job_monitor_loop(self) -> None:
        """Single per-replica coroutine that learns how queued jobs turned out. Started in lifespan.

        The counterpart to :meth:`scheduler_loop`: that one decides what to start, this one finds out what finished.
        They are separate coroutines because a poll must not be able to stall a dispatch, or vice versa.

        Each pass is one batched read covering every job this replica is waiting on, so the query rate tracks the
        number of replicas rather than the fan-out width of the requests they are running. A pass that fails is simply
        logged: whatever it missed is still on the row for the next one to read.
        """
        if self._read_job_states is None:
            logger.info("Durable job monitor not started: no job-state reader is configured")
            return
        logger.info("Durable job monitor started")
        while True:
            await asyncio.sleep(self._poll_delay())
            try:
                await self.poll_durable_jobs()
            except Exception:
                logger.exception("Durable job poll raised; continuing")
            try:
                await self.reap_expired_jobs()
            except Exception:
                logger.exception("Expired-job reap raised; continuing")

    async def reap_expired_jobs(self) -> None:
        """Return jobs whose owner stopped renewing to the queue, so a live pod can pick them up.

        Runs from every replica on the monitor's jittered interval and needs no leader, because the requeue is a single
        conditional update: whichever replica gets there first moves the row and the others match nothing. It covers
        jobs this replica knows nothing about, which is the point -- a request whose arbitrator replica also died would
        otherwise leave its jobs owned by a dead pod forever.
        """
        if self._requeue_expired_jobs is None:
            return
        requeued = await self._requeue_expired_jobs()
        for row in requeued:
            logger.warning(
                "Requeued checker execution %s (%s) after its owner stopped renewing its lease",
                getattr(row, "execution_id", None),
                getattr(row, "checker_name", None),
            )

    async def poll_durable_jobs(self) -> None:
        """Read the rows of every in-flight durable job once and act on what they say.

        One read and one write per pass, both batched, regardless of how many jobs this replica is watching. The
        write comes first and covers the whole pass: see :meth:`_record_observed_claims` for why acting row-by-row
        does not survive a wide fan-out.

        Public rather than private because it is the monitor's whole behavior: tests drive it directly instead of
        waiting out a sleep, and an operator-facing endpoint could reasonably force a pass.
        """
        if self._read_job_states is None or not self._durable_jobs:
            return

        rows = await self._read_job_states(list(self._durable_jobs))
        actions = self._classify_job_rows(rows)
        refused = await self._record_observed_claims(actions)
        for row, action in actions:
            await self._apply_job_row(row, action, refused)

        # Jobs whose row did not come back are deliberately left alone. An unreachable database reads as zero rows, and
        # failing every in-flight job over a transient outage would be far worse than waiting for the next pass; a row
        # that never reappears is caught by the budget below instead.
        for entry in list(self._durable_jobs.values()):
            await self._enforce_job_budget(entry)

    def _classify_job_rows(
        self, rows: Iterable[CheckerExecutionStateRow]
    ) -> list[tuple[CheckerExecutionStateRow, _RowAction]]:
        """Decide what each polled row needs, before anything is acted on.

        Rows this replica no longer tracks are dropped here rather than carried through as a no-op action: a job that
        left ``_durable_jobs`` between the read and now has already been settled by whoever removed it.
        """
        classified = []
        for row in rows:
            entry = self._durable_jobs.get(row.execution_id)
            if entry is not None:
                classified.append((row, self._classify_job_row(entry, row)))
        return classified

    def _classify_job_row(self, entry: _DurableJob, row: CheckerExecutionStateRow) -> _RowAction:
        """Read one row against what this replica last believed about the job."""
        status = (row.status or "").lower()
        if status == "success":
            return _RowAction.COMPLETED
        if status in CHECKER_TERMINAL_STATUSES:
            return _RowAction.FAILED
        if row.claimed_by:
            # A claim already applied needs nothing: the write refining ``start_time`` happened on the pass that
            # first saw it, and repeating it every pass would keep moving the start time forward under a running job.
            return _RowAction.NOTHING if entry.in_progress else _RowAction.CLAIM_OBSERVED
        if entry.in_progress:
            # Claimed a moment ago and unclaimed now: the reaper handed the job back to the queue because its owner
            # stopped renewing. Nothing to dispatch — the row being queued again *is* the redelivery.
            return _RowAction.REQUEUED
        return _RowAction.NOTHING

    async def _record_observed_claims(
        self, actions: Iterable[tuple[CheckerExecutionStateRow, _RowAction]]
    ) -> Mapping[int, str | None]:
        """Write back every claim this pass is seeing for the first time, in one call.

        Batched for the same reason the read is, and it matters more here: every claim in the batch was read from a
        snapshot taken at the top of the pass, so the gap between reading a claim and recording it is what decides
        whether the job can have finished in between. One round trip holds that gap to the length of a single call
        however wide the fan-out, where a write per job stretches it across the whole walk -- and a claim recorded
        minutes after it was read is exactly the write the database-side guard exists to refuse.

        Returns the refused ids mapped to the status that won. A failed write returns no refusals, which reads as
        every claim accepted: an unreachable database leaves the jobs running, as they were, and the next pass
        retries. Only a *known* terminal status stops a job from being treated as live.
        """
        if self._record_claims is None:
            return {}
        observed = [row.execution_id for row, action in actions if action is _RowAction.CLAIM_OBSERVED]
        if not observed:
            return {}
        try:
            return await self._record_claims(observed, datetime.now(UTC))
        except Exception:
            logger.exception("Failed to record %d observed claim(s); leaving them for the next pass", len(observed))
            return {}

    def _poll_delay(self) -> float:
        """Seconds until the next poll, jittered so replicas do not converge on the same instant."""
        spread = self.JOB_POLL_INTERVAL * self.JOB_POLL_JITTER
        return self.JOB_POLL_INTERVAL + random.uniform(-spread, spread)

    async def _apply_job_row(
        self, row: CheckerExecutionStateRow, action: _RowAction, refused: Mapping[int, str | None]
    ) -> None:
        """Reconcile one job's in-memory state with what its row now says.

        The row was classified before this pass wrote anything (:meth:`_classify_job_rows`), so a job settled by an
        earlier row in the same pass may have left ``_durable_jobs`` since. Re-reading it here rather than trusting
        the classification's entry is what keeps that from being acted on twice.
        """
        entry = self._durable_jobs.get(row.execution_id)
        if entry is None:
            return

        if action is _RowAction.COMPLETED:
            await self._complete_durable_job(entry, row)
            return
        if action is _RowAction.FAILED:
            # The pod wrote its own diagnosis; passing ``record=None`` keeps the arbitrator from overwriting it with a
            # vaguer one.
            self._durable_jobs.pop(entry.execution_id, None)
            status = (row.status or "").lower()
            await self._fail_job(
                entry.req, entry.job, row.error_message or f"Checker reported status '{status}'", record=None
            )
            await self._after_job_settled(entry.req)
            return
        if action is _RowAction.CLAIM_OBSERVED:
            await self._note_job_in_progress(entry, row, refused)
            return
        if action is _RowAction.REQUEUED:
            await self._handle_requeued_job(entry, row)

    async def _note_job_in_progress(
        self, entry: _DurableJob, row: CheckerExecutionStateRow, refused: Mapping[int, str | None]
    ) -> None:
        """Advance a job whose claim this pass observed and recorded.

        Acceptance is observed rather than reported: the arbitrator never hears from the pod that took the work, so the
        claim on the row is the whole signal. It matters because it is what refines the job's start time from "when the
        arbitrator queued it" to "when something actually began working on it", and what flips the parent execution to
        ``in_progress``.

        The claim was read at the top of the pass, so the job may already be over by the time the write lands -- a
        short job can be claimed and finished inside one interval. The row settles it: a claim the batch could not
        record because the row had gone terminal leaves the job exactly as it is, for the next pass to read as
        finished, rather than starting it on a state that has already been superseded.
        """
        if entry.in_progress:
            return
        logger.info("Checker execution %s (%s) claimed by %s", row.execution_id, entry.job.name, row.claimed_by)
        if entry.execution_id in refused:
            logger.info(
                "Checker execution %s (%s) finished as '%s' before its claim was recorded; leaving the row alone",
                entry.execution_id,
                entry.job.name,
                refused[entry.execution_id],
            )
            return
        await self._mark_in_progress_locally(entry)

    async def _mark_in_progress_locally(self, entry: _DurableJob) -> None:
        """Advance this replica's own view of a job to IN_PROGRESS, without writing to the row.

        Separate from :meth:`_note_job_in_progress` because a job that is already terminal must not have ``in_progress``
        written back over its recorded outcome, and a short job can reach a poll having been claimed *and* finished
        since the last one.
        """
        if entry.in_progress:
            return
        entry.in_progress = True
        # Start the budget clock here rather than at enqueue: this is the first moment the job is someone's
        # responsibility, and the only window in which it can hang.
        entry.started_at = time.monotonic()
        entry.req.queue.mark_in_progress(entry.job.name)
        await self._mark_in_progress_if_needed(entry.req)

    async def _handle_requeued_job(self, entry: _DurableJob, row: CheckerExecutionStateRow) -> None:
        """React to a job the reaper put back on the queue: let it go round again, or stop it if it has had enough goes.

        The job stays outstanding, because from this replica's point of view nothing about it has finished -- its result
        will still arrive on its row, just from a different pod. What does change is that nobody is running it, and
        only the attempt count can end the cycle: work that reliably kills whichever pod picks it up has to fail loudly
        rather than tour the fleet forever.
        """
        attempts = row.attempt or 0
        if attempts >= self.JOB_MAX_ATTEMPTS:
            self._durable_jobs.pop(entry.execution_id, None)
            await self._fail_job(
                entry.req,
                entry.job,
                f"Checker stopped responding on {attempts} attempts",
                entry.req.checker_executions.get(entry.job.name),
            )
            await self._after_job_settled(entry.req)
            return

        entry.in_progress = False
        # Nobody holds the job now, so it is back to waiting and must not be charged for the attempt that lapsed or for
        # the wait until the next pod takes it. The next claim re-stamps this.
        entry.started_at = None
        # Follow the row back to QUEUED. Left IN_PROGRESS, the job would be counted as running on ``/metrics`` when no
        # pod holds it, and the next claim's ``mark_in_progress`` would be a no-op that never refines its start time.
        # Restored rather than transitioned: QUEUED is a state this job reached on its row, not one this replica moved
        # it through, and the row's attempt count is the authoritative one.
        entry.req.queue.restore(entry.job.name, JobState.QUEUED, attempt=attempts)
        logger.warning(
            "Checker execution %s (%s) was requeued after %s attempt(s); waiting for another pod to claim it",
            row.execution_id,
            entry.job.name,
            attempts,
        )

    async def _complete_durable_job(self, entry: _DurableJob, row: CheckerExecutionStateRow) -> None:
        """Adopt a finished job's result off its row. The row is already terminal, so nothing is written back."""
        envelope = row.response_json or {}
        data = envelope.get("data", {}) if isinstance(envelope, dict) else {}

        self._durable_jobs.pop(entry.execution_id, None)
        # A short job can be claimed and finished entirely between two polls, so its claim may never have been
        # observed on its own. Applying it here keeps the job's lifecycle the same either way.
        await self._mark_in_progress_locally(entry)
        entry.req.queue.mark_completed(entry.job.name, data)
        self._store_result(entry.req.aggregated_results, entry.job, data)
        entry.req.db_terminalized.add(entry.job.name)
        logger.info("Checker execution %s (%s) completed", row.execution_id, entry.job.name)
        await self._after_job_settled(entry.req)

    async def _enforce_job_budget(self, entry: _DurableJob) -> None:
        """Stop a job that has outlived its wall-clock ceiling.

        A pod that heartbeats forever holds its lease forever, so a job that hangs after starting is invisible to the
        stale-lease path. This is the backstop: ask the owner to stop, then close the job out locally rather than let
        one wedged clip hold a request open indefinitely. A pod that does eventually notice the cancellation writes its
        own terminal state over this one, which is a more accurate record of what happened and changes nothing here.

        Only a job some pod has claimed is eligible. One still waiting has nothing that could hang, and failing it for
        sitting in a backlog would kill work that was never given a chance to run.
        """
        if entry.started_at is None:
            return

        if time.monotonic() - entry.started_at < self.JOB_BUDGET_SECONDS:
            return

        if not entry.cancel_sent and self._request_job_cancel is not None:
            entry.cancel_sent = True
            try:
                await self._request_job_cancel(entry.execution_id)
            except Exception:
                logger.exception("Failed to request cancellation of checker execution %s", entry.execution_id)

        self._durable_jobs.pop(entry.execution_id, None)
        logger.error(
            "Checker execution %s (%s) exceeded its %.0fs budget; failing it",
            entry.execution_id,
            entry.job.name,
            self.JOB_BUDGET_SECONDS,
        )
        await self._fail_job(
            entry.req,
            entry.job,
            f"Checker exceeded its {self.JOB_BUDGET_SECONDS:.0f}s budget",
            entry.req.checker_executions.get(entry.job.name),
            timed_out=True,
        )
        await self._after_job_settled(entry.req)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def _fail_job(
        self,
        req: ActiveRequest,
        job: CheckerJob,
        error: str,
        record: CheckerExecutionRecord | None,
        *,
        timed_out: bool = False,
    ) -> None:
        """Terminalize one job as failed: queue state, aggregated result, row and dependents.

        Pass ``record=None`` when the row already carries a terminal state of its own. A checker that failed its own
        job wrote a better error than the arbitrator can reconstruct from the outside, and overwriting it loses it.
        """
        req.queue.mark_failed(job.name, error)
        self._store_result(req.aggregated_results, job, {"error": error})
        if record is None:
            # The row already carries a terminal state written by whoever knew better than us.
            req.db_terminalized.add(job.name)
        else:
            try:
                written = (
                    await record.timeout(error_message=error)
                    if timed_out
                    else await record.fail(response_json={"error": error}, duration_ms=0, error_message=error)
                )
            except Exception:
                logger.exception("Failed to mark checker_execution %s terminal", job.name)
                written = False
            if written:
                req.db_terminalized.add(job.name)
        await self._apply_skipped_cascade(req)

    async def _after_job_settled(self, req: ActiveRequest) -> None:
        """Wake the scheduler and finalize the request if that job was the last one outstanding.

        The enqueue path gets this for free from ``_enqueue``'s ``finally``; the monitor has no such wrapper, so
        every path that settles a job from a poll has to come through here.
        """
        self._wakeup.set()
        await self._maybe_finalize(req)

    async def _mark_in_progress_if_needed(self, req: ActiveRequest) -> None:
        """Flip parent ``arbitrator_executions.status`` to ``in_progress`` once.

        Fires the first time a job of this request is seen being worked on: a pod's claim observed on the row, or the
        synthetic ``Not yet implemented`` completion.
        """
        if req.in_progress_marked:
            return
        req.in_progress_marked = True
        try:
            await req.execution.in_progress()
        except Exception:
            logger.exception("Failed to flip arbitrator_executions to in_progress")

    async def _apply_skipped_cascade(self, req: ActiveRequest) -> None:
        """After a fail, update DB rows for any newly-SKIPPED jobs."""
        for name, info in req.queue.get_results().items():
            if info["state"] != "skipped":
                continue
            if name in req.db_terminalized:
                continue
            record = req.checker_executions.get(name)
            if record is None:
                continue
            error_msg = info.get("error") or "Dependency failed"
            try:
                if await record.skipped(error_message=str(error_msg)):
                    req.db_terminalized.add(name)
            except Exception:
                logger.exception("Failed to mark checker_execution %s skipped", name)

    async def _maybe_finalize(self, req: ActiveRequest) -> None:
        """If the request is now terminal, run finalize and remove it from the FIFO."""
        if not req.queue.is_done():
            return
        if req.completion_future.done():
            return
        # Make sure any SKIPPED-on-cascade DB updates have been applied.
        await self._apply_skipped_cascade(req)
        # Nothing should still be in flight for a request whose every job is terminal, but a leaked entry would keep
        # this request's objects alive and get polled forever, so drop any that survived.
        for execution_id, entry in list(self._durable_jobs.items()):
            if entry.req is req:
                self._durable_jobs.pop(execution_id, None)
        try:
            self.queue.remove(req)
        except ValueError:
            # Already removed; safe to continue.
            pass
        # Counted from here rather than from the top of the method: past the removal above the request is invisible to
        # ``self.queue``, and a drain that stopped waiting now would abandon the very write that records the outcome.
        self._finalizing += 1
        try:
            result = await req.finalize(req)
        except Exception as exc:
            logger.exception("finalize() raised for request_id=%s; setting future exception", _request_id(req))
            if not req.completion_future.done():
                req.completion_future.set_exception(exc)
            return
        finally:
            self._finalizing -= 1
        if not req.completion_future.done():
            req.completion_future.set_result(result)


def _request_id(req: ActiveRequest) -> Any:
    """Best-effort request_id extraction for log messages."""
    try:
        return req.execution.request_id
    except AttributeError:
        return None
