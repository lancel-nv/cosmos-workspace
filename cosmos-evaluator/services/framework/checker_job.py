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

"""Running a claimed checker job to a recorded conclusion.

A checker pod that has claimed a job owes the row three things for as long as it holds it: a heartbeat often enough
that nobody concludes the pod died, attention to whether a cancellation has been requested, and exactly one terminal
write at the end. This module is that obligation, expressed once.

Splitting it out of the transport keeps it testable without a FastAPI app, and keeps the interesting failure modes --
losing the claim mid-flight, cancellation arriving between heartbeats, the pipeline raising after the lease expired --
in one place with one set of tests rather than spread across the request handler of every checker.
"""

import asyncio
from collections.abc import Awaitable, Callable, Coroutine
import dataclasses
import logging
import socket
import threading
import time
from typing import Any, ClassVar, Protocol
import uuid

from database.api_client import JobOwnershipLost
from database.api_models import CheckerJobState
from utils.cancellation import ProcessingCancelled

logger = logging.getLogger(__name__)


class JobHandle(Protocol):
    """The writes a job owner is allowed to make, as this module uses them.

    Structural rather than concrete so the runner's decisions can be tested without an HTTP client behind them.
    ``database.api_client.CheckerJobHandle`` is the implementation; it is also where the fencing that makes these
    writes safe actually lives.
    """

    @property
    def execution_id(self) -> int: ...

    async def heartbeat(self, lease_seconds: float) -> CheckerJobState: ...

    async def complete(
        self,
        response_json: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        output_metadata: dict[str, Any] | None = None,
    ) -> None: ...

    async def fail(
        self,
        error_message: str,
        duration_ms: int | None = None,
        response_json: dict[str, Any] | None = None,
        output_metadata: dict[str, Any] | None = None,
        timed_out: bool = False,
    ) -> None: ...


@dataclasses.dataclass(frozen=True)
class LeasePolicy:
    """How long a claim is good for and how its owner keeps it alive.

    One object rather than four loose floats because the values are a single decision that only makes sense together:
    the lease has to outlast however long the heartbeat path can take to notice it is failing and try again. Stated as
    separate arguments that relationship is invisible, and it is easy to pick a lease that a slow database can run out
    on its own -- a live pod losing work it is in the middle of. :meth:`tolerated_failures` names the margin and
    ``__post_init__`` enforces it, so the invariant fails at construction rather than under load.

    Attributes:
        lease_seconds: How far ahead each heartbeat pushes ``lease_expires``.
        heartbeat_interval: Seconds between the end of one heartbeat and the start of the next.
        heartbeat_timeout: Ceiling on a single heartbeat call. Bounded independently of the HTTP client's own deadline
            and retries: without a ceiling here, the time a heartbeat spends failing is charged against the very lease
            it exists to renew, and the client's per-call budget (its read timeout multiplied by its retry count) can
            exceed the whole lease.
        retry_delay: How long to wait after a failed heartbeat before trying again. Much shorter than
            ``heartbeat_interval`` -- a beat that failed is the moment to hurry, not to idle, since the lease is now
            closer to expiring than a successful beat would have left it.
    """

    # The per-beat ceiling is wider than a healthy beat needs: the Database REST API can take this long to answer when
    # Postgres is slow to hand out connections, and a beat that succeeds late is worth more than one abandoned early.
    lease_seconds: float = 150.0
    heartbeat_interval: float = 30.0
    heartbeat_timeout: float = 15.0
    retry_delay: float = 5.0

    # Consecutive failed heartbeats a lease must survive for the policy to be considered safe. Two would mean a single
    # unlucky pair of blips costs a pod its job; more than a handful would leave a genuinely dead pod's work stranded
    # for much longer than the lease itself.
    MINIMUM_TOLERATED_FAILURES: ClassVar[int] = 3

    def __post_init__(self) -> None:
        for name in ("lease_seconds", "heartbeat_interval", "heartbeat_timeout", "retry_delay"):
            value = getattr(self, name)
            if not isinstance(value, int | float) or value <= 0:
                raise ValueError(f"{name} must be a positive number, got {value!r}")
        if self.heartbeat_timeout >= self.heartbeat_interval:
            raise ValueError(
                f"heartbeat_timeout ({self.heartbeat_timeout}) must be shorter than heartbeat_interval "
                f"({self.heartbeat_interval}), or a slow beat alone consumes the gap between beats"
            )
        if self.tolerated_failures < self.MINIMUM_TOLERATED_FAILURES:
            raise ValueError(
                f"lease_seconds ({self.lease_seconds}) leaves room for only {self.tolerated_failures} failed "
                f"heartbeat(s); at least {self.MINIMUM_TOLERATED_FAILURES} are required"
            )

    @property
    def tolerated_failures(self) -> int:
        """Consecutive failed heartbeats this policy survives without the lease expiring.

        Worst case after a successful beat: a full interval of silence, then each failed attempt costing its timeout
        plus the delay before the next one. The lease has to cover all of it.
        """
        budget = self.lease_seconds - self.heartbeat_interval
        return max(0, int(budget // (self.heartbeat_timeout + self.retry_delay)))


def worker_identity() -> str:
    """Mint a fencing token for one claim.

    Unique per claim rather than per pod: a pod whose lease expired can legitimately claim the same job again later,
    and reusing an identity would make the revived claim indistinguishable from the stale one it is meant to fence
    out. The hostname is only there to make the token readable in the row.
    """
    return f"{socket.gethostname()}/{uuid.uuid4().hex}"


async def run_claimed_job(
    handle: JobHandle,
    pipeline: Callable[[], Coroutine[Any, Any, dict[str, Any]]],
    cancel_event: threading.Event,
    cleanup: Callable[[], Coroutine[Any, Any, None]],
    policy: LeasePolicy,
) -> None:
    """Run one claimed job, heartbeating until it ends and recording its outcome on the row.

    Never raises: this runs as a detached task with nobody waiting on it, so an escaping exception would be reported
    only as an "exception was never retrieved" warning at garbage-collection time. Every outcome is either written to
    the row or logged here.

    Args:
        handle: Ownership of the job. Supplies the fencing token for every write.
        pipeline: The actual work. Its return value becomes the row's ``response_json``.
        cancel_event: Cooperative stop flag the pipeline's executor thread polls. Set when the row asks for
            cancellation, and when the claim is lost.
        cleanup: Releases the job's working files. Runs after the pipeline has genuinely stopped, in every outcome.
        policy: Lease duration and the heartbeat schedule that keeps it alive.
    """
    started = time.perf_counter()
    work = asyncio.create_task(pipeline())
    owned = True

    try:
        # Scheduled against a deadline rather than by sleeping between calls, so that however long a beat takes does not
        # push the following one further out. Sleeping a fixed interval after each call makes the effective period
        # "interval plus however slow the database is", which is exactly the drift that lets a lease lapse under load.
        next_beat = time.monotonic() + policy.heartbeat_interval
        while True:
            done, _ = await asyncio.wait({work}, timeout=max(0.0, next_beat - time.monotonic()))
            if done:
                break
            try:
                state = await asyncio.wait_for(handle.heartbeat(policy.lease_seconds), policy.heartbeat_timeout)
            except JobOwnershipLost:
                # Somebody else already owns this job. Stop the work and write nothing: the replacement's result is
                # now the truth, and a terminal write from here would overwrite it.
                logger.warning("Job %s was taken over by another pod; abandoning it", handle.execution_id)
                owned = False
                cancel_event.set()
                break
            except Exception:
                # A transient database blip should not kill a job that is otherwise fine, so retry rather than give up.
                # Retried after a short delay instead of a full interval: the lease is closer to expiring than a
                # successful beat would have left it, so this is the moment to hurry. ``policy`` guarantees the lease
                # survives at least ``MINIMUM_TOLERATED_FAILURES`` of these; past that the takeover path above is what
                # notices, which is correct -- a pod that cannot reach the database cannot report a result either.
                logger.warning(
                    "Heartbeat for job %s failed; retrying in %.1fs",
                    handle.execution_id,
                    policy.retry_delay,
                    exc_info=True,
                )
                next_beat = time.monotonic() + policy.retry_delay
                continue
            next_beat = time.monotonic() + policy.heartbeat_interval

            if state.cancel_requested:
                logger.info("Job %s was asked to stop", handle.execution_id)
                cancel_event.set()

        # The pipeline runs in an executor thread that task cancellation cannot interrupt, so wait for it to actually
        # finish before cleaning up. Otherwise cleanup deletes files out from under live work.
        outcome = await asyncio.gather(work, return_exceptions=True)
        error = outcome[0] if isinstance(outcome[0], BaseException) else None
        duration_ms = int((time.perf_counter() - started) * 1000)

        if not owned:
            return
        if error is None:
            response_json = outcome[0]
            assert isinstance(response_json, dict)
            await _record(handle, handle.complete(response_json=response_json, duration_ms=duration_ms))
        elif isinstance(error, ProcessingCancelled | asyncio.CancelledError):
            await _record(handle, handle.fail("Cancelled before completion", duration_ms=duration_ms))
        else:
            logger.error("Job %s failed", handle.execution_id, exc_info=error)
            await _record(handle, handle.fail(str(error) or type(error).__name__, duration_ms=duration_ms))
    except Exception:
        logger.exception("Unexpected error running job %s", handle.execution_id)
    finally:
        try:
            await cleanup()
        except Exception:
            logger.warning("Cleanup for job %s failed", handle.execution_id, exc_info=True)


async def _record(handle: JobHandle, write: Awaitable[None]) -> None:
    """Perform a terminal write, tolerating the two ways it can legitimately fail.

    Losing the claim means the outcome is no longer ours to report. Anything else means the result exists on disk but
    could not be recorded, which the arbitrator will eventually see as an expired lease and re-dispatch.
    """
    try:
        await write
    except JobOwnershipLost:
        logger.warning("Job %s finished but had already been taken over; result discarded", handle.execution_id)
    except Exception:
        logger.exception("Could not record the outcome of job %s", handle.execution_id)
