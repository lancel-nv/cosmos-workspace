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

"""Tests for running a claimed checker job to a recorded conclusion."""

import asyncio
import sys
import threading
from typing import Any

import pytest

from database.api_client import JobOwnershipLost
from database.api_models import CheckerJobState
from services.framework.checker_job import LeasePolicy, run_claimed_job, worker_identity
from utils.cancellation import ProcessingCancelled

# The production schedule compressed so these tests do not spend real seconds waiting for beats. Every relationship
# ``LeasePolicy`` enforces still holds, so it exercises the same code paths a real policy does.
_FAST = LeasePolicy(lease_seconds=120.0, heartbeat_interval=0.01, heartbeat_timeout=0.005, retry_delay=0.005)

# Ceiling on every wait below that only a working heartbeat loop can end -- whether waiting on the beats directly or on
# a cancellation the beats are what deliver. Orders of magnitude above the intervals in ``_FAST``, so it expires only
# when beats have genuinely stopped rather than when the machine is loaded. Bounded at all because a regression in that
# loop is precisely what makes these conditions never come true, and an unbounded wait turns that into the whole target
# hanging until Bazel kills it, with nothing said about which invariant broke.
_STALL_TIMEOUT = 5.0


class FakeHandle:
    """A job handle that records what the runner wrote to it.

    Stands in for ``CheckerJobHandle`` so these tests exercise the runner's decisions rather than the HTTP client's;
    the fencing itself is the handle's own responsibility and is tested against the client.
    """

    def __init__(self, execution_id: int = 7, cancel_after: int | None = None, lose_after: int | None = None) -> None:
        self.execution_id = execution_id
        self.claimed_by = "pod-a/abc"
        self.attempt = 1
        self.heartbeats = 0
        self.completed: list[dict[str, Any] | None] = []
        self.failures: list[str] = []
        self._cancel_after = cancel_after
        self._lose_after = lose_after

    async def heartbeat(self, lease_seconds: float) -> CheckerJobState:
        self.heartbeats += 1
        if self._lose_after is not None and self.heartbeats >= self._lose_after:
            raise JobOwnershipLost("taken over")
        cancel = self._cancel_after is not None and self.heartbeats >= self._cancel_after
        return CheckerJobState(execution_id=self.execution_id, cancel_requested=cancel)

    async def complete(
        self,
        response_json: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        output_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.completed.append(response_json)

    async def fail(
        self,
        error_message: str,
        duration_ms: int | None = None,
        response_json: dict[str, Any] | None = None,
        output_metadata: dict[str, Any] | None = None,
        timed_out: bool = False,
    ) -> None:
        self.failures.append(error_message)


async def _wait_for_heartbeats(handle: FakeHandle, count: int, timeout: float = _STALL_TIMEOUT) -> None:
    """Wait until ``handle`` has taken at least ``count`` beats, reporting the count reached if they stop first."""

    async def _poll() -> None:
        while handle.heartbeats < count:
            await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(_poll(), timeout)
    except TimeoutError:
        pytest.fail(f"heartbeats stalled at {handle.heartbeats}; expected at least {count} within {timeout}s")


def test_worker_identity_is_unique_per_claim() -> None:
    """Reusing an identity would make a revived stale claim indistinguishable from the live one meant to fence it."""
    assert worker_identity() != worker_identity()


@pytest.mark.asyncio
async def test_success_is_recorded_with_the_pipeline_result() -> None:
    handle = FakeHandle()
    cleaned = asyncio.Event()

    async def pipeline() -> dict[str, Any]:
        return {"score": 1}

    async def cleanup() -> None:
        cleaned.set()

    await run_claimed_job(handle, pipeline, threading.Event(), cleanup, _FAST)

    assert handle.completed == [{"score": 1}]
    assert handle.failures == []
    assert cleaned.is_set()


@pytest.mark.asyncio
async def test_failure_is_recorded_with_its_message() -> None:
    handle = FakeHandle()

    async def pipeline() -> dict[str, Any]:
        raise RuntimeError("segformer exploded")

    async def cleanup() -> None:
        return None

    await run_claimed_job(handle, pipeline, threading.Event(), cleanup, _FAST)

    assert handle.completed == []
    assert handle.failures == ["segformer exploded"]


@pytest.mark.asyncio
async def test_cleanup_runs_even_when_recording_the_outcome_fails() -> None:
    """The working files are this pod's to delete regardless of whether anybody accepted its answer."""

    class LostHandle(FakeHandle):
        async def complete(self, *args: Any, **kwargs: Any) -> None:
            raise JobOwnershipLost("taken over")

    handle = LostHandle()
    cleaned = asyncio.Event()

    async def pipeline() -> dict[str, Any]:
        return {}

    async def cleanup() -> None:
        cleaned.set()

    await run_claimed_job(handle, pipeline, threading.Event(), cleanup, _FAST)

    assert cleaned.is_set()


@pytest.mark.asyncio
async def test_the_lease_is_extended_while_the_work_runs() -> None:
    handle = FakeHandle()
    release = asyncio.Event()

    async def pipeline() -> dict[str, Any]:
        await release.wait()
        return {}

    async def cleanup() -> None:
        return None

    job = asyncio.create_task(run_claimed_job(handle, pipeline, threading.Event(), cleanup, _FAST))
    await _wait_for_heartbeats(handle, 2)
    release.set()
    await job

    assert handle.heartbeats >= 2
    assert handle.completed == [{}]


@pytest.mark.asyncio
async def test_a_cancel_request_reaches_the_worker_thread() -> None:
    """Cancellation arrives on the heartbeat rather than through an endpoint, because a request would hit any pod."""
    handle = FakeHandle(cancel_after=1)
    cancel_event = threading.Event()

    async def pipeline() -> dict[str, Any]:
        while not cancel_event.is_set():
            await asyncio.sleep(0.01)
        raise ProcessingCancelled("stopped")

    async def cleanup() -> None:
        return None

    await asyncio.wait_for(run_claimed_job(handle, pipeline, cancel_event, cleanup, _FAST), _STALL_TIMEOUT)

    assert cancel_event.is_set()
    assert handle.failures == ["Cancelled before completion"]


@pytest.mark.asyncio
async def test_a_superseded_pod_stops_and_writes_nothing() -> None:
    """The pod that took over owns the answer now; a terminal write from here would overwrite it."""
    handle = FakeHandle(lose_after=1)
    cancel_event = threading.Event()

    async def pipeline() -> dict[str, Any]:
        while not cancel_event.is_set():
            await asyncio.sleep(0.01)
        return {"late": True}

    cleaned = asyncio.Event()

    async def cleanup() -> None:
        cleaned.set()

    await asyncio.wait_for(run_claimed_job(handle, pipeline, cancel_event, cleanup, _FAST), _STALL_TIMEOUT)

    assert cancel_event.is_set()
    assert handle.completed == []
    assert handle.failures == []
    assert cleaned.is_set()


@pytest.mark.asyncio
async def test_a_transient_heartbeat_failure_does_not_kill_the_job() -> None:
    """One missed beat is well inside the lease; treating it as fatal would throw away good work."""

    class FlakyHandle(FakeHandle):
        async def heartbeat(self, lease_seconds: float) -> CheckerJobState:
            self.heartbeats += 1
            if self.heartbeats == 1:
                raise TimeoutError("database blip")
            return CheckerJobState(execution_id=self.execution_id)

    handle = FlakyHandle()
    release = asyncio.Event()

    async def pipeline() -> dict[str, Any]:
        await release.wait()
        return {"ok": True}

    async def cleanup() -> None:
        return None

    job = asyncio.create_task(run_claimed_job(handle, pipeline, threading.Event(), cleanup, _FAST))
    await _wait_for_heartbeats(handle, 2)
    release.set()
    await job

    assert handle.completed == [{"ok": True}]


@pytest.mark.asyncio
async def test_a_heartbeat_that_hangs_is_abandoned_rather_than_waited_out() -> None:
    """The reason a live pod used to lose its job: an unbounded beat spends the lease it is trying to renew.

    A hung beat has to be given up on well inside the lease and retried, so a slow database costs the pod a beat rather
    than the job. Without the ceiling this test hangs for as long as the HTTP client's own deadline and retries allow,
    which can exceed the whole lease.
    """

    class HangingHandle(FakeHandle):
        async def heartbeat(self, lease_seconds: float) -> CheckerJobState:
            self.heartbeats += 1
            if self.heartbeats == 1:
                await asyncio.sleep(3600)
            return CheckerJobState(execution_id=self.execution_id)

    handle = HangingHandle()
    release = asyncio.Event()

    async def pipeline() -> dict[str, Any]:
        await release.wait()
        return {"ok": True}

    async def cleanup() -> None:
        return None

    job = asyncio.create_task(run_claimed_job(handle, pipeline, threading.Event(), cleanup, _FAST))
    # A second beat can only happen if the first was abandoned instead of awaited to its (hour-long) end.
    await _wait_for_heartbeats(handle, 2)
    release.set()
    await job

    assert handle.completed == [{"ok": True}]


class TestLeasePolicy:
    """The lease and the heartbeat schedule have to be chosen together, which is what this type exists to enforce."""

    def test_the_production_default_survives_several_failed_beats(self) -> None:
        assert LeasePolicy().tolerated_failures >= LeasePolicy.MINIMUM_TOLERATED_FAILURES

    def test_the_production_default_tolerates_a_slow_database_rest_api(self) -> None:
        """The incident behind these numbers: beats against a struggling Database REST API took over 10s each.

        Asserted as a floor rather than equality so the defaults can be widened further without editing this test, but
        not narrowed back below what a slow-but-recovering API needs.
        """
        policy = LeasePolicy()
        assert policy.heartbeat_timeout >= 15.0
        assert policy.lease_seconds >= 150.0

    def test_a_lease_too_short_for_the_retry_schedule_is_rejected(self) -> None:
        """The bug this guards: a lease that looks generous next to the interval but that the retry path can run out."""
        with pytest.raises(ValueError, match="failed heartbeat"):
            LeasePolicy(lease_seconds=45.0, heartbeat_interval=30.0, heartbeat_timeout=10.0, retry_delay=5.0)

    def test_a_beat_allowed_to_outlast_the_gap_between_beats_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="shorter than heartbeat_interval"):
            LeasePolicy(heartbeat_timeout=30.0, heartbeat_interval=30.0)

    def test_nonpositive_intervals_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be a positive number"):
            LeasePolicy(retry_delay=0.0)

    def test_tolerated_failures_counts_timeout_and_delay_against_the_lease(self) -> None:
        """Each failure costs its own ceiling plus the wait before the next try, all of it inside the lease."""
        policy = LeasePolicy(lease_seconds=130.0, heartbeat_interval=10.0, heartbeat_timeout=5.0, retry_delay=5.0)
        assert policy.tolerated_failures == 12


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
