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

"""Tests for taking over an arbitration request whose replica went away.

Running work off rows makes the checker *work* survive a lost pod, but a request's aggregation lives in one arbitrator
replica's memory: lose that replica and the checkers still finish while nobody assembles their answer, leaving an
execution at ``in_progress`` forever. A lease makes the loss visible from the outside and a sweep on every replica
picks the request back up, rebuilding it from the rows the jobs were already reporting through.

Two layers are covered here. The waiting room's side is a small contract -- adopting has to resume watching jobs a pod
is already running rather than starting second copies. The rebuild is the substantial part: it has to land every job in
the state its row describes, keep pointing at the existing rows rather than inserting new ones, and leave the scheduler
with exactly the work that is genuinely outstanding.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any, cast
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from database.api_models import (
    AdoptedExecutionRow,
    CheckerExecutionStateRow,
    ExecutionLeaseRenewal,
    LostExecutionLease,
)
from services.arbitrator.checker_queue import JobState
from services.arbitrator.dependency_graph import DependencyGraph
from services.arbitrator.rest_api_common import ArbitratorApp
from services.arbitrator.settings import Settings
from services.arbitrator.tests.waiting_room_test_helpers import make_active_request
from services.arbitrator.waiting_room_queue import ActiveRequest, WaitingRoom

_CAMERA = "camera_front_wide_120fov"
_OTHER_CAMERA = "camera_cross_left_120fov"
_SINGLE_VIEW = "av.objects_single_view"
_MULTI_VIEW = "av.objects_multi_view"

_CONFIG = {
    _SINGLE_VIEW: {"enabled": True, "deps": []},
    _MULTI_VIEW: {"enabled": True, "deps": [_SINGLE_VIEW]},
}
_ENDPOINTS = {_SINGLE_VIEW: "http://objects", _MULTI_VIEW: "http://multi"}

_REQUEST_JSON: dict[str, Any] = {
    "inputs": {
        "augmented_video_urls": {_CAMERA: "s3://bucket/front.mp4", _OTHER_CAMERA: "s3://bucket/left.mp4"},
        "world_model_video_urls": {_CAMERA: "s3://bucket/front_wm.mp4", _OTHER_CAMERA: "s3://bucket/left_wm.mp4"},
        "rds_hq_url": "s3://bucket/rds.zip",
    },
    "checks": [_SINGLE_VIEW],
}


def _empty_room(capacity: int | None = None) -> WaitingRoom:
    """A waiting room with every collaborator stubbed; adoption never reaches any of them."""
    return WaitingRoom(
        build_request=lambda job, request, queue: {},
        store_result=lambda agg, job, data: agg.__setitem__(job.base_name, data),
        capacity=capacity,
    )


class TestWaitingRoomAdoption(unittest.IsolatedAsyncioTestCase):
    """What the scheduler and the monitor each see after a takeover."""

    async def test_a_job_a_pod_is_running_goes_to_the_monitor_not_the_scheduler(self) -> None:
        room = _empty_room()
        req = make_active_request(["check_a"], execution_ids={"check_a": 77})
        req.queue.restore("check_a", JobState.IN_PROGRESS)
        assert room.try_admit()

        room.adopt(req, {"check_a": 77})

        self.assertIn(77, room._durable_jobs)
        self.assertIs(room._durable_jobs[77].req, req)
        self.assertEqual(req.queue.get_ready_jobs(), [], "a second copy of running work would be started otherwise")

    async def test_an_adopted_job_is_budget_eligible_immediately(self) -> None:
        """Everything adopted was already claimed, so the budget covers it from the takeover onwards.

        Left unstamped it would read as a job still waiting its turn, which is exempt, and an adopted job whose row
        stayed unreadable would then never time out at all.
        """
        room = _empty_room()
        req = make_active_request(["check_a"], execution_ids={"check_a": 77})
        req.queue.restore("check_a", JobState.IN_PROGRESS)
        assert room.try_admit()

        room.adopt(req, {"check_a": 77})

        self.assertIsNotNone(room._durable_jobs[77].started_at)

    async def test_a_request_with_nothing_in_flight_is_simply_admitted(self) -> None:
        room = _empty_room()
        req = make_active_request(["check_a"])
        assert room.try_admit()

        room.adopt(req, {})

        self.assertEqual(list(room.queue), [req])
        self.assertEqual(room._durable_jobs, {})

    async def test_an_adopted_request_that_is_already_finished_finalizes_itself(self) -> None:
        """Its last job completed while it had no owner, so nothing else will ever wake it."""
        room = _empty_room()
        req = make_active_request(["check_a"])
        req.queue.restore("check_a", JobState.COMPLETED, result={"ok": True})
        assert room.try_admit()

        room.adopt(req, {})
        await asyncio.wait_for(req.completion_future, timeout=1.0)

        self.assertEqual(list(room.queue), [])

    async def test_free_capacity_is_unbounded_when_no_limit_is_set(self) -> None:
        self.assertIsNone(_empty_room().free_capacity())

    async def test_free_capacity_counts_reservations_that_have_not_committed(self) -> None:
        """A sweep that ignored them would claim work the room turns out to have no room for."""
        room = _empty_room(capacity=2)
        assert room.try_admit()

        self.assertEqual(room.free_capacity(), 1)

    async def test_a_draining_replica_has_no_room_for_anybody_else_s_work(self) -> None:
        room = _empty_room(capacity=5)
        room.begin_draining()

        self.assertEqual(room.free_capacity(), 0)


def _build_settings() -> Settings:
    """A real base ``Settings`` with YAML/runfiles resolution stubbed out."""
    graph = DependencyGraph(_CONFIG)
    graph.validate()
    with (
        patch.object(Settings, "_resolve_checker_endpoints", return_value=_ENDPOINTS),
        patch.object(Settings, "_resolve_database_api_url", return_value=""),
        patch.object(Settings, "_load_dependency_graph", return_value=graph),
        patch("services.settings_base.SettingsBase.get_env_files", return_value=[]),
    ):
        return Settings(_env_file=[])


def _job_row(
    execution_id: int,
    checker_name: str,
    status: str,
    camera_name: str | None = None,
    response_json: dict[str, Any] | None = None,
    error_message: str | None = None,
    claimed_by: str | None = None,
    attempt: int = 1,
) -> CheckerExecutionStateRow:
    """One persisted ``checker_executions`` row as the adoption read returns it."""
    return CheckerExecutionStateRow(
        execution_id=execution_id,
        request_id=7,
        checker_name=checker_name,
        camera_name=camera_name,
        status=status,
        response_json=response_json,
        error_message=error_message,
        claimed_by=claimed_by,
        lease_expires=(datetime.now(UTC) + timedelta(seconds=60)).isoformat() if claimed_by else None,
        attempt=attempt,
    )


def _adopted(status: str = "in_progress") -> AdoptedExecutionRow:
    return AdoptedExecutionRow(
        request_id=7,
        client_id=3,
        status=status,
        request_json=_REQUEST_JSON,
        claimed_by="replica-b",
        version="1.0.0",
        commit_sha="abc",
    )


class _RebuildCase(unittest.IsolatedAsyncioTestCase):
    """An arbitrator app with a stubbed database, ready to rebuild a request from scripted rows."""

    def setUp(self) -> None:
        patcher = patch("services.utils.get_git_sha", return_value="test_sha")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app = ArbitratorApp(settings=_build_settings())
        self.api_client = AsyncMock()
        self.app._api_client = self.api_client
        self.app._waiting_room = _empty_room()

    def room(self) -> WaitingRoom:
        room = self.app._waiting_room
        assert room is not None, "the test installed one in setUp"
        return room

    def script(self, *rows: CheckerExecutionStateRow) -> None:
        self.api_client.get_checker_execution_states_for_request.return_value = list(rows)

    async def rebuild(self, row: AdoptedExecutionRow | None = None) -> tuple[ActiveRequest, dict[str, int]]:
        return await self.app._rebuild_active_request(row or _adopted())


class TestRebuild(_RebuildCase):
    """Reconstructing a request from its own row plus its checker rows."""

    async def test_a_finished_job_comes_back_with_its_result(self) -> None:
        self.script(
            _job_row(101, _SINGLE_VIEW, "success", _CAMERA, response_json={"data": {"score": 0.9}}),
            _job_row(102, _SINGLE_VIEW, "queued", _OTHER_CAMERA),
        )

        active, in_flight = await self.rebuild()

        self.assertEqual(active.aggregated_results[_SINGLE_VIEW][_CAMERA], {"score": 0.9})
        self.assertEqual(active.queue.get_job(f"{_SINGLE_VIEW}[{_CAMERA}]").state, JobState.COMPLETED)
        self.assertEqual(in_flight, {})

    async def test_a_finished_job_is_not_written_again(self) -> None:
        """Its row is already terminal; rewriting it would replace the checker's own account of the run."""
        self.script(_job_row(101, _SINGLE_VIEW, "success", _CAMERA, response_json={"data": {}}))

        active, _ = await self.rebuild()

        self.assertIn(f"{_SINGLE_VIEW}[{_CAMERA}]", active.db_terminalized)

    async def test_a_failed_job_comes_back_with_the_error_the_checker_recorded(self) -> None:
        self.script(_job_row(101, _SINGLE_VIEW, "fail", _CAMERA, error_message="cuda OOM"))

        active, _ = await self.rebuild()

        self.assertEqual(active.queue.get_job(f"{_SINGLE_VIEW}[{_CAMERA}]").state, JobState.FAILED)
        self.assertEqual(active.aggregated_results[_SINGLE_VIEW][_CAMERA], {"error": "cuda OOM"})

    async def test_a_job_a_pod_is_still_running_is_reported_as_in_flight(self) -> None:
        self.script(_job_row(101, _SINGLE_VIEW, "in_progress", _CAMERA, claimed_by="pod-a"))

        active, in_flight = await self.rebuild()

        self.assertEqual(in_flight, {f"{_SINGLE_VIEW}[{_CAMERA}]": 101})
        self.assertEqual(active.queue.get_job(f"{_SINGLE_VIEW}[{_CAMERA}]").state, JobState.IN_PROGRESS)

    async def test_a_queued_job_no_pod_has_claimed_is_left_for_the_scheduler(self) -> None:
        """Nobody ever took it, so there is no lease to wait on and nothing to report as in flight."""
        self.script(_job_row(101, _SINGLE_VIEW, "queued", _CAMERA, claimed_by=None))

        active, in_flight = await self.rebuild()

        self.assertEqual(in_flight, {})
        self.assertEqual(active.queue.get_job(f"{_SINGLE_VIEW}[{_CAMERA}]").state, JobState.READY)

    async def test_a_job_that_was_never_offered_is_left_for_the_scheduler(self) -> None:
        """A 'pending' row is the state of a job whose payload was never written, so it is simply outstanding work.

        Reading it as terminal would drop a job the request still needs, and reading it as in flight would leave the
        monitor waiting on a lease no pod ever took.
        """
        self.script(_job_row(101, _SINGLE_VIEW, "pending", _CAMERA, claimed_by=None))

        active, in_flight = await self.rebuild()

        self.assertEqual(in_flight, {})
        self.assertEqual(active.queue.get_job(f"{_SINGLE_VIEW}[{_CAMERA}]").state, JobState.READY)
        self.assertNotIn(f"{_SINGLE_VIEW}[{_CAMERA}]", active.db_terminalized)

    async def test_a_never_offered_aggregator_waits_on_its_dependencies_again(self) -> None:
        """The row says nothing about the deps, so the rebuilt graph is what has to hold the job back.

        This is the shape every multi-view request is adopted in: its own row has never been offered, and whether it
        may run is a question about its dependencies rather than about its status.
        """
        self.script(
            _job_row(101, _SINGLE_VIEW, "success", _CAMERA, response_json={"data": {}}),
            _job_row(102, _SINGLE_VIEW, "pending", _OTHER_CAMERA),
            _job_row(103, _MULTI_VIEW, "pending"),
        )

        active, in_flight = await self.rebuild()

        self.assertEqual(in_flight, {})
        self.assertEqual(active.queue.get_job(_MULTI_VIEW).state, JobState.PENDING)
        self.assertEqual([job.name for job in active.queue.get_ready_jobs()], [f"{_SINGLE_VIEW}[{_OTHER_CAMERA}]"])

    async def test_the_attempt_count_survives_the_takeover(self) -> None:
        """Otherwise a job that has already exhausted the fleet gets a fresh set of attempts on every adoption."""
        self.script(_job_row(101, _SINGLE_VIEW, "in_progress", _CAMERA, claimed_by="pod-a", attempt=3))

        active, _ = await self.rebuild()

        self.assertEqual(active.queue.get_job(f"{_SINGLE_VIEW}[{_CAMERA}]").attempt, 3)

    async def test_the_rebuilt_records_point_at_the_existing_rows(self) -> None:
        self.script(_job_row(101, _SINGLE_VIEW, "queued", _CAMERA))

        active, _ = await self.rebuild()

        record = active.checker_executions[f"{_SINGLE_VIEW}[{_CAMERA}]"]
        self.assertEqual(record.execution_id, 101)
        self.api_client.create_checker_execution.assert_not_awaited()

    async def test_the_rebuilt_execution_takes_over_the_parent_row(self) -> None:
        self.script(_job_row(101, _SINGLE_VIEW, "queued", _CAMERA))

        active, _ = await self.rebuild()

        self.assertEqual(active.execution.request_id, 7)
        self.assertEqual(active.execution.claimed_by, self.app._replica_id)
        self.api_client.create_execution.assert_not_awaited()

    async def test_a_request_already_past_queued_is_not_flipped_again(self) -> None:
        self.script(_job_row(101, _SINGLE_VIEW, "in_progress", _CAMERA, claimed_by="pod-a"))

        active, _ = await self.rebuild(_adopted(status="in_progress"))

        self.assertTrue(active.in_progress_marked)

    async def test_a_request_adopted_before_its_first_dispatch_still_gets_the_flip(self) -> None:
        self.script(_job_row(101, _SINGLE_VIEW, "queued", _CAMERA))

        active, _ = await self.rebuild(_adopted(status="queued"))

        self.assertFalse(active.in_progress_marked)

    async def test_the_checker_set_comes_from_the_rows_not_from_health(self) -> None:
        """A checker that has gone unhealthy since admission must not silently vanish from an admitted request."""
        self.script(
            _job_row(101, _SINGLE_VIEW, "success", _CAMERA, response_json={"data": {}}),
            _job_row(102, _SINGLE_VIEW, "success", _OTHER_CAMERA, response_json={"data": {}}),
            _job_row(103, _MULTI_VIEW, "queued"),
        )

        active, _ = await self.rebuild()

        self.assertIn(_MULTI_VIEW, {job.base_name for job in active.queue.iter_jobs()})
        self.api_client.get_checker_execution_states_for_request.assert_awaited_once_with(7)

    async def test_only_genuinely_outstanding_work_is_left_for_the_scheduler(self) -> None:
        self.script(
            _job_row(101, _SINGLE_VIEW, "success", _CAMERA, response_json={"data": {}}),
            _job_row(102, _SINGLE_VIEW, "queued", _OTHER_CAMERA),
        )

        active, _ = await self.rebuild()

        self.assertEqual([job.name for job in active.queue.get_ready_jobs()], [f"{_SINGLE_VIEW}[{_OTHER_CAMERA}]"])

    async def test_a_row_the_graph_no_longer_produces_is_skipped_rather_than_misfiled(self) -> None:
        """The camera set or the graph changed under a live request; the result belongs to no job here."""
        self.script(
            _job_row(101, _SINGLE_VIEW, "queued", _CAMERA),
            _job_row(999, "av.egomotion", "success", _CAMERA, response_json={"data": {"drift": 1}}),
        )

        active, _ = await self.rebuild()

        self.assertNotIn("av.egomotion", active.aggregated_results)
        self.assertNotIn("av.egomotion", active.checker_executions)


class TestSweep(_RebuildCase):
    """Claiming stale requests, and the limits on how many.

    The claim has to happen for the request to be seen at all, so a sweep that takes more than the replica can run
    would leave the surplus owned by somebody who never starts it -- recoverable only after another whole lease.
    """

    async def test_an_adopted_request_reaches_the_scheduler(self) -> None:
        self.api_client.adopt_executions.return_value = [_adopted()]
        self.script(_job_row(101, _SINGLE_VIEW, "in_progress", _CAMERA, claimed_by="pod-a"))

        await self.app.sweep_adoptable()

        room = self.room()
        self.assertEqual([req.execution.request_id for req in room.queue], [7])
        self.assertIn(101, room._durable_jobs)

    async def test_nothing_stale_means_nothing_admitted(self) -> None:
        self.api_client.adopt_executions.return_value = []

        await self.app.sweep_adoptable()

        self.assertEqual(list(self.room().queue), [])

    async def test_the_claim_is_bounded_by_the_room_that_is_actually_free(self) -> None:
        self.app._waiting_room = _empty_room(capacity=2)
        self.api_client.adopt_executions.return_value = []

        await self.app.sweep_adoptable()

        self.assertEqual(self.api_client.adopt_executions.call_args[1]["limit"], 2)

    async def test_the_batch_limit_caps_a_large_free_room(self) -> None:
        self.app._waiting_room = _empty_room(capacity=1000)
        self.api_client.adopt_executions.return_value = []

        await self.app.sweep_adoptable()

        self.assertEqual(self.api_client.adopt_executions.call_args[1]["limit"], ArbitratorApp.ADOPTION_BATCH_LIMIT)

    async def test_a_full_replica_does_not_claim_at_all(self) -> None:
        room = _empty_room(capacity=1)
        assert room.try_admit()
        self.app._waiting_room = room

        await self.app.sweep_adoptable()

        self.api_client.adopt_executions.assert_not_awaited()

    async def test_a_draining_replica_does_not_claim_at_all(self) -> None:
        self.room().begin_draining()

        await self.app.sweep_adoptable()

        self.api_client.adopt_executions.assert_not_awaited()

    async def test_one_unrebuildable_request_does_not_stop_the_rest(self) -> None:
        self.api_client.adopt_executions.return_value = [_adopted(), _adopted()]
        self.api_client.get_checker_execution_states_for_request.side_effect = [
            RuntimeError("db hiccup"),
            [_job_row(101, _SINGLE_VIEW, "queued", _CAMERA)],
        ]

        await self.app.sweep_adoptable()

        room = self.room()
        self.assertEqual(len(room.queue), 1)
        self.assertIsNone(room.free_capacity())

    async def test_a_request_already_running_here_is_not_rebuilt_on_top_of_itself(self) -> None:
        """The database excludes this replica's own rows, so getting here means a row came back some other way -- a
        request adopted away and then stranded by its new owner. Rebuilding it would leave two copies of one request
        dispatching everything twice, with whichever finished first stranding the other on a terminal row.
        """
        room = self.room()
        req = make_active_request(["check_a"], request_id=7)
        assert room.try_admit()
        room.admit(req)
        self.api_client.adopt_executions.return_value = [_adopted()]

        with self.assertLogs("services.arbitrator.rest_api_common", level="WARNING"):
            await self.app.sweep_adoptable()

        self.assertEqual([r.execution.request_id for r in room.queue], [7])
        self.api_client.get_checker_execution_states_for_request.assert_not_awaited()

    async def test_a_failed_rebuild_gives_its_reservation_back(self) -> None:
        """A leaked reservation counts against capacity forever and eventually wedges admissions altogether."""
        room = _empty_room(capacity=2)
        self.app._waiting_room = room
        self.api_client.adopt_executions.return_value = [_adopted()]
        self.api_client.get_checker_execution_states_for_request.side_effect = RuntimeError("db hiccup")

        await self.app.sweep_adoptable()

        self.assertEqual(room.free_capacity(), 2)


class TestRenewal(_RebuildCase):
    """Keeping hold of what this replica is running, and telling the truth about what it has lost."""

    def _admit(self, request_id: int = 1) -> MagicMock:
        """Admit one request and hand back its stub execution record."""
        room = self.room()
        req = make_active_request(["check_a"], request_id=request_id)
        assert room.try_admit()
        room.admit(req)
        return cast(MagicMock, req.execution)

    def _answer(self, *, renewed: list[int] | None = None, lost: list[dict[str, Any]] | None = None) -> None:
        self.api_client.heartbeat_executions.return_value = ExecutionLeaseRenewal(
            renewed=renewed or [], lost=[LostExecutionLease.model_validate(entry) for entry in lost or []]
        )

    async def test_every_admitted_request_is_renewed_in_a_single_call(self) -> None:
        """One round trip for the whole set: a pass that scaled with load would expire the leases it is renewing."""
        self._admit(1)
        self._admit(2)
        self._answer(renewed=[1, 2])

        await self.app.renew_ownership()

        self.api_client.heartbeat_executions.assert_awaited_once()
        self.assertEqual(self.api_client.heartbeat_executions.call_args.kwargs["request_ids"], [1, 2])

    async def test_a_replica_running_nothing_does_not_call_at_all(self) -> None:
        await self.app.renew_ownership()

        self.api_client.heartbeat_executions.assert_not_awaited()

    async def test_a_request_taken_over_names_the_replica_that_has_it(self) -> None:
        self._admit(1)
        self._answer(lost=[{"request_id": 1, "status": "in_progress", "claimed_by": "replica-b"}])

        with self.assertLogs("services.arbitrator.rest_api_common", level="ERROR") as logs:
            await self.app.renew_ownership()

        self.assertIn("Lost the claim on request_id=1 to replica replica-b", logs.output[0])

    async def test_a_request_that_finished_underneath_is_not_called_a_takeover(self) -> None:
        """The row is still claimed here, so nobody took it. Reporting a takeover sends the reader looking for a
        replica that does not exist; the real fault is that this replica was running the request twice."""
        self._admit(1)
        self._answer(lost=[{"request_id": 1, "status": "success", "claimed_by": self.app._replica_id}])

        with self.assertLogs("services.arbitrator.rest_api_common", level="ERROR") as logs:
            await self.app.renew_ownership()

        self.assertIn("already terminal (status=success)", logs.output[0])
        self.assertNotIn("Lost the claim", logs.output[0])

    async def test_a_row_that_no_longer_exists_says_that_instead_of_guessing(self) -> None:
        self._admit(1)
        self._answer(lost=[{"request_id": 1}])

        with self.assertLogs("services.arbitrator.rest_api_common", level="ERROR") as logs:
            await self.app.renew_ownership()

        self.assertIn("no longer exists", logs.output[0])

    async def test_a_lost_claim_stops_the_record_from_writing_again(self) -> None:
        """Its aggregation is stale now; landing it on top of the real one is the outcome all the fencing prevents."""
        execution = self._admit(1)
        self._answer(lost=[{"request_id": 1, "status": "in_progress", "claimed_by": "replica-b"}])

        await self.app.renew_ownership()

        execution.stand_down.assert_called_once_with()

    async def test_an_outage_leaves_every_claim_where_it_is(self) -> None:
        """An unreachable database is not evidence of adoption, and standing down over one strands a live request."""
        execution = self._admit(1)
        self.api_client.heartbeat_executions.side_effect = RuntimeError("db down")

        with self.assertRaises(RuntimeError):
            await self.app.renew_ownership()

        execution.stand_down.assert_not_called()


if __name__ == "__main__":
    unittest.main()
