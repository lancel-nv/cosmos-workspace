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

"""Tests for the arbitrator's graceful shutdown.

A replica keeps every admitted request's aggregation in its own memory, so one killed while holding some strands those
executions at ``in_progress`` with nobody left to finish them. Rollouts cause nearly all of these, and a rollout arrives
as SIGTERM, so the replica intercepts it: stop admitting, finish what is already admitted, then let the shutdown it
interrupted carry on.

The signal tests drive the lifespan directly and send real signals, because the interception has to sit ahead of the
handler uvicorn installed and there is no way to observe that ordering through a stubbed signal module. What they hold
in flight is an uncommitted reservation rather than a queued request: it is the same thing to a drain, and it stays put
until the test releases it instead of racing the scheduler.
"""

import asyncio
from collections.abc import Callable
import os
import signal
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.arbitrator.dependency_graph import DependencyGraph
from services.arbitrator.rest_api_common import ArbitratorApp
from services.arbitrator.settings import Settings
from services.arbitrator.waiting_room_queue import WaitingRoom

_CONFIG = {"av.objects_single_view": {"enabled": True, "deps": []}}
_ENDPOINTS = {"av.objects_single_view": "http://objects"}

_REQUEST_BODY = {
    "inputs": {"augmented_video_urls": {"camera_front_wide_120fov": "s3://bucket/front.mp4"}},
    "checks": ["av.objects_single_view"],
}


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


class _AppCase(unittest.TestCase):
    """Builds one app per test with the git-sha genrule stubbed, since it is not in this test's runfiles."""

    def setUp(self) -> None:
        patcher = patch("services.utils.get_git_sha", return_value="test_sha")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.app = ArbitratorApp(settings=_build_settings())
        self.fastapi_app: FastAPI = self.app.create_app()

    def room(self) -> WaitingRoom:
        """The waiting room the running lifespan installed."""
        room = self.app._waiting_room
        assert room is not None, "the lifespan has not started yet"
        return room


class TestDrainingIsVisibleToCallers(_AppCase):
    """What a client sees from a replica on its way out.

    ``TestClient`` runs the lifespan off the main thread, so no signal handler can be installed here -- which also
    covers that startup degrades to the lifespan drain instead of failing -- and the drain is started directly.
    """

    def test_health_reports_a_replica_that_is_still_taking_work(self) -> None:
        with TestClient(self.fastapi_app) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["data"]["draining"])

    def test_health_reports_the_drain_without_calling_itself_unhealthy(self) -> None:
        """A liveness probe must not read a graceful drain as a crash and escalate straight to SIGKILL."""
        with TestClient(self.fastapi_app) as client:
            self.room().begin_draining()
            response = client.get("/health")

        data = response.json()["data"]
        self.assertTrue(data["draining"])
        self.assertEqual(data["status"], "healthy")

    def test_submissions_are_turned_away_with_a_reason_of_their_own(self) -> None:
        """Not the queue-full 503: nothing is full, and a retry reaches a replica that will take the work."""
        with TestClient(self.fastapi_app) as client:
            self.room().begin_draining()
            response = client.post("/process/async", json=_REQUEST_BODY)

        self.assertEqual(response.status_code, 503)
        self.assertIn("shutting down", response.json()["error"]["message"])


class TestSigterm(unittest.IsolatedAsyncioTestCase, _AppCase):
    """The signal path, driven on the main thread where handlers can actually be installed."""

    def setUp(self) -> None:
        super().setUp()
        # Restored as it stands, not flattened to SIG_DFL: SIG_IGN is a disposition a test runner may legitimately have
        # installed, and swapping it for the default would leave the process dying on a signal it was told to ignore.
        # Only ``None`` -- a handler Python did not install and cannot name -- has nothing to put back.
        previous = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)
        self.exits: list[int] = []
        signal.signal(signal.SIGTERM, lambda signum, _frame: self.exits.append(signum))

    async def _await_until(self, what: str, done: Callable[[], bool], timeout: float = 2.0) -> None:
        """Let the loop run until ``done`` holds, so signal delivery is never assumed to be instantaneous."""
        for _ in range(int(timeout / 0.01)):
            if done():
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"timed out waiting for {what}")

    async def _await_exit(self) -> None:
        await self._await_until("the shutdown to be handed on", lambda: bool(self.exits))

    async def _await_drain_start(self) -> None:
        await self._await_until("the drain to start", lambda: self.app._drain_task is not None)

    async def test_the_handler_defers_to_the_one_it_displaced(self) -> None:
        """Uvicorn's ``handle_exit`` is what sits there in production, and it still has to run."""
        async with self.app._lifespan(self.fastapi_app):
            os.kill(os.getpid(), signal.SIGTERM)
            await self._await_exit()

            self.assertEqual(self.exits, [signal.SIGTERM], "the ordinary shutdown never got to happen")
            self.assertTrue(self.room().is_draining)

    async def test_the_drain_waits_for_work_in_flight_and_then_lets_go(self) -> None:
        with patch.object(WaitingRoom, "DRAIN_POLL_INTERVAL", 0.01):
            async with self.app._lifespan(self.fastapi_app):
                room = self.room()
                assert room.try_admit(), "an unbounded room takes the reservation"

                os.kill(os.getpid(), signal.SIGTERM)
                await self._await_drain_start()
                await asyncio.sleep(0.05)
                self.assertEqual(self.exits, [], "shutdown proceeded while a submission was still in flight")

                room.release()
                await self._await_exit()

        self.assertEqual(self.exits, [signal.SIGTERM])

    async def test_the_drain_gives_up_when_its_budget_runs_out(self) -> None:
        """Bounded because the platform's grace period is: waiting past it only means being killed mid-wait."""
        with (
            patch.object(WaitingRoom, "DRAIN_POLL_INTERVAL", 0.01),
            patch.object(ArbitratorApp, "SHUTDOWN_DRAIN_TIMEOUT", 0.05),
        ):
            async with self.app._lifespan(self.fastapi_app):
                assert self.room().try_admit()

                os.kill(os.getpid(), signal.SIGTERM)
                await self._await_exit()

        self.assertEqual(self.exits, [signal.SIGTERM], "a stuck submission must not stop the process from exiting")

    async def test_a_second_signal_stops_the_waiting(self) -> None:
        with patch.object(WaitingRoom, "DRAIN_POLL_INTERVAL", 0.01):
            async with self.app._lifespan(self.fastapi_app):
                assert self.room().try_admit()

                os.kill(os.getpid(), signal.SIGTERM)
                await self._await_drain_start()
                await asyncio.sleep(0.05)
                self.assertEqual(self.exits, [], "the drain should still be waiting on the reservation")

                os.kill(os.getpid(), signal.SIGTERM)
                await self._await_exit()

        self.assertEqual(self.exits, [signal.SIGTERM], "the impatient operator was made to keep waiting")

    async def test_the_handler_is_put_back_on_shutdown(self) -> None:
        """Leaving it installed would let a later SIGTERM drain a waiting room that no longer exists."""
        installed = signal.getsignal(signal.SIGTERM)

        async with self.app._lifespan(self.fastapi_app):
            self.assertIsNot(signal.getsignal(signal.SIGTERM), installed)

        self.assertIs(signal.getsignal(signal.SIGTERM), installed)


class TestTeardownWithoutASignal(unittest.IsolatedAsyncioTestCase, _AppCase):
    """Shutdowns that never went through SIGTERM at all -- SIGINT, an embedded app, a test client."""

    def setUp(self) -> None:
        super().setUp()
        previous = signal.getsignal(signal.SIGTERM)
        self.addCleanup(signal.signal, signal.SIGTERM, previous if previous is not None else signal.SIG_DFL)

    async def test_teardown_stops_admitting(self) -> None:
        async with self.app._lifespan(self.fastapi_app):
            room = self.room()

        self.assertTrue(room.is_draining)
        self.assertFalse(room.try_admit())

    async def test_teardown_does_not_spend_the_drain_budget(self) -> None:
        """The budget is sized to the platform's grace period, which a Ctrl-C or a test client is not bound by."""
        waiting = patch.object(WaitingRoom, "wait_until_drained", new_callable=AsyncMock)
        never_waited = waiting.start()
        self.addCleanup(waiting.stop)

        async with self.app._lifespan(self.fastapi_app):
            assert self.room().try_admit(), "leave work in flight, so waiting would be observable"

        never_waited.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
