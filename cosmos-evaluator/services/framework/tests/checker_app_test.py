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

"""Tests for the shared checker transport.

These cover the surface every checker inherits -- how it claims work, the slot ceiling, the temp-directory lifecycle,
status mapping -- against a minimal stand-in checker, so each real checker's own tests only have to cover the two
functions it actually supplies.
"""

import asyncio
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
import signal
import threading
from typing import Any
import unittest
from unittest.mock import patch

from fastapi import FastAPI, Response
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from pydantic import BaseModel
from pydantic_settings import SettingsConfigDict

from database.api_models import CheckerJobState
from services.framework import checker_app
from services.framework.checker_app import CheckerSpec, build_checker_app
from services.framework.request_handlers.json_request_handler import RequestValidationError
from services.framework.worker_settings import WorkerDatabaseSettings
from utils.environment import Environment
from utils.gpu import GpuUnavailableError


class StubCloudRequest(BaseModel):
    """Minimal cloud request: one URL to "download", one flag to make the pipeline misbehave."""

    source_url: str = "s3://bucket/input.mp4"
    fail_with: str | None = None


class StubResult(BaseModel):
    """Minimal result carrying the one field the transport reads."""

    output_dir: str
    clip_id: str = "clip-1"


class StubLocalRequest(BaseModel):
    local_path: str


class StubSettings(WorkerDatabaseSettings):
    """Settings for the stand-in checker, with storage pinned to the local backend."""

    model_config = SettingsConfigDict(env_prefix="STUB_CHECKER_TEST_", frozen=True, extra="ignore")


def _settings(**overrides: Any) -> StubSettings:
    values: dict[str, Any] = {
        "env": Environment.LOCAL,
        "input_storage_type": "local",
        "output_storage_type": "local",
    }
    values.update(overrides)
    return StubSettings(**values)


def _coordinating_settings() -> StubSettings:
    return _settings(database_api_url="https://db.example", database_worker_key="worker-key")


class StubService:
    """A service whose process step is instant and whose failures are requested by the caller."""

    def __init__(self) -> None:
        self.processed: list[StubLocalRequest] = []
        self.cleaned: list[Path] = []
        self.cancel_events: list[threading.Event | None] = []
        self.block: asyncio.Event | None = None
        self.running = asyncio.Event()

    async def process(self, request_data: StubLocalRequest, cancel_event: threading.Event | None = None) -> StubResult:
        self.processed.append(request_data)
        self.cancel_events.append(cancel_event)
        self.running.set()
        if self.block is not None:
            await self.block.wait()
        output_dir = Path(request_data.local_path) / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        return StubResult(output_dir=str(output_dir))

    async def validate_input(self, request_data: StubLocalRequest) -> bool:
        return True

    async def get_default_config(self) -> dict[str, Any]:
        return {"threshold": 0.5}

    async def cleanup(self, output_dir: Path) -> None:
        self.cleaned.append(output_dir)


def _build(
    service: StubService | None = None,
    settings: StubSettings | None = None,
    concurrency_limit: int = 2,
    **spec_overrides: Any,
) -> tuple[FastAPI, StubService, list[Path]]:
    """Assemble a stand-in checker app, returning it alongside its service and the temp dirs it was handed."""
    service = service or StubService()
    temp_dirs: list[Path] = []

    async def prepare_request(request: StubCloudRequest, provider: Any, temp_dir: Path) -> StubLocalRequest:
        temp_dirs.append(temp_dir)
        if request.fail_with == "validation":
            raise RequestValidationError("malformed body")
        if request.fail_with == "value":
            raise ValueError("no such camera")
        if request.fail_with == "boom":
            raise RuntimeError("unexpected")
        if request.fail_with == "timeout":
            raise TimeoutError("download timed out")
        if request.fail_with == "gpu":
            raise GpuUnavailableError("cublasCreate failed")
        (temp_dir / "input.mp4").write_bytes(b"data")
        return StubLocalRequest(local_path=str(temp_dir))

    async def build_response_data(request: StubCloudRequest, result: StubResult, provider: Any) -> dict[str, Any]:
        return {"clip_id": result.clip_id, "output_dir": result.output_dir}

    spec = CheckerSpec(
        name="stub_checker",
        checker_name="test.stub_checker",
        title="Stub Checker API",
        description="A checker that exists only to exercise the shared transport",
        version="1.2.3",
        config_description="Default configuration for the stub checker",
        settings=settings or _settings(),
        concurrency_limit=concurrency_limit,
        cloud_request_model=StubCloudRequest,
        build_service=lambda: service,
        prepare_request=prepare_request,
        build_response_data=build_response_data,
        **spec_overrides,
    )
    return build_checker_app(spec), service, temp_dirs


class TestReadOnlyEndpoints(unittest.TestCase):
    """/health, /metrics and /config, which every checker gets identically."""

    def test_health_reports_identity_and_occupancy(self) -> None:
        app, _, _ = _build(concurrency_limit=4)
        with TestClient(app) as client:
            body = client.get("/health").json()["data"]

        self.assertEqual(body["service"], "Stub Checker API")
        self.assertEqual(body["status"], "healthy")
        self.assertEqual(body["version"], "1.2.3")
        self.assertEqual(body["checker"], "test.stub_checker")
        self.assertEqual(body["slots"], {"active": 0, "limit": 4})
        self.assertEqual(body["jobs"], 0)
        self.assertFalse(body["coordination"])

    def test_health_says_when_a_pod_is_not_drawing_work(self) -> None:
        """A pod that answers healthy but claims nothing runs nothing, which is otherwise invisible from outside."""
        app, _, _ = _build()
        with TestClient(app) as client:
            body = client.get("/health").json()["data"]

        self.assertFalse(body["claiming"])

    def test_health_answers_before_startup_has_run(self) -> None:
        """The arbitrator discovers pods through this endpoint, so it must not depend on the service existing."""
        app, _, _ = _build()
        client = TestClient(app)
        self.addCleanup(client.close)

        response = client.get("/health")

        self.assertEqual(response.status_code, 200)

    def test_metrics_exposes_the_checkers_own_label(self) -> None:
        app, _, _ = _build(concurrency_limit=3)
        with TestClient(app) as client:
            response = client.get("/metrics")

        self.assertIn("version=0.0.4", response.headers["content-type"])
        self.assertIn('checker_slots_limit{checker="test.stub_checker"} 3', response.text)

    def test_config_returns_the_services_defaults(self) -> None:
        app, _, _ = _build()
        with TestClient(app) as client:
            body = client.get("/config").json()["data"]

        self.assertEqual(body["default_config"], {"threshold": 0.5})
        self.assertEqual(body["description"], "Default configuration for the stub checker")

    def test_endpoints_that_need_the_service_answer_503_before_startup(self) -> None:
        app, _, _ = _build()
        client = TestClient(app)
        self.addCleanup(client.close)

        self.assertEqual(client.get("/config").status_code, 503)
        self.assertEqual(client.post("/process", json={}).status_code, 503)


class TestProcess(unittest.TestCase):
    """The direct path: one clip in, one result out, nothing left behind."""

    def test_a_successful_request_returns_the_uploaded_response(self) -> None:
        app, service, _ = _build()
        with TestClient(app) as client:
            response = client.post("/process", json={})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["data"]["clip_id"], "clip-1")
        self.assertIn("duration_ms", body["metadata"])
        self.assertEqual(len(service.processed), 1)

    def test_the_documented_request_body_is_the_checkers_own_model(self) -> None:
        """``/process`` must advertise the concrete model the spec named, not an open dict.

        Pins the one thing the request annotation buys that no other test would miss: the generated schema. The
        annotation has to reach FastAPI as a class for the body to be validated and for ``/docs`` to describe it, and
        a change that satisfied the type checker by loosening it to ``dict`` would break neither the 200 path nor the
        422 path -- it would only quietly empty the schema.
        """
        app, _, _ = _build()
        with TestClient(app) as client:
            schema = client.get("/openapi.json").json()

        body = schema["paths"]["/process"]["post"]["requestBody"]["content"]["application/json"]["schema"]
        self.assertEqual(body["$ref"], "#/components/schemas/StubCloudRequest")
        self.assertEqual(
            sorted(schema["components"]["schemas"]["StubCloudRequest"]["properties"]),
            ["fail_with", "source_url"],
        )

    def test_extra_metadata_from_the_checker_reaches_the_envelope(self) -> None:
        app, _, _ = _build(response_metadata=lambda result: {"processing_rate": "42.0%"})
        with TestClient(app) as client:
            body = client.post("/process", json={}).json()

        self.assertEqual(body["metadata"]["processing_rate"], "42.0%")

    def test_the_temp_directory_is_deleted_and_outputs_are_handed_back(self) -> None:
        app, service, temp_dirs = _build()
        with TestClient(app) as client:
            client.post("/process", json={})

        self.assertEqual(len(temp_dirs), 1)
        self.assertFalse(temp_dirs[0].exists())
        self.assertEqual(len(service.cleaned), 1)

    def test_the_temp_directory_is_deleted_when_the_request_fails(self) -> None:
        """Nothing else deletes it, so a failure that skipped cleanup would leak a directory per bad request."""
        app, _, temp_dirs = _build()
        with TestClient(app) as client:
            client.post("/process", json={"fail_with": "boom"})

        self.assertEqual(len(temp_dirs), 1)
        self.assertFalse(temp_dirs[0].exists())

    def test_failures_map_onto_the_status_that_says_whose_fault_it_was(self) -> None:
        app, _, _ = _build()
        with TestClient(app) as client:
            for fail_with, expected in (("validation", 400), ("value", 422), ("boom", 500)):
                with self.subTest(fail_with=fail_with):
                    response = client.post("/process", json={"fail_with": fail_with})
                    self.assertEqual(response.status_code, expected)

    def test_the_worker_thread_is_told_to_stop_once_the_request_ends(self) -> None:
        """Without this a client disconnect leaves the executor thread running the whole clip out."""
        app, service, _ = _build()
        with TestClient(app) as client:
            client.post("/process", json={})

        self.assertEqual(len(service.cancel_events), 1)
        event = service.cancel_events[0]
        assert event is not None
        self.assertTrue(event.is_set())

    def test_an_epilogue_failure_does_not_change_the_response(self) -> None:
        async def explode() -> None:
            raise RuntimeError("trim failed")

        app, _, _ = _build(after_request=explode)
        with TestClient(app) as client:
            response = client.post("/process", json={})

        self.assertEqual(response.status_code, 200)

    def test_the_slot_is_returned_so_the_next_request_is_not_locked_out(self) -> None:
        """A leaked slot on a single-slot replica is indistinguishable from a pod that has stopped working."""
        app, _, _ = _build(concurrency_limit=1)
        with TestClient(app) as client:
            first = client.post("/process", json={})
            second = client.post("/process", json={})

        self.assertEqual((first.status_code, second.status_code), (200, 200))
        self.assertEqual(app.state.slots.active, 0)


class _FakeHandle:
    """One claimed job, recording the writes its owner makes."""

    def __init__(
        self,
        execution_id: int = 42,
        request_json: dict[str, Any] | None = None,
        cancel_requested: bool = False,
    ) -> None:
        self.execution_id = execution_id
        self.claimed_by = "pod-a/abc"
        self.attempt = 1
        self.request_json = {} if request_json is None else request_json
        self.cancel_requested = cancel_requested
        self.completed: dict[str, Any] | None = None
        self.failure: str | None = None
        self.settled = asyncio.Event()

    async def heartbeat(self, lease_seconds: float) -> CheckerJobState:
        return CheckerJobState(execution_id=self.execution_id, cancel_requested=False)

    async def complete(
        self,
        response_json: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        output_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.completed = response_json
        self.settled.set()

    async def fail(
        self,
        error_message: str,
        duration_ms: int | None = None,
        response_json: dict[str, Any] | None = None,
        output_metadata: dict[str, Any] | None = None,
        timed_out: bool = False,
    ) -> None:
        self.failure = error_message
        self.settled.set()


class _StallingHandle(_FakeHandle):
    """A claimed job whose success write never returns, the way a call to a hung Database REST API does not."""

    def __init__(self) -> None:
        super().__init__()
        self.writing = asyncio.Event()

    async def complete(
        self,
        response_json: dict[str, Any] | None = None,
        duration_ms: int | None = None,
        output_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.writing.set()
        await asyncio.Event().wait()


class _FakeQueue:
    """Stands in for the job table the replica claims from.

    Serves the prepared batches in order and then holds every further claim open, the way an empty queue's long poll
    does. That keeps a test's assertions about what was claimed stable instead of racing a retry loop.
    """

    def __init__(self, *batches: list[_FakeHandle]) -> None:
        self._batches = list(batches)
        self.claims: list[dict[str, Any]] = []
        self.exhausted = asyncio.Event()
        self.fail_next = 0
        # Called synchronously as each claim begins. The claim loop asks again straight after starting a batch, so on
        # the second call the jobs it just started exist as tasks but have not yet had a turn on the loop.
        self.on_claim: Callable[[], None] | None = None

    async def claim_next_checker_jobs(self, **kwargs: Any) -> list[_FakeHandle]:
        if self.on_claim is not None:
            self.on_claim()
        self.claims.append(kwargs)
        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError("database is unreachable")
        if self._batches:
            return self._batches.pop(0)
        self.exhausted.set()
        await asyncio.sleep(3600)
        return []

    async def aclose(self) -> None:
        return None


@asynccontextmanager
async def _running(app: FastAPI, queue: _FakeQueue) -> AsyncGenerator[None]:
    """Run the app's lifespan in the test's own event loop, with the queue standing in for the database client.

    Driving the lifespan directly rather than through ``TestClient`` is what makes the claim loop observable: it runs
    in this loop, so a test can await the claims it makes instead of hoping a request gave it a turn.
    """
    with patch("services.framework.checker_app.ApiClient", return_value=queue):
        async with app.router.lifespan_context(app):
            yield


async def _settle(handles: list[_FakeHandle], timeout: float = 5.0) -> None:
    """Wait until every job has written its terminal state."""
    await asyncio.wait_for(asyncio.gather(*(handle.settled.wait() for handle in handles)), timeout)


def _process_endpoint(app: FastAPI) -> Callable[[StubCloudRequest], Coroutine[Any, Any, Response]]:
    """The ``/process`` handler itself, so a test can cancel it or put a deadline on it the way a caller does."""
    route = next(route for route in app.routes if isinstance(route, APIRoute) and route.path == "/process")
    endpoint: Callable[[StubCloudRequest], Coroutine[Any, Any, Response]] = route.endpoint
    return endpoint


class TestClaiming(unittest.IsolatedAsyncioTestCase):
    """Work arrives by being claimed: the replica asks for jobs, runs them, and reports through their rows."""

    async def test_a_replica_runs_work_it_was_never_sent(self) -> None:
        app, service, _ = _build(settings=_coordinating_settings())
        handle = _FakeHandle(request_json={"source_url": "s3://bucket/a.mp4"})
        queue = _FakeQueue([handle])

        async with _running(app, queue):
            await _settle([handle])

        self.assertEqual(len(service.processed), 1)
        self.assertIsNone(handle.failure)

    async def test_the_claim_names_the_checkers_own_queue(self) -> None:
        """A pod claiming under the wrong name idles while its work waits for somebody who asks for it."""
        app, _, _ = _build(settings=_coordinating_settings())
        queue = _FakeQueue()

        async with _running(app, queue):
            await asyncio.wait_for(queue.exhausted.wait(), 5.0)

        self.assertEqual(queue.claims[0]["checker_name"], "test.stub_checker")

    async def test_the_claim_carries_the_replica_s_own_build(self) -> None:
        """The claimed row records the version and commit of the pod that runs the job, not of any other replica."""
        app, _, _ = _build(settings=_coordinating_settings())
        queue = _FakeQueue()

        with patch("services.utils.get_git_sha", return_value="abc123"):
            async with _running(app, queue):
                await asyncio.wait_for(queue.exhausted.wait(), 5.0)

        self.assertEqual(queue.claims[0]["version"], "1.2.3")
        self.assertEqual(queue.claims[0]["commit_sha"], "abc123")

    async def test_a_replica_asks_for_exactly_the_work_it_has_room_for(self) -> None:
        """The batch size is the free-slot count, which is what makes per-pod capacity self-enforcing."""
        app, _, _ = _build(settings=_coordinating_settings(), concurrency_limit=3)
        queue = _FakeQueue()

        async with _running(app, queue):
            await asyncio.wait_for(queue.exhausted.wait(), 5.0)

        self.assertEqual(queue.claims[0]["limit"], 3)

    async def test_an_idle_pod_waiting_on_the_queue_reports_itself_free(self) -> None:
        """A pod that reserved its capacity for the length of the wait would look full while doing nothing at all."""
        app, _, _ = _build(settings=_coordinating_settings(), concurrency_limit=2)
        queue = _FakeQueue()

        async with _running(app, queue):
            await asyncio.wait_for(queue.exhausted.wait(), 5.0)
            self.assertEqual(app.state.slots.active, 0)

    async def test_a_claimed_job_holds_a_slot_while_it_runs(self) -> None:
        app, service, _ = _build(settings=_coordinating_settings(), concurrency_limit=2)
        service.block = asyncio.Event()
        handle = _FakeHandle()
        queue = _FakeQueue([handle])

        async with _running(app, queue):
            await asyncio.wait_for(queue.exhausted.wait(), 5.0)
            self.assertEqual(app.state.slots.active, 1)
            service.block.set()
            await _settle([handle])

    async def test_the_row_records_the_same_envelope_a_direct_call_would_return(self) -> None:
        """Everything downstream reads ``response_json.data``, so the row must not depend on how the job arrived."""
        app, _, _ = _build(settings=_coordinating_settings())
        handle = _FakeHandle()
        queue = _FakeQueue([handle])

        async with _running(app, queue):
            await _settle([handle])

        assert handle.completed is not None
        self.assertEqual(handle.completed["data"]["clip_id"], "clip-1")
        self.assertIn("duration_ms", handle.completed["metadata"])

    async def test_a_finished_job_releases_its_slot_and_deletes_its_files(self) -> None:
        app, service, temp_dirs = _build(settings=_coordinating_settings())
        handle = _FakeHandle()
        queue = _FakeQueue([handle])

        async with _running(app, queue):
            await _settle([handle])

        self.assertEqual(app.state.slots.active, 0)
        self.assertEqual(len(temp_dirs), 1)
        self.assertFalse(temp_dirs[0].exists())
        self.assertEqual(len(service.cleaned), 1)

    async def test_a_job_whose_payload_cannot_be_read_is_failed_rather_than_left_for_the_next_pod(self) -> None:
        app, service, _ = _build(settings=_coordinating_settings())
        handle = _FakeHandle(request_json={"source_url": ["not", "a", "url"]})
        queue = _FakeQueue([handle])

        async with _running(app, queue):
            await _settle([handle])

        self.assertEqual(service.processed, [])
        self.assertIsNotNone(handle.failure)
        self.assertEqual(app.state.slots.active, 0, "the rejected job's slot must come back")

    async def test_a_rejected_payload_is_described_without_echoing_back_what_was_rejected(self) -> None:
        """The recorded reason names the field and the problem. It must not carry the presigned URL that failed."""
        app, _, _ = _build(settings=_coordinating_settings())
        secret = "https://bucket.s3.amazonaws.com/clip.mp4?X-Amz-Signature=deadbeef"
        handle = _FakeHandle(request_json={"source_url": {"nested": secret}})
        queue = _FakeQueue([handle])

        async with _running(app, queue):
            await _settle([handle])

        self.assertIsNotNone(handle.failure)
        self.assertIn("source_url", str(handle.failure))
        self.assertNotIn(secret, str(handle.failure))

    async def test_a_job_cancelled_while_it_was_queued_is_never_started(self) -> None:
        """The claim carries the cancel forward, so the pod records it instead of running work nobody wants."""
        app, service, _ = _build(settings=_coordinating_settings())
        handle = _FakeHandle(request_json={"source_url": "s3://bucket/a.mp4"}, cancel_requested=True)
        queue = _FakeQueue([handle])

        async with _running(app, queue):
            await _settle([handle])

        self.assertEqual(service.processed, [])
        self.assertEqual(handle.failure, "Cancelled before completion")
        self.assertEqual(app.state.slots.active, 0, "the cancelled job's slot must come back")

    async def test_a_failing_pipeline_is_recorded_on_the_row_rather_than_lost(self) -> None:
        app, _, _ = _build(settings=_coordinating_settings())
        handle = _FakeHandle(request_json={"fail_with": "boom"})
        queue = _FakeQueue([handle])

        async with _running(app, queue):
            await _settle([handle])

        self.assertIsNotNone(handle.failure)
        self.assertIn("unexpected", str(handle.failure))

    async def test_a_claim_that_fails_is_retried_rather_than_ending_the_loop(self) -> None:
        """A database blip must not leave a pod permanently unable to take work once the blip is over."""
        app, service, _ = _build(settings=_coordinating_settings())
        handle = _FakeHandle()
        queue = _FakeQueue([handle])
        queue.fail_next = 1

        with patch("services.framework.checker_app._CLAIM_RETRY_SECONDS", 0.01):
            async with _running(app, queue):
                await _settle([handle])

        self.assertGreaterEqual(len(queue.claims), 2)
        self.assertEqual(len(service.processed), 1)

    async def test_shutdown_stops_claiming_before_it_waits_for_running_work(self) -> None:
        """Claiming during a drain commits the replica to another clip it has no time left to finish."""
        app, _, _ = _build(settings=_coordinating_settings())
        queue = _FakeQueue()

        async with _running(app, queue):
            await asyncio.wait_for(queue.exhausted.wait(), 5.0)
            claims_before_shutdown = len(queue.claims)

        self.assertIsNone(app.state.claim_task)
        self.assertEqual(len(queue.claims), claims_before_shutdown)

    async def test_nothing_is_claimed_when_no_queue_is_configured(self) -> None:
        """Local runs and tests have no database, and must still serve ``/process``."""
        app, _, _ = _build()
        queue = _FakeQueue()

        async with _running(app, queue):
            await asyncio.sleep(0)

        self.assertEqual(queue.claims, [])


class TestDrain(unittest.IsolatedAsyncioTestCase):
    """Shutdown gives detached jobs a bounded chance to record their outcome."""

    async def test_shutdown_waits_for_a_running_job_to_record_its_outcome(self) -> None:
        app, service, _ = _build(settings=_coordinating_settings())
        service.block = asyncio.Event()
        handle = _FakeHandle()
        queue = _FakeQueue([handle])

        async with _running(app, queue):
            await asyncio.wait_for(queue.exhausted.wait(), 5.0)
            # Let the job finish only once the drain is already waiting on it.
            asyncio.get_running_loop().call_soon(service.block.set)

        self.assertIsNotNone(handle.completed)

    async def test_a_job_that_outlasts_the_grace_period_does_not_hold_shutdown_open(self) -> None:
        """A checker job can legitimately run for twenty minutes, and SIGTERM does not wait that long."""
        app, service, _ = _build(settings=_coordinating_settings())
        service.block = asyncio.Event()
        handle = _FakeHandle()
        queue = _FakeQueue([handle])

        drain = checker_app._drain_jobs

        async def brief_drain(target: FastAPI) -> None:
            await drain(target, grace_seconds=0.05)

        with patch.object(checker_app, "_drain_jobs", brief_drain):
            async with _running(app, queue):
                await asyncio.wait_for(queue.exhausted.wait(), 5.0)
                jobs = set(app.state.jobs)

        self.assertIsNone(handle.completed, "shutdown blocked on a job that had not finished")
        # Let the abandoned job end so this loop closes cleanly. In production it simply stops heartbeating and the
        # reaper puts it back on the queue.
        service.block.set()
        await asyncio.wait_for(asyncio.gather(*jobs), 5.0)


class TestSlotRelease(unittest.IsolatedAsyncioTestCase):
    """A slot comes back however the work holding it ends: success, error, timeout or cancellation.

    A leaked slot is permanent. The replica runs below its limit until it restarts, and a single-slot replica stops
    taking work altogether while still answering ``/health`` as if nothing were wrong.
    """

    async def test_a_direct_request_that_fails_or_times_out_returns_its_slot(self) -> None:
        for failure in ("boom", "timeout"):
            with self.subTest(failure=failure):
                app, _, _ = _build(concurrency_limit=1)
                async with _running(app, _FakeQueue()):
                    response = await _process_endpoint(app)(StubCloudRequest(fail_with=failure))

                    self.assertEqual(response.status_code, 500)
                    self.assertEqual(app.state.slots.active, 0)

    async def test_a_direct_request_cancelled_mid_pipeline_returns_its_slot(self) -> None:
        """A client disconnect cancels the handler, which never reaches anything after the await it is suspended in."""
        app, service, temp_dirs = _build(concurrency_limit=1)
        service.block = asyncio.Event()

        async with _running(app, _FakeQueue()):
            request = asyncio.create_task(_process_endpoint(app)(StubCloudRequest()))
            await asyncio.wait_for(service.running.wait(), 5.0)
            self.assertEqual(app.state.slots.active, 1)

            request.cancel()
            await asyncio.gather(request, return_exceptions=True)

            self.assertEqual(app.state.slots.active, 0)
        self.assertFalse(temp_dirs[0].exists())

    async def test_a_direct_request_that_overruns_a_deadline_returns_its_slot(self) -> None:
        app, service, _ = _build(concurrency_limit=1)
        service.block = asyncio.Event()

        async with _running(app, _FakeQueue()):
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(_process_endpoint(app)(StubCloudRequest()), timeout=0.05)

            self.assertEqual(app.state.slots.active, 0)

    async def test_a_claimed_job_that_fails_or_times_out_returns_its_slot(self) -> None:
        for failure in ("boom", "timeout"):
            with self.subTest(failure=failure):
                app, _, _ = _build(settings=_coordinating_settings())
                handle = _FakeHandle(request_json={"fail_with": failure})
                queue = _FakeQueue([handle])

                async with _running(app, queue):
                    await _settle([handle])

                self.assertIsNotNone(handle.failure)
                self.assertEqual(app.state.slots.active, 0)

    async def test_a_claimed_job_cancelled_mid_flight_returns_its_slot(self) -> None:
        app, _, temp_dirs = _build(settings=_coordinating_settings())
        handle = _StallingHandle()
        queue = _FakeQueue([handle])

        async with _running(app, queue):
            await asyncio.wait_for(handle.writing.wait(), 5.0)
            jobs = set(app.state.jobs)
            self.assertEqual(app.state.slots.active, 1)

            for job in jobs:
                job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)

        self.assertEqual([job.cancelled() for job in jobs], [True])
        self.assertEqual(app.state.slots.active, 0)
        self.assertFalse(temp_dirs[0].exists())

    async def test_a_claimed_job_cancelled_before_it_first_runs_returns_its_slot(self) -> None:
        """A task cancelled before its first step never enters its coroutine, so no ``finally`` inside it runs."""
        for case, request_json in (("runnable", {}), ("rejected", {"source_url": ["not", "a", "url"]})):
            with self.subTest(case=case):
                await self._cancel_before_first_step(_FakeHandle(request_json=request_json))

    async def _cancel_before_first_step(self, handle: _FakeHandle) -> None:
        app, service, _ = _build(settings=_coordinating_settings())
        queue = _FakeQueue([handle])
        cancelled: list[asyncio.Task[None]] = []

        def cancel_unstarted_jobs() -> None:
            for job in app.state.jobs:
                job.cancel()
                cancelled.append(job)

        queue.on_claim = cancel_unstarted_jobs

        async with _running(app, queue):
            await asyncio.wait_for(queue.exhausted.wait(), 5.0)

        self.assertEqual([job.cancelled() for job in cancelled], [True])
        self.assertEqual(app.state.slots.active, 0)
        # Neither the pipeline nor the rejection write ran, so the job really was cancelled before its first step.
        self.assertEqual(service.processed, [])
        self.assertIsNone(handle.failure)


class TestStartup(unittest.TestCase):
    """Lifespan wiring, including the checks a checker asks to fail fast on."""

    def test_a_failed_startup_check_prevents_the_app_from_serving(self) -> None:
        def refuse() -> None:
            raise RuntimeError("CUDA unavailable")

        app, _, _ = _build(verify_startup=refuse)

        with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"), TestClient(app):
            pass

    def test_startup_publishes_collaborators_and_shutdown_clears_them(self) -> None:
        app, service, _ = _build()
        with TestClient(app) as client:
            client.get("/health")
            self.assertIs(app.state.service, service)
            self.assertIsNotNone(app.state.input_storage_provider)
            self.assertIsNotNone(app.state.output_storage_provider)

        self.assertIsNone(app.state.input_storage_provider)
        self.assertIsNone(app.state.output_storage_provider)

    def test_the_claim_loop_is_started_when_a_queue_is_configured(self) -> None:
        app, _, _ = _build(settings=_coordinating_settings())
        with patch("services.framework.checker_app.ApiClient", return_value=_FakeQueue()), TestClient(app) as client:
            body = client.get("/health").json()["data"]

        self.assertTrue(body["coordination"])
        self.assertTrue(body["claiming"])


class TestGpuFailure(unittest.IsolatedAsyncioTestCase):
    """What a replica does when its GPU dies under it.

    The behaviour under test is the difference between losing a job and losing a pod. A poisoned CUDA context does
    not heal, so a replica that keeps claiming after one becomes a sink: it draws work out of the shared queue in
    order to fail it, faster than the healthy replicas can drain the rest.

    ``raise_signal`` is patched throughout, for the obvious reason that the code under test is trying to terminate
    the process running the tests.
    """

    async def test_a_gpu_failure_stops_the_replica_claiming_and_asks_for_a_restart(self) -> None:
        app, _, _ = _build(settings=_coordinating_settings())
        handle = _FakeHandle(request_json={"source_url": "s3://bucket/a.mp4", "fail_with": "gpu"})
        queue = _FakeQueue([handle])

        with patch("services.framework.checker_app.signal.raise_signal") as raise_signal:
            async with _running(app, queue):
                await _settle([handle])
                await asyncio.wait_for(asyncio.shield(app.state.retire_task), 5.0)

                self.assertFalse(app.state.gpu_health.is_healthy)
                self.assertIsNone(app.state.claim_task)
                raise_signal.assert_called_once_with(signal.SIGTERM)

        # The job itself is still recorded as failed rather than abandoned, so the arbitrator hears about it instead
        # of waiting out a lease on a pod that is leaving.
        self.assertIsNotNone(handle.failure)

    async def test_the_failing_job_is_still_recorded_before_the_replica_leaves(self) -> None:
        """Retirement must not be a way for a job to vanish: the row is the only thing the arbitrator can see."""
        app, _, _ = _build(settings=_coordinating_settings())
        handle = _FakeHandle(request_json={"source_url": "s3://bucket/a.mp4", "fail_with": "gpu"})
        queue = _FakeQueue([handle])

        with patch("services.framework.checker_app.signal.raise_signal"):
            async with _running(app, queue):
                await _settle([handle])
                await asyncio.wait_for(asyncio.shield(app.state.retire_task), 5.0)

        self.assertIn("cublasCreate", str(handle.failure))

    async def test_a_probe_that_passes_keeps_the_replica_in_service(self) -> None:
        """One job's GPU failure is not evidence the GPU is gone, and a working probe is the cheap way to tell.

        Without this the first transient failure would cost a replica, which over a fleet is a worse trade than the
        occasional failed job it was meant to prevent.
        """
        app, _, _ = _build(settings=_coordinating_settings(), probe_gpu=lambda: None)
        handle = _FakeHandle(request_json={"source_url": "s3://bucket/a.mp4", "fail_with": "gpu"})
        queue = _FakeQueue([handle])

        with patch("services.framework.checker_app.signal.raise_signal") as raise_signal:
            async with _running(app, queue):
                await _settle([handle])

                self.assertTrue(app.state.gpu_health.is_healthy)
                self.assertIsNone(app.state.retire_task)
                self.assertIsNotNone(app.state.claim_task)
                raise_signal.assert_not_called()

    async def test_a_probe_that_fails_confirms_the_replica_is_finished(self) -> None:
        def dead_gpu() -> None:
            raise RuntimeError("CUDA error: unspecified launch failure")

        app, _, _ = _build(settings=_coordinating_settings(), probe_gpu=dead_gpu)
        handle = _FakeHandle(request_json={"source_url": "s3://bucket/a.mp4", "fail_with": "gpu"})
        queue = _FakeQueue([handle])

        with patch("services.framework.checker_app.signal.raise_signal") as raise_signal:
            async with _running(app, queue):
                await _settle([handle])
                await asyncio.wait_for(asyncio.shield(app.state.retire_task), 5.0)

                self.assertFalse(app.state.gpu_health.is_healthy)
                raise_signal.assert_called_once_with(signal.SIGTERM)

    async def test_a_probe_that_never_answers_is_not_treated_as_one_that_passed(self) -> None:
        """A wedged device can leave an allocation or synchronize waiting as long as the driver allows.

        Unbounded, that await would strand the replica with the latch untripped and the claim loop still running --
        every job it goes on to claim failing -- which is the exact state retirement exists to end.
        """
        released = threading.Event()
        self.addCleanup(released.set)

        def wedged_gpu() -> None:
            # Bounded so a regression cannot leave a non-daemon thread behind; cleanup releases it immediately.
            released.wait(timeout=30.0)

        app, _, _ = _build(settings=_coordinating_settings(), probe_gpu=wedged_gpu)
        handle = _FakeHandle(request_json={"source_url": "s3://bucket/a.mp4", "fail_with": "gpu"})
        queue = _FakeQueue([handle])

        with (
            patch("services.framework.checker_app._GPU_PROBE_TIMEOUT_SECONDS", 0.05),
            patch("services.framework.checker_app.signal.raise_signal") as raise_signal,
        ):
            async with _running(app, queue):
                await _settle([handle])
                await asyncio.wait_for(asyncio.shield(app.state.retire_task), 5.0)

                self.assertFalse(app.state.gpu_health.is_healthy)
                raise_signal.assert_called_once_with(signal.SIGTERM)

    async def test_an_ordinary_failure_does_not_cost_the_replica(self) -> None:
        """Only a GPU verdict retires a pod. A bad clip is a bad clip, however many of them arrive."""
        app, _, _ = _build(settings=_coordinating_settings())
        handle = _FakeHandle(request_json={"source_url": "s3://bucket/a.mp4", "fail_with": "boom"})
        queue = _FakeQueue([handle])

        with patch("services.framework.checker_app.signal.raise_signal") as raise_signal:
            async with _running(app, queue):
                await _settle([handle])

                self.assertTrue(app.state.gpu_health.is_healthy)
                raise_signal.assert_not_called()


class TestGpuFailureSurface(unittest.TestCase):
    """What the outside world can see once a replica's GPU has been declared unusable."""

    def test_health_reports_the_reason_rather_than_just_a_status(self) -> None:
        """The pod is usually gone by the time anyone asks, so the reason is the only evidence of which death it was."""
        app, _, _ = _build()
        app.state.gpu_health.trip("CUBLAS failure 14")

        with TestClient(app) as client:
            body = client.get("/health").json()["data"]

        self.assertEqual(body["status"], "unhealthy")
        self.assertIs(body["gpu"]["healthy"], False)
        self.assertEqual(body["gpu"]["failure"], "CUBLAS failure 14")

    def test_healthy_replicas_still_say_so(self) -> None:
        app, _, _ = _build()

        with TestClient(app) as client:
            body = client.get("/health").json()["data"]

        self.assertEqual(body["status"], "healthy")
        self.assertIs(body["gpu"]["healthy"], True)
        self.assertIsNone(body["gpu"]["failure"])

    def test_process_refuses_with_503_rather_than_taking_work_it_cannot_do(self) -> None:
        """503 because the request is fine and would succeed elsewhere, which is advice the caller can act on."""
        app, service, _ = _build()
        app.state.gpu_health.trip("CUBLAS failure 14")

        with TestClient(app) as client:
            response = client.post("/process", json={"source_url": "s3://bucket/a.mp4"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(service.processed, [])

    def test_metrics_expose_the_one_replica_state_occupancy_cannot_show(self) -> None:
        """A pod whose GPU died reads as idle, exactly like a pod waiting on an empty queue."""
        app, _, _ = _build()

        with TestClient(app) as client:
            healthy = client.get("/metrics").text
            app.state.gpu_health.trip("CUBLAS failure 14")
            tripped = client.get("/metrics").text

        self.assertIn('checker_gpu_healthy{checker="test.stub_checker"} 1', healthy)
        self.assertIn('checker_gpu_healthy{checker="test.stub_checker"} 0', tripped)


if __name__ == "__main__":
    unittest.main()
