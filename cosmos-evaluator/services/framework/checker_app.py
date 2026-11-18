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

"""One transport implementation for every AV checker REST API.

Every checker runs the same way, differing only in its request and result types and in two genuinely checker-specific
steps: turning a cloud request into local files, and turning a result into an uploaded cloud response. Everything else
-- the claim loop that fetches work, the slot ceiling, the storage providers, the temp-directory lifecycle, cooperative
cancellation, error-to-status mapping, ``/health``, ``/metrics`` and ``/config`` -- is checker-independent, and until
this module existed it was written out three times.

Work arrives by being claimed, not by being sent. The replica's claim loop asks the job queue for as many jobs as it
has free slots, runs each one detached, and reports through the job's row. Nothing about a job travels over the
connection that started it, because no connection starts it: a pod behind a gateway cannot be addressed, so anything
pushed to it is a guess about which pod is free and anything held open across a job is something the gateway will
eventually recycle out from under the work.

A factory rather than a base class, because routes register by decorator against a concrete app and inheritance fits
that badly: a subclass cannot add a route without either re-running the decorators or reaching into the parent's
router. ``build_checker_app`` closes over the spec instead, which also puts every piece of per-request state on
``app.state`` rather than in module globals.

A checker's module becomes: define ``prepare_request`` and ``build_response_data``, hand them to ``build_checker_app``
with its settings and models, and export the resulting ``app``.
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
import logging
from pathlib import Path
import random
import shutil
import signal
import tempfile
import threading
import time
from typing import Any, Protocol

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ValidationError

from database.api_client import ApiClient, CheckerJobHandle
from services import utils
from services.framework.auth import install_api_key_auth
from services.framework.checker_job import LeasePolicy, run_claimed_job, worker_identity
from services.framework.gpu_health import GpuHealth
from services.framework.job_slots import JobSlots
from services.framework.metrics import PROMETHEUS_CONTENT_TYPE, render_checker_metrics
from services.framework.protocols.storage_provider import StorageProvider
from services.framework.request_handlers.json_request_handler import RequestValidationError
from services.framework.response_formatters.json_response_formatter import JsonResponseFormatter
from services.framework.storage_providers.factory import (
    build_input_storage_provider,
    build_output_storage_provider,
)
from services.framework.worker_settings import WorkerDatabaseSettings
from utils.environment import Environment
from utils.gpu import GpuUnavailableError

logger = logging.getLogger(__name__)

# How long a claim is good for and how its owner keeps it alive. The defaults leave room for several consecutive failed
# heartbeats, which ``LeasePolicy`` checks rather than assumes; see it for what each value has to be weighed against.
_LEASE_POLICY = LeasePolicy()

# How long each claim request may be held open waiting for work to appear. An idle pod therefore costs one request per
# window rather than one per second, and work that arrives mid-window is picked up within milliseconds instead of
# waiting for the next poll. The server enforces its own ceiling on this.
_CLAIM_WAIT_SECONDS = 20.0

# How long to wait before asking again after a failed claim, and how much of that is jittered away. Long enough that a
# database outage does not turn the whole fleet into a retry storm against a service that is already struggling, and
# jittered so the fleet does not come back all at once when it recovers.
_CLAIM_RETRY_SECONDS = 5.0
_CLAIM_RETRY_JITTER = 0.5

# How long the GPU probe gets to answer before it is counted as a failure. Generous against a healthy device, where
# the probe is an allocation and a synchronize, and bounded because a wedged one can leave those waiting as long as
# the driver allows. The replica keeps claiming until the probe resolves, so every second here is a second in which
# more jobs can be taken by a pod that may already be finished.
_GPU_PROBE_TIMEOUT_SECONDS = 30.0


class CheckerResult(Protocol):
    """The part of a checker's result the transport needs: where its artifacts were written, if anywhere.

    A read-only property rather than an attribute so a checker that writes nothing can answer ``None`` honestly,
    instead of naming a directory that does not exist to satisfy the type. Checkers whose results carry a plain
    ``output_dir: str`` field satisfy this unchanged.
    """

    @property
    def output_dir(self) -> str | None: ...


class CheckerService[LocalRequestT, ResultT](Protocol):
    """The service surface the transport drives.

    Wider than ``ServiceBase`` in one respect: ``process`` takes a cancellation event. Checkers do their real work in
    an executor thread, which does not stop when the surrounding task is cancelled, so "stop early" has to be
    communicated as a flag the thread polls rather than as task cancellation.
    """

    async def process(self, request_data: LocalRequestT, cancel_event: threading.Event | None = None) -> ResultT: ...

    async def validate_input(self, request_data: LocalRequestT) -> bool: ...

    async def get_default_config(self) -> dict[str, Any]: ...

    async def cleanup(self, output_dir: Path) -> None: ...


@dataclass(frozen=True)
class CheckerSpec[CloudRequestT: BaseModel, LocalRequestT, ResultT: CheckerResult]:
    """Everything that differs between one checker's REST API and another's.

    Args:
        name: Short identifier used for the ``checker`` metrics label and the temp-directory prefix
            (e.g. ``objects_single_view``)
        checker_name: The value this checker's jobs carry in ``checker_executions.checker_name``
            (e.g. ``av.objects_single_view``). This is the queue the claim loop draws from, so it has to match what the
            arbitrator writes exactly -- a mismatch is a pod that idles while its work waits.
        title: FastAPI title, also echoed as ``service`` on ``/health``
        description: FastAPI description
        version: Version string, typically read from the deployment's ``version.txt`` runfile
        config_description: Human-readable blurb returned alongside the defaults on ``/config``
        settings: The checker's resolved settings; supplies the environment, API key, storage configuration and the
            worker credential used to coordinate job rows
        concurrency_limit: Per-replica ceiling on simultaneous jobs, and the batch size the claim loop asks for. Not on
            the settings base, because each checker sizes it against its own memory and GPU footprint.
        cloud_request_model: Pydantic model that request payloads parse into
        build_service: Constructs the service singleton. Called once during lifespan startup, because model loads
            take seconds and must not happen per request.
        prepare_request: Downloads the request's inputs into the supplied temp directory and returns the local
            request. The transport owns that directory's creation and deletion, so this need not clean up after
            itself on failure.
        build_response_data: Uploads the result's artifacts and returns the serialized cloud response.
        verify_startup: Optional fail-fast check run first during startup, for preconditions worth crashing over
            rather than degrading past (e.g. a GPU that is configured but absent).
        probe_gpu: Optional check that the GPU is still usable, called after a job reports it was not. Its job is to
            tell a job-specific failure apart from a dead CUDA context, so a checker that supplies one costs a job
            where it would otherwise cost a replica. Raise to confirm the GPU is gone; return to say it works.
            Called in a worker thread, so it may block. A checker that omits it is taken at its word instead: a
            reported GPU failure is treated as the real thing, since without a probe there is no second opinion.
        response_metadata: Optional extra metadata merged into a successful ``/process`` response envelope.
        after_request: Optional per-request epilogue, run after cleanup on both the success and failure paths.
    """

    name: str
    checker_name: str
    title: str
    description: str
    version: str
    config_description: str
    settings: WorkerDatabaseSettings
    concurrency_limit: int
    cloud_request_model: type[CloudRequestT]
    build_service: Callable[[], CheckerService[LocalRequestT, ResultT]]
    prepare_request: Callable[[CloudRequestT, StorageProvider, Path], Awaitable[LocalRequestT]]
    build_response_data: Callable[[CloudRequestT, ResultT, StorageProvider], Awaitable[dict[str, Any]]]
    verify_startup: Callable[[], None] | None = None
    probe_gpu: Callable[[], None] | None = None
    response_metadata: Callable[[ResultT], dict[str, Any]] | None = None
    after_request: Callable[[], Awaitable[None]] | None = None


@dataclass
class _Run[ResultT: CheckerResult]:
    """Mutable per-request state that the cleanup path needs to see even when the pipeline never reached its end.

    Populated as the pipeline progresses rather than returned at the end, because on cancellation there is no end: the
    cleanup still has to know which temp directory to delete and which output directory to hand back to the service.
    """

    temp_dir: Path | None = None
    result: ResultT | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


def build_checker_app[CloudRequestT: BaseModel, LocalRequestT, ResultT: CheckerResult](
    spec: CheckerSpec[CloudRequestT, LocalRequestT, ResultT],
) -> FastAPI:
    """Assemble a checker's complete REST surface from its spec.

    The service, both storage providers and the database client are constructed once in the lifespan and published on
    ``app.state``, so a test can substitute any of them and a handler never has to guess whether startup has finished.
    """
    json_formatter = JsonResponseFormatter()
    settings = spec.settings

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        logger.info("Initializing %s for %s environment...", spec.title, settings.env.value)

        if spec.verify_startup is not None:
            spec.verify_startup()

        app.state.service = spec.build_service()

        # The stack owns both providers from the moment the first one opens, so a failure anywhere later in startup --
        # the second provider, the database client, the claim loop -- still releases the connections already made
        # instead of leaking them into a process that is about to be restarted anyway.
        async with AsyncExitStack() as storage_stack:
            app.state.input_storage_provider = await storage_stack.enter_async_context(
                build_input_storage_provider(settings)
            )
            app.state.output_storage_provider = await storage_stack.enter_async_context(
                build_output_storage_provider(settings)
            )

            try:
                if settings.database_coordination_enabled:
                    app.state.database_client = ApiClient(settings.database_api_url, settings.database_worker_key)
                    app.state.claim_task = asyncio.create_task(claim_loop())
                    logger.info(
                        "Claiming '%s' jobs from %s, up to %d at a time",
                        spec.checker_name,
                        settings.database_api_url,
                        spec.concurrency_limit,
                    )
                else:
                    logger.info("Job coordination disabled: no database endpoint or worker key configured")

                logger.info("Service initialized successfully")
                yield
            finally:
                logger.info("Shutting down service...")
                # Stop taking work before waiting for what is already running, or the drain races a claim that has
                # just committed this replica to another twenty-minute clip.
                if app.state.claim_task is not None:
                    app.state.claim_task.cancel()
                    await asyncio.gather(app.state.claim_task, return_exceptions=True)
                    app.state.claim_task = None
                # Usually already finished, since it is what raised the SIGTERM that got us here. It is still awaited
                # for the case where shutdown came from outside while a retirement was mid-drain.
                if app.state.retire_task is not None:
                    app.state.retire_task.cancel()
                    await asyncio.gather(app.state.retire_task, return_exceptions=True)
                    app.state.retire_task = None
                # Detached jobs still hold leases and still need the storage clients and the event loop, so they have
                # to reach a recorded conclusion before either goes away. Whatever is still running past the grace
                # period stops heartbeating, and the reaper returns it to the queue for another pod.
                await _drain_jobs(app)
                if app.state.database_client is not None:
                    await app.state.database_client.aclose()
                    app.state.database_client = None

        app.state.input_storage_provider = None
        app.state.output_storage_provider = None
        logger.info("Service shutdown complete")

    app = FastAPI(
        title=spec.title,
        description=spec.description,
        version=spec.version,
        lifespan=lifespan,
        debug=settings.env == Environment.LOCAL,
    )
    app.state.service = None
    app.state.input_storage_provider = None
    app.state.output_storage_provider = None
    app.state.database_client = None
    app.state.claim_task = None
    app.state.slots = JobSlots(spec.concurrency_limit)
    app.state.jobs = set()
    app.state.gpu_health = GpuHealth()
    # Held apart from ``app.state.jobs`` on purpose: retirement waits for that set to empty, so a retirement task
    # inside it would be waiting for itself.
    app.state.retire_task = None

    # No-op unless the environment configures a key.
    install_api_key_auth(app, settings.active_api_key)

    def _service() -> CheckerService[LocalRequestT, ResultT] | None:
        service: CheckerService[LocalRequestT, ResultT] | None = app.state.service
        return service

    def _slots() -> JobSlots:
        slots: JobSlots = app.state.slots
        return slots

    def _gpu_health() -> GpuHealth:
        health: GpuHealth = app.state.gpu_health
        return health

    async def _not_initialized() -> JSONResponse:
        logger.error("Service not initialized")
        return await json_formatter.format_error(Exception("Service not initialized"), status_code=503)

    async def _run_pipeline(cloud_request: CloudRequestT, run: _Run[ResultT]) -> dict[str, Any]:
        """Download, validate, process and upload, recording progress on ``run`` for the cleanup path."""
        service = _service()
        if service is None:
            raise ValueError("Service not initialized")
        input_provider: StorageProvider | None = app.state.input_storage_provider
        if input_provider is None:
            raise ValueError("Input storage provider not initialized")
        output_provider: StorageProvider | None = app.state.output_storage_provider
        if output_provider is None:
            raise ValueError("Output storage provider not initialized")

        run.temp_dir = Path(tempfile.mkdtemp(prefix=f"{spec.name}_"))
        logger.info("Processing request using temp directory: %s", run.temp_dir)

        # Both entry points funnel through here, which is why the GPU verdict is read off this one call rather than
        # from each caller's error handling: a claimed job's failure is recorded by ``run_claimed_job`` and never
        # passes through an ``except`` clause in this module at all.
        try:
            local_request = await spec.prepare_request(cloud_request, input_provider, run.temp_dir)

            logger.info("Validating %s input", spec.name)
            await service.validate_input(local_request)

            logger.info("Processing %s...", spec.name)
            run.result = await service.process(local_request, run.cancel_event)
            return await spec.build_response_data(cloud_request, run.result, output_provider)
        except GpuUnavailableError as exc:
            # The job fails either way; the question this answers is whether the *replica* should survive it.
            await _consider_retirement(exc)
            raise

    async def _consider_retirement(exc: GpuUnavailableError) -> None:
        """Decide whether a reported GPU failure means this replica is finished, and start retiring it if so.

        The asymmetry is deliberate. A GPU failure that clears is a job lost; a GPU failure that does not clear and
        goes unnoticed is every job the pod claims from then on, which is worse than the pod not existing. So the
        probe gets one chance to say the GPU still works, and anything else retires the replica.
        """
        if spec.probe_gpu is not None:
            try:
                # In a thread because a wedged device can make an otherwise instant allocation or synchronize block,
                # and the event loop here is carrying other jobs that still have a chance of finishing. Bounded
                # because that block has no inherent end: an un-timed await would hold this replica in limbo with the
                # latch untripped and the claim loop still running, which is the failure retirement exists to end.
                await asyncio.wait_for(asyncio.to_thread(spec.probe_gpu), timeout=_GPU_PROBE_TIMEOUT_SECONDS)

            # A probe that cannot answer is not a probe that passed, so the timeout joins the failure path rather
            # than getting its own. It is reported separately only because ``TimeoutError`` carries no message.
            # The thread is left to finish on its own: there is no way to interrupt it, and the process is on its
            # way out regardless.
            except TimeoutError:
                logger.error(
                    "GPU probe did not answer within %.0fs after a reported GPU failure; treating it as failed",
                    _GPU_PROBE_TIMEOUT_SECONDS,
                )

            # Deliberately broad: narrowing this would mean listing the exception types every GPU library can fail
            # with -- getting that list wrong re-creates the bug this whole change exists to fix, only with the
            # failure now escaping into a task nobody awaits.
            except Exception as probe_exc:  # noqa: BLE001
                logger.error("GPU probe failed after a reported GPU failure: %s", probe_exc)
            else:
                logger.warning(
                    "A job reported the GPU was unavailable (%s) but the GPU probe passed; "
                    "treating it as specific to that job and staying in service",
                    exc,
                )
                return

        if _gpu_health().trip(str(exc)) and app.state.retire_task is None:
            app.state.retire_task = asyncio.create_task(_retire_replica())

    async def _retire_replica() -> None:
        """Stop claiming, let what is already running finish, then ask to be replaced.

        Self-termination rather than merely reporting unhealthy, because nothing else would act on the report: the
        queue has no notion of an ineligible replica, and a checker's ``/health`` is not wired to a Kubernetes probe
        that would restart the container. Exiting is the one move available that ends with a working pod.

        Via SIGTERM rather than ``os._exit`` so the exit runs the shutdown path the process already has -- lifespan
        teardown, storage clients closed, the database client closed -- and so a job that finishes inside the grace
        period still records its own outcome instead of being reaped on a lease timeout.
        """
        claim_task: asyncio.Task[None] | None = app.state.claim_task
        if claim_task is not None:
            claim_task.cancel()
            await asyncio.gather(claim_task, return_exceptions=True)
            app.state.claim_task = None
            logger.info("Stopped claiming %s work", spec.checker_name)

        await _drain_jobs(app)
        logger.error("Exiting because the GPU is unusable: %s", _gpu_health().failure)
        signal.raise_signal(signal.SIGTERM)

    async def _cleanup(run: _Run[ResultT]) -> None:
        """Delete the request's working files. Safe whether the pipeline finished, failed or never started.

        The checker's own teardown runs first, because a checker that wrote its outputs inside the temp directory --
        which is the usual arrangement, since that is the directory it was handed -- would otherwise be asked to clean
        up a path that no longer exists.
        """
        service = _service()
        output_dir = run.result.output_dir if run.result is not None else None
        if output_dir is not None and service is not None:
            # Best effort, like the rmtree below: this runs from the caller's ``finally``, so an exception here would
            # replace the request's own outcome with a cleanup error.
            try:
                await service.cleanup(Path(output_dir))
            except Exception:
                logger.warning("Error cleaning up output directory: %s", output_dir, exc_info=True)

        if run.temp_dir is not None and run.temp_dir.exists():
            try:
                shutil.rmtree(run.temp_dir)
                logger.info("Cleaned up temp directory: %s", run.temp_dir)
            except OSError as exc:
                logger.warning("Error cleaning up temp directory: %s", exc)

        if spec.after_request is not None:
            # Best effort: an epilogue failure must not mask the request's own outcome.
            try:
                await spec.after_request()
            except Exception:
                logger.warning("%s after_request epilogue failed", spec.name, exc_info=True)

    async def _error_response(exc: Exception, started: float) -> JSONResponse:
        """Map a pipeline failure onto the status code that says whose fault it was.

        A malformed request is 400, a request that parsed but describes something unusable is 422, and anything else is
        ours at 500.
        """
        metadata = {"duration_ms": (time.perf_counter() - started) * 1000}
        if isinstance(exc, RequestValidationError):
            status_code = 400
        elif isinstance(exc, ValueError):
            status_code = 422
        else:
            status_code = 500
        return await json_formatter.format_error(exc, status_code=status_code, metadata=metadata)

    @app.get("/health")
    async def health_check() -> JSONResponse:
        """Health check endpoint.

        Reports this replica's occupancy alongside its identity, and whether it is actually drawing work: a pod that
        answers healthy while ``claiming`` is false will never run anything, which is otherwise invisible from outside.
        Occupancy does not affect the status -- a full pod is a working pod, and its queue is shared with every other
        replica, so there is nothing for a reader to route around.

        An unusable GPU does affect it, and is the only thing that does. It is not a condition to route around either,
        since the replica is already on its way out; the value of saying so is that this is the one place a human
        looking at a pod can see *why* it stopped working, before it disappears and takes its logs with it.
        """
        slots = _slots()
        claim_task: asyncio.Task[None] | None = app.state.claim_task
        health = _gpu_health()
        return await json_formatter.format_success(
            {
                "service": spec.title,
                "status": "healthy" if health.is_healthy else "unhealthy",
                "version": spec.version,
                "git_sha": utils.get_git_sha(),
                "environment": settings.env.value,
                "checker": spec.checker_name,
                "slots": {"active": slots.active, "limit": slots.limit},
                "jobs": len(app.state.jobs),
                "coordination": settings.database_coordination_enabled,
                "claiming": claim_task is not None and not claim_task.done(),
                "gpu": {"healthy": health.is_healthy, "failure": health.failure},
            }
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    async def get_metrics() -> PlainTextResponse:
        """Per-replica occupancy in Prometheus exposition format.

        Scrapers address pods directly rather than through the gateway, so summing across a deployment's targets gives
        the fleet-wide picture that a single ``/health`` response cannot.
        """
        return PlainTextResponse(
            content=render_checker_metrics(
                spec.checker_name, _slots(), len(app.state.jobs), gpu_healthy=_gpu_health().is_healthy
            ),
            media_type=PROMETHEUS_CONTENT_TYPE,
        )

    # The request annotation is a value rather than a name, which is how FastAPI learns the checker's concrete request
    # model from the spec: Python evaluates it to the class at definition time and FastAPI reads it normally. Only mypy
    # objects, because a type position may only hold names it can resolve statically and ``spec`` is a parameter.
    #
    # A suppression-free version does exist: annotate ``CloudRequestT`` and overwrite ``__annotations__["request"]``
    # with the model before registering the route. It is not an improvement. The signature a reader sees is then not
    # the one FastAPI enforces, the route stops being registered where it is defined, and a parameter added later to
    # the signature would silently not be patched. The ignore states on the line what is unusual about the line.
    #
    # Taking a ``dict`` and calling ``model_validate`` in the body is also ignore-free, but it empties the request
    # schema that ``/docs`` is generated from, to be re-supplied by hand via ``openapi_extra``. And a route per checker
    # is the duplication this module exists to remove.
    #
    # Whichever form this takes, ``/process`` has to keep advertising and validating the spec's own model rather than an
    # open dict, which ``test_the_documented_request_body_is_the_checkers_own_model`` is there to hold to.
    @app.post("/process", response_model=dict[str, Any])
    async def process(
        request: spec.cloud_request_model,  # type: ignore[name-defined]
    ) -> JSONResponse:
        """Process one clip and return its result on this connection.

        The direct path, for callers that hold a payload rather than a job row: local runs, smoke tests and anything
        outside the queue. Waits for a free slot rather than refusing when the replica is full -- the caller reached
        this pod through a gateway and cannot choose a different one, so "try elsewhere" is not advice it can take.

        Not how production work arrives. A result that has to come back on this connection is exactly what the queue
        exists to avoid, so anything long-running belongs on a job row instead.
        """
        if _service() is None:
            return await _not_initialized()
        health = _gpu_health()
        if not health.is_healthy:
            # 503 rather than 500: the request is fine and would succeed on another replica, which is what this code
            # tells a caller that can retry. Waiting for a slot first would only delay the same answer.
            return await json_formatter.format_error(
                Exception(f"GPU unusable on this replica: {health.failure}"), status_code=503
            )
        slots = _slots()
        await slots.acquire()
        try:
            return await _run_process(request)
        finally:
            slots.release()

    async def _run_process(cloud_request: CloudRequestT) -> JSONResponse:
        """Execute the pipeline synchronously, once a slot is held."""
        started = time.perf_counter()
        run: _Run[ResultT] = _Run()
        try:
            response_data = await _run_pipeline(cloud_request, run)
            metadata: dict[str, Any] = {"duration_ms": (time.perf_counter() - started) * 1000}
            if spec.response_metadata is not None and run.result is not None:
                metadata.update(spec.response_metadata(run.result))
            return await json_formatter.format_success(data=response_data, metadata=metadata)
        except Exception as exc:
            logger.exception("%s /process failed", spec.name)
            return await _error_response(exc, started)
        finally:
            # Signal the executor thread before cleaning up: on a client disconnect or a shutdown this stops the work
            # at its next checkpoint instead of running the whole clip out. ``service.process`` joins the thread
            # either way, so the delete below cannot race it.
            run.cancel_event.set()
            await _cleanup(run)

    async def claim_loop() -> None:
        """Keep this replica's slots full of claimed jobs, forever.

        The replica's only source of production work. It asks for as many jobs as it has free slots and asks for
        nothing when it has none, which is what makes per-pod capacity self-enforcing: there is no fleet-wide count to
        keep accurate and no way for a caller to hand this pod more than it can run.

        Runs until cancelled at shutdown, and survives everything short of that. A failed claim is a wait and a retry:
        an empty queue and an unreachable database are equally routine from here, and neither is a reason for a pod to
        stop being able to work when the condition clears.

        Public within the closure so tests can drive a bounded number of iterations instead of racing a background
        task.
        """
        client: ApiClient | None = app.state.database_client
        if client is None:
            return
        slots = _slots()
        # Stamped onto every claimed row, so each job records the build that ran it.
        commit_sha = utils.get_git_sha()
        while True:
            if not _gpu_health().is_healthy:
                # Retirement cancels this task, so reaching here means the trip landed while the loop was already
                # awake. Returning rather than waiting to be cancelled keeps the window in which a doomed replica can
                # still claim a job down to zero.
                logger.warning("Not claiming %s work: the GPU is unusable", spec.checker_name)
                return
            await slots.wait_for_room()
            wanted = slots.free
            if wanted == 0:
                continue
            # Slots are reserved for the jobs that come back rather than for the request itself: the wait is held open
            # for up to a claim window, and a pod that reserved its whole capacity for the duration would report itself
            # full while sitting idle.
            try:
                handles = await client.claim_next_checker_jobs(
                    checker_name=spec.checker_name,
                    claimed_by=worker_identity(),
                    lease_seconds=_LEASE_POLICY.lease_seconds,
                    limit=wanted,
                    wait_seconds=_CLAIM_WAIT_SECONDS,
                    version=spec.version,
                    commit_sha=commit_sha,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Could not claim %s work; retrying", spec.checker_name, exc_info=True)
                await asyncio.sleep(_CLAIM_RETRY_SECONDS + random.uniform(0.0, _CLAIM_RETRY_JITTER))
                continue

            for handle in handles:
                _start_job(handle)

    def _start_job(handle: CheckerJobHandle) -> None:
        """Run one claimed job detached, in a slot of its own.

        Detached because nothing is waiting: liveness, cancellation and the result all travel through the job's row, so
        the work outlives whatever asked for it. A payload that will not parse is the job's own failure and is recorded
        as one -- re-queueing it would hand the same unusable row to the next pod, and every pod after that.
        """
        if handle.cancel_requested:
            # Called off while it was still queued. Recording it now costs nothing; starting the pipeline first would
            # mean burning a full heartbeat interval of work on a result nobody will read.
            logger.info("Claimed job %s was already cancelled; not starting it", handle.execution_id)
            _reject_job(handle, "Cancelled before completion")
            return

        try:
            if handle.request_json is None:
                raise ValueError(f"Checker execution {handle.execution_id} has no recorded request payload")
            cloud_request = spec.cloud_request_model.model_validate(handle.request_json)
        except ValidationError as exc:
            reason = _describe_validation_error(exc)
            logger.error("Claimed job %s cannot be run: %s", handle.execution_id, reason)
            _reject_job(handle, reason)
            return
        except ValueError as exc:
            logger.error("Claimed job %s cannot be run: %s", handle.execution_id, exc)
            _reject_job(handle, str(exc))
            return

        run: _Run[ResultT] = _Run()

        async def _job_pipeline() -> dict[str, Any]:
            """Run the pipeline and shape its result the way ``/process`` shapes a response body.

            The row's ``response_json`` is read by the arbitrator and by anything else consuming results out of the
            database, all of which expect the ``{"data": ..., "metadata": ...}`` envelope. Building it here keeps the
            row's shape independent of how the job reached this pod, so nothing downstream has to care.
            """
            started = time.perf_counter()
            data = await _run_pipeline(cloud_request, run)
            metadata: dict[str, Any] = {"duration_ms": (time.perf_counter() - started) * 1000}
            if spec.response_metadata is not None and run.result is not None:
                metadata.update(spec.response_metadata(run.result))
            return {"data": data, "metadata": metadata}

        async def _finish() -> None:
            run.cancel_event.set()
            await _cleanup(run)

        _spawn_job(run_claimed_job(handle, _job_pipeline, run.cancel_event, _finish, _LEASE_POLICY))

    def _reject_job(handle: CheckerJobHandle, error: str) -> None:
        """Record a claimed job as failed without running it."""

        async def _fail() -> None:
            try:
                await handle.fail(error, duration_ms=0)
            except Exception:
                logger.warning("Could not record the rejection of job %s", handle.execution_id, exc_info=True)

        _spawn_job(_fail())

    def _spawn_job(job: Coroutine[Any, Any, None]) -> None:
        """Run a claimed job detached, holding one slot from now until its task ends, however it ends.

        The slot is returned by a done callback on the task rather than by a ``finally`` inside ``job``. A task
        cancelled before its first step never enters its coroutine, so no ``finally`` in it runs and the slot would be
        held for the life of the process; a done callback fires on every terminal state -- returned, raised, or
        cancelled, started or not. The slot is reserved only after the task exists, which no job can observe because
        the task cannot start before this returns, so nothing between the reservation and its release callback can
        raise and strand it.

        The task is also held in ``app.state.jobs``, so the loop cannot collect it and shutdown can wait for it.
        """
        task = asyncio.create_task(job)
        slots = _slots()
        slots.commit()
        task.add_done_callback(lambda _: slots.release())
        app.state.jobs.add(task)
        task.add_done_callback(app.state.jobs.discard)

    @app.get("/config")
    async def get_default_config() -> JSONResponse:
        """Return the checker's default configuration, which is also what its CLI defaults to."""
        service = _service()
        if service is None:
            return await _not_initialized()
        return await json_formatter.format_success(
            {"default_config": await service.get_default_config(), "description": spec.config_description}
        )

    return app


def _describe_validation_error(exc: ValidationError) -> str:
    """Summarize a payload rejection by field and reason, without repeating the values that were rejected.

    A checker request carries presigned URLs, and the default rendering of a ``ValidationError`` echoes the offending
    input back. That text would land in the pod's logs and in the job row's ``error_message``, both of which are read
    by people who have no business holding those credentials, and neither of which expires when the URL does. The
    field path and the reason are what makes the failure diagnosable; the value is not.
    """
    errors = exc.errors(include_url=False, include_input=False, include_context=False)
    fields = [f"{'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}" for err in errors]
    return f"Invalid request payload ({len(fields)} error(s)): " + "; ".join(fields)


async def _drain_jobs(app: FastAPI, grace_seconds: float = 30.0) -> None:
    """Give detached jobs a bounded chance to record their outcome before the process goes away.

    Bounded because a checker job can legitimately run for twenty minutes and SIGTERM does not wait that long. A job
    still running when the grace period ends simply stops heartbeating, and the reaper returns it to the queue once the
    lease expires -- which is exactly the recovery path a pod being killed outright already exercises.
    """
    jobs: set[asyncio.Task[None]] = app.state.jobs
    if not jobs:
        return
    logger.info("Waiting up to %.0fs for %d in-flight job(s) to finish", grace_seconds, len(jobs))
    _, pending = await asyncio.wait(set(jobs), timeout=grace_seconds)
    if pending:
        logger.warning("%d job(s) did not finish before shutdown; their leases will expire", len(pending))
