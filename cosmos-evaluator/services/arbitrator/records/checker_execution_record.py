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

"""Checker execution record for tracking individual checker executions via the Database REST API."""

from dataclasses import dataclass, field
import logging
from typing import Any

from database.api_client import ApiClient

logger = logging.getLogger(__name__)


@dataclass
class CheckerExecutionRecord:
    """Tracks execution of a single checker via the Database REST API.

    Lifecycle (matches ``checker_executions.status``):

    - ``save()`` — INSERT; the server creates the row as ``status='pending'``, carrying whatever payload admission could
      assemble. Not claimable: the job exists but has not been offered.
    - ``enqueue()`` — write the payload the pod will run and move the row to ``queued``, which together are what
      actually offer the job to the fleet.
    - ``complete()`` / ``fail()`` / ``timeout()`` / ``skipped()`` — terminal states.

    The ``in_progress`` step has no method here on purpose. The arbitrator observes claims by polling every job it owns
    at once, so that write is issued for the whole pass by ``ApiClient.mark_checker_executions_in_progress`` rather than
    per record. A per-record version would put one round trip per job back into a pass that is otherwise two calls wide.

    For per-camera fan-out checkers (e.g. single-view), ``checker_name`` stores the base name (e.g.
    ``av.objects_single_view``) and ``camera_name`` holds the full RDS camera stream id (e.g.
    ``camera_front_wide_120fov``). Non-parameterized checkers, such as cross-camera aggregators, leave
    ``camera_name`` as ``None``.

    The row's ``version`` / ``commit_sha`` are not this record's to write: they stay empty until a checker pod claims
    the job and stamps its own build, so an empty value means no pod ever ran it.
    """

    api_client: ApiClient
    request_id: int
    checker_name: str
    request_json: dict[str, Any]
    camera_name: str | None = None
    execution_id: int | None = field(default=None, init=False)

    @classmethod
    def attach(
        cls,
        api_client: ApiClient,
        request_id: int,
        execution_id: int,
        checker_name: str,
        request_json: dict[str, Any] | None = None,
        camera_name: str | None = None,
    ) -> "CheckerExecutionRecord":
        """Bind a record to a row that already exists, instead of inserting a new one.

        A replica adopting a request another one was running needs to write to that request's existing rows; calling
        ``save()`` would insert duplicates and leave the originals to be recovered forever.
        """
        record = cls(
            api_client=api_client,
            request_id=request_id,
            checker_name=checker_name,
            request_json=request_json or {},
            camera_name=camera_name,
        )
        record.execution_id = execution_id
        return record

    async def save(self) -> bool:
        """Insert this job's row, which the server creates as ``pending``. Returns whether it was written.

        The status is deliberately not this layer's to pass: a row is claimable only once its payload is on it, and
        that promotion belongs to :meth:`enqueue`, so there is no argument here that could make a job runnable early.

        What is written is the payload admission could assemble -- a real request for a job whose inputs are already
        known, ``{}`` for one still waiting on its dependencies. Informational either way: it exists so the snapshot
        endpoint and an adopting replica can see the job, not so a pod can run it.
        """
        try:
            self.execution_id = await self.api_client.create_checker_execution(
                request_id=self.request_id,
                checker_name=self.checker_name,
                request_json=self.request_json,
                camera_name=self.camera_name,
            )
            if self.execution_id:
                logger.debug("Created checker execution record: %s (id: %s)", self.checker_name, self.execution_id)
                return True
            return False
        except Exception:
            logger.exception("Failed to create checker execution record")
            return False

    async def enqueue(self, request_json: dict[str, Any]) -> bool:
        """Publish the job's payload and leave it ``queued`` for a checker pod to claim. Returns whether it was written.

        The whole handover: writing the payload to the row *is* offering the job, because ``queued`` is the status the
        claim query looks for and the row sat in ``pending`` until now. Nothing is sent anywhere -- an individual pod
        behind a load balancer cannot be addressed, and whichever one has room will find this row.

        Payload and status move in the same update, which is what makes the payload's presence a precondition of
        claimability rather than a race against it. What admission recorded cannot be relied on instead (see
        :meth:`save`): a dependent job's inputs are only knowable once its dependencies have produced results, and
        that is precisely the job whose payload matters most. A job whose payload never reached the row cannot be run
        by anybody, so a caller that sees ``False`` must fail the job rather than leave it sitting in ``pending``.

        An already-terminal row is the one case that returns ``True`` without writing anything. It happens on the
        adoption path, where a replica rebuilds a request and re-offers a job that a pod claimed and finished in the
        meantime: the job needs no payload because it is over, and its result is on the row waiting to be read. That
        is the opposite of a job nobody can run, so it must not be reported as one.
        """
        if not self.execution_id:
            logger.warning(
                "Cannot enqueue checker execution: execution_id is None (request_id: %s, checker: %s)",
                self.request_id,
                self.checker_name,
            )
            return False
        try:
            write = await self.api_client.update_checker_execution(
                execution_id=self.execution_id, status="queued", request_json=request_json
            )
        except Exception:
            logger.exception("Failed to enqueue checker execution %s", self.execution_id)
            return False
        if not write.written:
            logger.info(
                "Not offering checker execution %s: it already finished as '%s'",
                self.execution_id,
                write.existing_status,
            )
            return True
        logger.debug("Enqueued checker execution %s for a pod to claim", self.execution_id)
        return True

    async def complete(
        self,
        response_json: dict[str, Any],
        duration_ms: int,
        output_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update status to 'success' with response. Returns whether the row reached that state."""
        return await self._update_final("success", response_json, duration_ms, output_metadata=output_metadata)

    async def fail(
        self,
        response_json: dict[str, Any],
        duration_ms: int,
        error_message: str | None = None,
    ) -> bool:
        """Update status to 'fail' with error response. Returns whether the row reached that state."""
        return await self._update_final("fail", response_json, duration_ms, error_message=error_message)

    async def timeout(self, error_message: str) -> bool:
        """Update status to 'timeout'. Returns whether the row reached that state."""
        return await self._update_final("timeout", {}, 0, error_message=error_message)

    async def skipped(self, error_message: str | None = None) -> bool:
        """Update status to 'skipped' (a dependency hit a non-success terminal state).

        Returns whether the row reached that state.
        """
        return await self._update_final("skipped", {}, 0, error_message=error_message)

    async def _update_final(
        self,
        status: str,
        response_json: dict[str, Any],
        duration_ms: int,
        error_message: str | None = None,
        output_metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update status with final response data. Returns whether the row actually reached that state.

        Failure is reported rather than raised because a request must still finish for its caller even when the
        database is unreachable. The caller needs the answer regardless: a row that did not reach a terminal state is
        one the reaper will hand to another pod, so treating a failed write as done would silently lose the result.

        A row that already holds a *different* terminal status keeps it, and that is reported as failure too. The
        arbitrator reaches here with a verdict formed from the outside -- most often a job that outlived its budget
        -- while the pod that ran the work writes what actually happened. When the two race, the pod's account is
        the better one, so the refusal is logged with the status that won rather than retried.
        """
        if not self.execution_id:
            logger.warning(
                "Cannot update checker execution to final status '%s': execution_id is None "
                "(request_id: %s, checker: %s)",
                status,
                self.request_id,
                self.checker_name,
            )
            return False
        try:
            write = await self.api_client.update_checker_execution(
                execution_id=self.execution_id,
                status=status,
                response_json=response_json,
                duration_ms=duration_ms,
                error_message=error_message,
                output_metadata=output_metadata,
            )
        except Exception:
            logger.exception("Failed to update checker execution record %s", self.execution_id)
            return False
        if not write.written:
            logger.warning(
                "Not writing '%s' to checker execution %s: it already finished as '%s' (request_id: %s, checker: %s)",
                status,
                self.execution_id,
                write.existing_status,
                self.request_id,
                self.checker_name,
            )
            return False
        logger.info(
            "Updated checker execution %s to final status: %s (request_id: %s, checker: %s)",
            self.execution_id,
            status,
            self.request_id,
            self.checker_name,
        )
        return True
