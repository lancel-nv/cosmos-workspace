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

"""Execution record for tracking arbitrator requests via the Database REST API."""

from dataclasses import dataclass, field
import logging
import time
from typing import Any

from database.api_client import ApiClient

logger = logging.getLogger(__name__)


@dataclass
class ExecutionRecord:
    """Tracks execution of an arbitrator request via the Database REST API.

    Lifecycle (matches ``arbitrator_executions.status``):

    - ``save()`` — INSERT default ``status='queued'``.
    - ``in_progress()`` — flip to ``in_progress`` once a pod is first seen holding a job from this request.
    - ``complete()`` / ``fail()`` — terminal updates with response payload and
      duration.

    ``claimed_by`` names the arbitrator replica orchestrating the request, and every write this record makes carries it
    as a predicate. The aggregation lives in one replica's memory, so if that replica stops renewing its lease another
    adopts the request and rebuilds it; the predicate is what stops the original from later writing its now-stale
    aggregation over the adopter's. A record without a ``claimed_by`` writes unfenced and is never adopted, which is
    the right behavior for a caller that is not an arbitrator replica.
    """

    api_client: ApiClient
    client_id: int
    request_json: dict[str, Any]
    version: str
    commit_sha: str
    metadata: dict[str, Any] | None = None
    claimed_by: str | None = None
    lease_seconds: float | None = None
    request_id: int | None = field(default=None, init=False)
    # Set once a fenced write comes back empty. From then on this record is a bystander: it keeps no claim and must
    # not attempt further writes, because the replica that adopted the request owns the outcome.
    adopted_away: bool = field(default=False, init=False)
    _start_time: float = field(default_factory=time.monotonic, init=False)

    def _duration_ms(self) -> int:
        return int((time.monotonic() - self._start_time) * 1000)

    async def save(self) -> bool:
        """Create execution record with 'queued' status. Returns True if successful."""
        try:
            self.request_id = await self.api_client.create_execution(
                client_id=self.client_id,
                request_json=self.request_json,
                version=self.version,
                commit_sha=self.commit_sha,
                metadata=self.metadata,
                claimed_by=self.claimed_by,
                lease_seconds=self.lease_seconds,
            )
        except Exception:
            logger.exception("Failed to create execution record")
            return False
        else:
            if self.request_id:
                logger.debug("Created execution record with request_id: %s", self.request_id)
                return True
            return False

    def stand_down(self) -> None:
        """Give up the right to write this row, because its outcome is no longer this replica's to record.

        Called when a renewal pass finds the claim gone. Idempotent, and deliberately not a database call: whoever holds
        the row now is the only one entitled to write it, so the correct action here is purely local -- stop trying.
        """
        self.adopted_away = True

    async def in_progress(self) -> None:
        """Flip status from 'queued' to 'in_progress'.

        Fires once per request, when a pod is first seen holding any of this request's jobs. Sets only ``status``;
        ``response_json`` / ``duration_ms`` are left unchanged so the row reflects the in-progress state without a
        placeholder response.
        """
        await self._update_status("in_progress", response_json=None, duration_ms=None)

    async def complete(self, response_json: dict[str, Any]) -> int:
        """Update status to 'success' with response. Returns the recorded duration_ms."""
        duration_ms = self._duration_ms()
        await self._update_status("success", response_json=response_json, duration_ms=duration_ms)
        return duration_ms

    async def fail(self, response_json: dict[str, Any]) -> int:
        """Update status to 'fail' with error response. Returns the recorded duration_ms."""
        duration_ms = self._duration_ms()
        await self._update_status("fail", response_json=response_json, duration_ms=duration_ms)
        return duration_ms

    async def _update_status(
        self,
        status: str,
        response_json: dict[str, Any] | None,
        duration_ms: int | None,
    ) -> None:
        """Persist the status (and optional response / duration) to the DB REST API."""
        if not self.request_id:
            logger.warning(
                "Cannot update execution status to '%s': request_id is None (client_id: %s)",
                status,
                self.client_id,
            )
            return
        if self.adopted_away:
            logger.warning(
                "Not writing status '%s' for execution %s: this replica was superseded and its state is stale",
                status,
                self.request_id,
            )
            return
        try:
            held = await self.api_client.update_execution(
                request_id=self.request_id,
                status=status,
                response_json=response_json,
                duration_ms=duration_ms,
                claimed_by=self.claimed_by,
            )
        except Exception:
            logger.exception("Failed to update execution record %s", self.request_id)
            return
        if not held:
            self.adopted_away = True
            logger.warning(
                "Execution %s is no longer claimed by this replica; discarding the '%s' update", self.request_id, status
            )
            return
        logger.debug("Updated execution record %s to status: %s", self.request_id, status)
