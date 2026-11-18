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

"""
Per-request checker job lifecycle management for the Arbitrator.

Tracks each checker job through its in-memory state machine, exposes simple iteration helpers used by the
waiting-room scheduler, and cascades dependency failures. This module schedules nothing itself: a job leaves it
the moment its payload is on the queue, and the states past that point are the arbitrator's read of what the
checker fleet has done with it.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
import enum
from typing import Any


class JobState(enum.Enum):
    PENDING = "pending"  # has unsatisfied deps
    READY = "ready"  # dependencies met, eligible to be put on the queue
    QUEUED = "queued"  # on the queue, waiting for a pod to claim it
    IN_PROGRESS = "in_progress"  # a pod claimed it and is processing
    COMPLETED = "completed"  # terminal success
    FAILED = "failed"  # terminal failure (cascades to dependents)
    SKIPPED = "skipped"  # terminal: a dependency failed


_TERMINAL_STATES = frozenset({JobState.COMPLETED, JobState.FAILED, JobState.SKIPPED})


@dataclass
class CheckerJob:
    """A single checker job tracked by the queue.

    ``base_name`` identifies the underlying checker (e.g. ``av.objects_single_view``) and is used for endpoint / health
    lookups. ``name`` is the queue-unique job identity: for per-camera jobs it includes a camera-scoped suffix
    (e.g. ``av.objects_single_view[camera_front_wide_120fov]``) so that fan-out jobs dedupe against each other. For
    aggregator jobs, ``name == base_name`` and ``params`` is empty.
    """

    name: str
    priority: int  # lower number = higher priority; typically topological depth
    deps: set[str]  # names of checker jobs this job depends on
    base_name: str = ""  # Backing checker service name (for endpoint lookups). Defaults to name.
    params: dict[str, str] = field(default_factory=dict)  # Per-job parameters (e.g. camera_name).
    state: JobState = field(default=JobState.PENDING, init=True)
    result: Any = field(default=None, init=False)
    error: str | None = field(default=None, init=False)
    attempt: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.base_name:
            self.base_name = self.name


class CheckerQueue:
    """Manages the lifecycle of all checker jobs for a single Arbitrator request.

    Jobs progress through a state machine:
      PENDING -> READY (when all deps complete)
      READY -> QUEUED (once the job's payload is on the queue)
      QUEUED -> IN_PROGRESS (a pod has claimed it)
      IN_PROGRESS -> COMPLETED (the pod wrote a result)
      IN_PROGRESS -> FAILED (the pod wrote an error, or the job outlived its budget)
      FAILED -> cascades SKIPPED to all transitive dependents

    Past QUEUED the states are read, not driven: the arbitrator learns of each move by polling the job's row,
    and a job that is handed to a second pod after the first one died stays where it is, because from here nothing
    about it has changed -- it is still outstanding and its result will still arrive on the same row.
    """

    def __init__(self, jobs: list[CheckerJob] | None = None) -> None:
        self._jobs: dict[str, CheckerJob] = {}

        # Reverse dependency map: name -> set of job names that depend on it
        self._dependents: dict[str, set[str]] = {}

        if jobs:
            self.add_jobs(jobs)

    def add_jobs(self, jobs: list[CheckerJob]) -> None:
        """Add jobs to the queue using union-merge semantics.

        If a job with the same ``name`` is already present, the incoming job is treated as a no-op (existing job is
        preserved). This lets fan-out builders safely emit overlapping parameterized jobs (e.g. single-view jobs
        requested directly plus single-view jobs synthesized as an aggregator's dependencies) without deduplicating
        ahead of time.

        The initial state of a newly added job is set here:
          - PENDING if it has dependencies, READY otherwise.
        Callers that want to inject jobs in a terminal state (e.g. SKIPPED) should mutate the job's ``state`` (and
        ``error``) immediately after calling ``add_jobs``.
        """
        new_jobs: list[CheckerJob] = []
        for job in jobs:
            if job.name in self._jobs:
                continue
            self._jobs[job.name] = job
            self._dependents.setdefault(job.name, set())
            new_jobs.append(job)

        for job in new_jobs:
            for dep in job.deps:
                self._dependents.setdefault(dep, set()).add(job.name)

        for job in new_jobs:
            job.state = JobState.PENDING if job.deps else JobState.READY

    def get_job(self, name: str) -> CheckerJob:
        """Return a job by name."""
        return self._jobs[name]

    def iter_jobs(self) -> Iterator[CheckerJob]:
        """Iterate all jobs in insertion order. Used by the waiting-room scheduler."""
        return iter(self._jobs.values())

    def get_ready_jobs(self) -> list[CheckerJob]:
        """Return all READY jobs, sorted by priority (ascending). Every one of them is eligible right now."""
        ready = [j for j in self._jobs.values() if j.state == JobState.READY]
        return sorted(ready, key=lambda j: j.priority)

    def mark_queued(self, name: str) -> None:
        """READY -> QUEUED. Increments the attempt counter and clears any error from a previous attempt."""
        job = self._jobs[name]
        assert job.state == JobState.READY, f"{name}: expected READY, got {job.state}"
        job.state = JobState.QUEUED
        job.error = None
        job.attempt += 1

    def mark_in_progress(self, name: str) -> None:
        """QUEUED -> IN_PROGRESS. A pod has claimed the job and is running it. Idempotent.

        Idempotent because this records an observation rather than an action: the arbitrator learns a job was claimed by
        polling its row, and the same claim can be seen more than once. A repeat is the normal case, not an error.
        """
        job = self._jobs[name]
        if job.state == JobState.IN_PROGRESS:
            return
        assert job.state == JobState.QUEUED, f"{name}: expected QUEUED, got {job.state}"
        job.state = JobState.IN_PROGRESS

    def mark_completed(self, name: str, result: Any) -> None:
        """IN_PROGRESS -> COMPLETED. Promotes PENDING dependents whose all deps are now done."""
        job = self._jobs[name]
        assert job.state == JobState.IN_PROGRESS, f"{name}: expected IN_PROGRESS, got {job.state}"
        job.state = JobState.COMPLETED
        job.result = result

        for dep_name in self._dependents.get(name, set()):
            dep_job = self._jobs[dep_name]
            if dep_job.state == JobState.PENDING and self._all_deps_completed(dep_job):
                dep_job.state = JobState.READY

    def mark_failed(self, name: str, error: str) -> None:
        """QUEUED/IN_PROGRESS -> FAILED. Cascades SKIPPED to all transitive dependents."""
        job = self._jobs[name]
        assert job.state in (JobState.QUEUED, JobState.IN_PROGRESS), (
            f"{name}: expected QUEUED or IN_PROGRESS, got {job.state}"
        )
        job.state = JobState.FAILED
        job.error = error
        self._cascade_skip(name)

    def restore(
        self,
        name: str,
        state: JobState,
        result: Any = None,
        error: str | None = None,
        attempt: int = 0,
    ) -> None:
        """Put a job into a state it already reached elsewhere, without replaying the transitions that got it there.

        Used when one replica takes over a request another was running: each job's history is read back off its
        ``checker_executions`` row rather than lived through. The guarded transitions cannot be replayed, because they
        never happened in this process -- restoring a COMPLETED job through the state machine would mean walking it
        through QUEUED and IN_PROGRESS, counting attempts this replica never made and firing their side effects.

        The *consequences* of a terminal state are still applied, since they are properties of the state rather than of
        the transition: a restored COMPLETED job promotes dependents whose dependencies are now all satisfied, and a
        restored FAILED one cascades SKIPPED downstream.
        """
        job = self._jobs[name]
        job.state = state
        job.result = result
        job.error = error
        job.attempt = attempt
        if state == JobState.COMPLETED:
            for dep_name in self._dependents.get(name, set()):
                dep_job = self._jobs[dep_name]
                if dep_job.state == JobState.PENDING and self._all_deps_completed(dep_job):
                    dep_job.state = JobState.READY
        elif state == JobState.FAILED:
            self._cascade_skip(name)

    def is_done(self) -> bool:
        """True when all jobs are in a terminal state."""
        return all(j.state in _TERMINAL_STATES for j in self._jobs.values())

    def get_results(self) -> dict[str, Any]:
        """Return a summary of all jobs keyed by checker name."""
        return {
            name: {
                "state": job.state.value,
                "result": job.result,
                "error": job.error,
                "attempts": job.attempt,
            }
            for name, job in self._jobs.items()
        }

    def _all_deps_completed(self, job: CheckerJob) -> bool:
        return all(self._jobs[dep].state == JobState.COMPLETED for dep in job.deps if dep in self._jobs)

    def _cascade_skip(self, failed_name: str) -> None:
        """BFS: transitively mark all downstream dependents as SKIPPED."""
        queue = list(self._dependents.get(failed_name, set()))
        visited: set[str] = set()
        while queue:
            dep_name = queue.pop()
            if dep_name in visited:
                continue
            visited.add(dep_name)
            dep_job = self._jobs[dep_name]
            if dep_job.state not in _TERMINAL_STATES:
                dep_job.state = JobState.SKIPPED
                dep_job.error = f"Dependency '{failed_name}' failed"
                queue.extend(self._dependents.get(dep_name, set()))
