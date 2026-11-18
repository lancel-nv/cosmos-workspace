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
Client-agnostic dependency graph for the Arbitrator's checker execution order.

Parses the ``deps`` field from each checker section in the checker config, validates the resulting DAG (no cycles, no
unknown refs, no self-deps), computes topological priorities, and builds a CheckerQueue for a given set of requested
checkers.

This base supports the generic per-view checkers ``av.objects_single_view``, ``metropolis.hallucination``,
``metropolis.attribute_verification``, and ``metropolis.physical_plausibility``. Per-view checkers are expanded into N
parameterized jobs, one per entry in the request's ``augmented_video_urls`` dict. A single-view camera missing its
per-camera world-model (control) video runs camera-only (the static check on the camera stream alone) rather than
being skipped; a hallucination camera missing its original video is inserted already ``SKIPPED`` so the other cameras
still run.

A deployment that adds more checkers extends this base by overriding :meth:`DependencyGraph._per_camera_checkers` (to
fan additional checkers out per camera) and/or :meth:`DependencyGraph.build_queue` (to add cross-camera aggregator jobs
that consume every per-camera result), composing the protected helpers this base itself uses: :meth:`_resolve_needed`,
:meth:`_per_camera_priorities`, and :meth:`_build_queue_for`.
"""

import logging
from typing import Any

from services.arbitrator.checker_queue import CheckerJob, CheckerQueue, JobState

logger = logging.getLogger(__name__)

# Base name of the per-camera single-view check (the only per-camera checker that consumes a world-model video).
_SINGLE_VIEW_CHECKER: str = "av.objects_single_view"
# Base name of the per-camera hallucination check (the only per-camera checker that consumes an original video).
_HALLUCINATION_CHECKER: str = "metropolis.hallucination"
# Base name of the per-camera attribute-verification check (fans out per camera; consumes only the augmented video).
_ATTRIBUTE_VERIFICATION_CHECKER: str = "metropolis.attribute_verification"
# Base name of the per-camera physical-plausibility check (consumes only the augmented video).
_PHYSICAL_PLAUSIBILITY_CHECKER: str = "metropolis.physical_plausibility"


def single_view_job_name(camera_full_name: str) -> str:
    """Return the parameterized job name for a per-camera single-view job."""
    return f"av.objects_single_view[{camera_full_name}]"


def parameterized_name(base_name: str, camera_full_name: str) -> str:
    """Return the queue-unique job name for a per-camera job of ``base_name``."""
    return f"{base_name}[{camera_full_name}]"


def world_model_url(
    camera_full_name: str,
    world_model_video_urls: dict[str, str],
) -> str | None:
    """Return the per-camera world-model (control) video URL for a single-view camera, or ``None``.

    ``world_model_video_urls`` is the authoritative per-camera source for the single-view static (CWIP) check — treated
    exactly like ``augmented_video_urls``. There is **no** shared fallback (a top-down BEV would collapse static scores
    to ~0): a camera with a missing/empty entry returns ``None``, and the caller skips that single-view job rather than
    running it against a wrong video.
    """
    return world_model_video_urls.get(camera_full_name) or None


def original_video_url(
    camera_full_name: str,
    original_video_urls: dict[str, str],
) -> str | None:
    """Return the per-camera original/reference video URL for a hallucination camera, or ``None``.

    ``original_video_urls`` is the authoritative per-camera source for the hallucination check's motion baseline —
    treated exactly like ``augmented_video_urls``. There is **no** shared fallback: a camera with a missing/empty entry
    returns ``None``, and the caller skips that hallucination job rather than running it without a reference video.
    """
    return original_video_urls.get(camera_full_name) or None


def missing_original_video_reason(
    camera_full_name: str,
    original_video_urls: dict[str, str],
) -> str:
    """Human-readable skip reason for a hallucination camera lacking its per-camera original/reference video."""
    return (
        f"original_video_urls has no original/reference video for hallucination camera {camera_full_name!r}; a "
        f"per-camera original video is required (there is no shared fallback). Provided cameras: "
        f"{sorted(original_video_urls)}"
    )


def per_camera_job_params(
    camera_full_name: str,
    augmented_video_url: str,
    world_model_video_url: str | None = None,
    original_video_url: str | None = None,
) -> dict[str, str]:
    """Build the ``params`` dict for a fanned-out per-camera job.

    Every per-camera job carries ``camera_name`` and ``augmented_video_url``. The single-view static (CWIP) check
    additionally needs the *perspective* control video matching its camera's view, so a single-view job is also given
    ``world_model_video_url`` — resolved per camera by the caller (see :func:`world_model_url`), symmetric with
    ``augmented_video_url``. The hallucination check needs a per-camera original/reference video, so a hallucination
    job is given ``original_video_url`` (see :func:`original_video_url`). Checkers that do not consume a given
    video pass ``None`` and that key is omitted.
    """
    params: dict[str, str] = {"camera_name": camera_full_name, "augmented_video_url": augmented_video_url}
    if world_model_video_url is not None:
        params["world_model_video_url"] = world_model_video_url
    if original_video_url is not None:
        params["original_video_url"] = original_video_url
    return params


class DependencyGraphError(Exception):
    """Raised when the dependency graph is structurally invalid."""


class DependencyGraph:
    """Represents the dependency relationships between checker jobs.

    Constructed from the parsed checker config dict. Each checker section may contain an optional ``deps`` list naming
    other checkers that must complete before this one is queued.

    Example config structure::

        av.my_aggregator:
          deps:
            - av.objects_single_view
          ...other config...

    Checkers without a ``deps`` field are treated as having no dependencies.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._deps: dict[str, list[str]] = {}
        for name, checker_config in config.items():
            if isinstance(checker_config, dict):
                raw_deps = checker_config.get("deps") or []
                self._deps[name] = list(raw_deps)
            else:
                self._deps[name] = []

    def _per_camera_checkers(self) -> frozenset[str]:
        """Base names that fan out to one parameterized job per camera. Subclasses extend this set."""
        return frozenset(
            {
                _SINGLE_VIEW_CHECKER,
                _HALLUCINATION_CHECKER,
                _ATTRIBUTE_VERIFICATION_CHECKER,
                _PHYSICAL_PLAUSIBILITY_CHECKER,
            }
        )

    def validate(self) -> None:
        """Validate the dependency graph.

        Raises:
            DependencyGraphError: If the graph contains self-dependencies, references to unknown checkers, or cycles.
        """
        all_names = set(self._deps.keys())

        for name, deps in self._deps.items():
            for dep in deps:
                if dep == name:
                    raise DependencyGraphError(f"Checker '{name}' depends on itself")
                if dep not in all_names:
                    raise DependencyGraphError(f"Checker '{name}' depends on unknown checker '{dep}'")

        # Cycle detection via DFS coloring (white=0, gray=1, black=2)
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {name: WHITE for name in self._deps}

        def _dfs(node: str) -> None:
            color[node] = GRAY
            for dep in self._deps[node]:
                if color[dep] == GRAY:
                    raise DependencyGraphError(f"Cycle detected in dependency graph: '{node}' -> '{dep}'")
                if color[dep] == WHITE:
                    _dfs(dep)
            color[node] = BLACK

        for name in self._deps:
            if color[name] == WHITE:
                _dfs(name)

    def compute_priority(self, checker: str) -> int:
        """Return the topological depth of a checker (0 = no deps = highest priority).

        A checker's priority is one more than the maximum priority of its dependencies, so that all dependencies are
        always queued at a lower priority number (higher urgency) than their dependents.
        """
        memo: dict[str, int] = {}

        def _depth(name: str) -> int:
            if name in memo:
                return memo[name]
            deps = self._deps.get(name, [])
            memo[name] = (max(_depth(dep) for dep in deps) + 1) if deps else 0
            return memo[name]

        return _depth(checker)

    def build_queue(
        self,
        checkers: list[str],
        augmented_video_urls: dict[str, str] | None = None,
        world_model_video_urls: dict[str, str] | None = None,
        original_video_urls: dict[str, str] | None = None,
    ) -> CheckerQueue:
        """Build a CheckerQueue for the given checkers.

        Per-view checkers (see :meth:`_per_camera_checkers`) are expanded into N parameterized jobs, one per entry in
        ``augmented_video_urls``. Single-view jobs additionally carry a per-camera ``world_model_video_url`` taken from
        ``world_model_video_urls``; hallucination jobs additionally carry a per-camera ``original_video_url`` taken from
        ``original_video_urls`` — both authoritative per-camera sources, fanned out symmetrically with
        ``augmented_video_url`` (there is no shared fallback). A single-view camera missing its per-camera world-model
        video runs camera-only (the static check on the camera stream alone) rather than being skipped. A hallucination
        camera missing its per-camera original video has its job inserted directly in the ``SKIPPED`` terminal state, so
        the remaining cameras still run and the response surfaces the per-camera skip reason. Every other checker
        becomes a single plain job.

        Transitively includes any dependencies not already in ``checkers``, logging a warning for each auto-added
        checker.

        Args:
            checkers: Names of checkers explicitly requested for this run.
            augmented_video_urls: Mapping of full RDS camera stream id to per-camera augmented video URL. Keys drive
                per-view fan-out. Ignored for non-per-view checkers.
            world_model_video_urls: Mapping of full RDS camera stream id to per-camera world-model (control) video URL.
                The authoritative source consumed only by single-view jobs; a single-view camera without an entry runs
                camera-only (no shared fallback).
            original_video_urls: Mapping of full RDS camera stream id to per-camera original/reference video URL. The
                authoritative source consumed only by hallucination jobs; a hallucination camera without an entry has
                its job SKIPPED (no shared fallback).

        Returns:
            A CheckerQueue with all required jobs at their correct priorities. A single-view camera without a
            world-model video runs camera-only; a hallucination camera without an original video is present already in
            the ``SKIPPED`` terminal state.
        """
        augmented_video_urls = augmented_video_urls or {}
        world_model_video_urls = world_model_video_urls or {}
        original_video_urls = original_video_urls or {}
        needed = self._resolve_needed(checkers)
        return self._build_queue_for(
            needed,
            augmented_video_urls,
            world_model_video_urls,
            original_video_urls,
            self._per_camera_priorities(),
        )

    def _resolve_needed(self, checkers: list[str]) -> set[str]:
        """Return the transitive dependency closure of ``checkers``, warning about any auto-added checkers."""
        needed: set[str] = set()
        stack = list(checkers)
        while stack:
            name = stack.pop()
            if name in needed:
                continue
            needed.add(name)
            for dep in self._deps.get(name, []):
                if dep not in needed:
                    stack.append(dep)

        auto_added = needed - set(checkers)
        if auto_added:
            logger.warning("Auto-including transitive dependencies not in the request: %s", sorted(auto_added))
        return needed

    def _per_camera_priorities(self) -> dict[str, int]:
        """Topological priority for each per-camera checker base name (keys drive per-camera fan-out)."""
        return {name: self.compute_priority(name) for name in self._per_camera_checkers()}

    def _build_queue_for(
        self,
        names: set[str],
        augmented_video_urls: dict[str, str],
        world_model_video_urls: dict[str, str],
        original_video_urls: dict[str, str],
        per_camera_priority_by_base: dict[str, int],
    ) -> CheckerQueue:
        """Fan out per-camera checkers in ``names`` and create a single plain job for every other checker.

        Per-camera checkers (see :meth:`_per_camera_checkers`) expand to one parameterized job per entry in
        ``augmented_video_urls``; a single-view job additionally carries a per-camera ``world_model_video_url`` when one
        is available (else it runs camera-only), and a hallucination job carries a per-camera ``original_video_url``.
        A hallucination job whose original video is missing is inserted already ``SKIPPED`` (no shared fallback) so the
        remaining cameras still run. Every other checker becomes one plain job carrying its raw config deps. Subclasses
        compose this helper (typically excluding any names they handle as cross-camera aggregators) to extend the queue.
        """
        per_camera_checkers = self._per_camera_checkers()
        jobs: list[CheckerJob] = []
        # Per-camera jobs whose required per-camera video is missing: the job is created (so it surfaces in the
        # response) and then marked SKIPPED (job name -> skip reason), so the rest of the request still runs instead of
        # failing the whole admission.
        per_camera_skips: dict[str, str] = {}
        for name in names:
            if name in per_camera_checkers:
                for cam_full, video_url in augmented_video_urls.items():
                    # Single-view attaches a per-camera world-model video when one is available (a two-stream run) and
                    # runs camera-only when it is absent; hallucination consumes a per-camera original video.
                    world_url = (
                        world_model_url(cam_full, world_model_video_urls) if name == _SINGLE_VIEW_CHECKER else None
                    )
                    original_url = (
                        original_video_url(cam_full, original_video_urls) if name == _HALLUCINATION_CHECKER else None
                    )
                    job_name = parameterized_name(name, cam_full)
                    jobs.append(
                        CheckerJob(
                            name=job_name,
                            base_name=name,
                            priority=per_camera_priority_by_base[name],
                            deps=set(),
                            params=per_camera_job_params(cam_full, video_url, world_url, original_url),
                        )
                    )
                    # A single-view camera with no world-model video is not skipped: the static (CWIP) check runs
                    # camera-only on the camera stream alone. Only hallucination still requires its per-camera video.
                    if name == _HALLUCINATION_CHECKER and original_url is None:
                        per_camera_skips[job_name] = missing_original_video_reason(cam_full, original_video_urls)
            else:
                jobs.append(
                    CheckerJob(
                        name=name,
                        base_name=name,
                        priority=self.compute_priority(name),
                        deps=set(self._deps.get(name, [])),
                    )
                )

        queue = CheckerQueue(jobs)

        # Mark per-camera jobs whose camera lacks a required video as SKIPPED (terminal) so the remaining cameras
        # still run.
        for job_name, reason in per_camera_skips.items():
            skipped_job = queue.get_job(job_name)
            skipped_job.state = JobState.SKIPPED
            skipped_job.error = reason
            logger.warning("Per-camera job skipped at admission: %s", reason)

        return queue

    def get_all_deps(self) -> dict[str, list[str]]:
        """Return the full dependency map keyed by checker name."""
        return dict(self._deps)
