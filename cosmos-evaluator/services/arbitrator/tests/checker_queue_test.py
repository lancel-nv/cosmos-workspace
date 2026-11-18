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

import unittest

from services.arbitrator.checker_queue import CheckerJob, CheckerQueue, JobState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _job(name: str, priority: int = 0, deps: set[str] | None = None) -> CheckerJob:
    return CheckerJob(name=name, priority=priority, deps=deps or set())


def _ready_queue(*names: str) -> CheckerQueue:
    """Queue of independent jobs (no deps) — all start READY."""
    return CheckerQueue([_job(n) for n in names])


def _claim(q: CheckerQueue, name: str) -> None:
    q.mark_queued(name)
    q.mark_in_progress(name)


def _complete(q: CheckerQueue, name: str, result: object = None) -> None:
    _claim(q, name)
    q.mark_completed(name, result)


class TestInitialState(unittest.TestCase):
    def test_jobs_without_deps_start_ready(self) -> None:
        q = _ready_queue("a", "b")
        self.assertEqual(q.get_job("a").state, JobState.READY)
        self.assertEqual(q.get_job("b").state, JobState.READY)

    def test_jobs_with_deps_start_pending(self) -> None:
        q = CheckerQueue([_job("a"), _job("b", deps={"a"})])
        self.assertEqual(q.get_job("a").state, JobState.READY)
        self.assertEqual(q.get_job("b").state, JobState.PENDING)


class TestHappyPathTransitions(unittest.TestCase):
    def test_happy_path_state_transitions(self) -> None:
        q = _ready_queue("a")
        self.assertEqual(q.get_job("a").state, JobState.READY)

        q.mark_queued("a")
        self.assertEqual(q.get_job("a").state, JobState.QUEUED)

        q.mark_in_progress("a")
        self.assertEqual(q.get_job("a").state, JobState.IN_PROGRESS)

        q.mark_completed("a", result={"score": 1.0})
        self.assertEqual(q.get_job("a").state, JobState.COMPLETED)
        self.assertEqual(q.get_job("a").result, {"score": 1.0})

    def test_mark_queued_increments_attempt(self) -> None:
        q = _ready_queue("a")
        self.assertEqual(q.get_job("a").attempt, 0)
        q.mark_queued("a")
        self.assertEqual(q.get_job("a").attempt, 1)


class TestFailureAndCascade(unittest.TestCase):
    def test_in_progress_job_can_be_failed(self) -> None:
        q = _ready_queue("a")
        _claim(q, "a")
        q.mark_failed("a", error="CUDA OOM")
        self.assertEqual(q.get_job("a").state, JobState.FAILED)
        self.assertEqual(q.get_job("a").error, "CUDA OOM")

    def test_queued_job_can_be_failed(self) -> None:
        """mark_failed is valid from QUEUED (e.g. NotImplementedError before HTTP call)."""
        q = _ready_queue("a")
        q.mark_queued("a")
        q.mark_failed("a", error="Not implemented")
        self.assertEqual(q.get_job("a").state, JobState.FAILED)

    def test_failure_cascades_to_direct_dependent(self) -> None:
        q = CheckerQueue([_job("a"), _job("b", deps={"a"})])
        _claim(q, "a")
        q.mark_failed("a", "error")
        self.assertEqual(q.get_job("b").state, JobState.SKIPPED)

    def test_failure_cascades_transitively(self) -> None:
        # a -> b -> c
        q = CheckerQueue([_job("a"), _job("b", deps={"a"}), _job("c", deps={"b"})])
        _claim(q, "a")
        q.mark_failed("a", "error")
        self.assertEqual(q.get_job("b").state, JobState.SKIPPED)
        self.assertEqual(q.get_job("c").state, JobState.SKIPPED)

    def test_cascade_does_not_overwrite_already_completed_job(self) -> None:
        # a -> c, b -> c; a completes, b fails — c was never queued so still PENDING
        q = CheckerQueue([_job("a"), _job("b"), _job("c", deps={"a", "b"})])
        _complete(q, "a")
        # c is still PENDING (b not done yet)
        _claim(q, "b")
        q.mark_failed("b", "error")
        # c should be SKIPPED, not COMPLETED
        self.assertEqual(q.get_job("c").state, JobState.SKIPPED)


class TestDependencyUnblocking(unittest.TestCase):
    def test_dependent_becomes_ready_when_dep_completes(self) -> None:
        q = CheckerQueue([_job("a"), _job("b", deps={"a"})])
        self.assertEqual(q.get_job("b").state, JobState.PENDING)
        _complete(q, "a")
        self.assertEqual(q.get_job("b").state, JobState.READY)

    def test_dependent_with_two_deps_waits_for_both(self) -> None:
        q = CheckerQueue([_job("a"), _job("b"), _job("c", deps={"a", "b"})])
        _complete(q, "a")
        self.assertEqual(q.get_job("c").state, JobState.PENDING)  # still waiting for b
        _complete(q, "b")
        self.assertEqual(q.get_job("c").state, JobState.READY)

    def test_diamond_dependency(self) -> None:
        # a -> b, a -> c, b -> d, c -> d
        q = CheckerQueue(
            [
                _job("a"),
                _job("b", deps={"a"}),
                _job("c", deps={"a"}),
                _job("d", deps={"b", "c"}),
            ]
        )
        _complete(q, "a")
        self.assertEqual(q.get_job("b").state, JobState.READY)
        self.assertEqual(q.get_job("c").state, JobState.READY)
        self.assertEqual(q.get_job("d").state, JobState.PENDING)

        _complete(q, "b")
        self.assertEqual(q.get_job("d").state, JobState.PENDING)  # c not done yet

        _complete(q, "c")
        self.assertEqual(q.get_job("d").state, JobState.READY)


class TestPriorityOrdering(unittest.TestCase):
    def test_get_ready_jobs_sorted_by_priority(self) -> None:
        q = CheckerQueue(
            [
                _job("high", priority=0),
                _job("low", priority=2),
                _job("mid", priority=1),
            ]
        )
        names = [j.name for j in q.get_ready_jobs()]
        self.assertEqual(names, ["high", "mid", "low"])

    def test_get_ready_jobs_excludes_queued_and_pending(self) -> None:
        q = CheckerQueue([_job("a"), _job("b"), _job("c", deps={"a"})])
        q.mark_queued("b")
        ready_names = {j.name for j in q.get_ready_jobs()}
        self.assertEqual(ready_names, {"a"})


class TestIsDone(unittest.TestCase):
    def test_is_done_false_when_active_jobs_exist(self) -> None:
        q = _ready_queue("a")
        self.assertFalse(q.is_done())

    def test_is_done_true_when_all_terminal(self) -> None:
        q = _ready_queue("a")
        _complete(q, "a")
        self.assertTrue(q.is_done())

    def test_is_done_true_when_mixed_terminal_states(self) -> None:
        q = CheckerQueue([_job("a"), _job("b", deps={"a"})])
        _claim(q, "a")
        q.mark_failed("a", "error")
        # a=FAILED, b=SKIPPED — both terminal
        self.assertTrue(q.is_done())


class TestIterJobs(unittest.TestCase):
    """iter_jobs is used by the waiting room to scan per-checker job state."""

    def test_iter_jobs_yields_every_job_exactly_once(self) -> None:
        q = CheckerQueue([_job("a"), _job("b", deps={"a"}), _job("c")])
        names = sorted(j.name for j in q.iter_jobs())
        self.assertEqual(names, ["a", "b", "c"])

    def test_iter_jobs_reflects_post_transition_state(self) -> None:
        q = _ready_queue("a", "b")
        q.mark_queued("a")
        states = {j.name: j.state for j in q.iter_jobs()}
        self.assertEqual(states, {"a": JobState.QUEUED, "b": JobState.READY})


class TestGetResults(unittest.TestCase):
    def test_get_results_format(self) -> None:
        q = _ready_queue("a")
        _complete(q, "a", result={"score": 0.9})
        results = q.get_results()
        self.assertEqual(results["a"]["state"], "completed")
        self.assertEqual(results["a"]["result"], {"score": 0.9})
        self.assertIsNone(results["a"]["error"])
        self.assertEqual(results["a"]["attempts"], 1)

    def test_get_results_includes_all_jobs(self) -> None:
        q = CheckerQueue([_job("a"), _job("b", deps={"a"})])
        results = q.get_results()
        self.assertEqual(set(results.keys()), {"a", "b"})


class TestFlatParallel(unittest.TestCase):
    """Jobs with no deps at all."""

    def test_all_independent_jobs_start_ready(self) -> None:
        q = _ready_queue("egomotion", "objects_single_view", "objects_multi_view")
        ready = {j.name for j in q.get_ready_jobs()}
        self.assertEqual(ready, {"egomotion", "objects_single_view", "objects_multi_view"})


class TestAddJobsAndParameterizedIdentity(unittest.TestCase):
    """add_jobs union-merge semantics and parameterized job identity."""

    def test_add_jobs_union_merge_is_noop_for_duplicate_names(self) -> None:
        """Adding a job whose name already exists leaves the existing job untouched."""
        q = CheckerQueue([_job("a")])
        _complete(q, "a", result={"score": 1.0})

        # Re-add a job with the same name — existing completed job must be preserved.
        q.add_jobs([_job("a")])
        self.assertEqual(q.get_job("a").state, JobState.COMPLETED)
        self.assertEqual(q.get_job("a").result, {"score": 1.0})
        self.assertEqual(len(q.get_results()), 1)

    def test_add_jobs_adds_new_jobs_with_correct_initial_state(self) -> None:
        q = CheckerQueue([_job("a")])
        q.add_jobs([_job("b"), _job("c", deps={"a"})])
        self.assertEqual(q.get_job("b").state, JobState.READY)
        self.assertEqual(q.get_job("c").state, JobState.PENDING)

    def test_add_jobs_wires_dependents_for_unblocking(self) -> None:
        """After add_jobs, completing an existing job unblocks a newly added dependent."""
        q = CheckerQueue([_job("a")])
        q.add_jobs([_job("b", deps={"a"})])
        self.assertEqual(q.get_job("b").state, JobState.PENDING)
        _complete(q, "a")
        self.assertEqual(q.get_job("b").state, JobState.READY)

    def test_checker_job_base_name_defaults_to_name(self) -> None:
        """CheckerJob.base_name defaults to the queue-unique name when not set explicitly."""
        job = CheckerJob(name="foo", priority=0, deps=set())
        self.assertEqual(job.base_name, "foo")

    def test_checker_job_parameterized_base_name_distinct_from_name(self) -> None:
        """For parameterized fan-out jobs, base_name and name are different."""
        job = CheckerJob(
            name="av.objects_single_view[camera_front_wide_120fov]",
            base_name="av.objects_single_view",
            priority=0,
            deps=set(),
            params={"camera_name": "camera_front_wide_120fov"},
        )
        self.assertEqual(job.base_name, "av.objects_single_view")
        self.assertEqual(job.name, "av.objects_single_view[camera_front_wide_120fov]")

    def test_parameterized_jobs_with_same_base_name_coexist(self) -> None:
        """Two fan-out jobs with the same base_name but different params are distinct queue entries."""
        q = CheckerQueue(
            [
                CheckerJob(
                    name="av.egomotion[cam_a]",
                    base_name="av.egomotion",
                    priority=0,
                    deps=set(),
                    params={"camera_name": "cam_a"},
                ),
                CheckerJob(
                    name="av.egomotion[cam_b]",
                    base_name="av.egomotion",
                    priority=0,
                    deps=set(),
                    params={"camera_name": "cam_b"},
                ),
            ]
        )
        self.assertEqual(q.get_job("av.egomotion[cam_a]").state, JobState.READY)
        self.assertEqual(q.get_job("av.egomotion[cam_b]").state, JobState.READY)
        self.assertEqual(q.get_job("av.egomotion[cam_a]").base_name, "av.egomotion")
        self.assertEqual(q.get_job("av.egomotion[cam_b]").base_name, "av.egomotion")


class TestRestore(unittest.TestCase):
    """Reinstating a job's state from a record of what happened to it in another process."""

    def test_a_completed_job_is_restored_without_being_queued_first(self) -> None:
        """Replaying the transitions would mean inventing a dispatch, which is the thing there is no record of."""
        q = _ready_queue("a")

        q.restore("a", JobState.COMPLETED, result={"score": 1}, attempt=2)

        job = q.get_job("a")
        self.assertEqual((job.state, job.result, job.attempt), (JobState.COMPLETED, {"score": 1}, 2))

    def test_a_restored_completion_promotes_the_dependents_it_unblocks(self) -> None:
        q = CheckerQueue([_job("a"), _job("b", deps={"a"})])
        self.assertEqual(q.get_job("b").state, JobState.PENDING)

        q.restore("a", JobState.COMPLETED, result={})

        self.assertEqual(q.get_job("b").state, JobState.READY)

    def test_a_dependent_with_work_left_stays_pending(self) -> None:
        q = CheckerQueue([_job("a"), _job("b"), _job("c", deps={"a", "b"})])

        q.restore("a", JobState.COMPLETED, result={})

        self.assertEqual(q.get_job("c").state, JobState.PENDING)

    def test_a_restored_failure_cascades_downstream(self) -> None:
        q = CheckerQueue([_job("a"), _job("b", deps={"a"}), _job("c", deps={"b"})])

        q.restore("a", JobState.FAILED, error="checker died")

        self.assertEqual(q.get_job("b").state, JobState.SKIPPED)
        self.assertEqual(q.get_job("c").state, JobState.SKIPPED)

    def test_a_job_a_pod_is_still_working_on_is_restored_as_in_progress(self) -> None:
        q = _ready_queue("a")

        q.restore("a", JobState.IN_PROGRESS, attempt=1)

        self.assertEqual(q.get_job("a").state, JobState.IN_PROGRESS)
        self.assertEqual(q.get_ready_jobs(), [], "the scheduler must not start a second copy of work already running")

    def test_a_fully_restored_queue_is_done(self) -> None:
        q = _ready_queue("a", "b")

        q.restore("a", JobState.COMPLETED, result={})
        q.restore("b", JobState.FAILED, error="boom")

        self.assertIs(q.is_done(), True)


if __name__ == "__main__":
    unittest.main()
