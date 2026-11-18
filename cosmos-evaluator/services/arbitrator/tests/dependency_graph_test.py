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

"""Tests for the client-agnostic DependencyGraph.

The module's graph supports a single per-camera checker (``av.objects_single_view``); every other checker becomes a
plain job carrying its declared dependencies.
"""

import unittest

from services.arbitrator.checker_queue import JobState
from services.arbitrator.dependency_graph import (
    DependencyGraph,
    DependencyGraphError,
    parameterized_name,
    single_view_job_name,
)

_SINGLE_VIEW = "av.objects_single_view"
_HALLUCINATION = "metropolis.hallucination"
_ATTRIBUTE_VERIFICATION = "metropolis.attribute_verification"

_CONFIG = {
    _SINGLE_VIEW: {"enabled": True, "deps": []},
    _HALLUCINATION: {"enabled": True, "deps": []},
    _ATTRIBUTE_VERIFICATION: {"enabled": True, "deps": []},
}


def _graph(config: dict) -> DependencyGraph:
    g = DependencyGraph(config)
    g.validate()
    return g


class TestConstructionAndValidation(unittest.TestCase):
    def test_checker_without_deps_field_defaults_to_empty(self) -> None:
        g = DependencyGraph({_SINGLE_VIEW: {"enabled": True}})
        self.assertEqual(g.get_all_deps()[_SINGLE_VIEW], [])

    def test_non_dict_checker_config_defaults_to_empty_deps(self) -> None:
        g = DependencyGraph({_SINGLE_VIEW: None})
        self.assertEqual(g.get_all_deps()[_SINGLE_VIEW], [])

    def test_self_dependency_raises(self) -> None:
        g = DependencyGraph({"a": {"deps": ["a"]}})
        with self.assertRaisesRegex(DependencyGraphError, "depends on itself"):
            g.validate()

    def test_unknown_dependency_raises(self) -> None:
        g = DependencyGraph({"a": {"deps": ["b"]}})
        with self.assertRaisesRegex(DependencyGraphError, "unknown checker"):
            g.validate()

    def test_cycle_raises(self) -> None:
        g = DependencyGraph({"a": {"deps": ["b"]}, "b": {"deps": ["a"]}})
        with self.assertRaisesRegex(DependencyGraphError, "Cycle detected"):
            g.validate()


class TestPriorities(unittest.TestCase):
    def test_compute_priority_no_deps_is_zero(self) -> None:
        g = _graph(_CONFIG)
        self.assertEqual(g.compute_priority(_SINGLE_VIEW), 0)

    def test_compute_priority_increments_with_depth(self) -> None:
        g = _graph({"a": {"deps": []}, "b": {"deps": ["a"]}, "c": {"deps": ["b"]}})
        self.assertEqual(g.compute_priority("a"), 0)
        self.assertEqual(g.compute_priority("b"), 1)
        self.assertEqual(g.compute_priority("c"), 2)


class TestSingleViewFanOut(unittest.TestCase):
    """build_queue: single-view per-camera fan-out."""

    def test_single_view_fans_out_per_camera(self) -> None:
        g = _graph(_CONFIG)
        queue = g.build_queue(
            [_SINGLE_VIEW],
            augmented_video_urls={"cam_a": "s3://a.mp4", "cam_b": "s3://b.mp4"},
            world_model_video_urls={"cam_a": "s3://ca.mp4", "cam_b": "s3://cb.mp4"},
        )
        names = {job.name for job in queue.iter_jobs()}
        self.assertIn(single_view_job_name("cam_a"), names)
        self.assertIn(single_view_job_name("cam_b"), names)

    def test_single_view_missing_world_model_runs_camera_only(self) -> None:
        # A single-view camera absent from world_model_video_urls runs camera-only (no world URL in its params)
        # rather than being skipped; a camera that has a world URL runs two-stream.
        g = _graph(_CONFIG)
        queue = g.build_queue(
            [_SINGLE_VIEW],
            augmented_video_urls={"cam_a": "s3://a.mp4", "cam_b": "s3://b.mp4"},
            world_model_video_urls={"cam_a": "s3://ca.mp4"},  # cam_b missing
        )
        camera_only_job = queue.get_job(single_view_job_name("cam_b"))
        self.assertNotEqual(camera_only_job.state, JobState.SKIPPED)
        self.assertNotIn("world_model_video_url", camera_only_job.params)
        two_stream_job = queue.get_job(single_view_job_name("cam_a"))
        self.assertNotEqual(two_stream_job.state, JobState.SKIPPED)
        self.assertEqual(two_stream_job.params["world_model_video_url"], "s3://ca.mp4")

    def test_single_view_no_world_model_map_runs_all_camera_only(self) -> None:
        # No world-model videos at all: every single-view camera runs camera-only (none skipped).
        g = _graph(_CONFIG)
        queue = g.build_queue(
            [_SINGLE_VIEW],
            augmented_video_urls={"cam_a": "s3://a.mp4", "cam_b": "s3://b.mp4"},
            world_model_video_urls={},  # none provided
        )
        for cam in ("cam_a", "cam_b"):
            job = queue.get_job(single_view_job_name(cam))
            self.assertNotEqual(job.state, JobState.SKIPPED)
            self.assertNotIn("world_model_video_url", job.params)


class TestHallucinationFanOut(unittest.TestCase):
    """build_queue: hallucination per-camera fan-out."""

    def test_hallucination_fans_out_per_camera(self) -> None:
        g = _graph(_CONFIG)
        queue = g.build_queue(
            [_HALLUCINATION],
            augmented_video_urls={"cam_a": "s3://a.mp4", "cam_b": "s3://b.mp4"},
            original_video_urls={"cam_a": "s3://oa.mp4", "cam_b": "s3://ob.mp4"},
        )
        by_name = {job.name: job for job in queue.iter_jobs()}
        self.assertIn(parameterized_name(_HALLUCINATION, "cam_a"), by_name)
        self.assertIn(parameterized_name(_HALLUCINATION, "cam_b"), by_name)
        # Each hallucination job carries its per-camera augmented + original video.
        job_a = by_name[parameterized_name(_HALLUCINATION, "cam_a")]
        self.assertEqual(job_a.params["augmented_video_url"], "s3://a.mp4")
        self.assertEqual(job_a.params["original_video_url"], "s3://oa.mp4")
        self.assertNotEqual(job_a.state, JobState.SKIPPED)

    def test_hallucination_missing_original_video_is_skipped(self) -> None:
        g = _graph(_CONFIG)
        queue = g.build_queue(
            [_HALLUCINATION],
            augmented_video_urls={"cam_a": "s3://a.mp4", "cam_b": "s3://b.mp4"},
            original_video_urls={"cam_a": "s3://oa.mp4"},  # cam_b missing
        )
        skipped = queue.get_job(parameterized_name(_HALLUCINATION, "cam_b"))
        self.assertEqual(skipped.state, JobState.SKIPPED)
        self.assertIn("original", skipped.error or "")
        running = queue.get_job(parameterized_name(_HALLUCINATION, "cam_a"))
        self.assertNotEqual(running.state, JobState.SKIPPED)

    def test_hallucination_without_original_urls_skips_every_camera(self) -> None:
        g = _graph(_CONFIG)
        queue = g.build_queue(
            [_HALLUCINATION],
            augmented_video_urls={"cam_a": "s3://a.mp4", "cam_b": "s3://b.mp4"},
        )
        for cam in ("cam_a", "cam_b"):
            self.assertEqual(queue.get_job(parameterized_name(_HALLUCINATION, cam)).state, JobState.SKIPPED)


class TestAttributeVerificationFanOut(unittest.TestCase):
    """build_queue: attribute_verification per-camera fan-out."""

    def test_attribute_verification_fans_out_per_camera_without_extra_videos(self) -> None:
        g = _graph(_CONFIG)
        queue = g.build_queue(
            [_ATTRIBUTE_VERIFICATION],
            augmented_video_urls={"cam_a": "s3://a.mp4", "cam_b": "s3://b.mp4"},
        )
        by_name = {job.name: job for job in queue.iter_jobs()}
        job_a = by_name[parameterized_name(_ATTRIBUTE_VERIFICATION, "cam_a")]
        job_b = by_name[parameterized_name(_ATTRIBUTE_VERIFICATION, "cam_b")]
        # Attribute verification only needs the augmented video; no original/world-model video and never SKIPPED for
        # that.
        self.assertEqual(job_a.params["augmented_video_url"], "s3://a.mp4")
        self.assertNotIn("original_video_url", job_a.params)
        self.assertNotIn("world_model_video_url", job_a.params)
        self.assertNotEqual(job_a.state, JobState.SKIPPED)
        self.assertNotEqual(job_b.state, JobState.SKIPPED)


class TestNonPerCameraChecker(unittest.TestCase):
    def test_non_per_camera_checker_is_a_plain_job(self) -> None:
        """A non-per-camera checker becomes a single plain job carrying its raw config deps (no aggregator fan-out).

        The base has no cross-camera aggregator concept: a checker that merely declares a dependency on the per-camera
        single-view checker is scheduled as one ordinary job, and its single-view dependency is auto-included and
        fanned out per camera.
        """
        g = _graph({_SINGLE_VIEW: {"deps": []}, "report.summary": {"deps": [_SINGLE_VIEW]}})
        queue = g.build_queue(
            ["report.summary"],
            augmented_video_urls={"cam_a": "s3://a.mp4"},
            world_model_video_urls={"cam_a": "s3://ca.mp4"},
        )
        summary_jobs = [job for job in queue.iter_jobs() if job.base_name == "report.summary"]
        self.assertEqual(len(summary_jobs), 1)
        # Plain (unparameterized) job name and raw config deps -> no per-camera aggregator wiring.
        self.assertEqual(summary_jobs[0].name, "report.summary")
        self.assertEqual(summary_jobs[0].deps, {_SINGLE_VIEW})
        # The single-view dependency was auto-included and fanned out per camera.
        self.assertIn(single_view_job_name("cam_a"), {job.name for job in queue.iter_jobs()})


if __name__ == "__main__":
    unittest.main()
