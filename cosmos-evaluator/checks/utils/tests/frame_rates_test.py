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

"""Tests for the object check's fixed 10 fps evaluation timeline."""

import dataclasses
import logging
import unittest

from checks.utils.frame_rates import (
    OBJECT_CHECK_FPS,
    RDS_INDEX_FPS,
    RDS_INDEX_STRIDE,
    ObjectCheckTimeline,
    stride_for_fps,
    validate_annotation_axis,
)


class StrideForFpsTest(unittest.TestCase):
    """The per-source stride derivation and the rates it refuses."""

    def test_multiples_of_the_object_check_rate_give_whole_strides(self) -> None:
        self.assertEqual(stride_for_fps(10.0, "Camera video"), 1)
        self.assertEqual(stride_for_fps(20.0, "Camera video"), 2)
        self.assertEqual(stride_for_fps(30.0, "Camera video"), 3)
        self.assertEqual(stride_for_fps(60.0, "Camera video"), 6)

    def test_ntsc_rate_is_absorbed_by_the_tolerance(self) -> None:
        # A container reporting 30000/1001 is a 30 fps video; it must not be rejected as non-conforming.
        self.assertEqual(stride_for_fps(30000.0 / 1001.0, "World video"), 3)

    def test_non_multiple_rates_are_rejected(self) -> None:
        for fps in (25.0, 15.0, 24.0, 5.0):
            with self.subTest(fps=fps), self.assertRaises(ValueError) as ctx:
                stride_for_fps(fps, "Camera video")
            self.assertIn("integer multiple", str(ctx.exception))

    def test_rate_below_the_object_check_rate_is_rejected(self) -> None:
        # Nothing can manufacture 10 Hz evidence out of a 5 fps source, so this must fail rather than duplicate.
        with self.assertRaises(ValueError):
            stride_for_fps(5.0, "World video")

    def test_non_positive_rate_is_rejected(self) -> None:
        for fps in (0.0, -30.0):
            with self.subTest(fps=fps), self.assertRaises(ValueError) as ctx:
                stride_for_fps(fps, "Camera video")
            self.assertIn("must be positive", str(ctx.exception))

    def test_error_names_the_offending_source(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            stride_for_fps(25.0, "World video")
        self.assertIn("World video", str(ctx.exception))


class ObjectCheckTimelineBuildTest(unittest.TestCase):
    """Step-count negotiation across the four sources."""

    def test_all_thirty_fps_sources_subsample_to_ten(self) -> None:
        timeline = ObjectCheckTimeline.build(
            camera_video_fps=30.0,
            camera_frame_count=300,
            world_video_fps=30.0,
            world_frame_count=300,
            pose_frame_count=301,
            annotated_frame_count=101,
        )
        self.assertEqual(timeline.camera_stride, 3)
        self.assertEqual(timeline.world_stride, 3)
        self.assertEqual(timeline.num_steps, 100)

    def test_ten_fps_world_video_against_thirty_fps_camera(self) -> None:
        # The WSM v3 shape: a 10 fps world video already at the annotation rate, paired with a 30 fps camera video.
        timeline = ObjectCheckTimeline.build(
            camera_video_fps=30.0,
            camera_frame_count=300,
            world_video_fps=10.0,
            world_frame_count=100,
            pose_frame_count=298,
            annotated_frame_count=100,
        )
        self.assertEqual(timeline.camera_stride, 3)
        self.assertEqual(timeline.world_stride, 1)
        self.assertEqual(timeline.num_steps, 100)

    def test_step_count_is_bounded_by_the_shortest_source(self) -> None:
        timeline = ObjectCheckTimeline.build(
            camera_video_fps=30.0,
            camera_frame_count=300,
            world_video_fps=30.0,
            world_frame_count=150,
            annotated_frame_count=101,
        )
        self.assertEqual(timeline.num_steps, 50)

    def test_annotation_axis_can_bound_the_step_count(self) -> None:
        timeline = ObjectCheckTimeline.build(
            camera_video_fps=30.0,
            camera_frame_count=300,
            annotated_frame_count=40,
        )
        self.assertEqual(timeline.num_steps, 40)

    def test_pose_stream_can_bound_the_step_count(self) -> None:
        timeline = ObjectCheckTimeline.build(
            camera_video_fps=30.0,
            camera_frame_count=300,
            pose_frame_count=31,
        )
        self.assertEqual(timeline.num_steps, 11)

    def test_omitted_sources_do_not_constrain(self) -> None:
        timeline = ObjectCheckTimeline.build(camera_video_fps=10.0, camera_frame_count=100)
        self.assertEqual(timeline.num_steps, 100)
        self.assertIsNone(timeline.world_stride)
        self.assertIsNone(timeline.world_video_fps)

    def test_trial_steps_truncate_the_timeline(self) -> None:
        timeline = ObjectCheckTimeline.build(
            camera_video_fps=30.0,
            camera_frame_count=300,
            trial_steps=7,
        )
        self.assertEqual(timeline.num_steps, 7)

    def test_trial_steps_above_the_available_count_do_not_extend_it(self) -> None:
        timeline = ObjectCheckTimeline.build(
            camera_video_fps=30.0,
            camera_frame_count=300,
            trial_steps=5000,
        )
        self.assertEqual(timeline.num_steps, 100)

    def test_world_fps_without_a_frame_count_is_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            ObjectCheckTimeline.build(
                camera_video_fps=30.0,
                camera_frame_count=300,
                world_video_fps=30.0,
            )
        self.assertIn("world_frame_count", str(ctx.exception))

    def test_subsampling_is_logged_at_info(self) -> None:
        with self.assertLogs("checks.utils.frame_rates", level=logging.INFO) as captured:
            ObjectCheckTimeline.build(
                camera_video_fps=30.0,
                camera_frame_count=300,
                world_video_fps=30.0,
                world_frame_count=300,
            )
        joined = "\n".join(captured.output)
        self.assertIn("Subsampling", joined)
        self.assertIn(f"{OBJECT_CHECK_FPS} fps", joined)

    def test_rate_summary_is_logged_even_without_subsampling(self) -> None:
        with self.assertLogs("checks.utils.frame_rates", level=logging.INFO) as captured:
            ObjectCheckTimeline.build(camera_video_fps=10.0, camera_frame_count=100)
        self.assertIn("evaluates at", "\n".join(captured.output))


class ObjectCheckTimelineMappingTest(unittest.TestCase):
    """Step-to-source index translation."""

    def setUp(self) -> None:
        self.timeline = ObjectCheckTimeline.build(
            camera_video_fps=30.0,
            camera_frame_count=300,
            world_video_fps=10.0,
            world_frame_count=100,
        )

    def test_rds_index_advances_by_the_namespace_stride(self) -> None:
        self.assertEqual(self.timeline.rds_index(0), 0)
        self.assertEqual(self.timeline.rds_index(1), RDS_INDEX_STRIDE)
        self.assertEqual(self.timeline.rds_index(33), 33 * RDS_INDEX_STRIDE)

    def test_video_frames_follow_their_own_strides(self) -> None:
        self.assertEqual(self.timeline.camera_frame(4), 12)
        self.assertEqual(self.timeline.world_frame(4), 4)

    def test_index_sequences_cover_every_step(self) -> None:
        self.assertEqual(len(self.timeline.rds_indices()), self.timeline.num_steps)
        self.assertEqual(self.timeline.camera_frames()[:4], [0, 3, 6, 9])
        self.assertEqual(self.timeline.world_frames()[:4], [0, 1, 2, 3])

    def test_every_rds_index_lands_on_the_annotation_grid(self) -> None:
        # The whole point of the fixed rate: no step may address an unannotated RDS index.
        self.assertTrue(all(index % RDS_INDEX_STRIDE == 0 for index in self.timeline.rds_indices()))

    def test_world_frame_without_a_world_video_is_rejected(self) -> None:
        camera_only = ObjectCheckTimeline.build(camera_video_fps=10.0, camera_frame_count=10)
        with self.assertRaises(ValueError):
            camera_only.world_frame(0)
        with self.assertRaises(ValueError):
            camera_only.world_frames()

    def test_timeline_is_frozen(self) -> None:
        # The assignment is the behaviour under test, so mypy's (correct) frozen-field complaint is suppressed.
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.timeline.num_steps = 5  # type: ignore[misc]


class ValidateAnnotationAxisTest(unittest.TestCase):
    """Guarding the assumption that annotations sit on the expected grid."""

    def test_exact_grid_passes(self) -> None:
        validate_annotation_axis(list(range(0, 300, RDS_INDEX_STRIDE)), num_steps=100)

    def test_denser_axis_passes_and_reports_the_surplus(self) -> None:
        with self.assertLogs("checks.utils.frame_rates", level=logging.INFO) as captured:
            validate_annotation_axis(list(range(300)), num_steps=100)
        self.assertIn("beyond", "\n".join(captured.output))

    def test_missing_index_is_rejected(self) -> None:
        axis = [index for index in range(0, 300, RDS_INDEX_STRIDE) if index != 12]
        with self.assertRaises(ValueError) as ctx:
            validate_annotation_axis(axis, num_steps=100)
        self.assertIn("12", str(ctx.exception))

    def test_offset_grid_is_rejected(self) -> None:
        # A dataset annotating 1, 4, 7, ... would otherwise leave every frame silently unscoreable.
        with self.assertRaises(ValueError):
            validate_annotation_axis(list(range(1, 301, RDS_INDEX_STRIDE)), num_steps=100)

    def test_empty_axis_with_no_steps_passes(self) -> None:
        validate_annotation_axis([], num_steps=0)


class ConstantsTest(unittest.TestCase):
    """The constants the rest of the check depends on."""

    def test_object_check_rate_divides_the_index_namespace(self) -> None:
        self.assertEqual(RDS_INDEX_FPS % OBJECT_CHECK_FPS, 0)
        self.assertEqual(RDS_INDEX_STRIDE, RDS_INDEX_FPS // OBJECT_CHECK_FPS)


if __name__ == "__main__":
    unittest.main()
