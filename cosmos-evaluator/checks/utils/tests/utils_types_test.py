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

"""Unit tests for checks.utils.types (Pydantic models for hallucination-style checks)."""

import unittest

from pydantic import ValidationError

from checks.utils.types import (
    DynParams,
    Event,
    FrameMetadata,
    RuleFinding,
    Sample,
    ScoreValue,
)


class TestDynParams(unittest.TestCase):
    def test_defaults(self) -> None:
        p = DynParams()
        self.assertEqual(p.grad_thresh, 10.0)
        self.assertEqual(p.blur_ksize, 7)
        self.assertIsNone(p.max_frames)

    def test_override(self) -> None:
        p = DynParams(max_frames=100, resize_factor=0.5)
        self.assertEqual(p.max_frames, 100)
        self.assertEqual(p.resize_factor, 0.5)


class TestScoreValue(unittest.TestCase):
    def test_construct(self) -> None:
        s = ScoreValue(key="k", value=0.25)
        self.assertEqual(s.model_dump(), {"key": "k", "value": 0.25})


class TestFrameMetadata(unittest.TestCase):
    def test_optional_camera_id(self) -> None:
        f = FrameMetadata(index=0, timestamp_ms=1, width=1920, height=1080)
        self.assertIsNone(f.camera_id)

    def test_with_camera_id(self) -> None:
        f = FrameMetadata(index=2, timestamp_ms=10, width=640, height=480, camera_id="front")
        self.assertEqual(f.camera_id, "front")


class TestEvent(unittest.TestCase):
    def test_default_attributes(self) -> None:
        e = Event(type="x", start_frame=0, end_frame=1)
        self.assertEqual(e.attributes, {})

    def test_custom_attributes(self) -> None:
        e = Event(type="cut", start_frame=0, end_frame=5, attributes={"lane": 2})
        self.assertEqual(e.attributes["lane"], 2)


class TestSample(unittest.TestCase):
    def test_minimal(self) -> None:
        s = Sample(sample_id="abc")
        self.assertEqual(s.sample_id, "abc")
        self.assertIsNone(s.media_path)
        self.assertEqual(s.extra, {})

    def test_full_roundtrip(self) -> None:
        frames = [FrameMetadata(index=0, timestamp_ms=0, width=1, height=1)]
        events = [Event(type="e", start_frame=0, end_frame=1)]
        s = Sample(
            sample_id="s1",
            media_path="/m.mp4",
            original_video="/o.mp4",
            input_dir="/in",
            scenario_description="desc",
            frames=frames,
            events=events,
            extra={"k": 1},
        )
        d = s.model_dump()
        self.assertEqual(d["sample_id"], "s1")
        self.assertEqual(len(d["frames"]), 1)
        self.assertEqual(len(d["events"]), 1)


class TestRuleFinding(unittest.TestCase):
    def test_defaults(self) -> None:
        r = RuleFinding(rule_id="r1", stage="s1", passed=True)
        self.assertEqual(r.scores, [])
        self.assertIsNone(r.severity)
        self.assertIsNone(r.message)
        self.assertEqual(r.details, {})

    def test_with_scores(self) -> None:
        r = RuleFinding(
            rule_id="r",
            stage="st",
            passed=False,
            scores=[ScoreValue(key="overlap", value=0.3)],
            severity="high",
            message="failed",
            details={"x": 1},
        )
        self.assertEqual(len(r.scores), 1)
        self.assertEqual(r.scores[0].value, 0.3)


class TestValidationErrors(unittest.TestCase):
    def test_frame_metadata_requires_fields(self) -> None:
        with self.assertRaises(ValidationError):
            FrameMetadata()  # type: ignore[call-arg]

    def test_score_value_requires_key(self) -> None:
        with self.assertRaises(ValidationError):
            ScoreValue(value=1.0)  # type: ignore[call-arg]

    def test_frame_metadata_rejects_invalid_types(self) -> None:
        cases = [
            ("non-numeric width", {"index": 0, "timestamp_ms": 0, "width": "wide", "height": 1}),
            ("non-numeric height", {"index": 0, "timestamp_ms": 0, "width": 1, "height": "tall"}),
            ("non-numeric index", {"index": "first", "timestamp_ms": 0, "width": 1, "height": 1}),
        ]
        for name, kwargs in cases:
            with self.subTest(case=name), self.assertRaises(ValidationError):
                FrameMetadata(**kwargs)

    def test_score_value_rejects_uncoercible_value(self) -> None:
        with self.assertRaises(ValidationError):
            ScoreValue(key="k", value="not-a-number")

    def test_event_rejects_invalid_attributes_type(self) -> None:
        with self.assertRaises(ValidationError):
            Event(type="x", start_frame=0, end_frame=1, attributes=["not", "a", "dict"])


class TestBoundaryValues(unittest.TestCase):
    def test_score_value_accepts_extreme_floats(self) -> None:
        cases = [
            ("zero", 0.0),
            ("negative", -1.5),
            ("very large", 1e308),
            ("very small", 1e-308),
        ]
        for name, value in cases:
            with self.subTest(case=name):
                s = ScoreValue(key="k", value=value)
                self.assertEqual(s.value, value)

    def test_event_allows_zero_length_and_inverted_ranges(self) -> None:
        """No range constraint is enforced today; capture current behavior."""
        cases = [
            ("zero-length", 5, 5),
            ("inverted", 10, 5),
        ]
        for name, start, end in cases:
            with self.subTest(case=name):
                e = Event(type="x", start_frame=start, end_frame=end)
                self.assertEqual((e.start_frame, e.end_frame), (start, end))

    def test_dyn_params_accepts_zero_and_negative_overrides(self) -> None:
        """DynParams currently has no range constraints; lock in that behavior."""
        p = DynParams(grad_thresh=0.0, blur_ksize=0, resize_factor=-1.0, max_frames=0)
        self.assertEqual(p.grad_thresh, 0.0)
        self.assertEqual(p.blur_ksize, 0)
        self.assertEqual(p.resize_factor, -1.0)
        self.assertEqual(p.max_frames, 0)


if __name__ == "__main__":
    unittest.main()
