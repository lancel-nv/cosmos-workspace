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

"""Unit tests for the hallucination processor."""

from collections.abc import Iterator
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from checks.hallucination.processor import HallucinationProcessingError, HallucinationProcessor
from checks.utils.video import VideoDecodeError
from utils.cancellation import ProcessingCancelled


def _fake_frame_iter(frames: list[np.ndarray]) -> Iterator[np.ndarray]:
    """Generator test double for iter_video_frames (a successful decode yielding the given BGR frames)."""
    yield from frames


def _raising_frame_iter() -> Iterator[np.ndarray]:
    """Generator test double for iter_video_frames where ffmpeg fails to probe/open the video on first iteration."""
    yield from ()  # make this a generator that yields nothing, then fail when first advanced
    raise VideoDecodeError("simulated open failure")


class TestHallucinationProcessor(unittest.TestCase):
    def test_process_real_videos_smoke(self) -> None:
        """Real-data smoke test to guard against accidental API regressions."""
        processor = HallucinationProcessor({})
        result = processor.process(
            clip_id="test_clip_id",
            augmented_video_path="./checks/sample_data/augmented.mp4",
            original_video_path="./checks/sample_data/original.mp4",
        )
        self.assertTrue(result.passed)
        # Coupled to how the augmented/original fixtures are encoded, not just to the algorithm: the score is derived
        # from pixel gradients, so a delta this tight pins the fixture bitstream too. Re-derive it whenever those
        # fixtures are re-encoded (the AV1 transcode moved it from 0.7826).
        self.assertAlmostEqual(result.score, 0.7780, delta=0.0001)

    @patch("checks.hallucination.processor.HallucinationProcessor.get_default_config")
    def test_init_overrides_and_invalid_threshold(self, mock_default_config: MagicMock) -> None:
        mock_default_config.return_value = {"metropolis.hallucination": {"threshold": 0.2, "max_frames": 5}}

        processor = HallucinationProcessor({"threshold": 0.8, "max_frames": 2})
        self.assertEqual(processor.threshold, 0.8)
        self.assertEqual(processor.max_frames, 2)

        with self.assertRaises(ValueError):
            HallucinationProcessor({"threshold": 1.2})

    @patch("checks.hallucination.processor.os.remove")
    @patch("checks.hallucination.processor.os.path.exists")
    @patch("checks.hallucination.processor.iter_video_frames")
    @patch("checks.hallucination.processor.download_if_remote")
    def test_process_cleanup_paths_and_re_raise(
        self,
        mock_download_if_remote: MagicMock,
        mock_iter_video_frames: MagicMock,
        mock_exists: MagicMock,
        mock_remove: MagicMock,
    ) -> None:
        processor = HallucinationProcessor({})

        mock_iter_video_frames.side_effect = [_raising_frame_iter(), _fake_frame_iter([])]
        mock_download_if_remote.side_effect = ["/tmp/orig.mp4", "/tmp/aug.mp4"]
        mock_exists.return_value = True
        mock_remove.side_effect = [None, OSError("remove failed")]

        with self.assertRaises(HallucinationProcessingError):
            processor.process("clip", "s3://bucket/orig.mp4", "s3://bucket/aug.mp4")

        self.assertEqual(mock_remove.call_count, 2)

    @patch("checks.hallucination.processor.iter_video_frames")
    @patch("checks.hallucination.processor.download_if_remote")
    def test_process_raises_when_augmented_video_fails_to_open(
        self,
        mock_download_if_remote: MagicMock,
        mock_iter_video_frames: MagicMock,
    ) -> None:
        processor = HallucinationProcessor({})
        mock_download_if_remote.side_effect = ["orig.mp4", "aug.mp4"]

        mock_iter_video_frames.side_effect = [
            _fake_frame_iter([np.zeros((2, 2, 3), dtype=np.uint8)]),
            _raising_frame_iter(),
        ]

        with self.assertRaises(HallucinationProcessingError):
            processor.process("clip", "orig.mp4", "aug.mp4")

    @patch("checks.hallucination.processor.iter_video_frames")
    @patch("checks.hallucination.processor.download_if_remote")
    def test_process_raises_when_first_frame_read_fails(
        self,
        mock_download_if_remote: MagicMock,
        mock_iter_video_frames: MagicMock,
    ) -> None:
        processor = HallucinationProcessor({})
        # Each process() call resolves both the original and augmented paths, so seed enough values for both blocks.
        mock_download_if_remote.side_effect = ["orig.mp4", "aug.mp4", "orig.mp4", "aug.mp4"]

        # Original yields no frames -> first-frame read fails for the original.
        mock_iter_video_frames.side_effect = [
            _fake_frame_iter([]),
            _fake_frame_iter([np.zeros((2, 2, 3), dtype=np.uint8)]),
        ]
        with self.assertRaises(HallucinationProcessingError):
            processor.process("clip", "orig.mp4", "aug.mp4")

        # Augmented yields no frames -> first-frame read fails for the augmented.
        mock_iter_video_frames.side_effect = [
            _fake_frame_iter([np.zeros((2, 2, 3), dtype=np.uint8)]),
            _fake_frame_iter([]),
        ]
        with self.assertRaises(HallucinationProcessingError):
            processor.process("clip", "orig.mp4", "aug.mp4")

    @patch("checks.hallucination.processor.hallucination_counts")
    @patch("checks.hallucination.processor.compute_dynamic_mask")
    @patch("checks.hallucination.processor.to_gray")
    @patch("checks.hallucination.processor.ensure_same_size")
    @patch("checks.hallucination.processor.iter_video_frames")
    @patch("checks.hallucination.processor.download_if_remote")
    def test_process_scores_with_and_without_augmented_dynamic_pixels(
        self,
        mock_download_if_remote: MagicMock,
        mock_iter_video_frames: MagicMock,
        mock_ensure_same_size: MagicMock,
        mock_to_gray: MagicMock,
        mock_compute_dynamic_mask: MagicMock,
        mock_hallucination_counts: MagicMock,
    ) -> None:
        processor = HallucinationProcessor({"threshold": 0.75, "max_frames": 2})
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        mock_download_if_remote.side_effect = lambda path: path
        mock_ensure_same_size.side_effect = lambda f, _shape: f
        mock_to_gray.side_effect = lambda f: np.zeros((4, 4), dtype=np.uint8)
        mock_compute_dynamic_mask.side_effect = [
            (np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)),
            (np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8)),
        ]
        mock_hallucination_counts.return_value = (2, 10)

        mock_iter_video_frames.side_effect = [
            _fake_frame_iter([frame, frame]),
            _fake_frame_iter([frame, frame]),
        ]

        result = processor.process("clip", "orig.mp4", "aug.mp4")
        self.assertAlmostEqual(result.score, 0.8)
        self.assertTrue(result.passed)
        self.assertEqual(result.total_frames, 2)

        # Zero dynamic pixels path: score forced to 1.0.
        mock_hallucination_counts.return_value = (0, 0)
        mock_iter_video_frames.side_effect = [
            _fake_frame_iter([frame]),
            _fake_frame_iter([frame]),
        ]
        result = processor.process("clip2", "orig.mp4", "aug.mp4")
        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.passed)

    @patch("checks.hallucination.processor.iter_video_frames")
    @patch("checks.hallucination.processor.download_if_remote", side_effect=lambda path: path)
    def test_process_reports_mismatched_frame_counts(
        self,
        _mock_download_if_remote: MagicMock,
        mock_iter_video_frames: MagicMock,
    ) -> None:
        processor = HallucinationProcessor({})
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        cases = [
            (3, 2, "1 extra input + 0 extra output frame(s) ignored"),
            (2, 4, "0 extra input + 2 extra output frame(s) ignored"),
        ]

        for original_count, augmented_count, expected_warning in cases:
            with self.subTest(original_count=original_count, augmented_count=augmented_count):
                mock_iter_video_frames.side_effect = [
                    _fake_frame_iter([frame] * original_count),
                    _fake_frame_iter([frame] * augmented_count),
                ]

                with self.assertLogs(processor.logger, level="WARNING") as captured:
                    result = processor.process("clip", "orig.mp4", "aug.mp4")

                self.assertEqual(result.total_frames, min(original_count, augmented_count))
                self.assertIn(f"compared {result.total_frames} aligned frames", captured.output[0])
                self.assertIn(expected_warning, captured.output[0])

    @patch("checks.hallucination.processor.iter_video_frames")
    @patch("checks.hallucination.processor.download_if_remote", side_effect=lambda path: path)
    def test_max_frames_does_not_report_unexamined_tail_as_a_mismatch(
        self,
        _mock_download_if_remote: MagicMock,
        mock_iter_video_frames: MagicMock,
    ) -> None:
        processor = HallucinationProcessor({"max_frames": 2})
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        mock_iter_video_frames.side_effect = [
            _fake_frame_iter([frame] * 4),
            _fake_frame_iter([frame] * 2),
        ]

        with self.assertNoLogs(processor.logger, level="WARNING"):
            result = processor.process("clip", "orig.mp4", "aug.mp4")

        self.assertEqual(result.total_frames, 2)

    @patch("checks.hallucination.processor.perf_counter", side_effect=[10.0, 12.5])
    @patch("checks.hallucination.processor.iter_video_frames")
    @patch("checks.hallucination.processor.download_if_remote", side_effect=lambda path: path)
    def test_process_reports_duration_seconds(
        self,
        _mock_download_if_remote: MagicMock,
        mock_iter_video_frames: MagicMock,
        _mock_perf_counter: MagicMock,
    ) -> None:
        processor = HallucinationProcessor({})
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        mock_iter_video_frames.side_effect = [
            _fake_frame_iter([frame]),
            _fake_frame_iter([frame]),
        ]

        with self.assertNoLogs(processor.logger, level="WARNING"):
            result = processor.process("clip", "orig.mp4", "aug.mp4")

        self.assertEqual(result.duration_seconds, 2.5)

    @patch("checks.hallucination.processor.hallucination_counts")
    @patch("checks.hallucination.processor.compute_dynamic_mask")
    @patch("checks.hallucination.processor.to_gray")
    @patch("checks.hallucination.processor.ensure_same_size")
    @patch("checks.hallucination.processor.iter_video_frames")
    @patch("checks.hallucination.processor.download_if_remote")
    def test_a_set_cancel_event_stops_the_frame_loop(
        self,
        mock_download_if_remote: MagicMock,
        mock_iter_video_frames: MagicMock,
        mock_ensure_same_size: MagicMock,
        mock_to_gray: MagicMock,
        mock_compute_dynamic_mask: MagicMock,
        mock_hallucination_counts: MagicMock,
    ) -> None:
        """An abandoned job unwinds at the next frame instead of decoding the rest of the clip."""
        processor = HallucinationProcessor({})
        frame = np.zeros((4, 4, 3), dtype=np.uint8)

        mock_download_if_remote.side_effect = lambda path: path
        mock_ensure_same_size.side_effect = lambda f, _shape: f
        mock_to_gray.side_effect = lambda f: np.zeros((4, 4), dtype=np.uint8)
        mock_iter_video_frames.side_effect = [
            _fake_frame_iter([frame] * 10),
            _fake_frame_iter([frame] * 10),
        ]
        cancel_event = threading.Event()
        cancel_event.set()

        with self.assertRaises(ProcessingCancelled):
            processor.process("clip", "orig.mp4", "aug.mp4", cancel_event)

        # Cancellation was noticed before any frame pair was compared, not after the clip ran out.
        mock_hallucination_counts.assert_not_called()
        mock_compute_dynamic_mask.assert_not_called()

    @patch("checks.hallucination.processor.ConfigManager")
    def test_get_default_config_success_and_failure(self, mock_config_manager: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_instance.load_config.return_value = {"metropolis.hallucination": {"threshold": 0.9}}
        mock_config_manager.return_value = mock_instance

        config = HallucinationProcessor.get_default_config()
        self.assertEqual(config["metropolis.hallucination"]["threshold"], 0.9)

        mock_instance.load_config.side_effect = RuntimeError("bad config")
        with self.assertRaises(RuntimeError):
            HallucinationProcessor.get_default_config()


if __name__ == "__main__":
    unittest.main()
