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

"""Unit tests for the physical-plausibility CLI entrypoint."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from checks.physical_plausibility.run import main, parse_args, save_results_to_file, validate_args


class TestPhysicalPlausibilityRun(unittest.TestCase):
    """Covers CLI parsing, validation, result serialization, and orchestration."""

    def test_parse_args_accepts_all_options(self) -> None:
        with patch(
            "sys.argv",
            [
                "run.py",
                "--clip-id",
                "clip-123",
                "--video-path",
                "/video.mp4",
                "--bad-threshold",
                "0.25",
                "--checkpoint-uri",
                "/models/checker.pt",
                "--config",
                "custom",
                "--config-dir",
                "/configs",
                "--verbose",
                "DEBUG",
                "--env-file",
                "/env-file",
                "--output-dir",
                "/output",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.clip_id, "clip-123")
        self.assertEqual(args.video_path, "/video.mp4")
        self.assertEqual(args.bad_threshold, 0.25)
        self.assertEqual(args.checkpoint_uri, "/models/checker.pt")
        self.assertEqual(args.config, "custom")
        self.assertEqual(args.config_dir, "/configs")
        self.assertEqual(args.verbose, "DEBUG")
        self.assertEqual(args.env_file, "/env-file")
        self.assertEqual(args.output_dir, "/output")

    @patch("checks.physical_plausibility.run.validate_uri", return_value=True)
    def test_validate_args_enforces_threshold_and_validates_uri(self, validate_uri: MagicMock) -> None:
        self.assertTrue(validate_args(argparse.Namespace(bad_threshold=0.5, video_path="s3://bucket/video.mp4")))
        validate_uri.assert_called_once_with("s3://bucket/video.mp4")
        self.assertFalse(validate_args(argparse.Namespace(bad_threshold=1.1, video_path="ignored")))

    def test_save_results_to_file_writes_expected_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = save_results_to_file({"passed": True}, "clip-123", temp_dir)

            self.assertEqual(Path(output_file).name, "clip-123.physical_plausibility.results.json")
            with open(output_file, encoding="utf-8") as file:
                self.assertEqual(json.load(file), {"passed": True})

    def test_save_results_to_file_sanitizes_clip_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = save_results_to_file({"passed": True}, "../../escaped", temp_dir)

            self.assertEqual(Path(output_file).parent, Path(temp_dir).resolve())
            self.assertEqual(Path(output_file).name, ".._.._escaped.physical_plausibility.results.json")

    def test_save_results_to_file_rejects_escaping_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as outside_dir:
            output_path = Path(temp_dir)
            output_path.mkdir(exist_ok=True)
            output_file = output_path / "clip.physical_plausibility.results.json"
            output_file.symlink_to(Path(outside_dir) / "escaped.json")

            with self.assertRaises(ValueError):
                save_results_to_file({"passed": True}, "clip", temp_dir)

    @patch("checks.physical_plausibility.run.save_results_to_file", return_value="/output/result.json")
    @patch("checks.physical_plausibility.run.PhysicalPlausibilityProcessor")
    @patch("checks.physical_plausibility.run.validate_args", return_value=True)
    @patch(
        "checks.physical_plausibility.run.load_config", return_value={"metropolis.physical_plausibility": {"fps": 8}}
    )
    @patch("checks.physical_plausibility.run._resolve_config_dir", return_value="/configs")
    @patch("checks.physical_plausibility.run.setup_msc_config")
    @patch("checks.physical_plausibility.run.os.path.exists", return_value=False)
    def test_main_processes_video_and_serializes_result(
        self,
        _env_file_exists: MagicMock,
        setup_msc_config: MagicMock,
        resolve_config_dir: MagicMock,
        load_config: MagicMock,
        validate: MagicMock,
        processor_class: MagicMock,
        save_results: MagicMock,
    ) -> None:
        result = MagicMock()
        result.model_dump.return_value = {"passed": True}
        result.prediction = "plausible"
        result.passed = True
        result.bad_score = 0.1
        result.plausibility_score = 0.9
        result.bad_threshold = 0.5
        processor_class.return_value.process.return_value = result

        with patch("sys.argv", ["run.py", "--clip-id", "clip-123", "--video-path", "/video.mp4"]):
            main()

        setup_msc_config.assert_called_once()
        resolve_config_dir.assert_called_once_with(None)
        load_config.assert_called_once_with("cosmos_evaluator", "/configs")
        validate.assert_called_once()
        processor_class.assert_called_once_with(params={"fps": 8}, checkpoint_uri=None, config_dir="/configs")
        processor_class.return_value.process.assert_called_once_with("clip-123", "/video.mp4", bad_threshold=None)
        save_results.assert_called_once_with({"passed": True}, "clip-123", "outputs")


if __name__ == "__main__":
    unittest.main()
