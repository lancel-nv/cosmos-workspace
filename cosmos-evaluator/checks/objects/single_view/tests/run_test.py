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

"""Unit tests for run.py CLI script."""

import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from checks.objects.single_view.run import main, parse_args, validate_args
from utils.bazel import get_runfiles_path

logger = logging.getLogger(__name__)


class TestObjectsRun(unittest.TestCase):
    """Test case for the objects run.py CLI script."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Get path to the RDS data zip file
        rds_zip_path = get_runfiles_path("checks/sample_data/rds_hq_data.zip")
        if rds_zip_path is None:
            self.fail("Could not locate rds_hq_data.zip file")
        self.rds_zip_path: str = rds_zip_path
        self.assertTrue(os.path.exists(rds_zip_path), f"RDS zip file does not exist: {rds_zip_path}")

        # Get path to the video file
        self.video_path = get_runfiles_path("checks/sample_data/camera_front_wide_120fov.mp4")
        self.assertIsNotNone(self.video_path, "Could not locate camera_front_wide_120fov.mp4 file")
        assert self.video_path is not None
        self.assertTrue(os.path.exists(self.video_path), f"Video file does not exist: {self.video_path}")

        self.world_video_path = get_runfiles_path(
            "checks/sample_data/0d21d408-ceca-4af6-9c4b-4f6f78ee7459_1727936028200000_0.mp4"
        )
        self.assertIsNotNone(
            self.world_video_path, "Could not locate 0d21d408-ceca-4af6-9c4b-4f6f78ee7459_1727936028200000_0.mp4 file"
        )
        assert self.world_video_path is not None
        self.assertTrue(
            os.path.exists(self.world_video_path), f"World video file does not exist: {self.world_video_path}"
        )

    def test_objects_run_direct(self) -> None:
        """Test run.py by calling main() function directly."""

        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract RDS data
            rds_extracted_dir = os.path.join(temp_dir, "rds_hq")
            os.makedirs(rds_extracted_dir, exist_ok=True)

            with zipfile.ZipFile(self.rds_zip_path, "r") as zf:
                # Safe extraction with path validation: skip absolute or parent-traversal entries.
                safe_members = [name for name in zf.namelist() if not name.startswith("/") and ".." not in name]
                zf.extractall(rds_extracted_dir, members=safe_members)

            # Create output directory
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir)

            # Prepare command arguments (simulate CLI args)
            test_args = [
                "run.py",  # script name (sys.argv[0])
                "--input_data",
                rds_extracted_dir,
                "--clip_id",
                "ba3d3d1c-2eaf-4e0d-8b6e-f3f5b0c4cf80_26015042000",
                "--camera_name",
                "camera_front_wide_120fov",
                "--video_path",
                self.video_path,
                "--world_video_path",
                self.world_video_path,
                "--output_dir",
                output_dir,
                "--verbose",
                "INFO",
                "--trial",
                "5",  # Process only 5 frames for faster testing
                "--model_device",
                "cuda",
            ]

            # Mock sys.argv and call main() directly
            with patch.object(sys, "argv", test_args):
                try:
                    main()

                    # Check that output directory contains some files
                    output_files = list(Path(output_dir).rglob("*"))
                    self.assertGreater(len(output_files), 0, f"No output files generated in {output_dir}")

                    logger.info("✅ Test passed! Generated %d output files", len(output_files))

                except SystemExit as e:
                    if e.code != 0:
                        self.fail(f"main() exited with non-zero code: {e.code}")
                    # SystemExit with code 0 is normal for successful CLI tools
                except (RuntimeError, ValueError, ImportError) as e:
                    self.fail(f"main() raised an exception: {e}")

    def test_objects_run_direct_camera_only(self) -> None:
        """Run the static checker end-to-end in camera-only mode: omitting --world_video_path selects it.

        Exercises the full camera-only path (CWIP camera-only inference, type-head presence mask, static scoring) and
        asserts it produces a static ``results.json`` with a populated score matrix.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            rds_extracted_dir = os.path.join(temp_dir, "rds_hq")
            os.makedirs(rds_extracted_dir, exist_ok=True)
            with zipfile.ZipFile(self.rds_zip_path, "r") as zf:
                safe_members = [name for name in zf.namelist() if not name.startswith("/") and ".." not in name]
                zf.extractall(rds_extracted_dir, members=safe_members)

            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir)

            test_args = [
                "run.py",
                "--input_data",
                rds_extracted_dir,
                "--clip_id",
                "ba3d3d1c-2eaf-4e0d-8b6e-f3f5b0c4cf80_26015042000",
                "--camera_name",
                "camera_front_wide_120fov",
                "--video_path",
                self.video_path,
                # Deliberately no --world_video_path: this selects camera-only static.
                "--output_dir",
                output_dir,
                "--verbose",
                "INFO",
                "--trial",
                "5",
                "--model_device",
                "cuda",
            ]

            with patch.object(sys, "argv", test_args):
                try:
                    main()
                except SystemExit as e:
                    if e.code != 0:
                        self.fail(f"main() exited with non-zero code: {e.code}")
                except (RuntimeError, ValueError, ImportError) as e:
                    self.fail(f"main() raised an exception: {e}")

            static_results = list(Path(output_dir).rglob("*static*results*.json"))
            self.assertGreater(len(static_results), 0, f"No static results.json generated in {output_dir}")
            with open(static_results[0]) as f:
                static_data = json.load(f)
            # Camera-only static processing must have actually scored frames, not fallen through the empty-matrix path.
            self.assertGreater(static_data.get("processed_frames", 0), 0, "Camera-only static run scored no frames")

    def test_argument_parsing(self) -> None:
        """Test that CLI arguments are parsed correctly."""

        test_args = [
            "run.py",
            "--input_data",
            "/path/to/data",
            "--clip_id",
            "test_clip_123",
            "--camera_name",
            "test_camera",
            "--video_path",
            "/path/to/video.mp4",
            "--world_video_path",
            "/path/to/world.mp4",
            "--output_dir",
            "/path/to/output",
            "--verbose",
            "DEBUG",
            "--trial",
            "10",
        ]

        with patch.object(sys, "argv", test_args):
            args = parse_args()

            # Verify arguments were parsed correctly
            self.assertEqual(args.input_data, "/path/to/data")
            self.assertEqual(args.clip_id, "test_clip_123")
            self.assertEqual(args.camera_name, "test_camera")
            self.assertEqual(args.video_path, "/path/to/video.mp4")
            self.assertEqual(args.world_video_path, "/path/to/world.mp4")
            self.assertEqual(args.output_dir, "/path/to/output")
            self.assertEqual(args.verbose, "DEBUG")
            self.assertEqual(args.trial, 10)

    def test_validate_args_rejects_missing_world_video(self) -> None:
        """Test that validate_args returns False when the world video file does not exist."""
        base_args = [
            "run.py",
            "--input_data",
            "/data",
            "--clip_id",
            "clip1",
            "--camera_name",
            "cam",
            "--video_path",
            "/vid.mp4",
            "--output_dir",
            "/out",
        ]
        with patch.object(sys, "argv", [*base_args, "--world_video_path", "/nonexistent/world.mp4"]):
            self.assertFalse(validate_args(parse_args()))
        # Omitting the world video is valid — the static check then runs camera-only.
        with patch.object(sys, "argv", base_args):
            self.assertTrue(validate_args(parse_args()))


if __name__ == "__main__":
    unittest.main()
