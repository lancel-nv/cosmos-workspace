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

"""Unit tests for checks.utils.cli_common module."""

import os
import tempfile
import unittest

from checks.utils.cli_common import validate_paths


class TestValidatePaths(unittest.TestCase):
    """Test suite for validate_paths function."""

    def test_validate_paths_missing_input_data(self) -> None:
        """Test validation fails when input_data directory doesn't exist."""
        self.assertFalse(
            validate_paths(
                input_data="/does/not/exist",
                video_path="/also/missing.mp4",
                output_dir="/tmp/output",
            )
        )

    def test_validate_paths_missing_video(self) -> None:
        """Test validation fails when video file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_file = os.path.join(tmp_dir, "data.txt")
            open(data_file, "a").close()

            self.assertFalse(
                validate_paths(
                    input_data=data_file,
                    video_path="/does/not/exist.mp4",
                    output_dir=tmp_dir,
                )
            )

    def test_validate_paths_success_without_creating_output_dir(self) -> None:
        """Test validate_paths with valid paths without creating output dir."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create temporary files
            data_file = os.path.join(tmp_dir, "data.txt")
            video_file = os.path.join(tmp_dir, "video.mp4")
            output_dir_path = os.path.join(tmp_dir, "output")
            open(data_file, "a").close()
            open(video_file, "a").close()

            self.assertTrue(
                validate_paths(
                    input_data=data_file,
                    video_path=video_file,
                    output_dir=output_dir_path,
                    create_output_dir=False,
                )
            )
            # Verify output dir was NOT created
            self.assertFalse(os.path.exists(output_dir_path))

    def test_validate_paths_success_with_creating_output_dir(self) -> None:
        """Test validate_paths with valid paths and creates output dir."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create temporary files
            data_file = os.path.join(tmp_dir, "data.txt")
            video_file = os.path.join(tmp_dir, "video.mp4")
            output_dir_path = os.path.join(tmp_dir, "output")
            open(data_file, "a").close()
            open(video_file, "a").close()

            self.assertTrue(
                validate_paths(
                    input_data=data_file,
                    video_path=video_file,
                    output_dir=output_dir_path,
                    create_output_dir=True,
                )
            )
            # Verify output dir WAS created
            self.assertTrue(os.path.exists(output_dir_path))


if __name__ == "__main__":
    unittest.main()
