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
Common CLI utilities shared across object-evaluation checks.

This module provides reusable functions for command-line interface operations that are shared across multiple check
modules (objects, hallucination, attribute verification, physical plausibility, etc.).
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_paths(
    input_data: str,
    video_path: str,
    output_dir: str,
    create_output_dir: bool = True,
) -> bool:
    """
    Validate that input paths exist and optionally create output directory.

    This function validates the actual file paths regardless of how they were
    obtained (CLI args or S3 download).

    Args:
        input_data: Path to input data directory (e.g., RDS HQ dataset)
        video_path: Path to video file
        output_dir: Path to output directory
        create_output_dir: Whether to create output directory if it doesn't exist

    Returns:
        True if validation passes, False otherwise
    """
    # Check if input data directory exists
    if not Path(input_data).exists():
        logger.error("Input data directory does not exist: %s", input_data)
        return False

    # Check if video file exists
    if not Path(video_path).exists():
        logger.error("Video file does not exist: %s", video_path)
        return False

    # Create output directory if requested
    if create_output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    return True
