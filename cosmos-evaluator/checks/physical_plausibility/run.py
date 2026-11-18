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

"""Command-line entry point for physical-plausibility checking."""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any
import uuid

from dotenv import load_dotenv
import yaml

from checks.physical_plausibility.processor import PhysicalPlausibilityProcessor
from checks.utils.config_manager import ConfigManager
from checks.utils.multistorage import setup_msc_config, validate_uri
from utils import bazel

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Physical Plausibility Checker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--clip-id", type=str, default=str(uuid.uuid4()), help="Clip ID for processing")
    parser.add_argument("--video-path", type=str, required=True, help="Path or URI to the video file")
    parser.add_argument(
        "--bad-threshold",
        type=float,
        default=None,
        help="Override the implausibility score threshold from the checkpoint or configuration",
    )
    parser.add_argument(
        "--checkpoint-uri",
        type=str,
        default=None,
        help="Override the attention bundle URI from the configuration",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="cosmos_evaluator",
        help="Configuration file name (without .yaml extension)",
    )
    parser.add_argument("--config-dir", type=str, default=None, help="Path to the configuration directory")
    parser.add_argument(
        "--verbose",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default="checks/physical_plausibility/.env",
        help="Path to the .env file to load",
    )
    parser.add_argument("--output-dir", type=str, default="outputs", help="Path to the output directory")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> bool:
    """Validate command-line arguments."""
    if args.bad_threshold is not None and not 0 <= args.bad_threshold <= 1:
        logger.error("bad-threshold must be in [0, 1]")
        return False
    return validate_uri(args.video_path)


def load_config(config_name: str = "cosmos_evaluator", config_dir: str | None = None) -> dict[str, Any] | None:
    """Load a configuration file by name."""
    return ConfigManager(config_dir).load_config(config_name)


def save_results_to_file(results: dict[str, Any], clip_id: str, output_dir: str) -> str:
    """Save a check result as JSON and return its path."""
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    clip_id = "".join(character if character.isalnum() or character in ".-_" else "_" for character in clip_id)
    clip_id = clip_id or "clip"
    output_file = (output_path / f"{clip_id}.physical_plausibility.results.json").resolve()
    if output_path not in output_file.parents:
        raise ValueError("results output file must be within the output directory")
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    return str(output_file)


def _resolve_config_dir(config_dir: str | None) -> str | None:
    """Locate the bundled configuration when invoked through Bazel."""
    if config_dir is not None:
        return config_dir
    resolved = bazel.get_runfiles_path("checks/cosmos_evaluator.yaml", external_repo="cosmos_eval", source_repo="")
    if resolved is not None:
        return str(Path(resolved).parent)
    return None


def main() -> None:
    """Run physical-plausibility checking for one video."""
    args = parse_args()

    if os.path.exists(args.env_file):
        load_dotenv(dotenv_path=args.env_file)
    else:
        logger.warning(".env file not found: %s", args.env_file)
    setup_msc_config(os.environ.get("MULTISTORAGECLIENT_CONFIGURATION"))

    config_dir = _resolve_config_dir(args.config_dir)
    try:
        config = load_config(args.config, config_dir)
    except FileNotFoundError as error:
        logger.error("Configuration file '%s.yaml' not found: %s", args.config, error)
        sys.exit(1)
    except yaml.YAMLError as error:
        logger.error("Invalid YAML syntax in configuration file '%s.yaml': %s", args.config, error)
        sys.exit(1)

    if config is None:
        logger.error("Failed to load configuration")
        sys.exit(1)
    params = config.get("metropolis.physical_plausibility", {})

    if not validate_args(args):
        logger.error("Argument validation failed")
        sys.exit(1)

    try:
        processor = PhysicalPlausibilityProcessor(
            params=params,
            checkpoint_uri=args.checkpoint_uri,
            config_dir=config_dir,
        )
        result = processor.process(args.clip_id, args.video_path, bad_threshold=args.bad_threshold)
        output_file = save_results_to_file(result.model_dump(), args.clip_id, args.output_dir)
        logger.info("\nCheck complete, results saved to: %s", output_file)
        logger.info("\nPrediction: %s", result.prediction)
        logger.info("Passed: %s", result.passed)
        logger.info("Bad score: %s", result.bad_score)
        logger.info("Plausibility score: %s", result.plausibility_score)
        logger.info("Bad threshold: %s", result.bad_threshold)
    except Exception:
        logger.exception("Error processing video")
        sys.exit(1)


if __name__ == "__main__":
    main()
