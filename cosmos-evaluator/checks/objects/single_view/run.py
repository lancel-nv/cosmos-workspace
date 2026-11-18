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
Entry point for object processing.

This script provides a unified interface for running object checks with configuration management and command line
argument parsing.
"""

import argparse
import logging
from pathlib import Path
import sys
import traceback
from typing import Any

from checks.objects.single_view.api import run_object_processors
from checks.utils.cli_common import validate_paths
from checks.utils.config_manager import ConfigManager

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Expected CLI:
      dazel run //checks/objects/single_view:run -- [args...]
    """
    parser = argparse.ArgumentParser(
        description="Objects Processing Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--camera_name", type=str, required=True, help="Camera name for processing")
    parser.add_argument(
        "--verbose",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="cosmos_evaluator",
        help="Configuration file name (without .yaml extension)",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=None,
        help="Trial mode: evaluate only the first N steps of the 10 fps object-check timeline (for debugging)",
    )
    parser.add_argument(
        "--model_device", type=str, choices=["cuda", "cpu"], default="cuda", help="Device to run the model on"
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help=(
            "Disable model inference caching. By default, SegFormer masks and CWIP inference "
            "results are cached under ~/.cache/cosmos_evaluator/inference/<clip_id>/ and reused on "
            "subsequent runs with the same video, skipping inference. Pass this flag to disable "
            "caching (e.g. when testing with modified models)."
        ),
    )
    parser.add_argument("--input_data", type=str, required=True, help="Path to the input data directory")
    parser.add_argument("--clip_id", type=str, required=True, help="Clip ID for processing")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the camera video file")
    parser.add_argument(
        "--world_video_path",
        type=str,
        default=None,
        help="Path to the world video file for CWIP static processing. If omitted, the static (CWIP) check runs "
        "camera-only on the camera stream alone: CWIP runs without a world-model reference and the per-class presence "
        "mask is derived from the type head, tuned via the av.objects_single_view.presence.type_mask config block. To "
        "run camera-only, omit this flag.",
    )

    return parser.parse_args()


def load_config(config_name: str = "cosmos_evaluator") -> dict[str, Any] | None:
    """
    Load configuration from the specified config file.

    Args:
        config_name: Name of the config file (without .yaml extension)

    Returns:
        Configuration dictionary or None if loading fails
    """
    try:
        # Try default ConfigManager (works for direct Python execution)
        config_manager = ConfigManager()
        config = config_manager.load_config(config_name)
        return config
    except FileNotFoundError as e:
        logger.warning("Configuration file '%s.yaml' not found: %s", config_name, e)
    except Exception as e:  # noqa: BLE001 - config load fallback; log and return None
        logger.error("Error loading configuration: %s", e)
    return None


def validate_args(args: argparse.Namespace) -> bool:
    """Validate command line arguments."""
    # A missing --world_video_path is valid: the static check then runs camera-only. Only a *provided* path that does
    # not exist is an error.
    if args.world_video_path is not None and not Path(args.world_video_path).exists():
        logger.error("World video file does not exist: %s", args.world_video_path)
        return False
    return True


def _resolve_mask_cache_dir(no_cache: bool, clip_id: str, *subpath: str) -> str | None:
    """Return the inference cache directory, or None if caching is disabled.

    The default location is ~/.cache/cosmos_evaluator/inference/<clip_id>/[subpath...]. The directory is created if it
    does not already exist.
    """
    if no_cache:
        return None
    safe_clip_id = clip_id.replace("/", "_").replace("\\", "_")
    cache_dir = Path.home() / ".cache" / "cosmos_evaluator" / "inference" / safe_clip_id
    for part in subpath:
        cache_dir = cache_dir / part
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


def _print_results(results: dict[str, Any], verbose: bool) -> None:
    """Print processing results summary.

    Args:
        results: Results dictionary from a processor.
        verbose: Whether to print detailed summary.
    """
    print(f"  - Total frames processed: {results['processed_frames']}")
    if results.get("track_ids"):
        print(f"  - Unique track IDs: {len(results['track_ids'])}")
    if "score_matrix" in results and results["score_matrix"] is not None:
        print(f"  - Score matrix shape: {results['score_matrix'].shape}")
    print(f"  - Mean correspondence score: {results['mean_score']:.3f}")
    print(f"  - Score range: {results['min_score']:.3f} - {results['max_score']:.3f}")
    print(f"  - Standard deviation: {results['std_score']:.3f}")
    signal_group_scores = results.get("signal_group_scores") or {}
    if signal_group_scores:
        print("  - Signal group scores:")
        for group, group_stats in signal_group_scores.items():
            print(f"    * {group}: {group_stats['mean_score']:.3f}")

    if verbose:
        # Print a more readable summary instead of the raw results dictionary
        print("\nDetailed Summary:")
        print(f"  - Processed frames: {results['processed_frames']}")
        print(f"  - Unique track IDs: {len(results['track_ids'])}")
        print("  - Score statistics:")
        print(f"    * Mean: {results['mean_score']:.3f}")
        print(f"    * Std: {results['std_score']:.3f}")
        print(f"    * Min: {results['min_score']:.3f}")
        print(f"    * Max: {results['max_score']:.3f}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if not validate_args(args):
        logger.error("Argument validation failed. Exiting.")
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)
    if config is None:
        print("Failed to load configuration. Exiting.")
        sys.exit(1)

    if args.verbose:
        print(f"Loaded configuration from '{args.config}.yaml'")
        print(f"Configuration: {config}")

    inputs = {
        "video_path": args.video_path,
        "rds_hq_path": args.input_data,
        "clip_id": args.clip_id,
    }
    world_video_path: str | None = args.world_video_path

    # The static (CWIP) check always runs. It runs camera-only when no world-model reference is available, i.e. when
    # --world_video_path was omitted.
    camera_only = world_video_path is None
    if camera_only:
        logger.info("Camera-only static mode: CWIP will run on the camera video alone (no world-model reference).")

    # Validate input paths exist and ensure output dir exists
    if not validate_paths(
        input_data=inputs["rds_hq_path"],
        video_path=inputs["video_path"],
        output_dir=args.output_dir,
        create_output_dir=True,
    ):
        print("Input path validation failed. Exiting.")
        sys.exit(1)

    mask_cache_dir = _resolve_mask_cache_dir(args.no_cache, inputs["clip_id"])

    # Import and run the actual processing
    try:
        # Display configuration summary
        print("Configuration loaded successfully:")
        print(f"  - Overlap check: {config.get('overlap_check', {})}")
        print(f"  - Importance filter: {config.get('importance_filter', {})}")
        print(f"  - Model device: {args.model_device}")

        print("\nProcessing with arguments:")
        print(f"  - Input data: {inputs['rds_hq_path']}")
        print(f"  - Clip ID: {inputs['clip_id']}")
        print(f"  - Camera name: {args.camera_name}")
        print(f"  - Video path: {inputs['video_path']}")
        if camera_only:
            print("  - World video path: N/A (camera-only static mode)")
        else:
            print(f"  - World video path: {world_video_path}")
        print(f"  - Output directory: {args.output_dir}")
        print(f"  - Log level: {args.verbose}")
        if args.trial is not None:
            print(f"  - Trial mode: {args.trial} frames")
        if mask_cache_dir is not None:
            print(f"  - Inference cache dir: {mask_cache_dir}")
        else:
            print("  - Inference cache: disabled (--no_cache)")

        # Process the clip using the unified run() function
        print("\nStarting object processing...")
        combined_results = run_object_processors(
            config=config,
            input_data=inputs["rds_hq_path"],
            clip_id=inputs["clip_id"],
            camera_name=args.camera_name,
            video_path=inputs["video_path"],
            output_dir=args.output_dir,
            world_video_path=world_video_path,
            model_device=args.model_device,
            verbose=args.verbose,
            trial_frames=args.trial,
            mask_cache_dir=mask_cache_dir,
        )

        # Display results for both processors
        print("\nProcessing complete!")
        for processor_name, results in combined_results.items():
            print(f"\n=== {processor_name.upper()} Processor Results ===")
            _print_results(results, args.verbose)

    except ImportError as e:
        print(f"Error importing processing module: {e}")
        sys.exit(1)
    except Exception:  # noqa: BLE001 - CLI top-level handler prints traceback and exits non-zero
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
