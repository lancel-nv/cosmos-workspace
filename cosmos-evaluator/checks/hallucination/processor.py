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
Module for the hallucination checker.

This module handles the detection of hallucinations in the augmented video.
"""

from itertools import zip_longest
import logging
import os
import threading
from time import perf_counter
from typing import Any

from pydantic import BaseModel

from checks.hallucination.frame_processing import hallucination_counts
from checks.utils.config_manager import ConfigManager
from checks.utils.frame_processing import compute_dynamic_mask, ensure_same_size, to_gray
from checks.utils.multistorage import download_if_remote
from checks.utils.types import DynParams
from checks.utils.video import VideoDecodeError, iter_video_frames
from utils.cancellation import ProcessingCancelled, check_cancelled

logger = logging.getLogger(__name__)


class HallucinationProcessingError(Exception):
    """Raised when the hallucination processor cannot read or decode the input videos."""


class HallucinationResult(BaseModel):
    clip_id: str
    passed: bool
    threshold: float
    score: float
    total_frames: int
    total_hallucinated_dynamic_pixels: int
    total_augmented_dynamic_pixels: int
    duration_seconds: float

    @property
    def output_dir(self) -> str | None:
        """Always ``None``: this check reports pixel counts and writes no artifacts for a transport to upload.

        A property rather than a field so the answer stays out of the serialized result, which is recorded on the job
        row and read by callers who have no interest in the transport's cleanup contract.
        """
        return None


class HallucinationProcessor:
    """
    Processor for the hallucination checker.
    """

    def __init__(self, params: dict[str, Any] | None, config_dir: str | None = None, verbose: str = "INFO") -> None:
        """
        Initialize the processor.
        """
        self.config = HallucinationProcessor.get_default_config(config_dir=config_dir).get(
            "metropolis.hallucination", {}
        )
        if params is not None:
            self.config.update(params)
        self.grad_thresh = float(self.config.get("grad_thresh", 10.0))
        self.blur_ksize = int(self.config.get("blur_ksize", 7))
        self.morph_k = int(self.config.get("morph_k", 3))
        self.dist_tol_px = float(self.config.get("dist_tol_px", 7.0))
        self.moving_window_size = int(self.config.get("moving_window_size", 0))
        max_frames_val = self.config.get("max_frames")
        self.max_frames = int(max_frames_val) if max_frames_val is not None else None
        self.threshold = float(self.config.get("threshold", 0.682))
        if self.threshold < 0.0 or self.threshold > 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        self._setup_logging(verbose)

        self.logger.info("Hallucination processor initialized with the following parameters:")
        self.logger.info("- grad_thresh: %s", self.grad_thresh)
        self.logger.info("- blur_ksize: %s", self.blur_ksize)
        self.logger.info("- morph_k: %s", self.morph_k)
        self.logger.info("- dist_tol_px: %s", self.dist_tol_px)
        self.logger.info("- moving_window_size: %s", self.moving_window_size)
        self.logger.info("- max_frames: %s", self.max_frames)
        self.logger.info("- threshold: %s", self.threshold)

    def _setup_logging(self, verbose: str) -> None:
        """Setup logging."""
        self.logger = logging.getLogger(__name__)

        # Set logging level based on verbose mode
        log_level = getattr(logging, verbose.upper())
        self.logger.setLevel(log_level)

        # Ensure propagation is enabled
        self.logger.propagate = True

        # Clear any existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        self.logger.info("Logging level set to: %s", verbose)

    def process(
        self,
        clip_id: str,
        original_video_path: str,
        augmented_video_path: str,
        cancel_event: threading.Event | None = None,
    ) -> HallucinationResult:
        """
        Check for hallucinated motion in augmented video.

        Args:
            original_video_path: Path to original video
            augmented_video_path: Path to augmented video
            cancel_event: Optional cooperative-cancellation event polled once per frame pair; when set, the loop
                raises ``ProcessingCancelled``. ``None`` disables cancellation (local CLI).

        Returns:
            HallucinationResult: Result of the hallucination check

        The result contains:
            - score: Overall hallucination score [0, 1]
            - total_frames: Number of frames processed
            - total_hallucinated_pixels: Total hallucinated pixels
            - total_augmented_dynamic_pixels: Total dynamic pixels in augmented video
            - duration_seconds: Elapsed processing time, including input download and decoding
        """
        self.logger.info("Starting hallucination check...")
        self.logger.debug("Original video: %s", original_video_path)
        self.logger.debug("Augmented video: %s", augmented_video_path)
        start_time = perf_counter()

        params = DynParams(
            grad_thresh=self.grad_thresh,
            blur_ksize=self.blur_ksize,
            morph_k=self.morph_k,
            dist_tol_px=self.dist_tol_px,
            max_frames=self.max_frames,
        )

        temp_files = []

        try:
            # Download videos if remote
            orig_path = download_if_remote(original_video_path)
            aug_path = download_if_remote(augmented_video_path)

            if orig_path != original_video_path:
                temp_files.append(orig_path)
            if aug_path != augmented_video_path:
                temp_files.append(aug_path)

            # Decode videos via the vendored LGPL ffmpeg binary (cv2.VideoCapture replacement, so cv2 ships no bundled
            # ffmpeg). The generators advance in lockstep over their shared frames; closing them kills ffmpeg.
            orig_frames = iter_video_frames(orig_path)
            aug_frames = iter_video_frames(aug_path)

            try:
                # Read first frames (a failure to probe/launch ffmpeg surfaces here as VideoDecodeError).
                try:
                    fo = next(orig_frames, None)
                except VideoDecodeError as e:
                    error_msg = f"Failed to open original video downloaded from {original_video_path}"
                    self.logger.error(error_msg)
                    raise HallucinationProcessingError(error_msg) from e
                try:
                    fa = next(aug_frames, None)
                except VideoDecodeError as e:
                    error_msg = f"Failed to open augmented video downloaded from {augmented_video_path}"
                    self.logger.error(error_msg)
                    raise HallucinationProcessingError(error_msg) from e

                if fo is None:
                    error_msg = f"Failed to read first frames from original video downloaded from {original_video_path}"
                    self.logger.error(error_msg)
                    raise HallucinationProcessingError(error_msg)
                if fa is None:
                    error_msg = (
                        f"Failed to read first frames from augmented video downloaded from {augmented_video_path}"
                    )
                    self.logger.error(error_msg)
                    raise HallucinationProcessingError(error_msg)

                # Get dimensions and ensure augmented matches original
                h, w = fo.shape[:2]
                fa = ensure_same_size(fa, (h, w))
                prev_o = to_gray(fo)
                prev_a = to_gray(fa)

                # Process frames
                total_h = 0
                total_aug = 0
                frame_count = 1

                for fo, fa in zip_longest(orig_frames, aug_frames):
                    # Once per frame pair, which is the only loop long enough to matter: a full clip is thousands of
                    # iterations of mask computation, and an abandoned job that runs it out holds a slot the whole time.
                    check_cancelled(cancel_event)

                    if fo is None or fa is None:
                        # Count the first surplus frame already pulled by zip_longest plus the remaining tail. The
                        # unmatched frames are deliberately not compared, but the mismatch should be visible to users.
                        extra_o = 0 if fo is None else 1
                        while True:
                            check_cancelled(cancel_event)
                            try:
                                next(orig_frames)
                            except StopIteration:
                                break
                            extra_o += 1

                        extra_a = 0 if fa is None else 1
                        while True:
                            check_cancelled(cancel_event)
                            try:
                                next(aug_frames)
                            except StopIteration:
                                break
                            extra_a += 1
                        self.logger.warning(
                            "Hallucination check: input/output frame counts differ "
                            "(compared %s aligned frames; %s extra input + %s extra output frame(s) ignored).",
                            frame_count,
                            extra_o,
                            extra_a,
                        )
                        break

                    fa = ensure_same_size(fa, (h, w))
                    go = to_gray(fo)
                    ga = to_gray(fa)

                    # Compute dynamic masks
                    mo, prev_o = compute_dynamic_mask(prev_o, go, params)
                    ma, prev_a = compute_dynamic_mask(prev_a, ga, params)

                    # Count hallucinated pixels
                    nh, nau = hallucination_counts(mo, ma, self.dist_tol_px)
                    total_h += nh
                    total_aug += nau

                    frame_count += 1

                    # Check frame limit
                    if self.max_frames is not None and frame_count >= self.max_frames:
                        self.logger.debug("Reached max_frames limit: %s", self.max_frames)
                        break
            finally:
                # Close generators so ffmpeg is terminated/reaped even when we stop early or raise.
                orig_frames.close()
                aug_frames.close()

            # Calculate score
            if total_aug == 0:
                score = 1.0
            else:
                score = max(0.0, 1.0 - (float(total_h) / float(total_aug)))

            self.logger.info("Hallucination check complete: score=%.4f", score)
            self.logger.debug("Frames processed: %s", frame_count)
            self.logger.debug("Hallucinated pixels: %s / %s", total_h, total_aug)

            return HallucinationResult(
                clip_id=clip_id,
                passed=score >= self.threshold,
                threshold=self.threshold,
                score=float(score),
                total_frames=frame_count,
                total_hallucinated_dynamic_pixels=total_h,
                total_augmented_dynamic_pixels=total_aug,
                duration_seconds=perf_counter() - start_time,
            )

        except ProcessingCancelled:
            # An expected outcome, not a fault: the job was abandoned or the pod is shutting down. Logged without a
            # traceback so it does not read as a failure of the check.
            self.logger.info("Hallucination check cancelled before completion")
            raise

        except Exception:
            self.logger.exception("Error during hallucination check")
            raise

        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        self.logger.debug("Cleaned up temp file: %s", temp_file)
                except OSError as e:
                    self.logger.warning("Failed to cleanup temp file %s: %s", temp_file, e)

    @staticmethod
    def get_default_config(config_dir: str | None = None) -> dict[str, Any]:
        """
        Get the default configuration for hallucination processing.

        Args:
            config_dir: Path to the configuration directory. If None, defaults to
                       checks/config relative to the current working directory.

        Returns:
            Default configuration for hallucination processing loaded from cosmos_evaluator.yaml
        """
        try:
            config_manager = ConfigManager(config_dir)
            config = config_manager.load_config("cosmos_evaluator")
            return {"metropolis.hallucination": config["metropolis.hallucination"]}
        except Exception:
            logger.exception("Error loading default configuration")
            raise
