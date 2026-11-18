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
ObjectsService implementation for processing objects analysis.

This service wraps the ObjectsProcessor and provides a standardized
service interface for deployment as a microservice.
"""

import asyncio
import contextlib
import logging
from pathlib import Path
import tempfile
import threading
from typing import Any

from checks.objects.single_view import api
from checks.utils.video import get_video_dimensions
from services.framework.service_base import ServiceBase
from services.objects.single_view.models import (
    ObjectsRequest,
    ObjectsResult,
)
from utils.cancellation import ProcessingCancelled


class ObjectsService(ServiceBase[ObjectsRequest, ObjectsResult]):
    """
    Service for processing objects analysis.

    This service processes video clips to analyze correspondence between
    segmented objects and tracked objects in the world model.
    """

    def __init__(self, seg_model: Any = None, seg_session: Any = None) -> None:
        """Initialize the service.

        Args:
            seg_model: Optional pre-loaded CWIP model. When provided (e.g. loaded once in the FastAPI lifespan), it
                is reused across every request instead of loading CWIP from disk per request.
            seg_session: Optional pre-built SegFormer inference session, loaded once in the same place and for the same
                reason. Sharing it also keeps concurrent requests from each initializing their own CUDA execution
                provider, which is what made two jobs starting together able to fail on ``cublasCreate``.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self._cwip_model = seg_model
        self._segformer_session = seg_session

    async def validate_input(self, request_data: ObjectsRequest) -> bool:
        """
        Validate input data for processing.

        Args:
            request_data: Request data to validate

        Returns:
            True if valid for processing

        Raises:
            ValueError: If validation fails
            GpuUnavailableDecodeError: If the videos exist but CUDA would not initialize for the probe. Deliberately
                not folded into the ValueError above: that one means the caller sent something unusable (422), whereas
                this means the input is fine and this replica is not (500).
        """
        # Validate required paths exist
        input_path = Path(request_data.input_data_path)
        if not input_path.is_dir():
            raise ValueError(f"Input data directory does not exist: {request_data.input_data_path}")

        video_path = Path(request_data.video_path)
        if not video_path.is_file():
            raise ValueError(f"Video file does not exist: {request_data.video_path}")

        await self._require_decodable("Video", video_path)

        # An absent world-model video is a supported camera-only request, not an error: the static check then runs on
        # the camera stream alone. A world video is validated only when one is provided (a two-stream run). The static
        # processor reads it the same way the augmented video is read, so probing it here turns an unreadable file
        # into an immediate rejection instead of a mid-run failure.
        if request_data.world_video_path is not None:
            world_video_path = Path(request_data.world_video_path)
            if not world_video_path.is_file():
                raise ValueError(f"World model video file does not exist: {request_data.world_video_path}")
            await self._require_decodable("World model video", world_video_path)

        # NOTE: Config is already validated as dict by Pydantic model validation

        # Validate trial_frames if provided
        if request_data.trial_frames is not None and request_data.trial_frames <= 0:
            raise ValueError("trial_frames must be positive if specified")

        return True

    async def _require_decodable(self, label: str, video_path: Path) -> None:
        """Reject a video that downloaded but cannot be read, before any model or RDS work starts.

        ``is_file()`` alone lets a zero-byte or truncated download through, and the failure then surfaces from inside
        the frame loop as an opaque decode error on a job that has already paid for the RDS zip and the model load.
        One ffprobe here turns that into an immediate, self-explanatory rejection.

        The probe is also what separates the two failure modes that previously looked identical: an unreadable file
        returns None and becomes a ``ValueError``, while a CUDA initialization failure raises
        ``GpuUnavailableDecodeError`` from underneath and is left to propagate.

        Args:
            label: Human-readable name for the input, used in the rejection message (e.g. ``"Video"``).
            video_path: Path to a file that already exists.

        Raises:
            ValueError: If ffprobe found no readable video stream in the file.
        """
        # Offloaded because the probe is a subprocess that may sleep between retries while it waits out a transient
        # GPU state, and this runs on the event loop.
        if await asyncio.to_thread(get_video_dimensions, str(video_path)) is None:
            raise ValueError(f"{label} file is not a readable video (ffprobe found no video stream): {video_path}")

    @staticmethod
    async def get_default_config() -> dict[str, Any]:
        """
        Get the default configuration for object processing.

        Returns:
            Default configuration for object processing
        """
        return api.get_default_config()

    async def process(self, request_data: ObjectsRequest, cancel_event: threading.Event | None = None) -> ObjectsResult:
        """
        Process the objects analysis.

        Args:
            request_data: Parsed and validated request data
            cancel_event: Optional cooperative-cancellation event. When the caller abandons this coroutine — a claimed
                job whose lease renewal came back with ``cancel_requested`` — the event is set so the executor thread
                running the processors stops promptly instead of continuing on a temp dir that is about to be deleted.
                ``None`` for the direct ``/process`` path, which cannot be cancelled mid-flight.

        Returns:
            Processing result
        """
        output_dir = self._setup_output_dir()
        self.logger.info("Starting object processing for clip %s", request_data.clip_id)
        self.logger.info("RDS HQ Data: %s", request_data.input_data_path)
        self.logger.info("Augmented Video: %s", request_data.video_path)
        self.logger.info("World Model Video: %s", request_data.world_video_path)
        self.logger.info("Output directory: %s", output_dir)

        try:
            # Run object processing using the unified entry point
            loop = asyncio.get_running_loop()

            # A non-empty request config is used as the FULL config for the run (we deliberately do NOT merge it over
            # the image defaults). An omitted (None) OR empty ({}) config is treated identically: we fall back to the
            # image defaults, since an empty dict almost always means "just use the defaults", not "run with no
            # configuration". This keeps a non-empty run self-describing with no hidden inherited defaults.
            config = request_data.config if request_data.config else api.get_default_config()

            fut = loop.run_in_executor(
                None,
                lambda: api.run_object_processors(
                    config=config,
                    input_data=request_data.input_data_path,
                    clip_id=request_data.clip_id,
                    camera_name=request_data.camera_name,
                    video_path=request_data.video_path,
                    output_dir=str(output_dir),
                    world_video_path=request_data.world_video_path,
                    model_device=request_data.model_device,
                    verbose=request_data.verbose.upper(),  # Convert to uppercase for consistency
                    trial_frames=request_data.trial_frames,
                    cancel_event=cancel_event,
                    static_seg_model=self._cwip_model,
                    dynamic_seg_session=self._segformer_session,
                ),
            )
            # Shield the executor future from task cancellation: cancelling this coroutine does NOT stop the running
            # thread, so on cancellation we instead flip the cooperative-cancellation flag and wait for the thread to
            # unwind. This guarantees the pipeline is no longer touching files by the time the caller deletes temp_dir.
            try:
                processing_results = await asyncio.shield(fut)
            except asyncio.CancelledError:
                if cancel_event is not None:
                    cancel_event.set()
                with contextlib.suppress(Exception):
                    await fut
                raise

            # Create result object
            result = ObjectsResult(
                processed_frames=processing_results.get("dynamic", {}).get("processed_frames", 0),
                total_video_frames=processing_results.get("dynamic", {}).get("total_video_frames", 0),
                mean_score=processing_results.get("dynamic", {}).get("mean_score", 0.0),
                std_score=processing_results.get("dynamic", {}).get("std_score", 0.0),
                min_score=processing_results.get("dynamic", {}).get("min_score", 0.0),
                max_score=processing_results.get("dynamic", {}).get("max_score", 0.0),
                unique_track_ids=sorted(processing_results.get("dynamic", {}).get("track_ids", set())),
                processed_frame_ids=processing_results.get("dynamic", {}).get("processed_frame_ids", []),
                config_summary=config,
                clip_id=request_data.clip_id,
                output_dir=str(output_dir),
            )

            self.logger.info("Processing completed successfully for clip %s", request_data.clip_id)
            self.logger.info("Processed %s frames with mean score %.3f", result.processed_frames, result.mean_score)

            return result

        except ProcessingCancelled:
            # Not a failure of this clip: the job was taken away (lease lost, or the arbitrator asked for a stop) and
            # somebody else owns the outcome now. Logged as the routine event it is so a load test's takeovers do not
            # read as a fleet full of errors.
            self.logger.info("Processing of clip %s was cancelled", request_data.clip_id)
            raise
        except Exception as e:
            self.logger.error("Error processing clip %s: %s", request_data.clip_id, e)
            raise

    def _setup_output_dir(self) -> Path:
        """Create a temporary working directory for processing outputs.

        Returns:
            Path to the temporary output directory
        """
        try:
            output_dir_path = Path(tempfile.mkdtemp(prefix="objects_"))
        except Exception as e:
            self.logger.error("Failed to create output directory: %s", e)
            raise

        self.logger.info("Using output directory: %s", output_dir_path)
        return output_dir_path
