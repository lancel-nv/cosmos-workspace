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

"""The fixed evaluation timeline the object check steps, and the strides that map it onto each input.

The object check draws on four independently-rated sources: RDS camera poses, RDS object annotations, the camera
video and the world-model video. Rather than negotiating a rate between them, the check pins its own timeline to
:data:`OBJECT_CHECK_FPS` (10 Hz) and strides every source down to it. 10 Hz is the rate RDS-HQ annotates objects at,
so it is the densest timeline on which every scored frame actually has ground truth behind it; the pass/fail
thresholds downstream of the check are calibrated against results produced at this rate.

Everything else follows from two facts:

- RDS frame indices — poses, ``all_object_info`` keys, per-frame static label keys — share one index namespace
  clocked at :data:`RDS_INDEX_FPS` (30 Hz). Poses populate every index; object annotations only every third one.
  So object-check step ``k`` reads RDS index ``3k``, which is always an annotated index.
- The videos carry their own rates, which need only be integer multiples of :data:`OBJECT_CHECK_FPS`. A 30 fps
  video contributes every 3rd frame; a 10 fps one every frame.

Running denser than 10 Hz does not buy evidence — it re-scores frames whose object geometry is absent (leaving them
unscoreable) while still paying for segmentation and hallucination detection on them. Running at exactly 10 Hz is
also what anything reading the results back assumes when it reconstructs which pose frame backs each scored frame.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# The rate the object check evaluates at, and the rate RDS-HQ annotates objects at. Not configurable: results at any
# other rate are not comparable against thresholds calibrated at this rate.
OBJECT_CHECK_FPS: int = 10

# The rate the shared RDS-HQ frame-index namespace is clocked at. Pose files exist at every index; object and static
# label files only every ``RDS_INDEX_FPS // OBJECT_CHECK_FPS``-th one.
RDS_INDEX_FPS: int = 30

# Frames of RDS index namespace per object-check step.
RDS_INDEX_STRIDE: int = RDS_INDEX_FPS // OBJECT_CHECK_FPS

# How far a source's frames-per-step ratio may sit from a whole number before it is rejected. Sized to absorb the
# rational frame rates a container may report (30000/1001 = 29.97 gives a ratio of 2.997) without admitting a rate
# that is genuinely not a multiple of OBJECT_CHECK_FPS (25 fps gives 2.5).
_STRIDE_TOLERANCE: float = 0.01


def stride_for_fps(source_fps: float, source_name: str) -> int:
    """Frames of ``source_name`` consumed per object-check step.

    Args:
        source_fps: The source's native frame rate.
        source_name: Human-readable source name, used only in the error message.

    Returns:
        The whole number of source frames per object-check step (1 for a 10 fps source, 3 for a 30 fps one).

    Raises:
        ValueError: If ``source_fps`` is not a positive integer multiple of :data:`OBJECT_CHECK_FPS`. Subsampling a
            source that is not a clean multiple would silently misalign it against the RDS annotation grid, so this
            is an error rather than a nearest-frame approximation.
    """
    if source_fps <= 0:
        raise ValueError(f"{source_name} frame rate must be positive, got {source_fps}")
    ratio = source_fps / OBJECT_CHECK_FPS
    stride = round(ratio)
    if stride < 1 or abs(ratio - stride) > _STRIDE_TOLERANCE:
        raise ValueError(
            f"{source_name} frame rate {source_fps} is not an integer multiple of the {OBJECT_CHECK_FPS} fps "
            f"object-check rate; cannot subsample it onto the object-annotation grid"
        )
    return stride


def _strided_count(frame_count: int, stride: int) -> int:
    """Number of strided samples available in ``frame_count`` frames, i.e. ``len(range(0, frame_count, stride))``."""
    if frame_count <= 0:
        return 0
    return -(-frame_count // stride)


@dataclass(frozen=True)
class ObjectCheckTimeline:
    """The object check's own frame axis, plus the per-source strides that project it onto each input.

    Step ``k`` runs from 0 to ``num_steps - 1`` and denotes one instant ``k / OBJECT_CHECK_FPS`` seconds into the
    clip. :meth:`rds_index`, :meth:`camera_frame` and :meth:`world_frame` translate it into each source's own index
    space; nothing downstream should re-derive those mappings from raw frame rates.

    Attributes:
        camera_video_fps: Native rate of the camera video.
        camera_stride: Camera video frames per step.
        world_video_fps: Native rate of the world-model video, or ``None`` for checks that do not read one (the
            dynamic-object check scores the camera video alone).
        world_stride: World video frames per step, or ``None`` when ``world_video_fps`` is.
        num_steps: Number of steps every contributing source can supply, after any trial-mode truncation.
    """

    camera_video_fps: float
    camera_stride: int
    world_video_fps: float | None
    world_stride: int | None
    num_steps: int

    def rds_index(self, step: int) -> int:
        """The RDS frame index — pose, object and static-label namespace — backing ``step``."""
        return step * RDS_INDEX_STRIDE

    def camera_frame(self, step: int) -> int:
        """The camera-video frame index backing ``step``."""
        return step * self.camera_stride

    def world_frame(self, step: int) -> int:
        """The world-video frame index backing ``step``.

        Raises:
            ValueError: If this timeline was built without a world video.
        """
        if self.world_stride is None:
            raise ValueError("This timeline has no world video; world_frame() is not available")
        return step * self.world_stride

    def rds_indices(self) -> list[int]:
        """The RDS frame index for every step, in order."""
        return [self.rds_index(step) for step in range(self.num_steps)]

    def camera_frames(self) -> list[int]:
        """The camera-video frame index for every step, in order."""
        return [self.camera_frame(step) for step in range(self.num_steps)]

    def world_frames(self) -> list[int]:
        """The world-video frame index for every step, in order.

        Raises:
            ValueError: If this timeline was built without a world video.
        """
        return [self.world_frame(step) for step in range(self.num_steps)]

    @classmethod
    def build(
        cls,
        *,
        camera_video_fps: float,
        camera_frame_count: int,
        world_video_fps: float | None = None,
        world_frame_count: int | None = None,
        pose_frame_count: int | None = None,
        annotated_frame_count: int | None = None,
        trial_steps: int | None = None,
        log: logging.Logger | None = None,
    ) -> "ObjectCheckTimeline":
        """Build the timeline for one clip, striding every supplied source down to :data:`OBJECT_CHECK_FPS`.

        ``num_steps`` is the largest step count every supplied source can satisfy, so the check never walks off the
        end of a video, the pose stream or the annotation axis. Sources left as ``None`` simply do not constrain it.

        Args:
            camera_video_fps: Native rate of the camera video.
            camera_frame_count: Total frames in the camera video.
            world_video_fps: Native rate of the world-model video, or ``None`` if the check reads none.
            world_frame_count: Total frames in the world-model video; required when ``world_video_fps`` is given.
            pose_frame_count: Number of RDS pose frames available, in the 30 Hz index namespace.
            annotated_frame_count: Number of RDS object-annotation frames available, already at
                :data:`OBJECT_CHECK_FPS`.
            trial_steps: Trial-mode cap, in object-check steps rather than source frames.
            log: Logger for the rate summary; defaults to this module's logger.

        Returns:
            The timeline for this clip.

        Raises:
            ValueError: If a source rate is not an integer multiple of :data:`OBJECT_CHECK_FPS`, or if
                ``world_video_fps`` was given without ``world_frame_count``.
        """
        active_log = logger if log is None else log

        camera_stride = stride_for_fps(camera_video_fps, "Camera video")
        limits = {"camera video": _strided_count(camera_frame_count, camera_stride)}

        world_stride: int | None = None
        if world_video_fps is not None:
            if world_frame_count is None:
                raise ValueError("world_frame_count is required when world_video_fps is given")
            world_stride = stride_for_fps(world_video_fps, "World video")
            limits["world video"] = _strided_count(world_frame_count, world_stride)
        if pose_frame_count is not None:
            limits["RDS poses"] = _strided_count(pose_frame_count, RDS_INDEX_STRIDE)
        if annotated_frame_count is not None:
            limits["RDS object annotations"] = max(annotated_frame_count, 0)

        num_steps = min(limits.values())
        binding = sorted(name for name, limit in limits.items() if limit == num_steps)

        if trial_steps is not None and trial_steps < num_steps:
            active_log.info("Trial mode: evaluating %d of %d available steps", trial_steps, num_steps)
            num_steps = max(trial_steps, 0)

        cls._log_rates(active_log, camera_video_fps, camera_stride, world_video_fps, world_stride, num_steps, binding)
        return cls(
            camera_video_fps=camera_video_fps,
            camera_stride=camera_stride,
            world_video_fps=world_video_fps,
            world_stride=world_stride,
            num_steps=num_steps,
        )

    @staticmethod
    def _log_rates(
        log: logging.Logger,
        camera_video_fps: float,
        camera_stride: int,
        world_video_fps: float | None,
        world_stride: int | None,
        num_steps: int,
        binding: Sequence[str],
    ) -> None:
        """Log the rate of every source, the stride applied to it, and which source bounds the step count."""
        parts = [f"camera video {camera_video_fps:g} fps (every {camera_stride}), "]
        if world_video_fps is not None and world_stride is not None:
            parts.append(f"world video {world_video_fps:g} fps (every {world_stride}), ")
        parts.append(f"RDS indices {RDS_INDEX_FPS} fps (every {RDS_INDEX_STRIDE})")
        log.info(
            "Object check evaluates at %d fps: %s; %d steps, bounded by %s",
            OBJECT_CHECK_FPS,
            "".join(parts),
            num_steps,
            " and ".join(binding),
        )
        subsampled = [f"camera video {camera_video_fps:g}"] if camera_stride > 1 else []
        if world_stride is not None and world_stride > 1 and world_video_fps is not None:
            subsampled.append(f"world video {world_video_fps:g}")
        if RDS_INDEX_STRIDE > 1:
            subsampled.append(f"RDS poses {RDS_INDEX_FPS}")
        if subsampled:
            log.info(
                "Subsampling to the %d fps object-annotation rate from: %s fps. Denser inputs carry no extra object "
                "ground truth — RDS-HQ annotates objects at %d fps — and results at another rate would not match "
                "the calibrated thresholds",
                OBJECT_CHECK_FPS,
                ", ".join(subsampled),
                OBJECT_CHECK_FPS,
            )


def validate_annotation_axis(axis: Sequence[int], num_steps: int, log: logging.Logger | None = None) -> None:
    """Check that the RDS annotation axis covers every index the object check is about to read.

    The check assumes object annotations sit on the ``RDS_INDEX_STRIDE`` grid. A dataset whose annotations land
    elsewhere would leave frames silently unscoreable rather than failing, so the mismatch is raised here instead.

    Args:
        axis: Sorted RDS frame indices carrying object annotations, e.g.
            ``RdsDataLoader.get_canonical_static_frame_axis()``.
        num_steps: Number of object-check steps about to be walked.
        log: Logger for the denser-than-needed notice; defaults to this module's logger.

    Raises:
        ValueError: If any index the check will read is absent from ``axis``.
    """
    active_log = logger if log is None else log
    available = set(axis)
    required = [step * RDS_INDEX_STRIDE for step in range(num_steps)]
    missing = [index for index in required if index not in available]
    if missing:
        raise ValueError(
            f"RDS object annotations are missing {len(missing)} of the {len(required)} frame indices the object "
            f"check reads at {OBJECT_CHECK_FPS} fps (first missing: {missing[:5]}); the dataset's annotation grid "
            f"does not match the expected every-{RDS_INDEX_STRIDE} spacing"
        )
    extra = len(available) - len(set(required))
    if extra > 0:
        active_log.info(
            "RDS carries %d annotated frames beyond the %d the object check reads; the surplus is not evaluated "
            "because %d fps is the rate the check's thresholds are calibrated for",
            extra,
            len(required),
            OBJECT_CHECK_FPS,
        )
