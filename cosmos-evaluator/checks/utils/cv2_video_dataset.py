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

"""Single camera mp4 dataset that uses open cv backend."""

from collections.abc import Callable, Generator, Iterator
import logging
import os

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset

from checks.utils.video import get_video_dimensions, get_video_fps, get_video_frame_count, iter_video_frames

logger = logging.getLogger(__name__)


class VideoDataset(IterableDataset):
    """VideoDataset class."""

    def __init__(
        self,
        video_path: str,
        transforms_fn: list[Callable[[torch.Tensor], torch.Tensor]] | None = None,
        skip: int = 1,
        start_frame: int | None = None,
    ) -> None:
        """Constructor.

        Args:
            video_path: Path to the input mp4 video. The filename should be the camera name where it was recorded.
                Example filename: `camera_front_wide_120fov.mp4`
            transforms_fn: a function pipeline that is called to transform the frame before yielding it to downstream
                task
            skip: Value for skipping frames in an video
            start_frame: start yielding frames from this position (1-based index)
        Raises:
            FileNotFoundError: If the `video_path` does not exist.
            ValueError: If the `video_path` does not have `.mp4` extension.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"{video_path} does not exist!")

        _, filename = os.path.split(video_path)
        _, ext = os.path.splitext(filename)
        if ext != ".mp4":
            raise ValueError(f"{video_path} is not a mp4 file!")
        self._video_path = video_path

        # Video metadata via the vendored LGPL ffprobe (cv2 here is built without the ffmpeg backend).
        self.number_frames = get_video_frame_count(video_path) or 0
        dims = get_video_dimensions(video_path)
        self.image_width, self.image_height = dims if dims is not None else (0, 0)
        fps = get_video_fps(video_path)
        self.frames_per_second = int(fps) if fps is not None else 0

        logger.info("number_frames: %s, frames_per_second: %s", self.number_frames, self.frames_per_second)
        self._transforms_fn = [] if transforms_fn is None else transforms_fn
        assert isinstance(self._transforms_fn, list)
        if skip < 1:
            raise ValueError(f"`skip` must be a positive integer (got {skip})")
        self._skip = skip
        self.start_frame = 0
        if start_frame is not None:
            assert start_frame >= 0
            self.start_frame = start_frame
        # Active decode generator for the most recent __iter__, so release() can promptly stop ffmpeg.
        self._frames: Generator[np.ndarray] | None = None

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Load and process a sample.

        Returns:
            Iterator object that yields processed image.
        """
        # Decode from start_frame onward; skip is applied on the absolute frame number to preserve prior semantics.
        self._frames = iter_video_frames(self._video_path, start_frame=self.start_frame)
        try:
            for offset, image in enumerate(self._frames):
                frame_number = self.start_frame + offset
                if frame_number % self._skip != 0:
                    continue
                # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # convert from H,W,C to C,H,W
                image = np.transpose(image, (2, 0, 1))
                # convert to float32
                image = image.astype(np.float32)

                image_tensor = torch.from_numpy(image)
                for fn in self._transforms_fn:
                    image_tensor = fn(image_tensor)
                yield image_tensor
        finally:
            self._frames.close()
            self._frames = None

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return (self.number_frames - self.start_frame) // self._skip

    def release(self) -> None:
        """Stop the active decode generator (terminating its ffmpeg child), if any."""
        if self._frames is not None:
            self._frames.close()
            self._frames = None
