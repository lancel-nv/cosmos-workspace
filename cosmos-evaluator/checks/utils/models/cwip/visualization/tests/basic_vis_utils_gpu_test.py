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

import os
import tempfile
import unittest

import numpy as np

from checks.utils.models.cwip.visualization.basic_vis_utils import (
    TilingOutputMode,
    tile_and_save_media,
)


class TestTileAndSaveMediaVideoModeGpu(unittest.TestCase):
    """Test tile_and_save_media VIDEO mode with real NVENC GPU encoding."""

    def test_video_mode_nvenc(self) -> None:
        """Test VIDEO mode encodes with NVENC hardware encoder."""
        num_frames = 2
        H, W = 256, 256
        camera_images = (np.random.rand(num_frames, H, W, 3) * 255).astype(np.uint8)
        world_images = (np.random.rand(num_frames, H, W, 3) * 255).astype(np.uint8)
        camera_spatial = (np.random.rand(num_frames, H, W, 3) * 255).astype(np.uint8)
        world_spatial = (np.random.rand(num_frames, H, W, 3) * 255).astype(np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "video.mp4")
            result = tile_and_save_media(
                camera_images=camera_images,
                world_images=world_images,
                camera_spatial_features=camera_spatial,
                world_spatial_features=world_spatial,
                output_mode=TilingOutputMode.VIDEO,
                output_path=output_path,
                spatial_cosine_sim=None,
                use_gpu_encoding=True,
            )
            self.assertIsNone(result)
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)


if __name__ == "__main__":
    unittest.main()
