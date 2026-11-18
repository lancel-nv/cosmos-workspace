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

from contextlib import nullcontext
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np
import torch

from checks.physical_plausibility.attention import CONTRACT
from checks.physical_plausibility.features import (
    EMBEDDING_DIM,
    IMAGE_MEAN,
    IMAGE_STD,
    RESOLUTION,
    SPATIAL_TOKENS,
    TEMPORAL_TOKENS,
    letterbox_frames,
)
from checks.physical_plausibility.model import (
    CHECKPOINT_SHA256,
    MODEL_REVISION,
    NUM_FRAMES,
    SOURCE_ARCHIVE_ROOT,
    VjepaTokenEncoder,
    duration_spanning_indices,
    ensure_verified_encoder_source_archive,
    load_verified_encoder_model,
    sample_frames_uniform,
)


class ModelTest(unittest.TestCase):
    def test_token_contract_matches_encoder_settings(self) -> None:
        self.assertEqual(CONTRACT["checkpoint_sha256"], CHECKPOINT_SHA256)
        self.assertEqual(CONTRACT["source_commit"], MODEL_REVISION)
        self.assertEqual(CONTRACT["sampling"], f"uniform-duration-{NUM_FRAMES}-v1")
        self.assertEqual(CONTRACT["spatial"], f"letterbox{RESOLUTION}")
        self.assertEqual(CONTRACT["shape"], [TEMPORAL_TOKENS, SPATIAL_TOKENS, SPATIAL_TOKENS, EMBEDDING_DIM])

    def test_duration_sampler_has_unique_endpoints(self) -> None:
        indices = duration_spanning_indices(93)
        self.assertEqual(indices.shape, (64,))
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 92)
        self.assertTrue(np.all(np.diff(indices) > 0))

    def test_sampler_preserves_frame_indices_and_rgb_order(self) -> None:
        frames = [np.full((2, 3, 3), [i, i + 1, i + 2], dtype=np.uint8) for i in range(93)]
        with (
            patch("checks.physical_plausibility.model.get_video_frame_count", return_value=93),
            patch("checks.physical_plausibility.model.iter_video_frames", return_value=(f for f in frames)),
        ):
            actual = sample_frames_uniform("mock.mp4")
        expected = np.stack([frames[i][:, :, ::-1] for i in duration_spanning_indices(93)])
        np.testing.assert_array_equal(actual, expected)

    def test_short_videos_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 64 real frames"):
            duration_spanning_indices(63)

    def test_source_archive_is_verified_before_it_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cached = Path(directory) / "source.zip"
            downloaded = b"verified source archive"

            def download(_url: str, destination: str, **_kwargs: object) -> None:
                Path(destination).write_bytes(downloaded)

            with (
                patch("checks.physical_plausibility.model._source_archive_path", return_value=cached),
                patch(
                    "checks.physical_plausibility.model.SOURCE_ARCHIVE_SHA256",
                    hashlib.sha256(downloaded).hexdigest(),
                ),
                patch(
                    "checks.physical_plausibility.model.torch.hub.download_url_to_file", side_effect=download
                ) as fetch,
            ):
                self.assertEqual(ensure_verified_encoder_source_archive(), cached)
                self.assertEqual(ensure_verified_encoder_source_archive(), cached)
            fetch.assert_called_once()
            self.assertEqual(cached.read_bytes(), downloaded)

    def test_corrupt_source_archive_is_rejected_before_caching(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cached = Path(directory) / "source.zip"

            def download(_url: str, destination: str, **_kwargs: object) -> None:
                Path(destination).write_bytes(b"untrusted")

            with (
                patch("checks.physical_plausibility.model._source_archive_path", return_value=cached),
                patch("checks.physical_plausibility.model.SOURCE_ARCHIVE_SHA256", "0" * 64),
                patch("checks.physical_plausibility.model.torch.hub.download_url_to_file", side_effect=download),
                self.assertRaisesRegex(ValueError, "source archive hash mismatch"),
            ):
                ensure_verified_encoder_source_archive()
            self.assertFalse(cached.exists())

    def test_verified_source_is_loaded_locally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "source.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(f"{SOURCE_ARCHIVE_ROOT}/hubconf.py", "# verified")

            model = object()

            def local_load(repository: str, factory: str, **kwargs: object) -> object:
                self.assertTrue((Path(repository) / "hubconf.py").is_file())
                self.assertEqual(kwargs, {"source": "local", "pretrained": False})
                return model

            with (
                patch(
                    "checks.physical_plausibility.model.ensure_verified_encoder_source_archive",
                    return_value=archive_path,
                ),
                patch("checks.physical_plausibility.model.torch.hub.load", side_effect=local_load) as hub_load,
            ):
                self.assertIs(load_verified_encoder_model(), model)
            hub_load.assert_called_once()

    def test_source_archive_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "source.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(f"{SOURCE_ARCHIVE_ROOT}/hubconf.py", "# verified")
                archive.writestr(f"{SOURCE_ARCHIVE_ROOT}/../escaped.py", "# untrusted")
            with (
                patch(
                    "checks.physical_plausibility.model.ensure_verified_encoder_source_archive",
                    return_value=archive_path,
                ),
                patch("checks.physical_plausibility.model.torch.hub.load") as hub_load,
                self.assertRaisesRegex(ValueError, "unsafe path"),
            ):
                load_verified_encoder_model()
            hub_load.assert_not_called()

    def test_encoder_keeps_fractional_weights_and_reference_tensor_layout(self) -> None:
        frames = np.random.default_rng(19).integers(0, 256, (64, 15, 29, 3), dtype=np.uint8)
        canvas, weights = letterbox_frames(frames)
        expected = ((torch.from_numpy(canvas) - torch.from_numpy(IMAGE_MEAN)) / torch.from_numpy(IMAGE_STD)).permute(
            3, 0, 1, 2
        )[None]
        encoder = VjepaTokenEncoder.__new__(VjepaTokenEncoder)
        encoder.device = "cpu"
        encoder.model = MagicMock(return_value=torch.zeros(1, 18432, 1024))
        with (
            patch("checks.physical_plausibility.model.sample_frames_uniform", return_value=frames),
            patch("checks.physical_plausibility.model.torch.autocast", return_value=nullcontext()),
        ):
            tokens, actual_weights = encoder.encode("mock.mp4")
        actual = encoder.model.call_args.args[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.stride(), expected.stride())
        self.assertIs(encoder.model.call_args.kwargs["training"], False)
        repeated = np.broadcast_to(weights, (32, 24, 24)).reshape(-1)
        np.testing.assert_array_equal(actual_weights.numpy(), repeated[repeated > 0])
        self.assertTrue(((actual_weights > 0) & (actual_weights < 1)).any())
        self.assertEqual(tokens.dtype, torch.float16)
        self.assertFalse(tokens.requires_grad)


if __name__ == "__main__":
    unittest.main()
