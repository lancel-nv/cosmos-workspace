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

from collections.abc import Iterator
import contextlib
from dataclasses import dataclass
import os
import subprocess
import tempfile
import tracemalloc
from typing import Self
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from checks.utils.models.cwip.visualization.image_video_utils import (
    WorldCameraVideoPairSampleMethod,
    _decode_video_frames_to_tensor,
    _write_video_with_ffmpeg_internal,
    get_pe_image_transform,
    preprocess_world_camera_video_pair,
    tiled_frames_generator,
    to_uint8,
    write_video_with_ffmpeg,
)
from checks.utils.video import NVENC_H264_FFMPEG_ARGS, SVTAV1_FFMPEG_ARGS, VideoDecodeError

_MODULE = "checks.utils.models.cwip.visualization.image_video_utils"


@dataclass(frozen=True)
class _StreamSpec:
    """Synthetic properties of one fake video stream."""

    num_frames: int
    fps: float = 30.0
    height: int = 64
    width: int = 64


def _expected_bgr_frame(spec: _StreamSpec, frame_idx: int) -> np.ndarray:
    """Build the BGR frame :class:`_FakeFrameReader` serves for ``frame_idx``.

    Each channel carries a different value derived from the index, so a test can tell apart both the frame that
    was decoded and the channel order it arrived in.
    """
    frame = np.empty((spec.height, spec.width, 3), dtype=np.uint8)
    for channel in range(3):
        frame[:, :, channel] = (frame_idx + channel) % 256
    return frame


class _FakeFrameReader:
    """In-memory stand-in for :class:`~checks.utils.video.SequentialFrameReader`.

    Records the indices it was asked for so tests can assert the sampling math, and returns ``None`` past the end
    of the stream exactly as the real reader does.
    """

    def __init__(self, spec: _StreamSpec) -> None:
        self._spec = spec
        self.requested_indices: list[int] = []
        self.closed = False

    def read(self, frame_idx: int) -> np.ndarray | None:
        self.requested_indices.append(frame_idx)
        if frame_idx < 0 or frame_idx >= self._spec.num_frames:
            return None
        return _expected_bgr_frame(self._spec, frame_idx)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@contextlib.contextmanager
def _fake_video_io(streams: dict[str, _StreamSpec]) -> Iterator[dict[str, list[_FakeFrameReader]]]:
    """Patch the module's ffmpeg/ffprobe entry points with in-memory fakes, keyed by video path.

    Yields a mapping of video path to every reader opened for it, so tests can assert which frames were pulled
    and how many decode passes each stream cost.
    """
    readers: dict[str, list[_FakeFrameReader]] = {path: [] for path in streams}

    def open_reader(video_path: str, *_args: object, **_kwargs: object) -> _FakeFrameReader:
        reader = _FakeFrameReader(streams[video_path])
        readers[video_path].append(reader)
        return reader

    with (
        patch(f"{_MODULE}.os.path.exists", return_value=True),
        patch(f"{_MODULE}.get_video_fps", side_effect=lambda path: streams[path].fps),
        patch(f"{_MODULE}.get_video_frame_count", side_effect=lambda path: streams[path].num_frames),
        patch(f"{_MODULE}.SequentialFrameReader", side_effect=open_reader),
    ):
        yield readers


_WORLD = "/fake/world.mp4"
_CAMERA = "/fake/camera.mp4"


def _pair(world: _StreamSpec, camera: _StreamSpec) -> dict[str, _StreamSpec]:
    """Shorthand for the world/camera stream mapping used by most tests."""
    return {_WORLD: world, _CAMERA: camera}


class TestToUint8(unittest.TestCase):
    """Tests for to_uint8 function."""

    def test_already_uint8(self) -> None:
        """Test that uint8 arrays pass through unchanged."""
        x = np.array([0, 128, 255], dtype=np.uint8)
        result = to_uint8(x)
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result, x)

    def test_from_float_normalized(self) -> None:
        """Test conversion from float [0, 1] to uint8."""
        x = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = to_uint8(x)
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result, np.array([0, 127, 255], dtype=np.uint8))

    def test_from_float_255_range(self) -> None:
        """Test conversion from float [0, 255] to uint8."""
        x = np.array([0.0, 127.5, 255.0], dtype=np.float32)
        result = to_uint8(x)
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result, np.array([0, 127, 255], dtype=np.uint8))

    def test_clips_negative_values(self) -> None:
        """Test that negative values are clipped to 0."""
        x = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
        result = to_uint8(x)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result[0], 0)

    def test_clips_values_above_one(self) -> None:
        """Test that values slightly above 1 are clipped."""
        x = np.array([0.5, 1.2, 1.4], dtype=np.float32)
        result = to_uint8(x)
        self.assertEqual(result.dtype, np.uint8)
        # Values > 1 but < 1.5 are clipped to 1, then scaled to 255
        self.assertEqual(result[1], 255)
        self.assertEqual(result[2], 255)

    def test_3d_array(self) -> None:
        """Test conversion of 3D array (like image)."""
        x = np.random.rand(64, 64, 3).astype(np.float32)
        result = to_uint8(x)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (64, 64, 3))

    def test_4d_array(self) -> None:
        """Test conversion of 4D array (like video frames)."""
        x = np.random.rand(10, 64, 64, 3).astype(np.float32)
        result = to_uint8(x)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(result.shape, (10, 64, 64, 3))


def _to_uint8_reference(x: np.ndarray) -> np.ndarray:
    """Whole-array form of the conversion, used as the oracle for the chunked implementation."""
    if x.dtype == np.uint8:
        return x
    original_max = x.max()
    clipped = np.clip(x, 0, 1) if original_max <= 1.5 else np.clip(x, 0, 255)
    return (clipped * 255).astype(np.uint8) if original_max <= 1.5 else clipped.astype(np.uint8)


def _to_uint8_peak_ratio(x: np.ndarray) -> float:
    """Peak bytes allocated during conversion, as a multiple of the input size."""
    tracemalloc.start()
    try:
        to_uint8(x)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return peak / x.nbytes


class TestToUint8Chunked(unittest.TestCase):
    """The chunked conversion matches the whole-array result without holding full-size float temporaries."""

    def test_values_match_the_whole_array_conversion(self) -> None:
        cases: dict[str, np.ndarray] = {
            "normalized_4d": np.linspace(0.0, 1.0, 8 * 6 * 5 * 3, dtype=np.float32).reshape(8, 6, 5, 3),
            "normalized_clipped": np.linspace(-0.4, 1.4, 4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3),
            "zero_255": np.linspace(0.0, 255.0, 4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3),
            "out_of_range_255": np.linspace(-20.0, 300.0, 4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3),
            "float64": np.linspace(0.0, 1.0, 12, dtype=np.float64).reshape(4, 3),
            "zeros": np.zeros((3, 2, 2), dtype=np.float32),
            "ones": np.ones((1, 1, 1), dtype=np.float32),
            "bool_mask": np.array([[True, False], [False, True]]),
            "all_false_mask": np.zeros((2, 2), dtype=bool),
            "int32_0_255": (np.arange(12, dtype=np.int32) * 21).reshape(4, 3),
        }
        for name, x in cases.items():
            with self.subTest(name):
                np.testing.assert_array_equal(to_uint8(x), _to_uint8_reference(x))

    def test_narrow_integer_input_scales_without_overflow(self) -> None:
        result = to_uint8(np.array([[0, 1], [1, 0]], dtype=np.int8))

        np.testing.assert_array_equal(result, np.array([[0, 255], [255, 0]], dtype=np.uint8))

    def test_uint8_input_is_returned_as_is(self) -> None:
        x = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)

        self.assertIs(to_uint8(x), x)

    def test_input_is_not_modified(self) -> None:
        x = np.linspace(-0.5, 1.5, 24, dtype=np.float32).reshape(2, 4, 3)
        before = x.copy()

        to_uint8(x)

        np.testing.assert_array_equal(x, before)

    def test_zero_dimensional_input_converts(self) -> None:
        self.assertEqual(to_uint8(np.array(0.5, dtype=np.float32)), np.uint8(127))

    def test_row_larger_than_the_chunk_budget_converts(self) -> None:
        """A single row above the 8 MB budget still gets a chunk of one row."""
        x = np.full((2, 1200, 1200, 3), 0.5, dtype=np.float32)

        result = to_uint8(x)

        self.assertEqual(result.shape, x.shape)
        self.assertTrue(np.all(result == 127))

    def test_no_full_size_float_temporary_is_held(self) -> None:
        """The whole-array form peaks at 2.25x a float32 input; the chunked form holds the uint8 result and scratch."""
        x = np.full((64, 720, 640, 3), 0.5, dtype=np.float32)

        self.assertLess(_to_uint8_peak_ratio(x), 0.5)

    def test_peak_does_not_grow_with_the_input(self) -> None:
        small = np.full((16, 720, 640, 3), 0.5, dtype=np.float32)
        large = np.full((64, 720, 640, 3), 0.5, dtype=np.float32)

        self.assertLessEqual(_to_uint8_peak_ratio(large), _to_uint8_peak_ratio(small) + 0.05)


class TestTiledFramesGenerator(unittest.TestCase):
    """Tests for tiled_frames_generator function."""

    def test_2x2(self) -> None:
        """Test tiled frames generator with 2x2 grid."""
        num_frames = 3
        tile_h, tile_w = 4, 6

        # Create 4 tiles for 2x2 grid
        tiles = [
            np.random.randint(0, 255, (num_frames, 8, 12, 3), dtype=np.uint8),
            np.random.randint(0, 255, (num_frames, 8, 12, 3), dtype=np.uint8),
            np.random.randint(0, 255, (num_frames, 8, 12, 3), dtype=np.uint8),
            np.random.randint(0, 255, (num_frames, 8, 12, 3), dtype=np.uint8),
        ]

        generator = tiled_frames_generator(
            tiles=tiles,
            num_rows=2,
            num_cols=2,
            num_frames=num_frames,
            tile_height=tile_h,
            tile_width=tile_w,
            use_gpu=False,  # Use CPU for testing
        )

        frames = list(generator)
        self.assertEqual(len(frames), num_frames)
        for frame in frames:
            self.assertEqual(frame.shape, (tile_h * 2, tile_w * 2, 3))
            self.assertEqual(frame.dtype, np.uint8)

    def test_with_none_tiles(self) -> None:
        """Test tiled frames generator with None tiles (black placeholders)."""
        num_frames = 2
        tile_h, tile_w = 4, 4

        # Create tiles with some None values
        tiles = [
            np.random.randint(0, 255, (num_frames, 8, 8, 3), dtype=np.uint8),
            None,  # This will be replaced with black tile
            None,
            np.random.randint(0, 255, (num_frames, 8, 8, 3), dtype=np.uint8),
        ]

        generator = tiled_frames_generator(
            tiles=tiles,
            num_rows=2,
            num_cols=2,
            num_frames=num_frames,
            tile_height=tile_h,
            tile_width=tile_w,
            use_gpu=False,
        )

        frames = list(generator)
        self.assertEqual(len(frames), num_frames)
        for frame in frames:
            self.assertEqual(frame.shape, (tile_h * 2, tile_w * 2, 3))
            self.assertEqual(frame.dtype, np.uint8)


class TestGetPeImageTransform(unittest.TestCase):
    """Tests for get_pe_image_transform function."""

    def test_with_int_size(self) -> None:
        """Test get_pe_image_transform with integer size."""
        transform = get_pe_image_transform(image_size=224, normalize=True)

        # Test on a sample tensor
        img = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
        result = transform(img)

        # Should be resized to 224x224 and normalized
        self.assertEqual(result.shape, (3, 224, 224))
        self.assertEqual(result.dtype, torch.float32)

    def test_with_tuple_size(self) -> None:
        """Test get_pe_image_transform with tuple size."""
        transform = get_pe_image_transform(image_size=(256, 384), normalize=False)

        # Test on a sample tensor
        img = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
        result = transform(img)

        # Should be resized to 256x384 and not normalized
        self.assertEqual(result.shape, (3, 256, 384))
        self.assertEqual(result.dtype, torch.float32)
        # Without normalization, values should be in [0, 1]
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)


class TestWorldCameraVideoPairSampleMethod(unittest.TestCase):
    """Tests for WorldCameraVideoPairSampleMethod enum."""

    def test_enum_values(self) -> None:
        """Test that WorldCameraVideoPairSampleMethod enum values are correct."""
        self.assertEqual(WorldCameraVideoPairSampleMethod.UNIFORM, "uniform")
        self.assertEqual(WorldCameraVideoPairSampleMethod.RANDOM, "random")
        self.assertEqual(WorldCameraVideoPairSampleMethod.FIRST, "first")
        self.assertEqual(WorldCameraVideoPairSampleMethod.LAST, "last")
        self.assertEqual(WorldCameraVideoPairSampleMethod.MIDDLE, "middle")
        self.assertEqual(WorldCameraVideoPairSampleMethod.MIXED, "mixed")


class TestTiledFramesGeneratorEdgeCases(unittest.TestCase):
    """Additional tests for tiled_frames_generator function."""

    def test_3x2_grid(self) -> None:
        """Test tiled frames generator with 3x2 grid."""
        num_frames = 2
        tile_h, tile_w = 8, 8

        # Create 6 tiles for 3x2 grid
        tiles = [np.random.randint(0, 255, (num_frames, 16, 16, 3), dtype=np.uint8) for _ in range(6)]

        generator = tiled_frames_generator(
            tiles=tiles,
            num_rows=3,
            num_cols=2,
            num_frames=num_frames,
            tile_height=tile_h,
            tile_width=tile_w,
            use_gpu=False,
        )

        frames = list(generator)
        self.assertEqual(len(frames), num_frames)
        for frame in frames:
            self.assertEqual(frame.shape, (tile_h * 3, tile_w * 2, 3))

    def test_wrong_tile_count_raises(self) -> None:
        """Test that wrong number of tiles raises ValueError."""
        num_frames = 2
        tile_h, tile_w = 8, 8

        # Create 3 tiles but specify 2x2 grid (should need 4)
        tiles = [np.random.randint(0, 255, (num_frames, 16, 16, 3), dtype=np.uint8) for _ in range(3)]

        with self.assertRaises(ValueError):
            list(
                tiled_frames_generator(
                    tiles=tiles,
                    num_rows=2,
                    num_cols=2,
                    num_frames=num_frames,
                    tile_height=tile_h,
                    tile_width=tile_w,
                    use_gpu=False,
                )
            )

    def test_1x1_grid(self) -> None:
        """Test tiled frames generator with 1x1 grid (single tile)."""
        num_frames = 3
        tile_h, tile_w = 32, 32

        tiles = [np.random.randint(0, 255, (num_frames, 64, 64, 3), dtype=np.uint8)]

        generator = tiled_frames_generator(
            tiles=tiles,
            num_rows=1,
            num_cols=1,
            num_frames=num_frames,
            tile_height=tile_h,
            tile_width=tile_w,
            use_gpu=False,
        )

        frames = list(generator)
        self.assertEqual(len(frames), num_frames)
        for frame in frames:
            self.assertEqual(frame.shape, (tile_h, tile_w, 3))


class TestGetPeImageTransformEdgeCases(unittest.TestCase):
    """Additional tests for get_pe_image_transform function."""

    def test_with_list_size(self) -> None:
        """Test get_pe_image_transform with list size."""
        transform = get_pe_image_transform(image_size=(128, 256), normalize=True)

        img = torch.randint(0, 255, (3, 512, 512), dtype=torch.uint8)
        result = transform(img)

        self.assertEqual(result.shape, (3, 128, 256))

    def test_batch_of_images(self) -> None:
        """Test transform on batch of images."""
        transform = get_pe_image_transform(image_size=64, normalize=True)

        # Batch of 4 images
        imgs = torch.randint(0, 255, (4, 3, 128, 128), dtype=torch.uint8)
        result = transform(imgs)

        self.assertEqual(result.shape, (4, 3, 64, 64))

    def test_normalization_range(self) -> None:
        """Test that normalization produces expected range."""
        transform = get_pe_image_transform(image_size=32, normalize=True)

        img = torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8)
        result = transform(img)

        # After ImageNet normalization, values can be negative
        # Just check it's float and transformed
        self.assertEqual(result.dtype, torch.float32)


class TestWriteVideoWithFfmpeg(unittest.TestCase):
    """Tests for write_video_with_ffmpeg function."""

    @patch("checks.utils.models.cwip.visualization.image_video_utils._write_video_with_ffmpeg_internal")
    @patch(
        "checks.utils.models.cwip.visualization.image_video_utils.get_video_encode_args",
        return_value=list(NVENC_H264_FFMPEG_ARGS),
    )
    def test_uses_hardware_encoder_when_available(self, _mock_args: MagicMock, mock_internal: MagicMock) -> None:
        """The writer passes whatever the shared ladder selected straight through."""
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        write_video_with_ffmpeg(frames, "/tmp/test.mp4", 64, 64, 30)

        mock_internal.assert_called_once()
        _, kwargs = mock_internal.call_args
        self.assertEqual(kwargs.get("encode_args"), list(NVENC_H264_FFMPEG_ARGS))

    @patch("checks.utils.models.cwip.visualization.image_video_utils._write_video_with_ffmpeg_internal")
    @patch(
        "checks.utils.models.cwip.visualization.image_video_utils.get_video_encode_args",
        return_value=list(SVTAV1_FFMPEG_ARGS),
    )
    def test_uses_svtav1_when_ladder_selects_it(self, _mock_args: MagicMock, mock_internal: MagicMock) -> None:
        """When the ladder has already bottomed out at SVT-AV1 the writer encodes once, with no retry."""
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        write_video_with_ffmpeg(frames, "/tmp/test.mp4", 64, 64, 30)

        mock_internal.assert_called_once()
        _, kwargs = mock_internal.call_args
        self.assertEqual(kwargs.get("encode_args"), list(SVTAV1_FFMPEG_ARGS))

    @patch("checks.utils.models.cwip.visualization.image_video_utils._write_video_with_ffmpeg_internal")
    @patch(
        "checks.utils.models.cwip.visualization.image_video_utils.get_video_encode_args",
        return_value=list(NVENC_H264_FFMPEG_ARGS),
    )
    def test_falls_back_to_cpu_on_hardware_error(self, _mock_args: MagicMock, mock_internal: MagicMock) -> None:
        """A tier that passes the probe but fails at encode time still falls back to SVT-AV1."""
        mock_internal.side_effect = [RuntimeError("NVENC failed"), None]
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        write_video_with_ffmpeg(frames, "/tmp/test.mp4", 64, 64, 30)

        self.assertEqual(mock_internal.call_count, 2)
        _, kwargs1 = mock_internal.call_args_list[0]
        self.assertEqual(kwargs1.get("encode_args"), list(NVENC_H264_FFMPEG_ARGS))
        _, kwargs2 = mock_internal.call_args_list[1]
        self.assertEqual(kwargs2.get("encode_args"), list(SVTAV1_FFMPEG_ARGS))

    @patch("checks.utils.models.cwip.visualization.image_video_utils._write_video_with_ffmpeg_internal")
    @patch(
        "checks.utils.models.cwip.visualization.image_video_utils.get_video_encode_args",
        return_value=list(NVENC_H264_FFMPEG_ARGS),
    )
    def test_oversized_frames_capped(self, _mock_args: MagicMock, mock_internal: MagicMock) -> None:
        """>4096px frames are downscaled to <=4096 per dim, keeping the encode within the hardware encoder's reach."""
        frames = [np.zeros((16, 16, 3), dtype=np.uint8)]
        write_video_with_ffmpeg(frames, "/tmp/big.mp4", 5000, 3000, 30)

        mock_internal.assert_called_once()
        _, kwargs = mock_internal.call_args
        self.assertEqual(kwargs.get("encode_args"), list(NVENC_H264_FFMPEG_ARGS))
        # Source dims preserved for the raw stdin read; encoded dims scaled to fit 4096, even, aspect-preserving.
        self.assertEqual(kwargs.get("enc_width"), 4096)
        self.assertEqual(kwargs.get("enc_height"), 2456)
        self.assertLessEqual(max(kwargs["enc_width"], kwargs["enc_height"]), 4096)

    @patch("checks.utils.models.cwip.visualization.image_video_utils._write_video_with_ffmpeg_internal")
    @patch(
        "checks.utils.models.cwip.visualization.image_video_utils.get_video_encode_args",
        return_value=list(SVTAV1_FFMPEG_ARGS),
    )
    def test_oversized_frames_without_nvenc_use_svtav1(self, _mock_args: MagicMock, mock_internal: MagicMock) -> None:
        """The 4096 cap is applied independently of which tier the ladder picked."""
        frames = [np.zeros((16, 16, 3), dtype=np.uint8)]
        write_video_with_ffmpeg(frames, "/tmp/big.mp4", 5000, 3000, 30)

        mock_internal.assert_called_once()
        _, kwargs = mock_internal.call_args
        self.assertEqual(kwargs.get("encode_args"), list(SVTAV1_FFMPEG_ARGS))
        self.assertEqual(kwargs.get("enc_width"), 4096)
        self.assertEqual(kwargs.get("enc_height"), 2456)

    @patch("checks.utils.models.cwip.visualization.image_video_utils._write_video_with_ffmpeg_internal")
    @patch(
        "checks.utils.models.cwip.visualization.image_video_utils.get_video_encode_args",
        return_value=list(NVENC_H264_FFMPEG_ARGS),
    )
    def test_converts_iterator_to_list(self, _mock_args: MagicMock, mock_internal: MagicMock) -> None:
        """Test that iterator frames are converted to list."""

        def frame_gen() -> Iterator[np.ndarray]:
            for _ in range(3):
                yield np.zeros((64, 64, 3), dtype=np.uint8)

        write_video_with_ffmpeg(frame_gen(), "/tmp/test.mp4", 64, 64, 30)
        mock_internal.assert_called_once()


class TestWriteVideoWithFfmpegInternal(unittest.TestCase):
    """Tests for _write_video_with_ffmpeg_internal function."""

    @patch("checks.utils.models.cwip.visualization.image_video_utils.subprocess.Popen")
    def test_successful_encoding(self, mock_popen: MagicMock) -> None:
        """Test successful encoding."""
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.returncode = 0
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
        _write_video_with_ffmpeg_internal(frames, "/tmp/test.mp4", 64, 64, 30, encode_args=list(NVENC_H264_FFMPEG_ARGS))

        # stdin.write should be called for each frame
        self.assertEqual(mock_proc.stdin.write.call_count, 3)
        mock_proc.stdin.close.assert_called_once()
        # Avoid PIPE deadlock: ffmpeg stderr must not fill an unread pipe
        self.assertEqual(mock_popen.call_args.kwargs.get("stdout"), subprocess.DEVNULL)
        self.assertEqual(mock_popen.call_args.kwargs.get("stderr"), subprocess.DEVNULL)

    @patch("checks.utils.models.cwip.visualization.image_video_utils.subprocess.Popen")
    def test_broken_pipe_raises(self, mock_popen: MagicMock) -> None:
        """Test BrokenPipeError handling."""
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write.side_effect = BrokenPipeError("pipe broken")
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.returncode = 1
        mock_proc.wait.return_value = None
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        with self.assertRaises(RuntimeError):
            _write_video_with_ffmpeg_internal(
                frames, "/tmp/test.mp4", 64, 64, 30, encode_args=list(NVENC_H264_FFMPEG_ARGS)
            )

    @patch("checks.utils.models.cwip.visualization.image_video_utils.subprocess.Popen")
    def test_nonzero_returncode_raises(self, mock_popen: MagicMock) -> None:
        """Test non-zero return code raises RuntimeError."""
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = b"encoding failed"
        mock_proc.returncode = 1
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        with self.assertRaises(RuntimeError):
            _write_video_with_ffmpeg_internal(
                frames, "/tmp/test.mp4", 64, 64, 30, encode_args=list(NVENC_H264_FFMPEG_ARGS)
            )

    @patch("checks.utils.models.cwip.visualization.image_video_utils.subprocess.Popen")
    def test_nvenc_encoding_args(self, mock_popen: MagicMock) -> None:
        """Test that NVENC encoding adds correct arguments."""
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.returncode = 0
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        _write_video_with_ffmpeg_internal(frames, "/tmp/test.mp4", 64, 64, 30, encode_args=list(NVENC_H264_FFMPEG_ARGS))

        # Check that h264_nvenc codec is used
        cmd = mock_popen.call_args[0][0]
        self.assertIn("h264_nvenc", cmd)

    @patch("checks.utils.models.cwip.visualization.image_video_utils.subprocess.Popen")
    def test_scale_filter_added_when_enc_dims_differ(self, mock_popen: MagicMock) -> None:
        """A ``scale`` video filter downscales to the encoded dims when they differ from the source dims."""
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.returncode = 0
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        frames = [np.zeros((16, 16, 3), dtype=np.uint8)]
        _write_video_with_ffmpeg_internal(
            frames,
            "/tmp/test.mp4",
            5000,
            3000,
            30,
            encode_args=list(NVENC_H264_FFMPEG_ARGS),
            enc_width=4096,
            enc_height=2456,
        )

        cmd = mock_popen.call_args[0][0]
        # Source dims drive the raw stdin read...
        self.assertIn("5000x3000", cmd)
        # ...while a scale filter downscales to the encoded (capped) dims.
        self.assertIn("-vf", cmd)
        self.assertIn("scale=4096:2456", cmd)

    @patch("checks.utils.models.cwip.visualization.image_video_utils.subprocess.Popen")
    def test_no_scale_filter_when_enc_dims_match(self, mock_popen: MagicMock) -> None:
        """No ``scale`` filter is added when encoded dims equal source dims (the common case)."""
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.returncode = 0
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        _write_video_with_ffmpeg_internal(frames, "/tmp/test.mp4", 64, 64, 30, encode_args=list(NVENC_H264_FFMPEG_ARGS))

        cmd = mock_popen.call_args[0][0]
        self.assertNotIn("-vf", cmd)

    @patch("checks.utils.models.cwip.visualization.image_video_utils.subprocess.Popen")
    def test_cpu_fallback_uses_svtav1(self, mock_popen: MagicMock) -> None:
        """The SVT-AV1 args reach the ffmpeg command line intact, with no hardware encoder anywhere in it."""
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.returncode = 0
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        _write_video_with_ffmpeg_internal(frames, "/tmp/test.mp4", 64, 64, 30, encode_args=list(SVTAV1_FFMPEG_ARGS))

        cmd = mock_popen.call_args[0][0]
        self.assertIn("libsvtav1", cmd)
        self.assertNotIn("h264_nvenc", cmd)
        self.assertIn("-crf", cmd)


class TestGetPeImageTransformExtended(unittest.TestCase):
    """Extended tests for get_pe_image_transform function."""

    def test_no_resize(self) -> None:
        """Test get_pe_image_transform without resize."""
        transform = get_pe_image_transform(image_size=None, normalize=True, include_resize=False)

        img = torch.randint(0, 255, (3, 128, 128), dtype=torch.uint8)
        result = transform(img)

        # Size should not change (no resize)
        self.assertEqual(result.shape, (3, 128, 128))
        self.assertEqual(result.dtype, torch.float32)

    def test_resize_required_without_size_raises(self) -> None:
        """Test that include_resize=True without image_size raises ValueError."""
        with self.assertRaises(ValueError):
            get_pe_image_transform(image_size=None, include_resize=True)

    def test_no_normalize(self) -> None:
        """Test get_pe_image_transform without normalization."""
        transform = get_pe_image_transform(image_size=64, normalize=False)

        img = torch.randint(0, 255, (3, 128, 128), dtype=torch.uint8)
        result = transform(img)

        self.assertEqual(result.shape, (3, 64, 64))
        # Without normalization, values should be in [0, 1] after ToDtype
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)


class TestToUint8Extended(unittest.TestCase):
    """Extended tests for to_uint8 function."""

    def test_int_array(self) -> None:
        """Test with int32 array in [0, 255] range."""
        x = np.array([0, 128, 255], dtype=np.int32)
        result = to_uint8(x)
        self.assertEqual(result.dtype, np.uint8)

    def test_float64_array(self) -> None:
        """Test with float64 array."""
        x = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        result = to_uint8(x)
        self.assertEqual(result.dtype, np.uint8)


class TestWriteVideoWithFfmpegMock(unittest.TestCase):
    """Mock-based tests for write_video_with_ffmpeg function."""

    @patch("checks.utils.models.cwip.visualization.image_video_utils._write_video_with_ffmpeg_internal")
    @patch(
        "checks.utils.models.cwip.visualization.image_video_utils.get_video_encode_args",
        return_value=list(NVENC_H264_FFMPEG_ARGS),
    )
    def test_nvenc_success(self, _mock_args: MagicMock, mock_internal: MagicMock) -> None:
        """Test that the hardware path is used when the ladder selects it."""
        frames = np.random.randint(0, 255, (4, 32, 32, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.mp4")
            mock_internal.return_value = None
            write_video_with_ffmpeg(frames, output_path, 32, 32, fps=30)
            mock_internal.assert_called_once()
            _, kwargs = mock_internal.call_args
            self.assertEqual(kwargs.get("encode_args"), list(NVENC_H264_FFMPEG_ARGS))


class TestTiledFramesGeneratorGrid(unittest.TestCase):
    """Grid-based tests for tiled_frames_generator."""

    def test_basic_1x2_tiling(self) -> None:
        """Test basic frame tiling with 1x2 grid."""
        tiles = [
            np.random.randint(0, 255, (4, 32, 32, 3), dtype=np.uint8),
            np.random.randint(0, 255, (4, 32, 32, 3), dtype=np.uint8),
        ]
        result = list(
            tiled_frames_generator(
                tiles, num_rows=1, num_cols=2, num_frames=4, tile_height=32, tile_width=32, use_gpu=False
            )
        )
        self.assertEqual(len(result), 4)  # 4 frames
        self.assertEqual(result[0].shape[0], 32)  # 1 row * 32
        self.assertEqual(result[0].shape[1], 64)  # 2 cols * 32

    def test_tiling_with_none_tile(self) -> None:
        """Test tiling when some tiles are None (black placeholder)."""
        tiles = [
            np.random.randint(0, 255, (4, 32, 32, 3), dtype=np.uint8),
            None,  # Black placeholder
        ]
        result = list(
            tiled_frames_generator(
                tiles, num_rows=1, num_cols=2, num_frames=4, tile_height=32, tile_width=32, use_gpu=False
            )
        )
        self.assertEqual(len(result), 4)

    def test_tiling_2x2_grid(self) -> None:
        """Test tiling with 2x2 grid."""
        tiles = [
            np.random.randint(0, 255, (2, 16, 16, 3), dtype=np.uint8),
            np.random.randint(0, 255, (2, 16, 16, 3), dtype=np.uint8),
            np.random.randint(0, 255, (2, 16, 16, 3), dtype=np.uint8),
            np.random.randint(0, 255, (2, 16, 16, 3), dtype=np.uint8),
        ]
        result = list(
            tiled_frames_generator(
                tiles, num_rows=2, num_cols=2, num_frames=2, tile_height=16, tile_width=16, use_gpu=False
            )
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (32, 32, 3))


class TestGetPeImageTransformAdditional(unittest.TestCase):
    """Additional tests for get_pe_image_transform."""

    def test_no_resize_no_normalize(self) -> None:
        """Test transform with no resize and no normalize."""
        transform = get_pe_image_transform(image_size=32, normalize=False, include_resize=False)
        self.assertIsNotNone(transform)

    def test_list_image_size(self) -> None:
        """Test transform with list image_size."""
        transform = get_pe_image_transform(image_size=(224, 224))
        self.assertIsNotNone(transform)

    def test_batch_of_images(self) -> None:
        """Test transform on a batch of images (4D tensor)."""
        transform = get_pe_image_transform(image_size=32, normalize=True)
        images = torch.randint(0, 255, (2, 3, 64, 64), dtype=torch.uint8)
        result = transform(images)
        self.assertEqual(result.shape, (2, 3, 32, 32))


class TestPreprocessWorldCameraVideoPair(unittest.TestCase):
    """Tests for preprocess_world_camera_video_pair function."""

    def test_basic_decode(self) -> None:
        """Test basic preprocessing through the ffmpeg reader."""
        with _fake_video_io(_pair(_StreamSpec(num_frames=10), _StreamSpec(num_frames=10))):
            result = preprocess_world_camera_video_pair(
                world_video_path=_WORLD,
                camera_video_path=_CAMERA,
                num_frames=4,
                sample_method="uniform",
            )
        self.assertIn("world_video", result)
        self.assertIn("camera_video", result)
        self.assertEqual(result["num_frames"], 4)
        # (N, C, H, W) RGB float, the contract the CWIP model consumes.
        self.assertEqual(result["camera_video"].shape, (4, 3, 64, 64))
        self.assertEqual(result["camera_video"].dtype, torch.float32)

    def test_decodes_exactly_the_sampled_indices(self) -> None:
        """The frames pulled off each reader are the sampled indices, in order."""
        with _fake_video_io(_pair(_StreamSpec(num_frames=10), _StreamSpec(num_frames=10))) as readers:
            result = preprocess_world_camera_video_pair(
                world_video_path=_WORLD,
                camera_video_path=_CAMERA,
                num_frames=3,
                sample_method="first",
            )
        self.assertEqual(readers[_CAMERA][0].requested_indices, result["camera_frame_indices"])
        self.assertEqual(readers[_WORLD][0].requested_indices, result["world_frame_indices"])

    def test_forward_pass_opens_one_reader_per_stream(self) -> None:
        """A full in-order pass costs exactly one decode per stream, not one per frame."""
        with _fake_video_io(_pair(_StreamSpec(num_frames=10), _StreamSpec(num_frames=10))) as readers:
            preprocess_world_camera_video_pair(
                world_video_path=_WORLD,
                camera_video_path=_CAMERA,
                num_frames=-1,
            )
        self.assertEqual(len(readers[_WORLD]), 1)
        self.assertEqual(len(readers[_CAMERA]), 1)
        self.assertTrue(readers[_WORLD][0].closed)
        self.assertTrue(readers[_CAMERA][0].closed)

    def test_first_sample_method(self) -> None:
        """Test 'first' sample method."""
        with _fake_video_io(_pair(_StreamSpec(num_frames=10), _StreamSpec(num_frames=10))):
            result = preprocess_world_camera_video_pair(
                world_video_path=_WORLD,
                camera_video_path=_CAMERA,
                num_frames=3,
                sample_method="first",
            )
        self.assertEqual(result["num_frames"], 3)
        self.assertEqual(result["camera_frame_indices"], [0, 1, 2])

    def test_last_sample_method(self) -> None:
        """Test 'last' sample method."""
        with _fake_video_io(_pair(_StreamSpec(num_frames=10), _StreamSpec(num_frames=10))):
            result = preprocess_world_camera_video_pair(
                world_video_path=_WORLD,
                camera_video_path=_CAMERA,
                num_frames=3,
                sample_method="last",
            )
        self.assertEqual(result["num_frames"], 3)
        self.assertEqual(result["camera_frame_indices"], [7, 8, 9])

    def test_middle_sample_method(self) -> None:
        """Test 'middle' sample method."""
        with _fake_video_io(_pair(_StreamSpec(num_frames=10), _StreamSpec(num_frames=10))):
            result = preprocess_world_camera_video_pair(
                world_video_path=_WORLD,
                camera_video_path=_CAMERA,
                num_frames=4,
                sample_method="middle",
            )
        self.assertEqual(result["num_frames"], 4)

    def test_random_sample_method(self) -> None:
        """Test 'random' sample method with seed."""
        with _fake_video_io(_pair(_StreamSpec(num_frames=10), _StreamSpec(num_frames=10))):
            result = preprocess_world_camera_video_pair(
                world_video_path=_WORLD,
                camera_video_path=_CAMERA,
                num_frames=4,
                sample_method="random",
                seed=42,
            )
        self.assertEqual(result["num_frames"], 4)

    def test_mixed_sample_method(self) -> None:
        """Test 'mixed' sample method (randomly picks one of the other methods)."""
        with _fake_video_io(_pair(_StreamSpec(num_frames=10), _StreamSpec(num_frames=10))):
            result = preprocess_world_camera_video_pair(
                world_video_path=_WORLD,
                camera_video_path=_CAMERA,
                num_frames=4,
                sample_method="mixed",
                seed=42,
            )
        self.assertEqual(result["num_frames"], 4)

    def test_all_frames(self) -> None:
        """Test num_frames=-1 to get all frames."""
        with _fake_video_io(_pair(_StreamSpec(num_frames=10), _StreamSpec(num_frames=10))):
            result = preprocess_world_camera_video_pair(
                world_video_path=_WORLD,
                camera_video_path=_CAMERA,
                num_frames=-1,
            )
        self.assertEqual(result["num_frames"], 10)

    def test_with_transform(self) -> None:
        """Test preprocessing with a transform applied."""
        transform = get_pe_image_transform(image_size=32, normalize=True)
        with _fake_video_io(_pair(_StreamSpec(num_frames=10), _StreamSpec(num_frames=10))):
            result = preprocess_world_camera_video_pair(
                world_video_path=_WORLD,
                camera_video_path=_CAMERA,
                num_frames=4,
                transform=transform,
            )
        self.assertIn("world_video", result)

    def test_with_paired_augmentation(self) -> None:
        """Test preprocessing with paired augmentation."""
        # Simple paired augmentation: identity
        paired_aug = MagicMock(side_effect=lambda w, c: (w, c))
        with _fake_video_io(_pair(_StreamSpec(num_frames=10), _StreamSpec(num_frames=10))):
            result = preprocess_world_camera_video_pair(
                world_video_path=_WORLD,
                camera_video_path=_CAMERA,
                num_frames=4,
                paired_augmentation=paired_aug,
                image_size=(64, 64),
            )
        self.assertIn("world_video", result)
        paired_aug.assert_called_once()

    def test_different_fps_videos(self) -> None:
        """Test preprocessing when world and camera have different FPS."""
        streams = _pair(_StreamSpec(num_frames=30, fps=30.0), _StreamSpec(num_frames=20, fps=20.0))
        with _fake_video_io(streams):
            result = preprocess_world_camera_video_pair(
                world_video_path=_WORLD,
                camera_video_path=_CAMERA,
                num_frames=4,
                verbose=True,
            )
        self.assertEqual(result["num_frames"], 4)

    def test_fewer_frames_than_requested(self) -> None:
        """Test when video has fewer frames than requested."""
        with _fake_video_io(_pair(_StreamSpec(num_frames=3), _StreamSpec(num_frames=3))):
            result = preprocess_world_camera_video_pair(
                world_video_path=_WORLD,
                camera_video_path=_CAMERA,
                num_frames=10,
                verbose=True,
            )
        # Should cap at available frames
        self.assertLessEqual(result["num_frames"], 3)

    def test_invalid_world_path(self) -> None:
        """Test that invalid world video path raises ValueError."""
        with self.assertRaises(ValueError):
            preprocess_world_camera_video_pair(
                world_video_path="/nonexistent/world.mp4",
                camera_video_path="/nonexistent/camera.mp4",
            )

    def test_paired_augmentation_without_image_size(self) -> None:
        """Test that paired_augmentation without image_size raises ValueError."""
        with self.assertRaises(ValueError):
            preprocess_world_camera_video_pair(
                world_video_path="/fake/world.mp4",
                camera_video_path="/fake/camera.mp4",
                paired_augmentation=lambda w, c: (w, c),
                image_size=None,
            )

    def test_with_nearest_interpolation_mode(self) -> None:
        """Test with 'nearest' world_frame_interpolation_mode."""
        paired_aug = MagicMock(side_effect=lambda w, c: (w, c))
        with _fake_video_io(_pair(_StreamSpec(num_frames=10), _StreamSpec(num_frames=10))):
            result = preprocess_world_camera_video_pair(
                world_video_path=_WORLD,
                camera_video_path=_CAMERA,
                num_frames=4,
                paired_augmentation=paired_aug,
                image_size=(32, 32),
                world_frame_interpolation_mode="nearest",
            )
        self.assertIn("world_video", result)

    def test_with_bilinear_interpolation_mode(self) -> None:
        """Test with 'bilinear' world_frame_interpolation_mode (falls back to nearest-exact)."""
        paired_aug = MagicMock(side_effect=lambda w, c: (w, c))
        with _fake_video_io(_pair(_StreamSpec(num_frames=10), _StreamSpec(num_frames=10))):
            result = preprocess_world_camera_video_pair(
                world_video_path=_WORLD,
                camera_video_path=_CAMERA,
                num_frames=4,
                paired_augmentation=paired_aug,
                image_size=(32, 32),
                world_frame_interpolation_mode="bilinear",
            )
        self.assertIn("world_video", result)

    def test_different_fps_verbose(self) -> None:
        """Test with different world/camera FPS and verbose=True."""
        # World at 30fps, camera at 15fps -> different effective frames
        streams = _pair(_StreamSpec(num_frames=30, fps=30.0), _StreamSpec(num_frames=15, fps=15.0))
        with _fake_video_io(streams):
            result = preprocess_world_camera_video_pair(
                world_video_path=_WORLD,
                camera_video_path=_CAMERA,
                num_frames=4,
                verbose=True,
            )
        self.assertIn("world_video", result)
        self.assertEqual(result["num_frames"], 4)

    def test_paired_augmentation_verbose(self) -> None:
        """Test with paired_augmentation and verbose=True."""
        paired_aug = MagicMock(side_effect=lambda w, c: (w, c))
        with _fake_video_io(_pair(_StreamSpec(num_frames=10), _StreamSpec(num_frames=10))):
            result = preprocess_world_camera_video_pair(
                world_video_path=_WORLD,
                camera_video_path=_CAMERA,
                num_frames=4,
                paired_augmentation=paired_aug,
                image_size=(32, 32),
                verbose=True,
            )
        self.assertIn("world_video", result)
        paired_aug.assert_called_once()

    def test_invalid_sample_method(self) -> None:
        """Test that invalid sample method raises ValueError."""
        with (
            _fake_video_io(_pair(_StreamSpec(num_frames=10), _StreamSpec(num_frames=10))),
            self.assertRaises(ValueError),
        ):
            preprocess_world_camera_video_pair(
                world_video_path=_WORLD,
                camera_video_path=_CAMERA,
                num_frames=4,
                sample_method="invalid_method",
            )

    def test_camera_only_materializes_black_world_by_default(self) -> None:
        """The camera-only fallback allocates a black world of the camera's shape unless opted out."""
        with _fake_video_io({_CAMERA: _StreamSpec(num_frames=10)}):
            result = preprocess_world_camera_video_pair(
                world_video_path="",
                camera_video_path=_CAMERA,
                num_frames=4,
                create_black_world_if_missing=True,
            )
        world_video = result["world_video"]
        self.assertIsNotNone(world_video)
        assert world_video is not None  # narrow for the shape comparison below
        self.assertEqual(world_video.shape, result["camera_video"].shape)
        self.assertEqual(float(world_video.abs().sum()), 0.0)

    def test_camera_only_skips_black_world_when_not_materialized(self) -> None:
        """``materialize_black_world=False`` returns ``world_video=None`` without allocating the full tensor."""
        with _fake_video_io({_CAMERA: _StreamSpec(num_frames=10)}):
            result = preprocess_world_camera_video_pair(
                world_video_path="",
                camera_video_path=_CAMERA,
                num_frames=4,
                create_black_world_if_missing=True,
                materialize_black_world=False,
            )
        self.assertIsNone(result["world_video"])
        self.assertEqual(result["camera_video"].shape[0], result["num_frames"])


class TestDecodeVideoFramesToTensor(unittest.TestCase):
    """Tests for the ffmpeg-reader decode helper and its BGR-to-RGB boundary conversion."""

    def test_converts_bgr_hwc_to_rgb_chw_float(self) -> None:
        """The reader's BGR uint8 HWC frames become RGB CHW floats in [0, 1]."""
        spec = _StreamSpec(num_frames=4, height=8, width=8)
        with _fake_video_io({_CAMERA: spec}):
            tensor = _decode_video_frames_to_tensor(_CAMERA, [0, 2])

        self.assertEqual(tensor.shape, (2, 3, 8, 8))
        self.assertEqual(tensor.dtype, torch.float32)
        # Frame 0 is BGR (0, 1, 2), so as RGB it must read (2, 1, 0) scaled by 1/255.
        for channel, expected in enumerate((2, 1, 0)):
            self.assertTrue(torch.allclose(tensor[0, channel], torch.full((8, 8), expected / 255.0)))
        # Frame 2 is BGR (2, 3, 4) -> RGB (4, 3, 2), confirming the right index was decoded.
        for channel, expected in enumerate((4, 3, 2)):
            self.assertTrue(torch.allclose(tensor[1, channel], torch.full((8, 8), expected / 255.0)))

    def test_in_order_indices_use_one_reader(self) -> None:
        """A forward walk opens a single reader and closes it."""
        spec = _StreamSpec(num_frames=6)
        with _fake_video_io({_CAMERA: spec}) as readers:
            _decode_video_frames_to_tensor(_CAMERA, [0, 1, 2, 3, 4, 5])
        self.assertEqual(len(readers[_CAMERA]), 1)
        self.assertTrue(readers[_CAMERA][0].closed)

    def test_unreadable_frame_raises(self) -> None:
        """A frame the reader cannot produce is an error, not a silently short tensor."""
        spec = _StreamSpec(num_frames=3)
        with _fake_video_io({_CAMERA: spec}), self.assertRaises(VideoDecodeError):
            _decode_video_frames_to_tensor(_CAMERA, [0, 99])

    def test_empty_indices_raises(self) -> None:
        """Requesting no frames is a caller bug rather than an empty tensor."""
        spec = _StreamSpec(num_frames=3)
        with _fake_video_io({_CAMERA: spec}), self.assertRaises(VideoDecodeError):
            _decode_video_frames_to_tensor(_CAMERA, [])


class TestWriteVideoWithFfmpegCpuFallbackReal(unittest.TestCase):
    """Real (un-mocked) encoding test verifying CPU fallback produces a valid video."""

    @patch(
        "checks.utils.models.cwip.visualization.image_video_utils.get_video_encode_args",
        return_value=list(SVTAV1_FFMPEG_ARGS),
    )
    def test_cpu_fallback_produces_valid_video(self, _mock_encode_args: MagicMock) -> None:
        """Write frames through the real ffmpeg subprocess with the SVT-AV1 CPU fallback."""
        num_frames = 3
        H, W = 128, 128
        frames = [np.random.randint(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(num_frames)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "cpu_fallback.mp4")
            write_video_with_ffmpeg(frames, output_path, W, H, fps=30)

            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)


if __name__ == "__main__":
    unittest.main()
