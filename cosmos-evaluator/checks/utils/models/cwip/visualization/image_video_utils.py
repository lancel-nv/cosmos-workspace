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

from collections.abc import Iterable, Iterator, Sequence
import enum
import logging
import os
import random
import subprocess
import time
from typing import IO, Any, TypedDict, cast

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torchvision.transforms import v2
from tqdm import tqdm

from checks.utils.video import (
    SVTAV1_FFMPEG_ARGS,
    SequentialFrameReader,
    VideoDecodeError,
    get_video_encode_args,
    get_video_fps,
    get_video_frame_count,
)

logger = logging.getLogger(__name__)


class WorldCameraVideoPairSampleMethod(enum.StrEnum):
    """How to pick frame indices along a timeline (single stream or logical paired timeline)."""

    UNIFORM = "uniform"
    RANDOM = "random"
    FIRST = "first"
    LAST = "last"
    MIDDLE = "middle"
    MIXED = "mixed"


class WorldCameraVideoPairResult(TypedDict):
    """Structured return from :func:`preprocess_world_camera_video_pair` and
    :func:`_preprocess_camera_only_with_black_world`.

    Fields:
        world_video: World stream frames, shape ``(N, C, H, W)``, dtype float in ``[0, 1]`` before transforms. ``None``
            on the camera-only fallback when ``materialize_black_world=False`` — a caller that forwards
            ``world_image=None`` has no use for a full-resolution black tensor, so it is not allocated.
        camera_video: Camera stream frames, same shape convention as ``world_video``.
        world_fps: Reported frames per second of the world video.
        camera_fps: Reported frames per second of the camera video.
        num_frames: Number of frames after sampling (``N``).
        camera_frame_indices: Decoder indices used for the camera stream.
        world_frame_indices: Decoder indices used for the world stream.
        logical_frame_indices: Index on the shared timeline for each sampled frame. ``camera_frame_indices`` and
            ``world_frame_indices`` are this projected onto each stream's own rate, so a caller that needs to map
            an output frame back to clip time reads this rather than re-deriving it from the two rates.
    """

    world_video: torch.Tensor | None
    camera_video: torch.Tensor
    world_fps: float
    camera_fps: float
    num_frames: int
    camera_frame_indices: list[int]
    world_frame_indices: list[int]
    logical_frame_indices: list[int]


def _normalize_sample_method(
    sample_method: str | WorldCameraVideoPairSampleMethod,
) -> WorldCameraVideoPairSampleMethod:
    """Coerce a sampling method string or enum member to :class:`WorldCameraVideoPairSampleMethod`.

    Args:
        sample_method: Enum member or string matching a member value (e.g. ``"uniform"``).

    Returns:
        The corresponding :class:`WorldCameraVideoPairSampleMethod` member.

    Raises:
        ValueError: If ``sample_method`` is not a valid enum value.
    """
    if isinstance(sample_method, WorldCameraVideoPairSampleMethod):
        return sample_method
    return WorldCameraVideoPairSampleMethod(sample_method)


def _probe_video_metadata(path: str) -> tuple[float, int]:
    """Probe a video's frame rate and frame count with the vendored ffprobe.

    Args:
        path: Filesystem path to the video file.

    Returns:
        Tuple ``(fps, total_frame_count)``.

    Raises:
        VideoDecodeError: If the file is missing, or either value cannot be determined.
    """
    fps = get_video_fps(path)
    if fps is None:
        raise VideoDecodeError(f"Could not determine FPS for video: {path}")
    total_frames = get_video_frame_count(path)
    if total_frames is None:
        raise VideoDecodeError(f"Could not determine frame count for video: {path}")
    return fps, total_frames


def _sample_frame_indices_single_stream(
    effective_total_frames: int,
    num_frames: int,
    sample_method: str | WorldCameraVideoPairSampleMethod,
    seed: int | None,
    *,
    invalid_fallback_uniform: bool = False,
) -> tuple[int, list[int], WorldCameraVideoPairSampleMethod]:
    """Pick frame indices along one timeline (single stream or shared logical index space before FPS mapping).

    When ``num_frames == -1``, all frames ``0 .. effective_total_frames - 1`` are used; ``sample_method`` does not
    affect which frames are chosen (only the returned resolved label).

    Args:
        effective_total_frames: Number of frames available on that timeline.
        num_frames: Target count, or ``-1`` to use every frame.
        sample_method: Sampling strategy; ``MIXED`` picks another strategy at random (when ``seed`` is set,
            sampling is deterministic for that stream).
        seed: Optional RNG seed for ``RANDOM``, ``MIXED``, and sub-method selection.
        invalid_fallback_uniform: If ``True``, strings that are not valid enum values fall back to uniform
            sampling (camera-only legacy). If ``False``, invalid methods raise :class:`ValueError`.

    Returns:
        Tuple ``(n, frame_indices, resolved_method)`` where ``n`` is the number of frames selected,
        ``frame_indices`` are integer indices into the stream, and ``resolved_method`` is the strategy used
        after resolving ``MIXED`` (and after fallback, if any).

    Raises:
        ValueError: If the sampling method is invalid and ``invalid_fallback_uniform`` is ``False``.
    """
    try:
        sm = _normalize_sample_method(sample_method)
    except ValueError:
        if not invalid_fallback_uniform:
            raise
        sm = WorldCameraVideoPairSampleMethod.UNIFORM
    if num_frames == -1:
        return effective_total_frames, list(range(effective_total_frames)), sm
    n = min(num_frames, effective_total_frames)
    rng = random.Random(seed) if seed is not None else random
    if sm == WorldCameraVideoPairSampleMethod.MIXED:
        other_methods = [m for m in WorldCameraVideoPairSampleMethod if m != WorldCameraVideoPairSampleMethod.MIXED]
        sm = rng.choice(other_methods)
    match sm:
        case WorldCameraVideoPairSampleMethod.UNIFORM:
            frame_indices = [int(i * (effective_total_frames / n)) for i in range(n)]
        case WorldCameraVideoPairSampleMethod.RANDOM:
            frame_indices = rng.sample(range(effective_total_frames), n)
        case WorldCameraVideoPairSampleMethod.FIRST:
            frame_indices = list(range(n))
        case WorldCameraVideoPairSampleMethod.LAST:
            frame_indices = list(range(effective_total_frames - n, effective_total_frames))
        case WorldCameraVideoPairSampleMethod.MIDDLE:
            start = (effective_total_frames - n) // 2
            frame_indices = list(range(start, start + n))
        case _:
            raise ValueError(f"Unhandled sample method: {sm}")
    return n, frame_indices, sm


def _decode_video_frames_to_tensor(video_path: str, frame_indices: Sequence[int]) -> torch.Tensor:
    """Decode the given frame indices of a video to a float tensor.

    Backed by :class:`~checks.utils.video.SequentialFrameReader`, so an in-order walk (the production case, where
    ``num_frames=-1`` asks for every frame) costs exactly one ffmpeg child. Only indices that move backwards --
    reachable via ``sample_method="random"`` -- restart the decoder.

    The reader yields BGR ``uint8`` HWC frames, matching cv2; this function converts to the RGB CHW float contract
    the CWIP model expects.

    Args:
        video_path: Filesystem path to the video file.
        frame_indices: Zero-based frame indices to decode.

    Returns:
        Tensor of shape ``(N, C, H, W)`` with values in ``[0, 1]`` (uint8 source scaled by ``1/255``).

    Raises:
        VideoDecodeError: If ``frame_indices`` is empty or any requested frame cannot be read.
    """
    if not frame_indices:
        raise VideoDecodeError(f"No frames requested from {video_path}")

    frames: list[np.ndarray] = []
    with SequentialFrameReader(video_path) as reader:
        for idx in frame_indices:
            frame = reader.read(idx)
            if frame is None:
                raise VideoDecodeError(f"Could not read frame {idx} from {video_path}")
            frames.append(frame)

    # (N, H, W, 3) BGR -> (N, 3, H, W) RGB. ascontiguousarray materializes the reversed/transposed view, which
    # torch.from_numpy requires (it rejects negative strides).
    bgr_nhwc = np.stack(frames)
    rgb_nchw = np.ascontiguousarray(bgr_nhwc[..., ::-1].transpose(0, 3, 1, 2))
    return torch.from_numpy(rgb_nchw).float() / 255.0


def _probe_world_camera_metadata(
    world_video_path: str,
    camera_video_path: str,
    verbose: bool,
) -> tuple[float, float, int, int]:
    """Probe frame rate and frame count for both streams.

    Args:
        world_video_path: Path to the world (e.g. multiview) video.
        camera_video_path: Path to the camera video.
        verbose: If ``True``, log basic stream statistics (module logger).

    Returns:
        Tuple ``(world_fps, camera_fps, world_total_frames, camera_total_frames)``.

    Raises:
        VideoDecodeError: If either stream's metadata cannot be determined.
    """
    world_fps, world_total_frames = _probe_video_metadata(world_video_path)
    camera_fps, camera_total_frames = _probe_video_metadata(camera_video_path)
    if verbose:
        logger.info("World video: %s frames at %.2f FPS", world_total_frames, world_fps)
        logger.info("Camera video: %s frames at %.2f FPS", camera_total_frames, camera_fps)
    return world_fps, camera_fps, world_total_frames, camera_total_frames


def to_uint8(x: np.ndarray) -> np.ndarray:
    """Convert an image array to ``uint8``, scaling from ``[0, 1]`` when appropriate.

    Args:
        x: Input array. If already ``uint8``, returned unchanged. If values appear normalized (max ``<= 1.5``),
            values are clipped to ``[0, 1]`` and multiplied by 255; otherwise clipped to ``[0, 255]`` and cast.

    Returns:
        ``uint8`` array with the same shape as ``x``.
    """
    if x.dtype == np.uint8:
        return x

    # Store the original max value to avoid double-checking
    # Use 1.5 instead of 1.0 in case of pixel values are slightly greater than 1.0
    original_max = x.max()
    normalized = original_max <= 1.5

    # Converted chunk-wise into a preallocated result: the whole-array form ``(np.clip(x, 0, 1) * 255).astype`` holds
    # two full-size float temporaries at once, 2.25x a float32 input before the cast completes.
    high = 1.0 if normalized else 255.0
    if x.ndim == 0:
        scaled = np.clip(x, 0, high) * 255.0 if normalized else np.clip(x, 0, high)
        return scaled.astype(np.uint8)

    result = np.empty(x.shape, dtype=np.uint8)
    # Only a float input can hold its clipped, rescaled values in its own dtype. Clipping a bool mask into a bool buffer
    # raises, and a narrow integer overflows on ``* 255``, so every non-float input goes through a float32 scratch.
    scratch_dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.dtype(np.float32)
    # The 8 MB budget is measured against the scratch, not the input, so a narrow input dtype cannot enlarge it.
    row_bytes = max(int(np.prod(x.shape[1:], dtype=np.int64)) * scratch_dtype.itemsize, 1)
    chunk_rows = max(1, int(8 * 1024 * 1024 // row_bytes))
    scratch = np.empty((min(chunk_rows, x.shape[0]), *x.shape[1:]), dtype=scratch_dtype)

    for start in range(0, x.shape[0], chunk_rows):
        stop = min(start + chunk_rows, x.shape[0])
        window = scratch[: stop - start]
        np.clip(x[start:stop], 0, high, out=window)
        if normalized:
            np.multiply(window, 255.0, out=window)
        # The values are already clipped into range, so the narrowing cast cannot wrap.
        np.copyto(result[start:stop], window, casting="unsafe")

    return result


def draw_topright_lines(
    frame: np.ndarray,
    lines: Sequence[str],
    *,
    font_scale: float = 0.6,
    thickness: int = 2,
    y_start: int = 25,
    y_spacing: int = 25,
    margin: int = 10,
    text_color: tuple[int, int, int] = (255, 255, 255),
    outline_color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw right-aligned, outlined text lines in the top-right corner of a single ``uint8`` RGB frame, in place.

    Each line is right-aligned ``margin`` px from the right edge and stacked vertically starting at baseline
    ``y_start`` with ``y_spacing`` px between baselines. Text is drawn in ``text_color`` over a thicker
    ``outline_color`` stroke (``thickness + 2``) so it stays legible over arbitrary imagery. Shared by the
    visualization label/overlay helpers, which all drew this same stacked top-right text loop.

    Args:
        frame: ``(H, W, 3)`` ``uint8`` RGB frame, mutated in place.
        lines: Text lines to draw, top to bottom.
        font_scale: OpenCV ``FONT_HERSHEY_SIMPLEX`` scale.
        thickness: Foreground stroke thickness (the outline uses ``thickness + 2``).
        y_start: Baseline y of the first line.
        y_spacing: Vertical spacing between consecutive baselines.
        margin: Right-edge margin in pixels.
        text_color: Foreground color (frames here are RGB).
        outline_color: Outline color.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    _h, w = frame.shape[:2]
    for i, text in enumerate(lines):
        (text_w, _text_h), _baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x = w - text_w - margin
        y = y_start + i * y_spacing
        cv2.putText(frame, text, (x, y), font, font_scale, outline_color, thickness + 2)
        cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)


def tiled_frames_generator(
    tiles: Sequence[np.ndarray | None],
    num_rows: int,
    num_cols: int,
    num_frames: int,
    tile_height: int,
    tile_width: int,
    use_gpu: bool = True,
) -> Iterator[np.ndarray]:
    """
    Generator that yields tiled frames for video creation with flexible grid layout.

    Uses GPU acceleration for frame resizing when available for 3-5x speedup.

    Args:
        tiles: List of tile frame arrays, each with shape (num_frames, H, W, 3) or None for black placeholder
               Order is row-major: [row0_col0, row0_col1, ..., row1_col0, row1_col1, ...]
               Length must equal num_rows * num_cols
        num_rows: Number of rows in the grid
        num_cols: Number of columns in the grid
        num_frames: Number of frames to process
        tile_height: Target height for each tile
        tile_width: Target width for each tile
        use_gpu: Whether to use GPU acceleration for frame processing (default: True)

    Yields:
        ``uint8`` RGB frames of shape ``(num_rows * tile_height, num_cols * tile_width, 3)``.
    """
    expected_tiles = num_rows * num_cols
    if len(tiles) != expected_tiles:
        raise ValueError(f"Expected {expected_tiles} tiles for {num_rows}x{num_cols} grid, got {len(tiles)}")

    # Check if GPU acceleration is available
    use_gpu_accel = use_gpu and torch.cuda.is_available()

    if use_gpu_accel:
        logger.info("[FRAMES] Using GPU acceleration for frame resizing")
    else:
        logger.info("[FRAMES] Using CPU for frame resizing")

    logger.info("[FRAMES] Generating %s frames (%sx%s layout)...", num_frames, num_rows, num_cols)

    try:
        _hide_tqdm = dist.is_initialized() and dist.get_rank() != 0
    except (ImportError, RuntimeError, AttributeError):
        # RuntimeError: e.g. get_rank() if process group state is inconsistent
        # AttributeError: missing is_initialized / get_rank on very old or stub dist
        _hide_tqdm = False

    for i in tqdm(
        range(num_frames),
        desc=f"Creating {num_rows}x{num_cols} tiled video",
        disable=_hide_tqdm,
    ):
        rows = []
        for row_idx in range(num_rows):
            row_tiles = []
            for col_idx in range(num_cols):
                tile_idx = row_idx * num_cols + col_idx
                tile = tiles[tile_idx]

                if use_gpu_accel:
                    # GPU path: use torch.nn.functional.interpolate
                    if tile is not None:
                        # Convert single frame to GPU tensor: (H, W, 3) -> (1, 3, H, W)
                        frame = torch.from_numpy(tile[i]).permute(2, 0, 1).unsqueeze(0).float().cuda()
                        # Resize using bilinear interpolation on GPU
                        resized = torch.nn.functional.interpolate(
                            frame, size=(tile_height, tile_width), mode="bilinear", align_corners=False
                        )[0]  # Remove batch dim -> (3, H, W)
                        # Convert back to numpy: (3, H, W) -> (H, W, 3)
                        resized_tile = resized.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    else:
                        resized_tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
                else:
                    # CPU path: use cv2.resize
                    if tile is not None:
                        resized_tile = cv2.resize(tile[i], (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                    else:
                        resized_tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)

                row_tiles.append(resized_tile)

            # Concatenate tiles horizontally to form a row
            row = np.concatenate(row_tiles, axis=1)
            rows.append(row)

        # Stack rows vertically
        tiled = np.concatenate(rows, axis=0)
        yield tiled

    logger.info("[FRAMES] Finished generating all %s frames", num_frames)


# H.264 NVENC max resolution per dimension on most GPUs. Encoded output is capped to this unconditionally so the
# result stays within the hardware encoder's reach at every tier.
H264_NVENC_MAX_DIM = 4096


def write_video_with_ffmpeg(
    tiled_frames: Iterable[np.ndarray],
    output_path: str,
    out_width: int,
    out_height: int,
    fps: float,
) -> None:
    """Write a sequence of RGB ``uint8`` frames to a video file via an ffmpeg subprocess.

    Encoder selection comes from the shared ladder in ``checks/utils/video.py`` (:func:`get_video_encode_args`):
    hardware H.264, then hardware AV1, then the royalty-free SVT-AV1 CPU encoder. A hardware tier that fails at
    runtime also falls back to SVT-AV1, which always works.

    Args:
        tiled_frames: Iterable or list of frames, each ``numpy.ndarray`` of shape ``(H, W, 3)``, RGB, ``uint8``.
            Non-list iterables are materialized to a list (may use substantial memory).
        output_path: Destination path for the encoded video file.
        out_width: Pixel width of each frame (must match frame content after any pipeline resize).
        out_height: Pixel height of each frame.
        fps: Output frame rate.

    Returns:
        ``None``. Raises :class:`RuntimeError` if ffmpeg exits non-zero after encoding.
    """
    logger.info(
        "[VIDEO] Entering video writer: will buffer frames (if streamed) then run ffmpeg/NVENC.",
    )
    logger.info("[VIDEO] Starting video encoding to: %s", output_path)
    logger.info("[VIDEO] Resolution: %sx%s, FPS: %s", out_width, out_height, fps)

    if not isinstance(tiled_frames, list):
        logger.info("[VIDEO] Materializing frame stream into memory (this can take a while)...")
        tiled_frames = list(tiled_frames)

    logger.info("[VIDEO] Have %s frames in memory; probing NVENC / CPU encoder...", len(tiled_frames))

    # Cap the *encoded* resolution to the H.264 NVENC per-dimension limit so the output stays H.264. Larger frames
    # would otherwise select HEVC, which ffmpeg tags ``hev1`` -- rejected by Apple players (QuickTime/Preview/Safari).
    # ffmpeg downscales the raw frames during encode (see ``enc_width``/``enc_height`` in the internal writer); dims are
    # rounded to even (yuv420p requirement). Codec selection is based on the encoded dims so it lands on H.264.
    enc_width, enc_height = out_width, out_height
    max_dim = max(out_width, out_height)
    if max_dim > H264_NVENC_MAX_DIM:
        scale = H264_NVENC_MAX_DIM / max_dim
        enc_width = max(2, (int(out_width * scale) // 2) * 2)
        enc_height = max(2, (int(out_height * scale) // 2) * 2)
        logger.info(
            "[VIDEO] %sx%s exceeds H.264 limit %s; encoding at %sx%s to keep output H.264/macOS-playable",
            out_width,
            out_height,
            H264_NVENC_MAX_DIM,
            enc_width,
            enc_height,
        )

    encode_args = get_video_encode_args()

    if encode_args != SVTAV1_FFMPEG_ARGS:
        try:
            _write_video_with_ffmpeg_internal(
                tiled_frames,
                output_path,
                out_width,
                out_height,
                fps,
                encode_args=encode_args,
                enc_width=enc_width,
                enc_height=enc_height,
            )
            return
        except RuntimeError:
            logger.warning("[VIDEO] Hardware encoding failed, falling back to SVT-AV1 CPU encoder")

    _write_video_with_ffmpeg_internal(
        tiled_frames,
        output_path,
        out_width,
        out_height,
        fps,
        encode_args=list(SVTAV1_FFMPEG_ARGS),
        enc_width=enc_width,
        enc_height=enc_height,
    )


def _write_video_with_ffmpeg_internal(
    tiled_frames: Iterable[np.ndarray],
    output_path: str,
    out_width: int,
    out_height: int,
    fps: float,
    encode_args: list[str],
    enc_width: int | None = None,
    enc_height: int | None = None,
) -> None:
    """Encode raw RGB frames with ffmpeg, reading RGB24 from stdin.

    Args:
        tiled_frames: Sequence of ``(H, W, 3)`` ``uint8`` RGB frames matching ``out_width`` and ``out_height``.
        output_path: Output video path (overwritten if present; ``-y`` is passed to ffmpeg).
        out_width: Source frame width in pixels (the raw RGB read from stdin).
        out_height: Source frame height in pixels.
        fps: Output frames per second.
        encode_args: Full ffmpeg encoder argument list from the shared ladder (``get_video_encode_args`` or
            ``SVTAV1_FFMPEG_ARGS``). Already carries ``-c:v``, the pixel format, and container flags.
        enc_width: Encoded output width; if it differs from ``out_width`` an ffmpeg ``scale`` filter downscales the
            frames during encode. Defaults to ``out_width`` (no rescale).
        enc_height: Encoded output height (see ``enc_width``). Defaults to ``out_height``.

    Returns:
        ``None``. Raises :class:`RuntimeError` on ffmpeg failure or broken pipe during write.
    """
    enc_width = enc_width if enc_width is not None else out_width
    enc_height = enc_height if enc_height is not None else out_height
    scale_args = (
        ["-vf", f"scale={enc_width}:{enc_height}"] if (enc_width != out_width or enc_height != out_height) else []
    )
    logger.info("[VIDEO] Using encoder args %s", encode_args)

    # Build ffmpeg command
    ffmpeg_cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-hide_banner",
        "-y",  # overwrite output file
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{out_width}x{out_height}",
        "-r",
        str(fps),
        "-i",
        "-",  # input from stdin
        "-an",  # no audio
        *scale_args,  # optional downscale to keep the encoded resolution within the H.264 limit
        *encode_args,
        output_path,
    ]

    logger.info("[VIDEO] Starting FFmpeg process...")
    # Never use PIPE for ffmpeg stdout/stderr without continuously reading them: ffmpeg may block once the ~64KiB stderr
    # buffer fills while Python is still writing raw frames to stdin. That looks like a hang right after CPU-side frame
    # generation (e.g. combined_patch 2x3 step).
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.info("[VIDEO] FFmpeg PID: %s", proc.pid)
    # ``stdin=subprocess.PIPE`` guarantees a writable stream; narrow away the ``IO[bytes] | None`` for mypy.
    proc_stdin = cast(IO[bytes], proc.stdin)

    try:
        logger.info("[VIDEO] Writing frames to FFmpeg stdin...")
        frame_count = 0
        last_progress_time = time.time()

        for frame in tiled_frames:
            proc_stdin.write(frame.tobytes())
            frame_count += 1

            # Show progress every 30 frames or every 2 seconds
            current_time = time.time()
            if frame_count % 30 == 0 or (current_time - last_progress_time) >= 2.0:
                logger.info("[VIDEO] Written %s frames to FFmpeg...", frame_count)
                last_progress_time = current_time

        logger.info("[VIDEO] Wrote %s frames to FFmpeg", frame_count)
        logger.info("[VIDEO] Closing FFmpeg stdin...")
        proc_stdin.close()
        mp_per_frame = (out_width * out_height) / 1_000_000
        logger.info(
            "[VIDEO] Raw RGB sent; NVENC is now encoding (%sx%s, ~%.1f MP/frame × %s f). "
            "This step has no per-frame output and can take several minutes at 4K-class sizes.",
            out_width,
            out_height,
            mp_per_frame,
            frame_count,
        )
    except BrokenPipeError as e:
        # FFmpeg died early (often encoder OOM / invalid params)
        logger.warning("[VIDEO] BrokenPipeError detected! FFmpeg process died at frame %s.", frame_count)
        try:
            proc_stdin.close()
        except (OSError, ValueError):
            pass  # stdin might already be closed or in error state
        proc.wait()
        raise RuntimeError(
            f"FFmpeg encoding failed after {frame_count} frames (stdin pipe broken; check NVENC limits, "
            f"resolution caps, and disk space). Original error: {e}"
        ) from e

    logger.info(
        "[VIDEO] Waiting for FFmpeg to exit (%s: encode + mux to disk). "
        "Stderr is disabled, so there is no encoder live log — heartbeats every 5s while the process runs.",
        encode_args,
    )
    # After stdin closes, ffmpeg may run a long time with no output (we discard stderr). Poll with timeout so users see
    # heartbeats instead of assuming a hang.
    wait_start = time.monotonic()
    heartbeat_s = 5.0
    while True:
        try:
            proc.wait(timeout=heartbeat_s)
            break
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - wait_start
            still_running = proc.poll() is None
            state = "still encoding/muxing" if still_running else "process ended (joining)"
            logger.info(
                "[VIDEO] FFmpeg %s — %.0fs since stdin closed, pid=%s, %sx%s, → %s",
                state,
                elapsed,
                proc.pid,
                out_width,
                out_height,
                output_path,
            )
    wait_elapsed = time.monotonic() - wait_start
    logger.info(
        "[VIDEO] FFmpeg finished with return code: %s (post-stdin wait %.1fs)",
        proc.returncode,
        wait_elapsed,
    )
    # Check if encoding was successful
    if proc.returncode != 0:
        logger.warning("[VIDEO] FFmpeg exited with non-zero code %s", proc.returncode)
        raise RuntimeError(
            f"FFmpeg encoding failed with code {proc.returncode}. "
            "If this is intermittent under multi-GPU runs, try use_gpu_video_encoding=False or a single process."
        )
    else:
        logger.info(
            "[VIDEO] Video successfully encoded to: %s "
            "(container write complete; continuing visualization pipeline if any)",
            output_path,
        )


def write_tiled_video(
    tiles: Sequence[np.ndarray | None],
    num_rows: int,
    num_cols: int,
    num_frames: int,
    tile_height: int,
    tile_width: int,
    output_path: str,
    fps: float,
    use_gpu: bool = True,
) -> None:
    """Tile ``tiles`` into a ``num_rows x num_cols`` grid and encode the result to ``output_path``.

    Convenience wrapper that chains :func:`tiled_frames_generator` and :func:`write_video_with_ffmpeg` with a
    consistent output resolution (``tile_width * num_cols`` by ``tile_height * num_rows``). Shared by callers that
    build a fixed grid of equally sized tiles (e.g. the camera-only tiled feature video and the type-only
    combined-patch video).

    Args:
        tiles: Row-major list of ``(num_frames, H, W, 3)`` tile arrays (or ``None`` for a black tile); length must
            equal ``num_rows * num_cols``.
        num_rows: Grid row count.
        num_cols: Grid column count.
        num_frames: Number of frames to emit.
        tile_height: Per-tile height in pixels (tiles are resized to this).
        tile_width: Per-tile width in pixels.
        output_path: Destination video path.
        fps: Output frame rate.
        use_gpu: Use GPU-accelerated tile resizing when available (passed to :func:`tiled_frames_generator`).
    """
    frame_generator = tiled_frames_generator(
        tiles, num_rows, num_cols, num_frames, tile_height, tile_width, use_gpu=use_gpu
    )
    write_video_with_ffmpeg(frame_generator, output_path, tile_width * num_cols, tile_height * num_rows, fps)


def get_pe_image_transform(
    image_size: int | tuple[int, int] | None = None,
    interpolation: v2.InterpolationMode = v2.InterpolationMode.BILINEAR,  # Match PE default resize behavior
    normalize: bool = True,
    include_resize: bool = True,
    inplace: bool = True,
) -> v2.Transform:
    """
    Get the image transform used in PE.

    Ref: https://github.com/facebookresearch/perception_models/blob/main/core/vision_encoder/transforms.py

    Args:
        image_size: the size of the image. Required when include_resize=True, ignored when include_resize=False.
        interpolation: the interpolation method. Only used when include_resize=True.
        normalize: whether to normalize the image.
        include_resize: whether to include resize operation. Set to False when resize is done before paired
            augmentation.
        inplace: whether Normalize mutates the input tensor in-place. Set to False when
            the input may still be referenced elsewhere (e.g. for visualization).

    Returns:
        A ``torchvision.transforms.v2.Compose`` pipeline applying resize (optional), ``ToDtype``, and optional
        ``Normalize`` to ImageNet-style ``[-1, 1]`` when ``normalize`` is ``True``.

    Raises:
        ValueError: If ``include_resize`` is ``True`` but ``image_size`` is ``None``.
    """
    transform_list: list[v2.Transform] = []

    if include_resize:
        if image_size is None:
            raise ValueError("image_size must be provided when include_resize=True")
        transform_list.append(v2.Resize(size=image_size, interpolation=interpolation))

    transform_list.append(v2.ToDtype(torch.float32, scale=True))

    if normalize:
        transform_list.append(v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=inplace))
    return v2.Compose(transform_list)


def _preprocess_camera_only_with_black_world(
    camera_video_path: str,
    num_frames: int = 8,
    transform: v2.Transform | None = None,
    sample_method: str | WorldCameraVideoPairSampleMethod = "uniform",
    paired_augmentation: Any = None,
    image_size: tuple[int, int] | None = None,
    seed: int | None = None,
    ref_fps: float | None = None,
    materialize_black_world: bool = True,
) -> WorldCameraVideoPairResult:
    """Load only the camera stream and build a black world tensor of the same shape.

    Used when the world video path is missing and :func:`preprocess_world_camera_video_pair` is called with
    ``create_black_world_if_missing=True`` (e.g. inference with ``blacken_input_world_frames``).

    Args:
        camera_video_path: Path to an existing camera video file.
        num_frames: Number of frames to sample, or ``-1`` for all frames in the file.
        transform: Optional per-stream transform (typically dtype + normalize) applied after decode.
        sample_method: Frame sampling strategy; invalid strings fall back to uniform (legacy behavior).
        paired_augmentation: Optional callable ``(world, camera) -> (world, camera)`` applied after resize when
            ``image_size`` is set.
        image_size: ``(H, W)`` resize target before ``paired_augmentation``; required if ``paired_augmentation`` is set.
        seed: RNG seed for stochastic sampling methods.
        ref_fps: Rate of the shared logical timeline the camera stream is resampled onto. ``None`` samples the camera
            frames directly (no rate conversion); a positive value no greater than the camera FPS strides the camera
            by ``camera_fps / ref_fps``, matching the camera side of the two-stream path.
        materialize_black_world: When ``True`` (default), allocate the full ``(N, C, H, W)`` black world tensor —
            required by a caller that feeds it somewhere (visualization tiling, a two-branch forward). When ``False``,
            return ``world_video=None`` without allocating it: a caller that forwards ``world_image=None`` only needs
            the world shape, which it can read from the camera stream, so the tensor would double the decoded clip's
            memory for nothing. ``paired_augmentation`` requires a world tensor, so it must be ``None`` when this is
            ``False``.

    Returns:
        :class:`WorldCameraVideoPairResult` with real ``camera_video``, identical FPS fields, matching frame index
        lists, and a black ``world_video`` (``None`` when ``materialize_black_world=False``).

    Raises:
        ValueError: If the camera path is missing, ``paired_augmentation`` is set without ``image_size``, or
            ``paired_augmentation`` is combined with ``materialize_black_world=False``.
    """
    if paired_augmentation is not None and not materialize_black_world:
        raise ValueError(
            "paired_augmentation requires a world tensor; it cannot be used with materialize_black_world=False"
        )
    if not camera_video_path or not os.path.exists(camera_video_path):
        raise ValueError(f"Invalid camera video path: '{camera_video_path}'")
    if paired_augmentation is not None and image_size is None:
        raise ValueError("image_size must be provided when paired_augmentation is used")

    camera_fps, camera_total_frames = _probe_video_metadata(camera_video_path)
    # Resample the camera stream onto a shared logical timeline at ref_fps, mirroring the camera side of the two-stream
    # path so the object-check timeline is identical whether or not a world video is present.
    if ref_fps is not None:
        if ref_fps <= 0 or ref_fps > camera_fps:
            raise ValueError(f"ref_fps must be positive and no greater than camera_fps={camera_fps}, got {ref_fps}")
        camera_every_n_frames = camera_fps / ref_fps
    else:
        camera_every_n_frames = 1.0
    effective_total_frames = int(camera_total_frames / camera_every_n_frames)
    if num_frames == -1:
        num_frames = effective_total_frames
        logical_indices = list(range(num_frames))
    else:
        num_frames, logical_indices, _ = _sample_frame_indices_single_stream(
            effective_total_frames,
            num_frames,
            sample_method,
            seed,
            invalid_fallback_uniform=True,
        )
    camera_frame_indices = [int(i * camera_every_n_frames) for i in logical_indices]

    camera_video = _decode_video_frames_to_tensor(camera_video_path, camera_frame_indices)

    # Black world: same shape as camera; use 0 so transform (Normalize 0.5,0.5) yields -1. Skipped entirely when the
    # caller forwards ``world_image=None`` and only needs the world shape (which equals the camera shape here).
    world_video: torch.Tensor | None = torch.zeros_like(camera_video) if materialize_black_world else None

    if paired_augmentation is not None:
        assert world_video is not None  # guaranteed: paired_augmentation + materialize_black_world=False raises above
        resize_size = list(image_size) if image_size is not None else None
        camera_video = v2.functional.resize(camera_video, resize_size, interpolation=v2.InterpolationMode.BILINEAR)
        world_video = v2.functional.resize(world_video, resize_size, interpolation=v2.InterpolationMode.NEAREST_EXACT)
        world_video, camera_video = paired_augmentation(world_video, camera_video)

    if transform is not None:
        if world_video is not None:
            world_video = transform(world_video)
        camera_video = transform(camera_video)

    result: WorldCameraVideoPairResult = {
        "world_video": world_video,
        "camera_video": camera_video,
        "world_fps": camera_fps,
        "camera_fps": camera_fps,
        "num_frames": num_frames,
        "camera_frame_indices": camera_frame_indices,
        "world_frame_indices": camera_frame_indices,
        "logical_frame_indices": logical_indices,
    }
    return result


def preprocess_world_camera_video_pair(
    world_video_path: str,
    camera_video_path: str,
    num_frames: int = 8,
    transform: v2.Transform | None = None,
    sample_method: str | WorldCameraVideoPairSampleMethod = "uniform",
    verbose: bool = False,
    paired_augmentation: Any = None,
    image_size: tuple[int, int] | None = None,
    seed: int | None = None,
    world_frame_interpolation_mode: str = "nearest-exact",
    create_black_world_if_missing: bool = False,
    ref_fps: float | None = None,
    materialize_black_world: bool = True,
) -> WorldCameraVideoPairResult:
    """Decode world and camera videos, sample aligned frames, optionally augment, and apply transforms.

    When streams differ in FPS, frames are aligned on a common logical timeline then mapped to each decoder's
    frame indices (see implementation). Delegates to :func:`_preprocess_camera_only_with_black_world` when
    ``create_black_world_if_missing`` is set and the world file is absent.

    Args:
        world_video_path: Path to the world (e.g. multiview) video, or empty/missing when
            ``create_black_world_if_missing`` is ``True``.
        camera_video_path: Path to the camera video (always required).
        num_frames: Target number of frames, or ``-1`` to use every effective frame on the shared timeline.
        transform: Optional transform applied independently to world and camera tensors after decode/resize.
        sample_method: :class:`WorldCameraVideoPairSampleMethod` or string value; ``MIXED`` picks another method
            at random (deterministic when ``seed`` is set).
        verbose: If ``True``, log timing and index information via the module logger.
        paired_augmentation: Optional ``(world, camera) -> (world, camera)`` applied after both are resized to
            ``image_size``.
        image_size: ``(H, W)`` for resize before ``paired_augmentation``; required when ``paired_augmentation`` is set.
        seed: RNG seed for sampling reproducibility.
        world_frame_interpolation_mode: Resize mode for world frames before paired augmentation: ``nearest-exact`` or
            ``nearest``; any other string uses nearest-exact (same as the ``else`` branch in the implementation).
        create_black_world_if_missing: If ``True``, when the world path is missing, run the camera-only black-world
            path instead of raising.
        materialize_black_world: Forwarded to the camera-only black-world path; when ``False``, ``world_video`` is
            returned as ``None`` instead of a full-resolution black tensor (see
            :func:`_preprocess_camera_only_with_black_world`). Has no effect on the two-stream path.
        ref_fps: Rate of the shared logical timeline the two streams are aligned on. ``None`` uses
            ``min(world_fps, camera_fps)``, which keeps every frame of the slower stream. Pass a lower rate to
            subsample both streams onto a coarser timeline — a caller scoring against annotations sparser than
            either video wants its annotation rate here. Must not exceed either stream's own rate.

    Returns:
        :class:`WorldCameraVideoPairResult` whose keys are documented on that type: decoded tensors, FPS values,
        final frame count ``num_frames``, and per-stream ``*_frame_indices`` used for decoding.

    Raises:
        ValueError: Invalid paths, ``paired_augmentation`` without ``image_size``, invalid ``sample_method`` when
            not using the camera-only fallback, or a ``ref_fps`` that is non-positive, exceeds either stream's
            rate, or was supplied alongside the camera-only fallback.
    """
    # Validate parameters
    if paired_augmentation is not None and image_size is None:
        raise ValueError("image_size must be provided when paired_augmentation is used")

    world_missing = not (world_video_path and world_video_path.strip() and os.path.exists(world_video_path))
    if world_missing and create_black_world_if_missing:
        # Load only camera and create black world frames (for inference with blacken_input_world_frames when TSV has no
        # world_video). ``ref_fps`` resamples the camera stream onto the shared logical timeline exactly as the
        # two-stream path does for its camera stream, so the object-check timeline is identical in both modes.
        return _preprocess_camera_only_with_black_world(
            camera_video_path=camera_video_path,
            num_frames=num_frames,
            transform=transform,
            sample_method=sample_method,
            paired_augmentation=paired_augmentation,
            image_size=image_size,
            seed=seed,
            ref_fps=ref_fps,
            materialize_black_world=materialize_black_world,
        )
    if world_missing:
        raise ValueError(f"Invalid world video path: '{world_video_path}'")

    # Debug: Check if paths are valid
    if verbose:
        logger.info("World video path: '%s' (type: %s)", world_video_path, type(world_video_path))
        logger.info("Camera video path: '%s' (type: %s)", camera_video_path, type(camera_video_path))
        logger.info(
            "World video exists: %s",
            os.path.exists(world_video_path) if world_video_path else False,
        )
        logger.info(
            "Camera video exists: %s",
            os.path.exists(camera_video_path) if camera_video_path else False,
        )

    # Validate camera path
    if not camera_video_path or not os.path.exists(camera_video_path):
        raise ValueError(f"Invalid camera video path: '{camera_video_path}'")

    world_fps, camera_fps, world_total_frames, camera_total_frames = _probe_world_camera_metadata(
        world_video_path, camera_video_path, verbose
    )

    # Calculate the reference FPS and the number of frames to sample from
    native_ref_fps = min(world_fps, camera_fps)
    if ref_fps is None:
        ref_fps = native_ref_fps
    elif ref_fps <= 0 or ref_fps > native_ref_fps:
        raise ValueError(
            f"ref_fps must be positive and no greater than min(world_fps={world_fps}, camera_fps={camera_fps})"
            f" = {native_ref_fps}, got {ref_fps}"
        )
    world_every_n_frames = world_fps / ref_fps
    camera_every_n_frames = camera_fps / ref_fps
    # Calculate the effective number of frames at the reference FPS
    world_effective_frames = world_total_frames / world_every_n_frames
    camera_effective_frames = camera_total_frames / camera_every_n_frames
    effective_total_frames = int(min(world_effective_frames, camera_effective_frames))

    if world_effective_frames != camera_effective_frames and verbose:
        logger.info(
            "World and camera video have different effective number of frames: %s != %s. "
            "Using the smaller effective number of frames.",
            world_effective_frames,
            camera_effective_frames,
        )

    if world_fps != camera_fps and verbose:
        logger.info(
            "World and camera video have different FPS: %s != %s. Sample frames at their respective FPS.",
            world_fps,
            camera_fps,
        )

    if effective_total_frames < num_frames:
        if verbose:
            logger.info(
                "Video has less effective number of frames than requested: %s < %s. Using the first %s frames.",
                effective_total_frames,
                num_frames,
                effective_total_frames,
            )
        num_frames = effective_total_frames

    # Sample logical frame indices (shared timeline), then map to each stream's FPS
    if num_frames == -1:
        num_frames = effective_total_frames
        logical_indices = list(range(num_frames))
        resolved_sample_method = _normalize_sample_method(sample_method)
    else:
        num_frames, logical_indices, resolved_sample_method = _sample_frame_indices_single_stream(
            effective_total_frames,
            num_frames,
            sample_method,
            seed,
            invalid_fallback_uniform=False,
        )
    camera_frame_indices = [int(i * camera_every_n_frames) for i in logical_indices]
    world_frame_indices = [int(i * world_every_n_frames) for i in logical_indices]
    if verbose:
        logger.info("Sampling method: %s", resolved_sample_method)
        logger.info("Sampled frames from camera video: %s", camera_frame_indices)
        logger.info("Sampled frames from world video: %s", world_frame_indices)

    # Get frames using appropriate decoder interface
    if verbose:
        logger.info("Decoding %s frames from camera video...", len(camera_frame_indices))
        logger.info("Decoding %s frames from world video...", len(world_frame_indices))

    decoding_start_time = time.time()

    if verbose:
        logger.info("Decoding world video frames...")
        world_decode_start = time.time()
    world_video = _decode_video_frames_to_tensor(world_video_path, world_frame_indices)
    if verbose:
        world_decode_time = time.time() - world_decode_start
        logger.info("Decoding camera video frames...")
        camera_decode_start = time.time()
    camera_video = _decode_video_frames_to_tensor(camera_video_path, camera_frame_indices)
    if verbose:
        camera_decode_time = time.time() - camera_decode_start
        logger.info("World video decoding: %.3fs", world_decode_time)
        logger.info("Camera video decoding: %.3fs", camera_decode_time)

    if verbose:
        total_decoding_time = time.time() - decoding_start_time
        logger.info("Total frame decoding time: %.3fs", total_decoding_time)
        logger.info("Decoded shapes - World: %s, Camera: %s", world_video.shape, camera_video.shape)

    # CRITICAL: Resize both to target size BEFORE paired augmentation so they have the same resolution
    # This ensures spatial augmentations (crop, flip) are aligned between world and camera
    #
    # CONTRACT: When paired_augmentation is provided, we MUST resize both videos to the same size before applying the
    # augmentation. The dataset will skip resize in the transform to avoid redundant resizing (transform will only do
    # ToDtype + Normalize).
    if paired_augmentation is not None:
        # Resize both videos to target size first (validated that image_size is not None)
        if verbose:
            logger.info("Resizing both videos to %s before paired augmentation...", image_size)
        # Map world_frame_interpolation_mode string to v2.InterpolationMode enum
        if world_frame_interpolation_mode == "nearest-exact":
            world_interpolation = v2.InterpolationMode.NEAREST_EXACT
        elif world_frame_interpolation_mode == "nearest":
            world_interpolation = v2.InterpolationMode.NEAREST
        else:
            # Default to nearest-exact for other modes
            world_interpolation = v2.InterpolationMode.NEAREST_EXACT
        resize_size = list(image_size) if image_size is not None else None
        # Use nearest interpolation for world frames to preserve exact colors
        world_video = v2.functional.resize(world_video, resize_size, interpolation=world_interpolation)
        # Use bilinear for camera frames (standard for natural images)
        camera_video = v2.functional.resize(camera_video, resize_size, interpolation=v2.InterpolationMode.BILINEAR)

        if verbose:
            logger.info("After resize - World: %s, Camera: %s", world_video.shape, camera_video.shape)
            logger.info("Applying paired augmentation...")
            paired_aug_start = time.time()

        world_video, camera_video = paired_augmentation(world_video, camera_video)

        if verbose:
            paired_aug_time = time.time() - paired_aug_start
            logger.info("Paired augmentation time: %.3fs", paired_aug_time)

    # Apply independent transforms (if provided)
    # Note: If paired_augmentation was applied, we already resized, but the transform still contains the essential
    # ToDtype (uint8 → float32) and Normalize (to [-1, 1]) operations
    if transform is not None:
        world_video = transform(world_video)
        camera_video = transform(camera_video)

    result: WorldCameraVideoPairResult = {
        "world_video": world_video,
        "camera_video": camera_video,
        "world_fps": world_fps,
        "camera_fps": camera_fps,
        "num_frames": num_frames,
        "camera_frame_indices": camera_frame_indices,
        "world_frame_indices": world_frame_indices,
        "logical_frame_indices": logical_indices,
    }
    return result
