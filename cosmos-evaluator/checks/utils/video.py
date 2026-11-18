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

"""Video helper utilities and writer."""

from collections.abc import Callable, Generator, Sequence
from contextlib import suppress
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import select
import shutil
import subprocess
import tempfile
import time
from typing import Any, Self

import cv2
import numpy as np

from checks.utils.video_codec import (
    DEFAULT_VIDEO_QUALITY,
    VideoCodec,
    VideoQuality,
    resolve_video_codec,
    resolve_video_quality,
)
from utils.gpu import (
    NVDEC_MISSING_SIGNATURE,
    GpuUnavailableError,
    is_gpu_init_failure,
    is_transient_gpu_init_failure,
)

logger = logging.getLogger(__name__)

# Memoized results of the one-frame probe encode in ``_is_nvenc_codec_available``. Keyed by encoder *and* quality
# tier, because availability is a property of the configuration rather than of the host: an Ampere card runs
# ``h264_nvenc`` but not ``av1_nvenc`` (AV1 encode needs Ada or newer) and both are listed by ``ffmpeg -encoders``
# regardless, while on an Ada card the same ``av1_nvenc`` both opens and refuses to open depending on the tier asked
# of it. One entry per encoder would let whichever tier probed first answer for the other.
_NVENC_AVAILABLE_CACHE: dict[tuple[str, VideoQuality], bool] = {}


@dataclass(frozen=True)
class _NvdecProbe:
    """What the one-time cuvid probe in ``_probe_nvdec`` learned about this host.

    Two separate facts, because the ladder acts on them differently: ``av1_cuvid_works`` chooses between AV1's two
    decode tiers, while ``nvdec_stack_loads`` decides whether H.264 -- which has no software tier -- can be decoded
    here at all.
    """

    av1_cuvid_works: bool
    nvdec_stack_loads: bool


# Memoized result of the encode-then-decode probe in ``_probe_nvdec``.
_NVDEC_PROBE: _NvdecProbe | None = None

# Hard cap (seconds) on each blocking read from / probe of the ffmpeg child so a stuck or corrupt input cannot hang the
# caller (and, by extension, an inflight request).
_FFMPEG_DECODE_TIMEOUT_SECONDS = 30

# Wall-clock cap (seconds) for the encoder to flush and finalize the output file when a VideoWriter is released. The
# encode runs incrementally as frames are piped in, so release() only finalizes; the budget is generous to tolerate
# large buffered tails without hanging forever on a stuck child.
_FFMPEG_ENCODE_TIMEOUT_SECONDS = 120

# Memoized ffprobe results, keyed by (absolute path, size, mtime_ns) so a file replaced in place is re-probed.
# Unbounded by design: entries are one tuple each and a process only ever handles a handful of videos per request,
# whereas an LRU would have to be sized larger than the worst-case working set to be useful.
_DIMENSION_CACHE: dict[tuple[str, int, int], tuple[int, int]] = {}

# Same scheme for the codec probe. Decoder selection needs the codec on every decode, and the random-access reader
# decodes one frame per call, so leaving this uncached would put an ffprobe in front of every frame.
_CODEC_CACHE: dict[tuple[str, int, int], str] = {}


class VideoDecodeError(Exception):
    """Raised when the vendored ffmpeg binary cannot probe or decode a video."""


class GpuUnavailableDecodeError(VideoDecodeError, GpuUnavailableError):
    """Raised when ffprobe could not initialize CUDA, leaving a readable video unreadable *on this host*.

    Both parents, and each for its own reason. :class:`VideoDecodeError` so every existing decode handler keeps
    catching it unchanged. :class:`~utils.gpu.GpuUnavailableError` so a caller that reacts to an unusable GPU --
    the checker framework, which stops claiming work and asks to be replaced -- catches this the same way it catches
    an ONNX session whose CUDA provider would not start, without knowing that this one came from a video probe.

    The distinction from a plain ``VideoDecodeError`` is the remedy: that one says the input is unusable and should be
    rejected, this one says the input is fine and the machine is not, so the work belongs on a different replica (or
    on this one once its GPU is usable) rather than being failed as bad data.
    """


def _file_identity(video_path: str) -> tuple[str, int, int]:
    """Return the ffprobe cache key for ``video_path``: (absolute path, size, mtime_ns).

    Size and mtime are part of the key so a file replaced in place (same path, new content) misses the cache instead
    of returning a stale probe. The ``stat`` doubles as the existence check.

    Raises:
        VideoDecodeError: If ``video_path`` does not exist (fail fast rather than probing a missing file).
    """
    try:
        stat = os.stat(video_path)
    except OSError as exc:
        raise VideoDecodeError(f"Video file does not exist: {video_path}") from exc
    return os.path.abspath(video_path), stat.st_size, stat.st_mtime_ns


# Quality arguments per (encoder, tier): preset, tune, and rate control only. Keyed by *encoder* rather than by codec
# because the two AV1 encoders do not share a scale -- ``av1_nvenc`` takes ``-cq`` over 0-51 and ``libsvtav1`` takes
# ``-crf`` over 0-63 -- so a per-codec number would mean different things depending on the host's GPU.
#
# The two tiers differ by flag *deletion*, not by flag value: ``-tune lossless`` makes ``-rc vbr -cq N -b:v 0`` inert
# and ``-lossless 1`` makes ``-crf N`` inert, so those are dropped rather than carried with a different number. A
# surviving ``-cq 23`` next to ``-tune lossless`` would read as though a quality target were still active and invite
# tuning the wrong knob. That is also why this maps to whole argument lists instead of substituting one number into a
# shared per-encoder template.
#
# Encoder traps captured here once, rather than left to whoever sets the env var. The costly one is ``libsvtav1``'s
# ``-crf 0``, documented on its row below. ``-b:v 0`` beside ``-cq``/``-crf`` is *not* one of them despite the folklore:
# measured against ffmpeg 9.0.1, ``h264_nvenc`` and ``libvpx-vp9`` produce byte-identical output with and without it,
# because ``b:v`` already defaults to 0. It is kept because it states the constant-quality intent explicitly and costs
# nothing, not because omitting it changes anything today. VP9 genuinely does need ``-deadline good`` and ``-cpu-used``
# to avoid being unusably slow, and ``-row-mt 1`` to thread its (slower still) lossless mode.
#
# Provenance of the values: the three pre-existing STANDARD rows are the arguments this module used before the tier
# existed, carried over token for token. Everything else was measured on an A6000 against the vendored build -- the two
# lossless rows as exact round trips, and the ``libsvtav1`` row as a preset/CRF sweep (see its row).
_QUALITY_ARGS: dict[tuple[str, VideoQuality], list[str]] = {
    ("h264_nvenc", VideoQuality.HIGHEST): ["-preset", "p7", "-tune", "lossless"],
    ("h264_nvenc", VideoQuality.STANDARD): ["-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", "23", "-b:v", "0"],
    # ``av1_nvenc`` deliberately has no HIGHEST row, which is why this table is allowed to be sparse. NVENC's lossless
    # tune is an H.264/HEVC feature; Ada's AV1 encoder has no lossless mode, and asking for one does not degrade to a
    # high-quality lossy encode -- NVENC rejects the whole configuration and ffmpeg reports the misleading "No capable
    # devices found", as though the L40 it is running on were not an AV1 encoder at all. Measured on an L40 (driver
    # 570.211.01): the STANDARD row below encodes fine on the same host in the same run.
    #
    # A best-effort lossy row could go here instead, but it would make ``highest`` mean "slightly better than standard"
    # on Ada and "very nearly lossless" everywhere else, for the same request against the same codec. Leaving the row
    # out instead sends an AV1 ``highest`` request down to ``libsvtav1``, which really does have a near-lossless mode,
    # so the tier means one thing across the fleet -- at the cost of the encode running on CPU.
    ("av1_nvenc", VideoQuality.STANDARD): ["-preset", "p4", "-tune", "hq", "-rc", "vbr", "-cq", "23", "-b:v", "0"],
    # ``-crf 1``, emphatically not ``-crf 0``: ffmpeg's libsvtav1 wrapper declares ``crf`` with a default of 0 and
    # treats 0 as "unset", so ``-crf 0`` silently leaves SVT at its own internal default of 35 -- coarser than this
    # table's own STANDARD row. Verified from SVT's startup banner ("rate factor : CRF / 35" for ``-crf 0`` against
    # "CRF / 1" for ``-crf 1``) and by output size, where ``-crf 0`` produced a *smaller* file than ``-crf 30``. A
    # request for the best available mode must not quietly become the worst row in the table.
    #
    # ``-preset 2``, not ``0``. Measured at ``-crf 1`` on 30 frames of 1280x704: preset 0 reaches 66.48 dB PSNR in
    # 5.2 s and preset 2 reaches 63.99 dB in 2.5 s, against 60.63 dB at preset 8. Preset 2 therefore buys 57% of the
    # total available quality gain for 48% of preset 0's time. SVT-AV1 4.x narrowed that trade -- under 3.0.2 the same
    # sweep gave 86% of the gain for 25% of the time -- by speeding preset 0 up almost threefold, so preset 0 is worth
    # revisiting if HIGHEST-tier quality ever comes to matter more than encode time. This row is the fleet's common
    # case rather than an edge one: ``COSMOS_EVALUATOR_VIDEO_CODEC=av1`` plus ``highest`` lands here on every
    # non-Ada host.
    ("libsvtav1", VideoQuality.HIGHEST): ["-preset", "2", "-crf", "1"],
    ("libsvtav1", VideoQuality.STANDARD): ["-preset", "8", "-crf", "30"],
    ("libvpx-vp9", VideoQuality.HIGHEST): ["-lossless", "1", "-row-mt", "1"],
    ("libvpx-vp9", VideoQuality.STANDARD): ["-b:v", "0", "-crf", "31", "-deadline", "good", "-cpu-used", "2"],
}

# Encoders that need the runtime probe in ``_is_nvenc_codec_available``. The other two are software encoders compiled
# into the vendored build, so they are available wherever the binary runs and probing them would only cost time.
_HARDWARE_ENCODERS = frozenset({"h264_nvenc", "av1_nvenc"})

# The one encoder guaranteed to exist: software, royalty-free, and the terminating tier of every ladder.
_SOFTWARE_AV1_ENCODER = "libsvtav1"


def _build_encode_args(encoder: str, quality: VideoQuality) -> list[str]:
    """Compose the full encode argument list for ``encoder`` at ``quality``.

    The codec and container invariants -- the encoder name, the pixel format the whole pipeline is normalized to, and
    the MP4 flag -- belong to the encoder and apply at both tiers; only the middle section varies. Quality arguments
    sit between them so that the STANDARD tier reproduces the previously hardcoded constants token for token.
    """
    return ["-c:v", encoder, "-pix_fmt", "yuv420p", *_QUALITY_ARGS[(encoder, quality)], "-movflags", "+faststart"]


# Hardware H.264 (tier 1) at the default tier. Retained as a module constant because two call sites pin H.264
# deliberately rather than taking whatever the ladder yields: the VLM payload subsampler, whose external endpoints have
# no guaranteed AV1 support, and the CWIP visualization tests.
NVENC_H264_FFMPEG_ARGS: list[str] = _build_encode_args("h264_nvenc", DEFAULT_VIDEO_QUALITY)

# Hardware AV1 (tier 2). Reached when the host has NVENC but cannot run ``h264_nvenc`` -- and, more usefully, it is
# the tier that keeps hardware encoding available on cards where AVC encode is restricted. Needs Ada (RTX 40 / L40) or
# newer; on older NVENC hardware the probe encode fails and the ladder drops to ``libsvtav1``.
AV1_NVENC_FFMPEG_ARGS: list[str] = _build_encode_args("av1_nvenc", DEFAULT_VIDEO_QUALITY)

# Royalty-free SVT-AV1 (CPU) fallback used whenever no hardware encoder is usable (e.g. A100, CPU-only hosts). AV1 has
# no software-patent royalty exposure, so this keeps the container free of any software H.264 encoding.
SVTAV1_FFMPEG_ARGS: list[str] = _build_encode_args(_SOFTWARE_AV1_ENCODER, DEFAULT_VIDEO_QUALITY)

# Encode ladders per requested codec, in preference order. Only the AV1 ladder is guaranteed to terminate: it ends in
# ``libsvtav1``, which is software and always available. The other two can be exhausted -- ``h264`` on any host
# without NVENC, ``vp9`` only if the software encoder itself fails -- and fall back to the AV1 ladder.
#
# The ``h264`` ladder falling through to AV1 is what makes ``h264`` the default: it yields
# ``h264_nvenc -> av1_nvenc -> libsvtav1``, the "NVIDIA-only H.264 with hardware- then software-AV1 fallback" policy.
_ENCODE_LADDERS: dict[VideoCodec, tuple[str, ...]] = {
    VideoCodec.H264: ("h264_nvenc",),
    VideoCodec.AV1: ("av1_nvenc", _SOFTWARE_AV1_ENCODER),
    VideoCodec.VP9: ("libvpx-vp9",),
}

# Hardware decoders the vendored ffmpeg ships, keyed by the codec name ffprobe reports. H.264 is hardware-only by
# design: the vendored build carries no software AVC decoder (see third_party/ffmpeg/tar.sh), so on a host without
# NVDEC H.264 cannot be decoded at all -- there is nothing to fall back to. AV1 has both tiers, because ``libsvtav1``
# output must stay decodable on GPU-less hosts. VP9 has no entry here at all and decodes through the software tier
# below on every host: ffmpeg's ``vp9_cuvid`` mistags frame colorspace as RGB, which breaks the pixel-format
# normalization the tiers depend on, so it is not in the build.
_CUVID_DECODERS: dict[str, str] = {
    "h264": "h264_cuvid",
    "av1": "av1_cuvid",
}

# Software decoders, keyed the same way. A codec in neither map is unsupported: the vendored build has no decoder for
# it at all. HEVC is the notable one -- deliberately absent from both, not merely un-accelerated.
_SOFTWARE_DECODERS: dict[str, str] = {
    "av1": "libdav1d",
    "vp9": "vp9",
}

# The signatures that classify a CUDA initialization failure -- ``NVDEC_MISSING_SIGNATURE`` for an absent decode
# library, and the tables behind ``is_gpu_init_failure`` / ``is_transient_gpu_init_failure`` -- live in ``utils.gpu``.
# They are shared rather than local because ONNX Runtime and PyTorch fail for the same underlying reasons, and the
# framework that reacts to an unusable GPU should not have to import a video module to recognize one.

# Attempt budget and backoff for a probe that hit one of those transient states. The waits are 0.5s, 1s and 2s, so a
# server that is merely starting is waited out while a GPU that is genuinely gone gives up after ~3.5s rather than
# holding a job slot. Deliberately much shorter than ``_FFMPEG_DECODE_TIMEOUT_SECONDS``: these attempts fail
# immediately at ``cuInit`` rather than running to the probe timeout.
_GPU_PROBE_MAX_ATTEMPTS = 4
_GPU_PROBE_BACKOFF_BASE_SECONDS = 0.5

# Filter that normalizes decoder output to a single pixel format before any conversion to BGR/RGB. This is a
# correctness requirement of the decode ladder, not a formatting nicety: cuvid decoders emit NV12 while ``libdav1d``
# emits YUV420P, and swscale's NV12->BGR24 and YUV420P->BGR24 paths are not equivalent (measured up to 14/255 across
# 65% of bytes). Forcing a common YUV420P intermediate -- a lossless deinterleave for NV12 -- makes every tier produce
# byte-identical pixels, so a clip scores the same whether it decoded on a GPU or a CPU host.
# Exposed as a bare filter string as well, because a call site that already passes ``-vf`` has to concatenate into that
# one filter chain rather than add a second ``-vf`` (ffmpeg keeps only the last).
DECODE_PIXEL_FORMAT_FILTER = "format=yuv420p"
DECODE_PIXEL_FORMAT_ARGS: list[str] = ["-vf", DECODE_PIXEL_FORMAT_FILTER]


def _is_nvenc_codec_available(encoder: str, quality: VideoQuality = DEFAULT_VIDEO_QUALITY) -> bool:
    """Return whether ``encoder`` can encode *at this tier* on this host, memoized per encoder and tier.

    Presence in ``ffmpeg -encoders`` is not enough: ``av1_nvenc`` is compiled in unconditionally but fails at runtime
    with "No capable devices found" on pre-Ada hardware. Only a real one-frame encode answers the question.

    The probe encodes with the tier's own arguments rather than a bare ``-c:v``, because NVENC accepts or rejects a
    *configuration*, not an encoder. An encoder that opens with default settings can still refuse a particular tune or
    rate-control mode on hardware that lacks it, and the refusal looks identical to the encoder being missing. Probing
    the real arguments means such a combination is skipped during ladder selection -- falling through to a tier that
    works -- instead of aborting the encode after the caller has already committed to it.

    ``-movflags`` is left off: it is a muxer option and this probe writes to the null muxer, so it has no bearing on
    whether the encoder opens.

    An encoder with no row for this tier does not offer it, and is reported unavailable without probing. Answering
    from a probe run with no quality arguments would be worse than useless: it would report the encoder's *default*
    configuration as though it were the requested tier, which is the precise confusion this function now exists to
    prevent.
    """
    quality_args = _QUALITY_ARGS.get((encoder, quality))
    if quality_args is None:
        return False

    cache_key = (encoder, quality)
    cached = _NVENC_AVAILABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=256x256:d=0.1",
                "-c:v",
                encoder,
                "-pix_fmt",
                "yuv420p",
                *quality_args,
                "-f",
                "null",
                "-",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        available = result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        available = False

    _NVENC_AVAILABLE_CACHE[cache_key] = available
    return available


def is_nvenc_available(quality: VideoQuality = DEFAULT_VIDEO_QUALITY) -> bool:
    """Check if ``h264_nvenc`` can encode at ``quality`` via a test encode. Result is cached per tier after first call.

    Callers that go on to encode at a non-default tier should say so, since NVENC can accept one tier's configuration
    and reject another's on the same encoder.
    """
    return _is_nvenc_codec_available("h264_nvenc", quality)


def _probe_nvdec() -> _NvdecProbe:
    """Probe this host's NVDEC stack once, answering both questions the decode ladder asks of it.

    As with the encoders, listing in ``ffmpeg -decoders`` proves nothing: the cuvid decoders are compiled in
    unconditionally, ``av1_cuvid`` needs GA10x or newer, and every cuvid decoder fails on a host with no
    ``libnvcuvid`` at all. The probe encodes one black frame with ``libsvtav1`` (which always works) and tries to
    decode it with ``av1_cuvid``, which is the only honest test.

    One probe answers both questions because the two failures are distinguishable in stderr: a host missing the driver
    library says so explicitly, while a host that has it but lacks AV1 decode hardware fails some other way. That is
    what lets H.264 -- which cannot be probed directly, since producing an H.264 sample would itself require the NVENC
    hardware in question -- rely on whether the stack loads. Every NVDEC block since Kepler decodes H.264, so a
    loadable stack implies a working ``h264_cuvid``; the realistic failure is the library being absent, which is what a
    container started without ``NVIDIA_DRIVER_CAPABILITIES=video`` looks like.
    """
    global _NVDEC_PROBE
    if _NVDEC_PROBE is not None:
        return _NVDEC_PROBE

    av1_works = False
    stack_loads = False
    try:
        with tempfile.TemporaryDirectory() as probe_dir:
            sample = Path(probe_dir) / "probe.mp4"
            encode = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=256x256:d=0.1",
                    "-c:v",
                    "libsvtav1",
                    "-preset",
                    "12",
                    str(sample),
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if encode.returncode == 0 and sample.exists():
                decode = subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-c:v",
                        "av1_cuvid",
                        "-i",
                        str(sample),
                        "-frames:v",
                        "1",
                        "-f",
                        "null",
                        "-",
                    ],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                av1_works = decode.returncode == 0
                stack_loads = av1_works or NVDEC_MISSING_SIGNATURE not in decode.stderr
    except (subprocess.TimeoutExpired, OSError):
        av1_works = False
        stack_loads = False

    _NVDEC_PROBE = _NvdecProbe(av1_cuvid_works=av1_works, nvdec_stack_loads=stack_loads)
    logger.info(
        "av1_cuvid hardware decoder %s; NVDEC stack %s",
        "detected" if av1_works else "unavailable, using libdav1d",
        "present" if stack_loads else "not loadable (no libnvcuvid)",
    )
    return _NVDEC_PROBE


def _is_av1_cuvid_available() -> bool:
    """Return whether ``av1_cuvid`` can decode on this host. AV1 is the only codec with a decoder in both tiers."""
    return _probe_nvdec().av1_cuvid_works


def get_encode_quality_args(encoder: str, quality: VideoQuality) -> list[str]:
    """Return only the preset/tune/rate-control arguments for ``encoder`` at ``quality``, without the codec invariants.

    Raises ``KeyError`` for a combination the table does not offer -- ``av1_nvenc`` at HIGHEST, today's only such gap.
    That is deliberate: a caller pinning an encoder by hand has ruled out the ladder that would otherwise route around
    the gap, so silently substituting another tier would hand it output at a quality it did not ask for.

    For call sites that build their own argument list because they need a different container than the ladder's MP4 --
    a bare H.264 elementary stream being the case in point, where ``-movflags`` would be meaningless. Prefer
    :func:`get_video_encode_args` anywhere the full list will do.
    """
    return list(_QUALITY_ARGS[(encoder, quality)])


def _first_available_encoder(ladder: tuple[str, ...], quality: VideoQuality) -> str | None:
    """Return the first encoder in ``ladder`` usable at ``quality`` on this host, or ``None`` if it is exhausted.

    Two ways to be unusable. An encoder with no ``_QUALITY_ARGS`` row for this tier does not offer it at all and is
    skipped without probing -- ``av1_nvenc`` at HIGHEST is the case that exists today. Otherwise, hardware encoders
    are probed at this specific tier, since NVENC can accept an encoder's default configuration and reject the tier's.
    Software encoders are compiled into the vendored build and available wherever the binary runs, so probing them
    would only cost a one-frame encode.
    """
    for encoder in ladder:
        if (encoder, quality) not in _QUALITY_ARGS:
            logger.debug("%s offers no %s tier; trying the next encoder in the ladder", encoder, quality.value)
            continue
        if encoder not in _HARDWARE_ENCODERS or _is_nvenc_codec_available(encoder, quality):
            return encoder
    return None


def get_video_encode_args(codec: VideoCodec | None = None, quality: VideoQuality | None = None) -> list[str]:
    """Return ffmpeg video-encode args for the best available encoder of the requested codec's ladder.

    With both arguments left at ``None`` -- the case for nearly every call site -- the codec and tier come from
    ``COSMOS_EVALUATOR_VIDEO_CODEC`` and ``COSMOS_EVALUATOR_VIDEO_QUALITY``, defaulting to ``h264`` and ``standard``.
    That yields ``h264_nvenc -> av1_nvenc -> libsvtav1``: the first two are NVIDIA-licensed hardware paths and the last
    is royalty-free and always works, so there is no hard-fail path. The output codec is therefore host-dependent, and
    every tier is decodable by the vendored ffmpeg (H.264 only on a host with NVDEC -- see ``get_video_decode_args``).

    Pass ``codec`` or ``quality`` explicitly only to override that policy for a call site whose output has different
    requirements than a throwaway visualization clip -- an archived artifact, or a stream handed to an external
    consumer. Requesting a codec does not guarantee getting it: an exhausted ladder falls back to AV1.

    The tier participates in encoder selection rather than merely decorating it, so the same codec can land on
    different encoders at different tiers. Requesting AV1 at HIGHEST is the standing example: it skips ``av1_nvenc``,
    which has no lossless mode, and encodes on ``libsvtav1`` even on a host with hardware AV1.

    Video only: the vendored ffmpeg ships no audio encoder, and no caller produces audio.
    """
    requested = codec if codec is not None else resolve_video_codec()
    tier = quality if quality is not None else resolve_video_quality()

    encoder = _first_available_encoder(_ENCODE_LADDERS[requested], tier)
    if encoder is None and requested is not VideoCodec.AV1:
        logger.warning(
            "No usable %s encoder at %s quality on this host; falling back to the %s ladder",
            requested.value,
            tier.value,
            VideoCodec.AV1.value,
        )
        requested = VideoCodec.AV1
        encoder = _first_available_encoder(_ENCODE_LADDERS[requested], tier)
    if encoder is None:
        # The AV1 ladder ends in libsvtav1, which is software and needs no hardware, so reaching here means the ffmpeg
        # binary itself is broken or missing. Proceed with it anyway rather than raising: the encode is about to fail
        # with ffmpeg's own diagnostics, which say far more about the real problem than anything this could raise.
        encoder = _SOFTWARE_AV1_ENCODER
        logger.error("No encoder available at all, not even %s; the vendored ffmpeg looks broken", encoder)

    logger.info("Encoding %s at %s quality with %s", requested.value, tier.value, encoder)
    return _build_encode_args(encoder, tier)


def get_video_decode_args(input_codec: str) -> list[str]:
    """Return ffmpeg **input-side** options naming the decoder for ``input_codec``. Must be placed before ``-i``.

    Selection deliberately names a standalone cuvid decoder rather than passing ``-hwaccel cuda``. The cuvid decoder
    delivers frames to system memory, so the YUV-to-RGB conversion stays on CPU swscale in every tier; ``-hwaccel
    cuda`` would keep frames on the device and convert via NPP, whose coefficients and rounding differ. Pair this with
    ``DECODE_PIXEL_FORMAT_ARGS`` on the output side to make the tiers bit-identical.

    AV1 falls back to ``libdav1d`` when hardware decode is unavailable, and VP9 is always software.

    Takes a codec name, not an optional one: "the probe failed" is a different condition from "this codec has no
    decoder", and only the caller that ran the probe can tell them apart. See :func:`get_video_decode_input_args`.

    Raises:
        VideoDecodeError: If the vendored build has no decoder for ``input_codec`` (HEVC, MPEG-4, ...), or if the only
            decoder it has is hardware and this host cannot load the NVDEC stack (H.264 on a CPU-only host). Raising
            beats returning ``[]`` and letting ffmpeg pick: ffmpeg's own failure arrives mid-decode as a generic "no
            decoder found", whereas the two cases here are a deliberate policy decision and a fixable deployment
            mistake respectively, and each deserves to say so.
    """
    codec = input_codec.lower()
    hardware = _CUVID_DECODERS.get(codec)
    software = _SOFTWARE_DECODERS.get(codec)

    if hardware is not None:
        # AV1 is asked about its own decoder because it has a software tier to fall back to; H.264 has none, so the
        # question for it is only whether the NVDEC stack loads at all.
        usable = _is_av1_cuvid_available() if codec == "av1" else _probe_nvdec().nvdec_stack_loads
        if usable:
            return ["-c:v", hardware]

    if software is not None:
        return ["-c:v", software]

    if hardware is not None:
        raise VideoDecodeError(
            f"cannot decode {codec} on this host: {hardware} is the only decoder the vendored ffmpeg has for it and "
            "the NVIDIA decode library could not be loaded. On a GPU host this usually means the container was "
            "started without 'video' in NVIDIA_DRIVER_CAPABILITIES; on a CPU-only host it is expected, because "
            f"software {codec} decoding is deliberately absent from the build for patent reasons."
        )

    raise VideoDecodeError(
        f"unsupported codec {codec}: the vendored ffmpeg build carries no decoder for it in either tier. Supported "
        f"inputs are {', '.join(sorted(set(_CUVID_DECODERS) | set(_SOFTWARE_DECODERS)))}."
    )


def _run_ffprobe(args: Sequence[str], video_path: str, what: str) -> str | None:
    """Run one ffprobe invocation, retrying past transient GPU states, and return its stdout.

    Args:
        args: The full ffprobe argument vector, including the binary name.
        video_path: The video being probed, for log and error messages only.
        what: Short description of the property being read (e.g. ``"dimensions"``), for the same.

    Returns:
        ffprobe's stdout on success, or None if the probe failed for a reason that says nothing about the GPU --
        a missing binary, a timeout, or a file ffprobe could not read. Every None path logs first.

    Raises:
        GpuUnavailableDecodeError: If ffprobe could not initialize CUDA. Transient MPS states are retried first, so this
            is raised only once waiting has either been ruled out or exhausted.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            result = subprocess.run(
                args,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                timeout=_FFMPEG_DECODE_TIMEOUT_SECONDS,
            )
        except (subprocess.TimeoutExpired, OSError):
            logger.exception("ffprobe failed to probe %s for %s", what, video_path)
            return None

        if result.returncode == 0:
            return result.stdout

        stderr = result.stderr.strip()
        if not is_gpu_init_failure(stderr):
            logger.error("ffprobe failed for %s: %s", video_path, stderr)
            return None
        if not is_transient_gpu_init_failure(stderr) or attempt >= _GPU_PROBE_MAX_ATTEMPTS:
            raise GpuUnavailableDecodeError(
                f"ffprobe could not initialize CUDA while reading {what} of {video_path} ({attempt} attempt(s)). "
                "The video itself is not implicated: this build decodes H.264 only through h264_cuvid, so a probe "
                f"fails whenever the GPU is unusable. ffprobe reported: {stderr}"
            )

        delay = _GPU_PROBE_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1)
        logger.warning(
            "ffprobe hit a transient GPU state reading %s of %s (attempt %d/%d), retrying in %.1fs: %s",
            what,
            video_path,
            attempt,
            _GPU_PROBE_MAX_ATTEMPTS,
            delay,
            stderr,
        )
        time.sleep(delay)


def _probe_stream_entries(
    video_path: str, fields: str, what: str, extra_args: Sequence[str] = ()
) -> list[dict[str, Any]] | None:
    """Read ``-show_entries stream=<fields>`` for a video's first video stream and return the parsed stream list.

    The single path every metadata probe in this module takes, so that "bad file" and "unusable GPU" are told apart
    once rather than separately in each of them.

    Args:
        video_path: The path to the video. Callers check existence first, so a failure here is about content.
        fields: Comma-separated field list for ``-show_entries stream=...`` (e.g. ``"width,height"``).
        what: Short description of the property being read, used in logs and errors.
        extra_args: Options placed before ``-show_entries`` (only ``-count_frames`` today).

    Returns:
        The non-empty ``streams`` list from ffprobe's JSON output, or None if the probe failed, emitted output that
        would not parse, or found no video stream at all.

    Raises:
        GpuUnavailableDecodeError: If ffprobe could not initialize CUDA. See :func:`_run_ffprobe`.
    """
    stdout = _run_ffprobe(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            *extra_args,
            "-show_entries",
            f"stream={fields}",
            "-of",
            "json",
            video_path,
        ],
        video_path,
        what,
    )
    if stdout is None:
        return None

    try:
        streams = json.loads(stdout).get("streams", [])
    except ValueError:
        logger.exception("Could not parse ffprobe %s for %s", what, video_path)
        return None
    if not streams:
        logger.error("ffprobe found no video stream in %s", video_path)
        return None
    return streams


def get_video_codec(video_path: str) -> str | None:
    """Return the codec name of the first video stream via the ffprobe binary (e.g. ``"h264"``, ``"hevc"``, ``"av1"``).

    Uses the repo's vendored LGPL ffprobe (on PATH via //services:ffmpeg_layer). Used to decide whether an input can
    be stream-copied into an H.264 elementary stream or must be re-encoded, and by
    :func:`get_video_decode_input_args` to pick a decoder. Successful probes are memoized on the file's identity
    (path, size, mtime); failures are not, so a transient probe error is retried on the next call.

    Args:
        video_path: The path to the video

    Returns:
        The lowercase codec name, or None if it cannot be determined (probe failure or no video stream).

    Raises:
        VideoDecodeError: If ``video_path`` does not exist (fail fast rather than retrying against a missing file).
        GpuUnavailableDecodeError: If ffprobe could not initialize CUDA. See :func:`_run_ffprobe`.
    """
    cache_key = _file_identity(video_path)
    cached = _CODEC_CACHE.get(cache_key)
    if cached is not None:
        return cached

    streams = _probe_stream_entries(video_path, "codec_name", "codec")
    if streams is None:
        return None

    try:
        codec = str(streams[0]["codec_name"]).lower()
    except (KeyError, TypeError):
        logger.exception("Could not parse ffprobe codec for %s", video_path)
        return None

    _CODEC_CACHE[cache_key] = codec
    return codec


def get_video_decode_input_args(video_path: str) -> list[str]:
    """Probe ``video_path`` and return the ffmpeg **input-side** decoder options for it (to be placed before ``-i``).

    The path-level counterpart to :func:`get_video_decode_args`, which takes an already-known codec name. Both the
    probe and the hardware-availability check are memoized, so calling this per decode is cheap.

    A probe failure yields ``[]``, letting ffmpeg pick a decoder itself. That is deliberate, and it is the one case
    that does not raise: not knowing the codec is not the same as knowing it is unsupported, so it should degrade to
    ffmpeg's default rather than fail a decode that might well have worked. With the allow-list build there is at most
    one enabled decoder per codec anyway, so ffmpeg's default is usually the decoder this would have named. This is the
    only place that decision is made; ``get_video_decode_args`` is reserved for codecs actually identified.

    Raises:
        VideoDecodeError: If the probe succeeds and names a codec this build cannot decode. See
            :func:`get_video_decode_args`.
    """
    codec = get_video_codec(video_path)
    if codec is None:
        return []
    return get_video_decode_args(codec)


def get_video_fps(video_path: str) -> float | None:
    """Get the FPS of a video via the ffprobe binary.

    Uses the repo's vendored LGPL ffmpeg/ffprobe (on PATH via //services:ffmpeg_layer). ``r_frame_rate`` is a rational
    string (e.g. ``"30000/1001"``), matching what cv2's ``CAP_PROP_FPS`` would return.

    Args:
        video_path: The path to the video

    Returns:
        The FPS of the video or None if the video can be opened but the FPS cannot be determined

    Raises:
        VideoDecodeError: If ``video_path`` does not exist. A genuinely missing input is a hard error (fail fast in
            seconds) rather than something to retry against; distinguishing it from a probe failure on an existing file
            (which still returns None) prevents callers from looping on a file that will never appear.
        GpuUnavailableDecodeError: If ffprobe could not initialize CUDA. See :func:`_run_ffprobe`.
    """
    if not os.path.exists(video_path):
        raise VideoDecodeError(f"Video file does not exist: {video_path}")

    streams = _probe_stream_entries(video_path, "r_frame_rate", "fps")
    if streams is None:
        return None

    try:
        rate = str(streams[0]["r_frame_rate"])
        num, _, den = rate.partition("/")
        fps = float(num) / float(den) if den else float(num)
    except (ValueError, KeyError, TypeError, ZeroDivisionError):
        logger.exception("Could not parse ffprobe fps for %s", video_path)
        return None

    if fps <= 0.0:
        logger.warning("Invalid FPS (%s) for video: %s", fps, video_path)
        return None

    return fps


def get_video_frame_count(video_path: str | Path) -> int | None:
    """Return the number of frames in a video's first video stream via ffprobe.

    Tries the fast container metadata field ``nb_frames`` first (analogous to cv2's ``CAP_PROP_FRAME_COUNT``), falling
    back to an exact ``-count_frames`` decode (``nb_read_frames``) when the container omits the count. Uses the repo's
    vendored LGPL ffprobe.

    Args:
        video_path: The path to the video.

    Returns:
        The frame count, or None if the video exists but the count cannot be determined.

    Raises:
        VideoDecodeError: If ``video_path`` does not exist (fail fast rather than looping on a missing input).
        GpuUnavailableDecodeError: If ffprobe could not initialize CUDA. See :func:`_run_ffprobe`.
    """
    path = str(video_path)
    if not os.path.exists(path):
        raise VideoDecodeError(f"Video file does not exist: {path}")

    def _probe(extra_args: list[str], field: str) -> int | None:
        streams = _probe_stream_entries(path, field, "frame count", extra_args)
        if streams is None:
            return None
        try:
            value = int(streams[0][field])
        except (ValueError, KeyError, TypeError):
            return None
        return value if value > 0 else None

    count = _probe([], "nb_frames")
    if count is not None:
        return count
    # nb_frames absent (common for some mp4 muxers): fall back to an exact decode count.
    return _probe(["-count_frames"], "nb_read_frames")


def _popen_ffmpeg(
    args: Sequence[str],
    *,
    stdin: int,
    stdout: int,
    nostdin: bool,
) -> subprocess.Popen[bytes]:
    """Spawn an ffmpeg child with the shared hardening flags.

    ``-hide_banner -loglevel error`` keep output quiet; readers also pass ``nostdin=True`` (``-nostdin``) so a stuck or
    corrupt input cannot make ffmpeg block on the terminal. The writer keeps ``nostdin=False`` because it feeds frames
    through stdin (``-i -``). stderr is always captured for diagnostics.

    Args:
        args: ffmpeg arguments following the global flags.
        stdin: ``subprocess`` stdin disposition (e.g. ``subprocess.DEVNULL`` or ``subprocess.PIPE``).
        stdout: ``subprocess`` stdout disposition.
        nostdin: Whether to pass ``-nostdin`` (readers) or not (the stdin-fed writer).

    Returns:
        The spawned ``Popen``.
    """
    cmd = ["ffmpeg"]
    if nostdin:
        cmd.append("-nostdin")
    cmd += ["-hide_banner", "-loglevel", "error", *args]
    return subprocess.Popen(cmd, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE, bufsize=0)


def _iter_raw_bgr_frames(
    ffmpeg_args: Sequence[str],
    width: int,
    height: int,
    video_path: str,
    timeout: float,
) -> Generator[np.ndarray]:
    """Spawn ffmpeg with ``ffmpeg_args`` (which must emit ``rawvideo``/``bgr24`` on stdout ``-``) and yield BGR frames.

    Shared decode core for :func:`iter_video_frames` and :func:`read_video_frame`. Every read is bounded by ``timeout``
    via ``select`` and the child is always terminated and reaped.
    """
    frame_size = width * height * 3
    try:
        proc = _popen_ffmpeg(ffmpeg_args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, nostdin=True)
    except OSError as e:
        raise VideoDecodeError(f"Failed to launch ffmpeg to decode video: {video_path}") from e

    assert proc.stdout is not None
    stdout_fd = proc.stdout.fileno()
    try:
        while True:
            buf = bytearray()
            while len(buf) < frame_size:
                ready, _, _ = select.select([stdout_fd], [], [], timeout)
                if not ready:
                    proc.kill()
                    raise VideoDecodeError(f"ffmpeg decode timed out after {timeout}s for video: {video_path}")
                chunk = os.read(stdout_fd, frame_size - len(buf))
                if not chunk:
                    break  # EOF
                buf += chunk
            if len(buf) < frame_size:
                break  # End of stream (clean EOF or trailing partial frame).
            # buf is a fresh bytearray each frame, so the frombuffer view does not alias across yields.
            yield np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3))
    finally:
        with suppress(Exception):
            proc.stdout.close()
        if proc.poll() is None:
            proc.kill()
        try:
            _, stderr = proc.communicate(timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            proc.kill()
            with suppress(Exception):
                proc.communicate(timeout=timeout)
        else:
            if proc.returncode not in (0, None) and stderr:
                logger.debug(
                    "ffmpeg decode of %s exited %s: %s",
                    video_path,
                    proc.returncode,
                    stderr.decode(errors="replace").strip(),
                )


def get_video_dimensions(video_path: str) -> tuple[int, int] | None:
    """Return the (width, height) of a video's first video stream via the ffprobe binary.

    Uses the repo's vendored LGPL ffmpeg/ffprobe. Successful probes are memoized on the file's identity
    (path, size, mtime) so repeatedly decoding from the same video does not respawn ffprobe per call - every decode
    helper below probes dimensions first, and the per-frame readers made that a per-frame subprocess. Failures are
    not cached, so a transient probe error is retried on the next call.

    Args:
        video_path: The path to the video.

    Returns:
        ``(width, height)`` or ``None`` if the video exists but reports invalid/unprobeable dimensions.

    Raises:
        VideoDecodeError: If ``video_path`` does not exist (fail fast rather than looping on a missing input).
        GpuUnavailableDecodeError: If ffprobe could not initialize CUDA. Dimensions are pure container metadata, but
            ffprobe opens the stream's decoder to gather stream info and this build has no software H.264 decoder,
            so for an H.264 input the probe is only as available as the GPU. See :func:`_run_ffprobe`.
    """
    cache_key = _file_identity(video_path)
    cached = _DIMENSION_CACHE.get(cache_key)
    if cached is not None:
        return cached

    streams = _probe_stream_entries(video_path, "width,height", "dimensions")
    if streams is None:
        return None

    try:
        width = int(streams[0]["width"])
        height = int(streams[0]["height"])
    except (ValueError, KeyError, TypeError):
        logger.exception("Could not parse ffprobe dimensions for %s", video_path)
        return None

    if width <= 0 or height <= 0:
        logger.error("ffprobe returned invalid dimensions (%sx%s) for %s", width, height, video_path)
        return None

    _DIMENSION_CACHE[cache_key] = (width, height)
    return width, height


def iter_video_frames(
    video_path: str,
    max_frames: int | None = None,
    start_frame: int = 0,
    step: int = 1,
    timeout: float = _FFMPEG_DECODE_TIMEOUT_SECONDS,
) -> Generator[np.ndarray]:
    """Decode a video into BGR frames using the vendored ffmpeg binary (a cv2.VideoCapture replacement).

    Each yielded frame is a BGR ``uint8`` array of shape ``(height, width, 3)`` - the same layout cv2.VideoCapture.read
    produces - so downstream image ops (to_gray, ensure_same_size, ...) are unaffected. Decoding goes through the repo's
    LGPL ffmpeg 9.0.1 binary. The return type is a generator (not a bare Iterator) so callers can ``close()`` it to
    promptly terminate the ffmpeg child.

    The ffmpeg child is isolated from stdin (``-nostdin`` plus ``stdin=DEVNULL``) and every read is bounded by
    ``timeout`` seconds via ``select``, so a stuck or corrupt input cannot hang the caller; the child is always
    terminated and reaped. Callers that only need a subset should pass ``max_frames`` / ``start_frame`` / ``step``
    and/or stop iterating early (closing the generator kills ffmpeg).

    Args:
        video_path: The path to the video to decode.
        max_frames: If set, yield at most this many frames (after start_frame/step filtering).
        start_frame: Skip this many leading frames (0-based index of the first yielded frame).
        step: Yield every ``step``-th frame starting from ``start_frame`` (``step=1`` yields all).
        timeout: Per-read wall-clock budget in seconds.

    Yields:
        One BGR ``uint8`` frame of shape ``(height, width, 3)`` per selected video frame.

    Raises:
        VideoDecodeError: If the video dimensions cannot be probed, ffmpeg fails to start, or a read times out. The
            ``GpuUnavailableDecodeError`` subclass is raised instead when the probe failed because CUDA would not
            initialize, which blames the host rather than the file.
        ValueError: If ``step`` is not a positive integer or ``start_frame`` is negative.
    """
    if step < 1:
        raise ValueError(f"`step` must be a positive integer (got {step})")
    if start_frame < 0:
        raise ValueError(f"`start_frame` must be non-negative (got {start_frame})")

    dims = get_video_dimensions(video_path)
    if dims is None:
        raise VideoDecodeError(f"Could not determine dimensions for video: {video_path}")
    width, height = dims

    args = [*get_video_decode_input_args(video_path), "-i", video_path]
    # Only push the cap down to ffmpeg when we are decoding contiguously from the start; with start_frame/step the
    # cap applies to *yielded* frames, so it must be enforced Python-side below.
    if max_frames is not None and start_frame == 0 and step == 1:
        args += ["-frames:v", str(int(max_frames))]
    args += [*DECODE_PIXEL_FORMAT_ARGS, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]

    frames = _iter_raw_bgr_frames(args, width, height, video_path, timeout)
    try:
        yielded = 0
        for idx, frame in enumerate(frames):
            if idx < start_frame or (idx - start_frame) % step != 0:
                continue
            yield frame
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
    finally:
        frames.close()


def read_video_frame(
    video_path: str,
    frame_idx: int,
    timeout: float = _FFMPEG_DECODE_TIMEOUT_SECONDS,
) -> np.ndarray | None:
    """Read a single frame (0-based ``frame_idx``) as a BGR ``uint8`` array, or None on failure.

    Frame-accurate random access (a drop-in for ``cap.set(CAP_PROP_POS_FRAMES, idx)`` + ``cap.read()``): the ffmpeg
    ``select`` filter decodes from the start and emits exactly the requested frame, avoiding the approximate keyframe
    seek of cv2. Returns BGR to match cv2's ``read()`` output; returns None (rather than raising) so callers keep the
    cv2-style "no frame" branch.

    Args:
        video_path: The path to the video.
        frame_idx: 0-based index of the frame to read.
        timeout: Per-read wall-clock budget in seconds.

    Returns:
        The BGR ``uint8`` frame of shape ``(height, width, 3)``, or None if it cannot be read.
    """
    if frame_idx < 0:
        logger.error("read_video_frame got negative frame index %s for %s", frame_idx, video_path)
        return None

    # Both of these raise rather than return: get_video_dimensions() fails fast on a missing input, and
    # get_video_decode_input_args() raises for a codec this build cannot decode (HEVC, or H.264 on a host with no
    # NVDEC). Preserve this function's documented "return None on failure" contract by converting both into None
    # here, so callers keep their cv2-style "no frame" branch instead of growing an exception handler. Other
    # exceptions still bubble.
    try:
        dims = get_video_dimensions(video_path)
        decode_args = get_video_decode_input_args(video_path)
    except VideoDecodeError:
        logger.exception("Cannot decode frame %s from %s", frame_idx, video_path)
        return None
    if dims is None:
        return None
    width, height = dims

    args = [
        *decode_args,
        "-i",
        video_path,
        # One chain rather than a second -vf: ffmpeg keeps only the last -vf, so the pixel-format normalization has
        # to ride along with the frame selection.
        "-vf",
        f"select=eq(n\\,{int(frame_idx)}),{DECODE_PIXEL_FORMAT_FILTER}",
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    frames = _iter_raw_bgr_frames(args, width, height, video_path, timeout)
    try:
        for frame in frames:
            # Copy so the returned array does not alias the (closed) decode buffer.
            return frame.copy()
        logger.warning("Frame %s not available in %s", frame_idx, video_path)
        return None
    except VideoDecodeError:
        logger.exception("Failed to read frame %s from %s", frame_idx, video_path)
        return None
    finally:
        frames.close()


class SequentialFrameReader:
    """A forward-only cursor over a video, backed by one long-lived ffmpeg decode.

    :func:`read_video_frame` is frame-accurate but decodes from frame 0 on every call, so walking a clip in order
    costs O(n^2) decode work plus one ffprobe per frame. Every overlay caller does exactly that walk, which is what
    made a single objects job long enough to span a gateway connection recycle. This class keeps one
    :func:`iter_video_frames` generator open and advances it, so a full in-order pass spawns exactly one ffmpeg.

    Random access still works: a request for an index behind the cursor transparently restarts the decoder, which is
    no worse than what :func:`read_video_frame` does today. That path only triggers for the non-default
    ``sample_method="random"``.

    Not thread-safe - one reader per consumer. Use as a context manager, or call :meth:`close` (idempotent) to
    terminate the ffmpeg child promptly rather than waiting for garbage collection.
    """

    def __init__(self, video_path: str, timeout: float = _FFMPEG_DECODE_TIMEOUT_SECONDS) -> None:
        """Open a cursor over ``video_path``. The ffmpeg child starts lazily on the first :meth:`read`."""
        self._video_path = video_path
        self._timeout = timeout
        self._frames: Generator[np.ndarray] | None = None
        # Index of the frame the next generator yield will produce.
        self._next_idx = 0
        # Diagnostic counter, asserted in tests to pin the one-decode-per-forward-pass guarantee.
        self.decode_passes = 0

    def read(self, frame_idx: int) -> np.ndarray | None:
        """Return frame ``frame_idx`` as a BGR ``uint8`` array, or None if it cannot be read.

        Mirrors :func:`read_video_frame`'s contract (None rather than an exception on failure, including past EOF)
        so callers keep their existing "no frame" branch.
        """
        if frame_idx < 0:
            logger.error("SequentialFrameReader got negative frame index %s for %s", frame_idx, self._video_path)
            return None

        if self._frames is None or frame_idx < self._next_idx:
            self._restart()

        try:
            return self._advance_to(frame_idx)
        except VideoDecodeError:
            logger.exception("Failed to read frame %s from %s", frame_idx, self._video_path)
            self.close()
            return None

    def _advance_to(self, frame_idx: int) -> np.ndarray | None:
        """Pull frames off the open decoder until ``frame_idx`` is produced, discarding the ones skipped over."""
        assert self._frames is not None
        for frame in self._frames:
            current = self._next_idx
            self._next_idx += 1
            if current == frame_idx:
                # Copy so the returned array does not alias the decode buffer, which is reused for the next frame.
                return frame.copy()

        logger.warning("Frame %s not available in %s", frame_idx, self._video_path)
        return None

    def _restart(self) -> None:
        """Tear down any open decode and start a fresh one from frame 0."""
        self.close()
        self._frames = iter_video_frames(self._video_path, timeout=self._timeout)
        self._next_idx = 0
        self.decode_passes += 1

    def close(self) -> None:
        """Terminate the underlying ffmpeg child. Idempotent, and safe to call from a ``finally``."""
        if self._frames is not None:
            with suppress(Exception):
                self._frames.close()
            self._frames = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def convert_video_ffmpeg(
    input_video_path: Path,
    ffmpeg_args: Sequence[str],
    hide_output: bool = False,
    output_video_path: Path | None = None,
    input_args: Sequence[str] | None = None,
) -> Path | None:
    """Convert a video using ffmpeg, optionally replacing the input file.

    Output is video-only (``-an``). Nothing in the repo consumes audio, and the vendored ffmpeg ships no audio
    encoder, so an audio-bearing input would otherwise fail the whole conversion rather than being ignored.

    Args:
        input_video_path: The path to the input video
        ffmpeg_args: The arguments to pass to ffmpeg, placed after ``-i`` (output options)
        hide_output: Whether to hide the output
        output_video_path: The path to the output video
        input_args: Options placed *before* ``-i``, where ffmpeg requires input options to go. Defaults to the
            decoder selected by :func:`get_video_decode_input_args` for this input; pass an explicit list to override
            it, or ``[]`` to let ffmpeg choose.

    Returns:
        The path to the output video or None if the video cannot be converted
    """
    if not input_video_path.exists():
        error_msg = f"Input video not found: {input_video_path}"
        logger.error(error_msg)
        return None

    temp_dir = None
    try:
        if not output_video_path:
            temp_dir = tempfile.mkdtemp(dir=input_video_path.parent)
            temp_path = Path(temp_dir) / f"{input_video_path.stem}.compatible.mp4"
            replace_original = True
        else:
            temp_path = output_video_path
            replace_original = False

        resolved_input_args = (
            get_video_decode_input_args(str(input_video_path)) if input_args is None else list(input_args)
        )
        cmd = ["ffmpeg", "-y", *resolved_input_args, "-i", str(input_video_path), "-an", *ffmpeg_args, str(temp_path)]

        logger.info("Converting video with ffmpeg: %s", cmd)
        subprocess.run(cmd, capture_output=hide_output, text=True, check=True)

        if temp_path.exists():
            if replace_original:
                input_video_path.unlink()
                temp_path.rename(input_video_path)
                logger.info("Successfully converted video: %s", input_video_path)
                return input_video_path
            else:
                logger.info("Successfully converted video: %s", temp_path)
                return temp_path
        else:
            logger.error("ffmpeg conversion completed but output file not found")
            return None

    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg conversion failed: %s", e.stderr)
        return None
    except Exception as e:  # noqa: BLE001 - ffmpeg/IO failure surface is broad; log and return None
        logger.error("Error during video conversion: %s", e)
        return None
    finally:
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
                logger.info("Cleaned up temp directory: %s", temp_dir)
            except Exception as e:  # noqa: BLE001 - best-effort temp cleanup must not mask the result
                logger.error("Error during temp directory cleanup: %s", e)


class VideoWriter:
    """
    Helper for writing videos with optional per-frame preprocessing, encoding directly with the vendored LGPL ffmpeg.

    Accepts RGB frames and handles color conversion and resizing internally. Frames are piped (as raw BGR) straight to
    an ffmpeg child that encodes with the repo's video-encode args (NVENC H.264 / SVT-AV1) - so no bundled opencv
    ffmpeg backend is needed and there is no separate post-conversion pass.
    """

    def __init__(
        self,
        output_path: Path,
        fps: float | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
        preprocessor: Callable[[np.ndarray, int], np.ndarray] | None = None,
        ffmpeg_args: Sequence[str] | None = None,
    ) -> None:
        """Initialize the VideoWriter

        Args:
            output_path: The path to the output video
            fps: The FPS of the video
            frame_width: The width of the video
            frame_height: The height of the video
            preprocessor: The preprocessor to apply to the video
            ffmpeg_args: The output ffmpeg encoder arguments (defaults to the repo's H.264 args)
        """
        self.output_path = output_path
        self.fps = fps if fps is not None else 10.0
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.preprocessor = preprocessor
        self.ffmpeg_args = list(ffmpeg_args) if ffmpeg_args is not None else get_video_encode_args()

        self._proc: subprocess.Popen[bytes] | None = None
        self._frames_written: int = 0
        # ensure directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _initialize_writer(self) -> bool:
        if self.frame_width is None or self.frame_height is None:
            logger.warning("Cannot initialize video writer without frame dimensions")
            return False
        try:
            # Raw BGR frames are fed on stdin; ffmpeg encodes to the output path with the configured encoder args.
            input_args = [
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{self.frame_width}x{self.frame_height}",
                "-r",
                str(self.fps),
                "-i",
                "-",
            ]
            args = [*input_args, *self.ffmpeg_args, str(self.output_path)]
            self._proc = _popen_ffmpeg(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, nostdin=False)
            return True
        except OSError as e:
            logger.warning("Video writer init failed: %s", e)
            return False

    def open(self) -> bool:
        """Eagerly start the underlying ffmpeg encoder.

        Returns:
            True if the encoder was started successfully
        """
        return self._initialize_writer()

    def write_frame(self, frame_rgb: np.ndarray, frame_idx: int) -> bool:
        """Write a frame (RGB) to the video, applying optional preprocessing.

        Args:
            frame_rgb: The frame to write
            frame_idx: The index of the frame

        Returns:
            True if the frame was written successfully
        """
        try:
            # Initialize dimensions on first write if not provided
            if self.frame_width is None or self.frame_height is None:
                self.frame_height, self.frame_width = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])

            # Lazy start the encoder
            if self._proc is None and not self._initialize_writer():
                return False

            # Ensure size matches
            if frame_rgb.shape[:2] != (self.frame_height, self.frame_width):
                frame_rgb = cv2.resize(frame_rgb, (self.frame_width, self.frame_height))

            # Apply preprocessor (e.g., legend), if any
            if self.preprocessor is not None:
                try:
                    frame_rgb = self.preprocessor(frame_rgb, frame_idx)
                except Exception as e:  # noqa: BLE001 - user preprocessor callback can raise anything
                    logger.error("Frame preprocessing failed at %s: %s", frame_idx, e)

            # Convert to BGR and uint8
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if frame_bgr.dtype != np.uint8:
                frame_bgr = frame_bgr.astype(np.uint8)
            frame_bgr = np.ascontiguousarray(frame_bgr)

            assert self._proc is not None and self._proc.stdin is not None
            self._proc.stdin.write(frame_bgr.tobytes())
            self._frames_written += 1
            return True
        except (BrokenPipeError, OSError) as e:
            # ffmpeg died mid-stream; surface its stderr for diagnostics and stop accepting frames.
            logger.error("ffmpeg encoder pipe broke at frame %s: %s", frame_idx, e)
            return False
        except Exception as e:  # noqa: BLE001 - per-frame write failure must not abort the video
            logger.error("Error writing frame %s: %s", frame_idx, e)
            return False

    def release(self) -> None:
        """Close the encoder stdin and finalize the output file."""
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        with suppress(Exception):
            if proc.stdin is not None:
                proc.stdin.close()
        try:
            _, stderr = proc.communicate(timeout=_FFMPEG_ENCODE_TIMEOUT_SECONDS)
        except (subprocess.TimeoutExpired, OSError):
            proc.kill()
            with suppress(Exception):
                proc.communicate(timeout=_FFMPEG_ENCODE_TIMEOUT_SECONDS)
            logger.error("ffmpeg encoder timed out finalizing %s", self.output_path)
            return
        if proc.returncode not in (0, None):
            logger.error(
                "ffmpeg encoder exited %s for %s: %s",
                proc.returncode,
                self.output_path,
                stderr.decode(errors="replace").strip() if stderr else "",
            )
