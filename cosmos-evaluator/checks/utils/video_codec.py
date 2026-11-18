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

"""Requested output codec and quality tier for the encode ladder in :mod:`checks.utils.video`.

Both ends of the configuration path import this module -- ``services/settings_base.py`` to declare the typed fields and
``checks/utils/video.py`` to read them back out of the environment -- so the accepted values, the defaults for an unset
variable, and the fallbacks for a value we cannot honor are all defined exactly once and cannot drift apart.

The two knobs are independent. The codec picks *which* encode ladder runs and the quality tier picks the arguments for
whichever encoder that ladder lands on, so the tier survives a hardware fallback that changes the codec.
"""

from enum import Enum
import logging
import os

logger = logging.getLogger(__name__)


class VideoCodec(str, Enum):
    """Output codec requested for re-encodes, naming a *codec* rather than one of the four encoders that implement one.

    A codec can have more than one encoder (AV1 has hardware ``av1_nvenc`` and software ``libsvtav1``) and which one
    runs depends on the host, so this is a request for an output bitstream format, not for a specific implementation.
    Every member is decodable by the vendored ffmpeg, though ``H264`` needs NVDEC on the decoding host as well.
    """

    H264 = "h264"
    AV1 = "av1"
    VP9 = "vp9"


class VideoQuality(str, Enum):
    """Quality tier requested for re-encodes, translated per encoder in ``checks.utils.video._QUALITY_ARGS``.

    Two named tiers rather than a numeric knob because the encoders do not share a scale: ``av1_nvenc`` takes ``-cq``
    over 0-51 and ``libsvtav1`` takes ``-crf`` over 0-63, so one number would mean different things on an Ada host and
    a CPU-only host -- reintroducing exactly the host-dependent divergence the decode normalization removes.

    ``HIGHEST`` requests each encoder's best available mode, which is mathematically lossless for only two of the four
    encoders (``h264_nvenc`` and ``libvpx-vp9``). It is named for the request, not for a guarantee; see the
    ``video_quality`` field on ``SettingsBase`` for the per-encoder table.
    """

    HIGHEST = "highest"
    STANDARD = "standard"


VIDEO_CODEC_ENV_VAR = "COSMOS_EVALUATOR_VIDEO_CODEC"
VIDEO_QUALITY_ENV_VAR = "COSMOS_EVALUATOR_VIDEO_QUALITY"

# Defaults when the variables are unset. Together these reproduce the pre-existing ladder and arguments exactly --
# ``h264`` starts at ``h264_nvenc`` and falls through to the AV1 ladder, and ``STANDARD`` carries the previously
# hardcoded rate control token for token -- so an existing deployment sees no change until it opts in.
DEFAULT_VIDEO_CODEC = VideoCodec.H264
DEFAULT_VIDEO_QUALITY = VideoQuality.STANDARD

# Where an unparseable *codec* lands, which is deliberately not ``DEFAULT_VIDEO_CODEC``. A value we cannot honor means
# the deployment asked for something specific and got it wrong (``h265``, ``hevc``, a typo), and quietly substituting
# the patent-encumbered H.264 path is the wrong way to resolve that ambiguity. AV1 is royalty-free and its ladder
# always terminates in ``libsvtav1``, so it is both the safe answer and the one that cannot fail to encode.
UNPARSEABLE_CODEC_FALLBACK = VideoCodec.AV1


def parse_video_codec(value: str | None) -> VideoCodec:
    """Parse a requested codec name, warning and falling back to AV1 on anything unrecognized.

    Case- and whitespace-insensitive. ``None`` or empty means "not configured" and yields ``DEFAULT_VIDEO_CODEC``
    silently; a non-empty value we do not recognize is a misconfiguration and warns.
    """
    if value is None or not value.strip():
        return DEFAULT_VIDEO_CODEC
    try:
        return VideoCodec(value.strip().lower())
    except ValueError:
        logger.warning(
            "Unrecognized %s=%r; falling back to %s. Supported codecs: %s.",
            VIDEO_CODEC_ENV_VAR,
            value,
            UNPARSEABLE_CODEC_FALLBACK.value,
            ", ".join(member.value for member in VideoCodec),
        )
        return UNPARSEABLE_CODEC_FALLBACK


def parse_video_quality(value: str | None) -> VideoQuality:
    """Parse a requested quality tier, warning and falling back to ``STANDARD`` on anything unrecognized.

    Same shape as :func:`parse_video_codec`. The fallback here is the default, because ``STANDARD`` is already the
    conservative tier -- there is no equivalent of the codec fallback's licensing consideration.
    """
    if value is None or not value.strip():
        return DEFAULT_VIDEO_QUALITY
    try:
        return VideoQuality(value.strip().lower())
    except ValueError:
        logger.warning(
            "Unrecognized %s=%r; falling back to %s. Supported tiers: %s.",
            VIDEO_QUALITY_ENV_VAR,
            value,
            DEFAULT_VIDEO_QUALITY.value,
            ", ".join(member.value for member in VideoQuality),
        )
        return DEFAULT_VIDEO_QUALITY


def resolve_video_codec() -> VideoCodec:
    """Read the requested codec from the environment.

    This and :func:`resolve_video_quality` are the sanctioned exception to the repo's "no ad-hoc ``os.environ`` reads"
    rule, and the only two places the exception applies. The processor layer may not import pydantic-settings, and
    threading a codec argument through every encode call site would be far more invasive than reading the variable
    that ``SettingsBase`` already documents. ``SettingsBase`` remains the typed surface; this is the same variable.

    Not memoized: the read is trivially cheap next to the probe encode it feeds, and memoizing would make the value
    depend on which test happened to import the module first.
    """
    return parse_video_codec(os.environ.get(VIDEO_CODEC_ENV_VAR))


def resolve_video_quality() -> VideoQuality:
    """Read the requested quality tier from the environment.

    See :func:`resolve_video_codec` on why reading ``os.environ`` here is sanctioned.
    """
    return parse_video_quality(os.environ.get(VIDEO_QUALITY_ENV_VAR))
