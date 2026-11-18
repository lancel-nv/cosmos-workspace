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

"""Shared audit for pip packages that bring their own codec policy, bypassing the vendored LGPL FFmpeg.

The build in this directory is meant to be the only thing deciding which codecs we can touch. It carries no software
AVC/HEVC, which is what keeps AVC royalty exposure off the table, and it is LGPL rather than GPL, which is what the
written offer in ``SOURCE_OFFER.txt`` covers. Two distinct kinds of pip package undo that, and the reason strings below
say which is which:

- **Bundles its own FFmpeg.** The wheel ships prebuilt libav* or an ffmpeg binary, so the image contains a second codec
  set nobody reviewed. ``decord`` shipped one configured ``--enable-gpl --enable-libx264 --enable-nonfree``: a GPL
  software H.264 encoder and a software AVC decoder, in an image whose licensing posture says neither exists.
- **A second decode stack over our FFmpeg.** The wheel links the system libav* but picks its own decoder inside it, so
  ``get_video_decode_args`` stops being the thing that chooses. ``torchcodec`` did this, preferring a CUDA-capable AV1
  decoder over ``libdav1d`` and producing host-dependent pixels.

This catches both at pip-resolution time, long before anything looks at a built image. It lives in the tree that owns
the vendored build so that the codec policy and the list of packages that would subvert it stay together: any guard
that needs the list imports it from here rather than keeping a copy, which would go stale unnoticed.

Callers assert **set equality** against the packages they expect to find rather than mere absence, so that a lockfile
with a known and accepted entry still fails when a new one appears alongside it.
"""

import re

# Packages that bring a codec implementation or codec-selection policy of their own, mapped to the reason. Keys are
# canonical (PEP 503) distribution names: lowercase, with runs of -_. collapsed to a single hyphen.
FOREIGN_CODEC_PACKAGES: dict[str, str] = {
    "av": "bundles its own FFmpeg: PyAV wheels ship prebuilt libav* shared libraries",
    "decord": "bundles its own FFmpeg, configured --enable-gpl --enable-libx264 --enable-nonfree",
    "eva-decord": "bundles its own FFmpeg: decord fork, same GPL/nonfree build",
    "imageio-ffmpeg": "bundles its own FFmpeg: fetches/ships an ffmpeg binary",
    "opencv-contrib-python": "bundles its own FFmpeg: stock wheels are built WITH_FFMPEG=ON",
    "opencv-python": "bundles its own FFmpeg: stock wheels are built WITH_FFMPEG=ON (we build ours WITH_FFMPEG=OFF)",
    "torchcodec": "second decode stack: selects its own decoder inside libavcodec instead of get_video_decode_args",
}

# A requirements.txt line pinning a distribution: "name==version", optionally with extras. Hash lines, comments, flags
# and continuations do not match.
#
# Keep the trailing `\s*` inside the optional group. Outside it, the two `\s*` sit adjacent whenever the group matches
# empty, and the engine then backtracks over every way of splitting a whitespace run between them.
_PIN = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)\s*(?:\[[^\]]*\]\s*)?==", re.MULTILINE)


def canonicalize(name: str) -> str:
    """Return the PEP 503 canonical form of a distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def pinned_distributions(requirements_text: str) -> set[str]:
    """Return the canonical names of every distribution pinned with ``==`` in a compiled requirements file."""
    return {canonicalize(match.group("name")) for match in _PIN.finditer(requirements_text)}


def find_foreign_codec_pins(requirements_text: str) -> dict[str, str]:
    """Return the :data:`FOREIGN_CODEC_PACKAGES` entries pinned in ``requirements_text``, mapped to their reasons."""
    pinned = pinned_distributions(requirements_text)
    return {name: reason for name, reason in FOREIGN_CODEC_PACKAGES.items() if name in pinned}
