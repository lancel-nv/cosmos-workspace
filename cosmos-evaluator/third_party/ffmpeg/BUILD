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

load("@rules_python//python:defs.bzl", "py_library")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

sh_binary(
    name = "tar",
    srcs = ["tar.sh"],
    tags = ["manual"],
)

py_library(
    name = "bundled_ffmpeg_audit",
    srcs = ["bundled_ffmpeg_audit.py"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "ffmpeg_lgpl_tar",
    srcs = ["ffmpeg-9.0.1-lgpl-nvenc-svtav1-linux-amd64.tar.gz"],
    visibility = ["//visibility:public"],
)

# LGPL-2.1 written offer for FFmpeg corresponding source, shipped into each checker image's license layer alongside
# THIRD-PARTY-NOTICES.txt so the offer travels with the binary distribution.
filegroup(
    name = "source_offer",
    srcs = ["SOURCE_OFFER.txt"],
    visibility = ["//visibility:public"],
)
