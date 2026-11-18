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

load("@rules_python//python:defs.bzl", "py_test")

# CPU-only guard pinning the shipped ffmpeg binary's codec set by exact set equality. Runs the vendored binary's
# `-encoders`/`-decoders` listings, which report compiled-in codecs without dlopening the NVENC runtime, so this needs
# no GPU. Not gpu-tagged so it runs on ordinary CPU-only machines.
py_test(
    name = "ffmpeg_encoders_test",
    srcs = ["ffmpeg_encoders_test.py"],
    data = ["//third_party/ffmpeg:ffmpeg_lgpl_tar"],
    deps = ["//utils:bazel"],
)

# The lockfile half of the "one FFmpeg" guarantee: the tarball test above pins what our own build contains, this pins
# that no pip package brings a second one.
py_test(
    name = "lockfile_codecs_test",
    srcs = ["lockfile_codecs_test.py"],
    data = ["//:requirements.txt"],
    deps = [
        "//third_party/ffmpeg:bundled_ffmpeg_audit",
        "//utils:bazel",
    ],
)
