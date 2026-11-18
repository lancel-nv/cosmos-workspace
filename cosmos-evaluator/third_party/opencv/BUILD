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

load("@rules_python//python:py_library.bzl", "py_library")

# Marks third_party/opencv as a package so the vendored wheel and its overlay BUILD file are addressable as labels by
# the local_wheel module extension (see ../local_wheel.bzl and the local_wheel.wheel(...) tag in ../../MODULE.bazel).
# BUILD.opencv is the overlay applied *inside* the materialized @opencv_python_headless repo; it is not loaded here.
exports_files([
    "opencv_python_headless-4.13.0.92+noffmpeg.noipp-cp313-cp313-manylinux_2_28_x86_64.whl",
    "BUILD.opencv",
])

# build_wheel.sh runs this script with the manylinux image's interpreter; the target exists so its tests can import it.
py_library(
    name = "curate_third_party_licenses",
    srcs = ["curate_third_party_licenses.py"],
    visibility = ["//third_party/tests:__pkg__"],
)
