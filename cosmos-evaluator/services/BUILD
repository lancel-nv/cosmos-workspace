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

load("@rules_multirun//:defs.bzl", "multirun")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")
load(":image_layers.bzl", "gpu_core_layer")

exports_files(
    [
        "dpkg_status.bzl",
        "image_layers.bzl",
    ],
    visibility = ["//visibility:public"],
)

# Runs all checker REST API services concurrently on the local machine
multirun(
    name = "rest_apis",
    commands = [
        "//services/arbitrator:rest_api_command",
        "//services/attribute_verification:rest_api_command",
        "//services/database:rest_api_command",
        "//services/hallucination:rest_api_command",
        "//services/objects/single_view:rest_api_command",
        "//services/physical_plausibility:rest_api_command",
    ],
    jobs = 0,  # 0 means run all targets concurrently
)

# Public alias to the CUDA CUDNN base image (pulled in MODULE.bazel).
alias(
    name = "cuda_cudnn_image",
    actual = "@cuda_cudnn_image",
    visibility = ["//visibility:public"],
)

# Public alias to the platform-specific (linux/amd64) CUDA base image layer index
alias(
    name = "cuda_cudnn_image_index",
    actual = "@cuda_cudnn_image_linux_amd64",
    visibility = ["//visibility:public"],
)

# Pre-downloaded NVIDIA CUDA .deb files (installed via dpkg -i at startup). These come from NVIDIA's flat apt repo
# which rules_distroless cannot resolve.
pkg_tar(
    name = "cuda_debs_layer",
    srcs = [
        "@cuda_cupti_12_8_deb//file",
        "@cusparselt_cuda_12_deb//file",
        "@libcusparselt0_cuda_12_deb//file",
        "@libcusparselt0_dev_cuda_12_deb//file",
        "@libnccl2_deb//file",
        "@libnvshmem3_cuda_12_deb//file",
    ],
    package_dir = "/tmp/cuda-debs",
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

# Separate layer for git SHA so it doesn't invalidate the app_layer cache on every commit. The deployed
# rest_api_deployment py_binary intentionally excludes //utils:git_sha from its data; the running service reads the
# SHA from /app/git_sha.txt provided by this layer (see services.utils.get_git_sha).
pkg_tar(
    name = "git_sha_layer",
    srcs = ["//utils:git_sha"],
    package_dir = "/app",
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

# Pre-built LGPL FFmpeg 9.0.1 with NVENC/CUVID hardware acceleration and SVT-AV1. Provides both CLI tools
# (ffmpeg, ffprobe) and shared libraries (libavutil, libavcodec, libSvtAv1Enc, etc.) as an OCI image layer tar.
# See //third_party/ffmpeg/tar.sh to rebuild.
alias(
    name = "ffmpeg_layer",
    actual = "//third_party/ffmpeg:ffmpeg_lgpl_tar",
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "settings_base",
    srcs = ["settings_base.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//checks/utils:video_codec",
        "//utils:environment",
        "//utils:storage_url",
        "@pypi//pydantic",
        "@pypi//pydantic_settings",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    visibility = ["//visibility:public"],
    deps = ["//utils:bazel"],
)

py_binary(
    name = "merge_dpkg_status",
    srcs = ["merge_dpkg_status.py"],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "apt_pins",
    srcs = ["apt_pins.py"],
    visibility = ["//visibility:public"],
)

# Anchor binary whose only purpose is to pull in torch's transitive closure (torch + triton + nvidia_*_cu12), the exact
# byte-identical GPU core shared by every GPU image. Not run in production; gpu_core_deps.py statically imports torch so
# this closure lands in runfiles for gpu_core_layer to package (and so Gazelle keeps @pypi//torch here automatically).
py_binary(
    name = "gpu_core_deps",
    srcs = ["gpu_core_deps.py"],
    tags = ["manual"],
    visibility = ["//:__subpackages__"],
    deps = ["@pypi//torch"],
)

# Shared GPU core OCI layer (~6.4 GiB torch/triton/CUDA closure), built once here and reused across every GPU image
# instead of each image emitting its own copy. Archive paths are rooted at /app/rest_api_deployment.runfiles/... so the
# layer overlays into each image's runfiles.
gpu_core_layer(
    name = "gpu_core",
    binary = ":gpu_core_deps",
    owner = "1001",
    ownername = "appuser",
    package_dir = "/app",
    tags = ["manual"],
    visibility = ["//visibility:public"],
)
