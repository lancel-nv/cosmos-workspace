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

load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_test(
    name = "config_manager_test",
    srcs = ["config_manager_test.py"],
    deps = [
        "//checks/utils:config_manager",
        "@pypi//pyyaml",
    ],
)

py_test(
    name = "ego_importance_test",
    srcs = ["ego_importance_test.py"],
    deps = ["//checks/utils:ego_importance"],
)

py_test(
    name = "multistorage_test",
    srcs = ["multistorage_test.py"],
    deps = ["//checks/utils:multistorage"],
)

py_test(
    name = "utils_types_test",
    srcs = ["utils_types_test.py"],
    deps = [
        "//checks/utils:types",
        "@pypi//pydantic",
    ],
)

py_test(
    name = "frame_processing_test",
    srcs = ["frame_processing_test.py"],
    deps = [
        "//checks/utils:frame_processing",
        "//checks/utils:types",
        "@pypi//numpy",
    ],
)

py_test(
    name = "cli_common_test",
    srcs = ["cli_common_test.py"],
    deps = ["//checks/utils:cli_common"],
)

py_test(
    name = "cuboid_test",
    srcs = ["cuboid_test.py"],
    deps = [
        "//checks/utils:cuboid",
        "//checks/utils:polyline",
        "@pypi//numpy",
    ],
)

py_test(
    name = "polyline_test",
    srcs = ["polyline_test.py"],
    deps = [
        "//checks/utils:polyline",
        "//checks/utils:surface",
        "@pypi//numpy",
    ],
)

py_test(
    name = "surface_test",
    srcs = ["surface_test.py"],
    deps = [
        "//checks/utils:cuboid",
        "//checks/utils:surface",
        "@pypi//numpy",
    ],
)

py_test(
    name = "multi_polyline_test",
    srcs = ["multi_polyline_test.py"],
    deps = [
        "//checks/utils:cuboid",
        "//checks/utils:multi_polyline",
        "//checks/utils:polyline",
        "@pypi//numpy",
    ],
)

py_test(
    name = "polyline_grouping_test",
    srcs = ["polyline_grouping_test.py"],
    deps = [
        "//checks/utils:polyline",
        "//checks/utils:polyline_grouping",
        "@pypi//numpy",
    ],
)

py_test(
    name = "static_object_tracker_test",
    srcs = ["static_object_tracker_test.py"],
    deps = [
        "//checks/utils:cuboid",
        "//checks/utils:multi_polyline",
        "//checks/utils:polyline",
        "//checks/utils:static_object_tracker",
        "@pypi//numpy",
    ],
)

py_test(
    name = "scene_rasterizer_test",
    srcs = ["scene_rasterizer_test.py"],
    deps = [
        "//checks/utils:rds_geometry",
        "//checks/utils:scene_rasterizer",
        "@cosmos_drive_dreams_toolkits//:camera_base",
        "@pypi//numpy",
    ],
)

py_test(
    name = "rasterization_test",
    srcs = ["rasterization_test.py"],
    deps = [
        "//checks/utils:rasterization",
        "@pypi//numpy",
    ],
)

py_test(
    name = "coord_transforms_test",
    srcs = ["coord_transforms_test.py"],
    deps = [
        "//checks/utils:coord_transforms",
        "//checks/utils:cuboid",
        "@pypi//numpy",
    ],
)

py_test(
    name = "cv2_video_dataset_test",
    srcs = ["cv2_video_dataset_test.py"],
    deps = [
        "//checks/utils:cv2_video_dataset",
        "//checks/utils:video",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_test(
    name = "video_helper_test",
    srcs = ["video_helper_test.py"],
    deps = [
        "//checks/utils:overlay_video_helper",
        "//checks/utils:video",
        "//checks/utils:video_codec",
        "//utils:gpu",
        "@pypi//numpy",
    ],
)

py_test(
    name = "video_codec_test",
    srcs = ["video_codec_test.py"],
    deps = ["//checks/utils:video_codec"],
)

# Shared by encode_quality_test and av1_nvenc_test, which are two targets only because they need different hardware.
py_library(
    name = "encode_round_trip_test_base",
    testonly = True,
    srcs = ["encode_round_trip_test_base.py"],
    visibility = ["//checks/utils/tests:__pkg__"],
    deps = [
        "//checks/utils:video",
        "//checks/utils:video_codec",
        "@pypi//numpy",
    ],
)

# Needs a GPU: h264_nvenc is the encoder whose losslessness is being pinned, and h264_cuvid is the only H.264 decoder
# in the vendored build, so the round trip has no software path to fall back to.
py_test(
    name = "encode_quality_test",
    srcs = ["encode_quality_test.py"],
    tags = ["gpu"],
    deps = [
        ":encode_round_trip_test_base",
        "//checks/utils:video",
        "//checks/utils:video_codec",
    ],
)

# Needs an Ada-class GPU (L40/L40S), not just any GPU (av1_nvenc has no AV1 encode block on Ampere)
py_test(
    name = "av1_nvenc_test",
    srcs = ["av1_nvenc_test.py"],
    tags = ["gpu_ada"],
    deps = [
        ":encode_round_trip_test_base",
        "//checks/utils:video",
        "//checks/utils:video_codec",
    ],
)

# Requires an AV1 NVDEC (GA10x or newer)
py_test(
    name = "decode_determinism_test",
    srcs = ["decode_determinism_test.py"],
    data = ["//checks/sample_data:augmented_mp4"],
    tags = ["gpu"],
    deps = [
        "//checks/utils:video",
        "//utils:bazel",
        "@pypi//numpy",
    ],
)

py_test(
    name = "onnx_test",
    srcs = ["onnx_test.py"],
    deps = [
        "//checks/utils:onnx",
        "//utils:gpu",
        "@pypi//onnxruntime_gpu",
    ],
)

py_test(
    name = "profiler_test",
    srcs = ["profiler_test.py"],
    deps = [
        "//checks/utils:profiler",
    ],
)

py_test(
    name = "object_class_names_test",
    srcs = ["object_class_names_test.py"],
    deps = ["//checks/utils:object_class_names"],
)

py_test(
    name = "lane_style_test",
    srcs = ["lane_style_test.py"],
    deps = ["//checks/utils:lane_style"],
)

py_test(
    name = "numeric_test",
    srcs = ["numeric_test.py"],
    deps = ["//checks/utils:numeric"],
)

py_test(
    name = "rds_data_loader_test",
    srcs = ["rds_data_loader_test.py"],
    deps = [
        "//checks/utils:rds_data_loader",
        "//checks/utils:rds_geometry",
        "@pypi//numpy",
    ],
)

py_test(
    name = "segformer_test",
    srcs = ["segformer_test.py"],
    deps = [
        "//checks/utils:segformer",
        "@pypi//numpy",
        "@pypi//torch",
        "@pypi//torchvision",
    ],
)

py_test(
    name = "utils_rds_geometry_test",
    srcs = ["utils_rds_geometry_test.py"],
    deps = [
        "//checks/utils:rds_geometry",
        "@cosmos_drive_dreams_toolkits//:camera_base",
        "@pypi//numpy",
    ],
)

py_test(
    name = "video_keyframes_writer_test",
    srcs = ["video_keyframes_writer_test.py"],
    deps = [
        "//checks/utils:video",
        "@pypi//numpy",
    ],
)

py_test(
    name = "overlay_video_helper_extra_test",
    srcs = ["overlay_video_helper_extra_test.py"],
    deps = [
        "//checks/utils:overlay_video_helper",
        "@pypi//numpy",
    ],
)

py_test(
    name = "frame_rates_test",
    srcs = ["frame_rates_test.py"],
    deps = ["//checks/utils:frame_rates"],
)
