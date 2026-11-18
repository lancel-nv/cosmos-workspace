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

py_library(
    name = "config_manager",
    srcs = ["config_manager.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//utils:bazel",
        "@pypi//pyyaml",
    ],
)

py_library(
    name = "ground_plane",
    srcs = ["ground_plane.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@cosmos_drive_dreams_toolkits//:camera_base",
        "@pypi//numpy",
    ],
)

py_library(
    name = "multistorage",
    srcs = ["multistorage.py"],
    visibility = ["//visibility:public"],
    deps = ["@pypi//multi_storage_client"],
)

py_library(
    name = "types",
    srcs = ["types.py"],
    visibility = ["//visibility:public"],
    deps = ["@pypi//pydantic"],
)

py_library(
    name = "frame_processing",
    srcs = ["frame_processing.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":types",
        "@opencv_python_headless",
        "@pypi//numpy",
    ],
)

py_library(
    name = "coord_transforms",
    srcs = ["coord_transforms.py"],
    visibility = ["//visibility:public"],
    deps = ["@pypi//numpy"],
)

py_library(
    name = "ego_importance",
    srcs = ["ego_importance.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "object_class_names",
    srcs = ["object_class_names.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "numeric",
    srcs = ["numeric.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "lane_style",
    srcs = ["lane_style.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "tl_state",
    srcs = ["tl_state.py"],
    visibility = ["//visibility:public"],
    deps = [":numeric"],
)

py_library(
    name = "profiler",
    srcs = ["profiler.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "cli_common",
    srcs = ["cli_common.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "onnx",
    srcs = ["onnx.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//utils:gpu",
        "@pypi//onnxruntime_gpu",
        "@pypi//torch",
    ],
)

py_library(
    name = "video",
    srcs = ["video.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":video_codec",
        "//utils:gpu",
        "@opencv_python_headless",
        "@pypi//numpy",
    ],
)

py_library(
    name = "overlay_video_helper",
    srcs = ["overlay_video_helper.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":video",
        "@opencv_python_headless",
        "@pypi//numpy",
    ],
)

py_library(
    name = "cv2_video_dataset",
    srcs = ["cv2_video_dataset.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":video",
        "@opencv_python_headless",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_library(
    name = "rasterization",
    srcs = ["rasterization.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":coord_transforms",
        "@opencv_python_headless",
        "@pypi//numba",
        "@pypi//numpy",
    ],
)

py_library(
    name = "rds_geometry",
    srcs = ["rds_geometry.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@cosmos_drive_dreams_toolkits//:camera_base",
        "@pypi//numpy",
    ],
)

py_library(
    name = "scene_rasterizer",
    srcs = ["scene_rasterizer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":rds_geometry",
        "@cosmos_drive_dreams_toolkits//:camera_base",
        "@pypi//numpy",
    ],
)

py_library(
    name = "static_object_tracker",
    srcs = ["static_object_tracker.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":multi_polyline",
        ":polyline",
        ":polyline_grouping",
        ":rds_geometry",
        "@pypi//numpy",
        "@pypi//scipy",
    ],
)

py_library(
    name = "cuboid",
    srcs = ["cuboid.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":coord_transforms",
        ":rasterization",
        ":rds_geometry",
        "@cosmos_drive_dreams_toolkits//:camera_base",
        "@pypi//numpy",
    ],
)

py_library(
    name = "polyline",
    srcs = ["polyline.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":rasterization",
        ":rds_geometry",
        "@cosmos_drive_dreams_toolkits//:camera_base",
        "@pypi//numpy",
    ],
)

py_library(
    name = "multi_polyline",
    srcs = ["multi_polyline.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":polyline",
        ":rds_geometry",
        "@cosmos_drive_dreams_toolkits//:camera_base",
        "@pypi//numpy",
    ],
)

py_library(
    name = "polyline_grouping",
    srcs = ["polyline_grouping.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":polyline",
        "@opencv_python_headless",
        "@pypi//numpy",
    ],
)

py_library(
    name = "surface",
    srcs = ["surface.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":rasterization",
        ":rds_geometry",
        "@cosmos_drive_dreams_toolkits//:camera_base",
        "@pypi//numpy",
        "@pypi//shapely",
    ],
)

py_library(
    name = "rds_data_loader",
    srcs = ["rds_data_loader.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuboid",
        ":frame_rates",
        ":object_class_names",
        ":polyline",
        ":rds_geometry",
        ":static_object_tracker",
        ":surface",
        "@cosmos_drive_dreams_toolkits//:camera_ftheta",
        "@pypi//numpy",
        "@pypi//webdataset",
    ],
)

py_library(
    name = "segformer",
    srcs = ["segformer.py"],
    data = ["//checks/utils/models/segformer:cityformer_onnx"],
    visibility = ["//visibility:public"],
    deps = [
        ":cv2_video_dataset",
        ":onnx",
        "//utils:bazel",
        "@pypi//numpy",
        "@pypi//onnxruntime_gpu",
        "@pypi//torch",
        "@pypi//torchvision",
    ],
)

py_library(
    name = "video_codec",
    srcs = ["video_codec.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "frame_rates",
    srcs = ["frame_rates.py"],
    visibility = ["//visibility:public"],
)
