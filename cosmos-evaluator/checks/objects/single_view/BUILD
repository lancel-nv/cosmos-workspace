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

load("@rules_python//python:defs.bzl", "py_binary", "py_library")

py_library(
    name = "importance_filter",
    srcs = ["importance_filter.py"],
    visibility = ["//checks/objects/single_view/tests:__pkg__"],
    deps = [
        "//checks/utils:coord_transforms",
        "//checks/utils:ego_importance",
        "@pypi//numpy",
    ],
)

py_library(
    name = "hallucination_detector",
    srcs = ["hallucination_detector.py"],
    visibility = ["//checks/objects/single_view/tests:__pkg__"],
    deps = [
        "//checks/utils:ground_plane",
        "//checks/utils:object_class_names",
        "//checks/utils:rds_data_loader",
        "//checks/utils:scene_rasterizer",
        "@cosmos_drive_dreams_toolkits//:camera_base",
        "@opencv_python_headless",
        "@pypi//numpy",
    ],
)

py_library(
    name = "hallucination_tracking",
    srcs = ["hallucination_tracking.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//checks/utils:polyline_grouping",
        "@pypi//numpy",
    ],
)

py_library(
    name = "overlap_detector",
    srcs = ["overlap_detector.py"],
    visibility = ["//checks/objects/single_view/tests:__pkg__"],
    deps = [
        ":importance_filter",
        "//checks/utils:rds_data_loader",
        "//checks/utils:scene_rasterizer",
        "@cosmos_drive_dreams_toolkits//:camera_base",
        "@opencv_python_headless",
        "@pypi//numpy",
    ],
)

py_library(
    name = "cwip_helper",
    srcs = ["cwip_helper.py"],
    visibility = ["//checks/objects/single_view/tests:__pkg__"],
    deps = [
        "//checks/utils/models/cwip",
        "//checks/utils/models/cwip/modules:class_mapping",
        "//checks/utils/models/cwip/visualization:world_model_utils",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_library(
    name = "traffic_light_state_checker",
    srcs = ["traffic_light_state_checker.py"],
    visibility = ["//checks/objects/single_view/tests:__pkg__"],
    deps = [
        ":cwip_helper",
        "//checks/utils:rds_data_loader",
        "//checks/utils:scene_rasterizer",
        "//checks/utils/models/cwip/modules:class_mapping",
        "//checks/utils/models/cwip/visualization:world_model_utils",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_library(
    name = "seg_helper",
    srcs = ["seg_helper.py"],
    visibility = ["//checks/objects/single_view/tests:__pkg__"],
    deps = [
        "//checks/utils:segformer",
        "@pypi//numpy",
        "@pypi//onnxruntime_gpu",
        "@pypi//torch",
    ],
)

py_library(
    name = "seg_mask_cache",
    srcs = ["seg_mask_cache.py"],
    visibility = ["//checks/objects/single_view/tests:__pkg__"],
    deps = [
        "@pypi//torch",
    ],
)

py_library(
    name = "cwip_inference_cache",
    srcs = ["cwip_inference_cache.py"],
    visibility = ["//checks/objects/single_view/tests:__pkg__"],
    deps = [
        "@pypi//torch",
    ],
)

py_library(
    name = "object_processor_base",
    srcs = ["object_processor_base.py"],
    data = ["//checks:config"],
    visibility = ["//visibility:public"],
    deps = [
        ":importance_filter",
        ":visualization_helper",
        "//checks/objects:results",
        "//checks/utils:config_manager",
        "//checks/utils:frame_rates",
        "//checks/utils:object_class_names",
        "//checks/utils:onnx",
        "//checks/utils:profiler",
        "//checks/utils:rds_data_loader",
        "//checks/utils:scene_rasterizer",
        "//checks/utils:segformer",
        "@cosmos_drive_dreams_toolkits//:camera_base",
        "@pypi//numpy",
    ],
)

py_library(
    name = "object_processor_dynamic",
    srcs = ["object_processor_dynamic.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":hallucination_detector",
        ":hallucination_tracking",
        ":object_processor_base",
        ":overlap_detector",
        ":seg_helper",
        ":seg_mask_cache",
        "//checks/utils:coord_transforms",
        "//checks/utils:frame_rates",
        "//checks/utils:rds_data_loader",
        "//checks/utils:scene_rasterizer",
        "//utils:cancellation",
        "@cosmos_drive_dreams_toolkits//:camera_base",
        "@pypi//numpy",
        "@pypi//onnxruntime_gpu",
    ],
)

py_library(
    name = "object_processor_static",
    srcs = ["object_processor_static.py"],
    data = [
        "//checks:config",
        "//checks/utils/models/cwip/checkpoints:cwip_model",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":cwip_helper",
        ":cwip_inference_cache",
        ":hallucination_detector",
        ":hallucination_tracking",
        ":object_processor_base",
        ":overlap_detector",
        ":traffic_light_state_checker",
        "//checks/utils:frame_rates",
        "//checks/utils:lane_style",
        "//checks/utils:object_class_names",
        "//checks/utils:onnx",
        "//checks/utils:rds_data_loader",
        "//checks/utils:scene_rasterizer",
        "//checks/utils:video",
        "//checks/utils/models/cwip/modules:class_mapping",
        "//checks/utils/models/cwip/modules:presence",
        "//checks/utils/models/cwip/tools:batch_utils",
        "//checks/utils/models/cwip/visualization:image_video_utils",
        "//checks/utils/models/cwip/visualization:metrics_utils",
        "//checks/utils/models/cwip/visualization:world_model_utils",
        "//utils:bazel",
        "//utils:cancellation",
        "@cosmos_drive_dreams_toolkits//:camera_base",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_library(
    name = "api",
    srcs = ["api.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":object_processor_base",
        ":object_processor_dynamic",
        ":object_processor_static",
        "//checks/utils:config_manager",
        "//checks/utils:onnx",
    ],
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    data = ["//checks:config"],
    visibility = ["//checks/objects/single_view/tests:__pkg__"],
    deps = [
        ":api",
        "//checks/utils:cli_common",
        "//checks/utils:config_manager",
    ],
)

py_library(
    name = "visualization_helper",
    srcs = ["visualization_helper.py"],
    visibility = ["//checks/objects/single_view/tests:__pkg__"],
    deps = [
        "//checks/utils:overlay_video_helper",
        "//checks/utils:rds_data_loader",
        "//checks/utils:scene_rasterizer",
        "//checks/utils:video",
        "@cosmos_drive_dreams_toolkits//:camera_base",
        "@opencv_python_headless",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)
