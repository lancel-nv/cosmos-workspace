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

py_test(
    name = "results_test",
    srcs = ["results_test.py"],
    deps = [
        "//checks/objects:results",
        "@pypi//numpy",
    ],
)

py_test(
    name = "importance_filter_test",
    srcs = ["importance_filter_test.py"],
    deps = [
        "//checks/objects/single_view:importance_filter",
        "//checks/utils:cuboid",
        "@pypi//numpy",
    ],
)

py_test(
    name = "object_processor_base_test",
    srcs = ["object_processor_base_test.py"],
    deps = [
        "//checks/objects/single_view:object_processor_base",
        "@pypi//numpy",
    ],
)

py_test(
    name = "object_processor_dynamic_test",
    srcs = ["object_processor_dynamic_test.py"],
    deps = [
        "//checks/objects/single_view:object_processor_dynamic",
        "//checks/objects/single_view:seg_mask_cache",
        "//checks/utils:frame_rates",
        "//utils:cancellation",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_test(
    name = "object_processor_static_test",
    srcs = ["object_processor_static_test.py"],
    deps = [
        "//checks/objects/single_view:cwip_helper",
        "//checks/objects/single_view:cwip_inference_cache",
        "//checks/objects/single_view:object_processor_static",
        "//checks/utils:frame_rates",
        "//checks/utils:object_class_names",
        "//checks/utils/models/cwip/modules:class_mapping",
        "//checks/utils/models/cwip/modules:presence",
        "//checks/utils/models/cwip/visualization:world_model_utils",
        "//utils:cancellation",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_test(
    name = "cwip_inference_cache_test",
    srcs = ["cwip_inference_cache_test.py"],
    deps = [
        "//checks/objects/single_view:cwip_inference_cache",
        "@pypi//torch",
    ],
)

py_test(
    name = "run_test",
    srcs = ["run_test.py"],
    data = [
        "//checks/sample_data:0d21d408_ceca_4af6_9c4b_4f6f78ee7459_1727936028200000_0_mp4",
        "//checks/sample_data:camera_front_wide_120fov_mp4",
        "//checks/sample_data:rds_hq_data_zip",
    ],
    tags = ["gpu"],
    deps = [
        "//checks/objects/single_view:run",
        "//utils:bazel",
    ],
)

py_test(
    name = "seg_helper_test",
    srcs = ["seg_helper_test.py"],
    deps = [
        "//checks/objects/single_view:seg_helper",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_test(
    name = "seg_mask_cache_test",
    srcs = ["seg_mask_cache_test.py"],
    deps = [
        "//checks/objects/single_view:seg_mask_cache",
        "@pypi//torch",
    ],
)

py_test(
    name = "visualization_helper_test",
    srcs = ["visualization_helper_test.py"],
    deps = [
        "//checks/objects/single_view:visualization_helper",
        "//checks/utils:rds_geometry",
        "//checks/utils:scene_rasterizer",
        "@pypi//numpy",
    ],
)

py_test(
    name = "hallucination_detector_test",
    srcs = ["hallucination_detector_test.py"],
    deps = [
        "//checks/objects/single_view:hallucination_detector",
        "@pypi//numpy",
    ],
)

py_test(
    name = "hallucination_tracking_test",
    srcs = ["hallucination_tracking_test.py"],
    deps = [
        "//checks/objects/single_view:hallucination_tracking",
        "@pypi//numpy",
    ],
)

py_test(
    name = "api_test",
    srcs = ["api_test.py"],
    deps = [
        "//checks/objects/single_view:api",
        "//checks/objects/single_view:object_processor_base",
    ],
)

py_test(
    name = "cwip_helper_test",
    srcs = ["cwip_helper_test.py"],
    deps = [
        "//checks/objects/single_view:cwip_helper",
        "//checks/utils/models/cwip/modules:class_mapping",
        "//checks/utils/models/cwip/visualization:world_model_utils",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_test(
    name = "traffic_light_state_checker_test",
    srcs = ["traffic_light_state_checker_test.py"],
    deps = [
        "//checks/objects/single_view:traffic_light_state_checker",
        "//checks/utils/models/cwip/modules:class_mapping",
        "//checks/utils/models/cwip/visualization:world_model_utils",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_test(
    name = "overlap_detector_test",
    srcs = ["overlap_detector_test.py"],
    data = ["//checks:config"],
    deps = [
        "//checks/objects/single_view:overlap_detector",
        "//checks/utils:config_manager",
        "@opencv_python_headless",
        "@pypi//numpy",
    ],
)
