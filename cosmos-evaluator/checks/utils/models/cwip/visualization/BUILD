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

# Utility libraries used by visualization modules
py_library(
    name = "image_video_utils",
    srcs = ["image_video_utils.py"],
    visibility = [
        "//checks/objects/single_view:__pkg__",
        "//checks/utils/models/cwip:__subpackages__",
    ],
    deps = [
        "//checks/utils:video",
        "@opencv_python_headless",
        "@pypi//numpy",
        "@pypi//torch",
        "@pypi//torchvision",
        "@pypi//tqdm",
    ],
)

py_library(
    name = "world_model_utils",
    srcs = ["world_model_utils.py"],
    data = ["@cosmos_drive_dreams_toolkits//:config"],
    visibility = [
        "//checks/objects/single_view:__pkg__",
        "//checks/objects/single_view/tests:__pkg__",
        "//checks/utils/models/cwip:__subpackages__",
    ],
)

py_library(
    name = "basic_vis_utils",
    srcs = ["basic_vis_utils.py"],
    visibility = [
        "//checks/utils/models/cwip:__pkg__",
        "//checks/utils/models/cwip/visualization:__pkg__",
        "//checks/utils/models/cwip/visualization/tests:__pkg__",
    ],
    deps = [
        ":image_video_utils",
        ":world_model_utils",
        "@opencv_python_headless",
        "@pypi//matplotlib",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_library(
    name = "feature_similarity_utils",
    srcs = ["feature_similarity_utils.py"],
    visibility = [
        "//checks/utils/models/cwip:__pkg__",
        "//checks/utils/models/cwip/visualization:__pkg__",
        "//checks/utils/models/cwip/visualization/tests:__pkg__",
    ],
    deps = [
        "@pypi//torch",
    ],
)

py_library(
    name = "feature_vis_utils",
    srcs = ["feature_vis_utils.py"],
    visibility = [
        "//checks/utils/models/cwip:__pkg__",
        "//checks/utils/models/cwip/visualization:__pkg__",
        "//checks/utils/models/cwip/visualization/tests:__pkg__",
    ],
    deps = [
        ":basic_vis_utils",
        ":feature_similarity_utils",
        ":image_video_utils",
        "@opencv_python_headless",
        "@pypi//einops",
        "@pypi//matplotlib",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_library(
    name = "patch_vis_utils",
    srcs = ["patch_vis_utils.py"],
    visibility = [
        "//checks/utils/models/cwip:__pkg__",
        "//checks/utils/models/cwip/modules:__pkg__",
        "//checks/utils/models/cwip/modules/tests:__pkg__",
        "//checks/utils/models/cwip/visualization:__pkg__",
        "//checks/utils/models/cwip/visualization/tests:__pkg__",
    ],
    deps = [
        ":basic_vis_utils",
        ":image_video_utils",
        ":metrics_utils",
        ":world_model_utils",
        "//checks/utils/models/cwip/modules:class_mapping",
        "@opencv_python_headless",
        "@pypi//numpy",
        "@pypi//torch",
        "@pypi//tqdm",
    ],
)

py_library(
    name = "metrics_utils",
    srcs = ["metrics_utils.py"],
    visibility = [
        "//checks/objects/single_view:__pkg__",
        "//checks/utils/models/cwip:__subpackages__",
    ],
    deps = [
        ":world_model_utils",
        "//checks/utils/models/cwip/modules:class_mapping",
        "@pypi//torch",
    ],
)

py_library(
    name = "presence_vis",
    srcs = ["presence_vis.py"],
    visibility = [
        "//checks/utils/models/cwip/tools:__pkg__",
        "//checks/utils/models/cwip/visualization/tests:__pkg__",
    ],
    deps = [
        ":image_video_utils",
        ":metrics_utils",
        ":world_model_utils",
        "//checks/utils/models/cwip/modules:class_mapping",
        "//checks/utils/models/cwip/modules:presence",
        "@opencv_python_headless",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)
