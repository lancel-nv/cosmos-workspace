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
    name = "basic_vis_utils_test",
    srcs = ["basic_vis_utils_test.py"],
    deps = [
        "//checks/utils:video",
        "//checks/utils/models/cwip/visualization:basic_vis_utils",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_test(
    name = "basic_vis_utils_gpu_test",
    srcs = ["basic_vis_utils_gpu_test.py"],
    tags = ["gpu"],
    deps = [
        "//checks/utils/models/cwip/visualization:basic_vis_utils",
        "@pypi//numpy",
    ],
)

py_test(
    name = "image_video_utils_test",
    srcs = ["image_video_utils_test.py"],
    deps = [
        "//checks/utils:video",
        "//checks/utils/models/cwip/visualization:image_video_utils",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_test(
    name = "image_video_utils_gpu_test",
    srcs = ["image_video_utils_gpu_test.py"],
    tags = ["gpu"],
    deps = [
        "//checks/utils:video",
        "//checks/utils/models/cwip/visualization:image_video_utils",
        "@pypi//numpy",
    ],
)

py_test(
    name = "feature_vis_utils_test",
    srcs = ["feature_vis_utils_test.py"],
    deps = [
        "//checks/utils/models/cwip/visualization:basic_vis_utils",
        "//checks/utils/models/cwip/visualization:feature_vis_utils",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_test(
    name = "patch_vis_utils_test",
    srcs = ["patch_vis_utils_test.py"],
    deps = [
        "//checks/utils/models/cwip/modules:class_mapping",
        "//checks/utils/models/cwip/visualization:patch_vis_utils",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_test(
    name = "metrics_utils_test",
    srcs = ["metrics_utils_test.py"],
    deps = [
        "//checks/utils/models/cwip/modules:class_mapping",
        "//checks/utils/models/cwip/visualization:metrics_utils",
        "@pypi//torch",
    ],
)

py_test(
    name = "feature_similarity_utils_test",
    srcs = ["feature_similarity_utils_test.py"],
    deps = [
        "//checks/utils/models/cwip/visualization:feature_similarity_utils",
        "@pypi//torch",
    ],
)

py_test(
    name = "world_model_utils_test",
    srcs = ["world_model_utils_test.py"],
    deps = ["//checks/utils/models/cwip/visualization:world_model_utils"],
)

py_test(
    name = "presence_vis_test",
    srcs = ["presence_vis_test.py"],
    deps = [
        "//checks/utils/models/cwip/modules:presence",
        "//checks/utils/models/cwip/visualization:presence_vis",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)
