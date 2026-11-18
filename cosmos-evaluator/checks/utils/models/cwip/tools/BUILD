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
    name = "batch_utils",
    srcs = ["batch_utils.py"],
    visibility = [
        "//checks/objects:__pkg__",
        "//checks/objects/single_view:__pkg__",
        "//checks/utils/models/cwip/tools:__pkg__",
        "//checks/utils/models/cwip/tools/tests:__pkg__",
    ],
    deps = ["@pypi//torch"],
)

py_binary(
    name = "inference",
    srcs = ["inference.py"],
    data = ["//checks/utils/models/cwip/checkpoints:cwip_model"],
    visibility = [
        "//checks/objects:__pkg__",
        "//checks/objects/single_view:__pkg__",
        "//checks/utils/models/cwip/tests:__pkg__",
        "//checks/utils/models/cwip/tools/tests:__pkg__",
    ],
    deps = [
        ":batch_utils",
        "//checks/utils/models/cwip",
        "//checks/utils/models/cwip/modules:class_mapping",
        "//checks/utils/models/cwip/modules:parallelism",
        "//checks/utils/models/cwip/modules:presence",
        "//checks/utils/models/cwip/visualization:image_video_utils",
        "//checks/utils/models/cwip/visualization:presence_vis",
        "//checks/utils/models/cwip/visualization:world_model_utils",
        "//utils:bazel",
        "@pypi//numpy",
        "@pypi//torch",
        "@pypi//torchinfo",
        "@pypi//torchvision",
    ],
)
