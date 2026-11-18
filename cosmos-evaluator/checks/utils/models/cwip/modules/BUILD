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
    name = "class_mapping",
    srcs = ["class_mapping.py"],
    visibility = [
        "//checks/objects:__pkg__",
        "//checks/objects/single_view:__pkg__",
        "//checks/objects/single_view/tests:__pkg__",
        "//checks/utils/models/cwip:__pkg__",
        "//checks/utils/models/cwip/modules:__pkg__",
        "//checks/utils/models/cwip/modules/configs/defaults/tests:__pkg__",
        "//checks/utils/models/cwip/modules/tests:__pkg__",
        "//checks/utils/models/cwip/tests:__pkg__",
        "//checks/utils/models/cwip/tools:__pkg__",
        "//checks/utils/models/cwip/visualization:__pkg__",
        "//checks/utils/models/cwip/visualization/tests:__pkg__",
    ],
    deps = ["//checks/utils/models/cwip/visualization:world_model_utils"],
)

py_library(
    name = "presence",
    srcs = ["presence.py"],
    visibility = [
        "//checks/objects/single_view:__pkg__",
        "//checks/objects/single_view/tests:__pkg__",
        "//checks/utils/models/cwip/modules/tests:__pkg__",
        "//checks/utils/models/cwip/tools:__pkg__",
        "//checks/utils/models/cwip/tools/tests:__pkg__",
        "//checks/utils/models/cwip/visualization:__pkg__",
        "//checks/utils/models/cwip/visualization/tests:__pkg__",
    ],
    deps = [
        "//checks/utils/models/cwip/visualization:world_model_utils",
        "@pypi//torch",
    ],
)

py_library(
    name = "hf_integration",
    srcs = ["hf_integration.py"],
    visibility = [
        "//checks/utils/models/cwip:__pkg__",
        "//checks/utils/models/cwip/modules/tests:__pkg__",
        "//checks/utils/models/cwip/tests:__pkg__",
        "//checks/utils/models/cwip/tools:__pkg__",
    ],
    deps = [
        "//checks/utils/models/cwip/modules/configs/defaults:model",
        "@perception_models//:vision_encoder",
        "@pypi//huggingface_hub",
        "@pypi//torch",
        "@pypi//transformers",
    ],
)

py_library(
    name = "attention",
    srcs = ["attention.py"],
    visibility = ["//checks/utils/models/cwip:__subpackages__"],
    deps = [
        ":parallelism",
        "@perception_models//:vision_encoder",
        "@pypi//einops",
        "@pypi//timm",
        "@pypi//torch",
    ],
)

py_library(
    name = "parallelism",
    srcs = ["parallelism.py"],
    visibility = ["//checks/utils/models/cwip:__subpackages__"],
    deps = [
        "@pypi//torch",
    ],
)

py_library(
    name = "parallel_pe",
    srcs = ["parallel_pe.py"],
    visibility = ["//checks/utils/models/cwip:__subpackages__"],
    deps = [
        ":attention",
        ":parallelism",
        "//checks/utils/models/cwip/modules/configs/defaults:pe_configs",
        "@perception_models//:vision_encoder",
        "@pypi//torch",
    ],
)

py_library(
    name = "parallel_patchformer",
    srcs = ["parallel_patchformer.py"],
    visibility = ["//checks/utils/models/cwip:__subpackages__"],
    deps = [
        ":attention",
        ":class_mapping",
        ":parallelism",
        "//checks/utils/models/cwip/visualization:patch_vis_utils",
        "//checks/utils/models/cwip/visualization:world_model_utils",
        "@perception_models//:vision_encoder",
        "@pypi//einops",
        "@pypi//torch",
    ],
)
