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
    name = "cwip",
    srcs = ["cwip.py"],
    visibility = [
        "//checks/objects:__pkg__",
        "//checks/objects/single_view:__pkg__",
        "//checks/utils/models/cwip/tests:__pkg__",
        "//checks/utils/models/cwip/tools:__pkg__",
    ],
    deps = [
        "//checks/utils/models/cwip/modules:class_mapping",
        "//checks/utils/models/cwip/modules:hf_integration",
        "//checks/utils/models/cwip/modules:parallel_patchformer",
        "//checks/utils/models/cwip/modules:parallel_pe",
        "//checks/utils/models/cwip/modules:parallelism",
        "//checks/utils/models/cwip/modules/configs/defaults:model",
        "//checks/utils/models/cwip/visualization:basic_vis_utils",
        "//checks/utils/models/cwip/visualization:feature_vis_utils",
        "//checks/utils/models/cwip/visualization:image_video_utils",
        "//checks/utils/models/cwip/visualization:metrics_utils",
        "//checks/utils/models/cwip/visualization:patch_vis_utils",
        "//checks/utils/models/cwip/visualization:world_model_utils",
        "@detectron2//:config",
        "@pypi//einops",
        "@pypi//numpy",
        "@pypi//omegaconf",
        "@pypi//safetensors",
        "@pypi//torch",
    ],
)
