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
    name = "cwip_test",
    timeout = "long",
    srcs = ["cwip_test.py"],
    data = [
        "//checks/sample_data:0d21d408_ceca_4af6_9c4b_4f6f78ee7459_1727936028200000_0_mp4",
        "//checks/sample_data:0d21d408_ceca_4af6_9c4b_4f6f78ee7459_1727936028200000_rgb_mp4",
    ],
    tags = ["gpu"],
    deps = [
        "//checks/utils/models/cwip",
        "//checks/utils/models/cwip/modules:parallel_patchformer",
        "//checks/utils/models/cwip/modules:parallel_pe",
        "//checks/utils/models/cwip/modules/configs/defaults:model",
        "//checks/utils/models/cwip/tools:inference",
        "//checks/utils/models/cwip/visualization:image_video_utils",
        "//checks/utils/models/cwip/visualization:metrics_utils",
        "//checks/utils/models/cwip/visualization:world_model_utils",
        "@detectron2//:config",
        "@pypi//numpy",
        "@pypi//omegaconf",
        "@pypi//torch",
    ],
)
