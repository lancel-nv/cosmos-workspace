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
    name = "class_mapping_test",
    srcs = ["class_mapping_test.py"],
    deps = [
        "//checks/utils/models/cwip/modules:class_mapping",
        "//checks/utils/models/cwip/visualization:world_model_utils",
    ],
)

py_test(
    name = "attention_test",
    srcs = ["attention_test.py"],
    deps = [
        "//checks/utils/models/cwip/modules:attention",
        "//checks/utils/models/cwip/modules:parallelism",
        "@pypi//torch",
    ],
)

py_test(
    name = "parallelism_test",
    srcs = ["parallelism_test.py"],
    deps = [
        "//checks/utils/models/cwip/modules:parallelism",
        "@pypi//torch",
    ],
)

py_test(
    name = "parallelism_extended_test",
    srcs = ["parallelism_extended_test.py"],
    deps = [
        "//checks/utils/models/cwip/modules:parallelism",
        "@pypi//torch",
    ],
)

py_test(
    name = "parallel_pe_test",
    srcs = ["parallel_pe_test.py"],
    deps = [
        "//checks/utils/models/cwip/modules:parallel_pe",
        "//checks/utils/models/cwip/modules:parallelism",
        "//checks/utils/models/cwip/modules/configs/defaults:pe_configs",
        "@perception_models//:vision_encoder",
        "@pypi//attrs",
        "@pypi//torch",
    ],
)

py_test(
    name = "parallel_patchformer_test",
    srcs = ["parallel_patchformer_test.py"],
    deps = [
        "//checks/utils/models/cwip/modules:class_mapping",
        "//checks/utils/models/cwip/modules:parallel_patchformer",
        "//checks/utils/models/cwip/modules:parallelism",
        "@pypi//torch",
    ],
)

py_test(
    name = "hf_integration_test",
    srcs = ["hf_integration_test.py"],
    deps = [
        "//checks/utils/models/cwip/modules:hf_integration",
        "//checks/utils/models/cwip/modules/configs/defaults:model",
        "@perception_models//:vision_encoder",
        "@pypi//huggingface_hub",
        "@pypi//omegaconf",
        "@pypi//torch",
        "@pypi//transformers",
    ],
)

py_test(
    name = "presence_test",
    srcs = ["presence_test.py"],
    deps = [
        "//checks/utils/models/cwip/modules:presence",
        "//checks/utils/models/cwip/visualization:world_model_utils",
        "@pypi//torch",
    ],
)
