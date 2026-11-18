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
    name = "model_test",
    srcs = ["model_test.py"],
    deps = [
        "//checks/physical_plausibility:attention",
        "//checks/physical_plausibility:features",
        "//checks/physical_plausibility:model",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_test(
    name = "processor_test",
    srcs = ["processor_test.py"],
    deps = [
        "//checks/physical_plausibility:attention",
        "//checks/physical_plausibility:model",
        "//checks/physical_plausibility:processor",
        "@pypi//torch",
    ],
)

py_test(
    name = "run_test",
    srcs = ["run_test.py"],
    deps = ["//checks/physical_plausibility:run"],
)

py_test(
    name = "attention_test",
    srcs = ["attention_test.py"],
    deps = [
        "//checks/physical_plausibility:attention",
        "//checks/physical_plausibility:evaluation",
        "//checks/physical_plausibility:metrics",
        "//checks/physical_plausibility:model",
        "//checks/physical_plausibility:tokens",
        "//checks/physical_plausibility:training",
        "@pypi//numpy",
        "@pypi//pytest",
        "@pypi//torch",
    ],
)

py_test(
    name = "notebook_test",
    srcs = ["notebook_test.py"],
    data = ["//docs/tutorials:train_physical_plausibility_checker.ipynb"],
    deps = [
        "//checks/physical_plausibility:attention",
        "//checks/physical_plausibility:evaluation",  # keep: notebook import
        "//checks/physical_plausibility:model",
        "//checks/physical_plausibility:tokens",
        "//checks/physical_plausibility:training",  # keep: notebook import
        "//utils:bazel",
        "@pypi//pytest",
        "@pypi//torch",
    ],
)
