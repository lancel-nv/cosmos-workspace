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

# Object-level metrics over an object-check result JSON: what "present", "missing", "misaligned" and "extra"
# mean for one world-model object.

load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "result_view",
    srcs = ["result_view.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "taxonomy",
    srcs = ["taxonomy.py"],
    visibility = ["//visibility:public"],
    deps = ["//checks/utils:object_class_names"],
)

py_library(
    name = "rules",
    srcs = ["rules.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":result_view",
        "//checks/utils:tl_state",
        "@pypi//numpy",
    ],
)

py_library(
    name = "policy",
    srcs = ["policy.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":result_view",
        ":rules",
        ":taxonomy",
        "//checks/utils:lane_style",
    ],
)

py_library(
    name = "existence",
    srcs = ["existence.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":policy",
        ":result_view",
        ":taxonomy",
        "//checks/utils:object_class_names",
    ],
)

py_library(
    name = "summary",
    srcs = ["summary.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":existence",
        ":policy",
        ":result_view",
        ":taxonomy",
    ],
)

# `data` carries the stamped git SHA into the runfiles so the emitted block can name the build that produced
# it; `build_git_sha()` returns None without it rather than failing.
py_library(
    name = "augment",
    srcs = ["augment.py"],
    data = ["//utils:git_sha"],
    visibility = ["//visibility:public"],
    deps = [
        ":existence",
        ":policy",
        ":result_view",
        ":summary",
        ":taxonomy",
        "//services:utils",
    ],
)
