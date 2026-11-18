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

load("@rules_python//python:defs.bzl", "py_library", "py_test")

py_library(
    name = "fixtures",
    testonly = True,
    srcs = ["fixtures.py"],
    data = [
        "dynamic_objects_results.json",
        "traffic_light_results.json",
    ],
    visibility = ["//kpi/automotive:__subpackages__"],
    deps = ["//utils:bazel"],
)

py_test(
    name = "result_view_test",
    srcs = ["result_view_test.py"],
    deps = [
        ":fixtures",
        "//kpi/automotive:result_view",
    ],
)

py_test(
    name = "taxonomy_test",
    srcs = ["taxonomy_test.py"],
    deps = [
        "//checks/utils:object_class_names",
        "//kpi/automotive:taxonomy",
    ],
)

py_test(
    name = "policy_test",
    srcs = ["policy_test.py"],
    deps = [
        "//kpi/automotive:policy",
        "//kpi/automotive:result_view",
        "//kpi/automotive:rules",
        "//kpi/automotive:taxonomy",
    ],
)

# The extension path: a rule defined in the test, not in the library, evaluated and stamped like a built-in.
py_test(
    name = "custom_rules_test",
    srcs = ["custom_rules_test.py"],
    deps = [
        ":fixtures",
        "//kpi/automotive:existence",
        "//kpi/automotive:policy",
        "//kpi/automotive:result_view",
        "//kpi/automotive:summary",
        "//kpi/automotive:taxonomy",
    ],
)

py_test(
    name = "existence_test",
    srcs = ["existence_test.py"],
    deps = [
        ":fixtures",
        "//kpi/automotive:existence",
        "//kpi/automotive:policy",
        "//kpi/automotive:result_view",
        "//kpi/automotive:rules",
        "//kpi/automotive:taxonomy",
    ],
)

py_test(
    name = "traffic_light_test",
    srcs = ["traffic_light_test.py"],
    deps = [
        ":fixtures",
        "//checks/utils:tl_state",
        "//kpi/automotive:existence",
        "//kpi/automotive:policy",
        "//kpi/automotive:result_view",
        "//kpi/automotive:summary",
        "//kpi/automotive:taxonomy",
    ],
)

py_test(
    name = "summary_test",
    srcs = ["summary_test.py"],
    deps = [
        ":fixtures",
        "//kpi/automotive:policy",
        "//kpi/automotive:result_view",
        "//kpi/automotive:summary",
    ],
)

py_test(
    name = "lane_graph_test",
    srcs = ["lane_graph_test.py"],
    deps = [
        "//kpi/automotive:augment",
        "//kpi/automotive:existence",
        "//kpi/automotive:policy",
        "//kpi/automotive:result_view",
        "//kpi/automotive:summary",
    ],
)
