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

load("@rules_python//python:defs.bzl", "py_binary")

# The CLI front end only; every determination lives in //kpi/automotive:augment, which carries the git SHA
# stamp with it.
py_binary(
    name = "object_metrics",
    srcs = ["object_metrics.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//kpi/automotive:augment",
        "//kpi/automotive:policy",
        "//kpi/automotive:result_view",
    ],
)
