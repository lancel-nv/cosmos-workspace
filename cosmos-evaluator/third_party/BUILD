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

load("@rules_python//python:py_library.bzl", "py_library")

# Gazelle does not index this package: the overlays under third_party/*/ describe repos that only exist once bzlmod has
# fetched them, so there is nothing here for it to resolve imports against. Targets in this package and its subpackages
# are hand-written.
# gazelle:python_extension disabled

py_library(
    name = "vendored_python",
    srcs = ["vendored_python.py"],
    visibility = ["//visibility:public"],
)
