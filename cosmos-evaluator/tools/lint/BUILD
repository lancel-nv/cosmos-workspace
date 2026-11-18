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

load("@aspect_rules_lint//format:defs.bzl", "format_multirun")
load("@rules_mypy//mypy:mypy.bzl", "mypy_cli")

format_multirun(
    name = "buildifier",
    jobs = 4,  # Number of parallel threads (corresponds to CPU cores) to use for formatting
    print_command = True,
    starlark = "@buildifier_prebuilt//:buildifier",
    # `third_party/*/BUILD.*` appends the vendored-archive overlays to rules_lint's Starlark file list (which covers
    # `BUILD` and `BUILD.bazel`, never `BUILD.<archive>`); it filters nothing. The shell expands it without `globstar`,
    # so `*` spans one path segment and the pattern must track the one-directory-per-archive layout, and must keep
    # matching something: an unmatched glob reaches buildifier as a literal path and fails.
    starlark_check_args = [
        "-warnings=-constant-glob,-module-docstring",
        "-lint=warn",
        "-mode=check",
        "third_party/*/BUILD.*",
    ],
    starlark_fix_args = [
        "-lint=fix",
        "-mode=fix",
        "third_party/*/BUILD.*",
    ],
    visibility = ["//:__pkg__"],
)

mypy_cli(
    name = "mypy_cli",
    mypy_requirement = "@pypi//mypy",
    python_version = "3.13",
    deps = ["@pypi//pydantic"],
)
