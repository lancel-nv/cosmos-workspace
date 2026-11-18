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

load("@rules_python//python:py_test.bzl", "py_test")

py_test(
    name = "apt_pins_test",
    srcs = ["apt_pins_test.py"],
    data = [
        "//:apt_pin_facts",
        "//services/attribute_verification:packages.yaml",
        "//services/objects/single_view:packages.yaml",
        "//services/physical_plausibility:packages.yaml",
    ],
    deps = [
        "//services:apt_pins",
        "//utils:bazel",
    ],
)

py_test(
    name = "merge_dpkg_status_test",
    srcs = ["merge_dpkg_status_test.py"],
    deps = ["//services:merge_dpkg_status"],
)

py_test(
    name = "settings_base_test",
    srcs = ["settings_base_test.py"],
    deps = [
        "//checks/utils:video_codec",
        "//services:settings_base",
        "//utils:environment",
        "@pypi//pydantic",
        "@pypi//pydantic_settings",
    ],
)

py_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    deps = [
        "//services:utils",
        "//services/framework/storage_providers/utils:s3",
    ],
)
