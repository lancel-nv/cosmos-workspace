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
    name = "bazel_test",
    srcs = ["bazel_test.py"],
    deps = ["//utils:bazel"],
)

py_test(
    name = "container_images_test",
    srcs = ["container_images_test.py"],
    deps = [
        "//utils:container_images",
        "@pypi//testcontainers",
    ],
)

py_test(
    name = "cancellation_test",
    srcs = ["cancellation_test.py"],
    deps = [
        "//utils:cancellation",
        "@pypi//pytest",
        "@pypi//pytest_asyncio",  # keep: no static import available for pytest asyncio
    ],
)

py_test(
    name = "utils_s3_test",
    srcs = ["utils_s3_test.py"],
    deps = [
        "//utils:s3",
        "@pypi//botocore",
    ],
)

py_test(
    name = "gpu_test",
    srcs = ["gpu_test.py"],
    deps = ["//utils:gpu"],
)

py_test(
    name = "storage_url_test",
    srcs = ["storage_url_test.py"],
    deps = ["//utils:storage_url"],
)
