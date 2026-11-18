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

load("@rules_oci//oci:defs.bzl", "oci_load")
load("@rules_python//python:py_test.bzl", "py_test")
load("//database:version.bzl", "VERSION")

# Test-only image load using //database:image_base (without git_sha labels) to keep the Bazel test cache
# stable across commits.
oci_load(
    name = "test_image_load",
    testonly = True,
    image = "//database:image_base",
    repo_tags = ["database-postgres:" + VERSION],
    tags = ["manual"],
)

py_test(
    name = "api_client_test",
    srcs = ["api_client_test.py"],
    deps = [
        "//database:api_client",
        "//database:api_models",
        "@pypi//httpx",
    ],
)

py_test(
    name = "db_config_test",
    srcs = ["db_config_test.py"],
    deps = ["//database:db_config"],
)

py_test(
    name = "image_test",
    # Each new container downloads PostgreSQL at boot, so the duration follows the network rather than the host. The
    # per-boot waits in image_test.py bound a genuine hang, and together they exceed the "long" limit.
    timeout = "eternal",
    srcs = ["image_test.py"],
    data = [":test_image_load"],
    tags = [
        "exclusive",
        "no-coverage",
    ],
    deps = [
        "//utils:container_images",
        "@pypi//asyncpg",
        "@pypi//testcontainers",
    ],
)
