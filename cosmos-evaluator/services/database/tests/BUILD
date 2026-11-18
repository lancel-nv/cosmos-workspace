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
load("//services:local_ports.bzl", "LOCAL_PORTS")
load("//services/database:version.bzl", "VERSION")

py_test(
    name = "settings_test",
    srcs = ["settings_test.py"],
    deps = [
        "//services/database:settings",
        "//utils:environment",
        "@pypi//pydantic",
    ],
)

py_test(
    name = "auth_test",
    srcs = ["auth_test.py"],
    deps = [
        "//services/database:auth",
        "@pypi//fastapi",
    ],
)

py_test(
    name = "service_test",
    srcs = ["service_test.py"],
    deps = [
        "//services/database:service",
        "//services/database:settings",
        "//services/database/queries:orchestration",
    ],
)

py_test(
    name = "response_contract_test",
    srcs = ["response_contract_test.py"],
    deps = [
        "//database:api_models",
        "//services/database:rest_api_common",
        "//services/database:settings",
        "//services/database/queries:orchestration",
        "//services/framework/response_formatters:json_response_formatter",
        "@pypi//fastapi",
        "@pypi//httpx",  # keep: runtime backend for starlette/fastapi TestClient, not a static import
        "@pypi//pydantic",
    ],
)

py_test(
    name = "rest_api_test",
    srcs = ["rest_api_test.py"],
    deps = [
        "//services:utils",
        "//services/database:app",
        "//services/database:rest_api",
        "//services/database:rest_api_common",
        "//services/database:service",
        "//services/database:settings",
        "//services/database/queries:orchestration",
        "@pypi//fastapi",
        "@pypi//httpx",  # keep: runtime backend for starlette/fastapi TestClient, not a static import
    ],
)

# Test-only image load using image_base (without git_sha) to avoid invalidating the Bazel test cache on every commit.
oci_load(
    name = "test_image_load",
    testonly = True,
    image = "//services/database:image_base",
    repo_tags = ["database-base:" + VERSION],
    tags = ["manual"],
)

py_test(
    name = "image_test",
    timeout = "long",
    srcs = ["image_test.py"],
    data = [":test_image_load"],
    env = {"UVICORN_PORT": str(LOCAL_PORTS["database"])},
    tags = [
        "exclusive",
        "no-coverage",
    ],
    deps = [
        "//utils:container_images",
        "@pypi//testcontainers",
    ],
)
