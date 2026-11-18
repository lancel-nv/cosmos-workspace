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
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//services:local_ports.bzl", "LOCAL_PORTS")
load("//services/arbitrator:version.bzl", "VERSION")

# Test-only image load using image_base (without git_sha) to avoid invalidating the Bazel test cache on every commit.
oci_load(
    name = "test_image_load",
    testonly = True,
    image = "//services/arbitrator:image_base",
    repo_tags = ["arbitrator-base:" + VERSION],
    tags = ["manual"],
)

py_test(
    name = "image_test",
    timeout = "long",
    srcs = ["image_test.py"],
    data = [":test_image_load"],
    env = {"UVICORN_PORT": str(LOCAL_PORTS["arbitrator"])},
    tags = [
        "exclusive",  # Run in isolation to avoid resource contention
        "no-coverage",  # No meaningful Python coverage is collected, so exclude it from coverage runs
    ],
    deps = [
        "//utils:container_images",
        "@pypi//testcontainers",
    ],
)

py_test(
    name = "checker_queue_test",
    srcs = ["checker_queue_test.py"],
    deps = ["//services/arbitrator:checker_queue"],
)

py_library(
    name = "waiting_room_test_helpers",
    testonly = True,
    srcs = ["waiting_room_test_helpers.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//services/arbitrator:checker_queue",
        "//services/arbitrator:waiting_room_queue",
    ],
)

py_test(
    name = "waiting_room_queue_test",
    srcs = ["waiting_room_queue_test.py"],
    deps = [
        ":waiting_room_test_helpers",
        "//services/arbitrator:checker_queue",
        "//services/arbitrator:waiting_room_queue",
        "@pypi//pytest",
        "@pypi//pytest_asyncio",  # keep: no static import available for pytest asyncio
    ],
)

py_test(
    name = "durable_dispatch_test",
    srcs = ["durable_dispatch_test.py"],
    deps = [
        ":waiting_room_test_helpers",
        "//database:api_models",
        "//services/arbitrator:checker_queue",
        "//services/arbitrator:waiting_room_queue",
        "@pypi//pytest",
        "@pypi//pytest_asyncio",  # keep: no static import available for pytest asyncio
    ],
)

py_test(
    name = "execution_adoption_test",
    srcs = ["execution_adoption_test.py"],
    deps = [
        ":waiting_room_test_helpers",
        "//database:api_models",
        "//services/arbitrator:checker_queue",
        "//services/arbitrator:dependency_graph",
        "//services/arbitrator:rest_api_common",
        "//services/arbitrator:settings",
        "//services/arbitrator:waiting_room_queue",
    ],
)

py_test(
    name = "graceful_shutdown_test",
    srcs = ["graceful_shutdown_test.py"],
    deps = [
        "//services/arbitrator:dependency_graph",
        "//services/arbitrator:rest_api_common",
        "//services/arbitrator:settings",
        "//services/arbitrator:waiting_room_queue",
        "@pypi//fastapi",
    ],
)

py_test(
    name = "rest_api_common_test",
    srcs = ["rest_api_common_test.py"],
    deps = [
        "//services/arbitrator:checker_queue",
        "//services/arbitrator:dependency_graph",
        "//services/arbitrator:rest_api_common",
        "//services/arbitrator:settings",
        "//services/arbitrator/models:contract",
        "//services/arbitrator/records:execution_record",
        "@pypi//fastapi",
    ],
)

py_test(
    name = "rest_api_app_test",
    srcs = ["rest_api_app_test.py"],
    deps = [
        "//database:api_client",
        "//database:api_models",
        "//database:statuses",
        "//services/arbitrator:dependency_graph",
        "//services/arbitrator:rest_api_common",
        "//services/arbitrator:settings",
        "//services/arbitrator:waiting_room_queue",
        "@pypi//fastapi",
        "@pypi//httpx",
    ],
)

py_test(
    name = "rest_api_test",
    srcs = ["rest_api_test.py"],
    deps = ["//services/arbitrator:rest_api"],
)

py_test(
    name = "dependency_graph_test",
    srcs = ["dependency_graph_test.py"],
    deps = [
        "//services/arbitrator:checker_queue",
        "//services/arbitrator:dependency_graph",
    ],
)

py_test(
    name = "settings_test",
    srcs = ["settings_test.py"],
    deps = [
        "//services:settings_base",
        "//services/arbitrator:dependency_graph",
        "//services/arbitrator:settings",
        "//utils:environment",
        "@pypi//pydantic",
    ],
)
