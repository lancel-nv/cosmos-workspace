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
load("//services/hallucination:version.bzl", "VERSION")

# Test-only image load using image_base (without git_sha) to avoid invalidating
# the Bazel test cache on every commit.
oci_load(
    name = "test_image_load",
    testonly = True,
    image = "//services/hallucination:image_base",
    repo_tags = ["hallucination-checker:" + VERSION],
    tags = ["manual"],
)

py_test(
    name = "image_test",
    timeout = "long",
    srcs = ["image_test.py"],
    data = [":test_image_load"],
    env = {"UVICORN_PORT": str(LOCAL_PORTS["hallucination"])},
    tags = [
        "exclusive",  # Run in isolation to avoid resource contention
        "no-coverage",  # No meaningful Python coverage is collected so exclude it from coverage runs to improve runtime
    ],
    deps = [
        "//utils:container_images",
        "@pypi//testcontainers",
    ],
)

py_test(
    name = "hallucination_service_test",
    srcs = ["hallucination_service_test.py"],
    deps = [
        "//checks/hallucination:processor",
        "//services/hallucination:hallucination_service",
    ],
)

py_test(
    name = "rest_api_test",
    srcs = ["rest_api_test.py"],
    deps = [
        "//checks/hallucination:processor",
        "//services:utils",
        "//services/hallucination:models",
        "//services/hallucination:rest_api",
        "//services/hallucination:rest_api_common",
        "@pypi//fastapi",
        "@pypi//httpx",  # keep: runtime backend for starlette/fastapi TestClient, not a static import
    ],
)

py_test(
    name = "settings_test",
    srcs = ["settings_test.py"],
    deps = [
        "//services:settings_base",
        "//services/hallucination:settings",
        "//utils:environment",
        "@pypi//pydantic",
    ],
)
