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
load("//services/physical_plausibility:version.bzl", "VERSION")

oci_load(
    name = "test_image_load",
    testonly = True,
    image = "//services/physical_plausibility:image_base",
    repo_tags = ["physical-plausibility-checker:" + VERSION],
    tags = ["manual"],
)

py_library(
    name = "image_test_base",
    testonly = True,
    srcs = ["image_test_base.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//services/objects/tests:image_test_mixin",
        "//utils:container_images",
        "@pypi//testcontainers",
    ],
)

py_test(
    name = "image_test",
    timeout = "long",
    srcs = ["image_test.py"],
    data = [":test_image_load"],
    env = {"UVICORN_PORT": str(LOCAL_PORTS["physical_plausibility"])},
    env_inherit = ["HF_TOKEN"],
    tags = [
        "exclusive",
        "gpu",
        "no-coverage",
    ],
    deps = [":image_test_base"],
)

py_test(
    name = "physical_plausibility_service_test",
    srcs = ["physical_plausibility_service_test.py"],
    deps = [
        "//checks/physical_plausibility:attention",
        "//checks/physical_plausibility:processor",
        "//services/arbitrator:dependency_graph",
        "//services/physical_plausibility:models",
        "//services/physical_plausibility:service",
    ],
)

py_test(
    name = "rest_api_test",
    srcs = ["rest_api_test.py"],
    deps = [
        "//checks/physical_plausibility:processor",
        "//services/physical_plausibility:models",
        "//services/physical_plausibility:rest_api",
        "//services/physical_plausibility:rest_api_common",
        "@pypi//fastapi",
        "@pypi//httpx",  # keep: runtime backend for starlette/fastapi TestClient, not a static import
    ],
)

py_test(
    name = "settings_test",
    srcs = ["settings_test.py"],
    deps = [
        "//services:settings_base",
        "//services/physical_plausibility:settings",
        "//utils:environment",
        "@pypi//pydantic",
    ],
)
