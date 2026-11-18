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
load("//services/objects/single_view:version.bzl", "VERSION")

# Test-only image load using image_base (without git_sha) to avoid invalidating
# the Bazel test cache on every commit.
oci_load(
    name = "test_image_load",
    testonly = True,
    image = "//services/objects/single_view:image_base",
    repo_tags = ["objects:" + VERSION],
    tags = ["manual"],
)

py_test(
    name = "image_test",
    timeout = "long",
    srcs = ["image_test.py"],
    data = [":test_image_load"],
    env = {"UVICORN_PORT": str(LOCAL_PORTS["objects_single_view"])},
    tags = [
        "exclusive",  # Run in isolation to avoid resource contention
        "gpu",
        "no-coverage",  # No meaningful Python coverage is collected so exclude it from coverage runs to improve runtime
    ],
    deps = ["//services/objects/tests:image_test_mixin"],
)

py_test(
    name = "service_test",
    srcs = ["service_test.py"],
    deps = [
        "//checks/objects/single_view:object_processor_base",
        "//checks/utils:video",
        "//services/objects/single_view:models",
        "//services/objects/single_view:service",
        "@pypi//pydantic",
    ],
)

py_test(
    name = "rest_api_common_test",
    srcs = ["rest_api_common_test.py"],
    deps = [
        "//services:utils",
        "//services/framework:auth",
        "//services/framework/protocols:storage_provider",
        "//services/framework/storage_providers:config",
        "//services/framework/storage_providers:local_storage_provider",
        "//services/framework/storage_providers:s3_storage_provider",
        "//services/objects/single_view:models",
        "//services/objects/single_view:rest_api_common",
        "@pypi//fastapi",
    ],
)

py_test(
    name = "rest_api_test",
    srcs = ["rest_api_test.py"],
    deps = [
        "//services/objects/single_view:rest_api",
        "//services/objects/single_view:rest_api_common",
        "@pypi//fastapi",
    ],
)

py_test(
    name = "settings_test",
    srcs = ["settings_test.py"],
    deps = [
        "//services:settings_base",
        "//services/objects/single_view:settings",
        "//utils:environment",
        "@pypi//pydantic",
    ],
)
