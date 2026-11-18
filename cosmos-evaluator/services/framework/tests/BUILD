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
    name = "auth_test",
    srcs = ["auth_test.py"],
    deps = [
        "//services/framework:auth",
        "@pypi//fastapi",
        "@pypi//httpx",  # keep: runtime backend for starlette/fastapi TestClient, not a static import
    ],
)

py_test(
    name = "job_slots_test",
    srcs = ["job_slots_test.py"],
    deps = ["//services/framework:job_slots"],
)

py_test(
    name = "service_base_test",
    srcs = ["service_base_test.py"],
    deps = [
        "//services/framework:service_base",
        "@pypi//pytest",
        "@pypi//pytest_asyncio",  # keep: no static import available for pytest asyncio
    ],
)

py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    deps = [
        "//services/framework:job_slots",
        "//services/framework:metrics",
    ],
)

py_test(
    name = "checker_job_test",
    srcs = ["checker_job_test.py"],
    deps = [
        "//database:api_client",
        "//database:api_models",
        "//services/framework:checker_job",
        "//utils:cancellation",
        "@pypi//pytest",
        "@pypi//pytest_asyncio",  # keep: no static import available for pytest asyncio
    ],
)

py_test(
    name = "checker_app_test",
    srcs = ["checker_app_test.py"],
    deps = [
        "//database:api_models",
        "//services/framework:checker_app",
        "//services/framework:worker_settings",
        "//services/framework/request_handlers:json_request_handler",
        "//utils:environment",
        "//utils:gpu",
        "@pypi//fastapi",
        "@pypi//httpx",  # keep: runtime backend for starlette/fastapi TestClient, not a static import
        "@pypi//pydantic",
        "@pypi//pydantic_settings",
    ],
)

py_test(
    name = "gpu_health_test",
    srcs = ["gpu_health_test.py"],
    deps = ["//services/framework:gpu_health"],
)
