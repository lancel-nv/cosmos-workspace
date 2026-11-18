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

load("@rules_python//python:py_library.bzl", "py_library")

py_library(
    name = "auth",
    srcs = ["auth.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@pypi//fastapi",
        "@pypi//starlette",
    ],
)

py_library(
    name = "checker_job",
    srcs = ["checker_job.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//database:api_client",
        "//database:api_models",
        "//utils:cancellation",
    ],
)

py_library(
    name = "job_slots",
    srcs = ["job_slots.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "checker_app",
    srcs = ["checker_app.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":auth",
        ":checker_job",
        ":gpu_health",
        ":job_slots",
        ":metrics",
        ":worker_settings",
        "//database:api_client",
        "//services:utils",
        "//services/framework/protocols:storage_provider",
        "//services/framework/request_handlers:json_request_handler",
        "//services/framework/response_formatters:json_response_formatter",
        "//services/framework/storage_providers:factory",
        "//utils:environment",
        "//utils:gpu",
        "@pypi//fastapi",
        "@pypi//pydantic",
    ],
)

py_library(
    name = "service_base",
    srcs = ["service_base.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    visibility = ["//visibility:public"],
    deps = [":job_slots"],
)

py_library(
    name = "worker_settings",
    srcs = ["worker_settings.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//services:settings_base",
        "@pypi//pydantic",
    ],
)

py_library(
    name = "gpu_health",
    srcs = ["gpu_health.py"],
    visibility = ["//visibility:public"],
)
