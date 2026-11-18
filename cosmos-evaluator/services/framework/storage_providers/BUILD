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
    name = "exceptions",
    srcs = ["exceptions.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "config",
    srcs = ["config.py"],
    visibility = ["//visibility:public"],
    deps = ["@pypi//pydantic"],
)

py_library(
    name = "s3_storage_provider",
    srcs = ["s3_storage_provider.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":exceptions",
        "//services:settings_base",
        "//services/framework/protocols:storage_provider",
        "//services/framework/storage_providers/utils:local",
        "//services/framework/storage_providers/utils:s3",
        "//services/framework/url:security",
        "//utils:storage_url",
        "@pypi//aioboto3",
        "@pypi//aiofiles",
        "@pypi//aiohttp",
        "@pypi//botocore",
        "@pypi//certifi",
    ],
)

py_library(
    name = "local_storage_provider",
    srcs = ["local_storage_provider.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//services/framework/protocols:storage_provider",
        "//services/framework/storage_providers/utils:local",
    ],
)

py_library(
    name = "factory",
    srcs = ["factory.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":azure_blob_storage_provider",
        ":config",
        ":local_storage_provider",
        ":s3_storage_provider",
        "//services:settings_base",
        "//services/framework/protocols:storage_provider",
        "//utils:s3",
        "//utils:storage_url",
    ],
)

py_library(
    name = "azure_blob_storage_provider",
    srcs = ["azure_blob_storage_provider.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":config",
        ":exceptions",
        "//services/framework/protocols:storage_provider",
        "//services/framework/storage_providers/utils:azure",
        "//services/framework/storage_providers/utils:local",
        "//services/framework/url:security",
        "@pypi//azure_core",
        "@pypi//azure_storage_blob",
    ],
)
