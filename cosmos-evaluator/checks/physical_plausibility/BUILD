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

load("@rules_python//python:defs.bzl", "py_binary", "py_library")

py_library(
    name = "model",
    srcs = ["model.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":features",
        "//checks/utils:video",
        "@pypi//einops",  # keep: imported by the verified local model source.
        "@pypi//filelock",
        "@pypi//numpy",
        "@pypi//timm",  # keep: imported by the verified local model source.
        "@pypi//torch",
    ],
)

py_binary(
    name = "run",
    srcs = ["run.py"],
    data = ["//checks:config"] + glob(
        [".env"],
        allow_empty = True,
    ),
    visibility = ["//visibility:public"],
    deps = [
        ":processor",
        "//checks/utils:config_manager",
        "//checks/utils:multistorage",
        "//utils:bazel",
        "@pypi//python_dotenv",
        "@pypi//pyyaml",
    ],
)

py_library(
    name = "processor",
    srcs = ["processor.py"],
    data = ["//checks:config"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":model",
        "//checks/utils:config_manager",
        "//checks/utils:multistorage",
        "@pypi//huggingface_hub",
        "@pypi//pydantic",
        "@pypi//torch",
    ],
)

py_library(
    name = "features",
    srcs = ["features.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@opencv_python_headless",
        "@pypi//numpy",
    ],
)

py_library(
    name = "attention",
    srcs = ["attention.py"],
    visibility = ["//visibility:public"],
    deps = ["@pypi//torch"],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_library(
    name = "tokens",
    srcs = ["tokens.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":model",
        "@pypi//filelock",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_library(
    name = "training",
    srcs = ["training.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":metrics",
        ":model",
        ":tokens",
        "@pypi//filelock",
        "@pypi//numpy",
        "@pypi//torch",
    ],
)

py_library(
    name = "evaluation",
    srcs = ["evaluation.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":attention",
        ":metrics",
        ":model",
        ":tokens",
        "@pypi//numpy",
    ],
)
