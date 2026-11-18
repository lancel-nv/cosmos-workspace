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

load("@rules_multirun//:defs.bzl", "command")
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_load", "oci_push")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("//services:local_ports.bzl", "LOCAL_PORTS")
load(":version.bzl", "VERSION")

# ======================================
# Application packages
# ======================================

py_library(
    name = "settings",
    srcs = ["settings.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//services:settings_base",
        "//utils:environment",
        "@pypi//pydantic",
        "@pypi//pydantic_settings",
    ],
)

py_library(
    name = "auth",
    srcs = ["auth.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":settings",
        "@pypi//fastapi",
    ],
)

py_library(
    name = "service",
    srcs = ["service.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":settings",
        "//services/database/queries:orchestration",
        "@pypi//asyncpg",
    ],
)

py_library(
    name = "app",
    srcs = ["app.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":auth",
        ":service",
        ":settings",
        "//services:utils",
        "//services/database/routers:arbitrator",
        "//services/database/routers:checker_work",
        "//services/framework/response_formatters:json_response_formatter",
        "//utils:environment",
        "@pypi//fastapi",
        "@pypi//pydantic",
    ],
)

# Generate version.txt from version.bzl for runtime access
genrule(
    name = "generate_version_txt",
    outs = ["version.txt"],
    cmd = "echo '" + VERSION + "' > $@",
    visibility = ["//visibility:public"],
)

# FastAPI app + uvicorn entry point (main). git_sha-free library shared by both binaries below so the deployment
# binary can reach main() without dragging //utils:git_sha into its runfiles.
py_library(
    name = "rest_api_common",
    srcs = ["rest_api_common.py"],
    data = [":generate_version_txt"],
    visibility = ["//services/database/tests:__pkg__"],
    deps = [
        ":app",
        ":settings",
        "//database:db_config",
        "//services:settings_base",
        "//services:utils",
        "//utils:environment",
        "@pypi//uvicorn",
    ],
)

py_binary(
    name = "rest_api",
    srcs = ["rest_api.py"],
    data = ["//utils:git_sha"],
    visibility = ["//services/database/tests:__pkg__"],
    deps = [":rest_api_common"],
)

# Run the Rest API command with the local port environment variable set.
command(
    name = "rest_api_command",
    command = ":rest_api",
    environment = {"UVICORN_PORT": str(LOCAL_PORTS["database"])},
    visibility = ["//visibility:public"],
)

# Deployment variant of the rest_api py_binary. Omits //utils:git_sha from data so the app_layer cache is not
# invalidated on every commit; the running service reads the SHA from /app/git_sha.txt supplied by
# //services:git_sha_layer.
py_binary(
    name = "rest_api_deployment",
    srcs = ["rest_api_deployment.py"],
    tags = ["manual"],
    visibility = ["//:__subpackages__"],
    deps = [":rest_api_common"],
)

# ======================================
# Container Image Build
# ======================================

# Startup script that creates a non-root user and launches the Bazel py_binary.
genrule(
    name = "startup_script",
    outs = ["startup.sh"],
    cmd = """
cat > $@ << 'EOF'
#!/bin/bash
set -e

# Create non-root user for running the application
useradd --create-home --shell /bin/bash --uid 1001 appuser

# Ensure /tmp is world-writable (base images may not set this)
chmod 1777 /tmp

# Switch to non-root user and run the application
cd /app
exec su appuser -c "./rest_api_deployment --host $${UVICORN_HOST:-0.0.0.0} --port $${UVICORN_PORT:-8000}"
EOF
""",
)

pkg_tar(
    name = "startup_layer",
    srcs = [":startup_script"],
    mode = "0755",
    package_dir = "/usr/local/bin",
    tags = ["manual"],
)

# App layer bundles the python executable with all its runfiles. owner/group are set to 1001 (appuser) at build time
# to avoid an expensive chown -R at startup.
pkg_tar(
    name = "app_layer",
    srcs = [":rest_api_deployment"],
    include_runfiles = True,
    owner = "1001.1001",
    ownername = "appuser.appuser",
    package_dir = "/app",
    tags = ["manual"],
)

# License layer adds the third party notices and license files to the container image
pkg_tar(
    name = "license_layer",
    srcs = ["//services/database:THIRD-PARTY-NOTICES.txt"],
    owner = "1001.1001",
    ownername = "appuser.appuser",
    package_dir = "/app",
    tags = ["manual"],
)

# Base image without git_sha layer/labels. Used directly by image tests to avoid
# invalidating the Bazel test cache on every commit.
oci_image(
    name = "image_base",
    base = "@ubuntu_22_04_image",
    entrypoint = ["/usr/local/bin/startup.sh"],
    env = {
        "DEBIAN_FRONTEND": "noninteractive",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    },
    exposed_ports = ["8000"],
    tags = ["manual"],
    tars = [
        ":startup_layer",
        ":app_layer",
        ":license_layer",
    ],
    visibility = ["//services/database/tests:__pkg__"],
    workdir = "/app",
)

oci_image(
    name = "image",
    base = ":image_base",
    labels = "//utils:git_sha",
    tags = ["manual"],
    tars = ["//services:git_sha_layer"],
)

# ======================================
# Container Image Publishing and Testing
# ======================================

oci_push(
    name = "image_push",
    image = ":image",
    remote_tags = [],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

# Load the container image for testing.
oci_load(
    name = "image_load",
    image = ":image",
    repo_tags = ["database:" + VERSION],
    tags = ["manual"],
)
