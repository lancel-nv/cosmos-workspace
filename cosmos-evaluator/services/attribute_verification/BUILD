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

load("@rules_distroless//distroless:defs.bzl", "flatten")
load("@rules_multirun//:defs.bzl", "command")
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_load", "oci_push")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("//services:dpkg_status.bzl", "merged_dpkg_status")
load("//services:local_ports.bzl", "LOCAL_PORTS")
load(":version.bzl", "VERSION")

exports_files(
    ["packages.yaml"],
    visibility = ["//services/tests:__pkg__"],
)

# ======================================
# Application packages
# ======================================

# Generate version.txt from version.bzl for runtime access
genrule(
    name = "generate_version_txt",
    outs = ["version.txt"],
    cmd = "echo '" + VERSION + "' > $@",
    visibility = ["//visibility:public"],
)

py_library(
    name = "settings",
    srcs = ["settings.py"],
    visibility = [
        "//services/attribute_verification:__pkg__",
        "//services/attribute_verification/tests:__pkg__",
    ],
    deps = [
        "//services:settings_base",
        "//services/framework:worker_settings",
        "@pypi//pydantic",
        "@pypi//pydantic_settings",
    ],
)

# Request models. Kept dependency-light (pydantic only) so the arbitrator can import the cloud request without
# pulling in the checker's heavy runtime.
py_library(
    name = "models",
    srcs = ["models.py"],
    visibility = ["//visibility:public"],
    deps = ["@pypi//pydantic"],
)

py_library(
    name = "attribute_verification_service",
    srcs = ["service.py"],
    visibility = ["//services/attribute_verification/tests:__pkg__"],
    deps = [
        ":models",
        ":settings",
        "//checks/attribute_verification:processor",
        "//checks/utils:multistorage",
        "//services/framework:service_base",
    ],
)

py_library(
    name = "rest_api_common",
    srcs = ["rest_api_common.py"],
    data = [":generate_version_txt"],
    visibility = ["//services/attribute_verification/tests:__pkg__"],
    deps = [
        ":attribute_verification_service",
        ":models",
        ":settings",
        "//checks/attribute_verification:processor",
        "//checks/utils:multistorage",
        "//services:settings_base",
        "//services:utils",
        "//services/framework:checker_app",
        "//services/framework/protocols:storage_provider",
        "@pypi//uvicorn",
    ],
)

py_binary(
    name = "rest_api",
    srcs = ["rest_api.py"],
    data = ["//utils:git_sha"],
    visibility = ["//services/attribute_verification/tests:__pkg__"],
    deps = [":rest_api_common"],
)

# Run the Rest API command with the local port environment variable set.
# NOTE: the Multistorage client configuration (MULTISTORAGECLIENT_CONFIGURATION), recommended role-specific model keys
# (LLM_API_KEY and VLM_API_KEY), legacy BUILD_NVIDIA_API_KEY fallback, and other runtime env vars are not set here; put
# them in ~/.cosmos_evaluator/.env, which .dazelrc forwards into the container.
command(
    name = "rest_api_command",
    command = ":rest_api",
    environment = {"UVICORN_PORT": str(LOCAL_PORTS["attribute_verification"])},
    visibility = ["//visibility:public"],
)

# Deployment variant of the rest_api py_binary. Like every other deployment binary, it omits //utils:git_sha so the
# app_layer cache is not invalidated on every commit; the running service reads the SHA from /app/git_sha.txt supplied
# by //services:git_sha_layer.
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

# Startup script that registers the packaged FFmpeg shared libraries, creates a non-root user, and launches the Bazel
# py_binary. Initialize the system CA bundle from the packaged Ubuntu certificates so HTTPS model clients can use
# the system trust store without host certificate mounts.
genrule(
    name = "startup_script",
    outs = ["startup.sh"],
    cmd = """
cat > $@ << 'EOF'
#!/bin/bash
set -e

# rules_distroless extracts package data without running Debian's post-install scripts, so create the certificate
# selection file those scripts normally generate before initializing the trust store.
if [ ! -f /etc/ca-certificates.conf ]; then
    find /usr/share/ca-certificates -type f -name '*.crt' | sort | sed 's,^/usr/share/ca-certificates/,,' > /etc/ca-certificates.conf
fi
update-ca-certificates
# The packaged Python's OpenSSL defaults to /etc/ssl/cert.pem rather than Ubuntu's bundle filename.
ln -sf certs/ca-certificates.crt /etc/ssl/cert.pem

# Register the packaged FFmpeg shared libraries (extracted to /usr/local/lib by //services:ffmpeg_layer) with the
# dynamic linker so the ffmpeg binary resolves libavcodec/libavformat/libSvtAv1Enc/etc. at runtime.
ldconfig

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

# App layer bundles the python executable with all its runfiles (python code + pip dependencies)
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
    srcs = [
        "//services/attribute_verification:THIRD-PARTY-NOTICES.txt",
        "//third_party/ffmpeg:source_offer",
    ],
    owner = "1001.1001",
    ownername = "appuser.appuser",
    package_dir = "/app",
    tags = ["manual"],
)

merged_dpkg_status(
    name = "dpkg_status_merged",
    apt_repo = "@attribute_verification_packages",
    base_image = "@ubuntu_22_04_image_linux_amd64//:ubuntu_22_04_image_linux_amd64",
    lock = "packages.lock.json",
)

flatten(
    name = "system_packages_layer",
    deduplicate = True,
    tags = ["manual"],
    tars = [
        "@attribute_verification_packages//:packages",
        ":dpkg_status_merged",
    ],
)

# Base image without git_sha layer/labels. Used directly by image tests to avoid invalidating the Bazel test cache on
# every commit.
oci_image(
    name = "image_base",
    base = "@ubuntu_22_04_image",
    entrypoint = ["/usr/local/bin/startup.sh"],
    env = {
        "DEBIAN_FRONTEND": "noninteractive",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        # Declared even though this image carries no CUDA toolkit: cuvid decoding needs only the driver libraries the
        # container runtime injects (libcuda from "compute", libnvcuvid from "video"), so this is what lets the
        # first-frame extraction decode H.264 when a GPU is allocated. On a GPU-less host it has no effect and, since
        # the vendored ffmpeg has no software H.264 codec, this image is then AV1-only via libdav1d.
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video",
    },
    exposed_ports = ["8000"],
    tags = ["manual"],
    tars = [
        ":system_packages_layer",
        "//services:ffmpeg_layer",
        ":startup_layer",
        ":app_layer",
        ":license_layer",
    ],
    visibility = ["//visibility:public"],
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
    visibility = ["//visibility:public"],
)

oci_load(
    name = "image_load",
    image = ":image",
    repo_tags = ["attribute-verification-checker:" + VERSION],
    tags = ["manual"],
)
