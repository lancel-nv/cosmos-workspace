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
load("//services:image_layers.bzl", "py_image_layers")
load("//services:local_ports.bzl", "LOCAL_PORTS")
load(":version.bzl", "VERSION")

exports_files(
    ["packages.yaml"],
    visibility = ["//services/tests:__pkg__"],
)

genrule(
    name = "generate_version_txt",
    outs = ["version.txt"],
    cmd = "echo '" + VERSION + "' > $@",
)

py_library(
    name = "settings",
    srcs = ["settings.py"],
    visibility = ["//:__subpackages__"],
    deps = [
        "//services:settings_base",
        "//services/framework:worker_settings",
        "@pypi//pydantic",
        "@pypi//pydantic_settings",
    ],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    visibility = ["//visibility:public"],
    deps = ["@pypi//pydantic"],
)

py_library(
    name = "service",
    srcs = ["service.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":models",
        "//checks/physical_plausibility:processor",
        "//checks/utils:multistorage",
        "//services/framework:service_base",
    ],
)

py_library(
    name = "rest_api_common",
    srcs = ["rest_api_common.py"],
    data = [":generate_version_txt"],
    visibility = ["//:__subpackages__"],
    deps = [
        ":models",
        ":service",
        ":settings",
        "//checks/physical_plausibility:processor",
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
    visibility = ["//visibility:public"],
    deps = [":rest_api_common"],
)

command(
    name = "rest_api_command",
    command = ":rest_api",
    environment = {"UVICORN_PORT": str(LOCAL_PORTS["physical_plausibility"])},
    visibility = ["//visibility:public"],
)

py_binary(
    name = "rest_api_deployment",
    srcs = ["rest_api_deployment.py"],
    # Thin container-only wrapper around rest_api_common.main; the shared implementation is covered by rest_api_test.
    tags = [
        "manual",
        "no_test_target",
    ],
    visibility = ["//:__subpackages__"],
    deps = [":rest_api_common"],
)

genrule(
    name = "startup_script",
    outs = ["startup.sh"],
    cmd = """
cat > $@ << 'EOF'
#!/bin/bash
set -e

dpkg -i /tmp/cuda-debs/*.deb
echo "/usr/lib/x86_64-linux-gnu/nvshmem/12" > /etc/ld.so.conf.d/nvshmem-cuda-12.conf
ldconfig
rm -rf /usr/local/cuda*/compat

useradd --create-home --shell /bin/bash --uid 1001 appuser
mkdir -p /cache/huggingface /cache/torch
chown appuser:appuser /cache/huggingface /cache/torch
chmod 1777 /tmp

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

py_image_layers(
    name = "app",
    binary = ":rest_api_deployment",
)

pkg_tar(
    name = "license_layer",
    srcs = [
        "//services/physical_plausibility:THIRD-PARTY-NOTICES.txt",
        "//third_party/ffmpeg:source_offer",
    ],
    owner = "1001.1001",
    ownername = "appuser.appuser",
    package_dir = "/app",
    tags = ["manual"],
)

# Preserve the CUDA base's package records while replacing the upgraded packages' versions and file lists.
merged_dpkg_status(
    name = "dpkg_status_merged",
    apt_repo = "@physical_plausibility_packages",
    base_image = "@cuda_cudnn_image_linux_amd64//:cuda_cudnn_image_linux_amd64",
    lock = "packages.lock.json",
)

flatten(
    name = "system_packages_layer",
    deduplicate = True,
    tags = ["manual"],
    tars = [
        "@physical_plausibility_packages//:packages",
        ":dpkg_status_merged",
    ],
)

oci_image(
    name = "image_base",
    base = "@cuda_cudnn_image",
    entrypoint = ["/usr/local/bin/startup.sh"],
    env = {
        "CUDA_COMPAT_DISABLED": "1",
        "DEBIAN_FRONTEND": "noninteractive",
        "HF_HOME": "/cache/huggingface",
        # Makes the container runtime inject the NVIDIA video-driver libraries required by FFmpeg's h264_cuvid decoder.
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "TORCH_HOME": "/cache/torch",
    },
    exposed_ports = ["8000"],
    tags = ["manual"],
    tars = [
        ":system_packages_layer",
        "//services:cuda_debs_layer",
        "//services:ffmpeg_layer",
        ":startup_layer",
        "//services:gpu_core_layer",
        ":app_pip_gpu_extras_layer",
        ":app_pip_other_deps_layer",
        ":license_layer",
        ":app_source_layer",
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

oci_load(
    name = "image_load",
    image = ":image",
    repo_tags = ["physical-plausibility-checker:" + VERSION],
    tags = ["manual"],
)

oci_push(
    name = "image_push",
    image = ":image",
    remote_tags = [],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)
