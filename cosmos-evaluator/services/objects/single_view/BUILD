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

# Read by //services/tests:apt_pins_test, which enforces that this manifest's snapshot pin is no older than the base
# image's build date.
exports_files(
    ["packages.yaml"],
    visibility = ["//services/tests:__pkg__"],
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
    visibility = ["//services/objects/single_view/tests:__pkg__"],
    deps = [
        ":models",
        "//checks/objects/single_view:api",
        "//checks/utils:video",
        "//services/framework:service_base",
        "//utils:cancellation",
    ],
)

py_library(
    name = "settings",
    srcs = ["settings.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//services:settings_base",
        "//services/framework:worker_settings",
        "@pypi//pydantic",
        "@pypi//pydantic_settings",
    ],
)

# Generate version.txt from version.bzl for runtime access
genrule(
    name = "generate_version_txt",
    outs = ["version.txt"],
    cmd = "echo '" + VERSION + "' > $@",
    visibility = ["//visibility:public"],
)

py_library(
    name = "rest_api_common",
    srcs = ["rest_api_common.py"],
    data = [":generate_version_txt"],
    visibility = ["//services/objects/single_view/tests:__pkg__"],
    deps = [
        ":models",
        ":service",
        ":settings",
        "//checks/objects/single_view:object_processor_static",
        "//checks/utils:segformer",
        "//services:settings_base",
        "//services:utils",
        "//services/framework:checker_app",
        "//services/framework/protocols:storage_provider",
        "//utils:cancellation",
        "@pypi//onnxruntime_gpu",
        "@pypi//torch",
        "@pypi//uvicorn",
    ],
)

py_binary(
    name = "rest_api",
    srcs = ["rest_api.py"],
    data = ["//utils:git_sha"],
    visibility = ["//services/objects/single_view/tests:__pkg__"],
    deps = [":rest_api_common"],
)

command(
    name = "rest_api_command",
    command = ":rest_api",
    environment = {"UVICORN_PORT": str(LOCAL_PORTS["objects_single_view"])},
    visibility = ["//visibility:public"],
)

# //utils:git_sha excluded from the deployed rest_api to avoid invalidating the cache for app_layer on every commit
py_binary(
    name = "rest_api_deployment",
    srcs = ["rest_api_deployment.py"],
    tags = ["manual"],
    visibility = ["//visibility:public"],
    deps = [":rest_api_common"],
)

# Create a startup script that installs packages and starts the app:
# - System packages are pre-installed via the @objects_packages deb layer
# - NVIDIA CUDA .deb files are pre-downloaded at build time and installed via dpkg -i
#   to avoid apt-get update/install network overhead at container startup
genrule(
    name = "startup_script",
    outs = ["startup.sh"],
    cmd = """
cat > $@ << 'EOF'
#!/bin/bash
set -e

dpkg -i /tmp/cuda-debs/*.deb

# Create symlinks that are normally set up by postinst scripts (via update-alternatives)
# but aren't created by rules_distroless since it only extracts package files.
ln -sf /usr/lib/x86_64-linux-gnu/blas/libblas.so.3 /usr/lib/x86_64-linux-gnu/libblas.so.3
ln -sf /usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3 /usr/lib/x86_64-linux-gnu/liblapack.so.3

# Register libnvshmem (installed to non-standard path) with the dynamic linker.
echo "/usr/lib/x86_64-linux-gnu/nvshmem/12" > /etc/ld.so.conf.d/nvshmem-cuda-12.conf
ldconfig

# Remove CUDA compat libraries to disable forward compatibility when host system
# has an older version of CUDA installed because this container already pins the
# CUDA version to 12.8.1
if ls /usr/local/cuda*/compat 1> /dev/null 2>&1; then rm -rf /usr/local/cuda*/compat; fi

# Create non-root user for running the application
useradd --create-home --shell /bin/bash --uid 1001 appuser

# Ensure /tmp is world-writable (distroless base images may not set this)
chmod 1777 /tmp

# Start NVIDIA MPS (Multi-Process Service) control daemon. MPS merges CUDA contexts from cooperating processes into one
# so kernels from different processes execute in parallel on free SMs instead of time-slicing the GPU, which is the
# canonical fix for many-process GPU underutilization.
# Started as root so clients running as appuser can connect to the same MPS server.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p "$$CUDA_MPS_PIPE_DIRECTORY" "$$CUDA_MPS_LOG_DIRECTORY"
chmod 1777 "$$CUDA_MPS_PIPE_DIRECTORY" "$$CUDA_MPS_LOG_DIRECTORY"
if command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  nvidia-cuda-mps-control -d \
    || echo "WARN: failed to start nvidia-cuda-mps-control daemon; continuing without MPS"

  # Force the MPS server into existence before the app can claim any work. The line above starts only the *control
  # daemon*; the server itself is created lazily on the first client connection, and every client arriving during
  # that window is refused with CUDA_ERROR_MPS_SERVER_NOT_READY. That window is not hypothetical here: the vendored
  # ffmpeg decodes H.264 only through h264_cuvid, so even reading a video's dimensions is a CUDA client, and a job
  # claimed moments after startup fails on a perfectly good video.
  #
  # Three details are load-bearing. The probe runs as appuser rather than root, because MPS servers are per-user and
  # a root-owned server is the wrong one for the app's own clients. The loop only keeps waiting while the outcome is
  # an MPS-not-ready state or a timed-out attempt: anything else (success, or an unrelated ffmpeg error) stops it
  # immediately, so a host without a usable GPU costs one probe instead of the whole budget. And each attempt is
  # bounded as well as the loop, because a probe against a wedged MPS server or GPU can block for as long as the
  # driver allows -- an unbounded one would let a single call outlive the budget entirely, which matters more here
  # than it would elsewhere: the exec below never runs, so the container sits short of starting with nothing left to
  # time it out. A timed-out attempt is counted as not-ready, which is what it nearly always means.
  mps_probe="ffmpeg -hide_banner -loglevel error -init_hw_device cuda=gpu"
  mps_probe="$$mps_probe -f lavfi -i color=c=black:s=16x16:d=0.04 -f null -"
  mps_env="CUDA_MPS_PIPE_DIRECTORY='$$CUDA_MPS_PIPE_DIRECTORY' CUDA_MPS_LOG_DIRECTORY='$$CUDA_MPS_LOG_DIRECTORY'"
  mps_budget=$${MPS_WARMUP_TIMEOUT_SECONDS:-30}
  # Generous against a healthy device, where this is a CUDA context creation and a 0.04s encode, and still short
  # enough that a hung attempt leaves most of the budget for a real one.
  mps_attempt_budget=10
  mps_deadline=$$((SECONDS + mps_budget))
  mps_ready=0
  while [ "$$SECONDS" -lt "$$mps_deadline" ]; do
    # Kept as the left operand of && because `set -e` is in force: a bare assignment would abort the whole startup
    # script the first time the probe exited non-zero, which is the normal case this loop exists to retry.
    mps_error=$$(timeout "$$mps_attempt_budget" su appuser -c "$$mps_env $$mps_probe" 2>&1) \
      && { mps_ready=1; break; }
    # 124 is how `timeout` reports that it killed the probe; $$? here is the assignment's status, since && short-
    # circuited before `break`.
    if [ "$$?" -ne 124 ]; then
      case "$$mps_error" in
        *MPS_SERVER_NOT_READY*|*MPS_CONNECTION_FAILED*|*SYSTEM_NOT_READY*) ;;
        *)
          echo "WARN: MPS warm-up probe failed for an unrelated reason, not waiting: $$mps_error"
          break
          ;;
      esac
    fi
    sleep 1
  done
  if [ "$$mps_ready" -eq 0 ] && [ "$$SECONDS" -ge "$$mps_deadline" ]; then
    echo "WARN: MPS server still not ready after $${mps_budget}s; starting anyway"
  fi
else
  echo "WARN: nvidia-cuda-mps-control not found; skipping MPS startup"
fi

# Switch to non-root user and run the application. CUDA_MPS_* env vars are
# explicitly forwarded so client CUDA processes connect to the daemon above.
cd /app
exec su appuser -c "CUDA_MPS_PIPE_DIRECTORY='$$CUDA_MPS_PIPE_DIRECTORY' CUDA_MPS_LOG_DIRECTORY='$$CUDA_MPS_LOG_DIRECTORY' ./rest_api_deployment --host $${UVICORN_HOST:-0.0.0.0} --port $${UVICORN_PORT:-8000}"
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

# Split the Python binary into separate GPU deps, pip deps, app source, and model layers. GPU pip packages
# (~5-6 GiB, torch/CUDA/triton — very stable) are separated from other pip packages (~1-2 GiB) and app source
# (~100-300 MB, changes frequently) to improve build cache hit rates and docker load times. The CWIP model weights
# (~180 MB) are split into a dedicated model layer solely for caching and layer-size isolation; the build produces a
# single image that includes the weights.
py_image_layers(
    name = "app",
    binary = ":rest_api_deployment",
    model_file_pattern = "model\\.safetensors",
)

# Merge overlay package metadata into the base image's dpkg database. This ensures dpkg -l and security scanners
# (e.g. nSpect) see all packages — both from the base image and from packages.yaml — in a single database, and attribute
# each file installed from packages.yaml to its package.
merged_dpkg_status(
    name = "dpkg_status_merged",
    apt_repo = "@objects_packages",
    base_image = "@cuda_cudnn_image_linux_amd64//:cuda_cudnn_image_linux_amd64",
    lock = "packages.lock.json",
)

# License layer adds the third party notices and license files to the container image
pkg_tar(
    name = "license_layer",
    srcs = [
        "//services/objects/single_view:THIRD-PARTY-NOTICES.txt",
        "//third_party/ffmpeg:source_offer",
    ],
    owner = "1001.1001",
    ownername = "appuser.appuser",
    package_dir = "/app",
    tags = ["manual"],
)

flatten(
    name = "system_packages_layer",
    deduplicate = True,
    tags = ["manual"],
    tars = [
        "@objects_packages//:packages",
        ":dpkg_status_merged",
    ],
)

# Base image with CWIP model weights, without git_sha layer/labels. Used directly by image tests to avoid invalidating
# the Bazel test cache on every commit.
oci_image(
    name = "image_base",
    base = "@cuda_cudnn_image",
    entrypoint = ["/usr/local/bin/startup.sh"],
    env = {
        "DEBIAN_FRONTEND": "noninteractive",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_COMPAT_DISABLED": "1",
        # NVDEC/NVENC exist only if the container runtime injects libnvcuvid/libnvidia-encode, which it gates on
        # "video". The vendored ffmpeg carries no software H.264 codec, so without this the image cannot read H.264
        # at all and every encode falls to the SVT-AV1 CPU tier.
        "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,video",
        # Bound glibc per-thread malloc arenas. The service offloads CPU-heavy work onto a thread pool, and each thread
        # otherwise gets its own arena (default 8 * ncpu) that retains freed memory, producing a steady RSS climb across
        # requests. Capping at 4 keeps freed memory consolidated so malloc_trim() can return it to the OS.
        "MALLOC_ARENA_MAX": "4",
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
        ":app_model_layer",
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

oci_push(
    name = "image_push",
    image = ":image",
    remote_tags = [],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

oci_load(
    name = "image_load",
    image = ":image",
    repo_tags = ["objects-checker:" + VERSION],
    tags = ["manual"],
)
