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

# Use base image that includes CUDA 12.8 with Ubuntu 22.04 and cuDNN
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/doc/supported-tags.md
# Pinned by digest so every machine builds from the same base; the tag names what the digest tracks. To take updates,
# resolve the tag with `docker buildx imagetools inspect` and replace the digest.
FROM docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85

# Set root user to install packages
USER root

# Disable prompt during packages installation
ENV DEBIAN_FRONTEND=noninteractive

# Set timezone to UTC
ENV TZ=Etc/UTC

# Use HTTPS for Ubuntu archive/security mirrors. Plain HTTP to archive.ubuntu.com from some egress paths is severely
# throttled / sometimes blocked, which slows `apt-get update` to the point where Canonical publishes a new snapshot
# mid-run and apt aborts with "File has unexpected size ... Mirror sync in progress?". HTTPS routes to a different,
# consistently faster Canonical backend and also protects InRelease+Packages consistency in transit. ca-certificates
# ships in the base CUDA image, so no extra packages are needed to enable HTTPS here.
RUN sed -i -E \
        's#https?://(archive|security|ports)\.ubuntu\.com/#https://\1.ubuntu.com/#g' \
        /etc/apt/sources.list

# Update package lists, install software-properties-common, add the deadsnakes PPA, and install Python 3.13
RUN apt-get update && \
    apt-get install --no-install-recommends --yes software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install --no-install-recommends --yes python3.13 python3.13-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt update && \
    # Upgrade system NCCL from 2.25.1 (base image) to 2.27.5 to match the version
    # required by torch 2.10+cu128. The base image holds NCCL at 2.25.1 which is missing
    # ncclCommWindowRegister and other symbols needed by the newer PyTorch.
    # First remove libnccl-dev which pins libnccl2 to the base image version.
    apt remove --yes --allow-change-held-packages libnccl-dev && \
    apt install --no-install-recommends --yes --allow-change-held-packages \
        libnccl2=2.27.5-1+cuda12.4 && \
    # Install base system packages (alphabetically sorted)
    # Note: ca-certificates and gnupg are already in the base CUDA image.
    apt install --no-install-recommends --yes \
        curl \
        cusparselt-cuda-12 \
        git \
        libegl1-mesa \
        libgl1 \
        libgl1-mesa-dri \
        libglu1-mesa \
        libnvshmem3-cuda-12 \
        libusb-1.0-0 \
        libxcursor1 \
        libxinerama1 \
        libxrandr2 \
        lsb-release \
        openssh-client \
        openssl \
        sudo && \
    # Register libnvshmem (installed to non-standard path) with the dynamic linker.
    # Required by torch 2.10+ which links against libnvshmem_host.so.3.
    echo "/usr/lib/x86_64-linux-gnu/nvshmem/12" > /etc/ld.so.conf.d/nvshmem-cuda-12.conf && \
    ldconfig && \
    # Clean up apt cache to reduce image size
    rm -rf /var/lib/apt/lists/*

# Install Docker CLI only (not the daemon since we'll use the host's daemon via mounted socket from .dazelrc) from
# the official Docker repository. Match DOCKER_CLI_VERSION default to minimum recommended version in tools/env_setup.sh.
ARG DOCKER_CLI_VERSION=5:28.1.1-1~ubuntu.22.04~jammy
RUN ARCH=$(dpkg --print-architecture) && \
    DOCKER_KEY=/usr/share/keyrings/docker-archive-keyring.gpg && \
    DOCKER_URL=https://download.docker.com/linux/ubuntu && \
    curl --proto "=https" --tlsv1.2 -fsSL "${DOCKER_URL}/gpg" | gpg --dearmor -o "${DOCKER_KEY}" && \
    echo "deb [arch=${ARCH} signed-by=${DOCKER_KEY}] ${DOCKER_URL} $(lsb_release -cs) stable" \
        | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt update && \
    apt install --no-install-recommends --yes docker-ce-cli=${DOCKER_CLI_VERSION} && \
    rm -rf /var/lib/apt/lists/*

# Install Bazelisk for Bazel version management.
RUN ARCH=$(dpkg --print-architecture) && \
    if [ "$ARCH" = "arm64" ]; then BAZELISK_ARCH="arm64"; else BAZELISK_ARCH="amd64"; fi && \
    BAZELISK_URL_BASE=https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0 && \
    curl --proto "=https" --tlsv1.2 -fL "${BAZELISK_URL_BASE}/bazelisk-linux-${BAZELISK_ARCH}" -o /usr/local/bin/bazelisk && \
    chmod +x /usr/local/bin/bazelisk && \
    /usr/local/bin/bazelisk version && \
    ln -sf /usr/local/bin/bazelisk /usr/local/bin/bazel && \
    /usr/local/bin/bazel version

# Install pre-built LGPL FFmpeg 9.0.1 with NVENC/CUVID hardware acceleration and SVT-AV1 CPU AV1 encoding. The tarball
# is packaged as an OCI image layer (paths prefixed with usr/local/) and passed in via Docker's
# `--build-context ffmpeg=...` (configured in .dazelrc). See third_party/ffmpeg/tar.sh to rebuild the tarball.
COPY --from=ffmpeg ffmpeg-9.0.1-lgpl-nvenc-svtav1-linux-amd64.tar.gz /tmp/ffmpeg.tar.gz
RUN tar -xzf /tmp/ffmpeg.tar.gz -C / && \
    rm /tmp/ffmpeg.tar.gz && \
    ldconfig

# Create a symbolic link for python3 pointing to python3.13
RUN ln -sf /usr/bin/python3.13 /usr/bin/python3

# Create a user with the same UID and GID as the host user (see .dazelrc for corresponding --build-arg args).
ARG home
ARG groups
ARG user_id
ARG user_name

RUN mkdir -p /etc/sudoers.d && \
    echo "ALL ALL = (ALL) NOPASSWD: ALL" > /etc/sudoers.d/host_user && \
    chmod 0440 /etc/sudoers.d/host_user && \
    mkdir -p "$(dirname $home)"

RUN if [ $user_id -ne 0 ]; \
    then \
        useradd \
            --shell /bin/bash \
            --uid $user_id \
            --non-unique \
            --comment "" \
            --no-log-init \
            --create-home $user_name \
            --home-dir $home; \
    else \
        user_name=root; \
    fi;

# Add user to all specified groups. Note that the group id is the important part. If the group doesn't exist, create it
# with a new name to mark it comes from the host so that we can add the user to that id.
RUN IS_PRIMARY_GROUP=1; \
    for g in $groups; \
    do \
      group_name=$(echo $g | cut -f1 -d:); \
      group_id=$(echo $g | cut -f2 -d:); \
      getent group $group_id > /dev/null; \
      if [ $? -ne 0 ]; \
      then \
        groupadd -f -g $group_id $(echo "host_$group_name" | cut -c 1-30); \
      fi; \
      if [ $IS_PRIMARY_GROUP -eq 1 ]; \
      then \
        usermod -g $group_id $user_name; \
        IS_PRIMARY_GROUP=0; \
      else \
        usermod -a -G $group_id $user_name; \
      fi; \
    done

# Remove CUDA compat libraries to disable forward compatibility when host system has an older version of CUDA installed
# because this container already pins the CUDA version to 12.8.1.
RUN if ls /usr/local/cuda*/compat 1> /dev/null 2>&1; then rm -rf /usr/local/cuda*/compat; fi && ldconfig
ENV CUDA_COMPAT_DISABLED=1

# Set environment variables for the user.
ENV HOME=$home
ENV LC_ALL=C.UTF-8
ENV USER=$user_name
USER $user_name

# Verify tools.
RUN bazel --version && \
    ffmpeg -version && \
    python3 --version
