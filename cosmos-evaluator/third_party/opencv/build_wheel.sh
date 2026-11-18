#!/usr/bin/env bash
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

# Builds opencv-python-headless from source with the bundled FFmpeg and Intel IPP support disabled
# (-DWITH_FFMPEG=OFF -DWITH_IPP=OFF).
#
# Stock opencv-python(-headless) PyPI wheels hard-link an old LGPL FFmpeg (libav*) into cv2's extension module via
# RUNPATH, which there is no supported way to swap at runtime. For strict LGPL compliance (and to avoid shipping a
# second, stale FFmpeg next to the repo's vendored LGPL FFmpeg 9.0.1) we rebuild the wheel with WITH_FFMPEG=OFF. The
# result keeps every cv2 image op (imread/imwrite/imencode/cvtColor/resize/morphology/...) and loses only the
# VideoCapture/VideoWriter FFmpeg backend, which callers route through the vendored ffmpeg binary instead
# (see checks/utils/video.py).
#
# By default OpenCV also downloads Intel IPP ICV and the IPP Integration Wrappers at configure time and links them
# statically into cv2. IPP ICV is licensed under the Intel Simplified Software License, which is not an open-source
# license, so WITH_IPP=OFF keeps it out of the wheel. cv2 falls back to OpenCV's own SIMD code paths for the operations
# IPP would otherwise accelerate; the API is unchanged.
#
# opencv-python ships one LICENSE-3RD-PARTY.txt with all of its wheels, so it attributes libraries this build does not
# contain and omits code that OpenCV's own build attributes. curate_third_party_licenses.py derives the wheel's copy the
# way OpenCV attributes third-party code: OpenCV's COPYRIGHT notice, opencv-python's sections for this build, and the
# license files OpenCV installs for this configuration. It fails the build on any section or dependency it has not
# classified.
#
# The wheel is committed to the repo via Git LFS and consumed as a Bazel py_library through the local_wheel module
# extension (see third_party/local_wheel.bzl + opencv/BUILD.opencv), mirroring how the vendored FFmpeg
# tarball is consumed (see third_party/ffmpeg/tar.sh).
#
# Supply-chain integrity: the build clones opencv/opencv-python at a pinned git tag and verifies the clone resolved
# to the expected immutable commit (git commit hashes are content-addressed). opencv-python recursively pins the
# opencv source submodule, so this transitively pins the OpenCV C++ sources. When bumping OPENCV_PYTHON_TAG, refresh
# OPENCV_PYTHON_COMMIT below from an authenticated source.
#
# Usage:
#   ./third_party/opencv/build_wheel.sh    # build the wheel; commit the result via Git LFS
#
# Requirements:
#   - Docker with access to the manylinux build image
#
# Environment overrides:
#   - MANYLINUX_IMAGE: base build image (default below, pinned by digest).
#   - OPENCV_PYTHON_TAG / OPENCV_PYTHON_COMMIT: upstream opencv-python tag and its expected commit.
#   - CMAKE_BUILD_PARALLEL_LEVEL: parallel compile jobs (default: the build container's CPU count). Lower it on hosts
#     with little memory per core.

set -euo pipefail

# opencv-python release. The git tag is the package build number; the resulting version string is
# <opencv_version>.<build_number> (e.g. 4.13.0.92). Keep this in sync with the opencv-python-headless version that
# was previously resolved from PyPI in requirements.txt.
OPENCV_PYTHON_TAG="${OPENCV_PYTHON_TAG:-92}"
OPENCV_PYTHON_VERSION="${OPENCV_PYTHON_VERSION:-4.13.0.92}"

# Expected immutable commit for OPENCV_PYTHON_TAG (tags are mutable; commit hashes are content-addressed). This is a
# REQUIRED pin: without it the integrity check below has nothing to verify against and could be silently skipped, which
# would defeat the supply-chain guarantee. Recorded here so a rebuild reproduces the committed wheel from the same
# sources and fails loudly if the tag ever moves; refresh from an authenticated source whenever OPENCV_PYTHON_TAG
# changes (`git ls-remote https://github.com/opencv/opencv-python.git refs/tags/<tag>^{}`).
OPENCV_PYTHON_COMMIT="${OPENCV_PYTHON_COMMIT:-4ddfc013fd1f13d9b9e379dbebf2cdbeb052e7f8}"

if [[ -z "${OPENCV_PYTHON_COMMIT:-}" ]]; then
  echo "ERROR: OPENCV_PYTHON_COMMIT must be set to the expected immutable commit for tag ${OPENCV_PYTHON_TAG}." >&2
  echo "       Tags are mutable; pinning the commit is what makes the supply-chain integrity check meaningful." >&2
  exit 1
fi

# CPython ABI to build for. Must match the rules_python python-build-standalone interpreter (CPython 3.13) so the
# resulting cp313 extension loads under it. The manylinux image ships this interpreter at /opt/python/cp313-cp313.
PYTHON_TAG="${PYTHON_TAG:-cp313-cp313}"

# manylinux build image. glibc 2.28 baseline => the wheel runs on glibc >= 2.28, including the ubuntu jammy (glibc
# 2.35) dazel image and service images. Pinned by digest for a reproducible toolchain; the tag names what the digest
# tracks.
MANYLINUX_IMAGE="${MANYLINUX_IMAGE:-quay.io/pypa/manylinux_2_28_x86_64:latest@sha256:4131ddd7acb9c6c966af02fe7e71edc51e078c2d41f475966697676430bfa2ae}"

# PEP 440 local version label identifying this build. It is applied to the wheel's internal metadata version, not just
# the filename: an SBOM scanner reads the version out of the installed .dist-info, so a wheel claiming plain
# OPENCV_PYTHON_VERSION would be indistinguishable from the PyPI build that bundles FFmpeg and IPP — the exact
# confusion this custom build exists to avoid.
LOCAL_LABEL="${LOCAL_LABEL:-noffmpeg.noipp}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_NAME="opencv_python_headless-${OPENCV_PYTHON_VERSION}+${LOCAL_LABEL}-${PYTHON_TAG}-manylinux_2_28_x86_64.whl"
OUTPUT="${SCRIPT_DIR}/${WHEEL_NAME}"

echo "==> Building opencv-python-headless ${OPENCV_PYTHON_VERSION} with WITH_FFMPEG=OFF WITH_IPP=OFF (${PYTHON_TAG})..."
docker run --rm -v "${SCRIPT_DIR}:/output" \
  -e OPENCV_PYTHON_TAG="${OPENCV_PYTHON_TAG}" \
  -e OPENCV_PYTHON_COMMIT="${OPENCV_PYTHON_COMMIT}" \
  -e OPENCV_PYTHON_VERSION="${OPENCV_PYTHON_VERSION}" \
  -e PYTHON_TAG="${PYTHON_TAG}" \
  -e LOCAL_LABEL="${LOCAL_LABEL}" \
  -e WHEEL_NAME="${WHEEL_NAME}" \
  -e CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-}" \
  "${MANYLINUX_IMAGE}" bash -c '
    set -euxo pipefail

    PYBIN="/opt/python/${PYTHON_TAG}/bin"

    # Headless main-modules build with FFmpeg and IPP disabled. ENABLE_HEADLESS=1 selects the headless package
    # (opencv-python-headless: no GUI/HighGUI). CMAKE_ARGS disables the libav backend so no libav* is linked/bundled,
    # and disables IPP so the ippicv/ippiw static libraries are neither downloaded nor linked.
    export ENABLE_HEADLESS=1
    export ENABLE_CONTRIB=0
    export CMAKE_ARGS="-DWITH_FFMPEG=OFF -DWITH_IPP=OFF"
    # opencv-python drives the build through scikit-build, whose `cmake --build` runs make serially unless this is set.
    export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"

    git clone --recursive https://github.com/opencv/opencv-python.git /tmp/opencv-python
    cd /tmp/opencv-python
    git checkout "${OPENCV_PYTHON_TAG}"
    git submodule update --init --recursive

    # Integrity: tags are mutable, so verify the clone resolved to the expected immutable commit (required pin,
    # validated as non-empty on the host before this container ran).
    actual_commit="$(git rev-parse HEAD)"
    if [ "${actual_commit}" != "${OPENCV_PYTHON_COMMIT}" ]; then
      echo "opencv-python commit mismatch: expected ${OPENCV_PYTHON_COMMIT}, got ${actual_commit}" >&2
      exit 1
    fi
    echo "opencv-python tag ${OPENCV_PYTHON_TAG} resolved to commit ${actual_commit}"

    "${PYBIN}/pip" install --upgrade pip
    "${PYBIN}/pip" wheel . --verbose -w /tmp/dist

    # Repair to a manylinux_2_28 wheel: bundles the non-system shared libs cv2 links into opencv_python_headless.libs/
    # and rewrites RUNPATHs. With FFmpeg off, no libav* should be pulled in.
    "${PYBIN}/pip" install auditwheel
    "${PYBIN}/python" -m auditwheel repair /tmp/dist/opencv_python_headless-*.whl \
      --plat manylinux_2_28_x86_64 -w /tmp/repaired

    repaired_wheel="$(ls /tmp/repaired/opencv_python_headless-*.whl)"

    # --- Gating verification: the rebuilt wheel must contain no FFmpeg/libav payload ---
    echo "==> Verifying no libav* is bundled in the wheel zip..."
    if "${PYBIN}/python" -m zipfile -l "${repaired_wheel}" | grep -i "libav"; then
      echo "ERROR: rebuilt wheel still bundles libav* (FFmpeg). WITH_FFMPEG=OFF did not take effect." >&2
      exit 1
    fi

    echo "==> Verifying cv2 extension has no libav* in its DT_NEEDED entries..."
    workdir="$(mktemp -d)"
    "${PYBIN}/python" -m zipfile -e "${repaired_wheel}" "${workdir}"
    for so in "${workdir}"/cv2/*.so; do
      if readelf -d "${so}" | grep -i "libav"; then
        echo "ERROR: ${so} still links libav* (FFmpeg)." >&2
        exit 1
      fi
    done
    echo "OK: no libav* found in wheel payload or cv2 NEEDED entries."

    # IPP is linked statically, so it never shows up as a bundled file or a DT_NEEDED entry. Check the binary itself:
    # an IPP-enabled build embeds ippicv* function names and the ippicv path in its build configuration.
    echo "==> Verifying no Intel IPP is linked into the cv2 extension..."
    for so in "${workdir}"/cv2/*.so; do
      if grep -a -q -i "ippicv" "${so}"; then
        echo "ERROR: ${so} still contains Intel IPP ICV. WITH_IPP=OFF did not take effect." >&2
        exit 1
      fi
    done
    echo "OK: no Intel IPP found in the cv2 extension."

    # --- Derive LICENSE-3RD-PARTY.txt from what this build contains ---
    # scikit-build installs OpenCV into the build tree, and OpenCV installs a license file for every piece of
    # third-party code the configured build compiles in. /output is the bind-mounted repo directory, so without
    # PYTHONDONTWRITEBYTECODE the container would drop a root-owned __pycache__ into the checkout.
    echo "==> Deriving LICENSE-3RD-PARTY.txt from the cv2 build configuration..."
    licenses_dirs=(/tmp/opencv-python/_skbuild/*/cmake-install/share/licenses/opencv4)
    if [ "${#licenses_dirs[@]}" -ne 1 ] || [ ! -d "${licenses_dirs[0]}" ]; then
      echo "ERROR: expected one OpenCV licenses directory in the install tree, found: ${licenses_dirs[*]}" >&2
      exit 1
    fi
    third_party_licenses="/tmp/LICENSE-3RD-PARTY.txt"
    PYTHONDONTWRITEBYTECODE=1 "${PYBIN}/python" /output/curate_third_party_licenses.py \
      "${workdir}/cv2/LICENSE-3RD-PARTY.txt" "${workdir}/cv2/cv2.abi3.so" /tmp/opencv-python/opencv/COPYRIGHT \
      "${licenses_dirs[0]}" "${third_party_licenses}"

    # --- Stamp the local version label and the derived license file into the wheel ---
    # Relabeling lives in stamp_local_version.py (mounted at /output alongside this script) rather than inline, so the
    # same code can be re-run against an already-built wheel without a full recompile.
    echo "==> Stamping +${LOCAL_LABEL} into the wheel metadata..."
    labeled_version="${OPENCV_PYTHON_VERSION}+${LOCAL_LABEL}"
    labeled_wheel="/tmp/${WHEEL_NAME}"
    PYTHONDONTWRITEBYTECODE=1 "${PYBIN}/python" /output/stamp_local_version.py \
      "${repaired_wheel}" "${labeled_wheel}" "${OPENCV_PYTHON_VERSION}" "${labeled_version}" "${third_party_licenses}"

    cp "${labeled_wheel}" "/output/${WHEEL_NAME}"
  '

echo "==> Built: ${OUTPUT}"
sha256sum "${OUTPUT}"
echo "==> Commit this file to Git (tracked by LFS via .gitattributes)."
