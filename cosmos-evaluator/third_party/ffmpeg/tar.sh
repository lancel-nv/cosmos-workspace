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

# Builds FFmpeg from source with LGPL licensing, NVENC/CUVID hardware acceleration, SVT-AV1 CPU-based AV1
# encoding, dav1d software AV1 decoding, and libvpx CPU-based VP9 encoding.
#
# Codec policy: the codec set is an allow-list. `--disable-encoders --disable-decoders` turns everything off, and only
# the codecs the repo actually invokes are switched back on, so no codec can arrive by default or be pulled in
# transitively by an `--enable-*` flag. The point is that no software AVC exists anywhere in the shipped images: H.264
# decodes only through NVIDIA's licensed hardware path, so H.264 input requires a GPU. HEVC is absent in both
# directions -- software HEVC carries royalty exposure comparable to AVC, and the hardware decoder has no caller to
# justify it -- so HEVC input is unsupported outright rather than merely un-accelerated.
# //third_party/ffmpeg:ffmpeg_encoders_test asserts exact set equality against the list below, so anything enabled here
# must be justified here.
#
# The configure flags are grouped by purpose -- install, build shape, licensing, libraries, codec reset, codec
# allow-list, compiler flags -- and sorted alphabetically within each group. Group order is load-bearing, not
# cosmetic: configure applies flags left to right, so the three `--disable-encoders/decoders/hwaccels` resets must
# stay ahead of the `--enable-encoder=`/`--enable-decoder=` entries, which would otherwise be wiped. The groups
# cannot be marked with comments because the block is one backslash-continued command inside a `bash -c` string.
#
# Encoders:
#   - h264_nvenc, av1_nvenc, libsvtav1: the three tiers of the AV1/H.264 encode ladder (see checks/utils/video.py).
#   - libvpx_vp9: the whole of VP9 encoding. There is no NVENC VP9, and FFmpeg ships no native VP9 encoder either, so
#     libvpx is the only way to produce VP9 at all -- which is the point of carrying it: VP9 is royalty-free and, at
#     `-lossless 1`, is one of only two bit-exact encoders in this build (see the quality tiers in video.py).
#     `configure` spells it with an underscore; the command line is `-c:v libvpx-vp9`.
#   - mjpeg: what the image2 muxer picks implicitly for .jpg output.
#   - rawvideo: every stdin pipe into ffmpeg.
#   - wrapped_avframe: FFmpeg's default encoder for `-f null` probes.
#
# Decoders. The last two are the easy ones to drop by accident, because no command line in the repo names them:
#   - h264_cuvid, av1_cuvid: hardware decode.
#   - libdav1d: the software AV1 decoder. AV1 is royalty-free and is what the CPU encode tier produces, so
#     encode->decode round-trips have to keep working on GPU-less hosts (CPU-only builders, etc.). It is also the
#     only way to get a software AV1 path at all, FFmpeg's built-in "av1" decoder being hwaccel-only.
#   - vp9: the *only* VP9 decoder here, and unlike av1 this one is a real software decoder rather than an hwaccel
#     shell, so VP9 needs no external library to decode. It is also faster than libvpx's own, which is why libvpx
#     below is built encoder-only.
#
# vp9_cuvid is deliberately absent, and it is the one asymmetry in the table: H.264 and AV1 both get a hardware
# decoder and VP9 does not. Measured on 8.1.2, vp9_cuvid does not propagate color metadata -- it emits every frame
# tagged AVCOL_SPC_RGB (the enum's zero value) regardless of what the bitstream says, verified against streams
# explicitly tagged bt709 in both MP4 and WebM. swscale then refuses the mandatory `format=yuv420p` normalization
# outright ("Unsupported input ... fmt:nv12 csp:gbr"), and overriding the tag to get past that would make the
# subsequent YUV->RGB conversion disagree with the software tier, which is exactly the host-dependent divergence the
# normalization exists to prevent. Nothing is lost: VP9 decode carries no royalty exposure, so there is no licensing
# reason to want it on hardware, and being single-tier makes VP9 decode host-independent by construction -- a
# stronger guarantee than H.264 or AV1 get. Re-admitting it is one flag, once upstream propagates the metadata; the
# cuvid decoder still copies the bitstream's matrix_coefficients through verbatim in 9.0.1, unchanged since 8.1.2.
#   - rawvideo: every stdout pipe out of ffmpeg.
#   - mjpeg: reads .jpg back in, not just writes it. The multiview montage extracts frames to .jpg, then feeds them
#     back to ffmpeg as xstack inputs.
#   - wrapped_avframe: every `-f lavfi` input (e.g. `color=c=black:...`) arrives tagged as wrapped_avframe and must be
#     *decoded*. Without it both NVENC probe encodes fail and the ladder concludes the host has no hardware encoder.
#
# `--disable-hwaccels` is load-bearing, not tidying: each `X_nvdec_hwaccel` carries `select="X_decoder"`, which drags
# the full software decoder set back in behind the allow-list. It costs nothing, because hardware decode runs through
# the standalone cuvid decoders (`-c:v h264_cuvid`), which depend on `cuvid` rather than on any hwaccel, and never
# through `-hwaccel cuda`. Staying off the hwaccel path also keeps YUV->RGB conversion on CPU swscale rather than
# on-GPU NPP, which is half of what makes decode deterministic across hosts.
#
# The other half belongs to the decode ladder: cuvid decoders emit NV12 while libdav1d emits YUV420P, and swscale's
# NV12->BGR24 and YUV420P->BGR24 paths are not equivalent (up to 14/255 across 65% of bytes). Callers must normalize
# with `-vf format=yuv420p` before converting to BGR. With it, av1_cuvid and libdav1d agree byte-for-byte, as do
# h264_cuvid and a software h264 decode.
#
# Scope is deliberately encoders and decoders only. Muxers, demuxers, parsers, filters, protocols, and bitstream
# filters stay enabled: they carry no royalty exposure, and restricting them would break the drawtext/hstack/vstack
# filters the multiview assessment scripts rely on.
#
# Produces a tarball structured as an OCI image layer (files at usr/local/...) committed to the repo via Git LFS and
# consumed directly by the baseline Dockerfile (see third_party/ffmpeg/BUILD :ffmpeg_lgpl_tar).
#
# The archive step is fully deterministic (mtime derived from SOURCE_DATE_EPOCH, zeroed ownership, sorted entries,
# no gzip header timestamp), so identical input files always yield a byte-identical .tar.gz. The build step is made
# best-effort reproducible via SOURCE_DATE_EPOCH, build-path normalization, a removed ELF build-id, and symbol
# stripping. The base image is pinned by digest (CUDA_IMAGE below). One source of residual nondeterminism remains and
# requires operator action for a true bit-for-bit rebuild:
#   - apt toolchain: `apt-get install` pulls whatever gcc/binutils the base repos currently serve; pin via a snapshot
#     mirror for full reproducibility.
#
# Supply-chain integrity: version pinning alone is NOT authenticity verification — git tags are mutable and a release
# URL can serve different bytes over time. Each of the three pinned upstream downloads is therefore checked against
# an expected digest, and the build aborts on mismatch:
#   - nv-codec-headers: cloned tag is verified to resolve to NV_CODEC_HEADERS_COMMIT (git commit hashes are
#     content-addressed, so this pins the exact tree).
#   - SVT-AV1 source: cloned tag is verified to resolve to SVTAV1_COMMIT. SVT-AV1 lives on GitLab, whose
#     auto-generated archive tarballs are NOT byte-stable, so we content-verify via the immutable commit hash
#     (same treatment as nv-codec-headers) rather than a tarball sha256.
#   - dav1d source: cloned tag is verified to resolve to DAV1D_COMMIT. dav1d also lives on a GitLab instance
#     (code.videolan.org), so it gets the same immutable-commit content-verification as SVT-AV1.
#   - libvpx source: cloned tag is verified to resolve to LIBVPX_COMMIT, for the same reason -- it is hosted on a
#     Gitiles instance whose generated archives are not byte-stable. Note that LIBVPX_COMMIT is the *commit* the tag
#     resolves to, not the tag object's own hash: v1.16.0 is an annotated tag, so the two differ.
#   - FFmpeg source tarball: verified against FFMPEG_SHA256 via `sha256sum -c` (this digest was authenticated
#     out-of-band against the FFmpeg release signing key's detached GPG signature).
# When bumping any *_VERSION, refresh the matching digest below from an authenticated source.
#
# Usage:
#   ./third_party/ffmpeg/tar.sh             # build the tarball; commit the result via Git LFS
#
# Requirements:
#   - Docker with access to nvcr.io/nvidia/cuda images
#
# Environment overrides:
#   - CUDA_IMAGE: base image (default below, pinned by digest).

set -euo pipefail

FFMPEG_VERSION="${FFMPEG_VERSION:-9.0.1}"
# Stay on the 13.0 NVENC API line. The headers' NVENCAPI_MINOR_VERSION is what FFmpeg's nvenc.c checks the installed
# driver against, and the n13.1.x line raises it to 13.1, which refuses to open any NVENC session on a driver older
# than 610.00 ("Driver does not support the required nvenc API version. Required: 13.1 Found: 13.0"). Our hosts run
# 580.x, so 13.1 headers would drop both hardware encoders off the ladder fleet-wide and leave only libsvtav1.
# Decoding is unaffected either way. Bump this past 13.0 only once the deployed driver floor reaches 610.00.
NV_CODEC_HEADERS_VERSION="${NV_CODEC_HEADERS_VERSION:-n13.0.19.1}"
SVTAV1_VERSION="${SVTAV1_VERSION:-v4.2.0}"
DAV1D_VERSION="${DAV1D_VERSION:-1.5.4}"
LIBVPX_VERSION="${LIBVPX_VERSION:-v1.16.0}"

# Expected integrity digests for the pinned downloads above (see "Supply-chain integrity" note). Refresh these whenever
# the corresponding *_VERSION changes, sourcing them from an authenticated channel (signed release, GPG-verified
# tarball, or a trusted git host over TLS).
NV_CODEC_HEADERS_COMMIT="${NV_CODEC_HEADERS_COMMIT:-88fee5c37318c991a8762d423530f91681e32e3a}"
SVTAV1_COMMIT="${SVTAV1_COMMIT:-9292ec8e32bce26f781f277ec8739b53426c4300}"
DAV1D_COMMIT="${DAV1D_COMMIT:-54706fc6bc0cdecab7e9593974a4039cc038fca7}"
LIBVPX_COMMIT="${LIBVPX_COMMIT:-1024874c5919305883187e2953de8fcb4c3d7fa6}"
FFMPEG_SHA256="${FFMPEG_SHA256:-cf38e0e28c7e5605942c4a77755349b0145804a397af37eb1fb4c77cb237f635}"

# CVE-2026-2447: a heap buffer overflow in libvpx's VP9 encoder, reported after v1.16.0 shipped and still unfixed in
# any tag -- there is no v1.16.1. `write_superframe_index` returned a byte count it had not written when the output
# buffer was full, and its fullness check was itself off by one. The affected function is on the ordinary VP9 encode
# path rather than in some optional feature, so shipping the release as tagged means shipping the overflow.
# We take Debian's approach (libvpx 1.16.0-3) instead: a release-validated base plus this one upstream commit,
# cherry-picked by hash so it is content-verified exactly like the tag is, rather than pinning an untagged main
# revision and inheriting ~180 unrelated commits of drift. Drop this once upstream tags a release containing it.
LIBVPX_CVE_2026_2447_COMMIT="${LIBVPX_CVE_2026_2447_COMMIT:-d5f35ac8d93cba7f7a3f7ddb8f9dc8bd28f785e1}"

# Base build image, pinned by digest for a reproducible toolchain; the tag names what the digest tracks.
CUDA_IMAGE="${CUDA_IMAGE:-nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85}"

# Fixed timestamp for both compiler-embedded dates (SOURCE_DATE_EPOCH) and tar mtimes; the tar --mtime below derives
# its value from this variable so the two never drift. 1577836800 == 2020-01-01T00:00:00Z.
SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1577836800}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARBALL="ffmpeg-${FFMPEG_VERSION}-lgpl-nvenc-svtav1-linux-amd64.tar.gz"
OUTPUT="${SCRIPT_DIR}/${TARBALL}"

echo "==> Building FFmpeg ${FFMPEG_VERSION} with NVENC/CUVID + SVT-AV1 + dav1d + libvpx (LGPL)..."
docker run --rm -v "${SCRIPT_DIR}:/output" \
  -e SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH}" \
  "${CUDA_IMAGE}" bash -c "
    set -euxo pipefail

    # Best-effort build reproducibility: normalize embedded build paths (/tmp -> .) and drop the ELF build-id (its
    # default value hashes nondeterministic inputs). SOURCE_DATE_EPOCH is honored by the compiler for __DATE__/__TIME__
    # and is exported into the container above.
    export CFLAGS=\"-ffile-prefix-map=/tmp=. -g0\"
    export LDFLAGS=\"-Wl,--build-id=none\"
    # SVT-AV1 installs its .pc under /usr/local/lib/pkgconfig; make ffmpeg's configure (which detects libsvtav1 via
    # pkg-config) look there.
    export PKG_CONFIG_PATH=\"/usr/local/lib/pkgconfig:/usr/local/lib/x86_64-linux-gnu/pkgconfig:\${PKG_CONFIG_PATH:-}\"

    # yasm is here for libvpx, whose configure looks for it by name before falling back to anything else; the other
    # builds are served by nasm.
    apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake curl git meson ninja-build nasm pkg-config xz-utils yasm

    # nv-codec-headers (MIT-licensed) for NVENC/CUVID support
    git clone --depth 1 --branch ${NV_CODEC_HEADERS_VERSION} \
      https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers
    # Integrity: tags are mutable, so verify the clone resolved to the expected immutable commit.
    actual_commit=\$(git -C /tmp/nv-codec-headers rev-parse HEAD)
    if [ \"\$actual_commit\" != \"${NV_CODEC_HEADERS_COMMIT}\" ]; then
      echo \"nv-codec-headers commit mismatch: expected ${NV_CODEC_HEADERS_COMMIT}, got \$actual_commit\" >&2
      exit 1
    fi
    make -C /tmp/nv-codec-headers -j\$(nproc)
    make -C /tmp/nv-codec-headers install

    # SVT-AV1 (BSD-3-Clause-Clear + AOM Patent License 1.0) for royalty-free CPU-based AV1 encoding
    git clone --depth 1 --branch ${SVTAV1_VERSION} \
      https://gitlab.com/AOMediaCodec/SVT-AV1.git /tmp/svt-av1
    # Integrity: GitLab archive tarballs are not byte-stable, so content-verify via the immutable commit hash.
    actual_svtav1_commit=\$(git -C /tmp/svt-av1 rev-parse HEAD)
    if [ \"\$actual_svtav1_commit\" != \"${SVTAV1_COMMIT}\" ]; then
      echo \"SVT-AV1 commit mismatch: expected ${SVTAV1_COMMIT}, got \$actual_svtav1_commit\" >&2
      exit 1
    fi
    cmake -S /tmp/svt-av1 -B /tmp/svt-av1/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_APPS=OFF \
      -DBUILD_TESTING=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build /tmp/svt-av1/build -j\$(nproc)
    cmake --install /tmp/svt-av1/build
    ldconfig

    # dav1d (BSD-2-Clause) for royalty-free software AV1 decoding (FFmpeg's built-in av1 decoder is hwaccel-only).
    git clone --depth 1 --branch ${DAV1D_VERSION} \
      https://code.videolan.org/videolan/dav1d.git /tmp/dav1d
    # Integrity: GitLab archive tarballs are not byte-stable, so content-verify via the immutable commit hash.
    actual_dav1d_commit=\$(git -C /tmp/dav1d rev-parse HEAD)
    if [ \"\$actual_dav1d_commit\" != \"${DAV1D_COMMIT}\" ]; then
      echo \"dav1d commit mismatch: expected ${DAV1D_COMMIT}, got \$actual_dav1d_commit\" >&2
      exit 1
    fi
    # --libdir lib keeps the install off the multiarch x86_64-linux-gnu path so its .pc and .so land where the
    # PKG_CONFIG_PATH above and the bundling step below already look.
    meson setup /tmp/dav1d/build /tmp/dav1d \
      --buildtype release \
      --default-library shared \
      --prefix /usr/local \
      --libdir lib \
      -Denable_tools=false \
      -Denable_tests=false
    ninja -C /tmp/dav1d/build
    ninja -C /tmp/dav1d/build install
    ldconfig

    # libvpx (BSD-3-Clause) for royalty-free CPU-based VP9 encoding. Neither NVENC nor FFmpeg itself can encode VP9.
    git clone --depth 1 --branch ${LIBVPX_VERSION} \
      https://chromium.googlesource.com/webm/libvpx.git /tmp/libvpx
    # Integrity: Gitiles archive tarballs are not byte-stable, so content-verify via the immutable commit hash.
    actual_libvpx_commit=\$(git -C /tmp/libvpx rev-parse HEAD)
    if [ \"\$actual_libvpx_commit\" != \"${LIBVPX_COMMIT}\" ]; then
      echo \"libvpx commit mismatch: expected ${LIBVPX_COMMIT}, got \$actual_libvpx_commit\" >&2
      exit 1
    fi
    # Apply the CVE-2026-2447 fix (see the pin above). --depth 2 is the minimum that works: cherry-pick diffs the
    # commit against its parent, so fetching the commit alone leaves nothing to diff against.
    git -C /tmp/libvpx fetch --depth 2 origin ${LIBVPX_CVE_2026_2447_COMMIT}
    git -C /tmp/libvpx -c user.email=build@localhost -c user.name=build \
      cherry-pick --no-commit ${LIBVPX_CVE_2026_2447_COMMIT}
    # Encoder-only, and VP9-only: decoding runs through FFmpeg's native vp9 decoder, which is both faster than
    # libvpx's and already in the allow-list, and nothing in the repo touches VP8 in either direction. Out-of-tree
    # build because libvpx's configure refuses to run in the source directory.
    mkdir -p /tmp/libvpx/build
    ( cd /tmp/libvpx/build && ../configure \
        --prefix=/usr/local \
        --disable-docs \
        --disable-examples \
        --disable-static \
        --disable-tools \
        --disable-unit-tests \
        --disable-vp8 \
        --disable-vp9-decoder \
        --enable-pic \
        --enable-shared \
        --enable-vp9-encoder )
    make -C /tmp/libvpx/build -j\$(nproc)
    make -C /tmp/libvpx/build install
    ldconfig

    # Download and build FFmpeg
    curl -fsSL -o /tmp/ffmpeg.tar.xz https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz
    # Integrity: verify before extracting so a tampered tarball stops the build immediately.
    echo \"${FFMPEG_SHA256}  /tmp/ffmpeg.tar.xz\" | sha256sum -c -
    tar -xJf /tmp/ffmpeg.tar.xz -C /tmp
    cd /tmp/ffmpeg-${FFMPEG_VERSION}

    ./configure \
      --prefix=/opt/ffmpeg \
      --disable-debug \
      --disable-doc \
      --disable-static \
      --enable-pic \
      --enable-shared \
      --disable-gpl \
      --disable-nonfree \
      --disable-cuda-nvcc \
      --disable-vulkan \
      --enable-cuvid \
      --enable-libdav1d \
      --enable-libsvtav1 \
      --enable-libvpx \
      --enable-nvenc \
      --disable-decoders \
      --disable-encoders \
      --disable-hwaccels \
      --enable-encoder=av1_nvenc \
      --enable-encoder=h264_nvenc \
      --enable-encoder=libsvtav1 \
      --enable-encoder=libvpx_vp9 \
      --enable-encoder=mjpeg \
      --enable-encoder=rawvideo \
      --enable-encoder=wrapped_avframe \
      --enable-decoder=av1_cuvid \
      --enable-decoder=h264_cuvid \
      --enable-decoder=libdav1d \
      --enable-decoder=mjpeg \
      --enable-decoder=rawvideo \
      --enable-decoder=vp9 \
      --enable-decoder=wrapped_avframe \
      --extra-cflags=\"-ffile-prefix-map=/tmp=. -g0\" \
      --extra-ldflags=\"-Wl,--build-id=none\"

    make -j\$(nproc)
    make install

    # Bundle the SVT-AV1, dav1d and libvpx shared libraries alongside FFmpeg's own libs
    cp /usr/local/lib/x86_64-linux-gnu/libSvtAv1Enc.so* /opt/ffmpeg/lib/ 2>/dev/null \
      || cp /usr/local/lib/libSvtAv1Enc.so* /opt/ffmpeg/lib/
    cp /usr/local/lib/libdav1d.so* /opt/ffmpeg/lib/ 2>/dev/null \
      || cp /usr/local/lib/x86_64-linux-gnu/libdav1d.so* /opt/ffmpeg/lib/
    cp /usr/local/lib/libvpx.so* /opt/ffmpeg/lib/ 2>/dev/null \
      || cp /usr/local/lib/x86_64-linux-gnu/libvpx.so* /opt/ffmpeg/lib/

    # Strip symbol tables (their content/ordering can vary build-to-build) for a more stable payload.
    strip --strip-unneeded /opt/ffmpeg/lib/*.so* 2>/dev/null || true
    strip /opt/ffmpeg/bin/ffmpeg /opt/ffmpeg/bin/ffprobe 2>/dev/null || true

    # Package runtime files as a reproducible OCI layer tar (paths prefixed with usr/local/).
    # Determinism: --sort + sorted find => stable entry order; --mtime (from SOURCE_DATE_EPOCH) and zeroed numeric
    # ownership => no build-time metadata; --pax-option drops atime/ctime; gzip -n omits the gzip header
    # timestamp/filename. Identical inputs therefore produce a byte-identical archive. The 'S' flag on --transform
    # restricts the rewrite to member names only: GNU tar otherwise also rewrites symlink *targets*, which would
    # turn the same-dir relative links (libavcodec.so -> libavcodec.so.63.1.101) into broken 'usr/local/...' targets.
    cd /opt/ffmpeg
    tar --sort=name \
        --mtime=\"@\$SOURCE_DATE_EPOCH\" \
        --owner=0 --group=0 --numeric-owner \
        --pax-option='exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime' \
        --transform='s,^,usr/local/,S' \
        -cf - \
        bin/ffmpeg bin/ffprobe \
        \$(find lib -maxdepth 1 -name '*.so*' \\( -type f -o -type l \\) | sort) \
      | gzip -n > /output/${TARBALL}
  "

echo "==> Built: ${OUTPUT}"
sha256sum "${OUTPUT}"
echo "==> Commit this file to Git (tracked by LFS via .gitattributes)."
