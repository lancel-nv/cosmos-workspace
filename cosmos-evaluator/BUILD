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

# gazelle:python_root
# gazelle:python_label_convention $distribution_name$
# Generate one target per file (matching the repo convention: py_library/py_binary/py_test named after the
# source file stem). Package mode would instead collapse a tests/ dir into a single "tests_test" target.
# gazelle:python_generation_mode file
# Name generated package build files "BUILD"
# gazelle:build_file_name BUILD
# Third-party imports served by a dedicated repo instead of @pypi (not in the Gazelle pip manifest).
# gazelle:resolve py cv2 @opencv_python_headless
# gazelle:resolve py python.runfiles @rules_python//python/runfiles
# GitHub archives vendored as their own Bazel repos. Gazelle does not index external repositories, so it cannot map a
# `third_party.<repo>...<mod>` import onto the repo's `@<repo>//:<target>` label on its own (the repo BUILD templates
# live under third_party/, which sets `# gazelle:python_extension disabled`). Pin each imported module explicitly, the
# same mechanism as cv2 above.
# gazelle:resolve py third_party.cosmos_drive_dreams_toolkits.utils.camera.base @cosmos_drive_dreams_toolkits//:camera_base
# gazelle:resolve py third_party.cosmos_drive_dreams_toolkits.utils.camera.ftheta @cosmos_drive_dreams_toolkits//:camera_ftheta
# gazelle:resolve py third_party.detectron2.config.instantiate @detectron2//:config
# gazelle:resolve py third_party.detectron2.config.lazy @detectron2//:config
# gazelle:resolve py third_party.perception_models.core.vision_encoder.pe @perception_models//:vision_encoder
# gazelle:resolve py third_party.perception_models.core.vision_encoder.rope @perception_models//:vision_encoder
load("@bazel_gazelle//:def.bzl", "gazelle")
load("@pypi//:requirements.bzl", "all_whl_requirements")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
load("@rules_python_gazelle_plugin//manifest:defs.bzl", "gazelle_python_manifest")
load("@rules_python_gazelle_plugin//modules_mapping:def.bzl", "modules_mapping")

exports_files(
    [
        "requirements.in",
        "requirements.txt",
        "pyproject.toml",
    ],
    visibility = ["//visibility:public"],
)

# Facts //services/tests:apt_pins_test needs about the apt layers, extracted from MODULE.bazel: the Ubuntu base image's
# build date, which is the oldest snapshot.ubuntu.com pin the manifests may use, and the manifest label of every
# apt.install() repo, so a newly added service cannot go unaudited. An apt layer pinned before the base image was built
# resolves packages the base already ships to lower versions, and the overlay wins at image assembly, so the image
# regresses. Deriving both facts from MODULE.bazel keeps them correct across base image and service changes, and reading
# that file from the package that owns it avoids exposing it as a public file target. An empty extraction fails the
# build on purpose: a base image tag carrying no date means the invariant needs a human decision, not a skipped check.
#
# MODULE.bazel is read via `tools` rather than `srcs`, whose contents are instrumentable, so that `dazel coverage` does
# not pull it into the coverage manifest of the tests that depend on these rules.
genrule(
    name = "apt_pin_facts",
    outs = [
        "ubuntu_base_snapshot_floor.txt",
        "apt_manifest_labels.txt",
    ],
    cmd = """awk '/nvcr.io\\/nvidia\\/base\\/ubuntu:jammy-/ {
        sub(/.*ubuntu:jammy-/, ""); sub(/[^0-9].*/, ""); print $$0 "T000000Z"; exit
    }' $(location MODULE.bazel) > $(location ubuntu_base_snapshot_floor.txt) &&
    awk '/^[[:space:]]*manifest = / {
        sub(/^[[:space:]]*manifest = /, ""); gsub(/"/, ""); sub(/,$$/, ""); print
    }' $(location MODULE.bazel) > $(location apt_manifest_labels.txt) &&
    test -s $(location ubuntu_base_snapshot_floor.txt) &&
    test -s $(location apt_manifest_labels.txt)""",
    tools = ["MODULE.bazel"],
    visibility = ["//services/tests:__pkg__"],
)

# Facts //third_party/tests:vendored_python_test needs about the vendored Python repos, extracted from MODULE.bazel: one
# <kind>\t<attribute>\t<value> line per attribute of interest on every http_archive.archive() and local_wheel.wheel()
# tag. The test compares each declaration against the .dist-info the repo actually ships, so a vendored library cannot
# reach an image without metadata that names and versions it. Reading the declarations rather than hardcoding them means
# a newly vendored repo is audited the moment it is added.
genrule(
    name = "vendored_python_facts",
    outs = ["vendored_python_repos.txt"],
    cmd = """awk '
        /^(http_archive[.]archive|local_wheel[.]wheel)[(]/ { kind = ($$0 ~ /^local_wheel/) ? "wheel" : "archive"; next }
        /^[)]/ { kind = ""; next }
        kind != "" && /^ +(name|src|build_file|dist_info_name|dist_info_root|dist_info_version) = "/ {
            value = $$3; gsub(/[",]/, "", value); print kind "\\t" $$1 "\\t" value
        }' $(location MODULE.bazel) > $@ &&
    test -s $@""",
    tools = ["MODULE.bazel"],
    visibility = ["//third_party/tests:__pkg__"],
)

alias(
    name = "format",
    actual = "//tools/format:format_multirun",
)

alias(
    name = "format.check",
    actual = "//tools/format:format_multirun.check",
)

# pip-compile target for refreshing requirements.txt from requirements.in. Run via:
#     dazel run //:requirements.update    (regenerate)
#     dazel test //:requirements.test     (checks that requirements.txt is up-to-date)
compile_pip_requirements(
    name = "requirements",
    timeout = "long",
    src = "requirements.in",
    extra_args = [
        "--extra-index-url=https://download.pytorch.org/whl/cu128",
        "--no-strip-extras",
        "--quiet",
    ],
    requirements_txt = "requirements.txt",
    tags = [
        "no-coverage",
        "no-mypy",
    ],
)

modules_mapping(
    name = "modules_map",
    include_stub_packages = True,
    wheels = all_whl_requirements,
)

gazelle_python_manifest(
    name = "gazelle_python_manifest",
    modules_mapping = ":modules_map",
    pip_repository_name = "pypi",
    requirements = "//:requirements.txt",
)

gazelle(
    name = "gazelle",
    gazelle = "@rules_python_gazelle_plugin//python:gazelle_binary",
)
