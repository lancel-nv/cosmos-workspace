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

"""
Module extension for vendoring a committed local wheel as a Bazel repository.

Unlike `pip.parse` (which downloads + resolves wheels from an index) this extension extracts a `.whl` file already
checked into the repo (via Git LFS) into its own external repository and overlays a hand-written `BUILD` file. It is
the wheel analog of the `http_archive` extension in this directory: same `build_file`-overlay pattern, but the source
is a local label instead of a download URL.

Used to vendor a custom-built `opencv-python-headless` wheel (compiled `WITH_FFMPEG=OFF` for strict LGPL compliance)
that cannot come from PyPI. See `third_party/opencv/build_wheel.sh`, `third_party/opencv/BUILD.opencv`, and the
`local_wheel.wheel(...)` tag in `MODULE.bazel`.
"""

def _local_wheel_impl(repository_ctx):
    """Repository rule that extracts a committed .whl (a zip archive) and overlays a BUILD file."""

    # Nothing below reads the wheel through a label -- `extract` is handed a symlink created inside the repo -- so Bazel
    # would not otherwise know this rule depends on the wheel's contents. A rebuilt wheel would then leave an
    # already-extracted repo untouched, serving the previous build's modules and metadata indefinitely (a fresh checkout
    # on a machine with a warm output base included, which is most CI runners). Watch the source explicitly.
    repository_ctx.watch(repository_ctx.attr.src)

    # A wheel is a zip archive, but `extract` infers the archive format from the file extension and does not recognize
    # `.whl` (and this Bazel's `extract` has no format override). Symlink the wheel under a `.zip` name so the zip
    # format is inferred, then extract into the repo root so the wheel's top-level `cv2/` package and
    # `opencv_python_headless.libs/` directory land at the repository root (where `imports = ["."]` puts them on
    # sys.path with their relative layout preserved for RUNPATH resolution).
    repository_ctx.symlink(repository_ctx.attr.src, "_wheel.zip")
    repository_ctx.extract(archive = "_wheel.zip", output = "")
    repository_ctx.delete("_wheel.zip")
    repository_ctx.symlink(repository_ctx.attr.build_file, "BUILD.bazel")

_local_wheel = repository_rule(
    implementation = _local_wheel_impl,
    attrs = {
        # Label to the committed .whl file. `attr.label` (not `attr.string`) so the label resolves against the
        # tag-authoring module's repo mapping; see the same rationale on `build_file` below.
        "src": attr.label(mandatory = True, allow_single_file = [".whl"]),
        # Must be `attr.label` (not `attr.string`) so bzlmod resolves the label against the *tag-authoring* module's
        # repo mapping at tag-parse time. Otherwise a relative label written in a consumer module would be
        # re-resolved against the extension-defining module's mapping (i.e. `cosmos_eval`) when the rule
        # materializes, silently pointing at the wrong file.
        "build_file": attr.label(mandatory = True, allow_single_file = True),
    },
)

def _local_wheel_extension_impl(module_ctx):
    for mod in module_ctx.modules:
        for wheel in mod.tags.wheel:
            _local_wheel(
                name = wheel.name,
                src = wheel.src,
                build_file = wheel.build_file,
            )

    return module_ctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

_wheel_tag = tag_class(
    attrs = {
        "name": attr.string(mandatory = True),
        "src": attr.label(mandatory = True, allow_single_file = [".whl"]),
        "build_file": attr.label(mandatory = True, allow_single_file = True),
    },
)

local_wheel = module_extension(
    implementation = _local_wheel_extension_impl,
    tag_classes = {"wheel": _wheel_tag},
)
