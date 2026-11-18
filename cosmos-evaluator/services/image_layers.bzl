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

"""Macro to split a py_binary into separate OCI image layers for pip dependencies and application source code.

Splitting the app layer improves build and container load times by separating large, stable pip packages (~6-7 GiB for
services with PyTorch/CUDA) from small, frequently changing application source code (~100-300 MB). On a typical
code-only change, only the small source layer is rebuilt.

When gpu_deps_pattern is set, pip packages are further split into a GPU dependencies layer (torch, CUDA, triton —
~5-6 GiB, very rarely changing) and a remaining pip dependencies layer (~1-2 GiB), so that routine pip changes don't
invalidate the massive GPU layer.
"""

load("@aspect_bazel_lib//lib:tar.bzl", "mtree_mutate", "mtree_spec", "tar")

# Matches the whole GPU package family in an mtree path (torch, torchvision, nvidia-*, triton, functorch).
# Used to split GPU pip packages away from the small, churny non-GPU pip packages. Matches the bzlmod pip repo segment
# (`rules_python++pip+pypi_<pyver>_<pkg>`) rather than the site-packages dir so packages like nvidia_ml_py are matched
# in full (module + dist-info), not just their dist-info.
GPU_DEPS_PATTERN = "rules_python++pip+pypi_[0-9][0-9]*_\\(torch\\|nvidia\\|triton\\|functorch\\)"

# Matches only the GPU core that is byte-identical across every GPU image: torch (incl. torchgen/functorch in the same
# repo), triton, and the CUDA runtime wheels (nvidia_*_cu12). The trailing `\(/\|_\)` anchors each alternative to a full
# package-name boundary: the package name is followed either by `/` (a bare repo dir like `pypi_<ver>_torch/`) or by `_`
# (the start of a wheel-tag suffix like `triton_cp313_cp313_...` or `nvidia_cuda_runtime_cu12_py3_none_...`). This
# boundary keeps `torch` from matching `torchvision` (followed by a letter, not `/`/`_`) and keeps
# `nvidia_*_cu12` from matching `nvidia_ml_py` (no `_cu12` token). A bare trailing `/` would miss the suffixed triton and
# nvidia_*_cu12 repos entirely, leaking ~5 GiB of CUDA/triton into each service's per-build gpu_extras layer instead of
# the shared core. This is the set extracted into the shared //services:gpu_core_layer.
# NOTE: depends on the rules_python `pypi_<pyver>_<pkg>` repo naming; revisit if rules_python switches to a single-venv
# site-packages layout.
GPU_CORE_PATTERN = "rules_python++pip+pypi_[0-9][0-9]*_\\(torch\\|triton\\|nvidia_[a-z0-9_]*_cu12\\)\\(/\\|_\\)"

def gpu_core_layer(
        name,
        binary,
        package_dir,
        owner,
        ownername,
        tags,
        core_pattern = GPU_CORE_PATTERN,
        runfiles_root_name = "rest_api_deployment",
        visibility = None):
    """Build the shared GPU core OCI layer (torch/triton/CUDA) once, for reuse across every GPU image.

    Produces a single tar target `{name}_layer` containing only the pip packages matching `core_pattern`, taken from
    `binary`'s runfiles. The archive paths are rooted at `{package_dir}/{runfiles_root_name}.runfiles/...` so the layer
    overlays cleanly into the runfiles tree of every consuming image whose deployment binary is named
    `runfiles_root_name` (all checker services name theirs `rest_api_deployment`). The source `binary` may therefore
    have any name — its runfiles-dir segment is rewritten to `runfiles_root_name` in the archive path only (the mtree
    `content=` field points at `external/<repo>/...`, never at `<binary>.runfiles`, so the rewrite cannot corrupt it).

    Args:
        name: Base name for generated targets; the layer is `{name}_layer`.
        binary: Label of a py_binary whose runfiles supply the GPU core wheels (e.g. a torch-only builder binary).
        package_dir: Directory prefix in the tar archive (e.g. "/app").
        owner: UID for file ownership.
        ownername: Username for file ownership.
        tags: Bazel tags for generated targets.
        core_pattern: grep pattern selecting the core pip packages. Defaults to GPU_CORE_PATTERN.
        runfiles_root_name: Deployment binary name of the consuming images; the archive runfiles root is rewritten to
            `{runfiles_root_name}.runfiles`. Defaults to "rest_api_deployment".
        visibility: Visibility for the emitted `{name}_layer` tar (consumed by GPU image oci_image `tars`).
    """
    binary_name = binary.split(":")[-1]

    mtree_spec(
        name = name + "_mtree",
        srcs = [binary],
    )

    # Keep only core packages, then rewrite each archive path to be rooted at `{runfiles_root_name}.runfiles/...`.
    # The leading `^[^ ]*{binary}.runfiles` strips the entire variable path prefix in one step and rewrites the
    # runfiles root, so the layer always lands at the runfiles path every consuming image's deployment binary expects.
    # Doing the strip here (rather than via mtree_mutate strip_prefix, which is a single fixed string) handles any
    # prefix value. The sed is anchored to the first token (the archive path); the `content=` field uses
    # `external/<repo>/...` and is left untouched.
    native.genrule(
        name = name + "_mtree_filter",
        srcs = [name + "_mtree"],
        outs = [name + "_core.mtree"],
        cmd = "grep '%s' $< | sed 's#^[^ ]*%s\\.runfiles#%s.runfiles#' > $@ || true" % (
            core_pattern,
            binary_name,
            runfiles_root_name,
        ),
        tags = tags,
    )

    mtree_mutate(
        name = name + "_core_mtree",
        mtree = name + "_mtree_filter",
        strip_prefix = "",
        owner = owner,
        ownername = ownername,
        package_dir = package_dir,
        tags = tags,
    )

    tar(
        name = name + "_layer",
        srcs = [binary],
        mtree = name + "_core_mtree",
        tags = tags,
        visibility = visibility,
    )

def py_image_layers(
        name,
        binary,
        package_dir = "/app",
        owner = "1001",
        ownername = "appuser",
        tags = ["manual"],
        model_file_pattern = None,
        gpu_deps_pattern = GPU_DEPS_PATTERN,
        extract_shared_gpu_core = True):
    """Split a py_binary's runfiles into separate layers for pip dependencies and application source.

    Produces tar layer targets suitable for use in oci_image(tars = [...]). Which pip layers are emitted depends on
    gpu_deps_pattern and extract_shared_gpu_core (the defaults emit pip_gpu_extras_layer + pip_other_deps_layer):
      {name}_pip_gpu_extras_layer: GPU packages beyond the shared core, e.g. torchvision/nvidia_ml_py
        (default: emitted when gpu_deps_pattern is set and extract_shared_gpu_core is True)
      {name}_pip_other_deps_layer: Remaining non-GPU pip packages (emitted whenever gpu_deps_pattern is set)
      {name}_pip_gpu_deps_layer: Whole GPU family matching gpu_deps_pattern, legacy non-shared-core mode
        (only when gpu_deps_pattern is set and extract_shared_gpu_core is False)
      {name}_pip_deps_layer: All pip packages in a single layer (only when gpu_deps_pattern is None, i.e. no GPU split)
      {name}_source_layer: Application source code and non-pip deps, small and frequently changing
      {name}_model_layer: (only when model_file_pattern is set) Model files matching the pattern

    When gpu_deps_pattern is provided (the default), pip packages are split to separate large, very stable GPU
    libraries (torch, CUDA wheels, triton) from smaller, more frequently updated pip packages, and the single
    pip_deps_layer is not produced. By default (extract_shared_gpu_core = True) the GPU split omits the byte-identical
    core (provided by the shared //services:gpu_core_layer) and emits only pip_gpu_extras_layer; see that arg below.
    Setting gpu_deps_pattern = None disables the GPU split entirely and emits a single pip_deps_layer.

    When model_file_pattern is provided, files matching the pattern are excluded from source_layer and placed into
    model_layer instead. This enables building image variants with or without model weights while sharing all other
    layers.

    Defaults match how every GPU checker image in the repo uses this macro (appuser-owned `/app`, manual tag, full
    GPU split with the shared core extracted), so a standard GPU image only needs `name` + `binary`. A future non-GPU
    image can opt out of the GPU split with `gpu_deps_pattern = None`.

    Args:
        name: Base name for generated targets.
        binary: Label of the py_binary to split.
        package_dir: Directory prefix in the tar archive. Defaults to "/app".
        owner: UID for file ownership. Defaults to "1001" (appuser).
        ownername: Username for file ownership. Defaults to "appuser".
        tags: Bazel tags for generated targets. Defaults to ["manual"].
        model_file_pattern: Optional grep pattern for model files to split into a separate layer.
            When set, matching files are excluded from source_layer and placed in model_layer.
            Example: "model\\.safetensors" to separate model weights.
        gpu_deps_pattern: grep pattern for GPU pip packages to split into a separate layer. Defaults to
            GPU_DEPS_PATTERN (the PyTorch/CUDA family). When set, matching pip packages go into the GPU layer and the
            rest into pip_other_deps_layer. Pass None for a non-GPU image (single pip_deps_layer, no GPU split).
        extract_shared_gpu_core: When True (the default; requires gpu_deps_pattern), the byte-identical GPU core
            (GPU_CORE_PATTERN: torch/triton/CUDA) is NOT emitted here — the image is expected to add the shared
            //services:gpu_core_layer instead. This macro then emits pip_gpu_extras_layer (GPU packages matching
            gpu_deps_pattern but not GPU_CORE_PATTERN, e.g. torchvision/nvidia_ml_py) and the unchanged
            pip_other_deps_layer. When False, the legacy pip_gpu_deps_layer (whole GPU family) is emitted instead.
    """

    mtree_spec(
        name = name + "_mtree",
        srcs = [binary],
    )

    # Normalize each mtree path token to a package-relative layout before splitting, stripping any leading
    # `../<segment>/` so a runfiles tree rooted at `../<segment>/<pkg>/<binary>.runfiles/...` becomes
    # `<pkg>/<binary>.runfiles/...`, which is what the `strip_prefix = native.package_name()` splits below expect. Paths
    # without a `../` prefix are left unchanged, and only the leading path token is rewritten (the `content=` field
    # never starts with `../`).
    native.genrule(
        name = name + "_mtree_normalized",
        srcs = [name + "_mtree"],
        outs = [name + "_normalized.mtree"],
        cmd = "sed 's#^\\.\\./[^/]*/##' $< > $@",
        tags = tags,
    )

    native.genrule(
        name = name + "_pip_mtree_filter",
        srcs = [name + "_mtree_normalized"],
        outs = [name + "_pip_deps.mtree"],
        cmd = "grep 'site-packages' $< > $@ || true",
        tags = tags,
    )

    native.genrule(
        name = name + "_app_mtree_filter",
        srcs = [name + "_mtree_normalized"],
        outs = [name + "_app_source.mtree"],
        cmd = "grep -v 'site-packages' $< > $@ || true",
        tags = tags,
    )

    mtree_mutate(
        name = name + "_pip_mtree",
        mtree = name + "_pip_mtree_filter",
        strip_prefix = native.package_name(),
        owner = owner,
        ownername = ownername,
        package_dir = package_dir,
        tags = tags,
    )

    mtree_mutate(
        name = name + "_app_mtree",
        mtree = name + "_app_mtree_filter",
        strip_prefix = native.package_name(),
        owner = owner,
        ownername = ownername,
        package_dir = package_dir,
        tags = tags,
    )

    if gpu_deps_pattern:
        # The non-GPU pip layer is identical in both modes: everything not matching gpu_deps_pattern.
        native.genrule(
            name = name + "_other_pip_mtree_filter",
            srcs = [name + "_pip_mtree_filter"],
            outs = [name + "_other_pip_deps.mtree"],
            cmd = "grep -v '{}' $< > $@ || true".format(gpu_deps_pattern),
            tags = tags,
        )

        mtree_mutate(
            name = name + "_other_pip_mtree",
            mtree = name + "_other_pip_mtree_filter",
            strip_prefix = native.package_name(),
            owner = owner,
            ownername = ownername,
            package_dir = package_dir,
            tags = tags,
        )

        tar(
            name = name + "_pip_other_deps_layer",
            srcs = [binary],
            mtree = name + "_other_pip_mtree",
            tags = tags,
        )

        if extract_shared_gpu_core:
            # GPU packages beyond the shared core (e.g. torchvision, nvidia_ml_py). The core itself
            # (GPU_CORE_PATTERN) is provided to the image by the shared //services:gpu_core_layer instead.
            native.genrule(
                name = name + "_gpu_extras_mtree_filter",
                srcs = [name + "_pip_mtree_filter"],
                outs = [name + "_gpu_extras.mtree"],
                cmd = "grep '{gpu}' $< | grep -v '{core}' > $@ || true".format(
                    gpu = gpu_deps_pattern,
                    core = GPU_CORE_PATTERN,
                ),
                tags = tags,
            )

            mtree_mutate(
                name = name + "_gpu_extras_mtree",
                mtree = name + "_gpu_extras_mtree_filter",
                strip_prefix = native.package_name(),
                owner = owner,
                ownername = ownername,
                package_dir = package_dir,
                tags = tags,
            )

            tar(
                name = name + "_pip_gpu_extras_layer",
                srcs = [binary],
                mtree = name + "_gpu_extras_mtree",
                tags = tags,
            )
        else:
            native.genrule(
                name = name + "_gpu_pip_mtree_filter",
                srcs = [name + "_pip_mtree_filter"],
                outs = [name + "_gpu_deps.mtree"],
                cmd = "grep '{}' $< > $@ || true".format(gpu_deps_pattern),
                tags = tags,
            )

            mtree_mutate(
                name = name + "_gpu_pip_mtree",
                mtree = name + "_gpu_pip_mtree_filter",
                strip_prefix = native.package_name(),
                owner = owner,
                ownername = ownername,
                package_dir = package_dir,
                tags = tags,
            )

            tar(
                name = name + "_pip_gpu_deps_layer",
                srcs = [binary],
                mtree = name + "_gpu_pip_mtree",
                tags = tags,
            )
    else:
        tar(
            name = name + "_pip_deps_layer",
            srcs = [binary],
            mtree = name + "_pip_mtree",
            tags = tags,
        )

    if model_file_pattern:
        native.genrule(
            name = name + "_source_mtree_filter",
            srcs = [name + "_app_mtree"],
            outs = [name + "_source.mtree"],
            cmd = "grep -v '{}' $< > $@ || true".format(model_file_pattern),
            tags = tags,
        )

        native.genrule(
            name = name + "_model_mtree_filter",
            srcs = [name + "_app_mtree"],
            outs = [name + "_model.mtree"],
            cmd = "grep '{}' $< > $@ || true".format(model_file_pattern),
            tags = tags,
        )

        tar(
            name = name + "_source_layer",
            srcs = [binary],
            mtree = name + "_source_mtree_filter",
            tags = tags,
        )

        tar(
            name = name + "_model_layer",
            srcs = [binary],
            mtree = name + "_model_mtree_filter",
            tags = tags,
        )
    else:
        tar(
            name = name + "_source_layer",
            srcs = [binary],
            mtree = name + "_app_mtree",
            tags = tags,
        )
