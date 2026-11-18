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
Module extension for http_archive with dynamic netrc path support.

Wraps Bazel's stock `http_archive` repository rule with two extra capabilities:

- the `netrc` attribute is resolved at fetch time, expanding `~` to `$HOME` (needed for archives whose download URL
  requires netrc-based auth from a credentials file under the user's home directory). Resolution at fetch time — rather
  than at module-resolution time — keeps the extension reproducible and excludes it from `MODULE.bazel.lock`.
- `patches` applies git-style (`-p1`) patches to the extracted tree, so an archive can be fixed up in place instead of
  being forked or shadowed by a hand-maintained copy. Paths in the patch are relative to the repository root, which
  includes `add_prefix` when that is set.
- the `dist_info_*` attributes synthesize a Python `.dist-info` directory for the extracted subset. Source archives
  carry no installed metadata, so without this an SBOM scanner walking the image sees the vendored `.py` files as
  anonymous source with no name, version, or license — the vendored code is simply invisible to attribution tooling.
  The synthesized directory holds `METADATA` (name, version, license expression, upstream download URL), the upstream
  license text under `licenses/`, and an `INSTALLER` naming this extension, and lands on the same `sys.path` root the
  overlay's `imports` attribute exposes, so `importlib.metadata` resolves it too.
"""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "read_netrc", "use_netrc")

# Characters that a distribution name escapes to `_` in a `.dist-info` directory name (PEP 503 / binary distribution
# format): each run of them collapses to a single underscore, and the whole name is lowercased.
_DIST_NAME_SEPARATORS = "-_."

def _escaped_dist_name(name):
    """Escape a distribution name for use in a `.dist-info` directory name."""
    escaped = []
    previous_was_separator = False
    for char in name.lower().elems():
        is_separator = char in _DIST_NAME_SEPARATORS
        if not is_separator:
            escaped.append(char)
        elif not previous_was_separator:
            escaped.append("_")
        previous_was_separator = is_separator
    return "".join(escaped)

def _write_dist_info(repository_ctx):
    """Synthesize a `.dist-info` directory describing the extracted subset of the archive."""
    attr = repository_ctx.attr
    if not attr.dist_info_name:
        if attr.dist_info_version or attr.dist_info_license or attr.dist_info_license_files:
            fail("dist_info_* attributes were set without dist_info_name, so no .dist-info would be written")
        return
    if not attr.dist_info_version:
        fail("dist_info_name was set without dist_info_version; a .dist-info directory name requires both")

    dist_info_dir = "{}-{}.dist-info".format(_escaped_dist_name(attr.dist_info_name), attr.dist_info_version)
    if attr.dist_info_root:
        dist_info_dir = "{}/{}".format(attr.dist_info_root, dist_info_dir)

    metadata = [
        "Metadata-Version: 2.4",
        "Name: {}".format(attr.dist_info_name),
        "Version: {}".format(attr.dist_info_version),
    ]
    if attr.dist_info_summary:
        metadata.append("Summary: {}".format(attr.dist_info_summary))
    if attr.dist_info_homepage:
        metadata.append("Home-page: {}".format(attr.dist_info_homepage))
    metadata.append("Download-URL: {}".format(attr.url))
    if attr.dist_info_license:
        metadata.append("License-Expression: {}".format(attr.dist_info_license))

    for license_file in attr.dist_info_license_files:
        # `License-File` values are paths relative to the project root, and the file itself goes under `licenses/`.
        metadata.append("License-File: {}".format(license_file.name))
        repository_ctx.file(
            "{}/licenses/{}".format(dist_info_dir, license_file.name),
            repository_ctx.read(license_file),
            executable = False,
        )

    # The description records that this is a vendored subset rather than an upstream release artifact, so anyone reading
    # the metadata out of an image knows the version is a pinned commit and the file set is narrower than upstream's.
    metadata += [
        "Description-Content-Type: text/plain",
        "",
        "Vendored subset of {}, extracted and described by the http_archive".format(attr.url),
        "module extension in cosmos_eval. This is not an upstream distribution: the file set is limited to the modules",
        "this build needs, and the local version segment identifies the pinned upstream commit rather than a release.",
    ]

    repository_ctx.file("{}/METADATA".format(dist_info_dir), "\n".join(metadata) + "\n", executable = False)
    repository_ctx.file("{}/INSTALLER".format(dist_info_dir), "cosmos_eval-http_archive\n", executable = False)

def _resolve_netrc_path(netrc, repository_ctx):
    """Resolve netrc path at fetch time, expanding ~ to home directory."""
    home = repository_ctx.os.environ.get("HOME", "")
    if not home:
        fail("HOME environment variable is not set")

    if not netrc:
        return "{}/.netrc".format(home)  # Default to $HOME/.netrc

    return netrc.replace("~", home)  # Replace ~ with home directory

def _http_archive_with_netrc_impl(repository_ctx):
    """Repository rule that resolves netrc paths at fetch time."""

    netrc_path = _resolve_netrc_path(repository_ctx.attr.netrc, repository_ctx)

    auth = {}
    if netrc_path:
        netrc_content = read_netrc(repository_ctx, netrc_path)
        auth = use_netrc(netrc_content, [repository_ctx.attr.url], repository_ctx.attr.auth_patterns)

    output_dir = repository_ctx.attr.add_prefix if repository_ctx.attr.add_prefix else ""
    repository_ctx.download_and_extract(
        url = repository_ctx.attr.url,
        output = output_dir,
        integrity = repository_ctx.attr.integrity if repository_ctx.attr.integrity else None,
        stripPrefix = repository_ctx.attr.strip_prefix,
        auth = auth,
    )

    for patch_file in repository_ctx.attr.patches:
        repository_ctx.patch(patch_file, strip = 1)

    _write_dist_info(repository_ctx)

    if repository_ctx.attr.build_file:
        repository_ctx.symlink(repository_ctx.attr.build_file, "BUILD.bazel")

_http_archive_with_netrc = repository_rule(
    implementation = _http_archive_with_netrc_impl,
    attrs = {
        "url": attr.string(mandatory = True),
        "integrity": attr.string(default = ""),
        "strip_prefix": attr.string(default = ""),
        "add_prefix": attr.string(default = ""),
        "build_file": attr.label(allow_single_file = True),
        "netrc": attr.string(default = "", doc = "Path to netrc file. Supports ~/ for home directory."),
        "auth_patterns": attr.string_dict(default = {}),
        "patches": attr.label_list(default = []),
        "dist_info_name": attr.string(default = ""),
        "dist_info_version": attr.string(default = ""),
        "dist_info_root": attr.string(default = ""),
        "dist_info_license": attr.string(default = ""),
        "dist_info_license_files": attr.label_list(default = [], allow_files = True),
        "dist_info_summary": attr.string(default = ""),
        "dist_info_homepage": attr.string(default = ""),
    },
    environ = ["HOME"],  # Declare env dependency for proper cache invalidation
)

def _http_archive_impl(module_ctx):
    for mod in module_ctx.modules:
        for archive in mod.tags.archive:
            # Pass through config - netrc resolution happens in repository rule
            _http_archive_with_netrc(
                name = archive.name,
                url = archive.url,
                integrity = archive.integrity,
                strip_prefix = archive.strip_prefix,
                add_prefix = archive.add_prefix,
                build_file = archive.build_file,
                netrc = archive.netrc,
                auth_patterns = archive.auth_patterns,
                patches = archive.patches,
                dist_info_name = archive.dist_info_name,
                dist_info_version = archive.dist_info_version,
                dist_info_root = archive.dist_info_root,
                dist_info_license = archive.dist_info_license,
                dist_info_license_files = archive.dist_info_license_files,
                dist_info_summary = archive.dist_info_summary,
                dist_info_homepage = archive.dist_info_homepage,
            )

    return module_ctx.extension_metadata(
        reproducible = True,
        root_module_direct_deps = "all",
        root_module_direct_dev_deps = [],
    )

_archive_tag = tag_class(
    attrs = {
        "name": attr.string(mandatory = True),
        "add_prefix": attr.string(default = ""),
        "auth_patterns": attr.string_dict(default = {}),
        # Must be `attr.label` (not `attr.string`) so bzlmod resolves the label against the *tag-authoring* module's
        # repo mapping at tag-parse time. Otherwise, a relative label like `//third_party:BUILD.foo` written in a
        # consumer module would be re-resolved against the extension-defining module's mapping (i.e. `cosmos_eval`) when
        # `_http_archive_with_netrc` materializes — and silently point at the wrong file. With `attr.label`, the rule
        # receives a fully-qualified Label object.
        "build_file": attr.label(allow_single_file = True),
        # Setting `dist_info_name` turns on synthesis of a `.dist-info` directory for the extracted subset; the rest
        # fill it in. Without it the vendored code carries no installed metadata and attribution tooling cannot see it.
        "dist_info_homepage": attr.string(
            default = "",
            doc = "Upstream project URL, recorded as `Home-page` in METADATA.",
        ),
        "dist_info_license": attr.string(
            default = "",
            doc = "SPDX license expression for the vendored subset, recorded as `License-Expression` in METADATA.",
        ),
        # `attr.label_list` for the same reason as `build_file` above: the license labels must resolve against the
        # tag-authoring module's repo mapping, not the extension-defining module's.
        "dist_info_license_files": attr.label_list(
            default = [],
            allow_files = True,
            doc = "Upstream license texts, copied under `licenses/` in the synthesized `.dist-info` directory.",
        ),
        "dist_info_name": attr.string(
            default = "",
            doc = "Distribution name for the synthesized `.dist-info` directory. Empty disables synthesis.",
        ),
        "dist_info_root": attr.string(
            default = "",
            doc = "Repo-relative directory to write the `.dist-info` into. Must be the `sys.path` root the overlay's " +
                  "`imports` attribute exposes, so `importlib.metadata` can discover the metadata at runtime.",
        ),
        "dist_info_summary": attr.string(
            default = "",
            doc = "One-line description of the vendored subset, recorded as `Summary` in METADATA.",
        ),
        "dist_info_version": attr.string(
            default = "",
            doc = "Version for the synthesized `.dist-info`. Use the upstream version with the pinned commit as a " +
                  "PEP 440 local segment (e.g. `0.6+a1ce2f95`), or `0+<commit>` when upstream declares no version.",
        ),
        "integrity": attr.string(default = ""),
        "netrc": attr.string(default = "", doc = "Path to netrc file. Supports ~/ for home directory."),
        # `attr.label_list` for the same reason as `build_file` above: the patch labels must resolve against the
        # tag-authoring module's repo mapping, not the extension-defining module's.
        "patches": attr.label_list(
            default = [],
            doc = "Patches applied to the extracted tree in order, with `-p1` paths relative to the repository root.",
        ),
        "strip_prefix": attr.string(default = ""),
        "url": attr.string(mandatory = True),
    },
)

http_archive = module_extension(
    implementation = _http_archive_impl,
    tag_classes = {"archive": _archive_tag},
)
