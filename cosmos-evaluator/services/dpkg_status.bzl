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

"""Macro that records a rules_distroless apt overlay in the dpkg database of the base image it is layered onto."""

def merged_dpkg_status(name, base_image, apt_repo, lock, tags = ["manual"], visibility = None):
    """Generate `{name}.tar`, holding the merged /var/lib/dpkg/status and the info files of every overlay package.

    The tar carries the base image's dpkg status with the stanzas of the packages installed from `apt_repo` applied
    over it, plus a `.list` and `.md5sums` file per overlay package built from that package's data layer, so dpkg and
    security scanners attribute every installed file to the package that shipped it. Place the tar in the same layer
    as `<apt_repo>//:packages`, above the base image.

    Args:
        name: Name of the generated genrule.
        base_image: Label of the base image's single-platform OCI layout.
        apt_repo: Apparent name of the rules_distroless `apt.install` repository, e.g. "@objects_packages". It must
            match the `name` given to `apt.install`, because the data layers are matched to lockfile entries through
            the `<name>_<key>` repositories that `apt.install` generates.
        lock: Label of the lockfile passed to that `apt.install`.
        tags: Bazel tags for the genrule. Defaults to ["manual"].
        visibility: Visibility of the genrule.
    """
    tool = Label("//services:merge_dpkg_status")
    native.genrule(
        name = name,
        srcs = [
            base_image,
            lock,
            apt_repo + "//:dpkg_status",
            apt_repo + "//:packages",
        ],
        outs = [name + ".tar"],
        cmd = " ".join([
            "$(execpath {})".format(tool),
            "--repo",
            apt_repo.lstrip("@"),
            "--lock",
            "$(location {})".format(lock),
            "$(location {})".format(base_image),
            "$(location {}//:dpkg_status)".format(apt_repo),
            "$@",
            "$(locations {}//:packages)".format(apt_repo),
        ]),
        tags = tags,
        tools = [tool],
        visibility = visibility,
    )
