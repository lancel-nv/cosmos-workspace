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

load("@rules_pkg//pkg:mappings.bzl", "pkg_files", "strip_prefix")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")

# The base orchestration schema, staged into the shared schema directory that //database's tables_manifest.sh reads.
# Adding a file here ships it, but does not apply it -- the applied set and its order come from
# //database:tables_manifest.txt, which a consuming image may override with its own.
#
# pkg_files rather than pkg_tar's own package_dir, so the schemas land 0644 instead of pkg_tar's default 0555 for
# bare srcs.
# The schema as plain files, for tests that apply it to an ephemeral server rather than baking it into an image.
# Public because a dependent module's tests need it: a schema with foreign keys into the orchestration tables can
# only be exercised by a test that applies this one alongside its own.
filegroup(
    name = "schemas",
    srcs = ["orchestration.sql"],
    visibility = ["//visibility:public"],
)

pkg_files(
    name = "schema_files",
    srcs = ["orchestration.sql"],
    prefix = "/etc/postgres-schemas",
    strip_prefix = strip_prefix.from_pkg(),
    visibility = ["//visibility:public"],
)

pkg_tar(
    name = "schema_layer",
    srcs = [":schema_files"],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)
