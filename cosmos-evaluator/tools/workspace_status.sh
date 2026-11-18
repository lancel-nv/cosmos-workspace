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

# Workspace status script for Bazel stamping
# Variables prefixed with STABLE_ go to stable-status.txt (triggers cache invalidation)
# Variables without STABLE_ prefix go to volatile-status.txt (frequently changing)

### STABLE VALUES - These will invalidate cache when changed

# Allow GIT_SHA to be set explicitly (e.g. a pipeline pinning a specific commit for reproducibility tests); otherwise
# fall back to `git rev-parse HEAD` against the workspace's git tree, and finally to "unknown" so the stamp line is
# never a key with an empty value.
sha="${GIT_SHA:-$(git rev-parse HEAD 2>/dev/null)}"
echo "STABLE_GIT_SHA ${sha:-unknown}"

### VOLATILE VALUES - These change frequently but don't invalidate cache
