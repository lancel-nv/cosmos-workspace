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

"""Anchor binary for the shared GPU core OCI layer.

This module exists only so that the //services:gpu_core_deps py_binary depends on torch, which materializes the
byte-identical torch/triton/CUDA wheel closure in its runfiles. //services:gpu_core (gpu_core_layer) then packages that
closure into the shared gpu_core_layer reused across every GPU image. The ``import torch`` below is what pulls in that
closure: it is a real, statically analyzable dependency (so Gazelle keeps @pypi//torch on the target automatically),
not a hidden or build-only one. The binary is not run in production; main() merely confirms the import resolved.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def main() -> None:
    """Confirm torch imported. The binary is a build-time runfiles anchor for the GPU core layer, not a service."""
    logger.info("gpu_core_deps anchor: torch %s import OK", torch.__version__)


if __name__ == "__main__":
    main()
