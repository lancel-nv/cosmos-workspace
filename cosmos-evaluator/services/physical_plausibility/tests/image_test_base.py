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

"""Shared container setup for the physical-plausibility image tests.

The same application can be packaged into more than one image, so the credentials and startup budget its model load
needs are defined once here and every image test inherits them; only the image under test differs.
"""

import os

from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from services.objects.tests.image_test_mixin import ImageTestMixin, ObjectsContainer
from utils.container_images import load_image

# Startup is far slower than the mixin's default: the checker eagerly downloads the V-JEPA source from GitHub and the
# ~5.15 GB 384px ViT-L encoder from Meta, SHA-256 verifying each, before the attention bundle is loaded.
STARTUP_TIMEOUT_S = 1800


class PhysicalPlausibilityImageTestBase(ImageTestMixin):
    """Starts a physical-plausibility image with the credentials and startup budget its eager model load needs.

    Not a unittest.TestCase subclass and deliberately not named Test*, so unittest does not discover it directly.
    Concrete tests inherit from this and unittest.TestCase, and set LOADER_SCRIPT_PATH to pick the image.
    """

    SERVICE_NAME = "Physical Plausibility Service API"

    @classmethod
    def setUpClass(cls) -> None:
        """Loads the image and starts the container with HF_TOKEN in its environment.

        Overrides the mixin's setUpClass rather than reusing it: the default attention bundle lives in the gated
        nvidia/physical-plausibility-evaluator repository, so without HF_TOKEN in the container environment the app
        aborts startup on a 401 and no assertion below ever runs. The target's env_inherit carries the token across
        the test boundary.
        """
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN is not set. This image resolves its default attention bundle from the gated "
                "nvidia/physical-plausibility-evaluator repository at startup, so the container cannot boot without "
                "a read-scoped token. Set it in ~/.cosmos_evaluator/.env or export it before running this test."
            )

        cls.image_name = load_image(cls.LOADER_SCRIPT_PATH)
        cls.container = ObjectsContainer(cls.image_name)
        cls.container.with_env("HF_TOKEN", hf_token)
        wait_strategy = LogMessageWaitStrategy("Application startup complete").with_startup_timeout(STARTUP_TIMEOUT_S)
        cls.container.waiting_for(wait_strategy)
        cls.container.start()
