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

import subprocess
import unittest

from testcontainers.core.config import testcontainers_config

from utils.container_images import load_image, load_ryuk_image


class TestLoadRyukImage(unittest.TestCase):
    def test_loads_the_ryuk_image_testcontainers_requests(self) -> None:
        """A testcontainers upgrade that requests a different Ryuk fails here instead of pulling it at run time."""
        self.assertEqual(load_ryuk_image(), testcontainers_config.ryuk_image)

    def test_loaded_ryuk_image_is_present_in_the_docker_daemon(self) -> None:
        tag = load_ryuk_image()
        result = subprocess.run(["docker", "image", "inspect", tag], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


class TestLoadImage(unittest.TestCase):
    def test_returns_the_loaded_tag(self) -> None:
        self.assertEqual(load_image("utils/ryuk_image_load.sh"), testcontainers_config.ryuk_image)

    def test_missing_loader_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_image("utils/no_such_image_load.sh")


if __name__ == "__main__":
    unittest.main()
