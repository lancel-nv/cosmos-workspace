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

"""Integration test for the Objects container image."""

import unittest

from services.objects.tests.image_test_mixin import ImageTestMixin


class TestObjectsImage(ImageTestMixin, unittest.TestCase):
    """Integration tests for the Objects container image."""

    LOADER_SCRIPT_PATH = "services/objects/single_view/tests/test_image_load.sh"
    SERVICE_NAME = "Objects Single-View Service API"


if __name__ == "__main__":
    unittest.main()
