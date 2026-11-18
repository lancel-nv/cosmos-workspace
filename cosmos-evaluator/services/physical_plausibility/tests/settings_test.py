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

"""Unit tests for physical-plausibility service settings."""

import os
import unittest
from unittest import mock

from pydantic import ValidationError

from services.physical_plausibility.settings import Settings, get_settings
from services.settings_base import LogLevel
from utils.environment import Environment


class TestSettings(unittest.TestCase):
    def test_defaults(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            settings = Settings()

        self.assertEqual(settings.env, Environment.LOCAL)
        self.assertEqual(settings.log_level, LogLevel.INFO)
        self.assertEqual(settings.process_concurrency_limit, 4)
        self.assertEqual(settings.max_batch_size, 4)
        self.assertEqual(settings.max_batch_wait_ms, 10.0)
        self.assertIsNone(settings.checkpoint_uri)
        self.assertIsNone(settings.config_dir)

    def test_physical_plausibility_environment_overrides(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "PHYSICAL_PLAUSIBILITY_PROCESS_CONCURRENCY_LIMIT": "8",
                "PHYSICAL_PLAUSIBILITY_MAX_BATCH_SIZE": "8",
                "PHYSICAL_PLAUSIBILITY_CHECKPOINT_URI": "/models/checker.pt",
                "PHYSICAL_PLAUSIBILITY_CONFIG_DIR": "/configs",
                "PHYSICAL_PLAUSIBILITY_MAX_BATCH_WAIT_MS": "12.5",
            },
            clear=True,
        ):
            settings = Settings()

        self.assertEqual(settings.process_concurrency_limit, 8)
        self.assertEqual(settings.checkpoint_uri, "/models/checker.pt")
        self.assertEqual(settings.config_dir, "/configs")
        self.assertEqual(settings.max_batch_size, 8)
        self.assertEqual(settings.max_batch_wait_ms, 12.5)

    def test_base_settings_environment_is_case_insensitive(self) -> None:
        with mock.patch.dict(os.environ, {"cosmos_evaluator_log_level": "DEBUG"}, clear=True):
            settings = Settings()

        self.assertEqual(settings.log_level, LogLevel.DEBUG)

    def test_batch_limits_must_be_positive(self) -> None:
        for field in ("process_concurrency_limit", "max_batch_size"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                Settings(**{field: 0})
        with self.assertRaises(ValidationError):
            Settings(max_batch_wait_ms=-0.1)
        with self.assertRaises(ValidationError):
            Settings(process_concurrency_limit=2, max_batch_size=3)


class TestGetSettings(unittest.TestCase):
    def tearDown(self) -> None:
        get_settings.cache_clear()

    def test_get_settings_is_cached(self) -> None:
        get_settings.cache_clear()
        self.assertIsInstance(get_settings(), Settings)
        self.assertIs(get_settings(), get_settings())


if __name__ == "__main__":
    unittest.main()
