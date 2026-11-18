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

"""Unit tests for the client-agnostic services.database.settings base."""

import os
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from services.database.settings import Settings, get_settings
from utils.environment import Environment


def _make_settings(**overrides: object) -> Settings:
    """Create a base Settings instance (the base performs no connection bootstrap)."""
    return Settings(**overrides)


class TestDefaults(unittest.TestCase):
    """Tests for default field values."""

    def test_default_db_env_is_local(self) -> None:
        s = _make_settings()
        self.assertEqual(s.db_env, Environment.LOCAL)

    def test_default_keys_are_empty(self) -> None:
        s = _make_settings()
        self.assertEqual(s.read_key, "")
        self.assertEqual(s.admin_key, "")

    def test_default_db_config_is_empty_dict(self) -> None:
        s = _make_settings()
        self.assertEqual(s.db_config, {})

    def test_worker_keys_default_to_empty(self) -> None:
        s = _make_settings()
        self.assertEqual(s.worker_keys, [])


class TestWorkerKeysFromEnvironment(unittest.TestCase):
    """Tests for parsing DATABASE_WORKER_KEYS out of the environment.

    These go through the env source rather than init kwargs on purpose: pydantic-settings decodes collection-typed
    fields inside that source, so a field validator alone cannot make a bare or comma-separated value loadable.
    """

    @staticmethod
    def _from_env(raw: str) -> list[str]:
        with patch.dict(os.environ, {"DATABASE_WORKER_KEYS": raw}):
            return Settings().worker_keys

    def test_a_single_bare_key_loads_as_one_key(self) -> None:
        self.assertEqual(self._from_env("kFuTX2ntsyPPMBbmZ9EM8w"), ["kFuTX2ntsyPPMBbmZ9EM8w"])

    def test_a_comma_separated_pair_loads_as_two_keys(self) -> None:
        self.assertEqual(self._from_env("outgoing-key,incoming-key"), ["outgoing-key", "incoming-key"])

    def test_surrounding_whitespace_and_empty_entries_are_dropped(self) -> None:
        self.assertEqual(self._from_env("  first , , second  ,"), ["first", "second"])

    def test_a_json_array_loads_as_its_elements(self) -> None:
        self.assertEqual(self._from_env('["first", "second"]'), ["first", "second"])

    def test_an_empty_value_loads_as_no_keys(self) -> None:
        self.assertEqual(self._from_env(""), [])

    def test_a_malformed_json_array_is_reported_against_the_field(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            self._from_env('["unterminated')
        self.assertIn("worker_keys", str(ctx.exception))


class TestDbConfigInjection(unittest.TestCase):
    """Tests for injecting db_config directly (the client-agnostic path)."""

    def test_injected_db_config_is_preserved(self) -> None:
        config = {"host": "localhost", "port": 5432, "database": "staging", "user": "u", "password": "p"}
        s = _make_settings(db_config=config)
        self.assertEqual(s.db_config, config)


class TestApiKeyValidation(unittest.TestCase):
    """Tests for _validate_api_keys_in_production."""

    def test_local_allows_empty_keys(self) -> None:
        s = _make_settings(db_env=Environment.LOCAL)
        self.assertEqual(s.read_key, "")
        self.assertEqual(s.admin_key, "")

    def test_dev_allows_empty_keys(self) -> None:
        s = _make_settings(db_env=Environment.DEV)
        self.assertEqual(s.read_key, "")

    def test_staging_requires_read_key(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            _make_settings(db_env=Environment.STAGING, read_key="", admin_key="admin-key")
        self.assertIn("DATABASE_READ_KEY", str(ctx.exception))

    def test_staging_requires_admin_key(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            _make_settings(db_env=Environment.STAGING, read_key="read-key", admin_key="")
        self.assertIn("DATABASE_ADMIN_KEY", str(ctx.exception))

    def test_prod_requires_both_keys(self) -> None:
        with self.assertRaises(ValidationError):
            _make_settings(db_env=Environment.PROD, read_key="", admin_key="")

    def test_staging_accepts_both_keys(self) -> None:
        s = _make_settings(db_env=Environment.STAGING, read_key="rk", admin_key="ak")
        self.assertEqual(s.read_key, "rk")
        self.assertEqual(s.admin_key, "ak")

    def test_prod_accepts_both_keys(self) -> None:
        s = _make_settings(db_env=Environment.PROD, read_key="rk", admin_key="ak")
        self.assertEqual(s.db_env, Environment.PROD)


class TestDbEnvAliasing(unittest.TestCase):
    """Tests for db_env field aliasing (strict mode requires enum instances, not strings)."""

    def test_accepts_enum_values(self) -> None:
        s = _make_settings(db_env=Environment.DEV)
        self.assertEqual(s.db_env, Environment.DEV)

    def test_rejects_plain_string(self) -> None:
        with self.assertRaises(ValidationError):
            _make_settings(db_env="staging", read_key="rk", admin_key="ak")


class TestGetSettings(unittest.TestCase):
    """Tests for get_settings caching."""

    def test_returns_same_instance(self) -> None:
        get_settings.cache_clear()
        with patch("services.settings_base.SettingsBase.get_env_files", return_value=[]):
            s1 = get_settings()
            s2 = get_settings()
            self.assertIs(s1, s2)

    def test_cache_clear_creates_new_instance(self) -> None:
        with patch("services.settings_base.SettingsBase.get_env_files", return_value=[]):
            get_settings.cache_clear()
            s1 = get_settings()
            get_settings.cache_clear()
            s2 = get_settings()
            self.assertIsNot(s1, s2)


if __name__ == "__main__":
    unittest.main()
