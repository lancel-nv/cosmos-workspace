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

"""Unit tests for the local database config loader."""

import unittest
from unittest.mock import patch

from database.db_config import load_db_config


class TestLoadDbConfig(unittest.TestCase):
    """Tests for resolving the ``local`` section of database/config.yaml."""

    def test_resolves_local_section_with_credentials(self) -> None:
        env = {"COSMOS_EVALUATOR_LOCAL_DB_USER": "cosmos_user", "COSMOS_EVALUATOR_LOCAL_DB_PASSWORD": "cosmos_pass"}
        with patch.dict("os.environ", env, clear=False):
            config = load_db_config("local")

        self.assertIsNotNone(config)
        assert config is not None  # narrow for the type checker
        self.assertEqual(config["host"], "localhost")
        self.assertEqual(config["port"], 5432)
        self.assertEqual(config["database"], "local")
        self.assertEqual(config["connect_timeout"], 10)
        self.assertEqual(config["user"], "cosmos_user")
        self.assertEqual(config["password"], "cosmos_pass")
        # The *_env_var indirection keys must not leak into the resolved mapping.
        self.assertNotIn("user_env_var", config)
        self.assertNotIn("password_env_var", config)

    def test_returns_none_when_credentials_missing(self) -> None:
        with patch.dict(
            "os.environ",
            {"COSMOS_EVALUATOR_LOCAL_DB_USER": "", "COSMOS_EVALUATOR_LOCAL_DB_PASSWORD": ""},
            clear=False,
        ):
            self.assertIsNone(load_db_config("local"))

    def test_returns_none_for_unknown_section(self) -> None:
        env = {"COSMOS_EVALUATOR_LOCAL_DB_USER": "u", "COSMOS_EVALUATOR_LOCAL_DB_PASSWORD": "p"}
        with patch.dict("os.environ", env, clear=False):
            self.assertIsNone(load_db_config("does_not_exist"))


if __name__ == "__main__":
    unittest.main()
