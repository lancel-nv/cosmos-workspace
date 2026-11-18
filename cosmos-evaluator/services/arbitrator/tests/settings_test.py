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

"""Unit tests for the client-agnostic Arbitrator Settings.

The module's settings resolve checker endpoints from the *required* ``COSMOS_EVALUATOR_CHECKER_ENDPOINTS`` JSON env var.
"""

import json
import os
import tempfile
import unittest
from unittest import mock
from unittest.mock import MagicMock

from pydantic import ValidationError

from services.arbitrator.dependency_graph import DependencyGraph
from services.arbitrator.settings import Settings, _redact_url_for_log, get_settings
from services.settings_base import LogLevel
from utils.environment import Environment

# A valid JSON object string for COSMOS_EVALUATOR_CHECKER_ENDPOINTS.
_ENDPOINTS_JSON = json.dumps({"av.objects_single_view": "http://localhost:8082"})


def _base_env(**overrides: str) -> dict[str, str]:
    """Build a minimal environment with the required checker-endpoints var set, plus any overrides."""
    env = {"COSMOS_EVALUATOR_CHECKER_ENDPOINTS": _ENDPOINTS_JSON}
    env.update(overrides)
    return env


class TestCheckerEndpointsEnvRequired(unittest.TestCase):
    """The base settings require the JSON checker-endpoints env var."""

    @mock.patch("services.arbitrator.settings.Settings._load_dependency_graph", return_value=MagicMock())
    def test_missing_env_raises(self, _mock_graph: MagicMock) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, "COSMOS_EVALUATOR_CHECKER_ENDPOINTS is required"),
        ):
            Settings()

    @mock.patch("services.arbitrator.settings.Settings._load_dependency_graph", return_value=MagicMock())
    def test_valid_json_is_parsed(self, _mock_graph: MagicMock) -> None:
        with mock.patch.dict(os.environ, _base_env(), clear=True):
            s = Settings()
            self.assertEqual(s.checker_endpoints, {"av.objects_single_view": "http://localhost:8082"})

    @mock.patch("services.arbitrator.settings.Settings._load_dependency_graph", return_value=MagicMock())
    def test_invalid_json_raises(self, _mock_graph: MagicMock) -> None:
        with (
            mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_CHECKER_ENDPOINTS": "{not json"}, clear=True),
            self.assertRaisesRegex(ValueError, "must be a JSON object string"),
        ):
            Settings()

    @mock.patch("services.arbitrator.settings.Settings._load_dependency_graph", return_value=MagicMock())
    def test_non_object_json_raises(self, _mock_graph: MagicMock) -> None:
        with (
            mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_CHECKER_ENDPOINTS": "[1, 2]"}, clear=True),
            self.assertRaisesRegex(ValueError, "non-empty JSON object"),
        ):
            Settings()

    @mock.patch("services.arbitrator.settings.Settings._load_dependency_graph", return_value=MagicMock())
    def test_non_string_value_raises(self, _mock_graph: MagicMock) -> None:
        with (
            mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_CHECKER_ENDPOINTS": json.dumps({"a": 1})}, clear=True),
            self.assertRaisesRegex(ValueError, "must be a string URL"),
        ):
            Settings()


class TestEnvironmentDefaults(unittest.TestCase):
    """Tests for environment + simple scalar fields."""

    @mock.patch("services.arbitrator.settings.Settings._load_dependency_graph", return_value=MagicMock())
    def test_defaults(self, _mock_graph: MagicMock) -> None:
        with mock.patch.dict(os.environ, _base_env(), clear=True):
            s = Settings()
            self.assertEqual(s.env, Environment.LOCAL)
            self.assertEqual(s.log_level, LogLevel.INFO)
            self.assertEqual(s.db_env, Environment.LOCAL)
            self.assertEqual(s.database_api_key, "")
            self.assertEqual(s.database_api_url, "")
            self.assertIsNone(s.queue_capacity)

    @mock.patch("services.arbitrator.settings.Settings._load_dependency_graph", return_value=MagicMock())
    def test_env_override_production(self, _mock_graph: MagicMock) -> None:
        with mock.patch.dict(
            os.environ, _base_env(COSMOS_EVALUATOR_ENV="production", COSMOS_EVALUATOR_LOG_LEVEL="DEBUG"), clear=True
        ):
            s = Settings()
            self.assertEqual(s.env, Environment.PROD)
            self.assertEqual(s.log_level, LogLevel.DEBUG)
            self.assertEqual(s.db_env, Environment.PROD)

    @mock.patch("services.arbitrator.settings.Settings._load_dependency_graph", return_value=MagicMock())
    def test_database_api_key_from_env(self, _mock_graph: MagicMock) -> None:
        with mock.patch.dict(os.environ, _base_env(ARBITRATOR_DATABASE_API_KEY="test-key-123"), clear=True):
            s = Settings()
            self.assertEqual(s.database_api_key, "test-key-123")

    @mock.patch("services.arbitrator.settings.Settings._load_dependency_graph", return_value=MagicMock())
    def test_database_api_url_from_env(self, _mock_graph: MagicMock) -> None:
        with mock.patch.dict(os.environ, _base_env(COSMOS_EVALUATOR_DATABASE_API_URL="http://db:8087"), clear=True):
            s = Settings()
            self.assertEqual(s.database_api_url, "http://db:8087")


class TestQueueCapacity(unittest.TestCase):
    """Tests for the optional ``queue_capacity`` field."""

    @mock.patch("services.arbitrator.settings.Settings._load_dependency_graph", return_value=MagicMock())
    def test_reads_from_env_var(self, _mock_graph: MagicMock) -> None:
        with mock.patch.dict(os.environ, _base_env(ARBITRATOR_QUEUE_CAPACITY="8"), clear=True):
            s = Settings()
            self.assertEqual(s.queue_capacity, 8)

    @mock.patch("services.arbitrator.settings.Settings._load_dependency_graph", return_value=MagicMock())
    def test_rejects_zero(self, _mock_graph: MagicMock) -> None:
        with (
            mock.patch.dict(os.environ, _base_env(ARBITRATOR_QUEUE_CAPACITY="0"), clear=True),
            self.assertRaises(ValidationError),
        ):
            Settings()

    @mock.patch("services.arbitrator.settings.Settings._load_dependency_graph", return_value=MagicMock())
    def test_rejects_negative(self, _mock_graph: MagicMock) -> None:
        with (
            mock.patch.dict(os.environ, _base_env(ARBITRATOR_QUEUE_CAPACITY="-5"), clear=True),
            self.assertRaises(ValidationError),
        ):
            Settings()


class TestGetSettingsSingleton(unittest.TestCase):
    """Tests for the cached get_settings accessor."""

    @mock.patch("services.arbitrator.settings.Settings._load_dependency_graph", return_value=MagicMock())
    @mock.patch("services.settings_base.SettingsBase.get_env_files", return_value=[])
    def test_get_settings_singleton(self, _mock_env: MagicMock, _mock_graph: MagicMock) -> None:
        with mock.patch.dict(os.environ, _base_env(), clear=True):
            get_settings.cache_clear()
            s1 = get_settings()
            s2 = get_settings()
            self.assertIs(s1, s2)
            self.assertIsInstance(s1, Settings)


# A small in-memory checker config standing in for ``checks/cosmos_evaluator.yaml`` (avoids a runfiles dependency).
_GRAPH_CONFIG = {"av.objects_single_view": {"enabled": True, "deps": []}}


class TestDependencyGraphLoading(unittest.TestCase):
    """The dependency graph is built and validated from the checker config at construction time."""

    def test_loads_and_validates_graph(self) -> None:
        # No mock of ``_load_dependency_graph``: exercise ``_dependency_graph_class`` + ``_load_dependency_graph``
        # (and the ``dependency_graph`` property) for real, feeding a small in-memory config.
        with (
            mock.patch.dict(os.environ, _base_env(), clear=True),
            mock.patch.object(Settings, "_dependency_graph_config", return_value=dict(_GRAPH_CONFIG)),
        ):
            graph = Settings().dependency_graph
        self.assertIsInstance(graph, DependencyGraph)
        self.assertIn("av.objects_single_view", graph.get_all_deps())

    def _settings_with_stubbed_graph(self) -> Settings:
        with (
            mock.patch.dict(os.environ, _base_env(), clear=True),
            mock.patch.object(Settings, "_load_dependency_graph", return_value=MagicMock()),
        ):
            return Settings()

    def test_config_loader_reads_yaml_from_runfiles(self) -> None:
        s = self._settings_with_stubbed_graph()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("av.objects_single_view:\n  enabled: true\n  deps: []\n")
            path = f.name
        try:
            with mock.patch("services.arbitrator.settings.get_runfiles_path", return_value=path):
                config = s._dependency_graph_config()
            self.assertIn("av.objects_single_view", config)
        finally:
            os.unlink(path)

    def test_config_loader_raises_when_file_missing(self) -> None:
        s = self._settings_with_stubbed_graph()
        with (
            mock.patch("services.arbitrator.settings.get_runfiles_path", return_value=None),
            self.assertRaisesRegex(FileNotFoundError, "cosmos_evaluator.yaml not found"),
        ):
            s._dependency_graph_config()

    def test_config_loader_raises_when_not_a_mapping(self) -> None:
        s = self._settings_with_stubbed_graph()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("- a\n- b\n")
            path = f.name
        try:
            with (
                mock.patch("services.arbitrator.settings.get_runfiles_path", return_value=path),
                self.assertRaisesRegex(ValueError, "non-empty mapping"),
            ):
                s._dependency_graph_config()
        finally:
            os.unlink(path)


class TestRedactUrlForLog(unittest.TestCase):
    """``_redact_url_for_log`` keeps scheme/host/port/path and drops userinfo + query without ever raising."""

    def test_strips_userinfo_and_query(self) -> None:
        self.assertEqual(_redact_url_for_log("https://user:pass@host:8087/api?token=abc"), "https://host:8087/api")

    def test_keeps_path_without_port(self) -> None:
        self.assertEqual(_redact_url_for_log("http://db.internal/path"), "http://db.internal/path")

    def test_malformed_port_is_omitted(self) -> None:
        # ``urlsplit(...).port`` raises ``ValueError`` on a non-numeric port; the helper must swallow it.
        self.assertEqual(_redact_url_for_log("http://host:notaport/path"), "http://host/path")


if __name__ == "__main__":
    unittest.main()
