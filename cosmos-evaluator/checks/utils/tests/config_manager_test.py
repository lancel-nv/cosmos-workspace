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

"""Unit tests for config_manager module."""

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import yaml

from checks.utils.config_manager import ConfigManager, resolve_check_config


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(config_dir=self.temp_dir)

        # Sample YAML config for testing
        self.sample_config = {
            "av.objects_single_view": {
                "overlap_check": {
                    "vehicle": {"method": "ratio"},
                    "pedestrian": {"method": "cluster"},
                    "motorcycle": {"method": "cluster"},
                },
                "importance_filter": {
                    "distance_threshold_m": 100,
                    "skip_oncoming_objects": False,
                    "relevant_lanes": ["ego", "left", "right"],
                },
            }
        }

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_init_default_config_dir_is_none(self) -> None:
        """With no config_dir, resolution is deferred to load time and _config_dir stays None."""
        cm = ConfigManager()
        self.assertIsNone(cm._config_dir)

    @patch("checks.utils.config_manager.bazel.get_runfiles_path")
    def test_load_config_default_resolves_by_name_from_runfiles(self, mock_get_runfiles_path: MagicMock) -> None:
        """With no config_dir, load_config resolves checks/<name>.yaml from runfiles by config name."""
        config_path = Path(self.temp_dir) / "myconfig.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.sample_config, f)
        mock_get_runfiles_path.return_value = str(config_path)

        result = ConfigManager().load_config("myconfig")

        self.assertEqual(result, self.sample_config)
        mock_get_runfiles_path.assert_called_once_with("checks/myconfig.yaml")

    @patch("checks.utils.config_manager.bazel.get_runfiles_path")
    def test_load_config_default_not_found(self, mock_get_runfiles_path: MagicMock) -> None:
        """With no config_dir, a missing runfiles config raises FileNotFoundError naming the config."""
        mock_get_runfiles_path.return_value = None

        with self.assertRaises(FileNotFoundError) as context:
            ConfigManager().load_config("missing_config")

        self.assertIn("Configuration file not found", str(context.exception))
        self.assertIn("checks/missing_config.yaml", str(context.exception))

    def test_init_custom_config_dir(self) -> None:
        """Test ConfigManager initialization with custom config directory."""
        self.assertEqual(self.config_manager._config_dir, Path(self.temp_dir))

    def test_load_config_success(self) -> None:
        """Test successful config loading."""
        config_path = Path(self.temp_dir) / "test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.sample_config, f)

        result = self.config_manager.load_config("test")
        self.assertEqual(result, self.sample_config)
        self.assertIn("test", self.config_manager._configs)
        self.assertEqual(self.config_manager._configs["test"], self.sample_config)

    def test_load_config_file_not_found(self) -> None:
        """Test config loading when file doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.config_manager.load_config("nonexistent")

    def test_load_config_invalid_yaml(self) -> None:
        """Test config loading with invalid YAML."""
        config_path = Path(self.temp_dir) / "invalid.yaml"
        with open(config_path, "w") as f:
            f.write("invalid: yaml: content: [")

        with self.assertRaises(yaml.YAMLError):
            self.config_manager.load_config("invalid")

    def test_load_config_path_traversal_forward_slash(self) -> None:
        """Test that config names with forward slashes are rejected."""
        with self.assertRaises(ValueError) as context:
            self.config_manager.load_config("../etc/passwd")

        self.assertIn("Invalid config name", str(context.exception))

    def test_load_config_path_traversal_backslash(self) -> None:
        """Test that config names with backslashes are rejected."""
        with self.assertRaises(ValueError) as context:
            self.config_manager.load_config("..\\windows\\system32")

        self.assertIn("Invalid config name", str(context.exception))

    def test_load_config_path_traversal_subdirectory(self) -> None:
        """Test that config names with directory separators are rejected."""
        with self.assertRaises(ValueError) as context:
            self.config_manager.load_config("subdir/config")

        self.assertIn("Invalid config name", str(context.exception))

    def test_load_config_path_traversal_parent_directory(self) -> None:
        """Test that config names with parent directory references are rejected."""
        with self.assertRaises(ValueError) as context:
            self.config_manager.load_config("..config")

        self.assertIn("Invalid config name", str(context.exception))

    def test_load_config_valid_name_with_hyphens_underscores(self) -> None:
        """Test that valid config names with hyphens and underscores work."""
        config_path = Path(self.temp_dir) / "valid-config_name.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"test": "value"}, f)

        result = self.config_manager.load_config("valid-config_name")
        self.assertEqual(result["test"], "value")

    def test_load_config_caching(self) -> None:
        """Test that config is cached and not reloaded."""
        config_path = Path(self.temp_dir) / "cached.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"test": "value"}, f)

        # Load first time
        result1 = self.config_manager.load_config("cached")

        # Modify file
        with open(config_path, "w") as f:
            yaml.dump({"test": "modified"}, f)

        # Load second time - should return cached version
        result2 = self.config_manager.load_config("cached")
        self.assertEqual(result1, result2)
        self.assertEqual(result2["test"], "value")

    def test_reload_config(self) -> None:
        """Test config reloading."""
        config_path = Path(self.temp_dir) / "reload.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"version": 1}, f)

        # Load initial config
        result1 = self.config_manager.load_config("reload")
        self.assertEqual(result1["version"], 1)

        # Modify file
        with open(config_path, "w") as f:
            yaml.dump({"version": 2}, f)

        # Reload config
        result2 = self.config_manager.reload_config("reload")
        self.assertEqual(result2["version"], 2)

    def test_reload_config_path_traversal(self) -> None:
        """Test that reload_config rejects invalid config names."""
        with self.assertRaises(ValueError) as context:
            self.config_manager.reload_config("../../secret/config")

        self.assertIn("Invalid config name", str(context.exception))

    def test_list_configs(self) -> None:
        """Test listing available configs."""
        # Create test config files
        for name in ["config1", "config2", "config3"]:
            config_path = Path(self.temp_dir) / f"{name}.yaml"
            with open(config_path, "w") as f:
                yaml.dump({"test": "data"}, f)

        configs = self.config_manager.list_configs()
        expected_configs = {"config1", "config2", "config3"}
        self.assertEqual(set(configs), expected_configs)

    def test_list_configs_empty_dir(self) -> None:
        """Test listing configs in empty directory."""
        configs = self.config_manager.list_configs()
        self.assertEqual(configs, [])


class TestResolveCheckConfig(unittest.TestCase):
    """Test cases for the shared resolve_check_config helper."""

    def setUp(self) -> None:
        self.defaults = {"av.check": {"stage_a": {"enabled": True}, "stage_b": {"threshold": 0.5}}}

    def test_none_resolves_to_defaults(self) -> None:
        """An omitted config resolves to the default section."""
        resolved = resolve_check_config(None, self.defaults, section_key="av.check")
        self.assertEqual(resolved, self.defaults["av.check"])

    def test_empty_dict_resolves_to_defaults(self) -> None:
        """An empty {} config is treated identically to None."""
        self.assertEqual(
            resolve_check_config({}, self.defaults, section_key="av.check"),
            resolve_check_config(None, self.defaults, section_key="av.check"),
        )

    def test_non_empty_config_replaces_defaults(self) -> None:
        """A non-empty config replaces the defaults instead of being merged over them."""
        resolved = resolve_check_config(
            {"av.check": {"stage_a": {"enabled": False}}}, self.defaults, section_key="av.check"
        )
        self.assertEqual(resolved, {"stage_a": {"enabled": False}})
        self.assertNotIn("stage_b", resolved)

    def test_resolved_section_is_a_copy(self) -> None:
        """Mutating the resolved section must not corrupt the caller's defaults."""
        resolved = resolve_check_config(None, self.defaults, section_key="av.check")
        resolved["stage_a"] = "mutated"
        self.assertEqual(self.defaults["av.check"]["stage_a"], {"enabled": True})

    def test_missing_section_key_raises(self) -> None:
        """A non-empty config lacking the section key is an error, not a silent fallback to defaults."""
        with self.assertRaises(ValueError) as ctx:
            resolve_check_config({"other": 1}, self.defaults, section_key="av.check")
        self.assertIn("av.check", str(ctx.exception))

    def test_missing_required_key_raises(self) -> None:
        """A non-empty section omitting a required key fails up front with the key named."""
        with self.assertRaises(ValueError) as ctx:
            resolve_check_config(
                {"av.check": {"stage_a": {}}},
                self.defaults,
                section_key="av.check",
                required_keys=("stage_a", "stage_b"),
            )
        self.assertIn("stage_b", str(ctx.exception))
        self.assertNotIn("stage_a", str(ctx.exception).split(":")[-1])

    def test_non_mapping_section_raises_value_error(self) -> None:
        """A non-mapping section raises ValueError so the service layer maps it to 422, not 500."""
        with self.assertRaises(ValueError) as ctx:
            resolve_check_config({"av.check": ["not", "a", "dict"]}, self.defaults, section_key="av.check")
        self.assertIn("must be a mapping", str(ctx.exception))

    def test_section_key_none_treats_dicts_as_the_section(self) -> None:
        """Without a section key the config itself is the section (the VLM preset_check shape)."""
        defaults = {"model": {"endpoint": "default"}, "keyframe_width": 640}
        self.assertEqual(resolve_check_config(None, defaults), defaults)
        self.assertEqual(
            resolve_check_config({"model": {"endpoint": "custom"}}, defaults, required_keys=("model",)),
            {"model": {"endpoint": "custom"}},
        )

    def test_section_key_none_missing_required_key_raises(self) -> None:
        """Required-key validation still applies when the config is the section itself."""
        with self.assertRaises(ValueError) as ctx:
            resolve_check_config({"keyframe_width": 800}, {"model": {}}, required_keys=("model",))
        self.assertIn("model", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
