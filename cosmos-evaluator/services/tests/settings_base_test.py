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

"""Unit tests for services/settings.py."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from pydantic import AliasChoices, ValidationError
from pydantic_settings import SettingsConfigDict

from checks.utils.video_codec import (
    DEFAULT_VIDEO_CODEC,
    VIDEO_CODEC_ENV_VAR,
    VIDEO_QUALITY_ENV_VAR,
    VideoCodec,
    VideoQuality,
)
from services.settings_base import GLOBAL_ONLY_FIELDS, LogLevel, S3Credentials, SettingsBase
from utils.environment import Environment


class _FakeServiceSettings(SettingsBase):
    """Test-only subclass used to exercise the per-service alias auto-injection."""

    model_config = SettingsConfigDict(
        strict=True, extra="ignore", env_prefix="FAKE_", case_sensitive=False, frozen=True
    )


class TestEnvironmentEnum(unittest.TestCase):
    """Tests for the Environment enum."""

    def test_local_value(self) -> None:
        """Test Environment.LOCAL has correct value."""
        self.assertEqual(Environment.LOCAL.value, "local")

    def test_prod_value(self) -> None:
        """Test Environment.PROD has correct value."""
        self.assertEqual(Environment.PROD.value, "production")

    def test_staging_value(self) -> None:
        """Test Environment.STAGING has correct value."""
        self.assertEqual(Environment.STAGING.value, "staging")

    def test_dev_value(self) -> None:
        """Test Environment.DEV has correct value."""
        self.assertEqual(Environment.DEV.value, "dev")

    def test_environment_is_string_enum(self) -> None:
        """Test Environment is both str and Enum."""
        self.assertIsInstance(Environment.PROD, str)
        self.assertEqual(Environment.PROD, "production")
        self.assertIsInstance(Environment.LOCAL, str)
        self.assertEqual(Environment.LOCAL, "local")
        self.assertIsInstance(Environment.STAGING, str)
        self.assertEqual(Environment.STAGING, "staging")
        self.assertIsInstance(Environment.DEV, str)
        self.assertEqual(Environment.DEV, "dev")

    def test_environment_from_string(self) -> None:
        """Test Environment can be created from string."""
        self.assertEqual(Environment("local"), Environment.LOCAL)
        self.assertEqual(Environment("production"), Environment.PROD)
        self.assertEqual(Environment("staging"), Environment.STAGING)
        self.assertEqual(Environment("dev"), Environment.DEV)


class TestLogLevelEnum(unittest.TestCase):
    """Tests for the LogLevel enum."""

    def test_debug_value(self) -> None:
        """Test LogLevel.DEBUG has correct value."""
        self.assertEqual(LogLevel.DEBUG.value, "DEBUG")

    def test_info_value(self) -> None:
        """Test LogLevel.INFO has correct value."""
        self.assertEqual(LogLevel.INFO.value, "INFO")

    def test_warning_value(self) -> None:
        """Test LogLevel.WARNING has correct value."""
        self.assertEqual(LogLevel.WARNING.value, "WARNING")

    def test_error_value(self) -> None:
        """Test LogLevel.ERROR has correct value."""
        self.assertEqual(LogLevel.ERROR.value, "ERROR")


class TestSettingsDefaults(unittest.TestCase):
    """Tests for SettingsBase default values."""

    def test_default_env_is_local(self) -> None:
        """Test default environment is LOCAL."""
        s = SettingsBase()
        self.assertEqual(s.env, Environment.LOCAL)

    def test_default_log_level_is_info(self) -> None:
        """Test default log level is INFO."""
        s = SettingsBase()
        self.assertEqual(s.log_level, LogLevel.INFO)

    def test_default_storage_fields_are_unset(self) -> None:
        """Test default storage credential and address fields are unset."""
        s = SettingsBase()
        self.assertIsNone(s.storage_access_key)
        self.assertIsNone(s.storage_secret_key)
        self.assertEqual(s.storage_url, "")
        self.assertEqual(s.input_storage_url, "")
        self.assertEqual(s.output_storage_url, "")


class TestSettingsConstructor(unittest.TestCase):
    """Tests for SettingsBase constructor arguments."""

    def test_env_set_to_production(self) -> None:
        """Test setting env to production via constructor."""
        s = SettingsBase(env=Environment.PROD)
        self.assertEqual(s.env, Environment.PROD)

    def test_env_set_to_staging(self) -> None:
        """Test setting env to staging via constructor."""
        s = SettingsBase(env=Environment.STAGING)
        self.assertEqual(s.env, Environment.STAGING)

    def test_log_level_set_to_debug(self) -> None:
        """Test setting log_level to DEBUG via constructor."""
        s = SettingsBase(log_level=LogLevel.DEBUG)
        self.assertEqual(s.log_level, LogLevel.DEBUG)

    def test_log_level_set_to_error(self) -> None:
        """Test setting log_level to ERROR via constructor."""
        s = SettingsBase(log_level=LogLevel.ERROR)
        self.assertEqual(s.log_level, LogLevel.ERROR)

    def test_invalid_env_value_raises_error(self) -> None:
        """Test that invalid environment value raises ValidationError."""
        with self.assertRaises(ValidationError):
            SettingsBase(env="invalid_env_value")

    def test_invalid_log_level_raises_error(self) -> None:
        """Test that invalid log level raises ValidationError."""
        with self.assertRaises(ValidationError):
            SettingsBase(log_level="INVALID")

    def test_storage_fields_can_be_set(self) -> None:
        """Test storage fields can be set via constructor."""
        s = SettingsBase(
            storage_access_key="dummy_key",
            storage_secret_key="dummy_secret",
            storage_url="s3://my-bucket",
        )
        self.assertEqual(s.storage_access_key, "dummy_key")
        self.assertEqual(s.storage_secret_key, "dummy_secret")
        self.assertEqual(s.storage_url, "s3://my-bucket")


class TestStorageCredentialsFromEnv(unittest.TestCase):
    """Tests for storage credentials loaded from environment variables."""

    def test_storage_access_key_from_new_env(self) -> None:
        """Test COSMOS_EVALUATOR_STORAGE_ACCESS_KEY env var sets storage_access_key field."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_STORAGE_ACCESS_KEY": "test_key"}, clear=True):
            s = SettingsBase()
            self.assertEqual(s.storage_access_key, "test_key")

    def test_storage_secret_key_from_new_env(self) -> None:
        """Test COSMOS_EVALUATOR_STORAGE_SECRET_KEY env var sets storage_secret_key field."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_STORAGE_SECRET_KEY": "test_secret"}, clear=True):
            s = SettingsBase()
            self.assertEqual(s.storage_secret_key, "test_secret")

    def test_storage_session_token_from_new_env(self) -> None:
        """Test COSMOS_EVALUATOR_STORAGE_SESSION_TOKEN env var sets storage_session_token field."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_STORAGE_SESSION_TOKEN": "test_token"}, clear=True):
            s = SettingsBase()
            self.assertEqual(s.storage_session_token, "test_token")

    def test_ambient_cloud_credentials_are_ignored(self) -> None:
        """Credentials exported for one target must not become the shared credentials used against another."""
        env_vars = {
            "AWS_ACCESS_KEY_ID": "test_access_key_id",
            "AWS_SECRET_ACCESS_KEY": "test_secret",
            "AWS_SESSION_TOKEN": "FwoGZXIvYXdzEExampleToken",
            "AWS_DEFAULT_REGION": "us-west-2",
            "AZURE_STORAGE_KEY": "azure_key",
            "AZURE_STORAGE_SAS_TOKEN": "sv=2024&sig=abc",
        }
        with mock.patch.dict(os.environ, env_vars, clear=True):
            s = SettingsBase()
            self.assertIsNone(s.storage_access_key)
            self.assertIsNone(s.storage_secret_key)
            self.assertIsNone(s.storage_session_token)

    def test_s3_credentials_read_aws_variables_from_env_file(self) -> None:
        """boto3 never sees an env file, so an ``~/.aws/.env``-style file must reach S3 through the settings."""
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("AWS_ACCESS_KEY_ID=AKIAFILE\nAWS_SECRET_ACCESS_KEY=filesecret\n")
            with mock.patch.dict(os.environ, {}, clear=True):
                s = SettingsBase(_env_file=[env_file])

        self.assertIsNone(s.storage_access_key)
        self.assertEqual(s.s3_credentials, S3Credentials("AKIAFILE", "filesecret", None))

    def test_s3_credentials_prefer_storage_fields_as_a_set(self) -> None:
        env_vars = {
            "COSMOS_EVALUATOR_STORAGE_ACCESS_KEY": "AKIAEXPLICIT",
            "COSMOS_EVALUATOR_STORAGE_SECRET_KEY": "explicitsecret",
            "AWS_ACCESS_KEY_ID": "AKIAAMBIENT",
            "AWS_SECRET_ACCESS_KEY": "ambientsecret",
            "AWS_SESSION_TOKEN": "ambienttoken",
        }
        with mock.patch.dict(os.environ, env_vars, clear=True):
            s = SettingsBase()

        self.assertEqual(s.s3_credentials, S3Credentials("AKIAEXPLICIT", "explicitsecret", None))

    def test_all_storage_credentials_from_env(self) -> None:
        """Test all storage credentials can be loaded from environment variables."""
        env_vars = {
            "COSMOS_EVALUATOR_STORAGE_ACCESS_KEY": "test_access_key_id",
            "COSMOS_EVALUATOR_STORAGE_SECRET_KEY": "test_secret_access_key",
            "COSMOS_EVALUATOR_STORAGE_SESSION_TOKEN": "test_session_token",
        }
        with mock.patch.dict(os.environ, env_vars, clear=True):
            s = SettingsBase()
            self.assertEqual(s.storage_access_key, "test_access_key_id")
            self.assertEqual(s.storage_secret_key, "test_secret_access_key")
            self.assertEqual(s.storage_session_token, "test_session_token")

    def test_storage_credentials_default_to_none_when_not_set(self) -> None:
        """Test storage credentials default to None when env vars not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            s = SettingsBase()
            self.assertIsNone(s.storage_access_key)
            self.assertIsNone(s.storage_secret_key)
            self.assertIsNone(s.storage_session_token)


class TestStorageTargets(unittest.TestCase):
    """Tests for the per-direction storage targets parsed from the storage URLs."""

    def test_both_directions_share_the_base_url(self) -> None:
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_STORAGE_URL": "s3://my-bucket"}, clear=True):
            s = SettingsBase()
            self.assertEqual(s.input_storage_target.container, "my-bucket")
            self.assertEqual(s.output_storage_target.container, "my-bucket")

    def test_per_direction_urls_override_the_base(self) -> None:
        env_vars = {
            "COSMOS_EVALUATOR_STORAGE_URL": "s3://base-bucket",
            "COSMOS_EVALUATOR_INPUT_STORAGE_URL": "s3://input-bucket",
            "COSMOS_EVALUATOR_OUTPUT_STORAGE_URL": "s3://output-bucket",
        }
        with mock.patch.dict(os.environ, env_vars, clear=True):
            s = SettingsBase()
            self.assertEqual(s.input_storage_target.container, "input-bucket")
            self.assertEqual(s.output_storage_target.container, "output-bucket")

    def test_target_reflects_the_direction_storage_type(self) -> None:
        """The URL is parsed against its own direction's backend, not a single global one."""
        env_vars = {
            "COSMOS_EVALUATOR_INPUT_STORAGE_TYPE": "s3",
            "COSMOS_EVALUATOR_INPUT_STORAGE_URL": "https://my-bucket.s3.us-west-2.amazonaws.com",
            "COSMOS_EVALUATOR_OUTPUT_STORAGE_TYPE": "azure",
            "COSMOS_EVALUATOR_OUTPUT_STORAGE_URL": "https://myacct.blob.core.windows.net/my-container",
        }
        with mock.patch.dict(os.environ, env_vars, clear=True):
            s = SettingsBase()
            self.assertEqual(s.input_storage_target.region, "us-west-2")
            self.assertIsNone(s.input_storage_target.account)
            self.assertEqual(s.output_storage_target.account, "myacct")
            self.assertEqual(s.output_storage_target.container, "my-container")

    def test_unset_url_raises(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            s = SettingsBase()
        with self.assertRaises(ValueError):
            _ = s.input_storage_target
        with self.assertRaises(ValueError):
            _ = s.output_storage_target


class TestPerServicePrefixOverride(unittest.TestCase):
    """Tests for the auto-injected ``<SERVICE_PREFIX><FIELD_NAME>`` aliases.

    Whenever a ``SettingsBase`` subclass declares its own ``env_prefix``
    (other than the parent ``COSMOS_EVALUATOR_``), every aliased field gets a
    new highest-priority alias so each service can retarget any setting
    individually without losing the global ``COSMOS_EVALUATOR_*`` knobs.
    """

    def test_input_storage_type_picks_up_service_prefix(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_INPUT_STORAGE_TYPE": "local"}, clear=True):
            self.assertEqual(_FakeServiceSettings().input_storage_type, "local")

    def test_output_storage_type_picks_up_service_prefix(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_OUTPUT_STORAGE_TYPE": "local"}, clear=True):
            self.assertEqual(_FakeServiceSettings().output_storage_type, "local")

    def test_storage_url_picks_up_service_prefix(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_STORAGE_URL": "s3://svc-bucket"}, clear=True):
            self.assertEqual(_FakeServiceSettings().storage_url, "s3://svc-bucket")

    def test_storage_access_key_picks_up_service_prefix(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_STORAGE_ACCESS_KEY": "svc-key"}, clear=True):
            self.assertEqual(_FakeServiceSettings().storage_access_key, "svc-key")

    def test_storage_secret_key_picks_up_service_prefix(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_STORAGE_SECRET_KEY": "svc-secret"}, clear=True):
            self.assertEqual(_FakeServiceSettings().storage_secret_key, "svc-secret")

    def test_storage_session_token_picks_up_service_prefix(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_STORAGE_SESSION_TOKEN": "svc-token"}, clear=True):
            self.assertEqual(_FakeServiceSettings().storage_session_token, "svc-token")

    def test_env_picks_up_service_prefix(self) -> None:
        """Generalization check - non-storage aliased fields participate too."""
        with mock.patch.dict(os.environ, {"FAKE_ENV": "production"}, clear=True):
            self.assertEqual(_FakeServiceSettings().env, Environment.PROD)

    def test_log_level_picks_up_service_prefix(self) -> None:
        """Generalization check - non-storage aliased fields participate too."""
        with mock.patch.dict(os.environ, {"FAKE_LOG_LEVEL": "DEBUG"}, clear=True):
            self.assertEqual(_FakeServiceSettings().log_level, LogLevel.DEBUG)

    def test_service_prefix_beats_cosmos_evaluator_specific_alias(self) -> None:
        """``FAKE_INPUT_STORAGE_TYPE`` outranks ``COSMOS_EVALUATOR_INPUT_STORAGE_TYPE``."""
        env = {
            "FAKE_INPUT_STORAGE_TYPE": "local",
            "COSMOS_EVALUATOR_INPUT_STORAGE_TYPE": "s3",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(_FakeServiceSettings().input_storage_type, "local")

    def test_service_prefix_beats_cosmos_evaluator_umbrella_alias(self) -> None:
        """``FAKE_INPUT_STORAGE_TYPE`` outranks the umbrella ``COSMOS_EVALUATOR_STORAGE_TYPE``."""
        env = {
            "FAKE_INPUT_STORAGE_TYPE": "local",
            "COSMOS_EVALUATOR_STORAGE_TYPE": "s3",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(_FakeServiceSettings().input_storage_type, "local")

    def test_service_prefix_beats_cosmos_evaluator_storage_url(self) -> None:
        """``FAKE_INPUT_STORAGE_URL`` outranks both the specific and the umbrella ``COSMOS_EVALUATOR_*`` names."""
        env = {
            "FAKE_INPUT_STORAGE_URL": "s3://svc-input-bucket",
            "COSMOS_EVALUATOR_INPUT_STORAGE_URL": "s3://input-bucket",
            "COSMOS_EVALUATOR_STORAGE_URL": "s3://base-bucket",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(_FakeServiceSettings().input_storage_url, "s3://svc-input-bucket")

    def test_cosmos_evaluator_aliases_still_work_without_service_prefix(self) -> None:
        """Without a ``FAKE_*`` override, the existing ``COSMOS_EVALUATOR_*`` chain still drives the value."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_INPUT_STORAGE_TYPE": "local"}, clear=True):
            self.assertEqual(_FakeServiceSettings().input_storage_type, "local")
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_STORAGE_TYPE": "local"}, clear=True):
            self.assertEqual(_FakeServiceSettings().input_storage_type, "local")

    def test_settings_base_itself_is_not_mutated(self) -> None:
        """The parent class's alias chain must remain unchanged so ``COSMOS_EVALUATOR_COSMOS_EVALUATOR_*`` never
        appears.
        """
        for field_name, expected in (
            ("env", ("COSMOS_EVALUATOR_ENV", "env")),
            ("log_level", ("COSMOS_EVALUATOR_LOG_LEVEL", "log_level")),
            (
                "input_storage_type",
                ("COSMOS_EVALUATOR_INPUT_STORAGE_TYPE", "COSMOS_EVALUATOR_STORAGE_TYPE", "input_storage_type"),
            ),
        ):
            alias = SettingsBase.model_fields[field_name].validation_alias
            self.assertIsInstance(alias, AliasChoices)
            assert isinstance(alias, AliasChoices)
            self.assertEqual(tuple(alias.choices), expected)

    def test_subclass_alias_chain_starts_with_service_prefix(self) -> None:
        """Confirms the auto-injection actually mutated the subclass's alias choices in declaration order."""
        for field_name in (
            "env",
            "log_level",
            "input_storage_type",
            "output_storage_type",
            "storage_url",
            "input_storage_url",
            "output_storage_url",
            "storage_access_key",
            "storage_secret_key",
            "storage_session_token",
        ):
            alias = _FakeServiceSettings.model_fields[field_name].validation_alias
            self.assertIsInstance(alias, AliasChoices)
            assert isinstance(alias, AliasChoices)
            self.assertEqual(alias.choices[0], f"FAKE_{field_name.upper()}")


class TestGlobalOnlyFields(unittest.TestCase):
    """The opt-out from the injection above, for fields whose value is also read outside the settings object.

    ``checks.utils.video`` reads the two video fields straight from the environment and knows only their global names,
    so a per-service alias would be honored by ``Settings`` and ignored by the encode ladder. Accepting a variable
    that does nothing is worse than not offering it, hence the exclusion -- and hence these tests, since nothing else
    would notice the alias reappearing.
    """

    def test_video_fields_get_no_service_prefixed_alias(self) -> None:
        for field_name in GLOBAL_ONLY_FIELDS:
            with self.subTest(field=field_name):
                alias = _FakeServiceSettings.model_fields[field_name].validation_alias
                self.assertIsInstance(alias, AliasChoices)
                assert isinstance(alias, AliasChoices)
                self.assertNotIn(f"FAKE_{field_name.upper()}", alias.choices)

    def test_video_fields_keep_their_global_alias(self) -> None:
        """Excluded from the *injection*, not from configuration: the global variable must still work on a subclass."""
        for field_name, variable, expected in (
            ("video_codec", VIDEO_CODEC_ENV_VAR, VideoCodec.VP9),
            ("video_quality", VIDEO_QUALITY_ENV_VAR, VideoQuality.HIGHEST),
        ):
            with (
                self.subTest(field=field_name),
                mock.patch.dict(os.environ, {variable: expected.value}, clear=True),
            ):
                self.assertEqual(getattr(_FakeServiceSettings(), field_name), expected)

    def test_a_service_prefixed_video_variable_is_ignored(self) -> None:
        """The trap this guards: ``FAKE_VIDEO_CODEC`` must not silently become the service's codec."""
        with mock.patch.dict(os.environ, {"FAKE_VIDEO_CODEC": "vp9"}, clear=True):
            self.assertEqual(_FakeServiceSettings().video_codec, DEFAULT_VIDEO_CODEC)

    def test_every_excluded_field_exists(self) -> None:
        """A stale name in the frozenset would silently exclude nothing, so pin that they all resolve."""
        for field_name in GLOBAL_ONLY_FIELDS:
            self.assertIn(field_name, SettingsBase.model_fields)


class TestActiveApiKey(unittest.TestCase):
    """Tests for the env-driven ``active_api_key`` resolver."""

    def test_local_resolves_to_none_even_when_keys_set(self) -> None:
        """Local stays open regardless of configured keys."""
        s = SettingsBase(env=Environment.LOCAL, staging_api_key="stg", prod_api_key="prd")
        self.assertIsNone(s.active_api_key)

    def test_dev_resolves_to_none_even_when_keys_set(self) -> None:
        """Dev stays open regardless of configured keys."""
        s = SettingsBase(env=Environment.DEV, staging_api_key="stg", prod_api_key="prd")
        self.assertIsNone(s.active_api_key)

    def test_staging_resolves_to_staging_key(self) -> None:
        """Staging selects the staging key and ignores the prod key."""
        s = SettingsBase(env=Environment.STAGING, staging_api_key="stg", prod_api_key="prd")
        self.assertEqual(s.active_api_key, "stg")

    def test_prod_resolves_to_prod_key(self) -> None:
        """Production selects the prod key and ignores the staging key."""
        s = SettingsBase(env=Environment.PROD, staging_api_key="stg", prod_api_key="prd")
        self.assertEqual(s.active_api_key, "prd")

    def test_staging_without_key_resolves_to_none(self) -> None:
        """An unset staging key leaves staging open."""
        s = SettingsBase(env=Environment.STAGING)
        self.assertIsNone(s.active_api_key)

    def test_prod_without_key_resolves_to_none(self) -> None:
        """An unset prod key leaves prod open."""
        s = SettingsBase(env=Environment.PROD)
        self.assertIsNone(s.active_api_key)

    def test_keys_loaded_from_env(self) -> None:
        """The keys are read from COSMOS_EVALUATOR_STAGING_API_KEY / COSMOS_EVALUATOR_PROD_API_KEY."""
        env = {
            "COSMOS_EVALUATOR_ENV": "staging",
            "COSMOS_EVALUATOR_STAGING_API_KEY": "stg-from-env",
            "COSMOS_EVALUATOR_PROD_API_KEY": "prd-from-env",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            s = SettingsBase()
            self.assertEqual(s.active_api_key, "stg-from-env")


class TestSettingsGetEnvFiles(unittest.TestCase):
    """Tests for SettingsBase.get_env_files method."""

    def test_get_env_files_returns_paths_for_local(self) -> None:
        """Test get_env_files returns paths in local mode."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_ENV": "local"}):
            paths = SettingsBase.get_env_files()
            self.assertEqual(len(paths), 2)
            self.assertTrue(any(".cosmos_evaluator" in str(p) for p in paths))
            self.assertTrue(any(".aws" in str(p) for p in paths))

    def test_get_env_files_returns_empty_for_production(self) -> None:
        """Test get_env_files returns empty list in production mode."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_ENV": "production"}):
            paths = SettingsBase.get_env_files()
            self.assertEqual(paths, [])

    def test_get_env_files_returns_empty_for_staging(self) -> None:
        """Test get_env_files returns empty list in staging mode."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_ENV": "staging"}):
            paths = SettingsBase.get_env_files()
            self.assertEqual(paths, [])

    def test_get_env_files_defaults_to_local(self) -> None:
        """Test get_env_files defaults to local when COSMOS_EVALUATOR_ENV not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            paths = SettingsBase.get_env_files()
            self.assertEqual(len(paths), 2)


if __name__ == "__main__":
    unittest.main()
