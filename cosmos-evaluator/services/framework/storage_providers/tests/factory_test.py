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

"""Tests for the directional storage provider factories."""

import os
import unittest
from unittest import mock

from services.framework.storage_providers.azure_blob_storage_provider import AzureBlobStorageProvider
from services.framework.storage_providers.factory import (
    build_input_storage_provider,
    build_output_storage_provider,
)
from services.framework.storage_providers.local_storage_provider import LocalStorageProvider
from services.framework.storage_providers.s3_storage_provider import S3StorageProvider
from services.settings_base import SettingsBase

# The regional AWS host is the addressing form S3 requires: it names the bucket and the signing region together.
_S3_ENV = {
    "COSMOS_EVALUATOR_STORAGE_URL": "https://my-bucket.s3.us-east-1.amazonaws.com",
}

# Azure takes the container from the same path position an S3 URL takes the bucket from, and the account from the
# hostname. The account key travels in the shared secret-key setting.
_AZURE_ENV = {
    "COSMOS_EVALUATOR_STORAGE_TYPE": "azure",
    "COSMOS_EVALUATOR_STORAGE_URL": "https://myacct.blob.core.windows.net/my-container",
    "COSMOS_EVALUATOR_STORAGE_SECRET_KEY": "accountsecret",
}

_CLUSTER_ENV = {
    "PBSS_AWS_ACCESS_KEY_ID": "pbsskey",
    "PBSS_AWS_SECRET_ACCESS_KEY": "pbsssecret",
    "PBSS_AWS_ENDPOINT_URL": "https://pbss.example.com",
    "PBSS_AWS_DEFAULT_REGION": "us-west-2",
}


class TestBuildInputStorageProvider(unittest.TestCase):
    """Tests for build_input_storage_provider."""

    def test_creates_s3_provider_by_default(self) -> None:
        with mock.patch.dict(os.environ, _S3_ENV, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings)

        assert isinstance(provider, S3StorageProvider)
        self.assertEqual(provider.config.bucket_name, "my-bucket")
        self.assertEqual(provider.config.region_name, "us-east-1")
        self.assertIsNone(provider.config.endpoint_url)
        self.assertEqual(provider.config.key_prefix, "")

    def test_custom_endpoint_url(self) -> None:
        """An S3-compatible cluster is addressed by its endpoint and bucket, and needs no region."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_STORAGE_URL": "https://pbss.s8k.io/my-bucket"}, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings)

        assert isinstance(provider, S3StorageProvider)
        self.assertEqual(provider.config.bucket_name, "my-bucket")
        self.assertEqual(provider.config.endpoint_url, "https://pbss.s8k.io")

    def test_input_storage_type_overrides_base(self) -> None:
        env = {
            **_S3_ENV,
            "COSMOS_EVALUATOR_STORAGE_TYPE": "local",
            "COSMOS_EVALUATOR_INPUT_STORAGE_TYPE": "s3",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings)

        self.assertIsInstance(provider, S3StorageProvider)

    def test_input_storage_url_overrides_base(self) -> None:
        env = {
            **_S3_ENV,
            "COSMOS_EVALUATOR_INPUT_STORAGE_URL": "https://input-bucket.s3.eu-west-1.amazonaws.com",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings)

        assert isinstance(provider, S3StorageProvider)
        self.assertEqual(provider.config.bucket_name, "input-bucket")
        self.assertEqual(provider.config.region_name, "eu-west-1")

    def test_falls_back_to_base_storage_type(self) -> None:
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_STORAGE_TYPE": "local"}, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings, key_prefix="/tmp/inputs")

        self.assertIsInstance(provider, LocalStorageProvider)

    def test_local_storage_needs_no_url(self) -> None:
        """Local storage resolves keys under its base directory, so an unset storage URL is not an error."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_STORAGE_TYPE": "local"}, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings, key_prefix="/tmp/inputs")

        self.assertIsInstance(provider, LocalStorageProvider)

    def test_applies_key_prefix(self) -> None:
        with mock.patch.dict(os.environ, _S3_ENV, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings, key_prefix="my/prefix/")

        assert isinstance(provider, S3StorageProvider)
        self.assertEqual(provider.config.key_prefix, "my/prefix/")

    def test_raises_when_url_missing(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            settings = SettingsBase()
        with self.assertRaises(ValueError) as ctx:
            build_input_storage_provider(settings)
        self.assertIn("URL", str(ctx.exception))

    def test_raises_when_region_missing(self) -> None:
        """``s3://<bucket>`` names no region, and AWS rejects requests signed for the wrong one."""
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_STORAGE_URL": "s3://my-bucket"}, clear=True):
            settings = SettingsBase()
        with self.assertRaises(ValueError) as ctx:
            build_input_storage_provider(settings)
        self.assertIn("region", str(ctx.exception))

    def test_unsupported_storage_type_raises(self) -> None:
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_INPUT_STORAGE_TYPE": "gcs"}, clear=True):
            settings = SettingsBase()
        with self.assertRaises(ValueError) as ctx:
            build_input_storage_provider(settings)
        self.assertIn("gcs", str(ctx.exception))

    def test_passes_credentials(self) -> None:
        env = {
            **_S3_ENV,
            "COSMOS_EVALUATOR_STORAGE_ACCESS_KEY": "AKIATEST",
            "COSMOS_EVALUATOR_STORAGE_SECRET_KEY": "secret123",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings)

        assert isinstance(provider, S3StorageProvider)
        assert provider.config.aws_access_key_id is not None
        assert provider.config.aws_secret_access_key is not None
        self.assertEqual(provider.config.aws_access_key_id.get_secret_value(), "AKIATEST")
        self.assertEqual(provider.config.aws_secret_access_key.get_secret_value(), "secret123")
        self.assertIsNone(provider.config.aws_session_token)

    def test_passes_session_token(self) -> None:
        """Assumed-role credentials reach the provider with their session token, which S3 requires."""
        env = {
            **_S3_ENV,
            "COSMOS_EVALUATOR_STORAGE_ACCESS_KEY": "ASIATEST",
            "COSMOS_EVALUATOR_STORAGE_SECRET_KEY": "secret123",
            "COSMOS_EVALUATOR_STORAGE_SESSION_TOKEN": "FwoGZXIvYXdzEExampleToken",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings)

        assert isinstance(provider, S3StorageProvider)
        assert provider.config.aws_session_token is not None
        self.assertEqual(provider.config.aws_session_token.get_secret_value(), "FwoGZXIvYXdzEExampleToken")
        self.assertIn("aws_session_token", provider.config.get_boto3_config())

    def test_falls_back_to_aws_credentials(self) -> None:
        """With no storage credential set, S3 signs with the standard AWS variables."""
        env = {
            **_S3_ENV,
            "AWS_ACCESS_KEY_ID": "AKIATEST",
            "AWS_SECRET_ACCESS_KEY": "awssecret",
            "AWS_SESSION_TOKEN": "awstoken",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings)

        assert isinstance(provider, S3StorageProvider)
        assert provider.config.aws_access_key_id is not None
        assert provider.config.aws_secret_access_key is not None
        assert provider.config.aws_session_token is not None
        self.assertEqual(provider.config.aws_access_key_id.get_secret_value(), "AKIATEST")
        self.assertEqual(provider.config.aws_secret_access_key.get_secret_value(), "awssecret")
        self.assertEqual(provider.config.aws_session_token.get_secret_value(), "awstoken")

    def test_storage_credentials_replace_aws_credentials_whole(self) -> None:
        """An ambient session token must not be attached to an explicit key pair it was not issued with."""
        env = {
            **_S3_ENV,
            "COSMOS_EVALUATOR_STORAGE_ACCESS_KEY": "AKIAEXPLICIT",
            "COSMOS_EVALUATOR_STORAGE_SECRET_KEY": "explicitsecret",
            "AWS_ACCESS_KEY_ID": "AKIAAMBIENT",
            "AWS_SECRET_ACCESS_KEY": "ambientsecret",
            "AWS_SESSION_TOKEN": "ambienttoken",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings)

        assert isinstance(provider, S3StorageProvider)
        assert provider.config.aws_access_key_id is not None
        self.assertEqual(provider.config.aws_access_key_id.get_secret_value(), "AKIAEXPLICIT")
        self.assertIsNone(provider.config.aws_session_token)

    def test_no_credentials_leaves_boto3_chain(self) -> None:
        with mock.patch.dict(os.environ, _S3_ENV, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings)

        assert isinstance(provider, S3StorageProvider)
        self.assertIsNone(provider.config.aws_access_key_id)
        self.assertNotIn("aws_access_key_id", provider.config.get_boto3_config())


class TestBuildOutputStorageProvider(unittest.TestCase):
    """Tests for build_output_storage_provider."""

    def test_creates_s3_provider_by_default(self) -> None:
        with mock.patch.dict(os.environ, _S3_ENV, clear=True):
            settings = SettingsBase()
            provider = build_output_storage_provider(settings)

        self.assertIsInstance(provider, S3StorageProvider)

    def test_output_storage_type_overrides_base(self) -> None:
        env = {
            **_S3_ENV,
            "COSMOS_EVALUATOR_STORAGE_TYPE": "s3",
            "COSMOS_EVALUATOR_OUTPUT_STORAGE_TYPE": "local",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_output_storage_provider(settings, key_prefix="/tmp/output")

        self.assertIsInstance(provider, LocalStorageProvider)

    def test_output_storage_url_overrides_base(self) -> None:
        env = {
            **_S3_ENV,
            "COSMOS_EVALUATOR_OUTPUT_STORAGE_URL": "https://output-bucket.s3.eu-west-1.amazonaws.com",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_output_storage_provider(settings)

        assert isinstance(provider, S3StorageProvider)
        self.assertEqual(provider.config.bucket_name, "output-bucket")

    def test_falls_back_to_base_storage_type(self) -> None:
        with mock.patch.dict(os.environ, {"COSMOS_EVALUATOR_STORAGE_TYPE": "local"}, clear=True):
            settings = SettingsBase()
            provider = build_output_storage_provider(settings, key_prefix="/tmp/output")

        self.assertIsInstance(provider, LocalStorageProvider)

    def test_applies_key_prefix(self) -> None:
        with mock.patch.dict(os.environ, _S3_ENV, clear=True):
            settings = SettingsBase()
            provider = build_output_storage_provider(settings, key_prefix="run-42/")

        assert isinstance(provider, S3StorageProvider)
        self.assertEqual(provider.config.key_prefix, "run-42/")

    def test_raises_when_url_missing(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            settings = SettingsBase()
        with self.assertRaises(ValueError) as ctx:
            build_output_storage_provider(settings)
        self.assertIn("URL", str(ctx.exception))


class TestAzureStorageProvider(unittest.TestCase):
    """Tests for the azure branch of the factory, in both directions."""

    def test_creates_azure_input_provider(self) -> None:
        with mock.patch.dict(os.environ, _AZURE_ENV, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings, key_prefix="inputs/")

        assert isinstance(provider, AzureBlobStorageProvider)
        self.assertEqual(provider.config.container_name, "my-container")
        self.assertEqual(provider.config.account_name, "myacct")
        self.assertIsNone(provider.config.endpoint_url)
        self.assertEqual(provider.config.key_prefix, "inputs/")
        assert provider.config.account_key is not None
        self.assertEqual(provider.config.account_key.get_secret_value(), "accountsecret")
        self.assertIsNone(provider.config.sas_token)

    def test_creates_azure_output_provider(self) -> None:
        with mock.patch.dict(os.environ, _AZURE_ENV, clear=True):
            settings = SettingsBase()
            provider = build_output_storage_provider(settings, key_prefix="run-42/")

        assert isinstance(provider, AzureBlobStorageProvider)
        self.assertEqual(provider.config.key_prefix, "run-42/")

    def test_sas_token_alone_is_sufficient(self) -> None:
        env = {
            "COSMOS_EVALUATOR_STORAGE_TYPE": "azure",
            "COSMOS_EVALUATOR_STORAGE_URL": "https://myacct.blob.core.windows.net/my-container",
            "COSMOS_EVALUATOR_STORAGE_SESSION_TOKEN": "sv=2024&sig=abc",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings)

        assert isinstance(provider, AzureBlobStorageProvider)
        self.assertIsNone(provider.config.account_key)
        assert provider.config.sas_token is not None
        self.assertEqual(provider.config.sas_token.get_secret_value(), "sv=2024&sig=abc")

    def test_ambient_azure_credentials_are_not_read(self) -> None:
        """Only the ``COSMOS_EVALUATOR_`` names configure the provider, so an ambient account key is never signed."""
        env = {
            "COSMOS_EVALUATOR_STORAGE_TYPE": "azure",
            "COSMOS_EVALUATOR_STORAGE_URL": "https://myacct.blob.core.windows.net/my-container",
            "AZURE_STORAGE_KEY": "aliaskey",
            "AZURE_STORAGE_SAS_TOKEN": "sv=2024&sig=abc",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
        with self.assertRaises(ValueError) as ctx:
            build_input_storage_provider(settings)
        self.assertIn("SECRET_KEY", str(ctx.exception))

    def test_ambient_aws_secret_is_not_used_as_account_key(self) -> None:
        """The S3-only AWS fallback must never reach Azure, where the secret would sign as the account key."""
        env = {
            "COSMOS_EVALUATOR_STORAGE_TYPE": "azure",
            "COSMOS_EVALUATOR_STORAGE_URL": "https://myacct.blob.core.windows.net/my-container",
            "AWS_ACCESS_KEY_ID": "AKIATEST",
            "AWS_SECRET_ACCESS_KEY": "awssecret",
            "AWS_SESSION_TOKEN": "awstoken",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
        with self.assertRaises(ValueError) as ctx:
            build_input_storage_provider(settings)
        self.assertIn("SECRET_KEY", str(ctx.exception))

    def test_private_endpoint_url(self) -> None:
        """A path-style endpoint (Azurite, private link) carries the account as its first path segment."""
        env = {
            **_AZURE_ENV,
            "COSMOS_EVALUATOR_STORAGE_URL": "http://127.0.0.1:10000/devstoreaccount1/my-container",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings)

        assert isinstance(provider, AzureBlobStorageProvider)
        self.assertEqual(provider.config.endpoint_url, "http://127.0.0.1:10000")
        self.assertEqual(provider.config.account_name, "devstoreaccount1")
        self.assertEqual(provider.config.container_name, "my-container")

    def test_trusted_internal_hosts_reach_config(self) -> None:
        env = {**_AZURE_ENV, "COSMOS_EVALUATOR_STORAGE_TRUSTED_INTERNAL_HOSTS": "blob.internal.example.com"}
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings)

        assert isinstance(provider, AzureBlobStorageProvider)
        self.assertEqual(provider.config.trusted_internal_hosts, frozenset({"blob.internal.example.com"}))

    def test_raises_when_container_missing(self) -> None:
        env = {**_AZURE_ENV, "COSMOS_EVALUATOR_STORAGE_URL": "https://myacct.blob.core.windows.net"}
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
        with self.assertRaises(ValueError) as ctx:
            build_input_storage_provider(settings)
        self.assertIn("container", str(ctx.exception))

    def test_raises_when_account_missing(self) -> None:
        env = {**_AZURE_ENV, "COSMOS_EVALUATOR_STORAGE_URL": "http://127.0.0.1:10000/my-container"}
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
        with self.assertRaises(ValueError) as ctx:
            build_input_storage_provider(settings)
        self.assertIn("account", str(ctx.exception))

    def test_raises_when_credentials_missing(self) -> None:
        env = {k: v for k, v in _AZURE_ENV.items() if k != "COSMOS_EVALUATOR_STORAGE_SECRET_KEY"}
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
        with self.assertRaises(ValueError) as ctx:
            build_output_storage_provider(settings)
        self.assertIn("SESSION_TOKEN", str(ctx.exception))


class TestClusterMode(unittest.TestCase):
    """Tests for the cluster-aware S3 provider path (e.g. PBSS)."""

    def test_cluster_resolves_endpoint_creds_and_v2_signing(self) -> None:
        """``s3://<bucket>`` names only the bucket, leaving region and endpoint to the cluster credentials."""
        env = {**_CLUSTER_ENV, "COSMOS_EVALUATOR_STORAGE_URL": "s3://cosmos-mads-dataset"}
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings, cluster="pbss")

        assert isinstance(provider, S3StorageProvider)
        self.assertEqual(provider.config.bucket_name, "cosmos-mads-dataset")
        self.assertEqual(provider.config.endpoint_url, "https://pbss.example.com")
        self.assertEqual(provider.config.region_name, "us-west-2")
        self.assertEqual(provider.config.signature_version, "s3")
        assert provider.config.aws_access_key_id is not None
        self.assertEqual(provider.config.aws_access_key_id.get_secret_value(), "pbsskey")

    def test_bucket_override_takes_precedence(self) -> None:
        env = {**_CLUSTER_ENV, "COSMOS_EVALUATOR_STORAGE_URL": "s3://settings-bucket"}
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_output_storage_provider(settings, cluster="pbss", bucket="explicit-bucket")
        assert isinstance(provider, S3StorageProvider)
        self.assertEqual(provider.config.bucket_name, "explicit-bucket")

    def test_bucket_override_keeps_url_endpoint(self) -> None:
        """The override replaces only the bucket; the endpoint the storage URL names still wins over the cluster's."""
        env = {**_CLUSTER_ENV, "COSMOS_EVALUATOR_STORAGE_URL": "https://override.example.com/settings-bucket"}
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings, cluster="pbss", bucket="explicit-bucket")

        assert isinstance(provider, S3StorageProvider)
        self.assertEqual(provider.config.bucket_name, "explicit-bucket")
        self.assertEqual(provider.config.endpoint_url, "https://override.example.com")

    def test_bucket_override_without_url(self) -> None:
        """An explicit bucket is enough on its own; the cluster credentials supply the rest."""
        with mock.patch.dict(os.environ, _CLUSTER_ENV, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings, cluster="pbss", bucket="explicit-bucket")

        assert isinstance(provider, S3StorageProvider)
        self.assertEqual(provider.config.bucket_name, "explicit-bucket")
        self.assertEqual(provider.config.endpoint_url, "https://pbss.example.com")

    def test_cluster_without_bucket_raises(self) -> None:
        with mock.patch.dict(os.environ, _CLUSTER_ENV, clear=True):
            settings = SettingsBase()
        with self.assertRaises(ValueError) as ctx:
            build_input_storage_provider(settings, cluster="pbss")
        self.assertIn("bucket", str(ctx.exception).lower())

    def test_storage_url_and_credentials_override_cluster_creds(self) -> None:
        """What the storage URL and the explicit credential settings name wins over the cluster-prefixed env vars."""
        env = {
            **_CLUSTER_ENV,
            "COSMOS_EVALUATOR_STORAGE_URL": "https://override.example.com/cosmos-mads-dataset",
            "COSMOS_EVALUATOR_STORAGE_ACCESS_KEY": "override-key",
            "COSMOS_EVALUATOR_STORAGE_SECRET_KEY": "override-secret",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings, cluster="pbss")

        assert isinstance(provider, S3StorageProvider)
        self.assertEqual(provider.config.endpoint_url, "https://override.example.com")
        assert provider.config.aws_access_key_id is not None
        self.assertEqual(provider.config.aws_access_key_id.get_secret_value(), "override-key")
        assert provider.config.aws_secret_access_key is not None
        self.assertEqual(provider.config.aws_secret_access_key.get_secret_value(), "override-secret")
        # The override endpoint names no region, so the cluster credentials still supply it.
        self.assertEqual(provider.config.region_name, "us-west-2")

    def test_cluster_creds_fill_unset_storage_settings(self) -> None:
        """A partial override leaves the rest sourced from the cluster-prefixed env vars."""
        env = {
            **_CLUSTER_ENV,
            "COSMOS_EVALUATOR_STORAGE_URL": "s3://cosmos-mads-dataset",
            "COSMOS_EVALUATOR_STORAGE_ACCESS_KEY": "override-key",
            "COSMOS_EVALUATOR_STORAGE_SECRET_KEY": "override-secret",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            provider = build_input_storage_provider(settings, cluster="pbss")

        assert isinstance(provider, S3StorageProvider)
        self.assertEqual(provider.config.endpoint_url, "https://pbss.example.com")  # from cluster
        self.assertEqual(provider.config.region_name, "us-west-2")  # from cluster
        assert provider.config.aws_access_key_id is not None
        self.assertEqual(provider.config.aws_access_key_id.get_secret_value(), "override-key")  # from settings


class TestMixedMode(unittest.TestCase):
    """Tests for directions that address different backends on a single settings instance."""

    def test_input_s3_output_local(self) -> None:
        env = {
            **_S3_ENV,
            "COSMOS_EVALUATOR_STORAGE_TYPE": "s3",
            "COSMOS_EVALUATOR_OUTPUT_STORAGE_TYPE": "local",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            input_provider = build_input_storage_provider(settings)
            output_provider = build_output_storage_provider(settings, key_prefix="/tmp/output")

        self.assertIsInstance(input_provider, S3StorageProvider)
        self.assertIsInstance(output_provider, LocalStorageProvider)

    def test_input_local_output_s3(self) -> None:
        env = {
            **_S3_ENV,
            "COSMOS_EVALUATOR_STORAGE_TYPE": "local",
            "COSMOS_EVALUATOR_OUTPUT_STORAGE_TYPE": "s3",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            input_provider = build_input_storage_provider(settings, key_prefix="/tmp/inputs")
            output_provider = build_output_storage_provider(settings)

        self.assertIsInstance(input_provider, LocalStorageProvider)
        self.assertIsInstance(output_provider, S3StorageProvider)

    def test_input_s3_output_azure(self) -> None:
        """Reading presigned S3 URLs while writing results to Azure: no single URL can address both targets."""
        env = {
            "COSMOS_EVALUATOR_INPUT_STORAGE_TYPE": "s3",
            "COSMOS_EVALUATOR_INPUT_STORAGE_URL": "https://my-bucket.s3.us-east-1.amazonaws.com",
            "COSMOS_EVALUATOR_OUTPUT_STORAGE_TYPE": "azure",
            "COSMOS_EVALUATOR_OUTPUT_STORAGE_URL": "https://myacct.blob.core.windows.net/my-container",
            "COSMOS_EVALUATOR_STORAGE_SECRET_KEY": "accountsecret",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            input_provider = build_input_storage_provider(settings)
            output_provider = build_output_storage_provider(settings, key_prefix="run-42/")

        assert isinstance(input_provider, S3StorageProvider)
        assert isinstance(output_provider, AzureBlobStorageProvider)
        self.assertEqual(input_provider.config.bucket_name, "my-bucket")
        self.assertEqual(output_provider.config.container_name, "my-container")
        self.assertEqual(output_provider.config.account_name, "myacct")

    def test_input_azure_output_local(self) -> None:
        env = {**_AZURE_ENV, "COSMOS_EVALUATOR_OUTPUT_STORAGE_TYPE": "local"}
        with mock.patch.dict(os.environ, env, clear=True):
            settings = SettingsBase()
            input_provider = build_input_storage_provider(settings)
            output_provider = build_output_storage_provider(settings, key_prefix="/tmp/output")

        self.assertIsInstance(input_provider, AzureBlobStorageProvider)
        self.assertIsInstance(output_provider, LocalStorageProvider)


if __name__ == "__main__":
    unittest.main()
