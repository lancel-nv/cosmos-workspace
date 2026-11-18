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

"""Unit tests for utils.s3."""

import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from utils import s3 as s3_mod


class TestParseS3Uri(unittest.TestCase):
    def test_valid(self) -> None:
        b, k = s3_mod.parse_s3_uri("s3://my-bucket/path/to/key.mp4")
        self.assertEqual(b, "my-bucket")
        self.assertEqual(k, "path/to/key.mp4")

    def test_invalid_scheme(self) -> None:
        with self.assertRaises(ValueError):
            s3_mod.parse_s3_uri("https://example.com/x")

    def test_missing_bucket(self) -> None:
        with self.assertRaises(ValueError):
            s3_mod.parse_s3_uri("s3:///only-path")


class TestIsS3Path(unittest.TestCase):
    def test_true(self) -> None:
        self.assertTrue(s3_mod.is_s3_path("s3://b/k"))

    def test_false(self) -> None:
        self.assertFalse(s3_mod.is_s3_path("/local/path"))


class TestListS3Media(unittest.TestCase):
    @patch("utils.s3.boto3.client")
    def test_filters_video_extensions(self, mock_client: MagicMock) -> None:
        s3 = MagicMock()
        mock_client.return_value = s3
        paginator = MagicMock()
        s3.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {
                "Contents": [
                    {"Key": "a/video.mp4"},
                    {"Key": "b/readme.txt"},
                    {"Key": "c/clips/X.MOV"},
                ]
            }
        ]
        out = s3_mod.list_s3_media("s3://bucket/prefix/")
        self.assertEqual(
            out,
            [
                "s3://bucket/a/video.mp4",
                "s3://bucket/c/clips/X.MOV",
            ],
        )


class TestDownloadToTemp(unittest.TestCase):
    @patch("utils.s3.boto3.client")
    @patch("utils.s3.tempfile.mkstemp", return_value=(3, "/tmp/fake.mp4"))
    @patch("utils.s3.os.close")
    def test_download_success(self, _os_close: MagicMock, _mkstemp: MagicMock, mock_client: MagicMock) -> None:
        s3 = MagicMock()
        mock_client.return_value = s3
        path = s3_mod.download_to_temp("s3://b/k/file.mp4")
        self.assertEqual(str(path), "/tmp/fake.mp4")
        s3.download_file.assert_called_once_with("b", "k/file.mp4", "/tmp/fake.mp4")

    @patch("utils.s3.boto3.client")
    @patch("utils.s3.tempfile.mkstemp", return_value=(3, "/tmp/gone.mp4"))
    @patch("utils.s3.os.close")
    @patch("utils.s3.os.remove")
    def test_download_failure_removes_temp(
        self, mock_remove: MagicMock, _os_close: MagicMock, _mkstemp: MagicMock, mock_client: MagicMock
    ) -> None:
        s3 = MagicMock()
        mock_client.return_value = s3
        s3.download_file.side_effect = RuntimeError("network")
        with self.assertRaises(RuntimeError):
            s3_mod.download_to_temp("s3://b/k/x.mp4")
        mock_remove.assert_called_once_with("/tmp/gone.mp4")


class TestCreateS3Client(unittest.TestCase):
    @patch("utils.s3.boto3.client")
    @patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "id",
            "AWS_SECRET_ACCESS_KEY": "secret",
        },
        clear=False,
    )
    def test_amazon_endpoint_omits_custom_url(self, mock_client: MagicMock) -> None:
        s3_mod.create_s3_client(endpoint_url="https://s3.us-west-2.amazonaws.com", region_name="us-west-2")
        kwargs = mock_client.call_args.kwargs
        self.assertNotIn("endpoint_url", kwargs)
        self.assertEqual(kwargs["region_name"], "us-west-2")
        self.assertEqual(kwargs["aws_access_key_id"], "id")

    @patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": '"quoted-id"',
            "AWS_SECRET_ACCESS_KEY": '"quoted-secret"',
            "AWS_DEFAULT_REGION": "us-west-2",
        },
        clear=True,
    )
    @patch("utils.s3.boto3.client")
    def test_quoted_env_file_values_are_stripped(self, mock_client: MagicMock) -> None:
        s3_mod.create_s3_client(endpoint_url="https://s3.us-west-2.amazonaws.com")
        kwargs = mock_client.call_args.kwargs
        self.assertEqual(kwargs["aws_access_key_id"], "quoted-id")
        self.assertEqual(kwargs["aws_secret_access_key"], "quoted-secret")

    @patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "id",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "AWS_DEFAULT_REGION": "us-west-2",
            "S3_ENDPOINT_URL": "https://custom.example/",
        },
        clear=False,
    )
    @patch("utils.s3.boto3.client")
    def test_custom_endpoint_uses_keys(self, mock_client: MagicMock) -> None:
        s3_mod.create_s3_client(endpoint_url=None, region_name=None)
        kwargs = mock_client.call_args.kwargs
        self.assertEqual(kwargs["endpoint_url"], "https://custom.example")
        self.assertEqual(kwargs["aws_access_key_id"], "id")

    @patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "", "AWS_SECRET_ACCESS_KEY": ""}, clear=False)
    def test_missing_credentials_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            s3_mod.create_s3_client(endpoint_url="https://not-aws.example")
        self.assertIn("credentials", str(ctx.exception).lower())

    @patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "id",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "S3_ENDPOINT_URL": "https://custom.example/",
        },
        clear=False,
    )
    @patch("utils.s3.boto3.client")
    def test_signature_version_default_is_s3v4(self, mock_client: MagicMock) -> None:
        s3_mod.create_s3_client(endpoint_url=None, region_name=None)
        cfg = mock_client.call_args.kwargs["config"]
        self.assertEqual(cfg.signature_version, "s3v4")

    @patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "id",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "S3_ENDPOINT_URL": "https://custom.example/",
        },
        clear=False,
    )
    @patch("utils.s3.boto3.client")
    def test_signature_version_override_propagates(self, mock_client: MagicMock) -> None:
        s3_mod.create_s3_client(endpoint_url=None, region_name=None, signature_version="s3")
        cfg = mock_client.call_args.kwargs["config"]
        self.assertEqual(cfg.signature_version, "s3")


class TestLoadS3ClusterCredentials(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "PBSS_AWS_ACCESS_KEY_ID": "pbss-id",
            "PBSS_AWS_SECRET_ACCESS_KEY": "pbss-secret",
            "PBSS_AWS_DEFAULT_REGION": "us-east-1",
            "PBSS_AWS_ENDPOINT_URL": "https://pbss.s8k.io",
        },
        clear=True,
    )
    def test_cluster_prefixed_vars_loaded(self) -> None:
        creds = s3_mod.load_s3_cluster_credentials("pbss")
        self.assertEqual(creds["access_key_id"], "pbss-id")
        self.assertEqual(creds["secret_access_key"], "pbss-secret")
        self.assertEqual(creds["region"], "us-east-1")
        self.assertEqual(creds["endpoint_url"], "https://pbss.s8k.io")

    @patch.dict(
        os.environ,
        {
            # Cluster-prefixed missing for some fields; plain AWS_* should fill them in.
            "PBSS_AWS_ACCESS_KEY_ID": "pbss-id",
            "AWS_SECRET_ACCESS_KEY": "aws-secret-fallback",
            "AWS_DEFAULT_REGION": "us-west-2",
        },
        clear=True,
    )
    def test_per_variable_silent_fallback_to_aws(self) -> None:
        creds = s3_mod.load_s3_cluster_credentials("pbss")
        # Cluster-prefixed wins where present.
        self.assertEqual(creds["access_key_id"], "pbss-id")
        # Falls back per-variable when prefixed is missing.
        self.assertEqual(creds["secret_access_key"], "aws-secret-fallback")
        self.assertEqual(creds["region"], "us-west-2")
        self.assertEqual(creds["endpoint_url"], "")

    @patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "id",
            "AWS_SECRET_ACCESS_KEY": "secret",
            "AWS_DEFAULT_REGION": "us-east-1",
        },
        clear=True,
    )
    def test_cluster_none_reads_plain_aws_vars(self) -> None:
        creds = s3_mod.load_s3_cluster_credentials(None)
        self.assertEqual(creds["access_key_id"], "id")
        self.assertEqual(creds["secret_access_key"], "secret")
        self.assertEqual(creds["region"], "us-east-1")
        self.assertEqual(creds["endpoint_url"], "")

    @patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": '"id"',
            "AWS_SECRET_ACCESS_KEY": "'secret'",
            "AWS_DEFAULT_REGION": "us-east-1",
        },
        clear=True,
    )
    def test_cluster_none_strips_env_file_quotes(self) -> None:
        creds = s3_mod.load_s3_cluster_credentials(None)
        self.assertEqual(creds["access_key_id"], "id")
        self.assertEqual(creds["secret_access_key"], "secret")

    @patch.dict(os.environ, {}, clear=True)
    def test_all_empty_when_no_env_set(self) -> None:
        creds = s3_mod.load_s3_cluster_credentials("pbss")
        for value in creds.values():
            self.assertEqual(value, "")


class TestCreateS3ClientWithCluster(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "PBSS_AWS_ACCESS_KEY_ID": "pbss-id",
            "PBSS_AWS_SECRET_ACCESS_KEY": "pbss-secret",
            "PBSS_AWS_DEFAULT_REGION": "us-east-1",
            "PBSS_AWS_ENDPOINT_URL": "https://pbss.s8k.io",
        },
        clear=True,
    )
    @patch("utils.s3.boto3.client")
    def test_cluster_routes_credentials_and_endpoint(self, mock_client: MagicMock) -> None:
        s3_mod.create_s3_client(cluster="pbss")
        kwargs = mock_client.call_args.kwargs
        self.assertEqual(kwargs["endpoint_url"], "https://pbss.s8k.io")
        self.assertEqual(kwargs["aws_access_key_id"], "pbss-id")
        self.assertEqual(kwargs["aws_secret_access_key"], "pbss-secret")
        self.assertEqual(kwargs["region_name"], "us-east-1")

    @patch.dict(
        os.environ,
        {
            # No PBSS_AWS_* set; AWS_* should provide everything for cluster='pbss'.
            "AWS_ACCESS_KEY_ID": "aws-id",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
            "AWS_DEFAULT_REGION": "us-west-2",
            "AWS_ENDPOINT_URL": "https://fallback.example",
        },
        clear=True,
    )
    @patch("utils.s3.boto3.client")
    def test_cluster_falls_back_to_aws_per_variable(self, mock_client: MagicMock) -> None:
        s3_mod.create_s3_client(cluster="pbss")
        kwargs = mock_client.call_args.kwargs
        self.assertEqual(kwargs["aws_access_key_id"], "aws-id")
        self.assertEqual(kwargs["endpoint_url"], "https://fallback.example")
        self.assertEqual(kwargs["region_name"], "us-west-2")


def _not_found_client_error(code: str) -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": code, "Message": "Not Found"}},
        operation_name="HeadObject",
    )


class TestObjectExists(unittest.TestCase):
    def test_true_when_head_succeeds(self) -> None:
        client = MagicMock()
        self.assertTrue(s3_mod.object_exists(client, "s3://bucket/path/to/key.mp4"))
        client.head_object.assert_called_once_with(Bucket="bucket", Key="path/to/key.mp4")

    def test_false_on_404(self) -> None:
        client = MagicMock()
        client.head_object.side_effect = _not_found_client_error("404")
        self.assertFalse(s3_mod.object_exists(client, "s3://bucket/missing.mp4"))

    def test_false_on_nosuchkey(self) -> None:
        client = MagicMock()
        client.head_object.side_effect = _not_found_client_error("NoSuchKey")
        self.assertFalse(s3_mod.object_exists(client, "s3://bucket/missing.mp4"))

    def test_false_on_notfound(self) -> None:
        client = MagicMock()
        client.head_object.side_effect = _not_found_client_error("NotFound")
        self.assertFalse(s3_mod.object_exists(client, "s3://bucket/missing.mp4"))

    def test_other_client_error_propagates(self) -> None:
        client = MagicMock()
        client.head_object.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDenied", "Message": "Forbidden"}},
            operation_name="HeadObject",
        )
        with self.assertRaises(ClientError):
            s3_mod.object_exists(client, "s3://bucket/denied.mp4")


class TestDownloadS3Object(unittest.TestCase):
    def test_downloads_via_client(self) -> None:
        client = MagicMock()
        s3_mod.download_s3_object(client, "s3://bucket/path/to/key.mp4", Path("/tmp/out.mp4"))
        client.head_object.assert_called_once_with(Bucket="bucket", Key="path/to/key.mp4")
        client.download_file.assert_called_once_with("bucket", "path/to/key.mp4", "/tmp/out.mp4")

    def test_translates_404_to_typed_error(self) -> None:
        client = MagicMock()
        client.head_object.side_effect = _not_found_client_error("404")
        with self.assertRaises(s3_mod.S3ObjectNotFoundError):
            s3_mod.download_s3_object(client, "s3://bucket/missing.mp4", Path("/tmp/out.mp4"))
        client.download_file.assert_not_called()

    def test_translates_nosuchkey_to_typed_error(self) -> None:
        client = MagicMock()
        client.head_object.side_effect = _not_found_client_error("NoSuchKey")
        with self.assertRaises(s3_mod.S3ObjectNotFoundError):
            s3_mod.download_s3_object(client, "s3://bucket/missing.mp4", Path("/tmp/out.mp4"))

    def test_other_client_error_propagates(self) -> None:
        client = MagicMock()
        client.head_object.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDenied", "Message": "Forbidden"}},
            operation_name="HeadObject",
        )
        with self.assertRaises(ClientError):
            s3_mod.download_s3_object(client, "s3://bucket/denied.mp4", Path("/tmp/out.mp4"))


if __name__ == "__main__":
    unittest.main()
