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

"""Tests for storage URL parsing."""

from __future__ import annotations

import unittest

from utils.storage_url import S3Endpoint, StorageTarget, parse_s3_endpoint_url, parse_storage_url


class TestParseS3(unittest.TestCase):
    """S3 accepts the s3:// shorthand, both AWS addressing styles, and custom endpoints."""

    def test_s3_uri(self) -> None:
        self.assertEqual(parse_storage_url("s3://my-bucket", "s3"), StorageTarget(container="my-bucket"))

    def test_s3_uri_tolerates_trailing_slash(self) -> None:
        self.assertEqual(parse_storage_url("s3://my-bucket/", "s3"), StorageTarget(container="my-bucket"))

    def test_virtual_hosted_url_carries_region(self) -> None:
        target = parse_storage_url("https://my-bucket.s3.us-west-2.amazonaws.com", "s3")
        self.assertEqual(target, StorageTarget(container="my-bucket", region="us-west-2"))

    def test_path_style_url_carries_region(self) -> None:
        target = parse_storage_url("https://s3.us-west-2.amazonaws.com/my-bucket", "s3")
        self.assertEqual(target, StorageTarget(container="my-bucket", region="us-west-2"))

    def test_virtual_hosted_bucket_may_contain_dots(self) -> None:
        target = parse_storage_url("https://my.dotted.bucket.s3.eu-central-1.amazonaws.com", "s3")
        self.assertEqual(target, StorageTarget(container="my.dotted.bucket", region="eu-central-1"))

    def test_legacy_dashed_region_host(self) -> None:
        target = parse_storage_url("https://s3-us-west-2.amazonaws.com/my-bucket", "s3")
        self.assertEqual(target, StorageTarget(container="my-bucket", region="us-west-2"))

    def test_regionless_aws_host_leaves_region_unset(self) -> None:
        target = parse_storage_url("https://my-bucket.s3.amazonaws.com", "s3")
        self.assertEqual(target, StorageTarget(container="my-bucket", region=None))

    def test_non_region_label_is_not_read_as_a_region(self) -> None:
        """A 'dualstack' label sits where a region does; handing it to boto3 would fail at signing time."""
        target = parse_storage_url("https://s3.dualstack.us-east-1.amazonaws.com/my-bucket", "s3")
        self.assertEqual(target, StorageTarget(container="my-bucket", region=None))

    def test_host_matching_is_case_insensitive(self) -> None:
        target = parse_storage_url("https://MY-BUCKET.S3.US-WEST-2.AMAZONAWS.COM", "s3")
        self.assertEqual(target, StorageTarget(container="my-bucket", region="us-west-2"))

    def test_custom_endpoint_keeps_origin_and_bucket(self) -> None:
        target = parse_storage_url("https://pbss.s8k.io/my-bucket", "s3")
        self.assertEqual(target, StorageTarget(container="my-bucket", endpoint_url="https://pbss.s8k.io"))

    def test_custom_endpoint_preserves_port(self) -> None:
        target = parse_storage_url("http://localhost:9000/my-bucket", "s3")
        self.assertEqual(target, StorageTarget(container="my-bucket", endpoint_url="http://localhost:9000"))

    def test_azure_blob_host_raises(self) -> None:
        """A public Azure host must not be mistaken for a custom S3 endpoint."""
        with self.assertRaises(ValueError) as ctx:
            parse_storage_url("https://myacct.blob.core.windows.net/my-container", "s3")
        self.assertIn("storage type to 'azure'", str(ctx.exception))

    def test_key_below_the_bucket_raises(self) -> None:
        for url in ("s3://my-bucket/prefix", "https://pbss.s8k.io/my-bucket/prefix"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                parse_storage_url(url, "s3")

    def test_endpoint_without_bucket_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_storage_url("https://pbss.s8k.io", "s3")

    def test_s3_uri_without_bucket_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_storage_url("s3://", "s3")

    def test_foreign_scheme_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_storage_url("gs://my-bucket", "s3")


class TestParseAzure(unittest.TestCase):
    """Azure is addressed by its public blob host or a path-style private/emulator endpoint."""

    def test_public_blob_host(self) -> None:
        target = parse_storage_url("https://myacct.blob.core.windows.net/my-container", "azure")
        self.assertEqual(target, StorageTarget(container="my-container", account="myacct"))

    def test_private_link_host(self) -> None:
        target = parse_storage_url("https://myacct.privatelink.blob.core.windows.net/my-container", "azure")
        self.assertEqual(target, StorageTarget(container="my-container", account="myacct"))

    def test_aws_host_raises(self) -> None:
        """A path-style AWS URL must not be read as <endpoint>/<account>/<container>."""
        with self.assertRaises(ValueError) as ctx:
            parse_storage_url("https://s3.us-west-2.amazonaws.com/my-bucket/prefix", "azure")
        self.assertIn("storage type to 's3'", str(ctx.exception))

    def test_emulator_endpoint(self) -> None:
        target = parse_storage_url("http://127.0.0.1:10000/devstoreaccount1/my-container", "azure")
        self.assertEqual(
            target,
            StorageTarget(container="my-container", account="devstoreaccount1", endpoint_url="http://127.0.0.1:10000"),
        )

    def test_host_matching_is_case_insensitive(self) -> None:
        target = parse_storage_url("https://MYACCT.BLOB.CORE.WINDOWS.NET/my-container", "azure")
        self.assertEqual(target, StorageTarget(container="my-container", account="myacct"))

    def test_public_host_without_container_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_storage_url("https://myacct.blob.core.windows.net", "azure")

    def test_endpoint_without_container_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_storage_url("http://127.0.0.1:10000/devstoreaccount1", "azure")

    def test_blob_below_the_container_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_storage_url("https://myacct.blob.core.windows.net/my-container/clip.mp4", "azure")

    def test_scheme_tagged_uri_raises(self) -> None:
        """There is no azure:// shorthand: the account is structural and such a URI cannot carry it."""
        with self.assertRaises(ValueError):
            parse_storage_url("azure://my-container", "azure")


class TestParseStorageUrl(unittest.TestCase):
    """Dispatch-level behavior shared by every backend."""

    def test_unsupported_storage_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_storage_url("s3://my-bucket", "gcs")

    def test_local_storage_type_raises(self) -> None:
        """Local storage resolves keys under a caller-supplied base directory and has no URL to parse."""
        with self.assertRaises(ValueError):
            parse_storage_url("file:///var/output", "local")

    def test_empty_url_raises(self) -> None:
        for storage_type in ("s3", "azure"):
            with self.subTest(storage_type=storage_type), self.assertRaises(ValueError):
                parse_storage_url("", storage_type)


class TestParseS3Endpoint(unittest.TestCase):
    """A store addressed without a bucket: an AWS service endpoint or an S3-compatible origin."""

    def test_aws_regional_endpoint(self) -> None:
        endpoint = parse_s3_endpoint_url("https://s3.us-east-1.amazonaws.com")
        self.assertEqual(endpoint, S3Endpoint(endpoint_url=None, region="us-east-1"))
        self.assertIs(endpoint.is_aws, True)

    def test_aws_legacy_dashed_endpoint(self) -> None:
        endpoint = parse_s3_endpoint_url("https://s3-eu-west-1.amazonaws.com")
        self.assertEqual(endpoint, S3Endpoint(endpoint_url=None, region="eu-west-1"))

    def test_aws_regionless_endpoint_leaves_region_to_the_caller(self) -> None:
        self.assertEqual(parse_s3_endpoint_url("https://s3.amazonaws.com"), S3Endpoint(endpoint_url=None, region=None))

    def test_compatible_endpoint_keeps_origin(self) -> None:
        endpoint = parse_s3_endpoint_url("http://localhost:9000")
        self.assertEqual(endpoint, S3Endpoint(endpoint_url="http://localhost:9000", region=None))
        self.assertIs(endpoint.is_aws, False)

    def test_trailing_slash_is_not_a_path(self) -> None:
        endpoint = parse_s3_endpoint_url("https://pbss.s8k.io/")
        self.assertEqual(endpoint, S3Endpoint(endpoint_url="https://pbss.s8k.io", region=None))

    def test_path_raises(self) -> None:
        for url in ("https://pbss.s8k.io/my-bucket", "https://s3.us-east-1.amazonaws.com/my-bucket"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                parse_s3_endpoint_url(url)

    def test_virtual_hosted_bucket_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_s3_endpoint_url("https://my-bucket.s3.us-east-1.amazonaws.com")
        self.assertIn("names a bucket", str(ctx.exception))

    def test_azure_host_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_s3_endpoint_url("https://myacct.blob.core.windows.net")

    def test_unsupported_input_raises(self) -> None:
        for url in ("", "s3://my-bucket", "https://"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                parse_s3_endpoint_url(url)


if __name__ == "__main__":
    unittest.main()
