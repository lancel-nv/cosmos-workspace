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

"""Unit tests for the pure Azure Blob URL helpers.

These have no Azure SDK dependency, so they are exercised directly rather than through the provider. The negative
cases matter as much as the positive ones: ``is_azure_blob_url`` is what keeps a foreign presigned URL from being
handed to the Azure SDK, so every non-Azure scheme and host must stay unrecognized.
"""

from __future__ import annotations

import unittest

from services.framework.storage_providers.utils.azure import (
    build_account_url,
    build_azure_uri,
    build_blob_url,
    is_azure_blob_url,
    parse_azure_url,
)

_AZURITE_ENDPOINT = "http://127.0.0.1:10000"


class TestIsAzureBlobUrl(unittest.TestCase):
    """Classification rule: azure:// URIs, blob.core.windows.net hosts, and the configured endpoint."""

    def test_azure_uri(self) -> None:
        self.assertTrue(is_azure_blob_url("azure://container/path/to/blob.mp4"))

    def test_public_blob_host(self) -> None:
        self.assertTrue(is_azure_blob_url("https://acct.blob.core.windows.net/container/blob.mp4"))

    def test_public_blob_host_with_sas(self) -> None:
        self.assertTrue(is_azure_blob_url("https://acct.blob.core.windows.net/container/blob.mp4?sig=abc&se=xyz"))

    def test_private_link_blob_host(self) -> None:
        self.assertTrue(is_azure_blob_url("https://acct.privatelink.blob.core.windows.net/container/blob.mp4"))

    def test_host_matching_is_case_insensitive(self) -> None:
        self.assertTrue(is_azure_blob_url("https://ACCT.BLOB.CORE.WINDOWS.NET/container/blob.mp4"))

    def test_custom_endpoint_requires_endpoint_argument(self) -> None:
        url = f"{_AZURITE_ENDPOINT}/devstoreaccount1/container/blob.mp4"
        self.assertFalse(is_azure_blob_url(url))
        self.assertTrue(is_azure_blob_url(url, _AZURITE_ENDPOINT))

    def test_endpoint_argument_with_trailing_path_still_matches_origin(self) -> None:
        url = f"{_AZURITE_ENDPOINT}/devstoreaccount1/container/blob.mp4"
        self.assertTrue(is_azure_blob_url(url, f"{_AZURITE_ENDPOINT}/devstoreaccount1"))

    def test_endpoint_match_spans_the_whole_origin_not_a_prefix(self) -> None:
        # Each of these starts with the endpoint origin as a string but resolves to a different host, so a prefix
        # test would accept them. Callers skip the SSRF check on a match, so only the exact origin may match.
        for url in (
            "http://127.0.0.1:10000@evil.com/devstoreaccount1/container/blob.mp4",
            "http://127.0.0.1:10000.evil.com/devstoreaccount1/container/blob.mp4",
            "http://127.0.0.1:100000/devstoreaccount1/container/blob.mp4",
            "https://127.0.0.1:10000/devstoreaccount1/container/blob.mp4",
        ):
            with self.subTest(url=url):
                self.assertFalse(is_azure_blob_url(url, _AZURITE_ENDPOINT))

    def test_public_host_suffix_match_is_label_aligned(self) -> None:
        self.assertFalse(is_azure_blob_url("https://acct.blob.core.windows.net.evil.com/container/blob.mp4"))

    def test_endpoint_default_port_may_be_spelled_out(self) -> None:
        url = "https://blob.internal.example:443/devstoreaccount1/container/blob.mp4"
        self.assertTrue(is_azure_blob_url(url, "https://blob.internal.example"))

    def test_s3_uri_is_not_azure(self) -> None:
        self.assertFalse(is_azure_blob_url("s3://bucket/key"))

    def test_s3_presigned_url_is_not_azure(self) -> None:
        url = "https://bucket.s3.us-west-2.amazonaws.com/key?X-Amz-Signature=abc"
        self.assertFalse(is_azure_blob_url(url))
        # An endpoint override must not widen the match to unrelated hosts.
        self.assertFalse(is_azure_blob_url(url, _AZURITE_ENDPOINT))

    def test_gs_uri_is_not_azure(self) -> None:
        self.assertFalse(is_azure_blob_url("gs://bucket/key"))

    def test_arbitrary_https_url_is_not_azure(self) -> None:
        self.assertFalse(is_azure_blob_url("https://example.com/blob.mp4"))

    def test_bare_paths_are_not_azure(self) -> None:
        self.assertFalse(is_azure_blob_url("/abs/path/blob.mp4"))
        self.assertFalse(is_azure_blob_url("rel/path/blob.mp4"))

    def test_file_uri_is_not_azure(self) -> None:
        self.assertFalse(is_azure_blob_url("file:///abs/path/blob.mp4"))


class TestParseAzureUrl(unittest.TestCase):
    """Parsing of all three accepted URL forms."""

    def test_azure_uri(self) -> None:
        components = parse_azure_url("azure://my-container/nested/key/blob.mp4")
        self.assertEqual(components.container_name, "my-container")
        self.assertEqual(components.blob_name, "nested/key/blob.mp4")
        self.assertIsNone(components.account_name)
        self.assertIsNone(components.sas_token)

    def test_azure_uri_percent_decodes_blob_name(self) -> None:
        components = parse_azure_url("azure://container/dir/my%20blob.mp4")
        self.assertEqual(components.blob_name, "dir/my blob.mp4")

    def test_azure_uri_without_blob_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_azure_url("azure://container")

    def test_azure_uri_without_container_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_azure_url("azure:///blob.mp4")

    def test_public_blob_url(self) -> None:
        components = parse_azure_url("https://myacct.blob.core.windows.net/my-container/nested/blob.mp4")
        self.assertEqual(components.account_name, "myacct")
        self.assertEqual(components.container_name, "my-container")
        self.assertEqual(components.blob_name, "nested/blob.mp4")
        self.assertIsNone(components.sas_token)

    def test_public_blob_url_with_sas(self) -> None:
        components = parse_azure_url(
            "https://myacct.blob.core.windows.net/c/blob.mp4?sv=2024-11-04&sig=abc%2Fdef&se=2026-01-01T00%3A00%3A00Z"
        )
        self.assertEqual(components.account_name, "myacct")
        self.assertEqual(components.sas_token, "sv=2024-11-04&sig=abc%2Fdef&se=2026-01-01T00%3A00%3A00Z")

    def test_private_link_url_takes_first_label_as_account(self) -> None:
        components = parse_azure_url("https://myacct.privatelink.blob.core.windows.net/c/blob.mp4")
        self.assertEqual(components.account_name, "myacct")
        self.assertEqual(components.container_name, "c")

    def test_public_blob_url_without_blob_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_azure_url("https://myacct.blob.core.windows.net/only-a-container")

    def test_path_style_url_with_endpoint(self) -> None:
        components = parse_azure_url(
            f"{_AZURITE_ENDPOINT}/devstoreaccount1/my-container/nested/blob.mp4", _AZURITE_ENDPOINT
        )
        self.assertEqual(components.account_name, "devstoreaccount1")
        self.assertEqual(components.container_name, "my-container")
        self.assertEqual(components.blob_name, "nested/blob.mp4")

    def test_path_style_url_without_endpoint_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_azure_url(f"{_AZURITE_ENDPOINT}/devstoreaccount1/my-container/blob.mp4")

    def test_path_style_url_missing_segments_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_azure_url(f"{_AZURITE_ENDPOINT}/devstoreaccount1/container", _AZURITE_ENDPOINT)

    def test_unsupported_scheme_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_azure_url("s3://bucket/key")

    def test_unrecognized_https_host_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_azure_url("https://example.com/container/blob.mp4")


class TestBuildUrls(unittest.TestCase):
    """URL construction, including the round trip back through the parser."""

    def test_build_azure_uri(self) -> None:
        self.assertEqual(build_azure_uri("container", "nested/blob.mp4"), "azure://container/nested/blob.mp4")

    def test_build_azure_uri_strips_leading_slash(self) -> None:
        self.assertEqual(build_azure_uri("container", "/nested/blob.mp4"), "azure://container/nested/blob.mp4")

    def test_build_account_url_public(self) -> None:
        self.assertEqual(build_account_url("myacct"), "https://myacct.blob.core.windows.net")

    def test_build_account_url_custom_endpoint(self) -> None:
        self.assertEqual(
            build_account_url("devstoreaccount1", _AZURITE_ENDPOINT), f"{_AZURITE_ENDPOINT}/devstoreaccount1"
        )

    def test_build_blob_url_public(self) -> None:
        self.assertEqual(
            build_blob_url("container", "nested/blob.mp4", "myacct"),
            "https://myacct.blob.core.windows.net/container/nested/blob.mp4",
        )

    def test_build_blob_url_appends_sas(self) -> None:
        self.assertEqual(
            build_blob_url("container", "blob.mp4", "myacct", sas_token="sig=abc"),
            "https://myacct.blob.core.windows.net/container/blob.mp4?sig=abc",
        )

    def test_build_blob_url_tolerates_leading_question_mark_on_sas(self) -> None:
        self.assertEqual(
            build_blob_url("container", "blob.mp4", "myacct", sas_token="?sig=abc"),
            "https://myacct.blob.core.windows.net/container/blob.mp4?sig=abc",
        )

    def test_build_blob_url_custom_endpoint(self) -> None:
        self.assertEqual(
            build_blob_url("container", "blob.mp4", "devstoreaccount1", _AZURITE_ENDPOINT),
            f"{_AZURITE_ENDPOINT}/devstoreaccount1/container/blob.mp4",
        )

    def test_round_trip_public_url(self) -> None:
        url = build_blob_url("container", "nested/blob.mp4", "myacct")
        components = parse_azure_url(url)
        self.assertEqual(components.account_name, "myacct")
        self.assertEqual(components.container_name, "container")
        self.assertEqual(components.blob_name, "nested/blob.mp4")

    def test_round_trip_path_style_url(self) -> None:
        url = build_blob_url("container", "nested/blob.mp4", "devstoreaccount1", _AZURITE_ENDPOINT)
        components = parse_azure_url(url, _AZURITE_ENDPOINT)
        self.assertEqual(components.account_name, "devstoreaccount1")
        self.assertEqual(components.container_name, "container")
        self.assertEqual(components.blob_name, "nested/blob.mp4")

    def test_round_trip_azure_uri(self) -> None:
        uri = build_azure_uri("container", "nested/blob.mp4")
        components = parse_azure_url(uri)
        self.assertEqual(components.container_name, "container")
        self.assertEqual(components.blob_name, "nested/blob.mp4")

    def test_builders_encode_reserved_characters_but_keep_separators(self) -> None:
        self.assertEqual(build_azure_uri("container", "dir/a?b#c d.mp4"), "azure://container/dir/a%3Fb%23c%20d.mp4")
        self.assertEqual(
            build_blob_url("container", "dir/a?b#c d.mp4", "myacct"),
            "https://myacct.blob.core.windows.net/container/dir/a%3Fb%23c%20d.mp4",
        )

    def test_round_trip_preserves_reserved_characters(self) -> None:
        # Unencoded, the '?' would open a SAS query and the '#' a fragment, truncating the name the parser reads back.
        blob_name = "dir/a?b#c d%e.mp4"
        for url, endpoint in (
            (build_azure_uri("container", blob_name), None),
            (build_blob_url("container", blob_name, "myacct"), None),
            (build_blob_url("container", blob_name, "devstoreaccount1", _AZURITE_ENDPOINT), _AZURITE_ENDPOINT),
        ):
            with self.subTest(url=url):
                components = parse_azure_url(url, endpoint)
                self.assertEqual(components.container_name, "container")
                self.assertEqual(components.blob_name, blob_name)
                self.assertIsNone(components.sas_token)


if __name__ == "__main__":
    unittest.main()
