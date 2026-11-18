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

"""Tests for :mod:`services.framework.url.expiry`.

Covers all four signature schemes (AWS SigV4, AWS SigV2, Azure SAS, GCS V4)
plus the unsigned / malformed edge cases. All tests use an explicit ``now``
so the suite is deterministic and clock-skew independent.
"""

from datetime import UTC, datetime, timedelta
import unittest

from services.framework.url.expiry import SignedUrlExpiry, inspect_signed_url

# Fixed clock used as ``now`` in every test below. All sample URLs are constructed with absolute timestamps relative to
# this anchor.
_NOW = datetime(2026, 5, 19, 12, 0, 0, tzinfo=UTC)


def _aws_sigv4_url(*, issued: datetime, expires_seconds: int) -> str:
    """Build a SigV4 URL whose validity window straddles ``issued``."""
    date_str = issued.strftime("%Y%m%dT%H%M%SZ")
    return (
        "https://bucket.s3.us-west-2.amazonaws.com/key"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
        f"&X-Amz-Date={date_str}"
        f"&X-Amz-Expires={expires_seconds}"
        "&X-Amz-Credential=AKIA/20260519/us-west-2/s3/aws4_request"
        "&X-Amz-SignedHeaders=host"
        "&X-Amz-Signature=deadbeef"
    )


def _gcs_v4_url(*, issued: datetime, expires_seconds: int) -> str:
    date_str = issued.strftime("%Y%m%dT%H%M%SZ")
    return (
        "https://storage.googleapis.com/bucket/key"
        "?X-Goog-Algorithm=GOOG4-RSA-SHA256"
        f"&X-Goog-Date={date_str}"
        f"&X-Goog-Expires={expires_seconds}"
        "&X-Goog-Credential=service-account%40project.iam.gserviceaccount.com"
        "&X-Goog-SignedHeaders=host"
        "&X-Goog-Signature=cafebabe"
    )


def _aws_sigv2_url(*, expires_unix: int) -> str:
    return f"https://bucket.s3.amazonaws.com/key?AWSAccessKeyId=AKIA&Expires={expires_unix}&Signature=abcdef%3D"


def _azure_sas_url(*, expires: datetime, url_encode_se: bool = False) -> str:
    se = expires.strftime("%Y-%m-%dT%H:%M:%SZ")
    if url_encode_se:
        # ``parse_qs`` will URL-decode this back to the colon form before our parser sees it.
        se = se.replace(":", "%3A")
    return f"https://account.blob.core.windows.net/container/blob?sv=2022-11-02&se={se}&sr=b&sp=r&sig=base64%3D%3D"


class TestAwsSigV4(unittest.TestCase):
    def test_fresh_url_reports_not_expired(self) -> None:
        url = _aws_sigv4_url(issued=_NOW - timedelta(minutes=5), expires_seconds=3600)
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)
        self.assertEqual(result.signature_scheme, "aws_sigv4")
        self.assertEqual(result.expires_at, _NOW - timedelta(minutes=5) + timedelta(seconds=3600))
        self.assertIn("valid for", result.reason or "")

    def test_expired_url_reports_expired(self) -> None:
        url = _aws_sigv4_url(issued=_NOW - timedelta(hours=2), expires_seconds=3600)
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, True)
        self.assertEqual(result.signature_scheme, "aws_sigv4")
        self.assertEqual(result.expires_at, _NOW - timedelta(hours=1))
        # ``expires_at`` is the authoritative signal here; the reason string is human diagnostic text and intentionally
        # not pinned to an exact format.
        self.assertIn("expired", result.reason or "")

    def test_at_exactly_expiry_is_expired(self) -> None:
        url = _aws_sigv4_url(issued=_NOW - timedelta(seconds=3600), expires_seconds=3600)
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_expired, True)

    def test_unparseable_date_is_signed_but_not_expired(self) -> None:
        url = "https://bucket.s3.amazonaws.com/key?X-Amz-Date=not-a-date&X-Amz-Expires=3600&X-Amz-Signature=x"
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)
        self.assertEqual(result.signature_scheme, "aws_sigv4")
        self.assertIsNone(result.expires_at)
        self.assertIn("unparseable X-Amz-Date", result.reason or "")

    def test_unparseable_expires_is_signed_but_not_expired(self) -> None:
        url = "https://bucket.s3.amazonaws.com/key?X-Amz-Date=20260519T120000Z&X-Amz-Expires=abc&X-Amz-Signature=x"
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)
        self.assertIn("unparseable X-Amz-Expires", result.reason or "")

    def test_missing_date_is_signed_but_not_expired(self) -> None:
        url = "https://bucket.s3.amazonaws.com/key?X-Amz-Expires=3600&X-Amz-Signature=x"
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)
        self.assertIn("missing X-Amz-Date", result.reason or "")

    def test_overflowing_expires_is_signed_but_not_expired(self) -> None:
        # ``X-Amz-Expires`` is parseable as int but pushes the resulting absolute expiry past ``datetime.max`` (year
        # 9999). Must surface as a parse error rather than OverflowError.
        url = (
            "https://bucket.s3.amazonaws.com/key"
            "?X-Amz-Date=20260519T120000Z"
            "&X-Amz-Expires=9999999999999999"
            "&X-Amz-Signature=x"
        )
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)
        self.assertIsNone(result.expires_at)
        self.assertIn("unparseable X-Amz-Expires", result.reason or "")


class TestGcsV4(unittest.TestCase):
    def test_fresh_url(self) -> None:
        url = _gcs_v4_url(issued=_NOW - timedelta(minutes=1), expires_seconds=900)
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)
        self.assertEqual(result.signature_scheme, "gcs_v4")

    def test_expired_url(self) -> None:
        url = _gcs_v4_url(issued=_NOW - timedelta(hours=3), expires_seconds=3600)
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, True)
        self.assertEqual(result.signature_scheme, "gcs_v4")

    # GCS V4 shares the AWS SigV4 parsing path (``_inspect_amz_style``), so the malformed-input behavior must be
    # symmetric across the two key sets. These tests mirror the SigV4 cases above with the X-Goog-* keys so a regression
    # that only breaks one keyset is caught.

    def test_missing_x_goog_date(self) -> None:
        url = "https://storage.googleapis.com/bucket/key?X-Goog-Expires=900&X-Goog-Signature=x"
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)
        self.assertEqual(result.signature_scheme, "gcs_v4")
        self.assertIn("missing X-Goog-Date", result.reason or "")

    def test_unparseable_x_goog_date(self) -> None:
        url = "https://storage.googleapis.com/bucket/key?X-Goog-Date=not-a-date&X-Goog-Expires=900&X-Goog-Signature=x"
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)
        self.assertEqual(result.signature_scheme, "gcs_v4")
        self.assertIsNone(result.expires_at)
        self.assertIn("unparseable X-Goog-Date", result.reason or "")

    def test_missing_x_goog_expires(self) -> None:
        url = "https://storage.googleapis.com/bucket/key?X-Goog-Date=20260519T120000Z&X-Goog-Signature=x"
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)
        self.assertEqual(result.signature_scheme, "gcs_v4")
        self.assertIn("missing X-Goog-Expires", result.reason or "")

    def test_nonint_x_goog_expires(self) -> None:
        url = (
            "https://storage.googleapis.com/bucket/key"
            "?X-Goog-Date=20260519T120000Z&X-Goog-Expires=abc&X-Goog-Signature=x"
        )
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)
        self.assertEqual(result.signature_scheme, "gcs_v4")
        self.assertIn("unparseable X-Goog-Expires", result.reason or "")


class TestAwsSigV2(unittest.TestCase):
    def test_fresh_url(self) -> None:
        expires_at = _NOW + timedelta(hours=1)
        url = _aws_sigv2_url(expires_unix=int(expires_at.timestamp()))
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)
        self.assertEqual(result.signature_scheme, "aws_sigv2")
        self.assertEqual(result.expires_at, expires_at)

    def test_expired_url(self) -> None:
        expires_at = _NOW - timedelta(hours=1)
        url = _aws_sigv2_url(expires_unix=int(expires_at.timestamp()))
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, True)
        self.assertEqual(result.signature_scheme, "aws_sigv2")

    def test_missing_expires_is_signed_but_not_expired(self) -> None:
        url = "https://bucket.s3.amazonaws.com/key?AWSAccessKeyId=AKIA&Signature=sig"
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)
        self.assertIn("missing Expires", result.reason or "")

    def test_unparseable_expires_is_signed_but_not_expired(self) -> None:
        url = "https://bucket.s3.amazonaws.com/key?AWSAccessKeyId=AKIA&Signature=sig&Expires=tomorrow"
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)
        self.assertIn("unparseable Expires", result.reason or "")

    def test_large_expires_is_signed_but_not_expired(self) -> None:
        # Mirrors the SigV4/GCS overflow guardrail: ``int()`` accepts arbitrarily large values, but
        # ``datetime.fromtimestamp`` raises OverflowError/OSError/ValueError past the platform's ``time_t`` range.
        # ``_inspect_sigv2`` must catch that and surface a diagnostic rather than 500-ing the caller.
        url = "https://bucket.s3.amazonaws.com/key?AWSAccessKeyId=AKIA&Signature=sig&Expires=9999999999999999"
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)
        # Current message is ``out-of-range Expires=...``; pinning to ``Expires`` keeps the assertion stable if the
        # wording is later unified with SigV4's ``unparseable``.
        self.assertIn("Expires", result.reason or "")


class TestAzureSas(unittest.TestCase):
    def test_fresh_url(self) -> None:
        url = _azure_sas_url(expires=_NOW + timedelta(hours=2))
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)
        self.assertEqual(result.signature_scheme, "azure_sas")
        self.assertEqual(result.expires_at, _NOW + timedelta(hours=2))

    def test_expired_url(self) -> None:
        url = _azure_sas_url(expires=_NOW - timedelta(minutes=30))
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, True)
        self.assertEqual(result.signature_scheme, "azure_sas")

    def test_accepts_url_encoded_se(self) -> None:
        # Many Azure SAS generators URL-encode the colons in the expiry timestamp; parse_qs decodes those back
        # transparently, so the parser must accept both.
        url = _azure_sas_url(expires=_NOW + timedelta(hours=2), url_encode_se=True)
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)
        self.assertEqual(result.signature_scheme, "azure_sas")

    def test_missing_se_is_signed_but_not_expired(self) -> None:
        url = "https://account.blob.core.windows.net/container/blob?sv=2022-11-02&sig=base64"
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)
        self.assertIn("missing se", result.reason or "")

    def test_unparseable_se_is_signed_but_not_expired(self) -> None:
        url = "https://account.blob.core.windows.net/container/blob?sv=2022-11-02&se=garbage&sig=base64"
        result = inspect_signed_url(url, now=_NOW)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)
        self.assertIn("unparseable se", result.reason or "")


class TestUnsignedAndEdgeCases(unittest.TestCase):
    def test_plain_https_is_unsigned(self) -> None:
        result = inspect_signed_url("https://example.com/foo/bar.mp4", now=_NOW)
        self.assertEqual(
            result,
            SignedUrlExpiry(
                is_signed=False,
                is_expired=False,
                signature_scheme=None,
                expires_at=None,
                reason=None,
            ),
        )

    def test_https_with_unrelated_query_is_unsigned(self) -> None:
        result = inspect_signed_url("https://example.com/foo?token=abc&version=2", now=_NOW)
        self.assertIs(result.is_signed, False)
        self.assertIs(result.is_expired, False)

    def test_s3_uri_is_unsigned(self) -> None:
        result = inspect_signed_url("s3://bucket/path/to/key.mp4", now=_NOW)
        self.assertIs(result.is_signed, False)

    def test_empty_url_is_unsigned(self) -> None:
        result = inspect_signed_url("", now=_NOW)
        self.assertIs(result.is_signed, False)

    def test_naive_now_is_treated_as_utc(self) -> None:
        # Passing a tz-naive ``now`` must not raise and must be interpreted as UTC.
        url = _aws_sigv4_url(issued=_NOW - timedelta(minutes=5), expires_seconds=3600)
        naive_now = _NOW.replace(tzinfo=None)
        result = inspect_signed_url(url, now=naive_now)
        self.assertIs(result.is_signed, True)
        self.assertIs(result.is_expired, False)


if __name__ == "__main__":
    unittest.main()
