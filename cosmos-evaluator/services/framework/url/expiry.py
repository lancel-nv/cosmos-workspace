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

"""Presigned-URL expiry inspection across cloud providers.

Pure-offline parser that looks at a URL's query string to identify whether it carries a presigned signature and, if so,
when that signature expires. Used by the Arbitrator's admission gate to fail fast on inputs whose access has already
lapsed, before any checker work runs.

Supported signature schemes:

- **AWS SigV4** (S3, R2, MinIO, ...) — ``X-Amz-Signature`` plus ``X-Amz-Date`` + ``X-Amz-Expires`` (seconds since
  issue).
- **AWS SigV2 / legacy S3** — ``Signature`` + ``AWSAccessKeyId`` plus ``Expires`` (absolute Unix seconds).
- **Azure SAS** — ``sig`` plus ``se`` (ISO 8601 expiry; URL-decoding handled by ``parse_qs``).
- **GCS V4 signed URLs** — ``X-Goog-Signature`` plus ``X-Goog-Date`` + ``X-Goog-Expires`` (mirrors AWS SigV4 layout).

URLs without any recognized signature parameters are returned with ``is_signed=False`` and are not classified as
expired; they are left for the underlying storage provider to handle at fetch time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

# AWS SigV4 and GCS V4 both use ISO 8601 "basic" format with no separators.
_COMPACT_ISO_DATE_FORMAT = "%Y%m%dT%H%M%SZ"


@dataclass(frozen=True)
class SignedUrlExpiry:
    """Result of inspecting a URL for a presigned signature and its expiry.

    Attributes:
        is_signed: True when at least one recognized signature parameter is present on the URL.
        is_expired: True only when ``is_signed`` is True *and* the signature's expiry is in the past relative to
            ``now``. Unparseable expiries yield False here so callers can decide whether to fail closed.
        signature_scheme: Identifier of the detected scheme, or ``None`` when no signature was found.
        expires_at: Absolute UTC expiry, or ``None`` when the URL is unsigned or the expiry could not be parsed.
        reason: Human-readable detail (e.g. ``"expired 312s ago"``, ``"valid for 1782s more"``, or a parse-error
            message). ``None`` for unsigned URLs.
    """

    is_signed: bool
    is_expired: bool
    signature_scheme: str | None
    expires_at: datetime | None
    reason: str | None


def inspect_signed_url(url: str, *, now: datetime | None = None) -> SignedUrlExpiry:
    """Inspect *url* for a presigned signature and determine whether it has expired.

    The check is purely offline (no network calls, no signature verification — only expiry-window arithmetic). When a
    recognized scheme is present but the expiry parameters are missing or malformed, the URL is reported as
    ``is_signed=True`` with ``is_expired=False`` and a ``reason`` explaining why, so callers can fail open or closed as
    appropriate.

    Args:
        url: The URL to inspect.
        now: Reference timestamp for the expiry comparison. Defaults to ``datetime.now(timezone.utc)``. Naive datetimes
        are interpreted as UTC.

    Returns:
        A :class:`SignedUrlExpiry` describing what was found.
    """
    if now is None:
        now = datetime.now(UTC)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    try:
        parsed = urlparse(url)
    except (ValueError, TypeError):
        return _unsigned()

    if not parsed.query:
        return _unsigned()

    params = parse_qs(parsed.query, keep_blank_values=True)

    # AWS SigV4 — also used by S3-compatible stores (R2, MinIO, ...).
    if _first(params, "X-Amz-Signature") is not None:
        return _inspect_amz_style(
            params, scheme="aws_sigv4", date_key="X-Amz-Date", expires_key="X-Amz-Expires", now=now
        )

    # GCS V4 signed URLs — identical layout to AWS SigV4 with X-Goog-* keys.
    if _first(params, "X-Goog-Signature") is not None:
        return _inspect_amz_style(
            params, scheme="gcs_v4", date_key="X-Goog-Date", expires_key="X-Goog-Expires", now=now
        )

    # AWS SigV2 / legacy S3 presigned URLs.
    if _first(params, "Signature") is not None and _first(params, "AWSAccessKeyId") is not None:
        return _inspect_sigv2(params, now=now)

    # Azure SAS — ``sig`` plus ``se`` (signed expiry).
    if _first(params, "sig") is not None:
        return _inspect_azure_sas(params, now=now)

    return _unsigned()


def _unsigned() -> SignedUrlExpiry:
    return SignedUrlExpiry(
        is_signed=False,
        is_expired=False,
        signature_scheme=None,
        expires_at=None,
        reason=None,
    )


def _first(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key)
    return values[0] if values else None


def _inspect_amz_style(
    params: dict[str, list[str]],
    *,
    scheme: str,
    date_key: str,
    expires_key: str,
    now: datetime,
) -> SignedUrlExpiry:
    """Inspect AWS SigV4 or GCS V4 layouts (issue-date + relative expiry-seconds)."""
    date_str = _first(params, date_key)
    expires_str = _first(params, expires_key)

    if not date_str:
        return _signed_parse_error(scheme, f"missing {date_key}")
    if not expires_str:
        return _signed_parse_error(scheme, f"missing {expires_key}")

    issued_at = _parse_compact_iso_date(date_str)
    if issued_at is None:
        return _signed_parse_error(scheme, f"unparseable {date_key}={date_str!r}")

    try:
        expires_seconds = int(expires_str)
    except ValueError:
        return _signed_parse_error(scheme, f"unparseable {expires_key}={expires_str!r}")

    try:
        # Pathological / malicious inputs (e.g. ``X-Amz-Expires=9999999999999999``) can blow past ``timedelta``'s
        # internal range or push the absolute expiry past ``datetime.max``; treat both as 'unparseable' rather than
        # letting OverflowError bubble up as a 500.
        expires_at = issued_at + timedelta(seconds=expires_seconds)
    except OverflowError:
        return _signed_parse_error(scheme, f"unparseable {expires_key}={expires_str!r}")

    return _build_result(scheme, expires_at, now)


def _inspect_sigv2(params: dict[str, list[str]], *, now: datetime) -> SignedUrlExpiry:
    """Inspect AWS SigV2 layout: ``Expires`` is an absolute Unix timestamp (seconds)."""
    expires_str = _first(params, "Expires")
    if not expires_str:
        return _signed_parse_error("aws_sigv2", "missing Expires")

    try:
        expires_unix = int(expires_str)
    except ValueError:
        return _signed_parse_error("aws_sigv2", f"unparseable Expires={expires_str!r}")

    try:
        expires_at = datetime.fromtimestamp(expires_unix, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return _signed_parse_error("aws_sigv2", f"out-of-range Expires={expires_str!r}")

    return _build_result("aws_sigv2", expires_at, now)


def _inspect_azure_sas(params: dict[str, list[str]], *, now: datetime) -> SignedUrlExpiry:
    """Inspect Azure SAS layout: ``se`` is an ISO 8601 absolute expiry."""
    se = _first(params, "se")
    if not se:
        return _signed_parse_error("azure_sas", "missing se")

    expires_at = _parse_iso_8601(se)
    if expires_at is None:
        return _signed_parse_error("azure_sas", f"unparseable se={se!r}")

    return _build_result("azure_sas", expires_at, now)


def _parse_compact_iso_date(value: str) -> datetime | None:
    """Parse AWS/GCS basic ISO 8601 dates like ``20240107T134500Z``."""
    try:
        return datetime.strptime(value, _COMPACT_ISO_DATE_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None


def _parse_iso_8601(value: str) -> datetime | None:
    """Parse a delimited ISO 8601 date used by Azure SAS ``se``.

    Accepts both ``Z`` and ``+HH:MM`` zone designators; naive timestamps are
    interpreted as UTC.
    """
    candidate = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _build_result(scheme: str, expires_at: datetime, now: datetime) -> SignedUrlExpiry:
    if expires_at <= now:
        delta = int((now - expires_at).total_seconds())
        return SignedUrlExpiry(
            is_signed=True,
            is_expired=True,
            signature_scheme=scheme,
            expires_at=expires_at,
            reason=f"expired {delta}s ago",
        )

    delta = int((expires_at - now).total_seconds())
    return SignedUrlExpiry(
        is_signed=True,
        is_expired=False,
        signature_scheme=scheme,
        expires_at=expires_at,
        reason=f"valid for {delta}s more",
    )


def _signed_parse_error(scheme: str, detail: str) -> SignedUrlExpiry:
    """Return a 'signed but unparseable' result — caller chooses whether to fail closed."""
    return SignedUrlExpiry(
        is_signed=True,
        is_expired=False,
        signature_scheme=scheme,
        expires_at=None,
        reason=detail,
    )
