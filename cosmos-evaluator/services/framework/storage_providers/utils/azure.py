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

"""Pure utility functions and data classes for Azure Blob Storage URL parsing and construction.

These utilities have no Azure SDK dependency — they operate purely on URL strings. Lives under
``services/framework/storage_providers/utils/`` alongside ``s3.py`` and ``local.py``; the per-URL-family layout keeps
URL helpers grouped without conflating them with the concrete provider implementations.

Three URL forms are recognized:

- ``azure://<container>/<blob>`` — the scheme-tagged canonical form this codebase emits, mirroring ``s3://``. It
  carries no account, so the reading provider supplies its own.
- ``https://<account>.blob.core.windows.net/<container>/<blob>[?<sas>]`` — the public Azure endpoint, optionally
  carrying a SAS query string.
- ``http(s)://<host>[:<port>]/<account>/<container>/<blob>[?<sas>]`` — the path-style form used by the Azurite
  emulator and by sovereign / private-link endpoints. Because this shape is indistinguishable from an arbitrary HTTPS
  URL, it is only recognized when the caller passes the configured ``endpoint_url``.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote, unquote, urlparse

AZURE_URI_SCHEME = "azure"

# Schemes this module will compare origins for, with the port each implies when the URL omits one.
_DEFAULT_PORTS = {"http": 80, "https": 443}

# Public Azure Blob endpoint suffix. Private-link hostnames such as ``<account>.privatelink.blob.core.windows.net``
# also end with this, and still place the account in the first label, so the same parse applies to them.
AZURE_BLOB_HOST_SUFFIX = ".blob.core.windows.net"


@dataclass
class AzureBlobUrlComponents:
    """Parsed components of an Azure Blob Storage URL."""

    container_name: str
    blob_name: str
    # ``None`` for ``azure://`` URIs, which name no account; the provider fills in its configured one.
    account_name: str | None = None
    # SAS query string without the leading ``?``, or ``None`` when the URL carries no query.
    sas_token: str | None = None


def _endpoint_origin(endpoint_url: str) -> str:
    """Normalizes ``endpoint_url`` to a bare ``scheme://host[:port]`` origin with no trailing slash."""
    parsed = urlparse(endpoint_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _origin_parts(url: str) -> tuple[str, str, int] | None:
    """Splits ``url`` into a comparable ``(scheme, host, port)`` triple, or ``None`` when it has no usable origin.

    The port is resolved to the scheme's default when absent, so ``https://host`` and ``https://host:443`` compare
    equal. A malformed authority — including a port that is not a number — yields ``None`` rather than a partial
    match, because such a URL cannot be attributed to a host with confidence.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _DEFAULT_PORTS or not parsed.hostname:
        return None
    try:
        port = parsed.port
    except ValueError:
        return None
    return parsed.scheme, parsed.hostname.lower(), port or _DEFAULT_PORTS[parsed.scheme]


def matches_endpoint(url: str, endpoint_url: str | None) -> bool:
    """Reports whether ``url`` sits on the same origin as the configured ``endpoint_url``.

    Comparing origins rather than whole prefixes means an ``endpoint_url`` written with or without the account path
    segment behaves identically, and a caller cannot escape the configured host by varying the path.

    The comparison is on the parsed ``(scheme, host, port)`` triple, never on a string prefix. Callers use this to
    decide whether an SSRF check can be skipped, and a prefix test would hand that exemption to
    ``http://127.0.0.1:10000@evil.com/...`` (the authority's userinfo swallows the endpoint) and to
    ``https://<endpoint-host>.evil.com/...`` (a longer hostname sharing the endpoint's leading labels). Both resolve
    to an attacker-controlled host while still starting with the endpoint's origin.
    """
    if not endpoint_url:
        return False
    endpoint_origin = _origin_parts(endpoint_url)
    return endpoint_origin is not None and _origin_parts(url) == endpoint_origin


def is_azure_blob_url(url: str, endpoint_url: str | None = None) -> bool:
    """Reports whether ``url`` addresses Azure Blob Storage.

    Deliberately conservative: a plain ``https://`` URL is only treated as Azure when its host is a
    ``blob.core.windows.net`` endpoint, or when it sits under the ``endpoint_url`` the provider was configured with.
    Anything else (an S3 presigned URL, an arbitrary web host) must stay unrecognized so callers fall through to their
    own handling rather than feeding a foreign URL to the Azure SDK.

    Args:
        url: URL or URI to classify.
        endpoint_url: Optional configured blob endpoint, enabling the path-style form for emulators and private
            endpoints.

    Returns:
        True if ``url`` is an ``azure://`` URI or an HTTP(S) URL on a recognized blob endpoint.
    """
    parsed = urlparse(url)
    if parsed.scheme == AZURE_URI_SCHEME:
        return True
    if parsed.scheme not in ("http", "https"):
        return False
    hostname = (parsed.hostname or "").lower()
    if hostname.endswith(AZURE_BLOB_HOST_SUFFIX):
        return True
    return matches_endpoint(url, endpoint_url)


def parse_azure_url(url: str, endpoint_url: str | None = None) -> AzureBlobUrlComponents:
    """Parses an Azure Blob Storage URL into its account, container, and blob components.

    Args:
        url: ``azure://<container>/<blob>``, ``https://<account>.blob.core.windows.net/<container>/<blob>``, or — when
            ``endpoint_url`` is supplied — ``<endpoint>/<account>/<container>/<blob>``.
        endpoint_url: Optional configured blob endpoint, enabling the path-style form.

    Returns:
        AzureBlobUrlComponents with the container, blob name, account (``None`` for ``azure://``), and SAS token.

    Raises:
        ValueError: If the URL is not a recognized Azure Blob URL, or names a container without a blob.
    """
    parsed = urlparse(url)

    if parsed.scheme == AZURE_URI_SCHEME:
        container_name = parsed.netloc
        blob_name = unquote(parsed.path.lstrip("/"))
        if not container_name or not blob_name:
            raise ValueError(f"Invalid azure:// URI, expected azure://<container>/<blob>: {url}")
        return AzureBlobUrlComponents(
            container_name=container_name, blob_name=blob_name, sas_token=parsed.query or None
        )

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme for Azure Blob Storage: {parsed.scheme or url!r}")

    hostname = (parsed.hostname or "").lower()
    path = unquote(parsed.path.lstrip("/"))
    sas_token = parsed.query or None

    if hostname.endswith(AZURE_BLOB_HOST_SUFFIX):
        account_name = hostname[: -len(AZURE_BLOB_HOST_SUFFIX)].split(".", 1)[0]
        container_name, _, blob_name = path.partition("/")
        if not account_name or not container_name or not blob_name:
            raise ValueError(f"Invalid Azure blob URL, expected <account>host/<container>/<blob>: {url}")
        return AzureBlobUrlComponents(
            container_name=container_name, blob_name=blob_name, account_name=account_name, sas_token=sas_token
        )

    if matches_endpoint(url, endpoint_url):
        # Path style: the account is the first segment after the endpoint host.
        account_name, _, remainder = path.partition("/")
        container_name, _, blob_name = remainder.partition("/")
        if not account_name or not container_name or not blob_name:
            raise ValueError(
                f"Invalid path-style Azure blob URL, expected <endpoint>/<account>/<container>/<blob>: {url}"
            )
        return AzureBlobUrlComponents(
            container_name=container_name, blob_name=blob_name, account_name=account_name, sas_token=sas_token
        )

    raise ValueError(f"Unrecognized Azure Blob URL format: {url}")


def _quote_blob_path(blob_name: str) -> str:
    """Percent-encodes ``blob_name`` for use as a URL path, leaving ``/`` intact as a separator.

    Blob names may legally contain characters that are structural in a URL: an unescaped ``?`` would open the SAS
    query string and ``#`` a fragment, so :func:`parse_azure_url` — which unquotes the path it reads — would hand back
    a name truncated at that character. Encoding on the way out is what makes that round trip exact.
    """
    return quote(blob_name.lstrip("/"), safe="/")


def build_azure_uri(container_name: str, blob_name: str) -> str:
    """Builds the canonical ``azure://<container>/<blob>`` URI for a blob.

    This is the scheme-tagged form stored as ``StorageUrls.raw``, chosen over the HTTPS endpoint URL for the same
    reason S3 outputs are recorded as ``s3://``: it is unambiguously a storage reference rather than something a
    client might try to fetch directly, and it stays valid if the account's endpoint changes.
    """
    return f"{AZURE_URI_SCHEME}://{container_name}/{_quote_blob_path(blob_name)}"


def build_account_url(account_name: str, endpoint_url: str | None = None) -> str:
    """Builds the account-scoped service URL the Azure SDK's ``BlobServiceClient`` expects.

    With no ``endpoint_url`` this is the public ``https://<account>.blob.core.windows.net``. With one, the account
    becomes the first path segment under that endpoint's origin — the layout Azurite and private endpoints use, and
    the one :func:`parse_azure_url` reads back.
    """
    if endpoint_url:
        return f"{_endpoint_origin(endpoint_url)}/{account_name}"
    return f"https://{account_name}{AZURE_BLOB_HOST_SUFFIX}"


def build_blob_url(
    container_name: str,
    blob_name: str,
    account_name: str,
    endpoint_url: str | None = None,
    sas_token: str | None = None,
) -> str:
    """Builds the HTTP(S) URL for a blob, optionally with a SAS query string appended.

    Args:
        container_name: Container holding the blob.
        blob_name: Blob name (key) within the container.
        account_name: Storage account name.
        endpoint_url: Optional blob endpoint override, e.g. ``http://127.0.0.1:10000`` for Azurite.
        sas_token: Optional SAS query string, with or without a leading ``?``.

    Returns:
        An absolute HTTP(S) URL addressing the blob.
    """
    base = build_account_url(account_name, endpoint_url)
    url = f"{base}/{container_name}/{_quote_blob_path(blob_name)}"
    if sas_token:
        url = f"{url}?{sas_token.lstrip('?')}"
    return url
