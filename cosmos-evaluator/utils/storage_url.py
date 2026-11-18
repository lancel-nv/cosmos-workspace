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

"""Parsing of the single storage URL that addresses a remote storage backend.

One URL carries every addressing component a backend needs — the bucket or container, the Azure account, a
non-default endpoint, and the AWS region — so a deployment names its target once instead of setting four
loosely-coupled variables that must agree with each other.

Lives under ``utils/`` rather than beside the storage providers because the settings layer and the provider factory
both parse these URLs, and the settings layer sits below the providers.

The backend is supplied by the caller rather than inferred from the URL. Only the well-known public hosts are
self-identifying; a custom endpoint such as ``https://pbss.s8k.io/my-bucket`` is indistinguishable from a private
Azure endpoint, so guessing would silently address the wrong cloud.

Local storage has no address to parse: it resolves keys under a base directory supplied by its caller.

A caller whose objects span buckets addresses the store alone, with no bucket, through :func:`parse_s3_endpoint_url`.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from urllib.parse import ParseResult, unquote, urlparse

S3_URI_SCHEME = "s3"

_AWS_HOST_SUFFIX = ".amazonaws.com"
_AZURE_BLOB_HOST_SUFFIX = ".blob.core.windows.net"
_HTTP_SCHEMES = ("http", "https")

# AWS region labels ("us-west-2", "ap-southeast-1"). Matching a shape rather than a fixed list keeps new regions
# working, while still rejecting the other labels that appear in that host position (e.g. "dualstack"), which would
# otherwise be handed to boto3 as a region and fail at signing time.
_REGION_PATTERN = re.compile(r"^[a-z]{2}(-[a-z]+)+-\d+$")


@dataclass(frozen=True)
class StorageTarget:
    """Addressing components for one storage backend, parsed from a storage URL."""

    # S3 bucket or Azure container.
    container: str
    # Azure storage account. ``None`` for every other backend, which has no equivalent.
    account: str | None = None
    # Origin of a non-default endpoint (``scheme://host[:port]``), for S3-compatible clusters, Azurite, and private
    # Azure endpoints. ``None`` when the URL names a public cloud endpoint the SDK derives on its own.
    endpoint_url: str | None = None
    # AWS region, when the host carries one. ``None`` for every other backend and for custom S3 endpoints, leaving
    # the caller to supply a fallback.
    region: str | None = None


@dataclass(frozen=True)
class S3Endpoint:
    """Where an S3 store is reached, without naming a bucket in it."""

    # Origin (``scheme://host[:port]``) of an S3-compatible endpoint. ``None`` for AWS, whose endpoint the SDK derives
    # from the region.
    endpoint_url: str | None
    # AWS region named by the host. ``None`` for an S3-compatible endpoint, and for an AWS host naming no region.
    region: str | None

    @property
    def is_aws(self) -> bool:
        """Whether the URL named AWS itself rather than an S3-compatible endpoint."""
        return self.endpoint_url is None


def _origin(parsed: ParseResult) -> str:
    """Rebuilds the bare ``scheme://host[:port]`` origin, dropping path, query, and any authority userinfo."""
    host = parsed.hostname or ""
    return f"{parsed.scheme}://{host}:{parsed.port}" if parsed.port else f"{parsed.scheme}://{host}"


def _path_segments(path: str) -> list[str]:
    """Splits a URL path into non-empty, percent-decoded segments."""
    return [unquote(segment) for segment in path.split("/") if segment]


def _reject_extra_segments(extra: list[str], url: str) -> None:
    """Rejects path segments beyond the ones that address the container.

    A storage URL names a container, never a key inside it. Key prefixes are passed separately by the caller, so a
    deeper path is a configuration error that would otherwise be silently discarded.
    """
    if extra:
        raise ValueError(f"Storage URL carries a path below the container ({'/'.join(extra)!r}): {url!r}")


def _parse_aws_host(host: str, segments: list[str], url: str) -> tuple[str, str | None, int]:
    """Extracts the bucket and region from an ``*.amazonaws.com`` host.

    Handles both addressing styles the SDK emits: virtual-hosted (``<bucket>.s3.<region>.amazonaws.com``), where the
    bucket may itself contain dots, and path-style (``s3.<region>.amazonaws.com/<bucket>``).

    Returns:
        The bucket, the region when the host carries a recognizable one, and how many path segments were consumed
        naming the bucket (one for path-style, zero for virtual-hosted).
    """
    labels = host[: -len(_AWS_HOST_SUFFIX)].split(".")
    service_index = next((i for i, label in enumerate(labels) if label == "s3" or label.startswith("s3-")), None)
    if service_index is None:
        raise ValueError(f"Not an S3 host, expected an 's3' label in {host!r}: {url!r}")

    service_label = labels[service_index]
    if service_label.startswith("s3-"):
        candidate = service_label[len("s3-") :]
    else:
        candidate = labels[service_index + 1] if service_index + 1 < len(labels) else ""
    region = candidate if _REGION_PATTERN.match(candidate) else None

    if service_index == 0:
        if not segments:
            raise ValueError(f"Path-style S3 URL names no bucket: {url!r}")
        return segments[0], region, 1
    return ".".join(labels[:service_index]), region, 0


def _parse_s3(url: str) -> StorageTarget:
    """Reads an S3 target from an ``s3://`` URI, an AWS endpoint URL, or a custom S3-compatible endpoint."""
    parsed = urlparse(url)

    if parsed.scheme == S3_URI_SCHEME:
        if not parsed.netloc:
            raise ValueError(f"S3 URI names no bucket, expected s3://<bucket>: {url!r}")
        _reject_extra_segments(_path_segments(parsed.path), url)
        return StorageTarget(container=parsed.netloc)

    if parsed.scheme not in _HTTP_SCHEMES:
        raise ValueError(f"Unsupported scheme for S3 storage, expected s3://, http:// or https://: {url!r}")

    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError(f"S3 endpoint URL names no host: {url!r}")
    segments = _path_segments(parsed.path)

    if host.endswith(_AWS_HOST_SUFFIX):
        bucket, region, consumed = _parse_aws_host(host, segments, url)
        _reject_extra_segments(segments[consumed:], url)
        return StorageTarget(container=bucket, region=region)

    if host.endswith(_AZURE_BLOB_HOST_SUFFIX):
        raise ValueError(f"URL addresses Azure Blob Storage, not S3; set the storage type to 'azure': {url!r}")

    if not segments:
        raise ValueError(f"S3 endpoint URL names no bucket, expected <endpoint>/<bucket>: {url!r}")
    _reject_extra_segments(segments[1:], url)
    return StorageTarget(container=segments[0], endpoint_url=_origin(parsed))


def _parse_azure(url: str) -> StorageTarget:
    """Reads an Azure target from a public blob host or a path-style private/emulator endpoint.

    Azure is addressed only by its HTTPS forms. The account is structural rather than a credential — it is the first
    label of the public hostname, and the first path segment behind a custom endpoint — so there is no shorthand that
    can omit it.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _HTTP_SCHEMES:
        raise ValueError(
            "Unsupported scheme for Azure storage, expected https://<account>.blob.core.windows.net/<container> "
            f"or <endpoint>/<account>/<container>: {url!r}"
        )

    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError(f"Azure storage URL names no host: {url!r}")
    segments = _path_segments(parsed.path)

    if host.endswith(_AZURE_BLOB_HOST_SUFFIX):
        account = host[: -len(_AZURE_BLOB_HOST_SUFFIX)].split(".", 1)[0]
        if not account or not segments:
            raise ValueError(
                "Azure storage URL names no account or container, expected "
                f"https://<account>.blob.core.windows.net/<container>: {url!r}"
            )
        _reject_extra_segments(segments[1:], url)
        return StorageTarget(container=segments[0], account=account)

    if host.endswith(_AWS_HOST_SUFFIX):
        raise ValueError(f"URL addresses AWS, not Azure Blob Storage; set the storage type to 's3': {url!r}")

    if len(segments) < 2:
        raise ValueError(f"Azure endpoint URL expects <endpoint>/<account>/<container>: {url!r}")
    _reject_extra_segments(segments[2:], url)
    return StorageTarget(container=segments[1], account=segments[0], endpoint_url=_origin(parsed))


def parse_storage_url(url: str, storage_type: str) -> StorageTarget:
    """Parses a storage URL into the addressing components ``storage_type``'s backend needs.

    Accepted forms per backend:

    - ``s3``: ``s3://my-bucket``, an AWS endpoint in either addressing style
      (``https://my-bucket.s3.us-west-2.amazonaws.com``, ``https://s3.us-west-2.amazonaws.com/my-bucket``), or a
      custom S3-compatible endpoint (``https://pbss.s8k.io/my-bucket``).
    - ``azure``: ``https://myacct.blob.core.windows.net/my-container``, or a path-style private endpoint or emulator
      (``http://127.0.0.1:10000/devstoreaccount1/my-container``).

    Only an AWS host carries a region; ``s3://`` and custom endpoints leave :attr:`StorageTarget.region` as ``None``
    for the caller to fill in.

    Args:
        url: The storage URL naming the target container.
        storage_type: Remote backend the URL addresses (``"s3"`` or ``"azure"``).

    Returns:
        The parsed addressing components.

    Raises:
        ValueError: If ``storage_type`` names no remote backend, or the URL does not match a form that backend
            accepts.
    """
    if not url:
        raise ValueError(f"Storage URL is empty for {storage_type!r} storage")
    if storage_type == "s3":
        return _parse_s3(url)
    if storage_type == "azure":
        return _parse_azure(url)
    if storage_type == "local":
        raise ValueError("Local storage is addressed by its base directory, not by a storage URL")
    raise ValueError(f"Unsupported storage type: {storage_type}")


def parse_s3_endpoint_url(url: str) -> S3Endpoint:
    """Parses a URL naming an S3 store but no bucket in it.

    For callers whose objects span buckets, such as a provider that downloads absolute object URLs and signs each
    request for the bucket its URL names. Accepts an AWS service endpoint (``https://s3.us-east-1.amazonaws.com``, the
    legacy ``https://s3-us-east-1.amazonaws.com``, or the regionless ``https://s3.amazonaws.com``) or the origin of an
    S3-compatible endpoint (``https://pbss.s8k.io``).

    Raises:
        ValueError: If the URL is empty, is not ``http(s)``, carries a path, names a bucket in an AWS virtual-hosted
            host, or is a public Azure blob host.
    """
    if not url:
        raise ValueError("S3 endpoint URL is empty")
    parsed = urlparse(url)
    if parsed.scheme not in _HTTP_SCHEMES:
        raise ValueError(f"Unsupported scheme for an S3 endpoint, expected http:// or https://: {url!r}")
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError(f"S3 endpoint URL names no host: {url!r}")
    if _path_segments(parsed.path):
        raise ValueError(f"S3 endpoint URL carries a path; it names a store, not a bucket in it: {url!r}")
    if host.endswith(_AZURE_BLOB_HOST_SUFFIX):
        raise ValueError(f"URL addresses Azure Blob Storage, not S3: {url!r}")

    if host.endswith(_AWS_HOST_SUFFIX):
        labels = host[: -len(_AWS_HOST_SUFFIX)].split(".")
        service_label = labels[0]
        if service_label.startswith("s3-"):
            candidate = service_label[len("s3-") :]
        elif service_label == "s3":
            candidate = labels[1] if len(labels) > 1 else ""
        else:
            raise ValueError(f"AWS host names a bucket; expected the endpoint s3.<region>.amazonaws.com: {url!r}")
        return S3Endpoint(endpoint_url=None, region=candidate if _REGION_PATTERN.match(candidate) else None)

    return S3Endpoint(endpoint_url=_origin(parsed), region=None)
