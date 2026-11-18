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

import os
from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class S3ObjectNotFoundError(RuntimeError):
    """Raised when an expected S3 object is missing (404 / NoSuchKey).

    Distinguishes "the object isn't there" from generic ``ClientError`` so callers can decide policy per object (e.g.
    fail fast on required inputs but skip optional ones silently).
    """


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """
    Parse an S3 URI into a bucket and key.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def list_s3_media(prefix_uri: str) -> list[str]:
    """
    List all media files in the given S3 prefix.
    """
    bucket, prefix = parse_s3_uri(prefix_uri)
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    media: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key: str = obj["Key"]
            ext = os.path.splitext(key)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                media.append(f"s3://{bucket}/{key}")
    return media


def download_to_temp(path: str) -> Path:
    """
    Download an S3 file to a temp file.
    """
    bucket, key = parse_s3_uri(path)
    suffix = os.path.splitext(path)[1]
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket, key, tmp_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    return Path(tmp_path)


def is_s3_path(path: str) -> bool:
    return path.startswith("s3://")


def load_s3_cluster_credentials(cluster: str | None) -> dict[str, str]:
    """Load S3 credentials and connection details for a named cluster.

    Resolves four logical fields — ``access_key_id``, ``secret_access_key``, ``region``, ``endpoint_url`` — by reading
    cluster-prefixed environment variables of the form ``<PREFIX>_AWS_<SUFFIX>``. ``PREFIX`` is the upper-cased
    ``cluster`` argument (e.g. ``PBSS_AWS_ACCESS_KEY_ID`` for ``cluster='pbss'``). ``SUFFIX`` is the upper-cased suffix
    of the environment variable (e.g. ``ACCESS_KEY_ID``, ``SECRET_ACCESS_KEY``, ``DEFAULT_REGION``, ``ENDPOINT_URL``).
    When ``cluster`` is ``None``, the plain ``AWS_<SUFFIX>`` names are used directly.

    Per-variable silent fallback: if a cluster-prefixed variable is unset, the plain ``AWS_<SUFFIX>`` value is used in
    its place. This lets users keep cluster-specific creds in ``~/.cosmos_evaluator/.env`` (forwarded by dazel
    automatically) and shared AWS creds in ``~/.aws/.env``.

    Args:
        cluster: A cluster name (``'pbss'``, ``'pdx'``), or ``None`` to use unprefixed ``AWS_*`` directly.

    Returns:
        Dict with string values for ``access_key_id``, ``secret_access_key``, ``region``, and ``endpoint_url``. Any
        field may be the empty string when no env var is set; callers decide what's required.
    """
    prefix = f"{cluster.upper()}_AWS_" if cluster else "AWS_"

    def _resolve(suffix: str) -> str:
        value = os.environ.get(prefix + suffix)
        if value:
            return value.strip().strip("\"'")
        # Per-variable fallback to plain AWS_<suffix>.
        return os.environ.get("AWS_" + suffix, "").strip().strip("\"'")

    return {
        "access_key_id": _resolve("ACCESS_KEY_ID"),
        "secret_access_key": _resolve("SECRET_ACCESS_KEY"),
        "region": _resolve("DEFAULT_REGION") or _resolve("REGION"),
        "endpoint_url": _resolve("ENDPOINT_URL"),
    }


def create_s3_client(
    endpoint_url: str | None = None,
    region_name: str | None = None,
    max_pool_connections: int = 100,
    signature_version: str = "s3v4",
    cluster: str | None = None,
) -> Any:
    """Create an S3 client supporting both AWS and custom endpoints (e.g. PDX, PBSS).

    For AWS endpoints (amazonaws.com), omits endpoint_url so boto3 uses default regional routing.  For custom endpoints,
    uses credentials from env vars AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY — or, when a ``cluster`` is given, the
    cluster-prefixed equivalents (e.g. ``PBSS_AWS_ACCESS_KEY_ID``) with silent per-variable fallback to plain ``AWS_*``.
    See :func:`load_s3_cluster_credentials`.

    The default ``signature_version='s3v4'`` works for AWS and most S3-compatible endpoints. NVIDIA PBSS
    (``https://pbss.s8k.io``) rejects HeadObject under s3v4 with HTTP 400, so callers that need to use ``download_file``
    (which HEADs the object before transfer) against PBSS must pass ``signature_version='s3'``. GET/PUT/LIST work under
    both signing versions, which is why this gap had not surfaced before.

    Endpoint precedence: ``endpoint_url`` arg > cluster's ``<PREFIX>_AWS_ENDPOINT_URL`` (with ``AWS_ENDPOINT_URL``
    fallback) > ``S3_ENDPOINT_URL`` env > ``https://pdx.s8k.io`` (legacy default).

    Env vars are read at call time (not import time) so load_dotenv() takes effect.
    """
    creds = load_s3_cluster_credentials(cluster)
    region = region_name or creds["region"] or "us-east-1"
    access_key = creds["access_key_id"]
    secret_key = creds["secret_access_key"]
    if endpoint_url and "amazonaws.com" in endpoint_url:
        kwargs: dict[str, Any] = {
            "region_name": region,
            "config": BotoConfig(signature_version=signature_version, max_pool_connections=max_pool_connections),
        }
        if access_key and secret_key:
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key
        return boto3.client("s3", **kwargs)
    url = (
        (endpoint_url or creds["endpoint_url"] or os.getenv("S3_ENDPOINT_URL") or "https://pdx.s8k.io")
        .strip()
        .rstrip("/")
    )
    if not access_key or not secret_key:
        raise ValueError(
            "S3 credentials not set. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (or cluster-prefixed variants "
            "like PBSS_AWS_ACCESS_KEY_ID) for non-AWS endpoints."
        )
    return boto3.client(
        "s3",
        endpoint_url=url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        config=BotoConfig(signature_version=signature_version, max_pool_connections=max_pool_connections),
    )


def object_exists(client: Any, s3_uri: str) -> bool:
    """Return whether the object at ``s3_uri`` exists, via a HEAD request.

    A ``404`` / ``NoSuchKey`` / ``NotFound`` HEAD response means "the object isn't there" and yields ``False``. Other
    ``ClientError`` cases (auth, throttling, ...) propagate unchanged so a transient or permission failure is never
    silently reported as a missing object.

    Args:
        client: A boto3 S3 client, typically from :func:`create_s3_client`.
        s3_uri: Object URI in the form ``s3://<bucket>/<key>``.

    Returns:
        ``True`` if the object exists, ``False`` if it is definitively absent.
    """
    bucket, key = parse_s3_uri(s3_uri)
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code") if exc.response else ""
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def download_s3_object(client: Any, s3_uri: str, local_path: Path) -> None:
    """Download one S3 object to ``local_path`` with a typed not-found error.

    Probes the object with :func:`object_exists` first and raises :class:`S3ObjectNotFoundError` when it is absent, so
    callers can distinguish "the object isn't there" from generic transfer failures. The caller controls the
    destination layout and any caching strategy on top.

    Args:
        client: A boto3 S3 client, typically from :func:`create_s3_client`.
        s3_uri: Source URI in the form ``s3://<bucket>/<key>``.
        local_path: Destination path on disk. The parent directory is the caller's responsibility.

    Raises:
        S3ObjectNotFoundError: If the object does not exist.
    """
    if not object_exists(client, s3_uri):
        raise S3ObjectNotFoundError(f"S3 object not found: {s3_uri}")
    bucket, key = parse_s3_uri(s3_uri)
    client.download_file(bucket, key, str(local_path))
