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

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
import json
import logging
import mimetypes
from pathlib import Path
import ssl
from types import TracebackType
from typing import Any, NoReturn, Self
from urllib.parse import urljoin

import aioboto3
import aiofiles
import aiohttp
from botocore.config import Config as BotoCoreConfig
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
import certifi

from services.framework.protocols.storage_provider import StorageUrls
from services.framework.storage_providers.config import S3Config
from services.framework.storage_providers.exceptions import (
    S3StorageError,
    StorageAuthenticationError,
    StorageConnectionError,
    StorageNotFoundError,
    StoragePermissionError,
    StorageQuotaExceededError,
)
from services.framework.storage_providers.utils.local import is_local_url, symlink_local_url
from services.framework.storage_providers.utils.s3 import parse_s3_url
from services.framework.url.security import check_url_security
from services.settings_base import SettingsBase
from utils.storage_url import parse_storage_url

logger = logging.getLogger(__name__)

# Failure modes that should degrade the parallel/probe download path to a single streaming GET rather than abort:
# HTTP/network errors (aiohttp), socket-read timeouts, file I/O errors (open/seek/write/truncate; ssl.SSLError is an
# OSError), and the S3StorageError we raise explicitly on a non-206 response or a short read. A genuine programming
# error (AttributeError, etc.) is intentionally left to propagate.
_RANGED_DOWNLOAD_FALLBACK_ERRORS = (aiohttp.ClientError, OSError, TimeoutError, S3StorageError)

# Transient transfer failures worth retrying in place before giving up on a download attempt. ClientPayloadError covers
# aiohttp 3.14's spurious ContentLengthError when a clean close lands while the parser is paused by back-pressure (all
# bytes received, message never finalized); ClientConnectionError covers disconnects plus connector and OS-level socket
# errors; asyncio.TimeoutError covers sock_read/sock_connect expiry (SocketTimeoutError subclasses it).
_RETRYABLE_DOWNLOAD_ERRORS = (aiohttp.ClientPayloadError, aiohttp.ClientConnectionError, asyncio.TimeoutError)


def _is_retryable_download_error(exc: BaseException) -> bool:
    """Reports whether a failed download attempt is worth retrying as-is.

    Deliberately narrower than :data:`_RANGED_DOWNLOAD_FALLBACK_ERRORS`: the ``S3StorageError`` raised for a non-206
    response, a short read, or a refused redirect must keep degrading to the single-stream path immediately rather than
    burning every attempt first. A ``ClientResponseError`` is retried only for 5xx — a 403 from an expired presigned URL
    or a 404 will never succeed on retry.
    """
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status >= 500
    return isinstance(exc, _RETRYABLE_DOWNLOAD_ERRORS)


def _describe_download_error(exc: BaseException) -> str:
    """Renders ``exc`` for a log line without echoing the URL it may be carrying.

    ``ClientResponseError`` embeds ``request_info.real_url`` in both ``__str__`` and ``__repr__``, and for these
    downloads that URL is a presigned one whose query string holds ``X-Amz-Credential`` / ``X-Amz-Signature``. Only
    the type and status are reportable. Every other transfer failure carries a URL-free message (a payload error
    quotes byte counts, a connector error quotes host:port), so its text is kept for diagnostics.
    """
    if isinstance(exc, aiohttp.ClientResponseError):
        return f"{type(exc).__name__}(status={exc.status})"
    detail = str(exc)
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


class S3StorageProvider:
    """Amazon S3 storage provider implementation.

    Implements the StorageProvider protocol for Amazon S3 and S3-compatible storage services. Supports async operations,
    multipart uploads, server-side encryption, and comprehensive error handling.
    """

    DEFAULT_PRESIGNED_URL_EXPIRATION = 43200  # 12 hours

    def __init__(self, config: S3Config) -> None:
        """Initializes S3 storage provider.

        Args:
            config: S3 configuration object
        """
        self.config = config
        self._session: aioboto3.Session | None = None
        self._client: Any = None
        self._client_ctx: Any = None
        self._client_lock = asyncio.Lock()

        # Setup boto3 configuration
        boto_config_kwargs: dict[str, Any] = {
            "retries": {"max_attempts": config.max_attempts, "mode": config.retry_mode},
            "max_pool_connections": config.max_concurrency,
        }
        # Only override the signature version when explicitly configured; otherwise leave
        # botocore on its default (v4). v2 ('s3') is needed for clusters like PBSS.
        if config.signature_version:
            boto_config_kwargs["signature_version"] = config.signature_version
        self._boto_config = BotoCoreConfig(**boto_config_kwargs)

    @classmethod
    def from_settings(cls, bucket_name: str) -> S3StorageProvider:
        """Creates an S3StorageProvider from SettingsBase environment configuration.

        The caller names the bucket, which need not be the one the configured storage URL addresses; only the region
        is taken from that URL, and only when it is an S3 URL carrying one. Signing against the wrong region makes
        AWS reject every request, so an unset region falls back to ``us-east-1`` and its global endpoint.

        Args:
            bucket_name: S3 bucket name

        Returns:
            Configured S3StorageProvider instance
        """
        settings = SettingsBase(_env_file=SettingsBase.get_env_files())
        storage_url = settings.output_storage_url or settings.storage_url
        region = (
            parse_storage_url(storage_url, settings.output_storage_type).region
            if storage_url and settings.output_storage_type == "s3"
            else None
        )
        credentials = settings.s3_credentials
        config = S3Config(
            bucket_name=bucket_name,
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.session_token,
            region_name=region or "us-east-1",
        )
        return cls(config)

    async def _get_client(self) -> Any:
        """Gets or creates S3 client."""
        async with self._client_lock:
            if self._client is None:
                if self._session is None:
                    self._session = aioboto3.Session()

                boto3_config = self.config.get_boto3_config()
                self._client_ctx = self._session.client("s3", config=self._boto_config, **boto3_config)
                self._client = await self._client_ctx.__aenter__()
        return self._client

    def _full_key(self, key: str) -> str:
        """Prepends the configured key_prefix to a storage key.

        Args:
            key: Storage key/path

        Returns:
            Key with prefix prepended
        """
        return f"{self.config.key_prefix}{key}"

    def make_key_prefix(self, *parts: str | None) -> str:
        """Joins parts as object-store key segments.

        S3 keys are opaque strings with no notion of an absolute path; a leading ``/`` would create an empty top-level
        path component that confuses tooling and breaks this codebase's URL round-trip in ``utils.s3.parse_s3_url``
        (which calls ``lstrip("/")``). To keep keys clean we strip leading and trailing slashes from every part, drop
        empty/None parts, join the rest with ``/``, and append a trailing ``/`` so callers can concatenate filenames.
        """
        cleaned = [p.strip("/") for p in parts if p]
        cleaned = [p for p in cleaned if p]
        return ("/".join(cleaned) + "/") if cleaned else ""

    def _handle_s3_error(self, error: Exception, operation: str, key: str) -> NoReturn:
        """Handles and converts S3 errors to storage exceptions.

        Args:
            error: Original exception
            operation: Operation being performed
            key: Storage key involved

        Raises:
            Appropriate StorageError subclass
        """
        if isinstance(error, ClientError):
            error_code = error.response["Error"]["Code"]
            error_message = error.response["Error"]["Message"]

            if error_code == "NoSuchBucket":
                raise S3StorageError(
                    f"Bucket '{self.config.bucket_name}' does not exist",
                    key=key,
                    operation=operation,
                    bucket=self.config.bucket_name,
                    aws_error_code=error_code,
                )
            elif error_code == "NoSuchKey":
                raise StorageNotFoundError(
                    f"Object '{key}' not found in bucket '{self.config.bucket_name}'", key=key, operation=operation
                )
            elif error_code in ["AccessDenied", "Forbidden"]:
                raise StoragePermissionError(
                    f"Access denied for operation '{operation}' on key '{key}': {error_message}",
                    key=key,
                    operation=operation,
                )
            elif error_code in ["QuotaExceeded", "ServiceQuotaExceeded"]:
                raise StorageQuotaExceededError(
                    f"Storage quota exceeded: {error_message}", key=key, operation=operation
                )
            else:
                raise S3StorageError(
                    f"S3 error during {operation}: {error_message}",
                    key=key,
                    operation=operation,
                    bucket=self.config.bucket_name,
                    aws_error_code=error_code,
                    details=error.response,
                )
        elif isinstance(error, NoCredentialsError):
            raise StorageAuthenticationError("AWS credentials not found or invalid", key=key, operation=operation)
        elif isinstance(error, BotoCoreError):
            raise StorageConnectionError(
                f"Connection error during {operation}: {error!s}", key=key, operation=operation
            )
        else:
            raise S3StorageError(
                f"Unexpected error during {operation}: {error!s}",
                key=key,
                operation=operation,
                bucket=self.config.bucket_name,
            )

    def _prepare_data_for_storage(self, data: Any) -> bytes:
        """Prepares data for storage by converting to bytes.

        Args:
            data: Data to prepare

        Returns:
            Data as bytes

        Raises:
            S3StorageError: If data cannot be serialized
        """
        if isinstance(data, bytes):
            return data
        elif isinstance(data, str):
            return data.encode("utf-8")
        elif isinstance(data, (dict, list)):
            try:
                return json.dumps(data, ensure_ascii=False).encode("utf-8")
            except (TypeError, ValueError) as e:
                raise S3StorageError(f"Failed to serialize data to JSON: {e}") from e
        else:
            try:
                return str(data).encode("utf-8")
            except Exception as e:
                raise S3StorageError(f"Failed to convert data to bytes: {e}") from e

    def _prepare_metadata(self, metadata: dict[str, Any] | None) -> dict[str, str]:
        """Prepares metadata for S3 storage.

        Args:
            metadata: Original metadata

        Returns:
            S3-compatible metadata (string values only)
        """
        if not metadata:
            return {}

        s3_metadata = {}
        for key, value in metadata.items():
            # S3 metadata keys must be lowercase and contain only letters, numbers, and hyphens
            clean_key = key.lower().replace("_", "-")
            if isinstance(value, str):
                s3_metadata[clean_key] = value
            else:
                try:
                    s3_metadata[clean_key] = json.dumps(value)
                except (TypeError, ValueError):
                    s3_metadata[clean_key] = str(value)

        return s3_metadata

    def _determine_content_type(self, data: Any, key: str) -> str:
        """Determines content type based on data and key.

        Args:
            data: Data being stored
            key: Storage key

        Returns:
            Content type string
        """
        if isinstance(data, str):
            return "text/plain; charset=utf-8"
        elif isinstance(data, (dict, list)):
            return "application/json; charset=utf-8"
        else:
            # Try to guess content type from file extension
            content_type, _ = mimetypes.guess_type(key)
            return content_type or self.config.default_content_type

    async def store(self, data: Any, key: str, metadata: dict[str, Any] | None = None) -> str:
        """Stores data in S3.

        Args:
            data: Data to store (bytes, string, or serializable object)
            key: Storage key/path
            metadata: Optional metadata to attach

        Returns:
            S3 URL for the stored object

        Raises:
            S3StorageError: If storage operation fails
        """
        full_key = self._full_key(key)
        try:
            client = await self._get_client()

            # Prepare data
            data_bytes = self._prepare_data_for_storage(data)
            content_type = self._determine_content_type(data, key)
            s3_metadata = self._prepare_metadata(metadata)

            # Prepare upload parameters
            upload_params = {
                "Bucket": self.config.bucket_name,
                "Key": full_key,
                "Body": data_bytes,
                "ContentType": content_type,
                "Metadata": s3_metadata,
                "StorageClass": self.config.storage_class,
            }

            # Add server-side encryption if configured
            if self.config.server_side_encryption:
                upload_params["ServerSideEncryption"] = self.config.server_side_encryption
                if self.config.kms_key_id and self.config.server_side_encryption == "aws:kms":
                    upload_params["SSEKMSKeyId"] = self.config.kms_key_id

            # Upload the object
            await client.put_object(**upload_params)

            logger.info("Successfully stored object: %s (%s bytes)", full_key, len(data_bytes))

            # Return S3 URL
            if self.config.endpoint_url:
                base_url = self.config.endpoint_url
                if not base_url.endswith("/"):
                    base_url += "/"
                return urljoin(base_url, f"{self.config.bucket_name}/{full_key}")
            else:
                return f"https://s3.{self.config.region_name}.amazonaws.com/{self.config.bucket_name}/{full_key}"

        except (ClientError, NoCredentialsError, BotoCoreError) as e:
            self._handle_s3_error(e, "store", full_key)

    async def retrieve(self, key: str) -> Any:
        """Retrieves data from S3.

        Args:
            key: Storage key/path

        Returns:
            Retrieved data (attempts to deserialize JSON, otherwise returns bytes)

        Raises:
            StorageNotFoundError: If object doesn't exist
            S3StorageError: If retrieval fails
        """
        full_key = self._full_key(key)
        try:
            client = await self._get_client()

            response = await client.get_object(Bucket=self.config.bucket_name, Key=full_key)

            # Read the data
            data_bytes = await response["Body"].read()

            # Try to determine the original data type and deserialize appropriately
            content_type = response.get("ContentType", "")

            if content_type.startswith("application/json") or full_key.endswith(".json"):
                try:
                    return json.loads(data_bytes.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # If JSON parsing fails, return as bytes
                    pass
            elif content_type.startswith("text/"):
                try:
                    return data_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    # If text decoding fails, return as bytes
                    pass

            # Default: return as bytes
            return data_bytes

        except (ClientError, NoCredentialsError, BotoCoreError) as e:
            self._handle_s3_error(e, "retrieve", full_key)

    async def delete(self, key: str) -> bool:
        """Deletes data from S3.

        Args:
            key: Storage key/path

        Returns:
            True if deletion was successful

        Raises:
            S3StorageError: If deletion fails
        """
        full_key = self._full_key(key)
        try:
            client = await self._get_client()

            await client.delete_object(Bucket=self.config.bucket_name, Key=full_key)

            logger.info("Successfully deleted object: %s", full_key)
            return True

        except (ClientError, NoCredentialsError, BotoCoreError) as e:
            self._handle_s3_error(e, "delete", full_key)

    async def exists(self, key: str) -> bool:
        """Checks if data exists in S3.

        Args:
            key: Storage key/path

        Returns:
            True if object exists
        """
        full_key = self._full_key(key)
        try:
            client = await self._get_client()

            await client.head_object(Bucket=self.config.bucket_name, Key=full_key)
            return True

        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "Not Found"):
                return False
            self._handle_s3_error(e, "exists", full_key)
        except (NoCredentialsError, BotoCoreError) as e:
            self._handle_s3_error(e, "exists", full_key)

    async def list_keys(self, prefix: str = "") -> AsyncIterator[str]:
        """Lists storage keys with optional prefix.

        Args:
            prefix: Optional prefix filter

        Yields:
            Storage keys matching the prefix

        Raises:
            S3StorageError: If listing fails
        """
        full_prefix = self._full_key(prefix)
        try:
            client = await self._get_client()

            paginator = client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=self.config.bucket_name, Prefix=full_prefix)

            async for page in page_iterator:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        yield obj["Key"]

        except (ClientError, NoCredentialsError, BotoCoreError) as e:
            self._handle_s3_error(e, "list_keys", full_prefix)

    async def get_metadata(self, key: str) -> dict[str, Any]:
        """Gets object metadata from S3.

        Args:
            key: Storage key/path

        Returns:
            Object metadata including S3-specific information

        Raises:
            StorageNotFoundError: If object doesn't exist
            S3StorageError: If operation fails
        """
        full_key = self._full_key(key)
        try:
            client = await self._get_client()

            response = await client.head_object(Bucket=self.config.bucket_name, Key=full_key)

            # Extract relevant metadata
            metadata = {
                "content_length": response.get("ContentLength", 0),
                "content_type": response.get("ContentType", ""),
                "last_modified": response.get("LastModified"),
                "etag": response.get("ETag", "").strip('"'),
                "storage_class": response.get("StorageClass", "STANDARD"),
                "server_side_encryption": response.get("ServerSideEncryption"),
                "user_metadata": response.get("Metadata", {}),
            }

            # Add version ID if versioning is enabled
            if "VersionId" in response:
                metadata["version_id"] = response["VersionId"]

            return metadata

        except (ClientError, NoCredentialsError, BotoCoreError) as e:
            self._handle_s3_error(e, "get_metadata", full_key)

    async def copy(self, source_key: str, destination_key: str, metadata: dict[str, Any] | None = None) -> str:
        """Copies an object within S3.

        Args:
            source_key: Source storage key
            destination_key: Destination storage key
            metadata: Optional new metadata for the copy

        Returns:
            S3 URL for the copied object

        Raises:
            StorageNotFoundError: If source object doesn't exist
            S3StorageError: If copy operation fails
        """
        full_source = self._full_key(source_key)
        full_dest = self._full_key(destination_key)
        try:
            client = await self._get_client()

            copy_source = {"Bucket": self.config.bucket_name, "Key": full_source}

            copy_params = {
                "CopySource": copy_source,
                "Bucket": self.config.bucket_name,
                "Key": full_dest,
            }

            # Add metadata if provided
            if metadata:
                s3_metadata = self._prepare_metadata(metadata)
                copy_params["Metadata"] = s3_metadata
                copy_params["MetadataDirective"] = "REPLACE"

            await client.copy_object(**copy_params)

            logger.info("Successfully copied object from %s to %s", full_source, full_dest)

            # Return destination URL
            if self.config.endpoint_url:
                base_url = self.config.endpoint_url
                if not base_url.endswith("/"):
                    base_url += "/"
                return urljoin(base_url, f"{self.config.bucket_name}/{full_dest}")
            else:
                return f"https://s3.{self.config.region_name}.amazonaws.com/{self.config.bucket_name}/{full_dest}"

        except (ClientError, NoCredentialsError, BotoCoreError) as e:
            self._handle_s3_error(e, "copy", f"{full_source} -> {full_dest}")

    async def generate_presigned_url(self, key: str, operation: str = "get_object", expiration: int = 3600) -> str:
        """Generates a presigned URL for S3 operations.

        Args:
            key: Storage key/path
            operation: S3 operation (get_object, put_object, etc.)
            expiration: URL expiration time in seconds

        Returns:
            Presigned URL

        Raises:
            S3StorageError: If URL generation fails
        """
        full_key = self._full_key(key)
        try:
            client = await self._get_client()

            url = await client.generate_presigned_url(
                operation, Params={"Bucket": self.config.bucket_name, "Key": full_key}, ExpiresIn=expiration
            )

            return url

        except (ClientError, NoCredentialsError, BotoCoreError) as e:
            self._handle_s3_error(e, "generate_presigned_url", full_key)

    async def upload_file(
        self,
        file_path: Path,
        key: str,
        content_type: str | None = None,
        content_disposition: str | None = None,
    ) -> str:
        """Uploads a file from disk to S3.

        Args:
            file_path: Local file path to upload
            key: S3 key/path for the uploaded object
            content_type: MIME content type (auto-detected from extension if not provided)
            content_disposition: Content-Disposition header value

        Returns:
            S3 URI (s3://bucket/key) for the uploaded object

        Raises:
            S3StorageError: If upload fails
        """
        full_key = self._full_key(key)
        try:
            client = await self._get_client()

            extra_args: dict[str, str] = {}
            if content_type:
                extra_args["ContentType"] = content_type
            else:
                guessed, _ = mimetypes.guess_type(str(file_path))
                if guessed:
                    extra_args["ContentType"] = guessed
            if content_disposition:
                extra_args["ContentDisposition"] = content_disposition

            # aioboto3's upload_fileobj requires a sync file-like (it calls f.read() directly), so we cannot use
            # aiofiles. Offload the blocking open() call to a thread so the event loop stays free.
            f = await asyncio.to_thread(open, file_path, "rb")
            try:
                await client.upload_fileobj(f, self.config.bucket_name, full_key, ExtraArgs=extra_args or None)
            finally:
                await asyncio.to_thread(f.close)

            s3_url = f"s3://{self.config.bucket_name}/{full_key}"
            logger.info("Successfully uploaded file to %s", s3_url)
            return s3_url

        except (ClientError, NoCredentialsError, BotoCoreError, OSError) as e:
            self._handle_s3_error(e, "upload_file", full_key)

    async def store_file(
        self,
        file_path: Path,
        key: str,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StorageUrls:
        """Uploads a file from disk to S3 and returns both raw and presigned URLs.

        Combines upload_file() and generate_presigned_url() into a single operation that satisfies the StorageProvider
        protocol.

        Args:
            file_path: Local path to the file to upload
            key: Storage key/path (key_prefix is applied automatically)
            content_type: MIME content type (auto-detected if not provided)
            metadata: Optional metadata to attach (not yet used for file uploads)

        Returns:
            StorageUrls with raw S3 URI and presigned GET URL
        """
        raw_url = await self.upload_file(file_path, key, content_type=content_type)
        presigned_url = await self.generate_presigned_url(key, expiration=self.DEFAULT_PRESIGNED_URL_EXPIRATION)
        return StorageUrls(raw=raw_url, presigned=presigned_url)

    async def download_file(self, key: str, destination: Path) -> Path:
        """Downloads an S3 object to a local file.

        Args:
            key: S3 key/path of the object to download
            destination: Local file path to write to

        Returns:
            Path to the downloaded file

        Raises:
            S3StorageError: If download fails
        """
        full_key = self._full_key(key)
        try:
            client = await self._get_client()

            destination.parent.mkdir(parents=True, exist_ok=True)
            # aioboto3's download_fileobj requires a sync file-like (it calls f.write() directly), so we cannot use
            # aiofiles. Offload the blocking open() call to a thread so the event loop stays free.
            f = await asyncio.to_thread(open, destination, "wb")
            try:
                await client.download_fileobj(self.config.bucket_name, full_key, f)
            finally:
                await asyncio.to_thread(f.close)

            logger.info("Successfully downloaded %s to %s", full_key, destination)
            return destination

        except (ClientError, NoCredentialsError, BotoCoreError, OSError) as e:
            self._handle_s3_error(e, "download_file", full_key)

    async def download_from_url(self, url: str, destination: Path) -> Path:
        """Downloads a file from a URL (presigned or raw) to a local path.

        ``s3://`` URIs are converted to a presigned HTTPS URL before downloading. ``https://`` URLs are downloaded
        directly via HTTP GET, making this compatible with presigned URLs from any cloud provider.

        Large objects are fetched with **concurrent HTTP Range requests** (``multipart_chunksize`` per part, up to
        ``max_concurrency`` in flight) whenever the server advertises range support and the object is at least
        ``multipart_threshold`` bytes. This avoids the single-stream throughput ceiling that previously let
        sub-gigabyte ``rds_hq.zip`` inputs exceed the download deadline. If range support can't be confirmed (or the
        ranged transfer errors out) the download transparently falls back to a single streaming GET.

        Timeouts use an **inactivity (``sock_read``) budget rather than aiohttp's default 300s *total* cap**, so a
        slow-but-progressing transfer of a large object is not killed mid-flight; only a stalled connection fails. Both
        the read and connect budgets come from :class:`S3Config` (``download_sock_read_timeout`` /
        ``download_sock_connect_timeout``).

        Transient transfer failures (payload/connection errors, socket timeouts, 5xx) are retried with exponential
        backoff per :func:`_is_retryable_download_error`, bounded by ``download_max_attempts``. In the ranged path each
        part retries independently. Non-transient failures (4xx, a refused redirect) fail on the first attempt.

        Local URLs (``file://`` URIs and plain absolute / relative filesystem paths) bypass the network entirely and are
        symlinked into ``destination`` via :func:`services.framework.storage_providers.utils.local.symlink_local_url`.
        That lets a single S3-backed input provider mix cloud and local-disk URLs in the same request without flipping
        the service's ``*_INPUT_STORAGE_TYPE`` setting.

        Plain ``http://`` URLs are rejected: :func:`services.framework.url.security.check_url_security` raises
        :class:`InsecureUrlError` for any unencrypted HTTP scheme. This aligns with
        ``checks/utils/multistorage.py:is_remote_path``, which intentionally recognizes only ``https://``.

        Args:
            url: ``s3://bucket/key`` URI, an ``https://`` URL (e.g. presigned), a ``file://`` URI, or a plain filesystem
                path
            destination: Local file path to write to

        Returns:
            Path to the downloaded file

        Raises:
            InsecureUrlError: If ``url`` uses an unencrypted ``http://`` scheme
            S3StorageError: If download fails
        """
        if is_local_url(url):
            return symlink_local_url(url, destination)

        download_url = url
        s3_key = None

        if url.startswith("s3://"):
            s3_key = parse_s3_url(url).s3_key
            download_url = await self.generate_presigned_url_from_s3_url(url, expiration=1800)
            logger.info("Generated presigned URL for download")

        logger.info("Downloading file to %s", destination)

        # Re-validate DNS right before fetching to mitigate TOCTOU / DNS-rebinding. Allowlisted hosts (e.g. an internal
        # cluster like pbss.s8k.io) skip the private-IP check; TLS cert verification on the fetch remains the backstop.
        check_url_security(download_url, trusted_hosts=self.config.trusted_internal_hosts)

        # total=None disables aiohttp's default 300s wall-clock cap; sock_read bounds inactivity instead, so a large
        # object that keeps making progress is not killed just for taking a while.
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=self.config.download_sock_connect_timeout,
            sock_read=self.config.download_sock_read_timeout,
        )

        try:
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            # read_bufsize is the StreamReader's low-water mark (high-water is twice it), so matching it to the chunk
            # size the readers below request is what actually keeps the parser unpaused — iter_chunked alone only
            # drains the buffer faster, it never widens it past aiohttp's 64 KiB default.
            async with aiohttp.ClientSession(timeout=timeout, read_bufsize=self.config.download_chunk_size) as session:
                destination.parent.mkdir(parents=True, exist_ok=True)
                total_bytes = await self._probe_download_size(session, download_url, ssl_ctx)

                if total_bytes is not None and total_bytes >= self.config.multipart_threshold:
                    try:
                        await self._download_ranged(session, download_url, destination, total_bytes, ssl_ctx)
                    except _RANGED_DOWNLOAD_FALLBACK_ERRORS as ranged_error:
                        # Any anomaly in the parallel path (a part not honoring Range, a write error, etc.) degrades to
                        # a single stream, which truncates and rewrites destination from scratch.
                        logger.warning("Ranged download failed (%s); falling back to single-stream GET", ranged_error)
                        await self._download_stream(session, download_url, destination, ssl_ctx)
                else:
                    await self._download_stream(session, download_url, destination, ssl_ctx)
        except Exception as e:
            if destination.exists():
                destination.unlink()
            raise S3StorageError(
                f"Failed to download from '{url}' to '{destination}': {e}",
                operation="download_from_url",
                key=s3_key,
                details=str(e),
            ) from e

        logger.info("Downloaded file successfully to %s", destination)
        return destination

    async def _probe_download_size(
        self, session: aiohttp.ClientSession, url: str, ssl_ctx: ssl.SSLContext
    ) -> int | None:
        """Probes an object's size and range support with a single ``bytes=0-0`` GET.

        Returns the total object size in bytes when the server honors the range request (HTTP 206 with a parseable
        ``Content-Range`` total), or ``None`` when range support can't be confirmed — in which case the caller uses a
        single streaming GET. Never raises: any probe failure simply downgrades to single-stream.
        """
        try:
            # Don't follow redirects: a 3xx could send the fetch to a target check_url_security never saw. A
            # redirect here simply isn't a 206, so the probe downgrades to the single-stream path (which rejects it).
            async with session.get(url, headers={"Range": "bytes=0-0"}, ssl=ssl_ctx, allow_redirects=False) as response:
                if response.status != 206:
                    return None
                content_range = response.headers.get("Content-Range")
                if not content_range or "/" not in content_range:
                    return None
                total_str = content_range.rsplit("/", 1)[-1].strip()
                return int(total_str) if total_str.isdigit() else None
        except _RANGED_DOWNLOAD_FALLBACK_ERRORS as exc:
            logger.debug("Size probe failed (%s); using single-stream download", exc)
            return None

    async def _transfer_with_retry(self, description: str, transfer: Callable[[], Awaitable[None]]) -> None:
        """Runs ``transfer`` with exponential-backoff retries on transient HTTP/network failures.

        Both call sites rewrite their destination region from scratch on every attempt (:meth:`_download_stream`
        reopens with ``"wb"``; a ranged part seeks back to its own offset), so a retry can never leave a partial or
        interleaved result behind. Non-transient failures — a refused redirect, a range request the server declined,
        a 4xx — propagate on the first attempt so the caller's fallback or error path runs without delay.

        Args:
            description: Human-readable label for the transfer, used in retry logs (e.g. ``"bytes=0-1048575"``)
            transfer: Zero-argument coroutine factory performing one complete attempt

        Raises:
            Exception: The last failure, once the attempt budget is exhausted or the error is not retryable
        """
        attempts = self.config.download_max_attempts
        for attempt in range(1, attempts + 1):
            try:
                await transfer()
                return
            except Exception as exc:
                if attempt == attempts or not _is_retryable_download_error(exc):
                    raise
                delay = self.config.download_retry_backoff_base * (
                    self.config.download_retry_backoff_factor ** (attempt - 1)
                )
                logger.warning(
                    "Download of %s failed (attempt %d/%d): %s — retrying in %.1fs",
                    description,
                    attempt,
                    attempts,
                    _describe_download_error(exc),
                    delay,
                )
                await asyncio.sleep(delay)

    async def _download_ranged(
        self,
        session: aiohttp.ClientSession,
        url: str,
        destination: Path,
        total_bytes: int,
        ssl_ctx: ssl.SSLContext,
    ) -> None:
        """Downloads ``url`` to ``destination`` via concurrent HTTP Range requests.

        The object is split into ``multipart_chunksize`` byte parts fetched with up to ``max_concurrency`` in flight.
        The destination is preallocated to ``total_bytes`` and each part seeks to its own offset, so the concurrent
        writers never overlap. A part that hits a transient network failure is retried on its own (releasing its
        concurrency slot while it backs off); a part that fails for a non-transient reason — the server declining the
        range request, or a short read — raises, letting the caller fall back to a single streaming GET.
        """
        chunk_size = self.config.multipart_chunksize
        ranges: list[tuple[int, int]] = []
        start = 0
        while start < total_bytes:
            end = min(start + chunk_size - 1, total_bytes - 1)
            ranges.append((start, end))
            start = end + 1

        await asyncio.to_thread(self._preallocate_file, destination, total_bytes)
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def fetch_part(part_start: int, part_end: int) -> None:
            # allow_redirects=False: a redirect would bypass the SSRF check; a 3xx then fails the 206 check below
            # and falls back to the single-stream path, which rejects redirects outright.
            async with (
                semaphore,
                session.get(
                    url, headers={"Range": f"bytes={part_start}-{part_end}"}, ssl=ssl_ctx, allow_redirects=False
                ) as response,
            ):
                if response.status != 206:
                    raise S3StorageError(
                        f"Range request not honored (status {response.status})",
                        operation="download_from_url",
                    )
                file_handle = await asyncio.to_thread(open, destination, "r+b")
                try:
                    await asyncio.to_thread(file_handle.seek, part_start)
                    bytes_written = 0
                    async for data in response.content.iter_chunked(self.config.download_chunk_size):
                        await asyncio.to_thread(file_handle.write, data)
                        bytes_written += len(data)
                    # A short read would silently leave the preallocated zero-fill in place, corrupting the file. Detect
                    # the mismatch (still inside try, so finally closes the handle) and raise so the caller falls back.
                    expected_size = part_end - part_start + 1
                    if bytes_written != expected_size:
                        raise S3StorageError(
                            f"Range request returned fewer bytes than requested ({bytes_written}/{expected_size})",
                            operation="download_from_url",
                        )
                finally:
                    await asyncio.to_thread(file_handle.close)

        async def fetch_part_with_retry(part_start: int, part_end: int) -> None:
            # Retrying the individual part keeps one flaky connection from demoting the whole object to the much
            # slower single-stream fallback. Each attempt seeks back to part_start, so a partial write is overwritten.
            await self._transfer_with_retry(f"bytes={part_start}-{part_end}", lambda: fetch_part(part_start, part_end))

        await asyncio.gather(*(fetch_part_with_retry(s, e) for s, e in ranges))

    @staticmethod
    def _preallocate_file(destination: Path, total_bytes: int) -> None:
        """Creates (or truncates) ``destination`` to ``total_bytes`` so ranged writers can seek to their offsets."""
        with open(destination, "wb") as file_handle:
            file_handle.truncate(total_bytes)

    async def _download_stream(
        self, session: aiohttp.ClientSession, url: str, destination: Path, ssl_ctx: ssl.SSLContext
    ) -> None:
        """Downloads ``url`` to ``destination`` with a single streaming GET (the fallback / small-object path).

        Each attempt reopens ``destination`` with ``"wb"``, truncating whatever a previous failed attempt wrote, so
        retrying is safe.
        """

        async def attempt() -> None:
            # SSRF hardening: aiohttp follows redirects by default, so a validated (or allowlisted) URL could
            # 3xx-redirect to a private/internal target that check_url_security never saw. Block redirects and reject
            # any 3xx explicitly (raise_for_status only covers 4xx/5xx) so the validated host is the one we fetch from.
            async with session.get(url, ssl=ssl_ctx, allow_redirects=False) as response:
                if 300 <= response.status < 400:
                    raise S3StorageError(
                        f"Refusing to follow a download redirect (HTTP {response.status})",
                        operation="download_from_url",
                    )
                response.raise_for_status()
                async with aiofiles.open(destination, "wb") as file:
                    async for chunk in response.content.iter_chunked(self.config.download_chunk_size):
                        await file.write(chunk)

        await self._transfer_with_retry("full object", attempt)

    async def generate_presigned_url_from_s3_url(
        self,
        s3_url: str,
        operation: str = "get_object",
        expiration: int = DEFAULT_PRESIGNED_URL_EXPIRATION,
    ) -> str:
        """Generates a presigned URL from a raw S3 URL.

        Parses the S3 URL to extract the bucket and key and calls the boto3 client directly. The bucket from the URL is
        used (not ``self.config.bucket_name``), so this method works for arbitrary S3 URLs that may point at a different
        bucket than the one this provider is configured for. ``_full_key`` is bypassed so the configured ``key_prefix``
        is not applied a second time.

        Args:
            s3_url: Raw S3 URL (virtual-hosted, path-style, or s3://)
            operation: S3 operation (get_object, put_object, etc.)
            expiration: URL expiration time in seconds

        Returns:
            Presigned URL string

        Raises:
            ValueError: If URL cannot be parsed
            S3StorageError: If presigned URL generation fails
        """
        components = parse_s3_url(s3_url)
        absolute_key = components.s3_key
        try:
            client = await self._get_client()
            return await client.generate_presigned_url(
                operation,
                Params={"Bucket": components.bucket_name, "Key": absolute_key},
                ExpiresIn=expiration,
            )
        except (ClientError, NoCredentialsError, BotoCoreError) as e:
            self._handle_s3_error(e, "generate_presigned_url_from_s3_url", absolute_key)

    async def close(self) -> None:
        """Closes the S3 client and releases resources."""
        try:
            if self._client_ctx:
                await self._client_ctx.__aexit__(None, None, None)
        finally:
            self._client = None
            self._client_ctx = None
            self._session = None

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Async context manager exit."""
        await self.close()
