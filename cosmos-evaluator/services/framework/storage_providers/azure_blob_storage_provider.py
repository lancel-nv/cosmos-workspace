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

"""Azure Blob Storage provider implementation."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
import json
import logging
import mimetypes
from pathlib import Path
from types import TracebackType
from typing import Any, NoReturn, Self
from urllib.parse import urlparse

from azure.core.exceptions import (
    AzureError,
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
)
from azure.storage.blob import BlobSasPermissions, ContentSettings, generate_blob_sas
from azure.storage.blob.aio import BlobClient, BlobServiceClient

from services.framework.protocols.storage_provider import StorageUrls
from services.framework.storage_providers.config import AzureBlobConfig
from services.framework.storage_providers.exceptions import (
    AzureBlobStorageError,
    StorageAuthenticationError,
    StorageConnectionError,
    StorageNotFoundError,
    StoragePermissionError,
    StorageQuotaExceededError,
)
from services.framework.storage_providers.utils.azure import (
    AZURE_URI_SCHEME,
    build_account_url,
    build_azure_uri,
    build_blob_url,
    is_azure_blob_url,
    matches_endpoint,
    parse_azure_url,
)
from services.framework.storage_providers.utils.local import is_local_url, symlink_local_url
from services.framework.url.security import check_url_security

logger = logging.getLogger(__name__)

# Azure service error codes that mean the account or container is over its limit, rather than a request being
# malformed or unauthorized. Mapped to StorageQuotaExceededError so callers can treat them as capacity problems.
_QUOTA_ERROR_CODES = frozenset({"AccountIsOverQuota", "ShareSizeLimitReached", "QuotaExceeded"})

# The protocol's ``operation`` argument carries S3's verb names, since S3 was the first backend. Map them onto the SAS
# permissions Azure needs. ``create`` accompanies ``write`` because a SAS meant for uploading must also be able to
# bring a not-yet-existing blob into being.
_READ_OPERATION = "get_object"
_WRITE_OPERATION = "put_object"


class AzureBlobStorageProvider:
    """Azure Blob Storage provider implementation.

    Implements the StorageProvider protocol on the Azure SDK's async ``BlobServiceClient``. Supports account-key and
    SAS-token authentication, SAS-based presigned URLs, and the SDK's own chunked parallel transfers — which is why,
    unlike :class:`S3StorageProvider`, this provider carries no hand-rolled ranged-download engine.
    """

    DEFAULT_PRESIGNED_URL_EXPIRATION = 43200  # 12 hours

    def __init__(self, config: AzureBlobConfig) -> None:
        """Initializes the Azure Blob storage provider.

        Args:
            config: Azure configuration object
        """
        self.config = config
        self._service_client: BlobServiceClient | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> BlobServiceClient:
        """Gets or creates the shared ``BlobServiceClient``.

        Built lazily behind a lock so concurrent first-use from several requests creates exactly one client (and
        therefore one HTTP transport / connection pool) rather than racing to build several.
        """
        async with self._client_lock:
            if self._service_client is None:
                self._service_client = BlobServiceClient(
                    account_url=build_account_url(self.config.account_name, self.config.endpoint_url),
                    credential=self.config.get_credential(),
                )
        return self._service_client

    async def _get_blob_client(self, key: str) -> BlobClient:
        """Gets a blob client for ``key`` within the configured container.

        The returned client shares the service client's transport, so callers must *not* close it — doing so would
        tear down the pooled connections every other in-flight operation is using.
        """
        client = await self._get_client()
        return client.get_blob_client(container=self.config.container_name, blob=self._full_key(key))

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

        Identical semantics to :meth:`S3StorageProvider.make_key_prefix`: blob names are opaque strings with no notion
        of an absolute path, so leading and trailing slashes are stripped from every part, empty/None parts are
        dropped, the rest are joined with ``/``, and a trailing ``/`` is appended so callers can concatenate filenames.
        A leading slash would otherwise create an empty top-level path component that confuses tooling and breaks the
        URL round-trip through ``utils.azure.parse_azure_url``.
        """
        cleaned = [p.strip("/") for p in parts if p]
        cleaned = [p for p in cleaned if p]
        return ("/".join(cleaned) + "/") if cleaned else ""

    def _handle_azure_error(self, error: Exception, operation: str, key: str | None) -> NoReturn:
        """Handles and converts Azure errors to storage exceptions.

        Mirrors :meth:`S3StorageProvider._handle_s3_error` so callers can catch the same backend-neutral exceptions
        regardless of which provider raised them. ``ResourceNotFoundError`` and ``ClientAuthenticationError`` are
        checked before ``HttpResponseError`` because both subclass it.

        Args:
            error: Original exception
            operation: Operation being performed
            key: Storage key involved, if any

        Raises:
            Appropriate StorageError subclass
        """
        if isinstance(error, ResourceNotFoundError):
            raise StorageNotFoundError(
                f"Blob '{key}' not found in container '{self.config.container_name}'", key=key, operation=operation
            )
        if isinstance(error, ClientAuthenticationError):
            raise StorageAuthenticationError(
                f"Azure Storage credentials rejected during {operation}: {error!s}", key=key, operation=operation
            )
        if isinstance(error, HttpResponseError):
            # ``error_code`` is not declared on HttpResponseError: azure-storage-blob attaches it to the exception it
            # raises (from the ``x-ms-error-code`` response header) in its own error handler, so it has to be read
            # defensively. Azure Storage answers in XML, which azure-core's OData body parser cannot read, so
            # ``error.error.code`` is not a usable alternative here.
            error_code = getattr(error, "error_code", None) or ""
            if error.status_code == 403:
                raise StoragePermissionError(
                    f"Access denied for operation '{operation}' on key '{key}': {error.message}",
                    key=key,
                    operation=operation,
                )
            if error_code in _QUOTA_ERROR_CODES:
                raise StorageQuotaExceededError(
                    f"Storage quota exceeded: {error.message}", key=key, operation=operation
                )
            raise AzureBlobStorageError(
                f"Azure Storage error during {operation}: {error.message}",
                key=key,
                operation=operation,
                container=self.config.container_name,
                error_code=error_code,
            )
        if isinstance(error, ServiceRequestError):
            raise StorageConnectionError(
                f"Connection error during {operation}: {error!s}", key=key, operation=operation
            )
        raise AzureBlobStorageError(
            f"Unexpected error during {operation}: {error!s}",
            key=key,
            operation=operation,
            container=self.config.container_name,
        )

    def _prepare_data_for_storage(self, data: Any) -> bytes:
        """Prepares data for storage by converting to bytes.

        Args:
            data: Data to prepare

        Returns:
            Data as bytes

        Raises:
            AzureBlobStorageError: If data cannot be serialized
        """
        if isinstance(data, bytes):
            return data
        if isinstance(data, str):
            return data.encode("utf-8")
        if isinstance(data, (dict, list)):
            try:
                return json.dumps(data, ensure_ascii=False).encode("utf-8")
            except (TypeError, ValueError) as e:
                raise AzureBlobStorageError(f"Failed to serialize data to JSON: {e}") from e
        try:
            return str(data).encode("utf-8")
        except Exception as e:
            raise AzureBlobStorageError(f"Failed to convert data to bytes: {e}") from e

    def _prepare_metadata(self, metadata: dict[str, Any] | None) -> dict[str, str]:
        """Prepares metadata for Azure Blob storage.

        Azure requires metadata names to be valid C# identifiers — letters, digits and underscores only — which is the
        mirror image of S3's restriction, where hyphens are the safe separator. Hyphens are therefore converted to
        underscores here, in the opposite direction to
        :meth:`S3StorageProvider._prepare_metadata`.

        Args:
            metadata: Original metadata

        Returns:
            Azure-compatible metadata (string values only)
        """
        if not metadata:
            return {}

        azure_metadata = {}
        for key, value in metadata.items():
            clean_key = key.replace("-", "_")
            if isinstance(value, str):
                azure_metadata[clean_key] = value
            else:
                try:
                    azure_metadata[clean_key] = json.dumps(value)
                except (TypeError, ValueError):
                    azure_metadata[clean_key] = str(value)

        return azure_metadata

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
        if isinstance(data, (dict, list)):
            return "application/json; charset=utf-8"
        content_type, _ = mimetypes.guess_type(key)
        return content_type or self.config.default_content_type

    async def store(self, data: Any, key: str, metadata: dict[str, Any] | None = None) -> str:
        """Stores data in Azure Blob Storage.

        Args:
            data: Data to store (bytes, string, or serializable object)
            key: Storage key/path
            metadata: Optional metadata to attach

        Returns:
            Canonical ``azure://<container>/<blob>`` URI for the stored blob

        Raises:
            AzureBlobStorageError: If the storage operation fails
        """
        full_key = self._full_key(key)
        try:
            blob_client = await self._get_blob_client(key)

            data_bytes = self._prepare_data_for_storage(data)
            content_type = self._determine_content_type(data, key)

            await blob_client.upload_blob(
                data_bytes,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
                metadata=self._prepare_metadata(metadata),
                max_concurrency=self.config.max_concurrency,
            )

            logger.info("Successfully stored blob: %s (%s bytes)", full_key, len(data_bytes))
            return build_azure_uri(self.config.container_name, full_key)

        except AzureError as e:
            self._handle_azure_error(e, "store", full_key)

    async def retrieve(self, key: str) -> Any:
        """Retrieves data from Azure Blob Storage.

        Args:
            key: Storage key/path

        Returns:
            Retrieved data (attempts to deserialize JSON, otherwise returns bytes)

        Raises:
            StorageNotFoundError: If the blob doesn't exist
            AzureBlobStorageError: If retrieval fails
        """
        full_key = self._full_key(key)
        try:
            blob_client = await self._get_blob_client(key)

            downloader = await blob_client.download_blob(max_concurrency=self.config.max_concurrency)
            data_bytes = await downloader.readall()

            content_settings = downloader.properties.content_settings
            content_type = (content_settings.content_type if content_settings else "") or ""

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

        except AzureError as e:
            self._handle_azure_error(e, "retrieve", full_key)

    async def delete(self, key: str) -> bool:
        """Deletes data from Azure Blob Storage.

        Args:
            key: Storage key/path

        Returns:
            True if deletion was successful

        Raises:
            AzureBlobStorageError: If deletion fails
        """
        full_key = self._full_key(key)
        try:
            blob_client = await self._get_blob_client(key)
            await blob_client.delete_blob()

            logger.info("Successfully deleted blob: %s", full_key)
            return True

        except AzureError as e:
            self._handle_azure_error(e, "delete", full_key)

    async def exists(self, key: str) -> bool:
        """Checks if data exists in Azure Blob Storage.

        Args:
            key: Storage key/path

        Returns:
            True if the blob exists
        """
        full_key = self._full_key(key)
        try:
            blob_client = await self._get_blob_client(key)
            return bool(await blob_client.exists())

        except AzureError as e:
            self._handle_azure_error(e, "exists", full_key)

    async def list_keys(self, prefix: str = "") -> AsyncIterator[str]:
        """Lists storage keys with optional prefix.

        Args:
            prefix: Optional prefix filter

        Yields:
            Storage keys matching the prefix

        Raises:
            AzureBlobStorageError: If listing fails
        """
        full_prefix = self._full_key(prefix)
        try:
            client = await self._get_client()
            container_client = client.get_container_client(self.config.container_name)

            async for blob in container_client.list_blobs(name_starts_with=full_prefix):
                yield blob.name

        except AzureError as e:
            self._handle_azure_error(e, "list_keys", full_prefix)

    async def upload_file(
        self,
        file_path: Path,
        key: str,
        content_type: str | None = None,
        content_disposition: str | None = None,
    ) -> str:
        """Uploads a file from disk to Azure Blob Storage.

        Args:
            file_path: Local file path to upload
            key: Blob key/path for the uploaded blob
            content_type: MIME content type (auto-detected from extension if not provided)
            content_disposition: Content-Disposition header value

        Returns:
            Canonical ``azure://<container>/<blob>`` URI for the uploaded blob

        Raises:
            AzureBlobStorageError: If the upload fails
        """
        full_key = self._full_key(key)
        try:
            blob_client = await self._get_blob_client(key)

            resolved_type = content_type
            if not resolved_type:
                guessed, _ = mimetypes.guess_type(str(file_path))
                resolved_type = guessed

            # The SDK's upload_blob takes a sync file-like object and reads from it on its own worker, so aiofiles is
            # not usable here. Offload the blocking open() call to a thread so the event loop stays free.
            file_handle = await asyncio.to_thread(open, file_path, "rb")
            try:
                await blob_client.upload_blob(
                    file_handle,
                    overwrite=True,
                    content_settings=ContentSettings(
                        content_type=resolved_type, content_disposition=content_disposition
                    ),
                    max_concurrency=self.config.max_concurrency,
                )
            finally:
                await asyncio.to_thread(file_handle.close)

            azure_uri = build_azure_uri(self.config.container_name, full_key)
            logger.info("Successfully uploaded file to %s", azure_uri)
            return azure_uri

        except (AzureError, OSError) as e:
            self._handle_azure_error(e, "upload_file", full_key)

    async def store_file(
        self,
        file_path: Path,
        key: str,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StorageUrls:
        """Uploads a file from disk to Azure Blob Storage and returns both raw and presigned URLs.

        Combines :meth:`upload_file` and :meth:`generate_presigned_url` into a single operation that satisfies the
        StorageProvider protocol.

        Args:
            file_path: Local path to the file to upload
            key: Storage key/path (key_prefix is applied automatically)
            content_type: MIME content type (auto-detected if not provided)
            metadata: Optional metadata to attach (not yet used for file uploads)

        Returns:
            StorageUrls with the raw ``azure://`` URI and a SAS-signed HTTPS URL
        """
        raw_url = await self.upload_file(file_path, key, content_type=content_type)
        presigned_url = await self.generate_presigned_url(key, expiration=self.DEFAULT_PRESIGNED_URL_EXPIRATION)
        return StorageUrls(raw=raw_url, presigned=presigned_url)

    async def generate_presigned_url(self, key: str, operation: str = _READ_OPERATION, expiration: int = 3600) -> str:
        """Generates a SAS-signed HTTPS URL for a blob.

        With an account key configured, a fresh blob SAS is minted with the requested lifetime. With only a
        pre-minted ``sas_token``, that token is appended instead: it is the only signature available, so the URL's
        real expiry is whatever the token carries and ``expiration`` cannot be honored.

        Args:
            key: Storage key/path
            operation: Storage operation. The protocol uses S3's verb names; ``put_object`` maps to a
                create+write SAS and anything else to a read SAS.
            expiration: URL expiration time in seconds (account-key mode only)

        Returns:
            HTTPS URL carrying a SAS query string

        Raises:
            AzureBlobStorageError: If SAS generation fails
        """
        full_key = self._full_key(key)

        if not self.config.account_key:
            logger.info(
                "Reusing the configured SAS token for %s; its own expiry applies instead of the requested %ss",
                full_key,
                expiration,
            )
            return build_blob_url(
                self.config.container_name,
                full_key,
                self.config.account_name,
                self.config.endpoint_url,
                self.config.get_credential(),
            )

        permission = (
            BlobSasPermissions(read=True, write=True, create=True)
            if operation == _WRITE_OPERATION
            else BlobSasPermissions(read=True)
        )
        try:
            sas_token = generate_blob_sas(
                account_name=self.config.account_name,
                container_name=self.config.container_name,
                blob_name=full_key,
                account_key=self.config.account_key.get_secret_value(),
                permission=permission,
                expiry=datetime.now(UTC) + timedelta(seconds=expiration),
            )
        except (AzureError, ValueError) as e:
            self._handle_azure_error(e, "generate_presigned_url", full_key)

        return build_blob_url(
            self.config.container_name,
            full_key,
            self.config.account_name,
            self.config.endpoint_url,
            sas_token,
        )

    async def download_file(self, key: str, destination: Path) -> Path:
        """Downloads a blob to a local file.

        Args:
            key: Blob key/path of the blob to download
            destination: Local file path to write to

        Returns:
            Path to the downloaded file

        Raises:
            AzureBlobStorageError: If the download fails
        """
        full_key = self._full_key(key)
        try:
            blob_client = await self._get_blob_client(key)
            await self._download_to_path(blob_client, destination)

            logger.info("Successfully downloaded %s to %s", full_key, destination)
            return destination

        except (AzureError, OSError) as e:
            # _download_to_path opens the destination before streaming into it, so a mid-transfer failure leaves a
            # truncated file behind. Remove it, as download_from_url already does, so a retry or a later reader
            # cannot mistake the partial artifact for a complete download.
            destination.unlink(missing_ok=True)
            self._handle_azure_error(e, "download_file", full_key)

    async def download_from_url(self, url: str, destination: Path) -> Path:
        """Downloads a blob addressed by URL to a local path.

        Accepts the three forms :func:`utils.azure.parse_azure_url` understands — ``azure://<container>/<blob>``, a
        ``blob.core.windows.net`` HTTPS URL (with or without a SAS), and the path-style form under a configured
        ``endpoint_url``. A SAS-bearing URL is fetched with the token it carries; otherwise this provider's own
        credential is used, which is what makes bare ``azure://`` URIs recorded by a previous run re-readable.

        Local URLs (``file://`` URIs and plain filesystem paths) bypass the network entirely and are symlinked into
        ``destination`` via :func:`services.framework.storage_providers.utils.local.symlink_local_url` — the same fast
        path :meth:`S3StorageProvider.download_from_url` offers, so a single provider can mix cloud and local-disk
        URLs within one request.

        Any other URL — including an arbitrary or presigned ``https://`` URL on a non-Azure host — is rejected rather
        than handed to the Azure SDK. Deployments that need to read such URLs should point
        ``COSMOS_EVALUATOR_INPUT_STORAGE_TYPE`` at ``s3``, whose provider fetches any HTTPS URL over plain HTTP and
        therefore already handles presigned URLs from any cloud.

        Args:
            url: ``azure://`` URI, Azure blob HTTPS URL, ``file://`` URI, or plain filesystem path
            destination: Local file path to write to

        Returns:
            Path to the downloaded file

        Raises:
            AzureBlobStorageError: If the URL is not an Azure blob URL, or the download fails
        """
        if is_local_url(url):
            return symlink_local_url(url, destination)

        if not is_azure_blob_url(url, self.config.endpoint_url):
            raise AzureBlobStorageError(
                f"Not an Azure Blob Storage URL: '{url}'. Set COSMOS_EVALUATOR_INPUT_STORAGE_TYPE=s3 to download "
                "arbitrary HTTPS or presigned URLs.",
                operation="download_from_url",
                container=self.config.container_name,
            )

        components = parse_azure_url(url, self.config.endpoint_url)
        if urlparse(url).scheme == AZURE_URI_SCHEME:
            # An azure:// URI names no account, so address it against the account this provider is configured for.
            blob_url = build_blob_url(
                components.container_name,
                components.blob_name,
                self.config.account_name,
                self.config.endpoint_url,
            )
        else:
            blob_url = url

        # Re-validate DNS right before fetching to mitigate TOCTOU / DNS-rebinding, and skip it for the endpoint the
        # operator configured. That exemption is what makes an Azurite emulator on ``http://127.0.0.1:10000`` usable
        # at all, since check_url_security rejects plain HTTP even for allowlisted hosts. It is safe because
        # is_azure_blob_url has already pinned the URL to that endpoint's origin, so a caller can vary only the path,
        # never the host. Public blob.core.windows.net URLs still go through the full check.
        if not matches_endpoint(blob_url, self.config.endpoint_url):
            check_url_security(blob_url, trusted_hosts=self.config.trusted_internal_hosts)

        logger.info("Downloading blob %s to %s", components.blob_name, destination)
        try:
            # from_blob_url builds a standalone client with its own transport, so unlike the container-scoped clients
            # above this one owns its connections and must be closed.
            async with BlobClient.from_blob_url(
                blob_url, credential=None if components.sas_token else self.config.get_credential()
            ) as blob_client:
                await self._download_to_path(blob_client, destination)
        except Exception as e:
            if destination.exists():
                destination.unlink()
            raise AzureBlobStorageError(
                f"Failed to download from '{url}' to '{destination}': {e}",
                operation="download_from_url",
                key=components.blob_name,
                container=components.container_name,
                details=str(e),
            ) from e

        logger.info("Downloaded blob successfully to %s", destination)
        return destination

    async def _download_to_path(self, blob_client: BlobClient, destination: Path) -> None:
        """Streams a blob into ``destination``, creating parent directories as needed.

        ``max_concurrency`` lets the SDK split the blob into chunks and fetch them in parallel, which is the Azure
        equivalent of the S3 provider's ranged-download path and the reason this provider needs none of that
        machinery itself. ``readinto`` writes through a plain sync file object, so the blocking ``open`` is offloaded
        to a thread to keep the event loop free.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        downloader = await blob_client.download_blob(max_concurrency=self.config.max_concurrency)
        file_handle = await asyncio.to_thread(open, destination, "wb")
        try:
            await downloader.readinto(file_handle)
        finally:
            await asyncio.to_thread(file_handle.close)

    async def close(self) -> None:
        """Closes the Azure Blob client and releases resources."""
        try:
            if self._service_client is not None:
                await self._service_client.close()
        finally:
            self._service_client = None

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
