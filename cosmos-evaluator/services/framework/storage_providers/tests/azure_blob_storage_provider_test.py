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

"""Tests for AzureBlobStorageProvider and AzureBlobConfig.

The Azure SDK client is mocked at the module boundary (``BlobServiceClient`` / ``BlobClient`` as imported by the
provider) rather than exercised against a live account or emulator, matching how ``s3_storage_provider_test.py``
mocks ``aioboto3``. The real ``azure.core.exceptions`` classes are used for the error-mapping tests so the mapping is
pinned against the SDK's actual hierarchy, not a stand-in.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from azure.core.exceptions import (
    ClientAuthenticationError,
    HttpResponseError,
    ResourceNotFoundError,
    ServiceRequestError,
)
from pydantic import ValidationError
import pytest

from services.framework.protocols.storage_provider import StorageProvider, StorageUrls
from services.framework.storage_providers.azure_blob_storage_provider import AzureBlobStorageProvider
from services.framework.storage_providers.config import AzureBlobConfig
from services.framework.storage_providers.exceptions import (
    AzureBlobStorageError,
    StorageAuthenticationError,
    StorageConnectionError,
    StorageNotFoundError,
    StoragePermissionError,
    StorageQuotaExceededError,
)

_PROVIDER_MODULE = "services.framework.storage_providers.azure_blob_storage_provider"
_ACCOUNT = "myacct"
_CONTAINER = "my-container"
_AZURITE_ENDPOINT = "http://127.0.0.1:10000"


class _StorageHttpResponseError(HttpResponseError):
    """An HttpResponseError carrying the ``error_code`` azure-storage-blob attaches before raising.

    The attribute is not declared on ``HttpResponseError``; the Storage SDK sets it from the ``x-ms-error-code``
    response header in its own error handler. Declaring it on a subclass reproduces the exception the provider
    actually sees, while keeping the test source free of dynamic attribute assignment.
    """

    def __init__(self, status_code: int, error_code: str | None, message: str) -> None:
        super().__init__(message=message)
        self.status_code = status_code
        self.error_code = error_code


def _http_response_error(status_code: int, error_code: str | None = None, message: str = "boom") -> HttpResponseError:
    """Builds an HttpResponseError with the status and service error code the provider branches on."""
    return _StorageHttpResponseError(status_code=status_code, error_code=error_code, message=message)


def _make_downloader(body: bytes, content_type: str | None = None) -> AsyncMock:
    """Builds a mock StorageStreamDownloader that yields ``body`` from readall/readinto."""
    downloader = AsyncMock()
    downloader.readall = AsyncMock(return_value=body)

    async def readinto(stream: Any) -> int:
        stream.write(body)
        return len(body)

    downloader.readinto = AsyncMock(side_effect=readinto)
    content_settings = MagicMock()
    content_settings.content_type = content_type
    downloader.properties = MagicMock(content_settings=content_settings)
    return downloader


def _make_blob_client(body: bytes = b"", content_type: str | None = None) -> AsyncMock:
    """Builds a mock BlobClient whose download_blob returns a downloader over ``body``."""
    blob_client = AsyncMock()
    blob_client.download_blob = AsyncMock(return_value=_make_downloader(body, content_type))
    return blob_client


def _make_from_blob_url(blob_client: AsyncMock) -> MagicMock:
    """Builds a mock ``BlobClient.from_blob_url`` returning ``blob_client`` as an async context manager.

    ``__aexit__`` must return None explicitly: a bare ``AsyncMock()`` resolves to a truthy Mock, which ``async with``
    reads as "exception handled" and would silently swallow any error raised inside the block.
    """
    return MagicMock(
        return_value=AsyncMock(__aenter__=AsyncMock(return_value=blob_client), __aexit__=AsyncMock(return_value=None))
    )


def _make_service_client(blob_client: AsyncMock, blob_names: list[str] | None = None) -> MagicMock:
    """Builds a mock BlobServiceClient handing out ``blob_client`` and, optionally, a listable container."""
    service = MagicMock()
    service.get_blob_client = MagicMock(return_value=blob_client)
    service.close = AsyncMock()

    container_client = MagicMock()

    def list_blobs(name_starts_with: str = "") -> AsyncIterator[Any]:
        async def iterator() -> AsyncIterator[Any]:
            for name in blob_names or []:
                if name.startswith(name_starts_with):
                    # ``name`` is special in the MagicMock constructor, so it has to be assigned afterwards.
                    blob = MagicMock()
                    blob.name = name
                    yield blob

        return iterator()

    container_client.list_blobs = MagicMock(side_effect=list_blobs)
    service.get_container_client = MagicMock(return_value=container_client)
    return service


class TestAzureBlobConfig:
    """Test suite for AzureBlobConfig."""

    def test_config_initialization_with_account_key(self) -> None:
        config = AzureBlobConfig(container_name=_CONTAINER, account_name=_ACCOUNT, account_key="secret")
        self.assert_common_defaults(config)
        assert config.account_key is not None
        assert config.account_key.get_secret_value() == "secret"
        assert config.sas_token is None

    def test_config_initialization_with_sas_token(self) -> None:
        config = AzureBlobConfig(container_name=_CONTAINER, account_name=_ACCOUNT, sas_token="sig=abc")
        self.assert_common_defaults(config)
        assert config.account_key is None
        assert config.sas_token is not None
        assert config.sas_token.get_secret_value() == "sig=abc"

    def assert_common_defaults(self, config: AzureBlobConfig) -> None:
        """Shared assertions on the defaults every construction path should produce."""
        assert config.container_name == _CONTAINER
        assert config.account_name == _ACCOUNT
        assert config.endpoint_url is None
        assert config.key_prefix == ""
        assert config.max_concurrency == 10
        assert config.default_content_type == "application/octet-stream"
        assert config.trusted_internal_hosts == frozenset()

    def test_config_requires_a_credential(self) -> None:
        with pytest.raises(ValidationError, match="account_key or sas_token"):
            AzureBlobConfig(container_name=_CONTAINER, account_name=_ACCOUNT)

    # The next two cases feed deliberately malformed payloads, so they go through ``model_validate`` rather than the
    # constructor: a missing required field or a stray extra one is exactly what the type checker would reject at the
    # call site, and validating a raw dict is how these configs are really built from settings anyway.
    def test_config_requires_container_and_account(self) -> None:
        with pytest.raises(ValidationError):
            AzureBlobConfig.model_validate({"account_name": _ACCOUNT, "account_key": "secret"})
        with pytest.raises(ValidationError):
            AzureBlobConfig.model_validate({"container_name": _CONTAINER, "account_key": "secret"})

    def test_config_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            AzureBlobConfig.model_validate(
                {
                    "container_name": _CONTAINER,
                    "account_name": _ACCOUNT,
                    "account_key": "secret",
                    "bucket_name": "wrong-backend",
                }
            )

    def test_config_rejects_non_positive_concurrency(self) -> None:
        with pytest.raises(ValidationError):
            AzureBlobConfig(container_name=_CONTAINER, account_name=_ACCOUNT, account_key="k", max_concurrency=0)

    def test_get_credential_prefers_account_key(self) -> None:
        config = AzureBlobConfig(
            container_name=_CONTAINER, account_name=_ACCOUNT, account_key="thekey", sas_token="sig=abc"
        )
        assert config.get_credential() == "thekey"

    def test_get_credential_falls_back_to_sas_token(self) -> None:
        config = AzureBlobConfig(container_name=_CONTAINER, account_name=_ACCOUNT, sas_token="sig=abc")
        assert config.get_credential() == "sig=abc"


class TestAzureBlobStorageProvider:
    """Test suite for AzureBlobStorageProvider."""

    @pytest.fixture
    def config(self) -> AzureBlobConfig:
        return AzureBlobConfig(container_name=_CONTAINER, account_name=_ACCOUNT, account_key="secret")

    @pytest.fixture
    def provider(self, config: AzureBlobConfig) -> AzureBlobStorageProvider:
        return AzureBlobStorageProvider(config=config)

    def test_storage_provider_implements_protocol(self, provider: AzureBlobStorageProvider) -> None:
        assert isinstance(provider, StorageProvider)

    def test_provider_initialization(self, provider: AzureBlobStorageProvider, config: AzureBlobConfig) -> None:
        assert provider.config == config
        assert provider._service_client is None

    @pytest.mark.asyncio
    async def test_get_client_creation_and_reuse(self, provider: AzureBlobStorageProvider) -> None:
        """The service client is built once and shared, so the transport / connection pool is not duplicated."""
        service = _make_service_client(_make_blob_client())
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)) as mock_ctor:
            first = await provider._get_client()
            second = await provider._get_client()

        assert first is service
        assert second is service
        assert mock_ctor.call_count == 1
        _, kwargs = mock_ctor.call_args
        assert kwargs["account_url"] == f"https://{_ACCOUNT}.blob.core.windows.net"
        assert kwargs["credential"] == "secret"

    @pytest.mark.asyncio
    async def test_get_client_uses_custom_endpoint(self) -> None:
        config = AzureBlobConfig(
            container_name=_CONTAINER, account_name=_ACCOUNT, account_key="k", endpoint_url=_AZURITE_ENDPOINT
        )
        provider = AzureBlobStorageProvider(config=config)
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock()) as mock_ctor:
            await provider._get_client()

        _, kwargs = mock_ctor.call_args
        assert kwargs["account_url"] == f"{_AZURITE_ENDPOINT}/{_ACCOUNT}"

    @pytest.mark.asyncio
    async def test_concurrent_client_access_builds_one_client(self, provider: AzureBlobStorageProvider) -> None:
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock()) as mock_ctor:
            clients = await asyncio.gather(*(provider._get_client() for _ in range(5)))

        assert mock_ctor.call_count == 1
        assert all(client is clients[0] for client in clients)

    def test_full_key_no_prefix(self, provider: AzureBlobStorageProvider) -> None:
        assert provider._full_key("a/b.json") == "a/b.json"

    def test_full_key_with_prefix(self) -> None:
        config = AzureBlobConfig(
            container_name=_CONTAINER, account_name=_ACCOUNT, account_key="k", key_prefix="runs/42/"
        )
        provider = AzureBlobStorageProvider(config=config)
        assert provider._full_key("a/b.json") == "runs/42/a/b.json"

    # make_key_prefix must behave identically to S3StorageProvider.make_key_prefix; these cases mirror that suite.
    @pytest.mark.parametrize(
        ("parts", "expected"),
        [
            ((), ""),
            ((None, None), ""),
            (("", ""), ""),
            (("/", "//"), ""),
            (("/tmp/x", "egomotion", "cam_front"), "tmp/x/egomotion/cam_front/"),
            (("a/", "/b/", "c"), "a/b/c/"),
            ((None, "objects", None, "cam"), "objects/cam/"),
        ],
    )
    def test_make_key_prefix(
        self, provider: AzureBlobStorageProvider, parts: tuple[str | None, ...], expected: str
    ) -> None:
        assert provider.make_key_prefix(*parts) == expected

    @pytest.mark.asyncio
    async def test_store_uploads_and_returns_azure_uri(self, provider: AzureBlobStorageProvider) -> None:
        blob_client = _make_blob_client()
        service = _make_service_client(blob_client)
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)):
            url = await provider.store({"a": 1}, "results.json", metadata={"clip-id": "abc", "frames": 7})

        assert url == f"azure://{_CONTAINER}/results.json"
        _, kwargs = blob_client.upload_blob.call_args
        assert blob_client.upload_blob.call_args[0][0] == b'{"a": 1}'
        assert kwargs["overwrite"] is True
        assert kwargs["content_settings"].content_type == "application/json; charset=utf-8"
        # Azure metadata names must be C# identifiers, so hyphens become underscores (the inverse of S3).
        assert kwargs["metadata"] == {"clip_id": "abc", "frames": "7"}

    @pytest.mark.asyncio
    async def test_store_applies_key_prefix(self) -> None:
        config = AzureBlobConfig(
            container_name=_CONTAINER, account_name=_ACCOUNT, account_key="k", key_prefix="runs/42/"
        )
        provider = AzureBlobStorageProvider(config=config)
        service = _make_service_client(_make_blob_client())
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)):
            url = await provider.store(b"raw", "out.bin")

        assert url == f"azure://{_CONTAINER}/runs/42/out.bin"
        assert service.get_blob_client.call_args.kwargs["blob"] == "runs/42/out.bin"

    @pytest.mark.asyncio
    async def test_store_raises_on_unserializable_data(self, provider: AzureBlobStorageProvider) -> None:
        service = _make_service_client(_make_blob_client())
        with (
            patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)),
            pytest.raises(AzureBlobStorageError, match="Failed to serialize"),
        ):
            await provider.store({"bad": {1, 2}}, "results.json")

    @pytest.mark.asyncio
    async def test_retrieve_parses_json_by_content_type(self, provider: AzureBlobStorageProvider) -> None:
        body = json.dumps({"hello": "world"}).encode()
        service = _make_service_client(_make_blob_client(body, "application/json"))
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)):
            result = await provider.retrieve("data.bin")

        assert result == {"hello": "world"}

    @pytest.mark.asyncio
    async def test_retrieve_parses_json_by_extension(self, provider: AzureBlobStorageProvider) -> None:
        service = _make_service_client(_make_blob_client(b'{"n": 1}', None))
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)):
            result = await provider.retrieve("data.json")

        assert result == {"n": 1}

    @pytest.mark.asyncio
    async def test_retrieve_decodes_text(self, provider: AzureBlobStorageProvider) -> None:
        service = _make_service_client(_make_blob_client(b"plain words", "text/plain; charset=utf-8"))
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)):
            result = await provider.retrieve("notes.txt")

        assert result == "plain words"

    @pytest.mark.asyncio
    async def test_retrieve_returns_bytes_for_binary(self, provider: AzureBlobStorageProvider) -> None:
        service = _make_service_client(_make_blob_client(b"\x00\x01\x02", "video/mp4"))
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)):
            result = await provider.retrieve("clip.mp4")

        assert result == b"\x00\x01\x02"

    @pytest.mark.asyncio
    async def test_retrieve_returns_bytes_when_json_is_malformed(self, provider: AzureBlobStorageProvider) -> None:
        service = _make_service_client(_make_blob_client(b"not json", "application/json"))
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)):
            result = await provider.retrieve("broken.json")

        assert result == b"not json"

    @pytest.mark.asyncio
    async def test_delete(self, provider: AzureBlobStorageProvider) -> None:
        blob_client = _make_blob_client()
        service = _make_service_client(blob_client)
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)):
            assert await provider.delete("gone.json") is True

        blob_client.delete_blob.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("present", [True, False])
    async def test_exists(self, provider: AzureBlobStorageProvider, present: bool) -> None:
        blob_client = _make_blob_client()
        blob_client.exists = AsyncMock(return_value=present)
        service = _make_service_client(blob_client)
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)):
            assert await provider.exists("maybe.json") is present

    @pytest.mark.asyncio
    async def test_list_keys_filters_by_prefix(self) -> None:
        config = AzureBlobConfig(
            container_name=_CONTAINER, account_name=_ACCOUNT, account_key="k", key_prefix="runs/42/"
        )
        provider = AzureBlobStorageProvider(config=config)
        names = ["runs/42/a.json", "runs/42/b.json", "runs/43/c.json"]
        service = _make_service_client(_make_blob_client(), blob_names=names)
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)):
            keys = [key async for key in provider.list_keys()]

        assert keys == ["runs/42/a.json", "runs/42/b.json"]

    @pytest.mark.asyncio
    async def test_upload_file_returns_azure_uri(self, provider: AzureBlobStorageProvider, tmp_path: Path) -> None:
        source = tmp_path / "clip.mp4"
        source.write_bytes(b"video-bytes")
        blob_client = _make_blob_client()
        service = _make_service_client(blob_client)
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)):
            url = await provider.upload_file(source, "out/clip.mp4")

        assert url == f"azure://{_CONTAINER}/out/clip.mp4"
        # Content type is guessed from the extension when the caller does not supply one.
        assert blob_client.upload_blob.call_args.kwargs["content_settings"].content_type == "video/mp4"

    @pytest.mark.asyncio
    async def test_upload_file_honors_explicit_content_type(
        self, provider: AzureBlobStorageProvider, tmp_path: Path
    ) -> None:
        source = tmp_path / "blob.bin"
        source.write_bytes(b"x")
        blob_client = _make_blob_client()
        service = _make_service_client(blob_client)
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)):
            await provider.upload_file(source, "out/blob.bin", content_type="application/json")

        assert blob_client.upload_blob.call_args.kwargs["content_settings"].content_type == "application/json"

    @pytest.mark.asyncio
    async def test_store_file_returns_raw_uri_and_sas_url(
        self, provider: AzureBlobStorageProvider, tmp_path: Path
    ) -> None:
        source = tmp_path / "results.json"
        source.write_text("{}")
        service = _make_service_client(_make_blob_client())
        with (
            patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)),
            patch(f"{_PROVIDER_MODULE}.generate_blob_sas", Mock(return_value="sig=minted")),
        ):
            urls = await provider.store_file(source, "out/results.json", content_type="application/json")

        assert isinstance(urls, StorageUrls)
        assert urls.raw == f"azure://{_CONTAINER}/out/results.json"
        assert urls.presigned == (f"https://{_ACCOUNT}.blob.core.windows.net/{_CONTAINER}/out/results.json?sig=minted")

    @pytest.mark.asyncio
    async def test_generate_presigned_url_mints_read_sas(self, provider: AzureBlobStorageProvider) -> None:
        with patch(f"{_PROVIDER_MODULE}.generate_blob_sas", Mock(return_value="sig=read")) as mock_sas:
            url = await provider.generate_presigned_url("out/clip.mp4", expiration=900)

        assert url == f"https://{_ACCOUNT}.blob.core.windows.net/{_CONTAINER}/out/clip.mp4?sig=read"
        kwargs = mock_sas.call_args.kwargs
        assert kwargs["account_name"] == _ACCOUNT
        assert kwargs["container_name"] == _CONTAINER
        assert kwargs["blob_name"] == "out/clip.mp4"
        assert kwargs["account_key"] == "secret"
        assert kwargs["permission"].read is True
        assert not kwargs["permission"].write

    @pytest.mark.asyncio
    async def test_generate_presigned_url_maps_put_operation_to_write_sas(
        self, provider: AzureBlobStorageProvider
    ) -> None:
        with patch(f"{_PROVIDER_MODULE}.generate_blob_sas", Mock(return_value="sig=write")) as mock_sas:
            await provider.generate_presigned_url("out/clip.mp4", operation="put_object")

        permission = mock_sas.call_args.kwargs["permission"]
        assert permission.write is True
        assert permission.create is True

    @pytest.mark.asyncio
    async def test_generate_presigned_url_applies_key_prefix(self) -> None:
        config = AzureBlobConfig(
            container_name=_CONTAINER, account_name=_ACCOUNT, account_key="k", key_prefix="runs/42/"
        )
        provider = AzureBlobStorageProvider(config=config)
        with patch(f"{_PROVIDER_MODULE}.generate_blob_sas", Mock(return_value="sig=read")) as mock_sas:
            url = await provider.generate_presigned_url("clip.mp4")

        assert mock_sas.call_args.kwargs["blob_name"] == "runs/42/clip.mp4"
        assert url.endswith(f"/{_CONTAINER}/runs/42/clip.mp4?sig=read")

    @pytest.mark.asyncio
    async def test_generate_presigned_url_reuses_configured_sas_token(self) -> None:
        """Without an account key there is nothing to sign with, so the configured token is appended as-is."""
        config = AzureBlobConfig(container_name=_CONTAINER, account_name=_ACCOUNT, sas_token="sv=2024&sig=fixed")
        provider = AzureBlobStorageProvider(config=config)
        with patch(f"{_PROVIDER_MODULE}.generate_blob_sas", Mock()) as mock_sas:
            url = await provider.generate_presigned_url("clip.mp4", expiration=60)

        mock_sas.assert_not_called()
        assert url == f"https://{_ACCOUNT}.blob.core.windows.net/{_CONTAINER}/clip.mp4?sv=2024&sig=fixed"

    @pytest.mark.asyncio
    async def test_download_file_writes_destination(self, provider: AzureBlobStorageProvider, tmp_path: Path) -> None:
        destination = tmp_path / "nested" / "clip.mp4"
        service = _make_service_client(_make_blob_client(b"video-bytes"))
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)):
            result = await provider.download_file("out/clip.mp4", destination)

        assert result == destination
        assert destination.read_bytes() == b"video-bytes"

    @pytest.mark.asyncio
    async def test_download_file_removes_partial_file_on_failure(
        self, provider: AzureBlobStorageProvider, tmp_path: Path
    ) -> None:
        destination = tmp_path / "clip.mp4"
        blob_client = _make_blob_client()
        # Fail during the transfer rather than before it, so the destination file has already been created and the
        # cleanup branch is what has to remove it.
        downloader = await blob_client.download_blob()
        downloader.readinto = AsyncMock(side_effect=ServiceRequestError(message="connection reset"))
        service = _make_service_client(blob_client)
        with (
            patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)),
            pytest.raises(StorageConnectionError),
        ):
            await provider.download_file("out/clip.mp4", destination)

        assert not destination.exists()

    @pytest.mark.asyncio
    async def test_download_from_url_symlinks_local_path(
        self, provider: AzureBlobStorageProvider, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.mp4"
        source.write_bytes(b"local-bytes")
        destination = tmp_path / "dest.mp4"

        result = await provider.download_from_url(str(source), destination)

        assert result == destination
        assert destination.is_symlink()
        assert destination.read_bytes() == b"local-bytes"

    @pytest.mark.asyncio
    async def test_download_from_url_symlinks_file_uri(
        self, provider: AzureBlobStorageProvider, tmp_path: Path
    ) -> None:
        source = tmp_path / "source.mp4"
        source.write_bytes(b"local-bytes")
        destination = tmp_path / "dest.mp4"

        result = await provider.download_from_url(f"file://{source}", destination)

        assert result == destination
        assert destination.read_bytes() == b"local-bytes"

    @pytest.mark.asyncio
    async def test_download_from_url_fetches_azure_uri_with_own_credential(
        self, provider: AzureBlobStorageProvider, tmp_path: Path
    ) -> None:
        """A bare azure:// URI carries no account or SAS, so it is resolved against the configured account."""
        destination = tmp_path / "clip.mp4"
        blob_client = _make_blob_client(b"blob-bytes")
        from_blob_url = _make_from_blob_url(blob_client)
        with (
            patch(f"{_PROVIDER_MODULE}.BlobClient.from_blob_url", from_blob_url),
            patch(f"{_PROVIDER_MODULE}.check_url_security", Mock()) as mock_security,
        ):
            result = await provider.download_from_url(f"azure://{_CONTAINER}/out/clip.mp4", destination)

        assert result == destination
        assert destination.read_bytes() == b"blob-bytes"
        args, kwargs = from_blob_url.call_args
        assert args[0] == f"https://{_ACCOUNT}.blob.core.windows.net/{_CONTAINER}/out/clip.mp4"
        assert kwargs["credential"] == "secret"
        # Public blob URLs are re-checked immediately before the fetch.
        mock_security.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_from_url_uses_sas_from_url_instead_of_credential(
        self, provider: AzureBlobStorageProvider, tmp_path: Path
    ) -> None:
        """A SAS-bearing URL is already authorized, so no credential is attached."""
        destination = tmp_path / "clip.mp4"
        blob_client = _make_blob_client(b"sas-bytes")
        from_blob_url = _make_from_blob_url(blob_client)
        url = f"https://{_ACCOUNT}.blob.core.windows.net/{_CONTAINER}/clip.mp4?sig=abc&se=2026-01-01T00%3A00%3A00Z"
        with (
            patch(f"{_PROVIDER_MODULE}.BlobClient.from_blob_url", from_blob_url),
            patch(f"{_PROVIDER_MODULE}.check_url_security", Mock()),
        ):
            await provider.download_from_url(url, destination)

        assert from_blob_url.call_args[0][0] == url
        assert from_blob_url.call_args.kwargs["credential"] is None

    @pytest.mark.asyncio
    async def test_download_from_url_rejects_non_azure_url(
        self, provider: AzureBlobStorageProvider, tmp_path: Path
    ) -> None:
        with pytest.raises(AzureBlobStorageError, match="Not an Azure Blob Storage URL"):
            await provider.download_from_url(
                "https://bucket.s3.us-west-2.amazonaws.com/key?X-Amz-Signature=abc", tmp_path / "x"
            )

    @pytest.mark.asyncio
    async def test_download_from_url_rejects_s3_uri(self, provider: AzureBlobStorageProvider, tmp_path: Path) -> None:
        with pytest.raises(AzureBlobStorageError, match="Not an Azure Blob Storage URL"):
            await provider.download_from_url("s3://bucket/key", tmp_path / "x")

    @pytest.mark.asyncio
    async def test_download_from_url_removes_partial_file_on_failure(
        self, provider: AzureBlobStorageProvider, tmp_path: Path
    ) -> None:
        destination = tmp_path / "clip.mp4"
        blob_client = _make_blob_client()
        # Fail during the transfer rather than before it, so the destination file has already been created and the
        # cleanup branch is what has to remove it.
        downloader = await blob_client.download_blob()
        downloader.readinto = AsyncMock(side_effect=ServiceRequestError(message="connection reset"))
        from_blob_url = _make_from_blob_url(blob_client)
        with (
            patch(f"{_PROVIDER_MODULE}.BlobClient.from_blob_url", from_blob_url),
            patch(f"{_PROVIDER_MODULE}.check_url_security", Mock()),
            pytest.raises(AzureBlobStorageError, match="Failed to download"),
        ):
            await provider.download_from_url(f"azure://{_CONTAINER}/out/clip.mp4", destination)

        assert not destination.exists()

    @pytest.mark.asyncio
    async def test_download_from_url_allows_http_endpoint_override(self, tmp_path: Path) -> None:
        """An operator-configured http:// endpoint (Azurite) must not be rejected by the SSRF guard."""
        config = AzureBlobConfig(
            container_name=_CONTAINER, account_name=_ACCOUNT, account_key="k", endpoint_url=_AZURITE_ENDPOINT
        )
        provider = AzureBlobStorageProvider(config=config)
        destination = tmp_path / "clip.mp4"
        blob_client = _make_blob_client(b"emulated")
        from_blob_url = _make_from_blob_url(blob_client)
        with (
            patch(f"{_PROVIDER_MODULE}.BlobClient.from_blob_url", from_blob_url),
            patch(f"{_PROVIDER_MODULE}.check_url_security", Mock()) as mock_security,
        ):
            await provider.download_from_url(f"azure://{_CONTAINER}/out/clip.mp4", destination)

        assert destination.read_bytes() == b"emulated"
        assert from_blob_url.call_args[0][0] == f"{_AZURITE_ENDPOINT}/{_ACCOUNT}/{_CONTAINER}/out/clip.mp4"
        # The real guard rejects plain http:// outright, so it must be skipped for the configured endpoint.
        mock_security.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_releases_client(self, provider: AzureBlobStorageProvider) -> None:
        service = _make_service_client(_make_blob_client())
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)):
            await provider._get_client()
            await provider.close()

        service.close.assert_awaited_once()
        assert provider._service_client is None

    @pytest.mark.asyncio
    async def test_async_context_manager_closes_client(self, config: AzureBlobConfig) -> None:
        service = _make_service_client(_make_blob_client())
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)):
            async with AzureBlobStorageProvider(config=config) as provider:
                await provider._get_client()

        service.close.assert_awaited_once()


class TestAzureErrorMapping:
    """The Azure exception hierarchy must map onto the same backend-neutral errors the S3 provider raises."""

    @pytest.fixture
    def provider(self) -> AzureBlobStorageProvider:
        config = AzureBlobConfig(container_name=_CONTAINER, account_name=_ACCOUNT, account_key="secret")
        return AzureBlobStorageProvider(config=config)

    async def _run_with_error(self, provider: AzureBlobStorageProvider, error: Exception) -> None:
        """Drives ``retrieve`` against a client whose download_blob raises ``error``."""
        blob_client = _make_blob_client()
        blob_client.download_blob = AsyncMock(side_effect=error)
        service = _make_service_client(blob_client)
        with patch(f"{_PROVIDER_MODULE}.BlobServiceClient", MagicMock(return_value=service)):
            await provider.retrieve("some/key.json")

    @pytest.mark.asyncio
    async def test_resource_not_found_maps_to_storage_not_found(self, provider: AzureBlobStorageProvider) -> None:
        with pytest.raises(StorageNotFoundError) as exc_info:
            await self._run_with_error(provider, ResourceNotFoundError(message="missing"))
        assert exc_info.value.key == "some/key.json"
        assert exc_info.value.operation == "retrieve"

    @pytest.mark.asyncio
    async def test_client_authentication_error_maps_to_storage_authentication(
        self, provider: AzureBlobStorageProvider
    ) -> None:
        with pytest.raises(StorageAuthenticationError):
            await self._run_with_error(provider, ClientAuthenticationError(message="bad key"))

    @pytest.mark.asyncio
    async def test_forbidden_maps_to_storage_permission(self, provider: AzureBlobStorageProvider) -> None:
        with pytest.raises(StoragePermissionError):
            await self._run_with_error(provider, _http_response_error(403, "AuthorizationFailure"))

    @pytest.mark.asyncio
    async def test_quota_code_maps_to_storage_quota_exceeded(self, provider: AzureBlobStorageProvider) -> None:
        with pytest.raises(StorageQuotaExceededError):
            await self._run_with_error(provider, _http_response_error(409, "AccountIsOverQuota"))

    @pytest.mark.asyncio
    async def test_service_request_error_maps_to_storage_connection(self, provider: AzureBlobStorageProvider) -> None:
        with pytest.raises(StorageConnectionError):
            await self._run_with_error(provider, ServiceRequestError(message="no route"))

    @pytest.mark.asyncio
    async def test_other_http_error_maps_to_azure_storage_error(self, provider: AzureBlobStorageProvider) -> None:
        with pytest.raises(AzureBlobStorageError) as exc_info:
            await self._run_with_error(provider, _http_response_error(500, "InternalError"))
        assert exc_info.value.container == _CONTAINER
        assert exc_info.value.error_code == "InternalError"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
