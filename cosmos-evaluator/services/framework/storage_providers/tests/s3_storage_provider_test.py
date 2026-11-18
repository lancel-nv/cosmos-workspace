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

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
from botocore.exceptions import ClientError, NoCredentialsError
from multidict import CIMultiDict, CIMultiDictProxy
from pydantic import ValidationError
import pytest
from yarl import URL

from services.framework.protocols.storage_provider import StorageProvider, StorageUrls
from services.framework.storage_providers.config import S3Config
from services.framework.storage_providers.exceptions import (
    S3StorageError,
    StorageAuthenticationError,
    StorageNotFoundError,
    StoragePermissionError,
    StorageQuotaExceededError,
)
from services.framework.storage_providers.s3_storage_provider import S3StorageProvider, _describe_download_error
from services.settings_base import S3Credentials

_CLIENT_SESSION_TARGET = "services.framework.storage_providers.s3_storage_provider.aiohttp.ClientSession"


def _make_response_cm(
    status: int,
    body: bytes = b"",
    headers: dict[str, str] | None = None,
    *,
    body_error: BaseException | None = None,
    status_error: BaseException | None = None,
) -> AsyncMock:
    """Builds a mock aiohttp response context manager.

    ``body_error`` is raised after ``body`` has been yielded, simulating a transfer that dies partway through the
    payload; ``status_error`` is raised from ``raise_for_status()`` instead, simulating an HTTP error response.
    """
    response = AsyncMock()
    response.status = status
    response.headers = headers or {}
    response.raise_for_status = MagicMock(side_effect=status_error)

    async def iter_chunked(size: int) -> AsyncIterator[bytes]:
        for offset in range(0, len(body), size):
            yield body[offset : offset + size]
        if body_error is not None:
            raise body_error

    response.content.iter_chunked = iter_chunked
    return AsyncMock(__aenter__=AsyncMock(return_value=response), __aexit__=AsyncMock(return_value=None))


def _make_session_cm(get_side_effect: Any) -> AsyncMock:
    """Wraps ``get_side_effect`` in a mock ClientSession context manager."""
    session = AsyncMock()
    session.get = MagicMock(side_effect=get_side_effect)
    return AsyncMock(__aenter__=AsyncMock(return_value=session), __aexit__=AsyncMock(return_value=None))


def _http_error(status: int) -> aiohttp.ClientResponseError:
    """Builds the ClientResponseError that ``raise_for_status()`` would raise for ``status``."""
    return aiohttp.ClientResponseError(MagicMock(), (), status=status, message=f"HTTP {status}")


def _parse_range(headers: dict[str, str] | None) -> tuple[int, int] | None:
    """Extracts ``(start, end)`` from a ``Range: bytes=<start>-<end>`` header, or None when there is no Range."""
    byte_range = (headers or {}).get("Range", "")
    if not byte_range.startswith("bytes="):
        return None
    start_s, end_s = byte_range[len("bytes=") :].split("-")
    return int(start_s), int(end_s)


class TestS3Config:
    """Tests suite for S3Config."""

    def test_config_initialization_minimal(self) -> None:
        """Tests S3Config with minimal required parameters."""
        config = S3Config(bucket_name="test-bucket")

        assert config.bucket_name == "test-bucket"
        assert config.region_name == "us-east-1"
        assert config.use_ssl is True
        assert config.verify_ssl is True
        assert config.storage_class == "STANDARD"
        assert config.key_prefix == ""

    def test_config_key_prefix(self) -> None:
        """Tests S3Config key_prefix field."""
        config = S3Config(bucket_name="test-bucket", key_prefix="my/prefix/")
        assert config.key_prefix == "my/prefix/"

    def test_config_initialization_full(self) -> None:
        """Tests S3Config with all parameters."""
        config = S3Config(
            bucket_name="test-bucket",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            aws_session_token="test-token",
            region_name="us-west-2",
            endpoint_url="https://custom-s3.example.com",
            use_ssl=False,
            verify_ssl=False,
            multipart_threshold=128 * 1024 * 1024,
            multipart_chunksize=32 * 1024 * 1024,
            max_concurrency=20,
            default_content_type="text/plain",
            server_side_encryption="aws:kms",
            kms_key_id="test-key-id",
            max_attempts=5,
            retry_mode="standard",
            storage_class="REDUCED_REDUNDANCY",
            enable_versioning=True,
        )

        assert config.bucket_name == "test-bucket"
        assert config.aws_access_key_id is not None
        assert config.aws_access_key_id.get_secret_value() == "test-key"
        assert config.aws_secret_access_key is not None
        assert config.aws_secret_access_key.get_secret_value() == "test-secret"
        assert config.aws_session_token is not None
        assert config.aws_session_token.get_secret_value() == "test-token"
        assert config.region_name == "us-west-2"
        assert config.endpoint_url == "https://custom-s3.example.com"
        assert config.use_ssl is False
        assert config.verify_ssl is False
        assert config.multipart_threshold == 128 * 1024 * 1024
        assert config.multipart_chunksize == 32 * 1024 * 1024
        assert config.max_concurrency == 20
        assert config.default_content_type == "text/plain"
        assert config.server_side_encryption == "aws:kms"
        assert config.kms_key_id == "test-key-id"
        assert config.max_attempts == 5
        assert config.retry_mode == "standard"
        assert config.storage_class == "REDUCED_REDUNDANCY"
        assert config.enable_versioning is True

    def test_config_download_defaults(self) -> None:
        """The HTTP download knobs default to a 1 MiB chunk and a 3-attempt doubling backoff."""
        config = S3Config(bucket_name="test-bucket")

        assert config.download_chunk_size == 1 << 20
        assert config.download_max_attempts == 3
        assert config.download_retry_backoff_base == 0.5
        assert config.download_retry_backoff_factor == 2.0
        # Distinct from botocore's client-level retry budget, which governs boto3 calls rather than HTTP downloads.
        assert config.max_attempts == 3

    @pytest.mark.parametrize(
        "field,value",
        [
            # A zero/negative chunk size makes iter_chunked yield empty bytes forever; zero attempts makes
            # _transfer_with_retry skip the transfer entirely and report success on an empty file.
            ("download_chunk_size", 0),
            ("download_chunk_size", -1),
            ("download_max_attempts", 0),
            ("download_retry_backoff_base", -0.1),
            ("download_retry_backoff_factor", -1.0),
        ],
    )
    def test_config_download_bounds_reject_invalid_values(self, field: str, value: float) -> None:
        """Out-of-range download knobs fail at the config boundary rather than mid-transfer."""
        with pytest.raises(ValidationError):
            S3Config(bucket_name="test-bucket", **{field: value})

    def test_get_boto3_config(self) -> None:
        """Tests boto3 config generation."""
        config = S3Config(
            bucket_name="test-bucket",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-west-2",
            endpoint_url="https://custom-s3.example.com",
            use_ssl=False,
            verify_ssl=False,
        )

        boto3_config = config.get_boto3_config()

        assert boto3_config["region_name"] == "us-west-2"
        assert boto3_config["endpoint_url"] == "https://custom-s3.example.com"
        assert boto3_config["aws_access_key_id"] == "test-key"
        assert boto3_config["aws_secret_access_key"] == "test-secret"
        assert boto3_config["use_ssl"] is False
        assert boto3_config["verify"] is False

    def test_get_transfer_config(self) -> None:
        """Tests transfer config generation."""
        MULTIPART_THRESHOLD = 128 * 1024 * 1024
        MULTIPART_CHUNKSIZE = 32 * 1024 * 1024
        MAX_CONCURRENCY = 20

        config = S3Config(
            bucket_name="test-bucket",
            multipart_threshold=MULTIPART_THRESHOLD,
            multipart_chunksize=MULTIPART_CHUNKSIZE,
            max_concurrency=MAX_CONCURRENCY,
        )

        transfer_config = config.get_transfer_config()

        assert transfer_config["multipart_threshold"] == MULTIPART_THRESHOLD
        assert transfer_config["multipart_chunksize"] == MULTIPART_CHUNKSIZE
        assert transfer_config["max_concurrency"] == MAX_CONCURRENCY
        assert transfer_config["use_threads"] is True


class TestS3StorageProvider:
    """Tests suite for S3StorageProvider."""

    @pytest.fixture
    def config(self) -> S3Config:
        """Provides a standard S3 config."""
        return S3Config(bucket_name="test-bucket", region_name="us-east-1")

    @pytest.fixture
    def provider(self, config: S3Config) -> S3StorageProvider:
        """Provides an S3 storage provider."""
        return S3StorageProvider(config)

    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        """Provides a mock S3 client."""
        return AsyncMock()

    def test_storage_provider_implements_protocol(self, provider: S3StorageProvider) -> None:
        assert isinstance(provider, StorageProvider)

    def test_provider_initialization(self, provider: S3StorageProvider, config: S3Config) -> None:
        """Tests that provider initializes correctly."""
        assert provider.config == config
        assert provider._session is None
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_get_client_creation(self, provider: S3StorageProvider) -> None:
        """Tests S3 client creation."""
        with patch("services.framework.storage_providers.s3_storage_provider.aioboto3") as mock_aioboto3:
            mock_session = AsyncMock()
            mock_client_ctx = AsyncMock()
            mock_client = AsyncMock()
            mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
            mock_aioboto3.Session.return_value = mock_session
            mock_session.client = Mock(return_value=mock_client_ctx)

            client = await provider._get_client()

            assert client == mock_client
            assert provider._session == mock_session
            assert provider._client == mock_client
            assert provider._client_ctx == mock_client_ctx
            mock_session.client.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_client_reuse(self, provider: S3StorageProvider) -> None:
        """Tests that S3 client is reused."""
        mock_client = AsyncMock()
        provider._client = mock_client

        client = await provider._get_client()

        assert client == mock_client

    @pytest.mark.asyncio
    async def test_concurrent_client_access(self, provider: S3StorageProvider) -> None:
        """Tests that concurrent client access is properly synchronized."""
        with patch("services.framework.storage_providers.s3_storage_provider.aioboto3") as mock_aioboto3:
            mock_session = AsyncMock()
            mock_client_ctx = AsyncMock()
            mock_client = AsyncMock()
            mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
            mock_aioboto3.Session.return_value = mock_session
            mock_session.client = Mock(return_value=mock_client_ctx)

            # Simulate concurrent access
            tasks = [provider._get_client() for _ in range(10)]
            clients = await asyncio.gather(*tasks)

            # Should only create one session and client
            assert mock_aioboto3.Session.call_count == 1
            assert mock_session.client.call_count == 1
            assert all(c == mock_client for c in clients)

    def test_full_key_no_prefix(self, provider: S3StorageProvider) -> None:
        """Tests _full_key with no key_prefix configured."""
        assert provider._full_key("some/key.txt") == "some/key.txt"

    def test_full_key_with_prefix(self) -> None:
        """Tests _full_key with key_prefix configured."""
        config = S3Config(bucket_name="test-bucket", key_prefix="my/prefix/")
        prov = S3StorageProvider(config)
        assert prov._full_key("file.json") == "my/prefix/file.json"

    def test_full_key_empty_key(self) -> None:
        """Tests _full_key with empty key."""
        config = S3Config(bucket_name="test-bucket", key_prefix="prefix/")
        prov = S3StorageProvider(config)
        assert prov._full_key("") == "prefix/"

    def test_make_key_prefix_no_parts(self, provider: S3StorageProvider) -> None:
        """No parts -> empty prefix."""
        assert provider.make_key_prefix() == ""

    def test_make_key_prefix_all_none(self, provider: S3StorageProvider) -> None:
        """All None parts -> empty prefix."""
        assert provider.make_key_prefix(None, None) == ""

    def test_make_key_prefix_all_empty(self, provider: S3StorageProvider) -> None:
        """All empty parts -> empty prefix."""
        assert provider.make_key_prefix("", "") == ""

    def test_make_key_prefix_only_slashes(self, provider: S3StorageProvider) -> None:
        """Parts that strip to nothing are skipped (no stray '/')."""
        assert provider.make_key_prefix("/", "//") == ""

    def test_make_key_prefix_strips_leading_slash(self, provider: S3StorageProvider) -> None:
        """Leading slashes on the user prefix are stripped (S3 keys are not absolute paths)."""
        assert provider.make_key_prefix("/tmp/x", "egomotion", "cam_front") == "tmp/x/egomotion/cam_front/"

    def test_make_key_prefix_strips_trailing_slash(self, provider: S3StorageProvider) -> None:
        """Trailing slashes are normalized."""
        assert provider.make_key_prefix("/tmp/x/", "egomotion", "cam_front") == "tmp/x/egomotion/cam_front/"

    def test_make_key_prefix_relative_first_part(self, provider: S3StorageProvider) -> None:
        """Relative first part joins normally."""
        assert (
            provider.make_key_prefix("my-team/run-42", "egomotion", "cam_front")
            == "my-team/run-42/egomotion/cam_front/"
        )

    def test_make_key_prefix_skips_none_and_empty(self, provider: S3StorageProvider) -> None:
        """None and empty parts are skipped without doubling separators."""
        assert provider.make_key_prefix("foo", None, "bar", "", "baz") == "foo/bar/baz/"

    def test_make_key_prefix_strips_repeated_slashes(self, provider: S3StorageProvider) -> None:
        """Per-segment strip removes any number of leading/trailing slashes."""
        assert provider.make_key_prefix("///foo///", "//bar//") == "foo/bar/"

    def test_make_key_prefix_preserves_inner_slashes(self, provider: S3StorageProvider) -> None:
        """Multi-level segments keep their inner slashes; only outer ones are stripped."""
        assert provider.make_key_prefix("my-team/run-42") == "my-team/run-42/"

    def test_make_key_prefix_single_segment_gets_trailing_slash(self, provider: S3StorageProvider) -> None:
        assert provider.make_key_prefix("foo") == "foo/"

    @pytest.mark.asyncio
    async def test_store_file_combines_upload_and_presign(self) -> None:
        """Tests store_file returns StorageUrls with raw and presigned URLs."""
        config = S3Config(bucket_name="test-bucket", key_prefix="output/")
        prov = S3StorageProvider(config)

        mock_client = AsyncMock()
        mock_client.upload_fileobj = AsyncMock()
        mock_client.generate_presigned_url = AsyncMock(return_value="https://presigned.url")
        prov._client = mock_client

        tmp = Path("/tmp/test_store_file.json")
        try:
            tmp.write_text('{"test": true}')
            result = await prov.store_file(tmp, "data.json", "application/json")
        finally:
            if tmp.exists():
                tmp.unlink()

        assert isinstance(result, StorageUrls)
        assert result.raw == "s3://test-bucket/output/data.json"
        assert result.presigned == "https://presigned.url"
        mock_client.upload_fileobj.assert_called_once()
        mock_client.generate_presigned_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_applies_key_prefix(self, mock_client: AsyncMock) -> None:
        """Tests that store() prepends key_prefix to the key."""
        config = S3Config(bucket_name="test-bucket", key_prefix="prefix/")
        prov = S3StorageProvider(config)
        prov._client = mock_client
        mock_client.put_object = AsyncMock()

        await prov.store("data", "file.txt")

        call_args = mock_client.put_object.call_args
        assert call_args.kwargs["Key"] == "prefix/file.txt"

    @pytest.mark.asyncio
    async def test_exists_applies_key_prefix(self, mock_client: AsyncMock) -> None:
        """Tests that exists() prepends key_prefix to the key."""
        config = S3Config(bucket_name="test-bucket", key_prefix="prefix/")
        prov = S3StorageProvider(config)
        prov._client = mock_client
        mock_client.head_object = AsyncMock()

        await prov.exists("file.txt")

        call_args = mock_client.head_object.call_args
        assert call_args.kwargs["Key"] == "prefix/file.txt"

    @pytest.mark.asyncio
    async def test_generate_presigned_url_applies_key_prefix(self, mock_client: AsyncMock) -> None:
        """Tests that generate_presigned_url() prepends key_prefix to the key."""
        config = S3Config(bucket_name="test-bucket", key_prefix="prefix/")
        prov = S3StorageProvider(config)
        prov._client = mock_client
        mock_client.generate_presigned_url = AsyncMock(return_value="https://presigned.url")

        await prov.generate_presigned_url("file.txt")

        call_args = mock_client.generate_presigned_url.call_args
        assert call_args.kwargs["Params"]["Key"] == "prefix/file.txt"

    @pytest.mark.asyncio
    async def test_concurrent_store_operations(self, provider: S3StorageProvider, mock_client: AsyncMock) -> None:
        """Tests that concurrent store operations work correctly."""
        provider._client = mock_client
        mock_client.put_object = AsyncMock()

        # Simulate multiple concurrent stores
        tasks = [provider.store(f"data-{i}", f"key-{i}.txt") for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert mock_client.put_object.call_count == 10

    def test_prepare_data_for_storage_bytes(self, provider: S3StorageProvider) -> None:
        """Tests data preparation with bytes input."""
        data = b"test data"
        result = provider._prepare_data_for_storage(data)
        assert result == data

    def test_prepare_data_for_storage_string(self, provider: S3StorageProvider) -> None:
        """Tests data preparation with string input."""
        data = "test string"
        result = provider._prepare_data_for_storage(data)
        assert result == data.encode("utf-8")

    def test_prepare_data_for_storage_dict(self, provider: S3StorageProvider) -> None:
        """Tests data preparation with dict input."""
        data = {"key": "value", "number": 42}
        result = provider._prepare_data_for_storage(data)
        expected = json.dumps(data, ensure_ascii=False).encode("utf-8")
        assert result == expected

    def test_prepare_data_for_storage_list(self, provider: S3StorageProvider) -> None:
        """Tests data preparation with list input."""
        data = [1, 2, 3, "test"]
        result = provider._prepare_data_for_storage(data)
        expected = json.dumps(data, ensure_ascii=False).encode("utf-8")
        assert result == expected

    def test_prepare_data_for_storage_other(self, provider: S3StorageProvider) -> None:
        """Tests data preparation with other types."""
        data = 42
        result = provider._prepare_data_for_storage(data)
        assert result == b"42"

    def test_prepare_metadata(self, provider: S3StorageProvider) -> None:
        """Tests metadata preparation."""
        metadata = {
            "user_id": "123",
            "upload_time": datetime.now(tz=UTC),
            "tags": ["tag1", "tag2"],
            "complex_data": {"nested": "value"},
        }

        result = provider._prepare_metadata(metadata)

        assert result["user-id"] == "123"
        assert "upload-time" in result
        assert "tags" in result
        assert "complex-data" in result

    def test_prepare_metadata_empty(self, provider: S3StorageProvider) -> None:
        """Tests metadata preparation with empty input."""
        result = provider._prepare_metadata(None)
        assert result == {}

        result = provider._prepare_metadata({})
        assert result == {}

    def test_determine_content_type_string(self, provider: S3StorageProvider) -> None:
        """Tests content type determination for strings."""
        assert provider._determine_content_type("test", "key") == "text/plain; charset=utf-8"

    def test_determine_content_type_dict(self, provider: S3StorageProvider) -> None:
        """Tests content type determination for dicts."""
        assert provider._determine_content_type({"key": "value"}, "key") == "application/json; charset=utf-8"

    def test_determine_content_type_by_extension(self, provider: S3StorageProvider) -> None:
        """Tests content type determination by file extension."""
        assert provider._determine_content_type(b"data", "file.json") == "application/json"
        assert provider._determine_content_type(b"data", "file.txt") == "text/plain"
        assert provider._determine_content_type(b"data", "file.html") == "text/html"
        assert provider._determine_content_type(b"data", "file.jpg") == "image/jpeg"
        assert provider._determine_content_type(b"data", "file.png") == "image/png"
        assert provider._determine_content_type(b"data", "file.pdf") == "application/pdf"

    def test_determine_content_type_default(self, provider: S3StorageProvider) -> None:
        """Tests default content type determination."""
        assert provider._determine_content_type(b"data", "unknown.unknownextension123") == "application/octet-stream"

    @pytest.mark.asyncio
    async def test_store_success(self, provider: S3StorageProvider, mock_client: AsyncMock) -> None:
        """Tests successful data storage."""
        provider._client = mock_client
        mock_client.put_object = AsyncMock()

        test_data = {"message": "test"}
        result = await provider.store(test_data, "test/key.json")

        assert result == "https://s3.us-east-1.amazonaws.com/test-bucket/test/key.json"
        mock_client.put_object.assert_called_once()

        call_args = mock_client.put_object.call_args[1]
        assert call_args["Bucket"] == "test-bucket"
        assert call_args["Key"] == "test/key.json"
        assert call_args["ContentType"] == "application/json; charset=utf-8"
        assert call_args["StorageClass"] == "STANDARD"

    @pytest.mark.asyncio
    async def test_store_with_metadata(self, provider: S3StorageProvider, mock_client: AsyncMock) -> None:
        """Tests data storage with metadata."""
        provider._client = mock_client
        mock_client.put_object = AsyncMock()

        test_data = "test string"
        test_metadata = {"user_id": "123", "category": "test"}

        await provider.store(test_data, "test/key.txt", test_metadata)

        call_args = mock_client.put_object.call_args[1]
        assert "user-id" in call_args["Metadata"]
        assert "category" in call_args["Metadata"]
        assert call_args["Metadata"]["user-id"] == "123"
        assert call_args["Metadata"]["category"] == "test"

    @pytest.mark.asyncio
    async def test_store_with_encryption(self, provider: S3StorageProvider, mock_client: AsyncMock) -> None:
        """Tests data storage with server-side encryption."""
        provider.config.server_side_encryption = "AES256"
        provider._client = mock_client
        mock_client.put_object = AsyncMock()

        await provider.store("test data", "test/key.txt")

        call_args = mock_client.put_object.call_args[1]
        assert call_args["ServerSideEncryption"] == "AES256"

    @pytest.mark.asyncio
    async def test_store_with_kms_encryption(self, provider: S3StorageProvider, mock_client: AsyncMock) -> None:
        """Tests data storage with KMS encryption."""
        provider.config.server_side_encryption = "aws:kms"
        provider.config.kms_key_id = "test-key-id"
        provider._client = mock_client
        mock_client.put_object = AsyncMock()

        await provider.store("test data", "test/key.txt")

        call_args = mock_client.put_object.call_args[1]
        assert call_args["ServerSideEncryption"] == "aws:kms"
        assert call_args["SSEKMSKeyId"] == "test-key-id"

    @pytest.mark.asyncio
    async def test_store_with_custom_endpoint(self, provider: S3StorageProvider, mock_client: AsyncMock) -> None:
        """Tests data storage with custom endpoint URL."""
        provider.config.endpoint_url = "https://custom-s3.example.com"
        provider._client = mock_client
        mock_client.put_object = AsyncMock()

        result = await provider.store("test data", "test/key.txt")

        assert result == "https://custom-s3.example.com/test-bucket/test/key.txt"

    @pytest.mark.asyncio
    async def test_retrieve_json_data(self, provider: S3StorageProvider, mock_client: AsyncMock) -> None:
        """Tests retrieving JSON data."""
        provider._client = mock_client

        test_data = {"message": "test", "number": 42}
        json_data = json.dumps(test_data).encode("utf-8")

        mock_response: dict[str, Any] = {"Body": AsyncMock(), "ContentType": "application/json"}
        mock_response["Body"].read = AsyncMock(return_value=json_data)
        mock_client.get_object = AsyncMock(return_value=mock_response)

        result = await provider.retrieve("test/key.json")

        assert result == test_data
        mock_client.get_object.assert_called_once_with(Bucket="test-bucket", Key="test/key.json")

    @pytest.mark.asyncio
    async def test_retrieve_text_data(self, provider: S3StorageProvider, mock_client: AsyncMock) -> None:
        """Tests retrieving text data."""
        provider._client = mock_client

        test_data = "test string content"

        mock_response: dict[str, Any] = {"Body": AsyncMock(), "ContentType": "text/plain"}
        mock_response["Body"].read = AsyncMock(return_value=test_data.encode("utf-8"))
        mock_client.get_object = AsyncMock(return_value=mock_response)

        result = await provider.retrieve("test/key.txt")

        assert result == test_data

    @pytest.mark.asyncio
    async def test_retrieve_binary_data(self, provider: S3StorageProvider, mock_client: AsyncMock) -> None:
        """Tests retrieving binary data."""
        provider._client = mock_client

        test_data = b"binary data content"

        mock_response: dict[str, Any] = {"Body": AsyncMock(), "ContentType": "application/octet-stream"}
        mock_response["Body"].read = AsyncMock(return_value=test_data)
        mock_client.get_object = AsyncMock(return_value=mock_response)

        result = await provider.retrieve("test/key.bin")

        assert result == test_data

    @pytest.mark.asyncio
    async def test_delete_success(self, provider: S3StorageProvider, mock_client: AsyncMock) -> None:
        """Tests successful data deletion."""
        provider._client = mock_client
        mock_client.delete_object = AsyncMock()

        result = await provider.delete("test/key.txt")

        assert result is True
        mock_client.delete_object.assert_called_once_with(Bucket="test-bucket", Key="test/key.txt")

    @pytest.mark.asyncio
    async def test_exists_true(self, provider: S3StorageProvider, mock_client: AsyncMock) -> None:
        """Tests exists check when object exists."""
        provider._client = mock_client
        mock_client.head_object = AsyncMock()

        result = await provider.exists("test/key.txt")

        assert result is True
        mock_client.head_object.assert_called_once_with(Bucket="test-bucket", Key="test/key.txt")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("error_code", ["404", "NoSuchKey", "Not Found"])
    async def test_exists_false(self, provider: S3StorageProvider, mock_client: AsyncMock, error_code: str) -> None:
        """Tests exists check when object doesn't exist."""
        provider._client = mock_client

        error_response = {"Error": {"Code": error_code}}
        mock_client.head_object = AsyncMock(side_effect=ClientError(error_response, "HeadObject"))

        result = await provider.exists("test/nonexistent.txt")

        assert result is False

    @pytest.mark.asyncio
    async def test_list_keys(self, provider: S3StorageProvider, mock_client: AsyncMock) -> None:
        """Tests listing storage keys."""
        provider._client = mock_client

        mock_paginator = AsyncMock()
        mock_client.get_paginator = MagicMock(return_value=mock_paginator)

        # Mock paginated results
        mock_pages = [
            {"Contents": [{"Key": "test/file1.txt"}, {"Key": "test/file2.txt"}]},
            {"Contents": [{"Key": "test/file3.txt"}]},
            {},  # Empty page to end iteration
        ]

        async def mock_paginate(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
            for page in mock_pages:
                yield page

        mock_paginator.paginate = MagicMock(return_value=mock_paginate())

        keys = []
        async for key in provider.list_keys("test/"):
            keys.append(key)

        assert keys == ["test/file1.txt", "test/file2.txt", "test/file3.txt"]
        mock_client.get_paginator.assert_called_once_with("list_objects_v2")

    @pytest.mark.asyncio
    async def test_get_metadata(self, provider: S3StorageProvider, mock_client: AsyncMock) -> None:
        """Tests getting object metadata."""
        provider._client = mock_client

        mock_response = {
            "ContentLength": 1024,
            "ContentType": "application/json",
            "LastModified": datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
            "ETag": '"abc123"',
            "StorageClass": "STANDARD",
            "ServerSideEncryption": "AES256",
            "Metadata": {"user-id": "123", "category": "test"},
            "VersionId": "v1.0",
        }
        mock_client.head_object = AsyncMock(return_value=mock_response)

        metadata = await provider.get_metadata("test/key.json")

        assert metadata["content_length"] == 1024
        assert metadata["content_type"] == "application/json"
        assert metadata["etag"] == "abc123"
        assert metadata["storage_class"] == "STANDARD"
        assert metadata["server_side_encryption"] == "AES256"
        assert metadata["user_metadata"] == {"user-id": "123", "category": "test"}
        assert metadata["version_id"] == "v1.0"

    @pytest.mark.asyncio
    async def test_copy_success(self, provider: S3StorageProvider, mock_client: AsyncMock) -> None:
        """Tests successful object copy."""
        provider._client = mock_client
        mock_client.copy_object = AsyncMock()

        result = await provider.copy("source/key.txt", "dest/key.txt")

        assert result == "https://s3.us-east-1.amazonaws.com/test-bucket/dest/key.txt"
        mock_client.copy_object.assert_called_once()

        call_args = mock_client.copy_object.call_args[1]
        assert call_args["CopySource"]["Bucket"] == "test-bucket"
        assert call_args["CopySource"]["Key"] == "source/key.txt"
        assert call_args["Bucket"] == "test-bucket"
        assert call_args["Key"] == "dest/key.txt"

    @pytest.mark.asyncio
    async def test_copy_with_metadata(self, provider: S3StorageProvider, mock_client: AsyncMock) -> None:
        """Tests object copy with new metadata."""
        provider._client = mock_client
        mock_client.copy_object = AsyncMock()

        metadata = {"new_field": "value"}
        await provider.copy("source/key.txt", "dest/key.txt", metadata)

        call_args = mock_client.copy_object.call_args[1]
        assert call_args["MetadataDirective"] == "REPLACE"
        assert "new-field" in call_args["Metadata"]

    @pytest.mark.asyncio
    async def test_generate_presigned_url(self, provider: S3StorageProvider, mock_client: AsyncMock) -> None:
        """Tests presigned URL generation."""
        provider._client = mock_client
        expected_url = "https://test-bucket.s3.amazonaws.com/test/key.txt?signature=..."
        mock_client.generate_presigned_url = AsyncMock(return_value=expected_url)

        url = await provider.generate_presigned_url("test/key.txt", "get_object", 3600)

        assert url == expected_url
        mock_client.generate_presigned_url.assert_called_once_with(
            "get_object", Params={"Bucket": "test-bucket", "Key": "test/key.txt"}, ExpiresIn=3600
        )

    def test_handle_s3_error_no_such_bucket(self, provider: S3StorageProvider) -> None:
        """Tests S3 error handling for NoSuchBucket."""
        error_response = {"Error": {"Code": "NoSuchBucket", "Message": "Bucket does not exist"}}
        error = ClientError(error_response, "PutObject")

        with pytest.raises(S3StorageError) as exc_info:
            provider._handle_s3_error(error, "store", "test/key.txt")

        assert "Bucket 'test-bucket' does not exist" in str(exc_info.value)
        assert exc_info.value.aws_error_code == "NoSuchBucket"

    def test_handle_s3_error_no_such_key(self, provider: S3StorageProvider) -> None:
        """Tests S3 error handling for NoSuchKey."""
        error_response = {"Error": {"Code": "NoSuchKey", "Message": "Key does not exist"}}
        error = ClientError(error_response, "GetObject")

        with pytest.raises(StorageNotFoundError) as exc_info:
            provider._handle_s3_error(error, "retrieve", "test/key.txt")

        assert "Object 'test/key.txt' not found" in str(exc_info.value)

    def test_handle_s3_error_access_denied(self, provider: S3StorageProvider) -> None:
        """Tests S3 error handling for AccessDenied."""
        error_response = {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}
        error = ClientError(error_response, "PutObject")

        with pytest.raises(StoragePermissionError) as exc_info:
            provider._handle_s3_error(error, "store", "test/key.txt")

        assert "Access denied" in str(exc_info.value)

    def test_handle_s3_error_quota_exceeded(self, provider: S3StorageProvider) -> None:
        """Tests S3 error handling for QuotaExceeded."""
        error_response = {"Error": {"Code": "QuotaExceeded", "Message": "Quota exceeded"}}
        error = ClientError(error_response, "PutObject")

        with pytest.raises(StorageQuotaExceededError) as exc_info:
            provider._handle_s3_error(error, "store", "test/key.txt")

        assert "Storage quota exceeded" in str(exc_info.value)

    def test_handle_s3_error_no_credentials(self, provider: S3StorageProvider) -> None:
        """Tests S3 error handling for NoCredentialsError."""
        error = NoCredentialsError()

        with pytest.raises(StorageAuthenticationError) as exc_info:
            provider._handle_s3_error(error, "store", "test/key.txt")

        assert "AWS credentials not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_close(self, provider: S3StorageProvider) -> None:
        """Tests closing the provider."""
        mock_client = AsyncMock()
        mock_client_ctx = AsyncMock()
        mock_session = MagicMock()
        provider._client = mock_client
        provider._client_ctx = mock_client_ctx
        provider._session = mock_session

        await provider.close()

        mock_client_ctx.__aexit__.assert_called_once_with(None, None, None)
        assert provider._client is None
        assert provider._client_ctx is None
        assert provider._session is None

    @pytest.mark.asyncio
    async def test_context_manager(self, provider: S3StorageProvider) -> None:
        """Tests async context manager functionality."""
        with patch.object(provider, "close") as mock_close:
            async with provider as h:
                assert h == provider

            mock_close.assert_called_once()

    def test_multiple_provider_instances(self) -> None:
        """Tests that multiple provider instances work independently."""
        config1 = S3Config(bucket_name="bucket1", region_name="us-east-1")
        config2 = S3Config(bucket_name="bucket2", region_name="us-west-2")

        provider1 = S3StorageProvider(config1)
        provider2 = S3StorageProvider(config2)

        assert provider1.config.bucket_name == "bucket1"
        assert provider1.config.region_name == "us-east-1"
        assert provider2.config.bucket_name == "bucket2"
        assert provider2.config.region_name == "us-west-2"


class TestS3StorageProviderNewMethods:
    """Tests for newly added S3StorageProvider methods."""

    @pytest.fixture
    def config(self) -> S3Config:
        """Provides a standard S3 config."""
        return S3Config(
            bucket_name="test-bucket",
            aws_access_key_id="test-key",
            aws_secret_access_key="test-secret",
            region_name="us-east-1",
        )

    @pytest.fixture
    def provider(self, config: S3Config) -> S3StorageProvider:
        """Provides an S3 storage provider."""
        return S3StorageProvider(config)

    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        """Provides a mock S3 client."""
        return AsyncMock()

    def test_default_constants(self) -> None:
        """Tests default class constants."""
        assert S3StorageProvider.DEFAULT_PRESIGNED_URL_EXPIRATION == 43200

    @pytest.mark.asyncio
    async def test_upload_file(self, provider: S3StorageProvider, mock_client: AsyncMock, tmp_path: Path) -> None:
        """Tests uploading a file from disk."""
        provider._client = mock_client
        mock_client.upload_fileobj = AsyncMock()

        test_file = tmp_path / "test.json"
        test_file.write_text('{"key": "value"}')

        result = await provider.upload_file(test_file, "results/test.json", content_type="application/json")

        assert "test-bucket" in result
        assert "results/test.json" in result
        mock_client.upload_fileobj.assert_called_once()
        call_args = mock_client.upload_fileobj.call_args
        assert call_args[0][1] == "test-bucket"
        assert call_args[0][2] == "results/test.json"
        assert call_args[1]["ExtraArgs"]["ContentType"] == "application/json"

    @pytest.mark.asyncio
    async def test_upload_file_with_content_disposition(
        self, provider: S3StorageProvider, mock_client: AsyncMock, tmp_path: Path
    ) -> None:
        """Tests uploading a file with content disposition."""
        provider._client = mock_client
        mock_client.upload_fileobj = AsyncMock()

        test_file = tmp_path / "video.mp4"
        test_file.write_bytes(b"fake video")

        await provider.upload_file(
            test_file, "videos/video.mp4", content_type="video/mp4", content_disposition="attachment"
        )

        call_args = mock_client.upload_fileobj.call_args
        assert call_args[1]["ExtraArgs"]["ContentDisposition"] == "attachment"

    @pytest.mark.asyncio
    async def test_download_file(self, provider: S3StorageProvider, mock_client: AsyncMock, tmp_path: Path) -> None:
        """Tests downloading a file to disk."""
        provider._client = mock_client
        mock_client.download_fileobj = AsyncMock()

        destination = tmp_path / "downloaded.json"
        result = await provider.download_file("results/test.json", destination)

        assert result == destination
        mock_client.download_fileobj.assert_called_once()
        call_args = mock_client.download_fileobj.call_args
        assert call_args[0][0] == "test-bucket"
        assert call_args[0][1] == "results/test.json"

    @pytest.mark.asyncio
    async def test_download_file_creates_parent_dirs(
        self, provider: S3StorageProvider, mock_client: AsyncMock, tmp_path: Path
    ) -> None:
        """Tests that download_file creates parent directories."""
        provider._client = mock_client
        mock_client.download_fileobj = AsyncMock()

        destination = tmp_path / "a" / "b" / "c" / "file.txt"
        await provider.download_file("key.txt", destination)

        assert destination.parent.exists()

    @pytest.mark.asyncio
    async def test_generate_presigned_url_from_s3_url(
        self, provider: S3StorageProvider, mock_client: AsyncMock
    ) -> None:
        """Tests generating presigned URL from an S3 URL."""
        provider._client = mock_client
        expected_url = "https://test-bucket.s3.amazonaws.com/path/file.mp4?signature=..."
        mock_client.generate_presigned_url = AsyncMock(return_value=expected_url)

        result = await provider.generate_presigned_url_from_s3_url("s3://test-bucket/path/file.mp4", expiration=7200)

        assert result == expected_url
        mock_client.generate_presigned_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_presigned_url_from_virtual_hosted_url(
        self, provider: S3StorageProvider, mock_client: AsyncMock
    ) -> None:
        """Tests generating presigned URL from a virtual-hosted S3 URL."""
        provider._client = mock_client
        expected_url = "https://presigned.url"
        mock_client.generate_presigned_url = AsyncMock(return_value=expected_url)

        result = await provider.generate_presigned_url_from_s3_url(
            "https://my-bucket.s3.us-west-2.amazonaws.com/path/file.mp4"
        )

        assert result == expected_url

    @pytest.mark.asyncio
    async def test_generate_presigned_url_from_s3_url_uses_bucket_from_url(
        self, provider: S3StorageProvider, mock_client: AsyncMock
    ) -> None:
        """The bucket parsed from the URL must override ``self.config.bucket_name``.

        Regression test: this method previously presigned against
        ``self.config.bucket_name``, which meant arbitrary S3 URLs could not
        be downloaded by an ``S3StorageProvider`` configured for a different
        bucket. Now that the input download path can point at any S3 bucket
        independently of the provider's output bucket, we must use the
        bucket parsed out of the URL.
        """
        provider._client = mock_client
        mock_client.generate_presigned_url = AsyncMock(return_value="https://presigned.url")

        # Provider is configured for ``test-bucket`` (see config fixture above)
        # but the URL points at a different bucket entirely.
        await provider.generate_presigned_url_from_s3_url("s3://other-bucket/path/file.mp4")

        mock_client.generate_presigned_url.assert_called_once()
        params = mock_client.generate_presigned_url.call_args.kwargs["Params"]
        assert params["Bucket"] == "other-bucket", "Should presign against the URL's bucket, not config.bucket_name"
        assert params["Key"] == "path/file.mp4"

    @pytest.mark.asyncio
    async def test_download_from_url_presigned(self, provider: S3StorageProvider, tmp_path: Path) -> None:
        """Tests downloading from a presigned URL."""
        destination = tmp_path / "downloaded.mp4"
        presigned_url = (
            "https://bucket.s3.amazonaws.com/file.mp4"
            "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Credential=KEY%2F20260109%2Fus-east-1%2Fs3%2Faws4_request"
            "&X-Amz-Signature=abc123"
        )

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()

        async def mock_iter_chunked(size: int) -> AsyncIterator[bytes]:
            yield b"video data chunk 1"
            yield b"video data chunk 2"

        mock_response.content.iter_chunked = mock_iter_chunked

        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=None),
            )
        )

        mock_session_cm = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=None),
        )

        with patch(
            "services.framework.storage_providers.s3_storage_provider.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ):
            result = await provider.download_from_url(presigned_url, destination)

        assert result == destination
        assert destination.read_bytes() == b"video data chunk 1video data chunk 2"

    @pytest.mark.asyncio
    async def test_download_from_url_sets_inactivity_timeout(self, provider: S3StorageProvider, tmp_path: Path) -> None:
        """ClientSession is built with an inactivity (sock_read) timeout instead of aiohttp's default total cap."""
        destination = tmp_path / "f.bin"
        url = "https://bucket.s3.amazonaws.com/f.bin?X-Amz-Signature=abc"

        mock_response = AsyncMock()
        mock_response.status = 200  # range not honored -> single-stream path
        mock_response.raise_for_status = MagicMock()

        async def iter_chunked(size: int) -> AsyncIterator[bytes]:
            yield b"data"

        mock_response.content.iter_chunked = iter_chunked
        get_cm = AsyncMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock(return_value=None))
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=get_cm)
        session_cm = AsyncMock(__aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock(return_value=None))

        with patch(
            "services.framework.storage_providers.s3_storage_provider.aiohttp.ClientSession",
            return_value=session_cm,
        ) as mock_client_session:
            await provider.download_from_url(url, destination)

        timeout = mock_client_session.call_args.kwargs["timeout"]
        assert timeout.total is None, "must disable aiohttp's default 300s total cap"
        assert timeout.sock_read == provider.config.download_sock_read_timeout
        assert timeout.sock_connect == provider.config.download_sock_connect_timeout
        assert destination.read_bytes() == b"data"

    @pytest.mark.asyncio
    async def test_download_from_url_multipart_reassembles(self, tmp_path: Path) -> None:
        """A range-capable, over-threshold object is fetched via concurrent ranged GETs and reassembled byte-exact."""
        payload = bytes(range(256)) * 4096  # 1 MiB of deterministic bytes
        total = len(payload)
        # Tiny threshold + chunk so the download fans out into many parts.
        config = S3Config(bucket_name="test-bucket", multipart_threshold=1024, multipart_chunksize=4096)
        provider = S3StorageProvider(config)
        destination = tmp_path / "big.bin"
        url = "https://bucket.s3.amazonaws.com/big.bin?X-Amz-Signature=abc"

        def make_cm(status: int, headers: dict[str, str], body: bytes) -> AsyncMock:
            response = AsyncMock()
            response.status = status
            response.headers = headers
            response.raise_for_status = MagicMock()

            async def iter_chunked(size: int) -> AsyncIterator[bytes]:
                for i in range(0, len(body), size):
                    yield body[i : i + size]

            response.content.iter_chunked = iter_chunked
            return AsyncMock(__aenter__=AsyncMock(return_value=response), __aexit__=AsyncMock(return_value=None))

        def session_get(
            req_url: str, headers: dict[str, str] | None = None, ssl: Any = None, allow_redirects: bool = True
        ) -> AsyncMock:
            assert allow_redirects is False, "download GETs must not follow redirects (SSRF guard)"
            byte_range = (headers or {}).get("Range", "")
            assert byte_range.startswith("bytes="), "every GET in the ranged path must carry a Range header"
            start_s, end_s = byte_range[len("bytes=") :].split("-")
            start, end = int(start_s), int(end_s)
            if (start, end) == (0, 0):  # size probe
                return make_cm(206, {"Content-Range": f"bytes 0-0/{total}"}, payload[0:1])
            return make_cm(206, {"Content-Range": f"bytes {start}-{end}/{total}"}, payload[start : end + 1])

        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=session_get)
        session_cm = AsyncMock(__aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock(return_value=None))

        with patch(
            "services.framework.storage_providers.s3_storage_provider.aiohttp.ClientSession",
            return_value=session_cm,
        ):
            result = await provider.download_from_url(url, destination)

        assert result == destination
        assert destination.read_bytes() == payload

    @pytest.mark.asyncio
    async def test_download_from_url_retries_payload_error(self, tmp_path: Path) -> None:
        """A ClientPayloadError mid-body is retried, and the retry overwrites the partial first attempt.

        This is the aiohttp 3.14 failure mode where a clean connection close while the parser is paused by
        back-pressure surfaces a ContentLengthError even though every byte arrived.
        """
        config = S3Config(bucket_name="test-bucket", download_retry_backoff_base=0.0)
        provider = S3StorageProvider(config)
        destination = tmp_path / "video.mp4"
        url = "https://bucket.s3.amazonaws.com/video.mp4?X-Amz-Signature=abc"
        full_gets = 0

        def session_get(
            req_url: str, headers: dict[str, str] | None = None, ssl: Any = None, allow_redirects: bool = True
        ) -> AsyncMock:
            if _parse_range(headers) is not None:
                return _make_response_cm(200)  # probe: no range support -> single-stream path
            nonlocal full_gets
            full_gets += 1
            if full_gets == 1:
                payload_error = aiohttp.ClientPayloadError("Response payload is not completed")
                return _make_response_cm(200, b"truncated", body_error=payload_error)
            return _make_response_cm(200, b"the complete object")

        with patch(_CLIENT_SESSION_TARGET, return_value=_make_session_cm(session_get)):
            result = await provider.download_from_url(url, destination)

        assert result == destination
        assert destination.read_bytes() == b"the complete object", "retry must truncate the failed attempt's bytes"
        assert full_gets == 2

    @pytest.mark.asyncio
    async def test_download_from_url_retry_budget_exhausted(self, tmp_path: Path) -> None:
        """Once every attempt fails, the error surfaces as S3StorageError and no partial file is left behind."""
        config = S3Config(bucket_name="test-bucket", download_retry_backoff_base=0.0, download_max_attempts=3)
        provider = S3StorageProvider(config)
        destination = tmp_path / "video.mp4"
        url = "https://bucket.s3.amazonaws.com/video.mp4?X-Amz-Signature=abc"
        full_gets = 0

        def session_get(
            req_url: str, headers: dict[str, str] | None = None, ssl: Any = None, allow_redirects: bool = True
        ) -> AsyncMock:
            if _parse_range(headers) is not None:
                return _make_response_cm(200)
            nonlocal full_gets
            full_gets += 1
            return _make_response_cm(200, b"partial", body_error=aiohttp.ClientPayloadError("payload not completed"))

        with (
            patch(_CLIENT_SESSION_TARGET, return_value=_make_session_cm(session_get)),
            pytest.raises(S3StorageError, match="Failed to download"),
        ):
            await provider.download_from_url(url, destination)

        assert full_gets == 3, "must stop at download_max_attempts"
        assert not destination.exists()

    @pytest.mark.asyncio
    async def test_download_from_url_does_not_retry_4xx(self, tmp_path: Path) -> None:
        """A 403 (e.g. an expired presigned URL) fails on the first attempt - retrying can never help."""
        config = S3Config(bucket_name="test-bucket", download_retry_backoff_base=0.0)
        provider = S3StorageProvider(config)
        destination = tmp_path / "video.mp4"
        url = "https://bucket.s3.amazonaws.com/video.mp4?X-Amz-Signature=abc"
        full_gets = 0

        def session_get(
            req_url: str, headers: dict[str, str] | None = None, ssl: Any = None, allow_redirects: bool = True
        ) -> AsyncMock:
            if _parse_range(headers) is not None:
                return _make_response_cm(403)
            nonlocal full_gets
            full_gets += 1
            return _make_response_cm(403, status_error=_http_error(403))

        with (
            patch(_CLIENT_SESSION_TARGET, return_value=_make_session_cm(session_get)),
            pytest.raises(S3StorageError, match="Failed to download"),
        ):
            await provider.download_from_url(url, destination)

        assert full_gets == 1, "a 4xx must not be retried"
        assert not destination.exists()

    def test_retry_log_description_redacts_presigned_url(self) -> None:
        """A retried 5xx is logged as type + status, never as the exception's URL-bearing repr."""
        url = "https://bucket.s3.amazonaws.com/video.mp4?X-Amz-Credential=AKIA%2Fus-west-2&X-Amz-Signature=deadbeef"
        request_info = aiohttp.RequestInfo(URL(url), "GET", CIMultiDictProxy(CIMultiDict()))
        exc = aiohttp.ClientResponseError(request_info, (), status=503, message="Service Unavailable")

        described = _describe_download_error(exc)

        assert described == "ClientResponseError(status=503)"
        # The redaction is load-bearing only because the exception really does carry the signature.
        assert "X-Amz-Signature" in repr(exc) and "X-Amz-Signature" in str(exc)

    def test_retry_log_description_keeps_url_free_detail(self) -> None:
        """Failures that cannot carry a presigned URL keep their message, which is the useful diagnostic."""
        described = _describe_download_error(aiohttp.ClientPayloadError("received 100 of 200 bytes"))

        assert described == "ClientPayloadError: received 100 of 200 bytes"

    @pytest.mark.asyncio
    async def test_download_from_url_retries_5xx(self, tmp_path: Path) -> None:
        """A transient 503 from S3 is retried, unlike a 4xx."""
        config = S3Config(bucket_name="test-bucket", download_retry_backoff_base=0.0)
        provider = S3StorageProvider(config)
        destination = tmp_path / "video.mp4"
        url = "https://bucket.s3.amazonaws.com/video.mp4?X-Amz-Signature=abc"
        full_gets = 0

        def session_get(
            req_url: str, headers: dict[str, str] | None = None, ssl: Any = None, allow_redirects: bool = True
        ) -> AsyncMock:
            if _parse_range(headers) is not None:
                return _make_response_cm(200)
            nonlocal full_gets
            full_gets += 1
            if full_gets == 1:
                return _make_response_cm(503, status_error=_http_error(503))
            return _make_response_cm(200, b"served on retry")

        with patch(_CLIENT_SESSION_TARGET, return_value=_make_session_cm(session_get)):
            await provider.download_from_url(url, destination)

        assert full_gets == 2
        assert destination.read_bytes() == b"served on retry"

    @pytest.mark.asyncio
    async def test_download_from_url_does_not_retry_refused_redirect(self, tmp_path: Path) -> None:
        """The SSRF redirect guard is not a transient failure - it must fail immediately, not burn the budget."""
        config = S3Config(bucket_name="test-bucket", download_retry_backoff_base=0.0)
        provider = S3StorageProvider(config)
        destination = tmp_path / "video.mp4"
        url = "https://bucket.s3.amazonaws.com/video.mp4?X-Amz-Signature=abc"
        full_gets = 0

        def session_get(
            req_url: str, headers: dict[str, str] | None = None, ssl: Any = None, allow_redirects: bool = True
        ) -> AsyncMock:
            if _parse_range(headers) is not None:
                return _make_response_cm(302)
            nonlocal full_gets
            full_gets += 1
            return _make_response_cm(302)

        with (
            patch(_CLIENT_SESSION_TARGET, return_value=_make_session_cm(session_get)),
            pytest.raises(S3StorageError, match="redirect"),
        ):
            await provider.download_from_url(url, destination)

        assert full_gets == 1

    @pytest.mark.asyncio
    async def test_download_from_url_uses_configured_chunk_size(self, tmp_path: Path) -> None:
        """The streaming read uses download_chunk_size, which also sets aiohttp's read high-water mark."""
        config = S3Config(bucket_name="test-bucket", download_chunk_size=4)
        provider = S3StorageProvider(config)
        destination = tmp_path / "f.bin"
        url = "https://bucket.s3.amazonaws.com/f.bin?X-Amz-Signature=abc"
        chunk_sizes: list[int] = []

        def session_get(
            req_url: str, headers: dict[str, str] | None = None, ssl: Any = None, allow_redirects: bool = True
        ) -> AsyncMock:
            if _parse_range(headers) is not None:
                return _make_response_cm(200)
            response = AsyncMock()
            response.status = 200
            response.headers = {}
            response.raise_for_status = MagicMock()

            async def iter_chunked(size: int) -> AsyncIterator[bytes]:
                chunk_sizes.append(size)
                yield b"abcdefgh"

            response.content.iter_chunked = iter_chunked
            return AsyncMock(__aenter__=AsyncMock(return_value=response), __aexit__=AsyncMock(return_value=None))

        with patch(_CLIENT_SESSION_TARGET, return_value=_make_session_cm(session_get)) as mock_session_cls:
            await provider.download_from_url(url, destination)

        assert chunk_sizes == [4]
        # read_bufsize is the half that actually widens the reader's watermarks; iter_chunked alone would leave the
        # parser pausing at aiohttp's 64 KiB default.
        assert mock_session_cls.call_args.kwargs["read_bufsize"] == 4
        assert destination.read_bytes() == b"abcdefgh"

    @pytest.mark.asyncio
    async def test_download_backoff_grows_exponentially(self, tmp_path: Path) -> None:
        """Retry delays follow base * factor ** (attempt - 1)."""
        config = S3Config(
            bucket_name="test-bucket",
            download_max_attempts=3,
            download_retry_backoff_base=0.25,
            download_retry_backoff_factor=3.0,
        )
        provider = S3StorageProvider(config)
        destination = tmp_path / "video.mp4"
        url = "https://bucket.s3.amazonaws.com/video.mp4?X-Amz-Signature=abc"
        delays: list[float] = []

        async def fake_sleep(delay: float) -> None:
            delays.append(delay)

        def session_get(
            req_url: str, headers: dict[str, str] | None = None, ssl: Any = None, allow_redirects: bool = True
        ) -> AsyncMock:
            if _parse_range(headers) is not None:
                return _make_response_cm(200)
            return _make_response_cm(200, b"x", body_error=aiohttp.ClientPayloadError("payload not completed"))

        with (
            patch(_CLIENT_SESSION_TARGET, return_value=_make_session_cm(session_get)),
            patch("services.framework.storage_providers.s3_storage_provider.asyncio.sleep", fake_sleep),
            pytest.raises(S3StorageError),
        ):
            await provider.download_from_url(url, destination)

        assert delays == [0.25, 0.75], "one sleep between each pair of attempts, doubling per the factor"

    @pytest.mark.asyncio
    async def test_download_from_url_multipart_retries_single_part(self, tmp_path: Path) -> None:
        """A transient failure on one ranged part is retried in place, not demoted to a single-stream GET."""
        payload = bytes(range(256)) * 64  # 16 KiB -> 4 parts at a 4 KiB chunk size
        total = len(payload)
        config = S3Config(
            bucket_name="test-bucket",
            multipart_threshold=1024,
            multipart_chunksize=4096,
            download_retry_backoff_base=0.0,
        )
        provider = S3StorageProvider(config)
        destination = tmp_path / "big.bin"
        url = "https://bucket.s3.amazonaws.com/big.bin?X-Amz-Signature=abc"
        flaky_part_attempts = 0
        rangeless_gets = 0

        def session_get(
            req_url: str, headers: dict[str, str] | None = None, ssl: Any = None, allow_redirects: bool = True
        ) -> AsyncMock:
            byte_range = _parse_range(headers)
            if byte_range is None:
                nonlocal rangeless_gets
                rangeless_gets += 1
                return _make_response_cm(200, payload)
            start, end = byte_range
            if (start, end) == (0, 0):
                return _make_response_cm(206, payload[0:1], {"Content-Range": f"bytes 0-0/{total}"})
            content_range = {"Content-Range": f"bytes {start}-{end}/{total}"}
            if (start, end) == (4096, 8191):
                nonlocal flaky_part_attempts
                flaky_part_attempts += 1
                if flaky_part_attempts == 1:
                    payload_error = aiohttp.ClientPayloadError("payload not completed")
                    return _make_response_cm(206, payload[start : start + 16], content_range, body_error=payload_error)
            return _make_response_cm(206, payload[start : end + 1], content_range)

        with patch(_CLIENT_SESSION_TARGET, return_value=_make_session_cm(session_get)):
            result = await provider.download_from_url(url, destination)

        assert result == destination
        assert destination.read_bytes() == payload, "the retried part must overwrite its own partial write"
        assert flaky_part_attempts == 2
        assert rangeless_gets == 0, "a retryable part failure must not demote the object to single-stream"

    @pytest.mark.asyncio
    async def test_download_from_url_multipart_non_206_falls_back_without_retry(self, tmp_path: Path) -> None:
        """A server declining Range is not transient: parts fail once each and the download degrades immediately."""
        payload = bytes(range(256)) * 32  # 8 KiB -> 2 parts at a 4 KiB chunk size
        total = len(payload)
        config = S3Config(
            bucket_name="test-bucket",
            multipart_threshold=1024,
            multipart_chunksize=4096,
            download_retry_backoff_base=0.0,
            download_max_attempts=3,
        )
        provider = S3StorageProvider(config)
        destination = tmp_path / "big.bin"
        url = "https://bucket.s3.amazonaws.com/big.bin?X-Amz-Signature=abc"
        part_gets = 0
        rangeless_gets = 0

        def session_get(
            req_url: str, headers: dict[str, str] | None = None, ssl: Any = None, allow_redirects: bool = True
        ) -> AsyncMock:
            byte_range = _parse_range(headers)
            if byte_range is None:
                nonlocal rangeless_gets
                rangeless_gets += 1
                return _make_response_cm(200, payload)
            if byte_range == (0, 0):
                return _make_response_cm(206, payload[0:1], {"Content-Range": f"bytes 0-0/{total}"})
            nonlocal part_gets
            part_gets += 1
            return _make_response_cm(200, payload)  # range silently ignored by the server

        with patch(_CLIENT_SESSION_TARGET, return_value=_make_session_cm(session_get)):
            result = await provider.download_from_url(url, destination)

        assert result == destination
        assert destination.read_bytes() == payload
        assert part_gets <= 2, "a declined Range must not be retried per part (2 parts, 1 attempt each)"
        assert rangeless_gets == 1, "the download must fall back to a single streaming GET"

    @pytest.mark.asyncio
    async def test_download_from_url_raw_s3(
        self, provider: S3StorageProvider, mock_client: AsyncMock, tmp_path: Path
    ) -> None:
        """Tests downloading from a raw S3 URL (generates presigned first)."""
        provider._client = mock_client
        destination = tmp_path / "downloaded.mp4"
        raw_url = "s3://test-bucket/path/file.mp4"

        mock_client.generate_presigned_url = AsyncMock(return_value="https://presigned.url/path/file.mp4?sig=abc")

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.raise_for_status = MagicMock()

        async def mock_iter_chunked(size: int) -> AsyncIterator[bytes]:
            yield b"data"

        mock_response.content.iter_chunked = mock_iter_chunked

        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=None),
            )
        )

        mock_session_cm = AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=None),
        )

        with patch(
            "services.framework.storage_providers.s3_storage_provider.aiohttp.ClientSession",
            return_value=mock_session_cm,
        ):
            result = await provider.download_from_url(raw_url, destination)

        assert result == destination
        mock_client.generate_presigned_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_from_url_rejects_redirect(self, provider: S3StorageProvider, tmp_path: Path) -> None:
        """A 3xx on the download is refused, not followed — a redirect can't bypass the SSRF check (TOCTOU guard)."""
        destination = tmp_path / "redirected.mp4"
        url = "https://bucket.s3.amazonaws.com/file.mp4?X-Amz-Signature=abc"

        mock_response = AsyncMock()
        mock_response.status = 302  # a redirect to a (potentially private) target
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()
        get_cm = AsyncMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock(return_value=None))
        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=get_cm)
        session_cm = AsyncMock(__aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock(return_value=None))

        with (
            patch(
                "services.framework.storage_providers.s3_storage_provider.aiohttp.ClientSession",
                return_value=session_cm,
            ),
            pytest.raises(S3StorageError, match="redirect"),
        ):
            await provider.download_from_url(url, destination)

        # The GET was issued with redirects disabled, and the refused 3xx left no file behind.
        assert mock_session.get.call_args.kwargs.get("allow_redirects") is False
        assert not destination.exists()

    @pytest.mark.asyncio
    async def test_download_from_url_local_absolute_path(
        self, provider: S3StorageProvider, mock_client: AsyncMock, tmp_path: Path
    ) -> None:
        """Local absolute paths bypass boto3/aiohttp and are symlinked into destination."""
        provider._client = mock_client
        mock_client.generate_presigned_url = AsyncMock()
        src = tmp_path / "src.bin"
        src.write_bytes(b"local-path-payload")
        destination = tmp_path / "out" / "downloaded.bin"

        with patch(
            "services.framework.storage_providers.s3_storage_provider.aiohttp.ClientSession"
        ) as mock_session_cls:
            result = await provider.download_from_url(str(src), destination)

        assert result == destination
        assert destination.is_symlink()
        assert destination.read_bytes() == b"local-path-payload"
        mock_session_cls.assert_not_called()
        mock_client.generate_presigned_url.assert_not_called()

    @pytest.mark.asyncio
    async def test_download_from_url_file_uri(
        self, provider: S3StorageProvider, mock_client: AsyncMock, tmp_path: Path
    ) -> None:
        """``file://`` URIs are treated like local paths - no network calls."""
        provider._client = mock_client
        mock_client.generate_presigned_url = AsyncMock()
        src = tmp_path / "src.bin"
        src.write_bytes(b"file-uri-payload")
        destination = tmp_path / "downloaded.bin"

        with patch(
            "services.framework.storage_providers.s3_storage_provider.aiohttp.ClientSession"
        ) as mock_session_cls:
            result = await provider.download_from_url(f"file://{src}", destination)

        assert result == destination
        assert destination.is_symlink()
        assert destination.read_bytes() == b"file-uri-payload"
        mock_session_cls.assert_not_called()
        mock_client.generate_presigned_url.assert_not_called()

    @pytest.mark.asyncio
    async def test_download_from_url_local_missing_source_raises(
        self, provider: S3StorageProvider, tmp_path: Path
    ) -> None:
        """A local URL pointing at a missing file raises FileNotFoundError, not S3StorageError."""
        destination = tmp_path / "downloaded.bin"

        with pytest.raises(FileNotFoundError):
            await provider.download_from_url(str(tmp_path / "does-not-exist.bin"), destination)

    def test_from_settings(self) -> None:
        """Tests factory method from_settings, which reads the region from the configured output storage URL."""
        mock_settings = MagicMock()
        mock_settings.s3_credentials = S3Credentials("test-access-key", "test-secret-key", "test-session-token")
        mock_settings.output_storage_type = "s3"
        mock_settings.output_storage_url = ""
        mock_settings.storage_url = "https://other-bucket.s3.us-west-2.amazonaws.com"

        with patch("services.framework.storage_providers.s3_storage_provider.SettingsBase") as mock_cls:
            mock_cls.return_value = mock_settings
            mock_cls.get_env_files.return_value = []
            provider = S3StorageProvider.from_settings(bucket_name="my-bucket")

        # The caller's bucket wins over the one the storage URL names; only the region is taken from the URL.
        assert provider.config.bucket_name == "my-bucket"
        assert provider.config.region_name == "us-west-2"
        # Without the session token a signature minted from assumed-role credentials is rejected by S3.
        assert provider.config.aws_session_token is not None
        assert provider.config.aws_session_token.get_secret_value() == "test-session-token"

    def test_from_settings_default_region(self) -> None:
        """Tests from_settings with no storage URL falls back to us-east-1."""
        mock_settings = MagicMock()
        mock_settings.s3_credentials = S3Credentials("test-access-key", "test-secret-key", None)
        mock_settings.output_storage_type = "s3"
        mock_settings.output_storage_url = ""
        mock_settings.storage_url = ""

        with patch("services.framework.storage_providers.s3_storage_provider.SettingsBase") as mock_cls:
            mock_cls.return_value = mock_settings
            mock_cls.get_env_files.return_value = []
            provider = S3StorageProvider.from_settings(bucket_name="my-bucket")

        assert provider.config.bucket_name == "my-bucket"
        assert provider.config.region_name == "us-east-1"
        assert provider.config.aws_session_token is None

    def test_from_settings_ignores_non_s3_storage_url(self) -> None:
        """An Azure output URL names no AWS region, so the fallback applies rather than a parse failure."""
        mock_settings = MagicMock()
        mock_settings.s3_credentials = S3Credentials(None, "account-key", None)
        mock_settings.output_storage_type = "azure"
        mock_settings.output_storage_url = "https://myacct.blob.core.windows.net/my-container"
        mock_settings.storage_url = ""

        with patch("services.framework.storage_providers.s3_storage_provider.SettingsBase") as mock_cls:
            mock_cls.return_value = mock_settings
            mock_cls.get_env_files.return_value = []
            provider = S3StorageProvider.from_settings(bucket_name="my-bucket")

        assert provider.config.region_name == "us-east-1"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
