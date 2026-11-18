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

from typing import Literal, Self

from pydantic import BaseModel, Field, SecretStr, model_validator


class S3Config(BaseModel):
    """Configuration for S3 storage handler."""

    # Required settings
    bucket_name: str = Field(..., description="S3 bucket name")

    # AWS credentials (optional if using IAM roles or env vars)
    aws_access_key_id: SecretStr | None = Field(None, description="AWS access key ID")
    aws_secret_access_key: SecretStr | None = Field(None, description="AWS secret access key")
    aws_session_token: SecretStr | None = Field(None, description="AWS session token for temporary credentials")

    # Connection settings
    region_name: str = Field("us-east-1", description="AWS region")
    endpoint_url: str | None = Field(None, description="Custom S3 endpoint URL (for S3-compatible services)")
    signature_version: str | None = Field(
        None,
        description=(
            "Botocore signature version (e.g. 's3' for v2, 's3v4' for v4). None uses the boto default (v4). "
            "Some S3-compatible clusters (e.g. PBSS) reject v4 on HeadObject and require v2 ('s3')."
        ),
    )

    # Key prefix (prepended to all keys in storage operations)
    key_prefix: str = Field("", description="Prefix prepended to all storage keys")

    # Behavior settings
    use_ssl: bool = Field(True, description="Use SSL for connections")
    verify_ssl: bool = Field(True, description="Verify SSL certificates")
    trusted_internal_hosts: frozenset[str] = Field(
        default_factory=frozenset,
        description=(
            "Exact hostnames allowed to resolve to private/reserved IPs when fetching presigned/HTTPS URLs in "
            "download_from_url (SSRF allowlist for internal S3-compatible clusters, e.g. 'pbss.s8k.io'). Empty "
            "disables the exemption."
        ),
    )

    # Upload settings
    multipart_threshold: int = Field(64 * 1024 * 1024, description="Threshold for multipart uploads (64MB)")
    multipart_chunksize: int = Field(16 * 1024 * 1024, description="Chunk size for multipart uploads (16MB)")
    max_concurrency: int = Field(10, description="Maximum concurrent uploads/downloads")

    # HTTP download settings (presigned / https URL fetches in download_from_url)
    download_sock_connect_timeout: float = Field(
        30.0, description="Socket connect timeout (s) for HTTP(S) presigned-URL downloads"
    )
    download_sock_read_timeout: float = Field(
        300.0,
        description=(
            "Per-read inactivity timeout (s) for HTTP(S) downloads. Replaces aiohttp's default 300s *total* cap so a "
            "slow-but-progressing transfer of a large object is not killed mid-flight; only a genuinely stalled "
            "connection (no bytes received within this window) fails."
        ),
    )
    # gt/ge bounds below are load-bearing, not decorative: a zero chunk size makes iter_chunked yield empty bytes
    # forever, and a zero attempt budget makes _transfer_with_retry skip the transfer and report success.
    download_chunk_size: int = Field(
        1 << 20,
        gt=0,
        description=(
            "Chunk size (bytes) for streaming HTTP(S) downloads to disk. Also raises aiohttp's StreamReader "
            "high-water mark to twice this value, so a larger chunk both cuts per-chunk thread hops and keeps the "
            "parser from being paused for most of the transfer. Worst-case buffering is max_concurrency * 2 * this."
        ),
    )
    download_max_attempts: int = Field(
        3, ge=1, description="Total attempts (initial + retries) per HTTP(S) download transfer"
    )
    download_retry_backoff_base: float = Field(
        0.5, ge=0.0, description="Initial retry delay (s) for HTTP(S) download retries"
    )
    download_retry_backoff_factor: float = Field(
        2.0, ge=0.0, description="Multiplier applied to the retry delay after each failed HTTP(S) download attempt"
    )

    # Metadata settings
    default_content_type: str = Field("application/octet-stream", description="Default content type")
    server_side_encryption: str | None = Field(None, description="Server-side encryption method (AES256, aws:kms)")
    kms_key_id: str | None = Field(None, description="KMS key ID for encryption")

    # Retry settings
    max_attempts: int = Field(3, description="Maximum retry attempts")
    retry_mode: Literal["legacy", "standard", "adaptive"] = Field(
        "adaptive", description="Retry mode (legacy, standard, adaptive)"
    )

    # Storage class
    storage_class: str = Field("STANDARD", description="S3 storage class")

    # Lifecycle settings
    enable_versioning: bool = Field(False, description="Enable S3 versioning")

    class Config:
        """Pydantic configuration."""

        extra = "forbid"  # Don't allow extra fields

    def get_boto3_config(self) -> dict:
        """Gets configuration for boto3 client."""
        config = {
            "region_name": self.region_name,
            "use_ssl": self.use_ssl,
            "verify": self.verify_ssl,
        }

        if self.endpoint_url:
            config["endpoint_url"] = self.endpoint_url

        # Include credentials only if we have both key and secret
        if self.aws_access_key_id and self.aws_secret_access_key:
            config["aws_access_key_id"] = self.aws_access_key_id.get_secret_value()
            config["aws_secret_access_key"] = self.aws_secret_access_key.get_secret_value()

            if self.aws_session_token:
                config["aws_session_token"] = self.aws_session_token.get_secret_value()

        return config

    def get_transfer_config(self) -> dict:
        """Gets configuration for S3 transfer manager."""
        return {
            "multipart_threshold": self.multipart_threshold,
            "multipart_chunksize": self.multipart_chunksize,
            "max_concurrency": self.max_concurrency,
            "use_threads": True,
        }


class AzureBlobConfig(BaseModel):
    """Configuration for the Azure Blob Storage handler."""

    # Required settings. ``container_name`` is Azure's equivalent of an S3 bucket and is named by the same path
    # position in the storage URL.
    container_name: str = Field(..., description="Azure Blob container name")
    account_name: str = Field(..., description="Azure Storage account name")

    # Credentials: exactly one of these is enough, and ``account_key`` is preferred because only it can mint fresh
    # SAS URLs with a caller-chosen expiry. A ``sas_token`` alone still authenticates every data-plane call, but
    # presigned URLs then reuse that token and inherit its fixed expiry.
    account_key: SecretStr | None = Field(None, description="Azure Storage account key")
    sas_token: SecretStr | None = Field(None, description="Pre-minted account or container SAS token")

    # Connection settings
    endpoint_url: str | None = Field(
        None,
        description=(
            "Custom blob endpoint origin (e.g. 'http://127.0.0.1:10000' for Azurite, or a private-link host). The "
            "account name is appended as the first path segment. None uses https://<account>.blob.core.windows.net."
        ),
    )

    # Key prefix (prepended to all keys in storage operations)
    key_prefix: str = Field("", description="Prefix prepended to all storage keys")

    # Transfer settings. The Azure SDK splits large blobs into chunks internally and fans them out up to this many
    # at a time, which is why this provider needs no equivalent of the S3 provider's hand-rolled ranged downloader.
    max_concurrency: int = Field(10, gt=0, description="Maximum concurrent chunks per upload/download")

    trusted_internal_hosts: frozenset[str] = Field(
        default_factory=frozenset,
        description=(
            "Exact hostnames allowed to resolve to private/reserved IPs when fetching blob URLs (SSRF allowlist for "
            "emulators and private endpoints, e.g. '127.0.0.1'). Empty disables the exemption."
        ),
    )

    # Metadata settings
    default_content_type: str = Field("application/octet-stream", description="Default content type")

    class Config:
        """Pydantic configuration."""

        extra = "forbid"  # Don't allow extra fields

    @model_validator(mode="after")
    def _require_a_credential(self) -> Self:
        """Rejects a config with no way to authenticate.

        Anonymous access would only ever work for a public container, and failing at construction gives a far clearer
        error than the 403 every subsequent operation would otherwise return.
        """
        if not self.account_key and not self.sas_token:
            raise ValueError("Either account_key or sas_token is required for Azure Blob Storage")
        return self

    def get_credential(self) -> str:
        """Gets the credential to hand the Azure SDK client.

        ``BlobServiceClient`` accepts either an account key or a SAS token as a plain string in this position, so one
        accessor covers both. The account key wins when both are set, since it additionally enables SAS minting.
        """
        if self.account_key:
            return self.account_key.get_secret_value()
        if self.sas_token:
            return self.sas_token.get_secret_value()
        raise ValueError("Azure config carries no credential; _require_a_credential should have rejected it")
