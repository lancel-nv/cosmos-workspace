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

"""Factories for creating directional StorageProvider instances from settings."""

from __future__ import annotations

from dataclasses import replace

from services.framework.protocols.storage_provider import StorageProvider
from services.framework.storage_providers.azure_blob_storage_provider import AzureBlobStorageProvider
from services.framework.storage_providers.config import AzureBlobConfig, S3Config
from services.framework.storage_providers.local_storage_provider import LocalStorageProvider
from services.framework.storage_providers.s3_storage_provider import S3StorageProvider
from services.settings_base import SettingsBase
from utils.s3 import load_s3_cluster_credentials
from utils.storage_url import StorageTarget, parse_storage_url

# S3-compatible clusters (e.g. PBSS) reject v4 signing on HeadObject; v2 ('s3') works for
# both those clusters and AWS S3, matching how utils.s3 accesses them.
_CLUSTER_SIGNATURE_VERSION = "s3"

# Region used when nothing else names one: a custom S3-compatible endpoint, which is not region-scoped but still
# needs a value to sign with. Matches S3Config's own default.
_DEFAULT_S3_REGION = "us-east-1"


def _trusted_internal_hosts(settings: SettingsBase) -> frozenset[str]:
    """Parse the comma-separated ``COSMOS_EVALUATOR_STORAGE_TRUSTED_INTERNAL_HOSTS`` into exact hostnames.

    These are the hosts a cloud provider's ``download_from_url`` will allow to resolve to private/reserved IPs
    (the SSRF allowlist for internal clusters like ``pbss.s8k.io``, or a private Azure blob endpoint). Empty when the
    env var is unset.
    """
    return frozenset(host.strip() for host in settings.storage_trusted_internal_hosts.split(",") if host.strip())


def _build(settings: SettingsBase, storage_type: str, storage_url: str, key_prefix: str) -> StorageProvider:
    """Builds a StorageProvider for the given resolved storage type and URL.

    The bucket or container, the Azure account, the endpoint, and the AWS region all come from ``storage_url``;
    credentials come from the shared ``settings.storage_*_key`` fields.

    Args:
        settings: Application settings supplying the credentials and the SSRF allowlist.
        storage_type: Resolved storage backend (``"local"``, ``"s3"`` or ``"azure"``).
        storage_url: Resolved storage URL for this direction. Unused by local storage.
        key_prefix: Prefix prepended to all storage keys (cloud) or used as the base directory (local).

    Returns:
        A configured StorageProvider instance.

    Raises:
        ValueError: If ``storage_type`` is not supported, if ``storage_url`` is unset or does not address that
            backend, if an S3 URL names neither a region nor a custom endpoint, or if Azure is requested without a
            credential.
    """
    if storage_type == "local":
        return LocalStorageProvider(base_path=key_prefix)

    target = parse_storage_url(storage_url, storage_type)

    if storage_type == "s3":
        # Talking to AWS without a region leaves boto3 signing against us-east-1, which every bucket hosted
        # elsewhere rejects at request time. Demand the region up front, where the operator can act on it. A custom
        # endpoint is exempt: S3-compatible clusters are not region-scoped and accept the default.
        if not target.region and not target.endpoint_url:
            raise ValueError(
                f"S3 storage URL must name a region (e.g. https://<bucket>.s3.us-west-2.amazonaws.com): {storage_url!r}"
            )
        credentials = settings.s3_credentials
        config = S3Config(
            bucket_name=target.container,
            region_name=target.region or _DEFAULT_S3_REGION,
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.session_token,
            endpoint_url=target.endpoint_url,
            key_prefix=key_prefix,
            trusted_internal_hosts=_trusted_internal_hosts(settings),
        )
        return S3StorageProvider(config=config)

    if storage_type == "azure":
        # Azure reuses the shared credential fields: ``storage_secret_key`` carries the account key and
        # ``storage_session_token`` a pre-minted SAS. Neither field is read from an ``AWS_*`` variable, so there is
        # no ambient AWS secret for an Azure deployment to pick up by accident.
        if not target.account:
            raise ValueError(f"Azure storage URL names no account: {storage_url!r}")
        if not settings.storage_secret_key and not settings.storage_session_token:
            raise ValueError(
                "COSMOS_EVALUATOR_STORAGE_SECRET_KEY (account key) or COSMOS_EVALUATOR_STORAGE_SESSION_TOKEN "
                "(SAS token) must be set for Azure storage"
            )
        azure_config = AzureBlobConfig(
            container_name=target.container,
            account_name=target.account,
            account_key=settings.storage_secret_key,
            sas_token=settings.storage_session_token,
            endpoint_url=target.endpoint_url,
            key_prefix=key_prefix,
            trusted_internal_hosts=_trusted_internal_hosts(settings),
        )
        return AzureBlobStorageProvider(config=azure_config)

    raise ValueError(f"Unsupported storage type: {storage_type}")


def _cluster_target(storage_type: str, storage_url: str, bucket: str | None) -> StorageTarget:
    """Resolves the target for a named-cluster provider from the storage URL, with ``bucket`` replacing its container.

    The override replaces only the container, so an endpoint or region the storage URL names still reaches the
    provider.

    Raises:
        ValueError: If neither is available.
    """
    if not storage_url:
        if not bucket:
            raise ValueError(
                "A bucket is required for cluster storage (pass bucket= or set COSMOS_EVALUATOR_STORAGE_URL)"
            )
        return StorageTarget(container=bucket)
    target = parse_storage_url(storage_url, storage_type)
    return replace(target, container=bucket) if bucket else target


def _build_cluster(settings: SettingsBase, cluster: str, target: StorageTarget, key_prefix: str) -> StorageProvider:
    """Builds an S3 provider for a named cluster (e.g. ``"pbss"``).

    Region / endpoint prefer what the storage URL names and otherwise fall back to the cluster-prefixed environment
    variables (e.g. ``PBSS_AWS_*``, themselves falling back to plain ``AWS_*``) via
    :func:`utils.s3.load_s3_cluster_credentials`; credentials prefer the explicit ``settings.storage_*_key`` values
    over the same cluster variables. This lets a deployment override the cluster's defaults through the standard
    storage settings without those settings being silently ignored, while still reaching the cluster out of the box
    when only the cluster-prefixed vars are present. v2 signing is pinned, which these clusters require.

    Args:
        settings: Application settings supplying the optional credential overrides.
        cluster: Cluster name (e.g. ``"pbss"``, ``"pdx"``) selecting the credential set.
        target: Addressing for the bucket on that cluster.
        key_prefix: Prefix prepended to all storage keys.

    Returns:
        A configured S3 StorageProvider for the cluster.
    """
    creds = load_s3_cluster_credentials(cluster)
    config = S3Config(
        bucket_name=target.container,
        region_name=target.region or creds["region"] or _DEFAULT_S3_REGION,
        aws_access_key_id=settings.storage_access_key or creds["access_key_id"] or None,
        aws_secret_access_key=settings.storage_secret_key or creds["secret_access_key"] or None,
        endpoint_url=target.endpoint_url or creds["endpoint_url"] or None,
        signature_version=_CLUSTER_SIGNATURE_VERSION,
        key_prefix=key_prefix,
        trusted_internal_hosts=_trusted_internal_hosts(settings),
    )
    return S3StorageProvider(config=config)


def build_input_storage_provider(
    settings: SettingsBase, key_prefix: str = "", *, cluster: str | None = None, bucket: str | None = None
) -> StorageProvider:
    """Builds the storage provider used for downloading request inputs.

    Dispatches on ``settings.input_storage_type``, which itself resolves
    via the alias chain ``COSMOS_EVALUATOR_INPUT_STORAGE_TYPE`` ->
    ``COSMOS_EVALUATOR_STORAGE_TYPE`` -> default ``"s3"``. The target is addressed by
    ``settings.input_storage_url``, falling back to ``settings.storage_url``.

    When ``cluster`` is given, an S3 provider is built for that named cluster (v2 signing); the
    credentials prefer the explicit ``settings.storage_*_key`` values and fall back to the
    cluster-prefixed env vars (see :func:`_build_cluster`). ``bucket`` then selects the target
    bucket on that cluster, overriding the one the storage URL names.

    Args:
        settings: Application settings
        key_prefix: Key prefix prepended to all storage keys (cloud) or used
            as the base directory (local). Default empty so the same provider
            instance can be reused across requests with the per-request prefix
            baked into the key passed to ``store_file``.
        cluster: Optional named S3 cluster to read from (e.g. ``"pbss"``).
        bucket: Optional bucket override (required when no input storage URL is set).

    Returns:
        A configured StorageProvider instance
    """
    storage_url = settings.input_storage_url or settings.storage_url
    if cluster:
        return _build_cluster(
            settings, cluster, _cluster_target(settings.input_storage_type, storage_url, bucket), key_prefix
        )
    return _build(settings, settings.input_storage_type, storage_url, key_prefix)


def build_output_storage_provider(
    settings: SettingsBase, key_prefix: str = "", *, cluster: str | None = None, bucket: str | None = None
) -> StorageProvider:
    """Builds the storage provider used for uploading checker outputs.

    Dispatches on ``settings.output_storage_type``, which itself resolves
    via the alias chain ``COSMOS_EVALUATOR_OUTPUT_STORAGE_TYPE`` ->
    ``COSMOS_EVALUATOR_STORAGE_TYPE`` -> default ``"s3"``. The target is addressed by
    ``settings.output_storage_url``, falling back to ``settings.storage_url``.

    When ``cluster`` is given, an S3 provider is built for that named cluster (see
    :func:`build_input_storage_provider`).

    Args:
        settings: Application settings
        key_prefix: Key prefix prepended to all storage keys (cloud) or used
            as the base directory (local). Default empty so the same provider
            instance can be reused across requests with the per-request prefix
            baked into the key passed to ``store_file``.
        cluster: Optional named S3 cluster to write to (e.g. ``"pbss"``).
        bucket: Optional bucket override (required when no output storage URL is set).

    Returns:
        A configured StorageProvider instance
    """
    storage_url = settings.output_storage_url or settings.storage_url
    if cluster:
        return _build_cluster(
            settings, cluster, _cluster_target(settings.output_storage_type, storage_url, bucket), key_prefix
        )
    return _build(settings, settings.output_storage_type, storage_url, key_prefix)
