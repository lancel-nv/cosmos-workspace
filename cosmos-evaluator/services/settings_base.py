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

"""
Configuration module for the services.

This module provides environment-specific configuration management.
"""

from dataclasses import dataclass
from enum import Enum
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from checks.utils.video_codec import (
    DEFAULT_VIDEO_CODEC,
    DEFAULT_VIDEO_QUALITY,
    VIDEO_CODEC_ENV_VAR,
    VIDEO_QUALITY_ENV_VAR,
    VideoCodec,
    VideoQuality,
)
from utils.environment import Environment
from utils.storage_url import StorageTarget, parse_storage_url


@dataclass(frozen=True)
class S3Credentials:
    """One S3 credential set. Any field may be ``None``, leaving boto3 to supply it from its default chain."""

    access_key: str | None
    secret_key: str | None
    session_token: str | None


class LogLevel(str, Enum):
    """Log level enumeration (uppercase to match logging module)."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# Shared log line format for every REST service: timestamp, level, logger name, message.
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

# Fields held back from the per-service alias injection in ``__pydantic_init_subclass__``, so they answer only to
# their global ``COSMOS_EVALUATOR_*`` variable.
#
# The two video fields qualify because settings are not the only reader. ``checks.utils.video`` resolves them straight
# from the environment (it cannot import pydantic-settings; see ``checks/utils/video_codec.py``) and knows only the
# global names. Injecting ``HALLUCINATION_VIDEO_CODEC`` would therefore create a variable that ``Settings`` accepts and
# reports faithfully while the encode ladder ignores it -- two sources of truth that disagree silently, which is worse
# than not offering the per-service knob at all. If a service ever genuinely needs its own codec, the fix is to thread
# the resolved value into the ladder rather than to delete this entry.
GLOBAL_ONLY_FIELDS: frozenset[str] = frozenset({"video_codec", "video_quality"})


class UvicornErrorRenameFilter(logging.Filter):
    """Relabel uvicorn's misleadingly-named ``uvicorn.error`` logger to ``uvicorn`` for display.

    Uvicorn routes *all* of its lifecycle records (startup, shutdown, "Uvicorn running on ...") through a logger named
    ``uvicorn.error`` regardless of severity. With ``%(name)s`` in ``LOG_FORMAT`` this renders ordinary INFO startup
    lines as ``... - INFO - uvicorn.error - ...``, which reads like an error. This filter rewrites the record's logger
    name to ``uvicorn`` so the label matches the ``INFO`` level; it never drops records (always returns ``True``) and
    leaves genuine error-level records untouched apart from the cosmetic name change.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == "uvicorn.error":
            record.name = "uvicorn"
        return True


def build_log_config(log_level: str) -> dict[str, Any]:
    """Build a uvicorn ``log_config`` dict that configures the *root* logger.

    ``uvicorn.run()`` applies this dictConfig in-process and, crucially, inside *each worker process* when
    ``workers > 1``. Without a configured root logger, application and third-party dependency records fall through
    to ``logging.lastResort`` (WARNING) and are dropped — only uvicorn's own loggers print. Defining the root here
    surfaces every module's logs at ``log_level`` regardless of worker count, and uvicorn's own loggers propagate
    to the same handler. ``disable_existing_loggers=False`` keeps loggers created at import time (before uvicorn
    applies this) alive. Pass via ``uvicorn.run(..., log_config=build_log_config(args.log_level))``.
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"uvicorn_error_rename": {"()": f"{__name__}.UvicornErrorRenameFilter"}},
        "formatters": {"default": {"format": LOG_FORMAT}},
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "filters": ["uvicorn_error_rename"],
            }
        },
        "root": {"handlers": ["default"], "level": log_level},
    }


class SettingsBase(BaseSettings):
    """Base settings for the services."""

    model_config = SettingsConfigDict(
        strict=True, extra="ignore", env_prefix="COSMOS_EVALUATOR_", case_sensitive=False, frozen=True
    )

    # Use validation_alias to ensure all services read from COSMOS_EVALUATOR_ENV regardless of the child class's
    # env_prefix
    env: Environment = Field(
        default=Environment.LOCAL,
        validation_alias=AliasChoices("COSMOS_EVALUATOR_ENV", "env"),
    )

    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        validation_alias=AliasChoices("COSMOS_EVALUATOR_LOG_LEVEL", "log_level"),
    )

    # Output codec for every re-encode the checkers perform (visualization clips, trimmed and concatenated artifacts).
    # This is a request for a codec, not for an encoder: the host decides which encoder implements it, so pinning this
    # does not by itself make output byte-identical across the fleet. ``h264`` is the default and yields the
    # ``h264_nvenc -> av1_nvenc -> libsvtav1`` ladder, since an exhausted ladder falls through to AV1's.
    #
    # Global-only, unlike most fields here: it is in ``GLOBAL_ONLY_FIELDS``, so there is deliberately no
    # ``<SERVICE>_VIDEO_CODEC`` override. See that constant for why.
    video_codec: VideoCodec = Field(
        default=DEFAULT_VIDEO_CODEC,
        validation_alias=AliasChoices(VIDEO_CODEC_ENV_VAR, "video_codec"),
    )

    # Quality tier for those same re-encodes, translated per encoder (``av1_nvenc`` takes ``-cq``, ``libsvtav1`` takes
    # ``-crf``, and the two scales are not comparable, which is why this is a named tier rather than a number). Also
    # global-only, for the same reason as ``video_codec`` above.
    #
    # ``highest`` requests each encoder's best available mode. That is mathematically lossless for only two of the four
    # encoders, and this asymmetry is the most important thing about the tier:
    #
    #   | Encoder      | ``highest``            | Bit-exact                                                      |
    #   | ------------ | ---------------------- | -------------------------------------------------------------- |
    #   | h264_nvenc   | ``-tune lossless``     | Yes -- verified as 0 differing bytes through an h264_cuvid     |
    #   |              |                        | round trip                                                     |
    #   | libvpx-vp9   | ``-lossless 1``        | Yes -- lossless by construction                                |
    #   | av1_nvenc    | (no ``highest`` tier)  | n/a -- see below                                               |
    #   | libsvtav1    | ``-preset 2 -crf 1``   | No -- and ``lossless=1`` misses by 5-17 chroma bytes/10 frames |
    #
    # The ``libsvtav1`` row asks for ``-crf 1`` rather than ``-crf 0`` on purpose: ffmpeg's wrapper reads 0 as "unset"
    # and leaves SVT at its own default of 35, which is coarser than ``standard``. See ``_QUALITY_ARGS`` in
    # ``checks/utils/video.py`` for the measurements behind both that and the preset.
    #
    # ``av1_nvenc`` has no ``highest`` row because NVENC's lossless tune is an H.264/HEVC feature that Ada's AV1
    # encoder does not implement, and asking for it fails the encode outright rather than degrading. An AV1 request at
    # ``highest`` therefore skips hardware and encodes on ``libsvtav1``, on Ada as everywhere else -- slower, but the
    # tier then means one thing fleet-wide instead of varying by card.
    #
    # No runtime lossless AV1 encoder exists in the shipped build, so anything that genuinely requires bit-exact output
    # must also pin ``video_codec`` to ``h264`` or ``vp9``. The tier is called "highest" rather than "lossless" for
    # exactly this reason.
    video_quality: VideoQuality = Field(
        default=DEFAULT_VIDEO_QUALITY,
        validation_alias=AliasChoices(VIDEO_QUALITY_ENV_VAR, "video_quality"),
    )

    # Optional per-environment API keys. When the key for the active environment is set, REST apps that install the
    # framework auth middleware require a matching ``X-API-Key`` header; when unset, those endpoints stay open. Kept
    # separate from the Database REST API's RBAC keys (DATABASE_READ_KEY / DATABASE_ADMIN_KEY), which use their own
    # bespoke per-route enforcement.
    staging_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("COSMOS_EVALUATOR_STAGING_API_KEY", "staging_api_key"),
    )

    prod_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("COSMOS_EVALUATOR_PROD_API_KEY", "prod_api_key"),
    )

    # Storage configuration (cloud-agnostic)
    #
    # Each checker service builds two storage providers: one for downloading request inputs and one for uploading
    # checker outputs. ``input_storage_type``and ``output_storage_type`` pick the backend for each direction
    # independently and fall back to ``COSMOS_EVALUATOR_STORAGE_TYPE`` (when set) so a single-mode deployment names
    # the backend once.
    #
    # The backend is configured rather than inferred from the storage URL: only the public cloud hostnames are
    # self-identifying, and a custom endpoint such as ``https://pbss.s8k.io/my-bucket`` is indistinguishable from a
    # private Azure endpoint, so inferring it would silently address the wrong cloud.
    input_storage_type: str = Field(
        default="s3",
        validation_alias=AliasChoices(
            "COSMOS_EVALUATOR_INPUT_STORAGE_TYPE", "COSMOS_EVALUATOR_STORAGE_TYPE", "input_storage_type"
        ),
    )

    output_storage_type: str = Field(
        default="s3",
        validation_alias=AliasChoices(
            "COSMOS_EVALUATOR_OUTPUT_STORAGE_TYPE", "COSMOS_EVALUATOR_STORAGE_TYPE", "output_storage_type"
        ),
    )

    # One URL naming the target collection and how to reach it, parsed by :func:`utils.storage_url.parse_storage_url`
    # into the bucket or container, the Azure storage account, a non-default endpoint origin, and the AWS region.
    # Addressing a target once keeps those components consistent by construction, where separate variables can
    # disagree — a region that contradicts the endpoint's, or a container on the wrong account.
    #
    # S3 accepts ``s3://my-bucket`` (leaving the region to the SDK or the cluster credentials), an AWS host in either
    # addressing style, or ``https://<endpoint>/<bucket>`` for an S3-compatible cluster. Azure accepts
    # ``https://<account>.blob.core.windows.net/<container>`` or ``https://<endpoint>/<account>/<container>``. See
    # that function for the exact grammar. Unused by local storage, which resolves keys under a base directory.
    storage_url: str = Field(
        default="",
        validation_alias=AliasChoices("COSMOS_EVALUATOR_STORAGE_URL", "storage_url"),
    )

    # Per-direction overrides, each falling back to ``storage_url``. Required only when the two directions address
    # different targets: a deployment reading inputs from S3 and writing outputs to Azure has no single URL that can
    # serve both.
    input_storage_url: str = Field(
        default="",
        validation_alias=AliasChoices("COSMOS_EVALUATOR_INPUT_STORAGE_URL", "input_storage_url"),
    )

    output_storage_url: str = Field(
        default="",
        validation_alias=AliasChoices("COSMOS_EVALUATOR_OUTPUT_STORAGE_URL", "output_storage_url"),
    )

    # Storage credentials, shared across backends so a deployment supplies one credential set whichever cloud it
    # addresses. Each is read only under its own ``COSMOS_EVALUATOR_`` name: reading the ambient ``AWS_*`` / ``AZURE_*``
    # variables here would let credentials exported for one target be signed into requests for another — an
    # ``AWS_SECRET_ACCESS_KEY`` would become the Azure account key. S3 reads the ``AWS_*`` names separately through
    # :attr:`s3_credentials`.
    storage_access_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("COSMOS_EVALUATOR_STORAGE_ACCESS_KEY", "storage_access_key"),
    )

    # The S3 secret access key, or the Azure storage account key.
    storage_secret_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("COSMOS_EVALUATOR_STORAGE_SECRET_KEY", "storage_secret_key"),
    )

    # The S3 session token, or an Azure SAS token used in place of an account key.
    #
    # Temporary (STS) credentials carry a session token, and S3 rejects every signature made with an assumed-role key
    # that omits it. On Azure, a pre-minted account or container SAS leaves the provider unable to mint fresh SAS
    # URLs, so it appends this token instead and the expiry is whatever the token carries rather than the caller's
    # requested lifetime.
    storage_session_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("COSMOS_EVALUATOR_STORAGE_SESSION_TOKEN", "storage_session_token"),
    )

    # The standard AWS credential variables, consulted only by S3 and only when no ``storage_*`` credential is set.
    # boto3's default chain already reads them from the process environment, but not from the env files returned by
    # :meth:`get_env_files`; reading them here keeps S3 credentials kept in ``~/.aws/.env`` working for local runs.
    aws_access_key_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AWS_ACCESS_KEY_ID", "aws_access_key_id"),
    )

    aws_secret_access_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AWS_SECRET_ACCESS_KEY", "aws_secret_access_key"),
    )

    aws_session_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AWS_SESSION_TOKEN", "aws_session_token"),
    )

    # Comma-separated exact hostnames allowed to resolve to private/reserved IPs when fetching presigned/HTTPS inputs
    # (SSRF allowlist for internal S3-compatible clusters, e.g. "pbss.s8k.io", or a private Azure blob endpoint). Empty
    # by default — no exemption; set per checker pod that needs internal-cluster input support. Parsed into
    # S3Config/AzureBlobConfig.trusted_internal_hosts by the storage-provider factory.
    storage_trusted_internal_hosts: str = Field(
        default="",
        validation_alias=AliasChoices(
            "COSMOS_EVALUATOR_STORAGE_TRUSTED_INTERNAL_HOSTS", "storage_trusted_internal_hosts"
        ),
    )

    @property
    def input_storage_target(self) -> StorageTarget:
        """Addressing components for the input backend, parsed from ``input_storage_url`` or ``storage_url``.

        Raises:
            ValueError: If neither URL is set, or the URL does not address ``input_storage_type``'s backend. Local
                storage has no URL to parse and always raises.
        """
        return parse_storage_url(self.input_storage_url or self.storage_url, self.input_storage_type)

    @property
    def output_storage_target(self) -> StorageTarget:
        """Addressing components for the output backend, parsed from ``output_storage_url`` or ``storage_url``.

        Raises:
            ValueError: If neither URL is set, or the URL does not address ``output_storage_type``'s backend. Local
                storage has no URL to parse and always raises.
        """
        return parse_storage_url(self.output_storage_url or self.storage_url, self.output_storage_type)

    @property
    def s3_credentials(self) -> S3Credentials:
        """Credentials for the plain S3 backend: the ``storage_*`` fields when any is set, else the ``AWS_*`` fields.

        The two sets are taken whole rather than merged field by field, since a session token only validates alongside
        the key pair it was issued with. When both sets are empty every field is ``None``, which leaves boto3 to resolve
        its default credential chain.
        """
        explicit = S3Credentials(self.storage_access_key, self.storage_secret_key, self.storage_session_token)
        if explicit.access_key or explicit.secret_key or explicit.session_token:
            return explicit
        return S3Credentials(self.aws_access_key_id, self.aws_secret_access_key, self.aws_session_token)

    @property
    def active_api_key(self) -> str | None:
        """API key required for the current environment, or ``None`` to allow unrestricted access.

        Driven entirely by ``COSMOS_EVALUATOR_ENV`` (the ``env`` field): staging resolves to ``staging_api_key`` and
        production to ``prod_api_key``. Local and dev always resolve to ``None`` so local development and the existing
        test suite (which runs under ``COSMOS_EVALUATOR_ENV=local``) stay open with no header required.
        """
        if self.env == Environment.STAGING:
            return self.staging_api_key or None
        if self.env == Environment.PROD:
            return self.prod_api_key or None
        return None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Auto-injects a service-prefixed alias for every aliased field.

        When a subclass declares its own ``env_prefix`` (e.g. ``HALLUCINATION_``),
        every field on this class whose ``validation_alias`` is an
        ``AliasChoices`` gets a new highest-priority alias of the form
        ``<env_prefix><FIELD_NAME_UPPER>`` (e.g. ``HALLUCINATION_INPUT_STORAGE_TYPE``).
        That lets each service override any aliased setting individually
        without losing the global ``COSMOS_EVALUATOR_*`` knobs that sit lower
        in the alias chain.

        We hook ``__pydantic_init_subclass__`` (not Python's
        ``__init_subclass__``) because pydantic populates ``cls.model_fields``
        during its metaclass setup; ``__init_subclass__`` runs before
        subclass-only fields land there, and the resulting
        ``model_rebuild`` would silently drop them.

        The ``"COSMOS_EVALUATOR_"`` early return is defensive: subclasses that
        re-declare the parent prefix would otherwise produce
        ``COSMOS_EVALUATOR_COSMOS_EVALUATOR_*`` aliases.

        Fields listed in ``GLOBAL_ONLY_FIELDS`` are skipped, because a
        per-service alias for them would be accepted here but ignored by the
        code that actually consumes the value.
        """
        super().__pydantic_init_subclass__(**kwargs)
        prefix = cls.model_config.get("env_prefix") or ""
        if not prefix or prefix == "COSMOS_EVALUATOR_":
            return
        mutated = False
        for name, field_info in cls.model_fields.items():
            if name in GLOBAL_ONLY_FIELDS:
                continue
            alias = field_info.validation_alias
            if not isinstance(alias, AliasChoices):
                continue
            service_alias = f"{prefix}{name.upper()}"
            if service_alias in alias.choices:
                continue
            field_info.validation_alias = AliasChoices(service_alias, *alias.choices)
            mutated = True
        if mutated:
            cls.model_rebuild(force=True)

    @staticmethod
    def get_env_files() -> list[Path]:
        """Gets the environment files."""
        env_str = os.environ.get("COSMOS_EVALUATOR_ENV", Environment.LOCAL.value)
        if Environment(env_str) == Environment.LOCAL:
            return [Path("~/.cosmos_evaluator/.env").expanduser(), Path("~/.aws/.env").expanduser()]

        return []
