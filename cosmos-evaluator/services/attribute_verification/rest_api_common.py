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

"""FastAPI application and uvicorn entry point for the Attribute Verification Service.

Only the checker-specific steps live here -- resolving this checker's input, shaping its response, and pointing
``multistorageclient`` at its profile config. The endpoints, lifespan, slot accounting and durable job protocol come
from ``services.framework.checker_app.build_checker_app``.

The two entry-point binaries (``rest_api`` for local dev, ``rest_api_deployment`` for the container) are thin wrappers
that call ``main()``; keeping the app here (a git_sha-free library) lets the deployment binary depend on it without
dragging //utils:git_sha into its runfiles, and lets Gazelle manage both binaries' dependencies.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import uvicorn

from checks.attribute_verification.processor import AttributeVerificationResult
from checks.utils.multistorage import setup_msc_config
from services import utils
from services.attribute_verification.models import (
    AttributeVerificationCloudRequest,
    AttributeVerificationRequest,
)
from services.attribute_verification.service import AttributeVerificationService
from services.attribute_verification.settings import get_settings
from services.framework.checker_app import CheckerSpec, build_checker_app
from services.framework.protocols.storage_provider import StorageProvider
from services.settings_base import LOG_FORMAT, LogLevel, build_log_config

settings = get_settings()

logger = logging.getLogger(__name__)

log_level: LogLevel = LogLevel.INFO


def _verify_startup() -> None:
    """Point ``multistorageclient`` at its profile config before the first request lands.

    The processor resolves the remote video itself, so this stands in for the download step the other checkers perform
    in ``prepare_request``. Done at startup rather than per request because it mutates process-global client state.
    """
    setup_msc_config(os.environ.get("MULTISTORAGECLIENT_CONFIGURATION"))


def _build_service() -> AttributeVerificationService:
    """Build the service at the verbosity ``main()`` parsed off the command line."""
    return AttributeVerificationService(verbose=log_level.value)


async def _process_request(
    api_request: AttributeVerificationCloudRequest, _input_storage_provider: StorageProvider, _temp_dir: Path
) -> AttributeVerificationRequest:
    """Map the URL-based cloud request onto the local-path request.

    The processor downloads the remote video itself via ``multistorageclient``, so the URL flows straight into the
    local-path field; neither the storage provider nor the scratch directory is used.
    """
    return AttributeVerificationRequest(
        clip_id=api_request.clip_id,
        augmented_video_path=api_request.augmented_video_url,
        config=api_request.config,
        verbose=api_request.verbose,
        output_storage_prefix=api_request.output_storage_prefix,
    )


async def _build_response_data(
    _request: AttributeVerificationCloudRequest,
    result: AttributeVerificationResult,
    _output_storage_provider: StorageProvider,
) -> dict[str, Any]:
    """Return the verdicts as-is: this check writes no artifacts, so there is nothing to upload."""
    return result.model_dump(mode="json")


app = build_checker_app(
    CheckerSpec(
        name="attribute_verification",
        checker_name="metropolis.attribute_verification",
        title="Attribute Verification Service API",
        description="REST API for attribute verification",
        version=utils.get_contents_from_runfile("services/attribute_verification/version.txt"),
        config_description="Default configuration for attribute verification processing",
        settings=settings,
        concurrency_limit=settings.process_concurrency_limit,
        cloud_request_model=AttributeVerificationCloudRequest,
        build_service=_build_service,
        prepare_request=_process_request,
        build_response_data=_build_response_data,
        verify_startup=_verify_startup,
    )
)


def main() -> None:
    """Parse CLI arguments and run the Attribute Verification API under uvicorn.

    Shared entry point for both the local (:rest_api) and container (:rest_api_deployment) binaries; the only
    difference between them is whether //utils:git_sha is on the binary's runfiles, controlled in the BUILD file.
    """
    global log_level

    parser = argparse.ArgumentParser(description="Run the Attribute Verification API")
    parser.add_argument("--host", default=os.getenv("UVICORN_HOST", "0.0.0.0"), help="Host to bind to")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("UVICORN_PORT", "8080")),
        choices=range(1, 65536),
        help="Port to bind to",
        metavar="[1-65535]",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level.value,
        choices=[log_level.value for log_level in LogLevel],
        help="Log level",
    )

    args = parser.parse_args()

    # log_level is a module global _build_service reads to set the service's verbosity, so keep it assigned here.
    log_level = LogLevel(args.log_level)
    logging.basicConfig(level=log_level.value, format=LOG_FORMAT)
    logger.info("Starting Attribute Verification API on %s:%s", args.host, args.port)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        log_config=build_log_config(log_level.value),
    )
