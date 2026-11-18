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

"""FastAPI application and uvicorn entry point for physical-plausibility checking.

Checker transport, slot accounting, and durable job dispatch are supplied by
``services.framework.checker_app.build_checker_app``. This module only maps the
checker-specific request and result types.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import uvicorn

from checks.physical_plausibility.processor import PhysicalPlausibilityResult
from checks.utils.multistorage import setup_msc_config
from services import utils
from services.framework.checker_app import CheckerSpec, build_checker_app
from services.framework.protocols.storage_provider import StorageProvider
from services.physical_plausibility.models import PhysicalPlausibilityCloudRequest, PhysicalPlausibilityRequest
from services.physical_plausibility.service import PhysicalPlausibilityService
from services.physical_plausibility.settings import get_settings
from services.settings_base import LOG_FORMAT, LogLevel, build_log_config

settings = get_settings()
logger = logging.getLogger(__name__)


def _verify_startup() -> None:
    """Configure MultiStorage before the processor resolves remote request URLs."""
    setup_msc_config(os.environ.get("MULTISTORAGECLIENT_CONFIGURATION"))


async def _process_request(
    api_request: PhysicalPlausibilityCloudRequest, _input_storage_provider: StorageProvider, _temp_dir: Path
) -> PhysicalPlausibilityRequest:
    """Map the supplied video URL/path into the processor's local request shape.

    The processor resolves remote URIs itself, so this checker does not use the
    storage provider or temporary directory supplied by the common transport.
    """
    return PhysicalPlausibilityRequest(
        clip_id=api_request.clip_id,
        video_path=api_request.video_url,
        config=api_request.config,
        output_storage_prefix=api_request.output_storage_prefix,
    )


async def _build_response_data(
    _request: PhysicalPlausibilityCloudRequest,
    result: PhysicalPlausibilityResult,
    _output_storage_provider: StorageProvider,
) -> dict[str, Any]:
    """Return the check result directly; this checker produces no uploaded artifacts."""
    return result.model_dump(mode="json")


app = build_checker_app(
    CheckerSpec(
        name="physical_plausibility",
        checker_name="metropolis.physical_plausibility",
        title="Physical Plausibility Service API",
        description="REST API for temporal physical-plausibility analysis",
        version=utils.get_contents_from_runfile("services/physical_plausibility/version.txt"),
        config_description="Default configuration for physical-plausibility processing",
        settings=settings,
        concurrency_limit=settings.process_concurrency_limit,
        cloud_request_model=PhysicalPlausibilityCloudRequest,
        build_service=lambda: PhysicalPlausibilityService(
            max_batch_size=settings.max_batch_size,
            max_batch_wait_ms=settings.max_batch_wait_ms,
            checkpoint_uri=settings.checkpoint_uri,
            config_dir=settings.config_dir,
        ),
        prepare_request=_process_request,
        build_response_data=_build_response_data,
        verify_startup=_verify_startup,
    )
)


def main() -> None:
    """Run the physical-plausibility API under uvicorn."""
    parser = argparse.ArgumentParser(description="Run the Physical Plausibility API")
    parser.add_argument("--host", default=os.getenv("UVICORN_HOST", "0.0.0.0"), help="Host to bind to")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("UVICORN_PORT", "8088")),
        choices=range(1, 65536),
        help="Port to bind to",
        metavar="[1-65535]",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level.value,
        choices=[level.value for level in LogLevel],
        help="Log level",
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT)
    logger.info("Starting Physical Plausibility API on %s:%s", args.host, args.port)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        log_config=build_log_config(args.log_level),
    )
