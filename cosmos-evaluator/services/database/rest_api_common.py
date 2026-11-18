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

"""FastAPI application and uvicorn entry point for the Database REST API.

Builds the base ``DatabaseRestApi`` app over the client-agnostic ``Settings`` (exposed as the module-level ``app``
singleton) and serves it with uvicorn via ``main()``. The two entry-point binaries (``rest_api`` for local dev,
``rest_api_deployment`` for the container) are thin wrappers that call ``main()``; keeping the app here (a git_sha-free
library) lets the deployment binary depend on it without dragging //utils:git_sha into its runfiles, and lets Gazelle
manage both binaries' dependencies. The reusable ``DatabaseRestApi`` builder stays in ``app.py`` so other entry points
can import the class without building this base app as an import side effect.

``db_config`` resolution: an explicitly injected ``DATABASE_DB_CONFIG`` always wins. Otherwise, in ``local`` mode
only, the ``local`` section of ``database/config.yaml`` is loaded (credentials resolved from
``COSMOS_EVALUATOR_LOCAL_DB_*``) so a standalone dev run auto-connects to the cosmos-eval-local-db container without
hand-writing the JSON. When neither is available the service runs unconfigured (no-DB mode).
"""

import argparse
import logging
import os

import uvicorn

from database.db_config import load_db_config
from services import utils
from services.database.app import DatabaseRestApi
from services.database.settings import get_settings
from services.settings_base import LOG_FORMAT, LogLevel, build_log_config
from utils.environment import Environment

logger = logging.getLogger(__name__)

settings = get_settings()

# Local-mode convenience: derive db_config from database/config.yaml when it was not injected directly. Gated on
# LOCAL so deployed (staging/prod) runs still require an explicit DATABASE_DB_CONFIG and never read the local file.
if settings.db_env == Environment.LOCAL and not settings.db_config:
    local_db_config = load_db_config()
    if local_db_config:
        settings = settings.model_copy(update={"db_config": local_db_config})
        logger.info("Loaded local db_config from database/config.yaml for %s environment", settings.db_env.value)

app = DatabaseRestApi(settings, version=utils.get_contents_from_runfile("services/database/version.txt")).build()


def main() -> None:
    """Run the Database REST API under uvicorn.

    Shared entry point for both the local (:rest_api) and container (:rest_api_deployment) binaries; the only
    difference between them is whether //utils:git_sha is on the binary's runfiles, controlled in the BUILD file.
    """
    parser = argparse.ArgumentParser(description="Run the Database REST API")
    parser.add_argument("--host", default=os.getenv("UVICORN_HOST", "0.0.0.0"), help="Host to bind to")
    parser.add_argument(
        "--port",
        type=int,
        default=os.getenv("UVICORN_PORT", "8080"),
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

    logging.basicConfig(level=args.log_level, format=LOG_FORMAT)
    logger.info("Starting Database REST API on %s:%s", args.host, args.port)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        log_config=build_log_config(args.log_level),
    )
