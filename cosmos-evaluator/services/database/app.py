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

"""Client-agnostic FastAPI app builder for the Database REST API service.

``DatabaseRestApi`` assembles a complete, runnable application surface — health check, the RBAC-gated arbitrator router,
and the service lifespan — over an injected ``Settings`` instance.

Authentication uses API keys via the X-API-Key header:
- Read key: grants access to GET endpoints and /health
- Admin key: grants access to all endpoints (read + write)
- Worker key: grants access only to the checker job-coordination endpoints
- /docs, /redoc, /openapi.json remain unauthenticated

Most routers are mounted behind a blanket read-key guard, so their routes can only ever narrow access further. The
checker-work router is the exception: a checker pod holds a worker key and nothing else, so a blanket read-key guard
would reject it before its own per-route dependency ever ran. Those routers declare their auth per route instead.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging

from fastapi import APIRouter, Depends, FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from services import utils
from services.database.auth import require_read_key
from services.database.routers import arbitrator, checker_work
from services.database.service import DatabaseService
from services.database.settings import Settings
from services.framework.response_formatters.json_response_formatter import (
    ErrorEnvelope,
    JsonResponseFormatter,
    SuccessEnvelope,
)
from utils.environment import Environment

logger = logging.getLogger(__name__)
json_formatter = JsonResponseFormatter()


class HealthStatus(BaseModel):
    """Payload of ``GET /health``, probed by container readiness checks and by ``ApiClient.check_health``.

    Declared here rather than in ``database.api_models`` because it describes the service rather than any row it
    serves, and ``check_health`` only looks at the status code. ``extra="forbid"`` for the same reason the row models
    deliberately allow extras: nothing forwards this payload onward, so there is no compatibility argument for letting
    undeclared fields through, and forbidding them keeps the declaration honest.
    """

    model_config = ConfigDict(extra="forbid")

    service: str
    status: str
    version: str
    git_sha: str
    environment: str


class DatabaseRestApi:
    """Builder for the Database REST API FastAPI app."""

    title: str = "Database REST API"
    description: str = "Database REST API Service"
    service_class: type[DatabaseService] = DatabaseService

    def __init__(self, settings: Settings, version: str) -> None:
        """Store the resolved settings and the version string surfaced by ``/health``.

        Args:
            settings: The resolved Settings instance; stored on ``app.state.settings`` for the RBAC dependencies.
            version: Version string surfaced by ``/health`` (typically read from the deployment's version.txt runfile).
        """
        self._settings = settings
        self._version = version

    def routers(self) -> list[APIRouter]:
        """Routers mounted under the read-key guard."""
        return [arbitrator.router]

    def self_guarded_routers(self) -> list[APIRouter]:
        """Routers that carry their own per-route auth and must not inherit the read-key guard.

        Every route in these routers is responsible for declaring its own dependency; a route that forgets is
        unauthenticated. That is the cost of admitting callers who do not hold a read key.
        """
        return [checker_work.router]

    def _build_service(self) -> DatabaseService:
        """Build the service for the app lifespan."""
        return self.service_class(self._settings)

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI) -> AsyncGenerator[None]:
        """Construct the service, initialize its backends, and attach it to ``app.state`` for the request handlers."""
        logger.info("Initializing %s for %s environment...", self.title, self._settings.env.value)

        service = self._build_service()
        try:
            await service.init()
            logger.info("Service initialized successfully")
        except Exception:
            logger.exception("Failed to initialize database service; endpoints will return errors until it recovers.")

        app.state.service = service

        yield

        logger.info("Shutting down service...")
        await service.close()
        logger.info("Service shutdown complete")

    def build(self) -> FastAPI:
        """Assemble and return the configured FastAPI application ready to serve."""
        app = FastAPI(
            title=self.title,
            description=self.description,
            version=self._version,
            lifespan=self._lifespan,
            debug=self._settings.env == Environment.LOCAL,
        )
        # The RBAC dependencies resolve the active read/admin keys from here per request (dependency injection).
        app.state.settings = self._settings

        for router in self.routers():
            app.include_router(router, dependencies=[Depends(require_read_key)])
        for router in self.self_guarded_routers():
            app.include_router(router)

        settings = self._settings

        @app.get(
            "/health",
            dependencies=[Depends(require_read_key)],
            response_model=SuccessEnvelope[HealthStatus],
            responses={"4XX": {"model": ErrorEnvelope}, "5XX": {"model": ErrorEnvelope}},
        )
        async def health_check() -> JSONResponse:
            """Health check endpoint."""
            return await json_formatter.format_success(
                {
                    "service": app.title,
                    "status": "healthy",
                    "version": app.version,
                    "git_sha": utils.get_git_sha(),
                    "environment": settings.env.value,
                }
            )

        return app
