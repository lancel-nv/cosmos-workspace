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

"""Reusable API key authentication middleware for checker REST apps.

Provides ``install_api_key_auth``: a one-line helper that registers an ``X-API-Key`` enforcement middleware on a
FastAPI app. When the supplied key is falsy the helper is a no-op, so endpoints stay open (the default for local/dev
and for any environment where the key is unset). When a key is supplied, every request is rejected with HTTP 401
unless it carries a matching ``X-API-Key`` header, except for the exempt paths (health, metrics and the OpenAPI docs
surface) which always pass through.

The key is also declared in the app's OpenAPI document, which is what lets the exempt ``/docs`` page send it.

This is intentionally a coarse, single-key service gate. It is kept separate from the Database REST API's role-based
keys.
"""

import secrets
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response

# Paths that must remain reachable without a key. ``/metrics`` is scraped by platform monitoring and only exposes
# occupancy gauges.
EXEMPT_PATHS = frozenset({"/health", "/metrics", "/docs", "/openapi.json", "/redoc"})

# Header carrying the API key. Matches the convention already used by the Database REST API's RBAC.
API_KEY_HEADER = "X-API-Key"

# Name the key is published under in the OpenAPI document's ``securitySchemes``. Only ever seen by schema consumers,
# which is who it is for: it is the label on the docs page's Authorize dialog.
API_KEY_SCHEME_NAME = "ApiKeyHeader"


def install_api_key_auth(
    app: FastAPI,
    api_key: str | None,
    exempt_paths: frozenset[str] = EXEMPT_PATHS,
) -> None:
    """Register ``X-API-Key`` enforcement on ``app``. No-op when ``api_key`` is falsy.

    Args:
        app: The FastAPI application to protect.
        api_key: The key required on incoming requests, or ``None``/empty to leave the app open.
        exempt_paths: Request paths that bypass the check (defaults to health, metrics and the OpenAPI docs surface).
    """
    if not api_key:
        return

    @app.middleware("http")
    async def _enforce_api_key(request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in exempt_paths:
            return await call_next(request)
        provided = request.headers.get(API_KEY_HEADER)
        if not provided or not secrets.compare_digest(provided, api_key):
            return JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)
        return await call_next(request)

    _declare_api_key_in_schema(app, exempt_paths)


def _declare_api_key_in_schema(app: FastAPI, exempt_paths: frozenset[str]) -> None:
    """Publish the header this app requires in its OpenAPI document.

    Enforcement lives in middleware, and middleware is invisible to FastAPI's schema generation. Left undeclared, the
    docs page -- which is exempt, and so always reachable -- renders with no Authorize dialog and no field for the
    header, making every "Try it out" on the page a guaranteed 401 with nothing the reader can do about it. Declaring
    the scheme here, rather than as a dependency on every route, means an app is documented by virtue of being gated
    and the enforcement stays in exactly one place.

    Only the paths the middleware actually guards are marked, so ``/health`` still reads as open to anything
    generating a client from this document.
    """
    build_schema = app.openapi

    def openapi_with_api_key() -> dict[str, Any]:
        # Wrapped rather than computed now: routes are still being registered when ``install_api_key_auth`` runs, so a
        # schema built at install time would describe an app with no endpoints. FastAPI caches its own result, and
        # every write below is idempotent, so repeat calls stay cheap and stable.
        schema = build_schema()
        components: dict[str, Any] = schema.setdefault("components", {})
        components.setdefault("securitySchemes", {})[API_KEY_SCHEME_NAME] = {
            "type": "apiKey",
            "in": "header",
            "name": API_KEY_HEADER,
        }
        for path, operations in schema.get("paths", {}).items():
            if path in exempt_paths:
                continue
            for operation in operations.values():
                # A path item holds more than operations (``parameters``, ``summary``); only the verbs take security.
                if isinstance(operation, dict):
                    operation.setdefault("security", [{API_KEY_SCHEME_NAME: []}])
        return schema

    # Replacing the method on the instance is FastAPI's documented hook for customizing the schema, and it is the only
    # one that composes with whatever the app already does; mypy has no way to express that.
    app.openapi = openapi_with_api_key  # type: ignore[method-assign]
