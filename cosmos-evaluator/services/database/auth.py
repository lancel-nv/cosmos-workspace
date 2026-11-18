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

"""RBAC authentication dependencies for the Database REST API.

Provides three FastAPI dependency functions:
- require_read_key: accepts read key or admin key
- require_admin_key: accepts only admin key, returns 403 for read-only keys
- require_worker_key: accepts a worker key or admin key, returns 403 for read-only keys

The active keys are resolved per request from ``request.app.state.settings`` (set by the app builder
``app.DatabaseRestApi``) rather than a module-global ``get_settings()`` call.

The worker role exists so checker pods can coordinate their own job rows without holding a secret that authorizes
writes to every other table. It is not a subset of read: a worker writes, it just writes to one narrow surface.
"""

import secrets

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from services.database.settings import Settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_settings(request: Request) -> Settings:
    """Resolve the Settings instance the app builder stored on ``request.app.state``."""
    return request.app.state.settings


def _is_valid_read_key(settings: Settings, api_key: str) -> bool:
    """Check if the key matches the read key or admin key."""
    if settings.read_key and secrets.compare_digest(api_key, settings.read_key):
        return True
    return bool(settings.admin_key) and secrets.compare_digest(api_key, settings.admin_key)


def _is_valid_admin_key(settings: Settings, api_key: str) -> bool:
    """Check if the key matches the admin key."""
    return bool(settings.admin_key) and secrets.compare_digest(api_key, settings.admin_key)


def _is_valid_worker_key(settings: Settings, api_key: str) -> bool:
    """Check if the key matches any configured worker key.

    Every candidate is compared even after a match so the running time does not depend on the key's position in the
    list, keeping the constant-time guarantee ``compare_digest`` provides for a single comparison.
    """
    matched = False
    for worker_key in settings.worker_keys:
        if worker_key and secrets.compare_digest(api_key, worker_key):
            matched = True
    return matched


async def require_read_key(request: Request, api_key: str | None = Security(api_key_header)) -> str:
    """Require a read key or admin key. Rejects with 401 if missing/invalid."""
    settings = _get_settings(request)
    if not settings.read_key and not settings.admin_key:
        return "no-auth"
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    if not _is_valid_read_key(settings, api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return "admin" if _is_valid_admin_key(settings, api_key) else "read"


async def require_admin_key(request: Request, api_key: str | None = Security(api_key_header)) -> str:
    """Require the admin key. Rejects with 401 if missing, 403 if read-only key."""
    settings = _get_settings(request)
    if not settings.admin_key:
        return "no-auth"
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    if not _is_valid_admin_key(settings, api_key):
        if _is_valid_read_key(settings, api_key):
            raise HTTPException(status_code=403, detail="Admin API key required")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return "admin"


async def require_worker_key(request: Request, api_key: str | None = Security(api_key_header)) -> str:
    """Require a worker key or the admin key. Rejects with 401 if missing/unknown, 403 for a read-only key.

    Deployments that have not provisioned worker keys yet still authenticate the admin key, so the checker-work
    surface is never wider than the admin surface already is; it only ever narrows access for the pods that hold a
    worker key. As with the other two roles, a deployment with no keys configured at all is unauthenticated.
    """
    settings = _get_settings(request)
    if not settings.worker_keys and not settings.admin_key:
        return "no-auth"
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    if _is_valid_admin_key(settings, api_key):
        return "admin"
    if _is_valid_worker_key(settings, api_key):
        return "worker"
    if _is_valid_read_key(settings, api_key):
        raise HTTPException(status_code=403, detail="Worker API key required")
    raise HTTPException(status_code=401, detail="Invalid API key")
