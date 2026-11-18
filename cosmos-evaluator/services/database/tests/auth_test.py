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

"""Unit tests for services.database.auth RBAC dependency functions.

The dependencies resolve the active read/admin keys from ``request.app.state.settings`` (dependency injection), so
these tests pass a fake request whose ``app.state.settings`` is a MagicMock with the relevant key fields.
"""

from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import MagicMock

from fastapi import HTTPException

from services.database import auth as auth_module

READ_KEY = "test-read-key-abc123"
ADMIN_KEY = "test-admin-key-xyz789"
WORKER_KEY = "test-worker-key-def456"


def _fake_request(read_key: str | None = "", admin_key: str | None = "", worker_keys: list[str] | None = None) -> Any:
    """Build a stand-in request exposing ``request.app.state.settings`` for the auth dependencies."""
    settings = MagicMock(read_key=read_key, admin_key=admin_key, worker_keys=worker_keys or [])
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(settings=settings)))


class TestIsValidReadKey(unittest.TestCase):
    """Tests for _is_valid_read_key helper."""

    def test_matches_read_key(self) -> None:
        self.assertTrue(auth_module._is_valid_read_key(MagicMock(read_key=READ_KEY, admin_key=""), READ_KEY))

    def test_matches_admin_key(self) -> None:
        self.assertTrue(auth_module._is_valid_read_key(MagicMock(read_key="", admin_key=ADMIN_KEY), ADMIN_KEY))

    def test_rejects_wrong_key(self) -> None:
        self.assertFalse(auth_module._is_valid_read_key(MagicMock(read_key=READ_KEY, admin_key=ADMIN_KEY), "wrong"))

    def test_rejects_when_no_keys_configured(self) -> None:
        self.assertFalse(auth_module._is_valid_read_key(MagicMock(read_key="", admin_key=""), "anything"))


class TestIsValidAdminKey(unittest.TestCase):
    """Tests for _is_valid_admin_key helper."""

    def test_matches_admin_key(self) -> None:
        self.assertTrue(auth_module._is_valid_admin_key(MagicMock(admin_key=ADMIN_KEY), ADMIN_KEY))

    def test_rejects_read_key(self) -> None:
        self.assertFalse(auth_module._is_valid_admin_key(MagicMock(read_key=READ_KEY, admin_key=ADMIN_KEY), READ_KEY))

    def test_rejects_when_admin_key_not_configured(self) -> None:
        self.assertFalse(auth_module._is_valid_admin_key(MagicMock(admin_key=""), "anything"))


class TestRequireReadKey(unittest.IsolatedAsyncioTestCase):
    """Tests for require_read_key dependency."""

    async def test_no_auth_when_no_keys_configured(self) -> None:
        result = await auth_module.require_read_key(_fake_request(read_key="", admin_key=""), api_key="")
        self.assertEqual(result, "no-auth")

    async def test_401_when_key_missing(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            await auth_module.require_read_key(_fake_request(read_key=READ_KEY, admin_key=""), api_key="")
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.detail, "API key required")

    async def test_401_when_key_is_none(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            await auth_module.require_read_key(_fake_request(read_key=READ_KEY, admin_key=""), api_key=None)
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_401_when_key_invalid(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            await auth_module.require_read_key(_fake_request(read_key=READ_KEY, admin_key=ADMIN_KEY), api_key="wrong")
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.detail, "Invalid API key")

    async def test_returns_read_for_read_key(self) -> None:
        result = await auth_module.require_read_key(
            _fake_request(read_key=READ_KEY, admin_key=ADMIN_KEY), api_key=READ_KEY
        )
        self.assertEqual(result, "read")

    async def test_returns_admin_for_admin_key(self) -> None:
        result = await auth_module.require_read_key(
            _fake_request(read_key=READ_KEY, admin_key=ADMIN_KEY), api_key=ADMIN_KEY
        )
        self.assertEqual(result, "admin")

    async def test_no_auth_when_only_none_keys(self) -> None:
        result = await auth_module.require_read_key(_fake_request(read_key=None, admin_key=None), api_key="anything")
        self.assertEqual(result, "no-auth")


class TestRequireAdminKey(unittest.IsolatedAsyncioTestCase):
    """Tests for require_admin_key dependency."""

    async def test_no_auth_when_admin_key_not_configured(self) -> None:
        result = await auth_module.require_admin_key(_fake_request(admin_key=""), api_key="")
        self.assertEqual(result, "no-auth")

    async def test_401_when_key_missing(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            await auth_module.require_admin_key(_fake_request(read_key="", admin_key=ADMIN_KEY), api_key="")
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.detail, "API key required")

    async def test_401_when_key_invalid(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            await auth_module.require_admin_key(_fake_request(read_key="", admin_key=ADMIN_KEY), api_key="wrong")
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.detail, "Invalid API key")

    async def test_403_when_read_key_used(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            await auth_module.require_admin_key(_fake_request(read_key=READ_KEY, admin_key=ADMIN_KEY), api_key=READ_KEY)
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(ctx.exception.detail, "Admin API key required")

    async def test_returns_admin_for_admin_key(self) -> None:
        result = await auth_module.require_admin_key(
            _fake_request(read_key=READ_KEY, admin_key=ADMIN_KEY), api_key=ADMIN_KEY
        )
        self.assertEqual(result, "admin")

    async def test_no_auth_when_admin_key_is_none(self) -> None:
        result = await auth_module.require_admin_key(_fake_request(admin_key=None), api_key="anything")
        self.assertEqual(result, "no-auth")


class TestIsValidWorkerKey(unittest.TestCase):
    """Tests for the _is_valid_worker_key helper."""

    def test_matches_any_configured_key(self) -> None:
        settings = MagicMock(worker_keys=["other", WORKER_KEY])
        self.assertTrue(auth_module._is_valid_worker_key(settings, WORKER_KEY))

    def test_rejects_unknown_key(self) -> None:
        self.assertFalse(auth_module._is_valid_worker_key(MagicMock(worker_keys=[WORKER_KEY]), "wrong"))

    def test_rejects_when_none_configured(self) -> None:
        self.assertFalse(auth_module._is_valid_worker_key(MagicMock(worker_keys=[]), WORKER_KEY))

    def test_ignores_empty_entries(self) -> None:
        """An empty string in the list must not authenticate a caller that sent no key material."""
        self.assertFalse(auth_module._is_valid_worker_key(MagicMock(worker_keys=["", ""]), ""))


class TestRequireWorkerKey(unittest.IsolatedAsyncioTestCase):
    """Tests for the require_worker_key dependency."""

    async def test_no_auth_when_nothing_configured(self) -> None:
        result = await auth_module.require_worker_key(_fake_request(), api_key="")
        self.assertEqual(result, "no-auth")

    async def test_401_when_key_missing(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            await auth_module.require_worker_key(_fake_request(worker_keys=[WORKER_KEY]), api_key="")
        self.assertEqual(ctx.exception.status_code, 401)

    async def test_401_when_key_invalid(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            await auth_module.require_worker_key(_fake_request(worker_keys=[WORKER_KEY]), api_key="wrong")
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.detail, "Invalid API key")

    async def test_returns_worker_for_worker_key(self) -> None:
        result = await auth_module.require_worker_key(
            _fake_request(admin_key=ADMIN_KEY, worker_keys=[WORKER_KEY]), api_key=WORKER_KEY
        )
        self.assertEqual(result, "worker")

    async def test_admin_key_is_accepted(self) -> None:
        """Admin already authorizes every write, so refusing it here would add no isolation."""
        result = await auth_module.require_worker_key(
            _fake_request(admin_key=ADMIN_KEY, worker_keys=[WORKER_KEY]), api_key=ADMIN_KEY
        )
        self.assertEqual(result, "admin")

    async def test_403_when_read_key_used(self) -> None:
        """A read key is a recognized credential lacking this authority, which is 403 rather than 401."""
        with self.assertRaises(HTTPException) as ctx:
            await auth_module.require_worker_key(
                _fake_request(read_key=READ_KEY, admin_key=ADMIN_KEY, worker_keys=[WORKER_KEY]), api_key=READ_KEY
            )
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(ctx.exception.detail, "Worker API key required")

    async def test_admin_still_authenticates_before_worker_keys_are_provisioned(self) -> None:
        """Rollout ordering: the deployment gets the dependency before it gets the keys."""
        result = await auth_module.require_worker_key(_fake_request(admin_key=ADMIN_KEY), api_key=ADMIN_KEY)
        self.assertEqual(result, "admin")


if __name__ == "__main__":
    unittest.main()
