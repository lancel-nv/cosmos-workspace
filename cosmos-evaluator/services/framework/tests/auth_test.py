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

"""Unit tests for the framework API key auth middleware."""

from typing import Any
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.framework.auth import API_KEY_SCHEME_NAME, install_api_key_auth

API_KEY = "s3cr3t-key"


def _build_app(api_key: str | None) -> FastAPI:
    """Build a small app exposing the same endpoint shapes the AV checkers use.

    The key is installed before the routes, as ``build_checker_app`` does it, so the schema assertions below exercise
    the ordering the real apps use.
    """
    app = FastAPI()
    install_api_key_auth(app, api_key)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/config")
    async def config() -> dict[str, str]:
        return {"config": "ok"}

    @app.get("/metrics")
    async def metrics() -> dict[str, str]:
        return {"metrics": "ok"}

    @app.post("/process")
    async def process() -> dict[str, str]:
        return {"processed": "ok"}

    return app


def _schema(api_key: str | None) -> dict[str, Any]:
    """The app's published OpenAPI document, read the way a docs page or client generator would."""
    return TestClient(_build_app(api_key)).get("/openapi.json").json()


class TestApiKeyEnforcement(unittest.TestCase):
    """The middleware's runtime behavior: which requests reach a route and which are turned away."""

    def test_no_key_leaves_all_endpoints_open(self) -> None:
        """When no key is configured the middleware is a no-op and every path is reachable."""
        client = TestClient(_build_app(None))

        self.assertEqual(client.get("/health").status_code, 200)
        self.assertEqual(client.get("/config").status_code, 200)
        self.assertEqual(client.post("/process").status_code, 200)

    def test_empty_key_leaves_all_endpoints_open(self) -> None:
        """An empty-string key is treated the same as no key (unrestricted)."""
        client = TestClient(_build_app(""))

        self.assertEqual(client.post("/process").status_code, 200)

    def test_protected_path_rejects_missing_key(self) -> None:
        """A non-exempt path returns 401 when no X-API-Key header is supplied."""
        client = TestClient(_build_app(API_KEY))

        response = client.post("/process")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid or missing API key")

    def test_protected_path_rejects_wrong_key(self) -> None:
        """A non-exempt path returns 401 when the X-API-Key header is incorrect."""
        client = TestClient(_build_app(API_KEY))

        response = client.post("/process", headers={"X-API-Key": "wrong"})

        self.assertEqual(response.status_code, 401)

    def test_protected_path_accepts_correct_key(self) -> None:
        """A non-exempt path is reachable when the correct X-API-Key header is supplied."""
        client = TestClient(_build_app(API_KEY))

        response = client.post("/process", headers={"X-API-Key": API_KEY})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["processed"], "ok")

    def test_config_path_is_protected(self) -> None:
        """/config is not exempt, so it requires the key when one is configured."""
        client = TestClient(_build_app(API_KEY))

        self.assertEqual(client.get("/config").status_code, 401)
        self.assertEqual(client.get("/config", headers={"X-API-Key": API_KEY}).status_code, 200)

    def test_health_is_exempt_even_with_key_set(self) -> None:
        """/health must stay reachable without a key."""
        client = TestClient(_build_app(API_KEY))

        self.assertEqual(client.get("/health").status_code, 200)

    def test_metrics_is_exempt_even_with_key_set(self) -> None:
        """Metrics are scraped by monitoring that cannot carry the key, so gating them just loses the metrics."""
        client = TestClient(_build_app(API_KEY))

        self.assertEqual(client.get("/metrics").status_code, 200)

    def test_docs_surface_is_exempt_even_with_key_set(self) -> None:
        """The OpenAPI docs surface stays browsable without a key."""
        client = TestClient(_build_app(API_KEY))

        self.assertEqual(client.get("/openapi.json").status_code, 200)
        self.assertEqual(client.get("/docs").status_code, 200)


class TestOpenApiDeclaration(unittest.TestCase):
    """What the published document tells a docs page or a generated client about the key."""

    def test_the_key_is_published_as_a_security_scheme(self) -> None:
        """Without this the docs page has no Authorize dialog, so nothing there can ever get past the middleware."""
        schemes = _schema(API_KEY)["components"]["securitySchemes"]

        self.assertEqual(schemes[API_KEY_SCHEME_NAME], {"type": "apiKey", "in": "header", "name": "X-API-Key"})

    def test_guarded_operations_declare_the_key(self) -> None:
        schema = _schema(API_KEY)

        self.assertEqual(schema["paths"]["/process"]["post"]["security"], [{API_KEY_SCHEME_NAME: []}])
        self.assertEqual(schema["paths"]["/config"]["get"]["security"], [{API_KEY_SCHEME_NAME: []}])

    def test_exempt_operations_stay_undeclared(self) -> None:
        """Exempt paths are reachable without the header, and a client generated from this document should know that."""
        paths = _schema(API_KEY)["paths"]

        self.assertNotIn("security", paths["/health"]["get"])
        self.assertNotIn("security", paths["/metrics"]["get"])

    def test_no_key_leaves_the_schema_untouched(self) -> None:
        """An open app must not advertise a credential nothing checks."""
        schema = _schema(None)

        self.assertNotIn("securitySchemes", schema.get("components", {}))
        self.assertNotIn("security", schema["paths"]["/process"]["post"])


if __name__ == "__main__":
    unittest.main()
