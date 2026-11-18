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

"""Integration test for the client-agnostic Arbitrator REST API container image.

Loads and runs the base Arbitrator OCI image to verify it starts successfully (with the required
``COSMOS_EVALUATOR_CHECKER_ENDPOINTS`` env set), responds to health checks, and exposes the ``/process`` route in its
OpenAPI schema. This is a CPU-only smoke test (no GPU / no DeviceRequest); the arbitrator runs without any checker
services reachable — it only needs its checker-endpoints map to boot.
"""

import json
import logging
import os
import time
from typing import Any
import unittest
import urllib.error
import urllib.request

from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from utils.container_images import load_image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


_raw_port = os.environ.get("UVICORN_PORT")
if _raw_port is None:
    raise RuntimeError(
        'UVICORN_PORT environment variable is not set. Set it in the test BUILD rule via env = {"UVICORN_PORT": ...}.'
    )
try:
    UVICORN_PORT = int(_raw_port)
except ValueError:
    raise RuntimeError(f"UVICORN_PORT environment variable must be an integer, got: {_raw_port!r}") from None

# The client-agnostic base requires COSMOS_EVALUATOR_CHECKER_ENDPOINTS (a JSON object string mapping checker base name
# -> endpoint URL). No checker service needs to actually be reachable for the boot/health smoke test.
_CHECKER_ENDPOINTS = json.dumps({"av.objects_single_view": "http://localhost:8082"})


class ArbitratorImageContainer(DockerContainer):
    """Custom container class for the client-agnostic Arbitrator REST API service."""

    def __init__(self, image: str, **kwargs: Any) -> None:
        """Initialize the Arbitrator container."""
        super().__init__(image, **kwargs)
        self.with_bind_ports(UVICORN_PORT, UVICORN_PORT)
        self.with_env("UVICORN_PORT", str(UVICORN_PORT))
        self.with_env("UVICORN_HOST", "0.0.0.0")
        # Use staging env to avoid local env file loading.
        self.with_env("COSMOS_EVALUATOR_ENV", "staging")
        # Required by the client-agnostic base settings; recording stays disabled (no database URL set).
        self.with_env("COSMOS_EVALUATOR_CHECKER_ENDPOINTS", _CHECKER_ENDPOINTS)

    def get_health_url(self) -> str:
        """Get the URL for the health endpoint."""
        host = self.get_container_host_ip()
        return f"http://{host}:{UVICORN_PORT}/health"

    def get_openapi_url(self) -> str:
        """Get the URL for the OpenAPI schema."""
        host = self.get_container_host_ip()
        return f"http://{host}:{UVICORN_PORT}/openapi.json"


class TestArbitratorImage(unittest.TestCase):
    """Integration tests for the standalone client-agnostic Arbitrator REST API container image."""

    container: ArbitratorImageContainer
    image_name: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        """Loads the Docker image and start the container."""
        cls.image_name = load_image("services/arbitrator/tests/test_image_load.sh")
        logger.info("Using image: %s", cls.image_name)

        cls.container = ArbitratorImageContainer(cls.image_name)
        cls.container.waiting_for(LogMessageWaitStrategy("Application startup complete").with_startup_timeout(600))
        cls.container.start()
        logger.info("Container is ready for tests")

    @classmethod
    def tearDownClass(cls) -> None:
        """Stops the shared container."""
        if hasattr(cls, "container") and cls.container:
            cls.container.stop()
            logger.info("Container stopped")

    def _fetch_json_with_retries(self, url: str, max_retries: int = 5) -> dict[str, Any]:
        """Fetch JSON from a URL, retrying on transient errors.

        Returns the parsed JSON dict on success, or calls self.fail() after exhausting retries.
        """
        logger.info("Fetching %s (max %s retries)", url, max_retries)
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    return json.loads(response.read().decode())
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning("Attempt %s/%s failed: %s", attempt + 1, max_retries, e)
                time.sleep(1)

        self.fail(f"{url} did not respond after {max_retries} attempts: {last_error}")

    def test_container_starts_and_becomes_healthy(self) -> None:
        """Tests that the container starts and the health endpoint responds."""
        data = self._fetch_json_with_retries(self.container.get_health_url(), max_retries=10)

        self.assertTrue(data.get("success"), f"Health check failed: {data}")
        self.assertEqual(data["data"]["status"], "healthy", f"Unexpected health status: {data}")
        self.assertIn("version", data["data"])
        self.assertEqual(data["data"]["service"], "Arbitrator Service API")
        logger.info("Health check passed: %s", data)

    def test_openapi_exposes_process_route(self) -> None:
        """Tests that the OpenAPI schema exposes the core ``/process/async`` submission route."""
        schema = self._fetch_json_with_retries(self.container.get_openapi_url(), max_retries=10)

        paths = schema.get("paths", {})
        self.assertIn("/process/async", paths, f"/process/async not found in OpenAPI paths: {sorted(paths)}")
        logger.info("OpenAPI exposes /process/async: %s", sorted(paths))

    def test_container_stays_running(self) -> None:
        """Tests that the container stays running without crashing."""
        time.sleep(5)

        container_info = self.container.get_wrapped_container()
        container_info.reload()

        self.assertEqual(
            container_info.status,
            "running",
            f"Container stopped unexpectedly. Logs: {self.container.get_logs()}",
        )
        logger.info("Container stayed running for 5 seconds")


if __name__ == "__main__":
    unittest.main()
