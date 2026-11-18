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

"""Shared test infrastructure for Objects container image tests.

Provides a reusable mixin and helpers for loading OCI images and running standard
integration checks (health, git SHA, OpenAPI, licensing, stability). Concrete test
modules combine ImageTestMixin with unittest.TestCase and set LOADER_SCRIPT_PATH
and SERVICE_NAME.
"""

import json
import logging
import os
import time
from typing import Any
import urllib.error
import urllib.request

from docker.types import DeviceRequest
from testcontainers.core.container import DockerContainer
from testcontainers.core.wait_strategies import LogMessageWaitStrategy

from utils.container_images import load_image

logger = logging.getLogger(__name__)


def get_uvicorn_port() -> int:
    """Read and validate the UVICORN_PORT environment variable."""
    raw_port = os.environ.get("UVICORN_PORT")
    if raw_port is None:
        raise RuntimeError(
            "UVICORN_PORT environment variable is not set. "
            'Set it in the test BUILD rule via env = {"UVICORN_PORT": ...}.'
        )
    try:
        return int(raw_port)
    except ValueError:
        raise RuntimeError(f"UVICORN_PORT environment variable must be an integer, got: {raw_port!r}")


UVICORN_PORT = get_uvicorn_port()


class ObjectsContainer(DockerContainer):
    """Container wrapper for the Objects service image variants."""

    def __init__(self, image: str, **kwargs: Any) -> None:
        super().__init__(image, **kwargs)
        self.with_bind_ports(UVICORN_PORT, UVICORN_PORT)
        self.with_env("UVICORN_PORT", str(UVICORN_PORT))
        self.with_env("UVICORN_HOST", "0.0.0.0")
        self.with_env("COSMOS_EVALUATOR_ENV", "staging")
        self.with_env("COSMOS_EVALUATOR_STORAGE_TYPE", "local")
        self.with_kwargs(device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])])

    def get_health_url(self) -> str:
        host = self.get_container_host_ip()
        return f"http://{host}:{UVICORN_PORT}/health"

    def get_config_url(self) -> str:
        host = self.get_container_host_ip()
        return f"http://{host}:{UVICORN_PORT}/config"


class ImageTestMixin:
    """Mixin providing integration tests for Objects container images.

    Not a unittest.TestCase subclass, so unittest will not discover or run it directly.
    Concrete test classes should inherit from both this mixin and unittest.TestCase,
    and set LOADER_SCRIPT_PATH and SERVICE_NAME.

    Class variables:
        LOADER_SCRIPT_PATH: Runfiles-relative path to the oci_load shell script.
        SERVICE_NAME: Expected service name in the health endpoint response.
        HAS_THIRD_PARTY_NOTICES: Whether the image includes THIRD-PARTY-NOTICES.txt.
    """

    LOADER_SCRIPT_PATH: str
    SERVICE_NAME: str
    HAS_THIRD_PARTY_NOTICES: bool = True

    container: ObjectsContainer
    image_name: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        """Loads the Docker image and starts the container."""
        cls.image_name = load_image(cls.LOADER_SCRIPT_PATH)
        logger.info("Using image: %s", cls.image_name)

        cls.container = ObjectsContainer(cls.image_name)
        cls.container.waiting_for(LogMessageWaitStrategy("Application startup complete").with_startup_timeout(600))
        cls.container.start()
        logger.info("Container is ready for tests")

    @classmethod
    def tearDownClass(cls) -> None:
        """Stops the shared container.

        The loaded Docker image is intentionally kept to preserve layer cache for
        subsequent runs. Cleanup is left to the caller's `docker image prune`.
        """
        if hasattr(cls, "container") and cls.container:
            cls.container.stop()
            logger.info("Container stopped")

    def test_container_starts_and_becomes_healthy(self) -> None:
        """Tests that the container starts and the health endpoint responds."""
        health_url = self.container.get_health_url()
        logger.info("Checking health endpoint: %s", health_url)

        max_retries = 10
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(health_url, timeout=5) as response:
                    data = json.loads(response.read().decode())

                    self.assertTrue(data.get("success"), f"Health check failed: {data}")  # type: ignore[attr-defined]
                    self.assertEqual(  # type: ignore[attr-defined]
                        data["data"]["status"],
                        "healthy",
                        f"Unexpected health status: {data}",
                    )
                    self.assertIn("version", data["data"])  # type: ignore[attr-defined]
                    self.assertEqual(data["data"]["service"], self.SERVICE_NAME)  # type: ignore[attr-defined]

                    logger.info("Health check passed: %s", data)
                    return
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                last_error = e
                logger.warning("Attempt %s/%s failed: %s", attempt + 1, max_retries, e)
                time.sleep(1)
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning("Attempt %s/%s failed: %s", attempt + 1, max_retries, e)
                time.sleep(1)

        self.fail(f"Health endpoint did not respond after {max_retries} attempts: {last_error}")  # type: ignore[attr-defined]

    def test_container_exposes_openapi_docs(self) -> None:
        """Tests that the container serves OpenAPI documentation."""
        host = self.container.get_container_host_ip()
        docs_url = f"http://{host}:{UVICORN_PORT}/openapi.json"

        logger.info("Checking OpenAPI docs: %s", docs_url)

        max_retries = 5
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(docs_url, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    self.assertIn("openapi", data)  # type: ignore[attr-defined]
                    self.assertIn("paths", data)  # type: ignore[attr-defined]
                    self.assertIn("/health", data["paths"])  # type: ignore[attr-defined]
                    self.assertIn("/process", data["paths"])  # type: ignore[attr-defined]
                    self.assertIn("/config", data["paths"])  # type: ignore[attr-defined]
                    logger.info("OpenAPI documentation is available")
                    return
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                logger.warning("Attempt %s/%s failed: %s", attempt + 1, max_retries, e)
                time.sleep(1)
            except json.JSONDecodeError as e:
                logger.warning("Attempt %s/%s failed: %s", attempt + 1, max_retries, e)
                time.sleep(1)

        self.fail("OpenAPI endpoint did not respond")  # type: ignore[attr-defined]

    def test_third_party_notices_matches_expectation(self) -> None:
        """Tests THIRD-PARTY-NOTICES.txt presence matches the image's HAS_THIRD_PARTY_NOTICES flag.

        Rather than skipping images that intentionally omit the notices, assert the file's presence
        or absence so a drift in either direction (missing when expected, or newly bundled) fails.
        """
        result = self.container.get_wrapped_container().exec_run("test -f /app/THIRD-PARTY-NOTICES.txt")
        present = result.exit_code == 0
        if self.HAS_THIRD_PARTY_NOTICES:
            self.assertTrue(  # type: ignore[attr-defined]
                present,
                "THIRD-PARTY-NOTICES.txt not found at /app/THIRD-PARTY-NOTICES.txt in the container image",
            )
        else:
            self.assertFalse(  # type: ignore[attr-defined]
                present,
                "THIRD-PARTY-NOTICES.txt unexpectedly present; set HAS_THIRD_PARTY_NOTICES=True if it is now bundled",
            )

    def test_container_stays_running(self) -> None:
        """Tests that the container stays running without crashing."""
        time.sleep(5)

        container_info = self.container.get_wrapped_container()
        container_info.reload()

        self.assertEqual(  # type: ignore[attr-defined]
            container_info.status,
            "running",
            f"Container stopped unexpectedly. Logs: {self.container.get_logs()}",
        )
        logger.info("Container stayed running for 5 seconds")
