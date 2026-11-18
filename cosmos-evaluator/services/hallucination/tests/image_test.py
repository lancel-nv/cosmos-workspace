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

"""Integration test for the Hallucination container image.

This test loads and runs the Hallucination OCI image to verify it starts
successfully, responds to health checks, and can load its bundled configuration
(via the /config endpoint, which resolves cosmos_evaluator.yaml from the runfiles bundled
in the app layer).
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

# Configure logging
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


class HallucinationImageContainer(DockerContainer):
    """Custom container class for the Hallucination service."""

    def __init__(self, image: str, **kwargs: Any) -> None:
        """Initialize the Hallucination container."""
        super().__init__(image, **kwargs)
        self.with_bind_ports(UVICORN_PORT, UVICORN_PORT)
        self.with_env("UVICORN_PORT", str(UVICORN_PORT))
        self.with_env("UVICORN_HOST", "0.0.0.0")
        # Use staging env to avoid local env file loading
        self.with_env("COSMOS_EVALUATOR_ENV", "staging")
        self.with_env("COSMOS_EVALUATOR_STORAGE_TYPE", "local")

    def get_health_url(self) -> str:
        """Get the URL for the health endpoint."""
        host = self.get_container_host_ip()
        return f"http://{host}:{UVICORN_PORT}/health"

    def get_config_url(self) -> str:
        """Get the URL for the config endpoint."""
        host = self.get_container_host_ip()
        return f"http://{host}:{UVICORN_PORT}/config"


class TestHallucinationImage(unittest.TestCase):
    """Integration tests for the Hallucination container image."""

    container: HallucinationImageContainer
    image_name: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        """Loads the Docker image and start the container."""
        cls.image_name = load_image("services/hallucination/tests/test_image_load.sh")
        logger.info("Using image: %s", cls.image_name)

        # Start a shared container for all tests
        cls.container = HallucinationImageContainer(cls.image_name)
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

                    self.assertTrue(data.get("success"), f"Health check failed: {data}")
                    self.assertEqual(
                        data["data"]["status"],
                        "healthy",
                        f"Unexpected health status: {data}",
                    )
                    self.assertIn("version", data["data"])
                    self.assertEqual(data["data"]["service"], "Hallucination Service API")

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

        self.fail(f"Health endpoint did not respond after {max_retries} attempts: {last_error}")

    def test_config_endpoint_loads_bundled_config(self) -> None:
        """Tests that /config loads the bundled cosmos_evaluator.yaml from the app-layer runfiles in the container."""
        config_url = self.container.get_config_url()
        logger.info("Checking config endpoint: %s", config_url)

        max_retries = 5
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(config_url, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    self.assertTrue(data.get("success"), f"Config request failed: {data}")
                    self.assertIn("default_config", data["data"])
                    self.assertTrue(
                        data["data"]["default_config"],
                        f"Expected a non-empty default_config from bundled cosmos_evaluator.yaml: {data}",
                    )
                    logger.info("Config endpoint loaded bundled config successfully")
                    return
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                last_error = e
                logger.warning("Attempt %s/%s failed: %s", attempt + 1, max_retries, e)
                time.sleep(1)
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning("Attempt %s/%s failed: %s", attempt + 1, max_retries, e)
                time.sleep(1)

        self.fail(f"Config endpoint did not load bundled config after {max_retries} attempts: {last_error}")

    def test_container_exposes_openapi_docs(self) -> None:
        """Tests that the container serves OpenAPI documentation."""
        host = self.container.get_container_host_ip()
        docs_url = f"http://{host}:{UVICORN_PORT}/openapi.json"

        logger.info("Checking OpenAPI docs: %s", docs_url)

        max_retries = 5
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(docs_url, timeout=5) as response:
                    data = json.loads(response.read().decode())
                    self.assertIn("openapi", data)
                    self.assertIn("paths", data)
                    self.assertIn("/health", data["paths"])
                    self.assertIn("/process", data["paths"])
                    self.assertIn("/config", data["paths"])
                    logger.info("OpenAPI documentation is available")
                    return
            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                last_error = e
                logger.warning("Attempt %s/%s failed: %s", attempt + 1, max_retries, e)
                time.sleep(1)
            except json.JSONDecodeError as e:
                last_error = e
                logger.warning("Attempt %s/%s failed: %s", attempt + 1, max_retries, e)
                time.sleep(1)

        self.fail(f"OpenAPI endpoint did not respond after {max_retries} attempts: {last_error}")

    def test_third_party_notices_exists(self) -> None:
        """Tests that THIRD-PARTY-NOTICES.txt is present in the container image."""
        result = self.container.get_wrapped_container().exec_run("test -f /app/THIRD-PARTY-NOTICES.txt")
        self.assertEqual(
            result.exit_code,
            0,
            "THIRD-PARTY-NOTICES.txt not found at /app/THIRD-PARTY-NOTICES.txt in the container image",
        )

    def test_ffmpeg_and_ffprobe_present(self) -> None:
        """Tests that the packaged ffmpeg and ffprobe binaries are on PATH (the checker decodes via them)."""
        wrapped = self.container.get_wrapped_container()
        for binary in ("ffmpeg", "ffprobe"):
            exit_code, output = wrapped.exec_run([binary, "-version"])
            self.assertEqual(
                exit_code,
                0,
                f"{binary} not runnable in the image (exit {exit_code}): {output.decode()}. "
                "Ensure the custom FFmpeg layer (//services:ffmpeg_layer) is in the image and ldconfig runs at "
                "startup.",
            )
        logger.info("ffmpeg and ffprobe are present and runnable")

    def test_ffmpeg_has_all_shared_libraries(self) -> None:
        """Tests that the packaged FFmpeg binary has all shared library dependencies satisfied (ldconfig wired)."""
        wrapped = self.container.get_wrapped_container()

        which_exit, which_output = wrapped.exec_run(["which", "ffmpeg"])
        which_decoded = which_output.decode()
        self.assertEqual(which_exit, 0, f"could not resolve ffmpeg on PATH: {which_decoded}")
        ffmpeg_path = which_decoded.strip()

        exit_code, output = wrapped.exec_run(["ldd", ffmpeg_path])
        ldd_output = output.decode()
        self.assertEqual(exit_code, 0, f"ldd failed on ffmpeg binary {ffmpeg_path}: {ldd_output}")

        missing = [line.strip() for line in ldd_output.splitlines() if "not found" in line]
        self.assertEqual(
            missing,
            [],
            "FFmpeg binary has missing shared libraries:\n" + "\n".join(missing),
        )
        logger.info("All shared library dependencies for FFmpeg are satisfied")

    def test_container_stays_running(self) -> None:
        """Tests that the container stays running without crashing."""
        # Check that container is still running after 5 seconds
        time.sleep(5)

        # Verify container is still running
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
