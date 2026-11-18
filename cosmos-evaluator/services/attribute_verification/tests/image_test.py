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

"""Integration test for the Attribute Verification container image.

This test loads and runs the Attribute Verification OCI image to verify it starts
successfully, responds to health checks, and can load its bundled configuration
(via the /config endpoint, which resolves cosmos_evaluator.yaml from the runfiles
bundled in the app layer).
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


class AttributeVerificationImageContainer(DockerContainer):
    """Custom container class for the Attribute Verification service."""

    def __init__(self, image: str, **kwargs: Any) -> None:
        """Initialize the Attribute Verification container."""
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


class TestAttributeVerificationImage(unittest.TestCase):
    """Integration tests for the Attribute Verification container image."""

    container: AttributeVerificationImageContainer
    image_name: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        """Loads the Docker image and start the container."""
        cls.image_name = load_image("services/attribute_verification/tests/test_image_load.sh")
        logger.info("Using image: %s", cls.image_name)

        # Start a shared container for all tests
        cls.container = AttributeVerificationImageContainer(cls.image_name)
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
                    self.assertEqual(data["data"]["service"], "Attribute Verification Service API")

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

    def test_system_trust_store_has_certificates(self) -> None:
        """The packaged Python must find system CAs without a host mount or SSL_CERT_FILE override."""
        wrapped = self.container.get_wrapped_container()
        exit_code, output = wrapped.exec_run(
            [
                "/bin/sh",
                "-c",
                'python_path=$(find /app/rest_api_deployment.runfiles -path "*/bin/python3" -print -quit); '
                'env -u SSL_CERT_FILE -u SSL_CERT_DIR "$python_path" -c '
                "'import ssl; context = ssl.create_default_context(); "
                "assert context.get_ca_certs(), context.cert_store_stats()'",
            ],
        )
        self.assertEqual(exit_code, 0, f"Python cannot load the system trust store: {output.decode()}")

    def test_dpkg_attributes_openssl_files_to_their_package(self) -> None:
        """Scanners match /usr/bin/openssl against the patched Ubuntu package only if dpkg records it as that package's.

        A file no package claims is reported at the upstream version embedded in the binary instead, which misses
        Ubuntu's backported fixes. The ``.md5sums`` that ``dpkg --verify`` checks are also what syft reads ownership
        from.
        """
        wrapped = self.container.get_wrapped_container()

        exit_code, output = wrapped.exec_run(["dpkg", "-S", "/usr/bin/openssl"])
        self.assertEqual(exit_code, 0, f"dpkg -S /usr/bin/openssl failed: {output.decode()}")
        self.assertEqual(output.decode().strip(), "openssl: /usr/bin/openssl")

        exit_code, output = wrapped.exec_run(["dpkg", "--verify", "openssl", "ca-certificates"])
        self.assertEqual((exit_code, output.decode()), (0, ""), "dpkg --verify reported missing or modified files")

    def test_ffmpeg_has_all_shared_libraries(self) -> None:
        """Tests that the packaged FFmpeg binary has all shared library dependencies satisfied."""
        wrapped = self.container.get_wrapped_container()

        exit_code, output = wrapped.exec_run(["ldd", "/usr/local/bin/ffmpeg"])
        ldd_output = output.decode()
        self.assertEqual(exit_code, 0, f"ldd failed on ffmpeg binary: {ldd_output}")

        missing = [line.strip() for line in ldd_output.splitlines() if "not found" in line]
        self.assertEqual(
            missing,
            [],
            "FFmpeg binary has missing shared libraries:\n" + "\n".join(missing),
        )
        logger.info("All shared library dependencies for FFmpeg are satisfied")

    def test_ffmpeg_has_expected_encoders(self) -> None:
        """Tests the packaged FFmpeg build has the whole encode ladder, and no software H.264.

        Membership only: exactness is owned by //third_party/ffmpeg/tests:ffmpeg_encoders_test, which pins the full set
        against the tarball on a CPU-only host. What this adds is that the layer actually reached *this* image.
        """
        wrapped = self.container.get_wrapped_container()

        exit_code, output = wrapped.exec_run(["/usr/local/bin/ffmpeg", "-encoders"])
        encoder_output = output.decode()
        self.assertEqual(exit_code, 0, f"ffmpeg -encoders failed: {encoder_output}")

        # Isolate the encoder-name column (2nd whitespace-delimited token of each row after the "------" divider) so the
        # absence check for libopenh264 can't be fooled by that name appearing in another encoder's description text.
        encoder_names: set[str] = set()
        in_table = False
        for line in encoder_output.splitlines():
            if not in_table:
                in_table = line.lstrip().startswith("------")
                continue
            fields = line.split()
            if len(fields) >= 2:
                encoder_names.add(fields[1])

        for expected in ("h264_nvenc", "av1_nvenc", "libsvtav1"):
            self.assertIn(
                expected,
                encoder_names,
                f"FFmpeg does not have the {expected} encoder. "
                "Ensure the custom FFmpeg layer (//services:ffmpeg_layer) is included in the image.",
            )
        # The self-built OpenH264 software H.264 encoder is not supported for patent reasons
        self.assertNotIn("libopenh264", encoder_names, "libopenh264 should not exist in the FFmpeg build")
        logger.info("FFmpeg encode ladder (h264_nvenc, av1_nvenc, libsvtav1) is available and libopenh264 is absent")

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
