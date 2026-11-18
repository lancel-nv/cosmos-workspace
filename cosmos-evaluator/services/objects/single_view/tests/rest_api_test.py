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

"""Tests for the Objects REST API app + entry point (rest_api_common.py)."""

import sys
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from services.objects.single_view.rest_api import main
from services.objects.single_view.rest_api_common import app


class TestHealthEndpoint(unittest.TestCase):
    """The /health endpoint surfaces the service git SHA."""

    def test_health_endpoint_returns_git_sha(self) -> None:
        """/health returns the SHA from get_git_sha, which is mocked so no git_sha runfile is needed."""
        with patch("services.utils.get_git_sha", return_value="a" * 40):
            client = TestClient(app)
            self.addCleanup(client.close)
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["git_sha"], "a" * 40)


class TestRestApiMain(unittest.TestCase):
    """Tests for main() that boots the Objects Single-View API under uvicorn."""

    def test_main_parses_args_and_runs_uvicorn(self) -> None:
        """main() parses CLI args and hands the FastAPI app to uvicorn.run."""
        argv = ["rest_api", "--port", "8085"]
        with (
            patch("uvicorn.run") as mock_run,
            patch.object(sys, "argv", argv),
        ):
            main()
        # The server is launched exactly once with the configured port.
        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args.kwargs["port"], 8085)


if __name__ == "__main__":
    unittest.main()
