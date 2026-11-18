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

"""Tests for the Arbitrator uvicorn entry point (``services.arbitrator.rest_api_common.main``).

``main()`` assembles the FastAPI app (``app = ArbitratorApp(settings=get_settings()).create_app()``) and hands it to
uvicorn, so it is the unit under test: it must wire ``get_settings()`` into ``ArbitratorApp`` and serve the built app.
The collaborators (settings, ``ArbitratorApp``, uvicorn) are patched so no real settings/YAML/runfiles/server are
needed. The app is built inside ``main()`` (not at module import) so code importing ``ArbitratorApp`` to subclass it
does not build a base app as a side effect.
"""

import unittest
from unittest.mock import MagicMock, patch

from services.arbitrator.rest_api import main


class TestRestApiEntrypoint(unittest.TestCase):
    def test_main_builds_app_from_settings_and_serves_it(self) -> None:
        fake_settings = MagicMock()
        # main() feeds settings.log_level.value into argparse's default and logging.basicConfig, which reject a
        # bare MagicMock, so pin a real level value.
        fake_settings.log_level.value = "INFO"
        fake_app = MagicMock(name="fastapi_app")
        app_instance = MagicMock()
        app_instance.create_app.return_value = fake_app
        app_cls = MagicMock(return_value=app_instance)

        with (
            patch("services.arbitrator.rest_api_common.get_settings", return_value=fake_settings) as mock_get_settings,
            patch("services.arbitrator.rest_api_common.ArbitratorApp", app_cls),
            patch("services.arbitrator.rest_api_common.uvicorn") as mock_uvicorn,
            patch("sys.argv", ["rest_api.py"]),
        ):
            main()

        mock_get_settings.assert_called_once()
        app_cls.assert_called_once_with(settings=fake_settings)
        app_instance.create_app.assert_called_once()
        # The assembled app is the object handed to uvicorn.run.
        mock_uvicorn.run.assert_called_once()
        self.assertIs(mock_uvicorn.run.call_args.args[0], fake_app)


if __name__ == "__main__":
    unittest.main()
