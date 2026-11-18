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

"""Tests for the lcov_to_cobertura conversion entrypoint."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from tools.coverage import lcov_to_cobertura

MINIMAL_LCOV = """TN:
SF:example.py
DA:1,1
DA:2,0
LF:2
LH:1
end_of_record
"""


class TestLcovToCobertura(unittest.TestCase):
    def test_module_imports(self) -> None:
        self.assertIsNotNone(lcov_to_cobertura)

    def test_converts_minimal_lcov_to_cobertura(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lcov_path = Path(tmp) / "coverage.dat"
            out_path = Path(tmp) / "coverage.xml"
            lcov_path.write_text(MINIMAL_LCOV, encoding="utf-8")
            argv = ["lcov_to_cobertura", "--input", str(lcov_path), "--output", str(out_path)]
            with mock.patch.object(sys, "argv", argv):
                lcov_to_cobertura.main()
            self.assertTrue(out_path.exists())
            self.assertIn("<coverage", out_path.read_text(encoding="utf-8"))

    def test_missing_input_exits_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # "no_such_dir" is never created, so the input path is guaranteed missing without depending on host layout.
            missing_path = Path(tmp) / "no_such_dir" / "coverage.dat"
            argv = ["lcov_to_cobertura", "--input", str(missing_path)]
            with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as ctx:
                lcov_to_cobertura.main()
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
