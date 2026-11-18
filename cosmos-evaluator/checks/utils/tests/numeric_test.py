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

"""Unit tests for the shared numeric guards (``checks.utils.numeric``)."""

import math
import unittest

from checks.utils.numeric import is_finite


class IsFiniteTest(unittest.TestCase):
    def test_real_numbers_are_finite(self) -> None:
        self.assertTrue(is_finite(0.0))
        self.assertTrue(is_finite(-3.5))
        self.assertTrue(is_finite(7))  # int
        self.assertTrue(is_finite(True))  # bool is an int subclass — intentionally accepted

    def test_nan_and_none_and_non_numeric_are_not_finite(self) -> None:
        self.assertFalse(is_finite(float("nan")))
        self.assertFalse(is_finite(None))
        self.assertFalse(is_finite("0.5"))  # the isinstance gate prevents a TypeError from math.isnan on a str
        self.assertFalse(is_finite([1.0]))

    def test_inf_is_finite_by_this_predicate(self) -> None:
        # This guard rejects only NaN (matching the prior per-module behavior); +/-inf pass. Documented, not a bug.
        self.assertTrue(is_finite(math.inf))
        self.assertTrue(is_finite(-math.inf))


if __name__ == "__main__":
    unittest.main()
