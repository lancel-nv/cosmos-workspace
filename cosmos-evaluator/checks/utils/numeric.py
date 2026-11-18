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

"""Generic numeric guards shared across checkers and the metrics code that reads their results.

Pure, dependency-free scalar predicates, kept out of any domain module (e.g. traffic-light code) so every consumer
tests "is this a real number" one way instead of re-deriving an isinstance/NaN check inline.
"""

import math
from typing import Any


def is_finite(x: Any) -> bool:
    """True only for a real, non-NaN numeric value; ``None``, NaN, and non-numeric types return False.

    The ``isinstance`` gate is load-bearing: ``math.isnan`` raises ``TypeError`` on a non-float, so a bare
    ``not math.isnan(x)`` would crash on a stray string. ``bool`` is intentionally accepted (it is an ``int``
    subclass), matching the prior per-module ``_is_finite`` behavior.
    """
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))
