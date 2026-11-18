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

"""Shared access to the dynamic-mode result fixture the metrics tests read.

The fixture mirrors the shape of a real *.dynamic.objects.results.json (sparse_compact matrix with an
importance column, per-track importance blocks, hallucination tracks holding indices into per-class
detection lists) while staying small enough to reason about cell by cell. It deliberately contains one track
that is scored only in non-important frames, so every test can exercise both importance policies against the
same file.
"""

import json
from pathlib import Path
from typing import Any

from utils.bazel import get_runfiles_path

FIXTURE_NAME = "dynamic_objects_results.json"
TRAFFIC_LIGHT_FIXTURE_NAME = "traffic_light_results.json"
FIXTURE_PACKAGE = "kpi/automotive/tests"


def _load(name: str) -> dict[str, Any]:
    """Parse the fixture ``name`` out of the test target's runfiles.

    Resolved through runfiles rather than ``Path(__file__).parent`` so the fixture is found by the label the
    ``data`` attribute declares, not by where the source file happens to sit at runtime - the two differ
    whenever cosmos_eval is consumed as a ``bazel_dep`` rather than being the root module. A missing file
    raises: it means the target's ``data`` is wrong, which must fail loudly rather than degrade.
    """
    resolved = get_runfiles_path(f"{FIXTURE_PACKAGE}/{name}")
    if resolved is None:
        raise FileNotFoundError(f"{name} is not in the runfiles; is it declared in the target's data?")
    with Path(resolved).open("r", encoding="utf-8") as handle:
        result: dict[str, Any] = json.load(handle)
    return result


def load_dynamic_result() -> dict[str, Any]:
    """The dynamic-mode result fixture, parsed."""
    return _load(FIXTURE_NAME)


def load_traffic_light_result() -> dict[str, Any]:
    """A static-mode result fixture carrying four traffic lights.

    Built so each light isolates one behavior: a clean light, one whose box was drawn empty, one whose color
    is wrong only in its second GT-color episode, and one showing two colors at once for a sustained run. The
    per-frame ``frame_idx`` values are spaced 3 apart, matching a 30 FPS run against 10 FPS annotations - so a
    reader that skips the ordinal normalization sees every episode as one frame long and scores nothing.
    """
    return _load(TRAFFIC_LIGHT_FIXTURE_NAME)
