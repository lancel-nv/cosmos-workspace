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

"""A lane line's predominant paint style, read from its RDS-HQ ``styles`` attribute.

RDS-HQ records a lane line's style per vertex (``SOLID_SINGLE``, ``LONG_DASHED_SINGLE``, ...). A dashed line's
periodic gaps leave much of its projected box unpainted, so its correspondence score runs far below a solid
line's even when it is rendered correctly, and an existence verdict has to know which kind of line it is judging.
This module is the one definition of that summary: the static object check records it on each lane-line track,
and scoring reads it back.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

LANE_LINE_OBJECT_TYPE = "LaneLine"

_DASHED_STYLE_MARKER = "dash"
_SOLID_STYLE_MARKER = "solid"


def modal_lane_style(styles: Any) -> str | None:
    """The most common non-empty value of a lane line's per-vertex ``styles``, or ``None`` when it carries none.

    Ties resolve to the value seen first. Styles are recorded per vertex, so this is a vertex count rather than a
    length-weighted vote: a line that changes style along its length reports the style on most of its vertices.
    """
    if not isinstance(styles, Sequence) or isinstance(styles, str):
        return None
    usable = [style for style in styles if isinstance(style, str) and style]
    return Counter(usable).most_common(1)[0][0] if usable else None


def is_dashed_style(lane_style: str | None) -> bool:
    """Whether ``lane_style`` names a dashed line: a dashed stroke and no solid one, case-insensitively.

    ``LONG_DASHED_SINGLE``, ``SHORT_DASHED_SINGLE`` and ``DOT_DASHED_SINGLE`` are dashed. A hybrid double line -
    ``SOLID_DASHED`` or ``DASHED_SOLID`` - is not: its solid stroke paints a continuous stripe, so it scores like a
    solid line. Measured on paired add-object trials, real hybrid lines average 0.57-0.67 against 0.05 for a
    long-dashed line, none falls below the solid cutoff, and the dashed cutoff misses a third of the hybrid lines a
    render omits.
    """
    style = str(lane_style or "").lower()
    return _DASHED_STYLE_MARKER in style and _SOLID_STYLE_MARKER not in style


def annotate_lane_styles(tracks: Sequence[dict[str, Any]], static_objects: Mapping[str, Mapping[str, Any]]) -> int:
    """Record each lane-line track's predominant style on its result entry, as ``lane_style``. Returns how many.

    ``tracks`` are result ``tracks`` entries (``track_id``, ``object_type``); ``static_objects`` is the loader's
    clip-level static-object map keyed by track id as a string. The style comes from the clip-level ``attributes``
    rather than any one frame's, since it is a property of the line. A lane line whose style is unrecorded gets no
    key, which a reader treats as unknown.
    """
    annotated = 0
    for entry in tracks:
        if entry.get("object_type") != LANE_LINE_OBJECT_TYPE:
            continue
        static_object = static_objects.get(str(entry.get("track_id"))) or {}
        attributes = static_object.get("attributes")
        style = modal_lane_style(attributes.get("styles") if isinstance(attributes, Mapping) else None)
        if style is not None:
            entry["lane_style"] = style
            annotated += 1
    return annotated
