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

"""Ground-truth primitives for traffic-light state and wrong-color analysis.

Pure algorithms over a light's per-frame ``traffic_light_state`` rows - episode segmentation and
consecutive-run counting - with no policy in them: the thresholds and what a trip *means* belong to whatever
questionnaire or metric is asking. Keeping them here lets the checks, the KPI library and any reporting tool
share one implementation without any of them importing another's layer.

A light cycles color over a clip, so there is no single expected color for a track; the ground truth is per
frame, in each row's ``expected_state``. That is what :func:`gt_color_episodes` partitions on.
"""

from collections.abc import Iterator
import math
from typing import Any

from checks.utils.numeric import is_finite

CONF_THR = 0.5


def isnan(x: float | None) -> bool:
    return isinstance(x, float) and math.isnan(x)


def toks(v: object) -> set[str]:
    """Parse a comma-separated multi-select answer string into its lowercased token set, dropping empties/``none``."""
    return {t.strip().lower() for t in str(v or "").split(",") if t.strip() and t.strip().lower() != "none"}


def gt_color_episodes(frames: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Partition a light's per-frame ``traffic_light_state`` into maximal runs of constant GT ``expected_state``.

    An **episode** is a period of constant ground-truth color. A traffic light cycles (green -> yellow -> red), so a
    single per-track expected color is meaningless; the ground truth is per frame. Splitting into single-color
    episodes lets wrong-color detection be evaluated -- and named -- per phase, so a light mis-rendered only in its
    yellow phase is attributed to ``yellow`` rather than the clip-dominant color. Returns ``[(gt_color, [frame, ...])]``
    ordered by frame index.

    ``traffic_light_state`` is **sparse**: the static checker only appends a row for frames it actually scored, so a
    light that is skipped/occluded for a stretch leaves a gap in ``frame_idx``. Two same-color spans separated by such
    a gap are *distinct* episodes -- merging them would average an out-of-view stretch into the score and misreport the
    episode's ``frame_start``/``frame_end`` -- so we also start a new episode whenever ``frame_idx`` is not contiguous
    with the previous frame.
    """
    episodes: list[tuple[str, list[dict[str, Any]]]] = []
    prev_idx: int | None = None
    for f in sorted(frames, key=lambda x: x.get("frame_idx", 0)):
        color = str(f.get("expected_state") or "").lower()
        idx = f.get("frame_idx")
        contiguous = prev_idx is not None and idx is not None and idx == prev_idx + 1
        if episodes and episodes[-1][0] == color and contiguous:
            episodes[-1][1].append(f)
        else:
            episodes.append((color, [f]))
        prev_idx = idx
    return episodes


def _consec_runs(frame_idxs: list[int]) -> Iterator[tuple[int, int]]:
    if not frame_idxs:
        return
    s = p = frame_idxs[0]
    for x in frame_idxs[1:]:
        if x == p + 1:
            p = x
        else:
            yield (s, p)
            s = p = x
    yield (s, p)


def max_consec(frame_idxs: list[int]) -> int:
    best = 0
    for s, e in _consec_runs(sorted(set(frame_idxs))):
        best = max(best, e - s + 1)
    return best


def episode_state_trip(
    ep_frames: list[dict[str, Any]], state_threshold: float, min_wrong_frames: int
) -> tuple[str | None, float | None]:
    """Whether one GT-color episode is wrong-state, plus its per-frame state-score nanmean.

    Returns ``(rule, mean_state)``: ``rule`` is ``"mean"`` when the episode nanmean is below ``state_threshold``,
    else ``"consec"`` when the episode has ``>= min_wrong_frames`` consecutive-``frame_idx`` frames below threshold,
    else ``None``. ``mean_state`` is ``None`` when the episode has no finite score. This is the one wrong-state
    predicate every caller shares — keep it that way so they all report the same recall.
    """
    scores = [f["traffic_light_state_score"] for f in ep_frames if is_finite(f.get("traffic_light_state_score"))]
    mean_state = sum(scores) / len(scores) if scores else None
    if mean_state is not None and mean_state < state_threshold:
        return "mean", mean_state
    low_idxs = [
        f["frame_idx"]
        for f in ep_frames
        if f.get("frame_idx") is not None
        and is_finite(f.get("traffic_light_state_score"))
        and f["traffic_light_state_score"] < state_threshold
    ]
    if max_consec(low_idxs) >= min_wrong_frames:
        return "consec", mean_state
    return None, mean_state


def conflicting_run(frames: list[dict[str, Any]], threshold: float) -> tuple[int, float | None]:
    """Longest run of contiguous-``frame_idx`` frames whose ``traffic_light_multi_color_score`` exceeds ``threshold``,
    plus the track's peak multi-color score (``None`` when no finite score). The one conflicting-signal predicate
    every caller shares — it catches a brief burst that the per-track mean washes out."""
    over_idxs = [
        f["frame_idx"]
        for f in frames
        if f.get("frame_idx") is not None
        and is_finite(f.get("traffic_light_multi_color_score"))
        and f["traffic_light_multi_color_score"] > threshold
    ]
    finite = [s for f in frames if (s := f.get("traffic_light_multi_color_score")) is not None and is_finite(s)]
    return max_consec(over_idxs), (max(finite) if finite else None)
