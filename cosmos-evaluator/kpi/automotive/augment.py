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
"""Build the ``object_metrics`` block for a result JSON, and write it back into the file.

Reads a ``*.objects.results.json``, resolves a verdict for every object whose type the library has rules for,
and returns the block - or writes the file back with that one added top-level key. Everything the check wrote
is left byte-identical, so deleting that key returns the original file and a reviewer can see the augmentation
is purely additive.

The file's contents decide what gets scored: types the registry knows are evaluated, types it does not are
listed under ``unevaluated_object_types``. There is nothing to declare about which processor produced the
file, and so nothing to declare wrongly.

The block carries the policy that produced it, not just the numbers. Which cells count is a required choice
(see :class:`~kpi.automotive.result_view.ImportancePolicy`) and it moves the denominator of every rate, so a
verdict without its policy recorded beside it is not interpretable - and two augmented files are only
comparable if they used the same one.

:mod:`kpi.automotive.tools.object_metrics` is the command-line front end over this module.
"""

import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Any

from kpi.automotive.existence import (
    governing_object_type,
    is_conflicting,
    is_misaligned,
    is_missing,
    is_unscored,
    is_wrong_state,
    relevant_hallucinations,
)
from kpi.automotive.policy import SignalPolicy
from kpi.automotive.result_view import ResultView, read_result
from kpi.automotive.summary import signal_summary
from kpi.automotive.taxonomy import Group, kind_of
from services.utils import get_git_sha

logger = logging.getLogger(__name__)

METRICS_KEY = "object_metrics"
RESULT_GLOB = "*.objects.results.json"

# ``workspace_status.sh`` emits this literal when it cannot reach the git tree, so an unstamped build yields
# the sentinel rather than nothing at all.
UNSTAMPED_SENTINEL = "unknown"


def build_git_sha() -> str | None:
    """The commit this build was stamped from, or ``None`` when it was not stamped.

    Normalizes the sentinel to ``None`` so the emitted block has exactly one representation of "unavailable".
    Leaving both through would make every consumer handle two spellings of the same fact, and ``"unknown"``
    reads like a value.
    """
    sha = get_git_sha()
    return None if sha is None or sha == UNSTAMPED_SENTINEL or not sha.strip() else sha


def _track_entry(view: ResultView, track_id: int, policy: SignalPolicy) -> dict[str, Any]:
    """One object's verdicts, with ``null`` wherever nothing was measured."""
    track = view.tracks[track_id]
    unscored = is_unscored(track)
    return {
        "track_id": track.track_id,
        "object_type": track.object_type,
        "scored_frames": len(track.frame_scores),
        # JSON has no NaN literal, and emitting the non-standard token would break strict parsers; an
        # unscored track carries a null mean and ``unscored: true`` instead.
        "mean_score": None if unscored else track.mean_score,
        "unscored": unscored,
        "missing": is_missing(track, policy),
        "present": None if is_missing(track, policy) is None else not is_missing(track, policy),
        "misaligned": is_misaligned(track, policy),
        # Null for every type but TrafficLight, and for a light the checker sampled no color state for. The
        # aggregate in ``by_type`` cannot be reduced to these, so both are published: the counts say how many
        # lights were wrong, these say which.
        "wrong_state": is_wrong_state(track, policy),
        "conflicting": is_conflicting(track, policy),
        # The style that selected a lane line's existence cutoff; null for every other type, and for a lane line
        # whose result records none, which is judged by the solid-line cutoff.
        "lane_style": track.lane_style,
    }


def _extra_objects(view: ResultView, policy: SignalPolicy) -> list[dict[str, Any]]:
    """The hallucinations that count, deduplicated across groups and tagged with the group that kept them."""
    by_id: dict[int, dict[str, Any]] = {}
    for group in Group:
        for hallucination in relevant_hallucinations(view, group, policy):
            by_id.setdefault(
                hallucination.id,
                {
                    "id": hallucination.id,
                    "class": hallucination.class_label,
                    "group": str(group),
                    "relevancy": hallucination.relevancy,
                    "detections": len(hallucination.detections),
                },
            )
    return [by_id[key] for key in sorted(by_id)]


def _unevaluated_object_types(view: ResultView) -> list[str]:
    """Object types present in the file that this library has no rules for, so their absence self-explains."""
    return sorted(
        {
            track.object_type
            for track in view.tracks.values()
            if track.object_type is not None and kind_of(track.object_type) is None
        }
    )


def _unevaluated_hallucination_classes(view: ResultView) -> list[str]:
    """Hallucination class labels in the file that no rule governs.

    Reported separately from :func:`_unevaluated_object_types` because the two are different key spaces: a
    track carries an RDS object type, a hallucination carries whatever label the detector emitted, and a class
    can appear here without any ground-truth track of a corresponding type existing in the file. Without this,
    a file whose hallucinations were all dropped for want of a rule would be indistinguishable from one that
    had none.

    A malformed track with no class at all surfaces as an empty string rather than being filtered out - it is
    still a hallucination nothing scored.
    """
    return sorted({h.class_label for h in view.hallucinations if governing_object_type(h.class_label) is None})


def build_metrics(result: dict[str, Any], policy: SignalPolicy) -> dict[str, Any]:
    """The ``object_metrics`` block for one result payload.

    ``git_sha`` records the build these numbers came from, which is what makes a stale block identifiable:
    the rules live in code, so the commit is the only complete statement of what produced a verdict. It is
    ``null`` when the build was not stamped - rather than absent - so the field's presence does not itself
    vary between environments. See :func:`build_git_sha` for what "not stamped" covers.
    """
    view = read_result(result, importance=policy.importance)
    return {
        "git_sha": build_git_sha(),
        "policy": policy.as_dict(),
        "tracks": [_track_entry(view, track_id, policy) for track_id in sorted(view.tracks)],
        "by_type": {name: row.as_dict() for name, row in sorted(signal_summary(view, policy).items())},
        "extra_objects": _extra_objects(view, policy),
        "untyped_track_ids": sorted(view.untyped_track_ids),
        "unevaluated_object_types": _unevaluated_object_types(view),
        "unevaluated_hallucination_classes": _unevaluated_hallucination_classes(view),
    }


def _write_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` via a temp file in the same directory, then rename over it.

    An interrupted run over a directory of files must not leave a truncated result JSON behind: every file is
    either its old contents or its new ones.
    """
    descriptor, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
    except BaseException:
        # Includes KeyboardInterrupt: a half-written temp file left beside the real one is litter that a
        # later glob would pick up as an input.
        temp_path.unlink(missing_ok=True)
        raise


def augment_file(path: Path, destination: Path, policy: SignalPolicy) -> dict[str, Any]:
    """Augment one result file and write it to ``destination``. Returns the block that was written."""
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise TypeError(f"{path} does not contain a JSON object")

    # Recomputed rather than merged: the block is derived data, so re-running with a different policy must
    # replace the previous answer instead of leaving two policies' numbers side by side.
    metrics = build_metrics(result, policy)
    result[METRICS_KEY] = metrics
    _write_atomically(destination, result)

    unevaluated = metrics["unevaluated_object_types"]
    if unevaluated:
        logger.warning(
            "%s: no rules for object type(s) %s - they are absent from by_type rather than counted as zero",
            path.name,
            ", ".join(unevaluated),
        )
    unevaluated_classes = metrics["unevaluated_hallucination_classes"]
    if unevaluated_classes:
        logger.warning(
            "%s: no rules for hallucination class(es) %s - they are absent from extra_objects, not counted as zero",
            path.name,
            ", ".join(label or "<no class>" for label in unevaluated_classes),
        )
    return metrics
