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

"""Per-object-type tabulation of the determinations in :mod:`kpi.automotive.existence`.

This is what a consumer reads off: one :class:`ObjectSignals` row per RDS object type, holding how many
objects of that type were observed, how many were present, missing or misaligned, and how many extra ones the
render invented.

Two conventions here are deliberate and easy to get wrong:

* **``None`` is not zero.** A count of ``None`` means no rule defines that signal for that type - misalignment
  is not defined for vulnerable road users - so no number may be reported. Defaulting it to 0 publishes a 0%
  misalignment rate for something that was never measured.
* **Correspondence and hallucination counts live in different key spaces, on purpose.** A correspondence row
  is keyed by the ground-truth RDS type (``Car``); a hallucination has no ground-truth object behind it to
  take a type from, so it is keyed by the detector class label (``Vehicle``). One class's numbers therefore
  land under two keys for dynamic types. Folding either way would lose real information - ``Car``/``Truck``
  granularity on one side, or the honest "we do not know which kind of vehicle this was" on the other.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from kpi.automotive.existence import (
    is_conflicting_over,
    is_misaligned,
    is_missing,
    is_unscored,
    is_wrong_state_over,
    relevant_hallucinations,
)
from kpi.automotive.policy import SignalPolicy
from kpi.automotive.result_view import ResultView
from kpi.automotive.taxonomy import Group, kind_of, state_block_object_type


@dataclass(frozen=True)
class ObjectSignals:
    """One row of the object-level metrics table."""

    object_type: str
    # Tracks of this type that were scored at least once under the view's importance policy. This is the
    # denominator of every rate below, and it moves with that policy - see ImportancePolicy.
    observed: int = 0
    # Registered but never scored: coverage, not quality. Excluded from ``observed`` deliberately.
    unscored: int = 0
    present: int = 0
    missing: int = 0
    misaligned: int | None = None  # ``None`` == no misalignment rule governs this type
    # Lights with color measurements that a color rule governs - the denominator for the two color signals.
    # Separate from ``observed`` because the color signals come from a different detector with its own
    # eligibility: the overlap check that fills ``observed`` gates on projected area, the state check gates on
    # edge truncation and segmentation mass, and neither population contains the other. Sharing one denominator
    # published a color rate over the geometry detector's population, which on real data was roughly double.
    color_observed: int = 0
    # Traffic lights only; ``None`` for every other type, which has no color state to be wrong about.
    wrong_state: int | None = None
    conflicting: int | None = None
    hallucinated: int = 0

    def rates(self) -> dict[str, float | None]:
        """Every signal's rate over ``observed``, or ``None`` where the signal was not measured.

        One rate per count, so a reader never has to divide by hand and never has to guess which denominator
        applies: it is ``observed`` for all of them.
        """
        return {
            "present_rate": _ratio(self.present, self.observed),
            "missing_rate": _ratio(self.missing, self.observed),
            "misaligned_rate": None if self.misaligned is None else _ratio(self.misaligned, self.observed),
            "wrong_state_rate": None if self.wrong_state is None else _ratio(self.wrong_state, self.color_observed),
            "conflicting_rate": None if self.conflicting is None else _ratio(self.conflicting, self.color_observed),
        }

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable record of this row, rates included."""
        return {
            "object_type": self.object_type,
            "observed": self.observed,
            "unscored": self.unscored,
            "present": self.present,
            "missing": self.missing,
            "misaligned": self.misaligned,
            "color_observed": self.color_observed,
            "wrong_state": self.wrong_state,
            "conflicting": self.conflicting,
            "hallucinated": self.hallucinated,
            **self.rates(),
        }


def _ratio(numerator: int, denominator: int) -> float | None:
    """``numerator / denominator``, or ``None`` when the denominator is zero (an unmeasured rate, not 0.0)."""
    if denominator <= 0:
        return None
    return numerator / denominator


@dataclass
class _Counters:
    """Mutable accumulator behind one :class:`ObjectSignals` row."""

    observed: int = 0
    unscored: int = 0
    present: int = 0
    missing: int = 0
    misaligned: int = 0
    misalignment_governed: bool = False
    color_observed: int = 0
    wrong_state: int = 0
    wrong_state_governed: bool = False
    conflicting: int = 0
    conflicting_governed: bool = False
    hallucinated: int = 0


def signal_summary(view: ResultView, policy: SignalPolicy) -> dict[str, ObjectSignals]:
    """Tabulate ``view`` under ``policy``, one row per object type seen.

    Types the registry does not know - every static type today - are skipped rather than scored by a
    substitute rule. A row appearing with ``observed == 0`` and a non-zero ``hallucinated`` is normal and
    informative: it means the render invented objects of a class the world model never contained.
    """
    counters: dict[str, _Counters] = {}

    for track in view.tracks.values():
        object_type = track.object_type
        if object_type is None or kind_of(object_type) is None:
            continue
        row = counters.setdefault(object_type, _Counters())
        if is_unscored(track):
            row.unscored += 1
            continue
        missing = is_missing(track, policy)
        # ``observed`` is the denominator of the existence rates, so it counts only tracks an existence rule
        # actually governed. A scored track whose type has no such rule contributes to neither the numerator
        # nor the denominator - reporting it as present would be exactly the substitute verdict this library
        # refuses to make.
        if missing is not None:
            row.observed += 1
            row.missing += int(missing)
            row.present += int(not missing)
        # Evaluated independently of existence: the two are separate questions, and a type can be governed by
        # one and not the other.
        misaligned = is_misaligned(track, policy)
        if misaligned is not None:
            row.misalignment_governed = True
            row.misaligned += int(misaligned)
        # Color-state signals are independent of existence: a light can be drawn and drawn wrong.

    # Color is counted over its own population, not over ``view.tracks``. A light is measured for color by a
    # different detector than the one that fills ``tracks`` - they gate on different criteria, so neither
    # population contains the other - and counting color over the geometry detector's roster both dropped
    # measured lights and put the rate on a denominator it was never measured against.
    for track_id, frames in view.state_frames.items():
        if not frames:
            continue
        known = view.tracks.get(track_id)
        # The file names a type only in ``tracks``; for a light absent from it the state block itself is the
        # evidence. Prefer the declared type where there is one.
        object_type = known.object_type if known is not None and known.object_type else state_block_object_type()
        wrong_state = is_wrong_state_over(frames, object_type, policy)
        conflicting = is_conflicting_over(frames, object_type, policy)
        if wrong_state is None and conflicting is None:
            # No color rule governs this type, so there is nothing to report. The row is created only after a
            # signal is measured: creating it first would publish an all-zero row for a type the existence loop
            # above deliberately skips, which reads as "measured and clean" rather than "not scored here".
            continue
        row = counters.setdefault(object_type, _Counters())
        row.color_observed += 1
        if wrong_state is not None:
            row.wrong_state_governed = True
            row.wrong_state += int(wrong_state)
        if conflicting is not None:
            row.conflicting_governed = True
            row.conflicting += int(conflicting)

    # Deduplicated by track id before counting: the dynamic groups do not overlap today, but a taxonomy that
    # ever let one hallucination match two groups would otherwise count it twice.
    label_by_id: dict[int, str] = {}
    for group in Group:
        for hallucination in relevant_hallucinations(view, group, policy):
            label_by_id[hallucination.id] = hallucination.class_label
    for class_label in label_by_id.values():
        # A track with no class label cannot be attributed to a type; it is dropped rather than filed under "".
        if class_label:
            counters.setdefault(class_label, _Counters()).hallucinated += 1

    return {
        object_type: ObjectSignals(
            object_type=object_type,
            observed=row.observed,
            unscored=row.unscored,
            present=row.present,
            missing=row.missing,
            misaligned=row.misaligned if row.misalignment_governed else None,
            color_observed=row.color_observed,
            wrong_state=row.wrong_state if row.wrong_state_governed else None,
            conflicting=row.conflicting if row.conflicting_governed else None,
            hallucinated=row.hallucinated,
        )
        for object_type, row in sorted(counters.items())
    }


def merge_summaries(target: dict[str, ObjectSignals], source: Mapping[str, ObjectSignals]) -> None:
    """Accumulate ``source`` into ``target`` in place, for tabulating across clips.

    ``misaligned`` merges as ``None`` unless at least one side measured it: summing a measured count into an
    unmeasured one would invent coverage the merged row does not have.
    """
    for object_type, row in source.items():
        existing = target.get(object_type)
        if existing is None:
            target[object_type] = row
            continue

        def merged(left: int | None, right: int | None) -> int | None:
            """Sum two counters, staying ``None`` unless at least one side actually measured the signal."""
            return None if left is None and right is None else (left or 0) + (right or 0)

        target[object_type] = ObjectSignals(
            object_type=object_type,
            observed=existing.observed + row.observed,
            color_observed=existing.color_observed + row.color_observed,
            unscored=existing.unscored + row.unscored,
            present=existing.present + row.present,
            missing=existing.missing + row.missing,
            misaligned=merged(existing.misaligned, row.misaligned),
            wrong_state=merged(existing.wrong_state, row.wrong_state),
            conflicting=merged(existing.conflicting, row.conflicting),
            hallucinated=existing.hallucinated + row.hallucinated,
        )
