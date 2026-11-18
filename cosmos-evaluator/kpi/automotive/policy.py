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

"""Which rule applies to which object type, and the canonical set of them.

A :class:`SignalPolicy` maps each RDS object type to the rule deciding each signal for it. Per-type keying is
the extension point: installing a different rule for one type is a dict entry, not a fork. A type absent from
a map has **no rule** for that signal, which the predicates report as ``None`` - "not measured" - rather than
as a zero.

The rules themselves live in :mod:`kpi.automotive.rules`, which is also where to look when writing your own.
This module only decides which rule goes where, and records that decision: :meth:`SignalPolicy.as_dict` stamps
the policy's name and every resolved rule alongside whatever it produced, so a number is always traceable to
the rules that made it - including rules this library did not define.

:func:`default_policy` is the canonical ruleset for every registered type. Build a variant from it with
``dataclasses.replace`` and give it its own :attr:`SignalPolicy.name`, so a swept threshold is distinguishable
from the default in whatever the numbers land in.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from checks.utils.lane_style import LANE_LINE_OBJECT_TYPE, is_dashed_style
from kpi.automotive.result_view import ImportancePolicy
from kpi.automotive.rules import (
    ConflictingColors,
    DrawnEmpty,
    HallucinationRule,
    MeanBelow,
    PercentileBetween,
    RelevancyGate,
    StateRule,
    TrackRule,
    WrongColorEpisode,
)
from kpi.automotive.taxonomy import KINDS, Group, Signal, defines_signal

# The ``existence`` key holding the rule for a lane line drawn in a dashed style. A policy key rather than an RDS
# object type: :meth:`SignalPolicy.existence_rule` selects it from the type and the line's style.
DASHED_LANE_LINE = "LaneLine:dashed"


@dataclass(frozen=True)
class SignalPolicy:
    """The complete set of rules one metrics run is computed under.

    Instances are frozen and self-describing: :meth:`as_dict` returns everything needed to reproduce a number,
    which is what should be stamped alongside any published metric so a rule change shows up as a diff rather
    than as unexplained drift.
    """

    name: str  # identifies this ruleset in stamped output: "default", or a swept variant's own name
    importance: ImportancePolicy  # which cells count; see ImportancePolicy for why it is never defaulted
    # By RDS object type, plus DASHED_LANE_LINE for a dashed lane line; an absent type has no existence rule.
    existence: Mapping[str, TrackRule]
    misalignment: Mapping[str, TrackRule]  # by RDS object type; an absent type has no misalignment rule
    hallucination: Mapping[str, HallucinationRule]  # by RDS object type; an absent type counts none
    wrong_state: Mapping[str, StateRule]  # traffic lights only; an absent type has no color-state rule
    conflicting: Mapping[str, StateRule]  # traffic lights only; an absent type has no conflict rule

    def existence_rule(self, object_type: str | None, lane_style: str | None = None) -> TrackRule | None:
        """The existence rule for ``object_type``, or ``None`` when no rule defines that signal for it.

        A lane line whose ``lane_style`` is dashed takes the rule installed under :data:`DASHED_LANE_LINE`, and its
        type's own rule when the policy installs none there. ``lane_style`` is ignored for every other type.
        """
        if object_type is None:
            return None
        if object_type == LANE_LINE_OBJECT_TYPE and is_dashed_style(lane_style) and DASHED_LANE_LINE in self.existence:
            return self.existence[DASHED_LANE_LINE]
        return self.existence.get(object_type)

    def misalignment_rule(self, object_type: str | None) -> TrackRule | None:
        """The misalignment rule for ``object_type``, or ``None`` when no rule defines that signal for it."""
        return None if object_type is None else self.misalignment.get(object_type)

    def hallucination_rule(self, object_type: str | None) -> HallucinationRule | None:
        """The hallucination rule for ``object_type``, or ``None`` when no rule defines that signal for it."""
        return None if object_type is None else self.hallucination.get(object_type)

    def wrong_state_rule(self, object_type: str | None) -> StateRule | None:
        """The wrong-color rule for ``object_type``, or ``None`` when no rule defines that signal for it."""
        return None if object_type is None else self.wrong_state.get(object_type)

    def conflicting_rule(self, object_type: str | None) -> StateRule | None:
        """The conflicting-signal rule for ``object_type``, or ``None`` when no rule defines it."""
        return None if object_type is None else self.conflicting.get(object_type)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable record of this policy, for stamping into whatever the metrics feed.

        Each rule serializes itself, so a caller's own rule appears here with its own parameters rather than
        being flattened to an unrecognized placeholder.
        """
        return {
            "name": self.name,
            "importance": str(self.importance),
            "existence": {name: rule.as_dict() for name, rule in sorted(self.existence.items())},
            "misalignment": {name: rule.as_dict() for name, rule in sorted(self.misalignment.items())},
            "hallucination": {name: rule.as_dict() for name, rule in sorted(self.hallucination.items())},
            "wrong_state": {name: rule.as_dict() for name, rule in sorted(self.wrong_state.items())},
            "conflicting": {name: rule.as_dict() for name, rule in sorted(self.conflicting.items())},
        }


DEFAULT_POLICY_NAME = "default"

# The zero-score tolerance is an in-code constant rather than a tunable: it distinguishes "exactly zero" from
# floating-point noise around zero, and is not a knob whose value trades precision against recall.
DEFAULT_EXISTENCE = DrawnEmpty()

# Lane lines and road boundaries: missing when the mean over every scored frame falls below 0.1, or below 0.002 for a
# lane line drawn in a dashed style, whose periodic gaps leave most of its box unpainted. Both cutoffs were tuned on
# paired add-object perturbation trials, a world-model copy of a real line against the real line it was copied from,
# over ~2000 lane-line and ~2000 road-boundary trials.
DEFAULT_LANE_GRAPH_EXISTENCE = MeanBelow(threshold=0.1)
DEFAULT_DASHED_LANE_EXISTENCE = MeanBelow(threshold=0.002)

# Misalignment: the 85th percentile of a track's scored frames strictly between 0 and 0.35. The percentile is
# high enough that a well-rendered object clears it on its best frames alone, and low enough that a handful of
# good frames cannot rescue an object that is mostly outside its box; the upper bound separates a defect from
# a merely imperfect projection, and the lower bound keeps the verdict disjoint from the existence rule. These
# are the values the rule has been characterized at, so sweep by building a named :class:`SignalPolicy`
# variant rather than by editing here, and the number stays attached to the rules that produced it.
DEFAULT_MISALIGNMENT = PercentileBetween(percentile=85.0, lower=0.0, upper=0.35)

# A hallucination counts once its box has overlapped the drivable surface in at least one frame.
DEFAULT_HALLUCINATION = RelevancyGate(min_relevancy=1)

# Color state. The threshold is the share of CWIP probability mass that has to sit on the expected color; the
# two-frame minimum is what separates a sustained wrong-color burst from a single noisy sample.
DEFAULT_WRONG_STATE = WrongColorEpisode(state_score_threshold=0.5, min_wrong_frames=2)
DEFAULT_CONFLICTING = ConflictingColors(multi_color_threshold=0.5, min_conflicting_frames=2)


def default_policy(importance: ImportancePolicy) -> SignalPolicy:
    """The canonical rules for every registered object type, under an explicitly chosen importance policy.

    A function rather than a module-level constant precisely because ``importance`` has no defensible default
    (see :class:`~kpi.automotive.result_view.ImportancePolicy`); materializing one choice as a constant would
    make it the path of least resistance.

    Each map is populated only for the types whose taxonomy entry defines that signal, so "no rule" and "not
    defined for this type" are the same absence rather than two things that can disagree.
    """
    existence: dict[str, TrackRule] = {}
    misalignment: dict[str, TrackRule] = {}
    hallucination: dict[str, HallucinationRule] = {}
    wrong_state: dict[str, StateRule] = {}
    conflicting: dict[str, StateRule] = {}
    for rds_type in sorted(KINDS):
        if defines_signal(rds_type, Signal.MISSING):
            lane_graph = KINDS[rds_type].group is Group.LANE_GRAPH
            existence[rds_type] = DEFAULT_LANE_GRAPH_EXISTENCE if lane_graph else DEFAULT_EXISTENCE
        if defines_signal(rds_type, Signal.MISALIGNED):
            misalignment[rds_type] = DEFAULT_MISALIGNMENT
        if defines_signal(rds_type, Signal.EXTRA):
            hallucination[rds_type] = DEFAULT_HALLUCINATION
        if defines_signal(rds_type, Signal.WRONG_STATE):
            wrong_state[rds_type] = DEFAULT_WRONG_STATE
        if defines_signal(rds_type, Signal.CONFLICTING):
            conflicting[rds_type] = DEFAULT_CONFLICTING
    if LANE_LINE_OBJECT_TYPE in existence:
        existence[DASHED_LANE_LINE] = DEFAULT_DASHED_LANE_EXISTENCE
    return SignalPolicy(
        name=DEFAULT_POLICY_NAME,
        importance=importance,
        existence=existence,
        misalignment=misalignment,
        hallucination=hallucination,
        wrong_state=wrong_state,
        conflicting=conflicting,
    )
