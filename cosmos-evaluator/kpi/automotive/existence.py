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

"""Per-object determinations: is this world-model object present, missing, or misaligned, and what was extra.

Everything here is a pure function of a :class:`~kpi.automotive.result_view.TrackView` and a
:class:`~kpi.automotive.policy.SignalPolicy`, so a verdict depends on nothing but the result file and the
rules it was computed under. The rules themselves decide; this module resolves which rule applies, guarantees
its inputs, and turns "no rule" into a reportable answer.

Each predicate returns ``bool | None``, and ``None`` is load-bearing: it means *no rule defines this signal
for this object type*, which is not the same as ``False``. Misalignment is not defined for vulnerable road
users by default, so ``is_misaligned`` on a pedestrian is ``None``; reporting it as ``False`` would publish a
0% misalignment rate for a signal that was never measured.
"""

from collections.abc import Sequence
import math

from checks.utils.object_class_names import object_class_matches
from kpi.automotive.policy import SignalPolicy
from kpi.automotive.result_view import HallucinationView, ResultView, ScoredFrame, StateFrame, TrackView
from kpi.automotive.taxonomy import Group, hallucination_label_for, kind_of, types_in_group


def is_unscored(track: TrackView) -> bool:
    """Whether ``track`` was never scored under the view's importance policy.

    An unscored track is a **coverage** fact, not a quality one: the object was registered but the algorithm
    never got to evaluate it (filtered before scoring, outside the processed frame range, or never ego-relevant
    under :data:`~kpi.automotive.result_view.ImportancePolicy.EGO_RELEVANT`). It must be excluded from
    present/missing counts rather than counted as missing, or the metric measures coverage instead of
    generation quality.

    Resolving this before any rule is consulted is what lets a rule take a non-empty, NaN-free score sequence
    as a precondition rather than having to decide what an absent measurement means.
    """
    return not track.frame_scores or math.isnan(track.mean_score)


def is_present(track: TrackView, policy: SignalPolicy) -> bool | None:
    """Whether the render actually drew ``track``'s object.

    ``None`` when the track is unscored or no existence rule is installed for its type. Under the default
    rule an object counts as present whenever its scores do not average to ~0, however low - a mean of 0.08 is
    a badly rendered object, not an absent one.
    """
    missing = is_missing(track, policy)
    return None if missing is None else not missing


def is_missing(track: TrackView, policy: SignalPolicy) -> bool | None:
    """Whether the world model contained ``track``'s object but the render did not draw it.

    ``None`` when the track is unscored, or no existence rule is installed for its type.
    """
    if is_unscored(track):
        return None
    rule = policy.existence_rule(track.object_type, track.lane_style)
    return None if rule is None else rule.holds(track.frame_scores)


def is_missing_over(
    frames: Sequence[ScoredFrame], object_type: str | None, policy: SignalPolicy, lane_style: str | None = None
) -> bool | None:
    """:func:`is_missing` over an arbitrary slice of a track's frames, e.g. one distance band.

    The band-sliced form exists because a banded metric asks the same question of a subset of frames, and
    re-deriving the rule at that call site is exactly how two consumers of one rule drift apart.

    Slicing keeps whole :class:`~kpi.automotive.result_view.ScoredFrame` values rather than bare scores so the
    rule sees the same frame identity it would on a full track. Note the ordinals of a slice are those of the
    parent track, so a band's frames are adjacent only where they were adjacent in the clip - which is what a
    rule testing contiguity should see. ``lane_style`` is the style :func:`is_missing` reads off the track, supplied
    by the caller here.
    """
    if not frames:
        return None
    rule = policy.existence_rule(object_type, lane_style)
    return None if rule is None else rule.holds(frames)


def is_misaligned(track: TrackView, policy: SignalPolicy) -> bool | None:
    """Whether ``track``'s object was drawn only partly inside its bounding box.

    ``None`` when the track is unscored, or no misalignment rule is installed for its type - which by default
    is every vulnerable road user.

    An object the existence rule calls missing is never also misaligned: it was not drawn, so it cannot have
    been drawn badly. Establishing that here rather than relying on the two rules' bounds not to overlap is
    what makes the exclusion hold for a caller's rules too - a misalignment rule cannot see what the existence
    rule decided, and two independently reasonable rules can otherwise both fire on a near-zero track.
    """
    if is_unscored(track):
        return None
    rule = policy.misalignment_rule(track.object_type)
    if rule is None:
        return None
    if is_missing(track, policy):
        return False
    return rule.holds(track.frame_scores)


def is_wrong_state(track: TrackView, policy: SignalPolicy) -> bool | None:
    """Whether ``track``'s light was rendered in a color the ground truth did not call for.

    ``None`` when the track is unscored, no wrong-color rule is installed for the type, the light has no state
    frames at all - a light the checker never sampled is unmeasured, not correct - or the rule itself declines,
    as the default one does for frames naming no ground-truth color. Note this is independent of the existence
    *rule*: a light can be drawn (so not missing) and drawn the wrong color.

    The unscored gate is not redundant with the state-frame check. State frames are parsed straight off the
    file, so they survive the importance filter that decides which cells count as scores; without the gate a
    light outside the selected population would still return a color verdict, putting the numerator and the
    denominator of any rate built from it on different populations.
    """
    if is_unscored(track):
        return None
    rule = policy.wrong_state_rule(track.object_type)
    if rule is None or not track.state_frames:
        return None
    return rule.holds(track.state_frames)


def is_conflicting(track: TrackView, policy: SignalPolicy) -> bool | None:
    """Whether ``track``'s light shows two colors at once for a sustained run of frames.

    ``None`` when the track is unscored, no conflict rule is installed for the type, or the light has no state
    frames. The unscored gate is there for the same population reason as in :func:`is_wrong_state`.
    """
    if is_unscored(track):
        return None
    rule = policy.conflicting_rule(track.object_type)
    if rule is None or not track.state_frames:
        return None
    return rule.holds(track.state_frames)


def is_wrong_state_over(frames: Sequence[StateFrame], object_type: str | None, policy: SignalPolicy) -> bool | None:
    """:func:`is_wrong_state` over an arbitrary run of state frames, e.g. one color episode.

    Takes the frames and the type rather than a :class:`~kpi.automotive.result_view.TrackView`, for two
    reasons the track-shaped form cannot serve.

    **Coverage.** A ``TrackView`` exists only for an entry in the file's ``tracks`` array, which is the roster
    of objects the *correspondence* pass scored - but color state is a separate measurement channel with its
    own, wider coverage. Over 120 real result files, 466 of 821 measured lights had no ``tracks`` entry at all.
    Reading color through the track view would answer for 43% of the population and silently skip the rest.
    Pair this with :func:`~kpi.automotive.result_view.state_frames_by_track`, which is keyed off the state
    block and therefore sees every measured light.

    **Granularity.** A per-track verdict cannot say *which* color phase was wrong, and cannot judge one run's
    frames against another run's episode boundaries. Handing the rule a slice does both.

    ``object_type`` is asserted by the caller, exactly as in :func:`is_missing_over`. There is no unscored gate
    here: passing a slice *is* the population choice, and re-imposing one would override it.
    """
    if not frames:
        return None
    rule = policy.wrong_state_rule(object_type)
    return None if rule is None else rule.holds(frames)


def is_conflicting_over(frames: Sequence[StateFrame], object_type: str | None, policy: SignalPolicy) -> bool | None:
    """:func:`is_conflicting` over an arbitrary run of state frames.

    Same reasoning as :func:`is_wrong_state_over` - see there for why the frames arrive directly rather than
    through a track view.
    """
    if not frames:
        return None
    rule = policy.conflicting_rule(object_type)
    return None if rule is None else rule.holds(frames)


def hallucination_matches(hallucination: HallucinationView, object_type: str) -> bool:
    """Whether ``hallucination`` would be filed under ``object_type``'s detector class."""
    return hallucination.class_label == hallucination_label_for(object_type)


def _matching_type(group: Group, class_label: str) -> str | None:
    """The first registered type in ``group`` whose checker class matches ``class_label``, if any.

    Matching goes through the shared alias table rather than by equality, because a hallucination's ``class``
    is whatever spelling the detector emitted while the registry is keyed by RDS type.
    """
    for rds_type in sorted(types_in_group(group)):
        kind = kind_of(rds_type)
        if kind is not None and object_class_matches(class_label, kind.class_name):
            return rds_type
    return None


def governing_object_type(class_label: str) -> str | None:
    """The registered object type a hallucination labeled ``class_label`` is scored under, or ``None``.

    ``None`` means no rule can apply to it - the label belongs to no registered type - which is different from
    a rule applying and rejecting it. A caller reporting on a file should surface the two differently, or a
    class the library cannot score yet looks identical to one that was scored and found irrelevant.
    """
    for group in Group:
        matched = _matching_type(group, class_label)
        if matched is not None:
            return matched
    return None


def relevant_hallucinations(view: ResultView, group: Group, policy: SignalPolicy) -> tuple[HallucinationView, ...]:
    """The extra objects of ``group`` the render invented, filtered by each type's hallucination rule.

    A hallucination has no ground-truth object behind it, so the type whose rule applies is resolved from its
    detector class label. Results keep the order they appear in the result file.
    """
    kept: list[HallucinationView] = []
    for hallucination in view.hallucinations:
        rds_type = _matching_type(group, hallucination.class_label)
        if rds_type is None:
            continue
        rule = policy.hallucination_rule(rds_type)
        kind = kind_of(rds_type)
        if rule is None or kind is None:
            continue
        if rule.holds(hallucination, relevancy_meaningful=kind.relevancy_meaningful):
            kept.append(hallucination)
    return tuple(kept)
