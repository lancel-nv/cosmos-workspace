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

"""The rules that turn a track's scores, or a hallucination track, into a verdict - and the extension point
for writing your own.

A rule is any object that evaluates itself. There is no closed set of rule kinds and no dispatch table: the
library calls :meth:`TrackRule.holds` / :meth:`HallucinationRule.holds` and serializes the rule through
``as_dict``, so a caller can define a rule this module has never heard of, install it for one object type in a
:class:`~kpi.automotive.policy.SignalPolicy`, and have it evaluated and stamped into provenance like any
built-in.

**A rule states its own comparison.** ``PercentileBelow`` and ``PercentileBetween`` are separate types rather
than one parameterized "percentile rule" whose meaning depends on which policy field it was stored in. That
matters for an extension point: a rule read out of a different slot must not quietly mean something else.

Implementing one means satisfying the protocol, nothing more - no registration, no base class:

    @dataclass(frozen=True)
    class MedianBelow:
        threshold: float

        def holds(self, frames: Sequence[ScoredFrame]) -> bool:
            return statistics.median(frame.score for frame in frames) < self.threshold

        def as_dict(self) -> dict[str, Any]:
            return {"rule": "median_below", "threshold": self.threshold}

A track rule receives frames rather than bare scores so that "for how long" is expressible. A rule that only
needs the numbers reads ``frame.score`` and ignores the rest; a rule that asks whether an object was drawn
empty for a *contiguous stretch*, or whether the gap sat in the middle of the track rather than at its ends,
needs the frame identity a sequence of floats has already thrown away.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import Any, Protocol

import numpy as np

from checks.utils.tl_state import conflicting_run, episode_state_trip, gt_color_episodes
from kpi.automotive.result_view import HallucinationView, ScoredFrame, StateFrame


class TrackRule(Protocol):
    """Decides one signal for one ground-truth track from its correspondence scores.

    ``frames`` is guaranteed non-empty and NaN-free: the library resolves unscored tracks to ``None`` before
    any rule is consulted, so an implementation never has to decide what an absent measurement means. It must
    return a plain ``bool`` - "not applicable" is expressed by leaving the type out of the policy, not by a
    rule returning ``None``.

    Frames arrive ordinal-ascending, each carrying both frame spaces (see
    :class:`~kpi.automotive.result_view.ScoredFrame`). Consecutive ``ordinal`` values are consecutive
    measurements whatever the cadence, so that is the axis to test adjacency on; ``frame_idx`` is there for a
    rule reasoning about elapsed video time.
    """

    def holds(self, frames: Sequence[ScoredFrame]) -> bool:
        """Whether this rule's condition is met by ``frames``."""
        ...

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable record of this rule, for the provenance stamp."""
        ...


class StateRule(Protocol):
    """Decides one signal for a traffic light from its per-frame color state.

    ``frames`` are ordinal-normalized (see :class:`~kpi.automotive.result_view.StateFrame`) and may carry NaN
    measurements, which an implementation must treat as unmeasured rather than as a low score - the checker
    writes NaN for a frame it could not sample, most often because the ground-truth color was unknown.
    ``expected_state`` is likewise empty on a frame that names no ground-truth color.
    """

    def holds(self, frames: Sequence[StateFrame]) -> bool | None:
        """Whether this rule's condition is met by ``frames``, or ``None`` when they cannot answer it.

        ``None`` is how a rule declines rather than guesses: :class:`WrongColorEpisode` returns it for a light
        whose frames carry no ground-truth color at all, where any verdict would be invented. A rule that
        needs nothing beyond the scores themselves - :class:`ConflictingColors` - never has cause to.
        """
        ...

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable record of this rule, for the provenance stamp."""
        ...


class HallucinationRule(Protocol):
    """Decides whether one hallucination track counts.

    ``relevancy_meaningful`` says whether ``hallucination.relevancy`` was populated at all for this object's
    mode; a rule that reads relevancy must honor it, or it will discard every hallucination of a mode whose
    tracker ran without a road mask.
    """

    def holds(self, hallucination: HallucinationView, *, relevancy_meaningful: bool) -> bool:
        """Whether this hallucination counts under this rule."""
        ...

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serializable record of this rule, for the provenance stamp."""
        ...


@dataclass(frozen=True)
class DrawnEmpty:
    """Holds when the render drew *nothing* inside the object's box - its scores average to zero.

    Deliberately not a score-below-threshold test: an object averaging 0.08 is *present but badly rendered*,
    which a misalignment rule is there to catch, not missing. Conflating the two makes a missing-object count
    unusable as a generation-quality signal, because it mixes the model that omitted an object with the model
    that drew a poor one.
    """

    atol: float = 1e-8

    def holds(self, frames: Sequence[ScoredFrame]) -> bool:
        return math.isclose(math.fsum(f.score for f in frames) / len(frames), 0.0, rel_tol=0.0, abs_tol=self.atol)

    def as_dict(self) -> dict[str, Any]:
        return {"rule": "drawn_empty", "atol": self.atol}


@dataclass(frozen=True)
class MeanBelow:
    """Holds when the mean of a track's scored frames falls below ``threshold``.

    The existence rule for polylines. A lane line or road boundary projects to a thin ribbon whose box is mostly
    road, so its correspondence score runs low even when it is rendered correctly, and its cutoff is measured for
    the type rather than shared with the cuboids.
    """

    threshold: float

    def holds(self, frames: Sequence[ScoredFrame]) -> bool:
        return math.fsum(frame.score for frame in frames) / len(frames) < self.threshold

    def as_dict(self) -> dict[str, Any]:
        return {"rule": "mean_below", "threshold": self.threshold}


@dataclass(frozen=True)
class PercentileBelow:
    """Holds when the ``percentile``-th of a track's scores falls below ``threshold``.

    A percentile rather than a mean because a partly-rendered object scores near zero on most of its frames:
    the mean cannot separate that from "never drawn at all", but the upper tail can.

    ``method="linear"`` is pinned explicitly: numpy offers nine interpolation methods and the rule was
    characterized against this one, so leaving it to the default would let a numpy upgrade move a verdict.
    """

    percentile: float
    threshold: float

    def holds(self, frames: Sequence[ScoredFrame]) -> bool:
        scores = [frame.score for frame in frames]
        return bool(np.percentile(scores, self.percentile, method="linear") < self.threshold)

    def as_dict(self) -> dict[str, Any]:
        return {"rule": "percentile_below", "percentile": self.percentile, "threshold": self.threshold}


@dataclass(frozen=True)
class PercentileBetween:
    """Holds when the ``percentile``-th of a track's scores falls strictly between ``lower`` and ``upper``.

    The lower bound is what keeps misalignment disjoint from existence: an object scoring zero was not drawn
    at all, which is a different verdict with a different answer, so it must not also be reported as badly
    placed. It is a parameter rather than a hardcoded zero so the rule states its own comparison in full -
    but nothing stops a caller choosing bounds that overlap the existence rule installed for the same type,
    in which case both signals can fire for one object. Keeping them disjoint is the caller's business.
    """

    percentile: float
    lower: float
    upper: float

    def holds(self, frames: Sequence[ScoredFrame]) -> bool:
        scores = [frame.score for frame in frames]
        point = float(np.percentile(scores, self.percentile, method="linear"))
        return self.lower < point < self.upper

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": "percentile_between",
            "percentile": self.percentile,
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass(frozen=True)
class RelevancyGate:
    """Holds when a hallucination's box overlapped the drivable surface in at least ``min_relevancy`` frames.

    Anything below that is scenery the ego never has to contend with. Where relevancy is not meaningful the
    gate passes everything through rather than discarding it: the field is structurally zero for a mode whose
    tracker ran without a road mask, so gating on it there would drop every hallucination.
    """

    min_relevancy: int = 1

    def holds(self, hallucination: HallucinationView, *, relevancy_meaningful: bool) -> bool:
        if not relevancy_meaningful:
            return True
        return hallucination.relevancy >= self.min_relevancy

    def as_dict(self) -> dict[str, Any]:
        return {"rule": "relevancy_gate", "min_relevancy": self.min_relevancy}


@dataclass(frozen=True)
class WrongColorEpisode:
    """Holds when any ground-truth color episode was rendered in the wrong color.

    A light cycles color over a clip, so there is no single expected color for a track: the frames are split
    into episodes of constant ground-truth color and each is judged on its own. An episode trips when its mean
    state score is below ``state_score_threshold``, **or** when it has ``min_wrong_frames`` consecutive frames
    below it - a brief wrong-color burst that the mean would otherwise wash out.

    Both arms delegate to the shared predicates in :mod:`checks.utils.tl_state`, so this cannot drift from the
    other consumers of the same rule.

    Returns ``None`` for a light whose frames name no ground-truth color. That is not a corner case to be
    tidy about: a result file written before the checker emitted ``expected_state`` carries a full set of
    state scores and no colors at all, and without the guard every frame would fall into one episode of
    color ``""`` split only on frame contiguity, yielding a confident wrong-color verdict for a file that
    contains nothing to have been wrong about.
    """

    state_score_threshold: float
    min_wrong_frames: int

    def holds(self, frames: Sequence[StateFrame]) -> bool | None:
        # Frames naming no color are dropped rather than judged: there is no ground truth to be wrong against.
        rows = [frame.as_row() for frame in frames if frame.expected_state]
        if not rows:
            return None
        for _color, episode in gt_color_episodes(rows):
            trip, _mean = episode_state_trip(episode, self.state_score_threshold, self.min_wrong_frames)
            if trip is not None:
                return True
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": "wrong_color_episode",
            "state_score_threshold": self.state_score_threshold,
            "min_wrong_frames": self.min_wrong_frames,
        }


@dataclass(frozen=True)
class ConflictingColors:
    """Holds when a light shows two colors at once for a sustained run of frames.

    The multi-color score is the second-most-prominent color's share of the dominant one, so a high value
    means one light is rendering conflicting signals. A conflict is typically brief and intermittent, which is
    why this reads a **run of contiguous frames** over threshold rather than a per-track mean: the mean drowns
    a few conflicting frames in a mostly-clean track, and a single spike is usually mask noise.

    The contiguity requirement is why the frames must be ordinal-normalized before they arrive here.
    """

    multi_color_threshold: float
    min_conflicting_frames: int

    def holds(self, frames: Sequence[StateFrame]) -> bool:
        run, _peak = conflicting_run([frame.as_row() for frame in frames], self.multi_color_threshold)
        return run >= self.min_conflicting_frames

    def as_dict(self) -> dict[str, Any]:
        return {
            "rule": "conflicting_colors",
            "multi_color_threshold": self.multi_color_threshold,
            "min_conflicting_frames": self.min_conflicting_frames,
        }
