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

"""Typed, frame-space-normalized views over an object-check ``*.objects.results.json``.

This is the plumbing half of the object-level metrics library: one implementation of "read a result JSON",
so no consumer has to carry its own sparse-matrix parser. The policy half - what "present", "missing",
"misaligned" or "hallucinated" *mean* - lives in :mod:`kpi.automotive.existence`; this module only decides
which numbers exist and what frame they sit on.

Frame spaces in the result JSON are genuinely different from each other, and conflating them is the bug this
module exists to prevent:

* ``score_matrix`` cells carry a **row index into** ``processed_frame_ids`` - a compacted, processed-order
  ordinal. That is what :attr:`ScoreCell.frame_ordinal` holds.
* ``hallucination_detections`` entries carry a **video frame index**, i.e. a *value* from
  ``processed_frame_ids`` rather than an index into it, and may name a frame that was never scored at all.
  That is what :attr:`Detection.frame_idx` holds, and the two must not be compared without conversion.

See ``checks/objects/single_view/README.md`` for the on-disk schema these views are built from.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
import enum
import math
from typing import Any

# The column layout every ``sparse_compact`` score matrix used before the format gained a self-describing
# ``columns`` field (which also moved ``score`` and introduced ``projection_area_px``). Files written since
# then declare their own layout and this fallback is unused; older files do not, and reading them positionally
# is the only way to recover their scores.
LEGACY_COMPACT_COLUMNS: tuple[str, ...] = ("frame_idx", "track_idx", "importance", "score")

# The ordinal assigned to a cell whose frame index is absent, unparseable, or outside the matrix. The score
# itself is still real evidence about the track, so it is kept; only its position in time is unknown.
UNKNOWN_FRAME_ORDINAL = -1


class ImportancePolicy(enum.StrEnum):
    """Which scored cells count, and therefore which objects are considered observed at all.

    The object check emits a per-cell ``importance`` flag: an object is *important* in a frame when it passed
    the check's ego-relevance filters there (lane relevance, distance, oncoming). Honoring the flag asks "how
    well is the model doing on the objects that matter to the ego?"; ignoring it asks "how well is the model
    doing overall?". Both are legitimate questions with different answers, and neither is a safe default.

    They are not close. Measured over 997 real dynamic result files, 82% of emitted cells carry
    ``importance == 0``, and 28.8% of tracks are scored under one reading while never-observed under the
    other. Because that changes the denominator of every rate this library computes, there is **no default**:
    :func:`read_result` requires the policy as a keyword argument and :attr:`ResultView.importance_policy`
    records the answer, so a number can always be traced back to the population it was computed over.
    """

    # Only cells the check marked important, so counts describe the objects the ego has to contend with.
    EGO_RELEVANT = "ego_relevant"
    # Every emitted cell, importance ignored, so counts describe the render as a whole.
    ALL_SCORED = "all_scored"


@dataclass(frozen=True)
class ScoreCell:
    """One scored (frame, track) entry of the correspondence score matrix."""

    frame_ordinal: int  # index into ``processed_frame_ids``; ``UNKNOWN_FRAME_ORDINAL`` when unavailable
    track_id: int  # resolved through ``track_ids``; never the raw positional ``track_idx``
    score: float
    important: bool | None  # ``None`` when the file declares no importance column


@dataclass(frozen=True)
class ScoredFrame:
    """One frame's correspondence score for one track, carrying both frame spaces.

    A rule gets the frame, not a bare number, so that "how long" questions are expressible at all: a track
    whose scores average well above zero can still have been drawn empty for a contiguous stretch, and a
    sequence of floats cannot tell that from the same zeros scattered at random, nor an interior dropout from
    an object entering or leaving view.

    Both axes are published because rules legitimately want different ones. ``ordinal`` is the position among
    *scored* frames and is what adjacency means - consecutive ordinals are consecutive measurements, whatever
    the cadence. ``frame_idx`` is the video frame that ordinal names, which is what a rule reasoning about
    elapsed time needs, since a fixed number of samples is a different duration at 30 FPS than at the 10 FPS
    annotation cadence. It is ``None`` only when the file's ``processed_frame_ids`` cannot name the frame.
    """

    ordinal: int
    frame_idx: int | None
    score: float


@dataclass(frozen=True)
class StateFrame:
    """One frame of a traffic light's per-frame state measurement.

    ``ordinal`` is the position of this frame within ``processed_frame_ids``, **not** the video frame index the
    file records. The predicates that consume these frames test adjacency as ``idx == prev + 1`` - a gap starts
    a new color episode, and a run of conflicting frames must be contiguous - so they are only meaningful on a
    dense axis. A run read at 30 FPS against 10 FPS annotations lands on video frames 0, 3, 6, on which every
    episode would be one frame long and every run length 1. Normalizing here is what makes the predicates mean
    what they say; measured across 80 real result files, a third have a cadence where this is not the identity.

    ``state_score``, ``multi_color_score`` and ``confidence`` are ``nan`` where the checker could not measure
    them - most often because ``expected_state`` is ``"unknown"``, which it pairs with a NaN state score.
    """

    ordinal: int
    frame_idx: int  # the video frame ``ordinal`` names, for a rule that reasons about elapsed time
    expected_state: str
    state_score: float
    multi_color_score: float
    confidence: float

    def as_row(self) -> dict[str, Any]:
        """This frame in the shape :mod:`checks.utils.tl_state`'s predicates read.

        Those predicates are shared with any reporting code that hands them raw result rows; keeping their
        signature means this library converts rather than forking them. The row's
        ``frame_idx`` key deliberately carries the **ordinal**, not this frame's :attr:`frame_idx` - those
        predicates test adjacency as ``idx == prev + 1``, so the ordinal is the axis they need. A rule wanting
        real video frames should read :attr:`frame_idx` off the frame rather than this row.
        """
        return {
            "frame_idx": self.ordinal,
            "expected_state": self.expected_state,
            "traffic_light_state_score": self.state_score,
            "traffic_light_multi_color_score": self.multi_color_score,
            "traffic_light_state_confidence": self.confidence,
        }


@dataclass(frozen=True)
class TrackView:
    """A ground-truth (world-model) object and the correspondence scores it earned.

    ``frame_scores`` and ``mean_score`` are already filtered by the view's :class:`ImportancePolicy`, so a
    track can be *unscored* under one policy and scored under another. :func:`kpi.automotive.existence.is_unscored`
    is the supported way to ask; do not test ``mean_score`` for NaN by hand.
    """

    track_id: int
    object_type: str | None  # RDS type: "Car", "Pedestrian", "TrafficLight", ...
    frame_scores: tuple[ScoredFrame, ...]  # NaN-free, ordinal-ascending
    mean_score: float  # ``nan`` when the track has no cells under the active policy
    synthetic_source_id: str | None  # set only for synthetically injected tracks
    importance: Mapping[str, Any] = field(default_factory=dict)  # min_distance_m, lane_counts, ...
    # Per-frame traffic-light state, ordinal-normalized. Empty for every type but TrafficLight, and for a
    # light the checker never managed to sample.
    state_frames: tuple[StateFrame, ...] = ()
    # A lane line's predominant RDS-HQ paint style (``tracks[].lane_style``), which selects its existence cutoff.
    # ``None`` for every other type, and for a lane line whose result does not record it.
    lane_style: str | None = None

    def scores(self) -> tuple[float, ...]:
        """The per-frame scores alone, in ordinal order, for a caller that needs no frame identity."""
        return tuple(frame.score for frame in self.frame_scores)


@dataclass(frozen=True)
class Detection:
    """One unexplained image region attributed to a hallucination track."""

    frame_idx: int  # a VIDEO frame index of this run, not an index into ``processed_frame_ids``
    bbox_xywh: tuple[int, int, int, int]  # in the processor's rescaled 1280x720 image space
    class_label: str


@dataclass(frozen=True)
class HallucinationView:
    """An object the render produced that the world model never contained.

    ``relevancy`` counts frames whose box overlapped the road mask. It is meaningful only for modes whose
    tracker was given a road mask - the static processor runs with ``road_mask=None``, so every static
    hallucination carries ``relevancy == 0`` and gating on it would silently drop all of them. Ask
    :func:`kpi.automotive.taxonomy.relevancy_is_meaningful` rather than assuming.
    """

    id: int
    class_label: str
    relevancy: int
    detections: tuple[Detection, ...]


@dataclass(frozen=True)
class ResultView:
    """One result JSON, parsed once.

    Carries no notion of which processor wrote the file. Every determination is made per object type from the
    registry, so a file's contents decide what can be scored - a caller does not have to declare it, and
    cannot declare it wrongly.
    """

    # Video frame indices, positionally addressed by a cell's ``frame_ordinal``; an unparseable entry is
    # ``None`` in place rather than removed, so the ordinals keep pointing at the frames they named.
    processed_frame_ids: tuple[int | None, ...]
    tracks: Mapping[int, TrackView]
    hallucinations: tuple[HallucinationView, ...]
    importance_policy: ImportancePolicy
    # Ids present in ``track_ids`` but absent from ``tracks``, so they carry no ``object_type`` and cannot be
    # classified. Empty on the dynamic mode (measured: 0 across 5 085 real tracks), but substantial on the
    # static mode, where most registered objects are never scored. Kept as a coverage diagnostic rather than
    # folded into an "unknown" object type, which would report them as a missing-object finding.
    untyped_track_ids: frozenset[int] = frozenset()
    # Every light the file measured for color, keyed by track id - including the ones with no ``tracks`` entry,
    # which :attr:`TrackView.state_frames` cannot reach. The two detectors gate on different criteria (the
    # overlap check on projected area, the state check on edge truncation and segmentation mass), so neither
    # population contains the other and color must be counted over its own.
    state_frames: Mapping[int, tuple[StateFrame, ...]] = field(default_factory=dict)


def _int_or_none(value: Any) -> int | None:
    """``value`` as an int, or ``None`` when it is missing or unparseable."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positional_ints(raw_values: Any) -> list[int | None]:
    """Parse an array that other fields address **by position**, keeping unparseable entries as ``None``.

    Dropping a bad entry instead would renumber every later position: a ``track_idx`` of 2 against
    ``[10, "x", 20]`` would silently resolve to a different object's id rather than being rejected, and the
    last cell would fall off the end. Callers skip a ``None`` rather than shifting past it.
    """
    if not isinstance(raw_values, Sequence) or isinstance(raw_values, str):
        return []
    return [_int_or_none(raw) for raw in raw_values]


def _track_id_list(result: Mapping[str, Any]) -> list[int | None]:
    """The ``track_ids`` array as ints; the positional ``track_idx`` of a score cell indexes into it."""
    return _positional_ints(result.get("track_ids"))


def _frame_count(matrix: Mapping[str, Any]) -> int | None:
    """The matrix's declared frame count, or ``None`` when it declares no usable shape."""
    shape = matrix.get("shape")
    if not isinstance(shape, Sequence) or isinstance(shape, str) or len(shape) < 1:
        return None
    return _int_or_none(shape[0])


def _frame_ordinal(raw: Any, frame_count: int | None) -> int:
    """Validate a raw frame index into an ordinal, or :data:`UNKNOWN_FRAME_ORDINAL`.

    An out-of-range ordinal is downgraded rather than dropped: the score is real evidence about the track and
    every existence rule is order-independent, so discarding the cell would lose a verdict to protect a field
    that rule never reads. (A reader materializing a dense matrix has to drop such a cell instead, having no
    row to put it in.)
    """
    ordinal = _int_or_none(raw)
    if ordinal is None or ordinal < 0:
        return UNKNOWN_FRAME_ORDINAL
    if frame_count is not None and ordinal >= frame_count:
        return UNKNOWN_FRAME_ORDINAL
    return ordinal


def _iter_sparse_cells(
    data: Sequence[Any], track_ids: Sequence[int | None], frame_count: int | None
) -> Iterator[ScoreCell]:
    """Cells of a ``sparse`` matrix: one dict per entry, keyed by name. Carries no importance flag."""
    for entry in data:
        if not isinstance(entry, Mapping):
            continue
        track_idx = _int_or_none(entry.get("track_idx"))
        try:
            score = float(entry["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if track_idx is None or math.isnan(score) or not 0 <= track_idx < len(track_ids):
            continue
        track_id = track_ids[track_idx]
        if track_id is None:
            continue
        yield ScoreCell(
            frame_ordinal=_frame_ordinal(entry.get("frame_idx"), frame_count),
            track_id=track_id,
            score=score,
            important=None,
        )


def _iter_compact_cells(
    matrix: Mapping[str, Any], data: Sequence[Any], track_ids: Sequence[int | None], frame_count: int | None
) -> Iterator[ScoreCell]:
    """Cells of a ``sparse_compact`` matrix: positional rows whose layout ``columns`` declares.

    Falls back to :data:`LEGACY_COMPACT_COLUMNS` when ``columns`` is absent, which is what makes files
    written before that field existed readable at all. A row too short for the columns it claims, or
    carrying a non-numeric track index or score, is skipped rather than aborting the matrix.
    """
    columns = [str(name) for name in (matrix.get("columns") or LEGACY_COMPACT_COLUMNS)]
    index_of = {name: position for position, name in enumerate(columns)}
    track_col = index_of.get("track_idx")
    score_col = index_of.get("score")
    if track_col is None or score_col is None:
        return
    frame_col = index_of.get("frame_idx")
    importance_col = index_of.get("importance")

    for row in data:
        if not isinstance(row, Sequence) or isinstance(row, str) or len(row) <= max(track_col, score_col):
            continue
        track_idx = _int_or_none(row[track_col])
        try:
            score = float(row[score_col])
        except (TypeError, ValueError):
            continue
        if track_idx is None or math.isnan(score) or not 0 <= track_idx < len(track_ids):
            continue
        track_id = track_ids[track_idx]
        if track_id is None:
            continue
        important: bool | None = None
        if importance_col is not None and importance_col < len(row):
            flag = _int_or_none(row[importance_col])
            important = None if flag is None else flag == 1
        raw_frame = row[frame_col] if frame_col is not None and frame_col < len(row) else None
        yield ScoreCell(
            frame_ordinal=_frame_ordinal(raw_frame, frame_count),
            track_id=track_id,
            score=score,
            important=important,
        )


def iter_score_cells(result: Mapping[str, Any]) -> Iterator[ScoreCell]:
    """Every scored cell of ``result``'s score matrix, unfiltered by importance.

    NaN scores are dropped rather than yielded. The serialized format is supposed to omit them, but Python's
    ``json`` accepts the non-standard ``NaN`` token by default, so one can reach a reader from a file
    written by something less strict. Dropping it here is what lets every downstream rule take a NaN-free
    sequence as a precondition instead of re-guarding it.

    Handles both serialized formats (``sparse`` and ``sparse_compact``) and resolves each cell's positional
    ``track_idx`` to its stable ``track_id``. A consumer needing a dense matrix should materialize it from
    these cells rather than re-parsing the file, so there is one parser to keep correct.
    """
    matrix = result.get("score_matrix")
    if not isinstance(matrix, Mapping):
        return
    data = matrix.get("data")
    if not isinstance(data, Sequence) or isinstance(data, str):
        return
    track_ids = _track_id_list(result)
    frame_count = _frame_count(matrix)
    fmt = matrix.get("format")
    if fmt == "sparse":
        yield from _iter_sparse_cells(data, track_ids, frame_count)
    elif fmt == "sparse_compact":
        yield from _iter_compact_cells(matrix, data, track_ids, frame_count)


def cell_is_counted(cell: ScoreCell, policy: ImportancePolicy) -> bool:
    """Whether ``policy`` counts ``cell``.

    A cell from a file with no importance column (``important is None``) is counted under both policies: the
    check only emitted entries it had already scored and admitted, so there is no flag to filter on and
    treating the absence as "not important" would empty the matrix.
    """
    if policy is ImportancePolicy.ALL_SCORED:
        return True
    return cell.important is not False


def _bbox_xywh(detection: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    """A detection's box as a 4-tuple, or ``None`` when it is missing or malformed."""
    raw = detection.get("bbox_xywh")
    if raw is None:
        raw = detection.get("bbox")
    if not isinstance(raw, Sequence) or isinstance(raw, str) or len(raw) != 4:
        return None
    x, y, width, height = (_int_or_none(component) for component in raw)
    if x is None or y is None or width is None or height is None:
        return None
    return (x, y, width, height)


def _resolve_detections(
    detection_indices: Any, class_detections: Sequence[Any], class_label: str
) -> tuple[Detection, ...]:
    """Resolve a hallucination track's detection indices against its class's detection list.

    ``hallucination_tracks[].detections`` holds integer positions into ``hallucination_detections[class]``
    rather than the detections themselves. Non-integer, out-of-range, and structurally incomplete entries are
    dropped: a track that loses some detections is still a real hallucination, whereas failing here would
    discard it entirely.
    """
    if not isinstance(detection_indices, Sequence) or isinstance(detection_indices, str):
        return ()
    resolved: list[Detection] = []
    for raw_index in detection_indices:
        if not isinstance(raw_index, int) or isinstance(raw_index, bool):
            continue
        if not 0 <= raw_index < len(class_detections):
            continue
        detection = class_detections[raw_index]
        if not isinstance(detection, Mapping):
            continue
        frame_idx = _int_or_none(detection.get("frame_idx"))
        bbox = _bbox_xywh(detection)
        if frame_idx is None or bbox is None:
            continue
        resolved.append(Detection(frame_idx=frame_idx, bbox_xywh=bbox, class_label=class_label))
    return tuple(resolved)


def hallucination_views(result: Mapping[str, Any]) -> tuple[HallucinationView, ...]:
    """Every hallucination track in ``result``, with its detections resolved to real boxes.

    No relevancy or class filtering happens here - that is a policy decision made by
    :func:`kpi.automotive.existence.relevant_hallucinations`, which knows whether relevancy means anything
    for the mode in hand.
    """
    detections_by_class = result.get("hallucination_detections")
    if not isinstance(detections_by_class, Mapping):
        detections_by_class = {}

    views: list[HallucinationView] = []
    for track in result.get("hallucination_tracks") or []:
        if not isinstance(track, Mapping):
            continue
        track_id = _int_or_none(track.get("id"))
        if track_id is None:
            continue
        class_label = str(track.get("class") or "")
        class_detections = detections_by_class.get(class_label) or []
        if not isinstance(class_detections, Sequence) or isinstance(class_detections, str):
            class_detections = []
        views.append(
            HallucinationView(
                id=track_id,
                class_label=class_label,
                relevancy=_int_or_none(track.get("relevancy")) or 0,
                detections=_resolve_detections(track.get("detections"), class_detections, class_label),
            )
        )
    return tuple(views)


def _float_or_nan(value: Any) -> float:
    """``value`` as a float, or ``nan`` when it is missing or unparseable.

    The checker writes a literal ``NaN`` token for a measurement it could not take, which Python's ``json``
    accepts; a reader that treated it as an error would discard the frame rather than the measurement.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def state_frames_by_track(result: Mapping[str, Any]) -> dict[int, tuple[StateFrame, ...]]:
    """``track_id -> ordinal-normalized state frames`` for every light the file measured.

    The normalization in :func:`read_result` is only reachable through :attr:`TrackView.state_frames`, and a
    :class:`TrackView` exists only for an entry in the file's ``tracks`` array - but the ``traffic_light_state``
    block is keyed independently, and a light is routinely in the block while absent from ``tracks``. Measured
    over 55 real static result files carrying that block, only **6** had every light represented in ``tracks``;
    18 had none at all. Reading state through the track view alone would therefore silently drop most of the
    population, which is why this seam is keyed off the block instead.

    Use this when the question is about color state and the answer must cover every measured light. Use
    :attr:`TrackView.state_frames` when the light must also be an object the correspondence pass scored, since
    that is what ties the two measurement channels to one object.
    """
    return _state_frames(result, tuple(_positional_ints(result.get("processed_frame_ids"))))


def _state_frames(
    result: Mapping[str, Any], processed_frame_ids: Sequence[int | None]
) -> dict[int, tuple[StateFrame, ...]]:
    """``track_id -> ordinal-normalized state frames`` from the file's ``traffic_light_state`` block.

    Rows whose ``frame_idx`` names a frame absent from ``processed_frame_ids`` are dropped: their position on
    the dense axis is undefined, and inventing one would fabricate adjacency the predicates then read as a
    contiguous episode. Measured over 43 298 real rows, none are dropped - the checker only emits a state row
    for a frame it processed - so this is a guard, not a routine filter.
    """
    ordinals = {frame: index for index, frame in enumerate(processed_frame_ids) if frame is not None}
    raw = result.get("traffic_light_state")
    if not isinstance(raw, Mapping):
        return {}

    by_track: dict[int, tuple[StateFrame, ...]] = {}
    for raw_track_id, rows in raw.items():
        track_id = _int_or_none(raw_track_id)
        if track_id is None or not isinstance(rows, Sequence) or isinstance(rows, str):
            continue
        frames: list[StateFrame] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            frame_idx = _int_or_none(row.get("frame_idx"))
            if frame_idx is None:
                continue
            ordinal = ordinals.get(frame_idx)
            if ordinal is None:
                continue
            frames.append(
                StateFrame(
                    ordinal=ordinal,
                    frame_idx=frame_idx,
                    expected_state=str(row.get("expected_state") or "").lower(),
                    state_score=_float_or_nan(row.get("traffic_light_state_score")),
                    multi_color_score=_float_or_nan(row.get("traffic_light_multi_color_score")),
                    confidence=_float_or_nan(row.get("traffic_light_state_confidence")),
                )
            )
        if frames:
            by_track[track_id] = tuple(sorted(frames, key=lambda f: f.ordinal))
    return by_track


def _video_frame_of(ordinal: int, processed_frame_ids: Sequence[int | None]) -> int | None:
    """The video frame ``ordinal`` names, or ``None`` when the file cannot say.

    ``UNKNOWN_FRAME_ORDINAL`` and any out-of-range ordinal resolve to ``None`` rather than raising or wrapping:
    a cell whose frame the file could not place is still real evidence about the track, so it keeps its score
    and loses only the frame identity.
    """
    if ordinal < 0 or ordinal >= len(processed_frame_ids):
        return None
    return processed_frame_ids[ordinal]


def _track_metadata(result: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    """``track_id -> tracks[] entry`` for every entry carrying a parseable id."""
    metadata: dict[int, Mapping[str, Any]] = {}
    for entry in result.get("tracks") or []:
        if not isinstance(entry, Mapping):
            continue
        track_id = _int_or_none(entry.get("track_id"))
        if track_id is not None:
            metadata[track_id] = entry
    return metadata


def read_result(result: Mapping[str, Any], *, importance: ImportancePolicy) -> ResultView:
    """Parse a result JSON into a :class:`ResultView`.

    ``importance`` is required rather than defaulted; see :class:`ImportancePolicy` for why the two existing
    consumers disagree and what the choice costs.

    A :class:`TrackView` is built for every entry in ``tracks``, because that is the only place an object's
    ``object_type`` appears and no existence rule can be applied without it. Ids that appear only in
    ``track_ids`` land in :attr:`ResultView.untyped_track_ids` instead of becoming typeless tracks.
    """
    metadata = _track_metadata(result)
    processed_frame_ids = tuple(_positional_ints(result.get("processed_frame_ids")))
    state_frames = _state_frames(result, processed_frame_ids)
    scores: dict[int, list[tuple[int, float]]] = {track_id: [] for track_id in metadata}
    for cell in iter_score_cells(result):
        if not cell_is_counted(cell, importance):
            continue
        bucket = scores.get(cell.track_id)
        if bucket is not None:
            bucket.append((cell.frame_ordinal, cell.score))

    tracks: dict[int, TrackView] = {}
    for track_id, entry in metadata.items():
        frame_scores = tuple(
            ScoredFrame(ordinal=ordinal, frame_idx=_video_frame_of(ordinal, processed_frame_ids), score=score)
            for ordinal, score in sorted(scores[track_id])
        )
        mean_score = math.nan
        if frame_scores:
            mean_score = math.fsum(frame.score for frame in frame_scores) / len(frame_scores)
        synthetic = entry.get("synthetic_source_id")
        track_importance = entry.get("importance")
        lane_style = entry.get("lane_style")
        tracks[track_id] = TrackView(
            track_id=track_id,
            object_type=entry.get("object_type"),
            frame_scores=frame_scores,
            mean_score=mean_score,
            synthetic_source_id=synthetic if isinstance(synthetic, str) and synthetic else None,
            importance=track_importance if isinstance(track_importance, Mapping) else {},
            state_frames=state_frames.get(track_id, ()),
            lane_style=lane_style if isinstance(lane_style, str) and lane_style else None,
        )

    return ResultView(
        processed_frame_ids=processed_frame_ids,
        tracks=tracks,
        hallucinations=hallucination_views(result),
        importance_policy=importance,
        untyped_track_ids=frozenset(t for t in _track_id_list(result) if t is not None) - frozenset(metadata),
        state_frames=state_frames,
    )
